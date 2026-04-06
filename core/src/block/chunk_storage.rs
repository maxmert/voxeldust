use std::collections::HashMap;

use super::block_def::DAMAGE_STAGES;
use super::block_id::BlockId;
use super::block_meta::{BlockFlags, BlockMeta, BlockOrientation};
use super::palette::{block_index, PaletteStorage, CHUNK_VOLUME};
use super::registry::BlockRegistry;

/// Result of applying damage to a block.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DamageResult {
    /// The block is air or otherwise unbreakable — no effect.
    NoEffect,
    /// The block took damage but did not break. `stage` is 0–9 for visual feedback.
    Damaged { stage: u8 },
    /// The block's health reached 0 and it should be destroyed.
    Broken,
}

/// The primary data structure for one 62³ chunk of blocks.
///
/// Combines palette-compressed block types with sparse per-block metadata.
/// Owned by shard-specific chunk managers (ship shard, planet shard).
pub struct ChunkStorage {
    /// Palette-compressed block type array.
    blocks: PaletteStorage,
    /// Sparse metadata for blocks that need it (damaged, oriented, functional).
    /// Key = flat block index (`block_index(x, y, z)` cast to u32).
    meta: HashMap<u32, BlockMeta>,
    /// Monotonic edit sequence number for causality ordering.
    /// Matches the `seq` field in `ChunkBlockMods` protocol messages.
    edit_seq: u64,
}

impl ChunkStorage {
    /// Create a chunk filled uniformly with `fill`.
    pub fn new_uniform(fill: BlockId) -> Self {
        Self {
            blocks: PaletteStorage::new_single(fill),
            meta: HashMap::new(),
            edit_seq: 0,
        }
    }

    /// Create an empty (all-air) chunk.
    pub fn new_empty() -> Self {
        Self::new_uniform(BlockId::AIR)
    }

    // -----------------------------------------------------------------------
    // Block type access
    // -----------------------------------------------------------------------

    /// O(1) block type lookup.
    #[inline(always)]
    pub fn get_block(&self, x: u8, y: u8, z: u8) -> BlockId {
        self.blocks.get(x, y, z)
    }

    /// Set a block type. Returns the previous type.
    ///
    /// Does NOT automatically create or remove metadata — the caller must
    /// manage metadata separately (e.g., clear meta when replacing a
    /// functional block).
    pub fn set_block(&mut self, x: u8, y: u8, z: u8, id: BlockId) -> BlockId {
        self.blocks.set(x, y, z, id)
    }

    /// Fill the entire chunk with one type, clearing all metadata.
    pub fn fill(&mut self, id: BlockId) {
        self.blocks.fill(id);
        self.meta.clear();
    }

    // -----------------------------------------------------------------------
    // Metadata access
    // -----------------------------------------------------------------------

    /// Get metadata for a block, if any exists.
    #[inline]
    pub fn get_meta(&self, x: u8, y: u8, z: u8) -> Option<&BlockMeta> {
        self.meta.get(&(block_index(x, y, z) as u32))
    }

    /// Get mutable metadata for a block, if any exists.
    #[inline]
    pub fn get_meta_mut(&mut self, x: u8, y: u8, z: u8) -> Option<&mut BlockMeta> {
        self.meta.get_mut(&(block_index(x, y, z) as u32))
    }

    /// Get or create metadata for a block.
    pub fn get_or_create_meta(&mut self, x: u8, y: u8, z: u8) -> &mut BlockMeta {
        self.meta
            .entry(block_index(x, y, z) as u32)
            .or_insert_with(|| BlockMeta::EMPTY)
    }

    /// Remove metadata for a block (e.g., when health is fully restored or
    /// block is replaced).
    pub fn remove_meta(&mut self, x: u8, y: u8, z: u8) {
        self.meta.remove(&(block_index(x, y, z) as u32));
    }

    /// Set the orientation for a block, creating metadata if needed.
    pub fn set_orientation(&mut self, x: u8, y: u8, z: u8, orientation: BlockOrientation) {
        let meta = self.get_or_create_meta(x, y, z);
        meta.orientation = orientation;
    }

    /// Number of metadata entries (blocks with non-default state).
    pub fn meta_count(&self) -> usize {
        self.meta.len()
    }

    /// Iterate all metadata entries with their flat indices.
    pub fn iter_meta(&self) -> impl Iterator<Item = (u32, &BlockMeta)> {
        self.meta.iter().map(|(&idx, meta)| (idx, meta))
    }

    // -----------------------------------------------------------------------
    // Block health / damage
    // -----------------------------------------------------------------------

    /// Apply damage to the block at `(x, y, z)`.
    ///
    /// Returns `DamageResult` indicating what happened. On `Broken`, the caller
    /// is responsible for setting the block to Air, removing metadata, recording
    /// the delta, and emitting any functional block destroy events.
    pub fn damage_block(
        &mut self,
        x: u8,
        y: u8,
        z: u8,
        amount: u16,
        registry: &BlockRegistry,
    ) -> DamageResult {
        let id = self.get_block(x, y, z);
        if id.is_air() {
            return DamageResult::NoEffect;
        }

        let def = registry.get(id);
        let max_hp = def.max_hp();
        if max_hp == 0 {
            // Instant break (hardness 0: tall grass, flowers, etc.)
            return DamageResult::Broken;
        }

        let meta = self.get_or_create_meta(x, y, z);
        if !meta.is_damaged() {
            // First hit: initialize health to max
            meta.health = max_hp;
            meta.flags.set(BlockFlags::DAMAGED, true);
        }

        meta.health = meta.health.saturating_sub(amount);
        if meta.health == 0 {
            DamageResult::Broken
        } else {
            // Compute visual damage stage (0 = barely damaged, 9 = about to break)
            let remaining_fraction =
                (meta.health as u32 * (DAMAGE_STAGES as u32)) / (max_hp as u32);
            let stage = (DAMAGE_STAGES - 1).saturating_sub(remaining_fraction as u8);
            DamageResult::Damaged { stage }
        }
    }

    /// Restore a block's health fully, removing the damaged flag.
    /// Removes metadata if it becomes empty.
    pub fn heal_block(&mut self, x: u8, y: u8, z: u8) {
        let idx = block_index(x, y, z) as u32;
        if let Some(meta) = self.meta.get_mut(&idx) {
            meta.health = 0;
            meta.flags.set(BlockFlags::DAMAGED, false);
            if meta.is_empty() {
                self.meta.remove(&idx);
            }
        }
    }

    // -----------------------------------------------------------------------
    // Sequence number
    // -----------------------------------------------------------------------

    /// Current edit sequence number.
    pub fn edit_seq(&self) -> u64 {
        self.edit_seq
    }

    /// Increment and return the new sequence number.
    pub fn next_edit_seq(&mut self) -> u64 {
        self.edit_seq += 1;
        self.edit_seq
    }

    // -----------------------------------------------------------------------
    // Palette statistics (delegated)
    // -----------------------------------------------------------------------

    /// Number of non-air blocks.
    pub fn non_air_count(&self) -> u32 {
        self.blocks.non_air_count()
    }

    /// Whether this chunk is completely empty (all air).
    pub fn is_empty(&self) -> bool {
        self.blocks.non_air_count() == 0
    }

    /// Whether the block array is uniform (all one type).
    pub fn is_uniform(&self) -> bool {
        self.blocks.is_uniform()
    }

    /// Number of unique block types.
    pub fn unique_type_count(&self) -> usize {
        self.blocks.unique_type_count()
    }

    /// Direct access to the palette storage (for meshing adapter).
    pub fn palette(&self) -> &PaletteStorage {
        &self.blocks
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_registry() -> BlockRegistry {
        BlockRegistry::new()
    }

    #[test]
    fn new_empty_chunk() {
        let chunk = ChunkStorage::new_empty();
        assert!(chunk.is_empty());
        assert_eq!(chunk.non_air_count(), 0);
        assert_eq!(chunk.get_block(0, 0, 0), BlockId::AIR);
        assert_eq!(chunk.edit_seq(), 0);
        assert_eq!(chunk.meta_count(), 0);
    }

    #[test]
    fn set_and_get_blocks() {
        let mut chunk = ChunkStorage::new_empty();
        chunk.set_block(5, 10, 15, BlockId::STONE);
        assert_eq!(chunk.get_block(5, 10, 15), BlockId::STONE);
        assert_eq!(chunk.get_block(0, 0, 0), BlockId::AIR);
        assert_eq!(chunk.non_air_count(), 1);
    }

    #[test]
    fn metadata_lifecycle() {
        let mut chunk = ChunkStorage::new_empty();
        chunk.set_block(1, 2, 3, BlockId::HULL_STANDARD);

        // No metadata by default
        assert!(chunk.get_meta(1, 2, 3).is_none());

        // Set orientation creates metadata
        chunk.set_orientation(1, 2, 3, BlockOrientation::new(3, 2));
        let meta = chunk.get_meta(1, 2, 3).unwrap();
        assert_eq!(meta.orientation, BlockOrientation::new(3, 2));

        // Remove metadata
        chunk.remove_meta(1, 2, 3);
        assert!(chunk.get_meta(1, 2, 3).is_none());
    }

    #[test]
    fn damage_instant_break() {
        let reg = test_registry();
        let mut chunk = ChunkStorage::new_empty();
        chunk.set_block(0, 0, 0, BlockId::TALL_GRASS); // hardness = 0
        let result = chunk.damage_block(0, 0, 0, 1, &reg);
        assert_eq!(result, DamageResult::Broken);
    }

    #[test]
    fn damage_progressive() {
        let reg = test_registry();
        let mut chunk = ChunkStorage::new_empty();
        chunk.set_block(0, 0, 0, BlockId::DIRT); // hardness = 5, HP = 50

        // First hit: initializes health to 50, applies 10 damage → 40 HP
        let result = chunk.damage_block(0, 0, 0, 10, &reg);
        match result {
            DamageResult::Damaged { stage } => {
                assert!(stage < 10, "stage should be valid: {}", stage);
            }
            other => panic!("Expected Damaged, got {:?}", other),
        }

        // Metadata should show damaged
        let meta = chunk.get_meta(0, 0, 0).unwrap();
        assert!(meta.is_damaged());
        assert_eq!(meta.health, 40);
    }

    #[test]
    fn damage_to_zero_breaks() {
        let reg = test_registry();
        let mut chunk = ChunkStorage::new_empty();
        chunk.set_block(0, 0, 0, BlockId::DIRT); // HP = 50

        // Hit with massive damage
        let result = chunk.damage_block(0, 0, 0, 9999, &reg);
        assert_eq!(result, DamageResult::Broken);
    }

    #[test]
    fn damage_air_no_effect() {
        let reg = test_registry();
        let mut chunk = ChunkStorage::new_empty();
        let result = chunk.damage_block(0, 0, 0, 10, &reg);
        assert_eq!(result, DamageResult::NoEffect);
    }

    #[test]
    fn heal_block_removes_damage() {
        let reg = test_registry();
        let mut chunk = ChunkStorage::new_empty();
        chunk.set_block(0, 0, 0, BlockId::STONE);

        chunk.damage_block(0, 0, 0, 10, &reg);
        assert!(chunk.get_meta(0, 0, 0).unwrap().is_damaged());

        chunk.heal_block(0, 0, 0);
        // Metadata should be removed since block has no other state
        assert!(chunk.get_meta(0, 0, 0).is_none());
    }

    #[test]
    fn heal_preserves_other_metadata() {
        let reg = test_registry();
        let mut chunk = ChunkStorage::new_empty();
        chunk.set_block(0, 0, 0, BlockId::HULL_STANDARD);

        // Set orientation first, then damage
        chunk.set_orientation(0, 0, 0, BlockOrientation::new(2, 1));
        chunk.damage_block(0, 0, 0, 10, &reg);

        // Heal should remove damage but keep orientation
        chunk.heal_block(0, 0, 0);
        let meta = chunk.get_meta(0, 0, 0).unwrap();
        assert!(!meta.is_damaged());
        assert_eq!(meta.orientation, BlockOrientation::new(2, 1));
    }

    #[test]
    fn edit_seq_increments() {
        let mut chunk = ChunkStorage::new_empty();
        assert_eq!(chunk.edit_seq(), 0);
        assert_eq!(chunk.next_edit_seq(), 1);
        assert_eq!(chunk.next_edit_seq(), 2);
        assert_eq!(chunk.edit_seq(), 2);
    }

    #[test]
    fn fill_clears_metadata() {
        let mut chunk = ChunkStorage::new_empty();
        chunk.set_block(5, 5, 5, BlockId::STONE);
        chunk.set_orientation(5, 5, 5, BlockOrientation::new(1, 0));
        assert_eq!(chunk.meta_count(), 1);

        chunk.fill(BlockId::AIR);
        assert_eq!(chunk.meta_count(), 0);
        assert!(chunk.is_empty());
    }
}
