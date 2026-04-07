//! Generic block grid trait for aggregation and other systems that need
//! to iterate blocks without depending on a specific storage implementation.

use glam::IVec3;

use super::block_id::BlockId;
use super::block_meta::BlockMeta;

/// Trait for any block storage that can be iterated for aggregation.
///
/// Implemented by `ShipGrid` (ships/stations) and will be implemented by
/// future planet chunk managers. Aggregation functions take `&impl BlockGridView`
/// instead of `&ShipGrid`, enabling reuse across shard types.
pub trait BlockGridView {
    /// Iterate all non-air blocks as (world_pos, block_id).
    fn iter_blocks(&self) -> Box<dyn Iterator<Item = (IVec3, BlockId)> + '_>;

    /// Get block at world position (returns AIR if not loaded).
    fn get_block(&self, x: i32, y: i32, z: i32) -> BlockId;

    /// Get block metadata at world position.
    fn get_meta(&self, x: i32, y: i32, z: i32) -> Option<&BlockMeta>;
}
