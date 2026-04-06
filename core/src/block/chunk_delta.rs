use std::collections::HashMap;

use super::block_id::BlockId;
use super::palette::{block_index, index_to_xyz};

/// A single block modification record.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BlockEdit {
    pub bx: u8,
    pub by: u8,
    pub bz: u8,
    pub new_type: BlockId,
}

/// Tracks modifications to a chunk since the last flush (network broadcast
/// or persistence snapshot).
///
/// Deduplicates: if the same block is edited multiple times, only the latest
/// state is kept. This is correct because downstream consumers (network,
/// persistence) only need the final state, not the edit history.
pub struct ChunkDelta {
    /// Key = flat block index, Value = new block type.
    edits: HashMap<u32, BlockId>,
}

impl ChunkDelta {
    pub fn new() -> Self {
        Self {
            edits: HashMap::new(),
        }
    }

    /// Record an edit. Overwrites any previous edit to the same position.
    pub fn record(&mut self, x: u8, y: u8, z: u8, new_type: BlockId) {
        let idx = block_index(x, y, z) as u32;
        self.edits.insert(idx, new_type);
    }

    /// Drain all recorded edits into a `Vec<BlockEdit>`, clearing the buffer.
    pub fn drain(&mut self) -> Vec<BlockEdit> {
        self.edits
            .drain()
            .map(|(idx, new_type)| {
                let (bx, by, bz) = index_to_xyz(idx as usize);
                BlockEdit {
                    bx,
                    by,
                    bz,
                    new_type,
                }
            })
            .collect()
    }

    /// Whether any edits have been recorded since the last drain.
    pub fn is_dirty(&self) -> bool {
        !self.edits.is_empty()
    }

    /// Number of pending edits.
    pub fn len(&self) -> usize {
        self.edits.len()
    }

    /// Clear without draining.
    pub fn clear(&mut self) {
        self.edits.clear();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn record_and_drain() {
        let mut delta = ChunkDelta::new();
        assert!(!delta.is_dirty());

        delta.record(1, 2, 3, BlockId::STONE);
        delta.record(4, 5, 6, BlockId::DIRT);
        assert!(delta.is_dirty());
        assert_eq!(delta.len(), 2);

        let edits = delta.drain();
        assert_eq!(edits.len(), 2);
        assert!(!delta.is_dirty());

        // Verify both edits are present (order not guaranteed by HashMap)
        assert!(edits.iter().any(|e| e.bx == 1 && e.by == 2 && e.bz == 3 && e.new_type == BlockId::STONE));
        assert!(edits.iter().any(|e| e.bx == 4 && e.by == 5 && e.bz == 6 && e.new_type == BlockId::DIRT));
    }

    #[test]
    fn deduplication() {
        let mut delta = ChunkDelta::new();
        delta.record(1, 2, 3, BlockId::STONE);
        delta.record(1, 2, 3, BlockId::DIRT); // overwrites
        delta.record(1, 2, 3, BlockId::GRASS); // overwrites again
        assert_eq!(delta.len(), 1);

        let edits = delta.drain();
        assert_eq!(edits.len(), 1);
        assert_eq!(edits[0].new_type, BlockId::GRASS);
    }

    #[test]
    fn clear() {
        let mut delta = ChunkDelta::new();
        delta.record(0, 0, 0, BlockId::STONE);
        assert!(delta.is_dirty());
        delta.clear();
        assert!(!delta.is_dirty());
        assert_eq!(delta.len(), 0);
    }
}
