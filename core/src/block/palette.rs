use super::block_id::BlockId;

pub const CHUNK_SIZE: usize = 62;
pub const CHUNK_VOLUME: usize = CHUNK_SIZE * CHUNK_SIZE * CHUNK_SIZE; // 238,328

/// Flat index from block coordinates within a 62³ chunk.
/// Layout: x-major: `index = x * 62 * 62 + y * 62 + z`
#[inline(always)]
pub fn block_index(x: u8, y: u8, z: u8) -> usize {
    debug_assert!(x < 62 && y < 62 && z < 62);
    (x as usize) * (CHUNK_SIZE * CHUNK_SIZE) + (y as usize) * CHUNK_SIZE + (z as usize)
}

/// Inverse: flat index → (x, y, z).
#[inline(always)]
pub fn index_to_xyz(idx: usize) -> (u8, u8, u8) {
    debug_assert!(idx < CHUNK_VOLUME);
    let x = (idx / (CHUNK_SIZE * CHUNK_SIZE)) as u8;
    let rem = idx % (CHUNK_SIZE * CHUNK_SIZE);
    let y = (rem / CHUNK_SIZE) as u8;
    let z = (rem % CHUNK_SIZE) as u8;
    (x, y, z)
}

/// Number of u64 words needed to pack `CHUNK_VOLUME` nibble (4-bit) entries.
/// Each u64 holds 16 nibbles.
const NIBBLE_WORDS: usize = (CHUNK_VOLUME + 15) / 16; // 14,896

/// Palette-compressed block type array for one 62³ chunk.
///
/// Four storage tiers, selected by the number of unique block types:
///
/// | Unique types | Tier    | Memory      |
/// |--------------|---------|-------------|
/// | 1            | Single  | 2 bytes     |
/// | 2–16         | Nibble  | ~119 KB     |
/// | 17–256       | Byte    | ~238 KB     |
/// | 257+         | Direct  | ~477 KB     |
///
/// Auto-promotes when a new type exceeds the current tier's capacity.
/// Does NOT auto-demote (compaction happens at serialization time).
pub struct PaletteStorage {
    inner: PaletteInner,
    non_air_count: u32,
}

enum PaletteInner {
    /// All blocks are the same type.
    Single(BlockId),

    /// 2–16 unique types. 4-bit palette indices packed into u64 words.
    Nibble {
        palette: Vec<BlockId>,
        indices: Box<[u64; NIBBLE_WORDS]>,
    },

    /// 17–256 unique types. 8-bit palette indices.
    Byte {
        palette: Vec<BlockId>,
        indices: Box<[u8; CHUNK_VOLUME]>,
    },

    /// 257+ unique types. Direct u16 block IDs (no palette).
    Direct(Box<[BlockId; CHUNK_VOLUME]>),
}

impl PaletteStorage {
    /// Create storage filled uniformly with `fill`.
    pub fn new_single(fill: BlockId) -> Self {
        let non_air = if fill.is_air() { 0 } else { CHUNK_VOLUME as u32 };
        Self {
            inner: PaletteInner::Single(fill),
            non_air_count: non_air,
        }
    }

    /// Number of non-air blocks in this chunk.
    #[inline]
    pub fn non_air_count(&self) -> u32 {
        self.non_air_count
    }

    /// Get the block type at `(x, y, z)`.
    #[inline]
    pub fn get(&self, x: u8, y: u8, z: u8) -> BlockId {
        let idx = block_index(x, y, z);
        match &self.inner {
            PaletteInner::Single(id) => *id,
            PaletteInner::Nibble { palette, indices } => {
                let word = idx / 16;
                let nibble = idx % 16;
                let pi = ((indices[word] >> (nibble * 4)) & 0xF) as usize;
                palette[pi]
            }
            PaletteInner::Byte { palette, indices } => {
                palette[indices[idx] as usize]
            }
            PaletteInner::Direct(blocks) => blocks[idx],
        }
    }

    /// Set the block type at `(x, y, z)`. Returns the previous type.
    pub fn set(&mut self, x: u8, y: u8, z: u8, id: BlockId) -> BlockId {
        let old = self.get(x, y, z);
        if old == id {
            return old;
        }

        // Update non-air count
        if old.is_air() && !id.is_air() {
            self.non_air_count += 1;
        } else if !old.is_air() && id.is_air() {
            self.non_air_count -= 1;
        }

        let idx = block_index(x, y, z);

        match &mut self.inner {
            PaletteInner::Single(current) => {
                // Need to upgrade to Nibble with two palette entries.
                let old_id = *current;
                let mut palette = vec![old_id, id];
                // Deduplicate: old_id can't equal id (checked above).
                let mut indices = Box::new([0u64; NIBBLE_WORDS]);
                // All existing blocks have palette index 0 (the old_id).
                // Set the new block to palette index 1.
                let word = idx / 16;
                let nibble = idx % 16;
                indices[word] |= 1u64 << (nibble * 4);

                self.inner = PaletteInner::Nibble { palette, indices };
            }
            PaletteInner::Nibble { palette, indices } => {
                let pi = palette_index_or_insert(palette, id);
                if pi < 16 {
                    let word = idx / 16;
                    let nibble = idx % 16;
                    let mask = 0xFu64 << (nibble * 4);
                    indices[word] = (indices[word] & !mask) | ((pi as u64) << (nibble * 4));
                } else {
                    // Promote to Byte tier
                    self.promote_nibble_to_byte();
                    // Now set in the Byte tier (recursive, but only one level)
                    self.set_byte_inner(idx, id);
                }
            }
            PaletteInner::Byte { palette, indices } => {
                let pi = palette_index_or_insert(palette, id);
                if pi < 256 {
                    indices[idx] = pi as u8;
                } else {
                    // Promote to Direct tier
                    self.promote_byte_to_direct();
                    self.set_direct_inner(idx, id);
                }
            }
            PaletteInner::Direct(blocks) => {
                blocks[idx] = id;
            }
        }

        old
    }

    /// Fill the entire chunk with one block type.
    pub fn fill(&mut self, id: BlockId) {
        self.non_air_count = if id.is_air() { 0 } else { CHUNK_VOLUME as u32 };
        self.inner = PaletteInner::Single(id);
    }

    /// Number of unique block types currently stored.
    pub fn unique_type_count(&self) -> usize {
        match &self.inner {
            PaletteInner::Single(_) => 1,
            PaletteInner::Nibble { palette, .. } => palette.len(),
            PaletteInner::Byte { palette, .. } => palette.len(),
            PaletteInner::Direct(_) => {
                // Expensive — count unique values. Only used for diagnostics.
                let mut seen = std::collections::HashSet::new();
                if let PaletteInner::Direct(blocks) = &self.inner {
                    for b in blocks.iter() {
                        seen.insert(*b);
                    }
                }
                seen.len()
            }
        }
    }

    /// Whether this chunk is completely uniform (all one type).
    pub fn is_uniform(&self) -> bool {
        matches!(&self.inner, PaletteInner::Single(_))
    }

    // -----------------------------------------------------------------------
    // Internal helpers
    // -----------------------------------------------------------------------

    fn set_byte_inner(&mut self, idx: usize, id: BlockId) {
        if let PaletteInner::Byte { palette, indices } = &mut self.inner {
            let pi = palette_index_or_insert(palette, id);
            if pi < 256 {
                indices[idx] = pi as u8;
            } else {
                self.promote_byte_to_direct();
                self.set_direct_inner(idx, id);
            }
        }
    }

    fn set_direct_inner(&mut self, idx: usize, id: BlockId) {
        if let PaletteInner::Direct(blocks) = &mut self.inner {
            blocks[idx] = id;
        }
    }

    fn promote_nibble_to_byte(&mut self) {
        let (palette, nibble_indices) = match std::mem::replace(
            &mut self.inner,
            PaletteInner::Single(BlockId::AIR),
        ) {
            PaletteInner::Nibble { palette, indices } => (palette, indices),
            _ => unreachable!(),
        };

        let mut byte_indices = vec![0u8; CHUNK_VOLUME].into_boxed_slice();
        let byte_indices: Box<[u8; CHUNK_VOLUME]> = unsafe {
            let ptr = Box::into_raw(byte_indices) as *mut [u8; CHUNK_VOLUME];
            Box::from_raw(ptr)
        };

        // Unpack nibbles into bytes
        let byte_slice: &mut [u8; CHUNK_VOLUME] = unsafe {
            &mut *(Box::into_raw(byte_indices) as *mut [u8; CHUNK_VOLUME])
        };
        for i in 0..CHUNK_VOLUME {
            let word = i / 16;
            let nibble = i % 16;
            byte_slice[i] = ((nibble_indices[word] >> (nibble * 4)) & 0xF) as u8;
        }

        self.inner = PaletteInner::Byte {
            palette,
            indices: unsafe { Box::from_raw(byte_slice as *mut [u8; CHUNK_VOLUME]) },
        };
    }

    fn promote_byte_to_direct(&mut self) {
        let (palette, byte_indices) = match std::mem::replace(
            &mut self.inner,
            PaletteInner::Single(BlockId::AIR),
        ) {
            PaletteInner::Byte { palette, indices } => (palette, indices),
            _ => unreachable!(),
        };

        let mut direct = vec![BlockId::AIR; CHUNK_VOLUME].into_boxed_slice();
        let direct: Box<[BlockId; CHUNK_VOLUME]> = unsafe {
            let ptr = Box::into_raw(direct) as *mut [BlockId; CHUNK_VOLUME];
            Box::from_raw(ptr)
        };

        let direct_slice: &mut [BlockId; CHUNK_VOLUME] = unsafe {
            &mut *(Box::into_raw(direct) as *mut [BlockId; CHUNK_VOLUME])
        };
        for i in 0..CHUNK_VOLUME {
            direct_slice[i] = palette[byte_indices[i] as usize];
        }

        self.inner = PaletteInner::Direct(unsafe {
            Box::from_raw(direct_slice as *mut [BlockId; CHUNK_VOLUME])
        });
    }
}

/// Find or insert `id` in the palette. Returns the palette index.
fn palette_index_or_insert(palette: &mut Vec<BlockId>, id: BlockId) -> usize {
    if let Some(pos) = palette.iter().position(|&p| p == id) {
        pos
    } else {
        let idx = palette.len();
        palette.push(id);
        idx
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn single_tier_uniform() {
        let ps = PaletteStorage::new_single(BlockId::STONE);
        assert!(ps.is_uniform());
        assert_eq!(ps.non_air_count(), CHUNK_VOLUME as u32);
        assert_eq!(ps.get(0, 0, 0), BlockId::STONE);
        assert_eq!(ps.get(61, 61, 61), BlockId::STONE);
    }

    #[test]
    fn single_to_nibble_promotion() {
        let mut ps = PaletteStorage::new_single(BlockId::AIR);
        assert!(ps.is_uniform());
        assert_eq!(ps.non_air_count(), 0);

        ps.set(10, 20, 30, BlockId::STONE);
        assert!(!ps.is_uniform());
        assert_eq!(ps.non_air_count(), 1);
        assert_eq!(ps.get(10, 20, 30), BlockId::STONE);
        assert_eq!(ps.get(0, 0, 0), BlockId::AIR);
        assert_eq!(ps.unique_type_count(), 2);
    }

    #[test]
    fn nibble_tier_16_types() {
        let mut ps = PaletteStorage::new_single(BlockId::AIR);
        // Insert 15 more unique types (16 total including AIR)
        for i in 1..16u16 {
            ps.set(i as u8, 0, 0, BlockId::from_u16(i));
        }
        assert_eq!(ps.unique_type_count(), 16);
        // Verify all are correct
        for i in 1..16u16 {
            assert_eq!(ps.get(i as u8, 0, 0), BlockId::from_u16(i));
        }
        assert_eq!(ps.get(0, 0, 0), BlockId::AIR);
    }

    #[test]
    fn nibble_to_byte_promotion() {
        let mut ps = PaletteStorage::new_single(BlockId::AIR);
        // Fill 16 types (max for nibble)
        for i in 1..16u16 {
            ps.set(i as u8, 0, 0, BlockId::from_u16(i));
        }
        // 17th type triggers promotion to Byte
        ps.set(16, 0, 0, BlockId::from_u16(16));
        assert_eq!(ps.unique_type_count(), 17);
        // Verify all previous values survived
        for i in 1..=16u16 {
            assert_eq!(ps.get(i as u8, 0, 0), BlockId::from_u16(i));
        }
        assert_eq!(ps.get(0, 0, 0), BlockId::AIR);
    }

    #[test]
    fn set_returns_old_value() {
        let mut ps = PaletteStorage::new_single(BlockId::STONE);
        let old = ps.set(5, 5, 5, BlockId::DIRT);
        assert_eq!(old, BlockId::STONE);
        let old2 = ps.set(5, 5, 5, BlockId::GRASS);
        assert_eq!(old2, BlockId::DIRT);
    }

    #[test]
    fn non_air_count_tracking() {
        let mut ps = PaletteStorage::new_single(BlockId::AIR);
        assert_eq!(ps.non_air_count(), 0);

        ps.set(0, 0, 0, BlockId::STONE);
        assert_eq!(ps.non_air_count(), 1);

        ps.set(1, 0, 0, BlockId::DIRT);
        assert_eq!(ps.non_air_count(), 2);

        // Replace stone with air
        ps.set(0, 0, 0, BlockId::AIR);
        assert_eq!(ps.non_air_count(), 1);

        // Replace with same type — no change
        ps.set(1, 0, 0, BlockId::DIRT);
        assert_eq!(ps.non_air_count(), 1);
    }

    #[test]
    fn fill_resets_to_single() {
        let mut ps = PaletteStorage::new_single(BlockId::AIR);
        ps.set(5, 5, 5, BlockId::STONE);
        ps.set(10, 10, 10, BlockId::DIRT);
        assert!(!ps.is_uniform());

        ps.fill(BlockId::GRASS);
        assert!(ps.is_uniform());
        assert_eq!(ps.get(5, 5, 5), BlockId::GRASS);
        assert_eq!(ps.non_air_count(), CHUNK_VOLUME as u32);
    }

    #[test]
    fn block_index_roundtrip() {
        for x in [0u8, 1, 30, 61] {
            for y in [0u8, 1, 30, 61] {
                for z in [0u8, 1, 30, 61] {
                    let idx = block_index(x, y, z);
                    let (rx, ry, rz) = index_to_xyz(idx);
                    assert_eq!((x, y, z), (rx, ry, rz));
                }
            }
        }
    }

    #[test]
    fn block_index_bounds() {
        assert_eq!(block_index(0, 0, 0), 0);
        assert_eq!(block_index(61, 61, 61), CHUNK_VOLUME - 1);
    }

    #[test]
    fn many_unique_types_byte_tier() {
        let mut ps = PaletteStorage::new_single(BlockId::AIR);
        // Insert 100 unique types
        for i in 1..101u16 {
            ps.set((i % 62) as u8, (i / 62) as u8, 0, BlockId::from_u16(i));
        }
        assert_eq!(ps.unique_type_count(), 101);
        // Verify a few
        assert_eq!(ps.get(1, 0, 0), BlockId::from_u16(1));
        assert_eq!(ps.get(0, 1, 0), BlockId::from_u16(62));
    }
}
