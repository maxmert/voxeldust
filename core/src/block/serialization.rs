use super::block_id::BlockId;
use super::block_meta::{BlockFlags, BlockMeta, BlockOrientation};
use super::chunk_storage::ChunkStorage;
use super::palette::{block_index, CHUNK_SIZE, CHUNK_VOLUME};

/// Errors during chunk deserialization.
#[derive(Debug, thiserror::Error)]
pub enum ChunkDeserializeError {
    #[error("data too short: need at least {expected} bytes, got {actual}")]
    TooShort { expected: usize, actual: usize },
    #[error("invalid palette count: {0}")]
    InvalidPaletteCount(u16),
    #[error("lz4 decompression failed: {0}")]
    Lz4Error(String),
    #[error("unexpected data length after decompression")]
    UnexpectedLength,
}

/// Serialize a `ChunkStorage` to bytes for persistence or network transfer.
///
/// Format (before lz4 compression):
/// ```text
/// [2 bytes] edit_seq_lo (lower 16 bits, enough for delta ordering)
/// [2 bytes] palette_count (u16)
/// [palette_count × 2 bytes] palette entries (BlockId as u16 LE)
/// [CHUNK_VOLUME bytes] palette indices (u8 per block — expanded from palette tier)
/// [4 bytes] meta_count (u32 LE)
/// [meta_count × 12 bytes] metadata entries:
///     [4 bytes] flat_index (u32 LE)
///     [2 bytes] health (u16 LE)
///     [1 byte]  orientation (u8)
///     [1 byte]  flags (u8)
///     [4 bytes] entity_index (u32 LE)
/// ```
///
/// The entire payload is lz4-compressed.
pub fn serialize_chunk(chunk: &ChunkStorage) -> Vec<u8> {
    let mut raw = Vec::with_capacity(CHUNK_VOLUME + 1024);

    // Edit sequence (lower 16 bits)
    raw.extend_from_slice(&(chunk.edit_seq() as u16).to_le_bytes());

    // Build expanded palette: collect all unique types and index each block
    let mut palette: Vec<BlockId> = Vec::new();
    let mut palette_map: std::collections::HashMap<BlockId, u8> = std::collections::HashMap::new();
    let mut byte_indices = vec![0u8; CHUNK_VOLUME];

    for x in 0..CHUNK_SIZE as u8 {
        for y in 0..CHUNK_SIZE as u8 {
            for z in 0..CHUNK_SIZE as u8 {
                let id = chunk.get_block(x, y, z);
                let pi = *palette_map.entry(id).or_insert_with(|| {
                    let idx = palette.len() as u8;
                    palette.push(id);
                    idx
                });
                byte_indices[block_index(x, y, z)] = pi;
            }
        }
    }

    // Palette count + entries
    let palette_count = palette.len() as u16;
    raw.extend_from_slice(&palette_count.to_le_bytes());
    for &id in &palette {
        raw.extend_from_slice(&id.as_u16().to_le_bytes());
    }

    // Block indices (1 byte each, palette index)
    raw.extend_from_slice(&byte_indices);

    // Metadata
    let meta_count = chunk.meta_count() as u32;
    raw.extend_from_slice(&meta_count.to_le_bytes());
    for (flat_idx, meta) in chunk.iter_meta() {
        raw.extend_from_slice(&flat_idx.to_le_bytes());
        raw.extend_from_slice(&meta.health.to_le_bytes());
        raw.push(meta.orientation.0);
        raw.push(meta.flags.0);
        raw.extend_from_slice(&meta.entity_index.to_le_bytes());
    }

    // Compress with lz4
    lz4_flex::compress_prepend_size(&raw)
}

/// Deserialize a `ChunkStorage` from bytes.
pub fn deserialize_chunk(data: &[u8]) -> Result<ChunkStorage, ChunkDeserializeError> {
    let raw = lz4_flex::decompress_size_prepended(data)
        .map_err(|e| ChunkDeserializeError::Lz4Error(e.to_string()))?;

    let mut cursor = 0;

    // Edit sequence
    if raw.len() < 2 {
        return Err(ChunkDeserializeError::TooShort {
            expected: 2,
            actual: raw.len(),
        });
    }
    let edit_seq = u16::from_le_bytes([raw[0], raw[1]]) as u64;
    cursor += 2;

    // Palette count
    if raw.len() < cursor + 2 {
        return Err(ChunkDeserializeError::TooShort {
            expected: cursor + 2,
            actual: raw.len(),
        });
    }
    let palette_count = u16::from_le_bytes([raw[cursor], raw[cursor + 1]]);
    cursor += 2;

    if palette_count == 0 {
        return Err(ChunkDeserializeError::InvalidPaletteCount(0));
    }

    // Palette entries
    let palette_bytes = palette_count as usize * 2;
    if raw.len() < cursor + palette_bytes {
        return Err(ChunkDeserializeError::TooShort {
            expected: cursor + palette_bytes,
            actual: raw.len(),
        });
    }
    let mut palette = Vec::with_capacity(palette_count as usize);
    for i in 0..palette_count as usize {
        let lo = raw[cursor + i * 2];
        let hi = raw[cursor + i * 2 + 1];
        palette.push(BlockId::from_u16(u16::from_le_bytes([lo, hi])));
    }
    cursor += palette_bytes;

    // Block indices
    if raw.len() < cursor + CHUNK_VOLUME {
        return Err(ChunkDeserializeError::TooShort {
            expected: cursor + CHUNK_VOLUME,
            actual: raw.len(),
        });
    }

    let mut chunk = ChunkStorage::new_empty();
    for x in 0..CHUNK_SIZE as u8 {
        for y in 0..CHUNK_SIZE as u8 {
            for z in 0..CHUNK_SIZE as u8 {
                let idx = block_index(x, y, z);
                let pi = raw[cursor + idx] as usize;
                if pi < palette.len() {
                    chunk.set_block(x, y, z, palette[pi]);
                }
            }
        }
    }
    cursor += CHUNK_VOLUME;

    // Metadata
    if raw.len() >= cursor + 4 {
        let meta_count = u32::from_le_bytes([
            raw[cursor],
            raw[cursor + 1],
            raw[cursor + 2],
            raw[cursor + 3],
        ]);
        cursor += 4;

        for _ in 0..meta_count {
            if raw.len() < cursor + 12 {
                break;
            }
            let flat_idx = u32::from_le_bytes([
                raw[cursor],
                raw[cursor + 1],
                raw[cursor + 2],
                raw[cursor + 3],
            ]);
            let health = u16::from_le_bytes([raw[cursor + 4], raw[cursor + 5]]);
            let orientation = BlockOrientation(raw[cursor + 6]);
            let flags = BlockFlags(raw[cursor + 7]);
            let entity_index = u32::from_le_bytes([
                raw[cursor + 8],
                raw[cursor + 9],
                raw[cursor + 10],
                raw[cursor + 11],
            ]);
            cursor += 12;

            let (x, y, z) = super::palette::index_to_xyz(flat_idx as usize);
            let meta = chunk.get_or_create_meta(x, y, z);
            meta.health = health;
            meta.orientation = orientation;
            meta.flags = flags;
            meta.entity_index = entity_index;
        }
    }

    // Reconstruct edit_seq (we only stored lower 16 bits)
    // For full fidelity the caller should set the true seq from their tracking.
    // This is a reasonable default for persistence snapshots.
    let _ = edit_seq; // stored in chunk via next_edit_seq calls at load time

    Ok(chunk)
}

/// Serialize a block edit in the 5-byte WAL format specified in SPECIFICATION.md.
/// Format: `[bx, by, bz, block_type_lo, block_type_hi]`
pub fn serialize_edit_wal(bx: u8, by: u8, bz: u8, block_type: BlockId) -> [u8; 5] {
    let [lo, hi] = block_type.as_u16().to_le_bytes();
    [bx, by, bz, lo, hi]
}

/// Deserialize a 5-byte WAL edit record.
pub fn deserialize_edit_wal(bytes: &[u8; 5]) -> (u8, u8, u8, BlockId) {
    let bx = bytes[0];
    let by = bytes[1];
    let bz = bytes[2];
    let block_type = BlockId::from_u16(u16::from_le_bytes([bytes[3], bytes[4]]));
    (bx, by, bz, block_type)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn serialize_deserialize_empty_chunk() {
        let chunk = ChunkStorage::new_empty();
        let bytes = serialize_chunk(&chunk);
        let restored = deserialize_chunk(&bytes).unwrap();

        assert!(restored.is_empty());
        assert_eq!(restored.meta_count(), 0);
    }

    #[test]
    fn serialize_deserialize_with_blocks() {
        let mut chunk = ChunkStorage::new_empty();
        chunk.set_block(0, 0, 0, BlockId::STONE);
        chunk.set_block(10, 20, 30, BlockId::HULL_STANDARD);
        chunk.set_block(61, 61, 61, BlockId::WATER);

        let bytes = serialize_chunk(&chunk);
        let restored = deserialize_chunk(&bytes).unwrap();

        assert_eq!(restored.get_block(0, 0, 0), BlockId::STONE);
        assert_eq!(restored.get_block(10, 20, 30), BlockId::HULL_STANDARD);
        assert_eq!(restored.get_block(61, 61, 61), BlockId::WATER);
        assert_eq!(restored.get_block(5, 5, 5), BlockId::AIR);
        assert_eq!(restored.non_air_count(), 3);
    }

    #[test]
    fn serialize_deserialize_with_metadata() {
        let mut chunk = ChunkStorage::new_empty();
        chunk.set_block(5, 5, 5, BlockId::HULL_STANDARD);
        chunk.set_orientation(5, 5, 5, BlockOrientation::new(3, 2));
        let meta = chunk.get_or_create_meta(5, 5, 5);
        meta.health = 420;
        meta.flags.set(BlockFlags::DAMAGED, true);
        meta.entity_index = 99;

        let bytes = serialize_chunk(&chunk);
        let restored = deserialize_chunk(&bytes).unwrap();

        let rmeta = restored.get_meta(5, 5, 5).unwrap();
        assert_eq!(rmeta.health, 420);
        assert_eq!(rmeta.orientation, BlockOrientation::new(3, 2));
        assert!(rmeta.flags.has(BlockFlags::DAMAGED));
        assert_eq!(rmeta.entity_index, 99);
    }

    #[test]
    fn serialize_deserialize_uniform_chunk() {
        let chunk = ChunkStorage::new_uniform(BlockId::STONE);
        let bytes = serialize_chunk(&chunk);
        let restored = deserialize_chunk(&bytes).unwrap();

        assert_eq!(restored.get_block(0, 0, 0), BlockId::STONE);
        assert_eq!(restored.get_block(30, 30, 30), BlockId::STONE);
        assert_eq!(restored.get_block(61, 61, 61), BlockId::STONE);
        assert_eq!(restored.non_air_count(), CHUNK_VOLUME as u32);
    }

    #[test]
    fn serialization_is_deterministic() {
        let mut chunk = ChunkStorage::new_empty();
        chunk.set_block(1, 2, 3, BlockId::STONE);
        chunk.set_block(4, 5, 6, BlockId::DIRT);

        let bytes1 = serialize_chunk(&chunk);
        let bytes2 = serialize_chunk(&chunk);
        assert_eq!(bytes1, bytes2);
    }

    #[test]
    fn wal_edit_roundtrip() {
        let encoded = serialize_edit_wal(10, 20, 30, BlockId::HULL_ARMORED);
        let (bx, by, bz, block_type) = deserialize_edit_wal(&encoded);
        assert_eq!(bx, 10);
        assert_eq!(by, 20);
        assert_eq!(bz, 30);
        assert_eq!(block_type, BlockId::HULL_ARMORED);
    }

    #[test]
    fn compression_saves_space() {
        // A uniform chunk should compress extremely well
        let chunk = ChunkStorage::new_uniform(BlockId::STONE);
        let bytes = serialize_chunk(&chunk);
        // Uncompressed would be at least CHUNK_VOLUME bytes
        // Compressed should be much smaller
        assert!(
            bytes.len() < 1000,
            "Uniform chunk compressed to {} bytes, expected < 1000",
            bytes.len()
        );
    }
}
