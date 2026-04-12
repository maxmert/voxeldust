use super::block_id::BlockId;
use super::chunk_storage::ChunkStorage;
use super::registry::BlockRegistry;

/// Size of the padded voxel buffer expected by `binary-greedy-meshing`.
pub const PADDED_SIZE: usize = 64;
pub const PADDED_VOLUME: usize = PADDED_SIZE * PADDED_SIZE * PADDED_SIZE; // 262,144

/// Index into the padded 64³ buffer in ZXY layout.
///
/// `binary-greedy-meshing` expects: `index = (z+1) + (x+1)*64 + (y+1)*4096`
/// where the +1 accounts for the padding border on each axis.
///
/// `x`, `y`, `z` are in range -1..=62 (signed, including padding).
#[inline(always)]
pub fn padded_index(x: i32, y: i32, z: i32) -> usize {
    debug_assert!((-1..=62).contains(&x) && (-1..=62).contains(&y) && (-1..=62).contains(&z));
    ((z + 1) as usize) + ((x + 1) as usize) * 64 + ((y + 1) as usize) * 4096
}

/// Neighbor directions: +X, -X, +Y, -Y, +Z, -Z
pub const NEIGHBOR_POS_X: usize = 0;
pub const NEIGHBOR_NEG_X: usize = 1;
pub const NEIGHBOR_POS_Y: usize = 2;
pub const NEIGHBOR_NEG_Y: usize = 3;
pub const NEIGHBOR_POS_Z: usize = 4;
pub const NEIGHBOR_NEG_Z: usize = 5;

/// Build the 64³ padded voxel buffer from a `ChunkStorage` and its 6 face neighbors.
///
/// Each entry is a `u16` block ID. The mesher treats 0 as "passable" (air/transparent),
/// so for the **opaque pass**, transparent blocks are remapped to 0.
/// For the **transparent pass**, opaque blocks are remapped to 0.
///
/// Neighbors array: `[+X, -X, +Y, -Y, +Z, -Z]`. `None` = treat as air.
pub fn build_padded_buffer(
    chunk: &ChunkStorage,
    neighbors: &[Option<&ChunkStorage>; 6],
    registry: &BlockRegistry,
    transparent_pass: bool,
) -> Vec<u16> {
    let mut buf = Vec::new();
    build_padded_buffer_into(&mut buf, chunk, neighbors, registry, transparent_pass);
    buf
}

/// Like `build_padded_buffer`, but writes into a caller-provided buffer for reuse.
/// The buffer is resized and zeroed; its prior allocation is preserved.
pub fn build_padded_buffer_into(
    buf: &mut Vec<u16>,
    chunk: &ChunkStorage,
    neighbors: &[Option<&ChunkStorage>; 6],
    registry: &BlockRegistry,
    transparent_pass: bool,
) {
    buf.clear();
    buf.resize(PADDED_VOLUME, 0);

    // Fill interior (62³ usable voxels)
    for x in 0..62i32 {
        for y in 0..62i32 {
            for z in 0..62i32 {
                let id = chunk.get_block(x as u8, y as u8, z as u8);
                let write = remap_for_pass(id, registry, transparent_pass);
                buf[padded_index(x, y, z)] = write;
            }
        }
    }

    // Fill 6 padding faces from neighbors
    // +X face: padded x=62, read from neighbor[0] at x=0
    if let Some(n) = neighbors[NEIGHBOR_POS_X] {
        for y in 0..62i32 {
            for z in 0..62i32 {
                let id = n.get_block(0, y as u8, z as u8);
                buf[padded_index(62, y, z)] = remap_for_pass(id, registry, transparent_pass);
            }
        }
    }

    // -X face: padded x=-1, read from neighbor[1] at x=61
    if let Some(n) = neighbors[NEIGHBOR_NEG_X] {
        for y in 0..62i32 {
            for z in 0..62i32 {
                let id = n.get_block(61, y as u8, z as u8);
                buf[padded_index(-1, y, z)] = remap_for_pass(id, registry, transparent_pass);
            }
        }
    }

    // +Y face: padded y=62, read from neighbor[2] at y=0
    if let Some(n) = neighbors[NEIGHBOR_POS_Y] {
        for x in 0..62i32 {
            for z in 0..62i32 {
                let id = n.get_block(x as u8, 0, z as u8);
                buf[padded_index(x, 62, z)] = remap_for_pass(id, registry, transparent_pass);
            }
        }
    }

    // -Y face: padded y=-1, read from neighbor[3] at y=61
    if let Some(n) = neighbors[NEIGHBOR_NEG_Y] {
        for x in 0..62i32 {
            for z in 0..62i32 {
                let id = n.get_block(x as u8, 61, z as u8);
                buf[padded_index(x, -1, z)] = remap_for_pass(id, registry, transparent_pass);
            }
        }
    }

    // +Z face: padded z=62, read from neighbor[4] at z=0
    if let Some(n) = neighbors[NEIGHBOR_POS_Z] {
        for x in 0..62i32 {
            for y in 0..62i32 {
                let id = n.get_block(x as u8, y as u8, 0);
                buf[padded_index(x, y, 62)] = remap_for_pass(id, registry, transparent_pass);
            }
        }
    }

    // -Z face: padded z=-1, read from neighbor[5] at z=61
    if let Some(n) = neighbors[NEIGHBOR_NEG_Z] {
        for x in 0..62i32 {
            for y in 0..62i32 {
                let id = n.get_block(x as u8, y as u8, 61);
                buf[padded_index(x, y, -1)] = remap_for_pass(id, registry, transparent_pass);
            }
        }
    }
}

/// Build a filtered padded buffer for sub-grid-aware meshing.
///
/// - `sub_grid_filter = None`: include only blocks NOT in any sub-grid (root grid).
/// - `sub_grid_filter = Some(sg_id)`: include only blocks in that specific sub-grid.
///
/// Fused single-pass: builds the padded buffer and applies the sub-grid filter in one
/// 62³ traversal instead of two separate passes. Neighbor padding is only filled for
/// the root grid; sub-grid meshes are self-contained (padding stays zero).
pub fn build_padded_buffer_filtered(
    chunk: &ChunkStorage,
    neighbors: &[Option<&ChunkStorage>; 6],
    registry: &BlockRegistry,
    transparent_pass: bool,
    chunk_key: glam::IVec3,
    sub_grid_assignments: &std::collections::HashMap<glam::IVec3, u32>,
    sub_grid_filter: Option<u32>,
) -> Vec<u16> {
    let mut buf = Vec::new();
    build_padded_buffer_filtered_into(
        &mut buf, chunk, neighbors, registry, transparent_pass,
        chunk_key, sub_grid_assignments, sub_grid_filter,
    );
    buf
}

/// Like `build_padded_buffer_filtered`, but writes into a caller-provided buffer for reuse.
/// The buffer is resized and zeroed; its prior allocation is preserved.
pub fn build_padded_buffer_filtered_into(
    buf: &mut Vec<u16>,
    chunk: &ChunkStorage,
    neighbors: &[Option<&ChunkStorage>; 6],
    registry: &BlockRegistry,
    transparent_pass: bool,
    chunk_key: glam::IVec3,
    sub_grid_assignments: &std::collections::HashMap<glam::IVec3, u32>,
    sub_grid_filter: Option<u32>,
) {
    buf.clear();
    buf.resize(PADDED_VOLUME, 0);
    let cs = super::CHUNK_SIZE as i32;

    // Build a dense lookup array from the HashMap for O(1) sub-grid membership
    // checks in the inner loop (replaces 238K HashMap lookups with array indexing).
    let mut sg_lookup = vec![0u32; 62 * 62 * 62];
    let base = chunk_key * cs;
    for (&world_pos, &sg_id) in sub_grid_assignments {
        let local = world_pos - base;
        if (0..62).contains(&local.x) && (0..62).contains(&local.y) && (0..62).contains(&local.z) {
            sg_lookup[(local.x * 62 * 62 + local.y * 62 + local.z) as usize] = sg_id;
        }
    }

    // Fused interior pass: remap for opaque/transparent AND filter by sub-grid.
    for x in 0..62i32 {
        for y in 0..62i32 {
            for z in 0..62i32 {
                let block_sg = sg_lookup[(x * 62 * 62 + y * 62 + z) as usize];
                let keep = match sub_grid_filter {
                    None => block_sg == 0,        // root: only blocks NOT in any sub-grid
                    Some(id) => block_sg == id,    // specific sub-grid
                };
                if keep {
                    let id = chunk.get_block(x as u8, y as u8, z as u8);
                    buf[padded_index(x, y, z)] = remap_for_pass(id, registry, transparent_pass);
                }
            }
        }
    }

    // Fill neighbor padding only for root grid (sub-grid meshes are self-contained
    // to avoid cross-chunk face merging artifacts when the sub-grid rotates).
    if sub_grid_filter.is_none() {
        if let Some(n) = neighbors[NEIGHBOR_POS_X] {
            for y in 0..62i32 {
                for z in 0..62i32 {
                    let id = n.get_block(0, y as u8, z as u8);
                    buf[padded_index(62, y, z)] = remap_for_pass(id, registry, transparent_pass);
                }
            }
        }
        if let Some(n) = neighbors[NEIGHBOR_NEG_X] {
            for y in 0..62i32 {
                for z in 0..62i32 {
                    let id = n.get_block(61, y as u8, z as u8);
                    buf[padded_index(-1, y, z)] = remap_for_pass(id, registry, transparent_pass);
                }
            }
        }
        if let Some(n) = neighbors[NEIGHBOR_POS_Y] {
            for x in 0..62i32 {
                for z in 0..62i32 {
                    let id = n.get_block(x as u8, 0, z as u8);
                    buf[padded_index(x, 62, z)] = remap_for_pass(id, registry, transparent_pass);
                }
            }
        }
        if let Some(n) = neighbors[NEIGHBOR_NEG_Y] {
            for x in 0..62i32 {
                for z in 0..62i32 {
                    let id = n.get_block(x as u8, 61, z as u8);
                    buf[padded_index(x, -1, z)] = remap_for_pass(id, registry, transparent_pass);
                }
            }
        }
        if let Some(n) = neighbors[NEIGHBOR_POS_Z] {
            for x in 0..62i32 {
                for y in 0..62i32 {
                    let id = n.get_block(x as u8, y as u8, 0);
                    buf[padded_index(x, y, 62)] = remap_for_pass(id, registry, transparent_pass);
                }
            }
        }
        if let Some(n) = neighbors[NEIGHBOR_NEG_Z] {
            for x in 0..62i32 {
                for y in 0..62i32 {
                    let id = n.get_block(x as u8, y as u8, 61);
                    buf[padded_index(x, y, -1)] = remap_for_pass(id, registry, transparent_pass);
                }
            }
        }
    }
}

/// Remap a block ID for the given meshing pass.
/// - Opaque pass: transparent/air → 0, opaque → original ID
/// - Transparent pass: opaque → 0, transparent (non-air) → original ID
#[inline(always)]
fn remap_for_pass(id: BlockId, registry: &BlockRegistry, transparent_pass: bool) -> u16 {
    if id.is_air() {
        return 0;
    }
    let is_transparent = registry.is_transparent(id);
    if transparent_pass {
        if is_transparent { id.as_u16() } else { 0 }
    } else {
        if is_transparent { 0 } else { id.as_u16() }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn padded_index_corners() {
        // First usable voxel at (0,0,0) maps to padded (1,1,1)
        assert_eq!(padded_index(0, 0, 0), 1 + 64 + 4096);
        // Last usable voxel at (61,61,61) maps to padded (62,62,62)
        assert_eq!(padded_index(61, 61, 61), 62 + 62 * 64 + 62 * 4096);
        // Padding at (-1,-1,-1) maps to (0,0,0) = index 0
        assert_eq!(padded_index(-1, -1, -1), 0);
    }

    #[test]
    fn opaque_pass_filters_transparent() {
        let reg = BlockRegistry::new();
        let mut chunk = ChunkStorage::new_empty();
        chunk.set_block(5, 5, 5, BlockId::STONE); // opaque
        chunk.set_block(6, 5, 5, BlockId::WATER); // transparent

        let no_neighbors: [Option<&ChunkStorage>; 6] = [None; 6];
        let buf = build_padded_buffer(&chunk, &no_neighbors, &reg, false);

        assert_ne!(buf[padded_index(5, 5, 5)], 0, "Stone should be visible in opaque pass");
        assert_eq!(buf[padded_index(6, 5, 5)], 0, "Water should be invisible in opaque pass");
    }

    #[test]
    fn transparent_pass_filters_opaque() {
        let reg = BlockRegistry::new();
        let mut chunk = ChunkStorage::new_empty();
        chunk.set_block(5, 5, 5, BlockId::STONE); // opaque
        chunk.set_block(6, 5, 5, BlockId::WATER); // transparent

        let no_neighbors: [Option<&ChunkStorage>; 6] = [None; 6];
        let buf = build_padded_buffer(&chunk, &no_neighbors, &reg, true);

        assert_eq!(buf[padded_index(5, 5, 5)], 0, "Stone should be invisible in transparent pass");
        assert_ne!(buf[padded_index(6, 5, 5)], 0, "Water should be visible in transparent pass");
    }

    #[test]
    fn neighbor_padding() {
        let reg = BlockRegistry::new();
        let chunk = ChunkStorage::new_empty();
        let mut neighbor_pos_x = ChunkStorage::new_empty();
        neighbor_pos_x.set_block(0, 10, 10, BlockId::STONE);

        let neighbors: [Option<&ChunkStorage>; 6] = [
            Some(&neighbor_pos_x), None, None, None, None, None,
        ];
        let buf = build_padded_buffer(&chunk, &neighbors, &reg, false);

        // Stone from neighbor should appear at padded x=62
        assert_ne!(buf[padded_index(62, 10, 10)], 0, "Neighbor block should be in padding");
        // Interior should still be air
        assert_eq!(buf[padded_index(61, 10, 10)], 0, "Interior should be air");
    }

    #[test]
    fn all_air_buffer_is_zero() {
        let reg = BlockRegistry::new();
        let chunk = ChunkStorage::new_empty();
        let no_neighbors: [Option<&ChunkStorage>; 6] = [None; 6];
        let buf = build_padded_buffer(&chunk, &no_neighbors, &reg, false);
        assert!(buf.iter().all(|&v| v == 0), "All-air chunk should produce all-zero buffer");
    }

    #[test]
    fn buffer_size_is_correct() {
        let reg = BlockRegistry::new();
        let chunk = ChunkStorage::new_empty();
        let no_neighbors: [Option<&ChunkStorage>; 6] = [None; 6];
        let buf = build_padded_buffer(&chunk, &no_neighbors, &reg, false);
        assert_eq!(buf.len(), PADDED_VOLUME);
    }
}
