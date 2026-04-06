//! Chunk meshing: runs binary-greedy-meshing on a ChunkStorage to produce
//! format-agnostic greedy quads.
//!
//! This module lives in `core/` because both client and server need meshing:
//! - **Client**: converts quads → GPU vertices (BlockVertex) for rendering
//! - **Server**: converts quads → Rapier trimesh vertices for collision (Plan B)
//!
//! The output (`ChunkQuads`) is a plain data structure with no GPU or physics
//! dependencies. Each consumer converts it to their specific format.

use std::collections::BTreeSet;

use binary_greedy_meshing as bgm;

use super::chunk_storage::ChunkStorage;
use super::palette::CHUNK_SIZE;
use super::registry::BlockRegistry;

/// A single greedy-merged quad face, produced by the mesher.
///
/// Contains all the information needed by any consumer (renderer, physics)
/// to convert this quad into their target format. Vertex positions are
/// computed by bgm's `Face::vertices_packed()` for correctness.
#[derive(Clone, Debug)]
pub struct MesherQuad {
    /// Face direction index (0–5): Up, Down, Right, Left, Front, Back.
    pub face: u8,
    /// Block type ID that produced this quad.
    pub block_id: u16,
    /// The 4 corner vertex positions in chunk-local coordinates.
    /// Range: typically 0..62 for block positions, -1..63 at boundaries.
    /// Computed by bgm's Face::vertices_packed() — guaranteed correct axis mapping.
    /// Order: matches bgm's winding for the face direction.
    pub vertices: [[i8; 3]; 4],
}

/// Face index to normal vector mapping (matches binary-greedy-meshing Face order).
pub const FACE_NORMALS: [[f32; 3]; 6] = [
    [0.0, 1.0, 0.0],   // 0: Up (+Y)
    [0.0, -1.0, 0.0],  // 1: Down (-Y)
    [1.0, 0.0, 0.0],   // 2: Right (+X)
    [-1.0, 0.0, 0.0],  // 3: Left (-X)
    [0.0, 0.0, 1.0],   // 4: Front (+Z)
    [0.0, 0.0, -1.0],  // 5: Back (-Z)
];

/// Result of meshing a chunk: a list of greedy-merged quads.
pub struct ChunkQuads {
    pub quads: Vec<MesherQuad>,
}

impl ChunkQuads {
    pub fn is_empty(&self) -> bool {
        self.quads.is_empty()
    }

    pub fn quad_count(&self) -> usize {
        self.quads.len()
    }
}

/// Run binary-greedy-meshing on a chunk and produce format-agnostic quads.
///
/// This is the shared meshing entry point used by both client and server.
///
/// - `chunk`: the chunk to mesh
/// - `neighbors`: `[+X, -X, +Y, -Y, +Z, -Z]`, `None` = air
/// - `registry`: block property lookup
/// - `transparent_pass`: if true, mesh only transparent blocks; if false, mesh only opaque
pub fn mesh_chunk(
    chunk: &ChunkStorage,
    neighbors: &[Option<&ChunkStorage>; 6],
    registry: &BlockRegistry,
    transparent_pass: bool,
) -> ChunkQuads {
    if chunk.is_empty() && !transparent_pass {
        return ChunkQuads { quads: Vec::new() };
    }

    // Build the padded 64³ voxel buffer.
    let voxels = super::meshing_adapter::build_padded_buffer(
        chunk, neighbors, registry, transparent_pass,
    );

    // The padded buffer already handles opaque/transparent separation by
    // remapping block IDs to 0. No additional transparent set needed.
    let transparent_ids = BTreeSet::new();

    // Run the mesher.
    let mut mesher = bgm::Mesher::<{ CHUNK_SIZE }>::new();
    let opaque_mask = bgm::compute_opaque_mask::<{ CHUNK_SIZE }>(&voxels, &transparent_ids);
    let trans_mask = bgm::compute_transparent_mask::<{ CHUNK_SIZE }>(&voxels, &transparent_ids);
    mesher.fast_mesh(&voxels, &opaque_mask, &trans_mask);

    // Convert bgm quads to our format-agnostic representation.
    // Use bgm's Face::vertices_packed() for correct vertex positions —
    // the library's own axis mapping per face direction.
    let total_quads: usize = mesher.quads.iter().map(|v| v.len()).sum();
    let mut quads = Vec::with_capacity(total_quads);

    for face_idx in 0..6u8 {
        let face = bgm::Face::from(face_idx);
        for quad in &mesher.quads[face_idx as usize] {
            let packed_verts = face.vertices_packed(*quad);
            // bgm's vertices_packed() already returns chunk-local positions.
            // The padding offset is internal to the mesher — the output quad
            // positions are in the 0..62 usable range, not padded 1..63.
            let to_local = |v: &bgm::Vertex| -> [i8; 3] {
                [v.x() as i8, v.y() as i8, v.z() as i8]
            };
            quads.push(MesherQuad {
                face: face_idx,
                block_id: quad.voxel_id() as u16,
                vertices: [
                    to_local(&packed_verts[0]),
                    to_local(&packed_verts[1]),
                    to_local(&packed_verts[2]),
                    to_local(&packed_verts[3]),
                ],
            });
        }
    }

    ChunkQuads { quads }
}


#[cfg(test)]
mod tests {
    use super::*;
    use crate::block::{BlockId, ChunkStorage, BlockRegistry};

    #[test]
    fn empty_chunk_produces_no_quads() {
        let reg = BlockRegistry::new();
        let chunk = ChunkStorage::new_empty();
        let no_neighbors: [Option<&ChunkStorage>; 6] = [None; 6];
        let result = mesh_chunk(&chunk, &no_neighbors, &reg, false);
        assert!(result.is_empty());
    }

    #[test]
    fn single_block_produces_6_quads() {
        let reg = BlockRegistry::new();
        let mut chunk = ChunkStorage::new_empty();
        chunk.set_block(10, 10, 10, BlockId::STONE);

        let no_neighbors: [Option<&ChunkStorage>; 6] = [None; 6];
        let result = mesh_chunk(&chunk, &no_neighbors, &reg, false);

        assert_eq!(result.quad_count(), 6, "Single block should have 6 face quads");
        for q in &result.quads {
            assert_eq!(q.block_id, BlockId::STONE.as_u16());
            // Each quad should have 4 vertices with valid coordinates
            for v in &q.vertices {
                assert!(v[0] >= -1 && v[0] <= 63 && v[1] >= -1 && v[1] <= 63 && v[2] >= -1 && v[2] <= 63,
                    "Vertex {:?} out of range", v);
            }
        }
    }

    #[test]
    fn adjacent_blocks_merge_faces() {
        let reg = BlockRegistry::new();
        let mut chunk = ChunkStorage::new_empty();
        chunk.set_block(10, 10, 10, BlockId::STONE);
        chunk.set_block(11, 10, 10, BlockId::STONE);

        let no_neighbors: [Option<&ChunkStorage>; 6] = [None; 6];
        let result = mesh_chunk(&chunk, &no_neighbors, &reg, false);

        // Adjacent same-type blocks: shared face is culled, other faces may merge
        assert!(result.quad_count() < 12, "Should have fewer than 12 quads ({} found)", result.quad_count());
        assert!(result.quad_count() > 0);
    }

    #[test]
    fn transparent_pass_separation() {
        let reg = BlockRegistry::new();
        let mut chunk = ChunkStorage::new_empty();
        chunk.set_block(10, 10, 10, BlockId::WATER); // transparent
        chunk.set_block(15, 15, 15, BlockId::STONE); // opaque

        let no_neighbors: [Option<&ChunkStorage>; 6] = [None; 6];

        let opaque = mesh_chunk(&chunk, &no_neighbors, &reg, false);
        let transparent = mesh_chunk(&chunk, &no_neighbors, &reg, true);

        assert!(!opaque.is_empty(), "Opaque pass should have quads");
        assert!(!transparent.is_empty(), "Transparent pass should have quads");

        // Opaque quads should only be STONE
        for q in &opaque.quads {
            assert_eq!(q.block_id, BlockId::STONE.as_u16());
        }
        // Transparent quads should only be WATER
        for q in &transparent.quads {
            assert_eq!(q.block_id, BlockId::WATER.as_u16());
        }
    }

    #[test]
    fn block_at_high_local_coords_produces_quads() {
        let reg = BlockRegistry::new();
        let mut chunk = ChunkStorage::new_empty();
        // Place a block at high local coordinates (like the starter ship does)
        chunk.set_block(59, 0, 57, BlockId::HULL_STANDARD);

        // Verify block is actually there
        assert_eq!(chunk.get_block(59, 0, 57), BlockId::HULL_STANDARD);
        assert_eq!(chunk.non_air_count(), 1);

        // Check padded buffer
        let no_neighbors: [Option<&ChunkStorage>; 6] = [None; 6];
        let buf = crate::block::meshing_adapter::build_padded_buffer(
            &chunk, &no_neighbors, &reg, false,
        );
        let pi = crate::block::meshing_adapter::padded_index(59, 0, 57);
        assert_ne!(buf[pi], 0, "Padded buffer at (59,0,57) should be non-zero, got 0. Index={pi}");

        let non_zero: usize = buf.iter().filter(|&&v| v != 0).count();
        assert_eq!(non_zero, 1, "Expected exactly 1 non-zero voxel in padded buffer, got {non_zero}");

        // Now mesh
        let result = mesh_chunk(&chunk, &no_neighbors, &reg, false);
        assert_eq!(result.quad_count(), 6, "Block at (59,0,57) should have 6 quads, got {}", result.quad_count());
    }

    #[test]
    fn starter_ship_serialization_roundtrip_meshes() {
        let reg = BlockRegistry::new();
        let layout = crate::block::ship_grid::StarterShipLayout::default_starter();
        let grid = crate::block::ship_grid::build_starter_ship(&layout);

        for (key, chunk) in grid.iter_chunks() {
            let non_air = chunk.non_air_count();
            if non_air == 0 {
                continue;
            }

            // Serialize and deserialize
            let compressed = crate::block::serialization::serialize_chunk(chunk);
            let restored = crate::block::serialization::deserialize_chunk(&compressed).unwrap();

            assert_eq!(
                restored.non_air_count(), non_air,
                "Chunk {key}: non_air mismatch after serialization roundtrip"
            );

            // Mesh the deserialized chunk (same as client would do)
            let neighbors: [Option<&ChunkStorage>; 6] = [None; 6];
            let quads_restored = mesh_chunk(&restored, &neighbors, &reg, false);
            let quads_original = mesh_chunk(chunk, &neighbors, &reg, false);

            assert_eq!(
                quads_restored.quad_count(), quads_original.quad_count(),
                "Chunk {key}: quad count mismatch after serialization ({} vs {})",
                quads_restored.quad_count(), quads_original.quad_count()
            );
            assert!(
                quads_original.quad_count() > 0,
                "Chunk {key}: expected quads from non-empty chunk (non_air={non_air})"
            );
        }
    }

    #[test]
    fn vertices_have_correct_positions() {
        let reg = BlockRegistry::new();
        let mut chunk = ChunkStorage::new_empty();
        chunk.set_block(5, 10, 15, BlockId::STONE);

        let no_neighbors: [Option<&ChunkStorage>; 6] = [None; 6];
        let result = mesh_chunk(&chunk, &no_neighbors, &reg, false);

        assert_eq!(result.quad_count(), 6);

        // Dump all vertex positions per face to understand the exact mapping.
        let face_names = ["Up", "Down", "Right", "Left", "Front", "Back"];
        for q in &result.quads {
            eprintln!("Face {} ({}):", q.face, face_names[q.face as usize]);
            for (i, v) in q.vertices.iter().enumerate() {
                eprintln!("  v{}: ({}, {}, {})", i, v[0], v[1], v[2]);
            }
        }
        // A block at chunk-local (5, 10, 15) occupies (5..6, 10..11, 15..16).
        // Every vertex coordinate must be on the block boundary.
        for q in &result.quads {
            for v in &q.vertices {
                assert!(v[0] >= 5 && v[0] <= 6,
                    "Face {}: X={} should be 5 or 6", face_names[q.face as usize], v[0]);
                assert!(v[1] >= 10 && v[1] <= 11,
                    "Face {}: Y={} should be 10 or 11", face_names[q.face as usize], v[1]);
                assert!(v[2] >= 15 && v[2] <= 16,
                    "Face {}: Z={} should be 15 or 16", face_names[q.face as usize], v[2]);
            }
        }
    }
}
