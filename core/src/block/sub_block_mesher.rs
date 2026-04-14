//! Sub-block element mesh generation.
//!
//! Generates thin geometry for sub-block elements (wires, rails, pipes, etc.)
//! placed on block faces. Cannot use binary-greedy-meshing (not full cubes).
//! Each element type produces a small set of triangles positioned on the
//! host block's face.

use super::chunk_storage::ChunkStorage;
use super::palette::{index_to_xyz, CHUNK_SIZE};
use super::sub_block::{SubBlockElement, SubBlockType};

/// A single vertex for sub-block rendering. Same format as BlockVertex
/// (position, normal, color) so sub-blocks share the block shader pipeline.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct SubBlockVertex {
    pub position: [f32; 3],
    pub normal: [f32; 3],
    pub color: [f32; 3],
}

#[cfg(feature = "bytemuck")]
unsafe impl bytemuck::Pod for SubBlockVertex {}
#[cfg(feature = "bytemuck")]
unsafe impl bytemuck::Zeroable for SubBlockVertex {}

/// GPU-ready mesh data for all sub-blocks in a chunk.
pub struct SubBlockMeshData {
    pub vertices: Vec<SubBlockVertex>,
    pub indices: Vec<u32>,
}

impl SubBlockMeshData {
    pub fn is_empty(&self) -> bool {
        self.indices.is_empty()
    }
}

/// Face normals indexed by face ID (0-5): +X, -X, +Y, -Y, +Z, -Z.
const FACE_NORMALS: [[f32; 3]; 6] = [
    [1.0, 0.0, 0.0],
    [-1.0, 0.0, 0.0],
    [0.0, 1.0, 0.0],
    [0.0, -1.0, 0.0],
    [0.0, 0.0, 1.0],
    [0.0, 0.0, -1.0],
];

/// Two tangent vectors for each face, defining the local UV plane.
/// (tangent_u, tangent_v) such that normal = u × v.
const FACE_TANGENTS: [([f32; 3], [f32; 3]); 6] = [
    ([0.0, 0.0, -1.0], [0.0, 1.0, 0.0]),  // +X: u=−Z, v=+Y
    ([0.0, 0.0, 1.0],  [0.0, 1.0, 0.0]),  // −X: u=+Z, v=+Y
    ([1.0, 0.0, 0.0],  [0.0, 0.0, 1.0]),  // +Y: u=+X, v=+Z
    ([1.0, 0.0, 0.0],  [0.0, 0.0, -1.0]), // −Y: u=+X, v=−Z
    ([1.0, 0.0, 0.0],  [0.0, 1.0, 0.0]),  // +Z: u=+X, v=+Y
    ([-1.0, 0.0, 0.0], [0.0, 1.0, 0.0]),  // −Z: u=−X, v=+Y
];

/// Generate mesh data for all sub-block elements in a chunk.
/// For chainable types (wires, rails, pipes), computes connection masks
/// from neighboring sub-blocks to generate correct joint geometry.
pub fn mesh_sub_blocks(chunk: &ChunkStorage) -> SubBlockMeshData {
    use super::sub_block::{compute_connection_mask, SubBlockElement as SBE};

    let mut vertices = Vec::new();
    let mut indices = Vec::new();

    for (flat_idx, elements) in chunk.iter_sub_blocks() {
        let (bx, by, bz) = index_to_xyz(flat_idx as usize);
        let block_origin = [bx as f32, by as f32, bz as f32];
        let local_pos = glam::IVec3::new(bx as i32, by as i32, bz as i32);

        for elem in elements {
            // For chainable types, compute connection mask from neighbors.
            let connection_mask = if elem.element_type.is_chainable() {
                compute_connection_mask(
                    local_pos,
                    elem.face,
                    elem.element_type,
                    |neighbor_pos| {
                        // Clamp to chunk bounds (0..62).
                        if neighbor_pos.x < 0 || neighbor_pos.x >= CHUNK_SIZE as i32
                            || neighbor_pos.y < 0 || neighbor_pos.y >= CHUNK_SIZE as i32
                            || neighbor_pos.z < 0 || neighbor_pos.z >= CHUNK_SIZE as i32
                        {
                            return Vec::new();
                        }
                        chunk.get_sub_blocks(
                            neighbor_pos.x as u8,
                            neighbor_pos.y as u8,
                            neighbor_pos.z as u8,
                        ).to_vec()
                    },
                )
            } else {
                0 // Non-chainable types don't auto-join.
            };

            generate_element_mesh(
                &block_origin,
                elem,
                connection_mask,
                &mut vertices,
                &mut indices,
            );
        }
    }

    SubBlockMeshData { vertices, indices }
}

/// Generate mesh for a single sub-block element.
/// `connection_mask`: 4-bit mask for chainable types (bit 0=+u, 1=-u, 2=+v, 3=-v).
fn generate_element_mesh(
    block_origin: &[f32; 3],
    elem: &SubBlockElement,
    connection_mask: u8,
    vertices: &mut Vec<SubBlockVertex>,
    indices: &mut Vec<u32>,
) {
    let face = elem.face.min(5) as usize;
    let normal = FACE_NORMALS[face];
    let (tan_u, tan_v) = FACE_TANGENTS[face];

    let color_rgb = elem.element_type.color();
    let color = [
        color_rgb[0] as f32 / 255.0,
        color_rgb[1] as f32 / 255.0,
        color_rgb[2] as f32 / 255.0,
    ];

    // Element-specific geometry. All dimensions are fractions of a block (1.0 = 1 block).
    match elem.element_type {
        SubBlockType::PowerWire | SubBlockType::SignalWire | SubBlockType::Cable => {
            // Bitmask auto-tiling: emit strip segments from center toward each connected edge.
            emit_wire_from_mask(block_origin, face, &normal, &tan_u, &tan_v, &color,
                connection_mask, 0.03, 0.03, vertices, indices);
        }
        SubBlockType::Rail | SubBlockType::ConveyorBelt => {
            // Bitmask auto-tiling for rails (two parallel strips + cross-ties).
            // For now, use mask to determine primary direction.
            let rot = rotation_from_mask(connection_mask);
            emit_face_strip(block_origin, face, &normal, &tan_u, &tan_v, &color,
                0.5, 0.3, 0.5, 0.025, 0.05, rot, vertices, indices);
            emit_face_strip(block_origin, face, &normal, &tan_u, &tan_v, &color,
                0.5, 0.7, 0.5, 0.025, 0.05, rot, vertices, indices);
            for i in 0..4 {
                let u = 0.125 + i as f32 * 0.25;
                emit_face_strip(block_origin, face, &normal, &tan_u, &tan_v, &color,
                    u, 0.5, 0.04, 0.22, 0.03, rot, vertices, indices);
            }
        }
        SubBlockType::Pipe | SubBlockType::PipeValve | SubBlockType::PipePump => {
            // Bitmask auto-tiling for pipes.
            emit_wire_from_mask(block_origin, face, &normal, &tan_u, &tan_v, &color,
                connection_mask, 0.08, 0.08, vertices, indices);
        }
        SubBlockType::Ladder => {
            // Two vertical side rails spanning full block height + rungs.
            // Side rails (full block span along v-axis).
            emit_face_strip(block_origin, face, &normal, &tan_u, &tan_v, &color,
                0.2, 0.5, 0.025, 0.5, 0.025, elem.rotation, vertices, indices);
            emit_face_strip(block_origin, face, &normal, &tan_u, &tan_v, &color,
                0.8, 0.5, 0.025, 0.5, 0.025, elem.rotation, vertices, indices);
            // Rungs: 4 horizontal bars.
            for i in 0..4 {
                let v = 0.125 + i as f32 * 0.25;
                emit_face_strip(block_origin, face, &normal, &tan_u, &tan_v, &color,
                    0.5, v, 0.28, 0.02, 0.04, elem.rotation, vertices, indices);
            }
        }
        SubBlockType::SurfaceLight => {
            // Small flat disc (square approximation), slightly proud of surface.
            emit_face_strip(block_origin, face, &normal, &tan_u, &tan_v, &color,
                0.5, 0.5, 0.08, 0.08, 0.02, elem.rotation, vertices, indices);
        }
        SubBlockType::RotorMount | SubBlockType::HingeMount => {
            // Bearing assembly: outer ring + axle pin.
            let bearing_color = [0.70, 0.74, 0.78]; // metallic silver
            let axle_color = [0.55, 0.58, 0.62]; // darker steel
            // Outer bearing disc (thick, visible ring).
            emit_face_strip(block_origin, face, &normal, &tan_u, &tan_v, &bearing_color,
                0.5, 0.5, 0.2, 0.2, 0.06, elem.rotation, vertices, indices);
            // Inner axle pin (thin, protruding from center).
            emit_face_strip(block_origin, face, &normal, &tan_u, &tan_v, &axle_color,
                0.5, 0.5, 0.05, 0.05, 0.12, elem.rotation, vertices, indices);
        }
        SubBlockType::PistonMount | SubBlockType::SliderMount => {
            // Piston housing: outer cylinder + rod guide opening.
            let housing_color = [0.55, 0.58, 0.62]; // dark metallic
            let guide_color = [0.45, 0.48, 0.52]; // darker interior
            // Outer housing box.
            emit_face_strip(block_origin, face, &normal, &tan_u, &tan_v, &housing_color,
                0.5, 0.5, 0.15, 0.15, 0.12, elem.rotation, vertices, indices);
            // Inner rod guide (slightly recessed center).
            emit_face_strip(block_origin, face, &normal, &tan_u, &tan_v, &guide_color,
                0.5, 0.5, 0.08, 0.08, 0.14, elem.rotation, vertices, indices);
        }
        _ => {
            // Generic small box for any unhandled type.
            emit_face_strip(block_origin, face, &normal, &tan_u, &tan_v, &color,
                0.5, 0.5, 0.12, 0.12, 0.05, elem.rotation, vertices, indices);
        }
    }
}

/// Determine primary rotation from connection mask (for non-wire chainable types like rails).
/// If connections along v-axis (bits 2,3): rotation=1. Otherwise rotation=0.
fn rotation_from_mask(mask: u8) -> u8 {
    let has_v = (mask & 0b1100) != 0;
    let has_u = (mask & 0b0011) != 0;
    if has_v && !has_u { 1 } else { 0 }
}

/// Emit wire/pipe mesh based on 4-bit connection mask.
/// Generates strip segments from the face center toward each connected edge.
/// If no connections, emits a small dot at center.
fn emit_wire_from_mask(
    block_origin: &[f32; 3],
    face: usize,
    normal: &[f32; 3],
    tan_u: &[f32; 3],
    tan_v: &[f32; 3],
    color: &[f32; 3],
    mask: u8,
    half_width: f32,
    thickness: f32,
    vertices: &mut Vec<SubBlockVertex>,
    indices: &mut Vec<u32>,
) {
    if mask == 0 {
        // No connections: small dot at center.
        emit_face_strip(block_origin, face, normal, tan_u, tan_v, color,
            0.5, 0.5, 0.06, 0.06, thickness, 0, vertices, indices);
        return;
    }

    // Emit a strip segment from center (0.5, 0.5) toward each connected edge.
    // +u (bit 0): center → right edge (u=0.5..1.0)
    if mask & 1 != 0 {
        emit_face_strip(block_origin, face, normal, tan_u, tan_v, color,
            0.75, 0.5, 0.25, half_width, thickness, 0, vertices, indices);
    }
    // -u (bit 1): center → left edge (u=0.0..0.5)
    if mask & 2 != 0 {
        emit_face_strip(block_origin, face, normal, tan_u, tan_v, color,
            0.25, 0.5, 0.25, half_width, thickness, 0, vertices, indices);
    }
    // +v (bit 2): center → top edge (v=0.5..1.0)
    if mask & 4 != 0 {
        emit_face_strip(block_origin, face, normal, tan_u, tan_v, color,
            0.5, 0.75, half_width, 0.25, thickness, 0, vertices, indices);
    }
    // -v (bit 3): center → bottom edge (v=0.0..0.5)
    if mask & 8 != 0 {
        emit_face_strip(block_origin, face, normal, tan_u, tan_v, color,
            0.5, 0.25, half_width, 0.25, thickness, 0, vertices, indices);
    }

    // Center junction dot (always present when any connections exist).
    emit_face_strip(block_origin, face, normal, tan_u, tan_v, color,
        0.5, 0.5, half_width, half_width, thickness, 0, vertices, indices);
}

/// Emit a 3D box (6 faces, 24 vertices, 36 indices) positioned on a block face.
///
/// `center_u/center_v`: center position on the face in [0,1] UV space.
/// `half_u/half_v`: half-extents along the tangent axes.
/// `thickness`: how far the element protrudes from the face.
/// `rotation`: 0-3, 90° increments around the face normal.
fn emit_face_strip(
    block_origin: &[f32; 3],
    face: usize,
    normal: &[f32; 3],
    tan_u: &[f32; 3],
    tan_v: &[f32; 3],
    color: &[f32; 3],
    center_u: f32,
    center_v: f32,
    half_u: f32,
    half_v: f32,
    thickness: f32,
    rotation: u8,
    vertices: &mut Vec<SubBlockVertex>,
    indices: &mut Vec<u32>,
) {
    // Apply rotation: swap/negate tangent axes.
    let (eff_u, eff_v) = match rotation & 0x03 {
        0 => (*tan_u, *tan_v),
        1 => (*tan_v, neg3(tan_u)),
        2 => (neg3(tan_u), neg3(tan_v)),
        3 => (neg3(tan_v), *tan_u),
        _ => (*tan_u, *tan_v),
    };

    // Face center: block_origin + 0.5 (block center) + normal * 0.5 (face surface)
    // + small epsilon outward to prevent z-fighting with the block face.
    const FACE_OFFSET_EPSILON: f32 = 0.005;
    let face_center = [
        block_origin[0] + 0.5 + normal[0] * (0.5 + FACE_OFFSET_EPSILON),
        block_origin[1] + 0.5 + normal[1] * (0.5 + FACE_OFFSET_EPSILON),
        block_origin[2] + 0.5 + normal[2] * (0.5 + FACE_OFFSET_EPSILON),
    ];

    // Element center on the face.
    let cx = face_center[0] + eff_u[0] * (center_u - 0.5) + eff_v[0] * (center_v - 0.5);
    let cy = face_center[1] + eff_u[1] * (center_u - 0.5) + eff_v[1] * (center_v - 0.5);
    let cz = face_center[2] + eff_u[2] * (center_u - 0.5) + eff_v[2] * (center_v - 0.5);

    // 8 corners of the box.
    let base_idx = vertices.len() as u32;
    for &du in &[-half_u, half_u] {
        for &dv in &[-half_v, half_v] {
            for &dn in &[0.0f32, thickness] {
                let px = cx + eff_u[0] * du + eff_v[0] * dv + normal[0] * dn;
                let py = cy + eff_u[1] * du + eff_v[1] * dv + normal[1] * dn;
                let pz = cz + eff_u[2] * du + eff_v[2] * dv + normal[2] * dn;
                vertices.push(SubBlockVertex {
                    position: [px, py, pz],
                    normal: *normal,
                    color: *color,
                });
            }
        }
    }

    // 6 faces × 2 triangles = 12 triangles = 36 indices.
    // Box vertices: 0=(-u,-v,0), 1=(-u,-v,t), 2=(-u,+v,0), 3=(-u,+v,t),
    //               4=(+u,-v,0), 5=(+u,-v,t), 6=(+u,+v,0), 7=(+u,+v,t)
    let box_indices: &[u32] = &[
        // Front (normal direction, +n):
        1, 3, 7, 1, 7, 5,
        // Back (−n):
        0, 6, 2, 0, 4, 6,
        // Top (+v):
        2, 3, 7, 2, 7, 6,
        // Bottom (−v):
        0, 5, 1, 0, 4, 5,
        // Right (+u):
        4, 7, 6, 4, 5, 7,
        // Left (−u):
        0, 2, 3, 0, 3, 1,
    ];
    for &i in box_indices {
        indices.push(base_idx + i);
    }
}

fn neg3(v: &[f32; 3]) -> [f32; 3] {
    [-v[0], -v[1], -v[2]]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mesh_empty_chunk() {
        let chunk = ChunkStorage::new_empty();
        let mesh = mesh_sub_blocks(&chunk);
        assert!(mesh.is_empty());
    }

    #[test]
    fn mesh_single_wire() {
        let mut chunk = ChunkStorage::new_empty();
        chunk.set_block(5, 5, 5, crate::block::BlockId::HULL_STANDARD);
        chunk.add_sub_block(5, 5, 5, SubBlockElement {
            face: 0,
            element_type: SubBlockType::PowerWire,
            rotation: 0,
            flags: 0,
        });
        let mesh = mesh_sub_blocks(&chunk);
        assert!(!mesh.is_empty());
        // One wire = 1 box = 8 vertices, 36 indices.
        assert_eq!(mesh.vertices.len(), 8);
        assert_eq!(mesh.indices.len(), 36);
    }

    #[test]
    fn mesh_ladder_produces_geometry() {
        let mut chunk = ChunkStorage::new_empty();
        chunk.set_block(3, 3, 3, crate::block::BlockId::HULL_STANDARD);
        chunk.add_sub_block(3, 3, 3, SubBlockElement {
            face: 4, // +Z face
            element_type: SubBlockType::Ladder,
            rotation: 0,
            flags: 0,
        });
        let mesh = mesh_sub_blocks(&chunk);
        // Ladder = 2 rails + 4 rungs = 6 boxes = 48 vertices, 216 indices.
        assert_eq!(mesh.vertices.len(), 48);
        assert_eq!(mesh.indices.len(), 216);
    }

    #[test]
    fn mesh_rail_with_ties() {
        let mut chunk = ChunkStorage::new_empty();
        chunk.set_block(0, 0, 0, crate::block::BlockId::HULL_STANDARD);
        chunk.add_sub_block(0, 0, 0, SubBlockElement {
            face: 2, // +Y face (floor)
            element_type: SubBlockType::Rail,
            rotation: 0,
            flags: 0,
        });
        let mesh = mesh_sub_blocks(&chunk);
        // Rail = 2 parallel rails + 4 cross-ties = 6 boxes = 48 vertices, 216 indices.
        assert_eq!(mesh.vertices.len(), 48);
        assert_eq!(mesh.indices.len(), 216);
    }
}
