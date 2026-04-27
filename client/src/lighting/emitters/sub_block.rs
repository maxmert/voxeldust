//! Sub-block emitters — flush-mounted lamps via
//! `SubBlockType::SurfaceLight`, `RedSurfaceLight`, `BlueSurfaceLight`,
//! and `Floodlight`.
//!
//! Each emitter sits on one face of a host block. The PointLight
//! position is offset along the face normal (so the light isn't buried
//! inside the host's solid voxel and can illuminate the cabin); the
//! emissive proxy sits at the host's centre and gets its own outset
//! inside `spawn_emissive_proxy` to land flush on the host face.

use bevy::prelude::*;

use voxeldust_core::block::block_meta::BlockOrientation;
use voxeldust_core::block::chunk_storage::ChunkStorage;
use voxeldust_core::block::palette::index_to_xyz;
use voxeldust_core::block::registry::BlockRegistry;
use voxeldust_core::block::sub_block::face_to_offset;

use super::{
    blackbody_with_tint, spawn_emissive_proxy, spawn_one_light, LocalShadowBudget,
    BLOCK_CENTRE_OFFSET,
};

/// Sub-block lamp face-outset — the light position sits this many
/// metres outside the host block's face along the face normal so the
/// PointLight isn't buried inside the host's solid voxel.
const SUB_BLOCK_LAMP_FACE_OUTSET: f32 = 0.51;

/// Iterate the chunk's sub-block elements and spawn a light + emissive
/// proxy for any element whose `SubBlockType` resolves to a
/// `LightSpec` via `BlockRegistry::sub_block_light_spec`. Returns
/// `(spawned, shadow_casters)` so the caller can aggregate diagnostic
/// counts across the full-block + sub-block paths.
#[allow(clippy::too_many_arguments)]
pub(super) fn spawn_sub_block_emitters(
    commands: &mut Commands,
    materials: &mut Assets<StandardMaterial>,
    chunk_entity: Entity,
    chunk: &ChunkStorage,
    registry: &BlockRegistry,
    shadow_budget: &mut LocalShadowBudget,
    cube_mesh: &Handle<Mesh>,
    face_mesh: &Handle<Mesh>,
) -> (u32, u32) {
    let mut spawned: u32 = 0;
    let mut shadow_casters: u32 = 0;
    for (flat_idx, elements) in chunk.iter_sub_blocks() {
        let (bx, by, bz) = index_to_xyz(flat_idx as usize);
        for elem in elements {
            let Some(spec) = registry.sub_block_light_spec(elem.element_type) else {
                continue;
            };
            let face_normal = face_normal_vec3(elem.face);
            let block_centre = Vec3::new(
                bx as f32 + BLOCK_CENTRE_OFFSET,
                by as f32 + BLOCK_CENTRE_OFFSET,
                bz as f32 + BLOCK_CENTRE_OFFSET,
            );
            // Two distinct positions:
            //   * `light_position` — just outside the host face along
            //     the face normal so the PointLight isn't buried inside
            //     the host's solid voxel.
            //   * Proxy is spawned at `block_centre`; the
            //     `spawn_emissive_proxy` helper adds its own face
            //     outset internally for `DirectionalFace` mode, so the
            //     proxy plane lands flush on the host face (no double
            //     offset).
            let light_position = block_centre + face_normal * SUB_BLOCK_LAMP_FACE_OUTSET;
            let color = blackbody_with_tint(spec.color_kelvin, spec.tint_linear_rgb);
            let shadows_enabled =
                spec.shadow_caster_class > 0 && shadow_budget.try_consume();
            // Promote the sub-block face index into a virtual
            // `BlockOrientation` so the proxy emits along the face
            // normal — `BlockOrientation::new(facing, rot)` accepts
            // the same 0–5 index as `face_to_offset`.
            let virtual_orientation = BlockOrientation::new(elem.face, 0);
            spawn_one_light(
                commands,
                chunk_entity,
                light_position,
                &spec,
                color,
                shadows_enabled,
                None, // sub-block lamps get their throttle channel from
                      // the per-element config in Slice C; until then
                      // they run at static intensity.
            );
            spawn_emissive_proxy(
                commands,
                materials,
                chunk_entity,
                block_centre,
                &spec,
                color,
                virtual_orientation,
                cube_mesh.clone(),
                face_mesh.clone(),
                None,
            );
            spawned += 1;
            if shadows_enabled {
                shadow_casters += 1;
            }
        }
    }
    (spawned, shadow_casters)
}

/// Convert a `SubBlockElement.face` index (0..6: ±X, ±Y, ±Z) to a
/// `Vec3` unit normal — same axis convention as
/// `core::block::sub_block::face_to_offset`, just promoted to the
/// `Vec3` Bevy uses for transforms.
fn face_normal_vec3(face: u8) -> Vec3 {
    let off = face_to_offset(face);
    Vec3::new(off.x as f32, off.y as f32, off.z as f32)
}
