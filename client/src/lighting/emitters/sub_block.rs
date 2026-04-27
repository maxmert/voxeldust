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
use voxeldust_core::block::palette::{index_to_xyz, CHUNK_SIZE};
use voxeldust_core::block::registry::BlockRegistry;
use voxeldust_core::block::sub_block::face_to_offset;

use super::{
    blackbody_with_tint, spawn_emissive_proxy, spawn_one_light, LampConfigs,
    LocalShadowBudget, BLOCK_CENTRE_OFFSET,
};
use crate::shard::ShardKey;

/// Sub-block lamp face-outset — the light position sits this many
/// metres outside the host block's face along the face normal so the
/// PointLight isn't buried inside the host's solid voxel.
const SUB_BLOCK_LAMP_FACE_OUTSET: f32 = 0.51;

/// Iterate the chunk's sub-block elements and spawn a light + emissive
/// proxy for any element whose `SubBlockType` resolves to a
/// `LightSpec` via `BlockRegistry::sub_block_light_spec`. Returns
/// `(spawned, shadow_casters)` so the caller can aggregate diagnostic
/// counts across the full-block + sub-block paths.
///
/// When `lamp_configs` carries a player-customised entry for the
/// lamp's `(shard, world_pos, face)`, the broadcast values **override**
/// the default `LightSpec`:
///
///   * `subscribe_channel` — non-empty channels feed through the
///     existing `ThrottleModulated*` path so the lamp's intensity
///     scales with the broadcast signal value (0 = off, 1 = full).
///   * `color_kelvin` + `tint_linear_rgb` — replace the spec's blackbody
///     temperature and tint with the player's pick.
///   * `intensity_scale` — multiplies the spec's `lumens` so the
///     player can dim a lamp without unbinding the channel.
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
    lamp_configs: &LampConfigs,
    shard: Option<ShardKey>,
    chunk_index: bevy::math::IVec3,
) -> (u32, u32) {
    let mut spawned: u32 = 0;
    let mut shadow_casters: u32 = 0;
    let chunk_origin = chunk_index * (CHUNK_SIZE as i32);
    for (flat_idx, elements) in chunk.iter_sub_blocks() {
        let (bx, by, bz) = index_to_xyz(flat_idx as usize);
        for elem in elements {
            let Some(default_spec) =
                registry.sub_block_light_spec(elem.element_type)
            else {
                continue;
            };
            // Look up player-customised settings for this lamp, if the
            // shard is known and the broadcast carried an override.
            // `LampConfigs` is keyed by `glam::IVec3` (workspace 0.29);
            // the per-frame Bevy IVec3 is incompatible, so build the
            // lookup key directly in `glam`.
            let world_pos = glam::IVec3::new(
                chunk_origin.x + bx as i32,
                chunk_origin.y + by as i32,
                chunk_origin.z + bz as i32,
            );
            let custom = shard
                .and_then(|s| lamp_configs.get(s, world_pos, elem.face));
            let mut spec = default_spec;
            if let Some(c) = custom {
                spec.color_kelvin = c.color_kelvin;
                spec.tint_linear_rgb = c.tint_linear_rgb;
                spec.lumens = default_spec.lumens * c.intensity_scale;
                spec.emissive_radiance_w_per_m2 =
                    default_spec.emissive_radiance_w_per_m2 * c.intensity_scale;
            }
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
            // Throttle channel: if the player bound a non-empty
            // `subscribe_channel`, route the lamp's intensity through
            // it via the existing `ThrottleModulated*` path. The
            // `&'static str` path used by full-block thrusters needs
            // a runtime String here — we convert by leaking, which is
            // safe because the count is bounded by the number of
            // player-bound lamps × ship lifetime (microscopic).
            let throttle_channel: Option<&'static str> = custom
                .map(|c| c.subscribe_channel.as_str())
                .filter(|s| !s.is_empty())
                .map(|s| Box::leak(s.to_string().into_boxed_str()) as &'static str);
            spawn_one_light(
                commands,
                chunk_entity,
                light_position,
                &spec,
                color,
                shadows_enabled,
                throttle_channel,
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
                throttle_channel,
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
