//! Full-block emitters — one emitter occupies an entire 1 m voxel.
//!
//! Today's full-block emitters:
//!
//!   * **Thrusters** (`FunctionalBlockKind::Thruster`) — directional
//!     plume on the exhaust face; intensity modulated by the throttle
//!     channel implied by the block's facing direction (linear
//!     thrusters only — RCS torque thrusters fall back to the matching
//!     linear channel).
//!   * **Reactors** (`FunctionalBlockKind::Reactor`) — full-cube core
//!     glow; static intensity (no signal modulation today).
//!
//! Adding a new full-block emitter type is two steps: add a
//! `LightSpec` entry in `core::block::registry`, and (optionally) add a
//! match arm here for any block-kind-specific channel resolution.

use bevy::prelude::*;

use voxeldust_core::block::block_meta::BlockOrientation;
use voxeldust_core::block::chunk_storage::ChunkStorage;
use voxeldust_core::block::registry::{BlockRegistry, FunctionalBlockKind};

use super::{
    blackbody_with_tint, spawn_emissive_proxy, spawn_one_light, LocalShadowBudget,
    BLOCK_CENTRE_OFFSET, CHUNK_SIZE_U8,
};

/// Iterate the chunk's 62³ voxel cells and spawn a light + emissive
/// proxy for any block whose registry entry has a `LightSpec`. Returns
/// `(spawned, shadow_casters)` so the caller can aggregate diagnostic
/// counts across the full-block + sub-block paths.
#[allow(clippy::too_many_arguments)]
pub(super) fn spawn_full_block_emitters(
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
    // Voxel iteration: 62³ = 238 k cells per chunk. The body is a
    // single registry lookup + branch; in benchmark this clears in
    // <1 ms even with no LightSpec hits, so we just walk the whole chunk
    // every remesh.
    for z in 0..CHUNK_SIZE_U8 {
        for y in 0..CHUNK_SIZE_U8 {
            for x in 0..CHUNK_SIZE_U8 {
                let id = chunk.get_block(x, y, z);
                let Some(spec) = registry.light_spec(id) else { continue };
                let local = Vec3::new(
                    x as f32 + BLOCK_CENTRE_OFFSET,
                    y as f32 + BLOCK_CENTRE_OFFSET,
                    z as f32 + BLOCK_CENTRE_OFFSET,
                );
                let color = blackbody_with_tint(spec.color_kelvin, spec.tint_linear_rgb);
                let shadows_enabled =
                    spec.shadow_caster_class > 0 && shadow_budget.try_consume();
                let orientation = chunk
                    .get_meta(x, y, z)
                    .map(|m| m.orientation)
                    .unwrap_or(BlockOrientation::DEFAULT);
                // Per-block-kind channel resolution. Today only
                // thrusters dim with their throttle input.
                let throttle_channel = match registry.functional_kind(id) {
                    Some(FunctionalBlockKind::Thruster) => {
                        Some(thruster_channel_for_facing(orientation.facing()))
                    }
                    _ => None,
                };
                spawn_one_light(
                    commands,
                    chunk_entity,
                    local,
                    &spec,
                    color,
                    shadows_enabled,
                    throttle_channel,
                );
                spawn_emissive_proxy(
                    commands,
                    materials,
                    chunk_entity,
                    local,
                    &spec,
                    color,
                    orientation,
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
    }
    (spawned, shadow_casters)
}

/// Map a thruster's exhaust direction (`BlockOrientation.facing()`) to
/// the canonical input throttle channel the starter ship uses. Newton's
/// 3rd law: a thruster with exhaust along +X pushes the ship in -X (the
/// "thrust-left" direction in ship frame), so the channel reads the
/// motion-direction the player wants, not the exhaust direction.
///
/// Slice C will replace this heuristic with a per-block broadcast of
/// the actual `channel_override` set on the block grid — at which point
/// RCS torque thrusters and any future custom-channel thruster also
/// dim correctly.
fn thruster_channel_for_facing(facing: u8) -> &'static str {
    match facing {
        0 => "thrust-left",     // exhaust +X → ship goes -X
        1 => "thrust-right",    // exhaust -X → ship goes +X
        2 => "thrust-down",     // exhaust +Y → ship goes -Y
        3 => "thrust-up",       // exhaust -Y → ship goes +Y
        4 => "thrust-forward",  // exhaust +Z → ship goes -Z (per starter-ship convention)
        _ => "thrust-reverse",  // exhaust -Z → ship goes +Z
    }
}
