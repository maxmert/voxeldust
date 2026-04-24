//! Cross-shard client-side raycast. HUD-only — server re-raycasts all
//! block edits. Iterates every loaded `ChunkSource` (primary AND every
//! secondary), transforming the camera ray into each shard's local
//! frame, DDA against `ChunkStorageCache`, keeping the closest hit.

use bevy::prelude::*;
use bevy::window::{CursorGrabMode, CursorOptions, PrimaryWindow};
use glam::{DQuat, DVec3};

use voxeldust_core::block::{
    block_id::BlockId, chunk_storage::ChunkStorage, raycast as core_raycast,
    ship_grid::ShipGrid,
};

use crate::chunk::stream::SharedBlockRegistry;
use crate::chunk::ChunkStorageCache;
use crate::input::InputSet;
use crate::net::NetConnection;
use crate::shard::{CameraWorldPos, ChunkSource, ShardKey, ShardOrigin};
use crate::MainCamera;

/// Matches the server's `BLOCK_EDIT_RANGE` in ship-shard main.rs.
pub const BLOCK_EDIT_RANGE: f32 = 8.0;

#[derive(SystemSet, Clone, Eq, PartialEq, Hash, Debug)]
pub struct BlockRaycastSet;

pub struct BlockRaycastPlugin;

impl Plugin for BlockRaycastPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<BlockTarget>().add_systems(
            Update,
            update_block_target
                .in_set(BlockRaycastSet)
                .after(InputSet),
        );
    }
}

/// Current block under the crosshair. `None` → no hit within range.
/// `eye` / `look` cached in the HIT's shard-local frame so Phase 15
/// dispatch can send the exact same values to the server, guaranteeing
/// server + client agree on the hit.
#[derive(Resource, Default, Debug, Clone)]
pub struct BlockTarget {
    pub hit: Option<BlockTargetHit>,
}

#[derive(Debug, Clone, Copy)]
pub struct BlockTargetHit {
    /// Which shard owns the block. Phase 15 routes `BlockEditRequest`
    /// to this shard's TCP.
    pub shard: ShardKey,
    /// The `ChunkSource` root entity the hit belongs to. The highlight
    /// uses `ChildOf(source)` so it follows the shard's transform.
    pub source_entity: Entity,
    /// Block position in the owning shard's local frame.
    pub block_pos: IVec3,
    /// Face normal of the hit (axis-aligned, shard-local). Placement
    /// target = `block_pos + face_normal`.
    pub face_normal: IVec3,
    pub distance: f32,
    pub block_type: BlockId,
    /// Eye position in the OWNING shard's local frame. Sent verbatim
    /// to the server so both sides raycast against the same geometry.
    pub eye_local: glam::Vec3,
    /// Look direction (normalized) in the owning shard's local frame.
    pub look_local: glam::Vec3,
    /// Sub-grid id when the hit belongs to a mechanical sub-grid
    /// (Phase 16). `None` for root-grid hits.
    pub sub_grid_id: Option<u32>,
}

fn update_block_target(
    mut target: ResMut<BlockTarget>,
    conn: Res<NetConnection>,
    cursors: Query<&CursorOptions, With<PrimaryWindow>>,
    camera_q: Query<&GlobalTransform, With<MainCamera>>,
    camera_world: Res<CameraWorldPos>,
    sources: Query<(Entity, &ChunkSource, &ShardOrigin)>,
    storage: Res<ChunkStorageCache>,
    registry: Res<SharedBlockRegistry>,
) {
    // Cheap gates.
    let grabbed = cursors
        .single()
        .map(|c| c.grab_mode != CursorGrabMode::None)
        .unwrap_or(false);
    if !conn.connected || !grabbed {
        target.hit = None;
        return;
    }
    let Ok(cam_global) = camera_q.single() else {
        target.hit = None;
        return;
    };

    // `camera_world` already includes the eye-height offset (baked in
    // by `player_sync::apply_worldstate_pose`), so use it directly as
    // the ray origin. Both camera rendering and raycast read the same
    // world-space eye — no mismatch possible.
    let cam_rot = cam_global.rotation();
    let fwd_world = cam_rot * Vec3::NEG_Z;
    if fwd_world.length_squared() < 1e-6 {
        target.hit = None;
        return;
    }
    let eye_world = camera_world.pos;
    let look_world = DVec3::new(
        fwd_world.x as f64,
        fwd_world.y as f64,
        fwd_world.z as f64,
    );

    // Iterate every loaded shard; keep the closest hit.
    let mut best: Option<BlockTargetHit> = None;
    for (entity, source, origin) in sources.iter() {
        let Some(hit) = raycast_shard(
            entity,
            source.key,
            origin,
            eye_world,
            look_world,
            &storage,
            &registry.0,
        ) else {
            continue;
        };
        match best {
            Some(ref cur) if hit.distance >= cur.distance => {}
            _ => best = Some(hit),
        }
    }
    target.hit = best;
}

/// Raycast into a single shard. Transforms `eye_world` + `look_world`
/// into the shard's local frame via its `ShardOrigin`, runs the core
/// DDA, converts the hit back into shard-local coordinates and returns.
fn raycast_shard(
    source_entity: Entity,
    key: ShardKey,
    origin: &ShardOrigin,
    eye_world: DVec3,
    look_world: DVec3,
    storage: &ChunkStorageCache,
    registry: &voxeldust_core::block::registry::BlockRegistry,
) -> Option<BlockTargetHit> {
    // World → shard-local: eye_local = rot^-1 * (eye_world - origin).
    let inv_rot: DQuat = origin.rotation.inverse();
    let eye_local_d = inv_rot * (eye_world - origin.origin);
    let look_local_d = inv_rot * look_world;

    // Core raycast takes f32 Vec3.
    let eye_local = glam::Vec3::new(
        eye_local_d.x as f32,
        eye_local_d.y as f32,
        eye_local_d.z as f32,
    );
    let look_local = glam::Vec3::new(
        look_local_d.x as f32,
        look_local_d.y as f32,
        look_local_d.z as f32,
    );
    if look_local.length_squared() < 1e-6 {
        return None;
    }

    let hit = core_raycast::raycast(eye_local, look_local, BLOCK_EDIT_RANGE, |x, y, z| {
        let (chunk_key, lx, ly, lz) = ShipGrid::world_to_chunk(x, y, z);
        storage
            .get(key, to_bevy_ivec3(chunk_key))
            .map(|c: &ChunkStorage| registry.is_solid(c.get_block(lx, ly, lz)))
            .unwrap_or(false)
    })?;

    // Re-read the block type at the hit for MMB pick / interaction
    // schema lookup.
    let (chunk_key, lx, ly, lz) = ShipGrid::world_to_chunk(
        hit.world_pos.x,
        hit.world_pos.y,
        hit.world_pos.z,
    );
    let block_type = storage
        .get(key, to_bevy_ivec3(chunk_key))
        .map(|c| c.get_block(lx, ly, lz))
        .unwrap_or_else(|| BlockId::from_u16(0));

    Some(BlockTargetHit {
        shard: key,
        source_entity,
        block_pos: to_bevy_ivec3(hit.world_pos),
        face_normal: to_bevy_ivec3(hit.face_normal),
        distance: hit.distance,
        block_type,
        eye_local,
        look_local,
        sub_grid_id: None,
    })
}

fn to_bevy_ivec3(v: glam::IVec3) -> IVec3 {
    IVec3::new(v.x, v.y, v.z)
}
