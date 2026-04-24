//! Client-side block targeting raycast.
//!
//! Produces a per-frame `BlockTarget` resource describing which block
//! the camera is currently aimed at — used by:
//! * The **highlight renderer** (visible cube around the targeted
//!   block, so the user sees what they'll break/place).
//! * The **mouse-button dispatch** in `block_interaction` (LMB=BREAK,
//!   RMB=PLACE, MMB=pick).
//!
//! The server always re-raycasts on `BlockEditRequest` — this client
//! raycast exists **only for visual feedback**. Authority for what
//! block actually gets broken or placed lives with the server, which
//! uses the `eye` + `look` vectors we send alongside.
//!
//! **Multi-grid support.** On ships, a single block world position may
//! belong to the root grid or to a mechanical sub-grid (rotor, hinge,
//! piston). Each sub-grid has its own local frame driven by its joint
//! transform, so raycasting against a sub-grid requires transforming
//! the ray into the sub-grid's local space first, running DDA there,
//! then comparing the hit distance against the root-grid hit. This
//! module implements the root-grid path today; sub-grid raycast is
//! gated behind `#23 Sub-grid mechanical rendering` — the hooks are in
//! place so it drops in cleanly once sub-grid transforms flow.
//!
//! **Server contract** (ship-shard/main.rs:3636-3680):
//! * `eye` in shard-local coords
//! * `look` normalized, in shard-local coords
//! * Server's max range = `BLOCK_EDIT_RANGE` (8m)
//! * Server rejects edits where `|eye − player_pos| > 3m` (anti-cheat)
//! We match those semantics so a client-highlighted block is the same
//! block the server will resolve on edit.

use bevy::prelude::*;
use voxeldust_core::block::{
    chunk_storage::ChunkStorage, raycast as core_raycast, registry::BlockRegistry,
    ship_grid::ShipGrid,
};

/// `voxeldust_core` uses a distinct glam version from Bevy's re-export,
/// so `glam::IVec3` and `bevy::prelude::IVec3` are nominally different
/// types even though they're structurally identical. Convert on the
/// boundary via this helper to keep call sites clean.
fn to_bevy_ivec3(v: glam::IVec3) -> IVec3 {
    IVec3::new(v.x, v.y, v.z)
}

use crate::camera_frame::CameraFrameSet;
use crate::chunk_stream::{ChunkStorageCache, SourceId};
use crate::input_system::{InputSet, EYE_HEIGHT};
use crate::net_plugin::NetConnection;
use crate::player_sync::PlayerState;
use crate::shard_transition::ShardContext;
use voxeldust_core::client_message::shard_type as st;

/// Maximum ray distance for block targeting. Matches legacy
/// `BLOCK_EDIT_RANGE` on the server (ship-shard/main.rs). Exceeding
/// this on the client would produce highlights for blocks the server
/// would reject, breaking feedback parity.
pub const BLOCK_EDIT_RANGE: f32 = 8.0;

/// System set for the raycast pass. Consumers (`block_interaction`,
/// highlight renderer) run `.after(BlockRaycastSet)` so they observe a
/// `BlockTarget` resource that's already been updated for this frame's
/// camera pose.
#[derive(SystemSet, Clone, Eq, PartialEq, Hash, Debug)]
pub struct BlockRaycastSet;

pub struct BlockRaycastPlugin;

impl Plugin for BlockRaycastPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<BlockTarget>().add_systems(
            Update,
            update_block_target
                .in_set(BlockRaycastSet)
                .after(InputSet)
                .after(CameraFrameSet),
        );
    }
}

/// Result of the per-frame block raycast. `None` inner when no block
/// is within range; otherwise carries the hit info for the block the
/// crosshair is currently pointed at.
#[derive(Resource, Default, Debug, Clone)]
pub struct BlockTarget {
    pub hit: Option<BlockTargetHit>,
    /// Eye position from this frame's raycast, in shard-local coords.
    /// Cached so `block_interaction` can send the exact same eye to
    /// the server — guaranteeing server + client agree on the hit.
    pub eye: glam::Vec3,
    /// Look direction (normalized) from this frame's raycast.
    pub look: glam::Vec3,
}

#[derive(Debug, Clone, Copy)]
pub struct BlockTargetHit {
    /// Which chunk source this block belongs to. For the foreseeable
    /// future this is always `ShardContext.primary_seed` — sub-grid
    /// secondary-source raycasts land in #23.
    pub source_seed: SourceId,
    /// World-space integer coordinates of the hit block.
    pub block_pos: IVec3,
    /// Unit vector normal to the face the ray entered through
    /// (axis-aligned: ±X/±Y/±Z). Used for placement: new block goes
    /// at `block_pos + face_normal`.
    pub face_normal: IVec3,
    /// Straight-line distance from `eye` to the hit face.
    pub distance: f32,
    /// Block type at the hit position. Used by MMB pick-block to fill
    /// the hotbar / selected block.
    pub block_type: voxeldust_core::block::BlockId,
    /// Sub-grid id the block belongs to, or `None` for root-grid blocks.
    /// Reserved for #23 — currently always `None`.
    pub sub_grid_id: Option<u32>,
}

/// Per-frame raycast. Runs only when we have a connection and are on a
/// SHIP primary (the only shard type where block-targeting is
/// meaningful today; PLANET surface editing lands later). Skipped
/// while the cursor is ungrabbed (user is in a menu) to avoid
/// targeting wandering mouse positions.
fn update_block_target(
    mut target: ResMut<BlockTarget>,
    ctx: Res<ShardContext>,
    conn: Res<NetConnection>,
    state: Res<PlayerState>,
    frame: Res<crate::camera_frame::CameraFrame>,
    storage: Res<ChunkStorageCache>,
    registry: Res<crate::chunk_stream::SharedBlockRegistry>,
    cursors: Query<&bevy::window::CursorOptions, With<bevy::window::PrimaryWindow>>,
) {
    // Cheap bail-outs in priority order. The `BlockTarget` is cleared
    // so stale highlights don't linger into menus / transitions.
    let grabbed = cursors
        .single()
        .map(|c| c.grab_mode != bevy::window::CursorGrabMode::None)
        .unwrap_or(false);
    if !conn.connected || !grabbed || state.last_tick == 0 || ctx.current != st::SHIP {
        target.hit = None;
        return;
    }
    let Some(seed) = ctx.primary_seed else {
        target.hit = None;
        return;
    };

    // Eye + look in **shard-local** coordinates. The server's raycast
    // runs in the same frame (ship-local on SHIP shard), so sending
    // these values straight through gives a pixel-accurate match on
    // `BlockEditRequest`.
    //
    // `state.world_pos` is the player's position in the current shard
    // frame (already in Bevy world frame since the camera-frame
    // pipeline has shifted it appropriately). The eye sits at
    // EYE_HEIGHT along the camera's up vector — same formula
    // `apply_server_pose` uses for the camera transform.
    let eye = state.world_pos + frame.up * EYE_HEIGHT;
    let look = (frame.rotation * Vec3::NEG_Z).normalize_or_zero();
    if look.length_squared() < 1e-6 {
        target.hit = None;
        return;
    }
    target.eye = glam::Vec3::new(eye.x, eye.y, eye.z);
    target.look = glam::Vec3::new(look.x, look.y, look.z);

    // Root-grid raycast. Sub-grid support is deferred to #23 — the
    // renderer has no sub-grid root transforms flowing yet, so there
    // is nothing to inverse-transform against. When #23 lands, this
    // block grows: raycast against root, then each active sub-grid
    // with ray transformed into its local frame, keep closest hit.
    let registry_ref = &registry.0;
    // `voxeldust_core`'s raycast takes glam::Vec3 — rebuild from our
    // Bevy Vec3 via component extraction (the two crates disagree on
    // the nominal type even though they're structurally identical).
    let core_eye = glam::Vec3::new(target.eye.x, target.eye.y, target.eye.z);
    let core_look = glam::Vec3::new(target.look.x, target.look.y, target.look.z);
    let core_hit = core_raycast::raycast(core_eye, core_look, BLOCK_EDIT_RANGE, |x, y, z| {
        let (chunk_key, lx, ly, lz) = ShipGrid::world_to_chunk(x, y, z);
        storage
            .get(seed, to_bevy_ivec3(chunk_key))
            .map(|c: &ChunkStorage| registry_ref.is_solid(c.get_block(lx, ly, lz)))
            .unwrap_or(false)
    });

    target.hit = core_hit.map(|h| {
        // Re-read the block type at the hit position for MMB pick.
        let (chunk_key, lx, ly, lz) =
            ShipGrid::world_to_chunk(h.world_pos.x, h.world_pos.y, h.world_pos.z);
        let block_type = storage
            .get(seed, to_bevy_ivec3(chunk_key))
            .map(|c| c.get_block(lx, ly, lz))
            .unwrap_or_else(|| voxeldust_core::block::BlockId::from_u16(0));
        BlockTargetHit {
            source_seed: seed,
            block_pos: to_bevy_ivec3(h.world_pos),
            face_normal: to_bevy_ivec3(h.face_normal),
            distance: h.distance,
            block_type,
            sub_grid_id: None,
        }
    });
}
