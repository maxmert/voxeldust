//! Phase 11 — Player sync: authoritative camera pose, no client
//! prediction, no interpolation. Writes `CameraWorldPos` from
//! `WorldState.entities[].is_own` (or the spawn pose latch) and sets
//! the camera entity's Transform directly.
//!
//! MVP: identity rotation on the camera. Per-shard-type camera-frame
//! math (ship forward-lock, tangent basis, body-relative EVA, galaxy
//! skybox) lands as subsequent increments via a trait method on
//! `ShardTypePlugin` (not in Phase 3's initial trait). Moving the
//! camera position off zero is enough to make the floating-origin
//! system's rebase visible in the HUD + logs.

use bevy::prelude::*;
use glam::{DQuat, DVec3};

use voxeldust_core::client_message::EntityKind;

use crate::input::{rotation_from_look, LocalLook, EYE_HEIGHT};
use crate::net::NetConnection;
use crate::shard::{
    CameraWorldPos, PrimaryShard, PrimaryWorldState, ShardOriginSet, SpawnPoseLatch,
    WorldStateIngestSet,
};
use crate::MainCamera;

#[derive(SystemSet, Clone, Debug, Hash, Eq, PartialEq)]
pub struct PlayerSyncSet;

pub struct PlayerSyncPlugin;

impl Plugin for PlayerSyncPlugin {
    fn build(&self, app: &mut App) {
        // Run AFTER WorldStateIngestSet (so we see this frame's
        // latest primary WorldState) and BEFORE ShardOriginSet (so
        // the rebase sees the updated CameraWorldPos for the current
        // frame).
        app.configure_sets(
            Update,
            PlayerSyncSet
                .after(WorldStateIngestSet)
                .before(ShardOriginSet),
        )
            .add_systems(
                Update,
                (
                    consume_spawn_pose_latch,
                    apply_worldstate_pose,
                    update_camera_rotation,
                )
                    .chain()
                    .in_set(PlayerSyncSet),
            );
    }
}

/// If a spawn pose is latched from a transition, apply it immediately.
/// Authoritative, one-shot — subsequent WorldState applies layer on top.
fn consume_spawn_pose_latch(
    mut latch: ResMut<SpawnPoseLatch>,
    mut camera_world: ResMut<CameraWorldPos>,
    mut cam_q: Query<&mut Transform, With<MainCamera>>,
) {
    let Some(pose) = latch.pose.take() else { return };
    latch.target_shard_type = None;
    camera_world.pos = DVec3::new(
        pose.position.x,
        pose.position.y,
        pose.position.z,
    );
    if let Ok(mut tf) = cam_q.single_mut() {
        // MVP: write rotation directly from the handoff pose.
        // Per-shard-type camera-frame math will layer on top once the
        // trait method lands.
        tf.rotation = dquat_to_bevy(DQuat::from_xyzw(
            pose.rotation.x,
            pose.rotation.y,
            pose.rotation.z,
            pose.rotation.w,
        ));
    }
    tracing::info!(
        cam_world = ?(camera_world.pos.x, camera_world.pos.y, camera_world.pos.z),
        "spawn pose latch consumed",
    );
}

/// Per-tick authoritative camera position. Server authority, no
/// client prediction — values applied directly from this frame's
/// `PrimaryWorldState`.
///
/// Source of truth (ordered by preference):
/// 1. `entities[]` entry with `is_own=true && kind.is_player()` —
///    the modern ObservableEntity path (PLANET / SYSTEM primaries).
/// 2. `players[]` entry with `player_id == conn.player_id` — the
///    legacy snapshot path. On SHIP primary the server populates this
///    with the player's ship-local position; the own-ship rotation
///    is read from the SHIP entity in `entities[]` (the ship always
///    has `is_own=true` when it's the player's ship).
///
/// Camera world-pos composition:
/// - `ws.origin` is the primary's system-space anchor (ship position
///   on SHIP shards, planet center on PLANET, star for SYSTEM).
/// - SHIP primary: world-pos = ws.origin + ship_rotation * player_local.
/// - Other primaries: world-pos = ws.origin + player_local (identity).
fn apply_worldstate_pose(
    ws: Res<PrimaryWorldState>,
    primary: Res<PrimaryShard>,
    conn: Res<NetConnection>,
    mut camera_world: ResMut<CameraWorldPos>,
) {
    let Some(ws) = ws.latest.as_ref() else { return };
    if primary.current.is_none() {
        return;
    }

    // Eye offset in shard-local frame — server's capsule uses Y-up,
    // so the eye sits `EYE_HEIGHT` above the player's capsule center
    // along local +Y. When composed with the ship's world rotation on
    // SHIP primaries the eye ends up at the correct world-space height
    // (and raycast reads `camera_world` as the eye without adding its
    // own offset, so both systems agree).
    let eye_offset_local = DVec3::new(0.0, EYE_HEIGHT as f64, 0.0);

    // Prefer the entities[] path when the server populates it.
    if let Some(own) = ws
        .entities
        .iter()
        .find(|e| e.is_own && e.kind.is_player())
    {
        let abs = DVec3::new(ws.origin.x, ws.origin.y, ws.origin.z)
            + DVec3::new(own.position.x, own.position.y, own.position.z)
            + eye_offset_local;
        camera_world.pos = abs;
        return;
    }

    // Fallback: legacy players[] snapshot. On SHIP primary the player
    // is reported in ship-local coords; compose with the ship's rotation
    // from entities[] to produce world-pos.
    let Some(own_player) = ws
        .players
        .iter()
        .find(|p| p.player_id == conn.player_id)
    else {
        return;
    };

    let own_ship = ws
        .entities
        .iter()
        .find(|e| e.is_own && e.kind == EntityKind::Ship);
    let ship_rot = own_ship
        .map(|s| DQuat::from_xyzw(s.rotation.x, s.rotation.y, s.rotation.z, s.rotation.w))
        .unwrap_or(DQuat::IDENTITY);

    let player_local = DVec3::new(
        own_player.position.x,
        own_player.position.y,
        own_player.position.z,
    );
    let origin = DVec3::new(ws.origin.x, ws.origin.y, ws.origin.z);
    // Eye offset is in SHIP-LOCAL frame (on SHIP primary) or world
    // Y-up (on SYSTEM/PLANET/GALAXY primary). When there is no own
    // ship (EVA), `ship_rot` is identity so eye_offset_local is
    // applied as-is along world +Y.
    camera_world.pos = origin + ship_rot * (player_local + eye_offset_local);
}

fn dquat_to_bevy(q: DQuat) -> Quat {
    Quat::from_xyzw(q.x as f32, q.y as f32, q.z as f32, q.w as f32)
}

/// Camera rotation — per-shard-type camera frame math lands as a
/// future `ShardTypePlugin::primary_camera_frame` trait method. Until
/// then this MVP composes:
///
///   SEATED  + !Alt  → `ship_rot` (forward-locked; mouse drives ship).
///   SEATED  + Alt   → `ship_rot * rotation_from_look(look.yaw, look.pitch)`
///                     (free-look: mouse pans view without moving ship).
///   WALKING         → `ship_rot * rotation_from_look(look.server_yaw,
///                     look.server_pitch)` — the player's avatar-head
///                     orientation relative to the ship frame.
///
/// `ship_rot` comes from the own-ship `ObservableEntity` on SHIP
/// primaries; on SYSTEM / PLANET / GALAXY primaries there's no own-ship
/// so it falls back to identity (the full per-shard-type camera is a
/// follow-up item).
fn update_camera_rotation(
    ws: Res<PrimaryWorldState>,
    look: Res<LocalLook>,
    seated: Res<crate::input::SeatedState>,
    mut cam_q: Query<&mut Transform, With<MainCamera>>,
) {
    let Ok(mut tf) = cam_q.single_mut() else { return };
    let ship_rot = ws
        .latest
        .as_ref()
        .and_then(|ws| {
            ws.entities
                .iter()
                .find(|e| e.is_own && e.kind == EntityKind::Ship)
        })
        .map(|s| {
            Quat::from_xyzw(
                s.rotation.x as f32,
                s.rotation.y as f32,
                s.rotation.z as f32,
                s.rotation.w as f32,
            )
        })
        .unwrap_or(Quat::IDENTITY);

    let local = if seated.is_seated && !look.free_look {
        // Forward-locked: camera looks along ship forward. Mouse is
        // being consumed by the seat evaluator to produce torque
        // signals — it must NOT also pan the view.
        Quat::IDENTITY
    } else {
        rotation_from_look(look.server_yaw, look.server_pitch)
    };
    tf.rotation = ship_rot * local;
}

// Future trait method impls will use this helper.
#[allow(dead_code)]
fn kind_is_own_player(k: EntityKind) -> bool {
    k.is_player() || k == EntityKind::Ship
}
