//! Phase 11 — Player sync: authoritative camera pose, no client
//! prediction of position. Writes `CameraWorldPos` from
//! `WorldState.entities[].is_own` (or the spawn pose latch).
//!
//! Camera **rotation** is composed every frame from
//! `ship_rot × rotation_from_look(LocalLook)` so mouse input feels
//! 60 Hz-smooth. Reading the server's pre-composed rotation directly
//! would clock the camera at 20 Hz and any off-center geometry would
//! visibly step every WS tick — a perceived position "blink" while
//! walking past walls. The server still composes the same rotation for
//! handoff / spawn-pose purposes; on transitions, the spawn pose's
//! rotation is decomposed back into `LocalLook` so the next-frame
//! composition reproduces it pixel-for-pixel. The two paths converge
//! because both share `ship_rot` and `LocalLook` (the client owns
//! `LocalLook` and ships it back to the server every tick).

use std::time::Instant;

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

/// Velocity-based extrapolation of the spawn-pose camera position
/// during the gap between latch consume and the new primary's first
/// `WorldState` arrival (typically 1–2 s on inter-shard transitions
/// while the new TCP/UDP path warms up).
///
/// Without this, when the camera is latched to the spawn position and
/// the EVA player on the server is co-moving with a planet at ~110 km/s
/// (SOI co-move), 1.5 s of stale-camera-vs-moving-server produces a
/// ~165 km position discrepancy that lands as a visible "teleport"
/// when the first WS finally arrives and `apply_worldstate_pose`
/// snaps the camera to the now-distant authoritative position.
///
/// `velocity` is from `SpawnPose.velocity` — server-authoritative,
/// not a client-side prediction. We're just continuing to apply the
/// server's last-known velocity until the next authoritative position
/// update, which IS what `SpawnPose.velocity` was designed for per
/// its own doc comment.
#[derive(Resource, Default, Clone, Copy)]
struct SpawnExtrapolation {
    velocity: DVec3,
    last_step: Option<Instant>,
}

#[derive(SystemSet, Clone, Debug, Hash, Eq, PartialEq)]
pub struct PlayerSyncSet;

pub struct PlayerSyncPlugin;

impl Plugin for PlayerSyncPlugin {
    fn build(&self, app: &mut App) {
        // Run AFTER WorldStateIngestSet (so we see this frame's
        // latest primary WorldState) and BEFORE ShardOriginSet (so
        // the rebase sees the updated CameraWorldPos for the current
        // frame).
        app.init_resource::<SpawnExtrapolation>()
            .configure_sets(
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
///
/// `pose.position` is the player's BODY position in system-space
/// (server's `EvaPosition` / ship-local `PlayerPosition`). The camera
/// sits `EYE_HEIGHT` above the body in the destination shard's body
/// frame — the same eye-offset that `apply_worldstate_pose` adds every
/// subsequent tick. We mirror that addition here so frame-0 (latch
/// consume) and frame-1 (first WS apply) place the camera at the same
/// height; without this the camera would visibly teleport `EYE_HEIGHT`
/// metres up at frame 1.
///
/// For SYSTEM target (EVA): body frame = world (after the EvaRotation
/// = IDENTITY fix), so eye = body + (0, EYE_HEIGHT, 0) world.
/// For SHIP target (boarding) and other body-rotated frames we'd need
/// the destination's body rotation, which isn't in the spawn pose;
/// we apply the world-+Y best-effort offset and the first WS may
/// visibly correct on rotated ships. Acceptable for now — primary
/// transition the user hits is SHIP→EVA.
///
/// `pose.rotation` is the full camera orientation at departure
/// (`body × head_look`). We write it directly to `Transform.rotation`
/// AND decompose its forward vector into `(yaw, pitch)` using the
/// convention `forward_from_look` / `rotation_from_look` use:
///   pitch = asin(D.y), yaw = atan2(D.z, D.x)
/// (NOT Bevy's Euler::YXZ — that's for the -Z-forward convention and
/// doesn't round-trip through `rotation_from_look`'s +X-forward
/// convention.) Roll is lost in this decomposition; for ships with
/// non-trivial roll the camera up-axis snaps to world +Y on frame 1.
fn consume_spawn_pose_latch(
    mut latch: ResMut<SpawnPoseLatch>,
    mut camera_world: ResMut<CameraWorldPos>,
    look: Res<LocalLook>,
    mut extrapolation: ResMut<SpawnExtrapolation>,
    mut cam_q: Query<&mut Transform, With<MainCamera>>,
) {
    let Some(pose) = latch.pose.take() else { return };
    latch.target_shard_type = None;
    let body_world = DVec3::new(pose.position.x, pose.position.y, pose.position.z);
    camera_world.pos = body_world + DVec3::new(0.0, EYE_HEIGHT as f64, 0.0);

    extrapolation.velocity = DVec3::new(
        pose.velocity.x,
        pose.velocity.y,
        pose.velocity.z,
    );
    extrapolation.last_step = Some(Instant::now());

    // `pose.rotation` from the server is the BODY frame (e.g.,
    // `exterior.rotation` for SHIP→EVA). The CAMERA at this instant is
    // `body × head_quat(LocalLook)` — exactly what
    // `update_camera_rotation` will compute every subsequent frame
    // once it reads body from the new primary's WS. We DON'T reset
    // LocalLook (head yaw/pitch); preserving it across the transition
    // is what keeps the player's view aim continuous, including any
    // body roll the server can't represent in the wire-format yaw/pitch.
    let body_rot = dquat_to_bevy(DQuat::from_xyzw(
        pose.rotation.x,
        pose.rotation.y,
        pose.rotation.z,
        pose.rotation.w,
    ));
    let head_rot = rotation_from_look(look.server_yaw, look.server_pitch);
    if let Ok(mut tf) = cam_q.single_mut() {
        tf.rotation = body_rot * head_rot;
    }

    tracing::info!(
        cam_world = ?(camera_world.pos.x, camera_world.pos.y, camera_world.pos.z),
        vel = ?(extrapolation.velocity.x, extrapolation.velocity.y, extrapolation.velocity.z),
        body_rot = ?(body_rot.x, body_rot.y, body_rot.z, body_rot.w),
        head_yaw_deg = look.server_yaw.to_degrees(),
        head_pitch_deg = look.server_pitch.to_degrees(),
        "spawn pose latch consumed (body × head preserved)",
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
    mut extrapolation: ResMut<SpawnExtrapolation>,
) {
    // While the new primary's first WS hasn't arrived yet, keep the
    // camera moving along the spawn velocity captured at latch time.
    // Each frame we add `velocity * dt` so the camera stays roughly
    // co-located with the still-moving server-side player. When the
    // first WS lands, we clear the extrapolation and let the
    // authoritative position take over with no visible snap.
    let Some(ws) = ws.latest.as_ref() else {
        if let Some(prev_step) = extrapolation.last_step {
            let now = Instant::now();
            let dt = now.saturating_duration_since(prev_step).as_secs_f64();
            if dt > 0.0 && extrapolation.velocity.length_squared() > 0.0 {
                camera_world.pos += extrapolation.velocity * dt;
            }
            extrapolation.last_step = Some(now);
        }
        return;
    };
    if primary.current.is_none() {
        return;
    }
    // Authoritative WS replaces the extrapolated value — stop
    // extrapolating until the next transition primes a new spawn
    // velocity.
    extrapolation.last_step = None;
    extrapolation.velocity = DVec3::ZERO;

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

/// Camera rotation = `ship_rot × rotation_from_look(LocalLook)`.
///
/// `ship_rot` comes from the own-ship `ObservableEntity` on SHIP
/// primaries (server-authoritative, 20 Hz); on SYSTEM / PLANET / GALAXY
/// primaries there's no own-ship so it falls back to identity. The
/// `rotation_from_look` half is driven by client-owned mouse input at
/// 60 Hz so the view tracks mouse motion smoothly between WS ticks.
/// LocalLook is round-tripped to the server every input packet, so the
/// server's parallel composition (used for handoffs / spawn poses /
/// other observers) lands on the same rotation within one tick of
/// network round-trip — convergence, not divergence.
///
/// Seated mode forward-locks the camera to ship: yaw/pitch in seated
/// inputs are pilot rates that drive seat torque signals, not head
/// angles, so composing them into the camera rotation would wobble the
/// view every mouse move.
fn update_camera_rotation(
    ws: Res<PrimaryWorldState>,
    conn: Res<NetConnection>,
    look: Res<LocalLook>,
    seated: Res<crate::input::SeatedState>,
    mut cam_q: Query<&mut Transform, With<MainCamera>>,
) {
    let Ok(mut tf) = cam_q.single_mut() else { return };
    // Body frame, in this order of precedence:
    //   1. The own-player's `players[].rotation` from the primary WS.
    //      Servers send the player's BODY rotation here (ship_rot for
    //      SHIP-piloting, EvaRotation for EVA, etc.). Composing
    //      body × LocalLook here recovers the camera while preserving
    //      arbitrary body roll — which a server-composed,
    //      client-decomposed approach cannot.
    //   2. Own-ship rotation from `entities[]` — fallback for shards
    //      that haven't migrated to body-only yet (and during the
    //      sub-tick window between transition and first WS).
    //   3. Identity.
    let body_rot = ws
        .latest
        .as_ref()
        .and_then(|ws| {
            ws.players
                .iter()
                .find(|p| p.player_id == conn.player_id)
                .map(|p| {
                    Quat::from_xyzw(
                        p.rotation.x as f32,
                        p.rotation.y as f32,
                        p.rotation.z as f32,
                        p.rotation.w as f32,
                    )
                })
        })
        .or_else(|| {
            ws.latest.as_ref().and_then(|ws| {
                ws.entities
                    .iter()
                    .find(|e| e.is_own && e.kind == EntityKind::Ship)
                    .map(|s| {
                        Quat::from_xyzw(
                            s.rotation.x as f32,
                            s.rotation.y as f32,
                            s.rotation.z as f32,
                            s.rotation.w as f32,
                        )
                    })
            })
        })
        .unwrap_or(Quat::IDENTITY);

    let head = if seated.is_seated && !look.free_look {
        Quat::IDENTITY
    } else {
        rotation_from_look(look.server_yaw, look.server_pitch)
    };
    tf.rotation = body_rot * head;
}
