//! Server-authoritative player camera + spawn-pose application.
//!
//! After the input and camera-frame extraction, this module's role is
//! narrow:
//!
//! * Consume `SpawnPoseLatch` on transitions → snap `PlayerState`,
//!   `LocalLook`, and `BodyRotation` so the first post-transition frame
//!   renders exactly at the server-authoritative pose.
//! * Ingest `NetEvent::WorldState` → maintain `PlayerState.world_pos` and
//!   (first time only) seed `LocalLook` from the server rotation so the
//!   camera starts oriented correctly on fresh connect.
//! * Apply `PlayerState.world_pos` + `CameraFrame.{rotation, up}` to the
//!   networked camera each frame — no yaw/pitch math here, the frame
//!   module owns that.
//!
//! Multiplayer: own-player filtered by `is_own` in `WorldState.entities`.
//! Everything about remote players + crewmates flows past unchanged for
//! the entity renderer (#11). Nothing here assumes single-player.

use bevy::prelude::*;
use glam::DVec3;

use crate::camera_frame::{planet_tangent_basis, BodyRotation, CameraFrame, CameraFrameSet};
use crate::input_system::{yaw_pitch_from_server_rotation, LocalLook, EYE_HEIGHT};
use crate::net_plugin::{GameEvent, NetConnection};
use crate::network::NetEvent;
use crate::shard_transition::{ShardContext, ShardTransitionSet, SpawnPoseLatch};
use voxeldust_core::client_message::{shard_type as st, EntityKind};

/// Marker component on the camera entity driven by this module.
#[derive(Component)]
pub struct NetworkedPlayerCamera;

pub struct PlayerSyncPlugin;

impl Plugin for PlayerSyncPlugin {
    fn build(&self, app: &mut App) {
        app
            .init_resource::<PlayerState>()
            // Pre-frame systems: consume spawn-pose latch, ingest server WS,
            // seed local look. Must run AFTER ShardTransitionSet (so the
            // latch is populated) and BEFORE CameraFrameSet (so the frame
            // is computed from up-to-date PlayerState / LocalLook /
            // BodyRotation).
            .add_systems(
                Update,
                (
                    consume_spawn_pose_latch,
                    ingest_worldstate,
                    seed_local_look,
                )
                    .chain()
                    .after(ShardTransitionSet)
                    .before(CameraFrameSet),
            )
            // Post-frame system: read CameraFrame + PlayerState, write
            // the camera Transform.
            .add_systems(Update, apply_server_pose.after(CameraFrameSet));
    }
}

/// Latest server-authoritative player pose, in whatever frame the current
/// primary shard broadcasts (ship-local on SHIP, planet-local on PLANET,
/// system-space on SYSTEM / GALAXY). Bevy's world origin = the primary's
/// `ws.origin` by construction of `shard_origin.rs`, so `world_pos`
/// applies directly to `Transform.translation` without any frame shift
/// beyond the eye-height offset along `CameraFrame.up`.
#[derive(Resource, Default)]
pub struct PlayerState {
    pub world_pos: Vec3,
    pub world_rot: Quat,
    pub last_tick: u64,
    /// Set when the first post-connect / post-transition WorldState with
    /// our player lands — lets `seed_local_look` run its one-shot yaw/pitch
    /// reseed exactly once per shard context.
    pub snapshot_seeded: bool,
}

/// Apply a server-authoritative spawn pose (from `ShardRedirect`) to the
/// player state, local look, and persistent `BodyRotation`. Runs before
/// `ingest_worldstate` so the first WorldState-driven update this frame
/// is already starting from the authoritative position.
///
/// Per-target-shard frame conversion:
///
/// * **SHIP** (boarding): pose.rotation is ship-local — normally
///   identity per the `core::handoff::SpawnPose` docs. We reset look to
///   0 and let `track_body_rotation` reset body_rot on the shard-type
///   edge. Ship rotation itself arrives via WorldState, not the pose.
/// * **SYSTEM** / **GALAXY** (EVA exit, warp): pose.rotation IS the body
///   rotation. We park it in `BodyRotation` and reset look to 0 so the
///   camera renders exactly `body_rot · Identity = pose.rotation`.
/// * **PLANET** (landing): pose.rotation is world-frame. Project the
///   world forward into the planet's tangent basis to derive the
///   tangent-frame yaw/pitch the PLANET camera arm expects. No special
///   body_rot — PLANET does not use it.
fn consume_spawn_pose_latch(
    mut latch: ResMut<SpawnPoseLatch>,
    mut state: ResMut<PlayerState>,
    mut look: ResMut<LocalLook>,
    mut body_rot: ResMut<BodyRotation>,
) {
    let Some(pose) = latch.pose.take() else { return };
    let pos = Vec3::new(
        pose.position.x as f32,
        pose.position.y as f32,
        pose.position.z as f32,
    );
    let rot_world = Quat::from_xyzw(
        pose.rotation.x as f32,
        pose.rotation.y as f32,
        pose.rotation.z as f32,
        pose.rotation.w as f32,
    );

    state.world_pos = pos;
    state.world_rot = rot_world;

    // `pose.rotation` is in the **server's** rotation convention (rotates
    // +X → "forward") — NOT Bevy's camera default. `yaw_pitch_from_server_rotation`
    // applies to +X to get forward in server convention, then extracts
    // legacy-convention (yaw, pitch) which is exactly the format the rest
    // of this module stores. This makes every shard arm below agree on
    // what "pose.rotation" means without per-shard Bevy-basis math.
    match pose.target_shard_type {
        t if t == st::PLANET => {
            // Project server-convention forward onto the planet tangent
            // frame. On PLANET the camera's local basis is `(north, up,
            // east)` per `camera_frame::compute_camera_frame`, so we
            // extract (yaw, pitch) such that `basis · forward_from_look
            // (yaw, pitch)` equals `pose.rotation · +X`.
            let cam_fwd_world = (rot_world * Vec3::X).normalize_or_zero();
            if let Some((east, up, north)) = planet_tangent_basis(pos) {
                let fwd_east = cam_fwd_world.dot(east);
                let fwd_up = cam_fwd_world.dot(up);
                let fwd_north = cam_fwd_world.dot(north);
                // rotation_from_look(yaw, pitch) produces local forward
                // (cos y · cos p, sin p, sin y · cos p); planet basis
                // maps (local +X → north, +Y → up, +Z → east), so:
                //   world_north = cos(yaw) · cos(pitch)
                //   world_up    = sin(pitch)
                //   world_east  = sin(yaw) · cos(pitch)
                look.pitch = fwd_up.clamp(-1.0, 1.0).asin();
                look.yaw = fwd_east.atan2(fwd_north);
            } else {
                look.yaw = 0.0;
                look.pitch = 0.0;
            }
            look.server_yaw = look.yaw;
            look.server_pitch = look.pitch;
            body_rot.rotation = Quat::IDENTITY;
        }
        t if t == st::SYSTEM || t == st::GALAXY => {
            // pose.rotation is the body rotation in server convention.
            // On SYSTEM/GALAXY our rendering math is `body_rot ·
            // rotation_from_look(yaw, pitch)`; we want that to equal
            // `rot_world` as a camera rotation, so setting `body_rot =
            // rot_world` (interpreted as "apply to +X to get forward")
            // and `look = (0, 0)` (which is forward = +X in server
            // convention) composes to the right forward.
            body_rot.rotation = rot_world;
            look.yaw = 0.0;
            look.pitch = 0.0;
            look.server_yaw = 0.0;
            look.server_pitch = 0.0;
        }
        t if t == st::SHIP => {
            // Ship-local frame: `pose.rotation` is identity by convention
            // (see `core::handoff::SpawnPose` doc — "For boarding, identity
            // (ship-local frame)"). Extract yaw/pitch anyway so any future
            // custom spawn pose authored on a specific seat still works.
            let (yaw, pitch) = yaw_pitch_from_server_rotation(rot_world);
            look.yaw = yaw;
            look.pitch = pitch;
            look.server_yaw = yaw;
            look.server_pitch = pitch;
            // Boarding resets body_rot; `track_body_rotation` also
            // enforces this on the shard-type edge but we do it here
            // too so the first post-latch frame is correct regardless
            // of ordering with the edge detector.
            body_rot.rotation = Quat::IDENTITY;
        }
        _ => {
            // Unknown target (legacy 255 sentinel): extract yaw/pitch
            // assuming server convention. body_rot untouched.
            let (yaw, pitch) = yaw_pitch_from_server_rotation(rot_world);
            look.yaw = yaw;
            look.pitch = pitch;
            look.server_yaw = yaw;
            look.server_pitch = pitch;
        }
    }

    state.snapshot_seeded = true;
    tracing::info!(
        target_shard_type = pose.target_shard_type,
        ?pos,
        yaw = look.yaw,
        pitch = look.pitch,
        "applied server-authoritative spawn pose"
    );
}

fn ingest_worldstate(
    mut state: ResMut<PlayerState>,
    ctx: Res<ShardContext>,
    mut events: MessageReader<GameEvent>,
) {
    for GameEvent(ev) in events.read() {
        match ev {
            NetEvent::Connected { .. } => {
                state.last_tick = 0;
                state.snapshot_seeded = false;
            }
            NetEvent::WorldState(ws) => {
                if ws.tick < state.last_tick {
                    continue;
                }
                let own_entity = ws.entities.iter().find(|e| e.is_own && e.kind.is_player());
                let (local_pos, local_rot) = if let Some(me) = own_entity {
                    (me.position, me.rotation)
                } else if let Some(p) = ws.players.first() {
                    (p.position, p.rotation)
                } else {
                    continue;
                };

                // Ship-local coordinates: compose with the ship's own
                // pose so `PlayerState.world_pos` is expressed in Bevy's
                // frame (where 0 = ws.origin). For the own ship this
                // simplifies to identity because the server places its
                // own origin at the ship; general composition keeps the
                // formula correct for every shard type without branches.
                let own_ship = ws
                    .entities
                    .iter()
                    .find(|e| e.is_own && e.kind == EntityKind::Ship);
                let (ship_pos, ship_rot) = if let Some(s) = own_ship {
                    (s.position, dquat_to_bevy(s.rotation))
                } else {
                    (DVec3::ZERO, Quat::IDENTITY)
                };

                let compose_with_ship = ctx.current == st::SHIP;
                let (world_pos, world_rot) = if compose_with_ship {
                    let local_pos_f32 = dvec3_to_bevy(local_pos);
                    let rotated_local = ship_rot * local_pos_f32;
                    let world_translation = dvec3_to_bevy(ship_pos) + rotated_local;
                    let player_local_rot = dquat_to_bevy(local_rot);
                    (world_translation, ship_rot * player_local_rot)
                } else {
                    (dvec3_to_bevy(local_pos), dquat_to_bevy(local_rot))
                };

                state.world_pos = world_pos;
                state.world_rot = world_rot;
                state.last_tick = ws.tick;
            }
            _ => {}
        }
    }
}

/// One-shot seeding of `LocalLook.yaw/pitch` on first post-connect
/// WorldState arrival. `consume_spawn_pose_latch` handles transitions
/// authoritatively; this handles fresh connects where no spawn pose is
/// present.
///
/// The server's `WorldState` does **not** broadcast an authoritative
/// head-yaw per player — what looks like a rotation on `PlayerSnapshotData`
/// / `ObservableEntityData` is actually the ship's world rotation
/// (see `ship-shard/main.rs:2497` and `:2567`), stored there for grace-
/// window rendering of departed ships. Deriving camera yaw/pitch from it
/// would point the camera in an arbitrary direction on every fresh
/// connect. Instead we seed to (0, 0) which in legacy convention means
/// "facing shard-local +X" — on SHIP that is ship-forward (pilot-seat
/// orientation), on PLANET it is east (arbitrary default, the next mouse
/// motion will move the camera responsively), on SYSTEM/GALAXY it is
/// the body frame's +X.
fn seed_local_look(
    mut state: ResMut<PlayerState>,
    mut look: ResMut<LocalLook>,
    _ctx: Res<ShardContext>,
) {
    if state.snapshot_seeded || state.last_tick == 0 {
        return;
    }
    let (yaw, pitch) = (0.0, 0.0);
    look.yaw = yaw;
    look.pitch = pitch;
    look.server_yaw = yaw;
    look.server_pitch = pitch;
    state.snapshot_seeded = true;
}

/// Apply the per-shard camera frame + server position to the camera
/// Transform. `CameraFrame.up` drives the eye-height offset so pitching
/// up inside a cockpit or on a planet does not shift the eye forward.
fn apply_server_pose(
    state: Res<PlayerState>,
    frame: Res<CameraFrame>,
    conn: Res<NetConnection>,
    mut q: Query<&mut Transform, With<NetworkedPlayerCamera>>,
) {
    if !conn.connected || state.last_tick == 0 {
        return;
    }
    let Ok(mut tf) = q.single_mut() else { return };
    tf.translation = state.world_pos + frame.up * EYE_HEIGHT;
    tf.rotation = frame.rotation;
    // Diagnostic: log what the camera is actually being set to, once a
    // second, so we can tell if mouse motion is reaching the transform
    // vs. being applied and then overwritten.
    {
        use std::sync::atomic::{AtomicU64, Ordering};
        static LAST_LOG: AtomicU64 = AtomicU64::new(0);
        let now_ms = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_millis() as u64)
            .unwrap_or(0);
        let last = LAST_LOG.load(Ordering::Relaxed);
        if now_ms.saturating_sub(last) > 1000 {
            LAST_LOG.store(now_ms, Ordering::Relaxed);
            let fwd = tf.rotation * Vec3::NEG_Z;
            tracing::info!(
                pos = ?tf.translation,
                fwd = ?fwd,
                rot = ?tf.rotation,
                "camera transform applied"
            );
        }
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn dquat_to_bevy(q: glam::DQuat) -> Quat {
    Quat::from_xyzw(q.x as f32, q.y as f32, q.z as f32, q.w as f32)
}

fn dvec3_to_bevy(v: DVec3) -> Vec3 {
    Vec3::new(v.x as f32, v.y as f32, v.z as f32)
}
