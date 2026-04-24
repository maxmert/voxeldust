//! Per-shard camera frame computation.
//!
//! Four coordinate frames in voxeldust each require a different mapping
//! from "local yaw + pitch" to a world-space camera rotation + up vector:
//!
//! | Shard  | Up (for eye offset)       | Yaw/pitch interpretation        |
//! |--------|---------------------------|---------------------------------|
//! | SHIP   | `ship_rot · Y`            | ship-local basis (walking) or   |
//! |        |                           | forward-locked (piloting)       |
//! | PLANET | radial (player pos norm.) | tangent-plane (yaw around up,   |
//! |        |                           | pitch around east)              |
//! | SYSTEM | `body_rot · Y`            | body-local basis; `body_rot`    |
//! |        |                           | persists across ship-exit       |
//! | GALAXY | `Y`                       | world-aligned (skybox / warp)   |
//!
//! Implementation maps each case to a Quat that, applied to Bevy's
//! camera default (forward = -Z, up = +Y), produces the correct world-
//! space look. No per-frame trig reconstruction in the render path —
//! everything lives in `CameraFrame` so consumers (`apply_server_pose`,
//! HUD horizon, future aerial perspective) read a single resource.
//!
//! Persistent `body_rotation` across SHIP → SYSTEM transition is the
//! detail that makes EVA exits feel seamless: at the moment the player
//! leaves the cockpit, the ship's current world rotation is copied into
//! `BodyRotation` so the view direction is visually continuous (no Y-up
//! snap in vacuum). The opposite transition (SYSTEM → SHIP, boarding)
//! resets body_rot to identity because ship-local yaw/pitch become the
//! new frame basis.
//!
//! Multiplayer: the `ShipRotation` resource tracks **our** ship (the one
//! the own-player is inside). Every other player's rendering — crewmates
//! on the same ship, pilots on nearby ships, EVA neighbors — flows
//! through `WorldState.entities` and is task #11 territory. The camera
//! frame only cares about the single ship the client is currently
//! attached to.

use bevy::math::Mat3;
use bevy::prelude::*;
use glam::{DQuat, DVec3};

use crate::input_system::{rotation_from_look, InputSet, LocalLook, PilotState};
use crate::net_plugin::GameEvent;
use crate::network::NetEvent;
use crate::shard_transition::{ShardContext, ShardTransitionSet};
use voxeldust_core::client_message::{shard_type as st, EntityKind};

/// System set for every camera-frame system. Consumers run after this set
/// so they observe a committed `CameraFrame` for the current tick.
#[derive(SystemSet, Clone, Eq, PartialEq, Hash, Debug)]
pub struct CameraFrameSet;

pub struct CameraFramePlugin;

impl Plugin for CameraFramePlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<CameraFrame>()
            .init_resource::<ShipRotation>()
            .init_resource::<BodyRotation>()
            .add_systems(
                Update,
                (
                    track_ship_rotation,
                    track_body_rotation,
                    compute_camera_frame,
                )
                    .chain()
                    .in_set(CameraFrameSet)
                    .after(ShardTransitionSet)
                    .after(InputSet),
            );
    }
}

// ---------------------------------------------------------------------------
// Resources
// ---------------------------------------------------------------------------

/// The output of the per-frame camera-frame computation. A single source
/// of truth for every consumer (`apply_server_pose`, HUD horizon, future
/// aerial-perspective sampling).
#[derive(Resource, Default, Debug, Clone, Copy)]
pub struct CameraFrame {
    /// World-space up used for the eye-height offset and any UI element
    /// anchored to the local "down." Distinct from `rotation * Y` —
    /// pitching up should never tilt the eye-height offset.
    pub up: Vec3,
    /// World-space rotation to apply to the camera. Takes Bevy's default
    /// camera basis (-Z forward, +Y up) to the correct world-frame look.
    pub rotation: Quat,
}

/// Our-ship world rotation. Updated every `WorldState` from the entity
/// marked `is_own && kind == Ship` on SHIP primaries. Left at identity on
/// non-SHIP shards. Used by the SHIP arm of `compute_camera_frame` and
/// by the SHIP → SYSTEM transition to seed `BodyRotation`.
#[derive(Resource, Default, Debug, Clone, Copy)]
pub struct ShipRotation {
    pub rotation: Quat,
    /// True if we have a live `is_own` ship entity this tick. Guards the
    /// SHIP camera arm from reading a stale rotation on a primary where
    /// the ship entity hasn't arrived yet.
    pub has_own_ship: bool,
}

/// Persistent EVA body rotation. Updated by:
///  - On SHIP → SYSTEM transition: seeded from `ShipRotation` so the
///    first EVA frame sees the same world-space forward as the last
///    ship-frame.
///  - On SYSTEM → SHIP transition: reset to identity because ship-local
///    yaw/pitch become the new frame basis and identity avoids a double-
///    rotation on boarding.
///  - On SPAWN pose: replaced with the `SpawnPose.rotation` because the
///    server-authoritative spawn pose already accounts for the correct
///    body frame at the handoff moment.
#[derive(Resource, Default, Debug, Clone, Copy)]
pub struct BodyRotation {
    pub rotation: Quat,
}

// ---------------------------------------------------------------------------
// Systems
// ---------------------------------------------------------------------------

fn track_ship_rotation(
    mut events: MessageReader<GameEvent>,
    ctx: Res<ShardContext>,
    mut ship_rot: ResMut<ShipRotation>,
) {
    for GameEvent(ev) in events.read() {
        let NetEvent::WorldState(ws) = ev else { continue };
        if ctx.current != st::SHIP {
            ship_rot.has_own_ship = false;
            continue;
        }
        if let Some(own) = ws
            .entities
            .iter()
            .find(|e| e.is_own && e.kind == EntityKind::Ship)
        {
            ship_rot.rotation = dquat_to_bevy(own.rotation);
            ship_rot.has_own_ship = true;
        }
    }
}

/// Maintain `BodyRotation` across local shard-type edges as a fallback
/// for when the server does NOT supply a `spawn_pose` on the transition.
/// When a spawn pose IS supplied, `player_sync::consume_spawn_pose_latch`
/// writes the authoritative body_rot earlier in the same frame — this
/// system respects that via change detection and skips.
///
/// SHIP → SYSTEM carries the ship's world rotation so EVA exits have no
/// yaw/pitch snap; `_ → SHIP` resets to identity because ship-local
/// yaw/pitch becomes the new basis.
fn track_body_rotation(
    ctx: Res<ShardContext>,
    ship_rot: Res<ShipRotation>,
    mut body_rot: ResMut<BodyRotation>,
    mut prev_shard_type: Local<Option<u8>>,
) {
    let prev = prev_shard_type.replace(ctx.current);
    let Some(prev) = prev else { return };
    if prev == ctx.current {
        return;
    }
    // If consume_spawn_pose_latch already set body_rot this frame, the
    // server-authoritative value takes precedence — don't overwrite
    // with the shard-edge fallback.
    if body_rot.is_changed() {
        return;
    }
    match (prev, ctx.current) {
        (st::SHIP, st::SYSTEM) => {
            // EVA exit without a spawn_pose: carry the ship's world
            // rotation so yaw/pitch are continuous across the hatch.
            if ship_rot.has_own_ship {
                body_rot.rotation = ship_rot.rotation;
            }
        }
        (_, st::SHIP) => {
            // Boarding any ship: ship-local yaw/pitch become the new
            // basis. body_rot should not layer on top of ship_rot at
            // render time.
            body_rot.rotation = Quat::IDENTITY;
        }
        _ => {}
    }
}

/// Compute the final camera `rotation` and `up` for the frame based on
/// current shard type, piloting state, and the three look inputs
/// (`LocalLook`, `ShipRotation`, `BodyRotation`).
fn compute_camera_frame(
    ctx: Res<ShardContext>,
    pilot: Res<PilotState>,
    look: Res<LocalLook>,
    ship_rot: Res<ShipRotation>,
    body_rot: Res<BodyRotation>,
    player_world_pos: Res<crate::player_sync::PlayerState>,
    mut frame: ResMut<CameraFrame>,
) {
    // All four arms: `rotation_from_look(yaw, pitch)` is the shard-local
    // camera rotation in Bevy's basis whose forward matches the legacy/
    // server convention (yaw=0 → local +X, positive yaw rotates toward
    // +Z). Each arm then composes it with the shard's world-frame
    // anchor (ship rotation, planet tangent basis, body rotation).
    match ctx.current {
        st::SHIP => {
            let ship_q = if ship_rot.has_own_ship {
                ship_rot.rotation
            } else {
                Quat::IDENTITY
            };
            // Piloting without free-look: camera is locked to the ship's
            // forward. Yaw/pitch deliberately ignored — the user torques
            // the ship via `PilotRates`, not the view.
            let local_look = if pilot.is_piloting && !look.free_look {
                rotation_from_look(0.0, 0.0)
            } else {
                rotation_from_look(look.yaw, look.pitch)
            };
            frame.rotation = ship_q * local_look;
            // Eye-height always along the ship's up axis so pitching up
            // inside the cockpit doesn't shift the head forward.
            frame.up = ship_q * Vec3::Y;
        }
        st::PLANET => {
            // Tangent-plane basis: (north, up, east) so that the local
            // rotation at yaw=0, pitch=0 produces a world-space forward
            // aligned with **north**. Matches legacy `camera.rs:88`
            // formula: `fwd = north·cos(y) + up·sin(p) + east·sin(y)`.
            //
            // Matrix columns are basis vectors: Bevy +X → north,
            // +Y → up, +Z → east. Then `basis · local_fwd` (where
            // local_fwd in the rotation_from_look formula is
            // (cos(y)cp, sin(p), sin(y)cp)) expands to
            // `north·cos(y)cp + up·sin(p) + east·sin(y)cp` exactly.
            if let Some((east, up, north)) = planet_tangent_basis(player_world_pos.world_pos) {
                let basis = Quat::from_mat3(&Mat3::from_cols(north, up, east));
                frame.rotation = basis * rotation_from_look(look.yaw, look.pitch);
                frame.up = up;
            } else {
                // Degenerate case — player at planet center. Fall back
                // to world-Y. Only happens during the 1-frame handoff
                // before the first PLANET WS arrives with a valid
                // planet-local position.
                frame.rotation = rotation_from_look(look.yaw, look.pitch);
                frame.up = Vec3::Y;
            }
        }
        st::SYSTEM | st::GALAXY => {
            // EVA / warp: body-local yaw/pitch composed onto persistent
            // `body_rot`. On SYSTEM this keeps the world-view continuous
            // across ship-exit (body_rot = ship_rot at exit); on GALAXY
            // it stays identity unless a spawn_pose overrides.
            frame.rotation = body_rot.rotation * rotation_from_look(look.yaw, look.pitch);
            frame.up = body_rot.rotation * Vec3::Y;
        }
        _ => {
            // Unknown shard type (e.g. pre-first-Connected). Render
            // world-aligned so the camera at least has valid up/rotation
            // rather than uninitialised junk.
            frame.rotation = rotation_from_look(look.yaw, look.pitch);
            frame.up = Vec3::Y;
        }
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Derive an orthonormal (east, up, north) tangent basis at the player's
/// planet-local position. Matches the server's basis so yaw = 0 always
/// faces the same cardinal direction on any planet seed.
///
/// `up` is the radial outward from planet center. `east` is chosen as
/// `pole × up` where `pole = Y` — the equatorial convention most
/// software uses. At the geographic poles where `pole × up` vanishes, we
/// fall back to `Z × up` so the basis is still well-defined.
///
/// Returns `None` when the player is at planet center (magnitude ≈ 0)
/// because `up` is undefined there.
pub fn planet_tangent_basis(pos: Vec3) -> Option<(Vec3, Vec3, Vec3)> {
    if pos.length_squared() < 1e-4 {
        return None;
    }
    let up = pos.normalize();
    let pole = Vec3::Y;
    let east_raw = pole.cross(up);
    let east = if east_raw.length_squared() > 1e-10 {
        east_raw.normalize()
    } else {
        Vec3::Z.cross(up).normalize()
    };
    let north = up.cross(east).normalize();
    Some((east, up, north))
}

fn dquat_to_bevy(q: DQuat) -> Quat {
    Quat::from_xyzw(q.x as f32, q.y as f32, q.z as f32, q.w as f32)
}
