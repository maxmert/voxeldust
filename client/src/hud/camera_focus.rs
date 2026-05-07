//! Generic camera focus for block-face HUD subblocks. Drives off
//! `HudFocusState`, not Terminal-specific state — any HUD widget on
//! any block face can engage the camera lerp by calling
//! `HudFocusState::engage_block_tile`.
//!
//! ## Behaviour
//!
//! When `HudFocusState.origin` becomes `BlockFace`, the camera
//! `Transform` smoothly leans in toward the focused tile's plane.
//! Held there while engaged. On disengage the camera lerps back to
//! the player's natural Transform (translation = ZERO, rotation =
//! body × head as written by `PlayerSyncSet`). All under floating-
//! origin: we override the camera's Bevy-local Transform; the f64
//! `CameraWorldPos` resource is never touched, so chunk streaming /
//! origin rebase keep working unchanged.
//!
//! ## Reading position
//!
//! Camera target = `tile_world_pos + tile_normal × READING_DISTANCE`.
//! `tile_world_pos` is the tile's `GlobalTransform.translation`
//! (already in Bevy-local space because it's a child of the chunk's
//! `ChunkSource` which the rebase pipeline anchors). The tile mesh
//! is a `Rectangle` in XY with normal +Z; in world space the +Z axis
//! of `GlobalTransform` is the outward face normal.
//!
//! ## Star Citizen idiom
//!
//! AAA "use the device" interactions lean the camera toward the
//! device a few centimetres while the player operates it. Half a
//! second of smoothstep ease-in-out reads as deliberate without
//! making the player wait. On disengage the camera glides back over
//! the same window — by t=1 the camera Transform exactly matches
//! what `PlayerSyncSet` would have written, so there's no snap when
//! the override stops.

use bevy::prelude::*;

use crate::camera::PlayerSyncSet;
use crate::hud::focus::{HudFocusOrigin, HudFocusState, HudFocusSet};
use crate::MainCamera;

/// Camera-focus animation duration, both directions. Star-Citizen's
/// terminal-focus is in this range; long enough to register as
/// "the camera moved," short enough that the player doesn't sit and
/// watch.
pub const CAMERA_FOCUS_DURATION_S: f32 = 0.30;

/// Distance from the surface, in metres, at which the camera settles
/// while engaged. The HUD panel is 0.9 m × 0.9 m; at the player's
/// default ~60° vertical FOV, a viewing distance of `d` m shows a
/// vertical span of roughly `1.15 · d`. We want the whole 0.9 m face
/// in view with a comfortable ~30 % margin (no glyph clipping at
/// edges, no FOV-edge fisheye), which works out to **≈ 1.05 m**.
/// Closer than that and the face's corners crop out of frame.
pub const READING_DISTANCE_M: f32 = 1.05;

/// Phase machine for the focus animation. Each state stores the data
/// the lerp needs:
/// - `Engaging`: started_at; lerps from current Transform → target.
/// - `Engaged`: hold at target.
/// - `Disengaging`: started_at + frozen "from" pose so the lerp-back
///   begins exactly where the engaged hold ended (no jitter from a
///   frame-1 surface-plane micro-update).
#[derive(Default, Clone, Copy, Debug)]
enum FocusPhase {
    #[default]
    Idle,
    Engaging {
        started_at: f32,
    },
    Engaged,
    Disengaging {
        started_at: f32,
        from_translation: Vec3,
        from_rotation: Quat,
    },
}

#[derive(Resource, Default)]
struct CameraFocusState {
    phase: FocusPhase,
    /// Last-engaged tile entity — used to detect target swaps mid-
    /// animation (re-engaging another block while leaning in).
    last_tile: Option<Entity>,
}

pub struct HudCameraFocusPlugin;

impl Plugin for HudCameraFocusPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<CameraFocusState>().add_systems(
            Update,
            apply_hud_camera_focus
                // After PlayerSyncSet (which writes body × head into
                // Transform.rotation each frame) so we override on
                // top. After HudFocusSet so this frame's focus state
                // is settled before we read it.
                .after(PlayerSyncSet)
                .after(HudFocusSet),
        );
    }
}

/// Cubic ease-in-out — C¹-continuous, no overshoot. Standard pick
/// for camera focus animation.
#[inline]
fn smoothstep(t: f32) -> f32 {
    let t = t.clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

fn apply_hud_camera_focus(
    focus: Res<HudFocusState>,
    mut state: ResMut<CameraFocusState>,
    time: Res<Time>,
    tile_q: Query<&GlobalTransform>,
    mut cam_q: Query<&mut Transform, With<MainCamera>>,
) {
    let Ok(mut tf) = cam_q.single_mut() else { return };
    let now = time.elapsed_secs();

    // Only block-face engagements trigger the lerp. Tablet focus is
    // camera-relative and doesn't need a camera move.
    let engaged_tile = match focus.origin {
        HudFocusOrigin::BlockFace => focus.focused_tile,
        _ => None,
    };

    // Compute target Transform from the tile's world transform. The
    // tile's `Rectangle` mesh lies in local XY with normal +Z, and
    // the spawn rotates local +Z onto the host block's face outward
    // normal. In Bevy's right-handed convention `Transform.forward()`
    // returns the -Z axis (camera looks toward -Z); the +Z axis —
    // i.e. the outward normal in world space — is `Transform.back()`.
    // Camera sits along the outward normal at READING_DISTANCE from
    // the surface, looking at the surface centre with surface +Y as
    // up (keeps glyph "up" matching view "up"). For ceiling / floor
    // tiles the surface +Y winds up sideways relative to world up,
    // which is correct: the player's view inherits the panel's local
    // orientation rather than the world.
    let target = engaged_tile
        .and_then(|tile| tile_q.get(tile).ok())
        .map(|gt| {
            let surface_pos = gt.translation();
            let outward = gt.back().as_vec3();
            let surface_up = gt.up().as_vec3();
            let cam_pos = surface_pos + outward * READING_DISTANCE_M;
            let look_dir = (surface_pos - cam_pos).normalize_or_zero();
            let rotation = if look_dir.length_squared() > 0.0 {
                Transform::IDENTITY
                    .looking_to(look_dir, surface_up)
                    .rotation
            } else {
                Quat::IDENTITY
            };
            (cam_pos, rotation)
        });

    // Re-engage on a different tile mid-animation: snap to a fresh
    // Engaging from the current pose. Disengage if the focus dropped.
    let tile_changed = state.last_tile != engaged_tile;
    state.last_tile = engaged_tile;

    let new_phase = match (engaged_tile.is_some(), tile_changed, state.phase) {
        (true, true, _) => FocusPhase::Engaging { started_at: now },
        (true, false, FocusPhase::Idle) => FocusPhase::Engaging { started_at: now },
        (true, false, FocusPhase::Engaging { started_at })
            if now - started_at >= CAMERA_FOCUS_DURATION_S =>
        {
            FocusPhase::Engaged
        }
        (false, _, FocusPhase::Engaged) => FocusPhase::Disengaging {
            started_at: now,
            from_translation: tf.translation,
            from_rotation: tf.rotation,
        },
        (false, _, FocusPhase::Engaging { .. }) => FocusPhase::Disengaging {
            started_at: now,
            from_translation: tf.translation,
            from_rotation: tf.rotation,
        },
        (false, _, FocusPhase::Disengaging { started_at, .. })
            if now - started_at >= CAMERA_FOCUS_DURATION_S =>
        {
            FocusPhase::Idle
        }
        (_, _, phase) => phase,
    };
    state.phase = new_phase;

    // Apply per-phase Transform.
    match state.phase {
        FocusPhase::Idle => {
            // Normal play. Body × head is in `tf.rotation` already
            // (PlayerSyncSet wrote it earlier this frame). Restore
            // translation to the floating-origin invariant ZERO.
            if tf.translation != Vec3::ZERO {
                tf.translation = Vec3::ZERO;
            }
        }
        FocusPhase::Engaging { started_at } => {
            let Some((target_t, target_r)) = target else {
                return;
            };
            let raw = ((now - started_at) / CAMERA_FOCUS_DURATION_S).clamp(0.0, 1.0);
            let t = smoothstep(raw);
            tf.translation = Vec3::ZERO.lerp(target_t, t);
            tf.rotation = tf.rotation.slerp(target_r, t);
        }
        FocusPhase::Engaged => {
            let Some((target_t, target_r)) = target else { return };
            tf.translation = target_t;
            tf.rotation = target_r;
        }
        FocusPhase::Disengaging {
            started_at,
            from_translation,
            from_rotation,
        } => {
            let raw = ((now - started_at) / CAMERA_FOCUS_DURATION_S).clamp(0.0, 1.0);
            let t = smoothstep(raw);
            tf.translation = from_translation.lerp(Vec3::ZERO, t);
            // `tf.rotation` is body × head this frame — slerping
            // toward it converges on the live trajectory at t=1, so
            // the moment Idle takes over there's no snap.
            tf.rotation = from_rotation.slerp(tf.rotation, t);
        }
    }
}
