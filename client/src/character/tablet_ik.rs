//! Tablet hold IK — both arms drive the held tablet pose.
//!
//! Phase J of the character system plan. Two cooperating bone chains:
//!
//! - **Right arm** (shoulder → elbow → wrist) — IK target is a fixed
//!   point on the tablet plane (`class.tablet_right_grip_uv`), so the
//!   right hand pins the device to a world position derived from the
//!   chest bone + a per-class offset / pitch.
//! - **Left arm** (shoulder → elbow → wrist) — IK target is the
//!   server-replicated cursor UV projected onto the tablet plane,
//!   raised by `class.tablet_finger_lift` so the index fingertip
//!   floats just above the screen.
//!
//! On top of those, the LEFT WRIST gets an extra rotation that aligns
//! its `hand_finger_axis_local` axis with the tablet plane's negative
//! normal — i.e. the index finger points at the screen instead of
//! whatever direction the animation graph left the wrist in.
//!
//! # Why anchor to the chest, not the camera
//!
//! Anchoring to a chest bone makes the tablet move with the body
//! rather than with the camera. That gives the same tablet pose to
//! every observer (local FP camera, remote 3rd-person, future replay)
//! without bespoke per-view code, and it lets the eventual ragdoll
//! / damage / IK-disable paths drop the tablet pose by simply
//! disabling this one system.
//!
//! # Composition order
//!
//! Runs in `PostUpdate`, after `AnimationSystems` (so we layer onto
//! the clip pose), after `CharacterCameraSet` (so the FP override is
//! already in effect for the local visual), and before
//! `TransformSystems::Propagate` (so the IK'd local rotations fold
//! into this frame's `GlobalTransform`). Same window as foot IK and
//! look-at IK — multiple IK passes happily layer because each writes
//! disjoint bone Transforms.

use bevy::app::AnimationSystems;
use bevy::input::mouse::MouseButton;
use bevy::prelude::*;
use bevy::transform::TransformSystems;

use glam::{Quat as GQuat, Vec3 as GVec3};

use voxeldust_core::character::{
    ik::{solve_aim_chain, solve_two_bone_ik, AimChainInput, TwoBoneInput},
    CharacterClass,
};

use crate::hud::focus::{HudFocusOrigin, HudFocusState};
use crate::hud::tablet::HeldTablet;
use crate::remote::RemotePlayers;

use super::assets::CharacterAssetRegistry;
use super::bind_pose::HandBindData;
use super::camera_attach::{CharacterCameraSet, LocalCharacterTag};
use super::foot_ik::advance_blend;
use super::render::{BoneRegistry, RemoteCharacterTag};

#[derive(SystemSet, Clone, Eq, PartialEq, Hash, Debug)]
pub struct CharacterTabletIkSet;

/// Per-character blend strength + sticky cursor + tap-pulse state.
/// `weight` ramps up when the character starts holding a tablet and
/// back down when they stop, fading both arms in / out smoothly.
/// `last_cursor_uv` holds the most recent cursor target across the
/// fade-out window so the finger doesn't snap to the screen centre
/// when the tablet closes. `tap_t` is `Some(elapsed_secs)` while a
/// click animation is in flight — the IK system reads it to drive
/// the fingertip down to the tablet plane and back up.
#[derive(Component, Debug, Default)]
pub struct TabletIkBlendState {
    pub weight: f32,
    pub last_cursor_uv: Vec2,
    pub tap_t: Option<f32>,
}

pub struct CharacterTabletIkPlugin;

impl Plugin for CharacterTabletIkPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<FpTabletCameraPitchBlend>();
        app.add_systems(Update, (update_tablet_blend, apply_fp_tablet_camera_pitch));
        app.add_systems(
            PostUpdate,
            (
                apply_tablet_head_look,
                apply_tablet_ik,
                apply_tablet_left_hand_finger_curl,
            )
                .chain()
                .after(AnimationSystems)
                .after(CharacterCameraSet)
                .before(TransformSystems::Propagate)
                .in_set(CharacterTabletIkSet),
        );
    }
}

/// Phase J — smoothed FP camera pitch offset applied while the local
/// player holds a tablet. Ramps to `class.tablet_fp_camera_pitch` over
/// `arm_blend_in_secs` on open, back to 0 over `arm_blend_out_secs`
/// on close — same easing as the arm IK so the camera dip and the
/// arm raise feel like a single gesture.
#[derive(Resource, Default, Debug)]
struct FpTabletCameraPitchBlend {
    /// Current applied pitch angle in radians (camera-local +X axis,
    /// negative = down).
    pub current: f32,
}

/// Apply a downward pitch to the main camera while the local player
/// holds a tablet in FIRST-PERSON. The chest bone is ~30 cm below the
/// FP eye line; with Bevy's 45° vertical FOV, a chest-held device
/// falls below the view frustum. Tilting the camera down brings the
/// pad into view without forcing the player to manually mouse-look.
///
/// Player input still works — `update_camera_rotation` writes the
/// mouse-driven rotation first, and we compose the pitch offset
/// AFTER that, so mouse-look navigates relative to the offset
/// orientation (look up brings the world horizon back into view,
/// look down dives further into the pad).
fn apply_fp_tablet_camera_pitch(
    held: Query<(), With<crate::hud::tablet::HeldTablet>>,
    mode: Res<crate::character::camera_attach::CameraMode>,
    asset_registry: Res<CharacterAssetRegistry>,
    time: Res<Time>,
    mut state: ResMut<FpTabletCameraPitchBlend>,
    mut cam_q: Query<&mut Transform, With<crate::MainCamera>>,
) {
    use crate::character::camera_attach::CameraMode;

    let Some(assets) = asset_registry.ready(0) else {
        return;
    };
    let class = assets.class;

    let in_fp = matches!(*mode, CameraMode::FirstPerson);
    let holding = !held.is_empty();
    let target = if in_fp && holding {
        class.tablet_fp_camera_pitch
    } else {
        0.0
    };

    let dt = time.delta().as_secs_f32().min(0.25);
    let going_up = target.abs() > state.current.abs();
    let blend_secs = if going_up {
        class.arm_blend_in_secs
    } else {
        class.arm_blend_out_secs
    };
    let denom = blend_secs.max(1e-3);
    let step = (target - state.current).clamp(-dt / denom, dt / denom);
    state.current += step;

    // Skip writing the transform if the pitch is negligible — the
    // camera is already where `update_camera_rotation` left it.
    if state.current.abs() < 1e-4 {
        return;
    }

    let Ok(mut tf) = cam_q.single_mut() else {
        return;
    };
    // Compose AFTER the player's mouse-look rotation: tf.rotation
    // is `body_rot * head` from `update_camera_rotation`. Right-
    // multiplying by a camera-local pitch around +X applies the
    // tilt in the camera's local frame.
    let auto_pitch = Quat::from_axis_angle(Vec3::X, state.current);
    tf.rotation = tf.rotation * auto_pitch;
}

/// Drive `TabletIkBlendState.weight` toward 1.0 for any character
/// that's currently holding a tablet (local: `HeldTablet` entity
/// exists; remote: `RemoteEntity.is_holding_tablet`) and toward 0.0
/// otherwise. Lazy-inserts the component on the first observed visual.
/// Also advances the local player's tap pulse — a fresh LMB press
/// while the tablet has focus starts a new tap, and `tap_t` ticks
/// forward each frame until the press + release window completes.
fn update_tablet_blend(
    mut commands: Commands,
    time: Res<Time>,
    asset_registry: Res<CharacterAssetRegistry>,
    remote_players: Res<RemotePlayers>,
    held_tablet: Query<(), With<HeldTablet>>,
    focus: Res<HudFocusState>,
    mouse: Res<ButtonInput<MouseButton>>,
    mut visuals: Query<(
        Entity,
        &RemoteCharacterTag,
        Has<LocalCharacterTag>,
        Option<&mut TabletIkBlendState>,
    )>,
) {
    let dt = time.delta().as_secs_f32().min(0.25);
    let local_holding = !held_tablet.is_empty();
    // A tap fires only on the LOCAL player and only while the tablet
    // has focus — clicks while the focus has moved off the tablet
    // (e.g. focus was kicked back to the world) belong to the world,
    // not the screen.
    let local_tap_pressed = local_holding
        && matches!(focus.origin, HudFocusOrigin::Tablet)
        && focus.active
        && mouse.just_pressed(MouseButton::Left);

    for (entity, tag, is_local, blend_opt) in &mut visuals {
        let Some(assets) = asset_registry.ready(tag.class_id) else {
            continue;
        };
        let class = assets.class;

        let (target, cursor_uv) = if is_local {
            (
                if local_holding { 1.0 } else { 0.0 },
                Vec2::new(focus.cursor_uv.x, focus.cursor_uv.y),
            )
        } else {
            match remote_players.by_id.get(&tag.player_id) {
                Some(r) if r.is_holding_tablet => (
                    1.0,
                    Vec2::new(r.tablet_cursor_uv.x, r.tablet_cursor_uv.y),
                ),
                Some(_) => (0.0, Vec2::ZERO),
                None => (0.0, Vec2::ZERO),
            }
        };

        match blend_opt {
            Some(mut blend) => {
                advance_blend(
                    &mut blend.weight,
                    target,
                    dt,
                    class.arm_blend_in_secs,
                    class.arm_blend_out_secs,
                );
                if target > 0.0 {
                    // Only freshen the cached cursor while a tablet is
                    // actively held. During the fade-out, hold the
                    // last value so the finger lingers at the final
                    // touch point rather than snapping to a stale
                    // remote default of (0, 0).
                    blend.last_cursor_uv = cursor_uv;
                }
                if is_local && local_tap_pressed {
                    // New tap — restart the animation even if a prior
                    // tap was still in flight. Avoids a stutter where
                    // a fast double-click would only register one
                    // visible press.
                    blend.tap_t = Some(0.0);
                }
                if let Some(t) = blend.tap_t.as_mut() {
                    *t += dt;
                    let total = (class.tap_press_secs + class.tap_release_secs).max(1e-3);
                    if *t >= total {
                        blend.tap_t = None;
                    }
                }
            }
            None => {
                commands
                    .entity(entity)
                    .insert(TabletIkBlendState {
                        weight: target,
                        last_cursor_uv: cursor_uv,
                        tap_t: None,
                    });
            }
        }
    }
}

/// Phase J — head + neck aim-IK toward the held tablet's centre.
/// Reuses [`solve_aim_chain`] (the same solver `look_ik` uses for
/// NPC attention) so the head ramps onto the device with the same
/// shared yaw/pitch limits and easing. Runs for ANY character with
/// `TabletIkBlendState` and `weight > 0` — local + remote, both FP
/// and TP. The local FP camera is independent of the head bone (per
/// `camera_follow_head`'s body-anchor design), so this can pitch the
/// head freely without disturbing the player's mouse-look.
fn apply_tablet_head_look(
    visuals: Query<(
        Entity,
        &RemoteCharacterTag,
        &BoneRegistry,
        &TabletIkBlendState,
    )>,
    asset_registry: Res<CharacterAssetRegistry>,
    parents: Query<&ChildOf>,
    mut transforms: Query<&mut Transform>,
) {
    /// Skip the math when the IK contribution is invisible.
    const MIN_WEIGHT: f32 = 0.01;

    for (visual_e, tag, bones, blend) in &visuals {
        if !bones.resolved || blend.weight < MIN_WEIGHT {
            continue;
        }
        let Some(assets) = asset_registry.ready(tag.class_id) else {
            continue;
        };
        let class = assets.class;

        let Some(chest_e) = bones.get(class.chest_bone) else { continue };
        let Some(neck_e) = bones.get(class.neck_bone) else { continue };
        let Some(head_e) = bones.get(class.head_bone) else { continue };

        let Some((_, visual_rot)) = walk_bone_world_pose(visual_e, &parents, &transforms) else {
            continue;
        };
        let Some((chest_pos, _)) = walk_bone_world_pose(chest_e, &parents, &transforms) else {
            continue;
        };
        let Some((neck_pos, neck_world_rot)) = walk_bone_world_pose(neck_e, &parents, &transforms)
        else {
            continue;
        };
        let Some((head_pos, head_world_rot)) = walk_bone_world_pose(head_e, &parents, &transforms)
        else {
            continue;
        };

        // Same anchor math as `apply_tablet_ik` so the head looks at
        // the EXACT tablet centre.
        let hold_offset_b = Vec3::new(
            class.tablet_hold_offset_local.x,
            class.tablet_hold_offset_local.y,
            class.tablet_hold_offset_local.z,
        );
        let tablet_centre = chest_pos + visual_rot * hold_offset_b;

        let solve = solve_aim_chain(AimChainInput {
            neck_pos: bevy_to_g_vec3(neck_pos),
            neck_world_rot: bevy_to_g_quat(neck_world_rot),
            head_pos: bevy_to_g_vec3(head_pos),
            head_world_rot: bevy_to_g_quat(head_world_rot),
            forward_local: class.look_forward_local,
            up_local: class.look_up_local,
            target: bevy_to_g_vec3(tablet_centre),
            neck_share: class.look_neck_share,
            neck_yaw_limit: class.look_neck_yaw_limit,
            neck_pitch_limit: class.look_neck_pitch_limit,
            head_yaw_limit: class.look_head_yaw_limit,
            head_pitch_limit: class.look_head_pitch_limit,
        });

        let neck_delta_full = g_to_bevy_quat(solve.neck_delta_world);
        let head_delta_full = g_to_bevy_quat(solve.head_delta_world);
        let neck_delta = Quat::IDENTITY.slerp(neck_delta_full, blend.weight);
        let head_delta = Quat::IDENTITY.slerp(head_delta_full, blend.weight);

        // Same conjugation pattern as `look_ik::apply_look_ik`: apply
        // neck first; then the head bone's effective world rotation
        // includes the neck delta, so the head conjugation uses the
        // post-neck world rotation.
        if let Ok(mut neck_tf) = transforms.get_mut(neck_e) {
            let delta_local = neck_world_rot.inverse() * neck_delta * neck_world_rot;
            neck_tf.rotation = neck_tf.rotation * delta_local;
        }
        let head_world_rot_after_neck = neck_delta * head_world_rot;
        if let Ok(mut head_tf) = transforms.get_mut(head_e) {
            let delta_local =
                head_world_rot_after_neck.inverse() * head_delta * head_world_rot_after_neck;
            head_tf.rotation = head_tf.rotation * delta_local;
        }
    }
}

fn apply_tablet_ik(
    visuals: Query<(
        Entity,
        &RemoteCharacterTag,
        &BoneRegistry,
        &TabletIkBlendState,
    )>,
    asset_registry: Res<CharacterAssetRegistry>,
    bind_q: Query<&HandBindData>,
    parents: Query<&ChildOf>,
    mut transforms: Query<&mut Transform>,
) {
    /// Below this fade weight the IK contribution is invisible; skip
    /// to save the bone walks + solver math.
    const MIN_WEIGHT: f32 = 0.01;

    for (visual_e, tag, bones, blend) in &visuals {
        if !bones.resolved || blend.weight < MIN_WEIGHT {
            continue;
        }
        let Some(assets) = asset_registry.ready(tag.class_id) else {
            continue;
        };
        let class = assets.class;
        // Walk the visual entity itself for its FRESH world rotation
        // (same staleness reasoning as foot_ik — `GlobalTransform`
        // hasn't propagated this frame). Visual root rotation drives
        // body-direction math; chest bone is only the translation
        // anchor since Mixamo Spine2 has world-aligned bind axes.
        let Some((_, visual_rot)) = walk_bone_world_pose(visual_e, &parents, &transforms) else {
            continue;
        };

        // Resolve every bone we touch up-front. If any are missing
        // (mid-load, future class swaps, …) skip the visual entirely
        // rather than producing a half-IK pose.
        let Some(chest_e) = bones.get(class.chest_bone) else { continue };
        let Some(r_arm_e) = bones.get(class.right_arm_bone) else { continue };
        let Some(r_fore_e) = bones.get(class.right_forearm_bone) else { continue };
        let Some(r_hand_e) = bones.get(class.right_hand_bone) else { continue };
        let Some(l_arm_e) = bones.get(class.left_arm_bone) else { continue };
        let Some(l_fore_e) = bones.get(class.left_forearm_bone) else { continue };
        let Some(l_hand_e) = bones.get(class.left_hand_bone) else { continue };

        // Walk fresh world poses (position + rotation) for every
        // chain joint. Rotation is needed for the world→local
        // delta conjugation; foot_ik uses the same pattern.
        let Some((chest_pos, _)) = walk_bone_world_pose(chest_e, &parents, &transforms) else {
            continue;
        };
        let Some((r_arm_pos, r_arm_world_rot)) =
            walk_bone_world_pose(r_arm_e, &parents, &transforms)
        else {
            continue;
        };
        let Some((r_fore_pos, r_fore_world_rot)) =
            walk_bone_world_pose(r_fore_e, &parents, &transforms)
        else {
            continue;
        };
        let Some((r_hand_pos, _)) = walk_bone_world_pose(r_hand_e, &parents, &transforms) else {
            continue;
        };
        let Some((l_arm_pos, l_arm_world_rot)) =
            walk_bone_world_pose(l_arm_e, &parents, &transforms)
        else {
            continue;
        };
        let Some((l_fore_pos, l_fore_world_rot)) =
            walk_bone_world_pose(l_fore_e, &parents, &transforms)
        else {
            continue;
        };
        let Some((l_hand_pos, l_hand_world_rot)) =
            walk_bone_world_pose(l_hand_e, &parents, &transforms)
        else {
            continue;
        };

        // -- Build the tablet plane in world space ----------------------
        //
        // Body-local axes (in visual root's local frame):
        //   +Z = body forward    -X = body right
        //   +Y = body up         -Z = body back (player's eye direction)
        //
        // `tablet_hold_offset_local` is in this frame; rotating by
        // `visual_rot` lifts it into world.
        let hold_offset_b = Vec3::new(
            class.tablet_hold_offset_local.x,
            class.tablet_hold_offset_local.y,
            class.tablet_hold_offset_local.z,
        );
        let tablet_centre = chest_pos + visual_rot * hold_offset_b;

        let body_right = visual_rot * Vec3::NEG_X;
        let body_up = visual_rot * Vec3::Y;
        let body_back = visual_rot * Vec3::NEG_Z;

        // Pitch around body_right tips the tablet top toward the
        // player. Right-hand rule: thumb along body_right, +ve angle
        // takes body_up → body_back → body_down. Positive class
        // pitch → top tips toward the player's face.
        let pitch_quat = Quat::from_axis_angle(body_right, class.tablet_hold_pitch);
        let tablet_right = body_right;
        let tablet_up = pitch_quat * body_up;
        let tablet_normal = pitch_quat * body_back; // screen faces this way (toward player).

        // Convert UV (centred at 0.5,0.5) to a tablet-local offset in
        // metres. UV.x grows rightward; UV.y grows DOWNWARD (screen
        // convention) → flip Y so the world offset grows upward.
        let uv_to_world = |uv: Vec2| -> Vec3 {
            let u = (uv.x - 0.5) * class.tablet_width;
            let v = -(uv.y - 0.5) * class.tablet_height;
            tablet_centre + tablet_right * u + tablet_up * v
        };

        // Right wrist target — derived directly from where the tablet
        // visual will end up under the parent-to-hand architecture
        // (`hud::tablet::follow_camera` places the tablet at the same
        // body-local offset, inverted). Wrist sits OUTSIDE the right
        // edge of the pad by `tablet_right_hand_outside_offset` so
        // the visible hand wraps the bezel rather than fingers
        // extending ACROSS the screen.
        //
        // hand_world = chest + visual_rot * (
        //     hold_offset.x - W/2 - outside_offset,   // body_right
        //     hold_offset.y,                          // (down/up)
        //     hold_offset.z + grip_back_depth         // body_forward (behind screen)
        // )
        //
        // Tightly coupled to the tablet visual's local Transform —
        // both must move together if either changes.
        let half_w = class.tablet_width / 2.0;
        let right_grip_world = chest_pos
            + visual_rot
                * Vec3::new(
                    class.tablet_hold_offset_local.x
                        - half_w
                        - class.tablet_right_hand_outside_offset,
                    class.tablet_hold_offset_local.y,
                    class.tablet_hold_offset_local.z + class.tablet_grip_back_depth,
                );
        // Suppress unused-warning on the legacy uv path's helpers.
        let _ = (uv_to_world, &class.tablet_right_grip_uv);

        // -- Desired LEFT wrist orientation (derived from rig bind) --
        //
        // The previous implementation hardcoded a per-rig Quat
        // (`left_hand_pointing_rotation_in_body`) which assumed a
        // specific bone-axis convention (Convention C: bone +X = thumb,
        // +Y = finger, +Z = palm-back). Mixamo Y-bot's actual axes
        // disagree with that convention, so the override produced
        // wrong palm orientations.
        //
        // The correct approach reads the bone's BIND world rotation
        // (via `HandBindData`, populated from
        // `SkinnedMeshInverseBindposes`) and derives bone-local axes
        // from data:
        //   * finger axis (bone-local) = the bone-local axis that, at
        //     bind, points body_LEFT (since LEFT hand bone +Y in T-pose
        //     extends body_LEFT). For Mixamo Y-bot this is bone +Y.
        //   * palm-normal axis (bone-local) = the bone-local axis that,
        //     at bind, points body_DOWN (since palms-down T-pose has
        //     palm normal = body_DOWN).
        //   * thumb axis (bone-local) = palm × finger (RH rule).
        //
        // The TARGET pose: palm PARALLEL to the pad surface (the
        // user's explicit request — "way more natural than palm 90
        // degrees to the tablet"). Geometrically:
        //   * Pad surface normal in body local =
        //     `pitch_quat * body_back` (pad rotated by
        //     `tablet_hold_pitch` around body_RIGHT axis from a
        //     vertical "back-facing" rest pose).
        //   * Palm normal aligns with the pad's INTO-surface
        //     direction = `-pad_normal`. This puts the palm flat on
        //     the screen, knuckles facing away from the screen
        //     (toward the player's eye line). The back of the hand
        //     is visible from both FP and TP cameras — exactly how
        //     a real hand looks resting palm-down on a touchscreen.
        //   * Finger lies in the pad surface plane, pointing
        //     body_RIGHT (= INWARD across the body for the LEFT
        //     hand). Wrist sits on the LEFT side of the cursor;
        //     fingers extend ACROSS the pad to the cursor — the
        //     natural reach for a centred LEFT-hand tap with the
        //     RIGHT hand holding the tablet edge.
        //   * Thumb (= palm × finger via RH rule) ends along
        //     `-pad_up` direction (toward the bottom edge of the
        //     screen) — natural anatomy for LEFT hand palm-down
        //     with fingers extended sideways.
        //
        // `tablet_finger_tilt` is unused in this pose since the
        // hand is rigidly aligned with the pad surface; the field
        // is kept for backward compat / future tweaks (e.g., a
        // small wrist-extension offset away from pure flat).
        //
        // Construct R such that R * bone_basis = target_basis →
        // R = target_basis * bone_basis^T (transpose = inverse for
        // orthonormal). Then the world-space override is
        // `visual_rot * R`.
        let _ = class.tablet_finger_tilt; // Reserved for future tweaks; pose is pad-aligned.
        // Body-local axes in mesh basis (visual root local = mesh
        // local for Mixamo): +X = body_LEFT, +Y = body_UP,
        //                    +Z = body_FORWARD. (NB: this codebase
        // uses `body_right = visual_rot * Vec3::NEG_X`, so +X mesh
        // is body_LEFT, NOT body_RIGHT. Verified empirically.)
        let body_right_local = Vec3::NEG_X;
        let body_back_local = Vec3::NEG_Z;
        let body_up_local = Vec3::Y;
        let pitch_quat_local =
            Quat::from_axis_angle(body_right_local, class.tablet_hold_pitch);
        let pad_normal_local = (pitch_quat_local * body_back_local).normalize_or_zero();
        let target_palm_normal = -pad_normal_local;
        // Within the pad plane, two orthogonal axes:
        //   * `body_right_local` — pure body_RIGHT (lies in pad plane
        //     because pad_normal has only YZ components).
        //   * `pad_up_local`     — direction from pad centre to top
        //     edge of the screen.
        // Rotate the finger 45° from body_RIGHT toward pad_up so it
        // angles diagonally up-right on the pad surface — natural
        // tap pose for the LEFT hand reaching across the body.
        let pad_up_local = (pitch_quat_local * body_up_local).normalize_or_zero();
        let finger_pad_angle = std::f32::consts::FRAC_PI_4;
        let target_finger = (finger_pad_angle.cos() * body_right_local
            + finger_pad_angle.sin() * pad_up_local)
            .normalize_or_zero();
        let target_thumb = target_palm_normal.cross(target_finger).normalize_or_zero();

        let bind_l = bind_q.get(visual_e).map(|b| b.left_hand).ok();
        let (target_in_body, finger_local_b) = match bind_l {
            Some(bind_in_body) => {
                let inv = bind_in_body.inverse();
                // CONVENTION: this codebase's body axes (verified
                // against `body_right = visual_rot * Vec3::NEG_X`):
                //   * body_LEFT  = Vec3::X      (mesh +X).
                //   * body_RIGHT = Vec3::NEG_X.
                //   * body_UP    = Vec3::Y.
                //   * body_DOWN  = Vec3::NEG_Y.
                //   * body_FWD   = Vec3::Z.
                //   * body_BACK  = Vec3::NEG_Z.
                //
                // For LEFT hand at palms-down T-pose bind:
                //   * Bone +Y (along bone, toward fingertip) points
                //     body_LEFT = +X mesh = `Vec3::X`.
                //   * Palm normal direction = body_DOWN = `Vec3::NEG_Y`.
                //
                // So the bone-local axis aligned with "finger" =
                // bind.inverse() * body_LEFT_in_body_local =
                // bind.inverse() * Vec3::X.
                let finger_axis_bone = (inv * Vec3::X).normalize_or_zero();
                let palm_normal_axis_bone = (inv * Vec3::NEG_Y).normalize_or_zero();
                let thumb_axis_bone =
                    palm_normal_axis_bone.cross(finger_axis_bone).normalize_or_zero();
                let bone_basis =
                    Mat3::from_cols(thumb_axis_bone, finger_axis_bone, palm_normal_axis_bone);
                let target_basis =
                    Mat3::from_cols(target_thumb, target_finger, target_palm_normal);
                (
                    Quat::from_mat3(&(target_basis * bone_basis.transpose())),
                    finger_axis_bone,
                )
            }
            None => {
                // Fallback during the brief window before
                // `HandBindData` populates: use the legacy hardcoded
                // Quat. Wrong orientation visible for ~1 frame, then
                // the bind-derived path takes over.
                let q = Quat::from_xyzw(
                    class.left_hand_pointing_rotation_in_body.x,
                    class.left_hand_pointing_rotation_in_body.y,
                    class.left_hand_pointing_rotation_in_body.z,
                    class.left_hand_pointing_rotation_in_body.w,
                );
                let local = Vec3::new(
                    class.hand_finger_axis_local.x,
                    class.hand_finger_axis_local.y,
                    class.hand_finger_axis_local.z,
                )
                .normalize_or_zero();
                (q, local)
            }
        };
        let desired_l_hand_world_rot = visual_rot * target_in_body;
        let target_finger_world = (desired_l_hand_world_rot * finger_local_b).normalize_or_zero();
        // `tablet_up` is no longer in the explicit basis — kept on
        // the wire so future tweaks can reference it.
        let _ = tablet_up;

        // Left index finger floats `tablet_finger_lift` metres above
        // the screen so it visually "touches" without z-fighting the
        // mesh. On click, the lift drops to 0 over `tap_press_secs`,
        // then returns over `tap_release_secs` — reading as a touch.
        let lift_factor = match blend.tap_t {
            Some(t) => tap_lift_factor(t, class.tap_press_secs, class.tap_release_secs),
            None => 1.0,
        };
        // World position the FINGERTIP must reach.
        let cursor_world = uv_to_world(blend.last_cursor_uv)
            + tablet_normal * (class.tablet_finger_lift * lift_factor);
        // Wrist target = cursor - (wrist_world_rot *
        // fingertip_offset_in_wrist_local). The fingertip offset is
        // a 3D vector read from the actual rig bind data
        // (`HandBindData::left_index_tip_offset_in_wrist_local`),
        // so the math accounts for the index finger's lateral
        // offset from the wrist bone's +Y axis (Mixamo's wrist
        // bone +Y points along the MIDDLE finger; the index is
        // shifted toward the thumb side). The legacy 1D
        // `index_finger_length` estimate ignored this offset and
        // left the cursor drifting from the visible fingertip.
        // Falls back to the legacy length-based offset only when
        // bind data hasn't populated yet (~1 frame at scene load).
        let fingertip_offset_world = match bind_l {
            Some(_) => {
                let offset_in_wrist = bind_q
                    .get(visual_e)
                    .ok()
                    .map(|b| b.left_index_tip_offset_in_wrist_local)
                    .unwrap_or(Vec3::new(0.0, class.index_finger_length, 0.0));
                desired_l_hand_world_rot * offset_in_wrist
            }
            None => target_finger_world * class.index_finger_length,
        };
        // `tablet_left_hand_outside_offset` adds an extra push of
        // the wrist along the bone +Y direction (away from the
        // cursor) — useful when the natural rig pose puts the wrist
        // inside the pad area's screen-space column and we want
        // the hand body OUTSIDE. Default 0 = no extra push.
        let wrist_outside_offset_world =
            target_finger_world * class.tablet_left_hand_outside_offset;
        let l_wrist_target_world =
            cursor_world - fingertip_offset_world - wrist_outside_offset_world;

        // -- Right arm IK ----------------------------------------------
        //
        // Pole = arm_pos + (body_back + body_down). Elbow tucks
        // BEHIND-AND-BELOW the shoulder, which is the natural pose
        // for holding a tablet at chest height — arms folded in
        // close, forearms reaching forward to the device.
        let pole_dir = body_back + Vec3::NEG_Y;
        let r_pole_world = r_arm_pos + pole_dir;
        apply_two_bone(
            r_arm_pos,
            r_fore_pos,
            r_hand_pos,
            right_grip_world,
            r_pole_world,
            r_arm_e,
            r_fore_e,
            r_arm_world_rot,
            r_fore_world_rot,
            blend.weight,
            &mut transforms,
        );

        // -- Override RIGHT wrist orientation to grip pose -------------
        //
        // Locks the right wrist's WORLD rotation to `visual_rot * grip`,
        // where `grip = class.right_hand_grip_rotation_in_body` rotates
        // the rig's hand-bone-local axes into body-local axes for the
        // tablet hold. With the wrist locked, the tablet (parented to
        // this bone in `hud::tablet::follow_camera`) sits at a known
        // local Transform that puts the screen at chest+offset facing
        // the player — independent of what the 2-bone IK chose for
        // the elbow and shoulder.
        let grip_in_body = right_hand_grip_in_body_rot(class);
        let desired_hand_world_rot = visual_rot * grip_in_body;

        if let Some((_, r_fore_rot_after)) =
            walk_bone_world_pose(r_fore_e, &parents, &transforms)
        {
            let desired_local = r_fore_rot_after.inverse() * desired_hand_world_rot;
            if let Ok(mut hand_tf) = transforms.get_mut(r_hand_e) {
                let blended = hand_tf.rotation.slerp(desired_local, blend.weight);
                hand_tf.rotation = blended;
            }
        }

        // -- Left arm IK ------------------------------------------------
        //
        // Target is the WRIST position that puts the fingertip at
        // the cursor. The wrist orientation is overridden separately
        // below to a fixed body-relative pose, so the user sees the
        // finger pointing at the screen while only the wrist position
        // changes with the cursor.
        let l_pole_world = l_arm_pos + pole_dir;
        apply_two_bone(
            l_arm_pos,
            l_fore_pos,
            l_hand_pos,
            l_wrist_target_world,
            l_pole_world,
            l_arm_e,
            l_fore_e,
            l_arm_world_rot,
            l_fore_world_rot,
            blend.weight,
            &mut transforms,
        );

        // -- Wrist orientation: align finger axis with -tablet_normal --
        //
        // After the 2-bone solve the parent (forearm) has rotated, so
        // the hand's effective world pose is roughly the same delta we
        // applied to the forearm composed with the hand's pre-IK world
        // rotation. We want the hand's local `hand_finger_axis_local`
        // to point along `-tablet_normal` (finger pressed down onto
        // the screen). Compute the post-IK forearm world rotation by
        // walking again — cheap enough at ~22-bone chains, and saves
        // mirroring the math with cached deltas.
        // -- Override LEFT wrist to the fixed pointing pose -----------
        //
        // After the 2-bone IK has rotated the forearm, override the
        // hand's LOCAL rotation so its WORLD rotation equals the
        // pre-computed `desired_l_hand_world_rot`. Slerp toward the
        // override by `blend.weight` so the fade-in is smooth.
        if let Some((_, l_fore_rot_after)) =
            walk_bone_world_pose(l_fore_e, &parents, &transforms)
        {
            let desired_local = l_fore_rot_after.inverse() * desired_l_hand_world_rot;
            if let Ok(mut hand_tf) = transforms.get_mut(l_hand_e) {
                let blended = hand_tf.rotation.slerp(desired_local, blend.weight);
                hand_tf.rotation = blended;
            }
            // Suppress unused-warning when the path isn't reached.
            let _ = l_hand_world_rot;
            let _: &CharacterClass = class;
        }
    }
}

/// Run `solve_two_bone_ik` and write the per-bone deltas back to the
/// LOCAL Transforms via the same conjugation pattern as foot IK
/// (`world_inv * δ_world * world` expresses a world rotation in the
/// bone's local frame). Caller passes pre-IK WORLD rotations so the
/// math is correct for bones nested deep in the rig hierarchy.
#[allow(clippy::too_many_arguments)]
fn apply_two_bone(
    root_pos: Vec3,
    mid_pos: Vec3,
    tip_pos: Vec3,
    target: Vec3,
    pole: Vec3,
    root_e: Entity,
    mid_e: Entity,
    root_world_rot: Quat,
    mid_world_rot: Quat,
    weight: f32,
    transforms: &mut Query<&mut Transform>,
) {
    let solve = solve_two_bone_ik(TwoBoneInput {
        root: bevy_to_g_vec3(root_pos),
        mid: bevy_to_g_vec3(mid_pos),
        tip: bevy_to_g_vec3(tip_pos),
        target: bevy_to_g_vec3(target),
        pole: bevy_to_g_vec3(pole),
    });
    let root_delta_full = g_to_bevy_quat(solve.root_delta_world);
    let mid_delta_full = g_to_bevy_quat(solve.mid_delta_world);
    let root_delta = Quat::IDENTITY.slerp(root_delta_full, weight);
    let mid_delta = Quat::IDENTITY.slerp(mid_delta_full, weight);

    // Apply root (shoulder) first — its world rotation goes from
    // `root_world_rot` to `root_delta * root_world_rot` after this.
    if let Ok(mut t) = transforms.get_mut(root_e) {
        let delta_local = root_world_rot.inverse() * root_delta * root_world_rot;
        t.rotation = t.rotation * delta_local;
    }
    // Mid (elbow) — its world rotation is now influenced by the
    // shoulder's delta. Recompute the post-root world rotation, then
    // conjugate the elbow's delta in that updated frame.
    let mid_world_after_root = root_delta * mid_world_rot;
    if let Ok(mut t) = transforms.get_mut(mid_e) {
        let delta_local =
            mid_world_after_root.inverse() * mid_delta * mid_world_after_root;
        t.rotation = t.rotation * delta_local;
    }
}

/// Same parent-chain walk as look-at / foot IK. Returns the bone's
/// CURRENT animated world pose without touching `GlobalTransform`.
fn walk_bone_world_pose(
    bone: Entity,
    parents: &Query<&ChildOf>,
    transforms: &Query<&mut Transform>,
) -> Option<(Vec3, Quat)> {
    let mut pos = Vec3::ZERO;
    let mut rot = Quat::IDENTITY;
    let mut e = bone;
    loop {
        let Ok(local) = transforms.get(e) else {
            return None;
        };
        pos = local.translation + local.rotation * (local.scale * pos);
        rot = local.rotation * rot;
        match parents.get(e) {
            Ok(child_of) => e = child_of.parent(),
            Err(_) => return Some((pos, rot)),
        }
    }
}

/// Phase J — curl the LEFT hand's non-index fingers into a fist
/// while the player holds a tablet. The index finger stays extended
/// as the cursor; Middle/Ring/Pinky/Thumb segments rotate toward
/// the palm so the hand reads as a "pointing" gesture instead of
/// a flat open palm. Blended by `TabletIkBlendState.weight` so the
/// fist closes/opens with the same easing as the arm IK.
///
/// Curl axis derivation: per-bone, computed from each bone's bind
/// world rotation. For each finger bone, bone +Y points along the
/// bone (= away from the wrist toward the fingertip) at bind. The
/// shortest rotation that takes bone +Y from its bind direction
/// toward body_DOWN (= palm direction at bind) lives around the axis
/// `(bone_y_at_bind_in_body) × body_DOWN`, expressed in body local.
/// Mapping that axis back into bone local via the bone's bind
/// inverse gives the per-bone curl axis without assuming any
/// thumb/palm/finger axis convention.
///
/// Why per-bone (not one shared axis): index/middle/ring/pinky bone
/// +Y all point body_LEFT at bind (along the LEFT arm), so they
/// share a curl axis (body_BACK). The thumb bone +Y at bind points
/// body_FORWARD (since the thumb sticks out in front of the palm in
/// palms-down T-pose), so its curl axis is body_LEFT — totally
/// different. Using one shared axis would bend the thumb sideways /
/// backwards instead of curling it into the palm. Read from
/// [`HandBindData::left_finger_bones`], populated once per visual at
/// scene resolution time.
fn apply_tablet_left_hand_finger_curl(
    visuals: Query<(
        Entity,
        &RemoteCharacterTag,
        &BoneRegistry,
        &TabletIkBlendState,
    )>,
    asset_registry: Res<CharacterAssetRegistry>,
    bind_q: Query<&HandBindData>,
    mut transforms: Query<&mut Transform>,
) {
    /// Skip when the IK contribution is invisible — the bones are
    /// at their bind/animation pose, no curl applied.
    const MIN_WEIGHT: f32 = 0.01;

    for (visual_e, tag, bones, blend) in &visuals {
        if !bones.resolved || blend.weight < MIN_WEIGHT {
            continue;
        }
        let Some(assets) = asset_registry.ready(tag.class_id) else {
            continue;
        };
        let class = assets.class;
        let Ok(bind_data) = bind_q.get(visual_e) else {
            // Bind data not yet populated — skip curl this frame
            // rather than apply with the wrong axis.
            continue;
        };

        // Magnitude of curl per segment (radians). Total across 3
        // segments approximates a closed fist (~170°). Sign comes
        // from the derived curl axis below (which already encodes
        // direction-toward-palm), so we take the magnitude.
        let curl_mag = class.tablet_left_hand_finger_curl_angle.abs() * blend.weight;

        for &name in class.tablet_left_hand_curl_bones {
            let Some(bone_e) = bones.get(name) else {
                continue;
            };
            let Some(bind_in_body) = bind_data.left_finger_bones.get(name) else {
                continue;
            };
            // Per-bone curl axis. For each bone:
            //   1. bone_y_in_body_at_bind = bind * Vec3::Y
            //      (= direction the bone +Y points in body local at
            //      bind: body_LEFT for index/middle/ring/pinky bones,
            //      body_FORWARD for thumb bones).
            //   2. curl_axis_in_body = bone_y × body_DOWN (the
            //      shortest-rotation axis that takes bone +Y toward
            //      the palm direction).
            //   3. curl_axis_in_bone = bind.inverse() *
            //      curl_axis_in_body (bring the rotation axis back
            //      into bone-local frame for axis-angle composition).
            // POSITIVE curl angle = curl toward palm.
            let bone_y_in_body = (*bind_in_body) * Vec3::Y;
            let curl_axis_in_body =
                bone_y_in_body.cross(Vec3::NEG_Y).normalize_or_zero();
            if curl_axis_in_body.length_squared() < 1e-6 {
                // bone +Y already aligned with body_DOWN — no curl.
                continue;
            }
            let curl_axis =
                (bind_in_body.inverse() * curl_axis_in_body).normalize_or_zero();
            if curl_axis.length_squared() < 1e-6 {
                continue;
            }
            let curl_quat = Quat::from_axis_angle(curl_axis, curl_mag);
            let Ok(mut bone_tf) = transforms.get_mut(bone_e) else {
                continue;
            };
            // Right-multiply: applies the curl in the bone's LOCAL
            // frame on top of whatever animation/bind rotation the
            // bone already had.
            bone_tf.rotation = bone_tf.rotation * curl_quat;
        }
    }
}

/// Bridge workspace `glam` 0.29 → bevy `glam` 0.30 for a per-class
/// `right_hand_grip_rotation_in_body` Quat.
pub(crate) fn right_hand_grip_in_body_rot(class: &CharacterClass) -> Quat {
    Quat::from_xyzw(
        class.right_hand_grip_rotation_in_body.x,
        class.right_hand_grip_rotation_in_body.y,
        class.right_hand_grip_rotation_in_body.z,
        class.right_hand_grip_rotation_in_body.w,
    )
}

#[inline]
fn bevy_to_g_vec3(v: Vec3) -> GVec3 {
    GVec3::new(v.x, v.y, v.z)
}

#[inline]
fn bevy_to_g_quat(q: Quat) -> GQuat {
    GQuat::from_xyzw(q.x, q.y, q.z, q.w)
}

#[inline]
fn g_to_bevy_quat(q: GQuat) -> Quat {
    Quat::from_xyzw(q.x, q.y, q.z, q.w)
}

/// Tap-pulse profile: lift fraction in [0, 1] given elapsed `t` since
/// the click. `1.0` = fingertip at full `tablet_finger_lift` above the
/// screen (resting). `0.0` = fingertip touching the screen. Smooth
/// (cosine half-cycle) press from 1 → 0 over `press_secs`, smooth
/// release from 0 → 1 over `release_secs`. Returns 1.0 once the total
/// window has elapsed (the caller should then drop `tap_t` to None).
fn tap_lift_factor(t: f32, press_secs: f32, release_secs: f32) -> f32 {
    let press = press_secs.max(1e-3);
    let release = release_secs.max(1e-3);
    if t < press {
        // Cosine half-cycle: 1 → 0 with zero derivative at the
        // endpoints, so the fingertip eases into AND out of contact
        // rather than snapping. Same easing curve foot IK uses for
        // landing weight.
        let phase = t / press;
        0.5 * (1.0 + (std::f32::consts::PI * phase).cos())
    } else if t < press + release {
        let phase = (t - press) / release;
        0.5 * (1.0 - (std::f32::consts::PI * phase).cos())
    } else {
        1.0
    }
}
