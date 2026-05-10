//! Camera-bone attachment, head/pitch additive layer, third-person
//! toggle, first-person body cull. Phase E of the character plan.
//!
//! # The four pieces
//!
//! 1. **Head additive rotation** ([`apply_head_local_rotation`]):
//!    server-broadcast `head_yaw + head_pitch` is applied as a
//!    *post-clip* rotation on every visible character's head bone, so
//!    other clients see your head turning within your body. Runs in
//!    `PostUpdate` between [`AnimationSystems`] (Bevy applies the clip
//!    pose first) and [`TransformSystems::Propagate`] (so the
//!    additive layer is folded into [`GlobalTransform`] this frame).
//!
//! 2. **Camera follows the local player's head bone**
//!    ([`camera_follow_head`]): once the local visual exists and bones
//!    resolve, [`CameraWorldPos`] is overwritten with the head bone's
//!    world position + class-defined `eye_offset_local`, so the
//!    camera physically rides the animation (subtle breathing,
//!    walk-cycle head bob, eventual look-at IK). Runs in `Update`
//!    *before* [`crate::shard::ShardOriginSet`] so this frame's
//!    floating-origin rebase sees the fresh value.
//!
//! 3. **V-toggle third-person** ([`toggle_camera_mode`]): switches
//!    between [`CameraMode::FirstPerson`] (eye at head bone) and
//!    [`CameraMode::ThirdPerson`] (camera offset behind head bone).
//!    Third-person uses the same head-bone read path; only the
//!    offset vector changes.
//!
//! 4. **First-person body cull** ([`apply_fp_body_cull`]): in
//!    [`CameraMode::FirstPerson`], scales the local visual's head +
//!    clavicles + upper arms to ~zero so the body doesn't intrude
//!    into the FP frame. Restored to identity in `ThirdPerson`.
//!
//! # Data flow / one-frame lag
//!
//! [`camera_follow_head`] reads the head bone's [`GlobalTransform`],
//! which is updated in `PostUpdate` (transform propagation). Reading
//! it from `Update` therefore returns the *previous* frame's value —
//! a ~16 ms lag for the camera position. Acceptable: fast-motion
//! lag is invisible at typical mouse-look speeds, and running the
//! whole loop in `PostUpdate` would feed a stale `CameraWorldPos`
//! into this frame's [`crate::shard::origin::rebase_shard_transforms`]
//! instead, which is more visible.
//!
//! # Coexistence with `crate::camera::pose::apply_worldstate_pose`
//!
//! `pose.rs` writes [`CameraWorldPos`] from the server-broadcast body
//! position + a fixed eye-height fallback. [`camera_follow_head`]
//! runs *after* it (in the same `Update`) and overwrites the value
//! when the local visual is ready. While the visual is still
//! materializing (first second after spawn), `pose.rs`'s value is
//! the one used — preventing a black-screen-during-load flash.

use bevy::app::AnimationSystems;
use bevy::input::keyboard::KeyCode;
use bevy::input::ButtonInput;
use bevy::prelude::*;
use bevy::transform::TransformSystems;

use glam::DVec3;

use crate::net::NetConnection;
use crate::shard::{CameraWorldPos, ShardOriginSet};

use super::assets::CharacterAssetRegistry;
use super::render::{BoneRegistry, RemoteCharacterTag};
use crate::remote::RemotePlayers;

/// Marker on the visual entity belonging to the local player.
/// Inserted by [`promote_local_visual`] when a visual is spawned with
/// `RemoteCharacterTag.player_id == NetConnection.player_id`. Used by
/// the camera-follow path to disambiguate the local from remote
/// visuals.
#[derive(Component, Debug, Default)]
pub struct LocalCharacterTag;

/// Cached pre-FP-cull bone scales, so toggling back to third-person
/// restores the rig exactly. Inserted alongside [`LocalCharacterTag`]
/// the first time the cull runs; absence on a tagged entity means
/// "not yet cull-prepared" and the cull system fills it lazily.
#[derive(Component, Debug, Default)]
struct OriginalBoneScales {
    /// Per-bone original scale, indexed by bone name. We only cache
    /// the bones we actually mutate (head + clavicles + upper arms
    /// per `CharacterClass::fp_cull_bones`).
    by_name: std::collections::HashMap<String, Vec3>,
}

/// Camera perspective mode. Default is [`Self::FirstPerson`]; V key
/// toggles to / from [`Self::ThirdPerson`] with the class-tunable
/// distance.
#[derive(Resource, Clone, Copy, Debug, PartialEq)]
pub enum CameraMode {
    /// Eye at the head bone + class.eye_offset_local.
    FirstPerson,
    /// Camera offset behind the head bone (along its local +Z axis,
    /// since glTF characters face +Z by Bevy convention) by
    /// `distance` metres. Future raycast pull-in will clamp this
    /// when the offset would put the camera inside geometry.
    ThirdPerson { distance: f32 },
}

impl Default for CameraMode {
    fn default() -> Self {
        Self::FirstPerson
    }
}

#[derive(SystemSet, Clone, Eq, PartialEq, Hash, Debug)]
pub struct CharacterCameraSet;

pub struct CharacterCameraPlugin;

impl Plugin for CharacterCameraPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<CameraMode>()
            // Update systems — local-visual tagging, mode toggle, FP
            // body cull, and the camera follow (gated to before the
            // floating-origin rebase reads CameraWorldPos).
            .add_systems(
                Update,
                (
                    promote_local_visual,
                    toggle_camera_mode,
                    apply_fp_body_cull,
                    camera_follow_head.before(ShardOriginSet),
                )
                    .in_set(CharacterCameraSet),
            )
            // PostUpdate: head additive layer between animation pose
            // application and transform propagation. Root-motion
            // stripping runs in the same window so the zeroed XZ of
            // the Hips bone (the rig root) propagates this frame —
            // otherwise walk / run clips that bake in forward
            // translation would shift the visual ahead of the
            // server-authoritative position by Δ per frame, and the
            // camera (anchored to the head bone) would drift in
            // lockstep.
            .add_systems(
                PostUpdate,
                (apply_head_local_rotation, strip_root_motion)
                    .after(AnimationSystems)
                    .before(TransformSystems::Propagate),
            );
    }
}

// ────────────────────────────────────────────────────────────────────
// 1. Local-visual tagging
// ────────────────────────────────────────────────────────────────────

/// Insert [`LocalCharacterTag`] on the visual whose `player_id`
/// matches the local [`NetConnection`]. Idempotent: filters
/// `Without<LocalCharacterTag>` so a tagged entity is never re-tagged.
fn promote_local_visual(
    mut commands: Commands,
    conn: Res<NetConnection>,
    visuals: Query<(Entity, &RemoteCharacterTag), Without<LocalCharacterTag>>,
) {
    if conn.player_id == 0 {
        return; // Pre-JoinResponse — no local id yet.
    }
    for (entity, tag) in &visuals {
        if tag.player_id == conn.player_id {
            commands.entity(entity).insert(LocalCharacterTag);
            info!(
                player_id = tag.player_id,
                "character camera: tagged local visual"
            );
        }
    }
}

// ────────────────────────────────────────────────────────────────────
// 2. Camera mode toggle (V key)
// ────────────────────────────────────────────────────────────────────

/// V key toggles between FirstPerson and ThirdPerson. Distance
/// default of 2.5 m matches AAA convention (close enough to feel
/// connected, far enough to see the body).
fn toggle_camera_mode(keys: Res<ButtonInput<KeyCode>>, mut mode: ResMut<CameraMode>) {
    if !keys.just_pressed(KeyCode::KeyV) {
        return;
    }
    *mode = match *mode {
        CameraMode::FirstPerson => CameraMode::ThirdPerson { distance: 2.5 },
        CameraMode::ThirdPerson { .. } => CameraMode::FirstPerson,
    };
    info!(?mode, "character camera: mode toggled");
}

// ────────────────────────────────────────────────────────────────────
// 3. Head additive rotation (all characters, including local)
// ────────────────────────────────────────────────────────────────────

/// Layer the server-broadcast `head_yaw + head_pitch` on top of the
/// clip-driven head pose so other clients see the head turning within
/// the body. Pure post-multiply means a clip with its own head
/// movement (e.g. an idle that gently sways the head) blends with the
/// look-at on top.
fn apply_head_local_rotation(
    visuals: Query<(&RemoteCharacterTag, &BoneRegistry)>,
    remote_players: Res<RemotePlayers>,
    asset_registry: Res<CharacterAssetRegistry>,
    mut transforms: Query<&mut Transform>,
) {
    for (tag, bones) in &visuals {
        if !bones.resolved {
            continue;
        }
        let Some(remote) = remote_players.by_id.get(&tag.player_id) else {
            continue;
        };
        let Some(assets) = asset_registry.ready(tag.class_id) else {
            continue;
        };
        let class = assets.class;

        let Some(head_bone) = bones.get(class.head_bone) else {
            continue;
        };
        let Ok(mut head_transform) = transforms.get_mut(head_bone) else {
            continue;
        };

        // YXZ Euler (yaw, pitch, roll) so the rotation reads as
        // "yaw the head left/right, then pitch it up/down". Layered
        // on top of the animation by post-multiplication: the bone
        // already holds the clip's local rotation, and we right-
        // multiply by the additive look-at.
        //
        // Per-class sign multipliers handle rigs whose head-bone
        // local frame doesn't match the AAA convention
        // (`head_yaw +ve = look left`, `head_pitch +ve = look up`).
        // Mixamo Y-bot needs `head_pitch_sign = -1`; pure-data fix,
        // no per-rig math in this system.
        let head_rot = Quat::from_euler(
            EulerRot::YXZ,
            remote.head_yaw * class.head_yaw_sign,
            remote.head_pitch * class.head_pitch_sign,
            0.0,
        );
        head_transform.rotation = head_transform.rotation * head_rot;
    }
}

// ────────────────────────────────────────────────────────────────────
// 4. Camera follows the local player's head bone
// ────────────────────────────────────────────────────────────────────

/// Read the local visual's head-bone world transform and overwrite
/// [`CameraWorldPos`] so the camera physically rides the animation +
/// the additive look-at applied above.
///
/// `head_world.translation()` is in *camera-relative* space (the
/// floating-origin rebase shifts every shard's transform relative to
/// the previous frame's [`CameraWorldPos`]). To get the head's
/// system-space position we add the previous-frame `CameraWorldPos`
/// back; that gives a stable result regardless of how far the player
/// has travelled in absolute world space.
fn camera_follow_head(
    local_visual: Query<(&RemoteCharacterTag, &BoneRegistry), With<LocalCharacterTag>>,
    asset_registry: Res<CharacterAssetRegistry>,
    global_transforms: Query<&GlobalTransform>,
    mode: Res<CameraMode>,
    mut camera_world: ResMut<CameraWorldPos>,
) {
    let Ok((tag, bones)) = local_visual.single() else {
        return;
    };
    if !bones.resolved {
        return;
    }
    let Some(assets) = asset_registry.ready(tag.class_id) else {
        return;
    };
    let class = assets.class;

    let Some(head_bone) = bones.get(class.head_bone) else {
        return;
    };
    let Ok(head_world) = global_transforms.get(head_bone) else {
        return;
    };

    // GlobalTransform decomposed: position is camera-relative,
    // rotation is world-space (rebase only shifts translation).
    let head_pos_relative = head_world.translation();
    let head_rot_world = head_world.rotation();

    // FP eye offset is in head-bone local frame; rotate by the
    // head's world rotation to get the world-space delta. The class
    // type uses workspace `glam::Vec3` (0.29) and Bevy uses its
    // vendored 0.30 — same memory layout, but the type-system
    // doesn't know it; cast through tuple.
    let eye_local_bevy = Vec3::new(
        class.eye_offset_local.x,
        class.eye_offset_local.y,
        class.eye_offset_local.z,
    );
    // ThirdPerson offset is along world-space camera-back (head's
    // local +Z, which faces backwards in glTF Y-up convention).
    let offset_world: Vec3 = match *mode {
        CameraMode::FirstPerson => head_rot_world * eye_local_bevy,
        CameraMode::ThirdPerson { distance } => head_rot_world * (Vec3::Z * distance),
    };

    // System-space head position = previous_camera_world + relative.
    // System-space target = head + offset.
    let target_system: DVec3 = camera_world.pos
        + DVec3::new(
            (head_pos_relative.x + offset_world.x) as f64,
            (head_pos_relative.y + offset_world.y) as f64,
            (head_pos_relative.z + offset_world.z) as f64,
        );

    camera_world.pos = target_system;
}

// ────────────────────────────────────────────────────────────────────
// 5. Root-motion stripping
// ────────────────────────────────────────────────────────────────────

/// Zero the root bone's XZ translation each frame after animation
/// has applied the clip. AAA standard for server-authoritative
/// movement: the server is the only source of world translation, the
/// clip drives **only** in-place skeletal motion (foot plant, hip
/// rotation, arm swing). A clip baked with forward root motion
/// (Mixamo's *In Place* checkbox off, or FBX2glTF mistakenly
/// re-introducing it) would otherwise add Δ per frame to the visual,
/// putting the body ahead of the camera + snapping back on every
/// server snapshot — exactly what the user reports when this system
/// is missing.
///
/// The bone's **Y** is preserved so the natural hip bob (subtle
/// up-down during walk, breathing during idle, sit-down lower) still
/// reads through. Sit / fall clips that intentionally translate the
/// hips along Y also work correctly.
///
/// Runs in `PostUpdate` after [`AnimationSystems`] and before
/// [`TransformSystems::Propagate`] so the zero is folded into this
/// frame's [`GlobalTransform`].
fn strip_root_motion(
    visuals: Query<(&RemoteCharacterTag, &BoneRegistry)>,
    asset_registry: Res<CharacterAssetRegistry>,
    mut transforms: Query<&mut Transform>,
) {
    for (tag, bones) in &visuals {
        if !bones.resolved {
            continue;
        }
        let Some(assets) = asset_registry.ready(tag.class_id) else {
            continue;
        };
        let class = assets.class;
        let Some(root_entity) = bones.get(class.root_bone) else {
            continue;
        };
        let Ok(mut t) = transforms.get_mut(root_entity) else {
            continue;
        };
        // Don't trigger Bevy's change-detection if no drift this
        // frame — a clean idle clip won't drive XZ at all.
        if t.translation.x != 0.0 || t.translation.z != 0.0 {
            t.translation.x = 0.0;
            t.translation.z = 0.0;
        }
    }
}

// ────────────────────────────────────────────────────────────────────
// 6. First-person body cull
// ────────────────────────────────────────────────────────────────────

/// Scale the local visual's head + clavicles + upper-arm bones to a
/// near-zero value when in [`CameraMode::FirstPerson`] so the body
/// doesn't intrude into the FP frame, and restore the original scales
/// when toggling back to [`CameraMode::ThirdPerson`].
///
/// AAA's preferred path is per-bone-mask shader culling, which
/// requires a custom material or a per-vertex bone-mask attribute.
/// The "scale to ~0" trick is not perfectly clean (collapsed
/// triangles still consume a vertex draw) but produces visually
/// indistinguishable results in FP and is a one-tick `Transform.scale`
/// write — minimal coupling, easy to swap for a shader path later.
fn apply_fp_body_cull(
    mut commands: Commands,
    mode: Res<CameraMode>,
    asset_registry: Res<CharacterAssetRegistry>,
    mut local: Query<
        (Entity, &RemoteCharacterTag, &BoneRegistry, Option<&mut OriginalBoneScales>),
        With<LocalCharacterTag>,
    >,
    mut transforms: Query<&mut Transform>,
) {
    /// Scale value used to "cull" a bone in FP. Not exactly zero —
    /// some Bevy paths divide by scale, so a true zero would NaN.
    /// 1e-4 collapses the bone's weighted vertices to a sub-pixel
    /// point (~0.1 mm at typical view distance) — invisible in
    /// practice.
    const FP_CULL_SCALE: Vec3 = Vec3::splat(1.0e-4);

    let Ok((entity, tag, bones, cached)) = local.single_mut() else {
        return;
    };
    if !bones.resolved {
        return;
    }
    let Some(assets) = asset_registry.ready(tag.class_id) else {
        return;
    };
    let class = assets.class;

    // Lazy-initialize the cache the first frame the local visual is
    // ready. Stores each cull-bone's original scale so toggling back
    // to ThirdPerson restores the rig exactly.
    let mut cache = match cached {
        Some(c) => c,
        None => {
            let mut by_name = std::collections::HashMap::new();
            for &name in class.fp_cull_bones {
                if let Some(bone) = bones.get(name) {
                    if let Ok(t) = transforms.get(bone) {
                        by_name.insert(name.to_string(), t.scale);
                    }
                }
            }
            commands
                .entity(entity)
                .insert(OriginalBoneScales { by_name });
            return; // Apply on the next frame once the cache is in place.
        }
    };

    let target_in_fp = matches!(*mode, CameraMode::FirstPerson);
    for &name in class.fp_cull_bones {
        let Some(bone) = bones.get(name) else { continue };
        let Ok(mut t) = transforms.get_mut(bone) else {
            continue;
        };
        let want = if target_in_fp {
            FP_CULL_SCALE
        } else {
            cache
                .by_name
                .get(name)
                .copied()
                .unwrap_or(Vec3::ONE)
        };
        if t.scale != want {
            t.scale = want;
        }
    }
    // Touch the cache so Bevy doesn't drop the &mut without commits.
    let _ = &mut cache;
}
