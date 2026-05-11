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

use glam::{DVec3, IVec3 as GIVec3, Vec3 as GVec3};

use voxeldust_core::block::{
    chunk_storage::ChunkStorage, raycast as core_raycast, ship_grid::ShipGrid,
};

use crate::chunk::{stream::SharedBlockRegistry, ChunkStorageCache};
use crate::net::NetConnection;
use crate::shard::{CameraWorldPos, ShardOrigin, ShardOriginSet};

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

/// Smooth-blend state for the camera-mode transition. Lerps from 0
/// (FP eye position) to 1 (TP behind-head position) when the user
/// toggles modes, instead of snapping. Lives as a [`Resource`] (one
/// camera per client) so the data flow is plain and the smoothing
/// constants come from `CharacterClass`.
#[derive(Resource, Default, Debug)]
pub struct CameraModeBlend {
    /// Currently rendered blend weight: 0 = pure FP, 1 = pure TP.
    pub current: f32,
    /// Where `current` is heading. Updated by [`toggle_camera_mode`]
    /// (or any future code that flips [`CameraMode`]).
    pub target: f32,
}

/// Per-camera low-pass state for the wall-collision-clamped TP
/// distance. Init to NaN so the first valid raw distance seeds the
/// filter without a transient jump.
#[derive(Resource, Debug)]
pub struct TpDistanceSmoothed {
    pub distance: f32,
}
impl Default for TpDistanceSmoothed {
    fn default() -> Self {
        Self { distance: f32::NAN }
    }
}

#[derive(SystemSet, Clone, Eq, PartialEq, Hash, Debug)]
pub struct CharacterCameraSet;

pub struct CharacterCameraPlugin;

impl Plugin for CharacterCameraPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<CameraMode>()
            .init_resource::<CameraModeBlend>()
            .init_resource::<TpDistanceSmoothed>()
            // `camera_follow_head` runs AFTER `PlayerSyncSet` so it
            // sees the FP eye position `apply_worldstate_pose` just
            // wrote and ADDITIVELY adds the TP body-back offset on
            // top. In FP mode the additive offset is zero, so the
            // value pose.rs wrote stays as-is (no math, no drift,
            // no GlobalTransform read — the FP path is unchanged).
            // In TP mode the offset is `body_back * tp_distance *
            // blend.current`, smoothly animating the camera out
            // behind the player as V toggles.
            .add_systems(
                Update,
                (
                    promote_local_visual,
                    toggle_camera_mode,
                    apply_fp_body_cull,
                    camera_follow_head
                        .after(crate::camera::PlayerSyncSet)
                        .before(ShardOriginSet),
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

/// V key toggles between FirstPerson and ThirdPerson. Distance is
/// the per-class TP convention (`tp_camera_distance` in
/// [`voxeldust_core::character::CharacterClass`]); the actual
/// rendered distance is wall-collision-clamped each frame in
/// [`camera_follow_head`].
fn toggle_camera_mode(
    keys: Res<ButtonInput<KeyCode>>,
    mut mode: ResMut<CameraMode>,
    mut blend: ResMut<CameraModeBlend>,
    asset_registry: Res<CharacterAssetRegistry>,
) {
    if !keys.just_pressed(KeyCode::KeyV) {
        return;
    }
    let tp_distance = asset_registry
        .ready(0)
        .map(|a| a.class.tp_camera_distance)
        .unwrap_or(2.5);
    *mode = match *mode {
        CameraMode::FirstPerson => CameraMode::ThirdPerson { distance: tp_distance },
        CameraMode::ThirdPerson { .. } => CameraMode::FirstPerson,
    };
    blend.target = match *mode {
        CameraMode::FirstPerson => 0.0,
        CameraMode::ThirdPerson { .. } => 1.0,
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
    local_visual: Query<&RemoteCharacterTag, With<LocalCharacterTag>>,
    asset_registry: Res<CharacterAssetRegistry>,
    mode: Res<CameraMode>,
    mut camera_world: ResMut<CameraWorldPos>,
    time: Res<Time>,
    mut blend: ResMut<CameraModeBlend>,
    mut tp_smoothed: ResMut<TpDistanceSmoothed>,
    cam_q: Query<&Transform, With<crate::MainCamera>>,
    remote_players: Res<RemotePlayers>,
    sources: Res<crate::shard::SourceIndex>,
    shard_origins: Query<&ShardOrigin>,
    chunk_storage: Res<ChunkStorageCache>,
    block_registry: Res<SharedBlockRegistry>,
) {
    let Ok(tag) = local_visual.single() else {
        return;
    };
    let Some(assets) = asset_registry.ready(0) else {
        return;
    };
    let class = assets.class;

    // -- Advance the FP↔TP smoothing blend ---------------------------
    blend.target = match *mode {
        CameraMode::FirstPerson => 0.0,
        CameraMode::ThirdPerson { .. } => 1.0,
    };
    let dt = time.delta().as_secs_f32().min(0.25);
    let blend_secs = class.camera_mode_blend_secs.max(1e-3);
    let step = dt / blend_secs;
    let delta = (blend.target - blend.current).clamp(-step, step);
    blend.current = (blend.current + delta).clamp(0.0, 1.0);

    // FP — pose.rs already wrote `camera_world.pos = body + EYE`.
    // We do nothing, leaving the value untouched. No GlobalTransform
    // read, no math, no per-frame jitter on the rebase base. This is
    // exactly the FP behaviour confirmed working without blink.
    if blend.current < 1e-3 {
        return;
    }

    // TP (or transitioning). Camera orbits around the body anchor
    // along the freshly-applied camera direction (`pose.rs` ran in
    // `PlayerSyncSet` just before us, so `MainCamera.rotation` is
    // current). Bevy cameras look down -Z, so `rotation * +Z` is
    // world-space camera-back.
    let Ok(cam_tf) = cam_q.single() else {
        return;
    };
    let cam_back_world = cam_tf.rotation * Vec3::Z;
    let raw_tp_distance = match *mode {
        CameraMode::ThirdPerson { distance } => distance,
        CameraMode::FirstPerson => class.tp_camera_distance,
    };

    // Wall-collision pull-in. RAYCAST FROM THE BODY ANCHOR
    // (`camera_world.pos`, which is the server-authoritative
    // `body + EYE_HEIGHT` written by `apply_worldstate_pose`) — NOT
    // from the head bone. The head bone bobs sub-block per frame
    // due to the idle / walk animation, which previously caused the
    // raycast start cell to flip between solid and air, blinking
    // walls. The body anchor is stable per server tick.
    let raw_clamped = compute_wall_clamped_distance(
        tag.player_id,
        camera_world.pos,
        cam_back_world,
        raw_tp_distance,
        class,
        &remote_players,
        &sources,
        &shard_origins,
        &chunk_storage,
        &block_registry.0,
    );
    // Low-pass smoothing on the clamped distance to bridge any
    // residual frame-to-frame raycast variation (e.g., edge of a
    // doorway sweeping in/out as the camera orbits past).
    let tp_distance = if tp_smoothed.distance.is_nan() {
        tp_smoothed.distance = raw_clamped;
        raw_clamped
    } else {
        let alpha = (dt / class.tp_distance_smoothing_secs.max(1e-3)).clamp(0.0, 1.0);
        tp_smoothed.distance = tp_smoothed.distance * (1.0 - alpha) + raw_clamped * alpha;
        tp_smoothed.distance
    };

    let blended_offset = cam_back_world * (tp_distance * blend.current);
    camera_world.pos += DVec3::new(
        blended_offset.x as f64,
        blended_offset.y as f64,
        blended_offset.z as f64,
    );
}

/// Wall-collision-clamped TP camera distance. Raycasts from
/// `anchor_world` (the body+EYE position, system-space) along
/// `direction_world` (the camera-back direction in world). Returns
/// `raw_distance` when there's no shard info or no obstruction.
#[allow(clippy::too_many_arguments)]
fn compute_wall_clamped_distance(
    player_id: u64,
    anchor_world: DVec3,
    direction_world: Vec3,
    raw_distance: f32,
    class: &voxeldust_core::character::CharacterClass,
    remote_players: &RemotePlayers,
    sources: &crate::shard::SourceIndex,
    shard_origins: &Query<&ShardOrigin>,
    chunk_storage: &ChunkStorageCache,
    block_registry: &voxeldust_core::block::registry::BlockRegistry,
) -> f32 {
    let Some(remote) = remote_players.by_id.get(&player_id) else {
        return raw_distance;
    };
    let Some(&source_entity) = sources.by_shard.get(&remote.shard) else {
        return raw_distance;
    };
    let Ok(shard_origin) = shard_origins.get(source_entity) else {
        return raw_distance;
    };

    // Anchor → shard-local block coords.
    let anchor_local_d = shard_origin.rotation.inverse() * (anchor_world - shard_origin.origin);
    let anchor_local = GVec3::new(
        anchor_local_d.x as f32,
        anchor_local_d.y as f32,
        anchor_local_d.z as f32,
    );

    // Direction (world) → shard-local.
    let dir_world_d = DVec3::new(
        direction_world.x as f64,
        direction_world.y as f64,
        direction_world.z as f64,
    );
    let dir_local_d = shard_origin.rotation.inverse() * dir_world_d;
    let dir_local = GVec3::new(
        dir_local_d.x as f32,
        dir_local_d.y as f32,
        dir_local_d.z as f32,
    )
    .normalize_or_zero();
    if dir_local.length_squared() < 0.5 {
        return raw_distance;
    }

    let shard = remote.shard;
    let hit = core_raycast::raycast(anchor_local, dir_local, raw_distance, |x, y, z| {
        let (chunk_key, lx, ly, lz) = ShipGrid::world_to_chunk(x, y, z);
        chunk_storage
            .get(shard, to_bevy_ivec3(chunk_key))
            .map(|c: &ChunkStorage| block_registry.is_solid(c.get_block(lx, ly, lz)))
            .unwrap_or(false)
    });
    if let Some(hit) = hit {
        // `face_normal == ZERO` = raycast started inside a solid
        // block (anchor inside a wall, e.g. server position
        // momentarily clipped through during transition). Treat as
        // no-hit so we don't snap-zoom into the player's face.
        if hit.face_normal == GIVec3::ZERO {
            return raw_distance;
        }
        (hit.distance - class.tp_camera_collision_buffer)
            .max(class.tp_camera_min_distance)
            .min(raw_distance)
    } else {
        raw_distance
    }
}

#[inline]
fn to_bevy_ivec3(v: GIVec3) -> bevy::math::IVec3 {
    bevy::math::IVec3::new(v.x, v.y, v.z)
}

// Wall-collision raycast deliberately not implemented at this stage.
// The previous attempts caused per-frame "transparent walls" because
// the head bone's animated Y position straddled voxel boundaries and
// the raycast result oscillated. We'll re-add it once we have a
// stable TP baseline confirmed and can sample multiple raycasts (toe
// + heel + horizontal sweep) for a robust pull-in that doesn't
// flicker.

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
