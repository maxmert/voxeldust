//! Foot IK — feet plant on terrain instead of floating / clipping.
//!
//! Phase G of the character system plan. Reads each foot's
//! animated world position, raycasts straight down against the
//! shard's [`ChunkStorageCache`] (the same DDA voxel raycast the
//! block-targeting system uses), and adjusts the leg via the
//! 2-bone analytical solver in `voxeldust_core::character::ik` so
//! the sole lands on the actual block surface.
//!
//! # When IK is active
//!
//! Only the [`LocomotionState::Grounded`] branch — airborne /
//! seated / ragdoll states leave the legs in pure animation pose.
//! Within Grounded the IK applies regardless of speed (idle, walk,
//! run, turn-in-place) so even the idle-on-stairs case looks
//! correct.
//!
//! # Per-foot pipeline (per-frame, per-character)
//!
//! 1. Read the ankle bone's CURRENT animated world position by
//!    walking up the parent chain and composing local Transforms
//!    (see [`walk_bone_world_pose`]). `GlobalTransform` is unsafe
//!    here because Bevy propagates AFTER this system, so reading it
//!    would yield the previous frame's IK output and the IK would
//!    fight itself (foot oscillates at half framerate, looks blurry).
//! 2. Convert to shard-local coords by subtracting the active
//!    shard's `ShardOrigin` (in system-space) and adding the
//!    previous-frame [`CameraWorldPos`] back.
//! 3. DDA raycast straight down (`Vec3::NEG_Y`) up to
//!    `class.foot_ik_max_descent` metres against the chunk cache.
//! 4. If hit: target world Y = (hit block top) +
//!    `class.foot_ik_sole_offset`. If no hit (foot in mid-air over a
//!    deep gap or on the edge of streamed chunks) skip — leg stays
//!    in animation pose.
//! 5. Solve 2-bone IK on hip → knee → ankle. Pole = character
//!    forward direction (from `body_yaw`) so knees bend forward.
//! 6. Apply the world-space rotation deltas to the LOCAL `Transform`
//!    of the hip and knee bones (cascading: hip's delta affects
//!    knee's effective parent rotation, the solver pre-accounts for
//!    that).
//!
//! # Skipping during airborne / pelvis-drop / blend-in/out
//!
//! Phase G v1 skips during Airborne / Seated / Ragdoll and applies
//! IK at full strength when in Grounded. A future polish pass adds:
//!  - Pelvis Y drop when both feet must descend more than the leg
//!    can stretch (going down deep stairs).
//!  - Smooth blend-in / blend-out (~150 ms each way) on state
//!    transitions to avoid pop on jump landing.
//!
//! For now the natural "knee bends more / less" is enough — the
//! v1 visual already nails the common cases (flat floor, single
//! steps, slopes within 0–30°).

use bevy::app::AnimationSystems;
use bevy::prelude::*;
use bevy::transform::TransformSystems;

use glam::{DVec3, IVec3 as GIVec3};
// Workspace glam (0.29) types for the IK solver — Bevy ships its
// own glam (0.30); same memory layout, but the type system treats
// them as distinct, so the boundary here converts component-wise.
use glam::{Quat as GQuat, Vec3 as GVec3};

use voxeldust_core::block::{
    chunk_storage::ChunkStorage, raycast as core_raycast, registry::BlockRegistry,
    ship_grid::ShipGrid, BlockId,
};
use voxeldust_core::character::{
    ik::{solve_two_bone_ik, TwoBoneInput},
    LocomotionState,
};

use crate::chunk::{stream::SharedBlockRegistry, ChunkStorageCache};
use crate::remote::RemotePlayers;
use crate::shard::{CameraWorldPos, ShardOrigin};

use super::assets::CharacterAssetRegistry;
use super::render::{BoneRegistry, RemoteCharacterTag};

#[derive(SystemSet, Clone, Eq, PartialEq, Hash, Debug)]
pub struct CharacterFootIkSet;

/// Per-character fade strength for foot IK. Ramps to 1 when the
/// character enters [`LocomotionState::Grounded`] and to 0 when it
/// leaves (jump, ragdoll, ...). Multiplied with the per-foot lift
/// weight inside `apply_one_leg`. AAA convention: foot IK fades IN
/// gradually after a landing (looks like the foot "settles") and
/// OUT quickly on lift-off (no ghost correction in the air).
#[derive(Component, Debug, Default)]
pub struct FootIkBlendState {
    /// 0..1. Driven each frame toward `target_weight` by
    /// `update_foot_blend`.
    pub weight: f32,
}

pub struct CharacterFootIkPlugin;

impl Plugin for CharacterFootIkPlugin {
    fn build(&self, app: &mut App) {
        // Blend update runs in the EARLIER `Update` schedule so by
        // the time the IK system samples the weight in `PostUpdate`
        // it reflects this tick's locomotion state.
        app.add_systems(Update, update_foot_blend);
        // Run after AnimationSystems (so we layer on the clip-driven
        // pose) and before TransformSystems::Propagate (so the IK'd
        // local rotations fold into this frame's GlobalTransform).
        // The world positions we read are 1-frame stale; acceptable
        // for foot planting (terrain is static at the timescale of
        // 16 ms).
        app.add_systems(
            PostUpdate,
            apply_foot_ik
                .after(AnimationSystems)
                .before(TransformSystems::Propagate)
                .in_set(CharacterFootIkSet),
        );
    }
}

/// Drives the per-character `FootIkBlendState` toward the target
/// weight implied by the locomotion state. Ensures any visual that
/// has a `BoneRegistry` (i.e. is ready for IK) also has the blend
/// component, lazily inserting it on the first tick.
fn update_foot_blend(
    mut commands: Commands,
    time: Res<Time>,
    asset_registry: Res<CharacterAssetRegistry>,
    remote_players: Res<RemotePlayers>,
    mut visuals: Query<
        (Entity, &RemoteCharacterTag, Option<&mut FootIkBlendState>),
    >,
) {
    let dt = time.delta().as_secs_f32().min(0.25);
    for (entity, tag, blend_opt) in &mut visuals {
        let Some(remote) = remote_players.by_id.get(&tag.player_id) else {
            continue;
        };
        let Some(assets) = asset_registry.ready(tag.class_id) else {
            continue;
        };
        let class = assets.class;
        let target = if matches!(
            LocomotionState::from_u8(remote.locomotion),
            LocomotionState::Grounded
        ) {
            1.0
        } else {
            0.0
        };
        match blend_opt {
            Some(mut blend) => {
                advance_blend(
                    &mut blend.weight,
                    target,
                    dt,
                    class.foot_blend_in_secs,
                    class.foot_blend_out_secs,
                );
            }
            None => {
                commands
                    .entity(entity)
                    .insert(FootIkBlendState { weight: target });
            }
        }
    }
}

/// Lerp `weight` toward `target` by `dt` at a per-direction rate.
/// Separate in/out rates let us shape the easing — fast disengage,
/// slow reattach, etc.
pub(super) fn advance_blend(
    weight: &mut f32,
    target: f32,
    dt: f32,
    blend_in_secs: f32,
    blend_out_secs: f32,
) {
    let going_up = target > *weight;
    let denom = if going_up { blend_in_secs } else { blend_out_secs };
    let step = if denom <= 1e-6 {
        target - *weight
    } else {
        (target - *weight).clamp(-dt / denom, dt / denom)
    };
    *weight = (*weight + step).clamp(0.0, 1.0);
}

fn apply_foot_ik(
    visuals: Query<(
        Entity,
        &RemoteCharacterTag,
        &BoneRegistry,
        Option<&FootIkBlendState>,
    )>,
    remote_players: Res<RemotePlayers>,
    asset_registry: Res<CharacterAssetRegistry>,
    chunk_storage: Res<ChunkStorageCache>,
    camera_world: Res<CameraWorldPos>,
    shard_origins: Query<&ShardOrigin>,
    sources: Res<crate::shard::SourceIndex>,
    block_registry: Res<SharedBlockRegistry>,
    parents: Query<&ChildOf>,
    mut transforms: Query<&mut Transform>,
) {

    for (visual_e, tag, bones, blend) in &visuals {
        if !bones.resolved {
            continue;
        }
        let Some(remote) = remote_players.by_id.get(&tag.player_id) else {
            continue;
        };
        // The blend weight smoothly fades IK in and out at state
        // transitions (see `update_foot_blend`). When a character is
        // mid-jump and still partially planted, this avoids a hard
        // pop. Once weight reaches zero we can skip the per-leg work.
        let blend_weight = blend.map(|b| b.weight).unwrap_or(0.0);
        if blend_weight < 0.01 {
            continue;
        }
        let Some(assets) = asset_registry.ready(tag.class_id) else {
            continue;
        };
        let class = assets.class;

        // Resolve the active shard's origin so we can convert bone
        // world positions ↔ shard-local block coords for the raycast.
        let Some(&source_entity) = sources.by_shard.get(&remote.shard) else {
            continue;
        };
        let Ok(shard_origin) = shard_origins.get(source_entity) else {
            continue;
        };

        for foot_side in [
            FootSide {
                hip: class.left_hip_bone,
                knee: class.left_knee_bone,
                ankle: class.left_foot_bone,
            },
            FootSide {
                hip: class.right_hip_bone,
                knee: class.right_knee_bone,
                ankle: class.right_foot_bone,
            },
        ] {
            apply_one_leg(
                &foot_side,
                visual_e,
                remote.locomotion_speed,
                blend_weight,
                bones,
                remote.shard,
                shard_origin,
                &camera_world,
                &chunk_storage,
                &block_registry.0,
                class,
                &parents,
                &mut transforms,
            );
        }
    }
}

/// Walk the parent chain from a bone up to its root entity, composing
/// **current** local Transforms to recover the bone's world-space pose.
///
/// Why not use [`GlobalTransform`]? Because Bevy's transform
/// propagation runs at the END of `PostUpdate` (= [`TransformSystems::Propagate`]),
/// so any `GlobalTransform` we read in our PostUpdate IK system is
/// from the PREVIOUS frame — and that value reflects the previous
/// frame's IK output, not the current frame's animation pose.
/// Reading stale GlobalTransform creates a feedback loop: the IK
/// "thinks" the foot is already on target, applies zero delta, the
/// next frame the bone snaps back to bind pose, IK applies full
/// delta again — visible as the foot oscillating at half framerate.
///
/// Walking the local Transforms directly gives the AAA-correct
/// FRESH pose: AnimationSystems already wrote this frame's clip
/// pose into each bone's Transform; we just compose them ourselves.
/// O(depth) per call (~5 ancestors for a Mixamo ankle); cheap.
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
        // pos in parent's frame = local.translation + local.rotation * (local.scale * pos)
        // rot in parent's frame = local.rotation * rot
        pos = local.translation + local.rotation * (local.scale * pos);
        rot = local.rotation * rot;
        match parents.get(e) {
            Ok(child_of) => e = child_of.parent(),
            Err(_) => return Some((pos, rot)),
        }
    }
}

struct FootSide {
    hip: &'static str,
    knee: &'static str,
    ankle: &'static str,
}

fn apply_one_leg(
    side: &FootSide,
    visual_e: Entity,
    locomotion_speed: f32,
    state_blend_weight: f32,
    bones: &BoneRegistry,
    shard: crate::shard::ShardKey,
    shard_origin: &ShardOrigin,
    camera_world: &CameraWorldPos,
    chunk_storage: &ChunkStorageCache,
    block_registry: &BlockRegistry,
    class: &voxeldust_core::character::CharacterClass,
    parents: &Query<&ChildOf>,
    transforms: &mut Query<&mut Transform>,
) {
    let (Some(hip_e), Some(knee_e), Some(ankle_e)) = (
        bones.get(side.hip),
        bones.get(side.knee),
        bones.get(side.ankle),
    ) else {
        return;
    };
    // FRESH world pose via manual hierarchy walk — see
    // `walk_bone_world_pose` for why GlobalTransform is unsafe here.
    let (Some((hip_w_bevy, hip_world_old)), Some((knee_w_bevy, knee_world_old)), Some((ankle_w_bevy, _))) = (
        walk_bone_world_pose(hip_e, parents, transforms),
        walk_bone_world_pose(knee_e, parents, transforms),
        walk_bone_world_pose(ankle_e, parents, transforms),
    ) else {
        return;
    };
    // Visual entity's world pose — its forward direction is the IK
    // pole (knee bends toward forward). Without an explicit pole the
    // solver's geometric fallback collapses when the leg is vertical
    // (hip directly above ankle), which is most of the time, and the
    // knee can flip backwards. Mixamo bind has the mesh facing +Z,
    // so visual_world_rot * Vec3::Z is the character's forward.
    let Some((_, visual_world_rot)) = walk_bone_world_pose(visual_e, parents, transforms) else {
        return;
    };
    let visual_forward = bevy_to_g_vec3(visual_world_rot * Vec3::Z);
    let hip_w = bevy_to_g_vec3(hip_w_bevy);
    let knee_w = bevy_to_g_vec3(knee_w_bevy);
    let ankle_w = bevy_to_g_vec3(ankle_w_bevy);

    // Convert ankle to shard-local for the raycast. shard_local =
    // (camera_world + bevy_relative) - shard_origin, with rotation
    // factored out (shard.rotation rotates shard-local → system).
    let ankle_system: DVec3 = camera_world.pos
        + DVec3::new(ankle_w_bevy.x as f64, ankle_w_bevy.y as f64, ankle_w_bevy.z as f64);
    let ankle_local_d: DVec3 = shard_origin.rotation.inverse() * (ankle_system - shard_origin.origin);
    let ankle_local = GVec3::new(
        ankle_local_d.x as f32,
        ankle_local_d.y as f32,
        ankle_local_d.z as f32,
    );

    // Raycast direction in shard-local: straight down. The shard's
    // rotation may not have +Y as world-up (e.g. rolled ship), so we
    // rotate world -Y into shard-local frame.
    let world_down = DVec3::new(0.0, -1.0, 0.0);
    let local_down_d = shard_origin.rotation.inverse() * world_down;
    let local_down = GVec3::new(
        local_down_d.x as f32,
        local_down_d.y as f32,
        local_down_d.z as f32,
    );

    // Begin the raycast from `max_ascent` ABOVE the ankle so we can
    // catch ground that's higher than the animated ankle (steps).
    let ray_origin = ankle_local - local_down * class.foot_ik_max_ascent;
    let max_dist = class.foot_ik_max_descent + class.foot_ik_max_ascent;

    let hit = core_raycast::raycast(ray_origin, local_down, max_dist, |x, y, z| {
        let (chunk_key, lx, ly, lz) = ShipGrid::world_to_chunk(x, y, z);
        chunk_storage
            .get(shard, to_bevy_ivec3(chunk_key))
            .map(|c: &ChunkStorage| block_registry.is_solid(c.get_block(lx, ly, lz)))
            .unwrap_or(false)
    });
    let Some(hit) = hit else {
        return; // No ground in range — leg stays animated.
    };

    // Reject non-floor hits (walls, ceilings, steep slopes). Generic
    // for any shard orientation: dot the block face normal against
    // the local "up" vector (=−`local_down`), require ≥ cos(45°).
    // Without this the foot would try to plant on a wall when the
    // character is up against geometry — visually as if climbing.
    let local_up = -local_down;
    let face_normal_g = GVec3::new(
        hit.face_normal.x as f32,
        hit.face_normal.y as f32,
        hit.face_normal.z as f32,
    );
    if face_normal_g.dot(local_up) < 0.7 {
        return;
    }

    // Surface is the +Y face of the hit block (block_pos.y + 1) since
    // we already filtered to floor-aligned hits.
    let surface_y_local = (hit.world_pos.y + 1) as f32;

    // Foot target: sole on surface, in shard-local coords.
    let target_local = GVec3::new(
        ankle_local.x,
        surface_y_local + class.foot_ik_sole_offset,
        ankle_local.z,
    );

    // ---- Plant / lift detection ----
    //
    // During the LIFT phase of the walk cycle the animated ankle is
    // significantly above the planted target. If we let IK run at
    // full strength then, it would yank the lifted foot back down to
    // the floor — that requires straightening the leg, which makes
    // the walk look like a pendulum on stilts (the user's "weird
    // strait-legs walk" symptom).
    //
    // We measure lift in the shard's local up direction (so this
    // generalises beyond Y-up planets to ships and rotated worlds)
    // and blend IK off as the foot rises:
    //  - lift ≤ PLANT_LIFT  → full IK (foot is planted)
    //  - lift ≥ STRIDE_LIFT → no   IK (foot is mid-stride)
    //  - between            → linear blend
    //
    // For static IDLE we always apply full IK regardless of measured
    // lift, because any class-level visual offset error shows up as a
    // constant non-zero "lift" and we still want the foot planted.
    const PLANT_LIFT: f32 = 0.05;
    const STRIDE_LIFT: f32 = 0.30;
    const SPEED_IDLE_THRESHOLD: f32 = 0.5;
    let raw_lift = (ankle_local.y - target_local.y).max(0.0);
    let lift_weight = if locomotion_speed < SPEED_IDLE_THRESHOLD {
        1.0
    } else if raw_lift <= PLANT_LIFT {
        1.0
    } else if raw_lift >= STRIDE_LIFT {
        0.0
    } else {
        1.0 - (raw_lift - PLANT_LIFT) / (STRIDE_LIFT - PLANT_LIFT)
    };
    // Final IK weight composes the per-foot lift detection (per-step
    // anti-leg-straightening) with the per-character state blend
    // (jump-landing fade-in / lift-off fade-out).
    let ik_weight = lift_weight * state_blend_weight;
    if ik_weight < 0.01 {
        return;
    }

    // Convert target back to BEVY camera-relative for the IK input
    // (so it's in the same frame as hip / knee / ankle world).
    let target_system_d: DVec3 = shard_origin.origin
        + shard_origin.rotation
            * DVec3::new(target_local.x as f64, target_local.y as f64, target_local.z as f64);
    let target_world = GVec3::new(
        (target_system_d.x - camera_world.pos.x) as f32,
        (target_system_d.y - camera_world.pos.y) as f32,
        (target_system_d.z - camera_world.pos.z) as f32,
    );

    // Pole = visual entity's forward direction in world space. Knee
    // bends toward the pole; using "character forward" guarantees a
    // natural forward-bending knee regardless of leg geometry. The
    // earlier `Y × (target − hip) × …` derivation was degenerate when
    // the leg was vertical (target directly below hip = the common
    // case), producing a near-zero pole and unstable knee direction
    // — that was the user's "knee sags backwards near walls" bug.
    let pole_world = hip_w + visual_forward;

    let solve = solve_two_bone_ik(TwoBoneInput {
        root: hip_w,
        mid: knee_w,
        tip: ankle_w,
        target: target_world,
        pole: pole_world,
    });

    // Apply the world-space deltas to the LOCAL bone Transforms,
    // SCALED by `ik_weight` so the IK fades smoothly off during the
    // lift phase of the walk cycle. `Quat::IDENTITY.slerp(δ, w)`
    // produces a fractional rotation between "no correction" and
    // "full correction".
    //
    // For each bone the conjugation `world_inv * δ_world * world`
    // expresses the world-rotation in the bone's local frame, which
    // is what `Transform.rotation` stores. Bevy uses its own glam
    // 0.30; convert the workspace 0.29 quats at the boundary.
    let root_delta_full = g_to_bevy_quat(solve.root_delta_world);
    let mid_delta_full = g_to_bevy_quat(solve.mid_delta_world);
    let root_delta = Quat::IDENTITY.slerp(root_delta_full, ik_weight);
    let mid_delta = Quat::IDENTITY.slerp(mid_delta_full, ik_weight);
    if let Ok(mut hip_tf) = transforms.get_mut(hip_e) {
        let delta_local = hip_world_old.inverse() * root_delta * hip_world_old;
        hip_tf.rotation = hip_tf.rotation * delta_local;
    }
    // Knee — its world rotation has just changed because hip rotated.
    // Apply mid_delta in the post-hip-rotation world frame.
    let knee_world_after_hip = root_delta * knee_world_old;
    if let Ok(mut knee_tf) = transforms.get_mut(knee_e) {
        let delta_local =
            knee_world_after_hip.inverse() * mid_delta * knee_world_after_hip;
        knee_tf.rotation = knee_tf.rotation * delta_local;
    }
}

#[inline]
fn bevy_to_g_vec3(v: Vec3) -> GVec3 {
    GVec3::new(v.x, v.y, v.z)
}

#[inline]
fn g_to_bevy_quat(q: GQuat) -> Quat {
    Quat::from_xyzw(q.x, q.y, q.z, q.w)
}

#[inline]
fn to_bevy_ivec3(v: GIVec3) -> bevy::math::IVec3 {
    bevy::math::IVec3::new(v.x, v.y, v.z)
}
