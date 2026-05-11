//! Ragdoll physics — Phase I infrastructure.
//!
//! When a character dies, the kinematic character body is replaced by
//! a chain of dynamic rigid bodies joined by spherical / revolute
//! joints with biomechanical limits. Rapier integrates them
//! authoritatively for `ragdoll_lifetime_secs` (default 5 s) and
//! broadcasts per-tick bone transforms to clients, which override
//! the corresponding visual bones each frame. After the lifetime
//! expires the entity despawns (or — once we have a loot system —
//! converts to a corpse interactable).
//!
//! # Server-only authority
//!
//! All physics integration happens on the shard that owned the
//! character at death. Clients never simulate ragdolls — they only
//! render bone transforms received from the wire. This matches the
//! "server-driven, no client prediction" mandate that keeps the
//! 100 K-player target tractable.
//!
//! # What's wired up today
//!
//! - `RagdollBoneSpec` (per-class, in `CharacterClass.ragdoll_bones`):
//!   bone topology, capsule sizes, joint types + limits, bind-pose
//!   parent-relative offsets.
//! - `spawn_ragdoll_bodies` / `despawn_ragdoll_bodies`: pure rapier
//!   operations, take the body / collider / joint sets by `&mut`
//!   so the caller's ECS scheduling is decoupled.
//! - `RagdollHandles` (component): per-character storage of the
//!   ragdoll's rigid bodies + joints.
//! - `RagdollLifetime` (component): countdown timer driving
//!   `update_ragdoll_lifetime` shard systems.
//!
//! # What's NOT wired up today (deliberately)
//!
//! - The DEATH EVENT. No `Health -> 0` transition triggers ragdoll
//!   spawn in this commit because we don't have a damage / death
//!   system yet. The hook is left in `transition_to_ragdoll` which
//!   future combat code will call.
//! - Quantization on the wire-format bone transforms (sent as full
//!   f32 today; ~84 B / character / tick at the planned 6-byte
//!   quantization is a follow-up once bandwidth measurement says
//!   it's worth doing).

#[cfg(feature = "rapier")]
use rapier3d::dynamics::{
    GenericJoint, ImpulseJointSet, IslandManager, JointAxesMask, JointAxis, JointLimits,
    MultibodyJointSet, RigidBodyBuilder, RigidBodyHandle, RigidBodySet,
};
#[cfg(feature = "rapier")]
use rapier3d::geometry::{ColliderBuilder, ColliderSet};
#[cfg(feature = "rapier")]
use rapier3d::math::{Pose, Vector};

use bevy_ecs::component::Component;
use glam::{Quat, Vec3};

/// Topology label for one ragdoll bone's joint to its parent.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RagdollJointType {
    /// Root body — no parent, no joint. Exactly one bone per spec
    /// must use this (typically the pelvis).
    Root,
    /// 3-DOF rotation, translation locked. Used at hips, shoulders,
    /// neck, wrists, ankles — anywhere the limb pivots in all three
    /// rotational axes within its anatomical limits.
    Spherical,
    /// 1-DOF rotation around the joint's local X axis, all other
    /// DOFs locked. Used at knees and elbows.
    Revolute,
}

/// Static per-rig description of one bone in a ragdoll's skeleton.
///
/// Stored in `CharacterClass.ragdoll_bones` and consumed by
/// `spawn_ragdoll_bodies` at death time. The spec is keyed by
/// `bone_name` (Mixamo-style identifier, e.g. `"mixamorig:LeftLeg"`)
/// so the wire-format bone transforms can be reattached to the
/// correct visual bone client-side without an additional id lookup.
#[derive(Clone, Copy, Debug)]
pub struct RagdollBoneSpec {
    pub bone_name: &'static str,
    /// Parent bone's `bone_name` in this same spec. `None` indicates
    /// the root (must match the bone with `joint_type = Root`).
    pub parent_bone: Option<&'static str>,
    /// Capsule half-height (cylindrical middle length / 2). Rapier
    /// `Capsule::new(half_height, radius)` convention. The capsule's
    /// principal axis runs along the bone's local +Y (head→tail).
    pub capsule_half_height: f32,
    pub capsule_radius: f32,
    /// Body mass. Sensible biomechanical defaults: pelvis ~10 kg,
    /// thigh ~8 kg, upper arm ~3 kg, head ~5 kg. Heavier root makes
    /// the ragdoll feel grounded; uniform low mass makes it
    /// "flutter."
    pub mass: f32,
    /// Translation from the PARENT joint to THIS joint, in the
    /// parent's local bind-pose frame. Ignored for the root.
    pub local_anchor_in_parent: Vec3,
    /// Translation from THIS bone's local origin to its joint
    /// anchor. For Mixamo: typically `Vec3::ZERO` because the bone
    /// origin coincides with the joint pivot.
    pub local_anchor_in_self: Vec3,
    pub joint_type: RagdollJointType,
    /// Joint angle limits (radians, half-cone). For `Revolute`,
    /// only `pitch_limit` is honoured (rotation around local X).
    /// For `Spherical`, all three constrain the swing.
    pub joint_yaw_limit: f32,
    pub joint_pitch_limit: f32,
    pub joint_roll_limit: f32,
}

/// Per-character storage of the rigid bodies that make up the
/// active ragdoll. Stored as a sorted `Vec` keyed by bone name so
/// the wire-format broadcast iterates in a deterministic order
/// (necessary for delta-encoding the snapshot down the road).
#[derive(Component, Default, Debug)]
#[cfg(feature = "rapier")]
pub struct RagdollHandles {
    /// `(bone_name, body_handle)` pairs, sorted by `bone_name`.
    pub bodies: Vec<(&'static str, RigidBodyHandle)>,
    /// Joint handles in the order they were inserted (parent before
    /// child — guarantees a clean tear-down order).
    pub joints: Vec<rapier3d::dynamics::ImpulseJointHandle>,
}

/// Countdown until the ragdoll despawns. Driven by
/// `update_ragdoll_lifetime` on the shard side. Once `remaining`
/// reaches zero the bodies + joints are removed from the Rapier
/// world and the entity is despawned (or — once loot lands —
/// promoted to a corpse).
#[derive(Component, Clone, Copy, Debug)]
pub struct RagdollLifetime {
    pub remaining_secs: f32,
}

impl RagdollLifetime {
    /// Start a fresh lifetime from the per-class default.
    pub fn new(total_secs: f32) -> Self {
        Self {
            remaining_secs: total_secs,
        }
    }

    /// Advance the timer. Returns `true` once expired.
    #[inline]
    pub fn tick(&mut self, dt: f32) -> bool {
        self.remaining_secs -= dt;
        self.remaining_secs <= 0.0
    }
}

/// Build the ragdoll's rigid bodies + joints in the Rapier world.
///
/// All bodies are spawned at world-space positions chained from the
/// `root_world_pose` through each bone's `local_anchor_in_parent`,
/// so the ragdoll instantly inhabits the character's last pose at
/// the moment of death (no teleport-from-bind-pose glitch).
///
/// The function is intentionally pure rapier — no Bevy
/// `Commands`, no `RapierContext` (each shard wraps differently).
/// Callers are responsible for inserting the returned
/// `RagdollHandles` as a component on the dying character's entity.
#[cfg(feature = "rapier")]
pub fn spawn_ragdoll_bodies(
    bodies: &mut RigidBodySet,
    colliders: &mut ColliderSet,
    joints: &mut ImpulseJointSet,
    class: &super::CharacterClass,
    root_world_pose: Pose,
) -> RagdollHandles {
    use std::collections::HashMap;

    // Pass 1: chain through specs in declared order, computing each
    // bone's world pose from its parent's world pose + the spec's
    // `local_anchor_in_parent` offset.
    let mut world_poses: HashMap<&'static str, Pose> = HashMap::new();
    let mut spawned: Vec<(&'static str, RigidBodyHandle)> = Vec::with_capacity(class.ragdoll_bones.len());

    for spec in class.ragdoll_bones {
        // World pose for this bone.
        let world_pose = match spec.parent_bone {
            None => root_world_pose,
            Some(parent_name) => {
                let Some(parent_pose) = world_poses.get(parent_name) else {
                    // Bad spec — parent declared AFTER child. Skip
                    // gracefully so a malformed config can't crash
                    // the shard. The ragdoll's topology will be
                    // partial but the shard stays up.
                    continue;
                };
                let offset_local = Vector::new(
                    spec.local_anchor_in_parent.x,
                    spec.local_anchor_in_parent.y,
                    spec.local_anchor_in_parent.z,
                );
                let world_translation = parent_pose.translation + parent_pose.rotation * offset_local;
                Pose::from_parts(world_translation, parent_pose.rotation)
            }
        };
        world_poses.insert(spec.bone_name, world_pose);

        // Build the rigid body. Dynamic; mass set explicitly so the
        // bones don't inherit the (often broken) auto-computed mass
        // from a capsule of small radius.
        let body = RigidBodyBuilder::dynamic()
            .position(world_pose)
            .additional_mass(spec.mass)
            .ccd_enabled(true)
            .build();
        let body_handle = bodies.insert(body);

        // Capsule collider — Y-aligned, matching the bone's local
        // +Y bone direction in Mixamo bind.
        let collider = ColliderBuilder::capsule_y(spec.capsule_half_height, spec.capsule_radius)
            .friction(0.8)
            .restitution(0.0)
            .build();
        colliders.insert_with_parent(collider, body_handle, bodies);

        spawned.push((spec.bone_name, body_handle));
    }

    // Pass 2: create the joints. We do this after all bodies exist
    // so the lookups can't fail in a topologically valid spec.
    let mut joint_handles = Vec::with_capacity(class.ragdoll_bones.len().saturating_sub(1));
    let body_by_name: HashMap<&'static str, RigidBodyHandle> =
        spawned.iter().copied().collect();
    for spec in class.ragdoll_bones {
        let Some(parent_name) = spec.parent_bone else {
            continue; // Root has no joint.
        };
        let (Some(&parent_handle), Some(&child_handle)) = (
            body_by_name.get(parent_name),
            body_by_name.get(spec.bone_name),
        ) else {
            continue;
        };
        let joint = build_joint(spec);
        let handle = joints.insert(parent_handle, child_handle, joint, true);
        joint_handles.push(handle);
    }

    // Sort handles by name to give the broadcast a deterministic order.
    spawned.sort_by_key(|(name, _)| *name);
    RagdollHandles {
        bodies: spawned,
        joints: joint_handles,
    }
}

/// Tear down everything `spawn_ragdoll_bodies` created. Idempotent:
/// missing handles are silently skipped so a partially-built
/// ragdoll can still be cleaned up.
#[cfg(feature = "rapier")]
pub fn despawn_ragdoll_bodies(
    bodies: &mut RigidBodySet,
    colliders: &mut ColliderSet,
    impulse_joints: &mut ImpulseJointSet,
    multibody_joints: &mut MultibodyJointSet,
    islands: &mut IslandManager,
    handles: &RagdollHandles,
) {
    // Joints first — removing a body would orphan its joints.
    for &handle in &handles.joints {
        impulse_joints.remove(handle, true);
    }
    for &(_, body_handle) in &handles.bodies {
        // `remove` also removes attached colliders + joints (defensive).
        bodies.remove(body_handle, islands, colliders, impulse_joints, multibody_joints, true);
    }
}

/// Build a `GenericJoint` matching the spec's topology + limits.
/// Returns a generic joint configured with the correct locked axes
/// and limit angles — rapier's specific `SphericalJoint` /
/// `RevoluteJoint` builders all reduce to this shape internally.
#[cfg(feature = "rapier")]
fn build_joint(spec: &RagdollBoneSpec) -> GenericJoint {
    let anchor1 = Vector::new(
        spec.local_anchor_in_parent.x,
        spec.local_anchor_in_parent.y,
        spec.local_anchor_in_parent.z,
    );
    let anchor2 = Vector::new(
        spec.local_anchor_in_self.x,
        spec.local_anchor_in_self.y,
        spec.local_anchor_in_self.z,
    );
    let mut joint = match spec.joint_type {
        RagdollJointType::Spherical => GenericJoint::new(JointAxesMask::LOCKED_SPHERICAL_AXES),
        RagdollJointType::Revolute => GenericJoint::new(JointAxesMask::LOCKED_REVOLUTE_AXES),
        RagdollJointType::Root => unreachable!("build_joint called for root bone"),
    };
    joint.set_local_anchor1(anchor1);
    joint.set_local_anchor2(anchor2);

    // Apply per-axis limits.
    match spec.joint_type {
        RagdollJointType::Spherical => {
            joint.set_limits(
                JointAxis::AngX,
                [-spec.joint_pitch_limit, spec.joint_pitch_limit],
            );
            joint.set_limits(
                JointAxis::AngY,
                [-spec.joint_yaw_limit, spec.joint_yaw_limit],
            );
            joint.set_limits(
                JointAxis::AngZ,
                [-spec.joint_roll_limit, spec.joint_roll_limit],
            );
        }
        RagdollJointType::Revolute => {
            // Knee / elbow — single-axis hinge around local X.
            joint.set_limits(
                JointAxis::AngX,
                [-spec.joint_pitch_limit, spec.joint_pitch_limit],
            );
        }
        RagdollJointType::Root => {}
    }
    joint
}

// -----------------------------------------------------------------
// Wire-format bone transform
// -----------------------------------------------------------------

/// One bone's pose sent over the wire during ragdoll simulation.
///
/// The bone is identified by its name (matching the visual rig's
/// `Name` component) and carries world-space translation + rotation.
/// f32 today; a future quantization pass shrinks this to ~6 bytes /
/// bone (16-bit unit-quat + 16-bit-per-axis position delta from a
/// shared anchor — that's a 2-3× wire-bandwidth win at peak ragdoll
/// density).
#[derive(Clone, Debug, Default)]
pub struct RagdollBoneTransform {
    pub bone_name: String,
    pub translation: Vec3,
    pub rotation: Quat,
}

// -----------------------------------------------------------------
// Death-trigger hook (NOT wired up to a damage system yet)
// -----------------------------------------------------------------

/// Wire-up point for future combat code.
///
/// When a damage system eventually drives a player's `Health` to 0,
/// it should:
///   1. Read the character's current visual bone world poses
///      (walking the rig like the IK systems do).
///   2. Call `spawn_ragdoll_bodies` with those bone poses chained
///      from the hips.
///   3. Insert the returned `RagdollHandles` + a fresh
///      `RagdollLifetime::new(class.ragdoll_lifetime_secs)` on the
///      character entity.
///   4. Transition `LocomotionState` to `Ragdoll` (which already
///      `skips_kcc` per `locomotion::LocomotionState::skips_kcc`).
///   5. Remove the kinematic character body via
///      `kcc_sys::destroy_character_body` (to be added when this
///      hook activates).
///
/// Until that combat system exists, this module's data structures
/// are dormant: no spawn, no lifetime tick, no broadcast. The shard
/// systems that DO exist (`update_ragdoll_lifetime`,
/// `broadcast_ragdoll_bones`) become no-ops when no entity has the
/// `RagdollHandles` component.
pub fn _docs_only_death_hook() {}
