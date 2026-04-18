//! Core character components — the Rapier handles + capsule geometry.

use bevy_ecs::component::Component;

/// Marker for character entities.
///
/// Queries like `With<IsCharacter>` are the canonical way to identify
/// "this entity is a player / NPC that walks" — the KCC systems gate
/// on this marker rather than on `Player` (which is shard-local and
/// shouldn't leak across crates).
#[derive(Component, Clone, Copy, Debug, Default)]
pub struct IsCharacter;

/// Capsule dimensions for the character collider.
///
/// Height is the **cylindrical middle length** — Rapier's
/// `capsule_y(half_height, radius)` convention — so total body height ≈
/// `height + 2 * radius`. Written by `apply_stance_transitions` when
/// stance changes; the KCC controller rebuilds its shape from this.
#[derive(Component, Clone, Copy, Debug, PartialEq)]
pub struct CharacterCapsule {
    pub height: f32,
    pub radius: f32,
}

impl Default for CharacterCapsule {
    fn default() -> Self {
        // Matches `CharacterStance::Standing::capsule()` — keep in sync.
        Self {
            height: 0.6,
            radius: 0.3,
        }
    }
}

/// Wraps Rapier's `KinematicCharacterController` plus the kinematic-
/// position-based `RigidBody` handle that other entities can still
/// collide against. The controller itself is `Copy` + cheap to reconfigure
/// per tick if stance or gravity changes.
#[cfg(feature = "rapier")]
#[derive(Component, Debug)]
pub struct CharacterController {
    pub kcc: rapier3d::control::KinematicCharacterController,
    pub body: rapier3d::dynamics::RigidBodyHandle,
    pub collider: rapier3d::geometry::ColliderHandle,
}
