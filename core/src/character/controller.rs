//! Construction + reconfiguration of the Rapier KCC and its backing body.
//!
//! The character's rigid body is `kinematic_position_based` — NOT dynamic.
//! This means other entities can still query/collide with it, but its
//! translation is authoritative (Rapier integrates an implicit velocity
//! from `next_position - position`, which correctly shoves dynamics).

use rapier3d::control::{
    CharacterAutostep, CharacterLength, KinematicCharacterController,
};
use rapier3d::dynamics::{IslandManager, RigidBodyBuilder, RigidBodyHandle, RigidBodySet};
use rapier3d::geometry::{Collider, ColliderBuilder, ColliderHandle, ColliderSet};
use rapier3d::math::{Isometry, Real, Vector};
use rapier3d::na::{UnitVector3, Vector3};

use super::components::{CharacterCapsule, CharacterController};
use super::stats::MovementStats;

/// Parameters for building a character.
///
/// Kept as a struct (rather than positional args) because a character's
/// shape + stats are stable at spawn time but additional knobs might be
/// added — new fields get `Default` values without breaking callers.
#[derive(Clone, Copy, Debug)]
pub struct CharacterBuildSpec {
    pub position: Vector<Real>,
    pub capsule: CharacterCapsule,
    pub stats: MovementStats,
    /// World-space up axis passed to the KCC. For ship interior and
    /// planet surface this is `Y` (KCC always runs in the locally-flat
    /// Rapier frame). Override for rotating stations.
    pub up_axis: UnitVector3<Real>,
}

impl Default for CharacterBuildSpec {
    fn default() -> Self {
        Self {
            position: Vector::zeros(),
            capsule: CharacterCapsule::default(),
            stats: MovementStats::default(),
            up_axis: Vector3::y_axis(),
        }
    }
}

/// Build the kinematic body + capsule + KCC for a new character and
/// insert them into the provided Rapier sets.
///
/// Returns a `CharacterController` component ready to be attached to the
/// ECS entity.
pub fn build_character(
    rigid_bodies: &mut RigidBodySet,
    colliders: &mut ColliderSet,
    spec: CharacterBuildSpec,
) -> CharacterController {
    // Kinematic-position-based: Rapier integrates velocity from the
    // per-tick position delta, so other dynamic bodies get shoved
    // correctly by walking players. `lock_rotations` is implicit — a
    // kinematic body never rotates from physics; only we set it.
    let body = RigidBodyBuilder::kinematic_position_based()
        .translation(spec.position)
        .build();
    let body_handle = rigid_bodies.insert(body);

    let collider = make_collider(&spec.capsule);
    let collider_handle =
        colliders.insert_with_parent(collider, body_handle, rigid_bodies);

    CharacterController {
        kcc: configure_kcc(&spec.stats, spec.up_axis),
        body: body_handle,
        collider: collider_handle,
    }
}

/// Rebuild the collider when stance changes the capsule dimensions.
///
/// The KCC itself is shape-agnostic (it takes `&dyn Shape` at `move_shape`
/// time), but the Rapier body needs its collider replaced so overlap
/// queries from other systems see the correct hitbox.
///
/// Caller passes the shard's `IslandManager` because Rapier's collider
/// removal updates island membership. `wake_sleeping = false` because the
/// body is kinematic and has no solver island in the usual sense.
pub fn resize_capsule(
    rigid_bodies: &mut RigidBodySet,
    colliders: &mut ColliderSet,
    islands: &mut IslandManager,
    ctrl: &mut CharacterController,
    new_capsule: CharacterCapsule,
) {
    colliders.remove(ctrl.collider, islands, rigid_bodies, false);
    let new_collider = make_collider(&new_capsule);
    ctrl.collider = colliders.insert_with_parent(new_collider, ctrl.body, rigid_bodies);
}

/// Rebuild the KCC if the up axis changes (artificial gravity, rotating
/// stations). Most code paths never call this — ship and planet both use
/// Y-up via the existing re-centering invariant.
pub fn reconfigure_kcc(
    ctrl: &mut CharacterController,
    stats: &MovementStats,
    up_axis: UnitVector3<Real>,
) {
    ctrl.kcc = configure_kcc(stats, up_axis);
}

// ---------------------------------------------------------------------------
// Internals
// ---------------------------------------------------------------------------

fn make_collider(capsule: &CharacterCapsule) -> Collider {
    // Rapier `capsule_y(half_height, radius)` → Y-aligned capsule.
    // `height` in `CharacterCapsule` is the **cylindrical middle length**,
    // so half-height is `height / 2`.
    ColliderBuilder::capsule_y(capsule.height * 0.5, capsule.radius).build()
}

fn configure_kcc(
    stats: &MovementStats,
    up_axis: UnitVector3<Real>,
) -> KinematicCharacterController {
    let mut kcc = KinematicCharacterController::default();
    kcc.up = up_axis;
    kcc.offset = CharacterLength::Absolute(stats.skin_offset);
    kcc.slide = true;
    kcc.max_slope_climb_angle = stats.max_slope_deg.to_radians();
    // `min_slope_slide_angle` must be STRICTLY GREATER than
    // `max_slope_climb_angle`. If they're equal every floor hit falls
    // into an ambiguous "wall OR slope" branch inside Rapier's
    // `handle_slopes` — on voxel terrain with adjacent box colliders
    // that seam hits every ~0.5 m and the character loses 30-60 % of
    // the tick's tangential motion to the "let it slide" fallback.
    // Measured effect: walking at 4 m/s produced 0.125 m/tick instead
    // of 0.200 m/tick on ~40 % of ticks. Separating the two angles by
    // 10° gives a clear walkable band and eliminates the stutter.
    //
    // With max_slope_deg=45, slide fires only past 55° — steep enough
    // that a stationary character on the actual 45° boundary still
    // doesn't slide.                                                    // TUNABLE
    kcc.min_slope_slide_angle = (stats.max_slope_deg + 10.0).to_radians();
    kcc.snap_to_ground = if stats.snap_distance > 0.0 {
        Some(CharacterLength::Absolute(stats.snap_distance))
    } else {
        None
    };
    kcc.autostep = if stats.autostep_height > 0.0 {
        Some(CharacterAutostep {
            max_height: CharacterLength::Absolute(stats.autostep_height),
            min_width: CharacterLength::Absolute(stats.autostep_min_width),
            include_dynamic_bodies: true,
        })
    } else {
        None
    };
    // Small nudge against contact normals avoids numerical stick-on-wall.
    kcc.normal_nudge_factor = 1.0e-4;
    kcc
}

/// Build an `Isometry` for the KCC's `character_pos` from the body's
/// current translation.
#[inline]
pub fn isometry_for_body(
    bodies: &RigidBodySet,
    handle: RigidBodyHandle,
) -> Option<Isometry<Real>> {
    bodies.get(handle).map(|rb| *rb.position())
}

/// Fetch the character's collider shape (borrowed reference — no clone).
/// Needed because `move_shape` takes `&dyn Shape`.
#[inline]
pub fn collider_shape<'a>(
    colliders: &'a ColliderSet,
    handle: ColliderHandle,
) -> Option<&'a dyn rapier3d::geometry::Shape> {
    colliders.get(handle).map(|c| c.shape())
}
