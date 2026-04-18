//! The KCC move resolver.
//!
//! This is the entire migration in one function: takes per-tick intent
//! (`DesiredMovement`, `PlatformDelta`), integrates gravity, blends
//! horizontal velocity toward the input target, and asks Rapier's
//! `KinematicCharacterController` to resolve the move.
//!
//! The function is self-contained — it doesn't depend on any shard-local
//! resource other than what's passed in — so it's unit-testable against
//! a hand-built Rapier world (see `tests` below) and reusable from both
//! ship-shard and planet-shard without duplication.

use bevy_ecs::message::Message;
use glam::{Vec2, Vec3};

use rapier3d::control::CharacterCollision;
use rapier3d::dynamics::RigidBodySet;
use rapier3d::geometry::ColliderSet;
use rapier3d::math::{Isometry, Real, Vector};
use rapier3d::pipeline::{QueryFilter, QueryPipeline};

use super::components::CharacterController;
use super::desired::{DesiredMovement, PlatformDelta};
use super::locomotion::{CharacterVelocity, LandedEvent, LocomotionState, PlatformSnapSuppressed};
use super::stats::MovementStats;

/// Terminal fall speed (m/s). Prevents gravity from integrating forever
/// during long falls. AAA MMOs typically clamp around 50–80 m/s. Matches
/// real-world skydiving belly-to-earth terminal velocity.                // TUNABLE
pub const TERMINAL_FALL_SPEED: f32 = 55.0;

/// Event forwarded to gameplay on KCC contacts. Wraps the raw Rapier
/// hit + the ECS entity for downstream handlers (footstep SFX, impact
/// damage, surface-type-aware friction).
#[derive(Message, Clone, Debug)]
pub struct CharacterCollisionEvent {
    pub entity: bevy_ecs::entity::Entity,
    pub hit: CharacterCollision,
}

/// Result of a single character move. Kept as a plain struct (no ECS)
/// so we can unit-test `move_one_character` in isolation.
///
/// `EffectiveCharacterMovement` from Rapier doesn't implement `Copy`/`Debug`
/// (`translation: Vector<Real>` is `Copy` but the struct itself isn't
/// marked) — we flatten the three fields we actually use instead.
#[derive(Clone, Copy, Debug)]
pub struct CharacterMoveResult {
    pub translation: Vector<Real>,
    pub grounded: bool,
    pub is_sliding_down_slope: bool,
    pub new_velocity: Vec3,
    pub landed_with_impact_speed: Option<f32>,
    pub new_state: LocomotionState,
}

/// Per-tick input + state for a single character's KCC move.
///
/// Collected from ECS by the shard-side glue and passed into
/// [`move_one_character`] (pure function over Rapier state). The ECS
/// system then writes back.
pub struct CharacterMoveInput<'a> {
    pub dt: f32,
    /// Previous state (used to decide grounded→airborne transitions + emit LandedEvent).
    pub prev_state: LocomotionState,
    pub velocity: Vec3,
    pub desired: DesiredMovement,
    pub platform_delta: PlatformDelta,
    pub gravity: Vec3,
    /// Character yaw (radians) — used to rotate the input's horizontal
    /// strafe/forward vector from character-local into the KCC frame
    /// (which is world-XZ on both ship and planet after re-centering).
    pub yaw: f32,
    pub stats: &'a MovementStats,
    pub crouching: bool,
    pub jump_grace_remaining: f32,
    pub snap_suppressed: bool,
}

/// Pure-function KCC move for one character.
///
/// Reads Rapier state (bodies + colliders + query pipeline), computes the
/// new velocity and position, and returns the result + the list of
/// collision callbacks. Performs **no** ECS writes — caller applies them.
///
/// This separation is load-bearing:
/// - Unit tests can call it with synthetic Rapier worlds.
/// - The parallel version (future) would iterate characters with
///   `rayon` since each call only needs `&` access to the shared sets.
pub fn move_one_character(
    bodies: &RigidBodySet,
    colliders: &ColliderSet,
    queries: &QueryPipeline,
    ctrl: &CharacterController,
    input: CharacterMoveInput,
    mut on_hit: impl FnMut(CharacterCollision),
) -> CharacterMoveResult {
    let body = bodies
        .get(ctrl.body)
        .expect("CharacterController.body missing from RigidBodySet");
    let collider = colliders
        .get(ctrl.collider)
        .expect("CharacterController.collider missing from ColliderSet");
    let shape = collider.shape();
    let char_pos: Isometry<Real> = *body.position();

    // 1. Horizontal target from input — rotated from character-local
    //    (strafe/forward) into the KCC frame by the current yaw.
    //
    //    Coordinate convention: yaw=0 faces +X; positive yaw rotates
    //    toward +Z. This matches the ship-shard's existing yaw→fwd
    //    formula (`cos(y), sin(y)`) used for `look_yaw` from the client.
    //    The planet-shard has a different yaw convention (yaw=0 faces
    //    +Z) which it handles by converting its input yaw to match
    //    before writing DesiredMovement.
    let speed = input
        .stats
        .horizontal_speed(input.crouching, input.desired.sprint);
    let clamped = clamp_horizontal_input(input.desired.horizontal);
    let (sin_y, cos_y) = input.yaw.sin_cos();
    // clamped.x = strafe (right), clamped.y = forward.
    // world_x = forward * cos + strafe * -sin
    // world_z = forward * sin + strafe * cos
    let world_x = clamped.y * cos_y - clamped.x * sin_y;
    let world_z = clamped.y * sin_y + clamped.x * cos_y;
    let target_horizontal = Vec2::new(world_x, world_z) * speed;

    // 2. Approach horizontal target with a capped acceleration.
    //    Fixes the original "patchy" bug: no instantaneous zeroing.
    //    Model: per-tick velocity delta capped at `accel * dt` in the
    //    direction of the target. Reaches the target in exactly
    //    `|target - current| / accel` seconds — predictable, testable,
    //    matches the feel of Source/Quake-era character movement.
    //
    //    `response_time` is retained as a **soft** upper bound: if the
    //    physical cap would take longer than `response_time`, we widen
    //    the cap so we hit the target in `response_time`. This lets
    //    designers tune feel with a single "time to target" knob while
    //    still respecting acceleration physics for small deltas.
    let prev_h = Vec2::new(input.velocity.x, input.velocity.z);
    let grounded_now = input.prev_state.is_grounded();
    let response = input.stats.response_time(grounded_now).max(1e-4);
    let diff = target_horizontal - prev_h;
    let diff_len = diff.length();
    let accel_cap = input.stats.accel_cap(grounded_now);
    let max_dv_from_accel = accel_cap * input.dt;
    // Implied acceleration to reach target in exactly `response_time`:
    let implied_accel = if response > 0.0 {
        diff_len / response
    } else {
        f32::MAX
    };
    let max_dv_from_response = implied_accel * input.dt;
    let max_dv = max_dv_from_accel.max(max_dv_from_response);
    let new_h = if diff_len <= max_dv {
        target_horizontal
    } else {
        prev_h + diff * (max_dv / diff_len)
    };

    // 3. Vertical: gravity integration + jump edge.
    let mut new_y = input.velocity.y;
    // `gravity.y` is signed (negative = downward in world frame).
    new_y += input.gravity.y * input.dt;
    if new_y < -TERMINAL_FALL_SPEED {
        new_y = -TERMINAL_FALL_SPEED;
    }
    let jumping = input.desired.jump && grounded_now && input.jump_grace_remaining <= 0.0;
    if jumping {
        new_y = input.stats.jump_speed;
    }

    // 4. Build desired translation = velocity * dt + platform delta.
    let mut desired_translation = Vector::new(new_h.x * input.dt, new_y * input.dt, new_h.y * input.dt);
    desired_translation += input.platform_delta.translation();

    // 5. KCC resolves collisions, autostep, slope slide, snap-to-ground.
    //    Snap is suppressed when the platform just lifted us large delta:
    //    preserves the platform's vertical translation instead of
    //    snapping back onto it.
    let kcc = if input.snap_suppressed {
        let mut k = ctrl.kcc;
        k.snap_to_ground = None;
        k
    } else if jumping {
        // Jumping implies we WANT to leave the ground; same reasoning.
        let mut k = ctrl.kcc;
        k.snap_to_ground = None;
        k
    } else {
        ctrl.kcc
    };

    let filter = QueryFilter::new().exclude_rigid_body(ctrl.body);
    let effective = kcc.move_shape(
        input.dt,
        bodies,
        colliders,
        queries,
        shape,
        &char_pos,
        desired_translation,
        filter,
        |c| on_hit(c),
    );

    // 6. Persist velocity: zero vertical if KCC blocked a fall/rise,
    //    clamp grounded y>=0 so gravity only accumulates in air.
    let mut persisted = Vec3::new(new_h.x, new_y, new_h.y);
    let epsilon = 1e-4;
    if effective.translation.y > desired_translation.y + epsilon && persisted.y < 0.0 {
        // We were falling but KCC clamped us above the desired y →
        // hit a floor.
        persisted.y = 0.0;
    } else if effective.translation.y < desired_translation.y - epsilon && persisted.y > 0.0 {
        // Rising but KCC clamped below → hit a ceiling.
        persisted.y = 0.0;
    }
    if effective.grounded && persisted.y < 0.0 {
        persisted.y = 0.0;
    }

    // 7. State transitions + LandedEvent.
    let new_state = if jumping {
        LocomotionState::Airborne
    } else if effective.grounded {
        LocomotionState::Grounded
    } else {
        LocomotionState::Airborne
    };

    let landed_impact = if matches!(input.prev_state, LocomotionState::Airborne)
        && matches!(new_state, LocomotionState::Grounded)
    {
        // Impact speed is the MAGNITUDE of the just-prior downward
        // velocity (positive m/s). Zero-clamp for sanity.
        Some(-input.velocity.y.min(0.0))
    } else {
        None
    };

    CharacterMoveResult {
        translation: effective.translation,
        grounded: effective.grounded,
        is_sliding_down_slope: effective.is_sliding_down_slope,
        new_velocity: persisted,
        landed_with_impact_speed: landed_impact,
        new_state,
    }
}

/// Clamp a raw input vector into the unit disk so diagonal movement isn't
/// √2 faster than cardinal. Preserves sub-unit magnitude for gamepads.
#[inline]
fn clamp_horizontal_input(raw: Vec2) -> Vec2 {
    let len_sq = raw.length_squared();
    if len_sq > 1.0 {
        raw / len_sq.sqrt()
    } else {
        raw
    }
}

// ---------------------------------------------------------------------------
// Shard-side ECS glue: kcc_move_characters
// ---------------------------------------------------------------------------
//
// Shards call this as a Bevy system in their `Physics` set. The function
// is kept here (in core) so there's a single implementation; the shards
// just need to wire it up with their `RapierContext` resource.
//
// The shard-local `RapierContext` struct is not known to core, so we
// expose a trait-based entry point: each shard adapts its own context
// to `RapierWorld` and passes it in.

/// Abstraction over the shard's Rapier resource. Each shard implements
/// this on a newtype wrapping its `RapierContext` to keep the core crate
/// ignorant of shard-specific resource layout.
pub trait RapierWorld {
    fn bodies(&self) -> &RigidBodySet;
    fn colliders(&self) -> &ColliderSet;
    fn queries(&self) -> &QueryPipeline;
    fn bodies_mut(&mut self) -> &mut RigidBodySet;
    /// Update the query pipeline from the current collider state.
    /// Must be idempotent within a tick — call only once per Physics set.
    fn refresh_query_pipeline(&mut self);
}

/// The ECS-side loop. Unit-tested indirectly via `move_one_character`.
/// Full ECS integration tests live in the shard crates once they wire this up.
pub fn kcc_move_all<W: RapierWorld>(
    world: &mut W,
    dt: f32,
    characters: &mut [CharacterRecord<'_>],
    mut emit_collision: impl FnMut(CharacterCollisionEvent),
    mut emit_landed: impl FnMut(LandedEvent),
) {
    world.refresh_query_pipeline();

    // Separate read and write phases so the borrow checker is happy.
    let results: Vec<(usize, CharacterMoveResult)> = {
        let bodies = world.bodies();
        let colliders = world.colliders();
        let queries = world.queries();

        characters
            .iter()
            .enumerate()
            .filter_map(|(idx, rec)| {
                if rec.state.skips_kcc() {
                    return None;
                }
                let entity = rec.entity;
                let input = CharacterMoveInput {
                    dt,
                    prev_state: *rec.state,
                    velocity: rec.velocity.0,
                    desired: *rec.desired,
                    platform_delta: *rec.platform_delta,
                    gravity: rec.gravity,
                    yaw: rec.yaw,
                    stats: rec.stats,
                    crouching: rec.crouching,
                    jump_grace_remaining: 0.0,
                    snap_suppressed: rec.snap_suppressed,
                };
                let result = move_one_character(bodies, colliders, queries, rec.ctrl, input, |hit| {
                    emit_collision(CharacterCollisionEvent { entity, hit });
                });
                Some((idx, result))
            })
            .collect()
    };

    // Apply writes (position + persisted state).
    let bodies_mut = world.bodies_mut();
    for (idx, result) in results {
        let rec = &mut characters[idx];
        if let Some(body) = bodies_mut.get_mut(rec.ctrl.body) {
            let cur = body.position().translation.vector;
            body.set_next_kinematic_translation(cur + result.translation);
        }
        rec.velocity.0 = result.new_velocity;
        *rec.state = result.new_state;
        if let Some(impact) = result.landed_with_impact_speed {
            emit_landed(LandedEvent {
                entity: rec.entity,
                impact_speed: impact,
            });
        }
    }
}

/// Borrowed handles to one character's per-tick inputs + outputs.
/// The shard ECS system collects a `Vec<CharacterRecord>` and hands
/// it to [`kcc_move_all`].
pub struct CharacterRecord<'a> {
    pub entity: bevy_ecs::entity::Entity,
    pub ctrl: &'a CharacterController,
    pub desired: &'a DesiredMovement,
    pub platform_delta: &'a PlatformDelta,
    pub stats: &'a MovementStats,
    pub gravity: Vec3,
    pub yaw: f32,
    pub velocity: &'a mut CharacterVelocity,
    pub state: &'a mut LocomotionState,
    pub crouching: bool,
    pub snap_suppressed: bool,
}

impl<'a> CharacterRecord<'a> {
    /// Convenience: read `PlatformSnapSuppressed` if present.
    pub fn with_snap_flag(mut self, flag: Option<&PlatformSnapSuppressed>) -> Self {
        self.snap_suppressed = flag.map(|f| f.0).unwrap_or(false);
        self
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use rapier3d::dynamics::{IslandManager, IntegrationParameters, CCDSolver, ImpulseJointSet, MultibodyJointSet};
    use rapier3d::geometry::{BroadPhaseMultiSap, ColliderBuilder, DefaultBroadPhase, NarrowPhase};
    use rapier3d::pipeline::PhysicsPipeline;

    use crate::character::controller::{build_character, CharacterBuildSpec};

    struct TestWorld {
        rigid_bodies: RigidBodySet,
        colliders: ColliderSet,
        queries: QueryPipeline,
        islands: IslandManager,
        broad: DefaultBroadPhase,
        narrow: NarrowPhase,
        joints: ImpulseJointSet,
        mb_joints: MultibodyJointSet,
        ccd: CCDSolver,
        params: IntegrationParameters,
        pipeline: PhysicsPipeline,
    }

    impl TestWorld {
        fn new() -> Self {
            let mut params = IntegrationParameters::default();
            params.dt = 0.05;
            Self {
                rigid_bodies: RigidBodySet::new(),
                colliders: ColliderSet::new(),
                queries: QueryPipeline::new(),
                islands: IslandManager::new(),
                broad: DefaultBroadPhase::new(),
                narrow: NarrowPhase::new(),
                joints: ImpulseJointSet::new(),
                mb_joints: MultibodyJointSet::new(),
                ccd: CCDSolver::new(),
                params,
                pipeline: PhysicsPipeline::new(),
            }
        }

        fn step(&mut self) {
            self.pipeline.step(
                &Vector::zeros(),
                &self.params,
                &mut self.islands,
                &mut self.broad,
                &mut self.narrow,
                &mut self.rigid_bodies,
                &mut self.colliders,
                &mut self.joints,
                &mut self.mb_joints,
                &mut self.ccd,
                Some(&mut self.queries),
                &(),
                &(),
            );
        }

        fn add_floor(&mut self) {
            // Large static box at y=-0.5, top face at y=0.
            let body = rapier3d::dynamics::RigidBodyBuilder::fixed()
                .translation(Vector::new(0.0, -0.5, 0.0))
                .build();
            let handle = self.rigid_bodies.insert(body);
            let col = ColliderBuilder::cuboid(50.0, 0.5, 50.0).build();
            self.colliders
                .insert_with_parent(col, handle, &mut self.rigid_bodies);
        }
    }

    fn spawn_character(world: &mut TestWorld, y: f32) -> CharacterController {
        build_character(
            &mut world.rigid_bodies,
            &mut world.colliders,
            CharacterBuildSpec {
                position: Vector::new(0.0, y, 0.0),
                ..Default::default()
            },
        )
    }

    #[test]
    fn flat_walk_reaches_target_speed() {
        let mut w = TestWorld::new();
        w.add_floor();
        let ctrl = spawn_character(&mut w, 0.31); // capsule bottom just above floor
        // Settle onto floor.
        w.step();
        w.queries.update(&w.colliders);

        let stats = MovementStats::default();
        let mut vel = CharacterVelocity::zero();
        let mut state = LocomotionState::Airborne;

        // 30 ticks of full-forward input.
        for _ in 0..30 {
            let result = move_one_character(
                &w.rigid_bodies,
                &w.colliders,
                &w.queries,
                &ctrl,
                CharacterMoveInput {
                    dt: 0.05,
                    prev_state: state,
                    velocity: vel.0,
                    desired: DesiredMovement {
                        horizontal: Vec2::new(0.0, 1.0),
                        ..Default::default()
                    },
                    platform_delta: PlatformDelta::default(),
                    gravity: Vec3::new(0.0, -9.81, 0.0),
                    yaw: 0.0,
                    stats: &stats,
                    crouching: false,
                    jump_grace_remaining: 0.0,
                    snap_suppressed: false,
                },
                |_| {},
            );
            if let Some(body) = w.rigid_bodies.get_mut(ctrl.body) {
                let cur = body.position().translation.vector;
                body.set_next_kinematic_translation(cur + result.translation);
            }
            vel.0 = result.new_velocity;
            state = result.new_state;
            w.step();
            w.queries.update(&w.colliders);
        }
        // By now horizontal-speed magnitude must be at (or very close to)
        // walk_speed. Direction depends on yaw convention; this test uses
        // yaw=0 where forward=+X (matches ship-shard's existing convention).
        let h_speed = (vel.0.x * vel.0.x + vel.0.z * vel.0.z).sqrt();
        assert!(
            h_speed >= stats.walk_speed - 0.1,
            "expected horizontal speed ≈ {}, got {} (vel = {:?})",
            stats.walk_speed,
            h_speed,
            vel.0,
        );
        assert_eq!(state, LocomotionState::Grounded);
    }

    #[test]
    fn release_input_decelerates_smoothly_not_instantly() {
        // The original bug: releasing movement caused INSTANT zero.
        // Verify that releasing input takes at least a couple of ticks
        // to fully decelerate.
        let mut w = TestWorld::new();
        w.add_floor();
        let ctrl = spawn_character(&mut w, 0.31);
        w.step();
        w.queries.update(&w.colliders);

        let stats = MovementStats::default();
        let mut vel = CharacterVelocity(Vec3::new(0.0, 0.0, stats.walk_speed));
        let mut state = LocomotionState::Grounded;

        // Release input for ONE tick — velocity should still be > 0.
        let result = move_one_character(
            &w.rigid_bodies,
            &w.colliders,
            &w.queries,
            &ctrl,
            CharacterMoveInput {
                dt: 0.05,
                prev_state: state,
                velocity: vel.0,
                desired: DesiredMovement::default(),
                platform_delta: PlatformDelta::default(),
                gravity: Vec3::new(0.0, -9.81, 0.0),
                yaw: 0.0,
                stats: &stats,
                crouching: false,
                jump_grace_remaining: 0.0,
                snap_suppressed: false,
            },
            |_| {},
        );
        vel.0 = result.new_velocity;
        state = result.new_state;
        assert!(
            vel.0.z > 0.5,
            "after 1 tick without input, velocity should still be positive (got {})",
            vel.0.z
        );
        // But within ~stop_response_s it should reach zero.
        let max_ticks = (stats.stop_response_s / 0.05).ceil() as i32 + 2;
        for _ in 0..max_ticks {
            let r = move_one_character(
                &w.rigid_bodies,
                &w.colliders,
                &w.queries,
                &ctrl,
                CharacterMoveInput {
                    dt: 0.05,
                    prev_state: state,
                    velocity: vel.0,
                    desired: DesiredMovement::default(),
                    platform_delta: PlatformDelta::default(),
                    gravity: Vec3::new(0.0, -9.81, 0.0),
                    yaw: 0.0,
                    stats: &stats,
                    crouching: false,
                    jump_grace_remaining: 0.0,
                    snap_suppressed: false,
                },
                |_| {},
            );
            vel.0 = r.new_velocity;
            state = r.new_state;
        }
        assert!(
            vel.0.z.abs() < 0.05,
            "velocity should have decelerated to ~0 (got {})",
            vel.0.z
        );
    }

    #[test]
    fn jump_transitions_grounded_to_airborne_to_grounded() {
        let mut w = TestWorld::new();
        w.add_floor();
        let ctrl = spawn_character(&mut w, 0.31);
        w.step();
        w.queries.update(&w.colliders);

        let stats = MovementStats::default();
        let mut vel = CharacterVelocity::zero();
        let mut state = LocomotionState::Grounded;
        let mut transitions = Vec::new();
        let mut landed_impact: Option<f32> = None;

        // Issue jump on tick 0, then hold empty input for 50 ticks.
        for tick in 0..50 {
            let desired = DesiredMovement {
                jump: tick == 0,
                ..Default::default()
            };
            let result = move_one_character(
                &w.rigid_bodies,
                &w.colliders,
                &w.queries,
                &ctrl,
                CharacterMoveInput {
                    dt: 0.05,
                    prev_state: state,
                    velocity: vel.0,
                    desired,
                    platform_delta: PlatformDelta::default(),
                    gravity: Vec3::new(0.0, -9.81, 0.0),
                    yaw: 0.0,
                    stats: &stats,
                    crouching: false,
                    jump_grace_remaining: 0.0,
                    snap_suppressed: false,
                },
                |_| {},
            );
            if let Some(body) = w.rigid_bodies.get_mut(ctrl.body) {
                let cur = body.position().translation.vector;
                body.set_next_kinematic_translation(cur + result.translation);
            }
            if result.new_state != state {
                transitions.push((tick, state, result.new_state));
            }
            if let Some(impact) = result.landed_with_impact_speed {
                landed_impact = Some(impact);
            }
            vel.0 = result.new_velocity;
            state = result.new_state;
            w.step();
            w.queries.update(&w.colliders);
        }
        // Must have seen Grounded → Airborne and back.
        assert!(
            transitions
                .iter()
                .any(|(_, from, to)| *from == LocomotionState::Grounded
                    && *to == LocomotionState::Airborne),
            "missing grounded→airborne: {:?}",
            transitions
        );
        assert!(
            transitions
                .iter()
                .any(|(_, from, to)| *from == LocomotionState::Airborne
                    && *to == LocomotionState::Grounded),
            "missing airborne→grounded: {:?}",
            transitions
        );
        assert!(
            landed_impact.is_some(),
            "expected a LandedEvent with impact speed"
        );
    }

    #[test]
    fn platform_delta_moves_character_with_zero_input() {
        let mut w = TestWorld::new();
        w.add_floor();
        let ctrl = spawn_character(&mut w, 0.31);
        w.step();
        w.queries.update(&w.colliders);

        let stats = MovementStats::default();
        let vel = CharacterVelocity::zero();
        let state = LocomotionState::Grounded;

        let start = w
            .rigid_bodies
            .get(ctrl.body)
            .unwrap()
            .position()
            .translation
            .vector;

        let mut platform = PlatformDelta::default();
        platform.0.translation.vector = Vector::new(1.0, 0.0, 0.0);

        let result = move_one_character(
            &w.rigid_bodies,
            &w.colliders,
            &w.queries,
            &ctrl,
            CharacterMoveInput {
                dt: 0.05,
                prev_state: state,
                velocity: vel.0,
                desired: DesiredMovement::default(),
                platform_delta: platform,
                gravity: Vec3::new(0.0, -9.81, 0.0),
                yaw: 0.0,
                stats: &stats,
                crouching: false,
                jump_grace_remaining: 0.0,
                snap_suppressed: false,
            },
            |_| {},
        );
        if let Some(body) = w.rigid_bodies.get_mut(ctrl.body) {
            let cur = body.position().translation.vector;
            body.set_next_kinematic_translation(cur + result.translation);
        }
        w.step();
        let end = w
            .rigid_bodies
            .get(ctrl.body)
            .unwrap()
            .position()
            .translation
            .vector;
        let delta = end - start;
        assert!(
            (delta.x - 1.0).abs() < 0.05,
            "expected ~1.0 platform translation in X, got {}",
            delta.x
        );
    }

    #[test]
    fn seated_state_skips_kcc() {
        // `LocomotionState::Seated` should make `kcc_move_all` treat the
        // character as if it weren't there. We verify via the `skips_kcc`
        // gate directly — full ECS integration test lives in shard crates.
        assert!(LocomotionState::Seated.skips_kcc());
        assert!(LocomotionState::Ragdoll.skips_kcc());
        assert!(!LocomotionState::Grounded.skips_kcc());
        assert!(!LocomotionState::Airborne.skips_kcc());
    }

    #[test]
    fn input_clamped_to_unit_disk() {
        // Diagonal [1,1] must not be sqrt(2) faster than axial.
        let h = clamp_horizontal_input(Vec2::new(1.0, 1.0));
        let len = h.length();
        assert!((len - 1.0).abs() < 1e-6, "diagonal should clamp to unit, got {}", len);
        // Sub-unit input is preserved (for gamepad).
        let h2 = clamp_horizontal_input(Vec2::new(0.3, 0.4));
        assert!((h2.length() - 0.5).abs() < 1e-6);
    }
}
