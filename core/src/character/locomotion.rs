//! Character locomotion state machine + persisted velocity.
//!
//! `CharacterVelocity` is the single source of truth for per-character
//! motion — the piece `set_linvel` was destroying every packet. Gravity
//! accumulates into `.y` while airborne; horizontal is blended toward the
//! input target with a bounded response time (see `kcc_move_characters`).

use bevy_ecs::component::Component;
use bevy_ecs::message::Message;
use glam::{DVec3, Vec3};
use serde::{Deserialize, Serialize};

/// Coarse locomotion state — drives animation selection, gameplay gating
/// (e.g. you can't jump from `Seated`), and KCC skip logic.
#[derive(Component, Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum LocomotionState {
    /// Standing/walking on a ground surface detected by the KCC.
    Grounded,
    /// In free-fall / mid-jump — gravity integrates into vertical velocity.
    Airborne,
    /// Occupying a seat — KCC skips; transform follows the seat.
    Seated,
    /// Holding a ladder / cliff edge (reserved).
    Climbing,
    /// Temporarily ragdolled (reserved — body swapped dynamic on knockdown).
    Ragdoll,
}

impl Default for LocomotionState {
    fn default() -> Self {
        // Fresh characters spawn airborne so the first KCC tick decides
        // whether they're on a floor — avoids false "just landed" events
        // when a player spawns 0.001 m above the deck plate.
        LocomotionState::Airborne
    }
}

impl LocomotionState {
    /// The KCC system is a no-op in these states.
    #[inline]
    pub fn skips_kcc(self) -> bool {
        matches!(
            self,
            LocomotionState::Seated | LocomotionState::Ragdoll
        )
    }

    #[inline]
    pub fn is_grounded(self) -> bool {
        matches!(self, LocomotionState::Grounded)
    }
}

/// Persisted character velocity in the character's local frame (same
/// orientation as the KCC `up` axis — Y-up on ship and planet after the
/// existing re-centering). Horizontal (XZ) survives between inputs;
/// vertical (Y) integrates gravity and jumps.
#[derive(Component, Clone, Copy, Debug, Default, Serialize, Deserialize)]
pub struct CharacterVelocity(pub Vec3);

impl CharacterVelocity {
    #[inline]
    pub fn zero() -> Self {
        Self(Vec3::ZERO)
    }

    #[inline]
    pub fn horizontal(&self) -> glam::Vec2 {
        glam::Vec2::new(self.0.x, self.0.z)
    }

    #[inline]
    pub fn set_horizontal(&mut self, h: glam::Vec2) {
        self.0.x = h.x;
        self.0.z = h.y;
    }

    #[inline]
    pub fn vertical(&self) -> f32 {
        self.0.y
    }

    #[inline]
    pub fn set_vertical(&mut self, y: f32) {
        self.0.y = y;
    }
}

/// Per-character gravity axis in **world space**. On the ship this is
/// `DVec3::NEG_Y` relative to the ship body frame; on a planet it's the
/// radial-in direction from `TangentFrame`. Informational for clients and
/// cross-shard serialization; the KCC runs in the locally-flat Rapier
/// frame so its internal `up` is `Y` regardless (see `controller.rs`).
#[derive(Component, Clone, Copy, Debug)]
pub struct LocalUp(pub DVec3);

impl Default for LocalUp {
    fn default() -> Self {
        Self(DVec3::Y)
    }
}

/// Overrides the automatic gravity derivation for edge cases (rotating
/// stations, artificial-gravity fields, planar-gravity ships). When
/// present, the KCC rebuilds with a non-Y up axis. Out of scope for the
/// current migration — reserved.
#[derive(Component, Clone, Copy, Debug, Default)]
pub struct GravityOverride(pub Option<Vec3>);

/// Suppresses `snap_to_ground` for one tick — written by systems that know
/// the character just acquired a large vertical displacement from platform
/// motion (e.g., rotor lifting the player 1 m in one tick). Without this,
/// the KCC would re-ground the character and erase the lift.
#[derive(Component, Clone, Copy, Debug, Default)]
pub struct PlatformSnapSuppressed(pub bool);

/// Fired on `Airborne → Grounded` transitions. Consumer hook for fall
/// damage, landing SFX, footstep VFX.
///
/// `impact_speed` is the magnitude of the vertical velocity just before
/// landing (positive m/s). Future handler:
/// ```ignore
/// // FUTURE: impact_speed > FALL_DAMAGE_THRESHOLD
/// //   → DamageEvent { amount: damage_curve(impact_speed) }
/// //   → DamageResistance reduces → Health.0 -= final
/// ```
#[derive(Message, Clone, Copy, Debug)]
pub struct LandedEvent {
    pub entity: bevy_ecs::entity::Entity,
    pub impact_speed: f32,
}
