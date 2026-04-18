//! Input → physics intermediate.
//!
//! `process_input` writes `DesiredMovement`; the Physics-set systems consume
//! it. This split means the input-side code never touches Rapier handles —
//! it's trivially serializable and unit-testable.

use bevy_ecs::component::Component;
use glam::Vec2;
use serde::{Deserialize, Serialize};

use super::stance::StanceAction;

/// Per-tick movement intent written by input, consumed by the KCC system.
///
/// Horizontal is a `[strafe, forward]` vector in character-local space
/// (not normalized — the KCC applies `horizontal_speed` to it). Magnitudes
/// above 1.0 are clamped by the consumer.
#[derive(Component, Clone, Copy, Debug, PartialEq, Serialize, Deserialize, Default)]
pub struct DesiredMovement {
    /// Horizontal input in **character-local** (body-yaw-aligned) XZ axes:
    /// `x` = right-strafe, `y` = forward. Coordinate choice matches the
    /// existing `input.movement[0]` (right) and `input.movement[2]`
    /// (forward) convention used by legacy `process_input`.
    pub horizontal: Vec2,
    /// Edge-triggered jump for this tick.
    pub jump: bool,
    /// Sprint modifier (future — routes through [`super::stats::MovementStats::horizontal_speed`]).
    pub sprint: bool,
    /// Crouch modifier (future).
    pub crouch: bool,
    /// Requested stance transition (`None` = keep current).
    pub stance_action: Option<StanceAction>,
}

impl DesiredMovement {
    /// Clear transient edges — call after a Physics tick has consumed them
    /// so the next tick doesn't re-trigger jump/stance.
    pub fn clear_edges(&mut self) {
        self.jump = false;
        self.stance_action = None;
    }
}

/// World-space delta transform accumulated by the platform-ride system.
///
/// Written by `update_player_platform`; consumed by `kcc_move_characters`
/// which folds it into the KCC's desired translation. Reset to identity at
/// the start of each Physics tick.
#[cfg(feature = "rapier")]
#[derive(Component, Clone, Copy, Debug)]
pub struct PlatformDelta(pub rapier3d::math::Isometry<f32>);

#[cfg(feature = "rapier")]
impl Default for PlatformDelta {
    fn default() -> Self {
        Self(rapier3d::math::Isometry::identity())
    }
}

#[cfg(feature = "rapier")]
impl PlatformDelta {
    /// Translation component extracted for KCC input.
    #[inline]
    pub fn translation(&self) -> rapier3d::math::Vector<f32> {
        self.0.translation.vector
    }

    /// Reset to identity (start of Physics tick).
    #[inline]
    pub fn reset(&mut self) {
        self.0 = rapier3d::math::Isometry::identity();
    }
}

/// Non-rapier fallback so the client and test binaries can still compile
/// against the character module without pulling Rapier.
#[cfg(not(feature = "rapier"))]
#[derive(Component, Clone, Copy, Debug, Default)]
pub struct PlatformDelta;
