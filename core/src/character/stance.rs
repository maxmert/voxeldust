//! Character stance (standing / crouching / prone).
//!
//! Stance drives:
//! - Collider capsule dimensions (resize on transition).
//! - Horizontal speed multiplier (crouch is slower; prone is slower still).
//! - Weapon handling flags (future — hip-fire vs. ADS vs. bipod).
//!
//! A full stance transition takes time (resize animation) — gameplay can
//! read `is_in_transition` to block weapon fire mid-blend. For now the
//! transition is instant; the hook is here.

use bevy_ecs::component::Component;
use serde::{Deserialize, Serialize};

/// Current body stance.
#[derive(Component, Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum CharacterStance {
    #[default]
    Standing,
    Crouching,
    Prone,
}

/// Requested stance transition (written by input, consumed by
/// `apply_stance_transitions` in the Physics set).
#[derive(Component, Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum StanceAction {
    /// Toggle between Standing ↔ Crouching.
    ToggleCrouch,
    /// Toggle between Standing ↔ Prone.
    TogglePron,
    /// Step down: Standing → Crouching → Prone.
    CycleDown,
    /// Step up: Prone → Crouching → Standing.
    CycleUp,
    /// Explicit set.
    SetStanding,
    SetCrouching,
    SetProne,
}

/// Per-stance capsule dimensions (meters).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct StanceCapsule {
    pub height: f32,
    pub radius: f32,
}

impl CharacterStance {
    /// Capsule size for this stance.
    ///
    /// Height is the **cylindrical middle** length of the capsule — Rapier's
    /// `capsule_y(half_height, radius)` convention — so total character
    /// height ≈ `height + 2 * radius`.
    pub fn capsule(self) -> StanceCapsule {
        match self {
            // Stand ≈ 1.8 m tall: 0.6 cylinder + 2×0.3 radius.
            CharacterStance::Standing => StanceCapsule {
                height: 0.6,
                radius: 0.3,
            },
            // Crouch ≈ 1.2 m tall.
            CharacterStance::Crouching => StanceCapsule {
                height: 0.3,
                radius: 0.3,
            },
            // Prone ≈ 0.6 m tall.
            CharacterStance::Prone => StanceCapsule {
                height: 0.0,
                radius: 0.3,
            },
        }
    }

    /// Resolve a requested action into the next stance.
    pub fn apply(self, action: StanceAction) -> Self {
        use CharacterStance::*;
        match action {
            StanceAction::ToggleCrouch => {
                if matches!(self, Crouching) {
                    Standing
                } else {
                    Crouching
                }
            }
            StanceAction::TogglePron => {
                if matches!(self, Prone) {
                    Standing
                } else {
                    Prone
                }
            }
            StanceAction::CycleDown => match self {
                Standing => Crouching,
                Crouching => Prone,
                Prone => Prone,
            },
            StanceAction::CycleUp => match self {
                Prone => Crouching,
                Crouching => Standing,
                Standing => Standing,
            },
            StanceAction::SetStanding => Standing,
            StanceAction::SetCrouching => Crouching,
            StanceAction::SetProne => Prone,
        }
    }

    /// Speed multiplier applied on top of [`MovementStats::walk_speed`].
    /// Sprint is NOT a stance — it's an independent modifier handled by
    /// [`MovementStats::horizontal_speed`].
    #[inline]
    pub fn speed_multiplier(self) -> f32 {
        match self {
            CharacterStance::Standing => 1.0,
            CharacterStance::Crouching => 0.45,
            CharacterStance::Prone => 0.25,
        }
    }
}
