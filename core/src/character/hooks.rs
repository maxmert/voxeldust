//! Reserved extension-point components for future character gameplay.
//!
//! **Not implemented today.** These stubs exist so the KCC migration sets
//! up insertion points once — the next gameplay feature (stamina cost on
//! sprint, equipment-weight slow-down, fall damage, armor resistances) can
//! land without touching the physics pipeline again.
//!
//! Each component carries the minimal fields its consumer will need, so
//! wire-format evolution (`PlayerHandoff::character_state` blob) already
//! has a defined shape per tag.

use bevy_ecs::component::Component;
use serde::{Deserialize, Serialize};

/// Tag enum for the append-only character-state blob serialized into
/// `PlayerHandoff::character_state`. Values are stable once published —
/// a new tag appends; removing a tag is a breaking change (bump
/// `schema_version`).
#[repr(u8)]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum CharacterComponentTag {
    Stamina = 1,
    EquipmentLoad = 2,
    ActiveItem = 3,
    DamageResistance = 4,
}

// ---------------------------------------------------------------------------
// Stamina
// ---------------------------------------------------------------------------

/// Stamina pool — drained by sprinting and jumping, regenerates when idle.
///
/// **Not yet consumed** by any system. Future wiring (in
/// `apply_movement_modifiers`):
/// ```ignore
/// // FUTURE:
/// // if desired.sprint && stamina.current > 0.0 {
/// //     stamina.current -= SPRINT_COST * dt;
/// // } else {
/// //     stamina.current = (stamina.current + stamina.regen * dt).min(stamina.max);
/// // }
/// // if stamina.current <= 0.0 { desired.sprint = false; }
/// ```
#[derive(Component, Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct Stamina {
    pub current: f32,
    pub max: f32,
    /// Regen rate (units / second).
    pub regen: f32,
}

impl Default for Stamina {
    fn default() -> Self {
        Self {
            current: 100.0,
            max: 100.0,
            regen: 20.0,
        }
    }
}

// ---------------------------------------------------------------------------
// Equipment load
// ---------------------------------------------------------------------------

/// Aggregate equipment mass (armor + weapons + gear).
///
/// **Not yet consumed.** Future wiring:
/// ```ignore
/// // FUTURE: speed_mult = 1.0 - (load.mass_kg / LOAD_CAP).clamp(0.0, 0.5);
/// //         target_horizontal *= speed_mult;
/// ```
#[derive(Component, Clone, Copy, Debug, PartialEq, Default, Serialize, Deserialize)]
pub struct EquipmentLoad {
    pub mass_kg: f32,
}

// ---------------------------------------------------------------------------
// Active item
// ---------------------------------------------------------------------------

/// Currently equipped weapon or tool.
///
/// **Not yet consumed.** Future wiring routes `PlayerInputData.actions_bits`
/// (PRIMARY_FIRE, SECONDARY_FIRE, RELOAD) to the entity referenced here.
/// Generalizes the existing seat-binding system — a weapon is "a seat you
/// wear" whose bindings come from an `ItemSchema`.
#[derive(Component, Clone, Copy, Debug, PartialEq, Default, Serialize, Deserialize)]
pub struct ActiveItem {
    /// Hotbar slot 0..=9; server looks up the item definition from the
    /// player's inventory.
    pub slot: u8,
    /// Item mode (fire mode, tool subtype). Item-specific meaning.
    pub mode: u8,
}

// ---------------------------------------------------------------------------
// Damage resistance
// ---------------------------------------------------------------------------

/// Per-damage-type reduction factors `[0.0, 1.0)` — 0 = full damage,
/// 0.5 = half damage. Sourced from worn armor pieces.
///
/// **Not yet consumed.** Future wiring in a `DamageEvent` handler.
#[derive(Component, Clone, Copy, Debug, PartialEq, Default, Serialize, Deserialize)]
pub struct DamageResistance {
    pub kinetic: f32,
    pub energy: f32,
    pub thermal: f32,
    pub chemical: f32,
}
