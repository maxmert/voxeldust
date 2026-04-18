//! Movement tunables per character.
//!
//! Replaces the ship-shard local `PlayerPhysics` struct and the duplicated
//! `WALK_SPEED` / `JUMP_IMPULSE` constants in planet-shard. Values marked
//! `// TUNABLE` below are gameplay knobs; changing them affects feel but not
//! correctness.

use bevy_ecs::component::Component;
use serde::{Deserialize, Serialize};

/// Per-character movement configuration.
///
/// Lives as a Component so equipment / stance / status-effect systems can
/// override a player's stats by inserting a replacement component (see
/// future [`crate::character::hooks::EquipmentLoad`]).
#[derive(Component, Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct MovementStats {
    // -- Horizontal speeds (m/s) --
    /// Default walking speed.                                     // TUNABLE
    pub walk_speed: f32,
    /// Sprinting speed (future; gated by stamina).                // TUNABLE
    pub sprint_speed: f32,
    /// Crouching speed (future; stance system).                   // TUNABLE
    pub crouch_speed: f32,

    // -- Vertical --
    /// Jump takeoff speed in m/s (directly assigned to vel.y).    // TUNABLE
    pub jump_speed: f32,

    // -- Acceleration curve --
    /// Time constant for reaching the target horizontal velocity
    /// when grounded. Smaller = snappier; `0.0` would be teleport.
    /// The original `set_linvel` bug was effectively `0.0`.       // TUNABLE
    pub stop_response_s: f32,
    /// Time constant while airborne. Heavy by design — avoids
    /// bunny-hop abuse and mid-air strafing exploits.             // TUNABLE
    pub air_response_s: f32,
    /// Maximum ground acceleration cap (m/s²). Clamps the per-
    /// tick velocity delta regardless of response time — lets
    /// short pulses remain predictable.                           // TUNABLE
    pub ground_accel: f32,
    /// Maximum air acceleration cap.                              // TUNABLE
    pub air_accel: f32,

    // -- KCC geometry --
    /// Maximum floor angle the character can climb (degrees).     // TUNABLE
    pub max_slope_deg: f32,
    /// Autostep height in absolute meters (0 = disabled).         // TUNABLE
    pub autostep_height: f32,
    /// Autostep minimum landing width in meters.                  // TUNABLE
    pub autostep_min_width: f32,
    /// Max distance the character snaps down to ground after
    /// a move (meters). 0 = disabled.                             // TUNABLE
    pub snap_distance: f32,
    /// KCC skin gap in meters — small positive buffer that keeps
    /// the shape cast numerically stable.                         // TUNABLE
    pub skin_offset: f32,
}

impl Default for MovementStats {
    /// Defaults match the legacy hardcoded values so Phase-B wiring is a
    /// behaviour-preserving swap (speeds/jump unchanged; the difference is
    /// the controller algorithm, not the numbers).
    fn default() -> Self {
        Self {
            walk_speed: 4.0,
            sprint_speed: 6.5,
            crouch_speed: 1.8,
            jump_speed: 5.0,
            stop_response_s: 0.10,
            air_response_s: 0.80,
            ground_accel: 50.0,
            air_accel: 10.0,
            max_slope_deg: 45.0,
            autostep_height: 0.5,
            autostep_min_width: 0.3,
            snap_distance: 0.2,
            skin_offset: 0.01,
        }
    }
}

impl MovementStats {
    /// Response time used by the velocity-blend curve, selected by
    /// locomotion state. Grounded is snappy; airborne is floaty.
    #[inline]
    pub fn response_time(&self, grounded: bool) -> f32 {
        if grounded {
            self.stop_response_s
        } else {
            self.air_response_s
        }
    }

    /// Acceleration cap for the same state.
    #[inline]
    pub fn accel_cap(&self, grounded: bool) -> f32 {
        if grounded {
            self.ground_accel
        } else {
            self.air_accel
        }
    }

    /// Horizontal speed for the given locomotion modifiers.
    /// Stance/sprint/crouch multipliers live here so gameplay code can
    /// query one value per tick.
    #[inline]
    pub fn horizontal_speed(&self, crouching: bool, sprinting: bool) -> f32 {
        // Crouching wins over sprinting — you can't sprint while prone.
        if crouching {
            self.crouch_speed
        } else if sprinting {
            self.sprint_speed
        } else {
            self.walk_speed
        }
    }
}
