//! `SignalProperty` — discriminator for what a functional-block signal
//! channel reads or writes.
//!
//! Lifted from the prior `voxeldust_core::signal::types` module. Lives at
//! the bottom of the crate graph because:
//!  * `voxeldust-signal::config` validates bindings against a property set
//!    keyed by [`crate::FunctionalBlockKind`], so both types must coexist
//!    without going through `voxeldust-core`;
//!  * the wire-protocol `as_ordinal`/`from_ordinal` mapping is stable
//!    cross-crate and does not depend on any signal infrastructure.

/// Which property of a functional block is read/written by a signal.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum SignalProperty {
    /// Boolean: on/off state.
    Active,
    /// Float 0.0–1.0: power/throttle level.
    Throttle,
    /// Float: angle in degrees (rotor target).
    Angle,
    /// Float 0.0–1.0: piston extension.
    Extension,
    /// Float: pressure reading (kPa).
    Pressure,
    /// Float: speed reading (m/s).
    Speed,
    /// Float 0.0–1.0: fill/charge level (battery).
    Level,
    /// u8: discrete switch state (junction branch index).
    SwitchState,
    /// Float: thrust boost multiplier (1.0 = normal). Set by cruise drives.
    Boost,
    /// Float: mechanical status code (0=Idle, 1=Moving, 2=Blocked, 3=Error).
    Status,
    /// Text: server-authored display string (body names, warp
    /// targets, ship callsigns). Only valid on the wire inside
    /// `HudSignalValue::Text`; core simulation channels stay numeric.
    Text,
}

impl SignalProperty {
    /// Stable u8 ordinal used by the HudSignalEntry wire protocol.
    /// Adding variants MUST keep existing ordinals stable.
    pub fn as_ordinal(self) -> u8 {
        match self {
            Self::Active => 0,
            Self::Throttle => 1,
            Self::Angle => 2,
            Self::Extension => 3,
            Self::Pressure => 4,
            Self::Speed => 5,
            Self::Level => 6,
            Self::SwitchState => 7,
            Self::Boost => 8,
            Self::Status => 9,
            Self::Text => 10,
        }
    }

    pub fn from_ordinal(v: u8) -> Option<Self> {
        Some(match v {
            0 => Self::Active,
            1 => Self::Throttle,
            2 => Self::Angle,
            3 => Self::Extension,
            4 => Self::Pressure,
            5 => Self::Speed,
            6 => Self::Level,
            7 => Self::SwitchState,
            8 => Self::Boost,
            9 => Self::Status,
            10 => Self::Text,
            _ => return None,
        })
    }
}
