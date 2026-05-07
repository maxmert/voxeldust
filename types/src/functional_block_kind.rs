//! `FunctionalBlockKind` — coarse category of functional block, plus its
//! static signal-property schema. The single source of truth used by:
//!
//!  * the configurator UI (drop-down filtering by kind)
//!  * `voxeldust-signal::config::ConfigInvalid::validate_against_kind`
//!    (server-side rejection of out-of-schema bindings)
//!  * the block registry (kind → block-def cross-reference)
//!
//! Lifted from `voxeldust_core::block::registry` so signal config can
//! validate against block kinds without `voxeldust-signal` depending on
//! `voxeldust-core` (which itself depends on signal — the cycle is now
//! broken at this layer).

use crate::signal_property::SignalProperty;

/// Category of functional block — determines which subsystems interact with it.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum FunctionalBlockKind {
    Thruster,
    Reactor,
    Battery,
    SolarPanel,
    PowerConduit,
    Seat,
    GravityGenerator,
    ShieldEmitter,
    ShieldGenerator,
    AirCompressor,
    /// Phase D: bidirectional radio bridge. Holds optional TX side
    /// (subscribes to a local channel and forwards values to a Radio
    /// frequency) + optional RX side (publishes Radio-received values
    /// to a local channel). Replaces the prior split Antenna (TX-only)
    /// + Listener (RX-only) pair.
    Antenna,
    Rotor,
    Piston,
    Rail,
    RailJunction,
    RailSignal,
    SignalConverter,
    Sensor,
    Computer,
    CruiseDrive,
    FlightComputer,
    HoverModule,
    Autopilot,
    WarpComputer,
    EngineController,
    /// Phase D: unified Terminal block — subscribes to a media channel
    /// and renders received `MediaPayload::Text` frames on its in-world
    /// surface (read), AND publishes new `Text` frames the player types
    /// on E-key activation (write). Either side can be left empty:
    /// read-only signs (no publish), input-only kiosks (no subscribe),
    /// or full chat panels (both, typically same channel). Replaces the
    /// prior split TextDisplay (R-only) + KeyboardTerminal (W-only) pair.
    Terminal,
}

/// The publish/subscribe `SignalProperty` set this block kind supports.
///
/// A thruster doesn't know what `Pressure` means, so it shouldn't appear in
/// the configurator's property dropdown — neither for publish (the thruster
/// can't generate pressure data) nor for subscribe (the thruster can't
/// consume it). The block kind authoritatively declares the supported set
/// here; the configurator UI filters by it; `apply_config_updates` rejects
/// any binding whose property is outside the set.
///
/// Antenna and Listener blocks are *transparent relays* — their property
/// list reflects only their own state (`Active`, `Status`); the *bridged*
/// channel's property is decided by the source/destination channel, not by
/// the antenna itself, and is configured through a separate dedicated UI
/// (the Antenna/Listener config tab).
#[derive(Clone, Copy, Debug)]
pub struct BlockKindSignalSchema {
    /// Properties this block can write to a channel, in display order.
    pub publish_properties: &'static [SignalProperty],
    /// Properties this block can read from a channel, in display order.
    pub subscribe_properties: &'static [SignalProperty],
    /// Per-property tooltip strings for the configurator UI.
    pub property_hints: &'static [(SignalProperty, &'static str)],
}

impl FunctionalBlockKind {
    /// Discriminant value used on the wire (matches `as u8` cast).
    /// Stable across the enum's lifetime — adding a variant always
    /// goes at the end so existing values never shift.
    #[inline]
    pub const fn as_u8(self) -> u8 {
        self as u8
    }

    /// Static signal-property schema describing what this block kind can
    /// publish or subscribe to. Single source of truth for both the
    /// configurator UI (drop-down filtering) and `apply_config_updates`
    /// (server-side validation of incoming bindings).
    pub fn signal_schema(self) -> BlockKindSignalSchema {
        use SignalProperty::*;
        match self {
            Self::Thruster => BlockKindSignalSchema {
                publish_properties: &[],
                subscribe_properties: &[Throttle, Boost, Active],
                property_hints: &[
                    (Throttle, "0.0–1.0 thrust fraction"),
                    (Boost, "Multiplier on top of throttle (cruise drives)"),
                    (Active, "False to disable this thruster"),
                ],
            },
            Self::Reactor => BlockKindSignalSchema {
                publish_properties: &[Level, Status, Active],
                subscribe_properties: &[Throttle, Active],
                property_hints: &[
                    (Level, "Output as a 0.0–1.0 fraction of rated power"),
                    (Status, "Operational state (running/throttled/fault)"),
                    (Active, "False to shut down the reactor"),
                    (Throttle, "Dial output up/down (0.0–1.0)"),
                ],
            },
            Self::Battery => BlockKindSignalSchema {
                publish_properties: &[Level, Status],
                subscribe_properties: &[],
                property_hints: &[
                    (Level, "0.0–1.0 charge fraction"),
                    (Status, "Charging/idle/depleted"),
                ],
            },
            Self::SolarPanel => BlockKindSignalSchema {
                publish_properties: &[Level, Active],
                subscribe_properties: &[],
                property_hints: &[
                    (Level, "0.0–1.0 instantaneous output fraction"),
                    (Active, "False if panel is occluded or stowed"),
                ],
            },
            Self::PowerConduit => BlockKindSignalSchema {
                publish_properties: &[],
                subscribe_properties: &[],
                property_hints: &[],
            },
            Self::Seat => BlockKindSignalSchema {
                // Per-key seat input bindings (W/A/S/D etc.) live in the
                // dedicated SeatChannelMapping UI tab. The Active property
                // here is for the "seat occupied" boolean only.
                publish_properties: &[Active],
                subscribe_properties: &[],
                property_hints: &[
                    (Active, "True while a player is seated"),
                ],
            },
            Self::GravityGenerator => BlockKindSignalSchema {
                publish_properties: &[Active, Status],
                subscribe_properties: &[Active, Throttle],
                property_hints: &[
                    (Throttle, "0.0–1.0 gravity strength"),
                    (Active, "Master enable"),
                ],
            },
            Self::ShieldEmitter => BlockKindSignalSchema {
                publish_properties: &[Level, Active, Status],
                subscribe_properties: &[Active, Throttle],
                property_hints: &[
                    (Level, "Shield charge 0.0–1.0"),
                    (Throttle, "Power draw fraction"),
                ],
            },
            Self::ShieldGenerator => BlockKindSignalSchema {
                publish_properties: &[Level, Active, Status],
                subscribe_properties: &[Active, Throttle],
                property_hints: &[
                    (Level, "Shield charge 0.0–1.0"),
                    (Active, "Master enable"),
                ],
            },
            Self::AirCompressor => BlockKindSignalSchema {
                publish_properties: &[Pressure, Active, Status],
                subscribe_properties: &[Active, Throttle],
                property_hints: &[
                    (Pressure, "Compressed gas pressure (kPa)"),
                ],
            },
            Self::Antenna => BlockKindSignalSchema {
                // Phase A3 UX cleanup: Antenna has NO standalone signal
                // bindings. Its only configuration is the TX + RX
                // channels in the dedicated Antenna panel. Removing the
                // PUBLISH/SUBSCRIBE binding rows from the configurator
                // (auto-hidden when both arrays are empty) eliminates
                // the prior "what do these do — aren't TX/RX my
                // pub/sub?" confusion.
                publish_properties: &[],
                subscribe_properties: &[],
                property_hints: &[],
            },
            Self::Rotor => BlockKindSignalSchema {
                publish_properties: &[Angle, Speed, Status, Active],
                subscribe_properties: &[Angle, Throttle, Speed, Active],
                property_hints: &[
                    (Angle, "Position in degrees"),
                    (Throttle, "-1.0..1.0 velocity fraction (signed)"),
                    (Speed, "Max angular speed override (deg/s)"),
                    (Active, "False to lock the rotor"),
                ],
            },
            Self::Piston => BlockKindSignalSchema {
                publish_properties: &[Extension, Speed, Status, Active],
                subscribe_properties: &[Extension, Throttle, Speed, Active],
                property_hints: &[
                    (Extension, "Position 0.0..1.0"),
                    (Throttle, "-1.0..1.0 velocity fraction (signed)"),
                    (Speed, "Max linear speed override (m/s)"),
                    (Active, "False to lock the piston"),
                ],
            },
            Self::Rail => BlockKindSignalSchema {
                publish_properties: &[],
                subscribe_properties: &[],
                property_hints: &[],
            },
            Self::RailJunction => BlockKindSignalSchema {
                publish_properties: &[SwitchState, Status],
                subscribe_properties: &[SwitchState, Active],
                property_hints: &[
                    (SwitchState, "Junction branch (0..N)"),
                ],
            },
            Self::RailSignal => BlockKindSignalSchema {
                publish_properties: &[Status],
                subscribe_properties: &[Active],
                property_hints: &[],
            },
            Self::SignalConverter => BlockKindSignalSchema {
                publish_properties: &[],
                subscribe_properties: &[],
                property_hints: &[],
            },
            Self::Sensor => BlockKindSignalSchema {
                // Generic sensor — concrete sensor block kinds (PressureSensor,
                // VelocitySensor, LevelSensor) will replace this with single-
                // property schemas as they're added.
                publish_properties: &[Active, Pressure, Speed, Level, Status],
                subscribe_properties: &[Active],
                property_hints: &[],
            },
            Self::Computer => BlockKindSignalSchema {
                publish_properties: &[Active, Status, Text],
                subscribe_properties: &[Active, Throttle, SwitchState, Text],
                property_hints: &[
                    (Text, "Programmable display / message channel"),
                ],
            },
            Self::CruiseDrive => BlockKindSignalSchema {
                publish_properties: &[Boost, Status, Active],
                subscribe_properties: &[Throttle, Active],
                property_hints: &[
                    (Boost, "Output thrust multiplier when engaged"),
                    (Throttle, "Engagement throttle (>0.5 = engaged)"),
                ],
            },
            Self::FlightComputer => BlockKindSignalSchema {
                publish_properties: &[Active, Status],
                subscribe_properties: &[Active, Throttle],
                property_hints: &[
                    (Active, "Master enable for rotation damping"),
                ],
            },
            Self::HoverModule => BlockKindSignalSchema {
                publish_properties: &[Status, Active],
                subscribe_properties: &[Throttle, Active],
                property_hints: &[
                    (Throttle, "Vertical hover input"),
                ],
            },
            Self::Autopilot => BlockKindSignalSchema {
                publish_properties: &[Status, Active],
                subscribe_properties: &[Throttle, Active],
                property_hints: &[
                    (Active, "Engage / disengage autopilot"),
                ],
            },
            Self::WarpComputer => BlockKindSignalSchema {
                publish_properties: &[Status, Active, Text],
                subscribe_properties: &[Active, SwitchState],
                property_hints: &[
                    (Text, "Current target body name (server-authoritative)"),
                    (SwitchState, "Cycle (0) / Accept (1) / Cancel (2)"),
                ],
            },
            Self::EngineController => BlockKindSignalSchema {
                publish_properties: &[Active, Status],
                subscribe_properties: &[Active],
                property_hints: &[
                    (Active, "Master engine enable"),
                ],
            },
            Self::Terminal => BlockKindSignalSchema {
                // Phase A3 UX cleanup: Terminal has NO standalone
                // signal bindings. Its only configuration is the
                // Read/Write channels in the dedicated Terminal panel.
                // Same reasoning as Antenna — keep one source of truth
                // per block kind so players aren't asked twice "where
                // does this thing publish?".
                publish_properties: &[],
                subscribe_properties: &[],
                property_hints: &[],
            },
        }
    }
}
