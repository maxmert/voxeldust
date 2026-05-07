//! Signal-related ECS components for functional block entities.
//!
//! Runtime binding structs use `ChannelId` for O(1) lookups.  String-based
//! channel names are resolved to IDs at the config/spawn boundary and live
//! only in `SignalChannelTable::name_to_id`.

use bevy_ecs::prelude::*;
use glam::DVec3;

use super::channel::ChannelId;
use super::converter::SignalRule;
use super::types::SignalProperty;

// ---------------------------------------------------------------------------
// Generic signal bindings (publish / subscribe)
// ---------------------------------------------------------------------------

/// A single publish binding: this block writes a property to a channel.
#[derive(Clone, Debug)]
pub struct PublishBinding {
    /// Resolved channel ID (O(1) table lookup).
    pub channel_id: ChannelId,
    /// Which property of this block to read and publish.
    pub property: SignalProperty,
}

/// A single subscribe binding: this block reads a channel and applies to a property.
#[derive(Clone, Debug)]
pub struct SubscribeBinding {
    /// Resolved channel ID (O(1) table lookup).
    pub channel_id: ChannelId,
    /// Which property of this block to drive from the channel value.
    pub property: SignalProperty,
}

/// Channels this functional block publishes to.
/// Attached to any functional block entity that produces signal data.
#[derive(Component, Default, Clone, Debug)]
pub struct SignalPublisher {
    pub bindings: Vec<PublishBinding>,
}

/// Channels this functional block subscribes to.
/// Attached to any functional block entity that consumes signal data.
#[derive(Component, Default, Clone, Debug)]
pub struct SignalSubscriber {
    pub bindings: Vec<SubscribeBinding>,
}

/// Configuration for a Signal Converter block — condition → action rules.
/// Only attached to entities whose FunctionalBlockKind == SignalConverter.
#[derive(Component, Default, Clone, Debug)]
pub struct SignalConverterConfig {
    pub rules: Vec<SignalRule>,
}

// ---------------------------------------------------------------------------
// Generic seat system — shard-agnostic, works anywhere blocks are supported
// ---------------------------------------------------------------------------

/// Physical input source type for a seat binding.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum SeatInputSource {
    /// Keyboard key or mouse button (binary: 0.0 released, 1.0 held).
    Key = 0,
    /// Horizontal mouse movement (spring-centered continuous axis).
    MouseMoveX = 1,
    /// Vertical mouse movement (spring-centered continuous axis).
    MouseMoveY = 2,
    /// Mouse scroll wheel (accumulative, persists between frames).
    ScrollWheel = 3,
}

impl SeatInputSource {
    pub fn from_u8(v: u8) -> Option<Self> {
        match v {
            0 => Some(Self::Key),
            1 => Some(Self::MouseMoveX),
            2 => Some(Self::MouseMoveY),
            3 => Some(Self::ScrollWheel),
            _ => None,
        }
    }

    pub fn label(self) -> &'static str {
        match self {
            Self::Key => "Key",
            Self::MouseMoveX => "Mouse X",
            Self::MouseMoveY => "Mouse Y",
            Self::ScrollWheel => "Scroll",
        }
    }
}

/// Key activation mode (only meaningful for `SeatInputSource::Key`).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum KeyMode {
    /// Value = 1.0 while held, 0.0 when released.
    Momentary = 0,
    /// Each press toggles between 0.0 and 1.0.
    Toggle = 1,
}

impl KeyMode {
    pub fn from_u8(v: u8) -> Option<Self> {
        match v {
            0 => Some(Self::Momentary),
            1 => Some(Self::Toggle),
            _ => None,
        }
    }

    pub fn label(self) -> &'static str {
        match self {
            Self::Momentary => "Hold",
            Self::Toggle => "Toggle",
        }
    }
}

/// Direction filter for mouse axis bindings.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum AxisDirection {
    /// Only positive values (right / up). Negative clamped to 0.
    Positive = 0,
    /// Only negative values, output made positive (left / down).
    Negative = 1,
    /// Full bipolar [-1, 1].
    Both = 2,
}

impl AxisDirection {
    pub fn from_u8(v: u8) -> Option<Self> {
        match v {
            0 => Some(Self::Positive),
            1 => Some(Self::Negative),
            2 => Some(Self::Both),
            _ => None,
        }
    }

    pub fn label(self) -> &'static str {
        match self {
            Self::Positive => "\u{2192} Positive",
            Self::Negative => "\u{2190} Negative",
            Self::Both => "\u{2194} Both",
        }
    }

    /// Apply direction filter to a raw bipolar value.
    pub fn apply(self, raw: f32) -> f32 {
        match self {
            Self::Positive => raw.max(0.0),
            Self::Negative => (-raw).max(0.0),
            Self::Both => raw,
        }
    }
}

/// A single input-to-channel binding in a seat.
#[derive(Clone, Debug)]
pub struct SeatInputBinding {
    /// Human-readable label (e.g., "Thrust Forward").
    pub label: String,
    /// What physical input triggers this binding.
    pub source: SeatInputSource,
    /// Key name for Key source (e.g., "KeyW", "Space", "MouseLeft"). Empty for axes/scroll.
    pub key_name: String,
    /// Key activation mode (only for Key source).
    pub key_mode: KeyMode,
    /// Direction filter (only for MouseMoveX/Y).
    pub axis_direction: AxisDirection,
    /// Target signal channel (resolved ID).
    pub channel_id: ChannelId,
    /// What property to publish.
    pub property: SignalProperty,
}

/// Seat channel mapping component. Maps physical inputs to signal channels.
/// Shard-agnostic — works in any shard where blocks are supported.
/// Attached to entities whose FunctionalBlockKind == Seat.
#[derive(Component, Clone, Debug)]
pub struct SeatChannelMapping {
    pub bindings: Vec<SeatInputBinding>,
    /// Channel to emit 1.0 when player is seated, 0.0 when not.
    pub seated_channel_id: Option<ChannelId>,
}

// ---------------------------------------------------------------------------
// Custom ship system block components — each fully independent
// ---------------------------------------------------------------------------

/// Flight computer — angular velocity damping when pilot input is near zero.
/// Each channel field is explicit — no implicit ordering.
#[derive(Component, Clone, Debug)]
pub struct FlightComputerState {
    pub yaw_cw_channel: Option<ChannelId>,
    pub yaw_ccw_channel: Option<ChannelId>,
    pub pitch_up_channel: Option<ChannelId>,
    pub pitch_down_channel: Option<ChannelId>,
    pub roll_cw_channel: Option<ChannelId>,
    pub roll_ccw_channel: Option<ChannelId>,
    pub toggle_channel: Option<ChannelId>,
    pub damping_gain: f32,
    pub dead_zone: f32,
    pub max_correction: f32,
    pub active: bool,
    pub prev_toggle_value: f32,
}

impl Default for FlightComputerState {
    fn default() -> Self {
        Self {
            yaw_cw_channel: None, yaw_ccw_channel: None,
            pitch_up_channel: None, pitch_down_channel: None,
            roll_cw_channel: None, roll_ccw_channel: None,
            toggle_channel: None,
            damping_gain: 0.6,
            dead_zone: 0.005,
            max_correction: 0.3,
            active: true,
            prev_toggle_value: 0.0,
        }
    }
}

/// Hover module — 6-DOF hover: attitude hold + gravity compensation + velocity damping.
#[derive(Component, Clone, Debug)]
pub struct HoverModuleState {
    // Thrust channels:
    pub thrust_forward_channel: Option<ChannelId>,
    pub thrust_reverse_channel: Option<ChannelId>,
    pub thrust_right_channel: Option<ChannelId>,
    pub thrust_left_channel: Option<ChannelId>,
    pub thrust_up_channel: Option<ChannelId>,
    pub thrust_down_channel: Option<ChannelId>,
    // Rotation channels:
    pub yaw_cw_channel: Option<ChannelId>,
    pub yaw_ccw_channel: Option<ChannelId>,
    pub pitch_up_channel: Option<ChannelId>,
    pub pitch_down_channel: Option<ChannelId>,
    pub roll_cw_channel: Option<ChannelId>,
    pub roll_ccw_channel: Option<ChannelId>,
    // Activation:
    pub activate_channel: Option<ChannelId>,
    pub cutoff_channel: Option<ChannelId>,
    // Runtime state:
    pub was_active: bool,
    pub captured_heading: DVec3,
    pub prev_velocity_local: DVec3,
}

impl Default for HoverModuleState {
    fn default() -> Self {
        Self {
            thrust_forward_channel: None, thrust_reverse_channel: None,
            thrust_right_channel: None, thrust_left_channel: None,
            thrust_up_channel: None, thrust_down_channel: None,
            yaw_cw_channel: None, yaw_ccw_channel: None,
            pitch_up_channel: None, pitch_down_channel: None,
            roll_cw_channel: None, roll_ccw_channel: None,
            activate_channel: None,
            cutoff_channel: None,
            was_active: false,
            captured_heading: DVec3::NEG_Z,
            prev_velocity_local: DVec3::ZERO,
        }
    }
}

/// Autopilot — target-tracking, publishes steering commands to rotation channels.
#[derive(Component, Clone, Debug)]
pub struct AutopilotBlockState {
    // Rotation channels (autopilot writes steering commands here):
    pub yaw_cw_channel: Option<ChannelId>,
    pub yaw_ccw_channel: Option<ChannelId>,
    pub pitch_up_channel: Option<ChannelId>,
    pub pitch_down_channel: Option<ChannelId>,
    pub roll_cw_channel: Option<ChannelId>,
    pub roll_ccw_channel: Option<ChannelId>,
    // Activation:
    pub engage_channel: Option<ChannelId>,
    // Runtime state:
    pub target_body_id: Option<u32>,
    pub pending_cmd: Option<(u32, u8)>,
    pub prev_engage_value: f32,
}

impl Default for AutopilotBlockState {
    fn default() -> Self {
        Self {
            yaw_cw_channel: None, yaw_ccw_channel: None,
            pitch_up_channel: None, pitch_down_channel: None,
            roll_cw_channel: None, roll_ccw_channel: None,
            engage_channel: None,
            target_body_id: None,
            pending_cmd: None,
            prev_engage_value: 0.0,
        }
    }
}

/// Warp computer — target selection and warp initiation.
#[derive(Component, Clone, Debug, Default)]
pub struct WarpComputerState {
    pub cycle_channel: Option<ChannelId>,
    pub accept_channel: Option<ChannelId>,
    pub cancel_channel: Option<ChannelId>,
    pub target_star_index: Option<u32>,
    pub pending_cmd: Option<u32>,
    pub prev_cycle_value: f32,
    pub prev_accept_value: f32,
    pub prev_cancel_value: f32,
}

/// Engine controller — master on/off toggle for all propulsion.
/// Rising edge on toggle channel flips engines_on. When off, zeros all managed channels.
#[derive(Component, Clone, Debug)]
pub struct EngineControllerState {
    // All channels zeroed when engines are off:
    pub thrust_forward_channel: Option<ChannelId>,
    pub thrust_reverse_channel: Option<ChannelId>,
    pub thrust_right_channel: Option<ChannelId>,
    pub thrust_left_channel: Option<ChannelId>,
    pub thrust_up_channel: Option<ChannelId>,
    pub thrust_down_channel: Option<ChannelId>,
    pub yaw_cw_channel: Option<ChannelId>,
    pub yaw_ccw_channel: Option<ChannelId>,
    pub pitch_up_channel: Option<ChannelId>,
    pub pitch_down_channel: Option<ChannelId>,
    pub roll_cw_channel: Option<ChannelId>,
    pub roll_ccw_channel: Option<ChannelId>,
    // Toggle:
    pub toggle_channel: Option<ChannelId>,
    pub engines_on: bool,
    pub prev_toggle_value: f32,
}

impl Default for EngineControllerState {
    fn default() -> Self {
        Self {
            thrust_forward_channel: None, thrust_reverse_channel: None,
            thrust_right_channel: None, thrust_left_channel: None,
            thrust_up_channel: None, thrust_down_channel: None,
            yaw_cw_channel: None, yaw_ccw_channel: None,
            pitch_up_channel: None, pitch_down_channel: None,
            roll_cw_channel: None, roll_ccw_channel: None,
            toggle_channel: None,
            engines_on: true,
            prev_toggle_value: 0.0,
        }
    }
}

// ---------------------------------------------------------------------------
// Phase 3E — Antenna and Listener block-bound state.
//
// Both blocks act as block-bound conveniences over the cross-shard grant
// infrastructure (which already works at the wire level via
// `RemoteSignalPublish` + `SignalSubscribe`). The blocks provide a stable
// in-world placement point + UI handle so a player doesn't have to bind a
// HUD widget to every cross-shard channel they care about — placing an
// Antenna pins an outgoing channel; placing a Listener pins an incoming one.
// ---------------------------------------------------------------------------

/// Antenna: bridges a Local channel on this shard to a remote
/// Phase D: bidirectional Antenna runtime state. Holds optional TX
/// side (forward local-channel values onto a Radio frequency) and
/// optional RX side (subscribe to a Radio frequency, mirror values
/// onto a local channel). Most antennas use both sides at one
/// frequency for full-duplex chat. Replaces the prior split
/// AntennaState (TX-only) + ListenerState (RX-only) components.
///
/// Either side's `local_channel_id == None` means "side unconfigured
/// yet" — the apply path resolves channel ids at config time. Either
/// side's `grant_id == None` means "open channel, no key required"
/// (CB-radio-style broadcast). Cross-shard ingress (`try_push_remote`)
/// validates the HMAC against the matching grant when set.
#[derive(Component, Clone, Debug, Default)]
pub struct AntennaState {
    pub tx: Option<AntennaTxSide>,
    pub rx: Option<AntennaRxSide>,
    /// Block owner session — used to look up grant keys in HeldGrants.
    /// Set at placement from `BlockOwnership`; never zero on a placed
    /// antenna.
    pub owner_session: u64,
    /// Master power: false ⇒ both sides paused (no forward, no mirror).
    /// Driven by the antenna's own Active subscriber binding.
    pub active: bool,
}

impl AntennaState {
    /// Antenna has at least one configured side?
    pub fn has_any_side(&self) -> bool {
        self.tx.is_some() || self.rx.is_some()
    }
}

/// TX side of an Antenna: the antenna SUBSCRIBES to `local_channel_id`
/// on this shard each tick, looks up its grant's HMAC key, stamps the
/// value with `next_sequence`, and forwards via QUIC to the remote
/// shard hosting the matching Radio channel at `frequency`. Same wire
/// path as the tablet `RemoteSignalPublish` flow, but block-driven.
#[derive(Clone, Debug, Default)]
pub struct AntennaTxSide {
    /// Local channel feeding the antenna. Antenna reads this each tick.
    pub local_channel_id: Option<ChannelId>,
    /// Radio frequency. Receiver-side filter on the target shard.
    pub frequency: u32,
    /// Optional HMAC grant id. None ⇒ open channel (no key required).
    pub grant_id: Option<u64>,
    /// Target shard the forwarded entries are addressed to. None ⇒
    /// orchestrator-routed via the relay's frequency-band table.
    pub remote_shard_id: Option<u64>,
    /// Monotonic outbound sequence counter for the publisher's half
    /// of the (channel, sender) replay window. Incremented on every
    /// successful frame ship.
    pub next_sequence: u64,
}

/// RX side of an Antenna: the antenna issues a `SignalSubscribe` for
/// the Radio channel at `frequency` (HMAC-stamped with its grant if
/// keyed; bare if open) and PUBLISHES values it receives on
/// `bridged_channel_id` (a local Radio-scope mirror) onto
/// `local_channel_id` (a Local-scope channel that in-shard subscribers
/// can wire to). The two-step "Radio-mirror → Local-publish" indirection
/// keeps the cross-shard auth surface contained — internal subscribers
/// don't need HMAC awareness.
#[derive(Clone, Debug, Default)]
pub struct AntennaRxSide {
    /// Local Local-scope channel that mirrors the bridged Radio value.
    /// Created at config time with `SignalScope::Local` so internal
    /// subscribers can wire to it like any native channel.
    pub local_channel_id: Option<ChannelId>,
    /// Local Radio-scope channel that receives forwarded entries from
    /// the publisher's shard. `try_push_remote` verifies the HMAC tag
    /// against this channel's signature (or grant), then push_pendings
    /// the value here; the antenna's RX-mirror system reads from this
    /// and writes to `local_channel_id`.
    pub bridged_channel_id: Option<ChannelId>,
    /// Radio frequency.
    pub frequency: u32,
    /// Optional HMAC grant id. None ⇒ open channel; no SignalSubscribe
    /// HMAC stamping required (the publisher's open-channel path
    /// accepts unauth subscriptions).
    pub grant_id: Option<u64>,
    /// The shard this antenna is subscribing TO (the publisher's shard).
    /// None ⇒ orchestrator-routed via the relay.
    pub remote_shard_id: Option<u64>,
    /// Tick number when our subscription on the publisher's side
    /// expires. Refreshed by the lease-renewal system before expiry.
    pub lease_until_tick: u64,
}

/// LEGACY type alias for the Phase 3E split-block state. Retained as
/// `ListenerState = AntennaState` so existing references in shard-
/// common's apply / listener-mirror systems compile during the Phase 3
/// pipeline rewrite. Removed in Phase 6 cleanup.
#[deprecated(note = "Use AntennaState — Listener is now an RX-only Antenna")]
pub type ListenerState = AntennaState;

/// Phase D: unified Terminal block runtime state. Subscribes (read)
/// AND publishes (write) media `Text` frames in one block. Either
/// side can be unused: read-only sign (`publish_channel.is_none()`),
/// input-only kiosk (`subscribe_channel.is_none()`), or full chat
/// panel (both). Replaces the prior split TextDisplayState +
/// KeyboardTerminalState components.
///
/// **Phase A1 shift**: media now flows through `ChannelMediaBuffer`
/// keyed by `ChannelId`, not via direct shard targeting. The Terminal
/// resolves its configured channel names to `ChannelId`s at apply
/// time and stores them here. The cross-shard routing (target shard,
/// grant) lives on the channel itself (its scope + auth + grants),
/// so the Terminal no longer carries `target_shard_id`/`grant_id` —
/// they were design smell that made same-shard chat impossible.
///
/// Read side: each tick `terminal_subscribe` reads frames from
/// `ChannelMediaBuffer[subscribe_channel]` and appends to
/// `recent_lines` (ring-buffered at `max_lines`).
///
/// Write side: each `KeyboardTerminalInput` event triggers
/// `terminal_publish` to push a frame to
/// `ChannelMediaBuffer[publish_channel]`. The same tick's
/// `terminal_subscribe` (for in-shard subscribers) and
/// `antenna_publish_media` (for cross-shard via Radio frequency) both
/// read it.
#[derive(Component, Clone, Debug, Default)]
pub struct TerminalState {
    /// Channel this terminal subscribes to. None = read side disabled
    /// (input-only kiosk). ChannelId is resolved at apply time from
    /// the configured channel name; the same-named channel must
    /// already exist or be created with Local scope by the apply path.
    pub subscribe_channel: Option<ChannelId>,
    /// Channel this terminal publishes to. None = write side disabled
    /// (read-only sign / status board).
    pub publish_channel: Option<ChannelId>,
    /// Most recent N inbound text lines (bounded scrollback).
    pub recent_lines: Vec<String>,
    /// Maximum scrollback lines retained.
    pub max_lines: u32,
    /// Block owner — used by future authorization paths and for
    /// engagement bookkeeping. Sourced from `BlockOwnership` at apply
    /// time.
    pub owner_session: u64,
    /// Monotonic outbound sequence counter for the publisher's half
    /// of the (channel, sender) replay window. Incremented on every
    /// successful frame ship; persisted across ticks so cross-shard
    /// replay rejects stale duplicates.
    pub next_sequence: u64,
    /// False = terminal powered down (display blank, keyboard locked).
    pub active: bool,
}

impl TerminalState {
    pub const DEFAULT_MAX_LINES: u32 = 64;

    /// Append an inbound line to scrollback. Drops when the terminal
    /// is muted; ring-buffers at `max_lines` capacity.
    pub fn push_line(&mut self, line: String) {
        if !self.active {
            return;
        }
        let cap = self.max_lines.max(1) as usize;
        if self.recent_lines.len() >= cap {
            self.recent_lines.remove(0);
        }
        self.recent_lines.push(line);
    }

    /// Read-side enabled?
    pub fn can_read(&self) -> bool {
        self.active && self.subscribe_channel.is_some()
    }

    /// Write-side enabled?
    pub fn can_write(&self) -> bool {
        self.active && self.publish_channel.is_some()
    }
}
