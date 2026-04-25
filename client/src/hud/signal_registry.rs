//! Client-side mirror of signal-graph values the server has authorised
//! this player to see. Populated from `WorldStateData.hud_signals` —
//! every tick the server sends a full snapshot of all HUD-relevant
//! channels (numeric + text), the drainer mirrors them into this
//! resource, and widgets read via `SignalRegistry::get(channel)`.
//!
//! Full-snapshot semantics (not delta): a channel absent from this
//! tick's `hud_signals` is NOT evicted — the previous value remains
//! valid. The server populates every channel each tick when it wants
//! live updates; if a server ever switches to delta semantics, add
//! eviction logic here. For MVP the snapshot pattern keeps widget
//! flicker-free between ticks where the server doesn't republish.

use std::collections::HashMap;

use bevy::prelude::*;

use voxeldust_core::client_message::{HudSignalValue, WorldStateData};
use voxeldust_core::signal::types::SignalProperty;

use crate::net::{GameEvent, NetEvent};

/// Current value of a signal channel. `Text` is carried natively so
/// string-valued signals (body names, warp targets, ship callsigns,
/// distant-object labels) are fully server-authoritative.
#[derive(Debug, Clone)]
pub enum SignalValue {
    Bool(bool),
    Float(f32),
    U8(u8),
    Text(String),
}

impl SignalValue {
    pub fn as_f32(&self) -> f32 {
        match self {
            SignalValue::Bool(b) => {
                if *b {
                    1.0
                } else {
                    0.0
                }
            }
            SignalValue::Float(f) => *f,
            SignalValue::U8(n) => *n as f32,
            SignalValue::Text(_) => 0.0,
        }
    }
    pub fn as_bool(&self) -> bool {
        match self {
            SignalValue::Bool(b) => *b,
            SignalValue::Float(f) => *f != 0.0,
            SignalValue::U8(n) => *n != 0,
            SignalValue::Text(s) => !s.is_empty(),
        }
    }
    pub fn as_u8(&self) -> u8 {
        match self {
            SignalValue::Bool(b) => *b as u8,
            SignalValue::Float(f) => *f as u8,
            SignalValue::U8(n) => *n,
            SignalValue::Text(_) => 0,
        }
    }
    pub fn as_text(&self) -> Option<&str> {
        match self {
            SignalValue::Text(s) => Some(s.as_str()),
            _ => None,
        }
    }
}

impl From<&HudSignalValue> for SignalValue {
    fn from(v: &HudSignalValue) -> Self {
        match v {
            HudSignalValue::Bool(b) => SignalValue::Bool(*b),
            HudSignalValue::Float(f) => SignalValue::Float(*f),
            HudSignalValue::State(s) => SignalValue::U8(*s),
            HudSignalValue::Text(s) => SignalValue::Text(s.clone()),
        }
    }
}

/// Registry keyed on `channel_name`. HUD widgets read via `get(channel)`.
#[derive(Resource, Default, Debug)]
pub struct SignalRegistry {
    pub by_channel: HashMap<String, RegisteredSignal>,
    /// Last server tick the registry was updated from. Widgets can
    /// use this for "stale signal" visual hints.
    pub last_tick: u64,
}

#[derive(Debug, Clone)]
pub struct RegisteredSignal {
    pub value: SignalValue,
    pub property: SignalProperty,
    /// Wall-clock instant when this value was received.
    pub received_at: std::time::Instant,
}

impl SignalRegistry {
    pub fn get(&self, channel: &str) -> Option<&RegisteredSignal> {
        self.by_channel.get(channel)
    }
}

pub struct SignalRegistryPlugin;

impl Plugin for SignalRegistryPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<SignalRegistry>()
            .add_systems(Update, drain_signal_broadcasts);
    }
}

/// Drain `NetEvent::WorldState` (primary only) into the
/// `SignalRegistry`. Snapshot-merge semantics: any channel present
/// in this tick's batch overwrites the registry's value; any channel
/// missing keeps its previous value.
///
/// **Primary-only**: every SHIP shard the client observes (own ship
/// as primary, every other ship as a SHIP secondary) publishes its
/// own `ship.speed` / `ship.thrust_tier` / etc. into `hud_signals`.
/// If we ingested secondaries too, the registry's `ship.speed` slot
/// would be overwritten by whichever shard's WS arrived last —
/// producing the "speed jumps between values every few seconds"
/// symptom when multiple ships are within AOI. The HUD shows the
/// player's own status, so primary-only is the correct scope.
/// (System-wide signals like nearest body / warp target are
/// published by the primary too: SHIP-primary inherits them via
/// SystemSceneUpdate caching, SYSTEM-primary publishes them
/// directly.)
fn drain_signal_broadcasts(
    mut events: MessageReader<GameEvent>,
    mut registry: ResMut<SignalRegistry>,
) {
    let now = std::time::Instant::now();
    for GameEvent(ev) in events.read() {
        if let NetEvent::WorldState(ws) = ev {
            ingest_ws(&mut registry, ws, now);
        }
    }
}

fn ingest_ws(
    registry: &mut SignalRegistry,
    ws: &WorldStateData,
    now: std::time::Instant,
) {
    if ws.hud_signals.is_empty() {
        return;
    }
    registry.last_tick = ws.tick;
    for entry in &ws.hud_signals {
        let property = SignalProperty::from_ordinal(entry.property)
            .unwrap_or(SignalProperty::Status);
        registry.by_channel.insert(
            entry.channel_name.clone(),
            RegisteredSignal {
                value: SignalValue::from(&entry.value),
                property,
                received_at: now,
            },
        );
    }
}
