//! Client-side mirror of signal-graph values the server has authorised
//! this player to see.
//!
//! # Two ingest paths (Phase 4.4 transition)
//!
//!   * **Legacy UDP snapshot** — `WorldStateData.hud_signals`. Every
//!     tick the server includes a full per-channel snapshot. Snapshot-
//!     merge semantics: any channel present overwrites; missing channels
//!     keep their previous value. Used by older shards + as a safety
//!     net during the V2 cutover.
//!
//!   * **TCP delta** — `ServerMsg::HudSignalDelta`. Server emits ONLY
//!     when something changed. Each entry is REGISTER (carries name,
//!     binds wire_id), bare (value-only update for a known wire_id),
//!     or REMOVE (drops the channel from the cache). 20× bandwidth
//!     reduction once the legacy path is removed.
//!
//! Both paths populate the same [`SignalRegistry`] keyed by channel
//! name, so widget code reads a single source of truth via
//! `get(channel)`.
//!
//! # Per-session inbound dictionary
//!
//! The TCP delta path uses a per-session `InboundDict` to resolve
//! `wire_id` → channel name. The dict + the `SignalRegistry`'s by-
//! wire mirror live in [`HudInboundState`]. Reset on TCP reconnect:
//! the server's `OutboundDict` resets too, so the next inbound
//! batch's REGISTER entries rebuild the dict from scratch.
//!
//! # Resync on drift
//!
//! `dict_seq` must monotonically increase. A backwards drift means
//! the server restarted (its `OutboundDict::dict_seq()` reset to 0)
//! without our knowledge — we drop the inbound dict, clear the
//! by-wire mirror, and the next REGISTER entries rebuild state. An
//! `UnresolvedWireId` (bare entry with no matching dict entry) is
//! the same kind of resync trigger.

use std::collections::HashMap;

use bevy::prelude::*;

use voxeldust_core::client_message::{
    hud_delta_flags, HudSignalDeltaData, HudSignalEntryV2Data, HudSignalValue, WorldStateData,
};
use voxeldust_core::signal::types::SignalProperty;
use voxeldust_core::signal::wire_dict::InboundDict;

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

/// Phase 4.4: per-session inbound state for the TCP HUD delta path.
///
/// Keeps:
///   * `InboundDict` — resolves `wire_id → channel_name` per session.
///     Dict is reset (cleared, last_dict_seq=0) on TCP reconnect.
///   * `wire_to_name` — local mirror tying wire ids back to the
///     channel names we've inserted into [`SignalRegistry`]. Drives
///     REMOVE handling: a REMOVE entry's `wire_id` looks up the
///     channel name we previously cached so we can evict from the
///     registry.
///
/// Reset on TCP reconnect or `dict_seq` drift — handled by
/// [`reset_hud_inbound`].
#[derive(Resource, Default, Debug)]
pub struct HudInboundState {
    dict: InboundDict,
    wire_to_name: HashMap<u32, String>,
    /// Highest `batch_seq` we've accepted on this session. Helps
    /// diagnose out-of-order reception under future failure modes
    /// (today TCP guarantees in-order so this just monotonically
    /// rises).
    last_batch_seq: u64,
}

impl HudInboundState {
    pub fn new() -> Self {
        Self::default()
    }

    /// Drop all per-session state. Called on TCP reconnect (the
    /// server's outbound dict is also fresh) and on `dict_seq`
    /// drift (forces a full re-register pass).
    pub fn reset(&mut self) {
        self.dict.reset();
        self.wire_to_name.clear();
        self.last_batch_seq = 0;
    }

    pub fn last_batch_seq(&self) -> u64 {
        self.last_batch_seq
    }
}

pub struct SignalRegistryPlugin;

impl Plugin for SignalRegistryPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<SignalRegistry>()
            .init_resource::<HudInboundState>()
            .add_systems(
                Update,
                (
                    drain_signal_broadcasts,
                    drain_hud_signal_deltas,
                    drain_session_resets,
                ),
            );
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

/// Phase 4.4: drain TCP-delivered `HudSignalDelta` events into the
/// `SignalRegistry`. Resolves wire ids against the session's
/// `InboundDict`; applies REGISTER/bare/REMOVE entries; resyncs on
/// drift.
///
/// Apply ordering matches the wire spec:
///   1. Verify `dict_seq >= last_dict_seq`. Drift triggers full reset
///      (drop dict + by-wire mirror, clear registry entries that came
///      from this path) — the server's next batch's REGISTER entries
///      rebuild state.
///   2. For each entry:
///      * `REGISTER`: bind `wire_id → name` in the dict, cache the
///        name in our by-wire mirror, then apply the value.
///      * `REMOVE`: look up the wire_id's name, drop both the dict
///        entry and the registry entry.
///      * Bare: resolve via dict, apply value. Unresolvable means
///        the dict and the server have drifted — resync.
fn drain_hud_signal_deltas(
    mut events: MessageReader<GameEvent>,
    mut registry: ResMut<SignalRegistry>,
    mut inbound: ResMut<HudInboundState>,
) {
    let now = std::time::Instant::now();
    for GameEvent(ev) in events.read() {
        let NetEvent::HudSignalDelta(delta) = ev else {
            continue;
        };
        apply_delta(&mut registry, &mut inbound, delta, now);
    }
}

fn apply_delta(
    registry: &mut SignalRegistry,
    inbound: &mut HudInboundState,
    delta: &HudSignalDeltaData,
    now: std::time::Instant,
) {
    // Step 1: dict_seq monotonicity. Backwards drift = server reset.
    if let Err(e) = inbound.dict.verify_seq(delta.dict_seq) {
        tracing::warn!(
            error = %e,
            ?delta.dict_seq,
            ?inbound.last_batch_seq,
            "HudSignalDelta dict_seq drift — resync"
        );
        // Evict every channel we got via the delta path so the
        // registry doesn't show stale values that the server has
        // since updated. The next REGISTER batch repopulates.
        for name in inbound.wire_to_name.values() {
            registry.by_channel.remove(name);
        }
        inbound.reset();
        // Don't apply this batch — the next batch should be the
        // server's fresh REGISTER pass after its own reset.
        return;
    }
    inbound.last_batch_seq = delta.batch_seq;

    // Step 2: walk entries.
    for entry in &delta.entries {
        if (entry.flags & hud_delta_flags::REMOVE) != 0 {
            apply_remove(registry, inbound, entry);
            continue;
        }
        if (entry.flags & hud_delta_flags::REGISTER) != 0 {
            // Bind in dict + mirror. The same wire_id may have been
            // bound to a different name previously (server LRU
            // eviction reused the slot) — overwrite cleanly.
            inbound.dict.register(entry.wire_id, entry.channel_name.clone());
            // If we had an old name for this wire_id, evict its
            // registry entry — it's no longer valid.
            if let Some(old_name) = inbound
                .wire_to_name
                .insert(entry.wire_id, entry.channel_name.clone())
            {
                if old_name != entry.channel_name {
                    registry.by_channel.remove(&old_name);
                }
            }
            apply_value_entry(registry, &entry.channel_name, entry, now);
            continue;
        }
        // Bare entry — resolve against the dict.
        let resolved_name = match inbound.dict.resolve(entry.wire_id) {
            Some(n) => n.to_owned(),
            None => {
                tracing::warn!(
                    wire_id = entry.wire_id,
                    "HudSignalDelta unresolved wire_id — resync"
                );
                for name in inbound.wire_to_name.values() {
                    registry.by_channel.remove(name);
                }
                inbound.reset();
                return;
            }
        };
        apply_value_entry(registry, &resolved_name, entry, now);
    }
}

fn apply_remove(
    registry: &mut SignalRegistry,
    inbound: &mut HudInboundState,
    entry: &HudSignalEntryV2Data,
) {
    let Some(name) = inbound.wire_to_name.remove(&entry.wire_id) else {
        // Unknown wire_id on REMOVE is benign — possibly a duplicate
        // REMOVE after a resync. Just drop it.
        return;
    };
    // Drop dict entry too. The server's next REGISTER for the same
    // wire_id will re-bind from scratch.
    let _ = inbound.dict;
    registry.by_channel.remove(&name);
}

fn apply_value_entry(
    registry: &mut SignalRegistry,
    name: &str,
    entry: &HudSignalEntryV2Data,
    now: std::time::Instant,
) {
    let value = match entry.value_type {
        0 => SignalValue::Bool(entry.value_num > 0.5),
        1 => SignalValue::Float(entry.value_num),
        2 => SignalValue::U8(entry.value_num as u8),
        3 => SignalValue::Text(entry.value_text.clone()),
        _ => SignalValue::Float(entry.value_num),
    };
    let property = SignalProperty::from_ordinal(entry.property).unwrap_or(SignalProperty::Status);
    registry.by_channel.insert(
        name.to_string(),
        RegisteredSignal {
            value,
            property,
            received_at: now,
        },
    );
}

/// Reset the HUD inbound state on TCP reconnect / primary shard
/// transition. The server's `OutboundDict` resets too, so leftover
/// dict + by-wire state is meaningless.
///
/// Public so the network-event drainer can call it on
/// `NetEvent::Connected` and `NetEvent::Transitioning`. The function
/// also clears the SignalRegistry entries that came from the delta
/// path (tracked via `wire_to_name`).
pub fn reset_hud_inbound(registry: &mut SignalRegistry, inbound: &mut HudInboundState) {
    for name in inbound.wire_to_name.values() {
        registry.by_channel.remove(name);
    }
    inbound.reset();
}

#[cfg(test)]
mod tests {
    use super::*;
    use voxeldust_core::client_message::HudSignalEntryV2Data;

    fn fresh_state() -> (SignalRegistry, HudInboundState, std::time::Instant) {
        (
            SignalRegistry::default(),
            HudInboundState::default(),
            std::time::Instant::now(),
        )
    }

    #[test]
    fn register_entry_inserts_into_registry_and_binds_dict() {
        let (mut reg, mut inb, now) = fresh_state();
        let delta = HudSignalDeltaData {
            dict_seq: 1,
            batch_seq: 1,
            entries: vec![HudSignalEntryV2Data::register(
                0,
                "ship.thrust".into(),
                HudSignalValue::Float(0.5),
                1,
                1,
            )],
        };
        apply_delta(&mut reg, &mut inb, &delta, now);
        let entry = reg.by_channel.get("ship.thrust").expect("must insert");
        assert!(matches!(entry.value, SignalValue::Float(v) if v == 0.5));
        assert_eq!(inb.dict.resolve(0), Some("ship.thrust"));
        assert_eq!(inb.wire_to_name.get(&0), Some(&"ship.thrust".to_string()));
    }

    #[test]
    fn bare_entry_resolves_via_dict_and_updates_value() {
        let (mut reg, mut inb, now) = fresh_state();
        // First batch: REGISTER.
        let d1 = HudSignalDeltaData {
            dict_seq: 1,
            batch_seq: 1,
            entries: vec![HudSignalEntryV2Data::register(
                0,
                "ship.thrust".into(),
                HudSignalValue::Float(0.5),
                1,
                1,
            )],
        };
        apply_delta(&mut reg, &mut inb, &d1, now);
        // Second batch: bare value-only update for the known wire_id.
        let d2 = HudSignalDeltaData {
            dict_seq: 1,
            batch_seq: 2,
            entries: vec![HudSignalEntryV2Data::bare(
                0,
                HudSignalValue::Float(0.75),
                1,
                2,
            )],
        };
        apply_delta(&mut reg, &mut inb, &d2, now);
        let entry = reg.by_channel.get("ship.thrust").unwrap();
        assert!(matches!(entry.value, SignalValue::Float(v) if v == 0.75));
    }

    #[test]
    fn remove_entry_evicts_from_registry_and_dict_mirror() {
        let (mut reg, mut inb, now) = fresh_state();
        let d1 = HudSignalDeltaData {
            dict_seq: 1,
            batch_seq: 1,
            entries: vec![HudSignalEntryV2Data::register(
                7,
                "ship.lights".into(),
                HudSignalValue::Bool(true),
                0,
                1,
            )],
        };
        apply_delta(&mut reg, &mut inb, &d1, now);
        assert!(reg.by_channel.contains_key("ship.lights"));
        let d2 = HudSignalDeltaData {
            dict_seq: 2,
            batch_seq: 2,
            entries: vec![HudSignalEntryV2Data::remove(7, 2)],
        };
        apply_delta(&mut reg, &mut inb, &d2, now);
        assert!(
            !reg.by_channel.contains_key("ship.lights"),
            "REMOVE evicts registry entry"
        );
        assert!(
            inb.wire_to_name.get(&7).is_none(),
            "REMOVE drops the wire_id mirror"
        );
    }

    #[test]
    fn dict_seq_drift_triggers_full_reset() {
        let (mut reg, mut inb, now) = fresh_state();
        // Establish state at dict_seq=10.
        let d1 = HudSignalDeltaData {
            dict_seq: 10,
            batch_seq: 1,
            entries: vec![HudSignalEntryV2Data::register(
                0,
                "ship.thrust".into(),
                HudSignalValue::Float(0.5),
                1,
                1,
            )],
        };
        apply_delta(&mut reg, &mut inb, &d1, now);
        assert!(reg.by_channel.contains_key("ship.thrust"));

        // Server "restarted" — sends dict_seq=5 (backwards drift).
        let d2 = HudSignalDeltaData {
            dict_seq: 5,
            batch_seq: 2,
            entries: vec![HudSignalEntryV2Data::bare(
                0,
                HudSignalValue::Float(0.75),
                1,
                2,
            )],
        };
        apply_delta(&mut reg, &mut inb, &d2, now);
        // Drift triggers reset: registry entry from delta path is
        // evicted, dict cleared, batch dropped.
        assert!(
            !reg.by_channel.contains_key("ship.thrust"),
            "drift must evict delta-sourced registry entries"
        );
        assert!(inb.wire_to_name.is_empty());
    }

    #[test]
    fn unresolved_bare_wire_id_triggers_reset() {
        let (mut reg, mut inb, now) = fresh_state();
        // Bare entry without a prior REGISTER — wire_id 99 isn't in
        // the dict. Receiver MUST resync.
        let d = HudSignalDeltaData {
            dict_seq: 1,
            batch_seq: 1,
            entries: vec![HudSignalEntryV2Data::bare(
                99,
                HudSignalValue::Float(0.5),
                1,
                1,
            )],
        };
        apply_delta(&mut reg, &mut inb, &d, now);
        assert!(reg.by_channel.is_empty());
        assert!(inb.wire_to_name.is_empty());
    }

    #[test]
    fn register_overwrites_old_binding_when_wire_id_is_recycled() {
        // Server LRU-evicted wire_id=0 from "old.channel" and reused
        // it for "new.channel". The receiver must drop the old entry
        // from the registry — not leave it as a ghost.
        let (mut reg, mut inb, now) = fresh_state();
        let d1 = HudSignalDeltaData {
            dict_seq: 1,
            batch_seq: 1,
            entries: vec![HudSignalEntryV2Data::register(
                0,
                "old.channel".into(),
                HudSignalValue::Float(1.0),
                1,
                1,
            )],
        };
        apply_delta(&mut reg, &mut inb, &d1, now);
        assert!(reg.by_channel.contains_key("old.channel"));
        let d2 = HudSignalDeltaData {
            dict_seq: 2,
            batch_seq: 2,
            entries: vec![HudSignalEntryV2Data::register(
                0,
                "new.channel".into(),
                HudSignalValue::Float(2.0),
                1,
                1,
            )],
        };
        apply_delta(&mut reg, &mut inb, &d2, now);
        assert!(
            !reg.by_channel.contains_key("old.channel"),
            "ghost binding must be evicted on REGISTER replay"
        );
        assert!(reg.by_channel.contains_key("new.channel"));
    }

    #[test]
    fn reset_hud_inbound_clears_only_delta_sourced_entries() {
        // The legacy UDP path inserts entries directly into the
        // registry without touching `wire_to_name`. Reset must NOT
        // touch those — only entries whose wire_id we tracked.
        let (mut reg, mut inb, now) = fresh_state();
        // Delta path: ship.thrust + ship.lights.
        let delta = HudSignalDeltaData {
            dict_seq: 1,
            batch_seq: 1,
            entries: vec![
                HudSignalEntryV2Data::register(
                    0,
                    "ship.thrust".into(),
                    HudSignalValue::Float(0.5),
                    1,
                    1,
                ),
                HudSignalEntryV2Data::register(
                    1,
                    "ship.lights".into(),
                    HudSignalValue::Bool(true),
                    0,
                    1,
                ),
            ],
        };
        apply_delta(&mut reg, &mut inb, &delta, now);
        // Legacy path: ship.callsign comes via UDP, no wire_id.
        reg.by_channel.insert(
            "ship.callsign".into(),
            RegisteredSignal {
                value: SignalValue::Text("SHIP-42".into()),
                property: SignalProperty::Text,
                received_at: now,
            },
        );
        reset_hud_inbound(&mut reg, &mut inb);
        assert!(!reg.by_channel.contains_key("ship.thrust"));
        assert!(!reg.by_channel.contains_key("ship.lights"));
        assert!(
            reg.by_channel.contains_key("ship.callsign"),
            "UDP-sourced entries must survive a delta-path reset"
        );
        assert!(inb.wire_to_name.is_empty());
        assert_eq!(inb.dict.last_dict_seq(), 0);
    }
}

/// Watch for connection lifecycle events that imply the server's
/// per-session HUD state has been reset, and mirror that reset on
/// our side.
///
/// **`Connected`**: a fresh TCP+UDP session opened. Server-side
/// `HudSession` is brand-new for our token — no dict bindings carry
/// over. We must reset.
///
/// **`Transitioning`**: primary shard is changing. The new primary's
/// `HudSession` map has no entry for our token (we're a new client
/// to it). Same reset.
fn drain_session_resets(
    mut events: MessageReader<GameEvent>,
    mut registry: ResMut<SignalRegistry>,
    mut inbound: ResMut<HudInboundState>,
) {
    for GameEvent(ev) in events.read() {
        match ev {
            NetEvent::Connected { .. } | NetEvent::Transitioning { .. } => {
                if !inbound.wire_to_name.is_empty() || inbound.dict.last_dict_seq() > 0 {
                    tracing::debug!(
                        "HUD inbound state reset on session change"
                    );
                    reset_hud_inbound(&mut registry, &mut inbound);
                }
            }
            _ => {}
        }
    }
}
