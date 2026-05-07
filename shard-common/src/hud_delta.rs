//! Phase 4.4 — generic per-session HUD signal delta emitter.
//!
//! The session-management + diff + TCP-emit logic is shared across
//! every shard that hosts Players. The per-shard variations (which
//! resources feed the snapshot, what the local newtype `SessionId`
//! looks like) stay in the shard binary; this module owns the
//! cluster-stable wire path.
//!
//! # The per-shard responsibility
//!
//! 1. Build the current per-tick snapshot of HUD-visible channel
//!    state, expressed as `Vec<(name, HudSignalValueRepr, property)>`.
//!    Ship-shard combines `ship.*` auto-publishes with the channel
//!    table; planet-shard just dumps the channel table.
//!
//! 2. Enumerate the live `SessionToken`s for connected players (each
//!    shard has its own `Player` ECS component + `SessionId` newtype,
//!    so a shard-common Bevy system would not satisfy the borrow
//!    checker without per-shard glue).
//!
//! 3. Call [`flush_hud_deltas`] with both. It walks every session,
//!    runs `HudSession::produce_delta`, and ships non-empty results
//!    over TCP via `ServerMsg::HudSignalDelta`. Empty deltas are
//!    skipped (no TCP send when nothing changed) — the steady-state
//!    common case.
//!
//! # Lifecycle
//!
//! `HudSessionMap` is a `Bevy` resource owned by each shard. Sessions
//! are lazy-inserted on first contact + per-tick pruned to drop
//! orphans (the caller passes the full live session set; entries not
//! in that set get evicted). No explicit disconnect-handler plumbing
//! needed — the prune is the cleanup.

use std::collections::{HashMap, HashSet};

use bevy_ecs::prelude::Resource;

use voxeldust_core::client_message::{
    HudSignalDeltaData, HudSignalEntryV2Data, ServerMsg,
};
use voxeldust_core::shard_types::SessionToken;
use voxeldust_core::signal::hud_session::{HudSession, HudSignalValueRepr};

use crate::harness::NetworkBridge;

/// Per-shard map of `SessionToken → HudSession`. Lazy-inserted on
/// first emit; per-tick pruned to drop sessions no longer in the
/// live set.
#[derive(Resource, Default)]
pub struct HudSessionMap {
    sessions: HashMap<SessionToken, HudSession>,
}

impl HudSessionMap {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn live_count(&self) -> usize {
        self.sessions.len()
    }
}

/// Drive every live session's delta + ship non-empty results to
/// clients over TCP.
///
/// **Ordering**: caller is responsible for invoking this at most
/// once per shard tick, after the snapshot has been built. The
/// function does NOT enforce any cadence — call it from a 20Hz
/// Bevy system to match server tick rate.
///
/// **Backpressure**: the TCP send is fire-and-forget via
/// `tokio::spawn` so a slow client never stalls the per-tick
/// schedule. Failed sends log at `debug!` (likely just a
/// disconnected client).
pub fn flush_hud_deltas<I>(
    bridge: &NetworkBridge,
    session_map: &mut HudSessionMap,
    snapshot: &[(String, HudSignalValueRepr, u8)],
    live_sessions: I,
) where
    I: IntoIterator<Item = SessionToken>,
{
    // Walk live sessions: lazy-insert on first contact, run produce_delta,
    // collect non-empty results into a per-(token, msg) dispatch list.
    let mut seen: HashSet<SessionToken> = HashSet::new();
    let mut to_send: Vec<(SessionToken, ServerMsg)> = Vec::new();

    for token in live_sessions {
        seen.insert(token);
        let session = session_map.sessions.entry(token).or_insert_with(HudSession::new);
        let entries = session.produce_delta(snapshot);
        if entries.is_empty() {
            continue;
        }
        // Convert HudDeltaEntry → wire-shaped HudSignalEntryV2Data.
        // The two types are isomorphic; the split exists so `core`'s
        // hud_session module doesn't depend on `client_message`'s
        // ordering details. Cheap conversion.
        let wire_entries: Vec<HudSignalEntryV2Data> = entries
            .into_iter()
            .map(|e| HudSignalEntryV2Data {
                flags: e.flags,
                wire_id: e.wire_id,
                channel_name: e.channel_name,
                value_type: e.value.type_code,
                value_num: e.value.num,
                value_text: e.value.text,
                property: e.property,
                seq: e.seq,
            })
            .collect();
        let msg = ServerMsg::HudSignalDelta(HudSignalDeltaData {
            dict_seq: session.dict_seq(),
            batch_seq: session.batch_seq(),
            entries: wire_entries,
        });
        to_send.push((token, msg));
    }

    // Prune sessions that disconnected since last call. Lazy cleanup
    // — no explicit on-disconnect hook needed.
    session_map.sessions.retain(|tok, _| seen.contains(tok));

    if to_send.is_empty() {
        return;
    }

    // TCP fan-out via `client_registry.send_tcp` — same fire-and-forget
    // pattern used by `apply_grant_create` etc. Deferred to a tokio
    // task so we don't block the per-tick schedule on socket writes.
    let cr = bridge.client_registry.clone();
    tokio::spawn(async move {
        let reg = cr.read().await;
        for (token, msg) in to_send {
            if let Err(e) = reg.send_tcp(token, &msg).await {
                tracing::debug!(
                    session = token.0,
                    %e,
                    "HudSignalDelta TCP send failed (client likely disconnected)"
                );
            }
        }
    });
}

/// Helper for shards whose HUD snapshot is "every channel in the
/// signal table." Most production shards layer ship-specific entries
/// (`ship.speed`, etc.) ON TOP of this base; planet-shard uses just
/// this.
///
/// Appends to `out` so callers can build composite snapshots with a
/// single allocation.
pub fn append_channel_table_entries(
    channels: &voxeldust_core::signal::SignalChannelTable,
    out: &mut Vec<(String, HudSignalValueRepr, u8)>,
) {
    use voxeldust_core::signal::types::{SignalProperty, SignalValue as CoreSignalValue};
    for (name, _id, value) in channels.iter_all() {
        let (repr, property) = match value {
            CoreSignalValue::Bool(b) => (HudSignalValueRepr::bool(b), SignalProperty::Active),
            CoreSignalValue::Float(f) => (HudSignalValueRepr::float(f), SignalProperty::Throttle),
            CoreSignalValue::State(s) => {
                (HudSignalValueRepr::state(s), SignalProperty::SwitchState)
            }
        };
        out.push((name.to_string(), repr, property.as_ordinal()));
    }
}
