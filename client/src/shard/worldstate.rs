//! WorldState fan-out: drain `NetEvent::WorldState` +
//! `NetEvent::SecondaryWorldState` into `PrimaryWorldState` and
//! `SecondaryWorldStates` resources so downstream systems (Phase 11+
//! camera pose, Phase 20 remote-entity tracking, HUD) read without
//! touching the raw event stream.

use std::collections::HashMap;

use bevy::prelude::*;

use voxeldust_core::client_message::WorldStateData;

use crate::net::{GameEvent, NetEvent};
use crate::shard::registry::ShardRegistrySet;

#[derive(SystemSet, Clone, Debug, Hash, Eq, PartialEq)]
pub struct WorldStateIngestSet;

/// Latest primary WorldState. Replaced every tick; downstream code
/// reads the current tick's authoritative game state.
#[derive(Resource, Default)]
pub struct PrimaryWorldState {
    pub latest: Option<WorldStateData>,
}

/// Latest WorldState per secondary shard-type. Keyed by shard_type
/// (u8) because the `SecondaryWorldState` NetEvent carries shard_type
/// but not seed — multiple secondaries of the same type are rare at
/// tick granularity (typically system-wide AOI).
#[derive(Resource, Default)]
pub struct SecondaryWorldStates {
    pub by_shard_type: HashMap<u8, WorldStateData>,
}

pub struct WorldStateIngestPlugin;

impl Plugin for WorldStateIngestPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<PrimaryWorldState>()
            .init_resource::<SecondaryWorldStates>()
            .configure_sets(Update, WorldStateIngestSet.after(ShardRegistrySet))
            .add_systems(
                Update,
                (
                    // Clear buffered ticks on shard swaps BEFORE ingesting
                    // this frame's WS — otherwise the new primary's tick
                    // (which may be lower than the old primary's) gets
                    // rejected by the monotonic-tick guard in `ingest_*`.
                    reset_on_shard_change,
                    (ingest_primary, ingest_secondary),
                )
                    .chain()
                    .in_set(WorldStateIngestSet),
            );
    }
}

/// On `Connected` (new primary) and `SecondaryDisconnected` (closed
/// secondary), drop the buffered WS for the affected shard so the
/// next WS we receive isn't rejected as "older than the previous
/// shard's last tick" by the monotonic guard.
fn reset_on_shard_change(
    mut events: MessageReader<GameEvent>,
    mut primary: ResMut<PrimaryWorldState>,
    mut secondary: ResMut<SecondaryWorldStates>,
) {
    for GameEvent(ev) in events.read() {
        match ev {
            NetEvent::Connected { .. } => {
                primary.latest = None;
            }
            NetEvent::SecondaryDisconnected { .. } => {
                // We don't get the shard_type on SecondaryDisconnected;
                // clear all secondaries so the next WS from any
                // remaining secondary is accepted. Cheap — ingest will
                // repopulate from the next packet.
                secondary.by_shard_type.clear();
            }
            _ => {}
        }
    }
}

/// WS arrives over UDP — packets can be reordered or arrive late. A
/// stale tick that overtakes a fresh one would overwrite `latest` with
/// older data; the next frame's `apply_worldstate_pose` would then place
/// the camera at the stale player position, producing a one-frame
/// position blink that snaps back when the next fresh WS lands. Drop
/// any WS whose tick is not newer than the one already buffered.
///
/// A wrap-aware comparison would matter at u64 saturation; for now we
/// reject equal-or-older ticks and let monotonic tick growth handle the
/// rest.
fn ingest_primary(
    mut events: MessageReader<GameEvent>,
    mut primary: ResMut<PrimaryWorldState>,
) {
    for GameEvent(ev) in events.read() {
        if let NetEvent::WorldState(ws) = ev {
            let stale = primary
                .latest
                .as_ref()
                .map(|prev| ws.tick <= prev.tick)
                .unwrap_or(false);
            if stale {
                continue;
            }
            primary.latest = Some(ws.clone());
        }
    }
}

fn ingest_secondary(
    mut events: MessageReader<GameEvent>,
    mut secondary: ResMut<SecondaryWorldStates>,
) {
    for GameEvent(ev) in events.read() {
        if let NetEvent::SecondaryWorldState { shard_type, ws } = ev {
            // Same monotonic-tick guard as `ingest_primary`. Stale UDP
            // packets on a secondary's stream would otherwise rotate /
            // translate that secondary's chunks (or its observable
            // entities used for cross-shard pose lookup) backward by
            // one frame, then snap forward on the next fresh tick.
            let stale = secondary
                .by_shard_type
                .get(shard_type)
                .map(|prev| ws.tick <= prev.tick)
                .unwrap_or(false);
            if stale {
                continue;
            }
            secondary.by_shard_type.insert(*shard_type, ws.clone());
        }
    }
}
