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
                (ingest_primary, ingest_secondary).in_set(WorldStateIngestSet),
            );
    }
}

fn ingest_primary(
    mut events: MessageReader<GameEvent>,
    mut primary: ResMut<PrimaryWorldState>,
) {
    for GameEvent(ev) in events.read() {
        if let NetEvent::WorldState(ws) = ev {
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
            secondary.by_shard_type.insert(*shard_type, ws.clone());
        }
    }
}
