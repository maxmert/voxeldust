//! Remote entity tracking (position-only, no rendering).
//!
//! Aggregates `ObservableEntity[]` from primary + every secondary's
//! WorldState into per-kind resources: `RemotePlayers`, `RemoteShips`,
//! `RemoteDebris`. Dedupes across shards by `entity_id`.
//!
//! **No placeholder rendering** (Design Principle #2 +
//! `feedback_no_placeholder_rendering`). Ships visible in-AOI already
//! render via their own SHIP secondary's chunk stream (Phase 5 +
//! Phase 16 transforms). Avatars are not rendered at all until a real
//! rigged mesh lands in a future plan. Debris will render via a
//! DEBRIS shard's chunks once that shard-type is declared server-side.
//!
//! This resource is the data source for:
//!   * Future radar UIs (Phase 21 HUD layer).
//!   * Cross-shard interaction range hints.
//!   * Thruster-signature detection mechanics later on.

use std::collections::HashMap;

use bevy::prelude::*;
use glam::{DQuat, DVec3};

use voxeldust_core::client_message::{EntityKind, ObservableEntityData};

use crate::shard::{
    PrimaryShard, PrimaryWorldState, SecondaryWorldStates, ShardKey, WorldStateIngestSet,
};

#[derive(SystemSet, Clone, Eq, PartialEq, Hash, Debug)]
pub struct RemoteEntitiesSet;

/// Position + velocity snapshot for one remote entity, current tick.
#[derive(Debug, Clone)]
pub struct RemoteEntity {
    pub entity_id: u64,
    pub kind: EntityKind,
    pub position: DVec3,
    pub rotation: DQuat,
    pub velocity: DVec3,
    /// Which shard this entity was observed on. For ships this is the
    /// authoritative SHIP shard id; for players on planet surfaces
    /// this is the PLANET shard id; etc.
    pub shard: ShardKey,
    pub name: String,
    pub health: f32,
    pub shield: f32,
}

/// Remote players (EVA + grounded + seated, excluding the own player).
#[derive(Resource, Default)]
pub struct RemotePlayers {
    pub by_id: HashMap<u64, RemoteEntity>,
}

/// Remote ships (excluding own-ship when we're on it).
#[derive(Resource, Default)]
pub struct RemoteShips {
    pub by_id: HashMap<u64, RemoteEntity>,
}

/// Remote debris — populated when a DEBRIS shard-type is declared
/// server-side and starts emitting `ObservableEntity` with a debris
/// kind. Today the kind enum doesn't distinguish debris, so this map
/// stays empty; kept as a forward-compatible resource so future
/// DEBRIS shard support is a one-line addition in the filter below.
#[derive(Resource, Default)]
pub struct RemoteDebris {
    pub by_id: HashMap<u64, RemoteEntity>,
}

pub struct RemoteEntitiesPlugin;

impl Plugin for RemoteEntitiesPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<RemotePlayers>()
            .init_resource::<RemoteShips>()
            .init_resource::<RemoteDebris>()
            .configure_sets(Update, RemoteEntitiesSet.after(WorldStateIngestSet))
            .add_systems(Update, track_remote_entities.in_set(RemoteEntitiesSet));
    }
}

fn track_remote_entities(
    primary: Res<PrimaryShard>,
    primary_ws: Res<PrimaryWorldState>,
    secondary_ws: Res<SecondaryWorldStates>,
    mut players: ResMut<RemotePlayers>,
    mut ships: ResMut<RemoteShips>,
    mut debris: ResMut<RemoteDebris>,
) {
    // Rebuild from scratch each tick — ObservableEntity broadcasts are
    // authoritative full-snapshots at AOI scope, not deltas. Clearing
    // + re-populating guarantees stale entries (entity left AOI) are
    // evicted immediately.
    players.by_id.clear();
    ships.by_id.clear();
    debris.by_id.clear();

    if let Some(ws) = primary_ws.latest.as_ref() {
        if let Some(key) = primary.current {
            ingest(ws, key, &mut players, &mut ships, &mut debris);
        }
    }
    for (&shard_type, ws) in &secondary_ws.by_shard_type {
        // We don't know the exact `seed` of the secondary from its
        // WorldState (wire format collapses to shard_type). Use
        // shard_type + 0 as a placeholder ShardKey; downstream
        // consumers who need the authoritative shard id should cross-
        // reference `Secondaries.runtimes`.
        let placeholder = ShardKey {
            shard_type,
            seed: 0,
        };
        ingest(ws, placeholder, &mut players, &mut ships, &mut debris);
    }
}

fn ingest(
    ws: &voxeldust_core::client_message::WorldStateData,
    observer: ShardKey,
    players: &mut RemotePlayers,
    ships: &mut RemoteShips,
    debris: &mut RemoteDebris,
) {
    for e in &ws.entities {
        if e.is_own {
            continue;
        }
        let remote = make_remote(e, observer);
        match e.kind {
            EntityKind::Ship => {
                ships.by_id.insert(e.entity_id, remote);
            }
            EntityKind::EvaPlayer
            | EntityKind::GroundedPlayer
            | EntityKind::Seated => {
                players.by_id.insert(e.entity_id, remote);
            }
            // Future DEBRIS kinds plug in here without a core change.
        }
        // Silence "unused variable" when the enum only has 4 variants
        // currently; avoids `_` in match arms that would suppress
        // future variants from triggering a compile warning.
        let _ = debris;
    }
}

fn make_remote(e: &ObservableEntityData, observer: ShardKey) -> RemoteEntity {
    RemoteEntity {
        entity_id: e.entity_id,
        kind: e.kind,
        position: DVec3::new(e.position.x, e.position.y, e.position.z),
        rotation: DQuat::from_xyzw(e.rotation.x, e.rotation.y, e.rotation.z, e.rotation.w),
        velocity: DVec3::new(e.velocity.x, e.velocity.y, e.velocity.z),
        shard: if e.shard_id != 0 {
            ShardKey {
                shard_type: e.shard_type,
                seed: e.shard_id,
            }
        } else {
            observer
        },
        name: e.name.clone(),
        health: e.health,
        shield: e.shield,
    }
}
