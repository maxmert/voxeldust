//! Shared ECS resource types used by server shards and client.

use std::collections::HashMap;

use bevy_ecs::prelude::*;

use crate::shard_types::SessionToken;

/// Tick counter (incremented each 20Hz tick).
#[derive(Resource, Default)]
pub struct TickCounter(pub u64);

/// Deterministic celestial time derived from universe epoch.
#[derive(Resource, Default)]
pub struct CelestialTime(pub f64);

/// Physics simulation time (may differ from celestial time during catchup).
#[derive(Resource, Default)]
pub struct PhysicsTime(pub f64);

/// Index mapping ship_id → Entity for O(1) lookup from network messages.
/// Synced automatically via `Added<ShipId>` / `RemovedComponents<ShipId>`.
#[derive(Resource, Default)]
pub struct ShipEntityIndex(pub HashMap<u64, Entity>);

/// Index mapping session_token → Entity for O(1) lookup from network messages.
/// Synced automatically via `Added<PlayerId>` / `RemovedComponents<PlayerId>`.
#[derive(Resource, Default)]
pub struct PlayerEntityIndex(pub HashMap<SessionToken, Entity>);
