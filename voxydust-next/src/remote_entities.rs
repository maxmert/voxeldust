//! Remote player / AOI-entity state tracking.
//!
//! **Rendering policy (binary, user feedback 2026-04-22):**
//!
//! - **In visibility range** → render the proper mesh.
//! - **Out of visibility range** → don't render.
//!
//! No placeholder geometry, no LOD substitutes, no colored-sphere
//! proxies. Ships in range render as their actual block meshes via
//! `chunk_stream`. Player avatars in range will render as proper
//! rigged meshes (future work — until those art assets land, remote
//! players are not drawn at all; we do not fill the gap with a
//! placeholder).
//!
//! Position tracking (this module, via the `RemotePlayers` resource)
//! is **orthogonal** to rendering — we keep every known remote
//! entity's pose here regardless of whether it is currently in
//! rendering range, because downstream subsystems need it:
//!
//! Why track without rendering:
//!
//! 1. **Radar / long-range detection** (future): a thruster-signature
//!    radar system needs to know where every ship/player is, even
//!    those outside the render frustum. Centralising the position map
//!    here gives that system a single source of truth.
//! 2. **Collision / interaction probes** (future): block-target raycasts
//!    and weapon-damage validation will check against entity positions.
//!    Having them in a resource keyed by `entity_id` makes that trivial.
//! 3. **Avatar-mesh hand-in** (future): when proper avatars land, the
//!    rendering system will walk this same resource and spawn/update
//!    meshes per entry. This module is the anchor; the mesh pipeline
//!    plugs in later without re-plumbing WS consumption.
//!
//! Composition: same multi-shard rules as `chunk_stream` and
//! `celestial.rs` — walk primary + every secondary + grace WS, dedupe
//! by `entity_id`. The first WS to declare an entity wins its state;
//! primary is always checked first so it takes priority.

use std::collections::HashSet;

use bevy::prelude::*;
use glam::{DQuat, DVec3};
use voxeldust_core::client_message::{EntityKind, ObservableEntityData};

use crate::camera_frame::CameraFrameSet;
use crate::shard_transition::{ShardTransitionSet, WorldStates};

pub struct RemoteEntitiesPlugin;

impl Plugin for RemoteEntitiesPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<RemotePlayers>().add_systems(
            Update,
            track_remote_players
                .after(ShardTransitionSet)
                .after(CameraFrameSet),
        );
    }
}

/// Snapshot of every non-own player currently known to the client,
/// deduped by `entity_id`. Positions are in the **same Bevy world frame**
/// the camera uses — already shifted by `source_origin − primary_origin`
/// so consumers can compare against camera transforms without further
/// math.
///
/// Ships are intentionally excluded here — they have their own full
/// lifecycle in `chunk_stream` (pre-connect → observer TCP → mesh
/// spawn → root Transform driven by AOI). Including them twice would
/// require coordinating two sources-of-truth for the same object.
#[derive(Resource, Default, Debug, Clone)]
pub struct RemotePlayers {
    pub entries: bevy::platform::collections::HashMap<u64, RemotePlayer>,
}

#[derive(Debug, Clone, Copy)]
pub struct RemotePlayer {
    pub entity_id: u64,
    pub kind: EntityKind,
    /// Bevy world position (primary-origin-relative, system-space-axes).
    pub world_pos: Vec3,
    /// World rotation as broadcast by the server.
    pub world_rot: DQuat,
    /// World-space velocity (primary-origin-relative axes). Carried
    /// forward for extrapolation / dead-reckoning when radar queries
    /// run between ticks.
    pub velocity: DVec3,
    pub bounding_radius: f32,
    /// Shard the entity is authoritative on (for follow-up pre-connect
    /// decisions once radar / long-range rendering lands).
    pub shard_id: u64,
    pub shard_type: u8,
}

fn track_remote_players(worlds: Res<WorldStates>, mut players: ResMut<RemotePlayers>) {
    let primary_origin = worlds
        .primary
        .as_ref()
        .map(|ws| ws.origin)
        .unwrap_or(DVec3::ZERO);

    let mut seen: HashSet<u64> = HashSet::new();

    let mut ingest = |e: &ObservableEntityData,
                      source_origin: DVec3,
                      seen: &mut HashSet<u64>,
                      players: &mut RemotePlayers| {
        if e.is_own {
            return;
        }
        if !e.kind.is_player() {
            // Ships are tracked by chunk_stream; other non-player
            // kinds don't exist yet.
            return;
        }
        if !seen.insert(e.entity_id) {
            return;
        }
        let world_pos = source_origin + e.position - primary_origin;
        let bevy_pos = Vec3::new(
            world_pos.x as f32,
            world_pos.y as f32,
            world_pos.z as f32,
        );
        players.entries.insert(
            e.entity_id,
            RemotePlayer {
                entity_id: e.entity_id,
                kind: e.kind,
                world_pos: bevy_pos,
                world_rot: e.rotation,
                velocity: e.velocity,
                bounding_radius: e.bounding_radius,
                shard_id: e.shard_id,
                shard_type: e.shard_type,
            },
        );
    };

    if let Some(ws) = worlds.primary.as_ref() {
        for e in &ws.entities {
            ingest(e, ws.origin, &mut seen, &mut players);
        }
    }
    for ws in worlds.secondary_by_type.values() {
        for e in &ws.entities {
            ingest(e, ws.origin, &mut seen, &mut players);
        }
    }
    if let Some(ws) = worlds.last_primary.as_ref() {
        for e in &ws.entities {
            ingest(e, ws.origin, &mut seen, &mut players);
        }
    }

    // Drop players that are no longer visible in any WS. Future radar
    // work may want a different decay policy (keep positions for a few
    // seconds, flag as "stale"), but for now an entity that leaves all
    // AOI is gone until it re-enters.
    players.entries.retain(|id, _| seen.contains(id));
}
