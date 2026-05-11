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

use crate::net::NetConnection;
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
    // -- Body / head decoupling (Phase A — surfaced for Phase C+) ----
    /// Server-broadcast body yaw in tangent frame (radians).
    pub body_yaw: f32,
    /// Server-broadcast head yaw relative to body (radians).
    pub head_yaw: f32,
    /// Server-broadcast head pitch (radians).
    pub head_pitch: f32,
    /// `LocomotionState as u8` — drives Phase D animation graph.
    pub locomotion: u8,
    /// Horizontal speed (m/s) — walk/run blend driver in Phase D.
    pub locomotion_speed: f32,
    /// Mid-turn-in-place flag — Phase F triggers TIP clip on rising edge.
    pub is_turning: bool,
    /// Target body yaw during the active turn (only meaningful when `is_turning`).
    pub turn_target_yaw: f32,
    /// 0..1 progress through the active turn-in-place clip.
    pub turn_t: f32,
    /// Server-broadcast look-at attention target encoded as a delta
    /// from `position` (Phase H). `None` = no target — head returns
    /// to the animation pose. The client converts to absolute world
    /// space at render time.
    pub look_target_delta: Option<glam::Vec3>,
    /// Per-bone world transforms during the active ragdoll window
    /// (Phase I). Empty for any state other than `Ragdoll`. Bones
    /// are identified by name (Mixamo convention) so the wire
    /// format is robust to per-class skeleton spec changes.
    pub ragdoll_bones: Vec<voxeldust_core::character::RagdollBoneTransform>,
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
    conn: Res<NetConnection>,
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

    let own_player_id = conn.player_id;
    if let Some(ws) = primary_ws.latest.as_ref() {
        if let Some(key) = primary.current {
            ingest(ws, key, own_player_id, &mut players, &mut ships, &mut debris);
        }
    }
    for (&shard_type, (ws, _)) in &secondary_ws.by_shard_type {
        // We don't know the exact `seed` of the secondary from its
        // WorldState (wire format collapses to shard_type). Use
        // shard_type + 0 as a placeholder ShardKey; downstream
        // consumers who need the authoritative shard id should cross-
        // reference `Secondaries.runtimes`.
        let placeholder = ShardKey {
            shard_type,
            seed: 0,
        };
        ingest(ws, placeholder, own_player_id, &mut players, &mut ships, &mut debris);
    }
}

fn ingest(
    ws: &voxeldust_core::client_message::WorldStateData,
    observer: ShardKey,
    _own_player_id: u64,
    players: &mut RemotePlayers,
    ships: &mut RemoteShips,
    debris: &mut RemoteDebris,
) {
    for e in &ws.entities {
        // `is_own` filters out the OWN-SHIP entry (system-shard sets
        // it on the observer's own ship). Ship-shard hardcodes it to
        // `false` for every player today, so the local player IS in
        // `RemotePlayers` — and that's intentional for Phase C, so
        // the local player gets a visual rendered via the same
        // skinned-mesh pipeline. Phase E adds the FP bone-mask cull
        // (head + clavicles + upper arms hidden in first-person) so
        // the body doesn't overlap the camera; until that lands the
        // user will see their own avatar at the camera position
        // (looking through the back of their own neck), which is the
        // right visual smoke-test that the spawn / pose-sync
        // pipeline is wired correctly.
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
    // Per-kind shard-key composition. Players are always observed by
    // their authoritative shard's WorldState (a ground player on this
    // ship is broadcast by THIS ship-shard; an EVA player by the
    // system-shard; a planet surface player by their planet-shard) —
    // so for them the `observer` ShardKey is the right entry into
    // `SourceIndex.by_shard` for parenting visuals. Ships, on the
    // other hand, may be cross-shard observable via the system-shard's
    // AOI feed; for those we honour `e.shard_id` so the renderer can
    // parent them under the secondary SHIP ChunkSource (matched by
    // the same seed `find_secondary_pose` uses).
    //
    // The asymmetry is load-bearing: ship-shard stamps
    // `entities[].shard_id = config.shard_id.0` (the orchestrator-
    // assigned id), which is *not* the wire seed used as
    // `ShardKey.seed`. For players that mismatch silently broke
    // SourceIndex lookups; for ships the secondary registration uses
    // `find_secondary_pose`'s own match against `shard_id` so it
    // doesn't go through SourceIndex.by_shard the same way.
    let shard = match e.kind {
        EntityKind::EvaPlayer | EntityKind::GroundedPlayer | EntityKind::Seated => observer,
        EntityKind::Ship => {
            if e.shard_id != 0 {
                ShardKey {
                    shard_type: e.shard_type,
                    seed: e.shard_id,
                }
            } else {
                observer
            }
        }
    };
    RemoteEntity {
        entity_id: e.entity_id,
        kind: e.kind,
        position: DVec3::new(e.position.x, e.position.y, e.position.z),
        rotation: DQuat::from_xyzw(e.rotation.x, e.rotation.y, e.rotation.z, e.rotation.w),
        velocity: DVec3::new(e.velocity.x, e.velocity.y, e.velocity.z),
        shard,
        name: e.name.clone(),
        health: e.health,
        shield: e.shield,
        body_yaw: e.body_yaw,
        head_yaw: e.head_yaw,
        head_pitch: e.head_pitch,
        locomotion: e.locomotion,
        locomotion_speed: e.locomotion_speed,
        is_turning: e.is_turning,
        turn_target_yaw: e.turn_target_yaw,
        turn_t: e.turn_t,
        look_target_delta: e.look_target_delta,
        // Phase I — the wire format on `ObservableEntity` mirrors
        // `PlayerSnapshot` so the unified ingestion path renders
        // ragdolls directly. Cloned (rather than moved) so the
        // source data stays intact for any other system reading
        // the WorldState the same tick.
        ragdoll_bones: e.ragdoll_bones.clone(),
    }
}
