//! Floating-origin rebase: f64 world-space stored canonically, f32
//! camera-relative translation computed each frame.
//!
//! Every `ChunkSource` root entity carries a `ShardOrigin` component
//! (DVec3 origin + DQuat rotation in system-space). A per-frame
//! `rebase_shard_transforms` system reads `CameraWorldPos` (the f64
//! camera world-space position) and writes every ChunkSource's Bevy
//! `Transform`:
//!
//!   translation = (origin - camera_world).as_vec3()
//!   rotation    = rotation.as_quat()
//!
//! The camera entity itself keeps `Transform.translation = ZERO`;
//! Phase 11 writes its rotation from the authoritative player pose.
//! This way the subtraction is always small-magnitude — sub-cm
//! precision at 10⁵ m, sub-mm at 10³ m — regardless of how far the
//! camera has travelled in world-space.

use bevy::prelude::*;
use glam::{DQuat, DVec3};

use voxeldust_core::client_message::EntityKind;

use crate::shard::registry::{PrimaryShard, ShardRegistrySet, SourceIndex};
use crate::shard::runtime::ChunkSource;
use crate::shard::worldstate::{PrimaryWorldState, SecondaryWorldStates, WorldStateIngestSet};

#[derive(SystemSet, Clone, Debug, Hash, Eq, PartialEq)]
pub struct ShardOriginSet;

/// Component on every ChunkSource root. Canonical f64 world-space pose.
/// Read by `rebase_shard_transforms`; write via `apply_world_pose`
/// helpers from per-shard-type plugins (Phases 6-9) when they ingest
/// `SystemEntitiesUpdate` / `ShipPositionUpdate`.
#[derive(Component, Debug, Clone, Copy)]
pub struct ShardOrigin {
    pub origin: DVec3,
    pub rotation: DQuat,
}

impl ShardOrigin {
    pub fn new(origin: DVec3, rotation: DQuat) -> Self {
        Self { origin, rotation }
    }
}

impl Default for ShardOrigin {
    fn default() -> Self {
        Self {
            origin: DVec3::ZERO,
            rotation: DQuat::IDENTITY,
        }
    }
}

/// The camera's f64 position in system-space (current primary's
/// coordinate frame). Default `ZERO` until Phase 11 starts writing
/// authoritative player pose each tick.
#[derive(Resource, Debug, Clone, Copy)]
pub struct CameraWorldPos {
    pub pos: DVec3,
}

impl Default for CameraWorldPos {
    fn default() -> Self {
        Self { pos: DVec3::ZERO }
    }
}

pub struct ShardOriginPlugin;

impl Plugin for ShardOriginPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<CameraWorldPos>()
            .configure_sets(
                Update,
                ShardOriginSet
                    .after(ShardRegistrySet)
                    .after(WorldStateIngestSet),
            )
            .add_systems(
                Update,
                (
                    // Per-tick: refresh ShardOrigin.origin from the
                    // authoritative WorldState broadcasts. Without this
                    // the initial `reference_position` captured at
                    // Connected goes stale as the ship moves.
                    refresh_origins_from_worldstate,
                    // Then rebase every entity's Transform against the
                    // fresh origin + camera_world.
                    rebase_shard_transforms,
                )
                    .chain()
                    .in_set(ShardOriginSet),
            );
    }
}

/// Update every ChunkSource's `ShardOrigin.origin` from this tick's
/// WorldState data. Two sources feed this:
///   * Primary: `PrimaryWorldState.origin` is the primary shard's
///     current system-space anchor. On SHIP primary, `origin` is
///     (ship_position, ship_rotation) so own-ship chunks render at
///     the ship's live world pose.
///   * Secondary: `SecondaryWorldStates.by_shard_type[shard_type].origin`
///     for that type's secondary shard; plus ObservableEntity[] on the
///     primary WS that lists ships/debris with their live poses for
///     cross-shard rendering (own-ship inside SYSTEM primary, parked
///     ship visible from PLANET primary, etc.).
fn refresh_origins_from_worldstate(
    primary: Res<PrimaryShard>,
    primary_ws: Res<PrimaryWorldState>,
    secondary_ws: Res<SecondaryWorldStates>,
    sources: Res<SourceIndex>,
    mut origins: Query<(&ChunkSource, &mut ShardOrigin)>,
) {
    let Some(primary_key) = primary.current else { return };
    let Some(pws) = primary_ws.latest.as_ref() else { return };

    // Primary origin = ws.origin, rotation = own-ship rotation if on SHIP.
    let primary_origin = DVec3::new(pws.origin.x, pws.origin.y, pws.origin.z);
    let primary_rotation = pws
        .entities
        .iter()
        .find(|e| e.is_own && e.kind == EntityKind::Ship)
        .map(|s| DQuat::from_xyzw(s.rotation.x, s.rotation.y, s.rotation.z, s.rotation.w))
        .unwrap_or(DQuat::IDENTITY);
    if let Some(&primary_entity) = sources.by_shard.get(&primary_key) {
        if let Ok((_, mut origin)) = origins.get_mut(primary_entity) {
            origin.origin = primary_origin;
            origin.rotation = primary_rotation;
        }
    }

    // Secondaries: look up each shard's freshest authoritative pose.
    // On `None`, the origin is LEFT UNCHANGED — a missing lookup should
    // not snap the source to zero (would manifest as a position
    // blink). This matters especially for SHIP secondaries: multiple
    // SHIP secondaries collapse into a single `by_shard_type[SHIP]`
    // entry on the client because the wire NetEvent doesn't carry
    // seed, so ship poses come from the primary SYSTEM's
    // authoritative `entities[]` broadcast instead.
    for (&key, &entity) in &sources.by_shard {
        if key == primary_key {
            continue;
        }
        if let Some((origin_d, rotation_d)) =
            find_secondary_pose(key, &secondary_ws, pws)
        {
            if let Ok((_, mut origin)) = origins.get_mut(entity) {
                origin.origin = origin_d;
                origin.rotation = rotation_d;
            }
        }
    }
}

/// Multi-source lookup for a secondary shard's live pose. Returns
/// `None` when no authoritative pose is available this tick — caller
/// leaves the previous origin in place rather than snapping to zero.
///
/// Precedence by shard_type:
/// - SHIP secondary: look up the matching ship in primary WS's
///   `entities[]` by `shard_id == key.seed` (or entity_id fallback).
///   Position from the entity + primary ws.origin. We intentionally
///   ignore `secondary_ws.by_shard_type[SHIP]` for ships because
///   multiple SHIP secondaries collapse into a single entry on the
///   client side (the NetEvent doesn't carry seed), and using the
///   collapsed entry would flicker ship A's render between A's real
///   position and ship B's.
/// - Any other secondary type (SYSTEM / PLANET / GALAXY / …): use
///   its own WorldState origin; orientation from primary entities
///   only when a matching ship-shard entry is found.
fn find_secondary_pose(
    key: crate::shard::ShardKey,
    secondary_ws: &SecondaryWorldStates,
    primary_ws: &voxeldust_core::client_message::WorldStateData,
) -> Option<(DVec3, DQuat)> {
    const SHIP_SHARD_TYPE: u8 = 2;

    if key.shard_type == SHIP_SHARD_TYPE {
        if let Some(ship) = primary_ws.entities.iter().find(|e| {
            e.kind == EntityKind::Ship
                && (e.shard_id == key.seed || e.entity_id == key.seed)
        }) {
            let pos = DVec3::new(
                primary_ws.origin.x,
                primary_ws.origin.y,
                primary_ws.origin.z,
            ) + DVec3::new(ship.position.x, ship.position.y, ship.position.z);
            let rot = DQuat::from_xyzw(
                ship.rotation.x,
                ship.rotation.y,
                ship.rotation.z,
                ship.rotation.w,
            );
            return Some((pos, rot));
        }
        return None;
    }

    // Non-SHIP secondaries: secondary WS's own origin is authoritative.
    if let Some(ws) = secondary_ws.by_shard_type.get(&key.shard_type) {
        let origin = DVec3::new(ws.origin.x, ws.origin.y, ws.origin.z);
        let rotation = primary_ws
            .entities
            .iter()
            .find(|e| e.shard_id == key.seed && e.kind == EntityKind::Ship)
            .map(|s| {
                DQuat::from_xyzw(
                    s.rotation.x,
                    s.rotation.y,
                    s.rotation.z,
                    s.rotation.w,
                )
            })
            .unwrap_or(DQuat::IDENTITY);
        return Some((origin, rotation));
    }
    None
}

fn rebase_shard_transforms(
    camera_world: Res<CameraWorldPos>,
    mut query: Query<(&ShardOrigin, &mut Transform), With<ChunkSource>>,
) {
    let cam = camera_world.pos;
    for (origin, mut transform) in &mut query {
        // `(origin - cam)` is computed in f64, then down-cast. Never
        // subtract big-magnitude values directly in f32.
        let delta = origin.origin - cam;
        transform.translation = dvec3_to_bevy(delta);
        transform.rotation = dquat_to_bevy(origin.rotation);
    }
}

// Workspace `glam` is 0.29; `bevy::prelude::{Vec3, Quat}` come from
// bevy's 0.30. Cast through the public field-by-field constructors.

fn dvec3_to_bevy(v: DVec3) -> Vec3 {
    Vec3::new(v.x as f32, v.y as f32, v.z as f32)
}

fn dquat_to_bevy(q: DQuat) -> Quat {
    Quat::from_xyzw(q.x as f32, q.y as f32, q.z as f32, q.w as f32)
}
