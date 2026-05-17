//! Shard transitions + grace window.
//!
//! Composes `src_plugin.depart()` + `dst_plugin.arrive()` via the
//! registry — same code path for `ShardRedirect` (legacy, full
//! reconnect) and `ShardHandoff` (modern, seamless secondary→primary
//! promotion). Grace window preserves the old primary's
//! `ChunkSource` for `grace_duration()` after a transition so the
//! scene doesn't blink.
//!
//! Phase 10 MVP: latch `SpawnPose` into `SpawnPoseLatch`
//! (consumed by Phase 11 player_sync); retain old primary's chunk
//! source with server-reported velocity for extrapolation; despawn
//! on grace expiry.
//!
//! Scene-context preservation is already handled by the network
//! layer (voxydust network.rs binds SYSTEM + GALAXY secondaries to
//! `session_cancel_tx` instead of `primary_cancel_tx`). The client
//! does not need to filter scene-context shards out of the
//! transition teardown — they simply never get a
//! `SecondaryDisconnected` event on primary change.

use std::collections::HashMap;
use std::time::{Duration, Instant};

use bevy::prelude::*;
use glam::{DQuat, DVec3};

use crate::net::{GameEvent, NetEvent};
use crate::shard::origin::ShardOrigin;
use crate::shard::registry::{PrimaryShard, ShardRegistrySet, ShardTypeRegistry, SourceIndex};
use crate::shard::runtime::{ChunkSource, ShardKey};
use crate::shard::worldstate::PrimaryWorldState;

use voxeldust_core::client_message::EntityKind;
use voxeldust_core::handoff::SpawnPose;

#[derive(SystemSet, Clone, Debug, Hash, Eq, PartialEq)]
pub struct ShardTransitionSet;

/// Pending `SpawnPose` from a `Transitioning` event. Phase 11's
/// player_sync drains this and applies it as the authoritative camera
/// pose for the first frame after the handoff.
#[derive(Resource, Default, Debug, Clone)]
pub struct SpawnPoseLatch {
    pub pose: Option<SpawnPose>,
    pub target_shard_type: Option<u8>,
}

/// Grace-retained sources: old primaries' `ChunkSource` entities kept
/// alive for `grace_duration` after a transition so their chunks stay
/// visible during the seconds between old-primary demotion and new-
/// primary's snapshots arriving.
#[derive(Resource, Default)]
pub struct GraceWindow {
    pub retained: HashMap<ShardKey, GracedSource>,
}

pub struct GracedSource {
    pub entity: Entity,
    pub expiry: Instant,
    /// Snapshot of the source's last known velocity (for extrapolation
    /// during grace so the chunk mesh appears to keep travelling with
    /// its frame of reference, not freeze mid-air).
    pub velocity: DVec3,
    /// Anchor pose at grace start for extrapolation.
    pub origin_at_start: DVec3,
    pub rotation_at_start: DQuat,
    pub started_at: Instant,
}

pub struct ShardTransitionPlugin;

/// `latch_transitions` MUST run BEFORE `handle_connected_events` /
/// `handle_secondary_connected` (in `ShardRegistrySet`). When both a
/// `Transitioning` event and a `Connected` event arrive in the same
/// frame (the typical SHIP→EVA flow: ShardRedirect arrives, network
/// task tears down old primary + opens new, JoinResponse arrives a
/// few ms later — all before the next ECS tick), if the registry runs
/// first it promotes the new primary in place and replaces
/// `primary.current` BEFORE we can grab the OLD primary for grace.
/// Then `latch_transitions.primary.current.take()` returns the WRONG
/// shard (the just-promoted one) and graces it instead of the SHIP
/// that was actually departed. The SHIP entity is then orphaned in
/// `source_index.by_shard` until something stomps it; the new
/// SHIP-secondary spawn collides with the orphan, producing the
/// "ship disappears after EVA exit" symptom because the new
/// `ChunkSource` is empty until chunks re-stream and the orphan's
/// transform never updates again.
#[derive(SystemSet, Clone, Debug, Hash, Eq, PartialEq)]
pub struct ShardTransitionLatchSet;

impl Plugin for ShardTransitionPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<SpawnPoseLatch>()
            .init_resource::<GraceWindow>()
            .configure_sets(
                Update,
                (
                    ShardTransitionLatchSet
                        .after(crate::net::NetworkBridgeSet)
                        .before(ShardRegistrySet),
                    ShardTransitionSet.after(ShardRegistrySet),
                ),
            )
            .add_systems(Update, latch_transitions.in_set(ShardTransitionLatchSet))
            .add_systems(
                Update,
                (drive_grace_transforms, expire_grace)
                    .chain()
                    .in_set(ShardTransitionSet),
            );
    }
}

/// Consume `NetEvent::Transitioning` events: stash the spawn pose for
/// Phase 11 and stash the current primary's `ChunkSource` into the
/// grace window.
fn latch_transitions(
    mut events: MessageReader<GameEvent>,
    mut latch: ResMut<SpawnPoseLatch>,
    mut grace: ResMut<GraceWindow>,
    mut primary: ResMut<PrimaryShard>,
    mut source_index: ResMut<SourceIndex>,
    registry: Res<ShardTypeRegistry>,
    origins: Query<&ShardOrigin>,
    primary_ws: Res<PrimaryWorldState>,
) {
    for GameEvent(ev) in events.read() {
        if let NetEvent::Transitioning { target_shard_type, spawn_pose } = ev {
            latch.pose = spawn_pose.clone();
            latch.target_shard_type = Some(*target_shard_type);

            // Move old primary into grace window (if any).
            if let Some(old_key) = primary.current.take() {
                if let Some(entity) = source_index.by_shard.remove(&old_key) {
                    let plugin = registry.get(old_key.shard_type);
                    let keep = plugin.map(|p| p.wants_grace()).unwrap_or(true);
                    if keep {
                        let (origin, rotation) = origins
                            .get(entity)
                            .map(|o| (o.origin, o.rotation))
                            .unwrap_or((DVec3::ZERO, DQuat::IDENTITY));
                        let dur = plugin
                            .map(|p| p.grace_duration())
                            .unwrap_or(Duration::from_millis(1500));
                        // Phase T4 — capture the source's last-known
                        // velocity so `drive_grace_transforms` can
                        // extrapolate the graced chunks along the
                        // source's trajectory. Without this an in-
                        // flight ship→EVA / warp-arrival handoff
                        // leaves the source ship's chunks frozen in
                        // system-space while the ship itself orbits
                        // away at ~110 km/s — visually the chunks
                        // appear to teleport when grace expires.
                        let velocity = source_velocity_for_grace(
                            &old_key, primary_ws.latest.as_ref(),
                        );
                        grace.retained.insert(
                            old_key,
                            GracedSource {
                                entity,
                                expiry: Instant::now() + dur,
                                velocity,
                                origin_at_start: origin,
                                rotation_at_start: rotation,
                                started_at: Instant::now(),
                            },
                        );
                        tracing::info!(
                            %old_key,
                            target_shard_type,
                            vel = ?(velocity.x, velocity.y, velocity.z),
                            "transition: old primary graced",
                        );
                    } else {
                        tracing::info!(
                            %old_key,
                            "transition: plugin opted out of grace — despawning",
                        );
                        // Despawn immediately; `registry.by_shard` already
                        // had this key removed above.
                    }
                }
            }
        }
    }
}

/// Recover a graced source's last-known velocity in system-space from
/// the primary's most recent `WorldState`. The primary at this point
/// is still the OLD primary (the transition latch runs before the
/// new primary's `Connected` event lands), so its `entities[]`
/// snapshot is exactly the right place to ask.
///
/// Per-shard-type wiring:
///
///   * **SHIP** — ship-shard's own broadcast includes the own-ship
///     entity at `entity_id = config.ship_id` (= `ShardKey.seed` for
///     SHIP shards on the wire) with `velocity = exterior.velocity`
///     in system-space. Match by `entity_id` (canonical) or
///     `shard_id` (defensive; some legacy paths stamped orchestrator
///     id there). Returns the ship's orbital velocity.
///   * **PLANET** — planet shard doesn't broadcast its own-planet
///     entity. Returns `DVec3::ZERO`; a future refinement can compute
///     orbital velocity from `(planet_seed, celestial_time)` on the
///     client without a server round-trip.
///   * **SYSTEM / GALAXY** — frame origins, stationary by design.
///     Returns `DVec3::ZERO`.
///
/// Unknown shard types fall through to `DVec3::ZERO` (identical to
/// pre-T4 behaviour — no regression possible).
fn source_velocity_for_grace(
    old_key: &ShardKey,
    primary_ws: Option<&voxeldust_core::client_message::WorldStateData>,
) -> DVec3 {
    const SHIP_SHARD_TYPE: u8 = 2;
    if old_key.shard_type != SHIP_SHARD_TYPE {
        return DVec3::ZERO;
    }
    let Some(ws) = primary_ws else {
        return DVec3::ZERO;
    };
    for e in &ws.entities {
        if e.kind != EntityKind::Ship {
            continue;
        }
        if e.entity_id == old_key.seed || e.shard_id == old_key.seed {
            return e.velocity;
        }
    }
    DVec3::ZERO
}

/// Extrapolate graced sources' `ShardOrigin` each frame. Phase 4's
/// rebase system then picks up the updated origin and writes the Bevy
/// Transform; no special-case code here beyond the extrapolation.
fn drive_grace_transforms(
    grace: Res<GraceWindow>,
    mut origins: Query<&mut ShardOrigin, With<ChunkSource>>,
) {
    let now = Instant::now();
    for graced in grace.retained.values() {
        let Ok(mut origin) = origins.get_mut(graced.entity) else { continue };
        let elapsed = now.saturating_duration_since(graced.started_at).as_secs_f64();
        origin.origin = graced.origin_at_start + graced.velocity * elapsed;
        origin.rotation = graced.rotation_at_start;
    }
}

/// Despawn graced sources whose grace has expired.
fn expire_grace(mut grace: ResMut<GraceWindow>, mut commands: Commands) {
    let now = Instant::now();
    let expired: Vec<ShardKey> = grace
        .retained
        .iter()
        .filter(|(_, g)| g.expiry <= now)
        .map(|(k, _)| *k)
        .collect();
    for key in expired {
        if let Some(g) = grace.retained.remove(&key) {
            commands.entity(g.entity).despawn();
            tracing::info!(%key, "grace expired — chunk source despawned");
        }
    }
}
