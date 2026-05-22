//! Shared celestial clock — single source of truth for `game_time`
//! across every server shard.
//!
//! ## Why a plugin instead of per-shard inline calls
//!
//! Before this module, every shard maintained its own `game_time`
//! tracking with subtle variations:
//!   * `system-shard`: inline `celestial_time.0 = celestial_time_from_epoch(...)`
//!     inside `physics_integrate`.
//!   * `planet-shard`: inline `celestial_time.0 = celestial_time_from_epoch(...)`
//!     inside `step_planet_physics`, against a *local* `CelestialTimeRes`
//!     wrapper distinct from the shared `core::ecs::CelestialTime`.
//!   * `ship-shard`: **mirrored** `scene.game_time = SystemSceneUpdate.game_time`
//!     received via QUIC — lagged the system-shard source by ≥ 1 inter-shard
//!     round-trip every tick.
//!   * `galaxy-shard`: derived from its own tick counter
//!     (`tick.0 * DT`) — drifts across shard restart and never matches
//!     the universe epoch the other shards anchor on.
//!
//! Four shard types, three distinct mechanisms, two of which lag.
//! The mirror lag manifested as a client-side rendering blink: a
//! remote player on a SYSTEM secondary observed from a SHIP primary
//! had its `RemoteEntity.game_time` (fresh from system) consistently
//! ~1 tick ahead of `CameraWorldPos.game_time` (from ship-mirror),
//! so the cross-shard interpolation target in
//! `client/src/character/render.rs::rebase_root_visuals` could not
//! sit inside both lerp windows at once.
//!
//! This module fixes the root cause: one plugin, one resource, one
//! per-tick system, calling the *one* canonical helper
//! `shard_common::harness::celestial_time_from_epoch`. Every shard
//! adopts `CelestialClockPlugin` and reads `Res<CelestialTime>` — no
//! exceptions, no special cases, no mirrors.
//!
//! ## Determinism contract
//!
//! `celestial_time_from_epoch(epoch, time_scale)` is a pure function
//! of `(wall_clock_now − epoch_ms) × time_scale`. Two shards running
//! on synchronised wall-clocks (the same host, or NTP-synced hosts)
//! compute byte-identical `f64` values at the same instant. The
//! `Arc<AtomicU64>` carrying the epoch is shared across every shard
//! in the cluster via the orchestrator's join handshake
//! (`harness.rs::epoch_arc()`), and `time_scale` is derived
//! deterministically from the system seed on each shard's startup.

use std::sync::atomic::AtomicU64;
use std::sync::Arc;

use bevy_app::{App, Plugin, Update};
use bevy_ecs::prelude::*;

use voxeldust_core::ecs::CelestialTime;

use crate::harness::celestial_time_from_epoch;

/// System set running the per-tick `advance_celestial_time` system.
/// Shards whose other systems must observe a fresh `CelestialTime`
/// for the current tick should be configured `.after(CelestialClockSet)`.
#[derive(SystemSet, Debug, Clone, PartialEq, Eq, Hash)]
pub struct CelestialClockSet;

/// Per-shard time-scale multiplier. Defaults to `1.0` (no scaling) so
/// that shards without celestial physics (e.g. galaxy in-transit)
/// still get a coherent monotonic clock without explicit setup.
/// Shards that own a `SystemParams` should `.insert_resource(TimeScale(sys.scale.time_scale))`
/// at startup so their `CelestialTime` matches the rest of the cluster.
#[derive(Resource, Debug, Clone, Copy)]
pub struct TimeScale(pub f64);

impl Default for TimeScale {
    fn default() -> Self {
        Self(1.0)
    }
}

/// Shared `Arc<AtomicU64>` carrying the universe epoch (Unix epoch
/// milliseconds), wrapped as a Bevy resource so the per-tick clock
/// system reads it locklessly. Obtain via `ShardHarness::epoch_arc()`
/// at startup and `.insert_resource(UniverseEpoch(arc))`.
#[derive(Resource, Clone)]
pub struct UniverseEpoch(pub Arc<AtomicU64>);

/// One-line registration for the shared celestial clock.
///
/// Required setup per shard:
/// ```rust,ignore
/// app.add_plugins(CelestialClockPlugin)
///    .insert_resource(UniverseEpoch(harness.epoch_arc()))
///    .insert_resource(TimeScale(sys_params.scale.time_scale));
/// ```
///
/// After this:
///   * Every consumer reads `celestial_time: Res<CelestialTime>` and
///     uses `.0`.
///   * No shard should call `celestial_time_from_epoch` directly any
///     more, nor mirror `game_time` from any cross-shard message.
pub struct CelestialClockPlugin;

impl Plugin for CelestialClockPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<CelestialTime>()
            .init_resource::<TimeScale>()
            .add_systems(
                Update,
                advance_celestial_time.in_set(CelestialClockSet),
            );
    }
}

/// Refresh `CelestialTime` once per tick from the wall-clock-relative
/// `(now − epoch) × time_scale` formula. Pure side-effect-free read
/// of `UniverseEpoch` (atomic) and `TimeScale` (Copy); safe to run
/// concurrently with anything else in the same set.
fn advance_celestial_time(
    mut celestial_time: ResMut<CelestialTime>,
    epoch: Res<UniverseEpoch>,
    scale: Res<TimeScale>,
) {
    celestial_time.0 = celestial_time_from_epoch(&epoch.0, scale.0);
}
