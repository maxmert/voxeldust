//! Per-frame planet rotation — Phase 4.
//!
//! Drives a connected PLANET shard's `ShardOrigin.rotation` from the
//! server-broadcast [`PlanetRotationParams`] and the per-frame
//! extrapolated `game_time`. Day/night emerges naturally:
//!
//! 1. The star's position is fixed in the system inertial frame.
//! 2. The planet's surface chunks rotate with `ShardOrigin.rotation`.
//! 3. The camera (player) is anchored to the rotating planet frame, so
//!    it carries through dawn → noon → dusk → midnight as
//!    `ShardOrigin.rotation` advances.
//! 4. `solar.rs`'s per-camera sun-direction recomputation
//!    `(star_pos − camera_world).normalize()` then produces a smoothly
//!    rotating sun direction without any extra plumbing.
//!
//! ## BE-authority + temporal correctness
//!
//! Both server and client invoke the **same**
//! [`core::planet_rotation::rotation_at`] pure function with byte-equal
//! inputs:
//!
//!   * `params` are static for the planet's session lifetime, broadcast
//!     verbatim in `CelestialBodySnapshot.rotation_params`.
//!   * `game_time_now` is extrapolated from the latest server tick at
//!     1:1 real-time:
//!     `game_time + (Instant::now() − last_tick_real_time).as_secs_f64()`.
//!
//! Two clients connected to the same server at the same wall-clock
//! moment compute the same `game_time_now` (assuming clocks roughly
//! sync) and the same quaternion bit-for-bit. There's no interpolation
//! between server samples — every client samples the closed-form
//! function at the same `t`.
//!
//! ## Seamless transition
//!
//! Because the rotation depends only on `params + game_time`, and
//! params don't change between SYSTEM-secondary's broadcast and
//! PLANET-primary's broadcast (both source from the system-shard's
//! single authoritative computation), no rotation discontinuity ever
//! occurs at primary swaps.

use bevy::prelude::*;

use voxeldust_core::client_message::WorldStateData;
use voxeldust_core::planet_rotation::{rotation_at, PlanetRotationParams};

use crate::shard::origin::{
    rebase_shard_transforms, refresh_origins_from_worldstate, ShardOrigin, ShardOriginSet,
};
use crate::shard::registry::{PrimaryShard, Secondaries, SourceIndex};
use crate::shard::runtime::ChunkSource;
use crate::shard::worldstate::{resolve_game_time_now, PrimaryWorldState, SecondaryWorldStates};
use crate::shard_types::planet::PLANET_SHARD_TYPE;

pub struct PlanetRotationPlugin;

impl Plugin for PlanetRotationPlugin {
    fn build(&self, app: &mut App) {
        // Insert between the existing `refresh_origins_from_worldstate`
        // (which sets every secondary's `ShardOrigin.rotation` to
        // identity / non-PLANET source) and `rebase_shard_transforms`
        // (which writes the rotation onto Bevy `Transform`). That makes
        // us the LAST writer for the PLANET shard's rotation field
        // before it gets consumed.
        //
        // We explicitly bind to those two systems by name rather than
        // putting ourselves in `ShardOriginSet`-then-`.after(ShardOriginSet)`,
        // because a system can't both belong to a set and run after it.
        app.add_systems(
            Update,
            apply_planet_rotation
                .in_set(ShardOriginSet)
                .after(refresh_origins_from_worldstate)
                .before(rebase_shard_transforms),
        );
    }
}

/// Each frame, for every PLANET shard whose home body has broadcast
/// `rotation_params`, write `ShardOrigin.rotation = rotation_at(params,
/// game_time_now)`. Runs after the base origin-refresh system so we're
/// the last writer for the PLANET shard's rotation field.
fn apply_planet_rotation(
    primary: Res<PrimaryShard>,
    secondaries: Res<Secondaries>,
    primary_ws: Res<PrimaryWorldState>,
    secondary_ws: Res<SecondaryWorldStates>,
    sources: Res<SourceIndex>,
    chunk_sources: Query<&ChunkSource>,
    mut origins: Query<&mut ShardOrigin>,
) {
    // Need an authoritative `game_time` reference + arrival timestamp to
    // extrapolate. Prefer the primary WS — it ticks at the highest rate
    // (the player's "now"). Fall back to the PLANET secondary's WS only
    // if the primary hasn't received its first WS yet (e.g., during the
    // transient handoff before primary promotion completes).
    let game_time_now = match resolve_game_time_now(&primary_ws, &secondary_ws) {
        Some(t) => t,
        None => return,
    };

    // For each connected PLANET shard, find its rotation_params (from
    // the PLANET WS body whose position is closest to that shard's
    // origin — the home planet) and apply.
    for (&key, &entity) in sources.by_shard.iter() {
        if key.shard_type != PLANET_SHARD_TYPE {
            continue;
        }
        // Confirm the shard is actually connected (primary or secondary).
        let connected = primary.current == Some(key) || secondaries.runtimes.contains_key(&key);
        if !connected {
            continue;
        }
        // Locate its WS — primary if PLANET-primary; secondary otherwise.
        let ws_opt: Option<&WorldStateData> = if primary.current == Some(key) {
            primary_ws.latest.as_ref()
        } else {
            secondary_ws
                .by_shard_type
                .get(&PLANET_SHARD_TYPE)
                .map(|(ws, _)| ws)
        };
        let Some(ws) = ws_opt else { continue };

        let Some(params) = home_planet_rotation_params(ws) else {
            continue;
        };
        let q = rotation_at(&params, game_time_now);

        if chunk_sources.get(entity).is_err() {
            continue;
        }
        if let Ok(mut origin) = origins.get_mut(entity) {
            origin.rotation = q;
        }
    }
}

/// Pull the home planet's broadcast `rotation_params` out of a PLANET
/// shard's WorldState. The home planet is the body closest to the
/// shard's origin (positions are shard-local; the home planet sits at
/// `≈ 0` after the `planet_sys_pos − planet_pos.0` subtraction in
/// `planet-shard/src/main.rs`).
fn home_planet_rotation_params(ws: &WorldStateData) -> Option<PlanetRotationParams> {
    ws.bodies
        .iter()
        .filter(|b| b.body_id != 0)
        .filter_map(|b| b.rotation_params.map(|r| (b, r)))
        .min_by(|(a, _), (b, _)| {
            a.position
                .length_squared()
                .partial_cmp(&b.position.length_squared())
                .unwrap_or(std::cmp::Ordering::Equal)
        })
        .map(|(_, params)| params)
}
