//! WorldState fan-out: drain `NetEvent::WorldState` +
//! `NetEvent::SecondaryWorldState` into `PrimaryWorldState` and
//! `SecondaryWorldStates` resources so downstream systems (Phase 11+
//! camera pose, Phase 20 remote-entity tracking, HUD) read without
//! touching the raw event stream.

use std::collections::HashMap;
use std::time::Instant;

use bevy::prelude::*;

use voxeldust_core::client_message::WorldStateData;

use crate::net::{GameEvent, NetEvent};
use crate::shard::registry::{PrimaryShard, ShardRegistrySet};
use crate::shard::runtime::ShardKey;

#[derive(SystemSet, Clone, Debug, Hash, Eq, PartialEq)]
pub struct WorldStateIngestSet;

/// Latest primary WorldState. Replaced every tick; downstream code
/// reads the current tick's authoritative game state.
///
/// `last_tick_real_time` records when the most recent fresh WorldState
/// was ingested, so per-frame systems (e.g.
/// `client/src/lighting/rotation.rs`) can extrapolate the
/// server-authoritative `game_time` between ticks at 1:1 real-time:
///
///     game_time_now = ws.game_time + (Instant::now() − last_tick_real_time)
///
/// This produces the same `game_time_now` on any client at any wall-clock
/// moment, which is the basis of the cross-client temporal-correctness
/// guarantee in `core::planet_rotation`.
#[derive(Resource, Default)]
pub struct PrimaryWorldState {
    pub latest: Option<WorldStateData>,
    pub last_tick_real_time: Option<Instant>,
}

/// Latest WorldState per secondary shard, keyed two ways:
///
///   * `by_shard_type` (legacy, lossy when multiple secondaries share
///     a type) — many existing consumers (lighting, AR, atmosphere,
///     eclipse, etc.) read this without needing per-secondary
///     disambiguation. Kept for backward-compat.
///   * `by_shard_key` (full `ShardKey` = `shard_type` + `seed`) —
///     the authoritative map. Use this anywhere correctness depends
///     on knowing exactly which secondary a WorldState came from
///     (e.g. cross-shard player visual parenting in
///     `client/src/remote/mod.rs`).
///
/// The `Instant` follows the same role as
/// `PrimaryWorldState.last_tick_real_time`: per-frame consumers
/// extrapolate `game_time_now` from `(ws, instant)`.
#[derive(Resource, Default)]
pub struct SecondaryWorldStates {
    pub by_shard_type: HashMap<u8, (WorldStateData, Instant)>,
    pub by_shard_key: HashMap<ShardKey, (WorldStateData, Instant)>,
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

/// Compute `game_time_now` from every WorldState we've received,
/// returning the MAX of the wall-clock-extrapolated values. Returns
/// `None` if no WS has been received yet.
///
/// Per-source extrapolation: `ws.game_time + (Instant::now() −
/// last_tick_real_time).as_secs_f64()`. The 1:1 real-time scaling
/// makes derived quantities (planet rotation, render-target
/// game-time) "live" between 20 Hz server ticks.
///
/// **Why max, not primary-preferred**: `game_time` is
/// deterministically derived per-shard from `(wall_clock − universe
/// epoch) × time_scale` (`shard-common::harness::celestial_time_from_epoch`),
/// so two shards on synchronised wall-clocks compute IDENTICAL
/// `game_time` values at any instant. But not every shard derives
/// fresh: ship-shard mirrors `game_time` from `SystemSceneUpdate.game_time`
/// (see `ship-shard/src/main.rs::drain_quic`), so its broadcasts
/// carry game_time values that lag system-shard's by ~1 QUIC
/// round-trip + send cadence (≈ 50 ms). If a client picks the
/// ship-mirror's anchor over the system-source's, the render-target
/// `target = anchor − INTERPOLATION_DELAY` lands BELOW system-shard's
/// per-entity lerp windows and every remote that came via the system
/// secondary clamps to `prev_position` every frame — visible as a
/// residual one-frame-per-tick blink.
///
/// Taking the max picks the freshest source (lowest end-to-end lag).
/// Mirror sources naturally lose; direct sources win.
///
/// Single source of truth for "what time is it on the server right
/// now". Used by:
///   * `client/src/lighting/rotation.rs` for planet rotation
///     (`rotation_at(params, game_time_now)`).
///   * `client/src/character/render.rs::rebase_root_visuals` and
///     `sync_remote_pose` to compute the cross-shard render target
///     `target = game_time_now − INTERPOLATION_DELAY`. Cross-shard
///     entity rendering MUST evaluate both `cam.pos` and
///     `remote.position` at the same simulation instant; this
///     function provides that instant.
pub fn resolve_game_time_now(
    primary: &PrimaryWorldState,
    secondary: &SecondaryWorldStates,
) -> Option<f64> {
    let now = Instant::now();
    let extrapolate = |ws_game_time: f64, last_real: Instant| -> f64 {
        ws_game_time + now.saturating_duration_since(last_real).as_secs_f64()
    };
    let mut best: Option<f64> = None;
    if let (Some(ws), Some(last_real)) = (primary.latest.as_ref(), primary.last_tick_real_time) {
        best = Some(extrapolate(ws.game_time, last_real));
    }
    for (ws, last_real) in secondary.by_shard_key.values() {
        let candidate = extrapolate(ws.game_time, *last_real);
        best = Some(match best {
            Some(prev) if prev >= candidate => prev,
            _ => candidate,
        });
    }
    best
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
                primary.last_tick_real_time = None;
                // We no longer remove the stale secondary WorldState here.
                // Doing so immediately on connection causes a visual blink/disappearance
                // for entities on that shard during the network handshake gap (before the
                // first primary WorldState packet arrives). Instead, we defer this cleanup
                // to `ingest_primary` when the first primary packet is actually ingested.
            }
            NetEvent::SecondaryDisconnected { seed } => {
                // Remove ONLY the disconnected secondary from the
                // authoritative `by_shard_key` map. Wire seeds are
                // unique per shard-instance across all shard_types
                // (system_seed != ship_id != planet_seed), so a
                // single-field match unambiguously identifies the
                // dropped secondary's entry.
                //
                // The legacy `by_shard_type` index is keyed by type
                // only (lossy when multiple secondaries share a
                // type — only the last-write-wins entry survives),
                // so without knowing which `shard_type` to clear
                // there isn't a safe per-key removal. Leave it
                // untouched: downstream consumers (lighting / AR /
                // atmosphere) querying a dropped secondary's stale
                // entry will see one tick of stale data at most,
                // and the next live WS overwrites it. Reading from
                // a dead secondary's last good WS is preferable to
                // clearing ALL secondaries' entries (which used to
                // happen here): clearing the system secondary's WS
                // for ~50 ms — the gap between this disconnect and
                // the next eva_broadcast packet — removed every EVA
                // player from `RemotePlayers.by_id`, ageing their
                // visuals into the 500 ms despawn grace; in the
                // seamless-promote scenario `handle_preconnect`
                // cancels-and-replaces secondaries routinely, so
                // this fired several times during a single
                // transition and the EVA player visibly blinked.
                secondary.by_shard_key.retain(|k, _| k.seed != *seed);
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
    primary_shard: Res<PrimaryShard>,
    mut primary: ResMut<PrimaryWorldState>,
    mut secondary: ResMut<SecondaryWorldStates>,
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
            primary.last_tick_real_time = Some(Instant::now());

            // Defer secondary WorldState cleanup of the promoted shard until
            // the first primary WorldState packet actually arrives.
            if let Some(key) = primary_shard.current {
                secondary.by_shard_key.remove(&key);
                secondary.by_shard_type.remove(&key.shard_type);
            }
        }
    }
}

fn ingest_secondary(
    mut events: MessageReader<GameEvent>,
    mut secondary: ResMut<SecondaryWorldStates>,
) {
    for GameEvent(ev) in events.read() {
        if let NetEvent::SecondaryWorldState { shard_type, seed, ws } = ev {
            let key = ShardKey::new(*shard_type, *seed);
            // Same monotonic-tick guard as `ingest_primary`. Stale UDP
            // packets on a secondary's stream would otherwise rotate /
            // translate that secondary's chunks (or its observable
            // entities used for cross-shard pose lookup) backward by
            // one frame, then snap forward on the next fresh tick.
            // The guard runs per full `ShardKey` (the authoritative
            // map) so a slow tick from ship A doesn't suppress a fresh
            // tick from ship B with the same shard_type.
            let stale = secondary
                .by_shard_key
                .get(&key)
                .map(|prev| ws.tick <= prev.0.tick)
                .unwrap_or(false);
            if stale {
                continue;
            }
            let now = Instant::now();
            secondary
                .by_shard_key
                .insert(key, (ws.clone(), now));
            // Backward-compat secondary index — last writer wins
            // when multiple secondaries share a shard_type. Kept so
            // existing read-sites (lighting, AR, atmosphere, etc.)
            // don't need a refactor in this phase.
            secondary
                .by_shard_type
                .insert(*shard_type, (ws.clone(), now));
        }
    }
}
