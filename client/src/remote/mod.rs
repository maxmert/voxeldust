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
use std::time::{Duration, Instant};

use bevy::prelude::*;
use glam::{DQuat, DVec3};

use voxeldust_core::client_message::{EntityKind, ObservableEntityData};

use crate::net::NetConnection;
use crate::shard::{
    PrimaryShard, PrimaryWorldState, SecondaryWorldStates, ShardKey, WorldStateIngestSet,
};

/// Phase T7 — render-time interpolation lag for remote players +
/// ships. The renderer reads at `now - INTERPOLATION_DELAY` and
/// lerps between the entity's previous server snapshot and the
/// most recent one. With only two snapshots stored per entity,
/// `INTERPOLATION_DELAY` must equal the server tick interval for
/// the lerp boundary to land exactly on each shift; any larger
/// value forces the lerp to reach `cur` BEFORE the next tick
/// arrives, then snap back to the new `prev` (= old `cur`) when
/// the shift happens — visible as a position bounce. Any smaller
/// value leaves the lerp short of `cur` when the shift happens
/// and snaps forward to the new `prev` (= old `cur`) — same
/// bounce, opposite direction.
///
/// 50 ms = exactly one server tick @ 20 Hz: at steady state the
/// lerp sweeps prev→cur over a full tick interval and the post-
/// shift `alpha == 0` lands on the same data point the pre-shift
/// `alpha == 1` lerp reached — zero jump.
///
/// Network jitter (ticks arriving early or late) still causes
/// small bounces with a 2-snapshot buffer; a future polish phase
/// can widen to 3+ snapshots + larger delay for Source-style
/// jitter absorption. Local player is rendered at zero lag —
/// server-authoritative pose applied directly, no smoothing —
/// to match the no-client-prediction project policy and keep the
/// user's own avatar feel snappy.
pub const INTERPOLATION_DELAY: Duration = Duration::from_millis(50);

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
    /// Phase J: server-side tablet hold gate. When `true`, the
    /// client renders a held tablet visual + drives both arms via
    /// IK. `tablet_cursor_uv` is the cursor position in
    /// tablet-screen UV space [0, 1] for the left-hand finger IK.
    pub is_holding_tablet: bool,
    pub tablet_cursor_uv: glam::Vec2,
    // ── Phase T7 interpolation state ────────────────────────────────
    /// Spatial state at the prior server snapshot for this entity.
    /// `sync_remote_pose` lerps `(prev_position → position)` and
    /// slerps `(prev_rotation → rotation)` based on the time
    /// fraction `(render_time − prev_update) / (last_update −
    /// prev_update)`. Set to `position`/`rotation` on first sight
    /// so the render lerp is a no-op until two snapshots exist.
    pub prev_position: DVec3,
    pub prev_rotation: DQuat,
    /// Real-time when `position`/`rotation` were last updated (last
    /// server snapshot for this entity).
    pub last_update: Instant,
    /// Real-time when `prev_position`/`prev_rotation` were captured
    /// (the snapshot one server tick before `last_update`).
    pub prev_update: Instant,
    /// Server tick of the snapshot that produced the current pose.
    /// Used by `ingest_entity_update` to detect when a new server
    /// tick has arrived for this entity — only then is the current
    /// pose shifted to prev. Multiple ingest calls in the same
    /// frame (primary + secondary WorldStates) at the same server
    /// tick produce identical writes that don't waste prev slots.
    pub last_server_tick: u64,
    /// Phase T2 — body/head decoupling carried alongside position
    /// for interpolation continuity. Today body_yaw etc. are
    /// applied directly (no interpolation); future polish phase
    /// could interpolate body_yaw with wrap-aware lerp.
    pub prev_body_yaw: f32,
    /// `WorldStateData.game_time` from the WS that produced
    /// `position` / `rotation`. The shared server clock — every
    /// shard mirrors `system-shard.celestial_time` into this field
    /// (`game_time = scene.game_time` on ship-shard,
    /// `game_time = celestial_time.0` on system / planet / galaxy),
    /// so values are comparable ACROSS shards.
    ///
    /// Cross-shard render (`rebase_root_visuals`) reads `cam.pos` from
    /// PRIMARY WS and `remote.position` from a SECONDARY WS that ticks
    /// on an independent wall-clock cadence. Naive subtraction
    /// `remote - cam` reads operands from different simulation
    /// instants, producing per-frame oscillation at orbital scales.
    /// Both operands now carry game_time; `interpolated_pose_at_game_time`
    /// + `CameraWorldPos::pos_at_game_time` evaluate them at the same
    /// `target_game_time`, eliminating the cross-shard tick-phase
    /// mismatch.
    pub game_time: f64,
    /// `game_time` from the WS that produced `prev_position` /
    /// `prev_rotation`. Equal to `game_time` on first sight (lerp is
    /// a no-op) and on cross-shard shifts (`reconcile_interpolation`
    /// resets both to current); strictly less than `game_time` once
    /// two same-shard snapshots have been observed.
    pub prev_game_time: f64,
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

/// Tracks the active authoritative shard and transition cooldowns for remote players.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct AuthTrack {
    /// The active authoritative shard for this player.
    pub active_shard: ShardKey,
    /// Real-time when the last authoritative update was ingested.
    pub last_update: Instant,
    /// Shard transition cooldown: `Some((stale_shard, ignore_until))` locks out the stale shard's
    /// direct updates until `ignore_until` to absorb asynchronous server-side handoff delays.
    pub cooldown: Option<(ShardKey, Instant)>,
}

#[allow(clippy::too_many_arguments)]
fn track_remote_entities(
    primary: Res<PrimaryShard>,
    primary_ws: Res<PrimaryWorldState>,
    secondary_ws: Res<SecondaryWorldStates>,
    conn: Res<NetConnection>,
    mut players: ResMut<RemotePlayers>,
    mut ships: ResMut<RemoteShips>,
    mut debris: ResMut<RemoteDebris>,
    mut last_authoritative: Local<HashMap<u64, AuthTrack>>,
) {
    // Phase T7 — preserve LAST frame's snapshots so we can do prev-
    // tracking for render-time interpolation. ObservableEntity
    // broadcasts are full snapshots (not deltas), so we still need
    // to evict entries that left AOI; that happens at the bottom
    // via "drop anything not seen this tick". The reconcile step
    // inside `ingest` knows to copy `prev_*` forward when an
    // entity's underlying server tick hasn't advanced, so a stable
    // entity holds its lerp window across multiple frame ingests
    // of the same server snapshot.
    let prev_players: HashMap<u64, RemoteEntity> = std::mem::take(&mut players.by_id);
    let prev_ships: HashMap<u64, RemoteEntity> = std::mem::take(&mut ships.by_id);
    let prev_debris: HashMap<u64, RemoteEntity> = std::mem::take(&mut debris.by_id);

    let own_player_id = conn.player_id;
    let now = Instant::now();

    // Clean up tracking entries older than 5 seconds to prevent memory leaks/bloat.
    last_authoritative.retain(|_, track| now.saturating_duration_since(track.last_update) < Duration::from_secs(5));

    let mut ingested_priorities = HashMap::new();

    let primary_key = primary.current;
    if let (Some(ws), Some(key)) = (primary_ws.latest.as_ref(), primary_key) {
        ingest(
            ws, key, primary_key, own_player_id,
            &prev_players, &prev_ships, &prev_debris,
            &mut players, &mut ships, &mut debris,
            &mut last_authoritative, now,
            &mut ingested_priorities,
        );
    }
    // Iterate the authoritative per-`ShardKey` secondary map (NOT
    // `by_shard_type`). With `by_shard_type` (legacy lossy index),
    // multiple SHIP secondaries collapse into one slot and downstream
    // visual parenting can't tell which ship's `ChunkSource` to bind
    // to — the practical symptom was an EVA observer not seeing
    // in-ship players even though the SHIP secondary's WorldState
    // was streaming and listed them. `by_shard_key` carries the real
    // seed end-to-end (NetEvent → resource → ingest call), so
    // `SourceIndex.by_shard.get(&observer)` always lands on the
    // correct ChunkSource.
    for (&observer, (ws, _)) in &secondary_ws.by_shard_key {
        // Skip the secondary entry whose key matches the CURRENT
        // primary. When a shard transitions secondary → primary
        // (e.g. system promoted on Ship→EVA), its last
        // secondary-mode WS lingers in
        // `secondary_ws.by_shard_key[primary_key]` because
        // `SecondaryDisconnected` is never emitted on an in-place
        // promote. Without this skip, the LOCAL player's
        // `RemoteEntity` would be overwritten every frame from the
        // pre-promote snapshot, pinning the local FP visual at a
        // stale world position while the camera advances. The
        // PRIMARY's authoritative WS arrives via
        // `PrimaryWorldState` and is iterated above — this branch
        // only handles other secondaries. Paired with the one-shot
        // cleanup in `client/src/shard/worldstate.rs::reset_on_shard_change`
        // (drops the stale entry on `NetEvent::Connected`); the skip
        // here covers the same-tick race before that cleanup runs.
        // Skip the secondary entry whose key matches the CURRENT primary ONLY when
        // we have actually received the first primary WorldState. During the handshake/transition
        // gap (when primary_ws.latest is None), we fall back to rendering with this secondary
        // entry to prevent entities on that shard from disappearing.
        if Some(observer) == primary_key && primary_ws.latest.is_some() {
            continue;
        }
        ingest(
            ws, observer, primary_key, own_player_id,
            &prev_players, &prev_ships, &prev_debris,
            &mut players, &mut ships, &mut debris,
            &mut last_authoritative, now,
            &mut ingested_priorities,
        );
    }

    // Carry forward any players/ships who had transient UDP drops or were skipped in this frame
    for (player_id, prev_entity) in prev_players {
        if !players.by_id.contains_key(&player_id) {
            if let Some(track) = last_authoritative.get(&player_id) {
                if now.saturating_duration_since(track.last_update) < Duration::from_millis(1500) {
                    players.by_id.insert(player_id, prev_entity);
                }
            }
        }
    }

    for (ship_id, prev_entity) in prev_ships {
        if !ships.by_id.contains_key(&ship_id) {
            if now.saturating_duration_since(prev_entity.last_update) < Duration::from_millis(1500) {
                ships.by_id.insert(ship_id, prev_entity);
            }
        }
    }
}


/// Frame-level ingestion priority for remote entity updates.
///
/// Under real-world conditions with asynchronous UDP updates streaming from multiple
/// shards in the same frame, this priority hierarchy ensures that authoritative direct updates
/// and high-fidelity direct projections are never overwritten by stale/late forwarded projections.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
enum IngestionPriority {
    /// Forwarded projection (e.g. ship/planet shards forwarding surrounding system-space entities).
    /// These are staler and double-transformed compared to direct projections.
    ForwardedProjection = 1,
    /// Direct projection from the host system shard (type 1) representing entities in other shards.
    DirectProjection = 2,
    /// Direct authoritative broadcast from the shard that actually owns and simulates the entity.
    Authoritative = 3,
}

/// Computes the ingestion priority for an entity update.
fn get_ingestion_priority(
    e: &voxeldust_core::client_message::ObservableEntityData,
    observer: ShardKey,
) -> IngestionPriority {
    let is_direct = observer.shard_type == e.shard_type;
    if is_direct {
        IngestionPriority::Authoritative
    } else if observer.shard_type == 1 {
        // Direct projection from SystemShard (host)
        IngestionPriority::DirectProjection
    } else {
        // Forwarded projection from ship/planet shards
        IngestionPriority::ForwardedProjection
    }
}

/// Should this `ObservableEntity` be ingested when arriving via this
/// `observer` shard's WorldState?
///
/// **Players** are authoritative on their owning shard. The system-
/// shard's AOI broadcast projects EVA + surface players into ship-
/// shards' WorldStates so distant observers can see them; if the
/// client ALSO has a direct connection to the player's owning shard
/// (the common case — the SYSTEM secondary is always pre-connected
/// in scene-context mode), those projections are redundant AND
/// dangerous: their `observer` ShardKey points to the projecting
/// shard, not the owning one, so `make_remote`'s `shard = observer`
/// rule would set the wrong `RemoteEntity.shard` and `reparent_
/// visuals_on_shard_change` would yank the visual under the wrong
/// `ChunkSource`. Skip the projection in that case — the
/// authoritative shard's broadcast covers the player on the next
/// tick with the correct shard binding.
///
/// **Ships** are independently authoritative — there's no AOI
/// projection that shadows their direct broadcast — so this filter
/// only applies to player kinds.
fn should_skip_cross_shard_projection(
    e: &voxeldust_core::client_message::ObservableEntityData,
    observer: ShardKey,
    last_authoritative: &HashMap<u64, AuthTrack>,
    now: Instant,
) -> bool {
    let is_player = matches!(
        e.kind,
        EntityKind::EvaPlayer | EntityKind::GroundedPlayer | EntityKind::Seated
    );
    if !is_player {
        return false;
    }
    // Same-shard direct broadcast — never skip. Checks both shard type and specific instance seed/shard_id.
    let is_direct = observer.shard_type == e.shard_type;
    if is_direct {
        return false;
    }
    // AAA temporal fallback: if we received an authoritative update for this entity from its
    // owning shard type within the last 1500 ms, strictly skip the cross-shard projection.
    // This provides robust insulation against UDP packet jitter, packet loss, or connection drops.
    if let Some(track) = last_authoritative.get(&e.entity_id) {
        if track.active_shard.shard_type == e.shard_type && now.saturating_duration_since(track.last_update) < Duration::from_millis(1500) {
            return true;
        }
    }
    false
}

#[allow(clippy::too_many_arguments)]
fn ingest(
    ws: &voxeldust_core::client_message::WorldStateData,
    observer: ShardKey,
    primary_key: Option<ShardKey>,
    own_player_id: u64,
    prev_players: &HashMap<u64, RemoteEntity>,
    prev_ships: &HashMap<u64, RemoteEntity>,
    prev_debris: &HashMap<u64, RemoteEntity>,
    players: &mut RemotePlayers,
    ships: &mut RemoteShips,
    debris: &mut RemoteDebris,
    last_authoritative: &mut HashMap<u64, AuthTrack>,
    now: Instant,
    ingested_priorities: &mut HashMap<u64, IngestionPriority>,
) {
    let is_primary_observer = primary_key == Some(observer);
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
        // The LOCAL player follows the PRIMARY shard authoritatively.
        // During an in-flight transition the player can be present in
        // both the soon-to-be-old primary's WorldState (still serving
        // the player as a ground/seated entity) AND the soon-to-be-
        // new primary's secondary broadcast (already serving them as
        // an EVA player) — both legitimately on their authoritative
        // shards, both with the same `entity_id == own_player_id`.
        // Last-write-wins would flip `RemoteEntity.shard` between
        // them and `reparent_visuals_on_shard_change` would yank the
        // local visual + camera between two `ChunkSource` parents
        // each tick. Anchoring the local player to the primary alone
        // makes the visual stable for the duration of the transition
        // — the actual swap happens atomically when `PrimaryShard`
        // changes (via either seamless promote-in-place or full
        // ShardRedirect), at which point the new primary's WorldState
        // is the only stream still feeding the local player_id.
        let is_player = matches!(
            e.kind,
            EntityKind::EvaPlayer | EntityKind::GroundedPlayer | EntityKind::Seated
        );
        if is_player && e.entity_id == own_player_id && !is_primary_observer {
            continue;
        }

        if is_player && e.entity_id != own_player_id {
            let is_direct = observer.shard_type == e.shard_type;
            let skip_proj = should_skip_cross_shard_projection(e, observer, last_authoritative, now);
            info!(
                player_id = e.entity_id,
                name = %e.name,
                observer = %observer,
                observer_seed = observer.seed,
                e_shard_type = e.shard_type,
                e_shard_id = e.shard_id,
                is_direct,
                priority = ?get_ingestion_priority(e, observer),
                skip_proj,
                pos = ?(e.position.x, e.position.y, e.position.z),
                "remote: player ingest debug"
            );
        }

        // Record same-shard authoritative updates for players to drive temporal deduplication
        let is_direct = observer.shard_type == e.shard_type;
        if is_player && is_direct {
            let mut skip_update = false;
            if let Some(track) = last_authoritative.get_mut(&e.entity_id) {
                if let Some((stale_shard, ignore_until)) = track.cooldown {
                    if now < ignore_until && stale_shard == observer {
                        skip_update = true;
                    }
                }
                if !skip_update {
                    if track.active_shard != observer {
                        // Shard transition detected! Put old shard on a 1-second cooldown
                        let old_shard = track.active_shard;
                        track.cooldown = Some((old_shard, now + Duration::from_millis(1000)));
                        track.active_shard = observer;
                        info!(
                            player_id = e.entity_id,
                            %old_shard,
                            new_shard = %observer,
                            "remote: player authority shard transition detected, old shard locked out for 1000ms"
                        );
                    }
                    track.last_update = now;
                }
            } else {
                last_authoritative.insert(
                    e.entity_id,
                    AuthTrack {
                        active_shard: observer,
                        last_update: now,
                        cooldown: None,
                    },
                );
            }
            if skip_update {
                continue;
            }
        }

        let priority = get_ingestion_priority(e, observer);
        if let Some(&existing_priority) = ingested_priorities.get(&e.entity_id) {
            if priority < existing_priority {
                continue; // Skip this update, we already have a higher-priority one in this frame
            }
        }

        if should_skip_cross_shard_projection(e, observer, last_authoritative, now) {
            continue;
        }
        let mut remote = make_remote(e, observer, now, ws.game_time);
        match e.kind {
            EntityKind::Ship => {
                reconcile_interpolation(&mut remote, prev_ships.get(&e.entity_id), ws.tick, now);
                ships.by_id.insert(e.entity_id, remote);
                ingested_priorities.insert(e.entity_id, priority);
            }
            EntityKind::EvaPlayer
            | EntityKind::GroundedPlayer
            | EntityKind::Seated => {
                reconcile_interpolation(&mut remote, prev_players.get(&e.entity_id), ws.tick, now);
                players.by_id.insert(e.entity_id, remote);
                ingested_priorities.insert(e.entity_id, priority);
            }
            // Future DEBRIS kinds plug in here without a core change.
        }
        // Silence "unused variable" when the enum only has 4 variants
        // currently; avoids `_` in match arms that would suppress
        // future variants from triggering a compile warning.
        let _ = debris;
        let _ = prev_debris;
    }
}

/// Phase T7 — reconcile a freshly-built `RemoteEntity` against its
/// previous-frame state for interpolation continuity.
///
///   * **Entity unchanged** (server tick same as previous frame's
///     last-known tick): the entity is being ingested from a
///     repeated WorldState. Copy the prev frame's
///     `prev_position` / `prev_rotation` / `prev_update` /
///     `last_update` forward so the lerp window stays anchored to
///     the actual server-tick boundary, not the per-frame ingest
///     time.
///   * **Entity advanced** (server tick is higher than previously
///     observed): shift the prev frame's current → this frame's
///     prev. `last_update = now`, fresh anchor.
///   * **Entity first sighting** (no prior frame state): leave
///     `prev_* == current` so the lerp is a no-op until a second
///     snapshot arrives next tick.
fn reconcile_interpolation(
    new: &mut RemoteEntity,
    prior: Option<&RemoteEntity>,
    server_tick: u64,
    now: Instant,
) {
    let Some(prior) = prior else {
        // First sighting — `make_remote` already set prev = current.
        new.last_server_tick = server_tick;
        return;
    };
    // Phase T7 — if the prior snapshot was sourced from a DIFFERENT
    // authoritative shard than this one (e.g. P1 transitioning from
    // ship-shard's GroundedPlayer broadcast to system-shard's EVA
    // broadcast during a hull exit), the two snapshots are in
    // INCOMPATIBLE coordinate frames (ship-local ≈ (5,1,0) vs.
    // system-space ≈ (3.5e9, …)). Lerping between them produces a
    // visual that flies across the world every frame — the
    // "blinks to a position aside" symptom. Reset prev = current
    // so the lerp is a no-op until two consecutive snapshots in
    // the SAME shard arrive.
    //
    // Server-tick numbers are also not comparable across shards
    // (each shard has its own tick counter), so the shift below
    // would be triggered every frame even when no transition was
    // happening, churning the lerp window pointlessly.
    if prior.shard != new.shard {
        new.last_server_tick = server_tick;
        // prev_* stay at the make_remote defaults (= current),
        // i.e. lerp is a no-op until the next same-shard tick.
        // Same rationale for `prev_game_time`: lerping a position
        // across a shard boundary on an incompatible coordinate
        // frame is the original "blink across the world" symptom.
        new.last_update = now;
        new.prev_update = now;
        // `new.game_time` / `new.prev_game_time` already equal
        // `ws_game_time` from `make_remote`; no shift.
        return;
    }
    if server_tick > prior.last_server_tick {
        // New server tick — shift prior's current into new's prev.
        new.prev_position = prior.position;
        new.prev_rotation = prior.rotation;
        new.prev_body_yaw = prior.body_yaw;
        new.prev_update = prior.last_update;
        new.last_update = now;
        new.last_server_tick = server_tick;
        // Mirror the wall-clock shift on game_time so
        // `interpolated_pose_at_game_time` has a non-degenerate
        // lerp window keyed on the server-authoritative clock.
        new.prev_game_time = prior.game_time;
        // `new.game_time` already set from `ws_game_time` in `make_remote`.
    } else {
        // Same server tick re-ingested this frame. Carry prior's
        // interpolation window forward verbatim so successive frames
        // continue to lerp from the same anchor pair.
        new.prev_position = prior.prev_position;
        new.prev_rotation = prior.prev_rotation;
        new.prev_body_yaw = prior.prev_body_yaw;
        new.prev_update = prior.prev_update;
        new.last_update = prior.last_update;
        new.last_server_tick = prior.last_server_tick;
        // Game-time mirror: keep the same anchor pair across all
        // re-ingests of this tick (primary + secondary WSes at the
        // same server-tick don't waste prev slots).
        new.prev_game_time = prior.prev_game_time;
        new.game_time = prior.game_time;
    }
}

impl RemoteEntity {
    /// Interpolated `(position, rotation)` at a target
    /// **server-authoritative game-time** instead of a wall-clock
    /// instant. The lerp window is `[prev_game_time, game_time]`
    /// derived from the `WorldStateData.game_time` field of each
    /// snapshot, NOT the per-frame wall-clock arrival time.
    ///
    /// Use this whenever the caller composes the interpolated value
    /// with another series that is also indexed on game-time — most
    /// importantly `CameraWorldPos::pos_at_game_time` for the
    /// cross-shard subtraction in
    /// `client/src/character/render.rs::rebase_root_visuals`. Without
    /// a shared temporal reference, `delta = remote − cam` reads its
    /// operands from independent 20 Hz WS streams and oscillates by
    /// `velocity × tick_phase_offset` (≈ 1.5–4 km per frame at
    /// orbital scales).
    ///
    /// Same clamp semantics as [`interpolated_pose`]:
    ///   * `target >= game_time` → return current.
    ///   * `target <= prev_game_time` → return prev.
    ///   * Between — lerp/slerp by `(target − prev_game_time) / (game_time − prev_game_time)`.
    ///
    /// Falls through to the current pose when `game_time ==
    /// prev_game_time` (first sight, or just-reset on a cross-shard
    /// shift); the denominator would be zero in that case.
    pub fn interpolated_pose_at_game_time(&self, target: f64) -> (DVec3, DQuat) {
        let span = self.game_time - self.prev_game_time;
        if span <= 0.0 || target >= self.game_time {
            return (self.position, self.rotation);
        }
        if target <= self.prev_game_time {
            return (self.prev_position, self.prev_rotation);
        }
        let alpha = ((target - self.prev_game_time) / span).clamp(0.0, 1.0);
        let pos = self.prev_position.lerp(self.position, alpha);
        let rot = self.prev_rotation.slerp(self.rotation, alpha);
        (pos, rot)
    }
}

fn make_remote(
    e: &ObservableEntityData,
    observer: ShardKey,
    now: Instant,
    ws_game_time: f64,
) -> RemoteEntity {
    // Per-kind shard-key composition.
    //
    // PLAYERS are always observed by the WorldState whose authoritative
    // shard owns them: a ground player on this ship is broadcast by
    // THIS ship-shard; an EVA player by the system-shard; a planet
    // surface player by their planet-shard. So for players the
    // `observer` ShardKey (= the WorldState's authoritative
    // `ShardKey`, looked up via `Secondaries.runtimes` for secondaries
    // and `PrimaryShard.current` for the primary) is the correct
    // entry into `SourceIndex.by_shard` for parenting visuals.
    //
    // Cross-shard AOI projections (e.g. system-shard projecting an
    // EVA player into a ship-shard's WorldState so distant ships
    // can see EVA traffic without opening a system secondary) are
    // filtered out upstream in `should_skip_cross_shard_projection`
    // when we have a direct connection to the authoritative shard,
    // so by the time we reach this function `observer` either IS
    // the authoritative shard or is an acceptable fallback (e.g.
    // viewer doesn't have a direct connection to the player's
    // owning shard, so the projection IS the only available view).
    //
    // SHIPS are different: the system-shard's AOI feed exposes nearby
    // ships in its primary `entities[]` even though each ship lives
    // on its own SHIP shard. For those entries we honour `e.shard_id`
    // so the renderer parents under the secondary SHIP `ChunkSource`
    // (`find_secondary_pose` does the symmetric lookup using the
    // same id).
    //
    // The shard_id-vs-seed asymmetry is load-bearing: ship-shard
    // stamps `entities[].shard_id = config.shard_id.0` (the
    // orchestrator-assigned id), which is *not* the wire `seed` used
    // as `ShardKey.seed`. For PLAYERS that mismatch silently broke
    // SourceIndex lookups in earlier code that tried to use
    // `e.shard_id` for them; for SHIPS the secondary registration
    // uses `find_secondary_pose`'s own match against `shard_id`,
    // bypassing `SourceIndex.by_shard` entirely.
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
    let position = DVec3::new(e.position.x, e.position.y, e.position.z);
    let rotation = DQuat::from_xyzw(e.rotation.x, e.rotation.y, e.rotation.z, e.rotation.w);
    RemoteEntity {
        entity_id: e.entity_id,
        kind: e.kind,
        position,
        rotation,
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
        is_holding_tablet: e.is_holding_tablet,
        tablet_cursor_uv: e.tablet_cursor_uv,
        // Phase T7 — first-sighting defaults. `reconcile_interpolation`
        // overrides prev_* with the actual prior frame's snapshot
        // (when one exists) so the lerp window aligns to the
        // server-tick boundary, not the per-frame ingest.
        prev_position: position,
        prev_rotation: rotation,
        prev_body_yaw: e.body_yaw,
        last_update: now,
        prev_update: now,
        last_server_tick: 0,
        // First-sighting: `game_time == prev_game_time` so
        // `interpolated_pose_at_game_time` returns the current pose
        // (lerp denominator is zero → no-op). `reconcile_interpolation`
        // shifts `prev_game_time` on the next same-shard tick.
        game_time: ws_game_time,
        prev_game_time: ws_game_time,
    }
}
