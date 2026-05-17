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
    secondaries: Res<crate::shard::Secondaries>,
    conn: Res<NetConnection>,
    mut players: ResMut<RemotePlayers>,
    mut ships: ResMut<RemoteShips>,
    mut debris: ResMut<RemoteDebris>,
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
    // Pre-compute which shard types we have a direct connection to.
    // Used to filter out cross-shard PLAYER projections — an EVA
    // player projected through a SHIP-shard's AOI would otherwise
    // overwrite the same player's authoritative entry from the
    // SYSTEM secondary's broadcast, parenting the visual under the
    // wrong `ChunkSource`. See `is_authoritative_for_player`.
    let mut connected_shard_types: u8 = 0;
    if let Some(key) = primary.current {
        connected_shard_types |= 1 << key.shard_type;
    }
    for k in secondaries.runtimes.keys() {
        connected_shard_types |= 1 << k.shard_type;
    }

    let primary_key = primary.current;
    if let Some(ws) = primary_ws.latest.as_ref() {
        if let Some(key) = primary_key {
            ingest(
                ws, key, primary_key, own_player_id, connected_shard_types,
                &prev_players, &prev_ships, &prev_debris,
                &mut players, &mut ships, &mut debris,
            );
        }
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
        ingest(
            ws, observer, primary_key, own_player_id, connected_shard_types,
            &prev_players, &prev_ships, &prev_debris,
            &mut players, &mut ships, &mut debris,
        );
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
    connected_shard_types: u8,
) -> bool {
    let is_player = matches!(
        e.kind,
        EntityKind::EvaPlayer | EntityKind::GroundedPlayer | EntityKind::Seated
    );
    if !is_player {
        return false;
    }
    // Same-shard direct broadcast — never skip.
    if observer.shard_type == e.shard_type {
        return false;
    }
    // Cross-shard projection. Skip iff we have a direct connection
    // to the player's authoritative shard type — its broadcast is
    // about to land with the correct shard binding.
    (connected_shard_types & (1 << e.shard_type)) != 0
}

fn ingest(
    ws: &voxeldust_core::client_message::WorldStateData,
    observer: ShardKey,
    primary_key: Option<ShardKey>,
    own_player_id: u64,
    connected_shard_types: u8,
    prev_players: &HashMap<u64, RemoteEntity>,
    prev_ships: &HashMap<u64, RemoteEntity>,
    prev_debris: &HashMap<u64, RemoteEntity>,
    players: &mut RemotePlayers,
    ships: &mut RemoteShips,
    debris: &mut RemoteDebris,
) {
    let is_primary_observer = primary_key == Some(observer);
    let now = Instant::now();
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
        if should_skip_cross_shard_projection(e, observer, connected_shard_types) {
            continue;
        }
        let mut remote = make_remote(e, observer, now);
        match e.kind {
            EntityKind::Ship => {
                reconcile_interpolation(&mut remote, prev_ships.get(&e.entity_id), ws.tick, now);
                ships.by_id.insert(e.entity_id, remote);
            }
            EntityKind::EvaPlayer
            | EntityKind::GroundedPlayer
            | EntityKind::Seated => {
                reconcile_interpolation(&mut remote, prev_players.get(&e.entity_id), ws.tick, now);
                players.by_id.insert(e.entity_id, remote);
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
        new.last_update = now;
        new.prev_update = now;
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
    }
}

impl RemoteEntity {
    /// Phase T7 — return the interpolated `(position, rotation)`
    /// for rendering at `target_time`. Looks up the entity's
    /// previous + current snapshots and lerps linearly between
    /// them based on the time fraction.
    ///
    /// `target_time` is expected to be `now - INTERPOLATION_DELAY`
    /// (= `now - 75 ms` for the standard 20 Hz server cadence).
    ///
    ///   * `target_time <= prev_update` — render at prev (we're
    ///     behind both snapshots; happens immediately after
    ///     respawn before any prev exists, or at the very first
    ///     sight of a new entity).
    ///   * `target_time >= last_update` — render at current
    ///     (we're caught up to the latest snapshot; happens
    ///     when network stalls past the next expected tick).
    ///   * Between — lerp/slerp with `alpha = (target_time −
    ///     prev_update) / (last_update − prev_update)`.
    ///
    /// Falls through to non-interpolated current pose when prev
    /// and current have identical timestamps (the entity hasn't
    /// received a second snapshot yet); the lerp ratio's
    /// denominator would be zero in that case.
    pub fn interpolated_pose(&self, target_time: Instant) -> (DVec3, DQuat) {
        let span = self.last_update.saturating_duration_since(self.prev_update);
        if span.is_zero() {
            return (self.position, self.rotation);
        }
        if target_time >= self.last_update {
            return (self.position, self.rotation);
        }
        if target_time <= self.prev_update {
            return (self.prev_position, self.prev_rotation);
        }
        let elapsed = target_time.saturating_duration_since(self.prev_update);
        let alpha = (elapsed.as_secs_f64() / span.as_secs_f64()).clamp(0.0, 1.0);
        let pos = self.prev_position.lerp(self.position, alpha);
        // slerp normalises the quaternion; safe even when prev_rotation
        // and rotation have opposite hemisphere signs (slerp goes the
        // short way).
        let rot = self.prev_rotation.slerp(self.rotation, alpha);
        (pos, rot)
    }
}

fn make_remote(e: &ObservableEntityData, observer: ShardKey, now: Instant) -> RemoteEntity {
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
    }
}
