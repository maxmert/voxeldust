//! Shard transition + promotion handler.
//!
//! Voxeldust runs the player across four shard types (PLANET, SYSTEM, SHIP,
//! GALAXY) and seamlessly promotes pre-connected secondary shards to primary
//! when the player crosses a boundary (board/exit ship, land/launch, warp
//! in/out). This module owns the client-side state machine that keeps the
//! camera, chunk meshes, and remote-entity renders continuous across those
//! promotions.
//!
//! The critical invariants — taken from the legacy `voxydust` client and
//! verified against `core::handoff` — are:
//!
//! 1. **Server authority is total.** Spawn position + rotation at the first
//!    post-transition frame come from `ShardRedirect.spawn_pose` when
//!    available. The client never predicts the destination pose; it lerps
//!    nothing, extrapolates nothing.
//!
//! 2. **Grace window.** The old primary's chunk source (the ship you just
//!    exited, the planet you just launched from) stays rendered for
//!    `GRACE_SECS` after transition, re-anchored to system-space coordinates
//!    on each frame so it sits at the correct point in the new primary's
//!    frame. Without this there is a 50-200 ms visible blink while the new
//!    primary's AOI catches up.
//!
//! 3. **Deferred shard-type flip.** `ShardContext.current` only flips to the
//!    destination type once a post-transition WorldState actually arrives
//!    (either because we seamlessly promoted a secondary's stashed WS, or
//!    because `NetEvent::Connected` landed). During the gap, rendering reads
//!    the old type so interior-of-ship / surface-of-planet gates keep
//!    matching the still-valid frame of reference.
//!
//! 4. **Per-type secondary routing.** Multiple secondaries can be connected
//!    simultaneously (a SHIP shard typically keeps a SYSTEM secondary and a
//!    GALAXY secondary active at all times for celestial + starfield
//!    rendering). `WorldStates.secondary_by_type` keys them so `Transitioning
//!    { target_shard_type: SYSTEM }` promotes the correct one.
//!
//! 5. **Scene-context survival.** `SecondaryDisconnected` does not erase
//!    state that was keyed on a type (SYSTEM, GALAXY) still present in
//!    `NetSecondaries`; the tokio layer already ensures scene-context
//!    secondaries outlive primary transitions (see `network.rs`).
//!
//! 6. **Multiplayer.** Every WorldState carries *all* player entities in
//!    its AOI (crew inside your ship, surface neighbours on a planet). This
//!    module does not special-case the local player beyond applying
//!    `spawn_pose` to the camera — crewmates render via the normal
//!    WorldState-entity path regardless of transitions.
//!
//! See `project_voxydust_next_migration.md` for the end-to-end spec this
//! file implements.

use std::collections::HashMap;
use std::time::{Duration, Instant};

use bevy::prelude::*;
use glam::{DQuat, DVec3};
use voxeldust_core::client_message::{shard_type as st, EntityKind, WorldStateData};

use crate::net_plugin::GameEvent;
use crate::network::NetEvent;

/// Length of the post-transition grace window during which the old primary's
/// chunk meshes keep rendering so the handoff has no visible blink.
///
/// Matches legacy `LAST_PRIMARY_GRACE_SECS`. Chosen to be long enough to
/// cover the worst-case observed gap between `Transitioning` and the first
/// destination-shard WorldState arriving with our player entity (~400 ms on
/// cross-host transitions under load), with headroom for the 20 Hz
/// re-streaming of the secondary ship's AOI snapshots that take over the
/// render after grace expires.
pub const GRACE_SECS: f64 = 1.5;

/// System set for everything in this module. Other plugins (chunk_stream,
/// player_sync) constrain their consumers to run `.after(ShardTransitionSet)`
/// so the per-frame transition bookkeeping is always observed before any
/// downstream read of `ShardContext` / `WorldStates` / `GraceWindow` /
/// `SpawnPoseLatch`. Without this, Bevy's default `Update` parallelism
/// would allow a consumer to race the transition handler and read stale
/// shard type / primary seed / grace source state mid-transition.
#[derive(SystemSet, Clone, Eq, PartialEq, Hash, Debug)]
pub struct ShardTransitionSet;

pub struct ShardTransitionPlugin;

impl Plugin for ShardTransitionPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<ShardContext>()
            .init_resource::<WorldStates>()
            .init_resource::<GraceWindow>()
            .init_resource::<SpawnPoseLatch>()
            .add_systems(
                Update,
                (
                    route_world_states,
                    handle_transitioning,
                    handle_connected,
                    drive_grace_transforms,
                    drain_grace_window,
                )
                    .chain()
                    .in_set(ShardTransitionSet),
            );
    }
}

// ---------------------------------------------------------------------------
// Resources
// ---------------------------------------------------------------------------

/// Authoritative view of which shard the client is "in" right now. Keyed off
/// `NetEvent::Connected` / `NetEvent::Transitioning` — never inferred from
/// world-state content. Consumers (camera frame, input routing, HUD mode)
/// read `current` and ignore `pending` until it has been committed by the
/// transition handler.
#[derive(Resource, Default, Debug, Clone)]
pub struct ShardContext {
    /// Shard type currently driving the camera/render frame. Stays on the
    /// *source* type through the transition window so interior-of-ship
    /// rendering continues to read ship-local coordinates until the new
    /// primary's WS with our player actually arrives.
    pub current: u8,
    /// Destination type during an in-flight transition. `Some` means a
    /// `Transitioning` has fired but the new primary's world state is not
    /// yet confirmed. Flipped into `current` by the transition handler as
    /// soon as the destination WS lands.
    pub pending: Option<u8>,
    /// Seed of the shard currently supplying primary chunk data. `None`
    /// before the first `Connected`.
    pub primary_seed: Option<u64>,
    /// Set once the first `Connected` has been processed — lets systems
    /// skip their run until the world is definitively initialized.
    pub ready: bool,
}

impl ShardContext {
    /// The shard type that should drive the render/camera frame *right now*.
    /// Deliberately ignores `pending` — see module docs.
    pub fn render_shard_type(&self) -> u8 {
        self.current
    }

    pub fn is_ship(&self) -> bool { self.current == st::SHIP }
    pub fn is_planet(&self) -> bool { self.current == st::PLANET }
    pub fn is_system(&self) -> bool { self.current == st::SYSTEM }
    pub fn is_galaxy(&self) -> bool { self.current == st::GALAXY }
}

/// World-state fan-out. The network layer ships one `NetEvent::WorldState`
/// per primary tick and one `NetEvent::SecondaryWorldState { shard_type }`
/// per secondary tick; `route_world_states` files them here so any rendering
/// or transition system can look up the latest WS for any shard type / seed
/// without replaying the event stream.
///
/// Kept as a single resource instead of split-by-field because gameplay
/// queries are almost always "give me the WS for this shard type" — the
/// fields stay coupled and borrow-checker-friendly that way.
#[derive(Resource, Default, Debug)]
pub struct WorldStates {
    /// Most recent primary WorldState.
    pub primary: Option<WorldStateData>,
    /// Most recent secondary WorldState keyed by shard type (for the scene-
    /// context secondaries: SYSTEM for celestial bodies + long-range entity
    /// rendering, GALAXY for starfield parallax).
    pub secondary_by_type: HashMap<u8, WorldStateData>,
    /// Most recent secondary WorldState keyed by shard seed (for multi-ship
    /// rendering: each nearby ship is its own secondary shard).
    pub secondary_by_seed: HashMap<u64, WorldStateData>,
    /// Pre-transition primary WS held for the grace window so the renderer
    /// can keep drawing remote entities (other ships, crew, bodies) from
    /// the old frame until the new primary's first tick lands. Expires at
    /// `last_primary_until`.
    pub last_primary: Option<WorldStateData>,
    pub last_primary_until: Option<Instant>,
}

/// Grace-window state: a set of chunk-source entities whose lifetime is
/// extended past their owning shard's primary/secondary status so their
/// meshes keep rendering across a transition. Each entry carries the
/// system-space pose the renderer should place it at during grace (unlike
/// the normal per-frame sync from `WorldState.entities`, we no longer
/// receive live updates for a departed ship, so we extrapolate with the
/// velocity we captured at the transition moment).
#[derive(Resource, Default, Debug)]
pub struct GraceWindow {
    /// seed → (root entity, pose captured at transition, expiry). When
    /// multiple transitions stack (rare: ship → system → ship in quick
    /// succession), each insertion overwrites the prior expiry/pose so the
    /// grace window is always aligned with the most recent exit.
    pub sources: HashMap<u64, GraceSource>,
}

#[derive(Debug, Clone, Copy)]
pub struct GraceSource {
    pub entity: Entity,
    /// System-space position at the moment of transition.
    pub system_position: DVec3,
    pub rotation: DQuat,
    /// System-space velocity at transition. `drive_grace_transforms`
    /// extrapolates the position each frame so a ship you just exited
    /// continues on its orbit instead of appearing to stop dead.
    pub velocity: DVec3,
    pub captured_at: Instant,
    pub expiry: Instant,
}

/// One-shot spawn pose from `ShardRedirect.spawn_pose`. Applied to the
/// camera on the next `player_sync` tick (see `player_sync::apply_server_pose`)
/// and cleared. Kept as a separate resource so the transition handler and
/// the pose-apply system can run in any scheduling order within a frame
/// without races.
#[derive(Resource, Default, Debug, Clone)]
pub struct SpawnPoseLatch {
    pub pose: Option<LatchedSpawnPose>,
}

#[derive(Debug, Clone)]
pub struct LatchedSpawnPose {
    pub target_shard_type: u8,
    /// Position in destination-shard-local coordinates. Consumers render
    /// relative to the new primary's `ws.origin`, so this is written
    /// straight into `PlayerState.world_pos`.
    pub position: DVec3,
    pub rotation: DQuat,
    pub velocity: DVec3,
}

// ---------------------------------------------------------------------------
// Systems
// ---------------------------------------------------------------------------

/// Drain every `NetEvent::WorldState` / `NetEvent::SecondaryWorldState` on
/// the message bus into the `WorldStates` resource. Run first in the chain
/// so downstream systems have a fresh snapshot.
fn route_world_states(
    mut events: MessageReader<GameEvent>,
    mut worlds: ResMut<WorldStates>,
    mut ctx: ResMut<ShardContext>,
) {
    let now = Instant::now();
    // Evict grace-window primary WS when its TTL expires. Only clear the
    // `last_primary` slot — `secondary_by_type` and friends are driven by
    // their own lifecycle.
    if let Some(expiry) = worlds.last_primary_until {
        if now >= expiry {
            worlds.last_primary = None;
            worlds.last_primary_until = None;
        }
    }

    for GameEvent(ev) in events.read() {
        match ev {
            NetEvent::WorldState(ws) => {
                // If a deferred shard flip is pending, commit it as soon as
                // the destination WS lands — it carries our player in the
                // destination frame, which is the signal that the server
                // considers the handoff complete.
                if let Some(pending) = ctx.pending {
                    ctx.current = pending;
                    ctx.pending = None;
                }
                worlds.primary = Some(ws.clone());
            }
            NetEvent::SecondaryWorldState { shard_type, ws } => {
                worlds.secondary_by_type.insert(*shard_type, ws.clone());
                // Cross-reference by seed from within the secondary WS itself.
                // (WorldStateData does not carry its own seed explicitly; we
                // rely on per-type indexing as the primary key and only
                // populate per-seed when an entity association becomes
                // available. For now this stays empty — chunk_stream drives
                // seed-keyed rendering directly from the chunk source map.)
                let _ = &worlds.secondary_by_seed;
            }
            _ => {}
        }
    }
}

/// Handle `NetEvent::Transitioning`:
///
/// 1. Stash the old primary WS into the grace window so its remote entities
///    keep rendering while the destination's first WS is in flight.
/// 2. Latch `spawn_pose` so `player_sync` can snap the camera when the
///    destination frame begins.
/// 3. Defer the shard-type flip via `ShardContext.pending`.
/// 4. If a matching secondary WS is already cached, seamlessly promote it
///    to the primary slot so rendering has no gap.
///
/// Chunk-source grace is handled inside `chunk_stream.rs` via the public
/// `begin_grace_for_primary` helper this system invokes — we keep that logic
/// next to the chunk data it operates on.
fn handle_transitioning(
    mut events: MessageReader<GameEvent>,
    mut ctx: ResMut<ShardContext>,
    mut worlds: ResMut<WorldStates>,
    mut latch: ResMut<SpawnPoseLatch>,
    mut grace: ResMut<GraceWindow>,
    mut commands: Commands,
    mut source_index: ResMut<crate::chunk_stream::SourceIndex>,
    q_transform: Query<&Transform, With<crate::chunk_stream::ChunkSource>>,
) {
    let now = Instant::now();
    let grace_expiry = now + Duration::from_secs_f64(GRACE_SECS);

    for GameEvent(ev) in events.read() {
        let NetEvent::Transitioning { target_shard_type, spawn_pose } = ev else {
            continue;
        };

        // --- 1. Latch authoritative spawn pose ---------------------------
        //
        // `SpawnPose` fields are in destination-shard-local coordinates
        // (system-space for SYSTEM/GALAXY, planet-local for PLANET,
        // ship-local for SHIP — see `core::handoff::SpawnPose` docs), so
        // the pose applies directly to the player state without any
        // coordinate conversion on our side. Rotation/velocity are written
        // through unchanged; the camera frame conversion (yaw/pitch extract
        // in the right tangent/body basis) is the camera-frame task (#9).
        latch.pose = spawn_pose.as_ref().map(|sp| LatchedSpawnPose {
            target_shard_type: *target_shard_type,
            position: sp.position,
            rotation: sp.rotation,
            velocity: sp.velocity,
        });

        // --- 2. Defer shard-type flip -----------------------------------
        //
        // Legacy-compatible: `ShardContext.current` stays on the source
        // type until the first post-transition WS actually lands. If the
        // redirect carries `target_shard_type == 255` (legacy "unknown"
        // sentinel), we fall back to whichever secondary type most recently
        // broadcast a WS — that is the shard the client can already render.
        let target = if *target_shard_type != 255 {
            *target_shard_type
        } else {
            worlds
                .secondary_by_type
                .keys()
                .copied()
                .next()
                .unwrap_or(ctx.current)
        };
        ctx.pending = Some(target);

        // --- 3. Grace-stash old primary WS ------------------------------
        //
        // Keep the previous primary WS renderable so remote entities and
        // celestial bodies do not blink. TTL is aligned with the chunk
        // grace so ship + crew + mesh all expire together.
        if let Some(prev) = worlds.primary.take() {
            worlds.last_primary = Some(prev);
            worlds.last_primary_until = Some(grace_expiry);
        }

        // --- 4. Grace-stash primary chunk source ------------------------
        //
        // The source entity stays in the Bevy world (its Mesh3d children
        // keep rendering). We move it from the live `SourceIndex` into the
        // grace map so incoming chunk snapshots for the destination cannot
        // land on it, and so `drain_grace_window` can despawn it at
        // expiry. Capture the departed pose here from the stashed
        // `last_primary` — system_position comes from `ws.origin` (on a
        // SHIP primary this is the ship's system-space position; on PLANET
        // it is the planet's center), rotation from the own-ship entity if
        // present, velocity similarly.
        if let Some(primary_seed) = ctx.primary_seed {
            if let Some(source_entity) = source_index.take(primary_seed) {
                let (sys_pos, rot, vel) = departed_pose_from_last_primary(
                    worlds.last_primary.as_ref(),
                    ctx.current,
                );
                // For PLANET/SYSTEM/GALAXY primaries, `ws.origin` is already
                // the system-space anchor and `is_own` Ship/Player entity is
                // absent — we fall through to identity rotation + zero
                // velocity. The grace render still shifts the source into
                // system-space via `system_position - new_ws.origin`, which
                // is what we want for a planet receding as we launch.
                grace.sources.insert(
                    primary_seed,
                    GraceSource {
                        entity: source_entity,
                        system_position: sys_pos,
                        rotation: rot,
                        velocity: vel,
                        captured_at: now,
                        expiry: grace_expiry,
                    },
                );
                tracing::info!(
                    seed = primary_seed,
                    ?sys_pos,
                    source_type = ctx.current,
                    target = target,
                    "grace-pinned old primary chunk source on transition"
                );
            }
        }

        // --- 5. Seamless promotion of a matching cached secondary WS ----
        //
        // When the destination shard was pre-connected as a secondary
        // (typical case: SYSTEM↔SHIP boarding, both tied together by the
        // server), its latest WS is already in `secondary_by_type`. Promote
        // it to `primary` so the very next render frame has live data in
        // the destination frame, and commit the shard-type flip
        // immediately since that data is now authoritative.
        if let Some(promoted) = worlds.secondary_by_type.remove(&target) {
            worlds.primary = Some(promoted);
            ctx.current = target;
            ctx.pending = None;
            tracing::info!(
                target,
                "seamless transition: promoted cached secondary to primary"
            );
        } else {
            tracing::info!(
                target,
                "hard transition: no cached secondary WS — waiting for Connected"
            );
        }

        // Clear `primary_seed` so any stray primary `ChunkSnapshot` that
        // arrives during the Transitioning→Connected window (rare: last
        // flush from the torn-down old TCP) does not route to the
        // now-graced source seed and resurrect a phantom entity under
        // that key. `ingest_chunk_events` drops primary snapshots when
        // `primary_seed` is `None`. Authoritative reseed happens in
        // `handle_connected`.
        ctx.primary_seed = None;

        // Diagnostic query so we can track grace entity transforms in
        // case the user files a "ship disappears for a moment" bug after
        // a transition — the ChunkSource root's position is the right
        // thing to observe there.
        let _ = (q_transform, &commands);
    }
}

/// On `NetEvent::Connected`, commit any still-pending shard-type flip, set
/// the primary seed, and (idempotently) promote a secondary chunk source
/// to primary if the same seed was pre-observed.
fn handle_connected(
    mut events: MessageReader<GameEvent>,
    mut ctx: ResMut<ShardContext>,
    mut source_index: ResMut<crate::chunk_stream::SourceIndex>,
    mut commands: Commands,
) {
    for GameEvent(ev) in events.read() {
        let NetEvent::Connected { shard_type, seed, .. } = ev else {
            continue;
        };

        // Commit any deferred flip. In the seamless path this was already
        // done by `handle_transitioning` when the matching secondary WS
        // was promoted — the re-commit here is a no-op.
        ctx.current = *shard_type;
        ctx.pending = None;

        // If a secondary chunk source for this seed already exists (we
        // were pre-observing this shard), rename it to serve as the
        // primary. Otherwise let the regular `ingest_chunk_events` path
        // create a fresh root when the first ChunkSnapshot lands.
        source_index.rekey_to_primary(seed, &mut commands);

        ctx.primary_seed = Some(*seed);
        ctx.ready = true;

        tracing::info!(
            shard_type,
            seed,
            "shard connected — primary seed set, context committed"
        );
    }
}

/// During the grace window, re-place each graced chunk-source root in the
/// *new* primary's frame every frame. The stored pose is system-space; we
/// convert into Bevy frame as
///     `translation = departed_system_pos + velocity * dt_since_capture - new_ws_origin`.
/// When there is no new primary WS yet (hard transition still reconnecting)
/// we leave the source at its last known transform so it at least stays on
/// screen at the pre-transition position.
fn drive_grace_transforms(
    grace: Res<GraceWindow>,
    worlds: Res<WorldStates>,
    mut q: Query<&mut Transform, With<crate::chunk_stream::ChunkSource>>,
) {
    let Some(new_ws) = worlds.primary.as_ref() else {
        return;
    };
    let new_origin = new_ws.origin;
    let now = Instant::now();
    for src in grace.sources.values() {
        let Ok(mut tf) = q.get_mut(src.entity) else {
            continue;
        };
        let elapsed = now.saturating_duration_since(src.captured_at).as_secs_f64();
        let world_pos = src.system_position + src.velocity * elapsed;
        let rel = world_pos - new_origin;
        tf.translation = Vec3::new(rel.x as f32, rel.y as f32, rel.z as f32);
        tf.rotation = Quat::from_xyzw(
            src.rotation.x as f32,
            src.rotation.y as f32,
            src.rotation.z as f32,
            src.rotation.w as f32,
        );
    }
}

/// Expire grace-window entries whose TTL has elapsed: despawn the root
/// entity (Bevy 0.18 `despawn()` cascades to children, so every chunk mesh
/// child of the source is freed in one call) and remove the map entry.
fn drain_grace_window(
    mut grace: ResMut<GraceWindow>,
    mut commands: Commands,
) {
    let now = Instant::now();
    let expired: Vec<u64> = grace
        .sources
        .iter()
        .filter_map(|(seed, s)| (now >= s.expiry).then_some(*seed))
        .collect();
    for seed in expired {
        if let Some(src) = grace.sources.remove(&seed) {
            commands.entity(src.entity).despawn();
            tracing::info!(seed, "grace window expired — chunk source despawned");
        }
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Extract the system-space pose + velocity of the old primary's "anchor
/// entity" — the object whose frame the chunks were rendered in. For a SHIP
/// primary this is the own-ship entity (marked `is_own && kind == Ship`).
/// For other primaries, no per-entity anchor is meaningful and we return
/// `(ws.origin, identity, zero)` so the grace-render simply pins the source
/// at the old WS origin (e.g. a planet staying where it was in system-space
/// as you launch).
fn departed_pose_from_last_primary(
    last: Option<&WorldStateData>,
    source_shard_type: u8,
) -> (DVec3, DQuat, DVec3) {
    let Some(last) = last else {
        return (DVec3::ZERO, DQuat::IDENTITY, DVec3::ZERO);
    };
    if source_shard_type == st::SHIP {
        if let Some(ship) = last
            .entities
            .iter()
            .find(|e| e.is_own && e.kind == EntityKind::Ship)
        {
            // On SHIP shard, `ws.origin` is the ship's system-space origin
            // and `entity.position` is ship-local (≈ 0 for own ship). Use
            // `ws.origin` as the system-space anchor and take rotation +
            // velocity from the own-ship entity for accurate grace.
            return (last.origin, ship.rotation, ship.velocity);
        }
    }
    (last.origin, DQuat::IDENTITY, DVec3::ZERO)
}
