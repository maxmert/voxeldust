use std::collections::HashMap;
use std::sync::Arc;

use bevy_app::prelude::*;
use bevy_ecs::prelude::*;
use clap::Parser;
use glam::{DQuat, DVec3, Vec3};
use rapier3d::prelude::*;
use tracing::{info, warn};

use voxeldust_core::client_message::{
    CelestialBodyData, EntityKind, JoinResponseData, LightingData, LodTier, ObservableEntityData,
    PlayerSnapshotData, ServerMsg, ShipRenderData, WorldStateData,
};
use voxeldust_core::ecs;
use voxeldust_core::handoff;
use voxeldust_core::shard_message::{
    AoiTarget, PlanetPlayerDigestData, PlanetPlayerDigestEntry, ShardMsg, ShipNearbyInfoData,
};
use voxeldust_core::shard_types::{SessionToken, ShardId, ShardType};
use voxeldust_core::system::{compute_lighting, compute_planet_position, SystemParams};
use voxeldust_shard_common::client_listener;
use voxeldust_shard_common::harness::{NetworkBridge, ShardHarness, ShardHarnessConfig};
use voxeldust_shard_common::signal_pipeline::SignalPipelinePlugin;
use voxeldust_shard_common::hud_delta::{
    append_channel_table_entries, flush_hud_deltas, HudSessionMap,
};
use voxeldust_shard_common::grant_persistence::{
    flush_grants_now, populate_registry_from_db, GrantsPersistenceQueue,
};

use voxeldust_core::character::{
    self, apply_update, build_character, move_one_character, step_body_head, BodyYaw,
    CharacterBuildSpec, CharacterCapsule, CharacterClassComp, CharacterCollisionEvent,
    CharacterController, CharacterMoveInput, CharacterVelocity, DesiredMovement, HeadPitch,
    HeadYaw, IsCharacter, LandedEvent, LocalUp, LocomotionState, MovementStats, PlatformDelta,
    PlatformSnapSuppressed, RapierWorld, TurnInPlace, HUMAN_DEFAULT,
};

// ---------------------------------------------------------------------------
// CLI
// ---------------------------------------------------------------------------

#[derive(Parser, Debug)]
#[command(name = "planet-shard", about = "Voxeldust planet shard — surface physics")]
struct Args {
    #[arg(long)]
    shard_id: u64,
    #[arg(long)]
    seed: u64,
    #[arg(long)]
    system_seed: Option<u64>,
    #[arg(long, default_value = "0")]
    planet_index: u32,
    #[arg(long, default_value = "http://127.0.0.1:8080")]
    orchestrator: String,
    #[arg(long, default_value = "127.0.0.1:9090")]
    orchestrator_heartbeat: String,
    #[arg(long, default_value = "7777")]
    tcp_port: u16,
    #[arg(long, default_value = "7778")]
    udp_port: u16,
    #[arg(long, default_value = "7779")]
    quic_port: u16,
    #[arg(long, default_value = "8081")]
    healthz_port: u16,
    #[arg(long)]
    advertise_host: Option<String>,
}

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

const G: f64 = 6.674e-11;

// Planet walking constants moved into the shared
// `voxeldust_core::character::MovementStats` defaults. Legacy
// `WALK_SPEED`/`JUMP_IMPULSE` values (4.0 / 5.0) were preserved as the
// new tunable defaults — speeds are unchanged; the physics is now
// kinematic-character-controller driven.

// ---------------------------------------------------------------------------
// Components (planet-shard-specific, on player entities)
// ---------------------------------------------------------------------------

/// Marker for a player entity on this planet.
#[derive(Component)]
struct PlanetPlayer;

// `PlayerBody` (RigidBodyHandle newtype) was removed in the KCC
// migration. The body handle now lives inside `CharacterController`
// from `voxeldust_core::character`.

/// Planet-local position (vector from planet center).
#[derive(Component)]
struct PlanetPosition(DVec3);

/// Orthonormal frame tangent to the sphere at the player's position.
#[derive(Component)]
struct TangentFrame {
    up: DVec3,
    north: DVec3,
    east: DVec3,
}

impl TangentFrame {
    fn from_up(up: DVec3) -> Self {
        let up = up.normalize();
        let pole = DVec3::Y;
        let east_raw = pole.cross(up);
        let east = if east_raw.length_squared() > 1e-10 {
            east_raw.normalize()
        } else {
            DVec3::Z.cross(up).normalize()
        };
        let north = up.cross(east).normalize();
        Self { up, north, east }
    }
}

/// Rapier flat-space position at last re-center.
#[derive(Component)]
struct RapierOrigin(DVec3);

// `BodyYaw`, `HeadYaw`, `HeadPitch` are imported from
// `voxeldust_core::character` — the shared body/head decoupling state.
// The planet-shard's old private `PlayerYaw(f32)` was renamed in Phase A
// (semantics preserved: it was always body yaw on the surface).

/// Per-player look input as received from the client this tick. Written
/// by `process_input`; consumed by `update_body_head_state`.
#[derive(Component, Default, Clone, Copy)]
struct CamLookInput {
    cam_yaw: f32,
    cam_pitch: f32,
}

/// Player session token.
#[derive(Component)]
struct SessionId(SessionToken);

/// Player name.
#[derive(Component)]
struct Name(String);

/// Input action state for edge detection.
#[derive(Component)]
struct ActionState {
    current: u8,
    previous: u8,
}

/// Marks a player as having a pending handoff.
#[derive(Component)]
struct HandoffPending;

// ---------------------------------------------------------------------------
// Components for nearby ships
// ---------------------------------------------------------------------------

/// Marker for a nearby ship entity.
#[derive(Component)]
struct NearbyShip;

/// Ship identifier.
#[derive(Component)]
struct NearbyShipId(u64);

/// Ship's shard for handoff routing.
#[derive(Component)]
struct NearbyShipShard(ShardId);

/// Ship position in planet-local coordinates.
#[derive(Component)]
struct ShipPosition(DVec3);

/// Ship rotation.
#[derive(Component)]
struct ShipRotation(DQuat);

/// Rapier rigid body handle for the ship's exterior collider (KinematicPositionBased).
/// Created when ShipColliderSync is received from the ship shard.
#[derive(Component)]
struct ShipColliderBody(rapier3d::dynamics::RigidBodyHandle);

/// Rapier collider handle for the ship's compound hull shape.
#[derive(Component)]
struct ShipColliderAttached(rapier3d::geometry::ColliderHandle);

// ---------------------------------------------------------------------------
// Resources
// ---------------------------------------------------------------------------

/// Phase 4-Persist.6: planet-shard's redb file. Mirrors ship-shard's
/// `ShipPersistence` pattern but currently houses only the grants
/// table; planet block persistence is a separate slice. The same
/// `voxeldust_shard_common::grant_persistence` helpers operate
/// against this `db` field.
#[derive(Resource)]
struct PlanetPersistence {
    db: redb::Database,
}

/// Rapier 3D physics context.
///
/// Rapier 0.32: `QueryPipeline` is no longer a stored resource — it's a
/// `Copy` view derived from `BroadPhaseBvh` + `NarrowPhase` + the body /
/// collider sets, built on demand via `BroadPhaseBvh::as_query_pipeline`.
#[derive(Resource)]
struct RapierContext {
    rigid_body_set: RigidBodySet,
    collider_set: ColliderSet,
    integration_params: IntegrationParameters,
    physics_pipeline: PhysicsPipeline,
    island_manager: IslandManager,
    broad_phase: BroadPhaseBvh,
    narrow_phase: NarrowPhase,
    impulse_joint_set: ImpulseJointSet,
    multibody_joint_set: MultibodyJointSet,
    ccd_solver: CCDSolver,
}

impl RapierWorld for RapierContext {
    fn bodies(&self) -> &RigidBodySet { &self.rigid_body_set }
    fn colliders(&self) -> &ColliderSet { &self.collider_set }
    fn query_pipeline(&self) -> rapier3d::pipeline::QueryPipeline<'_> {
        self.broad_phase.as_query_pipeline(
            self.narrow_phase.query_dispatcher(),
            &self.rigid_body_set,
            &self.collider_set,
            rapier3d::pipeline::QueryFilter::default(),
        )
    }
    fn bodies_mut(&mut self) -> &mut RigidBodySet { &mut self.rigid_body_set }
    // Default no-op `refresh_query_pipeline`: Rapier 0.32 keeps the
    // broad-phase BVH refreshed inside `physics_pipeline.step()` itself,
    // so there's no separate refresh phase to wire up.
}

/// Planet physics dt. Shared between KCC + Rapier step so they integrate
/// against the same clock.
#[derive(Resource, Clone, Copy)]
struct PlanetIntegrationDt(pub f32);

/// Planet configuration (immutable after init).
#[derive(Resource)]
#[allow(dead_code)]
struct PlanetConfig {
    shard_id: ShardId,
    planet_seed: u64,
    planet_radius: f64,
    planet_mass: f64,
    surface_gravity: f64,
    system_seed: Option<u64>,
    planet_index: u32,
}

/// System parameters for orbital computation.
#[derive(Resource)]
struct SystemParamsRes(Option<SystemParams>);

/// Planet position in system-space (updated each tick from Keplerian orbit).
#[derive(Resource)]
struct PlanetPositionInSystem(DVec3);

/// Cached system-space positions for all planets (avoids recomputing Kepler per broadcast).
#[derive(Resource, Default)]
struct CachedAllPlanetPositions(Vec<DVec3>);

/// Celestial time derived from epoch (matches system shard).
#[derive(Resource, Default)]
struct CelestialTimeRes(f64);

/// Physics simulation time.
#[derive(Resource, Default)]
struct PhysicsTimeRes(f64);

/// Universe epoch for deterministic celestial time.
#[derive(Resource)]
struct UniverseEpoch(Arc<std::sync::atomic::AtomicU64>);

/// Entity index: session_token → Entity for O(1) lookup.
#[derive(Resource, Default)]
struct PlayerEntityIndex(HashMap<SessionToken, Entity>);

/// Entity index: ship_id → Entity for nearby ships.
#[derive(Resource, Default)]
struct ShipEntityIndex(HashMap<u64, Entity>);

/// Handoff spawn info stored by process_handoffs, consumed by process_connects.
/// Avoids deferred-Commands visibility issues (Commands not materialized in same tick).
struct HandoffSpawnInfo {
    surface_pos: DVec3,
    /// `PlayerHandoff::schema_version` from the source shard. When >=
    /// the local `CHARACTER_SCHEMA_VERSION` we decode `character_state`
    /// to restore body/head pose; otherwise we spawn with defaults.
    schema_version: u16,
    /// Raw blob bytes; decoded in `process_connects` after the entity
    /// exists. Empty when the source shard didn't pack one.
    character_state: Vec<u8>,
}

/// Pending handoffs keyed by player name. process_handoffs inserts, process_connects consumes.
#[derive(Resource, Default)]
struct PendingHandoffs(HashMap<String, HandoffSpawnInfo>);

/// Latest AOI snapshot received from the system shard.
/// System shard sends entities in system-space coordinates with `observer_position`
/// set to this planet's system-space position. We transform to planet-local by
/// subtracting the planet's system-space position at broadcast time.
#[derive(Resource, Default)]
struct ExternalEntities {
    entities: Vec<ObservableEntityData>,
    observer_position: DVec3,
    tick: u64,
}

// ---------------------------------------------------------------------------
// Messages
// ---------------------------------------------------------------------------

/// A new TCP client connected.
#[derive(Message)]
struct ClientConnectedMsg {
    session_token: SessionToken,
    player_name: String,
    tcp_write: Arc<tokio::sync::Mutex<tokio::net::tcp::OwnedWriteHalf>>,
}

/// Player input from UDP.
#[derive(Message)]
struct PlayerInputMsg {
    session: SessionToken,
    input: voxeldust_core::client_message::PlayerInputData,
}

/// Player handoff from system shard.
#[derive(Message)]
struct InboundHandoffMsg {
    handoff: handoff::PlayerHandoff,
    relay_shard: ShardId,
}

/// Ship nearby info from system shard.
#[derive(Message)]
struct ShipNearbyMsg(ShipNearbyInfoData);

/// HandoffAccepted from ship shard (player re-entering ship).
#[derive(Message)]
struct HandoffAcceptedMsg {
    session: SessionToken,
    target_shard: ShardId,
}

/// Ship collider shapes from ship shard (for physical collision).
#[derive(Message)]
struct ShipColliderSyncMsg(voxeldust_core::shard_message::ShipColliderSyncData);


// ---------------------------------------------------------------------------
// System Sets
// ---------------------------------------------------------------------------

#[derive(SystemSet, Debug, Clone, PartialEq, Eq, Hash)]
enum PlanetSet {
    Bridge,
    Spawn,
    Input,
    Physics,
    Detection,
    Broadcast,
    Diagnostics,
}

// ---------------------------------------------------------------------------
// Bridge systems
// ---------------------------------------------------------------------------

fn drain_connects(
    mut bridge: ResMut<NetworkBridge>,
    mut events: MessageWriter<ClientConnectedMsg>,
) {
    for _ in 0..16 {
        let event = match bridge.connect_rx.try_recv() {
            Ok(e) => e,
            Err(_) => break,
        };
        let conn = event.connection;
        if let Ok(mut reg) = bridge.client_registry.try_write() {
            reg.register(&conn);
        }
        info!(player = %conn.player_name, session = conn.session_token.0, "player connected to planet");
        events.write(ClientConnectedMsg {
            session_token: conn.session_token,
            player_name: conn.player_name.clone(),
            tcp_write: conn.tcp_write.clone(),
        });
    }
}

fn drain_input(
    mut bridge: ResMut<NetworkBridge>,
    mut events: MessageWriter<PlayerInputMsg>,
    player_index: Res<PlayerEntityIndex>,
) {
    for _ in 0..64 {
        let (_src, input) = match bridge.input_rx.try_recv() {
            Ok(e) => e,
            Err(_) => break,
        };
        // Route to first player (same as original — multi-player session routing
        // needs UDP addr→session mapping, which will come with proper multi-player).
        if let Some((&session, _)) = player_index.0.iter().next() {
            events.write(PlayerInputMsg { session, input });
        }
    }
}

fn drain_quic(
    mut bridge: ResMut<NetworkBridge>,
    mut handoff_events: MessageWriter<InboundHandoffMsg>,
    mut ship_nearby_events: MessageWriter<ShipNearbyMsg>,
    mut accepted_events: MessageWriter<HandoffAcceptedMsg>,
    mut collider_sync_events: MessageWriter<ShipColliderSyncMsg>,
    mut external_entities: ResMut<ExternalEntities>,
    mut celestial_time: ResMut<CelestialTimeRes>,
    mut planet_pos: ResMut<PlanetPositionInSystem>,
    sys_params: Res<SystemParamsRes>,
    config: Res<PlanetConfig>,
) {
    for _ in 0..32 {
        let queued = match bridge.quic_msg_rx.try_recv() {
            Ok(q) => q,
            Err(_) => break,
        };
        match queued.msg {
            ShardMsg::PlayerHandoff(h) => {
                // Sync celestial time from system shard authority.
                if h.game_time > celestial_time.0 {
                    celestial_time.0 = h.game_time;
                    update_planet_position(&sys_params.0, config.planet_index, celestial_time.0, &mut planet_pos);
                }
                handoff_events.write(InboundHandoffMsg {
                    handoff: h,
                    relay_shard: queued.source_shard_id,
                });
            }
            ShardMsg::ShipNearbyInfo(info) => {
                if info.game_time > celestial_time.0 {
                    celestial_time.0 = info.game_time;
                    update_planet_position(&sys_params.0, config.planet_index, celestial_time.0, &mut planet_pos);
                }
                ship_nearby_events.write(ShipNearbyMsg(info));
            }
            ShardMsg::SystemEntitiesUpdate(data) => {
                // Accept only updates addressed to this planet.
                if let AoiTarget::Planet(idx) = data.target {
                    if idx == config.planet_index && data.tick >= external_entities.tick {
                        external_entities.entities = data.entities;
                        external_entities.observer_position = data.observer_position;
                        external_entities.tick = data.tick;
                    }
                }
            }
            ShardMsg::HandoffAccepted(accepted) => {
                accepted_events.write(HandoffAcceptedMsg {
                    session: accepted.session_token,
                    target_shard: accepted.target_shard,
                });
            }
            ShardMsg::ShipColliderSync(data) => {
                collider_sync_events.write(ShipColliderSyncMsg(data));
            }
            _ => {}
        }
    }
}

fn update_planet_position(
    sys_params: &Option<SystemParams>,
    planet_index: u32,
    celestial_time: f64,
    planet_pos: &mut PlanetPositionInSystem,
) {
    if let Some(sys) = sys_params {
        if let Some(planet) = sys.planets.get(planet_index as usize) {
            planet_pos.0 = compute_planet_position(planet, celestial_time);
        }
    }
}

/// Refresh all-planet position cache once per tick (after physics_step updates celestial_time).
/// Separated from update_planet_position to avoid redundant Kepler solves in drain_quic.
fn refresh_planet_position_cache(
    sys_params: Res<SystemParamsRes>,
    celestial_time: Res<CelestialTimeRes>,
    mut cached: ResMut<CachedAllPlanetPositions>,
) {
    if let Some(ref sys) = sys_params.0 {
        cached.0.clear();
        cached.0.reserve(sys.planets.len());
        for planet in &sys.planets {
            cached.0.push(compute_planet_position(planet, celestial_time.0));
        }
    }
}

// ---------------------------------------------------------------------------
// Spawn / process events
// ---------------------------------------------------------------------------

fn process_connects(
    mut commands: Commands,
    mut events: MessageReader<ClientConnectedMsg>,
    mut rapier: ResMut<RapierContext>,
    config: Res<PlanetConfig>,
    planet_pos: Res<PlanetPositionInSystem>,
    celestial_time: Res<CelestialTimeRes>,
    mut player_index: ResMut<PlayerEntityIndex>,
    mut pending: ResMut<PendingHandoffs>,
    bridge: Res<NetworkBridge>,
) {
    for event in events.read() {
        let token = event.session_token;
        let tcp_write = event.tcp_write.clone();
        let sys_seed = config.system_seed.unwrap_or(0);
        let game_time = celestial_time.0;
        let planet_pos_val = planet_pos.0;

        // Check if a handoff was received for this player (by name).
        let spawn_pos = if let Some(handoff_info) = pending.0.remove(&event.player_name) {
            info!(player = %event.player_name, "spawning from handoff data");
            let entity = spawn_player(
                &mut commands,
                &mut rapier,
                config.planet_radius,
                token,
                event.player_name.clone(),
                handoff_info.surface_pos,
            );
            player_index.0.insert(token, entity);
            // Restore body/head/turn pose from the handoff blob if the
            // source shard packed one (schema_version >= 2). Legacy
            // handoffs (version 0/1) leave the spawn defaults.
            if handoff_info.schema_version
                >= voxeldust_core::character::CHARACTER_SCHEMA_VERSION
            {
                let blob = voxeldust_core::character::decode_character_state(
                    &handoff_info.character_state,
                );
                commands.entity(entity).insert((
                    BodyYaw(blob.body_yaw),
                    HeadYaw(blob.head_yaw),
                    HeadPitch(blob.head_pitch),
                ));
                if let Some(turn) = blob.turn {
                    commands.entity(entity).insert(turn);
                }
            }
            handoff_info.surface_pos
        } else {
            // New player — spawn at default position.
            warn!(player = %event.player_name, "no handoff data — spawning at default");
            let default_pos = DVec3::new(0.0, config.planet_radius + 2.0, 0.0);
            let entity = spawn_player(
                &mut commands,
                &mut rapier,
                config.planet_radius,
                token,
                event.player_name.clone(),
                default_pos,
            );
            player_index.0.insert(token, entity);
            default_pos
        };

        info!(player = %event.player_name, session = token.0, "player spawned on planet");

        // Look up the host system shard's endpoints (for always-on scene secondary).
        let system_preconnect = if let Ok(reg) = bridge.peer_registry.try_read() {
            reg.find_by_type(ShardType::System).first().map(|info| {
                handoff::ShardPreConnect {
                    shard_type: 1, // System
                    tcp_addr: info.endpoint.tcp_addr.to_string(),
                    udp_addr: info.endpoint.udp_addr.to_string(),
                    seed: sys_seed,
                    planet_index: 0,
                    reference_position: DVec3::ZERO,
                    reference_rotation: DQuat::IDENTITY,
                    shard_id: info.id.0,
                }
            })
        } else {
            None
        };

        tokio::spawn(async move {
            let jr = ServerMsg::JoinResponse(JoinResponseData {
                seed: 0,
                planet_radius: 0,
                player_id: token.0,
                spawn_position: spawn_pos,
                spawn_rotation: DQuat::IDENTITY,
                spawn_forward: DVec3::NEG_Z,
                session_token: token,
                shard_type: 0, // Planet
                galaxy_seed: 0,
                system_seed: sys_seed,
                game_time,
                reference_position: planet_pos_val,
                reference_rotation: DQuat::IDENTITY,
            });
            let mut writer = tcp_write.lock().await;
            let _ = client_listener::send_tcp_msg(&mut *writer, &jr).await;
            // Follow immediately with a ShardPreConnect for the host system shard.
            // System and Galaxy secondaries are always-on and not counted against
            // the client's secondary cap.
            if let Some(pc) = system_preconnect {
                let _ = client_listener::send_tcp_msg(
                    &mut *writer,
                    &ServerMsg::ShardPreConnect(pc),
                )
                .await;
            }
        });
    }
}

fn spawn_player(
    commands: &mut Commands,
    rapier: &mut RapierContext,
    planet_radius: f64,
    session_token: SessionToken,
    name: String,
    planet_local_pos: DVec3,
) -> Entity {
    let radial = planet_local_pos.normalize();
    let height = (planet_local_pos.length() - planet_radius).max(0.5);
    let frame = TangentFrame::from_up(radial);

    // Build the KCC-backed kinematic body at the re-centered origin.
    // The planet keeps the same "character lives at (0, height, 0) in
    // flat Rapier space + re-center each tick" invariant — no change
    // to that pattern, only the body type.
    let stats = MovementStats::default();
    let capsule = CharacterCapsule::default();
    let ctrl = build_character(
        &mut rapier.rigid_body_set,
        &mut rapier.collider_set,
        CharacterBuildSpec {
            // Rapier 0.32: `vector!` macro produced a `nalgebra::Vector3`
            // — replace with a plain glam `Vec3` since the build spec's
            // `position` field is now `Vector` (= `Vec3`).
            position: rapier3d::math::Vector::new(0.0, height as f32, 0.0),
            capsule,
            stats,
            // Rapier 0.32: `up_axis` is a plain `Vector` (= glam Vec3),
            // no longer wrapped in `UnitVector3`.
            up_axis: rapier3d::math::Vector::Y,
        },
    );
    // Apply planet-specific collision groups so players don't get
    // filtered out by the existing planet-shard raycast filters.
    if let Some(col) = rapier.collider_set.get_mut(ctrl.collider) {
        // Rapier 0.32: `InteractionGroups::new` gained an
        // `InteractionTestMode` parameter (And vs Or). Old AND-style
        // matching is the default — use it explicitly to preserve
        // the previous behaviour.
        col.set_collision_groups(InteractionGroups::new(
            Group::GROUP_1,
            Group::GROUP_1 | Group::GROUP_2,
            InteractionTestMode::default(),
        ));
    }

    let identity_bundle = (
        PlanetPlayer,
        PlanetPosition(planet_local_pos),
        frame,
        RapierOrigin(DVec3::new(0.0, height, 0.0)),
        BodyYaw(0.0),
        HeadYaw(0.0),
        HeadPitch(0.0),
        voxeldust_core::character::LookTarget::cleared(),
        CamLookInput::default(),
        CharacterClassComp(HUMAN_DEFAULT),
        SessionId(session_token),
        Name(name),
        ActionState {
            current: 0,
            previous: 0,
        },
    );
    let character_bundle = (
        IsCharacter,
        ctrl,
        capsule,
        stats,
        CharacterVelocity::zero(),
        LocomotionState::Airborne,
        LocalUp(radial),
        DesiredMovement::default(),
        PlatformDelta::default(),
        PlatformSnapSuppressed::default(),
    );

    commands.spawn((identity_bundle, character_bundle)).id()
}

fn process_handoffs(
    mut events: MessageReader<InboundHandoffMsg>,
    config: Res<PlanetConfig>,
    sys_params: Res<SystemParamsRes>,
    planet_pos: Res<PlanetPositionInSystem>,
    mut pending: ResMut<PendingHandoffs>,
    bridge: Res<NetworkBridge>,
) {
    for event in events.read() {
        let h = &event.handoff;

        // Compute planet position at handoff time for accurate system→planet-local conversion.
        let planet_pos_at_handoff = sys_params
            .0
            .as_ref()
            .and_then(|sys| sys.planets.get(config.planet_index as usize))
            .map(|p| compute_planet_position(p, h.game_time))
            .unwrap_or(planet_pos.0);
        let surface_pos = h.position - planet_pos_at_handoff;

        info!(
            player = %h.player_name,
            session = h.session_token.0,
            surface = format!("({:.1},{:.1},{:.1})", surface_pos.x, surface_pos.y, surface_pos.z),
            height = format!("{:.1}", surface_pos.length() - config.planet_radius),
            "handoff received"
        );

        // Store handoff data for process_connects to consume when the
        // player's TCP connection arrives. We don't spawn here because
        // deferred Commands aren't visible to process_connects in the
        // same tick (bevy_ecs ApplyDeferred timing).
        pending.0.insert(h.player_name.clone(), HandoffSpawnInfo {
            surface_pos,
            schema_version: h.schema_version,
            character_state: h.character_state.clone(),
        });

        // Send HandoffAccepted back to the relay system shard.
        // TODO(phase-A-surface): populate spawn_pose with the planet-local
        // authoritative landing position. For now left None — client uses
        // the JoinResponse position when arriving at a planet shard.
        let accepted = ShardMsg::HandoffAccepted(handoff::HandoffAccepted {
            session_token: h.session_token,
            target_shard: config.shard_id,
            spawn_pose: None,
            // Phase T0: planet-shard does not yet support observer
            // promotion (no PendingOwnership equivalent for surface
            // players). Source shard will use ShardRedirect (full TCP
            // handshake), which is fine for the planet path until a
            // future phase plumbs observer-promote here too.
            observer_promoted: false,
        });
        let relay_shard = event.relay_shard;
        if let Ok(reg) = bridge.peer_registry.try_read() {
            if let Some(addr) = reg.quic_addr(relay_shard) {
                let _ = bridge
                    .quic_send_tx
                    .try_send((relay_shard, addr, accepted));
                info!(
                    target = relay_shard.0,
                    "sent HandoffAccepted to relay system shard"
                );
            } else {
                warn!(
                    target = relay_shard.0,
                    "relay system shard not in peer registry"
                );
            }
        }
    }
}

fn process_ship_nearby(
    mut commands: Commands,
    mut events: MessageReader<ShipNearbyMsg>,
    sys_params: Res<SystemParamsRes>,
    config: Res<PlanetConfig>,
    mut ship_index: ResMut<ShipEntityIndex>,
    mut existing_ships: Query<(&NearbyShipId, &mut ShipPosition, &mut ShipRotation)>,
    bridge: Res<NetworkBridge>,
) {
    for event in events.read() {
        let info = &event.0;

        // Convert to planet-local using planet position at the ship's celestial time.
        let planet_pos_at_ship_time = sys_params
            .0
            .as_ref()
            .and_then(|sys| sys.planets.get(config.planet_index as usize))
            .map(|p| compute_planet_position(p, info.game_time))
            .unwrap_or(DVec3::ZERO);
        let planet_local_pos = info.position - planet_pos_at_ship_time;

        if let Some(&entity) = ship_index.0.get(&info.ship_id) {
            // Update existing ship entity.
            if let Ok((_, mut pos, mut rot)) = existing_ships.get_mut(entity) {
                pos.0 = planet_local_pos;
                rot.0 = info.rotation;
            }
        } else {
            // Spawn new nearby ship entity.
            info!(
                ship_id = info.ship_id,
                pos = format!(
                    "({:.0},{:.0},{:.0})",
                    info.position.x, info.position.y, info.position.z
                ),
                "new ship near planet"
            );
            let entity = commands
                .spawn((
                    NearbyShip,
                    NearbyShipId(info.ship_id),
                    NearbyShipShard(info.ship_shard_id),
                    ShipPosition(planet_local_pos),
                    ShipRotation(info.rotation),
                ))
                .id();
            ship_index.0.insert(info.ship_id, entity);

            // Send ShardPreConnect to all connected planet clients so they
            // open observer connections to this ship shard and receive chunk data.
            let ship_shard_id = info.ship_shard_id;
            let ship_id = info.ship_id;
            if let Ok(reg) = bridge.peer_registry.try_read() {
                if let Some(peer) = reg.get(ship_shard_id) {
                    let pc = ServerMsg::ShardPreConnect(handoff::ShardPreConnect {
                        shard_type: 2, // Ship
                        tcp_addr: peer.endpoint.tcp_addr.to_string(),
                        udp_addr: peer.endpoint.udp_addr.to_string(),
                        seed: ship_id,
                        planet_index: 0,
                        reference_position: DVec3::ZERO,
                        reference_rotation: DQuat::IDENTITY,
                        shard_id: ship_shard_id.0,
                    });
                    let cr = bridge.client_registry.clone();
                    tokio::spawn(async move {
                        if let Ok(reg) = cr.try_read() {
                            for addr in reg.udp_addrs() {
                                if let Some(session) = reg.session_for_udp(addr) {
                                    let _ = reg.send_tcp(session, &pc).await;
                                }
                            }
                        }
                    });
                    tracing::info!(ship_id, shard = ship_shard_id.0, "sent ShardPreConnect for ship to planet clients");
                }
            }
        }
    }
}

/// Build or update Rapier compound colliders for nearby ships from ShipColliderSync.
/// Uses a KinematicPositionBased rigid body so we can update position each tick.
fn process_ship_colliders(
    mut events: MessageReader<ShipColliderSyncMsg>,
    ship_index: Res<ShipEntityIndex>,
    mut rapier: ResMut<RapierContext>,
    mut commands: Commands,
    ships: Query<(
        &ShipPosition,
        &ShipRotation,
        Option<&ShipColliderBody>,
        Option<&ShipColliderAttached>,
    ), With<NearbyShip>>,
) {
    use rapier3d::prelude::*;

    for event in events.read() {
        let data = &event.0;
        let Some(&entity) = ship_index.0.get(&data.ship_id) else {
            warn!(ship_id = data.ship_id, "ShipColliderSync for unknown ship — ignoring");
            continue;
        };

        let Ok((pos, rot, existing_body, existing_collider)) = ships.get(entity) else {
            continue;
        };

        // Remove existing body (which also removes its attached colliders).
        if let Some(body_comp) = existing_body {
            let ctx = &mut *rapier;
            ctx.rigid_body_set.remove(
                body_comp.0,
                &mut ctx.island_manager,
                &mut ctx.collider_set,
                &mut ctx.impulse_joint_set,
                &mut ctx.multibody_joint_set,
                true,
            );
        }

        // Build compound shape from all chunks' collider shapes.
        // Rapier 0.32: `Pose` (= glamx `Pose3`) replaces `Isometry<f32>`,
        // and `Pose::translation(x, y, z)` is the translation-only
        // constructor (not the `.translation` field).
        let mut shapes: Vec<(Pose, SharedShape)> = Vec::new();
        for chunk in &data.chunks {
            for &(center, half_extents) in &chunk.shapes {
                let iso = Pose::translation(center.x, center.y, center.z);
                let shape = SharedShape::cuboid(half_extents.x, half_extents.y, half_extents.z);
                shapes.push((iso, shape));
            }
        }

        if shapes.is_empty() {
            // No solid blocks — remove collider components.
            commands.entity(entity).remove::<ShipColliderBody>();
            commands.entity(entity).remove::<ShipColliderAttached>();
            continue;
        }

        // Convert ship position to Rapier coordinates.
        // Planet shard positions are planet-local (DVec3); Rapier uses f32.
        // For ships near the surface, planet-local coords are small enough for f32.
        let ship_pos_f32 = glam::Vec3::new(
            pos.0.x as f32, pos.0.y as f32, pos.0.z as f32,
        );
        let ship_rot_f32 = glam::Quat::from_xyzw(
            rot.0.x as f32, rot.0.y as f32, rot.0.z as f32, rot.0.w as f32,
        );

        // Create KinematicPositionBased body (position updated each tick).
        // Rapier 0.32: `RigidBodyBuilder::translation` takes `Vector` (=
        // glamx-bundled glam 0.30 `Vec3`); workspace `glam` is 0.29.
        // Same memory layout, distinct types — convert at the boundary.
        let rb = RigidBodyBuilder::kinematic_position_based()
            .translation(rapier3d::math::Vector::new(
                ship_pos_f32.x,
                ship_pos_f32.y,
                ship_pos_f32.z,
            ))
            .build();
        let ctx = &mut *rapier;
        let body_handle = ctx.rigid_body_set.insert(rb);

        // Attach compound collider.
        let compound = ColliderBuilder::compound(shapes).build();
        let collider_handle = ctx.collider_set.insert_with_parent(
            compound,
            body_handle,
            &mut ctx.rigid_body_set,
        );

        commands.entity(entity).insert(ShipColliderBody(body_handle));
        commands.entity(entity).insert(ShipColliderAttached(collider_handle));

        info!(
            ship_id = data.ship_id,
            chunks = data.chunks.len(),
            "built ship exterior compound collider"
        );
    }
}

/// Update Rapier body positions for ship exterior colliders when ShipNearbyInfo updates.
fn update_ship_collider_positions(
    ships: Query<(&ShipPosition, &ShipRotation, &ShipColliderBody), With<NearbyShip>>,
    mut rapier: ResMut<RapierContext>,
) {
    for (pos, rot, body_comp) in &ships {
        if let Some(body) = rapier.rigid_body_set.get_mut(body_comp.0) {
            // Rapier 0.32: `Pose::from_parts(translation: Vec3,
            // rotation: Quat)` — no separate `Translation3`/`UnitQuaternion`
            // wrappers. Workspace `glam` is 0.29 / rapier's `glamx`
            // re-exports glam 0.30; convert at the boundary.
            let p = rapier3d::math::Vector::new(
                pos.0.x as f32, pos.0.y as f32, pos.0.z as f32,
            );
            let r = rapier3d::math::Rotation::from_xyzw(
                rot.0.x as f32, rot.0.y as f32, rot.0.z as f32, rot.0.w as f32,
            );
            body.set_next_kinematic_position(rapier3d::math::Pose::from_parts(p, r));
        }
    }
}

fn process_handoff_accepted(
    mut commands: Commands,
    mut events: MessageReader<HandoffAcceptedMsg>,
    mut rapier: ResMut<RapierContext>,
    bridge: Res<NetworkBridge>,
    player_index: Res<PlayerEntityIndex>,
    players: Query<&CharacterController, With<PlanetPlayer>>,
) {
    for event in events.read() {
        let session = event.session;
        let target_shard = event.target_shard;
        info!(
            session = session.0,
            target = target_shard.0,
            "received HandoffAccepted for ship re-entry"
        );

        // Send ShardRedirect to client.
        if let Ok(reg) = bridge.peer_registry.try_read() {
            if let Some(peer_info) = reg.get(target_shard) {
                // TODO(phase-A-reentry): populate spawn_pose with the
                // ship-local spawn position for planet→ship re-entry.
                // Left None for now — client uses the ship shard's
                // JoinResponse position.
                let redirect = ServerMsg::ShardRedirect(handoff::ShardRedirect {
                    session_token: session,
                    target_tcp_addr: peer_info.endpoint.tcp_addr.to_string(),
                    target_udp_addr: peer_info.endpoint.udp_addr.to_string(),
                    shard_id: target_shard,
                    target_shard_type: peer_info.shard_type as u8,
                    spawn_pose: None,
                });
                let cr = bridge.client_registry.clone();
                tokio::spawn(async move {
                    if let Ok(reg) = cr.try_read() {
                        if let Err(e) = reg.send_tcp(session, &redirect).await {
                            tracing::warn!(%e, "failed to send ShardRedirect for re-entry");
                        }
                    }
                    if let Ok(mut reg) = cr.try_write() {
                        reg.unregister(&session);
                    }
                });
            }
        }

        // Remove player entity and KCC body.
        if let Some(&entity) = player_index.0.get(&session) {
            if let Ok(ctrl) = players.get(entity) {
                let ctx = &mut *rapier;
                ctx.rigid_body_set.remove(
                    ctrl.body,
                    &mut ctx.island_manager,
                    &mut ctx.collider_set,
                    &mut ctx.impulse_joint_set,
                    &mut ctx.multibody_joint_set,
                    true,
                );
            }
            commands.entity(entity).despawn();
            info!(session = session.0, "player removed from planet (re-entry)");
        }
    }
}

// ---------------------------------------------------------------------------
// Tablet interaction (Phase J)
// ---------------------------------------------------------------------------

/// Phase J — server-authoritative tablet open/close. Mirrors the ship-shard
/// implementation; the only difference is the planet-local position type
/// (`PlanetPosition`) used for the distance check. The block coordinate is
/// already in planet-local frame on the wire because the client transforms
/// before sending.
fn process_tablet_interact(
    mut commands: Commands,
    mut bridge: ResMut<NetworkBridge>,
    player_index: Res<PlayerEntityIndex>,
    players: Query<&PlanetPosition, With<PlanetPlayer>>,
) {
    /// 4 metres — matches the ship-shard radius; kept as a per-shard const
    /// rather than a `CharacterClass` field because it's a gameplay-tuning
    /// distance, not a per-character property.
    const TABLET_INTERACT_RADIUS_M: f32 = 4.0;
    for _ in 0..64 {
        let Ok((session, data)) = bridge.tablet_interact_rx.try_recv() else {
            break;
        };
        let Some(&entity) = player_index.0.get(&session) else {
            continue;
        };
        match data {
            voxeldust_core::client_message::TabletInteractData::Open { block_pos } => {
                let Ok(player_pos) = players.get(entity) else {
                    continue;
                };
                let block_centre = DVec3::new(
                    block_pos.x as f64 + 0.5,
                    block_pos.y as f64 + 0.5,
                    block_pos.z as f64 + 0.5,
                );
                if player_pos.0.distance(block_centre) > TABLET_INTERACT_RADIUS_M as f64 {
                    tracing::info!(
                        ?session,
                        ?block_pos,
                        player = ?player_pos.0,
                        "TabletInteract: rejected — too far from target block"
                    );
                    continue;
                }
                commands.entity(entity).insert((
                    voxeldust_core::character::IsHoldingTablet,
                    voxeldust_core::character::TabletCursor(glam::Vec2::new(0.5, 0.5)),
                ));
            }
            voxeldust_core::client_message::TabletInteractData::Close => {
                commands
                    .entity(entity)
                    .remove::<voxeldust_core::character::IsHoldingTablet>()
                    .remove::<voxeldust_core::character::TabletCursor>();
            }
        }
    }
}

/// Phase J — high-frequency cursor stream. Pure passthrough; client-side
/// `ClientMsg::deserialize` already clamps `u/v` to [0, 1].
fn process_tablet_cursor_updates(
    mut bridge: ResMut<NetworkBridge>,
    player_index: Res<PlayerEntityIndex>,
    mut cursors: Query<&mut voxeldust_core::character::TabletCursor>,
) {
    for _ in 0..256 {
        let Ok((session, uv)) = bridge.tablet_cursor_update_rx.try_recv() else {
            break;
        };
        let Some(&entity) = player_index.0.get(&session) else {
            continue;
        };
        if let Ok(mut cur) = cursors.get_mut(entity) {
            cur.0 = uv;
        }
    }
}

// ---------------------------------------------------------------------------
// Input
// ---------------------------------------------------------------------------

/// Translate client input into `DesiredMovement`; KCC system in Physics
/// set consumes it. Same pattern as ship-shard.
fn process_input(
    mut events: MessageReader<PlayerInputMsg>,
    player_index: Res<PlayerEntityIndex>,
    mut players: Query<
        (&mut CamLookInput, &mut ActionState, &mut DesiredMovement),
        With<PlanetPlayer>,
    >,
) {
    use voxeldust_core::client_message::input_action_bits as bits;
    for event in events.read() {
        let entity = match player_index.0.get(&event.session) {
            Some(&e) => e,
            None => continue,
        };
        let Ok((mut look, mut actions, mut desired)) = players.get_mut(entity) else {
            continue;
        };

        actions.previous = actions.current;
        actions.current = event.input.action;
        // Stash latest camera angles for the body/head state machine
        // (`update_body_head_state`, in the Physics set).
        look.cam_yaw = event.input.look_yaw;
        look.cam_pitch = event.input.look_pitch.clamp(
            -std::f32::consts::FRAC_PI_2,
            std::f32::consts::FRAC_PI_2,
        );

        // Planet-local horizontal input: [strafe, forward] — same
        // semantics as ship-shard.
        desired.horizontal = glam::Vec2::new(event.input.movement[0], event.input.movement[2]);
        // Jump is edge-triggered; OR so packets arriving same-tick don't lose it.
        desired.jump |= event.input.jump;
        desired.sprint = (event.input.actions_bits & bits::SPRINT) != 0;
        desired.crouch = (event.input.actions_bits & bits::CROUCH) != 0;
        let new_stance = if (event.input.actions_bits & bits::STANCE_CYCLE_UP) != 0 {
            Some(character::StanceAction::CycleUp)
        } else if (event.input.actions_bits & bits::STANCE_CYCLE_DOWN) != 0 {
            Some(character::StanceAction::CycleDown)
        } else {
            None
        };
        if new_stance.is_some() {
            desired.stance_action = new_stance;
        }
    }
}

/// Body / head decoupling — same shared state machine as ship-shard.
/// Runs in `PlanetSet::Physics` before `kcc_move_characters_system` so
/// the KCC reads the new `BodyYaw` for character heading. On the planet
/// the character lives in flat Rapier space (re-centered each tick) so
/// the velocity direction is read from the local XZ plane just like
/// ship interior — `tangent_frame_sync` handles the sphere remapping.
fn update_body_head_state(
    mut commands: Commands,
    integration: Res<PlanetIntegrationDt>,
    tick: Res<ecs::TickCounter>,
    mut players: Query<
        (
            Entity,
            &CharacterClassComp,
            &CamLookInput,
            &CharacterVelocity,
            &mut BodyYaw,
            &mut HeadYaw,
            &mut HeadPitch,
            Option<&TurnInPlace>,
        ),
        With<PlanetPlayer>,
    >,
) {
    let dt = integration.0;
    let one_hz = tick.0.is_multiple_of(20);
    for (entity, class, look, vel, mut by, mut hy, mut hp, prev_turn) in &mut players {
        // Body chases CAMERA direction, not velocity — see ship-shard
        // for the rationale.
        let speed = vel.horizontal().length();
        let update = step_body_head(
            &class.0,
            by.0,
            prev_turn.copied(),
            look.cam_yaw,
            look.cam_pitch,
            speed,
            dt,
        );
        apply_update(&mut by, &mut hy, &mut hp, &update);
        match (prev_turn.is_some(), update.turn) {
            (false, Some(t)) => {
                info!(
                    body_yaw = format!("{:.2}", by.0),
                    target = format!("{:.2}", t.target_body_yaw),
                    cam_yaw = format!("{:.2}", look.cam_yaw),
                    "character: turn-in-place START"
                );
                commands.entity(entity).insert(t);
            }
            (true, Some(t)) => {
                commands.entity(entity).insert(t);
            }
            (true, None) => {
                info!(
                    body_yaw = format!("{:.2}", by.0),
                    "character: turn-in-place END"
                );
                commands.entity(entity).remove::<TurnInPlace>();
            }
            (false, None) => {}
        }

        if one_hz {
            info!(
                body_yaw = format!("{:.2}", by.0),
                head_yaw = format!("{:.2}", hy.0),
                head_pitch = format!("{:.2}", hp.0),
                speed = format!("{:.2}", update.locomotion_speed),
                turning = update.turn.is_some(),
                "character diag (1Hz)"
            );
        }
    }
}

/// Phase I — read each ragdoll body's pose from rapier and pack into
/// the per-tick wire snapshot. See ship-shard's matching helper for
/// the full rationale; the two are kept in lockstep to avoid
/// observable shard-vs-shard divergence.
fn collect_ragdoll_bones(
    handles: Option<&voxeldust_core::character::RagdollHandles>,
    rapier: &RapierContext,
) -> Vec<voxeldust_core::character::RagdollBoneTransform> {
    let Some(handles) = handles else {
        return Vec::new();
    };
    handles
        .bodies
        .iter()
        .filter_map(|(name, h)| {
            let body = rapier.rigid_body_set.get(*h)?;
            let pose = body.position();
            Some(voxeldust_core::character::RagdollBoneTransform {
                bone_name: name.to_string(),
                translation: glam::Vec3::new(
                    pose.translation.x,
                    pose.translation.y,
                    pose.translation.z,
                ),
                rotation: glam::Quat::from_xyzw(
                    pose.rotation.x,
                    pose.rotation.y,
                    pose.rotation.z,
                    pose.rotation.w,
                ),
            })
        })
        .collect()
}

/// Phase I — tick down each `RagdollLifetime`. When the timer expires,
/// the rapier bodies are removed and the entity despawned. No-op
/// every tick until a death event ever inserts a ragdoll.
fn update_ragdoll_lifetime(
    mut commands: Commands,
    mut rapier: ResMut<RapierContext>,
    integration: Res<PlanetIntegrationDt>,
    mut ragdolls: Query<(
        Entity,
        &voxeldust_core::character::RagdollHandles,
        &mut voxeldust_core::character::RagdollLifetime,
    )>,
) {
    let dt = integration.0;
    for (entity, handles, mut lifetime) in &mut ragdolls {
        if !lifetime.tick(dt) {
            continue;
        }
        let ctx = &mut *rapier;
        voxeldust_core::character::despawn_ragdoll_bodies(
            &mut ctx.rigid_body_set,
            &mut ctx.collider_set,
            &mut ctx.impulse_joint_set,
            &mut ctx.multibody_joint_set,
            &mut ctx.island_manager,
            handles,
        );
        commands.entity(entity).despawn();
    }
}

/// Phase H — populate each player's `LookTarget` with the world-space
/// position of the *nearest other player* that lies inside their
/// peripheral attention cone. Falls back to `None` when the player
/// has no neighbour worth looking at, so the character's head simply
/// stays in its animation pose (no IK overhead, no fake "always
/// staring forward" behaviour).
///
/// All math is in the planet shard's flat Y-up frame — same frame
/// `PlanetPosition` lives in. The wire layer encodes the broadcast
/// target as a delta from `position` so f32 precision is fine.
fn update_look_targets(
    mut players: Query<
        (
            Entity,
            &PlanetPosition,
            &BodyYaw,
            &CharacterClassComp,
            &mut voxeldust_core::character::LookTarget,
        ),
        With<PlanetPlayer>,
    >,
) {
    let positions: Vec<(Entity, DVec3)> = players
        .iter()
        .map(|(e, p, _, _, _)| (e, p.0))
        .collect();
    for (entity, my_pos, body_yaw, class, mut look) in players.iter_mut() {
        let cls = &class.0;
        let max_dist_sq = (cls.look_attention_distance as f64).powi(2);
        let cos_fov = (cls.look_attention_fov_half as f64).cos();
        let eye_offset = cls.look_eye_world_offset as f64;
        // Body-forward in the planet shard's flat frame. Server's
        // `body_yaw` convention is `0 = +X`, `+ve = clockwise from
        // above` — see `client::character::render` for the matching
        // visual rotation derivation.
        let by = body_yaw.0 as f64;
        let forward = DVec3::new(by.cos(), 0.0, by.sin());

        let mut best: Option<(f64, DVec3)> = None;
        for (other_e, other_pos) in &positions {
            if *other_e == entity {
                continue;
            }
            let delta = *other_pos - my_pos.0;
            let dist_sq = delta.length_squared();
            if dist_sq < 0.25 || dist_sq > max_dist_sq {
                continue;
            }
            let dist = dist_sq.sqrt();
            let dir = delta / dist;
            if dir.dot(forward) < cos_fov {
                continue;
            }
            // Aim at the OTHER player's eye level, not their hip
            // (their `position` is the capsule centre).
            let target = *other_pos + DVec3::new(0.0, eye_offset, 0.0);
            if best.map(|(d, _)| dist < d).unwrap_or(true) {
                best = Some((dist, target));
            }
        }
        let new_target = best.map(|(_, t)| t);
        if look.0 != new_target {
            look.0 = new_target;
        }
    }
}

/// Drive the kinematic character controllers for planet-surface walkers.
///
/// The planet-shard uses the `TangentFrame` + `RapierOrigin` re-center
/// trick: the character body lives near (0, height, 0) in flat Rapier
/// space and its horizontal displacement is remapped to sphere-surface
/// rotation in `tangent_frame_sync`. The KCC operates IN that flat
/// frame — no special-case math here — because:
/// 1. Gravity is `-surface_g * flat_Y` (the tangent frame's local Y).
/// 2. Displacement is tangential; curvature is applied post-KCC.
fn kcc_move_characters_system(
    mut rapier: ResMut<RapierContext>,
    config: Res<PlanetConfig>,
    integration: Res<PlanetIntegrationDt>,
    mut characters: Query<
        (
            Entity,
            &CharacterController,
            &mut DesiredMovement,
            &PlatformDelta,
            &MovementStats,
            &mut CharacterVelocity,
            &mut LocomotionState,
            // KCC reads CAMERA yaw, not body yaw — see ship-shard's
            // matching kcc_move_characters_system for rationale. WASD
            // is camera-relative; body chases the resulting velocity
            // direction in `update_body_head_state`.
            &CamLookInput,
            &PlatformSnapSuppressed,
        ),
        With<IsCharacter>,
    >,
    mut landed_writer: MessageWriter<LandedEvent>,
    mut collision_writer: MessageWriter<CharacterCollisionEvent>,
) {
    let dt = integration.0;
    let gravity = Vec3::new(0.0, -(config.surface_gravity as f32), 0.0);

    rapier.refresh_query_pipeline();

    // Planet yaw convention: yaw=0 → facing +Z. Shared KCC convention:
    // yaw=0 → facing +X. Offset by +π/2 so pressing W at planet-yaw=0
    // walks toward +Z as before.
    const PLANET_YAW_OFFSET: f32 = std::f32::consts::FRAC_PI_2;

    for (entity, ctrl, mut desired, platform, stats, mut vel, mut state, look, snap) in
        characters.iter_mut()
    {
        if state.skips_kcc() {
            desired.clear_edges();
            continue;
        }
        let input = CharacterMoveInput {
            dt,
            prev_state: *state,
            velocity: vel.0,
            desired: *desired,
            platform_delta: *platform,
            gravity,
            yaw: look.cam_yaw + PLANET_YAW_OFFSET,
            stats,
            crouching: desired.crouch,
            jump_grace_remaining: 0.0,
            snap_suppressed: snap.0,
        };
        let result = move_one_character(
            &rapier.rigid_body_set,
            &rapier.collider_set,
            rapier.query_pipeline(),
            ctrl,
            input,
            |hit| {
                collision_writer.write(CharacterCollisionEvent { entity, hit });
            },
        );
        if let Some(body) = rapier.rigid_body_set.get_mut(ctrl.body) {
            // Rapier 0.32: `position().translation` is a plain `Vec3`
            // (no `.vector` accessor — `Translation` newtype is gone).
            let cur = body.position().translation;
            body.set_next_kinematic_translation(cur + result.translation);
        }
        vel.0 = result.new_velocity;
        *state = result.new_state;
        if let Some(impact) = result.landed_with_impact_speed {
            landed_writer.write(LandedEvent {
                entity,
                impact_speed: impact,
            });
        }
        desired.clear_edges();
    }
}

// ---------------------------------------------------------------------------
// Physics
// ---------------------------------------------------------------------------

fn physics_step(
    mut rapier: ResMut<RapierContext>,
    config: Res<PlanetConfig>,
    mut physics_time: ResMut<PhysicsTimeRes>,
    mut celestial_time: ResMut<CelestialTimeRes>,
    mut tick: ResMut<ecs::TickCounter>,
    sys_params: Res<SystemParamsRes>,
    mut planet_pos: ResMut<PlanetPositionInSystem>,
    epoch: Res<UniverseEpoch>,
) {
    physics_time.0 += 0.05;
    celestial_time.0 = voxeldust_shard_common::harness::celestial_time_from_epoch(
        &epoch.0,
        sys_params.0.as_ref().map(|s| s.scale.time_scale).unwrap_or(1.0),
    );
    tick.0 += 1;

    // Update planet position from Keplerian orbit.
    update_planet_position(
        &sys_params.0,
        config.planet_index,
        celestial_time.0,
        &mut planet_pos,
    );

    // Step Rapier with surface gravity. Rapier 0.32: `step` takes
    // `gravity` by VALUE (was `&Vector` in 0.22) and the broad-phase
    // is the only spatial-query store — `query_pipeline` is no longer
    // a separate argument here.
    let gravity = rapier3d::math::Vector::new(0.0, -(config.surface_gravity as f32), 0.0);
    let ctx = &mut *rapier;
    ctx.physics_pipeline.step(
        gravity,
        &ctx.integration_params,
        &mut ctx.island_manager,
        &mut ctx.broad_phase,
        &mut ctx.narrow_phase,
        &mut ctx.rigid_body_set,
        &mut ctx.collider_set,
        &mut ctx.impulse_joint_set,
        &mut ctx.multibody_joint_set,
        &mut ctx.ccd_solver,
        &(),
        &(),
    );
}

/// Tangent frame sync: maps Rapier flat-space deltas to sphere surface movement.
/// Re-centers bodies to prevent f32 drift, recomputes tangent frames.
/// Tangent-frame sync: map the character's flat-space horizontal
/// displacement to a rotation of the sphere-tangent basis, re-center the
/// kinematic body to the origin, and transform the character's
/// persisted velocity from the old frame to the new one.
///
/// Runs AFTER `physics_step` so the body's post-step translation
/// reflects the KCC move + any external dynamic interactions.
fn tangent_frame_sync(
    mut rapier: ResMut<RapierContext>,
    config: Res<PlanetConfig>,
    mut players: Query<
        (
            &CharacterController,
            &mut PlanetPosition,
            &mut TangentFrame,
            &mut RapierOrigin,
            &mut CharacterVelocity,
            &mut LocalUp,
        ),
        With<PlanetPlayer>,
    >,
) {
    let planet_radius = config.planet_radius;

    for (ctrl, mut position, mut frame, mut rapier_origin, mut char_vel, mut local_up) in
        &mut players
    {
        let body = match rapier.rigid_body_set.get_mut(ctrl.body) {
            Some(b) => b,
            None => continue,
        };
        let t = body.translation();
        let rapier_pos = DVec3::new(t.x as f64, t.y as f64, t.z as f64);

        let delta = rapier_pos - rapier_origin.0;
        if delta.length_squared() < 1e-8 {
            continue;
        }

        // Map horizontal displacement to sphere surface movement.
        let tangent_disp = frame.east * delta.x + frame.north * delta.z;
        let horiz_dist = tangent_disp.length();

        let new_up = if horiz_dist > 1e-12 {
            let tangent_dir = tangent_disp / horiz_dist;
            let angle = horiz_dist / planet_radius;
            (frame.up * angle.cos() + tangent_dir * angle.sin()).normalize()
        } else {
            frame.up
        };

        let height = rapier_pos.y as f64;

        // Transform the *persisted* character velocity from the old
        // frame to world space, then back into the new frame. This is
        // the piece that used to read the Rapier body's linvel — now
        // we own it via `CharacterVelocity`.
        let vel_local = char_vel.0;
        let vel_world = frame.east * vel_local.x as f64
            + frame.up * vel_local.y as f64
            + frame.north * vel_local.z as f64;

        // Update to new frame.
        position.0 = new_up * (planet_radius + height);
        *frame = TangentFrame::from_up(new_up);
        local_up.0 = new_up;

        let new_vx = vel_world.dot(frame.east) as f32;
        let new_vy = vel_world.dot(frame.up) as f32;
        let new_vz = vel_world.dot(frame.north) as f32;
        char_vel.0 = Vec3::new(new_vx, new_vy, new_vz);

        // Re-center the body to (0, height, 0) in flat space.
        body.set_translation(rapier3d::math::Vector::new(0.0, t.y, 0.0), true);

        rapier_origin.0 = DVec3::new(0.0, height, 0.0);
    }
}

// ---------------------------------------------------------------------------
// Detection: ship proximity, disconnect cleanup
// ---------------------------------------------------------------------------

fn ship_proximity(
    tick: Res<ecs::TickCounter>,
    config: Res<PlanetConfig>,
    planet_pos: Res<PlanetPositionInSystem>,
    celestial_time: Res<CelestialTimeRes>,
    players: Query<
        (
            Entity,
            &SessionId,
            &Name,
            &PlanetPosition,
            &ActionState,
            Has<HandoffPending>,
            &BodyYaw,
            &HeadYaw,
            &HeadPitch,
            &CharacterVelocity,
            &LocomotionState,
            Option<&TurnInPlace>,
        ),
        With<PlanetPlayer>,
    >,
    ships: Query<(&NearbyShipId, &NearbyShipShard, &ShipPosition), With<NearbyShip>>,
    bridge: Res<NetworkBridge>,
    mut commands: Commands,
) {
    if tick.0 % 10 != 0 {
        return;
    }

    for (
        entity,
        session_id,
        name,
        pos,
        actions,
        has_handoff,
        body_yaw,
        head_yaw,
        head_pitch,
        char_vel,
        loco,
        turn,
    ) in &players
    {
        if has_handoff {
            continue;
        }
        let action_pressed = actions.current == 3 && actions.previous != 3;
        if !action_pressed {
            continue;
        }

        for (ship_id, ship_shard, ship_pos) in &ships {
            let dist = (pos.0 - ship_pos.0).length();
            if dist >= 10.0 {
                continue;
            }

            let player_system = pos.0 + planet_pos.0;
            let char_blob = voxeldust_core::character::encode_character_state(
                &voxeldust_core::character::CharacterStateBlob {
                    body_yaw: body_yaw.0,
                    head_yaw: head_yaw.0,
                    head_pitch: head_pitch.0,
                    locomotion: loco.as_u8(),
                    locomotion_speed: char_vel.horizontal().length(),
                    turn: turn.copied(),
                },
            );

            let h = handoff::PlayerHandoff {
                session_token: session_id.0,
                player_name: name.0.clone(),
                position: player_system,
                velocity: DVec3::ZERO,
                rotation: DQuat::from_axis_angle(DVec3::Y, body_yaw.0 as f64),
                forward: DVec3::NEG_Z,
                fly_mode: false,
                speed_tier: 0,
                grounded: true,
                health: 100.0,
                shield: 100.0,
                source_shard: config.shard_id,
                source_tick: tick.0,
                target_star_index: None,
                galaxy_context: None,
                target_planet_seed: None,
                target_planet_index: None,
                target_ship_id: Some(ship_id.0),
                target_ship_shard_id: Some(ship_shard.0),
                ship_system_position: None,
                ship_rotation: None,
                game_time: celestial_time.0,
                warp_target_star_index: None,
                warp_velocity_gu: None,
                target_system_eva: false,
                schema_version: voxeldust_core::character::CHARACTER_SCHEMA_VERSION,
                character_state: char_blob,
            };

            // Mark player as pending handoff.
            commands.entity(entity).insert(HandoffPending);

            // Send to system shard for routing to ship shard.
            if let Ok(reg) = bridge.peer_registry.try_read() {
                let system_shard = reg
                    .find_by_type(ShardType::System)
                    .first()
                    .map(|s| (s.id, s.endpoint.quic_addr));
                if let Some((sid, addr)) = system_shard {
                    let _ = bridge
                        .quic_send_tx
                        .try_send((sid, addr, ShardMsg::PlayerHandoff(h)));
                    info!(
                        session = session_id.0.0,
                        ship_id = ship_id.0,
                        "ship re-entry handoff initiated"
                    );
                }
            }
            break;
        }
    }
}

fn disconnect_cleanup(
    mut commands: Commands,
    tick: Res<ecs::TickCounter>,
    bridge: Res<NetworkBridge>,
    mut rapier: ResMut<RapierContext>,
    mut player_index: ResMut<PlayerEntityIndex>,
    players: Query<(Entity, &SessionId, &CharacterController, &Name, Has<HandoffPending>), With<PlanetPlayer>>,
) {
    if tick.0 % 40 != 0 {
        return;
    }

    let connected_tokens: std::collections::HashSet<SessionToken> =
        if let Ok(reg) = bridge.client_registry.try_read() {
            players
                .iter()
                .filter(|(_, sid, _, _, _)| reg.has_client(&sid.0))
                .map(|(_, sid, _, _, _)| sid.0)
                .collect()
        } else {
            return;
        };

    let orphaned: Vec<(Entity, SessionToken, RigidBodyHandle, String)> = players
        .iter()
        .filter(|(_, sid, _, _, has_handoff)| {
            !connected_tokens.contains(&sid.0) && !has_handoff
        })
        .map(|(entity, sid, ctrl, name, _)| (entity, sid.0, ctrl.body, name.0.clone()))
        .collect();

    for (entity, token, body_handle, player_name) in orphaned {
        let ctx = &mut *rapier;
        ctx.rigid_body_set.remove(
            body_handle,
            &mut ctx.island_manager,
            &mut ctx.collider_set,
            &mut ctx.impulse_joint_set,
            &mut ctx.multibody_joint_set,
            true,
        );
        player_index.0.remove(&token);
        commands.entity(entity).despawn();
        info!(
            session = token.0,
            player = %player_name,
            "cleaned up disconnected player"
        );
    }
}

// ---------------------------------------------------------------------------
// Broadcast
// ---------------------------------------------------------------------------

/// Phase 4-Persist.6: load every persisted `RemoteAccessGrant` from
/// planet-shard's redb back into the in-memory `GrantsRegistry`.
///
/// Channel-name resolution against the `SignalChannelTable` is best-
/// effort: planet-shard doesn't yet persist block configs, so the
/// channel table is empty at boot. Persisted grants whose channels
/// can't resolve come back with an empty `channels` list — they're
/// inert until the player re-issues. Once block persistence ships
/// (separate slice), the same grant data will resolve correctly.
fn load_grants_from_db(
    persistence: Res<PlanetPersistence>,
    channels: Res<voxeldust_core::signal::SignalChannelTable>,
    mut registry: ResMut<voxeldust_core::signal::GrantsRegistry>,
) {
    let now_ms = voxeldust_core::signal::current_unix_millis();
    let _loaded = populate_registry_from_db(&persistence.db, &channels, &mut registry, now_ms);
}

/// Phase 4-Persist.6: drain `GrantsPersistenceQueue` against
/// planet-shard's redb at 0.5s cadence. Uses the shared
/// `flush_grants_now` helper so the heavy lifting is identical to
/// ship-shard's flush path — no per-shard duplication.
fn flush_grants_to_db(
    persistence: Res<PlanetPersistence>,
    registry: Res<voxeldust_core::signal::GrantsRegistry>,
    channels: Res<voxeldust_core::signal::SignalChannelTable>,
    mut queue: ResMut<GrantsPersistenceQueue>,
    tick: Res<ecs::TickCounter>,
) {
    if tick.0 % 10 != 0 {
        return;
    }
    flush_grants_now(&persistence.db, &registry, &channels, &mut queue);
}

/// Phase 4.4: planet-shard HUD delta emitter. Snapshot is the channel
/// table only — planet-shard doesn't have ship-specific auto-publish
/// channels like `ship.speed`. The shared `flush_hud_deltas` helper
/// owns the per-session diff + TCP fan-out.
fn emit_planet_hud_signal_deltas(
    players: Query<&SessionId, With<PlanetPlayer>>,
    channels: Res<voxeldust_core::signal::SignalChannelTable>,
    bridge: Res<NetworkBridge>,
    mut session_map: ResMut<HudSessionMap>,
) {
    let mut snapshot = Vec::new();
    append_channel_table_entries(&channels, &mut snapshot);
    flush_hud_deltas(
        &bridge,
        &mut session_map,
        &snapshot,
        players.iter().map(|sid| sid.0),
    );
}

fn broadcast_world_state(
    players: Query<
        (
            &SessionId,
            &Name,
            &PlanetPosition,
            &BodyYaw,
            &HeadYaw,
            &HeadPitch,
            &CharacterVelocity,
            &LocomotionState,
            Option<&TurnInPlace>,
            &voxeldust_core::character::LookTarget,
            // Phase I: present only while ragdolling.
            Option<&voxeldust_core::character::RagdollHandles>,
            // Phase J: present only while the player is engaged with a
            // functional block via the tablet HUD.
            Option<&voxeldust_core::character::IsHoldingTablet>,
            Option<&voxeldust_core::character::TabletCursor>,
        ),
        With<PlanetPlayer>,
    >,
    ships: Query<(&NearbyShipId, &ShipPosition, &ShipRotation), With<NearbyShip>>,
    config: Res<PlanetConfig>,
    sys_params: Res<SystemParamsRes>,
    planet_pos: Res<PlanetPositionInSystem>,
    cached_all_planets: Res<CachedAllPlanetPositions>,
    celestial_time: Res<CelestialTimeRes>,
    tick: Res<ecs::TickCounter>,
    bridge: Res<NetworkBridge>,
    external: Res<ExternalEntities>,
    // Phase I: needed for `collect_ragdoll_bones` to read each
    // active ragdoll body's pose. Read-only — broadcast doesn't
    // mutate the rapier world.
    rapier: Res<RapierContext>,
) {
    let player_snapshots: Vec<PlayerSnapshotData> = players
        .iter()
        .map(|(sid, _, pos, body, head_y, head_p, vel, loco, turn, look, ragdoll, holding, cursor)| PlayerSnapshotData {
            player_id: sid.0.0,
            position: pos.0,
            // Identity for the planet shard. The camera composes
            // `body × rotation_from_look(LocalLook)`, where head_look
            // already encodes the player's full camera yaw/pitch in
            // the planet's flat-Y-up frame — applying body_yaw here
            // would double-count it. Renderers wanting the player's
            // visual body rotation read the dedicated `body_yaw`
            // field below and apply `Quat::from_axis_angle(Y, body_yaw)`
            // to the planet shard's flat frame.
            rotation: DQuat::IDENTITY,
            velocity: DVec3::ZERO,
            grounded: true,
            health: 100.0,
            shield: 100.0,
            seated: false,
            body_yaw: body.0,
            head_yaw: head_y.0,
            head_pitch: head_p.0,
            locomotion: loco.as_u8(),
            locomotion_speed: vel.horizontal().length(),
            is_turning: turn.is_some(),
            turn_target_yaw: turn.map(|t| t.target_body_yaw).unwrap_or(0.0),
            turn_t: turn.map(|t| t.t).unwrap_or(0.0),
            // Look-at target encoded as a delta from `position` —
            // f32 deltas are precision-safe because the target sits
            // within `look_attention_distance` (~30 m) of the player.
            look_target_delta: look
                .0
                .map(|t| (t - pos.0).as_vec3()),
            ragdoll_bones: collect_ragdoll_bones(ragdoll, &rapier),
            is_holding_tablet: holding.is_some(),
            tablet_cursor_uv: cursor.map(|c| c.0).unwrap_or(glam::Vec2::ZERO),
        })
        .collect();

    // Sky bodies from system params (uses cached planet positions to avoid
    // redundant Kepler equation solving — positions are computed in physics_step).
    let mut bodies = Vec::new();
    if let Some(ref sys) = sys_params.0 {
        bodies.reserve(sys.planets.len() + 1);
        // Star — physics-derived stellar state propagated from the system-shard's
        // authoritative `SystemParams.star` (single source for the whole system).
        bodies.push(CelestialBodyData {
            body_id: 0,
            position: -planet_pos.0,
            radius: sys.star.radius_m,
            color: sys.star.color,
            stellar: Some(sys.star.stellar),
            planetary: None,
            rotation_params: None,
        });
        for (i, planet) in sys.planets.iter().enumerate() {
            let planet_sys_pos = cached_all_planets.0.get(i)
                .copied()
                .unwrap_or_else(|| compute_planet_position(planet, celestial_time.0));
            bodies.push(CelestialBodyData {
                body_id: (i + 1) as u32,
                position: planet_sys_pos - planet_pos.0,
                radius: planet.radius_m,
                color: planet.color,
                stellar: None,
                planetary: Some(planet.geophysics(&sys.star.stellar)),
                rotation_params: Some(planet.rotation_params()),
            });
        }
    }

    // Lighting from star.
    let first_player_pos = players
        .iter()
        .next()
        .map(|(_, _, p, ..)| p.0)
        .unwrap_or(DVec3::new(0.0, config.planet_radius + 2.0, 0.0));

    let lighting = if let Some(ref sys) = sys_params.0 {
        let star_dir = (-planet_pos.0 - first_player_pos).normalize();
        let l = compute_lighting(planet_pos.0 + first_player_pos, &sys.star);
        Some(LightingData {
            sun_direction: star_dir,
            sun_color: l.sun_color,
            sun_intensity: l.sun_intensity,
            ambient: l.ambient,
        })
    } else {
        None
    };

    let ship_renders: Vec<ShipRenderData> = ships
        .iter()
        .map(|(id, pos, rot)| ShipRenderData {
            ship_id: id.0,
            position: pos.0,
            rotation: rot.0,
            is_own_ship: false,
        })
        .collect();

    // Build unified entities list: external (system-space) entities transformed
    // into planet-local + every local grounded player as a GroundedPlayer.
    // Origin == planet_pos.0, so the client adds origin to reconstruct world-space.
    let mut entities: Vec<ObservableEntityData> = Vec::with_capacity(
        external.entities.len() + players.iter().count(),
    );
    for e in &external.entities {
        let mut entity = e.clone();
        // Convert system-space position to planet-local frame (origin = planet_pos.0).
        entity.position = e.position - planet_pos.0;
        entities.push(entity);
    }
    for (sid, name, pos, body, head_y, head_p, vel, loco, turn, look, ragdoll, holding, cursor) in players.iter() {
        entities.push(ObservableEntityData {
            entity_id: sid.0.0,
            kind: EntityKind::GroundedPlayer,
            position: pos.0,
            // Identity — same rationale as PlayerSnapshot above.
            rotation: DQuat::IDENTITY,
            velocity: DVec3::ZERO,
            bounding_radius: 1.0,
            lod_tier: LodTier::Full,
            shard_id: config.shard_id.0,
            shard_type: ShardType::Planet as u8,
            is_own: false,
            name: name.0.clone(),
            health: 100.0,
            shield: 100.0,
            body_yaw: body.0,
            head_yaw: head_y.0,
            head_pitch: head_p.0,
            locomotion: loco.as_u8(),
            locomotion_speed: vel.horizontal().length(),
            is_turning: turn.is_some(),
            turn_target_yaw: turn.map(|t| t.target_body_yaw).unwrap_or(0.0),
            turn_t: turn.map(|t| t.t).unwrap_or(0.0),
            look_target_delta: look
                .0
                .map(|t| (t - pos.0).as_vec3()),
            ragdoll_bones: collect_ragdoll_bones(ragdoll, &rapier),
            is_holding_tablet: holding.is_some(),
            tablet_cursor_uv: cursor.map(|c| c.0).unwrap_or(glam::Vec2::ZERO),
        });
    }

    let ws = ServerMsg::WorldState(WorldStateData {
        tick: tick.0,
        origin: planet_pos.0,
        players: player_snapshots,
        bodies,
        ships: ship_renders,
        lighting,
        game_time: celestial_time.0,
        warp_target_star_index: 0xFFFFFFFF,
        autopilot: None,
        sub_grids: vec![],
        entities,
        // Phase 4.4.5: HUD signal updates ship as `HudSignalDelta`
        // TCP messages emitted by `emit_planet_hud_signal_deltas`,
        // running alongside this WorldState builder in the same
        // broadcast schedule.
    });
    if bridge.broadcast_tx.try_send(ws).is_err() {
        tracing::warn!("WorldState broadcast dropped — channel full");
    }
}

/// Emit a PlanetPlayerDigest at 1 Hz so the system shard can include surface
/// players in its AOI for distant observers (ships, EVA).
fn send_player_digest(
    players: Query<(&SessionId, &Name, &PlanetPosition), With<PlanetPlayer>>,
    config: Res<PlanetConfig>,
    planet_pos: Res<PlanetPositionInSystem>,
    tick: Res<ecs::TickCounter>,
    bridge: Res<NetworkBridge>,
) {
    // ~1 Hz at 20 Hz tick.
    if tick.0 % 20 != 0 {
        return;
    }
    if players.is_empty() {
        return;
    }

    let entries: Vec<PlanetPlayerDigestEntry> = players
        .iter()
        .map(|(sid, name, pos)| PlanetPlayerDigestEntry {
            session_token: sid.0,
            player_name: name.0.clone(),
            // Transform planet-local position to system-space (what system shard expects).
            position: planet_pos.0 + pos.0,
            rotation: DQuat::IDENTITY,
            planet_index: config.planet_index,
        })
        .collect();

    let digest = ShardMsg::PlanetPlayerDigest(PlanetPlayerDigestData {
        planet_shard: config.shard_id,
        planet_seed: config.planet_seed,
        planet_index: config.planet_index,
        entries,
        tick: tick.0,
    });

    // Find the system shard and send.
    if let Ok(reg) = bridge.peer_registry.try_read() {
        if let Some(info) = reg.find_by_type(ShardType::System).first() {
            let _ = bridge.quic_send_tx.try_send((info.id, info.endpoint.quic_addr, digest));
        }
    }
}

// ---------------------------------------------------------------------------
// Diagnostics
// ---------------------------------------------------------------------------

fn log_state(
    players: Query<&PlanetPlayer>,
    config: Res<PlanetConfig>,
    physics_time: Res<PhysicsTimeRes>,
    celestial_time: Res<CelestialTimeRes>,
    tick: Res<ecs::TickCounter>,
) {
    if tick.0 % 100 != 0 || tick.0 == 0 {
        return;
    }
    info!(
        physics_time = format!("{:.1}s", physics_time.0),
        celestial_time = format!("{:.1}s", celestial_time.0),
        players = players.iter().count(),
        surface_g = format!("{:.2} m/s²", config.surface_gravity),
        "planet state"
    );
}

// ---------------------------------------------------------------------------
// App construction
// ---------------------------------------------------------------------------

fn build_app(
    shard_id: ShardId,
    planet_seed: u64,
    planet_radius: f64,
    planet_mass: f64,
    system_seed: Option<u64>,
    planet_index: u32,
    universe_epoch: Arc<std::sync::atomic::AtomicU64>,
) -> App {
    let surface_gravity = G * planet_mass / (planet_radius * planet_radius);

    let rigid_body_set = RigidBodySet::new();
    let mut collider_set = ColliderSet::new();

    // Ground: halfspace at Y=0. Rapier 0.32: `halfspace` still wraps
    // the normal in `Unit<Vector>`, but `Vector` is glam `Vec3` now and
    // doesn't impl nalgebra's `Normed`. `Unit::new_unchecked` doesn't
    // require it; we feed `Vector::Y` directly since it's known unit.
    let ground = ColliderBuilder::halfspace(nalgebra::Unit::new_unchecked(
        rapier3d::math::Vector::Y,
    ))
    .translation(rapier3d::math::Vector::ZERO)
    .collision_groups(InteractionGroups::new(
        Group::GROUP_2,
        Group::GROUP_1,
        InteractionTestMode::default(),
    ))
    .build();
    collider_set.insert(ground);

    let system_params = system_seed.map(SystemParams::from_seed);
    let planet_position_in_system = system_params
        .as_ref()
        .and_then(|sys| sys.planets.get(planet_index as usize))
        .map(|p| compute_planet_position(p, 0.0))
        .unwrap_or(DVec3::ZERO);

    let mut app = App::new();

    // Phase 4-Persist.6: open the planet-shard's redb file. Currently
    // houses only the grants table; future block-persistence work
    // will reuse the same `db` handle (mirror of ship-shard's
    // `ShipPersistence`).
    let db_path = format!("/tmp/voxeldust-planet-{}.redb", shard_id.0);
    let db = redb::Database::create(&db_path)
        .unwrap_or_else(|e| panic!("failed to create redb at {db_path}: {e}"));
    app.insert_resource(PlanetPersistence { db });

    const PLANET_TICK_DT: f32 = 0.05;
    app.insert_resource(PlanetIntegrationDt(PLANET_TICK_DT));
    app.insert_resource(RapierContext {
        rigid_body_set,
        collider_set,
        integration_params: {
            let mut p = IntegrationParameters::default();
            p.dt = PLANET_TICK_DT;
            p
        },
        physics_pipeline: PhysicsPipeline::new(),
        island_manager: IslandManager::new(),
        broad_phase: BroadPhaseBvh::new(),
        narrow_phase: NarrowPhase::new(),
        impulse_joint_set: ImpulseJointSet::new(),
        multibody_joint_set: MultibodyJointSet::new(),
        ccd_solver: CCDSolver::new(),
    });
    app.insert_resource(PlanetConfig {
        shard_id,
        planet_seed,
        planet_radius,
        planet_mass,
        surface_gravity,
        system_seed,
        planet_index,
    });
    app.insert_resource(SystemParamsRes(system_params));
    app.insert_resource(PlanetPositionInSystem(planet_position_in_system));
    app.insert_resource(CachedAllPlanetPositions::default());
    app.insert_resource(CelestialTimeRes::default());
    app.insert_resource(PhysicsTimeRes::default());
    app.insert_resource(ecs::TickCounter::default());
    app.insert_resource(UniverseEpoch(universe_epoch));
    app.insert_resource(PlayerEntityIndex::default());
    app.insert_resource(ShipEntityIndex::default());
    app.insert_resource(PendingHandoffs::default());
    app.insert_resource(ExternalEntities::default());
    // Phase 4.4.5: per-session HUD signal delta state for the
    // `emit_planet_hud_signal_deltas` system. Lazy-populated;
    // entries pruned on session disappearance.
    app.insert_resource(HudSessionMap::default());

    // Signal pipeline — same `SignalPipelinePlugin` as ship-shard. Enables
    // placing functional blocks (seats, doors, lights, future thrusters)
    // anywhere on the planet surface and wiring them via the standard
    // signal channels. Without this plugin a Seat block on a planet would
    // publish into a void; with it, in-shard publish/subscribe works
    // identically to a ship's interior. Per-block-kind subscriber systems
    // get added under `SignalSet::Subscribe` as planet-side block kinds
    // come online (a follow-up phase).
    app.add_plugins(SignalPipelinePlugin);

    // Messages.
    app.add_message::<ClientConnectedMsg>();
    app.add_message::<PlayerInputMsg>();
    app.add_message::<InboundHandoffMsg>();
    app.add_message::<ShipNearbyMsg>();
    app.add_message::<HandoffAcceptedMsg>();
    app.add_message::<ShipColliderSyncMsg>();
    // Character-layer events (reserved — consumed by future gameplay).
    app.add_message::<LandedEvent>();
    app.add_message::<CharacterCollisionEvent>();

    // System ordering.
    app.configure_sets(
        Update,
        (
            PlanetSet::Bridge,
            PlanetSet::Spawn,
            PlanetSet::Input,
            PlanetSet::Physics,
            PlanetSet::Detection,
            PlanetSet::Broadcast,
            PlanetSet::Diagnostics,
        )
            .chain(),
    );

    // Bridge.
    app.add_systems(
        Update,
        (drain_connects, drain_input, drain_quic).in_set(PlanetSet::Bridge),
    );

    // Spawn: process_handoffs stores handoff data in PendingHandoffs resource,
    // process_connects consumes it to spawn at the correct position.
    app.add_systems(
        Update,
        (
            process_connects,
            process_handoffs,
            process_ship_nearby,
            process_ship_colliders,
            process_handoff_accepted,
        )
            .in_set(PlanetSet::Spawn),
    );

    // apply_deferred so newly spawned entities are visible to Input/Physics.
    app.add_systems(
        Update,
        bevy_ecs::schedule::ApplyDeferred
            .after(PlanetSet::Spawn)
            .before(PlanetSet::Input),
    );

    // Input.
    app.add_systems(Update, process_input.in_set(PlanetSet::Input));

    // Physics.
    //
    // Order: `kcc_move_characters_system` drives character translation
    // from `DesiredMovement` → `set_next_kinematic_translation`. Then
    // `physics_step` runs Rapier's full step (kinematic bodies pick up
    // their new positions; any dynamic bodies integrate normally).
    // `tangent_frame_sync` re-centers the character in the flat Rapier
    // frame and updates the sphere-surface tangent basis.
    app.add_systems(
        Update,
        (
            // Body/head decoupling first, so the new BodyYaw drives KCC.
            update_body_head_state,
            // Phase H: attention/look-at runs after body_yaw is fresh
            // (so the FOV cone is in the right direction) and before
            // broadcast so the new target ships out this tick.
            update_look_targets,
            // Phase I: ragdoll lifetime tick — see ship-shard's
            // matching note for the rationale.
            update_ragdoll_lifetime,
            kcc_move_characters_system,
            physics_step,
            refresh_planet_position_cache,
            tangent_frame_sync,
            update_ship_collider_positions,
        )
            .chain()
            .in_set(PlanetSet::Physics),
    );

    // Detection.
    // `process_tablet_*` slot in here so their `Commands` mutations
    // (insert/remove `IsHoldingTablet` + `TabletCursor`) flush through
    // the post-Detection `apply_deferred` and are visible to
    // `broadcast_world_state` on the same tick. Putting them in
    // `Input` instead would leave the inserts pending until physics
    // ran a system with a `Commands` parameter.
    app.add_systems(
        Update,
        (
            ship_proximity,
            disconnect_cleanup,
            process_tablet_interact,
            process_tablet_cursor_updates,
        )
            .in_set(PlanetSet::Detection),
    );

    // apply_deferred so despawned entities don't appear in broadcast.
    app.add_systems(
        Update,
        bevy_ecs::schedule::ApplyDeferred
            .after(PlanetSet::Detection)
            .before(PlanetSet::Broadcast),
    );

    // Broadcast.
    app.add_systems(
        Update,
        (
            broadcast_world_state,
            send_player_digest,
            emit_planet_hud_signal_deltas,
        )
            .in_set(PlanetSet::Broadcast),
    );

    // Phase 4-Persist.6: load persisted grants once at boot. Channel-
    // name resolution is best-effort against the (initially empty)
    // SignalChannelTable; future block persistence will populate it.
    app.add_systems(Startup, load_grants_from_db);
    // Periodic flush of GrantsPersistenceQueue → redb. 0.5s cadence
    // mirrors ship-shard's flush rate so a planet-shard crash bounds
    // grant-loss to <1s of in-flight ops.
    app.add_systems(Update, flush_grants_to_db);

    // Diagnostics.
    app.add_systems(Update, log_state.in_set(PlanetSet::Diagnostics));

    app
}

// ---------------------------------------------------------------------------
// Entry point
// ---------------------------------------------------------------------------

fn main() {
    let args = Args::parse();

    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "info".into()),
        )
        .init();

    // Install Prometheus exporter before any subsystem emits a metric.
    // Healthz server reads the resulting handle to mount /metrics.
    voxeldust_shard_common::observability::install_prometheus_recorder();

    let (planet_radius, planet_mass) = if let Some(sys_seed) = args.system_seed {
        let sys = SystemParams::from_seed(sys_seed);
        if let Some(planet) = sys.planets.get(args.planet_index as usize) {
            (planet.radius_m, planet.mass_kg)
        } else {
            (6.371e6, 5.972e24)
        }
    } else {
        (6.371e6, 5.972e24)
    };

    let bind = "0.0.0.0";
    let config = ShardHarnessConfig {
        shard_id: ShardId(args.shard_id),
        shard_type: ShardType::Planet,
        tcp_addr: format!("{bind}:{}", args.tcp_port).parse().unwrap(),
        udp_addr: format!("{bind}:{}", args.udp_port).parse().unwrap(),
        quic_addr: format!("{bind}:{}", args.quic_port).parse().unwrap(),
        orchestrator_url: args.orchestrator,
        orchestrator_heartbeat_addr: args.orchestrator_heartbeat,
        healthz_addr: format!("{bind}:{}", args.healthz_port).parse().unwrap(),
        planet_seed: Some(args.seed),
        system_seed: args.system_seed,
        ship_id: None,
        galaxy_seed: None,
        host_shard_id: None,
        advertise_host: args.advertise_host,
        wire_dict_v2_send: true,
    };

    info!(
        shard_id = args.shard_id,
        planet_seed = args.seed,
        radius_km = planet_radius / 1000.0,
        surface_g = format!("{:.2}", G * planet_mass / (planet_radius * planet_radius)),
        "planet shard starting"
    );

    let rt = tokio::runtime::Runtime::new().expect("failed to create tokio runtime");
    rt.block_on(async {
        let harness = ShardHarness::new(config);
        let universe_epoch = harness.epoch_arc();
        let app = build_app(
            ShardId(args.shard_id),
            args.seed,
            planet_radius,
            planet_mass,
            args.system_seed,
            args.planet_index,
            universe_epoch,
        );

        info!("planet shard ECS app built, starting harness");
        harness.run_ecs(app).await;
    });
}
