use flatbuffers::FlatBufferBuilder;
use glam::{DQuat, DVec3};
use thiserror::Error;

use crate::handoff;
use crate::protocol_generated as fb;
use crate::shard_types::{self, ShardId, SessionToken};

/// All possible inter-shard messages in a type-safe enum.
#[derive(Debug, Clone)]
pub enum ShardMsg {
    PlayerHandoff(handoff::PlayerHandoff),
    HandoffAccepted(handoff::HandoffAccepted),
    GhostUpdate(handoff::GhostUpdate),
    Heartbeat(shard_types::ShardHeartbeat),
    SplitDirective(SplitDirective),
    MergeDirective(MergeDirective),
    ShipPositionUpdate(ShipPositionUpdate),
    ShipControlInput(ShipControlInput),
    CrossShardBlockEdits(CrossShardBlockEdits),
    SystemSceneUpdate(SystemSceneUpdateData),
    AutopilotCommand(AutopilotCommandData),
    ShipNearbyInfo(ShipNearbyInfoData),
    WarpAutopilotCommand(WarpAutopilotCommandData),
    HostSwitch(HostSwitchData),
    /// Ship physical properties update (ship shard → system shard).
    /// Sent when block composition changes (aggregation recomputes).
    ShipPropertiesUpdate(ShipPropertiesUpdateData),
    /// Cross-shard signal broadcast — single signal (legacy).
    SignalBroadcast(SignalBroadcastData),
    /// Batched signal broadcast — multiple signals in one message.
    SignalBroadcastBatch(SignalBroadcastBatchData),
    /// Visibility directive from system shard to ship shard.
    /// Tells a ship which other ships are within visual range.
    VisibilityDirective(VisibilityDirectiveData),
    /// Ship collider shapes synced from ship shard to host (planet/system) shard.
    /// Host shard builds Rapier compound colliders from these for physical collision.
    ShipColliderSync(ShipColliderSyncData),
    /// System shard → ship/planet shard: unified AOI entity set for a single observer.
    /// Replaces VisibilityDirective + ShipNearbyInfo + SystemSceneUpdate.ships
    /// with a single authoritative stream.
    SystemEntitiesUpdate(SystemEntitiesUpdateData),
    /// Planet shard → system shard: aggregate surface-player positions at 1 Hz.
    /// Feeds the system shard AOI so distant observers (ships, EVA) can see
    /// players on planets without a direct secondary connection.
    PlanetPlayerDigest(PlanetPlayerDigestData),
}

#[derive(Debug, Clone)]
pub struct SplitDirective {
    pub target_shard: ShardId,
    pub sectors_to_split: Vec<u8>,
    pub planet_seed: u64,
}

#[derive(Debug, Clone)]
pub struct MergeDirective {
    pub absorbing_shard: ShardId,
    pub merging_shard: ShardId,
    pub sectors: Vec<u8>,
}

/// Server-authoritative autopilot state snapshot sent alongside ship position.
#[derive(Debug, Clone)]
pub struct AutopilotSnapshotData {
    pub phase: u8,
    pub mode: u8,
    pub target_planet_index: u32,
    pub thrust_tier: u8,
    pub intercept_pos: DVec3,
    pub target_arrival_vel: DVec3,
    pub braking_committed: bool,
    pub eta_real_seconds: f64,
    pub target_orbit_altitude: f64,
}

/// Ship exterior state synced from system shard to ship shard.
#[derive(Debug, Clone)]
pub struct ShipPositionUpdate {
    pub ship_id: u64,
    pub position: DVec3,
    pub velocity: DVec3,
    pub rotation: DQuat,
    pub angular_velocity: DVec3,
    pub autopilot: Option<AutopilotSnapshotData>,
    /// Authoritative atmosphere state from system-shard physics.
    pub in_atmosphere: bool,
    /// Planet index if in atmosphere, -1 otherwise.
    pub atmosphere_planet_index: i32,
    /// Gravitational acceleration at ship position (m/s²), world frame.
    pub gravity_acceleration: DVec3,
    /// Atmospheric density at ship altitude (kg/m³), 0 if not in atmosphere.
    pub atmosphere_density: f64,
}

/// Pilot control input sent from ship shard to system shard.
#[derive(Debug, Clone)]
pub struct ShipControlInput {
    pub ship_id: u64,
    pub thrust: DVec3,
    pub torque: DVec3,
    pub braking: bool,
    pub tick: u64,
}

/// Scene update from system shard to ship/planet shards.
#[derive(Debug, Clone)]
pub struct SystemSceneUpdateData {
    pub game_time: f64,
    pub bodies: Vec<CelestialBodySnapshotData>,
    pub ships: Vec<ShipSnapshotEntryData>,
    pub lighting: LightingInfoData,
}

#[derive(Debug, Clone)]
pub struct CelestialBodySnapshotData {
    pub body_id: u32,
    pub position: DVec3,
    pub radius: f64,
    pub color: [f32; 3],
}

#[derive(Debug, Clone)]
pub struct ShipSnapshotEntryData {
    pub ship_id: u64,
    pub position: DVec3,
    pub rotation: DQuat,
    pub is_own_ship: bool,
}

#[derive(Debug, Clone)]
pub struct LightingInfoData {
    pub sun_direction: DVec3,
    pub sun_color: [f32; 3],
    pub sun_intensity: f32,
    pub ambient: f32,
}

/// Autopilot command sent from ship shard to system shard.
#[derive(Debug, Clone)]
pub struct AutopilotCommandData {
    pub ship_id: u64,
    /// Target body id: 1..N = planets. 0xFFFFFFFF = disengage.
    pub target_body_id: u32,
    /// Speed tier (0-4).
    pub speed_tier: u8,
    /// Autopilot mode: 0=DirectApproach, 1=OrbitInsertion, 2=Landing, 3=Takeoff, 4=Departure.
    pub autopilot_mode: u8,
}

/// Ship position/state sent from system shard to planet shards within SOI.
#[derive(Debug, Clone)]
pub struct ShipNearbyInfoData {
    pub ship_id: u64,
    pub ship_shard_id: ShardId,
    pub position: DVec3,
    pub rotation: DQuat,
    pub velocity: DVec3,
    /// System shard's authoritative celestial time for time synchronization.
    pub game_time: f64,
}

/// Warp autopilot command sent from ship shard to system shard.
#[derive(Debug, Clone)]
pub struct WarpAutopilotCommandData {
    pub ship_id: u64,
    /// Target star index in the galaxy. `u32::MAX` = disengage warp.
    pub target_star_index: u32,
    pub galaxy_seed: u64,
}

/// Host switch: tells a ship shard to change its physics host.
/// Includes full endpoint info so the ship shard can send ShardPreConnect
/// to its client for secondary UDP (dual-shard compositing).
#[derive(Debug, Clone)]
pub struct HostSwitchData {
    pub ship_id: u64,
    pub new_host_shard_id: ShardId,
    pub new_host_quic_addr: String,
    pub new_host_tcp_addr: String,
    pub new_host_udp_addr: String,
    pub new_host_shard_type: u8, // 1=System, 3=Galaxy
    pub seed: u64,               // galaxy_seed or system_seed
}

/// Ship physical properties derived from block composition.
/// Sent from ship shard to system shard when blocks change.
#[derive(Debug, Clone)]
pub struct ShipPropertiesUpdateData {
    pub ship_id: u64,
    pub mass_kg: f64,
    pub max_thrust_forward_n: f64,
    pub max_thrust_reverse_n: f64,
    pub max_torque_nm: f64,
    pub thrust_multiplier: f64,
    pub dimensions: (f64, f64, f64),
}

/// Cross-shard signal broadcast — single signal (legacy, kept for backward compat).
#[derive(Debug, Clone)]
pub struct SignalBroadcastData {
    pub source_shard_id: u64,
    pub channel_name: String,
    /// 0=Bool, 1=Float, 2=State
    pub value_type: u8,
    pub value_data: f32,
    /// 1=ShortRange, 2=LongRange, 3=Radio
    pub scope: u8,
    pub range_m: f64,
    pub source_position: DVec3,
}

/// A single signal entry within a batched broadcast.
#[derive(Debug, Clone)]
pub struct SignalBroadcastEntry {
    pub channel_name: String,
    /// 0=Bool, 1=Float, 2=State
    pub value_type: u8,
    pub value_data: f32,
    /// 1=ShortRange, 2=LongRange, 3=Radio
    pub scope: u8,
    /// Range in meters (ShortRange only; 0.0 for LongRange/Radio).
    pub range_m: f64,
    /// Radio frequency (Radio scope only; 0 for other scopes).
    pub frequency: u32,
}

/// Batched signal broadcast — multiple dirty signals packed into one message.
/// Sent once per destination per tick instead of once per signal per destination.
#[derive(Debug, Clone)]
pub struct SignalBroadcastBatchData {
    pub source_shard_id: u64,
    pub source_position: DVec3,
    pub entries: Vec<SignalBroadcastEntry>,
}

/// A single visible ship entry in a VisibilityDirective.
#[derive(Debug, Clone)]
pub struct VisibleShipEntryData {
    pub ship_id: u64,
    pub shard_id: ShardId,
    pub tcp_addr: String,
    pub udp_addr: String,
    pub distance: f64,
}

/// System shard → ship shard: which other ships are visible.
/// Sent when the visibility set changes (enter/leave events).
#[derive(Debug, Clone)]
pub struct VisibilityDirectiveData {
    pub ship_id: u64,
    pub visible_ships: Vec<VisibleShipEntryData>,
}

/// Collider shapes for one chunk of a ship grid.
#[derive(Debug, Clone)]
pub struct ChunkColliderData {
    pub chunk_key: glam::IVec3,
    /// Each pair is (center_position, half_extents) in ship-local coordinates.
    pub shapes: Vec<(glam::Vec3, glam::Vec3)>,
}

/// Ship hull collider shapes for physical collision on host shards.
/// Sent from ship shard to planet/system shard when blocks change.
#[derive(Debug, Clone)]
pub struct ShipColliderSyncData {
    pub ship_id: u64,
    pub chunks: Vec<ChunkColliderData>,
    /// Tight AABB of all solid blocks (block coords).
    pub hull_min: glam::Vec3,
    pub hull_max: glam::Vec3,
}

/// Block edits affecting chunks on adjacent shard boundaries.
#[derive(Debug, Clone)]
pub struct CrossShardBlockEdits {
    pub chunk_address: Vec<u8>,
    pub edits: Vec<u8>,
    pub seq: u64,
}

/// Target of a SystemEntitiesUpdate — identifies which shard should consume it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AoiTarget {
    /// Update is for a ship shard hosting this ship id.
    Ship(u64),
    /// Update is for the planet shard at this planet index.
    Planet(u32),
}

/// Unified AOI payload: system shard → ship/planet shard.
/// The ship/planet shard merges `entities` into its own WorldState broadcast,
/// giving every observer a single authoritative view of nearby ships, EVA
/// players, and surface-player aggregates.
#[derive(Debug, Clone)]
pub struct SystemEntitiesUpdateData {
    pub target: AoiTarget,
    /// Observer anchor in system coordinates (used for LOD tier computation).
    pub observer_position: DVec3,
    pub entities: Vec<crate::client_message::ObservableEntityData>,
    pub tick: u64,
}

/// One entry in a PlanetPlayerDigest.
#[derive(Debug, Clone)]
pub struct PlanetPlayerDigestEntry {
    pub session_token: SessionToken,
    pub player_name: String,
    /// System-space position (planet shard transforms before sending).
    pub position: DVec3,
    pub rotation: DQuat,
    pub planet_index: u32,
}

/// Planet shard → system shard: aggregate surface-player positions at 1 Hz.
#[derive(Debug, Clone)]
pub struct PlanetPlayerDigestData {
    pub planet_shard: ShardId,
    pub planet_seed: u64,
    pub planet_index: u32,
    pub entries: Vec<PlanetPlayerDigestEntry>,
    pub tick: u64,
}

#[derive(Debug, Error)]
pub enum MessageError {
    #[error("invalid flatbuffer: {0}")]
    InvalidBuffer(String),
    #[error("unknown payload type: {0}")]
    UnknownPayload(u8),
    #[error("missing required field: {0}")]
    MissingField(&'static str),
}

// -- Helpers for converting between glam and flatbuffers structs --

fn to_fb_vec3d(v: &DVec3) -> fb::Vec3d {
    fb::Vec3d::new(v.x, v.y, v.z)
}

fn from_fb_vec3d(v: &fb::Vec3d) -> DVec3 {
    DVec3::new(v.x(), v.y(), v.z())
}

fn to_fb_quatd(q: &DQuat) -> fb::Quatd {
    fb::Quatd::new(q.x, q.y, q.z, q.w)
}

fn from_fb_quatd(q: &fb::Quatd) -> DQuat {
    DQuat::from_xyzw(q.x(), q.y(), q.z(), q.w())
}

impl ShardMsg {
    /// Serialize this message into a FlatBuffer byte vector.
    pub fn serialize(&self) -> Vec<u8> {
        let mut builder = crate::builder_pool::acquire(512);

        match self {
            ShardMsg::PlayerHandoff(h) => {
                let name = builder.create_string(&h.player_name);
                let pos = to_fb_vec3d(&h.position);
                let vel = to_fb_vec3d(&h.velocity);
                let rot = to_fb_quatd(&h.rotation);
                let fwd = to_fb_vec3d(&h.forward);

                // Build optional galaxy context.
                let galaxy_ctx = h.galaxy_context.as_ref().map(|ctx| {
                    let sp = to_fb_vec3d(&ctx.star_position);
                    fb::GalaxyHandoffContext::create(
                        &mut builder,
                        &fb::GalaxyHandoffContextArgs {
                            galaxy_seed: ctx.galaxy_seed,
                            star_index: ctx.star_index,
                            star_position: Some(&sp),
                        },
                    )
                });

                // Build optional ship position/rotation for ship→planet handoffs.
                let ship_sys_pos = h.ship_system_position.as_ref().map(|p| to_fb_vec3d(p));
                let ship_rot = h.ship_rotation.as_ref().map(|r| to_fb_quatd(r));
                let warp_vel = h.warp_velocity_gu.as_ref().map(|v| to_fb_vec3d(v));

                let handoff = fb::PlayerHandoff::create(
                    &mut builder,
                    &fb::PlayerHandoffArgs {
                        session_token: h.session_token.0,
                        player_name: Some(name),
                        position: Some(&pos),
                        velocity: Some(&vel),
                        rotation: Some(&rot),
                        forward: Some(&fwd),
                        fly_mode: h.fly_mode,
                        speed_tier: h.speed_tier,
                        grounded: h.grounded,
                        health: h.health,
                        shield: h.shield,
                        source_shard_id: h.source_shard.0,
                        source_tick: h.source_tick,
                        target_star_index: h.target_star_index.unwrap_or(0xFFFFFFFF),
                        galaxy_context: galaxy_ctx,
                        target_planet_seed: h.target_planet_seed.unwrap_or(u64::MAX),
                        target_planet_index: h.target_planet_index.unwrap_or(u32::MAX),
                        target_ship_id: h.target_ship_id.unwrap_or(u64::MAX),
                        target_ship_shard_id: h.target_ship_shard_id.map(|s| s.0).unwrap_or(u64::MAX),
                        ship_system_position: ship_sys_pos.as_ref(),
                        ship_rotation: ship_rot.as_ref(),
                        game_time: h.game_time,
                        warp_target_star_index: h.warp_target_star_index.unwrap_or(0xFFFFFFFF),
                        warp_velocity: warp_vel.as_ref(),
                        target_system_eva: h.target_system_eva,
                    },
                );

                let msg = fb::ShardMessage::create(
                    &mut builder,
                    &fb::ShardMessageArgs {
                        payload_type: fb::ShardPayload::PlayerHandoff,
                        payload: Some(handoff.as_union_value()),
                    },
                );
                builder.finish(msg, None);
            }

            ShardMsg::HandoffAccepted(a) => {
                let accepted = fb::HandoffAccepted::create(
                    &mut builder,
                    &fb::HandoffAcceptedArgs {
                        session_token: a.session_token.0,
                        target_shard_id: a.target_shard.0,
                    },
                );
                let msg = fb::ShardMessage::create(
                    &mut builder,
                    &fb::ShardMessageArgs {
                        payload_type: fb::ShardPayload::HandoffAccepted,
                        payload: Some(accepted.as_union_value()),
                    },
                );
                builder.finish(msg, None);
            }

            ShardMsg::GhostUpdate(g) => {
                let pos = to_fb_vec3d(&g.position);
                let rot = to_fb_quatd(&g.rotation);
                let vel = to_fb_vec3d(&g.velocity);

                let ghost = fb::GhostUpdate::create(
                    &mut builder,
                    &fb::GhostUpdateArgs {
                        session_token: g.session_token.0,
                        position: Some(&pos),
                        rotation: Some(&rot),
                        velocity: Some(&vel),
                        tick: g.tick,
                    },
                );
                let msg = fb::ShardMessage::create(
                    &mut builder,
                    &fb::ShardMessageArgs {
                        payload_type: fb::ShardPayload::GhostUpdate,
                        payload: Some(ghost.as_union_value()),
                    },
                );
                builder.finish(msg, None);
            }

            ShardMsg::Heartbeat(hb) => {
                let heartbeat = fb::ShardHeartbeat::create(
                    &mut builder,
                    &fb::ShardHeartbeatArgs {
                        shard_id: hb.shard_id.0,
                        tick_ms: hb.tick_ms,
                        p99_tick_ms: hb.p99_tick_ms,
                        player_count: hb.player_count,
                        chunk_count: hb.chunk_count,
                    },
                );
                let msg = fb::ShardMessage::create(
                    &mut builder,
                    &fb::ShardMessageArgs {
                        payload_type: fb::ShardPayload::ShardHeartbeat,
                        payload: Some(heartbeat.as_union_value()),
                    },
                );
                builder.finish(msg, None);
            }

            ShardMsg::SplitDirective(s) => {
                let sectors = builder.create_vector(&s.sectors_to_split);
                let split = fb::SplitDirective::create(
                    &mut builder,
                    &fb::SplitDirectiveArgs {
                        target_shard_id: s.target_shard.0,
                        sectors_to_split: Some(sectors),
                        planet_seed: s.planet_seed,
                    },
                );
                let msg = fb::ShardMessage::create(
                    &mut builder,
                    &fb::ShardMessageArgs {
                        payload_type: fb::ShardPayload::SplitDirective,
                        payload: Some(split.as_union_value()),
                    },
                );
                builder.finish(msg, None);
            }

            ShardMsg::MergeDirective(m) => {
                let sectors = builder.create_vector(&m.sectors);
                let merge = fb::MergeDirective::create(
                    &mut builder,
                    &fb::MergeDirectiveArgs {
                        absorbing_shard_id: m.absorbing_shard.0,
                        merging_shard_id: m.merging_shard.0,
                        sectors: Some(sectors),
                    },
                );
                let msg = fb::ShardMessage::create(
                    &mut builder,
                    &fb::ShardMessageArgs {
                        payload_type: fb::ShardPayload::MergeDirective,
                        payload: Some(merge.as_union_value()),
                    },
                );
                builder.finish(msg, None);
            }

            ShardMsg::ShipPositionUpdate(s) => {
                let pos = to_fb_vec3d(&s.position);
                let vel = to_fb_vec3d(&s.velocity);
                let rot = to_fb_quatd(&s.rotation);
                let ang = to_fb_vec3d(&s.angular_velocity);
                let ap_offset = s.autopilot.as_ref().map(|ap| {
                    let ip = to_fb_vec3d(&ap.intercept_pos);
                    let av = to_fb_vec3d(&ap.target_arrival_vel);
                    fb::AutopilotSnapshot::create(
                        &mut builder,
                        &fb::AutopilotSnapshotArgs {
                            phase: ap.phase,
                            mode: ap.mode,
                            target_planet_index: ap.target_planet_index,
                            thrust_tier: ap.thrust_tier,
                            intercept_pos: Some(&ip),
                            target_arrival_vel: Some(&av),
                            braking_committed: ap.braking_committed,
                            eta_real_seconds: ap.eta_real_seconds,
                            target_orbit_altitude: ap.target_orbit_altitude,
                        },
                    )
                });
                let grav = to_fb_vec3d(&s.gravity_acceleration);
                let update = fb::ShipPositionUpdate::create(
                    &mut builder,
                    &fb::ShipPositionUpdateArgs {
                        ship_id: s.ship_id,
                        position: Some(&pos),
                        velocity: Some(&vel),
                        rotation: Some(&rot),
                        angular_velocity: Some(&ang),
                        autopilot: ap_offset,
                        in_atmosphere: s.in_atmosphere,
                        atmosphere_planet_index: s.atmosphere_planet_index,
                        gravity_acceleration: Some(&grav),
                        atmosphere_density: s.atmosphere_density,
                    },
                );
                let msg = fb::ShardMessage::create(
                    &mut builder,
                    &fb::ShardMessageArgs {
                        payload_type: fb::ShardPayload::ShipPositionUpdate,
                        payload: Some(update.as_union_value()),
                    },
                );
                builder.finish(msg, None);
            }

            ShardMsg::ShipControlInput(c) => {
                let thrust = to_fb_vec3d(&c.thrust);
                let torque = to_fb_vec3d(&c.torque);
                let input = fb::ShipControlInput::create(
                    &mut builder,
                    &fb::ShipControlInputArgs {
                        ship_id: c.ship_id,
                        thrust: Some(&thrust),
                        torque: Some(&torque),
                        braking: c.braking,
                        tick: c.tick,
                    },
                );
                let msg = fb::ShardMessage::create(
                    &mut builder,
                    &fb::ShardMessageArgs {
                        payload_type: fb::ShardPayload::ShipControlInput,
                        payload: Some(input.as_union_value()),
                    },
                );
                builder.finish(msg, None);
            }

            ShardMsg::CrossShardBlockEdits(e) => {
                let chunk_addr = builder.create_vector(&e.chunk_address);
                let edits = builder.create_vector(&e.edits);
                let block_edits = fb::CrossShardBlockEdits::create(
                    &mut builder,
                    &fb::CrossShardBlockEditsArgs {
                        chunk_address: Some(chunk_addr),
                        edits: Some(edits),
                        seq: e.seq,
                    },
                );
                let msg = fb::ShardMessage::create(
                    &mut builder,
                    &fb::ShardMessageArgs {
                        payload_type: fb::ShardPayload::CrossShardBlockEdits,
                        payload: Some(block_edits.as_union_value()),
                    },
                );
                builder.finish(msg, None);
            }

            ShardMsg::AutopilotCommand(a) => {
                let cmd = fb::AutopilotCommand::create(
                    &mut builder,
                    &fb::AutopilotCommandArgs {
                        ship_id: a.ship_id,
                        target_body_id: a.target_body_id,
                        speed_tier: a.speed_tier,
                        autopilot_mode: a.autopilot_mode,
                    },
                );
                let msg = fb::ShardMessage::create(
                    &mut builder,
                    &fb::ShardMessageArgs {
                        payload_type: fb::ShardPayload::AutopilotCommand,
                        payload: Some(cmd.as_union_value()),
                    },
                );
                builder.finish(msg, None);
            }

            ShardMsg::SystemSceneUpdate(s) => {
                let bodies: Vec<_> = s.bodies.iter().map(|b| {
                    let pos = to_fb_vec3d(&b.position);
                    fb::CelestialBodySnapshot::create(&mut builder, &fb::CelestialBodySnapshotArgs {
                        body_id: b.body_id, position: Some(&pos), radius: b.radius,
                        color_r: b.color[0], color_g: b.color[1], color_b: b.color[2],
                    })
                }).collect();
                let bodies_vec = builder.create_vector(&bodies);

                let ships: Vec<_> = s.ships.iter().map(|sh| {
                    let pos = to_fb_vec3d(&sh.position);
                    let rot = to_fb_quatd(&sh.rotation);
                    fb::ShipSnapshotEntry::create(&mut builder, &fb::ShipSnapshotEntryArgs {
                        ship_id: sh.ship_id, position: Some(&pos), rotation: Some(&rot),
                        is_own_ship: sh.is_own_ship,
                    })
                }).collect();
                let ships_vec = builder.create_vector(&ships);

                let sun_dir = to_fb_vec3d(&s.lighting.sun_direction);
                let lighting = fb::LightingInfoMsg::create(&mut builder, &fb::LightingInfoMsgArgs {
                    sun_direction: Some(&sun_dir),
                    sun_color_r: s.lighting.sun_color[0], sun_color_g: s.lighting.sun_color[1],
                    sun_color_b: s.lighting.sun_color[2],
                    sun_intensity: s.lighting.sun_intensity, ambient: s.lighting.ambient,
                });

                let update = fb::SystemSceneUpdate::create(&mut builder, &fb::SystemSceneUpdateArgs {
                    game_time: s.game_time, bodies: Some(bodies_vec),
                    ships: Some(ships_vec), lighting: Some(lighting),
                });
                let msg = fb::ShardMessage::create(&mut builder, &fb::ShardMessageArgs {
                    payload_type: fb::ShardPayload::SystemSceneUpdate,
                    payload: Some(update.as_union_value()),
                });
                builder.finish(msg, None);
            }

            ShardMsg::ShipNearbyInfo(info) => {
                let pos = to_fb_vec3d(&info.position);
                let rot = to_fb_quatd(&info.rotation);
                let vel = to_fb_vec3d(&info.velocity);
                let nearby = fb::ShipNearbyInfo::create(
                    &mut builder,
                    &fb::ShipNearbyInfoArgs {
                        ship_id: info.ship_id,
                        ship_shard_id: info.ship_shard_id.0,
                        position: Some(&pos),
                        rotation: Some(&rot),
                        velocity: Some(&vel),
                        game_time: info.game_time,
                    },
                );
                let msg = fb::ShardMessage::create(
                    &mut builder,
                    &fb::ShardMessageArgs {
                        payload_type: fb::ShardPayload::ShipNearbyInfo,
                        payload: Some(nearby.as_union_value()),
                    },
                );
                builder.finish(msg, None);
            }

            ShardMsg::WarpAutopilotCommand(w) => {
                let cmd = fb::WarpAutopilotCommand::create(
                    &mut builder,
                    &fb::WarpAutopilotCommandArgs {
                        ship_id: w.ship_id,
                        target_star_index: w.target_star_index,
                        galaxy_seed: w.galaxy_seed,
                    },
                );
                let msg = fb::ShardMessage::create(
                    &mut builder,
                    &fb::ShardMessageArgs {
                        payload_type: fb::ShardPayload::WarpAutopilotCommand,
                        payload: Some(cmd.as_union_value()),
                    },
                );
                builder.finish(msg, None);
            }

            ShardMsg::HostSwitch(h) => {
                let quic_addr = builder.create_string(&h.new_host_quic_addr);
                let tcp_addr = builder.create_string(&h.new_host_tcp_addr);
                let udp_addr = builder.create_string(&h.new_host_udp_addr);
                let hs = fb::HostSwitch::create(
                    &mut builder,
                    &fb::HostSwitchArgs {
                        ship_id: h.ship_id,
                        new_host_shard_id: h.new_host_shard_id.0,
                        new_host_quic_addr: Some(quic_addr),
                        new_host_tcp_addr: Some(tcp_addr),
                        new_host_udp_addr: Some(udp_addr),
                        new_host_shard_type: h.new_host_shard_type,
                        seed: h.seed,
                    },
                );
                let msg = fb::ShardMessage::create(
                    &mut builder,
                    &fb::ShardMessageArgs {
                        payload_type: fb::ShardPayload::HostSwitch,
                        payload: Some(hs.as_union_value()),
                    },
                );
                builder.finish(msg, None);
            }
            ShardMsg::ShipPropertiesUpdate(data) => {
                let spu = fb::ShipPropertiesUpdate::create(
                    &mut builder,
                    &fb::ShipPropertiesUpdateArgs {
                        ship_id: data.ship_id,
                        mass_kg: data.mass_kg,
                        max_thrust_forward_n: data.max_thrust_forward_n,
                        max_thrust_reverse_n: data.max_thrust_reverse_n,
                        max_torque_nm: data.max_torque_nm,
                        thrust_multiplier: data.thrust_multiplier,
                        dimensions_x: data.dimensions.0,
                        dimensions_y: data.dimensions.1,
                        dimensions_z: data.dimensions.2,
                    },
                );
                let msg = fb::ShardMessage::create(
                    &mut builder,
                    &fb::ShardMessageArgs {
                        payload_type: fb::ShardPayload::ShipPropertiesUpdate,
                        payload: Some(spu.as_union_value()),
                    },
                );
                builder.finish(msg, None);
            }
            ShardMsg::SignalBroadcast(data) => {
                let name = builder.create_string(&data.channel_name);
                let src_pos = to_fb_vec3d(&data.source_position);
                let sb = fb::SignalBroadcast::create(
                    &mut builder,
                    &fb::SignalBroadcastArgs {
                        source_shard_id: data.source_shard_id,
                        channel_name: Some(name),
                        value_type: data.value_type,
                        value_data: data.value_data,
                        scope: data.scope,
                        range_m: data.range_m,
                        source_position: Some(&src_pos),
                    },
                );
                let msg = fb::ShardMessage::create(
                    &mut builder,
                    &fb::ShardMessageArgs {
                        payload_type: fb::ShardPayload::SignalBroadcast,
                        payload: Some(sb.as_union_value()),
                    },
                );
                builder.finish(msg, None);
            }
            ShardMsg::SignalBroadcastBatch(data) => {
                let entries: Vec<_> = data.entries.iter().map(|e| {
                    let name = builder.create_string(&e.channel_name);
                    fb::SignalBroadcastEntry::create(
                        &mut builder,
                        &fb::SignalBroadcastEntryArgs {
                            channel_name: Some(name),
                            value_type: e.value_type,
                            value_data: e.value_data,
                            scope: e.scope,
                            range_m: e.range_m,
                            frequency: e.frequency,
                        },
                    )
                }).collect();
                let entries_vec = builder.create_vector(&entries);
                let src_pos = to_fb_vec3d(&data.source_position);
                let batch = fb::SignalBroadcastBatch::create(
                    &mut builder,
                    &fb::SignalBroadcastBatchArgs {
                        source_shard_id: data.source_shard_id,
                        source_position: Some(&src_pos),
                        entries: Some(entries_vec),
                    },
                );
                let msg = fb::ShardMessage::create(
                    &mut builder,
                    &fb::ShardMessageArgs {
                        payload_type: fb::ShardPayload::SignalBroadcastBatch,
                        payload: Some(batch.as_union_value()),
                    },
                );
                builder.finish(msg, None);
            }
            ShardMsg::VisibilityDirective(data) => {
                let entries: Vec<_> = data.visible_ships.iter().map(|e| {
                    let tcp = builder.create_string(&e.tcp_addr);
                    let udp = builder.create_string(&e.udp_addr);
                    fb::VisibleShipEntry::create(
                        &mut builder,
                        &fb::VisibleShipEntryArgs {
                            ship_id: e.ship_id,
                            shard_id: e.shard_id.0,
                            tcp_addr: Some(tcp),
                            udp_addr: Some(udp),
                            distance: e.distance,
                        },
                    )
                }).collect();
                let entries_vec = builder.create_vector(&entries);
                let dir = fb::VisibilityDirective::create(
                    &mut builder,
                    &fb::VisibilityDirectiveArgs {
                        ship_id: data.ship_id,
                        visible_ships: Some(entries_vec),
                    },
                );
                let msg = fb::ShardMessage::create(
                    &mut builder,
                    &fb::ShardMessageArgs {
                        payload_type: fb::ShardPayload::VisibilityDirective,
                        payload: Some(dir.as_union_value()),
                    },
                );
                builder.finish(msg, None);
            }
            ShardMsg::SystemEntitiesUpdate(data) => {
                let observer = to_fb_vec3d(&data.observer_position);
                let entities = crate::client_message::encode_observable_entities(
                    &mut builder,
                    &data.entities,
                );
                let (target_ship_id, target_planet_index) = match data.target {
                    AoiTarget::Ship(id) => (id, u32::MAX),
                    AoiTarget::Planet(idx) => (0, idx),
                };
                let upd = fb::SystemEntitiesUpdate::create(
                    &mut builder,
                    &fb::SystemEntitiesUpdateArgs {
                        target_ship_id,
                        target_planet_index,
                        observer_position: Some(&observer),
                        entities,
                        tick: data.tick,
                    },
                );
                let msg = fb::ShardMessage::create(
                    &mut builder,
                    &fb::ShardMessageArgs {
                        payload_type: fb::ShardPayload::SystemEntitiesUpdate,
                        payload: Some(upd.as_union_value()),
                    },
                );
                builder.finish(msg, None);
            }
            ShardMsg::PlanetPlayerDigest(data) => {
                let entries: Vec<_> = data
                    .entries
                    .iter()
                    .map(|e| {
                        let name = builder.create_string(&e.player_name);
                        let pos = to_fb_vec3d(&e.position);
                        let rot = to_fb_quatd(&e.rotation);
                        fb::PlanetPlayerDigestEntry::create(
                            &mut builder,
                            &fb::PlanetPlayerDigestEntryArgs {
                                session_token: e.session_token.0,
                                player_name: Some(name),
                                position: Some(&pos),
                                rotation: Some(&rot),
                                planet_index: e.planet_index,
                            },
                        )
                    })
                    .collect();
                let entries_vec = builder.create_vector(&entries);
                let digest = fb::PlanetPlayerDigest::create(
                    &mut builder,
                    &fb::PlanetPlayerDigestArgs {
                        planet_shard_id: data.planet_shard.0,
                        planet_seed: data.planet_seed,
                        planet_index: data.planet_index,
                        entries: Some(entries_vec),
                        tick: data.tick,
                    },
                );
                let msg = fb::ShardMessage::create(
                    &mut builder,
                    &fb::ShardMessageArgs {
                        payload_type: fb::ShardPayload::PlanetPlayerDigest,
                        payload: Some(digest.as_union_value()),
                    },
                );
                builder.finish(msg, None);
            }
            ShardMsg::ShipColliderSync(data) => {
                let chunks: Vec<_> = data.chunks.iter().map(|chunk| {
                    let shapes: Vec<fb::ColliderShape> = chunk.shapes.iter().map(|(c, h)| {
                        fb::ColliderShape::new(c.x, c.y, c.z, h.x, h.y, h.z)
                    }).collect();
                    let shapes_vec = builder.create_vector(&shapes);
                    fb::ChunkColliderDataMsg::create(
                        &mut builder,
                        &fb::ChunkColliderDataMsgArgs {
                            cx: chunk.chunk_key.x,
                            cy: chunk.chunk_key.y,
                            cz: chunk.chunk_key.z,
                            shapes: Some(shapes_vec),
                        },
                    )
                }).collect();
                let chunks_vec = builder.create_vector(&chunks);
                let sync = fb::ShipColliderSync::create(
                    &mut builder,
                    &fb::ShipColliderSyncArgs {
                        ship_id: data.ship_id,
                        chunks: Some(chunks_vec),
                        hull_min_x: data.hull_min.x,
                        hull_min_y: data.hull_min.y,
                        hull_min_z: data.hull_min.z,
                        hull_max_x: data.hull_max.x,
                        hull_max_y: data.hull_max.y,
                        hull_max_z: data.hull_max.z,
                    },
                );
                let msg = fb::ShardMessage::create(
                    &mut builder,
                    &fb::ShardMessageArgs {
                        payload_type: fb::ShardPayload::ShipColliderSync,
                        payload: Some(sync.as_union_value()),
                    },
                );
                builder.finish(msg, None);
            }
        }

        let result = builder.finished_data().to_vec();
        crate::builder_pool::release(builder);
        result
    }

    /// Deserialize a FlatBuffer byte slice into a ShardMsg.
    pub fn deserialize(buf: &[u8]) -> Result<Self, MessageError> {
        let msg = flatbuffers::root::<fb::ShardMessage>(buf)
            .map_err(|e| MessageError::InvalidBuffer(e.to_string()))?;

        match msg.payload_type() {
            fb::ShardPayload::PlayerHandoff => {
                let h = msg
                    .payload_as_player_handoff()
                    .ok_or(MessageError::MissingField("PlayerHandoff payload"))?;

                let pos = h
                    .position()
                    .ok_or(MessageError::MissingField("position"))?;
                let vel = h
                    .velocity()
                    .ok_or(MessageError::MissingField("velocity"))?;
                let rot = h
                    .rotation()
                    .ok_or(MessageError::MissingField("rotation"))?;
                let fwd = h.forward().ok_or(MessageError::MissingField("forward"))?;

                Ok(ShardMsg::PlayerHandoff(handoff::PlayerHandoff {
                    session_token: SessionToken(h.session_token()),
                    player_name: h
                        .player_name()
                        .ok_or(MessageError::MissingField("player_name"))?
                        .to_string(),
                    position: from_fb_vec3d(pos),
                    velocity: from_fb_vec3d(vel),
                    rotation: from_fb_quatd(rot),
                    forward: from_fb_vec3d(fwd),
                    fly_mode: h.fly_mode(),
                    speed_tier: h.speed_tier(),
                    grounded: h.grounded(),
                    health: h.health(),
                    shield: h.shield(),
                    source_shard: ShardId(h.source_shard_id()),
                    source_tick: h.source_tick(),
                    target_star_index: {
                        let idx = h.target_star_index();
                        if idx == 0xFFFFFFFF { None } else { Some(idx) }
                    },
                    galaxy_context: h.galaxy_context().map(|ctx| {
                        let sp = ctx.star_position().unwrap();
                        handoff::GalaxyHandoffContext {
                            galaxy_seed: ctx.galaxy_seed(),
                            star_index: ctx.star_index(),
                            star_position: from_fb_vec3d(sp),
                        }
                    }),
                    target_planet_seed: {
                        let v = h.target_planet_seed();
                        if v == u64::MAX { None } else { Some(v) }
                    },
                    target_planet_index: {
                        let v = h.target_planet_index();
                        if v == u32::MAX { None } else { Some(v) }
                    },
                    target_ship_id: {
                        let v = h.target_ship_id();
                        if v == u64::MAX { None } else { Some(v) }
                    },
                    target_ship_shard_id: {
                        let v = h.target_ship_shard_id();
                        if v == u64::MAX { None } else { Some(ShardId(v)) }
                    },
                    ship_system_position: h.ship_system_position().map(|p| from_fb_vec3d(p)),
                    ship_rotation: h.ship_rotation().map(|r| from_fb_quatd(r)),
                    game_time: h.game_time(),
                    warp_target_star_index: {
                        let idx = h.warp_target_star_index();
                        if idx == 0xFFFFFFFF { None } else { Some(idx) }
                    },
                    warp_velocity_gu: h.warp_velocity().map(|v| from_fb_vec3d(v)),
                    target_system_eva: h.target_system_eva(),
                }))
            }

            fb::ShardPayload::HandoffAccepted => {
                let a = msg
                    .payload_as_handoff_accepted()
                    .ok_or(MessageError::MissingField("HandoffAccepted payload"))?;

                Ok(ShardMsg::HandoffAccepted(handoff::HandoffAccepted {
                    session_token: SessionToken(a.session_token()),
                    target_shard: ShardId(a.target_shard_id()),
                }))
            }

            fb::ShardPayload::GhostUpdate => {
                let g = msg
                    .payload_as_ghost_update()
                    .ok_or(MessageError::MissingField("GhostUpdate payload"))?;

                let pos = g
                    .position()
                    .ok_or(MessageError::MissingField("position"))?;
                let rot = g
                    .rotation()
                    .ok_or(MessageError::MissingField("rotation"))?;
                let vel = g
                    .velocity()
                    .ok_or(MessageError::MissingField("velocity"))?;

                Ok(ShardMsg::GhostUpdate(handoff::GhostUpdate {
                    session_token: SessionToken(g.session_token()),
                    position: from_fb_vec3d(pos),
                    rotation: from_fb_quatd(rot),
                    velocity: from_fb_vec3d(vel),
                    tick: g.tick(),
                }))
            }

            fb::ShardPayload::ShardHeartbeat => {
                let hb = msg
                    .payload_as_shard_heartbeat()
                    .ok_or(MessageError::MissingField("ShardHeartbeat payload"))?;

                Ok(ShardMsg::Heartbeat(shard_types::ShardHeartbeat {
                    shard_id: ShardId(hb.shard_id()),
                    tick_ms: hb.tick_ms(),
                    p99_tick_ms: hb.p99_tick_ms(),
                    player_count: hb.player_count(),
                    chunk_count: hb.chunk_count(),
                }))
            }

            fb::ShardPayload::SplitDirective => {
                let s = msg
                    .payload_as_split_directive()
                    .ok_or(MessageError::MissingField("SplitDirective payload"))?;

                Ok(ShardMsg::SplitDirective(SplitDirective {
                    target_shard: ShardId(s.target_shard_id()),
                    sectors_to_split: s
                        .sectors_to_split()
                        .map(|v| v.iter().collect())
                        .unwrap_or_default(),
                    planet_seed: s.planet_seed(),
                }))
            }

            fb::ShardPayload::MergeDirective => {
                let m = msg
                    .payload_as_merge_directive()
                    .ok_or(MessageError::MissingField("MergeDirective payload"))?;

                Ok(ShardMsg::MergeDirective(MergeDirective {
                    absorbing_shard: ShardId(m.absorbing_shard_id()),
                    merging_shard: ShardId(m.merging_shard_id()),
                    sectors: m
                        .sectors()
                        .map(|v| v.iter().collect())
                        .unwrap_or_default(),
                }))
            }

            fb::ShardPayload::ShipPositionUpdate => {
                let s = msg
                    .payload_as_ship_position_update()
                    .ok_or(MessageError::MissingField("ShipPositionUpdate payload"))?;
                let pos = s.position().ok_or(MessageError::MissingField("position"))?;
                let vel = s.velocity().ok_or(MessageError::MissingField("velocity"))?;
                let rot = s.rotation().ok_or(MessageError::MissingField("rotation"))?;
                let ang = s.angular_velocity().ok_or(MessageError::MissingField("angular_velocity"))?;

                let autopilot = s.autopilot().map(|ap| {
                    let ip = ap.intercept_pos().map(|v| from_fb_vec3d(v)).unwrap_or(DVec3::ZERO);
                    let av = ap.target_arrival_vel().map(|v| from_fb_vec3d(v)).unwrap_or(DVec3::ZERO);
                    AutopilotSnapshotData {
                        phase: ap.phase(),
                        mode: ap.mode(),
                        target_planet_index: ap.target_planet_index(),
                        thrust_tier: ap.thrust_tier(),
                        intercept_pos: ip,
                        target_arrival_vel: av,
                        braking_committed: ap.braking_committed(),
                        eta_real_seconds: ap.eta_real_seconds(),
                        target_orbit_altitude: ap.target_orbit_altitude(),
                    }
                });

                Ok(ShardMsg::ShipPositionUpdate(ShipPositionUpdate {
                    ship_id: s.ship_id(),
                    position: from_fb_vec3d(pos),
                    velocity: from_fb_vec3d(vel),
                    rotation: from_fb_quatd(rot),
                    angular_velocity: from_fb_vec3d(ang),
                    autopilot,
                    in_atmosphere: s.in_atmosphere(),
                    atmosphere_planet_index: s.atmosphere_planet_index(),
                    gravity_acceleration: s.gravity_acceleration().map(|v| from_fb_vec3d(v)).unwrap_or(DVec3::ZERO),
                    atmosphere_density: s.atmosphere_density(),
                }))
            }

            fb::ShardPayload::ShipControlInput => {
                let c = msg
                    .payload_as_ship_control_input()
                    .ok_or(MessageError::MissingField("ShipControlInput payload"))?;
                let thrust = c.thrust().ok_or(MessageError::MissingField("thrust"))?;
                let torque = c.torque().ok_or(MessageError::MissingField("torque"))?;

                Ok(ShardMsg::ShipControlInput(ShipControlInput {
                    ship_id: c.ship_id(),
                    thrust: from_fb_vec3d(thrust),
                    torque: from_fb_vec3d(torque),
                    braking: c.braking(),
                    tick: c.tick(),
                }))
            }

            fb::ShardPayload::CrossShardBlockEdits => {
                let e = msg
                    .payload_as_cross_shard_block_edits()
                    .ok_or(MessageError::MissingField("CrossShardBlockEdits payload"))?;

                Ok(ShardMsg::CrossShardBlockEdits(CrossShardBlockEdits {
                    chunk_address: e.chunk_address().map(|v| v.iter().collect()).unwrap_or_default(),
                    edits: e.edits().map(|v| v.iter().collect()).unwrap_or_default(),
                    seq: e.seq(),
                }))
            }

            fb::ShardPayload::SystemSceneUpdate => {
                let s = msg.payload_as_system_scene_update()
                    .ok_or(MessageError::MissingField("SystemSceneUpdate payload"))?;

                let bodies = s.bodies().map(|v| v.iter().map(|b| {
                    let pos = b.position().unwrap();
                    CelestialBodySnapshotData {
                        body_id: b.body_id(), position: from_fb_vec3d(pos),
                        radius: b.radius(), color: [b.color_r(), b.color_g(), b.color_b()],
                    }
                }).collect()).unwrap_or_default();

                let ships = s.ships().map(|v| v.iter().map(|sh| {
                    let pos = sh.position().unwrap();
                    let rot = sh.rotation().unwrap();
                    ShipSnapshotEntryData {
                        ship_id: sh.ship_id(), position: from_fb_vec3d(pos),
                        rotation: from_fb_quatd(rot), is_own_ship: sh.is_own_ship(),
                    }
                }).collect()).unwrap_or_default();

                let lighting_fb = s.lighting().ok_or(MessageError::MissingField("lighting"))?;
                let sun_dir = lighting_fb.sun_direction().ok_or(MessageError::MissingField("sun_direction"))?;

                Ok(ShardMsg::SystemSceneUpdate(SystemSceneUpdateData {
                    game_time: s.game_time(),
                    bodies,
                    ships,
                    lighting: LightingInfoData {
                        sun_direction: from_fb_vec3d(sun_dir),
                        sun_color: [lighting_fb.sun_color_r(), lighting_fb.sun_color_g(), lighting_fb.sun_color_b()],
                        sun_intensity: lighting_fb.sun_intensity(),
                        ambient: lighting_fb.ambient(),
                    },
                }))
            }

            fb::ShardPayload::AutopilotCommand => {
                let a = msg
                    .payload_as_autopilot_command()
                    .ok_or(MessageError::MissingField("AutopilotCommand payload"))?;

                Ok(ShardMsg::AutopilotCommand(AutopilotCommandData {
                    ship_id: a.ship_id(),
                    target_body_id: a.target_body_id(),
                    speed_tier: a.speed_tier(),
                    autopilot_mode: a.autopilot_mode(),
                }))
            }

            fb::ShardPayload::ShipNearbyInfo => {
                let info = msg
                    .payload_as_ship_nearby_info()
                    .ok_or(MessageError::MissingField("ShipNearbyInfo payload"))?;
                let pos = info.position().ok_or(MessageError::MissingField("position"))?;
                let rot = info.rotation().ok_or(MessageError::MissingField("rotation"))?;
                let vel = info.velocity().ok_or(MessageError::MissingField("velocity"))?;

                Ok(ShardMsg::ShipNearbyInfo(ShipNearbyInfoData {
                    ship_id: info.ship_id(),
                    ship_shard_id: ShardId(info.ship_shard_id()),
                    position: from_fb_vec3d(pos),
                    rotation: from_fb_quatd(rot),
                    velocity: from_fb_vec3d(vel),
                    game_time: info.game_time(),
                }))
            }

            fb::ShardPayload::WarpAutopilotCommand => {
                let w = msg
                    .payload_as_warp_autopilot_command()
                    .ok_or(MessageError::MissingField("WarpAutopilotCommand payload"))?;

                Ok(ShardMsg::WarpAutopilotCommand(WarpAutopilotCommandData {
                    ship_id: w.ship_id(),
                    target_star_index: w.target_star_index(),
                    galaxy_seed: w.galaxy_seed(),
                }))
            }

            fb::ShardPayload::HostSwitch => {
                let h = msg
                    .payload_as_host_switch()
                    .ok_or(MessageError::MissingField("HostSwitch payload"))?;

                Ok(ShardMsg::HostSwitch(HostSwitchData {
                    ship_id: h.ship_id(),
                    new_host_shard_id: ShardId(h.new_host_shard_id()),
                    new_host_quic_addr: h.new_host_quic_addr().unwrap_or("").to_string(),
                    new_host_tcp_addr: h.new_host_tcp_addr().unwrap_or("").to_string(),
                    new_host_udp_addr: h.new_host_udp_addr().unwrap_or("").to_string(),
                    new_host_shard_type: h.new_host_shard_type(),
                    seed: h.seed(),
                }))
            }

            fb::ShardPayload::ShipPropertiesUpdate => {
                let spu = msg
                    .payload_as_ship_properties_update()
                    .ok_or(MessageError::MissingField("ShipPropertiesUpdate payload"))?;

                Ok(ShardMsg::ShipPropertiesUpdate(ShipPropertiesUpdateData {
                    ship_id: spu.ship_id(),
                    mass_kg: spu.mass_kg(),
                    max_thrust_forward_n: spu.max_thrust_forward_n(),
                    max_thrust_reverse_n: spu.max_thrust_reverse_n(),
                    max_torque_nm: spu.max_torque_nm(),
                    thrust_multiplier: spu.thrust_multiplier(),
                    dimensions: (spu.dimensions_x(), spu.dimensions_y(), spu.dimensions_z()),
                }))
            }

            fb::ShardPayload::SignalBroadcast => {
                let sb = msg
                    .payload_as_signal_broadcast()
                    .ok_or(MessageError::MissingField("SignalBroadcast payload"))?;
                let src_pos = sb.source_position()
                    .ok_or(MessageError::MissingField("source_position"))?;

                Ok(ShardMsg::SignalBroadcast(SignalBroadcastData {
                    source_shard_id: sb.source_shard_id(),
                    channel_name: sb.channel_name().unwrap_or("").to_string(),
                    value_type: sb.value_type(),
                    value_data: sb.value_data(),
                    scope: sb.scope(),
                    range_m: sb.range_m(),
                    source_position: from_fb_vec3d(src_pos),
                }))
            }

            fb::ShardPayload::SignalBroadcastBatch => {
                let batch = msg
                    .payload_as_signal_broadcast_batch()
                    .ok_or(MessageError::MissingField("SignalBroadcastBatch payload"))?;
                let src_pos = batch.source_position()
                    .ok_or(MessageError::MissingField("source_position"))?;
                let entries = batch.entries().map(|v| {
                    v.iter().map(|e| SignalBroadcastEntry {
                        channel_name: e.channel_name().unwrap_or("").to_string(),
                        value_type: e.value_type(),
                        value_data: e.value_data(),
                        scope: e.scope(),
                        range_m: e.range_m(),
                        frequency: e.frequency(),
                    }).collect()
                }).unwrap_or_default();

                Ok(ShardMsg::SignalBroadcastBatch(SignalBroadcastBatchData {
                    source_shard_id: batch.source_shard_id(),
                    source_position: from_fb_vec3d(src_pos),
                    entries,
                }))
            }

            fb::ShardPayload::VisibilityDirective => {
                let dir = msg
                    .payload_as_visibility_directive()
                    .ok_or(MessageError::MissingField("VisibilityDirective payload"))?;
                let entries: Vec<VisibleShipEntryData> = dir
                    .visible_ships()
                    .map(|v| {
                        v.iter()
                            .map(|e| VisibleShipEntryData {
                                ship_id: e.ship_id(),
                                shard_id: ShardId(e.shard_id()),
                                tcp_addr: e.tcp_addr().unwrap_or("").to_string(),
                                udp_addr: e.udp_addr().unwrap_or("").to_string(),
                                distance: e.distance(),
                            })
                            .collect()
                    })
                    .unwrap_or_default();
                Ok(ShardMsg::VisibilityDirective(VisibilityDirectiveData {
                    ship_id: dir.ship_id(),
                    visible_ships: entries,
                }))
            }

            fb::ShardPayload::ShipColliderSync => {
                let sync = msg
                    .payload_as_ship_collider_sync()
                    .ok_or(MessageError::MissingField("ShipColliderSync payload"))?;
                let chunks: Vec<ChunkColliderData> = sync
                    .chunks()
                    .map(|v| {
                        v.iter()
                            .map(|c| {
                                let shapes: Vec<(glam::Vec3, glam::Vec3)> = c
                                    .shapes()
                                    .map(|s| {
                                        s.iter()
                                            .map(|shape| {
                                                (
                                                    glam::Vec3::new(shape.cx(), shape.cy(), shape.cz()),
                                                    glam::Vec3::new(shape.hx(), shape.hy(), shape.hz()),
                                                )
                                            })
                                            .collect()
                                    })
                                    .unwrap_or_default();
                                ChunkColliderData {
                                    chunk_key: glam::IVec3::new(c.cx(), c.cy(), c.cz()),
                                    shapes,
                                }
                            })
                            .collect()
                    })
                    .unwrap_or_default();
                Ok(ShardMsg::ShipColliderSync(ShipColliderSyncData {
                    ship_id: sync.ship_id(),
                    chunks,
                    hull_min: glam::Vec3::new(sync.hull_min_x(), sync.hull_min_y(), sync.hull_min_z()),
                    hull_max: glam::Vec3::new(sync.hull_max_x(), sync.hull_max_y(), sync.hull_max_z()),
                }))
            }

            fb::ShardPayload::SystemEntitiesUpdate => {
                let upd = msg
                    .payload_as_system_entities_update()
                    .ok_or(MessageError::MissingField("SystemEntitiesUpdate payload"))?;
                let observer = upd
                    .observer_position()
                    .map(from_fb_vec3d)
                    .unwrap_or(DVec3::ZERO);
                let entities =
                    crate::client_message::decode_observable_entities(upd.entities());
                let target = if upd.target_ship_id() != 0 {
                    AoiTarget::Ship(upd.target_ship_id())
                } else {
                    AoiTarget::Planet(upd.target_planet_index())
                };
                Ok(ShardMsg::SystemEntitiesUpdate(SystemEntitiesUpdateData {
                    target,
                    observer_position: observer,
                    entities,
                    tick: upd.tick(),
                }))
            }

            fb::ShardPayload::PlanetPlayerDigest => {
                let digest = msg
                    .payload_as_planet_player_digest()
                    .ok_or(MessageError::MissingField("PlanetPlayerDigest payload"))?;
                let entries: Vec<PlanetPlayerDigestEntry> = digest
                    .entries()
                    .map(|v| {
                        v.iter()
                            .map(|e| {
                                let pos = e
                                    .position()
                                    .map(from_fb_vec3d)
                                    .unwrap_or(DVec3::ZERO);
                                let rot = e
                                    .rotation()
                                    .map(from_fb_quatd)
                                    .unwrap_or(DQuat::IDENTITY);
                                PlanetPlayerDigestEntry {
                                    session_token: SessionToken(e.session_token()),
                                    player_name: e
                                        .player_name()
                                        .unwrap_or("")
                                        .to_string(),
                                    position: pos,
                                    rotation: rot,
                                    planet_index: e.planet_index(),
                                }
                            })
                            .collect()
                    })
                    .unwrap_or_default();
                Ok(ShardMsg::PlanetPlayerDigest(PlanetPlayerDigestData {
                    planet_shard: ShardId(digest.planet_shard_id()),
                    planet_seed: digest.planet_seed(),
                    planet_index: digest.planet_index(),
                    entries,
                    tick: digest.tick(),
                }))
            }

            fb::ShardPayload::NONE => {
                Err(MessageError::UnknownPayload(0))
            }

            other => Err(MessageError::UnknownPayload(other.0)),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use glam::{DQuat, DVec3};

    fn make_test_handoff() -> handoff::PlayerHandoff {
        handoff::PlayerHandoff {
            session_token: SessionToken(12345),
            player_name: "TestPlayer".to_string(),
            position: DVec3::new(1.0, 150000.0, 3.0),
            velocity: DVec3::new(0.0, -9.8, 0.0),
            rotation: DQuat::from_xyzw(0.0, 0.707, 0.0, 0.707),
            forward: DVec3::new(0.0, 0.0, -1.0),
            fly_mode: true,
            speed_tier: 2,
            grounded: false,
            health: 85.5,
            shield: 42.0,
            source_shard: ShardId(99),
            source_tick: 10000,
            target_star_index: None,
            galaxy_context: None,
            target_planet_seed: None,
            target_planet_index: None,
            target_ship_id: None,
            target_ship_shard_id: None,
            ship_system_position: None,
            ship_rotation: None,
            game_time: 0.0,
            warp_target_star_index: None,
            warp_velocity_gu: None,
            target_system_eva: false,
        }
    }

    #[test]
    fn roundtrip_player_handoff() {
        let msg = ShardMsg::PlayerHandoff(make_test_handoff());
        let bytes = msg.serialize();
        let decoded = ShardMsg::deserialize(&bytes).unwrap();

        if let ShardMsg::PlayerHandoff(h) = decoded {
            assert_eq!(h.session_token, SessionToken(12345));
            assert_eq!(h.player_name, "TestPlayer");
            assert!((h.position.y - 150000.0).abs() < 1e-10);
            assert!((h.velocity.y - (-9.8)).abs() < 1e-10);
            assert!(h.fly_mode);
            assert_eq!(h.speed_tier, 2);
            assert!(!h.grounded);
            assert!((h.health - 85.5).abs() < 1e-5);
            assert!((h.shield - 42.0).abs() < 1e-5);
            assert_eq!(h.source_shard, ShardId(99));
            assert_eq!(h.source_tick, 10000);
        } else {
            panic!("expected PlayerHandoff");
        }
    }

    #[test]
    fn roundtrip_handoff_accepted() {
        let msg = ShardMsg::HandoffAccepted(handoff::HandoffAccepted {
            session_token: SessionToken(111),
            target_shard: ShardId(222),
        });
        let bytes = msg.serialize();
        let decoded = ShardMsg::deserialize(&bytes).unwrap();

        if let ShardMsg::HandoffAccepted(a) = decoded {
            assert_eq!(a.session_token, SessionToken(111));
            assert_eq!(a.target_shard, ShardId(222));
        } else {
            panic!("expected HandoffAccepted");
        }
    }

    #[test]
    fn roundtrip_ghost_update() {
        let msg = ShardMsg::GhostUpdate(handoff::GhostUpdate {
            session_token: SessionToken(333),
            position: DVec3::new(10.0, 20.0, 30.0),
            rotation: DQuat::IDENTITY,
            velocity: DVec3::new(1.0, 2.0, 3.0),
            tick: 5000,
        });
        let bytes = msg.serialize();
        let decoded = ShardMsg::deserialize(&bytes).unwrap();

        if let ShardMsg::GhostUpdate(g) = decoded {
            assert_eq!(g.session_token, SessionToken(333));
            assert!((g.position.x - 10.0).abs() < 1e-10);
            assert_eq!(g.tick, 5000);
        } else {
            panic!("expected GhostUpdate");
        }
    }

    #[test]
    fn roundtrip_heartbeat() {
        let msg = ShardMsg::Heartbeat(shard_types::ShardHeartbeat {
            shard_id: ShardId(1),
            tick_ms: 48.5,
            p99_tick_ms: 49.9,
            player_count: 50,
            chunk_count: 3000,
        });
        let bytes = msg.serialize();
        let decoded = ShardMsg::deserialize(&bytes).unwrap();

        if let ShardMsg::Heartbeat(hb) = decoded {
            assert_eq!(hb.shard_id, ShardId(1));
            assert!((hb.tick_ms - 48.5).abs() < 1e-5);
            assert_eq!(hb.player_count, 50);
            assert_eq!(hb.chunk_count, 3000);
        } else {
            panic!("expected Heartbeat");
        }
    }

    #[test]
    fn roundtrip_split_directive() {
        let msg = ShardMsg::SplitDirective(SplitDirective {
            target_shard: ShardId(10),
            sectors_to_split: vec![3, 4, 5],
            planet_seed: 42,
        });
        let bytes = msg.serialize();
        let decoded = ShardMsg::deserialize(&bytes).unwrap();

        if let ShardMsg::SplitDirective(s) = decoded {
            assert_eq!(s.target_shard, ShardId(10));
            assert_eq!(s.sectors_to_split, vec![3, 4, 5]);
            assert_eq!(s.planet_seed, 42);
        } else {
            panic!("expected SplitDirective");
        }
    }

    #[test]
    fn roundtrip_merge_directive() {
        let msg = ShardMsg::MergeDirective(MergeDirective {
            absorbing_shard: ShardId(1),
            merging_shard: ShardId(2),
            sectors: vec![0, 1, 2],
        });
        let bytes = msg.serialize();
        let decoded = ShardMsg::deserialize(&bytes).unwrap();

        if let ShardMsg::MergeDirective(m) = decoded {
            assert_eq!(m.absorbing_shard, ShardId(1));
            assert_eq!(m.merging_shard, ShardId(2));
            assert_eq!(m.sectors, vec![0, 1, 2]);
        } else {
            panic!("expected MergeDirective");
        }
    }

    #[test]
    fn roundtrip_ship_position_update() {
        let msg = ShardMsg::ShipPositionUpdate(ShipPositionUpdate {
            ship_id: 42,
            position: DVec3::new(1000.0, 2000.0, 3000.0),
            velocity: DVec3::new(10.0, 0.0, -5.0),
            rotation: DQuat::from_xyzw(0.0, 0.0, 0.707, 0.707),
            angular_velocity: DVec3::new(0.0, 0.1, 0.0),
            autopilot: None,
            in_atmosphere: true,
            atmosphere_planet_index: 2,
            gravity_acceleration: DVec3::new(0.0, -9.8, 0.0),
            atmosphere_density: 0.5,
        });
        let bytes = msg.serialize();
        let decoded = ShardMsg::deserialize(&bytes).unwrap();

        if let ShardMsg::ShipPositionUpdate(s) = decoded {
            assert_eq!(s.ship_id, 42);
            assert!((s.position.x - 1000.0).abs() < 1e-10);
            assert!((s.velocity.z - (-5.0)).abs() < 1e-10);
            assert!((s.angular_velocity.y - 0.1).abs() < 1e-10);
            assert!(s.autopilot.is_none());
            assert!(s.in_atmosphere);
            assert_eq!(s.atmosphere_planet_index, 2);
            assert!((s.gravity_acceleration.y - (-9.8)).abs() < 1e-10);
            assert!((s.atmosphere_density - 0.5).abs() < 1e-10);
        } else {
            panic!("expected ShipPositionUpdate");
        }
    }

    #[test]
    fn roundtrip_ship_position_update_with_autopilot() {
        let msg = ShardMsg::ShipPositionUpdate(ShipPositionUpdate {
            ship_id: 99,
            position: DVec3::new(1e9, 2e9, 3e9),
            velocity: DVec3::new(5000.0, 0.0, -3000.0),
            rotation: DQuat::IDENTITY,
            angular_velocity: DVec3::ZERO,
            in_atmosphere: false,
            atmosphere_planet_index: -1,
            gravity_acceleration: DVec3::ZERO,
            atmosphere_density: 0.0,
            autopilot: Some(AutopilotSnapshotData {
                phase: 2, // Brake
                mode: 1,  // OrbitInsertion
                target_planet_index: 3,
                thrust_tier: 2,
                intercept_pos: DVec3::new(4e9, 5e9, 6e9),
                target_arrival_vel: DVec3::new(100.0, 200.0, 300.0),
                braking_committed: true,
                eta_real_seconds: 120.5,
                target_orbit_altitude: 50000.0,
            }),
        });
        let bytes = msg.serialize();
        let decoded = ShardMsg::deserialize(&bytes).unwrap();

        if let ShardMsg::ShipPositionUpdate(s) = decoded {
            assert_eq!(s.ship_id, 99);
            let ap = s.autopilot.unwrap();
            assert_eq!(ap.phase, 2);
            assert_eq!(ap.mode, 1);
            assert_eq!(ap.target_planet_index, 3);
            assert_eq!(ap.thrust_tier, 2);
            assert!((ap.intercept_pos.x - 4e9).abs() < 1e-3);
            assert!((ap.target_arrival_vel.y - 200.0).abs() < 1e-10);
            assert!(ap.braking_committed);
            assert!((ap.eta_real_seconds - 120.5).abs() < 1e-10);
            assert!((ap.target_orbit_altitude - 50000.0).abs() < 1e-10);
        } else {
            panic!("expected ShipPositionUpdate");
        }
    }

    #[test]
    fn roundtrip_ship_control_input() {
        let msg = ShardMsg::ShipControlInput(ShipControlInput {
            ship_id: 99,
            thrust: DVec3::new(0.0, 0.0, 100.0),
            torque: DVec3::new(0.0, 0.5, 0.0),
            braking: true,
            tick: 7777,
        });
        let bytes = msg.serialize();
        let decoded = ShardMsg::deserialize(&bytes).unwrap();

        if let ShardMsg::ShipControlInput(c) = decoded {
            assert_eq!(c.ship_id, 99);
            assert!((c.thrust.z - 100.0).abs() < 1e-10);
            assert!((c.torque.y - 0.5).abs() < 1e-10);
            assert!(c.braking);
            assert_eq!(c.tick, 7777);
        } else {
            panic!("expected ShipControlInput");
        }
    }

    #[test]
    fn deserialize_garbage_fails() {
        let result = ShardMsg::deserialize(&[0xFF, 0x00, 0x01, 0x02]);
        assert!(result.is_err());
    }
}
