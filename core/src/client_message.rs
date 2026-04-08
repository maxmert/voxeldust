use flatbuffers::FlatBufferBuilder;
use glam::{DQuat, DVec3};

use crate::handoff::{self, ShardRedirect};
use crate::protocol_generated as fb;
use crate::shard_message::{AutopilotSnapshotData, MessageError};
use crate::signal::converter::{SignalCondition, SignalExpression};
use crate::signal::types::SignalProperty;
use crate::shard_types::{SessionToken, ShardId};

fn serialize_condition(cond: &SignalCondition) -> (u8, f32) {
    match cond {
        SignalCondition::GreaterThan(v) => (0, *v),
        SignalCondition::LessThan(v) => (1, *v),
        SignalCondition::Equals(v) => (2, *v),
        SignalCondition::NotEquals(v) => (3, *v),
        SignalCondition::Changed => (4, 0.0),
        SignalCondition::Always => (5, 0.0),
    }
}

fn deserialize_condition(ctype: u8, val: f32) -> SignalCondition {
    match ctype {
        0 => SignalCondition::GreaterThan(val),
        1 => SignalCondition::LessThan(val),
        2 => SignalCondition::Equals(val),
        3 => SignalCondition::NotEquals(val),
        4 => SignalCondition::Changed,
        5 => SignalCondition::Always,
        _ => SignalCondition::Always,
    }
}

fn serialize_expression(expr: &SignalExpression) -> (u8, f32, f32) {
    match expr {
        SignalExpression::Constant(v) => (0, v.as_f32(), 0.0),
        SignalExpression::PassThrough => (1, 0.0, 0.0),
        SignalExpression::Invert => (2, 0.0, 0.0),
        SignalExpression::Scale(f) => (3, *f, 0.0),
        SignalExpression::Clamp(min, max) => (4, *min, *max),
    }
}

fn deserialize_expression(etype: u8, val: f32, val2: f32) -> SignalExpression {
    match etype {
        0 => SignalExpression::Constant(crate::signal::types::SignalValue::Float(val)),
        1 => SignalExpression::PassThrough,
        2 => SignalExpression::Invert,
        3 => SignalExpression::Scale(val),
        4 => SignalExpression::Clamp(val, val2),
        _ => SignalExpression::PassThrough,
    }
}

fn u8_to_signal_property(v: u8) -> SignalProperty {
    match v {
        0 => SignalProperty::Active,
        1 => SignalProperty::Throttle,
        2 => SignalProperty::Angle,
        3 => SignalProperty::Extension,
        4 => SignalProperty::Pressure,
        5 => SignalProperty::Speed,
        6 => SignalProperty::Level,
        7 => SignalProperty::SwitchState,
        _ => SignalProperty::Active,
    }
}

/// Client → server messages.
#[derive(Debug, Clone)]
pub enum ClientMsg {
    Connect { player_name: String },
    PlayerInput(PlayerInputData),
    BlockEditRequest(BlockEditData),
    /// Signal config update for a functional block (client → server).
    BlockConfigUpdate(crate::signal::config::BlockConfigUpdateData),
    /// Place or remove a sub-block element on a block face.
    SubBlockEdit(SubBlockEditData),
}

/// Server → client messages.
#[derive(Debug, Clone)]
pub enum ServerMsg {
    JoinResponse(JoinResponseData),
    WorldState(WorldStateData),
    ChunkBlockMods(ChunkBlockModsData),
    ShardRedirect(ShardRedirect),
    DamageEvent(DamageEventData),
    PlayerDestroyed(PlayerDestroyedData),
    StarCatalog(StarCatalogData),
    ShardPreConnect(handoff::ShardPreConnect),
    GalaxyWorldState(GalaxyWorldStateData),
    /// Full chunk snapshot (lz4-compressed). Sent on initial connection.
    ChunkSnapshot(ChunkSnapshotData),
    /// Incremental block changes to a chunk. Shard-type agnostic.
    ChunkDelta(ChunkDeltaData),
    /// Signal config snapshot for a functional block (server → client).
    BlockConfigState(crate::signal::config::BlockSignalConfig),
}

#[derive(Debug, Clone)]
pub struct JoinResponseData {
    pub seed: u64,
    pub planet_radius: u32,
    pub player_id: u64,
    pub spawn_position: DVec3,
    pub spawn_rotation: DQuat,
    pub spawn_forward: DVec3,
    pub session_token: SessionToken,
    pub shard_type: u8, // 0=Planet, 1=System, 2=Ship, 3=Galaxy
    pub galaxy_seed: u64,
    pub system_seed: u64,
    pub game_time: f64,
    pub reference_position: DVec3,
    pub reference_rotation: DQuat,
}

#[derive(Debug, Clone)]
pub struct StarCatalogData {
    pub galaxy_seed: u64,
    pub stars: Vec<StarCatalogEntryData>,
}

#[derive(Debug, Clone)]
pub struct StarCatalogEntryData {
    pub index: u32,
    pub position: DVec3,
    pub system_seed: u64,
    pub star_class: u8,
    pub luminosity: f32,
}

/// Galaxy shard world state sent to client during warp travel.
#[derive(Debug, Clone)]
pub struct GalaxyWorldStateData {
    pub tick: u64,
    pub ship_position: DVec3,   // in galaxy units
    pub ship_velocity: DVec3,   // in galaxy units/s
    pub ship_rotation: DQuat,
    pub warp_phase: u8,         // FlightPhase as u8
    pub eta_seconds: f64,
    pub origin_star_index: u32,
    pub target_star_index: u32,
}

#[derive(Debug, Clone, PartialEq)]
pub struct PlayerInputData {
    pub movement: [f32; 3],
    pub look_yaw: f32,
    pub look_pitch: f32,
    pub jump: bool,
    pub fly_toggle: bool,
    /// Toggle orbit stabilizer: when true, SOI entry zeros planet-relative velocity.
    pub orbit_stabilizer_toggle: bool,
    pub speed_tier: u8,
    pub action: u8,
    pub block_type: u16,
    pub tick: u64,
    pub thrust_limiter: f32,
    /// Roll input: -1.0 (Q, CCW) to +1.0 (E, CW). 0.0 when neither pressed.
    pub roll: f32,
}

/// Block edit action codes (client → server).
pub mod action {
    pub const BREAK: u8 = 1;
    pub const PLACE: u8 = 2;
    pub const INTERACT: u8 = 3;
    pub const OPEN_CONFIG: u8 = 8;
    pub const EXIT_SEAT: u8 = 9;
    pub const PLACE_SUB: u8 = 10;
    pub const REMOVE_SUB: u8 = 11;
}

/// Shard type identifiers.
pub mod shard_type {
    pub const PLANET: u8 = 0;
    pub const SYSTEM: u8 = 1;
    pub const SHIP: u8 = 2;
    pub const GALAXY: u8 = 3;
}

#[derive(Debug, Clone)]
pub struct BlockEditData {
    pub action: u8,
    pub eye: DVec3,
    pub look: DVec3,
    pub block_type: u16,
}

/// Client → server: place or remove a sub-block element on a block face.
#[derive(Debug, Clone)]
pub struct SubBlockEditData {
    pub block_pos: glam::IVec3,
    pub face: u8,
    pub element_type: u8,
    pub rotation: u8,
    pub action: u8,
}

#[derive(Debug, Clone)]
pub struct WorldStateData {
    pub tick: u64,
    pub origin: DVec3,
    pub players: Vec<PlayerSnapshotData>,
    pub bodies: Vec<CelestialBodyData>,
    pub ships: Vec<ShipRenderData>,
    pub lighting: Option<LightingData>,
    pub game_time: f64,
    /// Server-authoritative warp target star index (0xFFFFFFFF = none).
    pub warp_target_star_index: u32,
    /// Server-authoritative autopilot state (None = autopilot inactive).
    pub autopilot: Option<AutopilotSnapshotData>,
}

#[derive(Debug, Clone)]
pub struct CelestialBodyData {
    pub body_id: u32,
    pub position: DVec3,
    pub radius: f64,
    pub color: [f32; 3],
}

#[derive(Debug, Clone)]
pub struct ShipRenderData {
    pub ship_id: u64,
    pub position: DVec3,
    pub rotation: DQuat,
    pub is_own_ship: bool,
}

#[derive(Debug, Clone)]
pub struct LightingData {
    pub sun_direction: DVec3,
    pub sun_color: [f32; 3],
    pub sun_intensity: f32,
    pub ambient: f32,
}

#[derive(Debug, Clone)]
pub struct PlayerSnapshotData {
    pub player_id: u64,
    pub position: DVec3,
    pub rotation: DQuat,
    pub velocity: DVec3,
    pub grounded: bool,
    pub health: f32,
    pub shield: f32,
}

#[derive(Debug, Clone)]
pub struct ChunkBlockModsData {
    pub sector: u8,
    pub shell: u16,
    pub cx: u16,
    pub cy: u16,
    pub cz: u16,
    pub seq: u64,
    pub mods: Vec<BlockModData>,
}

#[derive(Debug, Clone)]
pub struct BlockModData {
    pub bx: u8,
    pub by: u8,
    pub bz: u8,
    pub block_type: u16,
}

/// Full chunk snapshot for initial sync. Shard-type agnostic addressing via IVec3.
///
/// Used by both ship shards (Cartesian chunks) and planet shards (cubic sphere chunks
/// mapped to IVec3). The `data` field is the lz4-compressed output of
/// `core::block::serialization::serialize_chunk()`.
#[derive(Debug, Clone)]
pub struct ChunkSnapshotData {
    /// Chunk address in unified Cartesian coordinates.
    pub chunk_x: i32,
    pub chunk_y: i32,
    pub chunk_z: i32,
    /// Monotonic edit sequence for ordering.
    pub seq: u64,
    /// lz4-compressed chunk data (palette + indices + metadata).
    pub data: Vec<u8>,
}

/// Incremental block changes to a chunk. Shard-type agnostic.
///
/// Replaces the planet-specific `ChunkBlockMods` for new code paths.
/// Both ship and planet shards use this message format.
#[derive(Debug, Clone)]
pub struct ChunkDeltaData {
    /// Chunk address in unified Cartesian coordinates.
    pub chunk_x: i32,
    pub chunk_y: i32,
    pub chunk_z: i32,
    /// Monotonic edit sequence for ordering.
    pub seq: u64,
    /// Block modifications within this chunk.
    pub mods: Vec<BlockModData>,
    /// Sub-block element modifications within this chunk.
    pub sub_block_mods: Vec<SubBlockModData>,
}

/// A single sub-block modification within a chunk delta.
#[derive(Debug, Clone)]
pub struct SubBlockModData {
    pub bx: u8,
    pub by: u8,
    pub bz: u8,
    pub face: u8,
    pub element_type: u8,
    pub rotation: u8,
    pub action: u8, // 0=remove, 1=place
}

#[derive(Debug, Clone)]
pub struct DamageEventData {
    pub target_id: u64,
    pub source_id: u64,
    pub damage: f32,
    pub weapon_type: u8,
}

#[derive(Debug, Clone)]
pub struct PlayerDestroyedData {
    pub player_id: u64,
    pub killer_id: u64,
    pub position: DVec3,
}

// -- Helpers (reuse the same conversion pattern as shard_message.rs) --

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

impl ClientMsg {
    pub fn serialize(&self) -> Vec<u8> {
        let mut builder = crate::builder_pool::acquire(256);

        match self {
            ClientMsg::Connect { player_name } => {
                let name = builder.create_string(player_name);
                let connect = fb::Connect::create(
                    &mut builder,
                    &fb::ConnectArgs {
                        player_name: Some(name),
                    },
                );
                let msg = fb::ClientMessage::create(
                    &mut builder,
                    &fb::ClientMessageArgs {
                        payload_type: fb::ClientPayload::Connect,
                        payload: Some(connect.as_union_value()),
                    },
                );
                builder.finish(msg, None);
            }
            ClientMsg::PlayerInput(data) => {
                let input = fb::PlayerInput::create(
                    &mut builder,
                    &fb::PlayerInputArgs {
                        movement_x: data.movement[0],
                        movement_y: data.movement[1],
                        movement_z: data.movement[2],
                        look_yaw: data.look_yaw,
                        look_pitch: data.look_pitch,
                        jump: data.jump,
                        fly_toggle: data.fly_toggle,
                        orbit_stabilizer_toggle: data.orbit_stabilizer_toggle,
                        speed_tier: data.speed_tier,
                        action: data.action,
                        block_type: data.block_type,
                        tick: data.tick,
                        thrust_limiter: data.thrust_limiter,
                        roll: data.roll,
                    },
                );
                let msg = fb::ClientMessage::create(
                    &mut builder,
                    &fb::ClientMessageArgs {
                        payload_type: fb::ClientPayload::PlayerInput,
                        payload: Some(input.as_union_value()),
                    },
                );
                builder.finish(msg, None);
            }
            ClientMsg::BlockEditRequest(data) => {
                let req = fb::BlockEditRequest::create(
                    &mut builder,
                    &fb::BlockEditRequestArgs {
                        action: data.action,
                        eye_x: data.eye.x,
                        eye_y: data.eye.y,
                        eye_z: data.eye.z,
                        look_x: data.look.x,
                        look_y: data.look.y,
                        look_z: data.look.z,
                        block_type: data.block_type,
                    },
                );
                let msg = fb::ClientMessage::create(
                    &mut builder,
                    &fb::ClientMessageArgs {
                        payload_type: fb::ClientPayload::BlockEditRequest,
                        payload: Some(req.as_union_value()),
                    },
                );
                builder.finish(msg, None);
            }
            ClientMsg::BlockConfigUpdate(data) => {
                let pub_b: Vec<_> = data.publish_bindings.iter().map(|b| {
                    let n = builder.create_string(&b.channel_name);
                    fb::SignalBindingFB::create(&mut builder, &fb::SignalBindingFBArgs {
                        channel_name: Some(n), property: b.property as u8,
                    })
                }).collect();
                let sub_b: Vec<_> = data.subscribe_bindings.iter().map(|b| {
                    let n = builder.create_string(&b.channel_name);
                    fb::SignalBindingFB::create(&mut builder, &fb::SignalBindingFBArgs {
                        channel_name: Some(n), property: b.property as u8,
                    })
                }).collect();
                let rules: Vec<_> = data.converter_rules.iter().map(|r| {
                    let ic = builder.create_string(&r.input_channel);
                    let oc = builder.create_string(&r.output_channel);
                    let (ct, cv) = serialize_condition(&r.condition);
                    let (et, ev, ev2) = serialize_expression(&r.expression);
                    fb::SignalRuleFB::create(&mut builder, &fb::SignalRuleFBArgs {
                        input_channel: Some(ic), condition_type: ct, condition_value: cv,
                        output_channel: Some(oc), expression_type: et,
                        expression_value: ev, expression_value2: ev2,
                    })
                }).collect();
                let seats: Vec<_> = data.seat_mappings.iter().map(|s| {
                    let ch = builder.create_string(&s.channel_name);
                    fb::SeatBindingFB::create(&mut builder, &fb::SeatBindingFBArgs {
                        control: s.control as u8, channel_name: Some(ch), property: s.property as u8,
                    })
                }).collect();
                let pv = builder.create_vector(&pub_b);
                let sv = builder.create_vector(&sub_b);
                let rv = builder.create_vector(&rules);
                let stv = builder.create_vector(&seats);
                let bcu = fb::BlockConfigUpdate::create(&mut builder, &fb::BlockConfigUpdateArgs {
                    block_x: data.block_pos.x, block_y: data.block_pos.y, block_z: data.block_pos.z,
                    publish_bindings: Some(pv), subscribe_bindings: Some(sv),
                    converter_rules: Some(rv), seat_mappings: Some(stv),
                });
                let msg = fb::ClientMessage::create(&mut builder, &fb::ClientMessageArgs {
                    payload_type: fb::ClientPayload::BlockConfigUpdate,
                    payload: Some(bcu.as_union_value()),
                });
                builder.finish(msg, None);
            }
            ClientMsg::SubBlockEdit(data) => {
                let req = fb::SubBlockEditRequest::create(
                    &mut builder,
                    &fb::SubBlockEditRequestArgs {
                        block_x: data.block_pos.x,
                        block_y: data.block_pos.y,
                        block_z: data.block_pos.z,
                        face: data.face,
                        element_type: data.element_type,
                        rotation: data.rotation,
                        action: data.action,
                    },
                );
                let msg = fb::ClientMessage::create(&mut builder, &fb::ClientMessageArgs {
                    payload_type: fb::ClientPayload::SubBlockEditRequest,
                    payload: Some(req.as_union_value()),
                });
                builder.finish(msg, None);
            }
        }

        let result = builder.finished_data().to_vec();
        crate::builder_pool::release(builder);
        result
    }

    pub fn deserialize(buf: &[u8]) -> Result<Self, MessageError> {
        let msg = flatbuffers::root::<fb::ClientMessage>(buf)
            .map_err(|e| MessageError::InvalidBuffer(e.to_string()))?;

        match msg.payload_type() {
            fb::ClientPayload::Connect => {
                let c = msg
                    .payload_as_connect()
                    .ok_or(MessageError::MissingField("Connect payload"))?;
                Ok(ClientMsg::Connect {
                    player_name: c
                        .player_name()
                        .ok_or(MessageError::MissingField("player_name"))?
                        .to_string(),
                })
            }
            fb::ClientPayload::PlayerInput => {
                let p = msg
                    .payload_as_player_input()
                    .ok_or(MessageError::MissingField("PlayerInput payload"))?;
                Ok(ClientMsg::PlayerInput(PlayerInputData {
                    movement: [p.movement_x(), p.movement_y(), p.movement_z()],
                    look_yaw: p.look_yaw(),
                    look_pitch: p.look_pitch(),
                    jump: p.jump(),
                    fly_toggle: p.fly_toggle(),
                    orbit_stabilizer_toggle: p.orbit_stabilizer_toggle(),
                    speed_tier: p.speed_tier(),
                    action: p.action(),
                    block_type: p.block_type(),
                    tick: p.tick(),
                    thrust_limiter: p.thrust_limiter(),
                    roll: p.roll(),
                }))
            }
            fb::ClientPayload::BlockEditRequest => {
                let r = msg
                    .payload_as_block_edit_request()
                    .ok_or(MessageError::MissingField("BlockEditRequest payload"))?;
                Ok(ClientMsg::BlockEditRequest(BlockEditData {
                    action: r.action(),
                    eye: DVec3::new(r.eye_x(), r.eye_y(), r.eye_z()),
                    look: DVec3::new(r.look_x(), r.look_y(), r.look_z()),
                    block_type: r.block_type(),
                }))
            }
            fb::ClientPayload::BlockConfigUpdate => {
                let bcu = msg.payload_as_block_config_update()
                    .ok_or(MessageError::MissingField("BlockConfigUpdate payload"))?;
                let pub_b = bcu.publish_bindings().map(|v| v.iter().map(|b| {
                    crate::signal::config::PublishBindingConfig {
                        channel_name: b.channel_name().unwrap_or("").to_string(),
                        property: u8_to_signal_property(b.property()),
                    }
                }).collect()).unwrap_or_default();
                let sub_b = bcu.subscribe_bindings().map(|v| v.iter().map(|b| {
                    crate::signal::config::SubscribeBindingConfig {
                        channel_name: b.channel_name().unwrap_or("").to_string(),
                        property: u8_to_signal_property(b.property()),
                    }
                }).collect()).unwrap_or_default();
                let rules = bcu.converter_rules().map(|v| v.iter().map(|r| {
                    crate::signal::config::SignalRuleConfig {
                        input_channel: r.input_channel().unwrap_or("").to_string(),
                        condition: deserialize_condition(r.condition_type(), r.condition_value()),
                        output_channel: r.output_channel().unwrap_or("").to_string(),
                        expression: deserialize_expression(r.expression_type(), r.expression_value(), r.expression_value2()),
                    }
                }).collect()).unwrap_or_default();
                let seats = bcu.seat_mappings().map(|v| v.iter().filter_map(|s| {
                    let control = crate::signal::components::SeatControl::from_u8(s.control())?;
                    Some(crate::signal::config::SeatInputBindingConfig {
                        control,
                        channel_name: s.channel_name().unwrap_or("").to_string(),
                        property: u8_to_signal_property(s.property()),
                    })
                }).collect()).unwrap_or_default();
                Ok(ClientMsg::BlockConfigUpdate(crate::signal::config::BlockConfigUpdateData {
                    block_pos: glam::IVec3::new(bcu.block_x(), bcu.block_y(), bcu.block_z()),
                    publish_bindings: pub_b,
                    subscribe_bindings: sub_b,
                    converter_rules: rules,
                    seat_mappings: seats,
                }))
            }
            fb::ClientPayload::SubBlockEditRequest => {
                let req = msg.payload_as_sub_block_edit_request()
                    .ok_or(MessageError::MissingField("SubBlockEditRequest payload"))?;
                Ok(ClientMsg::SubBlockEdit(SubBlockEditData {
                    block_pos: glam::IVec3::new(req.block_x(), req.block_y(), req.block_z()),
                    face: req.face(),
                    element_type: req.element_type(),
                    rotation: req.rotation(),
                    action: req.action(),
                }))
            }
            fb::ClientPayload::NONE => Err(MessageError::UnknownPayload(0)),
            other => Err(MessageError::UnknownPayload(other.0)),
        }
    }
}

impl ServerMsg {
    pub fn serialize(&self) -> Vec<u8> {
        let mut builder = crate::builder_pool::acquire(512);

        match self {
            ServerMsg::JoinResponse(data) => {
                let pos = to_fb_vec3d(&data.spawn_position);
                let rot = to_fb_quatd(&data.spawn_rotation);
                let fwd = to_fb_vec3d(&data.spawn_forward);
                let jr = fb::JoinResponse::create(
                    &mut builder,
                    &fb::JoinResponseArgs {
                        seed: data.seed,
                        planet_radius: data.planet_radius,
                        player_id: data.player_id,
                        spawn_position: Some(&pos),
                        spawn_rotation: Some(&rot),
                        spawn_forward: Some(&fwd),
                        session_token: data.session_token.0,
                        shard_type: data.shard_type,
                        galaxy_seed: data.galaxy_seed,
                        system_seed: data.system_seed,
                        game_time: data.game_time,
                        reference_position: Some(&to_fb_vec3d(&data.reference_position)),
                        reference_rotation: Some(&to_fb_quatd(&data.reference_rotation)),
                    },
                );
                let msg = fb::ServerMessage::create(
                    &mut builder,
                    &fb::ServerMessageArgs {
                        payload_type: fb::ServerPayload::JoinResponse,
                        payload: Some(jr.as_union_value()),
                    },
                );
                builder.finish(msg, None);
            }
            ServerMsg::ShardRedirect(data) => {
                let tcp = builder.create_string(&data.target_tcp_addr);
                let udp = builder.create_string(&data.target_udp_addr);
                let sr = fb::ShardRedirectMsg::create(
                    &mut builder,
                    &fb::ShardRedirectMsgArgs {
                        session_token: data.session_token.0,
                        target_tcp_addr: Some(tcp),
                        target_udp_addr: Some(udp),
                        shard_id: data.shard_id.0,
                    },
                );
                let msg = fb::ServerMessage::create(
                    &mut builder,
                    &fb::ServerMessageArgs {
                        payload_type: fb::ServerPayload::ShardRedirectMsg,
                        payload: Some(sr.as_union_value()),
                    },
                );
                builder.finish(msg, None);
            }
            ServerMsg::WorldState(data) => {
                let origin = to_fb_vec3d(&data.origin);
                let snapshots: Vec<_> = data.players.iter().map(|p| {
                    let pos = to_fb_vec3d(&p.position);
                    let rot = to_fb_quatd(&p.rotation);
                    let vel = to_fb_vec3d(&p.velocity);
                    fb::PlayerSnapshot::create(&mut builder, &fb::PlayerSnapshotArgs {
                        player_id: p.player_id,
                        position: Some(&pos),
                        rotation: Some(&rot),
                        velocity: Some(&vel),
                        grounded: p.grounded,
                        health: p.health,
                        shield: p.shield,
                    })
                }).collect();
                let players = builder.create_vector(&snapshots);
                let body_fbs: Vec<_> = data.bodies.iter().map(|b| {
                    let pos = to_fb_vec3d(&b.position);
                    fb::CelestialBodySnapshot::create(&mut builder, &fb::CelestialBodySnapshotArgs {
                        body_id: b.body_id, position: Some(&pos), radius: b.radius,
                        color_r: b.color[0], color_g: b.color[1], color_b: b.color[2],
                    })
                }).collect();
                let bodies_vec = builder.create_vector(&body_fbs);

                let ship_fbs: Vec<_> = data.ships.iter().map(|s| {
                    let pos = to_fb_vec3d(&s.position);
                    let rot = to_fb_quatd(&s.rotation);
                    fb::ShipSnapshotEntry::create(&mut builder, &fb::ShipSnapshotEntryArgs {
                        ship_id: s.ship_id, position: Some(&pos), rotation: Some(&rot),
                        is_own_ship: s.is_own_ship,
                    })
                }).collect();
                let ships_vec = builder.create_vector(&ship_fbs);

                let lighting_fb = data.lighting.as_ref().map(|l| {
                    let sd = to_fb_vec3d(&l.sun_direction);
                    fb::LightingInfoMsg::create(&mut builder, &fb::LightingInfoMsgArgs {
                        sun_direction: Some(&sd),
                        sun_color_r: l.sun_color[0], sun_color_g: l.sun_color[1], sun_color_b: l.sun_color[2],
                        sun_intensity: l.sun_intensity, ambient: l.ambient,
                    })
                });

                let ap_offset = data.autopilot.as_ref().map(|ap| {
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

                let ws = fb::WorldState::create(&mut builder, &fb::WorldStateArgs {
                    tick: data.tick,
                    origin: Some(&origin),
                    players: Some(players),
                    bodies: Some(bodies_vec),
                    ships: Some(ships_vec),
                    lighting: lighting_fb,
                    game_time: data.game_time,
                    warp_target_star_index: data.warp_target_star_index,
                    autopilot: ap_offset,
                });
                let msg = fb::ServerMessage::create(&mut builder, &fb::ServerMessageArgs {
                    payload_type: fb::ServerPayload::WorldState,
                    payload: Some(ws.as_union_value()),
                });
                builder.finish(msg, None);
            }
            ServerMsg::ChunkBlockMods(data) => {
                let mods: Vec<_> = data.mods.iter().map(|m| {
                    fb::BlockMod::create(&mut builder, &fb::BlockModArgs {
                        bx: m.bx, by: m.by, bz: m.bz, block_type: m.block_type,
                    })
                }).collect();
                let mods_vec = builder.create_vector(&mods);
                let cbm = fb::ChunkBlockMods::create(&mut builder, &fb::ChunkBlockModsArgs {
                    sector: data.sector, shell: data.shell,
                    cx: data.cx, cy: data.cy, cz: data.cz,
                    seq: data.seq, mods: Some(mods_vec),
                });
                let msg = fb::ServerMessage::create(&mut builder, &fb::ServerMessageArgs {
                    payload_type: fb::ServerPayload::ChunkBlockMods,
                    payload: Some(cbm.as_union_value()),
                });
                builder.finish(msg, None);
            }
            ServerMsg::DamageEvent(data) => {
                let de = fb::DamageEvent::create(&mut builder, &fb::DamageEventArgs {
                    target_id: data.target_id, source_id: data.source_id,
                    damage: data.damage, weapon_type: data.weapon_type,
                });
                let msg = fb::ServerMessage::create(&mut builder, &fb::ServerMessageArgs {
                    payload_type: fb::ServerPayload::DamageEvent,
                    payload: Some(de.as_union_value()),
                });
                builder.finish(msg, None);
            }
            ServerMsg::PlayerDestroyed(data) => {
                let pos = to_fb_vec3d(&data.position);
                let pd = fb::PlayerDestroyed::create(&mut builder, &fb::PlayerDestroyedArgs {
                    player_id: data.player_id, killer_id: data.killer_id,
                    position: Some(&pos),
                });
                let msg = fb::ServerMessage::create(&mut builder, &fb::ServerMessageArgs {
                    payload_type: fb::ServerPayload::PlayerDestroyed,
                    payload: Some(pd.as_union_value()),
                });
                builder.finish(msg, None);
            }
            ServerMsg::StarCatalog(data) => {
                let entries: Vec<_> = data.stars.iter().map(|s| {
                    let pos = to_fb_vec3d(&DVec3::new(s.position.x, s.position.y, s.position.z));
                    fb::StarCatalogEntry::create(&mut builder, &fb::StarCatalogEntryArgs {
                        index: s.index, position: Some(&pos),
                        system_seed: s.system_seed, star_class: s.star_class,
                        luminosity: s.luminosity,
                    })
                }).collect();
                let stars = builder.create_vector(&entries);
                let sc = fb::StarCatalog::create(&mut builder, &fb::StarCatalogArgs {
                    galaxy_seed: data.galaxy_seed, stars: Some(stars),
                });
                let msg = fb::ServerMessage::create(&mut builder, &fb::ServerMessageArgs {
                    payload_type: fb::ServerPayload::StarCatalog,
                    payload: Some(sc.as_union_value()),
                });
                builder.finish(msg, None);
            }
            ServerMsg::ShardPreConnect(data) => {
                let tcp = builder.create_string(&data.tcp_addr);
                let udp = builder.create_string(&data.udp_addr);
                let ref_pos = to_fb_vec3d(&data.reference_position);
                let ref_rot = to_fb_quatd(&data.reference_rotation);
                let pc = fb::ShardPreConnect::create(&mut builder, &fb::ShardPreConnectArgs {
                    shard_type: data.shard_type,
                    tcp_addr: Some(tcp),
                    udp_addr: Some(udp),
                    seed: data.seed,
                    planet_index: data.planet_index,
                    reference_position: Some(&ref_pos),
                    reference_rotation: Some(&ref_rot),
                });
                let msg = fb::ServerMessage::create(&mut builder, &fb::ServerMessageArgs {
                    payload_type: fb::ServerPayload::ShardPreConnect,
                    payload: Some(pc.as_union_value()),
                });
                builder.finish(msg, None);
            }
            ServerMsg::GalaxyWorldState(data) => {
                let pos = to_fb_vec3d(&data.ship_position);
                let vel = to_fb_vec3d(&data.ship_velocity);
                let rot = to_fb_quatd(&data.ship_rotation);
                let gws = fb::GalaxyWorldState::create(&mut builder, &fb::GalaxyWorldStateArgs {
                    tick: data.tick,
                    ship_position: Some(&pos),
                    ship_velocity: Some(&vel),
                    ship_rotation: Some(&rot),
                    warp_phase: data.warp_phase,
                    eta_seconds: data.eta_seconds,
                    origin_star_index: data.origin_star_index,
                    target_star_index: data.target_star_index,
                });
                let msg = fb::ServerMessage::create(&mut builder, &fb::ServerMessageArgs {
                    payload_type: fb::ServerPayload::GalaxyWorldState,
                    payload: Some(gws.as_union_value()),
                });
                builder.finish(msg, None);
            }
            ServerMsg::ChunkSnapshot(data) => {
                let addr = fb::ChunkAddr::new(data.chunk_x, data.chunk_y, data.chunk_z);
                let payload_data = builder.create_vector(&data.data);
                let cs = fb::ChunkSnapshot::create(&mut builder, &fb::ChunkSnapshotArgs {
                    addr: Some(&addr),
                    seq: data.seq,
                    data: Some(payload_data),
                });
                let msg = fb::ServerMessage::create(&mut builder, &fb::ServerMessageArgs {
                    payload_type: fb::ServerPayload::ChunkSnapshot,
                    payload: Some(cs.as_union_value()),
                });
                builder.finish(msg, None);
            }
            ServerMsg::ChunkDelta(data) => {
                let addr = fb::ChunkAddr::new(data.chunk_x, data.chunk_y, data.chunk_z);
                let mods: Vec<_> = data.mods.iter().map(|m| {
                    fb::BlockMod::create(&mut builder, &fb::BlockModArgs {
                        bx: m.bx, by: m.by, bz: m.bz, block_type: m.block_type,
                    })
                }).collect();
                let mods_vec = builder.create_vector(&mods);
                let sb_mods: Vec<_> = data.sub_block_mods.iter().map(|m| {
                    fb::SubBlockMod::create(&mut builder, &fb::SubBlockModArgs {
                        bx: m.bx, by: m.by, bz: m.bz,
                        face: m.face, element_type: m.element_type,
                        rotation: m.rotation, action: m.action,
                    })
                }).collect();
                let sb_mods_vec = if sb_mods.is_empty() { None } else { Some(builder.create_vector(&sb_mods)) };
                let cd = fb::ChunkDelta::create(&mut builder, &fb::ChunkDeltaArgs {
                    addr: Some(&addr),
                    seq: data.seq,
                    mods: Some(mods_vec),
                    sub_block_mods: sb_mods_vec,
                });
                let msg = fb::ServerMessage::create(&mut builder, &fb::ServerMessageArgs {
                    payload_type: fb::ServerPayload::ChunkDelta,
                    payload: Some(cd.as_union_value()),
                });
                builder.finish(msg, None);
            }
            ServerMsg::BlockConfigState(data) => {
                let pub_bindings: Vec<_> = data.publish_bindings.iter().map(|b| {
                    let name = builder.create_string(&b.channel_name);
                    fb::SignalBindingFB::create(&mut builder, &fb::SignalBindingFBArgs {
                        channel_name: Some(name), property: b.property as u8,
                    })
                }).collect();
                let sub_bindings: Vec<_> = data.subscribe_bindings.iter().map(|b| {
                    let name = builder.create_string(&b.channel_name);
                    fb::SignalBindingFB::create(&mut builder, &fb::SignalBindingFBArgs {
                        channel_name: Some(name), property: b.property as u8,
                    })
                }).collect();
                let rules: Vec<_> = data.converter_rules.iter().map(|r| {
                    let input = builder.create_string(&r.input_channel);
                    let output = builder.create_string(&r.output_channel);
                    let (cond_type, cond_val) = serialize_condition(&r.condition);
                    let (expr_type, expr_val, expr_val2) = serialize_expression(&r.expression);
                    fb::SignalRuleFB::create(&mut builder, &fb::SignalRuleFBArgs {
                        input_channel: Some(input), condition_type: cond_type, condition_value: cond_val,
                        output_channel: Some(output), expression_type: expr_type,
                        expression_value: expr_val, expression_value2: expr_val2,
                    })
                }).collect();
                let seats: Vec<_> = data.seat_mappings.iter().map(|s| {
                    let channel = builder.create_string(&s.channel_name);
                    fb::SeatBindingFB::create(&mut builder, &fb::SeatBindingFBArgs {
                        control: s.control as u8, channel_name: Some(channel), property: s.property as u8,
                    })
                }).collect();
                let channels: Vec<_> = data.available_channels.iter()
                    .map(|c| builder.create_string(c)).collect();

                let pub_vec = builder.create_vector(&pub_bindings);
                let sub_vec = builder.create_vector(&sub_bindings);
                let rules_vec = builder.create_vector(&rules);
                let seats_vec = builder.create_vector(&seats);
                let channels_vec = builder.create_vector(&channels);

                let bcs = fb::BlockConfigState::create(&mut builder, &fb::BlockConfigStateArgs {
                    block_x: data.block_pos.x, block_y: data.block_pos.y, block_z: data.block_pos.z,
                    block_type: data.block_type, kind: data.kind,
                    publish_bindings: Some(pub_vec), subscribe_bindings: Some(sub_vec),
                    converter_rules: Some(rules_vec), seat_mappings: Some(seats_vec),
                    available_channels: Some(channels_vec),
                });
                let msg = fb::ServerMessage::create(&mut builder, &fb::ServerMessageArgs {
                    payload_type: fb::ServerPayload::BlockConfigState,
                    payload: Some(bcs.as_union_value()),
                });
                builder.finish(msg, None);
            }
        }

        let result = builder.finished_data().to_vec();
        crate::builder_pool::release(builder);
        result
    }

    pub fn deserialize(buf: &[u8]) -> Result<Self, MessageError> {
        let msg = flatbuffers::root::<fb::ServerMessage>(buf)
            .map_err(|e| MessageError::InvalidBuffer(e.to_string()))?;

        match msg.payload_type() {
            fb::ServerPayload::JoinResponse => {
                let jr = msg
                    .payload_as_join_response()
                    .ok_or(MessageError::MissingField("JoinResponse payload"))?;
                let pos = jr
                    .spawn_position()
                    .ok_or(MessageError::MissingField("spawn_position"))?;
                let rot = jr
                    .spawn_rotation()
                    .ok_or(MessageError::MissingField("spawn_rotation"))?;
                let fwd = jr
                    .spawn_forward()
                    .ok_or(MessageError::MissingField("spawn_forward"))?;

                Ok(ServerMsg::JoinResponse(JoinResponseData {
                    seed: jr.seed(),
                    planet_radius: jr.planet_radius(),
                    player_id: jr.player_id(),
                    spawn_position: from_fb_vec3d(pos),
                    spawn_rotation: from_fb_quatd(rot),
                    spawn_forward: from_fb_vec3d(fwd),
                    session_token: SessionToken(jr.session_token()),
                    shard_type: jr.shard_type(),
                    galaxy_seed: jr.galaxy_seed(),
                    system_seed: jr.system_seed(),
                    game_time: jr.game_time(),
                    reference_position: jr.reference_position().map(|v| from_fb_vec3d(v)).unwrap_or(DVec3::ZERO),
                    reference_rotation: jr.reference_rotation().map(|q| from_fb_quatd(q)).unwrap_or(DQuat::IDENTITY),
                }))
            }
            fb::ServerPayload::ShardRedirectMsg => {
                let sr = msg
                    .payload_as_shard_redirect_msg()
                    .ok_or(MessageError::MissingField("ShardRedirectMsg payload"))?;
                Ok(ServerMsg::ShardRedirect(ShardRedirect {
                    session_token: SessionToken(sr.session_token()),
                    target_tcp_addr: sr
                        .target_tcp_addr()
                        .ok_or(MessageError::MissingField("target_tcp_addr"))?
                        .to_string(),
                    target_udp_addr: sr
                        .target_udp_addr()
                        .ok_or(MessageError::MissingField("target_udp_addr"))?
                        .to_string(),
                    shard_id: ShardId(sr.shard_id()),
                }))
            }
            fb::ServerPayload::WorldState => {
                let ws = msg.payload_as_world_state()
                    .ok_or(MessageError::MissingField("WorldState payload"))?;
                let origin = ws.origin().ok_or(MessageError::MissingField("origin"))?;
                let players = ws.players().map(|v| {
                    v.iter().map(|p| {
                        let pos = p.position().unwrap();
                        let rot = p.rotation().unwrap();
                        let vel = p.velocity().unwrap();
                        PlayerSnapshotData {
                            player_id: p.player_id(),
                            position: from_fb_vec3d(pos),
                            rotation: from_fb_quatd(rot),
                            velocity: from_fb_vec3d(vel),
                            grounded: p.grounded(),
                            health: p.health(),
                            shield: p.shield(),
                        }
                    }).collect()
                }).unwrap_or_default();
                let bodies = ws.bodies().map(|v| v.iter().map(|b| {
                    let pos = b.position().unwrap();
                    CelestialBodyData {
                        body_id: b.body_id(), position: from_fb_vec3d(pos),
                        radius: b.radius(), color: [b.color_r(), b.color_g(), b.color_b()],
                    }
                }).collect()).unwrap_or_default();

                let ships = ws.ships().map(|v| v.iter().map(|s| {
                    let pos = s.position().unwrap();
                    let rot = s.rotation().unwrap();
                    ShipRenderData {
                        ship_id: s.ship_id(), position: from_fb_vec3d(pos),
                        rotation: from_fb_quatd(rot), is_own_ship: s.is_own_ship(),
                    }
                }).collect()).unwrap_or_default();

                let lighting = ws.lighting().map(|l| {
                    let sd = l.sun_direction().unwrap();
                    LightingData {
                        sun_direction: from_fb_vec3d(sd),
                        sun_color: [l.sun_color_r(), l.sun_color_g(), l.sun_color_b()],
                        sun_intensity: l.sun_intensity(), ambient: l.ambient(),
                    }
                });

                let autopilot = ws.autopilot().map(|ap| {
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

                Ok(ServerMsg::WorldState(WorldStateData {
                    tick: ws.tick(),
                    origin: from_fb_vec3d(origin),
                    players,
                    bodies,
                    ships,
                    lighting,
                    game_time: ws.game_time(),
                    warp_target_star_index: ws.warp_target_star_index(),
                    autopilot,
                }))
            }
            fb::ServerPayload::ChunkBlockMods => {
                let cbm = msg.payload_as_chunk_block_mods()
                    .ok_or(MessageError::MissingField("ChunkBlockMods payload"))?;
                let mods = cbm.mods().map(|v| {
                    v.iter().map(|m| BlockModData {
                        bx: m.bx(), by: m.by(), bz: m.bz(), block_type: m.block_type(),
                    }).collect()
                }).unwrap_or_default();
                Ok(ServerMsg::ChunkBlockMods(ChunkBlockModsData {
                    sector: cbm.sector(), shell: cbm.shell(),
                    cx: cbm.cx(), cy: cbm.cy(), cz: cbm.cz(),
                    seq: cbm.seq(), mods,
                }))
            }
            fb::ServerPayload::DamageEvent => {
                let de = msg.payload_as_damage_event()
                    .ok_or(MessageError::MissingField("DamageEvent payload"))?;
                Ok(ServerMsg::DamageEvent(DamageEventData {
                    target_id: de.target_id(), source_id: de.source_id(),
                    damage: de.damage(), weapon_type: de.weapon_type(),
                }))
            }
            fb::ServerPayload::PlayerDestroyed => {
                let pd = msg.payload_as_player_destroyed()
                    .ok_or(MessageError::MissingField("PlayerDestroyed payload"))?;
                let pos = pd.position().ok_or(MessageError::MissingField("position"))?;
                Ok(ServerMsg::PlayerDestroyed(PlayerDestroyedData {
                    player_id: pd.player_id(), killer_id: pd.killer_id(),
                    position: from_fb_vec3d(pos),
                }))
            }
            fb::ServerPayload::StarCatalog => {
                let sc = msg.payload_as_star_catalog()
                    .ok_or(MessageError::MissingField("StarCatalog payload"))?;
                let stars = sc.stars().map(|v| {
                    v.iter().map(|s| {
                        let pos = s.position().unwrap();
                        StarCatalogEntryData {
                            index: s.index(),
                            position: from_fb_vec3d(pos),
                            system_seed: s.system_seed(),
                            star_class: s.star_class(),
                            luminosity: s.luminosity(),
                        }
                    }).collect()
                }).unwrap_or_default();
                Ok(ServerMsg::StarCatalog(StarCatalogData {
                    galaxy_seed: sc.galaxy_seed(),
                    stars,
                }))
            }
            fb::ServerPayload::ShardPreConnect => {
                let pc = msg.payload_as_shard_pre_connect()
                    .ok_or(MessageError::MissingField("ShardPreConnect payload"))?;
                let ref_pos = pc.reference_position()
                    .ok_or(MessageError::MissingField("reference_position"))?;
                let ref_rot = pc.reference_rotation()
                    .ok_or(MessageError::MissingField("reference_rotation"))?;
                Ok(ServerMsg::ShardPreConnect(handoff::ShardPreConnect {
                    shard_type: pc.shard_type(),
                    tcp_addr: pc.tcp_addr().unwrap_or("").to_string(),
                    udp_addr: pc.udp_addr().unwrap_or("").to_string(),
                    seed: pc.seed(),
                    planet_index: pc.planet_index(),
                    reference_position: from_fb_vec3d(ref_pos),
                    reference_rotation: from_fb_quatd(ref_rot),
                }))
            }
            fb::ServerPayload::GalaxyWorldState => {
                let gws = msg.payload_as_galaxy_world_state()
                    .ok_or(MessageError::MissingField("GalaxyWorldState payload"))?;
                let pos = gws.ship_position()
                    .ok_or(MessageError::MissingField("ship_position"))?;
                let vel = gws.ship_velocity()
                    .ok_or(MessageError::MissingField("ship_velocity"))?;
                let rot = gws.ship_rotation()
                    .ok_or(MessageError::MissingField("ship_rotation"))?;
                Ok(ServerMsg::GalaxyWorldState(GalaxyWorldStateData {
                    tick: gws.tick(),
                    ship_position: from_fb_vec3d(pos),
                    ship_velocity: from_fb_vec3d(vel),
                    ship_rotation: from_fb_quatd(rot),
                    warp_phase: gws.warp_phase(),
                    eta_seconds: gws.eta_seconds(),
                    origin_star_index: gws.origin_star_index(),
                    target_star_index: gws.target_star_index(),
                }))
            }
            fb::ServerPayload::ChunkSnapshot => {
                let cs = msg.payload_as_chunk_snapshot()
                    .ok_or(MessageError::MissingField("ChunkSnapshot payload"))?;
                let addr = cs.addr().ok_or(MessageError::MissingField("ChunkSnapshot addr"))?;
                let data = cs.data().map(|v| v.iter().collect::<Vec<u8>>()).unwrap_or_default();
                Ok(ServerMsg::ChunkSnapshot(ChunkSnapshotData {
                    chunk_x: addr.x(),
                    chunk_y: addr.y(),
                    chunk_z: addr.z(),
                    seq: cs.seq(),
                    data,
                }))
            }
            fb::ServerPayload::ChunkDelta => {
                let cd = msg.payload_as_chunk_delta()
                    .ok_or(MessageError::MissingField("ChunkDelta payload"))?;
                let addr = cd.addr().ok_or(MessageError::MissingField("ChunkDelta addr"))?;
                let mods = cd.mods().map(|v| {
                    v.iter().map(|m| BlockModData {
                        bx: m.bx(), by: m.by(), bz: m.bz(), block_type: m.block_type(),
                    }).collect()
                }).unwrap_or_default();
                let sub_block_mods = cd.sub_block_mods().map(|v| {
                    v.iter().map(|m| SubBlockModData {
                        bx: m.bx(), by: m.by(), bz: m.bz(),
                        face: m.face(), element_type: m.element_type(),
                        rotation: m.rotation(), action: m.action(),
                    }).collect()
                }).unwrap_or_default();
                Ok(ServerMsg::ChunkDelta(ChunkDeltaData {
                    chunk_x: addr.x(),
                    chunk_y: addr.y(),
                    chunk_z: addr.z(),
                    seq: cd.seq(),
                    mods,
                    sub_block_mods,
                }))
            }
            fb::ServerPayload::BlockConfigState => {
                let bcs = msg.payload_as_block_config_state()
                    .ok_or(MessageError::MissingField("BlockConfigState payload"))?;
                let pub_b = bcs.publish_bindings().map(|v| v.iter().map(|b| {
                    crate::signal::config::PublishBindingConfig {
                        channel_name: b.channel_name().unwrap_or("").to_string(),
                        property: u8_to_signal_property(b.property()),
                    }
                }).collect()).unwrap_or_default();
                let sub_b = bcs.subscribe_bindings().map(|v| v.iter().map(|b| {
                    crate::signal::config::SubscribeBindingConfig {
                        channel_name: b.channel_name().unwrap_or("").to_string(),
                        property: u8_to_signal_property(b.property()),
                    }
                }).collect()).unwrap_or_default();
                let rules = bcs.converter_rules().map(|v| v.iter().map(|r| {
                    crate::signal::config::SignalRuleConfig {
                        input_channel: r.input_channel().unwrap_or("").to_string(),
                        condition: deserialize_condition(r.condition_type(), r.condition_value()),
                        output_channel: r.output_channel().unwrap_or("").to_string(),
                        expression: deserialize_expression(r.expression_type(), r.expression_value(), r.expression_value2()),
                    }
                }).collect()).unwrap_or_default();
                let seats = bcs.seat_mappings().map(|v| v.iter().filter_map(|s| {
                    let control = crate::signal::components::SeatControl::from_u8(s.control())?;
                    Some(crate::signal::config::SeatInputBindingConfig {
                        control,
                        channel_name: s.channel_name().unwrap_or("").to_string(),
                        property: u8_to_signal_property(s.property()),
                    })
                }).collect()).unwrap_or_default();
                let channels = bcs.available_channels().map(|v| {
                    v.iter().map(|s| s.to_string()).collect()
                }).unwrap_or_default();
                Ok(ServerMsg::BlockConfigState(crate::signal::config::BlockSignalConfig {
                    block_pos: glam::IVec3::new(bcs.block_x(), bcs.block_y(), bcs.block_z()),
                    block_type: bcs.block_type(),
                    kind: bcs.kind(),
                    publish_bindings: pub_b,
                    subscribe_bindings: sub_b,
                    converter_rules: rules,
                    seat_mappings: seats,
                    available_channels: channels,
                }))
            }
            fb::ServerPayload::NONE => Err(MessageError::UnknownPayload(0)),
            other => Err(MessageError::UnknownPayload(other.0)),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn roundtrip_connect() {
        let msg = ClientMsg::Connect {
            player_name: "Cosmonaut".to_string(),
        };
        let bytes = msg.serialize();
        let decoded = ClientMsg::deserialize(&bytes).unwrap();

        if let ClientMsg::Connect { player_name } = decoded {
            assert_eq!(player_name, "Cosmonaut");
        } else {
            panic!("expected Connect");
        }
    }

    #[test]
    fn roundtrip_join_response() {
        let msg = ServerMsg::JoinResponse(JoinResponseData {
            seed: 42,
            planet_radius: 150000,
            player_id: 1,
            spawn_position: DVec3::new(0.0, 150000.0, 0.0),
            spawn_rotation: DQuat::IDENTITY,
            spawn_forward: DVec3::NEG_Z,
            session_token: SessionToken(99),
            shard_type: 0,
            galaxy_seed: 100,
            system_seed: 200,
            game_time: 1000.5,
            reference_position: DVec3::new(1e11, 0.0, 0.0),
            reference_rotation: DQuat::IDENTITY,
        });
        let bytes = msg.serialize();
        let decoded = ServerMsg::deserialize(&bytes).unwrap();

        if let ServerMsg::JoinResponse(jr) = decoded {
            assert_eq!(jr.seed, 42);
            assert_eq!(jr.planet_radius, 150000);
            assert_eq!(jr.player_id, 1);
            assert!((jr.spawn_position.y - 150000.0).abs() < 1e-10);
            assert_eq!(jr.session_token, SessionToken(99));
            assert_eq!(jr.shard_type, 0);
            assert_eq!(jr.galaxy_seed, 100);
            assert_eq!(jr.system_seed, 200);
            assert!((jr.game_time - 1000.5).abs() < 1e-10);
            assert!((jr.reference_position.x - 1e11).abs() < 1.0);
        } else {
            panic!("expected JoinResponse");
        }
    }

    #[test]
    fn roundtrip_shard_redirect() {
        let msg = ServerMsg::ShardRedirect(ShardRedirect {
            session_token: SessionToken(555),
            target_tcp_addr: "127.0.0.1:7777".to_string(),
            target_udp_addr: "127.0.0.1:7778".to_string(),
            shard_id: ShardId(10),
        });
        let bytes = msg.serialize();
        let decoded = ServerMsg::deserialize(&bytes).unwrap();

        if let ServerMsg::ShardRedirect(sr) = decoded {
            assert_eq!(sr.session_token, SessionToken(555));
            assert_eq!(sr.target_tcp_addr, "127.0.0.1:7777");
            assert_eq!(sr.target_udp_addr, "127.0.0.1:7778");
            assert_eq!(sr.shard_id, ShardId(10));
        } else {
            panic!("expected ShardRedirect");
        }
    }

    #[test]
    fn roundtrip_player_input() {
        let msg = ClientMsg::PlayerInput(PlayerInputData {
            movement: [1.0, 0.0, -1.0],
            look_yaw: 3.14,
            look_pitch: -0.5,
            jump: true,
            fly_toggle: false,
            orbit_stabilizer_toggle: false,
            speed_tier: 2,
            action: 1,
            block_type: 3,
            tick: 1000,
            thrust_limiter: 0.75,
            roll: 0.0,
        });
        let bytes = msg.serialize();
        let decoded = ClientMsg::deserialize(&bytes).unwrap();

        if let ClientMsg::PlayerInput(p) = decoded {
            assert!((p.movement[0] - 1.0).abs() < 1e-5);
            assert!(p.jump);
            assert_eq!(p.speed_tier, 2);
            assert_eq!(p.tick, 1000);
        } else {
            panic!("expected PlayerInput");
        }
    }

    #[test]
    fn roundtrip_world_state() {
        let msg = ServerMsg::WorldState(WorldStateData {
            tick: 500,
            origin: DVec3::new(0.0, 150000.0, 0.0),
            players: vec![PlayerSnapshotData {
                player_id: 1,
                position: DVec3::new(10.0, 150010.0, 5.0),
                rotation: DQuat::IDENTITY,
                velocity: DVec3::new(1.0, 0.0, -1.0),
                grounded: true,
                health: 95.0,
                shield: 50.0,
            }],
            bodies: vec![CelestialBodyData {
                body_id: 0, position: DVec3::ZERO, radius: 6.96e8, color: [1.0, 0.95, 0.8],
            }],
            ships: vec![],
            lighting: Some(LightingData {
                sun_direction: DVec3::new(-1.0, 0.0, 0.0),
                sun_color: [1.0, 0.95, 0.8],
                sun_intensity: 0.9,
                ambient: 0.08,
            }),
            game_time: 42.0,
            warp_target_star_index: 0,
            autopilot: None,
        });
        let bytes = msg.serialize();
        let decoded = ServerMsg::deserialize(&bytes).unwrap();
        if let ServerMsg::WorldState(ws) = decoded {
            assert_eq!(ws.tick, 500);
            assert_eq!(ws.players.len(), 1);
            assert_eq!(ws.players[0].player_id, 1);
            assert!((ws.players[0].health - 95.0).abs() < 1e-5);
            assert!(ws.autopilot.is_none());
        } else {
            panic!("expected WorldState");
        }
    }

    #[test]
    fn roundtrip_damage_event() {
        let msg = ServerMsg::DamageEvent(DamageEventData {
            target_id: 1, source_id: 2, damage: 25.5, weapon_type: 1,
        });
        let bytes = msg.serialize();
        let decoded = ServerMsg::deserialize(&bytes).unwrap();
        if let ServerMsg::DamageEvent(de) = decoded {
            assert_eq!(de.target_id, 1);
            assert!((de.damage - 25.5).abs() < 1e-5);
        } else {
            panic!("expected DamageEvent");
        }
    }

    #[test]
    fn roundtrip_star_catalog() {
        let msg = ServerMsg::StarCatalog(StarCatalogData {
            galaxy_seed: 42,
            stars: vec![
                StarCatalogEntryData {
                    index: 0,
                    position: DVec3::new(100.0, 5.0, -200.0),
                    system_seed: 12345,
                    star_class: 4, // G
                    luminosity: 1.0,
                },
                StarCatalogEntryData {
                    index: 1,
                    position: DVec3::new(-300.0, 10.0, 500.0),
                    system_seed: 67890,
                    star_class: 6, // M
                    luminosity: 0.08,
                },
            ],
        });
        let bytes = msg.serialize();
        let decoded = ServerMsg::deserialize(&bytes).unwrap();
        if let ServerMsg::StarCatalog(sc) = decoded {
            assert_eq!(sc.galaxy_seed, 42);
            assert_eq!(sc.stars.len(), 2);
            assert_eq!(sc.stars[0].system_seed, 12345);
            assert_eq!(sc.stars[1].star_class, 6);
        } else {
            panic!("expected StarCatalog");
        }
    }
}
