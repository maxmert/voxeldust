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

fn encode_power_source<'a>(
    builder: &mut FlatBufferBuilder<'a>,
    src: &crate::signal::config::PowerSourceConfig,
) -> flatbuffers::WIPOffset<fb::PowerSourceConfigFB<'a>> {
    let circuits: Vec<_> = src.circuits.iter().map(|c| {
        let name = builder.create_string(&c.name);
        fb::PowerCircuitFB::create(builder, &fb::PowerCircuitFBArgs {
            name: Some(name), fraction: c.fraction,
        })
    }).collect();
    let circuits_vec = builder.create_vector(&circuits);
    let (access_mode, allow_list) = match &src.access {
        crate::signal::config::PowerAccessConfig::OwnerOnly => (0u8, None),
        crate::signal::config::PowerAccessConfig::AllowList(names) => {
            let strs: Vec<_> = names.iter().map(|n| builder.create_string(n)).collect();
            (1u8, Some(builder.create_vector(&strs)))
        }
        crate::signal::config::PowerAccessConfig::Open => (2u8, None),
    };
    fb::PowerSourceConfigFB::create(builder, &fb::PowerSourceConfigFBArgs {
        circuits: Some(circuits_vec), access_mode, allow_list,
    })
}

fn decode_power_source(fb: &fb::PowerSourceConfigFB<'_>) -> crate::signal::config::PowerSourceConfig {
    let circuits = fb.circuits().map(|v| v.iter().map(|c| {
        crate::signal::config::PowerCircuitConfig {
            name: c.name().unwrap_or("").to_string(),
            fraction: c.fraction(),
        }
    }).collect()).unwrap_or_default();
    let access = match fb.access_mode() {
        1 => {
            let names = fb.allow_list().map(|v| {
                v.iter().map(|s| s.to_string()).collect()
            }).unwrap_or_default();
            crate::signal::config::PowerAccessConfig::AllowList(names)
        }
        2 => crate::signal::config::PowerAccessConfig::Open,
        _ => crate::signal::config::PowerAccessConfig::OwnerOnly,
    };
    crate::signal::config::PowerSourceConfig { circuits, access }
}

fn encode_power_consumer<'a>(
    builder: &mut FlatBufferBuilder<'a>,
    con: &crate::signal::config::PowerConsumerConfig,
) -> flatbuffers::WIPOffset<fb::PowerConsumerConfigFB<'a>> {
    let circuit = builder.create_string(&con.circuit);
    let (rx, ry, rz, has) = match con.reactor_pos {
        Some(p) => (p.x, p.y, p.z, true),
        None => (0, 0, 0, false),
    };
    fb::PowerConsumerConfigFB::create(builder, &fb::PowerConsumerConfigFBArgs {
        reactor_x: rx, reactor_y: ry, reactor_z: rz,
        has_reactor: has, circuit: Some(circuit),
    })
}

fn decode_power_consumer(fb: &fb::PowerConsumerConfigFB<'_>) -> crate::signal::config::PowerConsumerConfig {
    let reactor_pos = if fb.has_reactor() {
        Some(glam::IVec3::new(fb.reactor_x(), fb.reactor_y(), fb.reactor_z()))
    } else {
        None
    };
    crate::signal::config::PowerConsumerConfig {
        reactor_pos,
        circuit: fb.circuit().unwrap_or("").to_string(),
    }
}

fn encode_nearby_reactors<'a>(
    builder: &mut FlatBufferBuilder<'a>,
    reactors: &[crate::signal::config::NearbyReactorInfo],
) -> Option<flatbuffers::WIPOffset<flatbuffers::Vector<'a, flatbuffers::ForwardsUOffset<fb::NearbyReactorFB<'a>>>>> {
    if reactors.is_empty() {
        return None;
    }
    let entries: Vec<_> = reactors.iter().map(|r| {
        let label = builder.create_string(&r.label);
        let circuit_strs: Vec<_> = r.circuits.iter().map(|c| builder.create_string(c)).collect();
        let circuits_vec = builder.create_vector(&circuit_strs);
        fb::NearbyReactorFB::create(builder, &fb::NearbyReactorFBArgs {
            pos_x: r.pos.x, pos_y: r.pos.y, pos_z: r.pos.z,
            label: Some(label), distance: r.distance,
            circuits: Some(circuits_vec),
        })
    }).collect();
    Some(builder.create_vector(&entries))
}

fn decode_nearby_reactors(
    v: Option<flatbuffers::Vector<'_, flatbuffers::ForwardsUOffset<fb::NearbyReactorFB<'_>>>>,
) -> Vec<crate::signal::config::NearbyReactorInfo> {
    v.map(|vec| vec.iter().map(|r| {
        crate::signal::config::NearbyReactorInfo {
            pos: glam::IVec3::new(r.pos_x(), r.pos_y(), r.pos_z()),
            label: r.label().unwrap_or("").to_string(),
            distance: r.distance(),
            circuits: r.circuits().map(|c| c.iter().map(|s| s.to_string()).collect()).unwrap_or_default(),
        }
    }).collect()).unwrap_or_default()
}

// ---------------------------------------------------------------------------
// Seat binding encode/decode (shared by BlockConfigState, BlockConfigUpdate, SeatBindingsNotify)
// ---------------------------------------------------------------------------

fn encode_seat_binding<'a>(
    builder: &mut FlatBufferBuilder<'a>,
    s: &crate::signal::config::SeatInputBindingConfig,
) -> flatbuffers::WIPOffset<fb::SeatBindingFB<'a>> {
    let lbl = builder.create_string(&s.label);
    let kn = builder.create_string(&s.key_name);
    let ch = builder.create_string(&s.channel_name);
    fb::SeatBindingFB::create(builder, &fb::SeatBindingFBArgs {
        label: Some(lbl),
        source: s.source as u8,
        key_name: Some(kn),
        key_mode: s.key_mode as u8,
        axis_direction: s.axis_direction as u8,
        channel_name: Some(ch),
        property: s.property as u8,
    })
}

fn decode_seat_binding(s: &fb::SeatBindingFB<'_>) -> crate::signal::config::SeatInputBindingConfig {
    use crate::signal::{AxisDirection, KeyMode, SeatInputSource};
    crate::signal::config::SeatInputBindingConfig {
        label: s.label().unwrap_or("").to_string(),
        source: SeatInputSource::from_u8(s.source()).unwrap_or(SeatInputSource::Key),
        key_name: s.key_name().unwrap_or("").to_string(),
        key_mode: KeyMode::from_u8(s.key_mode()).unwrap_or(KeyMode::Momentary),
        axis_direction: AxisDirection::from_u8(s.axis_direction()).unwrap_or(AxisDirection::Positive),
        channel_name: s.channel_name().unwrap_or("").to_string(),
        property: u8_to_signal_property(s.property()),
    }
}

fn encode_seat_bindings_vec<'a>(
    builder: &mut FlatBufferBuilder<'a>,
    bindings: &[crate::signal::config::SeatInputBindingConfig],
) -> flatbuffers::WIPOffset<flatbuffers::Vector<'a, flatbuffers::ForwardsUOffset<fb::SeatBindingFB<'a>>>> {
    let entries: Vec<_> = bindings.iter().map(|s| encode_seat_binding(builder, s)).collect();
    builder.create_vector(&entries)
}

fn decode_seat_bindings_vec(
    v: Option<flatbuffers::Vector<'_, flatbuffers::ForwardsUOffset<fb::SeatBindingFB<'_>>>>,
) -> Vec<crate::signal::config::SeatInputBindingConfig> {
    v.map(|vec| vec.iter().map(|s| decode_seat_binding(&s)).collect()).unwrap_or_default()
}

// ---------------------------------------------------------------------------
// Custom block config encode/decode
// ---------------------------------------------------------------------------

fn encode_flight_computer_config<'a>(
    builder: &mut FlatBufferBuilder<'a>,
    cfg: &crate::signal::config::FlightComputerConfig,
) -> flatbuffers::WIPOffset<fb::FlightComputerConfigFB<'a>> {
    let yc = builder.create_string(&cfg.yaw_cw_channel);
    let ycc = builder.create_string(&cfg.yaw_ccw_channel);
    let pu = builder.create_string(&cfg.pitch_up_channel);
    let pd = builder.create_string(&cfg.pitch_down_channel);
    let rc = builder.create_string(&cfg.roll_cw_channel);
    let rcc = builder.create_string(&cfg.roll_ccw_channel);
    let tc = builder.create_string(&cfg.toggle_channel);
    fb::FlightComputerConfigFB::create(builder, &fb::FlightComputerConfigFBArgs {
        yaw_cw_channel: Some(yc), yaw_ccw_channel: Some(ycc),
        pitch_up_channel: Some(pu), pitch_down_channel: Some(pd),
        roll_cw_channel: Some(rc), roll_ccw_channel: Some(rcc),
        toggle_channel: Some(tc),
        damping_gain: cfg.damping_gain, dead_zone: cfg.dead_zone, max_correction: cfg.max_correction,
    })
}

fn decode_flight_computer_config(fb: &fb::FlightComputerConfigFB<'_>) -> crate::signal::config::FlightComputerConfig {
    crate::signal::config::FlightComputerConfig {
        yaw_cw_channel: fb.yaw_cw_channel().unwrap_or("").into(),
        yaw_ccw_channel: fb.yaw_ccw_channel().unwrap_or("").into(),
        pitch_up_channel: fb.pitch_up_channel().unwrap_or("").into(),
        pitch_down_channel: fb.pitch_down_channel().unwrap_or("").into(),
        roll_cw_channel: fb.roll_cw_channel().unwrap_or("").into(),
        roll_ccw_channel: fb.roll_ccw_channel().unwrap_or("").into(),
        toggle_channel: fb.toggle_channel().unwrap_or("").into(),
        damping_gain: fb.damping_gain(), dead_zone: fb.dead_zone(), max_correction: fb.max_correction(),
    }
}

fn encode_hover_module_config<'a>(
    builder: &mut FlatBufferBuilder<'a>,
    cfg: &crate::signal::config::HoverModuleConfig,
) -> flatbuffers::WIPOffset<fb::HoverModuleConfigFB<'a>> {
    let tf = builder.create_string(&cfg.thrust_forward_channel);
    let tr = builder.create_string(&cfg.thrust_reverse_channel);
    let tri = builder.create_string(&cfg.thrust_right_channel);
    let tl = builder.create_string(&cfg.thrust_left_channel);
    let tu = builder.create_string(&cfg.thrust_up_channel);
    let td = builder.create_string(&cfg.thrust_down_channel);
    let yc = builder.create_string(&cfg.yaw_cw_channel);
    let ycc = builder.create_string(&cfg.yaw_ccw_channel);
    let pu = builder.create_string(&cfg.pitch_up_channel);
    let pd = builder.create_string(&cfg.pitch_down_channel);
    let rc = builder.create_string(&cfg.roll_cw_channel);
    let rcc = builder.create_string(&cfg.roll_ccw_channel);
    let ac = builder.create_string(&cfg.activate_channel);
    let cc = builder.create_string(&cfg.cutoff_channel);
    fb::HoverModuleConfigFB::create(builder, &fb::HoverModuleConfigFBArgs {
        thrust_forward_channel: Some(tf), thrust_reverse_channel: Some(tr),
        thrust_right_channel: Some(tri), thrust_left_channel: Some(tl),
        thrust_up_channel: Some(tu), thrust_down_channel: Some(td),
        yaw_cw_channel: Some(yc), yaw_ccw_channel: Some(ycc),
        pitch_up_channel: Some(pu), pitch_down_channel: Some(pd),
        roll_cw_channel: Some(rc), roll_ccw_channel: Some(rcc),
        activate_channel: Some(ac), cutoff_channel: Some(cc),
    })
}

fn decode_hover_module_config(fb: &fb::HoverModuleConfigFB<'_>) -> crate::signal::config::HoverModuleConfig {
    crate::signal::config::HoverModuleConfig {
        thrust_forward_channel: fb.thrust_forward_channel().unwrap_or("").into(),
        thrust_reverse_channel: fb.thrust_reverse_channel().unwrap_or("").into(),
        thrust_right_channel: fb.thrust_right_channel().unwrap_or("").into(),
        thrust_left_channel: fb.thrust_left_channel().unwrap_or("").into(),
        thrust_up_channel: fb.thrust_up_channel().unwrap_or("").into(),
        thrust_down_channel: fb.thrust_down_channel().unwrap_or("").into(),
        yaw_cw_channel: fb.yaw_cw_channel().unwrap_or("").into(),
        yaw_ccw_channel: fb.yaw_ccw_channel().unwrap_or("").into(),
        pitch_up_channel: fb.pitch_up_channel().unwrap_or("").into(),
        pitch_down_channel: fb.pitch_down_channel().unwrap_or("").into(),
        roll_cw_channel: fb.roll_cw_channel().unwrap_or("").into(),
        roll_ccw_channel: fb.roll_ccw_channel().unwrap_or("").into(),
        activate_channel: fb.activate_channel().unwrap_or("").into(),
        cutoff_channel: fb.cutoff_channel().unwrap_or("").into(),
    }
}

fn encode_autopilot_config<'a>(
    builder: &mut FlatBufferBuilder<'a>,
    cfg: &crate::signal::config::AutopilotBlockConfig,
) -> flatbuffers::WIPOffset<fb::AutopilotConfigFB<'a>> {
    let yc = builder.create_string(&cfg.yaw_cw_channel);
    let ycc = builder.create_string(&cfg.yaw_ccw_channel);
    let pu = builder.create_string(&cfg.pitch_up_channel);
    let pd = builder.create_string(&cfg.pitch_down_channel);
    let rc = builder.create_string(&cfg.roll_cw_channel);
    let rcc = builder.create_string(&cfg.roll_ccw_channel);
    let ec = builder.create_string(&cfg.engage_channel);
    fb::AutopilotConfigFB::create(builder, &fb::AutopilotConfigFBArgs {
        yaw_cw_channel: Some(yc), yaw_ccw_channel: Some(ycc),
        pitch_up_channel: Some(pu), pitch_down_channel: Some(pd),
        roll_cw_channel: Some(rc), roll_ccw_channel: Some(rcc),
        engage_channel: Some(ec),
    })
}

fn decode_autopilot_config(fb: &fb::AutopilotConfigFB<'_>) -> crate::signal::config::AutopilotBlockConfig {
    crate::signal::config::AutopilotBlockConfig {
        yaw_cw_channel: fb.yaw_cw_channel().unwrap_or("").into(),
        yaw_ccw_channel: fb.yaw_ccw_channel().unwrap_or("").into(),
        pitch_up_channel: fb.pitch_up_channel().unwrap_or("").into(),
        pitch_down_channel: fb.pitch_down_channel().unwrap_or("").into(),
        roll_cw_channel: fb.roll_cw_channel().unwrap_or("").into(),
        roll_ccw_channel: fb.roll_ccw_channel().unwrap_or("").into(),
        engage_channel: fb.engage_channel().unwrap_or("").into(),
    }
}

fn encode_warp_computer_config<'a>(
    builder: &mut FlatBufferBuilder<'a>,
    cfg: &crate::signal::config::WarpComputerConfig,
) -> flatbuffers::WIPOffset<fb::WarpComputerConfigFB<'a>> {
    let cyc = builder.create_string(&cfg.cycle_channel);
    let acc = builder.create_string(&cfg.accept_channel);
    let can = builder.create_string(&cfg.cancel_channel);
    fb::WarpComputerConfigFB::create(builder, &fb::WarpComputerConfigFBArgs {
        cycle_channel: Some(cyc), accept_channel: Some(acc), cancel_channel: Some(can),
    })
}

fn decode_warp_computer_config(fb: &fb::WarpComputerConfigFB<'_>) -> crate::signal::config::WarpComputerConfig {
    crate::signal::config::WarpComputerConfig {
        cycle_channel: fb.cycle_channel().unwrap_or("").into(),
        accept_channel: fb.accept_channel().unwrap_or("").into(),
        cancel_channel: fb.cancel_channel().unwrap_or("").into(),
    }
}

fn encode_engine_controller_config<'a>(
    builder: &mut FlatBufferBuilder<'a>,
    cfg: &crate::signal::config::EngineControllerConfig,
) -> flatbuffers::WIPOffset<fb::EngineControllerConfigFB<'a>> {
    let tf = builder.create_string(&cfg.thrust_forward_channel);
    let tr = builder.create_string(&cfg.thrust_reverse_channel);
    let tri = builder.create_string(&cfg.thrust_right_channel);
    let tl = builder.create_string(&cfg.thrust_left_channel);
    let tu = builder.create_string(&cfg.thrust_up_channel);
    let td = builder.create_string(&cfg.thrust_down_channel);
    let yc = builder.create_string(&cfg.yaw_cw_channel);
    let ycc = builder.create_string(&cfg.yaw_ccw_channel);
    let pu = builder.create_string(&cfg.pitch_up_channel);
    let pd = builder.create_string(&cfg.pitch_down_channel);
    let rc = builder.create_string(&cfg.roll_cw_channel);
    let rcc = builder.create_string(&cfg.roll_ccw_channel);
    let tc = builder.create_string(&cfg.toggle_channel);
    fb::EngineControllerConfigFB::create(builder, &fb::EngineControllerConfigFBArgs {
        thrust_forward_channel: Some(tf), thrust_reverse_channel: Some(tr),
        thrust_right_channel: Some(tri), thrust_left_channel: Some(tl),
        thrust_up_channel: Some(tu), thrust_down_channel: Some(td),
        yaw_cw_channel: Some(yc), yaw_ccw_channel: Some(ycc),
        pitch_up_channel: Some(pu), pitch_down_channel: Some(pd),
        roll_cw_channel: Some(rc), roll_ccw_channel: Some(rcc),
        toggle_channel: Some(tc),
    })
}

fn decode_engine_controller_config(fb: &fb::EngineControllerConfigFB<'_>) -> crate::signal::config::EngineControllerConfig {
    crate::signal::config::EngineControllerConfig {
        thrust_forward_channel: fb.thrust_forward_channel().unwrap_or("").into(),
        thrust_reverse_channel: fb.thrust_reverse_channel().unwrap_or("").into(),
        thrust_right_channel: fb.thrust_right_channel().unwrap_or("").into(),
        thrust_left_channel: fb.thrust_left_channel().unwrap_or("").into(),
        thrust_up_channel: fb.thrust_up_channel().unwrap_or("").into(),
        thrust_down_channel: fb.thrust_down_channel().unwrap_or("").into(),
        yaw_cw_channel: fb.yaw_cw_channel().unwrap_or("").into(),
        yaw_ccw_channel: fb.yaw_ccw_channel().unwrap_or("").into(),
        pitch_up_channel: fb.pitch_up_channel().unwrap_or("").into(),
        pitch_down_channel: fb.pitch_down_channel().unwrap_or("").into(),
        roll_cw_channel: fb.roll_cw_channel().unwrap_or("").into(),
        roll_ccw_channel: fb.roll_ccw_channel().unwrap_or("").into(),
        toggle_channel: fb.toggle_channel().unwrap_or("").into(),
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
    /// Observer-only connection (no player entity, receives chunk data + WorldState).
    ObserverConnect { observer_name: String },
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
    /// Seat bindings notification (server → client when entering a seat).
    SeatBindingsNotify(SeatBindingsNotifyData),
    /// Sub-grid block assignment update (TCP, reliable).
    SubGridAssignmentUpdate(SubGridAssignmentData),
    /// Tear down a specific secondary connection (by shard_type + seed).
    ShardDisconnectNotify(handoff::ShardDisconnectNotify),
    /// Seamless promotion handoff: promote an already-open secondary to primary.
    /// Used for in-game transitions (launch, land, board, warp) where
    /// `ShardRedirect`'s tear-down/rebuild would cause a visible micro-freeze.
    ShardHandoff(handoff::ShardHandoff),
}

/// Seat bindings sent to client when player enters a seat.
#[derive(Debug, Clone)]
pub struct SeatBindingsNotifyData {
    pub bindings: Vec<crate::signal::config::SeatInputBindingConfig>,
    pub seated_channel_name: String,
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
    /// Supercruise active (C key toggle). Legacy — use seat_values for generic seat.
    pub cruise: bool,
    /// Atmosphere compensation (hover) active (H key toggle). Legacy — use seat_values.
    pub atmo_comp: bool,
    /// Per-binding float values for the generic seat system.
    /// Length matches the seat's binding count. Empty when walking or using legacy path.
    pub seat_values: Vec<f32>,
    /// Bit-packed action modifiers.
    /// See [`actions`] for flag bit layout. Zero preserves legacy behaviour;
    /// old clients transmit no value for this field and servers see `0`.
    pub actions_bits: u32,
}

/// Action-bit layout for [`PlayerInputData::actions_bits`].
///
/// Bits are stable once assigned — add new flags at the next unused bit,
/// never repurpose. FlatBuffers field default is `0`, so older wire
/// formats deserialize cleanly (no sprint/crouch).
pub mod input_action_bits {
    pub const SPRINT: u32 = 1 << 0;
    pub const CROUCH: u32 = 1 << 1;
    pub const PRONE: u32 = 1 << 2;
    pub const STANCE_CYCLE_UP: u32 = 1 << 3;
    pub const STANCE_CYCLE_DOWN: u32 = 1 << 4;
    pub const INTERACT_ALT: u32 = 1 << 5;
    // bits 6..31 reserved
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
    /// Deprecated: kept during migration to `entities`. New code reads `entities`.
    pub players: Vec<PlayerSnapshotData>,
    pub bodies: Vec<CelestialBodyData>,
    /// Deprecated: kept during migration to `entities`. New code reads `entities`.
    pub ships: Vec<ShipRenderData>,
    pub lighting: Option<LightingData>,
    pub game_time: f64,
    /// Server-authoritative warp target star index (0xFFFFFFFF = none).
    pub warp_target_star_index: u32,
    /// Server-authoritative autopilot state (None = autopilot inactive).
    pub autopilot: Option<AutopilotSnapshotData>,
    /// Mechanical sub-grid body transforms (ship-local, updated at 20Hz).
    pub sub_grids: Vec<SubGridTransformData>,
    /// Unified observable entities (ships, players, EVA) with LOD tiers.
    /// Source-of-truth going forward; the legacy `ships`/`players` fields
    /// mirror a subset for backward compatibility and will be removed.
    pub entities: Vec<ObservableEntityData>,
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

/// Kind of observable entity carried in `WorldStateData.entities`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum EntityKind {
    Ship = 0,
    EvaPlayer = 1,
    GroundedPlayer = 2,
    Seated = 3,
}

impl EntityKind {
    pub fn from_u8(v: u8) -> Self {
        match v {
            1 => EntityKind::EvaPlayer,
            2 => EntityKind::GroundedPlayer,
            3 => EntityKind::Seated,
            _ => EntityKind::Ship,
        }
    }

    pub fn as_u8(self) -> u8 {
        self as u8
    }

    pub fn is_player(self) -> bool {
        matches!(
            self,
            EntityKind::EvaPlayer | EntityKind::GroundedPlayer | EntityKind::Seated
        )
    }
}

/// LOD tier for an observable entity, derived from distance to observer.
/// Full = close-range full fidelity; Reduced = medium; Coarse = far point/sprite.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum LodTier {
    Full = 0,
    Reduced = 1,
    Coarse = 2,
}

impl LodTier {
    pub fn from_u8(v: u8) -> Self {
        match v {
            1 => LodTier::Reduced,
            2 => LodTier::Coarse,
            _ => LodTier::Full,
        }
    }

    pub fn as_u8(self) -> u8 {
        self as u8
    }
}

/// Unified observable entity for multi-shard AOI visibility.
/// Produced by the system-shard AOI pass (for ships, EVA players, and surface
/// player aggregates) and by ship/planet shards for their local players.
#[derive(Debug, Clone)]
pub struct ObservableEntityData {
    /// Ship id for ships, session-token u64 for players.
    pub entity_id: u64,
    pub kind: EntityKind,
    /// Position in the WorldState origin frame (client adds origin for render).
    pub position: DVec3,
    pub rotation: DQuat,
    pub velocity: DVec3,
    /// Approximate bounding radius (meters) — drives frustum culling and LOD.
    pub bounding_radius: f32,
    pub lod_tier: LodTier,
    /// Authoritative shard id (0 if no secondary upgrade is beneficial).
    pub shard_id: u64,
    /// Shard type for secondary-connection routing (0=Planet, 1=System, 2=Ship, 3=Galaxy).
    pub shard_type: u8,
    /// True for the player's own avatar / ship entry.
    pub is_own: bool,
    /// Display name. Empty for ships without a pilot context.
    pub name: String,
    /// Health (0 for non-player kinds).
    pub health: f32,
    /// Shield (0 for non-player kinds).
    pub shield: f32,
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
    pub seated: bool,
}

/// Transform of a mechanical sub-grid body (rotor, piston, hinge, slider).
/// Ship-local coordinates — f32 precision is sufficient.
#[derive(Debug, Clone)]
pub struct SubGridTransformData {
    pub sub_grid_id: u32,
    pub translation: glam::Vec3,
    pub rotation: glam::Quat,
    pub parent_grid: u32,
    /// Original root-space anchor position (mount_pos + face offset). Never changes.
    pub anchor: glam::Vec3,
    /// Mount block position (for mechanism arm rendering + collision).
    pub mount_pos: glam::IVec3,
    /// Face direction 0-5 (determines arm axis).
    pub mount_face: u8,
    /// Joint type: 0 = revolute (rotor), 1 = prismatic (piston).
    pub joint_type: u8,
    /// Current angle (degrees, revolute) or extension (meters, prismatic).
    pub current_value: f32,
}

/// Block-to-sub-grid assignment update (sent via TCP on join + on change).
#[derive(Debug, Clone)]
pub struct SubGridAssignmentData {
    pub assignments: Vec<(glam::IVec3, u32)>,
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

fn to_fb_vec3f(v: &glam::Vec3) -> fb::Vec3f {
    fb::Vec3f::new(v.x, v.y, v.z)
}

fn from_fb_vec3f(v: &fb::Vec3f) -> glam::Vec3 {
    glam::Vec3::new(v.x(), v.y(), v.z())
}

fn to_fb_quatf(q: &glam::Quat) -> fb::Quatf {
    fb::Quatf::new(q.x, q.y, q.z, q.w)
}

fn from_fb_quatf(q: &fb::Quatf) -> glam::Quat {
    glam::Quat::from_xyzw(q.x(), q.y(), q.z(), q.w())
}

/// Encode an `ObservableEntityData` slice into a FlatBuffers vector.
/// Returns `None` when the input is empty so we don't emit a zero-length vector.
pub(crate) fn encode_observable_entities<'a>(
    builder: &mut FlatBufferBuilder<'a>,
    entities: &[ObservableEntityData],
) -> Option<
    flatbuffers::WIPOffset<
        flatbuffers::Vector<'a, flatbuffers::ForwardsUOffset<fb::ObservableEntity<'a>>>,
    >,
> {
    if entities.is_empty() {
        return None;
    }
    let offsets: Vec<_> = entities
        .iter()
        .map(|e| {
            let name = builder.create_string(&e.name);
            let pos = to_fb_vec3d(&e.position);
            let rot = to_fb_quatd(&e.rotation);
            let vel = to_fb_vec3d(&e.velocity);
            fb::ObservableEntity::create(
                builder,
                &fb::ObservableEntityArgs {
                    entity_id: e.entity_id,
                    kind: fb::EntityKind(e.kind.as_u8() as i8),
                    position: Some(&pos),
                    rotation: Some(&rot),
                    velocity: Some(&vel),
                    bounding_radius: e.bounding_radius,
                    lod_tier: e.lod_tier.as_u8(),
                    shard_id: e.shard_id,
                    shard_type: e.shard_type,
                    is_own: e.is_own,
                    name: Some(name),
                    health: e.health,
                    shield: e.shield,
                },
            )
        })
        .collect();
    Some(builder.create_vector(&offsets))
}

/// Decode a FlatBuffers `entities` vector into `ObservableEntityData`.
pub(crate) fn decode_observable_entities(
    entities: Option<
        flatbuffers::Vector<'_, flatbuffers::ForwardsUOffset<fb::ObservableEntity<'_>>>,
    >,
) -> Vec<ObservableEntityData> {
    let Some(v) = entities else { return Vec::new() };
    v.iter()
        .map(|e| {
            let pos = e.position().map(from_fb_vec3d).unwrap_or(DVec3::ZERO);
            let rot = e.rotation().map(from_fb_quatd).unwrap_or(DQuat::IDENTITY);
            let vel = e.velocity().map(from_fb_vec3d).unwrap_or(DVec3::ZERO);
            ObservableEntityData {
                entity_id: e.entity_id(),
                kind: EntityKind::from_u8(e.kind().0 as u8),
                position: pos,
                rotation: rot,
                velocity: vel,
                bounding_radius: e.bounding_radius(),
                lod_tier: LodTier::from_u8(e.lod_tier()),
                shard_id: e.shard_id(),
                shard_type: e.shard_type(),
                is_own: e.is_own(),
                name: e.name().unwrap_or("").to_string(),
                health: e.health(),
                shield: e.shield(),
            }
        })
        .collect()
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
                let sv = if data.seat_values.is_empty() {
                    None
                } else {
                    Some(builder.create_vector(&data.seat_values))
                };
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
                        cruise: data.cruise,
                        atmo_comp: data.atmo_comp,
                        seat_values: sv,
                        actions_bits: data.actions_bits,
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
                let stv = encode_seat_bindings_vec(&mut builder, &data.seat_mappings);
                let pv = builder.create_vector(&pub_b);
                let sv = builder.create_vector(&sub_b);
                let rv = builder.create_vector(&rules);
                let ps = data.power_source.as_ref().map(|s| encode_power_source(&mut builder, s));
                let pc = data.power_consumer.as_ref().map(|c| encode_power_consumer(&mut builder, c));
                let seated_ch = if data.seated_channel_name.is_empty() { None } else { Some(builder.create_string(&data.seated_channel_name)) };
                let fc = data.flight_computer.as_ref().map(|c| encode_flight_computer_config(&mut builder, c));
                let hm = data.hover_module.as_ref().map(|c| encode_hover_module_config(&mut builder, c));
                let ap = data.autopilot.as_ref().map(|c| encode_autopilot_config(&mut builder, c));
                let wc = data.warp_computer.as_ref().map(|c| encode_warp_computer_config(&mut builder, c));
                let ec = data.engine_controller.as_ref().map(|c| encode_engine_controller_config(&mut builder, c));
                let bcu = fb::BlockConfigUpdate::create(&mut builder, &fb::BlockConfigUpdateArgs {
                    block_x: data.block_pos.x, block_y: data.block_pos.y, block_z: data.block_pos.z,
                    publish_bindings: Some(pv), subscribe_bindings: Some(sv),
                    converter_rules: Some(rv), seat_mappings: Some(stv),
                    power_source: ps, power_consumer: pc,
                    seated_channel_name: seated_ch,
                    flight_computer_config: fc, hover_module_config: hm,
                    autopilot_config: ap, warp_computer_config: wc, engine_controller_config: ec,
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
            ClientMsg::ObserverConnect { observer_name } => {
                let name = builder.create_string(observer_name);
                let oc = fb::ObserverConnect::create(
                    &mut builder,
                    &fb::ObserverConnectArgs {
                        observer_name: Some(name),
                    },
                );
                let msg = fb::ClientMessage::create(&mut builder, &fb::ClientMessageArgs {
                    payload_type: fb::ClientPayload::ObserverConnect,
                    payload: Some(oc.as_union_value()),
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
                    cruise: p.cruise(),
                    atmo_comp: p.atmo_comp(),
                    seat_values: p.seat_values().map(|v| v.iter().collect()).unwrap_or_default(),
                    actions_bits: p.actions_bits(),
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
                let seats = decode_seat_bindings_vec(bcu.seat_mappings());
                let power_source = bcu.power_source().map(|ps| decode_power_source(&ps));
                let power_consumer = bcu.power_consumer().map(|pc| decode_power_consumer(&pc));
                Ok(ClientMsg::BlockConfigUpdate(crate::signal::config::BlockConfigUpdateData {
                    block_pos: glam::IVec3::new(bcu.block_x(), bcu.block_y(), bcu.block_z()),
                    publish_bindings: pub_b,
                    subscribe_bindings: sub_b,
                    converter_rules: rules,
                    seat_mappings: seats,
                    seated_channel_name: bcu.seated_channel_name().unwrap_or("").to_string(),
                    power_source,
                    power_consumer,
                    flight_computer: bcu.flight_computer_config().map(|c| decode_flight_computer_config(&c)),
                    hover_module: bcu.hover_module_config().map(|c| decode_hover_module_config(&c)),
                    autopilot: bcu.autopilot_config().map(|c| decode_autopilot_config(&c)),
                    warp_computer: bcu.warp_computer_config().map(|c| decode_warp_computer_config(&c)),
                    engine_controller: bcu.engine_controller_config().map(|c| decode_engine_controller_config(&c)),
                    mechanical: None, // TODO: decode from FlatBuffers when schema is extended
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
            fb::ClientPayload::ObserverConnect => {
                let oc = msg
                    .payload_as_observer_connect()
                    .ok_or(MessageError::MissingField("ObserverConnect payload"))?;
                Ok(ClientMsg::ObserverConnect {
                    observer_name: oc.observer_name().unwrap_or("").to_string(),
                })
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
                        target_shard_type: data.target_shard_type,
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
                        seated: p.seated,
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

                let sg_fbs: Vec<_> = data.sub_grids.iter().map(|sg| {
                    let t = to_fb_vec3f(&sg.translation);
                    let r = to_fb_quatf(&sg.rotation);
                    let a = to_fb_vec3f(&sg.anchor);
                    fb::SubGridTransform::create(&mut builder, &fb::SubGridTransformArgs {
                        sub_grid_id: sg.sub_grid_id,
                        translation: Some(&t),
                        rotation: Some(&r),
                        parent_grid: sg.parent_grid,
                        anchor: Some(&a),
                        mount_x: sg.mount_pos.x,
                        mount_y: sg.mount_pos.y,
                        mount_z: sg.mount_pos.z,
                        mount_face: sg.mount_face,
                        joint_type: sg.joint_type,
                        current_value: sg.current_value,
                    })
                }).collect();
                let sub_grids_vec = if sg_fbs.is_empty() { None } else { Some(builder.create_vector(&sg_fbs)) };

                let entities_vec = encode_observable_entities(&mut builder, &data.entities);

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
                    sub_grids: sub_grids_vec,
                    entities: entities_vec,
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
                    shard_id: data.shard_id,
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
                let seats_vec = encode_seat_bindings_vec(&mut builder, &data.seat_mappings);
                let channels: Vec<_> = data.available_channels.iter()
                    .map(|c| builder.create_string(c)).collect();

                let pub_vec = builder.create_vector(&pub_bindings);
                let sub_vec = builder.create_vector(&sub_bindings);
                let rules_vec = builder.create_vector(&rules);
                let channels_vec = builder.create_vector(&channels);
                let ps = data.power_source.as_ref().map(|s| encode_power_source(&mut builder, s));
                let pc = data.power_consumer.as_ref().map(|c| encode_power_consumer(&mut builder, c));
                let nr = encode_nearby_reactors(&mut builder, &data.nearby_reactors);
                let seated_ch = if data.seated_channel_name.is_empty() { None } else { Some(builder.create_string(&data.seated_channel_name)) };
                let fc = data.flight_computer.as_ref().map(|c| encode_flight_computer_config(&mut builder, c));
                let hm = data.hover_module.as_ref().map(|c| encode_hover_module_config(&mut builder, c));
                let ap = data.autopilot.as_ref().map(|c| encode_autopilot_config(&mut builder, c));
                let wc = data.warp_computer.as_ref().map(|c| encode_warp_computer_config(&mut builder, c));
                let ec = data.engine_controller.as_ref().map(|c| encode_engine_controller_config(&mut builder, c));

                let bcs = fb::BlockConfigState::create(&mut builder, &fb::BlockConfigStateArgs {
                    block_x: data.block_pos.x, block_y: data.block_pos.y, block_z: data.block_pos.z,
                    block_type: data.block_type, kind: data.kind,
                    publish_bindings: Some(pub_vec), subscribe_bindings: Some(sub_vec),
                    converter_rules: Some(rules_vec), seat_mappings: Some(seats_vec),
                    available_channels: Some(channels_vec),
                    power_source: ps, power_consumer: pc, nearby_reactors: nr,
                    seated_channel_name: seated_ch,
                    flight_computer_config: fc, hover_module_config: hm,
                    autopilot_config: ap, warp_computer_config: wc, engine_controller_config: ec,
                });
                let msg = fb::ServerMessage::create(&mut builder, &fb::ServerMessageArgs {
                    payload_type: fb::ServerPayload::BlockConfigState,
                    payload: Some(bcs.as_union_value()),
                });
                builder.finish(msg, None);
            }
            ServerMsg::SeatBindingsNotify(data) => {
                let bv = encode_seat_bindings_vec(&mut builder, &data.bindings);
                let seated_ch = if data.seated_channel_name.is_empty() { None } else { Some(builder.create_string(&data.seated_channel_name)) };
                let sbn = fb::SeatBindingsNotify::create(&mut builder, &fb::SeatBindingsNotifyArgs {
                    bindings: Some(bv),
                    seated_channel_name: seated_ch,
                });
                let msg = fb::ServerMessage::create(&mut builder, &fb::ServerMessageArgs {
                    payload_type: fb::ServerPayload::SeatBindingsNotify,
                    payload: Some(sbn.as_union_value()),
                });
                builder.finish(msg, None);
            }
            ServerMsg::SubGridAssignmentUpdate(data) => {
                let assign_fbs: Vec<_> = data.assignments.iter().map(|(pos, sg_id)| {
                    fb::SubGridBlockAssignment::create(&mut builder, &fb::SubGridBlockAssignmentArgs {
                        bx: pos.x,
                        by: pos.y,
                        bz: pos.z,
                        sub_grid_id: *sg_id,
                    })
                }).collect();
                let assignments = builder.create_vector(&assign_fbs);
                let upd = fb::SubGridAssignmentUpdate::create(&mut builder, &fb::SubGridAssignmentUpdateArgs {
                    assignments: Some(assignments),
                });
                let msg = fb::ServerMessage::create(&mut builder, &fb::ServerMessageArgs {
                    payload_type: fb::ServerPayload::SubGridAssignmentUpdate,
                    payload: Some(upd.as_union_value()),
                });
                builder.finish(msg, None);
            }
            ServerMsg::ShardHandoff(data) => {
                let tcp = builder.create_string(&data.target_tcp_addr);
                let udp = builder.create_string(&data.target_udp_addr);
                let pos = to_fb_vec3d(&data.handoff_position);
                let vel = to_fb_vec3d(&data.handoff_velocity);
                let rot = to_fb_quatd(&data.handoff_rotation);
                let sh = fb::ShardHandoffMsg::create(
                    &mut builder,
                    &fb::ShardHandoffMsgArgs {
                        session_token: data.session_token.0,
                        promote_shard_type: data.promote_shard_type,
                        promote_shard_id: data.promote_shard_id.0,
                        target_tcp_addr: Some(tcp),
                        target_udp_addr: Some(udp),
                        source_demote_after_ticks: data.source_demote_after_ticks,
                        handoff_position: Some(&pos),
                        handoff_velocity: Some(&vel),
                        handoff_rotation: Some(&rot),
                    },
                );
                let msg = fb::ServerMessage::create(
                    &mut builder,
                    &fb::ServerMessageArgs {
                        payload_type: fb::ServerPayload::ShardHandoffMsg,
                        payload: Some(sh.as_union_value()),
                    },
                );
                builder.finish(msg, None);
            }
            ServerMsg::ShardDisconnectNotify(data) => {
                let dn = fb::ShardDisconnectNotify::create(&mut builder, &fb::ShardDisconnectNotifyArgs {
                    shard_type: data.shard_type,
                    seed: data.seed,
                });
                let msg = fb::ServerMessage::create(&mut builder, &fb::ServerMessageArgs {
                    payload_type: fb::ServerPayload::ShardDisconnectNotify,
                    payload: Some(dn.as_union_value()),
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
                    target_shard_type: sr.target_shard_type(),
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
                            seated: p.seated(),
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

                let sub_grids = ws.sub_grids().map(|v| v.iter().map(|sg| {
                    let t = sg.translation().map(|v| from_fb_vec3f(v)).unwrap_or(glam::Vec3::ZERO);
                    let r = sg.rotation().map(|v| from_fb_quatf(v)).unwrap_or(glam::Quat::IDENTITY);
                    let a = sg.anchor().map(|v| from_fb_vec3f(v)).unwrap_or(glam::Vec3::ZERO);
                    SubGridTransformData {
                        sub_grid_id: sg.sub_grid_id(),
                        translation: t,
                        rotation: r,
                        parent_grid: sg.parent_grid(),
                        anchor: a,
                        mount_pos: glam::IVec3::new(sg.mount_x(), sg.mount_y(), sg.mount_z()),
                        mount_face: sg.mount_face(),
                        joint_type: sg.joint_type(),
                        current_value: sg.current_value(),
                    }
                }).collect()).unwrap_or_default();

                let entities = decode_observable_entities(ws.entities());

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
                    sub_grids,
                    entities,
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
                    shard_id: pc.shard_id(),
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
                let seats = decode_seat_bindings_vec(bcs.seat_mappings());
                let channels = bcs.available_channels().map(|v| {
                    v.iter().map(|s| s.to_string()).collect()
                }).unwrap_or_default();
                let power_source = bcs.power_source().map(|ps| decode_power_source(&ps));
                let power_consumer = bcs.power_consumer().map(|pc| decode_power_consumer(&pc));
                let nearby_reactors = decode_nearby_reactors(bcs.nearby_reactors());
                Ok(ServerMsg::BlockConfigState(crate::signal::config::BlockSignalConfig {
                    block_pos: glam::IVec3::new(bcs.block_x(), bcs.block_y(), bcs.block_z()),
                    block_type: bcs.block_type(),
                    kind: bcs.kind(),
                    publish_bindings: pub_b,
                    subscribe_bindings: sub_b,
                    converter_rules: rules,
                    seat_mappings: seats,
                    seated_channel_name: bcs.seated_channel_name().unwrap_or("").to_string(),
                    available_channels: channels,
                    power_source,
                    power_consumer,
                    nearby_reactors,
                    flight_computer: bcs.flight_computer_config().map(|c| decode_flight_computer_config(&c)),
                    hover_module: bcs.hover_module_config().map(|c| decode_hover_module_config(&c)),
                    autopilot: bcs.autopilot_config().map(|c| decode_autopilot_config(&c)),
                    warp_computer: bcs.warp_computer_config().map(|c| decode_warp_computer_config(&c)),
                    engine_controller: bcs.engine_controller_config().map(|c| decode_engine_controller_config(&c)),
                    mechanical: None, // TODO: decode from FlatBuffers when schema is extended
                }))
            }
            fb::ServerPayload::SeatBindingsNotify => {
                let sbn = msg.payload_as_seat_bindings_notify()
                    .ok_or(MessageError::MissingField("SeatBindingsNotify payload"))?;
                Ok(ServerMsg::SeatBindingsNotify(SeatBindingsNotifyData {
                    bindings: decode_seat_bindings_vec(sbn.bindings()),
                    seated_channel_name: sbn.seated_channel_name().unwrap_or("").to_string(),
                }))
            }
            fb::ServerPayload::SubGridAssignmentUpdate => {
                let upd = msg.payload_as_sub_grid_assignment_update()
                    .ok_or(MessageError::MissingField("SubGridAssignmentUpdate payload"))?;
                let assignments = upd.assignments().map(|v| v.iter().map(|a| {
                    (glam::IVec3::new(a.bx(), a.by(), a.bz()), a.sub_grid_id())
                }).collect()).unwrap_or_default();
                Ok(ServerMsg::SubGridAssignmentUpdate(SubGridAssignmentData { assignments }))
            }
            fb::ServerPayload::ShardDisconnectNotify => {
                let dn = msg
                    .payload_as_shard_disconnect_notify()
                    .ok_or(MessageError::MissingField("ShardDisconnectNotify payload"))?;
                Ok(ServerMsg::ShardDisconnectNotify(handoff::ShardDisconnectNotify {
                    shard_type: dn.shard_type(),
                    seed: dn.seed(),
                }))
            }
            fb::ServerPayload::ShardHandoffMsg => {
                let sh = msg
                    .payload_as_shard_handoff_msg()
                    .ok_or(MessageError::MissingField("ShardHandoffMsg payload"))?;
                let pos = sh
                    .handoff_position()
                    .map(from_fb_vec3d)
                    .unwrap_or(DVec3::ZERO);
                let vel = sh
                    .handoff_velocity()
                    .map(from_fb_vec3d)
                    .unwrap_or(DVec3::ZERO);
                let rot = sh
                    .handoff_rotation()
                    .map(from_fb_quatd)
                    .unwrap_or(DQuat::IDENTITY);
                Ok(ServerMsg::ShardHandoff(handoff::ShardHandoff {
                    session_token: SessionToken(sh.session_token()),
                    promote_shard_type: sh.promote_shard_type(),
                    promote_shard_id: ShardId(sh.promote_shard_id()),
                    target_tcp_addr: sh
                        .target_tcp_addr()
                        .ok_or(MessageError::MissingField("target_tcp_addr"))?
                        .to_string(),
                    target_udp_addr: sh
                        .target_udp_addr()
                        .ok_or(MessageError::MissingField("target_udp_addr"))?
                        .to_string(),
                    source_demote_after_ticks: sh.source_demote_after_ticks(),
                    handoff_position: pos,
                    handoff_velocity: vel,
                    handoff_rotation: rot,
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
            target_shard_type: 2,
        });
        let bytes = msg.serialize();
        let decoded = ServerMsg::deserialize(&bytes).unwrap();

        if let ServerMsg::ShardRedirect(sr) = decoded {
            assert_eq!(sr.session_token, SessionToken(555));
            assert_eq!(sr.target_tcp_addr, "127.0.0.1:7777");
            assert_eq!(sr.target_udp_addr, "127.0.0.1:7778");
            assert_eq!(sr.shard_id, ShardId(10));
            assert_eq!(sr.target_shard_type, 2);
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
            cruise: false,
            atmo_comp: false,
            seat_values: vec![0.5, 1.0, 0.0],
            actions_bits: input_action_bits::SPRINT,
        });
        let bytes = msg.serialize();
        let decoded = ClientMsg::deserialize(&bytes).unwrap();

        if let ClientMsg::PlayerInput(p) = decoded {
            assert!((p.movement[0] - 1.0).abs() < 1e-5);
            assert!(p.jump);
            assert_eq!(p.speed_tier, 2);
            assert_eq!(p.tick, 1000);
            assert_eq!(p.seat_values.len(), 3);
            assert!((p.seat_values[0] - 0.5).abs() < 1e-5);
            assert_eq!(p.actions_bits, input_action_bits::SPRINT);
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
                seated: false,
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
            sub_grids: vec![SubGridTransformData {
                sub_grid_id: 1,
                translation: glam::Vec3::new(2.5, 0.5, -3.5),
                rotation: glam::Quat::from_rotation_y(std::f32::consts::FRAC_PI_4),
                parent_grid: 0,
                anchor: glam::Vec3::new(2.5, 0.5, -3.5),
                mount_pos: glam::IVec3::new(2, 0, -4),
                mount_face: 2,
                joint_type: 0,
                current_value: 45.0,
            }],
            entities: vec![ObservableEntityData {
                entity_id: 7,
                kind: EntityKind::Ship,
                position: DVec3::new(100.0, 150000.0, 0.0),
                rotation: DQuat::IDENTITY,
                velocity: DVec3::new(5.0, 0.0, 0.0),
                bounding_radius: 32.0,
                lod_tier: LodTier::Reduced,
                shard_id: 42,
                shard_type: 2,
                is_own: false,
                name: String::new(),
                health: 0.0,
                shield: 0.0,
            }],
        });
        let bytes = msg.serialize();
        let decoded = ServerMsg::deserialize(&bytes).unwrap();
        if let ServerMsg::WorldState(ws) = decoded {
            assert_eq!(ws.tick, 500);
            assert_eq!(ws.players.len(), 1);
            assert_eq!(ws.players[0].player_id, 1);
            assert!((ws.players[0].health - 95.0).abs() < 1e-5);
            assert!(ws.autopilot.is_none());
            assert_eq!(ws.sub_grids.len(), 1);
            assert_eq!(ws.sub_grids[0].sub_grid_id, 1);
            assert!((ws.sub_grids[0].translation.x - 2.5).abs() < 1e-5);
            assert_eq!(ws.sub_grids[0].parent_grid, 0);
            assert_eq!(ws.entities.len(), 1);
            assert_eq!(ws.entities[0].entity_id, 7);
            assert_eq!(ws.entities[0].kind, EntityKind::Ship);
            assert_eq!(ws.entities[0].lod_tier, LodTier::Reduced);
            assert_eq!(ws.entities[0].shard_id, 42);
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
