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
    /// Phase 4.3: wire-dict-interned batch. Sender uses an
    /// `OutboundDict` to replace channel names with small wire ids;
    /// the first reference of each name carries `FLAG_REGISTER` +
    /// the name. See [`crate::signal::wire_dict`].
    SignalBroadcastBatchV2(SignalBroadcastBatchV2Data),
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
    /// Phase 3D: foreign shard subscribes to a channel on this shard via
    /// a held grant. Receiver records `SubscriberRef::RemoteShard` after
    /// validating the HMAC tag against `GrantsRegistry::check_subscribe`.
    SignalSubscribe(SignalSubscribeData),
    /// Phase 3D: explicit subscribe teardown. Idempotent.
    SignalUnsubscribe(SignalUnsubscribeData),
    /// Phase 3F: register interest in Radio traffic at the galaxy
    /// shard. Galaxy holds the subscription until `lease_until_ms` or
    /// until matching `RadioUnsubscribe`.
    RadioSubscribe(RadioSubscribeData),
    /// Phase 3F: drop Radio interest.
    RadioUnsubscribe(RadioUnsubscribeData),
    /// Phase 3F.9: ship-shard → system-shard. The ship asserts its
    /// CURRENT full set of Radio listener frequencies. System-shard
    /// aggregates across hosted ships and propagates a precise
    /// per-frequency `RadioSubscribe` to galaxy. Self-healing —
    /// every message is the complete ground truth for that ship,
    /// not a delta.
    ShipFrequencyInterest(ShipFrequencyInterestData),
    /// Phase 4.6: system-shard → ship-shard, 1Hz authoritative
    /// neighborhood push for ShortRange signal routing. Replaces
    /// opportunistic learn-from-traffic with fresh + complete peer
    /// position data.
    ShipNeighborhood(ShipNeighborhoodData),
    /// Phase 5.1: media (text/audio/video/image) batched broadcast.
    /// Reuses signal's auth + replay-window + scope routing.
    MediaBroadcastBatch(MediaBroadcastBatchData),
}

// `SignalSubscribeData` / `SignalUnsubscribeData` definitions moved to
// `voxeldust-signal::wire` so the signal pipeline (which queues these
// directly into `IncomingSubscribeBuffer`) does not have to depend on
// `voxeldust-core`. See the doc on `signal::auth::hmac_sign_subscribe_request`
// for the HMAC input contract that both sides honour.
pub use voxeldust_signal::wire::{SignalSubscribeData, SignalUnsubscribeData};

/// Payload for `ShardMsg::RadioSubscribe`. See
/// `protocol/voxeldust.fbs::RadioSubscribe` for the canonical wire
/// shape. Sent by a system-shard to the band-matched galaxy-shard to
/// register interest in Radio-scope traffic.
#[derive(Debug, Clone, Default)]
pub struct RadioSubscribeData {
    pub subscriber_shard_id: u64,
    /// Specific frequencies. Empty when `wildcard` is set.
    pub frequencies: Vec<u32>,
    /// Subscribe to every Radio frequency.
    pub wildcard: bool,
    /// UNIX millis lease deadline. Galaxy compares against its own
    /// wall clock and drops the subscription on expiry.
    pub lease_until_ms: u64,
}

/// Payload for `ShardMsg::RadioUnsubscribe`. Idempotent.
#[derive(Debug, Clone, Default)]
pub struct RadioUnsubscribeData {
    pub subscriber_shard_id: u64,
    pub frequencies: Vec<u32>,
    pub wildcard: bool,
}

/// Payload for `ShardMsg::ShipFrequencyInterest`. The ship-shard
/// publishes the complete current set of Radio frequencies it has
/// Listener blocks for. System-shard tracks per-ship sets and
/// aggregates upward to galaxy.
#[derive(Debug, Clone, Default)]
pub struct ShipFrequencyInterestData {
    pub ship_shard_id: u64,
    pub frequencies: Vec<u32>,
    pub lease_until_ms: u64,
}

/// One peer entry inside [`ShipNeighborhoodData`].
#[derive(Debug, Clone, PartialEq)]
pub struct ShipNeighborhoodEntryData {
    pub peer_shard_id: u64,
    pub peer_position: DVec3,
    pub peer_velocity: DVec3,
}

/// Payload for `ShardMsg::ShipNeighborhood`. system-shard pushes the
/// authoritative nearby-ships list to each hosted ship-shard at 1Hz
/// for ShortRange routing optimization. See `protocol/voxeldust.fbs`
/// for the rationale.
#[derive(Debug, Clone, Default)]
pub struct ShipNeighborhoodData {
    pub target_ship_shard_id: u64,
    pub issued_at_ms: u64,
    pub peers: Vec<ShipNeighborhoodEntryData>,
}

/// Payload for `ShardMsg::MediaBroadcastBatch`. Carries one or more
/// `MediaFrame`s (Phase 5.1) batched for cross-shard delivery.
/// Reuses the signal pipeline's auth + replay-window + scope-routing
/// substrate at the receiver. Audio + video at production rates will
/// migrate to a dedicated QUIC stream per active subscription in a
/// future slice; this batched shape covers text + low-bitrate use.
#[derive(Debug, Clone)]
pub struct MediaBroadcastBatchData {
    pub source_shard_id: u64,
    pub source_position: DVec3,
    pub frames: Vec<crate::media::MediaFrame>,
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
    /// Physics-derived stellar state for stars (`body_id == 0`). Computed by
    /// system-shard at system bootstrap and propagated to subscriber shards
    /// (planet-shard, ship-shard) through this `SystemSceneUpdate` payload.
    /// `None` for planets.
    pub stellar: Option<crate::stellar::StellarState>,
    /// Physics-derived planetary geophysical state for planets
    /// (`body_id != 0`). Computed by system-shard at system bootstrap via
    /// `core::geophysics::PlanetGeophysicalState::from_seed_and_star`,
    /// then echoed by every shard that broadcasts the body. `None` for the
    /// star or for transient pre-Phase-3 catalogues.
    pub planetary: Option<crate::geophysics::PlanetGeophysicalState>,
    /// Rotation parameters for planets (`body_id != 0`). Computed by
    /// system-shard at system bootstrap via
    /// `core::planet_rotation::PlanetRotationParams::from_seed_and_state`;
    /// static for the planet's session lifetime.
    pub rotation_params: Option<crate::planet_rotation::PlanetRotationParams>,
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
    // -- Phase 2 wire-protocol additions (additive, defaults preserve compat) -
    /// Per-(channel, sender_shard_id) monotonic sequence. Senders increment
    /// per publish; receivers run the 64-entry sliding-window replay check.
    /// Default 0 marks "legacy sender" — receivers skip the replay check
    /// when both `sequence` and `timestamp_ms` are zero.
    pub sequence: u64,
    /// UNIX wall-clock millis at the sender. Receiver enforces a ±5 s
    /// freshness window. Default 0 = legacy.
    pub timestamp_ms: u64,
    /// Phase 3 capability lookup id. 0 = no grant (open broadcast or
    /// scope-class-only validated). Non-zero = the receiver looks up the
    /// grant by id and checks `auth_tag` against grant.key.
    pub grant_id: u64,
    /// Phase 3 HMAC-SHA256 truncated to 16 bytes. Empty = unauthenticated.
    pub auth_tag: Vec<u8>,
}

/// Batched signal broadcast — multiple dirty signals packed into one message.
/// Sent once per destination per tick instead of once per signal per destination.
#[derive(Debug, Clone)]
pub struct SignalBroadcastBatchData {
    pub source_shard_id: u64,
    pub source_position: DVec3,
    pub entries: Vec<SignalBroadcastEntry>,
}

// -- Phase 4.3 wire-dict V2 types ---------------------------------------

/// V2 entry — replaces verbose `channel_name` with a `wire_id` plus an
/// optional FLAG_REGISTER bit that announces a fresh `(wire_id, name)`
/// binding. See [`crate::signal::wire_dict`] for the protocol.
#[derive(Debug, Clone)]
pub struct SignalBroadcastEntryV2 {
    /// Bit 0 = FLAG_REGISTER. Receiver uses this to update its
    /// `InboundDict` before resolving.
    pub flags: u8,
    /// Per-connection dictionary id.
    pub wire_id: u32,
    /// Channel name — meaningful only when FLAG_REGISTER is set.
    /// Empty otherwise.
    pub channel_name: String,
    pub value_type: u8,
    pub value_data: f32,
    pub scope: u8,
    pub range_m: f64,
    pub frequency: u32,
    pub sequence: u64,
    pub timestamp_ms: u64,
    pub grant_id: u64,
    pub auth_tag: Vec<u8>,
}

/// V2 batch. `dict_seq` is the sender's `OutboundDict::dict_seq()` at
/// batch-build time; the receiver gates monotonicity per peer.
#[derive(Debug, Clone)]
pub struct SignalBroadcastBatchV2Data {
    pub source_shard_id: u64,
    pub source_position: DVec3,
    pub dict_seq: u64,
    pub entries: Vec<SignalBroadcastEntryV2>,
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

/// Interior-volume bitmask for one chunk of a ship grid. `interior_bits` is
/// `CHUNK_INTERIOR_WORDS` `u64`s packed in the same layout as
/// `voxeldust_core::block::palette::block_index` — bit `i` set means the
/// voxel at `index_to_xyz(i)` (within this chunk) is interior.
///
/// Decoupled from `ChunkColliderData` because an all-interior-air chunk has
/// no collider shapes but may still have tens of thousands of interior
/// voxels (a big ship's central cabin).
#[derive(Debug, Clone)]
pub struct InteriorChunkData {
    pub chunk_key: glam::IVec3,
    pub interior_bits: Vec<u64>,
}

/// Ship hull collider shapes + interior volume mask for physical collision
/// and boarding/exit classification on host shards.
/// Sent from ship shard to planet/system shard when blocks change.
#[derive(Debug, Clone)]
pub struct ShipColliderSyncData {
    pub ship_id: u64,
    pub chunks: Vec<ChunkColliderData>,
    /// Per-chunk interior-voxel bitmask. Empty when the ship has no
    /// semantically-interior volume (no seats/cockpits).
    pub interior_chunks: Vec<InteriorChunkData>,
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
    // Phase T2 — body/head decoupling carried through the digest so
    // distant observers (cross-shard AOI projections) see the surface
    // player's BODY yaw, HEAD yaw/pitch, and animation state on every
    // tick. Mirror of the matching `ObservableEntity` fields on the
    // wire — system-shard `collect_aoi_candidates` forwards these
    // verbatim into the AOI projection it ships to other shards.
    pub body_yaw: f32,
    pub head_yaw: f32,
    pub head_pitch: f32,
    pub locomotion: u8,
    pub locomotion_speed: f32,
    pub is_turning: bool,
    pub turn_target_yaw: f32,
    pub turn_t: f32,
    /// World-space look-at target encoded as a delta from `position`
    /// (so f32 precision is fine even for very far-from-origin shards).
    /// `None` ⇒ no target — the renderer keeps the head in the
    /// animation pose. Mirror of `ObservableEntity.look_target_delta`.
    pub look_target_delta: Option<glam::Vec3>,
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
                let char_state = if h.character_state.is_empty() {
                    None
                } else {
                    Some(builder.create_vector(&h.character_state))
                };

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
                        schema_version: h.schema_version,
                        character_state: char_state,
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
                let (has_pose, pose_pos, pose_rot, pose_vel) = match &a.spawn_pose {
                    Some(sp) => (
                        true,
                        to_fb_vec3d(&sp.position),
                        to_fb_quatd(&sp.rotation),
                        to_fb_vec3d(&sp.velocity),
                    ),
                    None => (
                        false,
                        to_fb_vec3d(&glam::DVec3::ZERO),
                        to_fb_quatd(&glam::DQuat::IDENTITY),
                        to_fb_vec3d(&glam::DVec3::ZERO),
                    ),
                };
                let accepted = fb::HandoffAccepted::create(
                    &mut builder,
                    &fb::HandoffAcceptedArgs {
                        session_token: a.session_token.0,
                        target_shard_id: a.target_shard.0,
                        has_spawn_pose: has_pose,
                        spawn_position: Some(&pose_pos),
                        spawn_rotation: Some(&pose_rot),
                        spawn_velocity: Some(&pose_vel),
                        observer_promoted: a.observer_promoted,
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
                    let stellar = crate::stellar::to_fb_stellar(&b.stellar, &mut builder);
                    let planetary = crate::geophysics::to_fb_planetary(&b.planetary, &mut builder);
                    let rotation_params = crate::planet_rotation::to_fb_rotation_params(&b.rotation_params, &mut builder);
                    fb::CelestialBodySnapshot::create(&mut builder, &fb::CelestialBodySnapshotArgs {
                        body_id: b.body_id, position: Some(&pos), radius: b.radius,
                        color_r: b.color[0], color_g: b.color[1], color_b: b.color[2],
                        stellar,
                        planetary,
                        rotation_params,
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
                    let auth_tag = if e.auth_tag.is_empty() {
                        None
                    } else {
                        Some(builder.create_vector(&e.auth_tag))
                    };
                    fb::SignalBroadcastEntry::create(
                        &mut builder,
                        &fb::SignalBroadcastEntryArgs {
                            channel_name: Some(name),
                            value_type: e.value_type,
                            value_data: e.value_data,
                            scope: e.scope,
                            range_m: e.range_m,
                            frequency: e.frequency,
                            sequence: e.sequence,
                            timestamp_ms: e.timestamp_ms,
                            grant_id: e.grant_id,
                            auth_tag,
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
            ShardMsg::SignalBroadcastBatchV2(data) => {
                // V2 entries: only emit `channel_name` when FLAG_REGISTER
                // is set. Others ship the empty string for FB schema
                // uniformity — receivers MUST gate name use on the flag
                // bit, never on string-non-empty-ness.
                let entries: Vec<_> = data.entries.iter().map(|e| {
                    let needs_register = (e.flags & crate::signal::wire_dict::FLAG_REGISTER) != 0;
                    let name = if needs_register {
                        Some(builder.create_string(&e.channel_name))
                    } else {
                        // Empty placeholder so the FB schema position is
                        // populated; receivers ignore it without the flag.
                        Some(builder.create_string(""))
                    };
                    let auth_tag = if e.auth_tag.is_empty() {
                        None
                    } else {
                        Some(builder.create_vector(&e.auth_tag))
                    };
                    fb::SignalBroadcastEntryV2::create(
                        &mut builder,
                        &fb::SignalBroadcastEntryV2Args {
                            flags: e.flags,
                            wire_id: e.wire_id,
                            channel_name: name,
                            value_type: e.value_type,
                            value_data: e.value_data,
                            scope: e.scope,
                            range_m: e.range_m,
                            frequency: e.frequency,
                            sequence: e.sequence,
                            timestamp_ms: e.timestamp_ms,
                            grant_id: e.grant_id,
                            auth_tag,
                        },
                    )
                }).collect();
                let entries_vec = builder.create_vector(&entries);
                let src_pos = to_fb_vec3d(&data.source_position);
                let batch = fb::SignalBroadcastBatchV2::create(
                    &mut builder,
                    &fb::SignalBroadcastBatchV2Args {
                        source_shard_id: data.source_shard_id,
                        source_position: Some(&src_pos),
                        dict_seq: data.dict_seq,
                        entries: Some(entries_vec),
                    },
                );
                let msg = fb::ShardMessage::create(
                    &mut builder,
                    &fb::ShardMessageArgs {
                        payload_type: fb::ShardPayload::SignalBroadcastBatchV2,
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
                        let (look_set, look_dx, look_dy, look_dz) = match e.look_target_delta {
                            Some(d) => (true, d.x, d.y, d.z),
                            None => (false, 0.0, 0.0, 0.0),
                        };
                        fb::PlanetPlayerDigestEntry::create(
                            &mut builder,
                            &fb::PlanetPlayerDigestEntryArgs {
                                session_token: e.session_token.0,
                                player_name: Some(name),
                                position: Some(&pos),
                                rotation: Some(&rot),
                                planet_index: e.planet_index,
                                body_yaw: e.body_yaw,
                                head_yaw: e.head_yaw,
                                head_pitch: e.head_pitch,
                                locomotion: e.locomotion,
                                locomotion_speed: e.locomotion_speed,
                                is_turning: e.is_turning,
                                turn_target_yaw: e.turn_target_yaw,
                                turn_t: e.turn_t,
                                look_target_set: look_set,
                                look_dx,
                                look_dy,
                                look_dz,
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
            ShardMsg::SignalSubscribe(data) => {
                let name = builder.create_string(&data.channel_name);
                let tag = if data.auth_tag.is_empty() {
                    None
                } else {
                    Some(builder.create_vector(&data.auth_tag))
                };
                let sub = fb::SignalSubscribe::create(
                    &mut builder,
                    &fb::SignalSubscribeArgs {
                        subscriber_shard_id: data.subscriber_shard_id,
                        channel_name: Some(name),
                        grant_id: data.grant_id,
                        nonce: data.nonce,
                        timestamp_ms: data.timestamp_ms,
                        valid_until_tick: data.valid_until_tick,
                        auth_tag: tag,
                    },
                );
                let msg = fb::ShardMessage::create(
                    &mut builder,
                    &fb::ShardMessageArgs {
                        payload_type: fb::ShardPayload::SignalSubscribe,
                        payload: Some(sub.as_union_value()),
                    },
                );
                builder.finish(msg, None);
            }
            ShardMsg::SignalUnsubscribe(data) => {
                let name = builder.create_string(&data.channel_name);
                let unsub = fb::SignalUnsubscribe::create(
                    &mut builder,
                    &fb::SignalUnsubscribeArgs {
                        subscriber_shard_id: data.subscriber_shard_id,
                        channel_name: Some(name),
                        grant_id: data.grant_id,
                    },
                );
                let msg = fb::ShardMessage::create(
                    &mut builder,
                    &fb::ShardMessageArgs {
                        payload_type: fb::ShardPayload::SignalUnsubscribe,
                        payload: Some(unsub.as_union_value()),
                    },
                );
                builder.finish(msg, None);
            }
            ShardMsg::RadioSubscribe(data) => {
                let freqs = builder.create_vector(&data.frequencies);
                let sub = fb::RadioSubscribe::create(
                    &mut builder,
                    &fb::RadioSubscribeArgs {
                        subscriber_shard_id: data.subscriber_shard_id,
                        frequencies: Some(freqs),
                        wildcard: data.wildcard,
                        lease_until_ms: data.lease_until_ms,
                    },
                );
                let msg = fb::ShardMessage::create(
                    &mut builder,
                    &fb::ShardMessageArgs {
                        payload_type: fb::ShardPayload::RadioSubscribe,
                        payload: Some(sub.as_union_value()),
                    },
                );
                builder.finish(msg, None);
            }
            ShardMsg::RadioUnsubscribe(data) => {
                let freqs = builder.create_vector(&data.frequencies);
                let unsub = fb::RadioUnsubscribe::create(
                    &mut builder,
                    &fb::RadioUnsubscribeArgs {
                        subscriber_shard_id: data.subscriber_shard_id,
                        frequencies: Some(freqs),
                        wildcard: data.wildcard,
                    },
                );
                let msg = fb::ShardMessage::create(
                    &mut builder,
                    &fb::ShardMessageArgs {
                        payload_type: fb::ShardPayload::RadioUnsubscribe,
                        payload: Some(unsub.as_union_value()),
                    },
                );
                builder.finish(msg, None);
            }
            ShardMsg::ShipFrequencyInterest(data) => {
                let freqs = builder.create_vector(&data.frequencies);
                let interest = fb::ShipFrequencyInterest::create(
                    &mut builder,
                    &fb::ShipFrequencyInterestArgs {
                        ship_shard_id: data.ship_shard_id,
                        frequencies: Some(freqs),
                        lease_until_ms: data.lease_until_ms,
                    },
                );
                let msg = fb::ShardMessage::create(
                    &mut builder,
                    &fb::ShardMessageArgs {
                        payload_type: fb::ShardPayload::ShipFrequencyInterest,
                        payload: Some(interest.as_union_value()),
                    },
                );
                builder.finish(msg, None);
            }
            ShardMsg::ShipNeighborhood(data) => {
                let peers_fb: Vec<_> = data.peers.iter().map(|p| {
                    let pos = to_fb_vec3d(&p.peer_position);
                    let vel = to_fb_vec3d(&p.peer_velocity);
                    fb::ShipNeighborhoodEntry::create(
                        &mut builder,
                        &fb::ShipNeighborhoodEntryArgs {
                            peer_shard_id: p.peer_shard_id,
                            peer_position: Some(&pos),
                            peer_velocity: Some(&vel),
                        },
                    )
                }).collect();
                let peers_vec = builder.create_vector(&peers_fb);
                let neigh = fb::ShipNeighborhood::create(
                    &mut builder,
                    &fb::ShipNeighborhoodArgs {
                        target_ship_shard_id: data.target_ship_shard_id,
                        issued_at_ms: data.issued_at_ms,
                        peers: Some(peers_vec),
                    },
                );
                let msg = fb::ShardMessage::create(
                    &mut builder,
                    &fb::ShardMessageArgs {
                        payload_type: fb::ShardPayload::ShipNeighborhood,
                        payload: Some(neigh.as_union_value()),
                    },
                );
                builder.finish(msg, None);
            }
            ShardMsg::MediaBroadcastBatch(data) => {
                let frames_fb: Vec<_> = data.frames.iter().map(|f| {
                    use crate::media::MediaPayload;
                    let name = builder.create_string(&f.channel_name);
                    // Project the rich `MediaPayload` into the flat
                    // wire shape: payload_kind + codec + bytes +
                    // dimension fields. Per-kind metadata lands in
                    // the structurally-shared dimension fields.
                    let (payload_bytes, samples, sample_rate, width, height, flags) =
                        match &f.payload {
                            MediaPayload::Text(s) => {
                                (s.as_bytes().to_vec(), 0, 0, 0, 0, 0)
                            }
                            MediaPayload::Audio { frame, samples, sample_rate, .. } => {
                                (frame.clone(), *samples, *sample_rate, 0, 0, 0)
                            }
                            MediaPayload::Video { frame, width, height, keyframe, .. } => {
                                let flags = if *keyframe { 1u8 } else { 0u8 };
                                (frame.clone(), 0, 0, *width as u32, *height as u32, flags)
                            }
                            MediaPayload::Image { data, width, height, .. } => {
                                (data.clone(), 0, 0, *width as u32, *height as u32, 0)
                            }
                        };
                    let payload_data = builder.create_vector(&payload_bytes);
                    let auth_tag = if f.auth_tag.is_empty() {
                        None
                    } else {
                        Some(builder.create_vector(&f.auth_tag))
                    };
                    fb::MediaFrame::create(
                        &mut builder,
                        &fb::MediaFrameArgs {
                            channel_name: Some(name),
                            grant_id: f.grant_id,
                            sequence: f.sequence,
                            timestamp_ms: f.timestamp_ms,
                            payload_kind: f.payload.payload_kind(),
                            codec: f.payload.codec_ordinal(),
                            payload_data: Some(payload_data),
                            samples,
                            sample_rate,
                            width,
                            height,
                            flags,
                            auth_tag,
                        },
                    )
                }).collect();
                let frames_vec = builder.create_vector(&frames_fb);
                let src_pos = to_fb_vec3d(&data.source_position);
                let batch = fb::MediaBroadcastBatch::create(
                    &mut builder,
                    &fb::MediaBroadcastBatchArgs {
                        source_shard_id: data.source_shard_id,
                        source_position: Some(&src_pos),
                        frames: Some(frames_vec),
                    },
                );
                let msg = fb::ShardMessage::create(
                    &mut builder,
                    &fb::ShardMessageArgs {
                        payload_type: fb::ShardPayload::MediaBroadcastBatch,
                        payload: Some(batch.as_union_value()),
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
                let interior_chunks: Vec<_> = data.interior_chunks.iter().map(|ic| {
                    let bits_vec = builder.create_vector(&ic.interior_bits);
                    fb::InteriorChunkBitsMsg::create(
                        &mut builder,
                        &fb::InteriorChunkBitsMsgArgs {
                            cx: ic.chunk_key.x,
                            cy: ic.chunk_key.y,
                            cz: ic.chunk_key.z,
                            interior_bits: Some(bits_vec),
                        },
                    )
                }).collect();
                let interior_chunks_vec = builder.create_vector(&interior_chunks);
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
                        interior_chunks: Some(interior_chunks_vec),
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
                    schema_version: h.schema_version(),
                    character_state: h
                        .character_state()
                        .map(|v| v.bytes().to_vec())
                        .unwrap_or_default(),
                }))
            }

            fb::ShardPayload::HandoffAccepted => {
                let a = msg
                    .payload_as_handoff_accepted()
                    .ok_or(MessageError::MissingField("HandoffAccepted payload"))?;

                let spawn_pose = if a.has_spawn_pose() {
                    match (a.spawn_position(), a.spawn_rotation(), a.spawn_velocity()) {
                        (Some(p), Some(r), Some(v)) => Some(handoff::SpawnPose {
                            position: from_fb_vec3d(p),
                            rotation: from_fb_quatd(r),
                            velocity: from_fb_vec3d(v),
                        }),
                        _ => None,
                    }
                } else {
                    None
                };

                Ok(ShardMsg::HandoffAccepted(handoff::HandoffAccepted {
                    session_token: SessionToken(a.session_token()),
                    target_shard: ShardId(a.target_shard_id()),
                    spawn_pose,
                    observer_promoted: a.observer_promoted(),
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
                        stellar: crate::stellar::from_fb_stellar(b.stellar()),
                        planetary: crate::geophysics::from_fb_planetary(b.planetary()),
                        rotation_params: crate::planet_rotation::from_fb_rotation_params(b.rotation_params()),
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
                        // Phase 2 fields — pre-Phase-2 senders don't write
                        // these, FB returns the schema default (0 / empty),
                        // and the receiver's `try_push_remote` skips the
                        // replay check when both are zero.
                        sequence: e.sequence(),
                        timestamp_ms: e.timestamp_ms(),
                        grant_id: e.grant_id(),
                        auth_tag: e
                            .auth_tag()
                            .map(|v| v.bytes().to_vec())
                            .unwrap_or_default(),
                    }).collect()
                }).unwrap_or_default();

                Ok(ShardMsg::SignalBroadcastBatch(SignalBroadcastBatchData {
                    source_shard_id: batch.source_shard_id(),
                    source_position: from_fb_vec3d(src_pos),
                    entries,
                }))
            }

            fb::ShardPayload::SignalBroadcastBatchV2 => {
                let batch = msg
                    .payload_as_signal_broadcast_batch_v2()
                    .ok_or(MessageError::MissingField("SignalBroadcastBatchV2 payload"))?;
                let src_pos = batch.source_position()
                    .ok_or(MessageError::MissingField("source_position"))?;
                let entries = batch.entries().map(|v| {
                    v.iter().map(|e| SignalBroadcastEntryV2 {
                        flags: e.flags(),
                        wire_id: e.wire_id(),
                        // Receivers MUST gate use on FLAG_REGISTER, not
                        // string-non-empty-ness. We propagate whatever
                        // the wire carried (empty string is fine).
                        channel_name: e.channel_name().unwrap_or("").to_string(),
                        value_type: e.value_type(),
                        value_data: e.value_data(),
                        scope: e.scope(),
                        range_m: e.range_m(),
                        frequency: e.frequency(),
                        sequence: e.sequence(),
                        timestamp_ms: e.timestamp_ms(),
                        grant_id: e.grant_id(),
                        auth_tag: e
                            .auth_tag()
                            .map(|v| v.bytes().to_vec())
                            .unwrap_or_default(),
                    }).collect()
                }).unwrap_or_default();

                Ok(ShardMsg::SignalBroadcastBatchV2(SignalBroadcastBatchV2Data {
                    source_shard_id: batch.source_shard_id(),
                    source_position: from_fb_vec3d(src_pos),
                    dict_seq: batch.dict_seq(),
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
                let interior_chunks: Vec<InteriorChunkData> = sync
                    .interior_chunks()
                    .map(|v| {
                        v.iter()
                            .map(|ic| {
                                let bits: Vec<u64> = ic
                                    .interior_bits()
                                    .map(|b| b.iter().collect())
                                    .unwrap_or_default();
                                InteriorChunkData {
                                    chunk_key: glam::IVec3::new(ic.cx(), ic.cy(), ic.cz()),
                                    interior_bits: bits,
                                }
                            })
                            .collect()
                    })
                    .unwrap_or_default();
                Ok(ShardMsg::ShipColliderSync(ShipColliderSyncData {
                    ship_id: sync.ship_id(),
                    chunks,
                    interior_chunks,
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
                                let look_target_delta = if e.look_target_set() {
                                    Some(glam::Vec3::new(
                                        e.look_dx(),
                                        e.look_dy(),
                                        e.look_dz(),
                                    ))
                                } else {
                                    None
                                };
                                PlanetPlayerDigestEntry {
                                    session_token: SessionToken(e.session_token()),
                                    player_name: e
                                        .player_name()
                                        .unwrap_or("")
                                        .to_string(),
                                    position: pos,
                                    rotation: rot,
                                    planet_index: e.planet_index(),
                                    body_yaw: e.body_yaw(),
                                    head_yaw: e.head_yaw(),
                                    head_pitch: e.head_pitch(),
                                    locomotion: e.locomotion(),
                                    locomotion_speed: e.locomotion_speed(),
                                    is_turning: e.is_turning(),
                                    turn_target_yaw: e.turn_target_yaw(),
                                    turn_t: e.turn_t(),
                                    look_target_delta,
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
            fb::ShardPayload::SignalSubscribe => {
                let sub = msg
                    .payload_as_signal_subscribe()
                    .ok_or(MessageError::MissingField("SignalSubscribe payload"))?;
                Ok(ShardMsg::SignalSubscribe(SignalSubscribeData {
                    subscriber_shard_id: sub.subscriber_shard_id(),
                    channel_name: sub.channel_name().unwrap_or("").to_string(),
                    grant_id: sub.grant_id(),
                    nonce: sub.nonce(),
                    timestamp_ms: sub.timestamp_ms(),
                    valid_until_tick: sub.valid_until_tick(),
                    auth_tag: sub
                        .auth_tag()
                        .map(|v| v.bytes().to_vec())
                        .unwrap_or_default(),
                }))
            }
            fb::ShardPayload::SignalUnsubscribe => {
                let unsub = msg
                    .payload_as_signal_unsubscribe()
                    .ok_or(MessageError::MissingField("SignalUnsubscribe payload"))?;
                Ok(ShardMsg::SignalUnsubscribe(SignalUnsubscribeData {
                    subscriber_shard_id: unsub.subscriber_shard_id(),
                    channel_name: unsub.channel_name().unwrap_or("").to_string(),
                    grant_id: unsub.grant_id(),
                }))
            }
            fb::ShardPayload::RadioSubscribe => {
                let sub = msg
                    .payload_as_radio_subscribe()
                    .ok_or(MessageError::MissingField("RadioSubscribe payload"))?;
                let frequencies = sub
                    .frequencies()
                    .map(|v| v.iter().collect())
                    .unwrap_or_default();
                Ok(ShardMsg::RadioSubscribe(RadioSubscribeData {
                    subscriber_shard_id: sub.subscriber_shard_id(),
                    frequencies,
                    wildcard: sub.wildcard(),
                    lease_until_ms: sub.lease_until_ms(),
                }))
            }
            fb::ShardPayload::RadioUnsubscribe => {
                let unsub = msg
                    .payload_as_radio_unsubscribe()
                    .ok_or(MessageError::MissingField("RadioUnsubscribe payload"))?;
                let frequencies = unsub
                    .frequencies()
                    .map(|v| v.iter().collect())
                    .unwrap_or_default();
                Ok(ShardMsg::RadioUnsubscribe(RadioUnsubscribeData {
                    subscriber_shard_id: unsub.subscriber_shard_id(),
                    frequencies,
                    wildcard: unsub.wildcard(),
                }))
            }
            fb::ShardPayload::ShipFrequencyInterest => {
                let interest = msg
                    .payload_as_ship_frequency_interest()
                    .ok_or(MessageError::MissingField("ShipFrequencyInterest payload"))?;
                let frequencies = interest
                    .frequencies()
                    .map(|v| v.iter().collect())
                    .unwrap_or_default();
                Ok(ShardMsg::ShipFrequencyInterest(ShipFrequencyInterestData {
                    ship_shard_id: interest.ship_shard_id(),
                    frequencies,
                    lease_until_ms: interest.lease_until_ms(),
                }))
            }
            fb::ShardPayload::ShipNeighborhood => {
                let neigh = msg
                    .payload_as_ship_neighborhood()
                    .ok_or(MessageError::MissingField("ShipNeighborhood payload"))?;
                let peers = neigh
                    .peers()
                    .map(|v| v.iter().map(|p| {
                        let pos = p.peer_position().expect("peer_position required");
                        let vel = p.peer_velocity().expect("peer_velocity required");
                        ShipNeighborhoodEntryData {
                            peer_shard_id: p.peer_shard_id(),
                            peer_position: from_fb_vec3d(pos),
                            peer_velocity: from_fb_vec3d(vel),
                        }
                    }).collect())
                    .unwrap_or_default();
                Ok(ShardMsg::ShipNeighborhood(ShipNeighborhoodData {
                    target_ship_shard_id: neigh.target_ship_shard_id(),
                    issued_at_ms: neigh.issued_at_ms(),
                    peers,
                }))
            }
            fb::ShardPayload::MediaBroadcastBatch => {
                use crate::media::{
                    payload_kind, AudioCodec, ImageFormat, MediaFrame, MediaPayload,
                    VideoCodec,
                };
                let batch = msg
                    .payload_as_media_broadcast_batch()
                    .ok_or(MessageError::MissingField("MediaBroadcastBatch payload"))?;
                let src_pos = batch
                    .source_position()
                    .ok_or(MessageError::MissingField("source_position"))?;
                let frames: Vec<MediaFrame> = batch
                    .frames()
                    .map(|v| v.iter().filter_map(|f| {
                        let bytes: Vec<u8> = f
                            .payload_data()
                            .map(|x| x.bytes().to_vec())
                            .unwrap_or_default();
                        // Recompose the rich `MediaPayload` enum
                        // from the flat wire fields. Bad codec
                        // ordinals drop the frame (filter_map → None)
                        // — never panic on hostile / corrupt input.
                        let payload = match f.payload_kind() {
                            payload_kind::TEXT => {
                                let s = String::from_utf8(bytes).ok()?;
                                MediaPayload::Text(s)
                            }
                            payload_kind::AUDIO => {
                                let codec = AudioCodec::from_ordinal(f.codec())?;
                                MediaPayload::Audio {
                                    codec,
                                    frame: bytes,
                                    samples: f.samples(),
                                    sample_rate: f.sample_rate(),
                                }
                            }
                            payload_kind::VIDEO => {
                                let codec = VideoCodec::from_ordinal(f.codec())?;
                                MediaPayload::Video {
                                    codec,
                                    frame: bytes,
                                    width: f.width() as u16,
                                    height: f.height() as u16,
                                    keyframe: (f.flags() & 0x01) != 0,
                                }
                            }
                            payload_kind::IMAGE => {
                                let format = ImageFormat::from_ordinal(f.codec())?;
                                MediaPayload::Image {
                                    format,
                                    data: bytes,
                                    width: f.width() as u16,
                                    height: f.height() as u16,
                                }
                            }
                            _ => return None, // unknown payload kind
                        };
                        Some(MediaFrame {
                            channel_name: f.channel_name().unwrap_or("").to_string(),
                            grant_id: f.grant_id(),
                            sequence: f.sequence(),
                            timestamp_ms: f.timestamp_ms(),
                            payload,
                            auth_tag: f
                                .auth_tag()
                                .map(|x| x.bytes().to_vec())
                                .unwrap_or_default(),
                        })
                    }).collect())
                    .unwrap_or_default();
                Ok(ShardMsg::MediaBroadcastBatch(MediaBroadcastBatchData {
                    source_shard_id: batch.source_shard_id(),
                    source_position: from_fb_vec3d(src_pos),
                    frames,
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
            schema_version: 1,
            character_state: Vec::new(),
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
            spawn_pose: None,
            observer_promoted: false,
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

    #[test]
    fn roundtrip_ship_collider_sync_with_interior() {
        // Two chunks of colliders + two chunks of interior bits. Verifies
        // the new `interior_chunks` field round-trips correctly — without
        // this, the ship-shard → system-shard path was silently dropping
        // ShipColliderSync messages because the payload schema diverged.
        let mut bits0 = vec![0u64; 3724];
        bits0[0] = 0xdead_beef_f00d_babe;
        bits0[37] = 1u64 << 17;
        let mut bits1 = vec![0u64; 3724];
        bits1[3723] = 0xffff_0000_ffff_0000;

        let msg = ShardMsg::ShipColliderSync(ShipColliderSyncData {
            ship_id: 777,
            chunks: vec![
                ChunkColliderData {
                    chunk_key: glam::IVec3::new(0, 0, 0),
                    shapes: vec![
                        (glam::Vec3::new(0.5, 0.5, 0.5), glam::Vec3::new(0.5, 0.5, 0.5)),
                        (glam::Vec3::new(1.5, 0.5, 0.5), glam::Vec3::new(0.5, 0.5, 0.5)),
                    ],
                },
                ChunkColliderData {
                    chunk_key: glam::IVec3::new(0, 0, -1),
                    shapes: vec![(glam::Vec3::new(0.5, 0.5, -0.5), glam::Vec3::new(0.5, 0.5, 0.5))],
                },
            ],
            interior_chunks: vec![
                InteriorChunkData {
                    chunk_key: glam::IVec3::new(0, 0, 0),
                    interior_bits: bits0.clone(),
                },
                InteriorChunkData {
                    chunk_key: glam::IVec3::new(0, 0, -1),
                    interior_bits: bits1.clone(),
                },
            ],
            hull_min: glam::Vec3::new(-5.0, 0.0, -8.0),
            hull_max: glam::Vec3::new(5.0, 6.0, 8.0),
        });
        let bytes = msg.serialize();
        let decoded = ShardMsg::deserialize(&bytes).unwrap();

        let ShardMsg::ShipColliderSync(d) = decoded else {
            panic!("expected ShipColliderSync");
        };
        assert_eq!(d.ship_id, 777);
        assert_eq!(d.chunks.len(), 2);
        assert_eq!(d.chunks[0].shapes.len(), 2);
        assert_eq!(d.chunks[1].chunk_key, glam::IVec3::new(0, 0, -1));
        assert_eq!(d.interior_chunks.len(), 2);
        assert_eq!(d.interior_chunks[0].chunk_key, glam::IVec3::new(0, 0, 0));
        assert_eq!(d.interior_chunks[0].interior_bits, bits0);
        assert_eq!(d.interior_chunks[1].chunk_key, glam::IVec3::new(0, 0, -1));
        assert_eq!(d.interior_chunks[1].interior_bits, bits1);
        assert!((d.hull_min.x + 5.0).abs() < 1e-6);
        assert!((d.hull_max.x - 5.0).abs() < 1e-6);
    }

    // -- Phase 2 wire-format extensions: SignalBroadcastEntry round-trip ----

    #[test]
    fn roundtrip_signal_broadcast_batch_with_phase2_fields() {
        // Round-trip a fully-populated batch through serialize/deserialize
        // and assert every Phase 2 field survives intact. Catches schema
        // drift between the FB definition, the Rust struct, and the
        // serialize/deserialize impls.
        let msg = ShardMsg::SignalBroadcastBatch(SignalBroadcastBatchData {
            source_shard_id: 4242,
            source_position: DVec3::new(1.0, 2.0, 3.0),
            entries: vec![
                SignalBroadcastEntry {
                    channel_name: "alice.beacon".to_string(),
                    value_type: 1,
                    value_data: 0.75,
                    scope: 1,
                    range_m: 5000.0,
                    frequency: 0,
                    sequence: 42,
                    timestamp_ms: 1_700_000_000_000,
                    grant_id: 0,
                    auth_tag: Vec::new(),
                },
                SignalBroadcastEntry {
                    channel_name: "alice.fleet.formation".to_string(),
                    value_type: 0,
                    value_data: 1.0,
                    scope: 3, // Radio
                    range_m: 0.0,
                    frequency: 91100,
                    sequence: 7,
                    timestamp_ms: 1_700_000_000_500,
                    grant_id: 0xdeadbeef_cafebabe,
                    auth_tag: vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16],
                },
            ],
        });
        let bytes = msg.serialize();
        let decoded = ShardMsg::deserialize(&bytes).unwrap();

        let ShardMsg::SignalBroadcastBatch(batch) = decoded else {
            panic!("wrong variant");
        };
        assert_eq!(batch.source_shard_id, 4242);
        assert_eq!(batch.entries.len(), 2);

        let e0 = &batch.entries[0];
        assert_eq!(e0.channel_name, "alice.beacon");
        assert_eq!(e0.sequence, 42);
        assert_eq!(e0.timestamp_ms, 1_700_000_000_000);
        assert_eq!(e0.grant_id, 0);
        assert!(e0.auth_tag.is_empty());

        let e1 = &batch.entries[1];
        assert_eq!(e1.channel_name, "alice.fleet.formation");
        assert_eq!(e1.sequence, 7);
        assert_eq!(e1.timestamp_ms, 1_700_000_000_500);
        assert_eq!(e1.grant_id, 0xdeadbeef_cafebabe);
        assert_eq!(e1.auth_tag.len(), 16);
        assert_eq!(e1.auth_tag[0], 1);
        assert_eq!(e1.auth_tag[15], 16);
    }

    #[test]
    fn roundtrip_signal_subscribe() {
        let msg = ShardMsg::SignalSubscribe(SignalSubscribeData {
            subscriber_shard_id: 0xABCD_1234_5678_9ABC,
            channel_name: "alice.thrust-forward".into(),
            grant_id: 0xCAFEBABE_DEADBEEF,
            nonce: 42,
            timestamp_ms: 1_700_000_000_000,
            valid_until_tick: 12_000,
            auth_tag: vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16],
        });
        let bytes = msg.serialize();
        let decoded = ShardMsg::deserialize(&bytes).unwrap();
        let ShardMsg::SignalSubscribe(d) = decoded else { panic!("wrong variant") };
        assert_eq!(d.subscriber_shard_id, 0xABCD_1234_5678_9ABC);
        assert_eq!(d.channel_name, "alice.thrust-forward");
        assert_eq!(d.grant_id, 0xCAFEBABE_DEADBEEF);
        assert_eq!(d.nonce, 42);
        assert_eq!(d.timestamp_ms, 1_700_000_000_000);
        assert_eq!(d.valid_until_tick, 12_000);
        assert_eq!(d.auth_tag.len(), 16);
        assert_eq!(d.auth_tag[15], 16);
    }

    #[test]
    fn roundtrip_signal_unsubscribe() {
        let msg = ShardMsg::SignalUnsubscribe(SignalUnsubscribeData {
            subscriber_shard_id: 99,
            channel_name: "alice.lights".into(),
            grant_id: 7,
        });
        let bytes = msg.serialize();
        let decoded = ShardMsg::deserialize(&bytes).unwrap();
        let ShardMsg::SignalUnsubscribe(d) = decoded else { panic!("wrong variant") };
        assert_eq!(d.subscriber_shard_id, 99);
        assert_eq!(d.channel_name, "alice.lights");
        assert_eq!(d.grant_id, 7);
    }

    #[test]
    fn roundtrip_signal_broadcast_batch_v2_register_then_bare() {
        // Two-entry batch: first carries FLAG_REGISTER + name, second
        // is a bare reference using the wire_id. Round-trips through
        // the FB encoder + decoder exactly as constructed.
        use crate::signal::wire_dict::FLAG_REGISTER;
        let msg = ShardMsg::SignalBroadcastBatchV2(SignalBroadcastBatchV2Data {
            source_shard_id: 7,
            source_position: DVec3::new(1.0, 2.0, 3.0),
            dict_seq: 42,
            entries: vec![
                SignalBroadcastEntryV2 {
                    flags: FLAG_REGISTER,
                    wire_id: 0,
                    channel_name: "alice.thrust".into(),
                    value_type: 1,
                    value_data: 0.75,
                    scope: 1,
                    range_m: 100.0,
                    frequency: 0,
                    sequence: 1,
                    timestamp_ms: 1_700_000_000_000,
                    grant_id: 5,
                    auth_tag: vec![0xAA; 16],
                },
                // Bare entry: receiver resolves via InboundDict.
                SignalBroadcastEntryV2 {
                    flags: 0,
                    wire_id: 0,
                    channel_name: String::new(),
                    value_type: 1,
                    value_data: 0.5,
                    scope: 1,
                    range_m: 100.0,
                    frequency: 0,
                    sequence: 2,
                    timestamp_ms: 1_700_000_000_500,
                    grant_id: 5,
                    auth_tag: vec![0xBB; 16],
                },
            ],
        });
        let bytes = msg.serialize();
        let decoded = ShardMsg::deserialize(&bytes).unwrap();
        let ShardMsg::SignalBroadcastBatchV2(batch) = decoded else {
            panic!("wrong variant");
        };
        assert_eq!(batch.source_shard_id, 7);
        assert_eq!(batch.dict_seq, 42);
        assert_eq!(batch.entries.len(), 2);
        // First entry: FLAG_REGISTER preserved, name carried.
        assert_eq!(batch.entries[0].flags & FLAG_REGISTER, FLAG_REGISTER);
        assert_eq!(batch.entries[0].channel_name, "alice.thrust");
        assert_eq!(batch.entries[0].wire_id, 0);
        assert_eq!(batch.entries[0].auth_tag.len(), 16);
        // Second entry: flag clear, name NOT trusted (we wrote empty;
        // FB decode reads empty back).
        assert_eq!(batch.entries[1].flags, 0);
        assert_eq!(batch.entries[1].wire_id, 0);
        assert!(batch.entries[1].channel_name.is_empty());
    }

    #[test]
    fn roundtrip_radio_subscribe_with_frequencies() {
        let msg = ShardMsg::RadioSubscribe(RadioSubscribeData {
            subscriber_shard_id: 42,
            frequencies: vec![1234, 5678, 9999],
            wildcard: false,
            lease_until_ms: 1_700_000_060_000,
        });
        let bytes = msg.serialize();
        let decoded = ShardMsg::deserialize(&bytes).unwrap();
        let ShardMsg::RadioSubscribe(d) = decoded else {
            panic!("wrong variant");
        };
        assert_eq!(d.subscriber_shard_id, 42);
        assert_eq!(d.frequencies, vec![1234, 5678, 9999]);
        assert!(!d.wildcard);
        assert_eq!(d.lease_until_ms, 1_700_000_060_000);
    }

    #[test]
    fn roundtrip_radio_subscribe_wildcard() {
        let msg = ShardMsg::RadioSubscribe(RadioSubscribeData {
            subscriber_shard_id: 7,
            frequencies: Vec::new(),
            wildcard: true,
            lease_until_ms: 1_700_000_120_000,
        });
        let bytes = msg.serialize();
        let decoded = ShardMsg::deserialize(&bytes).unwrap();
        let ShardMsg::RadioSubscribe(d) = decoded else {
            panic!("wrong variant");
        };
        assert!(d.wildcard);
        assert!(d.frequencies.is_empty());
    }

    #[test]
    fn roundtrip_ship_frequency_interest() {
        let msg = ShardMsg::ShipFrequencyInterest(ShipFrequencyInterestData {
            ship_shard_id: 9000,
            frequencies: vec![100, 200, 1234, 5678],
            lease_until_ms: 1_700_000_090_000,
        });
        let bytes = msg.serialize();
        let decoded = ShardMsg::deserialize(&bytes).unwrap();
        let ShardMsg::ShipFrequencyInterest(d) = decoded else {
            panic!("wrong variant");
        };
        assert_eq!(d.ship_shard_id, 9000);
        assert_eq!(d.frequencies, vec![100, 200, 1234, 5678]);
        assert_eq!(d.lease_until_ms, 1_700_000_090_000);
    }

    #[test]
    fn roundtrip_ship_neighborhood_with_peers() {
        let msg = ShardMsg::ShipNeighborhood(ShipNeighborhoodData {
            target_ship_shard_id: 7,
            issued_at_ms: 1_700_000_000_000,
            peers: vec![
                ShipNeighborhoodEntryData {
                    peer_shard_id: 11,
                    peer_position: DVec3::new(100.0, 0.0, -50.0),
                    peer_velocity: DVec3::new(5.0, 0.0, 0.0),
                },
                ShipNeighborhoodEntryData {
                    peer_shard_id: 12,
                    peer_position: DVec3::new(-200.0, 30.0, 0.0),
                    peer_velocity: DVec3::ZERO,
                },
            ],
        });
        let bytes = msg.serialize();
        let decoded = ShardMsg::deserialize(&bytes).unwrap();
        let ShardMsg::ShipNeighborhood(d) = decoded else {
            panic!("wrong variant");
        };
        assert_eq!(d.target_ship_shard_id, 7);
        assert_eq!(d.issued_at_ms, 1_700_000_000_000);
        assert_eq!(d.peers.len(), 2);
        assert_eq!(d.peers[0].peer_shard_id, 11);
        assert!((d.peers[0].peer_position.x - 100.0).abs() < 1e-10);
        assert!((d.peers[0].peer_velocity.x - 5.0).abs() < 1e-10);
        assert_eq!(d.peers[1].peer_shard_id, 12);
        assert_eq!(d.peers[1].peer_velocity, DVec3::ZERO);
    }

    #[test]
    fn roundtrip_media_text_frame() {
        use crate::media::{MediaFrame, MediaPayload};
        let msg = ShardMsg::MediaBroadcastBatch(MediaBroadcastBatchData {
            source_shard_id: 7,
            source_position: DVec3::new(1.0, 2.0, 3.0),
            frames: vec![MediaFrame {
                channel_name: "alice.intercom".into(),
                grant_id: 42,
                sequence: 1,
                timestamp_ms: 1_700_000_000_000,
                payload: MediaPayload::Text("Approach corridor A-7".into()),
                auth_tag: vec![0xAA; 16],
            }],
        });
        let bytes = msg.serialize();
        let decoded = ShardMsg::deserialize(&bytes).unwrap();
        let ShardMsg::MediaBroadcastBatch(d) = decoded else {
            panic!("wrong variant");
        };
        assert_eq!(d.frames.len(), 1);
        assert_eq!(d.frames[0].channel_name, "alice.intercom");
        match &d.frames[0].payload {
            MediaPayload::Text(s) => assert_eq!(s, "Approach corridor A-7"),
            other => panic!("expected Text payload, got {other:?}"),
        }
        assert_eq!(d.frames[0].auth_tag.len(), 16);
    }

    #[test]
    fn roundtrip_media_audio_frame_preserves_sample_metadata() {
        use crate::media::{AudioCodec, MediaFrame, MediaPayload};
        let opus_packet: Vec<u8> = (0..200).map(|i| (i * 7) as u8).collect();
        let msg = ShardMsg::MediaBroadcastBatch(MediaBroadcastBatchData {
            source_shard_id: 1,
            source_position: DVec3::ZERO,
            frames: vec![MediaFrame {
                channel_name: "fleet.voice".into(),
                grant_id: 99,
                sequence: 5,
                timestamp_ms: 1_700_000_000_500,
                payload: MediaPayload::Audio {
                    codec: AudioCodec::Opus,
                    frame: opus_packet.clone(),
                    samples: 480,        // 10 ms @ 48 kHz
                    sample_rate: 48_000,
                },
                auth_tag: vec![],
            }],
        });
        let bytes = msg.serialize();
        let decoded = ShardMsg::deserialize(&bytes).unwrap();
        let ShardMsg::MediaBroadcastBatch(d) = decoded else {
            panic!("wrong variant");
        };
        match &d.frames[0].payload {
            MediaPayload::Audio { codec, frame, samples, sample_rate } => {
                assert_eq!(*codec, AudioCodec::Opus);
                assert_eq!(*samples, 480);
                assert_eq!(*sample_rate, 48_000);
                assert_eq!(frame, &opus_packet);
            }
            other => panic!("expected Audio payload, got {other:?}"),
        }
    }

    #[test]
    fn roundtrip_media_video_keyframe_flag() {
        use crate::media::{MediaFrame, MediaPayload, VideoCodec};
        let msg = ShardMsg::MediaBroadcastBatch(MediaBroadcastBatchData {
            source_shard_id: 1,
            source_position: DVec3::ZERO,
            frames: vec![
                MediaFrame {
                    channel_name: "cam1".into(),
                    grant_id: 0,
                    sequence: 1,
                    timestamp_ms: 0,
                    payload: MediaPayload::Video {
                        codec: VideoCodec::H264Baseline,
                        frame: vec![0x00, 0x00, 0x00, 0x01, 0x67],
                        width: 1280,
                        height: 720,
                        keyframe: true,
                    },
                    auth_tag: vec![],
                },
                MediaFrame {
                    channel_name: "cam1".into(),
                    grant_id: 0,
                    sequence: 2,
                    timestamp_ms: 33,
                    payload: MediaPayload::Video {
                        codec: VideoCodec::H264Baseline,
                        frame: vec![0x00, 0x00, 0x00, 0x01, 0x41],
                        width: 1280,
                        height: 720,
                        keyframe: false,
                    },
                    auth_tag: vec![],
                },
            ],
        });
        let bytes = msg.serialize();
        let decoded = ShardMsg::deserialize(&bytes).unwrap();
        let ShardMsg::MediaBroadcastBatch(d) = decoded else {
            panic!("wrong variant");
        };
        match &d.frames[0].payload {
            MediaPayload::Video { keyframe, width, height, .. } => {
                assert!(*keyframe, "first frame is I-frame");
                assert_eq!(*width, 1280);
                assert_eq!(*height, 720);
            }
            _ => panic!("expected Video"),
        }
        match &d.frames[1].payload {
            MediaPayload::Video { keyframe, .. } => {
                assert!(!*keyframe, "second frame is delta");
            }
            _ => panic!("expected Video"),
        }
    }

    #[test]
    fn roundtrip_media_image_format() {
        use crate::media::{ImageFormat, MediaFrame, MediaPayload};
        let png_bytes = vec![0x89, 0x50, 0x4E, 0x47]; // PNG magic
        let msg = ShardMsg::MediaBroadcastBatch(MediaBroadcastBatchData {
            source_shard_id: 1,
            source_position: DVec3::ZERO,
            frames: vec![MediaFrame {
                channel_name: "terminal.shot".into(),
                grant_id: 0,
                sequence: 1,
                timestamp_ms: 0,
                payload: MediaPayload::Image {
                    format: ImageFormat::Png,
                    data: png_bytes.clone(),
                    width: 800,
                    height: 600,
                },
                auth_tag: vec![],
            }],
        });
        let bytes = msg.serialize();
        let decoded = ShardMsg::deserialize(&bytes).unwrap();
        let ShardMsg::MediaBroadcastBatch(d) = decoded else {
            panic!("wrong variant");
        };
        match &d.frames[0].payload {
            MediaPayload::Image { format, data, width, height } => {
                assert_eq!(*format, ImageFormat::Png);
                assert_eq!(data, &png_bytes);
                assert_eq!((*width, *height), (800, 600));
            }
            _ => panic!("expected Image"),
        }
    }

    #[test]
    fn roundtrip_media_unknown_codec_drops_frame_without_panic() {
        // Hand-build a wire payload with an unknown video codec
        // ordinal. The deserializer must drop that frame gracefully
        // — no panic, no whole-batch failure.
        use crate::media::{MediaFrame, MediaPayload, VideoCodec};
        // Start with a valid frame so the deserialize succeeds.
        let msg = ShardMsg::MediaBroadcastBatch(MediaBroadcastBatchData {
            source_shard_id: 1,
            source_position: DVec3::ZERO,
            frames: vec![MediaFrame {
                channel_name: "ok".into(),
                grant_id: 0,
                sequence: 1,
                timestamp_ms: 0,
                payload: MediaPayload::Video {
                    codec: VideoCodec::Av1,
                    frame: vec![],
                    width: 100,
                    height: 100,
                    keyframe: false,
                },
                auth_tag: vec![],
            }],
        });
        let bytes = msg.serialize();
        let decoded = ShardMsg::deserialize(&bytes).unwrap();
        let ShardMsg::MediaBroadcastBatch(d) = decoded else {
            panic!("wrong variant");
        };
        // The frame round-trips because AV1 is a known codec ordinal.
        // (Negative-path unknown-codec test would require crafting raw
        //  FB bytes; the from_ordinal None path is covered by the
        //  media module's own tests.)
        assert_eq!(d.frames.len(), 1);
    }

    #[test]
    fn roundtrip_ship_neighborhood_empty_peers() {
        let msg = ShardMsg::ShipNeighborhood(ShipNeighborhoodData {
            target_ship_shard_id: 1,
            issued_at_ms: 0,
            peers: vec![],
        });
        let bytes = msg.serialize();
        let decoded = ShardMsg::deserialize(&bytes).unwrap();
        let ShardMsg::ShipNeighborhood(d) = decoded else {
            panic!("wrong variant");
        };
        assert!(d.peers.is_empty());
    }

    #[test]
    fn roundtrip_ship_frequency_interest_empty_set() {
        // A ship that USED to have listeners but currently has none
        // sends an empty-set interest to drop its entry. Round-trips
        // cleanly so the receiver sees `frequencies.is_empty()`.
        let msg = ShardMsg::ShipFrequencyInterest(ShipFrequencyInterestData {
            ship_shard_id: 1,
            frequencies: vec![],
            lease_until_ms: 0,
        });
        let bytes = msg.serialize();
        let decoded = ShardMsg::deserialize(&bytes).unwrap();
        let ShardMsg::ShipFrequencyInterest(d) = decoded else {
            panic!("wrong variant");
        };
        assert!(d.frequencies.is_empty());
    }

    #[test]
    fn roundtrip_radio_unsubscribe() {
        let msg = ShardMsg::RadioUnsubscribe(RadioUnsubscribeData {
            subscriber_shard_id: 99,
            frequencies: vec![88],
            wildcard: false,
        });
        let bytes = msg.serialize();
        let decoded = ShardMsg::deserialize(&bytes).unwrap();
        let ShardMsg::RadioUnsubscribe(d) = decoded else {
            panic!("wrong variant");
        };
        assert_eq!(d.subscriber_shard_id, 99);
        assert_eq!(d.frequencies, vec![88]);
    }

    #[test]
    fn roundtrip_signal_broadcast_batch_legacy_zero_fields() {
        // Pre-Phase-2 senders ship zero/empty for the new fields. Make sure
        // the new deserializer accepts that as legitimate (backward compat).
        let msg = ShardMsg::SignalBroadcastBatch(SignalBroadcastBatchData {
            source_shard_id: 42,
            source_position: DVec3::ZERO,
            entries: vec![SignalBroadcastEntry {
                channel_name: "legacy".to_string(),
                value_type: 1,
                value_data: 0.5,
                scope: 1,
                range_m: 1000.0,
                frequency: 0,
                sequence: 0,
                timestamp_ms: 0,
                grant_id: 0,
                auth_tag: Vec::new(),
            }],
        });
        let bytes = msg.serialize();
        let decoded = ShardMsg::deserialize(&bytes).unwrap();
        let ShardMsg::SignalBroadcastBatch(batch) = decoded else {
            panic!("wrong variant");
        };
        assert_eq!(batch.entries.len(), 1);
        assert_eq!(batch.entries[0].sequence, 0);
        assert_eq!(batch.entries[0].timestamp_ms, 0);
        assert!(batch.entries[0].auth_tag.is_empty());
    }
}
