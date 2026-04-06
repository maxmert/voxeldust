//! Shared ECS message types for cross-system communication within a tick.
//!
//! Bridge systems drain mpsc channels into MessageWriter<T> at the start of each
//! tick. Downstream systems consume via MessageReader<T>.

use bevy_ecs::prelude::*;
use glam::DVec3;

use crate::client_message::PlayerInputData;
use crate::handoff::PlayerHandoff;
use crate::shard_message::{AutopilotCommandData, ShipControlInput, WarpAutopilotCommandData};
use crate::shard_types::{SessionToken, ShardId};

/// Inbound player handoff from another shard (QUIC).
#[derive(Message)]
pub struct PlayerHandoffEvent(pub PlayerHandoff);

/// Handoff accepted by the target shard.
#[derive(Message)]
pub struct HandoffAcceptedEvent {
    pub session: SessionToken,
    pub shard: ShardId,
}

/// Ship control input from a pilot (forwarded from ship shard via QUIC).
#[derive(Message)]
pub struct ShipControlEvent(pub ShipControlInput);

/// Autopilot command (engage/disengage/change target).
#[derive(Message)]
pub struct AutopilotCommandEvent(pub AutopilotCommandData);

/// Warp autopilot command (engage interstellar warp).
#[derive(Message)]
pub struct WarpAutopilotCommandEvent(pub WarpAutopilotCommandData);

/// Player input from UDP (movement, look, actions).
#[derive(Message)]
pub struct PlayerInputEvent {
    pub session: SessionToken,
    pub input: PlayerInputData,
}

/// A ship entered a planet's sphere of influence.
#[derive(Message)]
pub struct SoiEntryEvent {
    pub entity: Entity,
    pub planet_index: usize,
}

/// A ship exited a planet's sphere of influence.
#[derive(Message)]
pub struct SoiExitEvent {
    pub entity: Entity,
    pub planet_index: usize,
}

/// A ship has landed on a planet.
#[derive(Message)]
pub struct ShipLandedEvent {
    pub entity: Entity,
    pub planet_index: usize,
}

/// Trigger an outbound handoff for an entity.
#[derive(Message)]
pub struct HandoffTriggerEvent {
    pub entity: Entity,
    pub target_shard: ShardId,
}

/// Ship position update from the system shard (received by ship/planet shards).
#[derive(Message)]
pub struct ShipPositionUpdateEvent {
    pub ship_id: u64,
    pub position: DVec3,
    pub velocity: DVec3,
    pub rotation: glam::DQuat,
    pub angular_velocity: DVec3,
}
