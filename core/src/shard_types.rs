use serde::{Deserialize, Serialize};
use std::fmt;
use std::net::SocketAddr;

/// Unique identifier for a shard instance.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct ShardId(pub u64);

impl fmt::Display for ShardId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "shard-{}", self.0)
    }
}

/// What kind of game world a shard manages.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ShardType {
    /// Manages a planet's surface (or a subset of its sectors).
    Planet,
    /// Manages a star system's space (everything between planets).
    System,
    /// Manages one ship's interior.
    Ship,
    /// Manages the space between star systems in a galaxy.
    Galaxy,
}

impl fmt::Display for ShardType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ShardType::Planet => write!(f, "planet"),
            ShardType::System => write!(f, "system"),
            ShardType::Ship => write!(f, "ship"),
            ShardType::Galaxy => write!(f, "galaxy"),
        }
    }
}

/// Lifecycle state of a shard from the orchestrator's perspective.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ShardState {
    /// Pod/process created, waiting for first heartbeat.
    Provisioning,
    /// First heartbeat received, shard is initializing.
    Starting,
    /// Shard is accepting players.
    Ready,
    /// Shard is shutting down, handing off remaining players.
    Draining,
    /// Shard has stopped.
    Stopped,
}

impl ShardState {
    /// Returns true if this state can transition to `next`.
    pub fn can_transition_to(self, next: ShardState) -> bool {
        use ShardState::*;
        matches!(
            (self, next),
            (Provisioning, Starting)
                | (Provisioning, Stopped) // failed to start
                | (Starting, Ready)
                | (Starting, Stopped) // failed during init
                | (Ready, Draining)
                | (Ready, Stopped) // crashed
                | (Draining, Stopped)
        )
    }
}

/// Network endpoints for connecting to a shard.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ShardEndpoint {
    /// TCP address for reliable messages (join, block edits, redirects).
    pub tcp_addr: SocketAddr,
    /// UDP address for fast messages (input, world state).
    pub udp_addr: SocketAddr,
    /// QUIC address for inter-shard communication.
    pub quic_addr: SocketAddr,
}

/// Opaque token that ties a client session across shard handoffs.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct SessionToken(pub u64);

/// Full shard registration info as tracked by the orchestrator.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ShardInfo {
    pub id: ShardId,
    pub shard_type: ShardType,
    pub state: ShardState,
    pub endpoint: ShardEndpoint,
    /// For planet shards: the planet seed.
    pub planet_seed: Option<u64>,
    /// For planet shards: which sectors this shard owns (0-5).
    pub sectors: Option<Vec<u8>>,
    /// For system shards: the system seed.
    pub system_seed: Option<u64>,
    /// For ship shards: the ship entity id.
    pub ship_id: Option<u64>,
    /// For galaxy shards: the galaxy seed.
    pub galaxy_seed: Option<u64>,
    /// For ship shards: the shard managing this ship's exterior (system or planet shard).
    pub host_shard_id: Option<ShardId>,
    /// CLI args used to launch this shard (for auto-restart).
    pub launch_args: Vec<String>,
}

/// Heartbeat data sent periodically from shard to orchestrator.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ShardHeartbeat {
    pub shard_id: ShardId,
    pub tick_ms: f32,
    pub p99_tick_ms: f32,
    pub player_count: u32,
    pub chunk_count: u32,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn shard_state_valid_transitions() {
        use ShardState::*;
        assert!(Provisioning.can_transition_to(Starting));
        assert!(Provisioning.can_transition_to(Stopped));
        assert!(Starting.can_transition_to(Ready));
        assert!(Starting.can_transition_to(Stopped));
        assert!(Ready.can_transition_to(Draining));
        assert!(Ready.can_transition_to(Stopped));
        assert!(Draining.can_transition_to(Stopped));
    }

    #[test]
    fn shard_state_invalid_transitions() {
        use ShardState::*;
        assert!(!Provisioning.can_transition_to(Ready)); // must go through Starting
        assert!(!Provisioning.can_transition_to(Draining));
        assert!(!Starting.can_transition_to(Draining)); // must go through Ready
        assert!(!Ready.can_transition_to(Starting)); // no going backwards
        assert!(!Draining.can_transition_to(Ready)); // no going backwards
        assert!(!Stopped.can_transition_to(Provisioning)); // terminal
    }

    #[test]
    fn shard_id_display() {
        assert_eq!(ShardId(42).to_string(), "shard-42");
    }

    #[test]
    fn shard_type_display() {
        assert_eq!(ShardType::Planet.to_string(), "planet");
        assert_eq!(ShardType::System.to_string(), "system");
        assert_eq!(ShardType::Ship.to_string(), "ship");
    }
}
