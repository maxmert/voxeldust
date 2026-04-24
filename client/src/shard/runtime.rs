//! Per-shard runtime state — shared data the registry + plugins need.

use std::time::Instant;

use bevy::prelude::*;
use glam::{DQuat, DVec3};

/// Wire-level identifier for a shard instance. `shard_type` is the u8 from
/// the protocol (`0=Planet`, `1=System`, `2=Ship`, `3=Galaxy`, future
/// variants per server); `seed` is the per-instance seed the server
/// provides at connection time. The client never enumerates shard types —
/// unknown values log a warning and fall back to generic secondary
/// behavior.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ShardKey {
    pub shard_type: u8,
    pub seed: u64,
}

impl ShardKey {
    pub fn new(shard_type: u8, seed: u64) -> Self {
        Self { shard_type, seed }
    }
}

impl std::fmt::Display for ShardKey {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "shard[{}/0x{:x}]", self.shard_type, self.seed)
    }
}

/// Per-shard client-side state. One instance per connected shard
/// (primary OR secondary), referenced by the registry + plugins.
///
/// The `origin` + `rotation` fields are in SYSTEM-SPACE (canonical
/// coordinate frame). Phase 4's `ShardOriginPlugin` rebases every
/// shard's Bevy `Transform` relative to the camera each frame. Don't
/// read Bevy transforms for gameplay math; read these f64 fields.
#[derive(Debug, Clone)]
pub struct ShardRuntime {
    pub key: ShardKey,
    /// The root `ChunkSource` entity for this shard. Chunks, sub-grids,
    /// sub-blocks, and block highlights parent under this entity.
    pub entity: Entity,
    /// System-space origin of this shard. SYSTEM's origin is typically
    /// `DVec3::ZERO`; SHIP's is the ship's position; PLANET's is the
    /// planet's system-space position.
    pub origin: DVec3,
    /// System-space orientation of this shard's local frame. Matters for
    /// SHIP (ship rotation) and PLANET (planet rotation for day/night
    /// alignment). SYSTEM / GALAXY typically use identity.
    pub rotation: DQuat,
    /// Server-provided reference pose at connection time. Stored so
    /// plugins can compute deltas (e.g., the PLANET tangent frame at
    /// spawn).
    pub reference_position: DVec3,
    pub reference_rotation: DQuat,
    pub connected_at: Instant,
}

/// Marker component on the root entity of each shard. `ShardKey` + this
/// marker identify "chunk source roots" that the chunk streamer, raycast
/// system, highlight system, and origin-rebase system iterate over.
#[derive(Component, Debug, Clone, Copy)]
pub struct ChunkSource {
    pub key: ShardKey,
}
