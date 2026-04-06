//! Authorized peers filter for host-scoped QUIC messages.
//!
//! Tracks which QUIC peers are allowed to send scene data (SystemSceneUpdate,
//! ShipPositionUpdate) to this shard. Updated by HostSwitch and ShardPreConnect
//! lifecycle events. Supports multi-shard compositing (multiple concurrent sources).

use std::collections::HashSet;

use voxeldust_core::shard_types::ShardId;

/// Tracks which shard IDs are authorized to send scene data.
/// Updated by HostSwitch and ShardPreConnect lifecycle events.
/// Verified against the source_shard_id in QueuedShardMsg.
#[derive(Default)]
pub struct AuthorizedPeers {
    shard_ids: HashSet<ShardId>,
}

impl AuthorizedPeers {
    /// Replace all authorized peers with a single new host.
    /// Used on HostSwitch — only the new host is authorized.
    pub fn set_host(&mut self, id: ShardId) {
        self.shard_ids.clear();
        self.shard_ids.insert(id);
    }

    /// Add an additional authorized peer.
    /// Used for secondary shard compositing (e.g., planet + system).
    pub fn add(&mut self, id: ShardId) {
        self.shard_ids.insert(id);
    }

    /// Remove a peer (e.g., when a secondary connection ends).
    pub fn remove(&mut self, id: &ShardId) {
        self.shard_ids.remove(id);
    }

    /// Check if a shard ID is authorized to send scene data.
    /// Returns true when the set is empty (initial state before first
    /// HostSwitch) to preserve backward compatibility.
    pub fn is_authorized(&self, id: ShardId) -> bool {
        self.shard_ids.is_empty() || self.shard_ids.contains(&id)
    }

    /// Clear all authorized peers.
    pub fn clear(&mut self) {
        self.shard_ids.clear();
    }
}
