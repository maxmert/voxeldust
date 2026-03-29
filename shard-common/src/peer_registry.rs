use std::collections::HashMap;
use std::net::SocketAddr;

use voxeldust_core::shard_types::{ShardEndpoint, ShardId, ShardInfo, ShardType};

/// Cached registry of peer shard endpoints, refreshed periodically
/// from the orchestrator.
pub struct PeerShardRegistry {
    peers: HashMap<ShardId, PeerEntry>,
}

pub struct PeerEntry {
    pub info: ShardInfo,
}

impl PeerShardRegistry {
    pub fn new() -> Self {
        Self {
            peers: HashMap::new(),
        }
    }

    /// Update the registry with a fresh list of shards from the orchestrator.
    pub fn update(&mut self, shards: Vec<ShardInfo>) {
        self.peers.clear();
        for info in shards {
            self.peers.insert(info.id, PeerEntry { info });
        }
    }

    /// Find a peer shard by ID.
    pub fn get(&self, id: ShardId) -> Option<&ShardInfo> {
        self.peers.get(&id).map(|e| &e.info)
    }

    /// Find peer shards by type.
    pub fn find_by_type(&self, shard_type: ShardType) -> Vec<&ShardInfo> {
        self.peers
            .values()
            .filter(|e| e.info.shard_type == shard_type)
            .map(|e| &e.info)
            .collect()
    }

    /// Find the QUIC address for a peer shard.
    pub fn quic_addr(&self, id: ShardId) -> Option<SocketAddr> {
        self.get(id).map(|info| info.endpoint.quic_addr)
    }

    /// Get the endpoint for a peer shard.
    pub fn endpoint(&self, id: ShardId) -> Option<&ShardEndpoint> {
        self.get(id).map(|info| &info.endpoint)
    }

    /// List all known peers.
    pub fn all(&self) -> Vec<&ShardInfo> {
        self.peers.values().map(|e| &e.info).collect()
    }
}

impl Default for PeerShardRegistry {
    fn default() -> Self {
        Self::new()
    }
}
