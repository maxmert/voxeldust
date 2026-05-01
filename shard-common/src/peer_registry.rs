use std::collections::HashMap;
use std::net::SocketAddr;

use voxeldust_core::shard_types::{ShardEndpoint, ShardId, ShardInfo, ShardState, ShardType};

/// Decision a system-shard's Radio relay makes about where to fan
/// out an inbound `SignalBroadcastBatch` of scope=Radio. See
/// [`PeerShardRegistry::radio_fanout_decision`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RadioFanout {
    /// Forward to every peer of type `Galaxy` in the registry.
    pub to_galaxy: bool,
    /// Forward to every locally-hosted ship-shard peer (those whose
    /// `host_shard_id == our_id`), excluding the source.
    pub to_local_ships: bool,
}

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
    /// Only includes live shards (not Stopped/Draining) to prevent stale entries
    /// from previous sessions triggering dead QUIC connections.
    pub fn update(&mut self, shards: Vec<ShardInfo>) {
        self.peers.clear();
        for info in shards {
            if info.state == ShardState::Stopped || info.state == ShardState::Draining {
                continue;
            }
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

    /// Phase 3F: decide where to relay a Radio-scope batch given its
    /// `source` peer and our own shard id. Encapsulates the routing
    /// rules so V1 and V2 relay arms can share one decision and a
    /// single test exercises both.
    ///
    /// **Rules**:
    ///   - From a *locally-hosted ship* (`shard_type == Ship` and
    ///     `host_shard_id == our_id`): outbound. Send to every
    ///     galaxy peer (cross-system fan-out) AND to every other
    ///     locally-hosted ship (same-system co-frequency listening
    ///     without round-tripping through galaxy).
    ///   - From a *galaxy peer*: inbound. Send to every locally-
    ///     hosted ship; do NOT echo back to galaxy (that's a relay
    ///     loop).
    ///   - From an *unknown* source (peer disappeared, registry
    ///     stale): drop. The decision returns `to_galaxy=false,
    ///     to_local_ships=false` and the caller does nothing.
    ///
    /// `source` not found in the registry is the unknown case.
    pub fn radio_fanout_decision(&self, source: ShardId, our_id: ShardId) -> RadioFanout {
        let info = self.get(source);
        let from_galaxy = info
            .map(|i| i.shard_type == ShardType::Galaxy)
            .unwrap_or(false);
        let from_local_ship = info
            .map(|i| {
                i.shard_type == ShardType::Ship && i.host_shard_id == Some(our_id)
            })
            .unwrap_or(false);
        RadioFanout {
            to_galaxy: from_local_ship,
            to_local_ships: from_galaxy || from_local_ship,
        }
    }

    /// Iterate locally-hosted ship peers (excluding `exclude` if
    /// non-zero). Used by the system-shard's Radio relay to fan out
    /// to "ships in this system" without re-walking + re-filtering
    /// `all()` at every call site.
    pub fn local_ships<'a>(
        &'a self,
        our_id: ShardId,
        exclude: ShardId,
    ) -> impl Iterator<Item = &'a ShardInfo> + 'a {
        self.peers.values().filter_map(move |e| {
            let info = &e.info;
            if info.shard_type != ShardType::Ship {
                return None;
            }
            if info.host_shard_id != Some(our_id) {
                return None;
            }
            if info.id == exclude {
                return None;
            }
            Some(info)
        })
    }
}

impl Default for PeerShardRegistry {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn shard_info(id: u64, ty: ShardType, host: Option<u64>) -> ShardInfo {
        use voxeldust_core::shard_types::ShardEndpoint;
        let port = 7777 + id as u16;
        ShardInfo {
            id: ShardId(id),
            shard_type: ty,
            state: ShardState::Ready,
            endpoint: ShardEndpoint {
                tcp_addr: format!("127.0.0.1:{}", port).parse().unwrap(),
                udp_addr: format!("127.0.0.1:{}", port + 1).parse().unwrap(),
                quic_addr: format!("127.0.0.1:{}", port + 2).parse().unwrap(),
            },
            planet_seed: None,
            sectors: None,
            system_seed: None,
            ship_id: None,
            galaxy_seed: None,
            host_shard_id: host.map(ShardId),
            launch_args: vec![],
        }
    }

    fn fixture_registry(our_system_id: u64) -> PeerShardRegistry {
        let mut r = PeerShardRegistry::new();
        // Our system + a peer system + a galaxy + ships hosted by each.
        r.update(vec![
            shard_info(our_system_id, ShardType::System, None),
            shard_info(200, ShardType::System, None),
            shard_info(900, ShardType::Galaxy, None),
            shard_info(11, ShardType::Ship, Some(our_system_id)),
            shard_info(12, ShardType::Ship, Some(our_system_id)),
            shard_info(21, ShardType::Ship, Some(200)),
        ]);
        r
    }

    #[test]
    fn radio_fanout_from_local_ship_goes_to_galaxy_and_other_ships() {
        let r = fixture_registry(100);
        let dec = r.radio_fanout_decision(ShardId(11), ShardId(100));
        assert_eq!(
            dec,
            RadioFanout {
                to_galaxy: true,
                to_local_ships: true
            }
        );
    }

    #[test]
    fn radio_fanout_from_galaxy_goes_to_local_ships_not_galaxy() {
        let r = fixture_registry(100);
        let dec = r.radio_fanout_decision(ShardId(900), ShardId(100));
        assert_eq!(
            dec,
            RadioFanout {
                to_galaxy: false,
                to_local_ships: true,
            },
            "echoing inbound back to galaxy is a relay loop"
        );
    }

    #[test]
    fn radio_fanout_from_remote_system_ship_drops_quietly() {
        // Ship 21 belongs to system 200, not us. Today this isn't a
        // realistic source (ship-shards never message us directly
        // without going through their own system-shard), but the
        // decision must be defensive.
        let r = fixture_registry(100);
        let dec = r.radio_fanout_decision(ShardId(21), ShardId(100));
        assert_eq!(
            dec,
            RadioFanout {
                to_galaxy: false,
                to_local_ships: false
            }
        );
    }

    #[test]
    fn radio_fanout_from_unknown_source_drops_quietly() {
        let r = fixture_registry(100);
        let dec = r.radio_fanout_decision(ShardId(9999), ShardId(100));
        assert_eq!(
            dec,
            RadioFanout {
                to_galaxy: false,
                to_local_ships: false
            }
        );
    }

    #[test]
    fn local_ships_returns_only_locally_hosted_ships_excluding_target() {
        let r = fixture_registry(100);
        let ids: Vec<u64> = r.local_ships(ShardId(100), ShardId(11)).map(|info| info.id.0).collect();
        // Only ship 12 — ship 11 is excluded, ship 21 belongs to a
        // different system-shard.
        assert_eq!(ids, vec![12]);
    }

    #[test]
    fn local_ships_with_zero_exclude_returns_all_locals() {
        let r = fixture_registry(100);
        let mut ids: Vec<u64> = r.local_ships(ShardId(100), ShardId(0)).map(|info| info.id.0).collect();
        ids.sort();
        assert_eq!(ids, vec![11, 12]);
    }
}
