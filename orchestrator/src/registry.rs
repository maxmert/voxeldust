use std::collections::{HashMap, VecDeque};
use std::path::Path;
use std::time::Instant;

use redb::Database;
use serde::{Deserialize, Serialize};
use tracing::{info, warn};

use voxeldust_core::shard_types::*;

use crate::persistence;

/// Configuration for how to launch a shard process.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LaunchConfig {
    pub binary: String,
    pub args: Vec<String>,
    pub shard_type: ShardType,
}

/// Runtime entry for a tracked shard.
pub struct ShardEntry {
    pub info: ShardInfo,
    pub last_heartbeat: Instant,
    pub last_metrics: Option<ShardHeartbeat>,
    /// Timestamps of recent restarts, for rate limiting.
    pub restart_timestamps: VecDeque<Instant>,
    pub launch_config: Option<LaunchConfig>,
    /// When player_count first became 0 (for idle shutdown).
    pub idle_since: Option<Instant>,
}

/// In-memory shard registry backed by redb.
pub struct ShardRegistry {
    shards: HashMap<ShardId, ShardEntry>,
    /// Index: planet_seed → shard IDs serving that planet.
    planet_index: HashMap<u64, Vec<ShardId>>,
    /// Index: system_seed → shard ID serving that system.
    system_index: HashMap<u64, ShardId>,
    /// Index: galaxy_seed → shard ID serving that galaxy.
    galaxy_index: HashMap<u64, ShardId>,
    /// Index: ship_id → shard ID serving that ship.
    ship_index: HashMap<u64, ShardId>,
    db: Database,
}

impl ShardRegistry {
    /// Open or create the registry from a database path.
    /// Restores previously persisted shards (marked as Provisioning since
    /// we don't know if they're still alive — heartbeats will update them).
    pub fn open(db_path: &Path) -> Result<Self, Box<dyn std::error::Error>> {
        let db = persistence::open_db(db_path)?;

        let mut registry = Self {
            shards: HashMap::new(),
            planet_index: HashMap::new(),
            system_index: HashMap::new(),
            galaxy_index: HashMap::new(),
            ship_index: HashMap::new(),
            db,
        };

        // Restore from persistence.
        if let Ok(saved_shards) = persistence::load_all_shards(&registry.db) {
            for (id, mut info) in saved_shards {
                // Mark as Provisioning — we don't know if the shard is alive.
                // Live shards will re-register via heartbeat.
                info.state = ShardState::Provisioning;
                info!(shard_id = %id, "restored shard from persistence");
                registry.insert_entry(
                    id,
                    ShardEntry {
                        info,
                        last_heartbeat: Instant::now(),
                        last_metrics: None,
                        restart_timestamps: VecDeque::new(),
                        launch_config: None,
                        idle_since: None,
                    },
                );
            }
        }

        // Restore launch configs.
        if let Ok(configs) = persistence::load_all_launch_configs(&registry.db) {
            for (id, config) in configs {
                if let Some(entry) = registry.shards.get_mut(&id) {
                    entry.launch_config = Some(config);
                }
            }
        }

        Ok(registry)
    }

    /// Register or update a shard. Validates state transitions.
    pub fn register(&mut self, info: ShardInfo, launch_config: Option<LaunchConfig>) {
        let id = info.id;

        if let Some(existing) = self.shards.get(&id) {
            if !existing.info.state.can_transition_to(info.state) {
                warn!(
                    shard_id = %id,
                    from = ?existing.info.state,
                    to = ?info.state,
                    "invalid state transition, forcing update"
                );
            }
        }

        // Persist.
        if let Err(e) = persistence::save_shard(&self.db, id, &info) {
            warn!(%e, shard_id = %id, "failed to persist shard");
        }

        if let Some(ref lc) = launch_config {
            if let Err(e) = persistence::save_launch_config(&self.db, id, lc) {
                warn!(%e, shard_id = %id, "failed to persist launch config");
            }
        }

        let entry = ShardEntry {
            info,
            last_heartbeat: Instant::now(),
            last_metrics: None,
            restart_timestamps: self
                .shards
                .get(&id)
                .map(|e| e.restart_timestamps.clone())
                .unwrap_or_default(),
            launch_config: launch_config.or_else(|| {
                self.shards
                    .get(&id)
                    .and_then(|e| e.launch_config.clone())
            }),
            idle_since: None,
        };

        // Remove old index entries if replacing.
        self.remove_from_indices(id);
        self.insert_entry(id, entry);

        info!(shard_id = %id, "shard registered");
    }

    /// Update heartbeat timestamp and metrics for a shard.
    pub fn update_heartbeat(&mut self, heartbeat: &ShardHeartbeat) {
        let id = heartbeat.shard_id;
        if let Some(entry) = self.shards.get_mut(&id) {
            entry.last_heartbeat = Instant::now();
            entry.last_metrics = Some(heartbeat.clone());

            // Track idle state.
            if heartbeat.player_count == 0 {
                if entry.idle_since.is_none() {
                    entry.idle_since = Some(Instant::now());
                }
            } else {
                entry.idle_since = None;
            }

            // Transition Provisioning/Starting → Ready on first heartbeat with metrics.
            if entry.info.state == ShardState::Provisioning
                || entry.info.state == ShardState::Starting
            {
                entry.info.state = ShardState::Ready;
                if let Err(e) = persistence::save_shard(&self.db, id, &entry.info) {
                    warn!(%e, shard_id = %id, "failed to persist state transition");
                }
                info!(shard_id = %id, "shard transitioned to Ready");
            }
        }
    }

    pub fn get(&self, id: ShardId) -> Option<&ShardEntry> {
        self.shards.get(&id)
    }

    pub fn get_mut(&mut self, id: ShardId) -> Option<&mut ShardEntry> {
        self.shards.get_mut(&id)
    }

    pub fn list(&self) -> Vec<&ShardInfo> {
        self.shards.values().map(|e| &e.info).collect()
    }

    pub fn entries(&self) -> &HashMap<ShardId, ShardEntry> {
        &self.shards
    }

    /// Find shards serving a planet by seed.
    pub fn find_by_planet(&self, seed: u64) -> Vec<&ShardInfo> {
        self.planet_index
            .get(&seed)
            .map(|ids| {
                ids.iter()
                    .filter_map(|id| self.shards.get(id).map(|e| &e.info))
                    .collect()
            })
            .unwrap_or_default()
    }

    /// Find the shard serving a star system by seed.
    pub fn find_by_system(&self, seed: u64) -> Option<&ShardInfo> {
        self.system_index
            .get(&seed)
            .and_then(|id| self.shards.get(id).map(|e| &e.info))
    }

    /// Find the shard serving a galaxy by seed.
    pub fn find_by_galaxy(&self, seed: u64) -> Option<&ShardInfo> {
        self.galaxy_index
            .get(&seed)
            .and_then(|id| self.shards.get(id).map(|e| &e.info))
    }

    /// Find the shard serving a ship by ship_id.
    pub fn find_by_ship(&self, ship_id: u64) -> Option<&ShardInfo> {
        self.ship_index
            .get(&ship_id)
            .and_then(|id| self.shards.get(id).map(|e| &e.info))
    }

    /// Remove a shard from the registry.
    pub fn remove(&mut self, id: ShardId) {
        self.remove_from_indices(id);
        self.shards.remove(&id);
        if let Err(e) = persistence::remove_shard(&self.db, id) {
            warn!(%e, shard_id = %id, "failed to remove shard from persistence");
        }
    }

    /// Flush any pending writes to disk.
    pub fn flush(&self) -> Result<(), redb::Error> {
        // redb transactions are already durable on commit.
        // This is a no-op but kept for API completeness.
        Ok(())
    }

    /// Get the next available shard ID.
    pub fn next_id(&self) -> ShardId {
        let max = self.shards.keys().map(|k| k.0).max().unwrap_or(0);
        ShardId(max + 1)
    }

    fn insert_entry(&mut self, id: ShardId, entry: ShardEntry) {
        // Build indices.
        if let Some(seed) = entry.info.planet_seed {
            self.planet_index.entry(seed).or_default().push(id);
        }
        if let Some(seed) = entry.info.system_seed {
            self.system_index.insert(seed, id);
        }
        if let Some(seed) = entry.info.galaxy_seed {
            self.galaxy_index.insert(seed, id);
        }
        if let Some(sid) = entry.info.ship_id {
            self.ship_index.insert(sid, id);
        }
        self.shards.insert(id, entry);
    }

    fn remove_from_indices(&mut self, id: ShardId) {
        if let Some(entry) = self.shards.get(&id) {
            if let Some(seed) = entry.info.planet_seed {
                if let Some(ids) = self.planet_index.get_mut(&seed) {
                    ids.retain(|&i| i != id);
                    if ids.is_empty() {
                        self.planet_index.remove(&seed);
                    }
                }
            }
            if let Some(seed) = entry.info.system_seed {
                if self.system_index.get(&seed) == Some(&id) {
                    self.system_index.remove(&seed);
                }
            }
            if let Some(seed) = entry.info.galaxy_seed {
                if self.galaxy_index.get(&seed) == Some(&id) {
                    self.galaxy_index.remove(&seed);
                }
            }
            if let Some(sid) = entry.info.ship_id {
                if self.ship_index.get(&sid) == Some(&id) {
                    self.ship_index.remove(&sid);
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::net::SocketAddr;

    fn test_endpoint() -> ShardEndpoint {
        ShardEndpoint {
            tcp_addr: "127.0.0.1:7777".parse::<SocketAddr>().unwrap(),
            udp_addr: "127.0.0.1:7778".parse::<SocketAddr>().unwrap(),
            quic_addr: "127.0.0.1:7779".parse::<SocketAddr>().unwrap(),
        }
    }

    fn planet_info(id: u64, seed: u64) -> ShardInfo {
        ShardInfo {
            id: ShardId(id),
            shard_type: ShardType::Planet,
            state: ShardState::Ready,
            endpoint: test_endpoint(),
            planet_seed: Some(seed),
            sectors: Some(vec![0, 1, 2, 3, 4, 5]),
            system_seed: None,
            ship_id: None,
            galaxy_seed: None,
            host_shard_id: None,
            launch_args: vec![],
        }
    }

    fn system_info(id: u64, seed: u64) -> ShardInfo {
        ShardInfo {
            id: ShardId(id),
            shard_type: ShardType::System,
            state: ShardState::Ready,
            endpoint: test_endpoint(),
            planet_seed: None,
            sectors: None,
            system_seed: Some(seed),
            ship_id: None,
            galaxy_seed: None,
            host_shard_id: None,
            launch_args: vec![],
        }
    }

    #[test]
    fn register_and_lookup() {
        let dir = tempfile::tempdir().unwrap();
        let mut reg = ShardRegistry::open(&dir.path().join("test.redb")).unwrap();

        reg.register(planet_info(1, 42), None);
        assert!(reg.get(ShardId(1)).is_some());
        assert_eq!(reg.list().len(), 1);
    }

    #[test]
    fn planet_index_lookup() {
        let dir = tempfile::tempdir().unwrap();
        let mut reg = ShardRegistry::open(&dir.path().join("test.redb")).unwrap();

        reg.register(planet_info(1, 42), None);
        reg.register(planet_info(2, 42), None); // second shard for same planet

        let results = reg.find_by_planet(42);
        assert_eq!(results.len(), 2);
        assert!(reg.find_by_planet(99).is_empty());
    }

    #[test]
    fn system_index_lookup() {
        let dir = tempfile::tempdir().unwrap();
        let mut reg = ShardRegistry::open(&dir.path().join("test.redb")).unwrap();

        reg.register(system_info(1, 100), None);
        assert!(reg.find_by_system(100).is_some());
        assert!(reg.find_by_system(200).is_none());
    }

    #[test]
    fn remove_cleans_indices() {
        let dir = tempfile::tempdir().unwrap();
        let mut reg = ShardRegistry::open(&dir.path().join("test.redb")).unwrap();

        reg.register(planet_info(1, 42), None);
        reg.register(system_info(2, 100), None);

        reg.remove(ShardId(1));
        assert!(reg.find_by_planet(42).is_empty());
        assert!(reg.get(ShardId(1)).is_none());

        reg.remove(ShardId(2));
        assert!(reg.find_by_system(100).is_none());
    }

    #[test]
    fn heartbeat_transitions_to_ready() {
        let dir = tempfile::tempdir().unwrap();
        let mut reg = ShardRegistry::open(&dir.path().join("test.redb")).unwrap();

        let mut info = planet_info(1, 42);
        info.state = ShardState::Provisioning;
        reg.register(info, None);

        reg.update_heartbeat(&ShardHeartbeat {
            shard_id: ShardId(1),
            tick_ms: 45.0,
            p99_tick_ms: 48.0,
            player_count: 1,
            chunk_count: 100,
        });

        assert_eq!(
            reg.get(ShardId(1)).unwrap().info.state,
            ShardState::Ready
        );
    }

    #[test]
    fn next_id() {
        let dir = tempfile::tempdir().unwrap();
        let mut reg = ShardRegistry::open(&dir.path().join("test.redb")).unwrap();

        assert_eq!(reg.next_id(), ShardId(1));
        reg.register(planet_info(1, 42), None);
        assert_eq!(reg.next_id(), ShardId(2));
        reg.register(planet_info(5, 43), None);
        assert_eq!(reg.next_id(), ShardId(6));
    }

    #[test]
    fn persistence_roundtrip() {
        let dir = tempfile::tempdir().unwrap();
        let db_path = dir.path().join("test.redb");

        {
            let mut reg = ShardRegistry::open(&db_path).unwrap();
            reg.register(planet_info(1, 42), None);
            reg.register(system_info(2, 100), None);
        }

        // Reopen — shards should be restored.
        let reg = ShardRegistry::open(&db_path).unwrap();
        assert_eq!(reg.list().len(), 2);
        // Restored shards start as Provisioning (don't know if alive).
        assert_eq!(
            reg.get(ShardId(1)).unwrap().info.state,
            ShardState::Provisioning
        );
    }
}
