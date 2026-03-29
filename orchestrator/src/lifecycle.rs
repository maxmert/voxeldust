use std::sync::Arc;
use std::time::Duration;

use tokio::sync::RwLock;
use tokio_util::sync::CancellationToken;
use tracing::{info, warn};

use voxeldust_core::shard_types::{ShardId, ShardState};

use crate::config::OrchestratorConfig;
use crate::provisioner::ShardProvisioner;
use crate::registry::ShardRegistry;

/// Runs the lifecycle management loop at 1-second intervals.
/// Detects dead shards, handles idle shutdown, triggers auto-restart.
pub async fn run_lifecycle_ticker(
    config: OrchestratorConfig,
    registry: Arc<RwLock<ShardRegistry>>,
    provisioner: Arc<dyn ShardProvisioner>,
    cancel: CancellationToken,
) {
    let mut interval = tokio::time::interval(Duration::from_secs(1));

    loop {
        tokio::select! {
            _ = cancel.cancelled() => {
                info!("lifecycle ticker shutting down");
                return;
            }
            _ = interval.tick() => {
                tick(&config, &registry, &provisioner).await;
            }
        }
    }
}

async fn tick(
    config: &OrchestratorConfig,
    registry: &Arc<RwLock<ShardRegistry>>,
    provisioner: &Arc<dyn ShardProvisioner>,
) {
    let mut reg = registry.write().await;
    let now = std::time::Instant::now();

    // Collect shard IDs to process (avoid borrow issues).
    let shard_ids: Vec<ShardId> = reg.entries().keys().copied().collect();

    let mut dead_shards = Vec::new();
    let mut idle_shards = Vec::new();

    for &id in &shard_ids {
        let entry = reg.entries().get(&id).unwrap();

        // Skip already-stopped shards.
        if entry.info.state == ShardState::Stopped {
            continue;
        }

        // Dead shard detection: no heartbeat for too long.
        if now.duration_since(entry.last_heartbeat) > config.heartbeat_timeout() {
            dead_shards.push(id);
            continue;
        }

        // Idle shutdown: 0 players for too long.
        if let Some(idle_since) = entry.idle_since {
            if entry.info.state == ShardState::Ready
                && now.duration_since(idle_since) > config.idle_shutdown_duration()
            {
                idle_shards.push(id);
            }
        }
    }

    // Process dead shards.
    for id in dead_shards {
        warn!(shard_id = %id, "shard missed heartbeat, marking as dead");
        let should_restart = {
            let entry = reg.get_mut(id).unwrap();
            entry.info.state = ShardState::Stopped;

            // Rate-limit restarts.
            let restart_window = config.restart_window();
            entry
                .restart_timestamps
                .retain(|t| now.duration_since(*t) < restart_window);
            let can_restart = (entry.restart_timestamps.len() as u32) < config.max_restarts
                && entry.launch_config.is_some();

            if can_restart {
                entry.restart_timestamps.push_back(now);
            }
            can_restart
        };

        if should_restart {
            let launch_config = reg.get(id).unwrap().launch_config.clone().unwrap();
            info!(shard_id = %id, "auto-restarting dead shard");
            match provisioner.start_shard(&launch_config).await {
                Ok(new_id) => {
                    info!(old_id = %id, new_id = %new_id, "shard restarted");
                }
                Err(e) => {
                    warn!(shard_id = %id, %e, "failed to restart shard");
                }
            }
        }
    }

    // Process idle shards.
    for id in idle_shards {
        info!(shard_id = %id, "shard idle too long, shutting down");
        if let Some(entry) = reg.get_mut(id) {
            entry.info.state = ShardState::Draining;
        }
        if let Err(e) = provisioner.stop_shard(id).await {
            warn!(shard_id = %id, %e, "failed to stop idle shard");
        }
        if let Some(entry) = reg.get_mut(id) {
            entry.info.state = ShardState::Stopped;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::provisioner::LocalProvisioner;
    use std::net::SocketAddr;
    use voxeldust_core::shard_types::*;

    fn test_endpoint() -> ShardEndpoint {
        ShardEndpoint {
            tcp_addr: "127.0.0.1:7777".parse::<SocketAddr>().unwrap(),
            udp_addr: "127.0.0.1:7778".parse::<SocketAddr>().unwrap(),
            quic_addr: "127.0.0.1:7779".parse::<SocketAddr>().unwrap(),
        }
    }

    #[tokio::test]
    async fn dead_shard_detection() {
        let dir = tempfile::tempdir().unwrap();
        let mut reg = ShardRegistry::open(&dir.path().join("test.redb")).unwrap();

        // Register a shard with last_heartbeat far in the past.
        reg.register(
            ShardInfo {
                id: ShardId(1),
                shard_type: ShardType::Planet,
                state: ShardState::Ready,
                endpoint: test_endpoint(),
                planet_seed: Some(42),
                sectors: Some(vec![0, 1, 2, 3, 4, 5]),
                system_seed: None,
                ship_id: None,
                galaxy_seed: None,
                host_shard_id: None,
                launch_args: vec![],
            },
            None,
        );

        // Backdate the heartbeat.
        reg.get_mut(ShardId(1)).unwrap().last_heartbeat =
            std::time::Instant::now() - Duration::from_secs(20);

        let registry = Arc::new(RwLock::new(reg));
        let provisioner: Arc<dyn ShardProvisioner> = Arc::new(LocalProvisioner::new());

        let config = OrchestratorConfig {
            http_addr: "127.0.0.1:0".parse().unwrap(),
            heartbeat_addr: "127.0.0.1:0".parse().unwrap(),
            db_path: dir.path().join("test.redb"),
            heartbeat_timeout_secs: 10,
            idle_shutdown_secs: 300,
            max_restarts: 3,
            restart_window_secs: 60,
            provisioner_mode: "local".to_string(),
            shard_image: "stub-shard:latest".to_string(),
            shard_namespace: "voxeldust".to_string(),
            shard_port_range_start: 10000,
            shard_port_range_end: 10100,
            shard_advertise_host: "127.0.0.1".to_string(),
        };

        tick(&config, &registry, &provisioner).await;

        let reg = registry.read().await;
        assert_eq!(
            reg.get(ShardId(1)).unwrap().info.state,
            ShardState::Stopped
        );
    }

    #[tokio::test]
    async fn idle_shutdown() {
        let dir = tempfile::tempdir().unwrap();
        let mut reg = ShardRegistry::open(&dir.path().join("test.redb")).unwrap();

        reg.register(
            ShardInfo {
                id: ShardId(1),
                shard_type: ShardType::Planet,
                state: ShardState::Ready,
                endpoint: test_endpoint(),
                planet_seed: Some(42),
                sectors: Some(vec![0, 1, 2, 3, 4, 5]),
                system_seed: None,
                ship_id: None,
                galaxy_seed: None,
                host_shard_id: None,
                launch_args: vec![],
            },
            None,
        );

        // Set idle since far in the past.
        reg.get_mut(ShardId(1)).unwrap().idle_since =
            Some(std::time::Instant::now() - Duration::from_secs(400));

        let registry = Arc::new(RwLock::new(reg));
        let provisioner: Arc<dyn ShardProvisioner> = Arc::new(LocalProvisioner::new());

        let config = OrchestratorConfig {
            http_addr: "127.0.0.1:0".parse().unwrap(),
            heartbeat_addr: "127.0.0.1:0".parse().unwrap(),
            db_path: dir.path().join("test.redb"),
            heartbeat_timeout_secs: 10,
            idle_shutdown_secs: 300,
            max_restarts: 3,
            restart_window_secs: 60,
            provisioner_mode: "local".to_string(),
            shard_image: "stub-shard:latest".to_string(),
            shard_namespace: "voxeldust".to_string(),
            shard_port_range_start: 10000,
            shard_port_range_end: 10100,
            shard_advertise_host: "127.0.0.1".to_string(),
        };

        tick(&config, &registry, &provisioner).await;

        let reg = registry.read().await;
        assert_eq!(
            reg.get(ShardId(1)).unwrap().info.state,
            ShardState::Stopped
        );
    }
}
