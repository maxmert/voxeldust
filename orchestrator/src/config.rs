use std::net::SocketAddr;
use std::path::PathBuf;
use std::time::Duration;

use clap::Parser;

#[derive(Parser, Debug, Clone)]
#[command(name = "orchestrator", about = "Voxeldust shard orchestrator")]
pub struct OrchestratorConfig {
    /// HTTP API listen address.
    #[arg(long, default_value = "0.0.0.0:8080")]
    pub http_addr: SocketAddr,

    /// UDP heartbeat listener address.
    #[arg(long, default_value = "0.0.0.0:9090")]
    pub heartbeat_addr: SocketAddr,

    /// Path to redb database file.
    #[arg(long, default_value = "orchestrator.redb")]
    pub db_path: PathBuf,

    /// Heartbeat timeout in seconds. Shards missing heartbeats for this long
    /// are considered dead.
    #[arg(long, default_value = "10")]
    pub heartbeat_timeout_secs: u64,

    /// Idle shutdown duration in seconds. Shards with 0 players for this long
    /// are shut down.
    #[arg(long, default_value = "300")]
    pub idle_shutdown_secs: u64,

    /// Maximum number of restart attempts per shard within the restart window.
    #[arg(long, default_value = "3")]
    pub max_restarts: u32,

    /// Restart rate-limiting window in seconds.
    #[arg(long, default_value = "60")]
    pub restart_window_secs: u64,

    /// Provisioner mode: "local" for child processes, "kubernetes" for K8s pods.
    #[arg(long, default_value = "local")]
    pub provisioner_mode: String,

    /// Container image for shard pods (kubernetes mode only).
    #[arg(long, default_value = "voxeldust-stub-shard:latest")]
    pub shard_image: String,

    /// K8s namespace for shard pods.
    #[arg(long, default_value = "voxeldust")]
    pub shard_namespace: String,

    /// Start of dynamic port range for shard pods (4 ports per shard).
    #[arg(long, default_value = "10000")]
    pub shard_port_range_start: u16,

    /// End of dynamic port range (exclusive).
    #[arg(long, default_value = "10100")]
    pub shard_port_range_end: u16,

    /// Host address for shard endpoints advertised to clients.
    #[arg(long, default_value = "127.0.0.1")]
    pub shard_advertise_host: String,
}

impl OrchestratorConfig {
    pub fn heartbeat_timeout(&self) -> Duration {
        Duration::from_secs(self.heartbeat_timeout_secs)
    }

    pub fn idle_shutdown_duration(&self) -> Duration {
        Duration::from_secs(self.idle_shutdown_secs)
    }

    pub fn restart_window(&self) -> Duration {
        Duration::from_secs(self.restart_window_secs)
    }
}
