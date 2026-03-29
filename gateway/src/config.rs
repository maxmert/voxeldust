use std::net::SocketAddr;
use std::path::PathBuf;

use clap::Parser;

#[derive(Parser, Debug, Clone)]
#[command(name = "gateway", about = "Voxeldust client gateway")]
pub struct GatewayConfig {
    /// TCP listen address for client connections.
    #[arg(long, default_value = "0.0.0.0:7777")]
    pub listen_addr: SocketAddr,

    /// Orchestrator HTTP URL.
    #[arg(long, default_value = "http://127.0.0.1:8080")]
    pub orchestrator_url: String,

    /// Path to session persistence database.
    #[arg(long, default_value = "gateway.redb")]
    pub db_path: PathBuf,

    /// Universe seed (root of the entire seed hierarchy).
    #[arg(long, default_value = "1")]
    pub universe_seed: u64,

    /// Default galaxy index for new players.
    #[arg(long, default_value = "0")]
    pub default_galaxy_index: u32,

    /// Default star index for new player spawn.
    #[arg(long, default_value = "0")]
    pub default_star_index: u32,
}
