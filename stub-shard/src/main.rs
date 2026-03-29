use std::net::SocketAddr;

use clap::Parser;
use tracing::info;

use voxeldust_core::shard_types::{ShardId, ShardType};
use voxeldust_shard_common::harness::{ShardHarness, ShardHarnessConfig};

#[derive(Parser, Debug)]
#[command(name = "stub-shard", about = "Voxeldust stub shard for testing")]
struct Args {
    /// Unique shard identifier.
    #[arg(long)]
    shard_id: u64,

    /// Shard type: "planet" or "system".
    #[arg(long, default_value = "planet")]
    shard_type: String,

    /// Planet or system seed.
    #[arg(long, default_value = "42")]
    seed: u64,

    /// Orchestrator HTTP URL.
    #[arg(long, default_value = "http://127.0.0.1:8080")]
    orchestrator: String,

    /// Orchestrator heartbeat UDP address (host:port, supports DNS names).
    #[arg(long, default_value = "127.0.0.1:9090")]
    orchestrator_heartbeat: String,

    /// TCP port for client connections.
    #[arg(long, default_value = "7777")]
    tcp_port: u16,

    /// UDP port for fast messages.
    #[arg(long, default_value = "7778")]
    udp_port: u16,

    /// QUIC port for inter-shard communication.
    #[arg(long, default_value = "7779")]
    quic_port: u16,

    /// Healthz HTTP port.
    #[arg(long, default_value = "8081")]
    healthz_port: u16,

    /// Host to advertise to orchestrator (overrides bind address in endpoints).
    /// Set to 127.0.0.1 in K8s with hostNetwork for k3d port mapping.
    #[arg(long)]
    advertise_host: Option<String>,

    /// For ship shards: the shard ID managing this ship's exterior.
    #[arg(long)]
    host_shard: Option<u64>,
}

fn main() {
    let args = Args::parse();

    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "info".into()),
        )
        .init();

    let shard_type = match args.shard_type.as_str() {
        "planet" => ShardType::Planet,
        "system" => ShardType::System,
        "ship" => ShardType::Ship,
        "galaxy" => ShardType::Galaxy,
        other => {
            eprintln!("unknown shard type: {other}");
            std::process::exit(1);
        }
    };

    let (planet_seed, system_seed, galaxy_seed) = match shard_type {
        ShardType::Planet => (Some(args.seed), None, None),
        ShardType::System => (None, Some(args.seed), None),
        ShardType::Galaxy => (None, None, Some(args.seed)),
        ShardType::Ship => (None, None, None),
    };

    let bind = "0.0.0.0";
    let config = ShardHarnessConfig {
        shard_id: ShardId(args.shard_id),
        shard_type,
        tcp_addr: format!("{bind}:{}", args.tcp_port).parse().unwrap(),
        udp_addr: format!("{bind}:{}", args.udp_port).parse().unwrap(),
        quic_addr: format!("{bind}:{}", args.quic_port).parse().unwrap(),
        orchestrator_url: args.orchestrator,
        orchestrator_heartbeat_addr: args.orchestrator_heartbeat,
        healthz_addr: format!("{bind}:{}", args.healthz_port).parse().unwrap(),
        planet_seed,
        system_seed,
        ship_id: None,
        galaxy_seed,
        host_shard_id: args.host_shard.map(ShardId),
        advertise_host: args.advertise_host,
    };

    info!(
        shard_id = args.shard_id,
        shard_type = %shard_type,
        seed = args.seed,
        "stub shard starting"
    );

    let rt = tokio::runtime::Runtime::new().expect("failed to create tokio runtime");
    rt.block_on(async {
        let mut harness = ShardHarness::new(config);

        // Add a minimal tick system that drains connect events.
        let mut connect_rx = std::mem::replace(
            &mut harness.connect_rx,
            tokio::sync::mpsc::unbounded_channel().1,
        );

        harness.add_system("drain_connects", move || {
            while let Ok(event) = connect_rx.try_recv() {
                info!(
                    player = %event.connection.player_name,
                    session = event.connection.session_token.0,
                    "player connected (stub: no game logic)"
                );
            }
        });

        harness.run().await;
    });
}
