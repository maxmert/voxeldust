use bevy_app::prelude::*;
use bevy_ecs::prelude::*;
use clap::Parser;
use tracing::info;

use voxeldust_core::shard_types::{ShardId, ShardType};
use voxeldust_shard_common::harness::{NetworkBridge, ShardHarness, ShardHarnessConfig};

#[derive(Parser, Debug)]
#[command(name = "stub-shard", about = "Voxeldust stub shard for testing")]
struct Args {
    #[arg(long)]
    shard_id: u64,
    #[arg(long, default_value = "planet")]
    shard_type: String,
    #[arg(long, default_value = "42")]
    seed: u64,
    #[arg(long, default_value = "http://127.0.0.1:8080")]
    orchestrator: String,
    #[arg(long, default_value = "127.0.0.1:9090")]
    orchestrator_heartbeat: String,
    #[arg(long, default_value = "7777")]
    tcp_port: u16,
    #[arg(long, default_value = "7778")]
    udp_port: u16,
    #[arg(long, default_value = "7779")]
    quic_port: u16,
    #[arg(long, default_value = "8081")]
    healthz_port: u16,
    #[arg(long)]
    advertise_host: Option<String>,
    #[arg(long)]
    host_shard: Option<u64>,
}

fn drain_connects(mut bridge: ResMut<NetworkBridge>) {
    while let Ok(event) = bridge.connect_rx.try_recv() {
        info!(
            player = %event.connection.player_name,
            session = event.connection.session_token.0,
            "player connected (stub: no game logic)"
        );
    }
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
        let harness = ShardHarness::new(config);

        let mut app = App::new();
        app.add_systems(Update, drain_connects);

        harness.run_ecs(app).await;
    });
}
