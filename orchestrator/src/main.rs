use std::sync::Arc;
use std::time::SystemTime;

use clap::Parser;
use tokio::sync::RwLock;
use tokio_util::sync::CancellationToken;
use tracing::info;

use orchestrator::config::OrchestratorConfig;
use orchestrator::heartbeat::run_heartbeat_listener;
use orchestrator::http_api::build_router;
use orchestrator::lifecycle::run_lifecycle_ticker;
use orchestrator::k8s_provisioner::{KubernetesProvisioner, KubernetesProvisionerConfig};
use orchestrator::provisioner::{LocalProvisioner, ShardProvisioner};
use orchestrator::registry::ShardRegistry;

fn main() {
    let config = OrchestratorConfig::parse();

    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "info".into()),
        )
        .init();

    let rt = tokio::runtime::Runtime::new().expect("failed to create tokio runtime");
    rt.block_on(run(config));
}

async fn run(config: OrchestratorConfig) {
    info!("orchestrator starting");
    info!(http = %config.http_addr, heartbeat = %config.heartbeat_addr, "listening");

    let registry = ShardRegistry::open(&config.db_path).expect("failed to open registry db");
    let registry = Arc::new(RwLock::new(registry));

    let provisioner: Arc<dyn ShardProvisioner> = match config.provisioner_mode.as_str() {
        "kubernetes" => {
            let k8s_config = KubernetesProvisionerConfig {
                namespace: config.shard_namespace.clone(),
                shard_image: config.shard_image.clone(),
                port_range_start: config.shard_port_range_start,
                port_range_end: config.shard_port_range_end,
                orchestrator_url: format!("http://orchestrator.{}.svc.cluster.local:8080", config.shard_namespace),
                orchestrator_heartbeat_addr: format!("orchestrator.{}.svc.cluster.local:9090", config.shard_namespace),
                advertise_host: config.shard_advertise_host.clone(),
            };
            Arc::new(
                KubernetesProvisioner::new(k8s_config)
                    .await
                    .expect("failed to initialize K8s provisioner"),
            )
        }
        _ => Arc::new(LocalProvisioner::new()),
    };

    // Universe epoch: the reference point for deterministic celestial time.
    // All shards derive celestial_time = (now - epoch) * time_scale.
    // For persistence across restarts, this would be stored in the registry DB.
    // For now, epoch = orchestrator start time (resets each dev cycle).
    let universe_epoch_ms = SystemTime::now()
        .duration_since(SystemTime::UNIX_EPOCH)
        .unwrap()
        .as_millis() as u64;
    info!(universe_epoch_ms, "universe epoch established");

    let cancel = CancellationToken::new();

    let http_registry = registry.clone();
    let http_provisioner = provisioner.clone();
    let http_cancel = cancel.clone();
    let http_addr = config.http_addr;
    let http_handle = tokio::spawn(async move {
        let app = build_router(http_registry, http_provisioner, universe_epoch_ms);
        let listener = tokio::net::TcpListener::bind(http_addr)
            .await
            .expect("failed to bind HTTP");
        info!(%http_addr, "HTTP API ready");
        axum::serve(listener, app)
            .with_graceful_shutdown(http_cancel.cancelled_owned())
            .await
            .expect("HTTP server error");
    });

    let hb_registry = registry.clone();
    let hb_cancel = cancel.clone();
    let hb_addr = config.heartbeat_addr;
    let hb_handle = tokio::spawn(async move {
        run_heartbeat_listener(hb_addr, hb_registry, hb_cancel).await;
    });

    let lc_registry = registry.clone();
    let lc_provisioner = provisioner.clone();
    let lc_cancel = cancel.clone();
    let lc_config = config.clone();
    let lc_handle = tokio::spawn(async move {
        run_lifecycle_ticker(lc_config, lc_registry, lc_provisioner, lc_cancel).await;
    });

    // Wait for shutdown signal.
    tokio::signal::ctrl_c()
        .await
        .expect("failed to listen for ctrl-c");
    info!("shutdown signal received");
    cancel.cancel();

    let _ = tokio::join!(http_handle, hb_handle, lc_handle);

    // Final persistence flush.
    let reg = registry.read().await;
    if let Err(e) = reg.flush() {
        tracing::error!(%e, "failed to flush registry on shutdown");
    }
    info!("orchestrator stopped");
}
