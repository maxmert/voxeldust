use tokio_util::sync::CancellationToken;
use tracing::info;

/// Listens for SIGTERM/SIGINT and cancels the token.
pub async fn wait_for_shutdown_signal(cancel: CancellationToken) {
    tokio::signal::ctrl_c()
        .await
        .expect("failed to listen for ctrl-c");
    info!("shutdown signal received");
    cancel.cancel();
}
