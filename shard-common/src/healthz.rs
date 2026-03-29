use std::net::SocketAddr;

use axum::http::StatusCode;
use axum::routing::get;
use axum::Router;
use tokio_util::sync::CancellationToken;
use tracing::info;

/// Runs a minimal HTTP server for K8s liveness/readiness probes.
pub async fn run_healthz_server(addr: SocketAddr, cancel: CancellationToken) {
    let app = Router::new().route("/healthz", get(|| async { StatusCode::OK }));

    let listener = match tokio::net::TcpListener::bind(addr).await {
        Ok(l) => l,
        Err(e) => {
            tracing::warn!(%e, "failed to bind healthz server");
            return;
        }
    };

    info!(%addr, "healthz server ready");

    axum::serve(listener, app)
        .with_graceful_shutdown(cancel.cancelled_owned())
        .await
        .ok();
}
