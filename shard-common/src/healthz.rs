use std::net::SocketAddr;

use axum::http::StatusCode;
use axum::routing::get;
use axum::Router;
use tokio_util::sync::CancellationToken;
use tracing::info;

use crate::observability;

/// Runs the shard's HTTP service: K8s liveness/readiness probe at
/// `/healthz` plus the Prometheus exposition format at `/metrics`.
///
/// The `/metrics` route is mounted iff
/// [`observability::install_prometheus_recorder`] was called earlier
/// in process startup (typically the top of `main`). When no recorder
/// is installed, the route is omitted entirely — a 404 is more honest
/// than serving an empty body that pretends scraping is wired.
pub async fn run_healthz_server(addr: SocketAddr, cancel: CancellationToken) {
    let mut app = Router::new().route("/healthz", get(|| async { StatusCode::OK }));
    let metrics_wired = if let Some(handle) = observability::current_handle() {
        // Each request gets a fresh clone of the handle (cheap — it's
        // `Arc<Inner>` underneath). The closure must own the clone
        // outright so the returned response captures no borrowed
        // state from the closure body.
        app = app.route(
            "/metrics",
            get(move || {
                let handle = handle.clone();
                async move {
                    (
                        StatusCode::OK,
                        [(
                            axum::http::header::CONTENT_TYPE,
                            "text/plain; version=0.0.4",
                        )],
                        handle.render(),
                    )
                }
            }),
        );
        true
    } else {
        false
    };

    let listener = match tokio::net::TcpListener::bind(addr).await {
        Ok(l) => l,
        Err(e) => {
            tracing::warn!(%e, "failed to bind healthz server");
            return;
        }
    };

    info!(
        %addr,
        metrics = metrics_wired,
        "healthz server ready"
    );

    axum::serve(listener, app)
        .with_graceful_shutdown(cancel.cancelled_owned())
        .await
        .ok();
}
