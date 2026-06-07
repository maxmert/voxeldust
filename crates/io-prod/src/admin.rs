//! The read-only admin HTTP shell (PLAN.md P0: "curl the orchestrator admin endpoint
//! and get an empty-but-shaped directory/saga dump"). The orchestrator bin mounts
//! this at P1 with its real state source; from day one the SHAPE is served so 2am
//! debugging is `curl`, never log-grepping.
//!
//! Routes:
//! - `GET /admin/snapshot` — the [`AdminSnapshot`] contract from `vd-wire` as JSON.
//! - `GET /metrics` — Prometheus exposition of the reviewed metric-name registry
//!   (values come live as subsystems land; the names are frozen now).
//!
//! READ-ONLY by construction: no mutating route exists to misuse.

use std::fmt::Write as _;
use std::sync::Arc;

use axum::Json;
use axum::extract::State;
use axum::routing::get;
use vd_wire::admin::{AdminSnapshot, metric_names};

/// Whatever can produce the current admin snapshot (the orchestrator's state at P1;
/// a fixed shaped-empty snapshot in tests and before subsystems land).
pub trait SnapshotSource: Send + Sync + 'static {
    fn snapshot(&self) -> AdminSnapshot;
}

/// A constant source: serves the same snapshot forever (the P0 shell).
pub struct FixedSnapshot(pub AdminSnapshot);

impl SnapshotSource for FixedSnapshot {
    fn snapshot(&self) -> AdminSnapshot {
        self.0.clone()
    }
}

/// Build the admin router around a snapshot source.
pub fn admin_router(source: Arc<dyn SnapshotSource>) -> axum::Router {
    axum::Router::new()
        .route("/admin/snapshot", get(snapshot_handler))
        .route("/metrics", get(metrics_handler))
        .with_state(source)
}

async fn snapshot_handler(State(source): State<Arc<dyn SnapshotSource>>) -> Json<AdminSnapshot> {
    Json(source.snapshot())
}

async fn metrics_handler() -> String {
    render_metrics_shell()
}

/// Prometheus text exposition of every registered metric name. Until live
/// instrumentation lands (P7), each reports 0 — shaped, scrapeable, honest.
#[must_use]
pub fn render_metrics_shell() -> String {
    let mut out = String::new();
    for name in metric_names::ALL {
        let _ = writeln!(out, "# TYPE {name} untyped");
        let _ = writeln!(out, "{name} 0");
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use tokio::io::{AsyncReadExt, AsyncWriteExt};
    use vd_core::{EpochId, UniverseTick};

    /// One blocking HTTP/1.1 GET against the served router (no client dependency —
    /// the raw bytes ARE the thing being verified: this is what `curl` sends).
    fn http_get(path: &str) -> (String, String) {
        let rt = tokio::runtime::Builder::new_multi_thread()
            .worker_threads(2)
            .enable_all()
            .build()
            .expect("tokio runtime");
        rt.block_on(async move {
            let source = Arc::new(FixedSnapshot(AdminSnapshot::shaped_empty(
                UniverseTick(7),
                EpochId(3),
            )));
            let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
                .await
                .expect("bind");
            let addr = listener.local_addr().expect("addr");
            tokio::spawn(async move {
                axum::serve(listener, admin_router(source))
                    .await
                    .expect("serve");
            });

            let mut conn = tokio::net::TcpStream::connect(addr).await.expect("connect");
            let request =
                format!("GET {path} HTTP/1.1\r\nHost: localhost\r\nConnection: close\r\n\r\n");
            conn.write_all(request.as_bytes()).await.expect("request");
            let mut response = Vec::new();
            conn.read_to_end(&mut response).await.expect("response");
            let text = String::from_utf8(response).expect("utf8 response");
            let (head, body) = text
                .split_once("\r\n\r\n")
                .expect("HTTP head/body separator");
            (head.to_owned(), body.to_owned())
        })
    }

    #[test]
    fn snapshot_endpoint_serves_the_shaped_contract_as_json() {
        let (head, body) = http_get("/admin/snapshot");
        assert!(head.starts_with("HTTP/1.1 200"), "got: {head}");
        assert!(head.to_ascii_lowercase().contains("application/json"));
        let snapshot: AdminSnapshot = serde_json::from_str(&body).expect("contract JSON");
        assert_eq!(
            snapshot,
            AdminSnapshot::shaped_empty(UniverseTick(7), EpochId(3))
        );
    }

    #[test]
    fn metrics_endpoint_exposes_every_registered_name() {
        let (head, body) = http_get("/metrics");
        assert!(head.starts_with("HTTP/1.1 200"), "got: {head}");
        for name in metric_names::ALL {
            assert!(body.contains(name), "missing metric: {name}");
        }
    }

    #[test]
    fn unknown_routes_are_not_found_nothing_mutable_exists() {
        let (head, _) = http_get("/admin/wipe-everything");
        assert!(head.starts_with("HTTP/1.1 404"), "got: {head}");
    }
}
