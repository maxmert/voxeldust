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

use arc_swap::ArcSwap;
use axum::Json;
use axum::extract::State;
use axum::routing::get;
use vd_wire::admin::{AdminSnapshot, metric_names};

use crate::mesh::MeshControl;

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

/// A LIVE source backed by a lock-free cell the sim thread republishes after every tick (RLM RG-4). The
/// HTTP scrape task holds `Arc<PublishedSnapshot>` and only ever `load()`s — it never touches the sim
/// thread, so a slow `curl` cannot stall a tick. Both the orchestrator (directory/saga/lease view) and the
/// gateway (session/dynamic-shard view) publish through this ONE type — a bin that grew its own private
/// `Published` would drift from this contract.
pub struct PublishedSnapshot(pub Arc<ArcSwap<AdminSnapshot>>);

impl SnapshotSource for PublishedSnapshot {
    fn snapshot(&self) -> AdminSnapshot {
        self.0.load().as_ref().clone()
    }
}

/// Whatever produces the current metric VALUES (the live mesh counters). Mirrors
/// [`SnapshotSource`]. `values()` MUST be cheap + non-blocking — the mesh impl is a pure
/// atomic load off the hot path.
pub trait MetricsSource: Send + Sync + 'static {
    fn values(&self) -> MetricValues;
}

/// The rendered metric values — a FLAT struct (not a map) so [`render_metrics`] is a
/// straight-line pairing against `metric_names` with no per-key lookup and no magic strings
/// at the render site. One field per live mesh counter (R-4d M4); the not-yet-wired
/// saga/clock/fsync gauges report 0 until their subsystems land.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct MetricValues {
    pub datagrams_dropped_too_large: u64,
    pub datagrams_dropped_send: u64,
    pub inbound_dropped_reliable: u64,
    pub inbound_dropped_unreliable: u64,
    pub stale_incarnation_drop: u64,
    pub stale_epoch_drop: u64,
    pub dedup_drop: u64,
    pub gap_drop: u64,
    pub reliable_shed: u64,
    pub reliable_acked: u64,
    /// CA-1 (RLM RG-4): unbooked dial-in peers dropped because the learned-peer table was at capacity —
    /// the reactive-greeting mesh's shed-loud counter. A nonzero value means a demand shard's return
    /// connection was refused; on a cloud tier it is the signal to raise the learned-peer cap.
    pub learned_peers_rejected: u64,
}

/// The LIVE mesh metrics source: reads [`MeshControl::stats`] per scrape (a pure atomic
/// load — never the hot path, never blocking). The orchestrator holds the `MeshControl`
/// and wires this into its admin serve.
pub struct MeshMetrics(pub Arc<MeshControl>);

impl MetricsSource for MeshMetrics {
    fn values(&self) -> MetricValues {
        let s = self.0.stats();
        MetricValues {
            datagrams_dropped_too_large: s.datagrams_dropped_too_large,
            datagrams_dropped_send: s.datagrams_dropped_send,
            inbound_dropped_reliable: s.inbound_dropped_reliable,
            inbound_dropped_unreliable: s.inbound_dropped_unreliable,
            stale_incarnation_drop: s.stale_incarnation_drop,
            stale_epoch_drop: s.stale_epoch_drop,
            dedup_drop: s.dedup_drop,
            gap_drop: s.gap_drop,
            reliable_shed: s.reliable_shed,
            reliable_acked: s.reliable_acked,
            learned_peers_rejected: s.learned_peers_rejected,
        }
    }
}

/// A constant metrics source (the P0 shell / tests): shaped, scrapeable, fixed.
pub struct FixedMetrics(pub MetricValues);

impl MetricsSource for FixedMetrics {
    fn values(&self) -> MetricValues {
        self.0
    }
}

/// The combined admin state: the snapshot source and the metrics source (R-4d M4). Both are
/// `Arc<dyn ...>` trait objects so the orchestrator wires live sources and tests wire fixed ones.
#[derive(Clone)]
struct AdminState {
    snapshot: Arc<dyn SnapshotSource>,
    metrics: Arc<dyn MetricsSource>,
}

/// Build the admin router around a snapshot source AND a metrics source.
pub fn admin_router(
    snapshot: Arc<dyn SnapshotSource>,
    metrics: Arc<dyn MetricsSource>,
) -> axum::Router {
    axum::Router::new()
        .route("/admin/snapshot", get(snapshot_handler))
        .route("/metrics", get(metrics_handler))
        .with_state(AdminState { snapshot, metrics })
}

async fn snapshot_handler(State(state): State<AdminState>) -> Json<AdminSnapshot> {
    Json(state.snapshot.snapshot())
}

async fn metrics_handler(State(state): State<AdminState>) -> String {
    render_metrics(&state.metrics.values())
}

/// Prometheus text exposition of every registered metric name (R-4d M4). The live mesh
/// counters report their current values; the not-yet-wired saga/clock/fsync gauges report 0 —
/// shaped, scrapeable, honest, never a 404.
///
/// The `MetricValues` is EXHAUSTIVELY DESTRUCTURED (no `..`): a new field is a COMPILE ERROR
/// until it is paired to a `metric_names` entry here — the registry-vs-struct completeness guard.
#[must_use]
pub fn render_metrics(values: &MetricValues) -> String {
    let MetricValues {
        datagrams_dropped_too_large,
        datagrams_dropped_send,
        inbound_dropped_reliable,
        inbound_dropped_unreliable,
        stale_incarnation_drop,
        stale_epoch_drop,
        dedup_drop,
        gap_drop,
        reliable_shed,
        reliable_acked,
        learned_peers_rejected,
    } = *values;
    let live: [(&str, u64); 11] = [
        (
            metric_names::DATAGRAMS_DROPPED_TOO_LARGE,
            datagrams_dropped_too_large,
        ),
        (
            metric_names::DATAGRAMS_DROPPED_SEND_TOTAL,
            datagrams_dropped_send,
        ),
        (
            metric_names::INBOUND_DROPPED_RELIABLE_TOTAL,
            inbound_dropped_reliable,
        ),
        (
            metric_names::INBOUND_DROPPED_UNRELIABLE_TOTAL,
            inbound_dropped_unreliable,
        ),
        (
            metric_names::STALE_INCARNATION_DROP_TOTAL,
            stale_incarnation_drop,
        ),
        (metric_names::STALE_EPOCH_DROP_TOTAL, stale_epoch_drop),
        (metric_names::DEDUP_DROP_TOTAL, dedup_drop),
        (metric_names::GAP_DROP_TOTAL, gap_drop),
        (metric_names::RELIABLE_SHED_TOTAL, reliable_shed),
        (metric_names::RELIABLE_ACKED_TOTAL, reliable_acked),
        (
            metric_names::LEARNED_PEERS_REJECTED_TOTAL,
            learned_peers_rejected,
        ),
    ];
    let mut out = String::new();
    for name in metric_names::ALL {
        // The named value if wired, else 0 (the saga/clock/fsync gauges land later) — keeps the
        // shaped-empty, never-a-404 P0 guarantee.
        let value = live
            .iter()
            .find(|(n, _)| *n == *name)
            .map_or(0, |(_, v)| *v);
        let _ = writeln!(out, "# TYPE {name} untyped");
        let _ = writeln!(out, "{name} {value}");
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use tokio::io::{AsyncReadExt, AsyncWriteExt};
    use vd_core::{EpochId, UniverseTick};

    /// Known-nonzero fixed metrics the served router exposes, so the `/metrics` scrape test
    /// asserts the values FLOW through (not just that the names are present).
    fn seeded_metrics() -> MetricValues {
        MetricValues {
            reliable_shed: 7,
            reliable_acked: 42,
            datagrams_dropped_send: 5,
            gap_drop: 0,
            ..MetricValues::default()
        }
    }

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
            let metrics = Arc::new(FixedMetrics(seeded_metrics()));
            let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
                .await
                .expect("bind");
            let addr = listener.local_addr().expect("addr");
            tokio::spawn(async move {
                axum::serve(listener, admin_router(source, metrics))
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
    fn a_published_snapshot_serves_the_latest_republished_cell() {
        // RLM RG-4: the live source a bin republishes after every tick. A read sees whatever the sim thread
        // last stored — so a `curl` after tick N observes tick N's snapshot, never a stale clone.
        let cell = Arc::new(ArcSwap::from_pointee(AdminSnapshot::shaped_empty(
            UniverseTick(1),
            EpochId(2),
        )));
        let source = PublishedSnapshot(Arc::clone(&cell));
        assert_eq!(
            source.snapshot(),
            AdminSnapshot::shaped_empty(UniverseTick(1), EpochId(2))
        );
        cell.store(Arc::new(AdminSnapshot::shaped_empty(
            UniverseTick(9),
            EpochId(2),
        )));
        assert_eq!(
            source.snapshot(),
            AdminSnapshot::shaped_empty(UniverseTick(9), EpochId(2)),
            "the read reflects the republished cell, not the boot snapshot"
        );
    }

    #[test]
    fn metrics_endpoint_exposes_every_registered_name_and_live_values() {
        let (head, body) = http_get("/metrics");
        assert!(head.starts_with("HTTP/1.1 200"), "got: {head}");
        for name in metric_names::ALL {
            assert!(body.contains(name), "missing metric: {name}");
        }
        // The LIVE mesh values flow through the served scrape (R-4d M4), not hardcoded 0.
        assert!(body.contains("vd_reliable_shed_total 7"), "body: {body}");
        assert!(body.contains("vd_reliable_acked_total 42"), "body: {body}");
        assert!(
            body.contains("vd_datagrams_dropped_send_total 5"),
            "body: {body}"
        );
        // A wired-but-zero counter renders 0 (not omitted), and an UN-wired gauge is still 0.
        assert!(body.contains("vd_gap_drop_total 0"), "body: {body}");
        assert!(body.contains("vd_saga_stuck 0"), "body: {body}");
    }

    #[test]
    fn render_metrics_pairs_live_values_and_zeroes_the_unwired() {
        // Distinct values in every field prove the destructure pairs each to the RIGHT name (no
        // transposition), and exercises BOTH `find` branches: hit (a wired mesh counter) and miss
        // (an ALL name with no MetricValues field, e.g. a saga/clock gauge, renders 0).
        let out = render_metrics(&MetricValues {
            datagrams_dropped_too_large: 1,
            datagrams_dropped_send: 2,
            inbound_dropped_reliable: 3,
            inbound_dropped_unreliable: 4,
            stale_incarnation_drop: 5,
            stale_epoch_drop: 6,
            dedup_drop: 7,
            gap_drop: 8,
            reliable_shed: 9,
            reliable_acked: 10,
            learned_peers_rejected: 11,
        });
        assert!(out.contains("vd_datagrams_dropped_too_large 1"));
        assert!(out.contains("vd_datagrams_dropped_send_total 2"));
        assert!(out.contains("vd_inbound_dropped_reliable_total 3"));
        assert!(out.contains("vd_inbound_dropped_unreliable_total 4"));
        assert!(out.contains("vd_stale_incarnation_drop_total 5"));
        assert!(out.contains("vd_stale_epoch_drop_total 6"));
        assert!(out.contains("vd_dedup_drop_total 7"));
        assert!(out.contains("vd_gap_drop_total 8"));
        assert!(out.contains("vd_reliable_shed_total 9"));
        assert!(out.contains("vd_reliable_acked_total 10"));
        assert!(out.contains("vd_learned_peers_rejected_total 11"));
        // The `find`-miss branch: an ALL name with no live pairing renders 0.
        assert!(out.contains("vd_transfers_in_flight 0"));
    }

    #[test]
    fn render_emits_exactly_one_value_line_per_registered_metric() {
        // The completeness invariant the render loop must hold: every `metric_names::ALL` entry is
        // emitted EXACTLY ONCE — never dropped, never duplicated. (The per-name VALUE correctness +
        // the field↔name pairing are guarded by `render_metrics_pairs_live_values_and_zeroes_the_
        // unwired`; the struct↔registry field completeness is a COMPILE error via the exhaustive
        // destructure. A `live`-name typo is caught by the value asserts, NOT here — this guards the
        // ALL-iteration render loop against a future dup/drop regression.)
        let out = render_metrics(&MetricValues::default());
        let value_lines: Vec<&str> = out
            .lines()
            .filter(|l| !l.starts_with('#') && !l.is_empty())
            .collect();
        assert_eq!(
            value_lines.len(),
            metric_names::ALL.len(),
            "one value line per registered metric — no drop, no dup"
        );
        for name in metric_names::ALL {
            let count = value_lines
                .iter()
                .filter(|l| l.split_whitespace().next() == Some(*name))
                .count();
            assert_eq!(count, 1, "metric {name} must be emitted exactly once");
        }
    }

    #[test]
    fn unknown_routes_are_not_found_nothing_mutable_exists() {
        let (head, _) = http_get("/admin/wipe-everything");
        assert!(head.starts_with("HTTP/1.1 404"), "got: {head}");
    }
}
