//! End-to-end verification of the Prometheus exporter wiring.
//!
//! Boots an in-process healthz server on a random localhost port,
//! emits a metric, scrapes `/metrics` over HTTP, and asserts the
//! emitted metric appears in the exposition payload.
//!
//! This is the canonical smoke test that proves:
//!   1. `install_prometheus_recorder` actually registers a working
//!      global recorder (not a no-op stub).
//!   2. `healthz::run_healthz_server` mounts `/metrics` when the
//!      recorder is installed.
//!   3. The scrape returns the right content type and a payload that
//!      Prometheus' line-protocol parser will accept.
//!
//! If any of these silently regress in a refactor, this test fails.

use std::net::SocketAddr;
use std::time::Duration;

use metrics::counter;
use tokio_util::sync::CancellationToken;

use voxeldust_shard_common::healthz;
use voxeldust_shard_common::observability;

/// Spawn the healthz server on `127.0.0.1:0` (kernel-assigned port),
/// emit a uniquely-named counter, scrape `/metrics`, and assert the
/// counter shows up. The full round-trip exercises the same code path
/// production scrapes hit.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn metrics_endpoint_serves_emitted_counter() {
    // First-installer-wins. If a sibling integration test in this
    // binary already installed (no others today), we accept whatever
    // recorder is global and proceed — the assertion below still
    // covers the path. Running tests in isolation
    // (`cargo test -p voxeldust-shard-common --test metrics_integration`)
    // gives a deterministic install.
    observability::install_prometheus_recorder();

    // Bind to an ephemeral port: the OS picks one and we extract it
    // back from the socket. Avoids hard-coded ports clashing under
    // parallel test runs in the same workspace.
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
        .await
        .expect("ephemeral bind must succeed");
    let bound: SocketAddr = listener
        .local_addr()
        .expect("local_addr from bound TcpListener");
    drop(listener); // release so the healthz server can rebind it.

    // Spawn the server. Uses CancellationToken for graceful shutdown
    // when the test completes.
    let cancel = CancellationToken::new();
    let server_cancel = cancel.clone();
    let server = tokio::spawn(async move {
        healthz::run_healthz_server(bound, server_cancel).await;
    });

    // Give the server a moment to bind. 50ms is way more than the
    // axum::serve startup typically needs (microseconds), but tests
    // run on overloaded CI; the cost is bounded.
    tokio::time::sleep(Duration::from_millis(50)).await;

    // Emit a uniquely-named counter so the assertion can't false-pass
    // by colliding with some unrelated metric a sibling test left
    // around in the global recorder.
    let metric_name = "voxeldust_metrics_integration_smoketest_total";
    counter!(metric_name).increment(7);

    // Scrape. We use reqwest here because it's already a workspace
    // dep; a hand-rolled hyper client would be one more thing to
    // maintain and gain nothing.
    let url = format!("http://{bound}/metrics");
    let resp = reqwest::Client::new()
        .get(&url)
        .timeout(Duration::from_secs(2))
        .send()
        .await
        .expect("scrape /metrics must succeed");

    assert_eq!(resp.status(), 200);

    let ctype = resp
        .headers()
        .get("content-type")
        .expect("/metrics must set Content-Type")
        .to_str()
        .unwrap()
        .to_owned();
    assert!(
        ctype.starts_with("text/plain"),
        "Prometheus expects text/plain exposition; got {ctype}"
    );

    let body = resp.text().await.expect("scrape body must decode as UTF-8");
    assert!(
        body.contains(metric_name),
        "/metrics payload missing the counter we emitted ({metric_name}). Body:\n{body}"
    );
    // Prometheus exposition format prints counters as
    // `<name> <value>`; the value side proves the increment landed.
    assert!(
        body.lines()
            .any(|l| l.starts_with(metric_name) && l.ends_with(" 7")),
        "counter value 7 missing from exposition. Body:\n{body}"
    );

    // Tear down. cancel.cancel() unblocks `with_graceful_shutdown`.
    cancel.cancel();
    let _ = server.await;
}

/// `/healthz` is unaffected by the metrics wiring — it always returns
/// 200 even when no recorder is installed. Regression test: if a
/// future refactor accidentally moves the healthz route behind the
/// metrics gate, this fails.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn healthz_route_returns_200_with_or_without_metrics() {
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let bound: SocketAddr = listener.local_addr().unwrap();
    drop(listener);

    let cancel = CancellationToken::new();
    let server_cancel = cancel.clone();
    let server = tokio::spawn(async move {
        healthz::run_healthz_server(bound, server_cancel).await;
    });

    tokio::time::sleep(Duration::from_millis(50)).await;

    let resp = reqwest::Client::new()
        .get(format!("http://{bound}/healthz"))
        .timeout(Duration::from_secs(2))
        .send()
        .await
        .expect("scrape /healthz must succeed");
    assert_eq!(resp.status(), 200);

    cancel.cancel();
    let _ = server.await;
}
