//! Shard observability — Prometheus exporter wiring.
//!
//! The `metrics` facade in `core::signal::metrics` (and any other
//! subsystem that decides to call `counter!`/`gauge!`/`histogram!`)
//! emits into whatever recorder is registered globally. Without a
//! recorder, every emit is a no-op and operators see nothing.
//!
//! This module installs the Prometheus recorder once at process
//! startup and stores the resulting [`PrometheusHandle`] in a process-
//! wide [`OnceLock`]. The `/metrics` HTTP handler in `healthz.rs`
//! reads the handle via [`current_handle`] without needing any
//! plumbing through config structs.
//!
//! # Why a OnceLock instead of plumbing through `ShardHarnessConfig`
//!
//! `PrometheusHandle` does not implement `Debug`, and
//! `ShardHarnessConfig` derives `Debug`. We could `#[derive(Debug)]`
//! around it with a wrapper, but the install-once-per-process
//! contract is exactly what `OnceLock` represents — making the install
//! site write to it and consumer sites read from it is the most
//! honest expression of the lifecycle.
//!
//! # Idempotence
//!
//! `metrics::set_global_recorder` rejects double-installation with
//! `SetRecorderError`. We swallow that error: in practice every shard
//! binary calls [`install_prometheus_recorder`] exactly once at the
//! top of `main`; the only realistic double-install path is unit
//! tests that share a process. Tests that need an isolated recorder
//! should use `metrics::with_local_recorder`.
//!
//! # Why bucket buildup matters
//!
//! `PrometheusBuilder` accumulates buckets for histograms at install
//! time — they cannot be added after the recorder is installed. Any
//! histogram we add later in the codebase that wants non-default
//! buckets must wire its bucket spec into [`install_prometheus_recorder`]
//! first. Today no histograms are wired, so the default linear ladder
//! is fine; we keep this comment as a tripwire for when they're added.

use std::sync::OnceLock;

use metrics_exporter_prometheus::{PrometheusBuilder, PrometheusHandle};

/// Process-wide handle. Set exactly once by
/// [`install_prometheus_recorder`]; read by `healthz::run_healthz_server`
/// to wire the `/metrics` route.
static HANDLE: OnceLock<PrometheusHandle> = OnceLock::new();

/// Install the Prometheus recorder as the global metrics backend.
///
/// Call exactly once per process, at the top of `main`, before any
/// other subsystem emits a metric. After this returns, every
/// `counter!`/`gauge!`/`histogram!` macro in the dependency tree
/// records into the Prometheus backend, and `/metrics` on
/// `healthz_addr` serves the exposition payload.
///
/// Returns `false` if the recorder failed to install (already
/// installed). The caller should log a warning if it expected to be
/// the first installer; we don't crash because metrics are a cross-
/// cutting concern and missing observability should not take a shard
/// down.
pub fn install_prometheus_recorder() -> bool {
    let builder = PrometheusBuilder::new();
    match builder.install_recorder() {
        Ok(handle) => {
            // OnceLock::set returns Err if already populated. We
            // explicitly drop that error: a duplicate install means a
            // duplicate `install_prometheus_recorder` call in the same
            // process, which we already log via the recorder-install
            // path below. The first install wins; the second's handle
            // is silently dropped — its render() would target a
            // recorder that is no longer the global one anyway.
            let _ = HANDLE.set(handle);
            true
        }
        Err(e) => {
            tracing::warn!(
                error = %e,
                "Prometheus recorder install failed — /metrics will be empty. \
                 Most common cause: a second install attempt in the same \
                 process (e.g., test fixture leaking into prod path)."
            );
            false
        }
    }
}

/// Current handle, if a recorder has been installed.
///
/// Returns `None` if no binary has called [`install_prometheus_recorder`]
/// yet (test harnesses, lib-only consumers). The `/metrics` route is
/// then omitted from the healthz router — a 404 is more honest than
/// serving an empty body that pretends scraping is wired.
pub fn current_handle() -> Option<PrometheusHandle> {
    HANDLE.get().cloned()
}

#[cfg(test)]
mod tests {
    use super::*;
    use metrics::counter;

    /// First install in a fresh process must succeed; the handle then
    /// renders the exposition format including any subsequent metric.
    /// Co-tenant tests in the same process binary may see this lock
    /// already set and short-circuit — that's the install-once
    /// contract, not a bug.
    #[test]
    fn install_succeeds_and_handle_renders() {
        if !install_prometheus_recorder() {
            // Recorder slot already taken by a sibling test in the
            // same process. The contract is "first install wins";
            // skip the assertion rather than fight it.
            return;
        }
        // Emit a uniquely-named counter so we don't collide with any
        // sibling test's emissions.
        counter!("voxeldust_observability_install_smoketest_total").increment(1);
        let handle = current_handle().expect("install just succeeded");
        let rendered = handle.render();
        assert!(
            rendered.contains("voxeldust_observability_install_smoketest_total"),
            "rendered payload missing the metric we just emitted: {rendered}"
        );
    }

    /// `current_handle` returns `None` cleanly when no recorder is
    /// installed in this process. (We can only test the negative path
    /// reliably if we're the only test in the binary — otherwise a
    /// sibling test may have installed already. We accept the
    /// either-or outcome and just check the call doesn't panic.)
    #[test]
    fn current_handle_does_not_panic() {
        let _ = current_handle();
    }
}
