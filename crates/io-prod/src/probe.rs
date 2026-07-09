//! S3 (cloud-ready k3d) — the shared k8s probe HTTP surface: ONE `probe_router` (`/healthz` + `/readyz`)
//! served identically by ALL THREE bins (HR3). A SEPARATE router from `admin_router` on purpose: `/admin/*`
//! serves topology (directory + leases + sagas) and must gain auth before any routable bind, while probes are
//! UNAUTHENTICATED by necessity (a kubelet `httpGet` presents no token) — coupling them would open the
//! topology surface to the unauth kubelet posture. The probe response is a BARE status code (no body), so no
//! path exists for cluster state to leak to an unauth caller; a partitioned-vs-full-vs-booting distinction
//! lives behind the auth-gated `/metrics`, never here.
//!
//! HR1/seam: this is thin axum GLUE (Tier-B) over the PURE decision surface in `vd_node::health` (Tier-A,
//! 100% region+branch). Each bin injects its own [`HealthSource`] whose `report()` reads a lock-free
//! `ArcSwap` the bin's tick loop publishes — the HTTP task never touches the sim `World`.

use std::sync::Arc;
use std::time::{Duration, Instant};

use arc_swap::ArcSwap;
use axum::extract::State;
use axum::http::StatusCode;
use axum::routing::get;
use vd_node::health::{HealthReport, health_report};

/// A tick-progress heartbeat the bin's tick loop stamps each iteration (`at` via `Instant::now()` on the OPS
/// plane — `Instant` is banned in the Tier-A `vd_node`, so it lives here). A FROZEN `at` (a deadlocked/wedged
/// `step_tick` that never reaches the stamp) is what the staleness liveness check detects; `tick` is the
/// partition-surviving local tick, so the detector still works while a node is partitioned but healthy.
#[derive(Clone, Copy, Debug)]
pub struct Heartbeat {
    pub tick: u64,
    pub at: Instant,
}

/// The node's current health, read lock-free by the probe handlers. Each bin injects an impl backed by an
/// `ArcSwap` its tick loop (+ the shutdown-edge publisher) stores into — the per-role polymorphism is the
/// trait object, NEVER a `match` on a node kind (HR3), and each node reports only its OWN state (HR1).
pub trait HealthSource: Send + Sync + 'static {
    fn report(&self) -> HealthReport;
}

/// A fixed report — the test double (the probe analog of `admin::FixedSnapshot`).
pub struct FixedHealth(pub HealthReport);

impl HealthSource for FixedHealth {
    fn report(&self) -> HealthReport {
        self.0
    }
}

/// The published node-health snapshot the tick loop writes and the probe handler reads — lock-free via
/// [`HealthCell`] (the proven admin `ArcSwap` pattern). Holds the tick-progress [`Heartbeat`], the role's
/// last-computed readiness, and the drain flag.
#[derive(Clone, Copy, Debug)]
pub struct HealthState {
    pub heartbeat: Heartbeat,
    pub ready: bool,
    pub draining: bool,
}

impl HealthState {
    /// The pre-boot seed: NotReady, not draining, heartbeat stamped now (so an early scrape before the first
    /// tick reads LIVE-but-NotReady, never a false not-live). `Instant::now()` here is ops-plane.
    #[must_use]
    pub fn booting() -> HealthState {
        HealthState {
            heartbeat: Heartbeat {
                tick: 0,
                at: Instant::now(),
            },
            ready: false,
            draining: false,
        }
    }
}

/// The shared lock-free health cell: the tick loop `store`s each iteration, the shutdown-edge publisher
/// `store`s once, the probe HTTP task `load`s. ONE type for all three bins (HR3).
pub type HealthCell = Arc<ArcSwap<HealthState>>;

/// A fresh cell seeded [`HealthState::booting`].
pub fn new_health_cell() -> HealthCell {
    Arc::new(ArcSwap::from_pointee(HealthState::booting()))
}

/// The tick loop's per-iteration publish: stamp progress + the role's computed readiness. PRESERVES an
/// already-set `draining` (and forces NotReady while draining) so the tick loop can NEVER un-drain a pod after
/// a SIGTERM edge fired. Uses `rcu` (read-copy-update: re-read + retry the swap if the cell changed under us),
/// NOT a load-then-`store` — the two publishers run on DIFFERENT threads (the tick loop paces on the main
/// thread via a blocking `TickPacer`; `publish_draining` fires from the signal-handler task on a tokio worker),
/// so a plain load-then-store loses the update on the interleaving `read draining=false → drain stores true →
/// store draining=false`, leaving a terminating pod Ready as its FINAL state (the loop then breaks on the
/// shutdown flag and never re-publishes) — kubelet re-adds it to the Service and routes live traffic to a
/// process about to exit. `rcu`'s compare-and-retry makes `draining` MONOTONE once the edge sets it.
pub fn publish_tick(cell: &HealthCell, tick: u64, ready: bool) {
    cell.rcu(|cur| {
        let draining = cur.draining;
        HealthState {
            heartbeat: Heartbeat {
                tick,
                at: Instant::now(),
            },
            ready: ready & !draining,
            draining,
        }
    });
}

/// The SIGTERM-EDGE publish (from the signal-handler task, independent of the tick loop): de-route (NotReady)
/// the INSTANT SIGTERM arrives — the loop-head `!shutdown` check would otherwise skip the in-body publish on
/// the exit tick, keeping the pod Ready through termination — and set `draining` so `/healthz` stays LIVE
/// through the final-fsync park (never a kubelet SIGKILL mid-write). Keeps the last heartbeat (moot: draining
/// forces live).
pub fn publish_draining(cell: &HealthCell) {
    let cur = **cell.load();
    cell.store(Arc::new(HealthState {
        heartbeat: cur.heartbeat,
        ready: false,
        draining: true,
    }));
}

/// The [`HealthSource`] every bin injects: reads the lock-free [`HealthCell`] + applies the pure
/// `vd_node::health::health_report` (the only `Instant::now()` is the staleness reference, ops-plane).
pub struct PublishedHealth {
    cell: HealthCell,
    stall_deadline: Duration,
}

impl PublishedHealth {
    #[must_use]
    pub fn new(cell: HealthCell, stall_deadline: Duration) -> PublishedHealth {
        PublishedHealth {
            cell,
            stall_deadline,
        }
    }
}

impl HealthSource for PublishedHealth {
    fn report(&self) -> HealthReport {
        let s = **self.cell.load();
        // The ONE `Instant::now()` — the staleness reference (ops-plane); the pure `health_report` decides.
        let staleness = Instant::now().saturating_duration_since(s.heartbeat.at);
        health_report(staleness, self.stall_deadline, s.ready, s.draining)
    }
}

#[derive(Clone)]
struct ProbeState {
    source: Arc<dyn HealthSource>,
}

/// Build the ONE probe router: `/healthz` (liveness) + `/readyz` (readiness), each a bare status code.
pub fn probe_router(source: Arc<dyn HealthSource>) -> axum::Router {
    axum::Router::new()
        .route("/healthz", get(healthz))
        .route("/readyz", get(readyz))
        .with_state(ProbeState { source })
}

/// k8s LIVENESS: `200` while live, `503` ONLY on a detected running wedge (a deliberate drain reads live).
/// A `503` here → kubelet RESTARTS the pod. It does NOT `503` for not-ready (a booting/syncing node is live).
async fn healthz(State(state): State<ProbeState>) -> StatusCode {
    if state.source.report().live {
        StatusCode::OK
    } else {
        StatusCode::SERVICE_UNAVAILABLE
    }
}

/// k8s READINESS: `200` when ready, `503` when not (booting, syncing, partitioned/self-fenced, or draining).
/// A `503` here → kubelet removes the pod from the Service endpoints but LEAVES it running.
async fn readyz(State(state): State<ProbeState>) -> StatusCode {
    if state.source.report().ready {
        StatusCode::OK
    } else {
        StatusCode::SERVICE_UNAVAILABLE
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tokio::io::{AsyncReadExt, AsyncWriteExt};

    /// One blocking HTTP/1.1 GET against the served probe router (raw bytes = what a kubelet `httpGet` sends);
    /// returns the numeric status. Mirrors `admin::tests::http_get`.
    fn probe_status(report: HealthReport, path: &str) -> u16 {
        let rt = tokio::runtime::Builder::new_multi_thread()
            .worker_threads(2)
            .enable_all()
            .build()
            .expect("tokio runtime");
        rt.block_on(async move {
            let source: Arc<dyn HealthSource> = Arc::new(FixedHealth(report));
            let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
                .await
                .expect("bind");
            let addr = listener.local_addr().expect("addr");
            tokio::spawn(async move {
                axum::serve(listener, probe_router(source))
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
            let status_line = text.lines().next().expect("status line");
            // "HTTP/1.1 200 OK" → 200
            status_line
                .split_whitespace()
                .nth(1)
                .expect("status code")
                .parse::<u16>()
                .expect("numeric status")
        })
    }

    #[test]
    fn healthz_maps_live_to_200_and_not_live_to_503() {
        assert_eq!(
            probe_status(
                HealthReport {
                    live: true,
                    ready: false
                },
                "/healthz"
            ),
            200
        );
        assert_eq!(
            probe_status(
                HealthReport {
                    live: false,
                    ready: true
                },
                "/healthz"
            ),
            503
        );
    }

    #[test]
    fn readyz_maps_ready_to_200_and_not_ready_to_503() {
        assert_eq!(
            probe_status(
                HealthReport {
                    live: true,
                    ready: true
                },
                "/readyz"
            ),
            200
        );
        assert_eq!(
            probe_status(
                HealthReport {
                    live: true,
                    ready: false
                },
                "/readyz"
            ),
            503
        );
    }

    #[test]
    fn unknown_route_is_404() {
        assert_eq!(
            probe_status(
                HealthReport {
                    live: true,
                    ready: true
                },
                "/nope"
            ),
            404
        );
    }

    #[test]
    fn publish_tick_preserves_draining_so_the_loop_cannot_un_drain() {
        let cell = new_health_cell();
        // A normal tick: ready as computed, not draining.
        publish_tick(&cell, 1, true);
        let s = **cell.load();
        assert!(s.ready);
        assert!(!s.draining);
        // SIGTERM edge → de-route + draining.
        publish_draining(&cell);
        let s = **cell.load();
        assert!(!s.ready);
        assert!(s.draining);
        // A LATER tick (the one-tick race after the edge) must NOT un-drain or re-route.
        publish_tick(&cell, 2, true);
        let s = **cell.load();
        assert!(
            !s.ready,
            "draining forces NotReady even if the role predicate is ready"
        );
        assert!(
            s.draining,
            "draining is monotone once set on the shutdown edge"
        );
        assert_eq!(
            s.heartbeat.tick, 2,
            "the heartbeat still advances (draining forces live anyway)"
        );
    }

    #[test]
    fn publish_tick_cannot_clobber_a_concurrent_drain() {
        use std::thread;
        // Race publish_tick (the tick loop) against publish_draining (the signal-handler task) on TWO
        // threads, many rounds. Once the drain edge fires, the tick loop must NEVER un-drain: `rcu` re-reads
        // and retries, so `draining` is monotone. The old load-then-`store` lost this update and would, on
        // the unlucky interleaving, publish {draining:false, ready:true} as the pod's FINAL state.
        for round in 0..4000u64 {
            let cell = new_health_cell();
            let ticker = Arc::clone(&cell);
            let t = thread::spawn(move || {
                // ready=true is the worst case: a lost-update would republish Ready.
                publish_tick(&ticker, round, true);
            });
            publish_draining(&cell);
            t.join().expect("tick thread");
            let s = **cell.load();
            assert!(
                s.draining,
                "draining must survive a concurrent publish_tick (round {round})"
            );
            assert!(
                !s.ready,
                "a drained cell is never ready, even under the race"
            );
        }
    }

    #[test]
    fn published_health_reports_booting_live_not_ready_then_ready() {
        let cell = new_health_cell();
        let src = PublishedHealth::new(Arc::clone(&cell), Duration::from_secs(60));
        // Booting: fresh heartbeat ⇒ LIVE; not yet ready.
        let r = src.report();
        assert!(r.live);
        assert!(!r.ready);
        // After a ready tick: live + ready.
        publish_tick(&cell, 1, true);
        let r = src.report();
        assert!(r.live);
        assert!(r.ready);
        // Draining: still LIVE (drain-safe) but de-routed.
        publish_draining(&cell);
        let r = src.report();
        assert!(r.live);
        assert!(!r.ready);
    }
}
