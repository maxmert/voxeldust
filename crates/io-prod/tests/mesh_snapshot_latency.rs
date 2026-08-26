//! SPIKE-3a: the 2nd hard latency GATE (P3 blocker) — the 20 Hz unreliable SNAPSHOT datagram hot path must
//! stay TIMELY while a bulk RELIABLE transfer saturates the SAME quinn connection. This is the exact
//! end-goal coupling ("hundreds in one location", signal-heavy): a ship-blueprint / terrain-chunk bulk
//! transfer must never starve the snapshot stream a client renders from. ONE sender -> ONE receiver = ONE
//! `quinn::Connection`, so both classes genuinely share the connection's send scheduler + congestion window
//! (a second sender node would be a second connection and prove nothing).
//!
//! What it measures: the END-TO-END delivered-snapshot latency (enqueue -> receiver drain), on ONE monotonic
//! clock (same process, loopback — no cross-host skew), while a sustained reliable `MsgClass::Saga` burst runs
//! on the same per-peer lane. The transport routes ALL classes to a peer through ONE FIFO `peer_writer`, so
//! this latency reflects the REAL, user-observed cost of bulk contention (both app-lane and wire), not an
//! idealized stream/datagram-independence microbench.
//!
//! Determinism / non-flakiness (mirrors the SPIKE-2a release-gate discipline): gate on p99 (a robust tail,
//! not max/mean); discard warmup snapshots (handshake / CC slow-start / cold code); a delivery-RATIO floor
//! (unreliable latest-wins loss under a bulk burst is CORRECT-by-design, never gated to zero) catches total
//! starvation that a p99-over-survivors would hide; `datagrams_dropped_too_large == 0` proves we measured
//! congestion latency, not an MTU cliff; the hard timing assert is RELEASE-ONLY (a debug/coverage-instrumented
//! tail is meaningless). Run via `just spike3a` (release, `--test-threads=1`); under debug `cargo test` the
//! send/drain/decode still run (coverage) with the timing assert compiled out.

use std::collections::BTreeMap;
use std::net::{SocketAddr, UdpSocket};
use std::time::{Duration, Instant};

use vd_core::NodeId;
use vd_io_prod::mesh::{MeshConfig, MeshControl, MeshTransport, spawn_mesh};
use vd_io_prod::trust::ClusterTrust;
use vd_sim::io::{Inbound, MsgClass, Transport};

const RECEIVER: NodeId = NodeId(1);
const SENDER: NodeId = NodeId(2);

/// The ONE config home for the SPIKE-3a load shape (HR5 no-magic-numbers): every knob justified, never an
/// inline literal at a use site. Budgets/deadlines are DERIVED from the shape.
struct LatencyShape {
    /// Snapshot datagrams to fire, at `snapshot_hz` (the measured window; the first `warmup` are discarded).
    snapshots_to_send: usize,
    /// Warmup snapshots discarded before measuring (QUIC handshake, CC slow-start, cold allocs/MTU jitter —
    /// the single biggest latency-gate flake source).
    warmup: usize,
    /// The realistic client snapshot rate (Hz) — sets the 50 ms inter-snapshot tick.
    snapshot_hz: u64,
    /// The real D-18 gateway->client snapshot datagram size (proven under loopback `max_datagram_size` by the
    /// `ca1_reply_on_connection_delivers_an_unreliable_snapshot_datagram` test), so the ONLY drops are
    /// congestion, never the MTU cliff.
    snapshot_payload_bytes: usize,
    /// Bulk reliable frame size — the burst is `MsgClass::Saga` (reliable → ordered uni-stream + ack lane),
    /// pushed to saturate the shared connection for the whole window.
    bulk_frame_bytes: usize,
    /// Leave the lane a drain margin at the END of each tick so the peer_writer clears a slot for the next
    /// snapshot (the single per-peer FIFO lane is shared; without a margin the snapshot is perpetually shed —
    /// which would measure lane starvation, a SEPARATE property from delivered-latency).
    tick_drain_margin: Duration,
    /// Delivery-ratio floor as an exact integer fraction (num/den): delivered_measured * den >= measured * num.
    min_delivered_num: usize,
    min_delivered_den: usize,
    /// The HARD p99 budget (release-only). Calibrated at bring-up to ~10x the OBSERVED loaded p99 AND under one
    /// snapshot tick — a fresh snapshot is never a full tick late. See the headroom note at the assert.
    snapshot_p99_budget: Duration,
    /// A BOUNDED tokio pool: the point is realistic core contention on the ONE connection's CC + the single
    /// RecvLedger mutex, not to dissolve it under one-thread-per-lane.
    worker_threads: usize,
    /// Grace after the last snapshot to drain the in-flight tail before accounting.
    tail_grace: Duration,
}

impl LatencyShape {
    fn from_env() -> LatencyShape {
        // Debug (`just test`) runs a SMALL smoke window: it exercises the send/drain/decode + the always-on
        // ratio/honesty/competitor asserts (the timing p99 gate is compiled OUT under debug), so a big window
        // buys nothing there and would only add ~10s to the inner loop AND expose the always-on asserts to
        // `just test`'s DEFAULT parallel-CPU contention. Release (`just spike3a`, `--test-threads=1`) runs the
        // full, isolated measurement window that the p99 gate needs.
        let (default_snaps, warmup) = if cfg!(debug_assertions) {
            (60usize, 10usize) // ~3s at 20 Hz; 50 measured is ample to run every path once
        } else {
            (200usize, 40usize) // ~10s; 160 measured → a stable p99 nearest-rank
        };
        // ONE calibration knob (mirrors mesh_load's single VD_MESH_LOAD_NODES) to scale the window for a soak;
        // floored strictly ABOVE `warmup` so `measured()` (= snaps − warmup) is ALWAYS ≥ 1 — a hostile override
        // can never collapse the measured set to zero (which would make the ratio floor + p99 vacuous).
        let snapshots_to_send = std::env::var("VD_SPIKE3A_SNAPS")
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
            .filter(|n| *n > warmup)
            .unwrap_or(default_snaps);
        LatencyShape {
            snapshots_to_send,
            warmup,
            snapshot_hz: 20,
            snapshot_payload_bytes: 1150,
            bulk_frame_bytes: 1024,
            tick_drain_margin: Duration::from_millis(6),
            min_delivered_num: 1,
            min_delivered_den: 2,
            // Calibrated (bring-up STEP 13): the observed release delivered-snapshot p99 is ~0.7 ms (5 runs on
            // a LOADED dev host: 678-728 us, a ~7% spread) while ~1.8 GB of reliable bulk saturated the same
            // connection. 15 ms is ~20x that observed p99 — generous headroom to absorb an oversubscribed /
            // throttled CI core (scheduler starvation of the drain/test thread can add several ms) — yet it
            // stays UNDER one 50 ms snapshot tick, so a snapshot delayed a full tick still fails. A REAL
            // regression is orders of magnitude larger: a Snapshot mis-routed down the reliable ORDERED lane
            // head-of-line-blocks behind ~1.8 GB (seconds); a per-write blocking call serializing datagrams
            // behind reliable writes, or removed datagram pacing starving the CC, likewise blow past 15 ms.
            // Per the SPIKE-2a discipline: widen toward a full tick if CI ever flakes — NEVER shrink.
            snapshot_p99_budget: Duration::from_millis(15),
            worker_threads: 4,
            tail_grace: Duration::from_millis(500),
        }
    }

    fn tick(&self) -> Duration {
        Duration::from_millis(1000 / self.snapshot_hz)
    }

    /// Measured (non-warmup) snapshots the p99/ratio are computed over.
    fn measured(&self) -> usize {
        self.snapshots_to_send - self.warmup
    }

    /// Generous overall wall-clock ceiling (setup + the full 20 Hz window + tail). A trip means "stopped
    /// arriving" = a real starvation failure, never a flake.
    fn deadline(&self) -> Duration {
        Duration::from_secs(20) + self.tick() * self.snapshots_to_send as u32
    }
}

fn runtime(worker_threads: usize) -> tokio::runtime::Runtime {
    tokio::runtime::Builder::new_multi_thread()
        .worker_threads(worker_threads)
        .enable_all()
        .build()
        .expect("tokio runtime")
}

/// Reserve an ephemeral loopback UDP port (bind `:0`, read back, drop) for a node to re-bind.
fn reserve() -> SocketAddr {
    UdpSocket::bind("127.0.0.1:0")
        .expect("reserve port")
        .local_addr()
        .expect("addr")
}

fn spawn_node(
    handle: &tokio::runtime::Handle,
    trust: &ClusterTrust,
    id: NodeId,
    addr: SocketAddr,
    book: &BTreeMap<NodeId, SocketAddr>,
    inbound_capacity: usize,
) -> (MeshTransport, MeshControl) {
    let mut cfg = MeshConfig::new(id, addr, book.clone(), 256, 1, 0);
    cfg.inbound_capacity = inbound_capacity;
    spawn_mesh(handle, trust, &cfg, None).expect("mesh node")
}

/// A snapshot datagram payload: `seq` LE in `[0..8]`, zero-padded to `bytes`. Unreliable (latest-wins) — a
/// full lane / congestion drop is BY DESIGN, so this reports Ok(admitted)/Err(shed), never retries.
fn send_snapshot(t: &mut MeshTransport, seq: u64, bytes: usize) -> bool {
    let mut payload = vec![0u8; bytes];
    payload[0..8].copy_from_slice(&seq.to_le_bytes());
    t.send(RECEIVER, MsgClass::Snapshot, payload.into()).is_ok()
}

/// Best-effort bulk push: one reliable `Saga` frame (filler keyed by `seq` so frames are distinct). Returns
/// whether it was admitted to the lane (QueueFull = the lane is momentarily saturated — the caller backs off).
fn push_bulk(t: &mut MeshTransport, seq: u64, bytes: usize) -> bool {
    let mut payload = vec![0u8; bytes];
    payload[0..8].copy_from_slice(&seq.to_le_bytes());
    t.send(RECEIVER, MsgClass::Saga, payload.into()).is_ok()
}

/// Drain the receiver, stamping a recv `Instant` for each delivered SNAPSHOT (keyed by its seq); ignore the
/// bulk `Saga` frames (filler). Panic on any other inbound (a clean loopback pair has none).
fn drain_snapshots(t: &mut MeshTransport, recv: &mut BTreeMap<u64, Instant>) {
    for ev in t.drain_inbound() {
        match ev {
            Inbound::Wire {
                class: MsgClass::Snapshot,
                bytes,
                ..
            } => {
                let mut b = [0u8; 8];
                b.copy_from_slice(&bytes[..8]);
                // The recv stamp is taken here, at DRAIN time on the polling thread (~100-200 us cadence), so a
                // measured latency includes up to one drain interval of polling jitter. That bias is CONSERVATIVE
                // (it inflates the number — makes the gate HARDER to pass, never hides a regression), and the
                // 15 ms budget absorbs it comfortably. Latest-wins: keep the FIRST arrival's stamp per seq.
                recv.entry(u64::from_le_bytes(b))
                    .or_insert_with(Instant::now);
            }
            Inbound::Wire {
                class: MsgClass::Saga,
                ..
            } => {} // the bulk competitor — drained (keeps the inbox clear), not measured
            other => panic!("unexpected inbound on a clean loopback pair: {other:?}"),
        }
    }
}

#[test]
fn snapshot_datagram_p99_stays_timely_under_a_bulk_reliable_burst() {
    let shape = LatencyShape::from_env();
    let deadline = shape.deadline();
    let overall_start = Instant::now();
    let rt = runtime(shape.worker_threads);
    let trust = ClusterTrust::generate("vd-spike3a").expect("trust");

    // Receiver binds FIRST, inbox generously sized (the bulk floods it; a big inbox + per-iteration drain keeps
    // reliable inbox drops from confounding the measurement — reported if nonzero, not gated).
    let recv_addr = reserve();
    let empty_book: BTreeMap<NodeId, SocketAddr> = BTreeMap::new();
    let (mut receiver, _recv_ctl) =
        spawn_node(rt.handle(), &trust, RECEIVER, recv_addr, &empty_book, 65536);

    // ONE sender, booking ONLY the receiver → ONE connection carries BOTH classes.
    let sender_book: BTreeMap<NodeId, SocketAddr> = [(RECEIVER, recv_addr)].into();
    let (mut sender, sender_ctl) =
        spawn_node(rt.handle(), &trust, SENDER, reserve(), &sender_book, 1024);

    let tick = shape.tick();
    let mut sent: BTreeMap<u64, Instant> = BTreeMap::new();
    let mut recv: BTreeMap<u64, Instant> = BTreeMap::new();
    let mut snapshot_shed: usize = 0;
    let mut bulk_pushed: u64 = 0;

    let mut next_tick = Instant::now() + tick;
    for seq in 0..shape.snapshots_to_send as u64 {
        // Fire ONE snapshot at the tick boundary (the lane has drained over the prior margin). Latest-wins: a
        // QueueFull is a shed, counted — never retried (that is how the production OutboundBox behaves).
        let t0 = Instant::now();
        if send_snapshot(&mut sender, seq, shape.snapshot_payload_bytes) {
            sent.insert(seq, t0);
        } else {
            snapshot_shed += 1;
        }

        // Saturate the connection with bulk for MOST of the tick, then stop early (tick_drain_margin) and drain
        // only — so the peer_writer clears a lane slot for the NEXT snapshot. Keeps the wire busy the whole
        // window (reliable_acked climbs) without perpetually starving the snapshot of a lane slot.
        let bulk_until = next_tick.saturating_duration_since(Instant::now());
        let bulk_deadline = Instant::now() + bulk_until.saturating_sub(shape.tick_drain_margin);
        while Instant::now() < bulk_deadline {
            if push_bulk(&mut sender, bulk_pushed, shape.bulk_frame_bytes) {
                bulk_pushed += 1;
            } else {
                std::thread::sleep(Duration::from_micros(100)); // lane full — let the writer drain
            }
            drain_snapshots(&mut receiver, &mut recv);
        }
        // Drain-only margin: no bulk pushed, peer_writer clears the lane for the next snapshot.
        while Instant::now() < next_tick {
            drain_snapshots(&mut receiver, &mut recv);
            std::thread::sleep(Duration::from_micros(200));
        }
        next_tick += tick;
        assert!(
            overall_start.elapsed() < deadline,
            "SPIKE-3a exceeded the generous wall-clock ceiling during the send window (starvation?)"
        );
    }

    // Drain the in-flight tail.
    let tail_end = Instant::now() + shape.tail_grace;
    while Instant::now() < tail_end {
        drain_snapshots(&mut receiver, &mut recv);
        std::thread::sleep(Duration::from_millis(1));
    }

    // Account: delivered latencies for MEASURED (non-warmup) snapshots, matched by seq on the one monotonic clock.
    let warmup = shape.warmup as u64;
    let mut deltas: Vec<Duration> = Vec::new();
    let mut measured_sent = 0usize;
    for (&seq, &t0) in &sent {
        if seq < warmup {
            continue;
        }
        measured_sent += 1;
        if let Some(&t1) = recv.get(&seq) {
            deltas.push(t1.saturating_duration_since(t0));
        }
    }
    let delivered = deltas.len();
    let p50 = vd_harness::latency::percentile_unstable(deltas.clone(), 50);
    let p99 = vd_harness::latency::percentile_unstable(deltas.clone(), 99);
    let stats = sender_ctl.stats();

    // REPORTED (never gated to zero — latest-wins loss under a bulk burst is correct-by-design).
    eprintln!(
        "[spike3a] snapshots sent={} (measured={}, warmup={}, lane-shed={}), delivered(measured)={} ratio={}/{}\n\
         [spike3a] latency p50={p50:?} p99={p99:?} budget={:?}\n\
         [spike3a] bulk pushed={bulk_pushed} reliable_acked={} reliable_shed={} datagrams_dropped_send={} \
         datagrams_dropped_too_large={} inbound_dropped_unreliable={}",
        sent.len() + snapshot_shed,
        measured_sent,
        shape.warmup,
        snapshot_shed,
        delivered,
        shape.min_delivered_num,
        shape.min_delivered_den,
        shape.snapshot_p99_budget,
        stats.reliable_acked,
        stats.reliable_shed,
        stats.datagrams_dropped_send,
        stats.datagrams_dropped_too_large,
        stats.inbound_dropped_unreliable,
    );

    // HONESTY (always): the 1150 B payload never tripped the MTU cliff, so measured drops are congestion.
    assert_eq!(
        stats.datagrams_dropped_too_large, 0,
        "a snapshot datagram exceeded the datagram budget — measuring framing loss, not latency"
    );
    // COMPETITOR-REAL (always): the bulk actually ran (climbed the reliable ack path), so the connection was
    // genuinely under load — otherwise the latency gate proves nothing.
    assert!(
        stats.reliable_acked >= (bulk_pushed / 2).max(1),
        "the bulk burst did not provably compete (reliable_acked {} vs pushed {bulk_pushed})",
        stats.reliable_acked
    );
    // DELIVERY-RATIO FLOOR (always, integer math): catches total snapshot starvation that a p99-over-survivors
    // would hide (percentile_unstable is TOTAL: empty -> ZERO < budget passes deceptively without this floor).
    assert!(
        delivered * shape.min_delivered_den >= shape.measured() * shape.min_delivered_num,
        "snapshot delivery collapsed under bulk: {delivered}/{} measured (floor {}/{})",
        shape.measured(),
        shape.min_delivered_num,
        shape.min_delivered_den,
    );

    // The HARD p99 latency GATE is RELEASE-ONLY (a debug/coverage-instrumented tail is meaningless — the same
    // discipline as the SPIKE-2a gate). Headroom: the budget is ~10x the OBSERVED loopback p99 AND under one
    // snapshot tick, so a fresh snapshot is never a full tick late; a REAL regression (Snapshot mis-routed down
    // the reliable ordered lane, a per-write blocking call serializing datagrams behind bulk, removed datagram
    // pacing starving the CC) is orders of magnitude larger and trips this by a wide margin.
    #[cfg(not(debug_assertions))]
    {
        assert!(
            p99 < shape.snapshot_p99_budget,
            "snapshot p99 {p99:?} under a bulk burst exceeded {:?} — the snapshot hot path was starved by bulk",
            shape.snapshot_p99_budget
        );
    }
    #[cfg(debug_assertions)]
    let _ = (p50, p99);

    // Keep both handles alive to the end so no lane tears down mid-account.
    let _ = (&mut sender, &mut receiver);
}
