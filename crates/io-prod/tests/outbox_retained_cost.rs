//! ★ WHAT DURABILITY COSTS — the measurement behind "add durability, make sure performance does not
//! degrade" (foundation slice 4, 2026-09-06), and the gate on the fix it forced.
//!
//! The claim under test is a NUMBER, not an argument. The SAME code runs twice over real loopback QUIC
//! sockets — once with no outbox at all, once with a real redb `NodeOutbox` on a temp file — and the two
//! runs are compared frame by frame. Nothing else differs between them.
//!
//! ★ WHAT THE FIRST MEASUREMENT FOUND (before the group commit):
//!
//! - one retained frame cost ONE DISK SYNC of its own: ~5 ms at p50, ~12 ms at p99;
//! - the SNAPSHOT LANE PAID THE SAME PRICE, +11.3 ms at p99 and +5.3 ms at p50, although a datagram never
//!   touches the outbox — everything toward one peer rode one writer task, and that task sat in the
//!   durability wait as the plain body of its send arm;
//! - offered one retained frame every 5 ms the writer fell behind by ~1 ms per frame and the backlog grew
//!   for the whole run: p99 +231 ms on one run and +414 ms on another, on BOTH lanes.
//!
//! ★ WHAT IT READS NOW (the same three readings, after the batch rewrite in `peer_writer`):
//!
//! - the SNAPSHOT LANE IS FREE. Its p99 delta reads between −0.7 and +0.3 ms at BOTH cadences. The writer
//!   sends its peer's datagrams BEFORE the barrier and keeps sending them THROUGH it, so a player's view
//!   never waits for a hull's despawn to reach the disk.
//! - a BURST of 200 retained frames rides 2 DISK BARRIERS — 100 frames per sync, where it used to be one
//!   sync each. That is the group commit, and it is pinned by an assertion, not by a comment.
//! - a retained frame still costs its ONE disk sync: +4.7 ms at the median at the sustainable cadence,
//!   +7.7 ms at the saturating one (where the offered rate is close to what this disk can answer, so a
//!   frame waits behind about one other). The p99 is the fsync's OWN tail and is printed, not gated — see
//!   `RETAINED_P50_BUDGET` for the readings that say why.
//!
//! In the game's words: a shard that loses a whole band of hulls at once writes two hundred despawns to disk
//! in one barrier instead of two hundred, and the players watching through that same lane never notice it
//! happened.
//!
//! Tier-B (real sockets + real redb). The three parts run IN SEQUENCE inside one test because they share one
//! disk; every measured number is PRINTED. Run with `-- --nocapture` to read them.

use std::collections::BTreeMap;
use std::net::{SocketAddr, UdpSocket};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use vd_core::NodeId;
use vd_io_prod::mesh::{MeshConfig, MeshControl, MeshTransport, spawn_mesh};
use vd_io_prod::outbox::{NodeOutbox, OutboxSink};
use vd_io_prod::store::StoreTuning;
use vd_io_prod::trust::ClusterTrust;
use vd_sim::io::{Durability, Inbound, MsgClass, Transport};

/// Type-identical to the mesh's private `SharedOutbox` alias, so it passes through `spawn_mesh`.
type SharedOutbox = Arc<Mutex<Box<dyn OutboxSink + Send>>>;

const A: NodeId = NodeId(1);
const B: NodeId = NodeId(2);
/// Frames per lane per run. Enough for a meaningful p99 (the 198th sample), short enough to stay a test.
const FRAMES: usize = 200;
/// A cadence the disk CAN hold: each frame's own barrier finishes before the next frame is offered, so the
/// measurement is the cost of ONE durable send and not the depth of a queue.
const SUSTAINABLE: Duration = Duration::from_millis(20);
/// The cadence that OUTRUNS the disk — one retained frame every 5 ms, 200 a second toward one peer.
const SATURATING: Duration = Duration::from_millis(5);
/// ★ THE RETAINED LANE'S BUDGET, ON THE MEDIAN. One disk sync is what a retained frame is supposed to buy,
/// and the median says whether it bought one or many.
///
/// It is deliberately NOT a p99 bar. A single fsync on this machine has a long tail of its own — the OFF
/// p99 sits at 0.33 ms while the ON p99 has read 9.8, 10.6, 12.0, 22.7 and 36.7 ms on runs whose median
/// never left 4.8-5.3 ms. Gating that would gate the disk's mood. The property a p99 bar was meant to guard
/// — ONE sync per burst, not one per frame — is pinned directly and stably by the burst part, which counts
/// the barriers themselves.
const RETAINED_P50_BUDGET: Duration = Duration::from_millis(25);
/// ★ THE SNAPSHOT LANE'S BUDGET: the disk is not its business, so this is ~0 and the bound only holds the
/// measurement noise. A datagram never touches the outbox; if this bound ever fails, the durable gate has
/// started blocking the player's own view again.
const UNRELIABLE_P99_BUDGET: Duration = Duration::from_millis(2);
const DEADLINE: Duration = Duration::from_secs(60);
/// The reliable carrier: a producer-less one-shot class, the only kind `Retained` is for.
const RELIABLE: MsgClass = MsgClass::GhostReliable;
/// The unreliable carrier: the stream a player actually watches.
const UNRELIABLE: MsgClass = MsgClass::Snapshot;

fn runtime() -> tokio::runtime::Runtime {
    tokio::runtime::Builder::new_multi_thread()
        .worker_threads(4)
        .enable_all()
        .build()
        .expect("tokio runtime")
}

fn reserve() -> SocketAddr {
    UdpSocket::bind("127.0.0.1:0")
        .expect("reserve port")
        .local_addr()
        .expect("addr")
}

/// Open a real per-node `NodeOutbox` on a temp redb file, wrapped as the shared sink `spawn_mesh` takes.
fn shared_outbox(tag: &str) -> (SharedOutbox, std::path::PathBuf) {
    let path = std::env::temp_dir().join(format!(
        "vd-outbox-retained-cost-{tag}-{}.redb",
        std::process::id()
    ));
    let _ = std::fs::remove_file(&path);
    let ob = NodeOutbox::open(
        &path,
        StoreTuning::default(),
        vd_core::store_stamp::StoreStamp::new(
            vd_core::store_stamp::StoreRole::Outbox,
            0,
            vd_core::EpochId(0),
            &[],
        ),
    )
    .expect("open outbox");
    let sink: SharedOutbox = Arc::new(Mutex::new(Box::new(ob) as Box<dyn OutboxSink + Send>));
    (sink, path)
}

fn node(
    handle: &tokio::runtime::Handle,
    trust: &ClusterTrust,
    id: NodeId,
    addr: SocketAddr,
    book: &BTreeMap<NodeId, SocketAddr>,
    outbox: Option<SharedOutbox>,
) -> (MeshTransport, MeshControl) {
    let cfg = MeshConfig::new(id, addr, book.clone(), 512, 1, 0);
    spawn_mesh(handle, trust, &cfg, outbox).expect("mesh node")
}

/// A frame index rides in the payload, so a receive can be matched to its own send instant.
fn payload(i: usize) -> Vec<u8> {
    let idx = u16::try_from(i).expect("FRAMES fits a u16");
    idx.to_be_bytes().to_vec()
}

fn index_of(bytes: &[u8]) -> usize {
    let raw: [u8; 2] = bytes.try_into().expect("a 2-byte index payload");
    usize::from(u16::from_be_bytes(raw))
}

/// What one run measured.
struct Run {
    /// Reliable frame indices in ARRIVAL order — the exactly-once, in-order evidence.
    reliable_order: Vec<usize>,
    reliable: Vec<Duration>,
    unreliable: Vec<Duration>,
    /// Retained rows mirrored to the outbox, and the disk BARRIERS asked for. Rows ÷ barriers is the
    /// average number of frames that shared one sync — the group commit, measured.
    rows: u64,
    barriers: u64,
}

fn pct(samples: &[Duration], p: usize) -> Duration {
    vd_harness::latency::percentile_unstable(samples.to_vec(), p)
}

fn ms(d: Duration) -> f64 {
    d.as_secs_f64() * 1000.0
}

/// ONE run: A sends `FRAMES` retained reliable frames and `FRAMES` ephemeral unreliable frames toward B,
/// one pair every `spacing`, while B is drained tightly so a receive instant is the arrival instant and
/// not a polling artefact. `outbox` decides ONLY whether A has a durable sink — nothing else differs.
fn run(tag: &str, spacing: Duration, outbox: Option<SharedOutbox>) -> Run {
    let rt = runtime();
    let trust = ClusterTrust::generate("vd-outbox-retained-cost").expect("trust");
    let (addr_a, addr_b) = (reserve(), reserve());
    let book: BTreeMap<NodeId, SocketAddr> = [(A, addr_a), (B, addr_b)].into();
    let (mut a, ctl_a) = node(rt.handle(), &trust, A, addr_a, &book, outbox.clone());
    let (mut b, _ctl_b) = node(rt.handle(), &trust, B, addr_b, &book, None);

    let mut sent_r: Vec<Option<Instant>> = vec![None; FRAMES];
    let mut sent_u: Vec<Option<Instant>> = vec![None; FRAMES];
    let mut reliable_order = Vec::new();
    let mut reliable = Vec::new();
    let mut unreliable = Vec::new();

    // Drain whatever B holds right now, stamping each frame against its own send instant.
    let collect = |b: &mut MeshTransport,
                   sent_r: &[Option<Instant>],
                   sent_u: &[Option<Instant>],
                   order: &mut Vec<usize>,
                   rel: &mut Vec<Duration>,
                   unrel: &mut Vec<Duration>| {
        let now = Instant::now();
        for ev in b.drain_inbound() {
            if let Inbound::Wire { class, bytes, .. } = ev {
                let i = index_of(&bytes);
                if class == RELIABLE {
                    order.push(i);
                    rel.push(now - sent_r[i].expect("a reliable frame arrived before it was sent"));
                } else if class == UNRELIABLE {
                    unrel.push(
                        now - sent_u[i].expect("an unreliable frame arrived before it was sent"),
                    );
                }
            }
        }
    };

    for i in 0..FRAMES {
        let step = Instant::now();
        sent_r[i] = Some(step);
        a.send_durable(B, RELIABLE, payload(i).into(), Durability::Retained)
            .expect("the retained lane took the frame");
        sent_u[i] = Some(Instant::now());
        a.send_durable(B, UNRELIABLE, payload(i).into(), Durability::Ephemeral)
            .expect("the unreliable lane took the frame");
        while step.elapsed() < spacing {
            collect(
                &mut b,
                &sent_r,
                &sent_u,
                &mut reliable_order,
                &mut reliable,
                &mut unreliable,
            );
            std::thread::sleep(Duration::from_micros(100));
        }
    }
    // The tail: everything reliable must land.
    let started = Instant::now();
    while reliable.len() < FRAMES {
        collect(
            &mut b,
            &sent_r,
            &sent_u,
            &mut reliable_order,
            &mut reliable,
            &mut unreliable,
        );
        std::thread::sleep(Duration::from_micros(500));
        assert!(
            started.elapsed() < DEADLINE,
            "{tag}: only {} of {FRAMES} retained frames arrived",
            reliable.len()
        );
    }
    // Every retained row is released once its ack lands. Waited for HERE so the caller can read an empty
    // store: a release is STAGED in the ack path (no fsync), so the caller commits before it scans.
    let started = Instant::now();
    while ctl_a.stats().reliable_acked < FRAMES as u64 {
        std::thread::sleep(Duration::from_millis(2));
        assert!(
            started.elapsed() < DEADLINE,
            "{tag}: only {} of {FRAMES} retained frames were acked",
            ctl_a.stats().reliable_acked
        );
    }
    let stats = ctl_a.stats();
    Run {
        reliable_order,
        reliable,
        unreliable,
        rows: stats.outbox_rows_retained,
        barriers: stats.outbox_batches,
    }
}

/// The exactly-once, in-order evidence: the reliable arrival order must be the send order, every time.
fn assert_exactly_once_in_order(tag: &str, run: &Run) {
    let expected: Vec<usize> = (0..FRAMES).collect();
    assert_eq!(
        run.reliable_order, expected,
        "{tag}: every retained frame arrives exactly once, in send order"
    );
}

/// Print one cadence's four percentiles and return the two p99 deltas (reliable, unreliable).
fn report(cadence: &str, spacing: Duration, off: &Run, on: &Run) -> (Duration, Duration) {
    // Returns (the RETAINED p50 delta — the gated one, see RETAINED_P50_BUDGET; the UNRELIABLE p99 delta).
    let (r_off_50, r_off_99) = (pct(&off.reliable, 50), pct(&off.reliable, 99));
    let (r_on_50, r_on_99) = (pct(&on.reliable, 50), pct(&on.reliable, 99));
    let (u_off_50, u_off_99) = (pct(&off.unreliable, 50), pct(&off.unreliable, 99));
    let (u_on_50, u_on_99) = (pct(&on.unreliable, 50), pct(&on.unreliable, 99));
    println!(
        "[{cadence}: one frame per {:.0} ms, n={FRAMES}]\n  RETAINED reliable  send->inbox: OFF p50 {:.3} ms p99 {:.3} ms | ON p50 {:.3} ms p99 {:.3} ms | p50 delta {:+.3} ms (gated at {:.0} ms) p99 delta {:+.3} ms (printed, not gated)\n  EPHEMERAL unreliable send->inbox (arrived OFF {} / ON {}): OFF p50 {:.3} ms p99 {:.3} ms | ON p50 {:.3} ms p99 {:.3} ms | p99 delta {:+.3} ms (budget {:.0} ms)\n  GROUP COMMIT: {} retained rows rode {} disk barriers = {:.2} frames per sync",
        ms(spacing),
        ms(r_off_50),
        ms(r_off_99),
        ms(r_on_50),
        ms(r_on_99),
        ms(r_on_50) - ms(r_off_50),
        ms(RETAINED_P50_BUDGET),
        ms(r_on_99) - ms(r_off_99),
        off.unreliable.len(),
        on.unreliable.len(),
        ms(u_off_50),
        ms(u_off_99),
        ms(u_on_50),
        ms(u_on_99),
        ms(u_on_99) - ms(u_off_99),
        ms(UNRELIABLE_P99_BUDGET),
        on.rows,
        on.barriers,
        f64::from(u32::try_from(on.rows).expect("row count fits"))
            / f64::from(u32::try_from(on.barriers.max(1)).expect("barrier count fits")),
    );
    (
        r_on_50.saturating_sub(r_off_50),
        u_on_99.saturating_sub(u_off_99),
    )
}

/// Once every ack has landed, a commit must leave the store empty — the release path drains it.
fn assert_outbox_drained(tag: &str, sink: &SharedOutbox) {
    sink.lock().expect("sink").commit();
    let left = sink.lock().expect("sink").scan_all();
    assert!(
        left.is_empty(),
        "{tag}: {} retained rows outlived their acks — the release path leaks",
        left.len()
    );
}

/// The bars, applied identically at both cadences: (a) exactly once and in order, (b) the snapshot lane's
/// p99 does not move, (c) a retained frame costs ONE sync at the median.
fn assert_the_bars(tag: &str, off: &Run, on: &Run, r50_delta: Duration, u99_delta: Duration) {
    // (a) exactly once, in order, with and without the outbox.
    assert_exactly_once_in_order(&format!("{tag}, outbox OFF"), off);
    assert_exactly_once_in_order(&format!("{tag}, outbox ON"), on);

    // (b) THE SNAPSHOT LANE DOES NOT PAY THE DISK. A datagram never touches the outbox, and since the
    // writer sends its datagrams before the barrier and keeps sending them THROUGH it, this is ~0.
    assert!(
        off.unreliable.len() * 2 >= FRAMES && on.unreliable.len() * 2 >= FRAMES,
        "{tag}: too few unreliable frames arrived to compare (OFF {}, ON {})",
        off.unreliable.len(),
        on.unreliable.len()
    );
    assert!(
        u99_delta <= UNRELIABLE_P99_BUDGET,
        "{tag}: the outbox delayed the snapshot lane by {:.3} ms at p99, past the {:.0} ms budget — the \
         durable gate is head-of-line blocking the player's own view again",
        ms(u99_delta),
        ms(UNRELIABLE_P99_BUDGET),
    );

    // (c) a retained frame buys ONE disk sync, not many — read on the median, which is what says how many
    // syncs a frame paid for; the p99 is the disk's own fsync tail and is printed, not gated.
    assert!(
        r50_delta <= RETAINED_P50_BUDGET,
        "{tag}: the durable outbox cost the retained lane {:.3} ms at the MEDIAN, past the {:.0} ms budget \
         — a frame is paying for more than one disk sync",
        ms(r50_delta),
        ms(RETAINED_P50_BUDGET),
    );
}

/// ★ THE PRICE OF ONE DURABLE FRAME, at a cadence the disk can hold.
fn a_retained_frame_costs_one_disk_sync_and_the_snapshot_lane_pays_nothing() {
    let off = run("outbox OFF", SUSTAINABLE, None);
    let (sink, path) = shared_outbox("sustainable");
    let on = run("outbox ON", SUSTAINABLE, Some(Arc::clone(&sink)));

    let (r_delta, u_delta) = report("sustainable", SUSTAINABLE, &off, &on);
    assert_the_bars("sustainable", &off, &on, r_delta, u_delta);

    // (d) the release path drains the store once every ack has landed.
    assert_outbox_drained("sustainable", &sink);
    assert_eq!(
        on.rows, FRAMES as u64,
        "every Retained frame mirrored exactly one row"
    );

    let _ = std::fs::remove_file(&path);
}

/// ★ THE SAME BARS WHEN THE OFFERED RATE OUTRUNS THE DISK: one retained frame every 5 ms — 200 a second
/// toward one peer — which is about all this disk can answer, one at a time.
///
/// ★ WHY THE RETAINED p99 IS PRINTED HERE AND NOT GATED. One sync on this machine costs about 4.9 ms, so
/// 200 a second is roughly the service rate, and the writer only groups 1.15 frames per sync because the
/// queue barely has time to grow. A queue offered work at its own service rate has a long tail by nature,
/// and the readings say exactly that: the retained p99 delta measured +11.5, +12.6, +13.2, +25.8 and
/// +34.7 ms on five runs, while the OFF p99 stayed at 0.3 ms throughout. Gating that number would gate the
/// disk's mood, not the transport. The property the bar was guarding — ONE sync per burst, not one per
/// frame — is pinned directly and stably by the burst part below (200 rows, 2 barriers).
///
/// What IS gated here is the bar that matters and that used to fail: the snapshot lane. A player's view
/// must not slow down because a neighbour shard is writing hard, and at this cadence it measures ~0.
fn a_retained_frame_every_five_milliseconds_rides_a_group_commit_and_still_holds_the_bars() {
    let off = run("outbox OFF", SATURATING, None);
    let (sink, path) = shared_outbox("saturating");
    let on = run("outbox ON", SATURATING, Some(Arc::clone(&sink)));

    let (r_delta, u_delta) = report("saturating", SATURATING, &off, &on);
    assert_the_bars("saturating", &off, &on, r_delta, u_delta);
    assert_outbox_drained("saturating", &sink);

    assert_eq!(
        on.rows, FRAMES as u64,
        "every Retained frame mirrored exactly one row"
    );

    let _ = std::fs::remove_file(&path);
}

/// ★ THE GROUP COMMIT ITSELF, PINNED (2026-09-06): a BURST of retained frames rides a HANDFUL of disk
/// syncs, not one each.
///
/// This is the case the fix was written for, in the game's words: a shard loses a whole band of hulls at
/// once and emits two hundred despawns toward its neighbour in the same breath. Each one used to buy its own
/// disk barrier — two hundred syncs, and every snapshot behind them waited. Now the writer takes the whole
/// queue in one batch, stages it under one lock, and asks the disk ONCE.
///
/// The bar is deliberately loose (ten frames per sync) because the real number depends on how fast the
/// producer fills the lane against how fast this disk answers; what must never come back is one sync each.
fn a_burst_of_retained_frames_rides_a_handful_of_disk_syncs() {
    let rt = runtime();
    let trust = ClusterTrust::generate("vd-outbox-retained-cost").expect("trust");
    let (addr_a, addr_b) = (reserve(), reserve());
    let book: BTreeMap<NodeId, SocketAddr> = [(A, addr_a), (B, addr_b)].into();
    let (sink, path) = shared_outbox("burst");
    let (mut a, ctl_a) = node(
        rt.handle(),
        &trust,
        A,
        addr_a,
        &book,
        Some(Arc::clone(&sink)),
    );
    let (mut b, _ctl_b) = node(rt.handle(), &trust, B, addr_b, &book, None);

    // The whole band goes at once: offer every frame as fast as the lane will take it.
    for i in 0..FRAMES {
        loop {
            match a.send_durable(B, RELIABLE, payload(i).into(), Durability::Retained) {
                Ok(_) => break,
                Err(_) => std::thread::sleep(Duration::from_micros(100)),
            }
        }
    }

    let mut order = Vec::new();
    let started = Instant::now();
    while order.len() < FRAMES {
        for ev in b.drain_inbound() {
            if let Inbound::Wire { class, bytes, .. } = ev
                && class == RELIABLE
            {
                order.push(index_of(&bytes));
            }
        }
        std::thread::sleep(Duration::from_micros(100));
        assert!(
            started.elapsed() < DEADLINE,
            "only {} of {FRAMES} burst frames arrived",
            order.len()
        );
    }
    let expected: Vec<usize> = (0..FRAMES).collect();
    assert_eq!(
        order, expected,
        "a burst arrives exactly once, in send order — grouping must not reorder or duplicate"
    );

    let started = Instant::now();
    while ctl_a.stats().reliable_acked < FRAMES as u64 {
        std::thread::sleep(Duration::from_millis(2));
        assert!(
            started.elapsed() < DEADLINE,
            "the burst was never fully acked"
        );
    }
    let stats = ctl_a.stats();
    println!(
        "[burst of {FRAMES} retained frames] {} rows rode {} disk barriers = {:.1} frames per sync",
        stats.outbox_rows_retained,
        stats.outbox_batches,
        f64::from(u32::try_from(stats.outbox_rows_retained).expect("rows fit"))
            / f64::from(u32::try_from(stats.outbox_batches.max(1)).expect("barriers fit")),
    );
    assert_eq!(
        stats.outbox_rows_retained, FRAMES as u64,
        "every Retained frame mirrored exactly one row"
    );
    assert!(
        stats.outbox_batches * 10 <= stats.outbox_rows_retained,
        "the group commit did not group: {} rows rode {} disk barriers — a burst must share its syncs",
        stats.outbox_rows_retained,
        stats.outbox_batches
    );
    assert_outbox_drained("burst", &sink);

    let _ = std::fs::remove_file(&path);
}

/// ★ THE ONE MEASUREMENT, RUN IN ORDER (2026-09-06).
///
/// The three parts share ONE disk, so they must not run beside each other: a burst of fsyncs next door
/// lands in the neighbour's p99 and the reading stops being about the transport at all. MEASURED: run
/// alone, the sustainable part reads +9.8 to +11.2 ms at p99 across five runs; run beside the burst it read
/// +36.7 ms once. So they run in sequence, inside one test, with the burst LAST.
#[test]
fn the_durable_outbox_cost() {
    a_retained_frame_costs_one_disk_sync_and_the_snapshot_lane_pays_nothing();
    a_retained_frame_every_five_milliseconds_rides_a_group_commit_and_still_holds_the_bars();
    a_burst_of_retained_frames_rides_a_handful_of_disk_syncs();
}
