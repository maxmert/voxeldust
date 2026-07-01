//! R-3' end-to-end: the redelivering mesh transport over REAL quinn loopback. Proves the whole
//! at-least-once machinery LINKS on the real directional mesh — the reverse cumulative-ack round-trip
//! (data receiver's `serve_connection` opens the ACK stream → data sender's `peer_writer` ack-reader →
//! `on_ack` retire), the `drop_connections` transient-blip recovery (epoch bump → replay → dedup →
//! exactly-once), the lone-frame idle-flush, and the sender-restart incarnation reset. The pure verdict
//! ladder + retire FSM are unit-tested in `mesh.rs`; this file gates the async glue against the wire.

use std::collections::BTreeMap;
use std::net::{SocketAddr, UdpSocket};
use std::time::{Duration, Instant};

use vd_core::NodeId;
use vd_io_prod::mesh::{MeshConfig, MeshControl, MeshTransport, spawn_mesh};
use vd_io_prod::trust::ClusterTrust;
use vd_sim::io::{Inbound, MsgClass, Transport};

const DEADLINE: Duration = Duration::from_secs(30);
const A: NodeId = NodeId(1);
const B: NodeId = NodeId(2);

fn runtime() -> tokio::runtime::Runtime {
    tokio::runtime::Builder::new_multi_thread()
        .worker_threads(2)
        .enable_all()
        .build()
        .expect("tokio runtime")
}

/// Reserve an ephemeral loopback UDP port (bind `:0`, read it back, drop) for a node to re-bind.
fn reserve() -> SocketAddr {
    UdpSocket::bind("127.0.0.1:0")
        .expect("reserve port")
        .local_addr()
        .expect("addr")
}

/// Spawn one mesh node with a chosen ack-idle-flush cadence + process incarnation.
fn node(
    handle: &tokio::runtime::Handle,
    trust: &ClusterTrust,
    id: NodeId,
    addr: SocketAddr,
    book: &BTreeMap<NodeId, SocketAddr>,
    ack_flush: Duration,
    incarnation: u64,
) -> (MeshTransport, MeshControl) {
    let mut cfg = MeshConfig::new(id, addr, book.clone(), 256, incarnation);
    cfg.reliability.ack_idle_flush_interval = ack_flush;
    spawn_mesh(handle, trust, &cfg).expect("mesh node")
}

/// Drain the node's inbound and collect the first payload byte of every delivered `Wire` event.
fn drain_wire_values(t: &mut MeshTransport, out: &mut Vec<u8>) {
    for ev in t.drain_inbound() {
        if let Inbound::Wire { bytes, .. } = ev {
            out.push(bytes[0]);
        }
    }
}

/// Enqueue one reliable frame, tolerating per-peer back-pressure (retry until accepted).
fn send_reliable(t: &mut MeshTransport, to: NodeId, value: u8) {
    loop {
        match t.send(to, MsgClass::Saga, vec![value].into()) {
            Ok(_) => break,
            Err(_) => std::thread::sleep(Duration::from_micros(200)),
        }
    }
}

/// Poll a predicate until it holds or the deadline trips.
fn wait_until(mut predicate: impl FnMut() -> bool, msg: &str) {
    let start = Instant::now();
    while !predicate() {
        std::thread::sleep(Duration::from_millis(2));
        assert!(start.elapsed() < DEADLINE, "timed out: {msg}");
    }
}

#[test]
fn happy_burst_acks_retire_the_whole_window_reliable_acked_reaches_n() {
    // The ack round-trip actually links: B acks, A retires its ENTIRE retry window (base -> next_seq).
    // A dead ack path (the CRITICAL#2 topology risk) would leave reliable_acked stuck at 0.
    const N: usize = 60;
    let rt = runtime();
    let trust = ClusterTrust::generate("vd-mesh-redeliver").expect("trust");
    let (addr_a, addr_b) = (reserve(), reserve());
    let book: BTreeMap<_, _> = [(A, addr_a), (B, addr_b)].into();
    let (mut a, ctl_a) = node(rt.handle(), &trust, A, addr_a, &book, Duration::from_millis(20), 1);
    let (mut b, ctl_b) = node(rt.handle(), &trust, B, addr_b, &book, Duration::from_millis(20), 1);

    for i in 0..N {
        send_reliable(&mut a, B, i as u8);
    }

    // B receives all N, exactly once (no loss, no dup).
    let mut got = Vec::new();
    wait_until(
        || {
            drain_wire_values(&mut b, &mut got);
            got.len() >= N
        },
        "B never received all N frames",
    );
    got.sort_unstable();
    assert_eq!(
        got,
        (0..N as u8).collect::<Vec<_>>(),
        "exactly-once delivery: no loss, no dup"
    );

    // The reverse-lane acks retire A's whole retry window.
    wait_until(
        || ctl_a.stats().reliable_acked as usize >= N,
        "A's retry window never fully retired (dead ack path?)",
    );
    let sa = ctl_a.stats();
    assert_eq!(sa.reliable_acked as usize, N, "every frame retired exactly once");
    assert_eq!(sa.reliable_shed, 0, "no shed on the happy path");
    // Receiver honesty: no gaps, no dedups on the happy (blip-free) path.
    let sb = ctl_b.stats();
    assert_eq!(
        (sb.gap_drop, sb.dedup_drop, sb.stale_epoch_drop),
        (0, 0, 0),
        "no gap/dedup/stale on a clean single-stream burst"
    );
}

#[test]
fn a_drop_connections_blip_mid_burst_delivers_exactly_once() {
    // The transient-blip lever: close A's dialed connection WHILE the writer is actively draining a burst.
    // A re-dials, bumps its epoch, and REPLAYS its un-acked window; every frame arrives EXACTLY ONCE despite
    // the break (the D-6 #1 producer-less-flow cure), and — R-4a — the recovering blip bounces ZERO spurious
    // NodeUnreachable (the peer is alive so the first re-dial recovers before `confirm_unreachable_after_retries`).
    // NOTE: the acks flow promptly (per-frame), so at the instant of the blip the retry window is small and
    // mostly un-delivered — the replay re-covers only the un-acked tail. That is OPTIMAL (don't resend what's
    // acked), so `dedup_drop` is NOT asserted (it fires only in the narrow delivered-but-un-acked race — its
    // logic is proven deterministically by the pure `classify_dedups_*` unit test). What IS deterministic and
    // asserted here: no loss, no dup, and R-4a's zero-spurious-bounce blip tolerance.
    // SCOPE: phase-2 keeps traffic flowing AFTER the blip (the send path also re-dials). The IDLE-after-blip
    // case (a lone frame that blips then goes quiet) is now re-driven by the R-4a retransmit TIMER off the
    // sender's own clock — covered by `an_idle_after_blip_lone_frame_is_re_driven_by_the_timer` below.
    const N: usize = 100; // < 256 so a single-byte payload is a unique per-frame id
    let rt = runtime();
    let trust = ClusterTrust::generate("vd-mesh-redeliver").expect("trust");
    let (addr_a, addr_b) = (reserve(), reserve());
    let book: BTreeMap<_, _> = [(A, addr_a), (B, addr_b)].into();
    let (mut a, ctl_a) = node(rt.handle(), &trust, A, addr_a, &book, Duration::from_millis(20), 1);
    let (mut b, ctl_b) = node(rt.handle(), &trust, B, addr_b, &book, Duration::from_millis(20), 1);

    // Phase 1: deliver a chunk, then blip. Let the CONNECTION_CLOSE propagate (sleep) BEFORE driving phase-2,
    // so phase-2's first write PROVABLY hits the dead connection (→ NodeUnreachable bounce → epoch bump →
    // redial → replay), independent of scheduler load. (An enqueue is instant, so without the sleep a
    // contended run can slip phase-2's writes through before quinn processes the close.)
    let mut got = Vec::new();
    for i in 0..N / 2 {
        send_reliable(&mut a, B, i as u8);
    }
    wait_until(
        || {
            drain_wire_values(&mut b, &mut got);
            got.len() >= N / 4
        },
        "no delivery chunk arrived before the blip",
    );
    ctl_a.drop_connections();
    std::thread::sleep(Duration::from_millis(150));
    for i in N / 2..N {
        send_reliable(&mut a, B, i as u8);
    }
    wait_until(
        || {
            drain_wire_values(&mut b, &mut got);
            got.len() >= N
        },
        "B never received all N frames across the blip",
    );
    got.sort_unstable();
    assert_eq!(
        got,
        (0..N as u8).collect::<Vec<_>>(),
        "exactly-once across the blip: no loss, no dup"
    );

    // R-4a BLIP-TOLERANCE: the peer is alive (only the connection blipped), so the first re-dial recovers
    // and NO lane ever reaches `confirm_unreachable_after_retries` consecutive failures ⇒ ZERO spurious
    // NodeUnreachable bounces. (Under R-3' this bounced once per in-flight failed write; R-4a's threshold-
    // gated confirm is exactly the "a blip = zero bounce" cure.) A genuinely-dead peer still bounces after
    // N retries — covered by the lib test `a_dead_peer_never_blocks_traffic_to_live_peers`.
    let unreachable = a
        .drain_inbound()
        .into_iter()
        .filter(|e| matches!(e, Inbound::NodeUnreachable { .. }))
        .count();
    assert_eq!(
        unreachable, 0,
        "a recovering blip must bounce ZERO NodeUnreachable (R-4a confirm-dead-after-N)"
    );
    // The epoch/replay kept contiguity — no frame was ever a MUST-BE-0 gap.
    assert_eq!(ctl_b.stats().gap_drop, 0, "gap_drop must stay 0 across the blip");
    // The endpoint SURVIVED (drop_connections is a blip, NOT a kill).
    assert!(
        ctl_a.local_addr().is_ok(),
        "the endpoint stayed bound across drop_connections"
    );
}

#[test]
fn an_idle_after_blip_lone_frame_is_re_driven_by_the_timer() {
    // THE R-4a cure (the gap the R-3' post-impl review found): a lone reliable frame that blips then goes
    // QUIET (no follow-up send) is re-driven off the sender's own retransmit TIMER. Under R-3' the re-dial +
    // replay fired ONLY inside a new send's dial block, so an idle-after-blip frame stranded in the retry
    // buffer indefinitely. Here exactly ONE frame is sent after the blip and then the lane IDLES: the ONLY
    // path to its delivery is the R-4a timer.
    let rt = runtime();
    let trust = ClusterTrust::generate("vd-mesh-redeliver").expect("trust");
    let (addr_a, addr_b) = (reserve(), reserve());
    let book: BTreeMap<_, _> = [(A, addr_a), (B, addr_b)].into();
    let (mut a, ctl_a) = node(rt.handle(), &trust, A, addr_a, &book, Duration::from_millis(20), 1);
    let (mut b, _ctl_b) = node(rt.handle(), &trust, B, addr_b, &book, Duration::from_millis(20), 1);

    // Establish the connection with one delivered frame.
    send_reliable(&mut a, B, 0);
    let mut got = Vec::new();
    wait_until(
        || {
            drain_wire_values(&mut b, &mut got);
            got.contains(&0)
        },
        "the first frame never arrived",
    );

    // Blip, and let the close land (so the lone frame's initial write PROVABLY hits the dead connection ⇒
    // WriteFail::Down ⇒ the timer is armed).
    ctl_a.drop_connections();
    std::thread::sleep(Duration::from_millis(150));

    // The LONE frame, then IDLE — no further sends. Only the R-4a retransmit timer can deliver it.
    send_reliable(&mut a, B, 1);
    wait_until(
        || {
            drain_wire_values(&mut b, &mut got);
            got.contains(&1)
        },
        "the idle-after-blip lone frame was NEVER re-driven (the R-4a timer gap the review found)",
    );
    got.sort_unstable();
    assert_eq!(
        got,
        vec![0, 1],
        "both frames delivered exactly once across the idle-after-blip re-drive"
    );
}

#[test]
fn a_lone_reliable_frame_is_acked_via_the_idle_flush() {
    // A single reliable frame with NO follow-up traffic is acked purely by the receiver's idle-flush timer
    // (the `interval` arm of the ack-egress select) — the lone-damage/death/despawn case.
    let rt = runtime();
    let trust = ClusterTrust::generate("vd-mesh-redeliver").expect("trust");
    let (addr_a, addr_b) = (reserve(), reserve());
    let book: BTreeMap<_, _> = [(A, addr_a), (B, addr_b)].into();
    let (mut a, ctl_a) = node(rt.handle(), &trust, A, addr_a, &book, Duration::from_millis(30), 1);
    let (mut b, _ctl_b) = node(rt.handle(), &trust, B, addr_b, &book, Duration::from_millis(30), 1);

    a.send(B, MsgClass::Saga, vec![7].into()).expect("enqueued");

    let mut got = Vec::new();
    wait_until(
        || {
            drain_wire_values(&mut b, &mut got);
            !got.is_empty()
        },
        "the lone frame never arrived",
    );
    assert_eq!(got, vec![7]);

    wait_until(
        || ctl_a.stats().reliable_acked == 1,
        "the lone frame was never idle-flush-acked",
    );
}

#[test]
fn a_sender_restart_at_a_higher_incarnation_is_not_deduped() {
    // The sender-restart seq-reset cure: A restarts at a HIGHER incarnation (its seq resets to 0), while B
    // keeps running with a SURVIVING ledger holding A's old incarnation + a high hw. The higher incarnation
    // RESETS B's dedup state, so the restarted sender's fresh seq0.. deliver — never silently deduped.
    const M: u8 = 20;
    let rt = runtime();
    let trust = ClusterTrust::generate("vd-mesh-redeliver").expect("trust");
    let addr_b = reserve();
    let addr_a1 = reserve();
    let addr_a2 = reserve();
    // B never dials A (it only receives + acks on the reverse stream), so A may restart on a NEW port.
    let book_b: BTreeMap<_, _> = [(A, addr_a1), (B, addr_b)].into();
    let (mut b, ctl_b) = node(rt.handle(), &trust, B, addr_b, &book_b, Duration::from_millis(30), 9);

    // Incarnation 1: A sends M frames (values 0..M). B records incarnation 1 with hw = M-1.
    let book_a1: BTreeMap<_, _> = [(B, addr_b)].into();
    let (mut a1, ctl_a1) = node(rt.handle(), &trust, A, addr_a1, &book_a1, Duration::from_millis(30), 1);
    for i in 0..M {
        send_reliable(&mut a1, B, i);
    }
    let mut got = Vec::new();
    wait_until(
        || {
            drain_wire_values(&mut b, &mut got);
            got.len() >= M as usize
        },
        "B never received the first incarnation's frames",
    );

    // Restart A: kill incarnation 1 (endpoint closes), come up at incarnation 2 on a NEW port.
    ctl_a1.kill();
    drop(a1);
    drop(ctl_a1);
    let book_a2: BTreeMap<_, _> = [(B, addr_b)].into();
    let (mut a2, _ctl_a2) = node(rt.handle(), &trust, A, addr_a2, &book_a2, Duration::from_millis(30), 2);

    // A' sends M fresh frames (values 100..100+M) with seq RESET to 0 — delivered, NOT deduped.
    for i in 0..M {
        send_reliable(&mut a2, B, 100 + i);
    }
    wait_until(
        || {
            drain_wire_values(&mut b, &mut got);
            (100..100 + M).all(|v| got.contains(&v))
        },
        "the restarted sender's frames were DEDUPED (silent loss)",
    );

    assert_eq!(
        ctl_b.stats().stale_incarnation_drop,
        0,
        "a HIGHER incarnation RESETS the ledger, it never drops the restarted sender's frames"
    );
}
