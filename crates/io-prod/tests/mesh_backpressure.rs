//! ★ THE RELIABLE LANE MUST NOT LOSE A FRAME WHEN NOBODY IS DRAINING (2026-08-29).
//!
//! THE DEFECT THIS GATE EXISTS FOR, measured on a live cluster and not argued. The gateway starts
//! accepting mesh traffic as its FIRST act, and the only place the inbound queue is emptied is the
//! first simulation tick — which is about fifty seconds later, because the gateway folds a galaxy's
//! sky in between. The queue filled and the application DISCARDED a reliable frame that QUIC had
//! delivered correctly.
//!
//! That drop was unrecoverable. This transport replays a lane only when its STREAM is gone
//! (`owes_redelivery`), and an inbox drop leaves the stream perfectly healthy — so nothing was ever
//! re-sent. The receiver honestly refused to advance its watermark past a frame it never got, and the
//! two nodes talked past the hole for the life of the process: ONE drop, then a contiguity-gap warning
//! roughly four hundred times every eight seconds, forever. No player could enter the world.
//!
//! ★ WHY `mesh_load` CANNOT CATCH THIS, IN ITS OWN WORDS. That gate sizes the receiver's inbox to the
//! FULL backlog so that "`inbound_dropped_reliable == 0` is a STRUCTURAL invariant, not a
//! drain-vs-fill race". That is the right call for a throughput gate and it is exactly why the real
//! case is invisible there: production cannot pre-size a queue to the traffic it has not seen yet.
//! So this gate uses the DEFAULT capacity and stalls the drain on purpose.
//!
//! THE CURE UNDER TEST: the reliable reader now RESERVES a slot before it reads, and waits when the
//! queue is full. Not reading is how QUIC's flow control is applied — the sender stalls on its own
//! stream window and the frame stays inside the transport, which is where an undeliverable frame
//! belongs. Nothing is dropped, so nothing needs re-requesting.

use std::collections::{BTreeMap, BTreeSet};
use std::net::{SocketAddr, UdpSocket};
use std::time::{Duration, Instant};

use vd_core::NodeId;
use vd_io_prod::mesh::{MeshConfig, MeshControl, MeshTransport, spawn_mesh};
use vd_io_prod::trust::ClusterTrust;
use vd_sim::io::{Inbound, MsgClass, Transport};

const SENDER: NodeId = NodeId(1);
const RECEIVER: NodeId = NodeId(2);

/// How many reliable frames the sender pushes. Chosen against three real limits, not picked:
/// - ABOVE the default inbox (256 outbound x 8 = 2048), so today's build must overflow and drop;
/// - the payload takes the total ABOVE quinn's ~1.25 MB default stream window, so the sender's own
///   `write_all` genuinely stalls and flow control is really exercised rather than assumed;
/// - BELOW the sender's 4 MiB retry buffer, so the SENDER never sheds and the only thing under test
///   is the receiver's behaviour.
const FRAMES: u64 = 4_000;
/// 512 B of payload puts the framed total near 2 MB — past the stream window, inside the retry buffer.
const PAYLOAD_BYTES: usize = 512;
/// How long the receiver refuses to drain. This is the gateway's boot fold, modelled.
const STALL: Duration = Duration::from_secs(10);

fn runtime(worker_threads: usize) -> tokio::runtime::Runtime {
    tokio::runtime::Builder::new_multi_thread()
        .worker_threads(worker_threads)
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

/// Spawn one node at the DEFAULT inbound capacity — the whole point of this gate.
fn spawn_node(
    handle: &tokio::runtime::Handle,
    trust: &ClusterTrust,
    id: NodeId,
    addr: SocketAddr,
    book: &BTreeMap<NodeId, SocketAddr>,
) -> (MeshTransport, MeshControl) {
    let cfg = MeshConfig::new(id, addr, book.clone(), 256, 1, 0);
    spawn_mesh(handle, trust, &cfg, None).expect("mesh node")
}

#[test]
fn a_receiver_that_stalls_loses_no_reliable_frame_and_reports_no_gap() {
    let rt = runtime(4);
    let trust = ClusterTrust::generate("voxeldust").expect("trust");
    let recv_addr = reserve();
    let send_addr = reserve();

    let (mut receiver, recv_ctl) =
        spawn_node(rt.handle(), &trust, RECEIVER, recv_addr, &BTreeMap::new());
    let (mut sender, _send_ctl) = spawn_node(
        rt.handle(),
        &trust,
        SENDER,
        send_addr,
        &BTreeMap::from([(RECEIVER, recv_addr)]),
    );

    // THE SENDER PUSHES EVERYTHING while the receiver is asleep at the till.
    let payload = vec![0_u8; PAYLOAD_BYTES];
    for seq in 0..FRAMES {
        let mut body = payload.clone();
        body[..8].copy_from_slice(&seq.to_le_bytes());
        loop {
            match sender.send(RECEIVER, MsgClass::Saga, body.clone().into()) {
                Ok(_) => break,
                // The SENDER's own staging is full — that is lawful back-pressure on this side and
                // not the thing under test. Wait and retry.
                Err(_) => std::thread::sleep(Duration::from_micros(200)),
            }
        }
    }

    // ★ NOBODY DRAINS. This is the gateway's fifty-second boot fold, modelled.
    std::thread::sleep(STALL);

    // Now start reading, and keep reading until everything has arrived.
    let mut seen: BTreeSet<u64> = BTreeSet::new();
    let deadline = Instant::now() + Duration::from_secs(60);
    while seen.len() < FRAMES as usize && Instant::now() < deadline {
        for ev in receiver.drain_inbound() {
            if let Inbound::Wire { from, class, bytes } = ev {
                assert_eq!(from, SENDER);
                assert_eq!(class, MsgClass::Saga);
                let mut n = [0_u8; 8];
                n.copy_from_slice(&bytes[..8]);
                assert!(
                    seen.insert(u64::from_le_bytes(n)),
                    "a frame was delivered twice"
                );
            }
        }
        std::thread::sleep(Duration::from_millis(2));
    }

    let stats = recv_ctl.stats();

    // ★ THE VERDICT, in three statements that fail for different reasons.
    //
    // (1) NOTHING WAS LOST. Before the cure this is where it failed: the queue holds 2048 and 4000
    //     were sent into a receiver that read none of them.
    assert_eq!(
        seen.len(),
        FRAMES as usize,
        "{} of {FRAMES} reliable frames arrived — a stalled receiver lost {}",
        seen.len(),
        FRAMES as usize - seen.len()
    );
    // (2) NOTHING WAS DISCARDED BY US. QUIC delivered them; the application must not bin what it was
    //     correctly given.
    assert_eq!(
        stats.inbound_dropped_reliable, 0,
        "the application discarded a reliable frame it had been delivered"
    );
    // (3) AND THE TRIPWIRE NEVER FIRED. The gap counter is not a recovery path and never was — after
    //     this cure it is an assertion. A gap on a QUIC stream cannot come from the wire, so a
    //     non-zero here means we made the hole ourselves.
    assert_eq!(
        stats.gap_drop, 0,
        "a contiguity gap appeared on a QUIC stream — the wire cannot reorder, so we dropped it"
    );
}
