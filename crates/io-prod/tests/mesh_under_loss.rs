//! R-5 (the D-6 #1 producer-less-flow CAPSTONE). A `GhostFlow::Despawn` is emitted exactly ONCE on
//! band-exit and has NO saga, NO `scan_deadlines` re-driver — a genuinely PRODUCER-LESS reliable one-shot
//! (unlike a transfer saga, nothing at the application layer will re-emit it). If it is lost to a
//! connection blip and the lane then goes IDLE, the ONLY path to its delivery is the transport's own R-4a
//! retransmit timer (off the sender's clock, no application re-drive). This test proves the transport
//! delivers such a flow AT-LEAST-ONCE across a blip for a source that STAYS UP — retiring the D-6 #1
//! argument at the transport layer — and that a real `InterShardFlow::Ghost` envelope round-trips exactly.
//!
//! Run by `just mesh-load` alongside the N-peer load gate. The remaining D-6 #1 residual (a producer-less
//! flow whose SOURCE CRASHES before the dest adopts — `BatchHandoff::AwaitAdopt`) is a SEPARATE, ledgered
//! item gated on M3 (durable incarnation) + L5 (addr re-plumb): it is NOT what this capstone proves.

use std::collections::BTreeMap;
use std::net::{SocketAddr, UdpSocket};
use std::time::{Duration, Instant};

use vd_core::entity_kind::EntityKind;
use vd_core::{EntityId, Fence, NodeId};
use vd_io_prod::mesh::{MeshConfig, MeshControl, MeshTransport, spawn_mesh};
use vd_io_prod::trust::ClusterTrust;
use vd_sim::io::{Inbound, MsgClass, Transport};
use vd_wire::intershard::{GhostFlow, InterShardFlow};

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

fn reserve() -> SocketAddr {
    UdpSocket::bind("127.0.0.1:0")
        .expect("reserve port")
        .local_addr()
        .expect("addr")
}

fn node(
    handle: &tokio::runtime::Handle,
    trust: &ClusterTrust,
    id: NodeId,
    addr: SocketAddr,
    book: &BTreeMap<NodeId, SocketAddr>,
    ack_flush: Duration,
) -> (MeshTransport, MeshControl) {
    let mut cfg = MeshConfig::new(id, addr, book.clone(), 256, 1);
    cfg.reliability.ack_idle_flush_interval = ack_flush;
    spawn_mesh(handle, trust, &cfg).expect("mesh node")
}

/// Enqueue one reliable frame on a chosen class, tolerating per-peer back-pressure.
fn send_on(t: &mut MeshTransport, to: NodeId, class: MsgClass, payload: Vec<u8>) {
    loop {
        match t.send(to, class, payload.clone().into()) {
            Ok(_) => break,
            Err(_) => std::thread::sleep(Duration::from_micros(200)),
        }
    }
}

/// Drain and collect the payloads of every delivered `GhostReliable` `Wire` event.
fn drain_ghost(t: &mut MeshTransport, out: &mut Vec<Vec<u8>>) {
    for ev in t.drain_inbound() {
        if let Inbound::Wire { class, bytes, .. } = ev
            && class == MsgClass::GhostReliable
        {
            out.push(bytes.to_vec());
        }
    }
}

fn wait_until(mut predicate: impl FnMut() -> bool, msg: &str) {
    let start = Instant::now();
    while !predicate() {
        std::thread::sleep(Duration::from_millis(2));
        assert!(start.elapsed() < DEADLINE, "timed out: {msg}");
    }
}

#[test]
fn a_producer_less_ghost_despawn_survives_an_idle_after_blip_via_the_timer() {
    let rt = runtime();
    let trust = ClusterTrust::generate("vd-mesh-under-loss").expect("trust");
    let (addr_a, addr_b) = (reserve(), reserve());
    let book: BTreeMap<_, _> = [(A, addr_a), (B, addr_b)].into();
    let (mut a, ctl_a) = node(
        rt.handle(),
        &trust,
        A,
        addr_a,
        &book,
        Duration::from_millis(20),
    );
    let (mut b, ctl_b) = node(
        rt.handle(),
        &trust,
        B,
        addr_b,
        &book,
        Duration::from_millis(20),
    );

    // Establish the GhostReliable lane with a SPAWN marker (delivered) so `drop_connections` has a live
    // connection to blip. Semantically: the ghost was spawned; the Despawn below is the band-exit one-shot.
    let spawn_marker = vec![0xEEu8];
    let mut got: Vec<Vec<u8>> = Vec::new();
    send_on(&mut a, B, MsgClass::GhostReliable, spawn_marker.clone());
    wait_until(
        || {
            drain_ghost(&mut b, &mut got);
            got.contains(&spawn_marker)
        },
        "the spawn marker never established the ghost lane",
    );

    // BLIP; let the CONNECTION_CLOSE land (so the lone Despawn's initial write PROVABLY hits the dead
    // connection ⇒ WriteFail::Down ⇒ the R-4a timer is armed).
    ctl_a.drop_connections();
    std::thread::sleep(Duration::from_millis(150));

    // The LONE producer-less Despawn — a REAL InterShardFlow::Ghost envelope — then IDLE (no follow-up
    // sends). The ONLY path to its delivery is the R-4a retransmit timer off A's own clock.
    let despawn = InterShardFlow::Ghost(GhostFlow::Despawn {
        entity: EntityId::pack(EntityKind::Ship, 1, 42, 7),
        source_fence: Fence(9),
    });
    let despawn_bytes = postcard::to_allocvec(&despawn).expect("encode the Despawn envelope");
    send_on(&mut a, B, MsgClass::GhostReliable, despawn_bytes.clone());

    // B receives the Despawn — delivered PURELY by the retransmit timer (no application re-drive exists
    // for a band-exit ghost Despawn).
    wait_until(
        || {
            drain_ghost(&mut b, &mut got);
            got.contains(&despawn_bytes)
        },
        "the producer-less Despawn was never re-driven by the timer (D-6 #1 not cured)",
    );
    // The full at-least-once CYCLE: the Despawn's ack round-trips B→A only AFTER B receives it, so wait for
    // both frames to RETIRE (delivered AND acked) — a bare post-delivery sample of reliable_acked would race
    // the in-flight ack and flake under load.
    wait_until(
        || ctl_a.stats().reliable_acked >= 2,
        "the spawn marker + timer-re-driven Despawn never both retired (acked)",
    );

    // Exactly once: the Despawn payload appears in B's delivered GhostReliable stream exactly one time.
    let despawn_count = got.iter().filter(|p| **p == despawn_bytes).count();
    assert_eq!(
        despawn_count, 1,
        "the producer-less Despawn delivered EXACTLY once"
    );
    // The real envelope round-trips byte-exact: decode B's RECEIVED copy (NOT the local sent bytes) so the
    // assertion stands alone — the wire payload B actually delivered decodes back to the exact envelope A
    // sent (proving the end-goal Ghost wire shape survives the loss path).
    let received = got
        .iter()
        .find(|p| **p == despawn_bytes)
        .expect("B received the Despawn payload");
    let decoded: InterShardFlow =
        postcard::from_bytes(received).expect("B's delivered Despawn decodes");
    assert_eq!(
        decoded, despawn,
        "the GhostFlow::Despawn envelope B received round-trips exactly"
    );
    // Receiver honesty + producer-less at-least-once: no contiguity gap; the source stayed UP (a blip is
    // NOT a death — ZERO NodeUnreachable bounced to A); A's endpoint survived; both frames retired.
    assert_eq!(
        ctl_b.stats().gap_drop,
        0,
        "no contiguity gap on the re-driven lane"
    );
    let a_inbound = a.drain_inbound();
    assert!(
        !a_inbound
            .iter()
            .any(|m| matches!(m, Inbound::NodeUnreachable { .. })),
        "a recovering blip is NOT a death — the timer re-drove before the confirm threshold"
    );
    assert!(ctl_a.local_addr().is_ok(), "A's endpoint survived the blip");
    let _ = &mut b;
}
