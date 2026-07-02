//! R-4e1: the N-peer real-QUIC LOAD gate for the redelivering MeshTransport. Fan-in — N senders
//! (`NodeId(2..=N+1)`) each drive a sustained reliable burst into ONE receiver (`NodeId(1)`) — the
//! RX-plane-collapse geometry the H2 concern predicts (every sender's reliable RX serializes on the
//! receiver's single `RecvLedger` mutex today). This slice commits with H2 UNCHANGED: it proves the
//! transport is CORRECT under N-peer fan-in (no-loss/no-dup, acks keep pace, `gap_drop==0`, and — via a
//! SIZED receiver inbox — `inbound_dropped_reliable==0` structurally), and becomes the regression baseline
//! the H2 per-peer re-key (R-4e2) is measured against.
//!
//! Determinism: every correctness assertion is EXACT + completion-gated (wait for the delivery COUNT to
//! reach the target, THEN assert the exact per-sender set / counters) — never a timed sample. The ONLY
//! wall-clock assertion is a GENEROUS progress ceiling that scales with N; an RX collapse manifests as
//! never reaching the count (a deadline trip = a correct failure), never a flake.
//!
//! N is `VD_MESH_LOAD_NODES` (default 16 — the audit's always-feasible floor); `just mesh-load` pins the
//! soak N. A bare `cargo test -p vd-io-prod --test mesh_load` at the 16 default stays inner-loop-feasible.

use std::collections::{BTreeMap, BTreeSet};
use std::net::{SocketAddr, UdpSocket};
use std::time::{Duration, Instant};

use vd_core::NodeId;
use vd_io_prod::mesh::{MeshConfig, MeshControl, MeshTransport, spawn_mesh};
use vd_io_prod::trust::ClusterTrust;
use vd_sim::io::{Inbound, MsgClass, Transport};

const RECEIVER: NodeId = NodeId(1);

/// The ONE config home for the load shape (HR5 no-magic-numbers): every knob here, justified, never
/// an inline literal at a use site. The receiver inbox capacity and the progress deadline are DERIVED
/// from the shape, not bare constants.
struct LoadShape {
    /// N sender peers fanning into the one receiver. `VD_MESH_LOAD_NODES` or the 16 floor.
    senders: usize,
    /// Sustained reliable depth per sender lane (matches the `mesh_volume` burst depth).
    frames_per_sender: usize,
    /// A deliberately BOUNDED tokio pool: the point of the load test is to provoke the single-ledger-
    /// mutex contention on a realistic core count, not to dissolve it under one-thread-per-peer.
    worker_threads: usize,
    /// Fixed floor of the progress deadline (connection setup + teardown slack).
    deadline_base: Duration,
    /// Per-sender slice of the progress deadline — scales the ceiling with N and the ack idle-flush tail
    /// so a larger fan-in gets proportionally more wall-clock before the (generous) ceiling trips.
    per_sender_budget: Duration,
}

impl LoadShape {
    fn from_env() -> LoadShape {
        let senders = std::env::var("VD_MESH_LOAD_NODES")
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
            .filter(|n| *n >= 1)
            .unwrap_or(16);
        LoadShape {
            senders,
            frames_per_sender: 200,
            worker_threads: 4,
            deadline_base: Duration::from_secs(15),
            per_sender_budget: Duration::from_millis(750),
        }
    }

    fn total_frames(&self) -> usize {
        self.senders * self.frames_per_sender
    }

    /// The receiver inbox sized to the FULL reliable backlog + margin (finding 1-CRITICAL): even if the
    /// sim thread never drained until the end, no reliable frame can be dropped — so
    /// `inbound_dropped_reliable == 0` is a STRUCTURAL invariant, not a drain-vs-fill race.
    fn receiver_inbox_capacity(&self) -> usize {
        self.total_frames() + 256
    }

    /// The generous progress ceiling — the only wall-clock bound. Scales with N.
    fn deadline(&self) -> Duration {
        self.deadline_base + self.per_sender_budget * self.senders as u32
    }
}

fn runtime(worker_threads: usize) -> tokio::runtime::Runtime {
    tokio::runtime::Builder::new_multi_thread()
        .worker_threads(worker_threads)
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

/// Spawn one mesh node. `inbound_capacity` overrides the derived default (the receiver sizes it to the
/// full backlog; senders keep the default). `book` is the node's dial book (empty for the receiver — it
/// dials nobody, only accepts + acks on the accepted connection).
fn spawn_node(
    handle: &tokio::runtime::Handle,
    trust: &ClusterTrust,
    id: NodeId,
    addr: SocketAddr,
    book: &BTreeMap<NodeId, SocketAddr>,
    inbound_capacity: Option<usize>,
) -> (MeshTransport, MeshControl) {
    let mut cfg = MeshConfig::new(id, addr, book.clone(), 256, 1);
    if let Some(cap) = inbound_capacity {
        cfg.inbound_capacity = cap;
    }
    spawn_mesh(handle, trust, &cfg).expect("mesh node")
}

/// Enqueue one reliable frame, tolerating per-peer back-pressure (retry until accepted). The payload is
/// the seq as `u64` LE so each frame within a sender's window has a unique id (frames_per_sender may
/// exceed 255 — a single-byte id would collide).
fn send_reliable(t: &mut MeshTransport, to: NodeId, seq: u64) {
    let payload = seq.to_le_bytes().to_vec();
    loop {
        match t.send(to, MsgClass::Saga, payload.clone().into()) {
            Ok(_) => break,
            Err(_) => std::thread::sleep(Duration::from_micros(200)),
        }
    }
}

/// Drain the receiver and fold every delivered reliable `Wire` into per-sender seq sets. Asserts NO
/// `SendShed`/`NodeUnreachable` ever surfaces on the receiver (a clean fan-in has neither).
fn drain_into(t: &mut MeshTransport, got: &mut BTreeMap<NodeId, BTreeSet<u64>>) {
    for ev in t.drain_inbound() {
        match ev {
            Inbound::Wire { from, bytes, .. } => {
                let mut b = [0u8; 8];
                b.copy_from_slice(&bytes[..8]);
                got.entry(from).or_default().insert(u64::from_le_bytes(b));
            }
            other => panic!("receiver saw a non-Wire inbound in a clean fan-in: {other:?}"),
        }
    }
}

/// Poll a predicate until it holds or the (generous, N-scaled) deadline trips. A collapse ⇒ never
/// reaching the target ⇒ this trips — a correct failure, not a flake.
fn wait_until(deadline: Duration, mut predicate: impl FnMut() -> bool, msg: &str) {
    let start = Instant::now();
    while !predicate() {
        std::thread::sleep(Duration::from_millis(2));
        assert!(
            start.elapsed() < deadline,
            "timed out after {:?}: {msg}",
            start.elapsed()
        );
    }
}

#[test]
fn n_peer_fan_in_sustained_reliable_is_loss_free_and_acks_keep_pace() {
    let shape = LoadShape::from_env();
    let deadline = shape.deadline();
    let rt = runtime(shape.worker_threads);
    let trust = ClusterTrust::generate("vd-mesh-load").expect("trust");

    // The receiver binds FIRST (listening before any sender dials) with an inbox SIZED to the full
    // backlog. It dials nobody — an empty book.
    let recv_addr = reserve();
    let empty_book: BTreeMap<NodeId, SocketAddr> = BTreeMap::new();
    let (mut receiver, recv_ctl) = spawn_node(
        rt.handle(),
        &trust,
        RECEIVER,
        recv_addr,
        &empty_book,
        Some(shape.receiver_inbox_capacity()),
    );

    // N senders, each with a book of ONLY the receiver (fan-in). Held on the test thread.
    let sender_book: BTreeMap<NodeId, SocketAddr> = [(RECEIVER, recv_addr)].into();
    let mut senders: Vec<MeshTransport> = Vec::with_capacity(shape.senders);
    let mut sender_ctls: Vec<MeshControl> = Vec::with_capacity(shape.senders);
    for i in 0..shape.senders {
        let id = NodeId((i + 2) as u64); // 2..=N+1 (RECEIVER is 1)
        let (tx, ctl) = spawn_node(rt.handle(), &trust, id, reserve(), &sender_book, None);
        senders.push(tx);
        sender_ctls.push(ctl);
    }

    // Drive: ROUND-ROBIN across senders per frame index, so all N lanes stay busy concurrently (the
    // sustained fan-in), rather than draining one sender's whole window before the next starts.
    for seq in 0..shape.frames_per_sender as u64 {
        for tx in &mut senders {
            send_reliable(tx, RECEIVER, seq);
        }
    }

    // EXACT, completion-gated: wait for the delivered COUNT to reach the target, THEN assert every
    // sender's set is exactly 0..frames_per_sender (per-`from` grouping catches a cancelling loss+dup a
    // global multiset would miss).
    let mut got: BTreeMap<NodeId, BTreeSet<u64>> = BTreeMap::new();
    wait_until(
        deadline,
        || {
            drain_into(&mut receiver, &mut got);
            got.values().map(BTreeSet::len).sum::<usize>() >= shape.total_frames()
        },
        "the receiver never delivered every fan-in frame (RX-plane collapse?)",
    );
    assert_eq!(
        got.len(),
        shape.senders,
        "every sender delivered at least one frame"
    );
    let expected: BTreeSet<u64> = (0..shape.frames_per_sender as u64).collect();
    for i in 0..shape.senders {
        let id = NodeId((i + 2) as u64);
        assert_eq!(
            got.get(&id),
            Some(&expected),
            "sender {id} must deliver EXACTLY 0..{} — no loss, no dup",
            shape.frames_per_sender
        );
    }

    // EXACT, per-sender: acks keep pace — each sender's whole window retired. Per-sender wait (NOT a
    // sum) so one lagging sender trips the deadline rather than a peer masking another.
    wait_until(
        deadline,
        || {
            sender_ctls
                .iter()
                .all(|c| c.stats().reliable_acked as usize >= shape.frames_per_sender)
        },
        "some sender's retry window never fully retired (a dead/contended ack path)",
    );
    for (i, c) in sender_ctls.iter().enumerate() {
        let s = c.stats();
        assert_eq!(
            s.reliable_acked as usize,
            shape.frames_per_sender,
            "sender {} retired exactly its window, no more no less",
            i + 2
        );
        assert_eq!(
            s.reliable_shed,
            0,
            "sender {} shed nothing on a healthy fan-in",
            i + 2
        );
    }

    // Receiver honesty: no gaps, no stale-epoch drops, and — via the SIZED inbox — NO reliable inbox
    // drop (structural, not a drain race). This is the RX-plane correctness under N-peer fan-in.
    let r = recv_ctl.stats();
    assert_eq!(
        r.gap_drop, 0,
        "no contiguity gap on a clean single-stream-per-sender fan-in"
    );
    assert_eq!(
        r.stale_epoch_drop, 0,
        "no stale-epoch drop on a blip-free run"
    );
    assert_eq!(
        r.stale_incarnation_drop, 0,
        "no stale-incarnation drop on a blip-free run"
    );
    assert_eq!(
        r.dedup_drop, 0,
        "exactly-once: nothing deduped on a blip-free run"
    );
    assert_eq!(
        r.inbound_dropped_reliable, 0,
        "the sized inbox makes a reliable drop STRUCTURALLY impossible for this backlog"
    );

    // Keep the receiver handle alive to the end so no lane tears down mid-assert.
    let _ = &mut receiver;
}
