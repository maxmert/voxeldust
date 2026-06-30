//! The production multi-peer transport: one QUIC endpoint per node, a static
//! address book (NodeId → SocketAddr), connections dialed ON DEMAND under the
//! cluster's mutual-TLS trust — the real-deployment shape of the SPIKE-0a bridge.
//!
//! Scalability is structural, not aspirational:
//! - **Per-peer send paths**: every destination gets its own bounded queue and its
//!   own writer task. A slow, dead, or unreachable peer back-pressures ONLY traffic
//!   toward itself — never head-of-line blocking across peers (asserted by test).
//! - `Transport::send` is a lock-free bounded enqueue (`try_send`); the only
//!   synchronous failure is `QueueFull(bytes)` per peer.
//! - Hard delivery failures surface in-band as `Inbound::NodeUnreachable` with the
//!   FIFO `MsgId` of the failed send (the R9 cure, identical to the mem transport).
//! - Connections are cached per peer; a broken connection is dropped and re-dialed
//!   on the next frame (fail fast while down, recover without operator action).
//!
//! Peer identity: the frame `from` (`ReliableFrame`/`DatagramFrame`) is sender-asserted and trustworthy
//! ONLY because every link is mutually authenticated against the cluster trust — identity is never
//! derived from source addresses (R2).

use std::collections::BTreeMap;
use std::net::SocketAddr;
use std::sync::Arc;
use std::sync::Mutex;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Duration;

use vd_core::{MsgId, NodeId};
use vd_sim::io::{BoundedInbox, Bytes, Inbound, MsgClass, Reliability, SendError, Transport};

use crate::trust::ClusterTrust;
use crate::{DatagramFrame, OutFrame, ProdIoError, write_reliable_frame};

/// Mesh configuration — ONE struct, no inline literals at use sites.
#[derive(Clone, Debug)]
pub struct MeshConfig {
    pub local: NodeId,
    pub bind: SocketAddr,
    /// The static address book (P1: from config; later phases learn it from the
    /// orchestrator's provisioning flow).
    pub peers: BTreeMap<NodeId, SocketAddr>,
    /// Per-peer outbound queue depth (the back-pressure point).
    pub outbound_capacity: usize,
    /// Receiver inbound bound: a fast sender cannot OOM a slow node — overflow drops
    /// stale UNRELIABLE frames first (latest-wins), reliable traffic last.
    pub inbound_capacity: usize,
    /// Max concurrent ACCEPTED inbound connections — a connection flood cannot
    /// task-flood the node (TRANSPORT-4). Excess incoming connections wait.
    pub max_inbound_connections: usize,
    /// Re-dial backoff bounds for a failed peer lane (no 20 Hz connect-storm against
    /// a dead host — TRANSPORT-3). Backoff doubles from `min` up to `max`.
    pub redial_backoff_min: Duration,
    pub redial_backoff_max: Duration,
}

/// The dest node's `BoundedInbox` capacity for a given per-peer outbound depth — the SINGLE
/// source of truth for the inbox floor (`outbound × 8`, never below 256). It is `const` so a
/// compile-time invariant (the gateway cut-buffer drain-burst bound, `bins` D-8) can assert
/// against it without duplicating the formula.
#[must_use]
pub const fn inbound_capacity_for(outbound_capacity: usize) -> usize {
    let scaled = outbound_capacity.saturating_mul(8);
    if scaled > 256 { scaled } else { 256 }
}

impl MeshConfig {
    /// Sane defaults for the operational fields (capacities/timeouts); the caller
    /// supplies topology (`local`/`bind`/`peers`).
    #[must_use]
    pub fn new(
        local: NodeId,
        bind: SocketAddr,
        peers: BTreeMap<NodeId, SocketAddr>,
        outbound_capacity: usize,
    ) -> MeshConfig {
        MeshConfig {
            local,
            bind,
            peers,
            outbound_capacity,
            inbound_capacity: inbound_capacity_for(outbound_capacity),
            max_inbound_connections: 256,
            redial_backoff_min: Duration::from_millis(50),
            redial_backoff_max: Duration::from_secs(5),
        }
    }
}

/// The shared bounded inbox: reader/writer tasks push, the sim thread drains. The
/// Mutex is only ever held for a queue push/drain — never across an await.
type SharedInbox = Arc<Mutex<BoundedInbox>>;

/// Transport honesty counters — every dropped datagram is COUNTED, never silent
/// (audit GW-1: the design's never-silent rule). Surfaced via [`MeshControl::stats`].
#[derive(Debug, Default)]
pub struct MeshStats {
    /// Unreliable datagrams dropped because the encoded payload exceeded the path
    /// datagram MTU — should be ZERO once snapshots are content-partitioned upstream;
    /// a nonzero value is a partitioner-budget misconfiguration ALERT.
    pub datagrams_dropped_too_large: AtomicU64,
    /// Unreliable datagrams dropped by a full send queue (latest-wins back-pressure).
    pub datagrams_dropped_send: AtomicU64,
    /// RELIABLE inbound events dropped by a full BoundedInbox — the design's explicit
    /// "genuine overload" ALERT case (a dropped grant/attach/revoke). Also warned loudly
    /// at the drop site; this must stay ~0 in any healthy run. NOTE: this is the per-NODE
    /// aggregate derived from the `push_inbox` verdict; the `BoundedInbox::dropped_reliable`
    /// counter (vd-sim) is the per-INBOX tally — the two are distinct surfaces (never
    /// summed): MeshStats for the live transport, the inbox tally for unit tests.
    pub inbound_dropped_reliable: AtomicU64,
    /// Unreliable inbound events evicted/dropped by a full BoundedInbox (latest-wins
    /// back-pressure — by design under load; the counter is the observability surface).
    pub inbound_dropped_unreliable: AtomicU64,
}

/// A snapshot of the mesh counters (loads the atomics).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct MeshStatsSnapshot {
    pub datagrams_dropped_too_large: u64,
    pub datagrams_dropped_send: u64,
    pub inbound_dropped_reliable: u64,
    pub inbound_dropped_unreliable: u64,
}

/// THE inbound-push chokepoint: applies the BoundedInbox overflow policy AND surfaces
/// the result (audit ROB-2 — the drop return exists to be surfaced, never discarded).
/// A RELIABLE drop is the design's explicit overload ALERT: warned + counted. An
/// unreliable drop is by-design latest-wins back-pressure: counted only.
pub(crate) fn push_inbox(inbox: &SharedInbox, stats: &MeshStats, event: Inbound) {
    let dropped = inbox
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
        .push(event);
    match dropped {
        Some(vd_sim::io::InboxDrop::Reliable) => {
            stats
                .inbound_dropped_reliable
                .fetch_add(1, Ordering::Relaxed);
            tracing::warn!(
                "BoundedInbox FULL of reliable events: a RELIABLE inbound was dropped \
                 (genuine overload — raise the inbound capacity or shed load upstream)"
            );
        }
        Some(vd_sim::io::InboxDrop::Unreliable) => {
            stats
                .inbound_dropped_unreliable
                .fetch_add(1, Ordering::Relaxed);
        }
        None => {}
    }
}

struct PeerLane {
    tx: tokio::sync::mpsc::Sender<OutFrame>,
}

/// The sim-thread side: implements [`Transport`] over per-peer bounded queues.
pub struct MeshTransport {
    local: NodeId,
    lanes: BTreeMap<NodeId, PeerLane>,
    inbox: SharedInbox,
    next_msg_id: u64,
}

/// Lifecycle handle: owns the endpoint (dropping closes it).
pub struct MeshControl {
    endpoint: quinn::Endpoint,
    stats: Arc<MeshStats>,
}

impl MeshControl {
    /// The actual bound address (resolves `:0` bindings for tests/config handoff).
    ///
    /// # Errors
    /// Propagates the socket query failure.
    pub fn local_addr(&self) -> Result<SocketAddr, ProdIoError> {
        Ok(self.endpoint.local_addr()?)
    }

    /// Hard-kill this node's endpoint: peers' subsequent sends surface as
    /// `NodeUnreachable` on their side.
    pub fn kill(&self) {
        self.endpoint
            .close(quinn::VarInt::from_u32(1), b"killed by harness");
    }

    /// The transport honesty counters (dropped-datagram metrics; GW-1 never-silent).
    #[must_use]
    pub fn stats(&self) -> MeshStatsSnapshot {
        MeshStatsSnapshot {
            datagrams_dropped_too_large: self
                .stats
                .datagrams_dropped_too_large
                .load(Ordering::Relaxed),
            datagrams_dropped_send: self.stats.datagrams_dropped_send.load(Ordering::Relaxed),
            inbound_dropped_reliable: self.stats.inbound_dropped_reliable.load(Ordering::Relaxed),
            inbound_dropped_unreliable: self
                .stats
                .inbound_dropped_unreliable
                .load(Ordering::Relaxed),
        }
    }
}

/// Spawn one mesh node onto an existing tokio runtime.
///
/// # Errors
/// TLS-config or socket-bind failures.
pub fn spawn_mesh(
    handle: &tokio::runtime::Handle,
    trust: &ClusterTrust,
    cfg: &MeshConfig,
) -> Result<(MeshTransport, MeshControl), ProdIoError> {
    // Shared transport tuning: keepalive holds connections open across idle ticks,
    // a bounded idle timeout reaps a truly-dead half-open connection, and a uni-stream
    // cap bounds per-connection reader tasks (TRANSPORT-3/4/8).
    let transport = {
        let mut t = quinn::TransportConfig::default();
        t.keep_alive_interval(Some(Duration::from_secs(5)));
        t.max_idle_timeout(Some(
            quinn::IdleTimeout::try_from(Duration::from_secs(20)).expect("20s is a valid idle"),
        ));
        t.max_concurrent_uni_streams(quinn::VarInt::from_u32(256));
        Arc::new(t)
    };
    let mut server_config = trust.quinn_server_config()?;
    server_config.transport_config(Arc::clone(&transport));
    let mut client_config = trust.quinn_client_config()?;
    client_config.transport_config(Arc::clone(&transport));

    let endpoint = {
        let _guard = handle.enter();
        let mut endpoint = quinn::Endpoint::server(server_config, cfg.bind)?;
        endpoint.set_default_client_config(client_config);
        endpoint
    };

    let inbox: SharedInbox = Arc::new(Mutex::new(BoundedInbox::new(cfg.inbound_capacity)));
    let stats = Arc::new(MeshStats::default());

    // Accept loop: a Semaphore caps concurrently-served connections so a connection
    // flood cannot task-flood the node (TRANSPORT-4). Each connection serves BOTH
    // reliable uni streams and unreliable datagrams.
    let accept_endpoint = endpoint.clone();
    let accept_inbox = Arc::clone(&inbox);
    let accept_stats = Arc::clone(&stats);
    let permits = Arc::new(tokio::sync::Semaphore::new(
        cfg.max_inbound_connections.max(1),
    ));
    handle.spawn(async move {
        while let Some(incoming) = accept_endpoint.accept().await {
            let Ok(permit) = Arc::clone(&permits).acquire_owned().await else {
                break; // semaphore closed: endpoint shutting down
            };
            let inbox = Arc::clone(&accept_inbox);
            let stats = Arc::clone(&accept_stats);
            tokio::spawn(async move {
                let _permit = permit; // held for the connection's lifetime
                let Ok(connection) = incoming.await else {
                    return; // handshake failed (foreign trust): drop, never serve
                };
                serve_connection(connection, inbox, stats).await;
            });
        }
    });

    // One isolated send lane per peer: bounded queue + dedicated writer task.
    let mut lanes = BTreeMap::new();
    for (&peer, &addr) in &cfg.peers {
        if peer == cfg.local {
            continue; // no self-lane: a node never dials itself
        }
        let (tx, rx) = tokio::sync::mpsc::channel::<OutFrame>(cfg.outbound_capacity);
        handle.spawn(peer_writer(PeerWriter {
            endpoint: endpoint.clone(),
            local: cfg.local,
            addr,
            rx,
            inbox: Arc::clone(&inbox),
            stats: Arc::clone(&stats),
            backoff_min: cfg.redial_backoff_min,
            backoff_max: cfg.redial_backoff_max,
        }));
        lanes.insert(peer, PeerLane { tx });
    }

    Ok((
        MeshTransport {
            local: cfg.local,
            lanes,
            inbox,
            next_msg_id: 0,
        },
        MeshControl { endpoint, stats },
    ))
}

/// Serve one accepted connection: reliable uni streams AND unreliable datagrams,
/// both feeding the shared bounded inbox.
async fn serve_connection(
    connection: quinn::Connection,
    inbox: SharedInbox,
    stats: Arc<MeshStats>,
) {
    let stream_conn = connection.clone();
    let stream_inbox = Arc::clone(&inbox);
    let stream_stats = Arc::clone(&stats);
    // Reliable streams.
    let streams = tokio::spawn(async move {
        while let Ok(recv) = stream_conn.accept_uni().await {
            let inbox = Arc::clone(&stream_inbox);
            let stats = Arc::clone(&stream_stats);
            tokio::spawn(async move {
                crate::read_frames_into(recv, &inbox, &stats).await;
            });
        }
    });
    // Unreliable datagrams on the same connection.
    while let Ok(datagram) = connection.read_datagram().await {
        if let Ok(frame) = postcard::from_bytes::<DatagramFrame>(&datagram) {
            push_inbox(
                &inbox,
                &stats,
                Inbound::Wire {
                    from: frame.from,
                    class: frame.class,
                    bytes: vd_sim::io::bytes(frame.bytes),
                },
            );
        }
        // A malformed datagram is silently dropped: unreliable carriers tolerate it.
    }
    streams.abort();
}

struct PeerWriter {
    endpoint: quinn::Endpoint,
    local: NodeId,
    addr: SocketAddr,
    rx: tokio::sync::mpsc::Receiver<OutFrame>,
    inbox: SharedInbox,
    stats: Arc<MeshStats>,
    backoff_min: Duration,
    backoff_max: Duration,
}

/// One peer's writer: drains its lane in FIFO order, dialing on demand, with
/// exponential re-dial backoff so a dead host is NOT hammered at the tick rate
/// (TRANSPORT-3). Reliable frames ride a persistent uni stream and bounce
/// `NodeUnreachable` on hard failure; unreliable frames ride datagrams (latest-wins:
/// loss is correct, no bounce).
async fn peer_writer(mut w: PeerWriter) {
    let mut connection: Option<quinn::Connection> = None;
    let mut reliable_stream: Option<quinn::SendStream> = None;
    // R2': the per-stream monotone sequence stamped on each reliable frame. (R-2a keeps the
    // pre-existing one-stream-per-peer model; the per-(peer,class) retry-buffer lane FSM replaces this
    // counter in the next sub-step.) Reset to 0 whenever a fresh stream is opened.
    let mut reliable_seq = 0u64;
    let mut backoff = w.backoff_min;
    while let Some(frame) = w.rx.recv().await {
        let sent = write_frame(
            &w.endpoint,
            w.addr,
            &mut connection,
            &mut reliable_stream,
            &mut reliable_seq,
            w.local,
            &frame,
            &w.stats,
        )
        .await;
        match sent {
            Ok(()) => backoff = w.backoff_min,
            Err(()) => {
                connection = None;
                reliable_stream = None;
                // Reliable senders are waiting on an ack path; tell them the peer is
                // unreachable. Unreliable (datagram) loss is silent by design.
                if frame.class.reliability() == Reliability::Reliable {
                    push_inbox(
                        &w.inbox,
                        &w.stats,
                        Inbound::NodeUnreachable {
                            to: frame.to,
                            class: frame.class,
                            undelivered: frame.msg_id,
                        },
                    );
                }
                tokio::time::sleep(backoff).await;
                backoff = (backoff * 2).min(w.backoff_max);
            }
        }
    }
}

/// Ensure a connection (dial on demand) and write one frame on the carrier its
/// class mandates.
#[allow(clippy::too_many_arguments)] // writer state threaded explicitly
async fn write_frame(
    endpoint: &quinn::Endpoint,
    addr: SocketAddr,
    connection: &mut Option<quinn::Connection>,
    reliable_stream: &mut Option<quinn::SendStream>,
    reliable_seq: &mut u64,
    local: NodeId,
    frame: &OutFrame,
    stats: &MeshStats,
) -> Result<(), ()> {
    if connection.is_none() {
        // SNI is pinned to the cluster-trust SAN; identity comes from mTLS, not DNS.
        let conn = endpoint
            .connect(addr, "localhost")
            .map_err(|_| ())?
            .await
            .map_err(|_| ())?;
        *connection = Some(conn);
        *reliable_stream = None;
    }
    let conn = connection.as_ref().ok_or(())?;

    match frame.class.reliability() {
        Reliability::Reliable => {
            // Reliable frames ride a persistent uni stream, framed through the ONE wire::framing
            // home (codec_flags + cap) via the shared writer — never hand-rolled here (HR3). R2': a
            // monotone per-stream `seq` is stamped (incarnation/epoch 0 in R-2a — the receiver's
            // contiguity verdict is R3'); a fresh stream restarts the sequence.
            if reliable_stream.is_none() {
                *reliable_stream = Some(conn.open_uni().await.map_err(|_| ())?);
                *reliable_seq = 0;
            }
            let send = reliable_stream.as_mut().ok_or(())?;
            let this_seq = *reliable_seq;
            *reliable_seq += 1;
            write_reliable_frame(send, local, frame.class, 0, 0, this_seq, &frame.bytes).await
        }
        Reliability::Unreliable => {
            // Datagrams are message-bounded (QUIC-delimited, NO stream framing — codec_flags is a
            // stream concept) and carry the BARE DatagramFrame (R1' hot/cold split — no reliability
            // metadata on the 20Hz path). TooLarge is a loud failure of the caller's framing.
            let payload = postcard::to_allocvec(&DatagramFrame {
                from: local,
                class: frame.class,
                bytes: frame.bytes.to_vec(),
            })
            .map_err(|_| ())?;
            if conn
                .max_datagram_size()
                .is_none_or(|max| payload.len() > max)
            {
                // COUNTED, never silent (GW-1): snapshots are content-partitioned
                // upstream, so a too-large datagram here is a budget misconfiguration.
                stats
                    .datagrams_dropped_too_large
                    .fetch_add(1, Ordering::Relaxed);
                tracing::warn!(
                    "datagram payload {} exceeds the path MTU budget; dropped (counted)",
                    payload.len()
                );
                return Ok(()); // dropped, but the connection is healthy
            }
            // A full send queue drops the datagram (latest-wins); only a dead
            // connection is an Err that triggers re-dial.
            match conn.send_datagram(payload.into()) {
                Ok(()) => Ok(()),
                Err(quinn::SendDatagramError::ConnectionLost(_)) => Err(()),
                Err(_) => {
                    // Unsupported/too-large/queue-full: drop (latest-wins), stay up.
                    stats.datagrams_dropped_send.fetch_add(1, Ordering::Relaxed);
                    Ok(())
                }
            }
        }
    }
}

impl Transport for MeshTransport {
    fn send(&mut self, to: NodeId, class: MsgClass, bytes: Bytes) -> Result<MsgId, SendError> {
        let Some(lane) = self.lanes.get(&to) else {
            // An unknown destination is permanent back-pressure: the payload is
            // returned, never silently dropped (the address book is config, and a
            // misconfigured route must be loud at the caller).
            return Err(SendError::QueueFull(bytes));
        };
        let msg_id = MsgId(self.next_msg_id);
        match lane.tx.try_send(OutFrame {
            to,
            class,
            bytes,
            msg_id,
        }) {
            Ok(()) => {
                self.next_msg_id += 1;
                Ok(msg_id)
            }
            Err(
                tokio::sync::mpsc::error::TrySendError::Full(frame)
                | tokio::sync::mpsc::error::TrySendError::Closed(frame),
            ) => Err(SendError::QueueFull(frame.bytes)),
        }
    }

    fn drain_inbound(&mut self) -> Vec<Inbound> {
        self.inbox
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .drain()
    }

    fn local_id(&self) -> NodeId {
        self.local
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::{Duration, Instant};

    const DEADLINE: Duration = Duration::from_secs(20);

    fn runtime() -> tokio::runtime::Runtime {
        tokio::runtime::Builder::new_multi_thread()
            .worker_threads(2)
            .enable_all()
            .build()
            .expect("tokio runtime")
    }

    /// Bind ephemeral ports for `n` nodes and build the shared address book.
    fn cluster(
        handle: &tokio::runtime::Handle,
        trust: &ClusterTrust,
        n: u64,
        capacity: usize,
    ) -> (Vec<MeshTransport>, Vec<MeshControl>) {
        // Two passes: bind everyone on :0 first, then a second mesh per node would
        // re-bind — instead bind once and patch the shared book afterward is not
        // possible with quinn. So: bind sequentially, accumulating the book; lanes
        // are created from the FULL book, which requires knowing addresses up
        // front. Solution: pre-bind plain UDP sockets to reserve ports.
        let mut book = BTreeMap::new();
        let mut reserved = Vec::new();
        for i in 0..n {
            let socket = std::net::UdpSocket::bind("127.0.0.1:0").expect("reserve port");
            let addr = socket.local_addr().expect("addr");
            book.insert(NodeId(i + 1), addr);
            reserved.push((NodeId(i + 1), socket));
        }
        let mut transports = Vec::new();
        let mut controls = Vec::new();
        for (id, socket) in reserved {
            let addr = socket.local_addr().expect("addr");
            drop(socket); // release the port for quinn to take
            let (t, c) = spawn_mesh(
                handle,
                trust,
                &MeshConfig::new(id, addr, book.clone(), capacity),
            )
            .expect("mesh node");
            transports.push(t);
            controls.push(c);
        }
        (transports, controls)
    }

    fn wait_for(
        t: &mut MeshTransport,
        mut predicate: impl FnMut(&[Inbound]) -> bool,
    ) -> Vec<Inbound> {
        let started = Instant::now();
        let mut got = Vec::new();
        while !predicate(&got) {
            got.extend(t.drain_inbound());
            std::thread::sleep(Duration::from_millis(2));
            assert!(started.elapsed() < DEADLINE, "timed out; got {got:?}");
        }
        got
    }

    #[test]
    fn three_node_mesh_full_exchange_with_verified_sender_identity() {
        let rt = runtime();
        let trust = ClusterTrust::generate("vd-mesh-test").expect("trust");
        let (mut nodes, _controls) = cluster(rt.handle(), &trust, 3, 64);
        // Every node sends one tagged message to every other node.
        let ids: Vec<NodeId> = nodes.iter().map(MeshTransport::local_id).collect();
        for sender in &mut nodes {
            let from = sender.local_id();
            for &to in &ids {
                if to == from {
                    continue;
                }
                let tag = vec![from.0 as u8, to.0 as u8];
                sender
                    .send(to, MsgClass::Control, vd_sim::io::bytes(tag))
                    .expect("accepted");
            }
        }
        for node in &mut nodes {
            let me = node.local_id();
            let got = wait_for(node, |g| g.len() >= 2);
            assert_eq!(got.len(), 2);
            for msg in got {
                let Inbound::Wire { from, bytes, .. } = msg else {
                    panic!("wire expected, got {msg:?}");
                };
                assert_eq!(
                    bytes.to_vec(),
                    vec![from.0 as u8, me.0 as u8],
                    "the in-frame sender identity matches the actual sender"
                );
            }
        }
    }

    #[test]
    fn a_dead_peer_never_blocks_traffic_to_live_peers() {
        let rt = runtime();
        let trust = ClusterTrust::generate("vd-mesh-test").expect("trust");
        let (mut nodes, controls) = cluster(rt.handle(), &trust, 3, 4);
        // Kill node 3; node 1 saturates the dead lane THEN talks to node 2.
        controls[2].kill();
        std::thread::sleep(Duration::from_millis(50));
        let dead = NodeId(3);
        let live = NodeId(2);
        for n in 0..16u8 {
            // RELIABLE sends to the dead peer bounce as NodeUnreachable.
            let _ = nodes[0].send(dead, MsgClass::Saga, vec![n].into());
        }
        nodes[0]
            .send(live, MsgClass::Control, vec![42].into())
            .expect("the live lane is unaffected by the dead one");
        let got = wait_for(&mut nodes[1], |g| !g.is_empty());
        let Inbound::Wire { from, bytes, .. } = &got[0] else {
            panic!("wire expected");
        };
        assert_eq!((*from, bytes.to_vec()), (NodeId(1), vec![42]));
        // And the dead lane surfaces unreachable notices for RELIABLE sends.
        let notices = wait_for(&mut nodes[0], |g| {
            g.iter()
                .filter(|m| matches!(m, Inbound::NodeUnreachable { .. }))
                .count()
                >= 1
        });
        let unreachable_to: Vec<NodeId> = notices
            .iter()
            .filter_map(|m| match m {
                Inbound::NodeUnreachable { to, .. } => Some(*to),
                Inbound::Wire { .. } => None,
            })
            .collect();
        assert!(unreachable_to.iter().all(|to| *to == dead));
    }

    #[test]
    fn unreliable_sends_to_a_dead_peer_are_dropped_silently_no_bounce() {
        // Datagrams are latest-wins: their loss is correct, never a NodeUnreachable.
        let rt = runtime();
        let trust = ClusterTrust::generate("vd-mesh-test").expect("trust");
        let (mut nodes, controls) = cluster(rt.handle(), &trust, 2, 8);
        controls[1].kill();
        std::thread::sleep(Duration::from_millis(50));
        for n in 0..8u8 {
            let _ = nodes[0].send(NodeId(2), MsgClass::Snapshot, vec![n].into());
        }
        // Give the writer time to attempt + fail + back off, then confirm NO bounce.
        std::thread::sleep(Duration::from_millis(300));
        let drained = nodes[0].drain_inbound();
        assert!(
            !drained
                .iter()
                .any(|m| matches!(m, Inbound::NodeUnreachable { .. })),
            "unreliable loss must not bounce: {drained:?}"
        );
    }

    #[test]
    fn per_peer_backpressure_returns_the_payload() {
        let rt = runtime();
        let trust = ClusterTrust::generate("vd-mesh-test").expect("trust");
        // Capacity 2 toward a peer that never drains (killed before traffic).
        let (mut nodes, controls) = cluster(rt.handle(), &trust, 2, 2);
        controls[1].kill();
        std::thread::sleep(Duration::from_millis(50));
        // The writer task pulls some frames into its in-flight dial attempt; the
        // queue itself holds `capacity`. Flood until refusal and check the refusal.
        let mut refused = None;
        for n in 0..64u8 {
            match nodes[0].send(NodeId(2), MsgClass::Input, vec![n].into()) {
                Ok(_) => {}
                Err(SendError::QueueFull(bytes)) => {
                    refused = Some((n, bytes));
                    break;
                }
            }
        }
        let (n, bytes) = refused.expect("the bounded lane eventually refuses");
        assert_eq!(
            bytes.to_vec(),
            vec![n],
            "the refused payload is returned intact"
        );
    }

    #[test]
    fn an_oversize_datagram_is_dropped_but_counted_never_silent() {
        // GW-1: a too-large unreliable payload (past the path MTU) is dropped — but
        // COUNTED on the transport's honesty metric, never silently lost.
        let rt = runtime();
        let trust = ClusterTrust::generate("vd-mesh-test").expect("trust");
        let (mut nodes, controls) = cluster(rt.handle(), &trust, 2, 64);
        // A payload far above any datagram MTU (well under the wire::framing stream cap so the
        // length guard doesn't reject it first).
        let huge = vd_sim::io::bytes(vec![7u8; 64 * 1024]);
        nodes[0]
            .send(NodeId(2), MsgClass::Snapshot, huge)
            .expect("enqueued");
        let started = Instant::now();
        loop {
            let dropped = controls[0].stats().datagrams_dropped_too_large;
            if dropped >= 1 {
                break;
            }
            assert!(
                started.elapsed() < DEADLINE,
                "the oversize datagram was never counted"
            );
            std::thread::sleep(Duration::from_millis(5));
        }
    }

    /// CA-2 characterization: the mesh routes ONLY to statically-booked peers — an
    /// id absent from the address book is permanent, loud back-pressure. This is the
    /// limitation CA-1 (reply-on-connection, see `ca1_reply_on_connection…` below)
    /// will lift; until then the dev launcher pre-seeds client addresses into the
    /// gateway book (`client_book` in `vd-devcluster`) to work around it.
    #[test]
    fn unknown_destinations_are_loud_backpressure() {
        let rt = runtime();
        let trust = ClusterTrust::generate("vd-mesh-test").expect("trust");
        let (mut nodes, _controls) = cluster(rt.handle(), &trust, 2, 4);
        let err = nodes[0]
            .send(NodeId(99), MsgClass::Control, vec![7].into())
            .expect_err("not in the address book");
        assert_eq!(err, SendError::QueueFull(vec![7].into()));
    }

    /// RED GUARD for CA-1 (deferred to M3). Today a peer is reachable only if it is in
    /// the sender's static book (the CA-2 characterization above). The dev launcher
    /// works around this by pre-seeding EVERY client's address into the gateway book
    /// — the crutch this milestone fences (`client_book` in `vd-devcluster`). When
    /// CA-1 lands, a peer that DIALS IN becomes replyable without being booked first;
    /// this asserts exactly that, and fails by construction today (hence `#[ignore]`).
    #[test]
    #[ignore = "M3: reply-on-connection (CA-1) not yet implemented"]
    fn ca1_reply_on_connection_reaches_a_peer_not_in_the_book() {
        let rt = runtime();
        let trust = ClusterTrust::generate("vd-mesh-test").expect("trust");
        let (mut nodes, _controls) = cluster(rt.handle(), &trust, 2, 4);
        // The target behavior: reaching an UNBOOKED id succeeds once an inbound dial
        // has taught the mesh that peer's return address. Today this is QueueFull.
        let reply = nodes[0].send(NodeId(99), MsgClass::Control, vec![1].into());
        assert!(
            reply.is_ok(),
            "CA-1: an inbound-dialed peer must be replyable without pre-booking",
        );
    }

    #[test]
    fn no_self_lane_exists() {
        let rt = runtime();
        let trust = ClusterTrust::generate("vd-mesh-test").expect("trust");
        let (mut nodes, _controls) = cluster(rt.handle(), &trust, 2, 4);
        let me = nodes[0].local_id();
        let err = nodes[0]
            .send(me, MsgClass::Control, vec![1].into())
            .expect_err("a node never dials itself");
        assert_eq!(err, SendError::QueueFull(vec![1].into()));
    }

    /// The scalability volume bound: 4 nodes, every pair exchanging a sustained
    /// burst concurrently. Catches serialization collapse across lanes (the
    /// property), not raw throughput. RELIABLE class so no-loss is a valid assertion.
    #[test]
    fn mesh_volume_all_pairs_burst() {
        const BURST: usize = 200;
        let rt = runtime();
        let trust = ClusterTrust::generate("vd-mesh-test").expect("trust");
        let (mut nodes, _controls) = cluster(rt.handle(), &trust, 4, 256);
        let ids: Vec<NodeId> = nodes.iter().map(MeshTransport::local_id).collect();
        let started = Instant::now();
        for round in 0..BURST {
            for sender in &mut nodes {
                let from = sender.local_id();
                for &to in &ids {
                    if to == from {
                        continue;
                    }
                    // Sustained send with per-peer back-pressure tolerated (retry).
                    let mut payload = vd_sim::io::bytes(vec![(round % 251) as u8]);
                    loop {
                        match sender.send(to, MsgClass::Saga, payload) {
                            Ok(_) => break,
                            Err(SendError::QueueFull(returned)) => {
                                payload = returned;
                                std::thread::sleep(Duration::from_micros(200));
                            }
                        }
                    }
                }
            }
        }
        // Every node receives 3 peers * BURST messages (reliable: no loss).
        for node in &mut nodes {
            let got = wait_for(node, |g| {
                g.iter()
                    .filter(|m| matches!(m, Inbound::Wire { .. }))
                    .count()
                    >= 3 * BURST
            });
            assert_eq!(got.len(), 3 * BURST, "no loss, no duplication on QUIC");
        }
        assert!(
            started.elapsed() < DEADLINE,
            "all-pairs burst collapsed: {:?}",
            started.elapsed()
        );
    }

    /// Unreliable datagrams deliver best-effort: on localhost a steady (non-flooding)
    /// snapshot stream arrives, exercising the end-to-end send_datagram/read_datagram
    /// path with newest-wins semantics.
    #[test]
    fn unreliable_datagrams_deliver_best_effort() {
        let rt = runtime();
        let trust = ClusterTrust::generate("vd-mesh-test").expect("trust");
        let (mut nodes, _controls) = cluster(rt.handle(), &trust, 2, 64);
        // Pace the sends so the datagram send-queue never saturates: every one lands.
        for n in 0..20u8 {
            nodes[0]
                .send(NodeId(2), MsgClass::Snapshot, vec![n].into())
                .expect("enqueued");
            std::thread::sleep(Duration::from_millis(5));
        }
        let got = wait_for(&mut nodes[1], |g| {
            g.iter()
                .filter(|m| matches!(m, Inbound::Wire { .. }))
                .count()
                >= 10
        });
        // At least half arrived (best-effort, no flooding) and all are snapshots.
        assert!(got.len() >= 10, "datagrams delivered: {}", got.len());
        for msg in &got {
            assert!(
                matches!(
                    msg,
                    Inbound::Wire {
                        class: MsgClass::Snapshot,
                        ..
                    }
                ),
                "snapshot datagram: {msg:?}"
            );
        }
    }
}
