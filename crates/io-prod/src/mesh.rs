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
//! Peer identity: `WireFrame::from` is sender-asserted and trustworthy ONLY because
//! every link is mutually authenticated against the cluster trust — identity is
//! never derived from source addresses (R2).

use std::collections::BTreeMap;
use std::net::SocketAddr;

use crossbeam_channel::{Receiver, Sender, unbounded};
use vd_core::{MsgId, NodeId};
use vd_sim::io::{Bytes, Inbound, MsgClass, SendError, Transport};

use crate::trust::ClusterTrust;
use crate::{MAX_FRAME_BYTES, ProdIoError, WireFrame};

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
}

struct PeerLane {
    tx: tokio::sync::mpsc::Sender<OutFrame>,
}

struct OutFrame {
    to: NodeId,
    class: MsgClass,
    bytes: Bytes,
    msg_id: MsgId,
}

/// The sim-thread side: implements [`Transport`] over per-peer bounded queues.
pub struct MeshTransport {
    local: NodeId,
    lanes: BTreeMap<NodeId, PeerLane>,
    inbound_rx: Receiver<Inbound>,
    next_msg_id: u64,
}

/// Lifecycle handle: owns the endpoint (dropping closes it).
pub struct MeshControl {
    endpoint: quinn::Endpoint,
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
    let server_config = trust.quinn_server_config()?;
    let client_config = trust.quinn_client_config()?;

    let endpoint = {
        let _guard = handle.enter();
        let mut endpoint = quinn::Endpoint::server(server_config, cfg.bind)?;
        endpoint.set_default_client_config(client_config);
        endpoint
    };

    let (inbound_tx, inbound_rx) = unbounded::<Inbound>();

    // Accept loop: every inbound connection gets a reader task per uni stream.
    let accept_endpoint = endpoint.clone();
    let accept_inbound = inbound_tx.clone();
    handle.spawn(async move {
        while let Some(incoming) = accept_endpoint.accept().await {
            let Ok(connection) = incoming.await else {
                continue; // handshake failed (foreign trust): drop, never serve
            };
            let inbound = accept_inbound.clone();
            tokio::spawn(async move {
                while let Ok(recv) = connection.accept_uni().await {
                    let inbound = inbound.clone();
                    tokio::spawn(async move {
                        crate::read_frames(recv, inbound).await;
                    });
                }
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
        handle.spawn(peer_writer(
            endpoint.clone(),
            cfg.local,
            addr,
            rx,
            inbound_tx.clone(),
        ));
        lanes.insert(peer, PeerLane { tx });
    }

    Ok((
        MeshTransport {
            local: cfg.local,
            lanes,
            inbound_rx,
            next_msg_id: 0,
        },
        MeshControl { endpoint },
    ))
}

/// One peer's writer: drains its lane in FIFO order, dialing on demand. A hard
/// failure bounces the frame as `NodeUnreachable` and drops the cached connection
/// so the next frame re-dials (fail fast while down, recover automatically).
async fn peer_writer(
    endpoint: quinn::Endpoint,
    local: NodeId,
    addr: SocketAddr,
    mut rx: tokio::sync::mpsc::Receiver<OutFrame>,
    inbound: Sender<Inbound>,
) {
    let mut stream: Option<quinn::SendStream> = None;
    while let Some(frame) = rx.recv().await {
        let wrote = write_to_peer(&endpoint, addr, &mut stream, local, &frame).await;
        if wrote.is_err() {
            stream = None;
            let _ = inbound.send(Inbound::NodeUnreachable {
                to: frame.to,
                class: frame.class,
                undelivered: frame.msg_id,
            });
        }
    }
}

/// Ensure a stream to the peer and write one frame (dial + open on demand).
async fn write_to_peer(
    endpoint: &quinn::Endpoint,
    addr: SocketAddr,
    stream: &mut Option<quinn::SendStream>,
    local: NodeId,
    frame: &OutFrame,
) -> Result<(), ()> {
    if stream.is_none() {
        // SNI is pinned to the cluster-trust SAN; identity comes from mTLS, not DNS.
        let connection = endpoint
            .connect(addr, "localhost")
            .map_err(|_| ())?
            .await
            .map_err(|_| ())?;
        *stream = Some(connection.open_uni().await.map_err(|_| ())?);
    }
    let send = stream.as_mut().ok_or(())?;
    let payload = postcard::to_allocvec(&WireFrame {
        from: local,
        class: frame.class,
        bytes: frame.bytes.clone(),
    })
    .map_err(|_| ())?;
    let len = u32::try_from(payload.len()).map_err(|_| ())?;
    if len > MAX_FRAME_BYTES {
        return Err(());
    }
    send.write_all(&len.to_be_bytes()).await.map_err(|_| ())?;
    send.write_all(&payload).await.map_err(|_| ())?;
    Ok(())
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
        self.inbound_rx.try_iter().collect()
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
                &MeshConfig {
                    local: id,
                    bind: addr,
                    peers: book.clone(),
                    outbound_capacity: capacity,
                },
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
                sender.send(to, MsgClass::Control, tag).expect("accepted");
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
                    bytes,
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
            // Beyond the lane capacity these back-pressure — per-peer, loudly.
            let _ = nodes[0].send(dead, MsgClass::Input, vec![n]);
        }
        nodes[0]
            .send(live, MsgClass::Control, vec![42])
            .expect("the live lane is unaffected by the dead one");
        let got = wait_for(&mut nodes[1], |g| !g.is_empty());
        let Inbound::Wire { from, bytes, .. } = &got[0] else {
            panic!("wire expected");
        };
        assert_eq!((*from, bytes.clone()), (NodeId(1), vec![42]));
        // And the dead lane surfaces unreachable notices (accepted sends only).
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
            match nodes[0].send(NodeId(2), MsgClass::Input, vec![n]) {
                Ok(_) => {}
                Err(SendError::QueueFull(bytes)) => {
                    refused = Some((n, bytes));
                    break;
                }
            }
        }
        let (n, bytes) = refused.expect("the bounded lane eventually refuses");
        assert_eq!(bytes, vec![n], "the refused payload is returned intact");
    }

    #[test]
    fn unknown_destinations_are_loud_backpressure() {
        let rt = runtime();
        let trust = ClusterTrust::generate("vd-mesh-test").expect("trust");
        let (mut nodes, _controls) = cluster(rt.handle(), &trust, 2, 4);
        let err = nodes[0]
            .send(NodeId(99), MsgClass::Control, vec![7])
            .expect_err("not in the address book");
        assert_eq!(err, SendError::QueueFull(vec![7]));
    }

    #[test]
    fn no_self_lane_exists() {
        let rt = runtime();
        let trust = ClusterTrust::generate("vd-mesh-test").expect("trust");
        let (mut nodes, _controls) = cluster(rt.handle(), &trust, 2, 4);
        let me = nodes[0].local_id();
        let err = nodes[0]
            .send(me, MsgClass::Control, vec![1])
            .expect_err("a node never dials itself");
        assert_eq!(err, SendError::QueueFull(vec![1]));
    }

    /// The scalability volume bound: 4 nodes, every pair exchanging a sustained
    /// burst concurrently. Catches serialization collapse across lanes (the
    /// property), not raw throughput.
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
                    let mut payload = vec![(round % 251) as u8];
                    loop {
                        match sender.send(to, MsgClass::Snapshot, payload) {
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
        // Every node receives 3 peers * BURST messages.
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
}
