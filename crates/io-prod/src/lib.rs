//! # vd-io-prod — production `ShardIo` implementations
//!
//! The async world stops HERE. Simulation code (`vd-sim`/`vd-node`) is synchronous and
//! reaches the network only through the [`vd_sim::io::Transport`] seam; this crate
//! implements that seam over quinn QUIC with the SPIKE-0a threading model
//! (`docs/design/test_harness.md` §2):
//!
//! ```text
//! ┌─ tokio reader task ─┐  crossbeam   ┌─ SIM THREAD (no tokio) ─┐  crossbeam  ┌─ writer thread ─┐
//! │ quinn recv stream → │ ─inbound──►  │ step_tick(): drain,     │ ─outbound─► │ drains, quinn   │
//! │ decode WireFrame    │              │ react, enqueue sends    │  (BOUNDED)  │ send; on failure│
//! └─────────────────────┘              │ NEVER awaits            │             │ → enqueue       │
//!                                      └─────────────────────────┘             │ NodeUnreachable │
//!                                                                              │ into INBOUND    │
//!                                                                              └─────────────────┘
//! ```
//!
//! - `Transport::send` is a non-blocking enqueue into the BOUNDED outbound channel:
//!   `Err(QueueFull)` is the only synchronous failure (back-pressure).
//! - A hard write failure surfaces on a later drain as `Inbound::NodeUnreachable`,
//!   carrying the FIFO `MsgId` of the failed send.
//! - TLS: the [`trust::ClusterTrust`] day-1 PSK model — one cluster bundle, MUTUAL
//!   TLS in both directions, NEVER certificate-verification skipping (which was
//!   audit finding R7's enabler). Per-node certificates from a real CA land at P3+
//!   behind the same seam.
//! - Peer identity in `WireFrame::from` is sender-asserted; it is trustworthy only
//!   because the channel is mutually authenticated — the design forbids ever
//!   deriving identity from source addresses (R2).
//!
//! Coverage: Tier-B — exercised by the process tier; ratcheted floor, never 100% (HR5).

pub mod admin;
pub mod mesh;
pub mod runtime;
pub mod store;
pub mod trust;

use std::net::SocketAddr;

use crossbeam_channel::{Receiver, Sender, TrySendError, bounded, unbounded};
use serde::{Deserialize, Serialize};
use vd_core::{MsgId, NodeId};
use vd_sim::io::{Bytes, Inbound, MsgClass, SendError, Transport};

use crate::trust::{ClusterTrust, TrustError};

/// The RELIABLE on-stream frame payload: a postcard of this struct, carried inside the `wire::framing`
/// stream frame (`[u32_be total_len][u8 codec_flags][payload]`). io-prod routes ALL stream framing
/// through `wire::framing` (the ONE codec/framing home — HR3 + the `codec_flags` reserved-bit
/// forward-compat path); the cap is `wire::framing::MAX_STREAM_FRAME_BYTES`.
///
/// R1' (the redelivering-transport hot/cold split): this type rides the RELIABLE uni-stream path ONLY.
/// The at-least-once metadata (incarnation/epoch/seq) is added to THIS type in R2'/R3' — it can grow
/// without touching the unreliable hot path because that path now has its own [`DatagramFrame`].
#[derive(Debug, Serialize, Deserialize)]
pub(crate) struct WireFrame {
    from: NodeId,
    class: MsgClass,
    /// The on-wire payload is a plain `Vec<u8>` (the seam's `Bytes = Arc<[u8]>` is
    /// an in-process sharing optimization; it converts at the serialize boundary).
    bytes: Vec<u8>,
}

/// The UNRELIABLE datagram payload (R1'): a BARE `{from, class, bytes}` carried QUIC-datagram-delimited
/// (NO `wire::framing` stream framing — `codec_flags` is a stream concept). Split from [`WireFrame`] so
/// the 20Hz latest-wins snapshot/input/ghost-delta hot path carries ZERO reliability metadata and never
/// touches the (future) dedup state — the PvP hot path stays byte-for-byte what it is today (R2). It is
/// CURRENTLY byte-identical to `WireFrame` (a `serialize_bytes_match` test pins that), and stays bare as
/// `WireFrame` grows its at-least-once fields in R2'/R3'.
#[derive(Debug, Serialize, Deserialize)]
pub(crate) struct DatagramFrame {
    from: NodeId,
    class: MsgClass,
    bytes: Vec<u8>,
}

/// The sim-thread→writer queue item for the loopback bridge. DRY pin (audit `wf_2c963246`): byte-identical
/// to `mesh.rs`'s `OutFrame` — unify both into ONE crate-root struct in R2'/R3' (when the at-least-once
/// seq/incarnation/epoch metadata lands on the reliable path), so the two transports cannot drift.
#[derive(Debug)]
struct OutboundFrame {
    to: NodeId,
    class: MsgClass,
    bytes: Bytes,
    msg_id: MsgId,
}

#[derive(Debug, thiserror::Error)]
pub enum ProdIoError {
    #[error("trust: {0}")]
    Trust(#[from] TrustError),
    #[error("quinn connect: {0}")]
    Connect(String),
    #[error("io: {0}")]
    Io(#[from] std::io::Error),
}

/// Sim-thread side of the bridge. Implements [`Transport`] over the crossbeam queues.
pub struct ProdTransport {
    local: NodeId,
    outbound_tx: Sender<OutboundFrame>,
    inbound_rx: Receiver<Inbound>,
    next_msg_id: u64,
}

impl Transport for ProdTransport {
    fn send(&mut self, to: NodeId, class: MsgClass, bytes: Bytes) -> Result<MsgId, SendError> {
        let msg_id = MsgId(self.next_msg_id);
        match self.outbound_tx.try_send(OutboundFrame {
            to,
            class,
            bytes,
            msg_id,
        }) {
            Ok(()) => {
                self.next_msg_id += 1;
                Ok(msg_id)
            }
            // A full queue is back-pressure. A disconnected bridge (writer thread gone,
            // only during shutdown) behaves as permanently-full: the node is being torn
            // down and no new sends can ever be accepted. The payload is returned in
            // both cases (a refusal is never a loss).
            Err(TrySendError::Full(frame) | TrySendError::Disconnected(frame)) => {
                Err(SendError::QueueFull(frame.bytes))
            }
        }
    }

    fn drain_inbound(&mut self) -> Vec<Inbound> {
        self.inbound_rx.try_iter().collect()
    }

    fn local_id(&self) -> NodeId {
        self.local
    }
}

/// Control handle for one node's bridge (test/process lifecycle: pausing the writer for
/// deterministic back-pressure tests, killing the endpoint for unreachability tests).
pub struct BridgeControl {
    endpoint: quinn::Endpoint,
    writer_gate: Option<Sender<()>>,
}

impl BridgeControl {
    /// Release a writer that was started paused (deterministic back-pressure testing:
    /// flood the bounded queue first, then let the writer drain).
    pub fn release_writer(&mut self) {
        if let Some(gate) = self.writer_gate.take() {
            let _ = gate.send(());
        }
    }

    /// Hard-kill this node's endpoint: peers' subsequent sends surface as
    /// `NodeUnreachable` on their side.
    pub fn kill(&self) {
        self.endpoint
            .close(quinn::VarInt::from_u32(1), b"killed by harness");
    }
}

/// A connected loopback pair of nodes, each with its own bridge. Owns the tokio
/// runtime; dropping the pair tears everything down.
pub struct LoopbackPair {
    pub a: ProdTransport,
    pub b: ProdTransport,
    pub a_ctl: BridgeControl,
    pub b_ctl: BridgeControl,
    _runtime: tokio::runtime::Runtime,
}

/// Build two nodes connected over real QUIC on 127.0.0.1, under the cluster's
/// mutual-TLS trust bundle.
///
/// `outbound_capacity` bounds each node's outbound queue (the back-pressure point).
/// `paused_writers` starts both writer threads gated, so a test can flood the queue
/// deterministically before any drain happens.
pub fn loopback_pair(
    trust: &ClusterTrust,
    a_id: NodeId,
    b_id: NodeId,
    outbound_capacity: usize,
    paused_writers: bool,
) -> Result<LoopbackPair, ProdIoError> {
    let runtime = tokio::runtime::Builder::new_multi_thread()
        .worker_threads(2)
        .enable_all()
        .build()?;
    let handle = runtime.handle().clone();

    let server_config = trust.quinn_server_config()?;
    let client_config = trust.quinn_client_config()?;

    let bind: SocketAddr = "127.0.0.1:0"
        .parse()
        .map_err(|_| ProdIoError::Connect("bad bind addr".into()))?;

    let (conn_a, conn_b, ep_a, ep_b) = handle.block_on(async move {
        let mut ep_b = quinn::Endpoint::server(server_config, bind)?;
        ep_b.set_default_client_config(client_config.clone());
        let b_addr = ep_b.local_addr()?;

        let mut ep_a = quinn::Endpoint::client(bind)?;
        ep_a.set_default_client_config(client_config);

        // Drive the server-side handshake on its own task: the client's `connect`
        // cannot complete until the server's `Incoming` is awaited.
        let accept_ep = ep_b.clone();
        let server_task = tokio::spawn(async move {
            let incoming = accept_ep.accept().await?;
            incoming.await.ok()
        });
        let conn_a = ep_a
            .connect(b_addr, "localhost")
            .map_err(|e| ProdIoError::Connect(e.to_string()))?
            .await
            .map_err(|e| ProdIoError::Connect(e.to_string()))?;
        let conn_b = server_task
            .await
            .map_err(|e| ProdIoError::Connect(e.to_string()))?
            .ok_or_else(|| ProdIoError::Connect("endpoint closed while accepting".into()))?;
        Ok::<_, ProdIoError>((conn_a, conn_b, ep_a, ep_b))
    })?;

    let (a, a_ctl) = spawn_bridge(
        &handle,
        a_id,
        conn_a,
        ep_a,
        outbound_capacity,
        paused_writers,
    );
    let (b, b_ctl) = spawn_bridge(
        &handle,
        b_id,
        conn_b,
        ep_b,
        outbound_capacity,
        paused_writers,
    );

    Ok(LoopbackPair {
        a,
        b,
        a_ctl,
        b_ctl,
        _runtime: runtime,
    })
}

/// Wire one node's reader task + writer thread around an established connection.
fn spawn_bridge(
    handle: &tokio::runtime::Handle,
    local: NodeId,
    conn: quinn::Connection,
    endpoint: quinn::Endpoint,
    outbound_capacity: usize,
    paused_writer: bool,
) -> (ProdTransport, BridgeControl) {
    let (outbound_tx, outbound_rx) = bounded::<OutboundFrame>(outbound_capacity);
    let (inbound_tx, inbound_rx) = unbounded::<Inbound>();

    // Reader task: accept the peer's uni stream, decode frames, feed the inbound queue.
    let reader_inbound = inbound_tx.clone();
    let reader_conn = conn.clone();
    handle.spawn(async move {
        let Ok(recv) = reader_conn.accept_uni().await else {
            return;
        };
        read_frames(recv, reader_inbound).await;
    });

    // Writer thread: the ONLY place outbound frames touch the network. A hard failure
    // re-enters the inbound queue as NodeUnreachable — the async half of the R9 cure.
    let writer_gate = if paused_writer {
        let (gate_tx, gate_rx) = bounded::<()>(1);
        spawn_writer(
            handle.clone(),
            local,
            conn,
            outbound_rx,
            inbound_tx,
            Some(gate_rx),
        );
        Some(gate_tx)
    } else {
        spawn_writer(handle.clone(), local, conn, outbound_rx, inbound_tx, None);
        None
    };

    (
        ProdTransport {
            local,
            outbound_tx,
            inbound_rx,
            next_msg_id: 0,
        },
        BridgeControl {
            endpoint,
            writer_gate,
        },
    )
}

fn spawn_writer(
    handle: tokio::runtime::Handle,
    local: NodeId,
    conn: quinn::Connection,
    outbound_rx: Receiver<OutboundFrame>,
    inbound_tx: Sender<Inbound>,
    gate: Option<Receiver<()>>,
) {
    std::thread::Builder::new()
        .name(format!("vd-writer-{}", local.0))
        .spawn(move || {
            if let Some(gate) = gate {
                // Started paused; released by BridgeControl::release_writer.
                let _ = gate.recv();
            }
            let mut stream: Option<quinn::SendStream> = None;
            while let Ok(frame) = outbound_rx.recv() {
                let wrote = handle.block_on(write_frame(&conn, &mut stream, local, &frame));
                if wrote.is_err() {
                    // Surface the failure in-band; drop the broken stream so the next
                    // frame re-attempts (and fails fast while the connection is dead).
                    stream = None;
                    let _ = inbound_tx.send(Inbound::NodeUnreachable {
                        to: frame.to,
                        class: frame.class,
                        undelivered: frame.msg_id,
                    });
                }
            }
        })
        .expect("spawn writer thread");
}

/// Frame + write ONE `WireFrame` on a reliable uni stream — the SHARED stream-write for BOTH the
/// bridge and the mesh transport, routed through `wire::framing::frame_payload` so the
/// `codec_flags` discipline + cap live in ONE home (HR3). The framed bytes already include the
/// length prefix, so it is a single `write_all`.
pub(crate) async fn write_wireframe(
    send: &mut quinn::SendStream,
    local: NodeId,
    class: MsgClass,
    bytes: &[u8],
) -> Result<(), ()> {
    let payload = postcard::to_allocvec(&WireFrame {
        from: local,
        class,
        bytes: bytes.to_vec(),
    })
    .map_err(|_| ())?;
    let framed = vd_wire::framing::frame_payload(Ok(payload)).map_err(|_| ())?;
    send.write_all(&framed).await.map_err(|_| ())
}

/// Read ONE `wire::framing` stream frame off a uni stream and decode its `WireFrame` — the SHARED
/// stream-read for the bridge AND the mesh transport (collapsing the previously-duplicated read
/// loops). Returns `None` on clean EOF OR any framing / decode error (the caller stops reading the
/// stream, exactly as before). The cap is checked on `total_len` BEFORE the body is allocated, so a
/// forged oversize header cannot OOM; `frame_body_payload` then applies the reserved-bit reject.
async fn read_one_wireframe(recv: &mut quinn::RecvStream) -> Option<WireFrame> {
    let mut len_buf = [0u8; 4];
    recv.read_exact(&mut len_buf).await.ok()?;
    let total_len = u32::from_be_bytes(len_buf);
    if total_len == 0 || total_len > vd_wire::framing::MAX_STREAM_FRAME_BYTES {
        return None;
    }
    let mut body = vec![0u8; total_len as usize];
    recv.read_exact(&mut body).await.ok()?;
    let payload = vd_wire::framing::frame_body_payload(&body).ok()?;
    postcard::from_bytes::<WireFrame>(payload).ok()
}

async fn write_frame(
    conn: &quinn::Connection,
    stream: &mut Option<quinn::SendStream>,
    local: NodeId,
    frame: &OutboundFrame,
) -> Result<(), ()> {
    if stream.is_none() {
        *stream = Some(conn.open_uni().await.map_err(|_| ())?);
    }
    let Some(send) = stream.as_mut() else {
        return Err(());
    };
    write_wireframe(send, local, frame.class, &frame.bytes).await
}

pub(crate) async fn read_frames(mut recv: quinn::RecvStream, inbound_tx: Sender<Inbound>) {
    while let Some(frame) = read_one_wireframe(&mut recv).await {
        if inbound_tx
            .send(Inbound::Wire {
                from: frame.from,
                class: frame.class,
                bytes: vd_sim::io::bytes(frame.bytes),
            })
            .is_err()
        {
            return;
        }
    }
}

/// Read reliable frames off a uni stream into a shared bounded inbox (the mesh transport's receive
/// path; bounding makes a slow consumer drop stale frames rather than OOM). Same `wire::framing`
/// read as [`read_frames`], surfacing through the bounded-inbox chokepoint.
pub(crate) async fn read_frames_into(
    mut recv: quinn::RecvStream,
    inbox: &std::sync::Arc<std::sync::Mutex<vd_sim::io::BoundedInbox>>,
    stats: &mesh::MeshStats,
) {
    while let Some(frame) = read_one_wireframe(&mut recv).await {
        // Through the ONE surfacing chokepoint (a dropped RELIABLE frame here is the loudest case
        // of all — this is the reliable-stream reader).
        mesh::push_inbox(
            inbox,
            stats,
            Inbound::Wire {
                from: frame.from,
                class: frame.class,
                bytes: vd_sim::io::bytes(frame.bytes),
            },
        );
    }
}

#[cfg(test)]
mod frame_tests {
    use super::{DatagramFrame, WireFrame};
    use vd_core::NodeId;
    use vd_sim::io::MsgClass;

    #[test]
    fn datagram_frame_is_byte_identical_to_the_pre_split_reliable_frame() {
        // R1' hot/cold split: the 20Hz datagram payload must NOT change on the wire. DatagramFrame is
        // the exact {from,class,bytes} shape WireFrame had on the datagram path before the split, so the
        // PvP hot path + the path-MTU budget are unchanged (R2). WireFrame grows seq/incarnation in
        // R2'/R3'; the datagram path stays bare BECAUSE it now uses this separate type.
        let (from, class, bytes) = (NodeId(42), MsgClass::Snapshot, vec![1u8, 2, 3, 4, 5]);
        let dg = postcard::to_allocvec(&DatagramFrame {
            from,
            class,
            bytes: bytes.clone(),
        })
        .expect("postcard encodes DatagramFrame");
        let wf = postcard::to_allocvec(&WireFrame { from, class, bytes })
            .expect("postcard encodes WireFrame");
        assert_eq!(
            dg, wf,
            "DatagramFrame must encode byte-identically to the pre-split datagram WireFrame (zero hot-path wire change)"
        );
    }
}
