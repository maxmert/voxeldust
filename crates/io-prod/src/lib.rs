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

/// The on-stream frame payload: a postcard of this struct, carried inside the `wire::framing`
/// stream frame (`[u32_be total_len][u8 codec_flags][payload]`). io-prod routes ALL stream framing
/// through `wire::framing` (the ONE codec/framing home — HR3 + the `codec_flags` reserved-bit
/// forward-compat path); the cap is `wire::framing::MAX_STREAM_FRAME_BYTES`.
///
/// RELIABLE frames only. Carries the R-1 at-least-once metadata: `incarnation` (the SENDER's
/// process-epoch — a restart bumps it so the receiver resets its dedup high-water) + `seq` (a
/// per-(dest,class) monotone sequence so the receiver dedups replays by high-water). NOT
/// postcard-additive — this is `pub(crate)` in io-prod (NOT a `vd-wire` contract), so both ends
/// upgrade in lockstep; do not rely on field-append interop (postcard is non-self-describing).
#[derive(Debug, Serialize, Deserialize)]
pub(crate) struct WireFrame {
    from: NodeId,
    class: MsgClass,
    /// Sender process-epoch (R-1): the receiver resets its per-(peer,class) dedup high-water when it
    /// sees a higher incarnation (a sender restart) — the fix for the sender-restart seq-reset hole.
    incarnation: u64,
    /// Per-(dest,class) monotone sequence (R-1), starting at 1. The receiver dedups by high-water
    /// (a reconnect-replayed frame with `seq <= delivered` is dropped). `0` is never a live seq.
    seq: u64,
    /// The on-wire payload is a plain `Vec<u8>` (the seam's `Bytes = Arc<[u8]>` is
    /// an in-process sharing optimization; it converts at the serialize boundary).
    bytes: Vec<u8>,
}

/// The UNRELIABLE datagram payload (R-1): BARE — no `incarnation`/`seq`. The 20Hz latest-wins
/// snapshot/input datagrams carry ZERO reliability metadata so the redelivery work adds no overhead
/// to the PvP hot path (R2). Split from [`WireFrame`] precisely so the hot path stays minimal.
#[derive(Debug, Serialize, Deserialize)]
pub(crate) struct DatagramFrame {
    from: NodeId,
    class: MsgClass,
    bytes: Vec<u8>,
}

/// Receiver-side at-least-once DEDUP (R-1). Per-(peer,class): the sender's last-seen `incarnation`
/// and the highest contiguous `seq` delivered. Lives at the MESH level (one per node, shared across
/// every connection) so a sender's RECONNECT-replay is deduped against what an EARLIER connection
/// already delivered — a per-connection map would re-deliver the whole replayed window. The reliable
/// uni stream is FIFO and replay is ascending, so the high-water test is exact (no gaps reach here).
#[derive(Debug, Default)]
pub(crate) struct DedupState {
    delivered: std::collections::BTreeMap<(NodeId, MsgClass), (u64, u64)>, // (peer,class) -> (incarnation, last_seq)
}

/// THE dedup verdict (R-1) — monomorphic, straight-line (HR5(a)). `true` = ACCEPT (deliver this
/// frame), `false` = DROP (a stale-incarnation or already-delivered replay/dup). A HIGHER incarnation
/// RESETS the high-water (the sender restarted); a LOWER incarnation is a stale old-process frame
/// (drop); the same incarnation dedups by strictly-increasing `seq`.
pub(crate) fn dedup_accept(
    state: &mut DedupState,
    from: NodeId,
    class: MsgClass,
    incarnation: u64,
    seq: u64,
) -> bool {
    let entry = state.delivered.entry((from, class)).or_insert((0, 0));
    if incarnation > entry.0 {
        *entry = (incarnation, seq);
        return true;
    }
    if incarnation < entry.0 {
        return false;
    }
    if seq > entry.1 {
        entry.1 = seq;
        return true;
    }
    false
}

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
    incarnation: u64,
    seq: u64,
    bytes: &[u8],
) -> Result<(), ()> {
    let payload = postcard::to_allocvec(&WireFrame {
        from: local,
        class,
        incarnation,
        seq,
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
    // The in-process loopback BRIDGE (test pairs) needs no at-least-once: it never loses a frame,
    // so it sends incarnation/seq = 0 and the bridge reader does not dedup. Redelivery is a mesh-only
    // concern (the at-most-once QUIC carrier); the bridge already models at-least-once by construction.
    write_wireframe(send, local, frame.class, 0, 0, &frame.bytes).await
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
    dedup: &std::sync::Arc<std::sync::Mutex<DedupState>>,
) {
    while let Some(frame) = read_one_wireframe(&mut recv).await {
        // R-1 at-least-once DEDUP (mesh-level, persistent across this peer's connections): drop a
        // replayed/stale frame BEFORE it reaches the sim, so the sim sees each reliable message
        // exactly once. Inert in a healthy run (monotone seq, no dups); only a reconnect-replay
        // (R-3) or a stale-incarnation frame is dropped here. (R-2 will also re-ACK a dropped dup.)
        let accept = {
            let mut state = dedup
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            dedup_accept(
                &mut state,
                frame.from,
                frame.class,
                frame.incarnation,
                frame.seq,
            )
        };
        if !accept {
            continue;
        }
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
mod dedup_tests {
    //! R-1 receiver dedup verdict (the at-least-once core). The reliable uni stream is FIFO + a
    //! reconnect replays ascending, so the high-water test is exact; the incarnation guard fixes the
    //! sender-restart seq-reset (a higher incarnation resets the high-water).
    use super::{DedupState, dedup_accept};
    use vd_core::NodeId;
    use vd_sim::io::MsgClass;

    const P: NodeId = NodeId(7);
    const C: MsgClass = MsgClass::Saga;
    const INC: u64 = 1000; // a realistic sender incarnation

    #[test]
    fn first_frame_and_in_order_run_are_accepted() {
        let mut s = DedupState::default();
        assert!(
            dedup_accept(&mut s, P, C, INC, 1),
            "first frame accepted (resets from genesis)"
        );
        assert!(dedup_accept(&mut s, P, C, INC, 2));
        assert!(
            dedup_accept(&mut s, P, C, INC, 3),
            "monotone run all accepted (inert healthy path)"
        );
    }

    #[test]
    fn a_replayed_or_duplicate_seq_is_dropped() {
        let mut s = DedupState::default();
        assert!(dedup_accept(&mut s, P, C, INC, 1));
        assert!(dedup_accept(&mut s, P, C, INC, 2));
        assert!(
            !dedup_accept(&mut s, P, C, INC, 2),
            "exact replay of the last seq dropped"
        );
        assert!(!dedup_accept(&mut s, P, C, INC, 1), "older replay dropped");
        assert!(
            dedup_accept(&mut s, P, C, INC, 3),
            "the next fresh seq still accepted after the dups"
        );
    }

    #[test]
    fn a_higher_incarnation_resets_the_high_water_sender_restart() {
        let mut s = DedupState::default();
        assert!(
            dedup_accept(&mut s, P, C, INC, 5),
            "delivered up to seq 5 at INC"
        );
        // Sender restarts: higher incarnation, seq resets to 1. Without the guard this would be
        // dropped as <= 5; WITH the guard it resets + accepts (the load-bearing fix).
        assert!(
            dedup_accept(&mut s, P, C, INC + 1, 1),
            "fresh incarnation seq 1 accepted (reset)"
        );
        assert!(dedup_accept(&mut s, P, C, INC + 1, 2));
        assert!(
            !dedup_accept(&mut s, P, C, INC + 1, 1),
            "and dedup resumes within the new incarnation"
        );
    }

    #[test]
    fn a_stale_lower_incarnation_frame_is_dropped() {
        let mut s = DedupState::default();
        assert!(
            dedup_accept(&mut s, P, C, INC + 1, 1),
            "current incarnation established"
        );
        assert!(
            !dedup_accept(&mut s, P, C, INC, 9),
            "a frame from an OLD process incarnation is dropped"
        );
    }

    #[test]
    fn peer_and_class_are_independent_lanes() {
        let mut s = DedupState::default();
        assert!(dedup_accept(&mut s, P, C, INC, 1));
        assert!(
            dedup_accept(&mut s, P, MsgClass::Control, INC, 1),
            "a different class is its own lane"
        );
        assert!(
            dedup_accept(&mut s, NodeId(8), C, INC, 1),
            "a different peer is its own lane"
        );
        assert!(
            !dedup_accept(&mut s, P, C, INC, 1),
            "the original (P,Saga) lane still dedups"
        );
    }
}
