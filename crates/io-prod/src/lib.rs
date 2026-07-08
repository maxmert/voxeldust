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
//! │ decode ReliableFrame│              │ react, enqueue sends    │  (BOUNDED)  │ send; on failure│
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
//! - Peer identity in `ReliableFrame::from` (and `DatagramFrame::from`) is sender-asserted; it is
//!   trustworthy only because the channel is mutually authenticated — the design forbids ever
//!   deriving identity from source addresses (R2).
//!
//! Coverage: Tier-B — exercised by the process tier; ratcheted floor, never 100% (HR5).

// R-6d3a F5 (post-impl review): the durable-before-send gate holds a `std::sync::Mutex` guard across the
// non-blocking outbox submit but DROPS it before the block-B durability wait + block-C QUIC `.await`s (MF-1 —
// a fsync wait under the shared lock would serialize every peer). This lint fails the build if a future edit
// ever lets a lock guard cross an `.await`, mechanically guarding the invariant the slice depends on.
#![warn(clippy::await_holding_lock)]

pub mod admin;
pub mod boot;
pub mod mesh;
pub mod outbox;
pub mod probe;
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
/// R2' (the redelivering-transport at-least-once layer): this RELIABLE frame carries the redelivery
/// metadata BELOW the frozen `Transport` seam (sim/node never see it). `incarnation` is the SENDER's
/// process-epoch (a higher value = the sender restarted ⇒ the receiver resets its dedup state); `epoch`
/// is a per-(peer,class) REDIAL counter (bumped on a write error + stream re-dial — the cross-stream-race
/// cure: a straggler on an OLD stream carries the OLD epoch and is dropped BEFORE it can touch the
/// high-water); `seq` is the per-(peer,class) monotone sequence. The bare unreliable hot path uses
/// [`DatagramFrame`] and carries NONE of this (zero PvP-hot-path cost — R1'). The receiver's contiguity
/// verdict (the dedup/replay state machine) lands in R3'; through R2' the receiver decodes the grown
/// frame and DELIVERS as before (the new fields are inert), so the wire grows behaviour-identically.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub(crate) struct ReliableFrame {
    from: NodeId,
    class: MsgClass,
    /// Sender process-epoch (a higher value ⇒ the sender restarted ⇒ the receiver resets its dedup
    /// state — the sender-restart seq-reset cure). **Stamped a literal `0` through R-2a (inert — no
    /// receiver reads it).** R-2b seeds it from a per-process incarnation source (a `MeshConfig`
    /// process-epoch field, threaded through the sender lane FSM); the durable monotone boot-counter
    /// that survives a clock rewind is R-6 (P6/P7).
    incarnation: u64,
    /// Per-(peer,class) redial counter — the R-2'/R-3' cross-stream-race cure. **Stamped `0` through
    /// R-2a; bumped on a write-error re-dial once the sender lane FSM lands (R-2b).**
    epoch: u32,
    /// Per-(peer,class) monotone sequence. Through R-2a a simple per-stream counter (reset on a fresh
    /// stream); R-2b replaces it with the buffer-first lane assignment (assign + retain THEN write).
    seq: u64,
    /// The on-wire payload is a plain `Vec<u8>` (the seam's `Bytes = Arc<[u8]>` is
    /// an in-process sharing optimization; it converts at the serialize boundary).
    bytes: Vec<u8>,
}

/// The UNRELIABLE datagram payload (R1'): a BARE `{from, class, bytes}` carried QUIC-datagram-delimited
/// (NO `wire::framing` stream framing — `codec_flags` is a stream concept). The former `WireFrame` was
/// split into THIS bare datagram type + the reliable [`ReliableFrame`] so the 20Hz latest-wins
/// snapshot/input/ghost-delta hot path carries ZERO reliability metadata and never touches the (future)
/// dedup state — the PvP hot path stays byte-for-byte what it is today (R2). It stays the bare
/// `{from,class,bytes}` shape — `datagram_frame_stays_the_bare_pre_split_shape_distinct_from_reliable`
/// pins it byte-identical to the bare `(from,class,bytes)` tuple AND distinct from `ReliableFrame` (which
/// grows the at-least-once incarnation/epoch/seq fields), so the hot/cold split stays real.
#[derive(Debug, Serialize, Deserialize)]
pub(crate) struct DatagramFrame {
    from: NodeId,
    class: MsgClass,
    bytes: Vec<u8>,
}

/// The cumulative ACK frame (R-3'): the reverse-lane, latest-wins acknowledgement a data RECEIVER sends
/// back to a data SENDER so the sender can RETIRE its retained (unacked) reliable window. One `AckFrame`
/// batches every reliable class on the connection. Framed through the SAME `wire::framing` home as
/// [`ReliableFrame`] (HR3), on a dedicated `STREAM_KIND_ACK` uni stream — never a datagram, never a
/// `MsgClass` arm: acks live BELOW the frozen `Transport` seam (sim/node never see them).
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub(crate) struct AckFrame {
    pub(crate) entries: Vec<AckEntry>,
}

/// One class's cumulative acknowledgement inside an [`AckFrame`]: `ack_through` is the highest CONTIGUOUS
/// seq the receiver has DELIVERED for `(peer, class)` at `epoch` (TCP-style — everything `<=` it retires).
/// `incarnation` ECHOES the sender's process-incarnation as recorded in the receiver's ledger, so the
/// sender's `on_ack` rejects an ack minted against a since-restarted incarnation. It is PER-CLASS (R-6c/L4):
/// a single `AckFrame`-level scalar was last-class-wins (`ack_egress` overwrote it each loop iteration), so
/// once the durable-incarnation work (R-6a) makes sender restart first-class — and if a connection is ever
/// REUSED across a restart carrying two classes at DIFFERENT incarnations — a shared scalar would retire one
/// class's window against the other's incarnation (a silent send stall). Per-entry eliminates that latent
/// invariant dependency (a greenfield hard cutover — no rolling-version mix before the first cloud deploy).
#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub(crate) struct AckEntry {
    pub(crate) class: MsgClass,
    pub(crate) incarnation: u64,
    pub(crate) epoch: u32,
    pub(crate) ack_through: u64,
}

/// The sim-thread→writer queue item, shared by BOTH io-prod transports (the loopback `ProdTransport`
/// bridge and the `mesh` `MeshTransport`) so their frame envelope cannot drift — the ONE crate-root
/// struct that unifies the formerly-duplicated `OutboundFrame`/`OutFrame` (audit `wf_2c963246` DRY pin,
/// closed in R2'). It is the SEAM-LEVEL envelope (`Bytes` + the FIFO `MsgId`); the at-least-once
/// redelivery metadata (incarnation/epoch/seq) is stamped BELOW here, on the [`ReliableFrame`] at the
/// write boundary, never on this queue item (the unreliable hot path never gets that metadata).
#[derive(Debug)]
pub(crate) struct OutFrame {
    pub(crate) to: NodeId,
    pub(crate) class: MsgClass,
    pub(crate) bytes: Bytes,
    pub(crate) msg_id: MsgId,
    /// R-6d producer-intent: the mesh writer (`write_frame`) lowers `Retained` to the reliable lane's durable
    /// write-through; the loopback bridge (ProdTransport) carries it for struct-completeness but ignores it.
    pub(crate) durability: vd_sim::io::Durability,
}

#[derive(Debug, thiserror::Error)]
pub enum ProdIoError {
    #[error("trust: {0}")]
    Trust(#[from] TrustError),
    #[error("quinn connect: {0}")]
    Connect(String),
    #[error("io: {0}")]
    Io(#[from] std::io::Error),
    /// A `MeshReliabilityTuning` field is out of range — rejected at `spawn_mesh` boot BEFORE the
    /// endpoint binds or any task spawns (a mis-tuned redelivery layer must fail loud, R-2b).
    #[error("reliability tuning: {0}")]
    Tuning(String),
}

/// Sim-thread side of the bridge. Implements [`Transport`] over the crossbeam queues.
pub struct ProdTransport {
    local: NodeId,
    outbound_tx: Sender<OutFrame>,
    inbound_rx: Receiver<Inbound>,
    next_msg_id: u64,
}

impl Transport for ProdTransport {
    fn send_durable(
        &mut self,
        to: NodeId,
        class: MsgClass,
        bytes: Bytes,
        durability: vd_sim::io::Durability,
    ) -> Result<MsgId, SendError> {
        let msg_id = MsgId(self.next_msg_id);
        match self.outbound_tx.try_send(OutFrame {
            to,
            class,
            bytes,
            msg_id,
            durability,
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
    let (outbound_tx, outbound_rx) = bounded::<OutFrame>(outbound_capacity);
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
    outbound_rx: Receiver<OutFrame>,
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
            // The bridge is reliable single-stream test infra (the SPIKE-0a loopback). It stamps a
            // monotone per-stream `seq` so its `ReliableFrame`s are well-formed, with `incarnation`/`epoch`
            // 0 — the bridge's receiver (`read_frames`) ignores the redelivery metadata through R2'; the
            // per-(peer,class) lane FSM (retry buffer + replay) lives on the production `mesh` transport.
            let mut seq = 0u64;
            while let Ok(frame) = outbound_rx.recv() {
                let wrote =
                    handle.block_on(write_frame(&conn, &mut stream, local, &frame, &mut seq));
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

/// Frame + write ONE [`ReliableFrame`] on a reliable uni stream — the SHARED stream-write for BOTH the
/// bridge and the mesh transport, routed through `wire::framing::frame_payload` so the `codec_flags`
/// discipline + cap live in ONE home (HR3). The framed bytes already include the length prefix, so it is
/// a single `write_all`. The caller supplies the at-least-once header (`incarnation`/`epoch`/`seq`) — the
/// mesh lane FSM from its per-(peer,class) state, the bridge a monotone per-stream `seq` with 0 epoch/incarnation.
pub(crate) async fn write_reliable_frame(
    send: &mut quinn::SendStream,
    from: NodeId,
    class: MsgClass,
    incarnation: u64,
    epoch: u32,
    seq: u64,
    bytes: &[u8],
) -> Result<(), ()> {
    let payload = postcard::to_allocvec(&ReliableFrame {
        from,
        class,
        incarnation,
        epoch,
        seq,
        bytes: bytes.to_vec(),
    })
    .map_err(|_| ())?;
    let framed = vd_wire::framing::frame_payload(Ok(payload)).map_err(|_| ())?;
    send.write_all(&framed).await.map_err(|_| ())
}

/// Read ONE `wire::framing` stream frame off a uni stream and decode its [`ReliableFrame`] — the SHARED
/// stream-read for the bridge AND the mesh transport (the ONE read home). Returns `None` on clean EOF OR
/// any framing / decode error (the caller stops reading the stream). The cap is checked on `total_len`
/// BEFORE the body is allocated, so a forged oversize header cannot OOM; `frame_body_payload` then
/// applies the reserved-bit reject.
pub(crate) async fn read_one_reliable_frame(recv: &mut quinn::RecvStream) -> Option<ReliableFrame> {
    let mut len_buf = [0u8; 4];
    recv.read_exact(&mut len_buf).await.ok()?;
    let total_len = u32::from_be_bytes(len_buf);
    if total_len == 0 || total_len > vd_wire::framing::MAX_STREAM_FRAME_BYTES {
        return None;
    }
    let mut body = vec![0u8; total_len as usize];
    recv.read_exact(&mut body).await.ok()?;
    let payload = vd_wire::framing::frame_body_payload(&body).ok()?;
    postcard::from_bytes::<ReliableFrame>(payload).ok()
}

/// Frame + write ONE [`AckFrame`] on the reverse ACK uni stream, through the ONE `wire::framing` home (HR3)
/// — mirrors [`write_reliable_frame`]. The framed bytes already carry the length prefix, so it is a single
/// `write_all`.
///
/// ⚠️ CANCEL-SAFETY: the caller MUST keep this `.await` OUTSIDE any `select!`/`timeout` — quinn
/// `write_all` is not cancel-safe, and a torn `AckFrame` permanently desyncs the sender-side ack reader.
pub(crate) async fn write_ack_frame(
    send: &mut quinn::SendStream,
    ack: &AckFrame,
) -> Result<(), ()> {
    let payload = postcard::to_allocvec(ack).map_err(|_| ())?;
    let framed = vd_wire::framing::frame_payload(Ok(payload)).map_err(|_| ())?;
    send.write_all(&framed).await.map_err(|_| ())
}

/// Read ONE `wire::framing` stream frame off the reverse ACK uni stream and decode its [`AckFrame`] —
/// mirrors [`read_one_reliable_frame`] (the same cap-before-alloc + reserved-bit reject). `None` on clean
/// EOF or any framing / decode error (the caller stops reading the stream).
pub(crate) async fn read_one_ack_frame(recv: &mut quinn::RecvStream) -> Option<AckFrame> {
    let mut len_buf = [0u8; 4];
    recv.read_exact(&mut len_buf).await.ok()?;
    let total_len = u32::from_be_bytes(len_buf);
    if total_len == 0 || total_len > vd_wire::framing::MAX_STREAM_FRAME_BYTES {
        return None;
    }
    let mut body = vec![0u8; total_len as usize];
    recv.read_exact(&mut body).await.ok()?;
    let payload = vd_wire::framing::frame_body_payload(&body).ok()?;
    postcard::from_bytes::<AckFrame>(payload).ok()
}

async fn write_frame(
    conn: &quinn::Connection,
    stream: &mut Option<quinn::SendStream>,
    local: NodeId,
    frame: &OutFrame,
    seq: &mut u64,
) -> Result<(), ()> {
    if stream.is_none() {
        *stream = Some(conn.open_uni().await.map_err(|_| ())?);
        *seq = 0; // a fresh stream restarts the bridge's monotone sequence
    }
    let Some(send) = stream.as_mut() else {
        return Err(());
    };
    let this_seq = *seq;
    *seq += 1;
    write_reliable_frame(send, local, frame.class, 0, 0, this_seq, &frame.bytes).await
}

pub(crate) async fn read_frames(mut recv: quinn::RecvStream, inbound_tx: Sender<Inbound>) {
    // R2': decode the grown ReliableFrame; the redelivery metadata (incarnation/epoch/seq) is INERT here
    // — the bridge delivers every frame as before. The contiguity dedup/verdict lands in R3' (mesh side).
    while let Some(frame) = read_one_reliable_frame(&mut recv).await {
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

#[cfg(test)]
mod frame_tests {
    use super::{DatagramFrame, ReliableFrame};
    use vd_core::NodeId;
    use vd_sim::io::MsgClass;

    #[test]
    fn datagram_frame_stays_the_bare_pre_split_shape_distinct_from_reliable() {
        // R1'/R2' hot/cold split: the 20Hz datagram payload must remain EXACTLY {from,class,bytes} —
        // postcard encodes a struct as its fields IN ORDER, so `DatagramFrame` must encode byte-identically
        // to the bare tuple `(from, class, bytes)`. This is the frozen golden (no hand-computed bytes): a
        // future field-add to `DatagramFrame` breaks it, proving NO reliability metadata ever leaks onto the
        // PvP hot path. The reliable `ReliableFrame` now carries incarnation/epoch/seq, so it is DELIBERATELY
        // no longer byte-identical to the datagram — asserted distinct so the split is real.
        let (from, class, bytes) = (NodeId(42), MsgClass::Snapshot, vec![1u8, 2, 3, 4, 5]);
        let dg = postcard::to_allocvec(&DatagramFrame {
            from,
            class,
            bytes: bytes.clone(),
        })
        .expect("postcard encodes DatagramFrame");
        let bare = postcard::to_allocvec(&(from, class, bytes.clone()))
            .expect("postcard encodes the bare tuple");
        assert_eq!(
            dg, bare,
            "the datagram payload stays the bare {{from,class,bytes}} shape — zero hot-path reliability metadata"
        );
        let reliable = postcard::to_allocvec(&ReliableFrame {
            from,
            class,
            incarnation: 0,
            epoch: 0,
            seq: 0,
            bytes,
        })
        .expect("postcard encodes ReliableFrame");
        assert_ne!(
            dg, reliable,
            "ReliableFrame carries redelivery metadata — deliberately NOT byte-identical to the datagram"
        );
    }
}
