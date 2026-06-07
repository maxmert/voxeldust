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
//! - TLS: a self-signed certificate trusted explicitly by the peer (the day-1
//!   single-cluster-cert PSK model — NEVER certificate-verification skipping, which was
//!   audit finding R7's enabler). The full cluster-cert plumbing lands at P3.
//! - Peer identity in `WireFrame::from` is sender-asserted for the SPIKE; it becomes
//!   trustworthy only because the channel is mutually authenticated (P3 mTLS) — the
//!   design forbids ever deriving identity from source addresses (R2).
//!
//! Coverage: Tier-B — exercised by the process tier; ratcheted floor, never 100% (HR5).

use std::net::SocketAddr;
use std::sync::Arc;

use crossbeam_channel::{Receiver, Sender, TrySendError, bounded, unbounded};
use serde::{Deserialize, Serialize};
use vd_core::{MsgId, NodeId};
use vd_sim::io::{Bytes, Inbound, MsgClass, SendError, Transport};

/// Maximum frame size accepted on a stream. Operational constant: migrates into
/// `TransportTuning` when that struct lands (connection-plane work, P1).
const MAX_FRAME_BYTES: u32 = 1 << 20;

/// The on-stream frame: 4-byte BE length prefix, then postcard of this struct.
#[derive(Debug, Serialize, Deserialize)]
struct WireFrame {
    from: NodeId,
    class: MsgClass,
    bytes: Bytes,
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
    #[error("tls setup: {0}")]
    Tls(String),
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

/// Build two nodes connected over real QUIC on 127.0.0.1.
///
/// `outbound_capacity` bounds each node's outbound queue (the back-pressure point).
/// `paused_writers` starts both writer threads gated, so a test can flood the queue
/// deterministically before any drain happens.
pub fn loopback_pair(
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

    // Day-1 trust model: one self-signed cert, explicitly trusted by the peer.
    let cert = rcgen::generate_simple_self_signed(vec!["localhost".into()])
        .map_err(|e| ProdIoError::Tls(e.to_string()))?;
    let cert_der = rustls::pki_types::CertificateDer::from(cert.cert.der().to_vec());
    let key_der = rustls::pki_types::PrivateKeyDer::Pkcs8(
        rustls::pki_types::PrivatePkcs8KeyDer::from(cert.key_pair.serialize_der()),
    );

    let server_config = quinn::ServerConfig::with_single_cert(vec![cert_der.clone()], key_der)
        .map_err(|e| ProdIoError::Tls(e.to_string()))?;
    let mut roots = rustls::RootCertStore::empty();
    roots
        .add(cert_der)
        .map_err(|e| ProdIoError::Tls(e.to_string()))?;
    let client_config = quinn::ClientConfig::with_root_certificates(Arc::new(roots))
        .map_err(|e| ProdIoError::Tls(e.to_string()))?;

    let bind: SocketAddr = "127.0.0.1:0"
        .parse()
        .map_err(|_| ProdIoError::Tls("bad bind addr".into()))?;

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

async fn read_frames(mut recv: quinn::RecvStream, inbound_tx: Sender<Inbound>) {
    loop {
        let mut len_buf = [0u8; 4];
        if recv.read_exact(&mut len_buf).await.is_err() {
            return;
        }
        let len = u32::from_be_bytes(len_buf);
        if len > MAX_FRAME_BYTES {
            return;
        }
        let mut buf = vec![0u8; len as usize];
        if recv.read_exact(&mut buf).await.is_err() {
            return;
        }
        match postcard::from_bytes::<WireFrame>(&buf) {
            Ok(frame) => {
                if inbound_tx
                    .send(Inbound::Wire {
                        from: frame.from,
                        class: frame.class,
                        bytes: frame.bytes,
                    })
                    .is_err()
                {
                    return;
                }
            }
            Err(_) => return,
        }
    }
}
