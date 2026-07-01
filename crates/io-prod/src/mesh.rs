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
use crate::{DatagramFrame, OutFrame, ProdIoError, ReliableFrame, write_reliable_frame};

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
    /// This process's incarnation, stamped on every reliable frame (R-2b). A higher value tells a
    /// receiver the sender RESTARTED (⇒ reset its dedup high-water — the sender-restart seq-reset cure).
    /// Bins pass `VD_PROCESS_INCARNATION` (default 0 in dev/tests); R-6 (P6/P7) makes it a durable
    /// monotone boot-counter that survives a clock rewind.
    pub process_incarnation: u64,
    /// The at-least-once redelivery tuning (R-2b). `validate`d loud at `spawn_mesh` boot.
    pub reliability: MeshReliabilityTuning,
}

/// Default per-lane unacked-retry-buffer ceiling (R-4' shed point; counted in R-2b). 4 MiB.
pub const DEFAULT_RETRY_BUFFER_MAX_BYTES: usize = 4 * 1024 * 1024;
/// Default cadence at which a receiver flushes a lone cumulative ack (R-3'); a lone damage/death/
/// despawn is acked within this even without follow-up traffic. 50 ms.
pub const DEFAULT_ACK_IDLE_FLUSH: Duration = Duration::from_millis(50);
/// Default replays-before-a-hard-`NodeUnreachable`-bounce (R-4'): a transient blip costs ZERO bounce;
/// only a peer that fails this many consecutive replays is declared unreachable.
pub const DEFAULT_CONFIRM_UNREACHABLE_AFTER_RETRIES: u32 = 3;

/// At-least-once redelivery tuning — the ONE config home for the redelivering transport (no inline
/// literals at use sites). Plain (not `Serialize`): operational tuning, never persisted.
///
/// ⚠️ R-2b CONSUMER NOTE: only `retry_buffer_max_bytes`'s constraint is structurally relevant in R-2b
/// (it bounds the cap). `ack_idle_flush_interval` (the ack cadence) and `confirm_unreachable_after_retries`
/// (blip-tolerance) are validated at boot now but NOT YET CONSUMED — R-2b still bounces `NodeUnreachable`
/// per failed frame. Their behaviour lands with R-3' (acks) / R-4' (confirmed-dead-after-N); do not expect
/// blip-tolerance from `confirm_unreachable_after_retries` until R-4'.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct MeshReliabilityTuning {
    /// Per-lane unacked-retry-buffer byte ceiling. The R-4' shed point (counted in R-2b); must be at
    /// least one max-size frame or a single large reliable frame could never be retained.
    pub retry_buffer_max_bytes: usize,
    /// Cadence for flushing a lone cumulative ack (R-3'). Must be > 0 (a 0 timer would busy-spin).
    pub ack_idle_flush_interval: Duration,
    /// Consecutive failed replays before a hard `NodeUnreachable` bounce (R-4'). Must be ≥ 1.
    pub confirm_unreachable_after_retries: u32,
}

impl Default for MeshReliabilityTuning {
    fn default() -> Self {
        MeshReliabilityTuning {
            retry_buffer_max_bytes: DEFAULT_RETRY_BUFFER_MAX_BYTES,
            ack_idle_flush_interval: DEFAULT_ACK_IDLE_FLUSH,
            confirm_unreachable_after_retries: DEFAULT_CONFIRM_UNREACHABLE_AFTER_RETRIES,
        }
    }
}

/// A `MeshReliabilityTuning` field out of range — surfaced loud at `spawn_mesh` boot.
#[derive(Clone, Copy, Debug, PartialEq, Eq, thiserror::Error)]
pub enum MeshReliabilityTuningError {
    #[error("retry_buffer_max_bytes must be >= the max stream frame size")]
    RetryBufferTooSmall,
    #[error("ack_idle_flush_interval must be > 0")]
    AckFlushZero,
    #[error("confirm_unreachable_after_retries must be >= 1")]
    ConfirmRetriesZero,
}

impl MeshReliabilityTuning {
    /// Reject an out-of-range field (fail-loud at boot). Branchless-faithful: one `if` per arm, in
    /// order, each returning its specific typed error (HR5(d) — equality-comparable, no `matches!`).
    ///
    /// # Errors
    /// The specific [`MeshReliabilityTuningError`] for the first out-of-range field.
    pub fn validate(&self) -> Result<(), MeshReliabilityTuningError> {
        if self.retry_buffer_max_bytes < vd_wire::framing::MAX_STREAM_FRAME_BYTES as usize {
            return Err(MeshReliabilityTuningError::RetryBufferTooSmall);
        }
        if self.ack_idle_flush_interval.is_zero() {
            return Err(MeshReliabilityTuningError::AckFlushZero);
        }
        if self.confirm_unreachable_after_retries < 1 {
            return Err(MeshReliabilityTuningError::ConfirmRetriesZero);
        }
        Ok(())
    }
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
        process_incarnation: u64,
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
            process_incarnation,
            reliability: MeshReliabilityTuning::default(),
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
    // --- R-2b at-least-once redelivery counters (the design's never-silent rule) ---
    /// RECEIVER: a reliable frame whose `incarnation` is BELOW the lane's recorded incarnation —
    /// a straggler from a since-restarted sender, dropped + counted (R-3'; 0 until then).
    pub stale_incarnation_drop: AtomicU64,
    /// RECEIVER: a reliable frame on an OLD `epoch` (a straggler from a pre-redial stream), dropped
    /// BEFORE it can touch the high-water — the cross-stream-race cure (R-3'; 0 until then).
    pub stale_epoch_drop: AtomicU64,
    /// RECEIVER: a reliable frame at or below the high-water — an expected replay, deduped + counted
    /// so the sim never sees a dup (R-3'; 0 until then).
    pub dedup_drop: AtomicU64,
    /// RECEIVER: a reliable frame leaving a contiguity GAP (`seq > hw+1`) — a MUST-BE-0 alert (the
    /// wire epoch keeps it genuinely 0; R-3', 0 until then).
    pub gap_drop: AtomicU64,
    /// SENDER: a reliable frame SHED — in R-2b this fires for a per-frame payload that exceeds the
    /// max stream frame size (un-framable ⇒ rejected, never retained, bounced `NodeUnreachable`).
    /// R-4' adds the bounded-retry-buffer total shed on the same counter.
    pub reliable_shed: AtomicU64,
}

/// A snapshot of the mesh counters (loads the atomics).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct MeshStatsSnapshot {
    pub datagrams_dropped_too_large: u64,
    pub datagrams_dropped_send: u64,
    pub inbound_dropped_reliable: u64,
    pub inbound_dropped_unreliable: u64,
    pub stale_incarnation_drop: u64,
    pub stale_epoch_drop: u64,
    pub dedup_drop: u64,
    pub gap_drop: u64,
    pub reliable_shed: u64,
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

/// One reliable send LANE — the per-(peer,class) at-least-once sender FSM (R-2b). Lazily created on
/// the first reliable frame for a class to a peer. ALL FSM logic (seq assign, retain, replay framing,
/// ack-retire) is SYNCHRONOUS + unit-testable WITHOUT tokio/quinn; only the stream open/write (in
/// `write_frame`) is async. One stream per class so a stalled class never head-of-line-blocks another.
struct ReliableLaneSender {
    /// THIS lane's QUIC uni stream. `None` until the first send opens it; reset to `None` on any write
    /// error (re-opened lazily on the next send after the peer re-dial).
    stream: Option<quinn::SendStream>,
    /// The sender's process-incarnation, stamped on every frame (constant for the lane's life). R-3'
    /// reads it to reset the receiver's dedup state when the sender restarts.
    incarnation: u64,
    /// Per-lane redial counter, bumped on EVERY write error (the cross-stream-race cure: an OLD-stream
    /// straggler carries the OLD epoch ⇒ R-3' drops it BEFORE it advances the high-water). Starts 0.
    epoch: u32,
    /// Next seq to ASSIGN — monotone, NEVER reset, NEVER rolled back on a failed write (buffer-first ⇒
    /// a burned write leaves no gap and re-uses no seq).
    next_seq: u64,
    /// Unacked frames keyed by seq. `BTreeMap` ⇒ ascending key = ascending seq = exact replay order;
    /// holds the FULLY-BUILT [`ReliableFrame`] so replay re-stamps `epoch` only. R-2b has no ack
    /// PRODUCER, so this never drains at runtime (the ledgered non-draining-retry window R-2b→R-3'/R-4').
    /// The cumulative-ack RETIRE (`on_ack` + a `base` watermark) and the byte-total for the buffer SHED
    /// (`retry_bytes`) are the sender half of the R-3'/R-4' protocols and land WITH their consumers —
    /// not added speculatively here.
    retry: BTreeMap<u64, ReliableFrame>,
}

impl ReliableLaneSender {
    fn new(incarnation: u64) -> ReliableLaneSender {
        ReliableLaneSender {
            stream: None,
            incarnation,
            epoch: 0,
            next_seq: 0,
            retry: BTreeMap::new(),
        }
    }

    /// BUFFER-FIRST: stamp a NEW seq, build the frame at `(incarnation, epoch, seq)`, and — IF it frames
    /// within the cap — retain it + advance `next_seq`, returning the assigned SEQ. Returns `Err(())`
    /// WITHOUT retaining or advancing `next_seq` if the framed `ReliableFrame` would exceed
    /// `MAX_STREAM_FRAME_BYTES`. The check is on the ENCODED frame (the envelope varints + codec byte can
    /// push a near-cap payload over the cap), routed through the ONE codec/framing home `encode_frame`
    /// (HR3) — NOT the raw payload length. An un-framable frame must NEVER enter `retry`: it can never be
    /// sent, so it would poison the lane (replayed-and-failed every redial, a permanent contiguity wall).
    /// There is exactly ONE frame object (in `retry`), which the write path reads back — a frame can never
    /// be written from two objects. Pure: no I/O; a write failure AFTER a successful assign never rolls
    /// it back (the no-burned-seq guarantee).
    fn assign_and_retain(
        &mut self,
        from: NodeId,
        class: MsgClass,
        bytes: &[u8],
    ) -> Result<u64, ()> {
        let seq = self.next_seq;
        // Build the frame stamped at the WORST-CASE epoch (`u32::MAX`, a 5-byte varint) for the framing
        // CHECK: `replay_batch` re-stamps a retained frame to an ever-HIGHER epoch on each redial, and the
        // epoch varint grows from 1 byte (epoch 0) to 5 bytes (>= 2^28). Checking at `u32::MAX` guarantees
        // an ACCEPTED frame frames within the cap at EVERY future re-stamp — so a near-cap frame can never
        // overflow on the replay path (the relocated-poison residual, audit `wf_93fc5909` re-verify). The
        // REAL epoch is set below before retaining; the same one frame object (ONE `bytes` clone) is reused.
        let mut frame = ReliableFrame {
            from,
            class,
            incarnation: self.incarnation,
            epoch: u32::MAX,
            seq,
            bytes: bytes.to_vec(),
        };
        // Reject on the FRAMED size, before retaining (the same encode+frame the write performs).
        vd_wire::framing::encode_frame(&frame).map_err(|_| ())?;
        frame.epoch = self.epoch; // the real epoch for retention + the first write
        self.retry.insert(seq, frame);
        self.next_seq += 1;
        Ok(seq)
    }

    /// Write-error recovery: drop the stream + bump the epoch (DISTINCT from the peer-writer's anti-storm
    /// backoff sleep). `wrapping_add` is unreachable in practice — 2^32 redials on one lane at the 50 ms
    /// backoff floor is ~6.8 years of continuous flapping, and R-4' `confirm_unreachable_after_retries`
    /// kills the lane long before — but it is wrapping (never a panic) rather than a debug overflow.
    fn on_write_error(&mut self) {
        self.stream = None;
        self.epoch = self.epoch.wrapping_add(1);
    }

    /// The ascending replay set after a re-dial: every retained frame RE-STAMPED with the CURRENT
    /// (already-bumped) epoch, ORIGINAL seq preserved. `BTreeMap` value order = ascending seq. When the
    /// stream was just (re)opened this is the COMPLETE write set — it INCLUDES the just-assigned frame
    /// as its highest entry (assign-then-replay ordering), so the new frame is written exactly once
    /// inside the batch (the CRITICAL double-write cure).
    fn replay_batch(&self) -> Vec<ReliableFrame> {
        self.retry
            .values()
            .map(|f| ReliableFrame {
                epoch: self.epoch,
                ..f.clone()
            })
            .collect()
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
            stale_incarnation_drop: self.stats.stale_incarnation_drop.load(Ordering::Relaxed),
            stale_epoch_drop: self.stats.stale_epoch_drop.load(Ordering::Relaxed),
            dedup_drop: self.stats.dedup_drop.load(Ordering::Relaxed),
            gap_drop: self.stats.gap_drop.load(Ordering::Relaxed),
            reliable_shed: self.stats.reliable_shed.load(Ordering::Relaxed),
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
    // FAIL LOUD before binding the endpoint or spawning any task: a mis-tuned redelivery layer must
    // never come up half-configured (R-2b). This is the FIRST statement.
    cfg.reliability
        .validate()
        .map_err(|e| ProdIoError::Tuning(e.to_string()))?;
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
            incarnation: cfg.process_incarnation,
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
    /// This process's incarnation, stamped on every reliable frame this lane sends (R-2b).
    incarnation: u64,
}

/// One peer's writer: drains its lane in FIFO order, dialing on demand, with
/// exponential re-dial backoff so a dead host is NOT hammered at the tick rate
/// (TRANSPORT-3). Reliable frames ride a persistent uni stream and bounce
/// `NodeUnreachable` on hard failure; unreliable frames ride datagrams (latest-wins:
/// loss is correct, no bounce).
async fn peer_writer(mut w: PeerWriter) {
    // ONE peer-level connection (shared across classes) + ONE FSM lane per reliable class. A reliable
    // OutFrame routes to its class lane; unreliable rides datagrams on the shared connection.
    let mut connection: Option<quinn::Connection> = None;
    let mut lanes: BTreeMap<MsgClass, ReliableLaneSender> = BTreeMap::new();
    let mut backoff = w.backoff_min;
    while let Some(frame) = w.rx.recv().await {
        let sent = write_frame(
            &w.endpoint,
            w.addr,
            &mut connection,
            &mut lanes,
            w.local,
            w.incarnation,
            &frame,
            &w.stats,
        )
        .await;
        match sent {
            Ok(()) => backoff = w.backoff_min,
            Err(()) => {
                // FULL connection drop — REQUIRED for old-stream cleanup: dropping the quinn
                // Connection tears down its streams, which terminates the RECEIVER's per-stream
                // reader tasks. NEVER downgrade this to a stream-only reset without adding a
                // receiver-side stream reaper (DEFERRED: the stream-only-error optimization).
                connection = None;
                // Every lane's stream lived on the dead connection: reset each + bump its epoch, so
                // the next send per lane re-dials, re-opens, and REPLAYS its unacked window under the
                // new epoch (the cross-stream-race cure). The epoch bump (recovery) is DISTINCT from
                // the backoff sleep (anti-connect-storm) below.
                for lane in lanes.values_mut() {
                    lane.on_write_error();
                }
                // Reliable senders are waiting on an ack path; tell them the peer is unreachable.
                // (R-4' refines this to confirmed-dead-after-N-retries; R-2b keeps the per-frame
                // bounce.) Unreliable (datagram) loss is silent by design.
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
    // Loop exit = the per-peer channel closed = clean shutdown; lane streams drop here (quinn
    // implicit-finish on the still-live connection, or no-op on an already-dead one).
}

/// Ensure a connection (dial on demand) and write one frame on the carrier its class mandates. The
/// reliable arm is the R-2b lane FSM; the unreliable arm is the unchanged bare-datagram hot path.
///
/// ⚠️ CANCEL-SAFETY: NEVER wrap this in `tokio::time::timeout`/`select!` — a cancelled `write_all`
/// would leave a half-written frame on a `Some` stream. A send deadline must be internal + an explicit
/// `on_write_error`. The only cancel point in the writer is the `w.rx.recv()` await above.
#[allow(clippy::too_many_arguments)] // writer state threaded explicitly
async fn write_frame(
    endpoint: &quinn::Endpoint,
    addr: SocketAddr,
    connection: &mut Option<quinn::Connection>,
    lanes: &mut BTreeMap<MsgClass, ReliableLaneSender>,
    local: NodeId,
    incarnation: u64,
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
        // A fresh connection invalidates every lane's stream (the old ones died with the old conn).
        for lane in lanes.values_mut() {
            lane.stream = None;
        }
    }
    let conn = connection.as_ref().ok_or(())?;

    match frame.class.reliability() {
        Reliability::Reliable => {
            // Route to the class lane (lazily created at THIS process incarnation, epoch 0).
            let lane = lanes
                .entry(frame.class)
                .or_insert_with(|| ReliableLaneSender::new(incarnation));

            // BUFFER-FIRST: assign + retain BEFORE any write. A failed write below NEVER rolls back the
            // seq (no burned seq, no gap). The ONE frame object lives in `lane.retry`. `assign_and_retain`
            // also performs the OVERSIZE REJECT on the FRAMED size (the envelope + codec byte can push a
            // near-cap payload over `MAX_STREAM_FRAME_BYTES`): an un-framable frame can NEVER be sent, so
            // retaining it would poison the lane forever (replayed-and-failed every redial). On that
            // rejection it returns `Err` WITHOUT retaining ⇒ shed-count + bounce + return, nothing in
            // `retry`. `reliable_shed` is the one R-2b-live counter.
            let seq = match lane.assign_and_retain(local, frame.class, &frame.bytes) {
                Ok(seq) => seq,
                Err(()) => {
                    stats.reliable_shed.fetch_add(1, Ordering::Relaxed);
                    tracing::warn!(
                        "reliable frame (payload {}) would exceed MAX_STREAM_FRAME_BYTES framed; \
                         rejected (counted, not retained)",
                        frame.bytes.len()
                    );
                    return Err(()); // ⇒ peer_writer bounces NodeUnreachable; nothing retained
                }
            };

            // STRUCTURAL if/else — NO fall-through (the CRITICAL double-write cure):
            if lane.stream.is_none() {
                // (re)opened stream: write EXACTLY replay_batch(), which by construction includes the
                // just-assigned frame as its HIGHEST entry ⇒ the new frame is written exactly once,
                // inside the batch. Never falls through to the steady-state write below.
                let send = conn.open_uni().await.map_err(|_| ())?;
                lane.stream = Some(send);
                let batch = lane.replay_batch();
                debug_assert_eq!(
                    batch.last().map(|f| f.seq),
                    Some(seq),
                    "buffer-first invariant: the just-assigned frame is the highest replay entry"
                );
                let send = lane.stream.as_mut().ok_or(())?;
                for f in &batch {
                    write_reliable_frame(
                        send,
                        f.from,
                        f.class,
                        f.incarnation,
                        f.epoch,
                        f.seq,
                        &f.bytes,
                    )
                    .await
                    .map_err(|_| ())?;
                }
                Ok(())
            } else {
                // steady state: stream open ⇒ write ONLY the new frame (NEVER the backlog). Read the
                // single frame object back out of `retry`, copying its fields so the immutable borrow of
                // `retry` ends before the mutable borrow of `stream`.
                let nf = lane.retry.get(&seq).ok_or(())?;
                let (f_from, f_class, f_inc, f_epoch, f_bytes) = (
                    nf.from,
                    nf.class,
                    nf.incarnation,
                    nf.epoch,
                    nf.bytes.clone(),
                );
                let send = lane.stream.as_mut().ok_or(())?;
                write_reliable_frame(send, f_from, f_class, f_inc, f_epoch, seq, &f_bytes).await
            }
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

    // ---- Pure ReliableLaneSender FSM tests (no tokio/quinn — the at-least-once correctness core) ----

    const FROM: NodeId = NodeId(1);
    const CLASS: MsgClass = MsgClass::Saga;

    #[test]
    fn sender_assigns_monotone_seq_and_retains() {
        let mut lane = ReliableLaneSender::new(7);
        assert_eq!(lane.assign_and_retain(FROM, CLASS, b"a"), Ok(0));
        assert_eq!(lane.assign_and_retain(FROM, CLASS, b"bb"), Ok(1));
        assert_eq!(lane.assign_and_retain(FROM, CLASS, b"ccc"), Ok(2));
        assert_eq!(lane.next_seq, 3);
        assert_eq!(
            lane.retry.keys().copied().collect::<Vec<_>>(),
            vec![0, 1, 2]
        );
        assert_eq!(
            lane.retry[&1].bytes, b"bb",
            "the retained frame holds its payload"
        );
        // Frames are stamped with the lane's incarnation + current epoch (0).
        let f0 = &lane.retry[&0];
        assert_eq!((f0.incarnation, f0.epoch, f0.seq), (7, 0, 0));
    }

    #[test]
    fn write_error_does_not_roll_back_seq() {
        // BUFFER-FIRST: a failed write (modelled by on_write_error after an assign) never burns or
        // re-uses a seq — the next assign is the next monotone value, at the bumped epoch.
        let mut lane = ReliableLaneSender::new(0);
        assert_eq!(lane.assign_and_retain(FROM, CLASS, b"x"), Ok(0));
        lane.on_write_error();
        assert_eq!(lane.assign_and_retain(FROM, CLASS, b"y"), Ok(1)); // not reused 0, not skipped 2
        assert_eq!(
            lane.retry[&1].epoch, 1,
            "the post-error assign carries the bumped epoch"
        );
        assert_eq!(lane.retry.keys().copied().collect::<Vec<_>>(), vec![0, 1]);
    }

    #[test]
    fn replay_batch_is_ascending_seq_with_current_epoch() {
        let mut lane = ReliableLaneSender::new(0);
        for b in [b"a".as_slice(), b"b", b"c"] {
            lane.assign_and_retain(FROM, CLASS, b).expect("fits");
        }
        lane.on_write_error(); // epoch -> 1
        let batch = lane.replay_batch();
        assert_eq!(
            batch.iter().map(|f| f.seq).collect::<Vec<_>>(),
            vec![0, 1, 2]
        );
        assert!(
            batch.iter().all(|f| f.epoch == 1),
            "re-stamped with the current epoch"
        );
        assert_eq!(batch[1].bytes, b"b", "original seq+payload preserved");
        lane.on_write_error(); // epoch -> 2
        assert!(
            lane.replay_batch().iter().all(|f| f.epoch == 2),
            "replay tracks the LIVE epoch (idempotent re-stamp)"
        );
    }

    #[test]
    fn replay_batch_restamps_latest_epoch_after_two_bumps_with_intervening_assign() {
        // No stale-epoch frame leaks into a fresh replay even when assigns interleave with bumps.
        let mut lane = ReliableLaneSender::new(0);
        lane.assign_and_retain(FROM, CLASS, b"0").expect("fits"); // seq0 @ e0
        lane.assign_and_retain(FROM, CLASS, b"1").expect("fits"); // seq1 @ e0
        lane.on_write_error(); // e1
        lane.assign_and_retain(FROM, CLASS, b"2").expect("fits"); // seq2 @ e1
        lane.on_write_error(); // e2
        let batch = lane.replay_batch();
        assert_eq!(
            batch.iter().map(|f| f.seq).collect::<Vec<_>>(),
            vec![0, 1, 2]
        );
        assert!(
            batch.iter().all(|f| f.epoch == 2),
            "ALL entries re-stamped at the latest epoch"
        );
    }

    #[test]
    fn the_open_branch_write_set_equals_retry_exactly_once() {
        // The CRITICAL double-write cure (sender side): replay_batch() — the write set when a stream
        // is (re)opened — contains EACH retained seq EXACTLY once, with the just-assigned seq present
        // exactly once and LAST (so the steady-state path never re-writes it).
        let mut lane = ReliableLaneSender::new(0);
        lane.assign_and_retain(FROM, CLASS, b"0").expect("fits");
        lane.on_write_error();
        let new_seq = lane.assign_and_retain(FROM, CLASS, b"1").expect("fits");
        let batch = lane.replay_batch();
        let seqs: Vec<u64> = batch.iter().map(|f| f.seq).collect();
        assert_eq!(
            seqs,
            vec![0, 1],
            "each retained seq exactly once, ascending"
        );
        assert_eq!(
            seqs.iter().filter(|&&s| s == new_seq).count(),
            1,
            "new seq exactly once"
        );
        assert_eq!(
            batch.last().map(|f| f.seq),
            Some(new_seq),
            "new seq is last"
        );
    }

    #[test]
    fn retry_keys_are_contiguous_below_next_seq() {
        // R-2b has no ack producer, so `retry` never drains: keys are 0..next_seq (the ledgered
        // non-draining-retry window). The cumulative-ack RETIRE (`on_ack`/`base`) + its tests land WITH
        // the R-3' ack producer; the byte-total for the buffer SHED lands with R-4'.
        let mut lane = ReliableLaneSender::new(0);
        for b in [b"a".as_slice(), b"b", b"c"] {
            lane.assign_and_retain(FROM, CLASS, b).expect("fits");
        }
        lane.on_write_error();
        lane.assign_and_retain(FROM, CLASS, b"d").expect("fits");
        assert_eq!(
            lane.retry.keys().copied().collect::<Vec<_>>(),
            vec![0, 1, 2, 3]
        );
        assert!(
            lane.retry.keys().all(|&k| k < lane.next_seq),
            "all retry keys < next_seq"
        );
    }

    #[test]
    fn an_unframable_reliable_frame_is_rejected_not_retained() {
        // The cap is on the FRAMED size (envelope varints + codec byte), NOT the raw payload — a payload
        // at exactly MAX frames OVER the cap. assign must REJECT it without retaining or burning a seq:
        // an un-framable frame can never be sent, so retaining it would poison the lane (replayed-and-
        // failed every redial). (Regression for the off-by-the-header poison window, audit wf_93fc5909.)
        let mut lane = ReliableLaneSender::new(7);
        let at_cap = vec![0u8; vd_wire::framing::MAX_STREAM_FRAME_BYTES as usize];
        assert_eq!(lane.assign_and_retain(FROM, CLASS, &at_cap), Err(()));
        assert!(
            lane.retry.is_empty(),
            "an un-framable frame is NOT retained"
        );
        assert_eq!(lane.next_seq, 0, "a rejected assign does not burn a seq");
        // The lane is NOT poisoned: a normal frame after a rejection assigns cleanly at seq 0.
        assert_eq!(lane.assign_and_retain(FROM, CLASS, b"ok"), Ok(0));
    }

    #[test]
    fn an_accepted_frame_stays_framable_at_every_replay_epoch() {
        // assign checks at epoch=u32::MAX, so a frame it ACCEPTS frames within the cap at EVERY replay
        // re-stamp (replay only raises the epoch). GUARD: find the LARGEST near-cap payload assign accepts,
        // then re-stamp the retained frame to u32::MAX and confirm it STILL frames. Removing the
        // worst-case-epoch headroom (checking at the live epoch instead) would accept a larger payload that
        // overflows once replay re-stamps it — the relocated poison (audit wf_93fc5909). incarnation =
        // u64::MAX for the worst-case envelope too.
        let max = vd_wire::framing::MAX_STREAM_FRAME_BYTES as usize;
        let accepted = (0..48)
            .map(|d| max - d)
            .find(|&n| {
                ReliableLaneSender::new(u64::MAX)
                    .assign_and_retain(FROM, CLASS, &vec![0u8; n])
                    .is_ok()
            })
            .expect("some near-cap payload is accepted");
        let mut lane = ReliableLaneSender::new(u64::MAX);
        let seq = lane
            .assign_and_retain(FROM, CLASS, &vec![0u8; accepted])
            .expect("accepted");
        let mut replayed = lane.retry[&seq].clone();
        replayed.epoch = u32::MAX;
        assert!(
            vd_wire::framing::encode_frame(&replayed).is_ok(),
            "an accepted near-cap frame must still frame at the max replay epoch"
        );
    }

    #[test]
    fn sibling_error_does_not_leak_epoch_into_a_fresh_lane() {
        // Lazy lane creation always starts epoch 0 — a sibling class's write error never bleeds into
        // a freshly-created lane.
        let mut a = ReliableLaneSender::new(5);
        a.assign_and_retain(FROM, MsgClass::Control, b"x")
            .expect("fits");
        a.on_write_error(); // A -> epoch 1
        let mut b = ReliableLaneSender::new(5);
        b.assign_and_retain(FROM, MsgClass::Saga, b"y")
            .expect("fits");
        assert_eq!(
            b.retry[&0].epoch, 0,
            "a fresh lane starts at epoch 0, never a sibling's"
        );
    }

    #[test]
    fn reliability_tuning_validate_rejects_each_zeroed_field() {
        assert_eq!(MeshReliabilityTuning::default().validate(), Ok(()));
        let too_small = MeshReliabilityTuning {
            retry_buffer_max_bytes: 0,
            ..MeshReliabilityTuning::default()
        };
        assert_eq!(
            too_small.validate().expect_err("rejected"),
            MeshReliabilityTuningError::RetryBufferTooSmall
        );
        let flush_zero = MeshReliabilityTuning {
            ack_idle_flush_interval: Duration::ZERO,
            ..MeshReliabilityTuning::default()
        };
        assert_eq!(
            flush_zero.validate().expect_err("rejected"),
            MeshReliabilityTuningError::AckFlushZero
        );
        let confirm_zero = MeshReliabilityTuning {
            confirm_unreachable_after_retries: 0,
            ..MeshReliabilityTuning::default()
        };
        assert_eq!(
            confirm_zero.validate().expect_err("rejected"),
            MeshReliabilityTuningError::ConfirmRetriesZero
        );
    }

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
                &MeshConfig::new(id, addr, book.clone(), capacity, 0),
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

    /// R-2b: TWO reliable classes to the SAME live peer ride SEPARATE per-(peer,class) uni streams on
    /// one connection — both arrive with no loss/dup. This is the regression guard the process tier
    /// (gateway↔orchestrator carries Saga+Membership; gateway↔shard carries Control+Saga) implicitly
    /// depends on; no prior unit test exercised the multi-stream-per-connection path. The per-class
    /// split REMOVES cross-class ordering, so either RELATIVE arrival order is acceptable (asserted).
    #[test]
    fn two_reliable_classes_to_one_live_peer_both_arrive() {
        let rt = runtime();
        let trust = ClusterTrust::generate("vd-mesh-test").expect("trust");
        let (mut nodes, _controls) = cluster(rt.handle(), &trust, 2, 64);
        nodes[0]
            .send(NodeId(2), MsgClass::Control, vec![0xC0].into())
            .expect("control enqueued");
        nodes[0]
            .send(NodeId(2), MsgClass::Saga, vec![0x5A].into())
            .expect("saga enqueued");
        let got = wait_for(&mut nodes[1], |g| {
            g.iter()
                .filter(|m| matches!(m, Inbound::Wire { .. }))
                .count()
                >= 2
        });
        // Both classes landed exactly once (no loss, no dup), in EITHER relative order.
        let mut by_class: BTreeMap<MsgClass, Vec<u8>> = BTreeMap::new();
        for msg in &got {
            if let Inbound::Wire { class, bytes, .. } = msg {
                by_class.insert(*class, bytes.to_vec());
            }
        }
        assert_eq!(by_class.get(&MsgClass::Control), Some(&vec![0xC0]));
        assert_eq!(by_class.get(&MsgClass::Saga), Some(&vec![0x5A]));
        assert_eq!(got.len(), 2, "exactly two reliable frames, no duplication");
    }

    /// R-2b: a reliable frame whose FRAMED size exceeds the cap is REJECTED before retention (counted on
    /// `reliable_shed`, bounced `NodeUnreachable`), so it can never poison the lane. The payload is EXACTLY
    /// `MAX_STREAM_FRAME_BYTES` — IN the off-by-the-header poison window (the framed envelope pushes it over
    /// the cap) that the prior raw-payload `> MAX` check let slip through to wedge the lane (audit
    /// `wf_93fc5909`). The follow-up normal frame proves the lane is NOT poisoned — it still delivers — and
    /// exercises the reconnect-resets-an-existing-lane-stream path.
    #[test]
    fn an_oversize_reliable_frame_is_rejected_and_the_lane_survives() {
        let rt = runtime();
        let trust = ClusterTrust::generate("vd-mesh-test").expect("trust");
        let (mut nodes, controls) = cluster(rt.handle(), &trust, 2, 8);
        // Exactly MAX: raw len is NOT > MAX (the old buggy check passed it), but framed (envelope + codec
        // byte) IS over the cap, so the framed-size check rejects it.
        let at_cap =
            vd_sim::io::bytes(vec![0u8; vd_wire::framing::MAX_STREAM_FRAME_BYTES as usize]);
        nodes[0]
            .send(NodeId(2), MsgClass::Saga, at_cap)
            .expect("enqueued (the bound is on the framed size, not the queue)");
        let started = Instant::now();
        loop {
            if controls[0].stats().reliable_shed >= 1 {
                break;
            }
            assert!(
                started.elapsed() < DEADLINE,
                "the over-cap reliable frame was never shed-counted"
            );
            std::thread::sleep(Duration::from_millis(5));
        }
        // The caller learns it did not deliver (bounced).
        let bounced = wait_for(&mut nodes[0], |g| {
            g.iter()
                .any(|m| matches!(m, Inbound::NodeUnreachable { .. }))
        });
        assert!(
            bounced.iter().any(|m| matches!(
                m,
                Inbound::NodeUnreachable {
                    class: MsgClass::Saga,
                    ..
                }
            )),
            "over-cap reliable bounces NodeUnreachable for its class"
        );
        // THE LANE IS NOT POISONED: a normal Saga frame on the same lane still delivers exactly once.
        nodes[0]
            .send(NodeId(2), MsgClass::Saga, vec![0x77].into())
            .expect("normal frame enqueued");
        let got = wait_for(&mut nodes[1], |g| {
            g.iter().any(|m| matches!(m, Inbound::Wire { .. }))
        });
        assert!(
            got.iter().any(|m| matches!(
                m,
                Inbound::Wire { class: MsgClass::Saga, bytes, .. } if bytes.as_ref() == [0x77]
            )),
            "the lane survived the over-cap rejection: the next reliable frame delivers"
        );
    }

    /// R-2b: a mis-tuned reliability field is rejected LOUD at `spawn_mesh` boot — before the endpoint
    /// binds or any task spawns.
    #[test]
    fn spawn_mesh_rejects_a_zero_reliability_field() {
        let rt = runtime();
        let trust = ClusterTrust::generate("vd-mesh-test").expect("trust");
        let mut cfg = MeshConfig::new(
            NodeId(1),
            "127.0.0.1:0".parse().expect("addr"),
            BTreeMap::new(),
            8,
            0,
        );
        cfg.reliability.retry_buffer_max_bytes = 0; // below one max frame ⇒ invalid
        // `match` (not `expect_err`): the Ok type `(MeshTransport, MeshControl)` is not `Debug`. The Ok
        // arm panics without formatting, so the validation must fail loud BEFORE any endpoint binds.
        match spawn_mesh(rt.handle(), &trust, &cfg) {
            Err(ProdIoError::Tuning(_)) => {}
            Err(other) => panic!("expected a Tuning error, got {other:?}"),
            Ok(_) => panic!("a zero tuning field must be rejected at boot"),
        }
    }
}
