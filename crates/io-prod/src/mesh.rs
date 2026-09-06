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

use std::collections::{BTreeMap, BTreeSet};
use std::net::SocketAddr;
use std::sync::Arc;
use std::sync::Mutex;
use std::sync::RwLock;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Duration;

use arc_swap::ArcSwap;
use tokio::sync::{Notify, watch};
use tokio::task::JoinHandle;
use vd_core::{MsgId, NodeId};
use vd_sim::io::{
    // (`InboxDrop` is gone from this file: the mesh can no longer produce one. The unreliable lane
    // pushes through `BoundedInbox` and only counts, and the reliable lane cannot drop at all.)
    BoundedInbox,
    Bytes,
    Inbound,
    MsgClass,
    PeerResetCause,
    Reliability,
    SendError,
    ShedReason,
    Transport,
};

use crate::outbox::{OutboxKey, OutboxSink, SharedOutbox};
use crate::store::DurabilityHandle;
use crate::trust::ClusterTrust;
use crate::{
    AckEntry, AckFrame, DatagramFrame, OutFrame, ProdIoError, ReliableFrame, read_one_ack_frame,
    read_one_reliable_frame, write_ack_frame, write_reliable_frame,
};

/// The 1-byte tag consumed FIRST on every mesh uni stream (before the `wire::framing` loop; never inside
/// `frame_payload`). A DATA stream carries [`ReliableFrame`]s (the data sender → data receiver direction);
/// an ACK stream carries [`AckFrame`]s back on the SAME connection (data receiver → data sender). The
/// loopback bridge (lib.rs) is a SEPARATE transport that writes NO tag and never interoperates with the mesh.
const STREAM_KIND_DATA: u8 = 0x00;
const STREAM_KIND_ACK: u8 = 0x01;

/// The role of an accepted uni stream (CA-1 S1). A pure classifier of the 1-byte `STREAM_KIND` tag so the ONE
/// unified per-connection dispatcher has exactly one branch per role (HR5 per-arm coverage) and an unknown tag
/// is a straight drop — never a mis-route. Replaces the two ad-hoc `kind[0] != STREAM_KIND_*` checks that lived
/// in the (now-merged) `serve_data_stream` + `ack_reader_task`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum StreamKind {
    Data,
    Ack,
}

/// Classify a `STREAM_KIND` tag byte. `None` ⇒ unknown tag (drop the stream wholesale, ledger untouched).
fn stream_kind(tag: u8) -> Option<StreamKind> {
    if tag == STREAM_KIND_DATA {
        Some(StreamKind::Data)
    } else if tag == STREAM_KIND_ACK {
        Some(StreamKind::Ack)
    } else {
        None
    }
}

/// Registry of this node's DIALED (outbound) connections, keyed by dest peer (latest-wins on redial ⇒
/// bounded). [`MeshControl::drop_connections`] closes them all — the transient-blip lever (the endpoint
/// stays bound), DISTINCT from [`MeshControl::kill`] (which closes the endpoint). Closing connection C
/// tears down BOTH the DATA streams (this→peer) AND their reverse ACK stream on C, coherently.
type ConnRegistry = Arc<Mutex<BTreeMap<NodeId, quinn::Connection>>>;

/// CA-1 — the LEARNED-peer table (reply-on-connection). An UNBOOKED peer that dialed IN and whose return path
/// this node learned from the first authenticated frame on its accepted connection. Keyed by the sender-asserted
/// NodeId (latest-wins per NodeId; capped by `learned_peers_max`). DISTINCT from [`ConnRegistry`] (our DIALED
/// connections, the `drop_connections` lever): a `LearnedConn` is an ACCEPTED connection we reply back over. Each
/// entry carries the connection AND the per-connection ack watch receiver the accepted-conn dispatcher feeds, so
/// a lazily-spawned learned lane retires its unacked window over that same connection.
type LearnedPeers = Arc<Mutex<BTreeMap<NodeId, LearnedConn>>>;

/// One learned inbound peer (CA-1): the held accepted connection + the ack receiver for the reply lane.
#[derive(Clone)]
struct LearnedConn {
    conn: quinn::Connection,
    ack_rx: watch::Receiver<Option<AckFrame>>,
    /// The sender incarnation of the frame that learned this connection. A STRICTLY HIGHER incarnation
    /// means a NEW process speaks for the same node id, so its connection REPLACES this one even while
    /// this one still looks alive (the vanished-client wedge: a killed client re-logs in as a new
    /// process, and the reply must reach the new process, never the corpse).
    incarnation: u64,
}

/// CA-1 S2 — the shared, UPDATABLE peer-address book: `NodeId → SocketAddr`, seeded from `cfg.peers` at
/// `spawn_mesh`, read lock-free (copy-on-write) by every Dial-lane on each dial, and rcu-updated at runtime via
/// [`MeshControl::update_peer_addr`] (the orchestrator/provisioning push). This is the outbound L5 re-plumb: a
/// peer rescheduled to a new IP becomes dialable at its new addr on the next dial — no more once-captured
/// spawn-time addr. Same ArcSwap idiom as the gateway route table (no new dep).
type PeerTopology = Arc<ArcSwap<BTreeMap<NodeId, SocketAddr>>>;

/// The connection source for a `peer_writer` lane (CA-1). A BOOKED lane DIALS the peer's CURRENT address, re-read
/// from the shared [`PeerTopology`] on every dial (CA-1 S2 — never a spawn-time copy); a LEARNED lane NEVER dials
/// — it adopts the held accepted connection the receive side recorded in `LearnedPeers` (its dispatcher +
/// ack_egress are already owned by that connection's `serve_connection`).
enum ConnSource {
    Dial(PeerTopology),
    Learned(LearnedPeers),
}

/// CA-1 — the context the accept-path dispatcher threads to the DATA reader so it can learn an unbooked dial-in
/// peer's return connection on the first authenticated frame. Absent (`None`) on a DIALED connection's dispatcher
/// (a dialer never learns — only the acceptor of an unbooked peer does).
#[derive(Clone)]
struct LearnCtx {
    learned: LearnedPeers,
    booked: Arc<BTreeSet<NodeId>>,
    local: NodeId,
    cap: usize,
    /// The accepted connection to record (this dispatcher's own connection).
    conn: quinn::Connection,
    /// The per-connection ack receiver a learned lane subscribes to (paired with the dispatcher's `ack_out`).
    ack_rx: watch::Receiver<Option<AckFrame>>,
    stats: Arc<MeshStats>,
}

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
    /// ★ THE WORLD THIS PROCESS SERVES, as the generation folded over its shape constants.
    ///
    /// It rides on the transport handshake beside the coordinate unit, and two nodes that disagree about
    /// EITHER cannot connect at all. The unit alone was not enough: two builds can count in the same
    /// millimetres and still disagree about how big the galaxy is, and then they shake hands, decode every
    /// message cleanly, and put the same player in two different places. Nothing crashes. Nothing logs.
    ///
    /// Supplied by the boot, because the world's shape lives in the world generator and this crate cannot
    /// reach it — which is the same reason the saved-data label takes it as an argument.
    pub world_generation: u64,
    /// The at-least-once redelivery tuning (R-2b). `validate`d loud at `spawn_mesh` boot.
    pub reliability: MeshReliabilityTuning,
    /// CA-1 — the cap on the LEARNED-peer table (reply-on-connection): how many distinct UNBOOKED dial-in peers
    /// this node will remember return-connections for. Bounds a churn/flood memory vector on the trusted
    /// static-roster tier (a new learned peer past the cap is rejected + counted via
    /// [`MeshStats::learned_peers_rejected`], never silently). M6 evicts a vanished unbooked peer's RECEIVE
    /// LEDGER ([`evict_recv_ledger`]); this TABLE is still only replaced-in-place, never emptied, so a churny
    /// tier still fills it to the cap. Default [`DEFAULT_LEARNED_PEERS_MAX`].
    pub learned_peers_max: usize,
}

/// Default learned-peer table cap (CA-1). Generously above the static-roster tier; raise (or add per-peer
/// stores) only with a real churn need — never speculatively.
pub const DEFAULT_LEARNED_PEERS_MAX: usize = 4096;

/// Default per-lane unacked-retry-buffer ceiling (R-4' shed point; counted in R-2b). 4 MiB.
pub const DEFAULT_RETRY_BUFFER_MAX_BYTES: usize = 4 * 1024 * 1024;
/// Default cadence at which a receiver flushes a lone cumulative ack (R-3'); a lone damage/death/
/// despawn is acked within this even without follow-up traffic. 50 ms.
pub const DEFAULT_ACK_IDLE_FLUSH: Duration = Duration::from_millis(50);
/// Default replays-before-a-hard-`NodeUnreachable`-bounce (R-4'): a transient blip costs ZERO bounce;
/// only a peer that fails this many consecutive replays is declared unreachable.
pub const DEFAULT_CONFIRM_UNREACHABLE_AFTER_RETRIES: u32 = 3;
/// The peer-writer redial backoff bounds — the FIRST failed dial re-arms at `MIN`, each subsequent failure
/// doubles, capped at `MAX`. ONE home (cloud-ready k3d Slice 2 hoisted these out of the inline `MeshConfig::new`
/// literals) so the cloud liveness window (`LivenessTuning::cloud`, which must dominate the confirmed-dead run
/// spread these bounds define — the R-4c cross-check) derives from the SAME source the mesh actually dials with.
pub const DEFAULT_REDIAL_BACKOFF_MIN: Duration = Duration::from_millis(50);
/// See [`DEFAULT_REDIAL_BACKOFF_MIN`].
pub const DEFAULT_REDIAL_BACKOFF_MAX: Duration = Duration::from_secs(5);

/// The peer-address auto-resolver cadence — the missing PRODUCTION caller of [`MeshControl::update_peer_addr`].
/// A background task re-resolves each booked peer's DNS name this often and pushes a CHANGED address, so a
/// rescheduled peer (new pod IP, SAME DNS name) is re-plumbed for INITIATED traffic. Deliberately a FEW HUNDRED
/// ms — WELL under the cloud saga-liveness confirmed-dead window (`LivenessTuning::cloud` ~1.5 s @ 50 Hz) so the
/// re-plumb lands before a moved-but-recoverable peer is declared dead; NOT a multiple of the redial backoff
/// (which is deliberately larger). It is a FIXED const (not operator-tunable), so this coherence holds BY
/// CONSTRUCTION — there is NO boot cross-check (`vd-bins` `spawn_peer_resolver_if_configured` records "not a
/// tunable, so no boot cross-check is owed"); a future move to an operator-tunable interval must add one.
pub const DEFAULT_PEER_RERESOLVE_INTERVAL: Duration = Duration::from_millis(500);
/// A short settle after boot before the FIRST re-resolve: the entrypoint already resolved the initial addrs,
/// so the first tick is a no-op on an unchanged cluster; this just avoids racing that boot resolution.
pub const DEFAULT_PEER_RERESOLVE_INITIAL_DELAY: Duration = Duration::from_secs(1);

/// Peer-address auto-resolver tuning — the ONE config home (no inline literals at use sites), mirroring
/// [`MeshReliabilityTuning`]. Plain (not `Serialize`): operational tuning, never persisted. A FAILED resolve is
/// simply retried at the next `interval` (no separate failure backoff — the interval already paces the resolves
/// far below any CoreDNS-hammering rate).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PeerResolveTuning {
    /// How often the auto-resolver re-resolves each peer's DNS name. Must be > 0 (a 0 interval busy-spins).
    pub interval: Duration,
    /// A settle after boot before the first re-resolve.
    pub initial_delay: Duration,
}

impl Default for PeerResolveTuning {
    fn default() -> Self {
        PeerResolveTuning {
            interval: DEFAULT_PEER_RERESOLVE_INTERVAL,
            initial_delay: DEFAULT_PEER_RERESOLVE_INITIAL_DELAY,
        }
    }
}

/// A `PeerResolveTuning` field out of range — surfaced loud at boot.
#[derive(Clone, Copy, Debug, PartialEq, Eq, thiserror::Error)]
pub enum PeerResolveTuningError {
    #[error("peer re-resolve interval must be > 0")]
    IntervalZero,
}

impl PeerResolveTuning {
    /// Reject an out-of-range field (fail-loud at boot). Equality-comparable typed error (HR5(d)).
    ///
    /// # Errors
    /// [`PeerResolveTuningError::IntervalZero`] if the interval is zero (a 0 timer busy-spins).
    pub fn validate(&self) -> Result<(), PeerResolveTuningError> {
        if self.interval.is_zero() {
            return Err(PeerResolveTuningError::IntervalZero);
        }
        Ok(())
    }
}

/// At-least-once redelivery tuning — the ONE config home for the redelivering transport (no inline
/// literals at use sites). Plain (not `Serialize`): operational tuning, never persisted.
///
/// CONSUMER NOTE: all three are now LIVE — `retry_buffer_max_bytes` bounds the retry buffer (R-4b BufferFull
/// shed), `ack_idle_flush_interval` paces the cumulative-ack flush (R-3' `ack_egress`), and
/// `confirm_unreachable_after_retries` is the blip-tolerance threshold `confirm_and_maybe_bounce` (R-4a) gates
/// the `NodeUnreachable` bounce on (a blip that recovers before N ⇒ zero bounce). R-4c cross-validates this
/// threshold + the redial backoff against the orchestrator saga liveness window at boot
/// (`LivenessTuning::validate_against`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct MeshReliabilityTuning {
    /// Per-lane unacked-retry-buffer byte ceiling (R-4b, the shed point). Must be at least one MAXIMAL FRAMED
    /// frame (`MAX_STREAM_FRAME_BYTES` `total_len` + the 4-byte length prefix) or a single large reliable frame
    /// could never be retained — the byte accounting is on the on-wire `framed_len`, not the raw payload.
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
    #[error(
        "retry_buffer_max_bytes must be >= one maximal framed frame (max stream frame + 4-byte prefix)"
    )]
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
        // The cap is on the ON-WIRE `framed_len` = `total_len` (<= MAX_STREAM_FRAME_BYTES) + the 4-byte length
        // prefix, so one MAXIMAL framed frame is `MAX + size_of::<u32>()`. Requiring at least that keeps R-4b
        // from BufferFull-rejecting a single max frame on an empty buffer (review aa95e10c off-by-envelope).
        if self.retry_buffer_max_bytes
            < vd_wire::framing::MAX_STREAM_FRAME_BYTES as usize + std::mem::size_of::<u32>()
        {
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
        world_generation: u64,
    ) -> MeshConfig {
        MeshConfig {
            local,
            bind,
            peers,
            outbound_capacity,
            inbound_capacity: inbound_capacity_for(outbound_capacity),
            max_inbound_connections: 256,
            redial_backoff_min: DEFAULT_REDIAL_BACKOFF_MIN,
            redial_backoff_max: DEFAULT_REDIAL_BACKOFF_MAX,
            process_incarnation,
            world_generation,
            reliability: MeshReliabilityTuning::default(),
            learned_peers_max: DEFAULT_LEARNED_PEERS_MAX,
        }
    }
}

/// ★ THREE QUEUES, SPLIT BY WHO IS ALLOWED TO LOSE WHAT (2026-08-29).
///
/// There was ONE bounded queue and it discarded a RELIABLE frame when full. That drop is
/// unrecoverable on this transport — the sender replays a lane only when its stream is GONE
/// (`owes_redelivery`), and an inbox drop leaves the stream healthy — so the receiver's watermark
/// stops advancing and the two nodes talk past a hole forever.
///
/// MEASURED on a live cluster: ONE drop during the gateway's boot, then a contiguity-gap warning
/// roughly four hundred times every eight seconds until the process died. A player could never enter
/// the world.
///
/// ★ AND THE QUEUE WAS NOT SLOW — IT WAS UNATTENDED. The accept loop starts before the gateway folds
/// the galaxy's sky, and the ONLY drain is the first tick, some fifty seconds later. Two thousand
/// slots fill long before anyone reads.
///
/// The lanes are separated because their correct overflow policies are OPPOSITE, and one queue cannot
/// hold both:
///
/// - RELIABLE — the reader WAITS for room. Nothing is ever lost. Not reading is precisely how QUIC's
///   flow control is applied: the frame stays inside the transport, which is the standard place for
///   it, and the sender is slowed without anyone asking.
/// - UNRELIABLE — unchanged. Latest-wins dropping is CORRECT here; next tick's position beats a
///   resend of last tick's, and a reader that waited would stall a carrier whose whole contract is
///   that loss is fine.
/// - LOCAL NOTICES — this node telling itself something (a peer is unreachable, a send was shed).
///   Neither waits nor drops: waiting would stall this node's own sender against itself, and dropping
///   would strand a saga. They get their own room so a bounce storm cannot starve a login, which it
///   can today.
pub(crate) struct InboundQueues {
    /// Reliable wire frames. Producers reserve a slot and wait when it is full.
    reliable_tx: tokio::sync::mpsc::Sender<Inbound>,
    reliable_rx: Mutex<tokio::sync::mpsc::Receiver<Inbound>>,
    /// Unreliable datagrams — the existing bounded queue, with the existing latest-wins policy.
    unreliable: Mutex<BoundedInbox>,
    /// This node's own notices to itself.
    notices_tx: tokio::sync::mpsc::Sender<Inbound>,
    notices_rx: Mutex<tokio::sync::mpsc::Receiver<Inbound>>,
}

impl InboundQueues {
    fn new(capacity: usize) -> Self {
        let (reliable_tx, reliable_rx) = tokio::sync::mpsc::channel(capacity.max(1));
        // A node's own notices are bounded by its peer count and its own backoff, so a small room is
        // ample — and a FULL one is a defect worth shouting about, not a runtime condition.
        let (notices_tx, notices_rx) = tokio::sync::mpsc::channel(capacity.max(1));
        Self {
            reliable_tx,
            reliable_rx: Mutex::new(reliable_rx),
            unreliable: Mutex::new(BoundedInbox::new(capacity)),
            notices_tx,
            notices_rx: Mutex::new(notices_rx),
        }
    }
}

/// The shared inbound path: reader tasks push, the sim thread drains.
type SharedInbox = Arc<InboundQueues>;

/// Transport honesty counters — every dropped datagram is COUNTED, never silent
/// (audit GW-1: the design's never-silent rule). Surfaced via [`MeshControl::stats`].
#[derive(Debug, Default)]
pub struct MeshStats {
    /// ★ WHY THE LAST DIAL TO EACH PEER FAILED, and the tag we offered it (slice S3's fleet half).
    ///
    /// A failed dial used to discard its reason twice (`map_err(|_| ())`) and retry forever. So two
    /// nodes that count positions in different units produced an ENDLESS RETRY WITH NO LOG LINE
    /// ANYWHERE: the cluster never formed and nothing said why. The refusal is correct — the transport
    /// reports `no_application_protocol`, which carries no field, no value and no unit — but a refusal
    /// nobody can read is indistinguishable from a network that is simply down.
    ///
    /// Held per peer so the alarm fires ONCE per distinct reason and re-arms when the reason changes or
    /// the peer connects. An error line per retry is not an alarm; it is how a real alarm gets muted by
    /// the people who most need to see it — the lesson the wedged-hand-off alarm already learned.
    pub dial_failure: Mutex<BTreeMap<NodeId, String>>,
    /// ★ THE TAG THIS NODE OFFERS ON EVERY HANDSHAKE — the coordinate unit AND the world's own shape.
    ///
    /// Computed once at spawn and carried here rather than threaded through the dial path, because it is
    /// a fact about this node and every function on that path already holds these stats.
    ///
    /// The unit alone was not enough. Two builds can count in the same millimetres and still disagree
    /// about how big the galaxy is — and then they connect, decode every message cleanly, and put the
    /// same player in two different places. Nothing crashes and nothing is logged. Both halves ride the
    /// handshake now, so a disagreement about either refuses the connection instead.
    pub offered_tag: String,
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
    /// max stream frame size (un-framable ⇒ rejected, never retained, bounced `SendShed{Unframable}`
    /// — R-4d M3: NOT `NodeUnreachable`; a shed says nothing about peer liveness). R-4' adds the
    /// bounded-retry-buffer total shed (`SendShed{RetryBufferFull}`) on the same counter.
    pub reliable_shed: AtomicU64,
    /// SENDER: cumulative reliable frames RETIRED by an incoming cumulative ack (the retry buffer draining,
    /// R-3'). A value stuck at 0 while reliable traffic flows is a DEAD ACK PATH alert (the ack never
    /// reached this peer's writer) — the retry buffer would then grow unbounded until the R-4' shed.
    pub reliable_acked: AtomicU64,
    /// RECEIVER (CA-1): a new UNBOOKED dial-in peer was NOT learned because the learned-peer table is at
    /// `learned_peers_max` (a churn/flood guard on the static-roster tier). Must stay ~0 on a fixed roster; a
    /// nonzero value is a cap-too-low or peer-churn ALERT. Also warned loudly at the reject site.
    pub learned_peers_rejected: AtomicU64,
    /// RECEIVER (CA-1): a LIVE learned return connection was REPLACED because a strictly HIGHER process
    /// incarnation dialed in for the same node id — a restarted client whose old connection nobody reads
    /// any more. The old connection is closed at the same moment, so the count also says how many stale
    /// reply paths this node retired. A steady climb means peer churn, not a defect.
    pub learned_peers_superseded: AtomicU64,
    /// Writer lanes that retired quietly because their peer's NEWER connection had superseded the one
    /// they were bound to (never a `ConnectionLost` for a peer that is still there).
    pub learned_lanes_superseded: AtomicU64,
    /// RECEIVER (M6): receive ledgers EVICTED — a learned dial-in peer whose connection died and that has
    /// NO booked address, so this node can never be reached by that process again and its dedup state is
    /// dead weight. A booked peer (a shard, the orchestrator) is NEVER counted here: it is re-dialed and its
    /// flows resume against the SAME watermarks. A steady climb means client churn, which is the healthy
    /// shape; a value stuck at 0 while clients come and go is the leak this counter exists to show.
    pub recv_ledgers_evicted: AtomicU64,
    /// SENDER (the 2026-09-06 group commit): retained rows MIRRORED to the durable outbox. With
    /// [`MeshStats::outbox_batches`] it says how well the batching works — rows ÷ batches is the average
    /// number of frames that shared one disk sync. A ratio stuck at 1.0 under a burst means the drain is
    /// not grouping and every frame is paying its own barrier again.
    pub outbox_rows_retained: AtomicU64,
    /// SENDER: durability BARRIERS submitted — one per drained batch that staged anything durable, never
    /// one per frame. This is the count of disk syncs the retained lane asked for.
    pub outbox_batches: AtomicU64,
    /// The peer book: lanes DIALED LAZILY on a send toward a booked peer that had no lane yet.
    pub lazy_dials: AtomicU64,
    /// The peer book: addresses booked through `Transport::book_peer` (the node runtime's half).
    pub peers_booked: AtomicU64,
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
    pub reliable_acked: u64,
    pub learned_peers_rejected: u64,
    pub learned_peers_superseded: u64,
    pub learned_lanes_superseded: u64,
    pub recv_ledgers_evicted: u64,
    pub outbox_rows_retained: u64,
    pub outbox_batches: u64,
    pub lazy_dials: u64,
    pub peers_booked: u64,
}

/// ★ THE UNRELIABLE PUSH — unchanged policy, and the only place that still drops (2026-08-29).
///
/// Latest-wins is CORRECT on this lane: next tick's position beats a resend of last tick's, and a
/// reader that waited here would stall a carrier whose whole contract is that loss is fine. Counted,
/// never silent.
pub(crate) fn push_unreliable(inbox: &SharedInbox, stats: &MeshStats, event: Inbound) {
    let dropped = inbox
        .unreliable
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
        .push(event);
    if dropped.is_some() {
        stats
            .inbound_dropped_unreliable
            .fetch_add(1, Ordering::Relaxed);
    }
}

/// ★ THIS NODE'S OWN NOTICE TO ITSELF — neither waits nor drops (2026-08-29).
///
/// A node cannot apply back-pressure to itself: waiting here would stall the very sender the notice
/// is about. And dropping would strand a saga waiting to hear that a peer is gone. So the notices get
/// their OWN room, and a full one is a DEFECT rather than a runtime condition — the rate is bounded
/// by the peer count and the writer's own backoff.
///
/// They used to share the wire's queue, where a bounce storm could starve a login.
pub(crate) fn push_notice(inbox: &SharedInbox, stats: &MeshStats, event: Inbound) {
    if inbox.notices_tx.try_send(event).is_err() {
        stats
            .inbound_dropped_reliable
            .fetch_add(1, Ordering::Relaxed);
        tracing::error!(
            "the local-notice queue is FULL — this node could not tell itself something. The rate is \
             bounded by the peer count and the writer backoff, so this is a defect, not overload."
        );
    }
}

/// The receiver dedup state for ONE `(peer, class)` — pure, `Copy`, the unit-testable core of the R-3'
/// contiguity verdict. `primed=false` distinguishes "hw=0, nothing delivered yet" from "delivered seq0
/// (hw=0)". Held in the node-wide [`RecvLedger`] that SURVIVES connection teardown — the SSOT that makes
/// the cross-stream cure work (a straggler on an old stream can never corrupt a fresh stream's watermark).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
struct RecvState {
    /// Sender process-incarnation. A HIGHER value ⇒ the sender restarted ⇒ RESET (the sender-restart
    /// seq-reset cure); a LOWER value ⇒ a straggler from a since-restarted sender ⇒ drop.
    incarnation: u64,
    /// Per-(peer,class) redial counter. A LOWER value ⇒ an old-stream straggler ⇒ `StaleEpoch` BEFORE any
    /// hw compare (the reverted-R-1 CRITICAL cure); a HIGHER value ⇒ adopt-forward WITHOUT resetting hw.
    epoch: u32,
    /// Highest CONTIGUOUS seq DELIVERED to the inbox; next expected = `hw + 1`.
    hw: u64,
    /// False until the first frame of this incarnation is delivered (so a first-frame seq>0 after a reset
    /// surfaces as a Gap, never a silent Dedup of a lower never-delivered seq).
    primed: bool,
}

/// The node-wide receiver ledger: one [`RecvState`] per `(peer, class)`, shared by ALL `serve_connection`
/// readers across ALL connections/redials. Keyed FOREVER — never per-connection/per-stream (that WAS the
/// reverted-R-1 bug). Created once in [`spawn_mesh`].
///
/// R-4e2 (H2 RX-plane re-key): keyed PER-PEER at the outer level so a frame from peer P no longer serializes
/// against a frame from peer Q (the RX twin of the per-peer SEND lanes). The OUTER [`RwLock`] is shared-read
/// on the steady-state hot path (fetch a peer's inner `Arc`); it is write-locked ONLY on the rare first frame
/// from a never-seen peer (insert the peer's inner map). The INNER per-peer [`Mutex`] serializes only frames
/// from the SAME peer — and that serialization is LOAD-BEARING: it is what makes the `StaleEpoch`-before-hw
/// verdict + the reliable-inbox-drop rollback (`*st = before`) atomic against a concurrent same-`(peer,class)`
/// frame (two `serve_data_stream` tasks for one `(peer,class)` genuinely overlap across a redial — quinn's
/// close is async). LOCK ORDER (the only one; never reversed): outer-read → (drop) → inner-`Mutex` →
/// (optionally, in `classify_and_deliver`) the inbox `Mutex`. `ack_egress` takes the ledger, NEVER the inbox;
/// `acked_keys` is always acquired-then-released ABOVE (never inside) an inner-`Mutex` hold. RESIDUAL: after
/// this re-key the node-wide `SharedInbox` `Mutex` is the NEXT RX serialization point (a future per-peer-inbox
/// / lock-free-drain scaling slice — DEFERRED.md); the ledger re-key alone does NOT deliver full RX isolation.
///
/// ★ REMOVAL (M6). The OUTER write lock is taken for exactly two reasons and no others: INSERT a never-seen
/// peer's inner map ([`classify_and_deliver`]), and REMOVE a vanished unbooked peer's whole entry
/// ([`evict_recv_ledger`]). Never nested inside an inner peer `Mutex`, in either direction. A booked peer (a
/// shard, the orchestrator) is never removed — it is re-dialed and its flows resume against the same
/// watermarks. A client that dialed in and died IS removed, because nothing can reach that process again.
///
/// KNOWN RESIDUAL: eviction rides the learned lane's death, so it fires only for a peer this node actually
/// REPLIED to (which is what creates the lane). A dial-in peer that only ever SENDS — it talks, it is never
/// answered, it vanishes — leaves its entry behind. The gateway answers every client it holds a session for,
/// so this is not the live shape; a send-only peer tier would need the accept side to evict as well.
type RecvLedger = Arc<RwLock<BTreeMap<NodeId, Arc<Mutex<PeerRecv>>>>>;

/// Everything the receiver remembers about ONE peer, under that peer's inner lock: the per-class dedup
/// states AND the highest process incarnation any class has seen from it.
///
/// The incarnation lives HERE, beside the classes rather than inside each of them, because a peer restarts
/// ONCE — not once per class. A restarted client sends login on Control and input on Input; both frames
/// carry the new incarnation, and the consumer must be told the session is over exactly once.
struct PeerRecv {
    /// The highest sender incarnation seen from this peer on ANY class. `None` until the first frame ever,
    /// so a first contact is never mistaken for a restart.
    incarnation: Option<u64>,
    /// The per-class dedup states (the pre-existing inner map, unchanged in meaning).
    classes: BTreeMap<MsgClass, RecvState>,
}

/// The contiguity verdict for one reliable frame. Distinct arms (not a bool) so each is equality-asserted
/// in unit tests (HR5(d)) and counted on its own never-silent [`MeshStats`] counter.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Verdict {
    /// Deliver: contiguous advance (`seq == hw+1`) or a fresh-entry prime.
    Accept,
    /// Deliver: the FIRST frame of a higher incarnation (the sender restarted). Distinct from `Accept`
    /// only for the counter/coverage; delivered identically.
    Reset,
    /// Drop: a straggler from a since-restarted sender (`inc < recorded`). hw untouched.
    StaleIncarnation,
    /// Drop: a straggler on an OLD stream (`epoch < recorded`), dropped BEFORE the hw compare — the
    /// cross-stream-race cure. hw untouched.
    StaleEpoch,
    /// Drop: an expected replay (`seq <= hw`) — the sim never sees a dup. hw untouched.
    Dedup,
    /// Drop: a contiguity GAP (`seq > hw+1`) — a MUST-BE-0 alert (the wire epoch keeps it genuinely 0 in a
    /// healthy run; a reset-window reorder or a mid-window inbox-drop can surface it transiently). hw untouched.
    Gap,
}

/// THE receiver verdict ladder — pure, tokio-free, fully unit-testable. Mutates `st` in place and returns
/// the [`Verdict`]. The order (incarnation ▸ epoch ▸ seq) is load-bearing: incarnation dominates epoch
/// dominates seq. STALE-EPOCH is decided BEFORE any hw compare — the reverted-R-1 CRITICAL cure (a6f8e7d /
/// 731b377). For `Accept`/`Reset` it advances `hw` (to `seq`) + sets `primed`; the CALLER captures `st.hw`
/// under the SAME lock for the ack watermark, and ROLLS the whole state back if the inbox then drops the
/// frame (advancing hw without delivering = permanent loss).
fn classify_reliable(st: &mut RecvState, incarnation: u64, epoch: u32, seq: u64) -> Verdict {
    // A1: higher incarnation ⇒ the sender restarted. Reset to the NOT-PRIMED shape and FALL THROUGH so the
    // first frame primes via the seq0 gate — NEVER set hw=seq here (that would bury a lower late frame: the
    // reverted CRITICAL's twin). A reset-window redial reorder then surfaces as Gap, never a silent Dedup.
    if incarnation > st.incarnation {
        *st = RecvState {
            incarnation,
            epoch,
            hw: 0,
            primed: false,
        };
        return prime_or_contiguous(st, seq, true);
    }
    // A2: lower incarnation ⇒ a straggler from a since-restarted sender. NEVER touch hw.
    if incarnation < st.incarnation {
        return Verdict::StaleIncarnation;
    }
    // A3: THE CURE — an older epoch is dropped WITHOUT consulting hw (an old-stream straggler can never
    // advance the watermark past a never-delivered low).
    if epoch < st.epoch {
        return Verdict::StaleEpoch;
    }
    // A4: a newer epoch ⇒ adopt-forward. hw is CARRIED (a redial replays base..; base == hw+1 keeps it
    // contiguous), NEVER reset (the bug one candidate design made).
    if epoch > st.epoch {
        st.epoch = epoch;
    }
    prime_or_contiguous(st, seq, false)
}

/// Monomorphic helper (keeps `classify_reliable`'s branching coverable per-arm; HR5(a)). Primes ONLY at
/// seq0 — a restarted sender ALWAYS resets `next_seq` to 0, so a fresh-incarnation first-frame seq>0 is a
/// genuine reset-window loss/reorder (Gap), never a prime. A brand-new ENTRY (not a restart) may legitimately
/// adopt-forward from a sender base>0 (documented: trust the sender's delivery frontier).
fn prime_or_contiguous(st: &mut RecvState, seq: u64, fresh_incarnation: bool) -> Verdict {
    if !st.primed {
        if seq == 0 {
            st.hw = 0;
            st.primed = true;
            return if fresh_incarnation {
                Verdict::Reset
            } else {
                Verdict::Accept
            };
        }
        if !fresh_incarnation {
            // adopt-forward: a brand-new entry whose first observed frame is seq>0.
            st.hw = seq;
            st.primed = true;
            return Verdict::Accept;
        }
        // fresh incarnation + first frame seq>0 ⇒ a reset-window loss/reorder. VISIBLE, never primed.
        return Verdict::Gap;
    }
    if seq == st.hw + 1 {
        // Contiguous advance. `fresh_incarnation` is always false here (a fresh incarnation is never primed
        // at this point — it routes through the prime branch above), so this arm is unconditionally Accept.
        st.hw = seq;
        return Verdict::Accept;
    }
    if seq <= st.hw {
        return Verdict::Dedup;
    }
    Verdict::Gap
}

/// Did this peer RESTART? Pure, monomorphic, unit-testable (HR5(a)) — the peer-level twin of the per-class
/// incarnation arm in [`classify_reliable`].
///
/// Records `incoming` as the peer's high-water incarnation and answers whether it is a RESTART. A first
/// contact (`None`) is recorded and is NOT a restart: nobody held a session for a peer never heard from.
/// An equal incarnation is the same process still talking; a lower one is a straggler from a process this
/// node has already replaced.
fn peer_restarted(seen: &mut Option<u64>, incoming: u64) -> bool {
    match *seen {
        None => {
            *seen = Some(incoming);
            false
        }
        Some(prev) if incoming > prev => {
            *seen = Some(incoming);
            true
        }
        Some(_) => false,
    }
}

/// R-6d4-A: expose the REAL receiver dedup ladder to the outbox both-ends-restart replay proptest via an
/// OPAQUE `RecvCell` wrapper — its `DedupLedger` drives the SAME [`classify_reliable`] the mesh receiver
/// uses, so a dedup/incarnation-reset mutation turns the proptest RED (not a hand-rolled dedup that could
/// silently drift from production). Test-only (`#[cfg(test)]`), no release surface. A child mod may `use`
/// its parent's private `classify_reliable`/`RecvState`/`Verdict` (no re-export, no visibility widening).
#[cfg(test)]
pub(crate) mod recv_test_hooks {
    use super::{RecvState, Verdict, classify_reliable};

    /// One opaque per-`(peer, class)` receiver dedup cell wrapping the REAL [`RecvState`] + ladder.
    pub(crate) struct RecvCell(RecvState);
    impl RecvCell {
        /// A fresh (never-primed) cell — the shape the real ledger inserts on a first frame.
        #[must_use]
        pub(crate) fn fresh() -> Self {
            RecvCell(RecvState {
                incarnation: 0,
                epoch: 0,
                hw: 0,
                primed: false,
            })
        }
        /// Deliver one frame through the REAL verdict ladder; `true` iff a FIRST delivery (`Accept`/`Reset`),
        /// `false` for a drop (`Dedup`/`StaleIncarnation`/`StaleEpoch`/`Gap`).
        pub(crate) fn accept(&mut self, incarnation: u64, epoch: u32, seq: u64) -> bool {
            matches!(
                classify_reliable(&mut self.0, incarnation, epoch, seq),
                Verdict::Accept | Verdict::Reset
            )
        }
    }
}

/// One reliable send LANE — the per-(peer,class) at-least-once sender FSM (R-2b). Lazily created on
/// the first reliable frame for a class to a peer. ALL FSM logic (seq assign, retain, replay framing,
/// ack-retire) is SYNCHRONOUS + unit-testable WITHOUT tokio/quinn; only the stream open/write (in
/// `write_staged_batch`) is async. One stream per class so a stalled class never head-of-line-blocks another.
/// R-4b: a retained frame + its FROZEN worst-case-epoch framed length. The on-wire varint for `epoch` grows
/// from 1 byte (epoch 0) to 5 bytes (>= 2^28) as `replay_batch` re-stamps on each redial, so the framed
/// length of a retained frame CHANGES over its life. `framed_len` is the length at `epoch=u32::MAX` (the same
/// `encode_frame` output `assign_and_retain` already produced for the oversize cap check), so it UPPER-BOUNDS
/// every future re-stamp — `retry_bytes` can never drift below the true on-wire size, and the cap check + the
/// accounting share ONE number (the H3 cure).
struct RetainedFrame {
    frame: ReliableFrame,
    framed_len: u32,
    /// R-6d2c: was this frame write-through-mirrored to the durable outbox (`durable && sink.is_some()` at
    /// assign)? `on_ack` releases the outbox key ONLY for `retained` frames — so the outbox is a STRICT
    /// SUBSET of `retry` (the Retained frames). Keyed by seq/incarnation (epoch-independent), so an epoch
    /// bump never orphans a row.
    retained: bool,
}

/// R-4b: why `assign_and_retain` refused to retain a reliable frame. Both are shed-loud (`reliable_shed` +
/// a `SendShed{reason}` bounce — R-4d M3: NOT `NodeUnreachable`; a shed says nothing about peer liveness);
/// nothing is retained, so neither can poison the lane.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum AssignReject {
    /// The FRAMED frame exceeds `MAX_STREAM_FRAME_BYTES` — a PERMANENT reject (never sendable).
    Unframable,
    /// The per-lane retry buffer is at `retry_buffer_max_bytes` — producer backpressure. The receiver is
    /// strictly contiguous, so a retained frame can NEVER be safely shed (dropping any seq wedges/loses the
    /// window); the only loss-safe shed is to refuse the NEW send. A full buffer means acks stopped (a dead
    /// ack path) — `reliable_acked` stuck-at-0 is the corroborating alarm.
    BufferFull,
}

struct ReliableLaneSender {
    /// THIS lane's QUIC uni stream. `None` until the first send opens it; reset to `None` on any write
    /// error (re-opened lazily on the next send after the peer re-dial).
    stream: Option<quinn::SendStream>,
    /// R-6d2c: the DEST peer this lane sends to (the ConnRegistry key = `dest`/`w.dest`, NOT
    /// `ReliableFrame.from` which is the LOCAL sender). Constant for the lane's life. Supplies
    /// `OutboxKey.peer` for the durable write-through/delete-through — `on_ack` has no `NodeId` in scope
    /// otherwise. Class is NOT stored (it lives in every `RetainedFrame.frame.class`; the per-class-lane
    /// invariant makes them equal, so `on_ack` reads `rf.frame.class`).
    peer: NodeId,
    /// The sender's process-incarnation, stamped on every frame (constant for the lane's life). R-3'
    /// reads it to reset the receiver's dedup state when the sender restarts.
    incarnation: u64,
    /// Per-lane redial counter, bumped on EVERY write error (the cross-stream-race cure: an OLD-stream
    /// straggler carries the OLD epoch ⇒ R-3' drops it BEFORE it advances the high-water). Starts 0.
    epoch: u32,
    /// Next seq to ASSIGN — monotone, NEVER reset, NEVER rolled back on a failed write (buffer-first ⇒
    /// a burned write leaves no gap and re-uses no seq).
    next_seq: u64,
    /// Unacked frames keyed by seq. `BTreeMap` ⇒ ascending key = ascending seq = exact replay order; each
    /// value is a [`RetainedFrame`] (the fully-built [`ReliableFrame`], which replay re-stamps `epoch` on, +
    /// its frozen worst-case framed length). R-3' drains it via [`ReliableLaneSender::on_ack`] as cumulative
    /// acks arrive; R-4b caps its byte total at `retry_buffer_max_bytes` (`retry_bytes`).
    retry: BTreeMap<u64, RetainedFrame>,
    /// R-4b: the sum of every retained frame's `framed_len` — kept in LOCKSTEP with `retry` (`assign_and_retain`
    /// adds, `on_ack` subtracts the SAME number). When it would exceed `retry_cap` a new send is refused
    /// (`AssignReject::BufferFull` ⇒ shed-loud) — producer backpressure, never a retained-frame drop.
    retry_bytes: usize,
    /// R-4b: this lane's retry-buffer byte ceiling (`MeshReliabilityTuning::retry_buffer_max_bytes`, threaded
    /// in at construction). Constant for the lane's life; the `assign_and_retain` BufferFull gate.
    retry_cap: usize,
    /// The sender's belief of the receiver's `hw + 1`: `retry` holds exactly `base..next_seq`. Starts 0.
    /// NEVER decreases (monotone — a lower/stale ack retires nothing). `on_ack` advances it as the acked
    /// prefix retires; `replay_batch` (via `retry.values()`) then starts at `base`, so the first replayed
    /// frame is always `<= receiver_hw + 1` (Accept-or-Dedup, never a Gap).
    base: u64,
    /// R-4a: consecutive FAILED replay CYCLES on THIS lane since its last successful (re)dial-and-write.
    /// Bumped by EXACTLY ONE `on_replay_failed` per lane whose OWN replay failed during a redial cycle (NEVER
    /// the connection-drop fan-out — a shared blip is ONE event, not N lane failures: the C2 cure), reset by
    /// `on_replay_ok` when THIS lane's own write/replay completes (per-lane, never a blanket for-all: the H2
    /// cure). A transient blip that recovers before `confirm_unreachable_after_retries` ⇒ ZERO bounce.
    consecutive_failures: u32,
    /// R-4a: the `MsgId` of the newest frame this lane sent — the `undelivered` id the threshold-gated
    /// `confirm_and_maybe_bounce` carries (the timer-driven bounce has no `OutFrame` to source one from).
    /// `None` until the lane sends its first frame (a never-sent lane never bounces).
    last_msg_id: Option<MsgId>,
}

impl ReliableLaneSender {
    fn new(peer: NodeId, incarnation: u64, retry_cap: usize) -> ReliableLaneSender {
        ReliableLaneSender {
            stream: None,
            peer,
            incarnation,
            epoch: 0,
            next_seq: 0,
            retry: BTreeMap::new(),
            retry_bytes: 0,
            retry_cap,
            base: 0,
            consecutive_failures: 0,
            last_msg_id: None,
        }
    }

    /// R-6d2c: the ONE `OutboxKey` constructor — `peer`/`incarnation` single-sourced from the lane, the two
    /// varying inputs (class, seq) passed explicitly. `retain` (in `assign_and_retain`) and `release` (in
    /// `on_ack`) build the SAME key iff they pass the same `(class, seq)`; the per-class-lane invariant makes
    /// `rf.frame.class == class`, so a release keys the exact row its retain wrote.
    fn outbox_key(&self, class: MsgClass, seq: u64) -> OutboxKey {
        OutboxKey {
            peer: self.peer,
            class,
            incarnation: self.incarnation,
            seq,
        }
    }

    /// R-4a: bump the consecutive-failure count (saturating) — called EXACTLY once per lane whose own
    /// replay attempt failed during a redial cycle, never from the connection-drop fan-out (the C2 cure).
    fn on_replay_failed(&mut self) {
        self.consecutive_failures = self.consecutive_failures.saturating_add(1);
    }

    /// R-4a: reset the consecutive-failure count — called ONLY for a lane whose own write/replay just
    /// completed Ok, never a blanket for-all reset (the H2 cure).
    fn on_replay_ok(&mut self) {
        self.consecutive_failures = 0;
    }

    /// R-4a: this lane OWES a redelivery iff its stream is closed AND its unacked window is non-empty. The
    /// retransmit-timer guard is `any lane owes` — a PER-LANE property, decoupled from connection state (the
    /// C1 cure: a sibling lane re-opening on a fresh send must not disarm the redelivery THIS lane still owes).
    /// A lane whose stream is OPEN but whose acks silently stopped does NOT owe here — that live-but-stuck
    /// dead-ack-path case is R-4b's shed job (`retry_bytes` cap + `reliable_acked`-stuck alarm), NOT the timer.
    fn owes_redelivery(&self) -> bool {
        self.stream.is_none() && !self.retry.is_empty()
    }

    /// Retire the acked prefix on a cumulative [`AckFrame`] entry (R-3'). Ignores an ack minted against a
    /// DIFFERENT incarnation (a since-restarted sender) or a since-superseded `epoch` — retiring on either
    /// could drop a frame the current lane still owes (the epoch guard is the finding-#4 base/hw-coupling
    /// cure). The retire loop is CLAMPED to `next_seq` (never remove a key we never assigned — defends a
    /// torn/duplicated/forged `ack_through`) and MONOTONE (`base` never rolls back). Release-safe: the
    /// clamp is real logic, not the `debug_assert` (which compiles out on the Tier-B stable build).
    /// Returns the number of frames RETIRED (0 for a stale/duplicate/lower ack) — the sender's
    /// `reliable_acked` observability (a stuck-at-0 counter is a dead ack path).
    fn on_ack(
        &mut self,
        ack_incarnation: u64,
        ack_epoch: u32,
        ack_through: u64,
        mut sink: Option<&mut dyn OutboxSink>,
    ) -> usize {
        if ack_incarnation != self.incarnation || ack_epoch != self.epoch {
            return 0;
        }
        let base_before = self.base;
        while self.base < self.next_seq && self.base <= ack_through {
            if let Some(rf) = self.retry.remove(&self.base) {
                // R-4b: keep `retry_bytes` in LOCKSTEP — subtract the SAME frozen framed_len assign added.
                self.retry_bytes -= rf.framed_len as usize;
                // R-6d2c: release ONLY a write-through-mirrored (durable) frame — an Ephemeral-heavy Saga
                // lane must NOT stage a tombstone per acked ephemeral seq. `class` from the retired frame
                // (per-class-lane invariant: `rf.frame.class` == this lane's class). `as_deref_mut` re-borrows
                // the `Option<&mut dyn>` so multiple releases in one ack are fine.
                if rf.retained
                    && let Some(s) = sink.as_deref_mut()
                {
                    s.release(&self.outbox_key(rf.frame.class, self.base));
                }
            }
            self.base += 1;
        }
        debug_assert!(self.base <= self.next_seq);
        (self.base - base_before) as usize
    }

    /// BUFFER-FIRST: stamp a NEW seq, build the frame at `(incarnation, epoch, seq)`, and — IF it frames
    /// within the stream cap AND retaining it stays within the per-lane byte cap — retain it + advance
    /// `next_seq`, returning the assigned SEQ. Returns without retaining or advancing `next_seq` on either
    /// reject (R-4b): `Unframable` if the framed `ReliableFrame` would exceed `MAX_STREAM_FRAME_BYTES` (the
    /// check is on the ENCODED frame — the envelope varints + codec byte can push a near-cap payload over —
    /// routed through the ONE codec/framing home `encode_frame`, HR3), or `BufferFull` if it would push
    /// `retry_bytes` over `cap` (producer backpressure — the receiver is strictly contiguous, so a retained
    /// frame can NEVER be safely shed). An un-framable frame must NEVER enter `retry` (it can never be sent
    /// ⇒ a permanent contiguity wall). There is exactly ONE frame object (in `retry`), which the write path
    /// reads back. Pure: no I/O; a write failure AFTER a successful assign never rolls it back (no burned seq).
    fn assign_and_retain(
        &mut self,
        from: NodeId,
        class: MsgClass,
        bytes: &[u8],
        durable: bool,
        sink: Option<&mut dyn OutboxSink>,
    ) -> Result<u64, AssignReject> {
        let seq = self.next_seq;
        // Build the frame stamped at the WORST-CASE epoch (`u32::MAX`, a 5-byte varint) for the framing
        // CHECK + the stored `framed_len`: `replay_batch` re-stamps a retained frame to an ever-HIGHER epoch
        // on each redial, and the epoch varint grows from 1 byte (epoch 0) to 5 bytes (>= 2^28). Encoding at
        // `u32::MAX` guarantees an ACCEPTED frame frames within the cap at EVERY future re-stamp (the
        // relocated-poison residual, audit `wf_93fc5909`) AND that `framed_len` UPPER-BOUNDS the on-wire size
        // for accounting (H3). The REAL epoch is set below before retaining; ONE `bytes` clone is reused.
        let mut frame = ReliableFrame {
            from,
            class,
            incarnation: self.incarnation,
            epoch: u32::MAX,
            seq,
            bytes: bytes.to_vec(),
        };
        // ONE encode, reused for BOTH the oversize reject AND the stored framed_len (H3 — no re-encode).
        let encoded =
            vd_wire::framing::encode_frame(&frame).map_err(|_| AssignReject::Unframable)?;
        let framed_len = encoded.len() as u32; // <= MAX_STREAM_FRAME_BYTES + envelope ⇒ fits u32
        // R-4b producer backpressure: refuse the NEW send if retaining it would exceed the byte cap. NEVER
        // shed a retained frame. `saturating_add` guards the (unreachable) usize overflow.
        if self.retry_bytes.saturating_add(framed_len as usize) > self.retry_cap {
            return Err(AssignReject::BufferFull);
        }
        frame.epoch = self.epoch; // the real epoch for retention + the first write
        // R-6d2c: mirror ONLY a Retained frame to the durable outbox — and only when a sink is wired (`None`
        // in prod until R-6d3). `retained` records the SAME predicate so `on_ack` releases exactly this
        // subset. `encoded` (the epoch=u32::MAX framed bytes from the check above) is the outbox VALUE (the
        // L2 seam contract: a full framed `ReliableFrame`; R-6d3 replay `decode_frame`s it + re-stamps epoch),
        // still owned here — the `frame` move below does NOT touch it. NO re-encode. Straight-line
        // monomorphic shim (HR5(a)): the only branch is the two-level `if let`; `retain` is a `dyn` call.
        // Placed BEFORE the RAM insert — an INTENTIONAL deviation from the design-of-record (wf_5ca0a751 §3
        // put it after): behaviour-identical (both reject arms already returned, so an un-retainable frame is
        // never mirrored; `encoded` is owned + read here before the `frame` move), and this order sidesteps the
        // move-then-borrow the design spent a paragraph reasoning about. `outbox ⊆ retained ⊆ never-shed` holds
        // regardless of order.
        let retained = durable && sink.is_some();
        if let Some(s) = sink
            && retained
        {
            s.retain(&self.outbox_key(class, seq), &encoded);
        }
        self.retry.insert(
            seq,
            RetainedFrame {
                frame,
                framed_len,
                retained,
            },
        );
        self.retry_bytes += framed_len as usize;
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
            .map(|rf| ReliableFrame {
                epoch: self.epoch,
                ..rf.frame.clone()
            })
            .collect()
    }
}

struct PeerLane {
    tx: tokio::sync::mpsc::Sender<OutFrame>,
    /// The learned-table incarnation this lane was spawned over. `None` for a BOOKED lane: it DIALS the
    /// peer's current address, so no learned entry governs it. `Some(n)` for a LEARNED lane: when the table
    /// later holds a HIGHER incarnation for the same peer, this lane writes into the previous process's
    /// connection and must be dropped, never used.
    learned_incarnation: Option<u64>,
}

/// The sim-thread side: implements [`Transport`] over per-peer bounded queues.
pub struct MeshTransport {
    local: NodeId,
    lanes: BTreeMap<NodeId, PeerLane>,
    inbox: SharedInbox,
    next_msg_id: u64,
    /// CA-1 — the bundle needed to LAZILY spawn a learned reply lane in `send_durable` (reply-on-connection to
    /// an unbooked dial-in peer). `stats` MUST be the SAME `Arc` `MeshControl::stats()` reads, or a learned
    /// lane's `reliable_acked` would be invisible. All are cheap `Arc`/`Copy` clones of the `spawn_mesh` state.
    handle: tokio::runtime::Handle,
    endpoint: quinn::Endpoint,
    learned: LearnedPeers,
    stats: Arc<MeshStats>,
    ledger: RecvLedger,
    connections: ConnRegistry,
    outbox: Option<SharedOutbox>,
    incarnation: u64,
    reliability: MeshReliabilityTuning,
    backoff_min: Duration,
    backoff_max: Duration,
    outbound_capacity: usize,
    /// ★ THE PEER BOOK on the sending side (D-RLM-6 mechanism C): the same address book the boot lanes
    /// dial from and `MeshControl::update_peer_addr` rewrites. `Transport::book_peer` writes it, and a
    /// send toward a booked peer that has no lane yet DIALS one — lazily, on that send — instead of
    /// refusing. A peer in neither the book nor the learned set is refused as `UnknownPeer`, which is
    /// what makes the node runtime ask.
    topology: PeerTopology,
}

/// Lifecycle handle: owns the endpoint (dropping closes it).
pub struct MeshControl {
    endpoint: quinn::Endpoint,
    stats: Arc<MeshStats>,
    /// This node's dialed (outbound) connections — the [`MeshControl::drop_connections`] blip lever.
    connections: ConnRegistry,
    /// CA-1 S2 — the shared updatable peer-address book (the same `Arc` every Dial-lane re-reads), refreshed by
    /// [`MeshControl::update_peer_addr`] (the orchestrator/provisioning push for a rescheduled peer).
    topology: PeerTopology,
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

    /// A transient CONNECTION blip (R-3'/R-5' lever): close every dialed connection this node holds — the
    /// endpoint STAYS BOUND, so the next reliable send re-dials, re-opens its lane streams, and REPLAYS its
    /// unacked window under a bumped epoch (at-least-once recovery). DISTINCT from [`MeshControl::kill`]
    /// (which closes the endpoint — a permanent death). Closing connection C tears down BOTH its DATA
    /// streams and their reverse ACK stream, so the epoch-bump + replay + ledger-survival story stays coherent.
    pub fn drop_connections(&self) {
        let mut reg = self
            .connections
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        for conn in reg.values() {
            conn.close(quinn::VarInt::from_u32(2), b"drop_connections blip");
        }
        reg.clear();
    }

    /// CA-1 S2 — the OUTBOUND L5 re-plumb: push a peer's CURRENT address into the shared topology (the
    /// orchestrator/provisioning tells this node where a (re)provisioned peer now lives), so this node can
    /// INITIATE to a peer it has never contacted or one that rescheduled to a new IP — no DNS, no static book
    /// edit. The rcu is copy-on-write (lock-free readers on the dial hot path). Any STALE dialed connection to
    /// that peer is proactively closed + forgotten so the peer_writer re-dials the new addr on its next attempt
    /// (rather than waiting out the idle timeout on the dead one). A booked peer's addr is updatable ONLY through
    /// this trusted control surface — never via an untrusted dial-in (which only ever populates `LearnedPeers`,
    /// the authority-split).
    pub fn update_peer_addr(&self, peer: NodeId, addr: SocketAddr) {
        self.topology.rcu(|cur| {
            let mut next = (**cur).clone();
            next.insert(peer, addr);
            next
        });
        let mut reg = self
            .connections
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        if let Some(conn) = reg.remove(&peer) {
            conn.close(quinn::VarInt::from_u32(3), b"peer addr updated");
        }
    }

    /// The peer's CURRENT dialed address in the shared topology. The auto-resolver seeds its
    /// change-detection baseline from this so the FIRST re-resolve is a no-op on an unchanged cluster (an
    /// unconditional first push would `update_peer_addr`-close every fine connection). `None` if unbooked.
    #[must_use]
    pub fn current_peer_addr(&self, peer: NodeId) -> Option<SocketAddr> {
        self.topology.load().get(&peer).copied()
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
            reliable_acked: self.stats.reliable_acked.load(Ordering::Relaxed),
            learned_peers_rejected: self.stats.learned_peers_rejected.load(Ordering::Relaxed),
            learned_peers_superseded: self.stats.learned_peers_superseded.load(Ordering::Relaxed),
            learned_lanes_superseded: self.stats.learned_lanes_superseded.load(Ordering::Relaxed),
            recv_ledgers_evicted: self.stats.recv_ledgers_evicted.load(Ordering::Relaxed),
            outbox_rows_retained: self.stats.outbox_rows_retained.load(Ordering::Relaxed),
            outbox_batches: self.stats.outbox_batches.load(Ordering::Relaxed),
            lazy_dials: self.stats.lazy_dials.load(Ordering::Relaxed),
            peers_booked: self.stats.peers_booked.load(Ordering::Relaxed),
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
    outbox: Option<SharedOutbox>,
) -> Result<(MeshTransport, MeshControl), ProdIoError> {
    // FAIL LOUD before binding the endpoint or spawning any task: a mis-tuned redelivery layer must
    // never come up half-configured (R-2b). This is the FIRST statement.
    cfg.reliability
        .validate()
        .map_err(|e| ProdIoError::Tuning(e.to_string()))?;
    // R-6d3a F3 (post-impl review): the durable-outbox non-blocking submit is `Full`-safe ONLY while the peer
    // count fits the outbox writer channel — peak in-flight = one un-durable batch per parked peer-writer ≤
    // peer count. Beyond `OUTBOX_WRITER_CHANNEL_DEPTH` a `try_send` would hit `Full` and PANIC. Fail LOUD at
    // boot (never a silent runtime crash) — but ONLY when an outbox is actually wired (bins pass `None` today).
    if outbox.is_some() && cfg.peers.len() > crate::outbox::OUTBOX_WRITER_CHANNEL_DEPTH {
        return Err(ProdIoError::Tuning(format!(
            "peer count {} exceeds the durable outbox writer channel depth {} — a durable send under this \
             fan-out could overflow the non-blocking submit; raise OUTBOX_WRITER_CHANNEL_DEPTH",
            cfg.peers.len(),
            crate::outbox::OUTBOX_WRITER_CHANNEL_DEPTH,
        )));
    }
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
    let mut server_config = trust.quinn_server_config(cfg.world_generation)?;
    server_config.transport_config(Arc::clone(&transport));
    let mut client_config = trust.quinn_client_config(cfg.world_generation)?;
    client_config.transport_config(Arc::clone(&transport));

    let endpoint = {
        let _guard = handle.enter();
        let mut endpoint = quinn::Endpoint::server(server_config, cfg.bind)?;
        endpoint.set_default_client_config(client_config);
        endpoint
    };

    let inbox: SharedInbox = Arc::new(InboundQueues::new(cfg.inbound_capacity));
    let stats = Arc::new(MeshStats {
        // COMPUTED ONCE, here, from what this node actually offers — so the line a failed dial prints is
        // the tag that was really on the wire and not a second spelling of it.
        offered_tag: String::from_utf8_lossy(&crate::trust::intershard_alpn(cfg.world_generation))
            .into_owned(),
        ..MeshStats::default()
    });
    // The node-wide receiver ledger (SURVIVES connection teardown — the cross-stream cure) and the dialed-
    // connection registry (the drop_connections blip lever), created ONCE and shared by every task.
    let ledger: RecvLedger = Arc::new(RwLock::new(BTreeMap::new()));
    let connections: ConnRegistry = Arc::new(Mutex::new(BTreeMap::new()));
    // CA-1: the LEARNED-peer table (unbooked dial-in return connections) + the BOOKED set (authority split — a
    // booked NodeId is never learned, so no learned lane can shadow a booked one). Created ONCE, shared by the
    // accept path (which learns) + `MeshTransport` (which lazily spawns a learned lane on send).
    let learned: LearnedPeers = Arc::new(Mutex::new(BTreeMap::new()));
    let booked: Arc<BTreeSet<NodeId>> = Arc::new(
        cfg.peers
            .keys()
            .copied()
            .filter(|&p| p != cfg.local)
            .collect(),
    );
    // CA-1 S2 — the shared updatable peer-address topology, seeded from the static book; every Dial-lane re-reads
    // it on each dial, and `MeshControl::update_peer_addr` rcu-refreshes it (the orchestrator/provisioning push).
    let topology: PeerTopology = Arc::new(ArcSwap::from_pointee(cfg.peers.clone()));

    // Accept loop: a Semaphore caps concurrently-served connections so a connection
    // flood cannot task-flood the node (TRANSPORT-4). Each connection serves BOTH
    // reliable uni streams and unreliable datagrams.
    let accept_endpoint = endpoint.clone();
    let accept_inbox = Arc::clone(&inbox);
    let accept_stats = Arc::clone(&stats);
    let accept_ledger = Arc::clone(&ledger);
    let accept_learned = Arc::clone(&learned);
    let accept_booked = Arc::clone(&booked);
    let accept_local = cfg.local;
    let accept_cap = cfg.learned_peers_max;
    let ack_flush = cfg.reliability.ack_idle_flush_interval;
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
            let ledger = Arc::clone(&accept_ledger);
            let learned = Arc::clone(&accept_learned);
            let booked = Arc::clone(&accept_booked);
            tokio::spawn(async move {
                let _permit = permit; // held for the connection's lifetime
                let Ok(connection) = incoming.await else {
                    return; // handshake failed (foreign trust): drop, never serve
                };
                serve_connection(
                    connection,
                    inbox,
                    stats,
                    ledger,
                    ack_flush,
                    learned,
                    booked,
                    accept_local,
                    accept_cap,
                )
                .await;
            });
        }
    });

    // One isolated send lane per peer: bounded queue + dedicated writer task. CA-1 S2: the lane reads the peer's
    // address from the shared `topology` on each dial (not a per-peer captured addr), so a rescheduled peer is
    // re-dialed at its refreshed addr.
    let mut lanes = BTreeMap::new();
    for &peer in cfg.peers.keys() {
        if peer == cfg.local {
            continue; // no self-lane: a node never dials itself
        }
        let (tx, rx) = tokio::sync::mpsc::channel::<OutFrame>(cfg.outbound_capacity);
        handle.spawn(peer_writer(PeerWriter {
            endpoint: endpoint.clone(),
            local: cfg.local,
            dest: peer,
            source: ConnSource::Dial(Arc::clone(&topology)),
            rx,
            inbox: Arc::clone(&inbox),
            stats: Arc::clone(&stats),
            ledger: Arc::clone(&ledger),
            book: Arc::clone(&topology),
            backoff_min: cfg.redial_backoff_min,
            backoff_max: cfg.redial_backoff_max,
            incarnation: cfg.process_incarnation,
            connections: Arc::clone(&connections),
            reliability: cfg.reliability,
            outbox: outbox.clone(),
            ack_rx_override: None, // booked: uses its own dialed-connection ack watch
        }));
        lanes.insert(
            peer,
            PeerLane {
                tx,
                learned_incarnation: None, // booked at boot: this lane dials, it never adopts
            },
        );
    }

    Ok((
        MeshTransport {
            local: cfg.local,
            lanes,
            inbox: Arc::clone(&inbox),
            next_msg_id: 0,
            handle: handle.clone(),
            endpoint: endpoint.clone(),
            learned: Arc::clone(&learned),
            stats: Arc::clone(&stats),
            ledger: Arc::clone(&ledger),
            connections: Arc::clone(&connections),
            outbox: outbox.clone(),
            incarnation: cfg.process_incarnation,
            reliability: cfg.reliability,
            backoff_min: cfg.redial_backoff_min,
            backoff_max: cfg.redial_backoff_max,
            outbound_capacity: cfg.outbound_capacity,
            topology: Arc::clone(&topology),
        },
        MeshControl {
            endpoint,
            stats,
            connections,
            topology,
        },
    ))
}

/// Serve one accepted connection (the DATA-RECEIVER side): reliable uni streams (through the R-3'
/// contiguity verdict) + unreliable datagrams into the shared inbox, AND the reverse cumulative-ACK stream
/// back to the data sender ON THIS SAME connection (the directional-mesh topology — no AckRouter).
#[allow(clippy::too_many_arguments)] // per-connection serve state threaded explicitly
async fn serve_connection(
    connection: quinn::Connection,
    inbox: SharedInbox,
    stats: Arc<MeshStats>,
    ledger: RecvLedger,
    ack_flush: Duration,
    learned: LearnedPeers,
    booked: Arc<BTreeSet<NodeId>>,
    local: NodeId,
    learned_peers_max: usize,
) {
    // Per-connection ack coordination: the (peer,class) keys seen on THIS connection (all share one peer =
    // the dialer) + a wake signal. The data readers populate/notify; the ack-egress task snapshots + writes.
    let acked_keys: Arc<Mutex<BTreeSet<(NodeId, MsgClass)>>> =
        Arc::new(Mutex::new(BTreeSet::new()));
    let ack_due = Arc::new(Notify::new());

    // ACK-EGRESS: acks for this connection's peer ride the reverse direction on THIS SAME connection.
    let ack_task = tokio::spawn(ack_egress(
        connection.clone(),
        Arc::clone(&ledger),
        Arc::clone(&acked_keys),
        Arc::clone(&ack_due),
        ack_flush,
    ));

    // CA-1 — the unified per-connection dispatcher (DATA + ACK) on this ACCEPTED connection, carrying the
    // LearnCtx so its DATA reader learns an unbooked dial-in peer's return connection on the first frame. The
    // per-connection ACK sink `acc_ack_tx` is fed by the dispatcher's ACK branch; a lazily-spawned learned lane
    // (send_durable) subscribes to `acc_ack_rx` (stored in the LearnedConn) to retire its window over THIS
    // connection. In the booked topology no learned lane exists and no ACK stream rides an accepted connection,
    // so `acc_ack_rx` simply has no live consumer (harmless).
    let (acc_ack_tx, acc_ack_rx) = watch::channel::<Option<AckFrame>>(None);
    let learn = LearnCtx {
        learned,
        booked,
        local,
        cap: learned_peers_max,
        conn: connection.clone(),
        ack_rx: acc_ack_rx,
        stats: Arc::clone(&stats),
    };
    let streams = tokio::spawn(dispatch_streams(
        connection.clone(),
        Arc::clone(&inbox),
        Arc::clone(&stats),
        Arc::clone(&ledger),
        Arc::clone(&acked_keys),
        Arc::clone(&ack_due),
        acc_ack_tx,
        Some(learn),
    ));

    // Unreliable datagrams on the same connection. Extracted to `read_datagrams` (D-18 fix) so the DIAL side
    // runs the IDENTICAL loop — a datagram must be readable regardless of who dialed, or a reply-on-connection
    // snapshot to a client (which DIALS the gateway) is silently dropped.
    read_datagrams(&connection, &inbox, &stats).await;
    // The connection closed (peer gone / drop_connections / kill): tear down its reader + ack tasks.
    streams.abort();
    ack_task.abort();
}

/// Read UNRELIABLE datagrams off `conn` into the shared inbox until the connection closes (`read_datagram`
/// errors on close). Runs on BOTH sides of a connection — the ACCEPT side ([`serve_connection`]) AND the DIAL
/// side ([`ensure_connection`]'s Dial arm) — because QUIC datagrams are bidirectional and each endpoint must
/// read the datagrams sent TO IT. Without a reader on the dialed side, a peer that DIALS (a client dialing the
/// gateway) never receives the datagrams the far end sends back over that same connection — the D-18 snapshot
/// loss: reliable frames traverse `dispatch_streams` (both sides) so login works, but snapshots (unreliable
/// datagrams) are dropped on the dialer with `send_datagram` succeeding on the sender (no drop counter moves).
async fn read_datagrams(conn: &quinn::Connection, inbox: &SharedInbox, stats: &Arc<MeshStats>) {
    while let Ok(datagram) = conn.read_datagram().await {
        if let Ok(frame) = postcard::from_bytes::<DatagramFrame>(&datagram) {
            push_unreliable(
                inbox,
                stats,
                Inbound::Wire {
                    from: frame.from,
                    class: frame.class,
                    bytes: vd_sim::io::bytes(frame.bytes),
                },
            );
        }
        // A malformed datagram is silently dropped: unreliable carriers tolerate it.
    }
}

/// CA-1 S1 — the ONE unified per-connection uni-stream dispatcher. Runs on EVERY connection (accepted AND
/// dialed): one `accept_uni` loop consumes the 1-byte `STREAM_KIND` tag and routes by role — DATA to
/// [`serve_data_stream_body`] (the contiguity verdict into the node-wide ledger), ACK to [`drain_ack_stream`]
/// (forwarded to the owning `peer_writer` via `ack_out`). This REPLACES the former split of a DATA-only accept
/// loop in `serve_connection` and an ACK-only `ack_reader_task`: two `accept_uni` loops on one connection would
/// RACE (quinn hands each incoming uni to exactly one waiter), so there must be exactly ONE loop per connection.
#[allow(clippy::too_many_arguments)] // per-connection serve state threaded explicitly
async fn dispatch_streams(
    conn: quinn::Connection,
    inbox: SharedInbox,
    stats: Arc<MeshStats>,
    ledger: RecvLedger,
    acked_keys: Arc<Mutex<BTreeSet<(NodeId, MsgClass)>>>,
    ack_due: Arc<Notify>,
    ack_out: watch::Sender<Option<AckFrame>>,
    learn: Option<LearnCtx>,
) {
    while let Ok(mut recv) = conn.accept_uni().await {
        let mut kind = [0u8; 1];
        if recv.read_exact(&mut kind).await.is_err() {
            continue; // peer opened + finished, or a torn stream: clean silent close, try the next
        }
        match stream_kind(kind[0]) {
            Some(StreamKind::Data) => {
                tokio::spawn(serve_data_stream_body(
                    recv,
                    Arc::clone(&inbox),
                    Arc::clone(&stats),
                    Arc::clone(&ledger),
                    Arc::clone(&acked_keys),
                    Arc::clone(&ack_due),
                    learn.clone(),
                ));
            }
            Some(StreamKind::Ack) => {
                tokio::spawn(drain_ack_stream(recv, ack_out.clone()));
            }
            None => continue, // unknown tag ⇒ drop the stream wholesale, ledger untouched
        }
    }
}

/// Consume the frames of ONE accepted DATA stream (the dispatcher already read + classified the tag). Each
/// frame runs the contiguity verdict into the node-wide ledger and marks its `(peer,class)` ack-relevant —
/// even a Dedup re-acks, so a lost ack is recovered when the sender replays until the cumulative ack advances.
async fn serve_data_stream_body(
    mut recv: quinn::RecvStream,
    inbox: SharedInbox,
    stats: Arc<MeshStats>,
    ledger: RecvLedger,
    acked_keys: Arc<Mutex<BTreeSet<(NodeId, MsgClass)>>>,
    ack_due: Arc<Notify>,
    learn: Option<LearnCtx>,
) {
    let mut recorded = false;
    while let Some(frame) = read_one_reliable_frame(&mut recv).await {
        // ★ WAIT FOR ROOM — THIS IS THE BACK-PRESSURE (2026-08-29). While this task is parked it does
        // not read the stream, so QUIC's own flow control stalls the sender once its stream window
        // fills. The frame waits inside the transport, which is the standard place for it: the
        // application never takes custody of something it cannot hold.
        //
        // Awaited OUTSIDE every lock (the crate lints `await_holding_lock`), and the reservation is
        // per `(peer, class)` stream because one reader task serves one stream — so a slow consumer
        // slows exactly its own lane and no other.
        //
        // `Err` means the receiver is gone, which happens only as the node shuts down: stop reading.
        let Ok(permit) = inbox.reliable_tx.reserve().await else {
            return;
        };
        classify_and_deliver(permit, &inbox, &stats, &ledger, &frame);
        // CA-1 record site (accept path only): learn an UNBOOKED dial-in peer's return connection ONCE, on the
        // first authenticated frame. A SEPARATE `learned.lock()` inside `learn_dial_in_peer` — NEVER nested in
        // the ledger's inner Mutex (classify_and_deliver above obeys outer-read→inner→inbox and has fully
        // returned here), preserving the global lock order.
        if !recorded && let Some(ctx) = &learn {
            recorded = true;
            learn_dial_in_peer(ctx, frame.from, frame.incarnation);
        }
        acked_keys
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .insert((frame.from, frame.class));
        ack_due.notify_one();
    }
}

/// CA-1 record site helper — record an UNBOOKED dial-in peer's return connection in `LearnedPeers`. AUTHORITY
/// SPLIT: a BOOKED NodeId (or `local`) is NEVER learned, so a learned lane can never shadow a booked one nor be
/// spoofed into existence for a booked id. Past the cap: reject + count (loud), never silent.
///
/// REPLACEMENT RULE (two reasons, one site):
/// 1. the held connection is DEAD — the old rule, unchanged;
/// 2. `incarnation` is STRICTLY HIGHER than the held entry's — a NEW PROCESS speaks for the same node id.
///    The old process is gone even though its connection may still look open (a killed client leaves one
///    behind until the idle timeout). The new connection WINS, and the old one is CLOSED at once so its
///    learned writer terminates now instead of writing into a socket nobody reads.
///
/// An EQUAL or LOWER incarnation on a LIVE entry changes nothing: the old process re-dialing after its
/// connection was closed must never take the reply path back from the new process.
fn learn_dial_in_peer(ctx: &LearnCtx, from: NodeId, incarnation: u64) {
    if from == ctx.local || ctx.booked.contains(&from) {
        return; // authority split: never learn a booked/self id
    }
    // SPIFFE cross-check MARKER (M6): the shared cluster cert proves cluster-membership, NOT this NodeId — so the
    // learned NodeId is sender-asserted (the SAME trust the receiver already grants every inbound frame). A
    // per-node SPIFFE SAN check drops in HERE before trusting `from`. Release-present (log), NOT an authority gate.
    if ctx.conn.peer_identity().is_none() {
        tracing::warn!(
            from = from.0,
            "learned dial-in peer presented no peer_identity (SPIFFE cross-check point; sender-asserted NodeId)"
        );
    }
    let mut t = ctx
        .learned
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    // What the table holds for this id right now: is it alive, and which process authored it?
    let held: Option<(bool, u64, quinn::Connection)> = t.get(&from).map(|e| {
        (
            e.conn.close_reason().is_none(),
            e.incarnation,
            e.conn.clone(),
        )
    });
    match held {
        // A live entry from the SAME (or a newer) process: keep it. Nothing to do.
        Some((true, held_inc, _)) if incarnation <= held_inc => {}
        Some((live, _, old_conn)) => {
            t.insert(
                from,
                LearnedConn {
                    conn: ctx.conn.clone(),
                    ack_rx: ctx.ack_rx.clone(),
                    incarnation,
                },
            );
            // A LIVE entry was superseded by a newer process. Close the old connection so its learned
            // writer stops NOW, and count it — a silent swap would hide peer churn.
            if live {
                old_conn.close(
                    quinn::VarInt::from_u32(4),
                    b"superseded by a newer incarnation",
                );
                ctx.stats
                    .learned_peers_superseded
                    .fetch_add(1, Ordering::Relaxed);
            }
        }
        None if t.len() < ctx.cap => {
            t.insert(
                from,
                LearnedConn {
                    conn: ctx.conn.clone(),
                    ack_rx: ctx.ack_rx.clone(),
                    incarnation,
                },
            );
        }
        None => {
            ctx.stats
                .learned_peers_rejected
                .fetch_add(1, Ordering::Relaxed);
            tracing::warn!(
                from = from.0,
                cap = ctx.cap,
                "learned-peer table FULL — rejecting a new dial-in peer (raise learned_peers_max; the \
                 receive ledger is evicted on a vanished peer, this table is not)"
            );
        }
    }
}

/// ★ M6 — THE VANISHED CLIENT'S RECEIVE LEDGER. The inverse of [`learn_dial_in_peer`]: a learned peer that
/// dialed in, was learned, and whose connection then DIED gives its [`RecvLedger`] entry back.
///
/// THE RULE, in one line: a peer this node CANNOT re-dial loses its dedup state; a peer it CAN re-dial keeps it.
///
/// - A CLIENT dials the gateway, plays, and is killed. It has no booked address, so this node can never reach
///   that process again; the next login is a NEW process at a NEW incarnation, which resets the ladder anyway.
///   Holding its `(peer, class)` watermarks buys nothing and the map grows for the life of the process.
/// - A SHARD or the ORCHESTRATOR is booked. Its lane is re-dialed and its flows resume against the SAME
///   watermarks, so dropping them here would re-open the cross-stream hole the node-wide ledger exists to
///   close. It keeps its entry, always.
///
/// LOCK ORDER (unchanged, see [`RecvLedger`]): this takes the OUTER write lock and NOTHING else — no inner
/// peer `Mutex`, no inbox. It runs on the writer task, off the receive hot path.
///
/// SUPERSEDE needs nothing here: when [`learn_dial_in_peer`] replaces an entry for a HIGHER incarnation, the
/// ledger is left alone ON PURPOSE — the new process's first frame carries the higher incarnation, and
/// `classify_and_deliver` resets that peer's `RecvState` through the ladder (and says `PeerReset`
/// `Reincarnated` once). Evicting there would be a second, racing mechanism for one job.
fn evict_recv_ledger(ledger: &RecvLedger, book: &PeerTopology, peer: NodeId, stats: &MeshStats) {
    if book.load().contains_key(&peer) {
        return; // booked: it is re-dialed, and its flows resume with the dedup ladder intact
    }
    let removed = ledger
        .write()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
        .remove(&peer)
        .is_some();
    if removed {
        stats.recv_ledgers_evicted.fetch_add(1, Ordering::Relaxed);
    }
}

/// Consume the frames of ONE accepted ACK stream (the dispatcher already read + classified the tag) and forward
/// every decoded `AckFrame` to the owning `peer_writer` via the latest-wins watch. Returns when the watch
/// receiver is gone (the writer dropped it) — an ACK stream on a connection with no local waiting writer (e.g.
/// an accepted connection in the booked topology) drains harmlessly.
async fn drain_ack_stream(mut recv: quinn::RecvStream, ack_out: watch::Sender<Option<AckFrame>>) {
    while let Some(ack) = read_one_ack_frame(&mut recv).await {
        if ack_out.send(Some(ack)).is_err() {
            return; // the owning peer_writer's watch receiver is gone
        }
    }
}

/// Classify ONE reliable frame under the ledger lock and deliver it (or count the drop). The ledger guard
/// is held across the SYNCHRONOUS `push_inbox` (never across an await): if the RELIABLE frame is dropped by
/// a full inbox (genuine overload), the whole `RecvState` is ROLLED BACK to its pre-classify snapshot —
/// advancing `hw` without delivering would Dedup every redelivery = permanent silent loss. Contiguity (the
/// Gap arm), not the epoch check, is the anti-burying guard, so this rollback cannot re-open the reverted-R-1
/// CRITICAL; a later redelivery re-drives the transition cleanly.
/// ★ THE SLOT IS RESERVED BEFORE THIS IS CALLED (2026-08-29), so delivery here cannot fail and cannot
/// wait. The caller awaits room OUTSIDE every lock; this function stays synchronous and keeps the
/// ledger's lock exactly where it was. `clippy::await_holding_lock` is a crate lint, and honouring it
/// is what forces the reservation to the caller — which is also where it belongs, because that is the
/// only place that can yield the task and let QUIC's flow control slow the sender.
fn classify_and_deliver(
    permit: tokio::sync::mpsc::Permit<'_, Inbound>,
    inbox: &SharedInbox,
    stats: &MeshStats,
    ledger: &RecvLedger,
    frame: &ReliableFrame,
) {
    // R-4e2: fetch this peer's inner lock. Steady state (the peer is already known) = a SHARED READ of the
    // outer map, so a frame from peer P never contends with a frame from peer Q. Only the RARE first frame
    // from a never-seen peer takes the outer WRITE lock to insert the peer's inner map.
    let peer_lock = {
        let outer = ledger
            .read()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        outer.get(&frame.from).cloned()
    };
    let peer_lock = match peer_lock {
        Some(l) => l,
        None => {
            let mut outer = ledger
                .write()
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            Arc::clone(outer.entry(frame.from).or_insert_with(|| {
                Arc::new(Mutex::new(PeerRecv {
                    incarnation: None,
                    classes: BTreeMap::new(),
                }))
            }))
        }
    };
    // The inner per-peer lock: serializes only same-peer frames, and is HELD across classify + push_inbox +
    // the rollback (the atomicity invariant — see the RecvLedger doc). The inner seed is byte-identical to
    // the pre-re-key node-wide entry (only the OUTER insert above is new).
    let mut peer = peer_lock
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    // ★ THE PEER RESTARTED (2026-09-05, the vanished-client wedge). A strictly higher incarnation says a NEW
    // process now speaks for this node id. Tell this node ONCE, at the peer level, BEFORE the first frame of
    // the new process reaches the reliable channel: the notice queue is drained first, so a consumer that
    // binds a session to a peer tears the old session down before it reads the new process's login.
    if peer_restarted(&mut peer.incarnation, frame.incarnation) {
        push_notice(
            inbox,
            stats,
            Inbound::PeerReset {
                node: frame.from,
                cause: PeerResetCause::Reincarnated,
            },
        );
    }
    let st = peer.classes.entry(frame.class).or_insert(RecvState {
        incarnation: frame.incarnation,
        epoch: frame.epoch,
        hw: 0,
        primed: false,
    });
    match classify_reliable(st, frame.incarnation, frame.epoch, frame.seq) {
        Verdict::Accept | Verdict::Reset => {
            // ★ INFALLIBLE. The room was reserved before this function was entered, so there is no drop
            // to surface and no watermark to roll back.
            //
            // The rollback that used to live here — "not delivered, so `hw` must NOT advance" — was
            // CORRECT and is now unreachable. It was the honest half of an unrecoverable situation: the
            // receiver refused to pretend it had a frame it discarded, and then nothing ever re-sent it,
            // because this transport replays a lane only when its STREAM is gone. One drop, and the two
            // nodes talked past a hole for the life of the process.
            permit.send(Inbound::Wire {
                from: frame.from,
                class: frame.class,
                bytes: vd_sim::io::bytes(frame.bytes.clone()),
            });
        }
        Verdict::StaleIncarnation => {
            stats.stale_incarnation_drop.fetch_add(1, Ordering::Relaxed);
        }
        Verdict::StaleEpoch => {
            stats.stale_epoch_drop.fetch_add(1, Ordering::Relaxed);
        }
        Verdict::Dedup => {
            stats.dedup_drop.fetch_add(1, Ordering::Relaxed);
        }
        Verdict::Gap => {
            stats.gap_drop.fetch_add(1, Ordering::Relaxed);
            tracing::warn!(
                from = frame.from.0,
                seq = frame.seq,
                "reliable contiguity GAP (seq > hw+1) — MUST-BE-0 alert"
            );
        }
    }
}

/// The reverse cumulative-ACK writer for one accepted connection: open ONE `STREAM_KIND_ACK` uni stream on
/// THIS connection, then flush the current cumulative acks whenever a data frame arrives (coalesced) or the
/// idle timer fires (so a lone damage/death/despawn is acked without follow-up traffic).
///
/// ⚠️ CANCEL-SAFETY: `write_ack_frame().await` is OUTSIDE the `select!` — the select only DECIDES to flush.
/// quinn `write_all` is not cancel-safe, and a torn `AckFrame` permanently desyncs the sender's ack reader
/// (the reverted-R-1 bug class, reverse lane). On a write error the stream is dropped and cumulative acks
/// self-recover on the next connection.
async fn ack_egress(
    connection: quinn::Connection,
    ledger: RecvLedger,
    acked_keys: Arc<Mutex<BTreeSet<(NodeId, MsgClass)>>>,
    ack_due: Arc<Notify>,
    flush_interval: Duration,
) {
    let mut send = match connection.open_uni().await {
        Ok(s) => s,
        Err(_) => return, // the connection is already gone
    };
    if send.write_all(&[STREAM_KIND_ACK]).await.is_err() {
        return;
    }
    let mut interval = tokio::time::interval(flush_interval);
    let mut last_sent: Vec<AckEntry> = Vec::new();
    loop {
        // DECIDE to flush; NO write future inside the select (cancel-safety).
        tokio::select! {
            () = ack_due.notified() => {}
            _ = interval.tick() => {}
        }
        // Snapshot the cumulative acks. R-4e2: copy the key set (acked_keys grows-only, cumulative acks are
        // monotone+idempotent, so a key added after this copy is picked up next flush — no atomicity needed),
        // DROP acked_keys, THEN read the ledger — so acked_keys is never held across an inner peer lock (the
        // lock-order invariant). Per key: outer-read → that peer's inner Mutex → read (incarnation,epoch,hw).
        // The result is copied out; the write below stays OUTSIDE all locks (cancel-safety).
        let keys: Vec<(NodeId, MsgClass)> = {
            let g = acked_keys
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            g.iter().copied().collect()
        };
        let entries = {
            let outer = ledger
                .read()
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            let mut entries = Vec::new();
            for (peer, class) in keys {
                if let Some(peer_lock) = outer.get(&peer) {
                    let recv = peer_lock
                        .lock()
                        .unwrap_or_else(std::sync::PoisonError::into_inner);
                    if let Some(st) = recv.classes.get(&class) {
                        // Stamp THIS class's incarnation per-entry (R-6c/L4) so the sender's on_ack matches
                        // it against that lane's own incarnation — correct even if the connection ever
                        // carries two classes at different incarnations (no last-class-wins scalar).
                        entries.push(AckEntry {
                            class,
                            incarnation: st.incarnation,
                            epoch: st.epoch,
                            ack_through: st.hw,
                        });
                    }
                }
            }
            entries
        };
        // Nothing to ack yet, or identical to the last flush (dedup the idle busy-write; race-free — any
        // note that raced in is reflected in this very snapshot).
        if entries.is_empty() || entries == last_sent {
            continue;
        }
        let frame = AckFrame {
            entries: entries.clone(),
        };
        if write_ack_frame(&mut send, &frame).await.is_err() {
            return;
        }
        last_sent = entries;
    }
}

struct PeerWriter {
    endpoint: quinn::Endpoint,
    local: NodeId,
    /// The peer this lane targets (the `ConnRegistry` key for a Dial lane; the `LearnedPeers` key for a Learned
    /// lane).
    dest: NodeId,
    /// CA-1: DIAL the static addr (booked) or adopt the held accepted connection (learned reply-on-connection).
    source: ConnSource,
    rx: tokio::sync::mpsc::Receiver<OutFrame>,
    inbox: SharedInbox,
    stats: Arc<MeshStats>,
    /// The node-wide receiver ledger (CA-1 S1): the DIALED connection now also runs the unified dispatcher, so
    /// its inbound DATA classifies into the SAME shared ledger the accept path uses (the StaleEpoch cross-stream
    /// cure requires ONE ledger Arc across both directions). M6: this lane's death also EVICTS `dest`'s ledger
    /// entry when `dest` has no booked address — see [`evict_recv_ledger`].
    ledger: RecvLedger,
    /// M6 — the LIVE peer address book, read at this lane's death to answer ONE question: can this node still
    /// reach `dest`? A booked peer (a shard, the orchestrator) is re-dialed and keeps its receive ledger; an
    /// unbooked dial-in peer (a client) can never be reached again, so its ledger is evicted. The same `Arc`
    /// `Transport::book_peer` and `MeshControl::update_peer_addr` write, so a peer booked AFTER its lane was
    /// learned is read as booked here — the answer is the CURRENT one, never a spawn-time copy.
    book: PeerTopology,
    backoff_min: Duration,
    backoff_max: Duration,
    /// This process's incarnation, stamped on every reliable frame this lane sends (R-2b).
    incarnation: u64,
    /// This node's dialed-connection registry — this writer inserts its connection on dial (for
    /// [`MeshControl::drop_connections`]).
    connections: ConnRegistry,
    /// The at-least-once redelivery tuning (R-4a reads `confirm_unreachable_after_retries`).
    reliability: MeshReliabilityTuning,
    /// R-6d3a: the ONE shared durable outbox sink (cloned `Arc`), or `None` when no outbox is wired (bins pass
    /// `None` until R-6d3b — byte-identical to pre-R-6d3a). The SAME handle reaches both the send path
    /// (`stage_reliable_batch` retain + the batch durable-before-send gate) and the ack path (`on_ack`
    /// release), so a retained
    /// durable row is released on the same store — no leak (LOW-4).
    outbox: Option<SharedOutbox>,
    /// CA-1: a LEARNED lane consumes acks from the accepted connection's per-connection watch (fed by that
    /// connection's dispatcher), because a learned lane never dials and thus has no dispatcher of its own to
    /// forward acks. `None` for a booked lane (it uses its own freshly-created watch). Stage 3 makes this
    /// re-assignable on a learned-connection refresh; Stage 2 captures it once (happy-path v1).
    ack_rx_override: Option<watch::Receiver<Option<AckFrame>>>,
}

/// One peer's writer: drains its lane in FIFO order, dialing on demand, with
/// exponential re-dial backoff so a dead host is NOT hammered at the tick rate
/// (TRANSPORT-3). Reliable frames ride a persistent uni stream and bounce
/// `NodeUnreachable` on hard failure; unreliable frames ride datagrams (latest-wins:
/// loss is correct, no bounce).
/// ★ A CLOSED LEARNED CONNECTION IS A LOSS ONLY IF THE PEER IS GONE (foundation slice 5, 2026-09-06).
/// When a dial-in peer restarts, its NEW connection supersedes the old one, and this mesh CLOSES the
/// old one itself. The writer lane bound to that old connection then sees its ack watch close — the
/// same signal a dead peer gives. It must ask the learned table before it speaks: a LIVE connection
/// held for the peer now is the newer incarnation, so the peer was superseded, never lost. MEASURED in
/// k3d: a re-login from a fresh client pod raced its own `Hello` — the gateway created the new session,
/// then the old lane's `ConnectionLost` ended it, and the client hung with no session and no `Close`.
/// On one host the loss notice always landed first, which is why the login-after-kill gate stayed
/// green. A dialed (booked) lane is never superseded this way: it re-dials, so it answers `false`.
fn peer_superseded(source: &ConnSource, dest: NodeId) -> bool {
    match source {
        ConnSource::Learned(table) => table
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .get(&dest)
            .is_some_and(|held| held.conn.close_reason().is_none()),
        ConnSource::Dial(_) => false,
    }
}

async fn peer_writer(mut w: PeerWriter) {
    // ONE peer-level connection (shared across classes) + ONE FSM lane per reliable class. A reliable
    // OutFrame routes to its class lane; unreliable rides datagrams on the shared connection.
    let mut connection: Option<quinn::Connection> = None;
    let mut lanes: BTreeMap<MsgClass, ReliableLaneSender> = BTreeMap::new();
    let mut backoff = w.backoff_min;
    // The reverse-ACK path (R-3'): a child ack-reader accept_uni's the peer's ACK stream on OUR dialed
    // connection and forwards each decoded AckFrame here via a latest-wins watch (a lost ack is superseded
    // by the next cumulative one — never a stale lower ack winning). Re-created per dial, aborted on drop.
    // CA-1: a LEARNED lane's ack watch is fed SOLELY by the accepted connection's dispatcher — so a closed watch
    // means that connection DIED (distinct from a booked lane, whose own `ack_tx` keeps the watch alive while the
    // writer runs). The Err arm below uses this to terminate a learned lane instead of hot-spinning on it.
    let is_learned = matches!(w.source, ConnSource::Learned(_));
    let (ack_tx, ack_rx_own) = watch::channel::<Option<AckFrame>>(None);
    // CA-1: a LEARNED lane subscribes to the ACCEPTED connection's ack watch (`ack_rx_override`, fed by that
    // connection's own dispatcher); a BOOKED lane uses its own watch (`ack_rx_own`), fed by its DIALED
    // connection's dispatcher. `ack_tx` is handed to the Dial arm's dispatcher only — a learned lane never spawns
    // one, so `ack_tx` is inert for it.
    let mut ack_rx = w.ack_rx_override.take().unwrap_or(ack_rx_own);
    // CA-1 S1: the per-connection serve tasks spawned on each (re)dial — the unified dispatcher AND a symmetric
    // ack_egress (so the DIALED connection also emits acks, not just reads them). Aborted + re-spawned on redial,
    // drained on drop. Was a single `ack_reader` JoinHandle before the symmetric-serve refactor. A learned lane
    // spawns NONE (the accepted connection's `serve_connection` owns its dispatcher + ack_egress).
    let mut serve_tasks: Vec<JoinHandle<()>> = Vec::new();
    // R-4a retransmit timer: reusable, pinned. Disarmed = a far-future deadline the guard never polls; armed
    // by an error path, re-driven when it fires. This is what closes the idle-after-blip gap — a lane that
    // blips then goes quiet is re-driven off THIS clock, not off the next OutFrame.
    let retransmit = tokio::time::sleep(w.backoff_max);
    tokio::pin!(retransmit);
    let mut counting = false;
    loop {
        tokio::select! {
            biased; // 1) acks retire promptly  2) new sends  3) retransmit LAST
            changed = ack_rx.changed() => {
                match changed {
                    Ok(()) => if let Some(ack) = ack_rx.borrow_and_update().clone() {
                        apply_ack(&ack, &mut lanes, w.outbox.as_ref(), &w.stats);
                    }
                    // CA-1 HIGH fix (learned-lane connection-death lifecycle): a closed ack watch on a LEARNED
                    // lane means its accepted connection died (the dispatcher that solely owned our watch sender
                    // was aborted). The lane can neither be acked nor redeliver over the dead conn, and it CANNOT
                    // re-dial (NAT). Bounce any undelivered window LOUDLY (never a silent loss — the producer
                    // re-drives, respawning a fresh learned lane over the peer's re-dialed connection via
                    // send_durable's corpse-as-miss) then TERMINATE. This pre-empts the biased-select arm-1
                    // busy-spin (Err is perpetually ready) AND the dead-lane re-bounce storm; RE-connection is
                    // served by respawn. (Window-preserving in-place re-adopt is a ledgered refinement.) A BOOKED
                    // lane holds its own `ack_tx`, so its watch never closes while the writer runs — no-op here.
                    Err(_) if is_learned => {
                        for (&class, lane) in &lanes {
                            // Bounce on a NON-EMPTY in-flight window (`retry`), NOT `owes_redelivery()`: at
                            // connection-death the lane's `stream` may still be Some (the write succeeded; no
                            // write-error nulled it), and `owes_redelivery()` requires `stream.is_none()` — so it
                            // would UNDER-fire and SILENTLY drop an acknowledged-to-the-producer, unacked-in-
                            // flight frame (milestone-audit HIGH). `!retry.is_empty()` surfaces it loudly for the
                            // producer's re-drive regardless of stream state.
                            if !lane.retry.is_empty()
                                && let Some(msg_id) = lane.last_msg_id
                            {
                                push_notice(
                                    &w.inbox,
                                    &w.stats,
                                    Inbound::NodeUnreachable { to: w.dest, class, undelivered: msg_id },
                                );
                            }
                        }
                        // ★ THE DIAL-IN PEER IS GONE (2026-09-05, the vanished-client wedge). A learned peer
                        // has no booked address, so this node can NEVER reach that process again. Say so
                        // ALWAYS — with an empty window too. A quiet client that vanishes owes nothing, and
                        // the old code therefore said NOTHING, which is exactly how a dead client's session
                        // survived it and wedged the next login. A BOOKED lane never says this: a booked peer
                        // is re-dialed, and its confirmed death is the NodeUnreachable bounce after N replays.
                        // Superseded by the peer's newer incarnation: it already announced itself
                        // (`Reincarnated`, on its first frame), so this lane just retires — a loss
                        // notice here would end the session that new incarnation is opening.
                        if peer_superseded(&w.source, w.dest) {
                            w.stats
                                .learned_lanes_superseded
                                .fetch_add(1, Ordering::Relaxed);
                            break;
                        }
                        push_notice(
                            &w.inbox,
                            &w.stats,
                            Inbound::PeerReset {
                                node: w.dest,
                                cause: PeerResetCause::ConnectionLost,
                            },
                        );
                        // ★ AND THE LEDGER GOES WITH IT (M6). The dedup watermarks this node kept for that
                        // client are now dead weight — nobody will ever send under that node id from that
                        // address again. A booked peer keeps its ledger (see `evict_recv_ledger`).
                        evict_recv_ledger(&w.ledger, &w.book, w.dest, &w.stats);
                        break;
                    }
                    Err(_) => {} // booked: unreachable while the writer holds ack_tx — keep draining sends
                }
            }
            maybe = w.rx.recv() => {
                let Some(first) = maybe else { break };
                // ★ ONE BATCH, ONE DISK SYNC (2026-09-06). Take what the await handed over, then sweep
                // whatever else is ALREADY queued for this peer — no waiting, just what is there. A burst of
                // a hundred band-exit despawns toward one shard then costs ONE disk barrier instead of a
                // hundred, and the snapshots caught up in that burst go out BEFORE the barrier rather than
                // behind it. The drain is bounded by the lane's own channel capacity.
                let mut batch = vec![first];
                while let Ok(more) = w.rx.try_recv() {
                    batch.push(more);
                }
                let outcome = write_batch(
                    &w.endpoint,
                    w.dest,
                    &w.source,
                    &mut connection,
                    &mut lanes,
                    &mut serve_tasks,
                    &ack_tx,
                    &w.connections,
                    w.local,
                    w.incarnation,
                    w.reliability.retry_buffer_max_bytes,
                    &mut w.rx,
                    &mut ack_rx,
                    batch,
                    &w.stats,
                    w.outbox.as_ref(),
                    &w.inbox,
                    &w.ledger,
                    w.reliability.ack_idle_flush_interval,
                )
                .await;
                // A lane's OWN write succeeded ⇒ reset ONLY its failure counter (per-lane, H2).
                for class in &outcome.ok_classes {
                    if let Some(l) = lanes.get_mut(class) {
                        l.on_replay_ok();
                    }
                }
                if outcome.down {
                    // The connection/write died: tear it down + arm the retransmit timer, ONCE for the whole
                    // batch. The bounce is NOT here — it is threshold-gated on the timer replay (a blip that
                    // recovers before `confirm_unreachable_after_retries` ⇒ ZERO bounce). The timer IS the
                    // non-blocking backoff clock.
                    handle_connection_drop(&mut connection, &mut serve_tasks, &mut lanes);
                    retransmit.as_mut().reset(tokio::time::Instant::now() + backoff);
                    counting = true;
                    backoff = (backoff * 2).min(w.backoff_max);
                } else if outcome.wrote {
                    backoff = w.backoff_min;
                }
            }
            () = &mut retransmit, if any_lane_owes(&lanes) => {
                // Idle-after-blip re-drive: dial-if-down, then replay EVERY owing lane's window — no new frame.
                let outcome = replay_lanes(
                    &w.endpoint,
                    w.dest,
                    &w.source,
                    &mut connection,
                    &mut lanes,
                    &mut serve_tasks,
                    &ack_tx,
                    &w.connections,
                    w.reliability,
                    &w.inbox,
                    &w.stats,
                    &w.ledger,
                )
                .await;
                match outcome {
                    ReplayOutcome::AllOk => backoff = w.backoff_min,
                    ReplayOutcome::SomeFailed => {
                        // replay_lanes already bumped/reset per-lane counters + bounced any lane past the
                        // threshold (§1.5). Just grow the backoff + re-arm for the next redial attempt.
                        retransmit.as_mut().reset(tokio::time::Instant::now() + backoff);
                        counting = true;
                        backoff = (backoff * 2).min(w.backoff_max);
                    }
                }
            }
        }
        // ONE edge-triggered re-evaluation per iteration (H1 cure): arm iff a lane is owed and no live
        // deadline is counting; disarm once every lane has drained. An error arm above already set a live
        // deadline (counting=true) ⇒ this leaves it untouched.
        rearm_retransmit(
            retransmit.as_mut(),
            &mut counting,
            &lanes,
            backoff,
            w.backoff_max,
        );
    }
    // Loop exit = the per-peer channel closed = clean shutdown; abort the per-connection serve tasks (dispatcher
    // + ack_egress), then lane streams drop here (quinn implicit-finish on the still-live connection, or no-op on
    // an already-dead one).
    for h in serve_tasks.drain(..) {
        h.abort();
    }
}

/// The result of one retransmit-timer replay pass (R-4a).
enum ReplayOutcome {
    AllOk,
    SomeFailed,
}

/// R-4a: does ANY lane owe a redelivery (closed stream + non-empty window)? The retransmit-timer guard —
/// a PER-LANE property, NOT `connection.is_none()` (the C1 cure: a sibling lane re-opening on a fresh send
/// leaves this true for a still-owing lane, so its redelivery is never disarmed).
fn any_lane_owes(lanes: &BTreeMap<MsgClass, ReliableLaneSender>) -> bool {
    lanes.values().any(ReliableLaneSender::owes_redelivery)
}

/// R-4a: the ONE edge-triggered re-evaluation of the retransmit timer, run at the TAIL of every peer_writer
/// loop iteration (the H1 cure — a single deterministic re-derivation from live lane state, never a
/// self-contradictory per-arm arm/disarm). `counting` guards against pushing back an already-live deadline
/// (which would stall retransmit) and against leaving an owed lane with no armed timer.
fn rearm_retransmit(
    timer: std::pin::Pin<&mut tokio::time::Sleep>,
    counting: &mut bool,
    lanes: &BTreeMap<MsgClass, ReliableLaneSender>,
    backoff: Duration,
    backoff_max: Duration,
) {
    let owed = any_lane_owes(lanes);
    if owed && !*counting {
        timer.reset(tokio::time::Instant::now() + backoff); // just became owed ⇒ arm
        *counting = true;
    } else if !owed && *counting {
        timer.reset(tokio::time::Instant::now() + backoff_max); // drained ⇒ disarm to a harmless far deadline
        *counting = false;
    }
    // owed && counting ⇒ leave the live deadline UNTOUCHED (pushing it back would stall retransmit).
}

/// R-4a: on a `WriteFail::Down`, tear the dead connection down so the next dial re-establishes it. Drops the
/// connection (terminating its streams + the receiver's per-stream readers), aborts the ack-reader (no leak,
/// no stale ack racing the fresh watch), and `on_write_error`s every lane (stream=None + epoch bump — the
/// cross-stream-race cure). Does NOT bounce or touch the failure counters — the bounce is threshold-gated in
/// `confirm_and_maybe_bounce` on the timer replay path (a blip that recovers ⇒ zero bounce, the C2 inversion).
fn handle_connection_drop(
    connection: &mut Option<quinn::Connection>,
    serve_tasks: &mut Vec<JoinHandle<()>>,
    lanes: &mut BTreeMap<MsgClass, ReliableLaneSender>,
) {
    *connection = None;
    for h in serve_tasks.drain(..) {
        h.abort();
    }
    for lane in lanes.values_mut() {
        lane.on_write_error();
    }
}

/// R-4a: alarm the saga/`LivenessTracker` iff THIS lane has failed `confirm_unreachable_after_retries`
/// CONSECUTIVE replay cycles (a transient blip that recovers first ⇒ ZERO bounce). Re-bounces past the
/// threshold to keep the tracker fresh; the re-bounce CADENCE is the peer_writer backoff (each replay cycle
/// is one backoff apart, growing to `backoff_max`), so it can never out-produce the `BoundedInbox` drain and
/// evict a genuine reliable inbound — no explicit coalesce state needed (DRY). A never-sent lane (no
/// `last_msg_id`) never bounces.
fn confirm_and_maybe_bounce(
    lane: &ReliableLaneSender,
    reliability: MeshReliabilityTuning,
    inbox: &SharedInbox,
    stats: &MeshStats,
    to: NodeId,
    class: MsgClass,
) {
    if lane.consecutive_failures >= reliability.confirm_unreachable_after_retries
        && let Some(msg_id) = lane.last_msg_id
    {
        push_notice(
            inbox,
            stats,
            Inbound::NodeUnreachable {
                to,
                class,
                undelivered: msg_id,
            },
        );
    }
}

/// Say why a dial failed — ONCE per distinct reason, per peer.
///
/// Two things go in the line that nothing else can supply. The REASON, because it used to be thrown
/// away. And THE TAG WE OFFERED, because the failures this catches — two nodes counting positions in
/// different units, or serving worlds of different SIZES — are both refused by the transport with
/// `no_application_protocol`, which carries no field, no value and no unit. An operator reading that
/// alone has nothing to compare; reading it beside our tag, they compare it with the other node's boot
/// line or admin view and are done.
///
/// ★ THE TAG HAS TWO SEGMENTS AND THE LINE SAYS SO. `unit-` differing means the two builds count
/// positions in different steps. `world-` differing means they count the same way but disagree about how
/// big the world is. They are different problems with different cures, and an operator should not have
/// to diff two long hexadecimal strings by eye to find out which one they have.
///
/// Once per distinct reason, because this is called on every retry and a line per retry is not an alarm.
fn report_dial_failure(stats: &Arc<MeshStats>, dest: NodeId, addr: SocketAddr, reason: &str) {
    let mut seen = stats
        .dial_failure
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    if seen.get(&dest).is_some_and(|last| last == reason) {
        return; // same peer, same reason, already said
    }
    seen.insert(dest, reason.to_owned());
    drop(seen);
    tracing::error!(
        peer = dest.0,
        %addr,
        reason,
        offered_tag = %stats.offered_tag,
        "DIAL FAILED. If the reason mentions the application protocol, this peer does not serve the same \
         world we do, and the two builds cannot exchange positions at all. Compare the tag above with \
         the other node's boot line or its admin view, and look at WHICH SEGMENT differs: a different \
         `unit-` means the two builds count positions in different steps; a different `world-` means \
         they count the same way but disagree about how big the world is. Retrying; this line repeats \
         only if the reason changes."
    );
}

/// R-4a: dial the peer on demand (the ONE dial home, shared by the batch write path + `replay_lanes` so the
/// ack-reader teardown/respawn + the lane-stream reset can never drift). On a fresh dial: register the
/// connection (drop_connections), abort any prior ack-reader + spawn a new one on THIS connection, reset
/// every lane's stream (the old streams died with the old connection). `Ok` iff the connection is up
/// (already-up = a no-op); `Err` on a dial failure.
#[allow(clippy::too_many_arguments)] // writer state threaded explicitly
async fn ensure_connection(
    endpoint: &quinn::Endpoint,
    dest: NodeId,
    source: &ConnSource,
    connection: &mut Option<quinn::Connection>,
    lanes: &mut BTreeMap<MsgClass, ReliableLaneSender>,
    serve_tasks: &mut Vec<JoinHandle<()>>,
    ack_tx: &watch::Sender<Option<AckFrame>>,
    connections: &ConnRegistry,
    inbox: &SharedInbox,
    stats: &Arc<MeshStats>,
    ledger: &RecvLedger,
    ack_flush: Duration,
) -> Result<(), ()> {
    if connection.is_some() {
        return Ok(());
    }
    let conn = match source {
        // BOOKED: DIAL the static addr on demand. Register the connection (drop_connections lever), then run the
        // unified dispatcher (DATA into the shared ledger; ACK forwarded to THIS writer's own `ack_tx`) AND a
        // symmetric ack_egress on this DIALED connection (CA-1 S1 — so a connection is served identically
        // regardless of who dialed it; without the dialer-side ack_egress a reverse DATA flow could never be
        // acked back). Fresh per-connection acked_keys/ack_due, as the accept path builds. SNI is pinned to the
        // cluster-trust SAN; identity comes from mTLS, not DNS.
        ConnSource::Dial(topology) => {
            // CA-1 S2: re-read the peer's CURRENT address from the shared topology on EVERY dial (never a
            // spawn-time copy) — so a rescheduled peer, once `update_peer_addr` refreshes its entry, is dialed at
            // its NEW addr on the next redial. A peer with no current address (removed / not-yet-provisioned) is
            // an Err (like a dead learned conn) → Down → the retransmit timer re-reads on its next fire.
            let Some(addr) = topology.load().get(&dest).copied() else {
                return Err(());
            };
            // THE REASON IS KEPT, not discarded. Both arms used to be `map_err(|_| ())`, which is how a
            // coordinate-unit mismatch became an endless silent retry.
            let conn = match endpoint.connect(addr, "localhost") {
                Ok(connecting) => match connecting.await {
                    Ok(conn) => {
                        // Connected: forget the last failure so a LATER one alarms again rather than
                        // being suppressed by a reason that is no longer true.
                        stats
                            .dial_failure
                            .lock()
                            .unwrap_or_else(std::sync::PoisonError::into_inner)
                            .remove(&dest);
                        conn
                    }
                    Err(e) => {
                        report_dial_failure(stats, dest, addr, &e.to_string());
                        return Err(());
                    }
                },
                Err(e) => {
                    report_dial_failure(stats, dest, addr, &e.to_string());
                    return Err(());
                }
            };
            connections
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner)
                .insert(dest, conn.clone());
            for h in serve_tasks.drain(..) {
                h.abort();
            }
            let acked_keys: Arc<Mutex<BTreeSet<(NodeId, MsgClass)>>> =
                Arc::new(Mutex::new(BTreeSet::new()));
            let ack_due = Arc::new(Notify::new());
            serve_tasks.push(tokio::spawn(dispatch_streams(
                conn.clone(),
                Arc::clone(inbox),
                Arc::clone(stats),
                Arc::clone(ledger),
                Arc::clone(&acked_keys),
                Arc::clone(&ack_due),
                ack_tx.clone(),
                None, // a dialer never learns — only the acceptor of an unbooked peer does
            )));
            serve_tasks.push(tokio::spawn(ack_egress(
                conn.clone(),
                Arc::clone(ledger),
                acked_keys,
                ack_due,
                ack_flush,
            )));
            // D-18: a DIALED connection must ALSO read datagrams — the far end (a gateway replying to a client
            // that dialed in, reply-on-connection) sends snapshots back over THIS connection, and without a
            // reader they are silently dropped. Symmetric with the accept-side reader in `serve_connection`.
            serve_tasks.push(tokio::spawn({
                let conn = conn.clone();
                let inbox = Arc::clone(inbox);
                let stats = Arc::clone(stats);
                async move { read_datagrams(&conn, &inbox, &stats).await }
            }));
            conn
        }
        // LEARNED (CA-1 reply-on-connection): NEVER dial. Adopt the held accepted connection the receive side
        // recorded — its dispatcher + ack_egress are already owned by that connection's `serve_connection`, and
        // this learned lane's acks arrive via `ack_rx_override`. A missing/dead held connection ⇒ `Err` ⇒
        // WriteFail::Down ⇒ the retransmit timer re-reads the table next fire (picks up a refreshed connection if
        // the peer re-dialed in, or bounces once + terminates — Stage 3). No ConnRegistry insert (that is
        // dial-side, for drop_connections), no serve-task spawn.
        ConnSource::Learned(table) => {
            let held = table
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner)
                .get(&dest)
                .map(|lc| lc.conn.clone());
            match held {
                Some(c) if c.close_reason().is_none() => c,
                _ => return Err(()),
            }
        }
    };
    *connection = Some(conn);
    for lane in lanes.values_mut() {
        lane.stream = None;
    }
    Ok(())
}

/// R-4a: (re)open ONE owing lane's uni stream and write its full `replay_batch()` (re-stamped at the current
/// epoch) — the redelivery of the unacked window, NO new assign. Mirrors `write_staged_batch`'s reopen block minus
/// the assign; the buffer-first + receiver-dedup guarantees make a replay idempotent.
async fn replay_one_lane(
    conn: &quinn::Connection,
    lane: &mut ReliableLaneSender,
) -> Result<(), ()> {
    let mut send = conn.open_uni().await.map_err(|_| ())?;
    send.write_all(&[STREAM_KIND_DATA]).await.map_err(|_| ())?;
    lane.stream = Some(send);
    let batch = lane.replay_batch();
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
}

/// R-4a: the retransmit-timer body — re-drive EVERY owing lane's unacked window WITHOUT waiting for a new
/// `OutFrame` (closing the idle-after-blip gap). Dial-if-down (a dial failure fails every owing lane), then
/// replay each owing lane; a lane's own replay Ok ⇒ `on_replay_ok`, Err ⇒ `on_replay_failed` +
/// `confirm_and_maybe_bounce` (per-lane, the C2/H2 cure). A mid-pass write error means the connection died:
/// stop the pass and tear the connection down so the NEXT timer fire re-dials — the remaining owing lanes
/// stay owed and are re-driven then.
#[allow(clippy::too_many_arguments)] // writer state threaded explicitly
async fn replay_lanes(
    endpoint: &quinn::Endpoint,
    dest: NodeId,
    source: &ConnSource,
    connection: &mut Option<quinn::Connection>,
    lanes: &mut BTreeMap<MsgClass, ReliableLaneSender>,
    serve_tasks: &mut Vec<JoinHandle<()>>,
    ack_tx: &watch::Sender<Option<AckFrame>>,
    connections: &ConnRegistry,
    reliability: MeshReliabilityTuning,
    inbox: &SharedInbox,
    stats: &Arc<MeshStats>,
    ledger: &RecvLedger,
) -> ReplayOutcome {
    if ensure_connection(
        endpoint,
        dest,
        source,
        connection,
        lanes,
        serve_tasks,
        ack_tx,
        connections,
        inbox,
        stats,
        ledger,
        reliability.ack_idle_flush_interval,
    )
    .await
    .is_err()
    {
        // Dial failed: every owing lane failed this cycle.
        for (&class, lane) in lanes.iter_mut() {
            if lane.owes_redelivery() {
                lane.on_replay_failed();
                confirm_and_maybe_bounce(lane, reliability, inbox, stats, dest, class);
            }
        }
        return ReplayOutcome::SomeFailed;
    }
    let mut all_ok = true;
    let mut conn_died = false;
    {
        let Some(conn) = connection.as_ref() else {
            return ReplayOutcome::SomeFailed;
        };
        for (&class, lane) in lanes.iter_mut() {
            if !lane.owes_redelivery() || conn_died {
                continue; // already-open lanes untouched; once the conn dies, leave the rest owed for next fire
            }
            match replay_one_lane(conn, lane).await {
                Ok(()) => lane.on_replay_ok(),
                Err(()) => {
                    lane.on_replay_failed();
                    confirm_and_maybe_bounce(lane, reliability, inbox, stats, dest, class);
                    all_ok = false;
                    conn_died = true; // a write error = a dead connection; the rest fail the same way
                }
            }
        }
    }
    if conn_died {
        // Drop the dead connection (now that the `conn` borrow has ended) so the next fire re-dials.
        handle_connection_drop(connection, serve_tasks, lanes);
    }
    if all_ok {
        ReplayOutcome::AllOk
    } else {
        ReplayOutcome::SomeFailed
    }
}

/// R-6d3c — WHAT ONE DRAINED BATCH STAGED, per class: the seqs this batch assigned, ascending, with the
/// classes in the order they first appeared. A list, not a map, because the write phase walks the classes in
/// batch order exactly once and never looks one up.
type StagedByClass = Vec<(MsgClass, Vec<u64>)>;

/// The staging result for ONE drained batch — what to write, and the ONE durability barrier that covers
/// every retained row the batch staged (the group commit).
struct StagedBatch {
    by_class: StagedByClass,
    /// The ONE `(store batch seq, handle)` barrier for the WHOLE batch. `None` when the batch retained
    /// nothing durable (no sink wired, or every frame Ephemeral) ⇒ there is no wait at all.
    gate: Option<(u64, DurabilityHandle)>,
    /// A durable row was staged into a LIVE sink and yet no barrier came back — the store is mid-Drop.
    /// The batch's reliable frames must NOT reach the wire; they stay retained and the retransmit re-drives.
    gate_missing: bool,
}

/// The outcome of ONE drained batch, read by the writer loop.
struct BatchOutcome {
    /// A write failed on a dead connection ⇒ drop it and arm the retransmit timer, ONCE for the batch (not
    /// once per frame — one connection death deserves one reaction, and one backoff step).
    down: bool,
    /// At least one frame reached the wire, or was dropped as a healthy latest-wins datagram. This is the
    /// same meaning the per-frame `Ok(())` carried before, and it is what resets the redial backoff.
    wrote: bool,
    /// The classes whose reliable write completed ⇒ their own failure counters reset (per-lane, H2).
    ok_classes: Vec<MsgClass>,
}

/// ★ THE GROUP COMMIT, HALF ONE (2026-09-06) — stage EVERY reliable frame of one drained batch under ONE
/// sink lock, then ask for ONE durability barrier for all of them.
///
/// This is what used to be `write_frame`'s block A, run once per frame. A shard that emits a hundred
/// band-exit despawns toward one neighbour paid a hundred disk syncs for them; it now pays one.
///
/// SYNCHRONOUS ON PURPOSE, and called BEFORE the batch's first `.await`: every frame the writer took out of
/// its queue is retained in its lane before anything can yield, so a dropped task can never lose one. That
/// is the buffer-first invariant, widened from one frame to a batch.
///
/// Under the lock there is ONLY a pure-RAM stage (`assign_and_retain`'s `retain`) and a NON-BLOCKING
/// `submit_barrier` — nothing that waits on the disk (MF-1), so a durable send never serializes other peers.
/// A rejected frame (un-framable, or a full retry buffer) is shed LOUD here, exactly as before: counted,
/// warned, and bounced as `SendShed` with its reason. Only reliable frames are seen at all, so the "a shed
/// can only arise on a reliable lane" rule is now structural instead of asserted.
#[allow(clippy::too_many_arguments)] // writer state threaded explicitly
fn stage_reliable_batch(
    lanes: &mut BTreeMap<MsgClass, ReliableLaneSender>,
    local: NodeId,
    dest: NodeId,
    incarnation: u64,
    retry_cap: usize,
    batch: &[OutFrame],
    stats: &MeshStats,
    outbox: Option<&SharedOutbox>,
    inbox: &SharedInbox,
) -> StagedBatch {
    let mut by_class: StagedByClass = Vec::new();
    let mut durable_rows: u64 = 0;
    // Lock the shared sink ONCE, and only when this batch actually carries something durable.
    let wants_sink = batch.iter().any(|f| {
        f.class.reliability() == Reliability::Reliable
            && matches!(f.durability, vd_sim::io::Durability::Retained)
    });
    let mut guard = if wants_sink {
        outbox.map(|o| o.lock().unwrap_or_else(std::sync::PoisonError::into_inner))
    } else {
        None
    };
    for f in batch {
        if f.class.reliability() != Reliability::Reliable {
            continue; // the datagrams are written by their own phase, never staged
        }
        let durable = matches!(f.durability, vd_sim::io::Durability::Retained);
        let lane = lanes
            .entry(f.class)
            .or_insert_with(|| ReliableLaneSender::new(dest, incarnation, retry_cap));
        let sink: Option<&mut dyn OutboxSink> = if durable {
            guard
                .as_deref_mut()
                .map(|b| &mut **b as &mut dyn OutboxSink)
        } else {
            None
        };
        let mirrored = durable && sink.is_some();
        match lane.assign_and_retain(local, f.class, &f.bytes, durable, sink) {
            Ok(seq) => {
                // R-4a: remember the id ONLY of a RETAINED frame, so the threshold-gated confirm bounce
                // (which has no OutFrame) can never carry the id of a rejected/never-retained frame.
                lane.last_msg_id = Some(f.msg_id);
                if mirrored {
                    durable_rows += 1;
                }
                match by_class.iter_mut().find(|(c, _)| *c == f.class) {
                    Some((_, seqs)) => seqs.push(seq),
                    None => by_class.push((f.class, vec![seq])),
                }
            }
            Err(reject) => {
                // R-4b: BOTH rejects are shed-loud (counted + bounced), nothing retained ⇒ neither can
                // poison the lane, neither arms the timer or drops the (healthy) connection.
                stats.reliable_shed.fetch_add(1, Ordering::Relaxed);
                let reason = match reject {
                    AssignReject::Unframable => {
                        tracing::warn!(
                            "reliable frame (payload {}) would exceed MAX_STREAM_FRAME_BYTES framed; \
                             rejected (counted, not retained)",
                            f.bytes.len()
                        );
                        ShedReason::Unframable
                    }
                    AssignReject::BufferFull => {
                        tracing::warn!(
                            "reliable retry buffer FULL ({} bytes) — the ack drain is not keeping up \
                             (check reliable_acked progress; a stuck-at-0 value = a dead ack path); \
                             shedding a new {}-byte send (producer backpressure, NO retained frame dropped)",
                            lane.retry_bytes,
                            f.bytes.len()
                        );
                        ShedReason::RetryBufferFull
                    }
                };
                push_notice(
                    inbox,
                    stats,
                    Inbound::SendShed {
                        to: f.to,
                        class: f.class,
                        undelivered: f.msg_id,
                        reason,
                    },
                );
            }
        }
    }
    // ONE submit for the WHOLE batch — the barrier the write phase awaits exactly once.
    let gate = if durable_rows > 0 {
        guard.as_deref_mut().and_then(|b| b.submit_barrier())
    } else {
        None
    };
    drop(guard); // the sink lock is released BEFORE any durability wait (MF-1)
    if durable_rows > 0 {
        stats
            .outbox_rows_retained
            .fetch_add(durable_rows, Ordering::Relaxed);
    }
    if gate.is_some() {
        stats.outbox_batches.fetch_add(1, Ordering::Relaxed);
    }
    let gate_missing = durable_rows > 0 && gate.is_none();
    StagedBatch {
        by_class,
        gate,
        gate_missing,
    }
}

/// ★ THE GROUP COMMIT, HALF TWO — write the batch's already-staged reliable frames, AFTER the one barrier.
///
/// PER CLASS, never per frame, because the double-write cure is a per-class property: a lane whose stream is
/// closed writes its WHOLE replay batch on re-open (which by construction already contains every frame this
/// batch staged for that class), so writing those frames again individually would send each twice. A lane
/// whose stream is open writes only this batch's new seqs, ascending — the send order, preserved.
///
/// STRUCTURAL if/else, NO fall-through (the CRITICAL double-write cure), unchanged in meaning.
#[allow(clippy::too_many_arguments)] // writer state threaded explicitly
async fn write_staged_batch(
    endpoint: &quinn::Endpoint,
    dest: NodeId,
    source: &ConnSource,
    connection: &mut Option<quinn::Connection>,
    lanes: &mut BTreeMap<MsgClass, ReliableLaneSender>,
    serve_tasks: &mut Vec<JoinHandle<()>>,
    ack_tx: &watch::Sender<Option<AckFrame>>,
    connections: &ConnRegistry,
    inbox: &SharedInbox,
    stats: &Arc<MeshStats>,
    ledger: &RecvLedger,
    ack_flush: Duration,
    staged: &StagedByClass,
    ok_classes: &mut Vec<MsgClass>,
) -> Result<(), ()> {
    // Dial on demand through the ONE dial home, ONCE for the batch. A dial failure ⇒ Down; every frame is
    // already retained above ⇒ the timer re-drives them.
    ensure_connection(
        endpoint,
        dest,
        source,
        connection,
        lanes,
        serve_tasks,
        ack_tx,
        connections,
        inbox,
        stats,
        ledger,
        ack_flush,
    )
    .await?;
    let conn = connection.as_ref().ok_or(())?;
    for (class, seqs) in staged {
        let lane = lanes.get_mut(class).ok_or(())?;
        if lane.stream.is_none() {
            // (re)opened stream: write EXACTLY replay_batch(), which by construction ends with this batch's
            // highest staged seq ⇒ every new frame of this class is written exactly once, inside the batch.
            let mut send = conn.open_uni().await.map_err(|_| ())?;
            // Tag the DATA stream ONCE, before any frame — consumed by the reader's read_exact(1) BEFORE
            // the framing loop.
            send.write_all(&[STREAM_KIND_DATA]).await.map_err(|_| ())?;
            lane.stream = Some(send);
            let replay = lane.replay_batch();
            debug_assert_eq!(
                replay.last().map(|f| f.seq),
                seqs.last().copied(),
                "buffer-first invariant: this batch's last staged frame is the highest replay entry"
            );
            let send = lane.stream.as_mut().ok_or(())?;
            for f in &replay {
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
        } else {
            // steady state: stream open ⇒ write ONLY this batch's new frames (NEVER the backlog). Read them
            // out of `retry` first, copying the fields, so the immutable borrow ends before `stream`'s.
            let mut pending = Vec::with_capacity(seqs.len());
            for &seq in seqs {
                let nf = &lane.retry.get(&seq).ok_or(())?.frame;
                pending.push((
                    nf.from,
                    nf.class,
                    nf.incarnation,
                    nf.epoch,
                    seq,
                    nf.bytes.clone(),
                ));
            }
            let send = lane.stream.as_mut().ok_or(())?;
            for (from, cls, inc, epoch, seq, bytes) in &pending {
                write_reliable_frame(send, *from, *cls, *inc, *epoch, *seq, bytes)
                    .await
                    .map_err(|_| ())?;
            }
        }
        ok_classes.push(*class);
    }
    Ok(())
}

/// Write ONE unreliable frame as a datagram on the shared connection. Unchanged policy: no retain, no
/// bounce, latest-wins loss, every drop counted. `Err(())` means the connection died.
#[allow(clippy::too_many_arguments)] // writer state threaded explicitly
async fn write_unreliable(
    endpoint: &quinn::Endpoint,
    dest: NodeId,
    source: &ConnSource,
    connection: &mut Option<quinn::Connection>,
    lanes: &mut BTreeMap<MsgClass, ReliableLaneSender>,
    serve_tasks: &mut Vec<JoinHandle<()>>,
    ack_tx: &watch::Sender<Option<AckFrame>>,
    connections: &ConnRegistry,
    local: NodeId,
    frame: &OutFrame,
    stats: &Arc<MeshStats>,
    inbox: &SharedInbox,
    ledger: &RecvLedger,
    ack_flush: Duration,
) -> Result<(), ()> {
    ensure_connection(
        endpoint,
        dest,
        source,
        connection,
        lanes,
        serve_tasks,
        ack_tx,
        connections,
        inbox,
        stats,
        ledger,
        ack_flush,
    )
    .await?;
    let conn = connection.as_ref().ok_or(())?;
    // Datagrams are message-bounded (QUIC-delimited, NO stream framing — codec_flags is a stream concept)
    // and carry the BARE DatagramFrame (R1' hot/cold split — no reliability metadata on the 20Hz path).
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
        // COUNTED, never silent (GW-1): snapshots are content-partitioned upstream, so a too-large
        // datagram here is a budget misconfiguration.
        stats
            .datagrams_dropped_too_large
            .fetch_add(1, Ordering::Relaxed);
        tracing::warn!(
            "datagram payload {} exceeds the path MTU budget; dropped (counted)",
            payload.len()
        );
        return Ok(()); // dropped, but the connection is healthy
    }
    // A full send queue drops the datagram (latest-wins); only a dead connection is an Err.
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

/// Retire what an incoming cumulative ack covers, on every class it names. ONE home, called from the
/// writer's own ack arm AND from inside a durability wait — a burst must not stop the window draining just
/// because the disk is busy.
///
/// R-6d3a: the shared durable sink is locked ONCE for the whole fan-out (LOW-4 — the SAME `Arc` the staging
/// retained on releases here). Release-through is STAGED only, no fsync in the ack path (a tombstone rides
/// the next retain's submit; a crash before it is durable harmlessly re-delivers an already-acked frame,
/// which the receiver dedups). The lock spans only the pure-RAM `store.delete` staging — no wait under it.
fn apply_ack(
    ack: &AckFrame,
    lanes: &mut BTreeMap<MsgClass, ReliableLaneSender>,
    outbox: Option<&SharedOutbox>,
    stats: &MeshStats,
) {
    let mut guard = outbox.map(|o| o.lock().unwrap_or_else(std::sync::PoisonError::into_inner));
    let mut sink: Option<&mut dyn OutboxSink> = guard
        .as_deref_mut()
        .map(|b| &mut **b as &mut dyn OutboxSink);
    for e in &ack.entries {
        if let Some(lane) = lanes.get_mut(&e.class) {
            // R-6c/L4: retire against THIS entry's own class incarnation (per-entry), not a single
            // frame-level scalar. Each `on_ack` gets a FRESH short-lived `&mut dyn` reborrow so each lane's
            // retired-durable rows are released (staged).
            let retired = lane.on_ack(
                e.incarnation,
                e.epoch,
                e.ack_through,
                sink.as_mut().map(|s| &mut **s as &mut dyn OutboxSink),
            );
            if retired > 0 {
                stats
                    .reliable_acked
                    .fetch_add(retired as u64, Ordering::Relaxed);
            }
        }
    }
}

/// How many times ONE batch may keep servicing its peer's queue through a durability barrier before it must
/// finish. A round happens ONLY when new reliable frames arrived DURING a disk sync, so a producer that
/// pauses at all never reaches the second one; it is a TERMINATION GUARANTEE against a producer that never
/// pauses, not a tuning knob.
///
/// It is set HIGH on purpose. The last round has to await its barrier plainly — there is no round left to
/// write what it would hold — and that one wait is a stop-the-world for the peer's datagrams. MEASURED: with
/// the cap at 8, that rare plain wait alone put ~4 ms back on the snapshot lane's p99 under an overloaded
/// retained lane; at this cap the same measurement reads ~0. Nothing is starved by the high value, because
/// the wait services the peer's DATAGRAMS and applies its ACKS ([`apply_ack`]) the whole time it is waiting.
const MAX_BATCH_ROUNDS: usize = 512;

/// ★ THE DISK MUST NOT STOP THE SNAPSHOTS (2026-09-06). Await this batch's ONE durability barrier while the
/// writer KEEPS SERVICING its queue.
///
/// An UNRELIABLE frame that arrives during the wait goes STRAIGHT OUT on the wire: a player's view of a hull
/// must never freeze because a despawn is being written to disk. A RELIABLE frame is HELD and returned, so
/// the caller stages it into the NEXT round — it is never sent ahead of its own barrier, which is what keeps
/// crash-table row 4 true.
///
/// Without this the wait was a stop-the-world for the whole peer, and it MEASURED as one: with the retained
/// lane pushed past what the disk can take, the snapshot lane's p99 rose by 8.4 ms — the length of a sync.
///
/// CANCEL-SAFETY: `Receiver::recv` is cancel-safe and the barrier future is pinned and re-polled, so the
/// select holds no state that a cancel could tear. A task dropped here loses only frames already moved into
/// `held` — the same exposure the outer `recv` has always had, and this writer task is never aborted.
#[allow(clippy::too_many_arguments)] // writer state threaded explicitly
async fn await_barrier_serving_datagrams(
    gate: (u64, DurabilityHandle),
    rx: &mut tokio::sync::mpsc::Receiver<OutFrame>,
    ack_rx: &mut watch::Receiver<Option<AckFrame>>,
    outbox: Option<&SharedOutbox>,
    endpoint: &quinn::Endpoint,
    dest: NodeId,
    source: &ConnSource,
    connection: &mut Option<quinn::Connection>,
    lanes: &mut BTreeMap<MsgClass, ReliableLaneSender>,
    serve_tasks: &mut Vec<JoinHandle<()>>,
    ack_tx: &watch::Sender<Option<AckFrame>>,
    connections: &ConnRegistry,
    local: NodeId,
    stats: &Arc<MeshStats>,
    inbox: &SharedInbox,
    ledger: &RecvLedger,
    ack_flush: Duration,
    out: &mut BatchOutcome,
) -> Vec<OutFrame> {
    let (batch_seq, durability) = gate;
    let wait = durability.wait_durable_through_async(batch_seq);
    tokio::pin!(wait);
    let mut held: Vec<OutFrame> = Vec::new();
    // Once the producer (or the ack sender) is gone its branch is ready forever; the guards stop those
    // branches so the barrier is awaited instead of spun on. A dead ack watch on a LEARNED lane is the
    // peer's death, and the OUTER select owns that news — here it is only a branch to stop polling.
    let mut producer_gone = false;
    let mut acks_gone = false;
    loop {
        tokio::select! {
            biased; // the barrier first: the moment the disk is done, the reliable writes go
            () = &mut wait => return held,
            maybe = rx.recv(), if !producer_gone => match maybe {
                None => producer_gone = true,
                Some(f) if f.class.reliability() == Reliability::Reliable => held.push(f),
                Some(f) => {
                    match write_unreliable(
                        endpoint, dest, source, connection, lanes, serve_tasks, ack_tx, connections,
                        local, &f, stats, inbox, ledger, ack_flush,
                    )
                    .await
                    {
                        Ok(()) => out.wrote = true,
                        Err(()) => out.down = true,
                    }
                }
            },
            changed = ack_rx.changed(), if !acks_gone => match changed {
                Ok(()) => {
                    let ack = ack_rx.borrow_and_update().clone();
                    if let Some(ack) = ack {
                        apply_ack(&ack, lanes, outbox, stats);
                    }
                }
                Err(_) => acks_gone = true,
            },
        }
    }
}

/// ★ ONE BATCH, ONE DISK SYNC, AND THE DATAGRAMS GO FIRST (2026-09-06).
///
/// The four phases of ONE ROUND, in this order and for these reasons:
///
/// 1. STAGE every reliable frame, synchronously, before anything can yield. Retention comes first so a
///    dropped task loses nothing, and the ONE submit starts the disk working immediately.
/// 2. WRITE THE DATAGRAMS, before the durability wait. A player's snapshot must never wait for a hull's
///    despawn to reach the disk; measurement said it did, by ~5 ms per retained frame ahead of it, because
///    everything toward one peer rides one writer task.
/// 3. AWAIT THE ONE BARRIER for the whole batch — the group commit the design intended — and keep sending
///    this peer's datagrams while the disk works ([`await_barrier_serving_datagrams`]). Reliable frames that
///    arrive during the wait are HELD; they open the next round.
/// 4. WRITE THE RELIABLE FRAMES, in batch order, per class.
///
/// A ROUND ONLY REPEATS while reliable frames kept arriving through a sync, and at most
/// [`MAX_BATCH_ROUNDS`] times, so a producer that never pauses cannot hold the writer here forever.
///
/// CRASH TABLE ROW 4 IS UNCHANGED: a retained frame is durable-or-becoming-durable BEFORE its wire send,
/// never sent before durable. Phase 3 stands between the staging and every reliable write, exactly where the
/// per-frame gate used to stand.
///
/// ⚠️ CANCEL-SAFETY: this is `.await`ed as the plain body of the `w.rx.recv()` select arm, never wrapped in
/// a `timeout`/`select!`, so the sole cancel point in the writer is still that `recv`. A task dropped inside
/// this function drops it AFTER phase 1, so every frame of the batch is retained and re-driven — the frames
/// are durable-or-becoming-durable but NOT sent, which is crash-table row 4.
#[allow(clippy::too_many_arguments)] // writer state threaded explicitly
async fn write_batch(
    endpoint: &quinn::Endpoint,
    dest: NodeId,
    source: &ConnSource,
    connection: &mut Option<quinn::Connection>,
    lanes: &mut BTreeMap<MsgClass, ReliableLaneSender>,
    serve_tasks: &mut Vec<JoinHandle<()>>,
    ack_tx: &watch::Sender<Option<AckFrame>>,
    connections: &ConnRegistry,
    local: NodeId,
    incarnation: u64,
    retry_cap: usize,
    rx: &mut tokio::sync::mpsc::Receiver<OutFrame>,
    ack_rx: &mut watch::Receiver<Option<AckFrame>>,
    batch: Vec<OutFrame>,
    stats: &Arc<MeshStats>,
    outbox: Option<&SharedOutbox>,
    inbox: &SharedInbox,
    ledger: &RecvLedger,
    ack_flush: Duration,
) -> BatchOutcome {
    let mut out = BatchOutcome {
        down: false,
        wrote: false,
        ok_classes: Vec::new(),
    };
    let mut batch = batch;
    for round in 0..MAX_BATCH_ROUNDS {
        // (0) SWEEP whatever queued while the previous round was writing. Without this a snapshot that
        // arrived during the reliable writes would sit in the queue until the NEXT barrier's servicing
        // picked it up — one whole disk sync late, which is exactly what the tail measured.
        while let Ok(more) = rx.try_recv() {
            batch.push(more);
        }
        // (1) STAGE — synchronous, before the first await of this round.
        let staged = stage_reliable_batch(
            lanes,
            local,
            dest,
            incarnation,
            retry_cap,
            &batch,
            stats,
            outbox,
            inbox,
        );

        // (2) THE DATAGRAMS, BEFORE THE DISK.
        for f in &batch {
            if f.class.reliability() == Reliability::Reliable {
                continue;
            }
            match write_unreliable(
                endpoint,
                dest,
                source,
                connection,
                lanes,
                serve_tasks,
                ack_tx,
                connections,
                local,
                f,
                stats,
                inbox,
                ledger,
                ack_flush,
            )
            .await
            {
                Ok(()) => out.wrote = true,
                Err(()) => {
                    out.down = true;
                    break; // the connection is gone; the rest are latest-wins loss
                }
            }
        }

        // (3) THE ONE BARRIER for every retained row this round staged. The LAST allowed round awaits it
        // plainly: it may hold nothing over, because there is no round left to write what it would hold.
        let mut held: Vec<OutFrame> = Vec::new();
        if let Some(gate) = staged.gate {
            if round + 1 < MAX_BATCH_ROUNDS {
                held = await_barrier_serving_datagrams(
                    gate,
                    rx,
                    ack_rx,
                    outbox,
                    endpoint,
                    dest,
                    source,
                    connection,
                    lanes,
                    serve_tasks,
                    ack_tx,
                    connections,
                    local,
                    stats,
                    inbox,
                    ledger,
                    ack_flush,
                    &mut out,
                )
                .await;
            } else {
                let (batch_seq, durability) = gate;
                durability.wait_durable_through_async(batch_seq).await;
            }
        }

        // (4) THE RELIABLE WRITES. A durable stage with a live sink that produced NO barrier means the store
        // was mid-Drop: refuse to write, LOUD — sending here would put a frame on the wire before its row is
        // durable (a silent durable-before-send violation in a release build). The frames stay retained; the
        // retransmit re-drives them.
        if staged.gate_missing {
            tracing::error!(
                "durable reliable send produced no outbox barrier (store mid-Drop?) — refusing to send the \
                 batch before its rows are durable (they stay retained; the retransmit re-drives them)"
            );
            out.down = true;
        } else if !staged.by_class.is_empty() {
            match write_staged_batch(
                endpoint,
                dest,
                source,
                connection,
                lanes,
                serve_tasks,
                ack_tx,
                connections,
                inbox,
                stats,
                ledger,
                ack_flush,
                &staged.by_class,
                &mut out.ok_classes,
            )
            .await
            {
                Ok(()) => out.wrote = true,
                Err(()) => out.down = true,
            }
        }

        if held.is_empty() {
            break; // nothing arrived through the sync ⇒ this batch is finished
        }
        batch = held;
    }
    out
}

impl MeshTransport {
    /// Is the lane this node already holds toward `to` still the RIGHT lane to write into?
    ///
    /// Three ways it is not:
    /// - there is no lane at all;
    /// - the lane's writer task exited, which closes the queue (a corpse);
    /// - the lane is LEARNED and the learned table now holds a STRICTLY HIGHER incarnation — a new process
    ///   dialed in for the same node id, and this lane still points at the old process's connection.
    ///
    /// The third case is the vanished-client cure on the send side: a killed client re-logs in as a new
    /// process, and the reply must ride the NEW connection. The caller drops a stale lane and respawns one
    /// over the current learned connection.
    fn lane_is_current(&self, to: NodeId) -> bool {
        let Some(lane) = self.lanes.get(&to) else {
            return false;
        };
        if lane.tx.is_closed() {
            return false;
        }
        let Some(spawned_for) = lane.learned_incarnation else {
            return true; // a booked lane dials; no learned entry governs it
        };
        let current = self
            .learned
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .get(&to)
            .map(|lc| lc.incarnation);
        match current {
            Some(now) => now <= spawned_for,
            // The entry is gone (evicted): nothing says this lane is stale, and its own death arm will
            // terminate it if its connection is dead.
            None => true,
        }
    }
}

impl Transport for MeshTransport {
    fn send_durable(
        &mut self,
        to: NodeId,
        class: MsgClass,
        bytes: Bytes,
        durability: vd_sim::io::Durability,
    ) -> Result<MsgId, SendError> {
        // CA-1: prefer a BOOKED (or already-spawned learned) lane that is LIVE. A present-but-CLOSED lane (a
        // terminated learned lane, Stage 3) is a MISS — drop it + re-consult LearnedPeers so a re-learned peer
        // re-spawns instead of hitting the corpse. A booked lane never closes (its writer runs while this
        // MeshTransport holds the sender).
        // A node never dials itself: its own id sits in the topology like any peer's, and the lazy
        // dial below would otherwise open a lane to the local endpoint. Refused as a full queue —
        // there is no lane to fill and no peer to ask about (the old answer, kept).
        if to == self.local {
            return Err(SendError::QueueFull(bytes));
        }
        let live = self.lane_is_current(to);
        if !live {
            self.lanes.remove(&to);
            // Reply-on-connection: if `to` is an UNBOOKED peer that dialed IN (learned its return connection),
            // lazily spawn a learned reply lane over that held accepted connection — reusing the FULL
            // ReliableLaneSender machinery (seq/epoch/replay/ack) so the receiver dedup ladder is satisfied. The
            // learned lane subscribes to the accepted connection's ack watch (`ack_rx_override`) to retire its
            // window. Else CA-2 loud back-pressure (a genuinely unknown destination — never a silent drop).
            let learned_conn = self
                .learned
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner)
                .get(&to)
                .cloned();
            // Three ways to a lane, in order: a learned return connection (reply-on-connection), a
            // BOOKED address with no lane yet (the peer book — dial it now), or neither (UnknownPeer:
            // the node runtime asks the orchestrator and re-sends once the answer is booked).
            let (source, ack_rx_override, learned_incarnation) = match learned_conn {
                Some(lc) => {
                    // The learned connection may have died since it was recorded (its entry is evicted
                    // only on the peer's re-dial). Do NOT spawn a lane over a dead connection — it would
                    // immediately terminate and drop this frame. Loud back-pressure instead; a re-dial
                    // refreshes the entry + a re-send respawns.
                    if lc.conn.close_reason().is_some() {
                        return Err(SendError::QueueFull(bytes));
                    }
                    (
                        ConnSource::Learned(Arc::clone(&self.learned)),
                        Some(lc.ack_rx),
                        Some(lc.incarnation),
                    )
                }
                None if self.topology.load().contains_key(&to) => {
                    self.stats.lazy_dials.fetch_add(1, Ordering::Relaxed);
                    (ConnSource::Dial(Arc::clone(&self.topology)), None, None)
                }
                None => return Err(SendError::UnknownPeer(bytes)),
            };
            let (tx, rx) = tokio::sync::mpsc::channel::<OutFrame>(self.outbound_capacity);
            self.handle.spawn(peer_writer(PeerWriter {
                endpoint: self.endpoint.clone(),
                local: self.local,
                dest: to,
                source,
                rx,
                inbox: Arc::clone(&self.inbox),
                stats: Arc::clone(&self.stats),
                ledger: Arc::clone(&self.ledger),
                book: Arc::clone(&self.topology),
                backoff_min: self.backoff_min,
                backoff_max: self.backoff_max,
                incarnation: self.incarnation,
                connections: Arc::clone(&self.connections),
                reliability: self.reliability,
                outbox: self.outbox.clone(),
                ack_rx_override,
            }));
            self.lanes.insert(
                to,
                PeerLane {
                    tx,
                    learned_incarnation,
                },
            );
        }
        let lane = self.lanes.get(&to).expect(
            "a live booked or freshly-spawned learned lane is present after the CA-1 ensure",
        );
        let msg_id = MsgId(self.next_msg_id);
        match lane.tx.try_send(OutFrame {
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
            Err(
                tokio::sync::mpsc::error::TrySendError::Full(frame)
                | tokio::sync::mpsc::error::TrySendError::Closed(frame),
            ) => Err(SendError::QueueFull(frame.bytes)),
        }
    }

    fn book_peer(&mut self, node: NodeId, ip: [u8; 16], port: u16) {
        // The same write `MeshControl::update_peer_addr` does, from the node runtime's side of the seam:
        // the book takes the address, a stale dialed connection is closed so the next dial goes to the
        // new one, and the lane itself is created lazily by the next send (see `send_durable`).
        let v6 = std::net::Ipv6Addr::from(ip);
        let addr = match v6.to_ipv4_mapped() {
            Some(v4) => SocketAddr::new(std::net::IpAddr::V4(v4), port),
            None => SocketAddr::new(std::net::IpAddr::V6(v6), port),
        };
        self.topology.rcu(|cur| {
            let mut next = (**cur).clone();
            next.insert(node, addr);
            next
        });
        let mut reg = self
            .connections
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        if let Some(conn) = reg.remove(&node) {
            conn.close(quinn::VarInt::from_u32(3), b"peer addr booked");
        }
        self.stats.peers_booked.fetch_add(1, Ordering::Relaxed);
    }

    fn drain_inbound(&mut self) -> Vec<Inbound> {
        // ORDER: this node's own notices, then reliable wire, then unreliable. Notices first because a
        // "peer unreachable" changes how the tick reads everything after it; unreliable last because it
        // is the only lane whose contents may be superseded by the next tick anyway.
        let mut out = Vec::new();
        let mut notices = self
            .inbox
            .notices_rx
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        while let Ok(ev) = notices.try_recv() {
            out.push(ev);
        }
        drop(notices);
        let mut reliable = self
            .inbox
            .reliable_rx
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        while let Ok(ev) = reliable.try_recv() {
            out.push(ev);
        }
        drop(reliable);
        out.extend(
            self.inbox
                .unreliable
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner)
                .drain(),
        );
        out
    }

    fn local_id(&self) -> NodeId {
        self.local
    }
}

/// R-6d3b-2b (RC-2b): a boot-replay liveness probe layered on top of the FROZEN [`Transport`] seam (a NEW
/// io-prod PUBLIC supertrait — the `sim::io::Transport` seam itself is UNTOUCHED). `replay_outbox` uses it to
/// distinguish a transiently-full lane (retry) from a permanently DEAD one (its `peer_writer` task exited ⇒
/// the mpsc closed), so a dead lane fast-fails ([`ReplayError::LaneDead`]) instead of a ~10s corpse-spin.
///
/// [`ReplayError::LaneDead`]: crate::outbox::ReplayError::LaneDead
pub trait ReplayTransport: Transport {
    /// True iff a live send lane exists for `peer` (its `peer_writer` task is alive). A dropped mpsc
    /// receiver (the writer exited) closes the sender ⇒ `is_closed()`. O(1). A peer with NO lane (a roster
    /// miss) also reports `false` — but `replay_outbox`'s pre-send membership check catches that first.
    fn lane_alive(&self, peer: NodeId) -> bool;

    /// ★ HOW MANY RETAINED ROWS THIS TRANSPORT HAS MIRRORED, ever — the boot replay's fence anchor.
    ///
    /// The fence used to count SUBMITS, on the rule "one send, one submit". The group commit (2026-09-06)
    /// ended that rule: a peer writer drains its whole queue into ONE batch and asks the disk ONCE, so two
    /// replayed rows toward one shard now produce ONE submit and a submit-counting fence waits forever.
    /// ROWS are exact however they were grouped. Counted AFTER the submit that carries them, so a reading of
    /// N means those N rows have been handed to the disk.
    fn durable_rows_retained(&self) -> u64;
}

impl ReplayTransport for MeshTransport {
    fn lane_alive(&self, peer: NodeId) -> bool {
        self.lanes.get(&peer).is_some_and(|l| !l.tx.is_closed())
    }

    fn durable_rows_retained(&self) -> u64 {
        self.stats.outbox_rows_retained.load(Ordering::Relaxed)
    }
}

// ---- Peer-address AUTO-RESOLVER (CA-1: the production caller of update_peer_addr) -------------------------

/// The DNS-resolution seam for the peer auto-resolver — INJECTED so production uses the system resolver
/// (getaddrinfo, which picks up k8s pod DNS) while a test injects a deterministic in-memory resolver (no
/// getaddrinfo → no flake). `resolve` returns `None` on a failed/NXDOMAIN/empty lookup — a strict NO-OP for the
/// caller (NEVER a topology eviction: a peer mid-reschedule must not be stranded).
pub trait AddrResolver: Send + Sync {
    /// Resolve `host:port` to ONE socket address, or `None` on any failure.
    fn resolve(&self, hostport: &str) -> Option<SocketAddr>;
}

/// The production [`AddrResolver`]: the BLOCKING system resolver (getaddrinfo via nsswitch/resolv.conf). The
/// auto-resolver calls it OFF the runtime worker (`spawn_blocking`).
pub struct SystemDnsResolver;

impl AddrResolver for SystemDnsResolver {
    fn resolve(&self, hostport: &str) -> Option<SocketAddr> {
        resolve_first(hostport)
    }
}

/// The first resolved address for `host:port`, or `None`. The MONOMORPHIC helper the trait impl shares — the
/// `?` lives here, not in a generic body (HR5(a) branchless-shim).
fn resolve_first(hostport: &str) -> Option<SocketAddr> {
    use std::net::ToSocketAddrs;
    hostport.to_socket_addrs().ok()?.next()
}

/// The change-gate for ONE resolved peer address (the monomorphic core — every arm exercised deterministically
/// by the tests): push + record ONLY on a successful-AND-CHANGED address; a failed (`None`) or unchanged resolve
/// is a strict NO-OP (never an eviction, never a re-dial storm — `update_peer_addr` unconditionally closes the
/// dialed conn, so pushing ONLY on a real change is load-bearing).
fn apply_resolved(
    control: &MeshControl,
    last: &mut BTreeMap<NodeId, SocketAddr>,
    id: NodeId,
    resolved: Option<SocketAddr>,
) {
    let Some(addr) = resolved else {
        return; // failed / NXDOMAIN: keep the last-known addr, retry next interval
    };
    if last.get(&id) == Some(&addr) {
        return; // unchanged: no re-dial storm
    }
    control.update_peer_addr(id, addr);
    // A re-plumb is a notable cloud event (a peer rescheduled to a new address) — log it LOUD so a live
    // reschedule proof (S6) can confirm the auto-resolver actually pushed, not just that recovery happened.
    tracing::info!(peer = id.0, %addr, "peer-resolver: re-plumbed a peer to its new address");
    last.insert(id, addr);
}

/// Spawn the peer-address AUTO-RESOLVER — the missing production caller of [`MeshControl::update_peer_addr`]. A
/// background task (OFF the tick path, on the node's existing runtime) that every `tuning.interval` re-resolves
/// each `(NodeId, "host:port")` via the injected [`AddrResolver`] (in `spawn_blocking` — getaddrinfo blocks) and
/// pushes a CHANGED address, so a rescheduled peer (new pod IP, same DNS name) is re-plumbed for INITIATED
/// reliable traffic. Change-detection is seeded from the CURRENT topology so the first tick is a no-op on an
/// unchanged cluster. Exits within one interval of `shutdown` being set (drains with the tick loop — it holds an
/// `Arc<MeshControl>`, so it MUST exit for the endpoint to close). Returns the task handle.
pub fn spawn_peer_resolver(
    handle: &tokio::runtime::Handle,
    control: Arc<MeshControl>,
    hosts: BTreeMap<NodeId, String>,
    resolver: Arc<dyn AddrResolver>,
    tuning: PeerResolveTuning,
    shutdown: Arc<std::sync::atomic::AtomicBool>,
) -> JoinHandle<()> {
    handle.spawn(peer_resolver_loop(
        control, hosts, resolver, tuning, shutdown,
    ))
}

async fn peer_resolver_loop(
    control: Arc<MeshControl>,
    hosts: BTreeMap<NodeId, String>,
    resolver: Arc<dyn AddrResolver>,
    tuning: PeerResolveTuning,
    shutdown: Arc<std::sync::atomic::AtomicBool>,
) {
    // Seed change-detection from the CURRENT topology (the entrypoint's boot resolution) so the first tick is a
    // no-op on an unchanged cluster — an unconditional first push would tear down every live connection.
    let mut last: BTreeMap<NodeId, SocketAddr> = hosts
        .keys()
        .filter_map(|id| control.current_peer_addr(*id).map(|addr| (*id, addr)))
        .collect();
    tokio::time::sleep(tuning.initial_delay).await;
    let mut ticker = tokio::time::interval(tuning.interval);
    ticker.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);
    while !shutdown.load(Ordering::Relaxed) {
        ticker.tick().await;
        if shutdown.load(Ordering::Relaxed) {
            return;
        }
        for (id, host) in &hosts {
            // getaddrinfo BLOCKS — off the runtime worker so a slow/hung resolver never stalls it.
            let host = host.clone();
            let resolver = Arc::clone(&resolver);
            let resolved = tokio::task::spawn_blocking(move || resolver.resolve(&host))
                .await
                .ok()
                .flatten();
            apply_resolved(&control, &mut last, *id, resolved);
        }
    }
}

/// The label a test opens a store under. One place, so a test never states a generation by hand — the
/// whole point of the mechanism is that those two numbers are derived and un-stateable.
#[cfg(test)]
fn test_stamp() -> vd_core::store_stamp::StoreStamp {
    vd_core::store_stamp::StoreStamp::new(
        vd_core::store_stamp::StoreRole::Directory,
        0,
        vd_core::EpochId(0),
        &[],
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::{Duration, Instant};

    const DEADLINE: Duration = Duration::from_secs(20);

    // ---- Pure ReliableLaneSender FSM tests (no tokio/quinn — the at-least-once correctness core) ----

    const FROM: NodeId = NodeId(1);
    /// R-6d2c: the lane's DEST peer (the first `OutboxKey` field). Distinct from `FROM` (the local sender)
    /// so a test that DOES wire a sink can tell a mis-keyed row from a correct one. For every sink-less FSM
    /// test its value is behaviourally inert (the outbox is only touched when a `Some(sink)` is passed).
    const PEER: NodeId = NodeId(2);
    const CLASS: MsgClass = MsgClass::Saga;
    /// A retry-buffer cap so large no FSM test ever trips `AssignReject::BufferFull` (R-4b is exercised by a
    /// dedicated small-cap test).
    const BIG_CAP: usize = usize::MAX;

    #[test]
    fn sender_assigns_monotone_seq_and_retains() {
        let mut lane = ReliableLaneSender::new(PEER, 7, BIG_CAP);
        assert_eq!(
            lane.assign_and_retain(FROM, CLASS, b"a", false, None),
            Ok(0)
        );
        assert_eq!(
            lane.assign_and_retain(FROM, CLASS, b"bb", false, None),
            Ok(1)
        );
        assert_eq!(
            lane.assign_and_retain(FROM, CLASS, b"ccc", false, None),
            Ok(2)
        );
        assert_eq!(lane.next_seq, 3);
        assert_eq!(
            lane.retry.keys().copied().collect::<Vec<_>>(),
            vec![0, 1, 2]
        );
        assert_eq!(
            lane.retry[&1].frame.bytes, b"bb",
            "the retained frame holds its payload"
        );
        // Frames are stamped with the lane's incarnation + current epoch (0).
        let f0 = &lane.retry[&0].frame;
        assert_eq!((f0.incarnation, f0.epoch, f0.seq), (7, 0, 0));
    }

    #[test]
    fn r6c_a_per_class_ack_incarnation_retires_each_lane_against_its_own() {
        // L4: acks carry a PER-CLASS incarnation, so two lanes at DIFFERENT incarnations (the future
        // connection-reuse-across-a-restart case) each retire against ITS OWN. The single frame-level scalar
        // that preceded R-6c was last-class-wins (ack_egress overwrote it each loop iteration) and would
        // stall whichever lane did not match the emitted scalar.
        let mut saga = ReliableLaneSender::new(PEER, 5, BIG_CAP);
        let mut ghost = ReliableLaneSender::new(PEER, 9, BIG_CAP);
        assert_eq!(
            saga.assign_and_retain(FROM, MsgClass::Saga, b"a", false, None),
            Ok(0)
        );
        assert_eq!(
            ghost.assign_and_retain(FROM, MsgClass::GhostReliable, b"b", false, None),
            Ok(0)
        );

        // The ack frame carries each class's own incarnation; the peer_writer fan-out retires each lane
        // against ITS entry's incarnation.
        let entries = [
            AckEntry {
                class: MsgClass::Saga,
                incarnation: 5,
                epoch: 0,
                ack_through: 0,
            },
            AckEntry {
                class: MsgClass::GhostReliable,
                incarnation: 9,
                epoch: 0,
                ack_through: 0,
            },
        ];
        assert_eq!(
            saga.on_ack(
                entries[0].incarnation,
                entries[0].epoch,
                entries[0].ack_through,
                None
            ),
            1
        );
        assert_eq!(
            ghost.on_ack(
                entries[1].incarnation,
                entries[1].epoch,
                entries[1].ack_through,
                None
            ),
            1
        );
        assert!(
            saga.retry.is_empty() && ghost.retry.is_empty(),
            "both classes' windows retired against their own incarnation"
        );

        // The MISFIRE the per-class design removes: a WRONG-incarnation ack (what a shared last-class-wins
        // scalar would apply to the non-matching lane) retires NOTHING — a silent send stall.
        let mut stalled = ReliableLaneSender::new(PEER, 9, BIG_CAP);
        assert_eq!(
            stalled.assign_and_retain(FROM, MsgClass::GhostReliable, b"c", false, None),
            Ok(0)
        );
        assert_eq!(
            stalled.on_ack(5, 0, 0, None),
            0,
            "an ack minted at a wrong incarnation retires nothing"
        );
        assert_eq!(
            stalled.retry.len(),
            1,
            "the lane's window is stranded — the latent stall R-6c removes"
        );
    }

    #[test]
    fn write_error_does_not_roll_back_seq() {
        // BUFFER-FIRST: a failed write (modelled by on_write_error after an assign) never burns or
        // re-uses a seq — the next assign is the next monotone value, at the bumped epoch.
        let mut lane = ReliableLaneSender::new(PEER, 0, BIG_CAP);
        assert_eq!(
            lane.assign_and_retain(FROM, CLASS, b"x", false, None),
            Ok(0)
        );
        lane.on_write_error();
        assert_eq!(
            lane.assign_and_retain(FROM, CLASS, b"y", false, None),
            Ok(1)
        ); // not reused 0, not skipped 2
        assert_eq!(
            lane.retry[&1].frame.epoch, 1,
            "the post-error assign carries the bumped epoch"
        );
        assert_eq!(lane.retry.keys().copied().collect::<Vec<_>>(), vec![0, 1]);
    }

    #[test]
    fn replay_batch_is_ascending_seq_with_current_epoch() {
        let mut lane = ReliableLaneSender::new(PEER, 0, BIG_CAP);
        for b in [b"a".as_slice(), b"b", b"c"] {
            lane.assign_and_retain(FROM, CLASS, b, false, None)
                .expect("fits");
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
        let mut lane = ReliableLaneSender::new(PEER, 0, BIG_CAP);
        lane.assign_and_retain(FROM, CLASS, b"0", false, None)
            .expect("fits"); // seq0 @ e0
        lane.assign_and_retain(FROM, CLASS, b"1", false, None)
            .expect("fits"); // seq1 @ e0
        lane.on_write_error(); // e1
        lane.assign_and_retain(FROM, CLASS, b"2", false, None)
            .expect("fits"); // seq2 @ e1
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
        let mut lane = ReliableLaneSender::new(PEER, 0, BIG_CAP);
        lane.assign_and_retain(FROM, CLASS, b"0", false, None)
            .expect("fits");
        lane.on_write_error();
        let new_seq = lane
            .assign_and_retain(FROM, CLASS, b"1", false, None)
            .expect("fits");
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
        let mut lane = ReliableLaneSender::new(PEER, 0, BIG_CAP);
        for b in [b"a".as_slice(), b"b", b"c"] {
            lane.assign_and_retain(FROM, CLASS, b, false, None)
                .expect("fits");
        }
        lane.on_write_error();
        lane.assign_and_retain(FROM, CLASS, b"d", false, None)
            .expect("fits");
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
        let mut lane = ReliableLaneSender::new(PEER, 7, BIG_CAP);
        let at_cap = vec![0u8; vd_wire::framing::MAX_STREAM_FRAME_BYTES as usize];
        assert_eq!(
            lane.assign_and_retain(FROM, CLASS, &at_cap, false, None),
            Err(AssignReject::Unframable)
        );
        assert!(
            lane.retry.is_empty(),
            "an un-framable frame is NOT retained"
        );
        assert_eq!(lane.next_seq, 0, "a rejected assign does not burn a seq");
        // The lane is NOT poisoned: a normal frame after a rejection assigns cleanly at seq 0.
        assert_eq!(
            lane.assign_and_retain(FROM, CLASS, b"ok", false, None),
            Ok(0)
        );
    }

    /// The worst-case (epoch=u32::MAX) framed length of a `bytes` payload — the SAME number assign stores as
    /// `framed_len` (so a test can size the cap exactly).
    fn framed_len(bytes: &[u8]) -> usize {
        vd_wire::framing::encode_frame(&ReliableFrame {
            from: FROM,
            class: CLASS,
            incarnation: 7,
            epoch: u32::MAX,
            seq: 0,
            bytes: bytes.to_vec(),
        })
        .expect("frames")
        .len()
    }

    #[test]
    fn a_full_retry_buffer_refuses_the_new_send_as_producer_backpressure() {
        // R-4b: the receiver is strictly contiguous, so a retained frame can NEVER be safely shed — the only
        // loss-safe shed is to refuse the NEW send when the byte cap is reached. A cap that holds exactly two
        // frames fills after two assigns; the third returns BufferFull WITHOUT retaining or burning a seq, and
        // the retained window + retry_bytes are untouched (NOTHING dropped). Backpressure is transient: an ack
        // frees a slot and the next send is accepted.
        let one = framed_len(b"hello");
        let mut lane = ReliableLaneSender::new(PEER, 7, one * 2);
        assert_eq!(
            lane.assign_and_retain(FROM, CLASS, b"hello", false, None),
            Ok(0)
        );
        assert_eq!(lane.retry_bytes, one, "one frame accounted");
        assert_eq!(
            lane.assign_and_retain(FROM, CLASS, b"hello", false, None),
            Ok(1)
        );
        assert_eq!(lane.retry_bytes, one * 2, "two frames = full");
        assert_eq!(
            lane.assign_and_retain(FROM, CLASS, b"hello", false, None),
            Err(AssignReject::BufferFull)
        );
        assert_eq!(lane.next_seq, 2, "a refused send does not burn a seq");
        assert_eq!(lane.retry.len(), 2, "nothing new retained");
        assert_eq!(
            lane.retry_bytes,
            one * 2,
            "retry_bytes unchanged — NO retained frame dropped (producer backpressure, not a shed)"
        );
        lane.on_ack(7, 0, 0, None); // retire seq0 -> a slot frees
        assert_eq!(lane.retry_bytes, one, "one frame freed");
        assert_eq!(
            lane.assign_and_retain(FROM, CLASS, b"hello", false, None),
            Ok(2),
            "backpressure is transient: a freed slot accepts the next send"
        );
    }

    #[test]
    fn retry_bytes_stays_in_lockstep_with_retry_across_assign_ack_and_re_stamp() {
        // R-4b (the H3 accounting invariant): retry_bytes == the sum of every retained frame's FROZEN
        // framed_len — invariant across assigns, acks, AND an epoch re-stamp (framed_len is the worst-case
        // length, so it never drifts even as the on-wire epoch varint grows).
        let mut lane = ReliableLaneSender::new(PEER, 7, BIG_CAP);
        for b in [b"a".as_slice(), b"bb", b"ccc", b"dddd"] {
            lane.assign_and_retain(FROM, CLASS, b, false, None)
                .expect("fits");
        }
        let sum = |l: &ReliableLaneSender| {
            l.retry
                .values()
                .map(|rf| rf.framed_len as usize)
                .sum::<usize>()
        };
        assert_eq!(lane.retry_bytes, sum(&lane), "assign accounting");
        lane.on_write_error(); // epoch -> 1: replay would re-stamp, but framed_len is frozen
        assert_eq!(
            lane.retry_bytes,
            sum(&lane),
            "an epoch re-stamp never drifts retry_bytes (H3)"
        );
        lane.on_ack(7, 1, 1, None); // retire seq0,1 at the bumped epoch
        assert_eq!(lane.retry_bytes, sum(&lane), "retire accounting");
        assert_eq!(lane.retry.len(), 2);
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
                ReliableLaneSender::new(PEER, u64::MAX, BIG_CAP)
                    .assign_and_retain(FROM, CLASS, &vec![0u8; n], false, None)
                    .is_ok()
            })
            .expect("some near-cap payload is accepted");
        let mut lane = ReliableLaneSender::new(PEER, u64::MAX, BIG_CAP);
        let seq = lane
            .assign_and_retain(FROM, CLASS, &vec![0u8; accepted], false, None)
            .expect("accepted");
        let mut replayed = lane.retry[&seq].frame.clone();
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
        let mut a = ReliableLaneSender::new(PEER, 5, BIG_CAP);
        a.assign_and_retain(FROM, MsgClass::Control, b"x", false, None)
            .expect("fits");
        a.on_write_error(); // A -> epoch 1
        let mut b = ReliableLaneSender::new(PEER, 5, BIG_CAP);
        b.assign_and_retain(FROM, MsgClass::Saga, b"y", false, None)
            .expect("fits");
        assert_eq!(
            b.retry[&0].frame.epoch, 0,
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

    #[test]
    fn retry_buffer_cap_must_hold_one_maximal_framed_frame() {
        // R-4b: the byte cap is on the ON-WIRE framed_len (total_len + the 4-byte length prefix). validate must
        // reject a cap that could not retain ONE maximal framed frame (review aa95e10c off-by-envelope):
        // `MAX` alone is short by the prefix; `MAX + size_of::<u32>()` is the minimum that boots.
        let max = vd_wire::framing::MAX_STREAM_FRAME_BYTES as usize;
        let at_max = MeshReliabilityTuning {
            retry_buffer_max_bytes: max,
            ..MeshReliabilityTuning::default()
        };
        assert_eq!(
            at_max
                .validate()
                .expect_err("MAX alone is short by the 4-byte prefix"),
            MeshReliabilityTuningError::RetryBufferTooSmall
        );
        let one_frame = MeshReliabilityTuning {
            retry_buffer_max_bytes: max + std::mem::size_of::<u32>(),
            ..MeshReliabilityTuning::default()
        };
        assert_eq!(one_frame.validate(), Ok(()));
    }

    // ---- Pure classify_reliable receiver-verdict tests (tokio-free — the R-3' contiguity correctness core) ----

    fn rs(incarnation: u64, epoch: u32, hw: u64, primed: bool) -> RecvState {
        RecvState {
            incarnation,
            epoch,
            hw,
            primed,
        }
    }

    #[test]
    fn classify_primes_at_seq0_then_advances_contiguously() {
        let mut st = rs(5, 0, 0, false);
        assert_eq!(classify_reliable(&mut st, 5, 0, 0), Verdict::Accept);
        assert_eq!((st.hw, st.primed), (0, true));
        assert_eq!(classify_reliable(&mut st, 5, 0, 1), Verdict::Accept);
        assert_eq!(classify_reliable(&mut st, 5, 0, 2), Verdict::Accept);
        assert_eq!(st.hw, 2);
    }

    #[test]
    fn classify_dedups_at_and_below_hw_without_moving_hw() {
        let mut st = rs(5, 0, 3, true);
        assert_eq!(classify_reliable(&mut st, 5, 0, 3), Verdict::Dedup);
        assert_eq!(classify_reliable(&mut st, 5, 0, 0), Verdict::Dedup);
        assert_eq!(st.hw, 3, "a dedup never moves the high-water");
    }

    #[test]
    fn classify_gaps_above_hw_plus_one_without_moving_hw() {
        let mut st = rs(5, 0, 3, true);
        assert_eq!(classify_reliable(&mut st, 5, 0, 5), Verdict::Gap);
        assert_eq!(st.hw, 3, "a gap never moves the high-water");
    }

    #[test]
    fn classify_drops_stale_epoch_before_any_hw_compare() {
        // THE reverted-R-1 CRITICAL cure (a6f8e7d/731b377): an old-stream straggler (lower epoch) is dropped
        // BEFORE the hw compare, so it can never advance the watermark past a never-delivered low.
        let mut st = rs(5, 3, 10, true);
        assert_eq!(classify_reliable(&mut st, 5, 2, 999), Verdict::StaleEpoch);
        assert_eq!(
            (st.epoch, st.hw),
            (3, 10),
            "a stale-epoch straggler leaves epoch AND hw untouched"
        );
    }

    #[test]
    fn classify_adopts_a_higher_epoch_and_carries_hw() {
        let mut st = rs(5, 3, 10, true);
        // A redial replays from base==hw+1 at the new epoch — contiguous, hw CARRIED (never reset).
        assert_eq!(classify_reliable(&mut st, 5, 4, 11), Verdict::Accept);
        assert_eq!((st.epoch, st.hw), (4, 11));
        // A replay of an already-delivered seq at the adopted epoch dedups.
        assert_eq!(classify_reliable(&mut st, 5, 4, 8), Verdict::Dedup);
        assert_eq!(st.hw, 11);
    }

    #[test]
    fn classify_drops_a_lower_incarnation_straggler_leaving_state_untouched() {
        let mut st = rs(5, 2, 7, true);
        let before = st;
        assert_eq!(
            classify_reliable(&mut st, 4, 0, 0),
            Verdict::StaleIncarnation
        );
        assert_eq!(
            st, before,
            "a straggler from a since-restarted sender changes nothing"
        );
    }

    #[test]
    fn classify_higher_incarnation_reset_window_reorder_is_gap_not_dedup() {
        // CRITICAL#1 fix: at a fresh incarnation, a reset-window reorder (seq2 before seq1) surfaces as Gap,
        // NEVER a silent Dedup that buries a lower never-delivered seq (the reverted CRITICAL's twin).
        let mut st = rs(5, 7, 10, true);
        assert_eq!(
            classify_reliable(&mut st, 6, 0, 0),
            Verdict::Reset,
            "a fresh incarnation primes at seq0"
        );
        assert_eq!(
            (st.incarnation, st.epoch, st.hw, st.primed),
            (6, 0, 0, true)
        );
        assert_eq!(
            classify_reliable(&mut st, 6, 0, 2),
            Verdict::Gap,
            "seq2 before seq1 ⇒ Gap, NOT a silent Dedup"
        );
        assert_eq!(st.hw, 0, "the gap did not advance hw");
        assert_eq!(
            classify_reliable(&mut st, 6, 0, 1),
            Verdict::Accept,
            "seq1 fills the gap"
        );
        assert_eq!(st.hw, 1);
    }

    #[test]
    fn classify_fresh_incarnation_first_frame_seq_gt_0_is_gap_and_unprimed() {
        // A restarted sender ALWAYS resets next_seq to 0 — a fresh-incarnation first-frame seq>0 is a genuine
        // reset-window loss, never a prime.
        let mut st = rs(5, 0, 4, true);
        assert_eq!(classify_reliable(&mut st, 9, 0, 3), Verdict::Gap);
        assert_eq!(
            (st.incarnation, st.hw, st.primed),
            (9, 0, false),
            "the reset shape, not primed"
        );
    }

    #[test]
    fn classify_fresh_entry_adopts_forward_from_a_nonzero_first_seq() {
        // A brand-new ENTRY (not a restart: inc == recorded) whose first observed frame is seq5 trusts the
        // sender's delivery frontier (documented adopt-forward).
        let mut st = rs(5, 0, 0, false);
        assert_eq!(classify_reliable(&mut st, 5, 0, 5), Verdict::Accept);
        assert_eq!((st.hw, st.primed), (5, true));
    }

    // ---- Pure ReliableLaneSender::on_ack retire tests (the sender half of the R-3' cumulative ack) ----

    #[test]
    fn on_ack_retires_the_acked_prefix_and_advances_base() {
        let mut lane = ReliableLaneSender::new(PEER, 7, BIG_CAP);
        for b in [b"a".as_slice(), b"b", b"c", b"d"] {
            lane.assign_and_retain(FROM, CLASS, b, false, None)
                .expect("fits");
        }
        assert_eq!(lane.on_ack(7, 0, 1, None), 2, "retires seq 0 and 1"); // epoch 0 == lane epoch
        assert_eq!(lane.base, 2);
        assert_eq!(lane.retry.keys().copied().collect::<Vec<_>>(), vec![2, 3]);
    }

    #[test]
    fn on_ack_ignores_a_stale_epoch_ack() {
        let mut lane = ReliableLaneSender::new(PEER, 7, BIG_CAP);
        lane.assign_and_retain(FROM, CLASS, b"a", false, None)
            .expect("fits");
        lane.on_write_error(); // epoch -> 1
        lane.assign_and_retain(FROM, CLASS, b"b", false, None)
            .expect("fits");
        assert_eq!(
            lane.on_ack(7, 0, 5, None),
            0,
            "an ack at the OLD epoch 0 retires nothing"
        );
        assert_eq!(lane.base, 0, "a stale-epoch ack retires nothing");
        assert_eq!(lane.retry.len(), 2);
    }

    #[test]
    fn on_ack_ignores_a_prior_incarnation_ack() {
        let mut lane = ReliableLaneSender::new(PEER, 7, BIG_CAP);
        lane.assign_and_retain(FROM, CLASS, b"a", false, None)
            .expect("fits");
        assert_eq!(
            lane.on_ack(6, 0, 0, None),
            0,
            "an ack against a DIFFERENT incarnation retires nothing"
        );
        assert_eq!(lane.base, 0);
    }

    #[test]
    fn on_ack_is_monotone_a_lower_ack_never_rolls_base_back() {
        let mut lane = ReliableLaneSender::new(PEER, 7, BIG_CAP);
        for b in [b"a".as_slice(), b"b", b"c", b"d", b"e", b"f"] {
            lane.assign_and_retain(FROM, CLASS, b, false, None)
                .expect("fits");
        }
        assert_eq!(lane.on_ack(7, 0, 4, None), 5, "retire 0..=4"); // base -> 5
        assert_eq!(lane.base, 5);
        assert_eq!(
            lane.on_ack(7, 0, 1, None),
            0,
            "a reordered LOWER ack retires nothing"
        );
        assert_eq!(lane.base, 5, "base is monotone");
        assert_eq!(lane.retry.keys().copied().collect::<Vec<_>>(), vec![5]);
    }

    #[test]
    fn on_ack_clamps_to_next_seq_so_base_never_overruns() {
        let mut lane = ReliableLaneSender::new(PEER, 7, BIG_CAP);
        lane.assign_and_retain(FROM, CLASS, b"a", false, None)
            .expect("fits");
        lane.assign_and_retain(FROM, CLASS, b"b", false, None)
            .expect("fits");
        assert_eq!(
            lane.on_ack(7, 0, 999, None),
            2,
            "a forged/torn ack_through is clamped to next_seq"
        );
        assert_eq!(lane.base, lane.next_seq, "base clamped to next_seq");
        assert_eq!(lane.base, 2);
        assert!(lane.retry.is_empty(), "everything assigned was acked");
    }

    #[test]
    fn replay_batch_starts_at_base_after_a_retire() {
        let mut lane = ReliableLaneSender::new(PEER, 7, BIG_CAP);
        for b in [b"a".as_slice(), b"b", b"c"] {
            lane.assign_and_retain(FROM, CLASS, b, false, None)
                .expect("fits");
        }
        lane.on_ack(7, 0, 0, None); // retire seq0, base -> 1
        lane.on_write_error(); // epoch -> 1
        let batch = lane.replay_batch();
        assert_eq!(
            batch.iter().map(|f| f.seq).collect::<Vec<_>>(),
            vec![1, 2],
            "replay starts at base, not 0"
        );
        assert!(batch.iter().all(|f| f.epoch == 1));
    }

    #[test]
    fn on_ack_never_advances_base_past_next_seq_under_interleaving() {
        // A bounded deterministic sweep of {assign, on_ack(current epoch, ahead-of-window), write_error,
        // on_ack(stale epoch)}: base <= next_seq must hold ALWAYS (so the first replayed frame is
        // Accept-or-Dedup, never a Gap) and retry stays exactly base..next_seq.
        let mut lane = ReliableLaneSender::new(PEER, 1, BIG_CAP);
        for step in 0u64..200 {
            match step % 4 {
                0 => {
                    lane.assign_and_retain(FROM, CLASS, b"x", false, None)
                        .expect("fits");
                }
                1 => {
                    lane.on_ack(1, lane.epoch, step / 2, None); // an ack sometimes ahead of next_seq
                }
                2 => lane.on_write_error(),
                _ => {
                    lane.on_ack(1, lane.epoch.wrapping_sub(1), step, None); // a stale-epoch ack: ignored
                }
            }
            assert!(
                lane.base <= lane.next_seq,
                "base must never overrun next_seq (step {step})"
            );
            assert!(
                lane.retry
                    .keys()
                    .all(|&k| k >= lane.base && k < lane.next_seq),
                "retry stays exactly base..next_seq (step {step})"
            );
        }
    }

    // ---- Pure R-4a lane-owes / failure-counter tests (the retransmit-timer guard + confirm-dead core) ----

    #[test]
    fn a_lane_owes_redelivery_iff_stream_closed_and_window_non_empty() {
        let mut lane = ReliableLaneSender::new(PEER, 1, BIG_CAP);
        assert!(
            !lane.owes_redelivery(),
            "a fresh lane (empty window) owes nothing"
        );
        lane.assign_and_retain(FROM, CLASS, b"x", false, None)
            .expect("fits");
        assert!(
            lane.owes_redelivery(),
            "closed stream + non-empty window ⇒ owes a redelivery"
        );
        lane.on_ack(1, 0, 0, None); // retire the only frame
        assert!(
            !lane.owes_redelivery(),
            "a drained window owes nothing (the timer stops)"
        );
    }

    #[test]
    fn lane_failure_counter_bumps_saturating_and_resets_per_lane() {
        let mut lane = ReliableLaneSender::new(PEER, 1, BIG_CAP);
        assert_eq!(lane.consecutive_failures, 0);
        lane.on_replay_failed();
        lane.on_replay_failed();
        assert_eq!(lane.consecutive_failures, 2, "one bump per failed replay");
        lane.on_replay_ok();
        assert_eq!(
            lane.consecutive_failures, 0,
            "a lane's OWN Ok resets ONLY its counter (the H2 cure)"
        );
    }

    #[test]
    fn any_lane_owes_reflects_the_owing_lanes() {
        let mut lanes: BTreeMap<MsgClass, ReliableLaneSender> = BTreeMap::new();
        assert!(!any_lane_owes(&lanes), "no lanes ⇒ nothing owes");
        let mut lane = ReliableLaneSender::new(PEER, 1, BIG_CAP);
        lane.assign_and_retain(FROM, CLASS, b"x", false, None)
            .expect("fits");
        lanes.insert(CLASS, lane);
        assert!(
            any_lane_owes(&lanes),
            "an owing lane ⇒ the timer guard is live"
        );
        lanes.get_mut(&CLASS).expect("lane").on_ack(1, 0, 0, None); // drain it
        assert!(
            !any_lane_owes(&lanes),
            "a drained lane ⇒ the timer guard goes false"
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
                &MeshConfig::new(id, addr, book.clone(), capacity, 0, 0),
                None,
            )
            .expect("mesh node");
            transports.push(t);
            controls.push(c);
        }
        (transports, controls)
    }

    #[test]
    fn a_booked_peer_with_no_lane_is_dialed_on_the_first_send_and_an_unbooked_one_is_refused() {
        // THE PEER BOOK, the mesh half (D-RLM-6 mechanism C): A knows nobody. A send to B is refused as
        // UnknownPeer — the trigger the node runtime asks on. Once A books B's address through the
        // seam, the SAME send dials a lane lazily and B receives the frame.
        let rt = runtime();
        let trust = ClusterTrust::generate("vd-mesh-test").expect("trust");
        let a = NodeId(1);
        let b = NodeId(2);
        let sock_a = std::net::UdpSocket::bind("127.0.0.1:0").expect("reserve a");
        let sock_b = std::net::UdpSocket::bind("127.0.0.1:0").expect("reserve b");
        let addr_a = sock_a.local_addr().expect("addr a");
        let addr_b = sock_b.local_addr().expect("addr b");
        drop(sock_a);
        drop(sock_b);
        let (mut ta, ctl_a) = spawn_mesh(
            rt.handle(),
            &trust,
            &MeshConfig::new(a, addr_a, BTreeMap::new(), 64, 0, 0),
            None,
        )
        .expect("spawn A");
        let (mut tb, _ctl_b) = spawn_mesh(
            rt.handle(),
            &trust,
            &MeshConfig::new(b, addr_b, BTreeMap::new(), 64, 0, 0),
            None,
        )
        .expect("spawn B");
        assert_eq!(
            ta.send(b, MsgClass::Control, vec![0xA].into()),
            Err(SendError::UnknownPeer(vec![0xA].into())),
            "no lane, no address: refused by name"
        );
        let ip = match addr_b.ip() {
            std::net::IpAddr::V4(v4) => v4.to_ipv6_mapped().octets(),
            std::net::IpAddr::V6(v6) => v6.octets(),
        };
        ta.book_peer(b, ip, addr_b.port());
        ta.send(b, MsgClass::Control, vec![0xA].into())
            .expect("booked: the send dials the lane itself");
        let got_b = wait_for(&mut tb, |g| {
            g.iter().any(
                |m| matches!(m, Inbound::Wire { from, bytes, .. } if *from == a && bytes[0] == 0xA),
            )
        });
        assert!(
            !got_b.is_empty(),
            "B received A's frame over the lazily dialed lane"
        );
        let stats = ctl_a.stats();
        assert_eq!((stats.peers_booked, stats.lazy_dials), (1, 1));
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
                // This dead-peer test drives no oversize/buffer-full sends, so no SendShed arises;
                // and every peer here is BOOKED, so no PeerReset arises either. Both arms are present
                // for Inbound exhaustiveness (R-4d M3).
                Inbound::Wire { .. } | Inbound::SendShed { .. } | Inbound::PeerReset { .. } => None,
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
                Err(SendError::UnknownPeer(_)) => panic!("a booked peer is never unknown"),
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
        // Since the peer book (D-RLM-6 mechanism C, 2026-09-03) a destination with no lane and no
        // address is refused as UNKNOWN — the frame is handed back so the node keeps it and asks
        // the orchestrator where the peer listens — never as a full queue.
        assert_eq!(err, SendError::UnknownPeer(vec![7].into()));
    }

    /// CA-1 reply-on-connection (Stage 1+2) — the headline gate. A does NOT book B; B books+dials A. Once B
    /// sends (A LEARNS B's return connection on the first authenticated frame), A can reply to the UNBOOKED B
    /// over that held accepted connection — DELIVERED (gate i) AND ACKED (gate ii: A's lazily-spawned learned
    /// lane retires its window over the same connection's reverse path), with NO false `NodeUnreachable` at the
    /// live B. Asymmetric ⇒ built by hand (not `cluster()`, which books everyone symmetrically).
    #[test]
    fn ca1_reply_on_connection_reaches_a_peer_not_in_the_book() {
        let rt = runtime();
        let trust = ClusterTrust::generate("vd-mesh-test").expect("trust");
        let a = NodeId(1);
        let b = NodeId(2);
        // Reserve two ephemeral ports (bind + drop, as `cluster()` does) so quinn can bind them.
        let sock_a = std::net::UdpSocket::bind("127.0.0.1:0").expect("reserve a");
        let sock_b = std::net::UdpSocket::bind("127.0.0.1:0").expect("reserve b");
        let addr_a = sock_a.local_addr().expect("addr a");
        let addr_b = sock_b.local_addr().expect("addr b");
        drop(sock_a);
        drop(sock_b);
        // A: EMPTY book (does not know B). B: books A (so B dials A).
        let (mut ta, ctl_a) = spawn_mesh(
            rt.handle(),
            &trust,
            &MeshConfig::new(a, addr_a, BTreeMap::new(), 64, 0, 0),
            None,
        )
        .expect("spawn A");
        let book_b: BTreeMap<NodeId, SocketAddr> = [(a, addr_a)].into_iter().collect();
        let (mut tb, _ctl_b) = spawn_mesh(
            rt.handle(),
            &trust,
            &MeshConfig::new(b, addr_b, book_b, 64, 0, 0),
            None,
        )
        .expect("spawn B");

        // B → A over B's booked lane: A LEARNS B's return connection on this first authenticated frame.
        tb.send(a, MsgClass::Control, vec![0xB].into())
            .expect("B books A, so B->A is a normal booked send");
        let got_a = wait_for(&mut ta, |g| {
            g.iter().any(
                |m| matches!(m, Inbound::Wire { from, bytes, .. } if *from == b && bytes[0] == 0xB),
            )
        });
        assert!(
            !got_a.is_empty(),
            "A received B's first frame (and learned B)"
        );

        // A → B: B is UNBOOKED in A, but LEARNED ⇒ the send is Ok (a lazily-spawned learned lane), not QueueFull.
        ta.send(b, MsgClass::Control, vec![0xA].into())
            .expect("CA-1: an inbound-learned peer is replyable without pre-booking");

        // GATE (i): B RECEIVES A's reply over the held (accepted-by-A / dialed-by-B) connection.
        let got_b = wait_for(&mut tb, |g| {
            g.iter().any(
                |m| matches!(m, Inbound::Wire { from, bytes, .. } if *from == a && bytes[0] == 0xA),
            )
        });
        assert!(
            !got_b.is_empty(),
            "CA-1 GATE (i): B received A's reply over the learned connection"
        );

        // GATE (ii): A's learned lane RETIRES its window (the round trip closes over the reverse-ack path) AND A
        // never false-bounces `NodeUnreachable` at the LIVE B (which a broken reverse-ack path would do). The
        // stats Arc is shared with `ctl_a`, so a wrong-Arc stall is distinguishable from a real dead-ack path.
        let started = Instant::now();
        loop {
            if ctl_a.stats().reliable_acked >= 1 {
                break;
            }
            let bounced = ta
                .drain_inbound()
                .into_iter()
                .any(|m| matches!(m, Inbound::NodeUnreachable { to, .. } if to == b));
            assert!(
                !bounced,
                "CA-1: A false-bounced NodeUnreachable at the LIVE learned peer B"
            );
            std::thread::sleep(Duration::from_millis(5));
            assert!(
                started.elapsed() < DEADLINE,
                "CA-1 GATE (ii): A's learned lane never retired (reliable_acked stuck at 0)"
            );
        }
    }

    /// D-18 REPRO (the gateway→client snapshot path): CA-1 reply-on-connection must deliver an UNRELIABLE
    /// SNAPSHOT DATAGRAM to a LEARNED (unbooked) peer — not just a reliable Control frame. This is the exact
    /// k3d failure (an unbooked in-cluster client gets 0 snapshots while login/Active — reliable frames over the
    /// SAME learned lane — works). Pure loopback quinn: NO Docker Desktop, NO k3d. If B never receives the
    /// datagram, the drop counters pin the failure mode (too_large ⇒ max_datagram_size None on the accepted
    /// reply connection; dropped_send ⇒ send_datagram errored; both 0 ⇒ the frame never reached the datagram arm).
    #[test]
    fn ca1_reply_on_connection_delivers_an_unreliable_snapshot_datagram() {
        let rt = runtime();
        let trust = ClusterTrust::generate("vd-mesh-test").expect("trust");
        let a = NodeId(1);
        let b = NodeId(2);
        let sock_a = std::net::UdpSocket::bind("127.0.0.1:0").expect("reserve a");
        let sock_b = std::net::UdpSocket::bind("127.0.0.1:0").expect("reserve b");
        let addr_a = sock_a.local_addr().expect("addr a");
        let addr_b = sock_b.local_addr().expect("addr b");
        drop(sock_a);
        drop(sock_b);
        // A (gateway): EMPTY book. B (client): books A, so B dials A.
        let (mut ta, ctl_a) = spawn_mesh(
            rt.handle(),
            &trust,
            &MeshConfig::new(a, addr_a, BTreeMap::new(), 64, 0, 0),
            None,
        )
        .expect("spawn A");
        let book_b: BTreeMap<NodeId, SocketAddr> = [(a, addr_a)].into_iter().collect();
        let (mut tb, _ctl_b) = spawn_mesh(
            rt.handle(),
            &trust,
            &MeshConfig::new(b, addr_b, book_b, 64, 0, 0),
            None,
        )
        .expect("spawn B");

        // B → A (reliable Control): A LEARNS B's return connection (exactly as a client's login does).
        tb.send(a, MsgClass::Control, vec![0xB].into())
            .expect("B->A booked send");
        let got_a = wait_for(&mut ta, |g| {
            g.iter().any(
                |m| matches!(m, Inbound::Wire { from, bytes, .. } if *from == b && bytes[0] == 0xB),
            )
        });
        assert!(!got_a.is_empty(), "A learned B");

        // A → B: UNRELIABLE Snapshot datagrams, re-sent at ~20 Hz (the real fan-out cadence — a fire-and-forget
        // datagram has no retransmit, so an early not-yet-ready-MTU drop must not read as a permanent failure).
        // Assert B receives at least one within the deadline.
        let started = Instant::now();
        let mut received = false;
        while started.elapsed() < DEADLINE {
            ta.send(b, MsgClass::Snapshot, vec![0xC5, 0xC5, 0xC5].into())
                .expect("learned-lane snapshot enqueues");
            if tb.drain_inbound().into_iter().any(
                |m| matches!(m, Inbound::Wire { from, bytes, .. } if from == a && bytes.first() == Some(&0xC5)),
            ) {
                received = true;
                break;
            }
            std::thread::sleep(Duration::from_millis(20));
        }
        let s = ctl_a.stats();
        assert!(
            received,
            "D-18: B never received a learned-lane Snapshot datagram (dropped_too_large={}, dropped_send={})",
            s.datagrams_dropped_too_large, s.datagrams_dropped_send
        );
    }

    #[test]
    fn stream_kind_classifies_the_three_arms() {
        // CA-1 pure classifier: DATA / ACK / unknown-drop (the dispatcher's three routing arms, HR5 per-arm).
        assert_eq!(stream_kind(STREAM_KIND_DATA), Some(StreamKind::Data));
        assert_eq!(stream_kind(STREAM_KIND_ACK), Some(StreamKind::Ack));
        assert_eq!(stream_kind(0x7F), None);
    }

    #[test]
    fn ca1_learned_lane_terminates_when_its_accepted_connection_dies() {
        // HIGH-fix regression (post-impl review wf_5310c8fb): after A learns + replies to B over the held
        // connection, KILLING B must make A's learned lane TERMINATE (loud-bounce + break), NOT hot-spin on a
        // perpetually-Err ack watch. Observable: A.send(B) cleanly returns QueueFull within the deadline (the
        // learned lane closed + the dead connection is refused a doomed respawn) and the test does not hang.
        let rt = runtime();
        let trust = ClusterTrust::generate("vd-mesh-test").expect("trust");
        let a = NodeId(1);
        let b = NodeId(2);
        let sock_a = std::net::UdpSocket::bind("127.0.0.1:0").expect("reserve a");
        let sock_b = std::net::UdpSocket::bind("127.0.0.1:0").expect("reserve b");
        let addr_a = sock_a.local_addr().expect("addr a");
        let addr_b = sock_b.local_addr().expect("addr b");
        drop(sock_a);
        drop(sock_b);
        let (mut ta, _ctl_a) = spawn_mesh(
            rt.handle(),
            &trust,
            &MeshConfig::new(a, addr_a, BTreeMap::new(), 64, 0, 0),
            None,
        )
        .expect("spawn A");
        let book_b: BTreeMap<NodeId, SocketAddr> = [(a, addr_a)].into_iter().collect();
        let (mut tb, ctl_b) = spawn_mesh(
            rt.handle(),
            &trust,
            &MeshConfig::new(b, addr_b, book_b, 64, 0, 0),
            None,
        )
        .expect("spawn B");
        tb.send(a, MsgClass::Control, vec![0xB].into())
            .expect("B->A");
        wait_for(&mut ta, |g| {
            g.iter()
                .any(|m| matches!(m, Inbound::Wire { from, .. } if *from == b))
        });
        ta.send(b, MsgClass::Control, vec![0xA].into())
            .expect("A->B learned");
        wait_for(&mut tb, |g| {
            g.iter()
                .any(|m| matches!(m, Inbound::Wire { from, .. } if *from == a))
        });
        // Kill B ⇒ A's accepted connection from B dies ⇒ A's learned lane's ack watch closes ⇒ it must TERMINATE.
        ctl_b.kill();
        let started = Instant::now();
        loop {
            if let Err(SendError::QueueFull(_)) = ta.send(b, MsgClass::Control, vec![0xA].into()) {
                break;
            }
            std::thread::sleep(Duration::from_millis(10));
            assert!(
                started.elapsed() < DEADLINE,
                "A never returned QueueFull to the dead learned peer B — the learned lane wedged (hot-spin)?"
            );
        }
    }

    #[test]
    fn ca1_learned_peers_table_cap_rejects_and_counts() {
        // A has an EMPTY book + learned_peers_max = 1. B and C both book+dial A. A learns the FIRST, then
        // REJECTS the second at the cap — counted LOUD (learned_peers_rejected), never silent.
        let rt = runtime();
        let trust = ClusterTrust::generate("vd-mesh-test").expect("trust");
        let (a, b, c) = (NodeId(1), NodeId(2), NodeId(3));
        let sa = std::net::UdpSocket::bind("127.0.0.1:0").expect("reserve a");
        let sb = std::net::UdpSocket::bind("127.0.0.1:0").expect("reserve b");
        let sc = std::net::UdpSocket::bind("127.0.0.1:0").expect("reserve c");
        let addr_a = sa.local_addr().expect("addr a");
        let addr_b = sb.local_addr().expect("addr b");
        let addr_c = sc.local_addr().expect("addr c");
        drop(sa);
        drop(sb);
        drop(sc);
        let mut cfg_a = MeshConfig::new(a, addr_a, BTreeMap::new(), 64, 0, 0);
        cfg_a.learned_peers_max = 1;
        let (mut ta, ctl_a) = spawn_mesh(rt.handle(), &trust, &cfg_a, None).expect("spawn A");
        let book: BTreeMap<NodeId, SocketAddr> = [(a, addr_a)].into_iter().collect();
        let (mut tb, _cb) = spawn_mesh(
            rt.handle(),
            &trust,
            &MeshConfig::new(b, addr_b, book.clone(), 64, 0, 0),
            None,
        )
        .expect("spawn B");
        let (mut tc, _cc) = spawn_mesh(
            rt.handle(),
            &trust,
            &MeshConfig::new(c, addr_c, book, 64, 0, 0),
            None,
        )
        .expect("spawn C");
        let _ = (&mut tb, &mut tc); // held so their tasks live
        tb.send(a, MsgClass::Control, vec![0xB].into())
            .expect("B->A");
        tc.send(a, MsgClass::Control, vec![0xC].into())
            .expect("C->A");
        wait_for(&mut ta, |g| {
            g.iter()
                .filter(|m| matches!(m, Inbound::Wire { .. }))
                .count()
                >= 2
        });
        let started = Instant::now();
        loop {
            if ctl_a.stats().learned_peers_rejected >= 1 {
                break;
            }
            std::thread::sleep(Duration::from_millis(5));
            assert!(
                started.elapsed() < DEADLINE,
                "learned_peers_rejected never incremented despite a 2nd distinct learned peer at cap=1"
            );
        }
    }

    /// Reserve N ephemeral loopback ports (bind + drop, as `cluster()` does) so quinn can bind them.
    /// The asymmetric CA-1 tests build their nodes by hand, and every one of them needs this.
    fn reserve_addrs<const N: usize>() -> [SocketAddr; N] {
        // Hold EVERY socket while the addresses are read, and drop them only afterwards. Binding and
        // dropping one at a time lets the OS hand the SAME port back for the next one, and two nodes on
        // one port is an AddrInUse at spawn.
        let socks: [std::net::UdpSocket; N] = std::array::from_fn(|_| {
            std::net::UdpSocket::bind("127.0.0.1:0").expect("reserve a loopback port")
        });
        let addrs = std::array::from_fn(|i| socks[i].local_addr().expect("the reserved address"));
        drop(socks);
        addrs
    }

    /// ★ THE VANISHED CLIENT, HALF ONE (2026-09-05): a dial-in peer that dies owing NOTHING is still a
    /// PEER RESET.
    ///
    /// The gate: an UNBOOKED peer dials in, this node replies once, and the reply is ACKED — so the lane's
    /// retry window is EMPTY. Then the peer is killed. The old code walked the empty window, bounced nothing
    /// and went quiet, so the gateway kept a session for a process that no longer existed and the next login
    /// for that node id wedged. Now the death itself is the news: `PeerReset { ConnectionLost }`, and NO
    /// `NodeUnreachable` (nothing was owed, so nothing was lost).
    #[test]
    fn a_dialed_in_peers_death_is_a_peer_reset_even_with_nothing_owed() {
        let rt = runtime();
        let trust = ClusterTrust::generate("vd-mesh-test").expect("trust");
        let a = NodeId(1);
        let b = NodeId(2);
        let [addr_a, addr_b] = reserve_addrs();
        // A: EMPTY book (it never learns B's address, so it can never re-dial B). B: books A and dials it.
        let (mut ta, ctl_a) = spawn_mesh(
            rt.handle(),
            &trust,
            &MeshConfig::new(a, addr_a, BTreeMap::new(), 64, 0, 0),
            None,
        )
        .expect("spawn A");
        let book_b: BTreeMap<NodeId, SocketAddr> = [(a, addr_a)].into_iter().collect();
        let (mut tb, ctl_b) = spawn_mesh(
            rt.handle(),
            &trust,
            &MeshConfig::new(b, addr_b, book_b, 64, 0, 0),
            None,
        )
        .expect("spawn B");

        tb.send(a, MsgClass::Control, vec![0xB].into())
            .expect("B->A booked send");
        wait_for(&mut ta, |g| {
            g.iter()
                .any(|m| matches!(m, Inbound::Wire { from, .. } if *from == b))
        });
        // A replies once over the learned connection, and WAITS for the ack — that is what empties the
        // window, which is the whole point of this test.
        ta.send(b, MsgClass::Control, vec![0xA].into())
            .expect("A->B learned");
        let started = Instant::now();
        while ctl_a.stats().reliable_acked == 0 {
            std::thread::sleep(Duration::from_millis(5));
            assert!(
                started.elapsed() < DEADLINE,
                "A's learned lane never retired its only frame, so the window is not empty"
            );
        }

        ctl_b.kill();
        let got = wait_for(&mut ta, |g| {
            g.iter().any(|m| matches!(m, Inbound::PeerReset { .. }))
        });
        let resets: Vec<Inbound> = got
            .iter()
            .filter(|m| matches!(m, Inbound::PeerReset { .. }))
            .cloned()
            .collect();
        assert_eq!(
            resets,
            vec![Inbound::PeerReset {
                node: b,
                cause: PeerResetCause::ConnectionLost,
            }],
            "a dead dial-in peer names itself and says its connection is gone"
        );
        let bounces: Vec<Inbound> = got
            .iter()
            .filter(|m| matches!(m, Inbound::NodeUnreachable { .. }))
            .cloned()
            .collect();
        assert_eq!(
            bounces,
            Vec::new(),
            "nothing was owed to B, so nothing may be reported undelivered"
        );
    }

    /// M6 unit half — the eviction RULE on its own, all three answers, no sockets.
    ///
    /// A client (unbooked) that is present LOSES its ledger and is counted once. A shard (booked) KEEPS its
    /// ledger and is never counted — it is re-dialed, and its flows resume against the same watermarks. A peer
    /// with no entry at all (this node never received a frame from it) is a no-op that counts nothing.
    #[test]
    fn evicting_a_receive_ledger_spares_a_booked_peer_and_counts_only_a_real_removal() {
        let client = NodeId(7);
        let shard = NodeId(8);
        let stranger = NodeId(9);
        let ledger: RecvLedger = Arc::new(RwLock::new(BTreeMap::new()));
        let seed = |peer: NodeId| {
            ledger.write().expect("ledger").insert(
                peer,
                Arc::new(Mutex::new(PeerRecv {
                    incarnation: Some(1),
                    classes: BTreeMap::new(),
                })),
            );
        };
        seed(client);
        seed(shard);
        // Only the shard has a booked address: the client dialed in, so this node cannot reach it.
        let book: PeerTopology = Arc::new(ArcSwap::from_pointee(
            [(shard, "127.0.0.1:1".parse::<SocketAddr>().expect("addr"))]
                .into_iter()
                .collect::<BTreeMap<NodeId, SocketAddr>>(),
        ));
        let stats = Arc::new(MeshStats::default());

        evict_recv_ledger(&ledger, &book, shard, &stats);
        assert!(
            ledger.read().expect("ledger").contains_key(&shard),
            "a booked peer keeps its ledger: it is re-dialed and its flows resume"
        );
        assert_eq!(stats.recv_ledgers_evicted.load(Ordering::Relaxed), 0);

        evict_recv_ledger(&ledger, &book, client, &stats);
        assert!(
            !ledger.read().expect("ledger").contains_key(&client),
            "a vanished unbooked peer gives its ledger back"
        );
        assert_eq!(stats.recv_ledgers_evicted.load(Ordering::Relaxed), 1);

        evict_recv_ledger(&ledger, &book, stranger, &stats);
        assert_eq!(
            stats.recv_ledgers_evicted.load(Ordering::Relaxed),
            1,
            "a peer that never sent anything has no entry, so nothing is removed and nothing is counted"
        );
    }

    /// ★ M6 — THE VANISHED CLIENT'S LEDGER GOES WITH IT. The residual half one left behind: the death was
    /// SAID (`PeerReset { ConnectionLost }`) but the dedup watermarks stayed forever, so a node that served
    /// a thousand clients over a day held a thousand dead entries.
    ///
    /// The gate: an UNBOOKED peer dials in, this node replies (which is what spawns the learned lane), the
    /// peer is killed — and after the reset is drained, the acceptor's ledger no longer names it.
    #[test]
    fn a_vanished_dial_in_peers_receive_ledger_is_evicted() {
        let rt = runtime();
        let trust = ClusterTrust::generate("vd-mesh-test").expect("trust");
        let a = NodeId(1);
        let b = NodeId(2);
        let [addr_a, addr_b] = reserve_addrs();
        // A: EMPTY book — it can never re-dial B, which is exactly what makes B's ledger dead weight.
        let (mut ta, ctl_a) = spawn_mesh(
            rt.handle(),
            &trust,
            &MeshConfig::new(a, addr_a, BTreeMap::new(), 64, 0, 0),
            None,
        )
        .expect("spawn A");
        let book_b: BTreeMap<NodeId, SocketAddr> = [(a, addr_a)].into_iter().collect();
        let (mut tb, ctl_b) = spawn_mesh(
            rt.handle(),
            &trust,
            &MeshConfig::new(b, addr_b, book_b, 64, 0, 0),
            None,
        )
        .expect("spawn B");

        tb.send(a, MsgClass::Control, vec![0xB].into())
            .expect("B->A booked send");
        wait_for(&mut ta, |g| {
            g.iter()
                .any(|m| matches!(m, Inbound::Wire { from, .. } if *from == b))
        });
        assert!(
            ta.ledger.read().expect("ledger").contains_key(&b),
            "A holds B's dedup state the moment B's first frame lands"
        );
        // A replies, which spawns the LEARNED lane whose death carries the eviction.
        ta.send(b, MsgClass::Control, vec![0xA].into())
            .expect("A->B learned");
        let started = Instant::now();
        while ctl_a.stats().reliable_acked == 0 {
            std::thread::sleep(Duration::from_millis(5));
            assert!(
                started.elapsed() < DEADLINE,
                "A's learned lane never retired its only frame"
            );
        }

        ctl_b.kill();
        wait_for(&mut ta, |g| {
            g.iter().any(|m| matches!(m, Inbound::PeerReset { .. }))
        });
        assert_eq!(
            ctl_a.stats().recv_ledgers_evicted,
            1,
            "the vanished client's ledger is given back, exactly once"
        );
        assert!(
            !ta.ledger.read().expect("ledger").contains_key(&b),
            "and A no longer names B in its receive ledger"
        );
    }

    /// ★ M6, THE OTHER HALF: a BOOKED peer that dies KEEPS its ledger. A shard is re-dialed, and its flows
    /// resume against the SAME watermarks — dropping them would re-open the cross-stream hole the node-wide
    /// ledger exists to close. The synchronisation point is the booked peer's own confirmed death, the
    /// `NodeUnreachable` bounce after the replay threshold.
    #[test]
    fn a_booked_peers_death_keeps_its_receive_ledger() {
        let rt = runtime();
        let trust = ClusterTrust::generate("vd-mesh-test").expect("trust");
        let a = NodeId(1);
        let b = NodeId(2);
        let [addr_a, addr_b] = reserve_addrs();
        let book: BTreeMap<NodeId, SocketAddr> = [(a, addr_a), (b, addr_b)].into_iter().collect();
        let (mut ta, ctl_a) = spawn_mesh(
            rt.handle(),
            &trust,
            &MeshConfig::new(a, addr_a, book.clone(), 64, 0, 0),
            None,
        )
        .expect("spawn A");
        let (mut tb, ctl_b) = spawn_mesh(
            rt.handle(),
            &trust,
            &MeshConfig::new(b, addr_b, book, 64, 0, 0),
            None,
        )
        .expect("spawn B");

        tb.send(a, MsgClass::Control, vec![0xB].into())
            .expect("B->A booked send");
        wait_for(&mut ta, |g| {
            g.iter()
                .any(|m| matches!(m, Inbound::Wire { from, .. } if *from == b))
        });
        ctl_b.kill();
        // ONE frame toward the corpse. It is retained, the retransmit clock re-drives it off A's own timer,
        // and the lane bounces once it passes the confirmed-unreachable threshold.
        ta.send(b, MsgClass::Control, vec![0xC].into())
            .expect("A->B booked send toward a killed peer");
        let got = wait_for(&mut ta, |g| {
            g.iter()
                .any(|m| matches!(m, Inbound::NodeUnreachable { .. }))
        });
        assert!(
            got.iter().all(|m| !matches!(
                m,
                Inbound::PeerReset {
                    cause: PeerResetCause::ConnectionLost,
                    ..
                }
            )),
            "a booked peer's death is a bounce, never the dial-in peer's connection-lost reset"
        );
        assert_eq!(
            ctl_a.stats().recv_ledgers_evicted,
            0,
            "a booked peer is re-dialed, so nothing about it is forgotten"
        );
        assert!(
            ta.ledger.read().expect("ledger").contains_key(&b),
            "B's dedup watermarks wait for B to come back"
        );
    }

    /// ★ THE VANISHED CLIENT, HALF TWO (2026-09-05): a RESTARTED dial-in peer is a REINCARNATION, and its
    /// NEW connection wins.
    ///
    /// Two processes speak for node id 2: the first at incarnation 0, the second at incarnation 1, both
    /// dialing in to A while the first is still alive. A must (i) say `PeerReset { Reincarnated }` exactly
    /// ONCE — a restart happens once, not once per class — (ii) count the live entry it replaced, and
    /// (iii) send its next reply to the NEW process, never to the corpse.
    #[test]
    fn a_restarted_dial_in_peer_is_a_reincarnation_and_its_new_connection_wins() {
        let rt = runtime();
        let trust = ClusterTrust::generate("vd-mesh-test").expect("trust");
        let a = NodeId(1);
        let b = NodeId(2);
        let [addr_a, addr_b1, addr_b2] = reserve_addrs();
        let (mut ta, ctl_a) = spawn_mesh(
            rt.handle(),
            &trust,
            &MeshConfig::new(a, addr_a, BTreeMap::new(), 64, 0, 0),
            None,
        )
        .expect("spawn A");
        let book: BTreeMap<NodeId, SocketAddr> = [(a, addr_a)].into_iter().collect();
        // The FIRST process for node id 2, at incarnation 0.
        let (mut tb1, _ctl_b1) = spawn_mesh(
            rt.handle(),
            &trust,
            &MeshConfig::new(b, addr_b1, book.clone(), 64, 0, 0),
            None,
        )
        .expect("spawn B (first process)");
        // The SECOND process for the SAME node id, at incarnation 1. It is NOT a restart of the first in
        // the test's process table — it is a second live process, which is the harder case: the first one's
        // connection is still open and would otherwise keep the reply path.
        let (mut tb2, _ctl_b2) = spawn_mesh(
            rt.handle(),
            &trust,
            &MeshConfig::new(b, addr_b2, book, 64, 1, 0),
            None,
        )
        .expect("spawn B (second process)");

        let mut a_events: Vec<Inbound> = Vec::new();
        tb1.send(a, MsgClass::Control, vec![0xB1].into())
            .expect("B1->A booked send");
        a_events.extend(wait_for(&mut ta, |g| {
            g.iter().any(
                |m| matches!(m, Inbound::Wire { from, bytes, .. } if *from == b && bytes[0] == 0xB1),
            )
        }));

        // A answers the FIRST process, so a learned writer lane toward node 2 is bound to that first
        // connection — the lane the gateway streams a player over. When the second process supersedes
        // it, that lane's connection closes under it; it must retire quietly, never as a loss.
        ta.send(b, MsgClass::Control, vec![0xA1].into())
            .expect("A->B1 learned");
        let _ = wait_for(&mut tb1, |g| {
            g.iter().any(
                |m| matches!(m, Inbound::Wire { from, bytes, .. } if *from == a && bytes[0] == 0xA1),
            )
        });
        tb2.send(a, MsgClass::Control, vec![0xB2].into())
            .expect("B2->A booked send");
        a_events.extend(wait_for(&mut ta, |g| {
            g.iter().any(
                |m| matches!(m, Inbound::Wire { from, bytes, .. } if *from == b && bytes[0] == 0xB2),
            )
        }));

        // (ii) the live entry was replaced, and counted.
        let started = Instant::now();
        while ctl_a.stats().learned_peers_superseded == 0 {
            std::thread::sleep(Duration::from_millis(5));
            assert!(
                started.elapsed() < DEADLINE,
                "A never superseded the first process's learned connection"
            );
        }
        assert_eq!(
            ctl_a.stats().learned_peers_superseded,
            1,
            "one live learned connection was replaced, so the count is one"
        );

        // (iii) A's reply rides the NEW connection.
        ta.send(b, MsgClass::Control, vec![0xA].into())
            .expect("A->B learned");
        let mut b1_events: Vec<Inbound> = Vec::new();
        let started = Instant::now();
        let mut b2_got = false;
        while !b2_got {
            b1_events.extend(tb1.drain_inbound());
            b2_got = tb2.drain_inbound().into_iter().any(
                |m| matches!(m, Inbound::Wire { from, bytes, .. } if from == a && bytes[0] == 0xA),
            );
            std::thread::sleep(Duration::from_millis(5));
            assert!(
                started.elapsed() < DEADLINE,
                "the new process never received A's reply; it went to the corpse"
            );
        }
        b1_events.extend(tb1.drain_inbound());
        let b1_replies: Vec<Inbound> = b1_events
            .iter()
            .filter(|m| matches!(m, Inbound::Wire { bytes, .. } if bytes[0] == 0xA))
            .cloned()
            .collect();
        assert_eq!(
            b1_replies,
            Vec::new(),
            "the replaced process must never receive the reply"
        );

        // (i) exactly ONE reincarnation notice, and it names the peer.
        a_events.extend(ta.drain_inbound());
        let resets: Vec<Inbound> = a_events
            .iter()
            .filter(|m| matches!(m, Inbound::PeerReset { .. }))
            .cloned()
            .collect();
        assert_eq!(
            resets,
            vec![Inbound::PeerReset {
                node: b,
                cause: PeerResetCause::Reincarnated,
            }],
            "a peer restarts ONCE — one notice, whatever the class count"
        );
        // (iv) the old lane retired as superseded, counted by name.
        let started = Instant::now();
        while ctl_a.stats().learned_lanes_superseded == 0 {
            std::thread::sleep(Duration::from_millis(5));
            assert!(
                started.elapsed() < DEADLINE,
                "A's old learned lane never retired as superseded"
            );
        }
        assert_eq!(ctl_a.stats().learned_lanes_superseded, 1);
    }

    /// The other side of the replacement rule: an EQUAL incarnation is the SAME process, so it never takes
    /// the reply path away from the connection already held. Without this the old process re-dialing after
    /// any blip would steal the lane back from itself and the supersede count would drift.
    #[test]
    fn a_dial_in_at_an_equal_incarnation_keeps_the_first_connection() {
        let rt = runtime();
        let trust = ClusterTrust::generate("vd-mesh-test").expect("trust");
        let a = NodeId(1);
        let b = NodeId(2);
        let [addr_a, addr_b1, addr_b2] = reserve_addrs();
        let (mut ta, ctl_a) = spawn_mesh(
            rt.handle(),
            &trust,
            &MeshConfig::new(a, addr_a, BTreeMap::new(), 64, 0, 0),
            None,
        )
        .expect("spawn A");
        let book: BTreeMap<NodeId, SocketAddr> = [(a, addr_a)].into_iter().collect();
        let (mut tb1, _ctl_b1) = spawn_mesh(
            rt.handle(),
            &trust,
            &MeshConfig::new(b, addr_b1, book.clone(), 64, 0, 0),
            None,
        )
        .expect("spawn B (first connection)");
        // The SAME node id at the SAME incarnation, on a second endpoint.
        let (mut tb2, _ctl_b2) = spawn_mesh(
            rt.handle(),
            &trust,
            &MeshConfig::new(b, addr_b2, book, 64, 0, 0),
            None,
        )
        .expect("spawn B (second connection, same incarnation)");

        tb1.send(a, MsgClass::Control, vec![0xB1].into())
            .expect("B1->A");
        wait_for(&mut ta, |g| {
            g.iter().any(
                |m| matches!(m, Inbound::Wire { from, bytes, .. } if *from == b && bytes[0] == 0xB1),
            )
        });
        // A DIFFERENT class, so the receiver's per-class dedup does not swallow this frame — its arrival
        // is what proves the second connection reached the learn site at all.
        tb2.send(a, MsgClass::Input, vec![0xB2].into())
            .expect("B2->A");
        wait_for(&mut ta, |g| {
            g.iter().any(
                |m| matches!(m, Inbound::Wire { from, bytes, .. } if *from == b && bytes[0] == 0xB2),
            )
        });
        // The learn runs just after the frame is delivered; settle so the decision has certainly been made.
        std::thread::sleep(Duration::from_millis(100));
        assert_eq!(
            ctl_a.stats().learned_peers_superseded,
            0,
            "an equal incarnation is the same process — it supersedes nothing"
        );

        ta.send(b, MsgClass::Control, vec![0xA].into())
            .expect("A->B learned");
        wait_for(&mut tb1, |g| {
            g.iter().any(
                |m| matches!(m, Inbound::Wire { from, bytes, .. } if *from == a && bytes[0] == 0xA),
            )
        });
        let resets: Vec<Inbound> = ta
            .drain_inbound()
            .into_iter()
            .filter(|m| matches!(m, Inbound::PeerReset { .. }))
            .collect();
        assert_eq!(
            resets,
            Vec::new(),
            "no process was replaced, so nothing was reset"
        );
    }

    /// The pure peer-level restart decision (HR5(a): every arm equality-asserted off the sockets).
    #[test]
    fn peer_restarted_answers_first_contact_restart_and_straggler() {
        let mut seen = None;
        assert!(
            !peer_restarted(&mut seen, 4),
            "a first contact is not a restart"
        );
        assert_eq!(seen, Some(4));
        assert!(!peer_restarted(&mut seen, 4), "the same process is quiet");
        assert_eq!(seen, Some(4));
        assert!(
            !peer_restarted(&mut seen, 3),
            "a straggler from an older process is not news"
        );
        assert_eq!(seen, Some(4), "and it never lowers the high-water mark");
        assert!(
            peer_restarted(&mut seen, 5),
            "a higher incarnation restarts"
        );
        assert_eq!(seen, Some(5));
        assert!(!peer_restarted(&mut seen, 5), "and it says so only once");
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
        // A node never dials itself, whatever the topology says about its own address.
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
                            Err(SendError::UnknownPeer(_)) => {
                                panic!("a booked peer is never unknown")
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

    /// R-4e3 (item 3): a CORRELATED multi-peer outage. 8 nodes all-pairs; kill HALF (4)
    /// SIMULTANEOUSLY; keep driving the 4 survivors. Each survivor's sends to the 4 DEAD peers bounce
    /// `NodeUnreachable` into its OWN inbox (backoff-paced, R-4a) — the scale risk is that bounce storm
    /// evicting a SURVIVING peer's reliable inbound from the shared node-wide inbox. Assert (exact,
    /// completion-gated) survivor 1 receives every frame from survivors 2/3/4 loss-free, AND
    /// `inbound_dropped_reliable == 0`. A SIZED inbox (capacity 256 ⇒ 2048 slots ≫ 3·BURST + the
    /// geometric re-bounce cadence) makes the non-eviction STRUCTURAL — the R-4a bounce cadence can't
    /// out-produce the drain. This converts the "confirm-dead can't flood the inbox" argument into
    /// evidence (the post-R-4b audit proved slow≠unreachable are disjoint; this proves a MASS outage too).
    #[test]
    fn a_correlated_half_cluster_outage_never_evicts_surviving_peer_traffic() {
        const N: u64 = 8;
        const SURVIVORS: usize = 4; // ids 1..=4 survive; 5..=8 die together
        const BURST: usize = 100; // < 256 ⇒ a u8 round is a unique per-frame id within a lane
        let rt = runtime();
        let trust = ClusterTrust::generate("vd-mesh-outage").expect("trust");
        let (mut nodes, controls) = cluster(rt.handle(), &trust, N, 256);
        let ids: Vec<NodeId> = nodes.iter().map(MeshTransport::local_id).collect();

        // Warm up all-pairs so every lane exists before the outage (sentinel payload 255, excluded below).
        for sender in &mut nodes {
            let from = sender.local_id();
            for &to in &ids {
                if to != from {
                    let _ = sender.send(to, MsgClass::Saga, vec![255u8].into());
                }
            }
        }
        // Kill the second half SIMULTANEOUSLY (the correlated outage).
        for c in &controls[SURVIVORS..] {
            c.kill();
        }
        std::thread::sleep(Duration::from_millis(50)); // let the CONNECTION_CLOSE propagate

        // The NO-LOSS traffic: each survivor sends BURST reliable to the OTHER survivors (this is what
        // must arrive loss-free). A FEW sends to each DEAD node then trigger the bounce stressor without
        // flooding a dead lane's retry buffer with QueueFull retry-spin (which would only slow the test,
        // not change what it proves).
        let survivor_ids = ids[..SURVIVORS].to_vec();
        let dead_ids = ids[SURVIVORS..].to_vec();
        for round in 0..BURST {
            for node in nodes[..SURVIVORS].iter_mut() {
                let from = node.local_id();
                for &to in &survivor_ids {
                    if to == from {
                        continue;
                    }
                    let mut payload = vd_sim::io::bytes(vec![round as u8]);
                    loop {
                        match node.send(to, MsgClass::Saga, payload) {
                            Ok(_) => break,
                            Err(SendError::QueueFull(returned)) => {
                                payload = returned;
                                std::thread::sleep(Duration::from_micros(200));
                            }
                            Err(SendError::UnknownPeer(_)) => {
                                panic!("a booked peer is never unknown")
                            }
                        }
                    }
                }
            }
        }
        // A handful to each dead node from every survivor — enough to make each dead lane owe a
        // redelivery and (after `confirm_unreachable_after_retries` backoff-paced failed replays) bounce.
        for node in nodes[..SURVIVORS].iter_mut() {
            for &dead in &dead_ids {
                for _ in 0..4u8 {
                    let _ = node.send(dead, MsgClass::Saga, vec![254u8].into());
                }
            }
        }

        // Survivor 1: wait until it has (a) every BURST round from EACH of survivors 2/3/4 AND (b) the
        // dead-lane bounce stressor is ACTIVELY present (≥ one NodeUnreachable per dead lane — the bounces
        // lag the Wire because they need `confirm_unreachable_after_retries` backoff-paced failed replays,
        // so gating on BOTH is what makes the non-eviction assertion non-vacuous). Both are drained into
        // node 1's ONE inbox concurrently — the exact eviction-competition the test exists to disprove.
        let survivor_senders = [NodeId(2), NodeId(3), NodeId(4)];
        let dead_lanes = N as usize - SURVIVORS; // 4 killed peers → ≥4 bounces expected
        let got = wait_for(&mut nodes[0], |g| {
            let wire_ok = survivor_senders.iter().all(|s| {
                g.iter()
                    .filter(|m| {
                        matches!(m, Inbound::Wire { from, bytes, .. } if from == s && bytes[0] != 255)
                    })
                    .count()
                    >= BURST
            });
            let bounces = g
                .iter()
                .filter(|m| matches!(m, Inbound::NodeUnreachable { .. }))
                .count();
            wire_ok && bounces >= dead_lanes
        });
        for s in survivor_senders {
            let rounds: BTreeSet<u8> = got
                .iter()
                .filter_map(|m| match m {
                    Inbound::Wire { from, bytes, .. } if *from == s && bytes[0] != 255 => {
                        Some(bytes[0])
                    }
                    _ => None,
                })
                .collect();
            assert_eq!(
                rounds.len(),
                BURST,
                "survivor 1 lost/duped frames from surviving peer {s}"
            );
        }
        // THE eviction assertion: with the dead-lane bounce storm demonstrably present in node 1's inbox
        // (the gate required ≥ dead_lanes NodeUnreachable) alongside the 3·BURST live-peer frames, the
        // sized inbox dropped ZERO reliable — the backoff-paced re-bounce cadence cannot out-produce the
        // drain, so a correlated outage never evicts surviving-peer reliable inbound. (If this ever REDs,
        // the R-4e design's coalesce cure — a `last_bounced_unreachable` flag capping one bounce per
        // dead-episode — is the fix; the passing test proves it is NOT needed at the current cadence.)
        assert_eq!(
            controls[0].stats().inbound_dropped_reliable,
            0,
            "a correlated-outage bounce storm must not evict surviving-peer reliable inbound"
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
    /// `reliable_shed`, bounced `SendShed{Unframable}` — R-4d M3: a shed, NOT a peer-unreachability), so it
    /// can never poison the lane. The payload is EXACTLY `MAX_STREAM_FRAME_BYTES` — IN the off-by-the-header
    /// poison window (the framed envelope pushes it over the cap) that the prior raw-payload `> MAX` check
    /// let slip through to wedge the lane (audit `wf_93fc5909`). The follow-up normal frame proves the lane
    /// is NOT poisoned — it still delivers — and exercises the reconnect-resets-an-existing-lane-stream path.
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
        // The caller learns it did not deliver (bounced as a SendShed, NOT NodeUnreachable — the peer is
        // alive; R-4d M3), with the Unframable reason (a permanent oversize reject).
        let bounced = wait_for(&mut nodes[0], |g| {
            g.iter().any(|m| matches!(m, Inbound::SendShed { .. }))
        });
        assert!(
            bounced.iter().any(|m| matches!(
                m,
                Inbound::SendShed {
                    class: MsgClass::Saga,
                    reason: ShedReason::Unframable,
                    ..
                }
            )),
            "over-cap reliable bounces SendShed{{Unframable}} for its class"
        );
        // And it is NEVER mis-reported as a peer-unreachability (the false-confirm cure).
        assert!(
            !bounced
                .iter()
                .any(|m| matches!(m, Inbound::NodeUnreachable { .. })),
            "an oversize shed must not surface as NodeUnreachable"
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
            0,
        );
        cfg.reliability.retry_buffer_max_bytes = 0; // below one max frame ⇒ invalid
        // `match` (not `expect_err`): the Ok type `(MeshTransport, MeshControl)` is not `Debug`. The Ok
        // arm panics without formatting, so the validation must fail loud BEFORE any endpoint binds.
        match spawn_mesh(rt.handle(), &trust, &cfg, None) {
            Err(ProdIoError::Tuning(_)) => {}
            Err(other) => panic!("expected a Tuning error, got {other:?}"),
            Ok(_) => panic!("a zero tuning field must be rejected at boot"),
        }
    }

    /// R-6d3a F3 (post-impl review): a peer fan-out larger than the outbox writer channel is rejected LOUD at
    /// boot — but ONLY when an outbox is actually wired (the depth is irrelevant without a durable sink). This
    /// closes the silent panic-at-scale ceiling of the non-blocking submit (a `Full` would panic a live node).
    #[test]
    fn spawn_mesh_rejects_an_oversized_peer_fan_out_only_when_an_outbox_is_wired() {
        let rt = runtime();
        let trust = ClusterTrust::generate("vd-mesh-test").expect("trust");
        let addr: std::net::SocketAddr = "127.0.0.1:0".parse().expect("addr");
        // One more peer than the outbox channel can hold.
        let mut book = BTreeMap::new();
        for i in 1..=(crate::outbox::OUTBOX_WRITER_CHANNEL_DEPTH as u64 + 1) {
            book.insert(NodeId(i), addr);
        }
        let cfg = MeshConfig::new(NodeId(1), addr, book, 8, 0, 0);
        let sink: SharedOutbox = Arc::new(Mutex::new(
            Box::new(MockOutboxSink::default()) as Box<dyn OutboxSink + Send>
        ));
        // With an outbox wired ⇒ rejected LOUD before any endpoint binds (the F3 boot check).
        match spawn_mesh(rt.handle(), &trust, &cfg, Some(sink)) {
            Err(ProdIoError::Tuning(_)) => {}
            Err(other) => panic!("expected a Tuning error, got {other:?}"),
            Ok(_) => panic!("an oversized peer fan-out with an outbox must be rejected at boot"),
        }
    }

    // ---- R-6d2c: durable-outbox write-through / delete-through (the FSM half; sink=None in prod) ----

    /// A recording `OutboxSink` for the FSM tests: `retain`/`release` push to `Vec`s so a test asserts the
    /// EXACT keys (and, for retain, the framed value) the lane staged. `commit`/`scan_all`/`gc_below` are
    /// inert — the R-6d2c FSM tests never call them (the fsync barrier + boot replay are R-6d3). R-6d3a adds
    /// `submit_barrier` returning a MONOTONE non-zero seq (never aliased to genesis `0`) + an always-durable
    /// handle, so a gate-fired assertion is unambiguous without a real store/writer.
    #[derive(Default)]
    struct MockOutboxSink {
        retained: Vec<(OutboxKey, Vec<u8>)>,
        released: Vec<OutboxKey>,
        next_submit: u64,
    }
    impl OutboxSink for MockOutboxSink {
        fn retain(&mut self, key: &OutboxKey, framed: &[u8]) {
            self.retained.push((*key, framed.to_vec()));
        }
        fn release(&mut self, key: &OutboxKey) {
            self.released.push(*key);
        }
        fn commit(&mut self) {}
        fn scan_all(&self) -> Vec<(OutboxKey, Vec<u8>)> {
            Vec::new()
        }
        fn gc_below(&mut self, _incarnation: u64) {}
        fn gc_replayed(&mut self, _replayed_keys: &[OutboxKey]) {}
        fn submit_barrier(&mut self) -> Option<(u64, DurabilityHandle)> {
            // Mirror `NodeOutbox`: a barrier only when something was staged (a retain happened). The FSM
            // tests always retain before submitting, so return a monotone seq + an always-durable handle.
            self.next_submit += 1;
            Some((self.next_submit, DurabilityHandle::already_durable()))
        }
        fn durability(&self) -> DurabilityHandle {
            DurabilityHandle::already_durable()
        }
    }

    /// The EXACT framed value the write-through mirrors for `(from, class, incarnation, seq, bytes)`: the
    /// `encode_frame` output at the worst-case epoch `u32::MAX` (the L2 seam contract — R-6d3 replay strips
    /// the envelope, `decode_frame`s, then re-stamps epoch). Reproduces `assign_and_retain`'s `encoded` byte
    /// for byte, so T1 pins the stored VALUE, not just its key.
    fn expected_encoded(
        from: NodeId,
        class: MsgClass,
        incarnation: u64,
        seq: u64,
        bytes: &[u8],
    ) -> Vec<u8> {
        vd_wire::framing::encode_frame(&ReliableFrame {
            from,
            class,
            incarnation,
            epoch: u32::MAX,
            seq,
            bytes: bytes.to_vec(),
        })
        .expect("frame encodes")
    }

    /// A per-process-unique redb path for the T8 real-`NodeOutbox` glue test; the guard removes it on drop.
    fn temp_outbox_path(tag: &str) -> std::path::PathBuf {
        std::env::temp_dir().join(format!(
            "vd-mesh-outbox-{tag}-{}-{:p}.redb",
            std::process::id(),
            &tag
        ))
    }
    struct TempOutbox {
        path: std::path::PathBuf,
    }
    impl Drop for TempOutbox {
        fn drop(&mut self) {
            let _ = std::fs::remove_file(&self.path);
        }
    }

    #[test]
    fn r6d2c_t1_durable_with_sink_retains_exact_key_and_encoded_value() {
        let mut mock = MockOutboxSink::default();
        let mut lane = ReliableLaneSender::new(PEER, 7, BIG_CAP);
        assert_eq!(
            lane.assign_and_retain(FROM, CLASS, b"x", true, Some(&mut mock)),
            Ok(0)
        );
        // ONE row: keyed peer=DEST (PEER≠FROM), the lane's incarnation, the assigned seq, the per-send class,
        // AND the u32::MAX-epoch framed value (the replay contract). Covers `durable ∧ sink=Some`.
        assert_eq!(
            mock.retained,
            vec![(
                OutboxKey {
                    peer: PEER,
                    class: CLASS,
                    incarnation: 7,
                    seq: 0,
                },
                expected_encoded(FROM, CLASS, 7, 0, b"x"),
            )]
        );
        assert!(mock.released.is_empty());
    }

    #[test]
    fn r6d2c_t2_durable_false_with_sink_records_nothing() {
        let mut mock = MockOutboxSink::default();
        let mut lane = ReliableLaneSender::new(PEER, 7, BIG_CAP);
        // durable=false with a LIVE sink: the `&& durable` false-arm ⇒ no retain, `retained` flag stays false.
        assert_eq!(
            lane.assign_and_retain(FROM, CLASS, b"x", false, Some(&mut mock)),
            Ok(0)
        );
        assert!(mock.retained.is_empty());
    }

    #[test]
    fn r6d2c_t3_durable_true_no_sink_is_a_noop_matching_prod() {
        let mut lane = ReliableLaneSender::new(PEER, 7, BIG_CAP);
        // durable=true but sink=None (the prod path until R-6d3): the RAM assign happens; nothing is mirrored.
        assert_eq!(lane.assign_and_retain(FROM, CLASS, b"x", true, None), Ok(0));
        // seq still advances exactly as an ordinary send — `durable` is behaviourally inert without a sink.
        assert_eq!(lane.assign_and_retain(FROM, CLASS, b"y", true, None), Ok(1));
    }

    #[test]
    fn r6d2c_t4_on_ack_releases_only_the_retained_retired_keys() {
        let mut mock = MockOutboxSink::default();
        let mut lane = ReliableLaneSender::new(PEER, 7, BIG_CAP);
        // seq0 durable (mirrored), seq1 ephemeral (not) — same lane, same ack window.
        assert_eq!(
            lane.assign_and_retain(FROM, CLASS, b"a", true, Some(&mut mock)),
            Ok(0)
        );
        assert_eq!(
            lane.assign_and_retain(FROM, CLASS, b"b", false, Some(&mut mock)),
            Ok(1)
        );
        // The ack retires BOTH (base -> 2) ...
        assert_eq!(lane.on_ack(7, 0, 1, Some(&mut mock)), 2);
        // ... but ONLY the durable seq0 releases an outbox row — the strict-subset property.
        assert_eq!(
            mock.released,
            vec![OutboxKey {
                peer: PEER,
                class: CLASS,
                incarnation: 7,
                seq: 0,
            }]
        );
    }

    #[test]
    fn r6d2c_t5_on_ack_with_no_sink_releases_nothing() {
        let mut lane = ReliableLaneSender::new(PEER, 7, BIG_CAP);
        // A durable frame retained WITHOUT a sink (nothing mirrored) then acked WITHOUT a sink: retire, no panic.
        assert_eq!(lane.assign_and_retain(FROM, CLASS, b"a", true, None), Ok(0));
        assert_eq!(lane.on_ack(7, 0, 0, None), 1);
    }

    #[test]
    fn r6d2c_t6_a_stale_ack_releases_nothing_even_with_a_live_sink() {
        let mut mock = MockOutboxSink::default();
        let mut lane = ReliableLaneSender::new(PEER, 7, BIG_CAP);
        assert_eq!(
            lane.assign_and_retain(FROM, CLASS, b"a", true, Some(&mut mock)),
            Ok(0)
        );
        // Wrong incarnation ⇒ the guard returns 0 BEFORE the retire loop ⇒ no release despite the live sink.
        assert_eq!(lane.on_ack(999, 0, 0, Some(&mut mock)), 0);
        assert!(mock.released.is_empty());
    }

    #[test]
    fn r6d2c_t7_a_redial_epoch_bump_does_not_double_retain() {
        let mut mock = MockOutboxSink::default();
        let mut lane = ReliableLaneSender::new(PEER, 7, BIG_CAP);
        assert_eq!(
            lane.assign_and_retain(FROM, CLASS, b"a", true, Some(&mut mock)),
            Ok(0)
        );
        // A write-error redial bumps epoch + replays the retained frame for RE-SEND — it must NOT re-mirror
        // (replay_batch is `&self`; on_write_error has no sink). Pins the §b subset invariant across redials.
        lane.on_write_error();
        let _ = lane.replay_batch();
        assert_eq!(
            mock.retained.len(),
            1,
            "replay re-sends but never re-mirrors"
        );
    }

    #[test]
    fn r6d2c_t8_real_node_outbox_stores_a_replay_decodable_frame() {
        let path = temp_outbox_path("glue");
        let _g = TempOutbox { path: path.clone() };
        let mut ob = crate::outbox::NodeOutbox::open(
            &path,
            crate::store::StoreTuning::default(),
            test_stamp(),
        )
        .expect("open");
        let mut lane = ReliableLaneSender::new(PEER, 3, BIG_CAP);
        lane.assign_and_retain(FROM, MsgClass::Saga, b"payload", true, Some(&mut ob))
            .expect("assign");
        ob.commit();
        let scanned = ob.scan_all();
        assert_eq!(scanned.len(), 1);
        let (k, v) = &scanned[0];
        let expected_key = OutboxKey {
            peer: PEER,
            class: MsgClass::Saga,
            incarnation: 3,
            seq: 0,
        };
        assert_eq!(*k, expected_key);
        // `scan_all` ALREADY stripped the OUTBOX_FORMAT_VERSION envelope — `decode_frame` reads the value
        // DIRECTLY (a second strip would eat the frame's u32 length-prefix). The stored value is a wire frame,
        // byte-identical to `write_reliable_frame`'s output, so R-6d3 boot replay round-trips it.
        let (rf, _) = vd_wire::framing::decode_frame::<ReliableFrame>(v).expect("replay-decodes");
        assert_eq!(rf.bytes, b"payload");
        // F2 (review): pin the FULL stored value byte-for-byte through the REAL redb `encode_value`/`decode_value`
        // envelope round-trip — not just `rf.bytes`. R-6d3 boot-replay dedups on (from, incarnation, seq), so a
        // future envelope/version-byte bug that shifted those header bytes must fail HERE, not silently survive.
        assert_eq!(
            *v,
            expected_encoded(FROM, MsgClass::Saga, 3, 0, b"payload"),
            "the redb-stored value is the exact u32::MAX-epoch wire frame"
        );
        // F1 (review): the FULL lifecycle through REAL redb — an ack RELEASES the row, netting `scan_all` to
        // empty. This is precisely what the MockOutboxSink tests CANNOT prove: their `released` log is driven by
        // the `retained` BOOKKEEPING bit, so neutering the actual `s.retain()` store write leaves them green
        // (mutation-blind). Here the retain WROTE a real redb row and on_ack's release must DELETE that SAME row
        // — proving `outbox_key` is single-sourced identically across retain + release, and that neither half is
        // a no-op. seq0 acked at the lane's (incarnation=3, epoch=0) ⇒ base -> 1, one release fired.
        assert_eq!(lane.on_ack(3, 0, 0, Some(&mut ob)), 1);
        ob.commit();
        assert!(
            ob.scan_all().is_empty(),
            "the acked durable row is deleted from redb — retain+release net to empty"
        );
    }

    #[test]
    fn r6d2c_t10_on_ack_releases_every_retained_key_in_a_multi_durable_window() {
        let mut mock = MockOutboxSink::default();
        let mut lane = ReliableLaneSender::new(PEER, 7, BIG_CAP);
        // TWO durable frames retired by ONE ack — the ONLY test exercising the `sink.as_deref_mut()` re-borrow
        // across loop iterations (on_ack:585). A regression to a move-out API (`sink.take()`) would release only
        // seq0 and slip every other test (each of which retires ≤1 durable seq). Pins the multi-release contract.
        assert_eq!(
            lane.assign_and_retain(FROM, CLASS, b"a", true, Some(&mut mock)),
            Ok(0)
        );
        assert_eq!(
            lane.assign_and_retain(FROM, CLASS, b"b", true, Some(&mut mock)),
            Ok(1)
        );
        assert_eq!(lane.on_ack(7, 0, 1, Some(&mut mock)), 2);
        assert_eq!(
            mock.released,
            vec![
                OutboxKey {
                    peer: PEER,
                    class: CLASS,
                    incarnation: 7,
                    seq: 0,
                },
                OutboxKey {
                    peer: PEER,
                    class: CLASS,
                    incarnation: 7,
                    seq: 1,
                },
            ],
            "both durable seqs released, in ascending base order"
        );
    }

    /// R-6d3b-2a: the boot-replay END-TO-END over real QUIC — the headline proof that `replay_outbox`
    /// re-drives every retained prior-incarnation row, DELIVERS it, and GCs the prior window, all WITHOUT the
    /// deadlock the vetted §2 would have caused (the peer_writers re-mirror while replay holds no lock) and
    /// WITHOUT the premature-gc data loss (the count-anchored fence waits for all fresh rows durable before
    /// gc). Pre-seeds a real `NodeOutbox` with rows at incarnation 1, spawns A at incarnation 2 wired to it,
    /// replays, asserts B receives the payloads + the incarnation-1 window is swept. (A hang ⇒ the deadlock
    /// regressed; the `wait_for`/`DEADLINE` backstop fails loud.)
    #[test]
    fn replay_outbox_redrives_retained_rows_delivers_and_gcs_the_prior_incarnation() {
        let rt = runtime();
        let trust = ClusterTrust::generate("vd-mesh-test").expect("trust");
        // Reserve two loopback ports (bind + read + drop — quinn re-binds them).
        let mut book = BTreeMap::new();
        let mut reserved = Vec::new();
        for id in [NodeId(1), NodeId(2)] {
            let socket = std::net::UdpSocket::bind("127.0.0.1:0").expect("reserve port");
            let addr = socket.local_addr().expect("addr");
            book.insert(id, addr);
            reserved.push((id, addr, socket));
        }
        // Pre-seed a real NodeOutbox: 2 retained rows keyed to B (NodeId(2)) at the PRIOR incarnation 1,
        // simulating rows left durable by a crashed prior process.
        let path = temp_outbox_path("replay");
        let _g = TempOutbox { path: path.clone() };
        let mut ob = crate::outbox::NodeOutbox::open(
            &path,
            crate::store::StoreTuning::default(),
            test_stamp(),
        )
        .expect("open");
        for seq in 0..2u64 {
            let framed = expected_encoded(NodeId(1), MsgClass::Saga, 1, seq, &[50 + seq as u8]);
            ob.retain(
                &OutboxKey {
                    peer: NodeId(2),
                    class: MsgClass::Saga,
                    incarnation: 1,
                    seq,
                },
                &framed,
            );
        }
        // R-6d3b-2b: a THIRD row keyed to a peer NOT in the book (NodeId 9, roster-gone) — the MIXED F1 proof
        // that a re-driven row is swept while a QUARANTINED row is RETAINED (never blanket-gc'd by incarnation).
        ob.retain(
            &OutboxKey {
                peer: NodeId(9),
                class: MsgClass::Saga,
                incarnation: 1,
                seq: 0,
            },
            &expected_encoded(NodeId(1), MsgClass::Saga, 1, 0, &[99]),
        );
        ob.commit();
        assert_eq!(
            ob.scan_all().len(),
            3,
            "2 routable + 1 roster-gone prior-incarnation rows pre-seeded"
        );
        let shared: SharedOutbox = Arc::new(Mutex::new(Box::new(ob) as Box<dyn OutboxSink + Send>));

        // Spawn A at the FRESH incarnation 2 wired to the shared outbox; B plain.
        let mut a = None;
        let mut b = None;
        for (id, addr, socket) in reserved {
            drop(socket); // release the port for quinn
            let cfg = MeshConfig::new(id, addr, book.clone(), 64, 2, 0); // incarnation 2 == new_incarnation
            let outbox = (id == NodeId(1)).then(|| shared.clone());
            let (t, _c) = spawn_mesh(rt.handle(), &trust, &cfg, outbox).expect("mesh");
            if id == NodeId(1) {
                a = Some(t);
            } else {
                b = Some(t);
            }
        }
        let mut a = a.expect("A");
        let mut b = b.expect("B");

        // Boot replay: re-drive the 2 routable rows through A (fence on durability, gc ONLY the re-driven
        // keys); QUARANTINE the roster-gone NodeId-9 row (retained, counted).
        let counts = crate::outbox::replay_outbox(&shared, &mut a, &book).expect("replay ok");
        assert_eq!(
            counts,
            crate::outbox::ReplayCounts {
                replayed: 2,
                quarantined: 1
            },
            "2 re-driven + 1 quarantined"
        );

        // B receives BOTH replayed payloads (re-drive proof; the test COMPLETING is the no-deadlock proof).
        let got = wait_for(&mut b, |g| {
            g.iter()
                .filter(|m| {
                    matches!(
                        m,
                        Inbound::Wire {
                            class: MsgClass::Saga,
                            ..
                        }
                    )
                })
                .count()
                >= 2
        });
        let payloads: std::collections::BTreeSet<u8> = got
            .iter()
            .filter_map(|m| match m {
                Inbound::Wire {
                    class: MsgClass::Saga,
                    bytes,
                    ..
                } => Some(bytes[0]),
                _ => None,
            })
            .collect();
        assert!(
            payloads.contains(&50) && payloads.contains(&51),
            "both replayed payloads delivered, got {payloads:?}"
        );

        // MIXED F1 proof: the RE-DRIVEN peer-2 incarnation-1 rows are gc_replayed-swept (gc ran STRICTLY after
        // they were durable — finding D / HIGH-3), BUT the QUARANTINED roster-gone peer-9 incarnation-1 row is
        // RETAINED (never blanket-swept by incarnation — the F1 no-loss invariant).
        let remaining = shared
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .scan_all();
        assert!(
            !remaining
                .iter()
                .any(|(k, _)| k.peer == NodeId(2) && k.incarnation == 1),
            "the re-driven peer-2 prior-incarnation rows were gc_replayed-swept; remaining {remaining:?}"
        );
        assert!(
            remaining
                .iter()
                .any(|(k, _)| k.peer == NodeId(9) && k.incarnation == 1),
            "the QUARANTINED roster-gone peer-9 row is RETAINED (F1 no-loss); remaining {remaining:?}"
        );
        let _ = &mut a;
    }
}

#[cfg(test)]
mod dial_failure_is_never_silent {
    //! A REFUSAL NOBODY CAN READ IS A NETWORK OUTAGE (slice S3's fleet half).
    //!
    //! Two nodes counting positions in different units are refused by the transport, which reports
    //! `no_application_protocol` — no field, no value, no unit. The dial used to discard that reason
    //! twice and retry forever, so the whole event was an endless silent loop.
    use super::*;

    #[test]
    fn the_same_reason_alarms_once_and_a_new_reason_alarms_again() {
        // ONCE PER DISTINCT REASON. This is called on every retry, and a line per retry is not an
        // alarm — it is how a real alarm gets muted by the people who most need to see it.
        let stats = Arc::new(MeshStats::default());
        let dest = NodeId(7);
        let addr: SocketAddr = "127.0.0.1:9".parse().expect("a test address");

        let seen = |s: &Arc<MeshStats>| {
            s.dial_failure
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner)
                .get(&dest)
                .cloned()
        };

        report_dial_failure(&stats, dest, addr, "no application protocol");
        assert_eq!(seen(&stats).as_deref(), Some("no application protocol"));

        // The same reason again changes nothing — the alarm does not repeat.
        report_dial_failure(&stats, dest, addr, "no application protocol");
        assert_eq!(seen(&stats).as_deref(), Some("no application protocol"));

        // A DIFFERENT reason is a different fault and must be heard.
        report_dial_failure(&stats, dest, addr, "connection refused");
        assert_eq!(seen(&stats).as_deref(), Some("connection refused"));
    }

    #[test]
    fn a_peer_that_connects_re_arms_its_alarm() {
        // Without this, a peer that failed once and recovered would stay silent through its NEXT
        // failure — the alarm would be a one-shot latch rather than a report of the current state.
        let stats = Arc::new(MeshStats::default());
        let dest = NodeId(7);
        let addr: SocketAddr = "127.0.0.1:9".parse().expect("a test address");

        report_dial_failure(&stats, dest, addr, "no application protocol");
        // What the dial does on success:
        stats
            .dial_failure
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .remove(&dest);
        assert!(
            stats
                .dial_failure
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner)
                .get(&dest)
                .is_none(),
            "a connected peer forgets its last failure"
        );
    }

    #[test]
    fn peers_alarm_independently() {
        // A shared latch would let one noisy peer silence a second, genuinely different fault.
        let stats = Arc::new(MeshStats::default());
        let addr: SocketAddr = "127.0.0.1:9".parse().expect("a test address");
        report_dial_failure(&stats, NodeId(1), addr, "no application protocol");
        report_dial_failure(&stats, NodeId(2), addr, "no application protocol");
        let seen = stats
            .dial_failure
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        assert_eq!(seen.len(), 2, "each peer keeps its own reason");
    }
}
