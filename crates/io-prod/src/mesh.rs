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

use tokio::sync::{Notify, watch};
use tokio::task::JoinHandle;
use vd_core::{MsgId, NodeId};
use vd_sim::io::{
    BoundedInbox, Bytes, Inbound, InboxDrop, MsgClass, Reliability, SendError, ShedReason,
    Transport,
};

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

/// Registry of this node's DIALED (outbound) connections, keyed by dest peer (latest-wins on redial ⇒
/// bounded). [`MeshControl::drop_connections`] closes them all — the transient-blip lever (the endpoint
/// stays bound), DISTINCT from [`MeshControl::kill`] (which closes the endpoint). Closing connection C
/// tears down BOTH the DATA streams (this→peer) AND their reverse ACK stream on C, coherently.
type ConnRegistry = Arc<Mutex<BTreeMap<NodeId, quinn::Connection>>>;

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
    /// max stream frame size (un-framable ⇒ rejected, never retained, bounced `SendShed{Unframable}`
    /// — R-4d M3: NOT `NodeUnreachable`; a shed says nothing about peer liveness). R-4' adds the
    /// bounded-retry-buffer total shed (`SendShed{RetryBufferFull}`) on the same counter.
    pub reliable_shed: AtomicU64,
    /// SENDER: cumulative reliable frames RETIRED by an incoming cumulative ack (the retry buffer draining,
    /// R-3'). A value stuck at 0 while reliable traffic flows is a DEAD ACK PATH alert (the ack never
    /// reached this peer's writer) — the retry buffer would then grow unbounded until the R-4' shed.
    pub reliable_acked: AtomicU64,
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
}

/// THE inbound-push chokepoint: applies the BoundedInbox overflow policy AND surfaces
/// the result (audit ROB-2 — the drop return exists to be surfaced, never discarded).
/// A RELIABLE drop is the design's explicit overload ALERT: warned + counted. An
/// unreliable drop is by-design latest-wins back-pressure: counted only.
pub(crate) fn push_inbox(
    inbox: &SharedInbox,
    stats: &MeshStats,
    event: Inbound,
) -> Option<InboxDrop> {
    let dropped = inbox
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
        .push(event);
    match dropped {
        Some(InboxDrop::Reliable) => {
            stats
                .inbound_dropped_reliable
                .fetch_add(1, Ordering::Relaxed);
            tracing::warn!(
                "BoundedInbox FULL of reliable events: a RELIABLE inbound was dropped \
                 (genuine overload — raise the inbound capacity or shed load upstream)"
            );
        }
        Some(InboxDrop::Unreliable) => {
            stats
                .inbound_dropped_unreliable
                .fetch_add(1, Ordering::Relaxed);
        }
        None => {}
    }
    dropped
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
type RecvLedger = Arc<RwLock<BTreeMap<NodeId, Arc<Mutex<BTreeMap<MsgClass, RecvState>>>>>>;

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

/// One reliable send LANE — the per-(peer,class) at-least-once sender FSM (R-2b). Lazily created on
/// the first reliable frame for a class to a peer. ALL FSM logic (seq assign, retain, replay framing,
/// ack-retire) is SYNCHRONOUS + unit-testable WITHOUT tokio/quinn; only the stream open/write (in
/// `write_frame`) is async. One stream per class so a stalled class never head-of-line-blocks another.
/// R-4b: a retained frame + its FROZEN worst-case-epoch framed length. The on-wire varint for `epoch` grows
/// from 1 byte (epoch 0) to 5 bytes (>= 2^28) as `replay_batch` re-stamps on each redial, so the framed
/// length of a retained frame CHANGES over its life. `framed_len` is the length at `epoch=u32::MAX` (the same
/// `encode_frame` output `assign_and_retain` already produced for the oversize cap check), so it UPPER-BOUNDS
/// every future re-stamp — `retry_bytes` can never drift below the true on-wire size, and the cap check + the
/// accounting share ONE number (the H3 cure).
struct RetainedFrame {
    frame: ReliableFrame,
    framed_len: u32,
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
    fn new(incarnation: u64, retry_cap: usize) -> ReliableLaneSender {
        ReliableLaneSender {
            stream: None,
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
    fn on_ack(&mut self, ack_incarnation: u64, ack_epoch: u32, ack_through: u64) -> usize {
        if ack_incarnation != self.incarnation || ack_epoch != self.epoch {
            return 0;
        }
        let base_before = self.base;
        while self.base < self.next_seq && self.base <= ack_through {
            if let Some(rf) = self.retry.remove(&self.base) {
                // R-4b: keep `retry_bytes` in LOCKSTEP — subtract the SAME frozen framed_len assign added.
                self.retry_bytes -= rf.framed_len as usize;
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
        self.retry.insert(seq, RetainedFrame { frame, framed_len });
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
    /// This node's dialed (outbound) connections — the [`MeshControl::drop_connections`] blip lever.
    connections: ConnRegistry,
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
    // The node-wide receiver ledger (SURVIVES connection teardown — the cross-stream cure) and the dialed-
    // connection registry (the drop_connections blip lever), created ONCE and shared by every task.
    let ledger: RecvLedger = Arc::new(RwLock::new(BTreeMap::new()));
    let connections: ConnRegistry = Arc::new(Mutex::new(BTreeMap::new()));

    // Accept loop: a Semaphore caps concurrently-served connections so a connection
    // flood cannot task-flood the node (TRANSPORT-4). Each connection serves BOTH
    // reliable uni streams and unreliable datagrams.
    let accept_endpoint = endpoint.clone();
    let accept_inbox = Arc::clone(&inbox);
    let accept_stats = Arc::clone(&stats);
    let accept_ledger = Arc::clone(&ledger);
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
            tokio::spawn(async move {
                let _permit = permit; // held for the connection's lifetime
                let Ok(connection) = incoming.await else {
                    return; // handshake failed (foreign trust): drop, never serve
                };
                serve_connection(connection, inbox, stats, ledger, ack_flush).await;
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
            dest: peer,
            addr,
            rx,
            inbox: Arc::clone(&inbox),
            stats: Arc::clone(&stats),
            backoff_min: cfg.redial_backoff_min,
            backoff_max: cfg.redial_backoff_max,
            incarnation: cfg.process_incarnation,
            connections: Arc::clone(&connections),
            reliability: cfg.reliability,
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
        MeshControl {
            endpoint,
            stats,
            connections,
        },
    ))
}

/// Serve one accepted connection (the DATA-RECEIVER side): reliable uni streams (through the R-3'
/// contiguity verdict) + unreliable datagrams into the shared inbox, AND the reverse cumulative-ACK stream
/// back to the data sender ON THIS SAME connection (the directional-mesh topology — no AckRouter).
async fn serve_connection(
    connection: quinn::Connection,
    inbox: SharedInbox,
    stats: Arc<MeshStats>,
    ledger: RecvLedger,
    ack_flush: Duration,
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

    // DATA streams: one reader task per accepted uni stream.
    let stream_conn = connection.clone();
    let stream_inbox = Arc::clone(&inbox);
    let stream_stats = Arc::clone(&stats);
    let stream_ledger = Arc::clone(&ledger);
    let streams = tokio::spawn(async move {
        while let Ok(recv) = stream_conn.accept_uni().await {
            tokio::spawn(serve_data_stream(
                recv,
                Arc::clone(&stream_inbox),
                Arc::clone(&stream_stats),
                Arc::clone(&stream_ledger),
                Arc::clone(&acked_keys),
                Arc::clone(&ack_due),
            ));
        }
    });

    // Unreliable datagrams on the same connection (UNCHANGED hot path — byte-for-byte the R-1' split shape).
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
    // The connection closed (peer gone / drop_connections / kill): tear down its reader + ack tasks.
    streams.abort();
    ack_task.abort();
}

/// Read ONE accepted uni stream: consume the 1-byte `STREAM_KIND` tag FIRST (before the framing loop, never
/// inside `frame_payload`), then — for a DATA stream — run every frame through the contiguity verdict. An
/// ACK stream misrouted here (acks are consumed by `peer_writer`'s ack-reader, not `serve_connection`) or an
/// unknown tag or a torn stream is dropped WHOLESALE, never touching the ledger.
async fn serve_data_stream(
    mut recv: quinn::RecvStream,
    inbox: SharedInbox,
    stats: Arc<MeshStats>,
    ledger: RecvLedger,
    acked_keys: Arc<Mutex<BTreeSet<(NodeId, MsgClass)>>>,
    ack_due: Arc<Notify>,
) {
    let mut kind = [0u8; 1];
    if recv.read_exact(&mut kind).await.is_err() {
        return; // peer opened + finished, or a torn stream: clean silent close
    }
    if kind[0] != STREAM_KIND_DATA {
        return; // an ACK stream never rides an accepted conn; unknown tag ⇒ drop, ledger untouched
    }
    while let Some(frame) = read_one_reliable_frame(&mut recv).await {
        classify_and_deliver(&inbox, &stats, &ledger, &frame);
        // Every DATA frame (even a Dedup) makes this (peer,class) ack-relevant and re-acks it — a lost ack
        // is recovered because the sender replays until it sees the cumulative ack advance.
        acked_keys
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .insert((frame.from, frame.class));
        ack_due.notify_one();
    }
}

/// Classify ONE reliable frame under the ledger lock and deliver it (or count the drop). The ledger guard
/// is held across the SYNCHRONOUS `push_inbox` (never across an await): if the RELIABLE frame is dropped by
/// a full inbox (genuine overload), the whole `RecvState` is ROLLED BACK to its pre-classify snapshot —
/// advancing `hw` without delivering would Dedup every redelivery = permanent silent loss. Contiguity (the
/// Gap arm), not the epoch check, is the anti-burying guard, so this rollback cannot re-open the reverted-R-1
/// CRITICAL; a later redelivery re-drives the transition cleanly.
fn classify_and_deliver(
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
            Arc::clone(
                outer
                    .entry(frame.from)
                    .or_insert_with(|| Arc::new(Mutex::new(BTreeMap::new()))),
            )
        }
    };
    // The inner per-peer lock: serializes only same-peer frames, and is HELD across classify + push_inbox +
    // the rollback (the atomicity invariant — see the RecvLedger doc). The inner seed is byte-identical to
    // the pre-re-key node-wide entry (only the OUTER insert above is new).
    let mut classes = peer_lock
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let st = classes.entry(frame.class).or_insert(RecvState {
        incarnation: frame.incarnation,
        epoch: frame.epoch,
        hw: 0,
        primed: false,
    });
    let before = *st;
    match classify_reliable(st, frame.incarnation, frame.epoch, frame.seq) {
        Verdict::Accept | Verdict::Reset => {
            let dropped = push_inbox(
                inbox,
                stats,
                Inbound::Wire {
                    from: frame.from,
                    class: frame.class,
                    bytes: vd_sim::io::bytes(frame.bytes.clone()),
                },
            );
            if matches!(dropped, Some(InboxDrop::Reliable)) {
                // Not delivered ⇒ hw must NOT advance (push_inbox already counted+warned the reliable drop).
                *st = before;
            }
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
                    let classes = peer_lock
                        .lock()
                        .unwrap_or_else(std::sync::PoisonError::into_inner);
                    if let Some(st) = classes.get(&class) {
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
    /// The peer this writer dials (the ConnRegistry key; latest-wins on redial).
    dest: NodeId,
    addr: SocketAddr,
    rx: tokio::sync::mpsc::Receiver<OutFrame>,
    inbox: SharedInbox,
    stats: Arc<MeshStats>,
    backoff_min: Duration,
    backoff_max: Duration,
    /// This process's incarnation, stamped on every reliable frame this lane sends (R-2b).
    incarnation: u64,
    /// This node's dialed-connection registry — this writer inserts its connection on dial (for
    /// [`MeshControl::drop_connections`]).
    connections: ConnRegistry,
    /// The at-least-once redelivery tuning (R-4a reads `confirm_unreachable_after_retries`).
    reliability: MeshReliabilityTuning,
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
    // The reverse-ACK path (R-3'): a child ack-reader accept_uni's the peer's ACK stream on OUR dialed
    // connection and forwards each decoded AckFrame here via a latest-wins watch (a lost ack is superseded
    // by the next cumulative one — never a stale lower ack winning). Re-created per dial, aborted on drop.
    let (ack_tx, mut ack_rx) = watch::channel::<Option<AckFrame>>(None);
    let mut ack_reader: Option<JoinHandle<()>> = None;
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
                if changed.is_ok()
                    && let Some(ack) = ack_rx.borrow_and_update().clone()
                {
                    for e in &ack.entries {
                        if let Some(lane) = lanes.get_mut(&e.class) {
                            // R-6c/L4: retire against THIS entry's own class incarnation (per-entry), not a
                            // single frame-level scalar.
                            let retired = lane.on_ack(e.incarnation, e.epoch, e.ack_through);
                            if retired > 0 {
                                w.stats
                                    .reliable_acked
                                    .fetch_add(retired as u64, Ordering::Relaxed);
                            }
                        }
                    }
                }
                // changed Err ⇒ every ack-reader Sender dropped; peer_writer still holds `ack_tx`, so this
                // is effectively unreachable while the writer runs — keep draining sends regardless.
            }
            maybe = w.rx.recv() => {
                let Some(frame) = maybe else { break };
                let sent = write_frame(
                    &w.endpoint,
                    w.dest,
                    w.addr,
                    &mut connection,
                    &mut lanes,
                    &mut ack_reader,
                    &ack_tx,
                    &w.connections,
                    w.local,
                    w.incarnation,
                    w.reliability.retry_buffer_max_bytes,
                    &frame,
                    &w.stats,
                )
                .await;
                match sent {
                    Ok(()) => {
                        backoff = w.backoff_min;
                        // This lane's own write succeeded ⇒ reset ONLY its failure counter (per-lane, H2).
                        if let Some(l) = lanes.get_mut(&frame.class) {
                            l.on_replay_ok();
                        }
                    }
                    Err(WriteFail::Down) => {
                        // The connection/write died: tear it down + arm the retransmit timer. The bounce is
                        // NOT here — it is threshold-gated on the timer replay (a blip that recovers before
                        // `confirm_unreachable_after_retries` ⇒ ZERO bounce). The inline backoff `sleep`
                        // R-3' had is REMOVED (it blocked the whole select — acks/sends couldn't drain during
                        // backoff); the timer IS the non-blocking backoff clock now.
                        handle_connection_drop(&mut connection, &mut ack_reader, &mut lanes);
                        retransmit.as_mut().reset(tokio::time::Instant::now() + backoff);
                        counting = true;
                        backoff = (backoff * 2).min(w.backoff_max);
                    }
                    Err(WriteFail::Shed(reason)) => {
                        // The frame was REJECTED (oversize, or the retry buffer is full; nothing retained;
                        // the connection may be fine). Bounce once as `SendShed` (R-4d M3: NOT
                        // NodeUnreachable — a shed says nothing about peer liveness) so the caller learns
                        // it did not deliver. No timer arm, no connection drop. Both shed reasons arise
                        // ONLY on the reliable path (`assign_and_retain` is reliable-only), so the gate is
                        // load-bearing for neither today — the debug_assert makes that invariant executable
                        // (a future unreliable shed would trip it here, not silently mis-route).
                        debug_assert!(
                            frame.class.reliability() == Reliability::Reliable,
                            "a SendShed ({reason:?}) can only arise on a reliable lane"
                        );
                        if frame.class.reliability() == Reliability::Reliable {
                            push_inbox(
                                &w.inbox,
                                &w.stats,
                                Inbound::SendShed {
                                    to: frame.to,
                                    class: frame.class,
                                    undelivered: frame.msg_id,
                                    reason,
                                },
                            );
                        }
                    }
                }
            }
            () = &mut retransmit, if any_lane_owes(&lanes) => {
                // Idle-after-blip re-drive: dial-if-down, then replay EVERY owing lane's window — no new frame.
                let outcome = replay_lanes(
                    &w.endpoint,
                    w.dest,
                    w.addr,
                    &mut connection,
                    &mut lanes,
                    &mut ack_reader,
                    &ack_tx,
                    &w.connections,
                    w.reliability,
                    &w.inbox,
                    &w.stats,
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
    // Loop exit = the per-peer channel closed = clean shutdown; abort the ack-reader, then lane streams drop
    // here (quinn implicit-finish on the still-live connection, or no-op on an already-dead one).
    if let Some(h) = ack_reader.take() {
        h.abort();
    }
}

/// The child ack-reader for one dialed connection (R-3'): the peer's `serve_connection` opens ONE reverse
/// `STREAM_KIND_ACK` uni stream on THIS connection; accept it, consume the tag, and forward every decoded
/// `AckFrame` to `peer_writer` via the latest-wins watch. Ends when the connection closes (peer gone /
/// drop_connections) or `peer_writer` drops the watch.
async fn ack_reader_task(connection: quinn::Connection, ack_tx: watch::Sender<Option<AckFrame>>) {
    while let Ok(mut recv) = connection.accept_uni().await {
        let mut kind = [0u8; 1];
        if recv.read_exact(&mut kind).await.is_err() {
            continue; // a torn/finished stream: try the next
        }
        if kind[0] != STREAM_KIND_ACK {
            continue; // only ACK streams ride the dialed connection; ignore anything else
        }
        while let Some(ack) = read_one_ack_frame(&mut recv).await {
            if ack_tx.send(Some(ack)).is_err() {
                return; // peer_writer gone
            }
        }
    }
}

/// How a `write_frame` attempt failed (R-4a). `Down` = the connection/write died ⇒ drop the connection +
/// arm the retransmit timer; `Shed(reason)` = the frame was REJECTED (oversize, or the retry buffer is
/// full) ⇒ nothing retained, the connection may be fine, bounce once as `Inbound::SendShed{reason}` but do
/// NOT arm the timer or drop the connection. The `reason` (R-4d M3) rides through to the bounce so a
/// consumer can tell a permanent oversize reject from transient dead-ack-path backpressure.
enum WriteFail {
    Down,
    Shed(ShedReason),
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
    ack_reader: &mut Option<JoinHandle<()>>,
    lanes: &mut BTreeMap<MsgClass, ReliableLaneSender>,
) {
    *connection = None;
    if let Some(h) = ack_reader.take() {
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
        push_inbox(
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

/// R-4a: dial the peer on demand (the ONE dial home, shared by `write_frame` + `replay_lanes` so the
/// ack-reader teardown/respawn + the lane-stream reset can never drift). On a fresh dial: register the
/// connection (drop_connections), abort any prior ack-reader + spawn a new one on THIS connection, reset
/// every lane's stream (the old streams died with the old connection). `Ok` iff the connection is up
/// (already-up = a no-op); `Err` on a dial failure.
#[allow(clippy::too_many_arguments)] // writer state threaded explicitly
async fn ensure_connection(
    endpoint: &quinn::Endpoint,
    dest: NodeId,
    addr: SocketAddr,
    connection: &mut Option<quinn::Connection>,
    lanes: &mut BTreeMap<MsgClass, ReliableLaneSender>,
    ack_reader: &mut Option<JoinHandle<()>>,
    ack_tx: &watch::Sender<Option<AckFrame>>,
    connections: &ConnRegistry,
) -> Result<(), ()> {
    if connection.is_some() {
        return Ok(());
    }
    // SNI is pinned to the cluster-trust SAN; identity comes from mTLS, not DNS.
    let conn = endpoint
        .connect(addr, "localhost")
        .map_err(|_| ())?
        .await
        .map_err(|_| ())?;
    connections
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
        .insert(dest, conn.clone());
    if let Some(h) = ack_reader.take() {
        h.abort();
    }
    *ack_reader = Some(tokio::spawn(ack_reader_task(conn.clone(), ack_tx.clone())));
    *connection = Some(conn);
    for lane in lanes.values_mut() {
        lane.stream = None;
    }
    Ok(())
}

/// R-4a: (re)open ONE owing lane's uni stream and write its full `replay_batch()` (re-stamped at the current
/// epoch) — the redelivery of the unacked window, NO new assign. Mirrors `write_frame`'s reopen block minus
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
    addr: SocketAddr,
    connection: &mut Option<quinn::Connection>,
    lanes: &mut BTreeMap<MsgClass, ReliableLaneSender>,
    ack_reader: &mut Option<JoinHandle<()>>,
    ack_tx: &watch::Sender<Option<AckFrame>>,
    connections: &ConnRegistry,
    reliability: MeshReliabilityTuning,
    inbox: &SharedInbox,
    stats: &MeshStats,
) -> ReplayOutcome {
    if ensure_connection(
        endpoint,
        dest,
        addr,
        connection,
        lanes,
        ack_reader,
        ack_tx,
        connections,
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
        handle_connection_drop(connection, ack_reader, lanes);
    }
    if all_ok {
        ReplayOutcome::AllOk
    } else {
        ReplayOutcome::SomeFailed
    }
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
    dest: NodeId,
    addr: SocketAddr,
    connection: &mut Option<quinn::Connection>,
    lanes: &mut BTreeMap<MsgClass, ReliableLaneSender>,
    ack_reader: &mut Option<JoinHandle<()>>,
    ack_tx: &watch::Sender<Option<AckFrame>>,
    connections: &ConnRegistry,
    local: NodeId,
    incarnation: u64,
    retry_cap: usize,
    frame: &OutFrame,
    stats: &MeshStats,
) -> Result<(), WriteFail> {
    match frame.class.reliability() {
        Reliability::Reliable => {
            // R-6d2b: the reliable lane is the SOLE consumer of the per-send `Durability` marker — lower it
            // to a bool here. R-6d2c threads `durable` into `assign_and_retain`'s durable-outbox write-through
            // (retain-on-send) + `on_ack` delete-through; today it is computed but not yet consumed (the
            // `OutboxSink` handle is injected in R-6d2c).
            let _durable = matches!(frame.durability, vd_sim::io::Durability::Retained);
            // BUFFER-FIRST, BEFORE any connection work: create the lane, capture the id, assign + retain. A
            // first-dial failure to a dead peer then leaves the frame RETAINED (the retransmit timer re-drives
            // it) and the confirm bounce has an id — a failed dial NEVER silently drops the frame. A failed
            // write below NEVER rolls back the seq (no burned seq, no gap); the ONE frame object lives in
            // `lane.retry`. `assign_and_retain` also performs the OVERSIZE REJECT on the FRAMED size (the
            // envelope + codec byte can push a near-cap payload over `MAX_STREAM_FRAME_BYTES`): an un-framable
            // frame can NEVER be sent, so retaining it would poison the lane forever; it returns `Err` WITHOUT
            // retaining ⇒ shed-count + Shed (peer_writer bounces), nothing in `retry`.
            let seq = {
                let lane = lanes
                    .entry(frame.class)
                    .or_insert_with(|| ReliableLaneSender::new(incarnation, retry_cap));
                match lane.assign_and_retain(local, frame.class, &frame.bytes) {
                    Ok(seq) => {
                        // R-4a: remember the id ONLY of a RETAINED frame, so the threshold-gated confirm
                        // bounce (which has no OutFrame) can never carry the id of a rejected/never-retained
                        // frame — that frame already got its own immediate Shed bounce (post-impl review
                        // wf_b1d0610c).
                        lane.last_msg_id = Some(frame.msg_id);
                        seq
                    }
                    Err(reject) => {
                        // R-4b: BOTH rejects are shed-loud (counted + a bounce via the Shed arm), nothing
                        // retained ⇒ neither can poison the lane, neither arms the timer or drops the
                        // (healthy) connection. Unframable = a permanent oversize reject; BufferFull = producer
                        // backpressure (a full buffer means the ack path is dead — reliable_acked stuck).
                        stats.reliable_shed.fetch_add(1, Ordering::Relaxed);
                        // Map the mesh-private `AssignReject` onto the seam's closed `ShedReason` (R-4d
                        // M3) — `AssignReject` never crosses the crate boundary; the bounce carries the
                        // reason so a consumer distinguishes a permanent oversize reject from transient
                        // dead-ack backpressure.
                        let reason = match reject {
                            AssignReject::Unframable => {
                                tracing::warn!(
                                    "reliable frame (payload {}) would exceed MAX_STREAM_FRAME_BYTES \
                                     framed; rejected (counted, not retained)",
                                    frame.bytes.len()
                                );
                                ShedReason::Unframable
                            }
                            AssignReject::BufferFull => {
                                tracing::warn!(
                                    "reliable retry buffer FULL ({} bytes) — the ack drain is not keeping \
                                     up (check reliable_acked progress; a stuck-at-0 value = a dead ack \
                                     path); shedding a new {}-byte send (producer backpressure, NO \
                                     retained frame dropped)",
                                    lane.retry_bytes,
                                    frame.bytes.len()
                                );
                                ShedReason::RetryBufferFull
                            }
                        };
                        return Err(WriteFail::Shed(reason));
                    }
                }
            };

            // Dial on demand through the ONE dial home (shared with the retransmit path's replay_lanes). A
            // dial failure ⇒ Down; the frame is already retained above ⇒ the timer re-drives it.
            ensure_connection(
                endpoint,
                dest,
                addr,
                connection,
                lanes,
                ack_reader,
                ack_tx,
                connections,
            )
            .await
            .map_err(|()| WriteFail::Down)?;
            let conn = connection.as_ref().ok_or(WriteFail::Down)?;
            // Re-fetch the lane: a fresh dial in ensure_connection reset every lane's `stream` to None.
            let lane = lanes.get_mut(&frame.class).ok_or(WriteFail::Down)?;

            // STRUCTURAL if/else — NO fall-through (the CRITICAL double-write cure):
            if lane.stream.is_none() {
                // (re)opened stream: write EXACTLY replay_batch(), which by construction includes the
                // just-assigned frame as its HIGHEST entry ⇒ the new frame is written exactly once,
                // inside the batch. Never falls through to the steady-state write below.
                let mut send = conn.open_uni().await.map_err(|_| WriteFail::Down)?;
                // Tag the DATA stream ONCE, before any frame — consumed by serve_data_stream's read_exact(1)
                // BEFORE the framing loop, never inside frame_payload (read_one_reliable_frame stays byte-
                // identical, so the loopback bridge that shares it is unaffected).
                send.write_all(&[STREAM_KIND_DATA])
                    .await
                    .map_err(|_| WriteFail::Down)?;
                lane.stream = Some(send);
                let batch = lane.replay_batch();
                debug_assert_eq!(
                    batch.last().map(|f| f.seq),
                    Some(seq),
                    "buffer-first invariant: the just-assigned frame is the highest replay entry"
                );
                let send = lane.stream.as_mut().ok_or(WriteFail::Down)?;
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
                    .map_err(|_| WriteFail::Down)?;
                }
                Ok(())
            } else {
                // steady state: stream open ⇒ write ONLY the new frame (NEVER the backlog). Read the
                // single frame object back out of `retry`, copying its fields so the immutable borrow of
                // `retry` ends before the mutable borrow of `stream`.
                let nf = &lane.retry.get(&seq).ok_or(WriteFail::Down)?.frame;
                let (f_from, f_class, f_inc, f_epoch, f_bytes) = (
                    nf.from,
                    nf.class,
                    nf.incarnation,
                    nf.epoch,
                    nf.bytes.clone(),
                );
                let send = lane.stream.as_mut().ok_or(WriteFail::Down)?;
                write_reliable_frame(send, f_from, f_class, f_inc, f_epoch, seq, &f_bytes)
                    .await
                    .map_err(|_| WriteFail::Down)
            }
        }
        Reliability::Unreliable => {
            // Dial on demand (the ONE dial home). No retain, no bounce — datagram loss is latest-wins.
            ensure_connection(
                endpoint,
                dest,
                addr,
                connection,
                lanes,
                ack_reader,
                ack_tx,
                connections,
            )
            .await
            .map_err(|()| WriteFail::Down)?;
            let conn = connection.as_ref().ok_or(WriteFail::Down)?;
            // Datagrams are message-bounded (QUIC-delimited, NO stream framing — codec_flags is a
            // stream concept) and carry the BARE DatagramFrame (R1' hot/cold split — no reliability
            // metadata on the 20Hz path). TooLarge is a loud failure of the caller's framing.
            let payload = postcard::to_allocvec(&DatagramFrame {
                from: local,
                class: frame.class,
                bytes: frame.bytes.to_vec(),
            })
            .map_err(|_| WriteFail::Down)?;
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
                Err(quinn::SendDatagramError::ConnectionLost(_)) => Err(WriteFail::Down),
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
    fn send_durable(
        &mut self,
        to: NodeId,
        class: MsgClass,
        bytes: Bytes,
        durability: vd_sim::io::Durability,
    ) -> Result<MsgId, SendError> {
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
    /// A retry-buffer cap so large no FSM test ever trips `AssignReject::BufferFull` (R-4b is exercised by a
    /// dedicated small-cap test).
    const BIG_CAP: usize = usize::MAX;

    #[test]
    fn sender_assigns_monotone_seq_and_retains() {
        let mut lane = ReliableLaneSender::new(7, BIG_CAP);
        assert_eq!(lane.assign_and_retain(FROM, CLASS, b"a"), Ok(0));
        assert_eq!(lane.assign_and_retain(FROM, CLASS, b"bb"), Ok(1));
        assert_eq!(lane.assign_and_retain(FROM, CLASS, b"ccc"), Ok(2));
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
        let mut saga = ReliableLaneSender::new(5, BIG_CAP);
        let mut ghost = ReliableLaneSender::new(9, BIG_CAP);
        assert_eq!(saga.assign_and_retain(FROM, MsgClass::Saga, b"a"), Ok(0));
        assert_eq!(
            ghost.assign_and_retain(FROM, MsgClass::GhostReliable, b"b"),
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
                entries[0].ack_through
            ),
            1
        );
        assert_eq!(
            ghost.on_ack(
                entries[1].incarnation,
                entries[1].epoch,
                entries[1].ack_through
            ),
            1
        );
        assert!(
            saga.retry.is_empty() && ghost.retry.is_empty(),
            "both classes' windows retired against their own incarnation"
        );

        // The MISFIRE the per-class design removes: a WRONG-incarnation ack (what a shared last-class-wins
        // scalar would apply to the non-matching lane) retires NOTHING — a silent send stall.
        let mut stalled = ReliableLaneSender::new(9, BIG_CAP);
        assert_eq!(
            stalled.assign_and_retain(FROM, MsgClass::GhostReliable, b"c"),
            Ok(0)
        );
        assert_eq!(
            stalled.on_ack(5, 0, 0),
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
        let mut lane = ReliableLaneSender::new(0, BIG_CAP);
        assert_eq!(lane.assign_and_retain(FROM, CLASS, b"x"), Ok(0));
        lane.on_write_error();
        assert_eq!(lane.assign_and_retain(FROM, CLASS, b"y"), Ok(1)); // not reused 0, not skipped 2
        assert_eq!(
            lane.retry[&1].frame.epoch, 1,
            "the post-error assign carries the bumped epoch"
        );
        assert_eq!(lane.retry.keys().copied().collect::<Vec<_>>(), vec![0, 1]);
    }

    #[test]
    fn replay_batch_is_ascending_seq_with_current_epoch() {
        let mut lane = ReliableLaneSender::new(0, BIG_CAP);
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
        let mut lane = ReliableLaneSender::new(0, BIG_CAP);
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
        let mut lane = ReliableLaneSender::new(0, BIG_CAP);
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
        let mut lane = ReliableLaneSender::new(0, BIG_CAP);
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
        let mut lane = ReliableLaneSender::new(7, BIG_CAP);
        let at_cap = vec![0u8; vd_wire::framing::MAX_STREAM_FRAME_BYTES as usize];
        assert_eq!(
            lane.assign_and_retain(FROM, CLASS, &at_cap),
            Err(AssignReject::Unframable)
        );
        assert!(
            lane.retry.is_empty(),
            "an un-framable frame is NOT retained"
        );
        assert_eq!(lane.next_seq, 0, "a rejected assign does not burn a seq");
        // The lane is NOT poisoned: a normal frame after a rejection assigns cleanly at seq 0.
        assert_eq!(lane.assign_and_retain(FROM, CLASS, b"ok"), Ok(0));
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
        let mut lane = ReliableLaneSender::new(7, one * 2);
        assert_eq!(lane.assign_and_retain(FROM, CLASS, b"hello"), Ok(0));
        assert_eq!(lane.retry_bytes, one, "one frame accounted");
        assert_eq!(lane.assign_and_retain(FROM, CLASS, b"hello"), Ok(1));
        assert_eq!(lane.retry_bytes, one * 2, "two frames = full");
        assert_eq!(
            lane.assign_and_retain(FROM, CLASS, b"hello"),
            Err(AssignReject::BufferFull)
        );
        assert_eq!(lane.next_seq, 2, "a refused send does not burn a seq");
        assert_eq!(lane.retry.len(), 2, "nothing new retained");
        assert_eq!(
            lane.retry_bytes,
            one * 2,
            "retry_bytes unchanged — NO retained frame dropped (producer backpressure, not a shed)"
        );
        lane.on_ack(7, 0, 0); // retire seq0 -> a slot frees
        assert_eq!(lane.retry_bytes, one, "one frame freed");
        assert_eq!(
            lane.assign_and_retain(FROM, CLASS, b"hello"),
            Ok(2),
            "backpressure is transient: a freed slot accepts the next send"
        );
    }

    #[test]
    fn retry_bytes_stays_in_lockstep_with_retry_across_assign_ack_and_re_stamp() {
        // R-4b (the H3 accounting invariant): retry_bytes == the sum of every retained frame's FROZEN
        // framed_len — invariant across assigns, acks, AND an epoch re-stamp (framed_len is the worst-case
        // length, so it never drifts even as the on-wire epoch varint grows).
        let mut lane = ReliableLaneSender::new(7, BIG_CAP);
        for b in [b"a".as_slice(), b"bb", b"ccc", b"dddd"] {
            lane.assign_and_retain(FROM, CLASS, b).expect("fits");
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
        lane.on_ack(7, 1, 1); // retire seq0,1 at the bumped epoch
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
                ReliableLaneSender::new(u64::MAX, BIG_CAP)
                    .assign_and_retain(FROM, CLASS, &vec![0u8; n])
                    .is_ok()
            })
            .expect("some near-cap payload is accepted");
        let mut lane = ReliableLaneSender::new(u64::MAX, BIG_CAP);
        let seq = lane
            .assign_and_retain(FROM, CLASS, &vec![0u8; accepted])
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
        let mut a = ReliableLaneSender::new(5, BIG_CAP);
        a.assign_and_retain(FROM, MsgClass::Control, b"x")
            .expect("fits");
        a.on_write_error(); // A -> epoch 1
        let mut b = ReliableLaneSender::new(5, BIG_CAP);
        b.assign_and_retain(FROM, MsgClass::Saga, b"y")
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
        let mut lane = ReliableLaneSender::new(7, BIG_CAP);
        for b in [b"a".as_slice(), b"b", b"c", b"d"] {
            lane.assign_and_retain(FROM, CLASS, b).expect("fits");
        }
        assert_eq!(lane.on_ack(7, 0, 1), 2, "retires seq 0 and 1"); // epoch 0 == lane epoch
        assert_eq!(lane.base, 2);
        assert_eq!(lane.retry.keys().copied().collect::<Vec<_>>(), vec![2, 3]);
    }

    #[test]
    fn on_ack_ignores_a_stale_epoch_ack() {
        let mut lane = ReliableLaneSender::new(7, BIG_CAP);
        lane.assign_and_retain(FROM, CLASS, b"a").expect("fits");
        lane.on_write_error(); // epoch -> 1
        lane.assign_and_retain(FROM, CLASS, b"b").expect("fits");
        assert_eq!(
            lane.on_ack(7, 0, 5),
            0,
            "an ack at the OLD epoch 0 retires nothing"
        );
        assert_eq!(lane.base, 0, "a stale-epoch ack retires nothing");
        assert_eq!(lane.retry.len(), 2);
    }

    #[test]
    fn on_ack_ignores_a_prior_incarnation_ack() {
        let mut lane = ReliableLaneSender::new(7, BIG_CAP);
        lane.assign_and_retain(FROM, CLASS, b"a").expect("fits");
        assert_eq!(
            lane.on_ack(6, 0, 0),
            0,
            "an ack against a DIFFERENT incarnation retires nothing"
        );
        assert_eq!(lane.base, 0);
    }

    #[test]
    fn on_ack_is_monotone_a_lower_ack_never_rolls_base_back() {
        let mut lane = ReliableLaneSender::new(7, BIG_CAP);
        for b in [b"a".as_slice(), b"b", b"c", b"d", b"e", b"f"] {
            lane.assign_and_retain(FROM, CLASS, b).expect("fits");
        }
        assert_eq!(lane.on_ack(7, 0, 4), 5, "retire 0..=4"); // base -> 5
        assert_eq!(lane.base, 5);
        assert_eq!(
            lane.on_ack(7, 0, 1),
            0,
            "a reordered LOWER ack retires nothing"
        );
        assert_eq!(lane.base, 5, "base is monotone");
        assert_eq!(lane.retry.keys().copied().collect::<Vec<_>>(), vec![5]);
    }

    #[test]
    fn on_ack_clamps_to_next_seq_so_base_never_overruns() {
        let mut lane = ReliableLaneSender::new(7, BIG_CAP);
        lane.assign_and_retain(FROM, CLASS, b"a").expect("fits");
        lane.assign_and_retain(FROM, CLASS, b"b").expect("fits");
        assert_eq!(
            lane.on_ack(7, 0, 999),
            2,
            "a forged/torn ack_through is clamped to next_seq"
        );
        assert_eq!(lane.base, lane.next_seq, "base clamped to next_seq");
        assert_eq!(lane.base, 2);
        assert!(lane.retry.is_empty(), "everything assigned was acked");
    }

    #[test]
    fn replay_batch_starts_at_base_after_a_retire() {
        let mut lane = ReliableLaneSender::new(7, BIG_CAP);
        for b in [b"a".as_slice(), b"b", b"c"] {
            lane.assign_and_retain(FROM, CLASS, b).expect("fits");
        }
        lane.on_ack(7, 0, 0); // retire seq0, base -> 1
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
        let mut lane = ReliableLaneSender::new(1, BIG_CAP);
        for step in 0u64..200 {
            match step % 4 {
                0 => {
                    lane.assign_and_retain(FROM, CLASS, b"x").expect("fits");
                }
                1 => {
                    lane.on_ack(1, lane.epoch, step / 2); // an ack sometimes ahead of next_seq
                }
                2 => lane.on_write_error(),
                _ => {
                    lane.on_ack(1, lane.epoch.wrapping_sub(1), step); // a stale-epoch ack: ignored
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
        let mut lane = ReliableLaneSender::new(1, BIG_CAP);
        assert!(
            !lane.owes_redelivery(),
            "a fresh lane (empty window) owes nothing"
        );
        lane.assign_and_retain(FROM, CLASS, b"x").expect("fits");
        assert!(
            lane.owes_redelivery(),
            "closed stream + non-empty window ⇒ owes a redelivery"
        );
        lane.on_ack(1, 0, 0); // retire the only frame
        assert!(
            !lane.owes_redelivery(),
            "a drained window owes nothing (the timer stops)"
        );
    }

    #[test]
    fn lane_failure_counter_bumps_saturating_and_resets_per_lane() {
        let mut lane = ReliableLaneSender::new(1, BIG_CAP);
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
        let mut lane = ReliableLaneSender::new(1, BIG_CAP);
        lane.assign_and_retain(FROM, CLASS, b"x").expect("fits");
        lanes.insert(CLASS, lane);
        assert!(
            any_lane_owes(&lanes),
            "an owing lane ⇒ the timer guard is live"
        );
        lanes.get_mut(&CLASS).expect("lane").on_ack(1, 0, 0); // drain it
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
                // This dead-peer test drives no oversize/buffer-full sends, so no SendShed arises;
                // present for Inbound exhaustiveness (R-4d M3).
                Inbound::Wire { .. } | Inbound::SendShed { .. } => None,
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
