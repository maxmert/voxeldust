//! The `ShardIo` seam: the ONLY way simulation code touches the outside world.
//!
//! Landed with SPIKE-0a (the `Transport` half); `Clock`, `Store`, `DetRng`, and
//! `Provisioner` land with task P0.5. Production implementations live in `vd-io-prod`
//! (quinn/tokio/redb); deterministic test implementations live in [`mem`].
//!
//! Design: `docs/design/test_harness.md` §1–§2. The contract corrections that came out
//! of the adversarial review are load-bearing:
//! - `Transport::send` is **enqueue-only**. It can report exactly one synchronous,
//!   testable failure — back-pressure (`SendError::QueueFull`). It does NOT and CANNOT
//!   report whether the peer received anything: QUIC send completion is asynchronous,
//!   so a synchronous success/failure claim would either lie or buffer unboundedly.
//! - Hard delivery failures surface LATER, in-band, as [`Inbound::NodeUnreachable`]
//!   drained on a subsequent tick. The saga's error handling is therefore reachable in
//!   two distinct, separately-tested ways (sync `QueueFull` + async `NodeUnreachable`)
//!   — the structural cure for the old fire-and-forget `try_send` (R9).

pub mod mem;

use serde::{Deserialize, Serialize};
use vd_core::{MsgId, NodeId};

/// Opaque wire payload. Always produced by a `vd-wire` encoder — never hand-rolled
/// bytes (HR1: the closed flow taxonomies are the only byte producers).
///
/// `Arc<[u8]>` (not `Vec<u8>`): a snapshot fanned out to S sessions, or a frame
/// cloned to N mesh peers, is a refcount bump — never an O(payload) copy. The
/// gateway holds ONE body and shares it across every subscriber (SCALE-1).
pub type Bytes = std::sync::Arc<[u8]>;

/// Build [`Bytes`] from any byte container (a `vd-wire` encoder's `Vec<u8>`).
#[must_use]
pub fn bytes(v: Vec<u8>) -> Bytes {
    Bytes::from(v)
}

/// Coarse message class. Used by the transport for prioritization (control/saga ahead
/// of bulk) and by the fault fabric to target faults at a class of traffic. The class
/// also DETERMINES the carrier reliability ([`MsgClass::reliability`]) — the design's
/// channel table is enforced by the type, not by per-call-site choice.
///
/// ⚠️ WIRE-FROZEN, APPEND-ONLY. This enum rides EVERY postcard frame (`ReliableFrame`/`DatagramFrame`) via
/// its implicit variant index AND the R-6d durable outbox key's stable class byte. NEVER reorder or remove a
/// variant, and never add explicit `#[repr]`/discriminants that diverge from declaration order — either would
/// silently shift the on-wire class identity of live traffic and every retained-on-disk outbox row. ADD new
/// variants at the END ONLY. Pinned by `msgclass_wire_discriminant_is_frozen_append_only` (a reorder fails the
/// build), mirroring the `class_to_byte` golden pin (io-prod `outbox.rs`) and the `intershard.rs` append-only
/// arms — the three encodings of `MsgClass` identity must never drift.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub enum MsgClass {
    /// Reliable control-plane traffic (session lifecycle, subscriptions).
    Control,
    /// Transfer-saga envelopes and acks.
    Saga,
    /// Per-tick world-state snapshots (latest-wins, UNRELIABLE — `connection_plane.md`
    /// channel table: a fresh frame must never head-of-line-block behind a stale one).
    Snapshot,
    /// Player input frames (latest-wins, UNRELIABLE).
    Input,
    /// Liveness / lease / membership traffic.
    Membership,
    /// Ghost LIFECYCLE (`InterShardFlow::Ghost(GhostFlow::Spawn|Despawn)`, `wire/intershard.rs`):
    /// the reliable+acked spawn/despawn of a cross-shard collider ghost. RELIABLE — a dropped
    /// Spawn would strand a never-spawned collider; a dropped Despawn would leak a ghost forever.
    /// (Producer/consumer land at 1d.5b.3b; the variant is UNROUTED until then.)
    GhostReliable,
    /// Ghost POSE FEED (`InterShardFlow::Ghost(GhostFlow::Delta)`): the 20Hz latest-wins kinematic
    /// pose stream that keeps a fed source-ghost a live collider. UNRELIABLE — a fresh ghost pose
    /// must never head-of-line-block behind a stale one (identical contract to `Snapshot`/`Input`).
    /// (Producer/consumer land at 1d.5b.3b; the variant is UNROUTED until then.)
    GhostDelta,
}

/// Carrier reliability: whether a class rides a reliable ordered stream or a
/// best-effort latest-wins datagram. THE binding mapping (`connection_plane.md` §1).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Reliability {
    /// Reliable ordered delivery (QUIC uni stream): never dropped, never reordered.
    Reliable,
    /// Best-effort latest-wins (QUIC datagram): may be dropped on loss or overflow,
    /// newest supersedes oldest. Dropping is correct by design, not a failure.
    Unreliable,
}

impl MsgClass {
    /// The carrier this class MUST ride. Snapshot/Input are unreliable latest-wins;
    /// everything else is reliable. (`connection_plane.md` channel table.)
    #[must_use]
    pub fn reliability(self) -> Reliability {
        match self {
            MsgClass::Snapshot | MsgClass::Input | MsgClass::GhostDelta => Reliability::Unreliable,
            MsgClass::Control | MsgClass::Saga | MsgClass::Membership | MsgClass::GhostReliable => {
                Reliability::Reliable
            }
        }
    }
}

/// The only SYNCHRONOUS transport failure: the bounded outbound queue is saturated.
/// The refused payload is RETURNED so the caller can requeue without cloning — a
/// refusal is back-pressure, never a loss (the R9 drop-on-full is unrepresentable).
///
/// Everything else (peer death, link partition, write failure) is asynchronous by
/// nature and surfaces as [`Inbound::NodeUnreachable`] on a later drain.
#[derive(Clone, Debug, PartialEq, Eq, thiserror::Error)]
pub enum SendError {
    #[error("outbound queue full (back-pressure); payload returned")]
    QueueFull(Bytes),
}

/// Everything a node can observe from the outside world on a tick.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Inbound {
    /// A delivered peer message.
    Wire {
        from: NodeId,
        class: MsgClass,
        bytes: Bytes,
    },
    /// An asynchronous delivery failure: a previously-accepted `send` could not be
    /// delivered. `undelivered` is the transport-assigned FIFO sequence of that send,
    /// so callers that track their own send count can correlate exactly which message
    /// failed without the transport peeking into opaque payloads.
    NodeUnreachable {
        to: NodeId,
        class: MsgClass,
        undelivered: MsgId,
    },
    /// A LOCAL send was SHED by the transport — refused at THIS node's sender before it
    /// reached the wire, so it says NOTHING about `to`'s liveness (the peer may be alive).
    /// DISTINCT from [`Inbound::NodeUnreachable`] precisely so a liveness tracker never
    /// counts a local shed as evidence a peer is dead (R-4d M3: the false-confirm cure — a
    /// shed routed to `record_unreachable` would trip a destructive re-home of a
    /// live-but-ack-stalled peer). Both causes come from the io-prod mesh's bounded retry
    /// buffer (R-4b). Reliable feedback, like `NodeUnreachable` — never dropped to make
    /// inbox room. `undelivered` obeys the same FIFO-correlation contract as
    /// `NodeUnreachable.undelivered`.
    SendShed {
        to: NodeId,
        class: MsgClass,
        undelivered: MsgId,
        /// Why it was shed — lets a consumer/metric tell a permanent oversize reject from
        /// transient dead-ack-path backpressure without peeking at opaque payloads.
        reason: ShedReason,
    },
}

/// Why the transport shed a local send (the [`Inbound::SendShed`] cause). A CLOSED taxonomy.
/// Kept in `sim::io` (the seam owns its taxonomy): the io-prod mesh maps its private
/// `AssignReject` onto it, so `AssignReject` never leaks across the crate boundary. Never
/// serialized (`Inbound` is a local runtime enum), so it belongs here, not in `vd-wire`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ShedReason {
    /// The framed frame exceeds the stream frame cap — a PERMANENT reject (a partitioner/
    /// budget defect). Retrying is futile; the send can NEVER be delivered as framed.
    Unframable,
    /// The per-lane retry buffer hit its byte ceiling — the ack drain stalled (a dead ack
    /// path; the peer MAY be alive). Producer backpressure, never a retained-frame drop.
    RetryBufferFull,
}

impl Inbound {
    /// The carrier reliability of this inbound event (a `NodeUnreachable`/`SendShed` notice
    /// is reliable control feedback — it must never be dropped to make room).
    #[must_use]
    pub fn reliability(&self) -> Reliability {
        match self {
            Inbound::Wire { class, .. } => class.reliability(),
            // An unreachability notice is a reliable signal regardless of the failed
            // message's class: losing it would strand the sender's error handling.
            Inbound::NodeUnreachable { .. } => Reliability::Reliable,
            // A shed notice is reliable feedback for the same reason (losing it strands the
            // sender's accounting), independent of `ShedReason`.
            Inbound::SendShed { .. } => Reliability::Reliable,
        }
    }
}

/// Why a bounded inbox dropped a message (counted, never silent).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum InboxDrop {
    /// An unreliable (Snapshot/Input) message was evicted to bound the queue —
    /// correct by design (latest-wins): a newer frame supersedes it.
    Unreliable,
    /// A RELIABLE message was dropped because the inbox was full of reliable traffic
    /// — genuine overload, an ALERT. The sender's at-least-once redelivery (fabric)
    /// or QUIC stream flow-control (mesh) is the recovery path.
    Reliable,
}

/// A bounded inbound queue with a reliability-aware overflow policy — the deterministic
/// analog of a real node falling behind (TRANSPORT-5). Shared by every in-process
/// transport so the fault model can reproduce inbound overflow.
///
/// Policy on a full push:
/// 1. evict the OLDEST unreliable entry to make room (latest-wins: newest snapshots
///    survive, reliable traffic is never displaced); else
/// 2. if the queue is entirely reliable, drop the INCOMING message — if it is
///    unreliable that is latest-wins; if it is reliable that is the loud overload case.
///
/// The queue therefore never exceeds `capacity`, and reliable messages are dropped
/// ONLY when the inbox is saturated with reliable traffic (genuine overload).
#[derive(Debug)]
pub struct BoundedInbox {
    queue: std::collections::VecDeque<Inbound>,
    capacity: usize,
    dropped_unreliable: u64,
    dropped_reliable: u64,
}

impl BoundedInbox {
    #[must_use]
    pub fn new(capacity: usize) -> BoundedInbox {
        BoundedInbox {
            queue: std::collections::VecDeque::new(),
            capacity: capacity.max(1),
            dropped_unreliable: 0,
            dropped_reliable: 0,
        }
    }

    /// Push one inbound event, applying the overflow policy. Returns the drop that
    /// happened (if any) so the caller can surface a metric.
    pub fn push(&mut self, event: Inbound) -> Option<InboxDrop> {
        if self.queue.len() < self.capacity {
            self.queue.push_back(event);
            return None;
        }
        // Full: try to evict the oldest unreliable entry to admit the newcomer.
        if let Some(idx) = self
            .queue
            .iter()
            .position(|e| e.reliability() == Reliability::Unreliable)
        {
            self.queue.remove(idx);
            self.queue.push_back(event);
            self.dropped_unreliable += 1;
            return Some(InboxDrop::Unreliable);
        }
        // Entirely reliable and full: the incoming message is dropped.
        let drop = match event.reliability() {
            Reliability::Unreliable => InboxDrop::Unreliable,
            Reliability::Reliable => InboxDrop::Reliable,
        };
        match drop {
            InboxDrop::Unreliable => self.dropped_unreliable += 1,
            InboxDrop::Reliable => self.dropped_reliable += 1,
        }
        Some(drop)
    }

    /// Drain everything queued (FIFO).
    pub fn drain(&mut self) -> Vec<Inbound> {
        self.queue.drain(..).collect()
    }

    /// Clear without draining (crash modeling: undrained inbound is wiped).
    pub fn clear(&mut self) -> usize {
        let n = self.queue.len();
        self.queue.clear();
        n
    }

    #[must_use]
    pub fn len(&self) -> usize {
        self.queue.len()
    }
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.queue.is_empty()
    }
    #[must_use]
    pub fn dropped_unreliable(&self) -> u64 {
        self.dropped_unreliable
    }
    #[must_use]
    pub fn dropped_reliable(&self) -> u64 {
        self.dropped_reliable
    }
}

/// The time half of `ShardIo`: simulation code NEVER reads wall clocks (clippy-
/// enforced); it sees time only through this trait. Production: an io-prod
/// implementation disciplined toward the orchestrator's analytic clock (monotonic,
/// never stepped backward). Test: [`mem::VirtualClock`], advanced explicitly by the
/// topology driver — 12,000 ticks run in milliseconds with zero real sleeping.
pub trait Clock {
    /// This node's committed local simulation tick (NOT globally synchronized;
    /// every cross-shard message carries the sender's `source_tick`).
    fn local_tick(&self) -> vd_core::TickId;
    /// This node's view of the analytic universe clock (synced, monotonic).
    fn universe_tick(&self) -> vd_core::UniverseTick;
    /// The persisted universe epoch this node is participating in.
    fn epoch(&self) -> vd_core::EpochId;
}

/// The message-I/O half of `ShardIo`.
///
/// Contract (enforced by the SPIKE-0a tests against BOTH implementations):
/// 1. `send` never blocks and never performs network I/O on the calling thread.
/// 2. `send` returns `Err(QueueFull)` exactly when the bounded outbound queue is
///    saturated; an accepted send is assigned the next FIFO [`MsgId`].
/// 3. Messages accepted toward one peer WITHIN a single `MsgClass` (a `(peer,class)` lane) are
///    delivered in FIFO order (or surface as `NodeUnreachable`, also in order). **ACROSS classes there
///    is NO ordering guarantee** — the io-prod mesh rides one QUIC stream per `(peer,class)` so a stalled
///    class never head-of-line-blocks another, so every reliable consumer MUST be order-INDEPENDENT
///    across classes (a cross-class ordering dependence must ride the SAME class). CAVEAT: the mem
///    `FaultFabric` still drains one node-wide FIFO across all classes, so a cross-class-order-dependent
///    consumer passes in-process yet reorders on the real mesh — R-5' pins this with a parity test
///    (audit `wf_d0a91a43` H1).
/// 4. `drain_inbound` returns everything delivered since the previous drain, in
///    delivery order, without blocking.
/// 5. A LOCAL send the transport refuses at THIS node's sender (never reached the wire)
///    surfaces as [`Inbound::SendShed`] — NOT [`Inbound::NodeUnreachable`]: a shed is
///    orthogonal to `to`'s liveness (R-4d M3). Async local-failure feedback, like
///    `NodeUnreachable`; a consumer that tracks peer liveness MUST route the two
///    differently (a shed to a metric, an unreachability to the liveness tracker).
///
/// ⚠️ DELIVERY SEMANTICS (R-3' UPDATE, DEFERRED.md D-6): the io-prod `MeshTransport` is now AT-LEAST-ONCE
/// across a connection blip FOR A `(peer,class)` LANE THAT KEEPS CARRYING TRAFFIC (per-lane seq +
/// retry-buffer replay + a receiver contiguity ledger). Two residuals keep the
/// saga-must-not-depend-on-transport-redelivery rule BINDING until R-4'/R-5': (a) an IDLE-after-blip lane's
/// un-acked tail is RETAINED but not re-driven without the R-4' retransmit timer; (b) the receiver dedup
/// ledger is RAM-only, so a receiver restart wipes it. So transport dedup is BEST-EFFORT WITHIN A RECEIVER
/// INCARNATION — durable cross-restart exactly-once is the APPLICATION's job (the redb-persisted
/// `(correlation_id, step_id)` `applied_steps`; the HR2 decode-to-Default ban). The harness `FaultFabric` is
/// unconditionally at-least-once (redelivery-until-acked, surviving a receiver crash) — the STRICTER oracle.
/// Saga forward progress must therefore NOT depend on this trait redelivering a message lost to a peer
/// restart or an idle-after-blip lane — `scan_deadlines` re-drives every saga phase that HAS orchestrator
/// egress. The D-37 re-home ADOPT
/// gained that egress in Slice 2d (a re-homed `Promoting` carries `rehome_target`, so its Timeout
/// re-drives `A::ReHomeAdopt → target` — not `Promote → ctx.dest`), leaving `BatchHandoff::AwaitAdopt`
/// (the dest adopts off the source's envelope) as the ONE remaining producer-less phase (DEFERRED.md
/// D-6 precondition 1), whose owed cure (the same re-solicit-egress pattern, or a sender-side durable
/// outbox) is owed before the redb backend / a real rolling deploy.
pub trait Transport {
    /// Enqueue an outbound message toward `to`.
    fn send(&mut self, to: NodeId, class: MsgClass, bytes: Bytes) -> Result<MsgId, SendError>;

    /// Drain everything delivered to this node since the last drain.
    fn drain_inbound(&mut self) -> Vec<Inbound>;

    /// This node's identity.
    fn local_id(&self) -> NodeId;
}

/// The persistence half of `ShardIo` (D-6): durable state that survives a process kill-9 — the
/// orchestrator's saga WAL + directory + clock ceiling. Simulation/node code NEVER touches a disk
/// directly (clippy-enforced); durability is seen ONLY through this trait. Production: an io-prod
/// redb-backed impl with an off-tick fsync thread (DEFERRED — the backend redb-vs-sled-vs-custom is a
/// joint investigation). Test: [`mem::MemStore`], a staged/committed two-tier map whose `committed`
/// survives a node rebuild (the harness retains the handle) while `staged` is dropped on a crash —
/// modeling the fsync window the kill-9 cells must crash ACROSS.
///
/// Contract (the durability barrier, asserted by the contract tests):
/// 1. `put`/`delete` STAGE a mutation; neither is durable until `commit`.
/// 2. `commit` makes every staged mutation durable ATOMICALLY (the group-commit fsync point). A crash
///    BEFORE `commit` loses the staged batch; a crash AFTER it keeps the whole batch.
/// 3. `scan(prefix)` returns every COMMITTED `(key, value)` whose key starts with `prefix`, in ascending
///    key order — the rehydrate primitive (one prefix tag per key family); staged writes are invisible.
/// 4. LAST-WRITE-WINS within one fsync window: of several staged `put`/`delete`s at the SAME key before a
///    `commit`, only the last takes effect (a `delete` then `put` nets to the put; a `put` then `delete`
///    nets to the delete). The directory RECONCILE (delete-all-then-put-current in one barrier) relies on
///    this: a still-present record nets to a put, a revoked one to a lone delete (the audit COMP-2 cure).
///
/// OBJECT-SAFE by construction (used as `&mut dyn Store`), so there is NO per-monomorphization region
/// gotcha (HR5); ALL postcard encode/decode lives at the monomorphic call sites, never in the trait body.
pub trait Store {
    /// Stage a durable write of `value` at `key` (overwrites any staged/committed value at the key).
    fn put(&mut self, key: &[u8], value: &Bytes);
    /// Stage a durable delete of `key` (idempotent; a no-op at `commit` if the key is absent).
    fn delete(&mut self, key: &[u8]);
    /// Every COMMITTED `(key, value)` whose key starts with `prefix`, ascending by key.
    fn scan(&self, prefix: &[u8]) -> Vec<(Vec<u8>, Bytes)>;
    /// THE durability barrier: make every staged `put`/`delete` durable atomically (group-commit).
    fn commit(&mut self);
}

#[cfg(test)]
mod tests {
    use super::*;

    fn wire(class: MsgClass) -> Inbound {
        Inbound::Wire {
            from: NodeId(1),
            class,
            bytes: vec![0].into(),
        }
    }

    #[test]
    fn msgclass_wire_discriminant_is_frozen_append_only() {
        // MsgClass is wire-frozen: it rides every postcard frame (ReliableFrame/DatagramFrame) via its
        // variant index. Pin each variant's ACTUAL postcard byte so a REORDER or REMOVAL fails the build
        // (postcard encodes a fieldless enum as its 0-based variant index, a single varint byte for 0..=6).
        // Keep in lockstep with io-prod `outbox.rs` class_to_byte (the durable outbox KEY encoding).
        let pc = |c: MsgClass| postcard::to_allocvec(&c).expect("MsgClass encodes");
        assert_eq!(pc(MsgClass::Control), vec![0]);
        assert_eq!(pc(MsgClass::Saga), vec![1]);
        assert_eq!(pc(MsgClass::Snapshot), vec![2]);
        assert_eq!(pc(MsgClass::Input), vec![3]);
        assert_eq!(pc(MsgClass::Membership), vec![4]);
        assert_eq!(pc(MsgClass::GhostReliable), vec![5]);
        assert_eq!(pc(MsgClass::GhostDelta), vec![6]);
        // Round-trip closes the loop: the byte decodes back to the same variant.
        for (b, c) in [
            (0u8, MsgClass::Control),
            (1, MsgClass::Saga),
            (2, MsgClass::Snapshot),
            (3, MsgClass::Input),
            (4, MsgClass::Membership),
            (5, MsgClass::GhostReliable),
            (6, MsgClass::GhostDelta),
        ] {
            assert_eq!(
                postcard::from_bytes::<MsgClass>(&[b]).expect("decodes"),
                c,
                "byte {b} must decode to {c:?}"
            );
        }
    }

    #[test]
    fn reliability_maps_the_channel_table() {
        assert_eq!(MsgClass::Snapshot.reliability(), Reliability::Unreliable);
        assert_eq!(MsgClass::Input.reliability(), Reliability::Unreliable);
        assert_eq!(MsgClass::Control.reliability(), Reliability::Reliable);
        assert_eq!(MsgClass::Saga.reliability(), Reliability::Reliable);
        assert_eq!(MsgClass::Membership.reliability(), Reliability::Reliable);
        // 1d.5b.3a: ghost lifecycle is reliable (a lost Spawn/Despawn strands/leaks a collider);
        // the ghost pose feed is unreliable latest-wins (a fresh pose must never HOL-block).
        assert_eq!(MsgClass::GhostReliable.reliability(), Reliability::Reliable);
        assert_eq!(MsgClass::GhostDelta.reliability(), Reliability::Unreliable);
        // A NodeUnreachable notice is reliable feedback regardless of failed class.
        assert_eq!(
            Inbound::NodeUnreachable {
                to: NodeId(2),
                class: MsgClass::Input,
                undelivered: MsgId(0),
            }
            .reliability(),
            Reliability::Reliable
        );
        // A SendShed notice is reliable for BOTH shed reasons (constructs each variant,
        // covering the closed ShedReason enum + the new reliability() arm — R-4d M3).
        assert_eq!(
            Inbound::SendShed {
                to: NodeId(2),
                class: MsgClass::Saga,
                undelivered: MsgId(0),
                reason: ShedReason::Unframable,
            }
            .reliability(),
            Reliability::Reliable
        );
        assert_eq!(
            Inbound::SendShed {
                to: NodeId(2),
                class: MsgClass::Saga,
                undelivered: MsgId(0),
                reason: ShedReason::RetryBufferFull,
            }
            .reliability(),
            Reliability::Reliable
        );
        assert_eq!(
            wire(MsgClass::Snapshot).reliability(),
            Reliability::Unreliable
        );
    }

    #[test]
    fn bounded_inbox_admits_up_to_capacity_in_fifo_order() {
        let mut inbox = BoundedInbox::new(3);
        assert!(inbox.is_empty());
        for class in [MsgClass::Control, MsgClass::Saga, MsgClass::Membership] {
            assert_eq!(inbox.push(wire(class)), None);
        }
        assert_eq!(inbox.len(), 3);
        let drained = inbox.drain();
        assert_eq!(drained.len(), 3);
        assert!(inbox.is_empty());
        assert_eq!(drained[0], wire(MsgClass::Control));
    }

    #[test]
    fn full_inbox_evicts_oldest_unreliable_first() {
        // [snapshot, control, input] full; a new control evicts the OLDEST unreliable
        // (the snapshot), keeping reliable traffic and the newest unreliable.
        let mut inbox = BoundedInbox::new(3);
        inbox.push(wire(MsgClass::Snapshot));
        inbox.push(wire(MsgClass::Control));
        inbox.push(wire(MsgClass::Input));
        assert_eq!(
            inbox.push(wire(MsgClass::Saga)),
            Some(InboxDrop::Unreliable)
        );
        assert_eq!(inbox.dropped_unreliable(), 1);
        let drained = inbox.drain();
        // The first snapshot is gone; order is [control, input, saga].
        assert_eq!(
            drained,
            vec![
                wire(MsgClass::Control),
                wire(MsgClass::Input),
                wire(MsgClass::Saga),
            ]
        );
    }

    #[test]
    fn all_reliable_full_drops_the_newcomer_by_its_class() {
        let mut inbox = BoundedInbox::new(2);
        inbox.push(wire(MsgClass::Control));
        inbox.push(wire(MsgClass::Saga));
        // An incoming UNRELIABLE message can't displace reliable traffic: dropped
        // as latest-wins.
        assert_eq!(
            inbox.push(wire(MsgClass::Snapshot)),
            Some(InboxDrop::Unreliable)
        );
        // An incoming RELIABLE message into an all-reliable full inbox: the loud
        // overload case.
        assert_eq!(
            inbox.push(wire(MsgClass::Control)),
            Some(InboxDrop::Reliable)
        );
        assert_eq!(inbox.dropped_unreliable(), 1);
        assert_eq!(inbox.dropped_reliable(), 1);
        assert_eq!(inbox.len(), 2, "never exceeds capacity");
    }

    #[test]
    fn clear_wipes_without_draining() {
        let mut inbox = BoundedInbox::new(4);
        inbox.push(wire(MsgClass::Control));
        inbox.push(wire(MsgClass::Snapshot));
        assert_eq!(inbox.clear(), 2);
        assert!(inbox.is_empty());
        // Capacity floors at 1 even if constructed with 0.
        let mut tiny = BoundedInbox::new(0);
        assert_eq!(tiny.push(wire(MsgClass::Control)), None);
        assert_eq!(
            tiny.push(wire(MsgClass::Control)),
            Some(InboxDrop::Reliable)
        );
    }
}
