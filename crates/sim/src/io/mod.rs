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
}

impl Inbound {
    /// The carrier reliability of this inbound event (a `NodeUnreachable` notice is
    /// reliable control feedback — it must never be dropped to make room).
    #[must_use]
    pub fn reliability(&self) -> Reliability {
        match self {
            Inbound::Wire { class, .. } => class.reliability(),
            // An unreachability notice is a reliable signal regardless of the failed
            // message's class: losing it would strand the sender's error handling.
            Inbound::NodeUnreachable { .. } => Reliability::Reliable,
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
/// 3. Messages accepted toward one peer are delivered in FIFO order (or surface as
///    `NodeUnreachable`, also in order).
/// 4. `drain_inbound` returns everything delivered since the previous drain, in
///    delivery order, without blocking.
pub trait Transport {
    /// Enqueue an outbound message toward `to`.
    fn send(&mut self, to: NodeId, class: MsgClass, bytes: Bytes) -> Result<MsgId, SendError>;

    /// Drain everything delivered to this node since the last drain.
    fn drain_inbound(&mut self) -> Vec<Inbound>;

    /// This node's identity.
    fn local_id(&self) -> NodeId;
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
