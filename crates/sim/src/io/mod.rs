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
pub type Bytes = Vec<u8>;

/// Coarse message class. Used by the transport for prioritization (control/saga ahead
/// of bulk) and by the fault fabric to target faults at a class of traffic.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub enum MsgClass {
    /// Reliable control-plane traffic (session lifecycle, subscriptions).
    Control,
    /// Transfer-saga envelopes and acks.
    Saga,
    /// Per-tick world-state snapshots (latest-wins).
    Snapshot,
    /// Player input frames (latest-wins).
    Input,
    /// Liveness / lease / membership traffic.
    Membership,
}

/// The only SYNCHRONOUS transport failure: the bounded outbound queue is saturated.
///
/// Everything else (peer death, link partition, write failure) is asynchronous by
/// nature and surfaces as [`Inbound::NodeUnreachable`] on a later drain.
#[derive(Clone, Copy, Debug, PartialEq, Eq, thiserror::Error)]
pub enum SendError {
    #[error("outbound queue full (back-pressure)")]
    QueueFull,
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
