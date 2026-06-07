//! Identity and correlation primitives.
//!
//! Minimal set landed with SPIKE-0a; the full family (`EntityId`, `SessionId`,
//! `AccountId`, `TransferId`, `Fence`, `EpochId`) lands with task P0.3.
//! Nothing here is ever derived from wall-clock time (R7).

use serde::{Deserialize, Serialize};

/// Identifies a process-level node (shard, gateway, orchestrator, scripted client).
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct NodeId(pub u64);

impl core::fmt::Display for NodeId {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "node-{}", self.0)
    }
}

/// A node-local simulation tick counter (monotonic; NOT globally synchronized —
/// every cross-shard message carries the sender's `source_tick`).
#[derive(
    Clone, Copy, Debug, Default, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize,
)]
pub struct TickId(pub u64);

impl TickId {
    /// The next tick. Plain saturating increment: a tick counter never wraps in practice
    /// (u64 at 20 Hz outlives the universe), but saturation keeps the type total.
    #[must_use]
    pub fn next(self) -> TickId {
        TickId(self.0.saturating_add(1))
    }
}

impl core::fmt::Display for TickId {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "tick-{}", self.0)
    }
}

/// Transport-assigned, per-sender monotonic message sequence number.
///
/// Used for at-least-once delivery bookkeeping and to correlate
/// `Inbound::NodeUnreachable { undelivered }` back to a send: sends on one
/// transport are FIFO, so the k-th accepted `send` carries `MsgId(k)`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct MsgId(pub u64);

impl core::fmt::Display for MsgId {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "msg-{}", self.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tick_next_increments() {
        assert_eq!(TickId(0).next(), TickId(1));
        assert_eq!(TickId(41).next(), TickId(42));
    }

    #[test]
    fn tick_next_saturates_at_max() {
        assert_eq!(TickId(u64::MAX).next(), TickId(u64::MAX));
    }

    #[test]
    fn display_formats() {
        assert_eq!(NodeId(7).to_string(), "node-7");
        assert_eq!(TickId(3).to_string(), "tick-3");
        assert_eq!(MsgId(9).to_string(), "msg-9");
    }

    #[test]
    fn ids_roundtrip_postcard() {
        let n = NodeId(123);
        let bytes = postcard::to_allocvec(&n).expect("encode");
        let back: NodeId = postcard::from_bytes(&bytes).expect("decode");
        assert_eq!(n, back);
    }

    #[test]
    fn ids_are_ordered() {
        assert!(NodeId(1) < NodeId(2));
        assert!(MsgId(1) < MsgId(2));
        assert!(TickId::default() < TickId(1));
    }
}
