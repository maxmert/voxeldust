//! The per-tick runtime resources every node's systems read and write — the pure
//! half of the drain → run → flush tick shape (`vd-node` owns the phases; systems
//! see only these resources, never the transport — sealed shards).
//!
//! Lives in `vd-sim` so feature systems (sim) and gateway systems
//! (`vd-connection-plane`) share one vocabulary without depending on `vd-node`
//! (dependency law: bins → node → sim → wire → core).

use bevy_ecs::prelude::Resource;
use vd_core::{EpochId, NodeId, TickId, UniverseTick};

use crate::capability::NodeKind;
use crate::io::{Bytes, Inbound, MsgClass};

/// Messages drained from the transport this tick, readable by systems.
#[derive(Resource, Debug, Default)]
pub struct InboundBox(pub Vec<Inbound>);

/// Messages systems want sent; flushed to the transport at tick end. Refusals stay
/// queued for the next tick (bounded by the transport's own capacity discipline).
#[derive(Resource, Debug, Default)]
pub struct OutboundBox(pub Vec<(NodeId, MsgClass, Bytes)>);

/// This node's identity + kind, readable by systems.
#[derive(Resource, Clone, Copy, Debug, PartialEq, Eq)]
pub struct NodeIdentity {
    pub node_id: NodeId,
    pub kind: NodeKind,
}

/// This node's current view of time, updated each tick by the node shell (local
/// tick) and by clock-sync observation (analytic clock — the FollowerClock's
/// monotonic clamp lives in `vd-node`; systems only read the sample).
#[derive(Resource, Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct ClockSample {
    pub local_tick: TickId,
    pub universe_tick: UniverseTick,
    pub epoch: EpochId,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn boxes_default_empty() {
        assert_eq!(InboundBox::default().0, Vec::new());
        assert!(OutboundBox::default().0.is_empty());
        assert_eq!(ClockSample::default().local_tick, TickId(0));
    }

    #[test]
    fn identity_is_value_comparable() {
        let a = NodeIdentity {
            node_id: NodeId(1),
            kind: NodeKind::StubShard,
        };
        assert_eq!(a, a);
        let b = NodeIdentity {
            node_id: NodeId(2),
            ..a
        };
        assert_ne!(a, b);
    }
}
