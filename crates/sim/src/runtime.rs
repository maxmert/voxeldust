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

/// Messages systems want sent; flushed to the transport at tick end. A refusal
/// (`QueueFull`) is never a loss — the refused payload is returned and stays queued
/// for the next tick. The carried-over backlog is itself BOUNDED by
/// [`OutboundStagingCap`] (a peer that stays congested while systems keep producing
/// toward it would otherwise grow this `Vec` without limit — an OOM vector at load);
/// the node's flush sheds the oldest staged frames past the cap with a loud counted
/// drop. The transport's own per-peer capacity bounds what it *accepts*, not what
/// this staging buffer *holds* — those are distinct ceilings.
#[derive(Resource, Debug, Default)]
pub struct OutboundBox(pub Vec<(NodeId, MsgClass, Bytes)>);

/// Hard ceiling on the [`OutboundBox`] backlog a node carries across ticks under
/// sustained per-peer back-pressure. Beyond it the flush sheds OLDEST-first but
/// UNRELIABLE-first: latest-wins Snapshot/Input frames are the disposable casualties;
/// a RELIABLE Saga/Control/Membership frame is shed only when the unreliable backlog
/// is exhausted and the backlog is STILL over cap, and that reliable loss is a
/// distinct `error!`-level ALERT. This is the BoundedInbox ROB-2 reliability-aware
/// discipline applied outbound — a shed is a loud overload signal, never a silent
/// unbounded grow. A peer congested long enough to overflow this is genuinely
/// unreachable; the lease/self-fence machinery reassigns its authority.
///
/// Operational param with ONE home (this `Default`, never an inline literal at the
/// flush site — HR "no magic numbers"); it folds into `TransportTuning` when that
/// struct reaches the node tier (P1, alongside the existing `wire::framing` /
/// `io-prod` tuning notes). Tests/bins override by inserting a different value.
#[derive(Resource, Clone, Copy, Debug, PartialEq, Eq)]
pub struct OutboundStagingCap(pub usize);

impl Default for OutboundStagingCap {
    fn default() -> Self {
        // ~several ticks of full-peer-set fan-out: generous enough never to trip in
        // healthy operation (transient tick-skew drains within a tick or two), small
        // enough to bound worst-case staging memory at a few hundred KiB of frames.
        OutboundStagingCap(4096)
    }
}

impl OutboundBox {
    /// Encode one `InterShardFlow` and enqueue it to `to` on `class` — the ONE
    /// encode-and-push for every shard-bound flow (DRY-1: was hand-rolled at four sites —
    /// the saga runtime, the stub, and two orchestrator inlines). `class` is a PARAMETER,
    /// never hardcoded, so each caller states its message class deliberately (lease/CAS
    /// ride `Saga`; the clock broadcast rides `Membership`). The closed wire enums encode
    /// infallibly; the body is an `Arc`-backed `Bytes`.
    pub fn push_flow(
        &mut self,
        to: NodeId,
        class: MsgClass,
        flow: &vd_wire::intershard::InterShardFlow,
    ) {
        let bytes = crate::io::bytes(
            postcard::to_allocvec(flow).expect("closed wire enums serialize infallibly"),
        );
        self.0.push((to, class, bytes));
    }

    /// D-3 lease-liveness heartbeat: push one `LeaseRenew{key, fence}` per held key toward the
    /// orchestrator. A BRANCHLESS generic shim (HR5) — no `if`/`match`/`?`; ALL cadence + key-selection
    /// logic lives in the monomorphic callers (the gateway feeds its `Session` keys, the shard feeds its
    /// `Realm` + granted `Entity` keys — ONE mechanism, never a match-on-shard-kind, HR3). Renewals are
    /// idempotent-by-fence at `DirectoryCore::renew` and loss-tolerant (the next heartbeat covers a drop).
    pub fn push_renewals<I>(&mut self, keys: I, orchestrator: NodeId)
    where
        I: IntoIterator<Item = (vd_wire::seams::directory::DirectoryKey, vd_core::Fence)>,
    {
        for (key, fence) in keys {
            self.push_flow(
                orchestrator,
                MsgClass::Saga,
                &vd_wire::intershard::InterShardFlow::Directory(
                    vd_wire::seams::directory::DirectoryOp::LeaseRenew { key, fence },
                ),
            );
        }
    }
}

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
    fn staging_cap_default_is_the_one_documented_value() {
        // The cap lives ONLY here (no inline literal at the flush site); a positive,
        // generous ceiling that never trips in healthy operation but bounds the backlog.
        assert_eq!(OutboundStagingCap::default(), OutboundStagingCap(4096));
        assert!(OutboundStagingCap::default().0 > 0);
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
