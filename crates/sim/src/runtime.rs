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
pub struct OutboundBox(pub Vec<(NodeId, MsgClass, Bytes, crate::io::Durability)>);

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
        // The DEFAULT: Ephemeral (every re-driven flow — the common case). A producer-less one-shot uses
        // [`push_flow_durable`](Self::push_flow_durable) with `Retained`; that exception is enforced by the
        // `durability_class` conformance test, so the 99% of pushes stay a clean 3-arg call (DRY).
        self.push_flow_durable(to, class, flow, crate::io::Durability::Ephemeral);
    }

    /// [`push_flow`](Self::push_flow) with an explicit [`Durability`](crate::io::Durability) — the producer-
    /// less one-shots (`TransientBatch`, `GhostFlow::Despawn`, per `FlowDurabilityClass`) pass `Retained` so
    /// the R-6d durable outbox mirrors + replays them across a source crash (D-6 #1).
    pub fn push_flow_durable(
        &mut self,
        to: NodeId,
        class: MsgClass,
        flow: &vd_wire::intershard::InterShardFlow,
        durability: crate::io::Durability,
    ) {
        // R-6d2b HARDENING (post-impl review): with `send`/`push_flow` defaulting `Ephemeral`, a producer-less
        // reliable flow (`FlowDurabilityClass::ProducerLessReliable`) pushed WITHOUT `Retained` is silently
        // lost on a source crash (D-6 #1). This fires at the push site on EVERY such push in EVERY debug/test
        // build — so a forgotten marker (incl. on a future P9-Signal / P6-BlockEdit producer-less flow) fails
        // LOUD immediately, not on manual test-authoring discipline. Debug-only (release hot path untouched);
        // it forces the CORRECT durability, a guard stronger than the vetted explicit-4-arg (which forced only
        // that SOME value be stated). The compile-time nested tripwires (`intershard_closed.rs`) are the twin.
        debug_assert!(
            !(flow.durability_class()
                == vd_wire::intershard::FlowDurabilityClass::ProducerLessReliable
                && matches!(durability, crate::io::Durability::Ephemeral)),
            "producer-less-reliable flow pushed Ephemeral (needs Retained — D-6 #1 silent-loss): {flow:?}"
        );
        // interplay-01 (2026-07-11 holistic audit): the SYMMETRIC half of the HR1 send seam. The durability guard
        // above forbids a producer-less-reliable flow riding `Ephemeral`; THIS forbids a SIDE-EFFECTING flow
        // (authority-gating, non-idempotent — Transfer/Saga/SagaAck/side-effecting Directory ops) riding an
        // UNRELIABLE carrier (Snapshot/Input/GhostDelta), where datagram loss would SILENTLY drop an
        // authority-gating message with a green suite. FireAndForget flows (ghost pose deltas) MAY ride Unreliable
        // — loss is correct there. Debug-only (release hot path untouched); it fires at EVERY offending push, so a
        // future P9-Signal / P6-BlockEdit / PvP-fire push on the wrong class fails LOUD the moment it is written —
        // the send-time twin of the compile-time `intershard_closed.rs` classifiers.
        debug_assert!(
            !(matches!(
                flow.effect_class(),
                vd_wire::intershard::EffectClass::SideEffecting { .. }
            ) && class.reliability() == crate::io::Reliability::Unreliable),
            "side-effecting flow pushed on an Unreliable carrier (needs a Reliable MsgClass — datagram loss \
             would silent-drop an authority-gating message): class={class:?} flow={flow:?}"
        );
        let bytes = crate::io::bytes(
            postcard::to_allocvec(flow).expect("closed wire enums serialize infallibly"),
        );
        self.0.push((to, class, bytes, durability));
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

    /// The R-6d2b hardening guard: the DEFAULT `push_flow` (Ephemeral) must REJECT a producer-less-reliable
    /// flow — proving a forgotten `Retained` fails loud at the push site (not on manual test authoring), and
    /// covering the `debug_assert` panic arm. `#[cfg(debug_assertions)]` because the guard is debug-only.
    #[cfg(debug_assertions)]
    #[test]
    #[should_panic(expected = "producer-less-reliable flow pushed Ephemeral")]
    fn push_flow_default_rejects_a_producer_less_reliable_flow() {
        use vd_wire::intershard::{GhostFlow, InterShardFlow};
        let mut ob = OutboundBox::default();
        // Ghost::Despawn is FlowDurabilityClass::ProducerLessReliable — pushing it via the Ephemeral default
        // (instead of `push_flow_durable(.., Retained)`) is the D-6 #1 silent-loss the guard forbids.
        ob.push_flow(
            NodeId(2),
            MsgClass::GhostReliable,
            &InterShardFlow::Ghost(GhostFlow::Despawn {
                entity: vd_core::EntityId::pack(vd_core::entity_kind::EntityKind::Ship, 1, 7, 1),
                source_fence: vd_core::Fence(1),
            }),
        );
    }

    /// interplay-01 (holistic audit): the SYMMETRIC send-seam guard — a SIDE-EFFECTING flow pushed on an
    /// UNRELIABLE carrier must fail loud (else a datagram loss silently drops an authority-gating message),
    /// covering the second `debug_assert` panic arm. Debug-only, like its durability sibling above.
    #[cfg(debug_assertions)]
    #[test]
    #[should_panic(expected = "side-effecting flow pushed on an Unreliable carrier")]
    fn push_flow_rejects_a_side_effecting_flow_on_an_unreliable_carrier() {
        use vd_wire::intershard::InterShardFlow;
        use vd_wire::seams::directory::{DirectoryKey, DirectoryOp};
        let mut ob = OutboundBox::default();
        // A `LeaseRevoke` is SideEffecting + ReDriven (so the durability guard above does NOT fire); routing it on
        // `MsgClass::Snapshot` (Unreliable) is exactly the silent-authority-drop this guard forbids.
        ob.push_flow(
            NodeId(2),
            MsgClass::Snapshot,
            &InterShardFlow::Directory(DirectoryOp::LeaseRevoke {
                key: DirectoryKey::Entity(vd_core::EntityId::pack(
                    vd_core::entity_kind::EntityKind::Player,
                    1,
                    2,
                    3,
                )),
                fence: vd_core::Fence(1),
            }),
        );
    }

    /// interplay-01, the FireAndForget side: a ghost flow (re-derivable, loss-tolerant) is NOT subject to the
    /// carrier-reliability guard — it stages cleanly on any carrier. Covers the guard's `matches!`-false branch
    /// (pushed Retained so the durability sibling above also holds, isolating THIS guard's arm).
    #[test]
    fn push_flow_stages_a_fire_and_forget_flow_unguarded() {
        use vd_wire::intershard::{GhostFlow, InterShardFlow};
        let mut ob = OutboundBox::default();
        ob.push_flow_durable(
            NodeId(2),
            MsgClass::GhostReliable,
            &InterShardFlow::Ghost(GhostFlow::Despawn {
                entity: vd_core::EntityId::pack(vd_core::entity_kind::EntityKind::Ship, 1, 7, 1),
                source_fence: vd_core::Fence(1),
            }),
            crate::io::Durability::Retained,
        );
        assert_eq!(
            ob.0.len(),
            1,
            "a FireAndForget flow stages regardless of carrier reliability"
        );
    }
}
