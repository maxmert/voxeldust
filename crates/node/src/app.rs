//! `build_app(NodeKind, cfg, io)` and THE deterministic unit of progress:
//! `ShardNode::step_tick` (`docs/design/test_harness.md` §0/§2).
//!
//! ONE library constructs every node kind (HR3). A node is a private
//! `bevy_ecs::World` + a single-threaded `Schedule` + the injected I/O — its only
//! public surface is `step_tick() -> TickReport`. There is NO async here and NO
//! pacing: production pacing lives behind the io-prod seam; the test topology calls
//! `step_tick` directly on a virtual clock.
//!
//! Per the user mandate, P0 registers ZERO gameplay systems: kinds differ only in
//! their (validated) capability configuration. Feature systems attach by capability
//! in later phases through the same `build_app` — never by matching on a kind.
//!
//! Tick shape (fixed): drain inbound → run schedule → flush outbound. The schedule
//! reads `InboundBox` and writes `OutboundBox`; the transport is never visible to
//! systems (sealed shards: the flow encoder + `Transport::send` are the only egress).
//!
//! COVERAGE NOTE: `ShardNode<T>` is generic; per the branchless-shim discipline all
//! branching lives in monomorphic helpers over `&mut dyn Transport`.

use bevy_ecs::prelude::{Schedule, World};
use bevy_ecs::schedule::ExecutorKind;
use vd_core::{NodeId, TickId};
use vd_sim::capability::NodeKind;
use vd_sim::io::{Bytes, Inbound, MsgClass, SendError, Transport};
// The per-tick runtime resources live in vd-sim (shared with feature systems and
// the connection plane); re-exported here so node-level callers keep one path.
pub use vd_sim::runtime::{ClockSample, InboundBox, NodeIdentity, OutboundBox};

/// Static node configuration (operational values arrive via config structs, never
/// inline literals).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct NodeConfig {
    pub node_id: NodeId,
    pub kind: NodeKind,
}

/// What one tick did — the per-tick observability record (R10: structured signals,
/// not log archaeology) and the harness's per-step observation hook.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct TickReport {
    pub tick: TickId,
    /// Inbound messages drained at tick start.
    pub drained: usize,
    /// Outbound messages accepted by the transport.
    pub sent: usize,
    /// Outbound messages refused with `QueueFull` (back-pressure; the emitting
    /// system retries next tick — counted, never silent).
    pub backpressured: usize,
    /// `NodeUnreachable` notices observed among the drained inbound.
    pub unreachable: usize,
}

/// A node: private world, single-threaded schedule, injected transport.
pub struct ShardNode<T: Transport> {
    world: World,
    schedule: Schedule,
    transport: T,
    tick: TickId,
}

/// Construct a node. Kinds carry VALIDATED capability profiles (`ShardProfile::build`
/// already failed loud on incoherence); P0 attaches no systems for any kind.
pub fn build_app<T: Transport>(cfg: NodeConfig, transport: T) -> ShardNode<T> {
    let mut world = World::new();
    world.insert_resource(InboundBox::default());
    world.insert_resource(OutboundBox::default());
    world.insert_resource(ClockSample::default());
    world.insert_resource(NodeIdentity {
        node_id: cfg.node_id,
        kind: cfg.kind,
    });

    let mut schedule = Schedule::default();
    schedule.set_executor_kind(ExecutorKind::SingleThreaded);

    ShardNode {
        world,
        schedule,
        transport,
        tick: TickId(0),
    }
}

impl<T: Transport> ShardNode<T> {
    /// THE deterministic unit of progress. Synchronous by construction.
    pub fn step_tick(&mut self) -> TickReport {
        self.tick = self.tick.next();
        set_local_tick(&mut self.world, self.tick);
        let (drained, unreachable) = drain_phase(&mut self.transport, &mut self.world);
        self.schedule.run(&mut self.world);
        let (sent, backpressured) = flush_phase(&mut self.transport, &mut self.world);
        TickReport {
            tick: self.tick,
            drained,
            sent,
            backpressured,
            unreachable,
        }
    }

    #[must_use]
    pub fn tick(&self) -> TickId {
        self.tick
    }

    #[must_use]
    pub fn node_id(&self) -> NodeId {
        self.transport.local_id()
    }

    /// World access for the harness's ground-truth oracle and for capability-based
    /// feature registration in later phases. Tests and monitors only — gameplay
    /// never reaches a foreign node's world (HR1; crate-graph enforced).
    pub fn world_mut(&mut self) -> &mut World {
        &mut self.world
    }

    /// Schedule access for capability-based system registration (P1+).
    pub fn schedule_mut(&mut self) -> &mut Schedule {
        &mut self.schedule
    }

    /// Split borrow for registration helpers that install resources AND systems.
    pub fn parts_mut(&mut self) -> (&mut World, &mut Schedule) {
        (&mut self.world, &mut self.schedule)
    }
}

/// Monomorphic per-tick clock refresh: systems read time only via `ClockSample`.
fn set_local_tick(world: &mut World, tick: TickId) {
    world.resource_mut::<ClockSample>().local_tick = tick;
}

/// Monomorphic drain: pull everything delivered since last tick into the world.
fn drain_phase(transport: &mut dyn Transport, world: &mut World) -> (usize, usize) {
    let inbound = transport.drain_inbound();
    let drained = inbound.len();
    let unreachable = inbound
        .iter()
        .filter(|m| matches!(m, Inbound::NodeUnreachable { .. }))
        .count();
    let mut inbox = world.resource_mut::<InboundBox>();
    inbox.0 = inbound;
    (drained, unreachable)
}

/// Monomorphic flush: push system-emitted messages in order; on the FIRST refusal
/// the refused payload (returned by the transport — a refusal is never a loss) and
/// everything after it are requeued untried, preserving FIFO toward every peer.
fn flush_phase(transport: &mut dyn Transport, world: &mut World) -> (usize, usize) {
    let pending = std::mem::take(&mut world.resource_mut::<OutboundBox>().0);
    let mut sent = 0usize;
    let mut requeued: Vec<(NodeId, MsgClass, Bytes)> = Vec::new();
    let mut iter = pending.into_iter();
    while let Some((to, class, bytes)) = iter.next() {
        match transport.send(to, class, bytes) {
            Ok(_) => sent += 1,
            Err(SendError::QueueFull(returned)) => {
                requeued.push((to, class, returned));
                requeued.extend(iter);
                break;
            }
        }
    }
    let backpressured = requeued.len();
    world.resource_mut::<OutboundBox>().0 = requeued;
    (sent, backpressured)
}

#[cfg(test)]
mod tests {
    use super::*;
    use bevy_ecs::prelude::{Res, ResMut};
    use vd_sim::capability::profiles;
    use vd_sim::io::mem::MemHub;

    const A: NodeId = NodeId(1);
    const B: NodeId = NodeId(2);

    fn stub_cfg(id: NodeId) -> NodeConfig {
        NodeConfig {
            node_id: id,
            kind: NodeKind::StubShard,
        }
    }

    /// An infrastructure echo system: forwards every Wire payload back to a fixed
    /// peer. NOT game logic — it exists to exercise the tick pipeline.
    fn echo_system(inbox: Res<InboundBox>, mut outbox: ResMut<OutboundBox>) {
        for msg in &inbox.0 {
            if let Inbound::Wire { from, class, bytes } = msg {
                outbox.0.push((*from, *class, bytes.clone()));
            }
        }
    }

    #[test]
    fn tick_pipeline_drains_runs_and_flushes() {
        let hub = MemHub::new();
        let mut a = build_app(stub_cfg(A), hub.register(A, 8));
        let mut b = build_app(stub_cfg(B), hub.register(B, 8));
        b.schedule_mut().add_systems(echo_system);

        // A sends one message by writing the outbox directly (no systems on A).
        a.world_mut()
            .resource_mut::<OutboundBox>()
            .0
            .push((B, MsgClass::Control, vec![42].into()));
        let report_a = a.step_tick();
        assert_eq!(report_a.tick, TickId(1));
        assert_eq!((report_a.sent, report_a.backpressured), (1, 0));

        hub.pump();
        let report_b = b.step_tick();
        assert_eq!((report_b.drained, report_b.sent), (1, 1), "echo replied");

        hub.pump();
        let report_a = a.step_tick();
        assert_eq!(report_a.drained, 1, "echo arrived back");
        assert_eq!(report_a.unreachable, 0);
        assert_eq!(a.tick(), TickId(2));
        assert_eq!(a.node_id(), A);
    }

    #[test]
    fn backpressure_requeues_in_fifo_order_and_recovers() {
        let hub = MemHub::new();
        // Capacity 2: the third message backpressures; 4th stays untried behind it.
        let mut a = build_app(stub_cfg(A), hub.register(A, 2));
        let _b = hub.register(B, 8);

        {
            let mut outbox = a.world_mut().resource_mut::<OutboundBox>();
            for n in 0..4u8 {
                outbox.0.push((B, MsgClass::Input, vec![n].into()));
            }
        }
        let report = a.step_tick();
        assert_eq!((report.sent, report.backpressured), (2, 2));
        // The refused payloads are intact and ordered (returned, not lost).
        assert_eq!(
            a.world_mut().resource::<OutboundBox>().0,
            vec![(B, MsgClass::Input, vec![2].into()), (B, MsgClass::Input, vec![3].into())]
        );

        // Drain the hub; the next tick sends the remainder in order.
        hub.pump();
        let report = a.step_tick();
        assert_eq!((report.sent, report.backpressured), (2, 0));
    }

    #[test]
    fn unreachable_notices_are_counted() {
        let hub = MemHub::new();
        let mut a = build_app(stub_cfg(A), hub.register(A, 8));
        // The echo system also drains the NodeUnreachable notice (non-Wire path).
        a.schedule_mut().add_systems(echo_system);
        let _b = hub.register(B, 8);
        hub.kill(B);
        a.world_mut()
            .resource_mut::<OutboundBox>()
            .0
            .push((B, MsgClass::Saga, vec![1].into()));
        let r1 = a.step_tick();
        assert_eq!(r1.sent, 1, "enqueue succeeded; failure is async");
        hub.pump();
        let r2 = a.step_tick();
        assert_eq!((r2.drained, r2.unreachable), (1, 1));
    }

    #[test]
    fn identity_resource_reflects_config() {
        let hub = MemHub::new();
        let kind = NodeKind::Shard(profiles::ship().expect("ship"));
        let mut node = build_app(NodeConfig { node_id: A, kind }, hub.register(A, 8));
        let identity = *node.world_mut().resource::<NodeIdentity>();
        assert_eq!(identity, NodeIdentity { node_id: A, kind });
    }
}
