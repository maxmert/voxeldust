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

use std::collections::BTreeSet;

use bevy_ecs::prelude::{Schedule, World};
use bevy_ecs::schedule::ExecutorKind;
use vd_core::{NodeId, TickId};
use vd_sim::capability::NodeKind;
use vd_sim::io::{Bytes, Inbound, MsgClass, Reliability, SendError, Transport};
// The per-tick runtime resources live in vd-sim (shared with feature systems and
// the connection plane); re-exported here so node-level callers keep one path.
pub use vd_sim::runtime::{ClockSample, InboundBox, NodeIdentity, OutboundBox, OutboundStagingCap};

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
    /// Outbound messages refused with `QueueFull` and CARRIED OVER for retry next
    /// tick (back-pressure; counted, never silent). Excludes any frames shed to the
    /// staging cap — see `staging_shed`.
    pub backpressured: usize,
    /// Outbound frames SHED because the carried-over staging exceeded
    /// [`OutboundStagingCap`] (the loud overload-ALERT drop; oldest-first). Nonzero
    /// only under sustained, multi-tick congestion toward an effectively-unreachable
    /// peer — never in healthy operation.
    pub staging_shed: usize,
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
    world.insert_resource(OutboundStagingCap::default());
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
        let (sent, backpressured, staging_shed) = flush_phase(&mut self.transport, &mut self.world);
        TickReport {
            tick: self.tick,
            drained,
            sent,
            backpressured,
            staging_shed,
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

/// Monomorphic flush: push system-emitted messages in order, PER-PEER resilient.
///
/// A `QueueFull` is per-peer — io-prod gives each peer an independent lane, so a
/// refusal toward one peer must never abandon frames bound for a healthy peer
/// (SCALE-F1: the old break-on-first-refusal coupled all destinations, negating
/// that lane isolation). Once a peer back-pressures this tick we requeue the rest
/// of ITS frames untried — preserving that peer's FIFO (a later frame can never
/// jump ahead of an earlier refused one) — and keep flushing every other peer.
///
/// The carried-over staging is bounded by [`OutboundStagingCap`] (SCALE-F4): past
/// the cap the OLDEST staged frames are shed with a loud counted drop, so sustained
/// congestion toward an unreachable peer can never grow the buffer without limit.
///
/// Returns `(sent, backpressured, staging_shed)`.
fn flush_phase(transport: &mut dyn Transport, world: &mut World) -> (usize, usize, usize) {
    let pending = std::mem::take(&mut world.resource_mut::<OutboundBox>().0);
    let cap = world.resource::<OutboundStagingCap>().0;
    let mut sent = 0usize;
    // Peers that back-pressured THIS tick; their remaining frames requeue untried.
    let mut blocked: BTreeSet<NodeId> = BTreeSet::new();
    let mut requeued: Vec<(NodeId, MsgClass, Bytes)> = Vec::new();
    for (to, class, bytes) in pending {
        if blocked.contains(&to) {
            requeued.push((to, class, bytes));
            continue;
        }
        match transport.send(to, class, bytes) {
            Ok(_) => sent += 1,
            Err(SendError::QueueFull(returned)) => {
                blocked.insert(to);
                requeued.push((to, class, returned));
            }
        }
    }
    let (staging_shed, _reliable_shed) = shed_over_cap(&mut requeued, cap);
    let backpressured = requeued.len();
    world.resource_mut::<OutboundBox>().0 = requeued;
    (sent, backpressured, staging_shed)
}

/// Bound the carried-over staging to `cap`, shedding OLDEST-first but UNRELIABLE-FIRST
/// (the ROB-2 reliability-aware discipline, matching `BoundedInbox`): latest-wins
/// Snapshot/Input frames are the disposable casualties; a RELIABLE Saga/Control/
/// Membership frame is sacrificed ONLY when shedding every droppable unreliable frame
/// still leaves the backlog over cap — and that reliable loss is its own distinct
/// `error!`-level ALERT, never quietly lumped with snapshots. Survivors keep FIFO order.
/// Returns `(total_shed, reliable_shed)`. Monomorphic (the generic flush stays a shim).
fn shed_over_cap(requeued: &mut Vec<(NodeId, MsgClass, Bytes)>, cap: usize) -> (usize, usize) {
    let over = requeued.len().saturating_sub(cap);
    if over == 0 {
        return (0, 0);
    }
    let unreliable_total = requeued
        .iter()
        .filter(|(_, class, _)| class.reliability() == Reliability::Unreliable)
        .count();
    let unreliable_shed = over.min(unreliable_total);
    let reliable_shed = over - unreliable_shed;
    // Drop the oldest `unreliable_shed` unreliable + oldest `reliable_shed` reliable
    // frames (front-to-back walk = oldest-first); retain keeps the rest in FIFO order.
    let mut ud = unreliable_shed;
    let mut rd = reliable_shed;
    requeued.retain(|(_, class, _)| match class.reliability() {
        Reliability::Unreliable if ud > 0 => {
            ud -= 1;
            false
        }
        Reliability::Reliable if rd > 0 => {
            rd -= 1;
            false
        }
        _ => true,
    });
    if unreliable_shed > 0 {
        tracing::warn!(
            shed = unreliable_shed,
            cap,
            "OutboundBox staging over cap: shed oldest UNRELIABLE frames (latest-wins \
             Snapshot/Input) toward sustained-congested peer(s)"
        );
    }
    if reliable_shed > 0 {
        tracing::error!(
            shed = reliable_shed,
            cap,
            "OutboundBox staging over cap: a RELIABLE frame was SHED (overload ALERT — the \
             unreliable backlog was exhausted; the peer is effectively unreachable and \
             lease/self-fence will reassign its authority; this is never silent)"
        );
    }
    (unreliable_shed + reliable_shed, reliable_shed)
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
            vec![
                (B, MsgClass::Input, vec![2].into()),
                (B, MsgClass::Input, vec![3].into())
            ]
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

    // ---- SCALE-F1 / SCALE-F4: per-peer-resilient, bounded flush ----------------
    //
    // The canonical `MemHub` has ONE shared outbound queue, so it cannot express
    // "peer X congested while peer Y still has room" — the exact condition the
    // head-of-line fix guards. `PerPeerLanes` models io-prod's INDEPENDENT per-peer
    // lanes (each peer its own bounded lane) so the harness can actually catch this
    // class of bug. Lanes live behind a shared handle so the test inspects what each
    // peer received after `step_tick` (the node owns the transport privately).

    use std::cell::RefCell;
    use std::collections::BTreeMap;
    use std::rc::Rc;
    use vd_core::MsgId;

    #[derive(Default)]
    struct LaneState {
        caps: BTreeMap<NodeId, usize>,
        delivered: BTreeMap<NodeId, Vec<Bytes>>,
        next: u64,
    }

    struct PerPeerLanes {
        local: NodeId,
        state: Rc<RefCell<LaneState>>,
    }

    impl PerPeerLanes {
        fn with_caps(local: NodeId, caps: &[(NodeId, usize)]) -> (Self, Rc<RefCell<LaneState>>) {
            let state = Rc::new(RefCell::new(LaneState {
                caps: caps.iter().copied().collect(),
                ..LaneState::default()
            }));
            (
                PerPeerLanes {
                    local,
                    state: state.clone(),
                },
                state,
            )
        }
    }

    impl Transport for PerPeerLanes {
        fn send(&mut self, to: NodeId, _class: MsgClass, bytes: Bytes) -> Result<MsgId, SendError> {
            let mut st = self.state.borrow_mut();
            let cap = st.caps.get(&to).copied().unwrap_or(0);
            let lane = st.delivered.entry(to).or_default();
            if lane.len() >= cap {
                return Err(SendError::QueueFull(bytes));
            }
            lane.push(bytes);
            let id = MsgId(st.next);
            st.next += 1;
            Ok(id)
        }
        fn drain_inbound(&mut self) -> Vec<Inbound> {
            Vec::new()
        }
        fn local_id(&self) -> NodeId {
            self.local
        }
    }

    /// What `to` actually received this flush, as plain byte vecs (readable asserts).
    fn lane_bytes(state: &Rc<RefCell<LaneState>>, to: NodeId) -> Vec<Vec<u8>> {
        state
            .borrow()
            .delivered
            .get(&to)
            .map(|lane| lane.iter().map(|b| b.to_vec()).collect())
            .unwrap_or_default()
    }

    #[test]
    fn flush_is_per_peer_resilient_a_congested_peer_never_starves_a_healthy_one() {
        const X: NodeId = NodeId(10); // cap 1 — congests on its second frame
        const Y: NodeId = NodeId(11); // cap 8 — always has room

        let (transport, state) = PerPeerLanes::with_caps(A, &[(X, 1), (Y, 8)]);
        let mut node = build_app(stub_cfg(A), transport);
        assert_eq!(
            node.node_id(),
            A,
            "the node reports its transport's local id"
        );
        {
            let mut outbox = node.world_mut().resource_mut::<OutboundBox>();
            // Interleave X,Y,X,Y,X. A break-on-first-QueueFull would strand the
            // SECOND Y frame (idx 3) behind X's refusal at idx 2 — even though Y has room.
            outbox.0.push((X, MsgClass::Input, vec![0].into()));
            outbox.0.push((Y, MsgClass::Input, vec![1].into()));
            outbox.0.push((X, MsgClass::Input, vec![2].into()));
            outbox.0.push((Y, MsgClass::Input, vec![3].into()));
            outbox.0.push((X, MsgClass::Input, vec![4].into()));
        }
        let report = node.step_tick();

        // The healthy peer is FULLY delivered (both frames, in order) while X back-pressures.
        assert_eq!(lane_bytes(&state, Y), vec![vec![1u8], vec![3u8]]);
        // X accepted exactly its 1-deep lane.
        assert_eq!(lane_bytes(&state, X), vec![vec![0u8]]);
        assert_eq!(report.sent, 3); // 1 to X + 2 to Y
        assert_eq!(report.backpressured, 2); // X's two un-accepted frames carry over
        assert_eq!(report.staging_shed, 0);
        // The congested peer's FIFO is preserved — a later frame never jumps an earlier
        // refused one, and no other peer's frame leaks into its requeue.
        assert_eq!(
            node.world_mut().resource::<OutboundBox>().0,
            vec![
                (X, MsgClass::Input, vec![2].into()),
                (X, MsgClass::Input, vec![4].into()),
            ]
        );
    }

    #[test]
    fn staging_over_cap_sheds_oldest_unreliable_with_a_counted_drop() {
        const X: NodeId = NodeId(10); // cap 0 — refuses everything; all frames stage

        let (transport, _state) = PerPeerLanes::with_caps(A, &[(X, 0)]);
        let mut node = build_app(stub_cfg(A), transport);
        // Tighten the staging cap to 2 so the test does not need thousands of frames.
        node.world_mut().insert_resource(OutboundStagingCap(2));
        {
            let mut outbox = node.world_mut().resource_mut::<OutboundBox>();
            // All UNRELIABLE (Input) — the disposable class; oldest-first shed.
            for n in 0..5u8 {
                outbox.0.push((X, MsgClass::Input, vec![n].into()));
            }
        }
        let report = node.step_tick();

        assert_eq!(report.sent, 0); // cap-0 lane accepts nothing
        assert_eq!(report.staging_shed, 3); // 5 staged, cap 2 ⇒ 3 oldest shed
        assert_eq!(report.backpressured, 2); // exactly the cap carries over
        // The OLDEST (0,1,2) are shed; the NEWEST (3,4) survive, in order.
        assert_eq!(
            node.world_mut().resource::<OutboundBox>().0,
            vec![
                (X, MsgClass::Input, vec![3].into()),
                (X, MsgClass::Input, vec![4].into()),
            ]
        );
    }

    #[test]
    fn staging_shed_sacrifices_unreliable_before_any_reliable_frame() {
        // SCALE-F4 / SHED-CLASS-BLIND: a RELIABLE frame must outlive a disposable one.
        // Stage Saga(reliable), Input, Input, Saga(reliable), Input under cap 2 ⇒ 3 to
        // shed; all three Inputs go and BOTH Saga frames survive in FIFO order.
        const X: NodeId = NodeId(10); // cap 0 — everything stages
        let (transport, _state) = PerPeerLanes::with_caps(A, &[(X, 0)]);
        let mut node = build_app(stub_cfg(A), transport);
        node.world_mut().insert_resource(OutboundStagingCap(2));
        {
            let mut outbox = node.world_mut().resource_mut::<OutboundBox>();
            outbox.0.push((X, MsgClass::Saga, vec![0].into()));
            outbox.0.push((X, MsgClass::Input, vec![1].into()));
            outbox.0.push((X, MsgClass::Input, vec![2].into()));
            outbox.0.push((X, MsgClass::Saga, vec![3].into()));
            outbox.0.push((X, MsgClass::Input, vec![4].into()));
        }
        let report = node.step_tick();

        assert_eq!(report.staging_shed, 3); // the three Inputs
        assert_eq!(
            node.world_mut().resource::<OutboundBox>().0,
            vec![
                (X, MsgClass::Saga, vec![0].into()),
                (X, MsgClass::Saga, vec![3].into()),
            ],
            "both reliable frames survive; only the disposable unreliable ones are shed"
        );
    }

    #[test]
    fn staging_shed_sacrifices_reliable_only_when_no_unreliable_remain() {
        // When the backlog is ALL reliable and still over cap, reliable frames ARE shed
        // (oldest-first) — the distinct error-level ALERT path. Stage 3 Saga frames under
        // cap 1 ⇒ 2 oldest reliable shed, the newest survives.
        const X: NodeId = NodeId(10); // cap 0 — everything stages
        let (transport, _state) = PerPeerLanes::with_caps(A, &[(X, 0)]);
        let mut node = build_app(stub_cfg(A), transport);
        node.world_mut().insert_resource(OutboundStagingCap(1));
        {
            let mut outbox = node.world_mut().resource_mut::<OutboundBox>();
            for n in 0..3u8 {
                outbox.0.push((X, MsgClass::Saga, vec![n].into()));
            }
        }
        let report = node.step_tick();

        assert_eq!(report.staging_shed, 2); // 3 reliable staged, cap 1 ⇒ 2 shed
        assert_eq!(
            node.world_mut().resource::<OutboundBox>().0,
            vec![(X, MsgClass::Saga, vec![2].into())],
            "the newest reliable frame survives; the two oldest are the reliable casualties"
        );
    }
}
