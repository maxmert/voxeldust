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

use std::collections::{BTreeMap, BTreeSet};

use bevy_ecs::prelude::{Resource, Schedule, World};
use bevy_ecs::schedule::ExecutorKind;
use vd_core::{NodeId, TickId};
use vd_sim::capability::NodeKind;
use vd_sim::io::{Bytes, Durability, Inbound, MsgClass, Reliability, SendError, Transport};
use vd_wire::intershard::InterShardFlow;
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
    /// Sends refused this tick because the destination has no lane and no address (the peer book's
    /// trigger); the frames wait and the orchestrator is asked.
    pub unknown_peers: usize,
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
    /// peer — never in healthy operation. This is the TOTAL shed (unreliable + reliable);
    /// the reliable subset is broken out as [`Self::reliable_shed`].
    pub staging_shed: usize,
    /// The RELIABLE subset of `staging_shed` — a catastrophic drop of a Saga/Control/
    /// Membership frame (a lost transfer command, a stranded post-commit `OpenInputSlot`,
    /// a dropped resume `SessionInput`), NOT a benign latest-wins Snapshot/Input shed.
    /// Surfaced as its OWN machine-observable counter (matching `BoundedInbox`'s
    /// `dropped_reliable`/`dropped_unreliable` split) so an oracle/admin can gate
    /// `reliable_shed == 0` on any healthy run — a reliable post-commit shed must never
    /// be indistinguishable from a dropped snapshot. MUST be 0 in zero-fault operation.
    pub reliable_shed: usize,
    /// `NodeUnreachable` notices observed among the drained inbound.
    pub unreachable: usize,
    /// `SendShed` notices observed among the drained inbound (R-4d M3): a LOCAL send the
    /// transport refused — a lane's retry buffer full or an oversize frame. Counted SEPARATELY
    /// from `unreachable` because a shed is NOT a peer-unreachability (the peer may be alive):
    /// conflating them here would re-introduce, at the reporting layer, the exact ambiguity M3
    /// removes at the routing layer. MUST be 0 on a healthy run.
    pub shed: usize,
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
    world.insert_resource(PeerBook::default());
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

/// The mid-tick handoff (D-beta): the [`TickReport`] fields FIXED by
/// [`run_schedule`](ShardNode::run_schedule) before the outbox is flushed. Opaque by design — a
/// durability-aware bin holds it across a wait-for-durable, then hands it to
/// [`flush_outbox`](ShardNode::flush_outbox), without inspecting it. No derives: it is never formatted
/// or compared, so it adds no coverable surface (HR5).
pub struct TickPrologue {
    tick: TickId,
    drained: usize,
    unreachable: usize,
    shed: usize,
}

impl<T: Transport> ShardNode<T> {
    /// The FIRST half of a tick (D-beta): advance the local tick, drain inbound, and run the schedule —
    /// which, on the orchestrator, includes the group-commit barrier that STAGES + submits this tick's
    /// durable batch. The outbox is NOT sent here; that is [`flush_outbox`](Self::flush_outbox), split out
    /// so a durability-aware bin can interpose a wait-for-durable between the two (persist-before-effect:
    /// no effect leaves the node before the state authorizing it is durable — D-gamma's parked-flush). The
    /// returned [`TickPrologue`] carries the report fields fixed before the flush.
    pub fn run_schedule(&mut self) -> TickPrologue {
        self.tick = self.tick.next();
        set_local_tick(&mut self.world, self.tick);
        let (drained, unreachable, shed) = drain_phase(&mut self.transport, &mut self.world);
        self.schedule.run(&mut self.world);
        TickPrologue {
            tick: self.tick,
            drained,
            unreachable,
            shed,
        }
    }

    /// The SECOND half of a tick (D-beta): flush the outbox the schedule staged and assemble the
    /// [`TickReport`]. Takes the [`TickPrologue`] from [`run_schedule`](Self::run_schedule). Split so a
    /// durability-aware bin can wait for the prior tick's batch to be durable BEFORE this send.
    pub fn flush_outbox(&mut self, prologue: TickPrologue) -> TickReport {
        let (sent, backpressured, staging_shed, reliable_shed, unknown_peers) =
            flush_phase(&mut self.transport, &mut self.world);
        TickReport {
            tick: prologue.tick,
            unknown_peers,
            drained: prologue.drained,
            sent,
            backpressured,
            staging_shed,
            reliable_shed,
            unreachable: prologue.unreachable,
            shed: prologue.shed,
        }
    }

    /// THE deterministic unit of progress. Synchronous by construction. A thin wrapper running
    /// [`run_schedule`](Self::run_schedule) then [`flush_outbox`](Self::flush_outbox) back-to-back —
    /// byte-for-byte the pre-split single phase (the in-process harness + MemStore path, which needs no
    /// durability interposition). The durable-orchestrator bin calls the two halves directly instead.
    pub fn step_tick(&mut self) -> TickReport {
        let prologue = self.run_schedule();
        self.flush_outbox(prologue)
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

/// ★ THE PEER BOOK (D-RLM-6 mechanism C, built 2026-09-03 for the ruler switch's inward leg): the
/// node runtime's half. A send the transport refuses as `UnknownPeer` keeps its frame in the staging
/// and asks the clock's source — the orchestrator — where that node listens, once per tick while a
/// frame waits (a small Membership-class ask; unknown peers are rare and short-lived). The answer,
/// `PeerLocated`, is consumed HERE, below the schedule, and booked through the transport seam; the
/// sim never sees an address. Nothing in the sim changes: the up-lane's send to a new parent's node is
/// what misses, and the parent's statement toward the hull's node misses the same way from its side.
#[derive(Resource, Debug, Default)]
pub struct PeerBook {
    /// `PeerLocate` asks sent.
    pub locates_sent: u64,
    /// `PeerLocated` answers booked through the transport.
    pub located: u64,
    /// Asks withheld because no `ClockSync` has named the orchestrator yet.
    pub no_source: u64,
    /// ★ THE ASK RATE (owner 2026-09-04, item 6 of the flight plan): a peer that stays unknown is
    /// asked about on its first missed tick, then on the 2nd, 4th, 8th, … — a doubling backoff
    /// with no constant in it, so a peer nobody can locate costs `log₂(ticks)` asks instead of one
    /// per tick, and a peer located on the first answer costs exactly one. Per peer: the ticks it
    /// has missed so far, and the miss count the next ask goes out at. Cleared when the answer
    /// books the peer. Example: the hull's shard sends its first facts up to the galaxy's node,
    /// which it has never met; it asks the orchestrator at miss 1, the answer lands by miss 2, and
    /// the book never asks again for that node.
    pub pending: BTreeMap<NodeId, (u64, u64)>,
    /// Asks withheld by the backoff (the frame still waits; nothing is lost).
    pub asks_withheld: u64,
    /// Peers this process has warned about once (the warning is per peer, never per tick).
    pub warned: BTreeSet<NodeId>,
}

/// Monomorphic drain: pull everything delivered since last tick into the world. Counts the two
/// async delivery-failure notices SEPARATELY (R-4d M3): `unreachable` (peer down) vs `shed` (a
/// local send refused — orthogonal to peer liveness). The full inbound vec (both included) still
/// goes to the `InboundBox` for the schedule's consumers.
fn drain_phase(transport: &mut dyn Transport, world: &mut World) -> (usize, usize, usize) {
    let mut inbound = transport.drain_inbound();
    // The peer book's answers are consumed here and never reach the schedule (an address is the node
    // runtime's business, not the sim's). Only Membership-class frames are decoded — the cold lane.
    let mut located = 0u64;
    let mut booked: Vec<NodeId> = Vec::new();
    inbound.retain(|m| {
        let Inbound::Wire { class, bytes, .. } = m else {
            return true;
        };
        if *class != MsgClass::Membership {
            return true;
        }
        let Ok(InterShardFlow::PeerLocated(answer)) = postcard::from_bytes::<InterShardFlow>(bytes)
        else {
            return true;
        };
        transport.book_peer(answer.node, answer.ip, answer.port);
        located += 1;
        booked.push(answer.node);
        false
    });
    let mut book = world.resource_mut::<PeerBook>();
    book.located += located;
    for node in booked {
        // The answer closes the ask: a later miss toward this node starts a fresh backoff and a
        // fresh warning, because it is then a new event (the peer moved again).
        book.pending.remove(&node);
        book.warned.remove(&node);
    }
    let drained = inbound.len();
    let unreachable = inbound
        .iter()
        .filter(|m| matches!(m, Inbound::NodeUnreachable { .. }))
        .count();
    let shed = inbound
        .iter()
        .filter(|m| matches!(m, Inbound::SendShed { .. }))
        .count();
    let mut inbox = world.resource_mut::<InboundBox>();
    inbox.0 = inbound;
    (drained, unreachable, shed)
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
/// Returns `(sent, backpressured, staging_shed, reliable_shed)` — `reliable_shed` is the
/// reliable subset of `staging_shed`, surfaced distinctly so the loss of a transfer/control
/// frame is machine-observable, never lumped with benign latest-wins snapshot shedding.
fn flush_phase(
    transport: &mut dyn Transport,
    world: &mut World,
) -> (usize, usize, usize, usize, usize) {
    let pending = std::mem::take(&mut world.resource_mut::<OutboundBox>().0);
    let cap = world.resource::<OutboundStagingCap>().0;
    let mut sent = 0usize;
    // Peers that back-pressured THIS tick; their remaining frames requeue untried.
    let mut blocked: BTreeSet<NodeId> = BTreeSet::new();
    // Peers this process has no lane and no address for: their frames wait, and the book is asked.
    let mut unknown: BTreeSet<NodeId> = BTreeSet::new();
    let mut requeued: Vec<(NodeId, MsgClass, Bytes, Durability)> = Vec::new();
    for (to, class, bytes, durability) in pending {
        if blocked.contains(&to) {
            requeued.push((to, class, bytes, durability));
            continue;
        }
        match transport.send_durable(to, class, bytes, durability) {
            Ok(_) => sent += 1,
            Err(SendError::QueueFull(returned)) => {
                blocked.insert(to);
                requeued.push((to, class, returned, durability));
            }
            Err(SendError::UnknownPeer(returned)) => {
                blocked.insert(to);
                unknown.insert(to);
                requeued.push((to, class, returned, durability));
            }
        }
    }
    let (staging_shed, reliable_shed) = shed_over_cap(&mut requeued, cap);
    let backpressured = requeued.len();
    world.resource_mut::<OutboundBox>().0 = requeued;
    let unknown_peers = unknown.len();
    ask_peer_book(transport, world, &unknown);
    (
        sent,
        backpressured,
        staging_shed,
        reliable_shed,
        unknown_peers,
    )
}

/// Ask the clock's source where each unknown peer listens — one small Membership-class frame per
/// unknown peer on its first missed tick, then with a doubling backoff ([`PeerBook::pending`])
/// while its frames wait. Withheld, counted, until a `ClockSync` has named the orchestrator; a node
/// nobody has synced has nobody to ask. The first miss toward a peer is warned ONCE.
fn ask_peer_book(transport: &mut dyn Transport, world: &mut World, unknown: &BTreeSet<NodeId>) {
    if unknown.is_empty() {
        return;
    }
    let source = world
        .get_resource::<crate::follower::FollowerState>()
        .and_then(|f| f.source);
    let at = world.resource::<ClockSample>().universe_tick;
    let mut book = world.resource_mut::<PeerBook>();
    let Some(source) = source else {
        book.no_source += unknown.len() as u64;
        return;
    };
    for &node in unknown {
        if book.warned.insert(node) {
            tracing::warn!(
                peer = node.0,
                "no lane and no address for a peer — its frames wait while the orchestrator is asked"
            );
        }
        let (misses, next_ask) = book.pending.entry(node).or_insert((0, 1));
        *misses += 1;
        if *misses != *next_ask {
            book.asks_withheld += 1;
            continue;
        }
        *next_ask = next_ask.saturating_mul(2);
        let flow = InterShardFlow::PeerLocate(vd_wire::intershard::PeerLocate { node, at });
        let bytes = postcard::to_allocvec(&flow).expect("closed wire enums serialize infallibly");
        // A refused ask is covered by the next backoff step; nothing is owed on a miss here.
        let _ = transport.send(source, MsgClass::Membership, vd_sim::io::bytes(bytes));
        book.locates_sent += 1;
    }
}

/// Bound the carried-over staging to `cap`, shedding OLDEST-first but UNRELIABLE-FIRST
/// (the ROB-2 reliability-aware discipline, matching `BoundedInbox`): latest-wins
/// Snapshot/Input frames are the disposable casualties; a RELIABLE Saga/Control/
/// Membership frame is sacrificed ONLY when shedding every droppable unreliable frame
/// still leaves the backlog over cap — and that reliable loss is its own distinct
/// `error!`-level ALERT, never quietly lumped with snapshots. Survivors keep FIFO order.
/// Returns `(total_shed, reliable_shed)`. Monomorphic (the generic flush stays a shim).
fn shed_over_cap(
    requeued: &mut Vec<(NodeId, MsgClass, Bytes, Durability)>,
    cap: usize,
) -> (usize, usize) {
    let over = requeued.len().saturating_sub(cap);
    if over == 0 {
        return (0, 0);
    }
    let unreliable_total = requeued
        .iter()
        .filter(|(_, class, _, _)| class.reliability() == Reliability::Unreliable)
        .count();
    let unreliable_shed = over.min(unreliable_total);
    let reliable_shed = over - unreliable_shed;
    // Drop the oldest `unreliable_shed` unreliable + oldest `reliable_shed` reliable
    // frames (front-to-back walk = oldest-first); retain keeps the rest in FIFO order.
    let mut ud = unreliable_shed;
    let mut rd = reliable_shed;
    requeued.retain(|(_, class, _, _)| match class.reliability() {
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
                outbox.0.push((
                    *from,
                    *class,
                    bytes.clone(),
                    vd_sim::io::Durability::Ephemeral,
                ));
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
        a.world_mut().resource_mut::<OutboundBox>().0.push((
            B,
            MsgClass::Control,
            vec![42].into(),
            vd_sim::io::Durability::Ephemeral,
        ));
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
                outbox.0.push((
                    B,
                    MsgClass::Input,
                    vec![n].into(),
                    vd_sim::io::Durability::Ephemeral,
                ));
            }
        }
        let report = a.step_tick();
        assert_eq!((report.sent, report.backpressured), (2, 2));
        // The refused payloads are intact and ordered (returned, not lost).
        assert_eq!(
            a.world_mut().resource::<OutboundBox>().0,
            vec![
                (
                    B,
                    MsgClass::Input,
                    vec![2].into(),
                    vd_sim::io::Durability::Ephemeral
                ),
                (
                    B,
                    MsgClass::Input,
                    vec![3].into(),
                    vd_sim::io::Durability::Ephemeral
                )
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
        a.world_mut().resource_mut::<OutboundBox>().0.push((
            B,
            MsgClass::Saga,
            vec![1].into(),
            vd_sim::io::Durability::Ephemeral,
        ));
        let r1 = a.step_tick();
        assert_eq!(r1.sent, 1, "enqueue succeeded; failure is async");
        hub.pump();
        let r2 = a.step_tick();
        assert_eq!((r2.drained, r2.unreachable), (1, 1));
    }

    // ---- THE PEER BOOK (D-RLM-6 mechanism C) ----------------------------------------------------

    /// A transport with no lane to `unknown` until an address is booked; records every booking.
    struct BookedLanes {
        local: NodeId,
        unknown: BTreeSet<NodeId>,
        booked: Vec<(NodeId, [u8; 16], u16)>,
        sent: Vec<(NodeId, MsgClass, Vec<u8>)>,
        inbound: Vec<Inbound>,
    }

    impl Transport for BookedLanes {
        fn send_durable(
            &mut self,
            to: NodeId,
            class: MsgClass,
            bytes: Bytes,
            _durability: vd_sim::io::Durability,
        ) -> Result<MsgId, SendError> {
            if self.unknown.contains(&to) {
                return Err(SendError::UnknownPeer(bytes));
            }
            self.sent.push((to, class, bytes.to_vec()));
            Ok(MsgId(self.sent.len() as u64))
        }
        fn drain_inbound(&mut self) -> Vec<Inbound> {
            std::mem::take(&mut self.inbound)
        }
        fn local_id(&self) -> NodeId {
            self.local
        }
        fn book_peer(&mut self, node: NodeId, ip: [u8; 16], port: u16) {
            self.booked.push((node, ip, port));
            self.unknown.remove(&node);
        }
    }

    const ORCH: NodeId = NodeId(1);
    const STRANGER: NodeId = NodeId(1_007);

    fn located(node: NodeId) -> Inbound {
        Inbound::Wire {
            from: ORCH,
            class: MsgClass::Membership,
            bytes: postcard::to_allocvec(&InterShardFlow::PeerLocated(
                vd_wire::intershard::PeerLocated {
                    node,
                    ip: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0xff, 0xff, 127, 0, 0, 1],
                    port: 7_562,
                    at: vd_core::UniverseTick(9),
                },
            ))
            .expect("encode")
            .into(),
        }
    }

    #[test]
    fn a_send_to_an_unknown_peer_waits_and_asks_the_clocks_source_then_the_answer_books_it() {
        let transport = BookedLanes {
            local: A,
            unknown: BTreeSet::from([STRANGER]),
            booked: Vec::new(),
            sent: Vec::new(),
            inbound: Vec::new(),
        };
        let mut node = build_app(stub_cfg(A), transport);
        assert_eq!(
            node.transport.local_id(),
            A,
            "the transport serves THIS node: every ask below names the peer it misses, never itself"
        );
        // Nobody has synced this node: the ask is withheld and counted, the frame still waits.
        node.world_mut().resource_mut::<OutboundBox>().0.push((
            STRANGER,
            MsgClass::Control,
            vec![7].into(),
            vd_sim::io::Durability::Ephemeral,
        ));
        let report = node.step_tick();
        assert_eq!(
            (report.sent, report.backpressured, report.unknown_peers),
            (0, 1, 1)
        );
        assert_eq!(node.world_mut().resource::<PeerBook>().no_source, 1);
        assert!(node.transport.sent.is_empty(), "nobody to ask yet");
        // The clock's source is known: the ask goes out on the first miss while the frame waits.
        node.world_mut()
            .insert_resource(crate::follower::FollowerState {
                source: Some(ORCH),
                ..Default::default()
            });
        let report = node.step_tick();
        assert_eq!((report.backpressured, report.unknown_peers), (1, 1));
        let book = node.world_mut().resource::<PeerBook>();
        assert_eq!((book.locates_sent, book.located), (1, 0));
        let ask = node.transport.sent.last().expect("the ask was sent");
        assert_eq!((ask.0, ask.1), (ORCH, MsgClass::Membership));
        assert_eq!(
            postcard::from_bytes::<InterShardFlow>(&ask.2),
            Ok(InterShardFlow::PeerLocate(
                vd_wire::intershard::PeerLocate {
                    node: STRANGER,
                    at: node.world_mut().resource::<ClockSample>().universe_tick,
                }
            ))
        );
        // The answer is consumed below the schedule, booked through the seam, and never reaches the
        // inbox; the waiting frame goes out on the same tick.
        node.transport.inbound.push(located(STRANGER));
        let report = node.step_tick();
        assert_eq!(
            (report.drained, report.sent, report.backpressured),
            (0, 1, 0)
        );
        assert_eq!(
            node.transport.booked,
            vec![(
                STRANGER,
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0xff, 0xff, 127, 0, 0, 1],
                7_562
            )]
        );
        assert_eq!(node.world_mut().resource::<PeerBook>().located, 1);
        assert_eq!(node.transport.sent.last().map(|s| s.0), Some(STRANGER));
    }

    /// ★ THE ASK RATE (owner 2026-09-04): a peer that stays unknown is asked about at its 1st, 2nd,
    /// 4th, 8th missed tick — never every tick — and warned about once; the answer clears the
    /// backoff so a later move of the same peer is a fresh event.
    #[test]
    fn a_peer_that_stays_unknown_is_asked_with_a_doubling_backoff_and_warned_once() {
        let transport = BookedLanes {
            local: A,
            unknown: BTreeSet::from([STRANGER]),
            booked: Vec::new(),
            sent: Vec::new(),
            inbound: Vec::new(),
        };
        let mut node = build_app(stub_cfg(A), transport);
        node.world_mut()
            .insert_resource(crate::follower::FollowerState {
                source: Some(ORCH),
                ..Default::default()
            });
        node.world_mut().resource_mut::<OutboundBox>().0.push((
            STRANGER,
            MsgClass::Control,
            vec![7].into(),
            vd_sim::io::Durability::Ephemeral,
        ));
        for _ in 0..9 {
            let report = node.step_tick();
            assert_eq!((report.backpressured, report.unknown_peers), (1, 1));
        }
        let book = node.world_mut().resource::<PeerBook>();
        // Misses 1, 2, 4 and 8 asked; misses 3, 5, 6, 7 and 9 were withheld.
        assert_eq!((book.locates_sent, book.asks_withheld), (4, 5));
        assert_eq!(book.pending.get(&STRANGER), Some(&(9, 16)));
        assert_eq!(book.warned.len(), 1, "one warning per peer, not per tick");
        assert_eq!(node.transport.sent.len(), 4);
        // The answer books the peer and closes the ask; the frame goes out.
        node.transport.inbound.push(located(STRANGER));
        let report = node.step_tick();
        assert_eq!((report.sent, report.backpressured), (1, 0));
        let book = node.world_mut().resource::<PeerBook>();
        assert!(book.pending.is_empty() & book.warned.is_empty());
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
        /// Inbound to hand back on the NEXT `drain_inbound` (drained once). Lets a test inject an
        /// arbitrary notice — e.g. an `Inbound::SendShed` the MemHub cannot produce (R-4d M3).
        inbound: Vec<Inbound>,
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
        fn send_durable(
            &mut self,
            to: NodeId,
            _class: MsgClass,
            bytes: Bytes,
            _durability: vd_sim::io::Durability,
        ) -> Result<MsgId, SendError> {
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
            std::mem::take(&mut self.state.borrow_mut().inbound)
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
            outbox.0.push((
                X,
                MsgClass::Input,
                vec![0].into(),
                vd_sim::io::Durability::Ephemeral,
            ));
            outbox.0.push((
                Y,
                MsgClass::Input,
                vec![1].into(),
                vd_sim::io::Durability::Ephemeral,
            ));
            outbox.0.push((
                X,
                MsgClass::Input,
                vec![2].into(),
                vd_sim::io::Durability::Ephemeral,
            ));
            outbox.0.push((
                Y,
                MsgClass::Input,
                vec![3].into(),
                vd_sim::io::Durability::Ephemeral,
            ));
            outbox.0.push((
                X,
                MsgClass::Input,
                vec![4].into(),
                vd_sim::io::Durability::Ephemeral,
            ));
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
                (
                    X,
                    MsgClass::Input,
                    vec![2].into(),
                    vd_sim::io::Durability::Ephemeral
                ),
                (
                    X,
                    MsgClass::Input,
                    vec![4].into(),
                    vd_sim::io::Durability::Ephemeral
                ),
            ]
        );
    }

    #[test]
    fn a_send_shed_is_counted_separately_from_an_unreachable() {
        use vd_sim::io::ShedReason;
        // MemHub can never produce a SendShed (no retry buffer), so inject via the lane-state seam.
        let (transport, state) = PerPeerLanes::with_caps(A, &[]);
        // One shed + one unreachable toward the same peer, in one drain: drain_phase must tally them
        // into the DISTINCT report fields (R-4d M3 — a shed is NOT a peer-unreachability, so conflating
        // them at the reporting layer would re-introduce the ambiguity M3 removes at routing).
        state.borrow_mut().inbound = vec![
            Inbound::SendShed {
                to: B,
                class: MsgClass::Saga,
                undelivered: MsgId(0),
                reason: ShedReason::RetryBufferFull,
            },
            Inbound::NodeUnreachable {
                to: B,
                class: MsgClass::Saga,
                undelivered: MsgId(1),
            },
        ];
        let mut node = build_app(stub_cfg(A), transport);
        let report = node.step_tick();
        assert_eq!(report.drained, 2);
        assert_eq!(report.shed, 1, "the SendShed is counted as a shed");
        assert_eq!(
            report.unreachable, 1,
            "the NodeUnreachable is counted separately, not folded into shed"
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
                outbox.0.push((
                    X,
                    MsgClass::Input,
                    vec![n].into(),
                    vd_sim::io::Durability::Ephemeral,
                ));
            }
        }
        let report = node.step_tick();

        assert_eq!(report.sent, 0); // cap-0 lane accepts nothing
        assert_eq!(report.staging_shed, 3); // 5 staged, cap 2 ⇒ 3 oldest shed
        assert_eq!(
            report.reliable_shed, 0,
            "an all-unreliable shed surfaces ZERO reliable_shed — the disposable-class drop is NOT an alert",
        );
        assert_eq!(report.backpressured, 2); // exactly the cap carries over
        // The OLDEST (0,1,2) are shed; the NEWEST (3,4) survive, in order.
        assert_eq!(
            node.world_mut().resource::<OutboundBox>().0,
            vec![
                (
                    X,
                    MsgClass::Input,
                    vec![3].into(),
                    vd_sim::io::Durability::Ephemeral
                ),
                (
                    X,
                    MsgClass::Input,
                    vec![4].into(),
                    vd_sim::io::Durability::Ephemeral
                ),
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
            outbox.0.push((
                X,
                MsgClass::Saga,
                vec![0].into(),
                vd_sim::io::Durability::Ephemeral,
            ));
            outbox.0.push((
                X,
                MsgClass::Input,
                vec![1].into(),
                vd_sim::io::Durability::Ephemeral,
            ));
            outbox.0.push((
                X,
                MsgClass::Input,
                vec![2].into(),
                vd_sim::io::Durability::Ephemeral,
            ));
            outbox.0.push((
                X,
                MsgClass::Saga,
                vec![3].into(),
                vd_sim::io::Durability::Ephemeral,
            ));
            outbox.0.push((
                X,
                MsgClass::Input,
                vec![4].into(),
                vd_sim::io::Durability::Ephemeral,
            ));
        }
        let report = node.step_tick();

        assert_eq!(report.staging_shed, 3); // the three Inputs
        assert_eq!(
            report.reliable_shed, 0,
            "the reliable frames survived, so reliable_shed is 0 even though 3 frames were shed",
        );
        assert_eq!(
            node.world_mut().resource::<OutboundBox>().0,
            vec![
                (
                    X,
                    MsgClass::Saga,
                    vec![0].into(),
                    vd_sim::io::Durability::Ephemeral
                ),
                (
                    X,
                    MsgClass::Saga,
                    vec![3].into(),
                    vd_sim::io::Durability::Ephemeral
                ),
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
                outbox.0.push((
                    X,
                    MsgClass::Saga,
                    vec![n].into(),
                    vd_sim::io::Durability::Ephemeral,
                ));
            }
        }
        let report = node.step_tick();

        assert_eq!(report.staging_shed, 2); // 3 reliable staged, cap 1 ⇒ 2 shed
        assert_eq!(
            report.reliable_shed, 2,
            "the reliable shed is surfaced DISTINCTLY (== staging_shed here) — a transfer/control \
             loss is machine-observable, an oracle/admin can gate reliable_shed == 0",
        );
        assert_eq!(
            node.world_mut().resource::<OutboundBox>().0,
            vec![(
                X,
                MsgClass::Saga,
                vec![2].into(),
                vd_sim::io::Durability::Ephemeral
            )],
            "the newest reliable frame survives; the two oldest are the reliable casualties"
        );
    }
}
