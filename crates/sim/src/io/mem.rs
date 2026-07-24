//! In-memory deterministic `Transport` implementation.
//!
//! SPIKE-0a scope: a minimal hub with explicit, deterministic delivery. The full
//! `FaultFabric` (drop/dup/reorder/delay/partition/crash, at-least-once redelivery)
//! grows on top of this in task P0.7 — the *semantics* established here are binding:
//!
//! - `send` enqueues into a **bounded** per-node outbound queue (`Err(QueueFull)` when
//!   saturated) and performs no delivery — mirroring the prod writer-task model where
//!   the sim thread only ever touches queues.
//! - Delivery happens on an explicit [`MemHub::pump`] step (the analog of the prod
//!   writer task draining), moving outbound frames into peer inbound queues in FIFO
//!   order, or converting them into [`Inbound::NodeUnreachable`] if the peer is dead.
//!
//! The mem hub has NO bounded retry buffer and no oversize framing, so it never emits
//! [`Inbound::SendShed`] (R-4d M3) — a shed is an io-prod-mesh-only signal. Consumers treat
//! its in-process absence as normal; mem/mesh parity is preserved by ABSENCE, not by
//! fabricating a mem shed path.

use std::collections::{BTreeMap, BTreeSet, VecDeque};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex, MutexGuard, PoisonError};

use vd_core::realm_coord::RealmCoord;
use vd_core::taxonomy::ProfileKind;
use vd_core::{EpochId, MsgId, NodeId, TickId, UniverseTick};

use super::{
    BoundedInbox, Bytes, Clock, Inbound, MsgClass, RealmSpawner, SendError, SpawnError, Store,
    Transport,
};

/// The deterministic test clock: the topology driver advances it explicitly; nodes
/// only ever READ it through the [`Clock`] trait. Shared-handle semantics (clone =
/// same clock) so a driver and its node see one timeline.
#[derive(Clone, Debug)]
pub struct VirtualClock {
    inner: Arc<VirtualClockInner>,
}

#[derive(Debug)]
struct VirtualClockInner {
    local_tick: AtomicU64,
    universe_tick: AtomicU64,
    epoch: EpochId,
}

impl VirtualClock {
    #[must_use]
    pub fn new(epoch: EpochId) -> VirtualClock {
        VirtualClock {
            inner: Arc::new(VirtualClockInner {
                local_tick: AtomicU64::new(0),
                universe_tick: AtomicU64::new(0),
                epoch,
            }),
        }
    }

    /// Advance this node's local tick by one (the topology's per-node step).
    pub fn advance_local(&self) -> TickId {
        TickId(self.inner.local_tick.fetch_add(1, Ordering::Relaxed) + 1)
    }

    /// Set the analytic clock view (the topology plays the orchestrator's role).
    pub fn set_universe_tick(&self, tick: UniverseTick) {
        self.inner.universe_tick.store(tick.0, Ordering::Relaxed);
    }
}

impl Clock for VirtualClock {
    fn local_tick(&self) -> TickId {
        TickId(self.inner.local_tick.load(Ordering::Relaxed))
    }
    fn universe_tick(&self) -> UniverseTick {
        UniverseTick(self.inner.universe_tick.load(Ordering::Relaxed))
    }
    fn epoch(&self) -> EpochId {
        self.inner.epoch
    }
}

/// The deterministic test [`Store`] (D-6): a staged/committed two-tier map. `committed` survives a node
/// REBUILD (the harness retains this handle and re-attaches it to the fresh node) while `staged` is
/// dropped when the World dies — modeling the prod fsync window the orchestrator-kill cells must crash
/// ACROSS (a put-is-instantly-durable store would make the lose-the-uncommitted-batch crash untestable,
/// and the no-split-brain proof rests on that boundary). Shared-handle semantics (clone = same backing
/// log), exactly like [`MemHub`]/[`VirtualClock`].
#[derive(Clone, Debug, Default)]
pub struct MemStore {
    inner: Arc<Mutex<StoreInner>>,
}

#[derive(Debug, Default)]
struct StoreInner {
    /// Durable: survives a crash + a node rebuild.
    committed: BTreeMap<Vec<u8>, Bytes>,
    /// Pending this fsync window: `Some` = staged put, `None` = staged delete. Dropped on a crash
    /// before `commit` (the un-fsynced batch is lost), merged into `committed` on `commit`.
    staged: BTreeMap<Vec<u8>, Option<Bytes>>,
}

impl MemStore {
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    fn lock(&self) -> MutexGuard<'_, StoreInner> {
        self.inner.lock().unwrap_or_else(PoisonError::into_inner)
    }

    /// Whether anything is durably COMMITTED — the orchestrator's genesis-vs-recover discriminator on
    /// rebuild (empty ⇒ fresh genesis; non-empty ⇒ rehydrate). Staged-but-uncommitted does not count.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.lock().committed.is_empty()
    }
}

impl Store for MemStore {
    fn put(&mut self, key: &[u8], value: &Bytes) {
        self.lock().staged.insert(key.to_vec(), Some(value.clone()));
    }

    fn delete(&mut self, key: &[u8]) {
        self.lock().staged.insert(key.to_vec(), None);
    }

    fn scan(&self, prefix: &[u8]) -> Vec<(Vec<u8>, Bytes)> {
        self.lock()
            .committed
            .iter()
            .filter(|(k, _)| k.starts_with(prefix))
            .map(|(k, v)| (k.clone(), v.clone()))
            .collect()
    }

    fn commit(&mut self) {
        let mut inner = self.lock();
        let staged = std::mem::take(&mut inner.staged);
        for (key, value) in staged {
            match value {
                Some(bytes) => {
                    inner.committed.insert(key, bytes);
                }
                None => {
                    inner.committed.remove(&key);
                }
            }
        }
    }
}

#[derive(Debug)]
struct OutboundFrame {
    to: NodeId,
    class: MsgClass,
    bytes: Bytes,
    msg_id: MsgId,
}

#[derive(Debug)]
struct NodeQueues {
    /// Bounded by `outbound_capacity`; drained by `pump`.
    outbound: VecDeque<OutboundFrame>,
    outbound_capacity: usize,
    /// Delivered messages awaiting `drain_inbound` — BOUNDED with the seam's
    /// reliability-aware overflow policy (a fast sender cannot OOM a slow node).
    inbound: BoundedInbox,
    /// Next FIFO send sequence.
    next_msg_id: u64,
    /// Dead nodes stop receiving; sends toward them become `NodeUnreachable`.
    alive: bool,
}

#[derive(Debug, Default)]
struct HubInner {
    nodes: BTreeMap<NodeId, NodeQueues>,
}

/// A hub connecting any number of [`MemTransport`]s with explicit, deterministic pumping.
#[derive(Clone, Debug, Default)]
pub struct MemHub {
    inner: Arc<Mutex<HubInner>>,
}

impl MemHub {
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    fn lock(&self) -> MutexGuard<'_, HubInner> {
        self.inner.lock().unwrap_or_else(PoisonError::into_inner)
    }

    /// Register a node and obtain its transport endpoint.
    ///
    /// # Panics
    /// Panics if `id` is already registered — node identities are unique by
    /// construction, so a duplicate registration is a harness bug, not a runtime
    /// condition.
    #[must_use]
    pub fn register(&self, id: NodeId, outbound_capacity: usize) -> MemTransport {
        // The inbound bound is generous relative to outbound: it exists to make
        // overflow REPRODUCIBLE, not to throttle normal traffic. `register_bounded`
        // sets it explicitly for overflow scenarios.
        self.register_bounded(
            id,
            outbound_capacity,
            outbound_capacity.saturating_mul(8).max(64),
        )
    }

    /// Register with an explicit inbound capacity (overflow-scenario tests).
    ///
    /// # Panics
    /// On duplicate registration.
    #[must_use]
    pub fn register_bounded(
        &self,
        id: NodeId,
        outbound_capacity: usize,
        inbound_capacity: usize,
    ) -> MemTransport {
        let mut inner = self.lock();
        assert!(
            !inner.nodes.contains_key(&id),
            "duplicate node registration: {id}"
        );
        inner.nodes.insert(
            id,
            NodeQueues {
                outbound: VecDeque::new(),
                outbound_capacity,
                inbound: BoundedInbox::new(inbound_capacity),
                next_msg_id: 0,
                alive: true,
            },
        );
        MemTransport {
            local: id,
            hub: self.clone(),
        }
    }

    /// Re-attach a FRESH transport endpoint for an already-registered node — the kill-9-REBUILD analog
    /// (D-6): the node's process died (its in-flight outbound/inbound queues are LOST with the dead
    /// process) and a fresh process reclaims the same identity + peer connectivity. RESETS the node's
    /// queues + `next_msg_id` + `alive`. The node's DURABLE state survives SEPARATELY via its retained
    /// `Store` handle — exactly what the rebuilt node re-hydrates from (the World's RAM is gone, the WAL
    /// is not). Distinct from `register` (which refuses a duplicate) — rebuild PRESUMES a prior identity.
    ///
    /// # Panics
    /// If `id` was never registered (a rebuild reclaims an existing identity).
    #[must_use]
    pub fn reregister(&self, id: NodeId, outbound_capacity: usize) -> MemTransport {
        let mut inner = self.lock();
        let node = inner
            .nodes
            .get_mut(&id)
            .expect("reregister reclaims a prior registration");
        *node = NodeQueues {
            outbound: VecDeque::new(),
            outbound_capacity,
            inbound: BoundedInbox::new(outbound_capacity.saturating_mul(8).max(64)),
            next_msg_id: 0,
            alive: true,
        };
        MemTransport {
            local: id,
            hub: self.clone(),
        }
    }

    /// Kill a node: it stops receiving; in-flight and future frames toward it surface
    /// to their senders as [`Inbound::NodeUnreachable`] on the next pump.
    pub fn kill(&self, id: NodeId) {
        if let Some(node) = self.lock().nodes.get_mut(&id) {
            node.alive = false;
        }
    }

    /// REMOVE a node from the hub entirely (RLM Step 1): a torn-down realm's transport slot is
    /// reclaimed. DISTINCT from [`MemHub::kill`] (flag-only, for the kill-9 crash model where the id
    /// may be `reregister`ed): `deregister` is safe to fully remove ONLY because the caller
    /// ([`MemSpawner`], F2-monotone) NEVER re-registers a removed id — so there is no resurrection.
    /// A no-op on an unknown id (mirrors `kill`). Prevents unbounded hub accretion under the
    /// reconciler's spawn/kill churn.
    pub fn deregister(&self, id: NodeId) {
        self.lock().nodes.remove(&id);
    }

    /// The number of registered (not-yet-deregistered) nodes — the accretion assertion for RLM churn.
    #[must_use]
    pub fn node_count(&self) -> usize {
        self.lock().nodes.len()
    }

    /// A fresh transport HANDLE for an already-registered `id` (the RLM harness plant retrieves the
    /// spawned node's endpoint here, since [`MemSpawner::spawn_realm`] discards the register handle).
    /// `None` if `id` is not registered. A handle is a cheap `{id, hub}` view — several handles to one
    /// node all route through its single hub queue.
    #[must_use]
    pub fn transport_for(&self, id: NodeId) -> Option<MemTransport> {
        if self.lock().nodes.contains_key(&id) {
            Some(MemTransport {
                local: id,
                hub: self.clone(),
            })
        } else {
            None
        }
    }

    /// Deterministically deliver every queued outbound frame: FIFO per sender, senders
    /// processed in `NodeId` order (BTreeMap iteration — never hash order).
    pub fn pump(&self) {
        let mut inner = self.lock();
        // Two phases under ONE held lock: drain every sender, then route. The lock is
        // held throughout, so routing targets are guaranteed present: a deliverable
        // frame's receiver was just checked alive, and an undeliverable frame routes
        // back to its sender, whose entry produced the frame.
        let drained: Vec<(NodeId, Vec<OutboundFrame>)> = inner
            .nodes
            .iter_mut()
            .map(|(id, node)| (*id, node.outbound.drain(..).collect()))
            .collect();
        for (sender, frames) in drained {
            for frame in frames {
                let deliverable = inner.nodes.get(&frame.to).is_some_and(|n| n.alive);
                let (target, event) = if deliverable {
                    (
                        frame.to,
                        Inbound::Wire {
                            from: sender,
                            class: frame.class,
                            bytes: frame.bytes,
                        },
                    )
                } else {
                    (
                        sender,
                        Inbound::NodeUnreachable {
                            to: frame.to,
                            class: frame.class,
                            undelivered: frame.msg_id,
                        },
                    )
                };
                // Surface the overflow verdict (audit ROB-2): a dropped RELIABLE event is
                // the design's explicit overload ALERT — warn loudly even in the in-memory
                // hub (the BoundedInbox counters stay the queryable surface for tests).
                let dropped = inner
                    .nodes
                    .get_mut(&target)
                    .expect("pump target exists: receiver checked alive, or sender owns the frame")
                    .inbound
                    .push(event);
                if dropped == Some(crate::io::InboxDrop::Reliable) {
                    tracing::warn!(
                        node = target.0,
                        "BoundedInbox FULL of reliable events: a RELIABLE inbound was \
                         dropped (genuine overload)"
                    );
                }
            }
        }
    }
}

/// One node's endpoint into a [`MemHub`].
#[derive(Debug)]
pub struct MemTransport {
    local: NodeId,
    hub: MemHub,
}

impl Transport for MemTransport {
    // `_durability`: the mem tier has no durable outbox — a Retained one-shot is delivered like any other
    // (behavior-identical to Ephemeral); the R-6d write-through is io-prod-only.
    fn send_durable(
        &mut self,
        to: NodeId,
        class: MsgClass,
        bytes: Bytes,
        _durability: super::Durability,
    ) -> Result<MsgId, SendError> {
        let mut inner = self.hub.lock();
        let node = inner
            .nodes
            .get_mut(&self.local)
            .expect("registered node present in hub");
        if node.outbound.len() >= node.outbound_capacity {
            return Err(SendError::QueueFull(bytes));
        }
        let msg_id = MsgId(node.next_msg_id);
        node.next_msg_id += 1;
        node.outbound.push_back(OutboundFrame {
            to,
            class,
            bytes,
            msg_id,
        });
        Ok(msg_id)
    }

    fn drain_inbound(&mut self) -> Vec<Inbound> {
        let mut inner = self.hub.lock();
        match inner.nodes.get_mut(&self.local) {
            Some(node) => node.inbound.drain(),
            None => Vec::new(),
        }
    }

    fn local_id(&self) -> NodeId {
        self.local
    }
}

impl MemTransport {
    /// The count of RELIABLE inbound events this node's inbox has dropped under overload
    /// (the design's loud-ALERT case — also `tracing::warn`ed at the pump). A test/oracle
    /// surface for the ROB-2 never-silent guarantee; 0 in any healthy run.
    #[must_use]
    pub fn inbound_dropped_reliable(&self) -> u64 {
        self.hub
            .lock()
            .nodes
            .get(&self.local)
            .map_or(0, |node| node.inbound.dropped_reliable())
    }
}

/// The in-process [`RealmSpawner`] twin (RLM Step 1) — the deterministic harness counterpart to the
/// real-process spawner (Step 5), like [`MemStore`]/[`MemHub`]. Mints monotone `NodeId`s (F2: never
/// reused), plants each node's transport in a shared [`MemHub`], and records `(ProfileKind, at_tick)`
/// per live node. No wall clock, no RNG; all state `BTreeMap`/`BTreeSet` (deterministic order).
#[derive(Clone, Debug)]
pub struct MemSpawner {
    inner: Arc<Mutex<SpawnerInner>>,
}

#[derive(Debug)]
struct SpawnerInner {
    hub: MemHub,
    next_node: u64,
    live: BTreeMap<NodeId, (ProfileKind, UniverseTick)>,
    killed: BTreeSet<NodeId>,
    outbound_capacity: usize,
}

impl MemSpawner {
    /// `first_node` starts the monotone mint range (choose it ABOVE the scenario's statically-picked
    /// ids so a minted id never collides with a hand-picked one); `outbound_capacity` is the hub
    /// registration bound for a spawned node. Both are config values (no inline magic number).
    #[must_use]
    pub fn new(hub: MemHub, first_node: NodeId, outbound_capacity: usize) -> Self {
        MemSpawner {
            inner: Arc::new(Mutex::new(SpawnerInner {
                hub,
                next_node: first_node.0,
                live: BTreeMap::new(),
                killed: BTreeSet::new(),
                outbound_capacity,
            })),
        }
    }

    fn lock(&self) -> MutexGuard<'_, SpawnerInner> {
        self.inner.lock().unwrap_or_else(PoisonError::into_inner)
    }

    /// The live nodes' `(ProfileKind, at_tick)` witnesses — the reconciler reads this to compare
    /// desired-vs-actual (the `at_tick` says WHEN a realm was spawned), and the harness reads it to
    /// build the spawned node.
    #[must_use]
    pub fn live(&self) -> BTreeMap<NodeId, (ProfileKind, UniverseTick)> {
        self.lock().live.clone()
    }
}

impl RealmSpawner for MemSpawner {
    fn spawn_realm(&self, coord: &RealmCoord, at_tick: UniverseTick) -> Result<NodeId, SpawnError> {
        let kind = coord.profile_kind();
        let mut g = self.lock();
        let id = NodeId(g.next_node);
        g.next_node += 1; // F2: monotone — never decremented, never reused
        // Plant the node's transport in the hub (the harness retrieves the handle via
        // `MemHub::transport_for`); monotone ids mean `register` never sees a duplicate.
        let cap = g.outbound_capacity;
        let _transport = g.hub.register(id, cap);
        g.live.insert(id, (kind, at_tick));
        Ok(id)
    }

    fn kill_realm(&self, node: NodeId) -> Result<(), SpawnError> {
        let mut g = self.lock();
        if g.killed.contains(&node) {
            Err(SpawnError::AlreadyKilled(node))
        } else if !g.live.contains_key(&node) {
            Err(SpawnError::UnknownNode(node))
        } else {
            g.live.remove(&node);
            g.killed.insert(node);
            g.hub.deregister(node); // F-B: remove, not flag — no hub accretion under churn
            Ok(())
        }
    }

    fn live_nodes(&self) -> BTreeSet<NodeId> {
        self.lock().live.keys().copied().collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const A: NodeId = NodeId(1);
    const B: NodeId = NodeId(2);

    #[test]
    fn send_is_enqueue_only_until_pump() {
        let hub = MemHub::new();
        let mut a = hub.register(A, 8);
        let mut b = hub.register(B, 8);

        a.send(B, MsgClass::Control, vec![1].into())
            .expect("accepted");
        assert!(b.drain_inbound().is_empty(), "no delivery before pump");

        hub.pump();
        let got = b.drain_inbound();
        assert_eq!(
            got,
            vec![Inbound::Wire {
                from: A,
                class: MsgClass::Control,
                bytes: vec![1].into(),
            }]
        );
    }

    #[test]
    fn bounded_queue_backpressures_exactly_at_capacity() {
        let hub = MemHub::new();
        let mut a = hub.register(A, 4);
        let _b = hub.register(B, 4);

        let results: Vec<Result<MsgId, SendError>> = (0..6)
            .map(|n| a.send(B, MsgClass::Input, vec![n].into()))
            .collect();
        let accepted = results.iter().filter(|r| r.is_ok()).count();
        let rejected = results.iter().filter(|r| r.is_err()).count();
        assert_eq!((accepted, rejected), (4, 2));
        assert_eq!(results[4], Err(SendError::QueueFull(vec![4].into())));
    }

    #[test]
    fn fifo_order_and_fifo_msg_ids() {
        let hub = MemHub::new();
        let mut a = hub.register(A, 8);
        let mut b = hub.register(B, 8);
        assert_eq!((a.local_id(), b.local_id()), (A, B));

        let id0 = a
            .send(B, MsgClass::Control, vec![0].into())
            .expect("accepted");
        let id1 = a
            .send(B, MsgClass::Control, vec![1].into())
            .expect("accepted");
        assert_eq!((id0, id1), (MsgId(0), MsgId(1)));

        hub.pump();
        assert_eq!(
            b.drain_inbound(),
            vec![
                Inbound::Wire {
                    from: A,
                    class: MsgClass::Control,
                    bytes: vec![0].into(),
                },
                Inbound::Wire {
                    from: A,
                    class: MsgClass::Control,
                    bytes: vec![1].into(),
                },
            ]
        );
    }

    #[test]
    fn dead_peer_surfaces_node_unreachable_with_send_correlation() {
        let hub = MemHub::new();
        let mut a = hub.register(A, 8);
        let _b = hub.register(B, 8);

        let sent = a.send(B, MsgClass::Saga, vec![7].into()).expect("accepted");
        hub.kill(B);
        hub.pump();

        assert_eq!(
            a.drain_inbound(),
            vec![Inbound::NodeUnreachable {
                to: B,
                class: MsgClass::Saga,
                undelivered: sent,
            }]
        );
    }

    #[test]
    fn drain_after_deregistration_is_empty() {
        let hub = MemHub::new();
        let mut a = hub.register(A, 8);
        // Simulate a node whose hub entry vanished (crash modeling in P0.7 retains the
        // store but may drop transport state): drain must not panic.
        hub.lock().nodes.remove(&A);
        assert!(a.drain_inbound().is_empty());
    }

    #[test]
    #[should_panic(expected = "duplicate node registration")]
    fn duplicate_registration_panics() {
        let hub = MemHub::new();
        let _a = hub.register(A, 8);
        let _dup = hub.register(A, 8);
    }

    #[test]
    fn bounded_inbound_drops_stale_snapshots_under_a_flood() {
        // A slow consumer with a tiny inbound: a snapshot flood is bounded to the
        // newest frames (latest-wins), never an unbounded OOM.
        let hub = MemHub::new();
        let mut a = hub.register(A, 64);
        let mut b = hub.register_bounded(B, 64, 4);
        for n in 0..32u8 {
            a.send(B, MsgClass::Snapshot, vec![n].into())
                .expect("accepted");
            hub.pump();
        }
        let got = b.drain_inbound();
        assert!(got.len() <= 4);
        assert_eq!(
            *got.last().expect("some survived"),
            Inbound::Wire {
                from: A,
                class: MsgClass::Snapshot,
                bytes: vec![31].into(),
            },
            "the newest snapshot always survives"
        );
    }

    #[test]
    fn a_reliable_overflow_drops_the_newcomer_and_is_surfaced() {
        // The design's explicit overload ALERT (audit ROB-2): an inbox FULL of reliable
        // traffic drops the incoming RELIABLE event (warned at the pump site + counted by
        // the inbox). The earlier reliable events survive; the newcomer is the casualty.
        let hub = MemHub::new();
        let mut a = hub.register(A, 64);
        let mut b = hub.register_bounded(B, 64, 1);
        a.send(B, MsgClass::Control, vec![1].into()).expect("sent");
        a.send(B, MsgClass::Control, vec![2].into()).expect("sent");
        hub.pump();
        // The drop is SURFACED (audit ROB-2): the counter records it (and the pump warns).
        assert_eq!(
            b.inbound_dropped_reliable(),
            1,
            "the reliable drop is counted"
        );
        assert_eq!(
            b.drain_inbound(),
            vec![Inbound::Wire {
                from: A,
                class: MsgClass::Control,
                bytes: vec![1].into(),
            }],
            "the first reliable survives; the overflowing newcomer was dropped"
        );
    }

    #[test]
    fn killing_an_unknown_node_is_a_no_op() {
        let hub = MemHub::new();
        let mut a = hub.register(A, 8);
        let mut b = hub.register(B, 8);
        hub.kill(NodeId(999));
        a.send(B, MsgClass::Control, vec![5].into())
            .expect("accepted");
        hub.pump();
        assert_eq!(b.drain_inbound().len(), 1, "traffic unaffected");
    }

    #[test]
    fn virtual_clock_advances_and_shares_state_across_clones() {
        let clock = VirtualClock::new(EpochId(7));
        let view = clock.clone();
        assert_eq!(clock.local_tick(), TickId(0));
        assert_eq!(clock.advance_local(), TickId(1));
        assert_eq!(view.local_tick(), TickId(1), "clones share one timeline");
        clock.set_universe_tick(UniverseTick(40));
        assert_eq!(view.universe_tick(), UniverseTick(40));
        assert_eq!(view.epoch(), EpochId(7));
    }

    #[test]
    fn hub_recovers_from_a_poisoned_lock() {
        // A panic while holding the hub lock (e.g. the duplicate-registration assert)
        // poisons the mutex; the hub deliberately recovers via `into_inner` so one
        // harness assertion failure cannot wedge every other node's transport.
        let hub = MemHub::new();
        let mut a = hub.register(A, 8);
        let mut b = hub.register(B, 8);

        let hub2 = hub.clone();
        let poisoner = std::thread::spawn(move || {
            let _dup = hub2.register(A, 8); // panics while the lock is held
        });
        assert!(poisoner.join().is_err(), "poisoner thread must panic");

        a.send(B, MsgClass::Control, vec![1].into())
            .expect("post-poison send works");
        hub.pump();
        assert_eq!(b.drain_inbound().len(), 1, "post-poison delivery works");
    }

    #[test]
    fn reregister_resets_a_node_for_a_kill_9_rebuild() {
        // D-6: a rebuilt node reclaims its identity + peer connectivity but LOSES its in-flight queues
        // (the dead process's RAM) — only its durable Store survives (held separately by the harness).
        let hub = MemHub::new();
        let mut a = hub.register(A, 8);
        let mut b = hub.register(B, 8);
        a.send(B, MsgClass::Control, vec![1].into())
            .expect("pre-rebuild send accepted into A's outbound");
        // REBUILD A (kill-9): the in-flight outbound frame is dropped with the old process's queue.
        let mut a2 = hub.reregister(A, 8);
        hub.pump();
        assert!(
            b.drain_inbound().is_empty(),
            "the pre-rebuild in-flight send was lost with the dead process"
        );
        // The rebuilt endpoint is reachable + sends anew.
        a2.send(B, MsgClass::Control, vec![2].into())
            .expect("post-rebuild send accepted");
        hub.pump();
        assert_eq!(
            b.drain_inbound().len(),
            1,
            "the rebuilt node sends + delivers"
        );
    }

    // ---- MemStore (D-6): the staged/committed durability seam ----------------------------------

    #[test]
    fn memstore_stages_until_commit_then_survives_a_rebuild() {
        let mut s = MemStore::new();
        let k: &[u8] = b"k1";
        let v: Bytes = vec![1, 2, 3].into();
        assert!(s.is_empty(), "a fresh store is empty (genesis)");
        s.put(k, &v);
        // STAGED, not durable: a crash before commit loses it, so scan (committed-only) sees nothing.
        assert!(s.scan(b"").is_empty(), "an uncommitted put is not durable");
        assert!(
            s.is_empty(),
            "staged writes do not make the store non-empty"
        );
        s.commit();
        assert_eq!(
            s.scan(b""),
            vec![(k.to_vec(), v.clone())],
            "commit made it durable"
        );
        assert!(!s.is_empty(), "a committed store is non-empty (recover)");
        // Shared-handle: a clone sees the same committed log — exactly how a rebuilt node re-attaches.
        let clone = s.clone();
        assert_eq!(
            clone.scan(b""),
            vec![(k.to_vec(), v)],
            "the rebuilt handle sees committed state"
        );
    }

    #[test]
    fn memstore_crash_before_commit_drops_only_the_staged_batch() {
        let mut s = MemStore::new();
        let durable: Bytes = vec![9].into();
        s.put(b"a", &durable);
        s.commit(); // durable
        // A second batch, staged but NOT committed: the crash window.
        s.put(b"b", &(vec![8].into()));
        s.delete(b"a"); // a staged delete in the same window
        // CRASH = never commit; the committed view keeps `a` and never saw `b` or the delete.
        assert_eq!(
            s.scan(b""),
            vec![(b"a".to_vec(), durable)],
            "the staged put + staged delete are both lost on a crash-before-commit",
        );
    }

    #[test]
    fn memstore_commit_applies_puts_and_deletes_and_scan_filters_by_prefix() {
        let mut s = MemStore::new();
        s.put(b"x:1", &(vec![1].into()));
        s.put(b"x:2", &(vec![2].into()));
        s.put(b"y:1", &(vec![3].into()));
        s.commit();
        assert_eq!(
            s.scan(b"x:").len(),
            2,
            "a prefix scan returns exactly one key family"
        );
        assert_eq!(s.scan(b"y:"), vec![(b"y:1".to_vec(), vec![3].into())]);
        assert!(
            s.scan(b"z:").is_empty(),
            "a non-matching prefix returns nothing"
        );
        // delete + commit removes (the None arm of commit).
        s.delete(b"x:1");
        s.commit();
        assert_eq!(
            s.scan(b"x:"),
            vec![(b"x:2".to_vec(), vec![2].into())],
            "the committed delete removed x:1, x:2 remains",
        );
    }

    // ---- RLM Step 1: the MemSpawner twin + MemHub deregister / transport_for ------------------
    use vd_core::realm_path::{RealmKindTag, RealmLevel, RealmPath};

    /// A 1-level `RealmCoord` for `kind`/`seed` (the leaf kind drives `profile_kind`).
    fn spawn_coord(kind: RealmKindTag, seed: u64) -> RealmCoord {
        RealmCoord::from_path(RealmPath::from_levels(vec![RealmLevel::new(kind, seed)]))
            .expect("1-level path has a leaf")
    }

    /// A `MemSpawner` whose mint range starts at `first` (above the hand-picked ids A/B).
    fn spawner(first: u64) -> (MemHub, MemSpawner) {
        let hub = MemHub::new();
        let sp = MemSpawner::new(hub.clone(), NodeId(first), 8);
        (hub, sp)
    }

    #[test]
    fn spawn_mints_fresh_monotone_ids() {
        let (hub, sp) = spawner(100);
        let a = sp
            .spawn_realm(&spawn_coord(RealmKindTag::System, 7), UniverseTick(1))
            .expect("spawn");
        let b = sp
            .spawn_realm(&spawn_coord(RealmKindTag::Planet, 7), UniverseTick(1))
            .expect("spawn");
        let c = sp
            .spawn_realm(&spawn_coord(RealmKindTag::Area, 7), UniverseTick(1))
            .expect("spawn");
        assert_eq!((a, b, c), (NodeId(100), NodeId(101), NodeId(102)));
        assert_eq!(hub.node_count(), 3, "each spawn planted a hub transport");
    }

    #[test]
    fn spawn_derives_profile_from_kind() {
        let (_hub, sp) = spawner(100);
        let id = sp
            .spawn_realm(&spawn_coord(RealmKindTag::System, 7), UniverseTick(1))
            .expect("spawn");
        assert_eq!(sp.live()[&id].0, ProfileKind::System);
    }

    #[test]
    fn spawn_records_at_tick() {
        let (_hub, sp) = spawner(100);
        let id = sp
            .spawn_realm(&spawn_coord(RealmKindTag::System, 7), UniverseTick(42))
            .expect("spawn");
        assert_eq!(sp.live()[&id].1, UniverseTick(42));
    }

    #[test]
    fn kill_unknown_errors() {
        let (_hub, sp) = spawner(100);
        assert_eq!(
            sp.kill_realm(NodeId(999)).expect_err("never spawned"),
            SpawnError::UnknownNode(NodeId(999))
        );
    }

    #[test]
    fn kill_then_kill_again_errors() {
        let (_hub, sp) = spawner(100);
        let id = sp
            .spawn_realm(&spawn_coord(RealmKindTag::System, 7), UniverseTick(1))
            .expect("spawn");
        assert_eq!(sp.kill_realm(id), Ok(()));
        assert_eq!(
            sp.kill_realm(id).expect_err("second kill"),
            SpawnError::AlreadyKilled(id)
        );
    }

    #[test]
    fn killed_id_is_never_reminted() {
        let (_hub, sp) = spawner(100);
        let first = sp
            .spawn_realm(&spawn_coord(RealmKindTag::System, 7), UniverseTick(1))
            .expect("spawn");
        assert_eq!(first, NodeId(100));
        sp.kill_realm(first).expect("kill");
        let next = sp
            .spawn_realm(&spawn_coord(RealmKindTag::System, 8), UniverseTick(1))
            .expect("respawn");
        assert_ne!(next, first);
        assert_eq!(
            next,
            NodeId(101),
            "the mint counter never rewinds past a killed id"
        );
    }

    #[test]
    fn kill_deregisters_from_hub_no_accretion() {
        let (hub, sp) = spawner(100);
        let baseline = hub.node_count();
        for _ in 0..5 {
            let id = sp
                .spawn_realm(&spawn_coord(RealmKindTag::System, 7), UniverseTick(1))
                .expect("spawn");
            assert_eq!(hub.node_count(), baseline + 1);
            sp.kill_realm(id).expect("kill");
            assert_eq!(
                hub.node_count(),
                baseline,
                "kill deregistered the node — no accretion"
            );
        }
    }

    #[test]
    fn deregister_unknown_is_noop() {
        let hub = MemHub::new();
        hub.deregister(NodeId(777)); // the absent-key branch — must not panic
        assert_eq!(hub.node_count(), 0);
    }

    #[test]
    fn transport_for_returns_a_handle_for_registered_else_none() {
        let hub = MemHub::new();
        let _t = hub.register(A, 8);
        assert!(
            hub.transport_for(A).is_some(),
            "registered node has a handle"
        );
        assert!(
            hub.transport_for(B).is_none(),
            "unregistered node has no handle"
        );
    }

    #[test]
    fn mem_spawner_recovers_from_poisoned_lock() {
        let (_hub, sp) = spawner(100);
        let sp2 = sp.clone();
        let poisoner = std::thread::spawn(move || {
            let _g = sp2.lock();
            panic!("poison the spawner lock");
        });
        assert!(poisoner.join().is_err(), "poisoner thread must panic");
        // The next op recovers via `into_inner` — one panic cannot wedge the spawner.
        let id = sp
            .spawn_realm(&spawn_coord(RealmKindTag::System, 7), UniverseTick(1))
            .expect("post-poison spawn");
        assert_eq!(id, NodeId(100));
    }

    #[test]
    fn spawn_error_display() {
        assert_eq!(
            SpawnError::UnknownNode(NodeId(5)).to_string(),
            "unknown realm node: node-5"
        );
        assert_eq!(
            SpawnError::AlreadyKilled(NodeId(5)).to_string(),
            "realm node already killed: node-5"
        );
    }
}
