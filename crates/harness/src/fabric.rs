//! The `FaultFabric`: deterministic, seed-driven message delivery with first-class
//! faults (`docs/design/test_harness.md` §3).
//!
//! Contract (binding for everything built on it):
//! - AT-LEAST-ONCE: "delivered" means the receiver SURVIVED the step that drained
//!   the message (the topology acks on its behalf). An unacked message is redelivered
//!   after `retry_delay_ticks` — so a crash with a message sitting in the inbound
//!   queue forces redelivery, never silent loss (the mechanical guard for R1/R9).
//! - Faults are deterministic draws from the fabric's seeded RNG: a seed reproduces
//!   the exact fault tape, which is what makes chaos failures replayable.
//! - `drop` perturbs ONE delivery attempt; eventual delivery still holds through
//!   redelivery (loss-forever exists only via `kill` — permanent death — which
//!   surfaces as `NodeUnreachable` to senders).
//! - CONSERVATION: every accepted send is at all times exactly one of {acked,
//!   in flight, awaiting redelivery, returned unreachable} — asserted by tests and
//!   the oracle every step; nothing can leak.

use std::collections::{BTreeMap, VecDeque};

use vd_core::rng::SplitMix64;
use vd_core::{MsgId, NodeId, TickId};
use vd_sim::io::{Bytes, Inbound, MsgClass, SendError, Transport};

/// Per-link fault policy. Default: a perfect link (no faults, next-tick delivery).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct LinkPolicy {
    /// Probability one delivery ATTEMPT is dropped (redelivery still happens).
    pub drop_p: f64,
    /// Probability an attempt is duplicated (receivers must dedup at their layer).
    pub dup_p: f64,
    /// Extra delivery delay drawn uniformly from `[0, max_extra_delay_ticks]`.
    pub max_extra_delay_ticks: u64,
    /// A partitioned link delivers nothing (attempts requeue until healed).
    pub partitioned: bool,
    /// Probability `Transport::send` synchronously refuses (`QueueFull`) —
    /// the deterministic trigger for the back-pressure code path.
    pub send_reject_p: f64,
}

impl Default for LinkPolicy {
    fn default() -> LinkPolicy {
        LinkPolicy {
            drop_p: 0.0,
            dup_p: 0.0,
            max_extra_delay_ticks: 0,
            partitioned: false,
            send_reject_p: 0.0,
        }
    }
}

/// When, within a topology step, a scheduled crash fires
/// (the third matrix dimension — `docs/design/test_harness.md` §2).
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum CrashWhen {
    /// Before messages are injected: the node misses this tick's deliveries.
    PreInject,
    /// After injection, before stepping: deliveries sit UNACKED in its inbound —
    /// they MUST redeliver after resurrection.
    PostInject,
    /// After stepping: the node processed the tick but its effects-in-queues
    /// survive; subsequent ticks see it dead.
    PostStep,
}

/// A message the fabric is responsible for.
#[derive(Clone, Debug)]
struct Tracked {
    from: NodeId,
    to: NodeId,
    class: MsgClass,
    bytes: Bytes,
    msg_id: MsgId,
    /// Global, monotonically increasing send sequence (stable ordering key).
    global_seq: u64,
    kind: TrackedKind,
}

/// Typed payload kind — never smuggled through payload bytes (an in-band tag would
/// collide with user payloads that happen to start with it; found by coverage).
#[derive(Clone, Copy, Debug)]
enum TrackedKind {
    Wire,
    /// An unreachable notice bounced to the sender: `to` here is the DEAD node.
    Notice {
        dead: NodeId,
        undelivered: MsgId,
    },
}

#[derive(Debug, Default)]
struct NodeEndpoint {
    /// Released deliveries awaiting drain (and then ack-on-survival).
    inbound: VecDeque<Tracked>,
    /// Receiver buffer bound: a delivery that would overflow it is held in the
    /// unacked ledger and redelivered later (receiver back-pressure — the
    /// deterministic analog of a full OS socket buffer; 0 = unbounded). This
    /// NEVER drops a message (conservation holds): it just defers delivery.
    inbound_capacity: usize,
    /// Drained this step; acked when the node survives the step.
    drained_pending_ack: Vec<u64>,
    next_msg_id: u64,
    /// Crashed nodes neither send nor receive (until resurrected).
    crashed: bool,
    /// Killed nodes are permanently gone: sends toward them bounce as unreachable.
    killed: bool,
    /// WIRE TRUTH accounting: everything ever pushed into this inbound...
    delivered_total: u64,
    /// ...everything the node actually drained...
    drained_total: u64,
    /// ...and everything a crash wiped undrained.
    cleared_by_crash: u64,
}

/// Per-node delivery accounting — the ground truth the WireMonitor checks node
/// CLAIMS against (`delivered == drained + cleared + pending`, exactly).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct DeliveryAccounting {
    pub delivered: u64,
    pub drained: u64,
    pub cleared_by_crash: u64,
    pub pending: u64,
}

/// Counters proving conservation (and feeding the oracle).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct FabricStats {
    pub accepted: u64,
    pub acked: u64,
    pub unreachable_returned: u64,
    pub send_rejected: u64,
    pub attempts_dropped: u64,
    pub attempts_duplicated: u64,
    /// Unreachable notices that could not be returned because the SENDER was
    /// crashed at bounce time (counted, never silent).
    pub notices_missed_by_crashed_sender: u64,
    /// Delivery attempts deferred because the receiver buffer was full (receiver
    /// back-pressure — held in the unacked ledger, redelivered later, never lost).
    pub receiver_buffer_full: u64,
}

#[derive(Debug)]
struct FabricInner {
    endpoints: BTreeMap<NodeId, NodeEndpoint>,
    policies: BTreeMap<(NodeId, NodeId), LinkPolicy>,
    /// (deliver_tick, global_seq) -> attempt. BTreeMap = deterministic release order.
    inflight: BTreeMap<(TickId, u64), Tracked>,
    /// Accepted, not yet acked: the at-least-once ledger (keyed by global_seq).
    unacked: BTreeMap<u64, Tracked>,
    /// global_seq -> when to redeliver if still unacked.
    redeliver_at: BTreeMap<u64, TickId>,
    rng: SplitMix64,
    now: TickId,
    next_global_seq: u64,
    retry_delay_ticks: u64,
    stats: FabricStats,
}

/// The fabric: cloneable handle (one shared instance per topology).
#[derive(Clone, Debug)]
pub struct FaultFabric {
    inner: std::sync::Arc<std::sync::Mutex<FabricInner>>,
}

impl FaultFabric {
    #[must_use]
    pub fn new(seed: u64, retry_delay_ticks: u64) -> FaultFabric {
        FaultFabric {
            inner: std::sync::Arc::new(std::sync::Mutex::new(FabricInner {
                endpoints: BTreeMap::new(),
                policies: BTreeMap::new(),
                inflight: BTreeMap::new(),
                unacked: BTreeMap::new(),
                redeliver_at: BTreeMap::new(),
                rng: SplitMix64::new(seed),
                now: TickId(0),
                next_global_seq: 0,
                retry_delay_ticks: retry_delay_ticks.max(1),
                stats: FabricStats::default(),
            })),
        }
    }

    fn lock(&self) -> std::sync::MutexGuard<'_, FabricInner> {
        self.inner
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
    }

    /// Register a node endpoint.
    ///
    /// # Panics
    /// On duplicate registration (a harness bug, not a runtime condition).
    #[must_use]
    pub fn register(&self, id: NodeId) -> FabricTransport {
        self.register_bounded(id, 0)
    }

    /// Register with a receiver inbound bound (0 = unbounded). A bounded receiver
    /// applies back-pressure: deliveries that would overflow are deferred and
    /// redelivered, never dropped — the deterministic inbound-overflow scenario.
    ///
    /// # Panics
    /// On duplicate registration (a harness bug, not a runtime condition).
    #[must_use]
    pub fn register_bounded(&self, id: NodeId, inbound_capacity: usize) -> FabricTransport {
        let mut inner = self.lock();
        assert!(
            !inner.endpoints.contains_key(&id),
            "duplicate fabric registration: {id}"
        );
        inner.endpoints.insert(
            id,
            NodeEndpoint {
                inbound_capacity,
                ..NodeEndpoint::default()
            },
        );
        FabricTransport {
            local: id,
            fabric: self.clone(),
        }
    }

    /// Set the policy for the directed link `from -> to`.
    pub fn set_policy(&self, from: NodeId, to: NodeId, policy: LinkPolicy) {
        self.lock().policies.insert((from, to), policy);
    }

    /// Permanently kill a node: pending and future traffic toward it bounces back
    /// to senders as `NodeUnreachable`.
    pub fn kill(&self, id: NodeId) {
        let mut inner = self.lock();
        if let Some(ep) = inner.endpoints.get_mut(&id) {
            ep.killed = true;
            ep.crashed = true;
        }
    }

    /// Crash a node (volatile; resurrectable). The topology decides WHEN within a
    /// step this fires (`CrashWhen`).
    pub fn crash(&self, id: NodeId) {
        let mut inner = self.lock();
        if let Some(ep) = inner.endpoints.get_mut(&id) {
            ep.crashed = true;
            // Whatever sat undrained or un-acked in its inbound is NOT lost: the
            // unacked ledger redelivers it after resurrection.
            ep.cleared_by_crash += ep.inbound.len() as u64;
            ep.inbound.clear();
            ep.drained_pending_ack.clear();
        }
    }

    /// Bring a crashed (not killed) node back.
    pub fn resurrect(&self, id: NodeId) {
        let mut inner = self.lock();
        if let Some(ep) = inner.endpoints.get_mut(&id)
            && !ep.killed
        {
            ep.crashed = false;
        }
    }

    #[must_use]
    pub fn is_crashed(&self, id: NodeId) -> bool {
        self.lock().endpoints.get(&id).is_some_and(|e| e.crashed)
    }

    /// Is this node currently DOWN — killed permanently OR crashed and not yet resurrected (i.e. not
    /// delivering, not authoritative)? The dead-node-aware oracle filter (P3 crash matrix) excludes a
    /// dead node's authority claim so a corpse cannot FALSE-PASS AUTHORITY-UNIQUE (held@a dead node)
    /// nor a legitimate mid-flight park FALSE-RED. NB `kill` sets `crashed` too, so this is equivalent
    /// to `is_crashed`; spelled `killed || crashed` to document the intent at the call site.
    #[must_use]
    pub fn is_dead(&self, id: NodeId) -> bool {
        self.lock()
            .endpoints
            .get(&id)
            .is_some_and(|e| e.killed || e.crashed)
    }

    #[must_use]
    pub fn stats(&self) -> FabricStats {
        self.lock().stats
    }

    /// Advance the fabric one tick: release due attempts (policy-faulted,
    /// deterministically), schedule redeliveries for overdue unacked messages.
    /// Returns the number of deliveries released into inbound queues this tick.
    pub fn pump(&self, now: TickId) -> usize {
        let mut inner = self.lock();
        inner.now = now;

        // 1. Schedule redeliveries for unacked messages whose retry came due
        //    (deadlines for already-acked messages are simply discarded here).
        let due_retries: Vec<u64> = inner
            .redeliver_at
            .iter()
            .filter(|(_, at)| **at <= now)
            .map(|(seq, _)| *seq)
            .collect();
        for seq in due_retries {
            inner.redeliver_at.remove(&seq);
            if let Some(msg) = inner.unacked.get(&seq).cloned() {
                inner.schedule_attempt(msg, now);
            }
        }

        // 2. Release every attempt due at or before `now`, in (tick, seq) order.
        let mut released = 0usize;
        loop {
            let key = match inner.inflight.first_key_value() {
                Some((k, _)) if k.0 <= now => *k,
                _ => break,
            };
            let msg = inner
                .inflight
                .remove(&key)
                .expect("first key just observed under the same lock");
            released += inner.deliver_attempt(msg, now);
        }
        released
    }
}

impl FabricInner {
    /// Queue one delivery attempt, applying delay policy.
    fn schedule_attempt(&mut self, msg: Tracked, now: TickId) {
        let policy = self
            .policies
            .get(&(msg.from, msg.to))
            .copied()
            .unwrap_or_default();
        let extra = if policy.max_extra_delay_ticks > 0 {
            self.rng.range_u64(0, policy.max_extra_delay_ticks + 1)
        } else {
            0
        };
        let deliver_at = TickId(now.0 + 1 + extra);
        self.inflight.insert((deliver_at, msg.global_seq), msg);
    }

    /// Attempt delivery now: apply partition/drop/dup/death policies. Returns how
    /// many copies landed in the receiver's inbound.
    fn deliver_attempt(&mut self, msg: Tracked, now: TickId) -> usize {
        let policy = self
            .policies
            .get(&(msg.from, msg.to))
            .copied()
            .unwrap_or_default();
        let seq = msg.global_seq;

        let receiver_killed = self.endpoints.get(&msg.to).is_none_or(|e| e.killed);
        if receiver_killed {
            // Permanent death: bounce as NodeUnreachable to the sender (if alive;
            // a crashed sender just misses the notice — counted, not silent).
            self.unacked.remove(&seq);
            self.redeliver_at.remove(&seq);
            self.stats.unreachable_returned += 1;
            let sender_alive = self
                .endpoints
                .get(&msg.from)
                .is_some_and(|sender| !sender.crashed);
            if sender_alive {
                let sender = self
                    .endpoints
                    .get_mut(&msg.from)
                    .expect("sender liveness just checked");
                sender.delivered_total += 1;
                sender.inbound.push_back(Tracked {
                    from: msg.to,
                    to: msg.from,
                    class: msg.class,
                    bytes: vd_sim::io::bytes(Vec::new()),
                    msg_id: msg.msg_id,
                    global_seq: seq,
                    kind: TrackedKind::Notice {
                        dead: msg.to,
                        undelivered: msg.msg_id,
                    },
                });
            } else {
                self.stats.notices_missed_by_crashed_sender += 1;
            }
            return 0;
        }

        let receiver_crashed = self.endpoints.get(&msg.to).is_none_or(|e| e.crashed);
        let blocked = policy.partitioned || receiver_crashed;
        let dropped = !blocked && self.rng.chance(policy.drop_p);
        if blocked || dropped {
            if dropped {
                self.stats.attempts_dropped += 1;
            }
            // Not delivered: leave in the unacked ledger; retry later.
            self.redeliver_at
                .insert(seq, TickId(now.0 + self.retry_delay_ticks));
            return 0;
        }

        // Receiver back-pressure: a full inbound buffer defers the delivery (held in
        // the unacked ledger, redelivered later) — the deterministic analog of a
        // full OS socket buffer. Never drops: conservation/wire-truth hold.
        let receiver_cap = self
            .endpoints
            .get(&msg.to)
            .map_or(0, |e| e.inbound_capacity);
        let receiver_len = self.endpoints.get(&msg.to).map_or(0, |e| e.inbound.len());
        if receiver_cap > 0 && receiver_len >= receiver_cap {
            self.stats.receiver_buffer_full += 1;
            self.redeliver_at
                .insert(seq, TickId(now.0 + self.retry_delay_ticks));
            return 0;
        }

        let copies = if self.rng.chance(policy.dup_p) {
            self.stats.attempts_duplicated += 1;
            2
        } else {
            1
        };
        // Delivered-but-not-acked still carries a redelivery deadline: if the
        // receiver crashes with this sitting undrained (PostInject), the retry
        // fires after resurrection — delivery is only final at ack_survivor.
        self.redeliver_at
            .insert(seq, TickId(now.0 + self.retry_delay_ticks));
        for _ in 0..copies {
            let receiver = self
                .endpoints
                .get_mut(&msg.to)
                .expect("receiver verified alive above");
            receiver.delivered_total += 1;
            receiver.inbound.push_back(msg.clone());
        }
        copies
    }
}

/// One node's transport endpoint into the fabric.
#[derive(Debug)]
pub struct FabricTransport {
    local: NodeId,
    fabric: FaultFabric,
}

impl Transport for FabricTransport {
    fn send(&mut self, to: NodeId, class: MsgClass, bytes: Bytes) -> Result<MsgId, SendError> {
        let mut inner = self.fabric.lock();
        let policy = inner
            .policies
            .get(&(self.local, to))
            .copied()
            .unwrap_or_default();
        if inner.rng.chance(policy.send_reject_p) {
            inner.stats.send_rejected += 1;
            return Err(SendError::QueueFull(bytes));
        }
        let endpoint = inner
            .endpoints
            .get_mut(&self.local)
            .expect("registered endpoint present");
        let msg_id = MsgId(endpoint.next_msg_id);
        endpoint.next_msg_id += 1;
        let global_seq = inner.next_global_seq;
        inner.next_global_seq += 1;
        let msg = Tracked {
            from: self.local,
            to,
            class,
            bytes,
            msg_id,
            global_seq,
            kind: TrackedKind::Wire,
        };
        inner.stats.accepted += 1;
        inner.unacked.insert(global_seq, msg.clone());
        let now = inner.now;
        inner.schedule_attempt(msg, now);
        Ok(msg_id)
    }

    fn drain_inbound(&mut self) -> Vec<Inbound> {
        let mut inner = self.fabric.lock();
        let endpoint = inner
            .endpoints
            .get_mut(&self.local)
            .expect("transport endpoints are registered for life");
        let drained: Vec<Tracked> = endpoint.inbound.drain(..).collect();
        endpoint.drained_total += drained.len() as u64;
        let mut out = Vec::with_capacity(drained.len());
        let mut acks = Vec::new();
        for t in drained {
            acks.push(t.global_seq);
            out.push(decode_tracked(t));
        }
        endpoint.drained_pending_ack.extend(acks);
        out
    }

    fn local_id(&self) -> NodeId {
        self.local
    }
}

/// Reconstruct the typed Inbound from a tracked record.
fn decode_tracked(t: Tracked) -> Inbound {
    match t.kind {
        TrackedKind::Wire => Inbound::Wire {
            from: t.from,
            class: t.class,
            bytes: t.bytes,
        },
        TrackedKind::Notice { dead, undelivered } => Inbound::NodeUnreachable {
            to: dead,
            class: t.class,
            undelivered,
        },
    }
}

impl FaultFabric {
    /// The topology calls this after a node SURVIVED its step: everything it
    /// drained this step becomes durably acked (the at-least-once commit point).
    pub fn ack_survivor(&self, id: NodeId) {
        let mut inner = self.lock();
        let endpoint = inner
            .endpoints
            .get_mut(&id)
            .expect("ack target is a registered endpoint");
        let acks = std::mem::take(&mut endpoint.drained_pending_ack);
        for seq in acks {
            if inner.unacked.remove(&seq).is_some() {
                inner.stats.acked += 1;
            }
            // The redeliver_at deadline is left in place: the pump's due-retry scan
            // finds the seq acked (absent from `unacked`) and discards the deadline —
            // one cleanup path instead of two racing ones.
        }
    }

    /// Ground-truth delivery accounting for one node (the WireMonitor's source).
    #[must_use]
    pub fn delivery_accounting(&self, id: NodeId) -> DeliveryAccounting {
        let inner = self.lock();
        inner
            .endpoints
            .get(&id)
            .map(|ep| DeliveryAccounting {
                delivered: ep.delivered_total,
                drained: ep.drained_total,
                cleared_by_crash: ep.cleared_by_crash,
                pending: ep.inbound.len() as u64,
            })
            .unwrap_or_default()
    }

    /// CONSERVATION: accepted == acked + unreachable + outstanding (in flight or
    /// awaiting retry or sitting undrained/unacked in an inbound). Checked by the
    /// oracle every step; a leak is a fabric bug, loudly.
    #[must_use]
    pub fn conservation_holds(&self) -> bool {
        let inner = self.lock();
        let outstanding = inner.unacked.len() as u64;
        inner.stats.accepted == inner.stats.acked + inner.stats.unreachable_returned + outstanding
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const A: NodeId = NodeId(1);
    const B: NodeId = NodeId(2);

    fn perfect_pair() -> (FaultFabric, FabricTransport, FabricTransport) {
        let fabric = FaultFabric::new(11, 2);
        let a = fabric.register(A);
        let b = fabric.register(B);
        (fabric, a, b)
    }

    #[test]
    fn perfect_link_delivers_next_tick_and_acks_on_survival() {
        let (fabric, mut a, mut b) = perfect_pair();
        a.send(B, MsgClass::Control, vec![7].into())
            .expect("accepted");
        assert_eq!(fabric.pump(TickId(1)), 1);
        let got = b.drain_inbound();
        assert_eq!(
            got,
            vec![Inbound::Wire {
                from: A,
                class: MsgClass::Control,
                bytes: vec![7].into()
            }]
        );
        fabric.ack_survivor(B);
        assert!(fabric.conservation_holds());
        assert_eq!(fabric.stats().acked, 1);
        // The acked message's redelivery deadline is discarded when it comes due —
        // no ghost retry of something already final.
        assert_eq!(fabric.pump(TickId(3)), 0);
        assert!(b.drain_inbound().is_empty());
    }

    #[test]
    fn crashed_sender_misses_the_notice_counted_never_silent() {
        // B is permanently dead AND A is crashed when the bounce fires: the notice
        // has nowhere to go — it is counted, and conservation still holds.
        let (fabric, mut a, _b) = perfect_pair();
        a.send(B, MsgClass::Saga, vec![1].into()).expect("accepted");
        fabric.kill(B);
        fabric.crash(A);
        fabric.pump(TickId(1));
        assert_eq!(fabric.stats().notices_missed_by_crashed_sender, 1);
        assert_eq!(fabric.stats().unreachable_returned, 1);
        assert!(fabric.conservation_holds());
        fabric.resurrect(A);
        assert!(a.drain_inbound().is_empty(), "the notice was never queued");
    }

    #[test]
    fn unacked_messages_redeliver_after_a_post_inject_crash() {
        // THE at-least-once guarantee: B crashes WITH the message in its inbound
        // (PostInject); after resurrection the message arrives again.
        let (fabric, mut a, mut b) = perfect_pair();
        a.send(B, MsgClass::Saga, vec![9].into()).expect("accepted");
        fabric.pump(TickId(1));
        // Delivered into B's inbound, but B crashes before stepping/acking.
        fabric.crash(B);
        assert!(fabric.is_crashed(B));
        assert!(b.drain_inbound().is_empty(), "crashed node has nothing");
        // Retry comes due; B resurrects; the message arrives again.
        fabric.resurrect(B);
        let mut redelivered = Vec::new();
        for t in 2..8 {
            fabric.pump(TickId(t));
            redelivered.extend(b.drain_inbound());
            if !redelivered.is_empty() {
                break;
            }
        }
        assert_eq!(
            redelivered,
            vec![Inbound::Wire {
                from: A,
                class: MsgClass::Saga,
                bytes: vec![9].into()
            }]
        );
        fabric.ack_survivor(B);
        assert!(fabric.conservation_holds());
    }

    #[test]
    fn is_dead_tracks_killed_crashed_and_resurrected() {
        // The dead-node-aware-oracle predicate (P3 crash matrix): dead while killed (permanent),
        // dead while crashed, NOT dead after resurrect. Both `killed || crashed` operands covered.
        let (fabric, _a, _b) = perfect_pair();
        assert!(!fabric.is_dead(A), "a live node is not dead");
        fabric.kill(A);
        assert!(fabric.is_dead(A), "a killed node is dead (permanent)");
        fabric.crash(B);
        assert!(fabric.is_dead(B), "a crashed node is dead");
        fabric.resurrect(B);
        assert!(
            !fabric.is_dead(B),
            "a resurrected (not-killed) node is no longer dead"
        );
        // A killed node STAYS dead through a resurrect attempt (kill is permanent).
        fabric.resurrect(A);
        assert!(fabric.is_dead(A), "resurrect does not revive a killed node");
    }

    #[test]
    fn killed_receiver_bounces_unreachable_to_the_sender() {
        let (fabric, mut a, _b) = perfect_pair();
        let sent = a.send(B, MsgClass::Saga, vec![1].into()).expect("accepted");
        fabric.kill(B);
        fabric.pump(TickId(1));
        let notices = a.drain_inbound();
        assert_eq!(
            notices,
            vec![Inbound::NodeUnreachable {
                to: B,
                class: MsgClass::Saga,
                undelivered: sent
            }]
        );
        fabric.ack_survivor(A);
        assert!(fabric.conservation_holds());
        assert_eq!(fabric.stats().unreachable_returned, 1);
    }

    #[test]
    fn partition_holds_messages_until_healed() {
        let (fabric, mut a, mut b) = perfect_pair();
        fabric.set_policy(
            A,
            B,
            LinkPolicy {
                partitioned: true,
                ..LinkPolicy::default()
            },
        );
        a.send(B, MsgClass::Control, vec![3].into())
            .expect("accepted");
        for t in 1..6 {
            fabric.pump(TickId(t));
            assert!(b.drain_inbound().is_empty(), "partitioned at tick {t}");
        }
        // Heal: delivery resumes via redelivery.
        fabric.set_policy(A, B, LinkPolicy::default());
        let mut got = Vec::new();
        for t in 6..12 {
            fabric.pump(TickId(t));
            got.extend(b.drain_inbound());
            if !got.is_empty() {
                break;
            }
        }
        assert_eq!(got.len(), 1, "healed partition delivers");
        assert!(fabric.conservation_holds());
    }

    #[test]
    fn drop_policy_loses_attempts_never_messages() {
        let (fabric, mut a, mut b) = perfect_pair();
        fabric.set_policy(
            A,
            B,
            LinkPolicy {
                drop_p: 0.7,
                ..LinkPolicy::default()
            },
        );
        a.send(B, MsgClass::Input, vec![5].into())
            .expect("accepted");
        let mut got = Vec::new();
        for t in 1..64 {
            fabric.pump(TickId(t));
            got.extend(b.drain_inbound());
            if !got.is_empty() {
                break;
            }
        }
        assert_eq!(got.len(), 1, "eventual delivery through redelivery");
        assert!(
            fabric.stats().attempts_dropped > 0,
            "drops actually happened"
        );
        assert!(fabric.conservation_holds());
    }

    #[test]
    fn duplication_delivers_extra_copies() {
        let (fabric, mut a, mut b) = perfect_pair();
        fabric.set_policy(
            A,
            B,
            LinkPolicy {
                dup_p: 1.0,
                ..LinkPolicy::default()
            },
        );
        a.send(B, MsgClass::Control, vec![8].into())
            .expect("accepted");
        fabric.pump(TickId(1));
        assert_eq!(b.drain_inbound().len(), 2, "dup_p=1.0 doubles the attempt");
        assert_eq!(fabric.stats().attempts_duplicated, 1);
    }

    #[test]
    fn send_reject_policy_backpressures_synchronously() {
        let (fabric, mut a, _b) = perfect_pair();
        fabric.set_policy(
            A,
            B,
            LinkPolicy {
                send_reject_p: 1.0,
                ..LinkPolicy::default()
            },
        );
        let err = a
            .send(B, MsgClass::Input, vec![4].into())
            .expect_err("rejected");
        assert_eq!(
            err,
            SendError::QueueFull(vec![4].into()),
            "payload returned"
        );
        assert_eq!(fabric.stats().send_rejected, 1);
        assert!(
            fabric.conservation_holds(),
            "a rejection never enters the ledger"
        );
    }

    #[test]
    fn delay_policy_postpones_delivery_deterministically() {
        let (fabric, mut a, mut b) = perfect_pair();
        fabric.set_policy(
            A,
            B,
            LinkPolicy {
                max_extra_delay_ticks: 5,
                ..LinkPolicy::default()
            },
        );
        a.send(B, MsgClass::Control, vec![6].into())
            .expect("accepted");
        let mut arrival = None;
        for t in 1..10 {
            fabric.pump(TickId(t));
            if !b.drain_inbound().is_empty() {
                arrival = Some(t);
                break;
            }
        }
        let t = arrival.expect("delivered");
        assert!((1..=6).contains(&t), "within 1 + max extra delay");
    }

    #[test]
    fn same_seed_same_fault_tape() {
        let run = |seed: u64| -> (u64, u64) {
            let fabric = FaultFabric::new(seed, 2);
            let mut a = fabric.register(A);
            let mut b = fabric.register(B);
            fabric.set_policy(
                A,
                B,
                LinkPolicy {
                    drop_p: 0.5,
                    dup_p: 0.3,
                    max_extra_delay_ticks: 3,
                    ..LinkPolicy::default()
                },
            );
            for n in 0..16u8 {
                let _ = a.send(B, MsgClass::Input, vec![n].into());
            }
            let mut delivered = 0u64;
            for t in 1..32 {
                fabric.pump(TickId(t));
                delivered += b.drain_inbound().len() as u64;
                fabric.ack_survivor(B);
            }
            (delivered, fabric.stats().attempts_dropped)
        };
        assert_eq!(run(99), run(99), "identical fault tape from one seed");
    }

    #[test]
    fn a_bounded_receiver_defers_delivery_without_loss() {
        // B's inbound holds at most 2; A floods 5 — the overflow is held in the
        // unacked ledger and redelivered, NEVER dropped (conservation holds).
        let fabric = FaultFabric::new(7, 1);
        let mut a = fabric.register(A);
        let mut b = fabric.register_bounded(B, 2);
        for n in 0..5u8 {
            a.send(B, MsgClass::Snapshot, vec![n].into())
                .expect("accepted");
        }
        // First pump delivers up to the buffer cap; the rest defer.
        fabric.pump(TickId(1));
        let first = b.drain_inbound();
        assert!(first.len() <= 2);
        assert!(
            fabric.stats().receiver_buffer_full > 0,
            "back-pressure was exercised"
        );
        fabric.ack_survivor(B);
        // Drain over many ticks: every one of the 5 eventually arrives (no loss).
        let mut all = first;
        for t in 2..40 {
            fabric.pump(TickId(t));
            all.extend(b.drain_inbound());
            fabric.ack_survivor(B);
        }
        // No faults, no dup, no loss: exactly the 5 sent messages arrive (the
        // deferred ones via redelivery). All are Wire (this link never kills).
        assert_eq!(all.len(), 5, "all delivered, none lost, none duplicated");
        assert!(fabric.conservation_holds());
    }

    #[test]
    fn resurrect_does_not_revive_the_killed_and_unknowns_are_noops() {
        let (fabric, _a, _b) = perfect_pair();
        fabric.kill(B);
        fabric.resurrect(B);
        assert!(fabric.is_crashed(B), "killed is permanent");
        fabric.crash(NodeId(99));
        fabric.resurrect(NodeId(99));
        fabric.kill(NodeId(99));
        assert!(!fabric.is_crashed(NodeId(99)), "unknown nodes are no-ops");
    }

    #[test]
    #[should_panic(expected = "duplicate fabric registration")]
    fn duplicate_registration_panics() {
        let fabric = FaultFabric::new(1, 1);
        let _x = fabric.register(A);
        let _y = fabric.register(A);
    }
}
