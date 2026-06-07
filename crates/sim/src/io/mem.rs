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

use std::collections::{BTreeMap, VecDeque};
use std::sync::{Arc, Mutex, MutexGuard, PoisonError};

use vd_core::{MsgId, NodeId};

use super::{Bytes, Inbound, MsgClass, SendError, Transport};

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
    /// Delivered messages awaiting `drain_inbound`.
    inbound: VecDeque<Inbound>,
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
                inbound: VecDeque::new(),
                next_msg_id: 0,
                alive: true,
            },
        );
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
                inner
                    .nodes
                    .get_mut(&target)
                    .expect("pump target exists: receiver checked alive, or sender owns the frame")
                    .inbound
                    .push_back(event);
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
    fn send(&mut self, to: NodeId, class: MsgClass, bytes: Bytes) -> Result<MsgId, SendError> {
        let mut inner = self.hub.lock();
        let node = inner
            .nodes
            .get_mut(&self.local)
            .expect("registered node present in hub");
        if node.outbound.len() >= node.outbound_capacity {
            return Err(SendError::QueueFull);
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
            Some(node) => node.inbound.drain(..).collect(),
            None => Vec::new(),
        }
    }

    fn local_id(&self) -> NodeId {
        self.local
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

        a.send(B, MsgClass::Control, vec![1]).expect("accepted");
        assert!(b.drain_inbound().is_empty(), "no delivery before pump");

        hub.pump();
        let got = b.drain_inbound();
        assert_eq!(
            got,
            vec![Inbound::Wire {
                from: A,
                class: MsgClass::Control,
                bytes: vec![1],
            }]
        );
    }

    #[test]
    fn bounded_queue_backpressures_exactly_at_capacity() {
        let hub = MemHub::new();
        let mut a = hub.register(A, 4);
        let _b = hub.register(B, 4);

        let results: Vec<Result<MsgId, SendError>> = (0..6)
            .map(|n| a.send(B, MsgClass::Input, vec![n]))
            .collect();
        let accepted = results.iter().filter(|r| r.is_ok()).count();
        let rejected = results.iter().filter(|r| r.is_err()).count();
        assert_eq!((accepted, rejected), (4, 2));
        assert_eq!(results[4], Err(SendError::QueueFull));
    }

    #[test]
    fn fifo_order_and_fifo_msg_ids() {
        let hub = MemHub::new();
        let mut a = hub.register(A, 8);
        let mut b = hub.register(B, 8);
        assert_eq!((a.local_id(), b.local_id()), (A, B));

        let id0 = a.send(B, MsgClass::Control, vec![0]).expect("accepted");
        let id1 = a.send(B, MsgClass::Control, vec![1]).expect("accepted");
        assert_eq!((id0, id1), (MsgId(0), MsgId(1)));

        hub.pump();
        assert_eq!(
            b.drain_inbound(),
            vec![
                Inbound::Wire {
                    from: A,
                    class: MsgClass::Control,
                    bytes: vec![0],
                },
                Inbound::Wire {
                    from: A,
                    class: MsgClass::Control,
                    bytes: vec![1],
                },
            ]
        );
    }

    #[test]
    fn dead_peer_surfaces_node_unreachable_with_send_correlation() {
        let hub = MemHub::new();
        let mut a = hub.register(A, 8);
        let _b = hub.register(B, 8);

        let sent = a.send(B, MsgClass::Saga, vec![7]).expect("accepted");
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
    fn killing_an_unknown_node_is_a_no_op() {
        let hub = MemHub::new();
        let mut a = hub.register(A, 8);
        let mut b = hub.register(B, 8);
        hub.kill(NodeId(999));
        a.send(B, MsgClass::Control, vec![5]).expect("accepted");
        hub.pump();
        assert_eq!(b.drain_inbound().len(), 1, "traffic unaffected");
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

        a.send(B, MsgClass::Control, vec![1])
            .expect("post-poison send works");
        hub.pump();
        assert_eq!(b.drain_inbound().len(), 1, "post-poison delivery works");
    }
}
