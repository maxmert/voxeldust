//! SPIKE-0a tracer bullet: the smallest node that proves the threading model.
//!
//! A `TracerNode` is a synchronous `step_tick` state machine over the [`Transport`]
//! seam — no async, no sockets, no clocks. The SAME node, byte for byte, runs over the
//! in-memory hub (deterministic tier) and over the real quinn bridge (`vd-io-prod`),
//! and the SPIKE-0a exit criterion is that both produce identical logical traces.
//!
//! Protocol: a `Pinger` sends `Ping(n)` (one per tick, n strictly increasing, retrying
//! the same n after back-pressure); an `Echo` peer replies `Pong(n)` (buffering replies
//! it could not enqueue). The pinger's trace records pongs, back-pressure, and peer
//! unreachability — the normalized logical output.

use serde::{Deserialize, Serialize};
use vd_core::{NodeId, TickId};
use vd_sim::io::{Inbound, MsgClass, SendError, Transport};

/// Wire payload for the tracer protocol (postcard-encoded).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
enum TracerMsg {
    Ping(u64),
    Pong(u64),
}

/// Normalized logical trace event. Tick timing is deliberately absent: the in-memory
/// and real-network runs differ in latency but MUST agree on this sequence.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum TraceEvent {
    PongReceived(u64),
    /// `Ping(n)` was rejected with `QueueFull`; it will be retried.
    Backpressured(u64),
    /// The transport reported an accepted send as undeliverable.
    PeerUnreachable,
    /// A frame failed to decode — never expected; tests assert its absence.
    DecodeError,
}

pub enum TracerRole {
    /// Sends `total` pings, one per tick.
    Pinger { total: u64 },
    /// Echoes every ping back as a pong.
    Echo,
}

pub struct TracerNode<T: Transport> {
    transport: T,
    peer: NodeId,
    role: TracerRole,
    next_ping: u64,
    /// Echo-side replies that hit back-pressure, retried FIFO each tick.
    pending_replies: std::collections::VecDeque<u64>,
    tick: TickId,
    pub trace: Vec<TraceEvent>,
}

impl<T: Transport> TracerNode<T> {
    pub fn new(transport: T, peer: NodeId, role: TracerRole) -> Self {
        Self {
            transport,
            peer,
            role,
            next_ping: 0,
            pending_replies: std::collections::VecDeque::new(),
            tick: TickId::default(),
            trace: Vec::new(),
        }
    }

    /// Number of pongs received so far (the pinger's progress measure).
    #[must_use]
    pub fn pongs_received(&self) -> usize {
        self.trace
            .iter()
            .filter(|e| matches!(e, TraceEvent::PongReceived(_)))
            .count()
    }

    #[must_use]
    pub fn tick(&self) -> TickId {
        self.tick
    }

    fn send_msg(&mut self, msg: TracerMsg) -> Result<(), SendError> {
        let bytes = postcard::to_allocvec(&msg).expect("tracer messages always encode");
        self.transport
            .send(self.peer, MsgClass::Control, vd_sim::io::bytes(bytes))
            .map(|_| ())
    }

    /// THE deterministic unit of progress: drain inbound, react, emit outbound.
    /// Synchronous by construction — there is nothing here that *could* await.
    pub fn step_tick(&mut self) {
        self.tick = self.tick.next();

        for inbound in self.transport.drain_inbound() {
            match inbound {
                Inbound::Wire { bytes, .. } => match postcard::from_bytes::<TracerMsg>(&bytes) {
                    Ok(TracerMsg::Ping(n)) => self.pending_replies.push_back(n),
                    Ok(TracerMsg::Pong(n)) => self.trace.push(TraceEvent::PongReceived(n)),
                    Err(_) => self.trace.push(TraceEvent::DecodeError),
                },
                // A tracer is a bare connectivity probe: a peer-unreachable and a local send-shed
                // (R-4d M3) BOTH mean "the ping did not get through", so it records them identically
                // (one arm — a new Inbound variant still forces this to be reconsidered). The mem
                // transport this Tier-A tracer runs on never sheds, so a distinct SendShed event
                // would be an uncoverable region; the real-mesh tracer's shed path is proven in
                // io-prod (Tier-B).
                Inbound::NodeUnreachable { .. } | Inbound::SendShed { .. } => {
                    self.trace.push(TraceEvent::PeerUnreachable);
                }
            }
        }

        // Echo: flush as many pending replies as the outbound queue accepts.
        while let Some(&n) = self.pending_replies.front() {
            match self.send_msg(TracerMsg::Pong(n)) {
                Ok(()) => {
                    self.pending_replies.pop_front();
                }
                Err(SendError::QueueFull(_) | SendError::UnknownPeer(_)) => break,
            }
        }

        // Pinger: one ping per tick; back-pressure retries the same n next tick.
        if let TracerRole::Pinger { total } = self.role
            && self.next_ping < total
        {
            match self.send_msg(TracerMsg::Ping(self.next_ping)) {
                Ok(()) => self.next_ping += 1,
                Err(SendError::QueueFull(_) | SendError::UnknownPeer(_)) => {
                    self.trace.push(TraceEvent::Backpressured(self.next_ping));
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use vd_sim::io::mem::MemHub;

    const PINGER: NodeId = NodeId(10);
    const ECHO: NodeId = NodeId(20);

    #[test]
    fn decode_error_is_traced_not_fatal() {
        let hub = MemHub::new();
        let mut echo_t = hub.register(ECHO, 8);
        let pinger_t = hub.register(PINGER, 8);
        // Send garbage bytes that cannot decode as TracerMsg (postcard enum tag 250).
        echo_t
            .send(PINGER, MsgClass::Control, vec![250, 0, 0].into())
            .expect("accepted");
        hub.pump();

        let mut pinger = TracerNode::new(pinger_t, ECHO, TracerRole::Pinger { total: 0 });
        pinger.step_tick();
        assert_eq!(pinger.trace, vec![TraceEvent::DecodeError]);
        assert_eq!(pinger.pongs_received(), 0);
        assert_eq!(pinger.tick(), vd_core::TickId(1));
    }

    #[test]
    fn echo_buffers_replies_under_backpressure() {
        let hub = MemHub::new();
        let mut pinger_t = hub.register(PINGER, 16);
        let echo_t = hub.register(ECHO, 1); // echo can enqueue only ONE reply per pump

        // Three pings arrive at once.
        for n in 0..3u64 {
            let bytes = postcard::to_allocvec(&TracerMsg::Ping(n)).expect("encode");
            pinger_t
                .send(ECHO, MsgClass::Control, bytes.into())
                .expect("ok");
        }
        hub.pump();

        let mut echo = TracerNode::new(echo_t, PINGER, TracerRole::Echo);
        // Tick 1: drains 3 pings, can flush only 1 reply (capacity 1).
        echo.step_tick();
        hub.pump();
        // Ticks 2..3: flush the buffered replies one per tick.
        echo.step_tick();
        hub.pump();
        echo.step_tick();
        hub.pump();

        let mut pinger = TracerNode::new(pinger_t, ECHO, TracerRole::Pinger { total: 0 });
        pinger.step_tick();
        assert_eq!(
            pinger.trace,
            vec![
                TraceEvent::PongReceived(0),
                TraceEvent::PongReceived(1),
                TraceEvent::PongReceived(2),
            ]
        );
        assert_eq!(pinger.pongs_received(), 3);
    }

    #[test]
    fn pinger_sends_backpressures_and_observes_unreachable() {
        // Outbound capacity 1 and no pump: the first ping is accepted, the second
        // back-pressures and is retried. Killing the peer then surfaces the accepted
        // ping as PeerUnreachable in-band.
        let hub = MemHub::new();
        let pinger_t = hub.register(PINGER, 1);
        let _echo_t = hub.register(ECHO, 8);
        let mut pinger = TracerNode::new(pinger_t, ECHO, TracerRole::Pinger { total: 2 });

        pinger.step_tick(); // Ping(0) accepted
        pinger.step_tick(); // Ping(1) -> QueueFull (queue not pumped)
        assert_eq!(pinger.trace, vec![TraceEvent::Backpressured(1)]);

        hub.kill(ECHO);
        hub.pump(); // Ping(0) -> NodeUnreachable back to the pinger
        pinger.step_tick();
        assert_eq!(
            pinger.trace,
            vec![TraceEvent::Backpressured(1), TraceEvent::PeerUnreachable]
        );
    }
}
