//! SPIKE-0a exit criterion, part 1: the tracer bullet runs 1000 virtual ticks
//! deterministically, in milliseconds, with zero real sleeping and zero sockets.

use vd_core::{NodeId, TickId};
use vd_node::tracer::{TraceEvent, TracerNode, TracerRole};
use vd_sim::io::Transport;
use vd_sim::io::mem::MemHub;

const PINGER: NodeId = NodeId(1);
const ECHO: NodeId = NodeId(2);

#[test]
fn thousand_tick_ping_pong_is_complete_ordered_and_lossless() {
    const N: u64 = 1000;
    let hub = MemHub::new();
    let pinger_t = hub.register(PINGER, 64);
    let echo_t = hub.register(ECHO, 64);
    let mut pinger = TracerNode::new(pinger_t, ECHO, TracerRole::Pinger { total: N });
    let mut echo = TracerNode::new(echo_t, PINGER, TracerRole::Echo);

    // Deterministic drive: step both nodes, then pump the hub — one topology tick.
    let mut iterations = 0u64;
    while pinger.pongs_received() < N as usize {
        pinger.step_tick();
        echo.step_tick();
        hub.pump();
        iterations += 1;
        assert!(iterations < 3 * N, "tracer failed to converge");
    }

    // Every pong arrived, in exact FIFO order, with no other events.
    let expected: Vec<TraceEvent> = (0..N).map(TraceEvent::PongReceived).collect();
    assert_eq!(pinger.trace, expected);
    assert!(echo.trace.is_empty(), "echo observed unexpected events");
    // Round-trip through the queues costs a bounded, explainable number of ticks.
    assert!(
        iterations <= N + 5,
        "unexpected latency: {iterations} ticks for {N} pings"
    );
    // Each node ticked exactly once per topology iteration.
    assert_eq!(pinger.tick(), TickId(iterations));
    assert_eq!(echo.tick(), TickId(iterations));
}

#[test]
fn backpressure_is_observed_and_recovered() {
    // Outbound capacity 1 and NO pump between steps: the second ping backpressures,
    // is retried, and completes once the hub drains — nothing is ever lost.
    const N: u64 = 5;
    let hub = MemHub::new();
    let pinger_t = hub.register(PINGER, 1);
    let echo_t = hub.register(ECHO, 64);
    let mut pinger = TracerNode::new(pinger_t, ECHO, TracerRole::Pinger { total: N });
    let mut echo = TracerNode::new(echo_t, PINGER, TracerRole::Echo);

    // Two pinger steps with no pump: first fills the queue, second backpressures.
    pinger.step_tick();
    pinger.step_tick();
    assert_eq!(pinger.trace, vec![TraceEvent::Backpressured(1)]);

    let mut iterations = 0u64;
    while pinger.pongs_received() < N as usize {
        pinger.step_tick();
        echo.step_tick();
        hub.pump();
        iterations += 1;
        assert!(iterations < 10 * N, "tracer failed to converge");
    }
    let pongs: Vec<u64> = pinger
        .trace
        .iter()
        .filter_map(|e| match e {
            TraceEvent::PongReceived(n) => Some(*n),
            _ => None,
        })
        .collect();
    assert_eq!(pongs, vec![0, 1, 2, 3, 4], "no ping lost or reordered");
}

#[test]
fn garbage_frames_trace_decode_errors_without_breaking_the_node() {
    let hub = MemHub::new();
    let pinger_t = hub.register(PINGER, 8);
    let mut sender = hub.register(ECHO, 8);
    sender
        .send(
            PINGER,
            vd_sim::io::MsgClass::Control,
            vec![250, 0, 0].into(),
        )
        .expect("accepted");
    hub.pump();

    let mut pinger = TracerNode::new(pinger_t, ECHO, TracerRole::Pinger { total: 0 });
    pinger.step_tick();
    assert_eq!(pinger.trace, vec![TraceEvent::DecodeError]);
}

#[test]
fn echo_side_backpressure_loses_nothing() {
    // The echo's outbound queue holds ONE reply; replies buffer and flush over ticks.
    let hub = MemHub::new();
    let pinger_t = hub.register(PINGER, 8);
    let echo_t = hub.register(ECHO, 1);
    let mut pinger = TracerNode::new(pinger_t, ECHO, TracerRole::Pinger { total: 3 });
    let mut echo = TracerNode::new(echo_t, PINGER, TracerRole::Echo);

    let mut iterations = 0u64;
    while pinger.pongs_received() < 3 {
        pinger.step_tick();
        echo.step_tick();
        hub.pump();
        iterations += 1;
        assert!(iterations < 64, "echo failed to flush buffered replies");
    }
    let expected: Vec<TraceEvent> = (0..3).map(TraceEvent::PongReceived).collect();
    assert_eq!(pinger.trace, expected);
}

#[test]
fn dead_peer_surfaces_unreachable_in_band() {
    let hub = MemHub::new();
    let pinger_t = hub.register(PINGER, 8);
    let _echo_t = hub.register(ECHO, 8);
    let mut pinger = TracerNode::new(pinger_t, ECHO, TracerRole::Pinger { total: 1 });

    hub.kill(ECHO);
    pinger.step_tick(); // sends Ping(0)
    hub.pump(); //  -> NodeUnreachable re-enters pinger's inbound
    pinger.step_tick(); // drains it

    assert_eq!(pinger.trace, vec![TraceEvent::PeerUnreachable]);
}
