//! SPIKE-0a exit criterion, part 2: the SAME tracer node over REAL quinn loopback
//! produces logical output identical to the in-memory run, and the two R9-cure paths
//! (sync `QueueFull`, async `NodeUnreachable`) behave identically on both transports.

use std::time::{Duration, Instant};

use vd_core::{MsgId, NodeId};
use vd_io_prod::loopback_pair;
use vd_node::tracer::{TraceEvent, TracerNode, TracerRole};
use vd_sim::io::mem::MemHub;
use vd_sim::io::{Inbound, MsgClass, SendError, Transport};

const A: NodeId = NodeId(1);
const B: NodeId = NodeId(2);
const DEADLINE: Duration = Duration::from_secs(20);

/// The normalized logical output of the standard scenario on ANY transport.
fn run_scenario<T: Transport>(
    mut pinger: TracerNode<T>,
    mut echo: TracerNode<T>,
    n: u64,
    mut between_ticks: impl FnMut(),
) -> (Vec<TraceEvent>, Vec<TraceEvent>) {
    let started = Instant::now();
    while pinger.pongs_received() < n as usize {
        pinger.step_tick();
        echo.step_tick();
        between_ticks();
        assert!(started.elapsed() < DEADLINE, "scenario failed to converge");
    }
    (pinger.trace, echo.trace)
}

#[test]
fn identical_logical_output_mem_vs_quinn() {
    const N: u64 = 100;

    // In-memory run.
    let hub = MemHub::new();
    let mem_pinger = TracerNode::new(hub.register(A, 64), B, TracerRole::Pinger { total: N });
    let mem_echo = TracerNode::new(hub.register(B, 64), A, TracerRole::Echo);
    let (mem_trace, mem_echo_trace) = run_scenario(mem_pinger, mem_echo, N, || hub.pump());

    // Real-QUIC run: same nodes, same roles, same protocol bytes.
    let pair = loopback_pair(A, B, 64, false).expect("loopback pair");
    let quinn_pinger = TracerNode::new(pair.a, B, TracerRole::Pinger { total: N });
    let quinn_echo = TracerNode::new(pair.b, A, TracerRole::Echo);
    let (quinn_trace, quinn_echo_trace) = run_scenario(quinn_pinger, quinn_echo, N, || {
        std::thread::sleep(Duration::from_micros(200));
    });

    // THE spike assertion: byte-for-byte identical logical traces.
    assert_eq!(mem_trace, quinn_trace);
    assert_eq!(mem_echo_trace, quinn_echo_trace);
    let expected: Vec<TraceEvent> = (0..N).map(TraceEvent::PongReceived).collect();
    assert_eq!(quinn_trace, expected);
}

#[test]
fn backpressure_parity_mem_vs_quinn() {
    const CAPACITY: usize = 4;
    const FLOOD: u64 = 6;

    let flood = |t: &mut dyn Transport| -> Vec<Result<MsgId, SendError>> {
        (0..FLOOD)
            .map(|n| t.send(B, MsgClass::Input, vec![u8::try_from(n).unwrap_or(0)]))
            .collect()
    };

    // Mem: no pump -> queue saturates at CAPACITY.
    let hub = MemHub::new();
    let mut mem_a = hub.register(A, CAPACITY);
    let _mem_b = hub.register(B, CAPACITY);
    let mem_results = flood(&mut mem_a);

    // Quinn: writer started PAUSED -> deterministically saturates at CAPACITY.
    let mut pair = loopback_pair(A, B, CAPACITY, true).expect("loopback pair");
    let quinn_results = flood(&mut pair.a);

    let pattern = |results: &[Result<MsgId, SendError>]| -> Vec<bool> {
        results.iter().map(Result::is_ok).collect()
    };
    assert_eq!(pattern(&mem_results), pattern(&quinn_results));
    assert_eq!(
        pattern(&quinn_results),
        vec![true, true, true, true, false, false],
        "exactly CAPACITY accepted, the rest back-pressured"
    );

    // Release the quinn writer: the 4 accepted frames must all arrive at B, in order.
    pair.a_ctl.release_writer();
    let started = Instant::now();
    let mut received: Vec<u8> = Vec::new();
    while received.len() < CAPACITY {
        for inbound in pair.b.drain_inbound() {
            if let Inbound::Wire { bytes, .. } = inbound {
                received.push(bytes[0]);
            }
        }
        std::thread::sleep(Duration::from_millis(1));
        assert!(
            started.elapsed() < DEADLINE,
            "accepted frames never arrived"
        );
    }
    assert_eq!(received, vec![0, 1, 2, 3]);
}

#[test]
fn unreachable_parity_mem_vs_quinn() {
    // Mem: kill the peer, send once, observe exactly one NodeUnreachable.
    let hub = MemHub::new();
    let mut mem_a = hub.register(A, 8);
    let _mem_b = hub.register(B, 8);
    hub.kill(B);
    let mem_sent = mem_a.send(B, MsgClass::Saga, vec![9]).expect("enqueued");
    hub.pump();
    let mem_events = mem_a.drain_inbound();
    assert_eq!(
        mem_events,
        vec![Inbound::NodeUnreachable {
            to: B,
            class: MsgClass::Saga,
            undelivered: mem_sent,
        }]
    );

    // Quinn: kill B's endpoint, send once, the writer's failure re-enters inbound.
    let mut pair = loopback_pair(A, B, 8, false).expect("loopback pair");
    pair.b_ctl.kill();
    std::thread::sleep(Duration::from_millis(100)); // let CONNECTION_CLOSE propagate
    let quinn_sent = pair.a.send(B, MsgClass::Saga, vec![9]).expect("enqueued");

    let started = Instant::now();
    let mut quinn_events: Vec<Inbound> = Vec::new();
    while quinn_events.is_empty() {
        quinn_events = pair.a.drain_inbound();
        std::thread::sleep(Duration::from_millis(5));
        assert!(
            started.elapsed() < DEADLINE,
            "NodeUnreachable never surfaced"
        );
    }
    assert_eq!(
        quinn_events,
        vec![Inbound::NodeUnreachable {
            to: B,
            class: MsgClass::Saga,
            undelivered: quinn_sent,
        }]
    );
    assert_eq!(
        mem_events, quinn_events,
        "identical logical failure surface"
    );
}
