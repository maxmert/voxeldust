//! The P0 definition-of-done gates (PLAN.md P0 row). These are PERMANENT: no later
//! phase may regress them.
//!
//! - 12,000 virtual ticks run in milliseconds (no real sleeping anywhere).
//! - A chaos seed replays BYTE-IDENTICALLY across two SEPARATE PROCESSES (per-process
//!   state — hasher seeds, ASLR, allocation order — must not leak into the trace).

use std::time::{Duration, Instant};

use vd_core::NodeId;
use vd_harness::chaos::{ChaosConfig, run_chaos};
use vd_harness::fabric::FaultFabric;
use vd_harness::topology::{StaggerPlan, SteppableNode, Topology};
use vd_node::app::{InboundBox, NodeConfig, OutboundBox};
use vd_sim::capability::NodeKind;
use vd_sim::io::{Inbound, MsgClass, Transport};

fn relay(fabric: &FaultFabric, id: NodeId, peer: NodeId) -> Box<dyn SteppableNode> {
    let mut node = vd_node::build_app(
        NodeConfig {
            node_id: id,
            kind: NodeKind::StubShard,
        },
        fabric.register(id),
    );
    node.schedule_mut().add_systems(
        move |inbox: bevy_ecs::prelude::Res<InboundBox>,
              mut outbox: bevy_ecs::prelude::ResMut<OutboundBox>| {
            for msg in &inbox.0 {
                if let Inbound::Wire { class, bytes, .. } = msg {
                    outbox.0.push((
                        peer,
                        *class,
                        bytes.clone(),
                        vd_sim::io::Durability::Ephemeral,
                    ));
                }
            }
        },
    );
    Box::new(node)
}

/// P0 DoD: 12,000 virtual ticks of a live 4-node topology complete in milliseconds.
/// The bound is deliberately generous (2 s) for slow CI machines; typical runtime is
/// tens of milliseconds — the property being gated is NO REAL SLEEPING, not raw speed.
#[test]
fn twelve_thousand_virtual_ticks_run_in_milliseconds() {
    let fabric = FaultFabric::new(42, 2);
    let mut topo = Topology::new(fabric.clone(), StaggerPlan::lockstep());
    for i in 0..4u64 {
        topo.add_node(relay(&fabric, NodeId(i + 1), NodeId((i + 1) % 4 + 1)));
    }
    let mut seeder = fabric.register(NodeId(99));
    seeder
        .send(NodeId(1), MsgClass::Control, vec![1].into())
        .expect("seed accepted");

    let started = Instant::now();
    for _ in 0..12_000 {
        topo.step();
    }
    let elapsed = started.elapsed();
    assert_eq!(topo.tick().0, 12_000);
    assert!(
        elapsed < Duration::from_secs(2),
        "12k ticks took {elapsed:?} — the virtual clock must never sleep"
    );
    topo.verify_wire_truth()
        .expect("wire truth after 12k ticks");
}

/// P0 DoD: the chaos trace is byte-identical across SEPARATE PROCESSES. The child
/// re-executes this same test binary (filtered to the child test) with the seed in
/// the environment and prints the trace digest; the parent compares two child runs.
#[test]
fn chaos_seed_replays_byte_identically_across_processes() {
    if std::env::var("VD_CHAOS_CHILD").is_ok() {
        // Child mode is handled by chaos_child below; nothing to do here.
        return;
    }
    let exe = std::env::current_exe().expect("test binary path");
    let run_child = || {
        let output = std::process::Command::new(&exe)
            .args(["chaos_child", "--exact", "--nocapture"])
            .env("VD_CHAOS_CHILD", "9001")
            .output()
            .expect("spawn child test process");
        assert!(output.status.success(), "child failed: {output:?}");
        let stdout = String::from_utf8_lossy(&output.stdout);
        stdout
            .lines()
            .find(|l| l.starts_with("CHAOS_TRACE_HEX:"))
            .map(str::to_owned)
            .expect("child printed the trace digest")
    };
    let first = run_child();
    let second = run_child();
    assert_eq!(first, second, "cross-process determinism violated");
}

/// The child half of the two-process gate: runs the seeded chaos and prints the
/// full trace as hex (compared byte-for-byte by the parent).
#[test]
fn chaos_child() {
    let Ok(seed_str) = std::env::var("VD_CHAOS_CHILD") else {
        return; // not in child mode: a no-op test run
    };
    let seed: u64 = seed_str.parse().expect("numeric seed");
    let trace = run_chaos(seed, &ChaosConfig::default()).expect("wire truth holds");
    let hex: String = trace.iter().map(|b| format!("{b:02x}")).collect();
    println!("CHAOS_TRACE_HEX:{hex}");
}
