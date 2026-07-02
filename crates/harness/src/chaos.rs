//! `ChaosRunner(seed)`: a whole topology, fault tape, and crash schedule derived
//! from ONE seed (`docs/design/test_harness.md` §6). The reproducibility gate runs
//! the same seed twice in SEPARATE PROCESSES and requires byte-identical traces —
//! "reproducible" is a gate, not a claim.
//!
//! P0 scope: relay-ring nodes (points passing opaque messages — no game logic),
//! seed-derived link policies, crashes in all three phases, wire-truth and
//! conservation verified at the end. Later phases swap richer nodes/scenarios into
//! the SAME runner.

use vd_core::rng::SplitMix64;
use vd_core::{NodeId, TickId};
use vd_node::app::{InboundBox, NodeConfig, OutboundBox};
use vd_sim::capability::NodeKind;
use vd_sim::io::{Inbound, MsgClass, Transport};

use crate::fabric::{CrashWhen, FaultFabric, LinkPolicy};
use crate::topology::{StaggerPlan, SteppableNode, Topology, WireTruthViolation};

/// Bounded chaos parameters (operational, reviewed — never inline literals at use
/// sites). The DEFAULT is the P0 gate configuration.
#[derive(Clone, Copy, Debug)]
pub struct ChaosConfig {
    pub nodes: u64,
    pub ticks: u64,
    /// Upper bounds for seed-derived policy draws.
    pub max_drop_p: f64,
    pub max_dup_p: f64,
    pub max_extra_delay_ticks: u64,
    /// How many crash/resurrect pairs to schedule.
    pub crash_pairs: u64,
    /// Resurrection lag after each crash.
    pub resurrect_after_ticks: u64,
    pub retry_delay_ticks: u64,
    /// How many PERMANENT kills to schedule (seed-derived victims/ticks). Killed
    /// nodes never return; traffic toward them bounces as `NodeUnreachable`.
    pub kills: u64,
}

impl Default for ChaosConfig {
    fn default() -> ChaosConfig {
        ChaosConfig {
            nodes: 4,
            ticks: 256,
            max_drop_p: 0.4,
            max_dup_p: 0.2,
            max_extra_delay_ticks: 4,
            crash_pairs: 3,
            resurrect_after_ticks: 6,
            retry_delay_ticks: 2,
            kills: 0,
        }
    }
}

/// Build a relay node forwarding everything to `peer` (infrastructure, not gameplay).
fn relay_node(fabric: &FaultFabric, id: NodeId, peer: NodeId) -> Box<dyn SteppableNode> {
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
            // Forward at most ONE message per tick: dup_p would otherwise compound
            // exponentially around the ring (every duplicate re-forwarded forever) —
            // a message storm, not a useful chaos scenario. Bounded relays keep
            // steady traffic crossing every faulted link. Unreachable notices
            // (a killed peer) are observed and dropped, never forwarded.
            let first_wire = inbox.0.iter().find_map(|m| match m {
                Inbound::Wire { class, bytes, .. } => Some((*class, bytes.clone())),
                // Notices (a killed peer's NodeUnreachable, or a local SendShed — R-4d M3) are
                // observed and dropped, never forwarded around the ring. One arm: the fabric never
                // sheds, so a distinct SendShed arm would be uncoverable, and a new Inbound variant
                // still forces this to be reconsidered.
                Inbound::NodeUnreachable { .. } | Inbound::SendShed { .. } => None,
            });
            if let Some((class, bytes)) = first_wire {
                outbox.0.push((peer, class, bytes));
            }
        },
    );
    Box::new(node)
}

/// Run one fully seed-determined chaos scenario; returns the logical trace bytes,
/// or the wire-truth violation if any node's claims diverged from delivery reality.
///
/// # Errors
/// A `WireTruthViolation` names the lying node; pair it with the seed for exact
/// replay. (Conservation is asserted every step inside the topology — louder still.)
pub fn run_chaos(seed: u64, config: &ChaosConfig) -> Result<Vec<u8>, WireTruthViolation> {
    let mut rng = SplitMix64::new(seed);
    let fabric = FaultFabric::new(rng.next_u64(), config.retry_delay_ticks);
    let mut topo = Topology::new(fabric.clone(), StaggerPlan::lockstep());

    // A ring of relays: node i forwards to node (i+1) % n.
    let n = config.nodes.max(2);
    for i in 0..n {
        let id = NodeId(i + 1);
        let peer = NodeId((i + 1) % n + 1);
        topo.add_node(relay_node(&fabric, id, peer));
    }

    // Seed-derived per-link policies around the ring.
    for i in 0..n {
        let from = NodeId(i + 1);
        let to = NodeId((i + 1) % n + 1);
        let policy = LinkPolicy {
            drop_p: rng.next_f64() * config.max_drop_p,
            dup_p: rng.next_f64() * config.max_dup_p,
            max_extra_delay_ticks: rng.range_u64(0, config.max_extra_delay_ticks + 1),
            partitioned: false,
            send_reject_p: 0.0,
            // D-3 flap is a deterministic-window fault (not a probability), so the seeded chaos sweep
            // does not inject it — it is exercised by the dedicated CSCALE-1 flap cells (Slice 3).
            flap_until_tick: None,
        };
        fabric.set_policy(from, to, policy);
    }

    // Seed-derived crash schedule across all three phases.
    for _ in 0..config.crash_pairs {
        let victim = NodeId(rng.range_u64(0, n) + 1);
        let at = TickId(
            rng.range_u64(
                2,
                config
                    .ticks
                    .saturating_sub(config.resurrect_after_ticks + 2)
                    .max(3),
            ),
        );
        let when = match rng.range_u64(0, 3) {
            0 => CrashWhen::PreInject,
            1 => CrashWhen::PostInject,
            _ => CrashWhen::PostStep,
        };
        topo.schedule_crash(victim, at, when);
        topo.schedule_resurrect(victim, TickId(at.0 + config.resurrect_after_ticks));
    }

    // Seed-derived PERMANENT kills (only drawn when configured, so default-config
    // traces are unaffected). Sorted (tick, victim) plan, applied before the step.
    let mut kill_plan: Vec<(u64, NodeId)> = (0..config.kills)
        .map(|_| {
            let victim = NodeId(rng.range_u64(0, n) + 1);
            let at = rng.range_u64(2, config.ticks.max(3));
            (at, victim)
        })
        .collect();
    kill_plan.sort_unstable();

    // Seed traffic into the ring.
    let mut seeder = fabric.register(NodeId(9999));
    for k in 0..3u8 {
        let target = NodeId(rng.range_u64(0, n) + 1);
        seeder
            .send(target, MsgClass::Control, vd_sim::io::bytes(vec![k]))
            .expect("seed message accepted");
    }

    for t in 0..config.ticks {
        let next_topology_tick = t + 1;
        while kill_plan
            .first()
            .is_some_and(|(at, _)| *at <= next_topology_tick)
        {
            let (_, victim) = kill_plan.remove(0);
            fabric.kill(victim);
        }
        topo.step();
    }

    // Terminal verification: conservation held every step (topology asserts);
    // wire truth must hold at the end.
    topo.verify_wire_truth().map(|()| topo.trace_bytes())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn trace(seed: u64, config: &ChaosConfig) -> Vec<u8> {
        run_chaos(seed, config).expect("wire truth holds")
    }

    #[test]
    fn chaos_is_reproducible_in_process() {
        let config = ChaosConfig::default();
        assert_eq!(trace(1234, &config), trace(1234, &config));
        assert_ne!(
            trace(1234, &config),
            trace(1235, &config),
            "different seeds, different fault tapes"
        );
    }

    #[test]
    fn chaos_survives_many_seeds() {
        // A small seed sweep: every run must hold conservation (asserted every
        // step) + wire truth (the Ok) regardless of the fault tape.
        let config = ChaosConfig {
            ticks: 96,
            ..ChaosConfig::default()
        };
        for seed in 0..8 {
            let _ = trace(seed, &config);
        }
    }

    #[test]
    fn chaos_with_more_nodes_and_aggressive_faults() {
        let config = ChaosConfig {
            nodes: 6,
            ticks: 128,
            max_drop_p: 0.6,
            max_dup_p: 0.4,
            max_extra_delay_ticks: 6,
            crash_pairs: 5,
            ..ChaosConfig::default()
        };
        let _ = trace(77, &config);
    }

    #[test]
    fn chaos_with_permanent_kills_keeps_wire_truth_and_reproducibility() {
        // Killed nodes bounce traffic as NodeUnreachable through the surviving
        // relays — accounting must stay exact and the trace seed-determined.
        let config = ChaosConfig {
            ticks: 96,
            kills: 2,
            ..ChaosConfig::default()
        };
        assert_eq!(trace(55, &config), trace(55, &config));
        for seed in 50..56 {
            let _ = trace(seed, &config);
        }
    }
}
