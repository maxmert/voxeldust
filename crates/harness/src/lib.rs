//! # vd-harness — the deterministic feedback loop
//!
//! The cure for the old project's only feedback loop (rebuild k3d → fly client → grep
//! logs). An in-process `Topology` of N nodes + M scripted clients advances tick-by-tick
//! on a `VirtualClock` (12,000 ticks in milliseconds), with deliberate tick-skew
//! (`StaggerPlan`), seeded delivery shuffles, and seed-reproducible chaos
//! (`docs/design/test_harness.md`).
//!
//! Two monitors with a strict data-source rule:
//! - `ControlOracle` (ground truth: directory, saga records, node-reported held-sets):
//!   AUTHORITY-UNIQUE (`len == 1` exactly), TRANSIENT-AUTHORITY-HELD, ORPHAN-GHOST,
//!   NO-UNILATERAL-COMMIT, LIVENESS, DURABILITY, DURABLE-UNAFFECTED-BY-BURST.
//! - `WireMonitor` (consumes ONLY bytes the `FaultFabric` actually delivered):
//!   NO-VANISH, POSE-CONTINUITY, INPUT-CONSERVATION, TRANSIENT-CONSERVATION.
//!   A meta-test deliberately diverges ECS state from delivered snapshots and proves
//!   the WireMonitor FAILS — wire truth, not internal hope.
//!
//! `FaultFabric` is at-least-once ("delivered" == durably acked; redelivery until ack)
//! with drop/dup/reorder/delay/partition/send-reject and crash timing as a first-class
//! dimension (`CrashWhen ∈ {PreInject, PostInject, PostStep}`).
//!
//! `ChaosRunner(seed)`: failing seeds replay byte-identically (asserted twice in
//! separate processes) and auto-shrink into named regression tests.
//!
//! Coverage: Tier-A — 100% region + branch (HR5).

pub mod chaos;
pub mod client;
pub mod fabric;
pub mod oracle;
pub mod topology;
