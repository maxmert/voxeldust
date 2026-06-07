//! # vd-node — node construction and the deterministic tick
//!
//! ONE library for every node kind (HR3): `build_app(NodeKind, NodeConfig, impl ShardIo)`
//! returns a `ShardNode<IO>` whose `step_tick() -> TickReport` is the deterministic unit
//! of progress. The prod loop (`run_forever`) and the test `Topology` both drive the SAME
//! `step_tick`; production threading bridges async I/O to it via queue snapshots
//! (`docs/design/test_harness.md` §2) — the sim thread never awaits.
//!
//! Shard *types* (galaxy/system/planet/ship/station/…) are `ShardProfile` capability
//! configurations validated by `vd_sim::capability` — never separate codebases, never
//! `match`ed on in feature code (G-NO-SHARD-FORK).
//!
//! Also owns the `DurableUniverseClock` consumer side: write-ahead tick ceiling,
//! monotonic clamp (backward slew structurally rejected).
//!
//! bevy_ecs runs `ExecutorKind::SingleThreaded` with explicit ordering — total system
//! order is part of the determinism contract.
//!
//! Coverage: Tier-A — 100% region + branch (HR5).

pub mod app;
pub mod tracer;
pub mod universe_clock;

pub use app::{NodeConfig, ShardNode, TickReport, build_app};
