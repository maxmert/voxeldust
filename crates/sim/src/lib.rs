//! # vd-sim — pure simulation logic + the ShardIo seam
//!
//! All production simulation code obtains **time, randomness, persistence, and message I/O
//! only through the injected `sim::io` traits** — it never calls `tokio::time`, `quinn`,
//! `redb`, `SystemTime::now`, or `rand::thread_rng` directly. Enforced by this crate's
//! `clippy.toml` (`disallowed-methods` / `disallowed-types`), not by convention.
//! (`docs/design/test_harness.md` §1, §9.)
//!
//! Owns:
//! - `io`: the `ShardIo` trait bundle — `Transport` (enqueue-only `send → Err(QueueFull)`
//!   sync backpressure; hard failures surface async as `Inbound::NodeUnreachable`),
//!   `Clock`, `Store` (WAL-buffer puts, fsync OFF the tick thread), `DetRng`, `Provisioner`
//!   (persist-intent-before-spawn) — plus `io::mem`, the in-memory deterministic impls.
//! - `saga`: the Transfer Saga as a PURE `fn step(State, Event) -> (State, Vec<Action>)`
//!   (PREPARE → FLUSH → FENCE_DEMOTE → PROMOTE → CLEANUP, AWAIT_PROVISION, aborts,
//!   compensators, DurabilityClass fan-out incl. the batched TransientGo path). Proptested.
//! - `authority`: the per-entity authority FSM (Owned / Ghost / Frozen).
//! - `capability`: `NodeKind` + `ShardProfile::build()` — the validated capability DAG (HR4);
//!   incoherent profiles fail loud at config load, never at runtime.
//! - `coupling`: the `EffectFree` marker + `CouplingPort` trait (HR1) — ports land at P8.
//!
//! Coverage: Tier-A — 100% region + branch (HR5).

pub mod authority;
pub mod capability;
pub mod coupling;
pub mod directory;
pub mod io;
pub mod runtime;
pub mod saga;
pub mod stub;

// The remaining ShardIo traits (Clock, Store, DetRng, Provisioner) land with their
// first consumers: node's tick loop (P0.6) and the harness (P0.7) — grow on real
// pressure, never freeze unconsumed shapes.
