//! StubShard simulation systems: points in empty space (P1–P3 proving ground;
//! `docs/design/sealed_shards.md` §stub, roadmap P1). NO voxels, NO physics, NO
//! rendering — the point of a stub shard is that the CONNECTION and (later)
//! TRANSFER machinery around it is real.
//!
//! Binding behaviors encoded here:
//! - The shard NEVER sees a ticket: it acts on `(SessionId, Fence)` handed over the
//!   gateway↔shard flow, and drops stale-fence input with a logged reason — the
//!   stale-gateway drop branch exists (and is covered) from day one even though it
//!   cannot fire with a single gateway.
//! - Every avatar `EntityId` is minted HERE (`EntityId::pack`, seed-derived entropy,
//!   never time-derived — R7).
//! - Every applied or discarded input lands in [`InputLog`] — the ground truth the
//!   INPUT-CONSERVATION oracle audits against what the fabric delivered.
//! - Frames are emitted only while the shard HOLDS its realm authority (fence
//!   granted via the Directory seam), stamped with that fence.
//!
//! THE MODULE TREE (one lane per file; each file states what it owns and what it does not):
//! `config` dials + derived budgets · `stats` the honesty counters · `dot` the occupant and its
//! input · `session` the gateway lane · `realm_head` the directory round-trip · `greeting` a
//! demand-spawned shard making itself reachable · `regions` the containment forest · `placement`
//! the one placement writer · `containment` the crossing scan · `handoff` the hold ledger ·
//! `saga_arms` demote/promote/re-home · `crossing_receive` the destination adopt · `conversion`
//! the boundary pose conversion · `ghost` the retained ghost · `transient` the batch lane ·
//! `frames` the client snapshot · `window` the window lane · `relay` the sealed up-relay ·
//! `aoi` area of interest · `register` the world build and inbound dispatch.
//!
//! The whole surface is re-exported here, so `vd_sim::stub::X` resolves exactly as it did when
//! this was one file.

mod aoi;
pub mod built_store;
mod config;
mod containment;
mod conversion;
mod crossing_receive;
mod dot;
pub mod drive;
pub mod exterior;
mod frames;
mod ghost;
mod greeting;
mod handoff;
pub mod interest;
pub mod lineage;
mod placement;
pub mod reach;
mod realm_head;
mod regions;
mod register;
pub mod relay;
mod saga_arms;
mod session;
mod stats;
mod transient;
mod window;

#[cfg(test)]
mod tests;

pub use aoi::*;
pub use config::*;
pub use containment::*;
pub use conversion::*;
pub use crossing_receive::*;
pub use dot::*;
pub(crate) use frames::*;
pub use ghost::*;
pub use greeting::*;
pub use handoff::*;
pub use placement::*;
pub use realm_head::*;
pub use regions::*;
pub use register::*;
pub use relay::*;
pub(crate) use saga_arms::*;
pub use session::*;
pub use stats::*;
pub use transient::*;
pub use window::*;
