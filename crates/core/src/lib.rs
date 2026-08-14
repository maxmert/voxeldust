//! # vd-core — pure, shard-agnostic domain foundation
//!
//! The bottom of the dependency stack (`bins → node → sim → wire → core`).
//! Everything here is pure logic: no I/O, no clocks, no sockets, no persistence.
//!
//! Owns (per `docs/design/PLAN.md` and the hardened designs in `docs/design/`):
//! - Identity/correlation primitives: `EntityId`, `SessionId`, `AccountId`, `TransferId`,
//!   `Fence` (the ONE linearizability primitive), `TickId`, `EpochId`.
//! - The generic-transfer registry (HR2): `EntityKind`, `KindDef`, `TransferableKind`,
//!   `DurabilityClass`, `GhostPolicy`, `ContinuityModel`, `LossBudget`.
//! - TLV-framed entity-state blobs with version-floor evolution (`docs/design/generic_transfer.md` §A1).
//! - `StampedPose`/`FrameRef`: the one coordinate-frame transfer type.
//! - Overlap-band geometry: velocity-scaled widths, hysteresis, swept-segment crossing.
//!
//! The closed-form celestial math + the seed universe generator moved to `vd-physics` (the placement
//! arc S5): this crate carries the crossing/containment path, and SL4 demands that path cannot name a
//! motion — held by the crate graph (`tests/tests/crate_isolation.rs`), not by review.
//!
//! Coverage: Tier-A — 100% region + branch (HR5).

pub mod collections;
pub mod entity_kind;
pub mod fence;
pub mod frame;
pub mod geometry;
pub mod home;
pub mod ids;
pub mod incarnation;
pub mod kinematics;
pub mod placement;
pub mod pose;
pub mod realm_coord;
pub mod realm_path;
pub mod rng;
pub mod taxonomy;
pub mod tlv;
pub mod worldgen;

/// Re-exported so consumers use ONE glam version (pose types expose its vectors).
pub use glam;

pub use fence::Fence;
pub use ids::{
    AccountId, BatchId, EntityId, EpochId, MsgId, NodeId, SessionId, TickId, TransferId,
    UniverseTick,
};
