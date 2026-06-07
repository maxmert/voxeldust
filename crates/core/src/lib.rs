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
//! - Closed-form celestial math (Category A — analytic, bit-deterministic): Kepler solve,
//!   `planet_soi` (Hill sphere) and `system_soi` (luminosity) — deliberately DISTINCT functions.
//! - Overlap-band geometry: velocity-scaled widths, hysteresis, swept-segment crossing.
//!
//! Coverage: Tier-A — 100% region + branch (HR5).

pub mod celestial;
pub mod collections;
pub mod entity_kind;
pub mod fence;
pub mod geometry;
pub mod ids;
pub mod pose;
pub mod tlv;

pub use fence::Fence;
pub use ids::{
    AccountId, EntityId, EpochId, MsgId, NodeId, SessionId, TickId, TransferId, UniverseTick,
};
