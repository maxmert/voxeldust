//! Client block raycast, highlight, and interaction dispatch.
//!
//! The raycast is **cross-shard** (Design Principle #9): iterates every
//! loaded `ChunkSource`, inverse-transforms the camera ray into each
//! shard's local frame, runs the `voxeldust_core::block::raycast::raycast`
//! DDA against that shard's storage, and keeps the closest hit across
//! every shard. The hit carries a `ShardKey` so Phase 15's dispatch can
//! route `BlockEditRequest` to the TCP of the shard that OWNS the hit
//! target — a ship parked on the ground is fully editable from the
//! planet primary, EVA repairs work from SYSTEM primary, etc.

pub mod dispatch;
pub mod highlight;
pub mod raycast;

pub use dispatch::{Hotbar, InteractionPlugin, InteractionSet};
pub use highlight::{BlockHighlightPlugin, BlockHighlightSet};
pub use raycast::{BlockRaycastPlugin, BlockRaycastSet, BlockTarget, BlockTargetHit};
