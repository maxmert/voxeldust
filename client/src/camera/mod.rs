//! Generic camera driver — the camera entity stays at Bevy origin
//! (Transform::IDENTITY translation). Per-frame the driver reads the
//! authoritative player pose from `PrimaryWorldState` and updates:
//!
//! - `CameraWorldPos` → absolute world-space position in the primary's
//!   coordinate frame. Phase 4's origin rebase system consumes this to
//!   place every shard's `ChunkSource` Transform.
//! - Camera entity's Transform.rotation → world-space rotation of the
//!   player's view. Phase 11 MVP: identity rotation. Later work wires
//!   each shard-type's plugin camera-frame method to compute this
//!   correctly (ship lock, planet tangent, EVA body frame, galaxy
//!   parallax).
//!
//! `SpawnPoseLatch` from Phase 10 is drained here: a pending pose
//! overrides the WorldState-derived position for the first frame of a
//! new primary, so transitions are instantaneous.

pub mod pose;

pub use pose::{PlayerSyncPlugin, PlayerSyncSet};
