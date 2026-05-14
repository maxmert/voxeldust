//! Tablet-hold interaction state.
//!
//! When a player E/F-keys a functional block, the server inserts
//! [`IsHoldingTablet`] on their character entity. The client renders a
//! held-tablet visual + drives both arms via IK toward chest-relative
//! hold offsets and the tablet-screen cursor projection. The state is
//! broadcast to all observers so REMOTE players see the same hold pose
//! + finger position — Star Citizen-grade body language replication.
//!
//! # Server-driven by design
//!
//! The mandate for this project is server-authoritative state. Per
//! `feedback_no_client_prediction` and `project_block_system_plan`, the
//! tablet-open path is:
//!
//! ```text
//! client F-press → ClientMsg::TabletOpen → server validates distance →
//!     insert IsHoldingTablet + TabletCursor on character →
//!     broadcast on PlayerSnapshot.is_holding_tablet →
//!     client receives → spawns tablet visual + drives IK
//! ```
//!
//! The local player accepts a one-tick (~50 ms) latency between F-press
//! and tablet appearing; that's the cost of avoiding client prediction
//! and getting consistent state across all observers.
//!
//! # Wire bandwidth
//!
//! `is_holding_tablet: bool` (1 bit), `cursor_uv: Vec2` (8 B) per
//! tablet-active player. At 100 K simultaneous users with even 5%
//! tablet-active concurrency that's ~40 KB/s aggregate broadcast.
//! Negligible vs everything else PlayerSnapshot already carries.

use bevy_ecs::component::Component;
use glam::Vec2;

/// Marker that the character is holding a tablet (server-side
/// authoritative state). Inserted by the shard when a `TabletOpen`
/// client message validates; removed on `TabletClose` or when the
/// player moves out of interaction range.
#[derive(Component, Default, Clone, Copy, Debug, PartialEq, Eq)]
pub struct IsHoldingTablet;

/// Latest cursor position the holding client has reported. UV
/// coordinates in the tablet's screen space, [0, 1] both axes
/// (`(0, 0)` = top-left, `(1, 1)` = bottom-right).
///
/// The server never reads this for any gameplay decision — it's a
/// transparent passthrough to the broadcast so REMOTE players can
/// drive the holding character's left-hand IK to the same tablet
/// surface point. Server clamps to `[0, 1]` on receive as a
/// defence against bad clients (no exploit value here, but it
/// keeps the wire contract clean).
#[derive(Component, Clone, Copy, Debug, PartialEq, Default)]
pub struct TabletCursor(pub Vec2);
