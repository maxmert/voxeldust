//! World-space attention target for character look-at IK (Phase H).
//!
//! A character with `LookTarget(Some(world_pos))` will rotate its
//! neck and head bones to look at that point. `None` leaves the head
//! in whatever pose the animation clip + body/head decoupling
//! produced.
//!
//! # Authority
//!
//! Server-authoritative. The shard's `update_look_targets` system
//! recomputes each player's target every tick (initially: player
//! forward × `look_attention_distance`; later: nearest other player
//! within FOV for "lifelike crowd attention" behaviour) and the
//! server broadcasts it via [`crate::character::handoff_blob`] /
//! the wire `PlayerSnapshot.look_*` fields.
//!
//! # Why a world-space target instead of yaw/pitch angles?
//!
//! Yaw + pitch are body-relative — when the body turns, the head's
//! visible aim drifts. World-space target stays anchored: a head
//! looking at a fixed point keeps looking at that point even as the
//! character walks past, exactly the way real attention works.

use bevy_ecs::component::Component;
use glam::DVec3;

/// What the character is currently paying attention to, in
/// **system-space** (the same f64 frame as `Position`). `None` =
/// no target → head returns to the animation pose.
#[derive(Component, Clone, Copy, Debug, Default, PartialEq)]
pub struct LookTarget(pub Option<DVec3>);

impl LookTarget {
    /// Convenience: `Some(target)` constructor.
    #[inline]
    pub fn at(target: DVec3) -> Self {
        Self(Some(target))
    }

    /// Convenience: `None`.
    #[inline]
    pub fn cleared() -> Self {
        Self(None)
    }
}
