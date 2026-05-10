//! Body / head decoupling — the components and per-tick state that let the
//! head turn freely up to a per-class limit before the body steps to follow.
//!
//! Lives next to [`super::components::CharacterController`] but kept in its
//! own file because the concerns are different: capsule + KCC handles are
//! Rapier-flavoured, body/head yaw are pure data shared with the renderer
//! and broadcast over the wire as an animation-driver signal.
//!
//! # Authority
//!
//! Server-authoritative. Each shard that owns walking players (ship,
//! planet) runs [`super::state_machine::update_body_head_state`] every
//! Physics tick, after the KCC writes back a position, and broadcasts the
//! resulting `(BodyYaw, HeadYaw, HeadPitch, TurnInPlace?)` triplet so
//! every observer renders the same body-vs-head pose for that player.

use bevy_ecs::component::Component;
use serde::{Deserialize, Serialize};

/// Body's facing yaw in the player's local tangent frame (radians).
///
/// On a planet this is yaw in the local east/north tangent plane; on a
/// ship interior it is yaw in ship-local Y-up. Always wrapped to
/// `(-π, π]` by the state machine so wire deltas stay tight.
#[derive(Component, Clone, Copy, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct BodyYaw(pub f32);

/// Head yaw **relative to the body** (radians), clamped each tick to
/// `±CharacterClass::head_yaw_limit`. When the camera demands more swing
/// the state machine inserts [`TurnInPlace`] to rotate the body instead.
#[derive(Component, Clone, Copy, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct HeadYaw(pub f32);

/// Head pitch in the local tangent frame (radians), clamped each tick to
/// `±CharacterClass::head_pitch_limit`. Replaces the old
/// `PlayerPitch` component and is symmetrical between shards.
#[derive(Component, Clone, Copy, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct HeadPitch(pub f32);

/// Transient marker on a character that is mid-turn-in-place. Presence
/// = turn active; absence = no turn. The `update_body_head_state` system
/// inserts this when [`HeadYaw`] would exceed the per-class limit, and
/// removes it when `t` reaches 1.0.
///
/// `target_body_yaw` is the absolute (tangent-frame) yaw the body is
/// rotating toward; `t` is the eased progress in `[0, 1]`. The client
/// reads both to choose the turn-l vs turn-r animation clip and to drive
/// the clip's playback.
#[derive(Component, Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct TurnInPlace {
    pub target_body_yaw: f32,
    pub t: f32,
}

/// Marker component that points a character entity at its
/// [`super::class::CharacterClass`] config (eye offset, joint limits,
/// turn-in-place duration, asset paths). One class per character — when
/// a player picks a different race / size in a future cosmetic system,
/// the spawn code inserts a different `CharacterClassComp`.
///
/// Carried by **value** (the class is `Copy`) so reads are zero-cost
/// and don't need a resource lookup.
#[derive(Component, Clone, Copy, Debug)]
pub struct CharacterClassComp(pub super::class::CharacterClass);
