//! Player character system — client side.
//!
//! Phase B of the character system plan: load the per-class skeletal
//! assets (skeleton scene + animation clips) at startup, build a
//! shared [`bevy::animation::AnimationGraph`] for each class, and
//! expose the readiness state via the
//! [`assets::CharacterAssetRegistry`] resource.
//!
//! Phases C–F build on this:
//!  - **C** spawns one skinned-mesh entity per remote/local player
//!    using `Ready` class assets.
//!  - **D** drives the per-entity `AnimationPlayer` weights from the
//!    server-broadcast `Locomotion`.
//!  - **E** attaches the camera to the local player's head bone +
//!    third-person toggle + first-person body cull.
//!  - **F** wires the turn-in-place clip to fade in on
//!    `LocomotionState::TurningInPlace`.
//!
//! The registry is keyed by [`voxeldust_core::character::CharacterClass::id`],
//! so a future race / cosmetic system adds a class by registering a
//! new `CharacterClass` const in `core` and zero changes here.

use bevy::prelude::*;

pub mod anim;
pub mod assets;
pub mod camera_attach;
pub mod foot_ik;
pub mod loader;
pub mod look_ik;
pub mod render;

pub use anim::{CharacterAnimPlugin, CharacterAnimSet, LocomotionAnimState};
pub use assets::{AssetState, CharacterAssetRegistry, ClassAssets};
pub use camera_attach::{
    CameraMode, CharacterCameraPlugin, CharacterCameraSet, LocalCharacterTag,
};
pub use foot_ik::{CharacterFootIkPlugin, CharacterFootIkSet};
pub use look_ik::{CharacterLookIkPlugin, CharacterLookIkSet};
pub use render::{BoneRegistry, CharacterRenderPlugin, CharacterRenderSet, RemoteCharacterTag};

/// SystemSet for character-asset loading — placed after `Startup` so
/// downstream systems can `.after(CharacterAssetSet)` if they need a
/// guarantee the registry has at least one tick of progress.
#[derive(SystemSet, Clone, Eq, PartialEq, Hash, Debug)]
pub struct CharacterAssetSet;

/// Plugin: registers the asset registry resource, the load-enqueue
/// startup system, and the per-frame poll. Everything beyond this is
/// driven via the `CharacterAssetRegistry`.
pub struct CharacterAssetPlugin;

impl Plugin for CharacterAssetPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<CharacterAssetRegistry>()
            .add_systems(Startup, loader::enqueue_class_loads)
            .add_systems(
                Update,
                loader::poll_class_loads.in_set(CharacterAssetSet),
            );
    }
}

/// Top-level character plugin. Sub-plugins:
///  - [`CharacterAssetPlugin`] — Phase B: per-class glTF + clip load.
///  - [`CharacterRenderPlugin`] — Phase C: spawn skinned-mesh visuals
///    for every remote player and sync their pose each frame.
///  - [`CharacterAnimPlugin`] — Phase D: drive `AnimationPlayer`
///    weights from server-broadcast `Locomotion` state with
///    AAA-quality crossfades.
///  - [`CharacterCameraPlugin`] — Phase E: head additive look,
///    camera-bone attachment, V-toggle third-person, FP body cull.
///
/// Phase F+ (turn-in-place clip wiring, IK, ragdoll) will land as
/// additional sub-plugins under this umbrella.
pub struct CharacterPlugin;

impl Plugin for CharacterPlugin {
    fn build(&self, app: &mut App) {
        app.add_plugins(CharacterAssetPlugin)
            .add_plugins(CharacterRenderPlugin)
            .add_plugins(CharacterAnimPlugin)
            .add_plugins(CharacterCameraPlugin)
            .add_plugins(CharacterFootIkPlugin)
            .add_plugins(CharacterLookIkPlugin);
    }
}
