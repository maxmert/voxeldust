//! Shared character layer.
//!
//! Server-authoritative kinematic character controller used by every shard that
//! runs player walking physics (ship-shard, planet-shard). EVA in system-shard
//! uses custom Newtonian integration and does **not** go through this layer.
//!
//! # Why this module exists
//!
//! The original implementation drove the player with a `dynamic` Rapier rigid
//! body and re-issued `set_linvel` on every network packet. That pattern is
//! fundamentally incompatible with server-authoritative character movement:
//! every zero-movement packet instantly zeroed horizontal velocity, producing
//! the "patchy walking" the project has been debugging. Every AAA engine
//! (Unreal, Unity, Source, Havok/Bethesda, Cryengine) uses a kinematic
//! character controller for the same reason.
//!
//! # Architecture
//!
//! - [`CharacterController`](components::CharacterController) wraps Rapier's
//!   `KinematicCharacterController` + the kinematic-position-based rigid body
//!   that other entities can still query and collide against.
//! - Per-tick flow: `process_input` writes `DesiredMovement`; the `Physics`
//!   set runs stance transitions → movement modifiers →
//!   platform delta → **`kcc_move_characters`** → rapier step → position sync.
//! - `CharacterVelocity` persists horizontal + vertical velocity between
//!   ticks — the piece `set_linvel` was destroying every packet.
//!
//! # Extension points (reserved, not implemented)
//!
//! - [`hooks::Stamina`] — sprint/jump cost.
//! - [`hooks::EquipmentLoad`] — armor/gear mass modifies speed.
//! - [`hooks::ActiveItem`] — equipped weapon/tool slot.
//! - [`hooks::DamageResistance`] — armor damage reduction table.
//!
//! These are empty component stubs today; the KCC path reads them via a
//! placeholder `apply_movement_modifiers` system so future gameplay lands
//! without another character-physics refactor.

pub mod components;
pub mod desired;
pub mod hooks;
pub mod locomotion;
pub mod stance;
pub mod stats;

#[cfg(feature = "rapier")]
pub mod controller;
#[cfg(feature = "rapier")]
pub mod kcc_sys;

pub use components::{CharacterCapsule, IsCharacter};
pub use desired::{DesiredMovement, PlatformDelta};
pub use hooks::{ActiveItem, CharacterComponentTag, DamageResistance, EquipmentLoad, Stamina};
pub use locomotion::{
    CharacterVelocity, GravityOverride, LandedEvent, LocalUp, LocomotionState,
    PlatformSnapSuppressed,
};
pub use stance::{CharacterStance, StanceAction};
pub use stats::MovementStats;

#[cfg(feature = "rapier")]
pub use components::CharacterController;
#[cfg(feature = "rapier")]
pub use controller::{build_character, CharacterBuildSpec};
#[cfg(feature = "rapier")]
pub use kcc_sys::{kcc_move_all, move_one_character, CharacterCollisionEvent, CharacterMoveInput, CharacterMoveResult, CharacterRecord, RapierWorld};
