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

pub mod body_head;
pub mod class;
pub mod components;
pub mod desired;
pub mod handoff_blob;
pub mod hooks;
pub mod ik;
pub mod locomotion;
pub mod look_target;
pub mod stance;
pub mod state_machine;
pub mod stats;

#[cfg(feature = "rapier")]
pub mod controller;
#[cfg(feature = "rapier")]
pub mod kcc_sys;

pub use body_head::{BodyYaw, CharacterClassComp, HeadPitch, HeadYaw, TurnInPlace};
pub use class::{
    class_by_id, CharacterClass, ClipDef, ClipKind, ClipLabel, ALL_CLASSES, HUMAN_DEFAULT,
};
pub use components::{CharacterCapsule, IsCharacter};
pub use desired::{DesiredMovement, PlatformDelta};
pub use handoff_blob::{decode as decode_character_state, encode as encode_character_state, CharacterStateBlob, SCHEMA_VERSION as CHARACTER_SCHEMA_VERSION};
pub use hooks::{ActiveItem, CharacterComponentTag, DamageResistance, EquipmentLoad, Stamina};
pub use ik::{solve_aim_chain, solve_two_bone_ik, AimChainInput, AimChainOutput, TwoBoneInput, TwoBoneOutput};
pub use locomotion::{
    CharacterVelocity, GravityOverride, LandedEvent, LocalUp, LocomotionState,
    PlatformSnapSuppressed,
};
pub use look_target::{LookAtIkEnabled, LookTarget};
pub use stance::{CharacterStance, StanceAction};
pub use state_machine::{apply_update, step_body_head, wrap_pi, BodyHeadUpdate};
pub use stats::MovementStats;

#[cfg(feature = "rapier")]
pub use components::CharacterController;
#[cfg(feature = "rapier")]
pub use controller::{build_character, CharacterBuildSpec};
#[cfg(feature = "rapier")]
pub use kcc_sys::{kcc_move_all, move_one_character, CharacterCollisionEvent, CharacterMoveInput, CharacterMoveResult, CharacterRecord, RapierWorld};
