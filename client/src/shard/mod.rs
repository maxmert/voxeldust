//! Generic shard infrastructure — shard-type-agnostic.
//!
//! Shard-type-specific logic lives in `crate::shard_types::*`; this module
//! defines the trait and the registry that drive it.

pub mod origin;
pub mod plugin;
pub mod registry;
pub mod runtime;
pub mod transition;
pub mod worldstate;

pub use origin::{CameraWorldPos, ShardOrigin, ShardOriginPlugin, ShardOriginSet};
pub use plugin::ShardTypePlugin;
pub use registry::{
    PrimaryShard, Secondaries, ShardRegistryPlugin, ShardRegistrySet, ShardTypeRegistry,
    SourceIndex,
};
pub use runtime::{ChunkSource, ShardKey, ShardRuntime};
pub use transition::{
    GracedSource, GraceWindow, ShardTransitionPlugin, ShardTransitionSet, SpawnPoseLatch,
};
pub use worldstate::{
    PrimaryWorldState, SecondaryWorldStates, WorldStateIngestPlugin, WorldStateIngestSet,
};
