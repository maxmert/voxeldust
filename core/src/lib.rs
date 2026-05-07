pub mod block;
pub mod camera_math;
pub mod character;
pub mod ecs;
// Signal subsystem lives in `voxeldust-signal`; re-export under the historical
// name so `use voxeldust_core::signal::*` paths in shards/client keep
// resolving. See BUILD_PERF.md §"Workspace layout" for the firewall rationale.
pub use voxeldust_signal as signal;
pub mod builder_pool;
pub mod wire_codec;
pub mod shard_types;
pub mod handoff;
pub mod protocol_generated;
pub mod shard_message;
pub mod client_message;
pub mod seed;
pub mod galaxy;
pub mod system;
pub mod autopilot;
pub mod weather;
pub mod physics_constants;
pub mod blackbody;
pub mod stellar;
pub mod geophysics;
pub mod media;
pub mod planet_rotation;
pub mod spatial;
