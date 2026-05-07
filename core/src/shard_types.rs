//! Backward-compatible re-export shim.
//!
//! The actual definitions of `ShardId`, `ShardType`, `SessionToken`,
//! `ShardState`, `ShardEndpoint`, `ShardInfo`, `ShardHeartbeat` live in the
//! foundational `voxeldust-types` crate so that signal + protocol code can
//! reach them without depending on the rest of `voxeldust-core`. This file
//! exists so existing `use voxeldust_core::shard_types::ShardId` paths keep
//! resolving — see BUILD_PERF.md §"Workspace layout".

pub use voxeldust_types::shard::*;
