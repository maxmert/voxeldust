//! Backward-compatible re-export shim.
//!
//! The `BlockId` newtype + named constants live in `voxeldust-types` so that
//! `voxeldust-signal` (which matches on `BlockId::COCKPIT` etc.) can reach
//! them without depending on `voxeldust-core`. This file preserves the
//! existing `use voxeldust_core::block::BlockId` import paths.

pub use voxeldust_types::block_id::*;
