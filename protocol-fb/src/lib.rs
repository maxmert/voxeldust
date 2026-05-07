// FlatBuffers wire types — sole purpose of this crate is to be the compile
// firewall around the generated code. See BUILD_PERF.md.
//
// The generated file declares its own `pub mod voxeldust { pub mod protocol { … } }`,
// so we surface it as `voxeldust::protocol::*` for downstream consumers.

#![allow(unused_imports, dead_code, deprecated, clippy::all)]

#[path = "voxeldust_generated.rs"]
mod inner;

pub use inner::voxeldust;
