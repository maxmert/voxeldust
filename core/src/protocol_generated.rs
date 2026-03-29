// Re-export the flatbuffers-generated code.
// This module exists so the rest of the codebase imports from a stable path
// regardless of how flatc organizes its output.
#[path = "voxeldust_generated.rs"]
mod inner;

pub use inner::voxeldust::protocol::*;
