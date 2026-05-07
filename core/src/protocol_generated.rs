// Re-export the FlatBuffers wire types from the dedicated `voxeldust-protocol-fb`
// crate. The generated source itself lives at `protocol-fb/src/voxeldust_generated.rs`
// — split out so schema regen no longer cascades through every workspace crate.
//
// This thin wrapper preserves the existing import path:
//
//     use crate::protocol_generated as fb;
//
// for the 5 sites in `core/` that already use it (shard_message, client_message,
// stellar, geophysics, planet_rotation).

pub use voxeldust_protocol_fb::voxeldust::protocol::*;
