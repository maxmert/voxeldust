//! Per-planet rotation parameters and the closed-form rotation function.
//!
//! Two contracts:
//!   1. `PlanetRotationParams::from_seed_and_state(...)` is a pure derivation
//!      called by the server (system-shard) once at system bootstrap. The
//!      result is broadcast to clients in `CelestialBodySnapshot.rotation_params`.
//!   2. `rotation_at(params, game_time_s)` is a pure function called by both
//!      the server (when rendering authoritative rotation) and the client
//!      (each frame, with the extrapolated `game_time_s`). Bit-for-bit
//!      identical inputs produce bit-for-bit identical quaternions, which is
//!      the basis of the temporal-correctness guarantee.
//!
//! Phase 4 fills in the body. This skeleton exists so that Phase 0's
//! determinism harness can scan the file path.
