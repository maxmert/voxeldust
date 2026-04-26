//! Per-planet geophysical state derivation. Pure-functional; called exclusively
//! by server-shards (system-shard at system bootstrap; planet-shard receives
//! the result through orchestrator-mediated spawn config, not by re-deriving).
//! The client links this module but never invokes the `from_*` constructors —
//! `tests/no_client_seed_derivation.rs` enforces this.
//!
//! Phase 3 fills in `PlanetGeophysicalState` and `PlanetGeophysicalState::from_seed_and_star`,
//! which derives surface pressure, composition mix, Rayleigh/Mie/absorption
//! coefficients, scale height, gravity, and equilibrium temperature from the
//! planet seed plus parent-stellar context. This skeleton exists so that
//! Phase 0's determinism harness can scan the file path.
