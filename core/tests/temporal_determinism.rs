//! Temporal-determinism integration tests for the celestial-physics pure
//! functions in `core::stellar`, `core::geophysics`, and
//! `core::planet_rotation`.
//!
//! Two contracts are verified:
//!
//! 1. **Run-to-run determinism**: calling `f(args)` twice within the same
//!    process produces bit-identical outputs. This guards against accidental
//!    use of non-deterministic operations (`HashMap` iteration, OS RNG, etc.).
//!
//! 2. **Different inputs produce different outputs**: a sanity check that
//!    the function actually depends on its inputs (catches stub
//!    implementations that ignore parameters).
//!
//! Phase 1 covers `core::stellar`. Phases 3 and 4 will extend with
//! `PlanetGeophysicalState::from_seed_and_star` and
//! `PlanetRotationParams::rotation_at` respectively.

use voxeldust_core::galaxy::StarClass;
use voxeldust_core::stellar::StellarState;

#[test]
fn stellar_state_is_run_to_run_deterministic() {
    let class = StarClass::G;
    let seed = 0xC0FFEE_DEAD_BEEFu64;

    let a = StellarState::from_class_and_seed(class, seed);
    let b = StellarState::from_class_and_seed(class, seed);

    assert_eq!(
        a, b,
        "StellarState::from_class_and_seed must be bit-identical \
         for identical inputs (class={:?}, seed=0x{:x})",
        class, seed
    );
}

#[test]
fn stellar_state_varies_with_seed() {
    let a = StellarState::from_class_and_seed(StarClass::G, 0x1111);
    let b = StellarState::from_class_and_seed(StarClass::G, 0x2222);

    // Same class, different seeds: at least mass, T, L should differ
    // (variance from the within-class seed-driven mass roll).
    assert_ne!(a.mass_solar, b.mass_solar, "different seeds → different masses");
    assert_ne!(a.temperature_k, b.temperature_k, "different seeds → different T_eff");
    assert_ne!(a.luminosity_w, b.luminosity_w, "different seeds → different L");
}

#[test]
fn stellar_state_varies_with_class() {
    let g = StellarState::from_class_and_seed(StarClass::G, 0xAAAA);
    let m = StellarState::from_class_and_seed(StarClass::M, 0xAAAA);

    // Same seed, different class: M-class is much smaller than G-class.
    assert!(m.mass_solar < g.mass_solar, "M-class mass < G-class mass");
    assert!(m.temperature_k < g.temperature_k, "M-class T < G-class T");
    assert!(m.luminosity_w < g.luminosity_w, "M-class L < G-class L");
}

// Wire-level serde roundtrip is exercised in `core::client_message`'s
// `tests::roundtrip_*` suite, which goes through FlatBuffers — the canonical
// transport. No need to duplicate it with a JSON roundtrip here.
