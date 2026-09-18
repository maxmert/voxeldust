//! ★ THE WORLD TAG's DECLARED HALF (ruling V6 Part D; ruling V9 S5-2: tolerance ZERO).
//!
//! `GENERATOR_VERSION` is bumped by hand on ANY edit that moves one byte of one chunk: a new octave,
//! a changed constant, a reordered sum. Bumping it opens a world epoch: every store labelled with the
//! old tag refuses to open under the new one, and every saved edit is rebuilt against the new recipe,
//! never laid over a surface that slid under it. Detail that must not open an epoch is added on the
//! client as style, which never moves the shape.
//!
//! The declared half rides the store stamp, the mesh handshake, the realm's surface tag and the
//! client's world hello. The MEASURED half (`golden_self_check`) rides the client's hello only.
//!
//! **Example.** An artist wants craggier cliffs after launch. Craggy textures and small rocks are
//! style: no bump. A steeper ridge curve moves bytes: the version bumps, a new epoch opens, and every
//! moon's tunnels are rebuilt against the new ridges before anybody lands.

use vd_seed::digest::{FNV_OFFSET, fnv1a_u64};

/// The recipe's version. 1 was the first recipe of the voxel foundation (2026-09-08), on fenced
/// 64-bit floats. 2 is THE INTEGER RECIPE (ruling F7, 2026-09-12): the same world, computed on
/// fixed-point integers, which moved every surface by up to 1.5 mm and a quarter of a millimetre on
/// average (MEASURED over the 3 936 256 columns of the spike's square). ★ 3 IS THE EXTENDED LADDER
/// (owner, 2026-09-15): the top rung became ONE CHUNK per face edge, so every body's cell count
/// snaps to a coarser unit and every body's radius moves by up to 1.6 % — the home planet by
/// 28 683 m. Every address moved with it, so every chunk of every body is a new byte.
/// ★ 4 IS THE LANDFORM ARC's FIRST TWO SLICES (owner, 2026-09-16..18; rulings T1–T9): 8a — the
/// slope spectrum on the fine octaves, the ridged middle band, the per-column roughness factor on
/// its placeholder, the cap-rock bench, the wavelength survival rule with its stated alias, and the
/// crossfade under the ridge — and 8b — the relief law read from the body's own charter (the home
/// planet's mountains halved to 8 276 m). The charter's other words (the spin, the tilt, the derived
/// pressure, the greenhouse, the crust, the water, the sea) are STATED and read by no kernel yet, so
/// they move no byte; the recipe's sea keeps its draw until 8c (ruling T8).
pub const GENERATOR_VERSION: u32 = 4;

/// The declared world tag: the recipe's version folded with the universe seed.
#[must_use]
pub fn declared_world_tag(universe_seed: u64) -> u64 {
    fnv1a_u64(
        fnv1a_u64(FNV_OFFSET, u64::from(GENERATOR_VERSION)),
        universe_seed,
    )
}

/// ★ THE WORLD IDENTITY a process serves, the two numbers a client states at login (SL10 clause 3):
/// the DECLARED half, a constant, and the MEASURED half, the home body's eight golden chunks
/// evaluated by this binary on this chip. The gateway holds its own pair and refuses a client whose
/// pair differs, by name.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct WorldIdentity {
    pub declared: u64,
    pub measured: u64,
}

impl WorldIdentity {
    /// This binary's identity for `universe_seed`, measured on `home`; `None` when the home body
    /// cannot be self-checked (a refusal, never a fold of zeros).
    #[must_use]
    pub fn of(universe_seed: u64, home: &crate::body::BodyDefinition) -> Option<WorldIdentity> {
        Some(WorldIdentity {
            declared: declared_world_tag(universe_seed),
            measured: crate::digest::golden_self_check(home)?,
        })
    }
}

#[cfg(test)]
mod tests {
    //! ★ A TEST MAY DIVIDE (ruling F7's rule is about the SHIPPED path, not the measurement): a test
    //! states the exact quotient a reciprocal stands for, and a fixture picks its sample columns with a
    //! remainder. Neither runs in a kernel.
    #![allow(
        clippy::integer_division,
        clippy::modulo_arithmetic,
        reason = "a test states an exact quotient or picks a sample column; never a kernel's path"
    )]
    use super::*;

    #[test]
    fn the_declared_tag_folds_the_version_and_the_seed_and_nothing_else() {
        assert_eq!(declared_world_tag(2298), declared_world_tag(2298));
        assert_ne!(declared_world_tag(2298), declared_world_tag(2299));
        assert_ne!(declared_world_tag(2298), FNV_OFFSET);
        assert_eq!(
            GENERATOR_VERSION, 4,
            "bump by hand on any output-changing edit, and say so"
        );
        assert_eq!(
            declared_world_tag(2298),
            DECLARED_PIN,
            "the home world's declared tag"
        );
    }

    #[test]
    fn the_world_identity_pairs_the_declared_tag_with_the_measured_self_check() {
        let home = crate::home::home_planet();
        let id = WorldIdentity::of(2298, &home).expect("the home planet self-checks");
        assert_eq!(id.declared, DECLARED_PIN);
        assert_eq!(Some(id.measured), crate::digest::golden_self_check(&home));
        assert_eq!(
            Some(id),
            WorldIdentity::of(2298, &home),
            "the same binary, the same pair"
        );
    }

    const DECLARED_PIN: u64 = 4_196_931_793_486_802_419;
}
