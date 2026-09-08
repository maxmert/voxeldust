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

/// The recipe's version. 1 is the first recipe of the voxel foundation (2026-09-08).
pub const GENERATOR_VERSION: u32 = 1;

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
    use super::*;

    #[test]
    fn the_declared_tag_folds_the_version_and_the_seed_and_nothing_else() {
        assert_eq!(declared_world_tag(2298), declared_world_tag(2298));
        assert_ne!(declared_world_tag(2298), declared_world_tag(2299));
        assert_ne!(declared_world_tag(2298), FNV_OFFSET);
        assert_eq!(
            GENERATOR_VERSION, 1,
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

    const DECLARED_PIN: u64 = 10_807_444_098_716_102_726;
}
