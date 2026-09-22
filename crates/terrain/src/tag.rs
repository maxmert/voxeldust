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
/// 7 (2026-09-20): the far view's level pick — a rung reads the finest pyramid level whose node is at
/// least its cell (`PyramidField::level_for`), so every chunk from rung 10 up moved; the recipe's own
/// relief did not (the chunk golden tables stand).
/// ★ 8 (2026-09-21): THE PER-COLUMN ROUGHNESS FACTOR READS THE SOLVED FIELD'S OWN SLOPE. The factor
/// is now the GREATER of the placeholder noise's reading and the macro field's own slope as a share
/// of the body's first fine octave's slope (`artifact::slope_share`), so a mountain belt the solve
/// raised keeps its fine octaves whole. Every chunk ON THE ARTIFACT PATH whose share stands over its
/// noise factor moved; the recipe's own relief — a body with no artifact — did not, so the chunk
/// golden tables stand and the identity's measured half moves.
/// ★ 9 (2026-09-21): EACH BED GETS ITS OWN HARDNESS (slice 8d step 1; ruling W1). The cap-rock
/// bench's body-wide strength became a CEILING: the pull toward a bed top is that top's own drawn
/// share of it, and half the tops draw soft and pull nothing. EVERY chunk a bed top crosses moved,
/// so the recipe's own relief moves too and the chunk golden tables move with it.
/// ★ 10 (2026-09-21): THE ROCK MAP AND THE SUBSTANCE AT A FIXED RADIUS (slice 8d step 2; ruling W4
/// item 4). Every node row carries a PROVINCE byte, so the artifact's version bumps to 5 and a
/// version-4 store re-solves. Inside the veneer a cell's substance is no longer the body's one
/// sediment by depth: it is the rock of the BED at the cell's own radius, drawn from the province's
/// own four. No surface moved — the shape's arithmetic is untouched — but the SUBSTANCE of every
/// cell of the veneer's deepest band moved, so every chunk's digest moved and the golden tables move
/// with it.
/// ★ 11 (2026-09-21): THE SEA DECIDES THE SHORE (ruling W6). After the bench a column's surface is
/// held on its GROUND's side of its water by a quarter of the ground's own height over or under it
/// (`vd_recipe::height::shore`), so the shoreline stands where the solved ground crosses the water
/// at every rung and a ring swap cannot move it. Every column near a coast whose fine octaves
/// crossed the water moved; a body with no sea is untouched, so the seed-only golden tables stand
/// and only the measured identity may move.
/// ★ 12 (2026-09-22): THE LAKES AND THE CONTINENTS (ruling W7, the owner's three steps). The
/// flood every pass; the sediment laid in the first hollow downstream (`MacroSolve::deposit`), the
/// rebound reading the net; the freeboard law (`land::freeboard_factor`: the continental thickness
/// solved so the sea stands at the crust's shelf quantile, Earth's 29 of 40); the running sea
/// re-solved at every climate step; the erodibility by the province's rocks' tensile strength;
/// the envelope a per-node clamp, never a global scale. The SOLVE moved, so the artifact and the
/// identity's measured half move; the seed-only recipe did not, so the chunk golden tables stand.
/// ★ 13 (2026-09-22): THE COAST MASK (ruling W10; the owner, from 1 400 km: "during flight the
/// shores changes again all the time"). The SIDE of the water a column stands on is the fine row's
/// own word at EVERY rung — one bit per macro node, stored with the artifact and shipped with it —
/// and no host derives a side from a pyramid level's mean any more (`vd_recipe::height::shore`
/// takes the side; `artifact::sample_side` reads it). Every column at a rung that reads a level
/// and whose level mean disagreed with its own row moved; a chunk with no artifact reads the
/// unknown word and is untouched, so the seed-only golden tables stand and only the measured
/// identity may move. `ARTIFACT_VERSION` 6 with it: a version-5 store re-solves.
pub const GENERATOR_VERSION: u32 = 13;

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
    pub fn of(
        universe_seed: u64,
        home: &crate::body::BodyDefinition,
        fields: Option<&crate::artifact::GoldenFields>,
    ) -> Option<WorldIdentity> {
        Some(WorldIdentity {
            declared: declared_world_tag(universe_seed),
            measured: crate::digest::golden_self_check(home, fields)?,
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
            GENERATOR_VERSION, 13,
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
        let fields = crate::home::home_golden_fields();
        let id =
            WorldIdentity::of(2298, &home, Some(&fields)).expect("the home planet self-checks");
        assert_eq!(id.declared, DECLARED_PIN);
        assert_eq!(
            Some(id.measured),
            crate::digest::golden_self_check(&home, Some(&fields))
        );
        assert_eq!(
            Some(id),
            WorldIdentity::of(2298, &home, Some(&fields)),
            "the same binary, the same pair"
        );
        // ★ THE MEASURED HALF IS PINNED (slice 8c stage C4c): the eight chunks read through the
        // golden fields fold to this word on every host, or the world hello is refused by name.
        assert_eq!(id.measured, crate::home::HOME_IDENTITY_MEASURED);
        // Without the fields the eight chunks are the recipe's own relief: another word.
        let bare = WorldIdentity::of(2298, &home, None).expect("the recipe self-checks");
        assert_ne!(bare.measured, id.measured);
    }

    const DECLARED_PIN: u64 = 3_618_195_234_959_425_002;
}
