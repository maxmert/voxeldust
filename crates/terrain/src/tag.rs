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
/// ★ 14 (2026-09-22): THE ROUTING FILL IS NOT WATER (ruling W11; the report
/// `docs/investigation/2026-09-22/lakes_and_landscape_models.md` §6.1). The priority flood's
/// raised surface is a SCRATCH SURFACE: it decides the receivers, the flats, the order and the
/// sweep's base level, and nothing else. A hollow is a lake only where a finite amount of water
/// stands in it — the depression hierarchy of Fill–Spill–Merge (Barnes, Callaghan & Wickert 2021)
/// over the final field, and every hollow's own water balance (Langbein 1961) against the rain and
/// the potential evaporation the climate already computes. `MacroSolve::water_level` and
/// `MacroSolve::facies` read the budget's level, so the artifact's `water_m` row, its facies byte
/// and the pyramid's water words move on every hollow the rain cannot fill. The SHAPE did not move
/// — no sweep, no deposit, no ice and no envelope changed — so the seed-only chunk tables stand and
/// only the artifact's own words and the measured identity move.
/// ★ 15 (2026-09-22): THE SUMMER ICE LINE AND THE CRATER RECORD OF A WET SURFACE (ruling B2 step
/// 2; the report `docs/investigation/2026-09-22/lakes_and_landscape_models.md` §6.2 and §6.4).
/// Four laws moved, and every one of them moves the solved field, so the artifact and the
/// identity's measured half move with it; the seed-only recipe is untouched, so the chunk golden
/// tables stand.
/// * THE TEMPERATURE'S DATUM is the body's own MEAN SURFACE — the sea where there is one — and no
///   longer the LADDER RADIUS, a geometric datum the home planet's sea stands 3.4 km above. The
///   sea surface read −11 °C and the rain fell to a sixth of Earth's.
/// * THE AIR OVER WATER stands at the WATER's surface: the abyssal plain read 344 K, asked for
///   16 160 mm of evaporation a year and was given 6 708 mm of rain.
/// * THE EQUILIBRIUM LINE IS A SUMMER LINE. The season comes from the charter's own obliquity
///   through North's insolation expansion and a one-mode seasonal energy balance; the line stands
///   where the ablation-season temperature falls to the one Ohmura's 70 glaciers name for the
///   node's own rain. The ICE is Egholm's mass balance carried down the receiver tree, so no
///   extent is drawn anywhere; the ice and the talus now run INSIDE the pass loop, before every
///   sweep and every deposit, so the rivers answer what they cut.
/// * THE CRATER RECORD reads each province's own CRATER RETENTION AGE — the plate that carries the
///   ground and the rain that wears it — instead of the system's whole age. The home planet keeps
///   1 610 craters instead of 144 967; the airless moon's count does not move.
///
/// ★ 16 (2026-09-22): THE FLAT'S DISTANCE IS IN METRES, NOT IN HOPS (ruling B2 step 3; the report
/// `docs/investigation/2026-09-22/lakes_and_landscape_models.md` §3.4 and §6.3; Cordonnier, Bovy &
/// Braun 2019 §2.3.2). The priority flood turns every hollow into a FLAT, and the flat's receivers
/// were assigned by a breadth-first HOP COUNT, which charges the stencil's diagonal — √2 of a row —
/// the same one step as a row. So the cheapest way across every lake on the planet ran on one fixed
/// diagonal, and the receiver tree drew the scratches the owner flew over. The distance is now the
/// lattice's OWN CHORD IN WHOLE METRES summed along the path, and a flat node takes the neighbour
/// with the smallest `distance + chord`, ties to the smaller index. MEASURED on the home planet:
/// the valleys' long-axis flatness against the grid's four directions fell from 2.101 to 1.284 and
/// the 135° bin from 50 840 of 96 804 trunks to 23 953 of 96 893; on a filled disc the water's
/// detour out of the flat fell from 1.2406 to 1.0461 and its bearing error from 2.82° to 1.04°.
/// The receivers move, so the sweep, the deposit, the ice and the lakes move with them: the
/// artifact and the identity's measured half move. A field with NO flat routes byte for byte as
/// before (a test that could have failed), so the seed-only chunk tables stand.
/// ★ 17 (2026-09-22): THE DRAWN DRAINAGE (ruling B2 step 4; the owner, at 100–150 km: the plains
/// *"look like dunes"*). The fine relief under one macro node was a sum of noise octaves and
/// NOTHING DRAINED IT, so from the air it read as ripples. Now the solve's own river network —
/// the receiver slot and the discharge class the row has carried since 2026-09-19 and nothing read
/// back — is taken DOWN into the fine rungs (`crate::river`): every node's trunk line, four orders
/// of tributaries under it by Horton 1945's ratios, a channel from Leopold & Maddock 1953's width
/// law, a floodplain from Leopold & Wolman 1960's meander belt, and a valley half as wide as the
/// spacing between two streams of one order. A column inside a valley loses its fine octaves (the
/// roughness factor takes a CEILING, never a second factor) and gains the channel's trench; a
/// column on a divide keeps them whole. The relief is ARRANGED, never added.
///
/// EVERY chunk that reads an artifact and stands within a valley moved, so the artifact path's
/// chunks and the identity's measured half move. THE SOLVE DID NOT MOVE — not one pass, not one
/// row — so the artifact's own digest, the sea, the ocean share and the golden `Z` fields stand;
/// and a chunk with NO artifact reads a ceiling of one and a cut of zero, whose arithmetic is
/// exactly the one it ran before, so THE SEED-ONLY CHUNK TABLES STAND and the card's own column
/// pass is untouched (`just gpu-drift` compares like with like).
/// ★ 18 (2026-09-22): THE COAST MASK GETS A FOOTPRINT (ruling W15; the owner, from 41 000 km: the
/// globe *"shows squares of water on the land"*, and the same ground flips between water and land
/// as the rings sweep under a moving hull). Ruling W10 made the water's SIDE a BIT of the fine
/// row, read at every rung — and read it at the ONE fine node nearest the cell's centre. At rung 18
/// a cell is 262 km and covers about a thousand fine nodes, so one node in a thousand painted the
/// whole cell, and each rung sampled another centre node, so a cell flipped at every ring swap.
/// NOW a cell's side is the WET FRACTION of the fine nodes under its WHOLE FOOTPRINT — wet where
/// at least half of them are wet, on whole counts (`artifact::CoastCounts`, folded from the mask
/// by every host and shipped by none, so the artifact's own digest does not move). A parent's
/// count is the sum of its four children's, so a coarse cell shows the side most of its ground
/// stands on and the finer rung refines that edge instead of contradicting it. Every column at a
/// rung whose CELL IS WIDER THAN A MACRO NODE moved; the rungs under that read one node, as they
/// always did, so a pilot's own ground did not move. A chunk with no artifact reads the unknown
/// word and is untouched, so the seed-only golden tables stand and the card's bytes do not move.
/// ★ 19 (2026-09-23): ONE SHAPE AT EVERY DISTANCE (ruling W16; the owner, after flying: *"when I
/// fly over the water very close, it changes from water to surface and back … when I'm flying away
/// too far, it also changes … at some point water is not visible at all, just land"*). Three
/// changes to the shape, in one version:
///
/// 1. **STEP 4'S DRAWN RIVERS ARE RETIRED** (version 17's own note above). The stamped trunk and
///    tributary lines, the valley profile, the channel trench and the per-column river water word
///    are GONE, and the fine relief is the recipe's octaves under the roughness factor again.
///    MEASURED as the near flicker (ruling W15 §3): a stream's surface was written only where a
///    rung drew the valley at FULL strength, so one wet column in sixteen appeared or vanished at
///    ONE ring swap and the water's edge jumped up to 5.4 km. Every column that stood inside a
///    drawn valley moves back toward version 16's ground.
/// 2. **THE COAST MASK CARRIES A THREE-WAY SIDE** — land, sea or LAKE (`ARTIFACT_VERSION` 7).
///    MEASURED before it (`water_edge_step lake`): not ONE column of 300 000 stood under a lake's
///    own surface at any rung, because a lake node's bit was clear, the shore law read `SIDE_LAND`
///    and held every lake column at least a quarter of its ground's height ABOVE its own water.
///    Every column standing on a lake moves DOWN under its lake.
/// 3. The water SHEET is no longer a flat eight-cell block where its own chord would dive through
///    the ground — a DRAWING law, so it moves no column of the field, and it is named here only
///    because it ships in the same breath.
///
/// A chunk with NO artifact reads the unknown side and no drainage, so its arithmetic is the one
/// the card has always run: the SEED-ONLY golden tables stand and `gpu-drift` compares like with
/// like. The SOLVE did not move, so the sea, the ocean share and the rows stand; the artifact's own
/// DIGEST moves, because the mask's bytes double and carry the lake.
pub const GENERATOR_VERSION: u32 = 19;

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
            GENERATOR_VERSION, 19,
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

    const DECLARED_PIN: u64 = 11_554_407_130_392_614_244;
}
