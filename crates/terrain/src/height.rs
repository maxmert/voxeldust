//! ★ THE HEIGHT FIELD — where the surface is along one direction, at one rung: the ladder radius plus
//! the sum of the live octaves of noise, ON INTEGERS (ruling F7). At rung `L` the `L` finest octaves
//! are dropped, so the far view is the same hill with the small bumps left out, cheaper by exactly
//! those octaves. The biome of a column reads the same field and two slow noises.
//!
//! **The unit.** [`height`] answers in GAP STEPS at [`crate::units::LENGTH_BITS`] fraction bits — the
//! radius word plus the octave sum, added without a shift because both carry the same bits, and
//! floored ONCE by the density (`chunk::finish_cell`). [`height_m`] and [`biome_at`] are the same two
//! answers for a host that holds its direction as floats: the doors of [`crate::units`] on either side
//! of the integer field, and the only floats this module names.
//!
//! ★ **THE CAP-ROCK BENCH** (slice 8a stage 4) stands AFTER the sum: the column's surface is pulled
//! toward the nearest BED TOP, and a bed top stands at a FIXED RADIUS, so a bench runs along a whole
//! hillside at ONE HEIGHT. Its strength fades with the rung and is zero where a tread falls under
//! four cells, so a coarse rung draws the hill without it — which is why the dropped-octave bound
//! below gained the terrace's own Lipschitz factor and its fade's own step.
//!
//! ★ **THE PER-COLUMN ROUGHNESS FACTOR** (slice 8a stage 3) stands inside the sum: the FINE octaves
//! are multiplied ONCE by `m ∈ [M_MIN, 1]`, which the column's own slow field draws. So a plain is
//! flat and a range is rough out of one table, and the factor never depends on the rung — which is
//! why the dropped-octave bound below still holds exactly.
//!
//! ★ **AND ON A BODY THAT SHIPS AN ARTIFACT THE SOLVED FIELD DECIDES TOO** (2026-09-21): the factor
//! is the GREATER of that noise field's reading and the macro field's own slope as a share of the
//! body's slope reference ([`crate::artifact::slope_share`]). The ceiling is still one, so the band
//! and the dropped-octave bound below are untouched. [`height`] — the recipe's own relief, with no
//! artifact — reads the noise field alone.
//!
//! ★ **AND THE SEA DECIDES THE SHORE** (2026-09-21): after the bench, the surface is held on its
//! GROUND's side of the column's water — the ground being the radius, the field's `Z` and the coarse
//! octaves, which every rung shares — by a quarter of the ground's own height over or under it
//! ([`vd_recipe::height::shore`]). So the shoreline stands where the ground crosses the water at
//! every rung, and a ring swap cannot move it; before, the dropped octaves moved it by hundreds of
//! metres on a gentle coast. The clamp is a projection onto one side of the water, which never
//! widens a difference, so the dropped-octave bound below still holds.
//!
//! ★ **AND THE ROW SAYS WHICH SIDE** (2026-09-22, the coast mask; ruling W10): at a PYRAMID rung the
//! ground above is a LEVEL'S MEAN, and a mean crosses the sea somewhere else than its children do, so
//! the side comes from the fine node's own sea bit ([`crate::artifact::sample_side`], read on the
//! body's own macro lattice at every rung) and never from the ground's sign. MEASURED before it: the
//! crossing moved a median of 11.5 km at the swap from the rows to level 1 and about 20 km at each
//! level swap above. A host with no mask — the card, a body with no artifact — reads
//! [`vd_recipe::height::SIDE_UNKNOWN`] and the ground decides, as before.
//!
//! **Example.** Along the direction of the pilot's boots the field at rung 3 is the rung-0 hill
//! without the last three ripples, and the two are never further apart than the dropped amplitudes
//! promise (the test below measures it on the home planet).

use crate::body::BodyDefinition;
use crate::strata::Biome;
use crate::units::{direction_of_unit, metres_of_q28, q28_of_metres};
use vd_recipe::Gi;
use vd_recipe::height::{
    SIDE_UNKNOWN, biome_of as recipe_biome, relief_parts_from, roughness_factor, shore,
};
use vd_recipe::terrace::terrace;

/// The surface's radius along a unit direction at a rung, in GAP STEPS at
/// [`crate::units::LENGTH_BITS`] fraction bits. `dir` carries the bend's 40 fraction bits.
#[must_use]
pub fn height(body: &BodyDefinition, dir: [Gi; 3], rung: u8) -> Gi {
    // ★ THE BENCH, THEN THE SHORE (slice 8a stage 4; 2026-09-21): the octave sum answers the raw
    // surface, the terrace pulls it toward the nearest bed top, and the shore holds it on the
    // coarse ground's side of the body's sea. ONE SOURCE — `vd_recipe::terrace::terrace` and
    // `vd_recipe::height::shore` are the very functions the card's column pass runs, so a picture
    // and a pair of boots stand on one bench and one shore.
    let parts = relief_parts_from(
        body.octave_table(),
        0,
        body.octaves_at(rung).len(),
        dir,
        body.roughness(),
        Gi::ZERO,
    );
    let base = body.radius + parts.coarse;
    // No artifact here, so no coast mask: the ground's own sign decides the side, which is the
    // rule the card runs too.
    shore(
        base,
        body.sea_radius,
        terrace(&body.terrace_at(rung), base + parts.fine),
        SIDE_UNKNOWN,
    )
}

/// THE FLOAT SEAM of the height field: the surface's radius in METRES along a direction a host holds
/// as floats — a camera's radial, a ray from a pilot's boots. The direction enters through
/// [`crate::units::direction_of_unit`] and the answer leaves through [`crate::units::metres_of_q28`];
/// the shape between them is the integer recipe's.
#[must_use]
pub fn height_m(body: &BodyDefinition, dir: [f64; 3], rung: u8) -> f64 {
    metres_of_q28(height(body, direction_of_unit(dir), rung))
}

/// ★ THE HEIGHT WITH THE MACRO FIELD (slice 8c stage C4c): the surface's radius in metres along a
/// float direction at a rung when the host holds an artifact — `Z` read off the field at the rung-0
/// cell the direction falls in (within half a metre sideways, which the smooth macro field cannot
/// tell apart), plus the FINE octaves the rung keeps, then the bench: the column kernel's own sum
/// (`vd_recipe::plan::column_surface_from`) along a direction that is nobody's cell centre. `None`
/// where the field holds no row for the stencil (the coarser rung stands, ruling F9).
///
/// The client's geomorph reads this for a vertex with no parent triangle on its radial, so the
/// vertex morphs toward the artifact's coarser surface and never toward the recipe's own relief,
/// which the artifact replaced.
#[must_use]
pub fn height_field_m(
    body: &BodyDefinition,
    field: &dyn crate::artifact::ZField,
    dir: [f64; 3],
    rung: u8,
) -> Option<f64> {
    let fine = body.macro_lattice()?;
    let lattice = fine.coarser(field.level())?;
    let face = vd_seed::bend::face_of(dir);
    let (t, s) = vd_seed::bend::face_coords(face, dir);
    let n0 = body.ladder().cells_per_edge(0);
    let i = vd_seed::ladder::index_of(vd_seed::bend::unbend(t), n0);
    let j = vd_seed::ladder::index_of(vd_seed::bend::unbend(s), n0);
    let z = crate::artifact::sample_z(&lattice, field, face, 0, i, j)?;
    // ★ THE SAME SLOPE SHARE THE COLUMN READS (2026-09-21): the morph target and the chunk's own
    // column must stand at one height, so this reads the field the same way — the central difference
    // over one macro node at the SAME rung-0 cell the `Z` above came from.
    let slope_share = crate::artifact::slope_share(
        &crate::artifact::slope_charter(body, &lattice, 0),
        &lattice,
        field,
        face,
        0,
        i,
        j,
    )?;
    // ★ THE SAME WATER THE COLUMN READS (2026-09-21): the nearest row's level through the one
    // reader, so the morph target and the chunk's own column hold one shore.
    let (water, _) = crate::artifact::sample_water(body, &lattice, field, face, 0, i, j);
    // ★ THE SAME SIDE THE COLUMN READS (2026-09-22, the coast mask): the FINE node's own sea bit,
    // read on the body's own macro lattice whatever level the field stands at, so the morph target
    // and the chunk's own column stand on one side of one shoreline.
    let side =
        crate::artifact::sample_side(&fine, field, face, 0, i, j).map_or(SIDE_UNKNOWN, |sea| {
            if sea {
                vd_recipe::height::SIDE_SEA
            } else {
                vd_recipe::height::SIDE_LAND
            }
        });
    let parts = relief_parts_from(
        body.octave_table(),
        body.first_fine(),
        body.octaves_at(rung).len(),
        direction_of_unit(dir),
        body.roughness(),
        slope_share,
    );
    let base = body.radius + z + parts.coarse;
    Some(metres_of_q28(shore(
        base,
        water,
        terrace(&body.terrace_at(rung), base + parts.fine),
        side,
    )))
}

/// ★ THE PER-COLUMN ROUGHNESS FACTOR as a real number in `[M_MIN, 1]`, for a host outside the recipe
/// — the slope histogram's instrument reads it to say which columns are a PLAIN and which a RANGE
/// (slice 8a stage 3, measurement M-C). The shape is the recipe's own kernel; this crate only names
/// the answer, so an instrument can never measure a factor the field does not use.
///
/// ★ WITH NO FIELD THIS IS HALF THE ANSWER (2026-09-21). A column that reads an artifact takes the
/// GREATER of this factor and the solved field's own slope share, so an instrument that holds a
/// field must ask [`roughness_field_at`]; this one answers the noise placeholder's factor alone,
/// which is what a body with no artifact uses.
#[must_use]
pub fn roughness_at(body: &BodyDefinition, dir: [f64; 3]) -> f64 {
    crate::units::share_of_q28(roughness_factor(body.roughness(), direction_of_unit(dir)))
}

/// ★ THE PER-COLUMN ROUGHNESS FACTOR A COLUMN OF AN ARTIFACT ACTUALLY USES (2026-09-21): the GREATER
/// of the noise placeholder's factor and the solved field's own slope share at the same column, as a
/// real number in `[M_MIN, 1]`. The slope histogram reads this where it holds a field, so the
/// instrument can never print a factor the ground does not have.
///
/// `None` where the field holds no row for the column's stencil or its ring — the same refusal the
/// chunk builder makes (ruling F9).
///
/// **Example.** Over the home planet's highest belt the noise placeholder reads 0.31 and the solved
/// field's share reads 1.00, so the column's real factor is 1.00 and the instrument says RANGE.
#[must_use]
pub fn roughness_field_at(
    body: &BodyDefinition,
    field: &dyn crate::artifact::ZField,
    dir: [f64; 3],
) -> Option<f64> {
    let lattice = body.macro_lattice()?.coarser(field.level())?;
    let face = vd_seed::bend::face_of(dir);
    let (t, s) = vd_seed::bend::face_coords(face, dir);
    let n0 = body.ladder().cells_per_edge(0);
    let i = vd_seed::ladder::index_of(vd_seed::bend::unbend(t), n0);
    let j = vd_seed::ladder::index_of(vd_seed::bend::unbend(s), n0);
    let share = crate::artifact::slope_share(
        &crate::artifact::slope_charter(body, &lattice, 0),
        &lattice,
        field,
        face,
        0,
        i,
        j,
    )?;
    Some(crate::units::share_of_q28(crate::units::greater(
        roughness_factor(body.roughness(), direction_of_unit(dir)),
        share,
    )))
}

/// ★ THE CAP-ROCK BENCH as a real number, for a host outside the recipe (slice 8a stage 4): a
/// surface radius in METRES, pulled toward the nearest bed top at a rung. The skyline march reads it
/// so its own field is the field the world ships and not a field nobody draws; the shape between the
/// two doors is the recipe's own integer kernel, so an instrument can never measure a bench the
/// ground does not have.
#[must_use]
pub fn terrace_m(body: &BodyDefinition, surface_m: f64, rung: u8) -> f64 {
    metres_of_q28(terrace(&body.terrace_at(rung), q28_of_metres(surface_m)))
}

/// THE POLE AXIS of every body: `+Z` in the body's own frame — the axis the world's orbits turn
/// about (`crates/physics/src/celestial.rs`: the perifocal plane is `z = 0`), so ice caps face away
/// from the orbital plane, never into it. An obliquity the parent authors per body is a later
/// slice; until then every body's spin axis is its orbit's axis. Cross-pinned in
/// `crates/bins/tests/home_body_pin.rs`. The recipe's own kernel reads the same axis, and the test
/// below measures that the two agree — a kernel that turned the poles would turn them here too.
pub const POLE_AXIS: usize = vd_recipe::height::POLE_AXIS;

/// The biome of a column: cold near the poles and high up, dry where the humidity noise says so, and
/// highland where the surface stands far above the sea. `surface` is the column's surface radius in
/// gap steps at [`crate::units::LENGTH_BITS`], as [`height`] answers it.
///
/// ★ ONE SOURCE (step G2-A): the arithmetic is `vd_recipe::height::biome_of`, the very kernel the
/// card's column pass runs, and this crate only gives the answer its NAME. So the shard's cell pass
/// and the card's picture can never disagree about a column's biome.
#[must_use]
pub fn biome_of(body: &BodyDefinition, dir: [Gi; 3], surface: Gi) -> Biome {
    biome_of_code(recipe_biome(&body.biome_charter(), dir, surface))
}

/// The biome a recipe code names. The codes are the enum's own discriminants, and the recipe's own
/// kernel answers nothing else — a code past the four would be the recipe changed without this
/// table, so it reads as grassland and the test below measures the four that exist.
pub(crate) fn biome_of_code(code: Gi) -> Biome {
    match code.raw() {
        0 => Biome::Desert,
        2 => Biome::Tundra,
        3 => Biome::Highland,
        _ => Biome::Grassland,
    }
}

/// THE FLOAT SEAM of the biome field: the biome of a column a host names with a float direction and a
/// surface radius in metres.
#[must_use]
pub fn biome_at(body: &BodyDefinition, dir: [f64; 3], surface_m: f64) -> Biome {
    biome_of(body, direction_of_unit(dir), q28_of_metres(surface_m))
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
    use vd_recipe::bend::DIR_BITS;
    use vd_seed::bend::Face;

    fn home() -> BodyDefinition {
        crate::home::home_planet()
    }

    /// ★ THE MOON WITH A SEA AND ITS OWN COAST MASK, the fixture both shore statements read.
    ///
    /// The moon's own words draw NO water, so its rows are dry and its solve sets no sea bit. The
    /// fixture therefore STATES a sea at the field's own height in the middle of a face and writes
    /// the mask that sea implies — a node's bit set where its row stands at or under that sea,
    /// which is exactly what a wet body's own solve writes. So the moon stands in for a coast
    /// without a second world (SL5): one body, real rows, a stated water line.
    ///
    /// Answers `(the moon with its sea, the lattice, the artifact, the face, the row `j`)`.
    fn moon_with_a_stated_sea() -> (
        BodyDefinition,
        crate::macro_lattice::MacroLattice,
        crate::artifact::Artifact,
        Face,
        i32,
    ) {
        use crate::home::{HOME_SYSTEM_AGE_YR, home_moon, home_moon_solve_words};
        use crate::solve::{Schedule, solve_full};
        let moon = home_moon();
        let lattice = moon.macro_lattice().expect("a lattice");
        let words = home_moon_solve_words();
        let (state, facies, _) =
            solve_full(&moon, &words, Schedule::standard(HOME_SYSTEM_AGE_YR)).expect("a solve");
        let climate = crate::climate::climate(&moon, &lattice, &words, &state.z, Some(state.sea_z));
        let mut artifact =
            crate::artifact::Artifact::of(&state, &facies, &climate, words.water_km3 > 0);
        let n0 = moon.ladder().cells_per_edge(0) as i32;
        let face = Face::PosX;
        let j = n0 / 2;
        let sea_m = metres_of_q28(
            crate::artifact::sample_z(&lattice, &artifact, face, 0, n0 / 2, j).expect("a z"),
        ) as i32;
        // `sample_z` answers the field's own `Z` — metres OVER the ladder radius, the unit a row
        // states — so the stated sea and the rows are already in one unit.
        let sea_row = sea_m;
        // The mask the stated sea implies: a node at or under it is sea, as `Artifact::of` writes
        // it from the solve's own facies on a wet body.
        for (node, row) in artifact.rows.iter().enumerate() {
            let bit = 1u8 << (node % 8);
            if i32::from(row.z_m) <= sea_row {
                artifact.coast[node / 8] |= bit;
            } else {
                artifact.coast[node / 8] &= !bit;
            }
        }
        (moon.with_sea_m(Some(sea_m)), lattice, artifact, face, j)
    }

    /// The float direction of the rung-0 cell `(i, j)` of a face on this body.
    fn cell_dir(body: &BodyDefinition, face: Face, i: i32, j: i32) -> [f64; 3] {
        let d = vd_seed::bend::direction_q(face, i, j, body.inv_n(0));
        [
            d[0].raw() as f64 / (1u64 << DIR_BITS) as f64,
            d[1].raw() as f64 / (1u64 << DIR_BITS) as f64,
            d[2].raw() as f64 / (1u64 << DIR_BITS) as f64,
        ]
    }

    /// The direction of a face position, through the recipe's own bend: the cell nearest `(a, b)` of
    /// a face at rung 0 (the test names positions, the recipe names cells).
    fn dir(face: Face, a: f64, b: f64) -> [Gi; 3] {
        let m = home();
        let n_l = m.ladder().cells_per_edge(0);
        let i = vd_seed::ladder::index_of(a, n_l);
        let j = vd_seed::ladder::index_of(b, n_l);
        vd_seed::bend::direction_q(face, i, j, m.inv_n(0))
    }

    /// ★ THE HEIGHT WITH THE MACRO FIELD (C4c): through the moon's artifact it answers along any
    /// direction, agrees with the column kernel's own surface at a cell centre to a step, reads a
    /// pyramid level at a coarse rung, and answers nothing through a cache missing the tile.
    #[test]
    fn the_height_reads_the_artifacts_field_along_a_direction() {
        use crate::CHUNK_EDGE;
        use crate::artifact::{PyramidField, TileCache};
        use crate::home::{HOME_SYSTEM_AGE_YR, home_moon, home_moon_solve_words};
        use crate::solve::{Schedule, solve_full};
        let moon = home_moon();
        let lattice = moon.macro_lattice().expect("a lattice");
        let words = home_moon_solve_words();
        let (state, facies, _) =
            solve_full(&moon, &words, Schedule::standard(HOME_SYSTEM_AGE_YR)).expect("a solve");
        let climate = crate::climate::climate(&moon, &lattice, &words, &state.z, Some(state.sea_z));
        let artifact =
            crate::artifact::Artifact::of(&state, &facies, &climate, words.water_km3 > 0);
        // A cell centre on face +X: the column kernel's surface (with the field) and the height
        // along that centre's own direction agree to the cell's rounding.
        let n0 = moon.ladder().cells_per_edge(0);
        let (i, j) = (2_000, 3_000);
        let column = crate::chunk::column_field(
            &moon,
            Some(&artifact),
            Face::PosX,
            0,
            i / CHUNK_EDGE as i32,
            j / CHUNK_EDGE as i32,
        )
        .expect("a column");
        let idx = ((j % CHUNK_EDGE as i32) * CHUNK_EDGE as i32 + (i % CHUNK_EDGE as i32)) as usize;
        let (dir_q, h, _) = column.columns[idx];
        let dir = [
            dir_q[0].raw() as f64 / (1u64 << DIR_BITS) as f64,
            dir_q[1].raw() as f64 / (1u64 << DIR_BITS) as f64,
            dir_q[2].raw() as f64 / (1u64 << DIR_BITS) as f64,
        ];
        let along = height_field_m(&moon, &artifact, dir, 0).expect("a height");
        let want = metres_of_q28(h);
        assert!((along - want).abs() < 0.02, "{along} vs {want}");
        let _ = n0;
        // A coarse rung through a pyramid level answers; a torn cache does not; a level the
        // lattice cannot coarsen to and a body with no macro lattice answer nothing.
        let level = PyramidField::of(&artifact, 1).expect("level 1");
        assert!(height_field_m(&moon, &level, dir, 12).is_some());
        let empty = TileCache::new(lattice.edge);
        assert_eq!(height_field_m(&moon, &empty, dir, 0), None);
        let level_9 = PyramidField {
            level: 9,
            z_m: vec![],
            water_m: vec![],
            coast: None,
        };
        assert_eq!(height_field_m(&moon, &level_9, dir, 12), None);
        let rock = moon.without_macro_lattice();
        assert_eq!(height_field_m(&rock, &empty, dir, 0), None);
        // The recipe's own height along the same direction differs from the field's.
        assert!((height_m(&moon, dir, 0) - along).abs() > 0.0);
        // ★ THE INSTRUMENT READS THE FACTOR THE COLUMN USES (2026-09-21): never under the noise
        // placeholder's own, never over the ceiling, and it refuses wherever the height does.
        let with_field = roughness_field_at(&moon, &artifact, dir).expect("a factor");
        let noise_only = roughness_at(&moon, dir);
        assert!(
            with_field >= noise_only,
            "the field's factor {with_field} stands under the noise's {noise_only}"
        );
        assert!(with_field <= 1.0, "over the ceiling: {with_field}");
        assert_eq!(roughness_field_at(&moon, &empty, dir), None);
        assert_eq!(roughness_field_at(&moon, &level_9, dir), None);
        assert_eq!(roughness_field_at(&rock, &empty, dir), None);
    }

    /// ★ THE SHORELINE STANDS STILL ACROSS THE RUNGS (2026-09-21; the owner, flying the coast:
    /// "the shores are changing all the time"). Along one line of directions across the moon's
    /// stated shore, every rung from the metre to the 128 m cell puts each column on the SAME side
    /// of its water as its own ROW says. Three controls, each of which could fail: the line really
    /// crosses a shore (columns on both sides); the row's own word decides the side (2026-09-22,
    /// the coast mask — before it the interpolated ground decided, and a coarse rung's ground is a
    /// mean); and the clamp really acted somewhere (a column stands exactly at its held bound at
    /// some rung — without the clamp that bound is nobody's number).
    ///
    /// MEASURED before the law, on the home planet's belt coast (`vd-bins/examples/shore_step`):
    /// the crossing moved a median of 234 m between the 64 m and 128 m rungs.
    #[test]
    fn the_shoreline_stands_where_the_ground_crosses_the_water_at_every_rung() {
        let (moon, lattice, artifact, face, j) = moon_with_a_stated_sea();
        let n0 = moon.ladder().cells_per_edge(0) as i32;
        assert_ne!(moon.sea_radius, Gi::ZERO, "the moon states a sea");
        // ★ THE SIDE IS THE ROW'S OWN WORD (2026-09-22): the nearest fine node's coast bit, which
        // is the one word every rung reads.
        let ground_side = |i: i32| -> Option<(bool, Gi, Gi)> {
            let (water, _) =
                crate::artifact::sample_water(&moon, &lattice, &artifact, face, 0, i, j);
            if water == Gi::ZERO {
                return None;
            }
            let sea = crate::artifact::sample_side(&lattice, &artifact, face, 0, i, j)?;
            let z = crate::artifact::sample_z(&lattice, &artifact, face, 0, i, j)?;
            Some((!sea, moon.radius + z, water))
        };
        let mut i = 64;
        let first = ground_side(i).expect("a column with water");
        while i < n0 - 64 && ground_side(i).is_some_and(|(land, _, _)| land == first.0) {
            i += 64;
        }
        assert!(i < n0 - 64, "the moon's middle row crosses a shore");
        // The sixty-five columns from the last column on the first side to the first column on
        // the other, at every rung the ladder draws near the ground.
        let mut lands = 0;
        let mut seas = 0;
        let mut at_bound = 0;
        let mut c = i - 64;
        while c <= i {
            let (land, base, water) = ground_side(c).expect("a column with water");
            lands += i32::from(land);
            seas += i32::from(!land);
            let dir = cell_dir(&moon, face, c, j);
            // The bound the law holds the column at: a quarter of the ground's own distance from
            // the water, on the side the ROW named.
            let g = base - water;
            let away = if g >= Gi::ZERO { g } else { Gi::ZERO - g };
            let keep = away >> vd_recipe::height::SHORE_SHIFT;
            let bound_m = metres_of_q28(water + if land { keep } else { Gi::ZERO - keep });
            let water_m = metres_of_q28(water);
            let mut rung = 0u8;
            while rung <= 7 {
                let h = height_field_m(&moon, &artifact, dir, rung).expect("a height");
                assert_eq!(
                    h >= water_m,
                    land,
                    "column {c} at rung {rung}: {h} against the water {water_m}, row says land {land}"
                );
                at_bound += i32::from((h - bound_m).abs() < 1e-6);
                rung += 1;
            }
            c += 1;
        }
        assert!(
            lands > 0 && seas > 0,
            "both sides of the shore: {lands} land, {seas} sea"
        );
        assert!(at_bound > 0, "the clamp acted on the line at least once");
    }

    /// ★ A CHUNK'S COLUMN AT A PYRAMID RUNG STANDS ON THE MASK'S SIDE (2026-09-22, ruling W10).
    /// The chunk builder's own path — `chunk::column_field`, the very call the client's builders
    /// and the shard's collider make — over a chunk that straddles the moon's stated shore at the
    /// rung that reads level 1.
    ///
    /// Two statements, each of which could fail: (1) EVERY column of the chunk stands on the side
    /// its own fine row states; (2) the control that makes (1) worth making — the LEVEL'S OWN MEAN
    /// puts some of those columns on the OTHER side, which is exactly the disagreement the shore
    /// crawled on. RED before the mask: those columns followed the mean.
    ///
    /// **Example.** A hull at 1 400 km draws a chunk of 1 km cells over a bay. The level node under
    /// it is mostly water, so its mean stands under the sea; the headland's own rows say land, and
    /// the headland is drawn dry.
    #[test]
    fn a_chunk_column_at_a_pyramid_rung_stands_on_the_masks_side() {
        use crate::artifact::PyramidField;
        use crate::chunk::{CHUNK_EDGE, column_field};
        let (moon, lattice, artifact, face, j) = moon_with_a_stated_sea();
        let n0 = moon.ladder().cells_per_edge(0) as i32;
        let levels = artifact.pyramid.len() as u32;
        let mut rung = 0u8;
        while rung < moon.ladder().rungs && PyramidField::level_for(&lattice, levels, rung) != 1 {
            rung += 1;
        }
        assert!(rung < moon.ladder().rungs, "a rung reads level 1");
        let level = PyramidField::of(&artifact, 1).expect("level 1");
        let coarse = lattice.coarser(1).expect("a coarser lattice");
        // The first rung-0 cell along the middle row where the mask changes its word: the shore.
        let side_at = |i: i32| crate::artifact::sample_side(&lattice, &artifact, face, 0, i, j);
        let first = side_at(64).expect("a bit");
        let mut i = 64;
        while i < n0 - 64 && side_at(i) == Some(first) {
            i += 64;
        }
        assert!(i < n0 - 64, "the middle row crosses the shore");
        let x = (i >> rung) / CHUNK_EDGE as i32;
        let y = (j >> rung) / CHUNK_EDGE as i32;
        let built = column_field(&moon, Some(&level), face, rung, x, y).expect("the chunk builds");
        let mut on_the_masks_side = 0;
        let mut mean_disagrees = 0;
        for (k, site) in built.sites.iter().enumerate() {
            let site_face = Face::from_index(site.face).unwrap_or(face);
            let Some(sea) =
                crate::artifact::sample_side(&lattice, &artifact, site_face, rung, site.i, site.j)
            else {
                continue;
            };
            let (_, h, _) = built.columns[k];
            let water = built.water[k];
            assert_ne!(water, Gi::ZERO, "the moon states a sea for every column");
            if sea {
                assert!(
                    h <= water,
                    "column {k}: the row says sea, the surface stands over it"
                );
            } else {
                assert!(
                    h >= water,
                    "column {k}: the row says land, the surface stands under it"
                );
            }
            on_the_masks_side += 1;
            // The control: where does the LEVEL'S OWN mean put this column?
            let z = crate::artifact::sample_z(&coarse, &level, site_face, rung, site.i, site.j)
                .expect("a mean");
            mean_disagrees += i32::from((moon.radius + z >= water) == sea);
        }
        assert_eq!(
            on_the_masks_side,
            (CHUNK_EDGE * CHUNK_EDGE) as i32,
            "every column of the chunk was judged"
        );
        assert!(
            mean_disagrees > 0,
            "the level's mean disagrees with the mask somewhere in this chunk"
        );
    }

    /// ★ THE FAR-RUNG SHORE: ONE SHORELINE AT EVERY LEVEL (2026-09-22; the owner, from 1 400 km:
    /// "during flight the shores changes again all the time"; ruling W10). The instrument's own
    /// statement, as a unit test on the moon's stated shore.
    ///
    /// MEASURED BEFORE the coast mask (`vd-bins/examples/shore_step` on the home planet's belt
    /// coast, 400 lines of 600 km): the crossing of the sea moved a MEDIAN of 11 536 m at the swap
    /// from the rows to level 1 (rung 9 → 10), 20 823 m at level 1 → 2 and 19 982 m at level 2 → 3,
    /// because a level's `Z` is the mean of its children and a mean crosses the sea somewhere else.
    ///
    /// The statement, which could fail at any of three places: (1) the line really crosses a shore
    /// at every level; (2) the crossing moves by AT MOST ONE FINE NODE between the rows and level
    /// 1, and between level 1 and level 2; (3) every sampled column stands on the side its own
    /// fine row states, at every level.
    ///
    /// **Example.** A hull descends on the belt's coast from 1 400 km. At rung 15 it draws level 3,
    /// at rung 10 level 1, at rung 5 the rows; the beach under it is the same beach at all three.
    #[test]
    fn the_shoreline_stands_within_one_fine_node_at_every_pyramid_level() {
        use crate::artifact::PyramidField;
        let (moon, lattice, artifact, face, j) = moon_with_a_stated_sea();
        let n0 = moon.ladder().cells_per_edge(0) as i32;
        let levels = artifact.pyramid.len() as u32;
        assert!(levels >= 2, "the moon's pyramid holds two levels");
        // The rungs: the finest rung that reads each level, and the TOP rung for a level the
        // moon's own ladder stops before reaching. The moon has fifteen rungs and its coarsest
        // reads level 1 (MEASURED), so level 2 is drawn at the top rung — the very read
        // `PyramidField::of` gives any host, and the swap the home planet makes at rung 14 → 15.
        let top = moon.ladder().rungs - 1;
        let rung_for = |want: u32| -> u8 {
            let mut rung = 0u8;
            while rung < moon.ladder().rungs {
                if PyramidField::level_for(&lattice, levels, rung) == want {
                    return rung;
                }
                rung += 1;
            }
            top
        };
        let fields: Vec<(u8, Box<dyn crate::artifact::ZField>)> = vec![
            (rung_for(0), Box::new(artifact.clone())),
            (
                rung_for(1),
                Box::new(PyramidField::of(&artifact, 1).expect("level 1")),
            ),
            (
                rung_for(2),
                Box::new(PyramidField::of(&artifact, 2).expect("level 2")),
            ),
        ];
        let sea_r = metres_of_q28(moon.sea_radius);
        let step = 64i32;
        // The first crossing of the sea along the middle row, in metres from the row's start, per
        // field; and the count of columns whose side disagrees with their own fine row.
        let mut crossings = Vec::new();
        let mut disagreements = 0;
        let mut sampled = 0;
        for (rung, field) in &fields {
            let mut crossing: Option<f64> = None;
            let mut previous: Option<(f64, f64)> = None;
            let mut i = 64;
            while i < n0 - 64 {
                let dir = cell_dir(&moon, face, i, j);
                let over =
                    height_field_m(&moon, field.as_ref(), dir, *rung).expect("a height") - sea_r;
                // (3) the side the row states is the side the column stands on.
                let sea = crate::artifact::sample_side(&lattice, &artifact, face, 0, i, j)
                    .expect("a row's own bit");
                sampled += 1;
                disagreements += i32::from(if sea { over > 0.0 } else { over < 0.0 });
                if let Some((x0, v0)) = previous
                    && crossing.is_none()
                    && (v0 > 0.0) != (over > 0.0)
                {
                    let t = v0 / (v0 - over);
                    crossing = Some(x0 + t * f64::from(step));
                }
                previous = Some((f64::from(i), over));
                i += step;
            }
            crossings.push(crossing.expect("the line crosses the shore at this level"));
        }
        assert!(sampled > 0, "the line was walked");
        assert_eq!(
            disagreements, 0,
            "every column stands on the side its own fine row states"
        );
        // (2) the crossing moves by at most ONE FINE NODE at each level swap.
        let node_m = lattice.node_m();
        for pair in crossings.windows(2) {
            let step_m = (pair[0] - pair[1]).abs();
            assert!(
                step_m <= node_m,
                "the shore steps {step_m} m at a level swap, over one fine node of {node_m} m"
            );
        }
    }

    #[test]
    fn the_surface_stays_inside_the_band_and_coarser_rungs_stay_within_the_dropped_bound() {
        let m = home();
        let mut i = 0u32;
        while i < 400 {
            let a = -1.0 + f64::from(i % 20) / 10.0;
            let b = -1.0 + f64::from(i / 20) / 10.0;
            let d = dir(Face::ALL[(i % 6) as usize], a, b);
            let h0 = height(&m, d, 0);
            assert!(
                Gi::new((h0 - m.radius).unsigned_abs() as i64) <= m.relief_bound(0),
                "inside the band"
            );
            let mut rung = 1u8;
            while rung < m.ladder.rungs {
                let hl = height(&m, d, rung);
                assert!(
                    Gi::new((hl - h0).unsigned_abs() as i64) <= m.dropped_bound(rung),
                    "rung {rung}: {hl:?} vs {h0:?}"
                );
                assert!(Gi::new((hl - m.radius).unsigned_abs() as i64) <= m.relief_bound(rung));
                rung += 1;
            }
            i += 1;
        }
        assert_eq!(
            height(&m, dir(Face::PosX, 0.2, 0.3), 0),
            height(&m, dir(Face::PosX, 0.2, 0.3), 0)
        );
    }

    /// The float seam: the same answer as the integer path, in metres, for a direction a host holds as
    /// floats. The exactness of the unit is `units`'s own test; this one proves the two doors meet.
    #[test]
    fn the_float_seam_reads_the_integer_answer_in_metres() {
        let m = home();
        // A direction a camera holds: the +X axis, which the door in turns into the bend's own word.
        assert_eq!(
            height_m(&m, [1.0, 0.0, 0.0], 0),
            metres_of_q28(height(&m, [Gi::new(1 << DIR_BITS), Gi::ZERO, Gi::ZERO], 0))
        );
        let surface_m = height_m(&m, [1.0, 0.0, 0.0], 0);
        assert_eq!(
            biome_at(&m, [1.0, 0.0, 0.0], surface_m),
            biome_of(
                &m,
                [Gi::new(1 << DIR_BITS), Gi::ZERO, Gi::ZERO],
                q28_of_metres(surface_m)
            )
        );
        // ★ THE CAP-ROCK BENCH's own door (slice 8a stage 4): the same kernel the column pass runs,
        // in metres, so an instrument can never measure a bench the ground does not have. Three
        // statements: the door IS the kernel; the height field already carries it, so terracing a
        // terraced surface at rung 0 moves it again; and a rung the bench does not reach answers its
        // argument unchanged.
        let raw = m.radius
            + vd_recipe::height::relief_shaped(
                m.octaves_at(0),
                dir(Face::PosX, 0.2, 0.3),
                m.roughness(),
            );
        assert_eq!(
            terrace_m(&m, metres_of_q28(raw), 0),
            metres_of_q28(vd_recipe::terrace::terrace(&m.terrace_at(0), raw))
        );
        let benched = metres_of_q28(height(&m, dir(Face::PosX, 0.2, 0.3), 0));
        assert!(
            (terrace_m(&m, metres_of_q28(raw), 0) - benched).abs() < 1e-6,
            "the height field carries the bench already"
        );
        assert_eq!(terrace_m(&m, 1_234_567.0, 9), 1_234_567.0);
    }

    #[test]
    fn every_biome_appears_on_the_moon_and_the_poles_are_cold() {
        // A seed-built body has no sea and its datum is the ladder radius (C5); the biomes are
        // scanned on a body with a stated sea three kilometres under it, where highlands stand.
        let dry = home();
        assert_eq!(dry.biome_datum(), dry.radius);
        let m = dry.with_sea_m(Some(-3_000));
        let mut seen = [false; 4];
        let mut i = 0u32;
        while i < 2_000 {
            let a = -1.0 + f64::from(i % 40) / 20.0;
            let b = -1.0 + f64::from((i / 40) % 40) / 20.0;
            let d = dir(Face::ALL[(i % 6) as usize], a, b);
            let h = height(&m, d, 0);
            seen[biome_of(&m, d, h) as usize] = true;
            i += 1;
        }
        // Forced cases, so every arm is driven whatever the seed draws. The biome's datum is the
        // ladder radius on this seed-built body (no sea, C5) and the sea's radius on a wet one.
        let datum = m.biome_datum();
        assert_eq!(datum, m.sea_radius);
        assert_eq!(
            m.with_sea_m(Some(-700)).biome_datum(),
            m.radius - (Gi::new(700 * crate::units::STEPS_PER_M) << vd_recipe::cell::LENGTH_BITS)
        );
        let pole = dir(Face::PosZ, 0.0, 0.0);
        assert_eq!(biome_of(&m, pole, datum), Biome::Tundra, "the pole is cold");
        let anywhere = dir(Face::PosX, 0.1, 0.1);
        assert_eq!(
            biome_of(&m, anywhere, datum + m.biome.highland_above + Gi::ONE),
            Biome::Highland,
            "far above the datum is highland"
        );
        assert!(seen[Biome::Grassland as usize], "grassland exists");
        assert!(seen[Biome::Tundra as usize]);
        assert!(seen[Biome::Highland as usize], "highland exists");
        // A desert is a warm dry equator cell; find one by scanning the equator of the +X face.
        let mut desert = false;
        let mut j = 0;
        while j < 2_000 {
            let d = dir(Face::PosX, -1.0 + f64::from(j) / 1_000.0, 0.0);
            desert |= biome_of(&m, d, datum) == Biome::Desert;
            j += 1;
        }
        assert!(
            desert,
            "the home planet's equator holds a desert somewhere on the +X face"
        );
        // ★ MEASURED: HEIGHT COOLS A COLUMN. The same column at the sea and just under the highland
        // height differs in nothing but its height share, so a column whose biome changes between the
        // two proves the share carries the noise's fraction bits. (It read ZERO on the first run of
        // the integer recipe: the reciprocal's shift left the share a whole number, so every share
        // floored to nothing and height cooled no column at all.)
        let just_under = m.biome.highland_above - Gi::ONE;
        let mut cooled = 0;
        let mut j = 0;
        while j < 2_000 {
            let d = dir(Face::PosX, -1.0 + f64::from(j) / 1_000.0, 0.0);
            if biome_of(&m, d, m.sea_radius) != biome_of(&m, d, m.sea_radius + just_under) {
                cooled += 1;
            }
            j += 1;
        }
        assert!(cooled > 0, "height cools some column: {cooled}");
    }

    /// ★ THE FLOAT DOOR OF THE ROUGHNESS FACTOR answers the recipe's own word as a share (slice 8a
    /// stage 3). The instrument that measures the planet's plains and ranges (M-C, the slope
    /// histogram) reads THIS function, so a door that answered something else would make every
    /// number in that measurement a different field's.
    ///
    /// Three statements. (1) The door is the recipe's own kernel through the exit for a share —
    /// exactly, because the divisor is a power of two. (2) Over a scan the answer stays inside
    /// `[M_MIN, 1]`. (3) The scan meets a plain and a range, so the door carries the contrast and
    /// not a constant.
    #[test]
    fn the_float_door_of_the_roughness_factor_reads_the_recipes_own_word() {
        use vd_recipe::height::roughness_factor;
        let m = home();
        let (mut plains, mut ranges) = (0, 0);
        let mut i = 0u32;
        while i < 200 {
            let a = -1.0 + f64::from(i % 20) / 10.0;
            let b = -1.0 + f64::from(i / 20) / 5.0;
            let d = dir(Face::ALL[(i % 6) as usize], a, b);
            // The float direction the door takes in, through the same bend the integer path used.
            let unit = crate::units::unit_of_direction(d);
            let word = roughness_factor(m.roughness(), crate::units::direction_of_unit(unit));
            let share = roughness_at(&m, unit);
            assert_eq!(
                share,
                crate::units::share_of_q28(word),
                "the door is the word at {i}"
            );
            assert!(share >= 0.06, "under the floor at {i}: {share}");
            assert!(share <= 1.0, "over the ceiling at {i}: {share}");
            plains += i32::from(share <= 0.25);
            ranges += i32::from(share >= 0.75);
            i += 1;
        }
        assert!(plains > 0, "the door meets a plain: {plains}");
        assert!(ranges > 0, "and a range: {ranges}");
    }

    /// THE BIOME CODE AND THE BIOME NAME ARE ONE TABLE. The recipe answers a code and this crate
    /// names it; a code the recipe never answers reads as grassland, so the naming is total.
    #[test]
    fn every_biome_code_names_its_own_biome() {
        use vd_recipe::height::{BIOME_DESERT, BIOME_GRASSLAND, BIOME_HIGHLAND, BIOME_TUNDRA};
        assert_eq!(biome_of_code(BIOME_DESERT), Biome::Desert);
        assert_eq!(biome_of_code(BIOME_GRASSLAND), Biome::Grassland);
        assert_eq!(biome_of_code(BIOME_TUNDRA), Biome::Tundra);
        assert_eq!(biome_of_code(BIOME_HIGHLAND), Biome::Highland);
        assert_eq!(biome_of_code(Gi::new(9)), Biome::Grassland, "total");
        // Every biome's code is its own discriminant, which is what lets the charter's strata rows
        // stand in the enum's order.
        for b in Biome::ALL {
            assert_eq!(biome_of_code(Gi::new(b as i64)), b);
        }
    }
}
