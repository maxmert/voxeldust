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
//! ★ **AND NO WATER SHAPES THE GROUND** (2026-09-28, ruling B22 step 1). The shore law — which held
//! a column on its ground's side of its water, read the coast mask's side word and its footprint —
//! is removed with its inputs. The surface is the plain ground: the bench over the octave sum, at
//! every rung and on every host, and the dry sea floor shows until the one-volume fill step. The
//! solve's own water stands: it shaped this ground by its erosion, and the artifact keeps it.
//!
//! **Example.** Along the direction of the pilot's boots the field at rung 3 is the rung-0 hill
//! without the last three ripples, and the two are never further apart than the dropped amplitudes
//! promise (the test below measures it on the home planet).

use crate::body::BodyDefinition;
use crate::fine::read::FineZ;
use crate::strata::Biome;
use crate::units::{direction_of_unit, metres_of_q28, q28_of_metres};
use vd_recipe::Gi;
use vd_recipe::height::{
    OCTAVES_CAP, biome_of as recipe_biome, relief_parts_from, roughness_factor,
};
use vd_recipe::terrace::terrace;

/// The surface's radius along a unit direction at a rung, in GAP STEPS at
/// [`crate::units::LENGTH_BITS`] fraction bits. `dir` carries the bend's 40 fraction bits.
#[must_use]
pub fn height(body: &BodyDefinition, dir: [Gi; 3], rung: u8) -> Gi {
    // ★ THE BENCH OVER THE SUM (slice 8a stage 4): the octave sum answers the raw surface and the
    // terrace pulls it toward the nearest bed top. ONE SOURCE — `vd_recipe::terrace::terrace` is
    // the very function the card's column pass runs, so a picture and a pair of boots stand on one
    // bench. No water holds the surface since ruling B22 (2026-09-28).
    let parts = relief_parts_from(
        body.octave_table(),
        0,
        body.octaves_at(rung).len(),
        dir,
        body.roughness(),
        Gi::ZERO,
    );
    let base = body.radius + parts.coarse;
    terrace(&body.terrace_at(rung), base + parts.fine)
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
/// which the artifact replaced. ★ It is also THE SURFACE THE CHUNKS DRAW (2026-09-28, ruling B22):
/// no water sheet stands over a column any more, so an eye over the sea reads the sea floor.
///
/// **Example.** A hull hangs 20 m over the belt's sea level, where the floor lies 6.9 km down: this
/// answers the floor, and the eye reads 6.9 km to the drawn ground.
#[must_use]
pub fn height_field_m(
    body: &BodyDefinition,
    field: &dyn crate::artifact::ZField,
    dir: [f64; 3],
    rung: u8,
) -> Option<f64> {
    field_column(body, field, dir, rung).map(metres_of_q28)
}

/// ★ THE FINE WATER ALONG A DIRECTION at a rung (slice W2): the field's [`crate::artifact::ZField::fine_water`] at the
/// rung-0 cell the direction falls in — the same cell the ground's read takes ([`height_field_m`]).
#[must_use]
pub fn fine_water_at(
    body: &BodyDefinition,
    field: &dyn crate::artifact::ZField,
    dir: [f64; 3],
    rung: u8,
) -> crate::fine::read::FineWater {
    let face = vd_seed::bend::face_of(dir);
    let (t, s) = vd_seed::bend::face_coords(face, dir);
    let n0 = body.ladder().cells_per_edge(0);
    let i = vd_seed::ladder::index_of(vd_seed::bend::unbend(t), n0);
    let j = vd_seed::ladder::index_of(vd_seed::bend::unbend(s), n0);
    field.fine_water(face, rung, 0, i, j)
}

/// The ground the field states along a direction at a rung, as the recipe's fixed-point radius.
/// `None` where the field holds no row for the stencil.
fn field_column(
    body: &BodyDefinition,
    field: &dyn crate::artifact::ZField,
    dir: [f64; 3],
    rung: u8,
) -> Option<Gi> {
    let lattice = body.macro_lattice()?.coarser(field.level())?;
    let face = vd_seed::bend::face_of(dir);
    let (t, s) = vd_seed::bend::face_coords(face, dir);
    let n0 = body.ladder().cells_per_edge(0);
    let i = vd_seed::ladder::index_of(vd_seed::bend::unbend(t), n0);
    let j = vd_seed::ladder::index_of(vd_seed::bend::unbend(s), n0);
    let (z, first, slope_share) = match field.fine_z(face, rung, 0, i, j) {
        // ★ THE FINE GROUND (B1 (g)): the curve of the level the rung draws, at the rung-0 cell, and
        // no octave under it — the chunk's own column stands on the same curve at its sample.
        FineZ::Held(z) => (z, OCTAVES_CAP, Gi::ZERO),
        FineZ::Missing(_) => return None,
        // ★ THE SAME SLOPE SHARE THE COLUMN READS (2026-09-21): the morph target and the chunk's own
        // column must stand at one height, so this reads the field the same way — the central
        // difference over one macro node at the SAME rung-0 cell the `Z` came from.
        FineZ::Outside => (
            crate::artifact::sample_z(&lattice, field, face, 0, i, j)?,
            body.first_fine(),
            crate::artifact::slope_share(
                &crate::artifact::slope_charter(body, &lattice, 0),
                &lattice,
                field,
                face,
                0,
                i,
                j,
            )?,
        ),
    };
    let unit = direction_of_unit(dir);
    let parts = relief_parts_from(
        body.octave_table(),
        first,
        body.octaves_at(rung).len(),
        unit,
        body.roughness(),
        slope_share,
    );
    let base = body.radius + z + parts.coarse;
    Some(terrace(&body.terrace_at(rung), base + parts.fine))
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

    /// ★ THE FINE GROUND UNDER A FLOAT DIRECTION (B1 (g)): the eye's and the morph's read stands on the held fine
    /// height with no octave, the same bench over it as the chunk's column; a missing tile answers nothing.
    #[test]
    fn the_float_read_stands_on_held_fine_ground_and_answers_nothing_for_a_missing_tile() {
        use crate::fine::l1::TileKey;
        use crate::fine::read::test_field::OneFineRead;
        let body = home();
        let face = Face::from_index(2).expect("a face");
        let dir = cell_dir(&body, face, 1_000, 2_000);
        let z0 = Gi::new(-1_200 * crate::units::STEPS_PER_M) << crate::units::LENGTH_BITS;
        let held = OneFineRead(FineZ::Held(z0));
        let want = metres_of_q28(terrace(&body.terrace_at(5), body.radius + z0));
        assert_eq!(
            height_field_m(&body, &held, dir, 5).map(f64::to_bits),
            Some(want.to_bits())
        );
        let missing = OneFineRead(FineZ::Missing(TileKey { face, tx: 0, ty: 0 }));
        assert_eq!(height_field_m(&body, &missing, dir, 5), None);
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

    /// ★ A ONE-TILE CACHE AT ITS EDGE (2026-09-24): where the height's own stencil stands inside
    /// the one tile held and the slope's wider stencil reaches the tile not held, the field answers
    /// NOTHING — never a height with a slope read off a hole. Walks inward from the tile's edge, one
    /// node at a time, until the height reads and the slope does not; the whole artifact answers
    /// the same column, so the refusal is the missing tile's and not the column's.
    #[test]
    fn a_tile_whose_neighbour_is_missing_answers_nothing_at_its_edge() {
        use crate::artifact::{TILE_EDGE, TileCache};
        use crate::home::{HOME_SYSTEM_AGE_YR, home_moon, home_moon_solve_words};
        use crate::solve::Schedule;
        let moon = home_moon();
        let lattice = moon.macro_lattice().expect("a lattice");
        let words = home_moon_solve_words();
        let (state, facies, _) =
            crate::test_solves::solved(&moon, &words, Schedule::standard(HOME_SYSTEM_AGE_YR));
        let climate =
            crate::climate::climate(&moon, &lattice, &words, &state.z, Some(state.sea_view()));
        let artifact =
            crate::artifact::Artifact::of(&state, &facies, &climate, words.water_km3 > 0);
        assert!(
            artifact.tiles_per_edge() >= 2,
            "the moon's face holds two tiles a side"
        );
        let mut one = TileCache::new(lattice.edge);
        one.apply(artifact.tile(Face::PosX, 0, 0));
        let cpn = lattice.cells_per_node as i32;
        let edge = TILE_EDGE as i32;
        let j = (edge / 2) * cpn + cpn / 2;
        let found = (1..=8).find(|k| {
            let i = (edge - k) * cpn + cpn / 2;
            let z = crate::artifact::sample_z(&lattice, &one, Face::PosX, 0, i, j);
            let dir = cell_dir(&moon, Face::PosX, i, j);
            z.is_some() & height_field_m(&moon, &one, dir, 0).is_none()
        });
        let k = found.expect("a node whose height reads while its slope does not");
        let i = (edge - k) * cpn + cpn / 2;
        let dir = cell_dir(&moon, Face::PosX, i, j);
        assert!(height_field_m(&moon, &artifact, dir, 0).is_some());
    }

    /// ★ THE HEIGHT WITH THE MACRO FIELD (C4c): through the moon's artifact it answers along any
    /// direction, agrees with the column kernel's own surface at a cell centre to a step, reads a
    /// pyramid level at a coarse rung, and answers nothing through a cache missing the tile.
    #[test]
    fn the_height_reads_the_artifacts_field_along_a_direction() {
        use crate::CHUNK_EDGE;
        use crate::artifact::{PyramidField, TileCache};
        use crate::home::{HOME_SYSTEM_AGE_YR, home_moon, home_moon_solve_words};
        use crate::solve::Schedule;
        let moon = home_moon();
        let lattice = moon.macro_lattice().expect("a lattice");
        let words = home_moon_solve_words();
        let (state, facies, _) =
            crate::test_solves::solved(&moon, &words, Schedule::standard(HOME_SYSTEM_AGE_YR));
        let climate =
            crate::climate::climate(&moon, &lattice, &words, &state.z, Some(state.sea_view()));
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
        let level = PyramidField::of(&artifact, 1, &artifact.coast_counts()).expect("level 1");
        assert!(height_field_m(&moon, &level, dir, 12).is_some());
        let empty = TileCache::new(lattice.edge);
        assert_eq!(height_field_m(&moon, &empty, dir, 0), None);
        let level_9 = PyramidField {
            level: 9,
            z_m: vec![],
            water_m: vec![],
            coast: None,
            counts: None,
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

    /// ★ NO WATER SHAPES THE GROUND (2026-09-28, ruling B22 step 1). A stated sea moves no column:
    /// over a scan of the home planet's directions, the ground with a sea at the ladder radius and
    /// the ground with no sea are the same word at every rung the scan reads. The control, which
    /// could fail, says the scan met the columns the shore law used to move: some column's coarse
    /// ground and its surface stand on the two sides of the sea, and the law held such a column on
    /// its ground's side.
    ///
    /// RED before ruling B22: `height` held a column on its ground's side of the body's sea.
    ///
    /// **Example.** A plain 40 m over the sea whose fine octaves dig a valley 60 m deep: the
    /// valley's floor now stands 20 m under the sea's level, and the pilot sees that dry floor.
    #[test]
    fn no_water_shapes_the_ground() {
        let dry = home();
        let wet = dry.with_sea_m(Some(0));
        let sea = wet.sea_radius;
        assert_ne!(sea, Gi::ZERO, "the scan's body states a sea");
        let mut crossed = 0;
        let mut i = 0u32;
        while i < 2_000 {
            let a = -1.0 + f64::from(i % 40) / 20.0;
            let b = -1.0 + f64::from((i / 40) % 40) / 20.0;
            let d = dir(Face::ALL[(i % 6) as usize], a, b);
            let mut rung = 0u8;
            while rung < 4 {
                let h = height(&wet, d, rung);
                assert_eq!(h, height(&dry, d, rung), "direction {i} at rung {rung}");
                let parts = relief_parts_from(
                    wet.octave_table(),
                    0,
                    wet.octaves_at(rung).len(),
                    d,
                    wet.roughness(),
                    Gi::ZERO,
                );
                let base = wet.radius + parts.coarse;
                crossed += i32::from((base >= sea) != (h >= sea));
                rung += 1;
            }
            i += 1;
        }
        assert!(
            crossed > 0,
            "the scan meets a column whose fine octaves carry it across the sea"
        );
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
