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
//! **Example.** Along the direction of the pilot's boots the field at rung 3 is the rung-0 hill
//! without the last three ripples, and the two are never further apart than the dropped amplitudes
//! promise (the test below measures it on the home planet).

use crate::body::BodyDefinition;
use crate::strata::Biome;
use crate::units::{direction_of_unit, metres_of_q28, q28_of_metres};
use vd_recipe::Gi;
use vd_recipe::height::{biome_of as recipe_biome, relief_shaped, roughness_factor};
use vd_recipe::terrace::terrace;

/// The surface's radius along a unit direction at a rung, in GAP STEPS at
/// [`crate::units::LENGTH_BITS`] fraction bits. `dir` carries the bend's 40 fraction bits.
#[must_use]
pub fn height(body: &BodyDefinition, dir: [Gi; 3], rung: u8) -> Gi {
    // ★ THE BENCH IS LAST (slice 8a stage 4): the octave sum answers the raw surface and the terrace
    // pulls it toward the nearest bed top. ONE SOURCE — `vd_recipe::terrace::terrace` is the very
    // function the card's column pass runs, so a picture and a pair of boots stand on one bench.
    terrace(
        &body.terrace_at(rung),
        body.radius + relief_shaped(body.octaves_at(rung), dir, body.roughness()),
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
    let lattice = body.macro_lattice()?.coarser(field.level())?;
    let face = vd_seed::bend::face_of(dir);
    let (t, s) = vd_seed::bend::face_coords(face, dir);
    let n0 = body.ladder().cells_per_edge(0);
    let i = vd_seed::ladder::index_of(vd_seed::bend::unbend(t), n0);
    let j = vd_seed::ladder::index_of(vd_seed::bend::unbend(s), n0);
    let z = crate::artifact::sample_z(&lattice, field, face, 0, i, j)?;
    let relief = vd_recipe::height::relief_of_table_from(
        body.octave_table(),
        body.first_fine(),
        body.octaves_at(rung).len(),
        direction_of_unit(dir),
        body.roughness(),
    );
    Some(metres_of_q28(terrace(
        &body.terrace_at(rung),
        body.radius + z + relief,
    )))
}

/// ★ THE PER-COLUMN ROUGHNESS FACTOR as a real number in `[M_MIN, 1]`, for a host outside the recipe
/// — the slope histogram's instrument reads it to say which columns are a PLAIN and which a RANGE
/// (slice 8a stage 3, measurement M-C). The shape is the recipe's own kernel; this crate only names
/// the answer, so an instrument can never measure a factor the field does not use.
#[must_use]
pub fn roughness_at(body: &BodyDefinition, dir: [f64; 3]) -> f64 {
    crate::units::share_of_q28(roughness_factor(body.roughness(), direction_of_unit(dir)))
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

    /// The direction of a face position, through the recipe's own bend: the cell nearest `(a, b)` of
    /// a face at rung 0 (the test names positions, the recipe names cells).
    fn dir(face: Face, a: f64, b: f64) -> [Gi; 3] {
        let m = home();
        let n_l = m.ladder().cells_per_edge(0);
        let i = vd_seed::ladder::index_of(a, n_l);
        let j = vd_seed::ladder::index_of(b, n_l);
        vd_seed::bend::direction_q(face, i, j, m.inv_n(0))
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
        let m = home();
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
        // Forced cases, so every arm is driven whatever the seed draws.
        let pole = dir(Face::PosZ, 0.0, 0.0);
        assert_eq!(
            biome_of(&m, pole, m.sea_radius),
            Biome::Tundra,
            "the pole is cold"
        );
        let anywhere = dir(Face::PosX, 0.1, 0.1);
        assert_eq!(
            biome_of(
                &m,
                anywhere,
                m.sea_radius + m.biome.highland_above + Gi::ONE
            ),
            Biome::Highland,
            "far above the sea is highland"
        );
        assert!(seen[Biome::Grassland as usize], "grassland exists");
        assert!(seen[Biome::Tundra as usize]);
        assert!(seen[Biome::Highland as usize], "highland exists");
        // A desert is a warm dry equator cell; find one by scanning the equator of the +X face.
        let mut desert = false;
        let mut j = 0;
        while j < 2_000 {
            let d = dir(Face::PosX, -1.0 + f64::from(j) / 1_000.0, 0.0);
            desert |= biome_of(&m, d, m.sea_radius) == Biome::Desert;
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
