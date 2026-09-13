//! ★ THE CARVERS — where a cave hollows the rock, ON INTEGERS (ruling F7). Two kinds: a CAVERN FIELD,
//! a slow value noise in three dimensions that opens rooms where it rises above a threshold, and TUBE
//! CARVERS, seed-placed line segments of a stated radius that cut passages. Both give a "hollow" in
//! GAP STEPS: positive where the cell is air, and the cell's final gap is the greater of the rock's
//! and the cave's, so air wins and a tunnel mouth is round.
//!
//! **Caves are detail, like the fine octaves.** A cavern is sampled on a coarse lattice — one node per
//! [`CAVERN_STRIDE`] cells — and interpolated between nodes with weights that are powers of two, so
//! the field costs one sample per 64 cells and the interpolation is exact. A tube is a few metres wide,
//! so it is carved only where a cell is no wider than the tube; a cavern only where a cell is no
//! wider than a quarter of its wavelength. Coarser rungs draw the hill without its caves, exactly as
//! they draw it without its ripples, and the crossfade closes the detail's arrival.
//!
//! **No kernel divides.** The cavern's wavelength enters as the charter's RECIPROCAL; the tube's
//! region index is a shift, because the region's edge is a power of two metres; and a tube carries the
//! reciprocal of its own squared length, drawn once with the tube, so the projection of a cell onto
//! the segment is one two-word product. The distance itself is one integer square root.
//!
//! **Example.** Forty metres under a cliff the cavern field reads 0.71 against a threshold of 0.66:
//! the cell is one metre inside a room. A tube of radius three metres runs past two cells away: the
//! cell is one metre inside the tube. The greater says: hollow, by 128 gap steps.

use crate::body::{BodyDefinition, CAVERN_RECIP_BITS};
use vd_recipe::Gi;
use vd_recipe::noise::{NOISE_BITS, NOISE_ONE, unit_value, value3};
use vd_recipe::root::recip_pow2;

/// ★ THE TUBE CARVER IS THE RECIPE'S OWN (ruling F7, step G1): the segment, its radius, its stored
/// reciprocal and the distance from a point to it live in `vd_recipe::cell`, so the server's CPU,
/// the client's CPU and the client's GPU all measure a passage with ONE function. This crate DRAWS
/// the carvers ([`tubes_in`]) and the recipe MEASURES them.
pub use vd_recipe::cell::{TUBE_RECIP_BITS, Tube, tube_hollow_steps};
use vd_seed::ladder::cell_m;
use vd_seed::rng::child_seed;

/// Cells between two nodes of the cavern lattice.
pub const CAVERN_STRIDE: usize = 4;
/// The same stride as a shift: the lattice's weights are exact multiples of `1/CAVERN_STRIDE`.
pub const CAVERN_STRIDE_LOG2: u32 = 2;
const _: () = assert!(1 << CAVERN_STRIDE_LOG2 == CAVERN_STRIDE);

/// Whether tubes are carved at a rung: only where a cell is no wider than the tube.
#[must_use]
pub fn tubes_carve_at(body: &BodyDefinition, rung: u8) -> bool {
    cell_steps(rung) <= body.caves.tube_radius_steps * Gi::new(2)
}

/// Whether the cavern field is carved at a rung: only where a cell is no wider than a quarter of the
/// field's wavelength. Whole metres on both sides.
#[must_use]
pub fn caverns_carve_at(body: &BodyDefinition, rung: u8) -> bool {
    cell_m(rung) * 4 <= body.caves.cavern_wavelength_m
}

/// A cell's width at a rung, in gap steps — exact, a power of two times 128.
#[must_use]
pub fn cell_steps(rung: u8) -> Gi {
    Gi::new(i64::from(cell_m(rung)) * crate::units::STEPS_PER_M)
}

/// The cavern field's raw value at a point in the body's frame, in `[0, 1)` at the noise's fraction
/// bits. The lattice point is the point in gap steps times the wavelength's reciprocal.
#[must_use]
pub fn cavern_value(body: &BodyDefinition, point_steps: [Gi; 3]) -> Gi {
    let recip = body.caves.cavern_recip;
    // point_m / λ at NOISE_BITS: the point is 128 times the metres, so the shift takes those seven
    // bits back out along with the reciprocal's own.
    let shift = CAVERN_RECIP_BITS - NOISE_BITS + crate::units::STEPS_PER_M.trailing_zeros();
    value3(
        body.seed,
        [
            point_steps[0].mul_shr(recip, shift),
            point_steps[1].mul_shr(recip, shift),
            point_steps[2].mul_shr(recip, shift),
        ],
    )
}

/// The tube region a point falls in: a cube of the body's tube region edge, indexed by the shift the
/// charter holds (the edge is a power of two metres, so the index needs no division and an arithmetic
/// shift floors on both sides of the centre).
#[must_use]
pub fn tube_region(body: &BodyDefinition, point_steps: [Gi; 3]) -> [i64; 3] {
    let shift = body.caves.tube_region_shift;
    [
        (point_steps[0] >> shift).raw(),
        (point_steps[1] >> shift).raw(),
        (point_steps[2] >> shift).raw(),
    ]
}

/// The tubes a region draws: zero to three, each from the region's own hash, from a point in the
/// region toward a point in a neighbouring region.
#[must_use]
pub fn tubes_in(body: &BodyDefinition, region: [i64; 3]) -> Vec<Tube> {
    let edge = body.caves.tube_region_steps;
    let h = child_seed(
        child_seed(
            child_seed(body.caves.tube_seed, 1, region[0] as u64),
            2,
            region[1] as u64,
        ),
        3,
        region[2] as u64,
    );
    let count = (h & 3) as usize;
    let mut out = Vec::with_capacity(count);
    let mut t = 0;
    while t < count {
        let hs = child_seed(h, 4, t as u64);
        let unit = |salt: u64| unit_value(child_seed(hs, salt, 0));
        let base = [
            Gi::new(region[0]) * edge,
            Gi::new(region[1]) * edge,
            Gi::new(region[2]) * edge,
        ];
        let at = |salt: u64| (unit(salt) * edge) >> NOISE_BITS;
        let reach = |salt: u64| ((unit(salt) * Gi::new(3) - NOISE_ONE) * edge) >> NOISE_BITS;
        let start = [base[0] + at(10), base[1] + at(11), base[2] + at(12)];
        let end = [
            base[0] + reach(13),
            base[1] + reach(14),
            base[2] + reach(15),
        ];
        let mut len2 = 0u64;
        let mut c = 0;
        while c < 3 {
            let m = (end[c] - start[c]).unsigned_abs();
            len2 = len2.wrapping_add(m.wrapping_mul(m));
            c += 1;
        }
        out.push(Tube {
            start,
            end,
            radius_steps: body.caves.tube_radius_steps,
            inv_len2: Gi::new(recip_pow2(len2, TUBE_RECIP_BITS) as i64),
        });
        t += 1;
    }
    out
}

/// Every tube that can reach the regions `lo..=hi` (a chunk's own regions): a tube starts in its
/// region and ends up to TWO regions further along an axis or ONE back, so the regions from two below
/// to one above the chunk's own are asked, in one fixed order. The caller then keeps only the tubes
/// that come within their radius of the chunk.
#[must_use]
pub fn tubes_near(body: &BodyDefinition, lo: [i64; 3], hi: [i64; 3]) -> Vec<Tube> {
    let mut out = Vec::new();
    let mut x = lo[0] - 2;
    while x <= hi[0] + 1 {
        let mut y = lo[1] - 2;
        while y <= hi[1] + 1 {
            let mut z = lo[2] - 2;
            while z <= hi[2] + 1 {
                out.extend(tubes_in(body, [x, y, z]));
                z += 1;
            }
            y += 1;
        }
        x += 1;
    }
    out
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
    use crate::units::{STEPS_PER_M, metres_of_steps};

    fn home() -> BodyDefinition {
        crate::home::home_planet()
    }

    /// A whole number of metres as gap steps.
    fn s(metres: i64) -> Gi {
        Gi::new(metres * STEPS_PER_M)
    }

    /// A tube between two points stated in whole metres.
    fn tube(a: [i64; 3], b: [i64; 3], radius_m: i64) -> Tube {
        let start = [s(a[0]), s(a[1]), s(a[2])];
        let end = [s(b[0]), s(b[1]), s(b[2])];
        let mut len2 = 0u64;
        let mut c = 0;
        while c < 3 {
            let m = (end[c] - start[c]).unsigned_abs();
            len2 = len2.wrapping_add(m.wrapping_mul(m));
            c += 1;
        }
        Tube {
            start,
            end,
            radius_steps: s(radius_m),
            inv_len2: Gi::new(recip_pow2(len2, TUBE_RECIP_BITS) as i64),
        }
    }

    #[test]
    fn the_segment_distance_is_exact_on_the_ends_the_middle_and_a_degenerate_segment() {
        let t = tube([0, 0, 0], [10, 0, 0], 3);
        assert_eq!(t.distance_steps([s(5), s(3), s(4)]), s(5));
        assert_eq!(
            t.distance_steps([s(-3), s(4), s(0)]),
            s(5),
            "before the start"
        );
        assert_eq!(t.distance_steps([s(13), s(0), s(4)]), s(5), "past the end");
        let point = tube([0, 0, 0], [0, 0, 0], 3);
        assert_eq!(
            point.distance_steps([s(0), s(0), s(2)]),
            s(2),
            "a point segment"
        );
        assert_eq!(tube_hollow_steps(&[t], [s(5), s(1), s(0)]), s(2));
        assert_eq!(
            tube_hollow_steps(&[t], [s(5), s(9), s(0)]),
            Gi::ZERO,
            "outside stays zero"
        );
        assert_eq!(tube_hollow_steps(&[], [s(5), s(1), s(0)]), Gi::ZERO);
    }

    #[test]
    fn the_cavern_field_opens_rooms_and_the_tube_regions_draw_a_bounded_count() {
        let m = home();
        let charter = crate::chunk::charter_of(&m, 0, 0);
        let mut hollow = 0;
        let mut solid = 0;
        let mut i = 0i64;
        while i < 4_000 {
            let p = [Gi::new(47 * i), Gi::new(78 * i), Gi::new(166 * i)];
            if charter.cavern_hollow_steps(cavern_value(&m, p)) > Gi::ZERO {
                hollow += 1;
            } else {
                solid += 1;
            }
            i += 1;
        }
        assert!(hollow > 0, "some rooms");
        assert!(solid > hollow, "mostly rock");
        assert_eq!(charter.cavern_hollow_steps(Gi::ZERO), Gi::ZERO);
        assert!(charter.cavern_hollow_steps(NOISE_ONE) > Gi::ZERO);
        let mut total = 0;
        let mut r = 0i64;
        while r < 64 {
            let tubes = tubes_in(&m, [r, r * 7, -r]);
            assert!(tubes.len() <= 3);
            for t in &tubes {
                assert_eq!(t.radius_steps, m.caves.tube_radius_steps);
                assert!(t.start[0] >= Gi::new(r) * m.caves.tube_region_steps);
                assert!(t.start[0] < Gi::new(r + 1) * m.caves.tube_region_steps);
            }
            total += tubes.len();
            r += 1;
        }
        assert!(total > 0, "some regions draw tubes");
        assert_eq!(tube_region(&m, [s(1000), s(-1), Gi::ZERO]), [1, -1, 0]);
        assert_eq!(
            tubes_in(&m, [3, 21, -3]),
            tubes_in(&m, [3, 21, -3]),
            "the same region, the same tubes"
        );
        // Every tube's stored reciprocal is the exact floor of its own squared length's reciprocal.
        for t in tubes_in(&m, [3, 21, -3]) {
            let mut len2 = 0u64;
            let mut c = 0;
            while c < 3 {
                let d = (t.end[c] - t.start[c]).unsigned_abs();
                len2 += d * d;
                c += 1;
            }
            assert_eq!(t.inv_len2.raw() as u64, recip_pow2(len2, TUBE_RECIP_BITS));
        }
    }

    /// The refuter's finding 3: a tube may run from its region into the next two, so a chunk asks
    /// the regions from two below to one above its own, every one, in one order.
    #[test]
    fn the_tubes_near_a_chunk_come_from_the_regions_a_tube_can_reach() {
        let m = home();
        let near = tubes_near(&m, [3, 21, -3], [3, 21, -3]);
        let mut expected = Vec::new();
        for x in 1..=4 {
            for y in 19..=22 {
                for z in -5..=-2 {
                    expected.extend(tubes_in(&m, [x, y, z]));
                }
            }
        }
        assert_eq!(near, expected, "sixty-four regions, in order");
        assert!(!near.is_empty());
        // A tube of region (3, 21, -3) that ends in another region is still found from that other
        // region's neighbourhood.
        let own = tubes_in(&m, [3, 21, -3]);
        for t in &own {
            let end_region = tube_region(&m, t.end);
            let from_end = tubes_near(&m, end_region, end_region);
            assert!(
                from_end.contains(t),
                "the tube is reachable from where it ends"
            );
        }
    }

    #[test]
    fn caves_are_detail_that_stops_at_coarse_rungs() {
        let m = home();
        assert!(tubes_carve_at(&m, 0), "a one-metre cell sees a tube");
        assert!(!tubes_carve_at(&m, 6), "a 64 m cell does not");
        assert!(caverns_carve_at(&m, 0));
        assert!(!caverns_carve_at(&m, 10));
        assert_eq!(cell_steps(0), Gi::new(STEPS_PER_M));
        assert_eq!(metres_of_steps(cell_steps(3).raw()), 8.0);
    }
}
