//! ★ THE CARVERS — where a cave hollows the rock. Two kinds: a CAVERN FIELD, a slow value noise in
//! three dimensions that opens rooms where it rises above a threshold, and TUBE CARVERS, seed-placed
//! line segments of a stated radius that cut passages. Both give a "hollow" in metres: positive where
//! the cell is air, and the cell's final gap is the greater of the rock's and the cave's, so air wins
//! and a tunnel mouth is round.
//!
//! **Caves are detail, like the fine octaves.** A cavern is sampled on a coarse lattice — one node per
//! [`CAVERN_STRIDE`] cells — and interpolated between nodes with weights that are powers of two, so
//! the field costs one sample per 64 cells and the interpolation is exact. A tube is a few metres wide,
//! so it is carved only where a cell is no wider than the tube; a cavern only where a cell is no
//! wider than a quarter of its wavelength. Coarser rungs draw the hill without its caves, exactly as
//! they draw it without its ripples, and the crossfade closes the detail's arrival.
//!
//! **Example.** Forty metres under a cliff the cavern field reads 0.71 against a threshold of 0.66:
//! the cell is one metre inside a room. A tube of radius three metres runs past two cells away: the
//! cell is one metre inside the tube. The greater says: hollow, by one metre.

use crate::body::BodyDefinition;
use crate::gf::Gf;
use crate::noise::{unit_value, value3};
use vd_seed::rng::child_seed;

/// Cells between two nodes of the cavern lattice.
pub const CAVERN_STRIDE: usize = 4;

/// A tube carver: a segment in the body's frame, in metres, with a radius.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Tube {
    pub start: [Gf; 3],
    pub end: [Gf; 3],
    pub radius_m: Gf,
}

/// Whether tubes are carved at a rung: only where a cell is no wider than the tube.
#[must_use]
pub fn tubes_carve_at(body: &BodyDefinition, cell_m: Gf) -> bool {
    cell_m <= body.caves.tube_radius_m * Gf::TWO
}

/// Whether the cavern field is carved at a rung: only where a cell is no wider than a quarter of the
/// field's wavelength.
#[must_use]
pub fn caverns_carve_at(body: &BodyDefinition, cell_m: Gf) -> bool {
    cell_m * Gf::from_i64(4) <= body.caves.cavern_wavelength_m
}

/// The cavern field's raw value at a point in the body's frame, in `[0, 1)`.
#[must_use]
pub fn cavern_value(body: &BodyDefinition, point_m: [Gf; 3]) -> Gf {
    let f = Gf::ONE / body.caves.cavern_wavelength_m;
    value3(body.seed, [point_m[0] * f, point_m[1] * f, point_m[2] * f])
}

/// The hollow the cavern field opens for a field value, in metres: zero outside a room, positive
/// inside.
#[must_use]
pub fn cavern_hollow_m(body: &BodyDefinition, value: Gf) -> Gf {
    let over = value - body.caves.cavern_threshold;
    if over > Gf::ZERO {
        over * body.caves.cavern_scale_m
    } else {
        Gf::ZERO
    }
}

/// The tube region a point falls in: a cube of the body's tube region edge, indexed by floor.
#[must_use]
pub fn tube_region(body: &BodyDefinition, point_m: [Gf; 3]) -> [i64; 3] {
    let edge = Gf::from_i64(i64::from(body.caves.tube_region_m));
    [
        (point_m[0] / edge).to_i64_floor(),
        (point_m[1] / edge).to_i64_floor(),
        (point_m[2] / edge).to_i64_floor(),
    ]
}

/// The tubes a region draws: zero to three, each from the region's own hash, from a point in the
/// region toward a point in a neighbouring region.
#[must_use]
pub fn tubes_in(body: &BodyDefinition, region: [i64; 3]) -> Vec<Tube> {
    let edge = Gf::from_i64(i64::from(body.caves.tube_region_m));
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
            Gf::from_i64(region[0]) * edge,
            Gf::from_i64(region[1]) * edge,
            Gf::from_i64(region[2]) * edge,
        ];
        let start = [
            base[0] + unit(10) * edge,
            base[1] + unit(11) * edge,
            base[2] + unit(12) * edge,
        ];
        let end = [
            base[0] + (unit(13) * Gf::from_i64(3) - Gf::ONE) * edge,
            base[1] + (unit(14) * Gf::from_i64(3) - Gf::ONE) * edge,
            base[2] + (unit(15) * Gf::from_i64(3) - Gf::ONE) * edge,
        ];
        out.push(Tube {
            start,
            end,
            radius_m: body.caves.tube_radius_m,
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

/// The hollow a set of tubes opens at a point, in metres: the greatest of `radius − distance` over
/// the tubes, never below zero.
#[must_use]
pub fn tube_hollow_m(tubes: &[Tube], point_m: [Gf; 3]) -> Gf {
    let mut best = Gf::ZERO;
    for tube in tubes {
        let d = segment_distance_m(tube.start, tube.end, point_m);
        best = best.greater(tube.radius_m - d);
    }
    best
}

/// The distance from a point to a segment: the projection clamped to the segment, then one square
/// root.
#[must_use]
pub fn segment_distance_m(a: [Gf; 3], b: [Gf; 3], p: [Gf; 3]) -> Gf {
    let ab = [b[0] - a[0], b[1] - a[1], b[2] - a[2]];
    let ap = [p[0] - a[0], p[1] - a[1], p[2] - a[2]];
    let len2 = ab[0] * ab[0] + ab[1] * ab[1] + ab[2] * ab[2];
    let t = if len2 > Gf::ZERO {
        ((ap[0] * ab[0] + ap[1] * ab[1] + ap[2] * ab[2]) / len2).clamp(Gf::ZERO, Gf::ONE)
    } else {
        Gf::ZERO
    };
    let q = [a[0] + ab[0] * t, a[1] + ab[1] * t, a[2] + ab[2] * t];
    let d = [p[0] - q[0], p[1] - q[1], p[2] - q[2]];
    (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn home() -> BodyDefinition {
        crate::home::home_planet()
    }

    fn g(v: f64) -> Gf {
        Gf::from_f64(v)
    }

    #[test]
    fn the_segment_distance_is_exact_on_the_ends_the_middle_and_a_degenerate_segment() {
        let a = [g(0.0), g(0.0), g(0.0)];
        let b = [g(10.0), g(0.0), g(0.0)];
        assert_eq!(segment_distance_m(a, b, [g(5.0), g(3.0), g(4.0)]), g(5.0));
        assert_eq!(
            segment_distance_m(a, b, [g(-3.0), g(4.0), g(0.0)]),
            g(5.0),
            "before the start"
        );
        assert_eq!(
            segment_distance_m(a, b, [g(13.0), g(0.0), g(4.0)]),
            g(5.0),
            "past the end"
        );
        assert_eq!(
            segment_distance_m(a, a, [g(0.0), g(0.0), g(2.0)]),
            g(2.0),
            "a point segment"
        );
        let tube = Tube {
            start: a,
            end: b,
            radius_m: g(3.0),
        };
        assert_eq!(tube_hollow_m(&[tube], [g(5.0), g(1.0), g(0.0)]), g(2.0));
        assert_eq!(
            tube_hollow_m(&[tube], [g(5.0), g(9.0), g(0.0)]),
            Gf::ZERO,
            "outside stays zero"
        );
        assert_eq!(tube_hollow_m(&[], [g(5.0), g(1.0), g(0.0)]), Gf::ZERO);
    }

    #[test]
    fn the_cavern_field_opens_rooms_and_the_tube_regions_draw_a_bounded_count() {
        let m = home();
        let mut hollow = 0;
        let mut solid = 0;
        let mut i = 0i64;
        while i < 4_000 {
            let p = [
                g(0.37) * Gf::from_i64(i),
                g(0.61) * Gf::from_i64(i),
                g(1.3) * Gf::from_i64(i),
            ];
            if cavern_hollow_m(&m, cavern_value(&m, p)) > Gf::ZERO {
                hollow += 1;
            } else {
                solid += 1;
            }
            i += 1;
        }
        assert!(hollow > 0, "some rooms");
        assert!(solid > hollow, "mostly rock");
        assert_eq!(cavern_hollow_m(&m, Gf::ZERO), Gf::ZERO);
        assert!(cavern_hollow_m(&m, Gf::ONE) > Gf::ZERO);
        let mut total = 0;
        let mut r = 0i64;
        while r < 64 {
            let tubes = tubes_in(&m, [r, r * 7, -r]);
            assert!(tubes.len() <= 3);
            for t in &tubes {
                assert_eq!(t.radius_m, m.caves.tube_radius_m);
                assert!(t.start[0] >= Gf::from_i64(r) * Gf::from_i64(512));
                assert!(t.start[0] < Gf::from_i64(r + 1) * Gf::from_i64(512));
            }
            total += tubes.len();
            r += 1;
        }
        assert!(total > 0, "some regions draw tubes");
        assert_eq!(tube_region(&m, [g(1000.0), g(-1.0), g(0.0)]), [1, -1, 0]);
        assert_eq!(
            tubes_in(&m, [3, 21, -3]),
            tubes_in(&m, [3, 21, -3]),
            "the same region, the same tubes"
        );
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
        assert!(tubes_carve_at(&m, Gf::ONE), "a one-metre cell sees a tube");
        assert!(
            !tubes_carve_at(&m, Gf::from_i64(64)),
            "a 64 m cell does not"
        );
        assert!(caverns_carve_at(&m, Gf::ONE));
        assert!(!caverns_carve_at(&m, Gf::from_i64(1024)));
    }
}
