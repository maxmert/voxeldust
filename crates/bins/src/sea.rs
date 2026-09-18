//! ★ THE SEA FROM THE VOLUME (the landform arc, slice 8b stage 6; ruling T8 (a); the design
//! `slice_8b_design.md` §3.1–§3.3).
//!
//! Owns: the ONE solve that turns a body's water inventory into a sea level — a bisection over the
//! body's own shape, sampled at the cell centres of its ladder's coarsest rung, each sample weighted
//! by the solid angle its cell subtends. It runs ONCE, on the realm that owns the body (the shard's
//! boot), and its answer is STORED and STATED as a whole number of millimetres in the charter
//! (`BodyCharter::sea_offset_mm`); a client reads the integer and never re-solves (§3.3).
//!
//! Does NOT own: the water's presence (the charter's `water_km3`, stage 5), whether it is liquid at
//! the surface (the physics crate's own rule, read at the surface temperature and pressure), or the
//! recipe's fluid cell. IT LIVES IN THE BINS CRATE, NOT THE GENERATOR: the generator crate's fence
//! bans every libm-class call (SL10 clause 4) because two hosts must compute the same shape; this
//! solve runs ONCE on the owning realm's host, its answer is stored as an integer both hosts read,
//! and no client ever runs it — so it is a host-side instrument of the draw class, beside the
//! charter's own float→integer door. ★ RULING T8 (a): the recipe's `sea_radius` keeps its old draw until 8c gives
//! the ground its second hump, so the pictures are judged DRY; this module states the level the
//! water WOULD reach, and 8c's one-line switch makes the kernel read it.
//!
//! **THE SAMPLES COME FROM THE LADDER, never from a random sampler** (§3.1): the coarsest rung's
//! cell centres — a fixed list, a fixed order, no rejection loop. Each sample evaluates the FULL
//! field (every octave, the ridge, the roughness, the terrace), because that is the ground the water
//! stands on. **THE BISECTION TAKES A FIXED NUMBER OF STEPS**, never "until converged": a convergence
//! comparison is a drift class. Twenty-four halvings of the field's own span (under 20 km) reach a
//! millimetre.
//!
//! **THE ESCALATION IS STATED, NOT LOOPED** (§3.2): the coarsest rung answers first; when its ocean
//! share falls outside [0.05, 0.95] — where a coarse sample's level error grows — the next finer rung
//! answers once more, and the answer says which rung spoke.
//!
//! **Example.** The home planet holds 2.736 × 10⁹ km³. Its ladder's coarsest rung has 38 cells a
//! face edge, 8 664 samples over the globe. The solve finds the radius at which the water under it
//! equals that volume — on today's one-humped ground, a level over almost every sample — and the
//! shard states the offset from the ladder radius in millimetres. Nothing draws it yet.

use vd_seed::bend::{Face, direction};
use vd_terrain::BodyDefinition;
use vd_terrain::height::height_m;

/// The fixed number of halvings: the field's span (under 20 km on any body the ladder accepts)
/// over 2²⁴ is under a millimetre — finer than the charter's own unit.
pub const SEA_BISECTION_STEPS: u32 = 24;
/// The ocean-share band inside which the coarsest rung's level is trusted (§3.2: the sampling error
/// of the share is under one per cent of the surface near an even split, and the level's error
/// grows only where the height distribution's tail is flat).
pub const SEA_TRUSTED_SHARE: (f64, f64) = (0.05, 0.95);

/// One sample of the shape: a direction, the solid angle its cell subtends, and the surface radius
/// the full field puts there.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SeaSample {
    /// The cell's solid angle in steradians.
    pub weight_sr: f64,
    /// The surface radius under the cell's centre, in metres.
    pub surface_m: f64,
}

/// The solve's answer: the sea's radius, its offset from the ladder radius, the share of the
/// surface under it, and the rung whose cells answered.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SeaLevel {
    /// The sea's radius in metres.
    pub radius_m: f64,
    /// The sea's offset from the ladder radius in metres (the charter stores it in millimetres).
    pub offset_m: f64,
    /// The share of the sampled surface that lies under the sea, by solid angle.
    pub ocean_share: f64,
    /// The rung whose cell centres answered.
    pub rung: u8,
    /// How many samples answered.
    pub samples: usize,
}

/// The solid angle of the spherical triangle `(a, b, c)` on the unit sphere (Van Oosterom &
/// Strackee 1983): `2·atan2(|a · (b × c)|, 1 + a·b + b·c + c·a)`.
#[must_use]
pub fn triangle_solid_angle_sr(a: [f64; 3], b: [f64; 3], c: [f64; 3]) -> f64 {
    let cross = [
        b[1] * c[2] - b[2] * c[1],
        b[2] * c[0] - b[0] * c[2],
        b[0] * c[1] - b[1] * c[0],
    ];
    let triple = a[0] * cross[0] + a[1] * cross[1] + a[2] * cross[2];
    let ab = a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
    let bc = b[0] * c[0] + b[1] * c[1] + b[2] * c[2];
    let ca = c[0] * a[0] + c[1] * a[1] + c[2] * a[2];
    2.0 * triple.abs().atan2(1.0 + ab + bc + ca)
}

/// The face coordinate of the `k`-th of `n` cell edges: `−1 + 2k/n`.
fn edge_coord(k: u32, n: u32) -> f64 {
    -1.0 + 2.0 * f64::from(k) / f64::from(n)
}

/// THE SAMPLES of `body` at `rung`: every cell centre of every face, its solid angle from its four
/// corners (two spherical triangles), and the full field's surface radius under it.
#[must_use]
pub fn samples_at(body: &BodyDefinition, rung: u8) -> Vec<SeaSample> {
    let n = body.ladder().cells_per_edge(rung);
    let mut out = Vec::with_capacity(6 * (n as usize) * (n as usize));
    for face in Face::ALL {
        let mut j = 0;
        while j < n {
            let mut i = 0;
            while i < n {
                let (a0, a1) = (edge_coord(i, n), edge_coord(i + 1, n));
                let (b0, b1) = (edge_coord(j, n), edge_coord(j + 1, n));
                let c00 = direction(face, a0, b0);
                let c10 = direction(face, a1, b0);
                let c11 = direction(face, a1, b1);
                let c01 = direction(face, a0, b1);
                let weight_sr =
                    triangle_solid_angle_sr(c00, c10, c11) + triangle_solid_angle_sr(c00, c11, c01);
                let centre = direction(face, (a0 + a1) / 2.0, (b0 + b1) / 2.0);
                let surface_m = height_m(body, centre, 0);
                out.push(SeaSample {
                    weight_sr,
                    surface_m,
                });
                i += 1;
            }
            j += 1;
        }
    }
    out
}

/// The water that stands under sea radius `rs` over the sampled shape, in cubic metres: for each
/// sample the shell between its surface and the sea, `ω·(rs³ − h³)/3`, where the sea stands over
/// the ground, and nothing where the ground stands over the sea.
#[must_use]
pub fn water_under_m3(samples: &[SeaSample], sea_radius_m: f64) -> f64 {
    let mut sum = 0.0;
    for s in samples {
        let above = sea_radius_m - s.surface_m;
        if above > 0.0 {
            sum += s.weight_sr * (sea_radius_m.powi(3) - s.surface_m.powi(3)) / 3.0;
        }
    }
    sum
}

/// The share of the sampled surface under sea radius `rs`, by solid angle.
#[must_use]
pub fn ocean_share(samples: &[SeaSample], sea_radius_m: f64) -> f64 {
    let mut under = 0.0;
    let mut whole = 0.0;
    for s in samples {
        whole += s.weight_sr;
        if s.surface_m < sea_radius_m {
            under += s.weight_sr;
        }
    }
    under / whole
}

/// THE BISECTION: the sea radius at which the water under it equals `water_m3`, in exactly
/// [`SEA_BISECTION_STEPS`] halvings of the span from the lowest sample to the highest. A volume
/// past what the span can hold answers the top of the span (the whole shape is under water); zero
/// answers the bottom (nothing is).
#[must_use]
pub fn bisect_sea_radius_m(samples: &[SeaSample], water_m3: f64) -> f64 {
    let mut lo = f64::INFINITY;
    let mut hi = f64::NEG_INFINITY;
    for s in samples {
        if s.surface_m < lo {
            lo = s.surface_m;
        }
        if s.surface_m > hi {
            hi = s.surface_m;
        }
    }
    let mut step = 0;
    while step < SEA_BISECTION_STEPS {
        let mid = (lo + hi) / 2.0;
        if water_under_m3(samples, mid) < water_m3 {
            lo = mid;
        } else {
            hi = mid;
        }
        step += 1;
    }
    (lo + hi) / 2.0
}

/// THE SOLVE (§3.1–§3.2): the coarsest rung answers; if its ocean share lies outside the trusted
/// band the next finer rung answers once more. `None` for a body with no water.
#[must_use]
pub fn solve_sea_level(body: &BodyDefinition, water_m3: f64) -> Option<SeaLevel> {
    if water_m3.partial_cmp(&0.0) != Some(core::cmp::Ordering::Greater) {
        return None;
    }
    let top = body.ladder().rungs.saturating_sub(1);
    let first = solve_at(body, top, water_m3);
    let (lo, hi) = SEA_TRUSTED_SHARE;
    let trusted = (first.ocean_share >= lo) & (first.ocean_share <= hi);
    if trusted | (top == 0) {
        return Some(first);
    }
    Some(solve_at(body, top - 1, water_m3))
}

fn solve_at(body: &BodyDefinition, rung: u8, water_m3: f64) -> SeaLevel {
    let samples = samples_at(body, rung);
    let radius_m = bisect_sea_radius_m(&samples, water_m3);
    SeaLevel {
        radius_m,
        offset_m: radius_m - body.ladder().radius_m(),
        ocean_share: ocean_share(&samples, radius_m),
        rung,
        samples: samples.len(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use vd_terrain::BodyFacts;

    fn rock() -> BodyDefinition {
        BodyDefinition::from_seed(5, 3_000.0, BodyFacts::new(2_000, 3_000)).expect("a 3 km rock")
    }

    /// The cells tile the sphere: their solid angles sum to 4π at every rung of the rock's ladder.
    #[test]
    fn the_cells_solid_angles_sum_to_the_whole_sphere() {
        let body = rock();
        // The two coarsest rungs: the finest ones hold millions of cells on even a 3 km rock.
        let top = body.ladder().rungs - 1;
        for rung in [top - 1, top] {
            let whole: f64 = samples_at(&body, rung).iter().map(|s| s.weight_sr).sum();
            let err = (whole - 4.0 * core::f64::consts::PI).abs();
            assert!(err < 1e-9, "rung {rung}: {whole} sr");
        }
    }

    /// A triangle of an eighth of the sphere subtends π/2; a degenerate one nothing.
    #[test]
    fn a_spherical_octant_subtends_half_pi() {
        let sr = triangle_solid_angle_sr([1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]);
        assert!((sr - core::f64::consts::FRAC_PI_2).abs() < 1e-12, "{sr}");
        assert_eq!(
            triangle_solid_angle_sr([1.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]),
            0.0
        );
    }

    /// The bisection lands on the volume it was asked for, to the step's own precision; more water
    /// stands higher; none stands at the lowest ground; a flood stands at the highest.
    #[test]
    fn the_bisection_finds_the_level_that_holds_the_water() {
        let body = rock();
        let top = body.ladder().rungs - 1;
        let samples = samples_at(&body, top);
        let lowest = samples
            .iter()
            .map(|s| s.surface_m)
            .fold(f64::INFINITY, f64::min);
        let highest = samples
            .iter()
            .map(|s| s.surface_m)
            .fold(f64::NEG_INFINITY, f64::max);
        let span = highest - lowest;
        let precision = span / f64::from(1u32 << SEA_BISECTION_STEPS);
        assert!(span > 1.0, "a rock with relief: {span} m");
        let half = water_under_m3(&samples, lowest + span / 2.0);
        let level = bisect_sea_radius_m(&samples, half);
        assert!(
            (level - (lowest + span / 2.0)).abs() <= 2.0 * precision,
            "{level}"
        );
        let held = water_under_m3(&samples, level);
        assert!((held - half).abs() / half < 1e-6, "{held} against {half}");
        let more = bisect_sea_radius_m(&samples, 2.0 * half);
        assert!(more > level, "more water stands higher");
        assert!((bisect_sea_radius_m(&samples, 0.0) - lowest).abs() <= 2.0 * precision);
        let flood = bisect_sea_radius_m(&samples, water_under_m3(&samples, highest) * 4.0);
        assert!((flood - highest).abs() <= 2.0 * precision);
        assert_eq!(ocean_share(&samples, lowest), 0.0);
        assert!((ocean_share(&samples, highest + 1.0) - 1.0).abs() < 1e-12);
    }

    /// THE SOLVE: no water, no sea; a middling inventory answers on the coarsest rung; an inventory
    /// that floods almost everything escalates one rung finer and says so.
    #[test]
    fn the_solve_answers_on_the_coarsest_rung_and_escalates_once_at_the_edges() {
        let body = rock();
        assert_eq!(solve_sea_level(&body, 0.0), None);
        assert_eq!(solve_sea_level(&body, -1.0), None);
        let top = body.ladder().rungs - 1;
        let samples = samples_at(&body, top);
        let lowest = samples
            .iter()
            .map(|s| s.surface_m)
            .fold(f64::INFINITY, f64::min);
        let highest = samples
            .iter()
            .map(|s| s.surface_m)
            .fold(f64::NEG_INFINITY, f64::max);
        let middling = water_under_m3(&samples, (lowest + highest) / 2.0);
        let level = solve_sea_level(&body, middling).expect("a sea");
        assert_eq!(level.rung, top, "the coarsest rung answered");
        assert_eq!(level.samples, samples.len());
        assert!((level.offset_m - (level.radius_m - body.ladder().radius_m())).abs() < 1e-9);
        assert!(level.ocean_share > 0.05 && level.ocean_share < 0.95);
        let flood =
            solve_sea_level(&body, water_under_m3(&samples, highest) * 4.0).expect("a flood");
        assert_eq!(
            flood.rung,
            top - 1,
            "the next finer rung answered the flood"
        );
        assert!(flood.ocean_share > 0.95);
        assert!(flood.samples > samples.len());
    }
}
