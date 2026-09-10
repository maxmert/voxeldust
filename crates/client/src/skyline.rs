//! THE SKYLINE (the voxel foundation, slice 8 step 3; ruling V14 D8-1): what the near ground HIDES,
//! per azimuth, as a rigorous lower bound — so a far column is culled only when nothing of it can
//! be seen, and never a column the eye can see into.
//!
//! **The need, MEASURED.** Past the eye's own horizon a column is wanted only where some point of it
//! stands above the line of sight. Step 2 drew that line against the eye's own surface sphere: a
//! column was culled when its peak fell under the tangent that grazes that sphere at the horizon.
//! From the ground stand a ridge 15.5 km out peeked over one 13 km out, and the rung-4 column of
//! the valley floor between them (its peak 90 m UNDER the eye's surface, the tangent 3.9 m OVER
//! it) was culled — one row of sky between two crests, 47 pixels that survived every crossfade.
//! The ground between the eye and that valley lies below the eye's sphere, so the eye sees over
//! it into a valley the sphere would hide: the sphere test is UNSOUND wherever the ground has
//! relief. Its rigorous cousin — the tangent of the sphere lowered by the recipe's whole relief
//! bound (16 km on the home planet) — culls nothing inside the reach: MEASURED 8 775 chunks from
//! the ground against 4 107, every far ring a full annulus.
//!
//! **The structure.** The near ground is what actually hides the far ground, and the descent
//! visits every near column anyway. Each visited column inside the eye's horizon RAISES A WALL:
//! its guaranteed floor (the lowest sampled height less the column bound and the dropped
//! octaves' bound — a radius no point of the column lies under) as a plateau over the column's
//! own FOOTPRINT, the quad of its four bent corners on the ground around the eye. The skyline is
//! a fan of RAYS from the eye's foot, one per tenth of a degree; a ray that crosses the quad
//! leaves it at some ground distance, and a line of sight along that ray at elevation `θ` clears
//! the plateau only where `θ` exceeds the plateau's own elevation at that exit (any point of the
//! plateau the ray crosses bounds its blocking from below, so the exit is a safe choice, and it
//! is shortened by a margin for the chord read as an arc). Each ray keeps the highest wall
//! raised on it. Columns tile the ground — neighbours share their edges exactly — so the rays
//! crossing a ridge of several columns are blocked by each in turn and no gap opens between
//! them; MEASURED before this: a wall over the disc inscribed in each column left an open bin at
//! every corner, and a far column always found one — nothing was culled.
//!
//! A far column CLEARS the skyline when the highest elevation any point of it can reach — its
//! peak bound over its whole span of central angle, the tangent point included — is at or above
//! the LOWEST wall on every ray its circumscribed CAP touches: culled only when blocked at every
//! azimuth it spans. The cap is stated as ANGLES (its centre's central angle from the eye's foot
//! and its own angular radius), never as metres at some radius: MEASURED before this, a disc
//! stated in metres on the surface and read on the eye's chart was too narrow by the ratio of the
//! two radii — a per cent aloft, a quarter from orbit — and the azimuth span of a cap on the
//! sphere is `asin(sin ρ / sin φ)`, half again the flat reading at the orbit stand's limb. Every
//! bound leans the safe way: a wall may be lower than the ground (fewer culls), never higher; a
//! peak may be higher than the ground, never lower; the far cap may be wider than the column,
//! never narrower; a ray's exit shorter, never longer.
//!
//! Everything is exact spherical geometry — a point of radius `r` at central angle `φ` from an
//! eye of radius `e` stands at elevation `atan2(r cos φ − e, r sin φ)` — so no flat-earth sagitta
//! is approximated anywhere; the only approximations are the quad's edges drawn straight on the
//! azimuthal chart, which is why a wall is raised only for a column whose own angular size is
//! small ([`WALL_MAX_HALF_ANGLE`]): the near ground, which is what hides things, and where a
//! straight chord and the geodesic differ by less than the one-per-cent margin on the exit.

use std::f64::consts::{FRAC_PI_2, PI};

use vd_core::glam::{DVec2, DVec3};

/// Rays around the eye: a tenth of a degree apart.
pub const SKYLINE_RAYS: usize = 3600;
/// The margin a ray's exit gives up and the far cap takes on, for chords read as arcs at the
/// column's own scale.
pub const DISC_MARGIN: f64 = 0.01;
/// The largest angular radius of a column that raises a wall: 0.02 rad, about 127 km on the home
/// planet. A larger column's floor is loose anyway (its dropped octaves' bound), and its
/// straight-edged chart quad over-claims ground on the eye's side by the chord's sagitta.
pub const WALL_MAX_HALF_ANGLE: f64 = 0.02;

/// THE EYE'S TANGENT FRAME: up along the eye's radial; north and east across it.
#[derive(Clone, Copy, Debug)]
pub struct EyeFrame {
    /// The eye's radius from the body's centre, in metres.
    pub radius_m: f64,
    up: DVec3,
    north: DVec3,
    east: DVec3,
}

impl EyeFrame {
    /// The frame of an eye at `eye_m` in the body's frame; the eye is never at the centre.
    #[must_use]
    pub fn new(eye_m: DVec3) -> EyeFrame {
        let radius_m = eye_m.length();
        let up = eye_m / radius_m;
        let seed = if up.z.abs() < 0.9 { DVec3::Z } else { DVec3::X };
        let east = up.cross(seed).normalize();
        let north = east.cross(up);
        EyeFrame {
            radius_m,
            up,
            north,
            east,
        }
    }

    /// The ground distance (the arc at the eye's radius) and the azimuth of a direction on the
    /// body, from the eye's foot.
    #[must_use]
    pub fn ground(&self, dir: DVec3) -> (f64, f64) {
        let cos = dir.dot(self.up).clamp(-1.0, 1.0);
        let arc_m = cos.acos() * self.radius_m;
        let az = dir.dot(self.east).atan2(dir.dot(self.north));
        (arc_m, az)
    }

    /// A direction on the body as a point of THE CHART: the azimuthal equidistant plane around the
    /// eye's foot (north up, east right), in metres of ground distance.
    #[must_use]
    pub fn chart(&self, dir: DVec3) -> DVec2 {
        let (arc_m, az) = self.ground(dir);
        DVec2::new(arc_m * az.cos(), arc_m * az.sin())
    }
}

/// The elevation at which a point of radius `r_m` at central angle `phi` from the eye's radial
/// stands, from an eye of radius `eye_m`: exact on the sphere.
#[must_use]
pub fn elevation_rad(eye_m: f64, r_m: f64, phi: f64) -> f64 {
    (r_m * phi.cos() - eye_m).atan2(r_m * phi.sin())
}

/// The azimuth of a ray.
fn ray_az(ray: i64) -> f64 {
    ray as f64 * (2.0 * PI / SKYLINE_RAYS as f64) - PI
}

/// The rays within `beta` of azimuth `az`, as an index range (its ends may wrap; a reader takes
/// them modulo the fan). Every ray when `beta` reaches half a turn.
fn ray_span(az: f64, beta: f64) -> (i64, i64) {
    if beta >= PI {
        return (0, SKYLINE_RAYS as i64 - 1);
    }
    let width = 2.0 * PI / SKYLINE_RAYS as f64;
    let lo = ((az - beta + PI) / width).floor() as i64;
    let hi = ((az + beta + PI) / width).ceil() as i64;
    (lo, hi)
}

/// The azimuth half-width of a cap of angular radius `rho` whose centre stands at central angle
/// `phi` from the eye's foot, on the sphere: `asin(sin ρ / sin φ)`; half a turn when the cap
/// holds the foot or reaches past the antipode's meridian.
#[must_use]
pub fn cap_half_width(phi: f64, rho: f64) -> f64 {
    if phi <= rho {
        return PI;
    }
    let ratio = rho.sin() / phi.sin();
    if ratio >= 1.0 {
        return PI;
    }
    ratio.asin()
}

/// Where a ray from the chart's origin at azimuth `az` LEAVES a convex quad: the farthest of its
/// crossings with the quad's four edges, `None` when it misses the quad.
fn quad_exit_m(quad: &[DVec2; 4], az: f64) -> Option<f64> {
    let u = DVec2::new(az.cos(), az.sin());
    let mut exit: Option<f64> = None;
    let mut i = 0;
    while i < 4 {
        let p = quad[i];
        let v = quad[(i + 1) % 4] - p;
        let denom = u.perp_dot(v);
        // A ray along an edge crosses it nowhere in particular; the two corners belong to the
        // neighbouring edges.
        if denom.abs() > f64::EPSILON {
            let t = p.perp_dot(v) / denom;
            let s = p.perp_dot(u) / denom;
            if (0.0..=1.0).contains(&s) & (t >= 0.0) {
                exit = Some(exit.map_or(t, |e| e.max(t)));
            }
        }
        i += 1;
    }
    exit
}

/// Whether the chart's origin lies inside a convex quad (the eye stands over the column).
fn quad_holds_origin(quad: &[DVec2; 4]) -> bool {
    let mut positive = true;
    let mut negative = true;
    let mut i = 0;
    while i < 4 {
        let side = (quad[(i + 1) % 4] - quad[i]).perp_dot(-quad[i]);
        positive &= side >= 0.0;
        negative &= side <= 0.0;
        i += 1;
    }
    positive | negative
}

/// THE SKYLINE: the highest wall on every ray, as an elevation from the eye.
#[derive(Clone, Debug)]
pub struct Skyline {
    walls: Vec<f64>,
    eye_m: f64,
}

impl Skyline {
    /// An open sky around an eye of radius `eye_m`.
    #[must_use]
    pub fn new(eye_m: f64) -> Skyline {
        Skyline {
            walls: vec![-FRAC_PI_2; SKYLINE_RAYS],
            eye_m,
        }
    }

    /// The wall on one ray.
    #[must_use]
    pub fn wall(&self, ray: usize) -> f64 {
        self.walls[ray % SKYLINE_RAYS]
    }

    /// RAISE A WALL: a plateau of radius `floor_m` over a column's footprint `quad` on the chart.
    /// Every ray that crosses the quad is raised to the plateau's elevation at that ray's exit;
    /// every ray, when the eye stands over the quad. The rays tried are those between the
    /// corners' own azimuths (a convex quad's azimuth extremes are corners): MEASURED before
    /// this, a range taken from the corners' distance to the origin held every ray, always, so
    /// each wall cost the whole fan — 29 million edge tests per recompute from the ground.
    pub fn raise(&mut self, quad: &[DVec2; 4], floor_m: f64) {
        let (lo, hi) = if quad_holds_origin(quad) {
            (0, SKYLINE_RAYS as i64 - 1)
        } else {
            let centre = (quad[0] + quad[1] + quad[2] + quad[3]) * 0.25;
            let az = centre.y.atan2(centre.x);
            let mut beta: f64 = 0.0;
            let mut i = 0;
            while i < 4 {
                let off = quad[i].y.atan2(quad[i].x) - az;
                beta = beta.max(off.sin().atan2(off.cos()).abs());
                i += 1;
            }
            ray_span(az, beta)
        };
        let mut ray = lo;
        while ray <= hi {
            if let Some(exit_m) = quad_exit_m(quad, ray_az(ray)) {
                let theta = elevation_rad(
                    self.eye_m,
                    floor_m,
                    exit_m * (1.0 - DISC_MARGIN) / self.eye_m,
                );
                let i = ray.rem_euclid(SKYLINE_RAYS as i64) as usize;
                self.walls[i] = self.walls[i].max(theta);
            }
            ray += 1;
        }
    }

    /// Whether a column — the cap of angular radius `rho` whose centre stands at central angle
    /// `phi` and azimuth `az` from the eye's foot, no point of it over the radius `peak_m` — can
    /// show over the skyline anywhere in its span: the highest elevation any point of it can
    /// reach, plus `margin` (radians, the caller's hysteresis), against the lowest wall on the
    /// rays the cap touches.
    #[must_use]
    pub fn clears(&self, phi: f64, az: f64, rho: f64, peak_m: f64, margin: f64) -> bool {
        let phi_n = (phi - rho).max(0.0);
        let phi_f = phi + rho;
        let mut theta =
            elevation_rad(self.eye_m, peak_m, phi_n).max(elevation_rad(self.eye_m, peak_m, phi_f));
        // A peak under the eye stands highest at its tangent point, when the span holds it.
        if peak_m < self.eye_m {
            let phi_t = (peak_m / self.eye_m).acos().clamp(phi_n, phi_f);
            theta = theta.max(elevation_rad(self.eye_m, peak_m, phi_t));
        }
        let (lo, hi) = ray_span(az, cap_half_width(phi, rho));
        let mut lowest = FRAC_PI_2;
        let mut ray = lo;
        while ray <= hi {
            lowest = lowest.min(self.wall(ray.rem_euclid(SKYLINE_RAYS as i64) as usize));
            ray += 1;
        }
        theta + margin >= lowest
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const R: f64 = 6_373_000.0;

    /// A square footprint on the chart: `half` metres around a centre `north` and `east` of the
    /// eye's foot.
    fn square(north: f64, east: f64, half: f64) -> [DVec2; 4] {
        [
            DVec2::new(north - half, east - half),
            DVec2::new(north + half, east - half),
            DVec2::new(north + half, east + half),
            DVec2::new(north - half, east + half),
        ]
    }

    fn ray_at(deg: f64) -> usize {
        ((deg.to_radians() + PI) / (2.0 * PI / SKYLINE_RAYS as f64)).round() as usize % SKYLINE_RAYS
    }

    /// A column as a cap: its centre `arc_m` out, `r_m` around it, on the sphere of radius `R`.
    fn cap(arc_m: f64, r_m: f64) -> (f64, f64) {
        (arc_m / R, r_m / R)
    }

    #[test]
    fn the_frame_is_orthonormal_and_reads_ground_distance_azimuth_and_the_chart() {
        let eye = DVec3::new(1.0, 0.31, -0.22).normalize() * (R + 3.4);
        let f = EyeFrame::new(eye);
        assert!((f.up.dot(f.north)).abs() < 1e-12);
        assert!((f.up.dot(f.east)).abs() < 1e-12);
        assert!((f.north.dot(f.east)).abs() < 1e-12);
        assert!((f.north.length() - 1.0).abs() < 1e-12);
        // A point 1 km north stands at azimuth 0 and 1 km; 1 km east at π/2; on the chart they
        // are (1000, 0) and (0, 1000).
        let north = (f.up * (R + 3.4) + f.north * 1000.0).normalize();
        let (d, az) = f.ground(north);
        assert!((d - 1000.0).abs() < 0.01, "{d}");
        assert!(az.abs() < 1e-6, "{az}");
        let c = f.chart(north);
        assert!((c.x - 1000.0).abs() < 0.01, "{c}");
        assert!(c.y.abs() < 0.01, "{c}");
        let east = (f.up * (R + 3.4) + f.east * 1000.0).normalize();
        let (_, az) = f.ground(east);
        assert!((az - FRAC_PI_2).abs() < 1e-6, "{az}");
        let c = f.chart(east);
        assert!(c.x.abs() < 0.01, "{c}");
        assert!((c.y - 1000.0).abs() < 0.01, "{c}");
        // The pole seed switches near the poles.
        let polar = EyeFrame::new(DVec3::new(0.01, 0.0, 1.0).normalize() * R);
        assert!((polar.north.length() - 1.0).abs() < 1e-12);
        // Straight down reads the antipode: half the circumference.
        let (d, _) = f.ground(-f.up);
        assert!((d - PI * (R + 3.4)).abs() < 1e-3);
    }

    #[test]
    fn the_elevation_is_exact_on_the_sphere() {
        // Level ground at the eye's own radius stands below level by half the central angle.
        let e = R + 3.4;
        let phi = 1000.0 / e;
        let theta = elevation_rad(e, R, phi);
        // 3.4 m under over 1 km, and the sphere's own drop of 1 km²/2R = 78 mm.
        let expected = (-(3.4 + 0.0785) / 1000.0_f64).atan();
        assert!((theta - expected).abs() < 1e-6, "{theta} vs {expected}");
        // A point straight below is −90°; a point above the eye at φ = 0 is +90°.
        assert!((elevation_rad(e, R, 0.0) + FRAC_PI_2).abs() < 1e-12);
        assert!((elevation_rad(e, R + 10.0, 0.0) - FRAC_PI_2).abs() < 1e-12);
    }

    #[test]
    fn a_ray_leaves_a_quad_at_its_far_edge_and_misses_one_it_does_not_cross() {
        let q = square(1000.0, 0.0, 31.0);
        // Due north: in at 969, out at 1031.
        let exit = quad_exit_m(&q, 0.0).expect("a crossing");
        assert!((exit - 1031.0).abs() < 1e-9, "{exit}");
        // Past the quad's side: no crossing.
        assert_eq!(quad_exit_m(&q, 0.1), None);
        // A ray along an edge is no crossing of that edge, but it crosses the two it meets.
        let q2 = square(1000.0, 31.0, 31.0);
        let along = quad_exit_m(&q2, 0.0).expect("the corners");
        assert!((along - 1031.0).abs() < 1e-9, "{along}");
        // The eye over a quad: every ray leaves it.
        let own = square(0.0, 5.0, 31.0);
        assert!(quad_holds_origin(&own));
        assert!(!quad_holds_origin(&q));
        let mut ray = 0;
        while ray < SKYLINE_RAYS as i64 {
            assert!(quad_exit_m(&own, ray_az(ray)).is_some(), "ray {ray}");
            ray += 1;
        }
        // The ray spans: a cap of 31 m at 1 km spans about 36 rays; a half turn, every ray.
        let (lo, hi) = ray_span(0.0, cap_half_width(1000.0 / R, 31.0 / R));
        assert!(hi - lo >= 35, "{lo}..{hi}");
        assert!(hi - lo <= 38, "{lo}..{hi}");
        assert_eq!(ray_span(0.0, PI), (0, SKYLINE_RAYS as i64 - 1));
        // The cap's half-width on the sphere: the foot inside it is a half turn; a cap that
        // reaches the antipode's meridian too; far out it is wider than the flat reading
        // (`asin(r / d)`): at 0.7 rad a 0.05 rad cap spans 0.0776, the flat reading 0.0714.
        assert_eq!(cap_half_width(0.001, 0.002), PI);
        assert_eq!(cap_half_width(2.0, 1.5), PI);
        let wide = cap_half_width(0.7, 0.05);
        assert!((wide - (0.05_f64.sin() / 0.7_f64.sin()).asin()).abs() < 1e-12);
        assert!(wide > (0.05_f64 / 0.7).asin());
    }

    #[test]
    fn a_wall_hides_a_far_column_only_at_its_own_azimuths_and_never_a_peak_over_it() {
        let e = R + 3.4;
        let mut sky = Skyline::new(e);
        // Open sky: a valley floor 90 m under the eye's surface, 13.6 km out, clears.
        let far = (
            cap(13_600.0, 1_000.0).0,
            0.0,
            cap(13_600.0, 1_000.0).1,
            R - 90.0,
        );
        assert!(sky.clears(far.0, far.1, far.2, far.3, 0.0));
        // A ridge 2 m over the eye's level, 650 m north, as five tiled 62 m columns: it spans
        // ±13°, the far column's disc ±4.2°, so the valley behind is hidden.
        let mut k = -2;
        while k <= 2 {
            sky.raise(&square(650.0, f64::from(k) * 62.0, 31.0), e + 2.0);
            k += 1;
        }
        assert!(!sky.clears(far.0, far.1, far.2, far.3, 0.0));
        // The same valley east of the ridge is still seen.
        assert!(sky.clears(
            cap(far.0, far.2).0,
            FRAC_PI_2,
            cap(far.0, far.2).1,
            far.3,
            0.0
        ));
        // A far peak standing over the ridge's line is seen: 13.6 km out, the ridge's elevation
        // is about atan(2 / 650) ≈ 0.18°, so a peak 60 m over the eye clears (0.25°), 20 m does
        // not.
        assert!(sky.clears(
            cap(13_600.0, 1_000.0).0,
            0.0,
            cap(13_600.0, 1_000.0).1,
            e + 60.0,
            0.0
        ));
        assert!(!sky.clears(
            cap(13_600.0, 1_000.0).0,
            0.0,
            cap(13_600.0, 1_000.0).1,
            e + 20.0,
            0.0
        ));
        // The wall stands on the rays that cross the ridge (±14.05°) and on no other.
        assert!(sky.wall(ray_at(0.0)) > 0.0);
        assert!(sky.wall(ray_at(13.0)) > 0.0);
        assert!(sky.wall(ray_at(-13.0)) > 0.0);
        assert_eq!(sky.wall(ray_at(15.0)), -FRAC_PI_2);
        // No gap between tiled columns: every ray across the ridge is raised.
        let mut r = ray_at(-13.0);
        while r <= ray_at(13.0) {
            assert!(sky.wall(r) > 0.0, "ray {r}");
            r += 1;
        }
        // A ray's wall reads the plateau at its exit: the centre ray leaves at 681 m, a slanted
        // one later and lower.
        assert!(sky.wall(ray_at(2.5)) < sky.wall(ray_at(0.0)));
    }

    #[test]
    fn the_eye_over_a_column_raises_every_ray_and_a_wall_across_the_seam_is_whole() {
        let e = R + 3.4;
        let mut sky = Skyline::new(e);
        // The eye's own column: a floor 10 m under the surface, the eye 5 m off its centre. Every
        // ray gets a wall, all far under level.
        sky.raise(&square(0.0, 5.0, 31.0), R - 10.0);
        let mut ray = 0;
        while ray < SKYLINE_RAYS {
            let w = sky.wall(ray);
            assert!(w > -FRAC_PI_2, "ray {ray}: {w}");
            assert!(w < 0.0, "ray {ray}: {w}");
            ray += 1;
        }
        // A ridge due south, across the ±π seam: the rays on both sides of the seam rise, and a
        // far column just past the seam is hidden while one due north is seen.
        let mut sky = Skyline::new(e);
        let mut k = -2;
        while k <= 2 {
            sky.raise(&square(-650.0, f64::from(k) * 62.0, 31.0), e + 2.0);
            k += 1;
        }
        assert!(sky.wall(0) > 0.0);
        assert!(sky.wall(SKYLINE_RAYS - 1) > 0.0);
        assert!(!sky.clears(
            cap(13_600.0, 1_000.0).0,
            -PI + 0.001,
            cap(13_600.0, 1_000.0).1,
            R - 90.0,
            0.0
        ));
        assert!(sky.clears(
            cap(13_600.0, 1_000.0).0,
            0.0,
            cap(13_600.0, 1_000.0).1,
            R - 90.0,
            0.0
        ));
        // A far column the eye stands over reads the lowest wall of all: under an all-round
        // crest a low peak is hidden and a high one seen.
        let mut sky = Skyline::new(e);
        sky.raise(&square(0.0, 5.0, 31.0), e + 2.0);
        assert!(!sky.clears(cap(10.0, 31.0).0, 0.0, cap(10.0, 31.0).1, R - 90.0, 0.0));
        assert!(sky.clears(cap(10.0, 31.0).0, 0.0, cap(10.0, 31.0).1, e + 100.0, 0.0));
    }

    #[test]
    fn a_margin_lets_a_column_just_under_the_skyline_through() {
        let e = R + 3.4;
        let mut sky = Skyline::new(e);
        let mut k = -2;
        while k <= 2 {
            sky.raise(&square(650.0, f64::from(k) * 62.0, 31.0), e + 2.0);
            k += 1;
        }
        // A peak 20 m over the eye at 13.6 km is under the ridge's line by about 0.09°: refused
        // with no margin, let through with a 0.2° one; a valley far under stays refused.
        let (phi, rho) = cap(13_600.0, 1_000.0);
        assert!(!sky.clears(phi, 0.0, rho, e + 20.0, 0.0));
        assert!(sky.clears(phi, 0.0, rho, e + 20.0, 0.2_f64.to_radians()));
        assert!(!sky.clears(phi, 0.0, rho, R - 90.0, 0.2_f64.to_radians()));
    }

    #[test]
    fn a_peak_under_the_eye_is_judged_at_its_tangent_point() {
        let e = R + 3.4;
        let mut sky = Skyline::new(e);
        // A wall just under level on every ray: a plateau at the eye's own radius around the eye.
        sky.raise(&square(0.0, 0.0, 31.0), e);
        let wall = sky.wall(0);
        assert!(wall < 0.0, "{wall}");
        assert!(wall > -0.001, "{wall}");
        // A column at the eye's surface radius that spans the tangent point of that radius
        // (6.6 km) reaches its highest there: seen over nothing at all, hidden under the wall.
        let open = Skyline::new(e);
        assert!(open.clears(
            cap(6_500.0, 1_000.0).0,
            0.0,
            cap(6_500.0, 1_000.0).1,
            R,
            0.0
        ));
        assert!(!sky.clears(
            cap(6_500.0, 1_000.0).0,
            0.0,
            cap(6_500.0, 1_000.0).1,
            R,
            0.0
        ));
        // The same column with a peak 10 m over the eye clears the wall (the near edge test): at
        // 5.5 km the sphere drops 2.4 m under the eye's level, so 1 m over the eye does not.
        assert!(sky.clears(
            cap(6_500.0, 1_000.0).0,
            0.0,
            cap(6_500.0, 1_000.0).1,
            e + 10.0,
            0.0
        ));
        assert!(!sky.clears(
            cap(6_500.0, 1_000.0).0,
            0.0,
            cap(6_500.0, 1_000.0).1,
            e + 1.0,
            0.0
        ));
    }
}
