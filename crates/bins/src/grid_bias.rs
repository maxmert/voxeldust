//! ★ THE GRID'S SCRATCHES, AS NUMBERS (ruling B2 step 3; gate G-GRID, 2026-09-22).
//!
//! The owner flew the home planet at 100–150 km and saw ponds in a line along one diagonal. The
//! research report `docs/investigation/2026-09-22/lakes_and_landscape_models.md` §3.4 names the
//! defect the literature already measured: a D8 router on an eight-connected square grid sends
//! every drop along one of eight directions, and a flat — a filled hollow — carries the whole
//! pattern. This module is the INSTRUMENT that turns the picture into three readings a test can
//! fail. It measures the router; it never changes it.
//!
//! 1. **THE CONE** (Tarboton 1997, Water Resour. Res. 33(2), 309–319). A stated field: the ground
//!    rises with the great-circle distance from one node, so every direction is equal. The upslope
//!    area of a point on such a cone is a sector of its own annulus, which is an exact number, so
//!    the router's error is an exact number too. ★ A GRID-ALIGNED PLANE PASSES WHILE THE DEFECT IS
//!    WHOLE — Tarboton measures D8 at 0.065 on the plane and 118.88 on the inward cone — so the
//!    plane is never the gate.
//! 2. **THE ROTATION** (Hyväluoma 2017, IJGIS 31(11), 2272–2285). The cone is the same field at
//!    every azimuth, so a perfect router draws the same picture at every azimuth. Turn the picture
//!    about the cone's own pole and correlate it with itself: a perfect router scores one at every
//!    angle, and a router that follows the GRID instead of the FIELD falls away, worst near 45°.
//! 3. **THE LONG AXIS.** Fit a long axis to every lake patch and to every valley trunk, and
//!    histogram the axes against the face grid's own four directions. A spike at 45° names the
//!    tie-break; a flat histogram is the gate.
//!
//! **Example.** A pilot over the belt sees a line of ponds running north-east, one node wide,
//! straight as a ruler. The long-axis histogram is how that line becomes a number: the 45° bin
//! stands over the mean, and the flatness says by how much.

use std::f64::consts::PI;

use vd_core::glam::DVec3;
use vd_terrain::macro_lattice::{MacroLattice, NO_NODE};

/// The azimuth bins of the polar map the rotation test correlates: 5° each, so a 45° turn is an
/// exact shift of nine bins and no interpolation ever enters the reading.
pub const AZIMUTHS: usize = 72;

/// The great-circle bins of that map between the stated band's ends.
pub const RINGS: usize = 24;

/// The node's unit direction as a plain vector, from the lattice's own fixed-point direction.
#[must_use]
pub fn unit(lattice: &MacroLattice, node: u32) -> DVec3 {
    let d = lattice.direction(node);
    DVec3::new(d[0].raw() as f64, d[1].raw() as f64, d[2].raw() as f64).normalize()
}

/// ★ THE CONE'S FIELD, in sixteenths of a metre: the ground stands `slope` times the great-circle
/// distance from `apex`, so the apex is the one low point and every direction away from it is the
/// same. The whole sphere is one cone: water runs INWARD to the apex near it and OUTWARD from the
/// antipode near that, which is Tarboton's pair of cones in one stated field.
#[must_use]
pub fn cone_z(lattice: &MacroLattice, apex: DVec3, radius_m: f64, slope: f64) -> Vec<i32> {
    let apex = apex.normalize();
    (0..lattice.node_count() as u32)
        .map(|node| {
            let theta = unit(lattice, node).dot(apex).clamp(-1.0, 1.0).acos();
            (slope * radius_m * theta * 16.0).round() as i32
        })
        .collect()
}

/// The great-circle angle of a node from the cone's pole, in radians.
#[must_use]
pub fn theta_of(lattice: &MacroLattice, node: u32, apex: DVec3) -> f64 {
    unit(lattice, node)
        .dot(apex.normalize())
        .clamp(-1.0, 1.0)
        .acos()
}

/// ★ THE EXACT SPECIFIC CATCHMENT AREA at great-circle angle `theta` on a sphere of radius `r`,
/// in metres: everything farther from the pole than `theta` drains through the ring at `theta`, so
/// the upslope area of one sector is `r²(1 + cos θ)dφ` and the ring it crosses is `r sin θ dφ`.
/// The quotient is `r(1 + cos θ)/sin θ`, which is `r·cot(θ/2)` — the sphere's own reading of
/// Tarboton's `r/2` for a small cone.
#[must_use]
pub fn exact_specific_area(radius_m: f64, theta: f64) -> f64 {
    radius_m * (1.0 + theta.cos()) / theta.sin()
}

/// One band's reading of the router against the cone, in Tarboton's own quantity: his Table 2 is
/// *"Differences Between Theoretical and DEM-Computed **Upslope Area** … Bias, Mean(A − Â); MSE,
/// Mean((A − Â)²)"* with the area counted in PIXELS, so ours is counted in NODES. The relative
/// reading is the root of the mean square error over the band's own mean true area, in percent —
/// the one number of the three that two bodies of different node counts may be compared on, because
/// Tarboton's own 16 × 16 domain never reaches a hundred pixels upslope and a planet's node reaches
/// ten thousand.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ConeError {
    pub nodes: usize,
    pub mse: f64,
    pub bias: f64,
    pub relative_pct: f64,
}

/// ★ THE CONE'S ERROR over the band `[lo, hi]` of great-circle angle, measured on the accumulated
/// discharge the solve already holds, in NODES of upslope area (Tarboton's Table 2 quantity).
///
/// The true upslope area of a node at angle `θ` is the sector of everything farther from the pole
/// that its own ring width subtends: `w · r · (1 + cos θ)/sin θ`, with `w` the node's width — the
/// square root of its own true area, the one length the lattice states for a node that is not
/// square. The computed area is the accumulated discharge in square metres. Both are divided by the
/// node's own area, so the reading counts NODES and no metre survives into the square.
///
/// `discharge` must be the accumulation of one millimetre of rain a year on every node, so the
/// number it carries is the upslope area in square metres.
#[must_use]
pub fn cone_error(
    lattice: &MacroLattice,
    discharge: &[u64],
    apex: DVec3,
    radius_m: f64,
    band: (f64, f64),
) -> ConeError {
    let apex = apex.normalize();
    let (mut nodes, mut sum2, mut sum, mut truth) = (0usize, 0.0f64, 0.0f64, 0.0f64);
    for node in 0..lattice.node_count() as u32 {
        let theta = theta_of(lattice, node, apex);
        if theta < band.0 || theta > band.1 {
            continue;
        }
        let area = lattice.area_m2(node) as f64;
        let width = area.sqrt();
        let computed = discharge[node as usize] as f64 / area;
        let want = width * exact_specific_area(radius_m, theta) / area;
        // Tarboton's sign: the TRUE area less the computed one.
        let e = want - computed;
        nodes += 1;
        sum2 += e * e;
        sum += e;
        truth += want;
    }
    let n = nodes.max(1) as f64;
    let mse = sum2 / n;
    ConeError {
        nodes,
        mse,
        bias: sum / n,
        relative_pct: mse.sqrt() * 100.0 / (truth / n).max(1.0e-12),
    }
}

/// ★ THE POLAR MAP the rotation test correlates: the mean specific catchment area in each cell of a
/// ring-by-azimuth grid laid on the cone's own pole. The cone is the same field at every azimuth,
/// so a perfect router fills every cell of a ring with the same number.
#[derive(Clone, Debug, PartialEq)]
pub struct PolarMap {
    /// The cell means, ring-major; `f64::NAN` where no node fell in the cell.
    pub cells: Vec<f64>,
    /// The band of great-circle angle the rings divide.
    pub band: (f64, f64),
}

/// The azimuth of a node about the pole, in radians from `reference`, in `[0, 2π)`.
fn azimuth(v: DVec3, apex: DVec3, reference: DVec3) -> f64 {
    let east = apex.cross(reference).normalize();
    let north = east.cross(apex).normalize();
    let a = v.dot(north).atan2(v.dot(east));
    if a < 0.0 { a + 2.0 * PI } else { a }
}

/// ★ THE MAP of the accumulated field about `apex`, with `reference` fixing the zero azimuth.
#[must_use]
pub fn polar_map(
    lattice: &MacroLattice,
    discharge: &[u64],
    apex: DVec3,
    reference: DVec3,
    band: (f64, f64),
) -> PolarMap {
    let apex = apex.normalize();
    let mut sum = vec![0.0f64; RINGS * AZIMUTHS];
    let mut count = vec![0u32; RINGS * AZIMUTHS];
    let span = band.1 - band.0;
    for node in 0..lattice.node_count() as u32 {
        let v = unit(lattice, node);
        let theta = v.dot(apex).clamp(-1.0, 1.0).acos();
        if theta < band.0 || theta >= band.1 {
            continue;
        }
        let ring = (((theta - band.0) / span) * RINGS as f64) as usize;
        let phi = azimuth(v, apex, reference);
        let slot = ((phi / (2.0 * PI)) * AZIMUTHS as f64) as usize;
        let cell = ring.min(RINGS - 1) * AZIMUTHS + slot.min(AZIMUTHS - 1);
        let width = (lattice.area_m2(node) as f64).sqrt();
        sum[cell] += discharge[node as usize] as f64 / width;
        count[cell] += 1;
    }
    let cells = sum
        .iter()
        .zip(&count)
        .map(|(&s, &c)| if c == 0 { f64::NAN } else { s / f64::from(c) })
        .collect();
    PolarMap { cells, band }
}

impl PolarMap {
    /// ★ HYVÄLUOMA'S ROTATION SCORE at a turn of `slots` azimuth bins: the Pearson correlation of
    /// the map with itself turned about the pole. A perfect router draws a map that depends on the
    /// ring alone, so the turned map is the same map and the score is exactly one at every angle;
    /// a router that follows the grid instead of the field loses the match, worst near 45°.
    /// `f64::NAN` where the map holds fewer than two full cells.
    #[must_use]
    pub fn rotation_score(&self, slots: usize) -> f64 {
        // TWO PASSES, so the means leave before anything is squared: the accumulated areas run to
        // hundreds of millions of square metres a metre of ring, and a one-pass sum of squares
        // loses the very difference this reads.
        let pairs: Vec<(f64, f64)> = (0..RINGS)
            .flat_map(|ring| {
                (0..AZIMUTHS).map(move |slot| {
                    (
                        self.cells[ring * AZIMUTHS + slot],
                        self.cells[ring * AZIMUTHS + (slot + slots) % AZIMUTHS],
                    )
                })
            })
            .filter(|(x, y)| x.is_finite() && y.is_finite())
            .collect();
        let n = pairs.len() as f64;
        let mx = pairs.iter().map(|p| p.0).sum::<f64>() / n;
        let my = pairs.iter().map(|p| p.1).sum::<f64>() / n;
        let (mut cov, mut vx, mut vy) = (0.0f64, 0.0f64, 0.0f64);
        for (x, y) in pairs {
            cov += (x - mx) * (y - my);
            vx += (x - mx) * (x - mx);
            vy += (y - my) * (y - my);
        }
        cov / (vx * vy).sqrt()
    }

    /// ★ THE FOURFOLD AMPLITUDE, in percent: each ring is divided by its own mean, so the ring's
    /// own fall with distance leaves the reading, and the fourth Fourier mode of what remains is
    /// the square grid's own signature (Hyväluoma: *"a fourfold rotational symmetry which reflects
    /// the underlying grid structure"*). A perfect router scores zero.
    #[must_use]
    pub fn fourfold_pct(&self) -> f64 {
        let (mut cos, mut sin, mut n) = (0.0f64, 0.0f64, 0.0f64);
        for ring in 0..RINGS {
            let row = &self.cells[ring * AZIMUTHS..(ring + 1) * AZIMUTHS];
            let full: Vec<f64> = row.iter().copied().filter(|v| v.is_finite()).collect();
            if full.len() < AZIMUTHS {
                continue;
            }
            let mean = full.iter().sum::<f64>() / full.len() as f64;
            if mean <= 0.0 {
                continue;
            }
            for (slot, &v) in row.iter().enumerate() {
                let phi = 2.0 * PI * slot as f64 / AZIMUTHS as f64;
                cos += (v / mean - 1.0) * (4.0 * phi).cos();
                sin += (v / mean - 1.0) * (4.0 * phi).sin();
                n += 1.0;
            }
        }
        if n == 0.0 {
            return f64::NAN;
        }
        2.0 * (cos * cos + sin * sin).sqrt() / n * 100.0
    }
}

/// One shape's long axis: the angle against the face grid's `i` direction in degrees, folded into
/// `[0, 180)`, and the elongation — the square root of the ratio of the two principal spreads, so
/// a perfect disc reads one and a ten-times-long trough reads ten.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct LongAxis {
    pub degrees: f64,
    pub elongation: f64,
}

/// The principal axis of a cloud of face cells: the angle in degrees in `[0, 180)` and the
/// elongation. `None` where the cloud has no spread at all.
fn principal(points: &[(f64, f64)]) -> Option<LongAxis> {
    let n = points.len() as f64;
    let mx = points.iter().map(|p| p.0).sum::<f64>() / n;
    let my = points.iter().map(|p| p.1).sum::<f64>() / n;
    let (mut sxx, mut syy, mut sxy) = (0.0f64, 0.0f64, 0.0f64);
    for &(x, y) in points {
        let (dx, dy) = (x - mx, y - my);
        sxx += dx * dx;
        syy += dy * dy;
        sxy += dx * dy;
    }
    let trace = sxx + syy;
    if trace <= 0.0 {
        return None;
    }
    let root = ((sxx - syy) * (sxx - syy) + 4.0 * sxy * sxy).sqrt();
    let big = (trace + root) / 2.0;
    let small = (trace - root) / 2.0;
    let mut degrees = 0.5 * (2.0 * sxy).atan2(sxx - syy).to_degrees();
    while degrees < 0.0 {
        degrees += 180.0;
    }
    while degrees >= 180.0 {
        degrees -= 180.0;
    }
    Some(LongAxis {
        degrees,
        elongation: (big / small.max(1.0e-12)).sqrt(),
    })
}

/// ★ THE FRAME EVERY LONG AXIS IS STATED IN: the face's own cell indices `(i, j)`. The question the
/// gate asks is whether a shape lies along the GRID's directions, so the grid's own cells are the
/// right ruler and no metre ever enters: a row along `i` reads 0°, a row along `j` reads 90°, and
/// the stencil's diagonal reads exactly 45°. The two directions of a cube face are not at a right
/// angle in metres away from the face's middle, so a reading taken in metres turns a straight row
/// by several degrees — MEASURED, 174° for a row that is 0° (the first form of this instrument).
///
/// A shape that crosses a cube seam is read on the face that holds MOST of it, and the nodes on the
/// other face are not counted; twelve seams on a cube carry a vanishing share of any body's shapes.
#[must_use]
fn face_cells(lattice: &MacroLattice, nodes: &[u32]) -> Vec<(f64, f64)> {
    let mut count = [0usize; 6];
    for &node in nodes {
        let (face, _, _) = lattice.split(node);
        count[face.index() as usize] += 1;
    }
    let home = (0..6)
        .max_by_key(|&f| (count[f], std::cmp::Reverse(f)))
        .unwrap_or(0);
    nodes
        .iter()
        .filter_map(|&node| {
            let (face, i, j) = lattice.split(node);
            (face.index() as usize == home).then_some((f64::from(i), f64::from(j)))
        })
        .collect()
}

/// ★ A LAKE PATCH'S LONG AXIS against its own face grid, in the face's own cells. `None` where the
/// patch has no spread on its own face — a single node has no axis.
#[must_use]
pub fn patch_axis(lattice: &MacroLattice, nodes: &[u32]) -> Option<LongAxis> {
    principal(&face_cells(lattice, nodes))
}

/// ★ A VALLEY TRUNK'S LONG AXIS: the way the river runs over `steps` of its own receiver chain, in
/// the face's own cells. One step is one of eight directions by construction, so one step says
/// nothing; a chain of steps is a line on the ground and its angle is free to take any value.
/// `None` where the chain reaches an outlet, leaves its face, or has no spread.
#[must_use]
pub fn trunk_axis(
    lattice: &MacroLattice,
    receiver: &[u32],
    node: u32,
    steps: usize,
) -> Option<LongAxis> {
    let (face, i, j) = lattice.split(node);
    let mut points = Vec::with_capacity(steps + 1);
    points.push((f64::from(i), f64::from(j)));
    let mut at = node;
    for _ in 0..steps {
        let next = receiver[at as usize];
        if next == NO_NODE {
            return None;
        }
        at = next;
        let (f, i, j) = lattice.split(at);
        if f != face {
            return None;
        }
        points.push((f64::from(i), f64::from(j)));
    }
    principal(&points)
}

/// ★ THE STRAIGHT LINE between two nodes as an angle against the face grid, in the same frame every
/// long axis is stated in. The router's own trunk is compared against this: a flat whose water goes
/// the RIGHT way sends every drop along the line to its own exit, whatever direction that is.
/// `None` where the two nodes do not share a face or stand on one cell.
#[must_use]
pub fn pair_axis(lattice: &MacroLattice, a: u32, b: u32) -> Option<LongAxis> {
    let (fa, ia, ja) = lattice.split(a);
    let (fb, ib, jb) = lattice.split(b);
    if fa != fb {
        return None;
    }
    principal(&[
        (f64::from(ia), f64::from(ja)),
        (f64::from(ib), f64::from(jb)),
    ])
}

/// The angle between two long axes in degrees, in `[0, 90]` — an axis has no head or tail, so half
/// a turn apart is no difference at all.
#[must_use]
pub fn axis_gap(one: f64, other: f64) -> f64 {
    let d = (one - other).abs() % 180.0;
    if d > 90.0 { 180.0 - d } else { d }
}

/// ★ THE LONG-AXIS HISTOGRAM against the face grid's own directions (gate G-GRID). The bins divide
/// the half turn evenly, and the count is stated as a FLATNESS: the biggest bin over the mean bin.
/// A router that has no direction of its own scores one.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct AxisHistogram {
    pub bins: Vec<usize>,
}

impl AxisHistogram {
    /// An empty histogram of `bins` even bins over the half turn.
    #[must_use]
    pub fn new(bins: usize) -> AxisHistogram {
        AxisHistogram {
            bins: vec![0; bins],
        }
    }

    /// Counts one axis. The bin is CENTRED on its own direction, so the 45° bin holds the axes that
    /// lie within half a bin of the stencil's diagonal.
    pub fn add(&mut self, degrees: f64) {
        let width = 180.0 / self.bins.len() as f64;
        let slot = ((degrees + width / 2.0) / width).floor() as isize;
        let n = self.bins.len() as isize;
        let slot = slot.rem_euclid(n) as usize;
        self.bins[slot] += 1;
    }

    /// The axes counted.
    #[must_use]
    pub fn total(&self) -> usize {
        self.bins.iter().sum()
    }

    /// ★ THE FLATNESS: the biggest bin over the mean bin. One is flat; two says one direction holds
    /// twice its share. `f64::NAN` where nothing was counted.
    #[must_use]
    pub fn flatness(&self) -> f64 {
        let total = self.total();
        if total == 0 {
            return f64::NAN;
        }
        let mean = total as f64 / self.bins.len() as f64;
        self.bins.iter().copied().max().unwrap_or(0) as f64 / mean
    }

    /// The bins as a line, each labelled by its own direction in degrees.
    #[must_use]
    pub fn line(&self) -> String {
        let width = 180.0 / self.bins.len() as f64;
        self.bins
            .iter()
            .enumerate()
            .map(|(k, &c)| format!("{:.0}°:{c}", k as f64 * width))
            .collect::<Vec<_>>()
            .join(" ")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use vd_terrain::home::home_moon;
    use vd_terrain::solve::MacroSolve;

    /// The cone the whole measurement stands on, solved once on the home moon: the state with the
    /// stated field routed and accumulated, and the moon's own radius.
    fn moon_cone(apex: DVec3) -> (MacroSolve, f64) {
        let body = home_moon();
        let lattice = MacroLattice::of(&body).expect("the moon has a macro lattice");
        let radius_m = body.radius_m();
        let mut state = MacroSolve::new(&body).expect("the moon has a solve state");
        state.z = cone_z(&lattice, apex, radius_m, 0.01);
        state.route();
        state.accumulate();
        (state, radius_m)
    }

    #[test]
    fn the_exact_specific_area_falls_from_the_pole_to_the_antipode() {
        let r = 1_000.0;
        // Everything drains inward, so the ring near the pole carries the whole sphere and the
        // ring near the antipode carries almost nothing.
        assert!(exact_specific_area(r, 0.1) > exact_specific_area(r, 1.0));
        assert!(exact_specific_area(r, 1.0) > exact_specific_area(r, 3.0));
        // ★ NEAR THE ANTIPODE the cone DIVERGES, and there the sphere's reading is Tarboton's own
        // `r/2` at the same arc: at 0.002 rad from the antipode the arc is 2 m.
        let arc = r * 0.002;
        let out = exact_specific_area(r, PI - 0.002);
        assert!((out - arc / 2.0).abs() / (arc / 2.0) < 1.0e-5, "{out}");
    }

    #[test]
    fn the_cone_field_rises_with_the_distance_from_its_pole() {
        let body = home_moon();
        let lattice = MacroLattice::of(&body).expect("the moon has a macro lattice");
        let apex = unit(&lattice, 0);
        let z = cone_z(&lattice, apex, body.radius_m(), 0.01);
        assert_eq!(z[0], 0, "the pole is the one low point");
        let far = (0..lattice.node_count())
            .max_by_key(|&i| z[i])
            .expect("a highest node");
        let theta = theta_of(&lattice, far as u32, apex);
        assert!(
            theta > 3.0,
            "the highest node is the antipode, not a corner"
        );
    }

    /// ★ THE MEASUREMENT ITSELF, as a test that could fail: the D8 router on a cone carries a real
    /// error, and the error is bounded. The bound is a PIN of what was measured, never a tuning
    /// knob — it moves only when a cure moves it, and the cure states the new number.
    #[test]
    fn the_cone_measures_a_real_d8_error_in_both_bands() {
        let body = home_moon();
        let lattice = MacroLattice::of(&body).expect("the moon has a macro lattice");
        let apex = unit(&lattice, 0);
        let (state, radius_m) = moon_cone(apex);
        let inward = cone_error(
            &lattice,
            &state.discharge,
            apex,
            radius_m,
            (15f64.to_radians(), 45f64.to_radians()),
        );
        let outward = cone_error(
            &lattice,
            &state.discharge,
            apex,
            radius_m,
            (135f64.to_radians(), 165f64.to_radians()),
        );
        assert!(inward.nodes > 1_000, "the inward band holds nodes");
        assert!(outward.nodes > 1_000, "the outward band holds nodes");
        assert!(
            inward.mse > 0.0 && inward.mse < 1.0e6,
            "the inward cone's error: {inward:?}"
        );
        assert!(
            outward.mse > 0.0 && outward.mse < 1.0e4,
            "the outward cone's error: {outward:?}"
        );
        // Tarboton's own reading: the INWARD cone is the harder of the two for D8.
        assert!(
            inward.mse > outward.mse,
            "inward {:?} outward {:?}",
            inward.mse,
            outward.mse
        );
    }

    /// ★ THE ROTATION TEST, as a test that could fail: a perfect router scores one at every angle;
    /// ours does not, and the 45° turn is where it is worst.
    #[test]
    fn the_rotation_score_falls_away_from_a_whole_turn() {
        let body = home_moon();
        let lattice = MacroLattice::of(&body).expect("the moon has a macro lattice");
        let apex = unit(&lattice, 0);
        let (state, _) = moon_cone(apex);
        let reference = unit(&lattice, lattice.index(vd_seed::bend::Face::PosX, 1, 0));
        let map = polar_map(
            &lattice,
            &state.discharge,
            apex,
            reference,
            (20f64.to_radians(), 70f64.to_radians()),
        );
        assert_eq!(map.rotation_score(0), 1.0, "no turn is a perfect match");
        let quarter = map.rotation_score(AZIMUTHS / 4);
        let eighth = map.rotation_score(AZIMUTHS / 8);
        assert!(quarter <= 1.0 && eighth <= 1.0);
        assert!(
            eighth < quarter,
            "the 45° turn is the worse match: 45° {eighth}, 90° {quarter}"
        );
        assert!(map.fourfold_pct() > 0.0, "the grid leaves a fourfold mark");
    }

    /// A map with no node in it reads no score at all, and says so.
    #[test]
    fn an_empty_map_states_no_score() {
        let map = PolarMap {
            cells: vec![f64::NAN; RINGS * AZIMUTHS],
            band: (0.0, 1.0),
        };
        assert!(map.rotation_score(0).is_nan());
        assert!(map.fourfold_pct().is_nan());
    }

    #[test]
    fn a_long_axis_reads_the_grids_own_directions() {
        let body = home_moon();
        let lattice = MacroLattice::of(&body).expect("the moon has a macro lattice");
        let face = vd_seed::bend::Face::PosX;
        // A row of nodes along `i` is a shape whose long axis is the `i` direction, which is 0°.
        let row: Vec<u32> = (10..30).map(|i| lattice.index(face, i, 20)).collect();
        let axis = patch_axis(&lattice, &row).expect("the row has an axis");
        assert!(axis.degrees < 2.0 || axis.degrees > 178.0, "{axis:?}");
        assert!(axis.elongation > 5.0, "{axis:?}");
        // A row along `j` is 90°.
        let col: Vec<u32> = (10..30).map(|j| lattice.index(face, 20, j)).collect();
        let axis = patch_axis(&lattice, &col).expect("the column has an axis");
        assert!((axis.degrees - 90.0).abs() < 2.0, "{axis:?}");
        // The stencil's diagonal is 45°.
        let diag: Vec<u32> = (10..30).map(|k| lattice.index(face, k, k)).collect();
        let axis = patch_axis(&lattice, &diag).expect("the diagonal has an axis");
        assert!((axis.degrees - 45.0).abs() < 3.0, "{axis:?}");
        // A single node has no spread and no axis.
        assert_eq!(patch_axis(&lattice, &[lattice.index(face, 20, 20)]), None);
    }

    #[test]
    fn a_trunk_axis_follows_the_receiver_chain_and_stops_at_an_outlet() {
        let body = home_moon();
        let lattice = MacroLattice::of(&body).expect("the moon has a macro lattice");
        let face = vd_seed::bend::Face::PosX;
        let n = lattice.node_count();
        let mut receiver = vec![NO_NODE; n];
        for i in 10..30 {
            receiver[lattice.index(face, i, 20) as usize] = lattice.index(face, i - 1, 20);
        }
        let axis = trunk_axis(&lattice, &receiver, lattice.index(face, 29, 20), 8)
            .expect("the chain runs eight steps");
        assert!(axis.degrees < 2.0 || axis.degrees > 178.0, "{axis:?}");
        // A chain that reaches an outlet before the eighth step states nothing.
        assert_eq!(
            trunk_axis(&lattice, &receiver, lattice.index(face, 12, 20), 8),
            None
        );
    }

    #[test]
    fn the_histogram_bins_on_the_grids_directions_and_states_its_flatness() {
        let mut hist = AxisHistogram::new(4);
        assert!(hist.flatness().is_nan(), "nothing counted, nothing stated");
        for _ in 0..4 {
            hist.add(45.0);
        }
        hist.add(0.0);
        hist.add(90.0);
        hist.add(179.0);
        assert_eq!(hist.bins, vec![2, 4, 1, 0]);
        assert_eq!(hist.total(), 7);
        assert!((hist.flatness() - 4.0 / (7.0 / 4.0)).abs() < 1.0e-12);
        assert_eq!(hist.line(), "0°:2 45°:4 90°:1 135°:0");
        // A flat histogram scores one.
        let mut flat = AxisHistogram::new(4);
        for d in [0.0, 45.0, 90.0, 135.0] {
            flat.add(d);
        }
        assert!((flat.flatness() - 1.0).abs() < 1.0e-12);
    }

    #[test]
    fn a_straight_line_states_its_own_angle_and_two_axes_state_their_gap() {
        let body = home_moon();
        let lattice = MacroLattice::of(&body).expect("the moon has a macro lattice");
        let face = vd_seed::bend::Face::PosX;
        let row = pair_axis(
            &lattice,
            lattice.index(face, 10, 20),
            lattice.index(face, 30, 20),
        )
        .expect("a row has an angle");
        assert!(row.degrees < 1.0 || row.degrees > 179.0, "{row:?}");
        let diagonal = pair_axis(
            &lattice,
            lattice.index(face, 10, 10),
            lattice.index(face, 30, 30),
        )
        .expect("a diagonal has an angle");
        assert!((diagonal.degrees - 45.0).abs() < 1.0, "{diagonal:?}");
        // One cell has no line at all, and two faces share none.
        let one = lattice.index(face, 10, 10);
        assert_eq!(pair_axis(&lattice, one, one), None);
        let over = lattice.neighbours(lattice.index(face, 0, 20))[3];
        assert_eq!(pair_axis(&lattice, one, over), None);
        // The gap folds the half turn: 179° and 1° are two degrees apart, not a hundred and
        // seventy-eight.
        assert!((axis_gap(179.0, 1.0) - 2.0).abs() < 1.0e-12);
        assert!((axis_gap(0.0, 90.0) - 90.0).abs() < 1.0e-12);
        assert!((axis_gap(40.0, 50.0) - 10.0).abs() < 1.0e-12);
    }

    /// A shape that crosses a cube seam is read on the face that holds most of it, and a chain that
    /// leaves its face states nothing at all.
    #[test]
    fn a_seam_is_read_on_the_face_that_holds_most_of_the_shape() {
        let body = home_moon();
        let lattice = MacroLattice::of(&body).expect("the moon has a macro lattice");
        let face = vd_seed::bend::Face::PosX;
        let over = lattice.neighbours(lattice.index(face, 0, 20))[3];
        assert_ne!(over, NO_NODE, "the row's left neighbour crosses the seam");
        let mut row: Vec<u32> = (0..12).map(|i| lattice.index(face, i, 20)).collect();
        row.push(over);
        let axis = patch_axis(&lattice, &row).expect("the row has an axis");
        assert!(axis.degrees < 2.0 || axis.degrees > 178.0, "{axis:?}");
        let mut receiver = vec![NO_NODE; lattice.node_count()];
        receiver[lattice.index(face, 0, 20) as usize] = over;
        assert_eq!(
            trunk_axis(&lattice, &receiver, lattice.index(face, 0, 20), 1),
            None,
            "a chain that leaves its face states nothing"
        );
    }
}
