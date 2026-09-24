//! ★ THE SKYLINE MARCH AND ITS BREAK RULE — ONE RULE FOR EVERY STAND (`01_reference_target.md`
//! §6.1 and §1.4a; `11_reference_trace.md` §3; ruling B6 item 2, 2026-09-24).
//!
//! **WHY IT EXISTS.** Two instruments read a skyline: `skyline_march` (M1b, four named stations
//! through the height field) and `reference_stand` (the fourth stand, framed like the owner's
//! reference picture, through the SHIPPED column). The reference itself was traced by THIS rule
//! (`11` §3). A second copy of the rule can drift from the first, and then our number and the
//! reference's number come from two instruments again — the defect `01` §2.2 names. So the march and
//! the rule live here, once, and both instruments call them.
//!
//! **THE MARCH.** From an eye at a radius over a direction, for each bearing of a fan, step outward
//! along the great circle (M1b: 50 m steps to 5 km, then 300 m steps, out to 540 km) and keep the
//! greatest elevation angle of the ground over the eye's local horizontal. That greatest angle is
//! the skyline at that bearing. The caller states the ground: a radius ON the marched direction (a
//! field read at any direction), or a drawn COLUMN with its own direction and radius (a cell site,
//! which is where the chunk builder puts its column).
//!
//! **THE BREAK RULE.** `rise(b) = skyline(b) − median(skyline(b − 5) .. skyline(b + 5))` on the 1°
//! bearing grid, and a BREAK is a bearing whose skyline stands over BOTH neighbours AND whose rise
//! is over the threshold. The thresholds are 0.05°, 0.10°, 0.25° and 0.50°.
//!
//! **Example.** A pilot stands on a high point near the belt, 400 m over the ground. To the east a
//! range stands 40 km away, and one of its peaks stands 0.7° over the ridge line on either side of
//! it. The march finds that peak's elevation on its bearing. The rule reads the ±5° median of the
//! skyline there, finds a rise of about 0.7°, and calls the peak a break at all four thresholds,
//! the 0.50° one included.

use std::f64::consts::TAU;

/// The bearings of a full turn on the rule's 1° grid (`01` §6.1; `11` §3 samples the same grid).
pub const BEARINGS: usize = 360;

/// The half-window of the local median, in bearings: ±5° on the 1° grid (`01` §6.1).
pub const HALF_WINDOW: usize = 5;

/// The break thresholds, degrees (`01` §6.1; `11` §0).
pub const THRESHOLDS_DEG: [f64; 4] = [0.05, 0.10, 0.25, 0.50];

/// The march's steps along each bearing, in metres of ground distance (arc length).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Steps {
    /// The first sample and the step up to `near_limit_m`.
    pub near_step_m: f64,
    /// Where the near step hands over to the far step.
    pub near_limit_m: f64,
    /// The step past `near_limit_m`.
    pub far_step_m: f64,
    /// The last sample stands at or under this distance.
    pub limit_m: f64,
}

/// ★ M1b's STEPS (`01` §1.4a, converged): 50 m steps to 5 km, then 300 m steps, out to 540 km.
/// `01` §1.4a measured every station stable from 360 km on, so 540 km is past the convergence.
pub const M1B_STEPS: Steps = Steps {
    near_step_m: 50.0,
    near_limit_m: 5_000.0,
    far_step_m: 300.0,
    limit_m: 540_000.0,
};

/// The eye: its unit direction from the body's centre and its radius, in metres.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Eye {
    pub dir: [f64; 3],
    pub radius_m: f64,
}

/// The fan of bearings: how many over the full turn, and the first one's angle in radians in the
/// eye's tangent basis ([`tangent_basis`]). Bearing `b` stands at `first_rad + b · 2π / bearings`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Fan {
    pub bearings: usize,
    pub first_rad: f64,
}

impl Fan {
    /// A full turn of `bearings`, the first at the basis's own first axis — the M1b fan.
    #[must_use]
    pub const fn full_turn(bearings: usize) -> Fan {
        Fan {
            bearings,
            first_rad: 0.0,
        }
    }
}

/// What the caller's field answers for one marched sample.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Ground {
    /// The surface radius ON the marched direction, in metres (a field read at any direction).
    OnRay(f64),
    /// A drawn column: its own unit direction and its surface radius in metres (a cell site). The
    /// elevation is read from the column's own point, never from the marched direction.
    Column([f64; 3], f64),
    /// The field holds no row for this sample. The sample is skipped and counted.
    Missing,
}

/// A marched skyline.
#[derive(Clone, Debug, PartialEq)]
pub struct Skyline {
    /// The skyline per bearing of the fan, in degrees of elevation over the eye's local horizontal.
    pub deg: Vec<f64>,
    /// The ground distance of the sample that sets each bearing's skyline, in metres.
    pub at_m: Vec<f64>,
    /// How many samples the field could not state.
    pub missing: usize,
    /// How many samples the march read.
    pub samples: usize,
}

fn normalize(v: [f64; 3]) -> [f64; 3] {
    let l = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
    [v[0] / l, v[1] / l, v[2] / l]
}

fn cross(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

fn dot(a: [f64; 3], b: [f64; 3]) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

/// ★ THE EYE'S TANGENT BASIS, `(e1, e2)`: bearing zero along `e1`, bearing 90° along `e2`. The
/// basis M1b's march always used, so a bearing an instrument names on the lattice and the bearing
/// the march samples are one angle.
#[must_use]
pub fn tangent_basis(d: [f64; 3]) -> ([f64; 3], [f64; 3]) {
    let helper = if d[0].abs() < 0.9 {
        [1.0, 0.0, 0.0]
    } else {
        [0.0, 1.0, 0.0]
    };
    let e1 = normalize(cross(d, helper));
    let e2 = cross(d, e1);
    (e1, e2)
}

/// The unit tangent at bearing `phi` (radians) in the basis `(e1, e2)`.
#[must_use]
pub fn bearing_dir(e1: [f64; 3], e2: [f64; 3], phi: f64) -> [f64; 3] {
    let (sp, cp) = phi.sin_cos();
    [
        e1[0] * cp + e2[0] * sp,
        e1[1] * cp + e2[1] * sp,
        e1[2] * cp + e2[2] * sp,
    ]
}

/// ★ THE MARCH (`01` §6.1): the skyline from `eye` over every bearing of `fan`, stepping by `steps`
/// through `ground`, which is asked `(direction, ground distance in metres)` for each sample. The
/// angle around the centre is the ground distance over `body_radius_m`, the ladder radius.
pub fn march(
    eye: &Eye,
    body_radius_m: f64,
    fan: &Fan,
    steps: &Steps,
    ground: &mut dyn FnMut([f64; 3], f64) -> Ground,
) -> Skyline {
    let r = body_radius_m;
    let d = eye.dir;
    let r_c = eye.radius_m;
    let (e1, e2) = tangent_basis(d);
    let mut deg = Vec::with_capacity(fan.bearings);
    let mut at_m = Vec::with_capacity(fan.bearings);
    let mut missing = 0usize;
    let mut samples = 0usize;
    let mut b = 0;
    while b < fan.bearings {
        let phi = fan.first_rad + (b as f64) * TAU / (fan.bearings as f64);
        let t = bearing_dir(e1, e2, phi);
        let mut best = f64::NEG_INFINITY;
        let mut best_s = 0.0;
        let mut s = steps.near_step_m;
        while s <= steps.limit_m {
            let theta = s / r;
            let (st, ct) = theta.sin_cos();
            let dir = normalize([
                d[0] * ct + t[0] * st,
                d[1] * ct + t[1] * st,
                d[2] * ct + t[2] * st,
            ]);
            samples += 1;
            let elev = match ground(dir, s) {
                // M1b's own arithmetic, unchanged: the point stands on the marched direction.
                Ground::OnRay(r_t) => Some((r_t * ct - r_c).atan2(r_t * st)),
                // A column's own point: the rise over the eye's radial against the run across it.
                Ground::Column(cd, r_t) => {
                    let v = [
                        cd[0] * r_t - d[0] * r_c,
                        cd[1] * r_t - d[1] * r_c,
                        cd[2] * r_t - d[2] * r_c,
                    ];
                    let up = dot(v, d);
                    let across = [v[0] - up * d[0], v[1] - up * d[1], v[2] - up * d[2]];
                    Some(up.atan2(dot(across, across).sqrt()))
                }
                Ground::Missing => {
                    missing += 1;
                    None
                }
            };
            if let Some(elev) = elev
                && elev > best
            {
                best = elev;
                best_s = s;
            }
            s += if s < steps.near_limit_m {
                steps.near_step_m
            } else {
                steps.far_step_m
            };
        }
        deg.push(best.to_degrees());
        at_m.push(best_s);
        b += 1;
    }
    Skyline {
        deg,
        at_m,
        missing,
        samples,
    }
}

/// The rise of each bearing's skyline over the median of its ±[`HALF_WINDOW`] window, degrees. The
/// window wraps around the fan, so the fan must be a full turn.
#[must_use]
pub fn rises(sky: &[f64]) -> Vec<f64> {
    let n = sky.len();
    let mut rises = vec![0.0; n];
    for b in 0..n {
        let mut window: Vec<f64> = (0..=2 * HALF_WINDOW)
            .map(|k| sky[(b + n + k - HALF_WINDOW) % n])
            .collect();
        window.sort_by(|x, y| x.partial_cmp(y).expect("finite"));
        let median = window[HALF_WINDOW];
        rises[b] = sky[b] - median;
    }
    rises
}

/// Whether bearing `b` is a BREAK at `threshold_deg`: its skyline stands over both neighbours and
/// its rise is over the threshold.
#[must_use]
pub fn is_break(sky: &[f64], rises: &[f64], b: usize, threshold_deg: f64) -> bool {
    let n = sky.len();
    let local_max = sky[b] > sky[(b + n - 1) % n] && sky[b] > sky[(b + 1) % n];
    local_max && rises[b] > threshold_deg
}

/// ★ THE RULE OVER A FULL TURN (`01` §6.1): the largest rise over the ±5° trend, in degrees, and
/// the breaks per 60° of bearing at each of [`THRESHOLDS_DEG`].
#[must_use]
pub fn judge(sky: &[f64]) -> (f64, [f64; 4]) {
    let n = sky.len();
    let rises = rises(sky);
    let largest = rises.iter().fold(f64::NEG_INFINITY, |a, r| a.max(*r));
    let sixties = n as f64 / 60.0;
    let mut per_60 = [0.0; 4];
    for (k, t) in THRESHOLDS_DEG.iter().enumerate() {
        let count = (0..n).filter(|&b| is_break(sky, &rises, b, *t)).count();
        per_60[k] = count as f64 / sixties;
    }
    (largest, per_60)
}

/// The rule over a SECTOR of a full-turn skyline.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Sector {
    /// The largest rise over the ±5° trend among the sector's bearings, degrees.
    pub largest_rise_deg: f64,
    /// The breaks among the sector's bearings at each of [`THRESHOLDS_DEG`], as counts.
    pub breaks: [usize; 4],
    /// The sector's bearings.
    pub bearings: usize,
}

/// ★ THE RULE OVER A SECTOR: the bearings `centre − half ..= centre + half` of a full-turn skyline
/// (wrapping). Each bearing's ±5° window reads the full turn, so no bearing of the sector loses its
/// window — where the reference picture had to drop 5° at each edge (`11` §3 point 4).
#[must_use]
pub fn judge_sector(sky: &[f64], centre: usize, half: usize) -> Sector {
    let n = sky.len();
    let rises = rises(sky);
    let mut largest = f64::NEG_INFINITY;
    let mut breaks = [0usize; 4];
    for off in 0..=2 * half {
        let b = (centre + n + off - half) % n;
        largest = largest.max(rises[b]);
        for (k, t) in THRESHOLDS_DEG.iter().enumerate() {
            if is_break(sky, &rises, b, *t) {
                breaks[k] += 1;
            }
        }
    }
    Sector {
        largest_rise_deg: largest,
        breaks,
        bearings: 2 * half + 1,
    }
}
