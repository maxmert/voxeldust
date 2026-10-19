//! ★ THE SEA'S WAVES (slice S3; rulings B107, B110; `docs/investigation/2026-10-18/sea_waves_design.md`): plane waves in
//! the planet's own space. Each BAND of wavelengths (an octave) holds ONE 1D PROFILE (Jeschke et al. 2018, *Water Surface
//! Wavelets*) — a sum of its wavenumbers with phases from the world's seed, moving by the deep-water dispersion on the
//! universe's own clock — and SIXTEEN FIXED 3D DIRECTIONS, turned once for each band. A component is the profile read at
//! `x · d`, so it exists over the whole sphere with no chart and no seam; each place weighs the directions by its own
//! wind's spread and scales each band by its own spectrum (the shader's part). The waves are look only: no server reads
//! them, and every client builds the same ones from the same seed and the same clock.
//!
//! **Example.** The home planet's band 3 holds the waves 4–8 m long: 32 wavenumbers on a period of 256 m, 1 024 samples;
//! off the coast stand its profile is read along the directions that lie in the sea's plane there, and the 1.9 m/s breeze
//! weighs those that run with it.

use vd_core::glam::{DQuat, DVec3};

/// The bands, one octave each, from [`SHORTEST_M`].
pub const BANDS: usize = 10;
/// The shortest wave of the finest band, metres.
pub const SHORTEST_M: f64 = 0.5;
/// A band's profile's samples over its period.
pub const SAMPLES: usize = 1_024;
/// A band's period, in its longest waves: the wavenumbers of the octave are the period's harmonics 32 to 63.
pub const PERIOD_WAVES: usize = 32;
/// The fixed 3D directions of a band.
pub const DIRECTIONS: usize = 48;
/// The rows of a wave block (`[f32; 4]` each): the header, two rows per band, two rows per band and kept direction (the
/// list padded to [`DIRECTIONS`]), and the profiles' slopes four samples a row.
pub const BLOCK_ROWS: usize = 1 + 2 * BANDS + 2 * BANDS * DIRECTIONS + BANDS * SAMPLES / 4;
/// ★ THE FULLY DEVELOPED SEA's peak (Pierson & Moskowitz 1964): `ω_p = 0.855 g / U` at the 10 m wind; the shader's own
/// constant of the same name reads the same number for the energy.
pub const WAVE_PEAK: f64 = 0.855;
/// The share of a band's spread under which a direction is dropped from the band's list: it adds no visible slope.
pub const SPREAD_KEEP: f64 = 1e-3;
/// The seed's salt for the waves' phases and turns (an identity choice of the world, the same on every client).
const WAVE_SALT: u64 = 0x5741_5645_5300_0001;

/// ★ ONE BAND: its period, its wavenumbers' amplitudes (the saturation slope `k⁻³`, the profile's height variance one),
/// their seeded phases and their deep-water frequencies, and its turned directions.
#[derive(Clone, Debug, PartialEq)]
struct Band {
    period_m: f64,
    amp: Vec<f64>,
    phase: Vec<f64>,
    omega: Vec<f64>,
    dirs: [DVec3; DIRECTIONS],
}

/// ★ THE SEA'S WAVE BANDS of one body: its gravity and its ten bands.
#[derive(Clone, Debug, PartialEq)]
pub struct WaveBands {
    g_m_s2: f64,
    bands: Vec<Band>,
}

impl WaveBands {
    /// The bands of the body of `seed` under the gravity `g_m_s2`.
    #[must_use]
    pub fn new(seed: u64, g_m_s2: f64) -> WaveBands {
        let base = fibonacci_directions();
        let bands = (0..BANDS)
            .map(|b| {
                let mut rng = vd_seed::rng::SplitMix64::new(vd_seed::rng::child_seed(
                    seed, WAVE_SALT, b as u64,
                ));
                let period_m = PERIOD_WAVES as f64 * longest_m(b);
                let k = |j: usize| std::f64::consts::TAU * j as f64 / period_m;
                let js = PERIOD_WAVES..2 * PERIOD_WAVES;
                let raw: Vec<f64> = js.clone().map(|j| k(j).powf(-1.5)).collect();
                let norm = (raw.iter().map(|a| a * a / 2.0).sum::<f64>()).sqrt();
                let amp = raw.iter().map(|a| a / norm).collect();
                let phase = js
                    .clone()
                    .map(|_| rng.next_f64() * std::f64::consts::TAU)
                    .collect();
                let omega = js.map(|j| (g_m_s2 * k(j)).sqrt()).collect();
                let turn = uniform_turn(rng.next_f64(), rng.next_f64(), rng.next_f64());
                let dirs = base.map(|d| turn * d);
                Band {
                    period_m,
                    amp,
                    phase,
                    omega,
                    dirs,
                }
            })
            .collect();
        WaveBands { g_m_s2, bands }
    }

    /// ★ THE WAVE BLOCK's rows for the frame (the band table's, after the sun's row): the header (one, g, the bands,
    /// the samples); each band's period, its wavenumbers' ends and its shortest wave, then its list's length; each
    /// band's LIST of the directions its spread keeps under the wind over the eye (`wind_planet`, the planet's frame,
    /// m/s), the strongest first, each turned into this frame's render space (`facing`) with the eye's own place along
    /// it taken modulo the period in f64 (`eye_planet`, the eye in the planet's frame), and its share of the band's
    /// energy (as its root); and each band's profile's slope at the universe's second `t_s`, four samples a row.
    ///
    /// ★ WHY THE SPREAD IS THE EYE'S (measured 2026-10-18): the spread read at every pixel cost every pixel 48 weights a
    /// band; the waves a pixel can show lie within a few kilometres of the eye, where the wind and the sea's plane do
    /// not change, so the spread is read once, under the eye, and each place keeps its own energy (its own wind's speed
    /// and depth, the shader's part).
    #[must_use]
    pub fn block_rows(
        &self,
        facing: DQuat,
        eye_planet: DVec3,
        t_s: f64,
        wind_planet: DVec3,
    ) -> Vec<[f32; 4]> {
        let tau = std::f64::consts::TAU;
        let mut rows = Vec::with_capacity(BLOCK_ROWS);
        rows.push([1.0, self.g_m_s2 as f32, BANDS as f32, SAMPLES as f32]);
        let up = eye_planet.normalize_or_zero();
        let lists: Vec<Vec<(DVec3, f64)>> = self
            .bands
            .iter()
            .enumerate()
            .map(|(b, band)| self.spread(b, band, up, wind_planet))
            .collect();
        for (b, band) in self.bands.iter().enumerate() {
            rows.push([
                band.period_m as f32,
                (tau / longest_m(b)) as f32,
                (tau / shortest_m(b)) as f32,
                shortest_m(b) as f32,
            ]);
            rows.push([lists[b].len() as f32, 0.0, 0.0, 0.0]);
        }
        for (band, list) in self.bands.iter().zip(&lists) {
            for k in 0..DIRECTIONS {
                let Some(&(d, share)) = list.get(k) else {
                    rows.push([0.0; 4]);
                    rows.push([0.0; 4]);
                    continue;
                };
                let r = facing * d;
                let offset = eye_planet.dot(d).rem_euclid(band.period_m);
                rows.push([r.x as f32, r.y as f32, r.z as f32, offset as f32]);
                rows.push([share.sqrt() as f32, 0.0, 0.0, 0.0]);
            }
        }
        for band in &self.bands {
            let (_, s) = band.profile(t_s);
            for n in (0..SAMPLES).step_by(4) {
                rows.push([
                    s[n] as f32,
                    s[n + 1] as f32,
                    s[n + 2] as f32,
                    s[n + 3] as f32,
                ]);
            }
        }
        rows
    }
}

impl WaveBands {
    /// ★ BAND `b`'s SPREAD over the sea's plane at the normal `n` under the wind `wind` (Hasselmann et al. 1980,
    /// `cos^{2s}(θ/2)`, the exponent `s` by the band's frequency over the peak's, at the fully developed sea's
    /// `U/c_p = 0.855`): each direction whose local wavelength stays within half an octave of the band's (`|d_t|² ≥ ½`,
    /// a smooth taper over a tenth), weighed by its angle to the wind; those under [`SPREAD_KEEP`] of the band dropped;
    /// the rest as shares of one, the strongest first. No wind: every kept direction alike.
    fn spread(&self, b: usize, band: &Band, n: DVec3, wind: DVec3) -> Vec<(DVec3, f64)> {
        let speed = wind.length();
        let to_wind = wind.normalize_or_zero();
        let tau = std::f64::consts::TAU;
        let k_m = (tau / longest_m(b) * tau / shortest_m(b)).sqrt();
        let omega_p = WAVE_PEAK * self.g_m_s2 / speed.max(1e-9);
        let ratio = (k_m * self.g_m_s2).sqrt() / omega_p;
        let s = if ratio < 1.0 {
            6.97 * ratio.powf(4.06)
        } else {
            9.77 * ratio.powf(-1.873)
        };
        let mut weighed: Vec<(DVec3, f64)> = band
            .dirs
            .iter()
            .filter_map(|&d| {
                let dt = d - d.dot(n) * n;
                let t2 = dt.length_squared();
                let taper = ((t2 - 0.5) / 0.1).clamp(0.0, 1.0);
                let lean = if speed > 1e-9 {
                    (0.5 + 0.5 * dt.dot(to_wind) / t2.sqrt()).powf(s)
                } else {
                    1.0
                };
                let w = lean * taper;
                (w > 0.0).then_some((d, w))
            })
            .collect();
        let all: f64 = weighed.iter().map(|(_, w)| w).sum();
        weighed.retain(|(_, w)| *w >= SPREAD_KEEP * all);
        weighed.sort_by(|a, b| b.1.total_cmp(&a.1));
        let kept: f64 = weighed.iter().map(|(_, w)| w).sum();
        weighed.into_iter().map(|(d, w)| (d, w / kept)).collect()
    }
}

impl Band {
    /// The profile at the universe's second `t_s`: its height (variance one) and its slope along `s`, per sample — the
    /// two inverse transforms of the band's wavenumbers, each phase `φ − ω t` taken modulo a turn in f64.
    fn profile(&self, t_s: f64) -> (Vec<f64>, Vec<f64>) {
        let tau = std::f64::consts::TAU;
        let (mut hr, mut hi) = (vec![0.0; SAMPLES], vec![0.0; SAMPLES]);
        let (mut sr, mut si) = (vec![0.0; SAMPLES], vec![0.0; SAMPLES]);
        for (m, ((a, p), w)) in self
            .amp
            .iter()
            .zip(&self.phase)
            .zip(&self.omega)
            .enumerate()
        {
            let j = PERIOD_WAVES + m;
            let psi = (p - (w * t_s).rem_euclid(tau)).rem_euclid(tau);
            let k = tau * j as f64 / self.period_m;
            let (c, s) = (a * psi.cos(), a * psi.sin());
            hr[j] = c;
            hi[j] = s;
            // The slope's coefficient: i·k times the height's.
            sr[j] = -k * s;
            si[j] = k * c;
        }
        inverse_dft(&mut hr, &mut hi);
        inverse_dft(&mut sr, &mut si);
        (hr, sr)
    }
}

/// The longest wave of band `b`, metres.
#[must_use]
pub fn longest_m(b: usize) -> f64 {
    2.0 * shortest_m(b)
}

/// The shortest wave of band `b`, metres.
#[must_use]
pub fn shortest_m(b: usize) -> f64 {
    SHORTEST_M * f64::from(1u32 << b)
}

/// ★ SIXTEEN DIRECTIONS EVENLY OVER THE SPHERE: the spherical Fibonacci set (`z` in even steps, the longitude by the
/// golden angle).
fn fibonacci_directions() -> [DVec3; DIRECTIONS] {
    let golden = std::f64::consts::PI * (3.0 - 5.0_f64.sqrt());
    std::array::from_fn(|i| {
        let z = 1.0 - (2 * i + 1) as f64 / DIRECTIONS as f64;
        let r = (1.0 - z * z).sqrt();
        let a = golden * i as f64;
        DVec3::new(r * a.cos(), r * a.sin(), z)
    })
}

/// ★ A UNIFORM TURN from three uniform numbers in [0, 1) (Shoemake 1992).
fn uniform_turn(u1: f64, u2: f64, u3: f64) -> DQuat {
    let tau = std::f64::consts::TAU;
    let (a, b) = ((1.0 - u1).sqrt(), u1.sqrt());
    DQuat::from_xyzw(
        a * (tau * u2).sin(),
        a * (tau * u2).cos(),
        b * (tau * u3).sin(),
        b * (tau * u3).cos(),
    )
    .normalize()
}

/// ★ THE INVERSE TRANSFORM in place, `x_n = Σ_j X_j e^{+2πi jn/N}` (no scale), radix 2 with the bit-reversed order; `N`
/// a power of two.
fn inverse_dft(re: &mut [f64], im: &mut [f64]) {
    let n = re.len();
    let bits = n.trailing_zeros();
    for i in 0..n {
        let j = i.reverse_bits() >> (usize::BITS - bits);
        if j > i {
            re.swap(i, j);
            im.swap(i, j);
        }
    }
    let mut len = 2;
    while len <= n {
        let step = std::f64::consts::TAU / len as f64;
        for start in (0..n).step_by(len) {
            for k in 0..len / 2 {
                let (wr, wi) = ((step * k as f64).cos(), (step * k as f64).sin());
                let (a, b) = (start + k, start + k + len / 2);
                let tr = re[b] * wr - im[b] * wi;
                let ti = re[b] * wi + im[b] * wr;
                re[b] = re[a] - tr;
                im[b] = im[a] - ti;
                re[a] += tr;
                im[a] += ti;
            }
        }
        len <<= 1;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// ★ THE TRANSFORM IS THE SUM: the inverse transform of a few wavenumbers equals their direct sum at every sample.
    #[test]
    fn the_transform_is_the_direct_sum() {
        let n = 64;
        let (mut re, mut im) = (vec![0.0; n], vec![0.0; n]);
        re[3] = 0.5;
        im[3] = -0.25;
        re[10] = 1.0;
        let direct = |x: usize| {
            let tau = std::f64::consts::TAU;
            let a = tau * (3 * x) as f64 / n as f64;
            let b = tau * (10 * x) as f64 / n as f64;
            0.5 * a.cos() + 0.25 * a.sin() + b.cos()
        };
        inverse_dft(&mut re, &mut im);
        for (x, v) in re.iter().enumerate() {
            assert!((v - direct(x)).abs() < 1e-12, "{x}: {v} {}", direct(x));
        }
    }

    /// ★ A PROFILE HAS HEIGHT VARIANCE ONE and the slope its wavenumbers carry, and it moves with the time: the same
    /// world and time give the same bytes, another time other ones.
    #[test]
    fn a_profile_has_variance_one_and_moves_with_the_time() {
        let waves = WaveBands::new(4_030_111_653_607_004_909, 9.81);
        let band = &waves.bands[3];
        let (h, s) = band.profile(1_000.0);
        let var = h.iter().map(|v| v * v).sum::<f64>() / SAMPLES as f64;
        assert!((var - 1.0).abs() < 1e-9, "{var}");
        let k_lo = std::f64::consts::TAU / longest_m(3);
        let k_hi = std::f64::consts::TAU / shortest_m(3);
        let slope_var = s.iter().map(|v| v * v).sum::<f64>() / SAMPLES as f64;
        assert!(
            slope_var > k_lo * k_lo && slope_var < k_hi * k_hi,
            "{slope_var}"
        );
        assert_eq!(band.profile(1_000.0), (h.clone(), s));
        assert_ne!(band.profile(1_001.0).0, h);
        assert_eq!(WaveBands::new(4_030_111_653_607_004_909, 9.81), waves);
        assert_ne!(WaveBands::new(1, 9.81).bands[3].phase, band.phase);
    }

    /// ★ THE BLOCK: its rows in order — the header, the bands, the directions turned into render space with the eye's
    /// place along each modulo the period, the profiles.
    #[test]
    fn the_block_holds_the_header_the_bands_the_directions_and_the_profiles() {
        let waves = WaveBands::new(7, 9.81);
        let facing = DQuat::from_rotation_z(0.3);
        let eye = DVec3::new(6.0e6, -1.0e6, 2.0e6);
        let up = eye.normalize();
        let wind = DVec3::Z.cross(up).normalize() * 7.0;
        let rows = waves.block_rows(facing, eye, 12.5, wind);
        assert_eq!(rows.len(), BLOCK_ROWS);
        assert_eq!(rows[0], [1.0, 9.81, BANDS as f32, SAMPLES as f32]);
        assert_eq!(rows[1][3], SHORTEST_M as f32);
        assert_eq!(rows[1][0], (PERIOD_WAVES as f64 * 1.0) as f32);
        // The first band's list: its strongest direction first, turned, with the eye's place along it.
        let list = waves.spread(0, &waves.bands[0], up, wind);
        assert_eq!(rows[2][0], list.len() as f32);
        let (d, share) = list[0];
        let r = facing * d;
        let row = rows[1 + 2 * BANDS];
        assert_eq!(
            [row[0], row[1], row[2]],
            [r.x as f32, r.y as f32, r.z as f32]
        );
        assert_eq!(
            row[3],
            eye.dot(d).rem_euclid(waves.bands[0].period_m) as f32
        );
        assert_eq!(rows[2 + 2 * BANDS][0], share.sqrt() as f32);
        let (_, s) = waves.bands[0].profile(12.5);
        let first = rows[1 + 2 * BANDS + 2 * BANDS * DIRECTIONS];
        assert_eq!(first, [s[0] as f32, s[1] as f32, s[2] as f32, s[3] as f32]);
    }

    /// ★ THE DIRECTIONS cover the sphere evenly (unit length, their mean near the centre), and a turn keeps their length.
    #[test]
    fn the_directions_are_even_and_a_turn_keeps_them_unit() {
        let dirs = fibonacci_directions();
        let mean = dirs.iter().fold(DVec3::ZERO, |a, d| a + *d) / DIRECTIONS as f64;
        assert!(mean.length() < 0.1, "{mean}");
        let turn = uniform_turn(0.2, 0.7, 0.4);
        for d in dirs {
            assert!((d.length() - 1.0).abs() < 1e-12);
            assert!(((turn * d).length() - 1.0).abs() < 1e-12);
        }
    }

    /// ★ THE SPREAD UNDER THE EYE: the kept directions lie in the sea's plane, lean with the wind, their shares sum to
    /// one, the strongest first; with no wind every kept direction weighs alike or less (the taper).
    #[test]
    fn the_spread_keeps_the_directions_in_the_plane_that_run_with_the_wind() {
        let waves = WaveBands::new(11, 9.81);
        let n = DVec3::new(0.3, -0.2, 0.93).normalize();
        let wind = DVec3::Z.cross(n).normalize() * 8.0;
        for b in 0..BANDS {
            let list = waves.spread(b, &waves.bands[b], n, wind);
            assert!(!list.is_empty(), "band {b}");
            let sum: f64 = list.iter().map(|(_, w)| w).sum();
            assert!((sum - 1.0).abs() < 1e-12, "band {b}: {sum}");
            assert!(list.windows(2).all(|p| p[0].1 >= p[1].1));
            for (d, _) in &list {
                assert!(
                    d.dot(n).abs() <= 0.5_f64.sqrt() + 1e-9,
                    "band {b}: off the plane"
                );
            }
            assert!(
                list[0].0.dot(wind) > 0.0,
                "band {b}: the strongest runs with the wind"
            );
        }
        let still = waves.spread(4, &waves.bands[4], n, DVec3::ZERO);
        assert!(still.iter().all(|(_, w)| *w <= still[0].1));
    }
}
