//! ★ L1 — THE SUN BY LATITUDE AND SEASON (the design §3.1). The daily-mean sunlight on a level surface at the top of
//! the air, by exact geometry:
//!
//! `Q = (S/π)·(h₀·sin φ·sin δ + cos φ·cos δ·sin h₀)`, `cos h₀ = −tan φ·tan δ`, `sin δ = sin ε·sin λ`,
//!
//! `φ` the latitude, `δ` the sun's declination, `h₀` the hour angle of sunset, `ε` the tilt and `λ` the season's angle
//! along the year. Where `−tan φ·tan δ` passes 1 the sun does not rise (`h₀ = 0`, the polar night), and where it passes
//! −1 it does not set (`h₀ = π`, the polar day): the formula's own domain, never a clamp. No constant.
//!
//! **THE ORBIT.** A pass of the macro solve (~12 Myr) is far longer than a precession cycle (~20 kyr), so the landscape
//! sees the PRECESSION-AVERAGED year: the season's angle runs evenly in time, both hemispheres' seasons are equal, and
//! the year's mean flux is the orbit's time mean, `S/√(1 − e²)` for the flux `S` at the mean distance. A law of the
//! orbit's average, not a missing fact.
//!
//! **THE FENCE.** The year is `N = 2^shift` even samples, each at the MIDDLE of its own `1/N` of the year (so a season's
//! mean is the midpoint rule, its error falling as `1/N²`). Their angles come from square roots alone — `cos(π/2) = 0`
//! halved by `cos(θ/2) = √((1 + cos θ)/2)`, `sin(θ/2) = sin θ/(2·cos(θ/2))`, then one rotation a step on the circle of
//! `2N` — and the hour angle from [`Gf::acos`]. One answer on every target.
//!
//! **Example.** At Earth's tilt the June pole gets `S·sin 23.44°` = 541 W/m² and the equator `(S/π)·cos 23.44°` =
//! 397 W/m²; at the home planet's 55° with 0.75 of Earth's flux the June pole gets 836 W/m² and the equator 186 W/m².

use crate::gf::Gf;

/// The year's samples as a power of two (256): fine enough that the season's mean over a sixteenth of the year moves
/// under a thousandth between this and twice the samples (the table's test reads it).
pub const YEAR_SAMPLES_SHIFT: u32 = 8;

/// The cosine and sine of `2π·j/N` for every sample `j` of a year of `N = 2^shift` samples (`shift ≥ 2`), from square
/// roots and one rotation a sample.
#[must_use]
pub fn year_circle(shift: u32) -> Vec<(Gf, Gf)> {
    let n = 1usize << shift;
    let (mut c, mut s) = (Gf::ZERO, Gf::ONE);
    for _ in 2..shift {
        let half_c = ((Gf::ONE + c) * Gf::HALF).sqrt();
        s = s / (Gf::TWO * half_c);
        c = half_c;
    }
    let mut out = Vec::with_capacity(n);
    let (mut cj, mut sj) = (Gf::ONE, Gf::ZERO);
    for _ in 0..n {
        out.push((cj, sj));
        (cj, sj) = (cj * c - sj * s, sj * c + cj * s);
    }
    out
}

/// The daily-mean sunlight, W/m², at the latitude whose sine and cosine are `sin_lat`, `cos_lat` (`cos_lat ≥ 0`) when
/// the sun's declination has sine and cosine `sin_dec`, `cos_dec` (`cos_dec ≥ 0`), for the flux `s_w_m2`.
#[must_use]
pub fn daily_mean_w_m2(s_w_m2: Gf, sin_lat: Gf, cos_lat: Gf, sin_dec: Gf, cos_dec: Gf) -> Gf {
    let pi = Gf::from_f64(std::f64::consts::PI);
    let up = sin_lat * sin_dec;
    let across = cos_lat * cos_dec;
    // The hour angle of sunset and its sine: the polar day and night are where `−up/across` leaves [−1, 1] (or where
    // `across` is nought: a pole, or a sun over a pole).
    let (h0, sin_h0) = if across <= Gf::ZERO {
        if up > Gf::ZERO {
            (pi, Gf::ZERO)
        } else {
            (Gf::ZERO, Gf::ZERO)
        }
    } else {
        let x = -up / across;
        if x >= Gf::ONE {
            (Gf::ZERO, Gf::ZERO)
        } else if x <= -Gf::ONE {
            (pi, Gf::ZERO)
        } else {
            (x.acos(), (Gf::ONE - x * x).sqrt())
        }
    };
    s_w_m2 / pi * (h0 * up + across * sin_h0)
}

/// ★ THE SUN'S TABLE of one body: the daily-mean sunlight at `rows` latitudes even in the sine (row `r` at
/// `sin φ = −1 + 2r/(rows − 1)`, so each row stands for an equal area) and at every sample of the year. The energy
/// balance reads its harmonics, the seasons their means; both by linear reading in the sine of the latitude.
#[derive(Clone, Debug)]
pub struct SunTable {
    rows: usize,
    samples: usize,
    /// `rows × samples` daily means, W/m², row-major.
    q: Vec<Gf>,
    /// The circle of `2N` angles: sample `j` stands at angle `2j + 1` of it, and the harmonic `k` of sample `j` at
    /// `k·(2j + 1)` of it.
    circle: Vec<(Gf, Gf)>,
}

impl SunTable {
    /// The table for the flux `s_w_m2` at the orbit's mean distance, the eccentricity `ecc` and the tilt whose cosine is
    /// `cos_tilt`, on `rows ≥ 2` latitudes and `2^shift` samples a year. The season's angle `λ = 2π·(j + ½)/N`; `λ = 0`
    /// is the northern spring equinox.
    #[must_use]
    pub fn new(s_w_m2: Gf, ecc: Gf, cos_tilt: Gf, rows: usize, shift: u32) -> SunTable {
        let s = s_w_m2 / (Gf::ONE - ecc * ecc).sqrt();
        let sin_tilt = (Gf::ONE - cos_tilt * cos_tilt).greater(Gf::ZERO).sqrt();
        let circle = year_circle(shift + 1);
        let samples = circle.len() >> 1;
        let last = Gf::from_i64(rows as i64 - 1);
        let decs: Vec<(Gf, Gf)> = circle
            .iter()
            .skip(1)
            .step_by(2)
            .map(|&(_, sl)| {
                let sd = sin_tilt * sl;
                (sd, (Gf::ONE - sd * sd).sqrt())
            })
            .collect();
        let mut q = Vec::with_capacity(rows * samples);
        for r in 0..rows {
            let sin_lat = -Gf::ONE + Gf::TWO * Gf::from_i64(r as i64) / last;
            let cos_lat = (Gf::ONE - sin_lat * sin_lat).greater(Gf::ZERO).sqrt();
            for &(sd, cd) in &decs {
                q.push(daily_mean_w_m2(s, sin_lat, cos_lat, sd, cd));
            }
        }
        SunTable {
            rows,
            samples,
            q,
            circle,
        }
    }

    /// The samples a year.
    #[must_use]
    pub fn samples(&self) -> usize {
        self.samples
    }

    /// The row below `sin_lat` and the share toward the row above it.
    fn at(&self, sin_lat: Gf) -> (usize, Gf) {
        let t = (sin_lat + Gf::ONE) * Gf::from_i64(self.rows as i64 - 1) * Gf::HALF;
        let i = t.floor().to_i64_floor().clamp(0, self.rows as i64 - 2) as usize;
        (i, t - Gf::from_i64(i as i64))
    }

    /// The daily mean at `sin_lat` and the year's sample `j`.
    #[must_use]
    pub fn at_sample(&self, sin_lat: Gf, j: usize) -> Gf {
        let (i, w) = self.at(sin_lat);
        let a = self.q[i * self.samples + j];
        let b = self.q[(i + 1) * self.samples + j];
        a + (b - a) * w
    }

    /// ★ THE HARMONIC `k` of the year at `sin_lat`: `(a_k, b_k)` with `Q(t) = a₀ + Σ a_k·cos kλ + b_k·sin kλ` (`b₀ = 0`;
    /// `a₀` the annual mean), by the samples' own sums.
    #[must_use]
    pub fn harmonic(&self, sin_lat: Gf, k: usize) -> (Gf, Gf) {
        let (i, w) = self.at(sin_lat);
        let n = self.samples;
        let (mut a, mut b) = (Gf::ZERO, Gf::ZERO);
        for j in 0..n {
            let (c, s) = self.circle[(k * (2 * j + 1)) & (2 * n - 1)];
            let qa = self.q[i * n + j];
            let q = qa + (self.q[(i + 1) * n + j] - qa) * w;
            a += q * c;
            b += q * s;
        }
        let scale = if k == 0 {
            Gf::ONE / Gf::from_i64(n as i64)
        } else {
            Gf::TWO / Gf::from_i64(n as i64)
        };
        (a * scale, b * scale)
    }

    /// The mean over the season `m` of `2^seasons_shift` even seasons (season 0 starts at the northern spring equinox)
    /// at `sin_lat`; the seasons' count divides the samples.
    #[must_use]
    pub fn season_mean(&self, sin_lat: Gf, m: usize, seasons_shift: u32) -> Gf {
        let per = self.samples >> seasons_shift;
        let mut sum = Gf::ZERO;
        for j in m * per..(m + 1) * per {
            sum += self.at_sample(sin_lat, j);
        }
        sum / Gf::from_i64(per as i64)
    }
}

#[cfg(test)]
mod tests {
    #![allow(
        clippy::integer_division,
        reason = "a test states an exact quotient; never a kernel's path"
    )]
    use super::*;

    const EARTH_S: f64 = 1_361.0;

    fn close(got: Gf, want: f64, tol: f64) {
        assert!(
            (got.to_f64() - want).abs() < tol,
            "{} against {want}",
            got.to_f64()
        );
    }

    /// The circle's samples are the angles' own cosines and sines (literal values at the quarter and eighth).
    #[test]
    fn the_year_circle_stands_on_the_angles() {
        let c = year_circle(YEAR_SAMPLES_SHIFT);
        assert_eq!(c.len(), 256);
        close(c[0].0, 1.0, 1e-15);
        close(c[64].0, 0.0, 1e-14);
        close(c[64].1, 1.0, 1e-14);
        close(c[128].0, -1.0, 1e-14);
        close(c[32].0, std::f64::consts::FRAC_1_SQRT_2, 1e-14);
        close(c[192].1, -1.0, 1e-14);
        assert_eq!(year_circle(2).len(), 4);
    }

    /// ★ EARTH'S SOLSTICE AND EQUINOX by the geometry: the June pole `S·sin ε`, the June equator `(S/π)·cos ε`, the
    /// December pole's night, the equinox equator `S/π`; and the home planet's June pole and equator (the module's
    /// example).
    #[test]
    fn the_daily_sun_at_the_known_places() {
        let s = Gf::from_f64(EARTH_S);
        let (sd, cd) = (Gf::from_f64(0.397_777_2), Gf::from_f64(0.917_482_1)); // sin, cos 23.44°
        close(daily_mean_w_m2(s, Gf::ONE, Gf::ZERO, sd, cd), 541.4, 0.2);
        close(daily_mean_w_m2(s, Gf::ZERO, Gf::ONE, sd, cd), 397.5, 0.2);
        close(daily_mean_w_m2(s, Gf::ONE, Gf::ZERO, -sd, cd), 0.0, 1e-12);
        close(
            daily_mean_w_m2(s, Gf::ZERO, Gf::ONE, Gf::ZERO, Gf::ONE),
            EARTH_S / std::f64::consts::PI,
            1e-9,
        );
        // 70° N in June: the sun does not set (−tan 70°·tan 23.44° < −1); 70° S: it does not rise.
        let (s70, c70) = (Gf::from_f64(0.939_692_6), Gf::from_f64(0.342_020_1));
        let day = daily_mean_w_m2(s, s70, c70, sd, cd);
        close(day, EARTH_S * 0.939_692_6 * 0.397_777_2, 1e-9);
        close(daily_mean_w_m2(s, -s70, c70, sd, cd), 0.0, 1e-12);
        // 40° N in June: a sunset hour angle inside the domain.
        let q40 = daily_mean_w_m2(
            s,
            Gf::from_f64(0.642_787_6),
            Gf::from_f64(0.766_044_4),
            sd,
            cd,
        );
        close(q40, 498.86, 0.05);
        // The home planet: 0.75 of Earth's flux, tilt 55°.
        let sh = Gf::from_f64(0.75 * EARTH_S);
        let (s55, c55) = (Gf::from_f64(0.819_152_0), Gf::from_f64(0.573_576_4));
        close(daily_mean_w_m2(sh, Gf::ONE, Gf::ZERO, s55, c55), 836.1, 0.5);
        close(daily_mean_w_m2(sh, Gf::ZERO, Gf::ONE, s55, c55), 186.3, 0.5);
    }

    /// ★ THE TABLE'S YEAR: the global annual mean is `S/4` (the area mean of the even rows), Earth's annual equator and
    /// pole (415.5 and 172.3 W/m²), the hemispheres' seasons mirror each other half a year apart, the first harmonic is the
    /// season's swing, and sixteen seasons' means move under a thousandth between 256 and 512 samples.
    #[test]
    fn the_table_holds_the_year() {
        let rows = 513;
        let earth = SunTable::new(
            Gf::from_f64(EARTH_S),
            Gf::ZERO,
            Gf::from_f64(0.917_482_1),
            rows,
            YEAR_SAMPLES_SHIFT,
        );
        assert_eq!(earth.samples(), 256);
        // The area mean: the trapezoid over the even sine rows.
        let mut sum = Gf::ZERO;
        for r in 0..rows {
            let sin_lat =
                -Gf::ONE + Gf::TWO * Gf::from_i64(r as i64) / Gf::from_i64(rows as i64 - 1);
            let w = if r == 0 || r == rows - 1 {
                Gf::HALF
            } else {
                Gf::ONE
            };
            sum += earth.harmonic(sin_lat, 0).0 * w;
        }
        close(sum / Gf::from_i64(rows as i64 - 1), EARTH_S / 4.0, 0.3);
        close(earth.harmonic(Gf::ZERO, 0).0, 415.54, 0.05);
        close(earth.harmonic(Gf::ONE, 0).0, 172.32, 0.05);
        // The equator has no first harmonic (its two solstices match); the pole's swings by about half its year.
        close(earth.harmonic(Gf::ZERO, 1).1, 0.0, 1e-9);
        let (_, b1) = earth.harmonic(Gf::ONE, 1);
        assert!(b1.to_f64() > 200.0, "{}", b1.to_f64());
        let (_, b1s) = earth.harmonic(-Gf::ONE, 1);
        close(b1s, -b1.to_f64(), 1e-6);
        // A season in the north is the season half a year on in the south.
        let at = Gf::from_f64(0.6);
        close(
            earth.season_mean(at, 3, 4),
            earth.season_mean(-at, 11, 4).to_f64(),
            1e-6,
        );
        // The resolution: 512 samples against 256.
        let fine = SunTable::new(
            Gf::from_f64(EARTH_S),
            Gf::ZERO,
            Gf::from_f64(0.917_482_1),
            rows,
            YEAR_SAMPLES_SHIFT + 1,
        );
        for m in 0..16 {
            let a = earth.season_mean(at, m, 4).to_f64();
            let b = fine.season_mean(at, m, 4).to_f64();
            let scale = if b.abs() > 1.0 { b.abs() } else { 1.0 };
            assert!((a - b).abs() < 1e-3 * scale, "season {m}: {a} against {b}");
        }
        // The eccentricity's time mean: the year's mean grows by 1/√(1 − e²).
        let eccentric = SunTable::new(
            Gf::from_f64(EARTH_S),
            Gf::from_f64(0.6),
            Gf::from_f64(0.917_482_1),
            rows,
            YEAR_SAMPLES_SHIFT,
        );
        close(
            eccentric.harmonic(Gf::ZERO, 0).0,
            earth.harmonic(Gf::ZERO, 0).0.to_f64() / 0.8,
            1e-9,
        );
    }
}
