//! ★ THE FIRST SURFACE LOOK BY LAW (ruling B75 item 4, 2026-10-10): what covers the ground at a point — bare rock, snow,
//! or the soil between them — from the slope, the height and the latitude alone.
//!
//! The two laws are the ground step design's own (`docs/investigation/2026-10-02/ground_step_design.md`, K2 and K12):
//! - **THE SOIL HOLDS UNDER THE CRITICAL SLOPE** (K2: Roering, Kirchner & Dietrich 1999, `q = K·S / (1 − (S/S_c)²)`,
//!   `S_c` = 1.25 on the Oregon Coast Range, the limit of soil-mantled slopes). The soil's share is the law's own
//!   denominator `1 − (S/S_c)²`: the share of the hillslope's transport that slow soil creep can carry. All soil on the
//!   flat, all bare rock where the slope reaches `S_c` (51°). [`rock_share`] reads it in integers from one table.
//! - **THE SNOW LIES WHERE THE SUMMER FREEZES** (Egholm et al. 2009: the ablation season's `Ts = 0` isotherm): the
//!   height at which the climate law's SUMMER air reaches freezing ([`crate::climate::summer_k`]), by latitude, over
//!   the body's datum ([`SnowLine`]). Only a body with air AND water has snow ([`ClimateLaws::rains`]). The line is the
//!   almanac's PAINT (K12, F17: "the seasonal snow line as the almanac's paint, with no collision"): the drawing reads
//!   it, the static shape never does, and the almanac may later move it with the season.
//!
//! ★ INTERIMS (registered in `docs/design/DEFERRED.md`, "THE FIRST SURFACE LOOK"): the slope is the DRAWN slope, so a
//! coarser rung (a smoother surface) shows less rock than the average of the finer rung under it — K12's expected shares
//! under the dropped variance are owed; the soil law's second half (bare rock where the local lowering rate passes soil
//! production, Heimsath et al. 1997) waits for R3's lowering rate; the snow's own holding angle (Bernhardt & Schulz 2010)
//! is stood in for by the soil's; the line is the summer's, never the season's of the day; the glaciers' ice is not
//! drawn yet.
//!
//! **Example.** On the home planet's belt a pilot walks up a valley floor at 1 200 m: the slope is 10°, so the ground is
//! almost all soil. The valley's side steepens to 45°: about two thirds of it is bare rock. Over the summer's freezing
//! line the gentle shoulders are white, and the 60° face above them stays dark rock.

use std::sync::LazyLock;

use crate::climate::{ClimateLaws, FREEZING_K, summer_k};
use crate::gf::Gf;

/// ★ THE CRITICAL SLOPE `S_c` (K2: Roering, Kirchner & Dietrich 1999, the Oregon Coast Range): the gradient at which a
/// soil-mantled hillslope's transport runs away, so no soil stays on it.
pub const SOIL_CRITICAL_SLOPE: f64 = 1.25;

/// The slope's cosine is read in these steps (`cos θ · COS_STEPS`, rounded): the table holds one share a step.
pub const COS_STEPS: u32 = 4_096;

/// A whole share.
pub const FULL: u8 = 255;

/// The axis of a body's spin in the body's frame (the recipe's own).
pub const POLE_AXIS: usize = vd_recipe::height::POLE_AXIS;

/// The soil's share at each cosine step, `0..=COS_STEPS`, built once.
static SOIL: LazyLock<Vec<u8>> = LazyLock::new(soil_table);

/// ★ THE BARE ROCK'S SHARE at a slope whose cosine is `cos_q` in [`COS_STEPS`] (the ground's up against its own
/// normal), in 1/255: `(S/S_c)²` with `S = tan θ`, held to `[0, 1]`. A cosine past one reads as flat.
#[must_use]
pub fn rock_share(cos_q: u32) -> u8 {
    FULL - SOIL[cos_q.min(COS_STEPS) as usize]
}

/// The soil's share by the slope's cosine: `1 − (S/S_c)²` with `S = tan θ`, in 1/255, held to `[0, 1]`. A vertical
/// face (cosine zero) holds none.
fn soil_table() -> Vec<u8> {
    let steps = Gf::from_i64(i64::from(COS_STEPS));
    let sc2 = Gf::from_f64(SOIL_CRITICAL_SLOPE) * Gf::from_f64(SOIL_CRITICAL_SLOPE);
    let mut table = Vec::with_capacity(COS_STEPS as usize + 1);
    table.push(0);
    for step in 1..=COS_STEPS {
        let c = Gf::from_i64(i64::from(step)) / steps;
        let c2 = c * c;
        let tan2 = (Gf::ONE - c2) / c2;
        let soil = (Gf::ONE - tan2 / sc2).clamp(Gf::ZERO, Gf::ONE);
        table.push(share_of(soil));
    }
    table
}

/// A share in `[0, 1]` as 1/255, rounded half up.
fn share_of(share: Gf) -> u8 {
    (share * Gf::from_i64(i64::from(FULL)) + Gf::HALF).to_i64_floor() as u8
}

/// ★ THE SUMMER'S FREEZING LINE of a body: its height over the ladder radius, in metres, at a latitude whose sine is
/// `s` is `base_m + per_sin2_m · s² + per_sin_m · |s|`. That is the climate law's own form — the mean, the latitude
/// contrast `T₂·P₂(sin φ)` and the season's warmth `∝ |sin φ|`, climbed at the lapse rate — so three heights of the
/// law name it exactly ([`SnowLine::of`]).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SnowLine {
    pub base_m: Gf,
    pub per_sin2_m: Gf,
    pub per_sin_m: Gf,
}

impl SnowLine {
    /// The line of a body whose climate law is `laws` and whose datum stands `datum_m` whole metres over the ladder
    /// radius; `None` where the body has no snow (no air or no water) or no lapse to climb.
    #[must_use]
    pub fn of(laws: &ClimateLaws, datum_m: i32) -> Option<SnowLine> {
        if !laws.rains || laws.lapse_k_m <= Gf::ZERO {
            return None;
        }
        let at = |s: f64| line_at(laws, Gf::from_f64(s), datum_m);
        let (equator, half, pole) = (at(0.0), at(0.5), at(1.0));
        // From the form a + b·s² + c·|s|: the pole gives a + b + c, the half a + b/4 + c/2.
        let per_sin_m = Gf::from_f64(4.0) * (half - equator) - (pole - equator);
        Some(SnowLine {
            base_m: equator,
            per_sin2_m: pole - equator - per_sin_m,
            per_sin_m,
        })
    }

    /// The line's height at a latitude whose sine is `sin_lat`, metres over the ladder radius.
    #[must_use]
    pub fn at(&self, sin_lat: Gf) -> Gf {
        self.base_m + self.per_sin2_m * sin_lat * sin_lat + self.per_sin_m * sin_lat.abs()
    }
}

/// The law's own line at one latitude: the datum plus the summer's warmth over freezing at the datum, climbed at the
/// lapse rate (positive by [`SnowLine::of`]), metres over the ladder radius.
fn line_at(laws: &ClimateLaws, sin_lat: Gf, datum_m: i32) -> Gf {
    let over = summer_k(laws, sin_lat, Gf::ZERO) - Gf::from_f64(FREEZING_K);
    Gf::from_i64(i64::from(datum_m)) + over / laws.lapse_k_m
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::climate::Circulation;

    /// A body's climate law with Earth-like words: 288 K, a −40 K latitude contrast, 6.5 K/km, a season.
    fn earthlike(rains: bool, lapse: f64) -> ClimateLaws {
        ClimateLaws {
            t_mean_k: Gf::from_f64(288.0),
            t2_k: Gf::from_f64(-40.0),
            lapse_k_m: Gf::from_f64(lapse),
            circulation: Circulation { cells: 3 },
            rains,
            absorbed_w_m2: Gf::from_f64(240.0),
            s1: Gf::from_f64(0.8),
            seasonal_damping_w_m2_k: Gf::from_f64(20.0),
        }
    }

    #[test]
    fn the_flat_is_soil_and_the_critical_slope_is_rock() {
        // Flat ground: no rock; a cosine past one reads as flat.
        assert_eq!(rock_share(COS_STEPS), 0);
        assert_eq!(rock_share(COS_STEPS + 7), 0);
        // A vertical face: all rock.
        assert_eq!(rock_share(0), FULL);
        // 45° (S = 1): the soil share is 1 − 1/1.5625 = 0.36, so the rock is 0.64 of the whole.
        let c45 = (0.5f64.sqrt() * f64::from(COS_STEPS)).round() as u32;
        assert_eq!(rock_share(c45), FULL - 92);
        // Past the critical slope (S = 1.25, cos ≈ 0.625) the rock is whole.
        assert_eq!(rock_share(2_500), FULL);
    }

    #[test]
    fn a_body_with_no_air_or_no_lapse_has_no_snow_line() {
        assert_eq!(SnowLine::of(&earthlike(false, 0.0065), 0), None);
        assert_eq!(SnowLine::of(&earthlike(true, 0.0), 0), None);
    }

    #[test]
    fn the_line_is_the_laws_own_at_every_latitude() {
        let laws = earthlike(true, 0.0065);
        let line = SnowLine::of(&laws, 3_000).expect("a snow line");
        // At the equator the summer is the year: (288 − 40·(−½) − 273.15) / 0.0065 ≈ 5 362 m over the datum.
        assert!((line.at(Gf::ZERO).to_f64() - (3_000.0 + 34.85 / 0.0065)).abs() < 1.0e-6);
        // The three heights name the law at every other latitude, both hemispheres alike.
        for s in [0.1, 0.3, 0.7, 0.9, -0.4] {
            let law = line_at(&laws, Gf::from_f64(s), 3_000).to_f64();
            assert!(
                (line.at(Gf::from_f64(s)).to_f64() - law).abs() < 1.0e-6,
                "at {s}"
            );
        }
    }
}
