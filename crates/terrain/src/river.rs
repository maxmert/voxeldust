//! ★ THE WIDTH LAW — HOW WIDE AND HOW DEEP A CHANNEL IS AT A GIVEN DISCHARGE (slice 8d).
//!
//! ★ **WHAT THIS MODULE NO LONGER DOES** (2026-09-23, ruling W16; the survey's R4). It once also
//! DREW the drainage: a Horton–Strahler tributary synthesis, stamped channel lines, a valley
//! profile that damped the fine octaves and a per-column water word. That stamp is RETIRED. It was
//! measured as the near flicker the owner saw from a low pass (ruling W15 §3): the stamp wrote a
//! stream's surface only where a rung drew the valley at FULL strength, so one wet column in
//! sixteen appeared or vanished at a single ring swap and the water's edge jumped up to 5.4 km.
//! A rung boundary must be invisible, so the drawn lines are gone and the fine relief is the
//! recipe's own octaves under the roughness factor again.
//!
//! **WHAT STAYS, AND WHY.** The hydraulic geometry of Leopold & Maddock 1953 (USGS Professional
//! Paper 252), calibrated on EARTH: `w = 3.9·√Q` metres and `d = 0.4·Q^0.4` metres at Earth's
//! gravity, the downstream exponents `b = 0.5` and `f = 0.4` they measured. The SOLVE still needs a
//! width where it cuts a channel, and the deposition step still needs Leopold & Wolman 1960's
//! meander belt ([`BELT_OVER_WIDTH_NUM`] over [`BELT_OVER_WIDTH_DEN`] = 2.7 channel widths) for the
//! floodplain it lays.
//!
//! **GRAVITY IS NOT A CALIBRATION** (ruling T9): a bed's shear is `ρ·g·d·S`, so at a fixed critical
//! shear the depth falls as `1/g` and continuity `Q = w·d·U` puts the width up as `g`. A
//! low-gravity moon gets deep, narrow rivers with no new constant. The float fence forbids a power,
//! so both laws are 256-entry COMMITTED TABLES indexed by the row's own discharge class
//! ([`CHANNEL_WIDTH_MM`], [`CHANNEL_DEPTH_MM`]).
//!
//! **Example.** A trunk on the belt carries 9 500 m³/s. The law states a channel 391 m wide and
//! 16 m deep, and the solve cuts its bed to that width. Nothing draws a blue line: the river is
//! where the ground is lowest.

// ★ THE DIVISIONS HERE ARE CPU-ONLY AND INTEGER-EXACT (the crate's own `integer_division` deny, and
// the exception it states). The card holds no artifact and never runs one line of this module.
#![allow(
    clippy::integer_division,
    reason = "the width law is CPU-only: the card reads no artifact and runs no line of it"
)]

/// Earth's surface gravity in mm/s² — the body the width law is calibrated on, and the only place
/// the gravity factor is measured against.
pub const EARTH_GRAVITY_MM_S2: i64 = 9_807;

/// ★ THE MEANDER BELT over the channel's width (Leopold & Wolman 1960), numerator.
pub const BELT_OVER_WIDTH_NUM: i64 = 27;
/// The meander belt's denominator.
pub const BELT_OVER_WIDTH_DEN: i64 = 10;

mod tables;
pub use tables::{CHANNEL_DEPTH_MM, CHANNEL_WIDTH_MM};

/// ★ THE WIDTH LAW at one discharge class, in whole millimetres, on a body of this gravity
/// (Leopold & Maddock 1953; the gravity factor from the bed's own shear).
#[must_use]
pub fn channel_width_mm(class: u8, gravity_mm_s2: u32) -> i64 {
    let base = i64::from(CHANNEL_WIDTH_MM[class as usize]);
    (base * i64::from(gravity_mm_s2)) / EARTH_GRAVITY_MM_S2
}

/// ★ THE DEPTH LAW at one discharge class, in whole millimetres, on a body of this gravity: the
/// depth falls as gravity rises, because the bed's shear `ρ·g·d·S` must reach one critical value.
#[must_use]
pub fn channel_depth_mm(class: u8, gravity_mm_s2: u32) -> i64 {
    let base = i64::from(CHANNEL_DEPTH_MM[class as usize]);
    if gravity_mm_s2 == 0 {
        return 0;
    }
    (base * EARTH_GRAVITY_MM_S2) / i64::from(gravity_mm_s2)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::artifact::log2_class;

    /// The Julian year in seconds — the year the climate's own rain law is calibrated in.
    const YEAR_S: f64 = 31_557_600.0;

    /// The discharge class of a discharge in m³/s, by the very rule the row is written with.
    fn class_of(q_m3s: f64) -> u8 {
        log2_class((q_m3s * 1_000.0 * YEAR_S) as u64, 2)
    }

    /// wrong unit or a wrong exponent all fail.
    #[test]
    fn the_width_law_reads_earths_own_rivers() {
        let g = EARTH_GRAVITY_MM_S2 as u32;
        // The design's own committed table (`03_erosion_rivers.md` §6.1), which the class step
        // reproduces to within 3 %: 3.8, 12.0, 38, 120, 380 and 931 metres of width.
        let table = [
            (1.0_f64, 3.94_f64, 0.40_f64),
            (9.5, 12.21, 1.00),
            (95.0, 38.17, 2.48),
            (951.0, 117.38, 6.09),
            (9_506.0, 390.67, 15.95),
            (57_034.0, 939.05, 32.16),
        ];
        for (q, want_w, want_d) in table {
            let c = class_of(q);
            let w = channel_width_mm(c, g) as f64 / 1_000.0;
            let d = channel_depth_mm(c, g) as f64 / 1_000.0;
            assert!(
                (w - want_w).abs() <= want_w * 0.01,
                "Q = {q} m3/s, class {c}: the width reads {w} m against the law's {want_w} m"
            );
            assert!(
                (d - want_d).abs() <= want_d * 0.01,
                "Q = {q} m3/s, class {c}: the depth reads {d} m against the law's {want_d} m"
            );
        }
    }

    /// ★ GATE: EARTH'S TWO ANCHORS, AND WHAT THE LAW ACTUALLY SAYS ABOUT THEM.
    ///
    /// A stream of 1 m³/s: Earth's brooks of that flow are three to five metres wide, and the law
    /// reads 3.9 m. The Mississippi at its mean annual 17 000 m³/s: the law reads 19.6 m deep,
    /// inside the 10–20 m the river's own gauges give, and 504 m wide against an observed width
    /// near a kilometre.
    ///
    /// ★ THE WIDTH IS THE LAW'S OWN KNOWN NARROW BIAS AT THE LARGEST RIVERS, AND IT IS NOT TUNED
    /// AWAY. A coefficient that put the Mississippi at a kilometre would put the 1 m³/s brook at
    /// 7.7 m, and matching BOTH anchors needs a downstream exponent near 0.6, which no published
    /// fit of Leopold & Maddock's gives. Ruling B3 says a gate is never tuned to pass; the same
    /// holds for the law a gate measures. The band below is an ORDER-OF-MAGNITUDE band, so a
    /// wrong unit or a wrong exponent still goes red, and the exact readings are recorded here.
    #[test]
    fn earths_two_anchors_stand_where_the_law_puts_them() {
        let g = EARTH_GRAVITY_MM_S2 as u32;
        let brook = channel_width_mm(class_of(1.0), g) as f64 / 1_000.0;
        assert!(
            (1.0..=10.0).contains(&brook),
            "a 1 m3/s stream reads {brook} m wide; Earth's are three to five"
        );
        let miss_w = channel_width_mm(class_of(17_000.0), g) as f64 / 1_000.0;
        let miss_d = channel_depth_mm(class_of(17_000.0), g) as f64 / 1_000.0;
        assert!(
            (100.0..=2_000.0).contains(&miss_w),
            "the Mississippi reads {miss_w} m wide; it is about a kilometre"
        );
        assert!(
            (10.0..=20.0).contains(&miss_d),
            "the Mississippi reads {miss_d} m deep; its gauges give 10 to 20"
        );
    }

    /// ★ GRAVITY IS NOT A CALIBRATION: a low-gravity moon gets a DEEP, NARROW river at the same
    /// discharge, because the bed's shear `ρ·g·d·S` must still reach one critical value.
    #[test]
    fn a_low_gravity_moon_gets_a_deep_narrow_river() {
        let c = class_of(95.0);
        let earth_w = channel_width_mm(c, EARTH_GRAVITY_MM_S2 as u32);
        let earth_d = channel_depth_mm(c, EARTH_GRAVITY_MM_S2 as u32);
        let moon = (EARTH_GRAVITY_MM_S2 / 6) as u32;
        let moon_w = channel_width_mm(c, moon);
        let moon_d = channel_depth_mm(c, moon);
        assert!(
            moon_w * 5 < earth_w,
            "the moon's river reads {moon_w} mm wide against Earth's {earth_w} mm"
        );
        assert!(
            moon_d > earth_d * 5,
            "the moon's river reads {moon_d} mm deep against Earth's {earth_d} mm"
        );
        assert_eq!(channel_depth_mm(c, 0), 0);
    }
}
