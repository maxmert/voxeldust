//! Per-planet rotation parameters and the closed-form rotation function.
//!
//! ## Two contracts
//!
//! 1. [`PlanetRotationParams::from_seed_and_state`] is a pure derivation
//!    called by the server (`system-shard`) once at system bootstrap. The
//!    result is broadcast to clients in
//!    `CelestialBodySnapshot.rotation_params`.
//! 2. [`rotation_at`] is a pure function called by both the server (when
//!    rendering authoritative rotation) and the client (each frame, with
//!    the extrapolated `game_time_s`). Bit-for-bit identical inputs
//!    produce bit-for-bit identical quaternions, which is the basis of
//!    the temporal-correctness guarantee.
//!
//! ## Why the params are static
//!
//! A rotation period (~hours), axial tilt (~degrees), and epoch offset
//! (~seconds) are *static* properties of a planet for the duration of a
//! play session — they don't change with time. So we broadcast the
//! params **once per tick** (echoed for safety; ~6 floats/planet/tick)
//! rather than the time-varying quaternion. Both ends evaluate
//! `rotation_at(params, game_time)` from the same closed-form function:
//!
//! - Server and client compute the **same quaternion** at any wall-clock
//!   moment, with no interpolation between server samples.
//! - Two players standing at the same coordinates on the same planet at
//!   the same `game_time_now` see the **same sun direction** bit-for-bit.
//! - Replay determinism: recording broadcast bytes + `game_time` is
//!   sufficient to reconstruct any moment exactly.
//!
//! ## Physics-derived defaults
//!
//! `from_seed_and_state` derives the rotation period from the planet's
//! own bulk physics (mass, radius) plus seed-driven formation-history
//! variance. The natural physical timescale is the gravitational
//! free-fall time:
//!
//!   `t_freefall = √(R³ / (G·M))`
//!
//! which falls out of dimensional analysis on `G`, `M`, `R`. Its value
//! for Earth (≈ 806 s) is multiplied by a calibrated dimensionless
//! constant ([`PLANET_ROTATION_FREEFALL_MULTIPLIER`] = 107) to land
//! Earth at one Earth-day; planets with different `M`, `R` derive their
//! own rotation period from the same formula. Per-planet seed variance
//! covers formation-history randomness (impacts, tidal evolution) that
//! physics alone can't predict.
//!
//! ## Determinism
//!
//! Every numeric value comes from a named constant in
//! [`crate::physics_constants`]; the
//! `client/tests/no_magic_numbers.rs` integration test enforces this.

use glam::{DQuat, DVec3};
use serde::{Deserialize, Serialize};

use crate::physics_constants::{
    G_NEWTONIAN, PLANET_OBLIQUITY_MAX_RAD, PLANET_OBLIQUITY_MIN_RAD,
    PLANET_RETROGRADE_PROBABILITY, PLANET_ROTATION_FREEFALL_MULTIPLIER,
    PLANET_ROTATION_VARIANCE_AMPLITUDE, TAU,
};
use crate::seed::{derive_seed, seed_to_f64, seed_to_range};

/// Sub-seed indices reserved for planet-rotation derivations. Stable
/// across versions so seed → output remains reproducible.
mod sub_seed {
    pub const ROTATION_PERIOD_VARIANCE: u32 = 3_001;
    pub const OBLIQUITY: u32 = 3_002;
    pub const EPOCH_OFFSET: u32 = 3_003;
    pub const RETROGRADE_FLIP: u32 = 3_004;
}

/// Static rotational state of a single planet. Derived once from
/// `(planet_seed, mass_kg, radius_m)` and broadcast verbatim — the
/// time-varying quaternion is computed by `rotation_at(params, t)` on
/// both ends.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct PlanetRotationParams {
    /// Sidereal rotation period in seconds. Derived from the planet's
    /// gravitational free-fall timescale plus a log-uniform seed-driven
    /// formation-history multiplier. Always positive — retrograde
    /// motion is encoded by flipping the sign of `axis_system_space`.
    pub period_s: f64,

    /// Rotation axis in the system's inertial frame (the same frame the
    /// star catalog and orbital elements live in; +Z conventionally
    /// orbital-plane normal). Unit vector. The planet rotates about this
    /// axis with angular velocity `2π / period_s` in the right-handed
    /// sense: `quat = axis_angle(axis, +ω·t)` advances the rotation
    /// forward in time. Retrograde planets carry a flipped axis.
    pub axis_system_space: DVec3,

    /// Phase offset at `game_time = 0`. The full rotation phase at any
    /// time is `2π · (game_time + epoch_offset) / period`. Picked from
    /// the planet seed so neighbour planets aren't synchronised at boot.
    pub epoch_offset_s: f64,
}

impl PlanetRotationParams {
    /// Derive a planet's rotation parameters from its seed + bulk physics.
    ///
    /// Steps:
    /// 1. Compute the gravitational free-fall timescale
    ///    `t_ff = √(R³ / (G·M))`.
    /// 2. Multiply by [`PLANET_ROTATION_FREEFALL_MULTIPLIER`] to obtain
    ///    the physics-baseline rotation period. Heavier / larger planets
    ///    get smaller `t_ff` and so faster rotation.
    /// 3. Apply a log-uniform seed-driven multiplier in
    ///    `[1/AMP, AMP]` (see [`PLANET_ROTATION_VARIANCE_AMPLITUDE`])
    ///    for formation-history variance.
    /// 4. Pick obliquity from the seed within
    ///    `[PLANET_OBLIQUITY_MIN_RAD, PLANET_OBLIQUITY_MAX_RAD]`.
    /// 5. Roll for retrograde at probability
    ///    [`PLANET_RETROGRADE_PROBABILITY`]; flip the axis if so.
    /// 6. Pick the epoch offset uniformly in `[0, period_s)`.
    pub fn from_seed_and_state(planet_seed: u64, mass_kg: f64, radius_m: f64) -> Self {
        // Step 1: physics baseline period.
        let t_freefall = (radius_m.powi(3) / (G_NEWTONIAN * mass_kg)).sqrt();
        let baseline_period = t_freefall * PLANET_ROTATION_FREEFALL_MULTIPLIER;

        // Step 2: log-uniform seed-driven multiplier.
        let variance_t =
            seed_to_f64(derive_seed(planet_seed, sub_seed::ROTATION_PERIOD_VARIANCE));
        let log_amp = PLANET_ROTATION_VARIANCE_AMPLITUDE.ln();
        let variance_factor = (log_amp * (2.0 * variance_t - 1.0)).exp();
        let period_s = baseline_period * variance_factor;

        // Step 3: obliquity (axial tilt about the orbital plane).
        let obliquity_rad = seed_to_range(
            derive_seed(planet_seed, sub_seed::OBLIQUITY),
            PLANET_OBLIQUITY_MIN_RAD,
            PLANET_OBLIQUITY_MAX_RAD,
        );

        // Step 4: retrograde detection.
        let retrograde =
            seed_to_f64(derive_seed(planet_seed, sub_seed::RETROGRADE_FLIP))
                < PLANET_RETROGRADE_PROBABILITY;
        let direction = if retrograde { -1.0 } else { 1.0 };

        // Step 5: build the rotation axis. The orbital plane normal is +Z;
        // tilt by `obliquity` about the +X axis so the spin axis sweeps
        // a great circle in the YZ plane. The retrograde flip negates Z
        // so a Venus-like world spins the opposite sense.
        let axis = DVec3::new(0.0, obliquity_rad.sin(), direction * obliquity_rad.cos())
            .normalize();

        // Step 6: epoch offset uniform in [0, period_s).
        let epoch_offset_s = seed_to_range(
            derive_seed(planet_seed, sub_seed::EPOCH_OFFSET),
            0.0,
            period_s,
        );

        Self {
            period_s,
            axis_system_space: axis,
            epoch_offset_s,
        }
    }
}

/// Closed-form rotation quaternion at the given absolute server-time
/// (in seconds). Pure function: same `(params, t)` always returns the
/// exact same quaternion on every machine.
///
/// Rotation phase: `θ = 2π · (t + epoch_offset) / period`.
/// Quaternion:    `q = axis_angle(axis, θ)`.
pub fn rotation_at(params: &PlanetRotationParams, game_time_s: f64) -> DQuat {
    let phase = TAU * (game_time_s + params.epoch_offset_s) / params.period_s;
    DQuat::from_axis_angle(params.axis_system_space, phase)
}

// ─── FlatBuffers conversion helpers ────────────────────────────────────────

use crate::protocol_generated as fb;
use flatbuffers::{FlatBufferBuilder, WIPOffset};

/// Build a `PlanetRotationParamsData` table on the FlatBuffers builder.
/// `None` propagates through.
pub fn to_fb_rotation_params<'b>(
    s: &Option<PlanetRotationParams>,
    builder: &mut FlatBufferBuilder<'b>,
) -> Option<WIPOffset<fb::PlanetRotationParamsData<'b>>> {
    s.as_ref().map(|s| {
        fb::PlanetRotationParamsData::create(
            builder,
            &fb::PlanetRotationParamsDataArgs {
                period_s: s.period_s,
                axis_x: s.axis_system_space.x,
                axis_y: s.axis_system_space.y,
                axis_z: s.axis_system_space.z,
                epoch_offset_s: s.epoch_offset_s,
            },
        )
    })
}

/// Decode an optional `PlanetRotationParamsData` FB table into a typed
/// [`PlanetRotationParams`]. `None` propagates through.
pub fn from_fb_rotation_params(
    fb: Option<fb::PlanetRotationParamsData>,
) -> Option<PlanetRotationParams> {
    fb.map(|s| PlanetRotationParams {
        period_s: s.period_s(),
        axis_system_space: DVec3::new(s.axis_x(), s.axis_y(), s.axis_z()),
        epoch_offset_s: s.epoch_offset_s(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::physics_constants::{M_EARTH_KG, R_EARTH_M, SECONDS_PER_EARTH_DAY};

    /// Earth-bulk planet should land near one Earth-day rotation period
    /// (within the seed-variance envelope).
    #[test]
    fn earth_like_period() {
        for seed in [0x1111_u64, 0x2222, 0xC0FFEE, 0xDEAD_BEEF] {
            let p = PlanetRotationParams::from_seed_and_state(seed, M_EARTH_KG, R_EARTH_M);
            // Period is in [baseline / AMP, baseline · AMP] for the
            // configured envelope. Baseline ≈ Earth day, AMP = 3, so
            // 8h–72h is the legal range.
            let lower = SECONDS_PER_EARTH_DAY / PLANET_ROTATION_VARIANCE_AMPLITUDE;
            let upper = SECONDS_PER_EARTH_DAY * PLANET_ROTATION_VARIANCE_AMPLITUDE;
            assert!(
                (lower..=upper).contains(&p.period_s),
                "Earth-bulk seed {seed:x} period {p:?} outside [{lower}, {upper}]"
            );
        }
    }

    /// Determinism: same input → bit-identical output.
    #[test]
    fn deterministic() {
        let a = PlanetRotationParams::from_seed_and_state(0xC0FFEE, M_EARTH_KG, R_EARTH_M);
        let b = PlanetRotationParams::from_seed_and_state(0xC0FFEE, M_EARTH_KG, R_EARTH_M);
        assert_eq!(a, b);
    }

    /// Different seeds → different rotation params.
    #[test]
    fn different_seeds_differ() {
        let a = PlanetRotationParams::from_seed_and_state(0x1111, M_EARTH_KG, R_EARTH_M);
        let b = PlanetRotationParams::from_seed_and_state(0x2222, M_EARTH_KG, R_EARTH_M);
        assert_ne!(a.period_s, b.period_s);
    }

    /// Smaller planet (lower G·M, similar R) should rotate slower (larger
    /// `t_freefall`) all else equal — physics dominates the seed variance
    /// when held same.
    #[test]
    fn lighter_planet_rotates_slower() {
        // Same seed for both — variance factor is identical, so the only
        // changing input is the physical baseline.
        let earth = PlanetRotationParams::from_seed_and_state(
            0xABCD,
            M_EARTH_KG,
            R_EARTH_M,
        );
        // 10× lighter, same radius → larger `t_ff` → slower rotation.
        let light = PlanetRotationParams::from_seed_and_state(
            0xABCD,
            M_EARTH_KG / 10.0,
            R_EARTH_M,
        );
        assert!(
            light.period_s > earth.period_s,
            "lighter planet should rotate slower: light={}, earth={}",
            light.period_s,
            earth.period_s,
        );
    }

    /// `rotation_at` must be a pure function: same `(params, t)` always
    /// returns the same quaternion bit-for-bit.
    #[test]
    fn rotation_at_pure() {
        let p = PlanetRotationParams {
            period_s: 86_400.0,
            axis_system_space: DVec3::new(0.0, 0.4, 0.92).normalize(),
            epoch_offset_s: 12_345.6,
        };
        let q1 = rotation_at(&p, 1_234_567.89);
        let q2 = rotation_at(&p, 1_234_567.89);
        assert_eq!(q1, q2);
    }

    /// Periodicity: rotation at `t` and `t + period` should be identical.
    /// (Verifying the closed-form formula doesn't accidentally drift
    /// under integer-period offsets.)
    #[test]
    fn rotation_periodic() {
        let p = PlanetRotationParams {
            period_s: 86_400.0,
            axis_system_space: DVec3::new(0.0, 0.4, 0.92).normalize(),
            epoch_offset_s: 0.0,
        };
        let q_at_t = rotation_at(&p, 0.0);
        let q_at_t_plus_period = rotation_at(&p, p.period_s);
        // Quaternions q and -q represent the same rotation; either
        // should be considered equivalent. Use dot product.
        let dot = q_at_t.dot(q_at_t_plus_period).abs();
        assert!(
            (dot - 1.0).abs() < 1e-12,
            "rotation should be periodic: q(t)·q(t+T) = {dot}"
        );
    }

    /// Retrograde planets must produce a flipped axis (negative direction
    /// component on the spin axis). Use a seed known to roll retrograde
    /// from the 5%-probability check.
    #[test]
    fn retrograde_seed_flips_axis() {
        // Sweep many seeds and verify both prograde and retrograde
        // appear, with retrograde frequency near the named probability.
        let mut retrograde_count = 0;
        let total = 1_000;
        for i in 0..total {
            let p = PlanetRotationParams::from_seed_and_state(
                i as u64 ^ 0xABCD_DEAD,
                M_EARTH_KG,
                R_EARTH_M,
            );
            // Retrograde axis has negative Z component (flipped from
            // the +Z prograde convention).
            if p.axis_system_space.z < 0.0 {
                retrograde_count += 1;
            }
        }
        let observed_rate = retrograde_count as f64 / total as f64;
        // Allow ±2× tolerance — sample noise on a 5%-probability roll
        // over 1000 samples gives ≈ ±0.014 absolute error (~3 stddev).
        let expected = PLANET_RETROGRADE_PROBABILITY;
        assert!(
            (observed_rate / expected - 1.0).abs() < 1.0,
            "retrograde rate {observed_rate} far from expected {expected}"
        );
    }

    /// FlatBuffers round-trip preserves bit-identical params.
    #[test]
    fn flatbuffers_round_trip() {
        let original =
            PlanetRotationParams::from_seed_and_state(0xBEEF, M_EARTH_KG, R_EARTH_M);
        let mut builder = FlatBufferBuilder::new();
        let offset = to_fb_rotation_params(&Some(original), &mut builder).unwrap();
        builder.finish(offset, None);
        let buf = builder.finished_data();
        let decoded =
            unsafe { flatbuffers::root_unchecked::<fb::PlanetRotationParamsData>(buf) };
        let recovered = from_fb_rotation_params(Some(decoded)).unwrap();
        assert_eq!(original, recovered);
    }
}
