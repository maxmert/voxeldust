//! THE SPEED LAW (real-scale design §4 + addendum §A3 — the owner's warp mechanism): the pure,
//! closed-form pieces of "the CONTAINING REALM controls occupant speed". Four parts, one policy
//! number, no realm-kind test, no wire, nothing crosses a boundary:
//!
//! 1. **The realm ceiling** ([`realm_speed_cap_mps`]): every realm states ONE number to whoever is
//!    inside it — its own bound width over [`TRAVERSE_S`], floored at the occupant's thruster speed.
//!    Every realm under the 45 km break-even (`v_foot·T/2`) clamps to the floor, which is what makes
//!    the law PROVABLY INERT at human scale (walking, landing, building, colliding are untouched).
//! 2. **The geometric throttle** ([`throttle_axes_scale`]): half stick is the geometric middle, not
//!    the arithmetic one — the owner clause that keeps slow deep-space flight commandable (a linear
//!    map's smallest non-zero galaxy speed would be superluminal).
//! 3. **The ramp** ([`ramp_cap_mps`]): speed rises at a constant PROPORTIONAL rate `dv/dt = v/τ`,
//!    so a small change and a huge change cost the same seconds. Deceleration needs no ramp state:
//!    following the governor's falling ceiling IS the decel ramp (`dx/dt = −(v_c + x/τ)`).
//! 4. **The approach governor** ([`approach_ceiling_mps`]): near any body the ceiling falls back
//!    toward the body's own ceiling — you always arrive slowly and can never fly through anything.
//!    ★OQ-2 RULED (owner 2026-08-19, verbatim): *"the realm's ceiling governs everything the realm
//!    contains, piloted or not"* — the governor binds on the crossing path for EVERY subject kind
//!    (debris, projectiles, ballistic transients included), no exceptions.
//!
//! The closed-form leg time ([`leg_time_s`]) is the same law integrated — the flight-table gate and
//! every derived park/deadline read it from HERE, so the gates and the integrator cannot drift.
//!
//! HR5 shape: every function is monomorphic and branch-covered below; the couple of guards are
//! two-arm `if`s with both arms pinned by equality tests.

/// THE ONE POLICY NUMBER the speed law adds (real-scale design §4.2(a), addendum NEW-1): the
/// edge-to-edge full-throttle traverse time of ANY realm, in seconds. It is the owner's acceptance
/// bar — "journeys in MINUTES" — written down once: three minutes to cross a realm, at every level,
/// around every star, at any world size. `vd-physics`' geometry solve reads THIS constant (its
/// τ-free outset derivation), so the shells and the speed law can never disagree about T.
pub const TRAVERSE_S: f64 = 180.0;

/// The demand pipeline's post-boot hop count, in ticks — the τ derivation's fixed tail (§4.2(c):
/// τ = T_WAKE, "NOT a new constant"): the RLM reconciler's own interval (1) + the Q2 relay hop
/// (one tick at the woken child to author, one at the parent to forward: 2) + the gateway's
/// compose tick (1) + the client's draw tick (1). Each term is a landed cadence the process gates
/// (`warp_pixels::wake_budget_ticks`) count identically — one statement here, asserted there.
pub const WAKE_PIPELINE_TICKS: u64 = 5;

/// THE flight tuning — the speed law's numbers, derived in ONE place ([`FlightTuning::derive`]),
/// never inline. Carried per shard: `v_foot` and `tick_dt` are the shard's own config, τ is the
/// shard's own wake budget (a realm property — a realm with a slower demand pipeline ramps more
/// gently, which is exactly the §4.2(c) coupling: the ramp-down distance `τ·v` always covers the
/// wake lead).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct FlightTuning {
    /// The occupant's thruster speed — `move_speed_mps`, unchanged, with its true meaning: the
    /// floor of the ladder and the speed of everything at human scale.
    pub v_foot_mps: f64,
    /// The shard's tick length, seconds.
    pub tick_dt_s: f64,
    /// The traverse policy — [`TRAVERSE_S`], carried so a consumer never reaches for a global.
    pub traverse_s: f64,
    /// τ = T_WAKE: the ramp time-constant AND the governor's approach constant, seconds.
    pub tau_s: f64,
}

impl FlightTuning {
    /// Derive the tuning from what a shard already holds — NO new configuration enters:
    /// `τ = (2·aoi_cadence + boot_ticks_p99 + WAKE_PIPELINE_TICKS) · tick_dt` — the same T_WAKE
    /// expression the process gates derive (`2× the AoI demand beat` + the measured boot p99 +
    /// the fixed pipeline tail). At the DEV cluster (50 Hz, reactive boot 0) τ = 55 · 0.02 = 1.10 s.
    #[must_use]
    pub fn derive(
        v_foot_mps: f64,
        tick_dt_s: f64,
        aoi_cadence_ticks: u64,
        boot_ticks_p99: u32,
    ) -> FlightTuning {
        let wake_ticks = 2 * aoi_cadence_ticks + u64::from(boot_ticks_p99) + WAKE_PIPELINE_TICKS;
        FlightTuning {
            v_foot_mps,
            tick_dt_s,
            traverse_s: TRAVERSE_S,
            tau_s: wake_ticks as f64 * tick_dt_s,
        }
    }
}

/// (1) THE REALM CEILING: `max(v_foot, 2·bound_extent/T)` — one expression, no realm-kind test.
/// Reads the realm's OWN bound extent, a fact its own shard already holds (SL1/SL2/SL4/HR3 hold
/// structurally). The `max` is the collision floor: below the 45 km break-even extent
/// (`v_foot·T/2`) the ceiling IS the foot speed, exactly — `max` returns its first argument
/// bit-for-bit, so an interim-scale realm's behaviour is byte-identical by construction.
#[must_use]
pub fn realm_speed_cap_mps(bound_extent_m: f64, v_foot_mps: f64, traverse_s: f64) -> f64 {
    v_foot_mps.max(2.0 * bound_extent_m / traverse_s)
}

/// (4) ONE child's arm of the APPROACH GOVERNOR: the ceiling you may hold at `dist_to_bound_m`
/// from a body whose own ceiling is `child_cap_mps` — `child_cap + max(0, dist)/τ`. Following this
/// falling ceiling decelerates you exponentially onto the body's own speed: you always arrive
/// slowly, and a step can never out-run the ceiling into a fly-through. Kind-blind: a moon, a
/// station, a rock and a ship govern identically (SL4's own test — it reads an extent and a
/// distance, and cannot ask how anything moves).
#[must_use]
pub fn approach_ceiling_mps(child_cap_mps: f64, dist_to_bound_m: f64, tau_s: f64) -> f64 {
    child_cap_mps + dist_to_bound_m.max(0.0) / tau_s
}

/// (3) THE RAMP: the fastest speed reachable THIS tick from `v_prev` — a constant proportional
/// rate `dv/dt = +v/τ` compounded per tick (`·e^(dt/τ)`), floored at the foot speed so the ramp
/// binds only ABOVE human scale (from a standstill you always command your full thruster speed at
/// once, exactly as today — the ramp can never trap a stopped ship at zero). Deceleration is
/// deliberately un-ramped: stopping is instant (P3 "stopped means stopped"), and the gradual
/// arrival slow-down is the governor's falling ceiling, not a state here.
#[must_use]
pub fn ramp_cap_mps(v_prev_mps: f64, v_foot_mps: f64, tick_dt_s: f64, tau_s: f64) -> f64 {
    v_prev_mps.max(v_foot_mps) * (tick_dt_s / tau_s).exp()
}

/// (2) THE GEOMETRIC THROTTLE, as the ONE scale factor the integrator multiplies its existing
/// `axes · (move_speed·dt·time_multiplier)` step by — shaped so the inert case is BIT-IDENTICAL,
/// not merely equal:
///
/// `scale = (min(|axes|,1) / |axes|) · (v_cap/v_foot)^min(|axes|,1)`,  `0` for a zero stick.
///
/// - commanded speed = `throttle · v_foot · (v_cap/v_foot)^throttle` — §4.2(b) verbatim, the
///   geometric map (zero is stop: the MULTIPLY carries that, not the exponent);
/// - at a clamped ceiling (`v_cap == v_foot`, every sub-45 km realm) `ratio == 1.0` exactly
///   (IEEE `x/x`), `1^throttle == 1.0` exactly, and for a unit-or-less stick
///   `min(m,1)/m == m/m == 1.0` exactly — so `scale == 1.0` and the step is the OLD arithmetic
///   bit-for-bit (the S3 inertness measurement stands on this, and a pin below asserts it);
/// - an over-unit stick (a diagonal, `|axes| > 1`) is normalized to the ceiling: today's
///   `√3·v_foot` diagonal exceeded the realm's stated ceiling, which the law no longer permits —
///   the one measured behaviour change at human scale, and it is a cure, not a regression.
#[must_use]
pub fn throttle_axes_scale(axes_mag: f64, v_foot_mps: f64, v_cap_mps: f64) -> f64 {
    if axes_mag <= 0.0 {
        return 0.0;
    }
    let throttle = axes_mag.min(1.0);
    (throttle / axes_mag) * (v_cap_mps / v_foot_mps).powf(throttle)
}

/// THE CLOSED-FORM LEG (addendum §A3.2 — "the flight table was wrong at every level: the ramp was
/// never integrated"): the seconds a full-throttle leg of `distance_m` takes under the law —
/// ramp in from `v_start` (`τ·ln(v_cap/v_start)` over `τ·(v_cap−v_start)` metres), cruise at
/// `v_cap`, governor ramp out to `v_end` (`τ·ln(v_cap/v_end)` over `τ·(v_cap−v_end)` metres):
///
/// `leg = τ·ln(v_cap/v_start) + (D − τ·(v_cap−v_start) − τ·(v_cap−v_end)) / v_cap + τ·ln(v_cap/v_end)`
///
/// `None` when the distance cannot hold both ramps (no cruise exists — the short-leg regime is a
/// different closed form nobody flies in the gates; refusing loudly beats a wrong number). The
/// ramp is logarithmic in the speed ratio, so it does NOT scale away as the world grows — that is
/// the addendum's whole correction, and the flight-table gate measures the integrator against
/// exactly this expression.
#[must_use]
pub fn leg_time_s(
    distance_m: f64,
    v_cap_mps: f64,
    v_start_mps: f64,
    v_end_mps: f64,
    tau_s: f64,
) -> Option<f64> {
    let ramp_in_m = tau_s * (v_cap_mps - v_start_mps);
    let ramp_out_m = tau_s * (v_cap_mps - v_end_mps);
    let cruise_m = distance_m - ramp_in_m - ramp_out_m;
    if cruise_m < 0.0 {
        return None;
    }
    Some(
        tau_s * (v_cap_mps / v_start_mps).ln()
            + cruise_m / v_cap_mps
            + tau_s * (v_cap_mps / v_end_mps).ln(),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The DEV-cluster tuning every derived-gate number stands on: 50 Hz (cadence 25), reactive
    /// boot 0 ⇒ τ = (2·25 + 0 + 5)·0.02 = 1.10 s exactly.
    fn dev() -> FlightTuning {
        FlightTuning::derive(500.0, 0.02, 25, 0)
    }

    #[test]
    fn tau_is_the_wake_budget_in_seconds() {
        let t = dev();
        assert_eq!(t.tau_s, 1.1);
        assert_eq!(t.v_foot_mps, 500.0);
        assert_eq!(t.tick_dt_s, 0.02);
        assert_eq!(t.traverse_s, TRAVERSE_S);
        // A measured boot p99 lengthens τ by exactly its ticks: 122 boot ticks (the design's
        // placeholder) ⇒ (50 + 122 + 5)·0.02 = 3.54 s.
        assert_eq!(FlightTuning::derive(500.0, 0.02, 25, 122).tau_s, 3.54);
    }

    #[test]
    fn the_ceiling_clamps_below_break_even_and_scales_above() {
        // Below the 45 km break-even (v_foot·T/2 = 500·180/2) the ceiling IS the floor, exactly —
        // the inertness the whole interim world stands on.
        assert_eq!(realm_speed_cap_mps(150.0, 500.0, TRAVERSE_S), 500.0);
        assert_eq!(realm_speed_cap_mps(45_000.0, 500.0, TRAVERSE_S), 500.0);
        // Above it, the ceiling is the realm's width over T: the DEV galaxy row (addendum §A3.1's
        // 2.4986638e13 m/s), asserted as the expression's own arithmetic so no over-precise
        // literal is transcribed.
        let r_gal = 2.248_797_413_933_667_8e15;
        assert_eq!(
            realm_speed_cap_mps(r_gal, 500.0, TRAVERSE_S),
            2.0 * r_gal / TRAVERSE_S,
        );
        // The break-even edge itself: 2·45 000/180 = 500 exactly — max of equals is the floor.
        assert_eq!(realm_speed_cap_mps(45_000.0, 500.0, TRAVERSE_S), 500.0);
    }

    #[test]
    fn the_governor_arm_is_the_child_cap_plus_distance_over_tau() {
        // At the bound (dist 0): exactly the child's own ceiling — you arrive at ITS speed.
        assert_eq!(approach_ceiling_mps(500.0, 0.0, 1.1), 500.0);
        // 11 308 m out (the wake line of a 150 m system): 500 + 11 308/1.1.
        assert_eq!(
            approach_ceiling_mps(500.0, 11_308.0, 1.1),
            500.0 + 11_308.0 / 1.1
        );
        // INSIDE the bound (negative distance) clamps to the child cap — never below it.
        assert_eq!(approach_ceiling_mps(500.0, -25.0, 1.1), 500.0);
    }

    #[test]
    fn the_ramp_compounds_proportionally_and_floors_at_the_foot() {
        let t = dev();
        let g = (t.tick_dt_s / t.tau_s).exp();
        // From a standstill (and from anything below the foot) the ramp allows the FULL foot speed
        // at once — the floor, so a stopped ship is never trapped at zero.
        assert_eq!(ramp_cap_mps(0.0, 500.0, t.tick_dt_s, t.tau_s), 500.0 * g);
        assert_eq!(ramp_cap_mps(100.0, 500.0, t.tick_dt_s, t.tau_s), 500.0 * g);
        // Above the foot it compounds from the previous speed: constant proportional rate.
        assert_eq!(ramp_cap_mps(1.0e6, 500.0, t.tick_dt_s, t.tau_s), 1.0e6 * g);
        // n ticks compound to e^(n·dt/τ): the τ·ln(ratio) ramp time, discretely.
        let n = 100u32;
        let mut v = 500.0f64;
        for _ in 0..n {
            v = ramp_cap_mps(v, 500.0, t.tick_dt_s, t.tau_s);
        }
        let closed = 500.0 * (f64::from(n) * t.tick_dt_s / t.tau_s).exp();
        assert!((v - closed).abs() <= closed * 1e-12, "{v} vs {closed}");
    }

    #[test]
    fn a_clamped_ceiling_scales_by_exactly_one() {
        // THE INERTNESS PIN (bit-identity, not near-equality): at v_cap == v_foot the scale is
        // 1.0 EXACTLY for every unit-or-less stick — x/x == 1.0, 1^t == 1.0, and 1·1 == 1.0 in
        // IEEE 754 — so `axes · (scale · k)` is the old `axes · k` bit-for-bit.
        for mag in [1.0e-6, 0.03, 0.05, 0.5, 0.999_999, 1.0] {
            assert_eq!(throttle_axes_scale(mag, 500.0, 500.0), 1.0, "mag {mag}");
        }
        // The same identity at the fixtures' 2 m/s foot speed.
        assert_eq!(throttle_axes_scale(0.25, 2.0, 2.0), 1.0);
    }

    #[test]
    fn zero_stick_is_stop_and_the_map_is_geometric() {
        // ZERO IS STOP — the multiply carries it, not the exponent.
        assert_eq!(throttle_axes_scale(0.0, 500.0, 2.5e13), 0.0);
        assert_eq!(throttle_axes_scale(-1.0, 500.0, 2.5e13), 0.0);
        // Full stick reaches the ceiling exactly: scale·v_foot == v_cap.
        let cap = 1.750_767_8e9;
        assert_eq!(throttle_axes_scale(1.0, 500.0, cap) * 500.0, cap);
        // Half stick's BOOST is the geometric middle: v(0.5)/0.5 = √(v_foot·v_cap) — the owner
        // clause that keeps slow deep-space flight possible (a linear map's middle would be
        // v_cap/2, seven orders louder here).
        let v_half = throttle_axes_scale(0.5, 500.0, cap) * 500.0 * 0.5;
        let geometric_middle = 0.5 * (500.0f64 * cap).sqrt();
        assert!(
            (v_half - geometric_middle).abs() <= geometric_middle * 1e-12,
            "{v_half} vs {geometric_middle}",
        );
        // CREEP (0.05) in the home system stays a room-scale speed under a ~1.75e9 ceiling —
        // §4.2(b)'s whole point (the linear map's creep there is 8.75e7 m/s). The equation's own
        // value: 0.05·500·ratio^0.05.
        let creep = throttle_axes_scale(0.05, 500.0, cap) * 500.0 * 0.05;
        assert_eq!(creep, 0.05 * 500.0 * (cap / 500.0f64).powf(0.05));
        assert!(creep < 60.0, "creep {creep} stays room-scale");
    }

    #[test]
    fn an_over_unit_diagonal_is_normalized_to_the_ceiling() {
        // |axes| = √3 (a triple-key diagonal): the commanded speed is the FULL-throttle speed, not
        // √3× it — scale·|axes|·v_foot == v_cap at any ceiling (the cure §4.2(b)'s normalize buys).
        let mag = 3.0f64.sqrt();
        for cap in [500.0, 2.5e13] {
            let speed = throttle_axes_scale(mag, 500.0, cap) * mag * 500.0;
            assert!(
                (speed - cap).abs() <= cap * 1e-12,
                "cap {cap}: diagonal commanded {speed}",
            );
        }
    }

    #[test]
    fn the_closed_form_leg_integrates_both_ramps_and_refuses_a_ramp_only_distance() {
        let tau = 1.1;
        // The three-part sum, checked against an independently-spelled expansion.
        let (d, cap, v0, v1) = (
            2.248_490_504_408_914e15,
            2.0 * 2.248_797_413_933_667_8e15 / TRAVERSE_S,
            500.0,
            500.0,
        );
        let t = leg_time_s(d, cap, v0, v1, tau).expect("the warp leg holds a cruise");
        let by_parts = tau * (cap / v0).ln()
            + (d - tau * (cap - v0) - tau * (cap - v1)) / cap
            + tau * (cap / v1).ln();
        assert_eq!(t, by_parts);
        // A distance shorter than its own two ramps has no cruise: the typed refusal, never a
        // negative cruise silently folded in.
        assert_eq!(leg_time_s(1.0e3, cap, v0, v1, tau), None);
        // A ceiling equal to both ends degenerates to pure cruise: D/v exactly.
        assert_eq!(leg_time_s(1000.0, 500.0, 500.0, 500.0, tau), Some(2.0));
    }
}
