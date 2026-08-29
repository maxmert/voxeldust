//! A REGION'S INTEREST BAND: how far out something inside a realm can still be seen from.
//!
//! Owns: the interior reach (the largest distance from a realm's centre at which something INSIDE it
//! still matters), the band that reach becomes, and the velocity floor that keeps the band honest
//! for a realm moving at its own traverse speed.
//!
//! Does NOT own: any realm kind. The band is a uniform factor of a realm's own finite extent — a
//! bigger realm gets a bigger band by arithmetic, never by a match on what it is (HR3).

use super::{
    GeneratedBody, InterestConfig, T_TRAVERSE_S, VISIBILITY_THETA_MIN_RAD,
    worst_hop_excursion_capped_m,
};
use vd_core::geometry::AoiConfig;

/// A body's INTERIOR REACH (look_horizon.md §3.4.4): the largest distance from its centre at
/// which something INSIDE it is still visible — the max over its DIRECT children of (that
/// child's worst-instant excursion at the eccentricity cap + that child's visibility reach),
/// the same two terms the climb measurement walks with (§3.3.2's identity: one formula, one
/// worst-case convention). `0.0` for a childless leaf — nothing inside, nothing to reach.
/// (THE world's star-system reach used to be quoted here as `142.045826247 + 302.058663384 =
/// 444.104489631` m. That was the retired compressed geometry; since the true-size in-system
// (`interior_reach_m` IS DELETED, 2026-08-29. It filtered the WHOLE body list by parent and was
// called once per body from the lowering — a quadratic that made every boot 1.8e13 comparisons at
// THE world's census. `to_regions` solves every parent's reach in ONE pass and both paths read the
// same `child_reach_term_m`, so this slower spelling had no caller left. A dead slow version of a
// live fast one is a trap: the next author reaches for whichever they find first.)

/// ONE CHILD'S CONTRIBUTION to its parent's interior reach — the term
/// [`interior_reach_m`] takes the maximum of.
///
/// ★ LIFTED OUT SO THE TWO CALLERS CANNOT DRIFT (perf fix 2026-08-29). The lowering solves every
/// parent's reach in ONE pass instead of re-scanning the body list per body (see `to_regions` for
/// the measurement that forced it), and this is the term both paths read. Two spellings of one
/// formula is how a fast path and a slow path stop agreeing.
///
/// The visibility term reads the child's LOOK (the picture that can be seen), not its bound
/// (real-scale design §3.0); a look-less child contributes no reach.
pub(crate) fn child_reach_term_m(child: &GeneratedBody, ecc_cap: f64) -> f64 {
    worst_hop_excursion_capped_m(&child.placement, ecc_cap)
        + child.look.map_or(0.0, |look| {
            vd_core::geometry::visibility_reach_m(look.finite_extent(), VISIBILITY_THETA_MIN_RAD)
        })
}

/// The interior band for one child region (look_horizon.md §3.4.4, monomorphic — both arms
/// driven by named tests): spin-up AT the interior reach, tear-down widened by the SAME derived
/// velocity lead every AoI band carries (`|v_rel|·dt·(K_SAFETY + extra)` — no new number
/// anywhere). Inert for a leaf (zero reach) and wherever the whole AoI machinery is inert
/// (walk/canonical byte-identity: the live ctor's reject arm is never touched there, the same
/// discipline as [`InterestConfig::build`]).
pub(crate) fn interior_band(reach_m: f64, interest: &InterestConfig, v_child: f64) -> AoiConfig {
    if (reach_m <= 0.0) | !interest.is_live() {
        AoiConfig::inert()
    } else {
        AoiConfig::for_velocity_safe(
            reach_m,
            1.0,
            1.0,
            aoi_v_rel_mps(reach_m, interest.occupant_v_max_mps + v_child),
            interest.tick_dt_s,
            interest.grace_ticks,
            interest.k_safety_extra,
        )
        .expect("a positive reach with a positive closing speed builds a valid band")
    }
}

/// The AoI dead-zone's velocity input, FLOORED at the realm's own traverse speed
/// `2·extent / T_TRAVERSE_S` — the §4.2(a) speed-cap expression used here as the derived scale
/// floor the real-scale geometry needs (addendum H-11's cure, landed at the geometry slice: "the
/// AoI dead-zone is a constant 2.5 ticks at every scale" broke structurally the day the shells
/// went astronomic — at a 2.25e15 m ambient extent a 15 m/s pad is BELOW ONE ULP of the spin-up
/// radius, so `spin + pad == spin` and the band constructor rightly refused the collapsed
/// dead-zone). The floor is the same chain the outer geometry derives from (`T_TRAVERSE_S`), so
/// no new number enters; once the speed-law slice lands, `v_rel` can never be below the realm's
/// own cap anyway — this lands that clearance early. Byte-identical on every interim-scale row,
/// MEASURED: the floor binds only where `extent > v_rel·T/2` (the two ambient shells on THE
/// world; no walk row — walk's distinct tear factor out-binds the pad everywhere).
pub(crate) fn aoi_v_rel_mps(extent_m: f64, v_rel_mps: f64) -> f64 {
    v_rel_mps.max(2.0 * extent_m / T_TRAVERSE_S)
}
