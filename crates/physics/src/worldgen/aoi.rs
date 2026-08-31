//! A REGION'S INTEREST BAND: how far out a realm itself can still be seen from.
//!
//! Owns: the velocity floor that keeps a realm's band honest while the realm moves at its own
//! traverse speed.
//!
//! Does NOT own: any realm kind, and — since 2026-08-29 — anything about a realm's CHILDREN. The
//! band is a uniform factor of a realm's OWN finite extent. A bigger realm gets a bigger band by
//! arithmetic, never by a match on what it is (HR3), and never by looking inside it.
//!
//! ★ WHAT WAS DELETED HERE, AND WHY (owner ruling 2026-08-29). This module used to own a second
//! band: a realm's INTERIOR REACH, taken as the maximum over its DIRECT CHILDREN of their
//! excursion plus their visibility. A parent judged "is this child's inside worth waking" from it.
//!
//! It was the ONE place in the system where a parent read its child's contents, and it could not
//! survive the rule that a shard builds only its own subtree — a subtree stops one level down, so
//! the children the reach needs are absent. MEASURED on THE world: all 233 220 star systems on a
//! galaxy shard carried a reach of ZERO, so the galaxy told no system to warm anything, ever.
//!
//! The cure is that a realm carries ONE radius, derived from its own size, and a parent uses that
//! same radius for both jobs: whether to WAKE a child, and whether to TELL that child a looker is
//! near. One radius, one test, every realm kind, any depth.

use super::T_TRAVERSE_S;

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
