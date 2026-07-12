//! Overlap-band geometry: where ghosts exist and transfers trigger
//! (`docs/design/transfer_protocol.md` §1.3, hardened by Charter A).
//!
//! Bands are GEOMETRIC (never timeouts — the old system's R8 was timeout-based
//! eviction) with three protections:
//! - **Hysteresis**: ghosts are created crossing the inner edge and destroyed only
//!   beyond the outer edge, so a boundary-hovering entity cannot flap.
//! - **Velocity scaling**: band width must exceed `v_rel · dt · K_SAFETY` so a fast
//!   body cannot skip the band between ticks.
//! - **Swept-segment crossing**: membership evaluation tests the tick's whole motion
//!   segment against the shell, so a 5.5 km/tick body cannot tunnel undetected.
//!
//! Also home of [`system_soi`] — the LUMINOSITY-based, GALAXY-UNIT star-system SOI,
//! deliberately distinct from the mass-ratio, METERS [`crate::celestial::planet_soi`].

use glam::{DQuat, DVec3};
use serde::{Deserialize, Serialize};

use crate::pose::{LatticePos, RealmId};

/// Base star-system SOI radius in galaxy units (ported from the reference repo's
/// `galaxy.rs`; part of the galaxy-scale definition, not a tunable).
pub const BASE_SOI_RADIUS: f64 = 100.0;
/// SOI growth per unit of stellar luminosity, galaxy units.
pub const SOI_LUMINOSITY_SCALE: f64 = 200.0;

/// Minimum band-width safety factor: `(outer - inner) >= v_rel · dt · K_SAFETY`
/// (design value K ≥ 2; a per-deployment band-tuning config — owed with the D-2 band
/// geometry at P4/P5 — may raise it, never lower. No `TransferTuning` struct exists yet).
pub const K_SAFETY: f64 = 2.0;

/// Band-edge factors from the per-class band table (`transfer_protocol.md` §1.3).
pub const PLANET_SOI_CREATE_FACTOR: f64 = 1.15;
pub const PLANET_SOI_DESTROY_FACTOR: f64 = 1.30;
pub const SYSTEM_SOI_CREATE_FACTOR: f64 = 0.95;
pub const SYSTEM_SOI_DESTROY_FACTOR: f64 = 1.05;

/// Motion-scaled band edges in units of per-tick travel (`v_rel · dt`), for a boundary that
/// has NO sphere-of-influence geometry to scale off — the band is sized purely so a body moving
/// `per_tick_travel` per tick cannot skip it (velocity-safe) and exits only after travelling a
/// bounded multiple of its per-tick step. The destroy edge is set generously (`> ` the inner edge
/// by well over [`K_SAFETY`]) so a ghost anchored AT a boundary crossing is torn down strictly
/// AFTER the short handoff window the entity walks through — never during it.
///
/// This is the INTERIM band for the pre-spatial stub (and any future fine-grained boundary lacking
/// an SOI radius); production realm bands use [`OverlapBand::for_planet_soi`] /
/// [`OverlapBand::for_system_soi`], which derive their edges from `r_soi` and have no
/// anchor-at-crossing artifact. (`docs/design/DEFERRED.md` D-2.)
pub const MOTION_BAND_CREATE_TRAVELS: f64 = 16.0;
pub const MOTION_BAND_DESTROY_TRAVELS: f64 = 20.0;

/// Star-system sphere-of-influence radius in GALAXY UNITS:
/// `BASE_SOI_RADIUS + luminosity · SOI_LUMINOSITY_SCALE`.
///
/// Deliberately distinct from [`crate::celestial::planet_soi`] (different formula,
/// units, and meaning); a test below pins both to their reference ranges so a future
/// refactor cannot silently collapse them.
#[must_use]
pub fn system_soi(luminosity: f64) -> f64 {
    BASE_SOI_RADIUS + luminosity * SOI_LUMINOSITY_SCALE
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, thiserror::Error)]
pub enum BandError {
    #[error("band edges must satisfy 0 < create_below < destroy_above")]
    InvalidEdges,
}

/// A hysteresis overlap band around a boundary shell: ghosts/interest are created
/// when an entity moves inside `create_below` and destroyed only beyond
/// `destroy_above`.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct OverlapBand {
    create_below: f64,
    destroy_above: f64,
}

impl OverlapBand {
    pub fn new(create_below: f64, destroy_above: f64) -> Result<OverlapBand, BandError> {
        if create_below > 0.0 && create_below < destroy_above {
            Ok(OverlapBand {
                create_below,
                destroy_above,
            })
        } else {
            Err(BandError::InvalidEdges)
        }
    }

    /// The planet-SOI band (descent): create at `1.15·r_soi`, destroy at `1.30·r_soi`.
    #[must_use]
    pub fn for_planet_soi(r_soi_m: f64) -> OverlapBand {
        OverlapBand {
            create_below: r_soi_m * PLANET_SOI_CREATE_FACTOR,
            destroy_above: r_soi_m * PLANET_SOI_DESTROY_FACTOR,
        }
    }

    /// The system-SOI band (warp/interstellar): `0.95·r` / `1.05·r`.
    #[must_use]
    pub fn for_system_soi(r_soi: f64) -> OverlapBand {
        OverlapBand {
            create_below: r_soi * SYSTEM_SOI_CREATE_FACTOR,
            destroy_above: r_soi * SYSTEM_SOI_DESTROY_FACTOR,
        }
    }

    /// The motion-scaled band for a boundary with no SOI geometry: edges are multiples of the
    /// per-tick travel `v_rel · dt` ([`MOTION_BAND_CREATE_TRAVELS`] / [`MOTION_BAND_DESTROY_TRAVELS`]).
    /// Infallible (the factors are constant and ordered `0 < create < destroy`), velocity-safe by
    /// construction (the gap exceeds [`K_SAFETY`] travels), and sized so the destroy edge is many
    /// per-tick steps out — see the const docs. INTERIM for the pre-spatial stub; production realms
    /// use [`OverlapBand::for_planet_soi`] / [`OverlapBand::for_system_soi`].
    #[must_use]
    pub fn for_motion(per_tick_travel: f64) -> OverlapBand {
        OverlapBand {
            create_below: per_tick_travel * MOTION_BAND_CREATE_TRAVELS,
            destroy_above: per_tick_travel * MOTION_BAND_DESTROY_TRAVELS,
        }
    }

    #[must_use]
    pub fn create_below(&self) -> f64 {
        self.create_below
    }

    #[must_use]
    pub fn destroy_above(&self) -> f64 {
        self.destroy_above
    }

    /// Velocity-scaled width invariant: a band is safe for an entity moving at
    /// `v_rel` (units/s) over ticks of `dt_s` iff the hysteresis gap exceeds the
    /// per-tick travel times [`K_SAFETY`].
    #[must_use]
    pub fn width_safe_for(&self, v_rel: f64, dt_s: f64) -> bool {
        (self.destroy_above - self.create_below) >= v_rel.abs() * dt_s * K_SAFETY
    }

    /// Hysteresis membership update: given the previous membership and the current
    /// distance from the boundary center, is the entity in the band now?
    #[must_use]
    pub fn update_membership(&self, was_member: bool, distance: f64) -> bool {
        if was_member {
            distance <= self.destroy_above
        } else {
            distance <= self.create_below
        }
    }
}

/// How a motion segment relates to a spherical shell of radius `r` centered at the
/// origin — the anti-tunneling primitive.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum ShellCrossing {
    /// Both endpoints and the whole segment stay outside.
    StaysOutside,
    /// Both endpoints and the whole segment stay inside.
    StaysInside,
    /// The segment ends inside (entered through the shell).
    Inward,
    /// The segment ends outside having started inside (exited through the shell).
    Outward,
    /// Entered AND exited within one segment — the tunneling case a point sample
    /// would miss entirely.
    ThroughAndBack,
}

/// Classify the swept segment `p0 -> p1` against the shell `|p| = r`.
#[must_use]
pub fn segment_shell_crossing(p0: DVec3, p1: DVec3, r: f64) -> ShellCrossing {
    let start_inside = p0.length() <= r;
    let end_inside = p1.length() <= r;
    match (start_inside, end_inside) {
        (true, true) => ShellCrossing::StaysInside,
        (true, false) => ShellCrossing::Outward,
        (false, true) => ShellCrossing::Inward,
        (false, false) => {
            // Both endpoints outside: did the segment dip through the shell?
            // Closest point of the segment to the origin decides.
            let d = p1 - p0;
            let len_sq = d.length_squared();
            if len_sq == 0.0 {
                return ShellCrossing::StaysOutside; // degenerate: a stationary point
            }
            let t = (-(p0.dot(d)) / len_sq).clamp(0.0, 1.0);
            let closest = p0 + d * t;
            if closest.length() <= r {
                ShellCrossing::ThroughAndBack
            } else {
                ShellCrossing::StaysOutside
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Spatial boundaries (Slice 1): APPEND-ONLY pure geometry — the shapes a realm's
// overlap band lives on. NOTHING calls this yet (the crossing state machine and
// commit are Slice 3); it exists so the band hysteresis proven for spherical
// shells generalizes verbatim to axis-aligned and oriented boxes (stations,
// areas, ship bays) without re-proving the anti-tunneling / anti-flap invariants.
// ---------------------------------------------------------------------------

/// The direction an entity crosses a boundary, relative to the boundary interior.
/// `Ord` (`Inward < Outward`, declaration order) is the FINAL tiebreak in
/// [`resolve_winner`], keeping its total order strict even when two boundaries share a
/// realm at one depth (so the winner is slice-order-independent — determinism is
/// load-bearing cross-host).
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum Direction {
    /// Toward the interior (entering).
    Inward,
    /// Away from the interior (leaving).
    Outward,
}

/// The geometric shape of a realm boundary, centered at the boundary's own origin
/// (callers pass CENTER-RELATIVE positions — the caller subtracts the boundary
/// center before invoking any method here).
///
/// All three shapes expose the SAME three primitives — a swept crossing classifier
/// ([`Boundary::swept`], anti-tunneling), a monotone radial-analog membership scalar
/// the [`OverlapBand`] consumes ([`Boundary::membership_scalar`], anti-flap), and a
/// signed distance for the future commit line ([`Boundary::signed_distance`]) — so the
/// spherical-shell band math is reused unchanged for boxes.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub enum Boundary {
    /// A spherical shell of radius `r` — the SOI / planet-descent boundary.
    Shell { r: f64 },
    /// An axis-aligned box with per-axis half-extents `half` — a station volume.
    Aabb { half: DVec3 },
    /// An oriented box: half-extents `half` in a frame rotated by `orient` from the
    /// boundary-center frame. Every method rotates the query into box-local via
    /// `orient.inverse()` and then runs the SAME AABB algorithm (a shim), so there is
    /// exactly one box implementation.
    Obb { half: DVec3, orient: DQuat },
}

impl Boundary {
    /// Classify the swept segment `p0 -> p1` (center-relative) against this boundary.
    /// The anti-tunneling primitive — evaluates the whole motion segment, never a point
    /// sample. Obb rotates the segment into box-local and reuses the AABB algorithm.
    #[must_use]
    pub fn swept(&self, p0: DVec3, p1: DVec3) -> ShellCrossing {
        match self {
            Boundary::Shell { r } => segment_shell_crossing(p0, p1, *r),
            Boundary::Aabb { half } => segment_aabb_crossing(p0, p1, *half),
            Boundary::Obb { half, orient } => {
                let inv = orient.inverse();
                segment_aabb_crossing(inv * p0, inv * p1, *half)
            }
        }
    }

    /// The RADIAL-ANALOG monotone scalar the [`OverlapBand`] consumes: for a shell it is
    /// `|p|` (BIT-IDENTICAL to today's band input — a proptest pins this), for a box it is
    /// the normalized Chebyshev norm ([`chebyshev_norm`], 1.0 on the surface). Feed this to
    /// [`OverlapBand::update_membership`]; pair a box boundary with a normalized band
    /// ([`OverlapBand::for_box`]).
    #[must_use]
    pub fn membership_scalar(&self, p: DVec3) -> f64 {
        match self {
            Boundary::Shell { r: _ } => p.length(),
            Boundary::Aabb { half } => chebyshev_norm(p, *half),
            Boundary::Obb { half, orient } => chebyshev_norm(orient.inverse() * p, *half),
        }
    }

    /// Signed distance to the surface: NEGATIVE inside, positive outside, ~0 on the surface.
    /// This is for the (future, Slice 3) commit line — it is DISTINCT from
    /// [`Boundary::membership_scalar`] and must NEVER be fed to the band (the band consumes
    /// the radial-analog scalar, not a metric distance).
    #[must_use]
    pub fn signed_distance(&self, p: DVec3) -> f64 {
        match self {
            Boundary::Shell { r } => p.length() - r,
            Boundary::Aabb { half } => box_signed_distance(p, *half),
            Boundary::Obb { half, orient } => box_signed_distance(orient.inverse() * p, *half),
        }
    }
}

/// The normalized Chebyshev norm of `p` against box half-extents `half`: the max over axes
/// of `|p_i| / half_i`. Equals 1.0 on the box surface, `< 1` strictly inside, `> 1` outside;
/// its level sets are SCALED BOXES, so a band window on this scalar is a uniform-thickness
/// shell around the WHOLE perimeter (faces AND corners). `half` is a precondition `> 0`
/// (enforced by [`BoundaryTuning`] / the caller, not this branchless expression).
#[must_use]
fn chebyshev_norm(p: DVec3, half: DVec3) -> f64 {
    (p.abs() / half).max_element()
}

/// Signed distance from `p` to the surface of the axis-aligned box `|p_i| <= half_i`
/// (the standard exterior/interior box distance): NEGATIVE inside, positive outside.
#[must_use]
fn box_signed_distance(p: DVec3, half: DVec3) -> f64 {
    // Per-axis signed overshoot: >0 outside that slab, <0 inside it.
    let q = p.abs() - half;
    // Exterior part: Euclidean length of the positive overshoot (0 when inside every slab).
    let outside = q.max(DVec3::ZERO).length();
    // Interior part: the least-negative axis, capped at 0 (0 when outside any slab).
    let inside = q.max_element().min(0.0);
    outside + inside
}

/// Is the center-relative point `p` inside the axis-aligned box `|p_i| <= half_i`?
/// On-surface (`==`) counts as inside (the `<=` matches the shell's `<=`).
#[must_use]
fn aabb_contains(p: DVec3, half: DVec3) -> bool {
    p.abs().cmple(half).all()
}

/// Slab (Kay–Kajiya) test: does the segment `p0 -> p1` (`t ∈ [0, 1]`) intersect the
/// axis-aligned box `|p_i| <= half_i`? Used only when BOTH endpoints are outside, so a hit
/// means the segment passes THROUGH the box (the box tunneling case a point sample misses).
#[must_use]
fn segment_hits_aabb(p0: DVec3, p1: DVec3, half: DVec3) -> bool {
    let o = p0.to_array();
    let d = (p1 - p0).to_array();
    let h = half.to_array();
    let mut tmin = 0.0_f64;
    let mut tmax = 1.0_f64;
    for i in 0..3 {
        if d[i] == 0.0 {
            // Segment parallel to axis i: it can never enter this slab, so if the origin is
            // already outside the slab there is no intersection at all.
            if o[i] < -h[i] || o[i] > h[i] {
                return false;
            }
        } else {
            let inv = 1.0 / d[i];
            let t1 = (-h[i] - o[i]) * inv;
            let t2 = (h[i] - o[i]) * inv;
            // Order the two slab-plane hits (the segment may run in the -axis direction).
            let (lo, hi) = if t1 <= t2 { (t1, t2) } else { (t2, t1) };
            tmin = tmin.max(lo);
            tmax = tmax.min(hi);
            if tmin > tmax {
                return false;
            }
        }
    }
    true
}

/// Classify the swept segment `p0 -> p1` against the axis-aligned box `|p_i| <= half_i` — the
/// box analog of [`segment_shell_crossing`], mirroring its five cases. When both endpoints are
/// outside, the slab test decides through-and-back vs a clean miss.
#[must_use]
pub fn segment_aabb_crossing(p0: DVec3, p1: DVec3, half: DVec3) -> ShellCrossing {
    let start_inside = aabb_contains(p0, half);
    let end_inside = aabb_contains(p1, half);
    match (start_inside, end_inside) {
        (true, true) => ShellCrossing::StaysInside,
        (true, false) => ShellCrossing::Outward,
        (false, true) => ShellCrossing::Inward,
        (false, false) => {
            if segment_hits_aabb(p0, p1, half) {
                ShellCrossing::ThroughAndBack
            } else {
                ShellCrossing::StaysOutside
            }
        }
    }
}

/// What a boundary crossing drives: an AUTHORITY handoff (the entity's owning realm changes —
/// a transfer) versus an INTEREST change (ghost/subscription only — no authority move).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum CrossEffect {
    /// Crossing hands authority to `to_realm` (a transfer / commit).
    Authority,
    /// Crossing only creates or drops interest (a ghost), authority unchanged.
    Interest,
}

/// One realm boundary: its shape, its center in the coordinate lattice, the hysteresis band
/// around it, the realm it belongs to, the realm entered on an inward crossing, and whether a
/// crossing moves authority or only interest. Every field is `Copy`, so the whole descriptor is.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct RealmBoundary {
    /// The realm this boundary belongs to (its exterior side).
    pub realm: RealmId,
    /// The boundary center in the lattice (callers subtract this before calling shape methods).
    pub center: LatticePos,
    /// The boundary shape, centered at `center`.
    pub shape: Boundary,
    /// The hysteresis overlap band around the shape.
    pub band: OverlapBand,
    /// The parent realm, when this boundary nests inside another (`None` at the top level).
    pub parent: Option<RealmId>,
    /// The realm an entity enters on an INWARD crossing.
    pub to_realm: RealmId,
    /// Whether crossing moves authority (a transfer) or only interest (a ghost).
    pub effect: CrossEffect,
}

/// Per-deployment boundary tuning: the dwell/entry hysteresis counts, the minimum velocity pad,
/// the extra band-safety margin over [`K_SAFETY`], and the coordinate cell size. Validated once
/// at config load ([`BoundaryTuning::validate`]); the geometry fns take pre-validated scalars.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct BoundaryTuning {
    /// Consecutive in-band ticks required before an entry commit (anti-flap on entry).
    pub n_entry: u32,
    /// Consecutive out-of-band ticks required before teardown (anti-flap on exit).
    pub k_dwell: u32,
    /// The floor on the velocity pad added to the destroy edge (a slow body still gets a band).
    pub velocity_pad_floor: f64,
    /// Extra safety multiplier ADDED to [`K_SAFETY`] when widening the destroy edge.
    pub k_safety_extra: f64,
    /// The coordinate cell size in meters.
    pub cell_size_m: f64,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, thiserror::Error)]
pub enum BoundaryTuningError {
    #[error("n_entry, k_dwell, velocity_pad_floor, and cell_size_m must all be > 0")]
    NonPositive,
    #[error("k_safety_extra must be >= 0 (it is added to K_SAFETY, never subtracted)")]
    KSafetyTooLow,
}

impl BoundaryTuning {
    /// Sane defaults: 3-tick entry, 5-tick dwell, 0.5 m velocity floor, +1.0 safety over
    /// [`K_SAFETY`], 1 km cells.
    pub const DEFAULT: BoundaryTuning = BoundaryTuning {
        n_entry: 3,
        k_dwell: 5,
        velocity_pad_floor: 0.5,
        k_safety_extra: 1.0,
        cell_size_m: 1000.0,
    };

    /// Reject a zero/negative count or scale ([`BoundaryTuningError::NonPositive`]) and a
    /// negative safety extra ([`BoundaryTuningError::KSafetyTooLow`]); `Ok` otherwise.
    pub fn validate(&self) -> Result<(), BoundaryTuningError> {
        if self.n_entry == 0
            || self.k_dwell == 0
            || self.velocity_pad_floor <= 0.0
            || self.cell_size_m <= 0.0
        {
            return Err(BoundaryTuningError::NonPositive);
        }
        if self.k_safety_extra < 0.0 {
            return Err(BoundaryTuningError::KSafetyTooLow);
        }
        Ok(())
    }
}

/// Pick EXACTLY ONE winner among the boundaries an entity commits against on the same tick.
/// The total order is INNERMOST-first (MAX `depth`), ties broken by `RealmId`'s derived `Ord`
/// (MIN). `None` if empty. PERMUTATION-INVARIANT: the winner is independent of slice order
/// (the order is total, so the fold's result is order-free) — a proptest pins this.
#[must_use]
pub fn resolve_winner(candidates: &[(u32, RealmId, Direction)]) -> Option<(RealmId, Direction)> {
    candidates
        .iter()
        .copied()
        .reduce(|best, cur| {
            if candidate_beats(cur, best) {
                cur
            } else {
                best
            }
        })
        .map(|(_, realm, dir)| (realm, dir))
}

/// Does candidate `a` beat the current best `b` under the STRICT TOTAL order — deeper wins;
/// equal depth, smaller `RealmId` wins; equal depth AND realm, smaller `Direction` (`Inward`)
/// wins? The final direction tiebreak closes the reachable collision the reviewer found — two
/// boundary mouths into ONE realm at the same depth yield `(d, realm, Inward)` + `(d, realm,
/// Outward)`; without it the fold kept whichever came first (slice-order-dependent = non-
/// deterministic). A monotonic helper so the tie-break branching is fully covered here, not
/// smeared through the fold's closure.
#[must_use]
fn candidate_beats(a: (u32, RealmId, Direction), b: (u32, RealmId, Direction)) -> bool {
    if a.0 != b.0 {
        a.0 > b.0
    } else if a.1 != b.1 {
        a.1 < b.1
    } else {
        a.2 < b.2
    }
}

impl OverlapBand {
    /// An SOI band whose destroy edge is widened so the create -> destroy gap stays
    /// velocity-safe even on a CORNER (diagonal) approach: `create = r_soi · create_factor`;
    /// `destroy = max(r_soi · destroy_factor, create + max(pad_floor, |v_rel| · dt · (K_SAFETY +
    /// k_safety_extra)))`. The `debug_assert!` pins the [`OverlapBand::width_safe_for`] invariant.
    #[must_use]
    pub fn for_soi_velocity_safe(
        r_soi: f64,
        create_factor: f64,
        destroy_factor: f64,
        v_rel: f64,
        dt: f64,
        pad_floor: f64,
        k_safety_extra: f64,
    ) -> OverlapBand {
        let create = r_soi * create_factor;
        let velocity_pad = f64::max(pad_floor, v_rel.abs() * dt * (K_SAFETY + k_safety_extra));
        let destroy = f64::max(r_soi * destroy_factor, create + velocity_pad);
        let band = OverlapBand {
            create_below: create,
            destroy_above: destroy,
        };
        debug_assert!(band.width_safe_for(v_rel, dt));
        band
    }

    /// The UNITLESS band that pairs with the normalized [`chebyshev_norm`] box scalar: an entity
    /// enters when its Chebyshev norm drops below `create_factor` and leaves above `destroy_factor`
    /// (e.g. `1.15` / `1.30`). Because the norm is 1.0 on the surface, these factors are the same
    /// scale-relative fractions the shell bands use.
    ///
    /// Unlike the SOI constructors (which derive edges from ORDERED constant factors and so are
    /// infallible), this takes CALLER factors, so it is fallible: it routes through
    /// [`OverlapBand::new`] and returns [`BandError::InvalidEdges`] for a non-positive or inverted
    /// pair — an inverted band would INVERT the hysteresis (a guaranteed per-tick flap), the exact
    /// failure this module exists to prevent, so it must never be constructed silently.
    ///
    /// # Errors
    /// [`BandError::InvalidEdges`] unless `0 < create_factor < destroy_factor`.
    pub fn for_box(create_factor: f64, destroy_factor: f64) -> Result<OverlapBand, BandError> {
        OverlapBand::new(create_factor, destroy_factor)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;

    #[test]
    fn the_two_soi_functions_stay_distinct() {
        // Pin both to reference ranges so a refactor cannot collapse them
        // (generic_transfer.md §1.3 "a future refactor cannot collapse them").
        // Earth-Sun planet SOI: ~9.2e8 METERS.
        let planet = crate::celestial::planet_soi(1.496e11, 5.972e24, 1.989e30);
        assert!((8.0e8..1.1e9).contains(&planet));
        // Sun-like star (luminosity 1.0) system SOI: 300 GALAXY UNITS.
        let system = system_soi(1.0);
        assert!((system - 300.0).abs() < 1e-9);
        // Different formulas, units, and magnitudes by construction.
        assert!(planet / system > 1.0e5);
    }

    #[test]
    fn band_construction_validates_edges() {
        assert!(OverlapBand::new(10.0, 20.0).is_ok());
        assert_eq!(OverlapBand::new(20.0, 10.0), Err(BandError::InvalidEdges));
        assert_eq!(OverlapBand::new(10.0, 10.0), Err(BandError::InvalidEdges));
        assert_eq!(OverlapBand::new(0.0, 10.0), Err(BandError::InvalidEdges));
        assert_eq!(OverlapBand::new(-5.0, 10.0), Err(BandError::InvalidEdges));
    }

    #[test]
    fn class_band_factors_match_the_design_table() {
        let p = OverlapBand::for_planet_soi(1000.0);
        assert_eq!((p.create_below(), p.destroy_above()), (1150.0, 1300.0));
        let s = OverlapBand::for_system_soi(1000.0);
        assert_eq!((s.create_below(), s.destroy_above()), (950.0, 1050.0));
    }

    #[test]
    fn motion_band_is_velocity_safe_and_ordered() {
        // The interim motion-scaled band for an entity travelling 0.1 m/tick: edges are the
        // documented multiples of per-tick travel, ordered, and velocity-safe by construction
        // (a body that cannot skip the gap in one tick cannot tunnel the band).
        let travel = 0.1;
        let band = OverlapBand::for_motion(travel);
        assert_eq!(
            (band.create_below(), band.destroy_above()),
            (
                travel * MOTION_BAND_CREATE_TRAVELS,
                travel * MOTION_BAND_DESTROY_TRAVELS,
            ),
        );
        // Constructible via the validating `new` (edges satisfy 0 < create < destroy).
        assert!(OverlapBand::new(band.create_below(), band.destroy_above()).is_ok());
        // Velocity-safe at exactly the travel that produced it (dt folded into `travel`, so the
        // per-tick step is `travel` and the safety check uses dt = 1 tick).
        assert!(
            band.width_safe_for(travel, 1.0),
            "the motion band tolerates the velocity it was sized for"
        );
        // A body starting AT the boundary (distance 0, a member) stays a member until it has
        // travelled past the destroy edge — the post-handoff teardown guarantee.
        assert!(band.update_membership(true, band.destroy_above()));
        assert!(!band.update_membership(true, band.destroy_above() + travel));
    }

    #[test]
    fn hysteresis_prevents_flapping() {
        let band = OverlapBand::new(100.0, 130.0).expect("band");
        // Not a member: must come inside the CREATE edge.
        assert!(!band.update_membership(false, 115.0));
        assert!(band.update_membership(false, 99.0));
        // A member: stays until beyond the DESTROY edge.
        assert!(band.update_membership(true, 115.0));
        assert!(band.update_membership(true, 130.0));
        assert!(!band.update_membership(true, 130.1));
    }

    #[test]
    fn width_safety_scales_with_velocity() {
        let band = OverlapBand::new(100.0, 130.0).expect("band"); // gap 30
        // 30 >= v * 0.05s * 2  =>  safe up to v = 300 units/s.
        assert!(band.width_safe_for(300.0, 0.05));
        assert!(!band.width_safe_for(301.0, 0.05));
        assert!(
            band.width_safe_for(-300.0, 0.05),
            "speed is direction-agnostic"
        );
    }

    #[test]
    fn shell_crossing_classifies_all_cases() {
        let r = 10.0;
        let far = DVec3::new(20.0, 0.0, 0.0);
        let inside = DVec3::new(5.0, 0.0, 0.0);
        assert_eq!(
            segment_shell_crossing(far, DVec3::new(25.0, 0.0, 0.0), r),
            ShellCrossing::StaysOutside
        );
        assert_eq!(
            segment_shell_crossing(inside, DVec3::new(-5.0, 0.0, 0.0), r),
            ShellCrossing::StaysInside
        );
        assert_eq!(
            segment_shell_crossing(far, inside, r),
            ShellCrossing::Inward
        );
        assert_eq!(
            segment_shell_crossing(inside, far, r),
            ShellCrossing::Outward
        );
    }

    #[test]
    fn tunneling_is_detected() {
        // A 5.5 km/tick body: from (+x far) to (-x far) straight through a 10-unit
        // shell — both endpoints outside, but the sweep dips through.
        let r = 10.0;
        let a = DVec3::new(2750.0, 1.0, 0.0);
        let b = DVec3::new(-2750.0, 1.0, 0.0);
        assert_eq!(
            segment_shell_crossing(a, b, r),
            ShellCrossing::ThroughAndBack
        );
        // A grazing pass outside the shell is NOT a crossing.
        let graze_a = DVec3::new(2750.0, 11.0, 0.0);
        let graze_b = DVec3::new(-2750.0, 11.0, 0.0);
        assert_eq!(
            segment_shell_crossing(graze_a, graze_b, r),
            ShellCrossing::StaysOutside
        );
    }

    #[test]
    fn degenerate_zero_length_segment_outside_is_outside() {
        let p = DVec3::new(15.0, 0.0, 0.0);
        assert_eq!(
            segment_shell_crossing(p, p, 10.0),
            ShellCrossing::StaysOutside
        );
    }

    proptest! {
        /// The point-sample classification and the swept classification agree on
        /// endpoints; the sweep only ever ADDS the ThroughAndBack case.
        #[test]
        fn sweep_is_consistent_with_endpoints(
            ax in -100.0f64..100.0, ay in -100.0f64..100.0,
            bx in -100.0f64..100.0, by in -100.0f64..100.0,
            r in 1.0f64..50.0,
        ) {
            let a = DVec3::new(ax, ay, 0.0);
            let b = DVec3::new(bx, by, 0.0);
            let got = segment_shell_crossing(a, b, r);
            let (ain, bin) = (a.length() <= r, b.length() <= r);
            match (ain, bin) {
                (true, true) => prop_assert_eq!(got, ShellCrossing::StaysInside),
                (true, false) => prop_assert_eq!(got, ShellCrossing::Outward),
                (false, true) => prop_assert_eq!(got, ShellCrossing::Inward),
                (false, false) => prop_assert!(
                    got == ShellCrossing::StaysOutside || got == ShellCrossing::ThroughAndBack
                ),
            }
        }

        /// Hysteresis membership is monotone in distance for each prior state.
        #[test]
        fn membership_is_monotone(d1 in 0.0f64..200.0, d2 in 0.0f64..200.0) {
            let band = OverlapBand::new(100.0, 130.0).expect("band");
            let (near, far) = if d1 <= d2 { (d1, d2) } else { (d2, d1) };
            for was in [false, true] {
                // If the farther distance is a member, the nearer one must be too.
                if band.update_membership(was, far) {
                    prop_assert!(band.update_membership(was, near));
                }
            }
        }
    }

    // -----------------------------------------------------------------------
    // Slice 1 spatial-boundary tests.
    // -----------------------------------------------------------------------

    const UNIT_HALF: DVec3 = DVec3::new(1.0, 1.0, 1.0);

    #[test]
    fn segment_aabb_crossing_classifies_all_five_cases() {
        let half = DVec3::new(10.0, 10.0, 10.0);
        // StaysInside: both endpoints strictly inside.
        assert_eq!(
            segment_aabb_crossing(DVec3::new(1.0, 2.0, 3.0), DVec3::new(-4.0, 5.0, -6.0), half),
            ShellCrossing::StaysInside
        );
        // StaysOutside: both endpoints outside, no dip through.
        assert_eq!(
            segment_aabb_crossing(
                DVec3::new(20.0, 20.0, 0.0),
                DVec3::new(30.0, 20.0, 0.0),
                half
            ),
            ShellCrossing::StaysOutside
        );
        // Inward: outside -> inside.
        assert_eq!(
            segment_aabb_crossing(DVec3::new(20.0, 0.0, 0.0), DVec3::new(5.0, 0.0, 0.0), half),
            ShellCrossing::Inward
        );
        // Outward: inside -> outside.
        assert_eq!(
            segment_aabb_crossing(DVec3::new(5.0, 0.0, 0.0), DVec3::new(20.0, 0.0, 0.0), half),
            ShellCrossing::Outward
        );
        // ThroughAndBack: both outside, passes straight through.
        assert_eq!(
            segment_aabb_crossing(
                DVec3::new(-20.0, 0.0, 0.0),
                DVec3::new(20.0, 0.0, 0.0),
                half
            ),
            ShellCrossing::ThroughAndBack
        );
    }

    #[test]
    fn segment_hits_aabb_covers_every_branch() {
        let half = UNIT_HALF;
        // d == 0 on an axis AND origin below the slab (o[i] < -h[i]) => false (the
        // `o[i] < -h[i]` disjunct fires). Segment moves purely in +y at x = -5 (< -1).
        assert!(!segment_hits_aabb(
            DVec3::new(-5.0, -10.0, 0.0),
            DVec3::new(-5.0, 10.0, 0.0),
            half
        ));
        // d == 0 on an axis AND origin above the slab (o[i] > h[i]) => false (the
        // `o[i] > h[i]` disjunct fires; the first disjunct is false so short-circuit reaches it).
        assert!(!segment_hits_aabb(
            DVec3::new(5.0, -10.0, 0.0),
            DVec3::new(5.0, 10.0, 0.0),
            half
        ));
        // d == 0 on an axis but origin INSIDE that slab (no early return): the segment runs
        // along +y through x = 0 (inside the x-slab), so the parallel branch takes neither
        // return and the box is genuinely hit.
        assert!(segment_hits_aabb(
            DVec3::new(0.0, -10.0, 0.0),
            DVec3::new(0.0, 10.0, 0.0),
            half
        ));
        // Needs the t1 > t2 swap: moving in the -x direction makes the near/far plane hits
        // arrive in reverse t order, so `(lo, hi)` must swap them. A hit confirms the swap.
        assert!(segment_hits_aabb(
            DVec3::new(10.0, 0.0, 0.0),
            DVec3::new(-10.0, 0.0, 0.0),
            half
        ));
        // tmin > tmax miss: passes the +x slab test but is displaced far in +y so the y-slab
        // interval never overlaps the x-slab interval => the `tmin > tmax` return fires.
        assert!(!segment_hits_aabb(
            DVec3::new(-10.0, 5.0, 0.0),
            DVec3::new(10.0, 5.0, 0.0),
            half
        ));
        // A genuine through hit exercising the slab arithmetic on all three axes and the
        // final `true`.
        assert!(segment_hits_aabb(
            DVec3::new(-10.0, -0.5, 0.3),
            DVec3::new(10.0, 0.5, -0.3),
            half
        ));
    }

    #[test]
    fn aabb_contains_counts_the_surface_as_inside() {
        let half = DVec3::new(2.0, 3.0, 4.0);
        // Strictly inside.
        assert!(aabb_contains(DVec3::new(1.0, -1.0, 0.0), half));
        // Exactly on a face (== the half-extent) is inside via `<=`.
        assert!(aabb_contains(DVec3::new(2.0, 0.0, 0.0), half));
        assert!(aabb_contains(DVec3::new(-2.0, -3.0, 4.0), half));
        // Just outside one axis is not contained.
        assert!(!aabb_contains(DVec3::new(2.0001, 0.0, 0.0), half));
    }

    #[test]
    fn tunneling_is_detected_box() {
        // A fast body from far +x to far -x straight through a small box => ThroughAndBack.
        let half = DVec3::new(0.5, 0.5, 0.5);
        assert_eq!(
            segment_aabb_crossing(
                DVec3::new(1000.0, 0.0, 0.0),
                DVec3::new(-1000.0, 0.0, 0.0),
                half
            ),
            ShellCrossing::ThroughAndBack
        );
        // A grazing pass OUTSIDE the box (offset in +y beyond the half-extent) => StaysOutside.
        assert_eq!(
            segment_aabb_crossing(
                DVec3::new(1000.0, 1.0, 0.0),
                DVec3::new(-1000.0, 1.0, 0.0),
                half
            ),
            ShellCrossing::StaysOutside
        );
    }

    #[test]
    fn boundary_swept_dispatches_each_shape() {
        // Shell arm is segment_shell_crossing verbatim.
        assert_eq!(
            Boundary::Shell { r: 10.0 }
                .swept(DVec3::new(20.0, 0.0, 0.0), DVec3::new(5.0, 0.0, 0.0)),
            ShellCrossing::Inward
        );
        // Aabb arm is segment_aabb_crossing.
        assert_eq!(
            Boundary::Aabb {
                half: DVec3::splat(10.0)
            }
            .swept(DVec3::new(5.0, 0.0, 0.0), DVec3::new(20.0, 0.0, 0.0)),
            ShellCrossing::Outward
        );
        // Obb arm rotates into box-local: a 90deg-about-z box, segment along world +y hits the
        // (rotated) box exactly as an unrotated x-sweep would.
        let orient = DQuat::from_rotation_z(std::f64::consts::FRAC_PI_2);
        assert_eq!(
            Boundary::Obb {
                half: DVec3::new(10.0, 1.0, 1.0),
                orient
            }
            .swept(DVec3::new(0.0, -20.0, 0.0), DVec3::new(0.0, 20.0, 0.0)),
            ShellCrossing::ThroughAndBack
        );
    }

    #[test]
    fn membership_scalar_per_shape() {
        // Shell => |p|.
        let p = DVec3::new(3.0, 4.0, 0.0);
        assert_eq!(Boundary::Shell { r: 99.0 }.membership_scalar(p), 5.0);
        // Aabb => chebyshev_norm: (|p| / half).max_element.
        let aabb = Boundary::Aabb {
            half: DVec3::new(2.0, 8.0, 4.0),
        };
        // |3|/2 = 1.5, |4|/8 = 0.5, |0|/4 = 0 => 1.5.
        assert_eq!(aabb.membership_scalar(p), 1.5);
        // Obb => chebyshev_norm in the inverse-rotated frame. Identity orient equals the Aabb.
        let obb_id = Boundary::Obb {
            half: DVec3::new(2.0, 8.0, 4.0),
            orient: DQuat::IDENTITY,
        };
        assert_eq!(obb_id.membership_scalar(p), 1.5);
    }

    #[test]
    fn signed_distance_shell_aabb_obb() {
        // Shell: |p| - r. Inside negative, outside positive, ~0 on surface.
        let shell = Boundary::Shell { r: 10.0 };
        assert_eq!(shell.signed_distance(DVec3::new(4.0, 0.0, 0.0)), -6.0);
        assert_eq!(shell.signed_distance(DVec3::new(13.0, 0.0, 0.0)), 3.0);
        assert!(shell.signed_distance(DVec3::new(10.0, 0.0, 0.0)).abs() < 1e-12);
        // Aabb: negative inside (least-negative axis), positive outside, ~0 on surface.
        let aabb = Boundary::Aabb {
            half: DVec3::new(2.0, 2.0, 2.0),
        };
        // Inside at (1,0,0): q = (-1,-2,-2), outside=0, inside=max_element=-1 => -1.
        assert_eq!(aabb.signed_distance(DVec3::new(1.0, 0.0, 0.0)), -1.0);
        // Outside along one axis at (5,0,0): q=(3,-2,-2), outside=3, inside=min(3,0)=0 => 3.
        assert_eq!(aabb.signed_distance(DVec3::new(5.0, 0.0, 0.0)), 3.0);
        // On the surface (2,0,0): q=(0,-2,-2), outside=0, inside=0 => 0.
        assert_eq!(aabb.signed_distance(DVec3::new(2.0, 0.0, 0.0)), 0.0);
        // Outside a corner (5,5,0): q=(3,3,-2), outside=sqrt(18), inside=0.
        assert!((aabb.signed_distance(DVec3::new(5.0, 5.0, 0.0)) - 18.0_f64.sqrt()).abs() < 1e-12);
        // Obb: same distance in the inverse-rotated frame. Rotate the query and the box together
        // (90deg about z) and the distance matches the Aabb at the un-rotated point.
        let orient = DQuat::from_rotation_z(std::f64::consts::FRAC_PI_2);
        let obb = Boundary::Obb {
            half: DVec3::new(2.0, 2.0, 2.0),
            orient,
        };
        let world_p = orient * DVec3::new(1.0, 0.0, 0.0);
        assert!((obb.signed_distance(world_p) - (-1.0)).abs() < 1e-12);
    }

    #[test]
    fn for_soi_velocity_safe_widens_destroy_when_velocity_demands() {
        // Slow body: velocity pad is below the raw destroy_factor edge, so destroy stays at
        // r_soi * destroy_factor (the factor edge wins the max).
        let slow = OverlapBand::for_soi_velocity_safe(
            1000.0, // r_soi
            1.15,   // create_factor
            1.30,   // destroy_factor
            1.0,    // v_rel (slow)
            0.05,   // dt
            0.5,    // pad_floor
            1.0,    // k_safety_extra
        );
        assert_eq!(slow.create_below(), 1150.0);
        assert_eq!(slow.destroy_above(), 1300.0);
        // Fast body: |v| * dt * (K_SAFETY + extra) exceeds the factor gap, so the destroy edge is
        // widened to create + velocity_pad, strictly beyond the raw factor edge.
        let v = 5000.0;
        let dt = 0.05;
        let extra = 1.0;
        let fast = OverlapBand::for_soi_velocity_safe(1000.0, 1.15, 1.30, v, dt, 0.5, extra);
        assert_eq!(fast.create_below(), 1150.0);
        let expected_pad = v * dt * (K_SAFETY + extra);
        assert_eq!(fast.destroy_above(), 1150.0 + expected_pad);
        assert!(fast.destroy_above() > 1300.0);
        // The result is velocity-safe by construction.
        assert!(fast.width_safe_for(v, dt));
        // The pad_floor wins when both velocity and factor gap are tiny.
        let floored = OverlapBand::for_soi_velocity_safe(1.0, 1.15, 1.16, 0.0, 0.05, 0.5, 1.0);
        // factor destroy = 1.16; create = 1.15; create+floor = 1.65 => floor wins.
        assert_eq!(floored.destroy_above(), 1.15 + 0.5);
    }

    #[test]
    fn for_box_produces_the_given_factors_and_rejects_an_inverted_band() {
        let band = OverlapBand::for_box(1.15, 1.30).expect("ordered factors");
        assert_eq!(band.create_below(), 1.15);
        assert_eq!(band.destroy_above(), 1.30);
        // An inverted / equal / non-positive pair is REJECTED (an inverted band inverts the
        // hysteresis — a per-tick flap), routed through OverlapBand::new's validation.
        assert_eq!(
            OverlapBand::for_box(1.30, 1.15),
            Err(BandError::InvalidEdges)
        );
        assert_eq!(OverlapBand::for_box(1.2, 1.2), Err(BandError::InvalidEdges));
        assert_eq!(OverlapBand::for_box(0.0, 1.0), Err(BandError::InvalidEdges));
    }

    #[test]
    fn resolve_winner_empty_is_none() {
        assert_eq!(resolve_winner(&[]), None);
    }

    #[test]
    fn resolve_winner_innermost_max_depth_wins() {
        let a = (1u32, RealmId::Planet(10), Direction::Inward);
        let b = (5u32, RealmId::System(2), Direction::Outward);
        let c = (3u32, RealmId::Ship(ship_id()), Direction::Inward);
        // Depth 5 is innermost => b wins, carrying its direction.
        assert_eq!(
            resolve_winner(&[a, b, c]),
            Some((RealmId::System(2), Direction::Outward))
        );
    }

    #[test]
    fn resolve_winner_equal_depth_min_realm_wins() {
        // Same depth; RealmId's derived Ord orders Planet < System < Ship, and within Planet by
        // the u64. Planet(3) < Planet(9) < System(1), so Planet(3) (the min) wins.
        let a = (4u32, RealmId::Planet(9), Direction::Outward);
        let b = (4u32, RealmId::Planet(3), Direction::Inward);
        let c = (4u32, RealmId::System(1), Direction::Outward);
        assert_eq!(
            resolve_winner(&[a, b, c]),
            Some((RealmId::Planet(3), Direction::Inward))
        );
    }

    #[test]
    fn candidate_beats_covers_both_tie_break_arms() {
        let deep = (5u32, RealmId::Planet(0), Direction::Inward);
        let shallow = (2u32, RealmId::Planet(0), Direction::Inward);
        // Deeper beats shallower (the `a.0 > b.0` arm, true side).
        assert!(candidate_beats(deep, shallow));
        // Shallower does NOT beat deeper (the `a.0 > b.0` arm, false side).
        assert!(!candidate_beats(shallow, deep));
        // Equal depth, smaller realm beats larger (the `a.1 < b.1` arm, true side).
        let lo = (3u32, RealmId::Planet(1), Direction::Inward);
        let hi = (3u32, RealmId::Planet(2), Direction::Inward);
        assert!(candidate_beats(lo, hi));
        // Equal depth, larger realm does NOT beat smaller (the `a.1 < b.1` arm, false side).
        assert!(!candidate_beats(hi, lo));
        // Equal depth AND realm: Inward beats Outward (the `a.2 < b.2` final tiebreak, true side).
        let inw = (3u32, RealmId::Planet(1), Direction::Inward);
        let out = (3u32, RealmId::Planet(1), Direction::Outward);
        assert!(candidate_beats(inw, out));
        // ...and Outward does NOT beat Inward (the `a.2 < b.2` arm, false side).
        assert!(!candidate_beats(out, inw));
    }

    #[test]
    fn resolve_winner_is_total_even_for_same_depth_and_realm() {
        // The reachable collision the reviewer found: two boundary mouths into ONE realm at the
        // SAME depth, opposite directions. The winner MUST be slice-order-independent (Direction
        // is the final tiebreak: Inward < Outward), else the trigger is non-deterministic.
        let inw = (3u32, RealmId::Ship(ship_id()), Direction::Inward);
        let out = (3u32, RealmId::Ship(ship_id()), Direction::Outward);
        let want = Some((RealmId::Ship(ship_id()), Direction::Inward));
        assert_eq!(resolve_winner(&[inw, out]), want);
        assert_eq!(resolve_winner(&[out, inw]), want);
    }

    #[test]
    fn boundary_tuning_validate_truth_table() {
        assert_eq!(BoundaryTuning::DEFAULT.validate(), Ok(()));
        // n_entry == 0 => NonPositive.
        let mut t = BoundaryTuning::DEFAULT;
        t.n_entry = 0;
        assert_eq!(t.validate(), Err(BoundaryTuningError::NonPositive));
        // k_dwell == 0 => NonPositive.
        let mut t = BoundaryTuning::DEFAULT;
        t.k_dwell = 0;
        assert_eq!(t.validate(), Err(BoundaryTuningError::NonPositive));
        // velocity_pad_floor <= 0 => NonPositive (zero and negative).
        let mut t = BoundaryTuning::DEFAULT;
        t.velocity_pad_floor = 0.0;
        assert_eq!(t.validate(), Err(BoundaryTuningError::NonPositive));
        let mut t = BoundaryTuning::DEFAULT;
        t.velocity_pad_floor = -1.0;
        assert_eq!(t.validate(), Err(BoundaryTuningError::NonPositive));
        // cell_size_m <= 0 => NonPositive.
        let mut t = BoundaryTuning::DEFAULT;
        t.cell_size_m = 0.0;
        assert_eq!(t.validate(), Err(BoundaryTuningError::NonPositive));
        // k_safety_extra < 0 => KSafetyTooLow (the counts are all valid so this arm is reached).
        let mut t = BoundaryTuning::DEFAULT;
        t.k_safety_extra = -0.1;
        assert_eq!(t.validate(), Err(BoundaryTuningError::KSafetyTooLow));
        // k_safety_extra == 0 is allowed.
        let mut t = BoundaryTuning::DEFAULT;
        t.k_safety_extra = 0.0;
        assert_eq!(t.validate(), Ok(()));
    }

    #[test]
    fn chebyshev_norm_surface_inside_outside() {
        let half = DVec3::new(2.0, 4.0, 8.0);
        // On the surface of the tightest axis => exactly 1.0.
        assert_eq!(chebyshev_norm(DVec3::new(2.0, 0.0, 0.0), half), 1.0);
        // Strictly inside => < 1.
        assert!(chebyshev_norm(DVec3::new(1.0, 1.0, 1.0), half) < 1.0);
        // Outside => > 1.
        assert!(chebyshev_norm(DVec3::new(3.0, 0.0, 0.0), half) > 1.0);
        // A corner point at the surface of every axis is still 1.0 (uniform perimeter window).
        assert_eq!(chebyshev_norm(DVec3::new(2.0, 4.0, 8.0), half), 1.0);
    }

    #[test]
    fn spatial_types_serde_roundtrip() {
        // Each Boundary variant.
        for shape in [
            Boundary::Shell { r: 12.5 },
            Boundary::Aabb {
                half: DVec3::new(1.0, 2.0, 3.0),
            },
            Boundary::Obb {
                half: DVec3::new(4.0, 5.0, 6.0),
                orient: DQuat::from_rotation_x(0.3),
            },
        ] {
            let bytes = postcard::to_allocvec(&shape).expect("encode");
            let back: Boundary = postcard::from_bytes(&bytes).expect("decode");
            assert_eq!(back, shape);
        }
        // Direction.
        for d in [Direction::Inward, Direction::Outward] {
            let bytes = postcard::to_allocvec(&d).expect("encode");
            let back: Direction = postcard::from_bytes(&bytes).expect("decode");
            assert_eq!(back, d);
        }
        // CrossEffect.
        for e in [CrossEffect::Authority, CrossEffect::Interest] {
            let bytes = postcard::to_allocvec(&e).expect("encode");
            let back: CrossEffect = postcard::from_bytes(&bytes).expect("decode");
            assert_eq!(back, e);
        }
        // RealmBoundary (all fields Copy; parent Some and None both exercised).
        let rb = RealmBoundary {
            realm: RealmId::Planet(7),
            center: LatticePos::local(DVec3::new(100.0, 200.0, 300.0)),
            shape: Boundary::Aabb {
                half: DVec3::new(9.0, 9.0, 9.0),
            },
            band: OverlapBand::for_box(1.15, 1.30).expect("ordered factors"),
            parent: Some(RealmId::System(1)),
            to_realm: RealmId::Planet(7),
            effect: CrossEffect::Authority,
        };
        let bytes = postcard::to_allocvec(&rb).expect("encode");
        let back: RealmBoundary = postcard::from_bytes(&bytes).expect("decode");
        assert_eq!(back, rb);
        let rb_top = RealmBoundary {
            parent: None,
            effect: CrossEffect::Interest,
            ..rb
        };
        let bytes = postcard::to_allocvec(&rb_top).expect("encode");
        let back: RealmBoundary = postcard::from_bytes(&bytes).expect("decode");
        assert_eq!(back, rb_top);
    }

    fn ship_id() -> crate::ids::EntityId {
        crate::ids::EntityId::pack(crate::entity_kind::EntityKind::Ship, 3, 17, 0xABCDEF)
    }

    proptest! {
        /// BLOCKING: the shell membership scalar is BIT-IDENTICAL to `|p|` and drives the band
        /// to the identical result across the whole hysteresis window — the "hysteresis proven
        /// unchanged for shells" guarantee.
        #[test]
        fn shell_membership_scalar_equals_p_length_path(
            px in -500.0f64..500.0, py in -500.0f64..500.0, pz in -500.0f64..500.0,
            r in 1.0f64..300.0,
        ) {
            let p = DVec3::new(px, py, pz);
            let shape = Boundary::Shell { r };
            let scalar = shape.membership_scalar(p);
            // Bit-identical to the raw band input used today.
            prop_assert_eq!(scalar.to_bits(), p.length().to_bits());
            // And it drives update_membership identically across the whole band, for both prior
            // states and a create/destroy edge pair derived from r.
            let band = OverlapBand::for_planet_soi(r);
            for was in [false, true] {
                prop_assert_eq!(
                    band.update_membership(was, scalar),
                    band.update_membership(was, p.length())
                );
            }
        }

        /// BLOCKING: the box swept crossing agrees with endpoint containment (the sweep only ever
        /// ADDS ThroughAndBack over the point-sample), mirroring the shell proptest.
        #[test]
        fn sweep_is_consistent_with_endpoints_box(
            ax in -100.0f64..100.0, ay in -100.0f64..100.0, az in -100.0f64..100.0,
            bx in -100.0f64..100.0, by in -100.0f64..100.0, bz in -100.0f64..100.0,
            hx in 1.0f64..50.0, hy in 1.0f64..50.0, hz in 1.0f64..50.0,
        ) {
            let a = DVec3::new(ax, ay, az);
            let b = DVec3::new(bx, by, bz);
            let half = DVec3::new(hx, hy, hz);
            let got = segment_aabb_crossing(a, b, half);
            let ain = aabb_contains(a, half);
            let bin = aabb_contains(b, half);
            match (ain, bin) {
                (true, true) => prop_assert_eq!(got, ShellCrossing::StaysInside),
                (true, false) => prop_assert_eq!(got, ShellCrossing::Outward),
                (false, true) => prop_assert_eq!(got, ShellCrossing::Inward),
                (false, false) => prop_assert!(
                    got == ShellCrossing::StaysOutside || got == ShellCrossing::ThroughAndBack
                ),
            }
        }

        /// BLOCKING: a velocity-safe box band's create->destroy gap admits the per-tick travel on a
        /// CORNER (diagonal) approach, not merely a face — because the normalized-Chebyshev level
        /// sets are scaled boxes, the ABSOLUTE minimal thickness (at the nearest face, the tightest
        /// axis) still exceeds v*dt*K. We assert the minimal-axis absolute band thickness beats the
        /// per-tick corner travel.
        #[test]
        fn corner_hysteresis_velocity_safe(
            hx in 1.0f64..50.0, hy in 1.0f64..50.0, hz in 1.0f64..50.0,
            speed in 0.1f64..20.0, dt in 0.001f64..0.1,
        ) {
            let half = DVec3::new(hx, hy, hz);
            let create_factor = 1.15;
            // Size the destroy factor so the ABSOLUTE thickness on the tightest axis covers the
            // per-tick corner travel with the K_SAFETY margin. The tightest axis has the smallest
            // half-extent, so scale the factor gap off it.
            let min_half = half.min_element();
            let per_tick = speed * dt;
            // Required normalized gap so that gap * min_half >= per_tick * K_SAFETY.
            let needed_gap = (per_tick * K_SAFETY) / min_half;
            let destroy_factor = create_factor + needed_gap + 0.01; // strict margin
            // destroy_factor > create_factor > 0 by construction ⇒ always Ok.
            let band = OverlapBand::for_box(create_factor, destroy_factor).expect("ordered gap");
            // The absolute band thickness at the tightest face (worst case for a corner approach).
            let abs_thickness = (band.destroy_above() - band.create_below()) * min_half;
            // A diagonal (corner) step of length per_tick still fits inside the band with margin.
            prop_assert!(abs_thickness >= per_tick * K_SAFETY);
        }

        /// BLOCKING: the Obb swept classifier is EXACTLY the Aabb classifier on the inverse-rotated
        /// segment (the shim is exact, not approximate).
        #[test]
        fn obb_rotate_to_aabb_equivalence(
            ax in -50.0f64..50.0, ay in -50.0f64..50.0, az in -50.0f64..50.0,
            bx in -50.0f64..50.0, by in -50.0f64..50.0, bz in -50.0f64..50.0,
            hx in 1.0f64..30.0, hy in 1.0f64..30.0, hz in 1.0f64..30.0,
            rx in -3.0f64..3.0, ry in -3.0f64..3.0, rz in -3.0f64..3.0,
        ) {
            let half = DVec3::new(hx, hy, hz);
            let orient = DQuat::from_euler(glam::EulerRot::XYZ, rx, ry, rz);
            let p0 = DVec3::new(ax, ay, az);
            let p1 = DVec3::new(bx, by, bz);
            let obb = Boundary::Obb { half, orient };
            let aabb = Boundary::Aabb { half };
            let inv = orient.inverse();
            prop_assert_eq!(obb.swept(p0, p1), aabb.swept(inv * p0, inv * p1));
        }

        /// BLOCKING: shuffling the candidate slice yields the SAME winner (the total order makes
        /// resolve_winner permutation-invariant).
        #[test]
        fn resolve_winner_permutation_invariant(
            // NARROW depth/seed ranges so equal-(depth,realm) collisions are FREQUENT, and
            // direction is INDEPENDENT of depth (not depth-parity) so the same-(depth,realm)-
            // opposite-direction case — the one that would break a non-total order — actually
            // occurs in the generated inputs (the prior generator structurally excluded it).
            depths in prop::collection::vec(0u32..3, 1..6),
            seeds in prop::collection::vec(0u64..2, 1..6),
            kinds in prop::collection::vec(0u8..3, 1..6),
            dirs in prop::collection::vec(0u8..2, 1..6),
        ) {
            let n = depths.len().min(seeds.len()).min(kinds.len()).min(dirs.len());
            let mut cands: Vec<(u32, RealmId, Direction)> = Vec::new();
            for i in 0..n {
                let realm = match kinds[i] {
                    0 => RealmId::Planet(seeds[i]),
                    1 => RealmId::System(seeds[i]),
                    _ => RealmId::Ship(ship_id()),
                };
                let dir = if dirs[i] == 0 { Direction::Inward } else { Direction::Outward };
                cands.push((depths[i], realm, dir));
            }
            let baseline = resolve_winner(&cands);
            // A rotation of the slice is a permutation; the winner must not change.
            let mut rotated = cands.clone();
            rotated.rotate_left(1);
            prop_assert_eq!(resolve_winner(&rotated), baseline);
            // A reversal is another permutation.
            let mut reversed = cands.clone();
            reversed.reverse();
            prop_assert_eq!(resolve_winner(&reversed), baseline);
        }
    }
}
