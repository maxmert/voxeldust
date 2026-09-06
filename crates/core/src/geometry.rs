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
//!   segment against the shell, so a fast body cannot tunnel undetected. TRUE FOR `Shell` AND
//!   `Aabb` since S5, on exact integers ([`region_verdict`]). What it does NOT yet cover, named
//!   here so the sentence above is not read wider than it is:
//!   - **`Obb` is not swept.** A rotated box needs a rotation, and no rotation of an integer
//!     lattice vector is an integer lattice vector, so its swept form would decide on a
//!     square-cornered box while its point form decides on a rounded-corner distance field — a
//!     metre-scale disagreement between the two halves of one verdict. No shipped region is an
//!     `Obb`. Ledgered.
//!   - **A MOVING REGION sweeping over a PARKED subject is not covered.** Both endpoints are
//!     reframed through THIS tick's placement book, so the segment tested is the subject's own
//!     displacement, not the relative one. Closing it needs the prior endpoint folded through the
//!     PREVIOUS tick's book. Ledgered.
//!   - **One-tick fly-through is detected but not damped.** Acquiring a child and leaving it
//!     within one tick now issues a re-home the next tick reverses. That needs bands sized from
//!     real closing speed, which is the next slice; the condition under which it becomes
//!     reachable is pinned by a test rather than argued.
//!
//! Also home of [`system_soi`] — the LUMINOSITY-based, GALAXY-UNIT star-system SOI,
//! deliberately distinct from the mass-ratio, METERS [`crate::celestial::planet_soi`].

use glam::{DQuat, DVec3, I64Vec3};
use serde::{Deserialize, Serialize};

use crate::frame::{FrameError, transfer_frame};
use crate::placement::PlacementBook;
use crate::pose::{CELL_DOMAIN_MAX, FrameRef, LatticePos, RealmId, Separation, StampedPose, Tier};

/// Base star-system SOI radius in galaxy units (ported from the reference repo's
/// `galaxy.rs`; part of the galaxy-scale definition, not a tunable).
pub const BASE_SOI_RADIUS: f64 = 100.0;
/// SOI growth per unit of stellar luminosity, galaxy units.
pub const SOI_LUMINOSITY_SCALE: f64 = 200.0;

/// THE BAND A BOUNDARY NEEDS so that one step of travel at `v_mps` cannot carry a subject across it
/// unseen: `v · dt · ticks`.
///
/// Half of ONE solve, and the owner's own words for it (2026-08-24): *band width and top speed are two
/// readings of one solve*. A faster warp must be paid for with wider bands, and this is where that price
/// is stated rather than argued.
///
/// `ticks` is how many steps the subject must be observed INSIDE the band — one step is not enough,
/// because a subject sampled exactly on the boundary is a subject whose crossing nobody saw.
///
/// Branchless and monomorphic (HR5).
#[must_use]
pub fn band_for_speed(v_mps: f64, dt_s: f64, ticks: f64) -> f64 {
    v_mps.abs() * dt_s * ticks
}

/// THE FASTEST SPEED A BAND CAN CONTAIN — the inverse of [`band_for_speed`], and the shape the owner's
/// ruling takes on the movement side: *a realm's maximum speed is the speed at which one tick of travel
/// still fits inside the thinnest band in that realm*.
///
/// FAILS CLOSED. A window of zero time (`dt·ticks <= 0`) is a nonsense configuration, and the two
/// answers available are "any speed is safe" and "no speed is safe". It returns the second, because a
/// safety number that reads as infinity on a misconfiguration is how a misconfiguration becomes a ship
/// passing through a moon.
#[must_use]
pub fn speed_for_band(width_m: f64, dt_s: f64, ticks: f64) -> f64 {
    let window_s = dt_s * ticks;
    if window_s <= 0.0 {
        return 0.0;
    }
    width_m / window_s
}

/// HOW MANY COORDINATE CELLS THE THINNEST BAND MUST BE, so that the sub-cell residual the integer
/// verdict discards can never decide a membership. Three orders of magnitude, stated as the power of
/// two the lattice is actually built on rather than as a round decimal.
///
/// This is the fence [`region_verdict`] has always cited by name and which never existed: its doc says
/// *"`guard_quantum_band`-class fences keep every band ≥ 3 orders above"* the cell quantum. The claim
/// was true by 3.49 orders at the millimetre tier and NOTHING enforced it, so it would have gone
/// silently false the day a coarser tier lit up. [`guard_quantum_band`] now enforces it.
pub const QUANTUM_BAND_CELLS: f64 = 1024.0;

/// A band that is too thin for the grid it is measured on — the fence's refusal, carrying both sides
/// so the failure names itself rather than needing a debugger.
#[derive(Clone, Copy, Debug, PartialEq, thiserror::Error)]
#[error(
    "a band of {width_m} m is {cells} cells at this tier; every band must be at least \
     {QUANTUM_BAND_CELLS} cells wide, or the sub-cell residual the integer verdict discards could \
     decide a membership"
)]
pub struct BandTooThin {
    /// The offending band's total width, metres.
    pub width_m: f64,
    /// That width in whole coordinate cells at the tier it is measured on.
    pub cells: f64,
}

/// THE QUANTUM FENCE, finally written. Refuse a band thinner than [`QUANTUM_BAND_CELLS`] cells at its
/// own tier.
///
/// WHY IT MATTERS AND WHY IT IS NOT DECORATION. The containment verdict decides on whole cells and
/// DISCARDS the sub-cell remainder. That is safe only while a band is enormous compared to one cell —
/// otherwise the discarded remainder is a meaningful fraction of the band, and which side of a boundary
/// a subject is on becomes a question about rounding. Today a 3 m band is 3,072 cells, so the claim
/// holds with room to spare; at a coarser tier the same 3 m would be a fraction of one cell and the
/// claim would be false with nothing to say so.
///
/// # Errors
/// [`BandTooThin`] when the width is under the required cell count, including a zero or negative width
/// (which is thinner than anything and must never read as acceptable).
pub fn guard_quantum_band(width_m: f64, tier: Tier) -> Result<(), BandTooThin> {
    let cells = width_m / tier.cell_edge_m();
    if cells >= QUANTUM_BAND_CELLS {
        Ok(())
    } else {
        Err(BandTooThin { width_m, cells })
    }
}

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
///
/// ★ NOT THE CONTAINMENT ANSWER, AND IT MUST NOT BECOME ONE. Membership is decided on INTEGER CELLS
/// inside [`region_verdict`], so that it is exact at every magnitude and identical on every host.
/// This decimal form survives as an ORACLE and as a classifier for consumers that already work in
/// metres. It has **zero production callers**, deliberately.
///
/// **ITS VALIDITY WINDOW, because comparing the two is the obvious thing to try and it has a trap.**
/// Metres are exact only while a separation stays within `2⁵³` cells — about 8.8e12 m, roughly 59 AU.
/// Beyond that the flatten rounds: one f64 ulp equals one whole Fine cell at 4.4e12 m, which is
/// INSIDE the home system, and the loss reaches half a metre at the full coordinate domain. So an
/// equality between this and the integer verdict PASSES on small fixtures and starts failing exactly
/// when a test first runs at real scale — with the ORACLE at fault, not the implementation. Compare
/// them only inside the window, and say which window in the test.
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
/// `Ord` (`Inward < Outward`, declaration order) was the FINAL tiebreak in the retired
/// directional winner resolver, keeping its total order strict even when two boundaries
/// shared a realm at one depth. Retained (with `RealmBoundary`) as a swept-classifier
/// primitive; the containment model has no crossing direction.
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
    /// The FURTHEST any point of this shape lies from its own centre (the circumscribed radius). Used to
    /// ask "how far does this child reach?" — deliberately the outer bound, so the nesting fence is
    /// conservative: a child is rejected if ANY part of it could poke outside its parent.
    #[must_use]
    pub fn circumscribed_extent(&self) -> f64 {
        match self {
            Boundary::Shell { r } => *r,
            // A box's far corner. Orientation cannot change the distance to the corner, so Obb reuses it.
            Boundary::Aabb { half } | Boundary::Obb { half, .. } => half.length(),
        }
    }

    /// The FURTHEST any point of this shape lies from a DIFFERENT origin, when the shape's own centre sits
    /// at `offset` from it. The exact answer, not a bound: a sphere reaches `|offset| + r`; a box reaches
    /// its farthest CORNER, which is what actually decides whether it pokes out of its parent.
    ///
    /// WHY EXACT AND NOT `|offset| + circumscribed_extent()`. That sum treats a box as the sphere around it,
    /// which over-states a cube's reach by up to 73%. The nesting fence used it, and on the shipped walk
    /// forest it condemned an area that is entirely inside its planet: the area's box spans x∈[2,8] with
    /// half-extents 3 in a planet of radius 10, so its farthest corner is 9.06 m out, while the spherical
    /// bound reported 10.20 and called it an escape. A fence that rejects correct worlds gets widened or
    /// switched off; an exact one can stay armed.
    #[must_use]
    pub fn max_reach_from(&self, offset: DVec3) -> f64 {
        match self {
            Boundary::Shell { r } => offset.length() + r,
            Boundary::Aabb { half } => farthest_corner(offset, *half, DQuat::IDENTITY),
            Boundary::Obb { half, orient } => farthest_corner(offset, *half, *orient),
        }
    }

    /// The NEAREST any surface point lies to this shape's centre (the inscribed radius). Used to ask "how
    /// much interior can this parent guarantee?" — the inner bound, again so the fence is conservative:
    /// the parent only promises the sphere it fully contains.
    #[must_use]
    pub fn inscribed_extent(&self) -> f64 {
        match self {
            Boundary::Shell { r } => *r,
            Boundary::Aabb { half } | Boundary::Obb { half, .. } => half.min_element(),
        }
    }

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

    /// The largest linear extent of this shape from its center: the radius for a `Shell`, the max
    /// half-extent component for a box (`Aabb`/`Obb`). The ONE geometry-owned answer to "how big is
    /// this region", so the client's renderable-region filter (draw finite leaf realms, skip the
    /// ~unbounded ambient Galaxy/Universe shells — `worldgen::MAX_RENDERABLE_EXTENT_M`) never re-derives
    /// a per-shape size in a feature path. Monomorphic (the KIND branch lives here, not in the client).
    #[must_use]
    pub fn finite_extent(&self) -> f64 {
        match self {
            Boundary::Shell { r } => *r,
            Boundary::Aabb { half } | Boundary::Obb { half, .. } => half.max_element(),
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

/// The distance from the origin to the farthest of a box's eight corners, where the box has half-extents
/// `half`, is rotated by `orient`, and its centre sits at `offset`. Monomorphic and exhaustive — eight
/// corners, no closed form needed and none that stays exact under rotation. Boot-time only (the nesting
/// fence), so the eight-way loop is free.
#[must_use]
fn farthest_corner(offset: DVec3, half: DVec3, orient: DQuat) -> f64 {
    let mut max = 0.0_f64;
    for sx in [-1.0, 1.0] {
        for sy in [-1.0, 1.0] {
            for sz in [-1.0, 1.0] {
                let corner = offset + orient * (half * DVec3::new(sx, sy, sz));
                max = max.max(corner.length());
            }
        }
    }
    max
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

#[cfg(test)]
mod parent_centre_tests {
    use super::*;
    use crate::pose::{FrameRef, LatticePos, Tier};

    fn test_band() -> ContainmentBand {
        ContainmentBand::for_containment_velocity_safe(1.0, 2.0, 0.0, 1.0, 0.0)
            .expect("valid test band")
    }

    /// ★ THE TYPE COSTS THE WIRE NOTHING — MEASURED, not asserted from `#[serde(transparent)]`.
    ///
    /// A `regions.json` written before this type must still parse, and one written after must be
    /// byte-identical to what the bare `LatticePos` produced. `serde(transparent)` is what makes that
    /// true; this is what makes it CHECKED, because an attribute quietly removed is a silent on-disk
    /// format break, and the client and the dev cluster both read that file.
    #[test]
    fn a_parent_centre_is_wire_transparent() {
        for raw in [
            LatticePos::ORIGIN,
            LatticePos::local(DVec3::new(1.5, -2.5, 0.25)),
            LatticePos::at(I64Vec3::new(4096, -7, 0), DVec3::new(0.5, 0.0, 0.0)),
        ] {
            let bare = postcard::to_allocvec(&raw).expect("encode the bare position");
            let wrapped =
                postcard::to_allocvec(&ParentCentre::authored(raw)).expect("encode the centre");
            assert_eq!(bare, wrapped, "the wrapper must add no bytes");
            // …and it decodes from what the bare form wrote, which is the direction that matters for
            // a file already on disk.
            let back: ParentCentre =
                postcard::from_bytes(&bare).expect("decode from the bare bytes");
            assert_eq!(back, ParentCentre::authored(raw));
        }
    }

    /// ★ THE FENCE ITSELF: the step comes off the PARENT, so the child's cannot be substituted.
    ///
    /// This is the whole point of the type, so it is measured on the exact shape that went wrong
    /// sixteen times — a child on the FINE rung inside a parent on the GALAXY rung. Read correctly the
    /// answer is 100 m; read with the child's step it would be 100/2048 m, a plausible number for a
    /// world that does not exist.
    #[test]
    fn a_centre_is_read_in_its_parents_step_and_the_childs_cannot_be_passed() {
        let parent = RealmRegion {
            realm: RealmId::Galaxy(1),
            center: ParentCentre::ORIGIN,
            frame: FrameRef::GalaxySpace { galaxy_seed: 1 },
            shape: Boundary::Shell { r: 1.0e6 },
            look: None,
            band: test_band(),
            aoi: AoiConfig::inert(),
            parent: None,
        };
        assert_eq!(parent.frame.tier(), Tier::Galaxy);
        let centre = ParentCentre::authored(LatticePos::from_metres(
            DVec3::new(100.0, 0.0, 0.0),
            Tier::Galaxy,
        ));
        assert_eq!(centre.metres_in(&parent), DVec3::new(100.0, 0.0, 0.0));
        // THE CONTROL, so "the fence works" is a measurement and not a claim about the compiler: the
        // same bits read at the CHILD's step give the wrong world, and the ratio is exactly the one
        // that hid in sixteen places.
        let wrong = centre
            .in_parents_frame()
            .delta_m(LatticePos::ORIGIN, Tier::Fine);
        assert_eq!(wrong, DVec3::new(100.0 / 2048.0, 0.0, 0.0));
        assert_eq!(
            centre.metres_in(&parent).x / wrong.x,
            2048.0,
            "the defect this type prevents is a factor of 2048"
        );
    }

    /// The forest lookup's three answers, each driven: the ambient root, a child whose parent is
    /// present, and a child whose parent is NOT — which refuses rather than guessing a step.
    #[test]
    fn the_forest_lookup_refuses_when_the_parent_is_absent() {
        let galaxy = RealmRegion {
            realm: RealmId::Galaxy(1),
            center: ParentCentre::ORIGIN,
            frame: FrameRef::GalaxySpace { galaxy_seed: 1 },
            shape: Boundary::Shell { r: 1.0e6 },
            look: None,
            band: test_band(),
            aoi: AoiConfig::inert(),
            parent: None,
        };
        let system = RealmRegion {
            realm: RealmId::System(7),
            center: ParentCentre::authored(LatticePos::from_metres(
                DVec3::new(100.0, 0.0, 0.0),
                Tier::Galaxy,
            )),
            frame: FrameRef::SystemSpace { system_seed: 7 },
            parent: Some(RealmId::Galaxy(1)),
            ..galaxy
        };
        // (a) the ambient root: the origin by definition, and no lookup is needed to say so.
        assert_eq!(galaxy.centre_m(&[]), Some(DVec3::ZERO));
        // (b) the parent is present ⇒ its step is used.
        assert_eq!(
            system.centre_m(&[galaxy, system]),
            Some(DVec3::new(100.0, 0.0, 0.0))
        );
        // (c) the parent is ABSENT ⇒ refused. The step is unknowable, and the fallback this replaces
        // (the child's own) is exactly how the defect was written down in shipped code.
        assert_eq!(system.centre_m(&[system]), None);
    }
}

impl RealmRegion {
    /// WHERE THIS REALM SITS INSIDE ITS PARENT, in metres — the one-line spelling for a caller holding
    /// the forest. See [`ParentCentre::metres_in_forest`] for what `None` means.
    #[must_use]
    pub fn centre_m(&self, forest: &[RealmRegion]) -> Option<DVec3> {
        self.center.metres_in_forest(self.parent, forest)
    }
}

/// WHERE A REALM SITS INSIDE ITS PARENT — a position whose unit is the PARENT's, and a type that will
/// not let you forget it.
///
/// ★ WHY THIS IS A TYPE AND NOT A `LatticePos` (slice S9). A lattice position is a cell count plus a
/// sub-cell residual; it means nothing without the STEP those cells are counted in. A realm's centre is
/// counted in its PARENT's step, while the realm's own `frame` names its OWN — and those were the same
/// step for the whole life of the project, right up until the galaxy got a coarser one.
///
/// When they stopped being the same, THIRTEEN readers were found flattening a centre with the child's
/// step. Every one of them was wrong by exactly 2048× (or 33,554,432× at the rung above), and NOT ONE OF
/// THEM FAILED: a wrong step does not produce an error, it produces a plausible number for a world that
/// does not exist. One was an ORACLE whose fixture used the same wrong step, so the two agreed with each
/// other perfectly. One WROTE A GOLDEN VECTOR, and would have recorded the galaxy's rows 2048× wrong as
/// the world's own reference bits.
///
/// Fixing thirteen sites by hand does not stop a fourteenth. So the bare `delta_m` is gone from this
/// path: the only way to metres is [`ParentCentre::metres_in`], which takes the PARENT REGION and reads
/// the step off it, so the child's step cannot be passed. `#[serde(transparent)]` — the on-disk bytes of
/// a `regions.json` are unchanged, which is measured rather than assumed
/// (`a_parent_centre_is_wire_transparent`).
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(transparent)]
pub struct ParentCentre(LatticePos);

impl ParentCentre {
    /// The origin — where a realm sits when its parent placed it at its own centre, and the only
    /// lawful centre for the ambient root.
    pub const ORIGIN: ParentCentre = ParentCentre(LatticePos::ORIGIN);

    /// State a centre THE PARENT AUTHORED. Named for the act, because under SL1 the parent is the only
    /// writer: a child that constructs its own centre is the defect this whole area exists to prevent.
    #[must_use]
    pub const fn authored(at: LatticePos) -> ParentCentre {
        ParentCentre(at)
    }

    /// This centre in METRES, in the parent's frame — the step read off the PARENT, never passed in.
    #[must_use]
    pub fn metres_in(self, parent: &RealmRegion) -> DVec3 {
        self.0.delta_m(LatticePos::ORIGIN, parent.frame.tier())
    }

    /// This centre in metres, found through a FOREST rather than a parent row already in hand.
    ///
    /// `None` when the parent is not in `forest` — the step is then unknowable, and the whole point of
    /// this type is that guessing it is the defect. A `None`-parented row is the ambient root, whose
    /// centre is the origin by definition, so it answers `Some(ZERO)`.
    #[must_use]
    pub fn metres_in_forest(
        self,
        parent: Option<RealmId>,
        forest: &[RealmRegion],
    ) -> Option<DVec3> {
        match parent {
            None => Some(DVec3::ZERO),
            Some(p) => forest
                .iter()
                .find(|r| r.realm == p)
                .map(|r| self.metres_in(r)),
        }
    }

    /// The raw lattice value, still counted in THE PARENT'S STEP.
    ///
    /// For the arithmetic that already holds the parent's rung — the crossing path subtracts this from a
    /// pose measured in the same frame — and for equality and storage. It is deliberately verbose: the
    /// moment you hold the number you are holding a count whose unit is not written down beside it, and
    /// pairing it with any other realm's step is the error this type exists to make hard.
    #[must_use]
    pub const fn in_parents_frame(self) -> LatticePos {
        self.0
    }
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

impl RealmBoundary {
    /// Build a SPHERICAL-SHELL realm boundary paired — BY CONSTRUCTION — with its ABSOLUTE
    /// velocity-safe SOI band. The shape is `Shell { r: r_soi }` and the band is
    /// [`OverlapBand::for_soi_velocity_safe`] (edges in the SAME METRE units as the shell radius),
    /// so a shell can NEVER be handed a normalized (unit-Chebyshev) box band — the shape↔band unit
    /// mismatch the Slice-1 reviewer flagged is UNCONSTRUCTIBLE here, not merely detected. The SOI
    /// constructor derives its edges from ordered constant factors and is infallible, so this is too.
    #[must_use]
    #[allow(clippy::too_many_arguments)]
    pub fn shell(
        realm: RealmId,
        center: LatticePos,
        r_soi: f64,
        create_factor: f64,
        destroy_factor: f64,
        v_rel: f64,
        dt: f64,
        pad_floor: f64,
        k_safety_extra: f64,
        parent: Option<RealmId>,
        to_realm: RealmId,
        effect: CrossEffect,
    ) -> RealmBoundary {
        RealmBoundary {
            realm,
            center,
            shape: Boundary::Shell { r: r_soi },
            band: OverlapBand::for_soi_velocity_safe(
                r_soi,
                create_factor,
                destroy_factor,
                v_rel,
                dt,
                pad_floor,
                k_safety_extra,
            ),
            parent,
            to_realm,
            effect,
        }
    }

    /// Build a BOX realm boundary — `Aabb` when `orient` is `None`, `Obb` when `Some` — paired BY
    /// CONSTRUCTION with its NORMALIZED (unit-Chebyshev) band. The band is [`OverlapBand::for_box`]
    /// (factors relative to the box surface = 1.0), so a box can NEVER be handed an absolute SOI band;
    /// together with [`RealmBoundary::shell`] this makes every shape↔band pairing correct by the ONLY
    /// two constructors that exist. `for_box` is fallible (it takes CALLER factors, and an inverted
    /// pair would invert the hysteresis into a per-tick flap), so this returns the error loudly rather
    /// than constructing a flapping band.
    ///
    /// # Errors
    /// [`BandError::InvalidEdges`] unless `0 < create_factor < destroy_factor`.
    #[must_use = "the Result carries a BandError that must not be dropped"]
    #[allow(clippy::too_many_arguments)]
    pub fn boxed(
        realm: RealmId,
        center: LatticePos,
        half: DVec3,
        orient: Option<DQuat>,
        create_factor: f64,
        destroy_factor: f64,
        parent: Option<RealmId>,
        to_realm: RealmId,
        effect: CrossEffect,
    ) -> Result<RealmBoundary, BandError> {
        let shape = match orient {
            Some(orient) => Boundary::Obb { half, orient },
            None => Boundary::Aabb { half },
        };
        Ok(RealmBoundary {
            realm,
            center,
            shape,
            band: OverlapBand::for_box(create_factor, destroy_factor)?,
            parent,
            to_realm,
            effect,
        })
    }

    /// Build an AXIS-ALIGNED BOX realm boundary paired — BY CONSTRUCTION — with its NORMALIZED-Chebyshev
    /// VELOCITY-SAFE band, the exact MIRROR of [`RealmBoundary::shell`] for a Cartesian station volume.
    /// The shape is `Aabb { half }` and the band is [`OverlapBand::for_box_velocity_safe`]; the two
    /// constructors share the SAME arg shape (`create_factor`, `destroy_factor`, `v_rel`, `dt`,
    /// `pad_floor`, `k_safety_extra`) so ONE crossing-feature fixture drives both a Spherical Shell and a
    /// Cartesian Aabb boundary (the D-38 G-IDENTICAL discharge). The only difference is the UNIT: the
    /// shell scales its band off `r_soi` (metres), the box off the Chebyshev surface (`1.0`), so
    /// `create = create_factor` and `destroy = destroy_factor` are the normalized band edges — the box's
    /// per-tick velocity travel is converted into the Chebyshev scale by the band ctor (÷ the tightest
    /// half-extent). Because a box `membership_scalar` is [`chebyshev_norm`] (1.0 on the surface), pairing
    /// an Aabb with the ABSOLUTE SOI band would be a unit mismatch — so, mirroring `boxed`, an Aabb can
    /// NEVER be handed the metre-scaled shell band, and (unlike the infallible `shell`) this returns the
    /// band error LOUDLY rather than constructing a flapping/inverted band.
    ///
    /// # Errors
    /// [`BandError::InvalidEdges`] unless the resulting normalized band satisfies `0 < create < destroy`
    /// (see [`OverlapBand::for_box_velocity_safe`]).
    #[must_use = "the Result carries a BandError that must not be dropped"]
    #[allow(clippy::too_many_arguments)]
    pub fn aabb(
        realm: RealmId,
        center: LatticePos,
        half: DVec3,
        create_factor: f64,
        destroy_factor: f64,
        v_rel: f64,
        dt: f64,
        pad_floor: f64,
        k_safety_extra: f64,
        parent: Option<RealmId>,
        to_realm: RealmId,
        effect: CrossEffect,
    ) -> Result<RealmBoundary, BandError> {
        Ok(RealmBoundary {
            realm,
            center,
            shape: Boundary::Aabb { half },
            band: OverlapBand::for_box_velocity_safe(
                create_factor,
                destroy_factor,
                v_rel,
                dt,
                half.min_element(),
                pad_floor,
                k_safety_extra,
            )?,
            parent,
            to_realm,
            effect,
        })
    }
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
}
// (`cell_size_m` is DELETED — real-scale addendum §A4.5/H-31: validated `> 0` and never read, a
// name collision with the live lattice, not a member of it.)

#[derive(Clone, Copy, Debug, PartialEq, Eq, thiserror::Error)]
pub enum BoundaryTuningError {
    #[error("n_entry, k_dwell, and velocity_pad_floor must all be > 0")]
    NonPositive,
    #[error("k_safety_extra must be >= 0 (it is added to K_SAFETY, never subtracted)")]
    KSafetyTooLow,
}

impl BoundaryTuning {
    /// Sane defaults: 3-tick entry, 5-tick dwell, 0.5 m velocity floor, +1.0 safety over
    /// [`K_SAFETY`].
    pub const DEFAULT: BoundaryTuning = BoundaryTuning {
        n_entry: 3,
        k_dwell: 5,
        velocity_pad_floor: 0.5,
        k_safety_extra: 1.0,
    };

    /// Reject a zero/negative count or scale ([`BoundaryTuningError::NonPositive`]) and a
    /// negative safety extra ([`BoundaryTuningError::KSafetyTooLow`]); `Ok` otherwise.
    pub fn validate(&self) -> Result<(), BoundaryTuningError> {
        if self.n_entry == 0 || self.k_dwell == 0 || self.velocity_pad_floor <= 0.0 {
            return Err(BoundaryTuningError::NonPositive);
        }
        if self.k_safety_extra < 0.0 {
            return Err(BoundaryTuningError::KSafetyTooLow);
        }
        Ok(())
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

    /// The NORMALIZED-Chebyshev analog of [`OverlapBand::for_soi_velocity_safe`] — the box band whose
    /// destroy edge is widened so the create → destroy gap stays velocity-safe on a CORNER (diagonal)
    /// approach, in the UNITLESS Chebyshev scale ([`chebyshev_norm`], 1.0 = box surface). It mirrors the
    /// SOI constructor arm-for-arm with the box's "unit" fixed at 1.0:
    /// - `create = create_factor` (the SOI's `r_soi · create_factor`, with `r_soi ≡ 1.0` — the surface).
    /// - The velocity pad is the SOI's `|v_rel|·dt·(K_SAFETY + k_safety_extra)` (a METRE per-tick travel)
    ///   converted INTO the Chebyshev scale by dividing by `min_half` (the tightest half-extent — the
    ///   worst-case face, where a metre of travel spends the MOST normalized band). The `pad_floor` is
    ///   already unitless (a normalized-gap floor), so it enters the `max` directly.
    /// - `destroy = max(destroy_factor, create + velocity_pad)` (the SOI's `max(r_soi · destroy_factor,
    ///   create + velocity_pad)` with `r_soi ≡ 1.0`).
    ///
    /// This is INFALLIBLE for the same reason the SOI constructor is: `create > 0` (a `> 0` factor) and
    /// `destroy = max(destroy_factor, create + non_negative_pad) > create` whenever `destroy_factor >
    /// create` OR the pad is positive — but a CALLER could pass `destroy_factor <= create_factor` with a
    /// zero pad, so it routes through [`OverlapBand::new`] and returns the error LOUDLY (an inverted band
    /// inverts the hysteresis into a per-tick flap, the exact failure this module prevents), exactly like
    /// [`OverlapBand::for_box`]. `min_half` is a precondition `> 0` (the box is non-degenerate; the
    /// [`RealmBoundary::aabb`] caller derives it from `half.min_element()`).
    ///
    /// # Errors
    /// [`BandError::InvalidEdges`] unless the resulting `0 < create < destroy` holds.
    pub fn for_box_velocity_safe(
        create_factor: f64,
        destroy_factor: f64,
        v_rel: f64,
        dt: f64,
        min_half: f64,
        pad_floor: f64,
        k_safety_extra: f64,
    ) -> Result<OverlapBand, BandError> {
        let create = create_factor;
        // The SOI's metre per-tick pad, converted to the Chebyshev scale (÷ the tightest half-extent).
        let normalized_travel = (v_rel.abs() * dt * (K_SAFETY + k_safety_extra)) / min_half;
        let velocity_pad = f64::max(pad_floor, normalized_travel);
        let destroy = f64::max(destroy_factor, create + velocity_pad);
        OverlapBand::new(create, destroy)
    }
}

// ---------------------------------------------------------------------------
// Containment realm-membership (task #135) — APPEND-ONLY pure geometry. NOTHING
// calls this yet (the detector wiring is C-3). It REPLACES the DIRECTIONAL portal
// model (`RealmBoundary { to_realm }` + `should_commit`'s asymmetric in/out) with
// CONTAINMENT: a realm is a REGION of space; an entity's realm is the DEEPEST region
// CONTAINING it; re-home on change; SYMMETRIC (escaping and entering are one rule).
// See scripts/containment_realm_membership_design.md.
// ---------------------------------------------------------------------------

/// A signed-distance hysteresis band (METRES, negative inside) for CONTAINMENT membership. A DISTINCT
/// type from [`OverlapBand`] so it can NEVER be fed the radial-analog [`Boundary::membership_scalar`]
/// (and `OverlapBand` can never be fed [`Boundary::signed_distance`]) — the shape↔band unit mismatch
/// stays UNCONSTRUCTIBLE, exactly as the portal bands enforce it. ONE band flavour serves every shape:
/// [`box_signed_distance`] is a true Euclidean SDF, so a metre dead-zone is uniform-thickness at box
/// faces AND corners automatically (no normalized-Chebyshev split). The dead-zone (`inset` inside ..
/// `outset` outside) straddles the surface, so a surface-hovering entity cannot flap.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct ContainmentBand {
    inset: f64,
    outset: f64,
}

impl ContainmentBand {
    /// The VELOCITY-SAFE constructor — the ONLY way to build one. Widens the outer edge so the
    /// dead-zone `inset + outset >= |v_rel|·dt·(K_SAFETY + k_safety_extra)`, mirroring
    /// [`OverlapBand::for_soi_velocity_safe`]: a body moving `|v_rel|·dt` per tick can neither skip
    /// the band nor tunnel a thin region between ticks. The `debug_assert!` pins the
    /// [`ContainmentBand::width_safe_for`] invariant (like the SOI/box ctors). Fallible: an inverted
    /// or degenerate pair would invert the hysteresis into a per-tick flap, so it is rejected LOUD.
    ///
    /// # Errors
    /// [`BandError::InvalidEdges`] unless both resolved edges are strictly positive.
    #[must_use = "the Result carries a BandError that must not be dropped"]
    pub fn for_containment_velocity_safe(
        inset: f64,
        outset_min: f64,
        v_rel: f64,
        dt: f64,
        k_safety_extra: f64,
    ) -> Result<ContainmentBand, BandError> {
        let need = v_rel.abs() * dt * (K_SAFETY + k_safety_extra);
        let outset = f64::max(outset_min, need - inset);
        if inset > 0.0 && outset > 0.0 {
            let band = ContainmentBand { inset, outset };
            debug_assert!(band.width_safe_for(v_rel, dt));
            Ok(band)
        } else {
            Err(BandError::InvalidEdges)
        }
    }

    /// Velocity-scaled width invariant: the dead-zone `inset + outset` exceeds the per-tick travel
    /// times [`K_SAFETY`], so a body at `v_rel` (units/s) over `dt_s`-second ticks cannot skip the band.
    #[must_use]
    pub fn width_safe_for(&self, v_rel: f64, dt_s: f64) -> bool {
        (self.inset + self.outset) >= v_rel.abs() * dt_s * K_SAFETY
    }

    /// The inner (acquire) edge — metres INSIDE the surface.
    #[must_use]
    pub fn inset(&self) -> f64 {
        self.inset
    }

    /// The outer (release) edge — metres OUTSIDE the surface.
    #[must_use]
    pub fn outset(&self) -> f64 {
        self.outset
    }

    /// Sign-correct hysteresis membership from a SIGNED DISTANCE (negative inside — the
    /// [`Boundary::signed_distance`] convention). Not yet a member: acquire when at least `inset`
    /// INSIDE (`signed_distance <= -inset`). Already a member: hold until more than `outset` OUTSIDE
    /// (`signed_distance <= outset` keeps it). The dead-zone `[-inset, +outset]` straddles the
    /// surface — mirrors [`OverlapBand::update_membership`]'s smaller-is-more-inside `<=` convention,
    /// in signed-metre units.
    #[must_use]
    pub fn member(&self, was_member: bool, signed_distance: f64) -> bool {
        if was_member {
            signed_distance <= self.outset
        } else {
            signed_distance <= -self.inset
        }
    }
}

/// THE ONE VISIBILITY FORMULA (look_horizon.md §3.3.2): the factor `cot(θ/2)` for a minimum
/// angular size `theta_rad` — a body of finite extent `e` subtends `≥ theta_rad` (is VISIBLE)
/// out to exactly `e · cot(θ/2)`. It lives HERE, in `vd-core`, with three consumers and no
/// second copy anywhere: the WORLD SOLVE (`vd-physics` sizes the galaxy shell so nothing two
/// levels down is visible from outside it), the BOOT MEASUREMENT (`measure_visibility_climb` —
/// how many levels each body's picture must travel, and the fence that refuses a world whose
/// climb exceeds the look carrier's arity), and the RUNTIME TRIPWIRE (the AoI interest band's
/// spin-up/tear-down factors — the live wake/sleep test every realm runs). One equation written
/// once is what makes the live bound and the world's own size calculation provably the same
/// thing. Straight-line, branchless (HR5).
#[must_use]
pub fn visibility_factor(theta_rad: f64) -> f64 {
    1.0 / (theta_rad / 2.0).tan()
}

/// ★ THE DOT ANGLE — the minimum angular size at which a body of finite extent is still a dot worth
/// waking for: 1.5 degrees, in radians. It lived in `vd-physics`' scale module while the world solve
/// was its only reader; the reach (owner ruling 2026-09-02 R3, built 2026-09-04) makes every realm a
/// reader — a realm states its own reach by size as `extent · cot(θ/2)` with THIS θ, and the world
/// solve sizes its shells with the same one, so the two can never disagree. A single knob, and its
/// cost is stated where it was: every wake radius scales with it.
pub const VISIBILITY_THETA_MIN_RAD: f64 = 0.026_180;

/// The visibility REACH of a body of extent `extent_m` under minimum angle `theta_rad`: the
/// distance out to which it is still visible — `extent · cot(θ/2)`, the same formula's other
/// spelling ([`visibility_factor`]).
#[must_use]
pub fn visibility_reach_m(extent_m: f64, theta_rad: f64) -> f64 {
    extent_m * visibility_factor(theta_rad)
}

/// ★ THE REFERENCE VIEW (owner 2026-09-06, "use the drawable floor"): the vertical field of view and
/// the row count every "how big is one pixel" question is answered at — the capture camera's 45° over
/// 720 rows. ONE home: the renderer's capture height and the harness's fitted field of view read
/// these, and so does a shard deciding whether an occupant can still be drawn by an observer.
pub const REFERENCE_VIEW_FOV_Y_RAD: f64 = std::f64::consts::FRAC_PI_4;
/// See [`REFERENCE_VIEW_FOV_Y_RAD`].
pub const REFERENCE_VIEW_ROWS_PX: f64 = 720.0;

/// ★ THE DRAWABLE ANGLE: what ONE pixel subtends at the reference view — `2·tan(fov/2) / rows`.
/// Below it a body cannot be drawn at all, whatever its brightness. It is the floor for SHIPPING an
/// occupant to an observer (cheap), where the dot angle [`VISIBILITY_THETA_MIN_RAD`] is the bar for
/// WAKING a realm (expensive): a person is worth a row long before a planet is worth a shard.
/// Example: a two-metre character (circumscribed extent one metre) is one pixel at about 1.7 km and
/// is shipped inside that; a one-metre figure (extent half a metre) at about 870 m.
#[must_use]
pub fn drawable_theta_min_rad() -> f64 {
    2.0 * (REFERENCE_VIEW_FOV_Y_RAD * 0.5).tan() / REFERENCE_VIEW_ROWS_PX
}

/// A per-realm Area-of-Interest hysteresis band (METRES) for demand-driven realm lifecycle (RLM
/// Step 2). DISTINCT from [`OverlapBand`]/[`ContainmentBand`]: it drives a child SHARD's spin-up/down,
/// not entity membership. `spin_up_r_m` (a child within this range of an occupant is DEMANDED live)
/// and `tear_down_r_m > spin_up_r_m` (released only past this larger radius) straddle a dead-zone so an
/// occupant loitering at the edge cannot flap the child. `grace_ticks` holds a would-be release that
/// many ticks after the last in-range observation (the temporal anti-thrash; the geometric half is the
/// gap). Fields PRIVATE — the `0 < spin_up < tear_down` invariant is unconstructible-if-violated.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct AoiConfig {
    spin_up_r_m: f64,
    tear_down_r_m: f64,
    grace_ticks: u32,
}

impl AoiConfig {
    /// The VELOCITY-SAFE constructor — the ONLY fallible way to build a LIVE band (mirrors
    /// [`ContainmentBand::for_containment_velocity_safe`]). `spin_up = base_extent·spin_up_factor`;
    /// `tear_down = max(base_extent·tear_down_factor, spin_up + need)`, `need = |v_rel|·dt·(K_SAFETY +
    /// k_safety_extra)` — the gap WIDENED so a body closing at `v_rel` (m/s) over `dt`-second ticks can
    /// neither skip the band nor thrash the child. Fallible LOUD unless `0 < spin_up < tear_down` (the
    /// exact flap this type prevents). NOTE: at zero factor the generator calls [`AoiConfig::inert`],
    /// never this.
    ///
    /// # Errors
    /// [`BandError::InvalidEdges`] unless `0 < spin_up < tear_down`.
    #[must_use = "the Result carries a BandError that must not be dropped"]
    pub fn for_velocity_safe(
        base_extent: f64,
        spin_up_factor: f64,
        tear_down_factor: f64,
        v_rel: f64,
        dt: f64,
        grace_ticks: u32,
        k_safety_extra: f64,
    ) -> Result<AoiConfig, BandError> {
        let spin_up = base_extent * spin_up_factor;
        let need = v_rel.abs() * dt * (K_SAFETY + k_safety_extra);
        let tear_down = f64::max(base_extent * tear_down_factor, spin_up + need);
        if spin_up > 0.0 && tear_down > spin_up {
            let band = AoiConfig {
                spin_up_r_m: spin_up,
                tear_down_r_m: tear_down,
                grace_ticks,
            };
            debug_assert!(band.width_safe_for(v_rel, dt));
            Ok(band)
        } else {
            Err(BandError::InvalidEdges)
        }
    }

    /// The inert AoI (zero factor / walk-scale): `spin_up == 0` ⇒ nothing is ever in range ⇒ no demand.
    /// Distinct from the fallible ctor so the byte-identity path never touches the reject arm. This is
    /// the `#[serde(default)]` for [`RealmRegion::aoi`] — a legacy `regions.json` missing the field
    /// decodes to inert (behaviour-identical).
    #[must_use]
    pub fn inert() -> AoiConfig {
        AoiConfig {
            spin_up_r_m: 0.0,
            tear_down_r_m: 0.0,
            grace_ticks: 0,
        }
    }

    /// ★ THE SAME BAND AT A NEW INNER RADIUS (the reach, 2026-09-04): a child that states its reach
    /// re-bands at that reach without a rebuild. The tear-down keeps the WIDER of its old ratio to the
    /// spin-up and its old gap above it, so a band that was velocity-safe stays velocity-safe. An inert
    /// band stays inert (walk scale never wakes anything), and a non-positive reach changes nothing.
    /// Example: a planet's boot band was 2 million km in, 4 million km out; it states a reach of
    /// 5 million km; its band is now 5 million km in, 10 million km out.
    #[must_use]
    pub fn with_spin_up(self, spin_up_r_m: f64) -> AoiConfig {
        if (self.spin_up_r_m <= 0.0) | (spin_up_r_m <= 0.0) {
            return self;
        }
        let ratio = self.tear_down_r_m / self.spin_up_r_m;
        let gap = self.tear_down_r_m - self.spin_up_r_m;
        AoiConfig {
            spin_up_r_m,
            tear_down_r_m: f64::max(spin_up_r_m * ratio, spin_up_r_m + gap),
            grace_ticks: self.grace_ticks,
        }
    }

    /// The inner (spin-up) radius — metres; a child within this of an occupant is demanded live.
    #[must_use]
    pub fn spin_up_r_m(&self) -> f64 {
        self.spin_up_r_m
    }
    /// The outer (tear-down) radius — metres; a live child is released only past this.
    #[must_use]
    pub fn tear_down_r_m(&self) -> f64 {
        self.tear_down_r_m
    }
    /// Grace ticks held after the last in-range observation before release.
    #[must_use]
    pub fn grace_ticks(&self) -> u32 {
        self.grace_ticks
    }

    fn width_safe_for(&self, v_rel: f64, dt_s: f64) -> bool {
        (self.tear_down_r_m - self.spin_up_r_m) >= v_rel.abs() * dt_s * K_SAFETY
    }

    /// AoI hysteresis over a scalar min distance (BRANCHLESS bitwise `&`/`|`, like
    /// [`ShardProfile::satisfies`] — no short-circuit region to leave uncovered, HR5). `min_dist` is the
    /// smallest distance from ANY occupant to the child (the caller reduces). Acquire within `spin_up`;
    /// once in, hold within the larger `tear_down`.
    #[must_use]
    pub fn in_range(&self, was_in: bool, min_dist: f64) -> bool {
        let hold = min_dist <= self.tear_down_r_m;
        let acquire = min_dist <= self.spin_up_r_m;
        (was_in & hold) | acquire
    }
}

/// One realm REGION: a volume that, when it is the DEEPEST region CONTAINING a point, defines that
/// point's realm. CONTAINMENT, not a portal — there is no `to_realm` and no [`Direction`] (both were
/// the directional-trigger model's; the destination is DERIVED as the containing realm, symmetric by
/// construction). Every field `Copy`, so the whole descriptor is. Serde only for the client's
/// `--realm-boxes` single-source (identical bytes — regions are COMPUTED from the universe seed, not
/// authored; see `realm_regions_for`, C-2).
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct RealmRegion {
    /// The realm an entity IS IN while this is its deepest containing region.
    pub realm: RealmId,
    /// WHERE THIS REALM SITS INSIDE ITS PARENT, measured in the PARENT's frame. `None`-parented (the
    /// ambient root) means the origin. Callers subtract it before the shape methods — that subtraction IS
    /// the parent doing the downward conversion for its child.
    ///
    /// ★ ITS TYPE IS THE FENCE (slice S9). See [`ParentCentre`]: this used to be a bare [`LatticePos`],
    /// which any reader could flatten with any rung — and a dozen of them flattened it with the CHILD's.
    pub center: ParentCentre,
    /// THIS REALM'S OWN frame — the space its `shape` is drawn in and the label anything standing inside
    /// it wears. NOT the frame `center` is in: `center` is the parent's number, in the parent's frame.
    ///
    /// The comment here used to say "the frame `center`/`shape` are expressed in", from the days when
    /// every region shared one identity frame and the two were indistinguishable. They are not the same
    /// frame and never were on a real world; three separate pieces of code now depend on this field being
    /// the CHILD's own frame (the child-roster lookups a source flush and an arriving pose go through, and
    /// the realm feed's edge head), so the description had to stop describing the other one.
    pub frame: FrameRef,
    /// The region shape (`Shell` | `Aabb` | `Obb`) — reused verbatim from the portal model.
    /// Since the real-scale re-solve this is THE BOUND (the two-body law, real_scale_design
    /// §3.0): containment, crossing, nesting, the liveness AoI floor, the speed ceiling.
    /// **Never drawn.**
    pub shape: Boundary,
    /// THE LOOK (the BOUND/LOOK split, real_scale_design §3.0): the outline this realm DRAWS —
    /// angular size, visibility reach, the climb walk, the marker radius its parent authors,
    /// the realm's own self-look. `None` means *this realm draws nothing* (the ambient
    /// Universe/Galaxy of THE world — "a containment boundary is never drawn" made structurally
    /// impossible to violate rather than merely avoided). At interim/walk scale bound and look
    /// coincide; at real scale the ratio is ~200:1 and one number cannot be both.
    /// `#[serde(default)]`: a legacy `regions.json` decodes to `None` — draws nothing, the
    /// conservative side, mirroring the `aoi` append discipline.
    #[serde(default)]
    pub look: Option<Boundary>,
    /// The signed-distance hysteresis band around the surface (anti-flap on the containment edge).
    pub band: ContainmentBand,
    /// Per-realm AoI hysteresis (RLM Step 2), populated by `worldgen::to_regions` from the seed. Inert
    /// (`spin_up == 0`) at walk-scale ⇒ no demand ⇒ behaviour byte-identity. `#[serde(default)]` so a
    /// legacy `regions.json` (written before this field) decodes to inert — the client/devcluster
    /// on-disk contract stays forward-compatible (on-disk BYTES differ — the field is additive to the
    /// JSON — but a legacy file parses and a fresh one round-trips).
    #[serde(default = "AoiConfig::inert")]
    pub aoi: AoiConfig,
    /// The enclosing realm you fall to on LEAVING this region. `None` ONLY for the single ambient root
    /// (the Universe), whose volume contains all reachable space — so an entity is ALWAYS in ≥1 realm.
    pub parent: Option<RealmId>,
}

/// The containment depth-argmax order: depth DESC (the innermost realm wins), then `RealmId` ASC, then
/// slice `ix` ASC. NO [`Direction`] — containment has no crossing direction (that was the portal
/// model's concern). A STRICT TOTAL order ⇒ the argmax is permutation-invariant, which matters because
/// the winning realm is load-bearing cross-host and must not depend on region slice order. A monotonic
/// helper (branchless-shim shape) so every tiebreak branch is covered ONCE here, not smeared
/// through the container fold.
#[must_use]
pub fn depth_beats(a: (u32, RealmId, usize), b: (u32, RealmId, usize)) -> bool {
    if a.0 != b.0 {
        a.0 > b.0
    } else if a.1 != b.1 {
        a.1 < b.1
    } else {
        a.2 < b.2
    }
}

/// The ONE SYMMETRIC re-home decision — REPLACES the asymmetric [`should_commit`]. `Some(dest)` iff the
/// deepest hysteretic CONTAINER this tick (`container`) differs from the realm the shard OWNS the entity
/// in (`owning`), AND the post-commit cooldown has elapsed. NO [`Direction`]: escaping a realm
/// (System→Galaxy) and entering one (Galaxy→System) are the IDENTICAL path — the container simply
/// changed. The [`ContainmentBand`] hysteresis sits BEFORE this (the caller only flips `container` once
/// the dead-zone is crossed), so `should_rehome` needs no dwell of its own — the band IS the dwell.
/// `k_dwell` is a symmetric post-commit anti-thrash cooldown, direction-blind by design and CORRECT here
/// (the direction-blindness was a bug only in the asymmetric portal model). NOTHING calls it yet (the
/// detector wiring is C-3).
#[must_use]
pub fn should_rehome(
    owning: RealmId,
    container: RealmId,
    since_commit: Option<u32>,
    tuning: &BoundaryTuning,
) -> Option<RealmId> {
    if let Some(sc) = since_commit
        && sc < tuning.k_dwell
    {
        return None;
    }
    (container != owning).then_some(container)
}

/// A region's containment sort key: `(depth, realm, slice-ix)` — the input to [`depth_beats`]. `depth`
/// is the region's nesting depth (0 = the ambient root); `realm` is unique per region; `ix` is a final
/// determinism tiebreak. Computed once at boot (regions are static at P3), cached, and read per tick.
pub type DepthKey = (u32, RealmId, usize);

/// The nesting DEPTH of the region named `realm` = the number of ancestors up to the ambient root
/// (root = 0, its children = 1, grandchildren = 2, ...). Higher depth = more nested = wins containment.
/// Walks `parent` pointers, resolving each to the region whose `.realm` equals it (unique per region —
/// `guard_regions_nest`). Bounded by `regions.len()` hops so a MALFORMED cyclic set (which the boot
/// guard rejects) terminates instead of hanging. Computed ONCE at boot and cached, NOT per tick.
#[must_use]
pub fn region_depth(regions: &[RealmRegion], realm: RealmId) -> u32 {
    let mut depth = 0u32;
    let mut cur = realm;
    for _ in 0..regions.len() {
        let Some(region) = regions.iter().find(|r| r.realm == cur) else {
            return depth; // dangling parent (the boot guard rejects this; a safe stop if it slips)
        };
        let Some(parent) = region.parent else {
            return depth; // reached the ambient root
        };
        depth += 1;
        cur = parent;
    }
    depth // hop cap hit — a cycle (the boot guard rejects it; a safe stop, never a hang)
}

/// The entity's realm THIS TICK = the DEEPEST region whose membership holds, folded from the ambient
/// ROOT realm as the IDENTITY (depth 0). Returns [`RealmId`] — NOT `Option` — because the root seeds
/// the fold, so the argmax is never empty: there is **no `None` arm to leave uncoverable** (HR5-clean
/// by TYPE, the "always in a realm" mandate made structural). `members` are the [`DepthKey`]s whose
/// membership bit is set this tick (the root need not appear — it is the identity seed). SYMMETRIC and
/// direction-free: escaping a realm and entering one both fall out of which region now wins.
#[must_use]
pub fn container(root_realm: RealmId, members: &[DepthKey]) -> RealmId {
    let mut best: DepthKey = (0, root_realm, usize::MAX); // the root: depth 0, the fold identity
    for &m in members {
        if depth_beats(m, best) {
            best = m;
        }
    }
    best.1
}

/// The signed distance from `pose` to `region`'s surface, RE-EXPRESSED into the region's frame FIRST
/// (the input-side frame seam). The placements arrive as an authored [`PlacementBook`] — the region's
/// PARENT's rows at the pose's own instant — so this measurement cannot ask how the region moves
/// (SL4): it reads the same rows every other consumer reads. MONOMORPHIC: with one concrete book type
/// there is no per-`FrameContext` monomorphization left to multiply branches across (the old
/// `_resolved` split existed only to serve that genericity and is folded back in). The caller treats
/// an `Err` region as "not a member" (safe degrade), never a container.
///
/// # Errors
/// [`FrameError`] if the book speaks at a different instant than the pose's stamp, or has no placement
/// for `pose`'s frame or `region.frame`.
pub fn region_signed_distance(
    pose: &StampedPose,
    region: &RealmRegion,
    book: &PlacementBook,
) -> Result<f64, FrameError> {
    let p = transfer_frame(pose, region.frame, book)?;
    // EVERY REALM IS CENTRED ON ITSELF. The reframe above has already expressed the pose in this region's
    // OWN frame, and in its own frame a region sits at its own origin — so the distance is measured from
    // zero and there is NOTHING further to subtract.
    //
    // This used to subtract `region.center` here as well, which counted the region's position TWICE the
    // moment frames stopped being the identity: the reframe removed it, and then this removed it again.
    // While every realm sat at the origin both were zero and nothing showed. Once a realm sat anywhere
    // else, an occupant standing dead centre inside a five-metre box measured twenty metres OUTSIDE it,
    // so entry could never latch and a crossing into it simply never happened.
    //
    // `region.center` is not wrong — it is where this realm sits IN ITS PARENT'S FRAME, which is exactly
    // the thing a parent knows and a child does not. It belongs to the parent's downward question, which
    // is [`child_signed_distance`]. It has no business in the child's own-frame answer.
    //
    // `delta_m` is the exact form: it subtracts whole numbers as integers (no rounding is possible) and
    // leftovers as floats, then combines — so a position far from the origin is never truncated.
    Ok(region.shape.signed_distance(
        p.pos
            .delta_m(LatticePos::local(DVec3::ZERO), p.frame.tier()),
    ))
}

/// One region's CONTAINMENT MEMBERSHIP verdict plus the f64 signed distance beside it (for gauges
/// and logs) — the ONE band question every consumer asks (the scan, the departure guard, the
/// arrival guard), so no two sites can drift onto different rules.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RegionVerdict {
    /// The hysteretic membership answer — INTEGER for `Shell`/`Aabb` (see [`region_verdict`]).
    pub member: bool,
    /// The f64 signed distance to the region surface (negative inside) — a GAUGE value for logs and
    /// diagnosis, never the authority answer for the integer shapes.
    pub signed_distance_m: f64,
}

/// THE INTEGER CONTAINMENT VERDICT (real-scale addendum §A4.8 row 12 / §5.2 item 4, discharging the
/// standing `cell_index` control-quantization mandate): the hysteretic membership answer computed
/// **on integers only — no float, no square root — for `Shell` and `Aabb` regions**, so the verdict
/// is exact at every magnitude and bit-identical across hosts. `Obb` cannot take the integer path
/// (a rotated box needs a rotation, and no rotation of an integer lattice vector is an integer
/// lattice vector) and keeps the f64 arm with its quantum declared — a SHAPE branch that
/// [`Boundary::signed_distance`] already has, never a realm-kind fork (§A7.2).
///
/// The thresholds state their rounding direction ONCE: **floor the acquire edge, ceil the release
/// edge** — conservative on both sides, and the two edges can never cross (§A4.4). The sub-cell
/// residual is DISCARDED by the comparator; it can shift a verdict only inside one 0.98 mm cell,
/// and `guard_quantum_band`-class fences keep every band ≥ 3 orders above that.
///
/// The overflow hole (H-01) is cured structurally: the per-axis Chebyshev PRE-TEST runs before any
/// square, so a hostile-but-in-domain pair (`|Δ| = 2·CELL_DOMAIN_MAX` per axis, whose square-sum
/// overflows `i128`) is classified OUTSIDE without squaring; the guarded [`Separation::cells_sq`]
/// backstops the un-pre-testable remainder as a refusal, never a panic.
///
/// # The verdict tests the tick's MOTION, not the instant (slice S5)
///
/// `prev_pos` is where this subject was at the previous evaluation, in the same frame as `pose`. The
/// question asked is about the SEGMENT between them, which adds **exactly one thing** to the shipped
/// answer: the case where the path dipped inside while BOTH samples were outside. Inside now, inside
/// then, and LEAVING all keep the shipped verdict bit for bit, and the shipped point answer is asked
/// FIRST — so the upgrade can only turn `false` into `true`, and no subject that is a member today
/// can stop being one.
///
/// **"No prior" is spelled as a degenerate segment** (`prev_pos == pose.pos`), which is what every
/// caller holding a single instant passes and what the first tick after a spawn, an adopt or an
/// arrival carries. That is not a second rule: it takes the same path and short-circuits before any
/// new arithmetic runs. It is also LOSSLESS rather than merely cheap — the guard compares CELLS, and
/// a segment spanning at most one cell per axis provably cannot disagree with the point rule (a
/// disagreement needs `|d|² ≥ 4` on a shell and two cells on some axis for a box).
///
/// The segment arithmetic is EXACT AT EVERY MAGNITUDE and needs **256 bits** to be so: the squared
/// step length alone reaches 2.552e38 and overflows a signed 128-bit integer, and the deciding
/// product peaks at 254 bits. Every refusal along that path — a negative edge, an out-of-domain
/// endpoint, a prior the book cannot place — returns the shipped point verdict rather than a guess.
///
/// What it does NOT cover (a moving region over a parked subject, `Obb`, one-tick fly-through
/// damping) is named in this module's own header.
///
/// # Errors
/// [`FrameError`] exactly as [`region_signed_distance`]: an instant mismatch or a frame the book
/// cannot place. Callers treat `Err` as "not a member" (safe degrade), never a container.
pub fn region_verdict(
    pose: &StampedPose,
    prev_pos: LatticePos,
    region: &RealmRegion,
    book: &PlacementBook,
    was_member: bool,
) -> Result<RegionVerdict, FrameError> {
    // STEPS 1-4 ARE THE SHIPPED BODY, IN THE SHIPPED ORDER, WITH NOTHING MOVED ABOVE THEM — including
    // `signed_distance_m` computed UNCONDITIONALLY before the shape fork, so every gauge, log line and
    // sibling-claim diagnostic downstream is byte-identical.
    let p = transfer_frame(pose, region.frame, book)?;
    let tier = p.frame.tier();
    let sep = p.pos.separation(LatticePos::ORIGIN, tier);
    let signed_distance_m = region.shape.signed_distance(sep.metres());
    // STEP 5: the prior endpoint. An `Option`, never an `Err` — see `prior_separation`.
    let sep_prev = prior_separation(pose, prev_pos, region.frame, book, tier);
    // STEP 6: ONE shape fork. The swept logic lives INSIDE each shape's own arm, so no two shape
    // matches can drift onto different rules (the rule this type's own doc states).
    let member = match region.shape {
        Boundary::Shell { r } => {
            shell_member_swept(sep, sep_prev, r, &region.band, was_member, tier)
        }
        Boundary::Aabb { half } => {
            aabb_member_swept(sep, sep_prev, half, &region.band, was_member, tier)
        }
        // THE ROTATED-SHAPE DECIMAL ARM (§A7.2): the one lawful f64 verdict, quantum = the f64 ulp
        // at the region's own magnitude. No shipped region is an Obb; the arm stays covered.
        // NOT SWEPT — see the module header's exclusion list.
        Boundary::Obb { .. } => region.band.member(was_member, signed_distance_m),
    };
    Ok(RegionVerdict {
        member,
        signed_distance_m,
    })
}

/// The prior endpoint re-expressed in the region's frame, or `None` for "no usable prior".
///
/// NEVER an `Err`. The current pose's verdict is the shipped answer and may not be lost to a failure
/// about the prior, so an unplaceable prior degrades to the point verdict rather than propagating.
fn prior_separation(
    pose: &StampedPose,
    prev_pos: LatticePos,
    to: FrameRef,
    book: &PlacementBook,
    tier: Tier,
) -> Option<Separation> {
    // DEGENERACY IS DECIDED ON CELLS, NOT ON THE WHOLE POSITION. `LatticePos` compares an f64
    // `offset`, so `prev_pos == pose.pos` would be a FLOAT equality and a NaN offset would make an
    // identical pose compare UNEQUAL — the stationary-identity guarantee would have a hole in exactly
    // the case it exists for. Cells are integers: this guard is TOTAL.
    //
    // It is also PROVABLY LOSSLESS. Two NORMALIZED positions with equal cells differ by under one cell
    // per axis, so the reframed segment spans at most one cell per axis, i.e. `|d|² ≤ 3`; and a
    // disagreement between the segment and the point rule needs `|d|² ≥ 4` on the shell (a both-outside
    // pair forces `m² ≥ E²+1`, while `4(m² − E²) ≤ |d|²`) and `|d_i| ≥ 2` on some axis for the box.
    // The bound is ATTAINED at `a = (−1,0,0) → b = (1,0,0)`, `E = 0`, so it is tight, not padded.
    if prev_pos.cell() == pose.pos.cell() {
        return None;
    }
    // SANITIZE THE SYNTHESIZED PRIOR. `prev_offset` is a bare field no sanitizer touches, and
    // `transfer_frame` carries unchecked i64 cell arithmetic. `sanitized` passes every in-domain pose
    // through bit-for-bit, so this constrains only the new input and changes no shipped behaviour.
    let prev = StampedPose {
        pos: prev_pos,
        ..*pose
    }
    .sanitized();
    // An unplaceable prior degrades to the point answer. Reachable: `RotationBeyondExactReach` depends
    // on the separation MAGNITUDE, so it can refuse the prior while accepting the current pose.
    let q = transfer_frame(&prev, to, book).ok()?;
    Some(q.pos.separation(LatticePos::ORIGIN, tier))
}

/// A band edge in integer cells, with THE rounding rule stated once (§A4.4): the acquire edge
/// (`was_member == false`) rounds DOWN (harder to acquire), the release edge rounds UP (easier to
/// hold) — conservative on both sides, so the two integer edges can never cross even when the f64
/// edges sit within one cell of each other. The `as i64` cast saturates (a hostile/huge threshold
/// cannot wrap), and a negative acquire edge (a region thinner than its own inset) yields a
/// negative cell threshold no non-negative Chebyshev can pass — never-acquirable, exactly as the
/// f64 rule answers it.
fn band_edge_cells(edge_m: f64, was_member: bool, tier: Tier) -> i64 {
    let cells = edge_m / tier.cell_edge_m();
    if was_member {
        cells.ceil() as i64
    } else {
        cells.floor() as i64
    }
}

/// THE SHELL BAND EDGE, derived ONCE and threaded. The shipped expression, hoisted: `band_edge_cells`
/// is pure in unchanged arguments, so hoisting changes call sites, never values.
fn shell_edge_cells(r_m: f64, band: &ContainmentBand, was_member: bool, tier: Tier) -> i64 {
    let edge_m = if was_member {
        r_m + band.outset()
    } else {
        r_m - band.inset()
    };
    band_edge_cells(edge_m, was_member, tier)
}

/// The `Shell` arm of the integer verdict: `Σ Δcellᵢ²` as `i128` against the squared edge. The
/// Chebyshev pre-test rejects before any square (H-01); past it the edge bounds every axis, so the
/// square is safe wherever the edge is lawful, and the guarded `cells_sq` turns the unlawful
/// remainder into "outside" rather than a panic.
fn shell_point_member(sep: Separation, edge_cells: i64) -> bool {
    if sep.cells_chebyshev() > edge_cells {
        return false;
    }
    match sep.cells_sq() {
        Some(sq) => sq <= i128::from(edge_cells) * i128::from(edge_cells),
        // Reachable only when the edge itself exceeds the square-safe domain (an unlawful,
        // beyond-half-light-year shell): refuse membership rather than panic — H-01's cure.
        None => false,
    }
}

/// THE SHELL ARM, SWEPT. Adds EXACTLY ONE THING to the shipped answer: the case where the tick's
/// motion segment dipped inside while BOTH of its endpoints were outside. Every other case — inside
/// now, inside then, leaving — keeps the shipped verdict bit for bit.
///
/// The order matters and is the whole equivalence argument: the shipped point answer is asked FIRST
/// and returns immediately, so the upgrade can only ever turn `false` into `true`. No subject that is
/// a member today can stop being one.
fn shell_member_swept(
    cur: Separation,
    prev: Option<Separation>,
    r_m: f64,
    band: &ContainmentBand,
    was_member: bool,
    tier: Tier,
) -> bool {
    let edge = shell_edge_cells(r_m, band, was_member, tier);
    if shell_point_member(cur, edge) {
        return true;
    }
    // Degenerate segment, or a prior this book cannot place: the shipped answer stands.
    let Some(prev) = prev else {
        return false;
    };
    // LEAVING. The prior was inside and the current sample is not — the shipped answer is "released",
    // and holding the subject for one more tick would invert membership at exactly the speeds this
    // machinery exists for (at a galaxy ceiling the subject is already thousands of shell radii out).
    if shell_point_member(prev, edge) {
        return false;
    }
    shell_segment_dips(prev.cells(), cur.cells(), edge)
}

/// Does the closed segment `a → b` reach within `edge_cells` of the origin? EXACT, on integers, at
/// every magnitude — no float, no square root, no division.
///
/// PRECONDITION, established by the only caller: both endpoints are already known OUTSIDE under the
/// shipped point rule at this same `edge_cells`. Every violation of it FAILS CLOSED, and every refusal
/// below returns the shipped point verdict rather than a guess.
fn shell_segment_dips(a: I64Vec3, b: I64Vec3, edge_cells: i64) -> bool {
    // B1 A region thinner than its own inset. A negative edge must never be squared into a positive
    //    one: that would manufacture membership for a region the shipped rule calls never-acquirable.
    if edge_cells < 0 {
        return false;
    }
    // B2 DOMAIN GUARD. `unsigned_abs`, never `abs` (`i64::MIN.abs()` panics in debug). Refuses an
    //    out-of-domain endpoint exactly as `cells_sq` refuses an unsquarable separation: no upgrade.
    let reach =
        a.x.unsigned_abs()
            .max(a.y.unsigned_abs())
            .max(a.z.unsigned_abs())
            .max(b.x.unsigned_abs())
            .max(b.y.unsigned_abs())
            .max(b.z.unsigned_abs());
    if reach > CELL_DOMAIN_MAX.unsigned_abs() {
        return false;
    }

    let (ax, ay, az) = (i128::from(a.x), i128::from(a.y), i128::from(a.z));
    let (dx, dy, dz) = (
        i128::from(b.x) - ax,
        i128::from(b.y) - ay,
        i128::from(b.z) - az,
    );

    // B3 The nearest point of the segment is `a`, which is already known outside. This arm also
    //    swallows a zero-length segment (`P == 0 ≥ 0`), so there is no separate degenerate branch.
    //    Worst magnitude 3·CDM·2·CDM = 1.276e38, against i128::MAX = 1.701e38.
    let p = ax * dx + ay * dy + az * dz;
    if p >= 0 {
        return false;
    }

    // B4 The nearest point is `b`, also known outside. Compared UNSIGNED so that `d·d` — which reaches
    //    2.552e38 and does NOT fit a signed 128-bit integer — never has to.
    let pn = p.unsigned_abs();
    let s = dx.unsigned_abs() * dx.unsigned_abs()
        + dy.unsigned_abs() * dy.unsigned_abs()
        + dz.unsigned_abs() * dz.unsigned_abs();
    if pn >= s {
        return false;
    }

    // B5 INTERIOR MINIMUM. `|F|² = |a|² − (a·d)²/(d·d)`, so `|F|² ≤ E² ⟺ (|a|² − E²)·(d·d) ≤ (a·d)²`.
    //    `checked_sub` is a TOTALITY guard, and it is documented as one: the precondition makes the
    //    `None` arm unreachable from `region_verdict` (both-outside with `edge ≥ 0` forces `|a|² > E²`
    //    either through the Chebyshev arm or the square arm), and it fails CLOSED.
    //
    //    WHAT THE UNIT TEST BELOW CAN AND CANNOT CATCH, measured rather than assumed. Replacing this
    //    with a BARE subtraction panics in debug and the test goes red. Replacing it with a WRAPPING
    //    subtraction is INDISTINGUISHABLE, and that is a proof rather than a gap: a wrapped `g` is
    //    about `2¹²⁸`, while `g·s ≤ (a·d)²` requires `g ≤ |a|²`, so the wrapped form answers `false`
    //    on every input the checked form answers `false` on. The guard is kept for the panic it
    //    prevents and for saying out loud that the arm is reachable only from a violated contract.
    let a2 = ax.unsigned_abs() * ax.unsigned_abs()
        + ay.unsigned_abs() * ay.unsigned_abs()
        + az.unsigned_abs() * az.unsigned_abs();
    let e2 = u128::from(edge_cells.unsigned_abs()) * u128::from(edge_cells.unsigned_abs());
    let Some(g) = a2.checked_sub(e2) else {
        return false;
    };

    // B6 THE ONE 256-BIT COMPARISON. Both sides peak at 254 bits (`3·CDM²·3·(2·CDM)²` = 1.628e76
    //    against `2²⁵⁶−1` = 1.158e77), so 256 is the minimum sufficient width AND it is sufficient.
    le256(wide_mul(g, s), wide_mul(pn, pn))
}

/// 256-bit `<=` on `(hi, lo)` pairs, branchless, written out here rather than leaning on the derived
/// tuple ordering of a foreign type — this is the comparison the containment answer turns on.
#[inline]
fn le256(l: (u128, u128), r: (u128, u128)) -> bool {
    (l.0 < r.0) | ((l.0 == r.0) & (l.1 <= r.1))
}

/// `u128 × u128 → (hi, lo)`, exact, from four 64-bit limb products. ZERO branches — one region — which
/// is the point of the limb form: nothing here can be left undriven.
fn wide_mul(x: u128, y: u128) -> (u128, u128) {
    const M: u128 = u64::MAX as u128;
    let (x1, x0) = (x >> 64, x & M);
    let (y1, y0) = (y >> 64, y & M);
    let p00 = x0 * y0;
    let p01 = x0 * y1;
    let p10 = x1 * y0;
    let p11 = x1 * y1;
    let mid = (p00 >> 64) + (p01 & M) + (p10 & M);
    // The mask is EXPLICIT. `mid << 64` alone is correct only through the silent discard of the bits
    // shifted out, which is a fact about the language rather than a statement of the intent.
    let lo = (p00 & M) | ((mid & M) << 64);
    let hi = p11 + (p01 >> 64) + (p10 >> 64) + (mid >> 64);
    (hi, lo)
}

/// The `Aabb` arm: Chebyshev on integers, per axis. INSIDE a box the Euclidean SDF *is* the
/// Chebyshev excess, so the acquire arm is semantics-identical to the f64 rule; the release arm
/// holds over the box inflated by the outset PER AXIS (a box, where the f64 SDF's hold region had
/// rounded corners) — a declared, conservative-outward difference bounded by the outset itself.
fn aabb_edge_cells(half_m: f64, band: &ContainmentBand, was_member: bool, tier: Tier) -> i64 {
    let edge_m = if was_member {
        half_m + band.outset()
    } else {
        half_m - band.inset()
    };
    band_edge_cells(edge_m, was_member, tier)
}

/// The shipped per-axis rule, made TOTAL. `i64::abs` panics in debug on `i64::MIN`; `unsigned_abs`
/// cannot. The `edge_cells >= 0` term is REQUIRED, not decoration: an unsigned comparison against a
/// negative edge would compare against its magnitude and admit a region the shipped rule calls
/// never-acquirable. Bitwise `&`: one region, no branch.
#[inline]
fn axis_member(d: i64, edge_cells: i64) -> bool {
    (edge_cells >= 0) & (d.unsigned_abs() <= edge_cells.unsigned_abs())
}

fn aabb_point_member(sep: Separation, e: [i64; 3]) -> bool {
    let c = sep.cells();
    // Bitwise `&`, not `&&`: every operand is cheap + pure, and a short-circuit would leave the
    // tail axes uncoverable from a false-LHS side (the `AoiConfig::in_range` discipline).
    axis_member(c.x, e[0]) & axis_member(c.y, e[1]) & axis_member(c.z, e[2])
}

/// THE BOX ARM, SWEPT. Same shape as the shell arm, same one addition, same ordering guarantee.
fn aabb_member_swept(
    cur: Separation,
    prev: Option<Separation>,
    half: DVec3,
    band: &ContainmentBand,
    was_member: bool,
    tier: Tier,
) -> bool {
    // ONE derivation per axis, x then y then z — the shipped evaluation order.
    let e = [
        aabb_edge_cells(half.x, band, was_member, tier),
        aabb_edge_cells(half.y, band, was_member, tier),
        aabb_edge_cells(half.z, band, was_member, tier),
    ];
    if aabb_point_member(cur, e) {
        return true;
    }
    let Some(prev) = prev else {
        return false;
    };
    if aabb_point_member(prev, e) {
        return false;
    }
    box_segment_dips(prev.cells(), cur.cells(), e)
}

/// Segment versus axis-aligned box by the separating-axis test: three box FACE axes plus three CROSS
/// axes. Exact; no division, no rational parameter, no interval bookkeeping to keep consistent.
fn box_segment_dips(a: I64Vec3, b: I64Vec3, e: [i64; 3]) -> bool {
    // C1 An axis thinner than its own inset — the box is empty, exactly as the shipped rule says.
    if (e[0] < 0) | (e[1] < 0) | (e[2] < 0) {
        return false;
    }
    // C2 DOMAIN GUARD — the same rule and the same refusal as the shell arm.
    let reach =
        a.x.unsigned_abs()
            .max(a.y.unsigned_abs())
            .max(a.z.unsigned_abs())
            .max(b.x.unsigned_abs())
            .max(b.y.unsigned_abs())
            .max(b.z.unsigned_abs());
    if reach > CELL_DOMAIN_MAX.unsigned_abs() {
        return false;
    }

    // AN ANSWER-PRESERVING CLAMP, not an overflow cure. Past C2 every segment coordinate is within
    // ±CDM, so a half-edge wider than CDM already contains every reachable point and clamping leaves
    // the intersection SET identical. What it buys is margin: the unclamped radius term lands within
    // 5.5e19 of the signed 128-bit ceiling — a fit with a relative headroom of 3e-19 — and the clamp
    // restores a factor of two.
    let h = [
        i128::from(e[0].min(CELL_DOMAIN_MAX)),
        i128::from(e[1].min(CELL_DOMAIN_MAX)),
        i128::from(e[2].min(CELL_DOMAIN_MAX)),
    ];
    let a = [i128::from(a.x), i128::from(a.y), i128::from(a.z)];
    let b = [i128::from(b.x), i128::from(b.y), i128::from(b.z)];

    // C3 THE FACE AXES: both endpoints beyond the same face. Comparisons only, no products. This is
    //    the exact generalisation of the shipped Chebyshev pre-test — for `a == b` it IS it. ONE
    //    bitwise expression, so the six comparisons cost one branch rather than six.
    let miss = ((a[0].min(b[0]) > h[0]) | (a[0].max(b[0]) < -h[0]))
        | ((a[1].min(b[1]) > h[1]) | (a[1].max(b[1]) < -h[1]))
        | ((a[2].min(b[2]) > h[2]) | (a[2].max(b[2]) < -h[2]));
    if miss {
        return false;
    }

    // C4 THE CROSS AXES. The segment's own half-length vector is parallel to `d`, so its radius on
    //    every cross axis is ZERO and the midpoint's projection equals the START point's:
    //    `m × d = (a + d/2) × d = a × d`. The endpoint form is EXACT, not an approximation.
    let d = [b[0] - a[0], b[1] - a[1], b[2] - a[2]];
    let c = [
        a[1] * d[2] - a[2] * d[1],
        a[2] * d[0] - a[0] * d[2],
        a[0] * d[1] - a[1] * d[0],
    ];
    // `abs` on an i128 whose magnitude is at most 2·CDM — total. Never `i64::abs`.
    let r = [
        h[1] * d[2].abs() + h[2] * d[1].abs(),
        h[2] * d[0].abs() + h[0] * d[2].abs(),
        h[0] * d[1].abs() + h[1] * d[0].abs(),
    ];
    let separated = (c[0].abs() > r[0]) | (c[1].abs() > r[1]) | (c[2].abs() > r[2]);
    !separated
}

/// Why a realm-region forest is malformed — the boot fence (task #135, C-5). ONE variant per REJECT arm
/// so each is asserted with `expect_err` equality (HR5(d)), never `matches!`. These are PURE TOPOLOGY
/// checks; the geometric child-⊆-parent volume subset check needs cross-frame Shape×Shape math and is
/// LEDGERED to P4/P5 (DEFERRED D-45).
#[derive(Clone, Copy, Debug, PartialEq, thiserror::Error)]
pub enum RegionNestError {
    #[error("the forest has {found} ambient roots (parent: None); exactly one is required")]
    RootCount { found: usize },
    /// A band too thin for the grid it is measured on — see [`guard_quantum_band`].
    #[error(
        "realm {realm:?} has a band of {width_m} m, which is only {cells} cells at its own tier; a \
         band must be at least {QUANTUM_BAND_CELLS} cells wide or the sub-cell residual the integer \
         verdict discards could decide a membership"
    )]
    BandTooThin {
        realm: RealmId,
        width_m: f64,
        cells: f64,
    },
    #[error(
        "two regions share realm {realm:?}; each realm must be unique (region_depth determinism)"
    )]
    DuplicateRealm { realm: RealmId },
    #[error("region {realm:?} names parent {parent:?}, which is not a region in the forest")]
    DanglingParent { realm: RealmId, parent: RealmId },
    #[error("region {realm:?}'s parent chain cycles or never reaches the single root")]
    CycleOrOrphan { realm: RealmId },
    #[error(
        "region {realm:?} is not geometrically inside its parent {parent:?}: it reaches {reach} m from \
         the parent's centre but the parent's usable interior ends at {limit} m"
    )]
    ChildEscapesParent {
        realm: RealmId,
        parent: RealmId,
        reach: f64,
        limit: f64,
    },
    #[error(
        "region {realm:?} has no stated reach — the boot must supply every child's worst-instant \
         reach (a mover's apoapsis bound, a static's authored offset); a defaulted zero is banned"
    )]
    NoReachForChild { realm: RealmId },
}

/// A child's WORST-INSTANT reach description, supplied by the boot — the ONE party that may name how
/// things move (the placement arc S4). The fence itself consumes this KIND-BLIND: it cannot ask how a
/// child moves, only how far its motion can carry it.
/// - [`Fixed`](ChildReach::Fixed): a child that never moves off its authored offset — judged EXACTLY
///   at that point (a box's corners measured where they really are).
/// - [`Excursion`](ChildReach::Excursion): a bound on how far the child's motion can carry its centre
///   from the parent's origin (an orbit's APOAPSIS `a(1+e)`, a thruster budget at P8) — direction
///   unknown, so the child is judged as a shell swept over the whole excursion.
///
/// This is a statement about GEOMETRY ("an exact point" vs "a radius bound"), never a motion kind: a
/// moon, a ship on thrusters and a drifting rock all state an `Excursion`; the fence cannot tell them
/// apart. (The scalar-only form the design sketched was REFUTED by the shipped world: sphere-izing a
/// static box loses its corner exactness — walk Area A reaches 9.06 m of a 10 m planet measured
/// exactly, but 10.196 m sphere-ized, and that forest is correct.)
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum ChildReach {
    /// The child's exact authored offset in its parent's frame (metres).
    Fixed(DVec3),
    /// The worst-instant distance of the child's centre from the parent's origin (metres).
    Excursion(f64),
}

/// Does a child region fit inside its parent's usable interior AT ITS WORST INSTANT?
///
/// WHY THIS IS A BOOT FENCE AND NOT A LEDGERED NICETY. Containment hysteresis derives a subject's
/// prior from ANCESTRY: being authoritatively in a realm makes you a member of that realm and of every
/// realm containing it (task #177). That inference is only sound if "containing" is geometrically TRUE
/// — at EVERY instant, not just the epoch. A child poking outside its parent would hand a subject
/// membership of a region it is demonstrably outside — the derived prior would then assert a falsehood
/// every tick, and the containment fold could pick a parent the subject has physically left.
///
/// THE FENCE USED TO GO SIZE-ONLY FOR EVERY MOVER: it read the stored region `center`, and a mover's
/// stored centre is ZERO, so a planet whose orbit carried it outside its star system's shell booted
/// clean — MEASURED on THE world before the cure: two of the three systems' outer planets crossed
/// their own system's surface at apoapsis (152.57 m and 155.05 m against a 150 m shell). The fence now
/// stops asking for a position — no single instant can answer it — and asks for a BOUND
/// ([`ChildReach`]), which the boot derives from the one thing that may know how a child moves.
///
/// SCOPE, stated honestly:
/// - It compares the child's farthest point against the parent's inscribed extent — the largest sphere
///   the parent fully contains. Necessary, not sufficient: a box parent's corners are not promised to
///   anybody.
/// - It compares against the parent's BOUNDARY, deliberately NOT the boundary minus the parent's
///   hysteresis inset. The prior this fence protects says "authoritatively inside the child ⇒ a member
///   of the parent"; that is falsified only by a child poking outside the parent's boundary. The inset
///   is a release margin INSIDE that boundary — a subject there is still geometrically within the
///   parent, so the prior holds. MEASURED: subtracting it condemns the shipped walk forest's Area A,
///   whose farthest corner is 9.06 m from a planet of radius 10 with a 1 m inset. That forest is
///   correct; the stricter bound was not.
fn child_fits_in_parent(
    reach: &ChildReach,
    child: &RealmRegion,
    parent: &RealmRegion,
) -> Option<(f64, f64)> {
    let reach_m = match reach {
        ChildReach::Fixed(at) => child.shape.max_reach_from(*at),
        ChildReach::Excursion(r) => r + child.shape.max_reach_from(DVec3::ZERO),
    };
    let limit = parent.shape.inscribed_extent();
    (reach_m > limit).then_some((reach_m, limit))
}

/// The BOOT FENCE for a realm-region forest (task #135, C-5): validation run at shard boot BEFORE the
/// infallible [`RealmRegions::new`] (in `vd_sim`), so a malformed set fails LOUD rather than degrading
/// to the detector's `root_realm == None` no-op. Rejects, in order (fail-fast + cheap-first): count >
/// `max` (the membership-bitset width, passed by the caller — vd-core stays free of the bitset
/// detail); not exactly one `parent: None` root; a duplicate `.realm`; a dangling parent; a parent
/// chain that cycles or never reaches the root; and — the GEOMETRIC half, live here, not deferred — a
/// child whose worst-instant reach ([`ChildReach`], supplied by the boot per child) escapes its
/// parent's usable interior, or a child the boot stated NO reach for. A straight-line SHIM — each
/// fallible arm is a monomorphic helper, so the `?` branch regions are covered once here (HR5).
///
/// # Errors
/// [`RegionNestError`] — one variant per malformation above.
/// A shard was booted claiming to be a ROOT while the world says its realm has a parent.
///
/// Carries both sides so the failure names itself: the realm booted, and the parent the world gives it.
#[derive(Clone, Copy, Debug, PartialEq, Eq, thiserror::Error)]
#[error("shard hosts {realm:?}, whose parent is {parent:?}, but booted with a root lineage")]
pub struct LineageNotRooted {
    pub realm: RealmId,
    pub parent: RealmId,
}

/// BOOT FENCE — a realm may not exist without its complete chain to the root.
///
/// The owner's rule: "We can't boot the Realm without parents to the whole root." The only parentless
/// realm is the one true root, so a shard hosting anything else MUST know who contains it.
///
/// WHY THIS IS LOAD-BEARING NOW, and was not before. Leaving a realm used to be a geometric SEARCH: the
/// shard scanned every boundary it knew, ancestors included, and worked out which one an occupant had
/// ended up inside. A shard's declared lineage was a label nothing read, so a stub one was harmless.
/// Leaving is now "am I outside myself? — then hand up to my parent", and a shard's own lineage is the
/// ONLY place it learns who that is (knowing anything about where its ancestors ARE would be the leak
/// this whole model removes). So a shard that wrongly believes it is a root hands occupants to the
/// ambient root instead of to the star system twenty metres away — silently, with nothing crashing.
/// Refusing to start beats misrouting players.
///
/// # Errors
/// [`LineageNotRooted`] if `own_realm` has a parent in `regions` while `coord_parent` is `None`.
pub fn guard_lineage_reaches_root(
    regions: &[RealmRegion],
    own_realm: RealmId,
    coord_parent: Option<RealmId>,
) -> Result<(), LineageNotRooted> {
    let world_parent = regions
        .iter()
        .find(|r| r.realm == own_realm)
        .and_then(|r| r.parent);
    lineage_verdict(own_realm, world_parent, coord_parent)
}

/// The monomorphic core of [`guard_lineage_reaches_root`] — every branch covered once, here (HR5).
fn lineage_verdict(
    realm: RealmId,
    world_parent: Option<RealmId>,
    coord_parent: Option<RealmId>,
) -> Result<(), LineageNotRooted> {
    match (world_parent, coord_parent) {
        // The world gives this realm a parent and the lineage omits it — the state that misroutes.
        (Some(parent), None) => Err(LineageNotRooted { realm, parent }),
        // A true root (no parent either side), or a lineage that names one. Both fine.
        _ => Ok(()),
    }
}

/// The boot fence over a shard's region forest: exactly one root, unique realms, every parent
/// resolvable, every chain reaching the root, and every child geometrically inside its parent at its
/// worst instant.
///
/// **There is NO cap on how many regions a forest may hold (SL9).** One used to exist, because
/// membership was a fixed-width bitset with one bit per region; it is gone with that bitset. A galaxy
/// states a hundred and fifty thousand star systems and this fence judges every one of them.
///
/// # Errors
/// [`RegionNestError`] naming the first violated clause.
pub fn guard_regions_nest(
    regions: &[RealmRegion],
    reaches: &std::collections::BTreeMap<RealmId, ChildReach>,
) -> Result<(), RegionNestError> {
    guard_single_root(regions)?;
    // ★ ONE INDEX, BUILT ONCE, FOR EVERY CLAUSE BELOW (2026-08-31). Each of the next three used to
    // find a parent row by SCANNING the whole forest, so the fence was quadratic four times over.
    // MEASURED on THE world's galaxy: 233 222 regions, and the fence had not finished after ELEVEN
    // MINUTES — a galaxy shard could not boot at all, and the test that forks one leaked the process
    // it gave up on. Building the index also decides uniqueness, so the separate duplicate scan is
    // gone with it (SL9: finding which row holds a realm is a LOOKUP, never a walk).
    let by_realm = index_realms(regions)?;
    guard_parents_resolve(regions, &by_realm)?;
    guard_chains_reach_root(regions, &by_realm)?;
    guard_children_fit_parents(regions, &by_realm, reaches)?;
    guard_bands_clear_the_quantum(regions)?;
    Ok(())
}

/// Every band must be thick enough that the sub-cell residual the integer verdict DISCARDS cannot
/// decide a membership — [`guard_quantum_band`], applied to the forest that is about to boot.
///
/// This is the fence [`region_verdict`] has cited by name since it was written and which did not
/// exist. Its claim held by three and a half orders at the millimetre grid and nothing enforced it, so
/// it would have gone silently false the day a coarser grid lit up, or the day a band was sized down
/// to fit a small body.
fn guard_bands_clear_the_quantum(regions: &[RealmRegion]) -> Result<(), RegionNestError> {
    for r in regions {
        let width_m = r.band.inset() + r.band.outset();
        guard_quantum_band(width_m, r.frame.tier()).map_err(|e| RegionNestError::BandTooThin {
            realm: r.realm,
            width_m: e.width_m,
            cells: e.cells,
        })?;
    }
    Ok(())
}

/// Every child must sit inside its parent's usable interior AT ITS WORST INSTANT — see
/// [`child_fits_in_parent`] for why the ancestry-derived containment prior makes this load-bearing.
/// The boot supplies each child's [`ChildReach`]; a MISSING entry is a typed reject, never a defaulted
/// zero (the same discipline as "decode-to-Default is banned for Durable kinds").
fn guard_children_fit_parents(
    regions: &[RealmRegion],
    by_realm: &std::collections::BTreeMap<RealmId, usize>,
    reaches: &std::collections::BTreeMap<RealmId, ChildReach>,
) -> Result<(), RegionNestError> {
    for child in regions {
        let Some(parent_id) = child.parent else {
            continue; // the ambient root has nothing to fit inside
        };
        // An INVARIANT, not a case to handle: `guard_parents_resolve` ran two checks earlier in
        // `guard_regions_nest` and already refused any region naming a parent that is not present, so a
        // dangling parent cannot reach this line. It used to be written as a `continue`, which read as
        // defensiveness but was in fact an arm no input could take — permanently uncovered, and quietly
        // asserting that the guard above might not have run. Stating the invariant is honest about which
        // it is, and fails loudly if that ordering is ever broken.
        let parent = by_realm
            .get(&parent_id)
            .map(|ix| &regions[*ix])
            .expect("guard_parents_resolve already refused every unresolvable parent");
        let Some(reach) = reaches.get(&child.realm) else {
            return Err(RegionNestError::NoReachForChild { realm: child.realm });
        };
        if let Some((reach, limit)) = child_fits_in_parent(reach, child, parent) {
            return Err(RegionNestError::ChildEscapesParent {
                realm: child.realm,
                parent: parent_id,
                reach,
                limit,
            });
        }
    }
    Ok(())
}

fn guard_single_root(regions: &[RealmRegion]) -> Result<(), RegionNestError> {
    let found = regions.iter().filter(|r| r.parent.is_none()).count();
    if found != 1 {
        return Err(RegionNestError::RootCount { found });
    }
    Ok(())
}

/// Each realm's ROW, by realm — the one lookup table the rest of the fence reads, and the duplicate
/// check itself: a realm that is already in the table is a realm stated twice.
///
/// This replaced a prefix scan per row whose comment read "O(N²) over a bounded N (≤ `max` ≤ 64)".
/// That was honest while a forest held at most 64 regions. SL9 deleted the cap — a galaxy states a
/// hundred and fifty thousand star systems — and the scan never caught up with it.
///
/// # Errors
/// [`RegionNestError::DuplicateRealm`] naming the first realm stated twice, in forest order.
fn index_realms(
    regions: &[RealmRegion],
) -> Result<std::collections::BTreeMap<RealmId, usize>, RegionNestError> {
    let mut by_realm = std::collections::BTreeMap::new();
    for (ix, r) in regions.iter().enumerate() {
        if by_realm.insert(r.realm, ix).is_some() {
            return Err(RegionNestError::DuplicateRealm { realm: r.realm });
        }
    }
    Ok(by_realm)
}

fn guard_parents_resolve(
    regions: &[RealmRegion],
    by_realm: &std::collections::BTreeMap<RealmId, usize>,
) -> Result<(), RegionNestError> {
    for r in regions {
        if let Some(p) = r.parent
            && !by_realm.contains_key(&p)
        {
            return Err(RegionNestError::DanglingParent {
                realm: r.realm,
                parent: p,
            });
        }
    }
    Ok(())
}

/// Every region's parent chain must reach the single root. Runs AFTER the single-root / index /
/// resolve guards, so the `.parent` lookups are total. Never hangs on a cycle.
///
/// ★ EACH REALM IS PROVEN ONCE (2026-08-31). Every region used to re-walk its whole chain from
/// scratch, and each hop of that walk SCANNED the forest for the next row. On a galaxy every one of
/// 233 220 star systems repeated the same two-hop climb to the root. The realms already proven carry
/// forward instead, so a chain stops at the first realm whose answer is known.
fn guard_chains_reach_root(
    regions: &[RealmRegion],
    by_realm: &std::collections::BTreeMap<RealmId, usize>,
) -> Result<(), RegionNestError> {
    let mut proven: std::collections::BTreeSet<RealmId> = std::collections::BTreeSet::new();
    for r in regions {
        if !chain_reaches_root(regions, by_realm, r.realm, &mut proven) {
            return Err(RegionNestError::CycleOrOrphan { realm: r.realm });
        }
    }
    Ok(())
}

/// One realm's climb. Every realm on a successful climb is proven with it — they share its tail.
///
/// The hop budget is gone: standing on a realm the climb has ALREADY passed through is what a cycle
/// is, and saying so directly reads as the thing it detects. The old form counted hops against the
/// forest size, which was the same verdict written as arithmetic — and it re-derived that verdict for
/// every region.
fn chain_reaches_root(
    regions: &[RealmRegion],
    by_realm: &std::collections::BTreeMap<RealmId, usize>,
    start: RealmId,
    proven: &mut std::collections::BTreeSet<RealmId>,
) -> bool {
    let mut climbed: Vec<RealmId> = Vec::new();
    let mut cur = start;
    loop {
        if proven.contains(&cur) {
            proven.extend(climbed); // this chain joins one already known to reach the root
            return true;
        }
        if climbed.contains(&cur) {
            return false; // back where this climb already stood ⇒ a cycle
        }
        climbed.push(cur);
        match by_realm.get(&cur).and_then(|ix| regions[*ix].parent) {
            None => {
                proven.extend(climbed); // reached the ambient root (`parent: None`)
                return true;
            }
            Some(p) => cur = p,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;

    /// The verdict asked about ONE INSTANT — a degenerate segment, which is what every caller that
    /// holds a single pose passes. Every pre-existing membership test below reaches the rule through
    /// this alias, carrying the literal expected values it was written with, so the whole suite is a
    /// regression gate on the claim that a subject which has not moved gets exactly the shipped answer.
    fn point_verdict(
        pose: &StampedPose,
        region: &RealmRegion,
        book: &PlacementBook,
        was_member: bool,
    ) -> Result<RegionVerdict, FrameError> {
        region_verdict(pose, pose.pos, region, book, was_member)
    }

    // ======================= SLICE 5 — THE SWEPT MEMBERSHIP GATE =======================
    //
    // THE PLAN'S GATE AS WRITTEN IS FALSE, AND THAT IS RECORDED HERE RATHER THAN QUIETLY DROPPED.
    // It asked that "for every subject whose per-tick travel is below the band, the swept verdict
    // equals the point verdict, bit for bit". Two cells of travel — 1.95 mm, three orders below the
    // shipped 3 m band — genuinely clips a region at every size: `a = (E,−1,0) → b = (E,+1,0)` has
    // both endpoints outside and its midpoint exactly on the surface. Written as an equality that
    // gate would be RED for a correct implementation, and the cheapest way to make it green would be
    // to weaken the implementation. It is replaced below by four arms that can still fail:
    //   ARM 1  a subject that has not moved gets the shipped answer, bit for bit;
    //   ARM 2  one cell of travel per axis can never disagree — proved, then checked exhaustively;
    //   ARM 3  every disagreement obeys an exactly-tight integer envelope;
    //   ARM 4  an independent oracle carrying no floats at all.

    /// The point rule written from the SPECIFICATION rather than copied from the shipped body: edge
    /// is `r + outset` when held and `r − inset` when not, floored to acquire and ceiled to release,
    /// compared as squared cell lengths. A second transcription, so a typo in one is not a typo in
    /// both.
    fn frozen_point_member(
        sep: Separation,
        region: &RealmRegion,
        was_member: bool,
        tier: Tier,
    ) -> bool {
        let edge_of = |m: f64| -> i64 {
            let c = m / tier.cell_edge_m();
            if was_member {
                c.ceil() as i64
            } else {
                c.floor() as i64
            }
        };
        let widen = |m: f64| {
            if was_member {
                m + region.band.outset()
            } else {
                m - region.band.inset()
            }
        };
        let c = sep.cells();
        match region.shape {
            Boundary::Shell { r } => {
                let e = edge_of(widen(r));
                if e < 0 {
                    return false;
                }
                let sq = i128::from(c.x) * i128::from(c.x)
                    + i128::from(c.y) * i128::from(c.y)
                    + i128::from(c.z) * i128::from(c.z);
                sq <= i128::from(e) * i128::from(e)
            }
            Boundary::Aabb { half } => {
                // Bitwise, never `&&`: a short-circuit leaves the tail axes uncoverable from a
                // false left-hand side — this crate's own written discipline, which the first draft
                // of this very helper broke and the coverage gate caught.
                let ax = |d: i64, h: f64| {
                    let e = edge_of(widen(h));
                    (e >= 0) & (i128::from(d).abs() <= i128::from(e))
                };
                ax(c.x, half.x) & ax(c.y, half.y) & ax(c.z, half.z)
            }
            // THE ROTATED ARM IS UNTOUCHED BY THIS SLICE, and the fixture proves it by including one:
            // the reference states the decimal rule directly, and the identity gate then asserts that
            // a degenerate segment reaches exactly that and nothing new.
            Boundary::Obb { .. } => region
                .band
                .member(was_member, region.shape.signed_distance(sep.metres())),
        }
    }

    fn sep_of(pose: &StampedPose) -> Separation {
        pose.pos.separation(LatticePos::ORIGIN, Tier::Fine)
    }

    // ---------- ARM 1: a subject that has not moved gets the shipped answer ----------

    // ---------- SLICE S6: THE QUANTUM FENCE, WHICH WAS CITED BY NAME FOR MONTHS AND DID NOT EXIST ----

    #[test]
    fn a_band_thinner_than_the_grid_it_is_measured_on_is_refused() {
        // Both arms, and the boundary between them exactly. One cell under the requirement refuses;
        // exactly at it passes — an inclusive edge, stated by driving it rather than by a comment.
        let edge = Tier::Fine.cell_edge_m();
        let exactly = QUANTUM_BAND_CELLS * edge;
        assert_eq!(guard_quantum_band(exactly, Tier::Fine), Ok(()));
        assert_eq!(
            guard_quantum_band(exactly - edge, Tier::Fine).expect_err("one cell short is refused"),
            BandTooThin {
                width_m: exactly - edge,
                cells: QUANTUM_BAND_CELLS - 1.0,
            }
        );
        // The shipped floor clears it with room to spare, which is the claim the verdict's own doc
        // makes and which nothing checked until now.
        assert_eq!(guard_quantum_band(3.0, Tier::Fine), Ok(()));
        // A zero or negative width is thinner than anything and must never read as acceptable.
        assert!(guard_quantum_band(0.0, Tier::Fine).is_err());
        assert!(guard_quantum_band(-1.0, Tier::Fine).is_err());
    }

    #[test]
    fn the_boot_fence_refuses_a_forest_whose_band_is_too_thin_for_its_grid() {
        // The fence reaches the forest, not just the helper. A lawful forest passes; the same forest
        // with one band thinned below the grid is refused BY NAME.
        let root = test_region_r(RealmId::System(0), None, 1_000.0);
        let mut child = test_region_r(RealmId::Planet(1), Some(RealmId::System(0)), 10.0);
        let reaches = std::collections::BTreeMap::from([
            (RealmId::System(0), ChildReach::Fixed(DVec3::ZERO)),
            (RealmId::Planet(1), ChildReach::Fixed(DVec3::ZERO)),
        ]);
        assert_eq!(guard_regions_nest(&[root, child], &reaches), Ok(()));

        // A band of one millimetre is about one cell at this grid — far under the requirement.
        child.band = ContainmentBand::for_containment_velocity_safe(0.0004, 0.0006, 0.0, 1.0, 0.0)
            .expect("a thin band is still a valid band");
        let err = guard_regions_nest(&[root, child], &reaches).expect_err("a thin band is refused");
        assert_eq!(
            err,
            RegionNestError::BandTooThin {
                realm: RealmId::Planet(1),
                width_m: 0.001,
                cells: 0.001 / Tier::Fine.cell_edge_m(),
            }
        );
    }

    #[test]
    fn a_subject_that_has_not_moved_gets_the_shipped_point_answer_on_every_fixture() {
        let shell = test_region_r(RealmId::Planet(1), Some(RealmId::System(0)), 10.0);
        let boxed = RealmRegion {
            shape: Boundary::Aabb {
                half: DVec3::new(4.0, 6.0, 8.0),
            },
            look: Some(Boundary::Aabb {
                half: DVec3::new(4.0, 6.0, 8.0),
            }),
            ..test_region(RealmId::Station(2), Some(RealmId::System(0)))
        };
        // A region THINNER THAN ITS OWN INSET, whose acquire edge is negative. It is never
        // acquirable, at any distance, by either rule — and it is the fixture that drives the
        // negative-edge arm of both this reference and the shipped verdict.
        let thin = test_region_r(RealmId::Station(3), Some(RealmId::System(0)), 0.5);
        // THE ROTATED SHAPE, which this slice deliberately leaves on the decimal point rule. It is in
        // the fixture so that "the swept change does not touch it" is a measurement, not an omission.
        let turned = RealmRegion {
            shape: Boundary::Obb {
                half: DVec3::new(4.0, 6.0, 8.0),
                orient: DQuat::from_rotation_z(0.7),
            },
            look: None,
            ..test_region(RealmId::Station(4), Some(RealmId::System(0)))
        };
        let mut checked = 0u32;
        for region in [&shell, &boxed, &thin, &turned] {
            let book = verdict_book(region.frame);
            for m in [
                0.0, 1.0, 3.9, 4.0, 4.1, 8.5, 9.0, 9.5, 10.0, 11.5, 12.0, 12.5, 13.0, 1.0e3, 1.0e6,
                1.0e12,
            ] {
                for v in [
                    DVec3::new(m, 0.0, 0.0),
                    DVec3::new(0.0, m, 0.0),
                    DVec3::new(0.0, 0.0, m),
                    DVec3::new(m, m, m),
                ] {
                    for was in [false, true] {
                        let pose = pose_at(v);
                        let got =
                            region_verdict(&pose, pose.pos, region, &book, was).expect("placeable");
                        assert_eq!(
                            got.member,
                            frozen_point_member(sep_of(&pose), region, was, Tier::Fine),
                            "degenerate segment at {v:?} (was_member = {was}) must be the point answer"
                        );
                        checked += 1;
                    }
                }
            }
        }
        // The gate must have run. A fixture matrix that silently produced nothing would assert
        // nothing, and would look exactly like a pass.
        assert_eq!(checked, 4 * 16 * 4 * 2);
    }

    // ---------- ARM 2: one cell of travel can never disagree ----------

    #[test]
    fn one_cell_of_travel_per_axis_can_never_disagree_with_the_point_rule() {
        // This is what makes the "no prior" short-circuit LOSSLESS rather than merely cheap: two
        // normalized positions whose cells are equal differ by under one cell per axis, so the
        // reframed segment spans at most one cell per axis. A shell disagreement needs |d|² ≥ 4
        // (both endpoints outside forces m² ≥ E²+1, while 4(m² − E²) ≤ |d|²) and a box disagreement
        // needs |d_i| ≥ 2 on some axis. At one cell per axis |d|² ≤ 3, so neither is reachable.
        let mut shell_pairs = 0u64;
        let mut box_pairs = 0u64;
        for e in 0..=5i64 {
            let span = e + 2;
            for ax in -span..=span {
                for ay in -span..=span {
                    for az in -span..=span {
                        let a = I64Vec3::new(ax, ay, az);
                        for dx in -1..=1i64 {
                            for dy in -1..=1i64 {
                                for dz in -1..=1i64 {
                                    let b = a + I64Vec3::new(dx, dy, dz);
                                    let out_sphere =
                                        |p: I64Vec3| p.x * p.x + p.y * p.y + p.z * p.z > e * e;
                                    let out_box =
                                        |p: I64Vec3| p.x.abs().max(p.y.abs()).max(p.z.abs()) > e;
                                    if out_sphere(a) && out_sphere(b) {
                                        shell_pairs += 1;
                                        assert!(
                                            !shell_segment_dips(a, b, e),
                                            "shell: {a:?} -> {b:?} at edge {e} must not dip"
                                        );
                                    }
                                    if out_box(a) && out_box(b) {
                                        box_pairs += 1;
                                        assert!(
                                            !box_segment_dips(a, b, [e, e, e]),
                                            "box: {a:?} -> {b:?} at edge {e} must not dip"
                                        );
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
        // Pinned counts, so a loop that stopped enumerating would be caught rather than read as a
        // pass. These are the number of both-outside pairs the sweep actually saw.
        assert_eq!(shell_pairs, 187_008);
        assert_eq!(box_pairs, 135_492);
    }

    #[test]
    fn the_one_cell_bound_is_attained_so_it_is_tight_and_not_padded() {
        // Exactly two cells of travel DOES disagree — the minimum divergent step. If this ever went
        // quiet, the arm above would be proving a vacuous bound.
        assert!(shell_segment_dips(
            I64Vec3::new(-1, 0, 0),
            I64Vec3::new(1, 0, 0),
            0
        ));
        assert!(box_segment_dips(
            I64Vec3::new(-1, 0, 0),
            I64Vec3::new(1, 0, 0),
            [0, 0, 0]
        ));
        // And one cell either side of it does not.
        assert!(!shell_segment_dips(
            I64Vec3::new(-1, 0, 0),
            I64Vec3::new(0, 0, 0),
            0
        ));
    }

    // ---------- ARM 3: the divergence envelope ----------

    #[test]
    fn every_disagreement_obeys_the_exactly_tight_integer_envelope() {
        // Whenever the sweep adds a member the point rule refused, the segment can only have reached
        // inside by at most the sagitta of a chord of its own length. In integers, with no square
        // root and no float: 4·(min(|a|²,|b|²) − E²) ≤ |d|².
        let mut dips = 0u64;
        let mut attained = false;
        for e in [0i64, 1, 7, 64, 1_000, 5_120, 12_288] {
            for k in 1..=40i64 {
                for off in -2..=2i64 {
                    let a = I64Vec3::new(e + off, -k, 0);
                    let b = I64Vec3::new(e + off, k, 0);
                    let sq = |p: I64Vec3| {
                        i128::from(p.x) * i128::from(p.x)
                            + i128::from(p.y) * i128::from(p.y)
                            + i128::from(p.z) * i128::from(p.z)
                    };
                    let e2 = i128::from(e) * i128::from(e);
                    if (sq(a) <= e2) | (sq(b) <= e2) {
                        continue;
                    }
                    if !shell_segment_dips(a, b, e) {
                        continue;
                    }
                    dips += 1;
                    let d = b - a;
                    let dd = sq(I64Vec3::new(d.x, d.y, d.z));
                    let m2 = sq(a).min(sq(b));
                    assert!(
                        4 * (m2 - e2) <= dd,
                        "envelope violated at {a:?} -> {b:?}, edge {e}"
                    );
                    if 4 * (m2 - e2) == dd {
                        attained = true;
                    }
                }
            }
        }
        assert!(
            dips > 100,
            "the envelope arm must have seen real dips: {dips}"
        );
        // EXACTLY TIGHT. If the bound were padded this would never fire, and the arm would be
        // asserting something weaker than it claims.
        assert!(
            attained,
            "the envelope bound must be attained, not merely respected"
        );
    }

    // ---------- ARM 4: an independent oracle, with no floats ----------

    #[test]
    fn an_independent_lattice_oracle_agrees_at_every_magnitude_up_to_the_domain() {
        // Built on the lattice so it cannot degrade the way a decimal oracle does. For Q = (ρ,0,0)
        // and the perpendicular integer direction (0,1,0), the segment Q−k·u → Q+k·u has its nearest
        // point exactly at Q for every k ≥ 1, so the exact answer is (ρ ≤ E) — no arithmetic, no
        // rounding, nothing shared with the implementation.
        let mut cases = 0u32;
        for e in [
            1i64,
            7,
            5_120,
            12_288,
            6_523_904_000,
            360_960_000_000_000,
            2_302_768_551_868_075_776,
        ] {
            for rho in [e - 2, e - 1, e, e + 1, e + 2] {
                for k in [
                    1i64,
                    2,
                    3,
                    1_000,
                    1_000_000,
                    1_000_000_000,
                    1_000_000_000_000,
                    1_049_400_000_000_000_000,
                ] {
                    // Every construction here is IN DOMAIN by inspection: the widest edge is
                    // 0.4993 of the domain bound and the longest step 0.2276 of it. There is
                    // deliberately no guard, because a guard nothing can drive is a branch nothing
                    // can cover.
                    let a = I64Vec3::new(rho, -k, 0);
                    let b = I64Vec3::new(rho, k, 0);
                    // THE HELPER'S STATED PRECONDITION: both endpoints already point-outside. A few
                    // of these constructions put an endpoint inside, where the documented contract is
                    // a total `false` rather than the geometric answer. They belong to the totality
                    // test, not to the oracle.
                    let outside = i128::from(rho) * i128::from(rho) + i128::from(k) * i128::from(k)
                        > i128::from(e) * i128::from(e);
                    if !outside {
                        continue;
                    }
                    assert_eq!(
                        shell_segment_dips(a, b, e),
                        rho <= e,
                        "oracle disagrees: rho = {rho}, edge = {e}, k = {k}"
                    );
                    cases += 1;
                }
            }
        }
        // PINNED, not a threshold: 280 constructions, 49 of which put an endpoint inside the edge
        // and belong to the totality arm instead. A loop that quietly stopped enumerating would be
        // caught here rather than read as a pass.
        assert_eq!(cases, 231);
    }

    #[test]
    fn the_wide_multiply_is_exact_against_a_different_limb_split() {
        // The oracle uses 32-bit limbs, so it is not the implementation restated. Plus the cheap
        // invariant that the low half is the wrapping product.
        let oracle = |x: u128, y: u128| -> (u128, u128) {
            const M: u128 = u32::MAX as u128;
            let xs = [x & M, (x >> 32) & M, (x >> 64) & M, (x >> 96) & M];
            let ys = [y & M, (y >> 32) & M, (y >> 64) & M, (y >> 96) & M];
            let mut acc = [0u128; 8];
            for (i, &xi) in xs.iter().enumerate() {
                for (j, &yj) in ys.iter().enumerate() {
                    acc[i + j] += xi * yj;
                }
            }
            // Normalize the 32-bit limb carries, then reassemble.
            let mut carry = 0u128;
            let mut limbs = [0u128; 8];
            for (n, slot) in limbs.iter_mut().enumerate() {
                let v = acc[n] + carry;
                *slot = v & M;
                carry = v >> 32;
            }
            let lo = limbs[0] | (limbs[1] << 32) | (limbs[2] << 64) | (limbs[3] << 96);
            let hi = limbs[4] | (limbs[5] << 32) | (limbs[6] << 64) | (limbs[7] << 96);
            (hi, lo)
        };
        for (x, y) in [
            (0u128, 0u128),
            (1, 1),
            (u128::MAX, u128::MAX),
            (1 << 127, 1 << 127),
            (u64::MAX as u128, u64::MAX as u128),
            (
                3 * 4_611_686_018_427_387_903u128 * 4_611_686_018_427_387_903u128,
                1,
            ),
            (
                63_802_943_624_105_984_887_243_264_331_513_856_000,
                255_211_774_496_423_939_548_973_057_326_055_424_012,
            ),
        ] {
            assert_eq!(wide_mul(x, y), oracle(x, y), "wide multiply at {x} × {y}");
            assert_eq!(wide_mul(x, y).1, x.wrapping_mul(y));
        }
        // The 256-bit comparator, both directions and the tie.
        assert!(le256((1, 0), (1, 0)));
        assert!(le256((0, u128::MAX), (1, 0)));
        assert!(!le256((1, 0), (0, u128::MAX)));
        assert!(le256((1, 5), (1, 6)));
        assert!(!le256((1, 7), (1, 6)));
    }

    // ---------- THE HEADLINE: what the slice exists for ----------

    #[test]
    fn a_subject_travelling_past_a_whole_child_in_one_tick_still_acquires_it() {
        // The plan's headline: "a subject travelling one tick further than a child's whole diameter
        // still acquires that child". Before this slice acquisition needed a SAMPLE landing at least
        // one inset inside the surface, so no widening of any band could ever buy it.
        let region = test_region_r(RealmId::Planet(1), Some(RealmId::System(0)), 10.0);
        let book = verdict_book(region.frame);
        let before = pose_at(DVec3::new(-1_000.0, 0.0, 0.0));
        let after = pose_at(DVec3::new(1_000.0, 0.0, 0.0));
        // 2 km of travel across a 20 m child — a hundred diameters in one tick.
        assert!(
            !region_verdict(&after, after.pos, &region, &book, false)
                .expect("placeable")
                .member,
            "the point rule cannot see this crossing — that is the defect"
        );
        assert!(
            region_verdict(&after, before.pos, &region, &book, false)
                .expect("placeable")
                .member,
            "the swept rule must see it"
        );
        // AND IT IS NOT A BLANKET YES: a parallel pass that misses by a metre stays outside.
        let miss_a = pose_at(DVec3::new(-1_000.0, 11.0, 0.0));
        let miss_b = pose_at(DVec3::new(1_000.0, 11.0, 0.0));
        assert!(
            !region_verdict(&miss_b, miss_a.pos, &region, &book, false)
                .expect("placeable")
                .member,
            "a pass outside the acquire edge must stay outside"
        );
    }

    #[test]
    fn leaving_a_region_still_releases_on_the_tick_it_leaves() {
        // The one thing a swept rule must NOT do: hold a subject that has gone. At real speeds a
        // held subject can be thousands of radii outside, so a one-tick lag here would invert
        // membership at exactly the speeds this machinery exists for.
        let region = test_region_r(RealmId::Planet(1), Some(RealmId::System(0)), 10.0);
        let book = verdict_book(region.frame);
        let inside = pose_at(DVec3::new(0.0, 0.0, 0.0));
        let gone = pose_at(DVec3::new(1_000.0, 0.0, 0.0));
        assert!(
            !region_verdict(&gone, inside.pos, &region, &book, true)
                .expect("placeable")
                .member,
            "a subject whose prior was inside and whose sample is outside must release"
        );
    }

    // ---------- THE FLAP CONDITION, MEASURED RATHER THAN ARGUED ----------

    #[test]
    fn a_swept_acquire_holds_below_the_band_and_flaps_above_it() {
        // Acquiring a child and leaving it within one tick issues a re-home the next tick reverses.
        // That is not cured here — it is cured by bands sized from real closing speed, which is the
        // next slice. What IS established here is the exact condition, as a measurement that could
        // have come out either way: the hysteresis damps the flap while one step of travel stays
        // inside the band, and stops damping it above.
        let region = test_region_r(RealmId::Planet(1), Some(RealmId::System(0)), 10.0);
        let book = verdict_book(region.frame);
        let band_m = region.band.inset() + region.band.outset(); // 3 m on this fixture
        assert_eq!(band_m, 3.0);

        // A step INSIDE the band: graze the surface, acquire on the sweep, and still be held next
        // tick because the release edge sits a whole outset further out.
        let step = 2.0_f64;
        let graze_a = pose_at(DVec3::new(9.0, -step / 2.0, 0.0));
        let graze_b = pose_at(DVec3::new(9.0, step / 2.0, 0.0));
        let acquired = region_verdict(&graze_b, graze_a.pos, &region, &book, false)
            .expect("placeable")
            .member;
        assert!(acquired, "a graze inside the acquire edge must acquire");
        let next = pose_at(DVec3::new(9.0, step / 2.0 + step, 0.0));
        assert!(
            region_verdict(&next, graze_b.pos, &region, &book, true)
                .expect("placeable")
                .member,
            "below the band, one more step must NOT release — this is what stops the flap today"
        );

        // A step FAR ABOVE the band: acquire on the sweep, gone on the very next tick. The flap.
        let fast_a = pose_at(DVec3::new(-1_000.0, 0.0, 0.0));
        let fast_b = pose_at(DVec3::new(1_000.0, 0.0, 0.0));
        assert!(
            region_verdict(&fast_b, fast_a.pos, &region, &book, false)
                .expect("placeable")
                .member,
            "the fast crossing acquires"
        );
        let fast_c = pose_at(DVec3::new(3_000.0, 0.0, 0.0));
        assert!(
            !region_verdict(&fast_c, fast_b.pos, &region, &book, true)
                .expect("placeable")
                .member,
            "and releases the next tick — the one-tick fly-through, which the band slice cures"
        );
    }

    // ---------- EVERY REFUSAL RETURNS THE SHIPPED ANSWER ----------

    #[test]
    fn a_region_thinner_than_its_own_inset_stays_never_acquirable_even_swept() {
        // A negative edge must never be squared into a positive one.
        assert!(!shell_segment_dips(
            I64Vec3::new(1, 0, 0),
            I64Vec3::new(-1, 0, 0),
            -5
        ));
        assert!(!box_segment_dips(
            I64Vec3::new(1, 0, 0),
            I64Vec3::new(-1, 0, 0),
            [-5, 1, 1]
        ));
    }

    #[test]
    fn an_out_of_domain_endpoint_refuses_the_upgrade_rather_than_squaring_it() {
        // The same convention a separation too large to square already uses: refuse, never panic,
        // and never upgrade. Reachable from a lawful subject and region whose frames sit far apart.
        let far = I64Vec3::new(CELL_DOMAIN_MAX, 0, 0);
        let beyond = I64Vec3::new(CELL_DOMAIN_MAX / 2, 0, 0) + far;
        assert!(!shell_segment_dips(beyond, -far, 1_000));
        assert!(!box_segment_dips(beyond, -far, [1_000, 1_000, 1_000]));
        // And exactly AT the domain edge it still answers, so the guard is a boundary and not a wall.
        assert!(shell_segment_dips(far, -far, 1_000));
    }

    #[test]
    fn the_dip_predicate_is_total_when_its_precondition_is_violated() {
        // The interior-minimum step subtracts the squared edge from the squared distance. Reaching
        // it with an endpoint INSIDE the edge violates the caller's precondition, and the contract
        // is that the helper answers `false` rather than panicking. VERIFIED BY PLANTING IT: with a
        // bare subtraction this test goes red on a debug overflow. A WRAPPING subtraction is not
        // caught, and cannot be by any input — see the proof beside the guard itself.
        assert!(!shell_segment_dips(
            I64Vec3::new(1, 0, 0),
            I64Vec3::new(-1, 0, 0),
            5
        ));
    }

    #[test]
    fn a_receding_segment_and_a_zero_length_one_take_the_same_early_exit() {
        // The nearest point is the start, which the caller already knows is outside. A zero-length
        // segment lands in the same arm, so there is no separate degenerate branch to leave undriven.
        assert!(!shell_segment_dips(
            I64Vec3::new(10, 0, 0),
            I64Vec3::new(20, 0, 0),
            5
        ));
        assert!(!shell_segment_dips(
            I64Vec3::new(10, 0, 0),
            I64Vec3::new(10, 0, 0),
            5
        ));
        // The nearest point is the END, also known outside.
        assert!(!shell_segment_dips(
            I64Vec3::new(20, 0, 0),
            I64Vec3::new(10, 0, 0),
            5
        ));
    }

    #[test]
    fn the_box_sweep_separates_on_each_cross_axis_and_clips_a_true_corner() {
        // Each of the three cross axes must be the DECIDING one in at least one case, or an axis
        // could be transposed and nothing would notice.
        let h = [10i64, 10, 10];
        // A diagonal near-miss past a corner in each of the three coordinate planes.
        // Each cuts a plane at distance 21.2 from the centre, past the corner at (10,10) whose own
        // reach on that axis is 14.1 — a true miss, and one the FACE test cannot decide, because
        // every axis range straddles the box.
        for (name, a, b) in [
            ("z decides", I64Vec3::new(30, 0, 0), I64Vec3::new(0, 30, 0)),
            ("x decides", I64Vec3::new(0, 30, 0), I64Vec3::new(0, 0, 30)),
            ("y decides", I64Vec3::new(30, 0, 0), I64Vec3::new(0, 0, 30)),
        ] {
            assert!(
                !box_segment_dips(a, b, h),
                "{name}: {a:?} -> {b:?} must miss"
            );
        }
        // And a true clip through the middle, on each axis.
        assert!(box_segment_dips(
            I64Vec3::new(-30, 0, 0),
            I64Vec3::new(30, 0, 0),
            h
        ));
        assert!(box_segment_dips(
            I64Vec3::new(0, -30, 0),
            I64Vec3::new(0, 30, 0),
            h
        ));
        assert!(box_segment_dips(
            I64Vec3::new(0, 0, -30),
            I64Vec3::new(0, 0, 30),
            h
        ));
        // A face miss: both endpoints beyond the same face, on each axis.
        assert!(!box_segment_dips(
            I64Vec3::new(30, -30, 0),
            I64Vec3::new(30, 30, 0),
            h
        ));
        assert!(!box_segment_dips(
            I64Vec3::new(-30, 30, 0),
            I64Vec3::new(30, 30, 0),
            h
        ));
        assert!(!box_segment_dips(
            I64Vec3::new(0, -30, 30),
            I64Vec3::new(0, 30, 30),
            h
        ));
    }

    #[test]
    fn a_box_half_edge_wider_than_the_domain_is_clamped_without_changing_the_answer() {
        // Past the domain guard every coordinate is inside ±CDM, so a half-edge wider than CDM
        // already contains every reachable point: clamping leaves the intersection identical and
        // restores a factor of two of headroom in the radius term.
        for (a, b) in [
            (
                I64Vec3::new(CELL_DOMAIN_MAX, 1, 0),
                I64Vec3::new(-CELL_DOMAIN_MAX, -1, 0),
            ),
            (
                I64Vec3::new(CELL_DOMAIN_MAX, CELL_DOMAIN_MAX, 0),
                I64Vec3::new(-CELL_DOMAIN_MAX, CELL_DOMAIN_MAX, 0),
            ),
            (I64Vec3::new(7, 9, 11), I64Vec3::new(-7, -9, -11)),
        ] {
            let at_domain = box_segment_dips(a, b, [CELL_DOMAIN_MAX; 3]);
            for wider in [CELL_DOMAIN_MAX + 1, 2 * CELL_DOMAIN_MAX, i64::MAX] {
                assert_eq!(
                    box_segment_dips(a, b, [wider; 3]),
                    at_domain,
                    "clamping a half-edge of {wider} must not change the answer for {a:?} -> {b:?}"
                );
            }
        }
    }

    #[test]
    fn the_per_axis_box_rule_is_total_at_the_integer_floor() {
        // `i64::MIN.abs()` panics in debug — which is the whole test and coverage suite.
        assert!(!axis_member(i64::MIN, 10));
        assert!(!axis_member(0, -1));
        assert!(axis_member(10, 10));
        assert!(!axis_member(11, 10));
        assert!(axis_member(-10, 10));
    }

    #[test]
    fn a_prior_the_book_cannot_place_degrades_to_the_point_answer() {
        // A rotated destination placement refuses beyond the exact rotation reach, and that refusal
        // depends on the separation MAGNITUDE — so it can refuse the prior while accepting the
        // current pose. The verdict must then be the point answer, never an error.
        let region = RealmRegion {
            shape: Boundary::Shell { r: 10.0 },
            look: Some(Boundary::Shell { r: 10.0 }),
            ..test_region(RealmId::Planet(1), Some(RealmId::System(0)))
        };
        let rotated = crate::frame::FramePlacement {
            orientation: DQuat::from_rotation_z(0.5),
            ..crate::frame::FramePlacement::identity()
        };
        let book = crate::placement::PlacementBook::new(
            FrameRef::SystemSpace { system_seed: 0 },
            crate::ids::UniverseTick(0),
            vec![(region.frame, rotated)],
        );
        let near = pose_at(DVec3::new(1.0, 0.0, 0.0));
        // A prior far beyond the exact rotation reach for the Fine tier (2^42 m).
        let far_prior = LatticePos::from_metres(DVec3::new(1.0e14, 0.0, 0.0), Tier::Fine);
        let got = region_verdict(&near, far_prior, &region, &book, false).expect("placeable");
        assert!(
            got.member,
            "the current pose is inside; an unplaceable prior must not take that away"
        );
    }

    // ----- THE ONE visibility formula (look_horizon.md §3.3.2, slice 2) -----

    #[test]
    fn a_band_re_banded_at_a_stated_reach_keeps_its_ratio_or_its_gap_and_an_inert_band_stays_inert()
    {
        // Boot band: 2 000 in, 4 000 out (ratio 2, gap 2 000).
        let boot =
            AoiConfig::for_velocity_safe(1_000.0, 2.0, 4.0, 0.0, 0.05, 3, 0.0).expect("live");
        // A wider reach: the ratio wins (10 000 → 20 000 out; the gap alone would give 12 000).
        let wide = boot.with_spin_up(10_000.0);
        assert_eq!(
            (wide.spin_up_r_m(), wide.tear_down_r_m()),
            (10_000.0, 20_000.0)
        );
        assert_eq!(wide.grace_ticks(), 3);
        // A narrower reach: the gap wins (500 → 2 500 out; the ratio alone would give 1 000).
        let narrow = boot.with_spin_up(500.0);
        assert_eq!(
            (narrow.spin_up_r_m(), narrow.tear_down_r_m()),
            (500.0, 2_500.0)
        );
        // A non-positive reach changes nothing; an inert band stays inert.
        assert_eq!(boot.with_spin_up(0.0), boot);
        assert_eq!(AoiConfig::inert().with_spin_up(5_000.0), AoiConfig::inert());
        // The dot angle is the one the world solve and the reach share.
        assert!((VISIBILITY_THETA_MIN_RAD - 1.5_f64.to_radians()).abs() < 1e-5);
    }

    #[test]
    fn the_drawable_angle_is_one_pixel_at_the_reference_view() {
        let theta = drawable_theta_min_rad();
        assert!(
            (theta - 1.1506e-3).abs() < 1e-6,
            "2·tan(22.5°)/720 = {theta}"
        );
        // `extent` is the circumscribed extent — the radius: a figure one metre ACROSS is 0.5.
        let one_metre = visibility_reach_m(0.5, theta);
        assert!(
            (one_metre - 869.1).abs() < 1.0,
            "a one-metre figure: {one_metre} m"
        );
        let two_metres = visibility_reach_m(1.0, theta);
        assert!(
            (two_metres - 1738.3).abs() < 2.0,
            "a two-metre character: {two_metres} m"
        );
        assert!(
            theta < VISIBILITY_THETA_MIN_RAD,
            "a row is cheaper than a shard"
        );
    }

    #[test]
    fn the_visibility_formula_is_cot_half_theta_and_the_reach_is_its_other_spelling() {
        // cot(θ/2) with θ = π/2 is cot(π/4) — the measured f64 value of the identity (tan(π/4)
        // rounds a hair above 1, so the factor lands a hair below — pinned as measured, never
        // assumed to be exactly 1).
        let factor = visibility_factor(std::f64::consts::FRAC_PI_2);
        assert_eq!(factor, 1.0 / (std::f64::consts::FRAC_PI_4).tan());
        assert!((factor - 1.0).abs() < 1.0e-15);
        // The reach IS extent × the factor — the same formula's other spelling, bit-for-bit.
        assert_eq!(
            visibility_reach_m(150.0, std::f64::consts::FRAC_PI_2),
            150.0 * factor
        );
        // Monotone the right way: a smaller minimum angle sees farther.
        assert!(visibility_factor(0.026_180) > visibility_factor(0.14));
    }

    // ----- Containment realm-membership (task #135, C-1) -----

    #[test]
    fn containment_band_member_acquires_inside_and_holds_until_outside() {
        // v_rel = 0 ⇒ need = 0 ⇒ outset = max(3, 0 - 2) = 3; inset = 2.
        let b = ContainmentBand::for_containment_velocity_safe(2.0, 3.0, 0.0, 0.1, 0.0)
            .expect("valid band");
        assert_eq!(b.inset(), 2.0);
        assert_eq!(b.outset(), 3.0);
        // NOT yet a member: acquire ONLY at least `inset` inside (signed_distance <= -inset = -2).
        assert!(
            !b.member(false, 0.0),
            "on the surface: not inside enough to acquire"
        );
        assert!(
            !b.member(false, -1.9),
            "1.9 m inside is short of the 2 m acquire edge"
        );
        assert!(b.member(false, -2.0), "exactly `inset` inside acquires");
        assert!(b.member(false, -5.0), "deep inside acquires");
        // ALREADY a member: hold until MORE than `outset` outside (signed_distance > +3 releases).
        assert!(
            b.member(true, 0.0),
            "a member on the surface stays (dead-zone straddles it)"
        );
        assert!(
            b.member(true, 3.0),
            "a member exactly at the outset edge stays"
        );
        assert!(!b.member(true, 3.1), "past the outset edge releases");
    }

    #[test]
    fn containment_band_velocity_widens_the_outset() {
        // need = |v_rel|·dt·(K_SAFETY + extra) = 10 · 1 · (2 + 0) = 20; outset = max(1, 20 - 2) = 18.
        let b = ContainmentBand::for_containment_velocity_safe(2.0, 1.0, 10.0, 1.0, 0.0)
            .expect("valid band");
        assert_eq!(b.inset(), 2.0);
        assert_eq!(b.outset(), 18.0);
        assert!(
            b.width_safe_for(10.0, 1.0),
            "the dead-zone (inset+outset) covers a per-tick step × K_SAFETY"
        );
        assert!(
            !b.width_safe_for(100.0, 1.0),
            "a body 10× faster would skip this band"
        );
    }

    #[test]
    fn containment_band_rejects_degenerate_edges() {
        // inset <= 0 ⇒ Err (the `&&` short-circuits on the first operand).
        assert_eq!(
            ContainmentBand::for_containment_velocity_safe(0.0, 1.0, 0.0, 0.1, 0.0),
            Err(BandError::InvalidEdges)
        );
        // inset > 0 but the resolved outset is <= 0 (outset_min = 0 AND need - inset < 0) ⇒ Err.
        assert_eq!(
            ContainmentBand::for_containment_velocity_safe(1.0, 0.0, 0.0, 0.1, 0.0),
            Err(BandError::InvalidEdges)
        );
    }

    // ---- RLM Step 2: AoiConfig ---------------------------------------------------------------
    #[test]
    fn aoi_config_for_velocity_safe_ok() {
        // spin = 10·2 = 20; need = 0; tear = max(10·3, 20) = 30.
        let b = AoiConfig::for_velocity_safe(10.0, 2.0, 3.0, 0.0, 1.0, 5, 0.0).expect("valid");
        assert_eq!(b.spin_up_r_m(), 20.0);
        assert_eq!(b.tear_down_r_m(), 30.0);
        assert_eq!(b.grace_ticks(), 5);
        assert!(b.spin_up_r_m() < b.tear_down_r_m());
    }

    #[test]
    fn aoi_config_rejects_inverted_edges() {
        // tear_factor < spin_factor with no widening ⇒ tear == spin ⇒ reject (the second `&&` operand).
        assert_eq!(
            AoiConfig::for_velocity_safe(10.0, 3.0, 2.0, 0.0, 1.0, 0, 0.0),
            Err(BandError::InvalidEdges)
        );
    }

    #[test]
    fn aoi_config_rejects_zero_spin_up() {
        // spin_factor 0 ⇒ spin_up == 0 ⇒ reject (the first `&&` operand short-circuits).
        assert_eq!(
            AoiConfig::for_velocity_safe(10.0, 0.0, 2.0, 0.0, 1.0, 0, 0.0),
            Err(BandError::InvalidEdges)
        );
    }

    #[test]
    fn aoi_config_velocity_widens_the_gap() {
        // need = 100·1·(K_SAFETY 2 + 0) = 200; tear = max(10·2.5 = 25, 20 + 200 = 220) = 220.
        let b = AoiConfig::for_velocity_safe(10.0, 2.0, 2.5, 100.0, 1.0, 0, 0.0).expect("valid");
        assert_eq!(b.spin_up_r_m(), 20.0);
        assert_eq!(b.tear_down_r_m(), 220.0);
    }

    #[test]
    fn aoi_config_inert_is_zero() {
        let b = AoiConfig::inert();
        assert_eq!(b.spin_up_r_m(), 0.0);
        assert!(!b.in_range(false, 100.0));
        assert!(!b.in_range(true, 100.0));
    }

    #[test]
    fn aoi_in_range_acquire_and_hold() {
        let b = AoiConfig::for_velocity_safe(10.0, 2.0, 3.0, 0.0, 1.0, 5, 0.0).expect("valid"); // 20 / 30
        assert!(b.in_range(false, 19.0)); // acquire within spin_up
        assert!(!b.in_range(false, 21.0)); // not acquired; not yet in
        assert!(b.in_range(true, 29.0)); // hold within the larger tear_down
        assert!(!b.in_range(true, 31.0)); // released past tear_down
    }

    #[test]
    fn depth_beats_orders_deepest_then_realm_then_index() {
        let r = RealmId::System(1);
        // depth DESC: the deeper region wins (and the order is strict — not vice-versa).
        assert!(depth_beats((2, r, 0), (1, r, 0)));
        assert!(!depth_beats((1, r, 0), (2, r, 0)));
        // equal depth ⇒ RealmId ASC: the smaller realm wins.
        assert!(depth_beats(
            (2, RealmId::System(1), 0),
            (2, RealmId::System(2), 0)
        ));
        assert!(!depth_beats(
            (2, RealmId::System(2), 0),
            (2, RealmId::System(1), 0)
        ));
        // equal depth AND realm ⇒ slice ix ASC: the smaller index wins.
        assert!(depth_beats((2, r, 0), (2, r, 1)));
        assert!(!depth_beats((2, r, 1), (2, r, 0)));
        // identical ⇒ not strictly greater (irreflexive — a candidate never beats itself).
        assert!(!depth_beats((2, r, 0), (2, r, 0)));
    }

    #[test]
    fn should_rehome_fires_only_on_a_container_change_past_the_cooldown() {
        let t = BoundaryTuning::DEFAULT; // k_dwell = 5
        let owning = RealmId::System(7);
        let dest = RealmId::System(8);
        // container changed, never committed (None) ⇒ re-home to the new container.
        assert_eq!(should_rehome(owning, dest, None, &t), Some(dest));
        // container changed, cooldown elapsed (since_commit == k_dwell) ⇒ re-home.
        assert_eq!(should_rehome(owning, dest, Some(t.k_dwell), &t), Some(dest));
        // container changed but WITHIN the cooldown ⇒ suppressed.
        assert_eq!(should_rehome(owning, dest, Some(t.k_dwell - 1), &t), None);
        // container == owning (no change) ⇒ no re-home, regardless of cooldown.
        assert_eq!(should_rehome(owning, owning, None, &t), None);
        // SYMMETRIC: the reverse re-home (8→7) fires by the IDENTICAL rule — no direction.
        assert_eq!(should_rehome(dest, owning, None, &t), Some(owning));
    }

    proptest::proptest! {
        /// THE MEASUREMENT, not the argument. The claim being made is "at a zero whole-number part the
        /// exact form reduces to the old truncating one, so this change is byte-identical today". That is
        /// a claim about bits, so it is checked as one: both forms are computed over a wide range of real
        /// positions and compared by RAW BIT PATTERN, not by `==` (which would call +0.0 and -0.0 equal
        /// and hide exactly the case worth knowing about). Every position here carries the zero
        /// whole-number part that production carries today.
        #[test]
        fn at_a_zero_whole_number_part_the_exact_form_is_bit_identical_to_the_truncating_one(
            px in -1.0e9f64..1.0e9,
            py in -1.0e9f64..1.0e9,
            pz in -1.0e9f64..1.0e9,
            cx in -1.0e9f64..1.0e9,
            cy in -1.0e9f64..1.0e9,
            cz in -1.0e9f64..1.0e9,
        ) {
            let p = LatticePos::local(DVec3::new(px, py, pz));
            let centre = LatticePos::local(DVec3::new(cx, cy, cz));
            let exact = p.delta_m(centre, crate::pose::Tier::Fine);
            let truncating = p.offset() - centre.offset();
            proptest::prop_assert_eq!(
                [exact.x.to_bits(), exact.y.to_bits(), exact.z.to_bits()],
                [
                    truncating.x.to_bits(),
                    truncating.y.to_bits(),
                    truncating.z.to_bits()
                ],
                "the two forms disagree in bits at a zero whole-number part: exact {:?} vs truncating {:?}",
                exact,
                truncating
            );
        }
    }

    #[test]
    fn the_two_forms_agree_on_negative_zero_the_one_case_the_random_range_will_not_reach() {
        // The proptest above samples a wide range but will essentially never produce NEGATIVE zero, and
        // this project has been bitten before by treating +0.0 and -0.0 as interchangeable (they are
        // different bytes on the wire, and negative zero is reachable from the orbital maths). The exact
        // form adds `0.0 * edge` to the difference, and `0.0 + (-0.0)` is `+0.0` — so if any component
        // difference is negative zero the two forms DIVERGE IN BITS. Whether that happens is measured
        // here, not reasoned about.
        let p = LatticePos::local(DVec3::new(-0.0, 1.0, -0.0));
        let centre = LatticePos::local(DVec3::new(0.0, 1.0, 0.0));
        let exact = p.delta_m(centre, crate::pose::Tier::Fine);
        let truncating = p.offset() - centre.offset();

        // The measured answer: the bits DIFFER on the zero components (+0.0 versus -0.0)...
        assert_eq!(truncating.x.to_bits(), (-0.0f64).to_bits());
        assert_eq!(exact.x.to_bits(), 0.0f64.to_bits());
        assert_ne!(exact.x.to_bits(), truncating.x.to_bits());

        // ...and it does not reach any decision, because every shape reads this vector through a
        // magnitude, and the magnitude of negative zero is positive zero. Asserted for all three shapes
        // rather than argued for one.
        for shape in [
            Boundary::Shell { r: 1.0 },
            Boundary::Aabb {
                half: DVec3::splat(2.0),
            },
            Boundary::Obb {
                half: DVec3::splat(2.0),
                orient: DQuat::IDENTITY,
            },
        ] {
            assert_eq!(
                shape.signed_distance(exact).to_bits(),
                shape.signed_distance(truncating).to_bits(),
                "the sign of zero must not change a containment verdict for {shape:?}"
            );
        }
    }

    /// A test region with an explicit radius — so a fixture claiming to be a VALID forest can be
    /// geometrically nested (child strictly inside parent), not merely topologically well-formed.
    fn test_region_r(realm: RealmId, parent: Option<RealmId>, r: f64) -> RealmRegion {
        RealmRegion {
            shape: Boundary::Shell { r },
            look: Some(Boundary::Shell { r }),
            ..test_region(realm, parent)
        }
    }

    fn test_region(realm: RealmId, parent: Option<RealmId>) -> RealmRegion {
        RealmRegion {
            realm,
            center: ParentCentre::authored(LatticePos::local(DVec3::ZERO)),
            frame: FrameRef::SystemSpace { system_seed: 0 },
            shape: Boundary::Shell { r: 1.0 },
            look: Some(Boundary::Shell { r: 1.0 }),
            band: ContainmentBand::for_containment_velocity_safe(1.0, 2.0, 0.0, 1.0, 0.0)
                .expect("valid test band"),
            aoi: AoiConfig::inert(),
            parent,
        }
    }

    fn test_pose() -> StampedPose {
        StampedPose {
            frame: FrameRef::SystemSpace { system_seed: 7 },
            pos: LatticePos::local(DVec3::ZERO),
            vel: DVec3::ZERO,
            orient: DQuat::IDENTITY,
            universe_tick: crate::ids::UniverseTick(0),
        }
    }

    /// A book anchored on the test pose's frame at its tick, with one identity row for `frame`.
    fn verdict_book(frame: FrameRef) -> crate::placement::PlacementBook {
        crate::placement::PlacementBook::new(
            FrameRef::SystemSpace { system_seed: 0 },
            crate::ids::UniverseTick(0),
            vec![(frame, crate::frame::FramePlacement::identity())],
        )
    }

    fn pose_at(v: DVec3) -> StampedPose {
        StampedPose {
            frame: FrameRef::SystemSpace { system_seed: 0 },
            pos: crate::pose::LatticePos::from_metres(v, Tier::Fine),
            vel: DVec3::ZERO,
            orient: DQuat::IDENTITY,
            universe_tick: crate::ids::UniverseTick(0),
        }
    }

    #[test]
    fn region_verdict_shell_is_integer_and_matches_the_f64_rule_away_from_the_edges() {
        // The Shell arms: acquire (floor) and release (ceil), both outcomes, and P5-agreement with
        // the f64 rule at every tested point (all ≥ one cell from a band edge).
        let region = test_region_r(RealmId::Planet(1), Some(RealmId::System(0)), 10.0);
        let book = verdict_book(region.frame);
        for (v, was) in [
            (DVec3::new(8.5, 0.0, 0.0), false), // inside the acquire edge (r − inset = 9)
            (DVec3::new(9.5, 0.0, 0.0), false), // in the dead-zone: not acquired
            (DVec3::new(11.5, 0.0, 0.0), true), // held: inside the release edge (r + outset = 12)
            (DVec3::new(12.5, 0.0, 0.0), true), // released: beyond it
            (DVec3::new(13.0, 0.0, 0.0), false), // far outside: the Chebyshev pre-test arm
        ] {
            let got = point_verdict(&pose_at(v), &region, &book, was).expect("placeable");
            let sd = region_signed_distance(&pose_at(v), &region, &book).expect("placeable");
            assert_eq!(
                got.member,
                region.band.member(was, sd),
                "integer and f64 verdicts agree at {v:?} (was_member = {was})"
            );
            assert!((got.signed_distance_m - sd).abs() < 1e-12);
        }
    }

    #[test]
    fn region_verdict_shell_edges_round_floor_acquire_and_ceil_release() {
        // THE ROUNDING RULE (§A4.4), driven at the exact edges: a point EXACTLY on the acquire edge
        // is admitted (floor keeps the dyadic edge itself in), one cell beyond it is not; a point
        // exactly on the release edge holds, one cell beyond releases.
        let region = test_region_r(RealmId::Planet(1), Some(RealmId::System(0)), 10.0);
        let book = verdict_book(region.frame);
        let cell = Tier::Fine.cell_edge_m();
        // acquire edge = 9.0 m exactly (dyadic ⇒ floor is exact).
        let on_acquire = pose_at(DVec3::new(9.0, 0.0, 0.0));
        assert!(
            point_verdict(&on_acquire, &region, &book, false)
                .expect("ok")
                .member
        );
        let past_acquire = pose_at(DVec3::new(9.0 + 2.0 * cell, 0.0, 0.0));
        assert!(
            !point_verdict(&past_acquire, &region, &book, false)
                .expect("ok")
                .member
        );
        // release edge = 12.0 m exactly.
        let on_release = pose_at(DVec3::new(12.0, 0.0, 0.0));
        assert!(
            point_verdict(&on_release, &region, &book, true)
                .expect("ok")
                .member
        );
        let past_release = pose_at(DVec3::new(12.0 + 2.0 * cell, 0.0, 0.0));
        assert!(
            !point_verdict(&past_release, &region, &book, true)
                .expect("ok")
                .member
        );
    }

    #[test]
    fn region_verdict_shell_never_acquires_a_region_thinner_than_its_inset() {
        // r < inset ⇒ a negative acquire edge ⇒ a negative cell threshold no |Δ| can pass — the
        // integer spelling of the f64 rule's "never acquirable" answer.
        let region = test_region_r(RealmId::Planet(1), Some(RealmId::System(0)), 0.5);
        let book = verdict_book(region.frame);
        let at_centre = pose_at(DVec3::ZERO);
        assert!(
            !point_verdict(&at_centre, &region, &book, false)
                .expect("ok")
                .member
        );
    }

    #[test]
    fn region_verdict_shell_refuses_the_unsquarable_giant_instead_of_panicking() {
        // The H-01 backstop arm: a shell whose release edge exceeds the square-safe domain, probed
        // by a separation past that domain — the Chebyshev pre-test passes (edge > |Δ|) but the
        // square would overflow, so the guarded comparator answers "outside", never a panic.
        let region = test_region_r(
            RealmId::Planet(1),
            Some(RealmId::System(0)),
            8.9e15, // ≈ 9.1e18 cells > CELL_DOMAIN_MAX
        );
        let book = verdict_book(region.frame);
        let mut probe = pose_at(DVec3::ZERO);
        probe.pos = crate::pose::LatticePos::at(
            crate::glam::I64Vec3::new(5_000_000_000_000_000_000, 0, 0),
            DVec3::ZERO,
        );
        assert!(
            !point_verdict(&probe, &region, &book, true)
                .expect("ok")
                .member
        );
    }

    #[test]
    fn region_verdict_aabb_is_chebyshev_per_axis() {
        // The Aabb arms: acquire inside every axis's floor edge; the release hold is the box
        // inflated by the outset per axis (a box — the declared conservative-outward difference
        // from the f64 SDF's rounded corners, bounded by the outset).
        let region = RealmRegion {
            shape: Boundary::Aabb {
                half: DVec3::new(5.0, 4.0, 3.0),
            },
            ..test_region(RealmId::Station(1), Some(RealmId::System(0)))
        };
        let book = verdict_book(region.frame);
        // Acquire: inside every axis by ≥ inset (1 m).
        assert!(
            point_verdict(&pose_at(DVec3::new(3.9, 2.9, 1.9)), &region, &book, false)
                .expect("ok")
                .member
        );
        // One axis outside its acquire edge ⇒ not acquired (the per-axis `&` fold).
        assert!(
            !point_verdict(&pose_at(DVec3::new(4.5, 0.0, 0.0)), &region, &book, false)
                .expect("ok")
                .member
        );
        // Held out to half + outset (2 m) per axis…
        assert!(
            point_verdict(&pose_at(DVec3::new(6.5, 0.0, 0.0)), &region, &book, true)
                .expect("ok")
                .member
        );
        // …and released past it.
        assert!(
            !point_verdict(&pose_at(DVec3::new(0.0, 6.5, 0.0)), &region, &book, true)
                .expect("ok")
                .member
        );
        // At a CORNER the integer hold is the declared box: Chebyshev excess 2 m on each axis holds
        // where the Euclidean corner distance (2√3 ≈ 3.46 m) would have released — the documented
        // conservative-outward corner difference, asserted so it is a decision, not an accident.
        let corner = pose_at(DVec3::new(7.0 - 0.01, 6.0 - 0.01, 5.0 - 0.01));
        assert!(
            point_verdict(&corner, &region, &book, true)
                .expect("ok")
                .member
        );
        let sd = region_signed_distance(&corner, &region, &book).expect("ok");
        assert!(
            !region.band.member(true, sd),
            "the f64 corner rule releases here"
        );
    }

    #[test]
    fn region_verdict_obb_keeps_the_decimal_arm() {
        // The SHAPE branch (§A7.2): an Obb cannot take the integer path (a rotated box needs a
        // rotation), so its verdict is the f64 rule verbatim, quantum declared.
        let region = RealmRegion {
            shape: Boundary::Obb {
                half: DVec3::new(5.0, 4.0, 3.0),
                orient: DQuat::from_rotation_z(std::f64::consts::FRAC_PI_4),
            },
            ..test_region(RealmId::Station(1), Some(RealmId::System(0)))
        };
        let book = verdict_book(region.frame);
        for (v, was) in [
            (DVec3::new(1.0, 1.0, 0.0), false),
            (DVec3::new(9.0, 0.0, 0.0), true),
        ] {
            let got = point_verdict(&pose_at(v), &region, &book, was).expect("ok");
            let sd = region_signed_distance(&pose_at(v), &region, &book).expect("ok");
            assert_eq!(got.member, region.band.member(was, sd));
        }
    }

    #[test]
    fn region_verdict_propagates_the_frame_error() {
        // An unplaceable frame degrades exactly as region_signed_distance does — the caller's
        // `unwrap_or` treats it as non-member, never a container.
        let region = RealmRegion {
            frame: FrameRef::PlanetCentered { planet_seed: 1 },
            ..test_region(RealmId::Planet(1), Some(RealmId::System(0)))
        };
        let empty = crate::placement::PlacementBook::new(
            FrameRef::SystemSpace { system_seed: 0 },
            crate::ids::UniverseTick(0),
            Vec::new(),
        );
        assert_eq!(
            point_verdict(&pose_at(DVec3::ZERO), &region, &empty, false),
            Err(FrameError::UnknownDestFrame)
        );
    }

    #[test]
    fn container_folds_the_deepest_member_from_the_root_identity() {
        let root = RealmId::System(0);
        // No members ⇒ the root realm (the fold IDENTITY — "always in a realm", no None arm).
        assert_eq!(container(root, &[]), root);
        // Members present ⇒ the DEEPEST wins; a shallower member does NOT displace a deeper best.
        let members = [
            (1u32, RealmId::System(1), 1usize),
            (3u32, RealmId::Planet(7), 3usize),
            (2u32, RealmId::System(7), 2usize),
        ];
        assert_eq!(container(root, &members), RealmId::Planet(7));
        // A single shallow member still beats the depth-0 root identity.
        assert_eq!(
            container(root, &[(1, RealmId::System(1), 1)]),
            RealmId::System(1)
        );
    }

    #[test]
    fn region_depth_terminates_on_a_malformed_cycle() {
        // A 2-region CYCLE (each is the other's parent) — the boot guard rejects this, but region_depth
        // must TERMINATE via the hop cap, never hang. It returns after `regions.len()` hops.
        let cyclic = [
            test_region(RealmId::System(1), Some(RealmId::System(2))),
            test_region(RealmId::System(2), Some(RealmId::System(1))),
        ];
        assert_eq!(
            region_depth(&cyclic, RealmId::System(1)),
            cyclic.len() as u32,
            "the hop cap bounds a malformed cycle"
        );
    }

    #[test]
    fn region_signed_distance_resolves_ok_and_propagates_a_frame_error() {
        // Ok arm: the pose stands in the book's anchor frame and the region IS that anchor's realm
        // (its own frame == the anchor) ⇒ the reframe is the identity, and a pose at the unit shell's
        // center measures signed distance -r = -1.
        let mut region = test_region(RealmId::System(7), None);
        region.frame = test_pose().frame;
        let book = PlacementBook::new(test_pose().frame, test_pose().universe_tick, Vec::new());
        assert_eq!(
            region_signed_distance(&test_pose(), &region, &book),
            Ok(-1.0)
        );
        // Err arm: a frame error propagates — the region's frame has no row in the book (the
        // unknown-frame degrade the detector maps to "not a member").
        let stranger = test_region(RealmId::System(9), None);
        assert_eq!(
            region_signed_distance(&test_pose(), &stranger, &book),
            Err(FrameError::UnknownDestFrame)
        );
    }

    // ----- guard_regions_nest (C-5): each reject arm + the ok arm -----

    /// A well-formed forest: exactly one root, unique realms, resolvable acyclic parents, within `max`.
    fn valid_forest() -> Vec<RealmRegion> {
        vec![
            // Concentric and STRICTLY nested: each child fits well inside its parent's interior, so
            // this forest is valid GEOMETRICALLY as well as topologically. (It used to be three
            // identical unit shells at one point — fine for the topology fences, but it asserted a
            // containment relationship that was not actually true.)
            test_region_r(RealmId::System(0), None, 100.0), // the ambient root
            test_region_r(RealmId::System(1), Some(RealmId::System(0)), 10.0),
            test_region_r(RealmId::Planet(1), Some(RealmId::System(1)), 1.0),
        ]
    }

    #[test]
    fn a_shard_hosting_a_child_realm_may_not_boot_claiming_to_be_a_root() {
        // BOTH ARMS of the boot fence. The refused case is the one that used to be tolerated: a shard
        // declaring a root lineage while hosting a realm the world puts inside another. Harmless while
        // leaving was a geometric search that never read the lineage; now it is how a realm knows who to
        // hand an occupant to, so a wrong answer delivers players to the ambient root in silence.
        let forest = valid_forest();
        let child = forest
            .iter()
            .find(|r| r.parent.is_some())
            .copied()
            .expect("a valid forest nests something");
        let parent = child.parent.expect("just filtered on it");
        assert_eq!(
            guard_lineage_reaches_root(&forest, child.realm, None),
            Err(LineageNotRooted {
                realm: child.realm,
                parent
            })
        );
        // Naming the parent is accepted…
        assert_eq!(
            guard_lineage_reaches_root(&forest, child.realm, Some(parent)),
            Ok(())
        );
        // …and the ONE true root legitimately has none.
        let root = forest
            .iter()
            .find(|r| r.parent.is_none())
            .copied()
            .expect("a valid forest has one ambient root");
        assert_eq!(
            guard_lineage_reaches_root(&forest, root.realm, None),
            Ok(())
        );
    }

    /// The reach map the boot would supply for an all-static fixture forest: every child judged
    /// EXACTLY at its authored offset (the [`ChildReach::Fixed`] arm).
    fn fixed_reaches(regions: &[RealmRegion]) -> std::collections::BTreeMap<RealmId, ChildReach> {
        regions
            .iter()
            .filter(|r| r.parent.is_some())
            .filter_map(|r| {
                // A child whose parent is NOT in the forest is skipped rather than guessed at: its
                // centre has no step, and inventing one is the defect `ParentCentre` exists to stop.
                // The dangling parent is the nesting guard's OWN refusal to make — this map simply
                // does not pretend to a reach it cannot state.
                r.centre_m(regions)
                    .map(|at| (r.realm, ChildReach::Fixed(at)))
            })
            .collect()
    }

    #[test]
    fn the_band_and_the_speed_are_two_readings_of_one_solve() {
        // THE OWNER'S RULING, 2026-08-24, as an assertion: band width and top speed are the same
        // statement read in two directions. If these two ever stopped being inverses, a realm could
        // state a ceiling its own boundaries could not contain — and a ship would pass through a moon
        // without anything noticing.
        let dt = 0.02;
        let ticks = 3.0;
        for v in [0.5_f64, 1.0, 500.0, 1.0e6, 1.0e13] {
            let w = band_for_speed(v, dt, ticks);
            assert!(
                (speed_for_band(w, dt, ticks) - v).abs() <= v * f64::EPSILON * 4.0,
                "the round trip must return the same speed at {v} m/s"
            );
        }
        // A wider band affords a faster ceiling, monotonically — the trade the owner priced.
        assert!(speed_for_band(20.0, dt, ticks) > speed_for_band(10.0, dt, ticks));
        // And speed is signed-blind: approaching and receding need the same room.
        assert_eq!(
            band_for_speed(-7.0, dt, ticks),
            band_for_speed(7.0, dt, ticks)
        );
    }

    #[test]
    fn a_window_of_no_time_affords_no_speed_at_all() {
        // FAILS CLOSED, both arms driven. The arithmetic answer to "how fast may I go if I am never
        // observed" is infinity; the safe answer is zero. A ceiling that read as infinity on a
        // misconfigured band is how a misconfiguration becomes a ship inside a planet.
        assert_eq!(speed_for_band(10.0, 0.0, 3.0), 0.0);
        assert_eq!(speed_for_band(10.0, 0.02, 0.0), 0.0);
        assert_eq!(speed_for_band(10.0, -0.02, 3.0), 0.0);
        // And the ordinary arm still answers.
        assert_eq!(speed_for_band(6.0, 0.5, 4.0), 3.0);
    }

    #[test]
    fn guard_regions_nest_accepts_a_valid_forest() {
        let forest = valid_forest();
        assert_eq!(guard_regions_nest(&forest, &fixed_reaches(&forest)), Ok(()));
    }

    #[test]
    fn guard_regions_nest_rejects_a_child_with_no_stated_reach() {
        // TOTALITY: the boot must state EVERY child's worst-instant reach — a missing entry is a
        // typed reject, never a defaulted zero (the decode-to-Default discipline, applied to motion).
        let forest = valid_forest();
        let mut holes = fixed_reaches(&forest);
        let removed = forest
            .iter()
            .find(|r| r.parent.is_some())
            .expect("the valid forest has a child")
            .realm;
        holes.remove(&removed);
        assert_eq!(
            guard_regions_nest(&forest, &holes),
            Err(RegionNestError::NoReachForChild { realm: removed })
        );
    }

    #[test]
    fn guard_regions_nest_judges_a_mover_at_its_apoapsis_bound() {
        // THE ARM THE OLD FENCE COULD NOT HAVE (finding 27): a mover's stored centre is zero, so the
        // old read judged its SIZE only and a planet whose orbit left its system's shell booted clean.
        // The boot now states the mover's worst-instant EXCURSION (apoapsis + nothing else — the fence
        // cannot ask what produces it), and the same child passes or fails on that bound alone.
        let forest = vec![
            test_region_r(RealmId::System(0), None, 100.0),
            // A mover of radius 5 whose centre can wander 80 m out: reaches 85, inside 100. Fits.
            test_region_r(RealmId::Planet(1), Some(RealmId::System(0)), 5.0),
        ];
        let mut reaches = fixed_reaches(&forest);
        reaches.insert(RealmId::Planet(1), ChildReach::Excursion(80.0));
        assert_eq!(guard_regions_nest(&forest, &reaches), Ok(()));
        // The SAME child with an excursion of 98: reaches 103 > 100. Refused, with both numbers.
        reaches.insert(RealmId::Planet(1), ChildReach::Excursion(98.0));
        assert_eq!(
            guard_regions_nest(&forest, &reaches),
            Err(RegionNestError::ChildEscapesParent {
                realm: RealmId::Planet(1),
                parent: RealmId::System(0),
                reach: 103.0,
                limit: 100.0,
            })
        );
    }

    #[test]
    fn guard_regions_nest_rejects_a_child_that_escapes_its_parent() {
        // THE SOUNDNESS FENCE for the ancestry-derived containment prior (task #177): membership of a
        // parent is INFERRED from membership of a child, so a child poking outside its parent would make
        // that inference assert a falsehood every tick. Topology alone cannot catch it — this forest is
        // perfectly well-formed as a tree.
        let escaping = vec![
            test_region_r(RealmId::System(0), None, 100.0),
            test_region_r(RealmId::System(1), Some(RealmId::System(0)), 10.0),
            // A "child" of the r=10 region that is itself r=50 — it engulfs its own parent.
            test_region_r(RealmId::Planet(1), Some(RealmId::System(1)), 50.0),
        ];
        let err = guard_regions_nest(&escaping, &fixed_reaches(&escaping))
            .expect_err("an escaping child must be rejected");
        assert_eq!(
            err,
            RegionNestError::ChildEscapesParent {
                realm: RealmId::Planet(1),
                parent: RealmId::System(1),
                reach: 50.0,
                // The parent's own boundary — NOT boundary-minus-inset. See `child_fits_in_parent`:
                // the prior is falsified by leaving the parent, not by standing in its release band.
                limit: 10.0,
            }
        );
    }

    #[test]
    fn the_nesting_fence_measures_a_child_across_a_frame_boundary() {
        // THE ARM THAT WAS DEAD. Every real parent/child pair has DIFFERENT frames — a planet is
        // `PlanetCentered`, its system `SystemSpace` — and the fence used to return early on exactly that,
        // so it ran on nothing. This pair differs in frame and MUST still be judged.
        let parent = RealmRegion {
            frame: FrameRef::SystemSpace { system_seed: 1 },
            ..test_region_r(RealmId::System(1), Some(RealmId::System(0)), 10.0)
        };
        let escaping = RealmRegion {
            frame: FrameRef::PlanetCentered { planet_seed: 1 },
            center: ParentCentre::authored(LatticePos::local(DVec3::new(8.0, 0.0, 0.0))),
            ..test_region_r(RealmId::Planet(1), Some(RealmId::System(1)), 5.0)
        };
        assert_eq!(
            child_fits_in_parent(
                &ChildReach::Fixed(DVec3::new(8.0, 0.0, 0.0)),
                &escaping,
                &parent
            ),
            Some((13.0, 10.0)),
            "a cross-frame child that reaches 13 m out of a 10 m parent must be caught"
        );
        // …and the accept twin across the same frame boundary, so a fence that simply always rejects
        // would fail here.
        let fitting = RealmRegion {
            center: ParentCentre::authored(LatticePos::local(DVec3::new(4.0, 0.0, 0.0))),
            ..escaping
        };
        assert_eq!(
            child_fits_in_parent(
                &ChildReach::Fixed(DVec3::new(4.0, 0.0, 0.0)),
                &fitting,
                &parent
            ),
            None
        );
    }

    #[test]
    fn a_box_child_is_measured_by_its_farthest_corner_not_the_sphere_around_it() {
        // The shipped walk forest's Area A, to the metre: a half-3 box centred 5 m out inside a planet of
        // radius 10. Its farthest corner is at (8,3,3) ⇒ 9.06 m, so it FITS. The spherical over-estimate
        // (5 + |(3,3,3)| = 10.20) called it an escape, which is why the exact corner is the measure.
        let parent = test_region_r(RealmId::Planet(7), Some(RealmId::System(7)), 10.0);
        let area = RealmRegion {
            center: ParentCentre::authored(LatticePos::local(DVec3::new(5.0, 0.0, 0.0))),
            shape: Boundary::Aabb {
                half: DVec3::splat(3.0),
            },
            ..test_region_r(RealmId::Area(7), Some(RealmId::Planet(7)), 0.0)
        };
        let exact = Boundary::Aabb {
            half: DVec3::splat(3.0),
        }
        .max_reach_from(DVec3::new(5.0, 0.0, 0.0));
        assert_eq!(exact, DVec3::new(8.0, 3.0, 3.0).length());
        assert!(exact < 10.0, "measured reach {exact} must be inside r=10");
        assert_eq!(
            child_fits_in_parent(
                &ChildReach::Fixed(DVec3::new(5.0, 0.0, 0.0)),
                &area,
                &parent
            ),
            None
        );
    }

    #[test]
    fn a_rotated_box_reaches_the_same_distance_as_the_unrotated_one_about_its_own_centre() {
        // `max_reach_from` on an `Obb`: a cube spun about its centre reaches exactly as far as before
        // (its corner set is the same set of points), which is the property that lets a station be
        // authored at any orientation without the fence changing its verdict.
        let half = DVec3::splat(2.0);
        let spun = Boundary::Obb {
            half,
            orient: DQuat::from_rotation_z(std::f64::consts::FRAC_PI_4),
        };
        let flat = Boundary::Aabb { half };
        assert!(
            (spun.max_reach_from(DVec3::ZERO) - flat.max_reach_from(DVec3::ZERO)).abs() < 1e-12
        );
        // Off-centre the two DO differ (the spun corners point elsewhere), so this is not a vacuous pair.
        assert_ne!(
            spun.max_reach_from(DVec3::new(3.0, 0.0, 0.0)),
            flat.max_reach_from(DVec3::new(3.0, 0.0, 0.0))
        );
    }

    #[test]
    fn guard_regions_nest_allows_an_offset_child_that_still_fits() {
        // The ACCEPT twin, and the reason the check measures REACH (offset + extent) rather than size
        // alone: a small child placed off-centre is fine as long as its far side stays inside.
        let offset_child = RealmRegion {
            center: ParentCentre::authored(LatticePos::local(DVec3::new(5.0, 0.0, 0.0))),
            ..test_region_r(RealmId::Planet(1), Some(RealmId::System(1)), 3.0)
        };
        let forest = vec![
            test_region_r(RealmId::System(0), None, 100.0),
            test_region_r(RealmId::System(1), Some(RealmId::System(0)), 10.0),
            offset_child, // reaches 5 + 3 = 8, inside the parent's 10
        ];
        assert_eq!(guard_regions_nest(&forest, &fixed_reaches(&forest)), Ok(()));
    }

    #[test]
    fn guard_regions_nest_accepts_a_forest_far_wider_than_the_retired_bitset() {
        // SL9 — A PARENT'S CHILD COUNT IS UNBOUNDED. This forest gives ONE root 1024 direct children:
        // sixteen times the width of the `u64` membership bitset that used to cap the fence, and the
        // exact SHAPE a galaxy states (one root, every star system a direct child of it). The retired
        // count fence answered `TooManyRegions { found: 1025, max: 64 }` here; there is no width left
        // to exceed. Asserted rather than argued — this test is the slice's own proof.
        const WIDE: u64 = 1024;
        let mut forest = vec![test_region_r(RealmId::System(0), None, 100.0)];
        forest.extend(
            (1..=WIDE).map(|n| test_region_r(RealmId::Planet(n), Some(RealmId::System(0)), 1.0)),
        );
        assert_eq!(forest.len(), WIDE as usize + 1);
        assert_eq!(guard_regions_nest(&forest, &fixed_reaches(&forest)), Ok(()));
    }

    #[test]
    fn guard_regions_nest_rejects_zero_or_two_roots() {
        // TWO roots.
        let two_roots = vec![
            test_region(RealmId::System(0), None),
            test_region(RealmId::System(1), None),
        ];
        assert_eq!(
            guard_regions_nest(&two_roots, &fixed_reaches(&two_roots)),
            Err(RegionNestError::RootCount { found: 2 })
        );
        // ZERO roots (a pure cycle — caught by the root count BEFORE the chain walk).
        let no_root = vec![
            test_region(RealmId::System(1), Some(RealmId::System(2))),
            test_region(RealmId::System(2), Some(RealmId::System(1))),
        ];
        assert_eq!(
            guard_regions_nest(&no_root, &fixed_reaches(&no_root)),
            Err(RegionNestError::RootCount { found: 0 })
        );
    }

    #[test]
    fn guard_regions_nest_rejects_a_duplicate_realm() {
        let dup = vec![
            test_region(RealmId::System(0), None),
            test_region(RealmId::System(1), Some(RealmId::System(0))),
            test_region(RealmId::System(1), Some(RealmId::System(0))),
        ];
        assert_eq!(
            guard_regions_nest(&dup, &fixed_reaches(&dup)),
            Err(RegionNestError::DuplicateRealm {
                realm: RealmId::System(1)
            })
        );
    }

    #[test]
    fn guard_regions_nest_rejects_a_dangling_parent() {
        let dangling = vec![
            test_region(RealmId::System(0), None),
            test_region(RealmId::System(1), Some(RealmId::System(9))), // System(9) is not a region
        ];
        assert_eq!(
            guard_regions_nest(&dangling, &fixed_reaches(&dangling)),
            Err(RegionNestError::DanglingParent {
                realm: RealmId::System(1),
                parent: RealmId::System(9),
            })
        );
    }

    #[test]
    fn the_fit_check_declines_to_judge_a_child_measured_in_a_different_frame() {
        // The fit check compares a child's reach against its parent's promised interior — but only when
        // the two are measured in the SAME frame. Across frames the numbers are not commensurable, and
        // pretending otherwise would reject perfectly legal placements (a planet's surface region sits at
        // huge coordinates in system space and tiny ones in its own). It declines instead, and the real
        // cross-frame check is ledgered.
        //
        // NOT a subset proof either way: this is the necessary condition that catches the placement
        // mistakes a forest generator actually makes — a child too big, or centred too near the rim.
        let parent = test_region_r(RealmId::System(0), None, 100.0);
        let mut child = test_region_r(RealmId::Planet(1), Some(RealmId::System(0)), 1.0);
        child.frame = FrameRef::PlanetCentered { planet_seed: 1 };
        assert_eq!(
            guard_regions_nest(&[parent, child], &fixed_reaches(&[parent, child])),
            Ok(()),
            "a differently-framed child is not judged here, so the forest is accepted"
        );

        // The SAME child in the SAME frame IS judged — and a huge one is refused. Without this half the
        // test above would pass for the wrong reason (nothing is ever judged).
        let mut oversized = test_region_r(RealmId::Planet(1), Some(RealmId::System(0)), 5_000.0);
        oversized.frame = parent.frame;
        assert!(
            guard_regions_nest(&[parent, oversized], &fixed_reaches(&[parent, oversized])).is_err(),
            "same-frame IS judged: a child larger than its parent's interior is refused"
        );
    }

    #[test]
    fn guard_regions_nest_rejects_a_cycle() {
        // One valid root + a 2-region cycle whose parents resolve ⇒ passes count/root/unique/resolve,
        // then the chain walk exceeds `len` hops for the cycle members ⇒ CycleOrOrphan.
        let cyclic = vec![
            test_region(RealmId::System(0), None),
            test_region(RealmId::System(1), Some(RealmId::System(2))),
            test_region(RealmId::System(2), Some(RealmId::System(1))),
        ];
        assert_eq!(
            guard_regions_nest(&cyclic, &fixed_reaches(&cyclic)),
            Err(RegionNestError::CycleOrOrphan {
                realm: RealmId::System(1)
            })
        );
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
    fn finite_extent_is_the_largest_linear_extent_per_shape() {
        // Shell => the radius (the client renderable-extent filter reads this: systems r=40, planets
        // r=10 render; the ambient Galaxy r=1000 / Universe r=1e9 do NOT).
        assert_eq!(Boundary::Shell { r: 40.0 }.finite_extent(), 40.0);
        // Aabb => the max half-extent component.
        assert_eq!(
            Boundary::Aabb {
                half: DVec3::new(2.0, 8.0, 4.0),
            }
            .finite_extent(),
            8.0,
        );
        // Obb => the max half-extent component (orientation dropped — it does not change the extent).
        assert_eq!(
            Boundary::Obb {
                half: DVec3::new(3.0, 5.0, 1.0),
                orient: DQuat::from_rotation_z(0.7),
            }
            .finite_extent(),
            5.0,
        );
    }

    #[test]
    fn the_two_nesting_extents_bracket_every_shape() {
        // The pair the boot-time nesting fence is built from: the FARTHEST any surface point can be from
        // the centre (so a child's reach is over-estimated) and the NEAREST (so a parent's promise is
        // under-estimated). Conservative in both directions, which is what makes the fence refuse a
        // doubtful placement rather than wave it through.
        //
        // A SPHERE is the case both arms had never been tried with, and it is the interesting one: it is
        // the only shape where the two answers are EQUAL, because every surface point is the same distance
        // out. A box's are not, and that difference is the whole reason two functions exist.
        let shell = Boundary::Shell { r: 40.0 };
        assert_eq!(shell.circumscribed_extent(), 40.0);
        assert_eq!(shell.inscribed_extent(), 40.0);

        // A box: the far CORNER versus the nearest FACE — the corner is further out by the diagonal.
        let boxy = Boundary::Aabb {
            half: DVec3::new(2.0, 8.0, 4.0),
        };
        assert_eq!(
            boxy.circumscribed_extent(),
            DVec3::new(2.0, 8.0, 4.0).length()
        );
        assert_eq!(boxy.inscribed_extent(), 2.0);
        // Orientation cannot move either bound — spinning a box changes neither its corner distance nor
        // its nearest face.
        let spun = Boundary::Obb {
            half: DVec3::new(2.0, 8.0, 4.0),
            orient: DQuat::from_rotation_z(0.7),
        };
        assert_eq!(spun.circumscribed_extent(), boxy.circumscribed_extent());
        assert_eq!(spun.inscribed_extent(), boxy.inscribed_extent());
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
    fn for_box_velocity_safe_widens_destroy_in_normalized_units() {
        // The NORMALIZED-Chebyshev mirror of `for_soi_velocity_safe`: create = create_factor exactly
        // (the box's unit is the surface = 1.0), destroy = max(destroy_factor, create + velocity_pad).
        // SLOW body (small v_rel): the normalized travel is below the raw destroy_factor gap, so the
        // FACTOR edge wins the max (the `destroy_factor` arm) — but the pad_floor also loses here, so we
        // pick a pad_floor small enough that destroy_factor dominates.
        let slow = OverlapBand::for_box_velocity_safe(
            1.15, // create_factor → create edge 1.15
            1.30, // destroy_factor → the raw factor destroy edge 1.30
            1.0,  // v_rel (slow)
            0.05, // dt
            5.0,  // min_half
            0.01, // pad_floor (tiny — below the factor gap of 0.15)
            1.0,  // k_safety_extra
        )
        .expect("ordered factors");
        assert_eq!(slow.create_below(), 1.15);
        // normalized_travel = 1.0 * 0.05 * (2.0 + 1.0) / 5.0 = 0.03 < 0.15 factor gap; pad_floor 0.01 <
        // 0.15 too, so the destroy_factor arm (1.30) wins the outer max.
        assert_eq!(slow.destroy_above(), 1.30);
        // FAST body: the normalized travel exceeds the factor gap, so destroy widens to create + pad
        // (the `create + velocity_pad` arm of the outer max), strictly beyond the raw factor edge.
        let v = 5000.0;
        let dt = 0.05;
        let extra = 1.0;
        let min_half = 5.0;
        let fast = OverlapBand::for_box_velocity_safe(1.15, 1.30, v, dt, min_half, 0.5, extra)
            .expect("ok");
        assert_eq!(fast.create_below(), 1.15);
        let expected_pad = v * dt * (K_SAFETY + extra) / min_half;
        assert_eq!(fast.destroy_above(), 1.15 + expected_pad);
        assert!(fast.destroy_above() > 1.30);
        // The pad_floor wins when both the velocity travel and the factor gap are tiny (the `pad_floor`
        // arm of the INNER max): v_rel = 0 ⇒ normalized_travel = 0 < pad_floor 0.5, and destroy_factor
        // 1.16 < create + 0.5 = 1.65, so the floored destroy (1.65) wins the outer max too.
        let floored =
            OverlapBand::for_box_velocity_safe(1.15, 1.16, 0.0, 0.05, 5.0, 0.5, 1.0).expect("ok");
        assert_eq!(floored.destroy_above(), 1.15 + 0.5);
        // An inverted/degenerate normalized result is REJECTED LOUD (routes through `new`): a
        // destroy_factor below create_factor with a zero pad inverts the hysteresis.
        assert_eq!(
            OverlapBand::for_box_velocity_safe(1.30, 1.15, 0.0, 0.05, 5.0, 0.0, 1.0),
            Err(BandError::InvalidEdges)
        );
    }

    #[test]
    fn realm_boundary_aabb_pairs_an_aabb_with_a_normalized_velocity_safe_band() {
        // The MIRROR of `realm_boundary_shell_pairs_a_shell_with_an_absolute_band`: an Aabb shape ALWAYS
        // gets the normalized-Chebyshev velocity-safe band, and an Aabb can never be handed the absolute
        // SOI band — the mismatch is unconstructible. Same arg shape as `shell` (the D-38 fixture drives
        // both through one closure).
        let half = DVec3::new(5.0, 6.0, 7.0);
        let rb = RealmBoundary::aabb(
            RealmId::Station(2),
            LatticePos::local(DVec3::new(10.0, 20.0, 30.0)),
            half,
            1.15, // create_factor → create edge 1.15 (normalized)
            1.30, // destroy_factor → destroy edge 1.30 (normalized, slow body)
            1.0,  // v_rel (slow)
            0.05, // dt
            0.01, // pad_floor (tiny so the factor edge dominates)
            1.0,  // k_safety_extra
            Some(RealmId::Planet(9)),
            RealmId::Station(2),
            CrossEffect::Authority,
        )
        .expect("ordered factors");
        // The shape is the AABB with the given half-extents.
        assert_eq!(rb.shape, Boundary::Aabb { half });
        assert_eq!(rb.realm, RealmId::Station(2));
        assert_eq!(rb.to_realm, RealmId::Station(2));
        assert_eq!(rb.parent, Some(RealmId::Planet(9)));
        assert_eq!(rb.effect, CrossEffect::Authority);
        // The band is the normalized one: create at the create_factor, destroy strictly beyond it. For a
        // slow body the factor edges hold verbatim (create 1.15, destroy 1.30).
        assert_eq!(rb.band.create_below(), 1.15);
        assert_eq!(rb.band.destroy_above(), 1.30);
        assert!(rb.band.destroy_above() > rb.band.create_below());
        // Membership: the normalized Chebyshev scalar (1.0 on the surface) drives the band. A point at
        // half the tightest extent is a member (norm 0.5 < create 1.15); a point well outside is not.
        let inside = DVec3::new(2.5, 0.0, 0.0); // norm = 2.5 / 5.0 = 0.5
        let ms_in = rb.shape.membership_scalar(inside);
        assert_eq!(ms_in, 0.5);
        assert!(
            rb.band.update_membership(false, ms_in),
            "0.5 < create ⇒ member"
        );
        let outside = DVec3::new(10.0, 0.0, 0.0); // norm = 10.0 / 5.0 = 2.0
        let ms_out = rb.shape.membership_scalar(outside);
        assert_eq!(ms_out, 2.0);
        assert!(
            !rb.band.update_membership(true, ms_out),
            "2.0 > destroy ⇒ torn down even from a member prior state"
        );
        // A swept INWARD crossing on a DIAGONAL segment (≥2 non-zero axes) is classified Inward — the
        // slab-corner path of `segment_aabb_crossing`, the exact box surface a pure axis-aligned approach
        // would skip. From clearly outside the box to clearly inside it.
        let outside_diag = DVec3::new(20.0, 20.0, 0.0); // both axes outside (20 > 5, 20 > 6)
        let inside_diag = DVec3::new(1.0, 1.0, 0.0); // inside all axes
        assert_eq!(
            rb.shape.swept(outside_diag, inside_diag),
            ShellCrossing::Inward,
            "a diagonal descent into the box is an Inward swept crossing"
        );
        // The band error surfaces LOUD through this ctor too (an inverted factor pair with no pad).
        assert_eq!(
            RealmBoundary::aabb(
                RealmId::Station(2),
                LatticePos::local(DVec3::ZERO),
                DVec3::new(1.0, 1.0, 1.0),
                1.30, // create > destroy: inverted
                1.15,
                0.0, // v_rel 0 ⇒ no widening
                0.05,
                0.0, // pad_floor 0 ⇒ no floor rescue
                1.0,
                None,
                RealmId::Station(2),
                CrossEffect::Interest,
            )
            .expect_err("inverted band rejected"),
            BandError::InvalidEdges
        );
    }

    #[test]
    fn realm_boundary_shell_pairs_a_shell_with_an_absolute_band() {
        // The pairing constructor: a Shell shape ALWAYS gets the absolute velocity-safe SOI band,
        // and a Shell can never be constructed with a normalized box band — the mismatch is
        // unconstructible. destroy > create (a valid hysteresis band by construction).
        let rb = RealmBoundary::shell(
            RealmId::System(1),
            LatticePos::local(DVec3::new(10.0, 20.0, 30.0)),
            1000.0, // r_soi
            0.95,   // create_factor
            1.05,   // destroy_factor
            2.0,    // v_rel
            0.05,   // dt
            0.5,    // pad_floor
            1.0,    // k_safety_extra
            Some(RealmId::System(1)),
            RealmId::Planet(4),
            CrossEffect::Authority,
        );
        assert_eq!(rb.shape, Boundary::Shell { r: 1000.0 });
        assert_eq!(rb.realm, RealmId::System(1));
        assert_eq!(rb.to_realm, RealmId::Planet(4));
        assert_eq!(rb.parent, Some(RealmId::System(1)));
        assert_eq!(rb.effect, CrossEffect::Authority);
        // The band is the absolute one (edges in metres, ~950 / ~1050), destroy strictly beyond create.
        assert!(rb.band.destroy_above() > rb.band.create_below());
        assert_eq!(rb.band.create_below(), 950.0);
    }

    #[test]
    fn realm_boundary_boxed_picks_shape_from_orient_and_rejects_an_inverted_band() {
        // orient None ⇒ Aabb (axis-aligned).
        let aabb = RealmBoundary::boxed(
            RealmId::Station(2),
            LatticePos::local(DVec3::ZERO),
            DVec3::new(5.0, 6.0, 7.0),
            None,
            1.15,
            1.30,
            None,
            RealmId::Station(2),
            CrossEffect::Interest,
        )
        .expect("ordered factors");
        assert_eq!(
            aabb.shape,
            Boundary::Aabb {
                half: DVec3::new(5.0, 6.0, 7.0)
            }
        );
        assert_eq!(aabb.parent, None);
        assert_eq!(aabb.effect, CrossEffect::Interest);
        // The band is the normalized box band (factors verbatim), destroy > create.
        assert_eq!(aabb.band.create_below(), 1.15);
        assert_eq!(aabb.band.destroy_above(), 1.30);
        // orient Some ⇒ Obb (oriented).
        let orient = DQuat::from_rotation_z(std::f64::consts::FRAC_PI_4);
        let obb = RealmBoundary::boxed(
            RealmId::Area(3),
            LatticePos::local(DVec3::ZERO),
            DVec3::new(2.0, 3.0, 4.0),
            Some(orient),
            1.15,
            1.30,
            Some(RealmId::Planet(9)),
            RealmId::Area(3),
            CrossEffect::Authority,
        )
        .expect("ordered factors");
        assert_eq!(
            obb.shape,
            Boundary::Obb {
                half: DVec3::new(2.0, 3.0, 4.0),
                orient
            }
        );
        // An inverted factor pair fails LOUD (an inverted band inverts the hysteresis into a flap).
        assert_eq!(
            RealmBoundary::boxed(
                RealmId::Station(2),
                LatticePos::local(DVec3::ZERO),
                DVec3::new(1.0, 1.0, 1.0),
                None,
                1.30, // create > destroy: inverted
                1.15,
                None,
                RealmId::Station(2),
                CrossEffect::Interest,
            )
            .expect_err("inverted band"),
            BandError::InvalidEdges
        );
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
    }
}
