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

use crate::frame::{FrameContext, FrameError, transfer_frame};
use crate::pose::{FrameRef, LatticePos, RealmId, StampedPose};

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
    pub center: LatticePos,
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
    pub shape: Boundary,
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
/// (the input-side frame seam — the containment twin of [`crate::frame::rebind_pose_to_dest`]'s output
/// seam). At P3 `ctx` is [`crate::frame::IdentityFrames`] (a no-op reframe — position unchanged); at
/// P4/P5 the ephemeris `FrameContext` makes it a real transform, ADDITIVELY, with no caller reshape. A
/// BRANCHLESS generic shim (HR5): it looks the reframe up and delegates ALL branching to the MONOMORPHIC
/// [`region_signed_distance_resolved`], so a new `FrameContext` monomorphization adds ZERO uncovered
/// per-mono branch. The caller treats an `Err` region as "not a member" (safe degrade), never a container.
///
/// # Errors
/// [`FrameError`] if `ctx` has no placement for `pose`'s frame or `region.frame` (inert under identity).
pub fn region_signed_distance(
    pose: &StampedPose,
    region: &RealmRegion,
    ctx: &impl FrameContext,
) -> Result<f64, FrameError> {
    region_signed_distance_resolved(transfer_frame(pose, region.frame, ctx), region)
}

/// The MONOMORPHIC core of [`region_signed_distance`]: given the already-reframed pose (or its frame
/// error), fail loud on the error else take the shape's signed distance from the region center. Holding
/// the `?` branch here keeps [`region_signed_distance`] a straight-line generic shim (HR5).
fn region_signed_distance_resolved(
    reframed: Result<StampedPose, FrameError>,
    region: &RealmRegion,
) -> Result<f64, FrameError> {
    let p = reframed?;
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

/// Why a realm-region forest is malformed — the boot fence (task #135, C-5). ONE variant per REJECT arm
/// so each is asserted with `expect_err` equality (HR5(d)), never `matches!`. These are PURE TOPOLOGY
/// checks; the geometric child-⊆-parent volume subset check needs cross-frame Shape×Shape math and is
/// LEDGERED to P4/P5 (DEFERRED D-45).
#[derive(Clone, Copy, Debug, PartialEq, thiserror::Error)]
pub enum RegionNestError {
    #[error("the forest has {found} ambient roots (parent: None); exactly one is required")]
    RootCount { found: usize },
    #[error(
        "two regions share realm {realm:?}; each realm must be unique (region_depth determinism)"
    )]
    DuplicateRealm { realm: RealmId },
    #[error("region {realm:?} names parent {parent:?}, which is not a region in the forest")]
    DanglingParent { realm: RealmId, parent: RealmId },
    #[error("region {realm:?}'s parent chain cycles or never reaches the single root")]
    CycleOrOrphan { realm: RealmId },
    #[error("the forest has {found} regions; the membership bitset holds at most {max}")]
    TooManyRegions { found: usize, max: usize },
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
}

/// Does a CO-FRAMED child region fit inside its parent's usable interior?
///
/// WHY THIS IS NOW A BOOT FENCE AND NOT A LEDGERED NICETY. Containment hysteresis derives a subject's
/// prior from ANCESTRY: being authoritatively in a realm makes you a member of that realm and of every
/// realm containing it (task #177). That inference is only sound if "containing" is geometrically TRUE.
/// A child poking outside its parent would hand a subject membership of a region it is demonstrably
/// outside — the derived prior would then assert a falsehood every tick, and the containment fold could
/// pick a parent the subject has physically left.
///
/// THE FENCE WAS INERT, AND THIS IS WHAT WAS WRONG WITH IT. It used to begin `if child.frame !=
/// parent.frame { return None }` — and a parent and its child NEVER share a frame. Each realm is centred
/// on itself and holds its own frame; a planet is `PlanetCentered`, its star system `SystemSpace`. So the
/// early return fired on every pair in every real world and the check ran on nothing at all, while its
/// doc-comment went on claiming the ancestry prior was guarded. The frame difference is not an obstacle
/// here: `RealmRegion::center` MEANS "where I sit inside my parent, measured in my parent's frame", so the
/// number is already in the frame the comparison needs.
///
/// The second half of the same mistake was subtracting the parent's own centre. The parent's centre is
/// measured in ITS parent's frame — a different space again — so the subtraction mixed two frames and only
/// looked right while every realm sat at its parent's origin. The child's placement needs no adjustment at
/// all: it is already measured from the parent's centre.
///
/// SCOPE, stated honestly, three ways:
/// - It compares the child's EXACT farthest point ([`Boundary::max_reach_from`]) against the parent's
///   inscribed extent — the largest sphere the parent fully contains. Necessary, not sufficient: a box
///   parent's corners are not promised to anybody.
/// - It compares against the parent's BOUNDARY, deliberately NOT the boundary minus the parent's hysteresis
///   inset. The prior this fence protects says "authoritatively inside the child ⇒ a member of the parent";
///   that is falsified only by a child poking outside the parent's boundary. The inset is a release margin
///   INSIDE that boundary — a subject there is still geometrically within the parent, so the prior holds.
///   MEASURED: subtracting it condemns the shipped walk forest's Area A, whose farthest corner is 9.06 m
///   from a planet of radius 10 with a 1 m inset. That forest is correct; the stricter bound was not.
/// - A MOVING child is checked at its AUTHORED origin. An orbiting body's `center` is zero and its real
///   placement comes from the ephemeris every tick, so for those this fence checks the SIZE and not the
///   orbit. Saying so beats implying an orbit-wide guarantee it cannot give.
fn child_fits_in_parent(child: &RealmRegion, parent: &RealmRegion) -> Option<(f64, f64)> {
    // `center` is expressed in the PARENT's frame, so the parent's tier is the one that scales its cell
    // anchor into metres — never the child's own (which is what `RealmRegion::frame` names).
    let center_in_parent = child
        .center
        .delta_m(LatticePos::local(DVec3::ZERO), parent.frame.tier());
    let reach = child.shape.max_reach_from(center_in_parent);
    let limit = parent.shape.inscribed_extent();
    (reach > limit).then_some((reach, limit))
}

/// The BOOT FENCE for a realm-region forest (task #135, C-5): pure topological validation run at shard
/// boot BEFORE the infallible [`RealmRegions::new`] (in `vd_sim`), so a malformed set fails LOUD rather
/// than degrading to the detector's `root_realm == None` no-op. Rejects, in order (fail-fast + cheap-
/// first): count > `max` (the membership-bitset width, passed by the caller — vd-core stays free of the
/// bitset detail); not exactly one `parent: None` root; a duplicate `.realm`; a dangling parent; a parent
/// chain that cycles or never reaches the root. A straight-line SHIM — each fallible arm is a monomorphic
/// helper, so the `?` branch regions are covered once here (HR5). This validates only TOPOLOGY (the
/// geometric child-⊆-parent subset check is P4/P5, D-45).
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

pub fn guard_regions_nest(regions: &[RealmRegion], max: usize) -> Result<(), RegionNestError> {
    guard_region_count(regions, max)?;
    guard_single_root(regions)?;
    guard_unique_realms(regions)?;
    guard_parents_resolve(regions)?;
    guard_chains_reach_root(regions)?;
    guard_children_fit_parents(regions)?;
    Ok(())
}

/// Every CO-FRAMED child must sit inside its parent's usable interior — see [`child_fits_in_parent`]
/// for why the ancestry-derived containment prior makes this load-bearing.
fn guard_children_fit_parents(regions: &[RealmRegion]) -> Result<(), RegionNestError> {
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
        let parent = regions
            .iter()
            .find(|r| r.realm == parent_id)
            .expect("guard_parents_resolve already refused every unresolvable parent");
        if let Some((reach, limit)) = child_fits_in_parent(child, parent) {
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

fn guard_region_count(regions: &[RealmRegion], max: usize) -> Result<(), RegionNestError> {
    if regions.len() > max {
        return Err(RegionNestError::TooManyRegions {
            found: regions.len(),
            max,
        });
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

fn guard_unique_realms(regions: &[RealmRegion]) -> Result<(), RegionNestError> {
    // O(N²) over a bounded N (≤ `max` ≤ 64): no alloc, no hasher.
    for (i, a) in regions.iter().enumerate() {
        if regions[..i].iter().any(|b| b.realm == a.realm) {
            return Err(RegionNestError::DuplicateRealm { realm: a.realm });
        }
    }
    Ok(())
}

fn guard_parents_resolve(regions: &[RealmRegion]) -> Result<(), RegionNestError> {
    for r in regions {
        if let Some(p) = r.parent
            && !regions.iter().any(|x| x.realm == p)
        {
            return Err(RegionNestError::DanglingParent {
                realm: r.realm,
                parent: p,
            });
        }
    }
    Ok(())
}

/// Every region's parent chain must reach the single root within `regions.len()` hops (a longer walk
/// must cycle — parents resolve + one root ⇒ acyclic reaches the root). Runs AFTER the single-root /
/// unique / resolve guards, so the `.parent` lookups are total. Bounded ⇒ never hangs on a cycle.
fn guard_chains_reach_root(regions: &[RealmRegion]) -> Result<(), RegionNestError> {
    for r in regions {
        if !chain_reaches_root(regions, r.realm) {
            return Err(RegionNestError::CycleOrOrphan { realm: r.realm });
        }
    }
    Ok(())
}

fn chain_reaches_root(regions: &[RealmRegion], start: RealmId) -> bool {
    let mut cur = start;
    for _ in 0..regions.len() {
        match regions
            .iter()
            .find(|x| x.realm == cur)
            .and_then(|x| x.parent)
        {
            None => return true, // reached the ambient root (`parent: None`)
            Some(p) => cur = p,
        }
    }
    false // exceeded `len` hops without reaching the root ⇒ a cycle
}

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;

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
            ..test_region(realm, parent)
        }
    }

    fn test_region(realm: RealmId, parent: Option<RealmId>) -> RealmRegion {
        RealmRegion {
            realm,
            center: LatticePos::local(DVec3::ZERO),
            frame: FrameRef::SystemSpace { system_seed: 0 },
            shape: Boundary::Shell { r: 1.0 },
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
        let region = test_region(RealmId::System(7), None);
        // Ok arm: an already-reframed pose at the unit shell's center ⇒ signed distance -r = -1.
        assert_eq!(
            region_signed_distance_resolved(Ok(test_pose()), &region),
            Ok(-1.0)
        );
        // Err arm: a frame error propagates (the P4/P5 unknown-frame degrade; inert under identity).
        assert_eq!(
            region_signed_distance_resolved(Err(FrameError::UnknownSourceFrame), &region),
            Err(FrameError::UnknownSourceFrame)
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

    #[test]
    fn guard_regions_nest_accepts_a_valid_forest() {
        assert_eq!(guard_regions_nest(&valid_forest(), 64), Ok(()));
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
        let err =
            guard_regions_nest(&escaping, 64).expect_err("an escaping child must be rejected");
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
            center: LatticePos::local(DVec3::new(8.0, 0.0, 0.0)),
            ..test_region_r(RealmId::Planet(1), Some(RealmId::System(1)), 5.0)
        };
        assert_eq!(
            child_fits_in_parent(&escaping, &parent),
            Some((13.0, 10.0)),
            "a cross-frame child that reaches 13 m out of a 10 m parent must be caught"
        );
        // …and the accept twin across the same frame boundary, so a fence that simply always rejects
        // would fail here.
        let fitting = RealmRegion {
            center: LatticePos::local(DVec3::new(4.0, 0.0, 0.0)),
            ..escaping
        };
        assert_eq!(child_fits_in_parent(&fitting, &parent), None);
    }

    #[test]
    fn a_box_child_is_measured_by_its_farthest_corner_not_the_sphere_around_it() {
        // The shipped walk forest's Area A, to the metre: a half-3 box centred 5 m out inside a planet of
        // radius 10. Its farthest corner is at (8,3,3) ⇒ 9.06 m, so it FITS. The spherical over-estimate
        // (5 + |(3,3,3)| = 10.20) called it an escape, which is why the exact corner is the measure.
        let parent = test_region_r(RealmId::Planet(7), Some(RealmId::System(7)), 10.0);
        let area = RealmRegion {
            center: LatticePos::local(DVec3::new(5.0, 0.0, 0.0)),
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
        assert_eq!(child_fits_in_parent(&area, &parent), None);
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
            center: LatticePos::local(DVec3::new(5.0, 0.0, 0.0)),
            ..test_region_r(RealmId::Planet(1), Some(RealmId::System(1)), 3.0)
        };
        let forest = vec![
            test_region_r(RealmId::System(0), None, 100.0),
            test_region_r(RealmId::System(1), Some(RealmId::System(0)), 10.0),
            offset_child, // reaches 5 + 3 = 8, inside the parent's 10
        ];
        assert_eq!(guard_regions_nest(&forest, 64), Ok(()));
    }

    #[test]
    fn guard_regions_nest_rejects_too_many_regions() {
        // 3 regions with `max = 2` ⇒ the count fence trips FIRST (before any topology walk).
        assert_eq!(
            guard_regions_nest(&valid_forest(), 2),
            Err(RegionNestError::TooManyRegions { found: 3, max: 2 })
        );
    }

    #[test]
    fn guard_regions_nest_rejects_zero_or_two_roots() {
        // TWO roots.
        let two_roots = vec![
            test_region(RealmId::System(0), None),
            test_region(RealmId::System(1), None),
        ];
        assert_eq!(
            guard_regions_nest(&two_roots, 64),
            Err(RegionNestError::RootCount { found: 2 })
        );
        // ZERO roots (a pure cycle — caught by the root count BEFORE the chain walk).
        let no_root = vec![
            test_region(RealmId::System(1), Some(RealmId::System(2))),
            test_region(RealmId::System(2), Some(RealmId::System(1))),
        ];
        assert_eq!(
            guard_regions_nest(&no_root, 64),
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
            guard_regions_nest(&dup, 64),
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
            guard_regions_nest(&dangling, 64),
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
            guard_regions_nest(&[parent, child], 64),
            Ok(()),
            "a differently-framed child is not judged here, so the forest is accepted"
        );

        // The SAME child in the SAME frame IS judged — and a huge one is refused. Without this half the
        // test above would pass for the wrong reason (nothing is ever judged).
        let mut oversized = test_region_r(RealmId::Planet(1), Some(RealmId::System(0)), 5_000.0);
        oversized.frame = parent.frame;
        assert!(
            guard_regions_nest(&[parent, oversized], 64).is_err(),
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
            guard_regions_nest(&cyclic, 64),
            Err(RegionNestError::CycleOrOrphan {
                realm: RealmId::System(1)
            })
        );
    }

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
