//! THE VISIBILITY CLIMB: how far a body's own picture must travel before it can stop being drawn.
//!
//! Owns: the per-pair verdict numbers (a body at its worst instant, a viewer just outside an
//! ancestor's boundary at closest approach), the climb measurement over a whole forest, and the two
//! fences it feeds — the boot fence over the generated world and the admission fence a player's
//! candidate build must pass.
//!
//! Does NOT own: any realm kind or motion kind. A hop's excursion is a magnitude whichever way it is
//! produced, and the threshold enters as the same angular size the interest band already uses — one
//! rule, applied at two moments.

use super::{
    GeneratedBody, Placement, UniverseConfig, VISIBILITY_THETA_MIN_RAD, generate_system_forest,
    system_forest_cached,
};
#[cfg(test)]
use crate::motion::Motion;
use glam::DVec3;
use vd_core::geometry::Boundary;
#[cfg(test)]
use vd_core::geometry::visibility_factor;
use vd_core::pose::RealmId;
#[cfg(test)]
use vd_core::pose::Tier;

/// A grandchild-or-deeper body that would be VISIBLE from just outside one of its ancestors — the
/// two-level bound broken by geometry. Carries every number of the verdict so the failure names
/// itself: the worst-instant distance of the body's centre from the ancestor's centre, the body's
/// extent, the resulting minimum viewer distance, and the distance the visibility rule requires.
#[derive(Clone, Copy, Debug, PartialEq, thiserror::Error)]
#[error(
    "{body:?} subtends >= the visibility threshold from just outside its ancestor {ancestor:?}: \
     worst-instant centre distance {worst_dist_m} m, extent {extent_m} m, minimum viewer distance \
     {d_min_m} m, but the interest band keeps it visible out to {required_m} m — a not-running \
     realm two levels down would owe pixels its parent's placement marker cannot author"
)]
pub struct GrandchildVisibleOutside {
    pub body: RealmId,
    pub ancestor: RealmId,
    /// Worst-instant distance of the body's centre from the ancestor's centre (triangle bound —
    /// each hop contributes its own worst-instant offset magnitude).
    pub worst_dist_m: f64,
    /// The body's finite extent (the same extent the interest band judges visibility on).
    pub extent_m: f64,
    /// `R_ancestor − worst_dist − extent`: how close a viewer just outside the ancestor can get.
    pub d_min_m: f64,
    /// `extent · cot(θ_min/2)`: the distance out to which the interest band keeps the body visible.
    pub required_m: f64,
}

/// One hop's WORST-INSTANT offset magnitude — the same machinery the boot's `ChildReach` roster
/// states: a static child's authored offset, a mover's closed-form worst-instant excursion
/// ([`Motion::max_excursion_m`], the apoapsis — never a re-derived `a·(1+e)` beside it).
/// TEST-ONLY since look_horizon slice 2: the boot fence measures with the CAPPED excursion
/// ([`worst_hop_excursion_capped_m`] — the solve's own worst case); the drawn-eccentricity walk
/// below stays as the pinned drawn-margin history.
#[cfg(test)]
pub(crate) fn worst_hop_excursion_m(placement: &Placement) -> f64 {
    match placement {
        Placement::StaticOffset(at) => at.length(),
        Placement::Orbital(elements) => Motion::Kepler(*elements).max_excursion_m(Tier::Fine),
    }
}

/// The two-level VERDICT NUMBERS for EVERY `(body, ancestor)` pair — ancestor two or more levels
/// up — with the body at its worst-instant position and the viewer just outside the ancestor's
/// boundary at closest approach. PURE GEOMETRY over the roster: no realm kinds, no motion kinds (a
/// hop's excursion is a magnitude whichever way it is produced). The threshold enters as the SAME
/// `cot(θ/2)` the interest band uses ([`visibility_factor`]) — the condition
/// `angular_size(extent, d_min) < θ_min` is exactly `d_min > extent · cot(θ_min/2)`. A pair is an
/// OFFENCE iff `d_min ≤ required` ([`grandchild_visibility_offences`] filters); a green pair's
/// margin `d_min − required` is the measured headroom the re-solve pins.
#[cfg(test)]
pub(crate) fn grandchild_visibility_pairs(
    bodies: &[GeneratedBody],
    theta_min_rad: f64,
) -> Vec<GrandchildVisibleOutside> {
    let by_id: std::collections::BTreeMap<RealmId, &GeneratedBody> =
        bodies.iter().map(|b| (b.realm, b)).collect();
    let factor = visibility_factor(theta_min_rad);
    let mut pairs = Vec::new();
    for body in bodies {
        // The subject's PICTURE is its LOOK (real-scale design §3.0); a look-less body draws
        // nothing and has no two-level visibility question.
        let Some(look) = body.look else { continue };
        let extent_m = look.finite_extent();
        // Walk the ancestor chain, accumulating the worst-instant centre distance hop by hop.
        let mut worst_dist_m = worst_hop_excursion_m(&body.placement);
        let mut hops = 1_usize;
        let mut cursor = body.parent;
        while let Some(ancestor_id) = cursor {
            let ancestor = by_id
                .get(&ancestor_id)
                .expect("the generated forests resolve every parent (guarded at boot)");
            if hops >= 2 {
                let d_min_m = ancestor.shape.finite_extent() - worst_dist_m - extent_m;
                let required_m = extent_m * factor;
                pairs.push(GrandchildVisibleOutside {
                    body: body.realm,
                    ancestor: ancestor_id,
                    worst_dist_m,
                    extent_m,
                    d_min_m,
                    required_m,
                });
            }
            worst_dist_m += worst_hop_excursion_m(&ancestor.placement);
            hops += 1;
            cursor = ancestor.parent;
        }
    }
    pairs
}

/// Every pair of [`grandchild_visibility_pairs`] that IS an offence — the body would still be
/// VISIBLE (subtend ≥ `theta_min_rad`) from just outside its ancestor, equality included (the
/// margin the shell solve reserves is what keeps the worst lawful seed strictly clear).
#[cfg(test)]
pub(crate) fn grandchild_visibility_offences(
    bodies: &[GeneratedBody],
    theta_min_rad: f64,
) -> Vec<GrandchildVisibleOutside> {
    grandchild_visibility_pairs(bodies, theta_min_rad)
        .into_iter()
        .filter(|p| p.d_min_m <= p.required_m)
        .collect()
}

// ===== THE LOOK HORIZON's boot MEASUREMENT (look_horizon.md §3.3.2 — slice 2) ==================
// The boolean guard this replaces (`guard_grandchildren_invisible_outside`) gave a yes-or-no
// answer over the seed forest ONLY — §3.3.1 proves that cannot serve as a termination proof (the
// generator emits universe/galaxy/systems/planets and nothing else, so player-built content never
// entered it, and its predicate would refuse the first city). The MEASUREMENT below reports a
// NUMBER per body — how many levels its picture must travel — and the fence refuses a world whose
// number exceeds what the look carrier can carry (`vd_wire::session_flow::LOOK_CARRIER_ARITY`).

/// One hop's worst-instant offset at the ECCENTRICITY CAP — the SAME worst-case convention the
/// shell solve bounds against (`galaxy_shell_r_m`'s `outer_sma · (1 + ecc_cap)`), which is what
/// makes the measured stopping slack and the solve's reserved margin ONE equation written twice
/// (§3.3.2's identity; the drawn-eccentricity margin is the LOOSER `FROZEN_TWO_LEVEL_WORST_MARGIN_M`
/// — the engineering-relevant number is the reserved one). Straight-line per arm (HR5).
pub(crate) fn worst_hop_excursion_capped_m(placement: &Placement, ecc_cap: f64) -> f64 {
    match placement {
        Placement::StaticOffset(at) => at.length(),
        Placement::Orbital(elements) => elements.sma * (1.0 + ecc_cap),
    }
}

/// One body's measured VISIBILITY CLIMB (look_horizon.md §3.3.2): how far its own picture must
/// travel for every lawful observer to draw it.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct VisibilityClimb {
    pub body: RealmId,
    /// The HIGHEST ancestor from just outside which the body is STILL visible (the body itself
    /// when not even its parent's outside can see it — the degenerate one-level climb).
    pub top: RealmId,
    /// How many levels the body's own picture must travel: `1` (its own batch reaching its
    /// parent — every child's baseline) plus one per consecutive ancestor, parent upward, from
    /// outside which it is still visible.
    pub levels: usize,
    /// The slack at the level where visibility STOPPED: `d_min − required` at the first ancestor
    /// that does NOT see the body (POSITIVE — the measured headroom §3.3.2 pins at the world's
    /// own containment margin), or the ROOT's non-positive figure when the climb never stopped
    /// inside the forest (a pathological world the arity fence then refuses).
    pub slack_m: f64,
}

/// The climb walk over a generated forest — the SAME ancestor walk as
/// [`grandchild_visibility_pairs`] with the `hops >= 2` filter dropped (§3.3.2's construction)
/// and the excursions taken at the eccentricity CAP (the solve's own worst case). Bodies without
/// a parent (the root) have no climb and report nothing.
pub(crate) fn visibility_climbs(
    bodies: &[GeneratedBody],
    theta_min_rad: f64,
    ecc_cap: f64,
) -> Vec<VisibilityClimb> {
    let by_id: std::collections::BTreeMap<RealmId, &GeneratedBody> =
        bodies.iter().map(|b| (b.realm, b)).collect();
    let mut climbs = Vec::new();
    for body in bodies.iter().filter(|b| b.parent.is_some()) {
        // The climb carries the body's PICTURE — its LOOK (real-scale design §3.0). A body
        // with no look draws nothing, so there is no picture to carry and no climb (the
        // ambient galaxy's levels-2 root artifact of the outer re-solve dissolves here).
        let Some(look) = body.look else { continue };
        let extent_m = look.finite_extent();
        let required_m = vd_core::geometry::visibility_reach_m(extent_m, theta_min_rad);
        let mut worst_dist_m = worst_hop_excursion_capped_m(&body.placement, ecc_cap);
        let mut levels = 1_usize;
        let mut top = body.realm;
        let mut slack_m = f64::INFINITY;
        let mut cursor = body.parent;
        while let Some(ancestor_id) = cursor {
            let ancestor = by_id
                .get(&ancestor_id)
                .expect("the generated forests resolve every parent (guarded at boot)");
            let d_min_m = ancestor.shape.finite_extent() - worst_dist_m - extent_m;
            slack_m = d_min_m - required_m;
            if slack_m > 0.0 {
                break; // NOT visible from outside this ancestor: the climb stops HERE.
            }
            // Still visible (equality included — the same convention as the offence filter):
            // the picture must travel one level further.
            top = ancestor_id;
            levels += 1;
            worst_dist_m += worst_hop_excursion_capped_m(&ancestor.placement, ecc_cap);
            cursor = ancestor.parent;
        }
        climbs.push(VisibilityClimb {
            body: body.realm,
            top,
            levels,
            slack_m,
        });
    }
    climbs
}

/// THE BOOT MEASUREMENT (look_horizon.md §3.3.2, replacing the boolean guard): for every body of
/// the generated world, how many levels its picture must travel — the highest still-visible
/// ancestor, and the slack at the level where visibility stopped. On THE world today: max climb
/// 2, planet stopping slack exactly the containment margin (the solve identity, G-CLIMB's pin).
#[must_use]
pub fn measure_visibility_climb(
    seed_universe: u64,
    config: &UniverseConfig,
) -> Vec<VisibilityClimb> {
    visibility_climbs(
        &system_forest_cached(seed_universe, config),
        VISIBILITY_THETA_MIN_RAD,
        config.planet.ecc_cap,
    )
}

/// A world (or a candidate placement) whose measured visibility climb EXCEEDS what the look
/// carrier can carry — the fail-loud shape the fences print: the body and its numbers.
#[derive(Clone, Copy, Debug, PartialEq, thiserror::Error)]
#[error(
    "{body:?} needs its picture carried {levels} levels (visible from outside every ancestor up \
     to {top:?}; slack at the stop {slack_m} m), but the look carrier serves {arity} — the owner's \
     Q3 ruling (look_horizon.md RULINGS 2026-08-17): the arity STAYS 2 and this REFUSES at interim \
     scale; the near-real-scale re-solve is the scheduled cure, and its first gate run must \
     include measure_visibility_climb"
)]
pub struct VisibilityClimbExceeded {
    pub body: RealmId,
    pub top: RealmId,
    pub levels: usize,
    pub slack_m: f64,
    pub arity: usize,
}

/// The one comparison both fences share (monomorphic, both arms driven by named tests — HR5).
pub(crate) fn first_climb_over(
    climbs: &[VisibilityClimb],
    arity: usize,
) -> Result<(), VisibilityClimbExceeded> {
    match climbs.iter().find(|c| c.levels > arity) {
        Some(c) => Err(VisibilityClimbExceeded {
            body: c.body,
            top: c.top,
            levels: c.levels,
            slack_m: c.slack_m,
            arity,
        }),
        None => Ok(()),
    }
}

/// THE BOOT FENCE (look_horizon.md §3.3.4 instrument 1, wired into EVERY world-deriving
/// process's boot — the shard AND the gateway): the generated world's required climb must not
/// exceed the look carrier's arity. A refusal is a measurement; a wrong pixel is not.
///
/// # Errors
/// [`VisibilityClimbExceeded`] naming the first offending body with its numbers.
pub fn guard_visibility_climb_bounded(
    seed_universe: u64,
    config: &UniverseConfig,
    arity: usize,
) -> Result<(), VisibilityClimbExceeded> {
    first_climb_over(&measure_visibility_climb(seed_universe, config), arity)
}

/// A candidate player-built region put to the build-admission fence: a STATIC body (SL4 — a
/// built structure does not orbit) of `shape` at `offset_m` in `parent`'s frame.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CandidateRegion {
    pub realm: RealmId,
    pub parent: RealmId,
    pub shape: Boundary,
    pub offset_m: DVec3,
}

/// THE BUILD-ADMISSION FENCE (look_horizon.md §3.3.4 instrument 2 — D-LOOK-1): a candidate
/// placement whose required climb exceeds the carrier's arity is REFUSED — the PLACEMENT, never
/// the boot. The candidate joins the world's own generated forest (SL5: THE world, no variant)
/// and is measured by the same walk, the same formula, the same worst-case convention. At
/// today's interim scale a ~20 m surface structure measures a climb of 3 and refuses — the Q3
/// evidence, produced by a test rather than an argument; the near-real-scale re-solve is the
/// scheduled cure (owner ruling 2026-08-17).
///
/// # Errors
/// [`VisibilityClimbExceeded`] naming the candidate with its numbers.
pub fn guard_candidate_climb_bounded(
    candidate: &CandidateRegion,
    seed_universe: u64,
    config: &UniverseConfig,
    arity: usize,
) -> Result<(), VisibilityClimbExceeded> {
    let mut bodies = generate_system_forest(seed_universe, config);
    bodies.push(GeneratedBody {
        realm: candidate.realm,
        parent: Some(candidate.parent),
        shape: candidate.shape,
        placement: Placement::StaticOffset(candidate.offset_m),
        photometrics: None,
        // A built structure draws itself at its own bound (bound == look at build scale).
        taxon: None,
        look: Some(candidate.shape),
    });
    let climbs = visibility_climbs(&bodies, VISIBILITY_THETA_MIN_RAD, config.planet.ecc_cap);
    first_climb_over(
        &climbs
            .into_iter()
            .filter(|c| c.body == candidate.realm)
            .collect::<Vec<_>>(),
        arity,
    )
}
