//! THE REAL-SCALE DERIVATION: four numbers, each solved rather than chosen.
//!
//! Owns: the universe radius (solved from what the position store can represent), the galaxy radius,
//! the placement radius that is the gap between stars, the between-systems compression that gap
//! implies, and the derived mass cap — the greatest star this galaxy can host, found by a monotone
//! solve rather than declared.
//!
//! Does NOT own: any of them twice. Two spellings of one derivation is how a world quietly stops
//! agreeing with itself, so each number has exactly one home here and every consumer reads it.

use super::guards::ROOT_TIER;
use super::{
    IMF_MASS_HI_PHYSICAL_MSUN, IMF_MASS_LO_MSUN, PlanetConfig, StarPhotometrics, system_shell_r_m,
    world_planet_config,
};
use crate::taxonomy::{SpectralClass, classify_spectral, main_sequence_luminosity};
/// THE ONE visibility formula, re-exported from its home (look_horizon.md §3.3.2 — the formula
/// lives in `vd-core` with three consumers: the world solve, the boot measurement, the runtime
/// tripwire; this module is two of them and BUILDS the third's config). Never a second copy.
use vd_core::geometry::visibility_factor;
use vd_core::pose::CELL_DOMAIN_MAX;

/// The minimum angular size (radians) a realm must subtend to enter Area of Interest — a realm is visible
/// (streams in) out to `extent · cot(θ/2) + velocity-lead`. ONE config constant, no kind-branch: this
/// single number sets how far away EVERY realm kind appears, from a planet to a galaxy.
///
/// 1.5° (was 8°). Two independent measurements forced it down, and both are about the same thing — at 8°
/// a body only exists once it is ~14 of its own radii away, which is close enough to be already large:
///
/// 1. A planet was invisible from inside its own star system. Its wake radius was 59.5 m against a system
///    150 m in radius, so you could cross a system's boundary and find it apparently empty — measured, and
///    now pinned by `a_planet_is_visible_from_anywhere_inside_its_own_system`.
/// 2. Bodies POPPED IN at full size instead of growing from a dot, which is the opposite of the owner's
///    stated arrival/departure behaviour (a system shrinks to a dot as you leave, grows from one as you
///    arrive). Angular size IS that behaviour: the smaller this angle, the smaller a body is when it first
///    appears. At 1.5° it enters the scene ~76 radii out — a few pixels — and grows all the way in.
///
/// COST, stated because it is real and pays at every scale: every realm's wake radius grows by the same
/// 5.3×, so more realms are awake at once and the demand loop carries more of them. That is the ONE knob's
/// nature — it cannot be widened for planets alone without branching on kind, which is forbidden. Note the
/// inter-system ring is DERIVED from this same factor ([`UniverseConfig::visual_geometry`]), so a system
/// still sleeps until you approach it no matter what this is set to — that behaviour is invariant here.
pub(crate) const VISIBILITY_THETA_MIN_RAD: f64 = vd_core::geometry::VISIBILITY_THETA_MIN_RAD;

// ===== THE OUTER GEOMETRY (real-scale addendum §A2 — owner ruling 2026-08-18, OPTION C) =========
// The four changed numbers, each with its derivation. The chain runs DOWNWARD from the storage
// fence for the root and DOWNWARD from the star for everything inside a system; the two join at
// exactly one place (the placement radius), which is why everything in-system is BIT-IDENTICAL
// across this re-solve (§A1.1) — pinned by the unchanged in-system goldens.

/// ▲ 1. THE UNIVERSE RADIUS: `2⁵¹ m` — solved from the position store, never chosen (§A2.1/§A2.2).
/// The binding budget is F1, the wire-ingress domain: every per-axis cell of a sanitized pose must
/// survive `StampedPose::sanitized` unclamped, `|cell| ≤ CELL_DOMAIN_MAX = i64::MAX/2` (the clamp
/// exists so a cell DIFFERENCE cannot overflow — the hostile-sender panic cure). With one binary
/// octave of headroom against that clamp (`K_SPAN = 2` — the same discipline constant the exactness
/// budget always carried, with a new referent, §A2.2/OQ-3: a LAWFUL position may sit OUTSIDE the
/// root shell, so the clamp must not be reachable from there):
///
/// `K_SPAN · R_uni / FINE_CELL_EDGE_M ≤ CELL_DOMAIN_MAX + 1`
/// `R_uni = 2⁵¹ m = 2 251 799 813 685 248 m ≈ 0.238 ly` — passes with EXACT equality:
/// `2 × 2⁶¹ = 4 611 686 018 427 387 904 = CELL_DOMAIN_MAX + 1` (headroom exactly 2.0000×, domain
/// occupancy exactly 50.0000 %). `2⁵¹` is exactly f64-representable, so the fence and the shipped
/// shell radius agree bit-for-bit. [`guard_root_representable`] is the boot fence on this budget —
/// and its refusal is THE NAMED P10 TRIGGER (R3): a world that outgrows the FINE lattice is the day
/// the galaxy cell lattice is needed.
pub(crate) const REAL_UNIVERSE_R_M: f64 = root_radius_at(ROOT_TIER);

/// ★ THE ROOT RADIUS AS THE FENCE SOLVED AT EQUALITY (slice S8), never a typed-in number.
///
/// It used to be the literal `2_251_799_813_685_248.0` with a comment explaining that it came from the
/// step. A comment cannot follow a change: re-value the step and the literal stays where it was, the two
/// disagree, and only one pin at one rung would notice. Now the radius IS its derivation.
///
/// The fence asks that `K_SPAN · (R / step) ≤ CELL_DOMAIN_MAX + 1`. Solved at EQUALITY — which is the
/// construction, not slack — that is `R = (CELL_DOMAIN_MAX + 1) · step / K_SPAN`, i.e. `2⁶¹` cells at
/// whatever the rung's step happens to be. So occupancy is exactly one half and headroom exactly two at
/// EVERY rung, by construction rather than by three separate coincidences.
///
/// At today's millimetre root that is `2⁵¹ m` — the same number as before, to the bit.
pub(crate) const fn root_radius_at(tier: vd_core::pose::Tier) -> f64 {
    (CELL_DOMAIN_MAX as f64 + 1.0) * tier.cell_edge_m() / K_SPAN
}

/// The storage-fence headroom octave (§A2.2, `K_SPAN = 2`) — one binary octave between the root
/// shell and the sanitize clamp, so the clamp is unreachable from any lawful position (OQ-3's
/// recommendation, adopted: a silent clamp is the defect class this coordinate exists to prevent).
pub(crate) const K_SPAN: f64 = 2.0;

/// ▲ 2. THE GALAXY RADIUS — ★ ITS OWN FENCE, SOLVED AT EQUALITY (slice S9). This retires a
/// derivation (`R_uni − outset`, the τ-free band outset of §A2.2) that only ever existed because the
/// galaxy had no lattice of its own.
///
/// It used to be `R_universe MINUS a band`: the galaxy had to leave room inside the ROOT's storage
/// budget, because the galaxy and everything in it counted in the root's cells. S7 recorded that as a
/// SINGLE-LATTICE ARTEFACT and said the subtraction would die once the galaxy had its own lattice. It
/// has one now, so the subtraction is gone — the galaxy's radius is its own fence solved at equality,
/// exactly as the root's is.
///
/// `2⁶¹` cells at the galaxy's own two-metre step = **2⁶² m = 4.611686e18 m = 487.46 light years**,
/// against the 0.475 light years it had while it was counted in millimetres — a factor of 1,026. That is
/// what makes a hundred and fifty thousand star systems geometrically possible at all.
pub(crate) const REAL_GALAXY_R_M: f64 = root_radius_at(vd_core::pose::Tier::Galaxy);

/// The owner's acceptance bar ("journeys in MINUTES") — THE speed law's one policy number, whose
/// single home is [`vd_core::flight::TRAVERSE_S`] (the S3 slice landed the law:
/// `realm_speed_cap_mps`, the geometric throttle, the ramp and the governor all live in
/// `vd_core::flight`, consumed at the sim's one integrator seam). The geometry solve reads the SAME
/// constant through this alias (its τ-free outset above), so the shells and the speed law can never
/// disagree about T — one number, two readers, zero drift.
pub(crate) const T_TRAVERSE_S: f64 = vd_core::flight::TRAVERSE_S;
/// The geometry solve's FIXED reference tick (§A2.2's `tick_dt` input — a constant of the solve,
/// NEVER the cluster's live tick: two clusters at different tick rates must boot the identical
/// world, so no cluster parameter may enter a radius).
pub(crate) const GEOMETRY_TICK_DT_S: f64 = 0.02;
/// The geometry solve's FIXED reference FOOT SPEED — the same rule as [`GEOMETRY_TICK_DT_S`], one
/// argument further along. A band is part of the world's geometry, so it may not depend on how fast one
/// deployment happens to let a person walk, any more than a radius may depend on how fast that
/// deployment ticks. The value is the shipped default (`VD_SPEED`), read here as a constant of the
/// solve rather than from the cluster.
///
/// MEASURED, so that this is a bound and not a hope: across the derived seed sweep, the foot speed
/// binds on **0 of 13,144** boundaries of THE world — every generated realm is large enough that its
/// own geometric ceiling (`2·extent/T`) dominates. So this constant does not size a single band that
/// ships; it exists so that the small-realm arm has a lawful value instead of a cluster's.
pub(crate) const GEOMETRY_V_FOOT_MPS: f64 = 500.0;
/// The largest share of its own body a boundary's band may occupy (real-scale addendum §A3.4's
/// `BAND_EXTENT_CLAMP`). The acquire edge sits one third of the band inside the surface, so half the
/// inscribed extent leaves that edge at five sixths of the body — enterable at every size.
///
/// A band that WANTS to be wider than this belongs to a boundary whose own ceiling outruns its own
/// size. The clamp keeps the realm enterable and does not hide the condition; the ceiling fence names it.
pub(crate) const BAND_EXTENT_CLAMP: f64 = 0.5;
/// `N = max(K_SAFETY, n_entry) = 3` — the in-band tick count the band law owes (§A2.2/§A3.4).
pub(crate) const BAND_TICKS_N: f64 = 3.0;
/// The band-solvability fence's own factor (`dt·N ≤ τ/2` ⇒ the τ term can at most DOUBLE the
/// governed band) — not a new literal, the fence's ×2 (§A2.2, H-02's cure).
pub(crate) const BAND_TAU_HEADROOM: f64 = 2.0;

/// ▲ 3. THE PLACEMENT RADIUS (the star gap): `R_gal − clearance = 1.498979587153876e15 m
/// = 0.15843 ly = 10 019.98 AU` (★ RE-SOLVED 2026-08-20 with the DERIVED mass cap; it read
/// 2.248490504408914e15 m = 0.2376656 ly while the reservation was seed 0's own sample)
/// — what the storage budget leaves after the clearance the solve
/// owes (§A2.2/§A2.3; `VISUAL_RING_SLACK`, the old "THROWAWAY" padding fraction, is DELETED — the
/// radius is never padded by taste again). The clearance is the ONE clearance law (§3.2)
/// evaluated at the FUTURE in-system re-solve's targets, landed NOW so the outer geometry never
/// moves again when the in-system slices (taxonomy: star radius, real SOIs, shells) arrive:
///
/// `clearance = child_clearance(bound, look) = max(bound, look·(1+cot(θ/2))) + look·(1+cot(θ/2))`
/// with `bound = R_sys,max = 2.967026419e11 m` (the largest target system shell, §3.3.5 — solved
/// from the Demircan–Kahraman radius + re-anchored ladder of `System(13979593561158050752)`) and
/// `look = R★,max = 1.318892e8 m` (that star's photosphere radius under the same mass–radius law
/// from its ALREADY-PINNED mass draw 0.16179874709518627 M☉). Both enter as CITED derived targets
/// of the addendum's chain — the in-system machinery that recomputes them lands with the taxonomy
/// slice, and the named pin below flips loudly if that slice lands different numbers.
/// ★ RE-MEASURED AT THE FLAG DAY (the taxonomy arc's in-system re-solve): the shells became
/// SOLVED by `system_shell_r_m`, and the solved values superseded the addendum's hand-derived
/// 2.967026419e11 / 1.575690652e11.
///
/// ★★ AND THEN THE SAMPLE BROKE THE WORLD (owner ruling 2026-08-20, measured on the live cluster).
/// Both numbers above were SEED-0 SAMPLES wearing the name of a reservation: 2.967_034_259_82e11 m
/// was the shell of seed 0's heaviest star, an M dwarf of 0.1618 M☉. The owner flew a different
/// seed. Its galaxy drew a 0.18816 M☉ sibling whose shell solves to 3.525e11 m — 19 % past the
/// reservation — so the sibling poked 45 603 431 137 m through the galaxy's shell and the nest
/// fence refused to boot the galaxy shard, verbatim:
///
/// > region System(10487570625701098367) is not geometrically inside its parent System(1): it
/// > reaches 2248843017364804.8 m from the parent's centre but the parent's usable interior ends
/// > at 2248797413933667.8 m — refusing to boot
///
/// No galaxy ⇒ nowhere for an outbound crossing to land ⇒ the pilot drifts out of their own system
/// forever, and no real star ever streams. A per-child fit clamp was tried and REVERTED: moving a
/// child to where it fits disarms the very fence whose refusal is the world's negative control.
///
/// THE DEEPER CONTRADICTION the ruling closes: the mass draw ran to 120 M☉, and a 120 M☉ star's
/// ladder (anchored at `0.4·√L` AU) solves to a 2.43e16 m shell — TEN TIMES WIDER THAN THE WHOLE
/// GALAXY. No constant reservation could ever have covered every star the world may draw, so the
/// rule and the world contradicted each other. The cure is not a bigger constant: the CAP and the
/// RESERVATION become two readings of ONE derivation ([`DERIVED_MASS_CAP`]), and the fit is then
/// structural for every seed — see [`imf_mass_hi_msun`].
///
/// THE RESERVED SYSTEM BOUND: the system shell AT the derived mass cap. Every drawable star is at
/// or below the cap and the shell grows monotonically with mass, so `placement + any drawable
/// system's bound ≤ galaxy_r` holds for every seed BY CONSTRUCTION. `pub` for the flight table and
/// the story fixture, which size themselves off THE reservation rather than restating one.
#[must_use]
pub fn target_system_bound_max_m() -> f64 {
    DERIVED_MASS_CAP.system_bound_max_m
}

/// The largest star's photosphere radius (§3.3.1's `R★`) — `star_radius_m` AT the derived cap, the
/// `look` half of the clearance the galaxy reserves for its largest child.
pub(crate) fn target_star_look_max_m() -> f64 {
    DERIVED_MASS_CAP.star_look_max_m
}

/// ★ THE DERIVED MASS CAP — the greatest star THIS galaxy can host, and the reservation it implies.
///
/// WHY A CAP EXISTS AT ALL. This galaxy is 2.2487974139336678e15 m in radius — 0.2376981 ly. A
/// star's system shell grows with its mass (the orbit ladder is anchored at `0.4·√L` AU and `L`
/// climbs steeply with `M`), and above a certain mass the system is simply WIDER THAN THE GALAXY
/// THAT WOULD CONTAIN IT. Such a star was never possible here; until this ruling the world merely
/// failed to say so, and drew one anyway. The cap is the world stating its own geometry.
///
/// WHAT THE CAP IS, EXACTLY. The largest mass the galaxy can still PAY FOR, where the price of one
/// child of mass `m` is the §3.2 clearance the galaxy owes it plus the two system bounds that lie
/// on the line between the origin-anchored home system and any sibling:
///
/// `demand(m) = child_clearance_m(shell(m), R★(m), θ) + 2·shell(m) ≤ R_gal`
///
/// Every term is a shipped law, not a taste. The first is the ONE clearance law the placement
/// radius already subtracts (`placement = R_gal − clearance`). The `2·shell` is the SEPARATION
/// FENCE'S OWN INEQUALITY ([`seeded_systems_disjoint_3d`]: two siblings must be farther apart than
/// the sum of their extents) evaluated on the ONE pair whose geometry the construction fixes — the
/// home system is anchored at the galactic origin and every sibling sits at exactly the placement
/// radius, so that pair's separation IS the placement radius, and requiring `placement ≥ 2·shell`
/// at the cap makes the pair disjoint for every seed. (Sibling-against-sibling separation depends
/// on two seeded DIRECTIONS and can never be structural; it stays the boot fence's per-seed job.)
/// Dropping the `2·shell` term instead solves to a cap whose placement radius is ZERO — every
/// system stacked on the galactic centre. The strict form is also the CHEAPER one: it keeps two
/// thirds of the star gap where the loose form keeps none.
///
/// WHY IT LIFTS. The cap is a fact about a galaxy of THIS radius, and the radius is the FINE
/// lattice's storage budget (see [`guard_root_representable`]). When the galaxy cell lattice lands
/// (P10) the galaxy stops being one bounded shell and the cap rises with it — the same trigger that
/// drives the compression χ toward 1. Ledgered: `docs/design/DEFERRED.md` D-MASS-CAP.
///
/// HOW IT IS SOLVED. `demand` is strictly increasing in mass (both `shell` and `R★` are), so the
/// cap is the root of `demand(m) = R_gal`. The shell law is a `max` over nine ladder rungs of
/// piecewise power laws and does not invert in closed form, so it is a MONOTONE SOLVE: bracket by
/// doubling from the hydrogen-burning limit until the galaxy cannot pay, then bisect. The bracket
/// closes at relative width 1 and is halved [`MASS_CAP_BISECTION_STEPS`] times, i.e. to `2⁻⁵³`
/// relative — about one f64 ulp — and the LOWER end is returned, so the answer is affordable by
/// construction rather than by rounding luck.
static DERIVED_MASS_CAP: std::sync::LazyLock<MassCap> =
    std::sync::LazyLock::new(|| solve_mass_cap(REAL_GALAXY_R_M, SYSTEM_LATTICE_R_M));

/// How wide a star system may be before it overruns the lattice it counts its OWN positions on — the
/// second of the two limits on a star's mass, and the one that binds after the S9 climb.
///
/// It is the FINE rung's own fence solved at equality, exactly as the galaxy's radius is the galaxy
/// rung's. Named here rather than spelled at each use so the two readers cannot drift apart.
pub(crate) const SYSTEM_LATTICE_R_M: f64 = root_radius_at(vd_core::pose::Tier::Fine);

/// The three numbers the one solve produces (solved once, read everywhere — two spellings of a
/// derivation is how they would drift).
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct MassCap {
    /// The greatest stellar mass this galaxy can host (solar masses) — the IMF draw's upper bound.
    pub(crate) mass_hi_msun: f64,
    /// That star's system shell: THE reservation.
    pub(crate) system_bound_max_m: f64,
    /// That star's photosphere: the reservation's `look` half.
    pub(crate) star_look_max_m: f64,
}

/// How many times the bracket is halved: `f64::MANTISSA_DIGITS`. The doubling bracket ends at
/// relative width 1 (`hi == 2·lo`), so 53 halvings close it to `2⁻⁵³` of the cap — f64's own
/// resolution. Not a tolerance anyone chose; the type's.
const MASS_CAP_BISECTION_STEPS: u32 = f64::MANTISSA_DIGITS;

/// How many doublings the bracket may take before the solve gives up. `demand` grows without bound
/// in mass, so the break always fires long before this; the bound exists so a future law change
/// that broke monotonicity would end the loop rather than spin it. `f64::MAX_EXP` is the number of
/// doublings the type itself admits — again the type's number, not a chosen one.
const MASS_CAP_BRACKET_DOUBLINGS: u32 = f64::MAX_EXP as u32;

/// The star a mass draws — the SAME three-field chain [`generate_system_forest`] runs
/// (`sample_imf_mass` → `classify_spectral` → `main_sequence_luminosity`), with the draw already
/// resolved to a mass. One spelling, two callers (the generator and the cap solve).
pub(crate) fn star_at_mass(mass_msun: f64) -> StarPhotometrics {
    StarPhotometrics {
        mass_msun,
        class: classify_spectral(mass_msun, &SpectralClass::MASS_BOUNDS),
        luma_lsun: main_sequence_luminosity(mass_msun, &SpectralClass::MLR_SEGMENTS),
    }
}

/// What the galaxy must spend on ONE child of this mass — the inequality [`DERIVED_MASS_CAP`]
/// solves. Monomorphic, straight-line.
pub(crate) fn galaxy_child_demand_m(pl: &PlanetConfig, mass_msun: f64) -> f64 {
    let star = star_at_mass(mass_msun);
    let shell_m = system_shell_r_m(pl, &star);
    let look_m = crate::taxonomy::star_radius_m(mass_msun);
    child_clearance_m(shell_m, look_m, VISIBILITY_THETA_MIN_RAD) + 2.0 * shell_m
}

/// ★ CAN A STAR OF THIS MASS EXIST? — BOTH CONSTRAINTS, NOT ONE (slice S9).
///
/// The galaxy must be able to pay for the system around it, AND that system must fit its own coordinate
/// lattice. Until S9 only the first could ever bind, so only the first was asked: the galaxy was small,
/// and it ran out of room long before a star system ran out of numbers.
///
/// The climb reverses that. MEASURED at the new radius: the galaxy affords **1288.27** solar masses while
/// a system's own millimetre lattice holds **30.75**. Asking only the galaxy would now draw stars whose
/// systems cannot state their own positions — and nothing downstream would notice, because a shell that
/// overruns its lattice does not fail loudly, it wraps.
///
/// Bitwise `&`, not `&&`: both terms are cheap and pure, and a short-circuit would leave the second
/// uncoverable from a false first.
fn affordable_at(pl: &PlanetConfig, mass_msun: f64, budget_m: f64, lattice_m: f64) -> bool {
    binding_limit(pl, mass_msun, budget_m, lattice_m).is_none()
}

/// WHICH of the two constraints a star of this mass breaks, if either — the ONE affordability
/// definition, shared by the solve and the fence.
///
/// ★ WHY THIS EXISTS (slice S9). The solve and the fence used to disagree, and the climb is what
/// exposed it: [`affordable_at`] learned the lattice constraint, [`guard_galaxy_affords_its_stars`]
/// did not, so after the climb the fence would have PASSED a 120 M☉ star that the solve refuses at
/// 30.75. A fence that guards a door the world no longer uses is worse than no fence, because its
/// silence reads as approval. Both now read this.
pub(crate) fn binding_limit(
    pl: &PlanetConfig,
    mass_msun: f64,
    budget_m: f64,
    lattice_m: f64,
) -> Option<StarLimit> {
    let shell_m = system_shell_r_m(pl, &star_at_mass(mass_msun));
    let purse_ok = galaxy_child_demand_m(pl, mass_msun) <= budget_m;
    let lattice_ok = shell_m <= lattice_m;
    // Straight-line: the purse is reported first when both break, because a galaxy too small to
    // place the system is the coarser fact and the one an operator can act on.
    match (purse_ok, lattice_ok) {
        (true, true) => None,
        (false, _) => Some(StarLimit::GalaxyPurse),
        (true, false) => Some(StarLimit::SystemLattice),
    }
}

/// Which of the two limits on a star's mass actually bound.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum StarLimit {
    /// The galaxy has not the room to place the system this star needs around it.
    GalaxyPurse,
    /// The system would be too wide to state its own positions on the fine lattice.
    SystemLattice,
}

impl core::fmt::Display for StarLimit {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.write_str(match *self {
            StarLimit::GalaxyPurse => "the galaxy has no room to place its system",
            StarLimit::SystemLattice => "its system could not state its own positions",
        })
    }
}

/// The monotone solve itself (see [`DERIVED_MASS_CAP`] for the derivation and the why).
///
/// ★ THE BUDGET IS AN ARGUMENT, NOT A MODULE READ (slice S7). It used to read [`REAL_GALAXY_R_M`]
/// directly, which meant the one question this solve exists to answer — *what would the cap be if the
/// galaxy were a different size?* — could not be asked of it at all. That question is the whole of the
/// coordinate-step decision: the cap is a fact about a galaxy of THIS radius, and the radius is the
/// lattice's storage budget, so a change of step moves the budget and every star draw with it.
///
/// Turning a constant into a parameter must move NOTHING, and that is gated rather than asserted:
/// `the_cap_solved_at_todays_budget_is_bit_for_bit_the_shipped_one` re-solves at the shipped budget and
/// compares all three fields exactly.
pub(crate) fn solve_mass_cap(budget_m: f64, lattice_m: f64) -> MassCap {
    let pl = world_planet_config();
    // BRACKET: double from the hydrogen-burning limit until the galaxy cannot pay. `lo` therefore
    // always names a mass the galaxy CAN pay for (the limit itself costs 4.07e11 m against a
    // 2.25e15 m budget — six thousandths of a percent), and `hi` one it cannot.
    let mut lo = IMF_MASS_LO_MSUN;
    let mut hi = lo;
    for _ in 0..MASS_CAP_BRACKET_DOUBLINGS {
        if !affordable_at(&pl, hi, budget_m, lattice_m) {
            break;
        }
        lo = hi;
        hi *= 2.0;
    }
    // BISECT: keep the affordable half. Both assignments are total (no early exit), so the loop
    // runs a fixed, stated number of steps and the answer is a pure function of the laws above.
    for _ in 0..MASS_CAP_BISECTION_STEPS {
        let mid = 0.5 * (lo + hi);
        let affordable = affordable_at(&pl, mid, budget_m, lattice_m);
        lo = if affordable { mid } else { lo };
        hi = if affordable { hi } else { mid };
    }
    MassCap {
        mass_hi_msun: lo,
        system_bound_max_m: system_shell_r_m(&pl, &star_at_mass(lo)),
        star_look_max_m: crate::taxonomy::star_radius_m(lo),
    }
}

/// A galaxy that cannot pay for the stars physics says exist in it — the affordability verdict, with
/// every number of it, so the refusal names itself.
#[derive(Clone, Copy, Debug, PartialEq, thiserror::Error)]
#[error(
    "this world affords a {affordable_msun} solar-mass star, but physics states stars up to \
     {stated_top_msun}: at that mass {bound_by} — the system spans {shell_m} m and needs \
     {demand_m} m of room, against a galaxy radius of {budget_m} m and a system lattice of \
     {lattice_m} m"
)]
pub struct StarsUnaffordable {
    /// What physics says the heaviest star is.
    pub stated_top_msun: f64,
    /// What this world can actually pay for.
    pub affordable_msun: f64,
    /// WHICH of the two limits bound — see [`StarLimit`].
    pub bound_by: StarLimit,
    /// What the stated top would cost the galaxy.
    pub demand_m: f64,
    /// What there is to spend.
    pub budget_m: f64,
    /// How wide the stated top's own system would be.
    pub shell_m: f64,
    /// How wide a system may be before it overruns the fine lattice.
    pub lattice_m: f64,
}

/// ★ THE AFFORDABILITY QUESTION AS A FENCE RATHER THAN A SOLVE (slice S7).
///
/// The shipped arrangement solves for the heaviest star a galaxy of this radius can pay for, and hands
/// that answer to the star draw. So the universe's biggest star is a consequence of a COORDINATE
/// choice — change the step and every star in every system is re-drawn. The last time that happened,
/// 99 of 99 planet rows moved.
///
/// This asks the question the other way round, which is the way the owner's ruling states it: physics
/// says how big stars get, and the galaxy has to be big enough for them. Then a coordinate step is
/// judged rather than obeyed.
///
/// ★ IT USED TO REFUSE THE WORLD AS IT STOOD, AND THE CLIMB IS THE CURE IT ASKED FOR. Before S9 the
/// galaxy afforded 16.36 solar masses against a physical top of 120 — refused by a factor of about
/// seven — and that refusal WAS the argument for the coordinate step. The step changed. The galaxy is
/// now 2_051× wider and its purse affords 1288.27 solar masses, so the arm that refused has stopped
/// binding.
///
/// ★ AND THE LIMIT MOVED RATHER THAN VANISHED, which is why this still refuses. A star system counts
/// its own positions in millimetres and its shell must fit its own lattice (2⁵¹ m). At 120 M☉ the
/// system spans about 8.79e15 m against that 2.25e15 m, so the world is refused by roughly 3.9× —
/// **by the SYSTEM's numbers now, not by the galaxy's room.** [`StarLimit`] carries which.
///
/// So the fence stays unarmed at boot for the same reason as before (arming it would refuse the world
/// we ship) but the cure it names is different: the fine rung, not the galaxy's radius. That is the
/// P10 lift. It is driven from both sides by tests, and
/// [`what_the_heaviest_star_becomes_at_each_candidate_coordinate_step`] measures where it flips.
///
/// # Errors
/// [`StarsUnaffordable`] when a galaxy of `budget_m` cannot pay for a star of the stated physical top.
pub fn guard_galaxy_affords_its_stars(
    budget_m: f64,
    lattice_m: f64,
) -> Result<(), StarsUnaffordable> {
    let pl = world_planet_config();
    match binding_limit(&pl, IMF_MASS_HI_PHYSICAL_MSUN, budget_m, lattice_m) {
        None => Ok(()),
        Some(bound_by) => Err(StarsUnaffordable {
            stated_top_msun: IMF_MASS_HI_PHYSICAL_MSUN,
            affordable_msun: solve_mass_cap(budget_m, lattice_m).mass_hi_msun,
            bound_by,
            demand_m: galaxy_child_demand_m(&pl, IMF_MASS_HI_PHYSICAL_MSUN),
            budget_m,
            shell_m: system_shell_r_m(&pl, &star_at_mass(IMF_MASS_HI_PHYSICAL_MSUN)),
            lattice_m,
        }),
    }
}

/// THE IMF DRAW'S UPPER BOUND — the derived cap, read by every star draw
/// ([`StellarConfig::mass_hi_msun`]). See [`DERIVED_MASS_CAP`] for the derivation, why the cap
/// exists, and when it lifts.
#[must_use]
pub fn imf_mass_hi_msun() -> f64 {
    DERIVED_MASS_CAP.mass_hi_msun
}

/// ★ A SEED-0 PROVENANCE MARKER, AND ONLY THAT (restated 2026-08-21, the gate-pass arc).
///
/// The HOME system's target shell of §3.3.5 row 1 — **seed 0's `System(7)`**, not the shipped
/// default world's. Since 2026-08-20 a process with no `VD_UNIVERSE_SEED` boots
/// [`HOME_SEED`](crate::worldgen::HOME_SEED) (2298), whose home system solves to a shell some
/// THIRTY-TWO TIMES this one — its star is a G star where seed 0's is an M dwarf, and the orbit
/// ladder is anchored at `0.4·√L` AU. So this number describes ONE seed's world and is useful for
/// exactly one thing: being a fixed ruler that a generator change moves.
///
/// That is its job, and the job is checked. `g_star_shell_unmoved_the_stars_clearance_arm_never_binds`
/// asserts it EQUALS seed 0's generated home shell, so the constant cannot quietly drift away from
/// the solve it cites — it flips loudly instead. RE-MEASURED twice now, both times for the SAME
/// reason — the mass cap enters every star draw, so this seed's home star moves whenever the cap
/// does: 1.582261852875e11 → 1.582054016685e11 (2026-08-20, the galaxy-derived cap) →
/// 1.582181841711e11 m = 1.05723 AU (S9, the lattice-derived cap).
///
/// ★ IT IS NO LONGER A FLIGHT-TABLE LEG. It used to be the `flight_table` gate's system-leg
/// distance and warp-departure ceiling; that gate now DERIVES both from the home realm of the world
/// it boots, because a table whose two in-system legs came from seed 0 while its warp leg came from
/// the live world described no single world (SL5). The gate still PRINTS this target beside the
/// derived distance, which is what keeps the citation live. `pub` for that print and for the story
/// fixture.
pub const TARGET_SYSTEM_BOUND_HOME_M: f64 = 158_218_184_171.079_4;
/// The same seed-0 provenance marker, one level down: seed 0's home system's OUTER planet's target
/// SOI at the maximum mass draw (§3.3.4/§3.3.5). RE-MEASURED 2026-08-20 with the same cause
/// (8.567390468e9 → 8.566236992e9 m).
///
/// ★ UNLIKE ITS SIBLING ABOVE, NOTHING PINS THIS ONE. It is a hand-derived figure off the
/// addendum's §3.3 chain — a planet at the MAXIMUM mass draw on the outer rung — not a value the
/// shipped generator produces for any body, so there is no generated quantity to assert it equal
/// to. It is therefore documentation with a number attached, and it is `pub` only so the
/// `flight_table` gate can PRINT it beside the world's own outer-planet SOI. The D-REAL-1 equality
/// (realm shell == gravitational SOI) is what would finally give it a live pin; until then, read it
/// as a citation of the design, never as a fact about the world any player boots.
pub const TARGET_PLANET_SOI_OUTER_HOME_M: f64 = 8_566_236_992.362_801;

/// ▲ 4. THE COMPRESSION χ = 24.568× — stated as the measurement it is (§A2.3): the real mean
/// nearest-neighbour stellar separation over the placement radius. Real separation
/// `0.55396 · n^(−1/3)` at `n = 0.1 pc⁻³` (RECONS 10-parsec census) `= 3.682666e16 m = 3.8926 ly`;
/// `χ = 3.682666e16 / 1.4989796e15 = 24.568`. ★ IT ROSE FROM 16.378 on 2026-08-20: the DERIVED
/// mass cap makes the galaxy reserve room for the largest child it can actually host, and that
/// reservation comes out of the star gap. THE TRADE, stated: a third of the gap buys a world that
/// nests for EVERY seed instead of only for the one the old reservation was sampled from — see
/// [`DERIVED_MASS_CAP`] and `DEFERRED.md` D-MASS-CAP. The galaxy cell lattice (P10) exists to drive
/// χ toward 1 and lifts the cap at the same time; [`guard_root_representable`]'s refusal is its
/// named trigger.
const RECONS_STELLAR_DENSITY_PER_PC3: f64 = 0.1;
/// Mean nearest-neighbour coefficient for a Poisson point field (`0.55396·n^(−1/3)`).
const MEAN_NN_COEFF: f64 = 0.55396;
/// One parsec in metres (IAU): the census density's unit.
const PARSEC_M: f64 = 3.085_677_581_491_367e16;

/// THE NUMBERS THAT SHAPE THE FOREST — the list a durable file folds into its world-law label
/// ([`vd_core::store_stamp::world_generation`]), so a store written for one world's geometry is refused
/// by a process that would place those bodies somewhere else.
///
/// It is a LIST rather than a struct on purpose: the fold is over values and their order, and adding a
/// number here is exactly the act that should invalidate every existing store. Anything whose change
/// would move a body belongs in it; anything else does not, because a label that moves for a cosmetic
/// reason trains people to delete their data.
///
/// ⚠ NOT YET COMPLETE, and stated so rather than implied: the band law's own constants are owed here
/// when slice S2 gives each boundary its band. Adding them then is a deliberate act that moves every
/// label, which is correct.
#[must_use]
pub fn world_shape_constants() -> Vec<f64> {
    vec![
        REAL_UNIVERSE_R_M,
        REAL_GALAXY_R_M,
        T_TRAVERSE_S,
        GEOMETRY_TICK_DT_S,
        BAND_TICKS_N,
        BAND_TAU_HEADROOM,
        VISIBILITY_THETA_MIN_RAD,
    ]
}

/// The §3.2 clearance a parent owes ONE child: enough to CONTAIN it, plus enough that its picture
/// has already fallen below the minimum angle — the `+ vis` form, whose strictness proof costs
/// 3.3 % of what the rejected doubling form cost (§3.2's sidebar).
pub(crate) fn child_clearance_m(child_bound_m: f64, child_look_m: f64, theta_min_rad: f64) -> f64 {
    let vis = child_look_m * (1.0 + visibility_factor(theta_min_rad));
    child_bound_m.max(vis) + vis
}

/// The placement radius (the star gap) — see the ▲ 3 derivation above.
pub(crate) fn real_placement_r_m() -> f64 {
    REAL_GALAXY_R_M
        - child_clearance_m(
            target_system_bound_max_m(),
            target_star_look_max_m(),
            VISIBILITY_THETA_MIN_RAD,
        )
}

/// ▲ 4 as a number: the between-systems compression χ — the real mean nearest-neighbour stellar
/// separation over the placement radius (16.378× on THE world; 1.000000 in-system, exactly,
/// because no in-system compression factor exists to be anything else). Public so the pin and any
/// report read the ONE derivation.
#[must_use]
pub fn real_compression_chi() -> f64 {
    let real_separation_m =
        MEAN_NN_COEFF * RECONS_STELLAR_DENSITY_PER_PC3.powf(-1.0 / 3.0) * PARSEC_M;
    real_separation_m / real_placement_r_m()
}
