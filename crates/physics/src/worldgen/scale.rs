//! THE REAL-SCALE DERIVATION: four numbers, each solved rather than chosen.
//!
//! Owns: the universe radius (solved from what the position store can represent), the galaxy radius,
//! the placement radius that is the gap between stars, the between-systems compression that gap
//! implies, and the derived mass cap — the greatest star this galaxy can host, found by a monotone
//! solve rather than declared.
//!
//! Does NOT own: any of them twice. Two spellings of one derivation is how a world quietly stops
//! agreeing with itself, so each number has exactly one home here and every consumer reads it.

use super::{
    IMF_MASS_LO_MSUN, PlanetConfig, StarPhotometrics, system_shell_r_m, world_planet_config,
};
use crate::taxonomy::{SpectralClass, classify_spectral, main_sequence_luminosity};
/// THE ONE visibility formula, re-exported from its home (look_horizon.md §3.3.2 — the formula
/// lives in `vd-core` with three consumers: the world solve, the boot measurement, the runtime
/// tripwire; this module is two of them and BUILDS the third's config). Never a second copy.
use vd_core::geometry::visibility_factor;

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
pub(crate) const VISIBILITY_THETA_MIN_RAD: f64 = 0.026_180;

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
pub(crate) const REAL_UNIVERSE_R_M: f64 = 2_251_799_813_685_248.0; // 2^51, exact

/// The storage-fence headroom octave (§A2.2, `K_SPAN = 2`) — one binary octave between the root
/// shell and the sanitize clamp, so the clamp is unreachable from any lawful position (OQ-3's
/// recommendation, adopted: a silent clamp is the defect class this coordinate exists to prevent).
pub(crate) const K_SPAN: f64 = 2.0;

/// ▲ 2. THE GALAXY RADIUS: `R_uni − outset` — THE CORRECTED FORMULA (§A2.2, curing H-27: the main
/// design's `R_gal = R_uni/2` was a residue of the REJECTED doubling clearance form and cost a full
/// octave of star gap). The ambient realms carry no look, so their clearance degenerates to their
/// own bound and the shells would TOUCH; the strictness the nesting fence needs is the child's own
/// release band — but a band's governed arm contains τ = T_WAKE, a MEASURED boot latency, and a
/// world radius may never be a function of how fast a shard happens to boot (SL5 / determinism /
/// no-magic-numbers — H-02). The cure is the τ-FREE UPPER BOUND the band-solvability fence
/// supplies: `dt·N ≤ τ/2` ⇒ the governed band is at most `2 · v_cap · dt · N`, so
///
/// `outset = v_cap(R_uni) · GEOMETRY_TICK_DT_S · BAND_TICKS_N · BAND_TAU_HEADROOM`
///         `= (2·R_uni/T_TRAVERSE_S) · 0.02 · 3 · 2 = 3.0023997515803307e12 m`
/// `R_gal  = R_uni − outset = 2.2487974139336678e15 m ≈ 0.237698 ly`
///
/// Cost of the strictness margin: 0.133 % of `R_uni`, against the 50 % an octave would cost.
pub(crate) const REAL_GALAXY_R_M: f64 = REAL_UNIVERSE_R_M
    - (2.0 * REAL_UNIVERSE_R_M / T_TRAVERSE_S)
        * GEOMETRY_TICK_DT_S
        * BAND_TICKS_N
        * BAND_TAU_HEADROOM;

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
static DERIVED_MASS_CAP: std::sync::LazyLock<MassCap> = std::sync::LazyLock::new(solve_mass_cap);

/// The three numbers the one solve produces (solved once, read everywhere — two spellings of a
/// derivation is how they would drift).
#[derive(Clone, Copy, Debug, PartialEq)]
struct MassCap {
    /// The greatest stellar mass this galaxy can host (solar masses) — the IMF draw's upper bound.
    mass_hi_msun: f64,
    /// That star's system shell: THE reservation.
    system_bound_max_m: f64,
    /// That star's photosphere: the reservation's `look` half.
    star_look_max_m: f64,
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

/// The monotone solve itself (see [`DERIVED_MASS_CAP`] for the derivation and the why).
fn solve_mass_cap() -> MassCap {
    let pl = world_planet_config();
    let budget_m = REAL_GALAXY_R_M;
    // BRACKET: double from the hydrogen-burning limit until the galaxy cannot pay. `lo` therefore
    // always names a mass the galaxy CAN pay for (the limit itself costs 4.07e11 m against a
    // 2.25e15 m budget — six thousandths of a percent), and `hi` one it cannot.
    let mut lo = IMF_MASS_LO_MSUN;
    let mut hi = lo;
    for _ in 0..MASS_CAP_BRACKET_DOUBLINGS {
        if galaxy_child_demand_m(&pl, hi) > budget_m {
            break;
        }
        lo = hi;
        hi *= 2.0;
    }
    // BISECT: keep the affordable half. Both assignments are total (no early exit), so the loop
    // runs a fixed, stated number of steps and the answer is a pure function of the laws above.
    for _ in 0..MASS_CAP_BISECTION_STEPS {
        let mid = 0.5 * (lo + hi);
        let affordable = galaxy_child_demand_m(&pl, mid) <= budget_m;
        lo = if affordable { mid } else { lo };
        hi = if affordable { hi } else { mid };
    }
    MassCap {
        mass_hi_msun: lo,
        system_bound_max_m: system_shell_r_m(&pl, &star_at_mass(lo)),
        star_look_max_m: crate::taxonomy::star_radius_m(lo),
    }
}

/// THE IMF DRAW'S UPPER BOUND — the derived cap, read by every star draw
/// ([`StellarConfig::mass_hi_msun`]). See [`DERIVED_MASS_CAP`] for the derivation, why the cap
/// exists, and when it lifts.
#[must_use]
pub fn imf_mass_hi_msun() -> f64 {
    DERIVED_MASS_CAP.mass_hi_msun
}

/// The HOME system's target shell (§3.3.5 row 1, seed 0's `System(7)`) — RE-MEASURED 2026-08-20 at
/// the derived mass cap (1.582261852875e11 → 1.582054016685e11 m = 1.05714 AU; the cap enters
/// every star draw, so this seed's home star moved with it) —
/// the flight-table gate's system-leg distance (`2·R_sys` edge-to-edge) and its warp-departure
/// ceiling input. The SAME cited-target discipline as [`target_system_bound_max_m`]: the taxonomy
/// slice's in-system re-solve recomputes it, and the gate that reads it flips loudly if that slice
/// lands a different number. `pub` for exactly that gate.
pub const TARGET_SYSTEM_BOUND_HOME_M: f64 = 158_205_401_668.478_1;
/// The home system's OUTER planet's target SOI at the maximum mass draw (§3.3.4/§3.3.5) —
/// the flight-table gate's planet-leg distance ("planet surface out to its own shell").
/// RE-MEASURED 2026-08-20 with the same cause (8.567390468e9 → 8.566236992e9 m). Cited-target
/// discipline as above; the D-REAL-1 equality (realm shell == gravitational SOI) lands it for real
/// with the taxonomy slice.
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
