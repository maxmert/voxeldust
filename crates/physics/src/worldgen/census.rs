//! THE SWEEP: reading THE world to find where a person would want to live.
//!
//! Owns: the Earth-like predicate under the owner's stated criteria (a yellow sun, a rocky verdict, a
//! temperate band), the per-seed sweep, and the census numbers a report prints — planet and moon
//! counts, nearest and farthest sibling star.
//!
//! Does NOT own: a search-only world. Every number here is derived from THE one generator read at a
//! seed (SL5); the sweep chooses a SEED, never a different universe.

use super::{GeneratedBody, Placement, StarPhotometrics, UniverseConfig, generate_system_forest};
use crate::taxonomy::SpectralClass;
use vd_core::pose::RealmId;

/// One Earth-like candidate of a swept seed — everything the report prints, derived entirely
/// from `BodyTaxon` + `StarPhotometrics`, both of which the ONE generator emits (SL5: the tool
/// never builds a world; it READS THE world at a candidate seed and scores it).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct EarthLikeCandidate {
    pub system: RealmId,
    pub body: RealmId,
    pub star_mass_msun: f64,
    pub mass_kg: f64,
    pub radius_m: f64,
    pub insolation_rel: f64,
    pub t_eq_k: f64,
    // ---- T4b: the rest of the picture an owner needs in order to CHOOSE a home. Every field is
    // read off the same two rows the predicate reads (`StarPhotometrics`, `BodyTaxon`) or counted
    // off the forest already in hand — the tool still builds no world of its own (SL5).
    /// The star's Morgan-Keenan class (the predicate admits G only; carried so the report states
    /// what it measured rather than what it assumed).
    pub star_class: SpectralClass,
    /// The star's main-sequence luminosity, `L/L☉` — how bright the sky over this world is.
    pub star_luma_lsun: f64,
    /// The planet's derived composition class.
    pub planet_class: crate::taxonomy::PlanetType,
    /// Its Bond albedo (the class+atmosphere table's value — the input its `t_eq_k` came from).
    pub bond_albedo: f64,
    /// Whether the cosmic-shoreline + Jeans verdicts left it AIR.
    pub has_atmosphere: bool,
    /// How many PLANETS the candidate's system holds (its star is not counted).
    pub system_planets: u32,
    /// How many MOONS that system holds in total.
    pub system_moons: u32,
    /// How many moons THIS body holds.
    pub own_moons: u32,
    /// How many OTHER star systems this galaxy holds (the reachable neighbours).
    pub sibling_count: u32,
    /// The 3-D distance to the NEAREST sibling system (m) — the first warp's length.
    pub nearest_sibling_m: f64,
    /// The 3-D distance to the FARTHEST sibling system (m) — the galaxy's far corner from here.
    pub farthest_sibling_m: f64,
}

/// The owner's Earth-radius band (Earth radii) — the search's stated size criterion.
/// ★ THE HOME SEED — the universe every player starts in (owner ruling, 2026-08-20).
///
/// Chosen by the owner from the `vd-seedsearch` candidate table (`scratchpad/home_candidates.md`,
/// 42 candidates over 8 029 swept seeds — a measured rate of 1 in 191.2), NOT authored: the search
/// reads the generator, the generator is never biased toward the search (SL5). Seed 2298 ranked
/// first on the published desirability expression: a G-class star, an Earth-like world of 1.087 M⊕
/// and 1.023 R⊕ (6 515.5 km, ρ 5 601 kg/m³, g 10.20 m/s²) that is Rocky, temperate and RETAINS ITS
/// ATMOSPHERE, in a system of 9 planets, with two sibling stars to warp to.
///
/// ★ THE NUMBERS THIS PARAGRAPH USED TO NAME MOVED THE SAME DAY THE SEED LANDED, and are no longer
/// written down here. The DERIVED MASS CAP re-rolled the stellar draw hours after the choice: the
/// star went 1.0313 → 1.0151 M☉ and 1.1311 → 1.0616 L☉, the system's moon census 22 → 19, and the
/// sibling stars moved from 0.2377 ly to 0.1584 ly with the placement radius. The PLANET itself is
/// bit-unmoved — its own draws never read the stellar cap — which is why the choice still stands.
/// `earth_like_candidates_at_the_home_seed…` pins every one of those values as a MEASUREMENT, and
/// that is where to read them; the flight table measures what the warp actually costs. A prose
/// restatement here would only go stale again at the next re-solve.
///
/// ★ DISCOVERY PERMANENCE (owner's standing law): this number is a PRE-LAUNCH dial. At launch it
/// FREEZES FOREVER — a seed change is a different world, so once discovery begins it never moves.
/// The generator's draw stream is append-only for the same reason (see this module's header).
///
/// It is the DEFAULT every world-deriving process reads (`VD_UNIVERSE_SEED`); tests that pass an
/// explicit seed are unaffected by it, which is why the pinned f(seed) suites still pin seed 0.
pub const HOME_SEED: u64 = 2298;

pub const EARTH_LIKE_RADIUS_BAND_REARTH: (f64, f64) = (0.8, 1.25);

/// THE EARTH-LIKE PREDICATE (§8.1, under the owner's rulings): a YELLOW SUN (G class), a
/// ROCKY verdict (now a derived composition + envelope conclusion), the owner's radius band,
/// the Kopparapu CONSERVATIVE flux band (ruling B), the temperate `T_eq` band derived from
/// those same flux limits at the Rocky-with-atmosphere Bond albedo (no second literal), and
/// AIR — the shoreline verdict survived.
///
/// TWO OF THE SIX CLAUSES ARE STRUCTURAL NO-OPS ON ANY WORLD (§8.2, stated so the tool cannot
/// report false precision): insolation is QUANTISED AND SEED-FREE (rung 2 = 0.748 S⊕ for
/// every star at every seed — the ladder law), and a rocky rung-2 `T_eq` is one of two
/// class-albedo values, both inside the band. The search's real discriminants are the star's
/// class and the planet's drawn mass. `earth_like_no_op_clauses…` pins this as a measurement.
#[must_use]
pub fn earth_like(star: &StarPhotometrics, taxon: &crate::taxonomy::BodyTaxon) -> bool {
    use crate::taxonomy::{KOPPARAPU_FLUX_CONSERVATIVE, PlanetType, R_EARTH_M, SpectralClass};
    let (s_lo, s_hi) = KOPPARAPU_FLUX_CONSERVATIVE;
    let (r_lo, r_hi) = EARTH_LIKE_RADIUS_BAND_REARTH;
    let yellow_sun = star.class == SpectralClass::G;
    let rocky = taxon.class == PlanetType::Rocky;
    let earth_sized = (r_lo..=r_hi).contains(&(taxon.radius_m / R_EARTH_M));
    let temperate_flux = (s_lo..=s_hi).contains(&taxon.insolation_rel);
    // T grows with flux: the band's LOW temperature is the OUTER flux limit's.
    let temperate_k =
        (earth_like_t_bound_k(s_lo)..=earth_like_t_bound_k(s_hi)).contains(&taxon.t_eq_k);
    let air = taxon.atmosphere.is_some();
    yellow_sun & rocky & earth_sized & temperate_flux & temperate_k & air
}

/// The temperate band's temperature at flux `s` — the SAME Kopparapu limits converted once
/// through the one equilibrium law at the Rocky-with-atmosphere Bond albedo (the class
/// table's own value; no second literal): `T(1.10) ≈ 260.2 K`, `T(0.53) ≈ 216.7 K`.
#[must_use]
pub fn earth_like_t_bound_k(s_rel: f64) -> f64 {
    use crate::taxonomy::{AU_M, L_SUN_W, PlanetType, bond_albedo_of, equilibrium_temperature_k};
    equilibrium_temperature_k(
        L_SUN_W * s_rel,
        AU_M,
        bond_albedo_of(PlanetType::Rocky, true),
    )
}

/// SWEEP one seed: every Earth-like body of THE world at that seed (SL5: the one generator,
/// read — never a variant). The tool's whole read path, shared by the bin and the tests.
#[must_use]
pub fn earth_like_candidates(
    seed_universe: u64,
    config: &UniverseConfig,
) -> Vec<EarthLikeCandidate> {
    earth_like_in_forest(&generate_system_forest(seed_universe, config))
}

/// The sweep's ARITHMETIC over a forest already in hand — split from the generate-and-score shell
/// so its two defensive arms (a body whose parent is not in the forest; a chain that names no
/// photometric star) are reachable from a unit test. THE world never produces either: every
/// generated body's parent is present and every system carries its star.
pub(crate) fn earth_like_in_forest(bodies: &[GeneratedBody]) -> Vec<EarthLikeCandidate> {
    let mut out = Vec::new();
    for b in bodies.iter().filter(|b| b.taxon.is_some()) {
        let taxon = b.taxon.expect("filtered on presence");
        let Some(parent) = b.parent else { continue };
        // The illuminating STAR: the body's parent system (a planet) or grandparent (a moon).
        let system = match parent {
            RealmId::System(_) => parent,
            _ => match bodies
                .iter()
                .find(|p| p.realm == parent)
                .and_then(|p| p.parent)
            {
                Some(gp) => gp,
                None => continue,
            },
        };
        let Some(star) = bodies
            .iter()
            .find(|p| p.realm == system)
            .and_then(|p| p.photometrics)
        else {
            continue;
        };
        if earth_like(&star, &taxon) {
            let (nearest_sibling_m, farthest_sibling_m, sibling_count) =
                sibling_star_gaps(bodies, system);
            let (system_planets, system_moons) = system_census(bodies, system);
            out.push(EarthLikeCandidate {
                system,
                body: b.realm,
                star_mass_msun: star.mass_msun,
                mass_kg: taxon.mass_kg,
                radius_m: taxon.radius_m,
                insolation_rel: taxon.insolation_rel,
                t_eq_k: taxon.t_eq_k,
                star_class: star.class,
                star_luma_lsun: star.luma_lsun,
                planet_class: taxon.class,
                bond_albedo: taxon.bond_albedo,
                has_atmosphere: taxon.atmosphere.is_some(),
                system_planets,
                system_moons,
                own_moons: children_of(bodies, b.realm),
                sibling_count,
                nearest_sibling_m,
                farthest_sibling_m,
            });
        }
    }
    out
}

/// How many bodies name `parent` as their parent — the ONE counting expression the census reads
/// (a monomorphic helper: the report needs it for a system's planets and for a planet's moons).
fn children_of(bodies: &[GeneratedBody], parent: RealmId) -> u32 {
    let n = bodies.iter().filter(|b| b.parent == Some(parent)).count();
    // Branchless (HR5): the clamp is a `min`, never a fallible conversion with an unreachable arm.
    n.min(u32::MAX as usize) as u32
}

/// `(planets, moons)` of one system: its direct children that are NOT its star, and their own
/// children. Counted off the forest already generated — no second world, no second walk law.
fn system_census(bodies: &[GeneratedBody], system: RealmId) -> (u32, u32) {
    let mut planets = 0u32;
    let mut moons = 0u32;
    for b in bodies
        .iter()
        .filter(|b| b.parent == Some(system) && !matches!(b.realm, RealmId::Star(_)))
    {
        planets = planets.saturating_add(1);
        moons = moons.saturating_add(children_of(bodies, b.realm));
    }
    (planets, moons)
}

/// `(nearest, farthest, count)` 3-D distances from `system` to the galaxy's OTHER star systems —
/// what the owner's first warp will actually feel like. Both systems' placements are static
/// offsets in the one galaxy frame the generator authored them in (Q-B's seeded 3-D placements),
/// so this is a plain subtraction, never a fold from the root. An only child reports zeros with a
/// zero count — the honest answer, not a sentinel.
fn sibling_star_gaps(bodies: &[GeneratedBody], system: RealmId) -> (f64, f64, u32) {
    let offset_of = |realm: RealmId| {
        bodies
            .iter()
            .find(|b| b.realm == realm)
            .and_then(|b| match b.placement {
                Placement::StaticOffset(at) => Some(at),
                Placement::Orbital(_) => None,
            })
    };
    let Some(here) = offset_of(system) else {
        return (0.0, 0.0, 0);
    };
    // SL1: a placement is stated in the PARENT's frame, so only bodies sharing this system's
    // parent are comparable at all. Two id-shaped filters were MEASURED wrong before this one: the
    // ambient Universe/Galaxy shells wear the same `RealmId::System` spelling and sit at the
    // origin, and every system's own STAR sits at ZERO in ITS OWN frame — both reported a nearest
    // gap of 0 m on a home system that (by J1) sits at the galaxy origin itself.
    let Some(parent) = bodies
        .iter()
        .find(|b| b.realm == system)
        .and_then(|b| b.parent)
    else {
        return (0.0, 0.0, 0);
    };
    let mut nearest = f64::INFINITY;
    let mut farthest = 0.0_f64;
    let mut count = 0u32;
    for other in bodies
        .iter()
        .filter(|b| b.parent == Some(parent) && b.photometrics.is_some() && b.realm != system)
    {
        let Some(there) = offset_of(other.realm) else {
            continue;
        };
        let gap = (there - here).length();
        nearest = nearest.min(gap);
        farthest = farthest.max(gap);
        count = count.saturating_add(1);
    }
    if !nearest.is_finite() {
        nearest = 0.0;
    }
    (nearest, farthest, count)
}
