//! THE GENERATION: the draws that turn one seed into a star system.
//!
//! Owns: the per-kind seed salts, the FROZEN draw order (change it and every world changes), the
//! star's mass and photometrics, the planet ladder with its masses and orbital elements, the moon
//! pass, and the shell solve that sizes a system around the worst draw it could have made.
//!
//! Does NOT own: a second stream. The draw order IS the world's identity — a new quantity is
//! appended, never inserted — which is why the stream prefix width is a named constant rather than a
//! consequence of how the loop happens to be written.

use super::{
    ECC_CAP_SIGMAS, ECC_SIGMA, GeneratedBody, INCL_SIGMA, ORBITAL_A0_AU, ORBITAL_RATIO,
    PLANET_SOI_R_M, Placement, PlanetConfig, StarPhotometrics, StellarConfig, UniverseConfig,
    VISIBILITY_THETA_MIN_RAD, append_fixture_plant, child_clearance_m, reflected_photometrics,
};
use crate::celestial::OrbitalElements;
use crate::taxonomy::{
    FrostThresholds, SpectralClass, classify_spectral, habitable_zone_radius_au,
    main_sequence_luminosity, orbital_axis_au, sample_imf_mass, sample_rayleigh,
};
use core::f64::consts::TAU;
use glam::DVec3;
use vd_core::geometry::Boundary;
use vd_core::pose::RealmId;
use vd_core::rng::{SplitMix64, child_seed, realm_stream};
use vd_core::worldgen::{GALAXY, GALAXY_SEED, SYSTEM_A_SEED, UNIVERSE, UNIVERSE_SEED};

// --- FA-5 (D-45(a)) VISUAL-scale single-system generator — the COMPRESSED-REAL game scale (the ONE
// geometry; `visual_scale` is its static-render expression, `visual_demand` its live demand-cluster
// expression). Every visual geometry number is DERIVED (parameterized helpers below), not a literal. ---
/// `child_seed` salt distinguishing PLANET-kind children under a system (a fixed kind discriminant;
/// `child_seed` avalanches `(parent, salt, index)`, so a distinct salt keeps planet ids off other kinds).
pub(crate) const PLANET_SALT: u64 = 0x504c_414e_4554; // "PLANET"
/// The star realm's `child_seed` salt (taxonomy arc T2) — a star realm's seed avalanches off
/// its system exactly as a planet's does, on its own salt so the id spaces never collide.
pub(crate) const STAR_SALT: u64 = 0x0000_5354_4152; // "STAR"
/// The moon realm's `child_seed` salt (taxonomy arc T3) — a MOON IS A PLANET (the owner's
/// ruling, read literally: `RealmId::Planet` whose parent is a `RealmId::Planet`); its seed
/// avalanches off its parent planet on this salt, rung-indexed.
const MOON_SALT: u64 = 0x0000_4d4f_4f4e; // "MOON"
/// Canup & Ward's satellite-mass multiplier bounds (log-uniform, per-moon draw M.1): the
/// realised satellite-system mass spread around the 1e-4 accretion budget (the four solar
/// giants measure 1.21–2.48e-4).
const MOON_MASS_MULTIPLIER_BOUNDS: (f64, f64) = (0.5, 2.0);
/// `child_seed` salt distinguishing SYSTEM-kind children under a galaxy — the sibling-kind discriminant
/// for stars, exactly as [`PLANET_SALT`] is for planets. A system's identity is `f(galaxy, index)`, so two
/// galaxies never mint the same system id and a system's planets never collide with another system's.
const SYSTEM_SALT: u64 = 0x5359_5354_454d; // "SYSTEM"
/// How many star systems THE world's galaxy holds — the existing census parameter (owner ruling
/// Q-B: the placement law went 3-D and seeded; the census DERIVATION is P10's, this count is not).
pub(crate) const WORLD_SYSTEM_COUNT: u32 = 3;

/// THE FROZEN STREAM PREFIX WIDTH (the Stream Law, celestial_taxonomy_design §3.0/§7.1): the
/// per-system draw stream shipped with FIVE planets' element draws before the star/albedo/
/// placement draws. Growing the planet count re-rolls NOTHING because the draws for planets
/// beyond this prefix are APPENDED after the frozen prefix (elements 0..5, star, albedo 0..5,
/// placement ×2 — byte-identical forever), never inserted. A STREAM-shape constant, not a world
/// knob: the world's planet count is [`derived_planet_count`].
const LEGACY_STREAM_PLANETS: u32 = 5;

/// The smallest planet mass the generator draws, Earth masses — Mercury, the smallest confirmed
/// planet. A cited literal (real-scale design §3.3.4), stated plainly as one.
pub(crate) const PLANET_MASS_LO_MEARTH: f64 = 0.0553;
/// Protoplanetary disc-to-star mass fraction (Andrews & Williams 2005, ApJ 631:1134) — the
/// per-planet mass budget is `DISC_MASS_FRACTION · M★ / N_planets`.
pub(crate) const DISC_MASS_FRACTION: f64 = 0.01;
/// Jupiter's mass in Earth masses — the absolute per-planet draw cap (the disc arm binds on
/// every M dwarf; this cap is reachable only around heavier stars).
pub(crate) const M_JUP_MEARTH: f64 = 317.8;
/// The planet-mass draw is LOG-UNIFORM over its bounds: slope 1.0 through the existing
/// [`sample_imf_mass`] log-uniform limit branch (real-scale design §3.3.4).
const PLANET_MASS_SLOPE: f64 = 1.0;
/// Neptune's semi-major axis in AU — with [`crate::taxonomy::FROST_COEFF_AU`] it fixes the disc
/// outer edge as `NEPTUNE_SMA_AU/FROST_COEFF_AU = 11.137×` the frost line, measured on the one
/// planetary system we have (real-scale design §3.3.3).
pub(crate) const NEPTUNE_SMA_AU: f64 = 30.07;

/// The planet count, DERIVED and scale-free (real-scale design §3.3.3): the number of ladder
/// steps `a0·ratio^n` inside the disc edge `(NEPTUNE_SMA_AU/FROST_COEFF_AU) · frost_coeff ·
/// √L`. Both sides scale with √L, so the count is a property of the SPACING LAW, not the seed:
/// **9 for every star of every world** (`0.4·1.7^n ≤ 30.07 ⇔ n ≤ 8.14`). The √L cancellation is
/// why this needs no luminosity argument.
#[must_use]
pub fn derived_planet_count(a0_au: f64, ratio: f64, disc_edge_au: f64) -> u32 {
    (0..u32::MAX)
        .take_while(|&n| orbital_axis_au(n, a0_au, ratio) <= disc_edge_au)
        .count() as u32
}

/// One planet's five DRAWN orbital quantities, in the FROZEN order (ecc-u, incl-u, Ω, ω, M₀) so
/// the forest is pure `f(seed)`. ALL clamps are BRANCHLESS method calls (HR5): the ecc cap is
/// `.min(ecc_cap)` (an `if ecc > cap` would leave an UNREACHABLE true-arm at `ecc_sigma`≈0.03);
/// inclination is UN-clamped (no convergence domain, and `sample_rayleigh` is `≥ 0`).
///
/// The DERIVED halves — `sma` (the √L-anchored ladder) and `central_mass` (the star's drawn
/// mass) — are NOT here, because they consume NO draw and depend on the star's photometrics,
/// which the frozen stream order draws AFTER the legacy planets' elements: draws first,
/// derivations second ([`planet_orbital_elements`] assembles them).
#[derive(Clone, Copy, Debug)]
pub(crate) struct PlanetElementDraws {
    pub(crate) ecc: f64,
    inclination: f64,
    raan: f64,
    arg_periapsis: f64,
    mean_anomaly_epoch: f64,
}

pub(crate) fn planet_element_draws(
    config: &UniverseConfig,
    stream: &mut SplitMix64,
) -> PlanetElementDraws {
    PlanetElementDraws {
        ecc: sample_rayleigh(stream.next_f64(), config.planet.ecc_sigma).min(config.planet.ecc_cap),
        inclination: sample_rayleigh(stream.next_f64(), config.planet.incl_sigma),
        raan: stream.next_f64() * TAU,
        arg_periapsis: stream.next_f64() * TAU,
        mean_anomaly_epoch: stream.next_f64() * TAU,
    }
}

/// Assemble one planet's [`OrbitalElements`] from its frozen draws + the two DERIVED halves:
/// `sma = a0 · √L · ratio^n` in TRUE metres (THE LADDER LAW, real-scale design §3.3.2 — the
/// dead [`habitable_zone_radius_au`] made live: `√L` IS the habitable-zone radius at flux 1, so
/// the whole ladder is measured in habitable-zone radii and rung 2 sits at 0.748 S⊕ for every
/// star at every seed), and `central_mass` = the star's real drawn mass in kg.
fn planet_orbital_elements(
    config: &UniverseConfig,
    draws: PlanetElementDraws,
    n: u32,
    star: &StarPhotometrics,
) -> OrbitalElements {
    let a0_au = config.planet.orbital_a0_au * habitable_zone_radius_au(star.luma_lsun, 1.0);
    OrbitalElements {
        sma: orbital_axis_au(n, a0_au, config.planet.orbital_ratio) * crate::taxonomy::AU_M,
        ecc: draws.ecc,
        inclination: draws.inclination,
        raan: draws.raan,
        arg_periapsis: draws.arg_periapsis,
        mean_anomaly_epoch: draws.mean_anomaly_epoch,
        central_mass: star.mass_msun * crate::taxonomy::M_SUN_KG,
    }
}

/// Draw one star system's photometric identity from ITS OWN per-system `stream` (the window
/// lane's per-system draw — owner-approved 2026-08-15/16, `docs/design/window_lane.md` §2.2:
/// "luma drawn from the same seed stream the parent generated the child from"). ONE u01 draw:
/// mass through the bounded-IMF inverse-CDF, then class + luminosity as closed-form derivations
/// of that mass — the taxonomy discipline (no rejection loop, bit-reproducible). Straight-line,
/// monomorphic (HR5). Parameters are DATA off [`StellarConfig`] (no magic numbers); the MK mass
/// bounds are the taxonomy's canonical passable table, the MLR segments the config's.
fn draw_star_photometrics(st: &StellarConfig, stream: &mut SplitMix64) -> StarPhotometrics {
    let mass_msun = sample_imf_mass(
        stream.next_f64(),
        st.imf_slope,
        st.mass_lo_msun,
        st.mass_hi_msun,
    );
    StarPhotometrics {
        mass_msun,
        class: classify_spectral(mass_msun, &SpectralClass::MASS_BOUNDS),
        luma_lsun: main_sequence_luminosity(mass_msun, &st.mlr_segments),
    }
}

/// The seed of the `n`-th star system in a galaxy. System 0 keeps [`SYSTEM_A_SEED`] — it is the identity
/// every existing fixture, label and gate already names — and the rest avalanche off the galaxy through
/// the same [`child_seed`] every other sibling set uses, so a system's identity is a pure function of
/// (galaxy, index) and no two galaxies ever collide.
#[must_use]
pub(crate) fn system_seed_at(n: u32) -> u64 {
    // Branchless in the HR5 sense: ONE covered comparison, no nested control flow.
    if n == 0 {
        SYSTEM_A_SEED
    } else {
        child_seed(GALAXY_SEED, SYSTEM_SALT, u64::from(n))
    }
}

/// How many star systems a galaxy holds — its CENSUS, drawn from the galaxy's own stream inside
/// `[system_count_lo, system_count_hi]`. Pure `f(universe_seed)` against the galaxy lineage, so every
/// shard agrees on the population without exchanging a byte, and two galaxies from one universe differ
/// without anyone authoring either. `lo == hi` pins the count exactly (the walk roster's shape).
///
/// One draw, no rejection loop — the taxonomy discipline: a sampler is a closed-form map from one uniform.
#[must_use]
fn galaxy_system_count(seed_universe: u64, config: &UniverseConfig) -> u32 {
    let lo = config.galaxy.system_count_lo;
    let hi = config.galaxy.system_count_hi.max(lo);
    let span = u64::from(hi - lo) + 1;
    let mut stream = realm_stream(seed_universe, &[UNIVERSE_SEED, GALAXY_SEED]);
    // `next_f64` is [0,1); scaling by the inclusive span and truncating lands in [lo, hi] with the last
    // bucket reachable only at exactly 1.0, which the generator never produces — hence the `min`.
    let draw = (stream.next_f64() * span as f64) as u64;
    lo + u32::try_from(draw.min(span - 1)).unwrap_or(0)
}

/// Where the `n`-th system sits in its galaxy — THE 3-D SEEDED PLACEMENT LAW (owner ruling Q-B,
/// 2026-08-18: *"don't do 3 star systems in the line — get rid of this code; the whole world
/// generated from the seeds"*). The collinear ring (evenly-spaced angles on one circle in the XZ
/// plane — three stars on a line at N = 3, zero transverse parallax, §A7.1's named cost) is
/// DELETED; each non-home system takes a seeded uniform direction on the sphere at the derived
/// placement radius:
///
/// - `cos_polar = 2·u − 1`, `azimuth = TAU·v` — the standard closed-form uniform-on-the-sphere
///   construction from the system's own TWO appended stream draws (branchless, bit-reproducible);
/// - the RADIUS is the derived placement radius ([`real_placement_r_m`]) for every system — the
///   placement radius CLASS: the direction is the seed's, the magnitude is the storage budget's;
/// - system 0 (the HOME system) stays at the GALACTIC ORIGIN — the anchor the landed J1 fence
///   asserts ("a home↔galaxy crossing is numerically an identity in the drawn space"), the lineage
///   identity every fixture and gate names. The zero MULTIPLIER is the whole decision, no branch —
///   and the home draws its two placement u01s exactly like every sibling, so the stream SHAPE is
///   uniform across systems and a future re-rule of the home anchor shifts no draw.
///
/// The separation guarantee moved WITH the law: the ring's closed-form neighbour separation
/// (`2·r·sin(π/(n−1))`) is replaced by the pairwise 3-D fence over the seeded point set
/// ([`seeded_systems_disjoint_3d`] — closed-form per pair, refusal loud), measured on THE world by
/// the named pins beside it.
#[must_use]
pub(crate) fn system_center_at(
    config: &UniverseConfig,
    n: u32,
    dir_u01: f64,
    azim_u01: f64,
) -> DVec3 {
    let radius = config.stellar.system_ring_r_m;
    let cos_polar = 2.0 * dir_u01 - 1.0;
    let sin_polar = (1.0 - cos_polar * cos_polar).max(0.0).sqrt();
    let azimuth = TAU * azim_u01;
    let anchored = f64::from(u32::from(n != 0));
    DVec3::new(
        sin_polar * azimuth.cos(),
        cos_polar,
        sin_polar * azimuth.sin(),
    ) * (radius * anchored)
}

/// The config-driven star-system forest: Universe → Galaxy → `stellar.n_systems` star systems, each with
/// `planet.n_planets` `Orbital` planets (D-45(a) FA-5). A System shell IS its star's frame — the planets
/// orbit its center and the star is DATA (`stellar.central_mass_kg`), never a `RealmId::Star` (HR3).
///
/// EVERY system is built by the SAME loop from its own seed — there is no "system A" special case beyond
/// which seed index 0 carries. That is what makes a galaxy expressible at all: the previous shape named
/// one system in code and hung the planets off a constant, so a second star could not exist at any scale,
/// which is why the login side and the shard side ended up with two different worlds.
///
/// Pure `f(seed_universe, config)`: each system draws its planets from its OWN [`realm_stream`], keyed on
/// its own lineage, in a fixed order — so a shard hosting system 4 generates byte-identical planets to
/// every other shard's view of system 4, without any shared state (HR1). `n_planets == 0` emits no
/// planet; `n_systems == 0` emits the ambient forest alone.
pub(crate) fn generate_system_forest(
    seed_universe: u64,
    config: &UniverseConfig,
) -> Vec<GeneratedBody> {
    let sc = &config.scale;
    let st = &config.stellar;
    let pl = &config.planet;
    let shell = |r: f64| Boundary::Shell { r };
    let origin = Placement::StaticOffset(DVec3::ZERO);
    let mut bodies = vec![
        // Universe: the ambient ROOT (parent None) — the container-fold identity. NEVER drawn:
        // `look = None` (the two-body law — a containment boundary is structurally undrawable).
        GeneratedBody {
            realm: UNIVERSE,
            parent: None,
            shape: shell(sc.universe_r_m),
            placement: origin,
            photometrics: None,
            taxon: None,
            look: None,
        },
        // Galaxy: the finite between-systems space, nested in the Universe. NEVER drawn.
        GeneratedBody {
            realm: GALAXY,
            parent: Some(UNIVERSE),
            shape: shell(sc.galaxy_r_m),
            placement: origin,
            photometrics: None,
            taxon: None,
            look: None,
        },
    ];
    // HOW MANY STARS THIS GALAXY HOLDS — drawn from the galaxy's OWN stream against its census, so two
    // galaxies in one universe differ without anyone choosing. `lo == hi` pins it exactly.
    // (The star COUNT stays this existing config parameter — owner ruling Q-B: the placement law
    // went 3-D and seeded; the census census-derivation is P10's.)
    let n_systems = galaxy_system_count(seed_universe, config);
    for s in 0..n_systems {
        let seed = system_seed_at(s);
        let system = RealmId::System(seed);
        let system_ix = bodies.len();
        bodies.push(GeneratedBody {
            realm: system,
            parent: Some(GALAXY),
            // Filled below once the star is drawn: the shell is SOLVED (the clearance law) and
            // the look is the star's photosphere.
            shape: shell(0.0),
            placement: Placement::StaticOffset(DVec3::ZERO),
            photometrics: None,
            taxon: None,
            look: None,
        });
        // ============ THE PER-SYSTEM DRAW STREAM — the Stream Law (append-only) ============
        // FROZEN PREFIX (byte-identical since the interim world; the pins measure it):
        //   draws 1–25   five element draws for each of the first LEGACY_STREAM_PLANETS planets
        //   draw  26     the star's photometric mass (the pinned triples)
        //   draws 27–31  one geometric albedo per legacy planet
        //   draws 32–33  the two 3-D placement draws (owner ruling Q-B)
        // APPENDED by the taxonomy arc's in-system re-solve (new consumptions AFTER the frozen
        // prefix — planets 5–8 and every mass are new draws, never insertions):
        //   draws 34–53  five element draws for each planet beyond the legacy prefix
        //   draws 54–57  one geometric albedo per planet beyond the legacy prefix
        //   draws 58–66  one log-uniform mass per planet (ALL planets, index order)
        // Every `sma` and every `central_mass` is a DERIVATION (zero draws), so anchoring the
        // ladder to the star's √L consumes nothing and moves nothing.
        let mut stream = realm_stream(seed_universe, &[UNIVERSE_SEED, GALAXY_SEED, seed]);
        let legacy = pl.n_planets.min(LEGACY_STREAM_PLANETS);
        let mut element_draws: Vec<PlanetElementDraws> = (0..legacy)
            .map(|_| planet_element_draws(config, &mut stream))
            .collect();
        let star = draw_star_photometrics(st, &mut stream);
        bodies[system_ix].photometrics = Some(star);
        let mut albedo_draws: Vec<f64> = (0..legacy).map(|_| stream.next_f64()).collect();
        // THE 3-D PLACEMENT DRAWS (owner ruling Q-B, 2026-08-18) — the HOME system (index 0)
        // consumes its two draws exactly like every sibling (the anchor multiplier, not the
        // stream shape, pins it to the galactic origin).
        bodies[system_ix].placement = Placement::StaticOffset(system_center_at(
            config,
            s,
            stream.next_f64(),
            stream.next_f64(),
        ));
        // ---- the APPENDED draws (planets beyond the frozen prefix, then every mass) ----
        for _ in legacy..pl.n_planets {
            element_draws.push(planet_element_draws(config, &mut stream));
        }
        for _ in legacy..pl.n_planets {
            albedo_draws.push(stream.next_f64());
        }
        let mass_hi_mearth = planet_mass_hi_mearth(pl, &star);
        let masses_mearth: Vec<f64> = (0..pl.n_planets)
            .map(|_| {
                sample_imf_mass(
                    stream.next_f64(),
                    PLANET_MASS_SLOPE,
                    pl.mass_lo_mearth,
                    mass_hi_mearth,
                )
            })
            .collect();
        // ============ THE DERIVATIONS (consume nothing; true-size in-system space) ============
        let star_look_r_m = crate::taxonomy::star_radius_m(star.mass_msun);
        let frost_line_au =
            crate::taxonomy::frost_line_radius_au(star.luma_lsun, pl.frost_coeff_au);
        let th = pl.frost_thresholds();
        let mut planet_bodies = Vec::with_capacity(element_draws.len());
        let mut moon_inputs: Vec<(RealmId, u64, f64, f64, f64)> = Vec::new();
        for (n, (draws, mass_mearth)) in element_draws.iter().zip(&masses_mearth).enumerate() {
            let planet_seed = child_seed(seed, PLANET_SALT, n as u64);
            let elements = planet_orbital_elements(config, *draws, n as u32, &star);
            // THE PER-PLANET STREAM (T1 — a brand-new lineage nobody else reads; the Stream
            // Law's standing doctrine): P.1 = `f_env`, log-uniform over `F_ENV_BOUNDS` through
            // the existing sample_imf_mass log-uniform limit. Drawn UNCONDITIONALLY (the Stream
            // Law corollary: a draw is never conditional on a derived value); whether the
            // classifier READS it is downstream.
            let mut planet_stream = realm_stream(
                seed_universe,
                &[UNIVERSE_SEED, GALAXY_SEED, seed, planet_seed],
            );
            let f_env = sample_imf_mass(
                planet_stream.next_f64(),
                PLANET_MASS_SLOPE,
                crate::taxonomy::F_ENV_BOUNDS.0,
                crate::taxonomy::F_ENV_BOUNDS.1,
            );
            // THE ONE DERIVATION PATH (par 4.4): identical code for a planet under a star and
            // (T3) a moon under a planet.
            let taxon = crate::taxonomy::body_params(
                mass_mearth * crate::taxonomy::M_EARTH_KG,
                star.luma_lsun,
                star.class,
                elements.sma,
                frost_line_au,
                th,
                f_env,
            );
            let soi_m = planet_bound_m(pl, &star, n as u32, *mass_mearth);
            moon_inputs.push((
                RealmId::Planet(planet_seed),
                planet_seed,
                elements.sma,
                taxon.mass_kg,
                albedo_draws[n],
            ));
            planet_bodies.push(GeneratedBody {
                realm: RealmId::Planet(planet_seed),
                parent: Some(system),
                // D-REAL-1 (owner ruling 2026-08-18): the realm shell IS the gravitational
                // sphere of influence at the DRAWN mass (clamped by half the worst-instant
                // inter-orbit gap — an inert-but-live arm, fenced).
                shape: shell(soi_m),
                placement: Placement::Orbital(elements),
                // A planet's marker is REFLECTED LIGHT: the star's luma diluted over its orbit,
                // intercepted by its own DERIVED cross-section (its composition radius — its
                // LOOK, not its authority bound).
                photometrics: Some(reflected_photometrics(
                    &star,
                    albedo_draws[n],
                    taxon.radius_m,
                    elements.sma,
                )),
                taxon: Some(taxon),
                // SL3: the planet draws ITSELF at its own derived radius.
                look: Some(shell(taxon.radius_m)),
            });
        }
        // THE SYSTEM SHELL — the one clearance solve (real-scale design §3.2), bounded at the
        // MASS CAP (the worst lawful draw), so the shell is `f(star)` alone and no mass draw
        // can move it: every lawful planet fits by construction.
        bodies[system_ix].shape = shell(system_shell_r_m(&config.planet, &star));
        // THE SYSTEM'S LOOK IS ITS STAR: `star_radius_m` of the drawn mass — ONE function, TWO
        // call sites (celestial_taxonomy_design §5.2): the system's marker-side look here and
        // the Star realm's own look below carry the SAME value, so the marker→body handover is
        // the same radius at every depth. No double-draw: "a realm containing the eye is not a
        // subject" makes the two mutually exclusive.
        bodies[system_ix].look = Some(shell(star_look_r_m));
        bodies.extend(planet_bodies);
        // ★ THE STAR REALM (taxonomy arc T2, owner ruling 2026-08-19): the star becomes a
        // BODY-BEARING CHILD of its system — a separate authority volume whose crossing turns
        // on near-star physics. ZERO draws (its photometrics ARE the system's pinned draw);
        // pushed AFTER the system's planets (the §7.1 append-only forest order). Its BOUND is
        // the dust-sublimation radius (ruling E: the honest "solid matter is destroyed by heat
        // here" surface — `flux_radius_m` at 1500 K, the same equilibrium law as every planet
        // temperature, inverted); its LOOK is its own photosphere. The owner's Dyson-standoff
        // sub-item did NOT close honestly and is STOPPED per the ruling's own instruction —
        // see DEFERRED.md D-TAX-3 for the measured option set.
        bodies.push(GeneratedBody {
            realm: RealmId::Star(child_seed(seed, STAR_SALT, 0)),
            parent: Some(system),
            shape: shell(star_bound_m(&star)),
            placement: Placement::StaticOffset(DVec3::ZERO),
            // The star's marker IS its photometrics — the same kind-blind row the system
            // carries (a star's point of light is its own).
            photometrics: Some(star),
            taxon: None,
            look: Some(shell(star_look_r_m)),
        });
        // ★ THE MOON PASS (taxonomy arc T3, owner ruling: "for moons — it's the same as
        // Planet — just another realm"): moons are `RealmId::Planet` bodies whose parent is a
        // planet — zero new realm kinds, zero wire, the identical crossing/demand/draw code.
        // Pushed AFTER the star, planet order then rung order (the §7.1 append-only forest
        // order); every moon draws from its OWN per-moon lineage (the Stream Law one level
        // down: a count correction adds or removes moons without shifting a sibling's draws).
        for (planet_realm, planet_seed, planet_sma_m, planet_mass_kg, planet_albedo_u01) in
            moon_inputs
        {
            append_moons(
                &mut bodies,
                config,
                seed_universe,
                seed,
                &star,
                planet_realm,
                planet_seed,
                planet_sma_m,
                planet_mass_kg,
                planet_albedo_u01,
            );
        }
    }
    // The fixture plant (look_horizon slice 5 — G-IDENTICAL), appended LAST: with `None` (every
    // shipped constructor) this is a no-op and the forest is byte-identical to the pre-plant world.
    append_fixture_plant(&mut bodies, config);
    bodies
}

/// THE PLANET LAWS, in one place, `n_planets` apart — the ladder, the eccentricity cap, the frost
/// thresholds and the mass draw's bounds. The walk fixture forest carries none of them (`n = 0`);
/// THE world derives nine ([`derived_world_planet_count`]).
///
/// It exists as a function so the mass-cap solve can build a system's shell WITHOUT asking for a
/// `UniverseConfig` — which it could not do, because a `UniverseConfig` carries the very cap the
/// solve is solving for. Nothing here reads the stellar config; that is the whole point.
pub(crate) fn planet_config(n_planets: u32) -> PlanetConfig {
    PlanetConfig {
        planet_soi_r_m: PLANET_SOI_R_M,
        orbital_a0_au: ORBITAL_A0_AU,
        orbital_ratio: ORBITAL_RATIO,
        ecc_sigma: ECC_SIGMA,
        incl_sigma: INCL_SIGMA,
        // The GEOMETRY cap (4σ of the Rayleigh draw), NOT the solver bound: the compression
        // solves apoapsis-at-this-cap exactly onto the system shell (see ECC_CAP_SIGMAS).
        ecc_cap: ECC_SIGMA * ECC_CAP_SIGMAS,
        frost_coeff_au: crate::taxonomy::FROST_COEFF_AU,
        m_gas_mearth: FrostThresholds::CANONICAL.m_gas_mearth,
        m_core_crit_mearth: FrostThresholds::CANONICAL.m_core_crit_mearth,
        valley_r1_rearth: FrostThresholds::CANONICAL.valley_r1_rearth,
        valley_insolation_exp: FrostThresholds::CANONICAL.valley_insolation_exp,
        n_planets,
        mass_lo_mearth: PLANET_MASS_LO_MEARTH,
        disc_mass_fraction: DISC_MASS_FRACTION,
        mass_cap_mearth: M_JUP_MEARTH,
    }
}

/// THE world's planet count — the ladder steps inside the disc edge, scale-free (9 for every star
/// of every seed; both sides carry `√L`, so it cancels). ONE expression, two readers: the world
/// preset and the mass-cap solve.
pub(crate) fn derived_world_planet_count(pl: &PlanetConfig) -> u32 {
    derived_planet_count(
        pl.orbital_a0_au,
        pl.orbital_ratio,
        (NEPTUNE_SMA_AU / crate::taxonomy::FROST_COEFF_AU) * pl.frost_coeff_au,
    )
}

/// THE world's planet laws, count included — what the mass-cap solve measures a system against.
pub(crate) fn world_planet_config() -> PlanetConfig {
    let bare = planet_config(0);
    planet_config(derived_world_planet_count(&bare))
}

/// The planet mass draw's upper bound, Earth masses: `min(mass_cap, disc_fraction·M★/N)` —
/// the disc arm binds on every M dwarf (34.4/40.0/59.9 M⊕ for THE world's three stars), which
/// is observationally correct (M dwarfs do not host Jupiters). Monomorphic straight-line.
fn planet_mass_hi_mearth(pl: &PlanetConfig, star: &StarPhotometrics) -> f64 {
    let disc_budget_mearth = pl.disc_mass_fraction * star.mass_msun * crate::taxonomy::M_SUN_KG
        / crate::taxonomy::M_EARTH_KG
        / f64::from(pl.n_planets.max(1));
    pl.mass_cap_mearth.min(disc_budget_mearth)
}

/// One planet's BOUND: its gravitational sphere of influence at its drawn mass (D-REAL-1),
/// clamped by half the worst-instant gap to its neighbouring rungs — `.min` is a branchless
/// clamp; MEASURED never to bind on any lawful draw (`soi/a = (m/M★)^0.4 ≤ 0.0659` against a
/// worst-instant half-gap of `0.188·a`), kept as an inert-but-live arm with a fence rather than
/// deleted (real-scale design §3.3.4).
fn planet_bound_m(pl: &PlanetConfig, star: &StarPhotometrics, n: u32, mass_mearth: f64) -> f64 {
    let a0_au = pl.orbital_a0_au * habitable_zone_radius_au(star.luma_lsun, 1.0);
    let a_n = orbital_axis_au(n, a0_au, pl.orbital_ratio) * crate::taxonomy::AU_M;
    let soi = crate::celestial::planet_soi(
        a_n,
        mass_mearth * crate::taxonomy::M_EARTH_KG,
        star.mass_msun * crate::taxonomy::M_SUN_KG,
    );
    // Half the WORST-INSTANT gap to each REAL neighbouring rung (both rungs at the eccentricity
    // cap) — provable non-overlap at every instant for every seed (§3.3.4; the interim model
    // measured its own overlap at the cap, invisible to the static sibling fence). The first
    // rung has no inner neighbour and the last no outer one — no phantom clamp (both arms
    // driven by the rung sweep).
    let gap_out = if n + 1 < pl.n_planets {
        let a_next = orbital_axis_au(n + 1, a0_au, pl.orbital_ratio) * crate::taxonomy::AU_M;
        0.5 * (a_next * (1.0 - pl.ecc_cap) - a_n * (1.0 + pl.ecc_cap))
    } else {
        f64::INFINITY
    };
    let gap_in = if n > 0 {
        let a_prev = orbital_axis_au(n - 1, a0_au, pl.orbital_ratio) * crate::taxonomy::AU_M;
        0.5 * (a_n * (1.0 - pl.ecc_cap) - a_prev * (1.0 + pl.ecc_cap))
    } else {
        f64::INFINITY
    };
    soi.min(gap_out.min(gap_in))
}

/// THE STAR REALM'S BOUND (taxonomy arc §5.3, owner ruling E): the DUST-SUBLIMATION radius —
/// the distance at which silicate dust reaches [`crate::taxonomy::DUST_SUBLIMATION_K`]
/// (Dullemond & Monnier 2010: the inner rim of every observed protoplanetary disc). ONE law,
/// two consumers, two depths: this is [`crate::taxonomy::flux_radius_m`] — the exact inverse of
/// every planet's equilibrium temperature — at `A = 0`, `T = T_sub`. Scale-free identity:
/// `bound / a₀ = 0.0861` for every star at every seed (both scale with √L) — pinned by the T2
/// gates. NO sibling clamp ships (the identity makes one unable to ever bind — an arm HR5
/// would need a synthetic the world cannot produce); only the photosphere fence below.
fn star_bound_m(star: &StarPhotometrics) -> f64 {
    crate::taxonomy::flux_radius_m(
        star.luma_lsun * crate::taxonomy::L_SUN_W,
        crate::taxonomy::DUST_SUBLIMATION_K,
        0.0,
    )
}

/// ★ T3 — ONE planet's MOON PASS (celestial_taxonomy_design §4): the count law (density-
/// corrected Roche inner edge, the `ratio^(k+1)` offset ladder, the Machida `R_Hill/48` disc
/// edge), the Σ ∝ 1/a gas-starved mass partition with the per-moon log-uniform multiplier,
/// the potato-radius emission floor, and the climb-1 clearance clamp as the disc edge's
/// THIRD `child_clearance_m` consumer (live, measured never to bind). A moon IS a
/// `RealmId::Planet` under a planet: its bound is its OWN gravitational SOI (Hill-class,
/// the same `celestial::planet_soi`, different arguments), its look its own derived radius,
/// its taxon the ONE `body_params` path with the illuminator at its PARENT's orbit.
#[allow(clippy::too_many_arguments)] // one private pass; every argument is the parent's own datum
pub(crate) fn append_moons(
    bodies: &mut Vec<GeneratedBody>,
    config: &UniverseConfig,
    seed_universe: u64,
    system_seed: u64,
    star: &StarPhotometrics,
    planet_realm: RealmId,
    planet_seed: u64,
    planet_sma_m: f64,
    planet_mass_kg: f64,
    planet_albedo_u01: f64,
) -> u32 {
    use crate::taxonomy::{
        MOON_ECC_SIGMA, MOON_INCL_SIGMA, MOON_MIN_RADIUS_M, RHO_ICE_KGM3, RHO_ROCK_KGM3,
        SATELLITE_DISC_HILL_FRACTION, SATELLITE_MASS_FRACTION, hill_radius_m, roche_radius_m,
        satellite_disc_edge_m, satellite_ladder_a_m,
    };
    let pl = &config.planet;
    let frost_line_au = crate::taxonomy::frost_line_radius_au(star.luma_lsun, pl.frost_coeff_au);
    let inside_frost = planet_sma_m / crate::taxonomy::AU_M < frost_line_au;
    // Moon composition density: ice beyond the frost line, rock inside (the satellite census's
    // own calibration: Ganymede/Callisto/Titan ~1900 vs the Moon 3344).
    let rho_moon = if inside_frost {
        RHO_ROCK_KGM3
    } else {
        RHO_ICE_KGM3
    };
    let roche_m = roche_radius_m(planet_mass_kg, rho_moon);
    let star_mass_kg = star.mass_msun * crate::taxonomy::M_SUN_KG;
    let disc_edge_m = satellite_disc_edge_m(
        hill_radius_m(planet_sma_m, planet_mass_kg, star_mass_kg),
        SATELLITE_DISC_HILL_FRACTION,
    );
    let planet_soi_m = bodies
        .iter()
        .find(|b| b.realm == planet_realm)
        .expect("the moon pass runs after its planet is pushed")
        .shape
        .finite_extent();
    // THE COUNT: ladder rungs inside the disc edge — and inside the climb-1 clearance clamp
    // (the §4.5.1 third `child_clearance_m` arm: a rung whose worst instant + clearance would
    // breach the planet's SOI is not minted; measured 25×-slack inert on THE world, fenced by
    // the census pin printing the counts).
    let budget_kg = SATELLITE_MASS_FRACTION * planet_mass_kg;
    let worst_moon_mass_kg = MOON_MASS_MULTIPLIER_BOUNDS.1 * budget_kg;
    let mut rungs_m: Vec<f64> = Vec::new();
    for k in 0..64u32 {
        let a_k = satellite_ladder_a_m(k, roche_m, pl.orbital_ratio);
        if a_k > disc_edge_m {
            break;
        }
        let worst_bound_m = crate::celestial::planet_soi(a_k, worst_moon_mass_kg, planet_mass_kg);
        let worst_look_m = crate::taxonomy::composition_radius_m(
            crate::taxonomy::PlanetType::Rocky,
            inside_frost,
            worst_moon_mass_kg / crate::taxonomy::M_EARTH_KG,
            0.0,
            star.luma_lsun / (planet_sma_m / crate::taxonomy::AU_M).powi(2),
        );
        let ecc_cap = crate::taxonomy::MOON_ECC_SIGMA * ECC_CAP_SIGMAS;
        if a_k * (1.0 + ecc_cap)
            + child_clearance_m(worst_bound_m, worst_look_m, VISIBILITY_THETA_MIN_RAD)
            > planet_soi_m
        {
            break;
        }
        rungs_m.push(a_k);
    }
    let share_denominator: f64 = rungs_m.iter().sum();
    let th = pl.frost_thresholds();
    let mut emitted = 0u32;
    for (k, a_k) in rungs_m.iter().enumerate() {
        let moon_seed = child_seed(planet_seed, MOON_SALT, k as u64);
        // THE PER-MOON STREAM (M.1..M.6, frozen order): mass multiplier, ecc, incl, Ω, ω, M₀
        // — keyed on the MOON's own lineage, so a later count correction never shifts a
        // surviving sibling's draws (the Stream Law's own reason).
        let mut moon_stream = realm_stream(
            seed_universe,
            &[
                UNIVERSE_SEED,
                GALAXY_SEED,
                system_seed,
                planet_seed,
                moon_seed,
            ],
        );
        let multiplier = sample_imf_mass(
            moon_stream.next_f64(),
            PLANET_MASS_SLOPE,
            MOON_MASS_MULTIPLIER_BOUNDS.0,
            MOON_MASS_MULTIPLIER_BOUNDS.1,
        );
        let ecc = sample_rayleigh(moon_stream.next_f64(), MOON_ECC_SIGMA)
            .min(MOON_ECC_SIGMA * ECC_CAP_SIGMAS);
        let inclination = sample_rayleigh(moon_stream.next_f64(), MOON_INCL_SIGMA);
        let raan = moon_stream.next_f64() * TAU;
        let arg_periapsis = moon_stream.next_f64() * TAU;
        let mean_anomaly_epoch = moon_stream.next_f64() * TAU;
        // Σ ∝ 1/a partition: annulus mass ∝ a on a geometric ladder — masses increase
        // outward and the outermost dominates, the two observed facts the law reproduces.
        let mass_kg = budget_kg * (a_k / share_denominator) * multiplier;
        // THE ONE DERIVATION PATH (§4.4): the identical `body_params` a planet takes, with
        // the illuminator distance at the PARENT's orbit. `f_env = 0`: the per-moon stream
        // draws none (M.1..M.6), and no lawful moon mass can classify envelope-retaining
        // (the whole domain sits far under every rung's stripped ceiling).
        let taxon = crate::taxonomy::body_params(
            mass_kg,
            star.luma_lsun,
            star.class,
            planet_sma_m,
            frost_line_au,
            th,
            0.0,
        );
        // THE POTATO FLOOR (Lineweaver & Norman 2010): below hydrostatic equilibrium no
        // realm is emitted — rubble is a later debris slice. The DRAWS above happened
        // unconditionally (the Stream Law corollary); only the emission is gated.
        if taxon.radius_m < MOON_MIN_RADIUS_M {
            continue;
        }
        emitted += 1;
        bodies.push(GeneratedBody {
            // A MOON IS A PLANET (the ruling, literally): same kind, parent = a planet.
            realm: RealmId::Planet(moon_seed),
            parent: Some(planet_realm),
            // Its bound is its OWN gravitational SOI (Hill-class — the same one function).
            shape: Boundary::Shell {
                r: crate::celestial::planet_soi(*a_k, mass_kg, planet_mass_kg),
            },
            placement: Placement::Orbital(OrbitalElements {
                sma: *a_k,
                ecc,
                inclination,
                raan,
                arg_periapsis,
                mean_anomaly_epoch,
                central_mass: planet_mass_kg,
            }),
            // A moon's marker IS reflected light — the one existing function, the star as
            // illuminator, the PLANET's orbit as the dilution distance. Its albedo reuses its
            // parent planet's drawn u01 (per-moon albedo variety is the design's own named
            // deferral — a future M.7 append, never a re-roll).
            photometrics: Some(reflected_photometrics(
                star,
                planet_albedo_u01,
                taxon.radius_m,
                planet_sma_m,
            )),
            taxon: Some(taxon),
            look: Some(Boundary::Shell { r: taxon.radius_m }),
        });
    }
    emitted
}

/// THE SYSTEM SHELL SOLVE (real-scale design §3.2): `max` over the ladder rungs of
/// `worst_excursion + child_clearance_m(bound_at_cap, look_at_cap)` — each rung bounded at the
/// system's own MASS CAP, the worst lawful draw, so the shell is a pure function of the star
/// and no planet draw can move it. The §3.2 identity then gives every child a stopping slack
/// equal to exactly the clearance the solve reserved — the structural climb-1 proof.
pub(crate) fn system_shell_r_m(pl: &PlanetConfig, star: &StarPhotometrics) -> f64 {
    let cap_mearth = planet_mass_hi_mearth(pl, star);
    let a0_au = pl.orbital_a0_au * habitable_zone_radius_au(star.luma_lsun, 1.0);
    // The STAR child's own clearance arm (T2): zero excursion + the clearance its bound/look
    // owe. LIVE, and MEASURED never to bind (§5.3.2 Prediction A: the outer planet's term wins
    // by ~13× — `g_star_shell_unmoved` proves the shells byte-identical with the arm in).
    let star_term = child_clearance_m(
        star_bound_m(star),
        crate::taxonomy::star_radius_m(star.mass_msun),
        VISIBILITY_THETA_MIN_RAD,
    );
    (0..pl.n_planets)
        .map(|n| {
            let a_n = orbital_axis_au(n, a0_au, pl.orbital_ratio) * crate::taxonomy::AU_M;
            let insolation_rel = star.luma_lsun
                / (orbital_axis_au(n, a0_au, pl.orbital_ratio)
                    * orbital_axis_au(n, a0_au, pl.orbital_ratio));
            let bound_cap_m = planet_bound_m(pl, star, n, cap_mearth);
            let look_cap_m = worst_rung_look_m(pl, insolation_rel, cap_mearth);
            a_n * (1.0 + pl.ecc_cap)
                + child_clearance_m(bound_cap_m, look_cap_m, VISIBILITY_THETA_MIN_RAD)
        })
        .fold(star_term, f64::max)
}

/// The LARGEST look any lawful draw can put on a rung (the shell solve's per-rung bound): the
/// max over (a) the Chen–Kipping radius at the mass cap (the giant/population envelope), (b)
/// the ice core at the cap, and (c) the retained composition radius at both ENDPOINTS of the
/// retained-mass interval — endpoints suffice because `core + envelope` is `a·m^0.27 +
/// b·m^-0.21`, whose single interior critical point is a MINIMUM (the derivative goes negative
/// → positive), so the maximum sits at an end. Ice coefficient throughout (≥ rock). Monomorphic.
fn worst_rung_look_m(pl: &PlanetConfig, insolation_rel: f64, cap_mearth: f64) -> f64 {
    use crate::taxonomy::{
        F_ENV_BOUNDS, ICE_MR_SEGMENTS, PMR_SEGMENTS, R_EARTH_M, RADIUS_VALLEY_1SEARTH_REARTH,
        RADIUS_VALLEY_INSOLATION_EXP, SYSTEM_AGE_GYR, envelope_radius_rearth, radius_valley_rearth,
        segmented_power_law,
    };
    let ice_core = |m: f64| segmented_power_law(m, &ICE_MR_SEGMENTS);
    let retained = |m: f64| {
        ice_core(m) + envelope_radius_rearth(m, F_ENV_BOUNDS.1, insolation_rel, SYSTEM_AGE_GYR)
    };
    // The smallest mass the classifier can call retained at this insolation: the rock core
    // crosses the valley (`m^0.27 = valley` — the par 3.4.5 stripped ceiling), floored at the
    // draw's own lower bound.
    let valley = radius_valley_rearth(
        insolation_rel,
        RADIUS_VALLEY_1SEARTH_REARTH,
        RADIUS_VALLEY_INSOLATION_EXP,
    );
    let m_retained_lo = valley
        .powf(1.0 / ICE_MR_SEGMENTS[0].2)
        .clamp(pl.mass_lo_mearth, cap_mearth);
    let candidates = [
        segmented_power_law(cap_mearth, &PMR_SEGMENTS),
        ice_core(cap_mearth),
        retained(m_retained_lo),
        retained(cap_mearth),
    ];
    R_EARTH_M * candidates.iter().fold(0.0_f64, |a, &b| a.max(b))
}
