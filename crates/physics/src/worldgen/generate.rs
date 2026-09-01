//! THE GENERATION: the draws that turn one seed into a star system.
//!
//! Owns: the per-kind seed salts, the FROZEN draw order (change it and every world changes), the
//! star's mass and photometrics, the planet ladder with its masses and orbital elements, the moon
//! pass, and the shell solve that sizes a system around the worst draw it could have made.
//!
//! Does NOT own: a second stream. The draw order IS the world's identity — a new quantity is
//! appended, never inserted — which is why the stream prefix width is a named constant rather than a
//! consequence of how the loop happens to be written.

use std::collections::BTreeMap;

use super::{
    ECC_CAP_SIGMAS, ECC_SIGMA, GeneratedBody, INCL_SIGMA, ORBITAL_A0_AU, ORBITAL_RATIO,
    PLANET_SOI_R_M, Placement, PlanetConfig, StarPhotometrics, StellarConfig, UniverseConfig,
    VISIBILITY_THETA_MIN_RAD, append_fixture_plant, child_clearance_m, reflected_photometrics,
};
use crate::celestial::OrbitalElements;
use crate::taxonomy::{
    FrostThresholds, GalaxyType, SpectralClass, classify_spectral, habitable_zone_radius_au,
    main_sequence_luminosity, orbital_axis_au, sample_galaxy_type, sample_imf_mass,
    sample_rayleigh,
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
/// `child_seed` salt for the PUSH draw (G12) — its own lineage, so asking a star how far to move
/// consumes nothing from the streams that decide what it IS. The Stream Law's standing doctrine: a
/// new question gets a new stream, never a draw appended to someone else's.
const PUSH_SALT: u64 = 0x0000_5055_5348; // "PUSH"
// (`WORLD_SYSTEM_COUNT: u32 = 3` IS DELETED, S12/G8 2026-08-28. It stated a population, and the
// owner's ruling names it: *"Count is a result — I agree with that. 150K is the target number, not
// exact amount all galaxies should have. The amount also should come from the seed."* A galaxy's
// population now FOLLOWS from the volume its disc encloses and the density it drew — see
// `DISC_DENSITY_LO_PER_PC3`.)

/// THE DISC'S SCALE LENGTH, as a fraction of the storage rim — the radius at which its density has
/// fallen to `1/e` of the centre's.
///
/// ★ THE GALAXY MUST NOT END AT A LINE (owner, 2026-08-29). A quarter is chosen so the exponential
/// profile puts the great majority of stars well inside the rim and thins smoothly outward, with the
/// storage clamp reached by a small tail that is already almost empty. The Milky Way's own disc scale
/// length is about 2.6 kpc against a visible radius near 15 kpc — close to a sixth — so a quarter is
/// a slightly more compact disc than ours, which is the safer direction for a bounded lattice.
const DISC_SCALE_LENGTH_FRAC: f64 = 0.25;

/// How far the vertical exponential reaches, as a multiple of the drawn half-thickness. The drawn
/// thickness stays the disc's CHARACTERISTIC height; this turns a hard slab edge into a fade, so a
/// few stars sit well above the plane exactly as they do in a real disc.
const DISC_HEIGHT_SCALE_FRAC: f64 = 0.5;

/// The star systems per cubic parsec a galaxy's disc holds, at its sparsest — and the densest below.
///
/// ★ WHY THESE ARE ABOVE THE SOLAR NEIGHBOURHOOD, WHICH IS 0.1 (RECONS ten-parsec census, already
/// cited by [`super::scale::real_compression_chi`]). The Sun sits in a SPARSE OUTER REGION of its
/// own galaxy. A disc's density falls off outward, so the disc AVERAGE — which is what a count over
/// the whole disc reads — is several times the value measured where we happen to live. A few tenths
/// per cubic parsec is an ordinary galactic disc, not a crowded one.
///
/// ★ AND THE WORLD IS CURRENTLY FAR EMPTIER THAN REAL SPACE, which is the fact that makes this a
/// CORRECTION rather than a crowding. MEASURED 2026-08-28: the placement radius is 4.6094e18 m =
/// 487.2 light years, and three systems sit in it. Real space at the solar density would hold about
/// thirty thousand there. The compression note beside `real_compression_chi` still says the world is
/// packed 24.568x TIGHTER than real; that was true before the S9 climb moved the placement radius by
/// 3 075x, and the world is now about 125x SPARSER. Raising the count removes an emptiness that was
/// never intended.
const DISC_DENSITY_LO_PER_PC3: f64 = 0.25;
/// The star systems per cubic parsec a galaxy's disc holds, at its densest.
const DISC_DENSITY_HI_PER_PC3: f64 = 0.75;
/// A parsec in metres — the unit the density above is measured in, because that is the unit the
/// surveys publish.
const PARSEC_IN_M: f64 = 3.085_677_581_491_367_3e16;

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

/// ★ ONE GALAXY'S WHOLE PROFILE — its census, its kind and its shape, all drawn from ITS OWN stream
/// (S12, owner ruling G1: *"Density also should come from seed, otherwise any tiny change might
/// change positions."*).
///
/// ★ WHY ONE FUNCTION AND NOT THREE. The census and the kind used to be two functions, each rebuilding
/// this stream from scratch and then SKIPPING the draws it did not want — the kind's body read
/// `let _count_draw = stream.next_f64();` before taking its own. That works and it is fragile in a
/// silent way: a third reader must remember to skip two, a fourth to skip three, and forgetting does
/// not fail, it returns the WRONG NUMBER. Reading the stream once, in one place, in order, makes the
/// order a fact of the code rather than a rule to remember.
///
/// ★ THE STREAM IS APPENDED, NEVER INSERTED (the Stream Law). The first two draws are exactly what
/// they were, so every galaxy's count and kind are byte-identical across this change. The shape draws
/// follow them.
///
/// ★ AND NO SYSTEM'S OWN DRAW MOVES — verified in the code, not assumed. A system reads from
/// `realm_stream(seed, &[UNIVERSE_SEED, GALAXY_SEED, system_seed])`, a DIFFERENT stream keyed by its own
/// seed. Appending here cannot reach it. What moves the stars is the shape VALUES changing from chosen
/// literals to drawn numbers, which is the whole point of the ruling.
pub(crate) struct GalaxyProfile {
    /// How many star systems this galaxy holds.
    pub(crate) count: u32,
    /// Spiral, elliptical or irregular.
    pub(crate) kind: GalaxyType,
    /// What it looks like — every number of it drawn.
    pub(crate) shape: GalaxyShape,
    /// The disc density this galaxy drew, in star systems per cubic parsec. The count above is this
    /// times the volume its disc encloses, and nothing else.
    pub(crate) density_per_pc3: f64,
}

/// HOW MANY STAR SYSTEMS A GALAXY OF THIS SHAPE AT THIS DENSITY HOLDS (owner ruling G8).
///
/// The count is the volume the galaxy encloses at the density it drew, and it is a RESULT: nobody
/// states it, and it moves when the shape or the density moves, which is what the ruling asks for.
///
/// ★ THE VOLUME IS THE SHAPE'S OWN, NOT ALWAYS A DISC (corrected 2026-08-28, by measuring). A disc
/// of radius `R` and full thickness `t·R` encloses `π·R²·(t·R)`, and that is right for a spiral. It
/// is NOT right for an ELLIPTICAL, which is a round swell rather than a plate: its `t` is 0.5 by
/// construction, and putting that through the disc formula treats it as a plate half as thick as it
/// is wide. MEASURED across 64 seeds with the disc formula everywhere, the population ran 48 291 to
/// 3 891 262 — and the top of that range was ellipticals being counted as impossibly fat discs, not
/// galaxies genuinely differing.
///
/// So an elliptical is measured as the spheroid it is: `(4/3)·π·R³` flattened by its own roundness.
///
/// The bulge is not added on top of either. Its stars are drawn from the same population —
/// `bulge_fraction` is the SHARE of the galaxy that sits in the middle, not an extra crowd beside it.
fn galaxy_population(
    config: &UniverseConfig,
    kind: GalaxyType,
    shape: &GalaxyShape,
    density_per_pc3: f64,
) -> u32 {
    let r_m = config.stellar.galaxy_rim_r_m;
    let disc_m3 = core::f64::consts::PI * r_m * r_m * (shape.disc_thickness_frac * r_m);
    let spheroid_m3 = 4.0 / 3.0 * core::f64::consts::PI * r_m * r_m * r_m * shape.bulge_roundness;
    let volume_m3 = match kind {
        GalaxyType::Spiral | GalaxyType::Irregular => disc_m3,
        GalaxyType::Elliptical => spheroid_m3,
    };
    let volume_pc3 = volume_m3 / (PARSEC_IN_M * PARSEC_IN_M * PARSEC_IN_M);
    // At least one: a galaxy with no star at all is not a galaxy, and the home system is index 0.
    let n = (density_per_pc3 * volume_pc3).max(1.0);
    // Saturating rather than wrapping — a count that wrapped to nothing would be silent.
    if n >= f64::from(u32::MAX) {
        u32::MAX
    } else {
        n as u32
    }
}

/// Read a galaxy's whole profile off its own stream, in order.
#[must_use]
pub(crate) fn galaxy_profile(seed_universe: u64, config: &UniverseConfig) -> GalaxyProfile {
    let mut stream = realm_stream(seed_universe, &[UNIVERSE_SEED, GALAXY_SEED]);
    // ---- draw 1: THE DISC DENSITY ----
    //
    // ★ THIS DRAW USED TO BE THE COUNT ITSELF, read against a pair of config knobs a person set. The
    // owner's ruling G8 retired that: a population is a RESULT, never a number anyone states. The
    // draw keeps its POSITION so the kind and the shape below stay byte-identical — the Stream Law
    // is about where a draw sits, not what it means.
    let density_per_pc3 = spread(
        stream.next_f64(),
        DISC_DENSITY_LO_PER_PC3,
        DISC_DENSITY_HI_PER_PC3,
    );
    // ---- draw 2: THE KIND (unchanged since it landed earlier in S12) ----
    let kind = sample_galaxy_type(stream.next_f64(), &config.galaxy.type_cumulative);
    // ---- draws 3+: THE SHAPE ----
    let shape = draw_galaxy_shape(kind, &mut stream);
    // ---- AND THE COUNT FOLLOWS (G8) ----
    let count = galaxy_population(config, kind, &shape, density_per_pc3);
    GalaxyProfile {
        count,
        kind,
        shape,
        density_per_pc3,
    }
}

/// ★ THE SYSTEM LAYER — the galaxy's stars, and nothing inside them (owner ruling 2026-08-29).
///
/// The ambient Universe and Galaxy, then one body per star system: its place, its shell, its look and
/// its star's light. No planets, no moons, no star children.
///
/// ★ WHY IT EXISTS. The sky is a list of STAR SYSTEMS. The boot fold used to build the whole forest
/// and keep the systems out of it — MEASURED on THE world, 3 500 479 objects to state 233 220 rows.
/// The owner ruled that a boot builds what its consumer needs.
///
/// ★ AND IT IS NOT A SECOND WORLD. Every system here is placed by [`seed_one_system`], the same
/// function the full forest uses, reading the same stream in the same order — and pushed by the same
/// [`push_crowded_systems`]. A system's identity, place, shell and star are therefore identical in
/// both, which the byte-identity gate beside this proves rather than assumes. What differs is only
/// what is NOT built.
#[must_use]
pub(crate) fn generate_system_layer(
    seed_universe: u64,
    config: &UniverseConfig,
) -> Vec<GeneratedBody> {
    let sc = &config.scale;
    let shell = |r: f64| Boundary::Shell { r };
    let profile = galaxy_profile(seed_universe, config);
    let origin = Placement::StaticOffset(DVec3::ZERO);
    let mut bodies = vec![
        GeneratedBody {
            realm: UNIVERSE,
            parent: None,
            shape: shell(sc.universe_r_m),
            placement: origin,
            photometrics: None,
            taxon: None,
            look: None,
        },
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
    for s in 0..profile.count {
        let seed = system_seed_at(s);
        let system_ix = bodies.len();
        bodies.push(GeneratedBody {
            realm: RealmId::System(seed),
            parent: Some(GALAXY),
            shape: shell(0.0),
            placement: origin,
            photometrics: None,
            taxon: None,
            look: None,
        });
        let seeded = seed_one_system(
            seed_universe,
            config,
            &profile.shape,
            s,
            seed,
            &mut bodies[system_ix],
        );
        // The system's own shell and look are solved from its star — the same two derivations the
        // full forest makes, so a system is the same size in both.
        bodies[system_ix].shape = shell(system_shell_r_m(&config.planet, &seeded.star));
        bodies[system_ix].look = Some(shell(crate::taxonomy::star_radius_m(seeded.star.mass_msun)));
    }
    push_crowded_systems(&mut bodies, config);
    bodies
}

/// What [`seed_one_system`] hands back: the system is fully defined, and these are the draws its
/// CHILDREN need — carried, never re-derived, so the two callers cannot read the stream differently.
struct SeededSystem {
    stream: vd_core::rng::SplitMix64,
    star: StarPhotometrics,
    element_draws: Vec<PlanetElementDraws>,
    albedo_draws: Vec<f64>,
}

/// ★ ONE SYSTEM, DEFINED — its star, its shell, its look and its place, and nothing inside it.
///
/// ★ WHY THIS IS ITS OWN FUNCTION (2026-08-29). The sky needs STAR SYSTEMS. It does not need their
/// planets, their moons or their moons' moons. But the boot fold built the whole forest and then
/// kept the systems: MEASURED on THE world, 3 500 479 objects produced to state 233 220 star rows —
/// fifteen out of every sixteen built and thrown away.
///
/// The owner's rule (2026-08-29) is that a boot builds what its consumer needs, and that no path
/// gets a second spelling of a shared law. So the system-defining stage lives HERE and BOTH callers
/// use it: the sky's system-layer fold, and the full forest. There is exactly one place that reads a
/// system's stream, so the two can never disagree about where a star is or what colour it is.
///
/// The draws the CHILDREN need ride back in [`SeededSystem`] rather than being taken again, because
/// re-deriving them is precisely how a fast path and a slow path stop agreeing.
fn seed_one_system(
    seed_universe: u64,
    config: &UniverseConfig,
    shape: &GalaxyShape,
    s: u32,
    seed: u64,
    body: &mut GeneratedBody,
) -> SeededSystem {
    let pl = &config.planet;
    let st = &config.stellar;
    let mut stream = realm_stream(seed_universe, &[UNIVERSE_SEED, GALAXY_SEED, seed]);
    let legacy = pl.n_planets.min(LEGACY_STREAM_PLANETS);
    let element_draws: Vec<PlanetElementDraws> = (0..legacy)
        .map(|_| planet_element_draws(config, &mut stream))
        .collect();
    let star = draw_star_photometrics(st, &mut stream);
    body.photometrics = Some(star);
    let albedo_draws: Vec<f64> = (0..legacy).map(|_| stream.next_f64()).collect();
    // THE SHAPE'S PLACEMENT DRAWS (S12; owner rulings G2, G9, G13). SIX, where the sphere took
    // two: a population, a radius, an azimuth, TWO halves of an arm scatter and a height. The
    // HOME system consumes all six exactly like every sibling — the anchor multiplier, not the
    // stream shape, pins it to the galactic origin.
    // ★ EVERY STAR SYSTEM LANDS ON THE GALAXY'S OWN GRID (2026-08-31), not only the pushed ones.
    //
    // A galaxy counts in whole cells and a star catalogue row carries the CELL and no sub-cell part.
    // The shaped placement returns a continuous position, so a system landed between cells and the
    // row silently dropped the remainder — every client then drew that star up to half a cell from
    // where the world put it.
    //
    // MEASURED on the test galaxy: 33 of 48 systems sat off-cell, by up to 1.875 m on a 2 m grid. The
    // snap I first added inside the crowding push cured only the pushed ones, which is why the fence
    // stayed red — the fault was never the push, it was the placement.
    //
    // The cost is at most half a cell — one metre against inter-star distances of 1e16 m.
    body.placement = Placement::StaticOffset(on_galaxy_cell(system_center_at(
        config,
        shape,
        s,
        PlacementDraws {
            population: stream.next_f64(),
            radius: stream.next_f64(),
            radius_b: stream.next_f64(),
            azimuth: stream.next_f64(),
            scatter: stream.next_f64(),
            scatter_b: stream.next_f64(),
            height: stream.next_f64(),
        },
    )));
    SeededSystem {
        stream,
        star,
        element_draws,
        albedo_draws,
    }
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
    shape: &GalaxyShape,
    n: u32,
    draws: PlacementDraws,
) -> DVec3 {
    // THE HOME SYSTEM sits at the galactic origin. A multiplier, never a branch — and it consumes its
    // draws exactly like every sibling, so the stream SHAPE is uniform across systems and a future
    // re-rule of the home anchor shifts nothing.
    let anchored = f64::from(u32::from(n != 0));
    let r_max = config.stellar.galaxy_rim_r_m;

    // WHICH POPULATION this system belongs to. ONE draw decides all three, by reading it against two
    // thresholds — a second draw would buy nothing.
    //
    //   bulge      a round central swell
    //   inter-arm  in the disc, but between the arms
    //   arm        in the disc, following an arm
    //
    // ★ THE INTER-ARM POPULATION IS WHAT STOPS THE SPIRAL LOOKING DRAWN (owner, 2026-08-28: *"I'd also
    // add more randomness into the spiral. Now it's ideal, which never happens"*). A real spiral is not
    // empty between its arms — it has a smooth disc underneath them, and the arms are where stars
    // CROWD, not where they exclusively live. Arms with nothing between them read as painted.
    let in_bulge = draws.population < shape.bulge_fraction;
    let in_arm = draws.population >= shape.bulge_fraction + shape.interarm_fraction;

    // THE RADIUS, from a closed-form profile — never a rejection loop (the taxonomy discipline: a
    // sampler is a closed-form map from one uniform).
    //
    // `r = R · u^p` puts the surface density at `∝ r^(1/p − 2)`: p = 0.5 spreads stars evenly over the
    // disc, and larger p pulls them inward. The exponent is drawn per galaxy, so two galaxies from one
    // universe are concentrated differently without anyone choosing.
    let p = if in_bulge {
        shape.bulge_exponent
    } else {
        shape.disc_exponent
    };
    // ★ A DISC HAS NO LAST STAR (owner, 2026-08-29: "in reality there is no crisp line where galaxy
    // ends"). `r = R · u^p` cannot place a star past R, so the galaxy ended at a WALL — visible as a
    // hard straight edge across the sky, which is what the owner saw and no galaxy does.
    //
    // A real disc's surface density falls off EXPONENTIALLY with radius (Freeman 1970, the exponential
    // disc — the standard model for every spiral). Drawing from that profile has no edge at all: the
    // density simply becomes small, and the few far stars thin out instead of stopping.
    //
    // `r = -h · ln(1-u)` draws exactly that profile from one uniform, in closed form, with no
    // rejection loop. `h` is the scale length: the radius at which the density has fallen by `1/e`.
    // The old exponent still shapes the draw, so a galaxy that drew a concentrated profile stays
    // concentrated — it simply no longer ends abruptly.
    // THE RADIUS, from the exponential-disc profile — the standard model for every spiral (Freeman
    // 1970). Two uniforms give `r · exp(-r/h)` in closed form, with no rejection loop: few stars at
    // the very centre, a peak near the scale length, and a smooth fade outward with NO EDGE.
    //
    // The drawn exponent still shapes the profile, so a galaxy that drew a concentrated disc stays
    // concentrated — it simply has neither a wall nor a knot.
    // ★ THE BULGE GETS ITS OWN SCALE, IT DOES NOT SHRINK THE DISC'S DRAW (corrected 2026-08-29).
    // The bulge used to MULTIPLY an already-drawn radius by its fraction, which compounded into a
    // pinpoint knot with a hard edge — visible from the galactic pole as a tight blob, which is what
    // the owner saw. A bulge is the same kind of profile with a SHORTER scale length: a round central
    // concentration, not a squeezed copy of the disc.
    //
    // The concentration exponent modulates the SCALE rather than the drawn value, so a concentrated
    // galaxy has a tighter disc instead of a smaller one.
    let u1 = draws.radius.clamp(1.0e-12, 1.0);
    let u2 = draws.radius_b.clamp(1.0e-12, 1.0);
    let scale_length = r_max
        * if in_bulge {
            DISC_SCALE_LENGTH_FRAC * shape.bulge_radius_frac
        } else {
            DISC_SCALE_LENGTH_FRAC
        }
        * (2.0 * p);
    let r_soft = -scale_length * (u1.ln() + u2.ln());
    // ★ SQUEEZED TO THE RIM, NEVER CUT AT IT (corrected 2026-08-29, by measurement).
    //
    // The first attempt wrote `r_soft.min(r_max)`. That is a CLIFF wearing a different hat: it maps a
    // whole RANGE of draws onto ONE radius, so every star in the tail lands at exactly the rim. I
    // claimed it would catch "well under a percent" and did not measure it. MEASURED: 6 352 of
    // 233 220 systems — 2.7% — landed at an IDENTICAL distance, which is a shell at the rim, and
    // ruling G3 forbids two systems sharing a distance by name.
    //
    // `tanh` bounds without cutting. It is MONOTONE, so two different draws always give two different
    // radii and nothing can pile up; it is the identity to within a rounding error well inside the
    // rim, so the disc a viewer actually sees is untouched; and it approaches the rim without ever
    // reaching it, so the storage bound holds by construction rather than by a clamp.
    let r = r_max * (r_soft / r_max).tanh();

    // THE ANGLE. In the disc it follows an ARM; in the bulge there are no arms to follow.
    //
    // ★ A LOGARITHMIC SPIRAL is the arm's own law: its angle advances with the LOGARITHM of the
    // radius, at a fixed pitch. That is what makes an arm sweep rather than curl or straighten.
    let azimuth = if in_bulge || !in_arm {
        // A bulge has no arms to follow; an inter-arm system is between them by definition.
        TAU * draws.azimuth
    } else {
        let arm = (draws.azimuth * f64::from(shape.arms)).floor();
        let arm_base = TAU * arm / f64::from(shape.arms);
        // ln(r/R) is negative inward, so the arm winds BACK from the rim — the direction a real arm
        // trails. `max` floors the logarithm at the centre, where it diverges.
        let wind = (r / r_max).max(1.0e-6).ln() / shape.pitch_tan;
        // SCATTER, or the arm is a LINE — as unbelievable as the shell this law replaces.
        //
        // ★ TRIANGULAR, NOT FLAT. The mean of two uniforms is a triangular distribution: dense in the
        // middle, thinning toward the edges. A flat draw gives an arm a hard edge and an even density
        // across its width, which is the "drawn with a pen" look. This gives it a spine and a fade.
        //
        // ★ AND IT FRAYS OUTWARD. The width grows with radius, so arms are tight near the core and
        // loose at the rim — which is what a real arm does as it runs out of the density wave that
        // holds it together.
        let centred = 0.5 * (draws.scatter + draws.scatter_b) - 0.5;
        let fray = 1.0 + shape.arm_fray * (r / r_max);
        let scatter = centred * shape.arm_width_rad * fray;
        arm_base + wind + scatter
    };

    // THE HEIGHT. A disc is THIN: its thickness is a small fraction of its radius, and that ratio is
    // what a picture reads as "a galaxy seen edge-on" rather than "a ball".
    //
    // The bulge is ROUND, so its height scales with its own radius rather than with a disc thickness.
    let half_height = if in_bulge {
        r * shape.bulge_roundness
    } else {
        r_max * shape.disc_thickness_frac
    };
    // ★ AND A DISC HAS NO TOP OR BOTTOM EITHER (2026-08-29). This was a UNIFORM SLAB: full density
    // out to ±h, then nothing. Seen edge-on that is a bar with two straight edges.
    //
    // A real disc's vertical profile falls off exponentially from the mid-plane, so most stars are
    // near it and a few are far above. Same closed form as the radius, mirrored about zero: the sign
    // comes from which half of the draw we are in, and the magnitude from the exponential.
    let h_u = draws.height.clamp(0.0, 1.0 - 1.0e-12);
    let side = if h_u < 0.5 { -1.0 } else { 1.0 };
    // Fold the draw into [0,1) so both halves get the full profile rather than half of it.
    let folded = (h_u * 2.0 - if h_u < 0.5 { 0.0 } else { 1.0 }).clamp(0.0, 1.0 - 1.0e-12);
    let z_soft = -half_height * (1.0 - folded).ln() * DISC_HEIGHT_SCALE_FRAC;
    // Squeezed the same way, and bounded by the star's OWN radius rather than the disc's thickness:
    // an unbounded height would drive `planar` below zero and collapse the star onto the axis.
    let z = side * r * (z_soft / r.max(1.0)).tanh();

    let planar = (r * r - z * z).max(0.0).sqrt();
    DVec3::new(planar * azimuth.cos(), z, planar * azimuth.sin()) * anchored
}

/// The seven uniforms one system's placement consumes, named so a reader can see WHICH draw does what.
///
/// Two of these existed before (a direction on a sphere). The other four are what a SHAPE needs: a
/// population, BOTH halves of an arm scatter, and a height. Uniform on a sphere is a BALL — as many
/// stars above the disc as in it, and no arms, by construction.
#[derive(Clone, Copy, Debug)]
pub(crate) struct PlacementDraws {
    /// Bulge or disc.
    pub(crate) population: f64,
    /// Where between the centre and the rim.
    pub(crate) radius: f64,
    /// The radius draw's SECOND half.
    ///
    /// ★ A DISC NEEDS TWO (2026-08-29, and the first attempt got this wrong). A real disc's surface
    /// density falls off exponentially, so the number of stars in a ring at radius `r` goes as
    /// `r · exp(-r/h)` — few at the very centre, where a ring has almost no area, rising to a peak
    /// and then fading. That is a Gamma(2) distribution, and it is drawn in closed form from TWO
    /// uniforms: `r = -h·(ln u1 + ln u2)`.
    ///
    /// Drawing from ONE gave `exp(-r/h)`, which is the density of a LINE, not a disc — its maximum is
    /// at the centre. MEASURED over 20 000 draws, that put 7 844 stars in the innermost eighth where a
    /// real disc puts 1 776, and it showed up as a pinpoint knot at the galactic centre with a sharp
    /// edge, which the owner spotted immediately.
    pub(crate) radius_b: f64,
    /// Which arm, and where around the centre.
    pub(crate) azimuth: f64,
    /// How far off the arm's own line.
    pub(crate) scatter: f64,
    /// The scatter's second half — two uniforms averaged give a TRIANGULAR spread, so an arm has a
    /// dense spine and fading edges instead of a flat band with a hard edge.
    pub(crate) scatter_b: f64,
    /// How far above or below the disc plane.
    pub(crate) height: f64,
}

// ---- THE FAMILY, AND THE NUMBERS IT SPANS (S12/G1; owner-chosen 2026-08-28) --------------------
//
// ★ A GALAXY'S NUMBERS ARE NOT INDEPENDENT, AND DRAWING THEM AS IF THEY WERE MAKES MONSTERS.
// Real galaxies sit on a ONE-DIMENSIONAL family, known since Hubble 1926 and measured many times
// since. Run along it and the arms loosen while the central bulge shrinks, together:
//
//     EARLY  tight arms (~10°)   big bulge (~40% of the light)
//       │
//       │   a galaxy sits SOMEWHERE on this line — the stage draw says where
//       ▼
//     LATE   loose arms (~35°)   small bulge (~5%)
//
// Drawing the pitch and the bulge separately would eventually seed a galaxy with tightly wound arms
// and no bulge at all. No such galaxy is in the sky, and the owner asked for "ideally very faithful".
// So ONE draw places the galaxy on the family and the correlated numbers are DERIVED from it; each
// then takes its own small wobble, so two galaxies at the same stage are still not twins.
//
// Every endpoint below is a published measurement, cited, not a number anyone liked the look of.

/// The arms' pitch angle at the EARLY end of the family, in radians (~10°) — a tightly wound Sa.
/// Kennicutt 1981 (AJ 86:1847) measures pitch against Hubble stage over 113 spirals; the early end
/// sits near 10° and the late end near 35°, which is the span these two constants state.
const PITCH_EARLY_RAD: f64 = 0.174_532_925_199_432_96; // 10°
/// The arms' pitch angle at the LATE end of the family (~35°) — a loosely wound Sc/Sd.
const PITCH_LATE_RAD: f64 = 0.610_865_238_198_015_1; // 35°
/// The share of a galaxy's stars in its central bulge at the EARLY end. Graham & Worley 2008
/// (MNRAS 388:1708) and Weinzirl et al. 2009 measure bulge-to-total light falling from ~0.4 at Sa
/// to a few percent by Sd; these two constants state that span.
const BULGE_SHARE_EARLY: f64 = 0.40;
/// The share of a galaxy's stars in its central bulge at the LATE end.
const BULGE_SHARE_LATE: f64 = 0.05;
/// How far each derived number may wobble off the family line, as a fraction of its own span. Real
/// galaxies scatter about the sequence rather than sitting exactly on it — without this every galaxy
/// at one stage would be identical, which is its own kind of unbelievable.
const FAMILY_SCATTER: f64 = 0.18;

/// A disc's thickness as a fraction of its radius, at its thinnest and thickest. The Milky Way's
/// thin disc is ~300 pc of scale height against a ~15 kpc radius — about 0.02 — and the span here
/// brackets that with room for the thicker discs late types show.
const DISC_THICKNESS_FRAC_LO: f64 = 0.010;
/// A disc's thickness as a fraction of its radius, at its thickest.
const DISC_THICKNESS_FRAC_HI: f64 = 0.060;
/// The bulge's own radius as a fraction of the galaxy's, at its smallest and largest.
const BULGE_RADIUS_FRAC_LO: f64 = 0.10;
/// The bulge's own radius as a fraction of the galaxy's, at its largest.
const BULGE_RADIUS_FRAC_HI: f64 = 0.25;
/// How much of the SPACE BETWEEN ARMS one arm fills at the rim, at its narrowest and widest.
///
/// ★ A FRACTION OF THE SPACING, NEVER AN ABSOLUTE ANGLE — and this was FOUND BY LOOKING, which is
/// exactly what ruling G13 exists for. The width was a plain angle, and the space between arms is not:
/// two arms sit 180° apart, four sit 90° apart. At the wide end of the range an arm reached 137° and
/// so filled 152% of a four-armed galaxy's gap — every arm overlapping its neighbours into one smooth
/// disc. The census said "5.36× more stars in arms than between them" and was TRUE; the picture had no
/// arms at all, because a population can be assigned to an arm that is too wide to see.
///
/// Stated as a duty cycle, an arm covers the same share of its own gap however many arms there are.
const ARM_DUTY_LO: f64 = 0.14;
/// How much of the space between arms one arm fills at the rim, at its widest.
const ARM_DUTY_HI: f64 = 0.34;
/// The fewest arms a spiral is drawn with, and the most. Two is the grand-design case and by far
/// the most common; four is the upper end before a galaxy reads as flocculent rather than armed.
const ARMS_LO: u32 = 2;
/// The most arms a spiral is drawn with.
const ARMS_HI: u32 = 4;

/// Read one uniform and place it in `[lo, hi]`.
fn spread(u01: f64, lo: f64, hi: f64) -> f64 {
    lo + (hi - lo) * u01
}

/// Take a number DERIVED from the family and let it wobble off the line, staying inside the span.
///
/// The wobble is a fraction of the span, centred, and the result is clamped back into `[lo, hi]` so
/// a galaxy at an end of the family cannot be pushed off it.
fn wobble(value: f64, u01: f64, lo: f64, hi: f64) -> f64 {
    let span = hi - lo;
    let off = (u01 - 0.5) * 2.0 * FAMILY_SCATTER * span;
    (value + off).clamp(lo.min(hi), hi.max(lo))
}

/// DRAW a galaxy's shape — every number of it, from the galaxy's own stream (G1).
///
/// The draws are taken in a FIXED ORDER and every one is consumed for every kind, so the stream's
/// shape does not depend on what kind was drawn. A kind that ignores a number still costs its draw.
/// This is the same discipline the home system's placement follows: consume identically, branch on
/// the value, never on the stream.
fn draw_galaxy_shape(kind: GalaxyType, stream: &mut vd_core::rng::SplitMix64) -> GalaxyShape {
    // ---- THE STAGE: where on the family this galaxy sits. 0 is early, 1 is late. ----
    let stage = stream.next_f64();
    let u_pitch = stream.next_f64();
    let u_bulge_share = stream.next_f64();
    let u_bulge_radius = stream.next_f64();
    let u_thickness = stream.next_f64();
    let u_arms = stream.next_f64();
    let u_arm_width = stream.next_f64();
    let u_arm_fray = stream.next_f64();
    let u_interarm = stream.next_f64();
    let u_falloff = stream.next_f64();

    // THE TWO CORRELATED NUMBERS, derived from the stage and then wobbled off the line.
    let pitch_rad = wobble(
        spread(stage, PITCH_EARLY_RAD, PITCH_LATE_RAD),
        u_pitch,
        PITCH_EARLY_RAD,
        PITCH_LATE_RAD,
    );
    // The bulge share runs the OTHER WAY along the family: an early galaxy has the big bulge.
    let bulge_share = wobble(
        spread(stage, BULGE_SHARE_EARLY, BULGE_SHARE_LATE),
        u_bulge_share,
        BULGE_SHARE_LATE,
        BULGE_SHARE_EARLY,
    );
    // The rest are drawn on their own: they scatter across the sequence rather than tracking it.
    let bulge_radius_frac = spread(u_bulge_radius, BULGE_RADIUS_FRAC_LO, BULGE_RADIUS_FRAC_HI);
    let disc_thickness_frac = spread(u_thickness, DISC_THICKNESS_FRAC_LO, DISC_THICKNESS_FRAC_HI);
    let arms =
        ARMS_LO + ((u_arms * f64::from(ARMS_HI - ARMS_LO + 1)) as u32).min(ARMS_HI - ARMS_LO);
    let arm_fray = spread(u_arm_fray, 1.0, 2.2);
    // THE WIDTH IS A SHARE OF THE GAP (see `ARM_DUTY_LO`). The placement widens an arm by
    // `1 + fray·(r/R)`, reaching `1 + fray` at the rim, so dividing by that here makes the duty
    // cycle true where the arms are widest — and tighter than it everywhere inside.
    let spacing_rad = TAU / f64::from(arms);
    let arm_width_rad =
        spread(u_arm_width, ARM_DUTY_LO, ARM_DUTY_HI) * spacing_rad / (1.0 + arm_fray);
    let interarm_fraction = spread(u_interarm, 0.20, 0.38);
    let disc_exponent = spread(u_falloff, 0.50, 0.75);

    let spiral = GalaxyShape {
        arms,
        pitch_tan: pitch_rad.tan(),
        arm_width_rad,
        arm_fray,
        interarm_fraction,
        bulge_fraction: bulge_share,
        bulge_radius_frac,
        bulge_roundness: 0.6,
        bulge_exponent: 0.6,
        disc_exponent,
        disc_thickness_frac,
    };
    // AN ELLIPTICAL is one round population: no arms, no disc, denser toward the middle. Its own
    // stage draw sets how FLATTENED it is — Hubble's E0 (round) through E7 (lens-shaped).
    let elliptical = GalaxyShape {
        arms: 1,
        pitch_tan: 1.0,
        arm_width_rad: TAU,
        arm_fray: 0.0,
        interarm_fraction: 0.0,
        bulge_fraction: 1.0,
        bulge_radius_frac: 1.0,
        bulge_roundness: spread(stage, 0.30, 0.95),
        bulge_exponent: disc_exponent,
        disc_exponent,
        disc_thickness_frac: 0.5,
    };
    // AN IRREGULAR is neither: thick, loosely wound, mostly unstructured. That is what makes it
    // irregular, so it takes the loose end of every span rather than the family's line.
    let irregular = GalaxyShape {
        arms: 1,
        pitch_tan: spread(stage, 1.0, 2.0),
        arm_width_rad: arm_width_rad + 1.5,
        arm_fray: arm_fray + 1.0,
        interarm_fraction: interarm_fraction + 0.15,
        bulge_fraction: bulge_share,
        bulge_radius_frac: bulge_radius_frac * 2.0,
        bulge_roundness: 0.8,
        bulge_exponent: 0.5,
        disc_exponent,
        disc_thickness_frac: disc_thickness_frac * 5.0,
    };
    match kind {
        GalaxyType::Spiral => spiral,
        GalaxyType::Elliptical => elliptical,
        GalaxyType::Irregular => irregular,
    }
}

/// WHAT A GALAXY LOOKS LIKE — every number of it DRAWN FROM THE SEED, never a free literal (G1).
#[derive(Clone, Copy, Debug)]
pub(crate) struct GalaxyShape {
    pub(crate) arms: u32,
    /// tan(pitch angle) — how tightly the arms wind. A small value winds tightly.
    pub(crate) pitch_tan: f64,
    /// How far off its own line an arm scatters, in radians, at the CORE. The placement widens this
    /// outward by the fray. Derived from a duty cycle so it stays a share of the gap between arms —
    /// see `ARM_DUTY_LO` for what an absolute angle cost.
    pub(crate) arm_width_rad: f64,
    /// How much wider an arm is at the rim than at the core. Arms fray as they run out.
    pub(crate) arm_fray: f64,
    /// The share of the DISC that lies between the arms rather than in one. Without it a spiral looks
    /// painted: real arms are where stars crowd, not where they exclusively live.
    pub(crate) interarm_fraction: f64,
    pub(crate) bulge_fraction: f64,
    pub(crate) bulge_radius_frac: f64,
    pub(crate) bulge_roundness: f64,
    pub(crate) bulge_exponent: f64,
    pub(crate) disc_exponent: f64,
    pub(crate) disc_thickness_frac: f64,
}

// (THE CHOSEN TABLE IS DELETED, S12/G1 2026-08-28. `galaxy_shape` returned eleven fixed numbers per
// kind — a spiral was always two arms at 25° with a 15% bulge, whoever's galaxy it was. Ruling G1 says
// every number the placement reads must come from the seed, so the table became a SECOND SOURCE OF
// TRUTH for a fact the stream now states. `draw_galaxy_shape` is the only source. The owner judged the
// drawn shapes against the picture before this went, per ruling G13.)

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
/// ★ THE LAST WORLD THIS PROCESS BUILT, kept so it is not built again (2026-09-01).
///
/// **MEASURED, and this is why it exists.** A shard's boot ran 46.6 seconds in a debug build, and
/// about 30 of those seconds were THE SAME FOREST built four more times: the world build, the reach
/// roster, and three world fences each walked all 233,220 star systems from scratch. Every realm is a
/// process, so every ship a player spins up paid it — a hull that appears beside you after a
/// 46-second wait is a seam, and a seam is a defect.
///
/// **IT IS A PURE FUNCTION, so remembering its answer changes nothing.** The same seed and the same
/// config give the same forest, in every process and at every boot; that is the property the whole
/// world rests on. A cache of a pure function is not state — it is the same answer, not recomputed.
///
/// ONE entry, not a map. A process boots ONE world and asks about it repeatedly, so a single slot hits
/// every time after the first. A map would hold every world a test ever built and never free one.
static LAST_FOREST: std::sync::Mutex<Option<(u64, UniverseConfig, std::sync::Arc<Vec<GeneratedBody>>)>> =
    std::sync::Mutex::new(None);

/// The forest for this world, built once per process.
///
/// The lock is held only to look and to store — never across the build — so two threads asking at once
/// both build rather than one blocking the other. Building twice is wasteful; holding a lock across a
/// 4-second build is worse, because it turns a slow boot into a stalled one.
pub(crate) fn system_forest_cached(
    seed_universe: u64,
    config: &UniverseConfig,
) -> std::sync::Arc<Vec<GeneratedBody>> {
    if let Ok(slot) = LAST_FOREST.lock()
        && let Some((seed, cfg, forest)) = slot.as_ref()
        && *seed == seed_universe
        && cfg == config
    {
        return std::sync::Arc::clone(forest);
    }
    let built = std::sync::Arc::new(generate_system_forest(seed_universe, config));
    if let Ok(mut slot) = LAST_FOREST.lock() {
        *slot = Some((seed_universe, *config, std::sync::Arc::clone(&built)));
    }
    built
}

pub(crate) fn generate_system_forest(
    seed_universe: u64,
    config: &UniverseConfig,
) -> Vec<GeneratedBody> {
    // (`st` and `pl` moved into `seed_one_system` and `append_system_contents` with the stages that
    // read them — the extraction left this loop reading neither.)
    let sc = &config.scale;
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
    // ★ THE GALAXY STATES ITSELF — its census, its kind AND its shape, read ONCE from its own stream
    // (S12, owner ruling G1: *"Density also should come from seed, otherwise any tiny change might
    // change positions."*). This was three separate reads, each rebuilding the same stream; a shape
    // that a person had chosen is now a shape the seed draws, and the owner judged the picture before
    // the world adopted it (ruling G13, approved 2026-08-28).
    let GalaxyProfile {
        count: n_systems,
        kind: _kind,
        shape,
        density_per_pc3: _density,
    } = galaxy_profile(seed_universe, config);
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
        let SeededSystem {
            mut stream,
            star,
            mut element_draws,
            mut albedo_draws,
        } = seed_one_system(
            seed_universe,
            config,
            &shape,
            s,
            seed,
            &mut bodies[system_ix],
        );
        append_system_contents(
            &mut bodies,
            config,
            seed_universe,
            seed,
            system,
            system_ix,
            &mut stream,
            &star,
            &mut element_draws,
            &mut albedo_draws,
        );
    }
    // ★ THE PUSH (owner ruling G12) — where the shape and the gap disagree, the star moves OUT.
    // Runs INSIDE the generator, so no caller can hold an un-pushed galaxy: two processes with
    // different star positions is the very ambiguity the separation fence exists to prevent.
    push_crowded_systems(&mut bodies, config);
    // The fixture plant (look_horizon slice 5 — G-IDENTICAL), appended LAST: with `None` (every
    // shipped constructor) this is a no-op and the forest is byte-identical to the pre-plant world.
    append_fixture_plant(&mut bodies, config);
    bodies
}

/// How far one attempt moves a crowded star, as a share of its own distance from the centre. Small,
/// because the shortfalls are small: THE world's worst pair sits at 4.7% of the separation it needs,
/// which one step of this size clears with room to spare.
const PUSH_STEP_FRAC: f64 = 0.02;
/// How many times one star may step outward before the generator gives up and says so. A star that
/// cannot be placed in this many tries is a world that has gone wrong somewhere else, and a silent
/// infinite loop is the worst way to learn that.
const PUSH_MAX_ATTEMPTS: u32 = 64;

/// ★ ONE SYSTEM'S CONTENTS — its planets, its star child and its moons (owner ruling 2026-08-29).
///
/// ★ WHY IT IS ITS OWN FUNCTION. A shard runs ONE realm and needs the children it authors. It used
/// to get them by building the WHOLE forest and filtering: MEASURED on THE world, 3 500 479 regions
/// built to keep 13, which is the login timeout a player actually hits.
///
/// A realm AUTHORS its own children — that is the parent's job, and this realm is their parent — so
/// asking the seed "what is inside me" involves no other realm and crosses no boundary. What a realm
/// may NOT do is author its own PLACEMENT; that stays with the galaxy, and this function never
/// touches it.
///
/// ★ AND IT IS NOT A SECOND GENERATOR. This is the full forest's own loop body, lifted verbatim, and
/// the full forest calls it. There is exactly one piece of code that decides what a star system
/// contains, so a shard and the gateway cannot disagree about how many planets exist. The identity
/// gate beside it measures that rather than assuming it.
///
/// The system must ALREADY be seeded — [`seed_one_system`] read its stream and left it positioned at
/// the first appended draw, and that stream is handed in rather than re-made.
#[allow(clippy::too_many_arguments)] // one private stage; every argument is the system's own datum
fn append_system_contents(
    bodies: &mut Vec<GeneratedBody>,
    config: &UniverseConfig,
    seed_universe: u64,
    seed: u64,
    system: RealmId,
    system_ix: usize,
    stream: &mut vd_core::rng::SplitMix64,
    star: &StarPhotometrics,
    element_draws: &mut Vec<PlanetElementDraws>,
    albedo_draws: &mut Vec<f64>,
) {
    let pl = &config.planet;
    let star = *star;
    let shell = |r: f64| Boundary::Shell { r };

    let legacy = pl.n_planets.min(LEGACY_STREAM_PLANETS);
    // ---- the APPENDED draws (planets beyond the frozen prefix, then every mass) ----
    for _ in legacy..pl.n_planets {
        element_draws.push(planet_element_draws(config, stream));
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
    let frost_line_au = crate::taxonomy::frost_line_radius_au(star.luma_lsun, pl.frost_coeff_au);
    let th = pl.frost_thresholds();
    let mut planet_bodies = Vec::with_capacity(element_draws.len());
    let mut moon_inputs: Vec<(RealmId, u64, f64, f64, f64, f64)> = Vec::new();
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
            // ★ THE PLANET'S OWN SHELL, CARRIED (perf fix 2026-08-28). The moon pass used to
            // SEARCH the whole accumulated body list for this planet to read exactly this
            // number back off it. See `append_moons`.
            soi_m,
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
    for (
        planet_realm,
        planet_seed,
        planet_sma_m,
        planet_mass_kg,
        planet_albedo_u01,
        planet_soi_m,
    ) in moon_inputs
    {
        append_moons(
            bodies,
            config,
            seed_universe,
            seed,
            &star,
            planet_realm,
            planet_seed,
            planet_sma_m,
            planet_mass_kg,
            planet_albedo_u01,
            planet_soi_m,
        );
    }
}

/// ★ ONE REALM'S OWN SUBTREE — its chain, itself, and the children it authors (owner ruling
/// 2026-08-29). Nothing else in the galaxy is built.
///
/// A shard runs ONE realm. It used to get its 13 rows by building the whole forest and filtering:
/// MEASURED on THE world, 3 500 479 bodies to keep 13, which is the login timeout a player hits.
///
/// ★ WHAT COMES FROM WHERE, AND WHY IT BREAKS NO LAW.
/// - the CHAIN and this realm's own PLACEMENT come from the galaxy's own system layer. A realm never
///   authors its own place; its parent does, and the layer IS the parent's answer, crowding push
///   included.
/// - the CONTENTS come from [`append_system_contents`], the full forest's own stage, called for this
///   system alone. A parent authors its children, and this realm is their parent, so nothing crosses
///   a boundary and no message is needed where the seed already answers.
///
/// ★ ONE GENERATOR, NOT TWO. Both stages are the exact functions the full forest calls, reading the
/// same streams in the same order. The identity gate beside this measures that a subtree matches the
/// full build row for row, rather than assuming it.
/// Is this body a DIRECT child of a realm the shard holds? Monomorphic and named, so its two arms are
/// counted once (HR5 (a)) instead of inside a closure in a generic iterator chain.
fn child_of_held(b: &GeneratedBody, held: &std::collections::BTreeSet<RealmId>) -> bool {
    match b.parent {
        Some(parent) => held.contains(&parent),
        None => false,
    }
}

#[must_use]
pub(crate) fn realm_subtree(
    seed_universe: u64,
    config: &UniverseConfig,
    held: &std::collections::BTreeSet<RealmId>,
    lineage: &std::collections::BTreeSet<RealmId>,
) -> Vec<GeneratedBody> {
    // ★ THE LINEAGE IS WHAT LETS A DEEP SHARD EXIST AT ALL (owner ruling 2026-08-30).
    //
    // A shard below a star system cannot find itself from its own name. A planet's identifier is a
    // one-way hash of its system's, and a player's apartment is not in the seed at all — the
    // generator emits NO station and NO area, so no amount of generating will ever produce one.
    //
    // The parent already tells it: a spawn demand carries a `RealmCoord`, which names every ancestor
    // by kind and seed, and the orchestrator mints the whole chain in one sweep. This reads what was
    // already sent, so the star system that holds a held planet gets its contents built and the
    // planet appears in the world it is the centre of.
    //
    // MEASURED DEFECT IT CURES: a planet-hosting shard booted with an EMPTY forest and refused —
    // "the forest has 0 ambient roots" — because a subtree stops one level below a star system.
    let named = || held.iter().chain(lineage.iter()).copied();
    let layer = generate_system_layer(seed_universe, config);
    // The chain: the ambient root, the galaxy, and every held realm — placed by their parent.
    //
    // ★ AND THE DIRECT CHILDREN OF EVERY HELD REALM, which is not an extra: SL1 makes a parent the
    // ONLY writer of its children's placements, so a shard that cannot see its own children cannot do
    // the one job its realm has. The clause is stated as PARENTHOOD and tests no realm kind (SL4/HR3)
    // — for a held star system the layer holds no children at all (they arrive from the contents stage
    // below), so this adds exactly nothing there and the system shard's rows stay byte-identical.
    //
    // MEASURED DEFECT IT CURES (2026-08-29): with only the three clauses above, a GALAXY-hosting shard
    // booted with zero systems. `world_roster` panicked on THE world — "the galaxy authors the home
    // system's placement" — because the galaxy's own child list was empty. Live, that same shard would
    // author no placement, state no marker and hold no area of interest for any star.
    let mut bodies: Vec<GeneratedBody> = layer
        .iter()
        .filter(|b| {
            b.parent.is_none()
                || b.realm == GALAXY
                || held.contains(&b.realm)
                || lineage.contains(&b.realm)
                || child_of_held(b, held)
        })
        .copied()
        .collect();
    // The contents: for each star system this shard HOLDS or its lineage NAMES, that system's own
    // children, through the shared stage. Naming a system in the lineage is what puts a held planet
    // (and a held city, once players build them) into the forest at all.
    for hosted in named().collect::<std::collections::BTreeSet<_>>() {
        let RealmId::System(seed) = hosted else {
            continue; // only a star system has seed-generated contents today
        };
        let Some(system_ix) = bodies.iter().position(|b| b.realm == hosted) else {
            continue; // a held realm the layer does not name is not ours to populate
        };
        let Some(s) = (0..layer.len() as u32).find(|n| system_seed_at(*n) == seed) else {
            continue;
        };
        // Re-read this system's own stream to the point the contents begin — the SAME function the
        // full forest uses, so the draws are identical.
        let mut scratch = bodies[system_ix];
        let SeededSystem {
            mut stream,
            star,
            mut element_draws,
            mut albedo_draws,
        } = seed_one_system(
            seed_universe,
            config,
            &galaxy_profile(seed_universe, config).shape,
            s,
            seed,
            &mut scratch,
        );
        append_system_contents(
            &mut bodies,
            config,
            seed_universe,
            seed,
            hosted,
            system_ix,
            &mut stream,
            &star,
            &mut element_draws,
            &mut albedo_draws,
        );
    }
    bodies
}

/// ★ WHERE THE SHAPE AND THE GAP DISAGREE, PUSH — NEVER DROP (owner ruling G12, 2026-08-29).
///
/// A faithful bulge asks for density; the per-pair separation fence sets a floor. Where they meet,
/// one star must yield. The owner ruled which way: *"push the star out. Do not drop it."* Dropping
/// thins the bulge exactly where the shape is trying to be densest — a galaxy with a suspiciously
/// hollow middle, the shape defeated by its own rule.
///
/// ★ MEASURED ON THE WORLD BEFORE THIS EXISTED (2026-08-29). Seed 2298 — the HOME seed, the world the
/// dev cluster boots — holds 233 220 systems and SEVEN overlapping pairs, the worst at 4.7% of the
/// separation it needs, 3.3% of the way out from the centre. All of them in the bulge. Seed 0 has
/// none. The ruling predicted exactly this: its own warning says the density contrast is seed-derived
/// so "a galaxy MAY draw one above 550x", and that arms concentrate stars further still.
///
/// ★ THE LATER STAR YIELDS, AND THAT IS WHAT KEEPS RULING G4. A star's final place depends only on
/// stars placed BEFORE it, so growing a galaxy adds later stars that can never disturb earlier ones.
/// Nothing already placed moves, ever — which is the property the whole placement law exists to have.
///
/// ★ THE DISTANCE COMES FROM THE STAR'S OWN STREAM, so the push is seed-derived like everything else
/// (G1): same input, same answer, forever, on every process, with nothing exchanged. It also means no
/// two pushed stars land at the same distance — pushing each one to "just barely clear" would rebuild
/// the shell this slice deleted, and ruling G3 forbids equal distances by name.
///
/// The star keeps its DIRECTION and only its radius grows, because that is what preserves the shape:
/// a bulge pushed outward is still a bulge, very slightly less dense at its centre. A star nudged
/// sideways would still be in the crowd.
fn push_crowded_systems(bodies: &mut [GeneratedBody], _config: &UniverseConfig) {
    let ix: Vec<usize> = bodies
        .iter()
        .enumerate()
        .filter(|(_, b)| b.parent == Some(GALAXY) && matches!(b.realm, RealmId::System(_)))
        .map(|(i, _)| i)
        .collect();
    // ★ A GRID, NOT A SCAN — AND I WROTE THE SCAN FIRST (2026-08-29). The first version of this loop
    // checked every already-placed star for every star. MEASURED: it turned a 0.82 s galaxy into a
    // 22.11 s one, 27x slower, for seven pushes. That is the FIFTH loop of this exact shape found in
    // one day, and the only one I authored rather than inherited — which says the shape is easy to
    // write, not that the earlier authors were careless.
    //
    // Two stars can only be too close if they are within twice the largest reach, so a cell of that
    // width puts every possible partner in the caller's own cell or one of the twenty-six touching
    // it. The same argument the separation fence uses.
    let cell = 2.0
        * ix.iter()
            .map(|&i| bodies[i].shape.circumscribed_extent())
            .fold(0.0_f64, f64::max);
    let key = |v: DVec3| {
        (
            (v.x / cell).floor() as i64,
            (v.y / cell).floor() as i64,
            (v.z / cell).floor() as i64,
        )
    };
    // Placed so far, as (centre, reach) — a star is judged only against stars before it.
    let mut placed: Vec<(DVec3, f64)> = Vec::with_capacity(ix.len());
    let mut grid: BTreeMap<(i64, i64, i64), Vec<usize>> = BTreeMap::new();
    let crowded = |at: DVec3,
                   reach: f64,
                   grid: &BTreeMap<(i64, i64, i64), Vec<usize>>,
                   placed: &[(DVec3, f64)]| {
        let (cx, cy, cz) = key(at);
        for dx in -1..=1_i64 {
            for dy in -1..=1_i64 {
                for dz in -1..=1_i64 {
                    if let Some(bucket) = grid.get(&(cx + dx, cy + dy, cz + dz)) {
                        for &j in bucket {
                            let (p, r) = placed[j];
                            if (p - at).length() < r + reach {
                                return true;
                            }
                        }
                    }
                }
            }
        }
        false
    };
    for (n, &i) in ix.iter().enumerate() {
        let reach = bodies[i].shape.circumscribed_extent();
        let mut at = super::placement_offset(bodies[i].placement);
        // The star's own push draw: one number, from its own stream, so the distance it moves is
        // this star's and no other's.
        let mut stream = realm_stream(
            0,
            &[
                UNIVERSE_SEED,
                GALAXY_SEED,
                system_seed_at(n as u32),
                PUSH_SALT,
            ],
        );
        let jitter = stream.next_f64();
        let mut attempt = 0_u32;
        while attempt < PUSH_MAX_ATTEMPTS && crowded(at, reach, &grid, &placed) {
            attempt += 1;
            let grow = 1.0 + PUSH_STEP_FRAC * (f64::from(attempt) + jitter);
            at = on_galaxy_cell(super::placement_offset(bodies[i].placement) * grow);
        }
        grid.entry(key(at)).or_default().push(placed.len());
        placed.push((at, reach));
        bodies[i].placement = Placement::StaticOffset(at);
    }
}

/// ★ A PUSHED STAR LANDS ON THE GALAXY'S OWN GRID (2026-08-31).
///
/// The push scales a star's offset by a float, and a float multiply does not land on a cell edge.
/// A galaxy counts in whole cells, and a star catalogue row carries the CELL and no sub-cell part —
/// so an off-cell star is drawn up to half a cell from where the world placed it, silently, on every
/// client.
///
/// MEASURED, by the test that exists to catch exactly this: a pushed system sat 1.0 m off-cell, and
/// the row would have dropped it. Its own words: *"If a future generator starts placing a system
/// off-cell, THIS goes red rather than the position silently losing its residual on the way to every
/// client."* It went red. The defect was mine, introduced with the push.
///
/// Rounding is HALF-AWAY-FROM-ZERO and per component, so the snap is symmetric about the galactic
/// centre and cannot bias a whole galaxy one way.
pub(crate) fn on_galaxy_cell(v: DVec3) -> DVec3 {
    let edge = vd_core::pose::Tier::Galaxy.cell_edge_m();
    DVec3::new(
        (v.x / edge).round() * edge,
        (v.y / edge).round() * edge,
        (v.z / edge).round() * edge,
    )
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
    // The planet's own solved shell — its caller has it (`shape: shell(soi_m)`) and passes it,
    // where this used to search the whole body list for it. See the note at the first use.
    planet_soi_m: f64,
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
    // ★ THE PLANET'S SHELL IS PASSED IN, NOT SEARCHED FOR (perf fix 2026-08-28). This read
    //
    //     bodies.iter().find(|b| b.realm == planet_realm).shape.finite_extent()
    //
    // — a scan of EVERY body generated so far, run once per planet, to recover a number the caller
    // had already computed and pushed one line earlier (`shape: shell(soi_m)`).
    //
    // THAT MADE GENERATION QUADRATIC, and at three systems nobody could see it. MEASURED at
    // 1k/2k/4k/8k systems: 39.5 ms, 161.9 ms, 696.5 ms, 3654.2 ms — n^2.04, n^2.11, n^2.39. At 8000
    // systems the list holds 119 749 bodies and the scan runs 72 000 times, about 4.3 billion
    // comparisons. Extrapolated to the world's ~150 000 target it is hours, not seconds.
    //
    // Found while measuring what ruling G8 would cost, which is the only reason it surfaced: a
    // three-system world never pays a quadratic.
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
