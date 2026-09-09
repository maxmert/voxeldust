//! U1 — IS THE HOME PLANET THE RIGHT BODY? (the landform arc, ruling V13 L27).
//!
//! The voxel home planet is the first body of System 7 the ladder accepted, not a body chosen for
//! being earth-like. This prints its own facts from the census's row — mass, radius, density, surface
//! gravity, insolation, equilibrium temperature, class, atmosphere — with the earth-like verdict and
//! every clause of it, then every earth-like body the HOME SYSTEM holds, with whether the ladder
//! accepts it. It reads the system's subtree, never the galaxy.
//!
//! ```text
//! cargo run --release -p vd-bins --example home_body_facts
//! ```

use std::process::ExitCode;

use vd_bins::DEV;
use vd_core::pose::RealmId;
use vd_physics::celestial::G;
use vd_physics::taxonomy::{
    KOPPARAPU_FLUX_CONSERVATIVE, M_EARTH_KG, PlanetType, R_EARTH_M, SpectralClass,
};
use vd_physics::worldgen::{
    BodyFacts, EARTH_LIKE_RADIUS_BAND_REARTH, UniverseConfig, body_facts_in_subtree,
    earth_like_in_subtree, earth_like_t_bound_k,
};

fn print_facts(label: &str, realm: RealmId, f: &BodyFacts) {
    let t = &f.taxon;
    let volume = 4.0 / 3.0 * std::f64::consts::PI * t.radius_m.powi(3);
    let density = t.mass_kg / volume;
    let gravity = G * t.mass_kg / (t.radius_m * t.radius_m);
    let (s_lo, s_hi) = KOPPARAPU_FLUX_CONSERVATIVE;
    let (r_lo, r_hi) = EARTH_LIKE_RADIUS_BAND_REARTH;
    println!("{label}: {realm:?} in {:?}", f.system);
    println!(
        "  star: class {:?}, {:.4} Msun, {:.4} Lsun",
        f.star.class, f.star.mass_msun, f.star.luma_lsun
    );
    println!(
        "  mass {:.4} Mearth ({:.3e} kg), radius {:.4} Rearth ({:.1} km), density {density:.0} kg/m3, surface gravity {gravity:.2} m/s2",
        t.mass_kg / M_EARTH_KG,
        t.mass_kg,
        t.radius_m / R_EARTH_M,
        t.radius_m / 1000.0
    );
    println!(
        "  insolation {:.4} Searth, equilibrium temperature {:.1} K, Bond albedo {:.3}, class {:?}",
        t.insolation_rel, t.t_eq_k, t.bond_albedo, t.class
    );
    match t.atmosphere {
        Some(a) => println!(
            "  atmosphere: mean molecular weight {:.1}, scale height {:.0} m, reference density {:?} kg/m3",
            a.mean_molecular_weight, a.scale_height_m, a.reference_density_kgm3
        ),
        None => println!("  atmosphere: NONE (airless)"),
    }
    println!("  earth-like: {}", f.earth_like);
    println!(
        "    yellow sun (G):        {}",
        f.star.class == SpectralClass::G
    );
    println!(
        "    rocky:                 {}",
        t.class == PlanetType::Rocky
    );
    println!(
        "    radius in [{r_lo}, {r_hi}] Rearth: {}",
        (r_lo..=r_hi).contains(&(t.radius_m / R_EARTH_M))
    );
    println!(
        "    flux in [{s_lo}, {s_hi}] Searth: {}",
        (s_lo..=s_hi).contains(&t.insolation_rel)
    );
    println!(
        "    T_eq in [{:.1}, {:.1}] K:  {}",
        earth_like_t_bound_k(s_lo),
        earth_like_t_bound_k(s_hi),
        (earth_like_t_bound_k(s_lo)..=earth_like_t_bound_k(s_hi)).contains(&t.t_eq_k)
    );
    println!("    air:                   {}", t.atmosphere.is_some());
}

fn main() -> ExitCode {
    let config = UniverseConfig::world(DEV.move_speed, DEV.tick_dt);
    let Some(body) = vd_bins::home_body(DEV.universe_seed) else {
        println!("home_body_facts: REFUSED — the home system holds no planet the recipe accepts");
        return ExitCode::FAILURE;
    };
    let voxel_home = RealmId::Planet(body.seed());
    println!(
        "home_body_facts: the VOXEL home planet is {voxel_home:?}, ladder radius {:.1} km, {} rungs",
        body.ladder().radius_m() / 1000.0,
        body.ladder().rungs
    );
    // The home SYSTEM's subtree: seconds. (The galaxy-wide sweep is `vd-seedsearch`'s, and it
    // costs twenty minutes of one core — MEASURED 2026-09-09.)
    let held = std::collections::BTreeSet::from([RealmId::System(vd_bins::HOME_SYSTEM)]);
    let lineage = std::collections::BTreeSet::from([vd_core::worldgen::GALAXY]);
    match body_facts_in_subtree(DEV.universe_seed, &config, &held, &lineage, voxel_home) {
        Some(f) => print_facts("VOXEL HOME", voxel_home, &f),
        None => println!("home_body_facts: the census names no facts for the voxel home planet"),
    }
    // Every earth-like body of the home system, and whether the ladder accepts it.
    let found = earth_like_in_subtree(DEV.universe_seed, &config, &held, &lineage);
    println!(
        "home_body_facts: {} earth-like bodies in the home system at seed {}",
        found.len(),
        DEV.universe_seed
    );
    for c in &found {
        let in_home_system = c.system == RealmId::System(vd_bins::HOME_SYSTEM);
        let seed = match c.body {
            RealmId::Planet(s) => Some(s),
            _ => None,
        };
        let ladder = seed.and_then(|s| vd_terrain::BodyDefinition::from_seed(s, c.radius_m));
        println!(
            "  {:?} in {:?}{}: radius {:.1} km, mass {:.3} Mearth, gravity {:.2} m/s2, flux {:.3}, T_eq {:.1} K, air {}, ladder {}",
            c.body,
            c.system,
            if in_home_system {
                " (THE HOME SYSTEM)"
            } else {
                ""
            },
            c.radius_m / 1000.0,
            c.mass_kg / M_EARTH_KG,
            G * c.mass_kg / (c.radius_m * c.radius_m),
            c.insolation_rel,
            c.t_eq_k,
            c.has_atmosphere,
            match ladder {
                Some(b) => format!(
                    "ACCEPTS ({} rungs, radius {:.1} km)",
                    b.ladder().rungs,
                    b.ladder().radius_m() / 1000.0
                ),
                None => "REFUSES".to_owned(),
            }
        );
        if let Some(f) = body_facts_in_subtree(DEV.universe_seed, &config, &held, &lineage, c.body)
        {
            print_facts("  candidate", c.body, &f);
        }
    }
    // Every planet of the home system, earth-like or not, with the ladder's verdict — so the choice
    // of a home planet reads the whole roster, not the first row the ladder accepted.
    let (rows, _) = vd_physics::worldgen::shard_boot_world(
        DEV.universe_seed,
        &config,
        &held,
        RealmId::System(vd_bins::HOME_SYSTEM),
        &lineage,
    );
    println!("home_body_facts: the home system's planets, in the forest's order:");
    for r in rows
        .iter()
        .filter(|r| r.parent == Some(RealmId::System(vd_bins::HOME_SYSTEM)))
    {
        let RealmId::Planet(seed) = r.realm else {
            continue;
        };
        let look = match r.look {
            Some(vd_core::geometry::Boundary::Shell { r }) => r,
            _ => continue,
        };
        let ladder = vd_terrain::BodyDefinition::from_seed(seed, look);
        let facts = body_facts_in_subtree(DEV.universe_seed, &config, &held, &lineage, r.realm);
        let (mass, g, flux, t, air, class) = match &facts {
            Some(f) => (
                f.taxon.mass_kg / M_EARTH_KG,
                G * f.taxon.mass_kg / (f.taxon.radius_m * f.taxon.radius_m),
                f.taxon.insolation_rel,
                f.taxon.t_eq_k,
                f.taxon.atmosphere.is_some(),
                format!("{:?}", f.taxon.class),
            ),
            None => (
                f64::NAN,
                f64::NAN,
                f64::NAN,
                f64::NAN,
                false,
                "?".to_owned(),
            ),
        };
        println!(
            "  {:?}: look radius {:.1} km, {class}, mass {mass:.3} Mearth, g {g:.2} m/s2, flux {flux:.3}, T_eq {t:.1} K, air {air}, earth-like {}, ladder {}",
            r.realm,
            look / 1000.0,
            facts.as_ref().is_some_and(|f| f.earth_like),
            match ladder {
                Some(b) => format!(
                    "ACCEPTS ({} rungs, N {})",
                    b.ladder().rungs,
                    b.ladder().cells_per_edge(0)
                ),
                None => "REFUSES".to_owned(),
            }
        );
    }
    ExitCode::SUCCESS
}
