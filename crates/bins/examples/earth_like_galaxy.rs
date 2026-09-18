//! U1, the galaxy half — WHERE IS THE EARTH-LIKE BODY OF THE WORLD? (the landform arc, ruling V13
//! L27). The home system holds none (`home_body_facts`), so this sweeps the whole galaxy at the
//! home seed through the census's own predicate and prints every earth-like body with the ladder's
//! verdict on it. It folds the galaxy forest — twenty minutes of one core (MEASURED 2026-09-09) — so
//! run it detached and read its log.
//!
//! ```text
//! cargo run --release -p vd-bins --example earth_like_galaxy
//! ```

use std::process::ExitCode;

use vd_bins::DEV;
use vd_core::pose::RealmId;
use vd_physics::celestial::G;
use vd_physics::taxonomy::M_EARTH_KG;
use vd_physics::worldgen::{UniverseConfig, earth_like_candidates};

fn main() -> ExitCode {
    let config = UniverseConfig::world(DEV.move_speed, DEV.tick_dt);
    let start = std::time::Instant::now();
    let found = earth_like_candidates(DEV.universe_seed, &config);
    println!(
        "earth_like_galaxy: {} earth-like bodies in THE world at seed {} ({:.0} s)",
        found.len(),
        DEV.universe_seed,
        start.elapsed().as_secs_f64()
    );
    for c in &found {
        let seed = match c.body {
            RealmId::Planet(s) => Some(s),
            _ => None,
        };
        // ★ THE RELIEF LAW's TWO WORDS (slice 8b stage 3), through the census's own door. A
        // galaxy-wide walk cannot build a subtree per candidate, so it reads the pair from the
        // body's own mass and radius — the SAME expression the charter a realm states is built
        // from (`vd_physics::worldgen::relief_words`).
        let [g_word, rho_word] = vd_physics::worldgen::relief_words(c.mass_kg, c.radius_m);
        let facts = g_word
            .zip(rho_word)
            .map(|(g, rho)| vd_terrain::BodyFacts::new(g, rho));
        let ladder = seed
            .zip(facts)
            .and_then(|(s, f)| vd_terrain::BodyDefinition::from_seed(s, c.radius_m, f));
        println!(
            "  {:?} in {:?}{}: star {:?} {:.4} Msun {:.4} Lsun; radius {:.1} km, mass {:.3} Mearth, density {:.0} kg/m3, g {:.2} m/s2, flux {:.3}, T_eq {:.1} K, air {}, {} planets {} moons in the system; ladder {}",
            c.body,
            c.system,
            if c.system == RealmId::System(vd_bins::HOME_SYSTEM) {
                " (THE HOME SYSTEM)"
            } else {
                ""
            },
            c.star_class,
            c.star_mass_msun,
            c.star_luma_lsun,
            c.radius_m / 1000.0,
            c.mass_kg / M_EARTH_KG,
            c.mass_kg / (4.0 / 3.0 * std::f64::consts::PI * c.radius_m.powi(3)),
            G * c.mass_kg / (c.radius_m * c.radius_m),
            c.insolation_rel,
            c.t_eq_k,
            c.has_atmosphere,
            c.system_planets,
            c.system_moons,
            match ladder {
                Some(b) => format!(
                    "ACCEPTS ({} rungs, radius {:.1} km)",
                    b.ladder().rungs,
                    b.ladder().radius_m() / 1000.0
                ),
                None => "REFUSES".to_owned(),
            }
        );
    }
    ExitCode::SUCCESS
}
