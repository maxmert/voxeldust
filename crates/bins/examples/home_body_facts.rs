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
        "home_body_facts: the VOXEL home planet is {voxel_home:?}, ladder radius {:.1} km, {} rungs (home system {:?})",
        body.ladder().radius_m() / 1000.0,
        body.ladder().rungs,
        vd_core::worldgen::HOME_SYSTEM
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
        let [g_word, rho_word] = vd_physics::worldgen::relief_words(c.mass_kg, c.radius_m);
        let facts = g_word
            .zip(rho_word)
            .map(|(g, rho)| vd_terrain::BodyFacts::new(g, rho));
        let ladder = seed
            .zip(facts)
            .and_then(|(s, f)| vd_terrain::BodyDefinition::from_seed(s, c.radius_m, f));
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
        let facts = body_facts_in_subtree(DEV.universe_seed, &config, &held, &lineage, r.realm);
        let ladder = facts.as_ref().and_then(|f| {
            let [g_word, rho_word] =
                vd_physics::worldgen::relief_words(f.taxon.mass_kg, f.taxon.radius_m);
            g_word.zip(rho_word).and_then(|(g, rho)| {
                vd_terrain::BodyDefinition::from_seed(
                    seed,
                    look,
                    vd_terrain::BodyFacts::new(g, rho),
                )
            })
        });
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
            "  {:?}: look radius {:.1} km (bits {:#018x}), {class}, mass {mass:.3} Mearth, g {g:.2} m/s2, flux {flux:.3}, T_eq {t:.1} K, air {air}, earth-like {}, ladder {}",
            r.realm,
            look / 1000.0,
            look.to_bits(),
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
    relief_law_table();
    water_table();
    sea_table();
    ExitCode::SUCCESS
}

/// ★ M-R — THE RELIEF CAP (the landform arc, slice 8b stage 3; `slice_8b_design.md` §6.1).
///
/// For the home planet, its moons, and the LARGEST and SMALLEST planet of the home system the
/// ladder accepts: the charter's own gravity and density, the relief law's two arms, WHICH arm
/// binds, the relief the seed draws under the law, and the relief the OLD lottery drew — with the
/// ladder's floor and band on both sides.
///
/// **The old relief is arithmetic on measured numbers, not a second draw.** The law is
/// `relief = cap × (0.5 + 0.5u)` with `u` the stream's first draw, so `u = 2·relief/cap − 1`; the
/// old law was `relief = clamp(0.004·R, 200, 12 000) × (0.5 + u)` off the SAME `u`, because the
/// draw takes exactly one number from the stream now as it did then.
fn relief_law_table() {
    let config = UniverseConfig::world(DEV.move_speed, DEV.tick_dt);
    let system = RealmId::System(vd_bins::HOME_SYSTEM);
    let lineage = std::collections::BTreeSet::from([vd_core::worldgen::GALAXY]);
    let held = std::collections::BTreeSet::from([system]);
    let (rows, _) =
        vd_physics::worldgen::shard_boot_world(DEV.universe_seed, &config, &held, system, &lineage);

    let build = |realm: RealmId,
                 look: f64,
                 held: &std::collections::BTreeSet<RealmId>,
                 lineage: &std::collections::BTreeSet<RealmId>|
     -> Option<(vd_core::look::BodyCharter, vd_terrain::BodyDefinition)> {
        let RealmId::Planet(seed) = realm else {
            return None;
        };
        let charter = vd_bins::body_charter(DEV.universe_seed, held, lineage, realm)?;
        let body =
            vd_terrain::BodyDefinition::from_seed(seed, look, vd_bins::facts_of_charter(&charter))?;
        Some((charter, body))
    };

    // The home system's planets the ladder accepts, by look radius.
    let mut planets: Vec<(RealmId, f64)> = Vec::new();
    for r in rows.iter().filter(|r| r.parent == Some(system)) {
        if let (RealmId::Planet(_), Some(vd_core::geometry::Boundary::Shell { r: look })) =
            (r.realm, r.look)
        {
            planets.push((r.realm, look));
        }
    }
    planets.sort_by(|a, b| a.1.partial_cmp(&b.1).expect("finite"));
    let accepted: Vec<(RealmId, f64)> = planets
        .iter()
        .copied()
        .filter(|(realm, look)| build(*realm, *look, &held, &lineage).is_some())
        .collect();

    // The home planet's own moons, out of its own subtree.
    let home_realm = vd_core::worldgen::HOME_PLANET;
    let moon_held = std::collections::BTreeSet::from([home_realm]);
    let moon_lineage = std::collections::BTreeSet::from([
        vd_core::worldgen::GALAXY,
        vd_core::worldgen::HOME_SYSTEM,
    ]);
    let (moon_rows, _) = vd_physics::worldgen::shard_boot_world(
        DEV.universe_seed,
        &config,
        &moon_held,
        home_realm,
        &moon_lineage,
    );

    println!(
        "\nM-R — THE RELIEF CAP (slice 8b stage 3). sigma_y {:.4} MPa, crust share {} of the bulk, \
         shape share {} of the radius",
        vd_terrain::body::CRUST_YIELD_STRESS_PA / 1.0e6,
        vd_terrain::body::CRUST_DENSITY_SHARE,
        vd_terrain::body::SHAPE_RELIEF_SHARE,
    );
    println!(
        "body\trole\tladder_radius_m\tg_mm_s2\trho_kgm3\tstrength_arm_m\tshape_arm_m\tbinds\t\
         relief_new_m\trelief_old_m\tsum_amp_m\tfloor_m\tband_m"
    );
    let row = |role: &str,
               realm: RealmId,
               look: f64,
               held: &std::collections::BTreeSet<RealmId>,
               lineage: &std::collections::BTreeSet<RealmId>| {
        let Some((charter, body)) = build(realm, look, held, lineage) else {
            println!("{realm:?}\t{role}\tREFUSED");
            return;
        };
        let (strength, shape) = body.relief_arms_m();
        let cap = strength.min(shape);
        let relief = body.relief_m();
        // The stream's own draw, read back out of the law, then the OLD law on the same draw.
        let u = 2.0 * relief / cap - 1.0;
        let old_cap = (0.004 * body.ladder().radius_m()).clamp(200.0, 12_000.0);
        let old_relief = old_cap * (0.5 + u);
        println!(
            "{realm:?}\t{role}\t{:.1}\t{}\t{}\t{strength:.1}\t{shape:.1}\t{}\t{relief:.1}\t\
             {old_relief:.1}\t{:.1}\t{}\t{}",
            body.ladder().radius_m(),
            charter.gravity_mm_s2,
            charter.bulk_density_kgm3,
            if strength < shape {
                "strength"
            } else {
                "shape"
            },
            body.relief_bound_m(0),
            body.ladder().floor_m,
            body.ladder().band_m,
        );
    };
    let home_look = planets
        .iter()
        .find(|(realm, _)| *realm == home_realm)
        .map(|(_, look)| *look);
    if let Some(look) = home_look {
        row("the home planet", home_realm, look, &held, &lineage);
    }
    for r in moon_rows.iter().filter(|r| r.parent == Some(home_realm)) {
        if let (RealmId::Planet(_), Some(vd_core::geometry::Boundary::Shell { r: look })) =
            (r.realm, r.look)
        {
            row("a moon of it", r.realm, look, &moon_held, &moon_lineage);
        }
    }
    if let Some((realm, look)) = accepted.last() {
        row("the LARGEST planet", *realm, *look, &held, &lineage);
    }
    if let Some((realm, look)) = accepted.first() {
        row("the SMALLEST planet", *realm, *look, &held, &lineage);
    }
}

/// M-W — THE WATER CENSUS (slice 8b stage 5): every planet of the home system and every moon of
/// the home planet, with the charter's own stage-4 and stage-5 words: the sunlight, the water it
/// holds, whether it is locked to its star, its day, its surface temperature, its air and its crust.
fn water_table() {
    let config = UniverseConfig::world(DEV.move_speed, DEV.tick_dt);
    let system = RealmId::System(vd_bins::HOME_SYSTEM);
    let lineage = std::collections::BTreeSet::from([vd_core::worldgen::GALAXY]);
    let held = std::collections::BTreeSet::from([system]);
    let (rows, _) =
        vd_physics::worldgen::shard_boot_world(DEV.universe_seed, &config, &held, system, &lineage);
    println!(
        "\nM-W — THE WATER CENSUS (slice 8b stage 5): the formation zone against the snow line, \
         the census's own retention, the runaway edge."
    );
    println!(
        "body\trole\tlook_m\tS_rel\twater_km3\tearth_oceans\tlocked\tday_h\tT_surface_K\t\
         p_surf_Pa\tT_e_m"
    );
    let row = |role: &str,
               realm: RealmId,
               look: f64,
               held: &std::collections::BTreeSet<RealmId>,
               lineage: &std::collections::BTreeSet<RealmId>| {
        let Some(c) = vd_bins::body_charter(DEV.universe_seed, held, lineage, realm) else {
            println!("{realm:?}\t{role}\t{look:.0}\tNO CHARTER");
            return;
        };
        let water = c.water_km3.unwrap_or(0);
        println!(
            "{realm:?}\t{role}\t{look:.0}\t{:.4}\t{water}\t{:.3}\t{}\t{:.1}\t{:.1}\t{}\t{}",
            f64::from(c.insolation_q12) / 4096.0,
            water as f64 / 1.35e9,
            c.flags & vd_core::look::CHARTER_FLAG_TIDALLY_LOCKED != 0,
            c.day_s.map_or(0.0, |d| f64::from(d) / 3600.0),
            c.t_surface_mk.map_or(0.0, |t| f64::from(t) / 1000.0),
            c.p_surf_pa.map_or(0, |p| p),
            c.elastic_thickness_m.map_or(0, |t| t),
        );
    };
    let mut planets: Vec<(RealmId, f64)> = rows
        .iter()
        .filter(|r| r.parent == Some(system))
        .filter_map(|r| match (r.realm, r.look) {
            (RealmId::Planet(_), Some(vd_core::geometry::Boundary::Shell { r: look })) => {
                Some((r.realm, look))
            }
            _ => None,
        })
        .collect();
    planets.sort_by(|a, b| a.1.partial_cmp(&b.1).expect("finite"));
    for (realm, look) in &planets {
        let role = if *realm == vd_core::worldgen::HOME_PLANET {
            "the home planet"
        } else {
            "a planet"
        };
        row(role, *realm, *look, &held, &lineage);
    }
    let home_realm = vd_core::worldgen::HOME_PLANET;
    let moon_held = std::collections::BTreeSet::from([home_realm]);
    let moon_lineage = std::collections::BTreeSet::from([
        vd_core::worldgen::GALAXY,
        vd_core::worldgen::HOME_SYSTEM,
    ]);
    let (moon_rows, _) = vd_physics::worldgen::shard_boot_world(
        DEV.universe_seed,
        &config,
        &moon_held,
        home_realm,
        &moon_lineage,
    );
    for r in moon_rows.iter().filter(|r| r.parent == Some(home_realm)) {
        if let (RealmId::Planet(_), Some(vd_core::geometry::Boundary::Shell { r: look })) =
            (r.realm, r.look)
        {
            row(
                "a moon of the home planet",
                r.realm,
                look,
                &moon_held,
                &moon_lineage,
            );
        }
    }
}

/// M-S — THE SEA (slice 8b stage 6; ruling T8): for the four bodies, whether their water is liquid
/// at the surface (the boiling point at the charter's own pressure against its surface temperature),
/// and the level the water WOULD stand at over the body's own shape — the ocean share the owner
/// asked to see — solved here whether or not the charter states a sea.
fn sea_table() {
    let config = UniverseConfig::world(DEV.move_speed, DEV.tick_dt);
    let system = RealmId::System(vd_bins::HOME_SYSTEM);
    let lineage = std::collections::BTreeSet::from([vd_core::worldgen::GALAXY]);
    let held = std::collections::BTreeSet::from([system]);
    let (rows, _) =
        vd_physics::worldgen::shard_boot_world(DEV.universe_seed, &config, &held, system, &lineage);
    println!(
        "\nM-S — THE SEA (slice 8b stage 6): the liquid verdict at the charter's own surface, and the \
         level the water would stand at over the body's own shape."
    );
    println!(
        "body\trole\twater_km3\tT_surface_K\tp_surf_Pa\tboils_at_K\tliquid\tstated_sea_mm\t\
         would_stand_offset_m\tocean_share\trung\tsamples"
    );
    let row = |role: &str,
               realm: RealmId,
               look: f64,
               held: &std::collections::BTreeSet<RealmId>,
               lineage: &std::collections::BTreeSet<RealmId>| {
        let RealmId::Planet(seed) = realm else {
            return;
        };
        let Some(charter) = vd_bins::body_charter(DEV.universe_seed, held, lineage, realm) else {
            println!("{realm:?}\t{role}\tNO CHARTER");
            return;
        };
        let Some(body) =
            vd_terrain::BodyDefinition::from_seed(seed, look, vd_bins::facts_of_charter(&charter))
        else {
            println!("{realm:?}\t{role}\tREFUSED BY THE LADDER");
            return;
        };
        let stated = vd_bins::charter_with_sea(charter, Some(&body));
        let t_k = charter.t_surface_mk.map_or(0.0, |t| f64::from(t) / 1000.0);
        let p_pa = charter.p_surf_pa.map_or(0.0, f64::from);
        let boils = if p_pa > 0.0 {
            vd_physics::worldgen::water_boiling_point_k(p_pa)
        } else {
            0.0
        };
        let liquid = vd_physics::worldgen::water_is_liquid(t_k, p_pa);
        let water_m3 = charter.water_km3.map_or(0.0, |km3| km3 as f64 * 1.0e9);
        match vd_bins::sea::solve_sea_level(&body, water_m3) {
            Some(level) => println!(
                "{realm:?}\t{role}\t{}\t{t_k:.1}\t{p_pa:.0}\t{boils:.1}\t{liquid}\t{:?}\t{:.1}\t{:.4}\t{}\t{}",
                charter.water_km3.unwrap_or(0),
                stated.sea_offset_mm,
                level.offset_m,
                level.ocean_share,
                level.rung,
                level.samples,
            ),
            None => println!(
                "{realm:?}\t{role}\t{}\t{t_k:.1}\t{p_pa:.0}\t{boils:.1}\t{liquid}\t{:?}\tNO WATER",
                charter.water_km3.unwrap_or(0),
                stated.sea_offset_mm,
            ),
        }
    };
    let mut planets: Vec<(RealmId, f64)> = rows
        .iter()
        .filter(|r| r.parent == Some(system))
        .filter_map(|r| match (r.realm, r.look) {
            (RealmId::Planet(_), Some(vd_core::geometry::Boundary::Shell { r: look })) => {
                Some((r.realm, look))
            }
            _ => None,
        })
        .collect();
    planets.sort_by(|a, b| a.1.partial_cmp(&b.1).expect("finite"));
    let home_realm = vd_core::worldgen::HOME_PLANET;
    if let Some((_, look)) = planets.iter().find(|(realm, _)| *realm == home_realm) {
        row("the home planet", home_realm, *look, &held, &lineage);
    }
    if let Some((realm, look)) = planets.last() {
        row("the LARGEST planet", *realm, *look, &held, &lineage);
    }
    if let Some((realm, look)) = planets.first() {
        row("the SMALLEST planet", *realm, *look, &held, &lineage);
    }
    let moon_held = std::collections::BTreeSet::from([home_realm]);
    let moon_lineage = std::collections::BTreeSet::from([
        vd_core::worldgen::GALAXY,
        vd_core::worldgen::HOME_SYSTEM,
    ]);
    let (moon_rows, _) = vd_physics::worldgen::shard_boot_world(
        DEV.universe_seed,
        &config,
        &moon_held,
        home_realm,
        &moon_lineage,
    );
    for r in moon_rows.iter().filter(|r| r.parent == Some(home_realm)) {
        if let (RealmId::Planet(_), Some(vd_core::geometry::Boundary::Shell { r: look })) =
            (r.realm, r.look)
        {
            row(
                "a moon of the home planet",
                r.realm,
                look,
                &moon_held,
                &moon_lineage,
            );
        }
    }
}
