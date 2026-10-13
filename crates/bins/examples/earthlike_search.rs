//! ★ THE EARTH-LIKE STAND SEARCH (2026-10-13; the owner: "Maybe worth to try to find another home world? Which terrain
//! will be a bit more like earth? In that case it would be easier to understand if we are doing right things or wrong."
//! — "Yes, lets try the search"). A read-only instrument: it reads THE world (SL5: no variant, no second generator) —
//! the star systems nearest the home system, each one's planets and their charters through the generator's own
//! functions — and ranks the planets by how near their LAND-SHAPING facts stand to Earth's. The ranking is TOOL POLICY
//! (as `vd-seedsearch`'s): it lives here and never reaches the generator. Bench only.
//!
//! THE FACTS RANKED, each a distance from Earth's own value, all of one weight: the surface gravity (`|ln(g/g⊕)|`), the
//! axial tilt (`|ε − 23.44°| / 23.44°`), the surface temperature (`|ln(T/288 K)|`), the air (`|ln(P/101 325 Pa)|`), the
//! water inventory (`|ln(W/W⊕)|`, W⊕ = 1.386 × 10⁹ km³) and the day (`|ln(day/86 400 s)|`). A planet must have a solid
//! surface, air, and water liquid at its surface temperature and pressure.
//! `cargo run --release -p vd-bins --example earthlike_search -- [systems] [shown]`

use std::collections::BTreeSet;

use vd_core::look::{BodyCharter, CHARTER_FLAG_HAS_AIR, CHARTER_FLAG_SOLID_SURFACE};
use vd_core::pose::RealmId;
use vd_core::worldgen::{GALAXY, HOME_SYSTEM};

const UNIVERSE: u64 = vd_bins::DEV.universe_seed;
/// Earth's facts the distances read (the reference home, as `vd-seedsearch`'s).
const EARTH_G_MM_S2: f64 = 9_806.65;
const EARTH_TILT_DEG: f64 = 23.44;
const EARTH_T_K: f64 = 288.0;
const EARTH_P_PA: f64 = 101_325.0;
const EARTH_WATER_KM3: f64 = 1.386e9;
const EARTH_DAY_S: f64 = 86_400.0;

/// One planet's land-shaping facts, read off its charter.
struct Facts {
    g: f64,
    tilt_deg: f64,
    t_k: f64,
    p_pa: f64,
    water_km3: f64,
    day_s: f64,
    tidally_locked: bool,
}

fn facts(c: &BodyCharter) -> Option<Facts> {
    let solid = c.flags & CHARTER_FLAG_SOLID_SURFACE != 0;
    let air = c.flags & CHARTER_FLAG_HAS_AIR != 0;
    if !(solid && air) {
        return None;
    }
    let t_k = f64::from(c.t_surface_mk?) / 1_000.0;
    let p_pa = f64::from(c.p_surf_pa?);
    if !vd_physics::worldgen::water_is_liquid(t_k, p_pa) {
        return None;
    }
    let water_km3 = c.water_km3? as f64;
    if water_km3 <= 0.0 {
        return None;
    }
    let tilt_deg = (f64::from(c.obliquity_cos_q1024?) / 1_024.0)
        .clamp(-1.0, 1.0)
        .acos()
        .to_degrees();
    Some(Facts {
        g: f64::from(c.gravity_mm_s2),
        tilt_deg,
        t_k,
        p_pa,
        water_km3,
        day_s: f64::from(c.day_s.unwrap_or(0)),
        tidally_locked: c.flags & vd_core::look::CHARTER_FLAG_TIDALLY_LOCKED != 0,
    })
}

/// The distance from Earth: the six terms and their sum.
fn distance(f: &Facts) -> ([f64; 6], f64) {
    let day = if f.day_s > 0.0 {
        (f.day_s / EARTH_DAY_S).ln().abs()
    } else {
        f64::INFINITY
    };
    let terms = [
        (f.g / EARTH_G_MM_S2).ln().abs(),
        (f.tilt_deg - EARTH_TILT_DEG).abs() / EARTH_TILT_DEG,
        (f.t_k / EARTH_T_K).ln().abs(),
        (f.p_pa / EARTH_P_PA).ln().abs(),
        (f.water_km3 / EARTH_WATER_KM3).ln().abs(),
        day,
    ];
    (terms, terms.iter().sum())
}

fn row(name: &str, f: &Facts, d: ([f64; 6], f64)) -> String {
    format!(
        "{name:<34} | {:.2} g | tilt {:>5.1}° | {:>5.1} K | {:>7.3} bar | water {:>6.2} Earths | day {:>7.1} h{} | distance {:.2} (g {:.2} tilt {:.2} T {:.2} P {:.2} water {:.2} day {:.2})",
        f.g / EARTH_G_MM_S2,
        f.tilt_deg,
        f.t_k,
        f.p_pa / 100_000.0,
        f.water_km3 / EARTH_WATER_KM3,
        f.day_s / 3_600.0,
        if f.tidally_locked { " (locked)" } else { "" },
        d.1,
        d.0[0],
        d.0[1],
        d.0[2],
        d.0[3],
        d.0[4],
        d.0[5],
    )
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let systems: usize = args.first().map_or(2_000, |s| s.parse().expect("a count"));
    let shown: usize = args.get(1).map_or(15, |s| s.parse().expect("a count"));
    let config =
        vd_physics::worldgen::UniverseConfig::world(vd_bins::DEV.move_speed, vd_bins::DEV.tick_dt);
    let started = std::time::Instant::now();
    // THE STAR SYSTEMS, by their distance from the home system in the galaxy's frame.
    let view = vd_physics::worldgen::system_layer_view(UNIVERSE, &config);
    let regions = view.regions();
    let galaxy = regions
        .iter()
        .find(|r| r.realm == GALAXY)
        .expect("the galaxy's row");
    let home_at = regions
        .iter()
        .find(|r| r.realm == HOME_SYSTEM)
        .expect("the home system's row")
        .center
        .metres_in(galaxy);
    let mut near: Vec<(f64, RealmId)> = regions
        .iter()
        .filter(|r| matches!(r.realm, RealmId::System(_)) && r.parent == Some(GALAXY))
        .map(|r| ((r.center.metres_in(galaxy) - home_at).length(), r.realm))
        .collect();
    near.sort_by(|a, b| a.0.total_cmp(&b.0));
    near.truncate(systems);
    println!(
        "THE EARTH-LIKE STAND SEARCH: the {} star systems nearest the home system (out to {:.1} light-years), {:.1} s",
        near.len(),
        near.last().map_or(0.0, |n| n.0) / 9.460_730_472_580_8e15,
        started.elapsed().as_secs_f64()
    );
    let lineage = BTreeSet::from([GALAXY]);
    let (mut planets, mut ranked) = (0usize, Vec::new());
    for (dist_m, system) in &near {
        let held = BTreeSet::from([*system]);
        let (rows, _) =
            vd_physics::worldgen::shard_boot_world(UNIVERSE, &config, &held, *system, &lineage);
        for r in rows
            .iter()
            .filter(|r| matches!(r.realm, RealmId::Planet(_)) && r.parent == Some(*system))
        {
            planets += 1;
            let Some(c) = vd_bins::body_charter(UNIVERSE, &held, &lineage, r.realm) else {
                continue;
            };
            let Some(f) = facts(&c) else {
                continue;
            };
            let radius_km = match r.look {
                Some(vd_core::geometry::Boundary::Shell { r }) => r / 1_000.0,
                _ => 0.0,
            };
            let d = distance(&f);
            ranked.push((d.1, *system, r.realm, *dist_m, radius_km, f, d));
        }
    }
    ranked.sort_by(|a, b| a.0.total_cmp(&b.0));
    println!(
        "{planets} planets read; {} with a solid surface, air and liquid water ({:.1} s)\n",
        ranked.len(),
        started.elapsed().as_secs_f64()
    );
    // The home planet beside them, for the reader.
    let home_held = BTreeSet::from([HOME_SYSTEM]);
    if let Some(home) = vd_bins::home_body(UNIVERSE)
        && let Some(c) =
            vd_bins::body_charter(UNIVERSE, &home_held, &lineage, RealmId::Planet(home.seed()))
        && let Some(f) = facts(&c)
    {
        let d = distance(&f);
        println!("{}", row(&format!("HOME Planet({})", home.seed()), &f, d));
    }
    for (rank, (_, system, planet, dist_m, radius_km, f, d)) in
        ranked.iter().take(shown).enumerate()
    {
        println!(
            "{}",
            row(
                &format!(
                    "#{:<2} {planet:?} {radius_km:.0} km, {:.1} ly",
                    rank + 1,
                    dist_m / 9.460_730_472_580_8e15
                ),
                f,
                *d
            )
        );
        let _ = system;
    }
}
