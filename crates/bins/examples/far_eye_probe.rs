//! THE FAR EYE (slice 8, the two-radii cutoff's replacement; widened 2026-09-15 with the extended
//! ladder): what the ladder wants of a body at every distance, and what one descent costs there.
//! It reads the wanted set at 1.001, 1.5, 2, 3, 5, 10, 34, 100 and 1 000 body radii, and at the
//! body's own stated reach, on FOUR BODIES — the home planet, its moon, and the home system's
//! largest and smallest planets — so the ladder's behaviour at distance is read on a giant and on a
//! moon and not only on the world under the pilot's boots.
//!
//! ```text
//! cargo run --release -p vd-bins --example far_eye_probe
//! ```

use std::process::ExitCode;

use vd_bins::DEV;
use vd_core::glam::DVec3;

fn main() -> ExitCode {
    let Some(home) = vd_bins::home_body(DEV.universe_seed) else {
        eprintln!("far_eye_probe: the world names no home planet");
        return ExitCode::FAILURE;
    };
    let siblings = census(&home);
    let children = children_census(&home);
    // THE SUBJECTS: the home planet, every moon of it, and the LARGEST and SMALLEST planet of the
    // home system. One probe, four kinds of body, so the ladder's behaviour at distance is read on
    // a giant and on a moon and not only on the world under the pilot's boots.
    let mut subjects: Vec<(String, vd_terrain::BodyDefinition)> =
        vec![("the home planet".to_owned(), home)];
    for (name, body) in children {
        subjects.push((format!("the home planet's moon {name}"), body));
    }
    let mut by_radius = siblings.clone();
    by_radius.sort_by(|a, b| {
        a.1.ladder()
            .radius_m()
            .partial_cmp(&b.1.ladder().radius_m())
            .expect("finite")
    });
    if let Some((name, body)) = by_radius.last() {
        subjects.push((format!("the home system's LARGEST planet {name}"), *body));
    }
    if let Some((name, body)) = by_radius.first() {
        subjects.push((format!("the home system's SMALLEST planet {name}"), *body));
    }
    for (label, body) in &subjects {
        probe(label, body);
    }
    ExitCode::SUCCESS
}

/// What the ladder wants of ONE body at every distance, and what one descent costs there.
fn probe(label: &str, body: &vd_terrain::BodyDefinition) {
    let ladder = *body.ladder();
    let radius = ladder.radius_m();
    let rungs = ladder.rungs;
    let top = rungs.saturating_sub(1);
    let px = vd_client::ladder_view::pixel_rad();
    println!("\n===== {label} =====");
    println!(
        "far_eye_probe: seed {}, radius {radius:.0} m, {rungs} rungs, top cell {} m, top cells per \
         face edge {}, top chunk columns per face edge {}, six faces hold {} top columns",
        body.seed(),
        vd_seed::ladder::cell_m(top),
        ladder.cells_per_edge(top),
        (ladder.cells_per_edge(top) as i32 - 1) / vd_terrain::chunk::CHUNK_EDGE as i32 + 1,
        6 * (((ladder.cells_per_edge(top) as i32 - 1) / vd_terrain::chunk::CHUNK_EDGE as i32 + 1)
            .pow(2)),
    );
    println!(
        "far_eye_probe: the pixel {px:.6e} rad, the top rung's switch {:.4e} m ({:.4} radii), the \
         band {} m and {} cells at the top",
        vd_client::ladder_view::switch_m(top),
        vd_client::ladder_view::switch_m(top) / radius,
        ladder.band_m,
        ladder.cells_in_band(top),
    );
    let reach_m =
        vd_core::geometry::visibility_reach_m(radius, vd_core::geometry::VISIBILITY_THETA_MIN_RAD);
    println!(
        "far_eye_probe: the realm's stated reach by size {reach_m:.4e} m ({:.3} radii)",
        reach_m / radius,
    );
    // ★ THE CLIENT HAS NO FAR EDGE (owner 2026-09-15, *"agree"*): the server's visibility radius is
    // the only rule. The top-rung chunk column stays in the print because it says how coarse the
    // coarsest ground the ladder owns is.
    println!(
        "far_eye_probe: a top-rung chunk column is {:.0} m across, {:.2} body radii",
        vd_client::ladder_view::column_span_m(top),
        vd_client::ladder_view::column_span_m(top) / radius,
    );
    let d = vd_core::glam::DVec3::new(0.3, 0.5, 0.81).normalize();
    let mut radii: Vec<f64> = vec![1.001, 1.5, 2.0, 3.0, 5.0, 10.0, 34.0, 100.0, 1000.0];
    radii.push(reach_m / radius);
    radii.sort_by(|a, b| a.partial_cmp(b).expect("finite"));
    println!("radii\tdistance_m\tchunks\tcolumns\turgent\trevealed\tdescent_ms\tper_rung");
    for k in radii {
        let eye = d * radius * k;
        let mut view = vd_client::ladder_view::LadderView::default();
        let _ = view.wanted(body, eye.to_array());
        let started = std::time::Instant::now();
        let set = view.wanted(body, eye.to_array());
        let ms = started.elapsed().as_secs_f64() * 1e3;
        let per_rung: Vec<String> = set
            .per_rung()
            .iter()
            .map(|(r, n)| format!("{r}:{n}"))
            .collect();
        println!(
            "{k:.4}\t{:.4e}\t{}\t{}\t{}\t{}\t{ms:.2}\t{}",
            eye.length(),
            set.len(),
            set.columns(),
            set.urgent_count(),
            set.revealed_count(),
            per_rung.join(","),
        );
    }
}

/// THE CENSUS of the HOME PLANET'S OWN CHILDREN — the moons a pilot standing on the planet has in
/// her window beside the ground under her feet. Each one's distance from the planet's centre, in
/// its OWN radii, beside the two drawable floors (the whole body one pixel, one top-rung tile one
/// pixel) that the client no longer culls by — the server's visibility radius is the only rule.
fn children_census(home: &vd_terrain::BodyDefinition) -> Vec<(String, vd_terrain::BodyDefinition)> {
    let mut found = Vec::new();
    let config = vd_physics::worldgen::UniverseConfig::world(DEV.move_speed, DEV.tick_dt);
    let home_realm = vd_core::pose::RealmId::Planet(home.seed());
    let lineage = std::collections::BTreeSet::from([
        vd_core::worldgen::GALAXY,
        vd_core::worldgen::HOME_SYSTEM,
    ]);
    let held = std::collections::BTreeSet::from([home_realm]);
    let (rows, _) = vd_physics::worldgen::shard_boot_world(
        DEV.universe_seed,
        &config,
        &held,
        home_realm,
        &lineage,
    );
    let px = vd_client::ladder_view::pixel_rad();
    println!(
        "child_realm\tradius_m\tdistance_m\tdistance_own_radii\tbody_one_pixel_m\ttile_one_pixel_m"
    );
    for row in &rows {
        if row.parent != Some(home_realm) {
            continue;
        }
        let Some(vd_core::geometry::Boundary::Shell { r }) = row.look else {
            eprintln!(
                "far_eye_probe: the home planet's child {:?} has no shell look ({:?})",
                row.realm, row.look
            );
            continue;
        };
        let seed = match row.realm {
            vd_core::pose::RealmId::Planet(seed) => seed,
            other => {
                eprintln!("far_eye_probe: the home planet's child {other:?} is not a body");
                continue;
            }
        };
        let Some(child) = vd_terrain::BodyDefinition::from_seed(seed, r) else {
            continue;
        };
        let ladder = *child.ladder();
        let top = ladder.rungs.saturating_sub(1);
        let Some(parent_row) = rows.iter().find(|r| r.realm == home_realm) else {
            continue;
        };
        let distance = row.center.metres_in(parent_row).length();
        println!(
            "{:?}\t{:.4e}\t{distance:.4e}\t{:.2}\t{:.4e}\t{:.4e}",
            row.realm,
            ladder.radius_m(),
            distance / ladder.radius_m(),
            vd_core::geometry::visibility_reach_m(ladder.radius_m(), px),
            vd_client::ladder_view::column_span_m(top) / px,
        );
        found.push((format!("{:?}", row.realm), child));
    }
    found
}

/// THE CENSUS of the home system's bodies, from the home planet's own stand: what the pilot's
/// window holds beside the ground under her feet.
fn census(home: &vd_terrain::BodyDefinition) -> Vec<(String, vd_terrain::BodyDefinition)> {
    let mut found = Vec::new();
    let config = vd_physics::worldgen::UniverseConfig::world(DEV.move_speed, DEV.tick_dt);
    let system = vd_core::worldgen::HOME_SYSTEM;
    let lineage = std::collections::BTreeSet::from([vd_core::worldgen::GALAXY]);
    let held = std::collections::BTreeSet::from([system]);
    let (rows, movers) =
        vd_physics::worldgen::shard_boot_world(DEV.universe_seed, &config, &held, system, &lineage);
    let at = |realm: vd_core::pose::RealmId| -> Option<DVec3> {
        movers
            .iter()
            .find(|(r, _)| *r == realm)
            .map(|(_, o)| vd_physics::celestial::orbital_state(o, 0.0).position)
    };
    let home_realm = vd_core::pose::RealmId::Planet(home.seed());
    let Some(home_at) = at(home_realm) else {
        eprintln!("far_eye_probe: the home planet does not move in its system; no census");
        return found;
    };
    println!(
        "census_realm\tradius_m\tdistance_m\tdistance_own_radii\tbody_one_pixel_m\ttile_one_pixel_m"
    );
    for row in &rows {
        if row.parent != Some(vd_core::worldgen::HOME_SYSTEM) {
            continue;
        }
        let Some(vd_core::geometry::Boundary::Shell { r }) = row.look else {
            continue;
        };
        let vd_core::pose::RealmId::Planet(seed) = row.realm else {
            continue;
        };
        let Some(sibling) = vd_terrain::BodyDefinition::from_seed(seed, r) else {
            continue;
        };
        if row.realm == home_realm {
            eprintln!(
                "far_eye_probe: the home planet's own SHELL (its containment bound) is {:.4e} m, \
                 {:.1} radii; its look is {:?}",
                row.shape.circumscribed_extent(),
                row.shape.circumscribed_extent() / r,
                row.look,
            );
        }
        let Some(pos) = at(row.realm) else { continue };
        let distance = (pos - home_at).length();
        let ladder = *sibling.ladder();
        let top = ladder.rungs.saturating_sub(1);
        let px = vd_client::ladder_view::pixel_rad();
        let body_floor = vd_core::geometry::visibility_reach_m(ladder.radius_m(), px);
        let tile_floor = vd_client::ladder_view::column_span_m(top) / px;
        println!(
            "{:?}\t{:.4e}\t{distance:.4e}\t{:.2}\t{body_floor:.4e}\t{tile_floor:.4e}",
            row.realm,
            ladder.radius_m(),
            distance / ladder.radius_m(),
        );
        found.push((format!("{:?}", row.realm), sibling));
    }
    found
}
