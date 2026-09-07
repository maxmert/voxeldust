//! U-5 — how far the ladder moves each body's drawn radius on THE world (the voxel foundation,
//! slice 2). Walks the generated planets of one star system AND their moons, and reports the largest
//! snap in metres and as a fraction, against the rule's bound: at most half a ladder unit, and always
//! under one top-rung cell. REFUSES to report when the walk found no moon, because a planet below a
//! star system is found only through its lineage, and an empty moon list is a broken walk, not a
//! moonless system. Run in release:
//!
//! ```text
//! cargo run --release -p vd-bins --example ladder_snap            # the home system
//! VD_SYSTEM=214 cargo run --release -p vd-bins --example ladder_snap
//! ```

use std::collections::BTreeSet;
use std::process::ExitCode;

use vd_bins::DEV;
use vd_core::geometry::{Boundary, RealmRegion};
use vd_core::grid::{BandParams, ShellGrid};
use vd_core::pose::RealmId;
use vd_physics::worldgen::{UniverseConfig, shard_boot_world};

/// The direct children of `hosted` on THE world, from its own subtree, given its lineage (the
/// ancestors above it — a body below a star system cannot find itself from its own name).
fn children_of(
    config: &UniverseConfig,
    hosted: RealmId,
    lineage: &BTreeSet<RealmId>,
) -> Vec<RealmRegion> {
    let held = BTreeSet::from([hosted]);
    let (rows, _) = shard_boot_world(DEV.universe_seed, config, &held, hosted, lineage);
    rows.into_iter()
        .filter(|r| r.parent == Some(hosted))
        .collect()
}

fn main() -> ExitCode {
    let system: u64 = std::env::var("VD_SYSTEM")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(7);
    let config = UniverseConfig::world(DEV.move_speed, DEV.tick_dt);
    let system_realm = RealmId::System(system);
    let system_lineage = BTreeSet::from([vd_core::worldgen::GALAXY]);
    let planets = children_of(&config, system_realm, &system_lineage);
    let mut bodies: Vec<RealmRegion> = Vec::new();
    let mut moons = 0u32;
    for p in &planets {
        if let RealmId::Planet(_) = p.realm {
            bodies.push(*p);
            let lineage = BTreeSet::from([vd_core::worldgen::GALAXY, system_realm]);
            let children = children_of(&config, p.realm, &lineage);
            moons += u32::try_from(children.len()).unwrap_or(u32::MAX);
            bodies.extend(children);
        }
    }

    let mut round = 0u32;
    let mut too_large = 0u32;
    let mut worst_m = 0.0_f64;
    let mut worst_frac = 0.0_f64;
    let mut over_a_top_cell = 0u32;
    let mut over_half_unit = 0u32;
    for region in &bodies {
        let RealmId::Planet(_) = region.realm else {
            continue;
        };
        let Some(Boundary::Shell { r }) = region.look else {
            continue;
        };
        round += 1;
        let Some(grid) = ShellGrid::for_body(r, BandParams::provisional(r)) else {
            too_large += 1;
            println!(
                "ladder_snap: {:?} radius {r:.0} m exceeds the address (no grid)",
                region.realm
            );
            continue;
        };
        let snap = grid.radius_m() - r;
        let unit_m = f64::from(1u32 << (grid.rungs() - 1)) * std::f64::consts::FRAC_2_PI;
        let top_cell_m = f64::from(1u32 << (grid.rungs() - 1));
        if snap.abs() > unit_m / 2.0 + 1e-6 {
            over_half_unit += 1;
        }
        if snap.abs() > top_cell_m {
            over_a_top_cell += 1;
        }
        if snap.abs() > worst_m {
            worst_m = snap.abs();
        }
        if snap.abs() / r > worst_frac {
            worst_frac = snap.abs() / r;
        }
        let kind = if region.parent == Some(system_realm) {
            "planet"
        } else {
            "moon  "
        };
        println!(
            "ladder_snap: {kind} {:?} radius {r:.1} m -> {:.1} m (snap {snap:+.2} m, {} rungs, N {})",
            region.realm,
            grid.radius_m(),
            grid.rungs(),
            grid.n()
        );
    }
    println!(
        "ladder_snap: system {system} seed {}: {} planets, {moons} moons, {round} round bodies, {too_large} too large; worst snap {worst_m:.2} m ({:.5} %); over half a ladder unit: {over_half_unit}; over one top-rung cell: {over_a_top_cell}",
        DEV.universe_seed,
        planets.len(),
        worst_frac * 100.0
    );
    if moons == 0 {
        println!(
            "ladder_snap: REFUSED — the walk found no moon, so the subtree query is wrong; no result is reported"
        );
        return ExitCode::FAILURE;
    }
    ExitCode::SUCCESS
}
