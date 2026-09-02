//! Diagnosis probe: list the home star system's direct children with their distance from the
//! spawn point, so a test hull's rating can be sized against the real distances it must cover.
//! `cargo run -p vd-bins --example system_census`.
fn main() {
    let dev = vd_bins::DEV;
    let world = vd_bins::boot_world(dev.universe_seed, dev.move_speed, dev.tick_dt);
    let roster = vd_bins::world_roster(&dev);
    let spawn = world.default_home_offset_m();
    let regions = world.regions();
    let home = regions
        .iter()
        .find(|r| r.realm == roster.home)
        .expect("the home system is a region");
    println!(
        "home {:?} shape {:?} spawn {spawn:?}",
        roster.home, home.shape
    );
    // ONE index for the whole forest: the per-realm lookup rebuilds it otherwise, and a forest of
    // millions of regions turns a listing into a walk that never ends (measured: killed at 4 min).
    let ix = vd_core::worldgen::realm_index(regions);
    for r in regions {
        let Some(coord) = vd_core::worldgen::coord_of_realm_indexed(regions, &ix, r.realm) else {
            continue;
        };
        if coord.parent().map(|p| p.lowered()) != Some(roster.home) {
            continue;
        }
        let c = r.center.metres_in(home);
        println!(
            "child {:?} centre {c:?} from spawn {:.3e} m shape {:?}",
            r.realm,
            (c - spawn).length(),
            r.shape
        );
    }
}
