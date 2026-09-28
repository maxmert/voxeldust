//! A diagnostic (not a gate): THE HOME STAR SYSTEM'S WINDOW FRAME, CUT INTO ITS PARTS (2026-09-27,
//! the far planets that blink). The system ships every orbiting child in each tick's window frame,
//! its rows in realm order, cut by `partition_window_rows` under the snapshot budget; this builds the
//! same rows from the home system's own boot world (each planet at its orbit's place at the given
//! second) and prints which planets ride which part. Before minor 35 the planets of a late or lost
//! part left the gateway's picture for that tick; since minor 35 each part names its place and the
//! gateway draws a tick only whole.
//!
//! ```text
//! cargo run --release -p vd-bins --example window_parts -- [budget_bytes] [seconds]
//! ```
//!
//! **Example.** At the development budget of 1 100 bytes the nine planets ride two parts; the three
//! largest seeds ride the second, and those were the planets a walker saw blink.

use vd_core::glam::DQuat;
use vd_core::pose::{LatticePos, RealmId, StampedPose};
use vd_wire::channels::RealmSnap;

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let budget: usize = args.first().and_then(|s| s.parse().ok()).unwrap_or(1100);
    let secs: f64 = args.get(1).and_then(|s| s.parse().ok()).unwrap_or(0.0);
    let dev = vd_bins::DEV;
    let config = vd_physics::worldgen::UniverseConfig::world(dev.move_speed, dev.tick_dt);
    let system = RealmId::System(vd_bins::HOME_SYSTEM);
    let held = std::collections::BTreeSet::from([system]);
    let lineage = std::collections::BTreeSet::from([vd_core::worldgen::GALAXY]);
    let (regions, movers) =
        vd_physics::worldgen::shard_boot_world(dev.universe_seed, &config, &held, system, &lineage);
    let own = regions
        .iter()
        .find(|r| r.realm == system)
        .expect("the system's own row");
    let tier = own.frame.tier();
    let mut rows: Vec<RealmSnap> = movers
        .iter()
        .map(|(realm, orbit)| {
            let region = regions
                .iter()
                .find(|r| r.realm == *realm)
                .expect("a mover has a row");
            let state = vd_physics::celestial::orbital_state(orbit, secs);
            RealmSnap {
                realm: *realm,
                frame: region.frame,
                pose: StampedPose {
                    frame: own.frame,
                    pos: LatticePos::from_metres(state.position, tier),
                    vel: state.velocity,
                    orient: DQuat::IDENTITY,
                    universe_tick: vd_core::UniverseTick(1),
                },
            }
        })
        .collect();
    // The shard names its movers as a set, so its rows come in realm order.
    rows.sort_by_key(|r| r.realm);
    let parts = vd_wire::channels::partition_window_rows(&rows, budget);
    println!(
        "the home system's frame at {budget} bytes: {} movers in {} parts",
        rows.len(),
        parts.len()
    );
    for (k, part) in parts.iter().enumerate() {
        let names: Vec<String> = part
            .iter()
            .map(|r| {
                let m = r.pose.pos.delta_m(LatticePos::ORIGIN, tier).length();
                format!("{:?} ({:.3e} m out)", r.realm, m)
            })
            .collect();
        println!("  part {} of {}: {names:?}", k + 1, parts.len());
    }
}
