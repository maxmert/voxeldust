//! ★ THE SHIPYARD'S STAND-IN, AIMED AT THE SPAWN — berth one hull forty metres from where a player
//! logs in, into the running dev cluster's slot directory, named exactly as the shards name their
//! files. `cargo run -p vd-bins --example berth_hull -- <slot workdir> [owner account] [push m/s^2]
//! [turn rad/s^2]`. Runs AFTER `dev-cluster.sh up --demand` (the directory exists, no shard has booted
//! yet) and BEFORE the first login (the home system reads its berths at boot).
//!
//! THE RATING IS THIS HULL'S OWN DATA (owner ruling 2026-08-27: a rated push is a fact about what a
//! ship IS). The default here is a TEST hull's: strong enough to cross the home star system in
//! minutes at the top throttle tier, gentle enough at tier 1 to leave a berth (owner, 2026-09-02:
//! *"something faster, so we can really test travel inside the star system"*). It is stated on ONE
//! row by this tool, never a default every hull shares. MEASURED against the world: the home system
//! is a shell 5.2e12 m across, the spawn sits 1.08e10 m from its star, and its planets' shells are
//! 1.5e8 to 4.7e10 m in radius.

/// A test hull's push: ten million million metres per second per second at warp tier 10, one metre
/// per second per second at normal tier 1 (the ladder's floor is 1e-13 and the two modes meet at
/// 1e-5; see `input_map::ThrottleMode`).
/// Ten seconds at the top tier is a hundred million million metres per second, and the nearest
/// stars — thirty thousand million million metres away — slide across the window within a minute.
/// MEASURED first at a hundred million (owner flight, 2026-09-02): the movement path held, the
/// crossing out of the star system went through at speed, and the star field did not move.
///
/// ★ THE ROW'S GRID CAPS THE RATING: the facts hold whole MICRO-units in an `i64`, so the largest
/// push a row can state is about 9.22e12 m/s². MEASURED: asking for 1e13 saturated to
/// 9 223 372 036 854.775 m/s² (the tool printed it), so the default sits under the cap and the tool
/// refuses a number past it instead of rounding it in silence.
const DEFAULT_TEST_PUSH_MPS2: f64 = 9.0e12;
/// A test hull's turn: three radians per second, per second — a hull that answers the keys.
const DEFAULT_TEST_TURN_RADPS2: f64 = 3.0;

/// A rate in whole micro-units, refused past the row's `i64` grid rather than saturated.
fn micro(rate: f64, what: &str) -> i64 {
    let units = (rate * 1.0e6).round();
    assert!(
        units.abs() <= i64::MAX as f64,
        "{what} {rate} m/s^2 is past the row's grid (at most {} in micro-units)",
        i64::MAX
    );
    units as i64
}

fn main() {
    let dir = std::env::args().nth(1).expect("the slot's work directory");
    let owner = std::env::args().nth(2).unwrap_or_else(|| "1000".to_owned());
    let push_mps2: f64 = std::env::args().nth(3).map_or(DEFAULT_TEST_PUSH_MPS2, |s| {
        s.parse().expect("push in m/s^2")
    });
    let turn_radps2: f64 = std::env::args()
        .nth(4)
        .map_or(DEFAULT_TEST_TURN_RADPS2, |s| {
            s.parse().expect("turn in rad/s^2")
        });
    let push_micro = micro(push_mps2, "push").to_string();
    let turn_micro = micro(turn_radps2, "turn").to_string();
    let dev = vd_bins::DEV;
    let roster = vd_bins::world_roster(&dev);
    let spawn =
        vd_bins::boot_world(dev.universe_seed, dev.move_speed, dev.tick_dt).default_home_offset_m();
    let hull = vd_core::pose::RealmId::Ship(vd_core::ids::EntityId::pack(
        vd_core::entity_kind::EntityKind::Ship,
        1,
        1,
        0,
    ));
    let dir = std::path::Path::new(&dir);
    let parent_store = vd_bins::realm_store_path(dir, roster.home);
    let ship_store = vd_bins::realm_store_path(dir, hull);
    let tool = std::env::current_exe()
        .expect("own path")
        .parent()
        .and_then(|p| p.parent())
        .expect("target/debug")
        .join("vd-build-ship");
    let out = std::process::Command::new(&tool)
        .args([
            "--parent-store",
            &parent_store,
            "--ship-store",
            &ship_store,
            "--owner",
            &owner,
            "--berth-x-m",
            &(spawn.x + 40.0).to_string(),
            "--berth-y-m",
            &spawn.y.to_string(),
            "--berth-z-m",
            &spawn.z.to_string(),
            "--max-push-micro-mps2",
            &push_micro,
            "--max-turn-micro-radps2",
            &turn_micro,
        ])
        .output()
        .expect("vd-build-ship runs");
    assert!(
        out.status.success(),
        "vd-build-ship: {}",
        String::from_utf8_lossy(&out.stderr)
    );
    println!(
        "berthed {} in {} at spawn + 40 m along +x (spawn {:?}); files:\n  {parent_store}\n  {ship_store}\n  tool said: {}",
        hull,
        roster.home,
        spawn,
        String::from_utf8_lossy(&out.stdout).trim()
    );
}
