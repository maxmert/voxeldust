//! MEASURE what each boot shape builds and what it costs (foundation slice 5, 2026-09-06): the full
//! forest the gateway builds today, the system layer the sky needs, and the subtree a shard plants.
//! `cargo run --release -p vd-bins --example world_size`.
fn max_rss_mb() -> f64 {
    let mut u: libc::rusage = unsafe { std::mem::zeroed() };
    unsafe { libc::getrusage(libc::RUSAGE_SELF, &mut u) };
    // macOS reports ru_maxrss in bytes; Linux in kilobytes.
    if cfg!(target_os = "macos") {
        u.ru_maxrss as f64 / 1e6
    } else {
        u.ru_maxrss as f64 / 1e3
    }
}
fn main() {
    let p = &vd_bins::DEV;
    let which = std::env::args().nth(1).unwrap_or_else(|| "full".to_owned());
    let t = std::time::Instant::now();
    let cfg = vd_bins::process_world_config(p.move_speed, p.tick_dt);
    let n = match which.as_str() {
        "full" => vd_bins::boot_world(p.universe_seed, p.move_speed, p.tick_dt)
            .regions()
            .len(),
        "layer" => vd_physics::worldgen::system_layer_view(p.universe_seed, &cfg)
            .regions()
            .len(),
        "subtree" => {
            let layer = vd_physics::worldgen::system_layer_view(p.universe_seed, &cfg);
            let home =
                vd_core::worldgen::default_home_realm(layer.regions()).expect("a home system");
            let held = std::iter::once(home).collect();
            let lineage = [vd_core::worldgen::UNIVERSE, vd_core::worldgen::GALAXY, home]
                .into_iter()
                .collect();
            vd_bins::boot_world_lit(
                p.universe_seed,
                &held,
                home,
                p.move_speed,
                p.tick_dt,
                &lineage,
            )
            .0
            .len()
        }
        "guard" => {
            let layer = vd_physics::worldgen::system_layer_view(p.universe_seed, &cfg);
            let home =
                vd_core::worldgen::default_home_realm(layer.regions()).expect("a home system");
            let held = std::iter::once(home).collect();
            let lineage = [vd_core::worldgen::UNIVERSE, vd_core::worldgen::GALAXY, home]
                .into_iter()
                .collect();
            vd_physics::worldgen::guard_seeded_systems_disjoint(p.universe_seed, &cfg)
                .expect("disjoint");
            vd_physics::worldgen::guard_planted_subtree(p.universe_seed, &cfg, &held, &lineage, 2)
                .expect("planted");
            0
        }
        "gateway" => {
            let (view, lineage) =
                vd_bins::boot_world_for_home(p.universe_seed, p.move_speed, p.tick_dt);
            let home = vd_core::worldgen::default_home_realm(view.regions()).expect("home");
            vd_physics::worldgen::guard_seeded_systems_disjoint(p.universe_seed, &cfg)
                .expect("disjoint");
            vd_physics::worldgen::guard_planted_subtree(
                p.universe_seed,
                &cfg,
                &std::iter::once(home).collect(),
                &lineage,
                2,
            )
            .expect("planted");
            view.regions().len()
        }
        _ => 0,
    };
    println!(
        "{which}: regions={n} took={:.2}s max_rss={:.0} MB",
        t.elapsed().as_secs_f64(),
        max_rss_mb()
    );
}
