fn main() {
    let dev = vd_bins::DEV;
    let cfg = vd_bins::process_world_config(dev.move_speed, dev.tick_dt);
    let world = vd_physics::worldgen::WorldView::generated(dev.universe_seed, &cfg);
    let at = world.default_home_offset_m();
    println!("SPAWN {} {} {}", at.x, at.y, at.z);
}
