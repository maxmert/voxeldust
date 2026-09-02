fn main() {
    let dev = vd_bins::DEV;
    let held: std::collections::BTreeSet<_> =
        std::iter::once(vd_core::pose::RealmId::System(7)).collect();
    let berth = vd_core::built::Berth {
        child: vd_core::pose::RealmId::Ship(vd_core::ids::EntityId::pack(
            vd_core::entity_kind::EntityKind::Ship,
            1,
            1,
            0,
        )),
        offset_m: vd_core::glam::DVec3::new(40.0, 0.0, 10_822_131_358.930_182),
        bound: vd_core::geometry::Boundary::Shell { r: 20.0 },
        look: vd_core::geometry::Boundary::Shell { r: 20.0 },
        fence: vd_core::fence::Fence::GENESIS,
    };
    let (regions, _m, _l) = vd_bins::boot_world_built(
        dev.universe_seed,
        &held,
        vd_core::pose::RealmId::System(7),
        dev.move_speed,
        dev.tick_dt,
        &held,
        &[(vd_core::pose::RealmId::System(7), berth)],
        None,
    );
    for r in &regions {
        if matches!(r.realm, vd_core::pose::RealmId::Ship(_)) {
            println!(
                "WAKE spin_up={} tear_down={}",
                r.aoi.spin_up_r_m(),
                r.aoi.tear_down_r_m()
            );
        }
    }
}
