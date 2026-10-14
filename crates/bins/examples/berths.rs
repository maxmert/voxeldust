//! ★ THE BERTHS A REALM'S STORE HOLDS (2026-10-14, B5's first measurement): every built child the realm parked, its
//! berth in the realm's own frame, its distance from the centre and its facing — read only, so a bench can stand at a
//! parked hull without a running cluster.
//!
//! **Example.** On the home planet's store: hull 17 on the day-side hills and hull 19 on the valley floor, each with its
//! berth `X,Y,Z` for `fine_steps` or `valley_hull`.
//!
//! `VD_REALM_STORE=<a copy of a realm's store> cargo run --release -p vd-bins --example berths`

fn main() {
    let env = vd_bins::process_env();
    let realm = vd_core::pose::RealmId::Planet(0);
    let store = vd_bins::open_realm_store(&env, realm)
        .expect("the store opens")
        .expect("VD_REALM_STORE names a file");
    let (berths, body) = vd_bins::read_realm_store(&store).expect("the rows decode");
    println!(
        "body: {}",
        body.map_or("none".into(), |b| format!("{:?}", b.realm))
    );
    for b in &berths {
        let o = b.offset_m;
        println!(
            "{:?}: {},{},{}  r {:.1} m  facing {},{},{},{}",
            b.child,
            o.x,
            o.y,
            o.z,
            o.length(),
            b.facing.x,
            b.facing.y,
            b.facing.z,
            b.facing.w
        );
    }
}
