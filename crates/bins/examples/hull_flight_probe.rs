//! Fly the dev cluster's berthed hull into a drawn realm through the dev-control seam — the hull
//! flight helper's probe. `cargo run --release -p vd-bins --features dev-control --example
//! hull_flight_probe -- <devctl port> <target label prefix, e.g. Planet> [deadline s]`.
fn main() {
    let port: u16 = std::env::args()
        .nth(1)
        .expect("devctl port")
        .parse()
        .expect("port");
    let target = std::env::args()
        .nth(2)
        .unwrap_or_else(|| "Planet".to_owned());
    let deadline: u64 = std::env::args()
        .nth(3)
        .map_or(300, |s| s.parse().expect("seconds"));
    let aboard =
        vd_bins::flight::board_hull(port, &vd_bins::DEV, std::time::Duration::from_secs(120));
    println!("aboard: {:?}", aboard.location);
    let arrived = vd_bins::flight::fly_hull_to(
        port,
        &target,
        0.5,
        0.3,
        std::time::Duration::from_secs(deadline),
    );
    println!(
        "arrived: {:?} at tick {:?}",
        arrived.location, arrived.universe_tick
    );
}
