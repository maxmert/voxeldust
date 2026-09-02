//! Diagnosis probe: print the launch rows the orchestrator recorded under a kept fixture dir.
fn main() {
    let dir = std::env::args().nth(1).expect("the kept fixture dir");
    let trust = format!("{dir}/trust");
    for (k, v) in vd_bins::common_env(&trust, &vd_bins::DEV) {
        // SAFETY: single-threaded probe, before anything reads the environment.
        unsafe { std::env::set_var(k, v) };
    }
    let rows = vd_bins::launch_rows(std::path::Path::new(&format!("{dir}/launch.redb")));
    for (node, coord, pid) in rows {
        println!("node={} coord={coord:?} pid={pid:?}", node.0);
    }
}
