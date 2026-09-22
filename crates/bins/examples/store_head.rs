//! ★ WHAT A REALM'S STORE HOLDS (2026-09-22, the on-foot ground): the artifact head's presence and
//! the tile rows in one realm store file, so a step that touched the file can be held against
//! what it left behind. `VD_REALM_STORE=<file> cargo run --release -p vd-bins --example store_head`

use vd_sim::io::Store;
use vd_sim::stub::built_store::{artifact_head_key, artifact_tile_prefix};

fn main() {
    let env = vd_bins::process_env();
    let store = vd_bins::open_realm_store(&env, vd_core::pose::RealmId::Planet(0))
        .expect("the store opens")
        .expect("VD_REALM_STORE names a file");
    let head = store.get(&artifact_head_key());
    let tiles = store.scan(&artifact_tile_prefix()).len();
    println!(
        "artifact head {}, tile rows {tiles}",
        if head.is_some() { "PRESENT" } else { "ABSENT" }
    );
}
