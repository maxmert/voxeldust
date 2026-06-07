//! The crate-graph isolation gate (`docs/design/sealed_shards.md` §1; audit SEAL-2).
//!
//! HR1 sealing held BY MANUAL AUDIT before this gate existed; now it is enforced.
//! The dependency graph is parsed from `cargo metadata` and asserted: Tier-A
//! simulation crates cannot reach the async/network world (quinn/tokio/rustls/redb)
//! or the production I/O crate, and the gateway-only ticket crypto is unreachable
//! from shard simulation code. A violation of these edges is the structural way the
//! old project leaked shard state — this test makes re-introducing one a build break.

use std::collections::{BTreeMap, BTreeSet};
use std::process::Command;

/// Parse the workspace dependency graph: package name -> set of direct dependency
/// package names (normal + build, NOT dev — dev edges are test-only and don't ship).
fn dependency_graph() -> BTreeMap<String, BTreeSet<String>> {
    let output = Command::new(env!("CARGO"))
        .args(["metadata", "--format-version", "1", "--no-deps"])
        .output()
        .expect("cargo metadata runs");
    assert!(output.status.success(), "cargo metadata failed: {output:?}");
    let meta: serde_json::Value =
        serde_json::from_slice(&output.stdout).expect("metadata is JSON");

    let mut graph = BTreeMap::new();
    for pkg in meta["packages"].as_array().expect("packages array") {
        let name = pkg["name"].as_str().expect("package name").to_owned();
        let mut deps = BTreeSet::new();
        for dep in pkg["dependencies"].as_array().expect("deps array") {
            let kind = dep["kind"].as_str(); // null = normal, "build", "dev"
            if matches!(kind, Some("dev")) {
                continue; // dev-deps are test-only and never ship
            }
            deps.insert(dep["name"].as_str().expect("dep name").to_owned());
        }
        graph.insert(name, deps);
    }
    graph
}

/// The async/network/persistence world — none of it may be a NORMAL dependency of a
/// Tier-A simulation crate.
const FORBIDDEN_IN_TIER_A: &[&str] = &[
    "quinn",
    "tokio",
    "rustls",
    "rcgen",
    "redb",
    "crossbeam-channel",
    "axum",
    "vd-io-prod",
];

/// The crates that must stay pure (reachable I/O only through the injected seam).
const TIER_A: &[&str] = &[
    "vd-core",
    "vd-wire",
    "vd-sim",
    "vd-node",
    "vd-connection-plane",
    "vd-harness",
];

#[test]
fn tier_a_crates_never_depend_on_the_async_network_world() {
    let graph = dependency_graph();
    for crate_name in TIER_A {
        let deps = graph
            .get(*crate_name)
            .unwrap_or_else(|| panic!("{crate_name} present in the workspace"));
        for forbidden in FORBIDDEN_IN_TIER_A {
            assert!(
                !deps.contains(*forbidden),
                "HR1 VIOLATION: {crate_name} depends on {forbidden} — Tier-A code must \
                 reach I/O ONLY through the sim::io seam (io-prod is the sole impl)"
            );
        }
    }
}

#[test]
fn gateway_ticket_crypto_is_unreachable_from_shard_simulation() {
    // The ticket crypto (Ed25519/HMAC) lives in vd-connection-plane; vd-sim/vd-node
    // (shard simulation) must NOT depend on it, so a shard validating a client ticket
    // is structurally impossible (R7: the old triple-overloaded token cannot recur).
    let graph = dependency_graph();
    for shard_crate in ["vd-sim", "vd-node"] {
        let deps = &graph[shard_crate];
        assert!(
            !deps.contains("vd-connection-plane"),
            "R7 VIOLATION: {shard_crate} depends on vd-connection-plane — a shard could \
             reach the gateway-only ticket validator"
        );
        for crypto in ["ed25519-dalek", "hmac"] {
            assert!(
                !deps.contains(crypto),
                "R7 VIOLATION: {shard_crate} depends on {crypto} — credential crypto is \
                 gateway-only"
            );
        }
    }
}

#[test]
fn the_dependency_law_holds_bins_to_node_to_sim_to_wire_to_core() {
    // The layering rule: core depends on nothing internal; wire only on core; sim on
    // wire+core; node on sim+wire+core. Lower layers never depend up.
    let graph = dependency_graph();
    let internal = |deps: &BTreeSet<String>| -> BTreeSet<String> {
        deps.iter()
            .filter(|d| d.starts_with("vd-"))
            .cloned()
            .collect()
    };
    assert!(
        internal(&graph["vd-core"]).is_empty(),
        "vd-core is the root: it depends on no workspace crate"
    );
    assert_eq!(
        internal(&graph["vd-wire"]),
        BTreeSet::from(["vd-core".to_owned()]),
        "vd-wire depends only on core"
    );
    let sim = internal(&graph["vd-sim"]);
    assert!(sim.contains("vd-core") && sim.contains("vd-wire"));
    assert!(
        !sim.contains("vd-node") && !sim.contains("vd-connection-plane"),
        "vd-sim never depends UP the stack"
    );
}
