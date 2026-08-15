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
    let meta: serde_json::Value = serde_json::from_slice(&output.stdout).expect("metadata is JSON");

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
    "vd-physics",
    "vd-wire",
    "vd-sim",
    "vd-node",
    "vd-connection-plane",
    "vd-harness",
];

/// SL4: physics and re-home are separate machinery, one-way. Exactly these crates may name a motion —
/// i.e. carry a NORMAL dependency edge to `vd-physics` — each for a stated reason. Everything else is
/// a placement CONSUMER: it reads authored rows (and runs opaque injected `MotionFn`s) and must not be
/// ABLE to ask how a thing moves — an orbit symbol on the crossing path is an unresolved-crate compile
/// error, not a review finding. (Dev-dependency edges are exempt BY THE GRAPH PARSER, deliberately:
/// fixtures may plant a moving child and keep every branch covered; dev deps never ship.)
const MAY_NAME_MOTION: &[(&str, &str)] = &[
    (
        "vd-bins",
        "the composition root: builds the motion roster, runs the boot fences, injects MotionFns",
    ),
    // vd-connection-plane WAS allowlisted here ("the gateway resolves logins against THE world").
    // DELETED (batch review, MAJOR): the gateway only ever QUERIED the world — regions, the
    // neighbourhood scope, realm membership — so it now RECEIVES the lowered
    // `vd_core::worldgen::WorldRealms` from the composition root and carries a vd-physics edge only
    // in [dev-dependencies] (fixtures build worlds; the shipped library cannot). The universal
    // assert below covers the routing plane again — an orbit symbol there is a build break, not a
    // review finding. OBSERVED FAILING for this crate too (2026-08-14): with
    // `vd-physics = { workspace = true }` restored to vd-connection-plane's `[dependencies]`, this
    // test failed naming the crate, and passed again once the edge was removed.
    (
        "vd-tests",
        "the scenario library builds THE world's fixtures; a test-only crate nothing ships or depends on",
    ),
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
fn sl4_the_crossing_path_cannot_name_a_motion() {
    // THE STRUCTURAL HALF OF SL4 (audit finding 26; standing law: "Enforce structurally (a module/
    // crate dependency rule), not by care"). The crossing/containment path — core's frame/geometry
    // math, the wire, the sim that runs the detector and the flush, the node runtime, the client and
    // the harness — must not be able to NAME how anything moves: no normal edge to vd-physics, so
    // `use vd_physics::celestial::OrbitalElements` (the exact import the sim used to carry) is E0432.
    //
    // OBSERVED FAILING (the standing ground rule: a gate never observed failing is an argument, not a
    // measurement): with `vd-physics = { workspace = true }` added to vd-sim's `[dependencies]`, this
    // test failed with the message below (2026-08-14), and passed again once the edge was removed.
    let graph = dependency_graph();
    let allowed: BTreeSet<&str> = MAY_NAME_MOTION.iter().map(|(name, _)| *name).collect();
    for (name, deps) in &graph {
        if !name.starts_with("vd-") || allowed.contains(name.as_str()) || name == "vd-physics" {
            continue;
        }
        assert!(
            !deps.contains("vd-physics"),
            "SL4 VIOLATION: {name} carries a NORMAL dependency on vd-physics — the crossing path can \
             name a motion. Physics produces placements; consumers read the authored rows. If this \
             crate has a stated reason to hold motion, add it to MAY_NAME_MOTION with that reason."
        );
    }
    // The allowlist itself must not rot: every allowlisted crate exists in the workspace.
    for (name, reason) in MAY_NAME_MOTION {
        assert!(
            graph.contains_key(*name),
            "MAY_NAME_MOTION names {name} ({reason}) but the workspace has no such crate"
        );
    }
}

#[test]
fn the_window_composer_cannot_name_a_motion_or_generate_a_world() {
    // THE WINDOW LANE's structural guard 1 (docs/design/window_lane.md §2.6.1, the T1
    // resolution — build-level, not care): the gateway's composition engine
    // (`vd-connection-plane::window`) folds ATTESTED statements through the frame core and must
    // be STRUCTURALLY UNABLE to evaluate a placement or generate a world — so the crate that
    // carries it may hold NO normal edge to `vd-physics`. The universal SL4 sweep above already
    // covers every crate; this pin names the COMPOSER's crate specifically so the guard cannot
    // be silently dissolved by allowlisting vd-connection-plane into MAY_NAME_MOTION for some
    // other reason: the composer's crate must ALSO never appear there.
    //
    // (Dev-dependency edges are exempt by the graph parser, deliberately: the crate's fixtures
    // build worlds; the shipped library cannot — the same posture as the SL4 sweep.)
    let graph = dependency_graph();
    let deps = &graph["vd-connection-plane"];
    assert!(
        !deps.contains("vd-physics"),
        "WINDOW-LANE §2.6.1 VIOLATION: vd-connection-plane (the composer's crate) carries a \
         NORMAL vd-physics dependency — the gateway could evaluate motion/worldgen on the \
         composition path. The composer folds attested statements only; keep vd-physics in \
         [dev-dependencies]."
    );
    assert!(
        !MAY_NAME_MOTION
            .iter()
            .any(|(name, _)| *name == "vd-connection-plane"),
        "WINDOW-LANE §2.6.1 VIOLATION: vd-connection-plane was allowlisted into MAY_NAME_MOTION \
         — the composer's crate may never hold a stated reason to name a motion."
    );
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
    // The motion crate sits beside wire: on core, and on nothing else internal (SL4's producer half —
    // physics may not reach into the machinery it produces placements for).
    assert_eq!(
        internal(&graph["vd-physics"]),
        BTreeSet::from(["vd-core".to_owned()]),
        "vd-physics depends only on core"
    );
}
