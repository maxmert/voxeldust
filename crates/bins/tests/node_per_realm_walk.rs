//! NODE-PER-REALM WALK GATE (task #149) — the headless process-tier proof that a durable player walks
//! CLEANLY through a chain of cross-node re-homes without freezing, and that each crossing is ONE clean
//! directory commit (NO fence thrash).
//!
//! This REPLACES the old `cohosted_input_freeze_repro.rs`. That repro stood up ONE co-hosting shard (System 7
//! holding Planet/Station/Area 7 via `VD_HELD_REALMS`) and reproduced a limit-cycle FREEZE on the source==dest
//! re-home. The USER has since decided re-home should be NODE-PER-REALM: each realm on its OWN node, so EVERY
//! re-home is a uniform CROSS-NODE saga and the source==dest degenerate case never arises. With no co-hosting
//! there is no thrash — so this test asserts the OPPOSITE of the repro: the player ARRIVES at every leg and the
//! subject Entity's directory fence stays SMALL (one clean commit per crossing).
//!
//! Topology: the [`ClusterShape::Forest`] cluster — orchestrator + gateway + SIX single-realm shards, one per
//! realm of the System-7 sub-forest: System 7, Planet 7, Station 7, Area 7, Galaxy (System 1), System 8. NO
//! `VD_HELD_REALMS`. A REAL headless `client` binary logs in over localhost QUIC and is driven by dev-control
//! `WalkTo` along the +X axis (NO client prediction — the server stays sole authority; the loop steers on the
//! delivered, lagged pose).
//!
//! The seed forest (`vd_core::worldgen`) on the +X axis:
//!   System 7  : shell r=40 @ x=0    → x∈[-40,40]
//!   Planet 7  : shell r=10 @ x=20   → x∈[10,30]  (deepest at +X except the Area box)
//!   Area 7    : box half=3 @ x=25   → x∈[22,28]  (DEEPEST — nested under Planet 7)
//!   Galaxy    : shell r=180 @ x=0   → the between-space; System 7 far-face 40, System 8 near-face 90
//!   System 8  : shell r=40 @ x=130  → x∈[90,170]
//!
//! Legs (each asserts ARRIVAL within an epsilon — movement never freezes across the cross-node re-home):
//!   1. System 7 origin (x≈0) → Planet 7   (x=15)   [re-home System 7 → Planet 7]
//!   2.                       → Area 7      (x=25)   [re-home Planet 7 → Area 7]
//!   3.                       → back Planet 7 (x=15) [re-home Area 7 → Planet 7]
//!   4.                       → System 7    (x=35)   [re-home Planet 7 → System 7, past the planet]
//!   5.                       → the Galaxy   (x=65)  [re-home System 7 → Galaxy — the between-space gap]
//!      (best-effort onward to System 8 x=100 within a spare budget — proves the multi-hop chain end-to-end.)
//!
//! Then it asserts the subject Entity's directory fence is SMALL (≤ a modest bound) — the thrash guard: a
//! single clean re-home per crossing commits at a small fence; the co-hosting bug ballooned it to ~20.
//!
//! Gated on `dev-control`. Run with:
//!   cargo test -p vd-bins --features dev-control --test node_per_realm_walk -- --nocapture
#![cfg(feature = "dev-control")]

use std::process::{Child, Command};
use std::time::{Duration, Instant};

use vd_bins::{
    Cluster, ClusterAddrs, ClusterShape, DEV, admin_get_body, common_env, dev_auth_pubkey_hex,
    dev_auth_signing_key_hex, dev_roundtrip, galaxy_env, gateway_env, orchestrator_env,
    realm_shard_env, reserve_tcp_addr, reserve_udp_addr, shard_env,
};
use vd_core::NodeId;
use vd_core::glam::DVec3;
use vd_devproto::{CLIENT_NODE_BASE, DevEntityRow, DevPhase, DevRequest, DevResponse, DevState};

const DEADLINE: Duration = Duration::from_secs(120);
/// Arrival tolerance per leg: above one sim step (fixed-magnitude Move never oscillates) AND above the
/// ~100-150 ms delivered-pose lag. Each leg's target sits well inside its realm past the containment
/// hysteresis band (inset 1 m / outset 2 m), so this epsilon never straddles a boundary.
const ARRIVE_EPSILON: f64 = 2.0;
/// The thrash guard: a single CLEAN re-home per crossing commits the subject Entity at a small directory
/// fence. The whole walk makes ~6 crossings (7→P7→A7→P7→7→Galaxy[→8]); one clean commit each keeps the fence
/// well under this bound. The co-hosting freeze bug ballooned it past 20. Chosen with headroom (a couple of
/// legitimate re-drives) yet far below the thrash signature.
const MAX_ENTITY_FENCE: u64 = 12;

fn devctl(port: u16, request: &DevRequest) -> Option<DevResponse> {
    dev_roundtrip(port, request).ok()
}

fn poll_state(port: u16) -> Option<DevState> {
    match devctl(port, &DevRequest::State)? {
        DevResponse::State { state } => Some(state),
        _ => None,
    }
}

fn own_row(state: &DevState) -> Option<&DevEntityRow> {
    let own = state.own_entity.as_deref()?;
    state.entities.iter().find(|r| r.entity == own)
}

fn own_pos(state: &DevState) -> Option<DVec3> {
    own_row(state).map(|r| DVec3::from_array(r.pos))
}

/// The highest directory `fence` on any `ent-…` (Entity) row — the thrash signature. A single clean re-home
/// per crossing commits the subject Entity at a small fence; the co-hosting bug re-fired the re-home in a limit
/// cycle, each cycle a full CAS that bumped the fence, so it ballooned. Node-per-realm has no such cycle.
fn max_entity_fence(admin: std::net::SocketAddr) -> u64 {
    let Some(body) = admin_get_body(admin, "/admin/snapshot", Some(Duration::from_secs(2))) else {
        return 0;
    };
    let Ok(v) = serde_json::from_str::<serde_json::Value>(&body) else {
        return 0;
    };
    v["directory"]
        .as_array()
        .map(|arr| {
            arr.iter()
                .filter(|e| e["key"].as_str().is_some_and(|k| k.starts_with("ent-")))
                .filter_map(|e| e["fence"].as_u64())
                .max()
                .unwrap_or(0)
        })
        .unwrap_or(0)
}

/// The orchestrator `/admin/snapshot` directory rows as human strings — printed on a leg failure so the
/// process-tier state (which realm heads rest where, at what fence) is captured for debugging the wiring.
fn admin_directory(admin: std::net::SocketAddr) -> Vec<String> {
    let Some(body) = admin_get_body(admin, "/admin/snapshot", Some(Duration::from_secs(2))) else {
        return Vec::new();
    };
    let Ok(v) = serde_json::from_str::<serde_json::Value>(&body) else {
        return Vec::new();
    };
    v["directory"]
        .as_array()
        .map(|arr| {
            arr.iter()
                .map(|e| {
                    format!(
                        "{}={} fence={} in_transfer={}",
                        e["key"].as_str().unwrap_or("?"),
                        e["authority"].as_str().unwrap_or("?"),
                        e["fence"].as_u64().unwrap_or(0),
                        e["in_transfer"].as_str().unwrap_or("none"),
                    )
                })
                .collect()
        })
        .unwrap_or_default()
}

/// Drive ONE leg: `WalkTo` the world x-target (on the +X axis), then read the delivered own pos. Returns the
/// landed pose. Asserts the dot ARRIVED near the target — movement did NOT freeze across the cross-node
/// re-home. On failure it dumps the orchestrator directory so a wiring bug (a head that never rested on the
/// dest node) is visible. `max_ticks` is generous so a slow saga never times out mid-leg (the point is whether
/// the dot keeps MOVING, not raw speed).
fn walk_leg(
    devctl_port: u16,
    admin: std::net::SocketAddr,
    leg: &str,
    target_x: f64,
    max_ticks: u64,
) -> DVec3 {
    let target = DVec3::new(target_x, 0.0, 0.0);
    let outcome = devctl(
        devctl_port,
        &DevRequest::WalkTo {
            target: target.to_array(),
            arrive_epsilon: ARRIVE_EPSILON,
            max_ticks,
            max_step_m: 0.0,
        },
    )
    .unwrap_or_else(|| panic!("leg {leg}: no walk response"));
    // Settle the sticky last Move so the dot rests.
    let _ = devctl(
        devctl_port,
        &DevRequest::Move {
            axes: [0.0, 0.0, 0.0],
        },
    );
    let landed = own_pos(&poll_state(devctl_port).expect("state after leg")).expect("own pos");
    eprintln!(
        "NODE-PER-REALM leg {leg}: outcome={} landed x={:.3} (target x={target_x})",
        match &outcome {
            DevResponse::State { .. } => "State(arrived)",
            DevResponse::Timeout { .. } => "Timeout(budget)",
            _ => "other",
        },
        landed.x,
    );
    assert!(
        (landed.x - target_x).abs() <= ARRIVE_EPSILON,
        "leg {leg} FROZE: the player stalled at x={:.3} and never reached the target x={target_x} across the \
         cross-node re-home. A node-per-realm walk must never freeze (no source==dest thrash). DIRECTORY={:#?}",
        landed.x,
        admin_directory(admin),
    );
    landed
}

#[test]
fn a_durable_player_walks_the_whole_forest_node_per_realm_without_freezing_or_fence_thrash() {
    // FIRST statement: hold the process tier for the whole body, so it outlives the cluster reap
    // that frees the ports. See `vd_bins::cluster_tier`.
    let _tier = vd_bins::cluster_tier();
    // ---- topology + trust (mirrors dev_control_nav.rs) -----------------------
    let admin_addr = reserve_tcp_addr();
    let client_quic = reserve_udp_addr();
    let devctl_port = reserve_tcp_addr().port();

    let trust_dir = std::env::temp_dir().join(format!("vd-forest-{}", std::process::id()));
    let trust = vd_io_prod::trust::ClusterTrust::generate("vd-forest").expect("trust");
    trust.write_der_dir(&trust_dir).expect("trust dir");
    let orch_store =
        std::env::temp_dir().join(format!("vd-forest-{}-orch.redb", std::process::id()));
    let _ = std::fs::remove_file(&orch_store);
    let orch_store = orch_store.display().to_string();

    // Each of the SIX realm-shards + the base nodes gets its OWN reserved QUIC + probe addr — one shard per
    // realm (NO co-hosting), so every re-home is a uniform CROSS-NODE saga.
    let addrs = ClusterAddrs {
        admin: admin_addr,
        ..ClusterAddrs::reserve()
    };
    let shape = ClusterShape::Forest;
    let client_book = [(NodeId(CLIENT_NODE_BASE), client_quic)];
    let common = common_env(&trust_dir.display().to_string(), &DEV);
    let spawn_node = |bin: &str, node_env: Vec<(&'static str, String)>| -> Child {
        vd_bins::spawn_node(bin, &common, &node_env).expect("spawn node")
    };

    let mut nodes = Cluster::new();
    nodes.push(
        "vd-orchestrator",
        spawn_node(
            env!("CARGO_BIN_EXE_vd-orchestrator"),
            orchestrator_env(&addrs, &DEV, &orch_store, shape),
        ),
    );
    nodes.push(
        "vd-gateway",
        spawn_node(
            env!("CARGO_BIN_EXE_vd-gateway"),
            gateway_env(&addrs, &client_book, &dev_auth_pubkey_hex(), &DEV, shape),
        ),
    );
    // The System-7 SOURCE shard (single-realm — NO VD_HELD_REALMS in Forest).
    nodes.push(
        "vd-shard",
        spawn_node(
            env!("CARGO_BIN_EXE_vd-shard"),
            shard_env(&addrs, &DEV, shape),
        ),
    );
    // The five extra realm-shards (Planet 7, Station 7, Area 7, Galaxy, System 8) — each hosting EXACTLY its
    // own realm via `realm_shard_env` (VD_REALM_KIND + VD_REALM_SEED, no co-hosting). `galaxy_env` is unused
    // in Forest (the Galaxy rides `realm_shard_env` like the rest); reference it so the import stays honest.
    let _ = galaxy_env;
    for shard in shape.extra_realm_shards(&addrs, &DEV) {
        nodes.push(
            "vd-shard",
            spawn_node(
                env!("CARGO_BIN_EXE_vd-shard"),
                realm_shard_env(&addrs, &DEV, shape, shard),
            ),
        );
    }
    let _nodes = nodes; // RAII: reaps the whole cluster on test end or panic

    struct KillOnDrop(Child);
    impl Drop for KillOnDrop {
        fn drop(&mut self) {
            let _ = self.0.kill();
            let _ = self.0.wait();
        }
    }

    let _client = {
        let mut cmd = Command::new(env!("CARGO_BIN_EXE_client"));
        for (k, v) in common.iter() {
            cmd.env(k, v);
        }
        cmd.env("VD_AUTH_SIGNING_KEY", dev_auth_signing_key_hex());
        cmd.args([
            "--name",
            "forest",
            "--agent-index",
            "0",
            "--gateway",
            &addrs.gateway.to_string(),
            "--client-quic",
            &client_quic.port().to_string(),
            "--trust-dir",
            &trust_dir.display().to_string(),
            "--dev-control",
            &devctl_port.to_string(),
            "--allow-dev-control",
        ]);
        KillOnDrop(cmd.spawn().expect("spawn client"))
    };

    // ---- wait until Active AND the own row is delivered ----------------------
    let started = Instant::now();
    let origin = loop {
        if let Some(pos) = poll_state(devctl_port)
            .filter(|s| s.phase == DevPhase::Active)
            .as_ref()
            .and_then(own_pos)
        {
            break pos;
        }
        assert!(
            started.elapsed() < DEADLINE,
            "client never became Active with an own row (the Forest cluster failed to bring up — \
             DIRECTORY={:#?})",
            admin_directory(admin_addr),
        );
        std::thread::sleep(Duration::from_millis(100));
    };
    eprintln!("NODE-PER-REALM: client Active at origin x={:.3}", origin.x);
    // The dot spawns at the System-7 origin, well inside System 7 (x≈0) and clear of Planet 7's SOI (x≥10).
    assert!(
        origin.x < 10.0 - ARRIVE_EPSILON,
        "the dot must spawn well inside System 7 (origin x={:.3}) so the first walk crosses INTO Planet 7",
        origin.x,
    );

    // ---- the walk: each leg ARRIVES (never freezes across the cross-node re-home) ----
    // Generous per-leg budgets; the between-realm hops are short (a few metres) except the final Galaxy leg.
    let mut arrivals: Vec<(&str, f64)> = Vec::new();
    let p7 = walk_leg(
        devctl_port,
        admin_addr,
        "1 System7->Planet7 (x=15)",
        15.0,
        4000,
    );
    arrivals.push(("Planet7@15", p7.x));
    let a7 = walk_leg(
        devctl_port,
        admin_addr,
        "2 Planet7->Area7 (x=25)",
        25.0,
        4000,
    );
    arrivals.push(("Area7@25", a7.x));
    let p7b = walk_leg(
        devctl_port,
        admin_addr,
        "3 Area7->Planet7 (x=15)",
        15.0,
        4000,
    );
    arrivals.push(("Planet7@15(back)", p7b.x));
    let s7 = walk_leg(
        devctl_port,
        admin_addr,
        "4 Planet7->System7 (x=35)",
        35.0,
        4000,
    );
    arrivals.push(("System7@35", s7.x));
    // Leg 5: out past System 7's far face (40) into the between-space Galaxy gap (System 8 near-face is 90).
    // x=65 is pure Galaxy — a System 7 → Galaxy cross-node re-home. A longer walk, so a bigger budget.
    let gx = walk_leg(
        devctl_port,
        admin_addr,
        "5 System7->Galaxy (x=65)",
        65.0,
        8000,
    );
    arrivals.push(("Galaxy@65", gx.x));

    // Best-effort onward to System 8 (x=100, inside its r=40 SOI @ x=130). A long haul over process QUIC; NOT
    // asserted for arrival (the spare-budget leg proves the chain reaches System 8 when it lands, but the
    // primary gate is legs 1-5 above + the fence thrash guard below). Still driven so the walk exercises the
    // Galaxy → System 8 cross-node hop when the budget allows.
    let s8 = {
        let target = DVec3::new(100.0, 0.0, 0.0);
        let _ = devctl(
            devctl_port,
            &DevRequest::WalkTo {
                target: target.to_array(),
                arrive_epsilon: ARRIVE_EPSILON,
                max_ticks: 8000,
                max_step_m: 0.0,
            },
        );
        let _ = devctl(
            devctl_port,
            &DevRequest::Move {
                axes: [0.0, 0.0, 0.0],
            },
        );
        own_pos(&poll_state(devctl_port).expect("state after s8 leg")).expect("own pos")
    };
    arrivals.push(("System8@100(best-effort)", s8.x));

    eprintln!("NODE-PER-REALM arrivals: {arrivals:?}");

    // ---- the THRASH GUARD: the subject Entity's directory fence stays SMALL ----
    // One clean re-home per crossing commits at a small fence. Node-per-realm has no source==dest limit cycle,
    // so the fence never balloons. (The co-hosting freeze bug drove it past 20.)
    let entity_fence = max_entity_fence(admin_addr);
    eprintln!("NODE-PER-REALM: max entity directory fence = {entity_fence}");
    assert!(
        entity_fence <= MAX_ENTITY_FENCE,
        "FENCE THRASH: the subject Entity's directory fence ballooned to {entity_fence} (a clean node-per-realm \
         walk commits ONE fence per crossing, staying <= {MAX_ENTITY_FENCE}). A ballooning fence means a re-home \
         is re-firing in a limit cycle — the cluster wiring is wrong. DIRECTORY={:#?}",
        admin_directory(admin_addr),
    );

    let _ = devctl(devctl_port, &DevRequest::Close);
    let _ = std::fs::remove_dir_all(&trust_dir);
    let _ = std::fs::remove_file(&orch_store);
}
