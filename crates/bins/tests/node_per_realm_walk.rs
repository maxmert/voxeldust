//! NODE-PER-REALM WALK GATE (task #149) — the headless process-tier proof that a durable player
//! flies CLEANLY through a chain of cross-node re-homes on THE WORLD ITSELF, without freezing and
//! without fence thrash.
//!
//! Topology: the [`ClusterShape::Chain`] cluster — orchestrator + gateway + FOUR realm-shards, one
//! per realm of THE world's flight chain, every name derived through `world_roster` (never stated):
//! the HOME system, the GALAXY between-space (its parent), the home's INNER planet (smallest
//! semi-major axis), and the galaxy's lowest-seed ring SIBLING star. The retired walk fixture named
//! realms THE world does not contain (Planet/Station/Area 7, System 8), so four of its six shards
//! died at boot — invisibly, because bring-up never polled `Cluster::first_exited` — and every leg
//! target sat INSIDE the home system's 150 m shell, so a green run proved no crossing at all
//! (ledgered D-WORLD-7; the Station/Area legs return when the block/station slice grows THE world,
//! D-WORLD-1).
//!
//! THE FLIGHT LAW (the clusters plan §2.3): every leg flies the ±Z POLAR corridor — orbits lie near
//! the XY plane and the star ring near XZ, so ±Z is clear of both by the I-AXIS/I-POLE/I-RADIAL
//! margins `world_roster` asserts at derivation. The legs, in order (C, D, A, E, F):
//!
//!   C  home → inner planet   the shared rendezvous-and-park (`vd_bins::flight`), full orbit speed
//!   D  inner planet → home   planet-frame (0,0,−2·pole_altitude) — the polar lift, I-RADIAL-licensed
//!   A  home → galaxy         (0,0,−220) — out the pole, past the ~152 m release edge
//!   E  galaxy → sibling      waypoint sibling_centre+(0,0,−300), then a held-throttle CREEP up
//!                            the pole — the 12.8 km boost leg, clear of every sibling planet by
//!                            the SIBLING's own I-AXIS/I-POLE margins, which `world_roster` now
//!                            ASSERTS over the sibling's movers too (batch review: this clearance
//!                            was a bare comment); no coordinate in flight across the commit
//!   F  sibling → galaxy      (0,0,−300) in the sibling's own frame — back out the pole
//!
//! EVERY leg asserts the REALM LABEL reached (`FrameRef::label` via the delivered `location`), never
//! a coordinate: at a crossing the pose reframes and a target stated in the old frame is
//! meaningless. Then the TWO-SIDED thrash guard: the subject Entity's directory fence stays SMALL
//! (one clean CAS commit per crossing) AND at least one commit per crossing was actually recorded
//! (the floor — a zero-commit run or an admin blink used to satisfy the one-sided ceiling); the
//! observed maximum is PRINTED every run — the exact ceiling is UNMEASURED for this chain until a
//! green history accumulates.
//!
//! Gated on `dev-control`. Run with:
//!   cargo test -p vd-bins --features dev-control --test node_per_realm_walk -- --test-threads=1 --nocapture
#![cfg(feature = "dev-control")]

use std::process::{Child, Command};
use std::time::{Duration, Instant};

use vd_bins::flight::{creep_into, cross_leg, rendezvous_into_planet};
use vd_bins::{
    Cluster, ClusterAddrs, ClusterShape, DEV, GALAXY, PLANET_SHARD, admin_get_body, common_env,
    dev_auth_pubkey_hex, dev_auth_signing_key_hex, dev_roundtrip, gateway_env, orchestrator_env,
    realm_shard_env, realm_shards, reserve_tcp_addr, reserve_udp_addr, shard_env, world_roster,
};
use vd_core::NodeId;
use vd_core::glam::DVec3;
use vd_core::pose::frame_for_realm;
use vd_devproto::{CLIENT_NODE_BASE, DevEntityRow, DevPhase, DevRequest, DevResponse, DevState};

/// The whole gate's budget: bring-up (6 real processes) + login + the rendezvous (the slowest,
/// least deterministic leg — proven ≤ ~60 s at full orbit speed) + four corridor legs. Was 120 s
/// for the old inert walk; the real chain flies real distances.
const DEADLINE: Duration = Duration::from_secs(300);
/// Per-crossing-leg budget (the label flip, not a coordinate): the corridor legs are seconds of
/// flight plus the saga tail; generous so a slow commit never times a healthy leg out.
const LEG_DEADLINE: Duration = Duration::from_secs(60);
/// The thrash guard's CEILING: a single CLEAN re-home per crossing commits the subject Entity at a
/// small directory fence. The chain makes [`CROSSINGS`] crossings (C,D,A,E,F); one clean commit
/// each keeps the fence well under this bound (the co-hosting freeze bug ballooned it past 20).
/// The exact ceiling is UNMEASURED for the new chain until a green history accumulates — the
/// observed value is PRINTED every run; tighten later.
const MAX_ENTITY_FENCE: u64 = 12;
/// The label-asserted crossings the chain flies — and therefore the thrash guard's FLOOR: every
/// crossing commits the subject at a STRICTLY newer fence, so a run whose five label flips were
/// real leaves the fence at least this high. The guard used to be one-sided (batch review): a run
/// in which the directory recorded ZERO entity commits — or an admin blink read as 0 — satisfied
/// "one clean CAS commit per crossing" without a single commit observed.
const CROSSINGS: u64 = 5;

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

/// The highest directory `fence` on any `ent-…` (Entity) row — the thrash signature. A single clean
/// re-home per crossing commits the subject Entity at a small fence; a re-home re-firing in a limit
/// cycle bumps it per cycle, so it balloons. An unreachable or unparseable admin endpoint PANICS
/// (batch review: it used to return 0, which silently satisfied the one-sided guard — the guard
/// self-disabled on exactly the blink it should have surfaced).
fn max_entity_fence(admin: std::net::SocketAddr) -> u64 {
    let body = admin_get_body(admin, "/admin/snapshot", Some(Duration::from_secs(2)))
        .unwrap_or_else(|| {
            panic!(
                "the thrash guard could not read /admin/snapshot at {admin} — an admin blink \
                     is a loud failure, never a fence of 0"
            )
        });
    let v = serde_json::from_str::<serde_json::Value>(&body)
        .expect("the admin snapshot parses as JSON");
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

/// The orchestrator `/admin/snapshot` directory rows as human strings — printed on a failure so the
/// process-tier state (which realm heads rest where, at what fence) is captured for debugging.
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

/// A plain in-frame ROUTE step (no crossing, no label flip expected): WalkTo `target` in the space
/// the session currently stands in, then settle the sticky Move. Used only for leg E's approach
/// waypoint — the crossing itself is always a [`cross_leg`] with a label.
fn fly_waypoint(devctl_port: u16, leg: &str, target: DVec3, max_ticks: u64) {
    let outcome = devctl(
        devctl_port,
        &DevRequest::WalkTo {
            target: target.to_array(),
            arrive_epsilon: 2.0,
            max_ticks,
            max_step_m: 4.0 * DEV.move_speed * DEV.tick_dt,
        },
    )
    .unwrap_or_else(|| panic!("waypoint {leg}: no walk response"));
    assert!(
        matches!(outcome, DevResponse::State { .. }),
        "waypoint {leg}: the route step should arrive within its budget, got {outcome:?}",
    );
    let _ = devctl(
        devctl_port,
        &DevRequest::Move {
            axes: [0.0, 0.0, 0.0],
        },
    );
}

#[test]
fn a_durable_player_flies_the_chain_node_per_realm_without_freezing_or_fence_thrash() {
    // FIRST statement: hold the process tier for the whole body, so it outlives the cluster reap
    // that frees the ports. See `vd_bins::cluster_tier`.
    let _tier = vd_bins::cluster_tier();

    // THE ROSTER + THE LABELS — derived from THE world (`world_roster` asserts I-AXIS/I-POLE/
    // I-RADIAL/J1 at derivation, so an invalidated corridor fails HERE, before a process spawns).
    let roster = world_roster(&DEV);
    let home_label = frame_for_realm(roster.home, None)
        .expect("the home realm has a frame")
        .label();
    let galaxy_label = frame_for_realm(roster.galaxy, None)
        .expect("the galaxy realm has a frame")
        .label();
    let sibling_label = frame_for_realm(roster.sibling, None)
        .expect("the sibling realm has a frame")
        .label();

    // ---- topology + trust -----------------------------------------------------------------------
    let admin_addr = reserve_tcp_addr();
    let client_quic = reserve_udp_addr();
    let devctl_port = reserve_tcp_addr().port();

    let trust_dir = std::env::temp_dir().join(format!("vd-chain-{}", std::process::id()));
    let trust = vd_io_prod::trust::ClusterTrust::generate("vd-chain").expect("trust");
    trust.write_der_dir(&trust_dir).expect("trust dir");
    let orch_store =
        std::env::temp_dir().join(format!("vd-chain-{}-orch.redb", std::process::id()));
    let _ = std::fs::remove_file(&orch_store);
    let orch_store = orch_store.display().to_string();

    // Each realm-shard + the base nodes gets its OWN reserved QUIC + probe addr — one shard per
    // realm of the chain (NO co-hosting), so every re-home is a uniform CROSS-NODE saga.
    let addrs = ClusterAddrs {
        admin: admin_addr,
        ..ClusterAddrs::reserve()
    };
    let shape = ClusterShape::Chain;
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
    // The HOME login shard (single-realm — no shape carries VD_HELD_REALMS).
    nodes.push(
        "vd-shard",
        spawn_node(
            env!("CARGO_BIN_EXE_vd-shard"),
            shard_env(&addrs, &DEV, shape),
        ),
    );
    // The three extra realm-shards (galaxy / inner planet / sibling star) — each hosting EXACTLY its
    // own realm via `realm_shard_env` (VD_REALM_KIND + VD_REALM_SEED), booting THE world's seed
    // neighbourhood. The label only names the log/kill entry (HR3: one `vd-shard` binary).
    for shard in realm_shards(shape, &addrs, &DEV) {
        let label: &'static str = match shard.node {
            n if n == GALAXY => "vd-galaxy",
            n if n == PLANET_SHARD => "vd-planet",
            _ => "vd-shard-b",
        };
        nodes.push(
            label,
            spawn_node(
                env!("CARGO_BIN_EXE_vd-shard"),
                realm_shard_env(&addrs, &DEV, shape, shard),
            ),
        );
    }

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
            "chain",
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

    // ---- bring-up: Active + an own row, POLLING Cluster::first_exited THROUGHOUT ------------------
    // The old fixture never polled it, which is exactly how four dead shards stayed invisible while
    // the gate ran green on the two survivors (D-WORLD-7's standing requirement).
    let started = Instant::now();
    loop {
        if let Some((name, status)) = nodes.first_exited() {
            panic!(
                "{name} exited during bring-up ({status}) — a Chain shard died; DIRECTORY={:#?}",
                admin_directory(admin_addr),
            );
        }
        if let Some(state) = poll_state(devctl_port).filter(|s| s.phase == DevPhase::Active)
            && own_row(&state).is_some()
        {
            assert_eq!(
                state.location.as_deref(),
                Some(home_label.as_str()),
                "the login lands in the home system: {state:?}",
            );
            break;
        }
        assert!(
            started.elapsed() < DEADLINE,
            "client never became Active with an own row (the Chain cluster failed to bring up — \
             DIRECTORY={:#?})",
            admin_directory(admin_addr),
        );
        std::thread::sleep(Duration::from_millis(100));
    }
    let _nodes = nodes; // RAII: reaps the whole cluster on test end or panic
    eprintln!("NODE-PER-REALM: client Active in {home_label:?}");

    // ---- the chain: five label-asserted crossings, ONE pilot (`vd_bins::flight`) ------------------
    // C — home → inner planet: the shared rendezvous-and-park at full orbit speed. The slowest,
    // least deterministic leg (the plan's stated fallback if it flakes: drop C/D from the walk — a
    // decision, not a quiet descope).
    rendezvous_into_planet(
        devctl_port,
        &DEV,
        roster.inner,
        &roster.inner_elements,
        Duration::from_secs(150),
    );
    eprintln!("NODE-PER-REALM leg C: crossed into the inner planet");

    // D — inner planet → home: the polar lift, stated in the PLANET's frame (the space the session
    // now stands in): 2× the pole altitude clears the planet's release reach (I-RADIAL licenses the
    // lift; I-POLE keeps the park inside the home shell).
    let lift = DVec3::new(0.0, 0.0, -2.0 * roster.pole_altitude_m);
    cross_leg(
        devctl_port,
        "D planet->home (polar lift)",
        |_tick| lift,
        &home_label,
        LEG_DEADLINE,
    );

    // A — home → galaxy: out the pole past the ~152 m release edge (flight law leg A).
    cross_leg(
        devctl_port,
        "A home->galaxy (0,0,-220)",
        |_tick| DVec3::new(0.0, 0.0, -220.0),
        &galaxy_label,
        LEG_DEADLINE,
    );

    // E — galaxy → sibling: the 12.8 km boost leg. First a ROUTE waypoint 300 m below the sibling's
    // pole (still outside its 150 m shell — no crossing), clear of every sibling planet by the
    // SIBLING system's own asserted corridor margins (`world_roster` computes I-AXIS/I-POLE/I-RADIAL
    // over the sibling's movers too — measured, no longer a bare "≥140 m" claim; the ring is XZ,
    // orbits are near-XY); the waypoint budget covers the ~26 s full-speed run.
    // Then the crossing itself is a held-throttle CREEP up the pole with NO coordinate in flight —
    // a WalkTo aimed inside the shell straddles the commit and its absolute target re-reads in the
    // sibling's frame as a point 12 km away, driving the dot straight back out (measured as a
    // fence-burning sibling↔galaxy ping-pong; see `flight::creep_into`). The arrival criterion IS
    // the label.
    let approach = roster.sibling_centre + DVec3::new(0.0, 0.0, -300.0);
    fly_waypoint(
        devctl_port,
        "E.1 galaxy route to the sibling pole",
        approach,
        3000,
    );
    creep_into(
        devctl_port,
        "E.2 galaxy->sibling (pole creep)",
        vd_bins::flight::creep_axes_plus_z(),
        &sibling_label,
        LEG_DEADLINE,
    );

    // F — sibling → galaxy: back out the pole, stated in the SIBLING's frame (the session's space).
    cross_leg(
        devctl_port,
        "F sibling->galaxy (0,0,-300)",
        |_tick| DVec3::new(0.0, 0.0, -300.0),
        &galaxy_label,
        LEG_DEADLINE,
    );

    // ---- the THRASH GUARD: the subject Entity's directory fence stays SMALL -----------------------
    // One clean re-home per crossing commits at a small fence; a ballooning fence means a re-home is
    // re-firing in a limit cycle. PRINTED every run: the bound is UNMEASURED for this chain until a
    // green history accumulates (do not claim it as proof until a number exists — tighten later).
    let entity_fence = max_entity_fence(admin_addr);
    eprintln!(
        "NODE-PER-REALM: observed max entity directory fence = {entity_fence} \
         (floor {CROSSINGS}, bound {MAX_ENTITY_FENCE})"
    );
    // TWO-SIDED (batch review): the FLOOR is what makes "one clean CAS commit per crossing" the
    // assert's content and not just its message — five real crossings each advance the subject's
    // fence at least once, so a directory that recorded fewer commits than crossings (or none at
    // all) now fails here instead of passing silently.
    assert!(
        entity_fence >= CROSSINGS,
        "MISSING COMMITS: the chain flew {CROSSINGS} label-asserted crossings but the subject \
         Entity's directory fence is only {entity_fence} — the directory did not record one CAS \
         commit per crossing. DIRECTORY={:#?}",
        admin_directory(admin_addr),
    );
    assert!(
        entity_fence <= MAX_ENTITY_FENCE,
        "FENCE THRASH: the subject Entity's directory fence ballooned to {entity_fence} (a clean \
         node-per-realm chain commits ONE fence per crossing, staying <= {MAX_ENTITY_FENCE}). \
         DIRECTORY={:#?}",
        admin_directory(admin_addr),
    );

    let _ = devctl(devctl_port, &DevRequest::Close);
    let _ = std::fs::remove_dir_all(&trust_dir);
    let _ = std::fs::remove_file(&orch_store);
}
