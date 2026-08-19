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
//!                            the pole — THE GOVERNED WARP LEG (S3): the star gap is 0.2376656 ly
//!                            and the walk rides the galaxy's own ceiling (2·R_gal/T_TRAVERSE)
//!                            with the approach governor decelerating onto the sibling, so the
//!                            leg is ~2.5 minutes of wall clock; its budget and brake are the
//!                            governed closed form (`vd_core::flight::leg_time_s`), derived in
//!                            the body. Clear of every sibling planet by the SIBLING's own
//!                            I-AXIS/I-POLE margins, which `world_roster` ASSERTS over the
//!                            sibling's movers too; no coordinate in flight across the commit
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
use vd_core::flight::{
    FlightTuning, TRAVERSE_S, approach_ceiling_mps, leg_time_s, realm_speed_cap_mps,
};
use vd_core::glam::DVec3;
use vd_core::pose::frame_for_realm;
use vd_devproto::{CLIENT_NODE_BASE, DevEntityRow, DevPhase, DevRequest, DevResponse, DevState};

/// The BRING-UP budget: 6 real processes + the login grant-flip. Flight legs carry their own
/// DERIVED budgets (2× the governed closed form each — see the leg-budget block in the body);
/// this constant no longer bounds any flying.
const DEADLINE: Duration = Duration::from_secs(300);
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

/// The OWN entity's delivered world pose, when the row is delivered.
fn own_pos(state: &DevState) -> Option<DVec3> {
    own_row(state).map(|r| DVec3::from_array(r.pos))
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
/// CHUNKED (S3): a `WalkTo` blocks the devctl socket for its whole tick budget and the governed
/// star-gap route runs minutes — each chunk's blocking read stays a quarter of the socket's own
/// timeout, derived. The route ends when the delivered pose is within the walk's own epsilon
/// class of the waypoint, or the whole budget lapses (asserted).
fn fly_waypoint(
    devctl_port: u16,
    leg: &str,
    target: DVec3,
    max_ticks: u64,
    max_step_m: f64,
    arrive_within_m: f64,
) {
    // `arrive_within_m` is DERIVED PER LEG from the clearance geometry the park protects (a
    // shell fraction), never a fixed metre count: under the geometric throttle taper the last
    // stretch below ~1e9 m of a governed brake closes at FOOT speed, so a 3 m arrival demand
    // at true scale is an unbounded crawl (measured: A.2 never arrived in its whole budget).
    let chunk_ticks =
        ((vd_bins::DEVCTL_READ_TIMEOUT.as_secs_f64() / 4.0) / DEV.tick_dt).floor() as u64;
    let mut spent = 0u64;
    let arrived = loop {
        let chunk = chunk_ticks.min(max_ticks - spent);
        let outcome = devctl(
            devctl_port,
            &DevRequest::WalkTo {
                target: target.to_array(),
                arrive_epsilon: arrive_within_m,
                max_ticks: chunk,
                max_step_m,
            },
        )
        .unwrap_or_else(|| panic!("waypoint {leg}: no walk response"));
        // A chunk that ARRIVES answers `State`; a chunk that spends its whole slice still in
        // flight answers `Timeout{state}` — which on a governed star-gap route is the NORMAL
        // mid-journey outcome (measured: the first 75 s chunk covers ~half the 0.238 ly gap).
        // Anything else (an error string, a refusal) is a real failure.
        assert!(
            matches!(
                outcome,
                DevResponse::State { .. } | DevResponse::Timeout { .. }
            ),
            "waypoint {leg}: the route step should progress within its budget, got {outcome:?}",
        );
        spent += chunk;
        let st = poll_state(devctl_port)
            .unwrap_or_else(|| panic!("waypoint {leg}: no delivered state mid-route"));
        let pos =
            own_pos(&st).unwrap_or_else(|| panic!("waypoint {leg}: no delivered pose mid-route"));
        eprintln!(
            "[waypoint {leg}] spent={spent} tick={:?} dist_to_target={:.4e}",
            st.universe_tick,
            (pos - target).length(),
        );
        if (pos - target).length() <= arrive_within_m {
            break true;
        }
        if spent >= max_ticks {
            break false;
        }
    };
    let _ = devctl(
        devctl_port,
        &DevRequest::Move {
            axes: [0.0, 0.0, 0.0],
        },
    );
    assert!(
        arrived,
        "waypoint {leg}: the route never arrived within its {max_ticks}-tick governed budget",
    );
}

// UN-PARKED at the speed-law slice (S3, real-scale addendum §A3 + the OQ-2 ruling): legs E/F now
// fly the 0.2376656 ly star gap at GOVERNED speeds — the galaxy ceiling with the approach governor
// decelerating onto the sibling — re-derived from the governed closed form in the body. Every
// in-system assertion the park owed is back verbatim.
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

    // THE DERIVED LEG BUDGETS (true-scale restatement): the fixed 60 s LEG_DEADLINE was the
    // interim world's — a governed leg out of a 1.2e7 m planet or across a 1.6e11 m system is
    // MINUTES by the speed law's own closed form, so each leg's budget is 2× its governed
    // closed-form time plus a fixed commit/boot slack, derived off THE world's solved shells.
    let world_cfg = vd_physics::worldgen::UniverseConfig::world(DEV.move_speed, DEV.tick_dt);
    let world_regions =
        vd_physics::worldgen::realm_regions_for_config(DEV.universe_seed, &world_cfg);
    let shell_of = |realm: vd_core::pose::RealmId| {
        world_regions
            .iter()
            .find(|r| r.realm == realm)
            .map(|r| r.shape.finite_extent())
            .expect("a flown realm is rostered on THE world")
    };
    let tuning = FlightTuning::derive(
        DEV.move_speed,
        DEV.tick_dt,
        (u64::from(DEV.tick_hz) / 2).max(1),
        u32::try_from(DEV.boot_ticks_p99).expect("boot p99 fits"),
    );
    let budget =
        |dist: f64, cap: f64| -> Duration { vd_bins::flight::governed_leg_budget(&DEV, dist, cap) };

    // D — inner planet → home: the polar lift, stated in the PLANET's frame (the space the session
    // now stands in): 2× the pole altitude clears the planet's release reach (I-RADIAL licenses the
    // lift; I-POLE keeps the park inside the home shell). Budget: exiting the planet's own bound
    // under the planet's own ceiling (the label flips at the release edge, long before the aim).
    let cap_planet = realm_speed_cap_mps(shell_of(roster.inner), DEV.move_speed, TRAVERSE_S);
    let lift = DVec3::new(0.0, 0.0, -2.0 * roster.pole_altitude_m);
    cross_leg(
        devctl_port,
        "D planet->home (polar lift)",
        |_tick| lift,
        &home_label,
        budget(shell_of(roster.inner) + world_cfg.band.outset_m, cap_planet),
    );

    // A — home → galaxy: out the pole past the release edge (flight law leg A). THE EXIT
    // HEIGHT IS DERIVED FOR THE 3-D SIBLING (S3 re-derivation): leg E.1 then flies a STRAIGHT
    // line toward the sibling's seeded direction, whose closest approach to home is
    // `exit_z·sin(θ_off_pole)` — at the old 220 m exit the seed-0 sibling (u_z ≈ +0.748) leaves a
    // 146 m approach, INSIDE the 150 m shell (a mid-warp clip back into home). The derived height
    // clears the home's whole containment reach 2× by construction.
    // The TRUE-SIZE home shell (solved per system — no config radius exists any more).
    let home_reach = shell_of(roster.home) + world_cfg.band.outset_m;
    let sib_unit = roster.sibling_centre / roster.sibling_centre.length();
    let sin_off_pole = (1.0 - sib_unit.z * sib_unit.z).max(0.0).sqrt();
    // The floor is the interim world's 220 m relic made DERIVED: two release-reaches straight
    // down the pole always clears home whatever the sibling's off-pole angle.
    let exit_z = (2.0 * home_reach / sin_off_pole).max(2.0 * home_reach);
    let cap_home = realm_speed_cap_mps(shell_of(roster.home), DEV.move_speed, TRAVERSE_S);
    cross_leg(
        devctl_port,
        "A home->galaxy (polar, derived height)",
        |_tick| DVec3::new(0.0, 0.0, -exit_z),
        &galaxy_label,
        // The label flips at the home release edge; the whole path to it rides the home ceiling.
        budget(home_reach, cap_home),
    );
    // A.2 — DESCEND TO THE DERIVED EXIT HEIGHT before turning toward the sibling. `cross_leg`
    // ends the moment the LABEL flips — at the release edge (~1 home shell down the pole), NOT
    // at the aim — and the S3 clearance derivation (closest approach = exit_z·sinθ ≥ 2×reach)
    // assumed the onward line STARTS at the full exit height. Launched from the release edge,
    // the line to the +z-leaning sibling (u_z ≈ +0.748) re-enters the home shell (measured:
    // a System 1↔System 7 crossing ping-pong at fences 3→6 while E.1's waypoint never
    // arrived). The aim is FRAME-STABLE across the crossing: home sits AT the galaxy origin,
    // so (0,0,−exit_z) names the same point in both spaces.
    // The budget's cap is the HOME ceiling, conservatively: a receding leg's true ceiling is
    // the departed body's arm (child cap + d/τ, so ~cap_home right at the release edge where
    // the leg starts), and budgeting the whole distance at the galaxy cap measured one chunk
    // short. A generous ceiling only bounds a genuinely frozen leg.
    let a2_budget = vd_bins::flight::governed_leg_budget(&DEV, exit_z - home_reach, cap_home);
    fly_waypoint(
        devctl_port,
        "A.2 descend to the derived exit height",
        DVec3::new(0.0, 0.0, -exit_z),
        (a2_budget.as_secs_f64() / DEV.tick_dt) as u64,
        // THE TAPER IS THE ARRIVAL SLOP, never a governed-brake distance (MEASURED, this gate:
        // a taper zone wider than the slop is a RAMP-COLLAPSE TRAP — any throttle < 1 commands
        // a speed below the ramp, the ramp re-derives from the now-lower carried velocity, and
        // the two spiral down until the dot is wedged at walking pace with 1.2e11 m still to
        // go; integrate-diag runs 12-13 caught the spiral at seq 8704→9216, 3.1e10 → 441.7
        // m/s. The speed law is coherent — partial throttle IS exponentially slower by design
        // — so the pilot holds FULL throttle and the arrival is the governor's falling
        // ceiling + the slop, exactly the §4.2(c) division of labour.)
        0.5 * home_reach,
        // Half a home reach of slop keeps the onward clearance ≥ (exit_z − ε)·sinθ, still well
        // past the shell.
        0.5 * home_reach,
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
    // The route waypoint stands off the SIBLING's own solved shell by one shell (2× centre
    // distance = its shell doubled — outside, no crossing), down its pole.
    let sib_standoff_z = 2.0 * shell_of(roster.sibling);
    let approach = roster.sibling_centre + DVec3::new(0.0, 0.0, -sib_standoff_z);
    // THE GOVERNED LEG DERIVATION (S3): budget = 2× the governed closed form down the star gap
    // (ramp up from the foot, cruise at the galaxy ceiling, governor ramp-down onto the
    // waypoint's own ceiling); brake = the LAG-DERIVED governed brake
    // (`vd_bins::flight::governed_brake_m` — the pre-law 40 m brake is a limit cycle at a
    // governed arrival ceiling steered by a delivered pose).
    let cap_galaxy = realm_speed_cap_mps(world_cfg.scale.galaxy_r_m, DEV.move_speed, TRAVERSE_S);
    let v_waypoint = approach_ceiling_mps(
        realm_speed_cap_mps(shell_of(roster.sibling), DEV.move_speed, TRAVERSE_S),
        sib_standoff_z - shell_of(roster.sibling),
        tuning.tau_s,
    );
    let leg_s = leg_time_s(
        roster.sibling_centre.length(),
        cap_galaxy,
        DEV.move_speed,
        v_waypoint,
        tuning.tau_s,
    )
    .expect("the star-gap leg holds a cruise");
    // 3× + slack (the walk gate's own measured budget law — see `governed_leg_budget`).
    let leg_ticks =
        (vd_bins::flight::governed_leg_budget(&DEV, roster.sibling_centre.length(), cap_galaxy)
            .as_secs_f64()
            / DEV.tick_dt) as u64;
    let brake_m = vd_bins::flight::governed_brake_m(v_waypoint, DEV.tick_dt, tuning.tau_s);
    eprintln!(
        "NODE-PER-REALM leg E derivation: galaxy ceiling {cap_galaxy:.4e} m/s, waypoint ceiling \
         {v_waypoint:.1} m/s, governed closed-form leg {leg_s:.1} s => budget {leg_ticks} ticks, \
         brake {brake_m:.1} m",
    );
    fly_waypoint(
        devctl_port,
        "E.1 galaxy route to the sibling pole",
        approach,
        leg_ticks,
        // The taper IS the arrival slop (see A.2's ramp-collapse note); `brake_m` above stays
        // derived + printed as the overshoot yardstick, never the taper.
        0.5 * shell_of(roster.sibling),
        // Half a sibling shell of slop parks 1.5–2.5 shells below the pole — still outside the
        // release edge; E.2's held-axes approach flies the remainder.
        0.5 * shell_of(roster.sibling),
    );
    creep_into(
        devctl_port,
        "E.2 galaxy->sibling (held-axes governed approach)",
        // FULL throttle, not the 0.05 creep: the standoff is a whole sibling shell below the
        // bound (~1e11 m) — the approach governor, not a tiny throttle, shapes the arrival
        // (see `governed_axes_plus_z`). Held axes keep the no-coordinate-in-flight property.
        vd_bins::flight::governed_axes_plus_z(),
        &sibling_label,
        budget(sib_standoff_z, cap_galaxy),
    );

    // F — sibling → galaxy: back out the pole, stated in the SIBLING's frame (the session's
    // space). The interim world aimed (0,0,−300) — outside its 150 m shell; on the true-size
    // world 300 m is DEEP INSIDE (toward the sibling's own star), so the aim is the derived
    // 2× solved shell down the pole, and the label flips at the release edge on the way.
    let cap_sibling = realm_speed_cap_mps(shell_of(roster.sibling), DEV.move_speed, TRAVERSE_S);
    let sib_exit = DVec3::new(0.0, 0.0, -2.0 * shell_of(roster.sibling));
    cross_leg(
        devctl_port,
        "F sibling->galaxy (polar, derived height)",
        |_tick| sib_exit,
        &galaxy_label,
        budget(
            shell_of(roster.sibling) + world_cfg.band.outset_m,
            cap_sibling,
        ),
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
