//! THE dual-shard PROCESS-cluster crossing proof — a real-protocol client on the production mesh,
//! with ZERO navigation, crosses THE WORLD'S OWN 150 m home shell and the directory CAS re-homes its
//! `Entity` onto the pre-booked galaxy shard.
//!
//! Stands up `vd-devcluster up --dual` as REAL BINARIES over localhost QUIC under mTLS: orchestrator +
//! gateway + the HOME shard + the GALAXY shard (the home region's parent), every realm derived through
//! `world_roster` — NOTHING is injected. The old smoke planted an authored born-inside boundary
//! file whose geometry existed nowhere in THE world; this one flies the world as
//! shipped: the login spawns at the home star's centre, the client holds `movement: [1,0,0]` (world
//! −Z at identity orientation — the ±Z polar corridor, licensed by the I-AXIS assert below) for
//! EXACTLY `exit_ticks`, then holds zero. Leaving the shell, the home shard's containment detector
//! fires the `CrossingRequest` AUTONOMOUSLY, the orchestrator resolves the directory heads, THE
//! transfer saga runs (the client stamps the in-band CUT_MARKER on request), and the CAS re-homes the
//! dot's `Entity` authority onto the galaxy shard.
//!
//! ANTI-VACUITY: the crossing fires from THE world's own geometry (no `trigger`/`start_transfer`, no
//! planted boundary file); the SOURCE side is ESTABLISHED first — the home shard is observed owning
//! the admitted dot's Entity row BEFORE the leg, and afterwards THAT SAME row rests with the dest
//! while the home owns none (moved, never copied — batch review: the poll used to break on ANY
//! dest-owned row); the C1 gate (`AdminSnapshot::realms_present` over `roster_realms(Dual)`)
//! guarantees every pre-booked head resolves before the flight, so the crossing can never count
//! `crossing_unresolved` — which since J-0 is known to be a PERMANENT STRAND, not a soft failure.
//! The flight is BOUNDED: `exit_ticks` parks the dot ~3 release-edges out, provably still inside the
//! galaxy's own shell (asserted from THE world, never a literal), so no leg can reach a realm this
//! cluster does not host.
//!
//! SCOPE (M-2): the LOCAL process playground. The N-shard k3d roster generalization is ledgered to
//! cloud (#123) in `docs/design/DEFERRED.md`. This is Tier-B process glue (coverage-exempt); the
//! Tier-A verdicts it stands on (the env builders, the port scheme, `AdminSnapshot::realms_present`,
//! the roster derivation) are covered by their own unit tests.

use std::collections::BTreeMap;
use std::net::SocketAddr;
use std::time::{Duration, Instant};

use vd_bins::{
    CROSSING_SLOT, ClusterAddrs, ClusterShape, DEV, DEV_AUTH_SEED, DevClusterDown, GATEWAY, SHARD,
    admin_get_body, devcluster, loopback, realm_shards, roster_realms, slot_trust_dir,
    world_roster,
};
use vd_core::{AccountId, EpochId, NodeId, TickId};
use vd_devproto::{CLIENT_NODE_BASE, DevPortScheme};
use vd_io_prod::mesh::{MeshConfig, MeshTransport, spawn_mesh};
use vd_io_prod::runtime::TickPacer;
use vd_io_prod::trust::ClusterTrust;
use vd_physics::worldgen::UniverseConfig;
use vd_sim::io::{Inbound, MsgClass, Transport};
use vd_wire::admin::AdminSnapshot;
use vd_wire::channels::{ClientControlMsg, InputDatagram, ServerControlMsg};
use vd_wire::version::ProtoVersion;

/// A generous deadline: `up --dual` boots 4 processes + the C1 all-realms grant, then the client logs
/// in, flies the ~1 s bounded −Z exit leg, and the crossing + saga self-drive to the directory CAS.
/// Real QUIC handshakes over loopback.
const DEADLINE: Duration = Duration::from_secs(45);

/// Run one `vd-devcluster up --dual` against the crossing slot (the launcher path from `CARGO_BIN_EXE`).
fn up_dual(launcher: &str, slot: u16) -> std::process::ExitStatus {
    std::process::Command::new(launcher)
        .args(["up", "--slot", &slot.to_string(), "--dual"])
        .status()
        .expect("run vd-devcluster up --dual")
}

/// A minimal real-protocol client that logs in, flies the BOUNDED −Z exit leg, then parks — enough to
/// make the home shard grant its avatar and integrate it out of the home shell with zero navigation.
struct LoginClient {
    transport: MeshTransport,
    session: bool,
    subscribed: bool,
    /// Set when the gateway asks for the cut (`ServerControlMsg::RequestCut`): the NEXT input carries
    /// `is_cut_marker = true` — the client's half of the transfer cut (mirrors the real client / the
    /// harness `ScriptedClient`). Without this the saga parks at `Cutting` forever and never CAS-commits.
    emit_marker_next: bool,
    /// How many MOVING inputs remain before the throttle cuts to zero — the flight bound. Inputs keep
    /// flowing every tick after it reaches zero (the cut marker must still have a carrier); only the
    /// movement axes zero out, parking the dot well inside the galaxy shell.
    move_ticks_left: u64,
    next_seq: u64,
    tick: u64,
}

impl LoginClient {
    fn new(transport: MeshTransport, exit_ticks: u64) -> LoginClient {
        LoginClient {
            transport,
            session: false,
            subscribed: false,
            emit_marker_next: false,
            move_ticks_left: exit_ticks,
            next_seq: 0,
            tick: 0,
        }
    }

    fn send_hello(&mut self, account: AccountId) {
        let hello = ClientControlMsg::Hello {
            version: ProtoVersion::CURRENT,
            login: vd_connection_plane::tickets::mint_login(
                &DEV_AUTH_SEED,
                account,
                EpochId(1),
                account.0 as u64,
            ),
        };
        let bytes = postcard::to_allocvec(&hello).expect("encode hello");
        let _ = self
            .transport
            .send(GATEWAY, MsgClass::Control, bytes.into());
    }

    /// Drain inbound (noting the welcome + the first subscription), then FLY: once subscribed, send a
    /// forward input datagram every tick — `[1, 0, 0]` is world −Z at identity orientation, the polar
    /// corridor — for `exit_ticks` ticks, then hold `[0, 0, 0]`. The bounded leg takes the dot out of
    /// the home shell (the detector fires the crossing purely geometrically, nothing hand-fed) and
    /// parks it; every later tick still carries an input so the cut marker has a carrier.
    fn step(&mut self) {
        for msg in self.transport.drain_inbound() {
            if let Inbound::Wire {
                class: MsgClass::Control,
                bytes,
                ..
            } = msg
            {
                match postcard::from_bytes::<ServerControlMsg>(&bytes) {
                    Ok(ServerControlMsg::Welcome { .. }) => self.session = true,
                    Ok(ServerControlMsg::SubscriptionOpened { .. }) => self.subscribed = true,
                    // The gateway's cut request — stamp the CUT_MARKER on the next input so the transfer
                    // saga can advance past `Cutting` to the directory CAS (the client's half of the cut).
                    Ok(ServerControlMsg::RequestCut { .. }) => self.emit_marker_next = true,
                    _ => {}
                }
            }
        }
        self.tick += 1;
        if self.subscribed {
            self.next_seq += 1;
            let is_cut_marker = self.emit_marker_next;
            self.emit_marker_next = false;
            let movement = if self.move_ticks_left > 0 {
                self.move_ticks_left -= 1;
                [1.0, 0.0, 0.0] // forward = world −Z: the licensed polar corridor
            } else {
                [0.0, 0.0, 0.0] // PARKED — the flight is bounded by construction
            };
            let input = InputDatagram {
                seq: self.next_seq,
                is_cut_marker,
                client_tick: TickId(self.tick),
                movement,
                look: [0.0, 0.0],
                action_bits: 0,
            };
            let bytes = postcard::to_allocvec(&input).expect("encode input");
            let _ = self.transport.send(GATEWAY, MsgClass::Input, bytes.into());
        }
    }
}

/// Fetch the orchestrator admin snapshot's directory rows as `(key, authority)` pairs, or empty on a
/// not-yet-answering endpoint. Reuses the SHARED `admin_get_body` HTTP client (never a hand-inlined GET).
fn directory_rows(admin: SocketAddr) -> Vec<(String, String)> {
    let Some(body) = admin_get_body(admin, "/admin/snapshot", Some(Duration::from_secs(2))) else {
        return Vec::new();
    };
    let Ok(value) = serde_json::from_str::<serde_json::Value>(&body) else {
        return Vec::new();
    };
    value["directory"]
        .as_array()
        .map(|arr| {
            arr.iter()
                .filter_map(|e| {
                    Some((
                        e["key"].as_str()?.to_owned(),
                        e["authority"].as_str()?.to_owned(),
                    ))
                })
                .collect()
        })
        .unwrap_or_default()
}

/// A directory row proving `authority` OWNS a player Entity: an `ent-…` key (distinct from a realm
/// row) resting with that shard. Asked twice — of the HOME shard before the leg (the source side
/// ESTABLISHED, batch review: any galaxy-owned `ent-` row used to end the poll with no proof the
/// home ever held it) and of the GALAXY shard after (the CAS moved it).
fn owns_an_entity(rows: &[(String, String)], authority: &str) -> Option<String> {
    rows.iter()
        .find(|(key, a)| key.starts_with("ent-") && a == authority)
        .map(|(key, _)| key.clone())
}

#[test]
fn a_dot_re_homes_home_to_galaxy_over_the_process_dual_shard_tier() {
    // FIRST statement: hold the process tier for the whole body, so it outlives the cluster reap
    // that frees the ports. See `vd_bins::cluster_tier`.
    let _tier = vd_bins::cluster_tier();

    // ---- I-AXIS FIRST (the flight law's licence for the −Z leg) --------------------------------------
    // `world_roster` asserts I-AXIS/I-POLE/I-RADIAL/J1 internally; re-stated here with the measured
    // margin so a corridor regression names its number in THIS gate's failure, not a distant panic.
    let roster = world_roster(&DEV);
    let config = UniverseConfig::world(DEV.move_speed, DEV.tick_dt);
    let axis_floor = 2.0 * config.planet.planet_soi_r_m;
    assert!(
        roster.axis_clearance_m > axis_floor,
        "I-AXIS: the −Z corridor is licensed only while every orbit clears the polar axis by more \
         than {axis_floor:.2} m (2× planet SOI); THE world measures {:.2} m",
        roster.axis_clearance_m,
    );

    // ---- THE BOUNDED FLIGHT, computed from THE world (J3: exit bounds are computed, never inherited) --
    // One tick integrates `move_speed · tick_dt` = 10 m. The release edge is the home shell plus the
    // containment band's outset; flying 3 release-edges (~46 ticks, ~460 m) is provably OUT of the home
    // shell and provably INSIDE the galaxy's own shell, so the parked dot can never reach the ring
    // sibling (~12 km away) or any realm this cluster does not host.
    let release_edge_m = config.stellar.system_soi_r_m + config.band.outset_m;
    let step_m = DEV.move_speed * DEV.tick_dt;
    let exit_ticks = (3.0 * release_edge_m / step_m).ceil() as u64;
    let park_m = exit_ticks as f64 * step_m;
    let world = vd_bins::boot_world(DEV.universe_seed, DEV.move_speed, DEV.tick_dt);
    let galaxy_shell_m = world
        .regions()
        .iter()
        .find(|r| r.realm == roster.galaxy)
        .map(|r| match r.shape {
            vd_core::geometry::Boundary::Shell { r } => r,
            other => panic!("the galaxy region is a shell, got {other:?}"),
        })
        .expect("THE world contains the galaxy region");
    assert!(
        park_m > release_edge_m && park_m < galaxy_shell_m,
        "the bounded flight must park OUT of the home shell ({release_edge_m:.1} m) and INSIDE the \
         galaxy shell ({galaxy_shell_m:.1} m); computed park {park_m:.1} m over {exit_ticks} ticks",
    );

    let launcher = env!("CARGO_BIN_EXE_vd-devcluster");
    let slot = CROSSING_SLOT;
    let _ = devcluster(launcher, "down", slot); // clean slate (idempotent)
    let _guard = DevClusterDown::new(launcher, slot);

    let ports = DevPortScheme::DEFAULT
        .slot_ports(slot)
        .expect("crossing slot resolves");
    let addrs = ClusterAddrs::for_slot(ports);

    // The DEST authority DERIVED from the shape's own shard list — never a literal node string.
    let dest_node = realm_shards(ClusterShape::Dual, &addrs, &DEV)
        .first()
        .map(|s| s.node)
        .expect("Dual pre-books the galaxy realm-shard");
    let dest_authority = format!("shard:{dest_node}");
    let home_authority = format!("shard:{SHARD}");

    // ---- WIRING: stand up the 4-process dual cluster ------------------------------------------------
    // `up --dual` spawns orchestrator + gateway + the HOME shard + the GALAXY shard, every shard booting
    // THE world's seed neighbourhood (NO boundary injection), and — the C1 gate — returns 0 ONLY after
    // every pre-booked realm is granted in the ONE directory (so the crossing's dest head resolves,
    // never `unresolved`).
    assert!(
        up_dual(launcher, slot).success(),
        "up --dual must reach the C1 all-realms ready gate and exit 0 (gateway known_shards={{GALAXY}}, \
         orchestrator roster has the galaxy shard, the directory holds home + galaxy realm heads)",
    );

    // The C1 gate proof, read straight off the admin snapshot through the SAME Tier-A predicate the
    // launcher's readiness poll uses: every roster realm rests with a shard.
    let admin = loopback(ports.admin);
    let snapshot = admin_get_body(admin, "/admin/snapshot", Some(Duration::from_secs(2)))
        .and_then(|body| serde_json::from_str::<AdminSnapshot>(&body).ok())
        .expect("the orchestrator admin snapshot parses");
    let expected_realms = roster_realms(ClusterShape::Dual, &DEV);
    assert!(
        snapshot.realms_present(&expected_realms),
        "C1: every pre-booked realm ({expected_realms:?}) must be shard-granted in the ONE directory \
         before the flight: {:?}",
        directory_rows(admin),
    );

    // ---- THE CROSSING: a real client logs in; the bounded −Z leg fires the re-home AUTONOMOUSLY ------
    // Load the launcher's mTLS trust bundle + bind at the client-0 QUIC port the gateway pre-booked
    // (`CLIENT_NODE_BASE` → client_quic(0)), then log in on the home shard. The avatar spawns at the
    // home star's centre; the bounded forward flight takes it out of the shell.
    let trust =
        ClusterTrust::from_der_dir(&slot_trust_dir(slot)).expect("load launcher trust bundle");
    let rt = tokio::runtime::Builder::new_multi_thread()
        .worker_threads(2)
        .enable_all()
        .build()
        .expect("client runtime");
    // STRUCTURAL one-connection invariant: the client books ONLY the gateway.
    let client_book = BTreeMap::from([(GATEWAY, loopback(ports.gateway))]);
    let client_bind = loopback(ports.client_quic(0).expect("client-0 quic port"));
    let (transport, control) = spawn_mesh(
        rt.handle(),
        &trust,
        &MeshConfig::new(NodeId(CLIENT_NODE_BASE), client_bind, client_book, 64, 0),
        None,
    )
    .expect("client mesh");
    std::mem::forget(control); // keep the endpoint alive for the test's duration
    let mut client = LoginClient::new(transport, exit_ticks);

    let started = Instant::now();
    let mut pacer = TickPacer::new(DEV.tick_hz);
    let mut hello_retry = Instant::now();
    client.send_hello(AccountId(1000));

    // ---- SOURCE SIDE ESTABLISHED (batch review): the HOME shard owns the freshly admitted dot's
    // Entity row BEFORE the crossing can commit — measured off the ONE directory, so the headline
    // below is provably a MOVE off the home shard, never satisfiable by an entity that somehow
    // began life on the dest. The admit lands within the login handshake; the exit leg takes
    // ~`exit_ticks` more ticks and the saga longer still, so this poll always wins the race.
    let subject_key = loop {
        client.step();
        // Children may still be settling the QUIC handshake; re-send Hello until welcomed (idempotent).
        if !client.session && hello_retry.elapsed() > Duration::from_millis(500) {
            hello_retry = Instant::now();
            client.send_hello(AccountId(1000));
        }
        if let Some(key) = owns_an_entity(&directory_rows(admin), &home_authority) {
            break key;
        }
        assert!(
            started.elapsed() < DEADLINE,
            "the HOME shard never owned the admitted dot's Entity row (session={}) — the source \
             side was never established, so a crossing could prove nothing. directory={:?}",
            client.session,
            directory_rows(admin),
        );
        let _ = pacer.wait();
    };
    eprintln!("DUAL-CROSSING: source established — {home_authority} owns {subject_key}");

    loop {
        client.step();
        // THE PROOF: poll the ONE directory until the galaxy owns an Entity — the dot re-homed
        // home → galaxy across real processes.
        if owns_an_entity(&directory_rows(admin), &dest_authority).is_some() {
            break;
        }
        assert!(
            started.elapsed() < DEADLINE,
            "the dot never re-homed to the galaxy (session={}): the world's-own-shell crossing did \
             not flip an Entity head to {dest_authority}. directory={:?}",
            client.session,
            directory_rows(admin),
        );
        let _ = pacer.wait();
    }

    // The dest-owns flip IS the proof; assert it once more explicitly for the failure message, and
    // pin MOVED-NOT-COPIED (batch review: the count below used to be computed and then discarded):
    // THE established subject row flipped to the dest, and the home shard is left owning NO Entity
    // row at all.
    let final_rows = directory_rows(admin);
    assert!(
        owns_an_entity(&final_rows, &dest_authority).is_some(),
        "the galaxy shard must own the re-homed dot's Entity row: {final_rows:?}",
    );
    assert_eq!(
        final_rows
            .iter()
            .find(|(key, _)| *key == subject_key)
            .map(|(_, authority)| authority.as_str()),
        Some(dest_authority.as_str()),
        "THE SAME Entity row the home shard owned before the leg ({subject_key}) rests with the \
         dest after it — the CAS moved the established subject, not some other row: {final_rows:?}",
    );
    let home_entity_rows = final_rows
        .iter()
        .filter(|(key, authority)| key.starts_with("ent-") && *authority == home_authority)
        .count();
    let dest_entity_rows = final_rows
        .iter()
        .filter(|(key, authority)| key.starts_with("ent-") && *authority == dest_authority)
        .count();
    assert!(
        dest_entity_rows >= 1,
        "the galaxy holds >= 1 re-homed Entity ({dest_entity_rows}); the directory CAS committed the \
         crossing: {final_rows:?}",
    );
    assert_eq!(
        home_entity_rows, 0,
        "MOVED, NOT COPIED: the home shard holds no Entity authority after the commit (the \
         directory keys one row per entity — a lingering home-owned row would be a rival \
         authority): {final_rows:?}",
    );
    // Belt-and-suspenders anti-vacuity: the whole run's client actually established a session (the
    // crossing's `Session` head resolved), so the re-home was a real logged-in dot, not a phantom.
    assert!(
        client.session,
        "the client established a session (the crossing route's Session head resolved)"
    );
}
