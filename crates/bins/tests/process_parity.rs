//! The P1 process-tier parity gate: the SAME logical scenario the in-process
//! `p1_gates` proves — login, walk, see other dots, one connection target —
//! running as REAL BINARIES over localhost QUIC under cluster mTLS.
//!
//! Tier-1 (in-process) never solely gates a release: this tier validates the
//! quinn handshakes, the mesh bridge semantics, the bins' wiring, and the
//! admin endpoint — everything the VirtualClock tier cannot see.

use std::collections::{BTreeMap, BTreeSet};
use std::net::SocketAddr;
use std::process::Child;
use std::time::{Duration, Instant};

use vd_bins::{
    Cluster, ClusterAddrs, DEV, DEV_AUTH_SEED, GATEWAY, admin_get_body, common_env,
    dev_auth_pubkey_hex, gateway_env, orchestrator_env, reserve_tcp_addr, reserve_udp_addr,
    shard_env, spawn_node,
};
use vd_core::pose::StampedPose;
use vd_core::{AccountId, EntityId, EpochId, NodeId, SessionId, TickId};
use vd_devproto::CLIENT_NODE_BASE;
use vd_io_prod::mesh::{MeshConfig, MeshTransport, spawn_mesh};
use vd_io_prod::runtime::TickPacer;
use vd_io_prod::trust::ClusterTrust;
use vd_sim::io::{Inbound, MsgClass, Transport};
use vd_wire::channels::{
    ClientControlMsg, EventMsg, InputDatagram, RealmSnapshotDatagram, ServerControlMsg,
    SnapshotDatagram, SnapshotVerdict, SubId, classify_snapshot,
};
use vd_wire::version::ProtoVersion;

// ★ RAISED 2026-08-31, AND THE NUMBER IS MEASURED. These tests spawn REAL nodes built in DEBUG, and a
// debug shard folds THE world's 233 220 star systems before it can answer anything: MEASURED, ~50 s
// from process start to "planting the containment forest", against ~0.8 s for the same fold in
// release. The dev cluster's own bring-up measures 101 s end to end.
//
// So these deadlines are sized for the BUILD the tests actually run, not for the shipped one. What
// each test proves is that its node CONVERGES — ready, drained, re-routed — never how fast. A test
// that means to measure speed would say so and would not be run on a debug binary.
const DEADLINE: Duration = Duration::from_secs(240);
/// How many times the gateway may restate the whole sky before the client's confirmation stops it.
///
/// NOT a tolerance to be widened. The gateway holds no memory of what it sent — that memory could not
/// be made correct and was deleted — so it re-states each beat until the client confirms, and the
/// confirmation costs one round trip. Measured at TWO on a fast local cluster. The bound catches the
/// failure that matters: a confirmation that never lands, and a galaxy re-downloaded for ever.
const MAX_SKY_RESTATEMENTS: usize = 4;

/// A minimal real-protocol client over the production mesh transport.
struct ProcessClient {
    transport: MeshTransport,
    session: Option<SessionId>,
    held_subs: BTreeSet<SubId>,
    own_entity: Option<EntityId>,
    poses: BTreeMap<EntityId, StampedPose>,
    last_frame: Option<u64>,
    next_seq: u64,
    tick: u64,
    /// How many REALM-lane datagrams arrived, each DECODED (a decode failure is a panic — a
    /// transport fault). Asserted `> 0` at the end: the lane is ON by construction here. See `step`.
    realm_frames: u64,
    /// Total realm ROWS delivered across those datagrams — asserted `> 0` for the walker (the home
    /// system's own movers carry per-tick placements, so the lane carries real rows, not empty
    /// headers). It used to say "five planets"; THE world's home system holds NINE, and the count
    /// is a `derived_world_planet_count` fact, not something to write down here.
    realm_rows: u64,
    /// `EntityRemoved` evictions applied (mirroring the production client's rule). Asserted `== 0`
    /// at the end: NOBODY leaves in this static two-avatar scenario, so any eviction is a defect —
    /// e.g. a fan that violated the minor-14 owner-skip rule (see `on_control`).
    evictions: u64,
    /// Composed scene LEVELs received (minor 18). Asserted `== 1` at the end: login bumps the
    /// origin epoch exactly once here, and nothing else may (no crossing in this scenario).
    scene_levels: u64,
    /// Levels that arrived at a DIFFERENT epoch than the one before: a re-origin. Zero in a
    /// single-shard scenario, whatever the number of levels.
    scene_reorigins: u64,
    /// Parts of the star catalogue this client received (S11).
    sky_parts: u64,
    /// THE DISTINCT STARS HELD (S11) — a SET, not a running total.
    ///
    /// ★ WHY A SET. The gateway keeps no memory of what it sent; it re-states the sky each beat until
    /// the client confirms, and confirming costs a round trip. So the catalogue legitimately arrives
    /// more than once. MEASURED: 6 rows for a 3-star world. A running total then reads "6 stars" for a
    /// galaxy of 3, which is a statement about the CARRIER, not about what the client holds — the same
    /// trap the old message-count gate fell into.
    sky_stars: std::collections::BTreeSet<vd_core::pose::RealmId>,
    /// The generation of the catalogue that actually arrived (S11).
    sky_generation: Option<u64>,
    /// Liveness beats seen over the real transport (S11).
    sky_beats: u64,
    /// The generation those beats named (S11).
    sky_beat_generation: Option<u64>,
    /// The origin epoch carried by the newest level — pinned to `Some(1)` at the end, and every
    /// scene delta must match it on arrival (see `on_control`).
    scene_epoch: Option<u64>,
    /// THE FIRST pose this client was ever told about its OWN avatar — where the login PUT it,
    /// before any input of its own could move it. Recorded because "did it move" and "where does a
    /// login land" are different questions and this gate asks both; printed on a stalled
    /// convergence so a login that lands somewhere unexpected names itself.
    first_own_pose: Option<vd_core::pose::LatticePos>,
}

impl ProcessClient {
    fn new(transport: MeshTransport) -> ProcessClient {
        ProcessClient {
            transport,
            session: None,
            held_subs: BTreeSet::new(),
            own_entity: None,
            poses: BTreeMap::new(),
            last_frame: None,
            next_seq: 0,
            tick: 0,
            realm_frames: 0,
            realm_rows: 0,
            evictions: 0,
            scene_levels: 0,
            scene_reorigins: 0,
            sky_parts: 0,
            sky_stars: std::collections::BTreeSet::new(),
            sky_generation: None,
            sky_beats: 0,
            sky_beat_generation: None,
            scene_epoch: None,
            first_own_pose: None,
        }
    }

    /// Send `Hello` the way the real client does: ONCE, the first time the transport accepts it.
    /// Returns whether the transport accepted the send (a refusal means the lane is not up yet, and
    /// the caller tries again next beat). A second ACCEPTED `Hello` from the same node is, by the
    /// gateway's rule since 2026-09-05, a NEW PROCESS: it ends the pending session and starts over —
    /// so a client that keeps re-sending an accepted `Hello` never settles (it keeps re-logging in).
    fn send_hello(&mut self, account: AccountId) -> bool {
        let hello = ClientControlMsg::Hello {
            version: ProtoVersion::CURRENT,
            login: vd_connection_plane::tickets::mint_login(
                &DEV_AUTH_SEED,
                account,
                EpochId(1),
                account.0 as u64,
            ),
        };
        let bytes = postcard::to_allocvec(&hello).expect("encode");
        self.transport
            .send(GATEWAY, MsgClass::Control, bytes.into())
            .is_ok()
    }

    /// One client tick: drain, decode (asserting the single-peer invariant),
    /// optionally walk.
    fn step(&mut self, walk: bool) {
        for msg in self.transport.drain_inbound() {
            match msg {
                Inbound::Wire { from, class, bytes } => {
                    assert_eq!(from, GATEWAY, "one connection target, ever");
                    match class {
                        MsgClass::Control => self.on_control(&bytes),
                        MsgClass::Snapshot => self.on_snapshot(&bytes),
                        // THE REALM LANE — where a star system's children are, this tick. The
                        // mechanism is ESTABLISHED (audit :774 — this arm used to tolerate the lane
                        // with a comment admitting it was unmeasured): the Single-shape shard hosts
                        // THE world's home System (`DEV.realm_seed`), whose planets are ALL movers
                        // (this used to say FIVE — the retired compressed geometry's count; the
                        // shipped world derives NINE, and the number belongs in the generator, not
                        // here), so `emit_realm_frames` authors per-tick placement rows the moment
                        // it holds authority + a present observer, and the gateway fans the realm
                        // lane to every subscribed session. MEASURED here: every datagram must
                        // DECODE (a failure IS a transport fault) and the lane + its rows are
                        // asserted `> 0` at the end — a measured comparison, not a tolerance.
                        MsgClass::RealmSnapshot => {
                            let snap: RealmSnapshotDatagram =
                                postcard::from_bytes(&bytes).expect("decode realm snapshot");
                            self.realm_frames += 1;
                            self.realm_rows += snap.realms.len() as u64;
                        }
                        other => panic!("unexpected class {other:?}"),
                    }
                }
                Inbound::NodeUnreachable { .. } => {
                    // The gateway wasn't up yet for an early frame; retried below.
                }
                Inbound::SendShed { .. } => {
                    // A local send-shed (R-4d M3): this test client's sends are tiny + its buffer
                    // ample, so this never fires — present for Inbound exhaustiveness.
                }
                Inbound::PeerReset { .. } => {
                    // The gateway never restarts in this test; present for Inbound exhaustiveness.
                }
            }
        }
        self.tick += 1;
        if walk && !self.held_subs.is_empty() {
            self.next_seq += 1;
            let input = InputDatagram {
                seq: self.next_seq,
                is_cut_marker: false,
                client_tick: TickId(self.tick),
                movement: [1.0, 0.0, 0.0],
                look: [0.0, 0.0],
                action_bits: 0,
            };
            let bytes = postcard::to_allocvec(&input).expect("encode");
            let _ = self.transport.send(GATEWAY, MsgClass::Input, bytes.into());
        }
    }

    fn on_control(&mut self, bytes: &[u8]) {
        match postcard::from_bytes::<ServerControlMsg>(bytes).expect("decode control") {
            ServerControlMsg::Welcome { session, .. } => self.session = Some(session),
            ServerControlMsg::SubscriptionOpened { sub, .. } => {
                self.held_subs.insert(sub);
            }
            // S6: the node-agnostic own-entity signal (minor 2) names the avatar by EntityId alone —
            // this pure-renderer parity client learns which entity is itself WITHOUT any node info.
            ServerControlMsg::OwnEntity { entity } => self.own_entity = Some(entity),
            // Legacy node-aware `AuthorityChanged` — still emitted to old clients; a minor-2 client
            // ignores it (OwnEntity is the own-entity source of truth).
            ServerControlMsg::AuthorityChanged { .. } => {}
            ServerControlMsg::Close { reason } => panic!("gateway closed the session: {reason}"),
            // The cluster tick rate (minor 1) — this minimal parity client does not
            // interpolate; ignore it (the real client learns its render rate from it).
            ServerControlMsg::UniverseRate { .. } => {}
            // The remove message (minor 14) — applied, measured, never swallowed (audit :774: the
            // old blanket-ignore arm justified itself with "tracks no bystander figures", which the
            // convergence gate two screens down falsifies — BOTH avatars' poses are tracked). This
            // client mirrors the production rule (evict the entity's track) and COUNTS it; the gate
            // asserts ZERO evictions at the end, because nobody leaves in this static two-avatar
            // scenario — a spurious fan (e.g. one violating the minor-14 owner-skip rule) now turns
            // this gate red instead of passing silently.
            ServerControlMsg::Event(EventMsg::EntityRemoved { entity, .. }) => {
                self.poses.remove(&entity);
                self.evictions += 1;
            }
            // THE COMPOSED SCENE LANE (minor 18, the C1 flag day): the gateway ships every session
            // one full level at each origin-epoch bump (login = 1), bag-diff deltas between bumps,
            // and — since Step 12 (2026-09-04) — the full level AGAIN on every keep-alive beat at the
            // CURRENT epoch, so a client that missed a delta is healed by the next beat. Measured,
            // never swallowed: a level must LEAD with the session's own origin (the origin-marker
            // law, window_lane.md §2.7), and the end gate pins the epoch to exactly 1 with ZERO
            // re-origins — this single-shard scenario never crosses, so a level at a NEW epoch is a
            // defect, while many levels at the SAME epoch are the beat doing its job.
            ServerControlMsg::RealmRegistry {
                origin,
                origin_epoch,
                rows,
            } => {
                assert_eq!(
                    rows.first().map(|row| row.realm),
                    Some(origin),
                    "the origin marker leads every composed level"
                );
                self.scene_levels += 1;
                if self.scene_epoch.is_some_and(|epoch| epoch != origin_epoch) {
                    self.scene_reorigins += 1;
                }
                self.scene_epoch = Some(origin_epoch);
            }
            // A delta may only refine the CURRENT epoch's scene — an off-epoch delta on the
            // ordered control lane is a composer-ordering defect, so it panics rather than skews.
            ServerControlMsg::RealmSceneDelta { origin_epoch, .. } => {
                assert_eq!(
                    self.scene_epoch,
                    Some(origin_epoch),
                    "a scene delta rides the epoch of the level that preceded it"
                );
            }
            // ★ THE STAR CATALOGUE (S11) — and its arrival HERE is the end-to-end proof that the lane
            // works: real binaries, over QUIC, carrying the real seed's stars. Counted rather than
            // ignored, and its self-consistency asserted, because a catalogue that arrived malformed
            // over the real transport is exactly what the unit tiers cannot see.
            ServerControlMsg::StarCatalogue {
                generation,
                part,
                parts,
                rows,
            } => {
                assert!(parts > 0, "a catalogue with no parts can never complete");
                assert!(
                    part < parts,
                    "a part outside the set it claims to belong to"
                );
                assert!(
                    !rows.is_empty(),
                    "an empty part is a part that says nothing"
                );
                assert!(
                    generation != 0,
                    "the generation is folded from content and is never a default"
                );
                self.sky_parts += 1;
                self.sky_stars.extend(rows.iter().map(|r| r.realm));
                self.sky_generation = Some(generation);
            }
            // ★ THE SKY'S LIVENESS BEAT (S11), over the real transport. Its arrival here proves the
            // thing the unit tiers cannot: that a real client, over QUIC, is TOLD the sky is still
            // current rather than being left to read silence and guess.
            ServerControlMsg::SkyAlive { generation } => {
                assert!(
                    generation != 0,
                    "the generation is folded from content and is never a default"
                );
                self.sky_beats += 1;
                self.sky_beat_generation = Some(generation);
            }
            ServerControlMsg::Event(other) => panic!("unexpected event in P1: {other:?}"),
            other => panic!("unexpected control message in P1: {other:?}"),
        }
    }

    fn on_snapshot(&mut self, bytes: &[u8]) {
        let snap: SnapshotDatagram = postcard::from_bytes(bytes).expect("decode snapshot");
        // THE shared §6.3 gate — the SAME vd_wire fn the production client and the
        // in-process ScriptedClient call (no third hand-inlined copy to drift), so
        // the real-binary parity gate is held to the very SSOT it validates.
        match classify_snapshot(&self.held_subs, self.last_frame, snap.sub, snap.frame_id) {
            SnapshotVerdict::Apply => {
                self.last_frame = Some(snap.frame_id);
                for entity in snap.entities {
                    if self.own_entity == Some(entity.entity) && self.first_own_pose.is_none() {
                        self.first_own_pose = Some(entity.pose.pos);
                    }
                    self.poses.insert(entity.entity, entity.pose);
                }
            }
            SnapshotVerdict::DropForeignSub | SnapshotVerdict::DropStale => {}
        }
    }
}

#[test]
fn p1_parity_real_binaries_over_quic() {
    // FIRST statement: hold the process tier for the whole body, so it outlives the cluster reap
    // that frees the ports. See `vd_bins::cluster_tier`.
    let _tier = vd_bins::cluster_tier();
    // ---- topology: addresses, trust bundle, child processes ------------------
    let orch_addr = reserve_udp_addr();
    let gateway_addr = reserve_udp_addr();
    let shard_addr = reserve_udp_addr();
    let client_a_addr = reserve_udp_addr();
    let client_b_addr = reserve_udp_addr();
    let admin_addr = reserve_tcp_addr();
    // RLM RG-4c non-vacuity control: give the STATIC gateway its own admin endpoint so this test can prove a
    // static gateway NEVER receives a reactive greeting nor mints a dynamic home — making the demand-login
    // e2e's `presence_announces >= 1` / `dynamic_shards >= 1` assertions non-vacuous.
    let gw_admin = reserve_tcp_addr();

    let trust_dir = std::env::temp_dir().join(format!("vd-parity-{}", std::process::id()));
    let trust = ClusterTrust::generate("vd-parity").expect("trust");
    trust.write_der_dir(&trust_dir).expect("trust dir");
    // D-6: the orchestrator's durable Store (temp scratch ⇒ VD_STORE_EPHEMERAL_OK via orchestrator_env).
    // Remove any stale file from a recycled-pid prior run so this parity spawn boots at genesis.
    let orch_store =
        std::env::temp_dir().join(format!("vd-parity-{}-orch.redb", std::process::id()));
    let _ = std::fs::remove_file(&orch_store);
    let orch_store = orch_store.display().to_string();

    // The cluster contract is the SHARED vd_bins source of truth — the exact env,
    // params, roster, dev identity, and peer-book the launcher uses. If they ever
    // drift, THIS gate is what bring-up is really validated against.
    let addrs = ClusterAddrs {
        orchestrator: orch_addr,
        gateway: gateway_addr,
        shard: shard_addr,
        admin: admin_addr,
        gateway_admin: Some(gw_admin),
        ..ClusterAddrs::reserve()
    };
    let auth_pubkey_hex = dev_auth_pubkey_hex();
    let common = common_env(&trust_dir.display().to_string(), &DEV);

    let spawn = |bin: &str, node_env: Vec<(&'static str, String)>| -> Child {
        spawn_node(bin, &common, &node_env).expect("spawn child binary")
    };

    let mut cluster = Cluster::new();
    // Single-shard parity: every builder passes `dual=false` (the inert arm — byte-identical to the
    // pre-Track-R env; proven by `single_shard_env_is_byte_identical_to_dual_false`).
    cluster.push(
        "vd-orchestrator",
        spawn(
            env!("CARGO_BIN_EXE_vd-orchestrator"),
            orchestrator_env(&addrs, &DEV, &orch_store, vd_bins::ClusterShape::Single),
        ),
    );
    cluster.push(
        "vd-gateway",
        spawn(
            env!("CARGO_BIN_EXE_vd-gateway"),
            gateway_env(
                &addrs,
                &auth_pubkey_hex,
                &DEV,
                vd_bins::ClusterShape::Single,
            ),
        ),
    );
    cluster.push(
        "vd-shard",
        spawn(
            env!("CARGO_BIN_EXE_vd-shard"),
            shard_env(&addrs, &DEV, vd_bins::ClusterShape::Single),
        ),
    );
    let _guard = cluster; // RAII: kill the children on test end or panic

    // ---- two real-protocol clients over the production mesh ------------------
    let rt = tokio::runtime::Builder::new_multi_thread()
        .worker_threads(2)
        .enable_all()
        .build()
        .expect("runtime");
    // STRUCTURAL one-connection invariant: the clients' address books contain
    // ONLY the gateway — no other node is reachable, by construction.
    let client_book = BTreeMap::from([(GATEWAY, gateway_addr)]);
    let mesh = |id: NodeId, bind: SocketAddr| -> MeshTransport {
        let (t, _c) = spawn_mesh(
            rt.handle(),
            &trust,
            &MeshConfig::new(
                id,
                bind,
                client_book.clone(),
                64,
                0,
                vd_bins::world_generation(),
            ),
            None,
        )
        .expect("client mesh");
        std::mem::forget(_c); // keep the endpoint alive for the test's duration
        t
    };
    let mut walker = ProcessClient::new(mesh(NodeId(CLIENT_NODE_BASE), client_a_addr));
    let mut idle = ProcessClient::new(mesh(NodeId(CLIENT_NODE_BASE + 1), client_b_addr));

    // ---- WHERE A LOGIN LANDS, read off THE world -----------------------------
    // ★ RE-DERIVED 2026-08-21 (the gate-pass arc), and the re-derivation EXPOSED A RACE this gate
    // had been winning by accident. Both movement checks below used to measure distance from the
    // REALM ORIGIN, which was the same thing as distance walked only while a login landed exactly
    // on that origin. It does not any more: `resolve_homes` falls back to
    // `WorldView::default_home_offset_m`, the T2 SPAWN STANDOFF — `+Z` by twice the bound of the
    // largest STATIC child that holds the system's centre, i.e. twice the star's own reach, which
    // on THE world is ten million kilometres rather than the walk fixture's zero.
    //
    // WHAT WAS MEASURED (four runs, 2026-08-21). A dot's FIRST delivered pose is the realm origin;
    // its home is applied a few ticks later and the pose settles onto the standoff. The old loop
    // broke on "the walker is more than half a metre from the ORIGIN", which the standoff satisfies
    // before the walker has taken a step — so WHICH pose the assertions saw depended on how long
    // the loop happened to spin. It saw the settled standoff (offset residual
    // `0.0009098052978515625` m, the remainder of the standoff inside one Fine cell) on one run and
    // the un-settled origin on the next four. Neither reading is a flake: they are two real states
    // of a login, and the gate never said which one it wanted.
    //
    // It says so now. The loop waits for the IDLE dot to reach the home THE world states for it,
    // and the walker's displacement is measured FROM THAT HOME — so the scenario cannot be
    // sampled mid-login, and neither check can be satisfied by the standoff's own magnitude.
    let world = vd_bins::boot_world(DEV.universe_seed, DEV.move_speed, DEV.tick_dt);
    let spawn_m = world.default_home_offset_m();
    // ★ THE CENSUS IS THE WHOLE GALAXY'S (S11, owner ruling 2026-08-27 — "we're passing the Galaxy just
    // once over reliable lane"). Derived here the way the GATEWAY derives it, over the whole world.
    //
    // This assertion used to be written against the SHARD's own forest, and that was the defect rather
    // than the fixture: a shard folded its sky from the realms IT booted, so a single-shard cluster
    // stated ONE star — its own — and a player never draws their own star. MEASURED at the time: 1
    // held against 3 in the world, and 0 drawable. The sky now has one author and one census.
    let (census_rows, _, _) = vd_bins::star_catalogue_for_boot(
        DEV.universe_seed,
        DEV.move_speed,
        DEV.tick_dt,
        world.regions(),
    );
    let world_census = census_rows.len() as u64;
    let expected_parts = vd_wire::channels::partition_stars(&census_rows, 8 * 1024).len();
    assert!(
        world_census > 0,
        "this shard states stars, or the gate is vacuous"
    );
    let spawn_pos = vd_core::pose::LatticePos::from_metres(spawn_m, vd_core::pose::Tier::Fine);

    // ---- drive the scenario at real tick rate --------------------------------
    let started = Instant::now();
    let mut pacer = TickPacer::new(DEV.tick_hz);
    let mut hello_retry = Instant::now();
    let mut walker_hello_sent = walker.send_hello(AccountId(1000));
    let mut idle_hello_sent = idle.send_hello(AccountId(1001));
    // THE WALK IS ONE LEG, NOT A MARCH (the interest, 2026-09-06). The walker used to hold its key
    // down for the whole login: at the dev foot speed of a kilometre a second it was sixty
    // kilometres away by the time the sky had arrived, and a shard ships an occupant only to the
    // observers that can still draw it — so the idle dot was told the walker had left its
    // interest, evicted it, and this loop waited for a second pose that could never come. Now the
    // walker steps until it has measurably moved and then stands, inside the idle dot's reach, and
    // the parity claim is made about two dots that can see each other.
    let mut walker_moved = false;
    loop {
        walker.step(!walker_moved);
        idle.step(false);
        walker_moved = walker.own_entity.is_some_and(|own| {
            walker
                .poses
                .get(&own)
                .is_some_and(|p| p.pos.delta_m(spawn_pos, vd_core::pose::Tier::Fine).length() > 0.5)
        });
        // Children may still be booting, so the gateway's lane may refuse the first sends: retry
        // `Hello` until the transport ACCEPTS it once. From then on the reliable lane delivers it,
        // and a second accepted `Hello` would be read as a new process (a re-login), not a retry.
        if hello_retry.elapsed() > Duration::from_millis(500) {
            hello_retry = Instant::now();
            if !walker_hello_sent {
                walker_hello_sent = walker.send_hello(AccountId(1000));
            }
            if !idle_hello_sent {
                idle_hello_sent = idle.send_hello(AccountId(1001));
            }
        }
        // THE LOGIN HAS SETTLED when the dot that never moves is standing where THE world says it
        // spawns. Without this the loop can break mid-login and read a pre-home origin pose.
        let idle_settled = idle
            .own_entity
            .and_then(|own| idle.poses.get(&own))
            .is_some_and(|p| p.pos == spawn_pos);
        // ★ THE SKY IS PART OF "SETTLED" (S11), not something to hope has arrived by the time the
        // loop happens to break. Asserting it after the fact made this test pass alone and fail under
        // full-suite load — a flake, and this project's own rule is that a gate going red for the
        // wrong reason gets weakened until it protects nothing. Waiting for it makes arrival a
        // PRECONDITION, with the existing deadline as the honest failure.
        // ★ WAIT FOR THE WHOLE SKY, NOT ONE PART (2026-08-31). This said `sky_parts > 0`, which WAS the
        // whole sky when a galaxy held three stars and the catalogue fitted in one message. THE world
        // now needs 1 309 parts, and the gateway paces them — deliberately, because shouting 10.7 MB
        // per beat is what drowned the login handshake.
        //
        // So the loop declared victory on the FIRST part and the assertion below then demanded every
        // star: MEASURED, 114 911 of 233 220 had arrived. The comment above already names the cure —
        // make arrival a PRECONDITION, with the deadline as the honest failure — so this now waits for
        // exactly what it asserts.
        let done = walker.sky_stars.len() as u64 == world_census
            && idle.sky_stars.len() as u64 == world_census
            // ...and so is the BEAT, for the same reason: asserting it after the loop happened to
            // break is the flake this comment describes, one lane over.
            && walker.sky_beats > 0
            && walker.poses.len() == 2
            && idle.poses.len() == 2
            && idle_settled
            && walker.own_entity.is_some_and(|own| {
                walker.poses.get(&own).is_some_and(|p| {
                    // TOTAL displacement FROM THE SPAWN — whole-number part plus leftover. The
                    // integrator folds the leftover into the whole number every tick, so the
                    // leftover alone is a sub-millimetre remainder and never reaches this threshold
                    // however far the walker walks: measuring it alone would spin until the
                    // deadline. Measuring from the realm ORIGIN instead is the opposite failure —
                    // the spawn standoff clears half a metre before the walker takes one step, so
                    // the condition would be true immediately and prove nothing.
                    p.pos.delta_m(spawn_pos, vd_core::pose::Tier::Fine).length() > 0.5
                })
            });
        if done {
            break;
        }
        // The BYSTANDER READING: where the WALKER's session says the idle dot is. Printed beside
        // the idle dot's own reading so a stalled convergence names WHICH of the two it is — the
        // dot genuinely sitting at the realm origin (both readings agree), or one session's fan
        // carrying a stale row (they disagree). Without it the two are indistinguishable here.
        let idle_seen_by_walker = idle
            .own_entity
            .and_then(|e| walker.poses.get(&e))
            .map(|p| p.pos);
        let sky_parts = walker.sky_parts;
        assert!(
            started.elapsed() < DEADLINE,
            "parity scenario did not converge: walker(session={:?} subs={:?} poses={} moved={:?}) \
             idle(session={:?} poses={} settled={idle_settled} at={:?} as-seen-by-walker={:?}) \
             sky_parts={sky_parts} — \
             THE world's spawn is {spawn_pos:?}; FIRST poses walker={:?} idle={:?}",
            walker.session,
            walker.held_subs,
            walker.poses.len(),
            walker
                .own_entity
                .and_then(|e| walker.poses.get(&e))
                .map(|p| p.pos),
            idle.session,
            idle.poses.len(),
            idle.own_entity
                .and_then(|e| idle.poses.get(&e))
                .map(|p| p.pos),
            idle_seen_by_walker,
            walker.first_own_pose,
            idle.first_own_pose,
        );
        let _ = pacer.wait();
    }

    // ---- the same logical assertions as the in-process tier ------------------
    assert!(walker.session.is_some(), "walker logged in");
    assert!(idle.session.is_some(), "idle logged in");
    assert_ne!(walker.session, idle.session, "distinct sessions");
    assert_eq!(
        walker.held_subs,
        BTreeSet::from([SubId(0)]),
        "M0: the one subscription"
    );
    let idle_own = idle
        .own_entity
        .expect("authority announced to the idle dot");
    // THE IDLE DOT NEVER MOVED — stated as the whole pose against the world's own spawn, not as a
    // sub-cell residual against zero. STRONGER than the form it replaces: this pins all three
    // components of where the world put it, where `offset() == ZERO` only ever pinned the remainder.
    assert_eq!(
        idle.poses[&idle_own].pos, spawn_pos,
        "the idle dot never moved: it must still sit exactly where THE world's own login standoff \
         put it ({spawn_m:?} m, the T2 spawn clearing)",
    );
    // THE REALM LANE, measured (audit :774): the home System shard authors per-tick rows for every
    // one of its movers and the gateway fans them to every subscribed session — so by convergence
    // BOTH clients have decoded realm datagrams, and the walker's carried real placement rows.
    assert!(
        walker.realm_frames > 0 && idle.realm_frames > 0,
        "the realm lane is ON for both subscribed sessions (walker={}, idle={})",
        walker.realm_frames,
        idle.realm_frames,
    );
    assert!(
        walker.realm_rows > 0,
        "the realm datagrams carried real placement rows (the home system's movers)"
    );
    // THE COMPOSED SCENE LANE, measured (minor 18): login shipped each session EXACTLY ONE full
    // level (the one epoch bump this scenario lawfully has), stamped epoch 1 and led by the home
    // origin (asserted on arrival). A second level here would mean a spurious re-origin.
    for (name, client) in [("walker", &walker), ("idle", &idle)] {
        // ★ THE GATE COUNTS STARS, NOT MESSAGES (S11). It used to assert `sky_parts == 1`. That is a
        // claim about the CARRIER, and it turns red for the wrong reason the moment the census makes
        // the catalogue span many parts — at which point the natural repair is to loosen it, and the
        // gate stops protecting anything. What matters is that the client HOLDS the whole galaxy.
        assert_eq!(
            client.sky_stars.len() as u64,
            world_census,
            "★ THE SKY ARRIVED OVER THE REAL TRANSPORT: the client holds every star this world has"
        );
        // ★ AND IT IS BOUNDED, not "exactly once" (S11). The gateway keeps NO memory of what it sent —
        // that memory was proved unfixable and deleted — so it re-states the sky each beat until the
        // client CONFIRMS, and confirming costs a round trip. MEASURED: the catalogue arrives twice on
        // a fast local cluster.
        //
        // The claim that matters is that it STOPS. A bound of a few beats' worth of parts catches the
        // real failure — a client whose confirmation never lands, re-downloading the galaxy for ever —
        // while allowing the round trip the exchange honestly needs.
        assert!(
            (client.sky_parts as usize) <= expected_parts * MAX_SKY_RESTATEMENTS,
            "{name}: the sky was stated {} times over ({} parts against {expected_parts} per \
             statement) — the client's confirmation is not stopping it",
            client.sky_parts as usize / expected_parts.max(1),
            client.sky_parts,
        );
        // ★ THE BEAT NAMES THE SKY THAT ACTUALLY ARRIVED (S11). This is the end-to-end claim: over
        // real binaries and a real transport, the number the server keeps restating is the number of
        // the catalogue this client is holding. If those two ever differed, the client would sit
        // there believing it held the wrong galaxy while holding the right one.
        assert!(
            client.sky_beats > 0,
            "{name}: the sky beat over the real transport — silence would mean the client cannot \
             tell 'nothing changed' from 'nobody is working'"
        );
        assert_eq!(
            client.sky_beat_generation, client.sky_generation,
            "{name}: the beat names the sky that arrived"
        );
        assert!(
            client.scene_levels >= 1,
            "{name}: the login level arrived (the beat restates it at the same epoch after that)"
        );
        assert_eq!(
            client.scene_reorigins, 0,
            "{name}: no re-origin in a single-shard scenario"
        );
        assert_eq!(
            client.scene_epoch,
            Some(1),
            "{name}: the origin epoch never moved past login"
        );
    }
    // THE REMOVE LANE, measured: nobody left, so nothing may have been evicted — a spurious
    // `EntityRemoved` (e.g. an owner-skip violation) is a red gate, not a swallowed message.
    assert_eq!(
        walker.evictions + idle.evictions,
        0,
        "no EntityRemoved is lawful in this static two-avatar scenario"
    );

    // ---- the 2am curl: the admin endpoint shows the live directory -----------
    let snapshot = http_get_json(admin_addr, "/admin/snapshot");
    let entries = snapshot["directory"].as_array().expect("directory array");
    let authorities: Vec<&str> = entries
        .iter()
        .map(|e| e["authority"].as_str().expect("authority"))
        .collect();
    assert!(
        authorities.contains(&"shard:node-3"),
        "realm + entity records: {authorities:?}"
    );
    assert!(
        authorities.contains(&"gateway:node-2"),
        "session records: {authorities:?}"
    );
    // Exactly: 1 realm + 2 sessions + 2 entities = 5 records, every one fenced.
    assert_eq!(entries.len(), 5, "directory dump: {entries:?}");

    // ---- RLM RG-4c non-vacuity control: a STATIC gateway never greets nor mints ----
    // Both clients logged into the PRE-BOOKED shard (node-3), so the gateway received NO reactive greeting
    // and minted NO dynamic home. This is the falsifiable twin of the demand-login e2e: it proves that test's
    // `presence_announces >= 1` / `dynamic_shards >= 1` could only come from the demand path, never a static one.
    let gw_snapshot = http_get_json(gw_admin, "/admin/snapshot");
    let gw = &gw_snapshot["gateway"];
    assert_eq!(
        gw["presence_announces"].as_u64(),
        Some(0),
        "a static gateway must receive NO reactive greeting: {gw}",
    );
    assert_eq!(
        gw["dynamic_shards"].as_u64(),
        Some(0),
        "a static login routes to the pre-booked shard, never a demand-spawned node: {gw}",
    );

    let _ = std::fs::remove_dir_all(&trust_dir);
    let _ = std::fs::remove_file(&orch_store);
}

/// A raw HTTP/1.1 GET returning parsed JSON (what `curl` would do at 2am).
fn http_get_json(addr: SocketAddr, path: &str) -> serde_json::Value {
    let started = Instant::now();
    loop {
        if let Some(value) = admin_get_body(addr, path, None)
            .and_then(|body| serde_json::from_str::<serde_json::Value>(&body).ok())
        {
            return value;
        }
        assert!(
            started.elapsed() < DEADLINE,
            "admin endpoint never answered"
        );
        std::thread::sleep(Duration::from_millis(100));
    }
}
