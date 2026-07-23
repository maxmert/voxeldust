//! # vd-tests — accumulated scenario suites
//!
//! Shared scenario builders/fixtures used by the integration tests in `tests/tests/`.
//! Standing rule: every phase ADDS scenarios; nothing is deleted. The accumulated
//! suite re-running green is the release gate for every later phase.

use std::collections::{BTreeMap, BTreeSet};
use vd_connection_plane::gateway::{
    GatewayConfig, GatewayStats, TransportTuning, register_gateway,
};
use vd_connection_plane::tickets;

use vd_core::entity_kind::{DurabilityClass, EntityKind};
use vd_core::pose::{FrameRef, RealmId, StampedPose};
use vd_core::{AccountId, EntityId, EpochId, Fence, NodeId, SessionId, TickId, TransferId};
use vd_harness::client::{DeliveredWorldView, InputCmd, ScriptedClient};
use vd_harness::fabric::{CrashWhen, FabricTransport, FaultFabric};
use vd_harness::oracle::{
    AuthorityViolation, verify_authority_settled, verify_authority_unique_excluding,
    verify_transient_authority_held_excluding, verify_transient_conservation_tick_excluding,
    verify_transient_loss_budget,
};
use vd_harness::topology::{InspectReport, StaggerPlan, Topology};
use vd_node::ShardNode;
use vd_node::app::{NodeConfig, build_app};
use vd_node::follower::register_clock_follower;
use vd_node::orchestrator::{DirectoryRes, OrchestratorConfig, register_orchestrator_with_store};
use vd_node::saga_runtime::{ActiveTransfer, SagaRuntimeRes};
use vd_sim::capability::{CapRequest, NodeKind, ShardProfile};
use vd_sim::directory::DirectoryTuning;
use vd_sim::io::mem::MemStore;
use vd_sim::saga::{LivenessTuning, SagaCtx};
use vd_sim::stub::{StubConfig, register_stub_shard};
use vd_wire::seams::directory::{AuthorityRef, DirectoryKey, OwnerRecord};

/// The canonical P1 node ids (one orchestrator, one gateway, one stub shard;
/// clients from 100 upward).
pub const ORCH: NodeId = NodeId(1);
pub const GATEWAY: NodeId = NodeId(2);
pub const SHARD: NodeId = NodeId(3);
/// The transfer DESTINATION shard (P2): a second stub in a distinct realm a player crosses INTO.
pub const DEST: NodeId = NodeId(4);

/// The auth service's signing key for test-minted login tickets.
pub const AUTH_SIGNING_KEY: [u8; 32] = [0x42; 32];

#[must_use]
pub fn auth_verifying_key() -> [u8; 32] {
    ed25519_dalek::SigningKey::from_bytes(&AUTH_SIGNING_KEY)
        .verifying_key()
        .to_bytes()
}

/// P1 world parameters shared by every scenario.
#[must_use]
pub fn stub_config() -> StubConfig {
    StubConfig {
        realm: RealmId::System(7),
        // Single-realm (co-hosting is exercised by the --triple straight-walk smoke, not these cluster
        // scenarios — their crossings target realms hosted by OTHER shards). Byte-identical default.
        held_realms: StubConfig::single_realm(RealmId::System(7)),
        frame: FrameRef::SystemSpace { system_seed: 7 },
        move_speed_mps: 2.0,
        tick_dt_s: 0.05,
        // Realm SUBJECTIVE-time factor; 1.0 = objective universe time (byte-identical default).
        time_multiplier: 1.0,
        orchestrator: ORCH,
        mint_seed: 11,
        // Large window: the oracle must see a whole short test run.
        input_log_capacity: 1_000_000,
        // A small NON-ZERO recheck interval (the EXISTING knob, not a new magic number) so the
        // SOURCE periodically re-reads its REALM-lease head and self-fences the whole shard on a
        // lease revocation. The 1c.8 per-entity granted-key poll is GONE (1d.5b.2): the source
        // ENTITY self-fence is now driven by the saga-pushed ordered `Demote`, not a directory poll.
        realm_recheck_interval: 4,
        // D-3 INERT here: the cluster scenarios do not exercise the lease-renewal heartbeat (the D-3
        // cells set it explicitly). 0 = no heartbeat, matching pre-D-3 behavior.
        lease_renew_interval_ticks: 0,
        // D-3 Slice 5 INERT here (grace 0 vetoes the proactive self-fence even with recheck active);
        // the self-fence cells set it explicitly. Pre-D-3 behavior: no proactive fence.
        self_fence_grace_ticks: 0,
        snapshot_datagram_budget: 1100,
        // Slice 3e / C-3: the containment re-home trigger tuning. INERT here (the cluster scenarios plant
        // no `RealmRegions`, so `evaluate_realm_boundaries` early-returns — behaviour-identical).
        boundary: vd_core::geometry::BoundaryTuning::DEFAULT,
        request_ttl_ticks: 0,
    }
}

/// The DEST stub's params: a DISTINCT realm (`System(8)`) + frame + mint seed so it is a
/// genuinely separate shard the player transfers INTO (not a clone of the source).
#[must_use]
pub fn dest_stub_config() -> StubConfig {
    StubConfig {
        realm: RealmId::System(8),
        // Override the inherited `{System(7)}` — this shard hosts System 8 (single-realm).
        held_realms: StubConfig::single_realm(RealmId::System(8)),
        frame: FrameRef::SystemSpace { system_seed: 8 },
        mint_seed: 17,
        ..stub_config()
    }
}

/// task #149 — the CO-HOSTING login shard: the SAME [`SHARD`] node HOSTS its System 7 realm AND co-hosts the
/// nested Planet 7 + Area 7 child realms (the un-hosted-child cure). So a dot that walks from System 7 into
/// Planet 7 / Area 7 re-homes to a realm THIS node ALSO heads — `head(Realm(Area 7))` resolves to SHARD
/// (SOURCE==DEST), the degenerate case of the ONE uniform orchestrator saga. Matches the worldgen forest
/// (System 7 ⊃ Planet 7 ⊃ Area 7), so `plant_seed_neighbourhood` gives the shard all three regions.
#[must_use]
pub fn cohost_stub_config() -> StubConfig {
    StubConfig {
        held_realms: BTreeSet::from([RealmId::System(7), RealmId::Planet(7), RealmId::Area(7)]),
        ..stub_config()
    }
}

/// task #149 — the SOURCE==DEST cluster: orchestrator + gateway + ONE co-hosting [`SHARD`] (System 7 +
/// Planet 7 + Area 7). A durable dot that walks into Planet 7 / Area 7 drives the uniform crossing saga with
/// the SAME node as both source and dest — proving the co-hosted re-home completes (no local short-circuit).
#[must_use]
pub fn p2_cluster_cohost(fabric: &FaultFabric, max_sessions: usize) -> Topology {
    build_cluster(
        fabric,
        max_sessions,
        vec![GATEWAY, SHARD],
        vec![(SHARD, cohost_stub_config())],
        StaggerPlan::lockstep(),
        MemStore::new(),
        default_directory_tuning(),
    )
}

/// C-6c — the GALAXY shard (`RealmId::System(GALAXY_SEED)`, the seed forest's between-systems space): the
/// shard that OWNS the between-space + System 7/System 8 as its children, so a SIBLING crossing routes
/// THROUGH it (leave System 7 → land in the Galaxy → the Galaxy shard sees the entry into System 8). This
/// is the shard the 3-shard round-trip needs so `head(Realm(Galaxy))` resolves and authority can REST in
/// the between-space. Matches `vd_core::worldgen`'s `GALAXY = System(1)`.
pub const GALAXY: NodeId = NodeId(5);
/// The Galaxy realm seed — must match `vd_core::worldgen`'s Galaxy (`System(1)`), the between-systems space.
pub const GALAXY_SEED: u64 = 1;

/// The GALAXY stub's params: hosts `System(GALAXY_SEED)` (the between-space), a distinct mint seed so its
/// entities never alias the systems' (source=11, dest=17, galaxy=23).
#[must_use]
pub fn galaxy_stub_config() -> StubConfig {
    StubConfig {
        realm: RealmId::System(GALAXY_SEED),
        // Override the inherited `{System(7)}` — this shard hosts the Galaxy (single-realm; its System
        // 7/8 children are hosted by OTHER shards, so no co-hosting is needed on the Galaxy).
        held_realms: StubConfig::single_realm(RealmId::System(GALAXY_SEED)),
        frame: FrameRef::SystemSpace {
            system_seed: GALAXY_SEED,
        },
        mint_seed: 23,
        ..stub_config()
    }
}

/// The shared cluster spine: an orchestrator + a gateway + N stub shards on a fabric-backed
/// topology, built in a fixed order (orchestrator, gateway, then shards in list order — the
/// `p1_cluster` baseline must stay byte-identical). `clock_peers` are the nodes the
/// orchestrator drives the universe clock to; EVERY follower shard must be listed or its clock
/// never advances. Clients are added by the caller.
/// The cluster orchestrator's config (shared by `build_cluster` + the D-6 orchestrator-kill rebuild, so
/// a rebuilt orchestrator is built IDENTICALLY — same reserve_chunk/tuning → a clean recover).
#[must_use]
pub fn orch_config(
    clock_peers: Vec<NodeId>,
    roster: BTreeMap<NodeId, ShardProfile>,
) -> OrchestratorConfig {
    OrchestratorConfig {
        epoch: EpochId(1),
        reserve_chunk: 1024,
        clock_peers,
        directory: DirectoryTuning {
            lease_ttl_ticks: 1_000,
            ..DirectoryTuning::default()
        },
        // Slice 2a: the deadline producer runs LIVE in the capstone cluster (dev values 8/24).
        saga: vd_sim::saga::SagaTuning::default(),
        // D-3: kill-equivalent (n == 1) so the existing crash cells confirm a permanent kill on the
        // first NodeUnreachable; the CSCALE-1 flap cells override to n == 3 via set_liveness_tuning.
        liveness: vd_sim::saga::LivenessTuning::default(),
        // D-37: the re-home target roster (the D-6 rebuild passes the IDENTICAL one → clean recover).
        roster,
    }
}

/// D-37: the re-home target roster for a stub cluster — every stub shard maps to the EMPTY profile (a
/// bare-point P3 shard has no voxel/block caps). `select_rehome_target` picks the lowest LIVE one.
fn stub_roster(shard_ids: impl IntoIterator<Item = NodeId>) -> BTreeMap<NodeId, ShardProfile> {
    let empty = ShardProfile::build(CapRequest::default()).expect("empty profile is coherent");
    shard_ids.into_iter().map(|id| (id, empty)).collect()
}

/// The default orchestrator directory tuning: a long lease, the REAPER INERT (interval 0). Every cluster
/// uses this EXCEPT the D-37 standing-re-home cell ([`reaping_directory_tuning`]) — keeping the reaper
/// CELL-3-SPECIFIC so a short lease does not spuriously lapse + reap the other crash cells' live owners.
fn default_directory_tuning() -> DirectoryTuning {
    DirectoryTuning {
        lease_ttl_ticks: 1_000,
        ..DirectoryTuning::default()
    }
}

/// D-37 Slice 3b: the CELL-3-SPECIFIC directory tuning that ARMS the standing re-home — a SHORT lease (the
/// dead source's Entity lease is already lapsed by the time it is confirmed dead) + a non-zero reaper
/// interval (the sweep that detects the lapsed + confirmed-dead + unlocked orphan and arms its re-home).
/// Deliberately NOT the cluster default (a short global lease would lapse + perturb every other cell).
fn reaping_directory_tuning() -> DirectoryTuning {
    DirectoryTuning {
        lease_ttl_ticks: 16,
        reaper_interval_ticks: 8,
        ..DirectoryTuning::default()
    }
}

fn build_cluster(
    fabric: &FaultFabric,
    max_sessions: usize,
    clock_peers: Vec<NodeId>,
    shards: Vec<(NodeId, StubConfig)>,
    stagger: StaggerPlan,
    orch_store: MemStore,
    directory: DirectoryTuning,
) -> Topology {
    let mut topo = Topology::new(fabric.clone(), stagger);

    let mut orch = build_app(
        NodeConfig {
            node_id: ORCH,
            kind: NodeKind::Orchestrator,
        },
        fabric.register(ORCH),
    );
    let (world, schedule) = orch.parts_mut();
    // D-6: the orchestrator is built against a durable Store. Normal clusters pass a fresh (genesis)
    // `MemStore` (transparent); the D-6 orchestrator-kill driver passes a RETAINED handle so a rebuilt
    // orchestrator re-hydrates the SAME committed WAL.
    let roster = stub_roster(shards.iter().map(|(id, _)| *id));
    // D-37 Slice 3b: override the directory tuning (reaper/lease) — most clusters pass the inert default;
    // the standing-re-home cell passes the reaping tuning.
    let mut oc = orch_config(clock_peers, roster);
    oc.directory = directory;
    register_orchestrator_with_store(world, schedule, &oc, Box::new(orch_store));
    topo.add_node(Box::new(orch));

    let mut gateway = build_app(
        NodeConfig {
            node_id: GATEWAY,
            kind: NodeKind::Gateway,
        },
        fabric.register(GATEWAY),
    );
    // The STABLE routable-shard roster (FORK 5 / 1d.2): EVERY shard in the cluster, so a
    // (render-ready) dest's frames are node-class-dispatchable. The login shard `SHARD` is
    // always a member.
    let known_shards: std::collections::BTreeSet<NodeId> =
        shards.iter().map(|(id, _)| *id).collect();
    let (world, schedule) = gateway.parts_mut();
    register_clock_follower(world, schedule);
    register_gateway(
        world,
        schedule,
        GatewayConfig {
            orchestrator: ORCH,
            shard: SHARD,
            known_shards,
            auth_verifying_key: auth_verifying_key(),
            session_seed: 23,
            tick_hz: 50,
            // D-3 INERT: the cluster scenarios do not exercise the session heartbeat (the D-3 cells do).
            lease_renew_interval_ticks: 0,
            // D-3 Slice 5b INERT here (grace 0 vetoes the proactive self-fence; the self-fence cells set
            // these explicitly). Pre-D-3 behavior: no recheck, no proactive fence.
            session_recheck_interval: 0,
            self_fence_grace_ticks: 0,
            // 3g abort-leg lever INERT for every cluster (behaviour-identical); `arm_gateway_reject`
            // sets it live on the gateway node when a test wants the pre-CAS abort.
            reject_next_prepare: None,
            tuning: TransportTuning {
                max_sessions,
                max_buffered_inputs: TransportTuning::DEFAULT_MAX_BUFFERED_INPUTS,
            },
        },
    );
    topo.add_node(Box::new(gateway));

    for (id, cfg) in shards {
        let mut shard = build_app(
            NodeConfig {
                node_id: id,
                kind: NodeKind::StubShard,
            },
            fabric.register(id),
        );
        let (world, schedule) = shard.parts_mut();
        register_clock_follower(world, schedule);
        register_stub_shard(world, schedule, cfg);
        topo.add_node(Box::new(shard));
    }

    topo
}

/// Build the canonical P1 cluster (orchestrator + gateway + ONE stub shard) onto a
/// fabric-backed topology. Clients are added by the caller.
#[must_use]
pub fn p1_cluster(fabric: &FaultFabric, max_sessions: usize) -> Topology {
    build_cluster(
        fabric,
        max_sessions,
        vec![GATEWAY, SHARD],
        vec![(SHARD, stub_config())],
        StaggerPlan::lockstep(),
        MemStore::new(), // a fresh (genesis) store — this cluster is not rebuilt
        default_directory_tuning(),
    )
}

/// Build the P2 transfer cluster: P1 plus a SECOND stub shard at [`DEST`] in a distinct realm,
/// with `DEST` added to the orchestrator's clock peers so it follows the clock and wins its own
/// realm lease — the source/dest pair a cross-shard transfer hands a player between.
#[must_use]
pub fn p2_cluster(fabric: &FaultFabric, max_sessions: usize) -> Topology {
    build_cluster(
        fabric,
        max_sessions,
        vec![GATEWAY, SHARD, DEST],
        vec![(SHARD, stub_config()), (DEST, dest_stub_config())],
        StaggerPlan::lockstep(),
        MemStore::new(), // a fresh (genesis) store — this cluster is not rebuilt
        default_directory_tuning(),
    )
}

/// D-37 Slice 3b: the P2 transfer cluster with the REAPING directory tuning (short lease + non-zero reaper
/// interval) so the standing reaper-driven re-home (CELL 3) actually fires in the full kill scenario.
/// Identical to [`p2_cluster`] otherwise. CELL-3-specific (the other crash cells keep the inert default).
#[must_use]
pub fn p2_cluster_reaping(fabric: &FaultFabric, max_sessions: usize) -> Topology {
    build_cluster(
        fabric,
        max_sessions,
        vec![GATEWAY, SHARD, DEST],
        vec![(SHARD, stub_config()), (DEST, dest_stub_config())],
        StaggerPlan::lockstep(),
        MemStore::new(),
        reaping_directory_tuning(),
    )
}

/// The P2 transfer cluster whose orchestrator is built against a caller-RETAINED `MemStore` (D-6): the
/// returned handle lets the orchestrator-kill driver rebuild the orchestrator against the SAME committed
/// WAL (the kill-9 analog). Identical to [`p2_cluster`] otherwise.
#[must_use]
pub fn p2_cluster_durable_orch(fabric: &FaultFabric, max_sessions: usize) -> (Topology, MemStore) {
    let store = MemStore::new();
    let topo = build_cluster(
        fabric,
        max_sessions,
        vec![GATEWAY, SHARD, DEST],
        vec![(SHARD, stub_config()), (DEST, dest_stub_config())],
        StaggerPlan::lockstep(),
        store.clone(),
        default_directory_tuning(),
    );
    (topo, store)
}

/// The P2 transfer cluster under a deliberate `StaggerPlan` (D-7b): the source/dest process the
/// handoff at SKEWED local ticks, so a per-tick conservation gate can prove the structural
/// drop-before-promote never double-holds even when one shard lags the other.
#[must_use]
pub fn p2_cluster_staggered(
    fabric: &FaultFabric,
    max_sessions: usize,
    stagger: StaggerPlan,
) -> Topology {
    build_cluster(
        fabric,
        max_sessions,
        vec![GATEWAY, SHARD, DEST],
        vec![(SHARD, stub_config()), (DEST, dest_stub_config())],
        stagger,
        MemStore::new(), // a fresh (genesis) store — this cluster is not rebuilt
        default_directory_tuning(),
    )
}

/// C-6c — the 3-SHARD GALAXY cluster: orchestrator + gateway + THREE stub shards — System 7 (the login
/// shard, [`SHARD`]), the Galaxy between-space ([`GALAXY`], `System(GALAXY_SEED)`), and System 8
/// ([`DEST`]) — so a dot re-homes through the FULL containment chain System 7 → Galaxy → System 8 → Galaxy
/// → System 7. All three shards win their realm leases through the REAL directory, and the gateway's
/// `known_shards` = {SHARD, GALAXY, DEST} routes every shard's frames. This is the harness-tier substrate
/// for the round-trip gate; the seed-forest containment neighbourhood is planted per-shard by the caller.
#[must_use]
pub fn p3_galaxy_cluster(fabric: &FaultFabric, max_sessions: usize) -> Topology {
    build_cluster(
        fabric,
        max_sessions,
        vec![GATEWAY, SHARD, GALAXY, DEST],
        vec![
            (SHARD, stub_config()),
            (GALAXY, galaxy_stub_config()),
            (DEST, dest_stub_config()),
        ],
        StaggerPlan::lockstep(),
        MemStore::new(), // a fresh (genesis) store — this cluster is not rebuilt
        default_directory_tuning(),
    )
}

/// Borrow a topology node downcast to its concrete `ShardNode<FabricTransport>` so scenario code
/// can drive/read it directly (orchestrator = saga producer + directory; gateway = route/stats).
fn with_node<R>(
    topo: &mut Topology,
    id: NodeId,
    f: impl FnOnce(&mut ShardNode<FabricTransport>) -> R,
) -> R {
    let node = topo.node_mut(id).expect("node present");
    let shard = node
        .as_any_mut()
        .expect("ShardNode opts into downcasting")
        .downcast_mut::<ShardNode<FabricTransport>>()
        .expect("the node is a ShardNode");
    f(shard)
}

/// Borrow the orchestrator's concrete node — the saga producer + the directory live there.
fn with_orchestrator<R>(
    topo: &mut Topology,
    f: impl FnOnce(&mut ShardNode<FabricTransport>) -> R,
) -> R {
    with_node(topo, ORCH, f)
}

/// Slice 3g abort-leg: ARM the gateway's one-shot `reject_next_prepare` lever (the SOLE way to drive a
/// crossing-origin durable saga into its pre-CAS abort in a cluster — the gateway is the durable Prepare
/// decider). The NEXT `PrepareSubscribe` the gateway would answer `Ready` instead replies
/// `Prepared{ Rejected(reject) }`, then the gateway self-clears the lever (one-shot). `GatewayConfig` is a
/// live `#[derive(Resource)]` inserted by `register_gateway`, so `resource_mut::<GatewayConfig>()` mutates it.
pub fn arm_gateway_reject(
    topo: &mut Topology,
    reject: vd_wire::seams::transfer_control::PrepareReject,
) {
    with_node(topo, GATEWAY, |gw| {
        gw.world_mut()
            .resource_mut::<GatewayConfig>()
            .reject_next_prepare = Some(reject);
    });
}

/// Read the player's transfer SUBJECT live from the orchestrator directory: the `Session`, the
/// single `Entity` record (the avatar), and the fence it is recorded at — the CAS expectation
/// is the DIRECTORY record's fence, never a hardcoded literal or the shard's own copy.
#[must_use]
pub fn read_subject(topo: &mut Topology) -> (SessionId, EntityId, Fence) {
    with_orchestrator(topo, |orch| {
        let dir = orch.world_mut().resource::<DirectoryRes>();
        let mut session = None;
        let mut entity = None;
        for (key, record) in dir.0.entries() {
            match key {
                DirectoryKey::Session(s) => session = Some(*s),
                DirectoryKey::Entity(e) => entity = Some((*e, record.fence)),
                _ => {}
            }
        }
        let session = session.expect("the session is recorded");
        let (entity, fence) = entity.expect("the avatar entity is recorded");
        (session, entity, fence)
    })
}

/// Trigger a REAL transfer through the orchestrator's saga runtime — the same producer 1d
/// shards will use (never hand-fed acks): it emits the real
/// `PrepareSubscribe → RequestCut → FreezeSource → CommitAuthority` over the fabric.
pub fn trigger_transfer(topo: &mut Topology, ctx: SagaCtx) {
    with_orchestrator(topo, |orch| {
        orch.world_mut()
            .resource_mut::<SagaRuntimeRes>()
            .start_transfer(ctx, GATEWAY);
    });
}

/// Read a realm's recorded fence from the orchestrator directory (D-7: the transient go-token /
/// adopt anchor fence is read from the directory, never a hardcoded literal).
#[must_use]
pub fn realm_fence(topo: &mut Topology, realm: RealmId) -> Fence {
    with_orchestrator(topo, |orch| {
        orch.world_mut()
            .resource::<DirectoryRes>()
            .0
            .head(DirectoryKey::Realm(realm))
            .expect("realm recorded")
            .fence
    })
}

/// The seeded Debris origin (D-7b) — the SINGLE source of the crossing pose's `pos0`/`tick0`, shared
/// by [`seed_transient_crossing`] AND the e2e ballistic-trajectory assertion, so the expected
/// trajectory can NEVER silently drift from what the seed actually writes (no-drift discipline).
pub const TRANSIENT_SEED_POS0: vd_core::glam::DVec3 = vd_core::glam::DVec3::new(4.0, 5.0, 6.0);
/// See [`TRANSIENT_SEED_POS0`] — the seeded crossing pose's analytic-clock origin `tick0`.
pub const TRANSIENT_SEED_TICK0: vd_core::UniverseTick = vd_core::UniverseTick(1);

/// D-7: seed a Debris transient on the SOURCE shard ([`SHARD`]) as a pending Crossing to the DEST
/// realm — the TEST-driven boundary-heuristic stand-in (the autonomous geometric trigger is P4/P5).
/// The matching batch saga must be triggered separately via [`trigger_transfer`] with a Transient
/// ctx carrying the SAME `batch` id. `anchor` is the source's realm-lease fence; `dst_realm_fence`
/// the dest's (both via [`realm_fence`]).
pub fn seed_transient_crossing(
    topo: &mut Topology,
    entity: EntityId,
    batch: TransferId,
    anchor: Fence,
    dst_realm_fence: Fence,
    vel: vd_core::glam::DVec3,
) {
    with_node(topo, SHARD, |s| {
        let pose = vd_core::pose::StampedPose {
            vel,
            ..vd_core::pose::StampedPose::at_rest(
                stub_config().frame,
                TRANSIENT_SEED_POS0,
                TRANSIENT_SEED_TICK0,
            )
        };
        s.world_mut()
            .resource_mut::<vd_sim::stub::OwnedTransients>()
            .0
            .insert(
                entity,
                vd_sim::stub::Transient {
                    pose,
                    anchor_fence: anchor,
                    status: vd_sim::stub::TransientStatus::Crossing {
                        dest: DEST,
                        to_realm: dest_stub_config().realm,
                        dst_realm_fence,
                        batch,
                        to_parent: None,
                    },
                    // Slice 3d: the geometric-trigger prev-offset seed (the test plants no boundaries,
                    // so it is never read — seed to the pose offset for the degenerate first segment).
                    prev_offset: pose.pos.offset(),
                },
            );
    });
}

/// The ambient ROOT realm (`parent: None`) for a crossing-e2e region forest — a huge shell covering
/// everything, so the `container` fold is total (an entity is ALWAYS in ≥1 realm).
pub const CROSSING_ROOT_REALM: RealmId = RealmId::System(0);

/// Build a velocity-safe [`ContainmentBand`](vd_core::geometry::ContainmentBand) for a crossing-e2e region,
/// sized against the dot's own walk speed (the only motion in these fixtures) so it can never flap.
fn crossing_band() -> vd_core::geometry::ContainmentBand {
    vd_core::geometry::ContainmentBand::for_containment_velocity_safe(
        50.0,  // inset (m inside to acquire)
        100.0, // outset_min (m outside to release)
        stub_config().move_speed_mps,
        stub_config().tick_dt_s,
        1.0, // k_safety_extra
    )
    .expect("valid crossing containment band")
}

/// One `RealmRegion` shell at the origin of `realm`'s frame, radius `r`, nested under `parent`.
fn crossing_region(
    realm: RealmId,
    parent: Option<RealmId>,
    r: f64,
) -> vd_core::geometry::RealmRegion {
    use vd_core::pose::{LatticePos, frame_for_realm};
    vd_core::geometry::RealmRegion {
        realm,
        center: LatticePos::local(vd_core::glam::DVec3::ZERO),
        frame: frame_for_realm(realm, None).expect("System realm always resolves a frame"),
        shape: vd_core::geometry::Boundary::Shell { r },
        band: crossing_band(),
        parent,
    }
}

/// Slice 3g (C-3 CONTAINMENT) — install a REGION forest into the SOURCE shard's ([`SHARD`]) `RealmRegions`
/// resource, ARMING the containment re-home trigger (`evaluate_realm_boundaries` early-returns while the
/// registry is empty, so the cluster is behaviour-identical until this plant). The caller owns the region
/// set. INERT until this call.
pub fn plant_crossing_boundaries(
    topo: &mut Topology,
    regions: Vec<vd_core::geometry::RealmRegion>,
) {
    with_node(topo, SHARD, |s| {
        *s.world_mut().resource_mut::<vd_sim::stub::RealmRegions>() =
            vd_sim::stub::RealmRegions::new(regions);
    });
}

/// C-6c — plant the SEED-DERIVED containment neighbourhood (`vd_core::worldgen::realm_neighbourhood_for`)
/// on the shard `node` (its own realm + ancestors + owned children — the EXACT geometry the production
/// `shard.rs` boot computes). ARMS the containment detector on that shard against the canonical forest,
/// so a re-home is driven by real seed geometry, not an authored fixture. `universe_seed` matches the
/// shard's boot seed.
pub fn plant_seed_neighbourhood(
    topo: &mut Topology,
    node: NodeId,
    universe_seed: u64,
    hosted_realm: RealmId,
) {
    let regions = vd_core::worldgen::realm_neighbourhood_for(universe_seed, hosted_realm);
    with_node(topo, node, |s| {
        *s.world_mut().resource_mut::<vd_sim::stub::RealmRegions>() =
            vd_sim::stub::RealmRegions::new(regions);
    });
}

/// task #149 — plant the seed-derived containment neighbourhood for the UNION of a CO-HOSTED held set
/// (`vd_core::worldgen::realm_neighbourhood_for_held`), the EXACT geometry the production `shard.rs` boot
/// computes for a multi-realm shard. A single-realm `plant_seed_neighbourhood` gives only its own realm +
/// ancestors + DIRECT children — so a shard co-hosting `{System 7, Planet 7, Area 7}` needs THIS to evaluate
/// Area 7 (a GRANDCHILD of System 7, absent from System 7's own neighbourhood).
pub fn plant_seed_neighbourhood_held(
    topo: &mut Topology,
    node: NodeId,
    universe_seed: u64,
    held: &BTreeSet<RealmId>,
) {
    let regions = vd_core::worldgen::realm_neighbourhood_for_held(universe_seed, held);
    with_node(topo, node, |s| {
        *s.world_mut().resource_mut::<vd_sim::stub::RealmRegions>() =
            vd_sim::stub::RealmRegions::new(regions);
    });
}

/// C-6c — set the dot whose `entity == subject` on shard `node` to frame-local `off` (find it in `Dots`).
/// Returns true if found + set. The adopted/owned dot's manual write survives into
/// `evaluate_realm_boundaries` the same tick (schedule order: process_inbound → evaluate_realm_boundaries),
/// so the seed-forest detector re-homes it based on its scripted position — the round-trip's waypoint driver.
pub fn set_shard_subject_offset(
    topo: &mut Topology,
    node: NodeId,
    subject: EntityId,
    off: vd_core::glam::DVec3,
) -> bool {
    with_node(topo, node, |s| {
        let dots = s.world_mut().resource_mut::<vd_sim::stub::Dots>();
        for dot in dots.into_inner().0.values_mut() {
            if dot.entity == subject {
                dot.pose.pos = vd_core::pose::LatticePos::local(off);
                return true;
            }
        }
        false
    })
}

/// task #149 — the AUTHORITATIVE pose FRAME of the dot whose `entity == subject` on shard `node`, or `None`
/// if that shard holds no such dot. The frame flips to the dest realm's canonical frame once a re-home's
/// dest adopt runs `rebind_pose_to_dest` — on a co-hosted (source==dest) re-home that adopt is THIS node's
/// own promote, so the frame advances to the child realm (Planet/Area) here. The source==dest e2e reads
/// this to prove the crossed pose ends in the `AreaLocal` frame (the "Area label never flips" fix).
#[must_use]
pub fn shard_subject_frame(
    topo: &mut Topology,
    node: NodeId,
    subject: EntityId,
) -> Option<FrameRef> {
    with_node(topo, node, |s| {
        s.world_mut()
            .resource::<vd_sim::stub::Dots>()
            .0
            .values()
            .find(|d| d.entity == subject)
            .map(|d| d.pose.frame)
    })
}

/// C-6c — the node the directory records as `head(Entity(subject))`'s authority (arm-agnostic `.node()`),
/// or `None` if the subject is not (yet) recorded. This is `head(Realm(subject))` in the containment sense:
/// the shard that OWNS the dot's authority — it flips through System 7 → Galaxy → System 8 → Galaxy →
/// System 7 as the dot re-homes. Read straight off the orchestrator's ONE directory.
#[must_use]
pub fn entity_head_node(topo: &mut Topology, subject: EntityId) -> Option<NodeId> {
    with_orchestrator(topo, |orch| {
        orch.world_mut()
            .resource::<DirectoryRes>()
            .0
            .head(DirectoryKey::Entity(subject))
            .map(|r| r.authority.node())
    })
}

/// Slice 3g (C-3 CONTAINMENT) — the crossing-e2e convenience: plant a 3-level forest on the SOURCE
/// (root ⊃ own(`System(7)`) ⊃ dest(`System(8)`)), all origin-coincident at the dot's SPAWN offset
/// (`DVec3::ZERO` in the source frame — where login places the avatar, stub.rs `login`). The dot is a
/// MEMBER of the DEST region from spawn, so its deepest container is the DEST realm — `should_rehome`
/// fires a re-home to `System(8)` (the containment twin of the old `should_commit` Inward path). The OWN
/// region (`System(7)`) is the realm the shard owns the subject in, so a dot inside ONLY it would NOT
/// re-home; the deeper dest region is what triggers the crossing.
///
/// **M2 (load-bearing):** the dest region's `realm` MUST byte-equal the realm the DEST shard actually holds
/// ([`dest_stub_config`]`.realm`) so the orchestrator's `handle_crossing_request` resolves the dest head and
/// starts the saga; a mismatch silently counts `crossing_unresolved` (a vacuous green). The dest shell (r =
/// 1000 m) dwarfs the ~0.1 m/tick walk, so the dot stays a member for the whole run.
pub fn plant_one_crossing_shell(topo: &mut Topology) {
    let root = crossing_region(CROSSING_ROOT_REALM, None, 1.0e9);
    let own = crossing_region(stub_config().realm, Some(CROSSING_ROOT_REALM), 100_000.0);
    // M2: the dest region's realm byte-equals the realm the DEST shard holds, so the dest head resolves.
    let dest = crossing_region(dest_stub_config().realm, Some(stub_config().realm), 1000.0);
    plant_crossing_boundaries(topo, vec![root, own, dest]);
}

/// C-3 CONTAINMENT — DISARM the SOURCE shard's containment trigger by emptying its `RealmRegions`. Under
/// containment a re-home is POSITION-driven, so an abort resets the source's cooldown and the still-in-region
/// dot re-fires the SAME re-home next tick (the intended behaviour: the entity did not move). An abort test
/// that wants to observe EXACTLY ONE crossing (the one that aborts) calls this once the crossing has started,
/// so the source stops re-homing — leaving the abort as the terminal event under test.
pub fn clear_crossing_boundaries(topo: &mut Topology) {
    with_node(topo, SHARD, |s| {
        *s.world_mut().resource_mut::<vd_sim::stub::RealmRegions>() =
            vd_sim::stub::RealmRegions::default();
    });
}

/// Slice 3g — seed an OWNED `Held{outbound: None}` Debris transient on the SOURCE ([`SHARD`]), positioned
/// INSIDE the planted crossing band (at `pos`, in the source frame), so the geometric dwell fires its
/// Transient fan-out (`TransientCrossingRequest`) — the HR2 second-class autonomous crossing (no
/// test-seeded `Crossing` status, no hand-fed batch). `anchor` is the source's realm-lease fence (via
/// [`realm_fence`]). The dot is at rest (`vel = 0`), so its `prev_offset == pos` (a degenerate first
/// segment) and it stays a member of the shell centered at spawn.
pub fn seed_held_transient(
    topo: &mut Topology,
    entity: EntityId,
    anchor: Fence,
    pos: vd_core::glam::DVec3,
) {
    with_node(topo, SHARD, |s| {
        let pose =
            vd_core::pose::StampedPose::at_rest(stub_config().frame, pos, TRANSIENT_SEED_TICK0);
        s.world_mut()
            .resource_mut::<vd_sim::stub::OwnedTransients>()
            .0
            .insert(
                entity,
                vd_sim::stub::Transient {
                    pose,
                    anchor_fence: anchor,
                    status: vd_sim::stub::TransientStatus::Held { outbound: None },
                    prev_offset: pose.pos.offset(),
                },
            );
    });
}

/// C-6c — seed an OWNED `Held` transient (a Debris) on ANY shard `node` at frame-local `pos` (the generic
/// twin of [`seed_held_transient`], which is SHARD-only). `anchor` is that shard's realm-lease fence. A
/// transient re-homes via the batched `TransientGo` path (HR2 second class) with NO client cut-marker, so
/// it self-drives across shards — the round-trip subject whose OWNERSHIP (which shard holds it) flips
/// through the containment chain without the gateway session-migration a durable dot's directory head needs.
pub fn seed_held_transient_on(
    topo: &mut Topology,
    node: NodeId,
    frame: FrameRef,
    entity: EntityId,
    anchor: Fence,
    pos: vd_core::glam::DVec3,
) {
    with_node(topo, node, |s| {
        let pose = vd_core::pose::StampedPose::at_rest(frame, pos, TRANSIENT_SEED_TICK0);
        s.world_mut()
            .resource_mut::<vd_sim::stub::OwnedTransients>()
            .0
            .insert(
                entity,
                vd_sim::stub::Transient {
                    pose,
                    anchor_fence: anchor,
                    status: vd_sim::stub::TransientStatus::Held { outbound: None },
                    prev_offset: pos,
                },
            );
    });
}

/// C-6c — set a HELD transient's frame-local offset on shard `node` (find it in `OwnedTransients`). Returns
/// true if found + set. The manual write survives into `evaluate_realm_boundaries` the same tick, so the
/// seed-forest detector re-homes the transient based on its scripted waypoint position.
pub fn set_transient_offset_on(
    topo: &mut Topology,
    node: NodeId,
    entity: EntityId,
    off: vd_core::glam::DVec3,
) -> bool {
    with_node(topo, node, |s| {
        let mut owned = s
            .world_mut()
            .resource_mut::<vd_sim::stub::OwnedTransients>();
        if let Some(t) = owned.0.get_mut(&entity) {
            t.pose.pos = vd_core::pose::LatticePos::local(off);
            true
        } else {
            false
        }
    })
}

/// C-6c — remove the transient `entity` from shard `node`'s `OwnedTransients` (a no-op if absent). Used by
/// the round-trip driver to drop a completed leg's fresh subject so the next leg's holder read is clean.
pub fn remove_transient_on(topo: &mut Topology, node: NodeId, entity: EntityId) {
    with_node(topo, node, |s| {
        s.world_mut()
            .resource_mut::<vd_sim::stub::OwnedTransients>()
            .0
            .remove(&entity);
    });
}

/// C-6c — which of `nodes` currently HOLDS the transient `entity` authoritatively (the `is_held()` subset of
/// its `owned_transients`), or `None` if no shard holds it. This is the transient's OWNERSHIP head (the HR2
/// analog of `head(Realm(subject))` — a transient has no directory row by design). It flips through
/// System 7 → Galaxy → System 8 → Galaxy → System 7 as the transient re-homes.
#[must_use]
pub fn transient_holder(
    reports: &[(NodeId, InspectReport)],
    nodes: &[NodeId],
    entity: EntityId,
) -> Option<NodeId> {
    nodes.iter().copied().find(|n| {
        reports
            .iter()
            .find(|(id, _)| id == n)
            .is_some_and(|(_, r)| r.owned_transients.iter().any(|(e, _)| *e == entity))
    })
}

/// Total transients DROPPED as a LOSS (self-fence, no hand-off) across the source + dest shards
/// (D-7) — 0 on the happy path (a clean adopt-before-drop hand-off is NOT a loss).
#[must_use]
pub fn transient_dropped_total(topo: &mut Topology) -> u64 {
    [SHARD, DEST]
        .into_iter()
        .map(|n| {
            with_node(topo, n, |s| {
                s.world_mut()
                    .resource::<vd_sim::stub::StubStats>()
                    .transients_dropped
            })
        })
        .sum()
}

/// The SOURCE shard's count of `TransientBatch` envelopes EMITTED (D-7c G-TIER ack-cost dimension) —
/// one per batch, regardless of item count. The K-batch gate asserts this == K (round-trip cost is
/// O(batches), not O(items)).
#[must_use]
pub fn source_transients_emitted(topo: &mut Topology) -> u64 {
    with_node(topo, SHARD, |s| {
        s.world_mut()
            .resource::<vd_sim::stub::StubStats>()
            .transients_emitted
    })
}

/// D-7c G-TIER: seed a BURST of `count` distinct Debris transients on the SOURCE shard ([`SHARD`]),
/// ALL carrying the SAME `batch` id (so `emit_transient_batch` aggregates them into ONE envelope +
/// ONE go-token — the burst-isolation claim). Entity seqs are `seq_base..seq_base+count` (give each
/// batch a DISTINCT `seq_base` so K batches do not collide on entity id). Returns the seeded entity
/// set (for the `held_at_dest` positive guard). Reuses [`seed_transient_crossing`] (DRY); rest poses.
pub fn seed_transient_burst(
    topo: &mut Topology,
    batch: TransferId,
    count: u32,
    seq_base: u64,
    anchor: Fence,
    dst_realm_fence: Fence,
) -> BTreeSet<EntityId> {
    let mut seeded = BTreeSet::new();
    for i in 0..count {
        let entity = EntityId::pack(
            EntityKind::Debris,
            SHARD.0 as u32,
            seq_base + u64::from(i),
            0,
        );
        seed_transient_crossing(
            topo,
            entity,
            batch,
            anchor,
            dst_realm_fence,
            vd_core::glam::DVec3::ZERO,
        );
        seeded.insert(entity);
    }
    seeded
}

/// How many of the `entities` the DEST shard holds AUTHORITATIVELY (D-7c G-TIER positive guard — the
/// drop-axis vacuity closer: proves all N burst items actually settled at the DEST, not silently
/// dropped). Reads `owned_transients` (the `is_held()` subset) at DEST.
#[must_use]
pub fn held_at_dest(reports: &[(NodeId, InspectReport)], entities: &BTreeSet<EntityId>) -> usize {
    reports
        .iter()
        .filter(|(n, _)| *n == DEST)
        .flat_map(|(_, r)| r.owned_transients.iter())
        .filter(|(e, _)| entities.contains(e))
        .count()
}

/// The DURABLE-relevant projection of one node's [`InspectReport`], filtered to the durable subject
/// (D-7c DURABLE-UNAFFECTED-BY-BURST). Projects ONLY fields a transient burst must NEVER perturb (the
/// subject's directory row + held authority/pose/pending/departing/ghost + its live saga); EXCLUDES
/// the legitimately-moving transient/wire fields (`owned_transients`, `held_transient_poses`,
/// `batch_goes`, `batch_go_writes`, `transient_loss`, `held_realms`, `trace_bytes`). `PartialEq` (NOT
/// `Eq` — `StampedPose` carries `f64`); a deterministic same-seed run yields bit-identical poses.
#[derive(Clone, Debug, PartialEq)]
pub struct DurableProjection {
    pub directory_subject: Option<OwnerRecord>,
    pub directory_session: Option<OwnerRecord>,
    pub held_subject: Vec<(EntityId, Fence)>,
    pub held_pose_subject: Vec<(EntityId, StampedPose)>,
    pub pending_subject: Vec<EntityId>,
    pub departing_subject: Vec<EntityId>,
    pub ghost_subject: bool,
    pub active_durable: Vec<ActiveTransfer>,
}

/// Project the durable subject's footprint from every node's report (D-7c). The differential gate
/// asserts this is byte-identical with vs without a concurrent transient burst — any diff is a burst
/// leak into the durable path = an HR1/HR2 violation. PRECONDITION: byte-identity holds only under the
/// ZERO `LinkPolicy` (`p2_cluster` installs no faults); a faulted variant would need invariant-equality
/// (same terminal directory record + applied-input multiset), not raw `assert_eq!` on this projection.
#[must_use]
pub fn durable_subset(
    reports: &[(NodeId, InspectReport)],
    subject: EntityId,
    session: SessionId,
) -> Vec<(NodeId, DurableProjection)> {
    reports
        .iter()
        .map(|(node, r)| {
            (
                *node,
                DurableProjection {
                    directory_subject: r
                        .directory
                        .iter()
                        .find_map(|(k, rec)| (*k == DirectoryKey::Entity(subject)).then_some(*rec)),
                    directory_session: r.directory.iter().find_map(|(k, rec)| {
                        (*k == DirectoryKey::Session(session)).then_some(*rec)
                    }),
                    held_subject: r
                        .held_entities
                        .iter()
                        .filter(|(e, _)| *e == subject)
                        .copied()
                        .collect(),
                    held_pose_subject: r
                        .held_poses
                        .iter()
                        .filter(|(e, _)| *e == subject)
                        .copied()
                        .collect(),
                    pending_subject: r
                        .pending_entities
                        .iter()
                        .filter(|e| **e == subject)
                        .copied()
                        .collect(),
                    departing_subject: r
                        .departing_entities
                        .iter()
                        .filter(|e| **e == subject)
                        .copied()
                        .collect(),
                    ghost_subject: r.ghost_dots.contains(&subject),
                    active_durable: r
                        .active_transfers
                        .iter()
                        .filter(|at| at.subject == DirectoryKey::Entity(subject))
                        .copied()
                        .collect(),
                },
            )
        })
        .collect()
}

/// The `Debug` state strings of every live saga on the orchestrator (the admin `views()`
/// surface) — the deterministic phase observable the cut-window choreography polls on.
#[must_use]
pub fn saga_states(topo: &mut Topology) -> Vec<String> {
    with_orchestrator(topo, |orch| {
        orch.world_mut()
            .resource::<SagaRuntimeRes>()
            .views()
            .into_iter()
            .map(|v| v.state)
            .collect()
    })
}

/// Live saga count on the orchestrator.
#[must_use]
pub fn live_sagas(topo: &mut Topology) -> usize {
    with_orchestrator(topo, |orch| {
        orch.world_mut().resource::<SagaRuntimeRes>().live()
    })
}

/// D-7d — the count of dead-SOURCE self-promote resolutions on the orchestrator (the SOURCE-kill cell's
/// anti-vacuity observable: `> 0` proves the resolution actually fired, not that the happy path ran).
#[must_use]
pub fn source_unreachable_resolutions(topo: &mut Topology) -> u64 {
    with_orchestrator(topo, |orch| {
        orch.world_mut()
            .resource::<SagaRuntimeRes>()
            .source_unreachable_resolutions()
    })
}

/// D-7d — the count of dead-DEST abandon resolutions on the orchestrator (the DEST-kill cell's
/// anti-vacuity observable).
#[must_use]
pub fn dest_unreachable_resolutions(topo: &mut Topology) -> u64 {
    with_orchestrator(topo, |orch| {
        orch.world_mut()
            .resource::<SagaRuntimeRes>()
            .dest_unreachable_resolutions()
    })
}

/// D-3 — total `NodeUnreachable` notices the orchestrator observed (the CSCALE-1 flap cell's anti-vacuity
/// observable: `> 0` proves the blip genuinely exercised the path the old code would have abandoned on).
#[must_use]
pub fn liveness_notices(topo: &mut Topology) -> u64 {
    with_orchestrator(topo, |orch| {
        orch.world_mut()
            .resource::<SagaRuntimeRes>()
            .liveness_notices()
    })
}

/// D-7d — the total handover-attributable loss for `kind`, summed across every node's `transient_loss`
/// (the budget gate's population). Used by the DEST-kill cell to assert a NON-ZERO, within-budget loss.
#[must_use]
pub fn total_handover_loss(reports: &[(NodeId, InspectReport)], kind: EntityKind) -> u64 {
    reports
        .iter()
        .flat_map(|(_, r)| r.transient_loss.iter())
        .filter(|(k, _)| *k == kind)
        .map(|(_, n)| *n)
        .sum()
}

/// The gateway's count of client inputs BUFFERED for the transfer dest (the `seq > marker` cut
/// buffer fills) — so the conservation gate can assert the buffer/DRAIN path was actually
/// exercised, not merely that the dest applied SOMETHING post-marker (which a direct-forward
/// after the route swap also satisfies). Read via a gateway downcast (vd-harness does not depend
/// on connection-plane, so `GatewayStats` cannot ride `InspectReport`).
#[must_use]
pub fn gateway_buffered_count(topo: &mut Topology) -> u64 {
    with_node(topo, GATEWAY, |gw| {
        gw.world_mut()
            .resource::<GatewayStats>()
            .inputs_buffered_for_dest
    })
}

/// Build one scripted client with a freshly minted (real, Ed25519) login ticket.
#[must_use]
pub fn p1_client(
    fabric: &FaultFabric,
    id: NodeId,
    account: AccountId,
    script: impl FnMut(&DeliveredWorldView) -> Option<InputCmd> + Send + 'static,
) -> ScriptedClient {
    let ticket = tickets::mint_login(&AUTH_SIGNING_KEY, account, EpochId(1), id.0);
    ScriptedClient::new(fabric.register(id), GATEWAY, ticket, script)
}

/// A script that walks straight forward forever.
pub fn walk_forward() -> impl FnMut(&DeliveredWorldView) -> Option<InputCmd> + Send + 'static {
    |_view: &DeliveredWorldView| {
        Some(InputCmd {
            movement: [1.0, 0.0, 0.0],
            look: [0.0, 0.0],
        })
    }
}

// ====================================================================================================
// P3 Slice 1 — the crash/fault recovery matrix harness (the kill-9 thesis headline). ONE data-driven
// driver over a cell table (DRY, HR2-additive: parameterized on `DurabilityClass`); the dead-node-aware
// oracle (vd-harness) makes the deferred kill cells HONEST (a corpse never false-passes AUTHORITY-UNIQUE).
// SCOPE: crash+resurrect (fabric at-least-once recovers) + permanent kill (handled cell aborts to a live
// source; D-37 cells park/orphan honestly). Drop/partition + the seed-driven breadth chaos are P3 Slice 1b.
// ====================================================================================================

/// The crash-matrix client (distinct from the in-process capstone CLIENT 100, but same id is fine —
/// only one client per scenario).
pub const FAULT_CLIENT: NodeId = NodeId(100);

/// The fault a scenario injects at a chosen saga phase.
#[derive(Clone, Copy, Debug)]
pub enum Fault {
    /// Temporary crash (`CrashWhen` phase) then resurrect `after` ticks later — the fabric's
    /// at-least-once redelivers the unacked message on resurrection (the common-case recovery).
    CrashResurrect { after: u64 },
    /// Permanent death of the victim — sends toward it bounce `NodeUnreachable`; recovery depends on
    /// the cell (a pre-freeze dest-kill aborts to the live source; the rest park/orphan honestly, D-37).
    Kill,
}

/// One crash-matrix cell: crash/kill `victim` once the saga reaches `at_phase`, at the `crash_when`
/// sub-tick phase. `class` feeds `SagaCtx.class` (HR2 — Transient/Guided plug in additively, no
/// match-on-class in the driver).
#[derive(Clone, Copy, Debug)]
pub struct Scenario {
    pub class: DurabilityClass,
    pub victim: NodeId,
    pub at_phase: &'static str,
    pub crash_when: CrashWhen,
    pub fault: Fault,
    /// D-37 Slice 3b: this cell expects the STANDING reaper-driven re-home (CELL 3 — a pre-freeze SOURCE
    /// kill leaves a dead-owner orphan no live saga heals). When set, the scenario uses the REAPING cluster
    /// ([`p2_cluster_reaping`]: short lease + non-zero reaper interval) and runs the FULL quiesce window
    /// (the aborted saga tombstones BEFORE the reaper arms the re-home, so an early `live_sagas == 0` break
    /// would exit before the parked re-home saga is armed). Every other cell leaves this `false`.
    pub standing_rehome: bool,
}

/// The asserted end state of a scenario. Data-carrying (not 3 bare arms) so the deferred D-37 cells
/// name WHERE the orphan/park sits — proven HONEST (not false-green) by the dead-node-aware oracle.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum EndState {
    /// The transfer COMMITTED + settled at `node` (the recovered happy path).
    SettledAt(NodeId),
    /// A clean abort returned the avatar to the live SOURCE, lock cleared, saga terminal. NOTE: NO
    /// permanent-kill cell reaches this (killing the source/gateway makes THEM dead → an orphan; the
    /// abort-to-a-LIVE-source path is a NON-kill abort — a spatial rejection or a transient-fault
    /// pre-freeze timeout that heals). Exercised in P3 Slice 1b (when the Drop/Partition faults land).
    AbortedToSource,
    /// D-37 (RED, owed D-3+D-6): the saga PARKED (re-driving toward a dead node), directory names the
    /// dead `authority_at`; the dead-aware oracle surfaces the orphan rather than false-passing a corpse.
    ParkedHalfOpen { authority_at: NodeId },
    /// D-37 (RED): the saga TOMBSTONED to a dead-owner orphan (directory names the dead `authority_at`,
    /// lock cleared, no live saga) — the dead-aware oracle surfaces it (the orphan is a HeldNowhere: a
    /// dead owner holds nothing, so the fence value is moot here).
    DeadOwnerOrphan { authority_at: NodeId },
    /// D-37 entity-recovery slices (0/2, INTERMEDIATE): the ENTITY self-promoted / re-homed to the LIVE
    /// owner `entity_at` (it is no longer an orphan), but the dead SOURCE shard's REALM is STILL orphaned
    /// — the STANDING realm re-home is owed at Slice 3/4. GREEN entity, honest-RED realm; Slice 4 flips
    /// the cell to [`EndState::SettledAt`].
    EntityRecoveredRealmOrphaned { entity_at: NodeId },
    /// D-7d TRANSIENT analogue of [`EndState::SettledAt`]: the batch committed + settled at `node`'s
    /// HELD set (a transient has NO directory row), proven by the dead-aware held-set + a fired
    /// SOURCE-unreachable self-promote resolution; ZERO loss. (Dead source mid-handoff → the go-token
    /// authorizes the dest self-promote.)
    BatchCommittedAt { node: NodeId },
    /// D-7d accounted-loss outcome: the batch was DROPPED (the dead-DEST abandon) — held at NO live
    /// shard, lost WITHIN the `kind`'s `LossBudget` AND non-zero (anti-vacuity), via a fired
    /// DEST-unreachable resolution. (Dead dest mid-handoff → drop-within-budget, the transient answer.)
    BatchDroppedWithinBudget { kind: EntityKind },
}

fn fault_report(reports: &[(NodeId, InspectReport)], id: NodeId) -> &InspectReport {
    &reports
        .iter()
        .find(|(n, _)| *n == id)
        .expect("node present")
        .1
}

/// Step at most `max` ticks until `cond` holds; panic if never (bounded, deterministic).
fn fault_step_until(topo: &mut Topology, max: u64, mut cond: impl FnMut(&mut Topology) -> bool) {
    for _ in 0..max {
        topo.step();
        if cond(topo) {
            return;
        }
    }
    panic!("crash-matrix condition not reached within {max} ticks");
}

/// The orchestrator's directory record for the subject entity (the authority-of-record).
#[must_use]
fn fault_entity_record(reports: &[(NodeId, InspectReport)], entity: EntityId) -> OwnerRecord {
    fault_report(reports, ORCH)
        .directory
        .iter()
        .find_map(|(k, r)| (*k == DirectoryKey::Entity(entity)).then_some(*r))
        .expect("the subject entity is recorded in the directory")
}

/// Drive ONE crash-matrix scenario over the real fabric: warm up + trigger the transfer, drive to
/// `sc.at_phase` (asserting the saga is genuinely there — anti-vacuity), inject the fault, then
/// quiesce (settle, or cap for a parked cell). Returns the topology, the transferred entity, and the
/// dead-node set for the dead-aware oracle.
#[must_use]
pub fn run_fault_scenario(seed: u64, sc: Scenario) -> (Topology, EntityId, BTreeSet<NodeId>) {
    let fabric = FaultFabric::new(seed, 2);
    // D-37 Slice 3b: the standing-re-home cell needs the REAPING cluster (short lease + reaper) so the
    // orphan is detected + re-homed post-kill; every other cell uses the inert-reaper default.
    let mut topo = if sc.standing_rehome {
        p2_cluster_reaping(&fabric, 8)
    } else {
        p2_cluster(&fabric, 8)
    };
    topo.add_node(Box::new(p1_client(
        &fabric,
        FAULT_CLIENT,
        AccountId(1000),
        walk_forward(),
    )));
    // WARMUP: the player logs in + walks; the SOURCE grants its avatar and the DEST wins its realm.
    fault_step_until(&mut topo, 80, |t| {
        let r = t.inspect_all();
        !fault_report(&r, SHARD).held_entities.is_empty()
            && fault_report(&r, DEST)
                .held_realms
                .iter()
                .any(|(realm, _)| *realm == RealmId::System(8))
    });
    let (session, entity, fence) = read_subject(&mut topo);
    trigger_transfer(
        &mut topo,
        SagaCtx {
            transfer: TransferId(1),
            session,
            subject: DirectoryKey::Entity(entity),
            expected_fence: fence,
            source: SHARD,
            dest: DEST,
            class: sc.class,
            needs_provision: false,
            from_realm: RealmId::System(7),
            to_realm: RealmId::System(8),
            to_parent: None,
        },
    );
    // Drive to the target saga phase (the client auto-stamps the CUT_MARKER on RequestCut; the saga
    // progresses on its acks — no marker pause/resume needed for the crash matrix).
    fault_step_until(&mut topo, 60, |t| {
        saga_states(t).iter().any(|s| s.starts_with(sc.at_phase))
    });
    // ANTI-VACUITY: the saga is GENUINELY in the target phase on the fire tick (not advanced past it).
    assert!(
        saga_states(&mut topo)
            .iter()
            .any(|s| s.starts_with(sc.at_phase)),
        "the saga is in phase {} when the fault fires (anti-vacuity)",
        sc.at_phase,
    );
    match sc.fault {
        Fault::CrashResurrect { after } => {
            let fire = topo.tick().0 + 1;
            topo.schedule_crash(sc.victim, TickId(fire), sc.crash_when);
            topo.schedule_resurrect(sc.victim, TickId(fire + after));
        }
        Fault::Kill => fabric.kill(sc.victim),
    }
    // QUIESCE: settle to terminal, or cap (a D-37 parked cell never quiesces — the assert handles it).
    // D-37 Slice 3b: a standing-re-home cell must NOT break at the FIRST `live_sagas == 0` — the aborted
    // saga tombstones (live 0) BEFORE the reaper detects the lapsed orphan + arms the re-home (live 1 again,
    // parked) a few sweeps later. Run the FULL window so the parked re-home saga is armed by the end.
    for _ in 0..120 {
        topo.step();
        if !sc.standing_rehome && live_sagas(&mut topo) == 0 {
            break;
        }
    }
    let dead = topo.dead_nodes();
    (topo, entity, dead)
}

/// Assert the scenario reached its expected [`EndState`], using the dead-node-aware oracle so a killed
/// participant's corpse can neither false-pass uniqueness nor false-RED a legitimate park.
pub fn assert_end_state(
    topo: &mut Topology,
    entity: EntityId,
    dead: &BTreeSet<NodeId>,
    expected: EndState,
) {
    // The orphan that the dead-aware oracle MUST surface for a directory record naming a dead node:
    // the entity half returns HeldNowhere (the recorded owner holds nothing live) before the realm half.
    let orphan = |authority_at: NodeId| AuthorityViolation::HeldNowhere {
        entity,
        recorded: format!("{:?}", AuthorityRef::Shard(authority_at)),
    };
    match expected {
        EndState::SettledAt(node) => {
            // Post-recovery the LIVE view (a resurrect cell has no dead nodes, so inspect_live ==
            // inspect_all here) must be consistent + settled, with the avatar held once at `node`.
            let reports = topo.inspect_live();
            verify_authority_unique_excluding(&reports, dead)
                .expect("AUTHORITY-UNIQUE holds after recovery");
            verify_authority_settled(&reports).expect("settled after recovery");
            assert_eq!(
                live_sagas(topo),
                0,
                "the saga reached terminal (recovered to Done)"
            );
            assert!(
                fault_report(&reports, node)
                    .held_entities
                    .iter()
                    .any(|(e, _)| *e == entity),
                "the entity settled at {node} after recovery",
            );
        }
        EndState::AbortedToSource => {
            let reports = topo.inspect_all();
            let rec = fault_entity_record(&reports, entity);
            assert_eq!(
                rec.authority,
                AuthorityRef::Shard(SHARD),
                "aborted back to the source"
            );
            assert_eq!(rec.in_transfer, None, "the directory lock cleared on abort");
            assert_eq!(live_sagas(topo), 0, "the abort reached terminal");
            assert!(
                fault_report(&reports, SHARD)
                    .held_entities
                    .iter()
                    .any(|(e, _)| *e == entity),
                "the source still owns the avatar (never demoted)",
            );
            // FENCE-9 (the abort-path fence-neutrality fix): the surviving source's HELD fence equals the
            // directory's RECORDED fence — `abort_clear` did NOT bump (an abort is no ownership change), so
            // there is no owner-vs-directory split that would strand the source / wedge a logout LeaseRevoke.
            verify_authority_unique_excluding(&reports, dead)
                .expect("AUTHORITY-UNIQUE holds after the abort (no fence divergence — FENCE-9)");
            // SETTLED (parity with the SettledAt arm): the abort left NO lingering pending/departing/ghost
            // set — the AbortTransfer compensator tore the dest ghost down. Catches a leaked teardown that
            // the uniqueness check (which excludes Ghosts) would miss — load-bearing for a future
            // signal-grant / compound abort that is likelier to leak.
            verify_authority_settled(&reports)
                .expect("settled after the abort (no leaked dest ghost)");
        }
        EndState::ParkedHalfOpen { authority_at } => {
            let reports = topo.inspect_all();
            assert!(
                live_sagas(topo) >= 1,
                "the saga is PARKED (no recovery without D-37)"
            );
            let rec = fault_entity_record(&reports, entity);
            assert_eq!(
                rec.authority,
                AuthorityRef::Shard(authority_at),
                "the directory names the dead node {authority_at}",
            );
            // The dead-aware oracle surfaces the EXACT orphan (HeldNowhere at the dead node) — pinned
            // to the variant, not a loose is_err (HR5): a real split-brain would be WrongHolderCount.
            assert_eq!(
                verify_authority_unique_excluding(&reports, dead),
                Err(orphan(authority_at)),
                "the dead-aware oracle surfaces the orphan (NOT a false-passing corpse)",
            );
        }
        EndState::DeadOwnerOrphan { authority_at } => {
            let reports = topo.inspect_all();
            assert_eq!(
                live_sagas(topo),
                0,
                "the saga tombstoned (gateway-acked abort)"
            );
            let rec = fault_entity_record(&reports, entity);
            assert_eq!(
                rec.authority,
                AuthorityRef::Shard(authority_at),
                "the directory names the dead owner {authority_at}",
            );
            assert_eq!(rec.in_transfer, None, "the lock cleared (abort_clear)");
            assert_eq!(
                verify_authority_unique_excluding(&reports, dead),
                Err(orphan(authority_at)),
                "the dead-aware oracle surfaces the exact dead-owner orphan",
            );
        }
        EndState::EntityRecoveredRealmOrphaned { entity_at } => {
            // D-37 entity-recovery slices (0/2): the ENTITY self-promoted/re-homed to the LIVE `entity_at`
            // (no longer an orphan), but the dead SOURCE shard's REALM is still orphaned — the STANDING
            // realm re-home is owed at Slice 3/4. The dead-aware oracle honestly surfaces ONLY the realm
            // orphan now (the entity check PASSES), proving the entity recovery without false-passing the
            // realm gap. (vd-tests is the integration tier, not coverage-gated, so `matches!` is fine.)
            let reports = topo.inspect_all();
            let rec = fault_entity_record(&reports, entity);
            assert_eq!(
                rec.authority,
                AuthorityRef::Shard(entity_at),
                "the entity recovered to the live owner {entity_at}",
            );
            assert!(
                matches!(
                    verify_authority_unique_excluding(&reports, dead),
                    Err(AuthorityViolation::RealmHeldNowhere { .. })
                ),
                "the entity is no longer the orphan; the ONLY remaining orphan is the dead SOURCE shard's \
                 REALM (the standing realm re-home is owed at Slice 3/4): {:?}",
                verify_authority_unique_excluding(&reports, dead),
            );
        }
        // The TRANSIENT end states are held-set / loss-budget shaped (no directory row) — asserted by
        // [`assert_transient_end_state`], never this directory-shaped durable asserter.
        EndState::BatchCommittedAt { .. } | EndState::BatchDroppedWithinBudget { .. } => {
            panic!(
                "transient end states are asserted by assert_transient_end_state, not assert_end_state"
            )
        }
    }
}

/// Drive ONE crash-matrix scenario for a TRANSIENT batch (D-7d) — the held-set/loss-budget sibling of
/// [`run_fault_scenario`] (which is directory-shaped to the bone; a transient writes ZERO directory rows,
/// so forcing both through one driver would need a forbidden match-on-class). REUSES the shared spine
/// (cluster, realm-lease warmup, `seed_transient_crossing`, a Transient `trigger_transfer`, the
/// `Fault`/`CrashWhen` enums, `schedule_crash`/`kill`/`dead_nodes`) verbatim; only the transient-shaped
/// 40% differs (warmup waits for BOTH realm leases; drive-to-phase matches the `BatchHandoff` string;
/// per-tick conservation is dead-aware; quiesce on `live_sagas == 0`). Returns the topo + the debris
/// entity + the dead-node set for [`assert_transient_end_state`].
#[must_use]
pub fn run_transient_fault_scenario(
    seed: u64,
    sc: Scenario,
) -> (Topology, EntityId, BTreeSet<NodeId>) {
    let fabric = FaultFabric::new(seed, 2);
    let mut topo = p2_cluster(&fabric, 8);
    // WARMUP: BOTH shards win their realm leases (the source to cross from, the dest to adopt into).
    fault_step_until(&mut topo, 80, |t| {
        let r = t.inspect_all();
        fault_report(&r, SHARD)
            .held_realms
            .iter()
            .any(|(realm, _)| *realm == RealmId::System(7))
            && fault_report(&r, DEST)
                .held_realms
                .iter()
                .any(|(realm, _)| *realm == RealmId::System(8))
    });
    let src_fence = realm_fence(&mut topo, RealmId::System(7));
    let dst_fence = realm_fence(&mut topo, RealmId::System(8));
    // Seed a 1-item Debris crossing SHARD→DEST and trigger the matching Transient batch saga.
    let debris = EntityId::pack(EntityKind::Debris, SHARD.0 as u32, 1, 0);
    let batch = TransferId(1);
    seed_transient_crossing(
        &mut topo,
        debris,
        batch,
        src_fence,
        dst_fence,
        vd_core::glam::DVec3::ZERO,
    );
    trigger_transfer(
        &mut topo,
        SagaCtx {
            transfer: batch,
            session: SessionId(0),
            subject: DirectoryKey::Realm(RealmId::System(8)),
            expected_fence: dst_fence,
            source: SHARD,
            dest: DEST,
            class: sc.class,
            needs_provision: false,
            from_realm: RealmId::System(7),
            to_realm: RealmId::System(8),
            to_parent: None,
        },
    );
    // Drive to the target `BatchHandoff` phase (the saga progresses on the choreography acks).
    fault_step_until(&mut topo, 60, |t| {
        saga_states(t).iter().any(|s| s.starts_with(sc.at_phase))
    });
    assert!(
        saga_states(&mut topo)
            .iter()
            .any(|s| s.starts_with(sc.at_phase)),
        "the saga is in phase {} when the fault fires (anti-vacuity)",
        sc.at_phase,
    );
    match sc.fault {
        Fault::CrashResurrect { after } => {
            let fire = topo.tick().0 + 1;
            topo.schedule_crash(sc.victim, TickId(fire), sc.crash_when);
            topo.schedule_resurrect(sc.victim, TickId(fire + after));
        }
        Fault::Kill => fabric.kill(sc.victim),
    }
    // PHASE 1 — drive to SAGA quiescence: the dead-resolution fires ~2 redrive windows after the kill
    // (the first Timeout re-emit bounces `NodeUnreachable` → marks the node dead; the next due injects
    // the resolution), tombstoning the saga. Dead-aware per-tick TRANSIENT-CONSERVATION holds EVERY tick
    // (a killed participant's corpse is excluded, so a stale Held claim is never a phantom holder).
    let mut settled = false;
    for i in 0..160 {
        topo.step();
        let dead = topo.dead_nodes();
        let reports = topo.inspect_all();
        verify_transient_conservation_tick_excluding(&reports, &dead, TickId(i))
            .expect("no transient COUNTED-held by two LIVE shards at any tick under the crash");
        if live_sagas(&mut topo) == 0 {
            settled = true;
            break;
        }
    }
    assert!(
        settled,
        "the transient crash scenario did not reach saga quiescence"
    );
    // PHASE 2 — let the RESOLUTION'S EGRESS settle: the saga tombstones the SAME tick it emits the
    // self-promote (→ the dest flips `Arriving→Held`) / abandon (→ the source drops + buckets the loss),
    // so the SURVIVOR processes it only AFTER quiescence. Conservation still holds each settle tick.
    for i in 0..24 {
        topo.step();
        let dead = topo.dead_nodes();
        let reports = topo.inspect_all();
        verify_transient_conservation_tick_excluding(&reports, &dead, TickId(i)).expect(
            "no transient COUNTED-held by two LIVE shards while the resolution egress settles",
        );
    }
    let dead = topo.dead_nodes();
    (topo, debris, dead)
}

/// Assert a TRANSIENT crash scenario reached its expected [`EndState`] (D-7d), held-set / loss-budget
/// shaped (a transient has no directory row). The dead-aware `TRANSIENT-AUTHORITY-HELD` holds in EVERY
/// outcome (no double-hold; corpses excluded). `BatchCommittedAt` = the debris held SINGLY at the live
/// survivor + a fired source-unreachable resolution + zero loss; `BatchDroppedWithinBudget` = held at NO
/// live shard + a within-budget NON-ZERO loss + a fired dest-unreachable resolution.
pub fn assert_transient_end_state(
    topo: &mut Topology,
    debris: EntityId,
    dead: &BTreeSet<NodeId>,
    expected: EndState,
) {
    let reports = topo.inspect_all();
    verify_transient_authority_held_excluding(&reports, dead)
        .expect("TRANSIENT-AUTHORITY-HELD (dead-aware) holds after the crash settles");
    let live_holders: Vec<NodeId> = reports
        .iter()
        .filter(|(n, _)| !dead.contains(n))
        .filter(|(_, r)| r.owned_transients.iter().any(|(e, _)| *e == debris))
        .map(|(n, _)| *n)
        .collect();
    match expected {
        EndState::BatchCommittedAt { node } => {
            // OUTCOME-only (reusable by BOTH the SOURCE-kill cell AND the crash-resurrect control, which
            // reaches the same outcome WITHOUT a resolution): the debris settled SINGLY at the live
            // survivor, zero loss. Each cell adds its mechanism check (resolution fired vs NOT fired).
            assert_eq!(
                live_holders,
                vec![node],
                "the debris settled at the live survivor, held by exactly one shard"
            );
            assert_eq!(
                transient_dropped_total(topo),
                0,
                "BatchCommittedAt is ZERO loss (the item was kept, not dropped)"
            );
        }
        EndState::BatchDroppedWithinBudget { kind } => {
            assert!(
                live_holders.is_empty(),
                "the abandoned debris is held at NO live shard: {live_holders:?}"
            );
            verify_transient_loss_budget(&reports, kind)
                .expect("the loss is within the kind's budget");
            assert!(
                total_handover_loss(&reports, kind) > 0,
                "anti-vacuity: a loss was actually counted (not a silent vanish)"
            );
            assert!(
                dest_unreachable_resolutions(topo) > 0,
                "the dead-dest abandon resolution actually fired (anti-vacuity)"
            );
        }
        _ => panic!("assert_transient_end_state handles only the transient end states"),
    }
}

/// D-3 CSCALE-1 e2e: a recoverable BLIP (flap) toward a HEALTHY dest during a transient `BatchHandoff`
/// must NOT abandon the batch — the cure. The orchestrator is tuned to `n_consecutive_unreachable = 3`
/// (the prod margin), so the flap's `NodeUnreachable{DEST}` notices (the orch→dest promote bouncing)
/// never reach the confirmation threshold; the dest is never confirmed dead, the flap heals, the promote
/// redelivers, and the batch lands at DEST with zero loss. RED before Slice 3 (one `NodeUnreachable` →
/// `dead_participants` → the cheap-redrive abandon → irreversible loss of a HEALTHY batch). Returns the
/// settled topology + the debris entity (no node is killed, so the dead set is empty at the assert).
#[must_use]
pub fn run_transient_dest_flap(seed: u64) -> (Topology, EntityId) {
    let fabric = FaultFabric::new(seed, 2);
    let mut topo = p2_cluster(&fabric, 8);
    // The CSCALE-1 margin: a short blip (< 3 consecutive notices) never confirms the dest dead.
    with_orchestrator(&mut topo, |o| {
        o.world_mut()
            .resource_mut::<SagaRuntimeRes>()
            .set_liveness_tuning(LivenessTuning {
                n_consecutive_unreachable: 3,
                unreachable_window_ticks: 64,
                retry_delay_ticks_hint: 2,
            });
    });
    // WARMUP: both shards win their realm leases (the source to cross from, the dest to adopt into).
    fault_step_until(&mut topo, 80, |t| {
        let r = t.inspect_all();
        fault_report(&r, SHARD)
            .held_realms
            .iter()
            .any(|(realm, _)| *realm == RealmId::System(7))
            && fault_report(&r, DEST)
                .held_realms
                .iter()
                .any(|(realm, _)| *realm == RealmId::System(8))
    });
    let src_fence = realm_fence(&mut topo, RealmId::System(7));
    let dst_fence = realm_fence(&mut topo, RealmId::System(8));
    let debris = EntityId::pack(EntityKind::Debris, SHARD.0 as u32, 1, 0);
    let batch = TransferId(1);
    seed_transient_crossing(
        &mut topo,
        debris,
        batch,
        src_fence,
        dst_fence,
        vd_core::glam::DVec3::ZERO,
    );
    trigger_transfer(
        &mut topo,
        SagaCtx {
            transfer: batch,
            session: SessionId(0),
            subject: DirectoryKey::Realm(RealmId::System(8)),
            expected_fence: dst_fence,
            source: SHARD,
            dest: DEST,
            class: DurabilityClass::Transient,
            needs_provision: false,
            from_realm: RealmId::System(7),
            to_realm: RealmId::System(8),
            to_parent: None,
        },
    );
    // Drive to AwaitRelease (BEFORE AwaitPromote), so the orch→DEST promote — emitted on the AwaitPromote
    // transition — bounces during the flap window opened next.
    fault_step_until(&mut topo, 60, |t| {
        saga_states(t)
            .iter()
            .any(|s| s.starts_with("BatchHandoff { phase: AwaitRelease"))
    });
    // FLAP orch→DEST for a short window: the promote bounces (`NodeUnreachable{DEST}` → record_unreachable),
    // but < 3 consecutive + clear-on-ack → never confirmed; the flap heals → promote redelivers → done.
    fabric.flap(ORCH, DEST, TickId(topo.tick().0 + 10));
    for _ in 0..200 {
        topo.step();
        if live_sagas(&mut topo) == 0 {
            break;
        }
    }
    // Settle the handoff's terminal egress (the dest's promote-ack → the source retires).
    for _ in 0..24 {
        topo.step();
    }
    (topo, debris)
}

/// REBUILD the orchestrator on its existing identity (D-6 kill-9 recovery): drop the running World (its
/// in-memory saga set / directory / clock — gone with the process RAM) and stand up a FRESH one that
/// RE-HYDRATES from `store`. Mirrors a real restart: [`FaultFabric::reregister`] hands the rebuilt process
/// a fresh transport on the prior peer links (the at-least-once unacked ledger redelivers what the dead
/// process never acked) and [`Topology::replace_node`] swaps the new World in. Built IDENTICALLY to the
/// original (same [`orch_config`]) so recovery is a clean re-hydrate. Pass the RETAINED handle from
/// [`p2_cluster_durable_orch`] to recover; pass a fresh `MemStore::new()` to model a NON-durable restart
/// (the anti-theater control — nothing is recovered). The caller has typically `kill`ed the node first.
pub fn rebuild_orchestrator(topo: &mut Topology, fabric: &FaultFabric, store: MemStore) {
    let mut orch = build_app(
        NodeConfig {
            node_id: ORCH,
            kind: NodeKind::Orchestrator,
        },
        fabric.reregister(ORCH),
    );
    let (world, schedule) = orch.parts_mut();
    register_orchestrator_with_store(
        world,
        schedule,
        // D-6 rebuild: the IDENTICAL roster the original cluster used ({SHARD, DEST} stub shards) so the
        // rebuilt orchestrator recovers to the same re-home config (a clean recover).
        &orch_config(vec![GATEWAY, SHARD, DEST], stub_roster([SHARD, DEST])),
        Box::new(store),
    );
    topo.replace_node(Box::new(orch));
}

/// The outcome of an orchestrator kill-9 + rebuild ([`run_orch_kill_transient`] / [`run_orch_kill_durable`]).
pub struct OrchKillOutcome {
    /// The cluster, settled (recovered) or frozen right after the rebuild (anti-theater).
    pub topo: Topology,
    /// The transferred subject — a debris item (transient) or a player avatar (durable).
    pub subject: EntityId,
    /// Dead nodes after the rebuild — EMPTY (the orchestrator is alive again), for the dead-aware oracle.
    pub dead: BTreeSet<NodeId>,
    /// `saga_states` captured the INSTANT the rebuilt orchestrator boots, before it re-drives anything —
    /// THE recovery evidence: a `BatchHandoff` saga (re-hydrated from the retained WAL) when durable,
    /// EMPTY (nothing survived the dropped World) for the fresh-store anti-theater control.
    pub recovered_states: Vec<String>,
}

/// D-6 e2e: kill-9 the orchestrator MID-`BatchHandoff` of a live transient batch, then rebuild it. With
/// `durable`, the rebuilt orchestrator re-hydrates the retained durable WAL, re-drives the in-flight
/// handoff to `Done`, and the debris settles at [`DEST`] with zero loss — an orchestrator crash mid-handoff
/// never wedges nor loses the batch. With `!durable` (rebuilt against a FRESH empty store) it recovers
/// NOTHING — the anti-theater control proving the recovery rides the persisted WAL, not in-process World
/// survival across the kill. Conservation is asserted EVERY tick of the re-drive (no transient is ever
/// COUNTED-held by two live shards across the orchestrator's death + recovery).
#[must_use]
pub fn run_orch_kill_transient(seed: u64, durable: bool) -> OrchKillOutcome {
    let fabric = FaultFabric::new(seed, 2);
    let (mut topo, store) = p2_cluster_durable_orch(&fabric, 8);
    // WARMUP: BOTH shards win their realm leases (the source to cross from, the dest to adopt into).
    fault_step_until(&mut topo, 80, |t| {
        let r = t.inspect_all();
        fault_report(&r, SHARD)
            .held_realms
            .iter()
            .any(|(realm, _)| *realm == RealmId::System(7))
            && fault_report(&r, DEST)
                .held_realms
                .iter()
                .any(|(realm, _)| *realm == RealmId::System(8))
    });
    let src_fence = realm_fence(&mut topo, RealmId::System(7));
    let dst_fence = realm_fence(&mut topo, RealmId::System(8));
    let debris = EntityId::pack(EntityKind::Debris, SHARD.0 as u32, 1, 0);
    let batch = TransferId(1);
    seed_transient_crossing(
        &mut topo,
        debris,
        batch,
        src_fence,
        dst_fence,
        vd_core::glam::DVec3::ZERO,
    );
    trigger_transfer(
        &mut topo,
        SagaCtx {
            transfer: batch,
            session: SessionId(0),
            subject: DirectoryKey::Realm(RealmId::System(8)),
            expected_fence: dst_fence,
            source: SHARD,
            dest: DEST,
            class: DurabilityClass::Transient,
            needs_provision: false,
            from_realm: RealmId::System(7),
            to_realm: RealmId::System(8),
            to_parent: None,
        },
    );
    // Drive to the `BatchHandoff` tail — the saga is quiescent-but-persisted (the kill target).
    fault_step_until(&mut topo, 60, |t| {
        saga_states(t).iter().any(|s| s.starts_with("BatchHandoff"))
    });
    assert!(
        saga_states(&mut topo)
            .iter()
            .any(|s| s.starts_with("BatchHandoff")),
        "anti-vacuity: the saga is in BatchHandoff when the orchestrator is killed"
    );
    // The orchestrator PERSISTED durable state before the kill — else the recovery is vacuous. (The
    // BatchHandoff assert above proves the saga is the in-flight thing; the cell's `recovered_states`
    // assert proves the saga specifically rehydrated. This is the "there IS a committed WAL" floor.)
    assert!(
        !store.is_empty(),
        "the orchestrator committed durable state (incl. the in-flight saga) before the kill"
    );
    // KILL-9: `crash` (not `kill`) is the restart-able death — the orchestrator's in-process inbound is
    // cleared (RAM lost), but the at-least-once ledger HOLDS the in-flight shard acks (e.g. a dest
    // `BatchAdopted` sent while it is down) for redelivery once it is back; `kill` is permanent death
    // (bounces + DROPS those acks, which the producer-less `AwaitAdopt` phase could never re-solicit).
    // The World itself dies at `replace_node` below — `crash` + rehydrated `replace_node` = kill-9+restart.
    fabric.crash(ORCH);
    // OUTAGE: a few ticks down — the orchestrator is skipped; the dest adopts off the source's envelope
    // and acks `BatchAdopted`, which the fabric HOLDS (blocked on the crashed orchestrator, rescheduled).
    // Conservation holds even here (the dead orchestrator is excluded; the dest's `Arriving` is uncounted).
    for _ in 0..4 {
        topo.step();
        let tick = topo.tick();
        let dead = topo.dead_nodes();
        let reports = topo.inspect_all();
        verify_transient_conservation_tick_excluding(&reports, &dead, tick)
            .expect("no transient COUNTED-held by two LIVE shards while the orchestrator is down");
    }
    // REBUILD on the SAME identity: durable → re-hydrate the retained store; anti-theater → a FRESH
    // (empty) store, which recovers NOTHING (the proof the retained WAL is load-bearing).
    let recover_store = if durable {
        store.clone()
    } else {
        MemStore::new()
    };
    rebuild_orchestrator(&mut topo, &fabric, recover_store);
    // THE recovery evidence — captured the instant the rebuilt World boots, before it re-drives.
    let recovered_states = saga_states(&mut topo);
    if !durable {
        // The anti-theater control stops here: nothing re-hydrated, so there is nothing to re-drive.
        let dead = topo.dead_nodes();
        return OrchKillOutcome {
            topo,
            subject: debris,
            dead,
            recovered_states,
        };
    }
    // QUIESCE: the re-hydrated saga re-arms (since=0) → the deadline producer re-emits the `BatchHandoff`
    // egress, the redelivered acks land, the handoff completes. Conservation holds EVERY tick.
    let mut settled = false;
    for _ in 0..200 {
        topo.step();
        let tick = topo.tick();
        let dead = topo.dead_nodes();
        let reports = topo.inspect_all();
        verify_transient_conservation_tick_excluding(&reports, &dead, tick).expect(
            "no transient COUNTED-held by two LIVE shards at any tick across the orchestrator kill-9",
        );
        if live_sagas(&mut topo) == 0 {
            settled = true;
            break;
        }
    }
    assert!(
        settled,
        "the rebuilt orchestrator re-drove the in-flight transient batch to quiescence"
    );
    // Let the recovered handoff's resolution egress settle (mirror `run_transient_fault_scenario`).
    for _ in 0..24 {
        topo.step();
        let tick = topo.tick();
        let dead = topo.dead_nodes();
        let reports = topo.inspect_all();
        verify_transient_conservation_tick_excluding(&reports, &dead, tick)
            .expect("conservation holds while the recovered handoff's egress settles");
    }
    let dead = topo.dead_nodes();
    OrchKillOutcome {
        topo,
        subject: debris,
        dead,
        recovered_states,
    }
}

/// D-6 e2e: kill-9 the orchestrator mid-saga of a live DURABLE player transfer, then rebuild it — the
/// P3 headline for the class that carries PLAYERS (zero loss, AUTHORITY-UNIQUE). The durable twin of
/// [`run_orch_kill_transient`]: a separate driver because the durable path is directory-shaped (a client
/// logs in + walks, the avatar is a directory `Entity` record, the saga runs the full
/// `Preparing→…→Demoting→Promoting` choreography) where the transient is held-set-shaped — forcing both
/// through one driver would need a forbidden match-on-class. The SHARED spine is reused verbatim:
/// `p2_cluster_durable_orch` (retained Store), `fault_step_until`, `fabric.crash` + `rebuild_orchestrator`,
/// the `recovered_states` capture, the quiesce loop. Drive to `at_phase`, `crash` the orchestrator, rebuild
/// from the retained store (or a FRESH one for the anti-theater control), and let the rehydrated saga
/// re-drive. POST-commit phases (e.g. `Demoting`) re-drive forward to `Done`; the cell asserts via
/// [`assert_end_state`]. With `!durable_store` the rebuilt orchestrator recovers NOTHING (the control).
#[must_use]
pub fn run_orch_kill_durable(seed: u64, at_phase: &str, durable_store: bool) -> OrchKillOutcome {
    let fabric = FaultFabric::new(seed, 2);
    let (mut topo, store) = p2_cluster_durable_orch(&fabric, 8);
    topo.add_node(Box::new(p1_client(
        &fabric,
        FAULT_CLIENT,
        AccountId(1000),
        walk_forward(),
    )));
    // WARMUP: the player logs in + walks; the SOURCE grants its avatar and the DEST wins its realm.
    fault_step_until(&mut topo, 80, |t| {
        let r = t.inspect_all();
        !fault_report(&r, SHARD).held_entities.is_empty()
            && fault_report(&r, DEST)
                .held_realms
                .iter()
                .any(|(realm, _)| *realm == RealmId::System(8))
    });
    let (session, entity, fence) = read_subject(&mut topo);
    trigger_transfer(
        &mut topo,
        SagaCtx {
            transfer: TransferId(1),
            session,
            subject: DirectoryKey::Entity(entity),
            expected_fence: fence,
            source: SHARD,
            dest: DEST,
            class: DurabilityClass::Durable,
            needs_provision: false,
            from_realm: RealmId::System(7),
            to_realm: RealmId::System(8),
            to_parent: None,
        },
    );
    // Drive to the target saga phase (the client auto-stamps the CUT_MARKER on RequestCut).
    fault_step_until(&mut topo, 60, |t| {
        saga_states(t).iter().any(|s| s.starts_with(at_phase))
    });
    assert!(
        saga_states(&mut topo)
            .iter()
            .any(|s| s.starts_with(at_phase)),
        "anti-vacuity: the saga is in the target phase when the orchestrator is killed"
    );
    assert!(
        !store.is_empty(),
        "the orchestrator committed durable state (incl. the in-flight saga) before the kill"
    );
    // KILL-9: `crash` (restart-able) clears the orchestrator's in-process inbound (RAM lost) while the
    // at-least-once ledger HOLDS the in-flight shard acks for redelivery; the World dies at `replace_node`.
    fabric.crash(ORCH);
    // OUTAGE: a few ticks down. The client keeps emitting to the GATEWAY (unaffected by the orchestrator
    // crash); the source/dest acks are held for redelivery. (No per-tick AUTHORITY-UNIQUE here — that
    // oracle reads the directory on the crashed orchestrator; the end-state assert covers recovery.)
    for _ in 0..4 {
        topo.step();
    }
    let recover_store = if durable_store {
        store.clone()
    } else {
        MemStore::new()
    };
    rebuild_orchestrator(&mut topo, &fabric, recover_store);
    let recovered_states = saga_states(&mut topo);
    if !durable_store {
        let dead = topo.dead_nodes();
        return OrchKillOutcome {
            topo,
            subject: entity,
            dead,
            recovered_states,
        };
    }
    // QUIESCE: the re-hydrated saga re-arms (since=0) → the deadline producer re-drives the choreography
    // (POST-commit phases re-emit Demote/Promote idempotently → forward to Done at DEST; a PRE-commit phase
    // fires the abort deadline → ThawSource → terminal Aborted at SOURCE). Either way the saga tombstones.
    for _ in 0..200 {
        topo.step();
        if live_sagas(&mut topo) == 0 {
            break;
        }
    }
    // SETTLE the terminal egress: the saga tombstones the SAME tick it emits its last action (the final
    // Promote/ReleaseComplete on the commit path, or the ThawSource + AbortTransfer teardown on the abort
    // path), so the SOURCE/DEST applies it only AFTER quiescence (mirror `run_transient_fault_scenario`).
    // (The abort path is FENCE-NEUTRAL — no fence sync to wait for; this lets the dest ghost teardown land.)
    for _ in 0..24 {
        topo.step();
    }
    let dead = topo.dead_nodes();
    OrchKillOutcome {
        topo,
        subject: entity,
        dead,
        recovered_states,
    }
}
