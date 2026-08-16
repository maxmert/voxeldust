//! `vd-bins` library — the SINGLE SOURCE OF TRUTH for standing up a local cluster
//! of the node binaries, shared by the `vd-devcluster` launcher AND the
//! `process_parity` real-binary test. Without this shared module the two would
//! hand-duplicate the `VD_*` env contract (the dev-cluster keys + the optional prod-only R-6a durable-
//! incarnation keys `VD_BOOT_STATE_DIR`/`VD_BOOT_DURABLE_ROOT`/`VD_BOOT_STATE_EPHEMERAL_OK`), the 14 operational params, the
//! node roster, the dev auth identity, the peer-book formatter, and the child
//! teardown — and silently DRIFT (the parity gate would stay green while the
//! agent's cluster shipped half-configured). Everything that defines "what a dev
//! cluster IS" lives here, once.
//!
//! This is Tier-B (process glue): exercised by the process-tier parity + smoke
//! tests, not the coverage gate.

pub mod proc_launch;

// The dev-control-only helpers (Tier-B process-gate glue): `flight` is the ONE pilot every cluster
// gate flies (rendezvous-and-park + label-asserted crossing legs); `scene_camera` reconstructs the
// client's live capture camera for the GPU pixel gates. Both lean on `vd-client-harness`, which is
// an optional dep enabled by the `dev-control` feature.
#[cfg(feature = "dev-control")]
pub mod flight;
#[cfg(feature = "dev-control")]
pub mod scene_camera;
// The Slice-D pixel-gate instrument (window lane §2.8/§2.11): the straddled capture, the pilot
// camera reconstruction and the drawn-footprint reading the three acceptance gates share.
#[cfg(feature = "dev-control")]
pub mod pixel;

use std::net::SocketAddr;
use std::path::PathBuf;
use std::process::{Child, Command, ExitStatus};
use std::time::Duration;

use ed25519_dalek::SigningKey;
use vd_core::NodeId;
use vd_devproto::{DevRequest, DevResponse, SlotPorts, WORKTREE_SLOT_CEILING};
use vd_io_prod::runtime::hex32_encode;
use vd_io_prod::runtime::{ConfigError, EnvConfig};

// ---- graceful shutdown (cloud-ready k3d, Slice 1) ----------------------------

/// Install a SHUTDOWN flag flipped on SIGTERM (the signal k8s sends on pod-stop/rolling-deploy, BEFORE
/// its SIGKILL escalation after `terminationGracePeriodSeconds`) or SIGINT (dev Ctrl-C). The three server
/// bins' synchronous tick loops poll this flag each iteration and BREAK to run their graceful drain — so a
/// routine pod-stop stops being a hard SIGKILL crash-path (every restart re-hydrating from the durable
/// store) and becomes a clean flush → fsync → exit. Spawns ONE task on the node's EXISTING tokio runtime
/// (no new thread, no new dependency — tokio's `signal` rides `features = ["full"]`). Returns the flag the
/// loop polls; the caller runs the drain AFTER the loop (see each bin). If the SIGTERM handler cannot be
/// installed (unreachable on the unix deploy/dev targets) it degrades to Ctrl-C only — never a hard failure.
#[must_use]
pub fn install_shutdown_flag(
    runtime: &tokio::runtime::Handle,
) -> std::sync::Arc<std::sync::atomic::AtomicBool> {
    use std::sync::Arc;
    use std::sync::atomic::{AtomicBool, Ordering};
    let flag = Arc::new(AtomicBool::new(false));
    let set = Arc::clone(&flag);
    spawn_signal_watcher(runtime, move || set.store(true, Ordering::Relaxed));
    flag
}

/// [`install_shutdown_flag`] + the S3 readiness DE-ROUTE. On the SIGTERM edge the SAME signal task, BEFORE it
/// sets the flag, publishes NotReady+draining into the shared health cell (`probe::publish_draining`): `/readyz`
/// goes 503 the instant SIGTERM arrives — the loop-head `!shutdown` check would otherwise SKIP the in-body
/// health publish on the exit tick, keeping the pod Ready through termination — and `draining` keeps `/healthz`
/// LIVE through the graceful final-fsync park (never a kubelet SIGKILL mid-write). The tick loop still polls the
/// returned flag; `publish_tick` preserves `draining`, so it cannot un-drain in the one-tick race.
#[must_use]
pub fn install_shutdown_flag_with_health(
    runtime: &tokio::runtime::Handle,
    health: vd_io_prod::probe::HealthCell,
) -> std::sync::Arc<std::sync::atomic::AtomicBool> {
    use std::sync::Arc;
    use std::sync::atomic::{AtomicBool, Ordering};
    let flag = Arc::new(AtomicBool::new(false));
    let set = Arc::clone(&flag);
    spawn_signal_watcher(runtime, move || {
        vd_io_prod::probe::publish_draining(&health);
        set.store(true, Ordering::Relaxed);
    });
    flag
}

/// Spawn ONE task on the node's existing tokio runtime that awaits SIGTERM (the k8s pod-stop signal) or SIGINT
/// (dev Ctrl-C) and runs `on_signal` once. If the SIGTERM handler cannot be installed (not the unix
/// deploy/dev target) it degrades to Ctrl-C only — never a hard failure. Shared by the two shutdown installers.
fn spawn_signal_watcher(
    runtime: &tokio::runtime::Handle,
    on_signal: impl FnOnce() + Send + 'static,
) {
    runtime.spawn(async move {
        let ctrl_c = tokio::signal::ctrl_c();
        match tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate()) {
            Ok(mut term) => {
                tokio::select! {
                    _ = term.recv() => {}
                    _ = ctrl_c => {}
                }
            }
            // SIGTERM handler unavailable (not the unix deploy/dev path) — still honour Ctrl-C.
            Err(_) => {
                let _ = ctrl_c.await;
            }
        }
        on_signal();
    });
}

/// S3: spawn the k8s probe HTTP server (`/healthz` + `/readyz`) on the node's existing tokio runtime — the
/// sibling of [`install_shutdown_flag`], byte-identical to the orchestrator's admin spawn. Body-light,
/// unauthenticated (a kubelet `httpGet` presents no token; access is NetworkPolicy-gated in S4). Binds a plain
/// `TcpListener` + `axum::serve(probe_router(source))`; a bind failure is fatal (a probe-less pod would be
/// silently un-restartable / never-Ready in k8s).
/// `cookie` (RLM 5c) is `Some` ONLY on a spawned realm shard (its `VD_INCARNATION_COOKIE`) — it mounts
/// the `/whoami` echo the realm spawner's pid-reuse guard reads (Step 5e); the orchestrator + gateway pass
/// `None` (bare-status routes only).
pub fn spawn_probe_server(
    runtime: &tokio::runtime::Handle,
    addr: SocketAddr,
    source: std::sync::Arc<dyn vd_io_prod::probe::HealthSource>,
    cookie: Option<String>,
) {
    runtime.spawn(async move {
        let listener = tokio::net::TcpListener::bind(addr)
            .await
            .unwrap_or_else(|e| panic!("probe server failed to bind {addr}: {e}"));
        axum::serve(listener, vd_io_prod::probe::probe_router(source, cookie))
            .await
            .expect("probe server");
    });
}

/// RLM RG-4: spawn the read-only admin HTTP endpoint (`/admin/snapshot` + `/metrics`) on `addr` — the
/// orchestrator's + gateway's 2am `curl`. Mirrors [`spawn_probe_server`]: a detached task that binds then
/// serves forever, and a bind failure is a LOUD panic (a mis-set `VD_ADMIN_ADDR` must not boot silently
/// un-observable). ONE helper both bins call — the `snapshot` source is a lock-free cell the sim thread
/// republishes after every tick (a `curl` never touches the sim thread) and `metrics` is a pure atomic load
/// off the live mesh.
pub fn spawn_admin_server(
    runtime: &tokio::runtime::Handle,
    addr: SocketAddr,
    snapshot: std::sync::Arc<dyn vd_io_prod::admin::SnapshotSource>,
    metrics: std::sync::Arc<dyn vd_io_prod::admin::MetricsSource>,
) {
    runtime.spawn(async move {
        let listener = tokio::net::TcpListener::bind(addr)
            .await
            .unwrap_or_else(|e| panic!("admin server failed to bind {addr}: {e}"));
        axum::serve(listener, vd_io_prod::admin::admin_router(snapshot, metrics))
            .await
            .expect("admin server");
    });
}

// ---- node roster -------------------------------------------------------------

/// The fixed P1 node identities. ONE definition for the launcher + parity test.
pub const ORCH: NodeId = NodeId(1);
pub const GATEWAY: NodeId = NodeId(2);
pub const SHARD: NodeId = NodeId(3);
/// The SIBLING-star realm-shard ([`WorldRoster::sibling`] — the galaxy's lowest-seed ring sibling of
/// the home system) — a [`ClusterShape::Chain`] `up`'s sixth node. Matches the harness `DEST`
/// (`tests/src/lib.rs`). HR3: this is a `NodeId` in a SET, never a shard KIND — adding it is a roster
/// extension, never a code branch on a shard kind.
///
/// SCOPE (M-2): a fixed handful of extra shards for the LOCAL process clusters. The N-shard k3d
/// roster generalization (a `Vec` of shard ids/addrs, N-entry rosters) is ledgered as a separate
/// cloud (#123) slice in `docs/design/DEFERRED.md`.
pub const SHARD_B: NodeId = NodeId(4);
/// The GALAXY between-space shard ([`WorldRoster::galaxy`] — the home region's PARENT). It OWNS the
/// star systems as its CHILDREN, so every inter-system crossing routes THROUGH it (leave the home
/// system → land in the galaxy → the galaxy shard detects the entry into the next system), and
/// authority can REST in the between-space so a multi-hop dot is NEVER orphaned. Spawned by a
/// [`ClusterShape::Dual`] or [`Chain`](ClusterShape::Chain) `up`. HR3: one more `NodeId` in a SET,
/// never a shard KIND.
pub const GALAXY: NodeId = NodeId(5);

/// NODE-PER-REALM (task #149): the INNER-planet realm-shard ([`WorldRoster::inner`] — the home
/// system's smallest-sma mover) on its OWN node, so a flight into its SOI is a CROSS-NODE saga, not a
/// co-hosted local relabel. Present only in a [`ClusterShape::Chain`] `up`. HR3: one more `NodeId` in
/// the roster, never a shard KIND.
pub const PLANET_SHARD: NodeId = NodeId(6);
/// RESERVED (unused): the Station realm-shard slot. THE world generates no Station — stations are
/// player-built, so no seed emits one (D-WORLD-1) — but the node id and its `ClusterAddrs` /
/// `DevPortScheme` slots stay reserved, so the leg returns as a roster extension when the
/// block/station slice grows THE world, never as a Tier-A port renumbering.
pub const STATION_A_SHARD: NodeId = NodeId(7);
/// RESERVED (unused): the Area realm-shard slot — see [`STATION_A_SHARD`].
pub const AREA_A_SHARD: NodeId = NodeId(8);

/// The GALAXY realm seed — must match `vd_core::worldgen`'s Galaxy (`System(1)`, the between-systems
/// space) and the harness `GALAXY_SEED`. The galaxy shard hosts `RealmId::System(GALAXY_SEED)`.
pub const GALAXY_SEED: u64 = 1;

/// RLM RG-4 — the deterministic port band a [`Demand`](ClusterShape::Demand) cluster mints its
/// demand-spawned shards' QUIC + probe + admin ports from (`VD_RLM_FIRST_PORT`/`VD_RLM_PORT_LIMIT`). The
/// spawner mints ports deterministically off `first_port` (the k-th spawn's probe = `first_port + 2k + 1`)
/// for crash-recovery determinism, so it needs a FIXED band, not OS-assigned ports. Chosen ABOVE the
/// `rlm_kill9` bands (42000–45000) so the two process-tier test binaries never collide when run in parallel.
pub const RLM_DEMAND_FIRST_PORT: u32 = 45_000;
/// The exclusive upper bound of [`RLM_DEMAND_FIRST_PORT`]'s band (1000 ports ⇒ ~500 demand shards).
pub const RLM_DEMAND_PORT_LIMIT: u32 = 46_000;

/// The TOPOLOGY shape of a dev cluster — how many stub shards it spawns. A DATA value the shared env
/// builders fan out on (booking extra peers / rosters), NOT a shard-kind match in a feature (HR3): every
/// spawned shard runs the SAME `vd-shard` binary; the shape only says which node ids are in the roster.
/// Every static shape pre-books realms OF THE WORLD, derived through [`world_roster`] — the retired
/// `Triple`/`Forest` shapes named realms THE world does not contain (`System(8)`, `Planet(7)`,
/// `Station(7)`, `Area(7)`), so their extra shards died at boot and their gates pinned green on
/// clusters that never stood (D-WORLD-8).
///
/// - [`Single`](ClusterShape::Single): orchestrator + gateway + the home shard (the base `up`) —
///   login/snapshot/boot guards/render+parity smokes AND the shipped cloud topology
///   (`deploy/k3d/50-shard.yaml`).
/// - [`Dual`](ClusterShape::Dual): + the [`GALAXY`] shard hosting the home region's PARENT — the
///   one-hop autonomous crossing over THE world's own home shell (raw protocol, zero navigation).
/// - [`Chain`](ClusterShape::Chain): + [`GALAXY`], the [`PLANET_SHARD`] inner planet, and the
///   [`SHARD_B`] sibling star — down, up, out, and sideways-through-the-parent, each realm its OWN
///   node so every re-home is a uniform CROSS-NODE saga (no `VD_HELD_REALMS` co-hosting; the
///   source==dest degenerate case never arises).
/// - [`Demand`](ClusterShape::Demand): RLM RG-4 — orchestrator + gateway ONLY, with NO shard pre-booked. The
///   ONLY way a player reaches a world is the armed demand reconciler spinning one up on the fly at login
///   (`VD_DEMAND=1`, mutually exclusive with the static forest). The gateway reaches that just-spawned shard
///   with NO pre-booked address via the reactive greeting (RG-0..3). This is the shape the demand-login e2e
///   (`crates/bins/tests/rlm_demand_login.rs`) stands up.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ClusterShape {
    Single,
    Dual,
    Chain,
    Demand,
}

/// ONE extra realm-shard beyond the base orchestrator+gateway+home trio: its roster [`NodeId`], the
/// [`RealmId`](vd_core::pose::RealmId) it SINGLY hosts, and its QUIC + probe bind. The shape's fan-out
/// ([`realm_shards`]) returns these as DATA the env builders + the launcher iterate — never a
/// shard-KIND branch (HR3). Each carries its realm so the shard bin can boot the right realm KIND
/// (`VD_REALM_KIND`), not just a `System(seed)`.
#[derive(Clone, Copy, Debug)]
pub struct RealmShard {
    pub node: NodeId,
    pub realm: vd_core::pose::RealmId,
    pub quic: SocketAddr,
    pub probe: SocketAddr,
}

impl ClusterShape {
    /// RLM RG-4: is this the DEMAND shape (orchestrator + gateway only, the reconciler armed, NO static
    /// shard)? The `*_env` builders fan out on this to emit `VD_DEMAND` + drop `VD_STATIC_FOREST`/the booked
    /// shard — data, not a shard-KIND branch. `false` for every static shape (byte-identical).
    #[must_use]
    pub fn is_demand(self) -> bool {
        matches!(self, ClusterShape::Demand)
    }
}

/// The realms a shape pre-books BEYOND the always-present home realm, in spawn order — the ONE
/// fan-out both [`roster_realms`] and [`realm_shards`] read, so the realm list and the shard list can
/// never disagree. Every entry comes off [`world_roster`], the only place allowed to name a realm.
fn extra_roster_realms(shape: ClusterShape, roster: &WorldRoster) -> Vec<vd_core::pose::RealmId> {
    match shape {
        ClusterShape::Single | ClusterShape::Demand => Vec::new(),
        ClusterShape::Dual => vec![roster.galaxy],
        ClusterShape::Chain => vec![roster.galaxy, roster.inner, roster.sibling],
    }
}

/// EVERY realm a static `shape` pre-books (home first, then the extras in spawn order) — the
/// readiness roster: `up` returns 0 only once `AdminSnapshot::realms_present` holds over exactly this
/// list, so a crossing can never resolve an ungranted head. [`Demand`](ClusterShape::Demand)
/// pre-books NOTHING (its worlds spin up on login/AoI demand), so its list is EMPTY. Addr-free by
/// design (J5): enumerating realms must not require inventing placeholder addresses.
#[must_use]
pub fn roster_realms(shape: ClusterShape, p: &DevClusterParams) -> Vec<vd_core::pose::RealmId> {
    if shape.is_demand() {
        return Vec::new();
    }
    let roster = world_roster(p);
    let mut realms = vec![roster.home];
    realms.extend(extra_roster_realms(shape, &roster));
    realms
}

/// The single-realm shards `shape` stands up BEYOND the base orchestrator + gateway + home shard —
/// each extra roster realm zipped onto its fixed node/port slot ([`GALAXY`], [`PLANET_SHARD`],
/// [`SHARD_B`], in roster order). ONE data fan-out for the env builders and the launcher: Single and
/// Demand spawn none, Dual spawns the galaxy, Chain the galaxy + inner planet + sibling star. The
/// home shard is NOT listed (it is the always-present base).
#[must_use]
pub fn realm_shards(
    shape: ClusterShape,
    a: &ClusterAddrs,
    p: &DevClusterParams,
) -> Vec<RealmShard> {
    if shape.is_demand() || shape == ClusterShape::Single {
        return Vec::new(); // no world derivation on the shapes that book no extra shard
    }
    let roster = world_roster(p);
    let slots: [(NodeId, SocketAddr, SocketAddr); 3] = [
        (GALAXY, a.galaxy, a.galaxy_probe),
        (PLANET_SHARD, a.planet, a.planet_probe),
        (SHARD_B, a.shard_b, a.shard_b_probe),
    ];
    extra_roster_realms(shape, &roster)
        .into_iter()
        .zip(slots)
        .map(|(realm, (node, quic, probe))| RealmShard {
            node,
            realm,
            quic,
            probe,
        })
        .collect()
}

// ---- dev auth identity -------------------------------------------------------

/// DEV-ONLY Ed25519 login seed for the localhost cluster. The gateway boots with
/// its verifying key; clients mint logins with the seed. NEVER a production
/// secret — a fixed dev constant so the identity is reproducible. (Structurally
/// fencing it behind a dev/test compile guard is tracked for the prod-build phase.)
pub const DEV_AUTH_SEED: [u8; 32] = [0x42; 32];

/// THE LOG LEVEL EVERY NODE BOOTS AT — `RUST_LOG` when the operator sets one, [`DEFAULT_LOG_FILTER`]
/// otherwise.
///
/// One function, called by every binary, because it was five copies of the same literal. A hardcoded
/// filter also meant any diagnostic added to chase a live defect had to be shipped at `info` or never
/// seen, which is how a debugging line ends up permanently in a hot path.
pub fn init_tracing() {
    tracing_subscriber::fmt()
        .with_env_filter(
            std::env::var("RUST_LOG").unwrap_or_else(|_| DEFAULT_LOG_FILTER.to_owned()),
        )
        .init();
}

/// The level a node logs at when the operator names none.
pub const DEFAULT_LOG_FILTER: &str = "info";

/// The 32-byte DEV verifying key (derived once from [`DEV_AUTH_SEED`]). The gateway passes
/// `Some(dev_auth_pubkey_bytes())` to `vd_io_prod::boot::enforce_cloud_preflight` so a cloud profile REFUSES
/// to boot with the built-in dev key (cloud-ready k3d Slice 2). Passed by-value into io-prod (which cannot
/// depend on vd-bins) to avoid a circular dependency.
#[must_use]
pub fn dev_auth_pubkey_bytes() -> [u8; 32] {
    SigningKey::from_bytes(&DEV_AUTH_SEED)
        .verifying_key()
        .to_bytes()
}

/// The hex verifying key the gateway boots with (`VD_AUTH_PUBKEY`).
#[must_use]
pub fn dev_auth_pubkey_hex() -> String {
    hex32_encode(&dev_auth_pubkey_bytes())
}

/// The hex signing seed clients mint logins with (`VD_AUTH_SIGNING_KEY` in the
/// cluster contract). Dev-only — see [`DEV_AUTH_SEED`].
#[must_use]
pub fn dev_auth_signing_key_hex() -> String {
    hex32_encode(&DEV_AUTH_SEED)
}

/// S4b (cloud-ready k3d): derive the `(verifying-key hex, signing-seed hex)` pair for a PRODUCTION session-auth
/// key from a 32-byte `seed`. The verifying key → the gateway's `VD_AUTH_PUBKEY` / the `vd-auth` Secret (the
/// cloud profile VETOES the built-in dev key, `boot::CloudProfileError::DevAuthKey`); the signing seed → the
/// legitimate clients OUT-OF-BAND, NEVER a server Secret. PURE (the OS-CSPRNG read that supplies `seed` lives in
/// the `vd-devcluster gen-authkey` tool) so it is unit-testable; feeding [`DEV_AUTH_SEED`] reproduces the dev key.
#[must_use]
pub fn auth_keypair_hex_from_seed(seed: &[u8; 32]) -> (String, String) {
    let signing = SigningKey::from_bytes(seed);
    (
        hex32_encode(&signing.verifying_key().to_bytes()),
        hex32_encode(seed),
    )
}

// ---- operational parameters --------------------------------------------------

/// Every operational knob a dev cluster is configured with — ONE typed struct
/// (CLAUDE.md: operational params in one place, never scattered literals). The
/// launcher and the parity test both render their env from `DEV`.
#[derive(Clone, Copy, Debug)]
pub struct DevClusterParams {
    pub tick_hz: u32,
    pub outbound_cap: u32,
    pub epoch: u64,
    pub reserve_chunk: u64,
    pub lease_ttl: u64,
    pub session_seed: u64,
    pub max_sessions: u32,
    pub max_buffered_inputs: u32,
    pub realm_seed: u64,
    pub move_speed: f64,
    pub tick_dt: f64,
    pub mint_seed: u64,
    pub input_log_cap: u32,
    pub realm_recheck: u64,
    pub snapshot_budget: u32,
    /// RLM RG-4 (demand cluster): the predictive AoI horizon (`VD_BOOT_TICKS_P99`) — how far ahead a
    /// demand-spawned shard warms a child realm BEFORE an occupant reaches it, AND the orchestrator's own
    /// RLM-tuning read. 0 ⇒ reactive-only (spin up on arrival — the login IS the trigger); a visual walk-in
    /// bumps it. Emitted ONLY by the [`Demand`](ClusterShape::Demand) arm; absent (⇒ shard default) elsewhere.
    pub boot_ticks_p99: u64,
    /// RLM RG-4 (demand cluster): the universe seed (`VD_UNIVERSE_SEED`) the orchestrator + every
    /// demand-spawned shard generate their realm forest from. 0 = the default Walk-scale forest (the seed the
    /// shard bin already defaults to), so a demand cluster's forest matches the static clusters' byte-for-byte.
    pub universe_seed: u64,
}

/// The standard dev-cluster parameters (fast ticks for snappy bring-up).
pub const DEV: DevClusterParams = DevClusterParams {
    tick_hz: 50,
    outbound_cap: 256,
    epoch: 1,
    reserve_chunk: 4096,
    lease_ttl: 10_000,
    session_seed: 23,
    max_sessions: 8,
    // The canonical cut-window-sized default (kept below the transport per-tick caps — the
    // drain-burst invariant); sourced from the ONE config struct, never an inline literal.
    max_buffered_inputs: vd_connection_plane::gateway::TransportTuning::DEFAULT_MAX_BUFFERED_INPUTS
        as u32,
    realm_seed: 7,
    // The realistic-demo flight speed (RLM Slice 3). A ship spends real seconds flying from where a body
    // becomes VISIBLE (angular size crosses θ_min — ~318 m for a planet) to where it CROSSES containment
    // (~4 m planet SOI), so the demand loop boots the child DURING that flight — the spin-up-ahead is a
    // natural consequence of visibility-radius ≫ crossing-radius, not a predictive horizon
    // (`boot_ticks_p99` stays 0). Rides `VD_SPEED` into the shard, so `visual_demand`'s `occupant_v_max`
    // IS the sim's integrated speed, and the AoI lead is measured against the speed actually flown.
    //
    // THROWAWAY (tiny world). This is the BOOST/warp figure — the number the pilot gets holding the boost
    // key; releasing it cruises at a fraction of this (`CRUISE_FRACTION`, client-side, since the axes ride
    // the wire as a magnitude). Sized for the interstellar leg: neighbouring stars sit ~12 km apart, so a
    // hop is ~24 s — long enough to watch one star wake ahead while the one behind goes to sleep, short
    // enough to fly repeatedly. Cruise then lands at 15 m/s, which crosses a 300 m system in ~20 s.
    // Speed is a plain configured number with no cap anywhere — this is the whole reason warp needs no
    // special machinery, only a bigger one of these.
    move_speed: 500.0,
    tick_dt: 0.02,
    mint_seed: 11,
    input_log_cap: 4096,
    // The self-fence realm-head recheck cadence (D-3), INERT (0) in dev — the self-fence is off (grace 0). The
    // VU-AoI parent-resolution / up-relay the observation cascade needs is DECOUPLED from this in
    // `parent_headread_due` (it falls back to a tick-derived demand cadence when this is 0), so leaving the
    // self-fence off does not disable the cascade.
    realm_recheck: 0,
    snapshot_budget: 1100,
    boot_ticks_p99: 0, // reactive-only in dev (the login is the demand trigger); a visual walk-in bumps it.
    universe_seed: 0, // the default Walk-scale forest (matches the shard bin's VD_UNIVERSE_SEED default).
};

/// SCALE-MAXSESSIONS-VS-K (compile-time): the gateway must admit at least K =
/// `max_clients_per_worktree` sessions, or the last dev-control client window for a
/// slot could never log in. A regression here fails the BUILD, not a flaky test.
const _: () = assert!(
    DEV.max_sessions as u64 >= vd_devproto::DevPortScheme::DEFAULT.max_clients_per_worktree as u64,
    "DEV.max_sessions must be >= max_clients_per_worktree (K) so every client logs in",
);

/// TICK-PAIR (compile-time): `tick_hz` (the pacer rate) and `tick_dt` (the shard's
/// integration step) are ONE physical quantity — the tick period — expressed twice.
/// If they drift (e.g. retuning tick_hz without flipping tick_dt) the shard silently
/// integrates motion at the wrong dt while the pacer ticks at the wrong rate: avatar
/// speed, snapshot cadence, and the client interp buffer all desync with no error.
/// A drift fails the BUILD (audit DRY-A).
const _: () = assert!(
    {
        let diff = 1.0 / (DEV.tick_hz as f64) - DEV.tick_dt;
        // const-context abs(): both signs checked explicitly.
        diff < 1e-9 && diff > -1e-9
    },
    "DEV.tick_dt must equal 1/DEV.tick_hz (one tick period, two encodings)",
);

/// The tolerance for [`validate_tick_pair`] — matches the compile-time TICK-PAIR assert above.
pub const TICK_DT_EPSILON: f64 = 1e-9;

/// A `VD_TICK_DT` that does not match `1/VD_TICK_HZ` — rejected LOUD at shard boot. (Manual `Display`/`Error`
/// impls: `vd-bins` does not depend on `thiserror`, and this keeps it out of the dep set.)
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TickPairError {
    pub tick_hz: u32,
    pub tick_dt: f64,
    pub expected: f64,
}

impl std::fmt::Display for TickPairError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "VD_TICK_DT {} must equal 1/VD_TICK_HZ = {} (VD_TICK_HZ={}); they are ONE tick period expressed \
             twice — a cloud ConfigMap that retunes one but not the other silently integrates motion at the \
             wrong step",
            self.tick_dt, self.expected, self.tick_hz
        )
    }
}

impl std::error::Error for TickPairError {}

/// TICK-PAIR (RUNTIME): the shard reads `VD_TICK_DT` and `VD_TICK_HZ` as INDEPENDENT env vars, so the
/// compile-time [`DEV`] assert above cannot catch a cloud ConfigMap that changes one without the other. This
/// cross-checks the env-supplied pair at shard boot — a drift silently integrates the avatar's motion at the
/// wrong `dt` (no crash) otherwise. `tick_hz.max(1)` mirrors `TickPacer::new`'s degenerate-0 floor.
///
/// # Errors
/// [`TickPairError`] when `tick_dt` differs from `1/tick_hz` by more than [`TICK_DT_EPSILON`].
pub fn validate_tick_pair(tick_hz: u32, tick_dt: f64) -> Result<(), TickPairError> {
    let expected = 1.0 / f64::from(tick_hz.max(1));
    if (tick_dt - expected).abs() > TICK_DT_EPSILON {
        return Err(TickPairError {
            tick_hz,
            tick_dt,
            expected,
        });
    }
    Ok(())
}

/// DRAIN-BURST (compile-time): the gateway cut-buffer drain (`apply_commit`) pushes up to
/// `max_buffered_inputs` frames to the dest in ONE tick; it MUST stay within the dest
/// `BoundedInbox` floor or a full drain plus any co-arriving frame sheds a conserved (unreliable)
/// input (1c.5 / DEFERRED D-8). The floor formula lives ONCE in `vd_io_prod::mesh`; this asserts
/// the shipped DEV config honors it (enforce, don't just document — the only previously
/// convention-only half of the invariant). A retune that violates it fails the BUILD.
const _: () = assert!(
    DEV.max_buffered_inputs as usize
        <= vd_io_prod::mesh::inbound_capacity_for(DEV.outbound_cap as usize),
    "DEV.max_buffered_inputs must stay within the dest inbox floor inbound_capacity_for(outbound_cap) — the drain-burst invariant (D-8)",
);

/// The DRAIN-BURST invariant violated at RUNTIME: the gateway's cut-buffer drain (`max_buffered_inputs`
/// frames in one tick) exceeds the dest shard's `BoundedInbox` floor `inbound_capacity_for(outbound_cap)`,
/// so a full drain plus a co-arriving frame sheds a CONSERVED (unreliable) input at the transfer-commit
/// moment (D-8). Carries the offending pair + the computed floor for the boot log.
#[derive(Debug, PartialEq, Eq)]
pub struct DrainBurstError {
    pub outbound_cap: usize,
    pub max_buffered_inputs: usize,
    pub floor: usize,
}

impl std::fmt::Display for DrainBurstError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "DRAIN-BURST invariant violated (D-8): VD_MAX_BUFFERED_INPUTS={} exceeds the dest inbox floor \
             inbound_capacity_for(VD_OUTBOUND_CAP={})={} — a full cut-buffer drain would shed a conserved input",
            self.max_buffered_inputs, self.outbound_cap, self.floor
        )
    }
}

impl std::error::Error for DrainBurstError {}

/// DRAIN-BURST (RUNTIME): the runtime twin of the compile-time [`DEV`] assert above, mirroring
/// [`validate_tick_pair`]. The const assert covers only the DEV pair; a cloud ConfigMap can lower
/// `VD_OUTBOUND_CAP` (shrinking the dest inbox floor) WITHOUT lowering the gateway's
/// `VD_MAX_BUFFERED_INPUTS`, which silently sheds a conserved resume-input at the transfer-commit moment
/// (no crash). This cross-checks the env-supplied pair at gateway boot. The floor formula stays ONCE in
/// `vd_io_prod::mesh` (enforce, don't just document — the previously convention-only runtime half).
///
/// # Errors
/// [`DrainBurstError`] when `max_buffered_inputs` exceeds `inbound_capacity_for(outbound_cap)`.
pub fn validate_drain_burst(
    outbound_cap: usize,
    max_buffered_inputs: usize,
) -> Result<(), DrainBurstError> {
    let floor = vd_io_prod::mesh::inbound_capacity_for(outbound_cap);
    if max_buffered_inputs > floor {
        return Err(DrainBurstError {
            outbound_cap,
            max_buffered_inputs,
            floor,
        });
    }
    Ok(())
}

// ---- the env contract --------------------------------------------------------

/// The bind addresses of the three fixed cluster nodes (admin is the orchestrator's
/// read-only HTTP endpoint).
#[derive(Clone, Copy, Debug)]
pub struct ClusterAddrs {
    pub orchestrator: SocketAddr,
    pub gateway: SocketAddr,
    pub shard: SocketAddr,
    pub admin: SocketAddr,
    /// RLM RG-4 — the GATEWAY's read-only admin HTTP bind (the gateway analogue of [`admin`](Self::admin),
    /// which is the orchestrator's). `Some` ONLY where a cluster wants the gateway's `/admin/snapshot`
    /// (`GatewayView`) + `/metrics` scrapeable — the [`Demand`](ClusterShape::Demand) cluster and every cloud
    /// pod. `None` everywhere else keeps [`gateway_env`] from emitting `VD_ADMIN_ADDR`, so existing
    /// every static cluster is byte-identical (no extra port bound).
    pub gateway_admin: Option<SocketAddr>,
    /// The k8s /healthz+/readyz probe listeners (S3) — one per node (each needs its own kubelet-reachable
    /// port). Named fields (no offset math), matching the existing style.
    pub orchestrator_probe: SocketAddr,
    pub gateway_probe: SocketAddr,
    pub shard_probe: SocketAddr,
    /// The SIBLING-star realm-shard's ([`SHARD_B`]) QUIC bind + probe. Populated for every cluster
    /// (data, no branch); dialed/spawned ONLY by a [`Chain`](ClusterShape::Chain) `up`. N-shard
    /// generalization (a `Vec` of shard addrs) is ledgered to cloud #123 in `docs/design/DEFERRED.md`.
    pub shard_b: SocketAddr,
    pub shard_b_probe: SocketAddr,
    /// The GALAXY between-space shard's QUIC bind + probe. Populated for every cluster (data, no
    /// branch); dialed/spawned by a [`Dual`](ClusterShape::Dual)/[`Chain`](ClusterShape::Chain) `up`.
    /// Twin of the [`shard_b`](Self::shard_b) pair — one more shard in the roster, never a shard KIND.
    pub galaxy: SocketAddr,
    pub galaxy_probe: SocketAddr,
    /// NODE-PER-REALM (task #149) — the inner-planet realm-shard's ([`PLANET_SHARD`]) QUIC bind +
    /// probe, dialed/spawned ONLY by a [`Chain`](ClusterShape::Chain) `up`; plus the RESERVED
    /// station/area slots (see [`STATION_A_SHARD`] — THE world generates neither yet, so nothing binds
    /// them, but the port offsets hold so the legs return as a roster extension, never a renumbering).
    pub planet: SocketAddr,
    pub planet_probe: SocketAddr,
    pub station: SocketAddr,
    pub station_probe: SocketAddr,
    pub area: SocketAddr,
    pub area_probe: SocketAddr,
}

impl ClusterAddrs {
    /// Reserve a full, INTERNALLY DISTINCT set of loopback addresses for one process-tier cluster —
    /// the one fixture every such test builds its addresses from.
    ///
    /// WHY THIS EXISTS RATHER THAN 17 SEPARATE CALLS. [`reserve_udp_addr`]/[`reserve_tcp_addr`] bind
    /// `:0`, read the address, and DROP the socket immediately. Called seventeen times in a struct
    /// literal, each reservation is released before the next is taken, so the OS is free to hand the
    /// SAME ephemeral port back twice and two fields of one cluster silently alias. Holding all
    /// seventeen sockets simultaneously and dropping them together makes that aliasing STRUCTURALLY
    /// impossible. (It does NOT remove the machine-wide TOCTOU between the drop and the child's real
    /// bind — see `D-GATE-1` in `docs/design/DEFERRED.md`; the complete fix is per-fixture port bands.)
    ///
    /// DISTINCTNESS IS PER PROTOCOL, AND THAT IS DELIBERATE. UDP and TCP are SEPARATE port
    /// namespaces: a UDP field and a TCP field legitimately holding the same port number is not a
    /// collision. Asserting all seventeen are distinct as one set would therefore invent a fresh
    /// ~1-in-227 flake per call, across ~20 call sites — a flakiness fix that adds flakiness.
    ///
    /// `gateway_admin` is left `None` (the byte-identical default that keeps [`gateway_env`] from
    /// emitting `VD_ADMIN_ADDR`); a cluster that wants it adds it with [`Self::with_gateway_admin`].
    #[must_use]
    pub fn reserve() -> ClusterAddrs {
        // Every field bound AT ONCE. Each socket lives until the end of this function, so no two
        // fields can be handed the same port within their own namespace.
        let udp: [std::net::UdpSocket; 8] = std::array::from_fn(|_| {
            std::net::UdpSocket::bind("127.0.0.1:0").expect("reserve cluster udp")
        });
        let tcp: [std::net::TcpListener; 9] = std::array::from_fn(|_| {
            std::net::TcpListener::bind("127.0.0.1:0").expect("reserve cluster tcp")
        });
        let u = |i: usize| udp[i].local_addr().expect("udp addr");
        let t = |i: usize| tcp[i].local_addr().expect("tcp addr");
        ClusterAddrs {
            // QUIC binds (UDP) — every node's `VD_BIND`.
            orchestrator: u(0),
            gateway: u(1),
            shard: u(2),
            shard_b: u(3),
            galaxy: u(4),
            planet: u(5),
            station: u(6),
            area: u(7),
            // Admin + k8s probe listeners (TCP) — `VD_ADMIN_ADDR` / `VD_PROBE_ADDR`.
            admin: t(0),
            orchestrator_probe: t(1),
            gateway_probe: t(2),
            shard_probe: t(3),
            shard_b_probe: t(4),
            galaxy_probe: t(5),
            planet_probe: t(6),
            station_probe: t(7),
            area_probe: t(8),
            gateway_admin: None,
        }
    }

    /// Add the gateway's read-only admin HTTP bind (the Demand cluster + every cloud pod want it).
    #[must_use]
    pub fn with_gateway_admin(mut self, addr: SocketAddr) -> ClusterAddrs {
        self.gateway_admin = Some(addr);
        self
    }

    /// The slot-port → loopback-address mapping — ONE definition shared by the `vd-devcluster`
    /// launcher and the process smokes that read the same slot (a test re-deriving the mapping by
    /// hand is the exact hand-computed-port hazard `vd-devproto` forbids). The station/area fields
    /// stay populated from their RESERVED port slots (data, no branch); nothing binds them until the
    /// block/station slice grows THE world.
    #[must_use]
    pub fn for_slot(ports: SlotPorts) -> ClusterAddrs {
        ClusterAddrs {
            orchestrator: loopback(ports.orchestrator),
            gateway: loopback(ports.gateway),
            shard: loopback(ports.shard),
            admin: loopback(ports.admin),
            gateway_admin: None,
            orchestrator_probe: loopback(ports.probe_orchestrator),
            gateway_probe: loopback(ports.probe_gateway),
            shard_probe: loopback(ports.probe_shard),
            shard_b: loopback(ports.shard_b),
            shard_b_probe: loopback(ports.probe_shard_b),
            galaxy: loopback(ports.galaxy),
            galaxy_probe: loopback(ports.probe_galaxy),
            planet: loopback(ports.planet),
            planet_probe: loopback(ports.probe_planet),
            station: loopback(ports.station),
            station_probe: loopback(ports.probe_station),
            area: loopback(ports.area),
            area_probe: loopback(ports.probe_area),
        }
    }

    /// The eight QUIC (UDP) binds, in field order. Used by the distinctness tests.
    #[must_use]
    pub fn udp_binds(&self) -> [SocketAddr; 8] {
        [
            self.orchestrator,
            self.gateway,
            self.shard,
            self.shard_b,
            self.galaxy,
            self.planet,
            self.station,
            self.area,
        ]
    }

    /// The nine always-present admin/probe (TCP) listeners, in field order. `gateway_admin` is
    /// excluded because it is optional; a caller that sets it supplies its own reservation.
    #[must_use]
    pub fn tcp_binds(&self) -> [SocketAddr; 9] {
        [
            self.admin,
            self.orchestrator_probe,
            self.gateway_probe,
            self.shard_probe,
            self.shard_b_probe,
            self.galaxy_probe,
            self.planet_probe,
            self.station_probe,
            self.area_probe,
        ]
    }
}

/// Format a peer address book as the `id=addr,…` string the nodes parse from
/// `VD_PEERS` (`EnvConfig::peer_book`). ONE formatter (was `book`/`book_string`).
#[must_use]
pub fn book(pairs: &[(NodeId, SocketAddr)]) -> String {
    pairs
        .iter()
        .map(|(id, addr)| format!("{}={addr}", id.0))
        .collect::<Vec<_>>()
        .join(",")
}

fn str_pair(key: &'static str, value: impl ToString) -> (&'static str, String) {
    (key, value.to_string())
}

/// A per-LAUNCH monotone process incarnation (wall-clock milliseconds since the UNIX epoch). The mesh's
/// at-least-once receiver (R-3') resets its dedup high-water only when a peer's incarnation INCREASES — so a
/// node that restarts (its per-(peer,class) seq counter resets to 0) while a peer keeps running must come up
/// at a STRICTLY HIGHER incarnation than the value that peer's SURVIVING ledger still holds, or the restarted
/// sender's fresh seq0.. would be silently deduped away (a silent-data-loss landmine that only bites after a
/// restart — exactly the case R-3' exists to survive). A fresh `up` reads the clock again ⇒ a higher value (a real
/// process teardown+respawn is ≫1ms apart). BEST-EFFORT for dev: a sub-ms crash-loop restart could mint an EQUAL
/// incarnation (the same-ms residual) and a wall-clock REWIND a LOWER one — BOTH are accepted dev residuals closed by
/// R-6 (P6/P7)'s durable monotone boot counter. Not a production identity source.
#[must_use]
pub fn launch_incarnation() -> u64 {
    u64::try_from(
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis(),
    )
    .unwrap_or(u64::MAX)
}

/// The sidecar file (under `VD_BOOT_STATE_DIR`) holding the R-6a durable monotone boot-counter.
pub const BOOT_COUNTER_NAME: &str = "boot.counter";

/// Strict boolean env parse — the shared `1/true/yes | 0/false/no | error` ladder (a present-but-unrecognized
/// value is a LOUD config error, never a fail-OPEN footgun that silently disables a safety guard). Absent ⇒
/// `false` (the production-safe default).
///
/// # Errors
/// A present-but-non-boolean value.
pub fn parse_bool_env(env: &EnvConfig, key: &str) -> Result<bool, Box<dyn std::error::Error>> {
    // DRY (Slice 2): the bool-parsing lives ONCE in `EnvConfig::bool`; this thin wrapper keeps the historic
    // `Box<dyn Error>` signature its callers use.
    env.bool(key).map_err(Into::into)
}

/// Resolve THE process incarnation stamped on every reliable frame (R-6a / M3), in precedence order:
/// 1. `VD_PROCESS_INCARNATION` if EXPLICITLY set (the dev/test/loopback path via [`common_env`], and the
///    future orchestrator-issued path) — WINS, so dev keeps the cheap [`launch_incarnation`] value and the
///    provisioning slice can supersede at zero rework.
/// 2. else `VD_BOOT_STATE_DIR` present ⇒ the DURABLE MONOTONE self-counter ([`vd_io_prod::boot::BootCounter`]
///    — survives a k8s CrashLoop restart / a wall-clock rewind). Guarded by
///    [`vd_io_prod::boot::check_durable_path`] (an allow-list under `VD_BOOT_DURABLE_ROOT` if declared, else
///    the temp deny-list; `VD_BOOT_STATE_EPHEMERAL_OK=1` is the explicit dev escape). The genesis floor is
///    [`launch_incarnation`] so a re-provisioned pod with a FRESH volume still exceeds a peer's surviving
///    wall-clock-era ledger; steady state is purely counter+1.
/// 3. else ⇒ FAIL LOUD (a prod pod that forgets both must NOT silently revert to the unsafe wall-clock path)
///    unless `VD_BOOT_STATE_EPHEMERAL_OK=1` (then the wall-clock value, for a throwaway dev/test node).
///
/// # Errors
/// A non-durable/mis-configured boot path, a corrupt counter file, or a mis-set env value.
pub fn resolve_process_incarnation(env: &EnvConfig) -> Result<u64, Box<dyn std::error::Error>> {
    // (1) An explicit value wins (dev/test via common_env; a future orchestrator-issued value).
    if env.string("VD_PROCESS_INCARNATION").is_ok() {
        return Ok(env.parse::<u64>("VD_PROCESS_INCARNATION")?);
    }
    let ephemeral_ok = parse_bool_env(env, "VD_BOOT_STATE_EPHEMERAL_OK")?;
    // An EMPTY/whitespace value is treated as ABSENT — an unset-var expansion (`export X=$UNSET`) must NOT
    // fail-OPEN the durability guard (an empty `VD_BOOT_DURABLE_ROOT` would make the allow-list pass every
    // path) nor write the counter into CWD (an empty `VD_BOOT_STATE_DIR`).
    let non_empty = |key: &str| {
        env.string(key)
            .ok()
            .map(|s| s.trim().to_owned())
            .filter(|s| !s.is_empty())
    };
    // (2) The durable monotone self-counter on a persistent volume.
    if let Some(dir) = non_empty("VD_BOOT_STATE_DIR") {
        let path = PathBuf::from(dir).join(BOOT_COUNTER_NAME);
        let durable_root = non_empty("VD_BOOT_DURABLE_ROOT").map(PathBuf::from);
        vd_io_prod::boot::check_durable_path(&path, durable_root.as_deref(), ephemeral_ok)?;
        return Ok(vd_io_prod::boot::BootCounter::increment_on_boot(
            &path,
            launch_incarnation(),
        )?);
    }
    // (3) Neither set: fail loud unless the explicit ephemeral escape (then the unsafe wall-clock value).
    if ephemeral_ok {
        return Ok(launch_incarnation());
    }
    Err("no durable process incarnation: set VD_BOOT_STATE_DIR to a persistent path (the M3 durable monotone \
         boot-counter — required in the cloud so a CrashLoop/reschedule restart is not silently deduped), or \
         set VD_PROCESS_INCARNATION explicitly. Set VD_BOOT_STATE_EPHEMERAL_OK=1 ONLY to accept the wall-clock \
         incarnation for a throwaway dev/test node. Refusing to boot."
        .into())
}

/// The per-node durable OUTBOX (R-6d) — opened iff `VD_OUTBOX_PATH` names a non-empty path, else `None` (a
/// node with no producer-less durable reliable flow runs without one, and `step_tick` stays byte-identical).
/// The path is guarded by the SAME [`check_durable_path`] the M3 boot-counter uses (F2, DRY): an ALLOW-list
/// under `VD_STORE_DURABLE_ROOT` when declared (the mounted persistent volume — catches a k8s `emptyDir` a
/// prefix deny-list cannot enumerate), else the temp DENY-list; `VD_OUTBOX_EPHEMERAL_OK=1` is the explicit
/// dev/test escape (a throwaway cluster legitimately stores its outbox under `$TMPDIR`).
///
/// [`check_durable_path`]: vd_io_prod::boot::check_durable_path
///
/// # Errors
/// A non-durable / mis-configured outbox path, an unopenable store, or a mis-set boolean env value.
pub fn open_node_outbox(
    env: &EnvConfig,
) -> Result<Option<vd_io_prod::outbox::NodeOutbox>, Box<dyn std::error::Error>> {
    let path = match env.string("VD_OUTBOX_PATH") {
        Ok(p) if !p.trim().is_empty() => std::path::PathBuf::from(p),
        _ => return Ok(None), // absent / empty ⇒ no durable outbox on this node
    };
    // Create the parent FIRST so the guard can canonicalize it (resolve symlinks) — mirrors the store path.
    if let Some(parent) = path.parent().filter(|p| !p.as_os_str().is_empty()) {
        std::fs::create_dir_all(parent)?;
    }
    let durable_root = env
        .string("VD_STORE_DURABLE_ROOT")
        .ok()
        .filter(|s| !s.trim().is_empty())
        .map(std::path::PathBuf::from);
    let ephemeral_ok = parse_bool_env(env, "VD_OUTBOX_EPHEMERAL_OK")?;
    vd_io_prod::boot::check_durable_path(&path, durable_root.as_deref(), ephemeral_ok)?;
    Ok(Some(vd_io_prod::outbox::NodeOutbox::open(
        &path,
        vd_io_prod::store::StoreTuning::default(),
    )?))
}

/// R-6d3b-2b: THE one node boot sequence shared by `shard.rs` + `gateway.rs` (HR3 — no per-kind fork). It
/// (1) resolves the process incarnation EXACTLY ONCE (finding C — a second [`resolve_process_incarnation`]
/// would double-increment the durable BootCounter), (2) opens the durable outbox (`Some` iff `VD_OUTBOX_PATH`
/// is set) + wraps it as the cloneable `SharedOutbox`, (3) spawns the mesh WITH the sink (the
/// durable-before-send gate LIVE), then (4) REPLAYS the retained rows BEFORE the transport is handed to
/// `build_app`. Returns the transport + `MeshControl`; the caller MUST keep the control AND the owning tokio
/// runtime alive for the whole tick loop (dropping control closes the endpoint; dropping the runtime stops the
/// peer-writer tasks).
///
/// The replay fences on DURABILITY (block A submit), NOT DELIVERY (block C QUIC send) — so it does NOT hang
/// on peers down at boot; their rows are RETAINED + the R-4a retransmit timer re-drives on reconnect. A poison
/// or roster-gone row is QUARANTINED (RETAINED + loud-counted) and the boot PROCEEDS (RC-2a) — one bad row
/// never wedges the node; only a transient wedge (`LaneStuck`/`LaneDead`/`FenceTimeout`) refuses to boot.
///
/// # Errors
/// A bad incarnation source; an unopenable/non-durable outbox path; a `spawn_mesh` failure; or a replay that
/// hits a transient wedge.
/// Cloud-ready k3d Slice 2 — the shared shard/gateway D-3 boot step (HR3: ONE place, so the two node bins
/// cannot drift on the footgun preflight or the derived config). Runs `enforce_cloud_preflight` (fail-loud on
/// ephemeral escapes / manual incarnation / a bad durable root / the dev auth key) then `resolve_d3`. MUST be
/// called BEFORE [`boot_mesh_and_replay`] — the preflight has to veto a manual `VD_PROCESS_INCARNATION` /
/// ephemeral boot-state escape BEFORE that resolves the M3 durable boot-counter (a durable action). The gateway
/// passes `Some(dev_auth_pubkey_bytes())`; the shard passes `None` (it has no auth key). Returns the resolved
/// D-3 config; the node reads `node_self_fence_grace` + `node_recheck` from it (the orchestrator's
/// `directory`/`liveness` fields are inert for a node role and ignored).
///
/// # Errors
/// Any cloud-profile violation or incoherent D-3 tuning (boxed, printed `Refusing to boot`).
pub fn resolve_node_d3(
    env: &EnvConfig,
    role: vd_io_prod::boot::NodeRole,
    dev_pubkey: Option<[u8; 32]>,
) -> Result<vd_io_prod::boot::ResolvedD3, Box<dyn std::error::Error>> {
    let profile = vd_io_prod::boot::enforce_cloud_preflight(env, dev_pubkey)?;
    let tick_hz: u32 = env.parse("VD_TICK_HZ")?;
    vd_io_prod::boot::resolve_d3(env, profile, role, tick_hz)
}

/// S3: resolve the k8s probe server config, or `None` if `VD_PROBE_ADDR` is unset (the in-process/parity rigs
/// stay byte-identical until they opt in — the dev cluster, the probe process test, and every cloud pod set
/// it). `VD_PROBE_STALL_TICKS` overrides the wedge budget (defaults to [`ProbeTuning::DEFAULT`]); a present but
/// `< 2` value fails LOUD via `validate()`. NOT folded into `resolve_d3` — a probe is a different concern with
/// no inert/split-brain hazard, and it is meaningful in DevTest too (the process tests curl it).
///
/// # Errors
/// A malformed `VD_PROBE_ADDR` / `VD_PROBE_STALL_TICKS`, or a stall budget `< 2`.
pub fn resolve_probe(
    env: &EnvConfig,
) -> Result<Option<(SocketAddr, vd_node::health::ProbeTuning)>, Box<dyn std::error::Error>> {
    let Ok(addr_str) = env.string("VD_PROBE_ADDR") else {
        return Ok(None);
    };
    if addr_str.trim().is_empty() {
        return Ok(None);
    }
    let addr: SocketAddr = env.parse("VD_PROBE_ADDR")?;
    let tuning = vd_node::health::ProbeTuning {
        stall_deadline_ticks: env.parse_or(
            "VD_PROBE_STALL_TICKS",
            vd_node::health::ProbeTuning::DEFAULT.stall_deadline_ticks,
        )?,
    };
    tuning.validate()?;
    Ok(Some((addr, tuning)))
}

/// RLM RG-4: resolve the admin HTTP bind, or `None` if `VD_ADMIN_ADDR` is unset or empty (every
/// in-process/parity rig stays byte-identical until it opts in; the demand cluster + every cloud pod set it).
/// Twin of [`resolve_probe`] — a different concern (a read-only observability endpoint, no inert/split-brain
/// hazard), meaningful in DevTest too (the demand-login e2e curls the gateway's snapshot).
///
/// # Errors
/// A malformed `VD_ADMIN_ADDR`.
pub fn resolve_admin(env: &EnvConfig) -> Result<Option<SocketAddr>, Box<dyn std::error::Error>> {
    let Ok(addr_str) = env.string("VD_ADMIN_ADDR") else {
        return Ok(None);
    };
    if addr_str.trim().is_empty() {
        return Ok(None);
    }
    Ok(Some(env.parse("VD_ADMIN_ADDR")?))
}

pub fn boot_mesh_and_replay(
    env: &EnvConfig,
    runtime: &tokio::runtime::Handle,
    trust: &vd_io_prod::trust::ClusterTrust,
) -> Result<
    (
        vd_io_prod::mesh::MeshTransport,
        vd_io_prod::mesh::MeshControl,
    ),
    Box<dyn std::error::Error>,
> {
    use vd_io_prod::mesh::{MeshConfig, spawn_mesh};

    let local = env.node_id("VD_NODE_ID")?;
    let peers = env.peer_book("VD_PEERS")?;
    let new_incarnation = resolve_process_incarnation(env)?; // FINDING C: resolve ONCE, thread everywhere

    let shared: Option<vd_io_prod::outbox::SharedOutbox> = open_node_outbox(env)?.map(|ob| {
        std::sync::Arc::new(std::sync::Mutex::new(
            Box::new(ob) as Box<dyn vd_io_prod::outbox::OutboxSink + Send>
        ))
    });

    let (mut transport, control) = spawn_mesh(
        runtime,
        trust,
        &MeshConfig::new(
            local,
            env.parse("VD_BIND")?,
            peers.clone(),
            env.parse("VD_OUTBOUND_CAP")?,
            new_incarnation,
        ),
        shared.clone(),
    )?;

    // Boot replay BEFORE `build_app` consumes the transport (the peer-writer tasks are already live).
    if let Some(sh) = shared.as_ref() {
        let counts = vd_io_prod::outbox::replay_outbox(sh, &mut transport, &peers)?;
        if counts.replayed > 0 || counts.quarantined > 0 {
            tracing::info!(
                replayed = counts.replayed,
                quarantined = counts.quarantined,
                "boot replay: re-drove retained durable outbox rows (quarantined rows RETAINED for next boot)"
            );
        }
    }
    Ok((transport, control))
}

/// Spawn the peer-address AUTO-RESOLVER IFF `VD_PEER_HOSTS` is set — the production caller of
/// [`vd_io_prod::mesh::spawn_peer_resolver`] that re-plumbs a rescheduled BOOKED peer (new pod IP, same DNS
/// name) for INITIATED traffic. The cloud deploy sets `VD_PEER_HOSTS` (docker/entrypoint.sh); every in-process
/// / dev / parity rig leaves it UNSET, so this is a no-op — byte-identical to today. HR3: the ONE resolver
/// wiring shared by every node bin (shard / gateway / orchestrator). The default re-resolve interval
/// ([`vd_io_prod::mesh::DEFAULT_PEER_RERESOLVE_INTERVAL`], 500 ms) is well under the cloud confirmed-dead
/// liveness window, so a rescheduled peer is re-plumbed before the saga declares it dead — coherent by
/// construction (not a tunable, so no boot cross-check is owed).
///
/// # Errors
/// A MALFORMED `VD_PEER_HOSTS`, or a host for a peer NOT booked in `VD_PEERS` (a config drift), fails LOUD;
/// absence is a clean no-op.
pub fn spawn_peer_resolver_if_configured(
    env: &EnvConfig,
    runtime: &tokio::runtime::Handle,
    control: std::sync::Arc<vd_io_prod::mesh::MeshControl>,
    shutdown: std::sync::Arc<std::sync::atomic::AtomicBool>,
) -> Result<(), Box<dyn std::error::Error>> {
    let hosts = match env.peer_hosts("VD_PEER_HOSTS") {
        Ok(hosts) => hosts,
        // Absent ⇒ no auto-resolver (in-process / dev / parity rigs) — byte-identical to today.
        Err(vd_io_prod::runtime::ConfigError::Missing(_)) => return Ok(()),
        // Present but malformed ⇒ fail loud (never a silent mis-parse).
        Err(other) => return Err(Box::new(other)),
    };
    // Drift guard: every re-resolved peer MUST be booked in VD_PEERS — a host for an unbooked peer is a config
    // bug (the auto-resolver only re-plumbs booked peers; it never invents a dial target).
    let booked = env.peer_book("VD_PEERS")?;
    for id in hosts.keys() {
        if !booked.contains_key(id) {
            return Err(format!(
                "VD_PEER_HOSTS lists peer {} which is not in VD_PEERS — the auto-resolver only re-plumbs booked peers",
                id.0
            )
            .into());
        }
    }
    let resolver: std::sync::Arc<dyn vd_io_prod::mesh::AddrResolver> =
        std::sync::Arc::new(vd_io_prod::mesh::SystemDnsResolver);
    vd_io_prod::mesh::spawn_peer_resolver(
        runtime,
        control,
        hosts,
        resolver,
        vd_io_prod::mesh::PeerResolveTuning::default(),
        shutdown,
    );
    Ok(())
}

/// The env every node shares (trust bundle + transport knobs + the per-launch process incarnation).
#[must_use]
pub fn common_env(trust_dir: &str, p: &DevClusterParams) -> Vec<(&'static str, String)> {
    let mut env = vec![
        ("VD_TRUST_DIR", trust_dir.to_owned()),
        str_pair("VD_OUTBOUND_CAP", p.outbound_cap),
        str_pair("VD_TICK_HZ", p.tick_hz),
        str_pair("VD_PROCESS_INCARNATION", launch_incarnation()),
        // Step 5 slice B — every shard of THE (armed) world must carry the hand-off hold budget or
        // refuse to boot; only the shard bin reads the key, and it is deliberately NOT a spawn-anchor
        // harvest key, so a demand orchestrator still hands its children `handoff_hold_anchor`'s
        // value (the same derivation, from its own live tunings).
        static_handoff_hold_env(p),
    ];
    // D-WORLD-2 cure — every STATICALLY launched shard flies the armed crossing re-drive posture.
    // A static cluster hosts a SUBSET of THE world, which is exactly where an unresolved-dest
    // crossing bites (an unhosted realm's head can NEVER resolve), so the static path derives the
    // same pair the demand orchestrator hands its spawned children — from the default saga budget a
    // static cluster runs (its orchestrator env sets no VD_SAGA_* override).
    env.extend(crossing_redrive_env(&vd_sim::saga::SagaTuning::default()));
    env
}

/// THE CROSSING RE-DRIVE PAIR every shard boots with (the D-WORLD-2 cure): the request ttl and the
/// re-drive budget that turn an unresolved-dest crossing drop from a PERMANENT STRAND (latched
/// forever, no reply, no re-emit) into a bounded self-heal — re-drive `budget` times at the `ttl`
/// cadence, then abort locally, clear the latch, and let the entity's next crossing fire. DERIVED on
/// the launcher, never a literal, from the deployment's SAGA deadlines (the only inputs it has):
/// `ttl = abort + POST_COMMIT_STEPS·redrive + 1` (strictly outlasting the worst HEALTHY saga resolve,
/// so a re-drive never races a live-but-slow saga — [`vd_sim::saga::derive_request_ttl_ticks`]) and
/// `budget = abort / redrive` (the saga's own patience ratio —
/// [`vd_sim::saga::derive_crossing_redrive_budget`]). The static launcher feeds the default budget
/// ([`common_env`]); a DEMAND orchestrator feeds its LIVE budget into its spawn anchors, so the two
/// launch modes cannot disagree with the saga tuning they actually run.
#[must_use]
pub fn crossing_redrive_env(saga: &vd_sim::saga::SagaTuning) -> [(&'static str, String); 2] {
    [
        (
            "VD_CROSSING_TTL_TICKS",
            vd_sim::saga::derive_request_ttl_ticks(saga).to_string(),
        ),
        (
            "VD_CROSSING_REDRIVE_BUDGET",
            vd_sim::saga::derive_crossing_redrive_budget(saga).to_string(),
        ),
    ]
}

/// RLM RG-4: the shard-spawn boot anchors a DEMAND orchestrator carries in its OWN env — the 5 must-parse
/// params a forked shard consumes inside `spawn_realm` (a shard refuses to boot without them). The
/// orchestrator's [`spawn_anchors_from_env`] harvest forwards each to every demand-spawned child. In a static
/// cluster these already ride each shard's [`shard_env`]; a demand orchestrator has NO static shard, so it
/// must carry them itself. (`VD_TRUST_DIR`/`VD_TICK_HZ`/`VD_OUTBOUND_CAP` already ride [`common_env`], and the
/// seed + boot horizon ride the demand orchestrator arm directly — see [`orchestrator_env`].)
#[must_use]
pub fn shard_spawn_anchor_env(p: &DevClusterParams) -> Vec<(&'static str, String)> {
    vec![
        str_pair("VD_TICK_DT", p.tick_dt),
        str_pair("VD_SPEED", p.move_speed),
        str_pair("VD_SNAPSHOT_BUDGET", p.snapshot_budget),
        str_pair("VD_MINT_SEED", p.mint_seed),
        str_pair("VD_INPUT_LOG_CAP", p.input_log_cap),
    ]
}

/// THE HAND-OFF BUDGET a STATICALLY launched shard boots with (Step 5 slice B). Since the one-world
/// change every shard boots THE world, whose interest bands are ARMED — and an AoI-armed shard now
/// REFUSES to boot with a zero hold budget (the shard-side boot fence): a source that falls silent
/// about a departing occupant while its parent still counts on it is the gap the hand-off ledger
/// closes, and a fence beats running half a ledger. A DEMAND cluster's shards inherit the value from
/// [`handoff_hold_anchor`]; a static launcher has no orchestrator spawn-env to inherit from, so it
/// states the SAME derivation here — `derive_arrival_shield_ticks` over the armed RLM windows at this
/// cluster's tick rate and the default saga budget, the identical two inputs the demand path reads —
/// so the two launch modes cannot disagree about the one duration a hand-off has.
#[must_use]
pub fn static_handoff_hold_env(p: &DevClusterParams) -> (&'static str, String) {
    let rlm = vd_node::rlm_runtime::resolve_rlm_tuning(true, p.tick_hz, 0, 0);
    let saga = vd_sim::saga::SagaTuning::default();
    (
        "VD_HANDOFF_HOLD_TICKS",
        vd_sim::rlm::derive_arrival_shield_ticks(&rlm, &saga).to_string(),
    )
}

/// RLM RG-4 — the DEMAND orchestrator's env: orchestrator + gateway ONLY. NO static shard is booked (the
/// only shard is demand-spawned at runtime), NO `VD_STATIC_FOREST` and NO `VD_ROSTER` (nothing static to
/// re-home onto), and the reconciler is ARMED (`VD_DEMAND=1`, mutually exclusive with the static forest). It
/// carries every [`shard_spawn_anchor_env`] param + the universe seed + the predictive boot horizon + the RLM
/// port band in its OWN env, so its [`spawn_anchors_from_env`] harvest can boot a demand-spawned shard the
/// gateway then reaches with NO pre-booked address via the reactive greeting (RG-0..3).
fn demand_orchestrator_env(
    a: &ClusterAddrs,
    p: &DevClusterParams,
    store_path: &str,
) -> Vec<(&'static str, String)> {
    let mut env = vec![
        str_pair("VD_NODE_ID", ORCH.0),
        str_pair("VD_BIND", a.orchestrator),
        ("VD_PEERS", book(&[(GATEWAY, a.gateway)])), // NO shard — it does not exist until demand spawns it.
        str_pair("VD_EPOCH", p.epoch),
        str_pair("VD_RESERVE_CHUNK", p.reserve_chunk),
        ("VD_CLOCK_PEERS", GATEWAY.0.to_string()), // drive the universe clock to the gateway (the one booked follower).
        str_pair("VD_LEASE_TTL", p.lease_ttl),
        str_pair("VD_ADMIN_ADDR", a.admin),
        str_pair("VD_PROBE_ADDR", a.orchestrator_probe),
        ("VD_STORE_PATH", store_path.to_owned()),
        ("VD_STORE_EPHEMERAL_OK", "1".to_owned()),
        // ARM the reconciler. NO VD_STATIC_FOREST (the XOR gate the bin fail-loud-checks): an armed sweep atop
        // externally pre-spawned static heads would reap them (they carry no demand cell).
        ("VD_DEMAND", "1".to_owned()),
        str_pair("VD_UNIVERSE_SEED", p.universe_seed),
        // RLM realistic-demo Slice 3: a demand cluster is a VISUAL-DEMAND cluster — the compressed-real
        // 5-planet Kepler geometry with the per-realm AoI band LIVE, so a MOVING occupant's
        // `evaluate_realm_aoi` spins a child realm up as it crosses into visibility (angular size ≥ θ_min) and
        // reaps it as it falls back out (the whole point of the demand loop). `spawn_anchors_from_env`
        // forwards this to every demand-spawned shard (VD_SPEED + VD_TICK_DT — the band's occupant-speed/tick
        // inputs — already ride the anchors). A STATIONARY login (rlm_demand_login) lands at the star (System
        // 7): the inner planets already IN visibility spin up, the outer ones are culled — no spurious demand
        // beyond what is genuinely in view.
        str_pair("VD_BOOT_TICKS_P99", p.boot_ticks_p99),
        str_pair("VD_RLM_FIRST_PORT", RLM_DEMAND_FIRST_PORT),
        str_pair("VD_RLM_PORT_LIMIT", RLM_DEMAND_PORT_LIMIT),
    ];
    // The 5 shard-spawn anchors the orchestrator's harvest forwards to every demand-spawned child.
    env.extend(shard_spawn_anchor_env(p));
    env
}

/// The orchestrator's node-specific env (directory + clock + admin + the D-6 durable Store). `store_path`
/// is the redb file location (REQUIRED by the bin — no in-memory fallback). Dev/test clusters store under a
/// $TMPDIR work dir cleaned by `down`, so this also sets `VD_STORE_EPHEMERAL_OK` to clear the bin's HR1
/// persistent-volume guard (a production deploy builds its env elsewhere with a real volume + no opt-in).
#[must_use]
pub fn orchestrator_env(
    a: &ClusterAddrs,
    p: &DevClusterParams,
    store_path: &str,
    shape: ClusterShape,
) -> Vec<(&'static str, String)> {
    // RLM RG-4: the demand cluster is a wholly different roster (orchestrator + gateway only) — not the static
    // fan-out below, which always books a shard + marks VD_STATIC_FOREST.
    if shape.is_demand() {
        return demand_orchestrator_env(a, p, store_path);
    }
    // ONE data fan-out (HR3): in a MULTI-shard cluster the orchestrator INITIATES realm-grants /
    // re-home to every extra realm-shard, so it books them AND drives the universe clock to each (a
    // follower whose clock never advances can never win its realm lease). Each extra shard is ALSO
    // added to `VD_ROSTER` (the D-37 re-home candidate set; the crossing itself resolves via the
    // directory head, NOT the roster — see the bin note). In Single mode the list is EMPTY — these
    // keys are byte-identical to the pre-Track-R env. The peer/clock/roster lists GROW with the shape
    // (data, no per-kind branch): Dual adds the galaxy; Chain adds galaxy + inner planet + sibling.
    let extra_shards: Vec<(NodeId, SocketAddr)> = realm_shards(shape, a, p)
        .iter()
        .map(|s| (s.node, s.quic))
        .collect();
    let mut peers = vec![(GATEWAY, a.gateway), (SHARD, a.shard)];
    peers.extend_from_slice(&extra_shards);
    let mut clock_ids = vec![GATEWAY.0, SHARD.0];
    clock_ids.extend(extra_shards.iter().map(|(id, _)| id.0));
    let clock_peers = clock_ids
        .iter()
        .map(u64::to_string)
        .collect::<Vec<_>>()
        .join(",");
    let mut env = vec![
        str_pair("VD_NODE_ID", ORCH.0),
        str_pair("VD_BIND", a.orchestrator),
        ("VD_PEERS", book(&peers)),
        str_pair("VD_EPOCH", p.epoch),
        str_pair("VD_RESERVE_CHUNK", p.reserve_chunk),
        ("VD_CLOCK_PEERS", clock_peers),
        str_pair("VD_LEASE_TTL", p.lease_ttl),
        str_pair("VD_ADMIN_ADDR", a.admin),
        str_pair("VD_PROBE_ADDR", a.orchestrator_probe),
        ("VD_STORE_PATH", store_path.to_owned()),
        ("VD_STORE_EPHEMERAL_OK", "1".to_owned()),
        // RLM 5f: every harness shape STATICALLY pre-spawns its shards (each self-grants a realm head the
        // orchestrator did not demand), so an ARMED demand reconciler would reap them. Mark the static-boot
        // mode so the orchestrator's `VD_DEMAND` XOR `VD_STATIC_FOREST` gate fails loud on a misconfig; inert
        // while `VD_DEMAND` is unset (byte-identical). 5f-4's root-chain demand boot omits this marker.
        ("VD_STATIC_FOREST", "1".to_owned()),
    ];
    if !extra_shards.is_empty() {
        // The D-37 re-home candidate SET: every extra shard the orchestrator may re-home an orphan onto.
        // `select_rehome_target` is realm-BLIND (lowest live capable node) — a D-37 concern, NOT the
        // crossing (which resolves `head(Realm(to_realm))`, HR3-clean). For the crossing this roster's only
        // job is that each dest is a known re-home target; the head resolution comes from its realm grant.
        env.push((
            "VD_ROSTER",
            extra_shards
                .iter()
                .map(|(id, _)| id.0.to_string())
                .collect::<Vec<_>>()
                .join(","),
        ));
    }
    env
}

/// RLM RG-4 — the DEMAND gateway's env. Books ONLY the orchestrator + its dev-control clients — NO shard,
/// because the home shard is demand-spawned at runtime and the gateway learns its return connection via the
/// reactive greeting (RG-0..3), never a pre-booked address. `VD_DEMAND=1` arms the gateway's dynamic-home
/// login route (a login emits a `RealmDemand`, waits for the just-spawned home head, and routes to the
/// dynamically-minted node). `VD_SHARD` stays [`SHARD`] as the login-home DEFAULT id — the dynamic-home
/// resolve supersedes it per-login, but the bin still parses the key.
fn demand_gateway_env(
    a: &ClusterAddrs,
    clients: &[(NodeId, SocketAddr)],
    auth_pubkey_hex: &str,
    p: &DevClusterParams,
) -> Vec<(&'static str, String)> {
    let mut peers = vec![(ORCH, a.orchestrator)]; // NO shard booked — the reactive greeting learns it.
    peers.extend_from_slice(clients);
    let mut env = vec![
        str_pair("VD_NODE_ID", GATEWAY.0),
        str_pair("VD_BIND", a.gateway),
        ("VD_PEERS", book(&peers)),
        str_pair("VD_ORCH", ORCH.0),
        str_pair("VD_SHARD", SHARD.0),
        ("VD_AUTH_PUBKEY", auth_pubkey_hex.to_owned()),
        str_pair("VD_SESSION_SEED", p.session_seed),
        str_pair("VD_MAX_SESSIONS", p.max_sessions),
        str_pair("VD_MAX_BUFFERED_INPUTS", p.max_buffered_inputs),
        str_pair("VD_PROBE_ADDR", a.gateway_probe),
        ("VD_DEMAND", "1".to_owned()), // arm the dynamic-home login route + the trusted seed injector.
    ];
    env.extend(world_env(p));
    if let Some(admin) = a.gateway_admin {
        env.push(str_pair("VD_ADMIN_ADDR", admin));
    }
    env
}

/// The gateway's node-specific env. `clients` are seeded into the gateway's peer
/// book so it can route snapshots BACK to each dev-control client (the mesh dials
/// by address book; a missing client entry = the gateway can never reach it).
/// The inputs a node needs to BUILD THE WORLD — required by any node that answers a spatial question.
///
/// A shard has carried these forever (it integrates motion with them). The gateway needs them too and did
/// not have them: it decides which realm a login lands in, and it was doing that against a hardcoded
/// walk-scale world while the shards ran whatever `VD_UNIVERSE_SCALE` said — the login side and the
/// simulating side describing different universes from the same seed. Shared here so the two node kinds
/// cannot be handed different numbers by being edited in different places.
///
/// (`VD_UNIVERSE_SCALE` is deliberately absent: it is exported cluster-wide and inherited by every spawned
/// node, so listing it here would make one value have two sources.)
#[must_use]
fn world_env(p: &DevClusterParams) -> [(&'static str, String); 2] {
    [
        str_pair("VD_TICK_DT", p.tick_dt),
        str_pair("VD_SPEED", p.move_speed),
    ]
}

#[must_use]
pub fn gateway_env(
    a: &ClusterAddrs,
    clients: &[(NodeId, SocketAddr)],
    auth_pubkey_hex: &str,
    p: &DevClusterParams,
    shape: ClusterShape,
) -> Vec<(&'static str, String)> {
    // RLM RG-4: the demand gateway books no static shard — a wholly different roster from the static fan-out.
    if shape.is_demand() {
        return demand_gateway_env(a, clients, auth_pubkey_hex, p);
    }
    // ONE data fan-out (HR3): in a MULTI-shard cluster the gateway must BOOK every extra realm-shard
    // (to route a transferred client's inputs / cut-drains onto it — the durable session-route swap at
    // `CommitAuthority` migrates the route to each successive source shard, so ALL of them must be
    // dialable) AND class each as a KNOWN shard (`VD_KNOWN_SHARDS`, consumed by the gateway bin's
    // `known_shards` set) so a shard→gateway frame reaches `on_shard_frame` instead of dropping as an
    // unknown peer. `VD_SHARD` (the LOGIN shard) stays SHARD in every mode. In Single mode neither an
    // extra peer nor `VD_KNOWN_SHARDS` is emitted — byte-identical to the pre-Track-R env. The lists
    // GROW with the shape (data, no per-kind branch): Dual adds the galaxy; Chain adds galaxy + inner
    // planet + sibling.
    let extra_shards: Vec<(NodeId, SocketAddr)> = realm_shards(shape, a, p)
        .iter()
        .map(|s| (s.node, s.quic))
        .collect();
    let mut peers = vec![(ORCH, a.orchestrator), (SHARD, a.shard)];
    peers.extend_from_slice(&extra_shards);
    peers.extend_from_slice(clients);
    let mut env = vec![
        str_pair("VD_NODE_ID", GATEWAY.0),
        str_pair("VD_BIND", a.gateway),
        ("VD_PEERS", book(&peers)),
        str_pair("VD_ORCH", ORCH.0),
        str_pair("VD_SHARD", SHARD.0),
        ("VD_AUTH_PUBKEY", auth_pubkey_hex.to_owned()),
        str_pair("VD_SESSION_SEED", p.session_seed),
        str_pair("VD_MAX_SESSIONS", p.max_sessions),
        str_pair("VD_MAX_BUFFERED_INPUTS", p.max_buffered_inputs),
        str_pair("VD_PROBE_ADDR", a.gateway_probe),
    ];
    env.extend(world_env(p));
    // RLM RG-4: the gateway publishes its /admin/snapshot (GatewayView) ONLY where the cluster booked an
    // admin bind (the Demand cluster + cloud pods). `None` ⇒ no VD_ADMIN_ADDR ⇒ byte-identical.
    if let Some(admin) = a.gateway_admin {
        env.push(str_pair("VD_ADMIN_ADDR", admin));
    }
    if !extra_shards.is_empty() {
        // The gateway bin unions this into `known_shards` over the login `shard` (via `node_list`), so the
        // routable-shard roster is {SHARD, DEST[, GALAXY]} — every shard's frames are node-class dispatchable.
        env.push((
            "VD_KNOWN_SHARDS",
            extra_shards
                .iter()
                .map(|(id, _)| id.0.to_string())
                .collect::<Vec<_>>()
                .join(","),
        ));
    }
    env
}

/// The HOME stub-shard's node-specific env (realm + movement + snapshot budget). In a multi-shard
/// shape it also BOOKS every extra realm-shard so the cross-shard mesh can carry transfer traffic
/// (each shard books the others). In Single mode the peer book is byte-identical to the base `up`.
/// The home shard boots THE world's seed neighbourhood — its own 150 m shell IS the crossing
/// boundary; nothing is injected.
#[must_use]
pub fn shard_env(
    a: &ClusterAddrs,
    p: &DevClusterParams,
    shape: ClusterShape,
) -> Vec<(&'static str, String)> {
    // The home shard books every OTHER realm-shard so the cross-shard mesh can carry transfer traffic
    // (each shard books the others). The list GROWS with the shape (ONE data fan-out, no per-kind
    // branch): Dual books the galaxy; Chain books galaxy + inner planet + sibling. In Single mode the
    // book is byte-identical to the pre-Track-R env.
    let mut peers = vec![(ORCH, a.orchestrator), (GATEWAY, a.gateway)];
    peers.extend(realm_shards(shape, a, p).iter().map(|s| (s.node, s.quic)));
    let env = vec![
        str_pair("VD_NODE_ID", SHARD.0),
        str_pair("VD_BIND", a.shard),
        ("VD_PEERS", book(&peers)),
        // No `VD_REALM_KIND`: the System-7 source shard is a System realm, which is the shard bin's
        // ABSENT-default (`realm_from_kind_seed("", seed) == System(seed)`) — so the env stays BYTE-IDENTICAL
        // to the pre-NODE-PER-REALM shard env (the process_parity/inert-parity gate), and the Forest's extra
        // realm-shards carry their non-System KIND via `realm_shard_env`.
        str_pair("VD_REALM_SEED", p.realm_seed),
        str_pair("VD_SPEED", p.move_speed),
        str_pair("VD_TICK_DT", p.tick_dt),
        str_pair("VD_ORCH", ORCH.0),
        str_pair("VD_MINT_SEED", p.mint_seed),
        str_pair("VD_INPUT_LOG_CAP", p.input_log_cap),
        str_pair("VD_REALM_RECHECK", p.realm_recheck),
        str_pair("VD_SNAPSHOT_BUDGET", p.snapshot_budget),
        str_pair("VD_PROBE_ADDR", a.shard_probe),
    ];
    // NO `VD_HELD_REALMS` in ANY shape: co-hosting retired with the --triple shape (NODE-PER-REALM —
    // every realm its own shard, every re-home a uniform CROSS-NODE saga). The shard bin keeps the
    // parse (a shape with no consumer is exactly what rotted into the CRITICAL — ledgered D-WORLD-6).
    env
}

/// The `VD_REALM_KIND` env token for a realm KIND (`system` | `planet` | `station` | `area` | `ship`). The
/// shard bin ([`crate`]'s `shard.rs`) reads this beside `VD_REALM_SEED` to build its `own_realm` of the
/// right KIND (NODE-PER-REALM) — before this a shard could only host `System(seed)`. Monomorphic (the
/// `RealmId` arm match covered here), reusing the `VD_HELD_REALMS` `realm_token` kind vocabulary; the seed
/// is irrelevant to the KIND token, so any seed of the wanted kind maps to the right word.
#[must_use]
pub fn realm_kind_token(realm: vd_core::pose::RealmId) -> &'static str {
    use vd_core::pose::RealmId;
    match realm {
        RealmId::System(_) => "system",
        RealmId::Planet(_) => "planet",
        RealmId::Station(_) => "station",
        RealmId::Area(_) => "area",
        RealmId::Ship(_) => "ship",
    }
}

/// Parse a `VD_REALM_KIND` token + a `VD_REALM_SEED` into the shard's `own_realm` (the inverse of
/// [`realm_kind_token`]). A shard hosts EXACTLY ONE realm; its KIND is this token, its seed is `VD_REALM_SEED`.
/// ABSENT/empty token ⇒ `System(seed)` (the pre-NODE-PER-REALM default, so every legacy shard boot is
/// byte-identical). A `ship` kind is rejected here (a Ship realm keys on an entity id, not a seed — P8, never a
/// booted realm-shard).
///
/// # Errors
/// An unrecognized kind token.
pub fn realm_from_kind_seed(kind: &str, seed: u64) -> Result<vd_core::pose::RealmId, String> {
    use vd_core::pose::RealmId;
    match kind.trim() {
        "" | "system" => Ok(RealmId::System(seed)),
        "planet" => Ok(RealmId::Planet(seed)),
        "station" => Ok(RealmId::Station(seed)),
        "area" => Ok(RealmId::Area(seed)),
        other => Err(format!(
            "VD_REALM_KIND {other:?} is not one of system|planet|station|area (a realm-shard hosts one \
             seed-keyed realm; Ship realms key on an entity id and are never booted as a realm-shard)"
        )),
    }
}

/// THE node-specific env for ONE extra realm-shard (galaxy / inner planet / sibling star), each
/// hosting EXACTLY its own realm (NO `VD_HELD_REALMS`) — the ONE builder every extra shard boots
/// through (HR3: one tooling, never a per-realm builder). Twin of [`shard_env`]: it books ORCH +
/// GATEWAY + the home shard + every OTHER realm-shard of the shape (each shard books the others so
/// the cross-shard mesh carries a re-home between any pair), and emits `VD_REALM_KIND` +
/// `VD_REALM_SEED` so the shard bin boots the right realm KIND. A DISTINCT mint per node (derived
/// from the home mint) so no two shards alias entity ids. NO boundary-file injection — the SEED
/// neighbourhood boots (`realm_neighbourhood_for(own)` = own + ancestors + DIRECT children), which is
/// how the containment detector fires each cross-node re-home.
#[must_use]
pub fn realm_shard_env(
    a: &ClusterAddrs,
    p: &DevClusterParams,
    shape: ClusterShape,
    shard: RealmShard,
) -> Vec<(&'static str, String)> {
    let mut peers = vec![
        (ORCH, a.orchestrator),
        (GATEWAY, a.gateway),
        (SHARD, a.shard),
    ];
    // Book every OTHER extra realm-shard (skip self) so each shard can mesh a transfer to any other.
    peers.extend(
        realm_shards(shape, a, p)
            .iter()
            .filter(|s| s.node != shard.node)
            .map(|s| (s.node, s.quic)),
    );
    // A distinct mint per node so no two shards ever alias an entity id. The galaxy/sibling shards
    // keep their HISTORIC offsets (`+12`/`+6`, matching the harness galaxy=23 / dest=17 vs home=11);
    // any other realm-shard derives from its node id above a BASE (`+100`) that clears the small
    // historic offsets — so the planet's node-id 6 never collides with the sibling's `+6`. All mints
    // pairwise distinct (asserted in `every_shape_names_only_realms_of_the_world`).
    const REALM_SHARD_MINT_BASE: u64 = 100;
    let mint = match shard.node {
        GALAXY => p.mint_seed.wrapping_add(12),
        SHARD_B => p.mint_seed.wrapping_add(6),
        other => p
            .mint_seed
            .wrapping_add(REALM_SHARD_MINT_BASE)
            .wrapping_add(other.0),
    };
    vec![
        str_pair("VD_NODE_ID", shard.node.0),
        str_pair("VD_BIND", shard.quic),
        ("VD_PEERS", book(&peers)),
        str_pair("VD_REALM_KIND", realm_kind_token(shard.realm)),
        str_pair("VD_REALM_SEED", realm_seed_of(shard.realm)),
        str_pair("VD_SPEED", p.move_speed),
        str_pair("VD_TICK_DT", p.tick_dt),
        str_pair("VD_ORCH", ORCH.0),
        str_pair("VD_MINT_SEED", mint),
        str_pair("VD_INPUT_LOG_CAP", p.input_log_cap),
        str_pair("VD_REALM_RECHECK", p.realm_recheck),
        str_pair("VD_SNAPSHOT_BUDGET", p.snapshot_budget),
        str_pair("VD_PROBE_ADDR", shard.probe),
    ]
}

/// The seed a seed-keyed realm hosts (`VD_REALM_SEED`). A `Ship` realm has no seed (it keys on an entity id);
/// it is never a booted realm-shard, so it maps to 0 for token totality only.
fn realm_seed_of(realm: vd_core::pose::RealmId) -> u64 {
    use vd_core::pose::RealmId;
    match realm {
        RealmId::System(s) | RealmId::Planet(s) | RealmId::Station(s) | RealmId::Area(s) => s,
        RealmId::Ship(_) => 0,
    }
}

// THE AUTHORED PLAYGROUND IS GONE (SL5) — the born-inside source shell, the two-box crossing
// playground, the seed-forest and visual-scale scene emitters, and the boundary-file override that
// planted them. Each was a second world a cluster could stand up in place of THE one. The LAST
// emitter (`write_world_regions` / `vd-devcluster emit-world-scene`, the `--realm-boxes` file) is
// DELETED too (Slice C1, window_lane.md §2.11 — D-LANE-6 🟩, owner decision 10 THE DRAW LAW): the
// client draws its world from the COMPOSED STREAM alone. One world, one source — the stream.

// ---- shell-safe value quoting ------------------------------------------------

/// POSIX-single-quote a value for safe emission into a `KEY=VALUE` line that a
/// shell will `eval`/source. Wraps in `'…'` and renders each embedded `'` as
/// `'\''`, so a path with spaces or shell metacharacters (e.g. a `$TMPDIR`
/// containing `$(…)`) can never word-split or inject.
#[must_use]
pub fn sh_quote(value: &str) -> String {
    let mut out = String::with_capacity(value.len() + 2);
    out.push('\'');
    for ch in value.chars() {
        if ch == '\'' {
            out.push_str("'\\''");
        } else {
            out.push(ch);
        }
    }
    out.push('\'');
    out
}

// ---- shared process / net glue ----------------------------------------------

/// `127.0.0.1:port` — the ONE loopback-address constructor for the dev tooling (was
/// hand-inlined in every bin and test).
#[must_use]
pub fn loopback(port: u16) -> SocketAddr {
    SocketAddr::from(([127, 0, 0, 1], port))
}

/// Reserve an ephemeral loopback UDP address (bind `:0`, read it back, drop) — the
/// QUIC bind addr a node/client will reuse. Inherently TOCTOU-racy, fine for the
/// local test/dev tiers. (One definition; was `reserve_addr`/`reserve_udp`.)
#[must_use]
pub fn reserve_udp_addr() -> SocketAddr {
    std::net::UdpSocket::bind("127.0.0.1:0")
        .expect("reserve udp")
        .local_addr()
        .expect("addr")
}

/// Reserve an ephemeral loopback TCP address (for an admin / dev-control port).
#[must_use]
pub fn reserve_tcp_addr() -> SocketAddr {
    std::net::TcpListener::bind("127.0.0.1:0")
        .expect("reserve tcp")
        .local_addr()
        .expect("addr")
}

// ---- the dev-control round-trip ------------------------------------------------

/// Read budget for one dev-control reply. Generous: the longest legitimate server-side
/// stall is a `wait-until`/`screenshot --at-tick` bounded by its own tick budget, and a
/// `record` paces up to 3600 frames; past this the CLIENT is wedged and the caller should
/// fail loud rather than hang forever.
pub const DEVCTL_READ_TIMEOUT: Duration = Duration::from_secs(300);

/// ONE dev-control JSON-line round-trip (connect → one request line → one response line)
/// — the SINGLE definition of the wire framing the whole HR6 loop depends on, shared by
/// `vdctl`, the SCALE-1 load test, and the G-RENDER-SMOKE gate (was three hand copies
/// that could drift). A fresh connection per request, like `vdctl`. The read is bounded
/// by [`DEVCTL_READ_TIMEOUT`] so a wedged client can never hang the caller.
///
/// # Errors
/// A connect/IO failure, an empty reply (the client closed the connection), or an
/// undecodable response — each as a human-readable string.
pub fn dev_roundtrip(port: u16, request: &DevRequest) -> Result<DevResponse, String> {
    use std::io::{BufRead, BufReader, Write};
    let addr = loopback(port);
    let stream = std::net::TcpStream::connect(addr).map_err(|e| format!("connect {addr}: {e}"))?;
    stream
        .set_read_timeout(Some(DEVCTL_READ_TIMEOUT))
        .map_err(|e| format!("set read timeout: {e}"))?;
    let mut writer = stream
        .try_clone()
        .map_err(|e| format!("clone stream: {e}"))?;
    let mut line = serde_json::to_string(request).map_err(|e| e.to_string())?;
    line.push('\n');
    writer
        .write_all(line.as_bytes())
        .map_err(|e| format!("write: {e}"))?;
    writer.flush().ok();

    let mut reply = String::new();
    BufReader::new(stream)
        .read_line(&mut reply)
        .map_err(|e| format!("read: {e}"))?;
    if reply.trim().is_empty() {
        return Err("no response (client closed the connection)".to_owned());
    }
    serde_json::from_str(reply.trim()).map_err(|e| format!("decode response: {e}"))
}

// ---- the dev-cluster on-disk layout (ONE definition) ---------------------------

/// The runfile's filename inside a slot workdir — the atomic claim AND the durable kill
/// record (every PID in it is reaped by `down`, TERM→KILL by process group).
pub const RUNFILE_NAME: &str = "cluster.pids";
/// The mTLS trust-bundle dir name inside a slot workdir (generated by `up`).
pub const TRUST_DIR_NAME: &str = "trust";
/// The orchestrator's durable redb Store filename inside a slot workdir (D-6). Lives in the work dir so
/// `down`'s `remove_dir_all(work)` reaps it with the slot — ONE layout definition, shared by the launcher.
pub const ORCH_STORE_NAME: &str = "orchestrator.redb";
/// RLM Step 5e: the realm spawner's launch-ledger redb, a SIBLING of [`ORCH_STORE_NAME`] in the SAME dir
/// (so `down` reaps it too). A SEPARATE file/writer from the saga WAL — it isolates the write-ahead-
/// before-fork durability barrier and decouples launch fsyncs from the per-tick universe-clock barrier
/// (zero edits to the depth-1 D-6 writer core).
pub const LAUNCH_STORE_NAME: &str = "launch.redb";

/// A slot's working directory (`$TMPDIR/vd-devcluster/slot-N`) — the SINGLE definition of
/// the launcher's on-disk layout (was re-derived inline by the launcher AND each process
/// smoke test, a silent-drift risk on the exact contract leak-freedom depends on).
#[must_use]
pub fn slot_workdir(slot: u16) -> PathBuf {
    std::env::temp_dir()
        .join("vd-devcluster")
        .join(format!("slot-{slot}"))
}

/// The slot's mTLS trust bundle dir (clients read it via the env contract or, in tests,
/// directly from here).
#[must_use]
pub fn slot_trust_dir(slot: u16) -> PathBuf {
    slot_workdir(slot).join(TRUST_DIR_NAME)
}

/// The slot's runfile (see [`RUNFILE_NAME`]).
#[must_use]
pub fn slot_runfile(slot: u16) -> PathBuf {
    slot_workdir(slot).join(RUNFILE_NAME)
}

/// Append an EXTRA process to a slot's kill record — for a test-spawned process that must
/// not outlive the slot (e.g. the G-RENDER-SMOKE capture client: its in-test kill guard
/// dies with a SIGKILL of the test runner, but the runfile survives, so the next
/// pre-clean `down` reaps the orphan instead of leaving it to poison the slot's ports).
/// The process MUST be spawned in its own process group (`process_group(0)`), matching
/// the launcher's nodes — `down` signals by group id.
///
/// # Errors
/// If the runfile cannot be opened/written (the slot is not up).
pub fn record_extra_pid(slot: u16, pid: u32) -> Result<(), String> {
    use std::io::Write;
    let path = slot_runfile(slot);
    let mut file = std::fs::OpenOptions::new()
        .append(true)
        .open(&path)
        .map_err(|e| format!("open runfile {}: {e}", path.display()))?;
    writeln!(file, "{pid}").map_err(|e| format!("append pid: {e}"))
}

// ---- test-reserved slots + the cluster-down guard -------------------------------

/// Test-reserved slots ABOVE the worktree auto-derivation ceiling, so a process-tier test
/// can never collide with a developer's live `dev-cluster.sh up` (whose slot is always
/// `< WORKTREE_SLOT_CEILING`) — and each test owns a DISTINCT slot so the suites can
/// never collide with each other. ONE registry (was per-test consts that could silently
/// overlap).
pub const SMOKE_SLOT: u16 = WORKTREE_SLOT_CEILING + 16; // 80: dev_cluster_smoke up/down
pub const RECOVERY_SLOT: u16 = WORKTREE_SLOT_CEILING + 17; // 81: dev_cluster_smoke recovery
pub const RENDER_SMOKE_SLOT: u16 = WORKTREE_SLOT_CEILING + 18; // 82: G-RENDER-SMOKE
/// Track R / 1d.2: the dual-shard crossing smoke's slot (distinct from every other test slot).
pub const CROSSING_SLOT: u16 = WORKTREE_SLOT_CEILING + 19; // 83: dual_cluster_crossing_smoke
/// RLM demand-walk (VU): the `up --demand` launcher smoke's slot.
pub const DEMAND_SMOKE_SLOT: u16 = WORKTREE_SLOT_CEILING + 20; // 84: demand_cluster_smoke

/// Run one `vd-devcluster` subcommand against a slot (the launcher binary path comes from
/// the calling test's `env!("CARGO_BIN_EXE_vd-devcluster")`).
///
/// # Panics
/// If the launcher cannot be spawned at all (a build/setup failure, not a test outcome).
pub fn devcluster(launcher: &str, sub: &str, slot: u16) -> ExitStatus {
    Command::new(launcher)
        .args([sub, "--slot", &slot.to_string()])
        .status()
        .unwrap_or_else(|e| panic!("run vd-devcluster {sub}: {e}"))
}

/// Tear a test's dev cluster down on drop — even if an assertion panics, the slot's
/// processes are reaped and its workdir removed (the leak-freedom every process-tier
/// test stands on; was hand-duplicated per test).
pub struct DevClusterDown {
    launcher: String,
    slot: u16,
}

impl DevClusterDown {
    #[must_use]
    pub fn new(launcher: &str, slot: u16) -> DevClusterDown {
        DevClusterDown {
            launcher: launcher.to_owned(),
            slot,
        }
    }
}

impl Drop for DevClusterDown {
    fn drop(&mut self) {
        let _ = devcluster(&self.launcher, "down", self.slot);
    }
}

/// Spawn one node binary with the shared `common` env merged over its node-specific
/// env — the exact merge every in-process cluster bring-up needs, once. (The
/// `vd-devcluster` launcher wraps this with process-group + log redirection it alone
/// requires; the parity/load tests use this plain form.)
///
/// TEST SUPPORT ONLY — every caller is a process-tier test under `crates/bins/tests/`; the launcher
/// binary has its own wrapper over [`spawn_node_grouped`]. That is why this may assert the process
/// tier: see [`cluster_tier`].
///
/// # Panics
/// If the calling thread does not hold the process tier ([`cluster_tier`]) — forking a real node
/// binary concurrently with another cluster is what makes the gate lie, so it fails LOUD with the
/// remedy rather than flaking later. This is a tripwire, not the mechanism: it cannot see a test
/// that shells out to the launcher or goes through [`spawn_node_grouped`], which is why the
/// source-scanning test in this crate covers those bypass classes too.
///
/// # Errors
/// Propagates the OS spawn error (e.g. the binary is missing).
pub fn spawn_node(
    bin: &str,
    common: &[(&'static str, String)],
    node_env: &[(&'static str, String)],
) -> std::io::Result<Child> {
    assert!(tier_held_by_current_thread(), "{TIER_REMEDY}");
    let mut cmd = Command::new(bin);
    for (k, v) in common.iter().chain(node_env.iter()) {
        cmd.env(k, v);
    }
    cmd.spawn()
}

// ---- shared launcher primitives (RLM 5c DRY lift) ----------------------------
// These four were private to the `vd-devcluster` launcher; the RLM real-process realm spawner
// (`proc_launch::ProcLaunchBackend`) needs the IDENTICAL launch/kill/liveness primitives, so they live
// here as ONE path both share (HR3/DRY). No libc — process-group signalling is via the POSIX `kill` tool
// (RLM Step-5 OQ-3). The plain [`spawn_node`] above stays byte-identical for its 12+ process-tier callers.

/// Resolve a sibling binary (e.g. `vd-shard`) next to THIS process's executable in the cargo target dir.
/// Tries the exe's own dir first (the deploy/launcher layout: bins beside each other in `target/debug`),
/// then its parent (the `cargo test` layout: a test binary lives in `target/debug/deps`, one level BELOW
/// the real bins) — so the SAME resolver serves the launcher AND a process-tier test that forks a bin.
///
/// # Errors
/// The binary is absent in both candidate dirs (a build that never produced it).
pub fn sibling_binary(name: &str) -> Result<PathBuf, String> {
    let exe = std::env::current_exe().map_err(|e| format!("current_exe: {e}"))?;
    let dir = exe
        .parent()
        .ok_or("launcher has no parent dir")?
        .to_path_buf();
    let beside = dir.join(name);
    if beside.exists() {
        return Ok(beside);
    }
    // `cargo test` fallback: the bins are in the parent of `deps/`.
    if let Some(up) = dir.parent() {
        let up_candidate = up.join(name);
        if up_candidate.exists() {
            return Ok(up_candidate);
        }
    }
    Err(format!(
        "{name} not found next to the launcher ({}) or its parent; run `cargo build` first",
        beside.display()
    ))
}

/// Send `sig` to the process GROUP led by `pid` (the NEGATIVE `-pid` target) via the POSIX `kill` tool.
/// Best-effort; a "No such process" on an already-dead group is expected and silenced.
pub fn signal_group(pid: u32, sig: &str) {
    let _ = Command::new("kill")
        .arg(format!("-{sig}"))
        .arg(format!("-{pid}"))
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null())
        .status();
}

/// Send `sig` to the SINGLE process `pid` (the POSITIVE target) via the POSIX `kill` tool — the
/// one-process twin of [`signal_group`], for a child spawned WITHOUT its own process group.
/// Best-effort; a "No such process" on an already-exited child is expected and silenced.
pub fn signal_pid(pid: u32, sig: &str) {
    let _ = Command::new("kill")
        .arg(format!("-{sig}"))
        .arg(pid.to_string())
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null())
        .status();
}

/// Is `pid` still a live process? (`kill -0` is the POSIX existence probe.)
///
/// NOTE (RLM 5e): this is PID-REUSE-UNSAFE for an ADOPTED orphan after an orchestrator restart — a
/// recycled pid answers "alive". Owned-slot liveness uses [`std::process::Child::try_wait`] (a real
/// `waitpid` on a handle we own); the orphan path (5e) gates `pid_alive` behind an incarnation-cookie
/// probe, never `kill -0` alone.
#[must_use]
pub fn pid_alive(pid: u32) -> bool {
    Command::new("kill")
        .arg("-0")
        .arg(pid.to_string())
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null())
        .status()
        .map(|s| s.success())
        .unwrap_or(false)
}

/// Reopen a `launch.redb` read-only and decode the durable RLM launch rows to `(node, coord, pid)` — reading
/// back through the SAME frozen postcard shape the spawner wrote. DROPS the store (joins its writer, releases
/// the redb process-exclusive lock) before returning, so a subsequent boot can open it. The caller MUST ensure
/// NO orchestrator holds the file (kill + reap first). Shared by the kill-9 crash gate + the demand-login e2e
/// (RLM RG-4) — both reap the shards the ORCHESTRATOR forked, which its own process-group reap never reaches.
///
/// # Panics
/// If `launch.redb` cannot be reopened or a row fails to decode (a corrupt durable ledger is a hard test
/// failure, not a tolerated state).
#[must_use]
pub fn launch_rows(
    path: &std::path::Path,
) -> Vec<(NodeId, vd_core::realm_coord::RealmCoord, Option<u32>)> {
    use vd_sim::io::Store as _;
    let (store, _durability) =
        vd_io_prod::store::RedbStore::open(path, vd_io_prod::store::StoreTuning::default())
            .expect("reopen launch.redb");
    let rows: Vec<(NodeId, vd_core::realm_coord::RealmCoord, Option<u32>)> = store
        .scan(&vd_node::saga_runtime::rlm_launch_prefix())
        .into_iter()
        .map(|(_, v)| {
            let i: vd_node::rlm_spawn::LaunchIntent =
                postcard::from_bytes(&v).expect("decode LaunchIntent");
            (i.node, i.coord, i.pid)
        })
        .collect();
    drop(store);
    rows
}

/// SIGKILL + poll-until-gone every forked shard named by a durable launch row's confirmed pid (a row with no
/// pid — its v2 confirm not yet durable — has no reapable handle here). Shared by the kill-9 crash gate + the
/// demand-login e2e (RLM RG-4) to clean up the demand-spawned shards the orchestrator forked.
pub fn reap_forked(rows: &[(NodeId, vd_core::realm_coord::RealmCoord, Option<u32>)]) {
    for (_, _, pid) in rows {
        if let Some(pid) = pid {
            signal_group(*pid, "KILL");
            let start = std::time::Instant::now();
            while start.elapsed() < std::time::Duration::from_secs(5) && pid_alive(*pid) {
                std::thread::sleep(std::time::Duration::from_millis(50));
            }
        }
    }
}

/// Spawn one node binary with `common` env merged over `node_env`, redirecting stdout+stderr to `log`,
/// and (unix) making the child LEAD ITS OWN PROCESS GROUP (`process_group(0)` ⇒ child pid == pgid) so a
/// later [`signal_group`] reaches the whole subtree and a recycled bare pid can't be hit by accident.
/// `exe` is a resolved path (via [`sibling_binary`]); `log` is a PARAMETER (each caller names its own
/// per-node log file). The grouped+logged spawn the dev-cluster launcher AND the realm spawner share.
///
/// # Errors
/// Propagates the OS spawn error (e.g. the binary is missing) or a `log` handle-clone failure.
pub fn spawn_node_grouped(
    exe: &std::path::Path,
    common: &[(&'static str, String)],
    node_env: &[(&'static str, String)],
    log: std::fs::File,
) -> std::io::Result<Child> {
    let err = log.try_clone()?;
    let mut cmd = Command::new(exe);
    for (k, v) in common.iter().chain(node_env.iter()) {
        cmd.env(k, v);
    }
    cmd.stdout(log).stderr(err);
    #[cfg(unix)]
    {
        use std::os::unix::process::CommandExt;
        cmd.process_group(0);
    }
    cmd.spawn()
}

/// One blocking HTTP/1.1 GET of an admin path — the SINGLE place the request framing
/// lives (was hand-inlined in the launcher + two tests). Returns the response BODY on
/// a `200`, else `None` (unreachable / non-200 / malformed). `read_timeout` bounds the
/// read so a wedged endpoint can't hang the launcher's readiness poll.
#[must_use]
pub fn admin_get_body(
    addr: SocketAddr,
    path: &str,
    read_timeout: Option<Duration>,
) -> Option<String> {
    use std::io::{Read, Write};
    let mut stream = std::net::TcpStream::connect(addr).ok()?;
    if let Some(timeout) = read_timeout {
        stream.set_read_timeout(Some(timeout)).ok()?;
    }
    let request = format!("GET {path} HTTP/1.1\r\nHost: localhost\r\nConnection: close\r\n\r\n");
    stream.write_all(request.as_bytes()).ok()?;
    let mut response = String::new();
    stream.read_to_string(&mut response).ok()?;
    if !response.starts_with("HTTP/1.1 200") {
        return None;
    }
    response
        .split_once("\r\n\r\n")
        .map(|(_, body)| body.to_owned())
}

/// S3: the graceful-drain LINGER (`VD_SHUTDOWN_LINGER_MS`, default 0 = no linger). After SIGTERM de-routes
/// `/readyz` (503) on the shutdown edge, the bin sleeps this long in Terminating BEFORE its final drain +
/// exit, so k8s removes the pod from the Service endpoints and any in-flight requests finish (connection
/// draining) — the standard `preStop`-sleep pattern, here as an app-side knob so the de-route is observable
/// AND traffic stops before the process goes away. Must stay well below `terminationGracePeriodSeconds`.
///
/// # Errors
/// A malformed `VD_SHUTDOWN_LINGER_MS`.
pub fn resolve_shutdown_linger(env: &EnvConfig) -> Result<Duration, Box<dyn std::error::Error>> {
    Ok(Duration::from_millis(
        env.parse_or("VD_SHUTDOWN_LINGER_MS", 0)?,
    ))
}

/// One blocking HTTP/1.1 GET returning the numeric status code (`Some(200)` / `Some(503)` / `Some(404)`), or
/// `None` on a connect/IO failure (a not-yet-bound or refused probe port). S3 needs this because
/// [`admin_get_body`] collapses every non-200 to `None` and so cannot distinguish a Ready `/readyz` (200) from
/// a NotReady one (503). ONE shared status-parsing client (HR3), used by the probe process test.
#[must_use]
pub fn http_get_status(
    addr: SocketAddr,
    path: &str,
    read_timeout: Option<Duration>,
) -> Option<u16> {
    use std::io::{Read, Write};
    let mut stream = std::net::TcpStream::connect(addr).ok()?;
    if let Some(timeout) = read_timeout {
        stream.set_read_timeout(Some(timeout)).ok()?;
    }
    let request = format!("GET {path} HTTP/1.1\r\nHost: localhost\r\nConnection: close\r\n\r\n");
    stream.write_all(request.as_bytes()).ok()?;
    let mut response = String::new();
    stream.read_to_string(&mut response).ok()?;
    // "HTTP/1.1 200 OK" → 200
    response
        .lines()
        .next()?
        .split_whitespace()
        .nth(1)?
        .parse()
        .ok()
}

// THE SCALE KNOB IS GONE — the enum, its resolver, and its parser. It was read from the environment,
// and a live cluster was measured holding TWO of its values at once: the orchestrator on one world, its
// own gateway on another, in a single launch. The owner's ruling, made twice, is one world all the time.
// Nothing selects a universe any more, so nothing can select a different one. The shard's boundary-file
// override (the same knob class: a file that REPLACED the world's geometry at one process while its
// neighbours booted the real one) is gone with it — its error enum, resolver, parser, in-realm guard,
// and the containment-band builder that served only it (D-WORLD-4/4b).

/// The env keys a DEMAND-SPAWNED shard inherits from the orchestrator's own env (RLM 5f). ONE list so the
/// spawn-anchor set is single-sourced + unit-testable. Each is forwarded ONLY if present + non-empty (see
/// [`spawn_anchors_from_env`]), so an absent key ⇒ byte-identical child env; the whole set is unused while
/// the reconciler is inert (`spawn_realm` never runs).
#[must_use]
pub fn spawn_anchor_keys() -> &'static [&'static str] {
    &[
        "VD_TRUST_DIR",
        "VD_TICK_HZ",
        "VD_TICK_DT",
        "VD_SPEED",
        "VD_SNAPSHOT_BUDGET",
        "VD_UNIVERSE_SEED",
        "VD_OUTBOUND_CAP",
        // A forked shard's must-parse boot params (consumed only inside `spawn_realm`).
        "VD_MINT_SEED",
        "VD_INPUT_LOG_CAP",
        // RLM 5f-4: a demand-spawned shard needs these LIVE too — WITHOUT VD_BOOT_TICKS_P99 its predictive
        // AoI horizon is 0 (the walk-in predictive spin-up is silently dead); WITHOUT
        // VD_LEASE_RENEW_INTERVAL its realm lease is never renewed (liveness inert). The self-fence RECHECK
        // is deliberately NOT forwarded — the shard's `resolve_node_d3` self-derives a sane active default
        // from the `n` key (there is no `VD_REALM_RECHECK` read; a spawned shard defaults to hz/2).
        //
        // `VD_SPAWN_POSES` USED TO RIDE HERE and no longer does. It holds UNIVERSE-ABSOLUTE positions, and
        // a shard has no way to read one: turning it into "3 m from this planet's centre" means subtracting
        // the placements of every realm above, which is knowing where you yourself sit. The shard used to
        // load it and relabel, planting players at their star. The gateway reads it, converts once at
        // login, and hands the shard a pose already measured in the shard's own frame.
        "VD_BOOT_TICKS_P99",
        "VD_LEASE_RENEW_INTERVAL",
    ]
}

/// Build the spawned-child anchor env from the orchestrator's own env — each [`spawn_anchor_keys`] key
/// carried iff present + non-empty (absent ⇒ byte-identical child env). The caller appends the
/// node-specific anchors (VD_ORCH + the VD_PEERS ancestor closure) after this.
#[must_use]
pub fn spawn_anchors_from_env(env: &EnvConfig) -> Vec<(&'static str, String)> {
    spawn_anchor_keys()
        .iter()
        .filter_map(|&key| {
            env.string(key)
                .ok()
                .filter(|v| !v.is_empty())
                .map(|v| (key, v))
        })
        .collect()
}

/// THE HAND-OFF BUDGET a demand orchestrator hands every shard it launches, as a spawn anchor — or
/// `None` when the reconciler is inert, which leaves the child's budget at zero and its behaviour exactly
/// as it was before the hand-off ledger existed.
///
/// A hand-off has two ends and ONE duration. The destination is shielded from the reaper for it; the
/// source keeps speaking for its departing occupant — counting itself occupied, so its one-bit
/// `ChildLive` heartbeat keeps beating to its parent (NO pose crosses: the per-occupant position
/// up-relay is DELETED, Step 5 slice D / SL2) — for the same one. Both read
/// `derive_arrival_shield_ticks`, so the two ends cannot
/// drift apart under a later re-tune; a source that fell silent before its destination stopped waiting is
/// the gap the whole ledger exists to close.
///
/// It is derived on the ORCHESTRATOR because that is the only place where both inputs are in scope: how
/// long a hand-off may legitimately take (the saga deadlines) and how fast a realm can be reclaimed (the
/// lifecycle windows). A shard re-deriving it from the half it happens to know would be guessing.
#[must_use]
pub fn handoff_hold_anchor(
    demand: bool,
    rlm: &vd_sim::rlm::RlmTuning,
    saga: &vd_sim::saga::SagaTuning,
) -> Option<(&'static str, String)> {
    demand.then(|| {
        (
            "VD_HANDOFF_HOLD_TICKS",
            vd_sim::rlm::derive_arrival_shield_ticks(rlm, saga).to_string(),
        )
    })
}

/// Resolve the shard's realm SUBJECTIVE time multiplier (D-45(a)): `VD_REALM_TIME_MULTIPLIER` (the
/// per-realm override) else `VD_TIME_MULTIPLIER` (the orchestrator-wide default) else `1.0` (universe
/// rate — byte-identical). Feeds `StubConfig::time_multiplier` (dilates OCCUPANT movement inside the
/// realm, never the celestial orbit).
///
/// # Errors
/// [`ConfigError::Unparseable`] for a value that is not a finite `f64 > 0` (time must move forward at a
/// finite positive rate; a frozen/negative realm clock is an operator error, rejected LOUD at boot).
pub fn resolve_time_multiplier(env: &EnvConfig) -> Result<f64, ConfigError> {
    let raw = env
        .string("VD_REALM_TIME_MULTIPLIER")
        .or_else(|_| env.string("VD_TIME_MULTIPLIER"))
        .unwrap_or_default();
    parse_time_multiplier(raw.trim())
}

/// The `str -> multiplier` map (monomorphic, off the env body so each arm is covered once, HR5). Empty ⇒
/// `1.0`; a finite `f64 > 0` ⇒ that rate; anything else ⇒ loud.
fn parse_time_multiplier(raw: &str) -> Result<f64, ConfigError> {
    if raw.is_empty() {
        return Ok(1.0);
    }
    let unparseable = || ConfigError::Unparseable {
        key: "VD_TIME_MULTIPLIER".to_owned(),
        value: raw.to_owned(),
    };
    let value: f64 = raw.parse().map_err(|_| unparseable())?;
    if value.is_finite() && value > 0.0 {
        Ok(value)
    } else {
        Err(unparseable())
    }
}

/// RLM 5f RG-2 — the default reactive-greeting cadence in SECONDS (the shard reannounces its presence to
/// each booked peer it has not heard from for this long). Scaled to ticks at boot from the live tick rate;
/// `VD_PRESENCE_INTERVAL_TICKS` overrides the computed default outright.
pub const PRESENCE_REANNOUNCE_SECS: f64 = 1.0;

/// RLM 5f RG-2 — build a DEMAND-spawned shard's reactive-greeting policy ([`PresenceAnnounce`]), or `None`
/// for a STATIC shard (⇒ no greeting ⇒ byte-identical). The trigger is `VD_INCARNATION_COOKIE`, which ONLY
/// the demand spawner sets on a forked child (`proc_launch::child_env`); a static dev-cluster/k3d shard never
/// carries it. The greet set is EVERY booked peer (`VD_PEERS` = the ancestor closure ∪ {orchestrator,
/// gateway}) — the cloud-completeness widening (a parent must reach a just-spawned child for a crossing-in),
/// not the gateway alone. The cadence is `PRESENCE_REANNOUNCE_SECS` scaled to ticks by the live `tick_dt`,
/// overridable by `VD_PRESENCE_INTERVAL_TICKS`; a zero override fails LOUD ([`PresenceAnnounce::new`]).
///
/// # Errors
/// A malformed `VD_PEERS`, an unparseable `VD_PRESENCE_INTERVAL_TICKS`, or a zero interval — an operator
/// misconfig fails loud at boot rather than silently disabling reachability.
pub fn resolve_presence(
    env: &EnvConfig,
    tick_dt: f64,
) -> Result<Option<vd_sim::stub::PresenceAnnounce>, String> {
    // A static shard never carries the spawner's incarnation cookie ⇒ it inserts no greeting policy.
    if env
        .string("VD_INCARNATION_COOKIE")
        .unwrap_or_default()
        .is_empty()
    {
        return Ok(None);
    }
    // The default cadence scales with the tick rate (≈ PRESENCE_REANNOUNCE_SECS of real time); the env knob
    // overrides it with an explicit tick count.
    let default_ticks = (PRESENCE_REANNOUNCE_SECS / tick_dt).ceil().max(1.0) as u64;
    let interval_ticks: u64 = env
        .parse_or("VD_PRESENCE_INTERVAL_TICKS", default_ticks)
        .map_err(|e| e.to_string())?;
    let peers: std::collections::BTreeSet<vd_core::NodeId> = env
        .peer_book("VD_PEERS")
        .map_err(|e| e.to_string())?
        .into_keys()
        .collect();
    vd_sim::stub::PresenceAnnounce::new(peers, interval_ticks).map(Some)
}

/// THE HOMES a gateway boots with: which realm each account appears in, and where inside it.
///
/// Every account lives in the world's default home realm — a name, chosen by lineage position, never by
/// measuring anything. The fallback pose is that realm's OWN CENTRE, so there is no offset to justify; in
/// this world the nearest child's surface is about twelve metres away, which makes a star system's origin
/// empty space by construction rather than by a chosen number. The `VD_SPAWN_POSES` stand-in then supplies
/// per-account offsets FROM THAT CENTRE.
///
/// This replaces a map of UNIVERSE-ABSOLUTE positions that the router made usable by descending the whole
/// forest and subtracting each realm's stored centre. An orbiting realm stores its centre as zero, so that
/// walk read every orbiting planet as sitting on its own star: measured through the shipped boot, all five
/// planets of the first star system answered `4.16 m INSIDE` while answering `13.76`–`140.31 m OUTSIDE` in
/// their own frames. A home stated as a name plus a realm-local pose has nothing to descend.
///
/// # Errors
/// [`ConfigError::Unparseable`] on a malformed `VD_SPAWN_POSES` entry, or on a world with no home realm to
/// name (a degenerate forest — stated loud at boot rather than putting every account somewhere arbitrary).
pub fn resolve_homes(
    env: &EnvConfig,
    world: &vd_physics::worldgen::WorldView,
) -> Result<vd_core::home::HomeRegistry, ConfigError> {
    let offsets = resolve_spawn_poses(env)?;
    homes_from_offsets(world, &offsets)
}

/// The `(world, per-account offsets) -> registry` build (monomorphic, off the env body so each arm is
/// covered once, HR5). Each offset is read as METRES FROM THE HOME REALM'S OWN CENTRE.
fn homes_from_offsets(
    world: &vd_physics::worldgen::WorldView,
    offsets: &std::collections::BTreeMap<vd_core::AccountId, vd_core::pose::StampedPose>,
) -> Result<vd_core::home::HomeRegistry, ConfigError> {
    use vd_core::home::{HomeRegistry, StoredHome};
    let degenerate = || ConfigError::Unparseable {
        key: "VD_SPAWN_POSES".to_owned(),
        value: "this world names no home realm (its root has no grandchild)".to_owned(),
    };
    let realm = vd_core::worldgen::default_home_realm(world.regions()).ok_or_else(degenerate)?;
    let at = |offset| StoredHome::in_realm(world.regions(), realm, offset).ok_or_else(degenerate);
    let mut homes = HomeRegistry::new(at(vd_core::glam::DVec3::ZERO)?);
    for (account, pose) in offsets {
        homes = homes.with_account(*account, at(pose.pos.offset())?);
    }
    Ok(homes)
}

/// Resolve the per-account spawn OFFSETS from the `VD_SPAWN_POSES` STAND-IN — the seam the P7 durable
/// per-account home store replaces with ZERO caller reshape.
///
/// Format: `account=x,y,z` entries separated by `;`, where `account` is the decimal [`AccountId`] u128 and
/// `x,y,z` is a position measured FROM THE HOME REALM'S OWN CENTRE, at rest. ABSENT / empty ⇒ an EMPTY map
/// ⇒ every account takes the registry's fallback home. Models `resolve_time_multiplier` (an
/// `unwrap_or_default` read + a monomorphic parse).
///
/// # Errors
/// [`ConfigError::Unparseable`] on any malformed entry (a missing `=`, a non-numeric account, or a position
/// that is not exactly three finite `f64`s) — an operator typo fails LOUD at boot, never a silently dropped
/// spawn point.
pub fn resolve_spawn_poses(
    env: &EnvConfig,
) -> Result<std::collections::BTreeMap<vd_core::AccountId, vd_core::pose::StampedPose>, ConfigError>
{
    let raw = env.string("VD_SPAWN_POSES").unwrap_or_default();
    parse_spawn_poses(raw.trim())
}

/// The `str -> per-account spawn-pose map` parse (monomorphic, off the env body so each arm is covered
/// once, HR5). Empty ⇒ an empty map; each `account=x,y,z` entry ⇒ an at-rest absolute pose in the
/// Universe-root frame; anything malformed ⇒ loud.
fn parse_spawn_poses(
    raw: &str,
) -> Result<std::collections::BTreeMap<vd_core::AccountId, vd_core::pose::StampedPose>, ConfigError>
{
    use vd_core::glam::DVec3;
    use vd_core::pose::{FrameRef, StampedPose};
    let mut map = std::collections::BTreeMap::new();
    if raw.is_empty() {
        return Ok(map);
    }
    let unparseable = |entry: &str| ConfigError::Unparseable {
        key: "VD_SPAWN_POSES".to_owned(),
        value: entry.to_owned(),
    };
    for entry in raw.split(';').filter(|e| !e.trim().is_empty()) {
        let (acct_raw, coords_raw) = entry.split_once('=').ok_or_else(|| unparseable(entry))?;
        let account = vd_core::AccountId(
            acct_raw
                .trim()
                .parse::<u128>()
                .map_err(|_| unparseable(entry))?,
        );
        let coords = coords_raw
            .split(',')
            .map(|c| c.trim().parse::<f64>())
            .collect::<Result<Vec<f64>, _>>()
            .map_err(|_| unparseable(entry))?;
        if coords.len() != 3 {
            return Err(unparseable(entry));
        }
        map.insert(
            account,
            StampedPose::at_rest(
                FrameRef::SystemSpace { system_seed: 0 },
                DVec3::new(coords[0], coords[1], coords[2]),
                vd_core::UniverseTick(0),
            ),
        );
    }
    Ok(map)
}

/// A boot's seed-derived world: the shard's realm REGIONS and the per-realm orbital ELEMENTS of the movers
/// it authors. Named because both the visual and the scale boot arms return exactly this pair.
///
/// It used to be a triple, the third element being each realm's ORIGIN CHAIN — the ancestor list a shard
/// folded to work out where IT sat in the universe. A realm is never told where it sits; only its parent
/// knows that, and the conversion happens there. Nothing is handed a chain any more.
type BootWorld = (
    Vec<vd_core::geometry::RealmRegion>,
    std::collections::BTreeMap<vd_core::pose::RealmId, vd_physics::celestial::OrbitalElements>,
);

/// Build the containment forest + moving-child roster for a `Visual`/`VisualDemand` config.
/// THE ORBIT-SPEED KNOB IS GONE (Stage-C, SL5): `VD_VISUAL_ORBIT_SLOWDOWN` divided the star mass so
/// a pilot could catch a slowed planet — a scale knob on THE world, read by the SHARD alone (the
/// gateway's twin boot built the undivided world, so one env var could split the cluster across two
/// worlds). The crossing gates now fly the shared rendezvous at full orbit speed instead; nothing
/// reads a scale from the environment, so nothing can select a different world. Shared by BOTH
/// visual arms (DRY) so the static-render (`Visual`) and live-demand (`VisualDemand`) modes stay
/// ONE geometry expression.
fn visual_regions_and_movers(
    config: vd_physics::worldgen::UniverseConfig,
    universe_seed: u64,
    held: &std::collections::BTreeSet<vd_core::pose::RealmId>,
    hosted: vd_core::pose::RealmId,
) -> BootWorld {
    // The NEIGHBOURHOOD (own realm + ancestor chain + the children this shard authors) — NEVER its sibling
    // planets. A shard cannot place a realm it does not author, so folding siblings collapses them to the
    // origin and a hosted occupant reads as inside all of them at once (the production re-home flap). Scoping
    // them out is the standalone cure; the movers stay `hosted`'s authored children (single-realm demand shard
    // ⇒ exactly its own; the union-over-held is a co-hosting refinement not yet needed at visual scale).
    let regions =
        vd_physics::worldgen::realm_neighbourhood_for_config(universe_seed, held, &config);
    let moving = vd_physics::worldgen::moving_children_for_config(universe_seed, &config, hosted)
        .into_iter()
        .collect();
    (regions, moving)
}

/// The containment region forest + the moving-child roster for a shard booting at `scale`, hosting
/// `held_realms` (own realm `hosted`). `Walk` ⇒ the seed NEIGHBOURHOOD + an EMPTY roster — BYTE-IDENTICAL
/// to the pre-FA-5 boot (the exact `realm_neighbourhood_for_held`, and `RealmRegions::with_moving_children`
/// with an empty map is a no-op vs `new`). `Visual` ⇒ the FA-5 single-system forest (`realm_regions_for_config`)
/// plus its orbiting planets as movers (`moving_children_for_config`) — the SAME `(seed, visual config)`
/// builds BOTH, so the regions and the moving roster can NEVER derive from different elements (the vet
/// all-shard-seams-same-config rule; the shard authors each planet's live pose against the same region set).
/// `VisualDemand` ⇒ that SAME compressed-real geometry with the AoI band LIVE (`visual_demand(occupant_v_max_mps,
/// tick_dt_s)`) — the demand cluster spins its children up as a moving occupant crosses into their visibility
/// and reaps them behind. `occupant_v_max_mps` (= `move_speed · time_multiplier`) + `tick_dt_s` feed the
/// `VisualDemand` AoI band ONLY (ignored by `Walk`/`Visual`); they close the M-2 two-home owe (the band's tick
/// dt = the live cluster's). Both visual arms route through [`visual_regions_and_movers`] (the orbit-slowdown
/// knob + the one build path).
/// The WORST-INSTANT reach of every child region — what the boot hands `guard_regions_nest` so the
/// fence can judge a mover at its APOAPSIS instead of at the zeroed centre it stores (the placement
/// arc S4; audit finding 27's cure). The boot is the one party that may name how a child moves: a
/// mover's bound is [`vd_physics::motion::Motion::max_excursion_m`] — THE one closed-form
/// worst-instant accessor, never a re-derived `a·(1+e)` beside it (the batch-review DRY finding: the
/// declared single writer was the one expression nobody called); a static child is judged EXACTLY at
/// its authored offset. The fence itself consumes the map kind-blind.
///
/// THE MOVER ROSTER IS THE NEIGHBOURHOOD'S, NOT THE HOSTED REALM'S (batch review, MAJOR — the
/// zero-reach hole): a shard plants rows it does not author — its OWN row, its ancestors' — and a
/// MOVER among them stores a zeroed centre by construction, so keying the reach on the hosted
/// shard's own moving roster judged that mover at `Fixed(0,0,0)`: the size-only verdict the fence
/// was rebuilt to ban, silently lenient on every shard that does not host the mover's parent (one
/// forest, a different verdict per shard). The roster is therefore derived HERE, from THE world
/// itself, for EVERY parent a region row names — so every mover row gets its Excursion on every
/// shard that plants it, and the `None` arm below is reachable only by a genuinely static child.
/// Boot-time-only knowledge, discarded after the guard: the sim's runtime roster stays the hosted
/// realm's own authored children (SL1 — a realm is never handed its own placement to run with).
#[must_use]
pub fn child_reaches(
    universe_seed: u64,
    regions: &[vd_core::geometry::RealmRegion],
    occupant_v_max_mps: f64,
    tick_dt_s: f64,
) -> std::collections::BTreeMap<vd_core::pose::RealmId, vd_core::geometry::ChildReach> {
    use vd_core::geometry::ChildReach;
    use vd_physics::motion::Motion;
    let config = vd_physics::worldgen::UniverseConfig::world(occupant_v_max_mps, tick_dt_s);
    // Every mover of this neighbourhood, keyed by realm: the union of THE world's moving children
    // over every parent a region row names. The regions and this roster derive from the SAME
    // `(seed, config)` forest, so a mover row missing from it is unrepresentable — which is what
    // makes the static `None` arm honest rather than a defaulted zero wearing a `Fixed` label.
    let parents: std::collections::BTreeSet<vd_core::pose::RealmId> =
        regions.iter().filter_map(|r| r.parent).collect();
    let movers: std::collections::BTreeMap<
        vd_core::pose::RealmId,
        vd_physics::celestial::OrbitalElements,
    > = parents
        .iter()
        .flat_map(|p| vd_physics::worldgen::moving_children_for_config(universe_seed, &config, *p))
        .collect();
    regions
        .iter()
        .filter(|r| r.parent.is_some())
        .map(|r| {
            let reach = match movers.get(&r.realm) {
                Some(e) => ChildReach::Excursion(Motion::Kepler(*e).max_excursion_m()),
                None => {
                    // The stored offset is measured in the PARENT's frame, so the parent's tier
                    // scales its cell anchor into metres.
                    let tier = regions
                        .iter()
                        .find(|p| Some(p.realm) == r.parent)
                        .map_or(r.frame.tier(), |p| p.frame.tier());
                    ChildReach::Fixed(r.center.delta_m(
                        vd_core::pose::LatticePos::local(vd_core::glam::DVec3::ZERO),
                        tier,
                    ))
                }
            };
            (r.realm, reach)
        })
        .collect()
}

#[must_use]
pub fn boot_regions_and_movers(
    universe_seed: u64,
    held_realms: &std::collections::BTreeSet<vd_core::pose::RealmId>,
    hosted: vd_core::pose::RealmId,
    occupant_v_max_mps: f64,
    tick_dt_s: f64,
) -> BootWorld {
    // NO CHOICE, BY CONSTRUCTION. This used to `match` a scale read from the environment, and a live
    // cluster was MEASURED running two of them at once — the orchestrator on one world, its own gateway
    // on another, from a single launch of a single script. Logins were placed by one universe's rules and
    // simulated by another's. Deleting the parameter is the fix: two processes cannot disagree about a
    // value that does not exist.
    visual_regions_and_movers(
        vd_physics::worldgen::UniverseConfig::world(occupant_v_max_mps, tick_dt_s),
        universe_seed,
        held_realms,
        hosted,
    )
}

/// The WORLD a node boots into, for the SAME `scale` [`boot_regions_and_movers`] selects its regions from.
///
/// Both nodes must describe the same universe or the game is incoherent: the gateway decides which realm a
/// login lands in, and the shard decides what is around them once there. The gateway used to hardcode the
/// walk world regardless of what the shards were told to run, so a visual-scale cluster placed its players
/// by walk-scale geometry — a login resolved in a world with different stars in different places than the
/// one that would then simulate it. Deriving both from `scale` here is what makes that unstateable.
///
/// `Walk` is the HAND-PLACED world (its station and area are fixtures, not generated content); the visual
/// arms are seed-generated. `occupant_v_max_mps`/`tick_dt_s` feed the live AoI band exactly as they do for
/// the regions, so the world a gateway holds carries the same bands the shard evaluates.
#[must_use]
pub fn boot_world(
    universe_seed: u64,
    occupant_v_max_mps: f64,
    tick_dt_s: f64,
) -> vd_physics::worldgen::WorldView {
    use vd_physics::worldgen::{UniverseConfig, WorldView};
    // THE OTHER HALF OF THE MEASURED DISAGREEMENT. A live cluster was read process by process: the
    // orchestrator held `visual-demand` and its own gateway held `visual`, from one launch of one script.
    // This function is where the gateway's half of that came from — three arms, and nothing to stop the
    // two ends being handed different ones. Now there is one world and no argument to get wrong.
    WorldView::generated(
        universe_seed,
        &UniverseConfig::world(occupant_v_max_mps, tick_dt_s),
    )
}

/// THE WINDOW LANE's marker roster for one shard boot (Slice A, docs/design/window_lane.md
/// §2.2/§2.8): per DIRECT child of a held realm, the pre-encoded `TAG_LUMA` bag drawn from that
/// child's own generation stream (the owner-ruled R4 datum — the parent authors a sleeping
/// child's point of light). Derived from THE world (`system_photometrics_for_config`, the SAME
/// `(seed, config)` every other boot seam reads — SL5) and FILTERED to the booted forest's
/// direct children of the held realms, so a shard holds bags for exactly the children it may
/// state markers about and nothing else. Empty wherever no child carries a draw (walk
/// stations/areas; planets — their photometric ladder is an owed later draw).
#[must_use]
pub fn child_luma_bags(
    universe_seed: u64,
    occupant_v_max_mps: f64,
    tick_dt_s: f64,
    regions: &[vd_core::geometry::RealmRegion],
    held_realms: &std::collections::BTreeSet<vd_core::pose::RealmId>,
) -> std::collections::BTreeMap<vd_core::pose::RealmId, Vec<u8>> {
    let config = vd_physics::worldgen::UniverseConfig::world(occupant_v_max_mps, tick_dt_s);
    vd_physics::worldgen::system_photometrics_for_config(universe_seed, &config)
        .into_iter()
        .filter(|(realm, _)| {
            regions.iter().any(|r| {
                r.realm == *realm && r.parent.is_some_and(|parent| held_realms.contains(&parent))
            })
        })
        .map(|(realm, draw)| (realm, vd_physics::worldgen::marker_luma_bag(&draw)))
        .collect()
}

// ---- THE WORLD ROSTER — the ONE derivation point for every named realm of THE world -----------

/// THE named realms every process cluster stands on — DERIVED from THE world, never stated.
///
/// WHY THIS EXISTS. The retired cluster shapes named their realms as constants (`System(8)`,
/// `Planet(7)`, `Station(7)`, `Area(7)`) and THE world contains NONE of them: worldgen has no
/// Station/Area arm at all (stations are player-built — those legs return here as roster entries when
/// the block/station slice grows THE world), and a sibling star's seed is generated, never `8`. A
/// cluster naming a realm the world does not contain boots a shard that fails `guard_regions_nest`
/// with 0 ambient roots and DIES — and an occupant steered toward an unhosted realm is dropped with no
/// reply (`saga_runtime`'s "CROSSING UNRESOLVED"). That drop WAS a permanent strand; since the
/// D-WORLD-2 cure the source's armed ttl re-drive retries it a budgeted number of times and then
/// aborts locally, clearing its latch — bounded self-heal, but still a crossing the cluster cannot
/// serve. So this struct is the ONLY place allowed to name a realm of THE world, and it derives
/// every name through [`boot_regions_and_movers`] — the SAME call the shard bin boots with — so the
/// roster and the booted world cannot disagree.
///
/// The three `*_m` fields are the FLIGHT-LAW margins ([`world_roster`] asserts them): every static
/// cluster's gate flies the ±Z polar corridor, and these are what make that corridor provably unable
/// to reach a realm no shard hosts.
#[derive(Clone, Debug, PartialEq)]
pub struct WorldRoster {
    /// The login realm: `default_home_realm` over THE world's regions (the root's first grandchild) —
    /// a lineage position, never a stated seed.
    pub home: vd_core::pose::RealmId,
    /// The home region's PARENT — the between-systems space every departure lands in.
    pub galaxy: vd_core::pose::RealmId,
    /// The home system's INNERMOST mover — smallest SEMI-MAJOR AXIS, deliberately NOT smallest epoch
    /// radius (the two coincide only at e ≈ 0).
    pub inner: vd_core::pose::RealmId,
    /// The inner mover's orbital elements, exactly as the home shard authors them (the rendezvous
    /// aims by these).
    pub inner_elements: vd_physics::celestial::OrbitalElements,
    /// The galaxy's lowest-seed ring sibling of the home system (lowest seed for determinism).
    pub sibling: vd_core::pose::RealmId,
    /// The sibling's authored placement in the GALAXY's frame — read from the galaxy-hosted boot, the
    /// parent that authors it (SL1), never folded from the root.
    pub sibling_centre: vd_core::glam::DVec3,
    /// I-POLE: the farthest any mover reaches out of the orbital plane PLUS its whole containment
    /// reach (`max over movers of sma·(1+e)·|sin i| + planet_soi + outset`), taken over BOTH systems
    /// the static legs fly — the HOME system and its ring SIBLING, whose orbits are a different
    /// per-system draw (batch review: the sibling's margins used to be a bare comment). Every polar
    /// waypoint a flight plan states uses `|z| >= 2 · pole_altitude_m` — clear of every planet at
    /// every orbital phase by construction.
    pub pole_altitude_m: f64,
    /// I-AXIS: the closest any orbit — home OR ring sibling — comes to the polar (±Z) axis
    /// (`min over movers of sma·(1−e)·cos i`). Asserted `> 2 · planet_soi` per system — what
    /// licenses every ±Z corridor leg, including the chain gate's creep into the sibling.
    pub axis_clearance_m: f64,
    /// I-RADIAL: the smallest clear gap between adjacent orbits, in the home system AND the ring
    /// sibling (`min over adjacent movers of sma_{n+1}·(1−e_{n+1}) − sma_n·(1+e_n)`). Asserted wider
    /// than one planet's containment release reach per system — what licenses the rendezvous and the
    /// lift off the inner planet.
    pub radial_gap_m: f64,
}

/// The three ±Z corridor margins over ONE system's mover set — one derivation for the home system
/// AND its ring sibling, so the two cannot be computed by different arithmetic (the chain gate flies
/// the same polar corridor into both).
struct CorridorMargins {
    axis_clearance_m: f64,
    pole_altitude_m: f64,
    radial_gap_m: f64,
}

fn corridor_margins(
    movers: &std::collections::BTreeMap<
        vd_core::pose::RealmId,
        vd_physics::celestial::OrbitalElements,
    >,
    release_reach: f64,
) -> CorridorMargins {
    use vd_physics::motion::Motion;
    // The apoapsis terms read THE one worst-instant accessor (`Motion::max_excursion_m`) — never a
    // re-derived `a·(1+e)` beside it (the batch-review DRY finding).
    let axis_clearance_m = movers
        .values()
        .map(|e| e.sma * (1.0 - e.ecc) * e.inclination.cos())
        .fold(f64::INFINITY, f64::min);
    let pole_altitude_m = movers
        .values()
        .map(|e| Motion::Kepler(*e).max_excursion_m() * e.inclination.sin().abs() + release_reach)
        .fold(0.0_f64, f64::max);
    let mut by_sma: Vec<&vd_physics::celestial::OrbitalElements> = movers.values().collect();
    by_sma.sort_by(|a, b| a.sma.total_cmp(&b.sma));
    let radial_gap_m = by_sma
        .windows(2)
        .map(|w| w[1].sma * (1.0 - w[1].ecc) - Motion::Kepler(*w[0]).max_excursion_m())
        .fold(f64::INFINITY, f64::min);
    CorridorMargins {
        axis_clearance_m,
        pole_altitude_m,
        radial_gap_m,
    }
}

/// Assert the I-AXIS / I-POLE / I-RADIAL flight-law preconditions for ONE system's margins — run for
/// the home system AND the ring sibling, naming the system so a violated corridor names its world.
fn assert_corridor_margins(
    system: &str,
    m: &CorridorMargins,
    planet_soi: f64,
    release_reach: f64,
    system_soi_r_m: f64,
) {
    // I-AXIS: no orbit may come within 2× the planet SOI of the polar (±Z) axis, or a ±Z corridor leg
    // could thread a planet's shell and strand the dot on an unhosted realm (J-0).
    assert!(
        m.axis_clearance_m > 2.0 * planet_soi,
        "I-AXIS violated ({system}): an orbit approaches the polar (±Z) axis to {:.2} m, inside \
         2× the planet SOI ({:.2} m) — a ±Z corridor leg could thread a planet shell; the world \
         changed under the static flight law, so restate the corridor, never this assert",
        m.axis_clearance_m,
        2.0 * planet_soi,
    );
    // I-POLE: the polar park height (2 × pole_altitude) must itself stay INSIDE the system shell, or a
    // "lift off the planet" polar waypoint would exit the system and fire an unintended crossing.
    assert!(
        2.0 * m.pole_altitude_m + release_reach < system_soi_r_m,
        "I-POLE violated ({system}): the polar park height 2×{:.2} m (+ the planet release reach \
         {release_reach:.2} m) does not fit inside the system shell ({system_soi_r_m:.2} m) — an \
         in-system polar waypoint would exit the system; restate the corridor, never this assert",
        m.pole_altitude_m,
    );
    // I-RADIAL: adjacent orbits must be separated by more than one planet's release reach, so a ship
    // parked at one orbit's rendezvous is provably clear of both neighbours (and the lift off the
    // inner planet never grazes the next orbit out).
    assert!(
        m.radial_gap_m > release_reach,
        "I-RADIAL violated ({system}): adjacent orbits leave only {:.2} m of clear gap, within one \
         planet's containment release reach ({release_reach:.2} m) — the rendezvous/park and the \
         inner-planet lift-off are no longer licensed; restate the corridor, never this assert",
        m.radial_gap_m,
    );
}

/// Derive [`WorldRoster`] from THE world and ASSERT the flight-law preconditions (I-AXIS / I-POLE /
/// I-RADIAL, plus J1) — so a world change that invalidates the static clusters' ±Z corridor fails
/// LOUD at derivation, never as a dot stranded mid-gate on a realm no shard hosts.
///
/// # Panics
/// When THE world stops satisfying a precondition the static flight law stands on, or degenerates
/// (no home realm, no movers, no ring sibling). The cure is restating the corridor against the new
/// world — NEVER weakening an assert.
#[must_use]
pub fn world_roster(p: &DevClusterParams) -> WorldRoster {
    let world = boot_world(p.universe_seed, p.move_speed, p.tick_dt);
    let home = vd_core::worldgen::default_home_realm(world.regions())
        .expect("THE world names a home realm (root → galaxy → system)");
    let galaxy = world
        .regions()
        .iter()
        .find(|r| r.realm == home)
        .and_then(|r| r.parent)
        .expect("the home realm nests under a parent (the galaxy)");

    // The HOME shard's own boot: the movers are the children the home realm authors (SL1 — a parent
    // authors its children's placements; nothing here folds an absolute from the root).
    let held_home = std::collections::BTreeSet::from([home]);
    let (_, movers) =
        boot_regions_and_movers(p.universe_seed, &held_home, home, p.move_speed, p.tick_dt);
    assert!(
        !movers.is_empty(),
        "THE world's home system authors orbiting movers — a moverless home has no flight law to check",
    );
    let (inner, inner_elements) = movers
        .iter()
        .min_by(|a, b| a.1.sma.total_cmp(&b.1.sma))
        .map(|(realm, elements)| (*realm, *elements))
        .expect("non-empty movers");

    // ONE planet's containment reach: its SOI plus the band's release outset — past this a dot is
    // clear of the planet's authority. From the SAME config the boot builds THE world with.
    let config = vd_physics::worldgen::UniverseConfig::world(p.move_speed, p.tick_dt);
    let planet_soi = config.planet.planet_soi_r_m;
    let release_reach = planet_soi + config.band.outset_m;

    // I-AXIS / I-POLE / I-RADIAL over the HOME system's movers.
    let home_margins = corridor_margins(&movers, release_reach);
    assert_corridor_margins(
        "the home system",
        &home_margins,
        planet_soi,
        release_reach,
        config.stellar.system_soi_r_m,
    );

    // The GALAXY-hosted boot: the ring placements the galaxy authors for its systems — read from the
    // parent that authors them (SL1; the same discipline the demand suite's neighbour_system uses).
    let held_gal = std::collections::BTreeSet::from([galaxy]);
    let (gal_regions, _) =
        boot_regions_and_movers(p.universe_seed, &held_gal, galaxy, p.move_speed, p.tick_dt);
    let home_centre = gal_regions
        .iter()
        .find(|r| r.realm == home)
        .map(|r| r.center)
        .expect("the galaxy authors the home system's placement");
    // J1: the home system sits at the GALACTIC ORIGIN — a home↔galaxy crossing is numerically an
    // identity in the drawn space (the downward conversion subtracts zero), which is exactly why a
    // file-based scene stays valid across that one crossing. ASSERTED, not assumed.
    assert_eq!(
        home_centre.cell(),
        vd_core::glam::I64Vec3::ZERO,
        "J1: the home system's authored placement fits one lattice cell",
    );
    assert_eq!(
        home_centre.offset(),
        vd_core::glam::DVec3::ZERO,
        "J1: the home system sits at the galactic origin (ring index 0) — the identity the scene \
         emitter and the render gate stand on",
    );
    let (sibling, sibling_region) = gal_regions
        .iter()
        .filter(|r| r.parent == Some(galaxy) && r.realm != home)
        .filter_map(|r| match r.realm {
            vd_core::pose::RealmId::System(seed) => Some((seed, r)),
            _ => None,
        })
        .min_by_key(|(seed, _)| *seed)
        .map(|(seed, r)| (vd_core::pose::RealmId::System(seed), r))
        .expect("the multi-star galaxy has a ring sibling of the home system");
    assert_eq!(
        sibling_region.center.cell(),
        vd_core::glam::I64Vec3::ZERO,
        "a ring placement fits inside one lattice cell",
    );
    let sibling_centre = sibling_region.center.offset();
    assert_ne!(
        sibling_centre,
        vd_core::glam::DVec3::ZERO,
        "J1: the sibling is NOT at the origin — two systems on one point would be two authorities \
         over one position",
    );

    // The SIBLING shard's own boot: the chain gate's leg E flies the SAME ±Z polar corridor into the
    // ring sibling, whose planets are a DIFFERENT per-system draw (ecc/inclination come off the
    // sibling's own seed stream) — so its corridor margins are ASSERTED here too, never argued from
    // the home's (batch review: the "≥140 m under every sibling planet" clearance was a bare
    // comment; a seed or world-number change widening the sibling's inclinations would have strung
    // the creep through a planet shell with every assert still green).
    let held_sib = std::collections::BTreeSet::from([sibling]);
    let (_, sib_movers) =
        boot_regions_and_movers(p.universe_seed, &held_sib, sibling, p.move_speed, p.tick_dt);
    assert!(
        !sib_movers.is_empty(),
        "THE world's ring sibling authors orbiting movers — a moverless sibling has no corridor \
         margins to check",
    );
    let sib_margins = corridor_margins(&sib_movers, release_reach);
    assert_corridor_margins(
        "the ring sibling",
        &sib_margins,
        planet_soi,
        release_reach,
        config.stellar.system_soi_r_m,
    );

    WorldRoster {
        home,
        galaxy,
        inner,
        inner_elements,
        sibling,
        sibling_centre,
        // The roster carries the WORST case over both systems, so a flight plan sized by these
        // fields is licensed in whichever system a leg flies.
        pole_altitude_m: home_margins
            .pole_altitude_m
            .max(sib_margins.pole_altitude_m),
        axis_clearance_m: home_margins
            .axis_clearance_m
            .min(sib_margins.axis_clearance_m),
        radial_gap_m: home_margins.radial_gap_m.min(sib_margins.radial_gap_m),
    }
}

/// Serialize a co-hosted realm SET to the `VD_HELD_REALMS` env format (`kind:seed` comma-separated, e.g.
/// `system:7,planet:7,station:7,area:7`). NO launcher produces it since co-hosting retired with the
/// --triple shape (D-WORLD-6); the codec + the shard-side [`parse_held_realms`] keep unit coverage so
/// the parse cannot rot silently. Sorted (BTreeSet order) so the env is deterministic.
#[must_use]
pub fn held_realms_env(realms: &std::collections::BTreeSet<vd_core::pose::RealmId>) -> String {
    realms
        .iter()
        .map(|r| realm_token(*r))
        .collect::<Vec<_>>()
        .join(",")
}

/// The `kind:seed` token for ONE realm (the `VD_HELD_REALMS` element). Monomorphic — the `RealmId` arm
/// match is covered here, off the iterator body. A `Ship` realm keys on its entity id (a `u128`), encoded
/// as `ship:<u128>` for totality (ships are P8 — never in a P3 `VD_HELD_REALMS` — but the codec stays
/// exhaustive so a future ship co-host round-trips).
fn realm_token(realm: vd_core::pose::RealmId) -> String {
    use vd_core::pose::RealmId;
    match realm {
        RealmId::System(s) => format!("system:{s}"),
        RealmId::Planet(s) => format!("planet:{s}"),
        RealmId::Station(s) => format!("station:{s}"),
        RealmId::Area(s) => format!("area:{s}"),
        RealmId::Ship(id) => format!("ship:{}", id.0),
    }
}

/// Parse the `VD_HELD_REALMS` env value (`kind:seed` comma-separated) into the co-hosted realm set. ABSENT
/// or EMPTY ⇒ `{fallback}` (the single-realm default — a shard with no co-hosting knob holds exactly its
/// own realm, byte-identical). A malformed token fails LOUD. `fallback` (this shard's own realm) is ALWAYS
/// included even if the env omits it (a shard always hosts its own realm). The inverse of [`held_realms_env`].
///
/// # Errors
/// Returns the offending token string when a `kind:seed` element is unrecognized or the seed is not a `u64`.
pub fn parse_held_realms(
    raw: &str,
    fallback: vd_core::pose::RealmId,
) -> Result<std::collections::BTreeSet<vd_core::pose::RealmId>, String> {
    let mut set = std::collections::BTreeSet::from([fallback]);
    for token in raw.split(',').map(str::trim).filter(|t| !t.is_empty()) {
        set.insert(parse_realm_token(token)?);
    }
    Ok(set)
}

/// Parse ONE `kind:seed` token into a `RealmId` (monomorphic — the split/kind/seed error arms are covered
/// here off the CSV loop body). `ship:<u128>` decodes an entity-keyed Ship realm (the inverse of the ship
/// arm in [`realm_token`]; P8, never in a P3 rig — kept for round-trip totality).
fn parse_realm_token(token: &str) -> Result<vd_core::pose::RealmId, String> {
    use vd_core::pose::RealmId;
    let (kind, seed_str) = token
        .split_once(':')
        .ok_or_else(|| format!("VD_HELD_REALMS token {token:?} is not `kind:seed`"))?;
    if kind == "ship" {
        let raw: u128 = seed_str
            .parse()
            .map_err(|_| format!("VD_HELD_REALMS token {token:?} has a non-u128 ship id"))?;
        return Ok(RealmId::Ship(vd_core::EntityId(raw)));
    }
    let seed: u64 = seed_str
        .parse()
        .map_err(|_| format!("VD_HELD_REALMS token {token:?} has a non-u64 seed"))?;
    match kind {
        "system" => Ok(RealmId::System(seed)),
        "planet" => Ok(RealmId::Planet(seed)),
        "station" => Ok(RealmId::Station(seed)),
        "area" => Ok(RealmId::Area(seed)),
        other => Err(format!(
            "VD_HELD_REALMS token {token:?} has unknown kind {other:?}"
        )),
    }
}

// ---- the process-tier lock ---------------------------------------------------

/// THE lock every cluster-booting test holds for its whole body.
///
/// WHY IT EXISTS. `cargo test` runs the test fns inside ONE binary CONCURRENTLY (one thread per
/// test). A test that boots real node binaries needs several CPUs for a few seconds AND needs the
/// ports it reserved to still be free when the children actually bind them. Two such tests running
/// at once contend for both, so whichever wins passes and the others see `/readyz` 503 or an
/// unreachable listener — and WHICH ones fail shuffles run to run. That is a gate that lies: a real
/// regression is indistinguishable from the noise. Serializing at the source makes a green run mean
/// something, no matter how the binary is invoked — by `cargo test --workspace`, by a `just` recipe,
/// or by a developer running it directly. It is deliberately NOT a `justfile` flag, because a flag
/// leaves the documented `cargo test --workspace` still lying.
///
/// THREE THINGS THE CALLER MUST KNOW:
///
/// 1. **Declare it FIRST in the test body**, before any `Cluster` local:
///    ```ignore
///    #[test]
///    fn my_process_test() {
///        let _tier = vd_bins::cluster_tier();
///        let mut cluster = Cluster::new();
///        // ...
///    }
///    ```
///    Rust drops locals in REVERSE declaration order, so declaring the guard first makes it outlive
///    `Cluster::drop` ([`Cluster`]'s `Drop`, which kills AND `wait`s). That wait is what actually
///    frees the bound ports — see [`Cluster::kill_and_reap`], whose doc explains why the reap is
///    load-bearing. Releasing the tier before the reap would hand the next test a set of ports the
///    previous cluster's children still hold.
///
/// 2. **Poison is recovered, deliberately.** A test that panics while holding the tier poisons the
///    mutex; propagating that would turn ONE genuine failure into N spurious ones and re-create the
///    exact "which failure is real?" problem this removes. The lock guards scheduling, not data, so
///    there is no invariant a panic could have broken.
///
/// 3. **THE PREMISE — this is a PROCESS-LOCAL guard.** It is sufficient only because `cargo test`
///    runs test EXECUTABLES sequentially while parallelising WITHIN each one. It gives NO protection
///    under `cargo-nextest` (process per test), nor against two concurrent `cargo` invocations, nor
///    against anything else on the machine holding a port. The complete fix is deterministic
///    per-fixture port bands; see the ledger entry in `docs/design/DEFERRED.md`.
static CLUSTER_TIER: std::sync::Mutex<()> = std::sync::Mutex::new(());

/// The thread currently holding [`CLUSTER_TIER`], so [`spawn_node`] can refuse to fork a real binary
/// from a test that forgot the guard. A separate, always-briefly-held lock rather than a field on the
/// guard: the tripwire must be readable from code that does NOT hold the tier.
static TIER_OWNER: std::sync::Mutex<Option<std::thread::ThreadId>> = std::sync::Mutex::new(None);

/// RAII handle for the process tier. Hold it for the whole test body; see [`cluster_tier`].
pub struct ClusterTier {
    /// Held for the guard's lifetime. Never read — the lock IS the effect.
    _guard: std::sync::MutexGuard<'static, ()>,
}

impl Drop for ClusterTier {
    fn drop(&mut self) {
        // Clear the owner BEFORE `_guard` drops (Drop::drop runs ahead of field drops), so the tier is
        // never observably free-but-owned.
        *TIER_OWNER
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner) = None;
    }
}

/// Acquire THE process tier — the first statement of every test that boots real node binaries.
/// Blocks until the previous holder's cluster is fully reaped. See [`CLUSTER_TIER`] for the rules.
#[must_use]
pub fn cluster_tier() -> ClusterTier {
    let guard = CLUSTER_TIER
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    *TIER_OWNER
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner) = Some(std::thread::current().id());
    ClusterTier { _guard: guard }
}

/// Does the calling thread hold the process tier? The tripwire [`spawn_node`] checks.
#[must_use]
pub fn tier_held_by_current_thread() -> bool {
    *TIER_OWNER
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
        == Some(std::thread::current().id())
}

/// The remedy a forgotten-guard panic prints. One string so the message, the doc and the
/// source-scanning test cannot drift apart.
pub const TIER_REMEDY: &str = "this test forks a real node binary but does not hold the process tier — add \
     `let _tier = vd_bins::cluster_tier();` as the FIRST statement of the test body \
     (first, so it outlives the Cluster reap that frees the ports)";

// ---- RAII child guard --------------------------------------------------------

/// Owns spawned child processes and KILLS them on drop — so a partial spawn, an
/// early `?`, or a panic can never leak a node holding its ports. Disarm with
/// [`Cluster::into_pids`] only AFTER a durable kill-record (the runfile) exists;
/// the children then outlive the launcher and `down` reaps them by recorded PID.
#[derive(Default)]
pub struct Cluster {
    children: Vec<(&'static str, Child)>,
}

impl Cluster {
    #[must_use]
    pub fn new() -> Cluster {
        Cluster {
            children: Vec::new(),
        }
    }

    pub fn push(&mut self, name: &'static str, child: Child) {
        self.children.push((name, child));
    }

    #[must_use]
    pub fn pids(&self) -> Vec<u32> {
        self.children.iter().map(|(_, c)| c.id()).collect()
    }

    /// The display labels of the spawned children, in spawn order.
    #[must_use]
    pub fn names(&self) -> Vec<&'static str> {
        self.children.iter().map(|(name, _)| *name).collect()
    }

    /// The first child found already exited (a bring-up failure signal).
    pub fn first_exited(&mut self) -> Option<(&'static str, ExitStatus)> {
        for (name, child) in &mut self.children {
            if let Ok(Some(status)) = child.try_wait() {
                return Some((name, status));
            }
        }
        None
    }

    /// SIGKILL + REAP the most-recently-pushed child labelled `name`, and REMOVE it from the reap set (so
    /// its bind port frees for an immediate restart). Panics if no such child is present. For CrashLoop /
    /// restart process tests: the child stays RAII-reaped by the `Cluster` until this is called, so a panic
    /// between spawn and kill never leaks it (unlike a bare `Child`, which does not kill on drop).
    /// SIGKILL the named child and REAP it (block on `wait()`) before returning. The `wait()` reap is
    /// LOAD-BEARING for the immediate same-address restart pattern (e.g. `outbox_sigkill_restart`): it releases
    /// the kernel's hold on the bound UDP port + the redb lock synchronously, so the boot-2 process can rebind
    /// the SAME `reserve_udp_addr()` without a bind race. Dropping the `wait()` (or a future `into_pids`-style
    /// disarm here) would make same-addr restart tests non-deterministically flaky. Reaps the LAST child of the
    /// given name (rposition) so a restart re-`push`ed under the same name kills the right generation.
    pub fn kill_and_reap(&mut self, name: &'static str) {
        let idx = self
            .children
            .iter()
            .rposition(|(n, _)| *n == name)
            .unwrap_or_else(|| panic!("Cluster has no child named {name} to kill"));
        let (_, mut child) = self.children.remove(idx);
        let _ = child.kill(); // SIGKILL on Unix
        let _ = child.wait(); // reap (release the port + the redb lock + the zombie) — load-bearing, see doc
    }

    /// SIGTERM (the GRACEFUL k8s pod-stop signal) the most-recently-pushed child labelled `name`,
    /// and DELIBERATELY LEAVE IT IN THE REAP SET. Panics if no such child is present.
    ///
    /// This is the drain-observing twin of [`Cluster::kill_and_reap`] (which SIGKILLs and REMOVES).
    /// Keeping the child owned is the whole point: a graceful-shutdown test must stay alive across
    /// the drain window to observe it, and every assertion in that window can fail. A bare
    /// `std::process::Child` does NOT kill on drop, so a child held outside the `Cluster` purely to
    /// be signalled leaks a running node on the first failed assertion — which then competes for CPU
    /// with every later test in the binary and makes the next failure MORE likely. Signalling
    /// through the `Cluster` keeps RAII cover for the entire drain.
    pub fn sigterm(&mut self, name: &'static str) {
        let idx = self
            .children
            .iter()
            .rposition(|(n, _)| *n == name)
            .unwrap_or_else(|| panic!("Cluster has no child named {name} to SIGTERM"));
        signal_pid(self.children[idx].1.id(), "TERM");
    }

    /// Block until the most-recently-pushed child labelled `name` EXITS ON ITS OWN, then remove it
    /// from the reap set. Panics if no such child is present. The graceful counterpart to
    /// [`Cluster::kill_and_reap`]: it never signals, so it proves the child terminated by itself —
    /// which is exactly what a SIGTERM-drain test must assert. Returns the exit status.
    pub fn wait_for_exit(&mut self, name: &'static str) -> ExitStatus {
        let idx = self
            .children
            .iter()
            .rposition(|(n, _)| *n == name)
            .unwrap_or_else(|| panic!("Cluster has no child named {name} to wait for"));
        let (_, mut child) = self.children.remove(idx);
        child
            .wait()
            .unwrap_or_else(|e| panic!("wait for {name} to exit: {e}"))
    }

    /// Disarm: hand back the PIDs and forget the children WITHOUT killing them
    /// (the runfile is now the kill record). Call only once bring-up succeeded.
    #[must_use]
    pub fn into_pids(mut self) -> Vec<u32> {
        let pids = self.pids();
        self.children.clear(); // Drop now kills nothing
        pids
    }
}

impl Drop for Cluster {
    fn drop(&mut self) {
        for (_, child) in &mut self.children {
            let _ = child.kill();
            let _ = child.wait();
        }
    }
}

#[cfg(test)]
mod cluster_addrs_tests {
    use super::ClusterAddrs;
    use std::collections::BTreeSet;

    #[test]
    fn reserve_yields_eight_distinct_udp_and_nine_distinct_tcp() {
        let a = ClusterAddrs::reserve();
        // TWO separate sets, never one of seventeen — UDP and TCP are separate port namespaces, so a
        // cross-protocol port match is legal and asserting against it would invent a flake.
        let udp: BTreeSet<_> = a.udp_binds().into_iter().collect();
        let tcp: BTreeSet<_> = a.tcp_binds().into_iter().collect();
        assert_eq!(udp.len(), 8);
        assert_eq!(tcp.len(), 9);
        assert_eq!(a.gateway_admin, None);
    }

    #[test]
    fn reserve_binds_each_field_at_its_real_protocol() {
        // THE RED CONTROL for the reservation drift this fixture removes: `shard_b` becomes `VD_BIND`
        // (a QUIC/UDP bind), but thirteen hand-rolled literals reserved it with the TCP helper — so
        // they proved a TCP port free and then bound UDP on that number. Re-binding each field at the
        // protocol it will actually be used with is what pins the mapping.
        let a = ClusterAddrs::reserve();
        for addr in a.udp_binds() {
            std::net::UdpSocket::bind(addr)
                .unwrap_or_else(|e| panic!("{addr} must be free as UDP: {e}"));
        }
        for addr in a.tcp_binds() {
            std::net::TcpListener::bind(addr)
                .unwrap_or_else(|e| panic!("{addr} must be free as TCP: {e}"));
        }
    }

    #[test]
    fn with_gateway_admin_sets_only_that_field() {
        let base = ClusterAddrs::reserve();
        let extra = super::reserve_tcp_addr();
        let with = base.with_gateway_admin(extra);
        assert_eq!(with.gateway_admin, Some(extra));
        assert_eq!(with.udp_binds(), base.udp_binds());
        assert_eq!(with.tcp_binds(), base.tcp_binds());
    }

    #[test]
    fn two_reservations_do_not_overlap_within_a_protocol() {
        // Holding BOTH sets alive at once is the point: it proves `reserve()` does not hand the same
        // port to two live clusters, which a one-at-a-time reservation cannot promise.
        let a = ClusterAddrs::reserve();
        let b = ClusterAddrs::reserve();
        let udp: BTreeSet<_> = a.udp_binds().into_iter().chain(b.udp_binds()).collect();
        let tcp: BTreeSet<_> = a.tcp_binds().into_iter().chain(b.tcp_binds()).collect();
        assert_eq!(udp.len(), 16);
        assert_eq!(tcp.len(), 18);
    }
}

#[cfg(test)]
mod cluster_tier_tests {
    use super::{TIER_REMEDY, cluster_tier, spawn_node, tier_held_by_current_thread};
    use std::sync::Arc;
    use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

    /// Run `f` with the panic hook silenced, so a DELIBERATE panic does not spam the gate log and
    /// make a green run look alarming. Restores the previous hook.
    fn without_panic_noise<T>(f: impl FnOnce() -> T) -> T {
        let prev = std::panic::take_hook();
        std::panic::set_hook(Box::new(|_| {}));
        let out = f();
        std::panic::set_hook(prev);
        out
    }

    #[test]
    fn the_tier_admits_exactly_one_holder_at_a_time() {
        // THE property: N threads racing for the tier are serialized. `inside` counts concurrent
        // holders; `peak` records the largest count ever observed. If the lock did nothing, several
        // threads would overlap and peak would exceed 1.
        let inside = Arc::new(AtomicUsize::new(0));
        let peak = Arc::new(AtomicUsize::new(0));
        let threads: Vec<_> = (0..8)
            .map(|_| {
                let inside = Arc::clone(&inside);
                let peak = Arc::clone(&peak);
                std::thread::spawn(move || {
                    let _tier = cluster_tier();
                    let now = inside.fetch_add(1, Ordering::SeqCst) + 1;
                    peak.fetch_max(now, Ordering::SeqCst);
                    // Hold long enough that a broken lock would demonstrably overlap.
                    std::thread::sleep(std::time::Duration::from_millis(5));
                    inside.fetch_sub(1, Ordering::SeqCst);
                })
            })
            .collect();
        for t in threads {
            t.join().expect("tier thread");
        }
        assert_eq!(peak.load(Ordering::SeqCst), 1);
        assert_eq!(inside.load(Ordering::SeqCst), 0);
    }

    #[test]
    fn a_panicking_holder_does_not_poison_the_tier() {
        // One genuine failure must not cascade into N spurious ones — see the `CLUSTER_TIER` doc.
        let panicked = without_panic_noise(|| {
            std::thread::spawn(|| {
                let _tier = cluster_tier();
                panic!("a test failed while holding the tier");
            })
            .join()
        });
        assert!(panicked.is_err(), "the holder thread must have panicked");
        // The tier is still usable afterwards.
        let _tier = cluster_tier();
        assert!(tier_held_by_current_thread());
    }

    #[test]
    fn the_tier_is_still_held_while_a_later_declared_local_drops() {
        // WHY THIS MATTERS: a `Cluster`'s drop is what kills AND reaps its children, and the reap is
        // what actually frees the ports. Declaring the guard FIRST (so it drops LAST) is what keeps
        // the tier held across that reap. This asserts the convention really has that effect.
        struct Probe(Arc<AtomicBool>);
        impl Drop for Probe {
            fn drop(&mut self) {
                self.0
                    .store(tier_held_by_current_thread(), Ordering::SeqCst);
            }
        }

        let held_at_drop = Arc::new(AtomicBool::new(false));
        {
            let _tier = cluster_tier(); // FIRST  ⇒ drops LAST
            let _probe = Probe(Arc::clone(&held_at_drop)); // SECOND ⇒ drops FIRST
        }
        assert!(held_at_drop.load(Ordering::SeqCst));
    }

    #[test]
    fn declaring_the_guard_last_releases_the_tier_before_the_reap() {
        // The NEGATIVE twin of the test above — the mistake the doc warns against. Keeping both
        // arms makes the ordering rule an asserted property rather than a comment.
        struct Probe(Arc<AtomicBool>);
        impl Drop for Probe {
            fn drop(&mut self) {
                self.0
                    .store(tier_held_by_current_thread(), Ordering::SeqCst);
            }
        }

        let held_at_drop = Arc::new(AtomicBool::new(true));
        {
            let _probe = Probe(Arc::clone(&held_at_drop)); // FIRST  ⇒ drops LAST
            let _tier = cluster_tier(); // SECOND ⇒ drops FIRST
        }
        assert!(!held_at_drop.load(Ordering::SeqCst));
    }

    #[test]
    fn spawn_node_without_the_tier_guard_fails_loud() {
        // The tripwire fires BEFORE `Command::spawn`, so a path that does not exist is fine —
        // reaching an OS error would itself mean the assert did not fire.
        assert!(!tier_held_by_current_thread());
        let caught = without_panic_noise(|| {
            std::panic::catch_unwind(|| {
                let _ = spawn_node("/nonexistent/vd-shard", &[], &[]);
            })
        });
        let payload = caught.expect_err("spawn_node must panic without the tier");
        let msg = payload
            .downcast_ref::<String>()
            .map_or("<not a String>", String::as_str);
        assert!(
            msg.contains("cluster_tier"),
            "the panic must name the remedy; got: {msg}"
        );
        assert_eq!(msg, TIER_REMEDY);
    }

    // ---- the source-scanning completeness test --------------------------------------------------

    /// Every call that ends up forking a real node binary. `spawn_node` is tripwired at runtime, but
    /// the other three are not reachable by that assert — a test can shell out to the launcher, go
    /// through the grouped spawner, or drive the RLM process backend — so the only way to prove
    /// COMPLETE coverage is to read the sources.
    const BOOT_CALLS: [&str; 5] = [
        "Cluster::new(",
        "spawn_node(",
        "spawn_node_grouped(",
        "ProcLaunchBackend::new(",
        "devcluster(",
    ];

    /// `(name, is_test, body)` for every `fn` with a body in `src`.
    fn functions(src: &str) -> Vec<(String, bool, String)> {
        let bytes = src.as_bytes();
        let mut out = Vec::new();
        for (idx, _) in src.match_indices("fn ") {
            // A real definition is preceded by whitespace or a keyword boundary, and the name is an
            // identifier followed by `(` or `<`.
            let after = &src[idx + 3..];
            let name: String = after
                .chars()
                .take_while(|c| c.is_alphanumeric() || *c == '_')
                .collect();
            if name.is_empty() {
                continue;
            }
            let Some(rel) = src[idx..].find('{') else {
                continue;
            };
            let open = idx + rel;
            // Reject `fn` inside a string/comment cheaply: the char before must not be alphanumeric.
            if idx > 0 && (bytes[idx - 1].is_ascii_alphanumeric() || bytes[idx - 1] == b'_') {
                continue;
            }
            let mut depth = 0usize;
            let mut end = open;
            for (i, c) in src[open..].char_indices() {
                if c == '{' {
                    depth += 1;
                } else if c == '}' {
                    depth -= 1;
                    if depth == 0 {
                        end = open + i;
                        break;
                    }
                }
            }
            // `#[test]` (or `#[tokio::test]`) within the attributes just above the fn. Step back by
            // CHARS, not bytes — these sources contain multi-byte punctuation and a byte offset can
            // land mid-character.
            let start = src[..idx]
                .char_indices()
                .rev()
                .nth(200)
                .map_or(0, |(i, _)| i);
            let is_test = src[start..idx].contains("test]");
            out.push((name, is_test, src[open..=end].to_owned()));
        }
        out
    }

    /// Does `body` call `callee`? Substring plus a `(` is enough here: these are distinctive
    /// snake_case fixture names, and a false POSITIVE only adds a guard that is harmless.
    fn calls(body: &str, callee: &str) -> bool {
        body.contains(&format!("{callee}("))
    }

    #[test]
    fn every_cluster_booting_test_holds_the_tier() {
        let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tests");
        let mut offenders: Vec<String> = Vec::new();
        let mut checked = 0usize;
        let mut guarded = 0usize;

        let mut entries: Vec<_> = std::fs::read_dir(&dir)
            .unwrap_or_else(|e| panic!("read {}: {e}", dir.display()))
            .map(|e| e.expect("dir entry").path())
            .filter(|p| p.extension().is_some_and(|x| x == "rs"))
            .collect();
        entries.sort();
        assert!(!entries.is_empty(), "no process-tier test sources found");

        // ANTI-VACUITY, per file: if the parser ever regresses it would find nothing and the test
        // would pass while measuring nothing. A file that NAMES a boot call must yield at least one
        // booting test. Self-calibrating — no hard-coded count to drift.
        let mut silent_files: Vec<String> = Vec::new();

        for path in entries {
            let src = std::fs::read_to_string(&path).expect("read test source");
            let fns = functions(&src);
            let file_names_a_boot = BOOT_CALLS.iter().any(|c| src.contains(c));
            let mut found_here = 0usize;

            // Seed: functions that name a boot call directly.
            let mut boots: Vec<String> = fns
                .iter()
                .filter(|(_, _, body)| BOOT_CALLS.iter().any(|c| body.contains(c)))
                .map(|(n, _, _)| n.clone())
                .collect();
            // Close transitively: a helper that calls a booting helper also boots. This is the
            // bypass class the runtime tripwire structurally cannot see.
            loop {
                let before = boots.len();
                for (name, _, body) in &fns {
                    if !boots.contains(name) && boots.iter().any(|b| calls(body, b)) {
                        boots.push(name.clone());
                    }
                }
                if boots.len() == before {
                    break;
                }
            }

            for (name, is_test, body) in &fns {
                if !is_test || !boots.contains(name) {
                    continue;
                }
                checked += 1;
                found_here += 1;
                if body.contains("cluster_tier()") {
                    guarded += 1;
                } else {
                    offenders.push(format!(
                        "{}::{name}",
                        path.file_name().unwrap_or_default().to_string_lossy()
                    ));
                }
            }

            if file_names_a_boot && found_here == 0 {
                silent_files.push(
                    path.file_name()
                        .unwrap_or_default()
                        .to_string_lossy()
                        .into(),
                );
            }
        }

        assert!(
            checked > 0,
            "the scanner found no cluster-booting tests — it has stopped measuring anything"
        );
        assert_eq!(
            silent_files,
            Vec::<String>::new(),
            "these files name a cluster boot call but the scanner attributed it to no test — \
             the fn/attribute parser has regressed and is no longer measuring them"
        );
        assert_eq!(guarded, checked);
        assert_eq!(
            offenders,
            Vec::<String>::new(),
            "these tests boot a real cluster without the process tier. {TIER_REMEDY}"
        );
    }
}

#[cfg(test)]
mod world_roster_tests {
    use super::*;

    // (The per-case TempDir helper died with the deleted scene emitter's test — Slice C1,
    // D-LANE-6 🟩: nothing in this module writes a file any more.)

    #[test]
    fn child_luma_bags_cover_exactly_the_held_realms_direct_system_children() {
        // THE WINDOW LANE's boot plumbing (Slice A + C1): the marker roster a shard boots with
        // holds a TAG_LUMA bag for EXACTLY the direct children of its held realms that carry a
        // draw — on THE world, the galaxy shard gets its three systems (the stellar draws) and a
        // system shard gets its planets (the Slice-C1 REFLECTED draws).
        let world = boot_world(DEV.universe_seed, DEV.move_speed, DEV.tick_dt);
        let regions = world.regions();
        // Named via the lineage, never a seed literal: a planet's parent IS a star system, and
        // that system's parent IS the galaxy (the ambient Universe/Galaxy shells are System-tagged
        // shells too, so kind-matching alone cannot pick a star system out).
        let a_planet = regions
            .iter()
            .find(|r| matches!(r.realm, vd_core::pose::RealmId::Planet(_)))
            .expect("THE world holds planets");
        let a_system = a_planet.parent.expect("a planet nests under its system");
        let galaxy = regions
            .iter()
            .find(|r| r.realm == a_system)
            .and_then(|r| r.parent)
            .expect("a system nests under the galaxy");
        let bags = child_luma_bags(
            DEV.universe_seed,
            DEV.move_speed,
            DEV.tick_dt,
            regions,
            &std::collections::BTreeSet::from([galaxy]),
        );
        assert_eq!(bags.len(), 3, "one marker bag per system of THE world");
        for (realm, bag) in &bags {
            assert!(matches!(realm, vd_core::pose::RealmId::System(_)));
            vd_core::look::luma_of(bag).expect("a well-formed TAG_LUMA bag");
        }
        // A SYSTEM shard's roster (Slice C1): its planets carry the REFLECTED marker draw
        // (window_lane.md §1.1 item 3b, per direct child — the flag day made it load-bearing:
        // a sleeping realm appears ONLY as its parent's marker).
        let planets = child_luma_bags(
            DEV.universe_seed,
            DEV.move_speed,
            DEV.tick_dt,
            regions,
            &std::collections::BTreeSet::from([a_system]),
        );
        let expected_planets = regions
            .iter()
            .filter(|r| r.parent == Some(a_system))
            .count();
        assert_eq!(
            planets.len(),
            expected_planets,
            "one reflected marker bag per planet of the held system"
        );
        for (realm, bag) in &planets {
            assert!(matches!(realm, vd_core::pose::RealmId::Planet(_)));
            vd_core::look::luma_of(bag).expect("a well-formed TAG_LUMA bag");
        }
    }

    #[test]
    fn world_roster_derives_every_name_from_the_world_and_the_corridor_margins_hold() {
        // `world_roster` itself asserts I-AXIS/I-POLE/I-RADIAL/J1 — reaching the assertions below
        // means THE world passed them. This test pins the STRUCTURE of the derivation (lineage
        // positions, never seeds — a seed literal here would be a second place naming a realm).
        let roster = world_roster(&DEV);
        let world = boot_world(DEV.universe_seed, DEV.move_speed, DEV.tick_dt);
        for realm in [roster.home, roster.galaxy, roster.inner, roster.sibling] {
            assert!(
                world.contains_realm(realm),
                "{realm} must be a realm THE world contains",
            );
        }
        // Lineage: home nests under galaxy; the sibling is a DISTINCT system beside home; the inner
        // mover is one of the home shard's own authored movers, at the smallest semi-major axis.
        let region_parent = |realm| {
            world
                .regions()
                .iter()
                .find(|r| r.realm == realm)
                .and_then(|r| r.parent)
        };
        assert_eq!(region_parent(roster.home), Some(roster.galaxy));
        assert_eq!(region_parent(roster.sibling), Some(roster.galaxy));
        assert_ne!(roster.sibling, roster.home);
        assert_eq!(region_parent(roster.inner), Some(roster.home));
        let held = std::collections::BTreeSet::from([roster.home]);
        let (_, movers) = boot_regions_and_movers(
            DEV.universe_seed,
            &held,
            roster.home,
            DEV.move_speed,
            DEV.tick_dt,
        );
        assert_eq!(movers.get(&roster.inner), Some(&roster.inner_elements));
        for elements in movers.values() {
            assert!(
                roster.inner_elements.sma <= elements.sma,
                "the inner mover has the smallest semi-major axis",
            );
        }
        // The margins are real distances, not degenerate zeros — and PRINTED, so a world change
        // shows its measured numbers in the gate log rather than only a pass/fail.
        assert!(roster.axis_clearance_m > 0.0 && roster.axis_clearance_m.is_finite());
        assert!(roster.pole_altitude_m > 0.0 && roster.pole_altitude_m.is_finite());
        assert!(roster.radial_gap_m > 0.0 && roster.radial_gap_m.is_finite());
        assert!(roster.sibling_centre.length() > 0.0);
        eprintln!(
            "[world-roster] home {} | galaxy {} | inner {} | sibling {} at {:.1} m; axis clearance \
             {:.2} m, pole altitude {:.2} m, radial gap {:.2} m",
            roster.home,
            roster.galaxy,
            roster.inner,
            roster.sibling,
            roster.sibling_centre.length(),
            roster.axis_clearance_m,
            roster.pole_altitude_m,
            roster.radial_gap_m,
        );
    }
}

#[cfg(test)]
mod incarnation_tests {
    use super::*;
    use std::collections::BTreeMap;
    use vd_core::pose::RealmId;

    fn env(pairs: &[(&str, &str)]) -> EnvConfig {
        EnvConfig::new(
            pairs
                .iter()
                .map(|(k, v)| ((*k).to_owned(), (*v).to_owned()))
                .collect::<BTreeMap<_, _>>(),
        )
    }

    // ---- RLM 5f RG-2: the reactive-greeting boot policy -----------------------------------------
    #[test]
    fn resolve_presence_is_none_for_a_static_shard() {
        // No VD_INCARNATION_COOKIE ⇒ a static shard ⇒ no greeting policy (byte-identical boot).
        let e = env(&[("VD_PEERS", "2=127.0.0.1:9000")]);
        assert!(resolve_presence(&e, 0.02).expect("ok").is_none());
    }

    #[test]
    fn resolve_presence_greets_every_booked_peer_at_the_scaled_default() {
        let e = env(&[
            ("VD_INCARNATION_COOKIE", "abc123"),
            (
                "VD_PEERS",
                "2=127.0.0.1:9000,3=127.0.0.1:9001,30=127.0.0.1:9002",
            ),
        ]);
        let p = resolve_presence(&e, 0.02)
            .expect("ok")
            .expect("a demand shard greets");
        // Greets the FULL booked set (ancestors + orchestrator + gateway), not the gateway alone.
        assert_eq!(
            p.peers,
            std::collections::BTreeSet::from([
                vd_core::NodeId(2),
                vd_core::NodeId(3),
                vd_core::NodeId(30),
            ]),
        );
        // Default cadence scales with the tick rate: PRESENCE_REANNOUNCE_SECS (1.0s) / 0.02s = 50 ticks.
        assert_eq!(p.interval_ticks, 50);
    }

    #[test]
    fn resolve_presence_honors_the_tick_override() {
        let e = env(&[
            ("VD_INCARNATION_COOKIE", "abc"),
            ("VD_PEERS", "2=127.0.0.1:9000"),
            ("VD_PRESENCE_INTERVAL_TICKS", "77"),
        ]);
        assert_eq!(
            resolve_presence(&e, 0.02)
                .expect("ok")
                .expect("some")
                .interval_ticks,
            77,
        );
    }

    #[test]
    fn resolve_presence_rejects_a_zero_interval_override() {
        let e = env(&[
            ("VD_INCARNATION_COOKIE", "abc"),
            ("VD_PEERS", "2=127.0.0.1:9000"),
            ("VD_PRESENCE_INTERVAL_TICKS", "0"),
        ]);
        assert!(resolve_presence(&e, 0.02).is_err());
    }

    // ---- RLM RG-4a4: the admin HTTP bind resolver -----------------------------------------------
    #[test]
    fn resolve_admin_is_none_when_unset_or_empty() {
        // Unset ⇒ None (byte-identical: no admin server binds). An explicitly EMPTY value is also None
        // (a ConfigMap that sets VD_ADMIN_ADDR="" must not try to bind "").
        assert!(resolve_admin(&env(&[])).expect("ok").is_none());
        assert!(
            resolve_admin(&env(&[("VD_ADMIN_ADDR", "   ")]))
                .expect("ok")
                .is_none()
        );
    }

    #[test]
    fn resolve_admin_parses_a_socket_addr() {
        let a = resolve_admin(&env(&[("VD_ADMIN_ADDR", "127.0.0.1:9099")]))
            .expect("ok")
            .expect("some");
        assert_eq!(a, "127.0.0.1:9099".parse().expect("addr"));
    }

    #[test]
    fn resolve_admin_rejects_a_malformed_addr() {
        assert!(resolve_admin(&env(&[("VD_ADMIN_ADDR", "not-an-addr")])).is_err());
    }

    #[test]
    fn validate_tick_pair_accepts_matched_rejects_drift() {
        // Matched: 50 Hz ⇔ 0.02 s.
        assert_eq!(validate_tick_pair(50, 0.02), Ok(()));
        assert_eq!(validate_tick_pair(20, 0.05), Ok(()));
        assert_eq!(validate_tick_pair(10, 0.1), Ok(()));
        // Drift: 50 Hz but a 20 Hz dt (the ConfigMap-retune-one-not-the-other footgun) → loud.
        assert_eq!(
            validate_tick_pair(50, 0.05),
            Err(TickPairError {
                tick_hz: 50,
                tick_dt: 0.05,
                expected: 0.02
            })
        );
        // Within epsilon → accepted (float slack).
        assert_eq!(validate_tick_pair(50, 0.02 + TICK_DT_EPSILON / 2.0), Ok(()));
        // Degenerate tick_hz=0 floors to 1 (⇔ dt 1.0), mirroring TickPacer.
        assert_eq!(validate_tick_pair(0, 1.0), Ok(()));
    }

    #[test]
    fn validate_drain_burst_accepts_the_floor_rejects_overshoot() {
        // The shipped DEV pair is within the dest inbox floor (the runtime twin of the const assert).
        assert_eq!(
            validate_drain_burst(DEV.outbound_cap as usize, DEV.max_buffered_inputs as usize),
            Ok(())
        );
        let floor = vd_io_prod::mesh::inbound_capacity_for(DEV.outbound_cap as usize);
        // A ConfigMap lowering VD_OUTBOUND_CAP (shrinking the floor) without the gateway's drain burst is the
        // D-8 silent-shed footgun → loud (usize::MAX inputs overshoot ANY floor).
        assert_eq!(
            validate_drain_burst(DEV.outbound_cap as usize, usize::MAX),
            Err(DrainBurstError {
                outbound_cap: DEV.outbound_cap as usize,
                max_buffered_inputs: usize::MAX,
                floor,
            })
        );
        // Exactly AT the floor is accepted — the boundary is inclusive (`>` floor is the reject).
        assert_eq!(
            validate_drain_burst(DEV.outbound_cap as usize, floor),
            Ok(())
        );
    }

    #[test]
    fn auth_keypair_from_seed_is_deterministic_and_matches_dev() {
        // The dev seed reproduces the dev key exactly (the derivation is the same SigningKey path).
        let (pubkey, signing) = auth_keypair_hex_from_seed(&DEV_AUTH_SEED);
        assert_eq!(pubkey, dev_auth_pubkey_hex());
        assert_eq!(signing, dev_auth_signing_key_hex());
        // A different seed → a different verifying key (a real gen-authkey key is NOT the vetoed dev key).
        let (other_pubkey, other_signing) = auth_keypair_hex_from_seed(&[0x01u8; 32]);
        assert_ne!(other_pubkey, pubkey);
        assert_ne!(other_signing, signing);
    }

    #[test]
    fn parse_bool_env_ladder() {
        assert!(parse_bool_env(&env(&[("K", "1")]), "K").expect("1"));
        assert!(parse_bool_env(&env(&[("K", "TRUE")]), "K").expect("TRUE"));
        assert!(parse_bool_env(&env(&[("K", "yes")]), "K").expect("yes"));
        assert!(!parse_bool_env(&env(&[("K", "0")]), "K").expect("0"));
        assert!(!parse_bool_env(&env(&[("K", "false")]), "K").expect("false"));
        assert!(!parse_bool_env(&env(&[("K", "")]), "K").expect("empty"));
        assert!(!parse_bool_env(&env(&[]), "K").expect("absent = false"));
        assert!(
            parse_bool_env(&env(&[("K", "maybe")]), "K").is_err(),
            "a non-boolean is loud"
        );
    }

    #[test]
    fn resolve_prefers_an_explicit_incarnation() {
        assert_eq!(
            resolve_process_incarnation(&env(&[("VD_PROCESS_INCARNATION", "42")]))
                .expect("explicit wins"),
            42
        );
        // Explicit-but-unparseable is a loud error (not a silent fall-through).
        assert!(resolve_process_incarnation(&env(&[("VD_PROCESS_INCARNATION", "nope")])).is_err());
    }

    #[test]
    fn resolve_uses_the_durable_counter_and_increments() {
        let dir = std::env::temp_dir().join(format!("vd-resolve-test-{}", std::process::id()));
        std::fs::create_dir_all(&dir).expect("scratch");
        let _ = std::fs::remove_file(dir.join(BOOT_COUNTER_NAME));
        // The scratch dir is under temp ⇒ the deny-list would reject it; EPHEMERAL_OK=1 is the test escape.
        let e = env(&[
            ("VD_BOOT_STATE_DIR", dir.to_str().expect("utf8")),
            ("VD_BOOT_STATE_EPHEMERAL_OK", "1"),
        ]);
        let first = resolve_process_incarnation(&e).expect("boot1");
        let second = resolve_process_incarnation(&e).expect("boot2");
        assert!(first >= 1);
        assert_eq!(
            second,
            first + 1,
            "the durable counter increments monotonically"
        );
    }

    #[test]
    fn an_empty_durable_root_does_not_fail_open_the_guard() {
        // An empty VD_BOOT_DURABLE_ROOT (an `export X=$UNSET` mistake) must be treated as ABSENT — NOT
        // flip the allow-list into "root='' matches every path". A temp VD_BOOT_STATE_DIR must still be
        // rejected by the deny-list (no ephemeral escape).
        let temp_dir = std::env::temp_dir().join("vd-empty-root-test");
        let e = env(&[
            ("VD_BOOT_STATE_DIR", temp_dir.to_str().expect("utf8")),
            ("VD_BOOT_DURABLE_ROOT", ""),
        ]);
        assert!(
            resolve_process_incarnation(&e).is_err(),
            "an empty durable root must not disable the temp deny-list"
        );
    }

    #[test]
    fn resolve_fails_loud_when_no_durable_source() {
        // Neither VD_PROCESS_INCARNATION nor VD_BOOT_STATE_DIR, no ephemeral escape ⇒ refuse to boot.
        assert!(resolve_process_incarnation(&env(&[])).is_err());
        // With the explicit ephemeral escape, fall back to the wall-clock value (> 0).
        assert!(
            resolve_process_incarnation(&env(&[("VD_BOOT_STATE_EPHEMERAL_OK", "1")]))
                .expect("escape")
                > 0
        );
    }

    /// A unique temp dir per case, removed on drop (the success cases open a real redb file).
    struct TempDir(std::path::PathBuf);
    impl TempDir {
        fn new(tag: &str) -> TempDir {
            TempDir(std::env::temp_dir().join(format!(
                "vd-obx-{tag}-{}-{:p}",
                std::process::id(),
                &tag
            )))
        }
        fn file(&self, name: &str) -> String {
            self.0.join(name).display().to_string()
        }
    }
    impl Drop for TempDir {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }

    #[test]
    fn open_node_outbox_absent_or_blank_path_is_none() {
        assert!(open_node_outbox(&env(&[])).expect("absent ok").is_none());
        assert!(
            open_node_outbox(&env(&[("VD_OUTBOX_PATH", "   ")]))
                .expect("blank ok")
                .is_none(),
            "a blank path is treated as absent (an unset-var expansion must not open a CWD store)"
        );
    }

    #[test]
    fn open_node_outbox_refuses_a_temp_path_without_the_escape() {
        let d = TempDir::new("reject");
        // NodeOutbox is not Debug, so match rather than expect_err (which would need Ok: Debug).
        let e = match open_node_outbox(&env(&[("VD_OUTBOX_PATH", &d.file("out.redb"))])) {
            Err(e) => e,
            Ok(_) => panic!("a temp path with no durable-root + no escape must be refused"),
        };
        assert!(
            e.to_string().contains("temp"),
            "the refusal names the temp hazard: {e}"
        );
    }

    #[test]
    fn open_node_outbox_opens_under_the_ephemeral_escape() {
        let d = TempDir::new("escape");
        let path = d.file("out.redb");
        let ob = open_node_outbox(&env(&[
            ("VD_OUTBOX_PATH", &path),
            ("VD_OUTBOX_EPHEMERAL_OK", "1"),
        ]))
        .expect("opens with the ephemeral escape");
        assert!(ob.is_some(), "a Some outbox on a valid escape path");
        assert!(
            std::path::Path::new(&path).exists(),
            "the redb file was created"
        );
    }

    #[test]
    fn open_node_outbox_allow_lists_a_path_under_the_declared_root() {
        // F2: a declared VD_STORE_DURABLE_ROOT admits a path UNDER it (the cloud PV case) even under $TMPDIR,
        // where the deny-list would reject it; a path OUTSIDE the root is still refused.
        let d = TempDir::new("root");
        let root = d.0.display().to_string();
        let inside = open_node_outbox(&env(&[
            ("VD_OUTBOX_PATH", &d.file("sub/out.redb")),
            ("VD_STORE_DURABLE_ROOT", &root),
        ]))
        .expect("a path under the declared root is admitted");
        assert!(inside.is_some());

        let outside = TempDir::new("outside");
        let e = match open_node_outbox(&env(&[
            ("VD_OUTBOX_PATH", &outside.file("x.redb")),
            ("VD_STORE_DURABLE_ROOT", &root),
        ])) {
            Err(e) => e,
            Ok(_) => panic!("a path outside the declared root must be refused"),
        };
        assert!(
            e.to_string().contains("durable root"),
            "names the root violation: {e}"
        );
    }

    // ---- RLM 5f: the spawned-child anchor env ----------------------------------------------------

    #[test]
    fn spawn_anchors_forward_the_5f4_keys_and_stay_byte_identical_when_absent() {
        // Absent 5f-4 keys ⇒ EXACTLY the pre-5f-4 anchor set (byte-identical spawned-child env), and each
        // present key is carried in list order.
        let base = env(&[
            ("VD_TRUST_DIR", "/t"),
            ("VD_TICK_HZ", "50"),
            ("VD_UNIVERSE_SEED", "7"),
        ]);
        assert_eq!(
            spawn_anchors_from_env(&base),
            vec![
                ("VD_TRUST_DIR", "/t".to_owned()),
                ("VD_TICK_HZ", "50".to_owned()),
                ("VD_UNIVERSE_SEED", "7".to_owned()),
            ],
            "only the present pre-5f-4 keys, in list order — the 5f-4 keys absent ⇒ byte-identical"
        );
        // The 5f-4 keys are forwarded when present, in list order (after the earlier keys).
        let armed = env(&[
            ("VD_TICK_HZ", "50"),
            ("VD_BOOT_TICKS_P99", "100"),
            // Set, and deliberately NOT forwarded: a spawned shard cannot read a universe-absolute pose.
            ("VD_SPAWN_POSES", "5=1,2,3"),
            ("VD_LEASE_RENEW_INTERVAL", "25"),
        ]);
        assert_eq!(
            spawn_anchors_from_env(&armed),
            vec![
                ("VD_TICK_HZ", "50".to_owned()),
                ("VD_BOOT_TICKS_P99", "100".to_owned()),
                ("VD_LEASE_RENEW_INTERVAL", "25".to_owned()),
            ],
            "the 5f-4 keys reach the spawned shard so its predictive AoI and lease liveness live — and the \
             universe-absolute spawn map does NOT, because no realm can read one"
        );
        // An empty value is dropped, never forwarded as an empty string.
        assert!(
            spawn_anchors_from_env(&env(&[("VD_BOOT_TICKS_P99", "")])).is_empty(),
            "an empty value is not forwarded"
        );
    }

    #[test]
    fn the_handoff_budget_reaches_a_spawned_shard_and_matches_the_arrival_shield() {
        // The two ends of one hand-off are handed the SAME duration — not by two derivations that happen
        // to agree today, but by reading one. This test fails the day someone gives the source its own.
        let rlm = vd_sim::rlm::RlmTuning::cloud(50);
        let saga = vd_sim::saga::SagaTuning {
            redrive_deadline_ticks: 40,
            abort_deadline_ticks: 200,
        };
        let shield = vd_sim::rlm::derive_arrival_shield_ticks(&rlm, &saga);
        assert_eq!(
            handoff_hold_anchor(true, &rlm, &saga),
            Some(("VD_HANDOFF_HOLD_TICKS", shield.to_string())),
            "a demand orchestrator hands its shards the arrival-shield duration verbatim"
        );
        // The number is the real derivation, not an accidental zero that would make the equality vacuous.
        assert!(shield > 0, "the derived budget is a real window: {shield}");
        // An INERT reconciler hands down nothing, so the child parses no key, defaults to zero, and lets
        // go of a departing occupant the instant it is told to — exactly as before the ledger existed.
        assert_eq!(handoff_hold_anchor(false, &rlm, &saga), None);
    }

    #[test]
    fn the_crossing_redrive_pair_reaches_every_shard_and_matches_the_derivations() {
        // D-WORLD-2: the launcher hands its shards the SAME ttl/budget the vd-sim derivations state —
        // one derivation, read at both launch modes. Fails the day someone plants a literal.
        let saga = vd_sim::saga::SagaTuning {
            redrive_deadline_ticks: 40,
            abort_deadline_ticks: 200,
        };
        assert_eq!(
            crossing_redrive_env(&saga),
            [
                (
                    "VD_CROSSING_TTL_TICKS",
                    vd_sim::saga::derive_request_ttl_ticks(&saga).to_string(),
                ),
                (
                    "VD_CROSSING_REDRIVE_BUDGET",
                    vd_sim::saga::derive_crossing_redrive_budget(&saga).to_string(),
                ),
            ],
        );
        // Every STATIC cluster's shared env carries the pair (the walk gate's cluster is static — the
        // exact rig whose unhosted-planet graze stranded the dot before the cure).
        let common = common_env("trust", &DEV);
        let pair = crossing_redrive_env(&vd_sim::saga::SagaTuning::default());
        for (k, v) in pair {
            assert_eq!(
                common.iter().find(|(ck, _)| *ck == k).map(|(_, cv)| cv),
                Some(&v),
                "common_env must carry {k} at the derived value",
            );
        }
        // The defaults derive to a REAL armed posture (never a vacuous zero).
        let armed = vd_sim::saga::derive_request_ttl_ticks(&vd_sim::saga::SagaTuning::default());
        assert!(armed > 0, "the derived ttl is a real window: {armed}");
    }

    #[test]
    fn resolve_time_multiplier_defaults_reads_env_override_and_is_loud_on_bad() {
        // ABSENT / empty ⇒ 1.0 (universe rate, byte-identical).
        assert_eq!(resolve_time_multiplier(&env(&[])), Ok(1.0));
        assert_eq!(
            resolve_time_multiplier(&env(&[("VD_TIME_MULTIPLIER", "")])),
            Ok(1.0),
        );
        // The orchestrator-wide default is read + trimmed.
        assert_eq!(
            resolve_time_multiplier(&env(&[("VD_TIME_MULTIPLIER", "  0.5 ")])),
            Ok(0.5),
        );
        // The per-realm override WINS over the global.
        assert_eq!(
            resolve_time_multiplier(&env(&[
                ("VD_TIME_MULTIPLIER", "0.5"),
                ("VD_REALM_TIME_MULTIPLIER", "2.0"),
            ])),
            Ok(2.0),
        );
        // Non-numeric / zero / negative / non-finite all fail LOUD (time moves forward at a finite rate).
        assert!(resolve_time_multiplier(&env(&[("VD_TIME_MULTIPLIER", "fast")])).is_err());
        assert!(resolve_time_multiplier(&env(&[("VD_TIME_MULTIPLIER", "0")])).is_err());
        assert!(resolve_time_multiplier(&env(&[("VD_TIME_MULTIPLIER", "-1")])).is_err());
        assert!(resolve_time_multiplier(&env(&[("VD_TIME_MULTIPLIER", "inf")])).is_err());
    }

    #[test]
    fn resolve_spawn_poses_is_empty_by_default_parses_entries_and_is_loud_on_bad() {
        use vd_core::glam::DVec3;
        use vd_core::pose::{FrameRef, StampedPose};
        // ABSENT / empty ⇒ EMPTY map (every login origin-at-rest, byte-identical).
        assert_eq!(
            resolve_spawn_poses(&env(&[])),
            Ok(std::collections::BTreeMap::new())
        );
        assert_eq!(
            resolve_spawn_poses(&env(&[("VD_SPAWN_POSES", "  ")])),
            Ok(std::collections::BTreeMap::new()),
        );
        // A well-formed multi-entry value parses to at-rest ABSOLUTE poses in the Universe-root frame.
        let got = resolve_spawn_poses(&env(&[(
            "VD_SPAWN_POSES",
            " 5=11,-22,33 ; 7 = 1.5, 2.5, 3.5 ",
        )]))
        .expect("well-formed entries parse");
        let mut want = std::collections::BTreeMap::new();
        want.insert(
            vd_core::AccountId(5),
            StampedPose::at_rest(
                FrameRef::SystemSpace { system_seed: 0 },
                DVec3::new(11.0, -22.0, 33.0),
                vd_core::UniverseTick(0),
            ),
        );
        want.insert(
            vd_core::AccountId(7),
            StampedPose::at_rest(
                FrameRef::SystemSpace { system_seed: 0 },
                DVec3::new(1.5, 2.5, 3.5),
                vd_core::UniverseTick(0),
            ),
        );
        assert_eq!(got, want);
        // Malformed entries all fail LOUD: missing '=', a non-numeric account, wrong coord count, bad float.
        assert!(resolve_spawn_poses(&env(&[("VD_SPAWN_POSES", "5,1,2,3")])).is_err());
        assert!(resolve_spawn_poses(&env(&[("VD_SPAWN_POSES", "abc=1,2,3")])).is_err());
        assert!(resolve_spawn_poses(&env(&[("VD_SPAWN_POSES", "5=1,2")])).is_err());
        assert!(resolve_spawn_poses(&env(&[("VD_SPAWN_POSES", "5=1,2,x")])).is_err());
    }

    #[test]
    fn the_boot_builds_the_one_world_and_nothing_can_select_another() {
        use std::collections::BTreeSet;
        // ONE WORLD, ONE TEST. There used to be one of these PER SCALE, which is the defect's own shape
        // rather than a proof against it — a live cluster was measured running two scales at once, the
        // orchestrator on one and its gateway on another. Nothing selects a universe now, so what is left
        // to pin is the property that matters: the boot's REGIONS and its MOVER ROSTER come from the same
        // world, and therefore cannot describe different universes to the two halves that read them.
        let hosted = RealmId::System(7);
        let held: BTreeSet<RealmId> = [hosted].into_iter().collect();
        let (regions, moving) = boot_regions_and_movers(0, &held, hosted, 15.0, 0.02);
        let world = vd_physics::worldgen::UniverseConfig::world(15.0, 0.02);
        assert_eq!(
            regions,
            vd_physics::worldgen::realm_neighbourhood_for_config(0, &held, &world),
            "the boot scopes to THE world's neighbourhood: own realm, ancestors, authored children",
        );
        let expected_movers: std::collections::BTreeMap<_, _> =
            vd_physics::worldgen::moving_children_for_config(0, &world, hosted)
                .into_iter()
                .collect();
        assert_eq!(
            moving, expected_movers,
            "and the movers come from that SAME world, so regions and orbits cannot disagree",
        );
    }

    // ---- Co-hosting: the VD_HELD_REALMS env round-trip + parse guards ---------------------------

    #[test]
    fn held_realms_env_round_trips_all_four_realm_kinds() {
        // Every RealmId kind serializes + re-parses (the `realm_token` / `parse_realm_token` arms). NO
        // launcher produces VD_HELD_REALMS since co-hosting retired with --triple (D-WORLD-6); the codec
        // keeps unit coverage so the shard-side parse cannot rot silently.
        let set = std::collections::BTreeSet::from([
            RealmId::System(7),
            RealmId::Planet(7),
            RealmId::Station(7),
            RealmId::Area(7),
        ]);
        let wire = held_realms_env(&set);
        // Sorted by `RealmId`'s derived Ord (variant declaration order Planet<System<Ship<Station<Area).
        assert_eq!(
            wire, "planet:7,system:7,station:7,area:7",
            "sorted kind:seed CSV"
        );
        assert_eq!(
            parse_held_realms(&wire, RealmId::System(7)).expect("round-trips"),
            set,
        );
        // A Station/Area/Planet token each survives the round-trip (covers all four `parse_realm_token`
        // match arms via a distinct-kind fallback).
        assert_eq!(
            parse_held_realms("area:42", RealmId::Area(42)).expect("area token"),
            std::collections::BTreeSet::from([RealmId::Area(42)]),
        );
        // The Ship arm (P8, never in a P3 rig) round-trips for codec totality — `ship:<u128>`.
        let ship = RealmId::Ship(vd_core::EntityId(0xDEAD_BEEF));
        let ship_set = std::collections::BTreeSet::from([ship]);
        assert_eq!(
            parse_held_realms(&held_realms_env(&ship_set), RealmId::System(7)),
            Ok(std::collections::BTreeSet::from([RealmId::System(7), ship])),
        );
        assert!(
            parse_held_realms("ship:notanumber", RealmId::System(7))
                .expect_err("a non-u128 ship id is loud")
                .contains("non-u128 ship id"),
        );
    }

    #[test]
    fn parse_held_realms_defaults_to_the_fallback_when_absent_or_empty() {
        // ABSENT / EMPTY ⇒ exactly `{fallback}` (the single-realm byte-identical default). Whitespace-only
        // and stray commas degrade to the same (the `filter(!is_empty)` arm).
        let fallback = RealmId::System(9);
        let only = std::collections::BTreeSet::from([fallback]);
        assert_eq!(parse_held_realms("", fallback).expect("empty"), only);
        assert_eq!(parse_held_realms("  ", fallback).expect("blank"), only);
        assert_eq!(parse_held_realms(" , ,", fallback).expect("commas"), only);
        // The fallback is ALWAYS included even if the env omits it (a shard always hosts its own realm).
        assert_eq!(
            parse_held_realms("planet:9", fallback).expect("adds fallback"),
            std::collections::BTreeSet::from([fallback, RealmId::Planet(9)]),
        );
    }

    #[test]
    fn parse_held_realms_is_loud_on_a_malformed_token() {
        // Every parse-error arm fails LOUD (a co-hosting misconfig must never silently degrade): no colon,
        // an unknown kind, a non-u64 seed.
        let fb = RealmId::System(7);
        assert!(
            parse_held_realms("system7", fb)
                .expect_err("no colon is loud")
                .contains("not `kind:seed`")
        );
        assert!(
            parse_held_realms("moon:7", fb)
                .expect_err("unknown kind is loud")
                .contains("unknown kind")
        );
        assert!(
            parse_held_realms("system:x", fb)
                .expect_err("non-u64 seed is loud")
                .contains("non-u64 seed")
        );
    }

    // ---- Track R / 1d.2: the dual-shard env builders + boundaries helper --------------------------

    /// Loopback addrs distinct per role so a mis-booked peer is visible in an assert. Every field is a
    /// unique port, so a `book(...)` mismatch shows up as a wrong port string.
    fn dual_addrs() -> ClusterAddrs {
        ClusterAddrs {
            orchestrator: loopback(9001),
            gateway: loopback(9002),
            shard: loopback(9003),
            admin: loopback(9004),
            gateway_admin: None,
            orchestrator_probe: loopback(9005),
            gateway_probe: loopback(9006),
            shard_probe: loopback(9007),
            shard_b: loopback(9008),
            shard_b_probe: loopback(9009),
            galaxy: loopback(9010),
            galaxy_probe: loopback(9011),
            planet: loopback(9012),
            planet_probe: loopback(9013),
            station: loopback(9014),
            station_probe: loopback(9015),
            area: loopback(9016),
            area_probe: loopback(9017),
        }
    }

    /// Look up a key's value in a rendered env vec (None if absent).
    fn env_value<'a>(env: &'a [(&'static str, String)], key: &str) -> Option<&'a str> {
        env.iter().find(|(k, _)| *k == key).map(|(_, v)| v.as_str())
    }

    // ---- RLM RG-4b: the DEMAND cluster env ------------------------------------------------------
    #[test]
    fn demand_orchestrator_env_arms_the_reconciler_and_books_no_shard() {
        let a = dual_addrs();
        let d = orchestrator_env(&a, &DEV, "store", ClusterShape::Demand);
        // ARMED, and NOT the static forest (the exact XOR misconfig the bin fail-loud-rejects).
        assert_eq!(env_value(&d, "VD_DEMAND"), Some("1"));
        assert_eq!(env_value(&d, "VD_STATIC_FOREST"), None);
        assert_eq!(env_value(&d, "VD_ROSTER"), None);
        // The gateway is the ONLY booked peer + clock follower — NO shard exists until demand spawns one.
        assert_eq!(
            env_value(&d, "VD_PEERS"),
            Some(format!("{}={}", GATEWAY.0, a.gateway)).as_deref()
        );
        assert_eq!(
            env_value(&d, "VD_CLOCK_PEERS"),
            Some(GATEWAY.0.to_string()).as_deref()
        );
        // It carries the seed + boot horizon + RLM port band + the 5 shard-spawn anchors its harvest forwards.
        assert_eq!(
            env_value(&d, "VD_UNIVERSE_SEED"),
            Some(DEV.universe_seed.to_string()).as_deref()
        );
        // NO WORLD KNOB IS EXPORTED, and that is now the property worth pinning. This used to assert the
        // cluster exported a scale — and a live cluster was measured with its orchestrator on one value
        // and its own gateway on another, in a single launch, because a knob that exists can be set twice.
        // Asserting its ABSENCE is the assertion that could have caught that; asserting its value never
        // could, because both halves were individually "correct".
        assert_eq!(
            env_value(&d, "VD_UNIVERSE_SCALE"),
            None,
            "there is one world; nothing selects it, so nothing may be handed a different one",
        );
        assert_eq!(
            env_value(&d, "VD_BOOT_TICKS_P99"),
            Some(DEV.boot_ticks_p99.to_string()).as_deref()
        );
        assert_eq!(
            env_value(&d, "VD_RLM_FIRST_PORT"),
            Some(RLM_DEMAND_FIRST_PORT.to_string()).as_deref()
        );
        assert_eq!(
            env_value(&d, "VD_RLM_PORT_LIMIT"),
            Some(RLM_DEMAND_PORT_LIMIT.to_string()).as_deref()
        );
        assert_eq!(
            env_value(&d, "VD_TICK_DT"),
            Some(DEV.tick_dt.to_string()).as_deref()
        );
        assert_eq!(
            env_value(&d, "VD_MINT_SEED"),
            Some(DEV.mint_seed.to_string()).as_deref()
        );
    }

    #[test]
    fn demand_gateway_env_arms_dynamic_home_and_books_no_shard() {
        let a = dual_addrs();
        let clients = [(NodeId(30), loopback(9100))];
        let g = gateway_env(&a, &clients, "pub", &DEV, ClusterShape::Demand);
        assert_eq!(env_value(&g, "VD_DEMAND"), Some("1"));
        // VD_SHARD stays the login-home DEFAULT id, but SHARD is NOT booked in VD_PEERS.
        assert_eq!(
            env_value(&g, "VD_SHARD"),
            Some(SHARD.0.to_string()).as_deref()
        );
        let peers = env_value(&g, "VD_PEERS").expect("peers");
        assert!(
            !peers.contains(&format!("{}={}", SHARD.0, a.shard)),
            "no static shard booked: {peers}"
        );
        assert!(
            peers.contains(&format!("{}={}", ORCH.0, a.orchestrator)),
            "the orchestrator is booked: {peers}"
        );
        assert!(
            peers.contains(&format!("{}={}", 30, loopback(9100))),
            "the dev-control client is booked: {peers}"
        );
        // No static-shard roster key (there are no static shards).
        assert_eq!(env_value(&g, "VD_KNOWN_SHARDS"), None);
    }

    #[test]
    fn demand_gateway_env_emits_admin_addr_only_when_booked() {
        let mut a = dual_addrs();
        assert_eq!(
            env_value(
                &gateway_env(&a, &[], "pub", &DEV, ClusterShape::Demand),
                "VD_ADMIN_ADDR"
            ),
            None
        );
        a.gateway_admin = Some(loopback(9099));
        assert_eq!(
            env_value(
                &gateway_env(&a, &[], "pub", &DEV, ClusterShape::Demand),
                "VD_ADMIN_ADDR"
            ),
            Some(loopback(9099).to_string()).as_deref()
        );
    }

    #[test]
    fn orchestrator_env_dual_books_the_galaxy_clock_and_roster_inert_when_single() {
        let a = dual_addrs();
        let single = orchestrator_env(&a, &DEV, "store", ClusterShape::Single);
        let dual = orchestrator_env(&a, &DEV, "store", ClusterShape::Dual);

        // VD_ROSTER: absent single, present=GALAXY dual (the D-37 re-home candidate set).
        assert_eq!(env_value(&single, "VD_ROSTER"), None);
        assert_eq!(
            env_value(&dual, "VD_ROSTER"),
            Some(GALAXY.0.to_string()).as_deref()
        );

        // VD_CLOCK_PEERS: {GW,SHARD} single vs {GW,SHARD,GALAXY} dual (the galaxy's follower clock
        // must advance or it never wins its realm lease).
        assert_eq!(
            env_value(&single, "VD_CLOCK_PEERS"),
            Some(format!("{},{}", GATEWAY.0, SHARD.0)).as_deref()
        );
        assert_eq!(
            env_value(&dual, "VD_CLOCK_PEERS"),
            Some(format!("{},{},{}", GATEWAY.0, SHARD.0, GALAXY.0)).as_deref()
        );

        // VD_PEERS: the galaxy is booked ONLY in dual (each initiator books it).
        let galaxy_book = format!("{}={}", GALAXY.0, a.galaxy);
        assert!(
            !env_value(&single, "VD_PEERS")
                .expect("VD_PEERS is always emitted")
                .contains(&galaxy_book)
        );
        assert!(
            env_value(&dual, "VD_PEERS")
                .expect("VD_PEERS is always emitted")
                .contains(&galaxy_book)
        );
    }

    #[test]
    fn orchestrator_env_chain_books_all_three_extras_in_roster_clock_and_peers() {
        // The Chain orchestrator drives the clock to EVERY extra realm-shard and lists each in the
        // re-home roster, so the galaxy (the between-space parent), the inner planet, and the sibling
        // star all win their leases and rest authority.
        let a = dual_addrs();
        let chain = orchestrator_env(&a, &DEV, "store", ClusterShape::Chain);

        // VD_ROSTER = {GALAXY, PLANET, SIBLING} in roster order.
        assert_eq!(
            env_value(&chain, "VD_ROSTER"),
            Some(format!("{},{},{}", GALAXY.0, PLANET_SHARD.0, SHARD_B.0)).as_deref()
        );
        // VD_CLOCK_PEERS = {GW, SHARD, GALAXY, PLANET, SIBLING} — every follower's clock must advance.
        assert_eq!(
            env_value(&chain, "VD_CLOCK_PEERS"),
            Some(format!(
                "{},{},{},{},{}",
                GATEWAY.0, SHARD.0, GALAXY.0, PLANET_SHARD.0, SHARD_B.0
            ))
            .as_deref()
        );
        // VD_PEERS books ALL THREE extra shards.
        let peers = env_value(&chain, "VD_PEERS").expect("VD_PEERS is always emitted");
        assert!(peers.contains(&format!("{}={}", GALAXY.0, a.galaxy)));
        assert!(peers.contains(&format!("{}={}", PLANET_SHARD.0, a.planet)));
        assert!(peers.contains(&format!("{}={}", SHARD_B.0, a.shard_b)));
    }

    #[test]
    fn gateway_env_dual_books_the_galaxy_and_emits_known_shards_inert_when_single() {
        let a = dual_addrs();
        let single = gateway_env(&a, &[], "pub", &DEV, ClusterShape::Single);
        let dual = gateway_env(&a, &[], "pub", &DEV, ClusterShape::Dual);

        // VD_KNOWN_SHARDS: absent single, =GALAXY dual (so a galaxy frame is node-class dispatchable).
        assert_eq!(env_value(&single, "VD_KNOWN_SHARDS"), None);
        assert_eq!(
            env_value(&dual, "VD_KNOWN_SHARDS"),
            Some(GALAXY.0.to_string()).as_deref()
        );

        // The galaxy peer is booked ONLY in dual; VD_SHARD (the login shard) stays SHARD in both.
        let galaxy_book = format!("{}={}", GALAXY.0, a.galaxy);
        assert!(
            !env_value(&single, "VD_PEERS")
                .expect("VD_PEERS is always emitted")
                .contains(&galaxy_book)
        );
        assert!(
            env_value(&dual, "VD_PEERS")
                .expect("VD_PEERS is always emitted")
                .contains(&galaxy_book)
        );
        assert_eq!(
            env_value(&single, "VD_SHARD"),
            Some(SHARD.0.to_string()).as_deref()
        );
        assert_eq!(
            env_value(&dual, "VD_SHARD"),
            Some(SHARD.0.to_string()).as_deref()
        );
    }

    #[test]
    fn gateway_env_chain_books_and_lists_all_three_extras_as_known_shards() {
        // The Chain gateway must BOOK every extra shard (the durable session-route swaps to each
        // successive source across the multi-hop walk) and class each as known (so each shard→gateway
        // frame is node-class dispatchable), while VD_SHARD (the login shard) stays SHARD.
        let a = dual_addrs();
        let chain = gateway_env(&a, &[], "pub", &DEV, ClusterShape::Chain);

        assert_eq!(
            env_value(&chain, "VD_KNOWN_SHARDS"),
            Some(format!("{},{},{}", GALAXY.0, PLANET_SHARD.0, SHARD_B.0)).as_deref()
        );
        let peers = env_value(&chain, "VD_PEERS").expect("VD_PEERS is always emitted");
        assert!(peers.contains(&format!("{}={}", GALAXY.0, a.galaxy)));
        assert!(peers.contains(&format!("{}={}", PLANET_SHARD.0, a.planet)));
        assert!(peers.contains(&format!("{}={}", SHARD_B.0, a.shard_b)));
        assert_eq!(
            env_value(&chain, "VD_SHARD"),
            Some(SHARD.0.to_string()).as_deref()
        );
    }

    #[test]
    fn shard_env_dual_books_the_galaxy_inert_when_single_and_injects_no_boundaries() {
        let a = dual_addrs();
        let single = shard_env(&a, &DEV, ClusterShape::Single);
        let dual = shard_env(&a, &DEV, ClusterShape::Dual);
        let galaxy_book = format!("{}={}", GALAXY.0, a.galaxy);
        // Each shard books the other ONLY in dual (the cross-shard mesh); single is byte-identical.
        assert!(
            !env_value(&single, "VD_PEERS")
                .expect("VD_PEERS is always emitted")
                .contains(&galaxy_book)
        );
        assert!(
            env_value(&dual, "VD_PEERS")
                .expect("VD_PEERS is always emitted")
                .contains(&galaxy_book)
        );
        // The home shard hosts the home realm in both modes, and NO boundary is injected in ANY shape:
        // THE world's own home shell IS the crossing boundary (the seed neighbourhood boots).
        assert_eq!(
            env_value(&dual, "VD_REALM_SEED"),
            Some(DEV.realm_seed.to_string()).as_deref()
        );
        for env in [&single, &dual, &shard_env(&a, &DEV, ClusterShape::Chain)] {
            // The SL5 guard, key-shape not key-name: NO shape's env carries ANY boundary-file key
            // (the deleted injection knobs must stay deleted, whatever a revival would call itself).
            assert_eq!(
                env.iter().find(|(k, _)| k.contains("BOUNDARIES")),
                None,
                "no boundary-file injection key in any shape's shard env (SL5)"
            );
            assert_eq!(env_value(env, "VD_HELD_REALMS"), None);
        }
    }

    #[test]
    fn single_shard_env_is_byte_identical_to_dual_false() {
        // H-1 inert-parity: the `ClusterShape::Single` arm of every builder emits EXACTLY the pre-Track-R
        // env, so a single-shard `up` / the process_parity gate stays byte-identical. (Asserted
        // field-by-field against the hand-written expected today's env — a regression flips this loud.)
        let a = dual_addrs();

        let orch = orchestrator_env(&a, &DEV, "store", ClusterShape::Single);
        assert_eq!(
            orch,
            vec![
                ("VD_NODE_ID", ORCH.0.to_string()),
                ("VD_BIND", a.orchestrator.to_string()),
                ("VD_PEERS", book(&[(GATEWAY, a.gateway), (SHARD, a.shard)])),
                ("VD_EPOCH", DEV.epoch.to_string()),
                ("VD_RESERVE_CHUNK", DEV.reserve_chunk.to_string()),
                ("VD_CLOCK_PEERS", format!("{},{}", GATEWAY.0, SHARD.0)),
                ("VD_LEASE_TTL", DEV.lease_ttl.to_string()),
                ("VD_ADMIN_ADDR", a.admin.to_string()),
                ("VD_PROBE_ADDR", a.orchestrator_probe.to_string()),
                ("VD_STORE_PATH", "store".to_owned()),
                ("VD_STORE_EPHEMERAL_OK", "1".to_owned()),
                // RLM 5f-1: the static-boot marker (every harness shape pre-spawns its shards, so an armed
                // `VD_DEMAND` reconciler must be refused). INERT while `VD_DEMAND` is unset ⇒ boot behaviour
                // stays byte-identical; only the env LIST grew by this one marker.
                ("VD_STATIC_FOREST", "1".to_owned()),
            ]
        );

        let gw = gateway_env(&a, &[], "pub", &DEV, ClusterShape::Single);
        assert_eq!(
            gw,
            vec![
                ("VD_NODE_ID", GATEWAY.0.to_string()),
                ("VD_BIND", a.gateway.to_string()),
                (
                    "VD_PEERS",
                    book(&[(ORCH, a.orchestrator), (SHARD, a.shard)])
                ),
                ("VD_ORCH", ORCH.0.to_string()),
                ("VD_SHARD", SHARD.0.to_string()),
                ("VD_AUTH_PUBKEY", "pub".to_owned()),
                ("VD_SESSION_SEED", DEV.session_seed.to_string()),
                ("VD_MAX_SESSIONS", DEV.max_sessions.to_string()),
                (
                    "VD_MAX_BUFFERED_INPUTS",
                    DEV.max_buffered_inputs.to_string()
                ),
                ("VD_PROBE_ADDR", a.gateway_probe.to_string()),
                // The gateway now BUILDS THE WORLD (it decides which realm a login lands in), so it carries
                // the same two world inputs every shard always has. It used to resolve logins against a
                // hardcoded walk-scale universe no matter which one the shards were told to run.
                ("VD_TICK_DT", DEV.tick_dt.to_string()),
                ("VD_SPEED", DEV.move_speed.to_string()),
            ]
        );

        let shard = shard_env(&a, &DEV, ClusterShape::Single);
        assert_eq!(
            shard,
            vec![
                ("VD_NODE_ID", SHARD.0.to_string()),
                ("VD_BIND", a.shard.to_string()),
                (
                    "VD_PEERS",
                    book(&[(ORCH, a.orchestrator), (GATEWAY, a.gateway)])
                ),
                ("VD_REALM_SEED", DEV.realm_seed.to_string()),
                ("VD_SPEED", DEV.move_speed.to_string()),
                ("VD_TICK_DT", DEV.tick_dt.to_string()),
                ("VD_ORCH", ORCH.0.to_string()),
                ("VD_MINT_SEED", DEV.mint_seed.to_string()),
                ("VD_INPUT_LOG_CAP", DEV.input_log_cap.to_string()),
                ("VD_REALM_RECHECK", DEV.realm_recheck.to_string()),
                ("VD_SNAPSHOT_BUDGET", DEV.snapshot_budget.to_string()),
                ("VD_PROBE_ADDR", a.shard_probe.to_string()),
            ]
        );
    }

    // ---- NODE-PER-REALM realm-shard env builders ------------------------------------------------

    #[test]
    fn realm_kind_token_round_trips_through_realm_from_kind_seed() {
        use vd_core::pose::RealmId;
        for realm in [
            RealmId::System(7),
            RealmId::Planet(7),
            RealmId::Station(7),
            RealmId::Area(7),
        ] {
            let token = realm_kind_token(realm);
            let seed = realm_seed_of(realm);
            assert_eq!(
                realm_from_kind_seed(token, seed),
                Ok(realm),
                "the VD_REALM_KIND token round-trips for {realm}"
            );
        }
        // An ABSENT/empty kind defaults to System(seed) — the byte-identical legacy shard boot.
        assert_eq!(realm_from_kind_seed("", 7), Ok(RealmId::System(7)));
        // An unknown kind fails LOUD (never a silent wrong realm).
        assert!(realm_from_kind_seed("nope", 7).is_err());
    }

    #[test]
    fn realm_shards_zip_the_roster_realms_onto_their_fixed_slots() {
        // realm_shards and roster_realms read the SAME fan-out, so the shard list is exactly the
        // roster's extras zipped onto the fixed node/port slots — and the shapes that book nothing
        // list nothing.
        let a = dual_addrs();
        let roster = world_roster(&DEV);
        for shape in [ClusterShape::Single, ClusterShape::Demand] {
            assert!(
                realm_shards(shape, &a, &DEV).is_empty(),
                "{shape:?} books no extra realm-shard"
            );
        }
        let dual = realm_shards(ClusterShape::Dual, &a, &DEV);
        assert_eq!(
            dual.iter().map(|s| (s.node, s.realm)).collect::<Vec<_>>(),
            vec![(GALAXY, roster.galaxy)],
        );
        assert_eq!((dual[0].quic, dual[0].probe), (a.galaxy, a.galaxy_probe));
        let chain = realm_shards(ClusterShape::Chain, &a, &DEV);
        assert_eq!(
            chain.iter().map(|s| (s.node, s.realm)).collect::<Vec<_>>(),
            vec![
                (GALAXY, roster.galaxy),
                (PLANET_SHARD, roster.inner),
                (SHARD_B, roster.sibling),
            ],
        );
        assert_eq!((chain[1].quic, chain[1].probe), (a.planet, a.planet_probe));
        assert_eq!(
            (chain[2].quic, chain[2].probe),
            (a.shard_b, a.shard_b_probe)
        );
    }

    #[test]
    fn every_shape_names_only_realms_of_the_world() {
        // THE GUARD (D-WORLD-8's cure): every realm a shape pre-books must be a realm THE world
        // contains. The retired Triple/Forest shapes named System(8)/Planet(7)/Station(7)/Area(7) —
        // none exist on THE world — so their extra shards died at boot (`guard_regions_nest`, 0
        // ambient roots) and the gates atop them pinned green on clusters that never stood. This
        // test alone would have caught that rot the day the world changed.
        let world = boot_world(DEV.universe_seed, DEV.move_speed, DEV.tick_dt);
        let a = dual_addrs();
        let roster = world_roster(&DEV);
        // The home shard the env builders wire (`VD_REALM_SEED = DEV.realm_seed`) must BE the
        // roster's derived home realm — or the readiness gate would wait on a realm no shard hosts.
        assert_eq!(
            roster.home,
            vd_core::pose::RealmId::System(DEV.realm_seed),
            "DEV.realm_seed drifted from THE world's home realm",
        );
        for shape in [
            ClusterShape::Single,
            ClusterShape::Dual,
            ClusterShape::Chain,
            ClusterShape::Demand,
        ] {
            let realms = roster_realms(shape, &DEV);
            for realm in &realms {
                assert!(
                    world.contains_realm(*realm),
                    "{shape:?} pre-books {realm}, which THE world does not contain — a shard for it \
                     dies at boot and any occupant steered into it strands permanently",
                );
            }
            let shards = realm_shards(shape, &a, &DEV);
            // The realm list IS home + the shard list (ONE fan-out, no drift), and Demand books none.
            let expected: Vec<vd_core::pose::RealmId> = if shape.is_demand() {
                Vec::new()
            } else {
                std::iter::once(roster.home)
                    .chain(shards.iter().map(|s| s.realm))
                    .collect()
            };
            assert_eq!(realms, expected, "{shape:?}: roster == home + realm_shards");
            // Roster NODES pairwise distinct (including the base trio — an aliased node id would
            // silently merge two shards' peer-book entries).
            let mut nodes = vec![ORCH.0, GATEWAY.0, SHARD.0];
            nodes.extend(shards.iter().map(|s| s.node.0));
            let node_set: std::collections::BTreeSet<u64> = nodes.iter().copied().collect();
            assert_eq!(node_set.len(), nodes.len(), "{shape:?}: node ids alias");
            // MINTS pairwise distinct across the home shard + every realm-shard (an aliased mint
            // lets two shards mint the same entity id).
            let mut mints = vec![
                env_value(&shard_env(&a, &DEV, shape), "VD_MINT_SEED")
                    .expect("home mint emitted")
                    .to_owned(),
            ];
            for shard in &shards {
                mints.push(
                    env_value(&realm_shard_env(&a, &DEV, shape, *shard), "VD_MINT_SEED")
                        .expect("realm-shard mint emitted")
                        .to_owned(),
                );
            }
            let mint_set: std::collections::BTreeSet<&String> = mints.iter().collect();
            assert_eq!(mint_set.len(), mints.len(), "{shape:?}: mint seeds alias");
        }
    }

    #[test]
    fn chain_home_shard_env_books_all_three_extras_and_never_cohosts() {
        // The Chain home shard books EVERY other realm-shard (the cross-shard mesh) and, crucially,
        // sets NO `VD_HELD_REALMS` — each realm is its own node, so the source==dest co-hosting is gone.
        let a = dual_addrs();
        let env = shard_env(&a, &DEV, ClusterShape::Chain);
        let peers = env_value(&env, "VD_PEERS").expect("VD_PEERS is always emitted");
        for shard in realm_shards(ClusterShape::Chain, &a, &DEV) {
            assert!(
                peers.contains(&format!("{}={}", shard.node.0, shard.quic)),
                "the home shard books realm-shard node {}",
                shard.node.0,
            );
        }
        assert_eq!(
            env_value(&env, "VD_HELD_REALMS"),
            None,
            "no shape co-hosts — VD_HELD_REALMS must be absent (each realm on its own node)"
        );
        // The home shard is still a System realm via the absent-default (no VD_REALM_KIND emitted).
        assert_eq!(env_value(&env, "VD_REALM_KIND"), None);
        assert_eq!(
            env_value(&env, "VD_REALM_SEED"),
            Some(DEV.realm_seed.to_string()).as_deref()
        );
    }

    #[test]
    fn chain_realm_shard_env_hosts_one_realm_with_its_kind_and_books_the_others() {
        // The inner-planet realm-shard: hosts EXACTLY the roster's inner mover (VD_REALM_KIND=planet,
        // VD_REALM_SEED=its generated seed), no co-hosting, a distinct mint, and books ORCH + GATEWAY +
        // the home shard + every OTHER realm-shard (never itself).
        use vd_core::pose::RealmId;
        let a = dual_addrs();
        let roster = world_roster(&DEV);
        let planet = realm_shards(ClusterShape::Chain, &a, &DEV)
            .into_iter()
            .find(|s| s.realm == roster.inner)
            .expect("the inner-planet realm-shard is in the Chain set");
        let env = realm_shard_env(&a, &DEV, ClusterShape::Chain, planet);
        assert_eq!(
            env_value(&env, "VD_NODE_ID"),
            Some(planet.node.0.to_string()).as_deref()
        );
        assert_eq!(env_value(&env, "VD_REALM_KIND"), Some("planet"));
        let RealmId::Planet(inner_seed) = roster.inner else {
            panic!("the roster's inner mover is a planet, got {}", roster.inner);
        };
        assert_eq!(
            env_value(&env, "VD_REALM_SEED"),
            Some(inner_seed.to_string()).as_deref()
        );
        // NO co-hosting, NO boundary-file key of any name — the seed neighbourhood boots (SL5).
        assert_eq!(env_value(&env, "VD_HELD_REALMS"), None);
        assert_eq!(
            env.iter().find(|(k, _)| k.contains("BOUNDARIES")),
            None,
            "no boundary-file injection key in a realm-shard env (SL5)"
        );
        // Books ORCH + GATEWAY + the home shard + every OTHER realm-shard, but NOT itself.
        let peers = env_value(&env, "VD_PEERS").expect("VD_PEERS is always emitted");
        assert!(peers.contains(&format!("{}={}", ORCH.0, a.orchestrator)));
        assert!(peers.contains(&format!("{}={}", GATEWAY.0, a.gateway)));
        assert!(peers.contains(&format!("{}={}", SHARD.0, a.shard)));
        assert!(
            !peers.contains(&format!("{}={}", planet.node.0, planet.quic)),
            "a realm-shard does not book ITSELF"
        );
        for other in realm_shards(ClusterShape::Chain, &a, &DEV)
            .into_iter()
            .filter(|s| s.node != planet.node)
        {
            assert!(
                peers.contains(&format!("{}={}", other.node.0, other.quic)),
                "the inner-planet shard books the OTHER realm-shard node {}",
                other.node.0,
            );
        }
        // The galaxy/sibling realm-shards keep their HISTORIC distinct mints (harness galaxy=23 /
        // dest=17 vs home=11); the planet derives from its node id.
        let galaxy = realm_shards(ClusterShape::Chain, &a, &DEV)
            .into_iter()
            .find(|s| s.node == GALAXY)
            .expect("the galaxy realm-shard is in the Chain set");
        assert_eq!(
            env_value(
                &realm_shard_env(&a, &DEV, ClusterShape::Chain, galaxy),
                "VD_MINT_SEED"
            ),
            Some(DEV.mint_seed.wrapping_add(12).to_string()).as_deref()
        );
    }
}
