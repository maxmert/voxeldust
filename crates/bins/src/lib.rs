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

use std::net::SocketAddr;
use std::path::PathBuf;
use std::process::{Child, Command, ExitStatus};
use std::time::Duration;

use ed25519_dalek::SigningKey;
use vd_core::NodeId;
use vd_devproto::{DevRequest, DevResponse, WORKTREE_SLOT_CEILING};
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
/// Track R / 1d.2 — the DEST shard (realm B) a dot re-homes INTO. Matches the harness `DEST`
/// (`tests/src/lib.rs`). Absent from a single-shard `up` (only a dual `up` / `VD_KNOWN_SHARDS` /
/// `VD_ROSTER` brings it into any roster). HR3: this is a `NodeId` in a SET, never a shard KIND —
/// adding it is a roster extension, never a code branch on a shard kind.
///
/// SCOPE (M-2): this is a single extra shard for the LOCAL 2-process crossing playground. The
/// N-shard k3d roster generalization (a `Vec` of shard ids/addrs, N-entry rosters) is ledgered as a
/// separate cloud (#123) slice in `docs/design/DEFERRED.md`.
pub const SHARD_B: NodeId = NodeId(4);
/// S5b — the GALAXY between-space shard (`RealmId::System(GALAXY_SEED)`, the seed forest's
/// between-systems space). It OWNS System 7 + System 8 as its CHILDREN, so a SIBLING crossing routes
/// THROUGH it (leave System 7 → land in the Galaxy → the Galaxy shard detects the entry into System 8),
/// and authority can REST in the between-space so the multi-hop dot is NEVER orphaned. Matches the
/// harness `GALAXY` (`tests/src/lib.rs`) + `vd_core::worldgen`'s `GALAXY = System(1)`. Absent from a
/// single-shard / dual `up`; only a [`ClusterShape::Triple`] `up` brings it into any roster. HR3: one
/// more `NodeId` in a SET, never a shard KIND — adding it is a roster extension, never a code branch.
pub const GALAXY: NodeId = NodeId(5);

/// NODE-PER-REALM (task #149): the Planet 7 realm-shard (`RealmId::Planet(7)`) — its OWN node, so a
/// walk into Planet 7's SOI is a CROSS-NODE saga, not a co-hosted local relabel. Present only in a
/// [`ClusterShape::Forest`] `up`. HR3: one more `NodeId` in the roster, never a shard KIND.
pub const PLANET_A_SHARD: NodeId = NodeId(6);
/// NODE-PER-REALM: the Station 7 realm-shard (`RealmId::Station(7)`) — its OWN node. Forest-only.
pub const STATION_A_SHARD: NodeId = NodeId(7);
/// NODE-PER-REALM: the Area 7 realm-shard (`RealmId::Area(7)`) — its OWN node (the DEEPEST realm). Forest-only.
pub const AREA_A_SHARD: NodeId = NodeId(8);

/// The GALAXY realm seed — must match `vd_core::worldgen`'s Galaxy (`System(1)`, the between-systems
/// space) and the harness `GALAXY_SEED`. The galaxy shard hosts `RealmId::System(GALAXY_SEED)`.
pub const GALAXY_SEED: u64 = 1;

/// The seed of the System-7 sub-forest (Planet 7 / Station 7 / Area 7 all key on 7 in `vd_core::worldgen`).
/// The [`ClusterShape::Forest`] realm-shards host `Planet(FOREST_CHILD_SEED)` etc.
pub const FOREST_CHILD_SEED: u64 = 7;

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
///
/// - [`Single`](ClusterShape::Single): orchestrator + gateway + System 7 (the base `up`).
/// - [`Dual`](ClusterShape::Dual): + the DEST shard [`SHARD_B`] (System 8) — the `--dual` DIRECT-re-home
///   playground (a born-inside or injected walk-into trigger re-homes 7→8 in one hop, no Galaxy).
/// - [`Triple`](ClusterShape::Triple): + the DEST [`SHARD_B`] AND the [`GALAXY`] between-space shard —
///   the `--triple` SEED-FOREST cluster where a durable dot walks System 7 → Galaxy → System 8 and back,
///   coordinate-driven, with the Galaxy rendered as the containing box (never orphaned). S5b.
/// - [`Forest`](ClusterShape::Forest): NODE-PER-REALM (task #149) — SIX single-realm shards, one per realm
///   of the whole seed sub-forest: System 7, Planet 7, Station 7, Area 7, Galaxy (System 1), System 8. NO
///   `VD_HELD_REALMS` co-hosting — EVERY re-home (including into a System-7 child) is a uniform CROSS-NODE
///   saga, so the source==dest degenerate case never arises. This is the shape the node-per-realm walk gate
///   (`crates/bins/tests/node_per_realm_walk.rs`) + the `crossing-playground.sh` launcher stand up.
/// - [`Demand`](ClusterShape::Demand): RLM RG-4 — orchestrator + gateway ONLY, with NO shard pre-booked. The
///   ONLY way a player reaches a world is the armed demand reconciler spinning one up on the fly at login
///   (`VD_DEMAND=1`, mutually exclusive with the static forest). The gateway reaches that just-spawned shard
///   with NO pre-booked address via the reactive greeting (RG-0..3). This is the shape the demand-login e2e
///   (`crates/bins/tests/rlm_demand_login.rs`) stands up; the launcher's `up --demand` is deferred.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ClusterShape {
    Single,
    Dual,
    Triple,
    Forest,
    Demand,
}

/// ONE extra realm-shard beyond the base orchestrator+gateway+System-7 trio: its roster [`NodeId`], the
/// [`RealmId`](vd_core::pose::RealmId) it SINGLY hosts, and its QUIC + probe bind. The shape's fan-out
/// ([`ClusterShape::extra_realm_shards`]) returns these as DATA the env builders + the launcher iterate —
/// never a shard-KIND branch (HR3). Each carries its realm so the shard bin can boot the right realm KIND
/// (`VD_REALM_KIND`), not just a `System(seed)`.
#[derive(Clone, Copy, Debug)]
pub struct RealmShard {
    pub node: NodeId,
    pub realm: vd_core::pose::RealmId,
    pub quic: SocketAddr,
    pub probe: SocketAddr,
}

impl ClusterShape {
    /// Does this shape spawn the DEST shard [`SHARD_B`] (System 8)? Dual + Triple do; Single does not.
    /// (Forest hosts System 8 too, but as one of its uniform realm-shards — see [`extra_realm_shards`].)
    ///
    /// [`extra_realm_shards`]: ClusterShape::extra_realm_shards
    #[must_use]
    pub fn has_dest(self) -> bool {
        matches!(self, ClusterShape::Dual | ClusterShape::Triple)
    }

    /// RLM RG-4: is this the DEMAND shape (orchestrator + gateway only, the reconciler armed, NO static
    /// shard)? The `*_env` builders fan out on this to emit `VD_DEMAND` + drop `VD_STATIC_FOREST`/the booked
    /// shard — data, not a shard-KIND branch. `false` for every static shape (byte-identical).
    #[must_use]
    pub fn is_demand(self) -> bool {
        matches!(self, ClusterShape::Demand)
    }

    /// Does this shape spawn the [`GALAXY`] between-space shard (System 1) via the Triple wiring? Only
    /// Triple does (Forest also hosts the Galaxy, but through [`extra_realm_shards`], not this flag).
    #[must_use]
    pub fn has_galaxy(self) -> bool {
        matches!(self, ClusterShape::Triple)
    }

    /// The single-realm shards this shape stands up BEYOND the base orchestrator + gateway + System-7 shard
    /// (which always hosts `System(realm_seed)`). Non-empty ONLY for [`Forest`](ClusterShape::Forest): the
    /// Planet 7 / Station 7 / Area 7 children, the Galaxy between-space, and System 8 — each its OWN node so
    /// every re-home is a uniform CROSS-NODE saga. Single/Dual/Triple keep the legacy explicit wiring
    /// (their System 8 / Galaxy ride the `has_dest`/`has_galaxy` flags in the `*_env` builders), so this is
    /// EMPTY for them — byte-identical. The System-7 shard is NOT listed (it is the always-present base).
    #[must_use]
    pub fn extra_realm_shards(self, a: &ClusterAddrs, p: &DevClusterParams) -> Vec<RealmShard> {
        use vd_core::pose::RealmId;
        if self != ClusterShape::Forest {
            return Vec::new();
        }
        vec![
            RealmShard {
                node: PLANET_A_SHARD,
                realm: RealmId::Planet(FOREST_CHILD_SEED),
                quic: a.planet,
                probe: a.planet_probe,
            },
            RealmShard {
                node: STATION_A_SHARD,
                realm: RealmId::Station(FOREST_CHILD_SEED),
                quic: a.station,
                probe: a.station_probe,
            },
            RealmShard {
                node: AREA_A_SHARD,
                realm: RealmId::Area(FOREST_CHILD_SEED),
                quic: a.area,
                probe: a.area_probe,
            },
            RealmShard {
                node: GALAXY,
                realm: RealmId::System(GALAXY_SEED),
                quic: a.galaxy,
                probe: a.galaxy_probe,
            },
            RealmShard {
                node: SHARD_B,
                realm: RealmId::System(p.realm_seed_b),
                quic: a.shard_b,
                probe: a.shard_b_probe,
            },
        ]
    }
}

// ---- dev auth identity -------------------------------------------------------

/// DEV-ONLY Ed25519 login seed for the localhost cluster. The gateway boots with
/// its verifying key; clients mint logins with the seed. NEVER a production
/// secret — a fixed dev constant so the identity is reproducible. (Structurally
/// fencing it behind a dev/test compile guard is tracked for the prod-build phase.)
pub const DEV_AUTH_SEED: [u8; 32] = [0x42; 32];

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
    /// Track R / 1d.2 — the DEST realm seed: the shard at [`SHARD_B`] hosts `RealmId::System(realm_seed_b)`
    /// and a dot re-homes INTO it. MUST differ from [`realm_seed`](Self::realm_seed) (a same-realm
    /// "crossing" is a no-op) — enforced by a compile-time assert on [`DEV`]. Matches the harness
    /// `dest_stub_config` (`System(8)`). Inert in a single-shard `up`: no DEST is spawned, so no realm
    /// B is ever granted.
    pub realm_seed_b: u64,
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
    realm_seed_b: 8, // Track R / 1d.2 DEST realm (matches the harness `dest_stub_config` System(8)).
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

/// DEST-REALM-DISTINCT (compile-time, Track R / 1d.2): the DEST realm MUST differ from the source
/// realm — a crossing INTO the shard's own realm is a no-op (the geometric trigger's `to_realm` would
/// equal the exterior `realm`). A regression here fails the BUILD, not a flaky dual-cluster test.
const _: () = assert!(
    DEV.realm_seed != DEV.realm_seed_b,
    "DEV.realm_seed_b (the DEST realm) must differ from DEV.realm_seed (the source realm)",
);

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
    /// Single/Dual/Triple/Forest clusters are byte-identical (no extra port bound).
    pub gateway_admin: Option<SocketAddr>,
    /// The k8s /healthz+/readyz probe listeners (S3) — one per node (each needs its own kubelet-reachable
    /// port). Named fields (no offset math), matching the existing style.
    pub orchestrator_probe: SocketAddr,
    pub gateway_probe: SocketAddr,
    pub shard_probe: SocketAddr,
    /// Track R / 1d.2 (M-2 scope: the LOCAL 2-process crossing playground): the DEST shard's QUIC bind +
    /// probe. Populated for every cluster (data, no branch); dialed/spawned ONLY in dual mode (the `dual`
    /// arm of the env builders books it; a single-shard `up` never spawns the DEST). N-shard generalization
    /// (a `Vec` of shard addrs) is ledgered to cloud #123 in `docs/design/DEFERRED.md`.
    pub shard_b: SocketAddr,
    pub shard_b_probe: SocketAddr,
    /// S5b — the GALAXY between-space shard's QUIC bind + probe. Populated for every cluster (data, no
    /// branch); dialed/spawned ONLY in [`ClusterShape::Triple`]/[`Forest`](ClusterShape::Forest) mode. Twin
    /// of the [`shard_b`](Self::shard_b) pair — one more shard in the roster, never a shard KIND.
    pub galaxy: SocketAddr,
    pub galaxy_probe: SocketAddr,
    /// NODE-PER-REALM (task #149) — the Planet 7 / Station 7 / Area 7 realm-shards' QUIC binds + probes.
    /// Populated for every cluster (data, no branch); dialed/spawned ONLY in [`Forest`](ClusterShape::Forest)
    /// mode, where each hosts exactly ONE realm so every re-home is a uniform CROSS-NODE saga. Twins of the
    /// [`galaxy`](Self::galaxy) pair — three more shards in the roster, never a shard KIND.
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
    vec![
        ("VD_TRUST_DIR", trust_dir.to_owned()),
        str_pair("VD_OUTBOUND_CAP", p.outbound_cap),
        str_pair("VD_TICK_HZ", p.tick_hz),
        str_pair("VD_PROCESS_INCARNATION", launch_incarnation()),
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
        ("VD_UNIVERSE_SCALE", "visual-demand".to_owned()),
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
    // Track R / 1d.2 + S5b: in a MULTI-shard cluster the orchestrator INITIATES realm-grants / re-home to
    // every extra shard, so it books them AND drives the universe clock to each (a follower whose clock
    // never advances can never win its realm lease — the harness `clock_peers = {GW, SHARD, [DEST,
    // GALAXY]}` rule). Each extra shard is ALSO added to `VD_ROSTER` (the D-37 re-home candidate set; the
    // crossing itself resolves via the directory head, NOT the roster — see the bin note). In Single mode
    // these keys are byte-identical to the pre-Track-R env (the crossing INERT). The peer/clock/roster
    // lists GROW with the shape (data, no per-kind branch): Dual adds the DEST; Triple adds DEST + GALAXY.
    let mut extra_shards: Vec<(NodeId, SocketAddr)> = Vec::new();
    if shape.has_dest() {
        extra_shards.push((SHARD_B, a.shard_b));
    }
    if shape.has_galaxy() {
        extra_shards.push((GALAXY, a.galaxy));
    }
    // NODE-PER-REALM (Forest): fold the Planet/Station/Area/Galaxy/System-8 realm-shards in as extra
    // roster entries (empty for Single/Dual/Triple — byte-identical). The orchestrator books + drives the
    // clock + rosters each, exactly like the Dual/Triple extras.
    extra_shards.extend(
        shape
            .extra_realm_shards(a, p)
            .iter()
            .map(|s| (s.node, s.quic)),
    );
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
    // Track R / 1d.2 + S5b: in a MULTI-shard cluster the gateway must BOOK every extra shard (to route a
    // transferred client's inputs / cut-drains onto it — the durable session-route swap at
    // `CommitAuthority` migrates the route to each successive source shard, so ALL of them must be
    // dialable) AND class each as a KNOWN shard (`VD_KNOWN_SHARDS`, consumed by the gateway bin's
    // `known_shards` set) so a shard→gateway frame reaches `on_shard_frame` instead of dropping as an
    // unknown peer. `VD_SHARD` (the LOGIN shard) stays SHARD in every mode. In Single mode neither an
    // extra peer nor `VD_KNOWN_SHARDS` is emitted — byte-identical to the pre-Track-R env. The lists GROW
    // with the shape (data, no per-kind branch): Dual adds the DEST; Triple adds DEST + GALAXY.
    let mut extra_shards: Vec<(NodeId, SocketAddr)> = Vec::new();
    if shape.has_dest() {
        extra_shards.push((SHARD_B, a.shard_b));
    }
    if shape.has_galaxy() {
        extra_shards.push((GALAXY, a.galaxy));
    }
    // NODE-PER-REALM (Forest): the gateway must BOOK + CLASS-as-known every realm-shard (the durable
    // session-route swap at CommitAuthority migrates the client's input route onto each successive source
    // shard as the dot walks System 7 → Planet 7 → Area 7 → … so ALL must be dialable AND recognized). Empty
    // for Single/Dual/Triple — byte-identical.
    extra_shards.extend(
        shape
            .extra_realm_shards(a, p)
            .iter()
            .map(|s| (s.node, s.quic)),
    );
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

/// The SOURCE stub-shard's node-specific env (realm + movement + snapshot budget). In DUAL mode it also
/// BOOKS the DEST shard so the two shards can mesh cross-shard transfer traffic (each shard books the
/// other). Absent `dual` the peer book is byte-identical to a single-shard `up`. The SOURCE hosts the
/// crossing trigger into realm B (planted separately via `VD_REALM_BOUNDARIES`, set by the launcher).
#[must_use]
pub fn shard_env(
    a: &ClusterAddrs,
    p: &DevClusterParams,
    shape: ClusterShape,
) -> Vec<(&'static str, String)> {
    // The SOURCE (System 7) books every OTHER shard so the cross-shard mesh can carry transfer traffic
    // (each shard books the others). The list GROWS with the shape (data, no per-kind branch): Dual books
    // the DEST; Triple books DEST + GALAXY. In Single mode the book is byte-identical to the pre-Track-R env.
    let mut peers = vec![(ORCH, a.orchestrator), (GATEWAY, a.gateway)];
    if shape.has_dest() {
        peers.push((SHARD_B, a.shard_b));
    }
    if shape.has_galaxy() {
        peers.push((GALAXY, a.galaxy));
    }
    // NODE-PER-REALM (Forest): the System-7 shard books every OTHER realm-shard (Planet/Station/Area/
    // Galaxy/System 8) so the cross-shard mesh carries a re-home INTO any of them. Empty for the legacy
    // shapes — byte-identical.
    peers.extend(
        shape
            .extra_realm_shards(a, p)
            .iter()
            .map(|s| (s.node, s.quic)),
    );
    let mut env = vec![
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
    // CO-HOSTING (the un-hosted-child cure) — the --triple LEGACY shape only: the seed forest nests Planet 7
    // / Station 7 / Area 7 under System 7, but the --triple cluster hosts no shard for those children, so the
    // System-7 shard CO-HOSTS them (`VD_HELD_REALMS`) and a walk-into-child re-home resolves `head(Realm(
    // child))` to THIS node (source==dest). NODE-PER-REALM (Forest) is the SUPERSEDING shape: it gives each
    // child its OWN shard, so NO co-hosting — every re-home is a uniform CROSS-NODE saga (the source==dest
    // degenerate case never arises). Dual/Single don't co-host either — byte-identical there.
    if shape.has_galaxy() {
        env.push(str_pair("VD_HELD_REALMS", triple_source_held_realms_env()));
    }
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

/// NODE-PER-REALM (Forest) — the node-specific env for ONE extra realm-shard (Planet/Station/Area/Galaxy/
/// System 8), each hosting EXACTLY its own realm (NO `VD_HELD_REALMS`). Twin of [`shard_env`]/[`galaxy_env`]:
/// it books ORCH + GATEWAY + the System-7 source shard + every OTHER Forest realm-shard (each shard books the
/// others so the cross-shard mesh carries a re-home between any pair), and emits `VD_REALM_KIND` + `VD_REALM_SEED`
/// so the shard bin boots the right realm KIND. A DISTINCT mint per node (derived from the source mint by the
/// node id) so no two shards alias entity ids. The Galaxy/System-8 realm-shards get their historic distinct
/// mints (`+12`/`+6`) so their entity ids match the legacy Triple/Dual rigs; the new children derive from the
/// node id. NO `VD_REALM_BOUNDARIES` — the SEED neighbourhood boots (`realm_neighbourhood_for(own)` = own +
/// ancestors + DIRECT children), so a Planet-7 shard sees the Area-7 child boundary, System 7 sees the
/// Planet/Station boundaries, etc. — this is how the containment detector fires each cross-node re-home.
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
        shape
            .extra_realm_shards(a, p)
            .iter()
            .filter(|s| s.node != shard.node)
            .map(|s| (s.node, s.quic)),
    );
    // A distinct mint per node so no two shards ever alias an entity id. The Galaxy/System-8 shards keep
    // their HISTORIC offsets (matching the Triple/Dual rigs: `+12`/`+6`); the new Planet/Station/Area children
    // derive from the node id above a BASE (`+100`) that clears the small historic offsets — so a node-id of
    // 6 (Planet) never collides with System-8's `+6`. All five mints are pairwise distinct (asserted in
    // `forest_realm_shard_env_hosts_one_realm_with_its_kind_and_books_the_others`).
    const FOREST_CHILD_MINT_BASE: u64 = 100;
    let mint = match shard.node {
        GALAXY => p.mint_seed.wrapping_add(12),
        SHARD_B => p.mint_seed.wrapping_add(6),
        other => p
            .mint_seed
            .wrapping_add(FOREST_CHILD_MINT_BASE)
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

/// The `VD_HELD_REALMS` value the --triple System-7 shard co-hosts: its own realm PLUS its seed-forest
/// children Planet 7 / Station 7 / Area 7 (the un-hosted-child cure). SINGLE-SOURCED so the launcher, the
/// shard boot, and the smoke test agree on the exact co-hosted set. `realm_seed` is the SOURCE realm seed
/// (7 in every current rig), so the children key on the same seed (Planet(7)/Station(7)/Area(7)) — the
/// forest's `PLANET_A`/`STATION_A`/`AREA_A` all carry seed 7.
#[must_use]
pub fn triple_source_held_realms() -> std::collections::BTreeSet<vd_core::pose::RealmId> {
    use vd_core::pose::RealmId;
    std::collections::BTreeSet::from([
        RealmId::System(7),
        RealmId::Planet(7),
        RealmId::Station(7),
        RealmId::Area(7),
    ])
}

/// The env-string form of [`triple_source_held_realms`] (`VD_HELD_REALMS` wire).
#[must_use]
pub fn triple_source_held_realms_env() -> String {
    held_realms_env(&triple_source_held_realms())
}

/// Track R / 1d.2 — the DEST stub-shard's node-specific env (twin of [`shard_env`]). Hosts realm B
/// (`System(realm_seed_b)`) at [`SHARD_B`], with a DISTINCT mint seed (so its dots are genuinely separate,
/// mirroring the harness `dest_stub_config` mint 17 vs the source's 11) and its OWN QUIC bind + probe. Its
/// `VD_PEERS` books ORCH, GATEWAY, the SOURCE shard, AND (in [`ClusterShape::Triple`]) the GALAXY between-
/// space shard — each shard books the others (the cross-shard mesh), so the list GROWS with the shape.
/// NO `VD_REALM_BOUNDARIES`: in Triple the seed forest routes System 8 → Galaxy → System 7 through the
/// Galaxy parent; in Dual the SOURCE hosts the injected/born-inside trigger and the DEST just receives.
/// Spawned by a Dual or Triple `up`.
#[must_use]
pub fn shard_b_env(
    a: &ClusterAddrs,
    p: &DevClusterParams,
    shape: ClusterShape,
) -> Vec<(&'static str, String)> {
    let mut peers = vec![
        (ORCH, a.orchestrator),
        (GATEWAY, a.gateway),
        (SHARD, a.shard),
    ];
    if shape.has_galaxy() {
        peers.push((GALAXY, a.galaxy));
    }
    vec![
        str_pair("VD_NODE_ID", SHARD_B.0),
        str_pair("VD_BIND", a.shard_b),
        ("VD_PEERS", book(&peers)),
        str_pair("VD_REALM_SEED", p.realm_seed_b), // System(realm_seed_b) — DEST realm identity
        str_pair("VD_SPEED", p.move_speed),
        str_pair("VD_TICK_DT", p.tick_dt),
        str_pair("VD_ORCH", ORCH.0),
        // A distinct mint so DEST-minted entities never alias the source's (harness DEST=17 vs source=11).
        str_pair("VD_MINT_SEED", p.mint_seed.wrapping_add(6)),
        str_pair("VD_INPUT_LOG_CAP", p.input_log_cap),
        str_pair("VD_REALM_RECHECK", p.realm_recheck),
        str_pair("VD_SNAPSHOT_BUDGET", p.snapshot_budget),
        str_pair("VD_PROBE_ADDR", a.shard_b_probe),
    ]
}

/// S5b — the GALAXY between-space stub-shard's node-specific env (twin of [`shard_env`] / [`shard_b_env`]).
/// Hosts the Galaxy realm (`System(GALAXY_SEED)`, the seed forest's between-systems space) at [`GALAXY`],
/// with a DISTINCT mint seed (mirroring the harness `galaxy_stub_config` mint 23 vs source 11 / dest 17)
/// and its OWN QUIC bind + probe. Its `VD_PEERS` books ORCH, GATEWAY, AND BOTH the SOURCE (System 7) and
/// DEST (System 8) shards — the Galaxy is the SIBLING-ROUTING shard (it OWNS the two systems as children,
/// so a crossing routes System 7 → Galaxy → System 8 THROUGH it, and back). NO `VD_REALM_BOUNDARIES`: the
/// Galaxy boots the SEED-DERIVED neighbourhood (`realm_neighbourhood_for(System(1))` = {Universe, Galaxy,
/// System 7, System 8}), so its detector re-homes Galaxy→7 and Galaxy→8 by construction. Only ever spawned
/// by a [`ClusterShape::Triple`] `up`.
#[must_use]
pub fn galaxy_env(a: &ClusterAddrs, p: &DevClusterParams) -> Vec<(&'static str, String)> {
    vec![
        str_pair("VD_NODE_ID", GALAXY.0),
        str_pair("VD_BIND", a.galaxy),
        (
            "VD_PEERS",
            book(&[
                (ORCH, a.orchestrator),
                (GATEWAY, a.gateway),
                (SHARD, a.shard),
                (SHARD_B, a.shard_b),
            ]),
        ),
        str_pair("VD_REALM_SEED", GALAXY_SEED), // System(GALAXY_SEED) — the between-space realm identity
        str_pair("VD_SPEED", p.move_speed),
        str_pair("VD_TICK_DT", p.tick_dt),
        str_pair("VD_ORCH", ORCH.0),
        // A distinct mint so Galaxy-minted entities never alias the systems' (harness GALAXY=23 vs
        // source=11 / dest=17); derived from the source mint so it stays magic-number-free.
        str_pair("VD_MINT_SEED", p.mint_seed.wrapping_add(12)),
        str_pair("VD_INPUT_LOG_CAP", p.input_log_cap),
        str_pair("VD_REALM_RECHECK", p.realm_recheck),
        str_pair("VD_SNAPSHOT_BUDGET", p.snapshot_budget),
        str_pair("VD_PROBE_ADDR", a.galaxy_probe),
    ]
}

/// Track R / 1d.2 — the SINGLE-SOURCED source-realm crossing boundary set the SOURCE shard boots with
/// (`VD_REALM_BOUNDARIES`) so its geometric dwell detector fires a `CrossingRequest` into realm B. Returns
/// the `Vec<RealmBoundary>` (also serialize-roundtrippable to the `boxes.json` the shard's boot loader
/// reads — the same format the client's `--realm-boxes` loads). ONE born-inside authority shell centered at
/// the dot's login spawn offset (`LatticePos::local(ZERO)`), whose exterior `realm` is the SOURCE realm
/// (`System(realm_seed)`, so `guard_boundaries_in_realm` accepts it) and whose `to_realm` is DERIVED from
/// `realm_seed_b` (NEVER an inline `System(8)` — HR3 / M-1). Geometry mirrors the harness
/// `plant_one_crossing_shell` (born-inside ⇒ commits after the dwell without any walk).
#[must_use]
pub fn source_crossing_boundaries(p: &DevClusterParams) -> Vec<vd_core::geometry::RealmBoundary> {
    use vd_core::geometry::{CrossEffect, RealmBoundary};
    use vd_core::glam::DVec3;
    use vd_core::pose::{LatticePos, RealmId};
    vec![RealmBoundary::shell(
        RealmId::System(p.realm_seed), // exterior side = the SOURCE realm (guard-passing)
        LatticePos::local(DVec3::ZERO), // centered at the dot's login spawn offset (born-inside)
        1000.0,                        // r_soi
        1.15,                          // create_factor → create edge 1150 m
        1.30,                          // destroy_factor → destroy edge 1300 m
        p.move_speed,                  // v_rel (the dot's own walk speed)
        p.tick_dt,                     // dt
        0.5,                           // pad_floor
        1.0,                           // k_safety_extra
        None,                          // top-level (depth 0)
        RealmId::System(p.realm_seed_b), // to_realm DERIVED from realm_seed_b (M-1: never inline)
        CrossEffect::Authority,        // a TRANSFER crossing (hands authority to the dest realm)
    )]
}

/// Track R / 1d.2 — write [`source_crossing_boundaries`] as the client-identical `boundaries.json` into
/// `dir`, returning the path (as a `String` for the env value). The SOURCE shard reads it via
/// `VD_REALM_BOUNDARIES`; the file is SINGLE-SOURCED with [`source_crossing_boundaries`] so the launcher
/// and any client `--realm-boxes` project the SAME geometry.
///
/// # Errors
/// A directory-create, serialize, or write failure.
pub fn write_source_boundaries(
    dir: &std::path::Path,
    p: &DevClusterParams,
) -> Result<String, String> {
    std::fs::create_dir_all(dir).map_err(|e| format!("create boundaries dir: {e}"))?;
    let path = dir.join("source_boundaries.json");
    let json = serde_json::to_string(&source_crossing_boundaries(p))
        .map_err(|e| format!("serialize boundaries: {e}"))?;
    std::fs::write(&path, json).map_err(|e| format!("write boundaries: {e}"))?;
    Ok(path.display().to_string())
}

/// The path to the SOURCE shard's crossing boundaries for a `--dual` cluster: an operator/test
/// OVERRIDE via `VD_DEVCLUSTER_BOUNDARIES` (a dev-config hook so a crossing test — e.g. the visual
/// `render_crossing_smoke` — can inject its OWN *walk-into* geometry in place of the default
/// *born-inside* shell), else the born-inside shell [`write_source_boundaries`] writes into `dir`.
/// INERT by default: `env_override == None`/empty ⇒ the born-inside default, so every committed
/// dual-cluster path (`dual_cluster_crossing_smoke`, the launcher's own `up --dual`) is byte-identical.
/// The override file is validated as a readable file so a typo'd path fails LOUD rather than silently
/// planting the born-inside default (which would make a crossing test's geometry a mystery no-op).
///
/// # Errors
/// An override path that is not a readable file, or a [`write_source_boundaries`] failure when no
/// override is set.
pub fn resolve_source_boundaries(
    env_override: Option<String>,
    dir: &std::path::Path,
    p: &DevClusterParams,
) -> Result<String, String> {
    match env_override {
        Some(path) if !path.is_empty() => {
            if std::path::Path::new(&path).is_file() {
                Ok(path)
            } else {
                Err(format!(
                    "VD_DEVCLUSTER_BOUNDARIES is set but not a readable file: {path}"
                ))
            }
        }
        _ => write_source_boundaries(dir, p),
    }
}

/// The Visual Crossing Playground geometry (V4) — the SINGLE SOURCE for the walk-into crossing shared
/// by the `render_crossing_smoke` GPU gate AND the human `crossing-playground` launcher. Box A is the
/// SOURCE realm at the origin (the dot's login spawn); box B is the DEST realm offset on +X, far enough
/// that the two projected boxes are disjoint on screen. The dot walks +X out of box A across a shell
/// trigger (centered at box B, exterior `realm`=SOURCE so the shard's in-realm guard accepts it,
/// `to_realm`=DEST) into box B, where the transfer re-homes its authority.
pub mod crossing_playground {
    use super::DevClusterParams;
    use vd_core::geometry::{CrossEffect, RealmBoundary};
    use vd_core::glam::DVec3;
    use vd_core::pose::{LatticePos, RealmId};

    /// Box A (SOURCE realm) center — the dot's login-spawn origin.
    pub const BOX_A_CENTER: DVec3 = DVec3::new(0.0, 0.0, 0.0);
    /// Box B (DEST realm) center — offset on +X so the two boxes are disjoint on screen.
    pub const BOX_B_CENTER: DVec3 = DVec3::new(50.0, 0.0, 0.0);
    /// The (Chebyshev) half-extent of each render box.
    pub const BOX_HALF: DVec3 = DVec3::new(12.0, 12.0, 12.0);
    /// The trigger shell's SOI radius (create edge = `r_soi * 1.15` ≈ 11.5 → the dot enters at box B's
    /// near face ≈ +38.5, dwells `n_entry` ticks, and commits).
    pub const TRIGGER_R_SOI: f64 = 10.0;

    /// The SHARD's walk-into crossing trigger (planted on the SOURCE via `VD_REALM_BOUNDARIES`): a shell
    /// centered at box B, exterior `realm`=SOURCE (guard-passing), `to_realm`=DEST. Realms DERIVED from
    /// `DevClusterParams` (never inline — HR3).
    #[must_use]
    pub fn trigger(p: &DevClusterParams) -> Vec<RealmBoundary> {
        vec![RealmBoundary::shell(
            RealmId::System(p.realm_seed),
            LatticePos::local(BOX_B_CENTER),
            TRIGGER_R_SOI,
            1.15,
            1.30,
            p.move_speed,
            p.tick_dt,
            0.5,
            1.0,
            None,
            RealmId::System(p.realm_seed_b),
            CrossEffect::Authority,
        )]
    }

    /// The CLIENT render scene: TWO static region boxes keyed by their OWN realm (box A under SOURCE,
    /// box B under DEST) so `RealmScene::from_boundaries` draws two distinct boxes and `expected_box`
    /// resolves each. `v_rel = 0`: inert render geometry, never a crossing (distinct from [`trigger`]).
    #[must_use]
    pub fn scene(p: &DevClusterParams) -> Vec<RealmBoundary> {
        vec![
            region_box(RealmId::System(p.realm_seed), BOX_A_CENTER),
            region_box(RealmId::System(p.realm_seed_b), BOX_B_CENTER),
        ]
    }

    fn region_box(realm: RealmId, center: DVec3) -> RealmBoundary {
        RealmBoundary::aabb(
            realm,
            LatticePos::local(center),
            BOX_HALF,
            1.15,
            1.30,
            0.0,
            0.05,
            0.5,
            1.0,
            None,
            realm,
            CrossEffect::Authority,
        )
        .expect("valid region box")
    }

    /// Write the two playground fixtures into `dir` — `crossing-trigger.json` (the shard trigger, for
    /// `VD_DEVCLUSTER_BOUNDARIES`) and `crossing-scene.json` (the client's `--realm-boxes` two-box
    /// scene) — returning `(trigger_path, scene_path)`. Both are `Vec<RealmBoundary>` JSON, the same
    /// format the shard boot-loads and the client renders.
    ///
    /// # Errors
    /// A directory-create, serialize, or write failure.
    pub fn write_fixtures(
        dir: &std::path::Path,
        p: &DevClusterParams,
    ) -> Result<(String, String), String> {
        std::fs::create_dir_all(dir).map_err(|e| format!("create fixtures dir: {e}"))?;
        let trigger_path = dir.join("crossing-trigger.json");
        let scene_path = dir.join("crossing-scene.json");
        std::fs::write(
            &trigger_path,
            serde_json::to_string(&trigger(p)).map_err(|e| format!("serialize trigger: {e}"))?,
        )
        .map_err(|e| format!("write trigger: {e}"))?;
        std::fs::write(
            &scene_path,
            serde_json::to_string(&scene(p)).map_err(|e| format!("serialize scene: {e}"))?,
        )
        .map_err(|e| format!("write scene: {e}"))?;
        Ok((
            trigger_path.display().to_string(),
            scene_path.display().to_string(),
        ))
    }

    /// C-6b SINGLE-SOURCE: write the SEED-DERIVED containment forest (`realm_regions_for(seed)`) as
    /// `regions.json` into `dir`, returning the path. The client loads it via `--realm-boxes` (which tries
    /// `RealmScene::from_regions_json` first), so it draws EXACTLY the geometry the shard's detector
    /// evaluates — the canonical Universe⊃Galaxy⊃System⊃Planet forest, ambient shells auto-skipped by the
    /// renderable-extent filter. This is the single-sourced canonical render (distinct from
    /// [`write_fixtures`]'s authored two-box OVERRIDE scene used by the direct-re-home smoke).
    ///
    /// # Errors
    /// A directory-create, serialize, or write failure.
    pub fn write_seed_regions(dir: &std::path::Path, universe_seed: u64) -> Result<String, String> {
        write_regions_json(dir, &vd_core::worldgen::realm_regions_for(universe_seed))
    }

    /// FA-5 VISUAL-scale twin of [`write_seed_regions`]: write the visual single-system forest
    /// (`realm_regions_for_config(seed, visual_scale())` — the SAME forest a `VD_UNIVERSE_SCALE=visual`
    /// shard plants, single-sourced) as `regions.json`, so the client's `--realm-boxes` draws EXACTLY the
    /// orbiting-planet system the shard authors + ships (the planets ORBIT via the realm-frame overlay; the
    /// static `regions.json` seeds their tick-0 positions, the live feed animates them).
    ///
    /// # Errors
    /// A directory-create, serialize, or write failure.
    pub fn write_visual_regions(
        dir: &std::path::Path,
        universe_seed: u64,
    ) -> Result<String, String> {
        let config = vd_core::worldgen::UniverseConfig::visual_scale();
        write_regions_json(
            dir,
            &vd_core::worldgen::realm_regions_for_config(universe_seed, &config),
        )
    }

    /// Serialize a `RealmRegion` forest to `<dir>/regions.json` (the `--realm-boxes` format), returning
    /// the path — the shared body of [`write_seed_regions`] / [`write_visual_regions`] (single-sourced).
    fn write_regions_json(
        dir: &std::path::Path,
        regions: &[vd_core::geometry::RealmRegion],
    ) -> Result<String, String> {
        std::fs::create_dir_all(dir).map_err(|e| format!("create regions dir: {e}"))?;
        let path = dir.join("regions.json");
        let json = serde_json::to_string(regions).map_err(|e| format!("serialize regions: {e}"))?;
        std::fs::write(&path, json).map_err(|e| format!("write regions: {e}"))?;
        Ok(path.display().to_string())
    }
}

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

// ---- the shard boundary-plant OVERRIDE knob (C-6b; the seed forest is the default) ----------

/// A `VD_REALM_BOUNDARIES` file the shard could not turn into a valid, in-realm boundary set — rejected
/// LOUD at shard boot rather than silently booting an empty (inert) trigger. (Manual `Display`/`Error`
/// impls: `vd-bins` does not pull `thiserror`.) The two arms mirror the client's `--realm-boxes` failure
/// modes plus a shard-only config-drift guard (a boundary for a realm THIS shard does not host).
#[derive(Clone, Debug, PartialEq)]
pub enum RealmBoundariesError {
    /// The file could not be read or its JSON did not parse as `Vec<RealmBoundary>`.
    Malformed(String),
    /// A boundary's exterior `realm` is NOT the realm this shard hosts — a config drift (the same
    /// `boundaries.json` was handed to the wrong shard). Carries the offending + hosted realm.
    WrongRealm {
        boundary_realm: vd_core::pose::RealmId,
        hosted_realm: vd_core::pose::RealmId,
    },
}

impl std::fmt::Display for RealmBoundariesError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            RealmBoundariesError::Malformed(why) => write!(
                f,
                "VD_REALM_BOUNDARIES could not be loaded as a Vec<RealmBoundary>: {why} \
                 (the file is single-sourced with the client's --realm-boxes boxes.json)"
            ),
            RealmBoundariesError::WrongRealm {
                boundary_realm,
                hosted_realm,
            } => write!(
                f,
                "VD_REALM_BOUNDARIES has a boundary for realm {boundary_realm} but this shard hosts \
                 realm {hosted_realm} — a config drift (the wrong boundaries.json for this shard); the \
                 boundary's exterior `realm` must be the realm the shard is authoritative for"
            ),
        }
    }
}

impl std::error::Error for RealmBoundariesError {}

/// Resolve the shard's `VD_REALM_BOUNDARIES` OVERRIDE (C-6b — the process-seam playground knob). Reads
/// `VD_REALM_BOUNDARIES` (a path to a `boundaries.json` = a `Vec<RealmBoundary>`, the IDENTICAL format the
/// client loads as `--realm-boxes` — SINGLE-SOURCED), parses it, and validates every boundary's exterior
/// `realm` is the realm THIS shard hosts (`hosted_realm`, a config-drift guard). Returns:
/// - `Ok(None)` when the env var is ABSENT — the caller falls back to the SEED-DERIVED containment
///   neighbourhood (`realm_neighbourhood_for`), the production default (the detector is LIVE either way).
/// - `Ok(Some(vec))` when the file loads and every boundary is in-realm — the caller plants the authored
///   born-inside child crossing forest (the `dual_cluster_crossing_smoke` / `render_crossing_smoke` path).
///
/// Fails LOUD ([`RealmBoundariesError`]) on a malformed file or a boundary for a realm this shard
/// does not host — a misconfiguration must never become a silent wrong-geometry trigger.
///
/// # Errors
/// [`RealmBoundariesError`] on a read/parse failure or an out-of-realm boundary.
pub fn resolve_realm_boundaries(
    env: &EnvConfig,
    hosted_realm: vd_core::pose::RealmId,
) -> Result<Option<Vec<vd_core::geometry::RealmBoundary>>, RealmBoundariesError> {
    // ABSENT ⇒ inert (Ok(None)); this is the ONLY non-error absence path. A present path that fails to
    // read/parse/validate is LOUD.
    let Ok(path) = env.string("VD_REALM_BOUNDARIES") else {
        return Ok(None);
    };
    let json = std::fs::read_to_string(&path)
        .map_err(|e| RealmBoundariesError::Malformed(format!("read {path}: {e}")))?;
    let boundaries = parse_realm_boundaries(&json)?;
    guard_boundaries_in_realm(&boundaries, hosted_realm)?;
    Ok(Some(boundaries))
}

/// The world-SCALE a shard boots (D-45(a) FA-5). `Walk` is the production/byte-identity default (the P3
/// walk-scale mandate forest — all `StaticOffset`, no movers); `Visual` is the FA-5 window test: a single
/// star system whose planets ORBIT (the `worldgen::visual_scale` synthetic-scale preset — the STATIC-render
/// expression of the one compressed-real geometry); `VisualDemand` is that SAME geometry with the per-realm
/// AoI band turned LIVE (`worldgen::visual_demand`) — the demand-cluster expression, so a moving occupant
/// drives demand-driven realm spin-up/down.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum UniverseScale {
    Walk,
    Visual,
    /// RLM realistic-demo Slice 3: the compressed-real 5-planet Kepler geometry (IDENTICAL to `Visual`) with
    /// the per-realm AoI band LIVE (`VD_UNIVERSE_SCALE=visual-demand`) — a demand cluster boots this so a
    /// moving occupant's `evaluate_realm_aoi` spins the child realms up as they cross into visibility
    /// (angular size ≥ θ_min, i.e. within `extent · cot(θ/2)`) and reaps them as they fall back out. The
    /// STATIC-render twin is `Visual`; the two share `worldgen::visual_geometry` (one geometry, two drives).
    VisualDemand,
}

/// Resolve `VD_UNIVERSE_SCALE`: ABSENT / empty / `walk` ⇒ [`UniverseScale::Walk`] (the byte-identical
/// production default — EVERY existing shard boot is unchanged); `visual` ⇒ [`UniverseScale::Visual`] (the
/// compressed-real star system, STATIC render); `visual-demand` ⇒ [`UniverseScale::VisualDemand`] (the SAME
/// geometry with the LIVE AoI band — the demand cluster); ANY other value fails LOUD at boot (never a silent
/// degrade to Walk).
///
/// # Errors
/// [`ConfigError::Unparseable`] for an unrecognized value.
pub fn resolve_universe_scale(env: &EnvConfig) -> Result<UniverseScale, ConfigError> {
    universe_scale_of(env.string("VD_UNIVERSE_SCALE").unwrap_or_default().trim())
}

/// The `str -> UniverseScale` map (monomorphic, off the env body so each arm is covered once, HR5).
fn universe_scale_of(raw: &str) -> Result<UniverseScale, ConfigError> {
    match raw {
        "" | "walk" => Ok(UniverseScale::Walk),
        "visual" => Ok(UniverseScale::Visual),
        "visual-demand" => Ok(UniverseScale::VisualDemand),
        other => Err(ConfigError::Unparseable {
            key: "VD_UNIVERSE_SCALE".to_owned(),
            value: other.to_owned(),
        }),
    }
}

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
        "VD_UNIVERSE_SCALE",
        "VD_OUTBOUND_CAP",
        // A forked shard's must-parse boot params (consumed only inside `spawn_realm`).
        "VD_MINT_SEED",
        "VD_INPUT_LOG_CAP",
        // RLM 5f-4: a demand-spawned shard needs these LIVE too — WITHOUT VD_BOOT_TICKS_P99 its predictive
        // AoI horizon is 0 (the walk-in predictive spin-up is silently dead); WITHOUT VD_SPAWN_POSES the
        // gateway's server-derived login home and the shard's admit pose disagree; WITHOUT
        // VD_LEASE_RENEW_INTERVAL its realm lease is never renewed (liveness inert). The self-fence RECHECK
        // is deliberately NOT forwarded — the shard's `resolve_node_d3` self-derives a sane active default
        // from the `n` key (there is no `VD_REALM_RECHECK` read; a spawned shard defaults to hz/2).
        "VD_BOOT_TICKS_P99",
        "VD_SPAWN_POSES",
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
/// source keeps speaking for its departing occupant — telling its parent where they are, and counting
/// itself occupied — for the same one. Both read `derive_arrival_shield_ticks`, so the two ends cannot
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

/// Resolve the shard's per-account STORED spawn poses (RLM 5f-3b) from the `VD_SPAWN_POSES` STAND-IN — the
/// seam the P7 durable per-account pose store replaces with ZERO caller reshape (it fills the SAME
/// `StubConfig::spawn_poses` map). A login for an account with a stored pose is admitted at THAT pose
/// (organic bootstrap) instead of the origin; the shard-side lookup is keyed by the `AccountId` the
/// `AttachSession` arm already carries, so this grows NO frozen wire.
///
/// Format: `account=x,y,z` entries separated by `;`, where `account` is the decimal [`AccountId`] u128 and
/// `x,y,z` the ABSOLUTE Universe-root position (the `SystemSpace{system_seed: 0}` frame `container_coord_at`
/// reads), at rest. ABSENT / empty ⇒ an EMPTY map ⇒ every login births origin-at-rest (BYTE-IDENTICAL to
/// the pre-5f-3b boot). Models `resolve_time_multiplier` (an `unwrap_or_default` read + a monomorphic parse).
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

/// A boot's seed-derived world: the shard's realm REGIONS, the per-realm orbital ELEMENTS of the movers it
/// authors, and the per-realm ORIGIN CHAINS each realm folds its own absolute position from (A5). Named
/// because both the visual and the scale boot arms return exactly this triple.
type BootWorld = (
    Vec<vd_core::geometry::RealmRegion>,
    std::collections::BTreeMap<vd_core::pose::RealmId, vd_core::celestial::OrbitalElements>,
    std::collections::BTreeMap<vd_core::pose::RealmId, Vec<vd_core::worldgen::OriginLink>>,
);

/// Build the containment forest + moving-child roster for a `Visual`/`VisualDemand` config, applying the
/// DEV/TEST orbit-speed knob `VD_VISUAL_ORBIT_SLOWDOWN` (absent / <= 0 ⇒ 1.0 = the SHIPPED orbits,
/// byte-identical): dividing the synthetic star mass by K² makes every Kepler period K× longer
/// (`T = 2π·√(a³/GM)` ⇒ `T ∝ 1/√M`), so a pilot can CATCH a planet before real physics (P5) drags them
/// along — a temporary flight-aid, server-side only (the client is untouched, and the tick-0 region centers
/// the client's box scene loads are mass-INDEPENDENT, so the static scene still matches). Unset it to restore
/// the shipped orbit speed. Shared by BOTH visual arms (DRY) so the static-render (`Visual`) and live-demand
/// (`VisualDemand`) modes stay ONE geometry expression.
fn visual_regions_and_movers(
    mut config: vd_core::worldgen::UniverseConfig,
    universe_seed: u64,
    held: &std::collections::BTreeSet<vd_core::pose::RealmId>,
    hosted: vd_core::pose::RealmId,
) -> BootWorld {
    let orbit_slowdown = std::env::var("VD_VISUAL_ORBIT_SLOWDOWN")
        .ok()
        .and_then(|s| s.parse::<f64>().ok())
        .filter(|k| *k > 0.0)
        .unwrap_or(1.0);
    config.stellar.central_mass_kg /= orbit_slowdown * orbit_slowdown;
    // The NEIGHBOURHOOD (own realm + ancestor chain + the children this shard authors) — NEVER its sibling
    // planets. A shard cannot place a realm it does not author, so folding siblings collapses them to the
    // origin and a hosted occupant reads as inside all of them at once (the production re-home flap). Scoping
    // them out is the standalone cure; the movers stay `hosted`'s authored children (single-realm demand shard
    // ⇒ exactly its own; the union-over-held is a co-hosting refinement not yet needed at visual scale).
    let regions = vd_core::worldgen::realm_neighbourhood_for_config(universe_seed, held, &config);
    let moving = vd_core::worldgen::moving_children_for_config(universe_seed, &config, hosted)
        .into_iter()
        .collect();
    // A5 — the seed origin chain for EVERY neighbourhood realm, built from the SAME (seed, mutated config) that
    // just built the regions + movers, so the shard's folded absolutes (`FrameAbs`) can never derive from
    // different elements than its region set or its authored orbits. The chains ARM the server-authoritative
    // compose: `RealmRegions::origin_abs_of`/`frame_abs_map` fold them each tick, and every emitted pose is
    // composed to root-absolute before it ships.
    let origin_chains = regions
        .iter()
        .map(|r| {
            (
                r.realm,
                vd_core::worldgen::origin_chain_for_config(universe_seed, &config, r.realm),
            )
        })
        .collect();
    (regions, moving, origin_chains)
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
#[must_use]
pub fn boot_regions_and_movers(
    scale: UniverseScale,
    universe_seed: u64,
    held_realms: &std::collections::BTreeSet<vd_core::pose::RealmId>,
    hosted: vd_core::pose::RealmId,
    occupant_v_max_mps: f64,
    tick_dt_s: f64,
) -> BootWorld {
    use std::collections::BTreeMap;
    match scale {
        // Walk ⇒ an EMPTY chain roster: every walk body is `Fixed`, so a folded chain is the identity and an
        // empty roster (`FrameAbs` stays empty ⇒ the compose short-circuits to the raw frame-local pose) is
        // byte-identical to folding all-Fixed chains. Keeps the walk fixtures bit-for-bit unchanged post-flip.
        UniverseScale::Walk => (
            vd_core::worldgen::realm_neighbourhood_for_held(universe_seed, held_realms),
            BTreeMap::new(),
            BTreeMap::new(),
        ),
        UniverseScale::Visual => visual_regions_and_movers(
            vd_core::worldgen::UniverseConfig::visual_scale(),
            universe_seed,
            held_realms,
            hosted,
        ),
        // RLM realistic-demo Slice 3: the SAME compressed-real geometry as `Visual`, with the AoI band built
        // LIVE from the cluster's occupant speed + tick dt (`visual_demand`). Both visual arms share the one
        // build path (+ the orbit-slowdown knob) via `visual_regions_and_movers` (DRY).
        UniverseScale::VisualDemand => visual_regions_and_movers(
            vd_core::worldgen::UniverseConfig::visual_demand(occupant_v_max_mps, tick_dt_s),
            universe_seed,
            held_realms,
            hosted,
        ),
    }
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
    scale: UniverseScale,
    universe_seed: u64,
    occupant_v_max_mps: f64,
    tick_dt_s: f64,
) -> vd_core::worldgen::WorldView {
    use vd_core::worldgen::{UniverseConfig, WorldView};
    match scale {
        UniverseScale::Walk => WorldView::hand_placed(&UniverseConfig::walk_scale()),
        UniverseScale::Visual => {
            WorldView::generated(universe_seed, &UniverseConfig::visual_scale())
        }
        UniverseScale::VisualDemand => WorldView::generated(
            universe_seed,
            &UniverseConfig::visual_demand(occupant_v_max_mps, tick_dt_s),
        ),
    }
}

/// Parse a `boundaries.json` string into a `Vec<RealmBoundary>` (the SAME serde shape the client's
/// `RealmScene::from_boxes_json` loads — single-sourced). A monomorphic helper so the parse-error arm
/// is covered off the env/fs body.
fn parse_realm_boundaries(
    json: &str,
) -> Result<Vec<vd_core::geometry::RealmBoundary>, RealmBoundariesError> {
    serde_json::from_str(json).map_err(|e| RealmBoundariesError::Malformed(e.to_string()))
}

/// Serialize a co-hosted realm SET to the `VD_HELD_REALMS` env format (`kind:seed` comma-separated, e.g.
/// `system:7,planet:7,station:7,area:7`) — the wire the --triple launcher hands a co-hosting shard. Sorted
/// (BTreeSet order) so the env is deterministic; the round-trip inverse is [`parse_held_realms`].
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

/// The containment ACQUIRE edge sits this fraction of a region's OWN extent inside its surface. Fractional
/// (not a fixed metre) so it is reachable at ANY scale — a fixed inset is unreachable inside a shell SMALLER
/// than the inset (the r=10 crossing-trigger shell vs a fixed inset=50: acquire needed a dot 50 m inside a
/// 10 m shell ⇒ NO dot ever became a member ⇒ the walk-across crossing never fired). Mirrors how
/// `OverlapBand`'s SOI factors scale with `r_soi`.
const OVERRIDE_BAND_INSET_FRACTION: f64 = 0.1;
/// The RELEASE edge, a larger fraction OUTSIDE the surface — the dead-zone (inset+outset) straddles the
/// surface so a surface-hovering dot cannot flap.
const OVERRIDE_BAND_OUTSET_FRACTION: f64 = 0.2;

/// A [`ContainmentBand`](vd_core::geometry::ContainmentBand) SIZED TO `shape`'s own extent (a fraction of a
/// shell's radius / a box's smallest half-extent), velocity-widened for a body moving at `move_speed_mps`.
/// Scale-invariant: the acquire edge is reachable inside the shape whether it is a 10 m crossing shell or a
/// 100 km realm — the fixed-metre-inset bug this replaces made small shells un-enterable.
fn override_containment_band(
    shape: &vd_core::geometry::Boundary,
    move_speed_mps: f64,
    tick_dt_s: f64,
) -> vd_core::geometry::ContainmentBand {
    use vd_core::geometry::{Boundary, ContainmentBand};
    let extent = match shape {
        Boundary::Shell { r } => *r,
        Boundary::Aabb { half } | Boundary::Obb { half, .. } => half.min_element(),
    };
    let inset = extent * OVERRIDE_BAND_INSET_FRACTION;
    let outset_min = extent * OVERRIDE_BAND_OUTSET_FRACTION;
    ContainmentBand::for_containment_velocity_safe(
        inset,
        outset_min,
        move_speed_mps,
        tick_dt_s,
        1.0,
    )
    .unwrap_or_else(|_| {
        // A degenerate tick config can only shrink the velocity term; the positive-fraction edges of a
        // positive extent are valid at v_rel=0, so the plant never panics the shard boot.
        ContainmentBand::for_containment_velocity_safe(inset, outset_min, 0.0, 1.0, 0.0)
            .expect("a positive-fraction band of a positive extent is valid")
    })
}

/// The `VD_REALM_BOUNDARIES` OVERRIDE adapter (C-6b) — lift the SOURCE-shard's loaded `RealmBoundary` set
/// (the born-inside crossing geometry, single-sourced with the client's `--realm-boxes`) into the
/// containment [`RealmRegion`](vd_core::geometry::RealmRegion) forest the sim's `RealmRegions` resource
/// consumes. This is the PLAYGROUND override, NOT the production boot: the default boot computes the
/// SEED-DERIVED neighbourhood (`vd_core::worldgen::realm_neighbourhood_for`) directly, where a star-system's
/// disjoint sibling is reached THROUGH the shared Galaxy parent. The override models the crossing dest as a
/// DEEPER CHILD region whose `realm = boundary.to_realm`, nested under the shard's `hosted_realm`, which
/// nests under an ambient root — so a dot born INSIDE the shell has its deepest container == `to_realm` and
/// re-homes there in ONE step. This lets `dual_cluster_crossing_smoke` / `render_crossing_smoke` prove a
/// DIRECT source→dest re-home over the process tier WITHOUT standing up a Galaxy shard (the seed-forest
/// escape-SOI-to-sibling round-trip is the harness-tier gate `three_shard_round_trip_*`). The on-disk JSON
/// (and the client `--realm-boxes` scene) are UNCHANGED (still `RealmBoundary`); this converts only at the
/// shard's plant seam. The band is re-derived velocity-safe from the shard's tick params. The result is
/// `guard_regions_nest`-validated at the plant seam like the seed forest (an authored fixture is fenced too).
#[must_use]
pub fn override_regions_for_boundaries(
    boundaries: &[vd_core::geometry::RealmBoundary],
    hosted_realm: vd_core::pose::RealmId,
    move_speed_mps: f64,
    tick_dt_s: f64,
) -> Vec<vd_core::geometry::RealmRegion> {
    use vd_core::geometry::{Boundary, RealmRegion};
    use vd_core::pose::{LatticePos, RealmId, frame_for_realm};
    let root_realm = RealmId::System(0);
    let region = |realm: RealmId, parent: Option<RealmId>, shape: Boundary| RealmRegion {
        realm,
        center: LatticePos::local(vd_core::glam::DVec3::ZERO),
        frame: frame_for_realm(realm, None)
            .unwrap_or(vd_core::pose::FrameRef::SystemSpace { system_seed: 0 }),
        // The band is SIZED TO the region's own extent (a fraction of its radius), never a fixed metre —
        // see `override_containment_band` (the r=10-shell / inset=50 un-enterable bug).
        band: override_containment_band(&shape, move_speed_mps, tick_dt_s),
        shape,
        aoi: vd_core::geometry::AoiConfig::inert(),
        parent,
    };
    // Ambient root ⊃ the shard's own realm (large) ⊃ one deeper child per loaded boundary (its to_realm).
    let mut regions = vec![
        region(root_realm, None, Boundary::Shell { r: 1.0e9 }),
        region(
            hosted_realm,
            Some(root_realm),
            Boundary::Shell { r: 100_000.0 },
        ),
    ];
    for b in boundaries
        .iter()
        .filter(|b| b.effect == vd_core::geometry::CrossEffect::Authority)
    {
        // Nest the destination realm as a deeper AUTHORITY child, REUSING the boundary's own shape + center
        // so the client's `--realm-boxes` projection (same geometry) still frames the dot inside it. Only
        // Authority boundaries become containment regions — an Interest boundary (a ghost-only zone) is NOT
        // an authority re-home, so it is dropped here (the interim adapter carries no Interest path; C-6).
        regions.push(RealmRegion {
            realm: b.to_realm,
            center: b.center,
            frame: frame_for_realm(b.to_realm, None)
                .unwrap_or(vd_core::pose::FrameRef::SystemSpace { system_seed: 0 }),
            band: override_containment_band(&b.shape, move_speed_mps, tick_dt_s),
            shape: b.shape,
            aoi: vd_core::geometry::AoiConfig::inert(),
            parent: Some(hosted_realm),
        });
    }
    regions
}

/// The config-drift guard: every boundary's exterior `realm` must equal the realm this shard hosts. A
/// monomorphic helper so the reject arm is covered off the env/fs body.
fn guard_boundaries_in_realm(
    boundaries: &[vd_core::geometry::RealmBoundary],
    hosted_realm: vd_core::pose::RealmId,
) -> Result<(), RealmBoundariesError> {
    for b in boundaries {
        if b.realm != hosted_realm {
            return Err(RealmBoundariesError::WrongRealm {
                boundary_realm: b.realm,
                hosted_realm,
            });
        }
    }
    Ok(())
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
mod override_band_tests {
    use super::override_containment_band;
    use vd_core::geometry::Boundary;
    use vd_core::glam::DVec3;

    #[test]
    fn override_band_scales_to_the_region_extent_and_stays_reachable() {
        // The crossing-playground trigger scale: a small r=10 shell. The acquire inset must land WELL INSIDE
        // it so a dot walking to box B's centre becomes a member — the OLD fixed inset=50 made acquire need a
        // dot 50 m inside a 10 m shell (impossible ⇒ the walk-across crossing silently never fired).
        let band = override_containment_band(&Boundary::Shell { r: 10.0 }, 2.0, 0.05);
        assert!(
            band.inset() < 10.0,
            "the acquire inset {} must sit inside the r=10 shell",
            band.inset(),
        );
        // A dot 7 m inside the surface (signed distance -7) ACQUIRES; with the old inset=50 it would not.
        assert!(
            band.member(false, -7.0),
            "a dot 7 m inside a r=10 crossing shell must ACQUIRE membership",
        );
        // Scale-invariant: a 100 km realm gets a proportionally larger (still-reachable) band.
        let big = override_containment_band(&Boundary::Shell { r: 100_000.0 }, 2.0, 0.05);
        assert!(
            big.inset() > band.inset(),
            "the band scales with the region extent (not a fixed metre)",
        );
        // A box region sizes off its smallest half-extent (covers the Aabb/Obb arm of the extent match).
        let boxed = override_containment_band(
            &Boundary::Aabb {
                half: DVec3::splat(5.0),
            },
            2.0,
            0.05,
        );
        assert!(
            boxed.member(false, -3.0),
            "a dot 3 m inside a 5 m box half-extent must acquire membership",
        );
    }
}

#[cfg(test)]
mod incarnation_tests {
    use super::*;
    use std::collections::BTreeMap;

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

    // ---- the shard boundary-plant knob (DEFERRED 1-SIGKILL-OWED) --------------

    use vd_core::geometry::{CrossEffect, RealmBoundary};
    use vd_core::glam::DVec3;
    use vd_core::pose::{LatticePos, RealmId};

    /// One in-`realm` AUTHORITY shell whose `to_realm` is a distinct dest — the shape the crossing
    /// smoke plants (single-sourced with `plant_one_crossing_shell`).
    fn shell(realm: RealmId, to_realm: RealmId) -> RealmBoundary {
        RealmBoundary::shell(
            realm,
            LatticePos::local(DVec3::ZERO),
            1000.0,
            1.15,
            1.30,
            2.0,
            0.02,
            0.5,
            1.0,
            None,
            to_realm,
            CrossEffect::Authority,
        )
    }

    /// Write `boundaries` as the client-identical `boxes.json` into a fresh temp file, returning the
    /// path — so the loader is exercised end-to-end (fs → serde → guard), like the shard's boot read.
    fn write_boundaries(tag: &str, boundaries: &[RealmBoundary]) -> (TempDir, String) {
        let dir = TempDir::new(tag);
        std::fs::create_dir_all(&dir.0).expect("mk boundaries dir");
        let path = dir.file("boundaries.json");
        std::fs::write(&path, serde_json::to_string(boundaries).expect("serialize"))
            .expect("write");
        (dir, path)
    }

    #[test]
    fn resolve_realm_boundaries_is_inert_when_absent() {
        // ABSENT env var ⇒ Ok(None): the caller leaves the default-empty resource, byte-identical to today.
        let hosted = RealmId::System(7);
        assert_eq!(resolve_realm_boundaries(&env(&[]), hosted), Ok(None));
    }

    #[test]
    fn resolve_realm_boundaries_loads_an_in_realm_file() {
        // A present, well-formed, in-realm file ⇒ Ok(Some(vec)) with the exact planted boundary.
        let hosted = RealmId::System(7);
        let planted = vec![shell(hosted, RealmId::System(8))];
        let (_dir, path) = write_boundaries("load", &planted);
        let got = resolve_realm_boundaries(&env(&[("VD_REALM_BOUNDARIES", &path)]), hosted)
            .expect("the in-realm file loads");
        assert_eq!(got, Some(planted));
    }

    #[test]
    fn resolve_realm_boundaries_is_loud_on_a_missing_file() {
        // A present path that does NOT exist ⇒ LOUD Malformed (never a silent inert boot).
        let hosted = RealmId::System(7);
        let missing = TempDir::new("missing").file("nope.json");
        let err = resolve_realm_boundaries(&env(&[("VD_REALM_BOUNDARIES", &missing)]), hosted)
            .expect_err("a missing file is loud");
        assert_eq!(
            std::mem::discriminant(&err),
            std::mem::discriminant(&RealmBoundariesError::Malformed(String::new())),
        );
        assert!(err.to_string().contains("VD_REALM_BOUNDARIES"));
    }

    #[test]
    fn resolve_realm_boundaries_is_loud_on_malformed_json() {
        // A present file that is not a Vec<RealmBoundary> ⇒ LOUD Malformed.
        let hosted = RealmId::System(7);
        let dir = TempDir::new("bad");
        std::fs::create_dir_all(&dir.0).expect("mk");
        let path = dir.file("bad.json");
        std::fs::write(&path, "not json at all").expect("write");
        let err = resolve_realm_boundaries(&env(&[("VD_REALM_BOUNDARIES", &path)]), hosted)
            .expect_err("malformed json is loud");
        assert_eq!(
            std::mem::discriminant(&err),
            std::mem::discriminant(&RealmBoundariesError::Malformed(String::new())),
        );
    }

    #[test]
    fn resolve_realm_boundaries_rejects_a_boundary_for_a_realm_this_shard_does_not_host() {
        // The config-drift guard: a boundary whose exterior `realm` is NOT the hosted realm ⇒ loud
        // WrongRealm (the wrong boxes.json handed to this shard), never a silently mis-planted band.
        let hosted = RealmId::System(7);
        let drifted = RealmId::System(999);
        let planted = vec![shell(drifted, RealmId::System(8))];
        let (_dir, path) = write_boundaries("drift", &planted);
        let err = resolve_realm_boundaries(&env(&[("VD_REALM_BOUNDARIES", &path)]), hosted)
            .expect_err("an out-of-realm boundary is loud");
        assert_eq!(
            err,
            RealmBoundariesError::WrongRealm {
                boundary_realm: drifted,
                hosted_realm: hosted,
            },
        );
        assert!(err.to_string().contains("config drift"));
    }

    #[test]
    fn guard_boundaries_in_realm_accepts_an_empty_set() {
        // The guard's loop-never-fires arm: an empty (but present) file is in-realm-vacuously OK.
        assert_eq!(guard_boundaries_in_realm(&[], RealmId::System(7)), Ok(()));
    }

    // ---- FA-5 S2: the universe-scale boot selector ---------------------------------------------

    #[test]
    fn resolve_universe_scale_defaults_to_walk_when_absent_empty_or_named() {
        // ABSENT ⇒ Walk (the byte-identical production default); empty + "walk" also Walk.
        assert_eq!(resolve_universe_scale(&env(&[])), Ok(UniverseScale::Walk));
        assert_eq!(
            resolve_universe_scale(&env(&[("VD_UNIVERSE_SCALE", "")])),
            Ok(UniverseScale::Walk),
        );
        assert_eq!(
            resolve_universe_scale(&env(&[("VD_UNIVERSE_SCALE", "walk")])),
            Ok(UniverseScale::Walk),
        );
    }

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
            ("VD_SPAWN_POSES", "5=1,2,3"),
            ("VD_LEASE_RENEW_INTERVAL", "25"),
        ]);
        assert_eq!(
            spawn_anchors_from_env(&armed),
            vec![
                ("VD_TICK_HZ", "50".to_owned()),
                ("VD_BOOT_TICKS_P99", "100".to_owned()),
                ("VD_SPAWN_POSES", "5=1,2,3".to_owned()),
                ("VD_LEASE_RENEW_INTERVAL", "25".to_owned()),
            ],
            "the 5f-4 keys reach the spawned shard so its predictive AoI, admit pose, and lease liveness live"
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
    fn resolve_universe_scale_selects_visual_trims_and_is_loud_on_unknown() {
        assert_eq!(
            resolve_universe_scale(&env(&[("VD_UNIVERSE_SCALE", "visual")])),
            Ok(UniverseScale::Visual),
        );
        // RLM realistic-demo Slice 3: "visual-demand" is the demand-cluster scale (the compressed-real
        // geometry with the LIVE AoI band).
        assert_eq!(
            resolve_universe_scale(&env(&[("VD_UNIVERSE_SCALE", "visual-demand")])),
            Ok(UniverseScale::VisualDemand),
        );
        // Surrounding whitespace is trimmed.
        assert_eq!(
            resolve_universe_scale(&env(&[("VD_UNIVERSE_SCALE", "  visual  ")])),
            Ok(UniverseScale::Visual),
        );
        // An unrecognized value fails LOUD (never a silent Walk).
        assert_eq!(
            resolve_universe_scale(&env(&[("VD_UNIVERSE_SCALE", "galaxy")])),
            Err(ConfigError::Unparseable {
                key: "VD_UNIVERSE_SCALE".to_owned(),
                value: "galaxy".to_owned(),
            }),
        );
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
    fn boot_regions_and_movers_walk_is_the_seed_neighbourhood_with_an_empty_roster() {
        use std::collections::BTreeSet;
        let hosted = RealmId::System(7);
        let held: BTreeSet<RealmId> = [hosted].into_iter().collect();
        // The two dynamics args are IGNORED by Walk — pass live-cluster-ish values to prove it.
        let (regions, moving, chains) =
            boot_regions_and_movers(UniverseScale::Walk, 0, &held, hosted, 2.0, 0.02);
        // Walk ⇒ EXACTLY realm_neighbourhood_for_held + an EMPTY mover roster (the pre-FA-5 boot; the
        // empty map makes with_moving_children a no-op vs new — byte-identical).
        assert_eq!(
            regions,
            vd_core::worldgen::realm_neighbourhood_for_held(0, &held),
        );
        assert!(moving.is_empty());
        // A5 — Walk carries an EMPTY origin-chain roster (all Fixed ⇒ identity fold ⇒ FrameAbs empty ⇒ the
        // server-authoritative compose short-circuits to raw ⇒ the walk fixtures stay bit-for-bit unchanged).
        assert!(chains.is_empty());
    }

    #[test]
    fn boot_regions_and_movers_visual_demand_is_the_visual_geometry_with_a_live_band() {
        use std::collections::BTreeSet;
        // RLM realistic-demo Slice 3: `VisualDemand` builds the SAME compressed-real geometry as `Visual`,
        // via `visual_demand(occupant_v_max, tick_dt)` — regions + the orbiting-planet mover roster from ONE
        // config, with the per-realm AoI band LIVE. The occupant speed is the demand cluster's 15 m/s ship.
        let hosted = RealmId::System(7);
        let held: BTreeSet<RealmId> = [hosted].into_iter().collect();
        let (regions, moving, chains) =
            boot_regions_and_movers(UniverseScale::VisualDemand, 0, &held, hosted, 15.0, 0.02);
        // The SAME (seed, config) builds BOTH the regions and the mover roster (they can't disagree). The
        // orbit-slowdown knob is unset in the unit rig ⇒ mass unchanged. Slice 1: the boot scopes to the
        // NEIGHBOURHOOD (own realm + ancestors + authored children, NEVER sibling planets) — the origin-stacking
        // re-home flap cure — so the regions are the neighbourhood of `held`, not the full system forest.
        let config = vd_core::worldgen::UniverseConfig::visual_demand(15.0, 0.02);
        assert_eq!(
            regions,
            vd_core::worldgen::realm_neighbourhood_for_config(0, &held, &config)
        );
        assert_eq!(
            moving,
            vd_core::worldgen::moving_children_for_config(0, &config, hosted)
                .into_iter()
                .collect(),
        );
        // The visual-demand boot actually plants orbiting planets (vs the empty walk roster).
        assert!(!moving.is_empty());
        // Every mover is a region in the planted forest, so frame_context can author its live pose.
        for realm in moving.keys() {
            assert!(regions.iter().any(|r| r.realm == *realm));
        }
        // The AoI band is LIVE (vs Walk's inert): every planet region carries a positive spin-up radius, so a
        // moving occupant's `evaluate_realm_aoi` can spin it up on visibility.
        for r in regions
            .iter()
            .filter(|r| matches!(r.realm, vd_core::pose::RealmId::Planet(_)))
        {
            assert!(
                r.aoi.spin_up_r_m() > 0.0,
                "visual-demand planet region has a LIVE AoI band: {r:?}",
            );
        }
        // A5 — every region carries a seed origin chain (keyed by realm), built from the SAME (seed, config) as
        // the regions + movers; every MOVING child's chain VARIES (ends in an Orbital link), so its folded
        // absolute rides the orbit and the server ships it root-absolute.
        for r in &regions {
            assert!(
                chains.contains_key(&r.realm),
                "region {:?} has a chain",
                r.realm
            );
        }
        for realm in moving.keys() {
            assert!(
                vd_core::worldgen::origin_varies(&chains[realm]),
                "mover {realm:?} has a varying (orbital) origin chain",
            );
        }
    }

    #[test]
    fn boot_regions_and_movers_visual_builds_the_system_forest_and_orbiting_planets() {
        use std::collections::BTreeSet;
        let hosted = RealmId::System(7);
        let held: BTreeSet<RealmId> = [hosted].into_iter().collect();
        let (regions, moving, chains) =
            boot_regions_and_movers(UniverseScale::Visual, 0, &held, hosted, 2.0, 0.02);
        let config = vd_core::worldgen::UniverseConfig::visual_scale();
        // Slice 1: the boot scopes to the NEIGHBOURHOOD (own + ancestors + authored children, never sibling
        // planets — the flap cure); the SAME (seed, config) still builds regions + movers so they can't disagree.
        assert_eq!(
            regions,
            vd_core::worldgen::realm_neighbourhood_for_config(0, &held, &config)
        );
        assert_eq!(
            moving,
            vd_core::worldgen::moving_children_for_config(0, &config, hosted)
                .into_iter()
                .collect(),
        );
        // The visual boot actually plants orbiting planets (vs the empty walk roster).
        assert!(!moving.is_empty());
        // Every mover is a region in the planted forest, so frame_context can author its live pose.
        for realm in moving.keys() {
            assert!(regions.iter().any(|r| r.realm == *realm));
        }
        // A5 — the origin chains match origin_chain_for_config over the SAME (seed, config); a mover's chain
        // varies (folds to its orbit), the star-system root's does not (all Fixed).
        for r in &regions {
            assert_eq!(
                chains[&r.realm],
                vd_core::worldgen::origin_chain_for_config(0, &config, r.realm),
            );
        }
        for realm in moving.keys() {
            assert!(vd_core::worldgen::origin_varies(&chains[realm]));
        }
    }

    // ---- Co-hosting: the VD_HELD_REALMS env round-trip + parse guards ---------------------------

    #[test]
    fn held_realms_env_round_trips_all_four_realm_kinds() {
        // Every RealmId kind serializes + re-parses (the `realm_token` / `parse_realm_token` arms). The
        // --triple source set is the exercised value; the parse fallback (System 7) is already in it.
        let set = triple_source_held_realms();
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

    #[test]
    fn triple_source_held_realms_is_system_7_plus_its_seed_forest_children() {
        // The --triple co-hosted set SINGLE-SOURCED: System 7 + Planet/Station/Area 7 (its seed-forest
        // children). The launcher, the shard boot, and the smoke all read this one value.
        assert_eq!(
            triple_source_held_realms(),
            std::collections::BTreeSet::from([
                RealmId::System(7),
                RealmId::Planet(7),
                RealmId::Station(7),
                RealmId::Area(7),
            ]),
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
        // RLM realistic-demo Slice 3: the demand cluster arms the compressed-real geometry with the LIVE AoI
        // band (visual-demand) so a MOVING occupant's evaluate_realm_aoi spins a child up as it crosses into
        // visibility and reaps it behind.
        assert_eq!(env_value(&d, "VD_UNIVERSE_SCALE"), Some("visual-demand"));
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
    fn orchestrator_env_dual_books_dest_clock_and_roster_inert_when_single() {
        let a = dual_addrs();
        let single = orchestrator_env(&a, &DEV, "store", ClusterShape::Single);
        let dual = orchestrator_env(&a, &DEV, "store", ClusterShape::Dual);

        // VD_ROSTER: absent single, present=DEST dual (the D-37 re-home candidate set).
        assert_eq!(env_value(&single, "VD_ROSTER"), None);
        assert_eq!(
            env_value(&dual, "VD_ROSTER"),
            Some(SHARD_B.0.to_string()).as_deref()
        );

        // VD_CLOCK_PEERS: {GW,SHARD} single vs {GW,SHARD,DEST} dual (DEST's follower clock must advance).
        assert_eq!(
            env_value(&single, "VD_CLOCK_PEERS"),
            Some(format!("{},{}", GATEWAY.0, SHARD.0)).as_deref()
        );
        assert_eq!(
            env_value(&dual, "VD_CLOCK_PEERS"),
            Some(format!("{},{},{}", GATEWAY.0, SHARD.0, SHARD_B.0)).as_deref()
        );

        // VD_PEERS: the DEST is booked ONLY in dual (each initiator books the DEST).
        let dest_book = format!("{}={}", SHARD_B.0, a.shard_b);
        assert!(
            !env_value(&single, "VD_PEERS")
                .expect("VD_PEERS is always emitted")
                .contains(&dest_book)
        );
        assert!(
            env_value(&dual, "VD_PEERS")
                .expect("VD_PEERS is always emitted")
                .contains(&dest_book)
        );
    }

    #[test]
    fn orchestrator_env_triple_books_dest_and_galaxy_in_roster_clock_and_peers() {
        // S5b: the Triple orchestrator drives the clock to BOTH extra shards and lists both in the
        // re-home roster, so the Galaxy (the between-space parent) wins its lease and rests authority.
        let a = dual_addrs();
        let triple = orchestrator_env(&a, &DEV, "store", ClusterShape::Triple);

        // VD_ROSTER = {DEST, GALAXY} (in the shape's extend order: DEST then GALAXY).
        assert_eq!(
            env_value(&triple, "VD_ROSTER"),
            Some(format!("{},{}", SHARD_B.0, GALAXY.0)).as_deref()
        );
        // VD_CLOCK_PEERS = {GW, SHARD, DEST, GALAXY} — every follower's clock must advance to win its lease.
        assert_eq!(
            env_value(&triple, "VD_CLOCK_PEERS"),
            Some(format!(
                "{},{},{},{}",
                GATEWAY.0, SHARD.0, SHARD_B.0, GALAXY.0
            ))
            .as_deref()
        );
        // VD_PEERS books BOTH extra shards.
        let peers = env_value(&triple, "VD_PEERS").expect("VD_PEERS is always emitted");
        assert!(peers.contains(&format!("{}={}", SHARD_B.0, a.shard_b)));
        assert!(peers.contains(&format!("{}={}", GALAXY.0, a.galaxy)));
    }

    #[test]
    fn gateway_env_dual_books_dest_and_emits_known_shards_inert_when_single() {
        let a = dual_addrs();
        let single = gateway_env(&a, &[], "pub", &DEV, ClusterShape::Single);
        let dual = gateway_env(&a, &[], "pub", &DEV, ClusterShape::Dual);

        // VD_KNOWN_SHARDS: absent single, =DEST dual (so a DEST frame is node-class dispatchable).
        assert_eq!(env_value(&single, "VD_KNOWN_SHARDS"), None);
        assert_eq!(
            env_value(&dual, "VD_KNOWN_SHARDS"),
            Some(SHARD_B.0.to_string()).as_deref()
        );

        // The DEST peer is booked ONLY in dual; VD_SHARD (the login shard) stays SHARD in both.
        let dest_book = format!("{}={}", SHARD_B.0, a.shard_b);
        assert!(
            !env_value(&single, "VD_PEERS")
                .expect("VD_PEERS is always emitted")
                .contains(&dest_book)
        );
        assert!(
            env_value(&dual, "VD_PEERS")
                .expect("VD_PEERS is always emitted")
                .contains(&dest_book)
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
    fn gateway_env_triple_books_dest_and_galaxy_and_lists_both_known_shards() {
        // S5b: the Triple gateway must BOOK both extra shards (so the durable session-route can swap to
        // each successive source across the multi-hop) and class both as known (so each shard→gateway
        // frame is node-class dispatchable), while VD_SHARD (the login shard) stays SHARD.
        let a = dual_addrs();
        let triple = gateway_env(&a, &[], "pub", &DEV, ClusterShape::Triple);

        // VD_KNOWN_SHARDS = {DEST, GALAXY}.
        assert_eq!(
            env_value(&triple, "VD_KNOWN_SHARDS"),
            Some(format!("{},{}", SHARD_B.0, GALAXY.0)).as_deref()
        );
        // Both extra peers are booked; the login shard is unchanged.
        let peers = env_value(&triple, "VD_PEERS").expect("VD_PEERS is always emitted");
        assert!(peers.contains(&format!("{}={}", SHARD_B.0, a.shard_b)));
        assert!(peers.contains(&format!("{}={}", GALAXY.0, a.galaxy)));
        assert_eq!(
            env_value(&triple, "VD_SHARD"),
            Some(SHARD.0.to_string()).as_deref()
        );
    }

    #[test]
    fn shard_env_dual_books_the_other_shard_inert_when_single() {
        let a = dual_addrs();
        let single = shard_env(&a, &DEV, ClusterShape::Single);
        let dual = shard_env(&a, &DEV, ClusterShape::Dual);
        let dest_book = format!("{}={}", SHARD_B.0, a.shard_b);
        // Each shard books the other ONLY in dual (the cross-shard mesh); single is byte-identical.
        assert!(
            !env_value(&single, "VD_PEERS")
                .expect("VD_PEERS is always emitted")
                .contains(&dest_book)
        );
        assert!(
            env_value(&dual, "VD_PEERS")
                .expect("VD_PEERS is always emitted")
                .contains(&dest_book)
        );
        // The SOURCE shard is realm A in both modes; the boundaries are planted by the launcher, not here.
        assert_eq!(
            env_value(&dual, "VD_REALM_SEED"),
            Some(DEV.realm_seed.to_string()).as_deref()
        );
        assert_eq!(env_value(&dual, "VD_REALM_BOUNDARIES"), None);
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

    #[test]
    fn shard_b_env_is_the_dest_realm_with_a_distinct_mint_and_books_the_source() {
        let a = dual_addrs();
        let env = shard_b_env(&a, &DEV, ClusterShape::Dual);
        assert_eq!(
            env_value(&env, "VD_NODE_ID"),
            Some(SHARD_B.0.to_string()).as_deref()
        );
        assert_eq!(
            env_value(&env, "VD_BIND"),
            Some(a.shard_b.to_string()).as_deref()
        );
        assert_eq!(
            env_value(&env, "VD_REALM_SEED"),
            Some(DEV.realm_seed_b.to_string()).as_deref()
        );
        // A DISTINCT mint (never the source's) so DEST-minted entities don't alias the source's.
        assert_eq!(
            env_value(&env, "VD_MINT_SEED"),
            Some(DEV.mint_seed.wrapping_add(6).to_string()).as_deref()
        );
        assert_ne!(
            env_value(&env, "VD_MINT_SEED"),
            Some(DEV.mint_seed.to_string()).as_deref()
        );
        // Books ORCH, GATEWAY, and the SOURCE shard (the cross-shard mesh) — NOT itself, and NOT the
        // Galaxy in Dual (there is none).
        let peers = env_value(&env, "VD_PEERS").expect("VD_PEERS is always emitted");
        assert!(peers.contains(&format!("{}={}", SHARD.0, a.shard)));
        assert!(peers.contains(&format!("{}={}", ORCH.0, a.orchestrator)));
        assert!(peers.contains(&format!("{}={}", GATEWAY.0, a.gateway)));
        assert!(!peers.contains(&format!("{}=", SHARD_B.0)));
        assert!(!peers.contains(&format!("{}={}", GALAXY.0, a.galaxy)));
        // The DEST hosts realm B only — no crossing trigger (the SOURCE hosts it).
        assert_eq!(env_value(&env, "VD_REALM_BOUNDARIES"), None);
    }

    #[test]
    fn shard_b_env_triple_also_books_the_galaxy() {
        // S5b: in Triple the DEST (System 8) additionally books the GALAXY between-space shard — each
        // shard books the others (the sibling crossing routes System 8 → Galaxy → System 7 through it).
        let a = dual_addrs();
        let peers = env_value(&shard_b_env(&a, &DEV, ClusterShape::Triple), "VD_PEERS")
            .expect("VD_PEERS is always emitted")
            .to_owned();
        assert!(peers.contains(&format!("{}={}", GALAXY.0, a.galaxy)));
        assert!(peers.contains(&format!("{}={}", SHARD.0, a.shard)));
    }

    #[test]
    fn galaxy_env_is_the_between_space_realm_with_a_distinct_mint_and_books_both_systems() {
        // S5b: the GALAXY shard hosts System(GALAXY_SEED) (the between-space), with a mint distinct from
        // BOTH systems (so its entities never alias), and books ORCH, GATEWAY, AND both systems — it is the
        // sibling-routing shard. It boots the SEED neighbourhood (no boundary override).
        let a = dual_addrs();
        let env = galaxy_env(&a, &DEV);
        assert_eq!(
            env_value(&env, "VD_NODE_ID"),
            Some(GALAXY.0.to_string()).as_deref()
        );
        assert_eq!(
            env_value(&env, "VD_BIND"),
            Some(a.galaxy.to_string()).as_deref()
        );
        assert_eq!(
            env_value(&env, "VD_REALM_SEED"),
            Some(GALAXY_SEED.to_string()).as_deref()
        );
        // A mint distinct from BOTH the source (11) and the DEST (11+6) mints.
        let galaxy_mint = env_value(&env, "VD_MINT_SEED").expect("VD_MINT_SEED emitted");
        assert_eq!(galaxy_mint, DEV.mint_seed.wrapping_add(12).to_string());
        assert_ne!(galaxy_mint, DEV.mint_seed.to_string());
        assert_ne!(galaxy_mint, DEV.mint_seed.wrapping_add(6).to_string());
        // Books ORCH, GATEWAY, AND both systems (System 7 + System 8) — NOT itself.
        let peers = env_value(&env, "VD_PEERS").expect("VD_PEERS is always emitted");
        assert!(peers.contains(&format!("{}={}", ORCH.0, a.orchestrator)));
        assert!(peers.contains(&format!("{}={}", GATEWAY.0, a.gateway)));
        assert!(peers.contains(&format!("{}={}", SHARD.0, a.shard)));
        assert!(peers.contains(&format!("{}={}", SHARD_B.0, a.shard_b)));
        assert!(!peers.contains(&format!("{}=", GALAXY.0)));
        // The Galaxy boots the seed neighbourhood — no crossing-trigger override.
        assert_eq!(env_value(&env, "VD_REALM_BOUNDARIES"), None);
    }

    // ---- NODE-PER-REALM (Forest) env builders (task #149) --------------------------------------

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
    fn forest_extra_realm_shards_are_the_six_realm_minus_system_7() {
        // The Forest shape's extra shards = Planet 7, Station 7, Area 7, Galaxy, System 8 (System 7 is the
        // always-present base source shard, NOT listed). Non-Forest shapes list NONE.
        use vd_core::pose::RealmId;
        let a = dual_addrs();
        assert!(
            ClusterShape::Single.extra_realm_shards(&a, &DEV).is_empty(),
            "Single has no extra realm-shards"
        );
        assert!(
            ClusterShape::Triple.extra_realm_shards(&a, &DEV).is_empty(),
            "Triple keeps the legacy explicit wiring — no extra_realm_shards"
        );
        let forest = ClusterShape::Forest.extra_realm_shards(&a, &DEV);
        let realms: Vec<RealmId> = forest.iter().map(|s| s.realm).collect();
        assert_eq!(
            realms,
            vec![
                RealmId::Planet(FOREST_CHILD_SEED),
                RealmId::Station(FOREST_CHILD_SEED),
                RealmId::Area(FOREST_CHILD_SEED),
                RealmId::System(GALAXY_SEED),
                RealmId::System(DEV.realm_seed_b),
            ],
        );
        // Each realm-shard has a DISTINCT node id (no aliasing).
        let nodes: std::collections::BTreeSet<u64> = forest.iter().map(|s| s.node.0).collect();
        assert_eq!(
            nodes.len(),
            forest.len(),
            "distinct node ids per realm-shard"
        );
    }

    #[test]
    fn forest_source_shard_env_books_all_five_others_and_never_cohosts() {
        // The Forest System-7 source shard books EVERY other realm-shard (the cross-shard mesh) and, crucially,
        // sets NO `VD_HELD_REALMS` — each realm is its own node, so the source==dest co-hosting is gone.
        let a = dual_addrs();
        let env = shard_env(&a, &DEV, ClusterShape::Forest);
        let peers = env_value(&env, "VD_PEERS").expect("VD_PEERS is always emitted");
        for shard in ClusterShape::Forest.extra_realm_shards(&a, &DEV) {
            assert!(
                peers.contains(&format!("{}={}", shard.node.0, shard.quic)),
                "the source shard books realm-shard node {}",
                shard.node.0,
            );
        }
        // NO co-hosting in Forest (the KEY node-per-realm property).
        assert_eq!(
            env_value(&env, "VD_HELD_REALMS"),
            None,
            "Forest never co-hosts — VD_HELD_REALMS must be absent (each realm on its own node)"
        );
        // The System-7 source shard is still a System realm via the absent-default (no VD_REALM_KIND emitted).
        assert_eq!(env_value(&env, "VD_REALM_KIND"), None);
        assert_eq!(
            env_value(&env, "VD_REALM_SEED"),
            Some(DEV.realm_seed.to_string()).as_deref()
        );
    }

    #[test]
    fn forest_realm_shard_env_hosts_one_realm_with_its_kind_and_books_the_others() {
        // A Planet-7 realm-shard: hosts EXACTLY Planet 7 (VD_REALM_KIND=planet, VD_REALM_SEED=7), no
        // co-hosting, a distinct mint, and books ORCH + GATEWAY + System 7 + every OTHER realm-shard (never
        // itself).
        use vd_core::pose::RealmId;
        let a = dual_addrs();
        let planet = ClusterShape::Forest
            .extra_realm_shards(&a, &DEV)
            .into_iter()
            .find(|s| s.realm == RealmId::Planet(FOREST_CHILD_SEED))
            .expect("the Planet 7 realm-shard is in the Forest set");
        let env = realm_shard_env(&a, &DEV, ClusterShape::Forest, planet);
        assert_eq!(
            env_value(&env, "VD_NODE_ID"),
            Some(planet.node.0.to_string()).as_deref()
        );
        assert_eq!(env_value(&env, "VD_REALM_KIND"), Some("planet"));
        assert_eq!(
            env_value(&env, "VD_REALM_SEED"),
            Some(FOREST_CHILD_SEED.to_string()).as_deref()
        );
        // NO co-hosting, NO boundary override — the seed neighbourhood boots.
        assert_eq!(env_value(&env, "VD_HELD_REALMS"), None);
        assert_eq!(env_value(&env, "VD_REALM_BOUNDARIES"), None);
        // Books ORCH + GATEWAY + System 7 + every OTHER realm-shard, but NOT itself.
        let peers = env_value(&env, "VD_PEERS").expect("VD_PEERS is always emitted");
        assert!(peers.contains(&format!("{}={}", ORCH.0, a.orchestrator)));
        assert!(peers.contains(&format!("{}={}", GATEWAY.0, a.gateway)));
        assert!(peers.contains(&format!("{}={}", SHARD.0, a.shard)));
        assert!(
            !peers.contains(&format!("{}={}", planet.node.0, planet.quic)),
            "a realm-shard does not book ITSELF"
        );
        for other in ClusterShape::Forest
            .extra_realm_shards(&a, &DEV)
            .into_iter()
            .filter(|s| s.node != planet.node)
        {
            assert!(
                peers.contains(&format!("{}={}", other.node.0, other.quic)),
                "the Planet-7 shard books the OTHER realm-shard node {}",
                other.node.0,
            );
        }
        // The Galaxy + System-8 realm-shards keep their HISTORIC distinct mints so their entity ids match the
        // legacy Triple/Dual rigs; the new children derive from the node id — all pairwise distinct.
        let mints: std::collections::BTreeSet<String> = ClusterShape::Forest
            .extra_realm_shards(&a, &DEV)
            .into_iter()
            .map(|s| {
                env_value(
                    &realm_shard_env(&a, &DEV, ClusterShape::Forest, s),
                    "VD_MINT_SEED",
                )
                .expect("mint emitted")
                .to_owned()
            })
            .collect();
        assert_eq!(
            mints.len(),
            5,
            "every realm-shard has a distinct mint (no entity-id aliasing across shards)"
        );
    }

    #[test]
    fn forest_orchestrator_and_gateway_book_and_roster_all_six_realm_shards() {
        // The Forest orchestrator drives the clock + rosters every realm-shard; the gateway books + classes
        // each as known (so the session-route swap reaches every successive source shard as the dot walks).
        let a = dual_addrs();
        let orch = orchestrator_env(&a, &DEV, "store", ClusterShape::Forest);
        let gw = gateway_env(&a, &[], "pub", &DEV, ClusterShape::Forest);
        let orch_peers = env_value(&orch, "VD_PEERS").expect("orch VD_PEERS");
        let gw_peers = env_value(&gw, "VD_PEERS").expect("gw VD_PEERS");
        let roster = env_value(&orch, "VD_ROSTER").expect("VD_ROSTER emitted in Forest");
        let known = env_value(&gw, "VD_KNOWN_SHARDS").expect("VD_KNOWN_SHARDS emitted in Forest");
        for shard in ClusterShape::Forest.extra_realm_shards(&a, &DEV) {
            assert!(
                orch_peers.contains(&format!("{}={}", shard.node.0, shard.quic)),
                "orchestrator books realm-shard {}",
                shard.node.0
            );
            assert!(
                gw_peers.contains(&format!("{}={}", shard.node.0, shard.quic)),
                "gateway books realm-shard {}",
                shard.node.0
            );
            assert!(
                roster.split(',').any(|id| id == shard.node.0.to_string()),
                "VD_ROSTER lists realm-shard {}",
                shard.node.0
            );
            assert!(
                known.split(',').any(|id| id == shard.node.0.to_string()),
                "VD_KNOWN_SHARDS lists realm-shard {}",
                shard.node.0
            );
        }
        // The clock advances to every follower (else a follower never wins its lease).
        let clock = env_value(&orch, "VD_CLOCK_PEERS").expect("VD_CLOCK_PEERS");
        assert!(
            clock
                .split(',')
                .any(|id| id == PLANET_A_SHARD.0.to_string())
        );
        assert!(clock.split(',').any(|id| id == AREA_A_SHARD.0.to_string()));
    }

    #[test]
    fn source_crossing_boundaries_is_a_born_inside_source_realm_shell_to_dest() {
        // The planted shell's exterior realm = the SOURCE realm (so the shard guard accepts it); its
        // `to_realm` is DERIVED from realm_seed_b (M-1: never an inline System(8)).
        let boundaries = source_crossing_boundaries(&DEV);
        assert_eq!(boundaries.len(), 1);
        let b = &boundaries[0];
        assert_eq!(b.realm, RealmId::System(DEV.realm_seed));
        assert_eq!(b.to_realm, RealmId::System(DEV.realm_seed_b));
        assert_eq!(b.effect, CrossEffect::Authority);
        // The guard accepts it for the SOURCE realm, rejects it for a foreign realm (loud config drift).
        assert_eq!(
            guard_boundaries_in_realm(&boundaries, RealmId::System(DEV.realm_seed)),
            Ok(())
        );
        assert_eq!(
            guard_boundaries_in_realm(&boundaries, RealmId::System(DEV.realm_seed_b)),
            Err(RealmBoundariesError::WrongRealm {
                boundary_realm: RealmId::System(DEV.realm_seed),
                hosted_realm: RealmId::System(DEV.realm_seed_b),
            })
        );
    }

    #[test]
    fn write_source_boundaries_roundtrips_the_single_sourced_geometry() {
        // The written file is the client-identical boxes.json: it serialize-roundtrips to the SAME Vec the
        // helper returns AND loads back through the shard's boot loader (fs → serde → in-source-realm guard).
        let dir = TempDir::new("crossing");
        let path = write_source_boundaries(&dir.0, &DEV).expect("write boundaries");
        let loaded = resolve_realm_boundaries(
            &env(&[("VD_REALM_BOUNDARIES", &path)]),
            RealmId::System(DEV.realm_seed),
        )
        .expect("the written file loads in the source realm");
        assert_eq!(loaded, Some(source_crossing_boundaries(&DEV)));
    }

    #[test]
    fn resolve_source_boundaries_prefers_a_valid_override_else_born_inside_default() {
        let dir = TempDir::new("resolve-src");
        // (1) No override → the born-inside default, byte-identical to write_source_boundaries.
        let default_path = resolve_source_boundaries(None, &dir.0, &DEV)
            .expect("no override writes the born-inside default");
        let loaded = resolve_realm_boundaries(
            &env(&[("VD_REALM_BOUNDARIES", &default_path)]),
            RealmId::System(DEV.realm_seed),
        )
        .expect("the default file loads");
        assert_eq!(loaded, Some(source_crossing_boundaries(&DEV)));
        // (2) An EMPTY override is treated as unset → still the born-inside default path.
        let empty_path = resolve_source_boundaries(Some(String::new()), &dir.0, &DEV)
            .expect("empty override falls back to the default");
        assert_eq!(empty_path, default_path);
        // (3) A valid override FILE is returned verbatim (the test/operator's own walk-into geometry).
        let override_file = dir.0.join("override.json");
        std::fs::write(&override_file, "[]").expect("write override");
        let override_path = override_file.display().to_string();
        assert_eq!(
            resolve_source_boundaries(Some(override_path.clone()), &dir.0, &DEV),
            Ok(override_path),
        );
        // (4) An override that is NOT a readable file fails LOUD (never a silent fall-back).
        let missing = dir.0.join("nope.json").display().to_string();
        let err = resolve_source_boundaries(Some(missing), &dir.0, &DEV)
            .expect_err("a bad override path is loud");
        assert!(err.contains("VD_DEVCLUSTER_BOUNDARIES"));
    }
}
