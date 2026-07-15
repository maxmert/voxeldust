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

use std::net::SocketAddr;
use std::path::PathBuf;
use std::process::{Child, Command, ExitStatus};
use std::time::Duration;

use ed25519_dalek::SigningKey;
use vd_core::NodeId;
use vd_devproto::{DevRequest, DevResponse, WORKTREE_SLOT_CEILING};
use vd_io_prod::runtime::EnvConfig;
use vd_io_prod::runtime::hex32_encode;

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
pub fn spawn_probe_server(
    runtime: &tokio::runtime::Handle,
    addr: SocketAddr,
    source: std::sync::Arc<dyn vd_io_prod::probe::HealthSource>,
) {
    runtime.spawn(async move {
        let listener = tokio::net::TcpListener::bind(addr)
            .await
            .unwrap_or_else(|e| panic!("probe server failed to bind {addr}: {e}"));
        axum::serve(listener, vd_io_prod::probe::probe_router(source))
            .await
            .expect("probe server");
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
    move_speed: 2.0,
    tick_dt: 0.02,
    mint_seed: 11,
    input_log_cap: 4096,
    realm_recheck: 0,
    snapshot_budget: 1100,
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

/// The orchestrator's node-specific env (directory + clock + admin + the D-6 durable Store). `store_path`
/// is the redb file location (REQUIRED by the bin — no in-memory fallback). Dev/test clusters store under a
/// $TMPDIR work dir cleaned by `down`, so this also sets `VD_STORE_EPHEMERAL_OK` to clear the bin's HR1
/// persistent-volume guard (a production deploy builds its env elsewhere with a real volume + no opt-in).
#[must_use]
pub fn orchestrator_env(
    a: &ClusterAddrs,
    p: &DevClusterParams,
    store_path: &str,
    dual: bool,
) -> Vec<(&'static str, String)> {
    // Track R / 1d.2: in DUAL mode the orchestrator INITIATES realm-grants / re-home to the DEST, so it
    // books SHARD_B AND drives the universe clock to it (a follower whose clock never advances can never
    // win its realm lease — the harness `clock_peers = {GW, SHARD, DEST}` rule). The DEST is ALSO added to
    // `VD_ROSTER` (the D-37 re-home candidate set; the crossing itself resolves via the directory head, NOT
    // the roster — see the bin note). Absent `dual` these three keys are byte-identical to a single-shard
    // `up` (the crossing INERT).
    let mut peers = vec![(GATEWAY, a.gateway), (SHARD, a.shard)];
    let mut clock_peers = format!("{},{}", GATEWAY.0, SHARD.0);
    if dual {
        peers.push((SHARD_B, a.shard_b));
        clock_peers = format!("{},{},{}", GATEWAY.0, SHARD.0, SHARD_B.0);
    }
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
    ];
    if dual {
        // The D-37 re-home candidate SET: every DEST shard the orchestrator may re-home an orphan onto.
        // `select_rehome_target` is realm-BLIND (lowest live capable node) — a D-37 concern, NOT the
        // crossing (which resolves `head(Realm(to_realm))`, HR3-clean). For the crossing this roster's only
        // job is that the DEST is a known re-home target; the head resolution comes from DEST's realm grant.
        env.push(str_pair("VD_ROSTER", SHARD_B.0));
    }
    env
}

/// The gateway's node-specific env. `clients` are seeded into the gateway's peer
/// book so it can route snapshots BACK to each dev-control client (the mesh dials
/// by address book; a missing client entry = the gateway can never reach it).
#[must_use]
pub fn gateway_env(
    a: &ClusterAddrs,
    clients: &[(NodeId, SocketAddr)],
    auth_pubkey_hex: &str,
    p: &DevClusterParams,
    dual: bool,
) -> Vec<(&'static str, String)> {
    // Track R / 1d.2: in DUAL mode the gateway must BOOK the DEST (to route a transferred client's
    // inputs / cut-drains onto it) AND class the DEST as a KNOWN shard (`VD_KNOWN_SHARDS`, consumed by
    // the gateway bin's `known_shards` set) so a DEST→gateway frame reaches `on_shard_frame` instead of
    // dropping as an unknown peer. `VD_SHARD` (the LOGIN shard) stays SHARD in both modes. Absent `dual`
    // neither the DEST peer nor `VD_KNOWN_SHARDS` is emitted — byte-identical to a single-shard `up`.
    let mut peers = vec![(ORCH, a.orchestrator), (SHARD, a.shard)];
    if dual {
        peers.push((SHARD_B, a.shard_b));
    }
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
    if dual {
        // The gateway bin unions this into `known_shards` over the login `shard` (via `node_list`), so the
        // routable-shard roster is {SHARD, DEST} — every shard's frames are node-class dispatchable.
        env.push(str_pair("VD_KNOWN_SHARDS", SHARD_B.0));
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
    dual: bool,
) -> Vec<(&'static str, String)> {
    let mut peers = vec![(ORCH, a.orchestrator), (GATEWAY, a.gateway)];
    if dual {
        peers.push((SHARD_B, a.shard_b));
    }
    vec![
        str_pair("VD_NODE_ID", SHARD.0),
        str_pair("VD_BIND", a.shard),
        ("VD_PEERS", book(&peers)),
        str_pair("VD_REALM_SEED", p.realm_seed),
        str_pair("VD_SPEED", p.move_speed),
        str_pair("VD_TICK_DT", p.tick_dt),
        str_pair("VD_ORCH", ORCH.0),
        str_pair("VD_MINT_SEED", p.mint_seed),
        str_pair("VD_INPUT_LOG_CAP", p.input_log_cap),
        str_pair("VD_REALM_RECHECK", p.realm_recheck),
        str_pair("VD_SNAPSHOT_BUDGET", p.snapshot_budget),
        str_pair("VD_PROBE_ADDR", a.shard_probe),
    ]
}

/// Track R / 1d.2 — the DEST stub-shard's node-specific env (twin of [`shard_env`]). Hosts realm B
/// (`System(realm_seed_b)`) at [`SHARD_B`], with a DISTINCT mint seed (so its dots are genuinely separate,
/// mirroring the harness `dest_stub_config` mint 17 vs the source's 11) and its OWN QUIC bind + probe. Its
/// `VD_PEERS` books ORCH, GATEWAY, AND the SOURCE shard (each shard books the other — the cross-shard mesh).
/// NO `VD_REALM_BOUNDARIES`: the SOURCE hosts the crossing trigger INTO realm B; the DEST just receives.
/// Only ever spawned by a dual `up`, so it has no `dual` arm.
#[must_use]
pub fn shard_b_env(a: &ClusterAddrs, p: &DevClusterParams) -> Vec<(&'static str, String)> {
    vec![
        str_pair("VD_NODE_ID", SHARD_B.0),
        str_pair("VD_BIND", a.shard_b),
        (
            "VD_PEERS",
            book(&[
                (ORCH, a.orchestrator),
                (GATEWAY, a.gateway),
                (SHARD, a.shard),
            ]),
        ),
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
/// # Errors
/// Propagates the OS spawn error (e.g. the binary is missing).
pub fn spawn_node(
    bin: &str,
    common: &[(&'static str, String)],
    node_env: &[(&'static str, String)],
) -> std::io::Result<Child> {
    let mut cmd = Command::new(bin);
    for (k, v) in common.iter().chain(node_env.iter()) {
        cmd.env(k, v);
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

// ---- the shard boundary-plant knob (DEFERRED 1-SIGKILL-OWED) -----------------

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

/// Resolve the shard's boot-loaded realm boundaries (DEFERRED 1-SIGKILL-OWED — the process-seam
/// boundary-plant knob). Reads `VD_REALM_BOUNDARIES` (a path to a `boundaries.json` = a
/// `Vec<RealmBoundary>`, the IDENTICAL format the client loads as `--realm-boxes` — SINGLE-SOURCED),
/// parses it, and validates every boundary's exterior `realm` is the realm THIS shard hosts
/// (`hosted_realm`, a config-drift guard). Returns:
/// - `Ok(None)` when the env var is ABSENT — INERT: the caller leaves the shard's `RealmBoundaries`
///   at its `default()` empty, so `evaluate_realm_boundaries` early-returns exactly as today (every
///   existing run stays byte-identical).
/// - `Ok(Some(vec))` when the file loads and every boundary is in-realm — the caller plants it.
///
/// Fails LOUD ([`RealmBoundariesError`]) on a malformed file or a boundary for a realm this shard
/// does not host — a misconfiguration must never become a silent inert trigger.
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

/// Parse a `boundaries.json` string into a `Vec<RealmBoundary>` (the SAME serde shape the client's
/// `RealmScene::from_boxes_json` loads — single-sourced). A monomorphic helper so the parse-error arm
/// is covered off the env/fs body.
fn parse_realm_boundaries(
    json: &str,
) -> Result<Vec<vd_core::geometry::RealmBoundary>, RealmBoundariesError> {
    serde_json::from_str(json).map_err(|e| RealmBoundariesError::Malformed(e.to_string()))
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

    // ---- Track R / 1d.2: the dual-shard env builders + boundaries helper --------------------------

    /// Loopback addrs distinct per role so a mis-booked peer is visible in an assert. Every field is a
    /// unique port, so a `book(...)` mismatch shows up as a wrong port string.
    fn dual_addrs() -> ClusterAddrs {
        ClusterAddrs {
            orchestrator: loopback(9001),
            gateway: loopback(9002),
            shard: loopback(9003),
            admin: loopback(9004),
            orchestrator_probe: loopback(9005),
            gateway_probe: loopback(9006),
            shard_probe: loopback(9007),
            shard_b: loopback(9008),
            shard_b_probe: loopback(9009),
        }
    }

    /// Look up a key's value in a rendered env vec (None if absent).
    fn env_value<'a>(env: &'a [(&'static str, String)], key: &str) -> Option<&'a str> {
        env.iter().find(|(k, _)| *k == key).map(|(_, v)| v.as_str())
    }

    #[test]
    fn orchestrator_env_dual_books_dest_clock_and_roster_inert_when_single() {
        let a = dual_addrs();
        let single = orchestrator_env(&a, &DEV, "store", false);
        let dual = orchestrator_env(&a, &DEV, "store", true);

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
    fn gateway_env_dual_books_dest_and_emits_known_shards_inert_when_single() {
        let a = dual_addrs();
        let single = gateway_env(&a, &[], "pub", &DEV, false);
        let dual = gateway_env(&a, &[], "pub", &DEV, true);

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
    fn shard_env_dual_books_the_other_shard_inert_when_single() {
        let a = dual_addrs();
        let single = shard_env(&a, &DEV, false);
        let dual = shard_env(&a, &DEV, true);
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
        // H-1 inert-parity: the `dual=false` arm of every builder emits EXACTLY the pre-Track-R env, so a
        // single-shard `up` / the process_parity gate stays byte-identical. (Asserted field-by-field
        // against the hand-written expected today's env — a regression flips this loud.)
        let a = dual_addrs();

        let orch = orchestrator_env(&a, &DEV, "store", false);
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
            ]
        );

        let gw = gateway_env(&a, &[], "pub", &DEV, false);
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
            ]
        );

        let shard = shard_env(&a, &DEV, false);
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
        let env = shard_b_env(&a, &DEV);
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
        // Books ORCH, GATEWAY, and the SOURCE shard (the cross-shard mesh) — NOT itself.
        let peers = env_value(&env, "VD_PEERS").expect("VD_PEERS is always emitted");
        assert!(peers.contains(&format!("{}={}", SHARD.0, a.shard)));
        assert!(peers.contains(&format!("{}={}", ORCH.0, a.orchestrator)));
        assert!(peers.contains(&format!("{}={}", GATEWAY.0, a.gateway)));
        assert!(!peers.contains(&format!("{}=", SHARD_B.0)));
        // The DEST hosts realm B only — no crossing trigger (the SOURCE hosts it).
        assert_eq!(env_value(&env, "VD_REALM_BOUNDARIES"), None);
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
}
