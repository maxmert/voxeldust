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
        // Relaxed is sufficient: the tick loop polls this single flag with no other ordering dependency.
        set.store(true, Ordering::Relaxed);
    });
    flag
}

// ---- node roster -------------------------------------------------------------

/// The fixed P1 node identities. ONE definition for the launcher + parity test.
pub const ORCH: NodeId = NodeId(1);
pub const GATEWAY: NodeId = NodeId(2);
pub const SHARD: NodeId = NodeId(3);

// ---- dev auth identity -------------------------------------------------------

/// DEV-ONLY Ed25519 login seed for the localhost cluster. The gateway boots with
/// its verifying key; clients mint logins with the seed. NEVER a production
/// secret — a fixed dev constant so the identity is reproducible. (Structurally
/// fencing it behind a dev/test compile guard is tracked for the prod-build phase.)
pub const DEV_AUTH_SEED: [u8; 32] = [0x42; 32];

/// The hex verifying key the gateway boots with (`VD_AUTH_PUBKEY`).
#[must_use]
pub fn dev_auth_pubkey_hex() -> String {
    hex32_encode(
        &SigningKey::from_bytes(&DEV_AUTH_SEED)
            .verifying_key()
            .to_bytes(),
    )
}

/// The hex signing seed clients mint logins with (`VD_AUTH_SIGNING_KEY` in the
/// cluster contract). Dev-only — see [`DEV_AUTH_SEED`].
#[must_use]
pub fn dev_auth_signing_key_hex() -> String {
    hex32_encode(&DEV_AUTH_SEED)
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
    move_speed: 2.0,
    tick_dt: 0.02,
    mint_seed: 11,
    input_log_cap: 4096,
    realm_recheck: 0,
    snapshot_budget: 1100,
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

// ---- the env contract --------------------------------------------------------

/// The bind addresses of the three fixed cluster nodes (admin is the orchestrator's
/// read-only HTTP endpoint).
#[derive(Clone, Copy, Debug)]
pub struct ClusterAddrs {
    pub orchestrator: SocketAddr,
    pub gateway: SocketAddr,
    pub shard: SocketAddr,
    pub admin: SocketAddr,
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
    match env.string(key) {
        Err(_) => Ok(false),
        Ok(v) => match v.trim().to_ascii_lowercase().as_str() {
            "1" | "true" | "yes" => Ok(true),
            "0" | "false" | "no" | "" => Ok(false),
            other => Err(format!(
                "{key}={other:?} is not a boolean (use 1/true/yes or 0/false/no)"
            )
            .into()),
        },
    }
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
) -> Vec<(&'static str, String)> {
    vec![
        str_pair("VD_NODE_ID", ORCH.0),
        str_pair("VD_BIND", a.orchestrator),
        ("VD_PEERS", book(&[(GATEWAY, a.gateway), (SHARD, a.shard)])),
        str_pair("VD_EPOCH", p.epoch),
        str_pair("VD_RESERVE_CHUNK", p.reserve_chunk),
        ("VD_CLOCK_PEERS", format!("{},{}", GATEWAY.0, SHARD.0)),
        str_pair("VD_LEASE_TTL", p.lease_ttl),
        str_pair("VD_ADMIN_ADDR", a.admin),
        ("VD_STORE_PATH", store_path.to_owned()),
        ("VD_STORE_EPHEMERAL_OK", "1".to_owned()),
    ]
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
) -> Vec<(&'static str, String)> {
    let mut peers = vec![(ORCH, a.orchestrator), (SHARD, a.shard)];
    peers.extend_from_slice(clients);
    vec![
        str_pair("VD_NODE_ID", GATEWAY.0),
        str_pair("VD_BIND", a.gateway),
        ("VD_PEERS", book(&peers)),
        str_pair("VD_ORCH", ORCH.0),
        str_pair("VD_SHARD", SHARD.0),
        ("VD_AUTH_PUBKEY", auth_pubkey_hex.to_owned()),
        str_pair("VD_SESSION_SEED", p.session_seed),
        str_pair("VD_MAX_SESSIONS", p.max_sessions),
        str_pair("VD_MAX_BUFFERED_INPUTS", p.max_buffered_inputs),
    ]
}

/// The stub-shard's node-specific env (realm + movement + snapshot budget).
#[must_use]
pub fn shard_env(a: &ClusterAddrs, p: &DevClusterParams) -> Vec<(&'static str, String)> {
    vec![
        str_pair("VD_NODE_ID", SHARD.0),
        str_pair("VD_BIND", a.shard),
        (
            "VD_PEERS",
            book(&[(ORCH, a.orchestrator), (GATEWAY, a.gateway)]),
        ),
        str_pair("VD_REALM_SEED", p.realm_seed),
        str_pair("VD_SPEED", p.move_speed),
        str_pair("VD_TICK_DT", p.tick_dt),
        str_pair("VD_ORCH", ORCH.0),
        str_pair("VD_MINT_SEED", p.mint_seed),
        str_pair("VD_INPUT_LOG_CAP", p.input_log_cap),
        str_pair("VD_REALM_RECHECK", p.realm_recheck),
        str_pair("VD_SNAPSHOT_BUDGET", p.snapshot_budget),
    ]
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
}
