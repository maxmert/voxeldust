//! `vd-bins` library — the SINGLE SOURCE OF TRUTH for standing up a local cluster
//! of the node binaries, shared by the `vd-devcluster` launcher AND the
//! `process_parity` real-binary test. Without this shared module the two would
//! hand-duplicate the 23-key `VD_*` env contract, the 14 operational params, the
//! node roster, the dev auth identity, the peer-book formatter, and the child
//! teardown — and silently DRIFT (the parity gate would stay green while the
//! agent's cluster shipped half-configured). Everything that defines "what a dev
//! cluster IS" lives here, once.
//!
//! This is Tier-B (process glue): exercised by the process-tier parity + smoke
//! tests, not the coverage gate.

use std::net::SocketAddr;
use std::process::{Child, Command, ExitStatus};
use std::time::Duration;

use ed25519_dalek::SigningKey;
use vd_core::NodeId;
use vd_io_prod::runtime::hex32_encode;

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

/// The env every node shares (trust bundle + transport knobs).
#[must_use]
pub fn common_env(trust_dir: &str, p: &DevClusterParams) -> Vec<(&'static str, String)> {
    vec![
        ("VD_TRUST_DIR", trust_dir.to_owned()),
        str_pair("VD_OUTBOUND_CAP", p.outbound_cap),
        str_pair("VD_TICK_HZ", p.tick_hz),
    ]
}

/// The orchestrator's node-specific env (directory + clock + admin).
#[must_use]
pub fn orchestrator_env(a: &ClusterAddrs, p: &DevClusterParams) -> Vec<(&'static str, String)> {
    vec![
        str_pair("VD_NODE_ID", ORCH.0),
        str_pair("VD_BIND", a.orchestrator),
        ("VD_PEERS", book(&[(GATEWAY, a.gateway), (SHARD, a.shard)])),
        str_pair("VD_EPOCH", p.epoch),
        str_pair("VD_RESERVE_CHUNK", p.reserve_chunk),
        ("VD_CLOCK_PEERS", format!("{},{}", GATEWAY.0, SHARD.0)),
        str_pair("VD_LEASE_TTL", p.lease_ttl),
        str_pair("VD_ADMIN_ADDR", a.admin),
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
