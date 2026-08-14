//! `vd-devcluster` — the local-process dev-cluster launcher (HR6; replaces k3d for
//! the P1.5+ agent loop). It brings orchestrator + gateway + stub-shard up as real
//! binaries over localhost QUIC under a fresh mTLS trust bundle and the dev auth
//! identity, on the slot's [`vd_devproto::DevPortScheme`] ports, then waits until
//! the cluster has bootstrapped (a shard granted its realm to the directory —
//! proving the QUIC mesh works) with all node processes still alive.
//!
//! ```text
//! vd-devcluster up   --slot 0     # spawn + wait-ready + write runfile, then return
//! vd-devcluster status --slot 0   # are the children alive and the admin ready?
//! vd-devcluster down --slot 0     # kill the cluster and clean the runfile/trust
//! ```
//! `up` writes `<tmp>/vd-devcluster/slot-N/cluster.env` — the eval-able cluster
//! contract (gateway addr, admin addr, dev signing key, trust dir) that
//! `scripts/client.sh` sources.
//!
//! Robust process lifecycle (audited): children spawn into their own process
//! groups under a RAII guard, the runfile (kill record) is written BEFORE the
//! ready-wait, and `down` SIGTERMs then SIGKILL-escalates the recorded groups and
//! removes the runfile only once every process is confirmed dead. The env contract,
//! operational params, node roster, and dev identity are the SHARED `vd_bins`
//! source of truth — never re-typed here (so the parity test can't drift from it).
//! Tier-B process glue; gated by the Slice-0 smoke test + G-RENDER-SMOKE.

use std::fs::File;
use std::io::Write;
use std::net::SocketAddr;
use std::path::{Path, PathBuf};
use std::process::{Child, ExitCode};
use std::time::{Duration, Instant};

use vd_bins::{
    Cluster, ClusterAddrs, ClusterShape, DEV, GALAXY, ORCH_STORE_NAME, PLANET_SHARD, RUNFILE_NAME,
    TRUST_DIR_NAME, admin_get_body, common_env, dev_auth_pubkey_hex, gateway_env, loopback,
    orchestrator_env, realm_shards, roster_realms, sh_quote, shard_env, slot_workdir,
};
use vd_core::NodeId;
use vd_devproto::{CLIENT_NODE_BASE, DevPortScheme, SlotPorts};
use vd_io_prod::trust::ClusterTrust;
use vd_wire::admin::AdminSnapshot;

const READY_TIMEOUT: Duration = Duration::from_secs(20);
const POLL_INTERVAL: Duration = Duration::from_millis(200);
/// Grace between SIGTERM and the SIGKILL escalation in `down`.
const TERM_GRACE: Duration = Duration::from_secs(3);
/// Bounds the admin readiness GET so a wedged endpoint can't stall the poll.
const ADMIN_READ_TIMEOUT: Duration = Duration::from_secs(2);

/// One entry in the spawn list: (executable, log/kill-record label, env vars).
/// HR3: the DEST runs the SAME `vd-shard` binary as the SOURCE — the label only
/// names its log + kill-record entry, never a distinct binary.
type NodeSpec = (&'static str, &'static str, Vec<(&'static str, String)>);

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    match run(&args) {
        Ok(()) => ExitCode::SUCCESS,
        Err(msg) => {
            eprintln!("vd-devcluster: {msg}");
            eprintln!(
                "usage: vd-devcluster <up|down|status|env> --slot <N>  |  gen-trust <dir>  |  \
                 gen-authkey  |  emit-world-scene <dir>"
            );
            ExitCode::FAILURE
        }
    }
}

fn run(args: &[String]) -> Result<(), String> {
    let cmd = args.first().ok_or("missing subcommand")?.as_str();
    // `gen-trust <dir>` takes a DIRECTORY, not a slot — dispatch it before the slot-based
    // subcommands. It is the cluster-secret distribution used by BOTH the docker-run smoke
    // (mount the dir at VD_TRUST_DIR) and the k8s Secret (Slice 4: `kubectl create secret`
    // from the three DER files). ONE bundle serves every node/pod — the node cert SANs are
    // cluster_name + localhost and the mesh dials server-name "localhost" (mesh.rs), so pod
    // IP / Service DNS never has to appear in a cert.
    if cmd == "gen-trust" {
        let dir = args.get(1).ok_or("gen-trust needs a <dir>")?;
        return gen_trust(Path::new(dir));
    }
    // `gen-authkey` mints a fresh PRODUCTION session-auth keypair (the cloud profile VETOES the built-in dev
    // key). Prints VD_AUTH_PUBKEY (→ the gateway env / `vd-auth` Secret) + VD_AUTH_SIGNING_KEY (→ clients
    // OUT-OF-BAND, never a server Secret). No slot; dispatch before the slot-based subcommands.
    if cmd == "gen-authkey" {
        return gen_authkey();
    }
    // `emit-world-scene <dir>` writes THE world's home-shard neighbourhood (`regions.json`) for the
    // client's `--realm-boxes` — the ONE scene emitter (SL5): the drawn geometry IS the detector's,
    // single-sourced through `world_roster`/`boot_regions_and_movers` (which also runs the flight-law
    // asserts, so a world change fails the emit loudly). No options, no selector; dir arg only.
    if cmd == "emit-world-scene" {
        let dir = args.get(1).ok_or("emit-world-scene needs a <dir>")?;
        let scene = vd_bins::write_world_regions(Path::new(dir), &vd_bins::DEV)?;
        println!("VD_WORLD_SCENE={scene}");
        return Ok(());
    }
    // (The `emit-crossing-fixtures` / `emit-seed-fixtures` / `emit-visual-fixtures` subcommands are
    // DELETED — SL5: each emitted a scene of a world that is not THE world. `emit-world-scene` above
    // is the ONE emitter, with nothing to choose.)
    let slot = parse_slot(args)?;
    let ports = DevPortScheme::DEFAULT
        .slot_ports(slot)
        .map_err(|e| e.to_string())?;
    let work = work_dir(slot);
    match cmd {
        // The cluster shape (how many stub shards spawn) is a DATA value the shared env builders fan out
        // on (HR3 — never a shard-kind branch in a feature). Every static shape pre-books realms OF THE
        // WORLD via `roster_realms`/`realm_shards`:
        //   `up`            → Single: orchestrator + gateway + the home shard (3 procs).
        //   `up --dual`     → Dual: + the GALAXY shard (the home region's parent) — the one-hop
        //                     autonomous crossing over THE world's own home shell (4 procs).
        //   `up --chain`    → Chain: + GALAXY + the inner-planet shard + the sibling-star shard —
        //                     down, up, out, sideways-through-parent, each realm its OWN node (6 procs).
        //   `up --demand`   → Demand: orchestrator + gateway ONLY; worlds spin up on login/AoI demand.
        "up" => up(slot, ports, &work, cluster_shape(args)?),
        "down" => down(&work),
        "status" => status(ports, &work),
        "env" => env_cmd(&work),
        other => Err(format!("unknown subcommand `{other}`")),
    }
}

/// True iff `flag` appears anywhere in `args` (a bare boolean flag, e.g. `--dual`).
fn has_flag(args: &[String], flag: &str) -> bool {
    args.iter().any(|a| a == flag)
}

/// The cluster shape from the `up` flags: `--chain` > `--dual` > `--demand` > Single.
///
/// `--triple` and `--forest` ERROR LOUDLY: both named realms THE world does not contain (System 8,
/// Planet/Station/Area 7), so their extra shards died at boot and the shapes rotted invisibly —
/// helped by the old parser, where an unrecognised flag fell silently to Single. Refusing beats
/// silently standing up the wrong cluster.
fn cluster_shape(args: &[String]) -> Result<ClusterShape, String> {
    if has_flag(args, "--triple") {
        return Err(
            "`--triple` is retired: it named realms THE world does not contain (System 8 + the \
             co-hosted System-7 children), so its shards could never boot. The sibling crossing is \
             home → galaxy → sibling: use `up --chain`."
                .to_owned(),
        );
    }
    if has_flag(args, "--forest") {
        return Err(
            "`--forest` is retired: it named realms THE world does not contain (Planet/Station/Area 7, \
             System 8), so four of its six shards could never boot. Use `up --chain` (home + galaxy + \
             inner planet + sibling star, node-per-realm)."
                .to_owned(),
        );
    }
    Ok(if has_flag(args, "--chain") {
        ClusterShape::Chain
    } else if has_flag(args, "--dual") {
        ClusterShape::Dual
    } else if has_flag(args, "--demand") {
        // RLM demand-walk (VU): orchestrator + gateway ONLY, NO shard pre-booked — the ONLY way to a world is
        // the armed reconciler spinning one up on login/AoI demand. `up --demand` stands up the exact cluster
        // the rlm_demand_login/demand-walk proofs build in-harness, so a windowed client can fly it.
        ClusterShape::Demand
    } else {
        ClusterShape::Single
    })
}

/// Generate a fresh mTLS `ClusterTrust` bundle (`ca.der`/`node.der`/`key.der`) into `dir` — the
/// same generation `up` does inline, exposed standalone so the k3d smoke + the Slice-4 Secret can
/// mint the shared cluster secret WITHOUT spawning a cluster. Overwrites the dir's DER files.
fn gen_trust(dir: &Path) -> Result<(), String> {
    std::fs::create_dir_all(dir).map_err(|e| format!("create trust dir: {e}"))?;
    let trust = ClusterTrust::generate("voxeldust").map_err(|e| format!("trust: {e}"))?;
    trust
        .write_der_dir(dir)
        .map_err(|e| format!("write trust: {e}"))?;
    println!("wrote ca.der/node.der/key.der to {}", dir.display());
    Ok(())
}

/// Mint a fresh PRODUCTION Ed25519 session-auth keypair (the cloud profile vetoes the built-in dev key,
/// `boot::CloudProfileError::DevAuthKey`). Reads the 32-byte seed from the OS CSPRNG (`/dev/urandom`; no new
/// dependency — this is a Unix dev tool), derives the pair via [`vd_bins::auth_keypair_hex_from_seed`], and
/// prints (stdout, machine-readable) `VD_AUTH_PUBKEY=<hex>` (→ the gateway env / `vd-auth` Secret) +
/// `VD_AUTH_SIGNING_KEY=<hex>` (→ clients OUT-OF-BAND — NEVER a server Secret).
fn gen_authkey() -> Result<(), String> {
    use std::io::Read;
    let mut seed = [0u8; 32];
    std::fs::File::open("/dev/urandom")
        .and_then(|mut f| f.read_exact(&mut seed))
        .map_err(|e| format!("read OS CSPRNG (/dev/urandom): {e}"))?;
    let (pubkey_hex, signing_hex) = vd_bins::auth_keypair_hex_from_seed(&seed);
    println!("VD_AUTH_PUBKEY={pubkey_hex}");
    println!("VD_AUTH_SIGNING_KEY={signing_hex}");
    eprintln!(
        "gen-authkey: put VD_AUTH_PUBKEY in the gateway env / `vd-auth` Secret; deliver VD_AUTH_SIGNING_KEY \
         to clients OUT-OF-BAND (never a server Secret)."
    );
    Ok(())
}

/// Print the eval-able cluster contract for `client.sh` (keeps the temp-dir path
/// logic in ONE place — bash never reconstructs it).
fn env_cmd(work: &Path) -> Result<(), String> {
    let text = std::fs::read_to_string(env_file(work)).map_err(|e| {
        format!(
            "no cluster env ({}): {e}; is the cluster up?",
            env_file(work).display()
        )
    })?;
    print!("{text}");
    Ok(())
}

/// Bring the cluster up. The runfile is created O_EXCL FIRST — it is BOTH the
/// atomic slot claim (a second concurrent `up` loses the create) AND the kill
/// record: each child's pid is appended the instant it spawns. So a
/// SIGKILL/Ctrl-C/OOM at ANY point leaves a runfile a later `down` fully reaps and
/// clears — never a wedged slot (one artifact, not a lock-vs-runfile split) nor an
/// unrecorded port-holder (the pid lands before the child binds its port).
/// `Cluster::Drop` additionally reaps GRACEFUL in-process failures (an early `?` /
/// panic), but does NOT run on SIGKILL — the durable runfile is what covers that.
fn up(slot: u16, ports: SlotPorts, work: &Path, shape: ClusterShape) -> Result<(), String> {
    std::fs::create_dir_all(work).map_err(|e| format!("create work dir: {e}"))?;
    // The runfile IS the claim: O_EXCL create fails if another launcher holds it.
    let mut runfile = match std::fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(runfile(work))
    {
        Ok(file) => file,
        Err(e) if e.kind() == std::io::ErrorKind::AlreadyExists => {
            return Err(format!(
                "slot is already claimed ({}); run `down` first",
                work.display()
            ));
        }
        Err(e) => return Err(format!("claim slot: {e}")),
    };

    let result = up_inner(slot, ports, work, &mut runfile, shape);
    if result.is_err() {
        // Graceful failure: Cluster::Drop already killed the children; drop the
        // workdir (and the runfile/claim with it) so the slot is immediately reusable.
        let _ = std::fs::remove_dir_all(work);
    }
    result
}

fn up_inner(
    slot: u16,
    ports: SlotPorts,
    work: &Path,
    runfile: &mut std::fs::File,
    shape: ClusterShape,
) -> Result<(), String> {
    // mTLS trust bundle (shared by every node) + the dev auth identity.
    let trust_dir = work.join(TRUST_DIR_NAME);
    let trust = ClusterTrust::generate("vd-devcluster").map_err(|e| format!("trust: {e}"))?;
    trust
        .write_der_dir(&trust_dir)
        .map_err(|e| format!("write trust: {e}"))?;
    let trust_str = trust_dir.display().to_string();
    let auth_pubkey = dev_auth_pubkey_hex();

    // ONE slot-port → address mapping (`ClusterAddrs::for_slot`), shared with the process smokes that
    // read the same slot. Extra realm-shard addrs are bound only when the shape spawns them; the
    // station/area slots stay RESERVED-unbound.
    let addrs = ClusterAddrs::for_slot(ports);
    // CA-1 CRUTCH (fenced; deferred to M3): seed every dev-control client's QUIC
    // addr into the gateway book so the gateway can route snapshots back (the mesh
    // dials by static address book — `unknown_destinations_are_loud_backpressure`).
    // The cloud-correct design is reply-on-connection: the gateway learns a client's
    // return address from its INBOUND QUIC connection, so clients need no fixed addr
    // and no pre-seeding. Tracked by the ignored red guard
    // `ca1_reply_on_connection_reaches_a_peer_not_in_the_book`. Remove this seeding
    // when CA-1 lands.
    let clients = client_book(ports)?;
    let common = common_env(&trust_str, &DEV);
    // D-6: the orchestrator's durable Store lives IN the slot work dir (so `down` reaps it with the slot).
    // Under $TMPDIR ⇒ ephemeral; `orchestrator_env` sets VD_STORE_EPHEMERAL_OK so the bin's HR1 guard
    // allows it. A restart RECOVERS (clock resumes forward, in-flight sagas re-drive); `down` wipes it.
    let store_str = work.join(ORCH_STORE_NAME).display().to_string();

    // Spawn into a RAII guard (graceful-failure reaper); record each pid into the
    // runfile (SIGKILL reaper) the INSTANT its child exists, before the next spawn.
    // The base 3 nodes are shape-aware (their env builders book every extra shard + emit
    // VD_KNOWN_SHARDS/VD_ROSTER that grow with the shape). Extra shards are appended LAST (so the other
    // nodes' books tolerate a not-yet-bound peer with retry — process_parity's NodeUnreachable-then-retry
    // pattern). Each entry is (executable, log/kill-record label, env). HR3: EVERY shard runs the SAME
    // `vd-shard` binary — a distinct label (`vd-shard-b`/`vd-galaxy`) only names its log + kill-record entry.
    let mut nodes: Vec<NodeSpec> = vec![
        (
            "vd-orchestrator",
            "vd-orchestrator",
            orchestrator_env(&addrs, &DEV, &store_str, shape),
        ),
        (
            "vd-gateway",
            "vd-gateway",
            gateway_env(&addrs, &clients, &auth_pubkey, &DEV, shape),
        ),
    ];
    // The static login shard: present for every STATIC shape, ABSENT for Demand — the home shard is spawned on
    // the fly by the armed reconciler at login (that IS the demand cluster). The realm-shard fan-out below is
    // empty for Demand too (byte-identical to a Single roster minus the one static shard).
    if !shape.is_demand() {
        nodes.push(("vd-shard", "vd-shard", shard_env(&addrs, &DEV, shape)));
    }
    // NODE-PER-REALM: every extra realm-shard of the shape (galaxy / inner planet / sibling star) on
    // its OWN node so every re-home is a uniform CROSS-NODE saga (no `VD_HELD_REALMS` co-hosting). The
    // SAME `vd-shard` binary (HR3); each carries its `VD_REALM_KIND` + `VD_REALM_SEED` so the shard
    // boots the right realm KIND, and each boots THE world's SEED neighbourhood — NO boundary
    // injection in ANY shape: the world's own shells ARE the crossing boundaries. Appended LAST so the
    // earlier books tolerate the not-yet-bound peers with retry. Empty for Single/Demand
    // (byte-identical). The label (log + kill-record naming only, never behaviour) keys on the fixed
    // node slot.
    for shard in realm_shards(shape, &addrs, &DEV) {
        let label: &'static str = match shard.node {
            n if n == GALAXY => "vd-galaxy",
            n if n == PLANET_SHARD => "vd-planet",
            _ => "vd-shard-b",
        };
        nodes.push((
            "vd-shard",
            label,
            vd_bins::realm_shard_env(&addrs, &DEV, shape, shard),
        ));
    }
    let mut cluster = Cluster::new();
    for (bin, label, node_env) in nodes {
        let child = spawn_node(bin, label, work, &common, node_env)?;
        let pid = child.id();
        // Guard the child BEFORE the fallible record: if `record_pid` errors (a disk
        // fault mid-bring-up), `cluster` must already own this child so its Drop kills
        // it — otherwise an unrecorded, unkilled port-holder leaks when `up` then
        // removes the workdir. `push` is infallible and `record_pid` still runs before
        // the next spawn, so the pid is durable before the next child binds its port.
        cluster.push(label, child);
        record_pid(runfile, pid)?;
    }

    if let Err(msg) = await_ready(&mut cluster, ports.admin, shape) {
        for name in cluster.names() {
            dump_log_tail(work, name);
        }
        // `cluster` drops here on return → kills the children; the caller removes
        // the workdir (and the runfile/claim with it).
        return Err(msg);
    }

    write_env_contract(
        work,
        slot,
        addrs.gateway,
        addrs.admin,
        addrs.orchestrator_probe,
        addrs.gateway_probe,
        addrs.shard_probe,
        &trust_str,
    )?;
    let _ = cluster.into_pids(); // disarm: children outlive us; runfile is the record
    println!("dev-cluster slot {slot} UP");
    println!("  gateway   {}", addrs.gateway);
    println!("  admin     http://{}/admin/snapshot", addrs.admin);
    println!("  env       vd-devcluster env --slot {slot}");
    Ok(())
}

/// Wait until the cluster has bootstrapped with ALL node processes still alive — failing LOUD if a child
/// exits or the deadline hits. ONE readiness rule for every shape: every realm the shape pre-books
/// ([`roster_realms`]) rests with a shard in the ONE directory ([`AdminSnapshot::realms_present`]) —
/// a crossing whose dest head resolves nothing would only COUNT `crossing_unresolved` and never fire,
/// and (worse) an occupant steered at it strands permanently.
fn await_ready(cluster: &mut Cluster, admin_port: u16, shape: ClusterShape) -> Result<(), String> {
    let deadline = Instant::now() + READY_TIMEOUT;
    loop {
        if let Some((name, status)) = cluster.first_exited() {
            return Err(format!("{name} exited during bring-up ({status})"));
        }
        if admin_ready(admin_port, shape) {
            return Ok(()); // all node processes confirmed alive THIS iteration, and ready
        }
        if Instant::now() >= deadline {
            return Err("cluster did not become ready within the deadline".to_owned());
        }
        std::thread::sleep(POLL_INTERVAL);
    }
}

/// A blocking HTTP/1.1 GET of `/admin/snapshot`, parsed and judged by a Tier-A predicate (no brittle
/// substring match): ONE rule for every shape — [`AdminSnapshot::realms_present`] over
/// [`roster_realms`], the realms this shape pre-books (derived from THE world, never an inline
/// `System(…)` — the old Single `cluster_bootstrapped` special case is folded in, since Single's
/// roster is exactly its home realm). A DEMAND cluster pre-books NOTHING, so its roster is EMPTY and
/// `realms_present` is vacuously true the moment the snapshot parses — readiness is "the orchestrator
/// is UP and serving"; a client login spawns the first world (proven by rlm_demand_login).
fn admin_ready(admin_port: u16, shape: ClusterShape) -> bool {
    let Some(body) = admin_get_body(
        loopback(admin_port),
        "/admin/snapshot",
        Some(ADMIN_READ_TIMEOUT),
    ) else {
        return false;
    };
    let Ok(snap) = serde_json::from_str::<AdminSnapshot>(&body) else {
        return false;
    };
    snap.realms_present(&roster_realms(shape, &DEV))
}

/// Reap every realm shard the orchestrator forked, named by the durable launch ledger this cluster wrote.
///
/// BEST-EFFORT BY DESIGN: a demand cluster that never spawned has no ledger, and a `down` after a crash may
/// find a ledger it cannot open. Neither is a reason to refuse to tear the rest of the cluster down, so this
/// reports and returns rather than failing `down`. What it must NEVER do is silently skip a live survivor —
/// hence the count is printed whenever it is non-zero.
fn reap_demand_spawned(work: &Path) {
    let ledger = work.join(vd_bins::LAUNCH_STORE_NAME);
    if !ledger.exists() {
        return; // a static cluster, or a demand cluster that never spawned a realm
    }
    // `launch_rows` panics on a corrupt ledger (a hard failure for a test); here a corrupt ledger must not
    // block teardown, so isolate it and fall through to the recorded-group reap below.
    let rows = match std::panic::catch_unwind(|| vd_bins::launch_rows(&ledger)) {
        Ok(rows) => rows,
        Err(_) => {
            eprintln!(
                "dev-cluster: WARNING — could not read {} to reap demand-spawned shards; \
                 if a later `up --demand` fails to spawn realms, check for survivors holding the RLM port band",
                ledger.display()
            );
            return;
        }
    };
    let live = rows.iter().filter(|(_, _, pid)| pid.is_some()).count();
    if live > 0 {
        println!("dev-cluster: reaping {live} demand-spawned realm shard(s)");
    }
    vd_bins::reap_forked(&rows);
}

/// Tear the cluster down: SIGTERM each recorded process group, escalate to SIGKILL
/// after a grace period, and remove the runfile/workdir ONLY once every process is
/// confirmed dead — otherwise report LOUD and leave the record for a retry.
fn down(work: &Path) -> Result<(), String> {
    if !runfile(work).exists() {
        // No claim. Clear any stray workdir (e.g. a crash between create_dir_all
        // and the O_EXCL claim) so a future `up` is never wedged, then report
        // idempotently. `down` ALWAYS leaves the slot reusable.
        if work.exists() {
            std::fs::remove_dir_all(work).map_err(|e| format!("clean work dir: {e}"))?;
        }
        println!("dev-cluster: already down ({})", work.display());
        return Ok(());
    }
    // The runfile is the claim; it may hold 0..3 pids (a SIGKILL mid-bring-up). Kill
    // whatever was recorded, then clear the claim — a 0-pid runfile reaps to nothing
    // and is cleared, so a claim-only crash state is always recoverable here.
    let pids = read_pids(work)?;
    for pid in &pids {
        vd_bins::signal_group(*pid, "TERM");
    }
    let deadline = Instant::now() + TERM_GRACE;
    while Instant::now() < deadline && pids.iter().any(|p| vd_bins::pid_alive(*p)) {
        std::thread::sleep(POLL_INTERVAL);
    }
    for pid in &pids {
        if vd_bins::pid_alive(*pid) {
            vd_bins::signal_group(*pid, "KILL");
        }
    }
    // Brief settle, then confirm.
    std::thread::sleep(POLL_INTERVAL);
    let survivors: Vec<u32> = pids
        .into_iter()
        .filter(|p| vd_bins::pid_alive(*p))
        .collect();
    if !survivors.is_empty() {
        return Err(format!(
            "could not kill {survivors:?}; leaving the runfile ({}) for a retry",
            runfile(work).display()
        ));
    }
    // NOW reap the shards the ORCHESTRATOR forked (a demand cluster's realms). They are NOT in the runfile
    // and they LEAD THEIR OWN PROCESS GROUPS (`spawn_node_grouped`), so the signalling above never reaches
    // them — before this they simply outlived the cluster.
    //
    // WHY IT IS LOAD-BEARING, not tidiness: demand-spawned shards bind a FIXED port band
    // (`VD_RLM_FIRST_PORT`/`VD_RLM_PORT_LIMIT`), never ephemeral ports. One survivor therefore squats that
    // band FOREVER, and every later demand cluster's spawns die with `AddrInUse` the instant they start. The
    // symptom is remote from the cause and was silent: the player sat in `awaiting_subscription` with no
    // world while the orchestrator retried forever and nothing named the collision.
    //
    // ORDER MATTERS: this runs AFTER the recorded groups are confirmed dead, because the ledger is redb and
    // the LIVE orchestrator holds its file lock — reaping first can only ever fail to open it. It runs
    // BEFORE the workdir is removed, because the ledger lives in that directory.
    //
    // The reap primitive already existed, used by the kill-9 gate and the demand-login e2e: the TESTS
    // cleaned up after themselves while the launcher people actually run did not. Same primitive, one reap
    // path (HR3).
    reap_demand_spawned(work);
    std::fs::remove_dir_all(work).map_err(|e| format!("clean work dir: {e}"))?;
    println!("dev-cluster DOWN ({})", work.display());
    Ok(())
}

fn status(ports: SlotPorts, work: &Path) -> Result<(), String> {
    if !runfile(work).exists() {
        println!("dev-cluster: no runfile ({}) — not up", work.display());
        return Ok(());
    }
    let pids = read_pids(work).unwrap_or_default();
    let live = pids.iter().filter(|p| vd_bins::pid_alive(**p)).count();
    // `status` reports the base bootstrap signal (the home realm granted) — it does not re-derive the
    // shape (the pid count already reflects 3 vs 4 vs 6 nodes; a multi-shard cluster shows the base
    // signal once the home shard grants).
    let ready = admin_ready(ports.admin, ClusterShape::Single);
    println!(
        "dev-cluster: {live}/{} node processes alive; admin {} => {}",
        pids.len(),
        loopback(ports.admin),
        if ready { "READY" } else { "not ready" }
    );
    Ok(())
}

// ---- spawn plumbing ----------------------------------------------------------

/// Spawn a sibling node binary (resolved relative to this launcher) in its OWN
/// process group, with the merged env, redirecting its output to `<work>/<bin>.log`.
/// Thin wrapper over the SHARED [`vd_bins::spawn_node_grouped`] (the RLM 5c DRY lift): it owns only the
/// launcher's naming conventions (sibling-exe resolution + the `<work>/<label>.log` path).
fn spawn_node(
    bin: &str,
    label: &str,
    work: &Path,
    common: &[(&'static str, String)],
    node_env: Vec<(&'static str, String)>,
) -> Result<Child, String> {
    // HR3: ONE shard binary — the SOURCE and DEST both run `vd-shard`, distinguished by env + a distinct
    // `label` (the DEST logs to `vd-shard-b.log` while spawning the SAME `vd-shard` executable), never a
    // per-shard binary. `bin` is the sibling executable; `label` names the log + the kill-record entry.
    let exe = vd_bins::sibling_binary(bin)?;
    let log =
        File::create(work.join(format!("{label}.log"))).map_err(|e| format!("log file: {e}"))?;
    vd_bins::spawn_node_grouped(&exe, common, &node_env, log)
        .map_err(|e| format!("spawn {label}: {e}"))
}

/// The dev-control client address book seeded into the gateway: one
/// `(CLIENT_NODE_BASE + agent → 127.0.0.1:client_quic)` per slot client window.
/// This is the CA-1 crutch (see the call site) — superseded by reply-on-connection.
fn client_book(ports: SlotPorts) -> Result<Vec<(NodeId, SocketAddr)>, String> {
    (0..DevPortScheme::DEFAULT.max_clients_per_worktree)
        .map(|agent| {
            let port = ports.client_quic(agent).map_err(|e| e.to_string())?;
            Ok((NodeId(CLIENT_NODE_BASE + u64::from(agent)), loopback(port)))
        })
        .collect()
}

// ---- runfile / env contract --------------------------------------------------

/// Append one child's pid to the runfile (the kill record) and flush it durably —
/// called the instant the child spawns so `down` can always reap it.
fn record_pid(runfile: &mut std::fs::File, pid: u32) -> Result<(), String> {
    writeln!(runfile, "{pid}").map_err(|e| format!("record pid: {e}"))?;
    runfile.flush().map_err(|e| format!("flush runfile: {e}"))
}

#[allow(clippy::too_many_arguments)]
fn write_env_contract(
    work: &Path,
    slot: u16,
    gateway: SocketAddr,
    admin: SocketAddr,
    orch_probe: SocketAddr,
    gateway_probe: SocketAddr,
    shard_probe: SocketAddr,
    trust_dir: &str,
) -> Result<(), String> {
    // Every value sh-quoted so a path with spaces/metacharacters can't word-split
    // or inject when `client.sh` sources this. The three probe addrs are emitted so an S4 smoke /
    // shell scenario reads /healthz + /readyz ports from THIS one contract, never re-deriving the
    // DevPortScheme offsets by hand (the hand-computed-port hazard devproto forbids).
    let env = format!(
        "VD_SLOT={}\n\
         VD_GW_ADDR={}\n\
         VD_ADMIN_ADDR={}\n\
         VD_ORCH_PROBE_ADDR={}\n\
         VD_GW_PROBE_ADDR={}\n\
         VD_SHARD_PROBE_ADDR={}\n\
         VD_TRUST_DIR={}\n\
         VD_AUTH_SIGNING_KEY={}\n",
        sh_quote(&slot.to_string()),
        sh_quote(&gateway.to_string()),
        sh_quote(&admin.to_string()),
        sh_quote(&orch_probe.to_string()),
        sh_quote(&gateway_probe.to_string()),
        sh_quote(&shard_probe.to_string()),
        sh_quote(trust_dir),
        sh_quote(&vd_bins::dev_auth_signing_key_hex()),
    );
    std::fs::write(env_file(work), env).map_err(|e| format!("write env file: {e}"))
}

fn read_pids(work: &Path) -> Result<Vec<u32>, String> {
    let text = std::fs::read_to_string(runfile(work))
        .map_err(|e| format!("no runfile ({}): {e}", runfile(work).display()))?;
    text.lines()
        .map(|l| {
            l.trim()
                .parse::<u32>()
                .map_err(|_| format!("bad pid `{l}`"))
        })
        .collect()
}

/// Print the last few lines of a node's captured log to stderr (bring-up failure
/// diagnostics — the cause is shown, never swallowed by the cleanup).
fn dump_log_tail(work: &Path, bin: &str) {
    let path = work.join(format!("{bin}.log"));
    if let Ok(text) = std::fs::read_to_string(&path) {
        let tail: Vec<&str> = text.lines().rev().take(8).collect();
        if !tail.is_empty() {
            eprintln!("--- {bin} log tail ---");
            for line in tail.iter().rev() {
                eprintln!("  {line}");
            }
        }
    }
}

// ---- small helpers -----------------------------------------------------------

fn parse_slot(args: &[String]) -> Result<u16, String> {
    let mut it = args.iter();
    while let Some(a) = it.next() {
        if a == "--slot" {
            return it
                .next()
                .ok_or("--slot needs a value")?
                .parse::<u16>()
                .map_err(|_| "--slot must be a u16".to_owned());
        }
    }
    Err("--slot is required".to_owned())
}

fn work_dir(slot: u16) -> PathBuf {
    slot_workdir(slot) // ONE layout definition (vd_bins), shared with the process tests
}

fn runfile(work: &Path) -> PathBuf {
    work.join(RUNFILE_NAME)
}

fn env_file(work: &Path) -> PathBuf {
    work.join("cluster.env")
}

#[cfg(test)]
mod tests {
    use super::gen_trust;
    use vd_io_prod::trust::ClusterTrust;

    #[test]
    fn gen_trust_writes_a_bundle_the_mesh_can_load() {
        // `gen-trust <dir>` mints the shared cluster secret: assert the three DER files exist and
        // round-trip through `from_der_dir` — the exact load every node/pod does at boot (the k3d
        // Secret + the docker-run smoke both mount what this writes).
        let dir = std::env::temp_dir().join(format!("vd-gentrust-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        gen_trust(&dir).expect("gen-trust writes the bundle");
        assert!(dir.join("ca.der").exists(), "ca.der written");
        assert!(dir.join("node.der").exists(), "node.der written");
        assert!(dir.join("key.der").exists(), "key.der written");
        ClusterTrust::from_der_dir(&dir).expect("the written bundle loads back");
        std::fs::remove_dir_all(&dir).expect("cleanup");
    }
}
