//! The headless dev-control client (HR6) — the Tier-B process shell around the
//! Tier-A `vd-client` core. It wires three roles, matching the Slice-2 thread model:
//!
//! - **net** (the tokio runtime): the production `MeshTransport` — quinn reader/
//!   writer tasks bridging to the synchronous core over bounded queues.
//! - **core** (THIS thread): the single-threaded deterministic `ClientCore` — the
//!   SOLE owner of the client state. Each step drains the dev-command mailbox,
//!   steps the core (drain inbound → assemble input), then publishes the decoded
//!   delivered `DevState` to an `ArcSwap` for the listener to read lock-free.
//! - **dev-control** (`#[cfg(feature = "dev-control")]`, the tokio runtime): a
//!   loopback JSON-lines listener. Mutating commands are gated by
//!   `--allow-dev-control`; everything is shed-not-blocked through a bounded mailbox.
//!
//! The dev-control listener is `cfg`-gated, NON-default: a release build of
//! `vd-bins` contains no listener code at all (absence is a compile fact). The bin
//! reads wall-time from the OS (the lib never does — it takes `now_s` as an arg).

use std::collections::BTreeMap;
use std::net::SocketAddr;
use std::path::Path;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
#[cfg(feature = "render")]
use std::sync::mpsc::SyncSender;
use std::sync::mpsc::{Receiver, sync_channel};
use std::time::Instant;

use arc_swap::ArcSwap;
use vd_bins::GATEWAY;
// `loopback` is now used ONLY by the dev-control listener bind (the QUIC bind moved to a configurable
// `--bind` addr), which is itself feature-gated — so the import is too, else the default build sees it unused.
#[cfg(feature = "dev-control")]
use vd_bins::loopback;
use vd_client::net::{ClientCore, ClientPhase};
use vd_client::render_snapshot::RenderSnapshot;
use vd_client::tuning::ClientInterpTuning;
use vd_connection_plane::tickets::mint_login;
use vd_core::{AccountId, EpochId, NodeId};
use vd_devproto::{CLIENT_NODE_BASE, DevState, InputAction};
use vd_io_prod::mesh::{MeshConfig, MeshTransport, spawn_mesh};
use vd_io_prod::runtime::{EnvConfig, TickPacer};
use vd_io_prod::trust::ClusterTrust;

/// The dev-client account namespace — `AccountId(BASE + agent_index)`, matching the
/// process-parity convention (`AccountId(1000)`/`1001`). Distinct per agent ⇒
/// distinct gateway sessions; `max_sessions` (8) covers the K=4 client cap.
const DEV_CLIENT_ACCOUNT_BASE: u64 = 1000;

/// The mesh outbound queue depth (matches the parity client's book; one peer).
const CLIENT_OUTBOUND_CAP: usize = 64;

/// The bounded dev-command mailbox depth. Past this the listener sheds commands
/// LOUDLY (`DevError::Busy` + the `dev_commands_dropped` running total) rather than
/// blocking the deterministic step thread — back-pressure is observable, never a hang.
const COMMAND_MAILBOX_CAP: usize = 256;

/// The headless client's drain+input cadence (Hz). One step = one inbound drain +
/// one 20 Hz `InputDatagram`. Slice 3's real render loop will drive `step()` at the
/// display refresh instead; until then this is the single knob, overridable via
/// `--step-hz`.
const DEFAULT_STEP_HZ: u32 = 20;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    vd_bins::init_tracing();
    let args = parse_args()?;
    let env = EnvConfig::from_process_env();
    let signing_key = env.hex32("VD_AUTH_SIGNING_KEY")?;

    let local = NodeId(CLIENT_NODE_BASE + args.agent_index);
    let account = AccountId(u128::from(DEV_CLIENT_ACCOUNT_BASE + args.agent_index));
    // The auth service's half: a deterministic Ed25519 login over (account, epoch,
    // nonce) under the dev seed — what the gateway validates against `VD_AUTH_PUBKEY`.
    // The nonce is deterministic dev seeding (a restart re-presents it, idempotent);
    // single-use login-nonce replay protection is a later-phase auth-service concern.
    let ticket = mint_login(&signing_key, account, EpochId(1), account.0 as u64);
    tracing::info!(
        name = %args.name, node = local.0, account = account.0, gateway = %args.gateway,
        "client starting",
    );

    let runtime = tokio::runtime::Builder::new_multi_thread()
        .worker_threads(2)
        .enable_all()
        .build()?;
    let trust = ClusterTrust::from_der_dir(Path::new(&args.trust_dir))?;
    // STRUCTURAL one-connection invariant: the address book holds ONLY the gateway —
    // no other node is reachable, by construction (the client never re-homes).
    let peers = BTreeMap::from([(GATEWAY, args.gateway)]);
    let (transport, control) = spawn_mesh(
        runtime.handle(),
        &trust,
        &MeshConfig::new(
            local,
            // The QUIC bind: loopback for the local dev cluster (default), 0.0.0.0 (`--bind 0.0.0.0`) for an
            // in-cluster agent so it can reach the gateway over the pod network (a loopback source can't route
            // off-host). CA-1 reply-on-connection means the gateway learns this client's addr from the accepted
            // connection, so no advertised/booked address is needed.
            std::net::SocketAddr::new(args.bind_ip, args.client_quic),
            peers,
            CLIENT_OUTBOUND_CAP,
            // R-2b: a fresh process per client launch; 0 is correct here. NOTE: once R-3' dedup is
            // live, a crashed client reconnecting at incarnation 0 collides with its prior session's
            // buffered frames at the gateway — R-6's durable boot-counter / R-3' session reset close it.
            0,
        ),
        None, // R-6d3a: the client has no producer-less durable flows — no outbox needed.
    )?;
    let mut core = ClientCore::new(transport, GATEWAY, ticket, ClientInterpTuning::DEFAULT);

    // THE DEBUG BOOT SCENE (dev-control builds only — `debug_scene_arg` refuses it otherwise). Local file
    // geometry drawn while the client has been told nothing yet, so a playground smoke has something on
    // screen before the first streamed scene message arrives; that message REPLACES this whole scene with
    // what the shard chain shipped, and from then on the client draws only what it was sent.
    //
    // IT IS NOT CONVERTED AND MUST NOT BE. The centres in these files are each realm's placement in its
    // PARENT's frame, which is not the space this session draws in. Converting them here would mean the
    // client holding the whole forest and folding a chain of realms — the exact arrangement the server-side
    // chain replaced. Preferred: a `regions.json` (`Vec<RealmRegion>` = the SEED forest, ambient shells
    // auto-skipped). Legacy fallback: a `boxes.json` (`Vec<RealmBoundary>`) for the authored playground
    // OVERRIDE smokes. A file that parses as NEITHER fails LOUD at boot (never a silent empty scene).
    if let Some(path) = &args.realm_boxes {
        let json =
            std::fs::read_to_string(path).map_err(|e| format!("read --realm-boxes {path}: {e}"))?;
        let scene = vd_client::realm_scene::RealmScene::from_regions_json(&json)
            .or_else(|_| vd_client::realm_scene::RealmScene::from_boxes_json(&json))
            .map_err(|e| {
                format!("parse --realm-boxes {path} as regions.json or boxes.json: {e}")
            })?;
        core.state_mut().load_scene(scene);
    }

    // The lock-free step ↔ listener bridge: ArcSwap publishes the decoded delivered
    // DevState; the bounded mailbox carries injected InputActions; the atomics carry
    // the step clock (wait-until termination) and the shed-command running total.
    let published = Arc::new(ArcSwap::from_pointee(core.state().devstate(0.0, 0, 0)));
    let step_seq = Arc::new(AtomicU64::new(0));
    let dropped = Arc::new(AtomicU64::new(0));
    let (command_tx, command_rx) = sync_channel::<InputAction>(COMMAND_MAILBOX_CAP);
    // The capture seam (Capture mode): the dev-control screenshot handler → the Bevy
    // render thread. Created in any dev-control+render build; only WIRED into the
    // listener (and consumed by the capture app) when `--capture` is set.
    #[cfg(all(feature = "dev-control", feature = "render"))]
    let (capture_tx, capture_rx) = crossbeam_channel::unbounded::<vd_client_render::CaptureJob>();
    // The capture run dir + the shared run manifest (Capture mode): created up-front so the
    // dev-control listener — the manifest's SINGLE writer (one `Mutex`, so concurrent
    // connections can't race the file) — can record every capture (PNG path + aligned state
    // + tick). The render thread writes the PNGs INTO this same dir; the dev side owns
    // `manifest.json` and the `state/` dumps. `started_utc` is stamped here (the bin reads
    // the clock; the lib never does — the manifest takes the timestamp IN).
    #[cfg(all(feature = "dev-control", feature = "render"))]
    let capture_run: Option<(std::path::PathBuf, dev_control::SharedManifest)> =
        args.capture.then(|| {
            let secs = unix_secs();
            let manifest = std::sync::Arc::new(std::sync::Mutex::new(
                vd_client_harness::manifest::RunManifest::new(args.name.clone(), iso_utc(secs)),
            ));
            (
                capture_run_dir(secs, &args.name, args.agent_index),
                manifest,
            )
        });

    #[cfg(feature = "dev-control")]
    let _listener = match args.dev_control {
        Some(port) => {
            tracing::info!(
                port,
                allow_mutating = args.allow_dev_control,
                "dev-control up"
            );
            let handles = dev_control::Handles {
                commands: command_tx.clone(),
                published: published.clone(),
                step_seq: step_seq.clone(),
                dropped: dropped.clone(),
                allow_mutating: args.allow_dev_control,
                // Wire the capture sender only in `--capture` mode; otherwise `Screenshot`/
                // `Record` are `Unsupported` (windowed/headless have no offscreen readback).
                #[cfg(feature = "render")]
                captures: args.capture.then(|| capture_tx.clone()),
                // The run dir + manifest, so the listener can persist each capture. Cloned
                // per connection; the `Arc<Mutex<…>>` keeps one writer across them all.
                #[cfg(feature = "render")]
                runs_dir: capture_run.as_ref().map(|(dir, _)| dir.clone()),
                #[cfg(feature = "render")]
                manifest: capture_run.as_ref().map(|(_, m)| m.clone()),
            };
            Some(runtime.spawn(dev_control::serve(loopback(port), handles)))
        }
        // No listener: the later dispatch drops `command_tx` (headless) or hands it to
        // the window — so the mailbox's producer lifetime is owned by the dispatch, not
        // here.
        None => None,
    };
    #[cfg(not(feature = "dev-control"))]
    let _ = (&args.dev_control, args.allow_dev_control);

    let started_at = Instant::now();

    // `--window`/`--capture` need the render feature (Bevy); `--capture` also needs
    // dev-control (to receive the screenshot request). Reject early otherwise.
    #[cfg(not(feature = "render"))]
    if args.window || args.capture {
        return Err("--window/--capture require building with --features render".into());
    }
    #[cfg(all(feature = "render", not(feature = "dev-control")))]
    if args.capture {
        return Err("--capture requires --features dev-control".into());
    }

    // Windowed (T4) / headless-capture (T5): the core loop runs on a WORKER thread at a
    // deterministic 20 Hz while Bevy owns the MAIN thread; the render app reads the
    // published RenderSnapshot. Windowed feeds the input mailbox; capture writes PNGs.
    #[cfg(feature = "render")]
    if args.window {
        let render_published = Arc::new(ArcSwap::from_pointee(core.state().render_snapshot()));
        run_render(
            core,
            command_rx,
            &command_tx,
            published,
            render_published,
            step_seq,
            dropped,
            started_at,
            args.step_hz,
            vd_client_render::RenderMode::Windowed,
            None,
            std::path::PathBuf::from("runs"),
        );
        drop(control);
        return Ok(());
    }
    #[cfg(all(feature = "dev-control", feature = "render"))]
    if args.capture {
        let render_published = Arc::new(ArcSwap::from_pointee(core.state().render_snapshot()));
        // The same run dir the listener writes the manifest into (created up-front above).
        let runs_dir = capture_run
            .map(|(dir, _)| dir)
            .expect("capture mode ⇒ capture_run was created");
        // GPU PRECONDITION (cloud-1): headless capture renders through wgpu and REQUIRES a
        // working GPU adapter (the dev Apple-Silicon Metal GPU today). G-RENDER-SMOKE is a
        // LOCAL gate — there is no software-rasterizer fallback (it would poison the visual
        // baseline) and no CI yet. A host with no adapter fails in wgpu adapter selection;
        // steer the backend/power-pref with WGPU_BACKENDS / WGPU_POWER_PREF if needed.
        tracing::info!(
            dir = %runs_dir.display(),
            "headless capture run — requires a GPU adapter (steer via WGPU_BACKENDS/WGPU_POWER_PREF)",
        );
        run_render(
            core,
            command_rx,
            &command_tx,
            published,
            render_published,
            step_seq,
            dropped,
            started_at,
            args.step_hz,
            vd_client_render::RenderMode::Capture,
            Some(capture_rx),
            runs_dir,
        );
        drop(control);
        return Ok(());
    }

    // Headless (default + the process-tier path): the core loop owns THIS thread. Drop
    // our spare producer handle so the mailbox closes cleanly when the listener is gone.
    drop(command_tx);
    run_client_loop(
        core,
        command_rx,
        published,
        step_seq,
        dropped,
        None, // no render sink
        None, // no external stop signal (exits on vdctl/gateway Close)
        started_at,
        args.step_hz,
    );

    drop(control); // close the QUIC endpoint on a graceful exit
    Ok(())
}

/// Run the render client (Windowed or headless Capture): spawn the deterministic core
/// loop on a worker thread, then run Bevy on this (main) thread until it exits, then ask
/// the core to close. Shared by `--window` and `--capture` (only the `RenderHandles`
/// mode/captures/runs_dir differ — one render path, two front-ends).
#[cfg(feature = "render")]
#[allow(clippy::too_many_arguments)]
fn run_render(
    core: ClientCore<MeshTransport>,
    command_rx: Receiver<InputAction>,
    command_tx: &SyncSender<InputAction>,
    published: Arc<ArcSwap<DevState>>,
    render_published: Arc<ArcSwap<RenderSnapshot>>,
    step_seq: Arc<AtomicU64>,
    dropped: Arc<AtomicU64>,
    started_at: Instant,
    step_hz: u32,
    mode: vd_client_render::RenderMode,
    captures: Option<crossbeam_channel::Receiver<vd_client_render::CaptureJob>>,
    runs_dir: std::path::PathBuf,
) {
    // Reliable, BIDIRECTIONAL shutdown — independent of the bounded input mailbox:
    //  - `stop` (window → worker): set when the window closes; the worker checks it every
    //    tick and drives a graceful Close. An AtomicBool can never be "full"/shed, so the
    //    close signal can never be lost (the old best-effort `try_send(Close)` could be —
    //    and then `join()` would hang the process forever).
    //  - `core_alive` (worker → window): a drop-guard flips it false when the worker loop
    //    EXITS for ANY reason (a gateway Close, or a panic unwinding the thread); a Bevy
    //    system emits AppExit when it sees false, so a server-initiated disconnect or a
    //    dead core tears the window down too — no zombie frozen window.
    let stop = Arc::new(AtomicBool::new(false));
    let core_alive = Arc::new(AtomicBool::new(true));
    let render_sink = render_published.clone();
    let worker_stop = stop.clone();
    let worker_alive = core_alive.clone();
    let worker_dropped = dropped.clone();
    let core_thread = std::thread::spawn(move || {
        // Flip `core_alive` false on EXIT or PANIC so the window always learns.
        struct AliveGuard(Arc<AtomicBool>);
        impl Drop for AliveGuard {
            fn drop(&mut self) {
                self.0.store(false, Ordering::Relaxed);
            }
        }
        let _alive = AliveGuard(worker_alive);
        run_client_loop(
            core,
            command_rx,
            published,
            step_seq,
            worker_dropped,
            Some(render_sink),
            Some(worker_stop),
            started_at,
            step_hz,
        );
    });
    // Blocks on the main thread until the window closes — by the human, OR by the
    // AppExit the render crate emits when `core_alive` goes false.
    vd_client_render::run(vd_client_render::RenderHandles {
        snapshot: render_published,
        input: command_tx.clone(),
        dropped,
        core_alive,
        started_at,
        mode,
        captures,
        runs_dir,
    });
    // Window/app exited → reliably ask the core to close (AtomicBool — never shed), join.
    stop.store(true, Ordering::Relaxed);
    let _ = core_thread.join();
}

/// Wall-clock seconds since the unix epoch (the bin reads the OS clock; the lib never
/// does). Used for the run-dir name AND the manifest's `started_utc`.
#[cfg(all(feature = "dev-control", feature = "render"))]
fn unix_secs() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

/// Format unix `secs` as an RFC-3339 UTC timestamp (`YYYY-MM-DDThh:mm:ssZ`) for the run
/// manifest's `started_utc`. Uses Howard Hinnant's `civil_from_days` (exact, no leap-second
/// table) so the manifest carries a real calendar time with no new dependency.
#[cfg(all(feature = "dev-control", feature = "render"))]
fn iso_utc(secs: u64) -> String {
    let days = (secs / 86_400) as i64;
    let sod = secs % 86_400;
    let (hh, mm, ss) = (sod / 3600, (sod % 3600) / 60, sod % 60);
    // civil_from_days: days since 1970-01-01 → (year, month, day).
    let z = days + 719_468;
    let era = if z >= 0 { z } else { z - 146_096 } / 146_097;
    let doe = z - era * 146_097; // [0, 146096]
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365; // [0, 399]
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100); // [0, 365]
    let mp = (5 * doy + 2) / 153; // [0, 11]
    let d = doy - (153 * mp + 2) / 5 + 1; // [1, 31]
    let m = if mp < 10 { mp + 3 } else { mp - 9 }; // [1, 12]
    let y = yoe + era * 400 + i64::from(m <= 2);
    format!("{y:04}-{m:02}-{d:02}T{hh:02}:{mm:02}:{ss:02}Z")
}

/// The output dir for a capture run: `runs/<unix_secs>__<name>__a<agent>/` (created here);
/// the manifest + PNGs land inside. The timestamp keeps successive runs distinct, and the
/// AGENT INDEX keeps concurrent same-named clients distinct — the P2 paired visual scenario
/// launches K clients in the same second with the same default name, and each must own its
/// run dir (a shared dir would silently interleave two manifests). The name is sanitized by
/// the SAME Tier-A rule capture labels use (one filename discipline).
#[cfg(all(feature = "dev-control", feature = "render"))]
fn capture_run_dir(secs: u64, name: &str, agent_index: u64) -> std::path::PathBuf {
    let safe = vd_client_harness::capture::sanitize_stem(name);
    let dir = std::path::PathBuf::from("runs").join(format!("{secs}__{safe}__a{agent_index}"));
    let _ = std::fs::create_dir_all(&dir);
    dir
}

/// The deterministic core loop (the sole owner of `core` — on the main thread for the
/// headless client, or a worker thread for the windowed one): drain the dev-command
/// mailbox, step, publish the decoded `DevState` (+ the `RenderSnapshot` if a render sink
/// is wired), advance the step clock. Exits when the session closes (client `Close` or a
/// gateway `Close`). `started_at` is the shared monotonic epoch — the windowed renderer
/// computes its display cursor in this SAME timeline so motion stays continuous.
#[allow(clippy::too_many_arguments)]
fn run_client_loop(
    mut core: ClientCore<MeshTransport>,
    commands: Receiver<InputAction>,
    published: Arc<ArcSwap<DevState>>,
    step_seq: Arc<AtomicU64>,
    dropped: Arc<AtomicU64>,
    render_sink: Option<Arc<ArcSwap<RenderSnapshot>>>,
    stop: Option<Arc<AtomicBool>>,
    started_at: Instant,
    step_hz: u32,
) {
    let mut pacer = TickPacer::new(step_hz);
    // `applied` counts ALL drained dev actions (Move/Look/Action AND Close/Reset) —
    // it is the dev-command throughput counter, distinct from `sent_input_count`,
    // which honestly counts input FRAMES that rode the wire.
    let mut applied: u64 = 0;
    let mut stop_injected = false;
    loop {
        // Drain-at-top: apply every queued dev command before stepping, so an
        // injected input rides THIS tick's InputDatagram (the agent's seam).
        while let Ok(action) = commands.try_recv() {
            core.state_mut().apply_input_action(action);
            applied += 1;
        }
        // A stop request (window closed) injects a graceful Close ONCE — reliably, via
        // the AtomicBool, NOT the shed-able mailbox — so the next assemble sends Bye, the
        // phase flips to Closed, and the loop exits below (never a lost-close hang).
        if !stop_injected && stop.as_ref().is_some_and(|s| s.load(Ordering::Relaxed)) {
            core.state_mut().apply_input_action(InputAction::Close);
            stop_injected = true;
        }
        let now_s = started_at.elapsed().as_secs_f64();
        core.step(now_s);
        // The DevState is consumed ONLY by the dev-control listener (cfg dev-control). A
        // release client (render, no dev-control) builds none of this String-heavy struct
        // every step. `published` is created in `main` regardless (one cheap alloc) so the
        // signature stays uniform; without a listener it simply never updates.
        #[cfg(feature = "dev-control")]
        published.store(Arc::new(core.state().devstate(
            now_s,
            applied,
            dropped.load(Ordering::Relaxed),
        )));
        #[cfg(not(feature = "dev-control"))]
        let _ = (&published, applied, &dropped);
        if let Some(sink) = &render_sink {
            sink.store(Arc::new(core.state().render_snapshot()));
        }
        step_seq.fetch_add(1, Ordering::Relaxed);
        if core.state().phase() == ClientPhase::Closed {
            break;
        }
        pacer.wait();
    }
}

/// The parsed command line (`scripts/client.sh` supplies these flags; the auth
/// signing key arrives via `VD_AUTH_SIGNING_KEY` in the cluster env).
struct ClientArgs {
    name: String,
    gateway: SocketAddr,
    client_quic: u16,
    /// The IP the client's QUIC endpoint binds. Default `127.0.0.1` (the local dev cluster: client + gateway
    /// share loopback). MUST be `0.0.0.0` for an in-cluster (k3d) agent — a loopback-bound socket cannot reach
    /// a gateway on the pod network (the source addr never routes off loopback → `sendmsg` fails).
    bind_ip: std::net::IpAddr,
    trust_dir: String,
    dev_control: Option<u16>,
    agent_index: u64,
    allow_dev_control: bool,
    step_hz: u32,
    /// Open the Bevy window (Slice-3 T4) instead of running headless. Requires the
    /// `render` build feature.
    window: bool,
    /// Headless offscreen render + wgpu readback for `vdctl screenshot` (Slice-3 T5) — the
    /// agent's eyes, no display. Requires `--features dev-control,render` + `--dev-control`.
    capture: bool,
    /// A DEBUG boot scene read off local disk — a `regions.json` (`Vec<RealmRegion>`, the seed forest)
    /// or the legacy authored `boxes.json` (`Vec<RealmBoundary>`) — drawn as translucent realm boxes
    /// while the client has been told nothing (Visual Crossing Playground V2). `None` ⇒ no boxes.
    ///
    /// IT IS NOT WORLD TRUTH AND IS NOT ON THE NETWORKED PATH. The centres in those files are each
    /// realm's placement in its own PARENT's frame, which is not the space this session draws in, and
    /// nothing converts them — nothing can, because converting them would mean the client holding the
    /// whole forest and folding a chain of realms, which is precisely what the server-side chain exists
    /// to stop. It survives only until the first streamed scene message, which REPLACES the whole scene
    /// with what the shard chain shipped.
    ///
    /// It is therefore refused outright unless this is a `dev-control` build, so it cannot become a
    /// second source of world geometry in anything a player runs.
    realm_boxes: Option<String>,
}

fn parse_args() -> Result<ClientArgs, String> {
    let mut name = "client".to_owned();
    let mut gateway: Option<SocketAddr> = None;
    let mut client_quic: Option<u16> = None;
    let mut bind_ip: std::net::IpAddr = std::net::IpAddr::V4(std::net::Ipv4Addr::LOCALHOST);
    let mut trust_dir: Option<String> = None;
    let mut dev_control: Option<u16> = None;
    let mut agent_index: u64 = 0;
    let mut allow_dev_control = false;
    let mut step_hz: u32 = DEFAULT_STEP_HZ;
    let mut window = false;
    let mut capture = false;
    let mut realm_boxes: Option<String> = None;

    let mut it = std::env::args().skip(1);
    while let Some(arg) = it.next() {
        match arg.as_str() {
            "--name" => name = next_val(&mut it, "--name")?,
            "--gateway" => gateway = Some(parse_val(&mut it, "--gateway")?),
            "--client-quic" => client_quic = Some(parse_val(&mut it, "--client-quic")?),
            "--bind" => bind_ip = parse_val(&mut it, "--bind")?,
            "--trust-dir" => trust_dir = Some(next_val(&mut it, "--trust-dir")?),
            "--dev-control" => dev_control = Some(parse_val(&mut it, "--dev-control")?),
            "--agent-index" => agent_index = parse_val(&mut it, "--agent-index")?,
            "--allow-dev-control" => allow_dev_control = true,
            "--step-hz" => step_hz = parse_val(&mut it, "--step-hz")?,
            "--window" => window = true,
            "--capture" => capture = true,
            "--realm-boxes" => {
                realm_boxes = Some(debug_scene_arg(next_val(&mut it, "--realm-boxes")?)?);
            }
            other => return Err(format!("unknown argument: {other}")),
        }
    }
    if window && capture {
        return Err("--window and --capture are mutually exclusive".to_owned());
    }

    Ok(ClientArgs {
        name,
        gateway: gateway.ok_or("missing --gateway <addr>")?,
        client_quic: client_quic.ok_or("missing --client-quic <port>")?,
        bind_ip,
        trust_dir: trust_dir.ok_or("missing --trust-dir <dir>")?,
        dev_control,
        agent_index,
        allow_dev_control,
        step_hz,
        window,
        capture,
        realm_boxes,
    })
}

/// Accept `--realm-boxes` ONLY in a dev-control build — the flag names a local file of world geometry
/// nobody streamed, so it is a debugging affordance and is gated like one. See `ClientArgs::realm_boxes`.
#[cfg(feature = "dev-control")]
fn debug_scene_arg(path: String) -> Result<String, String> {
    Ok(path)
}

/// The non-dev twin: refuse LOUD rather than silently drawing file geometry beside streamed geometry.
#[cfg(not(feature = "dev-control"))]
fn debug_scene_arg(_path: String) -> Result<String, String> {
    Err(
        "--realm-boxes is a debug-only boot scene (local file geometry, not the streamed world) and \
         needs a `dev-control` build"
            .to_owned(),
    )
}

fn next_val(it: &mut impl Iterator<Item = String>, flag: &str) -> Result<String, String> {
    it.next().ok_or_else(|| format!("{flag} requires a value"))
}

fn parse_val<T: std::str::FromStr>(
    it: &mut impl Iterator<Item = String>,
    flag: &str,
) -> Result<T, String> {
    let raw = next_val(it, flag)?;
    raw.parse()
        .map_err(|_| format!("{flag}: cannot parse {raw:?}"))
}

/// The loopback JSON-lines dev-control listener (HR6) — present ONLY under the
/// `dev-control` feature, so a release build links none of it.
#[cfg(feature = "dev-control")]
mod dev_control {
    use std::net::SocketAddr;
    use std::sync::Arc;
    use std::sync::atomic::{AtomicU64, Ordering};
    use std::sync::mpsc::SyncSender;
    use std::time::Duration;

    use arc_swap::ArcSwap;
    #[cfg(feature = "render")]
    use crossbeam_channel::Sender;
    use tokio::io::{AsyncReadExt, AsyncWriteExt};
    use tokio::net::tcp::{OwnedReadHalf, OwnedWriteHalf};
    use tokio::net::{TcpListener, TcpStream};
    use tokio::sync::Semaphore;
    #[cfg(feature = "render")]
    use vd_client_harness::capture::{
        at_tick_predicate, capture_entry, plan_record, state_rel_for,
    };
    #[cfg(feature = "render")]
    use vd_client_harness::manifest::{CaptureKind, MANIFEST_FILENAME, RunManifest};
    // The closed-loop nav math (walk-to/look-at) — pure, 100% Tier-A covered in the
    // renderer-free vd-client-harness; the drivers below are branchless shims over it.
    use vd_client_harness::nav;
    use vd_core::glam::{DQuat, DVec3};
    use vd_devproto::{
        DevError, DevPhase, DevRequest, DevResponse, DevState, InputAction, WaitPredicate,
        decode_request, encode_response,
    };

    /// The run manifest behind one writer (the dev-control listener), shared across
    /// connections. One `Mutex` serializes appends + the `manifest.json` rewrite, so two
    /// concurrent captures can never lose an entry or corrupt the file.
    #[cfg(feature = "render")]
    pub type SharedManifest = Arc<std::sync::Mutex<RunManifest>>;

    /// The reply timeout for a capture (Capture mode): the render thread writes a PNG and
    /// replies; if it does not within this, the request fails (never hangs the connection).
    #[cfg(feature = "render")]
    const CAPTURE_REPLY_TIMEOUT: Duration = Duration::from_secs(5);

    /// Record-sequence safety bounds: a fat-fingered `--secs 1e9` or `--fps 100000` is
    /// clamped, never allowed to run unbounded. 3600 frames = 60 s at 60 fps — ample for a
    /// transition clip; the agent can issue another `record` to continue.
    #[cfg(feature = "render")]
    const MAX_RECORD_FRAMES: u64 = 3600;
    #[cfg(feature = "render")]
    const MAX_RECORD_FPS: u32 = 120;

    /// The `screenshot --at-tick` wait budget (step-ticks) before giving up on the tick.
    #[cfg(feature = "render")]
    const SCREENSHOT_WAIT_TICKS: u64 = 600;

    /// The `wait-until` re-evaluation interval — short relative to a step tick so a
    /// satisfied predicate is reported promptly. The wait is ALSO cancelled the instant
    /// the socket sees activity (EOF), so a killed `vdctl` never strands a task.
    const WAIT_POLL: Duration = Duration::from_millis(5);

    /// The cap on concurrent dev-control connections — bounds listener memory; past it
    /// new connections are refused (the agent retries), never queued unbounded.
    const MAX_DEV_CONNECTIONS: usize = 8;

    /// Backoff after a failed `accept()` so a persistent error (e.g. fd exhaustion,
    /// where the bad connection is NOT dequeued) cannot busy-spin a core + flood logs.
    const ACCEPT_RETRY_BACKOFF: Duration = Duration::from_millis(50);

    /// The hard cap on ONE request line. A `DevRequest` is tiny; this only exists so a
    /// peer that never sends a newline cannot grow the reader's buffer without bound
    /// (the one unbounded TCP ingress the mesh path's frame caps don't cover).
    const MAX_REQUEST_LINE: usize = 64 * 1024;
    const READ_CHUNK: usize = 8 * 1024;

    /// A bounded, cancel-safe line framer over a connection's read half. It owns the
    /// accumulation buffer, so a read cancelled by `tokio::select!` loses nothing (the
    /// bytes stay buffered), and it rejects a line past [`MAX_REQUEST_LINE`] instead of
    /// growing forever. Yields one `\n`-terminated line at a time.
    struct LineFramer {
        reader: OwnedReadHalf,
        buf: Vec<u8>,
    }

    impl LineFramer {
        fn new(reader: OwnedReadHalf) -> LineFramer {
            LineFramer {
                reader,
                buf: Vec::with_capacity(READ_CHUNK),
            }
        }

        /// The next complete line, or `None` at EOF. Reads only as needed; buffered
        /// bytes past a line (e.g. a pipelined request) are retained for the next call.
        async fn next_line(&mut self) -> std::io::Result<Option<String>> {
            loop {
                if let Some(line) = self.take_buffered_line() {
                    return Ok(Some(line));
                }
                if self.fill().await? {
                    return Ok(None); // EOF (a trailing partial line is dropped)
                }
            }
        }

        /// Split one `\n`-terminated line out of the buffer, if present (trims `\r\n`).
        fn take_buffered_line(&mut self) -> Option<String> {
            let nl = self.buf.iter().position(|&b| b == b'\n')?;
            let mut line: Vec<u8> = self.buf.drain(..=nl).collect();
            line.pop(); // the '\n'
            if line.last() == Some(&b'\r') {
                line.pop();
            }
            Some(String::from_utf8_lossy(&line).into_owned())
        }

        /// Read one chunk into the buffer. `Ok(true)` at EOF; `Ok(false)` on data.
        /// Cancel-safe: any bytes read are retained in `self.buf`.
        async fn fill(&mut self) -> std::io::Result<bool> {
            if self.buf.len() >= MAX_REQUEST_LINE {
                return Err(std::io::Error::new(
                    std::io::ErrorKind::InvalidData,
                    "dev-control request line exceeded the size cap",
                ));
            }
            self.buf.reserve(READ_CHUNK);
            Ok(self.reader.read_buf(&mut self.buf).await? == 0)
        }
    }

    /// The lock-free bridge to the step thread (cloned per connection).
    #[derive(Clone)]
    pub struct Handles {
        pub commands: SyncSender<InputAction>,
        pub published: Arc<ArcSwap<DevState>>,
        pub step_seq: Arc<AtomicU64>,
        pub dropped: Arc<AtomicU64>,
        pub allow_mutating: bool,
        /// The capture seam (Capture mode only): the screenshot/record handlers send jobs to
        /// the Bevy render thread. `None` in windowed/headless modes (then `Screenshot`/
        /// `Record` are `Unsupported`).
        #[cfg(feature = "render")]
        pub captures: Option<Sender<vd_client_render::CaptureJob>>,
        /// The capture run dir (`runs/<ts>__<name>/`) — where `manifest.json` + `state/`
        /// dumps land. `None` outside Capture mode.
        #[cfg(feature = "render")]
        pub runs_dir: Option<std::path::PathBuf>,
        /// The shared run manifest (single-writer). `None` outside Capture mode.
        #[cfg(feature = "render")]
        pub manifest: Option<SharedManifest>,
    }

    /// Accept loopback connections forever, one task per connection, bounded by a
    /// permit semaphore.
    pub async fn serve(addr: SocketAddr, handles: Handles) {
        let listener = match TcpListener::bind(addr).await {
            Ok(listener) => listener,
            Err(e) => {
                tracing::error!("dev-control bind {addr} failed: {e}");
                return;
            }
        };
        let limiter = Arc::new(Semaphore::new(MAX_DEV_CONNECTIONS));
        loop {
            match listener.accept().await {
                Ok((stream, _peer)) => {
                    let Ok(permit) = limiter.clone().try_acquire_owned() else {
                        tracing::warn!(
                            "dev-control at the connection cap ({MAX_DEV_CONNECTIONS}); refusing"
                        );
                        drop(stream); // close immediately; the agent retries
                        continue;
                    };
                    let handles = handles.clone();
                    tokio::spawn(async move {
                        let _permit = permit; // frees the slot when the connection ends
                        if let Err(e) = serve_conn(stream, handles).await {
                            tracing::debug!("dev-control connection ended: {e}");
                        }
                    });
                }
                Err(e) => {
                    tracing::warn!("dev-control accept failed: {e}");
                    tokio::time::sleep(ACCEPT_RETRY_BACKOFF).await;
                }
            }
        }
    }

    /// One connection: a request per line, a response per line, until EOF. A
    /// `wait-until` blocks here but is cancelled by socket EOF (a buffered pipelined
    /// line survives the wait and is served on the next iteration).
    async fn serve_conn(stream: TcpStream, handles: Handles) -> std::io::Result<()> {
        let (read_half, mut write_half) = stream.into_split();
        let mut framer = LineFramer::new(read_half);
        while let Some(line) = framer.next_line().await? {
            if line.trim().is_empty() {
                continue;
            }
            let response = match decode_request(&line) {
                Ok(DevRequest::WaitUntil {
                    predicate,
                    max_ticks,
                }) => match wait_until(&handles, &mut framer, predicate, max_ticks).await {
                    Some(response) => response,
                    None => return Ok(()), // socket closed mid-wait
                },
                // Closed loops: driven tick-by-tick here (like WaitUntil) — serve_conn owns
                // the read half the drive must cancel on (socket EOF).
                Ok(DevRequest::WalkTo {
                    target,
                    arrive_epsilon,
                    max_ticks,
                    max_step_m,
                }) => match drive_walk_to(
                    &handles,
                    &mut framer,
                    target,
                    arrive_epsilon,
                    max_ticks,
                    max_step_m,
                )
                .await
                {
                    Some(response) => response,
                    None => return Ok(()), // socket closed mid-drive
                },
                Ok(DevRequest::LookAt {
                    target,
                    align_epsilon,
                    max_ticks,
                }) => match drive_look_at(&handles, &mut framer, target, align_epsilon, max_ticks)
                    .await
                {
                    Some(response) => response,
                    None => return Ok(()), // socket closed mid-drive
                },
                // Capture mode only (a wired `captures` channel): handled here (not in
                // dispatch_immediate) because the optional `--at-tick` wait is async +
                // cancellable on EOF, like WaitUntil. Otherwise → Unsupported below.
                #[cfg(feature = "render")]
                Ok(DevRequest::Screenshot { at_tick, label }) if handles.captures.is_some() => {
                    match screenshot(&handles, &mut framer, at_tick, label).await {
                        Some(response) => response,
                        None => return Ok(()), // socket closed mid-wait
                    }
                }
                // Record (Capture mode): a frame SEQUENCE — paced here (off the reactor) and
                // cancellable on EOF, like Screenshot/WaitUntil; handled in serve_conn for
                // the same reason. Otherwise → Unsupported in dispatch_immediate.
                #[cfg(feature = "render")]
                Ok(DevRequest::Record { fps, secs, label }) if handles.captures.is_some() => {
                    match record(&handles, &mut framer, fps, secs, label).await {
                        Some(response) => response,
                        None => return Ok(()), // socket closed mid-record
                    }
                }
                Ok(request) => dispatch_immediate(&handles, request),
                Err(error) => DevResponse::Error { error },
            };
            write_line(&mut write_half, &response).await?;
        }
        Ok(())
    }

    async fn write_line(
        write_half: &mut OwnedWriteHalf,
        response: &DevResponse,
    ) -> std::io::Result<()> {
        let mut encoded = encode_response(response);
        encoded.push('\n');
        write_half.write_all(encoded.as_bytes()).await
    }

    /// Route a non-blocking request: `State` reads the published snapshot, everything
    /// else is an input command through the gated mailbox. (`WaitUntil` is handled in
    /// `serve_conn`, which owns the read half the wait must cancel on.)
    fn dispatch_immediate(handles: &Handles, request: DevRequest) -> DevResponse {
        match request {
            DevRequest::State => DevResponse::State {
                state: current(handles),
            },
            // Unreachable: serve_conn intercepts WaitUntil + the WalkTo/LookAt closed loops
            // (all owe the read half). Defensive, not a panic — never dishonest `Unsupported`
            // (they ARE supported, just driven in serve_conn).
            DevRequest::WaitUntil { .. }
            | DevRequest::WalkTo { .. }
            | DevRequest::LookAt { .. } => DevResponse::Error {
                error: DevError::BadRequest,
            },
            // Decodable but not wired in this build (capture needs `--features render` +
            // Capture mode): an HONEST `Unsupported`, NOT `BadRequest` (reserved for
            // malformed input). Reached when `captures` is None (headless/windowed).
            DevRequest::Screenshot { .. } | DevRequest::Record { .. } => DevResponse::Error {
                error: DevError::Unsupported,
            },
            input => apply(handles, input),
        }
    }

    fn current(handles: &Handles) -> DevState {
        handles.published.load().as_ref().clone()
    }

    /// Enqueue an input command (Move/Look/Action/Close/ResetInput), gating the
    /// mutating arm behind `--allow-dev-control` and shedding loudly when full.
    fn apply(handles: &Handles, input: DevRequest) -> DevResponse {
        if input.is_mutating() && !handles.allow_mutating {
            return DevResponse::Error {
                error: DevError::NotAllowed,
            };
        }
        match input.as_input_action() {
            Some(action) => match handles.commands.try_send(action) {
                Ok(()) => DevResponse::Ack,
                Err(_) => {
                    handles.dropped.fetch_add(1, Ordering::Relaxed);
                    DevResponse::Error {
                        error: DevError::Busy,
                    }
                }
            },
            // State/WaitUntil are routed in `dispatch`; any other non-input request
            // is a protocol error rather than a panic.
            None => DevResponse::Error {
                error: DevError::BadRequest,
            },
        }
    }

    /// Block until the predicate holds over the published state, `max_ticks`
    /// step-ticks elapse, the session closes, or the socket reaches EOF (cancel →
    /// `None`). The `Closed` check is the termination guarantee: once the step loop
    /// has exited, the step clock is FROZEN, so a `max_ticks` deadline would never
    /// trip — a Closed session reports an honest `Timeout` (carrying `phase: closed`)
    /// instead of hanging.
    async fn wait_until(
        handles: &Handles,
        framer: &mut LineFramer,
        predicate: WaitPredicate,
        max_ticks: u64,
    ) -> Option<DevResponse> {
        let start = handles.step_seq.load(Ordering::Relaxed);
        loop {
            let state = handles.published.load_full();
            if predicate.eval(state.as_ref()) {
                return Some(DevResponse::State {
                    state: state.as_ref().clone(),
                });
            }
            let elapsed = handles.step_seq.load(Ordering::Relaxed).wrapping_sub(start);
            if state.phase == DevPhase::Closed || elapsed >= max_ticks {
                return Some(DevResponse::Timeout {
                    state: state.as_ref().clone(),
                });
            }
            tokio::select! {
                _ = tokio::time::sleep(WAIT_POLL) => {}
                // EOF from a killed vdctl cancels the wait (frees the task at once);
                // a buffered pipelined line is retained by the framer for after the
                // wait responds, so it is never silently dropped.
                read = framer.fill() => {
                    if matches!(read, Ok(true) | Err(_)) {
                        return None;
                    }
                }
            }
        }
    }

    /// The OWN entity's delivered (lagged — NO prediction) world pose + facing, from the
    /// published `DevState`: the row whose id matches `own_entity`. `None` until a snapshot
    /// has anchored the own row (the drive loops keep polling until it appears). A branchless
    /// shim over the already-composited state; the closed-loop math it feeds is 100% Tier-A
    /// covered in `vd-client-harness::nav`.
    fn own_pose(state: &DevState) -> Option<(DVec3, DQuat)> {
        let own = state.own_entity.as_deref()?;
        state
            .entities
            .iter()
            .find(|row| row.entity == own)
            .map(|row| {
                let o = row.orient;
                (
                    DVec3::from_array(row.pos),
                    DQuat::from_xyzw(o[0], o[1], o[2], o[3]),
                )
            })
    }

    /// Push one closed-loop input (Move/Look) straight to the step mailbox — bypassing
    /// `apply`'s mutating gate (these decompose into ordinary NON-privileged input) — and
    /// counting a shed LOUD (never silent) if the bounded mailbox is full. The loop re-sends
    /// next tick, so a shed is harmless (Move is a held axis; a dropped Look just slows the
    /// turn), but it is always surfaced in `dev_commands_dropped`.
    fn push_action(handles: &Handles, action: InputAction) {
        if handles.commands.try_send(action).is_err() {
            handles.dropped.fetch_add(1, Ordering::Relaxed);
        }
    }

    /// One closed-loop tick's outcome: the error-reducing input, and whether converged.
    struct LoopStep {
        action: InputAction,
        done: bool,
    }

    /// The shared closed-loop driver for WalkTo/LookAt. Each ASSEMBLED sim step (gated on
    /// `step_seq` advancing — the load-bearing cadence fix: the loop polls every `WAIT_POLL`
    /// = 5 ms but the sim assembles input at the tick rate, and `Look` ACCUMULATES via
    /// `add_look`, so emitting per-poll would sum ~10 clamped deltas into ONE datagram = ~10×
    /// the intended turn → overshoot/oscillation; gating to one emit per step lands exactly
    /// one clamped delta per datagram, matching `nav`'s per-tick convergence model), it polls
    /// the DELIVERED (lagged) own pose, asks `step` for the error-reducing input, emits it
    /// (shed-loud), and returns `State` on convergence / `Timeout` on `max_ticks` or a Closed
    /// session / `None` on socket EOF. NO client prediction — the server stays sole authority;
    /// this only injects ordinary Move/Look toward the goal on the delivered state.
    async fn drive_closed_loop(
        handles: &Handles,
        framer: &mut LineFramer,
        max_ticks: u64,
        mut step: impl FnMut(DVec3, DQuat) -> LoopStep,
    ) -> Option<DevResponse> {
        let start = handles.step_seq.load(Ordering::Relaxed);
        let mut last_sent = start.wrapping_sub(1); // force an emit on the first observed step
        loop {
            let state = handles.published.load_full();
            let seq = handles.step_seq.load(Ordering::Relaxed);
            if seq != last_sent {
                last_sent = seq;
                if let Some((pos, orient)) = own_pose(state.as_ref()) {
                    let step = step(pos, orient);
                    if step.done {
                        return Some(DevResponse::State {
                            state: state.as_ref().clone(),
                        });
                    }
                    push_action(handles, step.action);
                }
            }
            if state.phase == DevPhase::Closed || seq.wrapping_sub(start) >= max_ticks {
                return Some(DevResponse::Timeout {
                    state: state.as_ref().clone(),
                });
            }
            tokio::select! {
                _ = tokio::time::sleep(WAIT_POLL) => {}
                read = framer.fill() => {
                    if matches!(read, Ok(true) | Err(_)) {
                        return None; // socket EOF: cancel the drive
                    }
                }
            }
        }
    }

    /// Closed loop: steer the own entity toward WORLD `target` with ordinary Move until
    /// within `arrive_epsilon` (→ `State`) or the budget/Closed (→ `Timeout`). `arrive_epsilon`
    /// MUST exceed one sim step or the fixed-magnitude Move overshoots and never settles — the
    /// contract `nav::walk_to` pins; vdctl defaults it well above one step.
    async fn drive_walk_to(
        handles: &Handles,
        framer: &mut LineFramer,
        target: [f64; 3],
        arrive_epsilon: f64,
        max_ticks: u64,
        max_step_m: f64,
    ) -> Option<DevResponse> {
        let target = DVec3::from_array(target);
        drive_closed_loop(handles, framer, max_ticks, move |pos, orient| {
            let step = nav::walk_to(pos, orient, target, arrive_epsilon, max_step_m);
            LoopStep {
                done: step.arrived,
                action: step.action(),
            }
        })
        .await
    }

    /// Closed loop: turn the own entity to face WORLD `target` with ordinary Look until within
    /// `align_epsilon` RADIANS (the true 3-D angle, honest at the pole) or the budget/Closed.
    async fn drive_look_at(
        handles: &Handles,
        framer: &mut LineFramer,
        target: [f64; 3],
        align_epsilon: f64,
        max_ticks: u64,
    ) -> Option<DevResponse> {
        let target = DVec3::from_array(target);
        drive_closed_loop(handles, framer, max_ticks, move |pos, orient| {
            let step = nav::look_at(pos, orient, target, align_epsilon);
            LoopStep {
                done: step.aligned,
                action: step.action(),
            }
        })
        .await
    }

    /// Serve a `Screenshot` (Capture mode): optionally wait for the universe tick to reach
    /// `at_tick` (reusing the wait-until machinery — cancellable on EOF), capture one frame,
    /// record it in the run manifest (PNG path + aligned state + tick), and reply with the
    /// PNG path. Non-mutating (a read). Returns `None` only on socket EOF (cancel).
    #[cfg(feature = "render")]
    async fn screenshot(
        handles: &Handles,
        framer: &mut LineFramer,
        at_tick: Option<u64>,
        label: Option<String>,
    ) -> Option<DevResponse> {
        if handles.captures.is_none() {
            return Some(DevResponse::Error {
                error: DevError::Unsupported,
            });
        }
        // --at-tick: block until the delivered universe tick reaches T (run-stable), via
        // the SAME wait machinery + the SAME Tier-A predicate the harness uses (no second
        // copy); a Timeout/closed/EOF there propagates straight back.
        if let Some(tick) = at_tick {
            match wait_until(
                handles,
                framer,
                at_tick_predicate(tick),
                SCREENSHOT_WAIT_TICKS,
            )
            .await
            {
                Some(DevResponse::State { .. }) => {} // tick reached → capture
                other => return other,                // Timeout / closed / EOF (None)
            }
        }
        match capture_one(handles, CaptureKind::Screenshot, label).await {
            Some(Ok(result)) => {
                // The render-sampled tick (the captured frame's), not a post-roundtrip poll.
                let tick = result.freshest_tick;
                let path = result.path.clone();
                persist_capture(handles, CaptureKind::Screenshot, result, at_tick, true).await;
                Some(DevResponse::Captured { path, tick })
            }
            // Render thread gone / timed out: shed honestly (it logs the real cause); retry.
            _ => Some(DevResponse::Error {
                error: DevError::Busy,
            }),
        }
    }

    /// Serve a `Record` (Capture mode): capture `round(fps*secs)` frames spaced by `1/fps`,
    /// each a `Frame` written to `frames/<base>-NNNN.png` + recorded in the manifest, then
    /// flush `manifest.json` ONCE. The cadence is driven HERE (off the reactor) by REUSING
    /// the one-frame capture path per frame — so there is NO record-mode state in the render
    /// thread (DRY). Cancellable on socket EOF; the frame count is clamped so a fat-fingered
    /// `--secs 1e9` can't run forever. Reply: `Recorded { dir, frames_written }`.
    #[cfg(feature = "render")]
    async fn record(
        handles: &Handles,
        framer: &mut LineFramer,
        fps: u32,
        secs: f64,
        label: Option<String>,
    ) -> Option<DevResponse> {
        if handles.captures.is_none() {
            return Some(DevResponse::Error {
                error: DevError::Unsupported,
            });
        }
        // The clamped frame count + interval are the Tier-A `plan_record` (the cadence math
        // is tested there, not in this bin shell); `None` rejects a bad duration loudly.
        let Some(plan) = plan_record(fps, secs, MAX_RECORD_FPS, MAX_RECORD_FRAMES) else {
            return Some(DevResponse::Error {
                error: DevError::BadRequest,
            });
        };
        let interval = Duration::from_secs_f64(plan.interval_secs);
        let base = label.unwrap_or_else(|| "rec".to_owned());
        let mut written = 0u64;
        for i in 0..plan.frames {
            match capture_one(handles, CaptureKind::Frame, Some(format!("{base}-{i:04}"))).await {
                Some(Ok(result)) => {
                    persist_capture(handles, CaptureKind::Frame, result, None, false).await;
                    written += 1;
                }
                // Render thread gone / timed out: stop, return what we captured so far.
                _ => break,
            }
            // Pace to the next frame, cancelling the instant a killed vdctl closes the socket
            // (EOF) so a long recording never strands the task.
            if i + 1 < plan.frames {
                tokio::select! {
                    _ = tokio::time::sleep(interval) => {}
                    read = framer.fill() => {
                        if matches!(read, Ok(true) | Err(_)) {
                            // The driver vanished mid-record (a killed vdctl — routine in
                            // agent loops). The frames captured SO FAR are on disk and in
                            // the in-memory manifest: flush it so manifest.json reflects
                            // every artifact that exists (never under-reports a run).
                            flush_manifest_blocking(handles).await;
                            return None;
                        }
                    }
                }
            }
        }
        flush_manifest_blocking(handles).await;
        let path = handles
            .runs_dir
            .as_ref()
            .map(|d| d.display().to_string())
            .unwrap_or_default();
        Some(DevResponse::Recorded {
            path,
            frames: written,
        })
    }

    /// Send ONE capture job to the render thread and await its reply off the reactor (a
    /// bounded blocking recv via `spawn_blocking`, so it never stalls the runtime or hangs).
    /// `None` ⇒ no capture channel (not Capture mode); `Some(Err)` ⇒ render thread gone or
    /// timed out; `Some(Ok)` ⇒ the written PNG. The ONE place a `CaptureJob` is sent —
    /// shared by `screenshot` and `record`.
    #[cfg(feature = "render")]
    async fn capture_one(
        handles: &Handles,
        kind: CaptureKind,
        label: Option<String>,
    ) -> Option<Result<vd_client_render::CaptureResult, ()>> {
        let captures = handles.captures.as_ref()?;
        let (reply_tx, reply_rx) = crossbeam_channel::bounded(1);
        if captures
            .send(vd_client_render::CaptureJob {
                kind,
                label,
                reply: reply_tx,
            })
            .is_err()
        {
            return Some(Err(())); // the render thread/app exited
        }
        match tokio::task::spawn_blocking(move || reply_rx.recv_timeout(CAPTURE_REPLY_TIMEOUT))
            .await
        {
            Ok(Ok(Ok(result))) => Some(Ok(result)),
            _ => Some(Err(())),
        }
    }

    /// Persist one served capture OFF the reactor (these are filesystem writes — the same
    /// never-block-the-listener discipline as the blocking capture reply): the state dump +
    /// the manifest entry, plus the `manifest.json` flush when `flush` (a screenshot flushes
    /// per capture; a record sequence flushes once at the end + on cancel).
    #[cfg(feature = "render")]
    async fn persist_capture(
        handles: &Handles,
        kind: CaptureKind,
        result: vd_client_render::CaptureResult,
        at_tick: Option<u64>,
        flush: bool,
    ) {
        let handles = handles.clone();
        let _ = tokio::task::spawn_blocking(move || {
            record_capture(&handles, kind, &result, at_tick);
            if flush {
                flush_manifest(&handles);
            }
        })
        .await;
    }

    /// Flush the manifest OFF the reactor (a filesystem write).
    #[cfg(feature = "render")]
    async fn flush_manifest_blocking(handles: &Handles) {
        let handles = handles.clone();
        let _ = tokio::task::spawn_blocking(move || flush_manifest(&handles)).await;
    }

    /// Record one served capture into the run: write its aligned `state/<stem>.json` dump
    /// (the decoded delivered DevState — wire truth at capture time) and append a
    /// `CaptureEntry` to the in-memory manifest via the Tier-A `capture_entry` alignment.
    /// Best-effort — a write failure is logged, never fatal (the PNG already exists). The
    /// `manifest.json` file is flushed separately so a record sequence rewrites it ONCE.
    /// Synchronous fs I/O — call via [`persist_capture`] (spawn_blocking), never directly
    /// on the reactor.
    #[cfg(feature = "render")]
    fn record_capture(
        handles: &Handles,
        kind: CaptureKind,
        result: &vd_client_render::CaptureResult,
        at_tick: Option<u64>,
    ) {
        let (Some(runs_dir), Some(manifest)) = (&handles.runs_dir, &handles.manifest) else {
            return;
        };
        // NOTE: `current` is a post-roundtrip poll, so the dumped DevState's `universe_tick`
        // may LEAD the render-sampled `freshest_tick` below by the render round-trip — a
        // bounded, low-severity skew. The MANIFEST's `freshest_tick`/`cursor` (from the
        // captured frame) are the aligned quantities; the state dump is a best-effort
        // diagnostic snapshot of the delivered world around the capture.
        let state = current(handles); // for the session-diagnostic count + the dump
        // The state dump sits beside the PNG (frames/foo.png → state/foo.json) — the
        // run-relative pairing is the Tier-A `state_rel_for`.
        let state_rel = state_rel_for(&result.rel_path);
        let wrote_state = write_state_dump(&runs_dir.join(&state_rel), &state);
        // freshest_tick + cursor are RENDER-sampled (the captured frame's), so the manifest
        // identifies the captured world; snapshots_applied is the session diagnostic.
        let entry = capture_entry(
            kind,
            result.rel_path.clone(),
            at_tick,
            result.freshest_tick,
            result.cursor,
            state.snapshots_applied,
            wrote_state.then_some(state_rel),
        );
        if let Ok(mut m) = manifest.lock() {
            m.push(entry);
        }
    }

    /// Serialize the in-memory manifest to `runs_dir/manifest.json` (best-effort).
    /// Synchronous fs I/O — call via [`flush_manifest_blocking`]/[`persist_capture`].
    #[cfg(feature = "render")]
    fn flush_manifest(handles: &Handles) {
        let (Some(runs_dir), Some(manifest)) = (&handles.runs_dir, &handles.manifest) else {
            return;
        };
        if let Ok(m) = manifest.lock()
            && let Err(e) = std::fs::write(runs_dir.join(MANIFEST_FILENAME), m.to_json())
        {
            tracing::warn!(error = %e, "manifest write failed");
        }
    }

    /// Write the aligned delivered state beside a capture (pretty JSON). Returns whether it
    /// was written, so the manifest only references a `state_path` that actually exists.
    #[cfg(feature = "render")]
    fn write_state_dump(path: &std::path::Path, state: &DevState) -> bool {
        let Ok(json) = serde_json::to_string_pretty(state) else {
            return false;
        };
        if let Some(parent) = path.parent()
            && std::fs::create_dir_all(parent).is_err()
        {
            return false;
        }
        std::fs::write(path, json).is_ok()
    }
}
