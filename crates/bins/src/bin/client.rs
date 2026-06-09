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
use vd_bins::{GATEWAY, loopback};
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
    tracing_subscriber::fmt().with_env_filter("info").init();
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
            loopback(args.client_quic),
            peers,
            CLIENT_OUTBOUND_CAP,
        ),
    )?;
    let core = ClientCore::new(transport, GATEWAY, ticket, ClientInterpTuning::DEFAULT);

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
                // Wire the capture sender only in `--capture` mode; otherwise `Screenshot`
                // is `Unsupported` (windowed/headless have no offscreen readback).
                #[cfg(feature = "render")]
                captures: args.capture.then(|| capture_tx.clone()),
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
        let runs_dir = capture_run_dir(&args.name);
        tracing::info!(dir = %runs_dir.display(), "headless capture run");
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

/// The output dir for a capture run: `runs/<unix_secs>__<name>/` (created here); the
/// manifest + PNGs land inside. The timestamp keeps successive runs distinct.
#[cfg(all(feature = "dev-control", feature = "render"))]
fn capture_run_dir(name: &str) -> std::path::PathBuf {
    let secs = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);
    let safe: String = name
        .chars()
        .map(|c| {
            if c.is_ascii_alphanumeric() || c == '-' || c == '_' {
                c
            } else {
                '_'
            }
        })
        .collect();
    let dir = std::path::PathBuf::from("runs").join(format!("{secs}__{safe}"));
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
        published.store(Arc::new(core.state().devstate(
            now_s,
            applied,
            dropped.load(Ordering::Relaxed),
        )));
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
}

fn parse_args() -> Result<ClientArgs, String> {
    let mut name = "client".to_owned();
    let mut gateway: Option<SocketAddr> = None;
    let mut client_quic: Option<u16> = None;
    let mut trust_dir: Option<String> = None;
    let mut dev_control: Option<u16> = None;
    let mut agent_index: u64 = 0;
    let mut allow_dev_control = false;
    let mut step_hz: u32 = DEFAULT_STEP_HZ;
    let mut window = false;
    let mut capture = false;

    let mut it = std::env::args().skip(1);
    while let Some(arg) = it.next() {
        match arg.as_str() {
            "--name" => name = next_val(&mut it, "--name")?,
            "--gateway" => gateway = Some(parse_val(&mut it, "--gateway")?),
            "--client-quic" => client_quic = Some(parse_val(&mut it, "--client-quic")?),
            "--trust-dir" => trust_dir = Some(next_val(&mut it, "--trust-dir")?),
            "--dev-control" => dev_control = Some(parse_val(&mut it, "--dev-control")?),
            "--agent-index" => agent_index = parse_val(&mut it, "--agent-index")?,
            "--allow-dev-control" => allow_dev_control = true,
            "--step-hz" => step_hz = parse_val(&mut it, "--step-hz")?,
            "--window" => window = true,
            "--capture" => capture = true,
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
        trust_dir: trust_dir.ok_or("missing --trust-dir <dir>")?,
        dev_control,
        agent_index,
        allow_dev_control,
        step_hz,
        window,
        capture,
    })
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
    use vd_devproto::{
        DevError, DevPhase, DevRequest, DevResponse, DevState, InputAction, WaitPredicate,
        decode_request, encode_response,
    };
    #[cfg(feature = "render")]
    use vd_devproto::{WaitField, WaitOp};

    /// The reply timeout for a capture (Capture mode): the render thread writes a PNG and
    /// replies; if it does not within this, the request fails (never hangs the connection).
    #[cfg(feature = "render")]
    const CAPTURE_REPLY_TIMEOUT: Duration = Duration::from_secs(5);

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
        /// The capture seam (Capture mode only): the screenshot handler sends a job to the
        /// Bevy render thread. `None` in windowed/headless modes (then `Screenshot` is
        /// `Unsupported`).
        #[cfg(feature = "render")]
        pub captures: Option<Sender<vd_client_render::CaptureJob>>,
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
            // Unreachable: serve_conn intercepts WaitUntil. Defensive, not a panic.
            DevRequest::WaitUntil { .. } => DevResponse::Error {
                error: DevError::BadRequest,
            },
            // Decodable but not yet wired (capture + closed loops land at T4/T5): an
            // HONEST `Unsupported`, NOT `BadRequest` (which stays reserved for malformed
            // input). Routed explicitly so the honesty surface is correct before the
            // feature lands.
            DevRequest::Screenshot { .. }
            | DevRequest::Record { .. }
            | DevRequest::WalkTo { .. }
            | DevRequest::LookAt { .. } => DevResponse::Error {
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

    /// Serve a `Screenshot` (Capture mode): optionally wait for the universe tick to reach
    /// `at_tick` (reusing the wait-until machinery — cancellable on EOF), then ask the
    /// render thread to write a PNG and reply with its path. Non-mutating (a read). Returns
    /// `None` only on socket EOF (cancel).
    #[cfg(feature = "render")]
    async fn screenshot(
        handles: &Handles,
        framer: &mut LineFramer,
        at_tick: Option<u64>,
        label: Option<String>,
    ) -> Option<DevResponse> {
        let Some(captures) = &handles.captures else {
            return Some(DevResponse::Error {
                error: DevError::Unsupported,
            });
        };
        // --at-tick: block until the delivered universe tick reaches T (run-stable), via
        // the SAME wait machinery; a Timeout/closed/EOF there propagates straight back.
        if let Some(tick) = at_tick {
            let predicate = WaitPredicate {
                field: WaitField::UniverseTick,
                op: WaitOp::Ge,
                value: tick,
            };
            match wait_until(handles, framer, predicate, SCREENSHOT_WAIT_TICKS).await {
                Some(DevResponse::State { .. }) => {} // tick reached → capture
                other => return other,                // Timeout / closed / EOF (None)
            }
        }
        // Hand the render thread a one-shot reply channel; it writes the PNG and replies.
        let (reply_tx, reply_rx) = crossbeam_channel::bounded(1);
        if captures
            .send(vd_client_render::CaptureJob {
                label,
                reply: reply_tx,
            })
            .is_err()
        {
            // The render thread/app is gone (exited): not a malformed request.
            return Some(DevResponse::Error {
                error: DevError::Unsupported,
            });
        }
        let tick = handles.published.load().universe_tick;
        // The reply is a BLOCKING crossbeam recv → run it off the async runtime via
        // spawn_blocking so it never stalls the reactor; bounded so it can never hang.
        let received =
            tokio::task::spawn_blocking(move || reply_rx.recv_timeout(CAPTURE_REPLY_TIMEOUT)).await;
        match received {
            Ok(Ok(Ok(result))) => Some(DevResponse::Captured {
                path: result.path,
                tick,
            }),
            // Timeout / render-side error / join error: shed honestly (the render side logs
            // the real cause); the agent retries.
            _ => Some(DevResponse::Error {
                error: DevError::Busy,
            }),
        }
    }
}
