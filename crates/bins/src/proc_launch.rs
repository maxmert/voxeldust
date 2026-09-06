//! `ProcLaunchBackend` (RLM Step 5c-2b) — the REAL-process [`LaunchBackend`], the "hands" that fork a
//! `vd-shard` behind the 100%-covered decision kernel ([`vd_node::rlm_spawn::SpawnCore`]).
//!
//! This is Tier-B (§1.1 of `scripts/rlm_step5_real_spawner_spec.md`): it carries NO policy — every method
//! is a syscall / mesh push the kernel drives — so it is proven by the process gate
//! (`crates/bins/tests/rlm_proc_spawn_smoke.rs`), never by coverage. All the branching worth covering lives
//! in the kernel; this module is straight-line effect.
//!
//! # Vet-corrected shape (workflow `wf_12f77998`)
//!
//! - **Owned-path ONLY (D4).** 5c holds an OS [`Child`] handle per launched node and answers liveness with
//!   [`Child::try_wait`] (a real non-blocking `waitpid`). The ADOPTED-orphan path (a survivor with no
//!   `Child` after an orchestrator restart, its liveness a cookie-probe over `/whoami`) is slice **5e** —
//!   it is where the kill-9-then-rehydrate process gate actually constructs orphans.
//! - **Reap on teardown (D2).** [`teardown`](ProcLaunchBackend::teardown) moves the `Child` into a DETACHED
//!   thread that SIGTERMs the group → polls `try_wait` to the grace deadline → SIGKILLs → **`wait()`s to
//!   REAP**. Off the reconcile thread, so `kill_realm` never blocks on the grace window; the final `wait`
//!   is what prevents a `<defunct>` zombie per spin-down (the self-inflicted `EAGAIN` DoS on RLM's up/DOWN
//!   churn).
//! - **Cookie minted here, PRE-FORK (D3).** [`mint_cookie`](ProcLaunchBackend::mint_cookie) draws std
//!   entropy (wall-clock nanos + a per-backend sequence) — NO `rand`/`nix` dep (OQ-3). The kernel persists
//!   it in the write-ahead intent before the fork; the child echoes it on `/whoami` for the 5e guard.
//! - **`book_peer` = the real [`MeshControl::update_peer_addr`] (the layering fix).** The mesh handle lives
//!   HERE (vd-bins), never in the vd-node kernel.
//!
//! The launch/kill primitives are the SHARED [`crate::spawn_node_grouped`]/[`crate::signal_group`]/
//! [`crate::pid_alive`]/[`crate::sibling_binary`] the dev-cluster launcher also uses (the 5c DRY lift).

use std::collections::BTreeMap;
use std::net::SocketAddr;
use std::path::PathBuf;
use std::process::Child;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex, MutexGuard, PoisonError};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use vd_core::NodeId;
use vd_core::incarnation::IncarnationCookie;
use vd_io_prod::mesh::MeshControl;
use vd_node::rlm_spawn::{LaunchBackend, LaunchSpec};

/// Teardown poll cadence between SIGTERM and the SIGKILL escalation (the detached-thread grace loop).
const TEARDOWN_POLL: Duration = Duration::from_millis(50);

/// How long a freshly forked realm shard is watched for an IMMEDIATE death before its launch is called a
/// success. Sized to comfortably cover a bind failure (which surfaces in low single-digit milliseconds — the
/// child fails before it does any real work) while never gating a healthy boot, which stays alive and simply
/// runs out the window. Not a boot timeout: a shard that is still coming up after this is a normal launch.
/// `pub` because it is a per-launch SERIALIZATION on the reconcile thread: a wake that spins N
/// children up in one sweep pays (N−1) of these before the last child even forks, so the pixel
/// gates' derived wake budgets state it as their own term (look_horizon.md slice 5).
pub const EARLY_DEATH_WINDOW: Duration = Duration::from_millis(250);
/// Poll cadence inside [`EARLY_DEATH_WINDOW`].
const EARLY_DEATH_POLL: Duration = Duration::from_millis(10);

/// The operational parameters of the real spawner — the ONE config struct (no inline magic numbers), like
/// the kernel's `SpawnTuning`. The orchestrator bin fills it at boot (5c-3); the process gate fills it for
/// the smoke test.
pub struct ProcSpawnTuning {
    /// The ONE shard executable (HR3) resolved next to this process — always `"vd-shard"`.
    pub exe: &'static str,
    /// The directory each realm's `realm-<node>.log` is written under (per-node stdout/stderr redirect).
    pub workdir: PathBuf,
    /// The bound the teardown thread waits between SIGTERM and the SIGKILL escalation.
    pub drain_grace: Duration,
    /// The fixed env EVERY spawned shard inherits (the cluster-wide anchors: `VD_TRUST_DIR`, tick rate,
    /// universe seed/scale, `VD_ORCH`, …). The per-realm vars (`VD_NODE_ID`/`VD_BIND`/`VD_OWN_COORD`/…) are
    /// derived per launch from the [`LaunchSpec`] and merged OVER these.
    pub anchors: Vec<(&'static str, String)>,
    /// RLM 5e-4: the slow cadence between ADOPTED-orphan `/whoami` cookie-probes. `is_alive` returns the
    /// cached result within this window (O(1)-amortized), re-probing only once it elapses — so a probe
    /// STORM never hits the tick thread and a survivor's later death is still eventually observed (D7/D8).
    pub orphan_probe_interval: Duration,
    /// RLM 5e-4: the bounded read timeout on the orphan cookie-probe, so a wedged/dead survivor's probe
    /// cannot stall the reconcile sweep.
    pub probe_timeout: Duration,
}

/// One realm shard this backend tracks. `Owned` = launched by THIS orchestrator process (holds the OS
/// `Child` for `try_wait` liveness + the teardown reap). `Orphan` = a survivor ADOPTED on rehydrate after a
/// restart (RLM 5e-4): a DIFFERENT process launched it, so there is no `Child` handle — liveness is the
/// `/whoami` incarnation-cookie probe (the pid-reuse guard), keyed on the persisted `(pid, cookie, probe)`.
enum Slot {
    Owned {
        child: Child,
        pid: u32,
    },
    Orphan {
        pid: u32,
        cookie: IncarnationCookie,
        probe: SocketAddr,
        /// `(when, result)` of the last cookie-probe — the cadence cache (`orphan_probe_interval`). Seeded
        /// optimistic-ALIVE by `adopt` so the first post-rehydrate sweep does NOT probe (defers the burst).
        cache: Option<(Instant, bool)>,
    },
}

/// The real-process [`LaunchBackend`]: forks a `vd-shard` per realm and books it into the orchestrator's
/// mesh. Cheap to hold (state behind a `Mutex`); the reconciler drives it as `&dyn LaunchBackend` inside a
/// [`vd_node::rlm_spawn::SpawnCore`].
pub struct ProcLaunchBackend {
    /// The orchestrator's own mesh control surface — the real [`LaunchBackend::book_peer`] target.
    control: Arc<MeshControl>,
    tuning: ProcSpawnTuning,
    /// Tracked shards (owned children + adopted orphans), keyed by the id the kernel minted. `Mutex` (not
    /// the kernel's — this is the backend's private OS-handle table); deterministic `BTreeMap` order. Only
    /// the single sim/reconcile thread touches it, so `is_alive`'s bounded probe may hold the lock briefly.
    slots: Mutex<BTreeMap<NodeId, Slot>>,
    /// A per-backend monotone sequence mixed into each cookie so two spawns in the same nanosecond still
    /// differ.
    mint_seq: AtomicU64,
}

impl ProcLaunchBackend {
    /// Build the backend against the orchestrator's mesh `control` (for `book_peer`) and `tuning`.
    #[must_use]
    pub fn new(control: Arc<MeshControl>, tuning: ProcSpawnTuning) -> ProcLaunchBackend {
        ProcLaunchBackend {
            control,
            tuning,
            slots: Mutex::new(BTreeMap::new()),
            mint_seq: AtomicU64::new(0),
        }
    }

    fn lock(&self) -> MutexGuard<'_, BTreeMap<NodeId, Slot>> {
        self.slots.lock().unwrap_or_else(PoisonError::into_inner)
    }

    /// The per-realm env the child shard reads at boot, merged OVER [`ProcSpawnTuning::anchors`]. The
    /// profile is DERIVED by the child from `VD_OWN_COORD` (HR3 — 5a's `profile_for(own_coord.profile_kind)`);
    /// `VD_INCARNATION_COOKIE`/`VD_PROBE_ADDR` arm the `/whoami` identity echo.
    ///
    /// ★ `VD_REALM_KIND`/`VD_REALM_SEED` ARE A LOSSY COPY OF THE LEAF, and since 2026-09-01 the child
    /// reads the LEAF instead. Both are still sent, for a child that predates the change and for a
    /// hand-launched shard that states no lineage.
    ///
    /// They cannot disagree with the coord, because they are MADE from it — one line below. But the
    /// copy is narrower than the original: a realm name is up to 128 bits and that seed is 64, so a
    /// built realm's identity did not survive the round trip. The word list refused "ship" outright,
    /// which hid the narrowing behind a louder refusal.
    fn child_env(&self, spec: &LaunchSpec) -> Vec<(&'static str, String)> {
        let realm = spec.coord.lowered();
        // The child's outbox lives beside its log in the realm workdir; under a dev slot that is
        // `$TMPDIR`, which the outbox guard refuses unless the ORCHESTRATOR itself runs on the dev
        // store escape — the one fact that says "this cluster is a throwaway".
        let dev_escape = self
            .tuning
            .anchors
            .iter()
            .any(|(k, v)| (*k == "VD_STORE_EPHEMERAL_OK") & (v == "1"));
        let mut env = vec![
            ("VD_NODE_ID", spec.node.0.to_string()),
            ("VD_BIND", spec.addr.to_string()),
            ("VD_PROBE_ADDR", spec.probe.to_string()),
            ("VD_OWN_COORD", spec.coord.path().to_env_string()),
            ("VD_INCARNATION_COOKIE", spec.cookie.to_env_string()),
            ("VD_REALM_KIND", crate::realm_kind_token(realm).to_string()),
            ("VD_REALM_SEED", crate::realm_seed_of(realm).to_string()),
            // RLM 5d: the ancestor-closure peer book (∪ anchors) the kernel computed — the child dials its
            // parent chain up to root without DNS. `book` is the ONE VD_PEERS formatter (DRY, reused).
            ("VD_PEERS", crate::book(&spec.peers)),
            // THE DURABLE OUTBOX (slice 4): one file per spawned shard, beside its log in the realm
            // workdir — a planet's shard that dies between a despawn and its ack replays it at boot.
            // The guard's facts (durable root / dev escape) ride the spawn anchors.
            (
                "VD_OUTBOX_PATH",
                crate::node_outbox_path(&self.tuning.workdir, spec.node),
            ),
        ];
        if dev_escape {
            env.push(("VD_OUTBOX_EPHEMERAL_OK", "1".to_owned()));
        }
        env
    }
}

impl LaunchBackend for ProcLaunchBackend {
    fn mint_cookie(&self, node: NodeId) -> IncarnationCookie {
        // A per-launch NONCE (not a secret — the guard relies on the child holding its probe port). Wall
        // nanos differ across a restart; the sequence differs within a process; the node id spreads them.
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_nanos())
            .unwrap_or(0);
        let seq = self.mint_seq.fetch_add(1, Ordering::Relaxed);
        IncarnationCookie(nanos ^ (u128::from(node.0) << 40) ^ (u128::from(seq) << 8))
    }

    fn launch(&self, spec: &LaunchSpec) -> Result<u32, String> {
        let exe = crate::sibling_binary(self.tuning.exe)?;
        std::fs::create_dir_all(&self.tuning.workdir)
            .map_err(|e| format!("realm workdir {}: {e}", self.tuning.workdir.display()))?;
        let log_path = self
            .tuning
            .workdir
            .join(format!("realm-{}.log", spec.node.0));
        let log = std::fs::File::create(&log_path)
            .map_err(|e| format!("realm log {}: {e}", log_path.display()))?;
        let node_env = self.child_env(spec);
        let mut child = crate::spawn_node_grouped(&exe, &self.tuning.anchors, &node_env, log)
            .map_err(|e| format!("spawn realm {}: {e}", spec.node.0))?;
        let pid = child.id();

        // A SUCCESSFUL FORK IS NOT A SUCCESSFUL LAUNCH. `spawn` returns as soon as the process exists, so a
        // child that dies milliseconds later — the common case being `AddrInUse` when something already holds
        // this realm's slot in the FIXED RLM port band — used to be recorded as a healthy launch. The
        // reconciler then saw `desired > running`, asked again, and repeated FOREVER: nine consecutive instant
        // deaths counted ZERO failures, no message named the port collision, and the player simply never got a
        // world. Silence is the defect; a spawn that cannot survive its own first breath must SAY SO.
        //
        // So: give the child a brief window to fall over, and treat an exit inside it as a failed launch. This
        // costs one short pause per spawn (spawns are demand-driven and rare) and NEVER waits on a healthy
        // boot — a live child is still running at the end of the window, which is the overwhelmingly common
        // case. A child that dies LATER is a different condition, caught by the liveness sweep, not here.
        let deadline = Instant::now() + EARLY_DEATH_WINDOW;
        while Instant::now() < deadline {
            match child.try_wait() {
                Ok(Some(status)) => {
                    return Err(format!(
                        "realm {} died immediately (exit {status}) — it never bound its port. \
                         Its own reason is in {}; the usual cause is a SURVIVOR from an earlier cluster \
                         still holding this realm's slot in the fixed RLM port band \
                         (VD_RLM_FIRST_PORT/VD_RLM_PORT_LIMIT), which `vd-devcluster down` now reaps",
                        spec.node.0,
                        log_path.display()
                    ));
                }
                Ok(None) => {}
                // The handle is unusable; do not invent a verdict — let the liveness sweep decide.
                Err(_) => break,
            }
            std::thread::sleep(EARLY_DEATH_POLL);
        }

        self.lock().insert(spec.node, Slot::Owned { child, pid });
        Ok(pid)
    }

    fn adopt(&self, node: NodeId, pid: u32, cookie: IncarnationCookie, probe: SocketAddr) {
        // RLM 5e-4: reconstruct an ORPHAN slot for a survivor recovered from the durable launch ledger on a
        // restart — a DIFFERENT orchestrator process launched it, so there is no `Child` handle; `is_alive`
        // confirms it via the `/whoami` cookie-probe (the pid-reuse guard). Seed the cache optimistic-ALIVE
        // so the FIRST post-rehydrate sweep does NOT probe (defers the recovery burst by one interval, D8);
        // it re-probes once the interval elapses, so a survivor's later death is still observed (D7).
        self.lock().insert(
            node,
            Slot::Orphan {
                pid,
                cookie,
                probe,
                cache: Some((Instant::now(), true)),
            },
        );
    }

    fn is_alive(&self, node: NodeId) -> bool {
        // Owned = a real non-blocking `waitpid` (`Ok(None)` = still running; else dead). Orphan = the slow-
        // cadence `/whoami` cookie-probe: return the cached result within `orphan_probe_interval`, else probe
        // and require the echoed cookie to EQUAL ours (a recycled pid answering with a different cookie —
        // or nothing bound — reads dead: the pid-reuse guard). Single sim-thread caller, so a bounded probe
        // may hold the lock.
        match self.lock().get_mut(&node) {
            Some(Slot::Owned { child, .. }) => matches!(child.try_wait(), Ok(None)),
            Some(Slot::Orphan {
                cookie,
                probe,
                cache,
                ..
            }) => match cache {
                Some((at, result)) if at.elapsed() < self.tuning.orphan_probe_interval => *result,
                _ => {
                    let alive =
                        crate::admin_get_body(*probe, "/whoami", Some(self.tuning.probe_timeout))
                            == Some(cookie.to_env_string());
                    *cache = Some((Instant::now(), alive));
                    alive
                }
            },
            None => false,
        }
    }

    fn teardown(&self, node: NodeId) {
        // Best-effort (the trait contract): an unknown/already-torn-down id is a no-op.
        let Some(slot) = self.lock().remove(&node) else {
            return;
        };
        let grace = self.tuning.drain_grace;
        // DETACHED so `kill_realm` never blocks on the grace window (D2). SIGTERM the group → poll → SIGKILL.
        match slot {
            // Owned: we hold the `Child`, so the final `wait()` REAPS it (a dropped-without-wait `Child`
            // would leak a `<defunct>` per spin-down — the self-DoS on RLM's up/DOWN churn the vet flagged).
            Slot::Owned { mut child, pid } => {
                std::thread::spawn(move || {
                    crate::signal_group(pid, "TERM");
                    let deadline = Instant::now() + grace;
                    loop {
                        match child.try_wait() {
                            Ok(Some(_)) => return, // exited gracefully + reaped by try_wait
                            Ok(None) => {}
                            Err(_) => break, // wait errored — fall through to the hard kill
                        }
                        if Instant::now() >= deadline {
                            break;
                        }
                        std::thread::sleep(TEARDOWN_POLL);
                    }
                    crate::signal_group(pid, "KILL");
                    let _ = child.kill(); // SIGKILL the direct child too
                    let _ = child.wait(); // REAP — never leave a zombie
                });
            }
            // Orphan: a DIFFERENT process's child (reparented to init on the crash), so we cannot `wait()` it
            // — signal its group, escalate to SIGKILL after the grace, and let init reap the corpse.
            Slot::Orphan { pid, .. } => {
                std::thread::spawn(move || {
                    crate::signal_group(pid, "TERM");
                    let deadline = Instant::now() + grace;
                    while Instant::now() < deadline && crate::pid_alive(pid) {
                        std::thread::sleep(TEARDOWN_POLL);
                    }
                    if crate::pid_alive(pid) {
                        crate::signal_group(pid, "KILL");
                    }
                });
            }
        }
    }

    fn book_peer(&self, node: NodeId, addr: SocketAddr) {
        // The production side of the 5b layering fix: push the child's addr into THIS orchestrator's mesh
        // so it can initiate to the freshly-launched shard (CA-1 S2), no DNS.
        self.control.update_peer_addr(node, addr);
    }
}
