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
}

/// One launched realm shard this backend owns — the OS `Child` (for `try_wait` liveness + the teardown
/// reap) and its pid (which, via `process_group(0)`, is also the pgid the group signal targets).
struct OwnedSlot {
    child: Child,
    pid: u32,
}

/// The real-process [`LaunchBackend`]: forks a `vd-shard` per realm and books it into the orchestrator's
/// mesh. Cheap to hold (state behind a `Mutex`); the reconciler drives it as `&dyn LaunchBackend` inside a
/// [`vd_node::rlm_spawn::SpawnCore`].
pub struct ProcLaunchBackend {
    /// The orchestrator's own mesh control surface — the real [`LaunchBackend::book_peer`] target.
    control: Arc<MeshControl>,
    tuning: ProcSpawnTuning,
    /// Live owned children, keyed by the id the kernel minted. `Mutex` (not the kernel's — this is the
    /// backend's private OS-handle table); deterministic `BTreeMap` order.
    slots: Mutex<BTreeMap<NodeId, OwnedSlot>>,
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

    fn lock(&self) -> MutexGuard<'_, BTreeMap<NodeId, OwnedSlot>> {
        self.slots.lock().unwrap_or_else(PoisonError::into_inner)
    }

    /// The per-realm env the child shard reads at boot, merged OVER [`ProcSpawnTuning::anchors`]. The
    /// profile is DERIVED by the child from `VD_OWN_COORD` (HR3 — 5a's `profile_for(own_coord.profile_kind)`);
    /// `VD_REALM_KIND`/`VD_REALM_SEED` give the child its `own_realm` `RealmId` (the lowered leaf), and
    /// `VD_INCARNATION_COOKIE`/`VD_PROBE_ADDR` arm the `/whoami` identity echo.
    fn child_env(&self, spec: &LaunchSpec) -> Vec<(&'static str, String)> {
        let realm = spec.coord.lowered();
        vec![
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
        ]
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
        let child = crate::spawn_node_grouped(&exe, &self.tuning.anchors, &node_env, log)
            .map_err(|e| format!("spawn realm {}: {e}", spec.node.0))?;
        let pid = child.id();
        self.lock().insert(spec.node, OwnedSlot { child, pid });
        Ok(pid)
    }

    fn is_alive(&self, node: NodeId) -> bool {
        // Owned-slot liveness: a real non-blocking `waitpid`. `Ok(None)` = still running; `Ok(Some(_))` =
        // exited (reaped here); `Err`/unknown = dead. (The orphan cookie-probe path is slice 5e.)
        match self.lock().get_mut(&node) {
            Some(slot) => matches!(slot.child.try_wait(), Ok(None)),
            None => false,
        }
    }

    fn teardown(&self, node: NodeId) {
        // Best-effort (the trait contract): an unknown/already-torn-down id is a no-op.
        let Some(OwnedSlot { mut child, pid }) = self.lock().remove(&node) else {
            return;
        };
        let grace = self.tuning.drain_grace;
        // DETACHED so `kill_realm` never blocks on the grace window (D2). SIGTERM the group → poll → SIGKILL
        // → REAP. The final `wait()` releases the zombie (a dropped-without-wait `Child` would leak a
        // `<defunct>` per spin-down — the exact self-DoS on RLM's up/DOWN churn the vet flagged).
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

    fn book_peer(&self, node: NodeId, addr: SocketAddr) {
        // The production side of the 5b layering fix: push the child's addr into THIS orchestrator's mesh
        // so it can initiate to the freshly-launched shard (CA-1 S2), no DNS.
        self.control.update_peer_addr(node, addr);
    }
}
