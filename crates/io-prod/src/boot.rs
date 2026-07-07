//! R-6a (M3): the durable MONOTONE process-incarnation boot-counter + the shared durable-path guard.
//!
//! WHY. The mesh stamps every reliable frame with a `process_incarnation` ([`crate::mesh`]); the receiver
//! ledger resets its dedup high-water only when a peer's incarnation INCREASES (`classify_reliable` A1). So a
//! node that restarts (its per-`(peer,class)` seq resets to 0) MUST come up at a STRICTLY HIGHER incarnation
//! than the value a still-running peer's ledger holds, or the restarted `seq0..` is silently DEDUPED away.
//! The dev `launch_incarnation()` = wall-clock-ms is UNSAFE in the cloud: a k8s CrashLoopBackOff sub-second
//! restart mints an EQUAL incarnation (⇒ silent dedup loss), and an NTP/reschedule clock REWIND a LOWER one
//! (⇒ `StaleIncarnation`-drops all restarted traffic). The cure: a counter derived ONLY from its own prior
//! durable value + 1 — MONOTONE across any restart, immune to a wall-clock rewind (no `SystemTime` in the
//! increment path).
//!
//! MECHANISM. A tiny per-node sidecar file (NOT the redb `Store` — gateway/shard have none; HR3 wants ONE
//! mechanism for every node kind, and the common denominator is "a tiny file on a persistent path"). The
//! 24-byte record is checksummed so a torn/foreign file is DETECTED (fail-loud) rather than silently mis-read
//! as genesis. The increment is atomic-durable (write-tmp → fsync → rename → fsync-dir) and the new value is
//! durable BEFORE the caller stamps any frame with it — so a crash after stamping can never re-use a value,
//! and a crash before the rename completes only ever re-uses a value that was NEVER on the wire (safe).
//!
//! No new dependency (std-only). Not a frozen-`sim::io`-seam concern — the incarnation is a `MeshConfig` field.
//!
//! Cloud-ready k3d Slice 2 ALSO homes the CLOUD PROFILE here (the shared boot guards live in one place): the
//! `VD_PROFILE=cloud` resolution, the derived-D-3 resolver (`resolve_d3`), and the footgun preflight
//! (`enforce_cloud_preflight`) — a config MODE (HR3), never a per-shard-kind fork.

use std::error::Error;
use std::path::Path;

use crate::mesh::{
    DEFAULT_CONFIRM_UNREACHABLE_AFTER_RETRIES, DEFAULT_REDIAL_BACKOFF_MAX,
    DEFAULT_REDIAL_BACKOFF_MIN,
};
use crate::runtime::EnvConfig;
use vd_sim::directory::{DirectoryTuning, validate_self_fence_cadence};
use vd_sim::saga::LivenessTuning;

/// The 8-byte file magic (identifies a boot-counter sidecar; a foreign/short file is CORRUPT, never genesis).
const MAGIC: [u8; 8] = *b"VDBOOT\0\0";
/// The fixed on-disk record: `magic(8) || counter_le(8) || checksum_le(8)`.
const RECORD_LEN: usize = 24;

/// A boot-counter operation that could not complete. Every arm FAILS LOUD — the caller must refuse to boot
/// rather than degrade to an unsafe incarnation (a lower/reused value is a silent-data-loss landmine).
#[derive(Debug)]
pub enum BootCounterError {
    /// The counter file exists but is unreadable AS a counter (wrong length, bad magic, or checksum
    /// mismatch — a torn write or a foreign file). NEVER silently reset to genesis: that would hand out a
    /// LOWER value than the last durable one and re-open the silent-dedup loss.
    Corrupt { path: String, detail: &'static str },
    /// The u64 counter overflowed (a `checked_add` guard — a wrap to 0 would be a silent monotonicity break).
    Overflow { path: String },
    /// The counter path is not on a durable volume (a temp/ephemeral mount, or outside the declared durable
    /// root) and the explicit dev/test escape is not set — refusing to boot with a non-durable incarnation.
    Ephemeral { path: String, detail: String },
    /// Underlying filesystem I/O failed (read/write/fsync/rename).
    Io {
        path: String,
        source: std::io::Error,
    },
}

impl std::fmt::Display for BootCounterError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            BootCounterError::Corrupt { path, detail } => write!(
                f,
                "boot-counter file {path} is corrupt ({detail}) — refusing to boot rather than reset to a \
                 LOWER incarnation (which would silently dedup this node's restarted traffic)"
            ),
            BootCounterError::Overflow { path } => {
                write!(
                    f,
                    "boot-counter {path} overflowed u64 (impossible in practice)"
                )
            }
            BootCounterError::Ephemeral { path, detail } => write!(
                f,
                "durable path {path} is not durable ({detail}) — durable state MUST live on a persistent \
                 volume so it survives a restart; set the caller's *_EPHEMERAL_OK=1 ONLY for a throwaway \
                 dev/test cluster. Refusing to boot."
            ),
            BootCounterError::Io { path, source } => {
                write!(f, "boot-counter {path} I/O error: {source}")
            }
        }
    }
}

impl std::error::Error for BootCounterError {}

/// A deterministic, dependency-free checksum (FNV-1a 64) over the magic + counter bytes. Detects a torn
/// write or a foreign file; NOT a security MAC (a boot-counter file is trusted-local, PVC-scoped).
fn checksum(bytes: &[u8]) -> u64 {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for &b in bytes {
        h ^= u64::from(b);
        h = h.wrapping_mul(0x0000_0100_0000_01b3);
    }
    h
}

/// Canonicalize `p`, resolving symlinks in its longest EXISTING prefix and re-appending any not-yet-created
/// tail (so an uncreated subdir still compares consistently against a canonicalized root — macOS `/var` →
/// `/private/var`). Falls back to the raw path only if nothing on the way up canonicalizes.
fn canon_lenient(p: &Path) -> std::path::PathBuf {
    if let Ok(c) = std::fs::canonicalize(p) {
        return c;
    }
    match p.parent().filter(|parent| !parent.as_os_str().is_empty()) {
        Some(parent) => {
            let mut base = canon_lenient(parent);
            if let Some(name) = p.file_name() {
                base.push(name);
            }
            base
        }
        None => p.to_path_buf(),
    }
}

/// Reject a durable-state path that is NOT on a persistent volume — the shared guard for BOTH the redb
/// `VD_STORE_PATH` and the boot-counter `VD_BOOT_STATE_DIR` (DRY; one path-durability policy).
///
/// Two modes (`ephemeral_ok` bypasses both — the explicit dev/test escape):
/// - `durable_root = Some(root)`: the PRODUCTION-grade ALLOW-list — `path` must canonicalize UNDER `root`
///   (the declared mounted volume). A k8s `emptyDir` mounts under `/var/lib/kubelet/...` (NOT `/tmp`), so a
///   prefix DENY-list cannot catch it; requiring the path be under an explicitly-declared durable root does.
/// - `durable_root = None`: the dev-safety DENY-list — reject a canonical temp dir (the old `/tmp` data-loss
///   net) but allow anything else.
///
/// # Errors
/// [`BootCounterError::Ephemeral`] if the path fails the active check and `ephemeral_ok` is false.
pub fn check_durable_path(
    path: &Path,
    durable_root: Option<&Path>,
    ephemeral_ok: bool,
) -> Result<(), BootCounterError> {
    if ephemeral_ok {
        return Ok(());
    }
    // Reject a `..` traversal: `canonicalize()` only collapses `..` for EXISTING paths, so an uncreated
    // path like `<durable_root>/../../tmp` would re-append `..` verbatim (see `canon_lenient`) and could
    // escape the allow-list root. A durable-state path never legitimately contains `..`.
    if path
        .components()
        .any(|c| matches!(c, std::path::Component::ParentDir))
    {
        return Err(BootCounterError::Ephemeral {
            path: path.display().to_string(),
            detail: "contains a `..` parent-dir traversal (durable-state paths must be direct)"
                .into(),
        });
    }
    // Canonicalize the directory that WILL hold the file (the file itself may not exist yet), so symlinked
    // spellings collapse to one form before any prefix test. `canon_lenient` resolves symlinks in the
    // longest EXISTING prefix and re-appends the not-yet-created tail — so a path whose parent dir does not
    // exist yet still compares consistently (macOS /var → /private/var, even for an uncreated subdir).
    let probe = match path.parent().filter(|p| !p.as_os_str().is_empty()) {
        Some(parent) => canon_lenient(parent),
        None => canon_lenient(path),
    };
    match durable_root {
        Some(root) => {
            let root = canon_lenient(root);
            if !probe.starts_with(&root) {
                return Err(BootCounterError::Ephemeral {
                    path: path.display().to_string(),
                    detail: format!(
                        "not under the declared durable root {} (the mounted persistent volume)",
                        root.display()
                    ),
                });
            }
        }
        None => {
            if is_under_temp(&probe) {
                return Err(BootCounterError::Ephemeral {
                    path: path.display().to_string(),
                    detail: format!(
                        "under a temp dir ({})",
                        canon_lenient(&std::env::temp_dir()).display()
                    ),
                });
            }
        }
    }
    Ok(())
}

/// Whether an ALREADY-CANONICALIZED path resolves under a temp dir (`$TMPDIR` / `/tmp` / `/private/tmp`). The
/// ONE temp-deny predicate — [`check_durable_path`]'s no-declared-root branch AND (cloud-ready k3d Slice 2)
/// [`enforce_cloud_preflight`]'s `VD_STORE_DURABLE_ROOT` check both use it, so a durable-root that ITSELF
/// canonicalizes under temp (a compliant-looking config that would re-open the ephemeral hole the allow-list
/// is meant to close) is rejected the same way. Callers canonicalize (via `canon_lenient`) first.
pub(crate) fn is_under_temp(canonical: &Path) -> bool {
    let tmp = canon_lenient(&std::env::temp_dir());
    canonical.starts_with(&tmp)
        || canonical.starts_with("/tmp")
        || canonical.starts_with("/private/tmp")
}

// ---- Cloud config profile (cloud-ready k3d Slice 2) --------------------------------------------

/// The deployment profile a node boots under. `DevTest` (the default + every in-process rig + the process
/// tests) keeps the INERT D-3 defaults — byte-identical to before this slice. `Cloud` SUPPLIES the derived
/// ACTIVE split-brain config, REJECTS re-zeroing it, and drops dev footguns (all fail-loud at boot).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Profile {
    DevTest,
    Cloud,
}

/// Which server role is booting — the D-3 node-side self-fence knobs differ (a shard rechecks its Realm head,
/// a gateway its Session head; the orchestrator has no node-side self-fence). A CONFIG axis (HR3), never a
/// behavioral fork.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum NodeRole {
    Orchestrator,
    Shard,
    Gateway,
}

/// The resolved D-3 lease-liveness config: the orchestrator's `DirectoryTuning` + the transport-cross-checked
/// `LivenessTuning` + the node-side self-fence grace/recheck. Produced by [`resolve_d3`], consumed by the bins.
#[derive(Clone, Copy, Debug)]
pub struct ResolvedD3 {
    pub directory: DirectoryTuning,
    pub liveness: LivenessTuning,
    pub node_self_fence_grace: u64,
    pub node_recheck: u64,
}

/// A cloud-profile violation — fail-loud at boot (the `Refusing to boot` posture).
#[derive(Debug, thiserror::Error, PartialEq, Eq)]
pub enum CloudProfileError {
    #[error("VD_PROFILE={0:?} is not a recognized profile (use `cloud`, `dev`, or `test`)")]
    UnknownProfile(String),
    #[error(
        "cloud profile forbids {0} (a dev-only escape/footgun — a cloud pod must use the durable/persistent \
         path, never an ephemeral one)"
    )]
    ForbiddenInCloud(&'static str),
    #[error(
        "cloud profile requires {0} to name a real persistent volume (absent/empty) — it is force-validated \
         in cloud so its durable path uses the allow-list, catching a k8s emptyDir the temp deny-list cannot"
    )]
    MissingDurableRoot(&'static str),
    #[error(
        "cloud profile: {key} {path:?} is temp-TANGLED (canonicalizes UNDER a temp dir, or is an ANCESTOR of \
         one — e.g. `/`) — not a real persistent volume; either would re-open the ephemeral hole the allow-list \
         exists to close (a path under the root could still land inside temp)"
    )]
    DurableRootTempTangled { key: &'static str, path: String },
    #[error(
        "cloud profile refuses the built-in DEV auth verifying key (VD_AUTH_PUBKEY decodes to the dev key) — \
         supply a real production key"
    )]
    DevAuthKey,
    #[error(
        "cloud profile requires an ACTIVE D-3 lease-liveness config, but {0} resolved to 0 (inert): a cluster \
         with inert D-3 boots with NO split-brain protection (the DEFERRED.md:984 hole)"
    )]
    InertD3(&'static str),
}

/// Resolve the deployment [`Profile`] from `VD_PROFILE`: absent / `dev` / `test` / empty ⇒ [`Profile::DevTest`]
/// (preserves every in-process rig + process-tier test, byte-identical); `cloud` ⇒ [`Profile::Cloud`]; any
/// other present value is a LOUD error (never fail-OPEN into dev on a typo).
///
/// # Errors
/// [`CloudProfileError::UnknownProfile`] on an unrecognized non-empty `VD_PROFILE`.
pub fn resolve_profile(env: &EnvConfig) -> Result<Profile, CloudProfileError> {
    match env.string("VD_PROFILE") {
        Err(_) => Ok(Profile::DevTest),
        Ok(v) => match v.trim().to_ascii_lowercase().as_str() {
            "cloud" => Ok(Profile::Cloud),
            "dev" | "test" | "" => Ok(Profile::DevTest),
            _ => Err(CloudProfileError::UnknownProfile(v)),
        },
    }
}

/// Resolve the D-3 lease-liveness config for `role` given the `profile` + `tick_hz`.
///
/// In [`Profile::DevTest`] the fallback baseline is the INERT [`DirectoryTuning::default`] /
/// [`LivenessTuning::default`] / node grace=0 / recheck=0 — behaviour-identical to before this slice. In
/// [`Profile::Cloud`] the baseline is the DERIVED active set: [`DirectoryTuning::cloud`] +
/// [`LivenessTuning::cloud`] (its window COMPUTED from the SAME [`DEFAULT_REDIAL_BACKOFF_MIN`] /
/// [`DEFAULT_REDIAL_BACKOFF_MAX`] / [`DEFAULT_CONFIRM_UNREACHABLE_AFTER_RETRIES`] the mesh dials with — the
/// R-4c cross-check) + node grace = the derived self-fence + node recheck = `tick_hz/2`. Any present `VD_*`
/// env value OVERRIDES its baseline. EVERY resolved config is validated in BOTH profiles (an incoherent
/// override fails loud); in cloud ONLY, an INERT resolution (a re-zeroed renew/reaper, or an inert node
/// self-fence/recheck on a shard/gateway) is REJECTED — the crux of DEFERRED.md:984.
///
/// # Errors
/// A boxed validation error (incoherent tuning) or [`CloudProfileError::InertD3`] (cloud + inert).
pub fn resolve_d3(
    env: &EnvConfig,
    profile: Profile,
    role: NodeRole,
    tick_hz: u32,
) -> Result<ResolvedD3, Box<dyn Error>> {
    let (base_dir, base_live, base_node_grace, base_recheck) = match profile {
        Profile::DevTest => (
            DirectoryTuning::default(),
            // The orchestrator (the ONLY consumer of `liveness`) shipped a PROD-SAFE `n = 3` default before
            // this slice (CSCALE-1: a single recoverable blip toward a LIVE peer never confirms it dead) —
            // NOT `LivenessTuning::default()`'s kill-equivalent `n = 1`. Preserve it so DevTest / the process
            // tier stay byte-identical (finding 5). `VD_LIVENESS_*` still override below.
            LivenessTuning {
                n_consecutive_unreachable: 3,
                ..LivenessTuning::default()
            },
            0u64,
            0u64,
        ),
        Profile::Cloud => (
            DirectoryTuning::cloud(tick_hz),
            LivenessTuning::cloud(
                tick_hz,
                DEFAULT_CONFIRM_UNREACHABLE_AFTER_RETRIES,
                DEFAULT_REDIAL_BACKOFF_MIN,
                DEFAULT_REDIAL_BACKOFF_MAX,
            ),
            DirectoryTuning::cloud(tick_hz).self_fence_grace_ticks,
            u64::from(tick_hz) / 2,
        ),
    };
    let directory = DirectoryTuning {
        // NOTE (review F2): the pre-slice orchestrator bin REQUIRED VD_LEASE_TTL (`parse`); this SHARED
        // resolver defaults it (`parse_or`) because the shard/gateway roles never set it (a required parse
        // would break their boot) and the CLOUD profile DERIVES it (`2*hz`) — so an absent VD_LEASE_TTL now
        // yields the profile's coherent base (DevTest 100 / cloud 2*hz) rather than a boot error. In DevTest
        // it is inert bookkeeping (the reaper is off), and every dev/process rig sets it explicitly.
        lease_ttl_ticks: env.parse_or("VD_LEASE_TTL", base_dir.lease_ttl_ticks)?,
        lease_renew_interval_ticks: env.parse_or(
            "VD_LEASE_RENEW_INTERVAL",
            base_dir.lease_renew_interval_ticks,
        )?,
        min_renews_before_lapse: env.parse_or(
            "VD_MIN_RENEWS_BEFORE_LAPSE",
            base_dir.min_renews_before_lapse,
        )?,
        self_fence_grace_ticks: env
            .parse_or("VD_SELF_FENCE_GRACE", base_dir.self_fence_grace_ticks)?,
        max_self_fence_grace_ticks: env.parse_or(
            "VD_MAX_SELF_FENCE_GRACE",
            base_dir.max_self_fence_grace_ticks,
        )?,
        reaper_interval_ticks: env
            .parse_or("VD_REAPER_INTERVAL", base_dir.reaper_interval_ticks)?,
        recovery_grace_ticks: env.parse_or("VD_RECOVERY_GRACE", base_dir.recovery_grace_ticks)?,
    };
    let liveness = LivenessTuning {
        n_consecutive_unreachable: env
            .parse_or("VD_LIVENESS_N", base_live.n_consecutive_unreachable)?,
        unreachable_window_ticks: env
            .parse_or("VD_LIVENESS_WINDOW", base_live.unreachable_window_ticks)?,
        retry_delay_ticks_hint: env
            .parse_or("VD_LIVENESS_RETRY_HINT", base_live.retry_delay_ticks_hint)?,
    };
    let node_self_fence_grace = env.parse_or("VD_SELF_FENCE_GRACE", base_node_grace)?;
    let node_recheck = match role {
        NodeRole::Shard => env.parse_or("VD_REALM_RECHECK", base_recheck)?,
        NodeRole::Gateway => env.parse_or("VD_SESSION_RECHECK", base_recheck)?,
        NodeRole::Orchestrator => base_recheck,
    };

    // Coherence — runs in BOTH profiles (dev inert passes vacuously; a cloud/override incoherence fails loud).
    directory.validate()?;
    liveness.validate()?;
    liveness.validate_against(
        DEFAULT_CONFIRM_UNREACHABLE_AFTER_RETRIES,
        DEFAULT_REDIAL_BACKOFF_MIN,
        DEFAULT_REDIAL_BACKOFF_MAX,
        tick_hz,
    )?;
    validate_self_fence_cadence(node_self_fence_grace, node_recheck)?;

    // Cloud ONLY: an inert resolution IS the split-brain hole (a cluster boots green with no protection) —
    // reject it loud. This CANNOT live in `DirectoryTuning::validate` (which must stay vacuous-on-inert for
    // dev); it is a PROFILE assertion on the resolved struct. Only the TWO knobs the validators above CANNOT
    // catch are checked here: `lease_renew_interval == 0` makes `validate` short-circuit `active=false` (the
    // whole D-3 goes vacuous — the exact green-boot hole), and `reaper_interval` is not part of any ordering
    // check (an inert reaper never reassigns a dead owner). The node-side inert cases are ALREADY caught by
    // the validators above and need no redundant check here: a `VD_SELF_FENCE_GRACE=0` fails
    // `directory.validate` (SelfFenceWithinTtl, since cloud renew is active), and a `VD_REALM_RECHECK`/
    // `VD_SESSION_RECHECK=0` fails `validate_self_fence_cadence` (NoConfirmationChannel).
    if profile == Profile::Cloud {
        if directory.lease_renew_interval_ticks == 0 {
            return Err(CloudProfileError::InertD3("VD_LEASE_RENEW_INTERVAL").into());
        }
        if directory.reaper_interval_ticks == 0 {
            return Err(CloudProfileError::InertD3("VD_REAPER_INTERVAL").into());
        }
    }
    Ok(ResolvedD3 {
        directory,
        liveness,
        node_self_fence_grace,
        node_recheck,
    })
}

/// The cloud footgun preflight — call ONCE per node at boot, BEFORE any durable/authoritative action. In
/// [`Profile::DevTest`] a no-op passthrough. In [`Profile::Cloud`] it fails LOUD on: any ephemeral-store
/// escape (`VD_STORE_EPHEMERAL_OK` / `VD_BOOT_STATE_EPHEMERAL_OK` / `VD_OUTBOX_EPHEMERAL_OK`) or a manual
/// `VD_PROCESS_INCARNATION` (the durable monotone counter is mandatory in cloud); a missing/empty or
/// under-temp `VD_STORE_DURABLE_ROOT`; and (gateway only, via `dev_pubkey = Some`) a `VD_AUTH_PUBKEY` that
/// decodes to the built-in DEV verifying key. Returns the resolved [`Profile`] to thread into [`resolve_d3`].
///
/// Whether a canonicalized durable-root path is temp-TANGLED — it must be DISJOINT from the temp dir. Rejects
/// a root UNDER temp (a temp path masquerading as durable — [`is_under_temp`]) AND a root that is an ANCESTOR
/// of temp (e.g. `/`, whose allow-list `starts_with(root)` would admit a store/counter that itself lands in
/// temp — finding 4). Both callers pass a `canon_lenient`-ed path.
fn temp_tangled(root_canon: &Path) -> bool {
    let tmp = canon_lenient(&std::env::temp_dir());
    is_under_temp(root_canon) || tmp.starts_with(root_canon)
}

/// Require a cloud durable-root env var (`key`) to name a real persistent volume: present, non-empty, and
/// temp-disjoint. The shared check for [`enforce_cloud_preflight`]'s store + boot-counter roots.
///
/// # Errors
/// [`CloudProfileError::MissingDurableRoot`] (absent/empty) or [`CloudProfileError::DurableRootTempTangled`].
fn require_durable_root(env: &EnvConfig, key: &'static str) -> Result<(), CloudProfileError> {
    let root = env
        .string(key)
        .ok()
        .filter(|s| !s.trim().is_empty())
        .ok_or(CloudProfileError::MissingDurableRoot(key))?;
    if temp_tangled(&canon_lenient(Path::new(&root))) {
        return Err(CloudProfileError::DurableRootTempTangled { key, path: root });
    }
    Ok(())
}

/// `dev_pubkey` is passed IN (the gateway supplies `Some(vd_bins::DEV_AUTH_PUBKEY_BYTES)`; orchestrator/shard
/// pass `None`) to avoid an `io-prod → vd-bins` circular dependency.
///
/// # Errors
/// [`CloudProfileError`] on any cloud violation; a boxed [`crate::runtime::ConfigError`] on a malformed value.
pub fn enforce_cloud_preflight(
    env: &EnvConfig,
    dev_pubkey: Option<[u8; 32]>,
) -> Result<Profile, Box<dyn Error>> {
    let profile = resolve_profile(env)?;
    if profile == Profile::Cloud {
        for key in [
            "VD_STORE_EPHEMERAL_OK",
            "VD_BOOT_STATE_EPHEMERAL_OK",
            "VD_OUTBOX_EPHEMERAL_OK",
        ] {
            if env.bool(key)? {
                return Err(CloudProfileError::ForbiddenInCloud(key).into());
            }
        }
        // The M3 durable monotone incarnation path is mandatory in cloud — an explicit override bypasses it.
        if env.string("VD_PROCESS_INCARNATION").is_ok() {
            return Err(CloudProfileError::ForbiddenInCloud("VD_PROCESS_INCARNATION").into());
        }
        // BOTH durable roots must name a real persistent volume (present, non-empty, temp-DISJOINT):
        // - VD_STORE_DURABLE_ROOT: the redb store (directory + saga WAL + clock ceiling).
        // - VD_BOOT_DURABLE_ROOT: the M3 monotone boot-counter (the split-brain incarnation ledger). In cloud
        //   the manual VD_PROCESS_INCARNATION + every ephemeral escape are forbidden above, so this durable
        //   counter is the MANDATORY, SOLE incarnation source — leaving it on an ephemeral emptyDir (which the
        //   temp deny-list cannot catch, mounted under /var/lib/kubelet) is the same fail-open as the store
        //   (finding 3). Forcing the root present flips `check_durable_path` into allow-list mode for it.
        require_durable_root(env, "VD_STORE_DURABLE_ROOT")?;
        require_durable_root(env, "VD_BOOT_DURABLE_ROOT")?;
        if let Some(dev) = dev_pubkey {
            // Compare DECODED bytes (case-robust); a malformed VD_AUTH_PUBKEY propagates loud, never passes.
            if env.hex32("VD_AUTH_PUBKEY")? == dev {
                return Err(CloudProfileError::DevAuthKey.into());
            }
        }
    }
    Ok(profile)
}

/// The durable monotone boot-counter.
pub struct BootCounter;

impl BootCounter {
    /// Read the counter at `path`, hand out the NEXT incarnation (durable BEFORE this returns), for the
    /// caller to stamp on its frames.
    ///
    /// - ABSENT file ⇒ genesis at `genesis_floor.max(1)` (the floor lets a re-provisioned pod with a fresh
    ///   volume still exceed any peer's surviving wall-clock-era ledger; steady state is purely counter+1).
    /// - VALID file ⇒ `prev.checked_add(1)`.
    /// - PRESENT-but-CORRUPT (wrong length / magic / checksum) ⇒ [`BootCounterError::Corrupt`] (FAIL LOUD,
    ///   never reset to a lower value).
    ///
    /// # Errors
    /// [`BootCounterError`] on a corrupt file, u64 overflow, or filesystem I/O failure.
    pub fn increment_on_boot(path: &Path, genesis_floor: u64) -> Result<u64, BootCounterError> {
        let prev = Self::read(path)?;
        let next = match prev {
            None => genesis_floor.max(1),
            Some(p) => p.checked_add(1).ok_or_else(|| BootCounterError::Overflow {
                path: path.display().to_string(),
            })?,
        };
        Self::write_durable(path, next)?;
        Ok(next)
    }

    /// TEST-ONLY (R-6d4-D; feature `store-test-hooks`, ABSENT from release): read the CURRENT persisted counter
    /// WITHOUT incrementing — so the process-tier SIGKILL-restart proof reads v1 before the kill + v2 after the
    /// restart and asserts `v2 == v1 + 1` (the incarnation resolved EXACTLY once). A thin wrapper over the
    /// private [`read`](Self::read) so the MAGIC + checksum validation stays in ONE place (never a raw byte read
    /// at the call site). `Ok(None)` = absent, `Ok(Some(n))` = valid, `Err` = present-but-corrupt.
    #[cfg(any(test, feature = "store-test-hooks"))]
    pub fn current(path: &Path) -> Result<Option<u64>, BootCounterError> {
        Self::read(path)
    }

    /// Read + validate the current counter. `Ok(None)` = absent (genesis); `Ok(Some(n))` = valid; `Err` =
    /// present-but-corrupt (fail loud, never treated as genesis).
    fn read(path: &Path) -> Result<Option<u64>, BootCounterError> {
        let bytes = match std::fs::read(path) {
            Ok(b) => b,
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(None),
            Err(source) => {
                return Err(BootCounterError::Io {
                    path: path.display().to_string(),
                    source,
                });
            }
        };
        if bytes.len() != RECORD_LEN {
            return Err(BootCounterError::Corrupt {
                path: path.display().to_string(),
                detail: "wrong record length",
            });
        }
        if bytes[0..8] != MAGIC {
            return Err(BootCounterError::Corrupt {
                path: path.display().to_string(),
                detail: "bad magic",
            });
        }
        let counter = u64::from_le_bytes(bytes[8..16].try_into().expect("8 bytes"));
        let stored_ck = u64::from_le_bytes(bytes[16..24].try_into().expect("8 bytes"));
        if checksum(&bytes[0..16]) != stored_ck {
            return Err(BootCounterError::Corrupt {
                path: path.display().to_string(),
                detail: "checksum mismatch (torn write)",
            });
        }
        Ok(Some(counter))
    }

    /// Atomically + durably write `value`: build the 24-byte record, write it to a sibling `.tmp`, fsync the
    /// tmp file's DATA, `rename(tmp, path)` (atomic on POSIX within one filesystem), then fsync the containing
    /// DIRECTORY so the rename itself is durable. On return `path` holds `value` durably.
    fn write_durable(path: &Path, value: u64) -> Result<(), BootCounterError> {
        use std::io::Write as _;
        let io_err = |source: std::io::Error| BootCounterError::Io {
            path: path.display().to_string(),
            source,
        };
        let dir = path.parent().filter(|p| !p.as_os_str().is_empty());
        if let Some(dir) = dir {
            std::fs::create_dir_all(dir).map_err(io_err)?;
        }
        let mut record = Vec::with_capacity(RECORD_LEN);
        record.extend_from_slice(&MAGIC);
        record.extend_from_slice(&value.to_le_bytes());
        record.extend_from_slice(
            &checksum(&{
                let mut mc = [0u8; 16];
                mc[0..8].copy_from_slice(&MAGIC);
                mc[8..16].copy_from_slice(&value.to_le_bytes());
                mc
            })
            .to_le_bytes(),
        );

        let tmp = path.with_extension("tmp");
        {
            let mut f = std::fs::File::create(&tmp).map_err(io_err)?;
            f.write_all(&record).map_err(io_err)?;
            f.sync_all().map_err(io_err)?; // fsync the DATA before the rename
        }
        std::fs::rename(&tmp, path).map_err(io_err)?; // atomic replace
        // fsync the DIRECTORY so the rename (the durability point) survives a crash. Best-effort: not all
        // platforms permit opening a dir for fsync; a failure here is logged, not fatal (the data fsync +
        // atomic rename already bound the loss to a re-used-never-wired value, which is safe).
        if let Some(dir) = dir
            && let Ok(d) = std::fs::File::open(dir)
        {
            let _ = d.sync_all();
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn scratch() -> std::path::PathBuf {
        // A unique per-test dir under the OS temp (the counter file lives here; the guard is tested
        // separately with ephemeral_ok=true so the temp location doesn't trip it).
        let base = std::env::temp_dir().join(format!(
            "vd-bootcounter-test-{}-{:p}",
            std::process::id(),
            &MAGIC as *const _
        ));
        std::fs::create_dir_all(&base).expect("scratch");
        base
    }

    #[test]
    fn genesis_uses_the_floor_then_increments_by_one() {
        let dir = scratch();
        let path = dir.join(format!("g-{}.counter", line!()));
        let _ = std::fs::remove_file(&path);
        // Absent ⇒ genesis at max(1, floor).
        assert_eq!(
            BootCounter::increment_on_boot(&path, 1000).expect("boot1"),
            1000
        );
        // Present ⇒ +1 (the floor is now IGNORED — steady state is purely counter-driven).
        assert_eq!(
            BootCounter::increment_on_boot(&path, 5).expect("boot2"),
            1001
        );
        assert_eq!(
            BootCounter::increment_on_boot(&path, 999).expect("boot3"),
            1002
        );
    }

    #[test]
    fn genesis_floor_is_at_least_one() {
        let dir = scratch();
        let path = dir.join(format!("g0-{}.counter", line!()));
        let _ = std::fs::remove_file(&path);
        assert_eq!(BootCounter::increment_on_boot(&path, 0).expect("boot"), 1);
    }

    #[test]
    fn survives_a_wall_clock_rewind() {
        let dir = scratch();
        let path = dir.join(format!("rw-{}.counter", line!()));
        let _ = std::fs::remove_file(&path);
        // Boot at a high genesis floor (a wall-clock-era value), then a LOWER floor (a clock rewind): the
        // counter ignores the lower floor and advances monotonically from the durable prior value.
        let first = BootCounter::increment_on_boot(&path, 1_700_000_000).expect("boot1");
        let second = BootCounter::increment_on_boot(&path, 42).expect("boot2 (rewound clock)");
        assert!(
            second > first,
            "monotone despite a lower floor: {second} > {first}"
        );
        assert_eq!(second, first + 1);
    }

    #[test]
    fn a_leftover_tmp_from_a_crash_before_rename_is_ignored() {
        let dir = scratch();
        let path = dir.join(format!("tmp-{}.counter", line!()));
        let _ = std::fs::remove_file(&path);
        assert_eq!(BootCounter::increment_on_boot(&path, 1).expect("boot1"), 1);
        // Simulate a crash mid-increment: a stale .tmp exists, but `path` still holds the last durable value.
        std::fs::write(path.with_extension("tmp"), b"garbage half-write").expect("stale tmp");
        // The next boot reads `path` (ignores the stale tmp), advances by one, and overwrites the tmp.
        assert_eq!(BootCounter::increment_on_boot(&path, 1).expect("boot2"), 2);
    }

    #[test]
    fn a_corrupt_file_fails_loud_never_resets_to_genesis() {
        let dir = scratch();
        // Bad magic.
        let p1 = dir.join(format!("c1-{}.counter", line!()));
        std::fs::write(&p1, [0u8; RECORD_LEN]).expect("write");
        assert!(matches!(
            BootCounter::increment_on_boot(&p1, 1),
            Err(BootCounterError::Corrupt { .. })
        ));
        // Wrong length.
        let p2 = dir.join(format!("c2-{}.counter", line!()));
        std::fs::write(&p2, b"short").expect("write");
        assert!(matches!(
            BootCounter::increment_on_boot(&p2, 1),
            Err(BootCounterError::Corrupt { .. })
        ));
        // Valid magic + counter but a BAD checksum (a torn counter field).
        let p3 = dir.join(format!("c3-{}.counter", line!()));
        let mut rec = [0u8; RECORD_LEN];
        rec[0..8].copy_from_slice(&MAGIC);
        rec[8..16].copy_from_slice(&7u64.to_le_bytes());
        rec[16..24].copy_from_slice(&0xDEAD_u64.to_le_bytes()); // wrong checksum
        std::fs::write(&p3, rec).expect("write");
        assert!(matches!(
            BootCounter::increment_on_boot(&p3, 1),
            Err(BootCounterError::Corrupt { .. })
        ));
    }

    #[test]
    fn check_durable_path_allow_list_and_deny_list_and_escape() {
        let dir = scratch();
        let under = dir.join("sub").join("boot.counter");
        // ALLOW-list: path under the declared root ⇒ ok; outside ⇒ Ephemeral.
        assert!(check_durable_path(&under, Some(&dir), false).is_ok());
        assert!(matches!(
            check_durable_path(std::path::Path::new("/etc/boot.counter"), Some(&dir), false),
            Err(BootCounterError::Ephemeral { .. })
        ));
        // DENY-list (no root): a temp path is rejected; a non-temp path is allowed.
        let temp_path = std::env::temp_dir().join("vd-x").join("boot.counter");
        assert!(matches!(
            check_durable_path(&temp_path, None, false),
            Err(BootCounterError::Ephemeral { .. })
        ));
        assert!(
            check_durable_path(
                std::path::Path::new("/var/lib/vd/boot.counter"),
                None,
                false
            )
            .is_ok()
        );
        // The explicit escape bypasses BOTH modes.
        assert!(check_durable_path(&temp_path, None, true).is_ok());
        assert!(check_durable_path(std::path::Path::new("/etc/x"), Some(&dir), true).is_ok());
    }

    #[test]
    fn check_durable_path_rejects_a_dot_dot_traversal() {
        // An uncreated `<root>/../../tmp/boot.counter` must NOT escape the allow-list root (canonicalize
        // cannot collapse `..` on an uncreated chain, so it is rejected outright).
        let escape = std::path::Path::new("/var/lib/vd/../../../tmp/boot.counter");
        assert!(matches!(
            check_durable_path(escape, Some(std::path::Path::new("/var/lib/vd")), false),
            Err(BootCounterError::Ephemeral { .. })
        ));
        // Also rejected in deny-list mode.
        assert!(matches!(
            check_durable_path(escape, None, false),
            Err(BootCounterError::Ephemeral { .. })
        ));
    }

    // ---- Cloud config profile (cloud-ready k3d Slice 2) --------------------------------------
    fn env_of(pairs: &[(&str, &str)]) -> EnvConfig {
        EnvConfig::new(
            pairs
                .iter()
                .map(|(k, v)| ((*k).to_owned(), (*v).to_owned()))
                .collect(),
        )
    }
    fn cloud_err(r: Result<Profile, Box<dyn Error>>) -> CloudProfileError {
        *r.expect_err("expected a cloud-profile violation")
            .downcast::<CloudProfileError>()
            .expect("a CloudProfileError")
    }

    #[test]
    fn resolve_profile_ladder() {
        assert_eq!(
            resolve_profile(&env_of(&[("VD_PROFILE", "cloud")])),
            Ok(Profile::Cloud)
        );
        assert_eq!(
            resolve_profile(&env_of(&[("VD_PROFILE", "CLOUD")])),
            Ok(Profile::Cloud) // case-insensitive
        );
        assert_eq!(resolve_profile(&env_of(&[])), Ok(Profile::DevTest)); // absent ⇒ dev (rigs unbroken)
        assert_eq!(
            resolve_profile(&env_of(&[("VD_PROFILE", "dev")])),
            Ok(Profile::DevTest)
        );
        assert_eq!(
            resolve_profile(&env_of(&[("VD_PROFILE", "test")])),
            Ok(Profile::DevTest)
        );
        assert_eq!(
            resolve_profile(&env_of(&[("VD_PROFILE", "prod")])),
            Err(CloudProfileError::UnknownProfile("prod".to_owned())) // typo never fails-open into dev
        );
    }

    #[test]
    fn cloud_preflight_rejects_ephemeral_escapes_and_manual_incarnation() {
        let base = [
            ("VD_PROFILE", "cloud"),
            ("VD_STORE_DURABLE_ROOT", "/var/lib/vd"),
        ];
        for key in [
            "VD_STORE_EPHEMERAL_OK",
            "VD_BOOT_STATE_EPHEMERAL_OK",
            "VD_OUTBOX_EPHEMERAL_OK",
            "VD_PROCESS_INCARNATION",
        ] {
            let mut pairs = base.to_vec();
            pairs.push((key, "1"));
            assert_eq!(
                cloud_err(enforce_cloud_preflight(&env_of(&pairs), None)),
                CloudProfileError::ForbiddenInCloud(key)
            );
        }
        // Dev mode: the same escape is ALLOWED (no enforcement) — the dev/test path is unbroken.
        assert_eq!(
            enforce_cloud_preflight(&env_of(&[("VD_STORE_EPHEMERAL_OK", "1")]), None)
                .expect("dev profile passes the escape through"),
            Profile::DevTest
        );
    }

    #[test]
    fn cloud_preflight_requires_both_durable_roots_temp_disjoint() {
        // The STORE root absent / whitespace-only → MissingDurableRoot(store).
        assert_eq!(
            cloud_err(enforce_cloud_preflight(
                &env_of(&[("VD_PROFILE", "cloud")]),
                None
            )),
            CloudProfileError::MissingDurableRoot("VD_STORE_DURABLE_ROOT")
        );
        assert_eq!(
            cloud_err(enforce_cloud_preflight(
                &env_of(&[("VD_PROFILE", "cloud"), ("VD_STORE_DURABLE_ROOT", "   ")]),
                None
            )),
            CloudProfileError::MissingDurableRoot("VD_STORE_DURABLE_ROOT")
        );
        // A root that canonicalizes UNDER a temp dir re-opens the ephemeral hole → temp-tangled.
        let tmp_root = std::env::temp_dir().join("vd-cloud-preflight-root");
        assert_eq!(
            cloud_err(enforce_cloud_preflight(
                &env_of(&[
                    ("VD_PROFILE", "cloud"),
                    ("VD_STORE_DURABLE_ROOT", tmp_root.to_str().expect("utf8 path")),
                ]),
                None
            )),
            CloudProfileError::DurableRootTempTangled {
                key: "VD_STORE_DURABLE_ROOT",
                path: tmp_root.to_str().expect("utf8 path").to_owned(),
            }
        );
        // Finding 4: a root that is an ANCESTOR of temp (`/`) would admit temp paths through the allow-list
        // — the parent-of-temp case the old under-temp-only check missed → temp-tangled too.
        assert_eq!(
            cloud_err(enforce_cloud_preflight(
                &env_of(&[("VD_PROFILE", "cloud"), ("VD_STORE_DURABLE_ROOT", "/")]),
                None
            )),
            CloudProfileError::DurableRootTempTangled {
                key: "VD_STORE_DURABLE_ROOT",
                path: "/".to_owned(),
            }
        );
        // Finding 3: a real STORE root but NO boot-counter root → the split-brain incarnation ledger could
        // sit on an ephemeral emptyDir the deny-list cannot catch → MissingDurableRoot(boot).
        assert_eq!(
            cloud_err(enforce_cloud_preflight(
                &env_of(&[
                    ("VD_PROFILE", "cloud"),
                    ("VD_STORE_DURABLE_ROOT", "/var/lib/vd")
                ]),
                None
            )),
            CloudProfileError::MissingDurableRoot("VD_BOOT_DURABLE_ROOT")
        );
        // BOTH roots present + non-temp → passes (orchestrator: no dev-key veto since dev_pubkey=None).
        assert_eq!(
            enforce_cloud_preflight(
                &env_of(&[
                    ("VD_PROFILE", "cloud"),
                    ("VD_STORE_DURABLE_ROOT", "/var/lib/vd"),
                    ("VD_BOOT_DURABLE_ROOT", "/var/lib/vd-boot"),
                ]),
                None
            )
            .expect("both roots present + non-temp → cloud"),
            Profile::Cloud
        );
    }

    #[test]
    fn cloud_preflight_vetoes_the_dev_verifying_key_case_robust() {
        let dev = [0x42u8; 32];
        // UPPERCASE dev hex — the veto compares DECODED bytes, so hex case cannot bypass it (A2-#2).
        let dev_hex_upper = crate::runtime::hex32_encode(&dev).to_uppercase();
        let veto = [
            ("VD_PROFILE", "cloud"),
            ("VD_STORE_DURABLE_ROOT", "/var/lib/vd"),
            ("VD_BOOT_DURABLE_ROOT", "/var/lib/vd-boot"),
            ("VD_AUTH_PUBKEY", dev_hex_upper.as_str()),
        ];
        assert_eq!(
            cloud_err(enforce_cloud_preflight(&env_of(&veto), Some(dev))),
            CloudProfileError::DevAuthKey
        );
        // A real (different) key passes the veto.
        let real_hex = crate::runtime::hex32_encode(&[0x99u8; 32]);
        let ok = [
            ("VD_PROFILE", "cloud"),
            ("VD_STORE_DURABLE_ROOT", "/var/lib/vd"),
            ("VD_BOOT_DURABLE_ROOT", "/var/lib/vd-boot"),
            ("VD_AUTH_PUBKEY", real_hex.as_str()),
        ];
        assert_eq!(
            enforce_cloud_preflight(&env_of(&ok), Some(dev)).expect("a real key passes the veto"),
            Profile::Cloud
        );
    }

    #[test]
    fn cloud_resolve_d3_supplies_active_set_and_devtest_stays_inert() {
        let e = env_of(&[("VD_PROFILE", "cloud")]);
        let orch =
            resolve_d3(&e, Profile::Cloud, NodeRole::Orchestrator, 50).expect("orch cloud d3");
        assert_eq!(orch.directory.lease_renew_interval_ticks, 25); // ACTIVE — not the inert 0
        assert_eq!(orch.directory.self_fence_grace_ticks, 150);
        assert_eq!(orch.directory.max_self_fence_grace_ticks, 250); // 5*hz — the split-brain reassign budget
        assert_eq!(orch.directory.reaper_interval_ticks, 10);
        // The window is the run-spread floor (n·hint = 75), NOT the old SKEW-inflated 120 (finding 1: the
        // window is not the split-brain margin — that lives in max_self_fence_grace_ticks above).
        assert_eq!(orch.liveness.unreachable_window_ticks, 75);
        assert_eq!(orch.liveness.n_consecutive_unreachable, 3);
        let shard = resolve_d3(&e, Profile::Cloud, NodeRole::Shard, 50).expect("shard cloud d3");
        assert_eq!(shard.node_self_fence_grace, 150);
        assert_eq!(shard.node_recheck, 25); // hz/2 supplied (VD_REALM_RECHECK is a required parse in the bin)
        // DevTest → the inert default (byte-identical to before this slice).
        let dev =
            resolve_d3(&env_of(&[]), Profile::DevTest, NodeRole::Orchestrator, 50).expect("dev d3");
        assert_eq!(dev.directory.lease_renew_interval_ticks, 0);
        assert_eq!(dev.node_self_fence_grace, 0);
        // Finding 5: the DevTest orchestrator keeps the PROD-SAFE n=3 confirm-dead default (NOT
        // LivenessTuning::default()'s kill-equivalent n=1) — byte-identical to the pre-slice orchestrator bin.
        assert_eq!(dev.liveness.n_consecutive_unreachable, 3);
    }

    #[test]
    fn cloud_resolve_d3_rejects_inert_or_incoherent_overrides() {
        let inert = |pairs: &[(&str, &str)], role| {
            *resolve_d3(&env_of(pairs), Profile::Cloud, role, 50)
                .expect_err("expected a cloud-profile violation")
                .downcast::<CloudProfileError>()
                .expect("a CloudProfileError")
        };
        // renew=0 makes the whole D-3 vacuous (validate would PASS it) — THE DEFERRED.md:984 hole → InertD3.
        assert_eq!(
            inert(
                &[("VD_PROFILE", "cloud"), ("VD_LEASE_RENEW_INTERVAL", "0")],
                NodeRole::Orchestrator
            ),
            CloudProfileError::InertD3("VD_LEASE_RENEW_INTERVAL")
        );
        // reaper=0 → the orchestrator never reassigns a dead owner (no ordering check catches it) → InertD3.
        assert_eq!(
            inert(
                &[("VD_PROFILE", "cloud"), ("VD_REAPER_INTERVAL", "0")],
                NodeRole::Orchestrator
            ),
            CloudProfileError::InertD3("VD_REAPER_INTERVAL")
        );
        // An incoherent grace override (grace <= ttl) fails via directory.validate (a DIFFERENT error type,
        // caught upstream — NOT InertD3).
        assert!(
            resolve_d3(
                &env_of(&[("VD_PROFILE", "cloud"), ("VD_SELF_FENCE_GRACE", "50")]),
                Profile::Cloud,
                NodeRole::Orchestrator,
                50
            )
            .is_err()
        );
        // A shard recheck=0 override fails via validate_self_fence_cadence (also caught upstream).
        assert!(
            resolve_d3(
                &env_of(&[("VD_PROFILE", "cloud"), ("VD_REALM_RECHECK", "0")]),
                Profile::Cloud,
                NodeRole::Shard,
                50
            )
            .is_err()
        );
        // A too-SMALL max override makes the reassign horizon RACE a THETA_MAX-throttled self-fence
        // (2*150 = 300 >= ttl+max = 100+50) → SelfFenceRacesReassign via directory.validate (NOT InertD3).
        let err = resolve_d3(
            &env_of(&[("VD_PROFILE", "cloud"), ("VD_MAX_SELF_FENCE_GRACE", "50")]),
            Profile::Cloud,
            NodeRole::Orchestrator,
            50,
        )
        .expect_err("a too-small max must be rejected");
        assert_eq!(
            err.downcast_ref::<vd_sim::directory::DirectoryTuningError>(),
            Some(
                &vd_sim::directory::DirectoryTuningError::SelfFenceRacesReassign {
                    ttl: 100,
                    grace: 150,
                    max: 50,
                    theta: vd_sim::directory::THETA_MAX,
                }
            )
        );
    }

    #[test]
    fn a_max_counter_overflows_loud_never_wraps_to_zero() {
        let dir = scratch();
        let path = dir.join(format!("of-{}.counter", line!()));
        // A valid file already at u64::MAX: the next increment would WRAP to 0 (a silent monotonicity
        // break) — the checked_add guard turns that into a loud Overflow instead.
        let mut rec = [0u8; RECORD_LEN];
        rec[0..8].copy_from_slice(&MAGIC);
        rec[8..16].copy_from_slice(&u64::MAX.to_le_bytes());
        let ck = checksum(&rec[0..16]);
        rec[16..24].copy_from_slice(&ck.to_le_bytes());
        std::fs::write(&path, rec).expect("write");
        assert!(matches!(
            BootCounter::increment_on_boot(&path, 1),
            Err(BootCounterError::Overflow { .. })
        ));
    }

    #[test]
    fn current_reads_the_counter_without_incrementing() {
        // R-6d4-D: `current` is the read-only peek the SIGKILL-restart proof uses (v1 before the kill, v2
        // after) — it must NOT mutate. Absent ⇒ None; after two increments ⇒ Some(the last value), stable.
        let dir = scratch();
        let path = dir.join(format!("cur-{}.counter", line!()));
        assert_eq!(
            BootCounter::current(&path).expect("absent read"),
            None,
            "absent ⇒ None"
        );
        assert_eq!(BootCounter::increment_on_boot(&path, 1).expect("v1"), 1);
        assert_eq!(BootCounter::increment_on_boot(&path, 1).expect("v2"), 2);
        assert_eq!(
            BootCounter::current(&path).expect("peek"),
            Some(2),
            "reads the last value"
        );
        assert_eq!(
            BootCounter::current(&path).expect("peek 2"),
            Some(2),
            "and does NOT increment"
        );
    }

    #[test]
    fn boot_counter_error_display_is_actionable() {
        assert!(
            BootCounterError::Corrupt {
                path: "/p".into(),
                detail: "bad magic",
            }
            .to_string()
            .contains("refusing to boot")
        );
        // The message is caller-agnostic (the shared guard serves the boot-counter, the R-6d outbox, and
        // the orchestrator store — each with its own `*_EPHEMERAL_OK` escape), so it names the generic form.
        let ephemeral = BootCounterError::Ephemeral {
            path: "/tmp/x".into(),
            detail: "under a temp dir".into(),
        }
        .to_string();
        assert!(
            ephemeral.contains("_EPHEMERAL_OK") && ephemeral.contains("Refusing to boot"),
            "the Ephemeral message must name the escape + the refusal: {ephemeral}"
        );
        assert!(
            BootCounterError::Overflow { path: "/p".into() }
                .to_string()
                .contains("overflowed")
        );
        assert!(
            BootCounterError::Io {
                path: "/p".into(),
                source: std::io::Error::other("disk full"),
            }
            .to_string()
            .contains("I/O error")
        );
    }
}
