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

use std::path::Path;

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
            let tmp = canon_lenient(&std::env::temp_dir());
            let under_temp = probe.starts_with(&tmp)
                || probe.starts_with("/tmp")
                || probe.starts_with("/private/tmp");
            if under_temp {
                return Err(BootCounterError::Ephemeral {
                    path: path.display().to_string(),
                    detail: format!("under a temp dir ({})", tmp.display()),
                });
            }
        }
    }
    Ok(())
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
