//! S3 (cloud-ready k3d) — the PURE liveness/readiness decision surface behind the `/healthz` + `/readyz`
//! probes. ONE shared module (HR3) whose fns the three bins' per-role publishers feed with values read from
//! their own World; each bin injects its own `HealthSource`, so the per-role difference is trait polymorphism,
//! never a `match` on a shard kind.
//!
//! DELIBERATELY Tier-A (100% region+branch): readiness gates whether k8s routes traffic to a pod, so a wrong
//! branch = route-to-a-zombie or never-join-the-Service — exactly the class HR5 exists to catch (io-prod's
//! region-floor coverage is not enough). Everything here is a pure fn over PRIMITIVES (`Duration`/`bool`/
//! `u64`): NO `Instant` (vd-node's clippy.toml bans `Instant::now()` even in tests) — the ops-plane bin stamps
//! the heartbeat time, computes the STALENESS (`Instant::now() - beat_at`), and passes that `Duration` in; no
//! `World` is touched. HR5 discipline: all branching is bitwise (`&`/`|`, one region, no short-circuit) so
//! `assert_eq!` covers both arms.

use std::time::Duration;

/// The two orthogonal health bits a probe reports. `live` drives k8s LIVENESS (restart the pod if false);
/// `ready` drives k8s READINESS (remove from the Service endpoints if false, but leave it running). A
/// booting-but-still-syncing node is LIVE yet NOT READY — restarting it would throw away warm state, so the
/// two must never be conflated.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct HealthReport {
    pub live: bool,
    pub ready: bool,
}

/// The operational tuning for the wedge detector: how many tick-periods without heartbeat progress before a
/// node is declared not-live. ONE reviewed struct (no inline literals), derived off `tick_hz` exactly like
/// `DirectoryTuning::cloud(tick_hz)`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ProbeTuning {
    pub stall_deadline_ticks: u32,
}

/// A mis-tuned probe budget — rejected LOUD at boot.
#[derive(Clone, Copy, Debug, PartialEq, Eq, thiserror::Error)]
pub enum ProbeTuningError {
    #[error(
        "VD_PROBE_STALL_TICKS must be >= 2: a single pacer overrun (a NORMAL event) must not trip liveness \
         and needlessly restart a healthy node"
    )]
    StallTicksTooSmall,
}

impl ProbeTuning {
    /// The shipped default: 20 tick-periods of wedge slack (400 ms @50Hz, 2 s @10Hz). It DWARFS a single-tick
    /// pacer overrun / GC or scheduler hiccup (a slow-but-progressing node is never killed — no flap) yet sits
    /// FAR below the D-3 self-fence grace (~150 ticks @50Hz), so a genuine wedge is caught long before the
    /// split-brain machinery could mis-fire. S4 invariant: keep it < the kubelet `periodSeconds ×
    /// failureThreshold` AND < `terminationGracePeriodSeconds`.
    pub const DEFAULT: ProbeTuning = ProbeTuning {
        stall_deadline_ticks: 20,
    };

    /// The wall-clock wedge deadline at `tick_hz` — `stall_deadline_ticks` × the `TickPacer` period
    /// (`1 s / tick_hz`). ONE physical quantity, ONE derivation (so a 10 Hz cloud shard and a 50 Hz dev shard
    /// both get 20 ticks of grace — the cloud profile needs no probe-specific default). `.max(1)` floors keep
    /// a degenerate `tick_hz`/`ticks` of 0 from yielding a zero deadline.
    #[must_use]
    pub fn stall_deadline(self, tick_hz: u32) -> Duration {
        (Duration::from_secs(1) / tick_hz.max(1)) * self.stall_deadline_ticks.max(1)
    }

    /// Reject a stall budget so small a single (normal) pacer overrun would trip liveness.
    ///
    /// # Errors
    /// [`ProbeTuningError::StallTicksTooSmall`] when `stall_deadline_ticks < 2`.
    pub fn validate(self) -> Result<(), ProbeTuningError> {
        if self.stall_deadline_ticks < 2 {
            return Err(ProbeTuningError::StallTicksTooSmall);
        }
        Ok(())
    }
}

/// Compile-time proof the shipped default passes [`ProbeTuning::validate`] — a bad default fails the BUILD,
/// not a test (mirrors the `DirectoryTuning::cloud` const assertion).
const _: () = assert!(
    ProbeTuning::DEFAULT.stall_deadline_ticks >= 2,
    "the default probe tuning must satisfy validate()"
);

/// Is the node LIVE — is the last heartbeat's `staleness` (`now - beat_at`, computed on the ops plane with a
/// saturating subtraction so an equal/future stamp reads zero) within `stall_deadline`? A frozen heartbeat
/// (a deadlocked/wedged `step_tick`) grows the staleness past the deadline. Pure Tier-A (no `Instant` here).
#[must_use]
pub fn is_live(staleness: Duration, stall_deadline: Duration) -> bool {
    staleness < stall_deadline
}

/// The D-3-shaped confirmed-staleness readiness predicate: the holder's last ownership-affirming round-trip
/// (`confirmed`) is no staler than `grace` LOCAL ticks. This is the SAME monotone staleness
/// `lease_self_fence_due` uses (its non-lapsed side), so readiness inherits the D-3 grace as its debounce and
/// does NOT snap back to Ready on a single lucky re-grant under an intermittent partition (the ~3 s Service
/// add/remove flap the raw authority bit would cause). `grace == 0` (disarmed) ⇒ never fresh. Bitwise `&`
/// (HR5). Saturating so a future-stamped `confirmed` reads fresh.
#[must_use]
pub fn is_confirmed_fresh(local_tick: u64, confirmed: u64, grace: u64) -> bool {
    (grace != 0) & (local_tick.saturating_sub(confirmed) <= grace)
}

/// Shard realm-authority readiness, correct in BOTH D-3 modes — ALWAYS gated on actually HOLDING the realm
/// (`held`), so a never-granted or self-fenced shard is never Ready (it owns no realm and would drop any
/// routed session). This closes the boot-window hole: at boot `RealmConfirmedAt` defaults to tick 0, which
/// [`is_confirmed_fresh`] reads as fresh for the first `grace` local ticks, so a `held`-blind predicate would
/// read Ready before the FIRST grant lands. ON TOP of `held`: when the self-fence is ARMED (`grace != 0`, the
/// cloud profile) readiness ALSO requires the last ownership-affirming round-trip to be confirmed-fresh
/// ([`is_confirmed_fresh`]) — the PARTITION-AWARE debounce that de-routes a partitioned holder and does not
/// snap back on a single lucky re-grant. When INERT (`grace == 0`, the dev / in-process default, where D-3 is
/// off and there IS no staleness concept) `held` alone decides. This now matches `lease_self_fence_due`'s own
/// `held &` gate (readiness and the self-fence are true duals). Branchless (HR5): `held & (is_confirmed_fresh
/// | grace == 0)` — the `grace == 0` disjunct restores the inert `held`-only path since `is_confirmed_fresh`
/// is vacuously false when disarmed.
#[must_use]
pub fn shard_authority_ready(held: bool, local_tick: u64, confirmed: u64, grace: u64) -> bool {
    held & (is_confirmed_fresh(local_tick, confirmed, grace) | (grace == 0))
}

/// Orchestrator readiness = its OWN serving capability (boot complete + tick loop running), NOT the
/// whole-cluster `cluster_bootstrapped()` latch — which would DEADLOCK a cold-start orchestrator (it reads
/// NotReady until a shard registers, but the shard must dial the orchestrator to register) and stay Ready
/// forever after every shard dies. Survives shard death: the orchestrator can still serve directory CAS,
/// which is exactly what lets a shard re-register.
#[must_use]
pub fn orch_ready(serving: bool) -> bool {
    serving
}

/// Shard readiness = clock-synced (a synced clock IMPLICITLY proves orch→shard reachability — closing the
/// no-per-peer-mesh-accessor gap) AND realm-authority-ready ([`shard_authority_ready`]: confirmed-fresh under
/// active D-3 so a partitioned shard self-de-routes, or authority-held when inert). Bitwise `&` = one region,
/// both operands always evaluated (HR5, no short-circuit).
#[must_use]
pub fn shard_ready(clock_synced: bool, authority_ready: bool) -> bool {
    clock_synced & authority_ready
}

/// Gateway readiness = clock-synced AND session capacity available (readiness gates on the SAME quantity the
/// admission gate rejects on). PARTITION-BLIND in S3 (the follower clock is monotone None→Some; a self-fenced
/// session still counts toward `len()`), so a fully-partitioned gateway still reads Ready — the shard-symmetric
/// partition-aware fix (`any_session_confirmed_within(grace)`) is a NAMED S4 blocker, stated here so no reader
/// believes the gateway self-de-routes on partition. Bitwise `&` (HR5).
#[must_use]
pub fn gateway_ready(clock_synced: bool, session_count: usize, max_sessions: usize) -> bool {
    clock_synced & (session_count < max_sessions)
}

/// Assemble the [`HealthReport`] from the ops-plane inputs. `live = draining || fresh-heartbeat`: a DELIBERATE
/// drain (SIGTERM → the final-fsync park stops stamping the heartbeat) reads LIVE so kubelet never SIGKILLs
/// mid-fsync and corrupts the durable state the drain exists to protect; a genuinely-wedged RUNNING loop
/// (not draining, stale heartbeat) reads not-live. A draining node is also never Ready — de-routed from the
/// Service on the shutdown edge. All bitwise (HR5).
#[must_use]
pub fn health_report(
    staleness: Duration,
    stall_deadline: Duration,
    ready: bool,
    draining: bool,
) -> HealthReport {
    HealthReport {
        live: draining | is_live(staleness, stall_deadline),
        ready: ready & !draining,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn is_live_boundary_both_sides() {
        let deadline = Duration::from_millis(100);
        // staleness just under the deadline → live.
        assert!(is_live(Duration::from_millis(99), deadline));
        // exactly at the deadline → NOT live (the `<` is strict).
        assert!(!is_live(Duration::from_millis(100), deadline));
        // past the deadline → not live.
        assert!(!is_live(Duration::from_millis(101), deadline));
        // zero staleness (a just-stamped heartbeat) → live.
        assert!(is_live(Duration::ZERO, deadline));
    }

    #[test]
    fn stall_deadline_is_derived_and_floored() {
        // ticks × (1s / hz): 20 × 20ms = 400ms @50Hz; 20 × 50ms = 1s @20Hz; 20 × 100ms = 2s @10Hz.
        assert_eq!(
            ProbeTuning::DEFAULT.stall_deadline(50),
            Duration::from_millis(400)
        );
        assert_eq!(
            ProbeTuning::DEFAULT.stall_deadline(20),
            Duration::from_secs(1)
        );
        assert_eq!(
            ProbeTuning::DEFAULT.stall_deadline(10),
            Duration::from_secs(2)
        );
        // tick_hz = 0 clamps to 1 (period = 1s) — never a zero deadline.
        assert_eq!(
            ProbeTuning::DEFAULT.stall_deadline(0),
            Duration::from_secs(20)
        );
        // stall_deadline_ticks = 0 clamps to 1.
        assert_eq!(
            ProbeTuning {
                stall_deadline_ticks: 0
            }
            .stall_deadline(50),
            Duration::from_millis(20)
        );
    }

    #[test]
    fn validate_rejects_below_two_accepts_at_two() {
        assert_eq!(
            ProbeTuning {
                stall_deadline_ticks: 1
            }
            .validate(),
            Err(ProbeTuningError::StallTicksTooSmall)
        );
        assert_eq!(
            ProbeTuning {
                stall_deadline_ticks: 0
            }
            .validate(),
            Err(ProbeTuningError::StallTicksTooSmall)
        );
        assert_eq!(
            ProbeTuning {
                stall_deadline_ticks: 2
            }
            .validate(),
            Ok(())
        );
        assert_eq!(ProbeTuning::DEFAULT.validate(), Ok(()));
    }

    #[test]
    fn is_confirmed_fresh_covers_disarmed_boundary_and_saturation() {
        // grace == 0 (disarmed) → never fresh, even at zero staleness.
        assert!(!is_confirmed_fresh(5, 5, 0));
        // local - confirmed == grace → fresh (the non-lapsed boundary).
        assert!(is_confirmed_fresh(10, 5, 5));
        // local - confirmed == grace + 1 → stale (lapsed).
        assert!(!is_confirmed_fresh(11, 5, 5));
        // confirmed in the future saturates to zero staleness → fresh.
        assert!(is_confirmed_fresh(3, 9, 5));
    }

    #[test]
    fn shard_authority_ready_requires_held_then_freshness_when_armed() {
        // ARMED (grace != 0): must HOLD the realm AND be confirmed-fresh.
        assert!(shard_authority_ready(true, 10, 8, 5)); // held + fresh (10-8<=5) → ready
        assert!(!shard_authority_ready(true, 20, 8, 5)); // held but stale (20-8>5) → NOT ready
        // ARMED but NOT held — the boot-window / never-granted hole: the default RealmConfirmedAt(0) makes
        // is_confirmed_fresh TRUE for the first `grace` local ticks, yet no realm is held ⇒ must be NOT ready
        // (else the shard joins the Service before its first grant and drops any routed session).
        assert!(!shard_authority_ready(false, 0, 0, 150)); // boot: local=0, confirmed=0 default, never granted
        assert!(!shard_authority_ready(false, 150, 0, 150)); // still inside grace, still never granted
        assert!(!shard_authority_ready(false, 10, 8, 5)); // fresh but not held → NOT ready
        // INERT (grace == 0): held alone decides; freshness is vacuously false.
        assert!(shard_authority_ready(true, 100, 0, 0)); // inert + held → ready
        assert!(!shard_authority_ready(false, 100, 0, 0)); // inert + not held → not ready
    }

    #[test]
    fn orch_ready_passes_through() {
        assert!(orch_ready(true));
        assert!(!orch_ready(false));
    }

    #[test]
    fn shard_ready_truth_table() {
        assert!(shard_ready(true, true));
        assert!(!shard_ready(true, false));
        assert!(!shard_ready(false, true));
        assert!(!shard_ready(false, false));
    }

    #[test]
    fn gateway_ready_capacity_boundary() {
        // clock synced + under capacity → ready.
        assert!(gateway_ready(true, 4, 5));
        // at capacity → not ready (the `<` is strict — same as the admission gate).
        assert!(!gateway_ready(true, 5, 5));
        // over capacity → not ready.
        assert!(!gateway_ready(true, 6, 5));
        // clock not synced → not ready regardless of capacity.
        assert!(!gateway_ready(false, 0, 5));
    }

    #[test]
    fn health_report_all_combinations() {
        let fresh = Duration::ZERO; // within the deadline ⇒ live
        let stale = Duration::from_secs(10); // past the deadline ⇒ not live
        let d = Duration::from_millis(100);
        // running + fresh heartbeat + ready + not draining → live & ready.
        assert_eq!(
            health_report(fresh, d, true, false),
            HealthReport {
                live: true,
                ready: true
            }
        );
        // running + fresh + NOT ready → live but not ready (booting/syncing).
        assert_eq!(
            health_report(fresh, d, false, false),
            HealthReport {
                live: true,
                ready: false
            }
        );
        // running + STALE heartbeat + not draining → wedged: NOT live.
        assert_eq!(
            health_report(stale, d, true, false),
            HealthReport {
                live: false,
                ready: true
            }
        );
        // DRAINING + stale heartbeat → live (the drain-safe fix: no SIGKILL mid-fsync) but NOT ready (de-routed).
        assert_eq!(
            health_report(stale, d, true, true),
            HealthReport {
                live: true,
                ready: false
            }
        );
        // draining also forces not-ready even if the underlying predicate is ready.
        assert_eq!(
            health_report(fresh, d, true, true),
            HealthReport {
                live: true,
                ready: false
            }
        );
    }
}
