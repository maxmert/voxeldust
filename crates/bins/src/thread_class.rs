//! ★ THE SOLVE'S CLASS (ruling B91, 2026-10-12): a solve thread lowers its own scheduling class below the shard's tick
//! and the player's window, so the cores a solve takes are the cores nobody else needs. MEASURED before it: two far
//! planets' first solves (7 threads each) filled the owner's 14 cores while the ground loaded, and the window's frame
//! waited for a core — 247 late frames in a minute's login against 2 with the solving shards lowered by hand, the
//! load 30 s in both. The bytes a solve gives do not depend on its class.
//!
//! The class is the operating system's word, set through `libc` (approved for the shard program by the same ruling):
//! - macOS: the "utility" quality-of-service class — below the default class every other thread runs in, above
//!   "background" (which would also slow the solve's store writes and keep it off the fast cores even on a quiet
//!   machine);
//! - Linux: a nice value of [`LINUX_SOLVE_NICE`] on the calling thread (the nice value is per thread there);
//! - any other system: the default class, unchanged.
//!
//! A thread does NOT pass its class to the threads it starts on macOS (MEASURED 2026-10-12: a child of a "utility"
//! thread starts at "default"), so a solve thread adopts [`lower_to_solve_class`] as the generator's start step
//! (`vd_terrain::par::adopt_start_step`) and every helper thread of the solve runs it first.
//!
//! **Example.** The desert world's shard starts its first solve: the solve thread and each of its seven helpers lower
//! themselves; the player flies hull 15 over the belt and the window's frame takes a core from the desert world's
//! solve, never the other way round; when the player lands and the machine is quiet, the solve runs on every core it
//! was given.

/// ★ THE SOLVE'S NICE VALUE ON LINUX: 10, the conventional batch value — a thread at it gets about a tenth of a core's
/// time against a thread at the default 0 when both want it, and every free core when nobody else does. Raising a
/// nice value needs no privilege, so a pod can always do it.
pub const LINUX_SOLVE_NICE: i32 = 10;

/// Lower the CALLING thread's scheduling class to the solve's (see the module). A refusal by the system is logged and
/// leaves the thread in its class: the solve then competes as before, it never stops.
#[cfg(target_os = "macos")]
pub fn lower_to_solve_class() {
    // SAFETY: a system call on the calling thread with constant arguments; it reads and writes no Rust memory.
    let rc =
        unsafe { libc::pthread_set_qos_class_self_np(libc::qos_class_t::QOS_CLASS_UTILITY, 0) };
    refused(rc);
}

/// Lower the CALLING thread's scheduling class to the solve's (see the module). A refusal by the system is logged and
/// leaves the thread in its class: the solve then competes as before, it never stops.
#[cfg(target_os = "linux")]
pub fn lower_to_solve_class() {
    // `who = 0` names the calling THREAD: under Linux's threads the nice value is a per-thread attribute.
    // SAFETY: a system call on the calling thread with constant arguments; it reads and writes no Rust memory.
    let rc = unsafe { libc::setpriority(libc::PRIO_PROCESS, 0, LINUX_SOLVE_NICE) };
    refused(rc);
}

/// Any other system: the solve keeps the default class.
#[cfg(not(any(target_os = "macos", target_os = "linux")))]
pub fn lower_to_solve_class() {}

/// Log a system's refusal (a non-zero answer).
#[cfg(any(target_os = "macos", target_os = "linux"))]
fn refused(rc: libc::c_int) {
    if rc != 0 {
        tracing::warn!(
            rc,
            "a solve thread could not lower its scheduling class; it competes with the tick and the window"
        );
    }
}

#[cfg(all(test, target_os = "macos"))]
mod tests {
    use super::*;

    unsafe extern "C" {
        fn qos_class_self() -> libc::qos_class_t;
    }

    /// The calling thread's class, as the system reads it.
    fn class() -> u32 {
        // SAFETY: a read of the calling thread's class; no arguments, no Rust memory.
        unsafe { qos_class_self() as u32 }
    }

    /// ★ A SOLVE THREAD AND EVERY HELPER IT STARTS RUN IN THE UTILITY CLASS (ruling B91): a plain thread starts at the
    /// default class; once it adopts the solve's class as the generator's start step, it and each helper of a step run
    /// at "utility"; a refusal is only a log line.
    #[test]
    fn a_solve_thread_and_its_helpers_run_below_the_default_class() {
        let utility = libc::qos_class_t::QOS_CLASS_UTILITY as u32;
        std::thread::spawn(move || {
            assert_eq!(class(), libc::qos_class_t::QOS_CLASS_DEFAULT as u32);
            vd_terrain::par::adopt_start_step(lower_to_solve_class);
            assert_eq!(class(), utility);
            assert_eq!(vd_terrain::par::map(4, 4, |_| class()), vec![utility; 4]);
            refused(1);
        })
        .join()
        .expect("the test's own thread");
    }
}
