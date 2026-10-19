//! ★ THE SOLVE'S CLASS (ruling B91, 2026-10-12): a solve thread lowers its own scheduling class below the shard's tick
//! and the player's window, so the cores a solve takes are the cores nobody else needs. MEASURED before it: two far
//! planets' first solves (7 threads each) filled the owner's 14 cores while the ground loaded, and the window's frame
//! waited for a core — 247 late frames in a minute's login against 2 with the solving shards lowered by hand, the
//! load 30 s in both. The bytes a solve gives do not depend on its class.
//!
//! The class is the operating system's word, set through `libc` (approved for the shard program by the same ruling):
//! - macOS: ★ THE DEFAULT QUALITY-OF-SERVICE CLASS AT ITS LOWEST RELATIVE PRIORITY, [`MACOS_SOLVE_RELATIVE_PRIORITY`]
//!   (ruling B117, 2026-10-19; amends B91's "utility"). MEASURED on the owner's machine (M4 Pro, 10 fast and 4 efficiency
//!   cores): the "utility" class kept every solve thread on the four efficiency cores while the ten fast cores stood idle
//!   — the home planet's seven solve threads at 51–53 % of a core each, a 128 m tile 17–32 s against 5.5 s on the bench;
//!   seven test spinners at "utility" got 0.26 of a core each, at the default class 0.92. At the default class with the
//!   lowest relative priority a spinner gets 0.91 of a core alone, and beside ten busy default threads 0.35 while they get
//!   0.68 each: the solve takes the idle fast cores and yields about two thirds of a contested core to the window and
//!   the tick;
//! - Linux: a nice value of [`LINUX_SOLVE_NICE`] on the calling thread (the nice value is per thread there);
//! - any other system: the default class, unchanged.
//!
//! A thread does NOT pass its class to the threads it starts on macOS (MEASURED 2026-10-12: a child of a "utility"
//! thread starts at "default"), so a solve thread adopts [`lower_to_solve_class`] as the generator's start step
//! (`vd_terrain::par::adopt_start_step`) and every helper thread of the solve runs it first.
//!
//! **Example.** The desert world's shard starts its first solve: the solve thread and each of its seven helpers lower
//! themselves; the player flies hull 15 over the belt and the window's frame takes the larger share of a contested core
//! from the desert world's solve; when the player lands and the machine is quiet, the solve runs on the fast cores it
//! was given.

/// ★ THE SOLVE'S RELATIVE PRIORITY ON macOS (ruling B117): the lowest the system takes inside a class (its
/// `QOS_MIN_RELATIVE_PRIORITY`), applied in the default class so the solve keeps the fast cores.
#[cfg(target_os = "macos")]
pub const MACOS_SOLVE_RELATIVE_PRIORITY: libc::c_int = -15;

/// ★ THE SOLVE'S NICE VALUE ON LINUX: 10, the conventional batch value — a thread at it gets about a tenth of a core's
/// time against a thread at the default 0 when both want it, and every free core when nobody else does. Raising a
/// nice value needs no privilege, so a pod can always do it.
pub const LINUX_SOLVE_NICE: i32 = 10;

/// Lower the CALLING thread's scheduling class to the solve's (see the module). A refusal by the system is logged and
/// leaves the thread in its class: the solve then competes as before, it never stops.
#[cfg(target_os = "macos")]
pub fn lower_to_solve_class() {
    // SAFETY: a system call on the calling thread with constant arguments; it reads and writes no Rust memory.
    let rc = unsafe {
        libc::pthread_set_qos_class_self_np(
            libc::qos_class_t::QOS_CLASS_DEFAULT,
            MACOS_SOLVE_RELATIVE_PRIORITY,
        )
    };
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

    /// The calling thread's class and relative priority, as the system reads them.
    fn class() -> (u32, libc::c_int) {
        let mut class = libc::qos_class_t::QOS_CLASS_UNSPECIFIED;
        let mut relative = 0;
        // SAFETY: a read of the calling thread's own class into two locals that outlive the call.
        let rc = unsafe {
            libc::pthread_get_qos_class_np(libc::pthread_self(), &raw mut class, &raw mut relative)
        };
        assert_eq!(rc, 0, "the system reads a thread's class");
        (class as u32, relative)
    }

    /// ★ A SOLVE THREAD AND EVERY HELPER IT STARTS RUN AT THE SOLVE'S PRIORITY (rulings B91, B117): a plain thread starts
    /// at the default class and relative priority nought; once it adopts the solve's class as the generator's start step,
    /// it and each helper of a step run in the default class at the lowest relative priority; a refusal is only a log
    /// line.
    #[test]
    fn a_solve_thread_and_its_helpers_run_below_the_default_class() {
        let default = libc::qos_class_t::QOS_CLASS_DEFAULT as u32;
        let solve = (default, MACOS_SOLVE_RELATIVE_PRIORITY);
        std::thread::spawn(move || {
            assert_eq!(class(), (default, 0));
            vd_terrain::par::adopt_start_step(lower_to_solve_class);
            assert_eq!(class(), solve);
            assert_eq!(vd_terrain::par::map(4, 4, |_| class()), vec![solve; 4]);
            refused(1);
        })
        .join()
        .expect("the test's own thread");
    }
}
