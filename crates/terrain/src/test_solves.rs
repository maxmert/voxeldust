//! ★ THE SHARED TEST SOLVES (ruling B74 item 2, 2026-10-10): a test that needs a solved body reads the ONE solve this
//! test binary makes of that body with those words on that schedule — made once, on all the host's cores, and handed to
//! every test as its own copy. The solve is the game's own (`solve_full_on`), and the cores give the bytes one core gives
//! (the test `the_cores_give_the_same_bytes`), so a test reads the very state it read when it solved alone. MEASURED
//! before (the coverage gate of 2026-10-09/10): the terrain and client crates' tests solved the moon about 36 times, each
//! on one core, about 15 hours under the coverage build.
//!
//! **Example.** Twelve tests read the dry moon with its own words: the first to ask solves it on fourteen threads while
//! the other eleven wait for that one solve, and each gets its copy — one solve where there were twelve.

use std::collections::BTreeMap;
use std::sync::{Arc, Mutex, OnceLock};

use crate::body::BodyDefinition;
use crate::solve::{FullReport, MacroSolve, Schedule, SolveWords, solve_full_on};

/// A solved body: the state, the facies bytes and the report ([`solve_full_on`]).
type Solved = (MacroSolve, Vec<u8>, FullReport);

/// The solve of `body` with `words` on `schedule`, shared by every test of this binary that asks for it (see the module).
pub(crate) fn solved(body: &BodyDefinition, words: &SolveWords, schedule: Schedule) -> Solved {
    static SOLVES: Mutex<BTreeMap<String, Arc<OnceLock<Solved>>>> = Mutex::new(BTreeMap::new());
    let key = format!("{}|{words:?}|{schedule:?}", body.seed());
    let cell = Arc::clone(
        SOLVES
            .lock()
            .expect("the shared solves' book")
            .entry(key)
            .or_default(),
    );
    let cores = crate::par::Cores(std::thread::available_parallelism().map_or(1, usize::from));
    cell.get_or_init(|| solve_full_on(body, words, schedule, cores).expect("a solve"))
        .clone()
}
