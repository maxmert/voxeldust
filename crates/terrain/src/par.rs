//! ★ THE SOLVE'S CORES (ruling B64, 2026-10-07: *"Can we give more cores for passes to build the realm for the first
//! time?"*). The passes of a first solve stay one after another — each reads the ground the one before left — but the
//! work inside a step runs on many cores, and gives THE SAME BYTES on any number of them: the nodes are cut into
//! contiguous ranges, each thread computes only its own range, and the ranges are joined in their order, so every value
//! is computed by the same arithmetic in the same order one core uses. The standard library's scoped threads, no other
//! dependency (the generator's rule: the recipe, the seed leaf, nothing else).
//!
//! **Example.** The climate of the home planet's 8.9 million nodes on ten threads: thread 3 computes nodes 2 661 120 to
//! 3 548 159 into its own vector, and the ten vectors are joined in order — the very vector one thread gives.

// ★ THE CORES' RANGES DIVIDE: they run on the CPU once per step, never in a kernel.
#![allow(
    clippy::integer_division,
    reason = "the ranges are cut on the CPU, never in a kernel; integer-exact on every host"
)]

use std::cell::Cell;
use std::ops::Range;

thread_local! {
    /// The start step this thread hands to the helper threads its steps start ([`adopt_start_step`]). `None` on every
    /// thread no host gave one — the client's, a test's, a tick's.
    static START_STEP: Cell<Option<fn()>> = const { Cell::new(None) };
}

/// ★ THE HOST'S START STEP FOR A SOLVE'S THREADS (ruling B91, 2026-10-12): runs `step` on this thread now, and on every
/// helper thread this thread's steps start, and on theirs. The generator calls no system itself (its rule: the recipe,
/// the seed leaf, nothing else), so the HOST hands it the step — the shard's lowers a thread below the tick and the
/// window. A host must hand it down: on macOS a thread's class is NOT inherited (MEASURED 2026-10-12: a child of a
/// "utility" thread starts at "default"). The bytes are the same whatever the step does to a thread's class.
///
/// **Example.** The planet's shard starts the desert world's first solve on its solve thread and adopts the step that
/// lowers a thread; each of the climate's seven helper threads runs it before its first node, so a frame of the window
/// or a tick of the shard gets a core before any of them.
pub fn adopt_start_step(step: fn()) {
    step();
    START_STEP.with(|s| s.set(Some(step)));
}

/// The start step this thread carries, read before a step starts its helpers.
fn inherited_step() -> Option<fn()> {
    START_STEP.with(Cell::get)
}

/// A helper thread's first act: adopt the step its parent carried, if it carried one.
fn begin_helper(step: Option<fn()>) {
    if let Some(step) = step {
        adopt_start_step(step);
    }
}

/// ★ HOW MANY THREADS A STEP OF THE SOLVE MAY USE — an operational word, never part of the world: two states that differ
/// only in their cores are the same state (their equality ignores this word), because the same bytes come out on any
/// number of them. `Cores(0)` and `Cores(1)` both run on the caller's own thread alone.
#[derive(Clone, Copy, Debug, Default)]
pub struct Cores(pub usize);

impl Cores {
    /// The threads a step uses: at least one.
    #[must_use]
    pub fn threads(self) -> usize {
        self.0.max(1)
    }
}

impl PartialEq for Cores {
    /// Always equal: the cores decide how fast a state is reached, never which state.
    fn eq(&self, _: &Cores) -> bool {
        true
    }
}

/// `0..n` cut into at most `threads` contiguous ranges in order, the first `n mod parts` one longer; none for `n = 0`.
#[must_use]
pub fn ranges(n: usize, threads: usize) -> Vec<Range<usize>> {
    let parts = threads.clamp(1, n.max(1));
    let (base, extra) = (n / parts, n % parts);
    let mut out = Vec::with_capacity(parts);
    let mut start = 0;
    for p in 0..parts {
        let len = base + usize::from(p < extra);
        out.push(start..start + len);
        start += len;
    }
    out.retain(|r| !r.is_empty());
    out
}

/// `f` over every range of `0..n` on `threads` threads, each range's own result in range order: the step's work over a
/// range, whatever it returns (a value per index, a sum, a scratch of its own). Each thread writes only its own slot;
/// the scope joins them all, and a thread's panic is the caller's.
pub fn per_range<T: Send>(
    n: usize,
    threads: usize,
    f: impl Fn(Range<usize>) -> T + Sync,
) -> Vec<T> {
    let parts = ranges(n, threads);
    let mut slots: Vec<Option<T>> = parts.iter().map(|_| None).collect();
    let f = &f;
    let step = inherited_step();
    std::thread::scope(|s| {
        for (r, slot) in parts.into_iter().zip(slots.iter_mut()) {
            s.spawn(move || {
                begin_helper(step);
                *slot = Some(f(r));
            });
        }
    });
    slots.into_iter().flatten().collect()
}

/// `f` over every index of `0..n` on `threads` threads, the results in index order — the vector one thread gives.
pub fn map<T: Send>(n: usize, threads: usize, f: impl Fn(usize) -> T + Sync) -> Vec<T> {
    join(per_range(n, threads, |r| r.map(&f).collect::<Vec<T>>()))
}

/// ★ `f` over every index of `0..n` on `threads` threads TAKEN IN TURN: each thread takes the next index when it finishes
/// one, so an index that takes long holds only its own thread, never a whole range; the results in index order — the
/// vector one thread gives (each result is its own index's alone).
///
/// **Example.** The seasons' climate runs its sixteen seasons on eight threads: a winter whose mountains' rain needs many
/// rounds keeps one thread busy while the other seven finish the remaining seasons.
pub fn map_in_turn<T: Send>(n: usize, threads: usize, f: impl Fn(usize) -> T + Sync) -> Vec<T> {
    let next = std::sync::atomic::AtomicUsize::new(0);
    let workers = threads.clamp(1, n.max(1));
    let (f, next) = (&f, &next);
    let step = inherited_step();
    let mut parts: Vec<Vec<(usize, T)>> = (0..workers).map(|_| Vec::new()).collect();
    std::thread::scope(|s| {
        for part in &mut parts {
            s.spawn(move || {
                begin_helper(step);
                loop {
                    let i = next.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                    if i >= n {
                        break;
                    }
                    part.push((i, f(i)));
                }
            });
        }
    });
    let mut all: Vec<(usize, T)> = parts.into_iter().flatten().collect();
    all.sort_unstable_by_key(|(i, _)| *i);
    all.into_iter().map(|(_, t)| t).collect()
}

/// The ranges' parts joined in their order into one vector — the vector one thread gives. Each part moves into the
/// answer whole (ruling B72 item 1: joining the home planet's 8.9 million values one at a time was a twentieth of a
/// solve's time on twelve cores, the profile of 2026-10-09).
pub fn join<T>(parts: Vec<Vec<T>>) -> Vec<T> {
    let mut out = Vec::with_capacity(parts.iter().map(Vec::len).sum());
    for mut part in parts {
        out.append(&mut part);
    }
    out
}

/// ★ UNIQUE KEYS SORTED ON `threads` THREADS (ruling B72 item 1): each thread sorts one range of the keys, and the
/// sorted runs are merged by `threads` threads, each writing the keys between two SPLITTERS (drawn evenly from the runs)
/// into its own part of the answer. Each key carries its node's index, so the keys are unique and exactly one order
/// sorts them: the order one thread's sort gives, byte for byte.
///
/// **Example.** The lakes' budget orders the home planet's 2.1 million land nodes by height, by their potential on a
/// flat, and by index: seven threads sort 300 000 keys each, and seven more merge the seven runs between six splitters.
#[must_use]
pub fn sort_keys(mut keys: Vec<u128>, threads: usize) -> Vec<u128> {
    let parts = ranges(keys.len(), threads);
    let step = inherited_step();
    std::thread::scope(|s| {
        let mut rest: &mut [u128] = &mut keys;
        for r in &parts {
            let (run, tail) = rest.split_at_mut(r.len());
            rest = tail;
            s.spawn(move || {
                begin_helper(step);
                run.sort_unstable();
            });
        }
    });
    let runs: Vec<&[u128]> = parts.iter().map(|r| &keys[r.clone()]).collect();
    let count = runs.len();
    // The splitters: from each run as many keys as there are threads, evenly spaced, and of all of them every run's
    // count-th — so each part of the answer holds about one run's worth of keys.
    let mut samples: Vec<u128> = runs
        .iter()
        .flat_map(|run| (0..count).map(move |k| run[k * run.len() / count]))
        .collect();
    samples.sort_unstable();
    let splitters: Vec<u128> = (1..count).map(|t| samples[t * count]).collect();
    // Each part: the keys at or over the splitter before it and under the splitter after it, from every run.
    let bounds: Vec<Vec<usize>> = runs
        .iter()
        .map(|run| {
            std::iter::once(0)
                .chain(splitters.iter().map(|&x| run.partition_point(|&k| k < x)))
                .chain(std::iter::once(run.len()))
                .collect()
        })
        .collect();
    let mut out = vec![0u128; keys.len()];
    std::thread::scope(|s| {
        let mut rest: &mut [u128] = &mut out;
        for t in 0..count {
            let pieces: Vec<&[u128]> = runs
                .iter()
                .zip(&bounds)
                .map(|(run, b)| &run[b[t]..b[t + 1]])
                .collect();
            let (part, tail) = rest.split_at_mut(pieces.iter().map(|p| p.len()).sum());
            rest = tail;
            s.spawn(move || {
                begin_helper(step);
                merge_into(&pieces, part);
            });
        }
    });
    out
}

/// The sorted `pieces` merged into `out`, which holds exactly their keys: at each place the least of the pieces' heads.
fn merge_into(pieces: &[&[u128]], out: &mut [u128]) {
    let mut head = vec![0usize; pieces.len()];
    for slot in out.iter_mut() {
        let mut best = usize::MAX;
        for (p, piece) in pieces.iter().enumerate() {
            let take = (head[p] < piece.len())
                && ((best == usize::MAX) || (piece[head[p]] < pieces[best][head[best]]));
            best = [best, p][usize::from(take)];
        }
        *slot = pieces[best][head[best]];
        head[best] += 1;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The cores are an operational word: at least one thread, and two states with other cores are equal.
    #[test]
    fn the_cores_are_never_part_of_the_world() {
        assert_eq!(Cores(0).threads(), 1);
        assert_eq!(Cores(6).threads(), 6);
        assert_eq!(Cores(1), Cores(9));
    }

    /// The ranges cover `0..n` in order with no gap and no overlap, at most `threads` of them, their lengths one apart;
    /// no thread at all and more threads than items still cover the items once.
    #[test]
    fn the_ranges_cover_the_nodes_once_in_order() {
        assert_eq!(ranges(10, 3), vec![0..4, 4..7, 7..10]);
        assert_eq!(ranges(10, 1), vec![0..10]);
        assert_eq!(ranges(10, 0), vec![0..10]);
        assert_eq!(ranges(3, 8), vec![0..1, 1..2, 2..3]);
        assert_eq!(ranges(0, 4), Vec::<Range<usize>>::new());
    }

    /// The map on many threads is the map on one, value for value and in order; a range's own result too.
    #[test]
    fn many_threads_give_the_bytes_one_gives() {
        let f = |i: usize| (i * 2_654_435_761) % 1_000_003;
        let one = map(10_000, 1, f);
        assert_eq!(map(10_000, 7, f), one);
        assert_eq!(one.len(), 10_000);
        let sums = per_range(10_000, 4, |r| r.map(f).sum::<usize>());
        assert_eq!(sums.len(), 4);
        assert_eq!(sums.iter().sum::<usize>(), one.iter().sum::<usize>());
        assert_eq!(map(0, 3, f), Vec::<usize>::new());
    }

    /// The map taken in turn is the map on one thread, value for value and in order, on any number of threads; no item
    /// gives an empty vector.
    #[test]
    fn indices_taken_in_turn_give_the_bytes_one_gives() {
        let f = |i: usize| (i * 2_654_435_761) % 1_000_003;
        let one = map(1_000, 1, f);
        assert_eq!(map_in_turn(1_000, 1, f), one);
        assert_eq!(map_in_turn(1_000, 7, f), one);
        assert_eq!(map_in_turn(3, 16, f), map(3, 1, f));
        assert_eq!(map_in_turn(0, 4, f), Vec::<usize>::new());
    }

    /// ★ The key sort on many threads is the one-thread sort, key for key: few keys and many, fewer keys than threads,
    /// none, and keys that crowd into a few values above their index bits.
    #[test]
    fn the_key_sort_on_many_threads_is_the_one_thread_sort() {
        let hash = |i: u64| {
            let mut x = i.wrapping_mul(0x9e37_79b9_7f4a_7c15);
            x ^= x >> 31;
            x.wrapping_mul(0xbf58_476d_1ce4_e5b9)
        };
        for (len, spread) in [(100_000u64, 8u32), (100_000, 61), (5, 8), (1, 8), (0, 8)] {
            let keys: Vec<u128> = (0..len)
                .map(|i| (u128::from(hash(i) >> spread) << 32) | u128::from(i))
                .collect();
            let mut want = keys.clone();
            want.sort_unstable();
            for threads in [1, 2, 3, 7, 14] {
                assert_eq!(
                    sort_keys(keys.clone(), threads),
                    want,
                    "{len} keys, {threads} threads"
                );
            }
        }
    }

    /// How many times the counting step ran — on the one test thread that adopts it and its helpers only.
    static STEPS: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);

    fn count_step() {
        STEPS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    }

    fn steps() -> usize {
        STEPS.load(std::sync::atomic::Ordering::Relaxed)
    }

    /// ★ THE HOST'S START STEP (ruling B91) runs on the thread that adopts it and on every helper its steps start — the
    /// map's, the sort's runs and merges, a helper's own helpers — and on no thread that never adopted it.
    #[test]
    fn every_helper_runs_the_hosts_start_step() {
        std::thread::spawn(|| {
            assert_eq!(map(8, 3, |i| i).len(), 8);
            assert_eq!(steps(), 0, "a thread with no step starts plain helpers");
            adopt_start_step(count_step);
            assert_eq!(steps(), 1, "the adopting thread runs it at once");
            assert_eq!(map(9, 3, |i| i), (0..9).collect::<Vec<_>>());
            assert_eq!(steps(), 4, "each of the map's three helpers");
            let keys: Vec<u128> = (0..10u128).rev().collect();
            assert_eq!(sort_keys(keys, 2), (0..10u128).collect::<Vec<_>>());
            assert_eq!(steps(), 8, "the sort's two runs and its two merges");
            assert_eq!(per_range(2, 2, |_| map(4, 2, |i| i).len()), vec![4, 4]);
            assert_eq!(steps(), 14, "two helpers, and each one's own two");
        })
        .join()
        .expect("the test's own thread");
    }

    /// A panic in a solve thread is the caller's panic.
    #[test]
    #[should_panic(expected = "a scoped thread panicked")]
    fn a_threads_panic_reaches_the_caller() {
        let _ = map(4, 2, |i| {
            assert!(i != 3, "a node's own failure");
            i
        });
    }
}
