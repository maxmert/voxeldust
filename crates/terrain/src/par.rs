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

use std::ops::Range;

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
    std::thread::scope(|s| {
        for (r, slot) in parts.into_iter().zip(slots.iter_mut()) {
            s.spawn(move || *slot = Some(f(r)));
        }
    });
    slots.into_iter().flatten().collect()
}

/// `f` over every index of `0..n` on `threads` threads, the results in index order — the vector one thread gives.
pub fn map<T: Send>(n: usize, threads: usize, f: impl Fn(usize) -> T + Sync) -> Vec<T> {
    per_range(n, threads, |r| r.map(&f).collect::<Vec<T>>())
        .into_iter()
        .flatten()
        .collect()
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
