//! Deterministic hashed collections.
//!
//! `HashMap`/`HashSet` with the default `RandomState` are BANNED in `sim`/`node`
//! (clippy `disallowed-types`): per-process random hash seeds make iteration order a
//! hidden input to anything that observes it, breaking seed-reproducible replay
//! (`docs/design/test_harness.md` §2). Where O(1) lookup matters and iteration order
//! is observable, use these fixed-seed aliases; otherwise prefer `BTreeMap`/`BTreeSet`.
//!
//! Determinism scope (honest): identical key/insertion sequences produce identical
//! iteration order within one pinned toolchain (`SipHasher13` with fixed zero keys via
//! `DefaultHasher::new()`). The chaos-replay gate runs twice in separate processes and
//! would catch any regression of this property.

use std::collections::{HashMap, HashSet};
use std::hash::BuildHasherDefault;

/// `BuildHasher` with a FIXED seed — same hashes in every process, every run.
pub type FixedSeedHasher = BuildHasherDefault<std::hash::DefaultHasher>;

/// Deterministic `HashMap`: use when O(1) lookup matters; `BTreeMap` otherwise.
pub type DetHashMap<K, V> = HashMap<K, V, FixedSeedHasher>;

/// Deterministic `HashSet`.
pub type DetHashSet<T> = HashSet<T, FixedSeedHasher>;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn identical_insertion_sequences_iterate_identically() {
        let build = |keys: &[u64]| -> Vec<u64> {
            let mut m: DetHashMap<u64, u64> = DetHashMap::default();
            for &k in keys {
                m.insert(k, k * 2);
            }
            m.keys().copied().collect()
        };
        let keys: Vec<u64> = (0..256).map(|i| i * 7919).collect();
        assert_eq!(build(&keys), build(&keys), "same inserts, same order");
    }

    #[test]
    fn det_set_behaves_as_a_set() {
        let mut s: DetHashSet<&str> = DetHashSet::default();
        assert!(s.insert("a"));
        assert!(!s.insert("a"));
        assert!(s.contains("a"));
        assert_eq!(s.len(), 1);
    }

    #[test]
    fn lookup_semantics_match_std() {
        let mut m: DetHashMap<String, u32> = DetHashMap::default();
        m.insert("x".into(), 1);
        m.insert("y".into(), 2);
        assert_eq!(m.get("x"), Some(&1));
        assert_eq!(m.remove("y"), Some(2));
        assert_eq!(m.get("y"), None);
    }
}
