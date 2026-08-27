//! THE ONE CONTENT DIGEST in the tree — FNV-1a, 64-bit.
//!
//! ★ WHY IT IS ITS OWN MODULE (slice S11). The same fold existed in two places: the saved-data
//! label's generation (`store_stamp`) and the window lane's send-on-change baselines (`vd-sim`).
//! Two copies of one arithmetic is two chances for them to stop being the same arithmetic, and a
//! digest that quietly differs between two callers is the kind of defect that shows up as "the
//! client keeps re-downloading the sky" long after the change that caused it.
//!
//! **`const`, deliberately.** `coordinate_generation` is a `pub const fn` and folds this at compile
//! time; a non-const helper would have forced a second definition again. A `const fn` is callable
//! from ordinary code too, so one definition serves both.
//!
//! **Dependency-free and stable across builds and machines**, which is load-bearing rather than
//! tidy: a generation that changed with the compiler would refuse every saved store after a
//! toolchain upgrade, and would re-issue the whole star catalogue to every player for nothing.
//!
//! It is a CONTENT digest, never a security one. A collision costs a deferred re-send or a stale
//! cache, never a wrong byte accepted as right — every caller states its own consequence.

/// FNV-1a's 64-bit offset basis — where a fold with nothing folded into it starts.
pub const FNV_OFFSET: u64 = 0xcbf2_9ce4_8422_2325;
/// FNV-1a's 64-bit prime.
const FNV_PRIME: u64 = 0x0000_0100_0000_01b3;

/// Fold `bytes` into `acc`. Chain the calls to digest several parts as one value.
///
/// An index loop rather than a `for`, because a `for` is not permitted in a `const fn`.
#[must_use]
pub const fn fnv1a(mut acc: u64, bytes: &[u8]) -> u64 {
    let mut i = 0;
    while i < bytes.len() {
        acc ^= bytes[i] as u64;
        acc = acc.wrapping_mul(FNV_PRIME);
        i += 1;
    }
    acc
}

/// One 64-bit value folded in, little-endian — the shape the saved-data label folds its numbers in.
#[must_use]
pub const fn fnv1a_u64(acc: u64, v: u64) -> u64 {
    fnv1a(acc, &v.to_le_bytes())
}
