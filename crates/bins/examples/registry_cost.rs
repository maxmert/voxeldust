//! The registry's one measurement (the voxel foundation, slice 3): the identity digest's cost at a
//! store open and at a handshake, which must stay under 200 microseconds each because both already
//! carry budgets in milliseconds. Run in release:
//!
//! ```text
//! cargo run --release -p vd-bins --example registry_cost
//! ```

use std::time::Instant;

use vd_core::registry::{ATTACHMENTS, KINDS, RegistryStamp};

fn main() {
    let rounds = 2_000u32;
    // The store open: this build stamps its own tables, then checks a stored stamp against them.
    let stored = RegistryStamp::current();
    let start = Instant::now();
    for _ in 0..rounds {
        let stamp = RegistryStamp::current();
        assert_eq!(stamp, stored);
        RegistryStamp::accepts(&stored).expect("the store this build wrote opens");
    }
    let open_us = start.elapsed().as_secs_f64() * 1e6 / f64::from(rounds);
    // The handshake: one prefix digest of the peer's stated lengths.
    let start = Instant::now();
    for _ in 0..rounds {
        RegistryStamp::accepts(&stored).expect("a peer with the same registry");
    }
    let handshake_us = start.elapsed().as_secs_f64() * 1e6 / f64::from(rounds);
    println!(
        "registry_cost: {} kinds, {} attachment kinds; open {open_us:.1} us (stamp + accept), handshake {handshake_us:.1} us (accept) — gate 200 us each",
        KINDS.len(),
        ATTACHMENTS.len()
    );
    // The gate can fail: a store open and a handshake each carry a budget in milliseconds.
    assert!(
        open_us < 200.0,
        "the store-open digest exceeds 200 us: {open_us:.1}"
    );
    assert!(
        handshake_us < 200.0,
        "the handshake digest exceeds 200 us: {handshake_us:.1}"
    );
    println!("registry_cost: PASS");
}
