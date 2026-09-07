//! U-4 — the cost of one address lookup, on three round bodies spanning the legal size range and
//! on a hull (the voxel foundation, slice 2). Prints microseconds per call side by side, so
//! constancy in the body's size is VISIBLE rather than asserted. Run in release:
//!
//! ```text
//! cargo run --release -p vd-bins --example grid_cost
//! ```
//!
//! "Off the containment path" is not measured here: it is a structural fence owed with the mesher
//! (the first consumer), the same shape SL4 uses.

use std::time::Instant;

use vd_core::glam::DVec3;
use vd_core::grid::{BandParams, GridMapping, IdentityGrid, Rung, ShellGrid};
use vd_core::ids::EntityId;
use vd_core::pose::{LatticePos, RealmId, Tier};

const SAMPLES: u64 = 1_000_000;

/// A deterministic walk over the surface band of a body of radius `r`.
fn surface_samples(r: f64) -> Vec<LatticePos> {
    let mut samples = Vec::with_capacity(SAMPLES as usize);
    for s in 0..SAMPLES {
        let t = (s as f64) * 0.000_618_033_988_749_9 % 1.0; // golden-ratio walk over [0, 1)
        let u = (s as f64) * 0.000_141_421_356_237_3 % 1.0;
        let theta = t * std::f64::consts::TAU;
        let phi = (u * 2.0 - 1.0).acos();
        let radius = r + (s % 40) as f64 - 20.0;
        let dir = DVec3::new(phi.sin() * theta.cos(), phi.sin() * theta.sin(), phi.cos());
        samples.push(LatticePos::from_metres(dir * radius, Tier::Fine));
    }
    samples
}

/// Microseconds per `addr_of` over `samples`, and how many landed inside the grid.
fn time_lookups(grid: &GridMapping, body: RealmId, samples: &[LatticePos]) -> (f64, u64) {
    let rung = Rung::ZERO;
    let mut hits = 0u64;
    let start = Instant::now();
    for &p in samples {
        if grid.addr_of(body, p, rung).is_some() {
            hits += 1;
        }
    }
    (
        start.elapsed().as_secs_f64() * 1e6 / samples.len() as f64,
        hits,
    )
}

fn main() {
    let planet = RealmId::Planet(7);
    // Three round bodies across the legal range: a 500 m asteroid, Earth, the largest legal body.
    let largest = f64::from(1u32 << 26) * std::f64::consts::FRAC_2_PI;
    let mut planet_us = Vec::new();
    for (name, r) in [
        ("asteroid 500 m", 500.0),
        ("Earth 6 371 km", 6_371_000.0),
        ("largest 42 723 km", largest),
    ] {
        let grid = GridMapping::CubeSphere(
            ShellGrid::for_body(r, BandParams::provisional(r)).expect("a legal body"),
        );
        let samples = surface_samples(r);
        let (us, hits) = time_lookups(&grid, planet, &samples);
        println!(
            "grid_cost: {name:<18} addr_of {us:.4} us/call ({hits}/{SAMPLES} inside the band)"
        );
        planet_us.push(us);
    }

    let hull = GridMapping::Identity(
        IdentityGrid::from_half_extent_m([64.0, 32.0, 128.0]).expect("a sane slot"),
    );
    let ship = RealmId::Ship(EntityId(44));
    let mut hull_samples = Vec::with_capacity(SAMPLES as usize);
    for s in 0..SAMPLES {
        let x = ((s * 7919) % 12_800) as f64 / 100.0 - 64.0;
        let y = ((s * 104_729) % 6_400) as f64 / 100.0 - 32.0;
        let z = ((s * 1_299_709) % 25_600) as f64 / 100.0 - 128.0;
        hull_samples.push(LatticePos::from_metres(DVec3::new(x, y, z), Tier::Fine));
    }
    let (hull_us, hull_hits) = time_lookups(&hull, ship, &hull_samples);
    println!(
        "grid_cost: {:<18} addr_of {hull_us:.4} us/call ({hull_hits}/{SAMPLES} inside the slot)",
        "hull 128x64x256 m"
    );

    let worst = planet_us.iter().copied().fold(0.0_f64, f64::max);
    let best = planet_us.iter().copied().fold(f64::MAX, f64::min);
    println!(
        "grid_cost: round-body spread {:.2}x across a factor of 85 000 in radius (constant if ~1)",
        worst / best
    );
    // A hundred lookers, each addressing its own cell plus a 3x3x3 neighbourhood per tick, at 50 Hz.
    let per_tick_us = worst * 27.0 * 100.0;
    println!(
        "grid_cost: 100 lookers x 27 lookups per tick = {per_tick_us:.1} us of a 20 000 us tick"
    );
}
