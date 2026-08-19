//! ★ THE SEED-SEARCH TOOL (celestial taxonomy arc T4; the owner's standing search for a
//! yellow-sun home). NOT wired into any boot: a report-only instrument that READS THE world
//! at candidate seeds through the ONE generator (SL5 — no second generator, no variant, no
//! world knob) and prints every Earth-like candidate with its numbers.
//!
//! THE SL5 FIREWALL: the ranking weights below are TOOL policy — they live HERE, never in
//! `UniverseConfig`, and never reach the generator (a seed-selection instrument may have
//! knobs; THE world may not — `the_ranking_weights_are_absent_from_the_one_config` pins it).
//!
//! Usage: `vd-seedsearch [start_seed] [count]` — defaults: seed 0, the derived sweep size.

/// The derived sweep size: the design's own predicted yield is ~1 Earth-like seed in 192
/// (§8.3 — `P(G) = 0.013323` through the tree's own IMF bounds × the stripped-rocky mass
/// window), so a sweep sized for ~16 expected hits is `16 × 192 = 3072` seeds — enough for
/// the measured rate to stand against the prediction without hand-picking a rounder number.
const DERIVED_SWEEP: u64 = 16 * 192;

/// TOOL-POLICY ranking weights (§8.4): log-distance from Earth in radius, from the Sun in
/// stellar mass, and from Earth's 254 K in temperature. Insolation carries ZERO information
/// (quantised and seed-free — §8.2) and is deliberately absent from the rank.
const W_RADIUS: f64 = 1.0;
const W_STAR_MASS: f64 = 1.0;
const W_TEMP: f64 = 1.0;

fn main() {
    let mut args = std::env::args().skip(1);
    let start: u64 = args.next().and_then(|a| a.parse().ok()).unwrap_or(0);
    let count: u64 = args
        .next()
        .and_then(|a| a.parse().ok())
        .unwrap_or(DERIVED_SWEEP);
    let config = vd_bins::process_world_config(vd_bins::DEV.move_speed, vd_bins::DEV.tick_dt);
    let mut hits: Vec<(f64, u64, vd_physics::worldgen::EarthLikeCandidate)> = Vec::new();
    let mut seeds_with_hits = 0u64;
    for seed in start..start + count {
        let found = vd_physics::worldgen::earth_like_candidates(seed, &config);
        if !found.is_empty() {
            seeds_with_hits += 1;
        }
        for c in found {
            let rank = W_RADIUS * (c.radius_m / vd_physics::taxonomy::R_EARTH_M).ln().abs()
                + W_STAR_MASS * c.star_mass_msun.ln().abs()
                + W_TEMP * (c.t_eq_k / 254.031).ln().abs();
            hits.push((rank, seed, c));
        }
    }
    hits.sort_by(|a, b| a.0.total_cmp(&b.0));
    println!(
        "[seedsearch] swept seeds {start}..{} ({count}): {seeds_with_hits} seeds hold an \
         Earth-like world ({} candidate bodies). Predicted ~1 in 192 (taxonomy design §8.3).",
        start + count,
        hits.len(),
    );
    println!(
        "[seedsearch] NOTE (§8.2): insolation and T_eq are structural no-ops given rocky+G \
         (the ladder is quantised and seed-free); the real discriminants are the star's class \
         and the planet's drawn mass."
    );
    for (rank, seed, c) in hits.iter().take(20) {
        println!(
            "  seed={seed} rank={rank:.4} {:?} in {:?}: star {:.4} Msun, mass {:.3e} kg, \
             radius {:.1} km, S {:.6}, T_eq {:.2} K",
            c.body,
            c.system,
            c.star_mass_msun,
            c.mass_kg,
            c.radius_m / 1.0e3,
            c.insolation_rel,
            c.t_eq_k,
        );
    }
}
