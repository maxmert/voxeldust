//! ★ HOW FAR THE SHORE MOVES BETWEEN RUNGS (the owner, 2026-09-21 evening: "when I'm flying, the
//! shores are changing all the time"): the shore is where the land crosses the sea's level. The land
//! at rung r is the solved field plus the fine octaves the rung keeps, and the octaves a coarser
//! rung drops move the crossing sideways by their amplitude over the land's slope. Along lines
//! across the coast near a stand, this prints, per rung pair, how far the crossing moves: the
//! shore's own step at a ring swap, in metres, which the crossfade slides the eye through.
//!
//! `cargo run --release -p vd-bins --example shore_step -- [dx dy dz] [lines] [half_length_km]
//! [step_m] [rung_lo] [rung_hi]`
//!
//! ★ THE OLD RULE, FOR THE BEFORE NUMBER (2026-09-22, ruling W15): with `VD_SHORE_RULE=old` in the
//! environment the pyramid levels are built with NO coast counts, so every cell takes the side of
//! the ONE fine node nearest its centre — the rule ruling W10 shipped and the rule that painted
//! squares of water on the land from 41 000 km. The same run with the counts is the after number.

use vd_bins::artifact_worker::{SolveJob, run_solve};
use vd_core::glam::DVec3;
use vd_terrain::height::height_field_m;
use vd_terrain::home::{home_planet, home_solve_words};

fn main() {
    let a: Vec<f64> = std::env::args()
        .skip(1)
        .map(|s| s.parse().expect("a number"))
        .collect();
    let aim = if a.len() >= 3 {
        DVec3::new(a[0], a[1], a[2]).normalize()
    } else {
        DVec3::new(-0.169_974, -0.983_964, -0.054_071).normalize()
    };
    let lines = a.get(3).map_or(400, |v| *v as usize);
    let half_km = a.get(4).copied().unwrap_or(60.0);
    let step_m = a.get(5).copied().unwrap_or(64.0);
    let rung_lo = a.get(6).map_or(3u8, |v| *v as u8);
    let rung_hi = a.get(7).map_or(18u8, |v| *v as u8);
    // The old rule: no counts on any level, so a cell reads its centre node's own bit.
    let old_rule = std::env::var("VD_SHORE_RULE").is_ok_and(|v| v == "old");
    let body = home_planet();
    let artifact = run_solve(&SolveJob {
        body,
        words: home_solve_words(),
    })
    .expect("the home planet solves");
    let body = home_planet().with_sea_m(artifact.sea());
    let sea_r = body.sea_radius_m();
    let radius = body.radius_m();
    let seed = if aim.z.abs() < 0.9 {
        DVec3::Z
    } else {
        DVec3::X
    };
    let east = aim.cross(seed).normalize();
    let north = east.cross(aim).normalize();
    let half = half_km * 1000.0;
    let n = (2.0 * half / step_m) as usize;
    // ★ EVERY RUNG A HULL DRAWS (2026-09-22): the fine rungs read the rows, the far rungs read the
    // pyramid's levels as the client's book picks them (`PyramidField::level_for`), so a swap from
    // the rows to level 1 and from level to level is measured like a swap between two fine rungs.
    let rungs: Vec<u8> = (rung_lo..=rung_hi).collect();
    let lattice = vd_terrain::macro_lattice::MacroLattice::of(&body).expect("a lattice");
    let levels = artifact.pyramid.len() as u32;
    let counts = artifact.coast_counts();
    let fields: Vec<Box<dyn vd_terrain::artifact::ZField>> = rungs
        .iter()
        .map(|&r| {
            let level = vd_terrain::artifact::PyramidField::level_for(&lattice, levels, r);
            if level == 0 {
                Box::new(artifact.clone()) as Box<dyn vd_terrain::artifact::ZField>
            } else {
                let mut field = vd_terrain::artifact::PyramidField::of(&artifact, level, &counts)
                    .expect("a level");
                if old_rule {
                    field.counts = None;
                }
                Box::new(field) as Box<dyn vd_terrain::artifact::ZField>
            }
        })
        .collect();
    println!(
        "shore_step rule: {}; step {step_m} m; rungs {rung_lo}..{rung_hi}",
        if old_rule {
            "OLD (the centre node)"
        } else {
            "the footprint's wet fraction"
        }
    );
    println!(
        "levels per rung: {:?}",
        rungs
            .iter()
            .map(|&r| vd_terrain::artifact::PyramidField::level_for(&lattice, levels, r))
            .collect::<Vec<_>>()
    );
    println!(
        "coast level per rung: {:?}",
        rungs
            .iter()
            .map(|&r| vd_terrain::artifact::coast_level(lattice.cells_per_node, r))
            .collect::<Vec<_>>()
    );
    // Per rung pair: the crossings' sideways steps.
    let mut steps: Vec<Vec<f64>> = vec![Vec::new(); rungs.len() - 1];
    let mut crossings_found = 0usize;
    let mut rng = 0x9E37_79B9_7F4A_7C15u64;
    let mut next = || {
        rng ^= rng << 13;
        rng ^= rng >> 7;
        rng ^= rng << 17;
        (rng >> 11) as f64 / (1u64 << 53) as f64
    };
    for _ in 0..lines {
        // A line: a random offset across the stand and a random heading.
        let off = east * ((next() - 0.5) * 2.0 * half) + north * ((next() - 0.5) * 2.0 * half);
        let ang = next() * std::f64::consts::TAU;
        let dir = (east * ang.cos() + north * ang.sin()).normalize();
        let centre = (aim * radius + off).normalize() * radius;
        // The land minus the sea along the line, per rung.
        let mut over: Vec<Vec<f64>> = vec![Vec::with_capacity(n); rungs.len()];
        for k in 0..n {
            let p = (centre + dir * ((k as f64) * step_m - half)).normalize();
            for (ri, r) in rungs.iter().enumerate() {
                let h = height_field_m(&body, &*fields[ri], p.to_array(), *r).unwrap_or(f64::NAN);
                over[ri].push(h - sea_r);
            }
        }
        // The first crossing per rung, walking from the line's start.
        let first_cross = |v: &[f64]| -> Option<f64> {
            for k in 1..v.len() {
                if v[k - 1].is_nan() || v[k].is_nan() {
                    continue;
                }
                if (v[k - 1] > 0.0) != (v[k] > 0.0) {
                    let t = v[k - 1] / (v[k - 1] - v[k]);
                    return Some(((k - 1) as f64 + t) * step_m);
                }
            }
            None
        };
        let xs: Vec<Option<f64>> = over.iter().map(|v| first_cross(v)).collect();
        if xs[0].is_some() {
            crossings_found += 1;
        }
        for ri in 0..rungs.len() - 1 {
            if let (Some(a), Some(b)) = (xs[ri], xs[ri + 1]) {
                steps[ri].push((a - b).abs());
            }
        }
    }
    println!(
        "shore_step: {lines} lines of {:.0} km across the coast near the stand; {crossings_found} lines cross the shore at rung {}",
        2.0 * half_km,
        rungs[0]
    );
    println!("rung pair | lines | shore step median m | p90 m | max m | cell m");
    for ri in 0..rungs.len() - 1 {
        let mut s = steps[ri].clone();
        s.sort_by(f64::total_cmp);
        if s.is_empty() {
            continue;
        }
        let med = s[s.len() / 2];
        let p90 = s[(s.len() * 9 / 10).min(s.len() - 1)];
        println!(
            "{:>3} → {:<3} | {:>5} | {:>19.0} | {:>5.0} | {:>5.0} | {}",
            rungs[ri],
            rungs[ri + 1],
            s.len(),
            med,
            p90,
            s[s.len() - 1],
            vd_seed::ladder::cell_m(rungs[ri + 1])
        );
    }
}
