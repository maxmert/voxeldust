//! ★ THE STEPPED COASTS' BENCH (rulings B58 item 2, B59–B61): the initial land of the home planet from the bench's copy
//! ([`vd_bins::land_copy`]) — first THE CONTROLS (the copy with the game's laws against the game's own initial land,
//! every row; the belts' copy with both laws on against the game's belts), then each run given — and each land's
//! heights written as face grids for the coast readings (`<out_prefix>.<label>.z0.i32`, sixteenths, the order
//! (face, j, i)), with its first sea and its belts' mask.
//!
//! `cargo run --release -p vd-bins --example coast_bench -- <out_prefix> [run …]`
//!
//! A run is a `+`-joined list applied to GENERATOR 25's land (`LandLaws::GENERATOR_25`, every law of generator 26
//! out): `kK` the cell's mean over K × K sub-cells, `wM` the diagnostic blend over ±M metres, `margins`, `spread`,
//! `airy`, `flex`, `spectrum`, `share`; `k8+margins+spread+spectrum+share` is the game's own land.
//!
//! **Example.** `coast_bench /tmp/coast k8 k8+margins` prints `CONTROL … MATCHES` twice, then writes
//! `/tmp/coast.game.z0.i32`, `/tmp/coast.k8.z0.i32` and `/tmp/coast.k8_margins.z0.i32`; the coast reader then states
//! how often each coastline faces a node row.

use std::io::Write;
use std::time::Instant;

use vd_seed::bend::Face;
use vd_terrain::artifact::node_at;
use vd_terrain::home::{home_planet, home_solve_words};
use vd_terrain::land::Land;
use vd_terrain::macro_lattice::MacroLattice;

/// The belts' mask of a land (ruling B59's old-plains reading): one where the belts' law raises a node by the end —
/// the game's law, or the bench's copy with the run's own candidates (ruling B60 item 3).
fn write_belts(
    path: &str,
    lattice: &MacroLattice,
    land: &Land,
    laws: vd_bins::land_copy::LandLaws,
) {
    let body = home_planet();
    let words = home_solve_words();
    let belts = if laws == vd_bins::land_copy::LandLaws::default() {
        vd_terrain::belts::belts(&body, lattice, &words, land, words.age_yr)
    } else {
        vd_bins::belts_copy::belts_copy(
            &body,
            lattice,
            &words,
            land,
            words.age_yr,
            laws.share,
            laws.margins,
        )
    };
    let edge = i64::from(lattice.edge);
    let mut bytes = Vec::with_capacity(lattice.node_count());
    for face in Face::ALL {
        for j in 0..edge {
            for i in 0..edge {
                let k = node_at(lattice, face, i, j) as usize;
                bytes.push(u8::from(belts.total[k] > 0));
            }
        }
    }
    std::fs::File::create(path)
        .and_then(|mut f| f.write_all(&bytes))
        .expect("the mask is written");
}

fn write_grid(path: &str, lattice: &MacroLattice, land: &Land) {
    let edge = i64::from(lattice.edge);
    let mut bytes = Vec::with_capacity(lattice.node_count() * 4);
    for face in Face::ALL {
        for j in 0..edge {
            for i in 0..edge {
                bytes.extend_from_slice(
                    &land.z[node_at(lattice, face, i, j) as usize].to_le_bytes(),
                );
            }
        }
    }
    std::fs::File::create(path)
        .and_then(|mut f| f.write_all(&bytes))
        .expect("the grid is written");
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    assert!(!args.is_empty(), "<out_prefix> [run …]");
    let prefix = &args[0];
    // Each run a `+`-joined list (ruling B59): `kK` the cell's mean over K × K sub-cells, `wM` the diagnostic blend over
    // ±M metres, `margins`, `spread`, `airy`, `flex`, `spectrum`; the run's label is the list with `_` for `+`.
    let runs: Vec<(String, vd_bins::land_copy::LandLaws)> = args[1..]
        .iter()
        .map(|a| {
            use vd_bins::land_copy::Relief;
            let mut laws = vd_bins::land_copy::LandLaws::GENERATOR_25;
            for part in a.split('+') {
                if let Some(k) = part.strip_prefix('k') {
                    laws.cell = Some(k.parse().expect("kK"));
                } else if let Some(m) = part.strip_prefix('w') {
                    laws.blend_m = Some(m.parse().expect("wM"));
                } else {
                    match part {
                        "margins" => laws.margins = true,
                        "spread" => laws.spread = true,
                        "airy" => laws.relief = Relief::Airy,
                        "flex" => laws.relief = Relief::Flex,
                        "spectrum" => laws.relief = Relief::Spectrum,
                        "share" => laws.share = true,
                        other => panic!("a run part is kK, wM, margins, spread, airy, flex or spectrum, not {other}"),
                    }
                }
            }
            (a.replace('+', "_"), laws)
        })
        .collect();
    let body = home_planet();
    let words = home_solve_words();
    let lattice = MacroLattice::of(&body).expect("a lattice");
    let t = Instant::now();
    let game = vd_terrain::land::initial_land(&body, &lattice, &words.land());
    println!(
        "the game's initial land in {:.1} s; first sea {:?}",
        t.elapsed().as_secs_f64(),
        game.sea_z
    );
    let t = Instant::now();
    let copy = vd_bins::land_copy::initial_land_copy(
        &body,
        &lattice,
        &words.land(),
        vd_bins::land_copy::LandLaws::default(),
    );
    let same = copy.z == game.z
        && copy.uplift == game.uplift
        && copy.crust == game.crust
        && copy.province == game.province
        && copy.plate == game.plate
        && copy.kind == game.kind
        && copy.boundary_m == game.boundary_m
        && copy.sea_z == game.sea_z
        && copy.ocean == game.ocean
        && copy.threshold == game.threshold
        && copy.freeboard_factor == game.freeboard_factor;
    println!(
        "CONTROL the copy with the game's laws ({:.1} s): {} the game's initial land",
        t.elapsed().as_secs_f64(),
        if same { "MATCHES" } else { "DIFFERS FROM" }
    );
    assert!(
        same,
        "the control must match, or no candidate reading can be trusted"
    );
    // THE BELTS' CONTROL: the copy of the belts' law with both laws on is the game's law.
    let game_belts = vd_terrain::belts::belts(&body, &lattice, &words, &game, words.age_yr);
    let copy_belts =
        vd_bins::belts_copy::belts_copy(&body, &lattice, &words, &game, words.age_yr, true, true);
    let belts_same = game_belts == copy_belts;
    println!(
        "CONTROL the belts' copy with both laws on: {} the game's belts ({} wedge nodes)",
        if belts_same {
            "MATCHES"
        } else {
            "DIFFERS FROM"
        },
        game_belts.wedge.len()
    );
    assert!(belts_same, "the belts' control must match");
    // THE ROCK MAP BY BOUNDARY KIND (generator 26's finding: the Andean margin's trench wall sinks continental crust):
    // the game's first land's provinces on each kind of boundary, before and after the belts' law marks its folded belt.
    let mut after = game.clone();
    let _ = vd_terrain::solve::belt_land(&game_belts, &mut after);
    let old = vd_bins::land_copy::initial_land_copy(
        &body,
        &lattice,
        &words.land(),
        vd_bins::land_copy::LandLaws::GENERATOR_25,
    );
    for (label, land) in [
        ("generator 25's first land", &old),
        ("the first land", &game),
        ("after the belts' law", &after),
    ] {
        let mut counts = std::collections::BTreeMap::new();
        for k in 0..land.province.len() {
            let province =
                vd_terrain::strata::Province::from_code(land.province[k]).expect("a province");
            if province != vd_terrain::strata::Province::RiftBasalt {
                continue;
            }
            let crust = if land.crust[k] >= 128 {
                "continental"
            } else {
                "oceanic"
            };
            let moves = if land.uplift[k] > 0 {
                "rising"
            } else {
                "sinking"
            };
            // The other plate's character: an oceanic node rising against a continent is on the diving plate (no arc).
            let gi = lattice.direction(k as u32);
            let one = vd_terrain::Gf::from_i64(1 << vd_recipe::bend::DIR_BITS);
            let d = gi.map(|c| vd_terrain::Gf::from_i64(c.raw()) / one);
            let b = vd_terrain::land::boundary_of(&land.plates, d);
            let other = match b.other {
                Some(o) if land.plates[usize::from(o)].affinity >= vd_terrain::Gf::ZERO => {
                    "against a continent"
                }
                Some(_) => "against an ocean",
                None => "alone",
            };
            *counts
                .entry((land.kind[k], crust, moves, other))
                .or_insert(0usize) += 1;
        }
        println!(
            "THE RIFT BASALT ({label}), nodes by (boundary kind: 1 convergent, 2 divergent, 3 transform; crust; uplift): {counts:?}"
        );
    }
    write_grid(&format!("{prefix}.game.z0.i32"), &lattice, &game);
    write_belts(
        &format!("{prefix}.game.belt.u8"),
        &lattice,
        &game,
        vd_bins::land_copy::LandLaws::default(),
    );
    println!("{prefix}.game.z0.i32 sea {:?}", game.sea_z);
    println!(
        "the home planet's flexural parameter {:.0} m; the belt's half-width {:.0} m; a trench's {:.0} m",
        vd_terrain::land::flexural_parameter_m(
            words.elastic_thickness_m,
            body.facts().gravity_mm_s2
        ),
        vd_terrain::land::flexural_parameter_m(
            words.elastic_thickness_m,
            body.facts().gravity_mm_s2
        ) * vd_terrain::land::BELT_HALF_WIDTHS,
        vd_terrain::land::flexural_parameter_m(
            words.elastic_thickness_m,
            body.facts().gravity_mm_s2
        ) * vd_terrain::land::BELT_HALF_WIDTHS
            * vd_terrain::land::NARROW_SHARE
    );
    for (label, candidate) in runs {
        let t = Instant::now();
        let land = vd_bins::land_copy::initial_land_copy(&body, &lattice, &words.land(), candidate);
        let moved = land.z.iter().zip(&game.z).filter(|(a, b)| a != b).count();
        let crust_moved = land
            .crust
            .iter()
            .zip(&game.crust)
            .filter(|(a, b)| a != b)
            .count();
        println!(
            "{candidate:?} ({:.1} s): first sea {:?} (the game's {:?}); the freeboard factor {:.5} (the game's {:.5}); {moved} heights and {crust_moved} crust words differ from the game's",
            t.elapsed().as_secs_f64(),
            land.sea_z,
            game.sea_z,
            land.freeboard_factor.to_f64(),
            game.freeboard_factor.to_f64()
        );
        write_grid(&format!("{prefix}.{label}.z0.i32"), &lattice, &land);
        write_belts(
            &format!("{prefix}.{label}.belt.u8"),
            &lattice,
            &land,
            candidate,
        );
        println!("{prefix}.{label}.z0.i32 sea {:?}", land.sea_z);
    }
}
