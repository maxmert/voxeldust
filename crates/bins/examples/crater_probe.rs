//! ★ THE CRATERS OVER ONE TILE (ruling B60 item 3: the second cliff's cause). The home planet's crater population
//! exactly as the game's solve draws it (the initial land, the climate over it, `vd_terrain::craters::crater_population`)
//! and, for every crater whose ejecta reaches a stated tile, its centre (face, i, j), its diameter, its shape (rim,
//! depth, complex), its age and the pass of the solve's axis it is stamped in.
//!
//! `cargo run --release -p vd-bins --example crater_probe -- [F,I0,J0,W]`
//!
//! **Example.** The Andean coast tile (face 3, nodes 1088.. × 128..) shows a cliff 3.5 km tall 200 km inside plate 1; the
//! probe lists no crater whose ejecta reaches the tile, which ruled the craters out (MEASURED: the cliff is the belts'
//! wedge stopping at the continent's half-share line, ruling B61).

use vd_seed::bend::Face;
use vd_terrain::Gf;
use vd_terrain::artifact::node_at;
use vd_terrain::craters::{Surface, crater_population, shape_of};
use vd_terrain::home::{home_planet, home_solve_words};
use vd_terrain::macro_lattice::MacroLattice;
use vd_terrain::solve::Schedule;

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let tile: Vec<i64> = args
        .first()
        .map_or_else(|| "3,1088,128,64".to_string(), Clone::clone)
        .split(',')
        .map(|x| x.parse().expect("F,I0,J0,W"))
        .collect();
    let (face, i0, j0, w) = (Face::ALL[tile[0] as usize], tile[1], tile[2], tile[3]);
    let body = home_planet();
    let words = home_solve_words();
    let lattice = MacroLattice::of(&body).expect("a lattice");
    let land = vd_terrain::land::initial_land(&body, &lattice, &words.land());
    let climate = vd_terrain::climate::climate(&body, &lattice, &words, &land.z, land.sea_view());
    let surface = Surface {
        crust: &land.crust,
        boundary_m: &land.boundary_m,
        plate: &land.plate,
        plates: &land.plates,
        province: &land.province,
        rain_mm_yr: &climate.rain_mm_yr,
    };
    let craters = crater_population(&body, &lattice, &words, &surface);
    let gravity = body.facts().gravity_mm_s2;
    let times = Schedule::standard(words.age_yr).times();
    let centre = node_at(&lattice, face, i0 + w / 2, j0 + w / 2);
    let tile_reach_m = lattice.node_m() * w as f64 * 0.75;
    println!(
        "{} craters on the home planet; those whose ejecta reaches the tile's middle ({} km around face {} node ({}, {})):",
        craters.len(),
        (tile_reach_m / 1000.0).round(),
        tile[0],
        i0 + w / 2,
        j0 + w / 2
    );
    for c in &craters {
        let shape = shape_of(Gf::from_i64(i64::from(c.diameter_m)), gravity);
        let reach = shape.radius_m.to_f64() * 2.0;
        let d = f64::from(lattice.chord_m(c.node, centre));
        if d > reach + tile_reach_m {
            continue;
        }
        let at = words.age_yr.saturating_sub(c.age_yr);
        let pass = times
            .iter()
            .position(|&t| t > at)
            .map_or(times.len() - 1, |p| p - 1);
        let (cf, ci, cj) = lattice.split(c.node);
        println!(
            "  centre face {} ({ci}, {cj}), {:.0} km from the tile's middle; diameter {:.0} km (rim {:.0} m, depth {:.0} m, {}); {:.1} Myr old, stamped in pass {pass}",
            cf.index(),
            d / 1000.0,
            f64::from(c.diameter_m) / 1000.0,
            shape.rim_m.to_f64(),
            shape.depth_m.to_f64(),
            if shape.complex { "complex" } else { "simple" },
            c.age_yr as f64 / 1e6
        );
    }
}
