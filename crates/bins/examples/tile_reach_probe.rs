//! ★ WHICH TILES THE SHARD'S REACH RULE NAMES for an eye over the home planet (2026-09-22, the
//! on-foot ground): the same `tile_reach_m` + `tiles_within` the shard and the gateway read, at a
//! stated eye in the planet's frame, so the list can be held against the tiles a client's chunks
//! name as missing.
//! `cargo run --release -p vd-bins --example tile_reach_probe -- ex ey ez`

use vd_terrain::home::home_planet;
use vd_terrain::macro_lattice::MacroLattice;

fn main() {
    let a: Vec<f64> = std::env::args()
        .skip(1)
        .map(|s| s.parse().expect("a number"))
        .collect();
    let eye = [a[0], a[1], a[2]];
    let body = home_planet();
    let lattice = MacroLattice::of(&body).expect("a lattice");
    let radial = (eye[0] * eye[0] + eye[1] * eye[1] + eye[2] * eye[2]).sqrt();
    let dir = [eye[0] / radial, eye[1] / radial, eye[2] / radial];
    let levels = 5;
    // ★ THE RING IS FLOORED BY THE HANDOVER'S OWN STEP (ruling W17), the shard's very line.
    let top = body.ladder().rungs - 1;
    let tile_rung = vd_terrain::artifact::tile_rung(&lattice, levels, top);
    let step_m = body.step_bound_m(tile_rung)
        + vd_terrain::artifact::field_step_m(&body, &lattice, levels, tile_rung);
    let reach = vd_terrain::artifact::tile_reach_m(
        &lattice,
        levels,
        top,
        body.radius_m(),
        body.relief_bound_m(0),
        radial,
        vd_core::geometry::drawable_theta_min_rad(),
        step_m,
    );
    let face = vd_seed::bend::face_of(dir);
    let (t, s) = vd_seed::bend::face_coords(face, dir);
    let n = lattice.edge;
    let i = vd_seed::ladder::index_of(vd_seed::bend::unbend(t), n);
    let j = vd_seed::ladder::index_of(vd_seed::bend::unbend(s), n);
    println!(
        "eye altitude {:.0} m; reach {:.0} m ({:.1} nodes of {:.0} m); the eye's face {:?} node ({i}, {j}) tile ({}, {})",
        radial - body.radius_m(),
        reach,
        reach / lattice.node_m(),
        lattice.node_m(),
        face,
        i / vd_terrain::artifact::TILE_EDGE as i32,
        j / vd_terrain::artifact::TILE_EDGE as i32
    );
    let tiles = vd_terrain::artifact::tiles_within(&lattice, dir, reach);
    println!("{} tiles within reach:", tiles.len());
    for t in &tiles {
        println!("  face {} tile ({}, {})", t.0, t.1, t.2);
    }
}
