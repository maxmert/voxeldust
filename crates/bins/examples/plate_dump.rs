//! ★ THE INITIAL LAND'S PLATES AS FACE GRIDS, beside [`art_dump`](./art_dump.rs)'s rows (ruling B58 item 2: the stepped
//! coasts' cause). Writes five little-endian files beside `<out_prefix>`, each `6 · edge · edge` values in the order
//! (face, j, i): `.plate.u8` (each node's plate), `.crust.u8` (the continental share, 0–255), `.boundary.u32` (the
//! distance to the nearest plate boundary, metres) and `.kind.u8` (that boundary's kind: 0 none, 1 convergent,
//! 2 divergent, 3 transform), `.z0.i32` (the initial land's height, sixteenths), `.area.u64` (each node's area, m² — the
//! weight every reading of the bench's field dumps takes), and prints each plate's character.
//!
//! `cargo run --release -p vd-bins --example plate_dump -- <out_prefix>`
//!
//! **Example.** A reader finds the home planet's coast nodes in the artifact's rows and asks this file how far each one
//! stands from a plate boundary: a coast that follows a boundary's straight line steps along the node rows.

use std::io::Write;

use vd_seed::bend::Face;
use vd_terrain::artifact::node_at;
use vd_terrain::home::{home_planet, home_solve_words};
use vd_terrain::macro_lattice::MacroLattice;

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    assert!(args.len() == 1, "<out_prefix>");
    let body = home_planet();
    let words = home_solve_words();
    let lattice = MacroLattice::of(&body).expect("a lattice");
    let land = vd_terrain::land::initial_land(&body, &lattice, &words.land());
    let edge = i64::from(lattice.edge);
    let (mut plate, mut crust, mut boundary, mut kind, mut z0, mut area) = (
        Vec::new(),
        Vec::new(),
        Vec::new(),
        Vec::new(),
        Vec::new(),
        Vec::new(),
    );
    for face in Face::ALL {
        for j in 0..edge {
            for i in 0..edge {
                let k = node_at(&lattice, face, i, j) as usize;
                plate.push(land.plate[k]);
                crust.push(land.crust[k]);
                boundary.extend_from_slice(&land.boundary_m[k].to_le_bytes());
                kind.push(land.kind[k]);
                z0.extend_from_slice(&land.z[k].to_le_bytes());
                area.extend_from_slice(&lattice.area_m2(k as u32).to_le_bytes());
            }
        }
    }
    for (ext, bytes) in [
        ("plate.u8", &plate),
        ("crust.u8", &crust),
        ("boundary.u32", &boundary),
        ("kind.u8", &kind),
        ("z0.i32", &z0),
        ("area.u64", &area),
    ] {
        let path = format!("{}.{ext}", args[0]);
        std::fs::File::create(&path)
            .and_then(|mut f| f.write_all(bytes))
            .expect("the file is written");
        println!("{path}");
    }
    for (p, plate) in land.plates.iter().enumerate() {
        println!(
            "plate {p}: affinity {:+.3}, crust scatter {:.3}, age share {:.3}",
            plate.affinity.to_f64(),
            plate.crust_scatter.to_f64(),
            plate.age.to_f64()
        );
    }
    println!(
        "the crust threshold {:.4}, the freeboard factor {:.4}, the first sea {:?} sixteenths",
        land.threshold.to_f64(),
        land.freeboard_factor.to_f64(),
        land.sea_z
    );
}
