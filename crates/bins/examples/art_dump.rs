//! ★ AN ARTIFACT FILE'S ROWS AS FACE GRIDS, for a reader outside Rust (ruling B58: the grid marks' causes are measured
//! on the shipped rows). Writes six little-endian files beside `<out_prefix>`, each `6 · edge · edge` values in the
//! order (face, j, i) of [`node_at`]: `.z.i16` (the height, whole metres), `.water.i16`, `.slot.u8` (the receiver's
//! slot, 255 for none), `.q.u8` (the discharge class), `.rain.u8` and `.province.u8`.
//!
//! `cargo run --release -p vd-bins --example art_dump -- <artifact_file> <out_prefix>` (it also prints the artifact's
//! digest, the one the release gate pins).
//!
//! **Example.** The belt tile of the home planet is face 3, `i` in 1088..1152 and `j` in 960..1024; a reader takes the
//! `.z.i16` file as an array of shape (6, 1216, 1216) and reads `[3, 960:1024, 1088:1152]`.

use std::io::Write;

use vd_seed::bend::Face;
use vd_terrain::artifact::{RECEIVER_NONE, RECEIVER_SLOT_MASK, node_at};
use vd_terrain::home::home_planet;
use vd_terrain::macro_lattice::MacroLattice;

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    assert!(args.len() == 2, "<artifact_file> <out_prefix>");
    let artifact = vd_bins::artifact_file::read(&args[0]).expect("an artifact file");
    // The artifact's own digest, the one the release gate pins (generator 26: the game's solve against the gate's).
    let digest = artifact.digest();
    println!("{}: digest [{:#x}, {:#x}]", args[0], digest[0], digest[1]);
    let lattice = MacroLattice::of(&home_planet()).expect("a lattice");
    assert_eq!(artifact.edge, lattice.edge, "the home planet's lattice");
    let edge = i64::from(lattice.edge);
    let (mut z, mut water, mut slot, mut q, mut rain, mut province) = (
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
                let r = artifact.rows[node_at(&lattice, face, i, j) as usize];
                z.extend_from_slice(&r.z_m.to_le_bytes());
                water.extend_from_slice(&r.water_m.to_le_bytes());
                slot.push(if r.receiver_facies & RECEIVER_NONE != 0 {
                    255
                } else {
                    r.receiver_facies & RECEIVER_SLOT_MASK
                });
                q.push(r.discharge_log);
                rain.push(r.rain);
                province.push(r.province);
            }
        }
    }
    for (ext, bytes) in [
        ("z.i16", &z),
        ("water.i16", &water),
        ("slot.u8", &slot),
        ("q.u8", &q),
        ("rain.u8", &rain),
        ("province.u8", &province),
    ] {
        let path = format!("{}.{ext}", args[1]);
        std::fs::File::create(&path)
            .and_then(|mut f| f.write_all(bytes))
            .expect("the file is written");
        println!("{path}");
    }
    println!("sea {} m, edge {}", artifact.sea_m, lattice.edge);
}
