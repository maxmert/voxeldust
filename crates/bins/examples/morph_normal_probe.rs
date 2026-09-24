//! ★ THE DARK EDGE OF THE FIRST RING (2026-09-24; the owner, low over the belt's highland after the
//! eye-ground cure: *"the underneath of the first rung is visible when flying"*).
//!
//! **WHY IT EXISTS.** A capture 494 m over the highland shows dark specks of ground. Its probe
//! says each one is RUNG 0 at about 950 m — the outer edge of rung 0's ring, where rung 1 takes
//! over — and the same specks stand with the shadows off (17 against 18) and with every chunk
//! casting its own shadow (18). The ground material culls back faces, so a skirt or a folded
//! triangle seen from behind would be a hole, not a dark piece. What is left is the SHADE: across
//! its fade-out band a vertex blends its own normal toward its MORPH NORMAL, the coarser rung's
//! shade where its radial meets the parent's mesh. This instrument builds the wanted chunks the
//! way the client does (the ladder's own descent, `vd_client::chunks::geometry_with` on the
//! artifact's cache, one shared parent cache) and reads, per surface vertex, its own normal and its
//! morph normal against its radial, by its distance from the eye.
//!
//! **WHAT IT READS.** The home planet solved once, the client's cache filled through its own
//! receiver, the eye stated in the planet's frame. For each rung it prints, over the vertices in
//! the rung's fade-out band: how many there are, how many whose morph normal turns more than 60°
//! from their own, and how many whose morph normal faces DOWN the radial (a shade from under the
//! ground); and the worst few with their distance, both normals' tilt from the radial and whether
//! their target came from a parent triangle or the field.
//!
//! `cargo run --release -p vd-bins --example morph_normal_probe -- ex ey ez`
//!
//! **Example.** A pilot hangs 494 m over the belt's highland. At 950 m ahead rung 0 hands over to
//! rung 1; a vertex there that blends its normal toward a shade facing down draws dark, and the
//! owner reads it as the underside of the first ring.

use std::collections::BTreeMap;

use vd_bins::artifact_worker::{SolveJob, run_solve};
use vd_client::artifact_book::ArtifactReceiver;
use vd_client::chunks::{ParentCache, drawn_field_m, geometry_with, oct_decode};
use vd_client::ladder_view::{LadderView, fade_bands};
use vd_core::glam::DVec3;
use vd_core::pose::RealmId;
use vd_seed::bend::Face;
use vd_terrain::chunk::ChunkKey;
use vd_terrain::home::{HOME_PLANET_SEED, home_planet, home_solve_words};
use vd_wire::channels::BulkMsg;

/// One band vertex's reading: (shade turn °, distance m, own tilt °, morph tilt °, shade from a
/// parent, the vertex's offset from its drawn ground m, its target's offset from the coarser ground m).
type BandVertex = (f64, f64, f64, f64, bool, f64, f64);

fn main() {
    let a: Vec<f64> = std::env::args()
        .skip(1)
        .map(|s| s.parse().expect("a number"))
        .collect();
    assert_eq!(a.len(), 3, "ex ey ez");
    let eye = [a[0], a[1], a[2]];
    let artifact = run_solve(&SolveJob {
        body: home_planet(),
        words: home_solve_words(),
    })
    .expect("the home planet solves");
    let body = home_planet().with_sea_m(artifact.sea());
    let realm = RealmId::Planet(HOME_PLANET_SEED);
    let mut rx = ArtifactReceiver::default();
    rx.accept(BulkMsg::ArtifactHead {
        realm,
        world_tag: 0,
        version: artifact.version,
        edge: artifact.edge,
        digest: artifact.digest(),
        tiles_per_edge: artifact.tiles_per_edge(),
        levels: artifact.pyramid.len() as u32,
        sea_m: artifact.sea_m,
        coast_parts: vd_bins::artifact_store::coast_parts_of(&artifact),
    });
    for (k, level) in artifact.pyramid.iter().enumerate().rev() {
        rx.accept(BulkMsg::ArtifactPyramid {
            realm,
            level: k as u32 + 1,
            part: 0,
            parts: 1,
            z_m: level.clone(),
            water_m: artifact.pyramid_water[k].clone(),
        });
    }
    for face in Face::ALL {
        for tx in 0..artifact.tiles_per_edge() {
            for ty in 0..artifact.tiles_per_edge() {
                rx.accept(BulkMsg::ArtifactTile {
                    realm,
                    face: face.index(),
                    tx,
                    ty,
                    rows: artifact.tile(face, tx, ty).to_bytes(),
                });
            }
        }
    }
    let parts = vd_bins::artifact_store::coast_parts_of(&artifact);
    for (part, bits) in artifact
        .coast
        .chunks(vd_bins::artifact_source::COAST_PART_BYTES)
        .enumerate()
    {
        rx.accept(BulkMsg::ArtifactCoast {
            realm,
            part: part as u32,
            parts,
            bits: bits.to_vec(),
        });
    }
    let cache = rx.book().get(realm).cloned().expect("a cache");
    let rungs = body.ladder().rungs;
    let mut view = LadderView::default();
    let wanted = view.wanted(&body, eye, Some(&cache));
    println!(
        "morph_normal_probe: eye {:.0} m over the drawn ground; rungs {}..{}; {} chunks wanted",
        wanted.altitude_m,
        wanted.rung_min,
        wanted.rung_max,
        wanted.keys.len()
    );
    let e = DVec3::from_array(eye);
    let parents = ParentCache::default();
    // Per rung: (band vertices, turned > 60°, facing down, worst list).
    let mut per_rung: BTreeMap<u8, (u64, u64, u64, Vec<BandVertex>)> = BTreeMap::new();
    // Over EVERY surface vertex of the rungs read (not only the band): how many face down, and how
    // far from the drawn ground those stand — a cave's roof stands far under it, a ground vertex on it.
    let mut down_own: BTreeMap<u8, (u64, u64, u64)> = BTreeMap::new();
    let mut ground_band: BTreeMap<u8, Vec<(f64, f64, f64, f64)>> = BTreeMap::new();
    for k in wanted.keys.iter().filter(|k| k.rung <= 2) {
        let key = ChunkKey {
            face: k.face,
            rung: k.rung,
            x: k.x,
            y: k.y,
            z: k.z,
        };
        let Some(g) = geometry_with(&body, Some(&cache), realm, key, &parents) else {
            continue;
        };
        let (_, out) = fade_bands(key.rung, rungs);
        let o = DVec3::from_array(g.origin_m);
        let entry = per_rung.entry(key.rung).or_insert((0, 0, 0, Vec::new()));
        for i in 0..g.skirt_start as usize {
            let p = o + DVec3::new(
                f64::from(g.vertices[i][0]),
                f64::from(g.vertices[i][1]),
                f64::from(g.vertices[i][2]),
            );
            let d = (p - e).length();
            let radial = p.normalize();
            let own = DVec3::from_array(g.normals[i].map(f64::from)).normalize_or_zero();
            let ground = drawn_field_m(&body, Some(&cache), radial.to_array(), key.rung);
            let off = ground.map_or(f64::NAN, |gr| p.length() - gr);
            let dn = down_own.entry(key.rung).or_insert((0, 0, 0));
            let own_down = own.dot(radial) < 0.0;
            dn.0 += 1;
            dn.1 += u64::from(own_down);
            dn.2 += u64::from(
                own_down & (off.abs() < 2.0 * f64::from(vd_seed::ladder::cell_m(key.rung))),
            );
            if d < out[0] || d > out[1] {
                continue;
            }
            let morph = DVec3::from_array(oct_decode(g.morph_normals[i]).map(f64::from));
            let turn = own.dot(morph).clamp(-1.0, 1.0).acos().to_degrees();
            let own_tilt = own.dot(radial).clamp(-1.0, 1.0).acos().to_degrees();
            let morph_tilt = morph.dot(radial).clamp(-1.0, 1.0).acos().to_degrees();
            let from_parent = g.morph_normals[i] != g.packed_normals[i];
            // The target: the vertex plus its radial times its morph metre, against the coarser
            // rung's drawn ground along the same radial.
            let coarser = (key.rung + 1).min(rungs - 1);
            let target = p.length() + f64::from(g.morph_m[i]);
            let coarse_ground = drawn_field_m(&body, Some(&cache), radial.to_array(), coarser);
            let target_off = coarse_ground.map_or(f64::NAN, |gr| target - gr);
            if off.abs() < 2.0 * f64::from(vd_seed::ladder::cell_m(key.rung)) {
                ground_band.entry(key.rung).or_default().push((
                    f64::from(g.morph_m[i]),
                    turn,
                    d,
                    off,
                ));
            }
            entry.0 += 1;
            entry.1 += u64::from(turn > 60.0);
            entry.2 += u64::from(morph_tilt > 90.0);
            entry
                .3
                .push((turn, d, own_tilt, morph_tilt, from_parent, off, target_off));
        }
    }
    // ★ THE GROUND ALONE in each band: the surface vertices within two cells of the drawn ground,
    // how far their morph carries them, and how far their shade turns.
    for rung in 0..=2u8 {
        let mut n = 0u64;
        let mut big_move = 0u64;
        let mut turned = 0u64;
        let mut worst: Vec<(f64, f64, f64, f64)> = Vec::new();
        for (r, list) in &ground_band {
            if *r != rung {
                continue;
            }
            for (m, turn, d, off) in list {
                n += 1;
                big_move += u64::from(m.abs() > 4.0 * f64::from(vd_seed::ladder::cell_m(rung)));
                turned += u64::from(*turn > 30.0);
                worst.push((m.abs(), *turn, *d, *off));
            }
        }
        worst.sort_by(|x, y| y.0.total_cmp(&x.0));
        println!(
            "rung {rung} GROUND in its band: {n} vertices; {big_move} moved by their morph more than four cells; {turned} whose shade turns over 30°"
        );
        for (m, turn, d, off) in worst.iter().take(6) {
            println!(
                "    morph {m:.2} m, shade turn {turn:.1}°, at {d:.0} m, the vertex {off:+.2} m off its ground"
            );
        }
        let mut by_turn = worst.clone();
        by_turn.sort_by(|x, y| y.1.total_cmp(&x.1));
        for (m, turn, d, off) in by_turn.iter().take(6) {
            println!(
                "    (by turn) shade turn {turn:.1}°, morph {m:.2} m, at {d:.0} m, the vertex {off:+.2} m off its ground"
            );
        }
    }
    for (rung, (all, own_down, on_ground)) in &down_own {
        println!(
            "rung {rung}: {all} surface vertices; {own_down} with their OWN normal facing down the radial, {on_ground} of those within two cells of the drawn ground"
        );
    }
    for (rung, (n, turned, down, mut worst)) in per_rung {
        let (_, out) = fade_bands(rung, rungs);
        println!(
            "rung {rung}: fade-out band {:.0}..{:.0} m; {n} surface vertices in it; {turned} turned over 60° from their own normal; {down} with a morph normal facing down the radial",
            out[0], out[1]
        );
        worst.sort_by(|x, y| y.0.total_cmp(&x.0));
        for (turn, d, own_tilt, morph_tilt, from_parent, off, target_off) in
            worst.into_iter().take(8)
        {
            println!(
                "    turned {turn:.1}° at {d:.0} m: own normal {own_tilt:.1}° off the radial, morph normal {morph_tilt:.1}° off it, the shade from {}; the vertex {off:+.2} m off its drawn ground, its target {target_off:+.2} m off the coarser drawn ground",
                if from_parent {
                    "a parent triangle"
                } else {
                    "its own normal (no hit)"
                }
            );
        }
    }
}
