//! ★ DOES A SKIRT REACH THE SUNK COARSER GROUND? (2026-09-24; the owner, flying very low at 29 km/s over
//! the belt's highland: *"I see cracks from time to time, only when I fly veeeery low"* — a long,
//! straight, dark line with light bits in it.)
//!
//! **WHY IT EXISTS.** In that flight 400 to 2 450 wanted chunks stood URGENT at a time: the builders
//! could not keep up. Where a finer chunk is still missing, the coarser rung shows instead, SUNK under
//! the finer by the coarser rung's sink so it never shows through the finer ground (coarse before fine,
//! SL8). So at the edge of a DRAWN finer chunk whose neighbour is MISSING, the ground steps down to the
//! sunk coarser stand-in, and only the finer chunk's SKIRT (two cells deep) can close that step. This
//! measures whether it does: for every skirt bottom of the finer chunks around a stated eye, inside the
//! band where the coarser is sunk whole, how far the bottom stands OVER the sunk coarser ground. A
//! positive number is a crack of that height, visible from a grazing eye.
//!
//! **WHAT IT READS.** The home planet solved once, the client's cache through its own receiver, the
//! ladder's own descent, the chunks the client builds (`geometry_with`, skirts included). The coarser
//! ground is the coarser rung's drawn FIELD (`drawn_field_m`), which its mesh follows within a cell;
//! the sink is `vd_client::chunks::sink_m` at the coarser rung, whole wherever the finer is nearer than
//! its fade-out band.
//!
//! `cargo run --release -p vd-bins --example skirt_gap -- ex ey ez`
//!
//! **Example.** A hull 30 m over the highland: a rung-0 chunk is drawn, its eastern neighbour is not yet
//! built, so rung 1 stands there 3.45 m under it. The rung-0 skirt hangs 2 m, so a strip 1.45 m tall
//! stays open along the chunk's edge — a straight dark crack, and through it whatever lies behind.

use std::collections::BTreeMap;

use vd_bins::artifact_worker::{SolveJob, run_solve};
use vd_client::artifact_book::ArtifactReceiver;
use vd_client::chunks::{ParentCache, drawn_field_m, geometry_with, sink_m};
use vd_client::ladder_view::{LadderView, fade_bands};
use vd_core::glam::DVec3;
use vd_core::pose::RealmId;
use vd_seed::bend::Face;
use vd_terrain::chunk::ChunkKey;
use vd_terrain::home::{HOME_PLANET_SEED, home_planet, home_solve_words};
use vd_wire::channels::BulkMsg;

fn main() {
    let a: Vec<f64> = std::env::args()
        .skip(1)
        .map(|s| s.parse().expect("a number"))
        .collect();
    assert_eq!(a.len(), 3, "ex ey ez");
    let eye = DVec3::new(a[0], a[1], a[2]);
    let artifact = run_solve(&SolveJob {
        body: home_planet(),
        words: home_solve_words(),
    })
    .expect("the home planet solves");
    let wet = home_planet().with_sea_m(artifact.sea());
    // ★ THE BAND THE HEAD SIZES (ruling B55), as the client's lane takes it from the same head.
    let drawn = artifact.drawn_range_m(&wet.macro_lattice().expect("a lattice"));
    let body = wet.with_band_m(drawn.0, drawn.1).unwrap_or(wet);
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
        low_m: drawn.0,
        high_m: drawn.1,
        wind_parts: 0,
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
    let wanted = view.wanted(&body, eye.to_array(), Some(&cache));
    println!(
        "skirt_gap: the eye {:.1} m over the drawn ground, rungs {}..{}, {} chunks wanted",
        wanted.altitude_m,
        wanted.rung_min,
        wanted.rung_max,
        wanted.keys.len()
    );
    let parents = ParentCache::default();
    // Per rung: skirt bottoms read, those over the sunk coarser ground, and every gap.
    let mut per_rung: BTreeMap<u8, (u64, u64, Vec<f64>, f64)> = BTreeMap::new();
    // Per rung: skirt vertices standing OVER their own rung's drawn ground by more than a cell — a
    // skirt piercing the ground toward the sky — and the highest.
    let mut pierce: BTreeMap<u8, (u64, f64)> = BTreeMap::new();
    for k in wanted.keys.iter().filter(|k| k.rung <= 3) {
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
        let coarser = (key.rung + 1).min(rungs - 1);
        let sink = sink_m(&body, coarser);
        let (_, out) = fade_bands(key.rung, rungs);
        let o = DVec3::from_array(g.origin_m);
        let entry = per_rung.entry(key.rung).or_insert((0, 0, Vec::new(), 0.0));
        // The skirt's own depth, read off its first strip: a bottom less its top.
        for j in g.skirt_start as usize..g.vertices.len() {
            let v = g.vertices[j];
            let p = o + DVec3::new(f64::from(v[0]), f64::from(v[1]), f64::from(v[2]));
            // Only where the finer is whole and the coarser sunk whole: nearer than the fade-out band.
            if (p - eye).length() >= out[0] {
                continue;
            }
            let u = p.normalize();
            let cell = f64::from(vd_seed::ladder::cell_m(key.rung));
            if let Some(own) = drawn_field_m(&body, Some(&cache), u.to_array(), key.rung) {
                let over = p.length() - own;
                let e = pierce.entry(key.rung).or_insert((0, f64::MIN));
                e.0 += u64::from(over > cell);
                e.1 = e.1.max(over);
            }
            let Some(ground) = drawn_field_m(&body, Some(&cache), u.to_array(), coarser) else {
                continue;
            };
            let gap = p.length() - (ground - sink);
            entry.0 += 1;
            entry.1 += u64::from(gap > 0.0);
            entry.2.push(gap);
            entry.3 = sink;
        }
    }
    for (rung, (n, over, mut gaps, sink)) in per_rung {
        gaps.sort_by(f64::total_cmp);
        let q = |f: f64| {
            gaps.get(((gaps.len() as f64 - 1.0) * f) as usize)
                .copied()
                .unwrap_or(f64::NAN)
        };
        println!(
            "rung {rung}: skirt drop {:.2} m, the coarser rung sunk {sink:.2} m under it; {n} skirt bottoms inside the whole band; {over} ({:.1} %) stand OVER the sunk coarser ground — gap median {:.2} m, p90 {:.2} m, max {:.2} m",
            vd_client::chunks::skirt_drop_m(&body, rung),
            100.0 * over as f64 / n.max(1) as f64,
            q(0.5),
            q(0.9),
            q(1.0)
        );
    }
    for (rung, (n, worst)) in pierce {
        println!(
            "rung {rung}: {n} skirt vertices stand more than a cell OVER their own drawn ground (a skirt piercing toward the sky); the highest {worst:+.2} m"
        );
    }
}
