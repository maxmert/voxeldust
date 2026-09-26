//! ★ COST (plan §8.6, readings): the shipped chunk build over the whole rung-7 and rung-9 rings a
//! hull 150 km over the plain and the belt asks for, one thread, one shared parent cache (the
//! parents built as the client builds them); the share of those chunks whose node block holds a
//! wet node (they would run a candidate path, finding 22); and the what-if's own extra time per
//! column, one thread, by candidate count.
//!
//! **Example.** A hull 150 km over the belt asks for 212 rung-7 chunks; each takes a median of
//! 18 ms to build with its parents; 61 % of them hold a lake or the sea in their node block.

use std::time::Instant;

use vd_bins::shore_rules::{Nodes, WORD_LAKE, WORD_SEA};
use vd_client::artifact_book::ArtifactCache;
use vd_client::chunks::{ParentCache, drawn_surface_m, geometry_with};
use vd_client::ladder_view::LadderView;
use vd_core::glam::DVec3;
use vd_core::pose::RealmId;
use vd_terrain::BodyDefinition;
use vd_terrain::artifact::nodes_of_chunk;
use vd_terrain::chunk::ChunkKey;

use crate::census::{Census, val};

/// ★ THE SHIPPED BUILD'S COST over the rings a hull 150 km over `dir` asks for, at `rungs`.
#[allow(clippy::too_many_arguments, reason = "a reading's own words")]
pub fn shipped(
    c: &mut Census,
    body: &BodyDefinition,
    cache: &ArtifactCache,
    realm: RealmId,
    nodes: &Nodes,
    name: &str,
    dir: DVec3,
    rungs: &[u8],
) {
    let fine = body.macro_lattice().expect("a lattice");
    let ground = drawn_surface_m(body, Some(cache), dir.to_array());
    let eye = dir * (ground + 150_000.0);
    let mut view = LadderView::default();
    let wanted = view.wanted(body, eye.to_array(), Some(cache));
    c.say(format!(
        "  COST {name}: the eye {:.0} m over the drawn ground, rungs {}..{}, {} chunks wanted",
        wanted.altitude_m,
        wanted.rung_min,
        wanted.rung_max,
        wanted.keys.len()
    ));
    let parents = ParentCache::default();
    for &rung in rungs {
        let keys: Vec<ChunkKey> = wanted
            .keys
            .iter()
            .filter(|k| k.rung == rung)
            .map(|k| ChunkKey {
                face: k.face,
                rung: k.rung,
                x: k.x,
                y: k.y,
                z: k.z,
            })
            .collect();
        let mut ms = Vec::with_capacity(keys.len());
        let mut wet_blocks = 0usize;
        let t_all = Instant::now();
        for key in &keys {
            let t = Instant::now();
            let _ = geometry_with(body, Some(cache), realm, *key, &parents);
            ms.push(t.elapsed().as_secs_f64() * 1_000.0);
            let wet = nodes_of_chunk(&fine, *key).iter().any(|&n| {
                let w = nodes.word[n as usize];
                (w == WORD_SEA) | (w == WORD_LAKE)
            });
            wet_blocks += usize::from(wet);
        }
        let total = t_all.elapsed().as_secs_f64() * 1_000.0;
        ms.sort_by(f64::total_cmp);
        let q = |f: f64| {
            if ms.is_empty() {
                f64::NAN
            } else {
                ms[((ms.len() as f64 * f) as usize).min(ms.len() - 1)]
            }
        };
        c.reading(
            &format!("COST-SHIPPED.{name}.r{rung:02}"),
            &format!(
                "{}chunks/{}total/p50{}/p90{}perchunk",
                keys.len(),
                val(total, 0, "ms"),
                val(q(0.5), 1, "ms"),
                val(q(0.9), 1, "ms")
            ),
            "one-thread;parents-included;no-pacing",
        );
        c.reading(
            &format!("CANDIDATE-PATH-SHARE.{name}.r{rung:02}"),
            &format!(
                "{}%/{wet_blocks}of{}chunks",
                val(100.0 * wet_blocks as f64 / keys.len().max(1) as f64, 1, ""),
                keys.len()
            ),
            "chunks-whose-node-block-holds-a-wet-node(F22)",
        );
    }
}
