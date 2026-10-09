//! ★ WHAT A PIXEL'S RAY MEETS (2026-09-24; the owner, low over the belt's highland: *"the
//! underneath of the first rung is visible when flying"*).
//!
//! **WHY IT EXISTS.** A capture 494 m over the highland shows dark specks whose probe says RUNG 0 at
//! 941–955 m, just inside rung 0's far edge (956 m). The shadows, the skirts seen from behind and the
//! ground's own morph shade are each refuted by measurement (`morph_normal_probe`). This casts the
//! capture camera's own ray through a stated pixel and lists EVERY triangle it crosses in the wanted
//! chunks of the finest rungs, each placed the way the crossfade shader places it (the morph toward
//! the coarser rung across the fade-out band, the sink under the finer rung across the fade-in band,
//! the fragment ended at and past the far edge, the back face culled), so the first front face the
//! ray meets and the shader keeps is what the pixel shows — and whether it is the GROUND, a SKIRT or a
//! CAVE surface under the ground.
//!
//! **WHAT IT READS.** The home planet solved once, the client's cache filled through its own
//! receiver, the ladder's own descent from the stated eye, and the chunks the client builds
//! (`vd_client::chunks::geometry_with`). The eye and the camera's rotation are the capture's state
//! file's (`terrain_stamp.eye_body_m`, `terrain_stamp.camera_body_xyzw`); the frame is the headless
//! capture's 1284 × 720 at the reference view's 45° of vertical field.
//!
//! `cargo run --release -p vd-bins --example ray_speck -- ex ey ez qx qy qz qw px py [px py ...]`
//!
//! **Example.** Pixel (630, 543) of the 494 m stand: the ray meets rung 0's ground at 955 m, which
//! the shader ends at 956 m — and what it meets next, and on which side, is the speck's answer.

use vd_bins::artifact_worker::{SolveJob, run_solve};
use vd_client::artifact_book::ArtifactReceiver;
use vd_client::chunks::{ParentCache, drawn_field_m, geometry_with};
use vd_client::ladder_view::{LadderView, fade_bands, sink_end_m};
use vd_core::glam::{DQuat, DVec3};
use vd_core::pose::RealmId;
use vd_seed::bend::Face;
use vd_terrain::chunk::ChunkKey;
use vd_terrain::home::{HOME_PLANET_SEED, home_planet, home_solve_words};
use vd_wire::channels::BulkMsg;

const W: f64 = 1284.0;
const H: f64 = 720.0;

/// One crossing of a ray with a drawn triangle.
struct Hit {
    t: f64,
    rung: u8,
    class: &'static str,
    front: bool,
    kept: bool,
    below_ground_m: f64,
    key: ChunkKey,
    /// The shade the shader draws at the crossing: the three vertices' own normals blended toward
    /// their morph normals by the fade-out weight, interpolated over the triangle.
    shade: DVec3,
    /// Whether a corner of the triangle stands on the chunk box's own boundary (its smooth normal
    /// then reads only this chunk's triangles).
    on_box_edge: bool,
    /// The triangle's three corners in the box's own quanta (the extractor's integers), `None` for a
    /// skirt's.
    corners: Option<[[i16; 3]; 3]>,
}

/// Möller–Trumbore: the distance along the unit ray from `o` to the triangle, if it crosses.
fn ray_tri(o: DVec3, d: DVec3, a: DVec3, b: DVec3, c: DVec3) -> Option<f64> {
    let e1 = b - a;
    let e2 = c - a;
    let p = d.cross(e2);
    let det = e1.dot(p);
    if det.abs() < 1e-12 {
        return None;
    }
    let inv = 1.0 / det;
    let s = o - a;
    let u = s.dot(p) * inv;
    if !(0.0..=1.0).contains(&u) {
        return None;
    }
    let q = s.cross(e1);
    let v = d.dot(q) * inv;
    if (v < 0.0) | (u + v > 1.0) {
        return None;
    }
    let t = e2.dot(q) * inv;
    (t > 0.0).then_some(t)
}

/// The barycentric weights of the crossing (for the shade).
fn bary(o: DVec3, d: DVec3, a: DVec3, b: DVec3, c: DVec3) -> (f64, f64) {
    let e1 = b - a;
    let e2 = c - a;
    let p = d.cross(e2);
    let inv = 1.0 / e1.dot(p);
    let s = o - a;
    let q = s.cross(e1);
    (s.dot(p) * inv, d.dot(q) * inv)
}

fn main() {
    let a: Vec<f64> = std::env::args()
        .skip(1)
        .map(|s| s.parse().expect("a number"))
        .collect();
    assert!(
        a.len() >= 12 && (a.len() - 10).is_multiple_of(2),
        "ex ey ez qx qy qz qw sx sy sz px py [px py ...]"
    );
    let eye = DVec3::new(a[0], a[1], a[2]);
    let q = DQuat::from_xyzw(a[3], a[4], a[5], a[6]).normalize();
    let sun = DVec3::new(a[7], a[8], a[9]).normalize();
    let pixels: Vec<(f64, f64)> = a[10..].chunks(2).map(|p| (p[0], p[1])).collect();
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
    // The camera: forward is −Z, up +Y, right +X of its rotation (the engine's convention).
    let forward = q * DVec3::NEG_Z;
    let up = q * DVec3::Y;
    let right = q * DVec3::X;
    let tan_half = (vd_core::geometry::REFERENCE_VIEW_FOV_Y_RAD * 0.5).tan();
    let rays: Vec<DVec3> = pixels
        .iter()
        .map(|(px, py)| {
            let nx = (px + 0.5) / W * 2.0 - 1.0;
            let ny = 1.0 - (py + 0.5) / H * 2.0;
            (forward + right * (nx * tan_half * (W / H)) + up * (ny * tan_half)).normalize()
        })
        .collect();
    let mut view = LadderView::default();
    let wanted = view.wanted(&body, eye.to_array(), Some(&cache));
    println!(
        "ray_speck: the eye {:.1} m over the drawn ground, rungs {}..{}, {} chunks wanted",
        wanted.altitude_m,
        wanted.rung_min,
        wanted.rung_max,
        wanted.keys.len()
    );
    for rung in 0..=2u8 {
        let (fi, fo) = fade_bands(rung, rungs);
        println!(
            "  rung {rung}: fade-in {:.0}..{:.0} m (sink ends {:.0} m, sink {:.2} m), fade-out {:.0}..{:.0} m",
            fi[0],
            fi[1],
            sink_end_m(&body, rung, rungs),
            vd_client::chunks::sink_m(&body, rung),
            fo[0],
            fo[1]
        );
    }
    let parents = ParentCache::default();
    let mut hits: Vec<Vec<Hit>> = (0..rays.len()).map(|_| Vec::new()).collect();
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
        // The same extraction the build ran, for the corners' integers (the vertex order is the
        // build's own up to the skirts).
        let mesh = vd_client::chunks::field_of(&body, Some(&cache), key)
            .and_then(|field| {
                vd_terrain::lattice::sample_box(
                    &body,
                    field
                        .as_deref()
                        .map(|f| f as &dyn vd_terrain::artifact::ZField),
                    key,
                )
            })
            .map(|samples| vd_terrain::extract::extract(&samples));
        let (fi, fo) = fade_bands(key.rung, rungs);
        let in_lo = fi[0];
        let sink_end = sink_end_m(&body, key.rung, rungs);
        let (out_lo, out_hi) = (fo[0], fo[1]);
        let sink = f64::from(g.sink_m);
        let o = DVec3::from_array(g.origin_m);
        let whole_at = |p: DVec3| ((out_hi - p.length()) / (out_hi - out_lo)).clamp(0.0, 1.0);
        let shade_of = |i: usize, w: f64| -> DVec3 {
            let own = DVec3::from_array(g.normals[i].map(f64::from)).normalize_or_zero();
            let morph =
                DVec3::from_array(vd_client::chunks::oct_decode(g.morph_normals[i]).map(f64::from));
            (morph * (1.0 - w) + own * w).normalize_or_zero()
        };
        // The box's own boundary, in the vertex's quanta: a corner on it reads a partial fan.
        let edge_q = f64::from(vd_terrain::chunk::CHUNK_EDGE as i32)
            * f64::from(vd_terrain::VERTEX_QUANTUM as i16);
        let _ = edge_q;
        // Every vertex where the shader puts it, in the eye-centred frame.
        let placed: Vec<DVec3> = (0..g.vertices.len())
            .map(|i| {
                let v = g.vertices[i];
                let own = o + DVec3::new(f64::from(v[0]), f64::from(v[1]), f64::from(v[2])) - eye;
                let d = own.length();
                let r = DVec3::new(
                    f64::from(g.radials[i][0]),
                    f64::from(g.radials[i][1]),
                    f64::from(g.radials[i][2]),
                )
                .normalize();
                let coarser = own + r * f64::from(g.morph_m[i]);
                let whole = ((out_hi - d) / (out_hi - out_lo)).clamp(0.0, 1.0);
                let risen = ((d - in_lo) / (sink_end - in_lo)).clamp(0.0, 1.0);
                coarser * (1.0 - whole) + own * whole - r * sink * (1.0 - risen)
            })
            .collect();
        // A cheap reject: the chunk's bounding sphere against each ray.
        let centre = placed.iter().fold(DVec3::ZERO, |s, p| s + *p) / placed.len().max(1) as f64;
        let radius = placed
            .iter()
            .map(|p| (*p - centre).length())
            .fold(0.0, f64::max);
        for (ri, ray) in rays.iter().enumerate() {
            let along = centre.dot(*ray);
            let miss = (centre - *ray * along).length();
            if (miss > radius) | (along + radius < 0.0) {
                continue;
            }
            for t in &g.triangles {
                let (pa, pb, pc) = (
                    placed[t[0] as usize],
                    placed[t[1] as usize],
                    placed[t[2] as usize],
                );
                let Some(dist) = ray_tri(DVec3::ZERO, *ray, pa, pb, pc) else {
                    continue;
                };
                let n = (pb - pa).cross(pc - pa);
                let skirt = t.iter().any(|i| *i >= g.skirt_start);
                let hit = *ray * dist + eye;
                let (u, v) = bary(DVec3::ZERO, *ray, pa, pb, pc);
                let w = whole_at(*ray * dist);
                let shade = (shade_of(t[0] as usize, w) * (1.0 - u - v)
                    + shade_of(t[1] as usize, w) * u
                    + shade_of(t[2] as usize, w) * v)
                    .normalize_or_zero();
                let ground =
                    drawn_field_m(&body, Some(&cache), hit.normalize().to_array(), key.rung);
                let below = ground.map_or(f64::NAN, |gr| gr - hit.length());
                let cell = f64::from(vd_seed::ladder::cell_m(key.rung));
                let class = if skirt {
                    "skirt"
                } else if below > 2.0 * cell {
                    "cave"
                } else {
                    "ground"
                };
                hits[ri].push(Hit {
                    t: dist,
                    rung: key.rung,
                    class,
                    front: n.dot(*ray) < 0.0,
                    kept: dist < out_hi,
                    below_ground_m: below,
                    key,
                    shade,
                    on_box_edge: false,
                    corners: mesh.as_ref().and_then(|m| {
                        (!skirt).then(|| {
                            [
                                m.vertices[t[0] as usize],
                                m.vertices[t[1] as usize],
                                m.vertices[t[2] as usize],
                            ]
                        })
                    }),
                });
            }
        }
    }
    for (ri, (px, py)) in pixels.iter().enumerate() {
        let list = &mut hits[ri];
        list.sort_by(|a, b| a.t.total_cmp(&b.t));
        let shown = list.iter().position(|h| h.front & h.kept);
        println!(
            "pixel ({px:.0}, {py:.0}): {} crossings; the first front face the shader keeps is #{}",
            list.len(),
            shown.map_or("none".to_owned(), |s| s.to_string())
        );
        for (i, h) in list.iter().enumerate().take(8) {
            let radial = (rays[ri] * h.t + eye).normalize();
            println!(
                "    #{i} {:.2} m  rung {}  {:6}  {}  {}  {:+.2} m under the ground  {:?}  shade {:.1}° off the radial, lit {:+.3} (n·sun)",
                h.t,
                h.rung,
                h.class,
                if h.front { "front" } else { "back " },
                if h.kept { "kept" } else { "ENDED" },
                h.below_ground_m,
                (h.key.face, h.key.x, h.key.y, h.key.z),
                h.shade.dot(radial).clamp(-1.0, 1.0).acos().to_degrees(),
                h.shade.dot(sun)
            );
            if let Some(c) = h.corners {
                println!(
                    "         corners (quanta; a cell is {} quanta, the box 0..{}): {:?}",
                    vd_terrain::VERTEX_QUANTUM,
                    vd_terrain::chunk::CHUNK_EDGE as i32
                        * i32::from(vd_terrain::VERTEX_QUANTUM as i16),
                    c
                );
            }
        }
        let _ = list.iter().any(|h| h.on_box_edge);
    }
}
