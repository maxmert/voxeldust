//! ★ THE WATER'S BENCH (B99 step 4; the one-volume revision `docs/investigation/2026-10-15/one_volume_revision.md`,
//! ruled by B105). Read only: nothing it does changes the world or the store.
//!
//! - `pits X,Y,Z HALF_KM STEP_M [coarse]` — ★ G-PIT-RASTER (the revision §3.10, a reading): the drawn ground around the
//!   stand at `X,Y,Z` (the planet's frame), sampled on a square grid of `STEP_M` over ±`HALF_KM` in the stand's tangent
//!   plane — the finest stored level (the 16 m ground), or with `coarse` the 128 m ground alone. Every water spreads from
//!   its own 8 km nodes through the points under its level: the sea at its level from the ocean's nodes, each lake at its
//!   water word from its own nodes. A point under the sea's level, or under its own lake node's level, that no water
//!   reaches is a PIT — a hollow the water fill would leave dry or seal; the counts, the areas and the largest pit.
//!
//! - `sides X,Y,Z WORLD.art HALF_KM` — the coast mask's words against the rows: the count of each side on the whole
//!   body, the nodes where the mask and the row's own side disagree, and a map of the sides around the stand (one
//!   character a node: `#` land, `~` the sea, `L` a lake, `p` a pocket's floor, `?` no word).
//!
//! - `reach WORLD.art` — the loading screen's wait (2026-10-18): for each rung the 128 m level serves, the client's
//!   effective switch distance (the ask bound as the live client builds it: the body's handover steps, the T7 floor, the
//!   fine ground's rows) beside the plain rule the shard's tile reach reads (`cell ÷ pixel`).
//!
//! - `rough X,Y,Z WORLD.art FACE RUNG CX CY CZ` — one waiting piece (the loading screen's diagnostic names it): for the
//!   piece and its parents, every 128 m tile each reads, whether the fine reach names it from the eye at `X,Y,Z` (the
//!   gateway's call on the head's band), at which mean level, and the tile's distance from the eye.
//!
//! - `wind WORLD.art X,Y,Z [THREADS]` — the sea's wind (ruling B109): the seasons' chain run once on the stored world, its
//!   time, the speeds over the sea, and the wind at the stand `X,Y,Z`.
//!
//! `VD_REALM_STORE=<a copy of the home planet's store> cargo run --release -p vd-bins --example water_bench -- pits
//! -5919734.9,-1422233.0,-1789183.2 6 16`
//!
//! **Example.** On a coast the sea's nodes reach every point of the bay under the sea's level; a hollow behind the dune
//! 1 m under that level, which no path under the level joins to the bay, is one pit of the reading.

use std::collections::BTreeSet;
use std::sync::Arc;

use vd_core::glam::DVec3;
use vd_core::pose::RealmId;
use vd_seed::bend::Face;
use vd_sim::stub::built_store::{FineAt, FineForm, read_fine_record};
use vd_terrain::BodyDefinition;
use vd_terrain::artifact::{DRY_M, FACIES_SHIFT, ZField};
use vd_terrain::fine::l1::TileKey;
use vd_terrain::fine::level::FineLevel;
use vd_terrain::fine::read::{FineBook, FineField};
use vd_terrain::fine::tile::{decode_levels, input_word_of};

const UNIVERSE: u64 = vd_bins::DEV.universe_seed;
const HOME: u64 = 4_030_111_653_607_004_909;

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.first().map(String::as_str) == Some("sides") {
        sides(&args);
        return;
    }
    if args.first().map(String::as_str) == Some("reach") {
        reach(&args);
        return;
    }
    if args.first().map(String::as_str) == Some("rough") {
        rough(&args);
        return;
    }
    if args.first().map(String::as_str) == Some("wind") {
        wind(&args);
        return;
    }
    assert_eq!(
        args.first().map(String::as_str),
        Some("pits"),
        "pits X,Y,Z HALF_KM STEP_M [coarse]"
    );
    let eye: Vec<f64> = args
        .get(1)
        .expect("X,Y,Z in the planet's frame")
        .split(',')
        .map(|x| x.trim().parse().expect("a number"))
        .collect();
    let half_m = args.get(2).map_or(6.0, |s| s.parse().expect("HALF_KM")) * 1_000.0;
    let step_m: f64 = args.get(3).map_or(16.0, |s| s.parse().expect("STEP_M"));
    let coarse = args.get(4).map(String::as_str) == Some("coarse");
    // The home planet, its store and its artifact, as `valley_hull` reads them.
    let realm = RealmId::Planet(HOME);
    let (rows, _) = vd_bins::home_system_world(UNIVERSE);
    let radius_m = rows
        .iter()
        .find(|r| r.realm == realm)
        .and_then(|r| match r.look {
            Some(vd_core::geometry::Boundary::Shell { r }) => Some(r),
            _ => None,
        })
        .expect("the home planet is round");
    let charter = vd_bins::body_charter(
        UNIVERSE,
        &BTreeSet::from([vd_core::worldgen::HOME_SYSTEM]),
        &BTreeSet::from([vd_core::worldgen::GALAXY]),
        realm,
    )
    .expect("a charter");
    let body = BodyDefinition::from_seed(HOME, radius_m, vd_bins::facts_of_charter(&charter))
        .expect("a ladder");
    let lattice = body.macro_lattice().expect("a lattice");
    let store = vd_bins::open_realm_store(&vd_bins::process_env(), realm)
        .expect("the store opens")
        .expect("VD_REALM_STORE names a file");
    let artifact = vd_bins::artifact_store::read_artifact(
        &store,
        vd_terrain::artifact_world_tag(UNIVERSE),
        &lattice,
    )
    .expect("the stored rows agree with their head")
    .expect("an artifact is stored");
    let at = DVec3::new(eye[0], eye[1], eye[2]);
    let up = at.normalize();
    let drawn = artifact.drawn_range_m(&lattice);
    let sea_r = body.radius_m() + f64::from(artifact.sea_m);
    // Each 8 km node's water: the sea (level 0 over the sea), a lake (its water word over the sea), or none.
    let node_level = |i: usize| -> Option<f64> {
        let r = &artifact.rows[i];
        if (r.receiver_facies >> FACIES_SHIFT) & vd_terrain::solve::FACIES_SEA != 0 {
            Some(0.0)
        } else if r.water_m != DRY_M && r.water_m > r.z_m {
            Some(f64::from(r.water_m) - f64::from(artifact.sea_m))
        } else {
            None
        }
    };
    let body = body.with_sea_m(artifact.sea());
    let book = |level: FineLevel| {
        let mut book =
            FineBook::new(level, &body, &lattice, artifact.sea_m).expect("the level applies");
        let input = input_word_of(level, artifact.digest());
        for ((f, tx, ty), _) in vd_terrain::artifact::fine_needs_of_head(
            level,
            &body,
            (drawn.0, drawn.1, artifact.sea_m),
            up.to_array(),
            at.length(),
            vd_core::geometry::drawable_theta_min_rad(),
        ) {
            let fat = FineAt {
                form: FineForm::Finished,
                input,
                level: level.id(),
                face: f,
                tx,
                ty,
            };
            if let Ok(Some(bytes)) = read_fine_record(&store, &fat)
                && let Some((_, levels)) = decode_levels(&bytes, level)
            {
                let key = TileKey {
                    face: Face::from_index(f).expect("a face"),
                    tx: tx as i32,
                    ty: ty as i32,
                };
                book.insert(key, levels);
            }
        }
        println!("{level:?}: {} stored tiles held", book.len());
        Arc::new(book)
    };
    let (l1, l2) = (book(FineLevel::L1), book(FineLevel::L2));
    let node_of = |d: DVec3| vd_bins::drainage_gates::node_under(&lattice, d) as usize;
    let levels: Vec<Option<f64>> = (0..artifact.rows.len()).map(node_level).collect();
    let inner: Arc<dyn ZField + Send + Sync> = Arc::new(artifact);
    let field = FineField {
        inner,
        book: l1,
        finer: (!coarse).then_some(l2),
    };
    // The grid in the stand's tangent plane: each point's height over the sea and its 8 km node's water level.
    let east = DVec3::Z.cross(up).normalize();
    let north = up.cross(east);
    let side = (2.0 * half_m / step_m).round() as usize + 1;
    let mut h = vec![0f64; side * side];
    let mut water = vec![None::<f64>; side * side];
    for y in 0..side {
        for x in 0..side {
            let (dx, dy) = (x as f64 * step_m - half_m, y as f64 * step_m - half_m);
            let d = (up * sea_r + east * dx + north * dy).normalize();
            let g = vd_terrain::height::height_field_m(&body, &field, d.to_array(), 0)
                .expect("the ground is held");
            h[y * side + x] = g - sea_r;
            water[y * side + x] = levels[node_of(d)];
        }
    }
    // Every distinct water level spreads from its own nodes' points under it, through the points under it.
    let mut wet = vec![false; side * side];
    let mut distinct: Vec<f64> = water.iter().flatten().copied().collect();
    distinct.sort_by(f64::total_cmp);
    distinct.dedup();
    for &level in &distinct {
        let mut seen = vec![false; side * side];
        let mut stack: Vec<usize> = (0..side * side)
            .filter(|&k| water[k] == Some(level) && h[k] < level)
            .collect();
        for &k in &stack {
            seen[k] = true;
        }
        while let Some(k) = stack.pop() {
            wet[k] = true;
            let (x, y) = (k % side, k / side);
            let around = [
                (x > 0).then(|| k - 1),
                (x + 1 < side).then(|| k + 1),
                (y > 0).then(|| k - side),
                (y + 1 < side).then(|| k + side),
            ];
            for m in around.into_iter().flatten() {
                if !seen[m] && h[m] < level {
                    seen[m] = true;
                    stack.push(m);
                }
            }
        }
    }
    // The pits: under the sea's level, or under the own node's lake level, and reached by no water.
    let under = |k: usize| h[k] < 0.0 || water[k].is_some_and(|l| h[k] < l);
    let cell_km2 = step_m * step_m / 1e6;
    let n_under = (0..side * side).filter(|&k| under(k)).count();
    let n_wet = (0..side * side).filter(|&k| wet[k]).count();
    let mut pit_seen = vec![false; side * side];
    let mut pits: Vec<(usize, f64)> = Vec::new();
    for start in 0..side * side {
        if pit_seen[start] || !under(start) || wet[start] {
            continue;
        }
        let (mut stack, mut size, mut deepest) = (vec![start], 0usize, 0f64);
        pit_seen[start] = true;
        while let Some(k) = stack.pop() {
            size += 1;
            deepest = deepest.min(h[k]);
            let (x, y) = (k % side, k / side);
            let around = [
                (x > 0).then(|| k - 1),
                (x + 1 < side).then(|| k + 1),
                (y > 0).then(|| k - side),
                (y + 1 < side).then(|| k + side),
            ];
            for m in around.into_iter().flatten() {
                if !pit_seen[m] && under(m) && !wet[m] {
                    pit_seen[m] = true;
                    stack.push(m);
                }
            }
        }
        pits.push((size, deepest));
    }
    pits.sort_by(|a, b| b.0.cmp(&a.0));
    let in_pits: usize = pits.iter().map(|p| p.0).sum();
    let land = (0..side * side).filter(|&k| h[k] >= 0.0).count();
    println!(
        "G-PIT-RASTER on the {} ground, ±{:.1} km at {step_m} m ({side} × {side} points, {:.1} km²): over the sea {:.1} %; under a water's level {} points ({:.2} km²), reached by its water {} ({:.2} km²); PITS {} points ({:.3} km², {:.2} % of the points under a level) in {} pits, the largest {} points ({:.3} km², its floor {:.1} m)",
        if coarse {
            "128 m"
        } else {
            "finest stored (16 m)"
        },
        half_m / 1_000.0,
        (side * side) as f64 * cell_km2,
        100.0 * land as f64 / (side * side) as f64,
        n_under,
        n_under as f64 * cell_km2,
        n_wet,
        n_wet as f64 * cell_km2,
        in_pits,
        in_pits as f64 * cell_km2,
        100.0 * in_pits as f64 / n_under.max(1) as f64,
        pits.len(),
        pits.first().map_or(0, |p| p.0),
        pits.first().map_or(0.0, |p| p.0 as f64 * cell_km2),
        pits.first().map_or(0.0, |p| p.1)
    );
    let sizes = [1usize, 10, 100, 1000]
        .iter()
        .map(|&b| pits.iter().filter(|p| p.0 >= b).count())
        .collect::<Vec<_>>();
    println!(
        "  pits by size: 1+ points {}, 10+ {}, 100+ {}, 1 000+ {}",
        sizes[0], sizes[1], sizes[2], sizes[3]
    );
}

/// The home planet's body and lattice, as `valley_hull` reads them.
fn home_body() -> (BodyDefinition, vd_terrain::macro_lattice::MacroLattice) {
    let realm = RealmId::Planet(HOME);
    let (rows, _) = vd_bins::home_system_world(UNIVERSE);
    let radius_m = rows
        .iter()
        .find(|r| r.realm == realm)
        .and_then(|r| match r.look {
            Some(vd_core::geometry::Boundary::Shell { r }) => Some(r),
            _ => None,
        })
        .expect("the home planet is round");
    let charter = vd_bins::body_charter(
        UNIVERSE,
        &BTreeSet::from([vd_core::worldgen::HOME_SYSTEM]),
        &BTreeSet::from([vd_core::worldgen::GALAXY]),
        realm,
    )
    .expect("a charter");
    let body = BodyDefinition::from_seed(HOME, radius_m, vd_bins::facts_of_charter(&charter))
        .expect("a ladder");
    let lattice = body.macro_lattice().expect("a lattice");
    (body, lattice)
}

/// `sides X,Y,Z WORLD.art HALF_KM`: the coast mask's words against the rows, and their map around the stand.
fn sides(args: &[String]) {
    use vd_terrain::artifact::{Side, coast_side, side_of_row};
    let eye: Vec<f64> = args[1]
        .split(',')
        .map(|x| x.trim().parse().expect("a number"))
        .collect();
    let a = vd_bins::artifact_file::read(args.get(2).expect("WORLD.art")).expect("the world reads");
    let half_m = args.get(3).map_or(60.0, |s| s.parse().expect("HALF_KM")) * 1_000.0;
    let (body, lattice) = home_body();
    let sea_m = a.sea_m;
    println!(
        "the body's radius {:.1} m, the sea's radius {:.1} m",
        body.radius_m(),
        body.radius_m() + f64::from(sea_m)
    );
    let n = a.rows.len();
    let mut count = std::collections::BTreeMap::<String, usize>::new();
    let mut differ = std::collections::BTreeMap::<String, usize>::new();
    for i in 0..n {
        let r = &a.rows[i];
        let row = side_of_row(r.receiver_facies >> FACIES_SHIFT, r.z_m, sea_m);
        let mask = coast_side(&a.coast, i as u32);
        *count.entry(format!("{mask:?}")).or_default() += 1;
        if mask != Some(row) {
            *differ
                .entry(format!("mask {mask:?} row {row:?}"))
                .or_default() += 1;
        }
    }
    println!(
        "{n} nodes, the mask {} bytes, the sea at {sea_m} m",
        a.coast.len()
    );
    println!("the mask's words: {count:?}");
    println!("the mask against the rows: {differ:?}");
    let at = DVec3::new(eye[0], eye[1], eye[2]);
    let up = at.normalize();
    let east = DVec3::Z.cross(up).normalize();
    let north = up.cross(east);
    let step_m: f64 = 4_096.0;
    let side = (2.0 * half_m / step_m).round() as i64;
    for y in (0..=side).rev() {
        let line: String = (0..=side)
            .map(|x| {
                let (dx, dy) = (x as f64 * step_m - half_m, y as f64 * step_m - half_m);
                let d = (up + east * (dx / at.length()) + north * (dy / at.length())).normalize();
                let node = vd_bins::drainage_gates::node_under(&lattice, d);
                if x == side / 2 && y == side / 2 {
                    return '@';
                }
                match coast_side(&a.coast, node) {
                    Some(Side::Land) => '#',
                    Some(Side::Sea) => '~',
                    Some(Side::Lake) => 'L',
                    Some(Side::Pocket) => 'p',
                    None => '?',
                }
            })
            .collect();
        println!("{line}");
    }
}

/// `reach WORLD.art`: the client's switch distance per rung against the shard's plain rule.
fn reach(args: &[String]) {
    use vd_client::ladder_view::AskBound;
    let a = vd_bins::artifact_file::read(args.get(1).expect("WORLD.art")).expect("the world reads");
    let (body, lattice) = home_body();
    let body = body.with_sea_m(a.sea());
    let drawn = a.drawn_range_m(&lattice);
    let body = body
        .with_fine_band_m(drawn.0, drawn.1, a.sea_m)
        .unwrap_or(body);
    let rungs = body.ladder().rungs;
    let levels = a.pyramid.len() as u32;
    let bound = AskBound::unbounded()
        .for_body(&body, rungs, levels)
        .for_fine_ground(true, rungs);
    let pixel = vd_core::geometry::drawable_theta_min_rad();
    println!("rungs {rungs}, levels {levels}, the pixel {pixel:.3e} rad");
    let edge = vd_terrain::artifact::ASK_HYSTERESIS_OUT * (1.0 + vd_terrain::artifact::ASK_SLACK);
    let pyramid = vd_terrain::artifact::pyramid_levels(&lattice);
    for rung in 4..rungs.min(16) {
        let plain = vd_terrain::artifact::switch_m(rung, pixel);
        let step = vd_terrain::artifact::handover_step_m(&body, &lattice, pyramid, rung);
        let shard = vd_terrain::artifact::rung_ask_m(rung, pixel, step) / edge;
        println!(
            "rung {rung:>2}: the client's switch {:>12.0} m, the shard's {:>12.0} m, the plain rule {:>12.0} m ({:.2}×), the handover step {:>9.1} m",
            bound.switch_m(rung),
            shard,
            plain,
            bound.switch_m(rung) / plain,
            bound.step_m(rung)
        );
    }
    println!("the pyramid: the head's {levels} levels, the lattice's own {pyramid}");
}

/// `rough X,Y,Z WORLD.art FACE RUNG CX CY CZ`: a waiting piece's tiles against the fine reach's names.
fn rough(args: &[String]) {
    use vd_terrain::chunk::ChunkKey;
    use vd_terrain::fine::level::FineLevel;
    let eye: Vec<f64> = args[1]
        .split(',')
        .map(|x| x.trim().parse().expect("a number"))
        .collect();
    let a = vd_bins::artifact_file::read(args.get(2).expect("WORLD.art")).expect("the world reads");
    let num = |i: usize| -> i32 { args.get(i).expect("a chunk key").parse().expect("a number") };
    let face = Face::from_index(num(3) as u8).expect("a face 0-5");
    let key = ChunkKey {
        face,
        rung: num(4) as u8,
        x: num(5),
        y: num(6),
        z: num(7),
    };
    let (body, lattice) = home_body();
    let body = body.with_sea_m(a.sea());
    let drawn = a.drawn_range_m(&lattice);
    let at = DVec3::new(eye[0], eye[1], eye[2]);
    let pixel = vd_core::geometry::drawable_theta_min_rad();
    let of_head: std::collections::BTreeMap<(u8, u32, u32), u8> =
        vd_terrain::artifact::fine_needs_of_head(
            FineLevel::L1,
            &body,
            (drawn.0, drawn.1, a.sea_m),
            at.normalize().to_array(),
            at.length(),
            pixel,
        )
        .into_iter()
        .collect();
    // The shard's own call (`vd_bins::fine_source`): the ground its artifact states under the eye, the band's relief.
    let banded = body
        .with_fine_band_m(drawn.0, drawn.1, a.sea_m)
        .unwrap_or(body);
    let ground = vd_terrain::height::height_field_m(&body, &a, at.normalize().to_array(), 0)
        .unwrap_or(at.length());
    let named: std::collections::BTreeMap<(u8, u32, u32), u8> = vd_terrain::artifact::fine_needs(
        FineLevel::L1,
        &banded,
        &lattice,
        at.normalize().to_array(),
        at.length(),
        (ground, ground),
        banded.ground_range_m().1,
        pixel,
    )
    .into_iter()
    .collect();
    println!(
        "the gateway's reach names {} tiles, the shard's {} (the eye {:.0} m over the ground its artifact states)",
        of_head.len(),
        named.len(),
        at.length() - ground
    );
    let far = named
        .keys()
        .map(|&(f, tx, ty)| tile_distance_km(&lattice, at, (f, tx, ty)))
        .fold(0.0f64, f64::max);
    println!(
        "the reach names {} tiles of the 128 m level, the farthest {far:.0} km",
        named.len()
    );
    let book = FineBook::new(FineLevel::L1, &body, &lattice, a.sea_m).expect("the level applies");
    let mut keys = vec![key];
    keys.extend(vd_client::chunks::parent_keys(&body, key));
    for k in keys {
        let tiles = book.tiles_of_chunk(k.face, k.rung, k.x, k.y);
        let read = k.rung.saturating_sub(FineLevel::L1.rung());
        let missing: Vec<String> = tiles
            .iter()
            .filter_map(|t| {
                let id = (t.face.index(), t.tx as u32, t.ty as u32);
                let d = tile_distance_km(&lattice, at, id);
                match named.get(&id) {
                    Some(&level) if level <= read => None,
                    Some(&level) => Some(format!(
                        "{:?}/{}/{} named at level {level} ({d:.0} km)",
                        t.face, t.tx, t.ty
                    )),
                    None => Some(format!(
                        "{:?}/{}/{} NOT NAMED ({d:.0} km)",
                        t.face, t.tx, t.ty
                    )),
                }
            })
            .collect();
        println!(
            "{k:?} reads level {read}: {} tiles, short {}: {missing:?}",
            tiles.len(),
            missing.len()
        );
    }
}

/// The distance from the eye to a 128 m tile's centre node, along the eye's ray to it, in km.
fn tile_distance_km(
    lattice: &vd_terrain::macro_lattice::MacroLattice,
    at: DVec3,
    (f, tx, ty): (u8, u32, u32),
) -> f64 {
    let nodes = 1u32 << vd_terrain::fine::level::FineLevel::L1.tile_nodes_shift();
    let face = Face::from_index(f).expect("a face");
    let (i, j) = (tx * nodes + nodes / 2, ty * nodes + nodes / 2);
    let i = i.min(lattice.edge - 1) as i32;
    let j = j.min(lattice.edge - 1) as i32;
    let g = lattice.direction(lattice.index(face, i, j));
    let unit = (1u64 << vd_recipe::bend::DIR_BITS) as f64;
    let d = DVec3::new(
        g[0].raw() as f64 / unit,
        g[1].raw() as f64 / unit,
        g[2].raw() as f64 / unit,
    )
    .normalize();
    (d * at.length() - at).length() / 1e3
}

/// `wind WORLD.art X,Y,Z [THREADS]`: the sea's wind on the stored world.
fn wind(args: &[String]) {
    use vd_terrain::gf::Gf;
    let a = vd_bins::artifact_file::read(args.get(1).expect("WORLD.art")).expect("the world reads");
    let eye: Vec<f64> = args
        .get(2)
        .expect("X,Y,Z")
        .split(',')
        .map(|x| x.trim().parse().expect("a number"))
        .collect();
    let threads: usize = args.get(3).map_or(12, |s| s.parse().expect("THREADS"));
    let (body, lattice) = home_body();
    let charter = vd_bins::body_charter(
        UNIVERSE,
        &BTreeSet::from([vd_core::worldgen::HOME_SYSTEM]),
        &BTreeSet::from([vd_core::worldgen::GALAXY]),
        RealmId::Planet(HOME),
    )
    .expect("a charter");
    let words = vd_bins::solve_words_of(&charter);
    let started = std::time::Instant::now();
    let w = vd_terrain::sea_wind::sea_wind(&body, &lattice, &words, &a, threads).expect("a wind");
    println!(
        "the chain once on the stored world: {:.1} s on {threads} threads; {} cells, {} bytes",
        started.elapsed().as_secs_f64(),
        w.speed_cms.len(),
        w.to_bytes().len()
    );
    let mut speeds: Vec<u16> = w.speed_cms.clone();
    speeds.sort_unstable();
    let q = |p: f64| f64::from(speeds[((speeds.len() - 1) as f64 * p) as usize]) / 100.0;
    println!(
        "the year's mean speed over every cell: 5 % {:.1}, median {:.1}, 95 % {:.1}, most {:.1} m/s",
        q(0.05),
        q(0.5),
        q(0.95),
        q(1.0)
    );
    let grid =
        vd_terrain::seasons::grid::ClimateGrid::new(Gf::from_f64(body.radius_m()), w.rows as usize);
    let d = DVec3::new(eye[0], eye[1], eye[2]).normalize();
    let c = grid.cell_at([Gf::from_f64(d.x), Gf::from_f64(d.y), Gf::from_f64(d.z)]);
    println!(
        "at the stand (cell {c}): toward the east {:.2}, toward the north {:.2}, speed {:.2} m/s; Cox–Munk's slope variance {:.4}",
        f64::from(w.east_cms[c]) / 100.0,
        f64::from(w.north_cms[c]) / 100.0,
        f64::from(w.speed_cms[c]) / 100.0,
        0.003 + 0.00512 * f64::from(w.speed_cms[c]) / 100.0
    );
}
