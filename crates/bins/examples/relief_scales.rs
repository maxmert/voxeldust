//! ★ THE RELIEF AT THREE SCALES, THE HOME PLANET BESIDE EARTH (2026-10-13; step 2 of the owner's order "the berth → one
//! measurement → the owner's choice", after the window look: "on the sides there are mountains, but the rest is mostly
//! flat, no hills, nothing"). Bench only; nothing in the game changes.
//!
//! ONE RULE FOR EVERY SOURCE. Each source becomes one grid of ~128 m cells, metres over its sea:
//! - **the home planet's fine ground**: the finished 128 m tiles around the hull's berth read back from the realm's store
//!   (level 0, the tile's own square), laid side by side on their face;
//! - **Earth**: SRTM 1″ tiles (NASA SRTM v3 through the public `elevation-tiles-prod` mirror; outside the repository,
//!   ruling B36 item 7), averaged to ~128 m cells (4 rows × the columns that make 128 m at the tile's latitude).
//!
//! The readings, over windows that do not overlap and hold land only (every cell over the sea): the LOCAL RELIEF (the
//! window's highest less its lowest) at 1 km (8 cells), at 5 km (39 cells), and at 41 km over the 4 096 m means (10 × 10
//! means of 32 × 32 cells — the scale of the 8 km rows the fine ground is held to) — each as the 10th, 50th and 90th
//! percentiles and the shares under 20, 50 and 100 m. And ONE SIDE VIEW of each source with the same camera (2 km over the
//! ground, looking north 8° down, 60° wide, the sun in the south-west 25° up, the haze by distance).
//! `VD_REALM_STORE=<home planet store> cargo run --release -p vd-bins --example relief_scales -- <out dir> <earth .hgt>…`

use std::collections::BTreeSet;
use std::io::Read;

use vd_core::pose::RealmId;
use vd_seed::bend::Face;
use vd_sim::stub::built_store::{FineAt, FineForm, read_fine_record};
use vd_terrain::BodyDefinition;
use vd_terrain::fine::tile::{decode_levels, input_word};

const UNIVERSE: u64 = vd_bins::DEV.universe_seed;
const HOME: u64 = 4_030_111_653_607_004_909;
/// Hull 15's standing centre over the home planet (metres in the planet's frame; the dev cluster's spawn).
const BERTH: [f64; 3] = [
    3_273_359.478_594_958_3,
    -2_850_257.825_591_502_7,
    -4_631_027.492_918_959,
];
/// The fine record's level (the store's key for the 128 m level).
const LEVEL_L1: u8 = 1;
/// The cells a window spans: 1 km and 5 km at 128 m, and 10 means of 4 096 m.
const WIN_1KM: usize = 8;
const WIN_5KM: usize = 39;
const MEAN_BLOCK: usize = 32;
const WIN_MEANS: usize = 10;

/// One grid of heights in metres over the sea, row 0 at the north; `NaN` where the source holds nothing.
struct Grid {
    w: usize,
    h: usize,
    cell_m: f64,
    z: Vec<f32>,
}

impl Grid {
    fn at(&self, x: usize, y: usize) -> f32 {
        self.z[y * self.w + x]
    }

    /// The `k × k` block means (a block with any `NaN` is `NaN`).
    fn means(&self, k: usize) -> Grid {
        let (w, h) = (self.w / k, self.h / k);
        let mut z = vec![f32::NAN; w * h];
        for by in 0..h {
            for bx in 0..w {
                let mut s = 0.0f64;
                let mut ok = true;
                for y in by * k..(by + 1) * k {
                    for x in bx * k..(bx + 1) * k {
                        let v = self.at(x, y);
                        ok &= v.is_finite();
                        s += f64::from(v);
                    }
                }
                if ok {
                    z[by * w + bx] = (s / (k * k) as f64) as f32;
                }
            }
        }
        Grid {
            w,
            h,
            cell_m: self.cell_m * k as f64,
            z,
        }
    }
}

/// The local relief over `win × win` windows that do not overlap and hold land only.
fn reliefs(g: &Grid, win: usize) -> Vec<f32> {
    let mut out = Vec::new();
    for wy in 0..g.h / win {
        for wx in 0..g.w / win {
            let (mut lo, mut hi, mut land) = (f32::INFINITY, f32::NEG_INFINITY, true);
            for y in wy * win..(wy + 1) * win {
                for x in wx * win..(wx + 1) * win {
                    let v = g.at(x, y);
                    land &= v.is_finite() && v > 0.0;
                    lo = lo.min(v);
                    hi = hi.max(v);
                }
            }
            if land {
                out.push(hi - lo);
            }
        }
    }
    out.sort_by(f32::total_cmp);
    out
}

fn line(name: &str, scale: &str, r: &[f32]) -> String {
    if r.is_empty() {
        return format!("  {name:<34} {scale:>6}: no land window");
    }
    let p = |q: f64| r[((r.len() - 1) as f64 * q).round() as usize];
    let under = |m: f32| 100.0 * r.iter().filter(|v| **v < m).count() as f64 / r.len() as f64;
    format!(
        "  {name:<34} {scale:>6}: p10 {:>6.0}  p50 {:>6.0}  p90 {:>6.0} m | under 20 m {:>5.1} %, 50 m {:>5.1} %, 100 m {:>5.1} % | {} windows",
        p(0.1),
        p(0.5),
        p(0.9),
        under(20.0),
        under(50.0),
        under(100.0),
        r.len()
    )
}

fn report(name: &str, g: &Grid) {
    println!("{}", line(name, "1 km", &reliefs(g, WIN_1KM)));
    println!("{}", line(name, "5 km", &reliefs(g, WIN_5KM)));
    println!(
        "{}",
        line(name, "41 km", &reliefs(&g.means(MEAN_BLOCK), WIN_MEANS))
    );
}

/// ★ THE SIDE VIEW: from the cell `from`, 2 km over the ground, looking along `heading_deg` (0 up the grid's rows, 90 along
/// its columns).
fn render(g: &Grid, from: (usize, usize), heading_deg: f64, path: &str) {
    let (wpx, hpx) = (960u32, 480u32);
    let fov = 60f64.to_radians();
    let (ex, ey) = (from.0 as f64 * g.cell_m, from.1 as f64 * g.cell_m);
    let ground = |x: f64, y: f64| -> Option<f64> {
        let (cx, cy) = (x / g.cell_m, y / g.cell_m);
        if cx < 0.0 || cy < 0.0 || cx >= (g.w - 1) as f64 || cy >= (g.h - 1) as f64 {
            return None;
        }
        let (ix, iy) = (cx as usize, cy as usize);
        let (fx, fy) = (cx - ix as f64, cy - iy as f64);
        let v = |a: usize, b: usize| f64::from(g.at(a, b));
        let z = v(ix, iy) * (1.0 - fx) * (1.0 - fy)
            + v(ix + 1, iy) * fx * (1.0 - fy)
            + v(ix, iy + 1) * (1.0 - fx) * fy
            + v(ix + 1, iy + 1) * fx * fy;
        z.is_finite().then_some(z.max(0.0))
    };
    let eye_z = ground(ex, ey).unwrap_or(0.0) + 2_000.0;
    let pitch = -8f64.to_radians();
    let sun = {
        let (az, el) = (225f64.to_radians(), 25f64.to_radians());
        [el.cos() * az.sin(), -el.cos() * az.cos(), el.sin()]
    };
    let mut img = image::RgbImage::new(wpx, hpx);
    for py in 0..hpx {
        for px in 0..wpx {
            let ax = ((f64::from(px) + 0.5) / f64::from(wpx) - 0.5) * fov;
            let ay = (0.5 - (f64::from(py) + 0.5) / f64::from(hpx)) * fov * f64::from(hpx)
                / f64::from(wpx)
                + pitch;
            // Up the rows is −y (row 0 at the north); the heading turns the view about the vertical.
            let h = heading_deg.to_radians() + ax;
            let d = [h.sin() * ay.cos(), -h.cos() * ay.cos(), ay.sin()];
            let (mut t, mut hit) = (0.0f64, None);
            let step = g.cell_m * 0.5;
            while t < 60_000.0 {
                let (x, y, z) = (ex + d[0] * t, ey + d[1] * t, eye_z + d[2] * t);
                match ground(x, y) {
                    None => break,
                    Some(gz) if z <= gz => {
                        hit = Some((x, y, gz, t));
                        break;
                    }
                    _ => {}
                }
                t += step * (1.0 + t / 8_000.0);
            }
            let sky = [150.0, 180.0, 215.0];
            let c = match hit {
                None => sky,
                Some((x, y, gz, t)) => {
                    let dzx = ground(x + g.cell_m, y).unwrap_or(gz)
                        - ground(x - g.cell_m, y).unwrap_or(gz);
                    let dzy = ground(x, y + g.cell_m).unwrap_or(gz)
                        - ground(x, y - g.cell_m).unwrap_or(gz);
                    let n = [-dzx, -dzy, 2.0 * g.cell_m];
                    let l = (n[0] * n[0] + n[1] * n[1] + n[2] * n[2]).sqrt();
                    let lit = ((n[0] * sun[0] + n[1] * sun[1] + n[2] * sun[2]) / l).max(0.0);
                    let base = if gz <= 0.0 {
                        [40.0, 70.0, 120.0]
                    } else {
                        [170.0, 160.0, 140.0]
                    };
                    let shade = 0.25 + 0.75 * lit;
                    let haze = 1.0 - (-t / 45_000.0).exp();
                    [0, 1, 2].map(|k| base[k] * shade * (1.0 - haze) + sky[k] * haze)
                }
            };
            img.put_pixel(px, py, image::Rgb(c.map(|v| v.clamp(0.0, 255.0) as u8)));
        }
    }
    img.save(path).expect("the picture saves");
}

/// The home planet's finished fine tiles around the berth, laid side by side on the berth's face.
fn home_grid() -> Grid {
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
    let env = vd_bins::process_env();
    let store = vd_bins::open_realm_store(&env, realm)
        .expect("the store opens")
        .expect("VD_REALM_STORE names a file");
    let artifact = vd_bins::artifact_store::read_artifact(
        &store,
        vd_terrain::artifact_world_tag(UNIVERSE),
        &lattice,
    )
    .expect("the stored rows agree with their head")
    .expect("an artifact is stored");
    let radial = (BERTH[0] * BERTH[0] + BERTH[1] * BERTH[1] + BERTH[2] * BERTH[2]).sqrt();
    let dir = [BERTH[0] / radial, BERTH[1] / radial, BERTH[2] / radial];
    let drawn = artifact.drawn_range_m(&lattice);
    let named = vd_terrain::artifact::fine_needs_of_head(
        &body,
        (drawn.0, drawn.1, artifact.sea_m),
        dir,
        radial,
        vd_core::geometry::drawable_theta_min_rad(),
    );
    let face = vd_seed::bend::face_of(dir).index();
    let input = input_word(artifact.digest());
    let mut tiles = Vec::new();
    for &((f, tx, ty), _) in &named {
        if f != face {
            continue;
        }
        let at = FineAt {
            form: FineForm::Finished,
            input,
            level: LEVEL_L1,
            face: f,
            tx,
            ty,
        };
        if let Ok(Some(bytes)) = read_fine_record(&store, &at)
            && let Some((ts, levels)) = decode_levels(&bytes)
        {
            tiles.push((
                tx,
                ty,
                1usize << ts,
                levels.into_iter().next().expect("level 0"),
            ));
        }
    }
    let side = tiles.first().expect("a stored tile").2;
    let (x0, x1) = tiles
        .iter()
        .fold((u32::MAX, 0), |(a, b), t| (a.min(t.0), b.max(t.0)));
    let (y0, y1) = tiles
        .iter()
        .fold((u32::MAX, 0), |(a, b), t| (a.min(t.1), b.max(t.1)));
    let (w, h) = (
        ((x1 - x0 + 1) as usize) * side,
        ((y1 - y0 + 1) as usize) * side,
    );
    let mut z = vec![f32::NAN; w * h];
    for (tx, ty, s, q) in &tiles {
        let (ox, oy) = ((tx - x0) as usize * s, (ty - y0) as usize * s);
        for r in 0..*s {
            for c in 0..*s {
                z[(oy + r) * w + ox + c] = q[r * s + c] as f32 / 4.0;
            }
        }
    }
    println!(
        "the home planet: {} finished tiles of face {} around the berth ({}..={} × {}..={}), {w} × {h} cells of 128 m",
        tiles.len(),
        Face::from_index(face).map_or(0, Face::index),
        x0,
        x1,
        y0,
        y1
    );
    Grid {
        w,
        h,
        cell_m: 128.0,
        z,
    }
}

/// An SRTM 1″ tile averaged to ~128 m cells: 4 rows, and the columns that make 128 m at the tile's mid latitude.
fn earth_grid(path: &str) -> Grid {
    let mut raw = Vec::new();
    std::fs::File::open(path)
        .and_then(|mut f| f.read_to_end(&mut raw))
        .expect("the tile reads");
    let n = 3601usize;
    let name = std::path::Path::new(path)
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("");
    let lat: f64 = name[1..3].parse::<f64>().expect("a latitude")
        * if name.starts_with('S') { -1.0 } else { 1.0 };
    let row_m = 30.87;
    let col_m = row_m * (lat + 0.5).to_radians().cos();
    let (ky, kx) = (4usize, (128.0 / col_m).round().max(1.0) as usize);
    let (w, h) = ((n - 1) / kx, (n - 1) / ky);
    let mut z = vec![f32::NAN; w * h];
    for by in 0..h {
        for bx in 0..w {
            let (mut s, mut ok) = (0.0f64, true);
            for y in by * ky..(by + 1) * ky {
                for x in bx * kx..(bx + 1) * kx {
                    let v = i16::from_be_bytes([raw[2 * (y * n + x)], raw[2 * (y * n + x) + 1]]);
                    ok &= v != i16::MIN;
                    s += f64::from(v);
                }
            }
            if ok {
                z[by * w + bx] = (s / (kx * ky) as f64) as f32;
            }
        }
    }
    Grid {
        w,
        h,
        cell_m: (row_m * ky as f64 + col_m * kx as f64) / 2.0,
        z,
    }
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let out = args.first().expect("an output directory").clone();
    std::fs::create_dir_all(&out).expect("the output directory");
    println!(
        "THE LOCAL RELIEF (the window's highest less its lowest), land windows only, cells of ~128 m"
    );
    let home = home_grid();
    report("the home planet, fine ground", &home);
    // Each home tile alone: which are flat and which hold mountains.
    let side = 512usize;
    let mut per_tile = Vec::new();
    for ty in 0..home.h / side {
        for tx in 0..home.w / side {
            let mut z = vec![f32::NAN; side * side];
            for r in 0..side {
                for c in 0..side {
                    z[r * side + c] = home.at(tx * side + c, ty * side + r);
                }
            }
            let g = Grid {
                w: side,
                h: side,
                cell_m: 128.0,
                z,
            };
            let r5 = reliefs(&g, WIN_5KM);
            let r41 = reliefs(&g.means(MEAN_BLOCK), 16);
            if !r5.is_empty() {
                per_tile.push((
                    tx,
                    ty,
                    r5[r5.len() / 2],
                    r41.first().copied().unwrap_or(f32::NAN),
                ));
            }
        }
    }
    println!(
        "  the home tiles one by one: (column, row) the 5 km relief's median | the tile's whole relief over its 4 096 m means"
    );
    for row in per_tile.chunks(9) {
        println!(
            "    {}",
            row.iter()
                .map(|(x, y, a, b)| format!("({x},{y}) {a:>4.0}|{b:>5.0}"))
                .collect::<Vec<_>>()
                .join("  ")
        );
    }
    // The berth's tile (14, 13) looking along the columns, over the plains east of the belt; a plain, (11, 9); hill
    // country, (6, 16); each from its tile's middle, looking up the rows unless named.
    let mid = |tx: usize, ty: usize| (tx * side + side / 2, ty * side + side / 2);
    render(
        &home,
        mid(14, 13),
        90.0,
        &format!("{out}/home_berth_east.png"),
    );
    render(&home, mid(11, 9), 0.0, &format!("{out}/home_plain.png"));
    render(&home, mid(6, 16), 0.0, &format!("{out}/home_hills.png"));
    for path in &args[1..] {
        let g = earth_grid(path);
        let name = std::path::Path::new(path)
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("earth");
        report(&format!("Earth {name}"), &g);
        render(
            &g,
            (g.w / 2, g.h * 3 / 4),
            0.0,
            &format!("{out}/earth_{name}.png"),
        );
    }
    println!("pictures in {out} (each: 2 km over the ground, 8° down, 60° wide)");
}
