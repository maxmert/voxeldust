//! ★ THE ARTIFACT'S BYTES, CLASS BY CLASS, AGAINST A SAVED ONE (plan step 2b, W4; ruling B8 item 2).
//!
//! Step 2b changes what two row bytes MEAN — the receiver slot and the discharge class — and the
//! plan states that nothing the client draws moves with them. That is a claim about bytes, so this
//! instrument measures it and never argues it. It solves the home planet exactly as the shard does
//! (`solve_full`, the climate over the connected ocean, `Artifact::of`), and then either SAVES the
//! artifact and the solve's own heights to a file, or COMPARES them with a saved file, one count a
//! class:
//!
//! - the solve's heights in sixteenths (`MacroSolve::z`) and its lake levels (`water_z`);
//! - the rows' height, water, facies bits, receiver slot, discharge class, temperature, rain,
//!   aridity and province bytes;
//! - every pyramid level's heights and water words, the coast mask and the sea.
//!
//! A class the client draws (the heights, the water, the facies, the pyramid, the water words, the
//! mask, the sea) must count 0; the receiver slot and the discharge class may move.
//!
//! **Example.** Before step 2b the scratchpad holds the generator-20 artifact. After it, a pilot's
//! view of the belt's lakes must not move by one metre: the compare run prints 0 in every drawn
//! class and a count only beside the receiver slot and the discharge class.
//!
//! `cargo run --release -p vd-bins --example artifact_diff -- save <file>`
//! `cargo run --release -p vd-bins --example artifact_diff -- compare <file>`

use std::io::{Read, Write};

use vd_terrain::artifact::{Artifact, FACIES_SHIFT, RECEIVER_NONE, RECEIVER_SLOT_MASK, Row};
use vd_terrain::home::{home_planet, home_solve_words};
use vd_terrain::solve::{Schedule, solve_full};

/// The file's first eight bytes.
const MAGIC: &[u8; 8] = b"VDARTDF1";

/// What one solve leaves for the comparison: the artifact and the two solve rows under it.
struct Saved {
    artifact: Artifact,
    z: Vec<i32>,
    water_z: Vec<i32>,
}

fn solve() -> Saved {
    let body = home_planet();
    let words = home_solve_words();
    let lattice = body.macro_lattice().expect("the home planet has a lattice");
    let started = std::time::Instant::now();
    let (state, facies, _) = solve_full(&body, &words, Schedule::standard(words.age_yr))
        .expect("the home planet solves");
    let climate =
        vd_terrain::climate::climate(&body, &lattice, &words, &state.z, Some(state.sea_view()));
    let artifact = Artifact::of(&state, &facies, &climate, words.water_km3 > 0);
    println!(
        "artifact_diff: solved in {:.1} s; generator {} artifact {}; digest {:?}",
        started.elapsed().as_secs_f64(),
        vd_terrain::GENERATOR_VERSION,
        artifact.version,
        artifact.digest()
    );
    Saved {
        artifact,
        z: state.z,
        water_z: state.water_z,
    }
}

fn put_u32(out: &mut Vec<u8>, v: u32) {
    out.extend_from_slice(&v.to_le_bytes());
}

fn put_i16s(out: &mut Vec<u8>, v: &[i16]) {
    put_u32(out, v.len() as u32);
    for x in v {
        out.extend_from_slice(&x.to_le_bytes());
    }
}

fn put_i32s(out: &mut Vec<u8>, v: &[i32]) {
    put_u32(out, v.len() as u32);
    for x in v {
        out.extend_from_slice(&x.to_le_bytes());
    }
}

fn encode(s: &Saved) -> Vec<u8> {
    let a = &s.artifact;
    let mut out = MAGIC.to_vec();
    put_u32(&mut out, a.edge);
    put_u32(&mut out, a.version);
    out.extend_from_slice(&a.sea_m.to_le_bytes());
    put_u32(&mut out, a.rows.len() as u32);
    for r in &a.rows {
        out.extend_from_slice(&r.to_bytes());
    }
    put_u32(&mut out, a.pyramid.len() as u32);
    for level in &a.pyramid {
        put_i16s(&mut out, level);
    }
    put_u32(&mut out, a.pyramid_water.len() as u32);
    for level in &a.pyramid_water {
        put_i16s(&mut out, level);
    }
    put_u32(&mut out, a.coast.len() as u32);
    out.extend_from_slice(&a.coast);
    put_i32s(&mut out, &s.z);
    put_i32s(&mut out, &s.water_z);
    out
}

/// A reader over the saved bytes.
struct Cursor<'a> {
    b: &'a [u8],
    at: usize,
}

impl Cursor<'_> {
    fn take(&mut self, n: usize) -> &[u8] {
        let s = &self.b[self.at..self.at + n];
        self.at += n;
        s
    }
    fn u32(&mut self) -> u32 {
        u32::from_le_bytes(self.take(4).try_into().expect("four bytes"))
    }
    fn i16(&mut self) -> i16 {
        i16::from_le_bytes(self.take(2).try_into().expect("two bytes"))
    }
    fn i16s(&mut self) -> Vec<i16> {
        let n = self.u32() as usize;
        (0..n).map(|_| self.i16()).collect()
    }
    fn i32s(&mut self) -> Vec<i32> {
        let n = self.u32() as usize;
        (0..n)
            .map(|_| i32::from_le_bytes(self.take(4).try_into().expect("four bytes")))
            .collect()
    }
}

fn decode(b: &[u8]) -> Saved {
    assert_eq!(&b[..8], MAGIC, "not an artifact_diff file");
    let mut c = Cursor { b, at: 8 };
    let edge = c.u32();
    let version = c.u32();
    let sea_m = c.i16();
    let n = c.u32() as usize;
    let rows = (0..n)
        .map(|_| Row::from_bytes(c.take(10).try_into().expect("ten bytes")))
        .collect();
    let levels = c.u32() as usize;
    let pyramid = (0..levels).map(|_| c.i16s()).collect();
    let wlevels = c.u32() as usize;
    let pyramid_water = (0..wlevels).map(|_| c.i16s()).collect();
    let coast_len = c.u32() as usize;
    let coast = c.take(coast_len).to_vec();
    let z = c.i32s();
    let water_z = c.i32s();
    Saved {
        artifact: Artifact {
            edge,
            version,
            sea_m,
            rows,
            pyramid,
            pyramid_water,
            coast,
        },
        z,
        water_z,
    }
}

/// The nodes where two rows of one class differ.
fn count<T: PartialEq>(a: &[T], b: &[T]) -> usize {
    if a.len() != b.len() {
        return usize::MAX;
    }
    a.iter().zip(b).filter(|(x, y)| x != y).count()
}

fn compare(old: &Saved, new: &Saved) {
    let (a, b) = (&old.artifact, &new.artifact);
    println!(
        "artifact_diff: the saved artifact is version {}, edge {}, sea {} m; this one version {}, edge {}, sea {} m",
        a.version, a.edge, a.sea_m, b.version, b.edge, b.sea_m
    );
    let field = |f: &dyn Fn(&Row) -> i64| -> usize {
        count(
            &a.rows.iter().map(f).collect::<Vec<_>>(),
            &b.rows.iter().map(f).collect::<Vec<_>>(),
        )
    };
    let drawn: Vec<(&str, usize)> = vec![
        ("solve-z-sixteenths", count(&old.z, &new.z)),
        ("solve-water-z", count(&old.water_z, &new.water_z)),
        ("row-z_m", field(&|r| i64::from(r.z_m))),
        ("row-water_m", field(&|r| i64::from(r.water_m))),
        (
            "row-facies",
            field(&|r| i64::from(r.receiver_facies >> FACIES_SHIFT)),
        ),
        ("row-temperature", field(&|r| i64::from(r.temperature))),
        ("row-rain", field(&|r| i64::from(r.rain))),
        ("row-aridity", field(&|r| i64::from(r.aridity))),
        ("row-province", field(&|r| i64::from(r.province))),
        (
            "pyramid-z",
            a.pyramid
                .iter()
                .zip(&b.pyramid)
                .map(|(x, y)| count(x, y))
                .sum::<usize>()
                + usize::from(a.pyramid.len() != b.pyramid.len()),
        ),
        (
            "pyramid-water",
            a.pyramid_water
                .iter()
                .zip(&b.pyramid_water)
                .map(|(x, y)| count(x, y))
                .sum::<usize>()
                + usize::from(a.pyramid_water.len() != b.pyramid_water.len()),
        ),
        ("coast-mask-bytes", count(&a.coast, &b.coast)),
        ("sea_m", usize::from(a.sea_m != b.sea_m)),
    ];
    let meaning: Vec<(&str, usize)> = vec![
        (
            "row-receiver-slot",
            field(&|r| i64::from(r.receiver_facies & (RECEIVER_SLOT_MASK | RECEIVER_NONE))),
        ),
        ("row-discharge_log", field(&|r| i64::from(r.discharge_log))),
    ];
    let mut moved = 0usize;
    for (name, k) in &drawn {
        println!("  DRAWN   {name:<20} {k} differ");
        moved += *k;
        println!(
            "CENSUS ARTIFACT-DIFF-{} {}differ 0[plan-step-2b:nothing-drawn-moves] {}",
            name.to_uppercase(),
            k,
            if *k == 0 { "GREEN" } else { "RED" }
        );
    }
    for (name, k) in &meaning {
        println!("  MEANING {name:<20} {k} differ (step 2b changes this byte's meaning)");
        println!(
            "CENSUS ARTIFACT-DIFF-{} {}differ no-band[plan-step-2b:the-byte's-meaning-changes] NO-BAND",
            name.to_uppercase(),
            k
        );
    }
    println!(
        "artifact_diff: {moved} drawn values differ in all; the saved digest {:?}, this digest {:?}",
        a.digest(),
        b.digest()
    );
}

/// ★ HOW DEEP THE LAKES ARE (2026-09-25; the generator-21 flight 1 000 km over the largest lake drew
/// stripes and dots, not one water surface — the look of a water level that stands almost exactly on a
/// wide flat ground). Over every lake node (a wet row that is not the ocean's facies): the depth, the
/// row's water metre less its ground metre, as a distribution; then each lake patch (lake nodes joined
/// through the lattice's own neighbours, one level a patch) with its size and its depths, the largest
/// first. A reading, no band: it says whether the lakes are basins or sheets laid on a flat.
///
/// **Example.** A tarn in a pass reads a median depth of forty metres; a sheet the brim laid on a
/// plain reads zero metres over most of its nodes, and from orbit its level and its ground fight for
/// every pixel.
fn depths(tag: &str, a: &Artifact) {
    use vd_terrain::artifact::DRY_M;
    use vd_terrain::macro_lattice::NO_NODE;
    use vd_terrain::solve::FACIES_SEA;
    let lattice = home_planet()
        .macro_lattice()
        .expect("the home planet has a lattice");
    let lake = |i: usize| {
        let r = a.rows[i];
        r.water_m != DRY_M && (r.receiver_facies >> FACIES_SHIFT) & FACIES_SEA == 0
    };
    let depth = |i: usize| i32::from(a.rows[i].water_m) - i32::from(a.rows[i].z_m);
    let km2 = |i: usize| lattice.area_m2(i as u32) as f64 / 1.0e6;
    let n = a.rows.len();
    let mut all: Vec<(i32, f64)> = (0..n)
        .filter(|&i| lake(i))
        .map(|i| (depth(i), km2(i)))
        .collect();
    all.sort_by_key(|x| x.0);
    let area: f64 = all.iter().map(|x| x.1).sum();
    let share =
        |lim: i32| all.iter().filter(|x| x.0 < lim).map(|x| x.1).sum::<f64>() / area.max(1.0);
    let mut acc = 0.0;
    let median = all
        .iter()
        .find(|x| {
            acc += x.1;
            acc >= area / 2.0
        })
        .map_or(0, |x| x.0);
    println!(
        "depths {tag}: {} lake nodes, {:.4e} km²; by area {:.1} % stand under 1 m of water, {:.1} % under 5 m, {:.1} % under 10 m, {:.1} % under 50 m; the area median depth {median} m; the deepest {} m",
        all.len(),
        area,
        100.0 * share(1),
        100.0 * share(5),
        100.0 * share(10),
        100.0 * share(50),
        all.last().map_or(0, |x| x.0)
    );
    // The patches: lake nodes of one level joined through the lattice's neighbours.
    let mut seen = vec![false; n];
    let mut patches: Vec<(f64, usize, i16, Vec<i32>, u32)> = Vec::new();
    for s in 0..n {
        if seen[s] || !lake(s) {
            continue;
        }
        let level = a.rows[s].water_m;
        let mut stack = vec![s as u32];
        seen[s] = true;
        let (mut km, mut ds, mut first) = (0.0, Vec::new(), s as u32);
        while let Some(v) = stack.pop() {
            let vi = v as usize;
            km += km2(vi);
            ds.push(depth(vi));
            first = first.min(v);
            for m in lattice.neighbours(v) {
                if m != NO_NODE
                    && !seen[m as usize]
                    && lake(m as usize)
                    && a.rows[m as usize].water_m == level
                {
                    seen[m as usize] = true;
                    stack.push(m);
                }
            }
        }
        ds.sort_unstable();
        let count = ds.len();
        patches.push((km, count, level, ds, first));
    }
    patches.sort_by(|x, y| y.0.total_cmp(&x.0));
    let thin: f64 = patches
        .iter()
        .filter(|p| p.3[p.3.len() / 2] < 10)
        .map(|p| p.0)
        .sum();
    println!(
        "depths {tag}: {} patches; {:.1} % of the lake area lies in patches whose median depth is under 10 m",
        patches.len(),
        100.0 * thin / area.max(1.0)
    );
    for (km, count, level, ds, first) in patches.iter().take(10) {
        let under = |lim: i32| ds.iter().filter(|&&d| d < lim).count() as f64 / *count as f64;
        let g = lattice.direction(*first);
        let unit = (1u64 << vd_recipe::bend::DIR_BITS) as f64;
        let d = [
            g[0].raw() as f64 / unit,
            g[1].raw() as f64 / unit,
            g[2].raw() as f64 / unit,
        ];
        let len = (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt();
        let d = [d[0] / len, d[1] / len, d[2] / len];
        println!(
            "  {count:>6} nodes {km:>10.0} km², level {level} m (the sea {} m): depth median {} m, p90 {} m, max {} m; {:.1} % of its nodes under 1 m, {:.1} % under 10 m; a node at {:.4} {:.4} {:.4}",
            a.sea_m,
            ds[count / 2],
            ds[count * 9 / 10],
            ds[count - 1],
            100.0 * under(1),
            100.0 * under(10),
            d[0],
            d[1],
            d[2]
        );
    }
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let (mode, path) = match args.as_slice() {
        [m, p] => (m.as_str(), p.as_str()),
        _ => panic!("usage: artifact_diff save|compare|depth <file>"),
    };
    if mode == "depth" {
        let mut bytes = Vec::new();
        std::fs::File::open(path)
            .and_then(|mut f| f.read_to_end(&mut bytes))
            .expect("the saved file is read");
        let saved = decode(&bytes);
        depths(
            &format!("saved artifact {}", saved.artifact.version),
            &saved.artifact,
        );
        let now = solve();
        depths(
            &format!("this artifact {}", now.artifact.version),
            &now.artifact,
        );
        return;
    }
    let now = solve();
    match mode {
        "save" => {
            let bytes = encode(&now);
            std::fs::File::create(path)
                .and_then(|mut f| f.write_all(&bytes))
                .expect("the file is written");
            println!("artifact_diff: saved {} bytes to {path}", bytes.len());
        }
        "compare" => {
            let mut bytes = Vec::new();
            std::fs::File::open(path)
                .and_then(|mut f| f.read_to_end(&mut bytes))
                .expect("the saved file is read");
            compare(&decode(&bytes), &now);
        }
        _ => panic!("usage: artifact_diff save|compare <file>"),
    }
}
