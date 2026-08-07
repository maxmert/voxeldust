//! Follow-up: the cave/density field dominates terrain generation cost (3.87 ms of 4.16 ms per
//! surface chunk). This measures the levers that move it, so the design picks on data.
//!
//! Lever 1 — octave count on the 3D field.
//! Lever 2 — COARSE-LATTICE EVALUATION: evaluate the 3D field on a grid N times coarser and
//!           trilinearly interpolate between samples. Caves are low-frequency features; sampling
//!           them per-voxel is oversampling. This is what shipped voxel engines do.
//!           Interpolation is exactly representable (powers of two) so it stays bit-deterministic.
//! Lever 3 — how much of a chunk actually needs the 3D field at all.

use std::hint::black_box;
use std::time::Instant;

#[inline(always)]
fn mix64(mut z: u64) -> u64 {
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

#[inline(always)]
fn hash3(x: i64, y: i64, z: i64, seed: u64) -> u64 {
    let mut h = seed ^ 0x9E37_79B9_7F4A_7C15;
    h = mix64(h ^ (x as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15));
    h = mix64(h ^ (y as u64).wrapping_mul(0xC2B2_AE3D_27D4_EB4F));
    mix64(h ^ (z as u64).wrapping_mul(0x1656_67B1_9E37_79F9))
}

const GRAD: [[f64; 3]; 16] = [
    [1.0, 1.0, 0.0], [-1.0, 1.0, 0.0], [1.0, -1.0, 0.0], [-1.0, -1.0, 0.0],
    [1.0, 0.0, 1.0], [-1.0, 0.0, 1.0], [1.0, 0.0, -1.0], [-1.0, 0.0, -1.0],
    [0.0, 1.0, 1.0], [0.0, -1.0, 1.0], [0.0, 1.0, -1.0], [0.0, -1.0, -1.0],
    [1.0, 1.0, 0.0], [0.0, -1.0, 1.0], [-1.0, 1.0, 0.0], [0.0, -1.0, -1.0],
];

#[inline(always)]
fn dot_grad(h: u64, dx: f64, dy: f64, dz: f64) -> f64 {
    let g = GRAD[(h & 15) as usize];
    g[0] * dx + g[1] * dy + g[2] * dz
}

#[inline(always)]
fn fade(t: f64) -> f64 {
    t * t * t * (t * (t * 6.0 - 15.0) + 10.0)
}

#[inline(always)]
fn lerp(a: f64, b: f64, t: f64) -> f64 {
    a + t * (b - a)
}

#[inline(always)]
fn noise3(x: f64, y: f64, z: f64, seed: u64) -> f64 {
    let xi = x.floor();
    let yi = y.floor();
    let zi = z.floor();
    let (x0, y0, z0) = (xi as i64, yi as i64, zi as i64);
    let (fx, fy, fz) = (x - xi, y - yi, z - zi);
    let (u, v, w) = (fade(fx), fade(fy), fade(fz));
    let n000 = dot_grad(hash3(x0, y0, z0, seed), fx, fy, fz);
    let n100 = dot_grad(hash3(x0 + 1, y0, z0, seed), fx - 1.0, fy, fz);
    let n010 = dot_grad(hash3(x0, y0 + 1, z0, seed), fx, fy - 1.0, fz);
    let n110 = dot_grad(hash3(x0 + 1, y0 + 1, z0, seed), fx - 1.0, fy - 1.0, fz);
    let n001 = dot_grad(hash3(x0, y0, z0 + 1, seed), fx, fy, fz - 1.0);
    let n101 = dot_grad(hash3(x0 + 1, y0, z0 + 1, seed), fx - 1.0, fy, fz - 1.0);
    let n011 = dot_grad(hash3(x0, y0 + 1, z0 + 1, seed), fx, fy - 1.0, fz - 1.0);
    let n111 = dot_grad(hash3(x0 + 1, y0 + 1, z0 + 1, seed), fx - 1.0, fy - 1.0, fz - 1.0);
    let x00 = lerp(n000, n100, u);
    let x10 = lerp(n010, n110, u);
    let x01 = lerp(n001, n101, u);
    let x11 = lerp(n011, n111, u);
    lerp(lerp(x00, x10, v), lerp(x01, x11, v), w)
}

#[inline(always)]
fn fbm(x: f64, y: f64, z: f64, seed: u64, octaves: u32) -> f64 {
    let mut freq = 1.0_f64;
    let mut amp = 1.0_f64;
    let mut sum = 0.0_f64;
    for o in 0..octaves {
        sum += amp * noise3(x * freq, y * freq, z * freq, seed ^ (o as u64));
        freq *= 2.0;
        amp *= 0.5;
    }
    sum
}

const CHUNK: usize = 62;
const CHUNK_VOL: usize = CHUNK * CHUNK * CHUNK;

/// Per-voxel 3D field over `frac` of a chunk's volume, `oct` octaves.
fn per_voxel(frac: f64, oct: u32, seed: u64) -> f64 {
    let n = (CHUNK_VOL as f64 * frac) as usize;
    let t0 = Instant::now();
    let mut carved = 0u32;
    for i in 0..n {
        let x = (i % CHUNK) as f64;
        let y = ((i / CHUNK) % CHUNK) as f64;
        let z = (i / (CHUNK * CHUNK)) as f64;
        carved += u32::from(fbm(x * 0.05, y * 0.05, z * 0.05, seed, oct) > 0.6);
    }
    let ms = t0.elapsed().as_secs_f64() * 1e3;
    black_box(carved);
    ms
}

/// COARSE LATTICE: sample the 3D field every `step` voxels, then trilinearly interpolate.
/// `step` is a power of two so the interpolation weights are exactly representable.
fn coarse_lattice(step: usize, oct: u32, seed: u64) -> (f64, usize) {
    let s = CHUNK / step + 2; // +2 for the interpolation apron
    let t0 = Instant::now();

    // Sample pass.
    let mut grid = vec![0.0_f64; s * s * s];
    for i in 0..s {
        for j in 0..s {
            for k in 0..s {
                grid[(i * s + j) * s + k] = fbm(
                    (i * step) as f64 * 0.05,
                    (j * step) as f64 * 0.05,
                    (k * step) as f64 * 0.05,
                    seed,
                    oct,
                );
            }
        }
    }

    // Interpolate pass — one trilinear lerp per voxel.
    let inv = 1.0 / step as f64; // step is a power of two => exact
    let mut carved = 0u32;
    for i in 0..CHUNK {
        for j in 0..CHUNK {
            for k in 0..CHUNK {
                let (gi, gj, gk) = (i / step, j / step, k / step);
                let (fi, fj, fk) = (
                    (i % step) as f64 * inv,
                    (j % step) as f64 * inv,
                    (k % step) as f64 * inv,
                );
                let idx = |a: usize, b: usize, c: usize| grid[(a * s + b) * s + c];
                let c00 = lerp(idx(gi, gj, gk), idx(gi + 1, gj, gk), fi);
                let c10 = lerp(idx(gi, gj + 1, gk), idx(gi + 1, gj + 1, gk), fi);
                let c01 = lerp(idx(gi, gj, gk + 1), idx(gi + 1, gj, gk + 1), fi);
                let c11 = lerp(idx(gi, gj + 1, gk + 1), idx(gi + 1, gj + 1, gk + 1), fi);
                let d = lerp(lerp(c00, c10, fj), lerp(c01, c11, fj), fk);
                carved += u32::from(d > 0.6);
            }
        }
    }
    let ms = t0.elapsed().as_secs_f64() * 1e3;
    black_box(carved);
    (ms, s * s * s)
}

fn main() {
    let seed = 0xDEAD_BEEF_CAFE_F00D_u64;
    println!("=== Cave / 3D-density field: the levers ===");
    println!("machine: Apple M4 Pro · rustc 1.94.1 · release+lto · single core");
    println!("baseline to beat: 3.87 ms per surface chunk (per-voxel, 1/3 volume, 4 octaves)\n");

    println!("LEVER 1 — octave count, per-voxel over 1/3 of the chunk");
    for oct in [1u32, 2, 3, 4, 5] {
        println!("  {oct} octave(s)                                       {:>8.3} ms", per_voxel(1.0 / 3.0, oct, seed));
    }

    println!("\nLEVER 2 — coarse-lattice sampling + trilinear interpolation (4 octaves, FULL chunk)");
    println!("  {:<12} {:>10} {:>12} {:>10}", "step", "samples", "ms", "vs 1/3 base");
    let base = 3.871_f64;
    for step in [1usize, 2, 4, 8] {
        let (ms, samples) = coarse_lattice(step, 4, seed);
        println!("  every {step:<6} voxel {samples:>10} {ms:>12.3} {:>9.1}x", base / ms);
    }

    println!("\nLEVER 3 — how much of the chunk needs the 3D field at all");
    for frac in [1.0_f64, 0.5, 1.0 / 3.0, 0.1, 0.05] {
        println!("  {:>5.1}% of volume, per-voxel, 4 oct               {:>8.3} ms", frac * 100.0, per_voxel(frac, 4, seed));
    }

    // The recommended combination, measured end to end.
    println!("\nRECOMMENDED COMBINATION — coarse lattice step 4, 4 octaves, full chunk");
    let (ms, samples) = coarse_lattice(4, 4, seed);
    let height_ms = 0.252_f64;
    let fill_ms = 0.037_f64;
    let total = height_ms + fill_ms + ms;
    println!("  height field (6 oct, 3844 columns)                   {height_ms:>8.3} ms");
    println!("  fill by comparison                                   {fill_ms:>8.3} ms");
    println!("  cave field ({samples} samples + trilinear)            {ms:>8.3} ms");
    println!("  ---------------------------------------------------------------");
    println!("  ONE SURFACE CHUNK, TOTAL                             {total:>8.3} ms");
    println!("  (was 4.160 ms per-voxel; naive 3D everywhere was 10.15 ms)");

    let radius_m = 161_671.0_f64;
    let horizon_m = (2.0 * radius_m * 1.7).sqrt();
    let columns = std::f64::consts::PI * horizon_m * horizon_m / (CHUNK * CHUNK) as f64;
    let chunks = columns * 3.0;
    println!("\n  horizon fill ({chunks:.0} chunks), single core        {:>8.2} s", chunks * total / 1e3);
    println!("  horizon fill, 8 cores                                {:>8.2} s", chunks * total / 1e3 / 8.0);
    println!("\n  SUSTAINED FLIGHT at this cost:");
    for &v in &[50.0_f64, 100.0, 250.0, 500.0, 1000.0] {
        let cps = 2.0 * horizon_m * v / (CHUNK * CHUNK) as f64 * 3.0;
        let ms_per_s = cps * total;
        println!("    {v:>6.0} m/s -> {cps:>6.0} chunks/s -> {:>5.1}% of one core, {:>5.1}% of eight", ms_per_s / 10.0, ms_per_s / 80.0);
    }
}
