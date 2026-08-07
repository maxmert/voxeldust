//! Terrain-noise throughput benchmark for the Voxeldust block-system design.
//!
//! Answers the ONE unmeasured constant named in `scripts/block_system_design.md` §8.7 item 2:
//! how much does it cost to generate terrain, and is client-side generation ahead of a moving
//! player affordable at the committed 62^3 chunk size?
//!
//! Determinism discipline mirrors the design's generation rules exactly, so the measured cost is
//! the cost of the ACTUAL shippable path, not an optimistic stand-in:
//!   - integer lattice hashing via the repo's landed SplitMix64 avalanche (no second hash),
//!   - scalar f64 only (no glam, no Vec3A, no runtime-SIMD dispatch),
//!   - NO transcendentals anywhere (no sin/cos/powf/exp) — quintic fade is a polynomial,
//!   - gradients are a const table indexed by hashed bits (no float noise library).

use std::hint::black_box;
use std::time::Instant;

// ---------------------------------------------------------------------------------------------
// Integer lattice hash — the repo's SplitMix64 avalanche (crates/core/src/rng.rs), inlined.
// ---------------------------------------------------------------------------------------------

#[inline(always)]
fn mix64(mut z: u64) -> u64 {
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

#[inline(always)]
fn hash3(x: i64, y: i64, z: i64, seed: u64) -> u64 {
    // Fold the three lattice coordinates and the seed through the avalanche.
    let mut h = seed ^ 0x9E37_79B9_7F4A_7C15;
    h = mix64(h ^ (x as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15));
    h = mix64(h ^ (y as u64).wrapping_mul(0xC2B2_AE3D_27D4_EB4F));
    mix64(h ^ (z as u64).wrapping_mul(0x1656_67B1_9E37_79F9))
}

// 12 edge-midpoint gradients (the standard improved-Perlin set), exact f64 constants.
const GRAD: [[f64; 3]; 16] = [
    [1.0, 1.0, 0.0], [-1.0, 1.0, 0.0], [1.0, -1.0, 0.0], [-1.0, -1.0, 0.0],
    [1.0, 0.0, 1.0], [-1.0, 0.0, 1.0], [1.0, 0.0, -1.0], [-1.0, 0.0, -1.0],
    [0.0, 1.0, 1.0], [0.0, -1.0, 1.0], [0.0, 1.0, -1.0], [0.0, -1.0, -1.0],
    // Repeat 4 to make the table a power of two (Perlin's own padding trick), so the index
    // is a mask and not a modulo.
    [1.0, 1.0, 0.0], [0.0, -1.0, 1.0], [-1.0, 1.0, 0.0], [0.0, -1.0, -1.0],
];

#[inline(always)]
fn dot_grad(h: u64, dx: f64, dy: f64, dz: f64) -> f64 {
    let g = GRAD[(h & 15) as usize];
    g[0] * dx + g[1] * dy + g[2] * dz
}

/// Quintic fade 6t^5 - 15t^4 + 10t^3 — a polynomial, so it is bit-identical everywhere.
#[inline(always)]
fn fade(t: f64) -> f64 {
    t * t * t * (t * (t * 6.0 - 15.0) + 10.0)
}

#[inline(always)]
fn lerp(a: f64, b: f64, t: f64) -> f64 {
    a + t * (b - a)
}

/// One 3D gradient-noise evaluation. Scalar f64, integer lattice, no transcendentals.
#[inline(always)]
fn noise3(x: f64, y: f64, z: f64, seed: u64) -> f64 {
    let xi = x.floor();
    let yi = y.floor();
    let zi = z.floor();
    let (x0, y0, z0) = (xi as i64, yi as i64, zi as i64);
    let (fx, fy, fz) = (x - xi, y - yi, z - zi);
    let (u, v, w) = (fade(fx), fade(fy), fade(fz));

    let h000 = hash3(x0, y0, z0, seed);
    let h100 = hash3(x0 + 1, y0, z0, seed);
    let h010 = hash3(x0, y0 + 1, z0, seed);
    let h110 = hash3(x0 + 1, y0 + 1, z0, seed);
    let h001 = hash3(x0, y0, z0 + 1, seed);
    let h101 = hash3(x0 + 1, y0, z0 + 1, seed);
    let h011 = hash3(x0, y0 + 1, z0 + 1, seed);
    let h111 = hash3(x0 + 1, y0 + 1, z0 + 1, seed);

    let n000 = dot_grad(h000, fx, fy, fz);
    let n100 = dot_grad(h100, fx - 1.0, fy, fz);
    let n010 = dot_grad(h010, fx, fy - 1.0, fz);
    let n110 = dot_grad(h110, fx - 1.0, fy - 1.0, fz);
    let n001 = dot_grad(h001, fx, fy, fz - 1.0);
    let n101 = dot_grad(h101, fx - 1.0, fy, fz - 1.0);
    let n011 = dot_grad(h011, fx, fy - 1.0, fz - 1.0);
    let n111 = dot_grad(h111, fx - 1.0, fy - 1.0, fz - 1.0);

    let x00 = lerp(n000, n100, u);
    let x10 = lerp(n010, n110, u);
    let x01 = lerp(n001, n101, u);
    let x11 = lerp(n011, n111, u);
    lerp(lerp(x00, x10, v), lerp(x01, x11, v), w)
}

/// Fractal Brownian motion: `octaves` evaluations, lacunarity 2, gain 0.5.
/// Powers of two only, so every scale factor is exactly representable.
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

// ---------------------------------------------------------------------------------------------
// Scenario constants — the committed scale from docs/design/roadmap.json + the design document.
// ---------------------------------------------------------------------------------------------

const CHUNK: usize = 62;
const CHUNK_VOL: usize = CHUNK * CHUNK * CHUNK; // 238,328
const CHUNK_AREA: usize = CHUNK * CHUNK; //       3,844

/// Octave counts: the design's "biome/geology noise stack". Terrain height wants more octaves
/// (it is 2D-ish and carries the silhouette); the 3D cave/density field wants fewer.
const HEIGHT_OCTAVES: u32 = 6;
const DENSITY_OCTAVES: u32 = 4;

fn bench<F: FnMut() -> f64>(label: &str, iters: u64, mut f: F) -> f64 {
    // Warm up so we measure steady state, not the first-touch page faults.
    let mut acc = 0.0;
    for _ in 0..(iters / 10).max(1000) {
        acc += f();
    }
    black_box(acc);

    let t0 = Instant::now();
    let mut sink = 0.0_f64;
    for _ in 0..iters {
        sink += f();
    }
    let dt = t0.elapsed();
    black_box(sink);

    let ns = dt.as_secs_f64() * 1e9 / iters as f64;
    println!("  {label:<52} {ns:>9.2} ns/op");
    ns
}

fn main() {
    println!("=== Voxeldust terrain-noise throughput ===");
    println!("machine: Apple M4 Pro · rustc 1.94.1 · opt-level=3, lto=fat, codegen-units=1");
    println!("scalar f64, integer SplitMix64 lattice hash, NO transcendentals, single core\n");

    let seed = 0xDEAD_BEEF_CAFE_F00D_u64;

    // ---- Primitive costs -------------------------------------------------------------------
    println!("PRIMITIVES");
    let mut c = 0.0_f64;
    let ns_hash = bench("integer lattice hash (1 corner)", 50_000_000, || {
        c += 1.0;
        hash3(black_box(c as i64), 7, 13, seed) as f64
    });
    let mut p = 0.0_f64;
    let ns_eval = bench("one 3D gradient-noise evaluation", 20_000_000, || {
        p += 0.001;
        noise3(black_box(p), p * 1.7, p * 0.3, seed)
    });
    let mut q = 0.0_f64;
    let ns_fbm4 = bench("fBm, 4 octaves (the 3D density field)", 5_000_000, || {
        q += 0.001;
        fbm(black_box(q), q * 1.7, q * 0.3, seed, DENSITY_OCTAVES)
    });
    let mut r = 0.0_f64;
    let ns_fbm6 = bench("fBm, 6 octaves (the surface height field)", 4_000_000, || {
        r += 0.001;
        fbm(black_box(r), r * 1.7, r * 0.3, seed, HEIGHT_OCTAVES)
    });

    // ---- Strategy A: 3D density per voxel --------------------------------------------------
    println!("\nSTRATEGY A — 3D density evaluated at EVERY voxel (the naive path)");
    let t0 = Instant::now();
    let mut solid = 0u32;
    for i in 0..CHUNK {
        for j in 0..CHUNK {
            for k in 0..CHUNK {
                let d = fbm(
                    i as f64 * 0.03,
                    j as f64 * 0.03,
                    k as f64 * 0.03,
                    seed,
                    DENSITY_OCTAVES,
                );
                solid += u32::from(d > 0.0);
            }
        }
    }
    let a_chunk_ms = t0.elapsed().as_secs_f64() * 1e3;
    black_box(solid);
    println!("  one 62^3 chunk ({CHUNK_VOL} voxels, {DENSITY_OCTAVES} octaves)   {a_chunk_ms:>9.2} ms");

    // ---- Strategy B: height-first, then fill by comparison ---------------------------------
    println!("\nSTRATEGY B — height field per COLUMN, fill by comparison (the design's path)");
    let t0 = Instant::now();
    let mut heights = vec![0.0_f64; CHUNK_AREA];
    for i in 0..CHUNK {
        for j in 0..CHUNK {
            heights[i * CHUNK + j] = fbm(
                i as f64 * 0.01,
                j as f64 * 0.01,
                0.5,
                seed,
                HEIGHT_OCTAVES,
            );
        }
    }
    let b_height_ms = t0.elapsed().as_secs_f64() * 1e3;

    let t0 = Instant::now();
    let mut solid = 0u32;
    for i in 0..CHUNK {
        for j in 0..CHUNK {
            let h = heights[i * CHUNK + j] * 20.0 + 31.0;
            for k in 0..CHUNK {
                solid += u32::from((k as f64) < h);
            }
        }
    }
    let b_fill_ms = t0.elapsed().as_secs_f64() * 1e3;
    black_box(solid);

    // Caves: 3D noise ONLY in the band the surface actually passes through. Measure the
    // realistic case where roughly a third of a surface chunk's voxels are near the surface.
    let cave_voxels = CHUNK_VOL / 3;
    let t0 = Instant::now();
    let mut carved = 0u32;
    for n in 0..cave_voxels {
        let i = (n % CHUNK) as f64;
        let j = ((n / CHUNK) % CHUNK) as f64;
        let k = (n / CHUNK_AREA) as f64;
        let d = fbm(i * 0.05, j * 0.05, k * 0.05, seed ^ 0x5EED, DENSITY_OCTAVES);
        carved += u32::from(d > 0.6);
    }
    let b_cave_ms = t0.elapsed().as_secs_f64() * 1e3;
    black_box(carved);

    let b_total_ms = b_height_ms + b_fill_ms + b_cave_ms;
    println!("  height field ({CHUNK_AREA} columns, {HEIGHT_OCTAVES} octaves)         {b_height_ms:>9.3} ms");
    println!("  fill by comparison ({CHUNK_VOL} voxels)              {b_fill_ms:>9.3} ms");
    println!("  cave carve (3D, surface band only, 1/3 volume)       {b_cave_ms:>9.3} ms");
    println!("  ---------------------------------------------------------------");
    println!("  one SURFACE chunk, total                             {b_total_ms:>9.3} ms");
    println!("  one INTERIOR/AIR chunk (band test says uniform)      {:>9.3} ms", b_height_ms);
    println!("  speed-up of B over A                                 {:>9.1}x", a_chunk_ms / b_total_ms);

    // ---- Horizon extrapolation --------------------------------------------------------------
    // R = 161,671 m (design row 4 recommendation). Ground horizon at eye height 1.7 m:
    //   d = sqrt(2*R*h) = sqrt(2 * 161671 * 1.7) ~= 741 m.
    // Chunk columns inside that disc: pi*d^2 / 62^2.
    let radius_m = 161_671.0_f64;
    let eye_m = 1.7_f64;
    let horizon_m = (2.0 * radius_m * eye_m).sqrt();
    let disc_area = std::f64::consts::PI * horizon_m * horizon_m;
    let columns = disc_area / (CHUNK * CHUNK) as f64;
    // Vertical: a walking player needs the surface column plus one chunk above and below.
    let surface_chunks = columns * 3.0;

    println!("\nHORIZON EXTRAPOLATION (R = 161,671 m, eye height 1.7 m)");
    println!("  ground horizon                                       {horizon_m:>9.0} m");
    println!("  chunk COLUMNS inside the horizon disc                 {columns:>9.0}");
    println!("  chunks to generate (3 vertical per column)            {surface_chunks:>9.0}");
    println!(
        "  strategy A, single core                              {:>9.1} s",
        surface_chunks * a_chunk_ms / 1e3
    );
    println!(
        "  strategy B, single core                              {:>9.2} s",
        surface_chunks * b_total_ms / 1e3
    );
    println!(
        "  strategy B, 8 cores                                  {:>9.2} s",
        surface_chunks * b_total_ms / 1e3 / 8.0
    );

    // ---- Flight speed ----------------------------------------------------------------------
    // Sustained flight needs a ring of new chunk columns per second at the horizon edge.
    // New columns per second at speed v = (2 * horizon_m * v) / 62^2  (a swept rectangle).
    println!("\nSUSTAINED FLIGHT (how fast can we move and still generate ahead?)");
    println!("  budget: one core, and separately 8 cores at 50% duty");
    for &v in &[10.0_f64, 50.0, 100.0, 250.0, 500.0, 1000.0] {
        let new_cols_per_s = 2.0 * horizon_m * v / (CHUNK * CHUNK) as f64;
        let chunks_per_s = new_cols_per_s * 3.0;
        let ms_per_s_1core = chunks_per_s * b_total_ms;
        println!(
            "  {v:>6.0} m/s -> {chunks_per_s:>7.0} chunks/s -> {:>7.0} ms/s of one core ({:>5.1}% of 1 core, {:>5.1}% of 8)",
            ms_per_s_1core,
            ms_per_s_1core / 10.0,
            ms_per_s_1core / 80.0
        );
    }

    println!("\nSUMMARY (the numbers to pin)");
    println!("  lattice hash            {ns_hash:.2} ns");
    println!("  one noise evaluation    {ns_eval:.2} ns   (8 hashes + 8 dots + 7 lerps)");
    println!("  fBm 4 octaves           {ns_fbm4:.2} ns");
    println!("  fBm 6 octaves           {ns_fbm6:.2} ns");
    println!("  62^3 chunk, strategy A  {a_chunk_ms:.2} ms");
    println!("  62^3 chunk, strategy B  {b_total_ms:.3} ms  (surface) / {b_height_ms:.3} ms (uniform)");
}
