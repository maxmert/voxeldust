//! ★ THE INTEGER BENCH (ruling F7, 2026-09-12, step (a) of its order): does an INTEGER-ONLY
//! recipe give the same bytes on the CPU and on the GPU, and what does it cost against today's
//! float recipe? The owner ruled the recipe integer-only and the GPU used to the maximum; this is
//! the number before the design.
//!
//! What runs: the height field's octave sum — the recipe's gradient noise (the same corner hash,
//! the same sixteen gradients, the same quintic fade, the same blend order) rewritten in FIXED
//! POINT on 64-bit integers, on the same 3 936 256 column directions the GPU spike used. Both hosts
//! read the same integer inputs (directions at 2⁻³⁰, frequencies at 2⁻⁸, amplitudes in 1/128 m) and
//! write the relief in 1/128 m as a 64-bit integer. The bench compares the two outputs bit for bit,
//! measures the fixed-point relief against the float recipe's on the same columns (the precision
//! the format must hold: the density byte is 1/128 of a cell), and times one CPU core on the
//! integer path, one CPU core on the float path, and the GPU with its upload and readback.
//!
//! THE GPU SIDE IS A THROWAWAY TRANSCRIPTION, an instrument for this measurement only: the
//! product's GPU path is ONE SOURCE compiled for both targets (F7 item 3), which this bench does not
//! build. Nothing here changes the world.
//!
//! Run: `cargo run --release -p vd-bins --features render --example integer_bench`.

use std::time::Instant;

use vd_client_render::wgpu;
use vd_seed::bend::Face;
use vd_terrain::BodyDefinition;
use vd_terrain::body::{octave_amplitude_m, octave_frequency};
use vd_terrain::chunk::{CHUNK_EDGE, ChunkKey, in_ladder};
use vd_terrain::digest::surface_chunk_z;
use vd_terrain::home::home_planet;
use vd_terrain::lattice::{site_dir, site_of};
use vd_terrain::noise::corner_hash;
use wgpu::util::DeviceExt;

/// The chunks the bench samples: the spike's square of rung-0 chunks on face +X.
const CHUNKS_ACROSS: i32 = 32;
const RUNG: u8 = 0;
const FACE: Face = Face::PosX;
const FIRST_X: i32 = 3;
const FIRST_Y: i32 = 5;
const WORKGROUP: u32 = 256;

/// THE FIXED-POINT FORMATS. A direction component at 2⁻³⁰ (|d| ≤ 1 fits 31 bits). An octave's
/// frequency (up to 2.1 × 10⁵ cells per unit direction on the home planet) is an INTEGER PART plus
/// a FRACTION at 2⁻²⁴, so the lattice point is two products: `dir × int` (2³⁰ × 2¹⁸ = 2⁴⁸, exact at
/// 30 fraction bits) plus `dir × frac >> 24` (2⁵⁴ before the shift), summed at 30 fraction bits
/// and shifted to the lattice point's 28. (A frequency rounded to 2⁻⁸ moved the coarsest lattice
/// point by a ten-thousandth of a cell, which eight kilometres of amplitude turned into metres —
/// the first run of this bench.) The fade, the dot and the blend stay at 28 fraction bits, every
/// intermediate under 2⁶⁰. An amplitude in 1/128 m (2²⁴ at most) times a noise value (2²⁶ at
/// most) is 2⁵⁰. THE SUM IS FLOORED ONCE: the octave products are summed at the noise's fraction
/// bits and shifted to 1/128 m at the end — flooring each octave to whole gap steps lost up to one
/// step (7.8 mm) per octave, 57 mm over fourteen (the second run of this bench); and the noise
/// carries 28 fraction bits, because 24 left the coarsest octave's eight kilometres of amplitude
/// a noise error of 6 × 10⁻⁷, five millimetres. THE AMPLITUDE carries eight fraction bits below the
/// gap step (1/32 768 m), each product shifted by those eight before the sum: an amplitude rounded
/// to whole steps cost up to 3.9 mm an octave (the third run). The bench compares the UNFLOORED
/// sum, since the recipe's own gap byte floors once at the end.
const AMP_BITS: u32 = 8;
const DIR_BITS: u32 = 30;
const FRAC_BITS: u32 = 28;
const FRAC_ONE: i64 = 1 << FRAC_BITS;
const FRAC_MASK: i64 = FRAC_ONE - 1;
/// The gap byte's unit: 1/128 of a cell, one metre at rung 0.
const GAP_STEPS_PER_M: f64 = 128.0;

fn main() {
    let body = home_planet();
    let (device, queue, int64) = device();
    let dirs = columns(&body);
    let octaves = body.octaves_at(RUNG);
    println!(
        "integer_bench: {} chunks, {} columns, {} octaves at rung {RUNG}; the GPU offers 64-bit \
         integers: {int64}",
        CHUNKS_ACROSS * CHUNKS_ACROSS,
        dirs.len(),
        octaves.len()
    );
    assert!(
        int64,
        "integer_bench: no 64-bit integers on this GPU — STOP"
    );

    // The integer inputs, the same bytes for both hosts.
    let dirs_q: Vec<i64> = dirs
        .iter()
        .flat_map(|d| d.iter().map(|c| to_fixed(*c, DIR_BITS)))
        .collect();
    let octaves_q: Vec<OctaveQ> = octaves
        .iter()
        .map(|o| {
            let f = octave_frequency(o);
            let f_int = f.floor();
            OctaveQ {
                seed: o.seed,
                frequency_int: f_int as i64,
                frequency_frac_q: to_fixed(f - f_int, FRAC_BITS),
                amplitude_q: (octave_amplitude_m(o) * GAP_STEPS_PER_M * f64::from(1u32 << AMP_BITS))
                    .round() as i64,
            }
        })
        .collect();
    let n = dirs.len();

    // One CPU core, the integer path.
    let started = Instant::now();
    let cpu_int: Vec<i64> = (0..n)
        .map(|i| relief_steps(&octaves_q, &dirs_q[i * 3..i * 3 + 3]))
        .collect();
    let cpu_int_s = started.elapsed().as_secs_f64();

    // One CPU core, THE SHIPPED RECIPE's own kernel (`vd_recipe::height::relief`) on the same
    // directions, widened from this bench's 30 fraction bits to the bend's 40 (the kernel shifts them
    // back, so the inputs are the same words).
    //
    // ★ THE FLOAT LEG IS GONE. This bench once measured the fixed-point relief against the FLOAT
    // recipe's, and that measurement is what carried ruling F7: the recipe is integer-only now, the
    // float recipe no longer exists, and `Gf` has left the shape. What is left to measure is an
    // IDENTITY between three hosts of ONE arithmetic — the bench's transcription, the shipped kernel
    // and the GPU — and the cost of each.
    let started = Instant::now();
    let cpu_recipe: Vec<i64> = dirs_q
        .chunks_exact(3)
        .map(|d| {
            let dir40 = [
                vd_recipe::Gi::new(d[0]) << (vd_recipe::bend::DIR_BITS - DIR_BITS),
                vd_recipe::Gi::new(d[1]) << (vd_recipe::bend::DIR_BITS - DIR_BITS),
                vd_recipe::Gi::new(d[2]) << (vd_recipe::bend::DIR_BITS - DIR_BITS),
            ];
            vd_recipe::height::relief(octaves, dir40).raw()
        })
        .collect();
    let cpu_recipe_s = started.elapsed().as_secs_f64();

    // The GPU, the same integer path, the same integer inputs.
    let mut octave_words: Vec<u32> = Vec::new();
    for o in &octaves_q {
        push_u64(&mut octave_words, o.seed);
        push_u64(&mut octave_words, o.frequency_int as u64);
        push_u64(&mut octave_words, o.frequency_frac_q as u64);
        push_u64(&mut octave_words, o.amplitude_q as u64);
    }
    let count_words = [octaves_q.len() as u32, 0, 0, 0];
    let started = Instant::now();
    let out = run_compute(
        &device,
        &queue,
        RELIEF_SHADER,
        &[
            Binding::Storage(as_bytes_i64(&dirs_q)),
            Binding::Storage(as_bytes_u32(&octave_words)),
            Binding::Output((n * 8) as u64),
            Binding::Uniform(as_bytes_u32(&count_words)),
        ],
        n as u32,
    );
    let gpu_s = started.elapsed().as_secs_f64();
    let gpu_int: Vec<i64> = out
        .chunks_exact(8)
        .map(|b| i64::from_le_bytes(b.try_into().expect("8 bytes")))
        .collect();

    // THE IDENTITY: bit for bit.
    let differing = cpu_int
        .iter()
        .zip(gpu_int.iter())
        .filter(|(a, b)| a != b)
        .count();
    println!(
        "integer_bench: IDENTITY — {n} columns: {differing} differ between the CPU and the GPU \
         on the fixed-point relief"
    );
    if differing > 0 {
        let i = cpu_int
            .iter()
            .zip(gpu_int.iter())
            .position(|(a, b)| a != b)
            .expect("one");
        println!(
            "integer_bench: first difference at column {i}: CPU {} GPU {} (1/128 m at 2⁻²⁸) — STOP",
            cpu_int[i], gpu_int[i]
        );
        std::process::exit(1);
    }

    // THE SHIPPED KERNEL against the transcription: the third host of the same arithmetic.
    let shipped_differing = cpu_int
        .iter()
        .zip(cpu_recipe.iter())
        .filter(|(a, b)| a != b)
        .count();
    println!(
        "integer_bench: IDENTITY — {n} columns: {shipped_differing} differ between this bench's \
         transcription and the shipped recipe's own kernel"
    );

    // THE COST.
    println!(
        "integer_bench: COST — {n} columns of {} octaves: the transcription on one CPU core {:.1} ms \
         ({:.0} ns a column), the shipped recipe on one CPU core {:.1} ms ({:.0} ns a column), the \
         GPU {:.1} ms with the upload and the readback",
        octaves_q.len(),
        cpu_int_s * 1.0e3,
        cpu_int_s * 1.0e9 / n as f64,
        cpu_recipe_s * 1.0e3,
        cpu_recipe_s * 1.0e9 / n as f64,
        gpu_s * 1.0e3
    );

    part_2_the_bend(&device, &queue, &body);
    part_3_the_bend_at_40_bits(&device, &queue, &body);
    part_4_the_one_source(&device, &queue, &body);
}

// ---------------------------------------------------------------- part 4: the one source

/// The environment variable naming the SPIR-V cargo-gpu built from `crates/recipe-gpu`.
const SPV_ENV: &str = "VD_RECIPE_SPV";

/// THE ONE SOURCE ON THE GPU (F8 decision 5): the recipe crate itself, compiled to SPIR-V through
/// rust-gpu (`crates/recipe-gpu`, entry `relief_columns`), loaded through wgpu's SPIR-V front end
/// (naga: SPIR-V → MSL here), run on the bench's columns over the recipe's own octaves, and
/// compared with the CPU's `vd_recipe::height::relief` bit for bit. No hand-written shader: the
/// kernel the GPU runs is the function the server links.
fn part_4_the_one_source(device: &wgpu::Device, queue: &wgpu::Queue, body: &BodyDefinition) {
    let Some(path) = std::env::var_os(SPV_ENV) else {
        println!("integer_bench: PART 4 SKIPPED — set {SPV_ENV} to the SPIR-V cargo-gpu built");
        return;
    };
    let spv = std::fs::read(&path).expect("the SPIR-V file reads");
    let n_l = body.ladder().cells_per_edge(RUNG);
    let inv_n = vd_recipe::bend::inv_n_of(n_l);
    // The columns' 40-bit directions from the recipe's own bend, the same words on both hosts.
    let mut dirs: Vec<i64> = Vec::new();
    for y in FIRST_Y..FIRST_Y + CHUNKS_ACROSS {
        for x in FIRST_X..FIRST_X + CHUNKS_ACROSS {
            let key = ChunkKey {
                face: FACE,
                rung: RUNG,
                x,
                y,
                z: surface_chunk_z(body, FACE, RUNG, x, y),
            };
            for b in 0..CHUNK_EDGE as i32 {
                for a in 0..CHUNK_EDGE as i32 {
                    let site = site_of(body, key, a, b);
                    let d = vd_seed::bend::direction_q(
                        Face::from_index(site.face).expect("a face"),
                        site.i,
                        site.j,
                        inv_n,
                    );
                    dirs.extend([d[0].raw(), d[1].raw(), d[2].raw()]);
                }
            }
        }
    }
    let n = dirs.len() / 3;
    let octaves = body.octaves_at(RUNG);
    let mut words: Vec<u64> = Vec::new();
    for o in octaves {
        words.extend([
            o.seed,
            o.frequency_int.raw() as u64,
            o.frequency_frac.raw() as u64,
            o.amplitude.raw() as u64,
        ]);
    }
    let started = Instant::now();
    let cpu: Vec<i64> = (0..n)
        .map(|i| {
            let d = [
                vd_recipe::Gi::new(dirs[i * 3]),
                vd_recipe::Gi::new(dirs[i * 3 + 1]),
                vd_recipe::Gi::new(dirs[i * 3 + 2]),
            ];
            vd_recipe::height::relief(octaves, d).raw()
        })
        .collect();
    let cpu_s = started.elapsed().as_secs_f64();
    // Twice: the first pass pays the pipeline's own compilation (naga's SPIR-V → MSL, then
    // Metal's compiler); the second is the kernel's cost with the upload and the readback. The
    // octaves go up as the recipe's own `Octave` words, read in place by the module.
    let mut gpu_s = [0.0_f64; 2];
    let mut out = Vec::new();
    let mut pass = 0;
    while pass < 2 {
        let started = Instant::now();
        out = run_compute_source(
            device,
            queue,
            wgpu::util::make_spirv(&spv),
            "relief_columns",
            &[
                Binding::Storage(as_bytes_i64(&dirs)),
                Binding::Storage(as_bytes_u64(&words)),
                Binding::Output((n * 8) as u64),
            ],
            n as u32,
        );
        gpu_s[pass] = started.elapsed().as_secs_f64();
        pass += 1;
    }
    let gpu: Vec<i64> = out
        .chunks_exact(8)
        .map(|b| i64::from_le_bytes(b.try_into().expect("8 bytes")))
        .collect();
    let differing = cpu.iter().zip(gpu.iter()).filter(|(a, b)| a != b).count();
    println!(
        "integer_bench: PART 4 THE ONE SOURCE — {n} columns of {} octaves through the recipe crate \
         compiled to SPIR-V: {differing} differ between the CPU and the GPU (CPU {:.1} ms; the GPU \
         {:.1} ms on the first pass with the pipeline's compilation, {:.1} ms on the second, both \
         with the upload and the readback)",
        octaves.len(),
        cpu_s * 1.0e3,
        gpu_s[0] * 1.0e3,
        gpu_s[1] * 1.0e3
    );
    if differing > 0 {
        let i = cpu
            .iter()
            .zip(gpu.iter())
            .position(|(a, b)| a != b)
            .expect("one");
        println!(
            "integer_bench: PART 4 first difference at column {i}: CPU {} GPU {} — STOP",
            cpu[i], gpu[i]
        );
        std::process::exit(1);
    }
}

fn as_bytes_u64(v: &[u64]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}

// ------------------------------------------------------------- part 3: the bend at 40 bits

/// The two-word product of two unsigned 64-bit words from 32-bit halves: `(hi, lo)`.
fn mul_wide(a: u64, b: u64) -> (u64, u64) {
    let (a0, a1) = (a & 0xFFFF_FFFF, a >> 32);
    let (b0, b1) = (b & 0xFFFF_FFFF, b >> 32);
    let p00 = a0 * b0;
    let p01 = a0 * b1;
    let p10 = a1 * b0;
    let p11 = a1 * b1;
    let mid = (p00 >> 32) + (p01 & 0xFFFF_FFFF) + (p10 & 0xFFFF_FFFF);
    let lo = (p00 & 0xFFFF_FFFF) | (mid << 32);
    let hi = p11 + (p01 >> 32) + (p10 >> 32) + (mid >> 32);
    (hi, lo)
}

/// `(hi, lo) >> s` for `s` in `1..=127`, as one word (the caller keeps the result under 64 bits).
fn shr_wide(hi: u64, lo: u64, s: u32) -> u64 {
    if s >= 64 {
        hi >> (s - 64)
    } else {
        (lo >> s) | (hi << (64 - s))
    }
}

/// `(a × b) >> s` on signed words by sign and magnitude: truncated toward zero on both hosts.
fn mul_shr(a: i64, b: i64, s: u32) -> i64 {
    let (hi, lo) = mul_wide(a.unsigned_abs(), b.unsigned_abs());
    let m = shr_wide(hi, lo, s) as i64;
    if (a < 0) != (b < 0) { -m } else { m }
}

const Q40: u32 = 40;
const ONE40: i64 = 1 << Q40;

struct Bend40 {
    k1: i64,
    k2: i64,
    k3: i64,
}

fn bend40_constants() -> Bend40 {
    let one = ONE40 as f64;
    let k1 = (vd_seed::bend::K1 * one).round() as i64;
    let k2 = (vd_seed::bend::K2 * one).round() as i64;
    Bend40 {
        k1,
        k2,
        k3: ONE40 - k1 - k2,
    }
}

fn bend40(k: &Bend40, a: i64) -> i64 {
    let a2 = mul_shr(a, a, Q40);
    let inner = k.k2 + mul_shr(a2, k.k3, Q40);
    let inner2 = k.k1 + mul_shr(a2, inner, Q40);
    mul_shr(a, inner2, Q40)
}

/// The unit direction of `(face, i, j)` at 40 fraction bits: the square sum at 80 bits in two
/// words; the reciprocal square root seeded by the 30-bit path's exact reciprocal of the 30-bit
/// root, then ONE Newton step `y ← y + y(1 − S·y²)/2` with the residual carried at 120 bits.
fn direction40(k: &Bend40, face: Face, inv_n: i64, i: i32, j: i32) -> [i64; 3] {
    let basis = face.basis();
    let wa = bend40(
        k,
        mul_shr(2 * i64::from(i) + 1, inv_n, INV_BITS - Q40) - ONE40,
    );
    let wb = bend40(
        k,
        mul_shr(2 * i64::from(j) + 1, inv_n, INV_BITS - Q40) - ONE40,
    );
    let axis = |c: usize| {
        i64::from(basis.n[c]) * ONE40 + wa * i64::from(basis.u[c]) + wb * i64::from(basis.v[c])
    };
    let v = [axis(0), axis(1), axis(2)];
    // S = Σ v² at 80 fraction bits, in two words.
    let (mut s_hi, mut s_lo) = (0u64, 0u64);
    let mut c = 0;
    while c < 3 {
        let m = v[c].unsigned_abs();
        let (h, l) = mul_wide(m, m);
        let (nl, carry) = s_lo.overflowing_add(l);
        s_lo = nl;
        s_hi = s_hi + h + u64::from(carry);
        c += 1;
    }
    // T = S >> 20: S at 60 fraction bits in one word (S < 3·2⁸⁰, so T < 2⁶²).
    let t = shr_wide(s_hi, s_lo, 20);
    // The seed: the 30-bit root and its exact reciprocal, widened to 40 bits.
    let y0 = recip_q(isqrt(t as i64)) << 10;
    // y0² at 60 fraction bits, in one word.
    let (yh, yl) = mul_wide(y0 as u64, y0 as u64);
    let y0sq = shr_wide(yh, yl, 20);
    // P = T × y0² = S·y0² at 120 fraction bits, in two words; E = 2¹²⁰ − P (signed).
    let (ph, pl) = mul_wide(t, y0sq);
    let one120_hi = 1u64 << 56;
    let (e_neg, e_hi, e_lo) = if ph > one120_hi || (ph == one120_hi && pl > 0) {
        let (l, borrow) = pl.overflowing_sub(0);
        (true, ph - one120_hi - u64::from(borrow), l)
    } else {
        let (l, borrow) = 0u64.overflowing_sub(pl);
        (false, one120_hi - ph - u64::from(borrow), l)
    };
    // e at 70 fraction bits in one word (|E| ≈ 2⁻³⁰ · 2¹²⁰ = 2⁹⁰, so E >> 50 < 2⁴¹).
    let e70 = shr_wide(e_hi, e_lo, 50) as i64;
    let e70 = if e_neg { -e70 } else { e70 };
    // y1 = y0 + y0·e/2: (y0 at 40) × (e at 70) >> 71.
    let y1 = y0 + mul_shr(y0, e70, 71);
    [
        mul_shr(v[0], y1, Q40),
        mul_shr(v[1], y1, Q40),
        mul_shr(v[2], y1, Q40),
    ]
}

fn part_3_the_bend_at_40_bits(device: &wgpu::Device, queue: &wgpu::Queue, body: &BodyDefinition) {
    let n_l = body.ladder().cells_per_edge(RUNG);
    let inv_n = inv_n_of(n_l);
    let radius_m = body.ladder().radius_m();
    let k = bend40_constants();
    let mut sites: Vec<i32> = Vec::new();
    for y in FIRST_Y..FIRST_Y + CHUNKS_ACROSS {
        for x in FIRST_X..FIRST_X + CHUNKS_ACROSS {
            let key = ChunkKey {
                face: FACE,
                rung: RUNG,
                x,
                y,
                z: surface_chunk_z(body, FACE, RUNG, x, y),
            };
            for b in 0..CHUNK_EDGE as i32 {
                for a in 0..CHUNK_EDGE as i32 {
                    let site = site_of(body, key, a, b);
                    sites.push(i32::from(site.face));
                    sites.push(site.i);
                    sites.push(site.j);
                }
            }
        }
    }
    let n = sites.len() / 3;

    let started = Instant::now();
    let cpu: Vec<i64> = (0..n)
        .flat_map(|c| {
            let face = Face::from_index(sites[c * 3] as u8).expect("a face");
            direction40(&k, face, inv_n, sites[c * 3 + 1], sites[c * 3 + 2])
        })
        .collect();
    let cpu_s = started.elapsed().as_secs_f64();

    let float: Vec<[f64; 3]> = (0..n)
        .map(|c| {
            let face = Face::from_index(sites[c * 3] as u8).expect("a face");
            vd_seed::bend::direction(
                face,
                vd_seed::ladder::face_param(sites[c * 3 + 1], n_l),
                vd_seed::ladder::face_param(sites[c * 3 + 2], n_l),
            )
        })
        .collect();

    let mut words: Vec<u32> = Vec::new();
    push_u64(&mut words, k.k1 as u64);
    push_u64(&mut words, k.k2 as u64);
    push_u64(&mut words, k.k3 as u64);
    push_u64(&mut words, inv_n as u64);
    push_u64(&mut words, 0);
    push_u64(&mut words, 0);
    let started = Instant::now();
    let out = run_compute(
        device,
        queue,
        BEND40_SHADER,
        &[
            Binding::Storage(as_bytes_i32(&sites)),
            Binding::Output((n * 24) as u64),
            Binding::Uniform(as_bytes_u32(&words)),
        ],
        n as u32,
    );
    let gpu_s = started.elapsed().as_secs_f64();
    let gpu: Vec<i64> = out
        .chunks_exact(8)
        .map(|b| i64::from_le_bytes(b.try_into().expect("8 bytes")))
        .collect();
    let differing = cpu.iter().zip(gpu.iter()).filter(|(a, b)| a != b).count();
    println!(
        "integer_bench: PART 3 THE BEND AT 40 BITS — {n} columns: {differing} of {} direction \
         components differ between the CPU and the GPU",
        n * 3
    );
    if differing > 0 {
        let i = cpu
            .iter()
            .zip(gpu.iter())
            .position(|(a, b)| a != b)
            .expect("one");
        println!(
            "integer_bench: PART 3 first difference at component {i}: CPU {} GPU {} — STOP",
            cpu[i], gpu[i]
        );
        std::process::exit(1);
    }
    let (mut widest_mm, mut sum_mm) = (0.0_f64, 0.0_f64);
    for c in 0..n {
        let mut d2 = 0.0_f64;
        for a in 0..3 {
            let q = cpu[c * 3 + a] as f64 / ONE40 as f64;
            let e = q - float[c][a];
            d2 += e * e;
        }
        let mm = d2.sqrt() * radius_m * 1.0e3;
        widest_mm = widest_mm.max(mm);
        sum_mm += mm;
    }
    println!(
        "integer_bench: PART 3 PRECISION — the 40-bit direction against the float bend's, \
         laterally on the surface: widest {widest_mm:.4} mm, mean {:.4} mm (a step of 2⁻⁴⁰ is \
         {:.4} mm here)",
        sum_mm / n as f64,
        radius_m * 1.0e3 / ONE40 as f64
    );
    println!(
        "integer_bench: PART 3 COST — the 40-bit bend on one CPU core {:.1} ms ({:.0} ns a \
         column, {:.3} ms a chunk of 3 844), the GPU {:.1} ms with the upload and the readback",
        cpu_s * 1.0e3,
        cpu_s * 1.0e9 / n as f64,
        cpu_s * 1.0e3 / n as f64 * 3844.0,
        gpu_s * 1.0e3
    );
}

/// THE THROWAWAY TRANSCRIPTION of the 40-bit bend (an instrument only, see the header).
const BEND40_SHADER: &str = r"
@group(0) @binding(0) var<storage, read> sites: array<i32>;
@group(0) @binding(1) var<storage, read_write> dirs: array<i64>;
@group(0) @binding(2) var<uniform> k: array<vec4<u32>, 3>;

const Q40: u32 = 40u;
const ONE40: i64 = 1099511627776li;
const BEND_BITS: u32 = 30u;
const BEND_ONE: i64 = 1073741824li;
const INV_SHIFT40: u32 = 16u;

fn u64_of(w: vec2<u32>) -> u64 { return (u64(w.y) << 32u) | u64(w.x); }

fn basis_n(face: i32) -> vec3<i64> {
    switch face {
        case 0: { return vec3<i64>(1li, 0li, 0li); }
        case 1: { return vec3<i64>(-1li, 0li, 0li); }
        case 2: { return vec3<i64>(0li, 1li, 0li); }
        case 3: { return vec3<i64>(0li, -1li, 0li); }
        case 4: { return vec3<i64>(0li, 0li, 1li); }
        default: { return vec3<i64>(0li, 0li, -1li); }
    }
}
fn basis_u(face: i32) -> vec3<i64> {
    switch face {
        case 0: { return vec3<i64>(0li, 1li, 0li); }
        case 1: { return vec3<i64>(0li, 0li, 1li); }
        case 2: { return vec3<i64>(0li, 0li, 1li); }
        case 3: { return vec3<i64>(1li, 0li, 0li); }
        case 4: { return vec3<i64>(1li, 0li, 0li); }
        default: { return vec3<i64>(0li, 1li, 0li); }
    }
}
fn basis_v(face: i32) -> vec3<i64> {
    switch face {
        case 0: { return vec3<i64>(0li, 0li, 1li); }
        case 1: { return vec3<i64>(0li, 1li, 0li); }
        case 2: { return vec3<i64>(1li, 0li, 0li); }
        case 3: { return vec3<i64>(0li, 0li, 1li); }
        case 4: { return vec3<i64>(0li, 1li, 0li); }
        default: { return vec3<i64>(1li, 0li, 0li); }
    }
}

fn mul_wide(a: u64, b: u64) -> vec2<u64> {
    let a0 = a & 0xFFFFFFFFlu; let a1 = a >> 32u;
    let b0 = b & 0xFFFFFFFFlu; let b1 = b >> 32u;
    let p00 = a0 * b0;
    let p01 = a0 * b1;
    let p10 = a1 * b0;
    let p11 = a1 * b1;
    let mid = (p00 >> 32u) + (p01 & 0xFFFFFFFFlu) + (p10 & 0xFFFFFFFFlu);
    let lo = (p00 & 0xFFFFFFFFlu) | (mid << 32u);
    let hi = p11 + (p01 >> 32u) + (p10 >> 32u) + (mid >> 32u);
    return vec2<u64>(hi, lo);
}

fn shr_wide(hi: u64, lo: u64, s: u32) -> u64 {
    if (s >= 64u) { return hi >> (s - 64u); }
    return (lo >> s) | (hi << (64u - s));
}

fn uabs(a: i64) -> u64 { return select(u64(a), u64(-a), a < 0li); }

fn mul_shr(a: i64, b: i64, s: u32) -> i64 {
    let p = mul_wide(uabs(a), uabs(b));
    let m = i64(shr_wide(p.x, p.y, s));
    return select(m, -m, (a < 0li) != (b < 0li));
}

fn bend40(k1: i64, k2: i64, k3: i64, a: i64) -> i64 {
    let a2 = mul_shr(a, a, Q40);
    let inner = k2 + mul_shr(a2, k3, Q40);
    let inner2 = k1 + mul_shr(a2, inner, Q40);
    return mul_shr(a, inner2, Q40);
}

fn isqrt(n: i64) -> i64 {
    var x = u64(n);
    var res = 0lu;
    var bit = 1lu << 62u;
    while (bit > x) { bit = bit >> 2u; }
    while (bit != 0lu) {
        if (x >= res + bit) {
            x = x - (res + bit);
            res = (res >> 1u) + bit;
        } else {
            res = res >> 1u;
        }
        bit = bit >> 2u;
    }
    return i64(res);
}

fn recip_q(len: i64) -> i64 {
    var r = BEND_ONE;
    for (var step = 0u; step < 6u; step = step + 1u) {
        let e = (1li << 60u) - len * r;
        r = r + ((r * (e >> BEND_BITS)) >> BEND_BITS);
    }
    let top = 1li << 60u;
    while (len * (r + 1li) <= top) { r = r + 1li; }
    while (len * r > top) { r = r - 1li; }
    return r;
}

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let c = id.x;
    if (c * 3u >= arrayLength(&dirs)) { return; }
    let k1 = i64(u64_of(k[0].xy));
    let k2 = i64(u64_of(k[0].zw));
    let k3 = i64(u64_of(k[1].xy));
    let inv_n = i64(u64_of(k[1].zw));
    let face = sites[c * 3u];
    let i = sites[c * 3u + 1u];
    let j = sites[c * 3u + 2u];
    let wa = bend40(k1, k2, k3, mul_shr(2li * i64(i) + 1li, inv_n, INV_SHIFT40) - ONE40);
    let wb = bend40(k1, k2, k3, mul_shr(2li * i64(j) + 1li, inv_n, INV_SHIFT40) - ONE40);
    let v = basis_n(face) * ONE40 + basis_u(face) * wa + basis_v(face) * wb;
    var s_hi = 0lu;
    var s_lo = 0lu;
    for (var q = 0u; q < 3u; q = q + 1u) {
        let m = uabs(v[q]);
        let p = mul_wide(m, m);
        let nl = s_lo + p.y;
        let carry = select(0lu, 1lu, nl < s_lo);
        s_lo = nl;
        s_hi = s_hi + p.x + carry;
    }
    let t = shr_wide(s_hi, s_lo, 20u);
    let y0 = recip_q(isqrt(i64(t))) << 10u;
    let ysq = mul_wide(u64(y0), u64(y0));
    let y0sq = shr_wide(ysq.x, ysq.y, 20u);
    let p = mul_wide(t, y0sq);
    let one120_hi = 1lu << 56u;
    var e_neg = false;
    var e_hi = 0lu;
    var e_lo = 0lu;
    if (p.x > one120_hi || (p.x == one120_hi && p.y > 0lu)) {
        e_neg = true;
        e_hi = p.x - one120_hi;
        e_lo = p.y;
    } else {
        e_lo = 0lu - p.y;
        e_hi = one120_hi - p.x - select(0lu, 1lu, p.y != 0lu);
    }
    var e70 = i64(shr_wide(e_hi, e_lo, 50u));
    e70 = select(e70, -e70, e_neg);
    let y1 = y0 + mul_shr(y0, e70, 71u);
    dirs[c * 3u] = mul_shr(v.x, y1, Q40);
    dirs[c * 3u + 1u] = mul_shr(v.y, y1, Q40);
    dirs[c * 3u + 2u] = mul_shr(v.z, y1, Q40);
}
";

// ---------------------------------------------------------------------- part 2: the bend

/// THE INTEGER BEND: the face parameter, the quintic bend, the basis, the normalise (an integer
/// square root and three integer divisions) at 30 fraction bits, on the CPU and on the GPU, on the
/// same columns' `(face, i, j)`; compared bit for bit, and against the float bend's directions as a
/// lateral distance on the home planet's surface.
const BEND_BITS: u32 = 30;
const BEND_ONE: i64 = 1 << BEND_BITS;

struct BendQ {
    k1: i64,
    k2: i64,
    k3: i64,
}

fn bend_constants() -> BendQ {
    let one = BEND_ONE as f64;
    let k1 = (vd_seed::bend::K1 * one).round() as i64;
    let k2 = (vd_seed::bend::K2 * one).round() as i64;
    BendQ {
        k1,
        k2,
        k3: BEND_ONE - k1 - k2,
    }
}

/// THE RECIPROCAL OF THE FACE'S CELL COUNT, computed ONCE per body: `floor(2⁵⁶ / n_l)`, so the
/// face parameter is a multiply and a shift per column — no division on either host. (naga 27's
/// Metal back end emits a guard for the 64-bit `/` operator that Metal's compiler rejects as
/// ambiguous, MEASURED by this bench; a recipe without `/` needs no such operator anywhere.)
const INV_BITS: u32 = 56;

fn inv_n_of(n_l: u32) -> i64 {
    ((1i128 << INV_BITS) / i128::from(n_l)) as i64
}

/// `(2i + 1)/n_l − 1` at 30 fraction bits by the reciprocal: `(2i + 1) × inv_n >> 26`.
fn face_param_q(i: i32, inv_n: i64) -> i64 {
    ((((2 * i64::from(i)) + 1) * inv_n) >> (INV_BITS - BEND_BITS)) - BEND_ONE
}

/// `2⁶⁰ / len` at 30 fraction bits by SIX Newton steps from the seed 1.0 (`len` is in [1, √3], so
/// the seed is under 2/len and the iteration converges): `r ← r + r·(2⁶⁰ − len·r) / 2⁶⁰`, the
/// residual shifted to 30 bits before the product so nothing passes 2⁶⁰.
fn recip_q(len: i64) -> i64 {
    let mut r = BEND_ONE;
    let mut step = 0;
    while step < 6 {
        let e = (1i64 << (2 * BEND_BITS)) - len * r;
        r += (r * (e >> BEND_BITS)) >> BEND_BITS;
        step += 1;
    }
    // THE EXACT LANDING: the Newton steps truncate a few units low; two compare loops make `r`
    // exactly `floor(2⁶⁰ / len)`, so the reciprocal is the division's own answer.
    let top = 1i64 << (2 * BEND_BITS);
    while len * (r + 1) <= top {
        r += 1;
    }
    while len * r > top {
        r -= 1;
    }
    r
}

/// `a(k₁ + a²(k₂ + a²k₃))` at 30 fraction bits, every product shifted back once.
fn bend_q(k: &BendQ, a: i64) -> i64 {
    let a2 = (a * a) >> BEND_BITS;
    let inner = k.k2 + ((a2 * k.k3) >> BEND_BITS);
    let inner2 = k.k1 + ((a2 * inner) >> BEND_BITS);
    (a * inner2) >> BEND_BITS
}

/// The integer square root of a non-negative 64-bit number, one bit at a time.
fn isqrt(n: i64) -> i64 {
    let mut x = n as u64;
    let mut res = 0u64;
    let mut bit = 1u64 << 62;
    while bit > x {
        bit >>= 2;
    }
    while bit != 0 {
        if x >= res + bit {
            x -= res + bit;
            res = (res >> 1) + bit;
        } else {
            res >>= 1;
        }
        bit >>= 2;
    }
    res as i64
}

/// The unit direction of `(face, i, j)` at 30 fraction bits.
fn direction_q(k: &BendQ, face: Face, inv_n: i64, i: i32, j: i32) -> [i64; 3] {
    let basis = face.basis();
    let wa = bend_q(k, face_param_q(i, inv_n));
    let wb = bend_q(k, face_param_q(j, inv_n));
    let axis = |c: usize| {
        i64::from(basis.n[c]) * BEND_ONE + wa * i64::from(basis.u[c]) + wb * i64::from(basis.v[c])
    };
    let v = [axis(0), axis(1), axis(2)];
    let r = recip_q(isqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]));
    let half = 1i64 << (BEND_BITS - 1);
    [
        (v[0] * r + half) >> BEND_BITS,
        (v[1] * r + half) >> BEND_BITS,
        (v[2] * r + half) >> BEND_BITS,
    ]
}

fn part_2_the_bend(device: &wgpu::Device, queue: &wgpu::Queue, body: &BodyDefinition) {
    let n_l = body.ladder().cells_per_edge(RUNG);
    let inv_n = inv_n_of(n_l);
    let radius_m = body.ladder().radius_m();
    let k = bend_constants();
    // The columns' (face, i, j), the same square as part 1.
    let mut sites: Vec<i32> = Vec::new();
    for y in FIRST_Y..FIRST_Y + CHUNKS_ACROSS {
        for x in FIRST_X..FIRST_X + CHUNKS_ACROSS {
            let key = ChunkKey {
                face: FACE,
                rung: RUNG,
                x,
                y,
                z: surface_chunk_z(body, FACE, RUNG, x, y),
            };
            for b in 0..CHUNK_EDGE as i32 {
                for a in 0..CHUNK_EDGE as i32 {
                    let site = site_of(body, key, a, b);
                    assert!(
                        Face::from_index(site.face).is_some(),
                        "a corner in the square"
                    );
                    sites.push(i32::from(site.face));
                    sites.push(site.i);
                    sites.push(site.j);
                }
            }
        }
    }
    let n = sites.len() / 3;

    let started = Instant::now();
    let cpu: Vec<i64> = (0..n)
        .flat_map(|c| {
            let face = Face::from_index(sites[c * 3] as u8).expect("a face");
            direction_q(&k, face, inv_n, sites[c * 3 + 1], sites[c * 3 + 2])
        })
        .collect();
    let cpu_s = started.elapsed().as_secs_f64();

    let started = Instant::now();
    let float: Vec<[f64; 3]> = (0..n)
        .map(|c| {
            let face = Face::from_index(sites[c * 3] as u8).expect("a face");
            vd_seed::bend::direction(
                face,
                vd_seed::ladder::face_param(sites[c * 3 + 1], n_l),
                vd_seed::ladder::face_param(sites[c * 3 + 2], n_l),
            )
        })
        .collect();
    let float_s = started.elapsed().as_secs_f64();

    let mut words: Vec<u32> = Vec::new();
    push_u64(&mut words, k.k1 as u64);
    push_u64(&mut words, k.k2 as u64);
    push_u64(&mut words, k.k3 as u64);
    push_u64(&mut words, inv_n as u64);
    push_u64(&mut words, 0);
    push_u64(&mut words, 0);
    let started = Instant::now();
    let out = run_compute(
        device,
        queue,
        BEND_SHADER,
        &[
            Binding::Storage(as_bytes_i32(&sites)),
            Binding::Output((n * 24) as u64),
            Binding::Uniform(as_bytes_u32(&words)),
        ],
        n as u32,
    );
    let gpu_s = started.elapsed().as_secs_f64();
    let gpu: Vec<i64> = out
        .chunks_exact(8)
        .map(|b| i64::from_le_bytes(b.try_into().expect("8 bytes")))
        .collect();
    let differing = cpu.iter().zip(gpu.iter()).filter(|(a, b)| a != b).count();
    println!(
        "integer_bench: PART 2 THE BEND — {n} columns: {differing} of {} direction components \
         differ between the CPU and the GPU",
        n * 3
    );
    if differing > 0 {
        let i = cpu
            .iter()
            .zip(gpu.iter())
            .position(|(a, b)| a != b)
            .expect("one");
        println!(
            "integer_bench: PART 2 first difference at component {i}: CPU {} GPU {} — STOP",
            cpu[i], gpu[i]
        );
        std::process::exit(1);
    }
    let (mut widest_mm, mut sum_mm) = (0.0_f64, 0.0_f64);
    for c in 0..n {
        let mut d2 = 0.0_f64;
        for a in 0..3 {
            let q = cpu[c * 3 + a] as f64 / BEND_ONE as f64;
            let e = q - float[c][a];
            d2 += e * e;
        }
        let mm = d2.sqrt() * radius_m * 1.0e3;
        widest_mm = widest_mm.max(mm);
        sum_mm += mm;
    }
    println!(
        "integer_bench: PART 2 PRECISION — the integer direction against the float bend's, as a \
         lateral distance on the surface: widest {widest_mm:.3} mm, mean {:.3} mm (a direction \
         step of 2⁻³⁰ is {:.3} mm here)",
        sum_mm / n as f64,
        radius_m * 1.0e3 / BEND_ONE as f64
    );
    println!(
        "integer_bench: PART 2 COST — the integer bend on one CPU core {:.1} ms ({:.0} ns a \
         column), the float bend {:.1} ms ({:.0} ns a column), the GPU {:.1} ms with the upload \
         and the readback",
        cpu_s * 1.0e3,
        cpu_s * 1.0e9 / n as f64,
        float_s * 1.0e3,
        float_s * 1.0e9 / n as f64,
        gpu_s * 1.0e3
    );
}

/// THE THROWAWAY TRANSCRIPTION of the integer bend (an instrument only, see the header).
const BEND_SHADER: &str = r"
@group(0) @binding(0) var<storage, read> sites: array<i32>;
@group(0) @binding(1) var<storage, read_write> dirs: array<i64>;
@group(0) @binding(2) var<uniform> k: array<vec4<u32>, 3>;

const BEND_BITS: u32 = 30u;
const BEND_ONE: i64 = 1073741824li;

fn u64_of(w: vec2<u32>) -> u64 { return (u64(w.y) << 32u) | u64(w.x); }

fn basis_n(face: i32) -> vec3<i64> {
    switch face {
        case 0: { return vec3<i64>(1li, 0li, 0li); }
        case 1: { return vec3<i64>(-1li, 0li, 0li); }
        case 2: { return vec3<i64>(0li, 1li, 0li); }
        case 3: { return vec3<i64>(0li, -1li, 0li); }
        case 4: { return vec3<i64>(0li, 0li, 1li); }
        default: { return vec3<i64>(0li, 0li, -1li); }
    }
}
fn basis_u(face: i32) -> vec3<i64> {
    switch face {
        case 0: { return vec3<i64>(0li, 1li, 0li); }
        case 1: { return vec3<i64>(0li, 0li, 1li); }
        case 2: { return vec3<i64>(0li, 0li, 1li); }
        case 3: { return vec3<i64>(1li, 0li, 0li); }
        case 4: { return vec3<i64>(1li, 0li, 0li); }
        default: { return vec3<i64>(0li, 1li, 0li); }
    }
}
fn basis_v(face: i32) -> vec3<i64> {
    switch face {
        case 0: { return vec3<i64>(0li, 0li, 1li); }
        case 1: { return vec3<i64>(0li, 1li, 0li); }
        case 2: { return vec3<i64>(1li, 0li, 0li); }
        case 3: { return vec3<i64>(0li, 0li, 1li); }
        case 4: { return vec3<i64>(0li, 1li, 0li); }
        default: { return vec3<i64>(1li, 0li, 0li); }
    }
}

const INV_SHIFT: u32 = 26u;

fn face_param_q(i: i32, inv_n: i64) -> i64 {
    return ((((2li * i64(i)) + 1li) * inv_n) >> INV_SHIFT) - BEND_ONE;
}

fn recip_q(len: i64) -> i64 {
    var r = BEND_ONE;
    for (var step = 0u; step < 6u; step = step + 1u) {
        let e = (1li << 60u) - len * r;
        r = r + ((r * (e >> BEND_BITS)) >> BEND_BITS);
    }
    let top = 1li << 60u;
    while (len * (r + 1li) <= top) { r = r + 1li; }
    while (len * r > top) { r = r - 1li; }
    return r;
}

fn bend_q(k1: i64, k2: i64, k3: i64, a: i64) -> i64 {
    let a2 = (a * a) >> BEND_BITS;
    let inner = k2 + ((a2 * k3) >> BEND_BITS);
    let inner2 = k1 + ((a2 * inner) >> BEND_BITS);
    return (a * inner2) >> BEND_BITS;
}

fn isqrt(n: i64) -> i64 {
    var x = u64(n);
    var res = 0lu;
    var bit = 1lu << 62u;
    while (bit > x) { bit = bit >> 2u; }
    while (bit != 0lu) {
        if (x >= res + bit) {
            x = x - (res + bit);
            res = (res >> 1u) + bit;
        } else {
            res = res >> 1u;
        }
        bit = bit >> 2u;
    }
    return i64(res);
}

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let c = id.x;
    if (c * 3u >= arrayLength(&dirs)) { return; }
    let k1 = i64(u64_of(k[0].xy));
    let k2 = i64(u64_of(k[0].zw));
    let k3 = i64(u64_of(k[1].xy));
    let inv_n = i64(u64_of(k[1].zw));
    let face = sites[c * 3u];
    let i = sites[c * 3u + 1u];
    let j = sites[c * 3u + 2u];
    let wa = bend_q(k1, k2, k3, face_param_q(i, inv_n));
    let wb = bend_q(k1, k2, k3, face_param_q(j, inv_n));
    let v = basis_n(face) * BEND_ONE + basis_u(face) * wa + basis_v(face) * wb;
    let r = recip_q(isqrt(v.x * v.x + v.y * v.y + v.z * v.z));
    let half = 1li << 29u;
    dirs[c * 3u] = (v.x * r + half) >> BEND_BITS;
    dirs[c * 3u + 1u] = (v.y * r + half) >> BEND_BITS;
    dirs[c * 3u + 2u] = (v.z * r + half) >> BEND_BITS;
}
";

/// One octave in fixed point.
struct OctaveQ {
    seed: u64,
    frequency_int: i64,
    frequency_frac_q: i64,
    amplitude_q: i64,
}

fn to_fixed(v: f64, bits: u32) -> i64 {
    (v * f64::from(1u32 << bits)).round() as i64
}

fn push_u64(words: &mut Vec<u32>, v: u64) {
    words.push(v as u32);
    words.push((v >> 32) as u32);
}

// -------------------------------------------------------------------------- the integer recipe

/// Perlin's sixteen gradients, as in the recipe.
const GRADIENTS: [[i64; 3]; 16] = [
    [1, 1, 0],
    [-1, 1, 0],
    [1, -1, 0],
    [-1, -1, 0],
    [1, 0, 1],
    [-1, 0, 1],
    [1, 0, -1],
    [-1, 0, -1],
    [0, 1, 1],
    [0, -1, 1],
    [0, 1, -1],
    [0, -1, -1],
    [1, 1, 0],
    [-1, 1, 0],
    [0, -1, 1],
    [0, -1, -1],
];

/// The quintic fade `t³(t(6t − 15) + 10)` at 28 fraction bits, every product shifted back once.
fn fade_q(t: i64) -> i64 {
    let t2 = (t * t) >> FRAC_BITS;
    let t3 = (t2 * t) >> FRAC_BITS;
    let inner = 6 * t - 15 * FRAC_ONE;
    let poly = ((inner * t) >> FRAC_BITS) + 10 * FRAC_ONE;
    (t3 * poly) >> FRAC_BITS
}

/// `a + t(b − a)` at 28 fraction bits.
fn lerp_q(a: i64, b: i64, t: i64) -> i64 {
    a + ((t * (b - a)) >> FRAC_BITS)
}

fn dot_q(g: [i64; 3], dx: i64, dy: i64, dz: i64) -> i64 {
    g[0] * dx + g[1] * dy + g[2] * dz
}

fn gradient_q(seed: u64, x: i64, y: i64, z: i64) -> [i64; 3] {
    GRADIENTS[(corner_hash(seed, x, y, z) & 15) as usize]
}

/// Gradient noise at a fixed-point lattice point (28 fraction bits), in about `[−1, 1]` at 28
/// fraction bits, blended in the recipe's order.
fn noise3_q(seed: u64, p: [i64; 3]) -> i64 {
    let x0 = p[0] >> FRAC_BITS;
    let y0 = p[1] >> FRAC_BITS;
    let z0 = p[2] >> FRAC_BITS;
    let dx = p[0] & FRAC_MASK;
    let dy = p[1] & FRAC_MASK;
    let dz = p[2] & FRAC_MASK;
    let u = fade_q(dx);
    let v = fade_q(dy);
    let w = fade_q(dz);
    let one = FRAC_ONE;
    let c000 = dot_q(gradient_q(seed, x0, y0, z0), dx, dy, dz);
    let c100 = dot_q(gradient_q(seed, x0 + 1, y0, z0), dx - one, dy, dz);
    let c010 = dot_q(gradient_q(seed, x0, y0 + 1, z0), dx, dy - one, dz);
    let c110 = dot_q(gradient_q(seed, x0 + 1, y0 + 1, z0), dx - one, dy - one, dz);
    let c001 = dot_q(gradient_q(seed, x0, y0, z0 + 1), dx, dy, dz - one);
    let c101 = dot_q(gradient_q(seed, x0 + 1, y0, z0 + 1), dx - one, dy, dz - one);
    let c011 = dot_q(gradient_q(seed, x0, y0 + 1, z0 + 1), dx, dy - one, dz - one);
    let c111 = dot_q(
        gradient_q(seed, x0 + 1, y0 + 1, z0 + 1),
        dx - one,
        dy - one,
        dz - one,
    );
    let x00 = lerp_q(c000, c100, u);
    let x10 = lerp_q(c010, c110, u);
    let x01 = lerp_q(c001, c101, u);
    let x11 = lerp_q(c011, c111, u);
    let y0v = lerp_q(x00, x10, v);
    let y1v = lerp_q(x01, x11, v);
    lerp_q(y0v, y1v, w)
}

/// The relief along one fixed-point direction in 1/128 m at `FRAC_BITS` fraction bits, unfloored:
/// the octave sum without the radius.
fn relief_steps(octaves: &[OctaveQ], dir_q: &[i64]) -> i64 {
    let mut h = 0i64;
    for o in octaves {
        let scale = |d: i64| {
            (d * o.frequency_int + ((d * o.frequency_frac_q) >> FRAC_BITS))
                >> (DIR_BITS - FRAC_BITS)
        };
        let p = [scale(dir_q[0]), scale(dir_q[1]), scale(dir_q[2])];
        h += (o.amplitude_q * noise3_q(o.seed, p)) >> AMP_BITS;
    }
    h
}

// ------------------------------------------------------------------------------- the GPU twin

/// THE THROWAWAY TRANSCRIPTION of the integer recipe above, an instrument only (see the header).
const RELIEF_SHADER: &str = r"
struct Octave { seed: vec2<u32>, f_int: vec2<u32>, f_frac: vec2<u32>, amplitude: vec2<u32> };
@group(0) @binding(0) var<storage, read> dirs: array<i64>;
@group(0) @binding(1) var<storage, read> octaves: array<Octave>;
@group(0) @binding(2) var<storage, read_write> relief: array<i64>;
@group(0) @binding(3) var<uniform> count: vec4<u32>;

const FRAC_BITS: u32 = 28u;
const FRAC_ONE: i64 = 268435456li;
const FRAC_MASK: i64 = 268435455li;
const DIR_SHIFT: u32 = 2u;
const AMP_BITS: u32 = 8u;

fn scale(d: i64, f_int: i64, f_frac: i64) -> i64 {
    return (d * f_int + ((d * f_frac) >> FRAC_BITS)) >> DIR_SHIFT;
}

fn u64_of(w: vec2<u32>) -> u64 { return (u64(w.y) << 32u) | u64(w.x); }

fn splitmix(key: u64) -> u64 {
    var z = key + 0x9E3779B97F4A7C15lu;
    z = (z ^ (z >> 30u)) * 0xBF58476D1CE4E5B9lu;
    z = (z ^ (z >> 27u)) * 0x94D049BB133111EBlu;
    return z ^ (z >> 31u);
}

fn corner_hash(s: u64, x: i64, y: i64, z: i64) -> u64 {
    let key = s ^ (u64(x) * 0x9E3779B97F4A7C15lu) ^ (u64(y) * 0xBF58476D1CE4E5B9lu) ^ (u64(z) * 0x94D049BB133111EBlu);
    return splitmix(key);
}

fn gradient(s: u64, x: i64, y: i64, z: i64) -> vec3<i64> {
    let g = u32(corner_hash(s, x, y, z) & 15lu);
    switch g {
        case 0u: { return vec3<i64>(1li, 1li, 0li); }
        case 1u: { return vec3<i64>(-1li, 1li, 0li); }
        case 2u: { return vec3<i64>(1li, -1li, 0li); }
        case 3u: { return vec3<i64>(-1li, -1li, 0li); }
        case 4u: { return vec3<i64>(1li, 0li, 1li); }
        case 5u: { return vec3<i64>(-1li, 0li, 1li); }
        case 6u: { return vec3<i64>(1li, 0li, -1li); }
        case 7u: { return vec3<i64>(-1li, 0li, -1li); }
        case 8u: { return vec3<i64>(0li, 1li, 1li); }
        case 9u: { return vec3<i64>(0li, -1li, 1li); }
        case 10u: { return vec3<i64>(0li, 1li, -1li); }
        case 11u: { return vec3<i64>(0li, -1li, -1li); }
        case 12u: { return vec3<i64>(1li, 1li, 0li); }
        case 13u: { return vec3<i64>(-1li, 1li, 0li); }
        case 14u: { return vec3<i64>(0li, -1li, 1li); }
        default: { return vec3<i64>(0li, -1li, -1li); }
    }
}

fn fade_q(t: i64) -> i64 {
    let t2 = (t * t) >> FRAC_BITS;
    let t3 = (t2 * t) >> FRAC_BITS;
    let inner = 6li * t - 15li * FRAC_ONE;
    let poly = ((inner * t) >> FRAC_BITS) + 10li * FRAC_ONE;
    return (t3 * poly) >> FRAC_BITS;
}

fn lerp_q(a: i64, b: i64, t: i64) -> i64 {
    return a + ((t * (b - a)) >> FRAC_BITS);
}

fn dot_q(g: vec3<i64>, dx: i64, dy: i64, dz: i64) -> i64 {
    return g.x * dx + g.y * dy + g.z * dz;
}

fn noise3_q(s: u64, p: vec3<i64>) -> i64 {
    let x0 = p.x >> FRAC_BITS;
    let y0 = p.y >> FRAC_BITS;
    let z0 = p.z >> FRAC_BITS;
    let dx = p.x & FRAC_MASK;
    let dy = p.y & FRAC_MASK;
    let dz = p.z & FRAC_MASK;
    let u = fade_q(dx);
    let v = fade_q(dy);
    let w = fade_q(dz);
    let one = FRAC_ONE;
    let c000 = dot_q(gradient(s, x0, y0, z0), dx, dy, dz);
    let c100 = dot_q(gradient(s, x0 + 1li, y0, z0), dx - one, dy, dz);
    let c010 = dot_q(gradient(s, x0, y0 + 1li, z0), dx, dy - one, dz);
    let c110 = dot_q(gradient(s, x0 + 1li, y0 + 1li, z0), dx - one, dy - one, dz);
    let c001 = dot_q(gradient(s, x0, y0, z0 + 1li), dx, dy, dz - one);
    let c101 = dot_q(gradient(s, x0 + 1li, y0, z0 + 1li), dx - one, dy, dz - one);
    let c011 = dot_q(gradient(s, x0, y0 + 1li, z0 + 1li), dx, dy - one, dz - one);
    let c111 = dot_q(gradient(s, x0 + 1li, y0 + 1li, z0 + 1li), dx - one, dy - one, dz - one);
    let x00 = lerp_q(c000, c100, u);
    let x10 = lerp_q(c010, c110, u);
    let x01 = lerp_q(c001, c101, u);
    let x11 = lerp_q(c011, c111, u);
    let y0v = lerp_q(x00, x10, v);
    let y1v = lerp_q(x01, x11, v);
    return lerp_q(y0v, y1v, w);
}

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let i = id.x;
    if (i >= arrayLength(&relief)) { return; }
    let d = vec3<i64>(dirs[i * 3u], dirs[i * 3u + 1u], dirs[i * 3u + 2u]);
    var h: i64 = 0li;
    for (var k = 0u; k < count.x; k = k + 1u) {
        let o = octaves[k];
        let s = u64_of(o.seed);
        let fi = i64(u64_of(o.f_int));
        let ff = i64(u64_of(o.f_frac));
        let a = i64(u64_of(o.amplitude));
        let p = vec3<i64>(scale(d.x, fi, ff), scale(d.y, fi, ff), scale(d.z, fi, ff));
        h = h + ((a * noise3_q(s, p)) >> AMP_BITS);
    }
    relief[i] = h;
}
";

// --------------------------------------------------------------------------------- the harness

/// A bare compute device, with 64-bit integers when the adapter offers them.
fn device() -> (wgpu::Device, wgpu::Queue, bool) {
    let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor::default());
    let adapter = block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
        power_preference: wgpu::PowerPreference::HighPerformance,
        ..Default::default()
    }))
    .expect("a GPU adapter");
    let info = adapter.get_info();
    println!(
        "integer_bench: adapter {} ({:?}, {:?})",
        info.name, info.backend, info.device_type
    );
    let int64 = adapter.features().contains(wgpu::Features::SHADER_INT64);
    let (device, queue) = block_on(adapter.request_device(&wgpu::DeviceDescriptor {
        label: Some("integer_bench"),
        required_features: if int64 {
            wgpu::Features::SHADER_INT64
        } else {
            wgpu::Features::empty()
        },
        required_limits: adapter.limits(),
        ..Default::default()
    }))
    .expect("a device");
    (device, queue, int64)
}

/// The column directions of the sampled chunks, as the recipe computes them (64-bit).
fn columns(body: &BodyDefinition) -> Vec<[f64; 3]> {
    let mut out = Vec::new();
    for y in FIRST_Y..FIRST_Y + CHUNKS_ACROSS {
        for x in FIRST_X..FIRST_X + CHUNKS_ACROSS {
            let key = ChunkKey {
                face: FACE,
                rung: RUNG,
                x,
                y,
                z: surface_chunk_z(body, FACE, RUNG, x, y),
            };
            assert!(in_ladder(body, key), "chunk ({x}, {y}) is off the ladder");
            for b in 0..CHUNK_EDGE as i32 {
                for a in 0..CHUNK_EDGE as i32 {
                    let d = site_dir(body, key, site_of(body, key, a, b));
                    out.push(vd_terrain::units::unit_of_direction(d));
                }
            }
        }
    }
    out
}

enum Binding {
    Storage(Vec<u8>),
    Output(u64),
    Uniform(Vec<u8>),
}

/// One compute pass from a WGSL string (the throwaway transcriptions): the bindings in order, `n`
/// invocations, the output buffer read back.
fn run_compute(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    source: &str,
    bindings: &[Binding],
    n: u32,
) -> Vec<u8> {
    run_compute_source(
        device,
        queue,
        wgpu::ShaderSource::Wgsl(source.into()),
        "main",
        bindings,
        n,
    )
}

/// One compute pass from any shader source and entry point (part 4 passes the SPIR-V the one
/// source compiled to).
fn run_compute_source(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    source: wgpu::ShaderSource<'_>,
    entry: &str,
    bindings: &[Binding],
    n: u32,
) -> Vec<u8> {
    let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("integer_bench"),
        source,
    });
    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("integer_bench"),
        layout: None,
        module: &module,
        entry_point: Some(entry),
        compilation_options: wgpu::PipelineCompilationOptions::default(),
        cache: None,
    });
    let mut buffers = Vec::new();
    let mut output: Option<(usize, u64)> = None;
    for (i, b) in bindings.iter().enumerate() {
        let buffer = match b {
            Binding::Storage(bytes) => {
                device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: None,
                    contents: bytes,
                    usage: wgpu::BufferUsages::STORAGE,
                })
            }
            Binding::Uniform(bytes) => {
                device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: None,
                    contents: bytes,
                    usage: wgpu::BufferUsages::UNIFORM,
                })
            }
            Binding::Output(size) => {
                output = Some((i, *size));
                device.create_buffer(&wgpu::BufferDescriptor {
                    label: None,
                    size: *size,
                    usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
                    mapped_at_creation: false,
                })
            }
        };
        buffers.push(buffer);
    }
    let entries: Vec<wgpu::BindGroupEntry> = buffers
        .iter()
        .enumerate()
        .map(|(i, b)| wgpu::BindGroupEntry {
            binding: i as u32,
            resource: b.as_entire_binding(),
        })
        .collect();
    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: None,
        layout: &pipeline.get_bind_group_layout(0),
        entries: &entries,
    });
    let (out_index, out_size) = output.expect("one output binding");
    let staging = device.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size: out_size,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor::default());
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        pass.dispatch_workgroups(n.div_ceil(WORKGROUP), 1, 1);
    }
    encoder.copy_buffer_to_buffer(&buffers[out_index], 0, &staging, 0, out_size);
    queue.submit([encoder.finish()]);
    let slice = staging.slice(..);
    let (tx, rx) = std::sync::mpsc::channel();
    slice.map_async(wgpu::MapMode::Read, move |r| {
        tx.send(r).expect("the receiver waits");
    });
    device
        .poll(wgpu::PollType::wait_indefinitely())
        .expect("the device polls");
    rx.recv().expect("a map result").expect("the map succeeds");
    let bytes = slice.get_mapped_range().to_vec();
    staging.unmap();
    bytes
}

fn as_bytes_i32(v: &[i32]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}

fn as_bytes_i64(v: &[i64]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}

fn as_bytes_u32(v: &[u32]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}

/// A minimal executor for the two async calls of device creation: poll with a no-op waker.
fn block_on<F: std::future::Future>(future: F) -> F::Output {
    let mut future = std::pin::pin!(future);
    let waker = std::task::Waker::noop();
    let mut cx = std::task::Context::from_waker(waker);
    loop {
        if let std::task::Poll::Ready(v) = future.as_mut().poll(&mut cx) {
            return v;
        }
        std::thread::yield_now();
    }
}
