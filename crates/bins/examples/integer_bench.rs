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
use vd_terrain::chunk::{CHUNK_EDGE, ChunkKey, in_ladder};
use vd_terrain::digest::surface_chunk_z;
use vd_terrain::home::home_planet;
use vd_terrain::lattice::{site_dir, site_of};
use vd_terrain::noise::{corner_hash, noise3};
use vd_terrain::{BodyDefinition, Gf};
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
            let f = o.frequency().to_f64();
            let f_int = f.floor();
            OctaveQ {
                seed: o.seed(),
                frequency_int: f_int as i64,
                frequency_frac_q: to_fixed(f - f_int, FRAC_BITS),
                amplitude_q: (o.amplitude_m().to_f64()
                    * GAP_STEPS_PER_M
                    * f64::from(1u32 << AMP_BITS))
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

    // One CPU core, today's float recipe (the octave sum without the radius, as `height_m`).
    let started = Instant::now();
    let cpu_float: Vec<f64> = dirs
        .iter()
        .map(|d| {
            let dir = [Gf::from_f64(d[0]), Gf::from_f64(d[1]), Gf::from_f64(d[2])];
            let mut h = Gf::ZERO;
            for o in octaves {
                let p = [
                    dir[0] * o.frequency(),
                    dir[1] * o.frequency(),
                    dir[2] * o.frequency(),
                ];
                h += o.amplitude_m() * noise3(o.seed(), p);
            }
            h.to_f64()
        })
        .collect();
    let cpu_float_s = started.elapsed().as_secs_f64();

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

    // THE PRECISION: the fixed-point relief against the float recipe's, in millimetres.
    let mut widest_mm = 0.0_f64;
    let mut sum_mm = 0.0_f64;
    let mut over_one_step = 0usize;
    for (q, f) in cpu_int.iter().zip(cpu_float.iter()) {
        let d_mm = ((*q as f64) / (FRAC_ONE as f64 * GAP_STEPS_PER_M) - f).abs() * 1.0e3;
        widest_mm = widest_mm.max(d_mm);
        sum_mm += d_mm;
        if d_mm > 1.0e3 / GAP_STEPS_PER_M {
            over_one_step += 1;
        }
    }
    println!(
        "integer_bench: PRECISION — the fixed-point relief against the float recipe's on {n} \
         columns: widest {widest_mm:.3} mm, mean {:.3} mm, {over_one_step} columns further than \
         one gap step (7.8 mm) apart",
        sum_mm / n as f64
    );

    // THE COST.
    println!(
        "integer_bench: COST — {n} columns of {} octaves: the integer path on one CPU core {:.1} ms \
         ({:.0} ns a column), the float recipe on one CPU core {:.1} ms ({:.0} ns a column), the \
         GPU {:.1} ms with the upload and the readback",
        octaves_q.len(),
        cpu_int_s * 1.0e3,
        cpu_int_s * 1.0e9 / n as f64,
        cpu_float_s * 1.0e3,
        cpu_float_s * 1.0e9 / n as f64,
        gpu_s * 1.0e3
    );
}

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
                    out.push([d[0].to_f64(), d[1].to_f64(), d[2].to_f64()]);
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

/// One compute pass: the bindings in order, `n` invocations, the output buffer read back.
fn run_compute(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    source: &str,
    bindings: &[Binding],
    n: u32,
) -> Vec<u8> {
    let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("integer_bench"),
        source: wgpu::ShaderSource::Wgsl(source.into()),
    });
    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("integer_bench"),
        layout: None,
        module: &module,
        entry_point: Some("main"),
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
