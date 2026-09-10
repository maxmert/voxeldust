//! ★ THE GPU SPIKE (ruling V17 item 3, 2026-09-10): can the world's recipe run on the GPU byte for
//! byte? The recipe computes with 64-bit floats (`Gf`), which the Mac's GPU does not have, so the
//! spike asks the two questions that remain, in order, each with a stop condition:
//!
//! 1. THE INTEGER HALF — the corner hash (`SplitMix64` on 64-bit integers) in a compute shader
//!    against the CPU, on millions of corners. Integers have no rounding: one differing bit means
//!    the GPU's 64-bit integer path is not what its feature flag says, and GPU generation ends.
//! 2. THE 32-BIT QUESTION — a SHADOW of the recipe's noise in 32-bit floats (integer lattice
//!    addressing, the same operations in the same order) on the CPU and in a compute shader, over
//!    the column directions of a thousand chunks, byte for byte. Differences mean the shader
//!    compiler contracts or reorders operations (fused multiply-add, fast math) beyond our
//!    control, and no float recipe can cross to a GPU without drift.
//!
//! The shadow changes nothing in the world: it lives here, outside the fence, as an instrument.
//!
//! Run: `cargo run --release -p vd-bins --features render --example gpu_spike`.

use std::time::Instant;

use vd_client_render::wgpu;
use vd_seed::bend::Face;
use vd_terrain::chunk::{CHUNK_EDGE, ChunkKey, in_ladder};
use vd_terrain::digest::surface_chunk_z;
use vd_terrain::home::home_planet;
use vd_terrain::lattice::{site_dir, site_of};
use vd_terrain::noise::corner_hash;
use vd_terrain::{BodyDefinition, Gf};
use wgpu::util::DeviceExt;

/// The chunks the spike samples: a square of rung-0 chunks on face +X around the ground stand.
const CHUNKS_ACROSS: i32 = 32;
const RUNG: u8 = 0;
const FACE: Face = Face::PosX;
const FIRST_X: i32 = 3;
const FIRST_Y: i32 = 5;
/// The compute shader's workgroup width.
const WORKGROUP: u32 = 256;

fn main() {
    let body = home_planet();
    let (device, queue, int64) = device();
    let columns = columns(&body);
    println!(
        "gpu_spike: {} chunks, {} columns, {} octaves at rung {RUNG}; the GPU offers 64-bit \
         integers: {int64}",
        CHUNKS_ACROSS * CHUNKS_ACROSS,
        columns.len(),
        body.octaves_at(RUNG).len()
    );
    if !int64 {
        println!(
            "gpu_spike: PART 1 SKIPPED — no 64-bit integers on this GPU; the emulated path is next"
        );
        return;
    }
    part_1_the_integer_half(&device, &queue, &body, &columns);
    part_2_the_32_bit_question(&device, &queue, &body, &columns);
}

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
        "gpu_spike: adapter {} ({:?}, {:?})",
        info.name, info.backend, info.device_type
    );
    let int64 = adapter.features().contains(wgpu::Features::SHADER_INT64);
    let (device, queue) = block_on(adapter.request_device(&wgpu::DeviceDescriptor {
        label: Some("gpu_spike"),
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

/// The column directions of the sampled chunks, as the recipe computes them (64-bit), narrowed
/// once to 32-bit for both hosts of the shadow.
fn columns(body: &BodyDefinition) -> Vec<[f32; 3]> {
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
                    out.push([
                        d[0].to_f64() as f32,
                        d[1].to_f64() as f32,
                        d[2].to_f64() as f32,
                    ]);
                }
            }
        }
    }
    out
}

// ------------------------------------------------------------------------------------ part 1

const HASH_SHADER: &str = r"
@group(0) @binding(0) var<storage, read> corners: array<i32>;
@group(0) @binding(1) var<storage, read_write> hashes: array<u64>;
@group(0) @binding(2) var<uniform> seed: vec2<u32>;

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

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let i = id.x;
    if (i >= arrayLength(&hashes)) { return; }
    let s = (u64(seed.y) << 32u) | u64(seed.x);
    let x = i64(corners[i * 3u]);
    let y = i64(corners[i * 3u + 1u]);
    let z = i64(corners[i * 3u + 2u]);
    hashes[i] = corner_hash(s, x, y, z);
}
";

/// The corner hash on the GPU against the CPU: the corners are the lattice cubes of every column
/// at every octave, and their eight neighbours' offsets folded in by the column index.
fn part_1_the_integer_half(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    body: &BodyDefinition,
    columns: &[[f32; 3]],
) {
    let octaves = body.octaves_at(RUNG);
    let seed = octaves[0].seed();
    let mut corners: Vec<i32> = Vec::with_capacity(columns.len() * 3);
    for (i, c) in columns.iter().enumerate() {
        let f = octaves[i % octaves.len()].frequency().to_f64() as f32;
        let k = (i & 7) as i32;
        corners.push((c[0] * f).floor() as i32 + (k & 1));
        corners.push((c[1] * f).floor() as i32 + ((k >> 1) & 1));
        corners.push((c[2] * f).floor() as i32 + ((k >> 2) & 1));
    }
    let n = columns.len();
    let cpu_started = Instant::now();
    let cpu: Vec<u64> = (0..n)
        .map(|i| {
            corner_hash(
                seed,
                i64::from(corners[i * 3]),
                i64::from(corners[i * 3 + 1]),
                i64::from(corners[i * 3 + 2]),
            )
        })
        .collect();
    let cpu_s = cpu_started.elapsed().as_secs_f64();
    let seed_words = [seed as u32, (seed >> 32) as u32];
    let gpu_started = Instant::now();
    let out = run_compute(
        device,
        queue,
        HASH_SHADER,
        &[
            Binding::Storage(as_bytes_i32(&corners)),
            Binding::Output((n * 8) as u64),
            Binding::Uniform(as_bytes_u32(&seed_words)),
        ],
        n as u32,
    );
    let gpu_s = gpu_started.elapsed().as_secs_f64();
    let gpu: Vec<u64> = out
        .chunks_exact(8)
        .map(|b| u64::from_le_bytes(b.try_into().expect("8 bytes")))
        .collect();
    let differing = cpu.iter().zip(gpu.iter()).filter(|(a, b)| a != b).count();
    println!(
        "gpu_spike: PART 1 — {n} corner hashes: {differing} differ between the CPU and the GPU \
         (CPU {:.1} ms, GPU {:.1} ms with the upload and the readback)",
        cpu_s * 1.0e3,
        gpu_s * 1.0e3
    );
    if differing > 0 {
        let i = cpu
            .iter()
            .zip(gpu.iter())
            .position(|(a, b)| a != b)
            .expect("one");
        println!(
            "gpu_spike: PART 1 first difference at corner {i}: CPU {:#018x}, GPU {:#018x} — STOP",
            cpu[i], gpu[i]
        );
        std::process::exit(1);
    }
}

// ------------------------------------------------------------------------------------ part 2

/// The shadow's noise on the CPU: `noise3` with every float a 32-bit one, the same operations in
/// the same order, the corner hash unchanged.
mod shadow {
    use vd_terrain::noise::corner_hash;

    const GRADIENTS: [[i8; 3]; 16] = [
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

    fn gradient(seed: u64, x: i64, y: i64, z: i64) -> [f32; 3] {
        let g = GRADIENTS[(corner_hash(seed, x, y, z) & 15) as usize];
        [f32::from(g[0]), f32::from(g[1]), f32::from(g[2])]
    }

    fn fade(t: f32) -> f32 {
        t * t * t * (t * (t * 6.0 - 15.0) + 10.0)
    }

    fn lerp(a: f32, b: f32, t: f32) -> f32 {
        a + t * (b - a)
    }

    fn dot(g: [f32; 3], dx: f32, dy: f32, dz: f32) -> f32 {
        g[0] * dx + g[1] * dy + g[2] * dz
    }

    pub fn noise3(seed: u64, p: [f32; 3]) -> f32 {
        let fx = p[0].floor();
        let fy = p[1].floor();
        let fz = p[2].floor();
        let x0 = fx as i64;
        let y0 = fy as i64;
        let z0 = fz as i64;
        let dx = p[0] - fx;
        let dy = p[1] - fy;
        let dz = p[2] - fz;
        let u = fade(dx);
        let v = fade(dy);
        let w = fade(dz);
        let c000 = dot(gradient(seed, x0, y0, z0), dx, dy, dz);
        let c100 = dot(gradient(seed, x0 + 1, y0, z0), dx - 1.0, dy, dz);
        let c010 = dot(gradient(seed, x0, y0 + 1, z0), dx, dy - 1.0, dz);
        let c110 = dot(gradient(seed, x0 + 1, y0 + 1, z0), dx - 1.0, dy - 1.0, dz);
        let c001 = dot(gradient(seed, x0, y0, z0 + 1), dx, dy, dz - 1.0);
        let c101 = dot(gradient(seed, x0 + 1, y0, z0 + 1), dx - 1.0, dy, dz - 1.0);
        let c011 = dot(gradient(seed, x0, y0 + 1, z0 + 1), dx, dy - 1.0, dz - 1.0);
        let c111 = dot(
            gradient(seed, x0 + 1, y0 + 1, z0 + 1),
            dx - 1.0,
            dy - 1.0,
            dz - 1.0,
        );
        let x00 = lerp(c000, c100, u);
        let x10 = lerp(c010, c110, u);
        let x01 = lerp(c001, c101, u);
        let x11 = lerp(c011, c111, u);
        let y0v = lerp(x00, x10, v);
        let y1v = lerp(x01, x11, v);
        lerp(y0v, y1v, w)
    }

    /// The height's octave sum, the radius left out (one add, the same on both hosts).
    pub fn relief(octaves: &[(u64, f32, f32)], dir: [f32; 3]) -> f32 {
        let mut h = 0.0f32;
        for (seed, frequency, amplitude) in octaves {
            let p = [dir[0] * frequency, dir[1] * frequency, dir[2] * frequency];
            h += amplitude * noise3(*seed, p);
        }
        h
    }
}

const NOISE_SHADER: &str = r"
struct Octave { seed_lo: u32, seed_hi: u32, frequency: f32, amplitude: f32 }
@group(0) @binding(0) var<storage, read> dirs: array<f32>;
@group(0) @binding(1) var<storage, read_write> relief: array<f32>;
@group(0) @binding(2) var<storage, read> octaves: array<Octave>;

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

fn gradient(s: u64, x: i64, y: i64, z: i64) -> vec3<f32> {
    let h = u32(corner_hash(s, x, y, z) & 15lu);
    // Perlin's twelve edges, four repeated: the same table as the recipe's.
    switch h {
        case 0u: { return vec3<f32>(1.0, 1.0, 0.0); }
        case 1u: { return vec3<f32>(-1.0, 1.0, 0.0); }
        case 2u: { return vec3<f32>(1.0, -1.0, 0.0); }
        case 3u: { return vec3<f32>(-1.0, -1.0, 0.0); }
        case 4u: { return vec3<f32>(1.0, 0.0, 1.0); }
        case 5u: { return vec3<f32>(-1.0, 0.0, 1.0); }
        case 6u: { return vec3<f32>(1.0, 0.0, -1.0); }
        case 7u: { return vec3<f32>(-1.0, 0.0, -1.0); }
        case 8u: { return vec3<f32>(0.0, 1.0, 1.0); }
        case 9u: { return vec3<f32>(0.0, -1.0, 1.0); }
        case 10u: { return vec3<f32>(0.0, 1.0, -1.0); }
        case 11u: { return vec3<f32>(0.0, -1.0, -1.0); }
        case 12u: { return vec3<f32>(1.0, 1.0, 0.0); }
        case 13u: { return vec3<f32>(-1.0, 1.0, 0.0); }
        case 14u: { return vec3<f32>(0.0, -1.0, 1.0); }
        default: { return vec3<f32>(0.0, -1.0, -1.0); }
    }
}

fn fade(t: f32) -> f32 { return t * t * t * (t * (t * 6.0 - 15.0) + 10.0); }
fn lerp(a: f32, b: f32, t: f32) -> f32 { return a + t * (b - a); }
fn dotg(g: vec3<f32>, dx: f32, dy: f32, dz: f32) -> f32 { return g.x * dx + g.y * dy + g.z * dz; }

fn noise3(s: u64, p: vec3<f32>) -> f32 {
    let fx = floor(p.x);
    let fy = floor(p.y);
    let fz = floor(p.z);
    let x0 = i64(fx);
    let y0 = i64(fy);
    let z0 = i64(fz);
    let dx = p.x - fx;
    let dy = p.y - fy;
    let dz = p.z - fz;
    let u = fade(dx);
    let v = fade(dy);
    let w = fade(dz);
    let c000 = dotg(gradient(s, x0, y0, z0), dx, dy, dz);
    let c100 = dotg(gradient(s, x0 + 1li, y0, z0), dx - 1.0, dy, dz);
    let c010 = dotg(gradient(s, x0, y0 + 1li, z0), dx, dy - 1.0, dz);
    let c110 = dotg(gradient(s, x0 + 1li, y0 + 1li, z0), dx - 1.0, dy - 1.0, dz);
    let c001 = dotg(gradient(s, x0, y0, z0 + 1li), dx, dy, dz - 1.0);
    let c101 = dotg(gradient(s, x0 + 1li, y0, z0 + 1li), dx - 1.0, dy, dz - 1.0);
    let c011 = dotg(gradient(s, x0, y0 + 1li, z0 + 1li), dx, dy - 1.0, dz - 1.0);
    let c111 = dotg(gradient(s, x0 + 1li, y0 + 1li, z0 + 1li), dx - 1.0, dy - 1.0, dz - 1.0);
    let x00 = lerp(c000, c100, u);
    let x10 = lerp(c010, c110, u);
    let x01 = lerp(c001, c101, u);
    let x11 = lerp(c011, c111, u);
    let y0v = lerp(x00, x10, v);
    let y1v = lerp(x01, x11, v);
    return lerp(y0v, y1v, w);
}

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let i = id.x;
    if (i >= arrayLength(&relief)) { return; }
    let d = vec3<f32>(dirs[i * 3u], dirs[i * 3u + 1u], dirs[i * 3u + 2u]);
    var h = 0.0;
    for (var o = 0u; o < arrayLength(&octaves); o = o + 1u) {
        let oc = octaves[o];
        let s = (u64(oc.seed_hi) << 32u) | u64(oc.seed_lo);
        let p = d * oc.frequency;
        h = h + oc.amplitude * noise3(s, p);
    }
    relief[i] = h;
}
";

/// The 32-bit shadow of the recipe's noise, CPU against GPU, over every column of the sampled
/// chunks; the real 64-bit recipe timed beside them for the payoff.
fn part_2_the_32_bit_question(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    body: &BodyDefinition,
    columns: &[[f32; 3]],
) {
    let octaves: Vec<(u64, f32, f32)> = body
        .octaves_at(RUNG)
        .iter()
        .map(|o| {
            (
                o.seed(),
                o.frequency().to_f64() as f32,
                o.amplitude_m().to_f64() as f32,
            )
        })
        .collect();
    let n = columns.len();
    let cpu_started = Instant::now();
    let cpu: Vec<f32> = columns
        .iter()
        .map(|d| shadow::relief(&octaves, *d))
        .collect();
    let cpu_s = cpu_started.elapsed().as_secs_f64();
    // The real recipe, 64-bit, on the same directions: the time GPU generation would buy.
    let real_started = Instant::now();
    let mut real_sum = 0.0f64;
    for d in columns {
        let dir = [
            Gf::from_f64(f64::from(d[0])),
            Gf::from_f64(f64::from(d[1])),
            Gf::from_f64(f64::from(d[2])),
        ];
        real_sum += vd_terrain::height::height_m(body, dir, RUNG).to_f64();
    }
    let real_s = real_started.elapsed().as_secs_f64();

    let mut octave_words: Vec<u32> = Vec::new();
    for (seed, f, a) in &octaves {
        octave_words.push(*seed as u32);
        octave_words.push((*seed >> 32) as u32);
        octave_words.push(f.to_bits());
        octave_words.push(a.to_bits());
    }
    let flat: Vec<f32> = columns.iter().flat_map(|d| d.iter().copied()).collect();
    let gpu_started = Instant::now();
    let out = run_compute(
        device,
        queue,
        NOISE_SHADER,
        &[
            Binding::Storage(as_bytes_f32(&flat)),
            Binding::Output((n * 4) as u64),
            Binding::Storage(as_bytes_u32(&octave_words)),
        ],
        n as u32,
    );
    let gpu_s = gpu_started.elapsed().as_secs_f64();
    let gpu: Vec<f32> = out
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes(b.try_into().expect("4 bytes")))
        .collect();
    let mut differing = 0usize;
    let mut widest_ulps = 0u32;
    let mut widest_m = 0.0f32;
    let mut first: Option<usize> = None;
    for (i, (a, b)) in cpu.iter().zip(gpu.iter()).enumerate() {
        if a.to_bits() != b.to_bits() {
            differing += 1;
            first.get_or_insert(i);
            widest_ulps = widest_ulps.max(a.to_bits().abs_diff(b.to_bits()));
            widest_m = widest_m.max((a - b).abs());
        }
    }
    println!(
        "gpu_spike: PART 2 — {n} columns of relief in 32-bit: {differing} differ between the CPU \
         and the GPU (the widest {widest_ulps} ulps, {widest_m:.3e} m); CPU 32-bit {:.1} ms, GPU \
         {:.1} ms with the upload and the readback, the real 64-bit recipe {:.1} ms (sum {real_sum:.1})",
        cpu_s * 1.0e3,
        gpu_s * 1.0e3,
        real_s * 1.0e3
    );
    if let Some(i) = first {
        println!(
            "gpu_spike: PART 2 first difference at column {i}: CPU {:.9} ({:#010x}), GPU {:.9} \
             ({:#010x})",
            cpu[i],
            cpu[i].to_bits(),
            gpu[i],
            gpu[i].to_bits()
        );
    }
}

// --------------------------------------------------------------------------------- the harness

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
        label: Some("gpu_spike"),
        source: wgpu::ShaderSource::Wgsl(source.into()),
    });
    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("gpu_spike"),
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

fn as_bytes_i32(v: &[i32]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}

fn as_bytes_u32(v: &[u32]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}

fn as_bytes_f32(v: &[f32]) -> Vec<u8> {
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
