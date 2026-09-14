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
//! THE GPU SIDE OF PARTS 1 TO 3 IS A THROWAWAY TRANSCRIPTION, an instrument for those measurements
//! only. Parts 4 and 5 run the product's own path: ONE SOURCE compiled for both targets (F7 item 3),
//! the recipe crate itself through `crates/recipe-gpu`. Nothing here changes the world.
//!
//! ★ PART 5 (step G1, 2026-09-13) is the CELL FIELD: every cell of a box — a chunk's 62³ plus its
//! one-cell halo — through the recipe's own `cell::cell_word` on the card, against the CPU's
//! `sample_box`, byte for byte. It runs the eight golden chunks the world identity folds and the
//! square of 1 024 chunks, and it reports what each host costs. Two PROBES run before it: each runs
//! ONE kernel over a list of words, because a box of a quarter of a million cells can only say that
//! something differs, and a probe says which function does.
//!
//! ★ PART 6 (step G2-A, 2026-09-13) GREW PART 5 INTO THE WHOLE CHAIN: the COLUMN pass and the NODE
//! pass now run on the card in front of the cell field, so the host uploads TOPOLOGY only. The part
//! compares the columns' own DIRECTIONS as well as the cells — a direction that differs in its last
//! bit can still pack the same cell byte — and it times the same three passes WITHOUT the readback,
//! which is the number that says whether the bus or the card's arithmetic is the cost.
//!
//! Run: `just recipe-gpu` first, then
//! `VD_RECIPE_SPV=target/recipe-gpu/vd_recipe_gpu.spv cargo run --release -p vd-bins --features \
//! render --example integer_bench` (`VD_BENCH_PART5=1` measures part 5 alone).

use std::time::Instant;

use vd_client_render::wgpu;
use vd_seed::bend::Face;
use vd_terrain::BodyDefinition;
use vd_terrain::body::{octave_amplitude_m, octave_frequency};
use vd_terrain::chunk::{CHUNK_EDGE, ChunkKey, in_ladder};
use vd_terrain::digest::{GOLDEN_SELF_CHECK_KEYS, self_check_key, surface_chunk_z};
use vd_terrain::home::home_planet;
use vd_terrain::lattice::{BOX_CELLS, BOX_EDGE, site_dir, site_of};
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
    // A measurement of part 5 alone (`VD_BENCH_PART5=1`) skips the four column parts, which cost
    // four million columns each and answer nothing about the cell field.
    if std::env::var_os("VD_BENCH_PART5").is_some() {
        part_5_the_cell_field(&device, &queue, &body);
        return;
    }
    // ★ THE DRIFT HUNT (`VD_BENCH_PART7=1`, 2026-09-14): the seam stand's own wanted set through
    // the CLIENT'S OWN GEAR, box for box against the CPU. It answers where the seam stand's hole
    // comes from — the kernel, or the client's path around it.
    if std::env::var_os("VD_BENCH_PART7").is_some() {
        part_7_the_drift_hunt(&device, &queue, &body);
        return;
    }
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
    part_5_the_cell_field(&device, &queue, &body);
}

/// A PROBE: the recipe's integer square root on the card against the CPU's.
fn probe_isqrt(device: &wgpu::Device, queue: &wgpu::Queue, spv: &[u8]) {
    // ★ THE WHOLE WORD'S RANGE, not just its bottom. The root takes thirty-two steps whatever the
    // value, and a probe whose widest input is 2⁴⁰ never runs the first eleven of them on the card:
    // a fault in the top steps would pass unseen. So the list carries the word's own ends, the
    // squares just under and just over each power of four's boundary, and a spread of full-width
    // words from the recipe's own hash.
    let mut input: Vec<i64> = vec![
        0,
        1,
        2,
        3,
        4,
        5,
        15,
        16,
        17,
        100,
        380_000,
        1 << 40,
        1 << 62,
        (1 << 62) + 12_345,
        1 << 63,
        (1 << 63) + 1,
        -1,
        i64::MAX,
        i64::MIN,
    ];
    let mut i = 1i64;
    while i < 500 {
        input.push(i * i * 7 + 3);
        // The same step at the word's top: a square of a 32-bit root, one under it and one over.
        let big = (i << 22) + 7_919;
        input.push(big.wrapping_mul(big));
        input.push(big.wrapping_mul(big).wrapping_sub(1));
        // A full-width word of the recipe's own hash: a value no pattern in this list reaches.
        input.push(vd_recipe::rng::corner_hash(0x5EED, i, i * 7, -i) as i64);
        i += 1;
    }
    let n = input.len();
    let out = run_compute_source(
        device,
        queue,
        wgpu::util::make_spirv(spv),
        "isqrt_probe",
        &[
            Binding::Storage(as_bytes_i64(&input)),
            Binding::Output((n * 8) as u64),
        ],
        n as u32,
    );
    let gpu: Vec<i64> = out
        .chunks_exact(8)
        .map(|b| i64::from_le_bytes(b.try_into().expect("8 bytes")))
        .collect();
    let mut differing = 0;
    let mut first = String::new();
    for (i, v) in input.iter().enumerate() {
        let cpu = vd_recipe::root::isqrt(*v as u64) as i64;
        if cpu != gpu[i] {
            differing += 1;
            if first.is_empty() {
                first = format!(
                    " — the first: isqrt({v}) is {cpu} on the CPU and {} on the GPU",
                    gpu[i]
                );
            }
        }
    }
    println!("integer_bench: PROBE isqrt — {n} words: {differing} differ{first}");
    if differing > 0 {
        std::process::exit(1);
    }
    probe_hollow(device, queue, spv);
}

/// A PROBE: the carvers' hollow on the card against the CPU's.
fn probe_hollow(device: &wgpu::Device, queue: &wgpu::Queue, spv: &[u8]) {
    let tube = vd_recipe::cell::Tube {
        start: [
            vd_recipe::Gi::new(0),
            vd_recipe::Gi::new(0),
            vd_recipe::Gi::new(0),
        ],
        end: [
            vd_recipe::Gi::new(1_000),
            vd_recipe::Gi::new(0),
            vd_recipe::Gi::new(0),
        ],
        radius_steps: vd_recipe::Gi::new(400),
        inv_len2: vd_recipe::Gi::new(vd_recipe::root::recip_pow2(
            1_000 * 1_000,
            vd_recipe::cell::TUBE_RECIP_BITS,
        ) as i64),
    };
    let tube_words: Vec<i64> = vec![
        tube.start[0].raw(),
        tube.start[1].raw(),
        tube.start[2].raw(),
        tube.end[0].raw(),
        tube.end[1].raw(),
        tube.end[2].raw(),
        tube.radius_steps.raw(),
        tube.inv_len2.raw(),
    ];
    let mut points: Vec<i64> = Vec::new();
    let mut i = 0i64;
    while i < 200 {
        points.extend([i * 11, i * 7 - 300, i * 3]);
        i += 1;
    }
    let n = points.len() / 3;
    let out = run_compute_source(
        device,
        queue,
        wgpu::util::make_spirv(spv),
        "hollow_probe",
        &[
            Binding::Storage(as_bytes_i64(&tube_words)),
            Binding::Storage(as_bytes_i64(&points)),
            Binding::Output((n * 3 * 8) as u64),
        ],
        n as u32,
    );
    let gpu: Vec<i64> = out
        .chunks_exact(8)
        .map(|b| i64::from_le_bytes(b.try_into().expect("8 bytes")))
        .collect();
    let mut hollow_differ = 0;
    let mut distance_differ = 0;
    let mut count_differ = 0;
    let mut first = String::new();
    for i in 0..n {
        let p = [
            vd_recipe::Gi::new(points[i * 3]),
            vd_recipe::Gi::new(points[i * 3 + 1]),
            vd_recipe::Gi::new(points[i * 3 + 2]),
        ];
        let hollow = vd_recipe::cell::tube_hollow_steps(&[tube], p).raw();
        let distance = tube.distance_steps(p).raw();
        if hollow != gpu[i * 3] {
            hollow_differ += 1;
            if first.is_empty() {
                first = format!(
                    " — the first at point {i}: the hollow is {hollow} on the CPU and {} on the GPU",
                    gpu[i * 3]
                );
            }
        }
        if distance != gpu[i * 3 + 1] {
            distance_differ += 1;
        }
        if gpu[i * 3 + 2] != 1 {
            count_differ += 1;
        }
    }
    println!(
        "integer_bench: PROBE the carvers — {n} points: {hollow_differ} hollows differ, \
         {distance_differ} distances differ, {count_differ} carver counts differ{first}"
    );
    if hollow_differ + distance_differ + count_differ > 0 {
        std::process::exit(1);
    }
}

// ------------------------------------------------------------- part 7: the drift hunt

/// ★ THE SEAM STAND'S EYE, in the home planet's own frame — the place the picture gate stands at
/// for its fifth picture (the gate's own log line: *landed in Planet 4030111653607004909 at
/// DVec3(4280492.264785528, −1990175.7967535623, −4280492.264785528)*). The stand sits ON a cube
/// face seam, which is why its wanted set holds the boxes that carry partner columns.
const SEAM_STAND_EYE_M: [f64; 3] = [
    4_280_492.264_785_528,
    -1_990_175.796_753_562_3,
    -4_280_492.264_785_528,
];

/// How many differing boxes the hunt names in full before it only counts them.
const HUNT_TOLD_MAX: usize = 4;

/// ★ PART 7 — THE DRIFT HUNT (2026-09-14). The seam stand draws a hole where the card builds
/// (14 959 pixels of nothing under drawn ground, §26.9), so ONE of these is true: the card's
/// KERNEL is wrong for some key, or the CLIENT's path around it is (the pooled buffers, the lanes,
/// a stale byte). This part settles that: it computes the stand's OWN wanted set, runs every box
/// through the CLIENT'S OWN GEAR — `vd_client_render::gpu_check::BoxGear`, the same pooled buffers
/// and bind groups, reused across boxes exactly as the builder reuses them — and compares each box
/// with `vd_terrain::lattice::sample_box`, cell for cell and direction for direction.
///
/// A difference here is the kernel's or the pool's. NO difference here means the drift lives in
/// the client's own path, and `VD_TERRAIN_GPU_VERIFY=1` is the instrument for that.
fn part_7_the_drift_hunt(device: &wgpu::Device, queue: &wgpu::Queue, body: &BodyDefinition) {
    let mut view = vd_client::ladder_view::LadderView::default();
    let wanted = view.wanted(body, SEAM_STAND_EYE_M);
    let lanes: usize = std::env::var("VD_BENCH_LANES")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(1);
    println!(
        "integer_bench: PART 7 — the seam stand wants {} chunks; every one through the client's \
         own gear ({lanes} in flight) against the CPU",
        wanted.keys.len()
    );
    // ★ THE FAILING KEY'S OWN CHARTER, word by word: the hunt's first question is whether the
    // card read the charter it was given, and this says what the host gave it.
    let failing = ChunkKey {
        face: Face::PosX,
        rung: 9,
        x: 69,
        y: 7,
        z: 0,
    };
    if let Some(plan) = vd_terrain::gpu::plan(body, failing) {
        println!(
            "integer_bench: PART 7 — the charter of {failing:?}: bedrock {} air {} water {} \
             box_edge {} rung {}; the words {:?}",
            plan.charter.bedrock_code.raw(),
            plan.charter.air_code.raw(),
            plan.charter.water_code.raw(),
            plan.charter.box_edge.raw(),
            plan.charter.rung.raw(),
            plan.charter_words(),
        );
        let below = plan.layers.iter().filter(|l| l.rule.raw() == 1).count();
        println!(
            "integer_bench: PART 7 — {below} of {} layers follow the BELOW rule; the tubes {} and \
             the strata rows {:?}",
            plan.layers.len(),
            plan.tubes.len(),
            plan.charter.strata,
        );
    }
    // ★ ONE BOX ON A FRESH GEAR (`VD_BENCH_ONE=1`): the failing key alone, on a gear that has
    // built nothing before it. A difference here is the KERNEL'S; agreement here with a difference
    // in the sweep is the POOL'S (a byte left by an earlier box).
    // ★ THE PREDECESSOR EXPERIMENT (`VD_BENCH_PAIR=1`): a rung-0 box FIRST, then the failing box
    // TWICE on the same gear. If the first answer drifts and the second does not, the fault is the
    // FIRST submit after the pool changed shape; if both drift, it is what the pool still holds.
    if std::env::var_os("VD_BENCH_PAIR").is_some() {
        let before = ChunkKey {
            face: vd_seed::bend::Face::NegZ,
            rung: 0,
            x: 35929,
            y: 161395,
            z: 305,
        };
        let mut gear = vd_client_render::gpu_check::BoxGear::new(device.clone(), queue.clone());
        if let Some(plan) = vd_terrain::gpu::plan(body, before) {
            let _ = gear.run(&plan).expect("the card answered");
        }
        let plan = vd_terrain::gpu::plan(body, failing).expect("the key is on the ladder");
        let cpu = vd_terrain::lattice::sample_box(body, failing).expect("the box");
        let mut attempt = 0;
        while attempt < 3 {
            let run = gear.run(&plan).expect("the card answered");
            let card = plan.box_of(&cells_of(&run.cells), &dirs_of(&run.dirs));
            let differing = cpu
                .cells
                .iter()
                .zip(card.cells.iter())
                .filter(|(a, b)| a != b)
                .count();
            println!(
                "integer_bench: PART 7 — the failing box after a rung-0 box, attempt {attempt}: \
                 {differing} cells differ"
            );
            attempt += 1;
        }
        return;
    }
    if std::env::var_os("VD_BENCH_ONE").is_some() {
        let plan = vd_terrain::gpu::plan(body, failing).expect("the key is on the ladder");
        let mut gear = vd_client_render::gpu_check::BoxGear::new(device.clone(), queue.clone());
        let run = gear.run(&plan).expect("the card answered");
        let card = plan.box_of(&cells_of(&run.cells), &dirs_of(&run.dirs));
        let cpu = vd_terrain::lattice::sample_box(body, failing).expect("the box");
        let differing = cpu
            .cells
            .iter()
            .zip(card.cells.iter())
            .filter(|(a, b)| a != b)
            .count();
        println!(
            "integer_bench: PART 7 — ONE BOX ON A FRESH GEAR at {failing:?}: {differing} cells \
             differ of {}",
            cpu.cells.len()
        );
        return;
    }
    let mut gears: Vec<vd_client_render::gpu_check::BoxGear> = (0..lanes.max(1))
        .map(|_| vd_client_render::gpu_check::BoxGear::new(device.clone(), queue.clone()))
        .collect();
    let started = Instant::now();
    let mut built = 0usize;
    let mut differing_boxes = 0usize;
    let mut told = 0usize;
    // ★ WHAT THE BOX BEFORE IT ASKED FOR: a pooled buffer that is grown and never shrunk keeps
    // the tail of a BIGGER predecessor, so a box that asks for LESS than the one before it is the
    // box that reads a stale byte. These counts name which buffer.
    let mut last: Option<(ChunkKey, usize, usize, usize, usize)> = None;
    let mut window: Vec<(ChunkKey, vd_terrain::gpu::BoxPlan)> = Vec::new();
    let mut keys = wanted.keys.iter();
    loop {
        // Fill the lanes, then collect them in the same order: the builder's own shape.
        window.clear();
        while window.len() < gears.len() {
            let Some(key) = keys.next() else { break };
            let Some(plan) = vd_terrain::gpu::plan(body, *key) else {
                continue;
            };
            gears[window.len()].submit(&plan);
            window.push((*key, plan));
        }
        if window.is_empty() {
            break;
        }
        for (lane, (key, plan)) in window.iter().enumerate() {
            let run = gears[lane].collect().expect("the card answered");
            let card = plan.box_of(&cells_of(&run.cells), &dirs_of(&run.dirs));
            let cpu =
                vd_terrain::lattice::sample_box(body, *key).expect("the box is on the ladder");
            // ★ THE THIRD ANSWER: the PLAN'S OWN CPU run — the same charter, the same layer rules,
            // the same kernel, on this host. It splits the verdict: a card that agrees with the
            // plan and disagrees with `sample_box` is INNOCENT, and the two CPU paths differ.
            let host_run = plan.run();
            let host = plan.cells_of(&host_run);
            built += 1;
            let mut cells = 0usize;
            let mut first = String::new();
            for (i, want) in cpu.cells.iter().enumerate() {
                if *want != card.cells[i] {
                    cells += 1;
                    if first.is_empty() {
                        let edge = BOX_EDGE;
                        let (a, b, c) = (i % edge, (i / edge) % edge, i / (edge * edge));
                        // ★ WHAT THE CELL'S OWN LAYER AND COLUMN SAY, on this host: the arm the
                        // kernel takes is decided by the layer's rule and by the DEPTH, so a drift
                        // names one of the two.
                        let layer = plan.layers[c];
                        let column = host_run.columns[b * edge + a];
                        let r = layer.r_steps << vd_recipe::cell::LENGTH_BITS;
                        let depth = column.h - r;
                        let depth_m =
                            depth >> (vd_recipe::cell::LENGTH_BITS + vd_recipe::cell::STEP_SHIFT);
                        println!(
                            "integer_bench: PART 7 — the cell's layer {c}: rule {} r_steps {}; \
                             the column's h {} biome {}; the depth {} = {} whole metres, and the \
                             charter's strata end at {} metres",
                            layer.rule.raw(),
                            layer.r_steps.raw(),
                            column.h.raw(),
                            column.biome.raw(),
                            depth.raw(),
                            depth_m.raw(),
                            plan.charter.strata_end_m.raw(),
                        );
                        first = format!(
                            " cell {i} (local {}, {}, {}): the CPU says substance {} gap {}, the \
                             card says substance {} gap {}",
                            a as i32 - 1,
                            b as i32 - 1,
                            c as i32 - 1,
                            want.stratum.code(),
                            want.gap,
                            card.cells[i].stratum.code(),
                            card.cells[i].gap,
                        );
                    }
                }
            }
            let mut dirs = 0usize;
            for (i, want) in cpu.dirs.iter().enumerate() {
                dirs += usize::from(*want != card.dirs[i]);
            }
            // ★ WHICH ARM DRIFTS: the differences by the layer's RULE and by the pair of
            // substances. An ABOVE or BELOW layer is a pure CHARTER READ; an EVALUATED layer is
            // the kernel's arithmetic. Which of the two drifts names the cause.
            let counts = (
                *key,
                plan.plan_charter_words().len(),
                plan.tubes.len(),
                plan.node_count,
                plan.lattice_words().len(),
            );
            if (cells > 0) & (told < HUNT_TOLD_MAX) {
                println!(
                    "integer_bench: PART 7 — this box asks for (plan charter, tubes, nodes, \
                     lattices) {:?}; the box before it {:?}",
                    (counts.1, counts.2, counts.3, counts.4),
                    last,
                );
            }
            last = Some(counts);
            if (cells > 0) & (told < HUNT_TOLD_MAX) {
                let edge = BOX_EDGE;
                let mut by_rule = [0usize; 3];
                let mut pairs: std::collections::BTreeMap<(i64, u8, u8), usize> =
                    std::collections::BTreeMap::new();
                for (i, want) in cpu.cells.iter().enumerate() {
                    if *want == card.cells[i] {
                        continue;
                    }
                    let rule = plan.layers[i / (edge * edge)].rule.raw();
                    by_rule[(rule as usize).min(2)] += 1;
                    *pairs
                        .entry((rule, want.stratum.code(), card.cells[i].stratum.code()))
                        .or_default() += 1;
                }
                println!(
                    "integer_bench: PART 7 — the differing cells by rule (evaluated, below, \
                     above): {by_rule:?}; by (rule, the CPU's substance, the card's): {pairs:?}"
                );
            }
            // WHICH PAIR DISAGREES: the card against the plan's own host run, and that host run
            // against `sample_box`.
            let mut card_vs_host = 0usize;
            let mut host_vs_sample = 0usize;
            for (i, want) in cpu.cells.iter().enumerate() {
                let host_cell = vd_terrain::chunk::cell_of_word(host[i]);
                card_vs_host += usize::from(card.cells[i] != host_cell);
                host_vs_sample += usize::from(host_cell != *want);
            }
            if (cells > 0) | (dirs > 0) {
                differing_boxes += 1;
                if told < HUNT_TOLD_MAX {
                    told += 1;
                    println!(
                        "integer_bench: PART 7 — ★ THE CARD'S BOX IS NOT THE CPU'S at {key:?}: \
                         {cells} cells and {dirs} directions differ; the card against the plan's \
                         own host run {card_vs_host}, that host run against sample_box \
                         {host_vs_sample}; the first layer's rule {} r_steps {};{first}",
                        plan.layers[0].rule.raw(),
                        plan.layers[0].r_steps.raw(),
                    );
                }
            }
        }
    }
    let seconds = started.elapsed().as_secs_f64();
    println!(
        "integer_bench: PART 7 — {built} boxes through the card's own gear in {seconds:.1} s; \
         {differing_boxes} of them differ from the CPU's"
    );
    assert_eq!(
        differing_boxes, 0,
        "PART 7: the card's box is not the CPU's on the seam stand's own wanted set"
    );
}

/// The cells of a readback, as the client's geometry stage decodes them.
fn cells_of(bytes: &[u8]) -> Vec<u32> {
    bytes
        .chunks_exact(4)
        .map(|b| u32::from_le_bytes(b.try_into().unwrap_or([0; 4])))
        .collect()
}

/// The column directions of a readback, as the client's geometry stage decodes them.
fn dirs_of(bytes: &[u8]) -> Vec<[vd_recipe::Gi; 3]> {
    bytes
        .chunks_exact(24)
        .map(|row| {
            let word = |k: usize| {
                vd_recipe::Gi::new(i64::from_le_bytes(
                    row[k * 8..k * 8 + 8].try_into().unwrap_or([0; 8]),
                ))
            };
            [word(0), word(1), word(2)]
        })
        .collect()
}

// ----------------------------------------------------------------- part 5: the cell field

/// ★ THE CELL FIELD ON THE GPU (step G1 of `slice_08_integer_recipe_design.md` §2): every cell of a
/// box — the chunk's 62³ plus its one-cell halo — computed on the card by the recipe's own
/// `cell_word`, through the shell's `cell_field` entry point, and compared BYTE FOR BYTE with the
/// CPU's `sample_box`. The eight golden chunks the world identity folds first, then the bench's
/// square of 1 024 rung-0 chunks on face +X, with the cost of each host beside it.
///
/// What the CPU still does for each box (G1 only): the column pass, the cavern lattice's node
/// values, the carvers that reach the box and the lattice's topology — `vd_terrain::gpu::plan`.
/// Its cost is reported on its own, because it is the next thing to move (step G2).
fn part_5_the_cell_field(device: &wgpu::Device, queue: &wgpu::Queue, body: &BodyDefinition) {
    let Some(path) = std::env::var_os(SPV_ENV) else {
        println!("integer_bench: PART 5 SKIPPED — set {SPV_ENV} to the SPIR-V cargo-gpu built");
        return;
    };
    let spv = std::fs::read(&path).expect("the SPIR-V file reads");
    // THE KERNEL PROBES first: the two kernels the cell field newly runs, on their own. A probe
    // names the fault in one function where a box of a quarter of a million cells names only that
    // something differs (both found a real one on 2026-09-13: the root's loop and the carvers').
    probe_isqrt(device, queue, &spv);
    // ★ THE WHOLE CHAIN (step G2-A): the column pass, the node pass and the cell field, three
    // compute passes in one command encoder, one submit, one readback — the CLIENT's own chain,
    // built from the module its build script compiled, so the bench measures the shipped path.
    let pipeline = BoxPipelines::new(device);

    // THE EIGHT GOLDEN CHUNKS: the keys the world identity folds, at the rungs they name.
    let golden: Vec<ChunkKey> = GOLDEN_SELF_CHECK_KEYS
        .iter()
        .map(|entry| self_check_key(body, *entry))
        .collect();
    let (differing, first, _, golden_with_carvers) =
        compare_boxes(device, queue, &pipeline, body, &golden);
    println!(
        "integer_bench: PART 5 THE GOLDEN CHUNKS — {} boxes of {} cells through the recipe's \
         cell_field: {differing} cells differ between the CPU and the GPU{} ({golden_with_carvers} \
         of the boxes hold a tube carver)",
        golden.len(),
        BOX_CELLS,
        first
    );
    if differing > 0 {
        std::process::exit(1);
    }

    // THE SQUARE: the same 1 024 chunks parts 1 to 4 read the columns of.
    let square: Vec<ChunkKey> = (FIRST_Y..FIRST_Y + CHUNKS_ACROSS)
        .flat_map(|y| (FIRST_X..FIRST_X + CHUNKS_ACROSS).map(move |x| (x, y)))
        .map(|(x, y)| ChunkKey {
            face: FACE,
            rung: RUNG,
            x,
            y,
            z: surface_chunk_z(body, FACE, RUNG, x, y),
        })
        .collect();
    let (differing, first, gpu, with_carvers) =
        compare_boxes(device, queue, &pipeline, body, &square);
    println!(
        "integer_bench: PART 5 THE SQUARE — {} boxes of {BOX_CELLS} cells ({} cells): {differing} \
         differ between the CPU and the GPU{}; {with_carvers} of the boxes hold a tube carver",
        square.len(),
        square.len() * BOX_CELLS,
        first
    );
    if differing > 0 {
        std::process::exit(1);
    }
    // ★ THE CARVER KERNEL IS ACTUALLY WALKED. The eight golden boxes hold no tube carver, so
    // without this the whole carver path — the segment distance, the squared guard, the hollow's
    // accumulator — could leave the GPU comparison silently, and the fault this very step found
    // would come back unnoticed. A change to `tubes_reaching` that emptied every list would go red
    // here instead of quietly narrowing the gate.
    if with_carvers == 0 {
        println!(
            "integer_bench: PART 5 — NO BOX OF THE SQUARE HOLDS A TUBE CARVER, so the carver \
             kernel was never compared on the GPU — STOP"
        );
        std::process::exit(1);
    }

    // ★ PART 6's OWN SET: THE BOXES THAT CROSS A SEAM. Neither the golden chunks nor the square
    // stands at a face's edge (MEASURED: every one of the 1 032 lays down ONE lattice), so the
    // column pass's two hardest cases — a column that belongs to a PARTNER face, and a CORNER
    // PHANTOM that belongs to none — were never compared on the card. These are the last chunk of
    // each face at rung 0 and at the coarsest rung, where a box holds all three.
    let last0 = (body.ladder().cells_per_edge(RUNG) as i32 - 1) / CHUNK_EDGE as i32;
    let top = body.ladder().rungs - 1;
    let last_top = (body.ladder().cells_per_edge(top) as i32 - 1) / CHUNK_EDGE as i32;
    let mut seams: Vec<ChunkKey> = Vec::new();
    for face in Face::ALL {
        for (rung, last) in [(RUNG, last0), (top, last_top)] {
            for (x, y) in [(last, last), (0, last), (last, 0), (0, 0)] {
                let key = ChunkKey {
                    face,
                    rung,
                    x,
                    y,
                    z: surface_chunk_z(body, face, rung, x, y),
                };
                if in_ladder(body, key) {
                    seams.push(key);
                }
            }
        }
    }
    let crossing = seams
        .iter()
        .filter(|k| {
            vd_terrain::gpu::plan(body, **k).is_some_and(|p| {
                p.sites.iter().any(|s| s.face != k.face.index())
                    && p.sites
                        .iter()
                        .any(|s| s.face == vd_terrain::lattice::CORNER_FACE)
            })
        })
        .count();
    let (differing, first, _, _) = compare_boxes(device, queue, &pipeline, body, &seams);
    println!(
        "integer_bench: PART 6 THE SEAMS — {} boxes at the faces' edges ({} of them hold BOTH a \
         partner face's columns and a corner phantom): {differing} cells or directions differ \
         between the CPU and the GPU{}",
        seams.len(),
        crossing,
        first
    );
    if differing > 0 {
        std::process::exit(1);
    }
    // ★ THE SEAM PATH IS ACTUALLY WALKED, for the same reason the carvers are: a set that never
    // crossed would pass every assertion above and leave the partner lattice and the phantom's
    // normalise uncompared on the card.
    // EVERY box of the set must cross, not merely one: the set IS the faces' corner chunks, so a
    // box of it that held no partner column or no phantom would mean the set is no longer the set
    // this part names, and the docs' claim of 48 of 48 would be quietly false.
    if crossing != seams.len() {
        println!(
            "integer_bench: PART 6 — only {crossing} of {} boxes cross a seam, so the column \
             pass's partner-face and corner-phantom arms are not compared on the GPU by the whole \
             set — STOP",
            seams.len()
        );
        std::process::exit(1);
    }

    // The GPU a second time: the first pass paid the pipeline's own compilation.
    let started = Instant::now();
    let mut second_plans = 0.0_f64;
    for key in &square {
        let at = Instant::now();
        let plan = vd_terrain::gpu::plan(body, *key).expect("the key is on the ladder");
        second_plans += at.elapsed().as_secs_f64();
        let _ = box_pass(device, queue, &pipeline, &plan);
    }
    let gpu_second = started.elapsed().as_secs_f64();

    // ★ THE READBACK'S OWN SHARE: the same three passes, submitted and waited for, with nothing
    // copied home. The difference against the line above is the megabyte crossing the bus — the
    // cost step G2 removes by extracting the mesh on the card.
    let started = Instant::now();
    for key in &square {
        let plan = vd_terrain::gpu::plan(body, *key).expect("the key is on the ladder");
        vd_client_render::gpu_check::dispatch_box_compute_only(device, queue, &pipeline, &plan);
    }
    let gpu_no_readback = started.elapsed().as_secs_f64();

    // ★ THE CPU'S OWN BOX, on one core and on the terrain's share of the cores (ruling F6) — and
    // MEASURED THE WAY THE GPU LEGS ARE. Three things the first version of this part did not do,
    // each of which flattered or punished one leg against another: the one-core leg ran 64 boxes
    // and multiplied by sixteen while the GPU legs ran all 1 024; the workers' leg DREW THE BODY
    // FROM ITS SEED inside the timed scope, once per worker; and the body each leg read was a
    // different object. Now every leg runs the same 1 024 keys, and every body is drawn before any
    // clock starts.
    let cores = std::thread::available_parallelism().map_or(1, std::num::NonZero::get);
    let workers = vd_client_render::terrain::worker_share(cores);
    let bodies: Vec<BodyDefinition> = (0..workers).map(|_| home_planet()).collect();
    let started = Instant::now();
    for key in &square {
        let _ = vd_terrain::lattice::sample_box(body, *key).expect("the box");
    }
    let cpu_one_core = started.elapsed().as_secs_f64();
    let started = Instant::now();
    std::thread::scope(|scope| {
        for (w, own) in bodies.iter().enumerate() {
            let keys = &square;
            scope.spawn(move || {
                let mut i = w;
                while i < keys.len() {
                    let _ = vd_terrain::lattice::sample_box(own, keys[i]).expect("the box");
                    i += workers;
                }
            });
        }
    });
    let cpu_workers = started.elapsed().as_secs_f64();
    println!(
        "integer_bench: PART 5 THE COST — {} boxes: the GPU {:.0} ms on the first pass (with the \
         pipeline's compilation) and {:.0} ms on the second, both with the plans, the upload and \
         the readback; of the second pass the PLANS on the CPU are {:.0} ms (the sites, the \
         carvers, the lattice extents and the layers — TOPOLOGY only since step G2-A) and the \
         card's own share is {:.0} ms. WITHOUT THE READBACK the same three passes take {:.0} ms, so \
         the bus carries {:.0} ms of it. The CPU's own sample_box: {:.0} ms on one core, {:.0} ms \
         on {workers} workers (the terrain's share of this machine).",
        square.len(),
        gpu * 1.0e3,
        gpu_second * 1.0e3,
        second_plans * 1.0e3,
        (gpu_second - second_plans) * 1.0e3,
        gpu_no_readback * 1.0e3,
        (gpu_second - gpu_no_readback) * 1.0e3,
        cpu_one_core * 1.0e3,
        cpu_workers * 1.0e3
    );
}

/// Every box of `keys` on the GPU against the CPU's own `sample_box`, cell for cell: the count of
/// differing cells, a line naming the first, the GPU's wall time, and how many of the boxes hold a
/// REAL tube carver (a plan is never given an empty carver buffer — a carver of no radius stands
/// in — so the count asks for a radius, not for a length).
fn compare_boxes(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    pipeline: &BoxPipelines,
    body: &BodyDefinition,
    keys: &[ChunkKey],
) -> (usize, String, f64, usize) {
    let mut differing = 0usize;
    let mut first = String::new();
    let mut seconds = 0.0_f64;
    let mut with_carvers = 0usize;
    let mut differing_dirs = 0usize;
    for key in keys {
        let started = Instant::now();
        let plan = vd_terrain::gpu::plan(body, *key).expect("the key is on the ladder");
        let (gpu, gpu_dirs) = box_pass(device, queue, pipeline, &plan);
        seconds += started.elapsed().as_secs_f64();
        with_carvers += usize::from(
            plan.tubes
                .iter()
                .any(|t| t.radius_steps > vd_recipe::Gi::ZERO),
        );
        let want = vd_terrain::lattice::sample_box(body, *key).expect("the box");
        // ★ THE COLUMN PASS ITSELF, not only its consequence: the directions the card wrote against
        // the ones the host's own column pass writes. A cell word folds the direction through the
        // whole cell kernel, so a direction that differs in its last bit could still pack the same
        // byte; this compares the words.
        for (i, d) in want.dirs.iter().enumerate() {
            if *d != gpu_dirs[i] {
                if differing_dirs == 0 {
                    println!(
                        "integer_bench: PART 6 — the first differing direction at {key:?} column \
                         {i} (site {:?}): the CPU says [{}, {}, {}], the GPU says [{}, {}, {}]",
                        want.sites[i],
                        d[0].raw(),
                        d[1].raw(),
                        d[2].raw(),
                        gpu_dirs[i][0].raw(),
                        gpu_dirs[i][1].raw(),
                        gpu_dirs[i][2].raw(),
                    );
                }
                differing_dirs += 1;
            }
        }
        for (i, cell) in want.cells.iter().enumerate() {
            let cpu = vd_recipe::cell::pack(
                vd_recipe::Gi::new(i64::from(cell.stratum.code())),
                vd_recipe::Gi::new(i64::from(cell.gap)),
            );
            if cpu != gpu[i] {
                differing += 1;
                if first.is_empty() {
                    first = format!(
                        " — the first at {key:?} cell {i}: the CPU says substance {} gap {}, the \
                         GPU says substance {} gap {}{}",
                        cell.stratum.code(),
                        cell.gap,
                        vd_recipe::cell::stratum_of_word(gpu[i]),
                        vd_recipe::cell::gap_of_word(gpu[i]),
                        why(&plan, i, &gpu)
                    );
                }
            }
        }
    }
    if differing_dirs > 0 {
        println!(
            "integer_bench: PART 6 — {differing_dirs} COLUMN DIRECTIONS differ between the card's \
             column pass and the host's"
        );
        differing += differing_dirs;
    }
    (differing, first, seconds, with_carvers)
}

/// WHY one cell differs: what the plan states about it and what the two carvers open there, on this
/// host. A diagnosis, never a gate.
fn why(plan: &vd_terrain::gpu::BoxPlan, index: usize, gpu: &[u32]) -> String {
    let edge = BOX_EDGE;
    let (a, b, c) = (index % edge, (index / edge) % edge, index / (edge * edge));
    let run = plan.run();
    let layer = &plan.layers[c];
    let column = &run.columns[b * edge + a];
    let value = vd_recipe::cell::cavern_at(column, layer, &run.nodes);
    let point = vd_recipe::cell::point_at(column.dir, layer.r_steps);
    let cavern = plan.charter.cavern_hollow_steps(value);
    let tube = vd_recipe::cell::tube_hollow_steps(&plan.tubes, point);
    let no_tubes = vd_recipe::cell::cell_word(
        &plan.charter,
        &vd_recipe::cell::CellAt {
            dir: column.dir,
            h: column.h,
            biome: column.biome,
            r_steps: layer.r_steps,
        },
        value,
        &[],
    );
    let no_cavern = vd_recipe::cell::cell_word(
        &plan.charter,
        &vd_recipe::cell::CellAt {
            dir: column.dir,
            h: column.h,
            biome: column.biome,
            r_steps: layer.r_steps,
        },
        vd_recipe::Gi::ZERO,
        &plan.tubes,
    );
    format!(
        " [at (a {a}, b {b}, c {c}); the layer: rule {} r_steps {} node {} weight {}; the column: \
         h {} biome {} has {} base {} na {} nb {} d0 {} d1 {} wa {} wb {}; the cavern value {} \
         opens {} steps, the {} carvers open {} steps; without the carvers the CPU word is {} \
         (tubes off) and {} (cavern off) against {}; the plan holds {} nodes]",
        layer.rule.raw(),
        layer.r_steps.raw(),
        layer.node.raw(),
        layer.weight.raw(),
        column.h.raw(),
        column.biome.raw(),
        column.has.raw(),
        column.base.raw(),
        column.na.raw(),
        column.nb.raw(),
        column.d0.raw(),
        column.d1.raw(),
        column.wa.raw(),
        column.wb.raw(),
        value.raw(),
        cavern.raw(),
        plan.tubes.len(),
        tube.raw(),
        no_tubes,
        no_cavern,
        gpu[index],
        run.nodes.len(),
    )
}

/// ★ THE WHOLE CHAIN OF ONE BOX (step G2-A) is `vd_client_render::gpu_check::dispatch_box` — the
/// SAME call the client's own self-check runs and the client's builder will run, so the bench
/// measures the shipped path and not an instrument beside it.
use vd_client_render::gpu_check::{BoxChain as BoxPipelines, dispatch_box as box_pass};

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
