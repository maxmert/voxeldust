//! ★ THE GPU'S SELF-CHECK (ruling F8 decision 3): the client MEASURES ITS OWN GPU before trusting
//! it with the world's shape. At start it runs the recipe's own kernel — the one source compiled
//! to SPIR-V at build time (`build.rs`, cargo-gpu) — on the columns of the eight golden chunks the
//! world identity is measured on, and compares every word with the CPU's call of the same
//! function. Equal: the GPU path is trusted. Different, or no 64-bit integers on this GPU: the
//! CPU path of the same source runs, and the client says so in its log — a measured refusal,
//! never a silent one. SL10's "no drift is a MEASUREMENT on every target": a GPU is a target the
//! build machine never sees, so the target measures itself.
//!
//! TWO kernels run (step G1, 2026-09-13): the octave sum over a column — the height field — and
//! THE CELL FIELD, every cell of the eight golden boxes with its substance and its gap. The cell
//! field is the one the client's chunk builder uses, so it is the one that must agree; the column
//! kernel stays because it is cheap and it names a narrower fault.
//!
//! **Example.** A laptop whose driver miscompiles a 64-bit shift builds the golden boxes wrong at
//! start: the check counts the differing cells, the log names them, the flag stays false, and the
//! player gets the CPU path — never a hill the server disagrees with.

use std::time::Instant;

use bevy::prelude::*;
use bevy::render::renderer::{RenderDevice, RenderQueue};
use vd_recipe::Gi;
use vd_terrain::BodyDefinition;
use vd_terrain::chunk::CHUNK_EDGE;
use vd_terrain::digest::{GOLDEN_SELF_CHECK_KEYS, self_check_key};
use vd_terrain::gpu::BoxPlan;
use vd_terrain::lattice::{site_dir, site_of};
use wgpu::util::DeviceExt;

/// The SPIR-V the build script compiled from the one source.
const RECIPE_SPV: &[u8] = include_bytes!(env!("VD_RECIPE_GPU_SPV"));
/// The entry point of the column kernel in the shell.
const ENTRY: &str = "relief_columns";
/// The entry point of the cell-field kernel in the shell.
const CELL_ENTRY: &str = "cell_field";
/// The column kernel's workgroup width.
const WORKGROUP: u32 = 256;
/// The cell-field kernel's workgroup width.
const CELL_WORKGROUP: u32 = 64;

/// The verdict of the self-check, a resource every draw may read.
#[derive(Resource, Clone, Copy, Debug, PartialEq, Eq)]
pub struct GpuRecipeCheck {
    /// Whether this GPU offers 64-bit integers at all.
    pub int64: bool,
    /// The columns compared.
    pub columns: u32,
    /// The words that differed between the GPU and the CPU.
    pub differing: u32,
    /// The cells of the eight golden boxes compared.
    pub cells: u32,
    /// The cells that differed between the GPU and the CPU.
    pub differing_cells: u32,
    /// The wall time of the whole check, in microseconds.
    pub micros: u32,
}

impl GpuRecipeCheck {
    /// Whether the GPU path is trusted: 64-bit integers present, both kernels run, and not one
    /// word and not one cell differed.
    #[must_use]
    pub const fn trusted(self) -> bool {
        self.int64
            & (self.differing == 0)
            & (self.columns > 0)
            & (self.differing_cells == 0)
            & (self.cells > 0)
    }
}

/// The columns of the eight golden chunks: their 40-bit directions, and the rung each is at. The
/// keys resolve through the digest's own rule (the top-rung sentinel, the wrap into the face), so
/// the check reads exactly the chunks the world identity folds.
fn golden_columns(body: &BodyDefinition) -> Vec<(u8, [Gi; 3])> {
    let mut out = Vec::new();
    for entry in GOLDEN_SELF_CHECK_KEYS {
        let key = self_check_key(body, entry);
        for b in 0..CHUNK_EDGE as i32 {
            for a in 0..CHUNK_EDGE as i32 {
                let site = site_of(body, key, a, b);
                out.push((key.rung, site_dir(body, key, site)));
            }
        }
    }
    out
}

/// Run the check on `device` for `body`: the golden columns through the shell's kernel against the
/// CPU's `vd_recipe::height::relief`, one dispatch per rung (the octaves differ by rung).
#[must_use]
pub fn run(device: &wgpu::Device, queue: &wgpu::Queue, body: &BodyDefinition) -> GpuRecipeCheck {
    let started = Instant::now();
    let int64 = device.features().contains(wgpu::Features::SHADER_INT64);
    if !int64 {
        return GpuRecipeCheck {
            int64: false,
            columns: 0,
            differing: 0,
            cells: 0,
            differing_cells: 0,
            micros: started.elapsed().as_micros() as u32,
        };
    }
    let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("recipe self-check"),
        source: wgpu::util::make_spirv(RECIPE_SPV),
    });
    let pipeline = compute_pipeline(device, &module, ENTRY);
    let cell_pipeline = compute_pipeline(device, &module, CELL_ENTRY);
    let all = golden_columns(body);
    let mut columns = 0u32;
    let mut differing = 0u32;
    let mut rung = 0u8;
    while rung <= vd_seed::ladder::RUNG_MAX {
        let dirs: Vec<i64> = all
            .iter()
            .filter(|(r, _)| *r == rung)
            .flat_map(|(_, d)| [d[0].raw(), d[1].raw(), d[2].raw()])
            .collect();
        let n = dirs.len() / 3;
        if n > 0 {
            let octaves = body.octaves_at(rung);
            let cpu: Vec<i64> = (0..n)
                .map(|i| {
                    let d = [
                        Gi::new(dirs[i * 3]),
                        Gi::new(dirs[i * 3 + 1]),
                        Gi::new(dirs[i * 3 + 2]),
                    ];
                    vd_recipe::height::relief(octaves, d).raw()
                })
                .collect();
            let gpu = dispatch(device, queue, &pipeline, &dirs, octaves, n);
            columns += n as u32;
            // ★ A SHORT READBACK IS EVERY WORD DIFFERING, never agreement. Zipping two lists stops
            // at the shorter one, so a card that wrote half a buffer — or none of it — would have
            // read as a card to be trusted. The length is part of the answer.
            differing += if gpu.len() == cpu.len() {
                cpu.iter().zip(gpu.iter()).filter(|(a, b)| a != b).count() as u32
            } else {
                n as u32
            };
        }
        rung += 1;
    }
    // ★ THE CELL FIELD: the eight golden boxes, every cell's substance and gap, on the card against
    // the same kernels on this machine's own CPU (`BoxPlan::cells`, which the generator's own test
    // measures against the cell pass byte for byte).
    let mut cells = 0u32;
    let mut differing_cells = 0u32;
    for entry in GOLDEN_SELF_CHECK_KEYS {
        let key = self_check_key(body, entry);
        let Some(plan) = vd_terrain::gpu::plan(body, key) else {
            continue;
        };
        let cpu = plan.cells();
        let gpu = dispatch_cells(device, queue, &cell_pipeline, &plan);
        cells += cpu.len() as u32;
        // A short readback is every cell differing, for the same reason as the columns above.
        differing_cells += if gpu.len() == cpu.len() {
            cpu.iter().zip(gpu.iter()).filter(|(a, b)| a != b).count() as u32
        } else {
            cpu.len() as u32
        };
    }
    GpuRecipeCheck {
        int64: true,
        columns,
        differing,
        cells,
        differing_cells,
        micros: started.elapsed().as_micros() as u32,
    }
}

/// One compute pipeline of the shell's module, by its entry point.
fn compute_pipeline(
    device: &wgpu::Device,
    module: &wgpu::ShaderModule,
    entry: &str,
) -> wgpu::ComputePipeline {
    device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("recipe self-check"),
        layout: None,
        module,
        entry_point: Some(entry),
        compilation_options: wgpu::PipelineCompilationOptions::default(),
        cache: None,
    })
}

/// ★ ONE BOX ON THE CARD: the plan's six buffers up, one word per cell back. The client's chunk
/// builder runs this same call; the check is that call on the eight boxes the world identity folds.
#[must_use]
pub fn dispatch_cells(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    pipeline: &wgpu::ComputePipeline,
    plan: &BoxPlan,
) -> Vec<u32> {
    let edge = plan.charter.box_edge.raw() as u32;
    let out_size = u64::from(edge) * u64::from(edge) * u64::from(edge) * 4;
    let charter = words_buffer(device, &plan.charter_words());
    let layers = words_buffer(device, &plan.layer_words());
    let columns = words_buffer(device, &plan.column_words());
    let nodes = words_buffer(device, &plan.node_words());
    let tubes = words_buffer(device, &plan.tube_words());
    let out = device.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size: out_size,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    let staging = device.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size: out_size,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let buffers = [&charter, &layers, &columns, &nodes, &tubes, &out];
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
    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor::default());
        pass.set_pipeline(pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        pass.dispatch_workgroups(edge.div_ceil(CELL_WORKGROUP), edge, edge);
    }
    encoder.copy_buffer_to_buffer(&out, 0, &staging, 0, out_size);
    queue.submit([encoder.finish()]);
    let bytes = read_back(device, &staging, out_size);
    bytes
        .chunks_exact(4)
        .map(|b| u32::from_le_bytes(b.try_into().expect("4 bytes")))
        .collect()
}

/// A storage buffer of words.
fn words_buffer(device: &wgpu::Device, words: &[i64]) -> wgpu::Buffer {
    device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: None,
        contents: &words
            .iter()
            .flat_map(|x| x.to_le_bytes())
            .collect::<Vec<u8>>(),
        usage: wgpu::BufferUsages::STORAGE,
    })
}

/// The bytes of a mapped staging buffer, once the card is done.
fn read_back(device: &wgpu::Device, staging: &wgpu::Buffer, size: u64) -> Vec<u8> {
    let slice = staging.slice(..size);
    let (tx, rx) = std::sync::mpsc::channel();
    slice.map_async(wgpu::MapMode::Read, move |r| {
        let _ = tx.send(r);
    });
    device
        .poll(wgpu::PollType::wait_indefinitely())
        .expect("the device polls");
    rx.recv().expect("a map result").expect("the map succeeds");
    let bytes = slice.get_mapped_range().to_vec();
    staging.unmap();
    bytes
}

/// One dispatch of the column kernel: the directions and the octaves up, the reliefs back.
fn dispatch(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    pipeline: &wgpu::ComputePipeline,
    dirs: &[i64],
    octaves: &[vd_recipe::height::Octave],
    n: usize,
) -> Vec<i64> {
    let dirs_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: None,
        contents: &dirs
            .iter()
            .flat_map(|x| x.to_le_bytes())
            .collect::<Vec<u8>>(),
        usage: wgpu::BufferUsages::STORAGE,
    });
    let mut octave_bytes = Vec::with_capacity(octaves.len() * 32);
    for o in octaves {
        octave_bytes.extend(o.seed.to_le_bytes());
        octave_bytes.extend(o.frequency_int.raw().to_le_bytes());
        octave_bytes.extend(o.frequency_frac.raw().to_le_bytes());
        octave_bytes.extend(o.amplitude.raw().to_le_bytes());
    }
    let octaves_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: None,
        contents: &octave_bytes,
        usage: wgpu::BufferUsages::STORAGE,
    });
    let out_size = (n * 8) as u64;
    let out_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size: out_size,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    let staging = device.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size: out_size,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: None,
        layout: &pipeline.get_bind_group_layout(0),
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: dirs_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: octaves_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: out_buf.as_entire_binding(),
            },
        ],
    });
    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor::default());
        pass.set_pipeline(pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        pass.dispatch_workgroups((n as u32).div_ceil(WORKGROUP), 1, 1);
    }
    encoder.copy_buffer_to_buffer(&out_buf, 0, &staging, 0, out_size);
    queue.submit([encoder.finish()]);
    read_back(device, &staging, out_size)
        .chunks_exact(8)
        .map(|b| i64::from_le_bytes(b.try_into().expect("8 bytes")))
        .collect()
}

/// The startup system: the check on the home planet (the body the world identity is measured
/// on), the verdict logged and kept as a resource.
pub fn gpu_recipe_self_check(
    mut commands: Commands,
    render_device: Res<RenderDevice>,
    render_queue: Res<RenderQueue>,
) {
    let body = vd_terrain::home::home_planet();
    let device: &wgpu::Device = render_device.wgpu_device();
    let queue: &wgpu::Queue = &render_queue.0;
    let check = run(device, queue, &body);
    if check.trusted() {
        tracing::info!(
            columns = check.columns,
            cells = check.cells,
            micros = check.micros,
            "GPU RECIPE SELF-CHECK PASSED: the GPU's columns and cells equal the CPU's, word for \
             word"
        );
    } else if check.int64 {
        tracing::warn!(
            columns = check.columns,
            differing = check.differing,
            cells = check.cells,
            differing_cells = check.differing_cells,
            "GPU RECIPE SELF-CHECK FAILED: the GPU differs from the CPU — the CPU path of the same \
             source runs on this machine"
        );
    } else {
        tracing::warn!(
            "GPU RECIPE SELF-CHECK SKIPPED: this GPU offers no 64-bit integers — the CPU path of \
             the same source runs on this machine"
        );
    }
    commands.insert_resource(check);
}
