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
//! ★ THE WHOLE CHAIN IS CHECKED SINCE STEP G2-A: the box now goes through [`dispatch_box`], which
//! runs the COLUMN pass and the NODE pass on the card before the cell field, and the check compares
//! the columns' own DIRECTIONS as well as every cell — a direction that differs in its last bit can
//! still pack the same cell byte, and the extractor places every vertex along it.
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
/// The entry point of the COLUMN pass (step G2-A).
const COLUMN_ENTRY: &str = "column_pass";
/// The entry point of the NODE pass (step G2-A).
const NODE_ENTRY: &str = "node_pass";
/// The column kernel's workgroup width.
const WORKGROUP: u32 = 256;
/// The cell-field kernel's workgroup width.
const CELL_WORKGROUP: u32 = 64;
/// The column pass's workgroup width.
const COLUMN_WORKGROUP: u32 = 64;
/// The node pass's workgroup width along a lattice's first face axis.
const NODE_WORKGROUP: u32 = 32;

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
    let chain = BoxChain::new(device);
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
        let run = plan.run();
        let cpu = plan.cells_of(&run);
        let cpu_dirs = BoxPlan::dirs_of(&run);
        let (gpu, gpu_dirs) = dispatch_box(device, queue, &chain, &plan);
        cells += cpu.len() as u32;
        // A short readback is every cell differing, for the same reason as the columns above.
        differing_cells += if gpu.len() == cpu.len() {
            cpu.iter().zip(gpu.iter()).filter(|(a, b)| a != b).count() as u32
        } else {
            cpu.len() as u32
        };
        // ★ THE COLUMN PASS ITSELF (step G2-A): the directions the card wrote against the host's.
        // The cells fold a direction through the whole cell kernel, so one that differs in its last
        // bit could still pack the same byte; these are the words themselves, and the extractor
        // places every vertex along them.
        differing_cells += if gpu_dirs.len() == cpu_dirs.len() {
            cpu_dirs
                .iter()
                .zip(gpu_dirs.iter())
                .filter(|(a, b)| a != b)
                .count() as u32
        } else {
            cpu_dirs.len() as u32
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

/// ★ THE BOX CHAIN — the three pipelines of the shell's module, in the order the card runs them
/// (step G2-A). One module, three entry points; a host builds this once and runs every box through
/// it.
pub struct BoxChain {
    /// The column pass: a column's direction, its surface and its biome.
    pub column: wgpu::ComputePipeline,
    /// The node pass: the cavern field at one lattice node.
    pub node: wgpu::ComputePipeline,
    /// The cell field: one word per cell of the box.
    pub cell: wgpu::ComputePipeline,
}

impl BoxChain {
    /// The chain built from the recipe's own SPIR-V — the module the build script compiled.
    #[must_use]
    pub fn new(device: &wgpu::Device) -> BoxChain {
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("the recipe's box chain"),
            source: wgpu::util::make_spirv(RECIPE_SPV),
        });
        BoxChain {
            column: compute_pipeline(device, &module, COLUMN_ENTRY),
            node: compute_pipeline(device, &module, NODE_ENTRY),
            cell: compute_pipeline(device, &module, CELL_ENTRY),
        }
    }
}

/// ★ ONE BOX ON THE CARD, WHOLE (step G2-A): the column pass, the node pass and the cell field, in
/// three compute passes of ONE command encoder, one submit, one wait. What goes up is TOPOLOGY —
/// the two charters, one site per column, the lattice extents, the layer rows, the slice table and
/// the carvers. What comes back is one word per cell and the columns' own directions, which the
/// extractor places its vertices along.
///
/// Three passes rather than three dispatches of one: the column buffer and the node buffer are
/// WRITTEN by the first two and READ by the third, and one compute pass may not hold both usages of
/// one buffer.
///
/// **Example.** Chunk (face 2, rung 0, 19, 1, 4): about 80 kB up, then 4 096 octave sums, 6 859
/// cavern nodes and 262 144 cells on the card, then 1.1 MB back — the same 262 144 words the
/// shard's CPU writes for the same key.
#[must_use]
pub fn dispatch_box(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    chain: &BoxChain,
    plan: &BoxPlan,
) -> (Vec<u32>, Vec<[Gi; 3]>) {
    box_chain(device, queue, chain, plan, true)
}

/// ★ THE CARD'S SHARE WITHOUT THE READBACK — the same three passes, submitted and waited for, with
/// nothing copied home. The instrument that says how much of a box's cost is the megabyte crossing
/// the bus, which is exactly what step G2 (the extraction on the card) removes.
pub fn dispatch_box_compute_only(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    chain: &BoxChain,
    plan: &BoxPlan,
) {
    let _ = box_chain(device, queue, chain, plan, false);
}

fn box_chain(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    chain: &BoxChain,
    plan: &BoxPlan,
    read: bool,
) -> (Vec<u32>, Vec<[Gi; 3]>) {
    let edge = plan.charter.box_edge.raw() as u32;
    let columns_n = u64::from(edge) * u64::from(edge);
    let cells_size = columns_n * u64::from(edge) * 4;
    let dirs_size = columns_n * 3 * 8;
    // What goes up.
    let plan_charter = words_buffer(device, &plan.plan_charter_words());
    let sites = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: None,
        contents: &plan
            .site_words()
            .iter()
            .flat_map(|x| x.to_le_bytes())
            .collect::<Vec<u8>>(),
        usage: wgpu::BufferUsages::STORAGE,
    });
    let lattices = words_buffer(device, &plan.lattice_words());
    let node_z = words_buffer(device, &plan.node_z_words());
    let radii = words_buffer(device, &plan.node_radius_words());
    let charter = words_buffer(device, &plan.charter_words());
    let layers = words_buffer(device, &plan.layer_words());
    let tubes = words_buffer(device, &plan.tube_words());
    // What the card writes for itself and never ships back.
    let scratch = |size: u64| {
        device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size,
            usage: wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        })
    };
    let columns = scratch(columns_n * vd_terrain::gpu::COLUMN_WORDS as u64 * 8);
    let nodes = scratch(plan.node_count as u64 * 8);
    let (dirs, dirs_staging) = read_pair(device, dirs_size);
    let (out, out_staging) = read_pair(device, cells_size);

    let group = |pipeline: &wgpu::ComputePipeline, buffers: &[&wgpu::Buffer]| {
        let entries: Vec<wgpu::BindGroupEntry> = buffers
            .iter()
            .enumerate()
            .map(|(i, b)| wgpu::BindGroupEntry {
                binding: i as u32,
                resource: b.as_entire_binding(),
            })
            .collect();
        device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &pipeline.get_bind_group_layout(0),
            entries: &entries,
        })
    };
    let column_group = group(
        &chain.column,
        &[&plan_charter, &sites, &lattices, &columns, &dirs],
    );
    let node_group = group(
        &chain.node,
        &[&plan_charter, &lattices, &node_z, &radii, &nodes],
    );
    let cell_group = group(
        &chain.cell,
        &[&charter, &layers, &columns, &nodes, &tubes, &out],
    );
    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor::default());
        pass.set_pipeline(&chain.column);
        pass.set_bind_group(0, &column_group, &[]);
        pass.dispatch_workgroups((columns_n as u32).div_ceil(COLUMN_WORKGROUP), 1, 1);
    }
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor::default());
        pass.set_pipeline(&chain.node);
        pass.set_bind_group(0, &node_group, &[]);
        pass.dispatch_workgroups(
            plan.node_extent[0].div_ceil(NODE_WORKGROUP),
            plan.node_extent[1],
            plan.node_z.len() as u32,
        );
    }
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor::default());
        pass.set_pipeline(&chain.cell);
        pass.set_bind_group(0, &cell_group, &[]);
        pass.dispatch_workgroups(edge.div_ceil(CELL_WORKGROUP), edge, edge);
    }
    if read {
        encoder.copy_buffer_to_buffer(&out, 0, &out_staging, 0, cells_size);
        encoder.copy_buffer_to_buffer(&dirs, 0, &dirs_staging, 0, dirs_size);
    }
    queue.submit([encoder.finish()]);
    if !read {
        device
            .poll(wgpu::PollType::wait_indefinitely())
            .expect("the device polls");
        return (Vec::new(), Vec::new());
    }
    let cell_bytes = read_back(device, &out_staging, cells_size);
    let dir_bytes = read_back(device, &dirs_staging, dirs_size);
    (
        cell_bytes
            .chunks_exact(4)
            .map(|b| u32::from_le_bytes(b.try_into().expect("4 bytes")))
            .collect(),
        dir_bytes
            .chunks_exact(24)
            .map(|row| {
                let word = |k: usize| {
                    Gi::new(i64::from_le_bytes(
                        row[k * 8..k * 8 + 8].try_into().expect("8 bytes"),
                    ))
                };
                [word(0), word(1), word(2)]
            })
            .collect(),
    )
}

/// ★ WHAT ONE BOX ON THE CARD COST, and what it wrote (ruling F9 item 2, review item 1).
pub struct CardRun {
    /// The cells, as the card wrote them — RAW BYTES. The decode is the geometry stage's work, not
    /// the card thread's (review item 1c).
    pub cells: Vec<u8>,
    /// The columns' directions, raw.
    pub dirs: Vec<u8>,
    /// THE CARD'S OWN SECONDS for this box: the device's own timestamps where it offers them, and
    /// the submit-to-map wall time where it does not.
    pub device_s: f64,
}

/// How many words of a query set one box writes: the first pass's start and the last pass's end.
const CARD_TIMESTAMPS: u32 = 2;

/// ★ THE CARD'S REUSED GEAR (review item 1b): every buffer one box needs, kept and GROWN, never
/// created again. A box writes into the buffers with `Queue::write_buffer` and the bind groups are
/// kept beside them, so the steady state of the card's builder allocates NOTHING — no buffer, no
/// bind group and no byte vector — for a box.
///
/// **Example.** The pilot flies at 528 m/s. The card builds forty boxes a second for a minute: the
/// first box creates fourteen buffers and three bind groups, and the other two thousand three
/// hundred write into the same ones.
pub struct BoxGear {
    device: wgpu::Device,
    queue: wgpu::Queue,
    chain: BoxChain,
    /// The uploads, in the order the passes read them.
    plan_charter: Slot,
    sites: Slot,
    lattices: Slot,
    node_z: Slot,
    radii: Slot,
    charter: Slot,
    layers: Slot,
    tubes: Slot,
    /// What the card writes for itself.
    columns: Slot,
    nodes: Slot,
    /// What comes home, and the staging it is mapped through.
    dirs: Slot,
    dirs_staging: Slot,
    out: Slot,
    out_staging: Slot,
    /// The three bind groups, rebuilt only when a buffer was grown.
    groups: Option<[wgpu::BindGroup; 3]>,
    /// One byte vector, reused for every upload of every box.
    scratch: Vec<u8>,
    /// THE CARD'S OWN CLOCK, where the device offers one: the query set, the buffer the card
    /// resolves it into, the staging the host maps, and the nanoseconds one tick is worth.
    clock: Option<(wgpu::QuerySet, wgpu::Buffer, wgpu::Buffer, f32)>,
}

/// One pooled buffer and the bytes it holds.
#[derive(Default)]
struct Slot {
    buffer: Option<wgpu::Buffer>,
    bytes: u64,
}

impl Slot {
    /// The buffer, grown to `need` bytes where it is too small. Answers whether it was rebuilt, so
    /// the bind groups follow. A buffer is never shrunk: a box at a finer rung asks for the same
    /// bytes as the last one of its rung.
    fn ensure(&mut self, device: &wgpu::Device, need: u64, usage: wgpu::BufferUsages) -> bool {
        if self.bytes >= need.max(1) {
            return false;
        }
        self.bytes = need.max(1);
        self.buffer = Some(device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: self.bytes,
            usage,
            mapped_at_creation: false,
        }));
        true
    }

    /// The buffer, once it has been ensured.
    fn get(&self) -> &wgpu::Buffer {
        self.buffer.as_ref().expect("the slot was ensured")
    }
}

/// The usages of a buffer the host writes and a pass reads.
const UPLOAD: wgpu::BufferUsages = wgpu::BufferUsages::STORAGE.union(wgpu::BufferUsages::COPY_DST);
/// The usages of a buffer the card writes for itself.
const SCRATCH: wgpu::BufferUsages = wgpu::BufferUsages::STORAGE;
/// The usages of a buffer the card writes and the host reads.
const READ_OUT: wgpu::BufferUsages =
    wgpu::BufferUsages::STORAGE.union(wgpu::BufferUsages::COPY_SRC);
/// The usages of the staging a readback is mapped through.
const STAGING: wgpu::BufferUsages =
    wgpu::BufferUsages::MAP_READ.union(wgpu::BufferUsages::COPY_DST);

impl BoxGear {
    /// The gear for one card thread. The clock is the device's own where it offers timestamps.
    #[must_use]
    pub fn new(device: wgpu::Device, queue: wgpu::Queue) -> BoxGear {
        let chain = BoxChain::new(&device);
        let clock = device
            .features()
            .contains(wgpu::Features::TIMESTAMP_QUERY)
            .then(|| {
                let set = device.create_query_set(&wgpu::QuerySetDescriptor {
                    label: Some("the card's own clock"),
                    ty: wgpu::QueryType::Timestamp,
                    count: CARD_TIMESTAMPS,
                });
                let size = u64::from(CARD_TIMESTAMPS) * 8;
                let resolve = device.create_buffer(&wgpu::BufferDescriptor {
                    label: None,
                    size,
                    usage: wgpu::BufferUsages::QUERY_RESOLVE | wgpu::BufferUsages::COPY_SRC,
                    mapped_at_creation: false,
                });
                let staging = device.create_buffer(&wgpu::BufferDescriptor {
                    label: None,
                    size,
                    usage: STAGING,
                    mapped_at_creation: false,
                });
                (set, resolve, staging, queue.get_timestamp_period())
            });
        BoxGear {
            device,
            queue,
            chain,
            plan_charter: Slot::default(),
            sites: Slot::default(),
            lattices: Slot::default(),
            node_z: Slot::default(),
            radii: Slot::default(),
            charter: Slot::default(),
            layers: Slot::default(),
            tubes: Slot::default(),
            columns: Slot::default(),
            nodes: Slot::default(),
            dirs: Slot::default(),
            dirs_staging: Slot::default(),
            out: Slot::default(),
            out_staging: Slot::default(),
            groups: None,
            scratch: Vec::new(),
            clock: None,
        }
        .with_clock(clock)
    }

    fn with_clock(
        mut self,
        clock: Option<(wgpu::QuerySet, wgpu::Buffer, wgpu::Buffer, f32)>,
    ) -> BoxGear {
        self.clock = clock;
        self
    }

    /// Whether the card's own time is READ FROM THE DEVICE (its timestamps) rather than taken from
    /// the host's wall clock around the submit. The stamp states which, because the budget rations
    /// this number (review item 1a).
    #[must_use]
    pub fn device_timed(&self) -> bool {
        self.clock.is_some()
    }

    /// Fill a pooled buffer from 64-bit words, through the one scratch vector.
    fn write_words(&mut self, which: Which, words: &[i64], usage: wgpu::BufferUsages) -> bool {
        let mut scratch = std::mem::take(&mut self.scratch);
        scratch.clear();
        for w in words {
            scratch.extend_from_slice(&w.to_le_bytes());
        }
        let grown = self.upload(which, &scratch, usage);
        self.scratch = scratch;
        grown
    }

    /// Fill a pooled buffer from 32-bit words.
    fn write_small(&mut self, which: Which, words: &[i32], usage: wgpu::BufferUsages) -> bool {
        let mut scratch = std::mem::take(&mut self.scratch);
        scratch.clear();
        for w in words {
            scratch.extend_from_slice(&w.to_le_bytes());
        }
        let grown = self.upload(which, &scratch, usage);
        self.scratch = scratch;
        grown
    }

    /// The bytes into the pooled buffer, growing it where this box is the widest one yet.
    fn upload(&mut self, which: Which, bytes: &[u8], usage: wgpu::BufferUsages) -> bool {
        let device = self.device.clone();
        let queue = self.queue.clone();
        let slot = self.slot(which);
        let grown = slot.ensure(&device, bytes.len() as u64, usage);
        queue.write_buffer(slot.get(), 0, bytes);
        grown
    }

    fn slot(&mut self, which: Which) -> &mut Slot {
        match which {
            Which::PlanCharter => &mut self.plan_charter,
            Which::Sites => &mut self.sites,
            Which::Lattices => &mut self.lattices,
            Which::NodeZ => &mut self.node_z,
            Which::Radii => &mut self.radii,
            Which::Charter => &mut self.charter,
            Which::Layers => &mut self.layers,
            Which::Tubes => &mut self.tubes,
        }
    }

    /// ★ ONE BOX ON THE CARD, WHOLE — the column pass, the node pass and the cell field, in three
    /// compute passes of ONE command encoder, one submit, one wait. Every failure is an answer,
    /// never a panic (review item 4): a card that stops answering detaches the builder and the CPU
    /// share carries the ladder alone.
    pub fn run(&mut self, plan: &BoxPlan) -> Result<CardRun, String> {
        let edge = plan.charter.box_edge.raw() as u32;
        let columns_n = u64::from(edge) * u64::from(edge);
        let cells_size = columns_n * u64::from(edge) * 4;
        let dirs_size = columns_n * 3 * 8;
        // What goes up, through the one scratch vector and the pooled buffers.
        let mut grown = self.write_words(Which::PlanCharter, &plan.plan_charter_words(), UPLOAD);
        grown |= self.write_small(Which::Sites, &plan.site_words(), UPLOAD);
        grown |= self.write_words(Which::Lattices, &plan.lattice_words(), UPLOAD);
        grown |= self.write_words(Which::NodeZ, &plan.node_z_words(), UPLOAD);
        grown |= self.write_words(Which::Radii, &plan.node_radius_words(), UPLOAD);
        grown |= self.write_words(Which::Charter, &plan.charter_words(), UPLOAD);
        grown |= self.write_words(Which::Layers, &plan.layer_words(), UPLOAD);
        grown |= self.write_words(Which::Tubes, &plan.tube_words(), UPLOAD);
        // What the card writes for itself, and what comes home.
        let column_bytes = columns_n * vd_terrain::gpu::COLUMN_WORDS as u64 * 8;
        grown |= self.columns.ensure(&self.device, column_bytes, SCRATCH);
        grown |= self
            .nodes
            .ensure(&self.device, plan.node_count as u64 * 8, SCRATCH);
        grown |= self.dirs.ensure(&self.device, dirs_size, READ_OUT);
        grown |= self.out.ensure(&self.device, cells_size, READ_OUT);
        self.dirs_staging.ensure(&self.device, dirs_size, STAGING);
        self.out_staging.ensure(&self.device, cells_size, STAGING);
        if grown || self.groups.is_none() {
            self.groups = Some(self.build_groups());
        }
        let groups = self.groups.as_ref().expect("the groups were built");

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: None,
                timestamp_writes: self.clock.as_ref().map(|(set, ..)| {
                    wgpu::ComputePassTimestampWrites {
                        query_set: set,
                        beginning_of_pass_write_index: Some(0),
                        end_of_pass_write_index: None,
                    }
                }),
            });
            pass.set_pipeline(&self.chain.column);
            pass.set_bind_group(0, &groups[0], &[]);
            pass.dispatch_workgroups((columns_n as u32).div_ceil(COLUMN_WORKGROUP), 1, 1);
        }
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor::default());
            pass.set_pipeline(&self.chain.node);
            pass.set_bind_group(0, &groups[1], &[]);
            pass.dispatch_workgroups(
                plan.node_extent[0].div_ceil(NODE_WORKGROUP),
                plan.node_extent[1],
                plan.node_z.len() as u32,
            );
        }
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: None,
                timestamp_writes: self.clock.as_ref().map(|(set, ..)| {
                    wgpu::ComputePassTimestampWrites {
                        query_set: set,
                        beginning_of_pass_write_index: None,
                        end_of_pass_write_index: Some(1),
                    }
                }),
            });
            pass.set_pipeline(&self.chain.cell);
            pass.set_bind_group(0, &groups[2], &[]);
            pass.dispatch_workgroups(edge.div_ceil(CELL_WORKGROUP), edge, edge);
        }
        encoder.copy_buffer_to_buffer(self.out.get(), 0, self.out_staging.get(), 0, cells_size);
        encoder.copy_buffer_to_buffer(self.dirs.get(), 0, self.dirs_staging.get(), 0, dirs_size);
        if let Some((set, resolve, staging, _)) = self.clock.as_ref() {
            encoder.resolve_query_set(set, 0..CARD_TIMESTAMPS, resolve, 0);
            encoder.copy_buffer_to_buffer(resolve, 0, staging, 0, u64::from(CARD_TIMESTAMPS) * 8);
        }
        let at = Instant::now();
        self.queue.submit([encoder.finish()]);
        let cells = map_bytes(&self.device, self.out_staging.get(), cells_size)?;
        let dirs = map_bytes(&self.device, self.dirs_staging.get(), dirs_size)?;
        let wall_s = at.elapsed().as_secs_f64();
        // ★ THE CARD'S OWN SECONDS: the device's clock where it has one, the submit-to-map wall
        // time where it has not. The budget rations THIS number, so which one it is matters.
        let device_s = match self.clock.as_ref() {
            Some((_, _, staging, period_ns)) => {
                let ticks = map_bytes(&self.device, staging, u64::from(CARD_TIMESTAMPS) * 8)?;
                let word = |k: usize| -> u64 {
                    u64::from_le_bytes(ticks[k * 8..k * 8 + 8].try_into().unwrap_or([0; 8]))
                };
                let span = word(1).saturating_sub(word(0)) as f64;
                span * f64::from(*period_ns) / 1.0e9
            }
            None => wall_s,
        };
        Ok(CardRun {
            cells,
            dirs,
            device_s,
        })
    }

    /// The three bind groups over the pooled buffers, in the order the passes read them.
    fn build_groups(&self) -> [wgpu::BindGroup; 3] {
        let group = |pipeline: &wgpu::ComputePipeline, buffers: [&wgpu::Buffer; 5]| {
            let entries: Vec<wgpu::BindGroupEntry> = buffers
                .iter()
                .enumerate()
                .map(|(i, b)| wgpu::BindGroupEntry {
                    binding: i as u32,
                    resource: b.as_entire_binding(),
                })
                .collect();
            self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: None,
                layout: &pipeline.get_bind_group_layout(0),
                entries: &entries,
            })
        };
        let group6 = |pipeline: &wgpu::ComputePipeline, buffers: [&wgpu::Buffer; 6]| {
            let entries: Vec<wgpu::BindGroupEntry> = buffers
                .iter()
                .enumerate()
                .map(|(i, b)| wgpu::BindGroupEntry {
                    binding: i as u32,
                    resource: b.as_entire_binding(),
                })
                .collect();
            self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: None,
                layout: &pipeline.get_bind_group_layout(0),
                entries: &entries,
            })
        };
        [
            group(
                &self.chain.column,
                [
                    self.plan_charter.get(),
                    self.sites.get(),
                    self.lattices.get(),
                    self.columns.get(),
                    self.dirs.get(),
                ],
            ),
            group(
                &self.chain.node,
                [
                    self.plan_charter.get(),
                    self.lattices.get(),
                    self.node_z.get(),
                    self.radii.get(),
                    self.nodes.get(),
                ],
            ),
            group6(
                &self.chain.cell,
                [
                    self.charter.get(),
                    self.layers.get(),
                    self.columns.get(),
                    self.nodes.get(),
                    self.tubes.get(),
                    self.out.get(),
                ],
            ),
        ]
    }
}

/// Which pooled upload a write goes to.
#[derive(Clone, Copy)]
enum Which {
    PlanCharter,
    Sites,
    Lattices,
    NodeZ,
    Radii,
    Charter,
    Layers,
    Tubes,
}

/// The bytes of a mapped staging buffer, once the card is done — AN ANSWER, never a panic. The
/// buffer is unmapped before the answer returns, so the same staging serves the next box.
fn map_bytes(device: &wgpu::Device, staging: &wgpu::Buffer, size: u64) -> Result<Vec<u8>, String> {
    let slice = staging.slice(..size);
    let (tx, rx) = std::sync::mpsc::channel();
    slice.map_async(wgpu::MapMode::Read, move |r| {
        let _ = tx.send(r);
    });
    device
        .poll(wgpu::PollType::wait_indefinitely())
        .map_err(|e| format!("the device did not poll: {e}"))?;
    rx.recv()
        .map_err(|e| format!("the map never answered: {e}"))?
        .map_err(|e| format!("the map failed: {e}"))?;
    let bytes = slice.get_mapped_range().to_vec();
    staging.unmap();
    Ok(bytes)
}

/// A buffer the card writes and a staging buffer the host maps.
fn read_pair(device: &wgpu::Device, size: u64) -> (wgpu::Buffer, wgpu::Buffer) {
    (
        device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        }),
        device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        }),
    )
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

/// ★ THE SEAM PROBE'S SWITCH (ruling F9 item 2, step 1): `VD_TERRAIN_GPU_SEAM=<seconds>` starts a
/// WORKER THREAD that many seconds after the client starts, and that thread dispatches the box
/// chain on the RENDERER'S OWN device — submitting and waiting — for as long as the client runs.
/// Absent, no thread starts and nothing changes.
///
/// It answers the two questions ruling F9 item 2 must answer before the card may build: does a
/// worker thread submitting compute to the renderer's device cost the renderer its frames, and
/// does a worker's `device.poll(wait)` stall the renderer's own submits? The client's frame counter
/// and its worst frame before and after the probe starts are the measurement; the probe's own log
/// line states its rate and its worst box.
pub const SEAM_PROBE_ENV: &str = "VD_TERRAIN_GPU_SEAM";
/// A box this long is a STALL, not a box: the probe counts it and names it. Half a second is
/// twenty frames of the capture client's own sixty-a-second loop — far past any box the bench ever
/// measured (about 2 ms), so a reading here is a seam fault and never a slow card.
const SEAM_STALL: std::time::Duration = std::time::Duration::from_millis(500);
/// HOW OFTEN THE PROBE STATES WHAT IT MEASURED: every two seconds. The capture client's own loop
/// runs at sixty frames a second, so a two-second window holds about a hundred frames — enough that
/// one slow frame moves the window's rate by a percent and not by a third, and short enough that a
/// thirty-five-second run holds seven quiet windows and nine busy ones (review item 10).
const SEAM_REPORT_S: f64 = 2.0;
/// HOW OFTEN THE IDLE PHASE LOOKS AT ITS CLOCK: a fiftieth of a second. It is the phase that builds
/// NOTHING, so the only thing this paces is when the probe notices that its idle seconds are up;
/// a fiftieth is a fortieth of the report window, so no window is ever short by a measurable part
/// (review item 10).
const SEAM_IDLE_TICK: std::time::Duration = std::time::Duration::from_millis(20);

/// ★ THE SEAM PROBE (ruling F9 item 2, step 1): a worker thread that builds the eight golden boxes
/// on the renderer's own device, over and over, while the renderer draws — and that STATES WHAT
/// THE RENDERER IS DOING as it does so, because no other thread can.
///
/// The probe runs in TWO PHASES, so one client run holds both halves of the comparison: it IDLES
/// for `after_s` seconds and reports the frames the renderer drew, then it BUILDS and reports the
/// same frames beside its own rate. The difference between the two phases is the seam's cost.
///
/// **Example.** The capture client stands in the home system. For fifteen seconds nothing but the
/// renderer touches the card, and the probe says so every two seconds; then the probe begins, and
/// the same line says how many boxes it built, what a box cost it, whether any box took past
/// [`SEAM_STALL`] — and what the renderer's frames did while it did that.
pub fn spawn_seam_probe(
    device: wgpu::Device,
    queue: wgpu::Queue,
    after_s: f64,
    meter: std::sync::Arc<crate::terrain::FrameMeter>,
) {
    std::thread::Builder::new()
        .name("terrain-gpu-seam".to_owned())
        .spawn(move || {
            let body = vd_terrain::home::home_planet();
            let plans: Vec<BoxPlan> = GOLDEN_SELF_CHECK_KEYS
                .iter()
                .filter_map(|entry| vd_terrain::gpu::plan(&body, self_check_key(&body, *entry)))
                .collect();
            // THE PROBE RUNS THE BUILDER'S OWN GEAR (review item 1): the same pooled buffers, the
            // same timestamps, the same read back — so what it measures ALONE is what the flight
            // then measures UNDER LOAD, and the two numbers are comparable.
            let mut gear = BoxGear::new(device, queue);
            tracing::info!(
                boxes = plans.len(),
                after_s,
                device_timed = gear.device_timed(),
                "THE SEAM PROBE is armed: it idles first, then submits the box chain to the \
                 renderer's own device and waits for it"
            );
            let started = Instant::now();
            // The building phase's own clock: the probe's rate is its own, never diluted by the
            // seconds it stood idle.
            let mut build_started = Instant::now();
            let mut building = false;
            // The phase's own counters, and the mark the report's window runs from.
            let mut built = 0u64;
            let mut nanos = 0u64;
            let mut worst_ns = 0u64;
            let mut frames_at = meter.frames.load(std::sync::atomic::Ordering::Relaxed);
            let mut mark = Instant::now();
            let mut peak_ns = 0u64;
            let mut stalls = 0u64;
            loop {
                if building {
                    for plan in &plans {
                        let at = Instant::now();
                        let Ok(run) = gear.run(plan) else {
                            tracing::warn!("THE SEAM PROBE stops: a box failed on the device");
                            return;
                        };
                        // The answer is READ, so nothing here can be optimised away.
                        assert!(
                            !run.cells.is_empty() & !run.dirs.is_empty(),
                            "the card wrote a box"
                        );
                        let dt = at.elapsed();
                        built += 1;
                        // THE CARD'S OWN SECONDS, not the thread's: the budget rations these.
                        nanos += (run.device_s * 1.0e9) as u64;
                        worst_ns = worst_ns.max(dt.as_nanos() as u64);
                        stalls += u64::from(dt >= SEAM_STALL);
                    }
                } else {
                    std::thread::sleep(SEAM_IDLE_TICK);
                }
                peak_ns = peak_ns.max(meter.peak_ns.load(std::sync::atomic::Ordering::Relaxed));
                let window_s = mark.elapsed().as_secs_f64();
                if window_s >= SEAM_REPORT_S {
                    let frames_now = meter.frames.load(std::sync::atomic::Ordering::Relaxed);
                    tracing::info!(
                        phase = if building { "building" } else { "idle" },
                        frames_per_s = frames_now.saturating_sub(frames_at) as f64 / window_s,
                        worst_frame_ms = peak_ns as f64 / 1.0e6,
                        boxes = built,
                        boxes_per_s =
                            built as f64 / build_started.elapsed().as_secs_f64().max(1.0e-9),
                        mean_ms = nanos as f64 / built.max(1) as f64 / 1.0e6,
                        worst_ms = worst_ns as f64 / 1.0e6,
                        stalls,
                        device_timed = gear.device_timed(),
                        "THE SEAM PROBE"
                    );
                    frames_at = frames_now;
                    peak_ns = 0;
                    mark = Instant::now();
                }
                if !building && (started.elapsed().as_secs_f64() >= after_s.max(0.0)) {
                    building = true;
                    build_started = Instant::now();
                    frames_at = meter.frames.load(std::sync::atomic::Ordering::Relaxed);
                    peak_ns = 0;
                    mark = Instant::now();
                    tracing::info!("THE SEAM PROBE starts building on the renderer's own device");
                }
            }
        })
        .expect("the seam probe's thread starts");
}

/// The startup system: the check on the home planet (the body the world identity is measured
/// on), the verdict logged and kept as a resource.
pub fn gpu_recipe_self_check(
    mut commands: Commands,
    mut terrain: ResMut<crate::terrain::Terrain>,
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
    // ★ THE CARD AS A SECOND BUILDER (ruling F9 item 2): the self-check's verdict decides at start
    // whether the card may build the world's shape at all. It runs here because this is where the
    // renderer's own device first exists and the verdict first stands.
    terrain.attach_card(device.clone(), queue.clone(), check.trusted());
    // ★ THE SEAM PROBE (ruling F9 item 2, step 1), only where a measurement asks for it by name.
    if let Some(after_s) = std::env::var(SEAM_PROBE_ENV)
        .ok()
        .and_then(|v| v.trim().parse::<f64>().ok())
    {
        if check.trusted() {
            spawn_seam_probe(
                device.clone(),
                queue.clone(),
                after_s,
                terrain.frame_meter(),
            );
        } else {
            tracing::warn!("THE SEAM PROBE is asked for, but this GPU is not trusted");
        }
    }
    commands.insert_resource(check);
}
