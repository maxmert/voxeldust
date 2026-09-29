//! ★ THE GPU'S OWN CLOCK (ruling B22 step 2, 2026-09-28): the GPU time of one frame's work,
//! read from the GPU itself.
//!
//! The window's frame time on the owner's display steps in whole refreshes. MEASURED on the
//! full-screen bench of 2026-09-28: every unpaced sample stood at 25.0, 33.3 or 41.7 ms — the
//! 120 Hz display's 8.33 ms steps — with the frame loop unpaced. So a cost smaller than one step
//! cannot be read from the frame time: the stand on foot read 33.3 ms whether its work took 26 ms
//! or 33 ms. This clock takes two GPU timestamps a frame and reads the pair back some frames later
//! without a stall.
//!
//! ★ THE STAMPS ARE TAKEN BY PASSES OF THE CLOCK'S OWN. This GPU samples its clock only at a pass's
//! start or end. MEASURED on the first try (stamps written outside any pass): the Metal backend
//! takes such a stamp at the start of the NEXT encoder — the second one fell on the start of the
//! very copy that read it — every reading was nought and the window's frame took 275 ms. Now:
//!   - the FIRST stamp is the start of an empty pass on the clock's own one-pixel picture, recorded
//!     before every other node of the frame's render graph;
//!   - the SECOND stamp is the end of an empty pass that loads and stores each camera's OUTPUT
//!     picture, recorded after every other node. The GPU cannot run that pass before the last
//!     pass that writes the same picture has finished, so the stamp stands after the frame's
//!     drawing. Its own price is one load and one store of that picture (about 0.3 ms at full
//!     screen), inside the reading.
//!
//! ONE query set of [`SLOTS`] pairs is made at the start and reused: the engine's own render
//! diagnostics made a new set for every frame whose readback had not returned, and on this GPU
//! that ran the driver out of counter buffers and lost the device (the crash of 2026-09-23, see
//! `RENDER_DIAG_ENV`). A frame whose slot is still being read back records nothing.
//!
//! Published as two diagnostics: `render/gpu_frame/elapsed_gpu`, the span from the first stamp to
//! the second (the frame's own work, and any of the previous frame's work the GPU still ran after
//! the first stamp); and `render/gpu_step/elapsed_gpu`, the time from one frame's second stamp to
//! the next frame's, when both frames were read (the GPU's time per frame when it never waits; with
//! waits for the display, those waits too). The stamp's list of passes carries both. A
//! measurement's switch: `VD_GPU_CLOCK=1`.
//!
//! Example: the owner's stand on foot draws 14 261 chunks at full screen; its frame time reads
//! 33.3 ms, and the clock reads the milliseconds the GPU worked on each of those frames.
//!
//! ★ THE PER-PASS STAMPS (the temporal path's slice T0 item 1; `VD_GPU_CLOCK=passes`): the same
//! proven method between the nodes of each camera's own graph. A stamp pass after a node loads and
//! stores the camera's picture, so the GPU cannot take that stamp before every earlier pass that
//! writes the picture has finished. The time from one stamp to the next is the node's. The nodes and
//! their stamps run as ONE chain, in an order that the graph's own edges allow, so no other node runs
//! between a node and its stamp.
//!   - A stamp waits only for the passes that write the camera's PICTURE. A node that writes only
//!     the shadow maps, the depth or a buffer (the shadow passes, the prepasses, the culling compute)
//!     can end after its stamp, and then its time lands in the next stamp that waits for it: the
//!     main pass samples the shadow maps and tests the depth. So those numbers are APPROXIMATE, and
//!     the price by absence stays the judge (the design's §11 T0 item 1).
//!   - Each stamp costs one load and one store of the picture (about 0.25 ms at full screen), inside
//!     the next node's reading and inside the frame's span. A per-pass run is a measurement of its
//!     own, never the frame's total.
//!   - The engine's five marker nodes do no work, and they get no stamp.
//!
//! Published as `render/gpu_pass_c<order>_<node>/elapsed_gpu` for each camera (by its order) and
//! node, and as `render/gpu_pass_sum/elapsed_gpu`, their sum. The sum must stand within the frame's
//! span (T0's gate). The stamp's list of passes carries them all.
//!
//! Example: the owner's stand on foot, full screen, MSAA off. The stamp's passes read
//! `gpu_pass_c0_MainOpaquePass` (the ground's geometry and its shading, with the shadow passes' time
//! in it where they overlap), `gpu_pass_c0_Bloom`, `gpu_pass_c0_Tonemapping` and the rest, and their
//! sum stands under the frame's span.

use std::collections::BTreeMap;
use std::sync::atomic::{AtomicBool, AtomicU8, AtomicU32, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

use bevy::core_pipeline::core_3d::graph::{Core3d, Node3d};
use bevy::diagnostic::{
    Diagnostic, DiagnosticMeasurement, DiagnosticPath, DiagnosticsStore, RegisterDiagnostic,
};
use bevy::platform::time::Instant;
use bevy::prelude::*;
use bevy::render::camera::ExtractedCamera;
use bevy::render::graph::CameraDriverLabel;
use bevy::render::render_graph::{
    Edge, InternedRenderLabel, Node, NodeRunError, RenderGraph, RenderGraphContext,
    RenderGraphError, RenderLabel,
};
use bevy::render::render_resource::{Buffer, TextureView};
use bevy::render::renderer::{RenderContext, RenderDevice, RenderQueue};
use bevy::render::view::ViewTarget;
use bevy::render::{Render, RenderApp, RenderSystems};

/// The switch: `VD_GPU_CLOCK=1` for the frame's two stamps; `VD_GPU_CLOCK=passes` adds the per-pass
/// stamps.
pub(crate) const GPU_CLOCK_ENV: &str = "VD_GPU_CLOCK";

/// The switch's value that adds the per-pass stamps.
const GPU_CLOCK_PASSES: &str = "passes";

/// Whether the clock runs.
pub(crate) fn gpu_clock() -> bool {
    std::env::var(GPU_CLOCK_ENV).is_ok_and(|v| v == "1" || v == GPU_CLOCK_PASSES)
}

/// Whether the clock also stamps each node of the cameras' graph.
fn gpu_clock_passes() -> bool {
    std::env::var(GPU_CLOCK_ENV).is_ok_and(|v| v == GPU_CLOCK_PASSES)
}

/// How many frames' stamps may be in flight at once: a readback returns a few frames after its
/// frame, and a frame whose slot is still out records nothing.
const SLOTS: usize = 8;

/// ★ A SLOT IS RESOLVED THIS MANY FRAMES AFTER ITS FRAME, never in it. MEASURED (2026-09-29, the
/// clock's raw ticks, headless): resolved in its own frame, about three pairs in four read an END
/// stamp that was the same slot's value four frames before — frame 2762's end stood 82 ms BEFORE its
/// begin and equalled frame 2758's end. The resolve's copy is not ordered after the end pass's
/// timestamp write (no tracked resource joins them), so it often ran first. Three frames later the
/// stamps are long written: MEASURED after, about 200 readings in every 5 s headless (one a frame)
/// and none out of order. Two earlier guesses were MEASURED and REFUTED on the way: a one-pixel draw
/// in each stamp pass (in case this GPU writes no timestamp for a pass in which no stage runs) left
/// the count unchanged, and so it is not kept; the device poll stays, because the window polls the
/// device for no other readback.
const RESOLVE_DELAY: u64 = 3;

/// The most per-pass stamps one frame records: a stamp for each working node of a camera's graph
/// (about thirty), for each camera (the capture has two). A frame's stamps past it are not taken.
const MAX_PASS_STAMPS: u32 = 192;

/// The queries of one slot: the frame's two stamps, then its per-pass stamps.
const SLOT_QUERIES: u32 = 2 + MAX_PASS_STAMPS;

/// The byte step between two slots in the resolve buffer: a slot's queries (eight bytes each),
/// rounded up to the resolve's alignment, because a resolve's destination offset must be a
/// multiple of it.
const SLOT_BYTES: u64 =
    (SLOT_QUERIES as u64 * 8).next_multiple_of(wgpu::QUERY_RESOLVE_BUFFER_ALIGNMENT);

/// The format of the clock's own one-pixel picture (the first stamp's attachment).
const OWN_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba8Unorm;

/// A readback's mapping: still pending, done and readable, or failed (the driver refused it).
const MAP_PENDING: u8 = 0;
const MAP_OK: u8 = 1;
const MAP_FAILED: u8 = 2;

/// ★ A MAPPING STILL PENDING AFTER THIS MANY FRAMES IS CANCELLED, and its slot is used again. A
/// frame's work ends a few frames after its submission, so a mapping this old is stuck (MEASURED
/// 2026-09-29, the T0 window session: 4 of 21 modes read one value for a whole minute, because a
/// slot that never came back stopped the clock for good — each frame waits for the NEXT slot).
const STUCK_FRAMES: u64 = 240;

/// How often the clock states its own health in the log, in seconds: the readings it delivered,
/// the pairs out of order, the mappings that failed or stuck, and the frames with no free slot.
const HEALTH_LOG_S: f64 = 5.0;

/// The engine's marker nodes of a camera's graph: they order the other nodes and do no work, so
/// they get no stamp.
const MARKER_NODES: [Node3d; 5] = [
    Node3d::EndPrepasses,
    Node3d::StartMainPass,
    Node3d::EndMainPass,
    Node3d::StartMainPassPostProcessing,
    Node3d::EndMainPassPostProcessing,
];

/// The span from the first stamp to the second, in milliseconds.
pub(crate) const GPU_FRAME_PATH: DiagnosticPath =
    DiagnosticPath::const_new("render/gpu_frame/elapsed_gpu");

/// The time from one read frame's second stamp to the next frame's, in milliseconds.
pub(crate) const GPU_STEP_PATH: DiagnosticPath =
    DiagnosticPath::const_new("render/gpu_step/elapsed_gpu");

/// The sum of a frame's per-pass readings, in milliseconds (`VD_GPU_CLOCK=passes`).
const GPU_PASS_SUM_PATH: DiagnosticPath =
    DiagnosticPath::const_new("render/gpu_pass_sum/elapsed_gpu");

#[derive(Debug, Hash, PartialEq, Eq, Clone, RenderLabel)]
struct GpuClockBegin;

#[derive(Debug, Hash, PartialEq, Eq, Clone, RenderLabel)]
struct GpuClockEnd;

/// The per-pass stamp nodes of a camera's graph: 0 starts the graph, and each other one follows
/// one working node.
#[derive(Debug, Hash, PartialEq, Eq, Clone, RenderLabel)]
struct GpuPassStamp(usize);

/// A frame's per-pass readings (each node's name and milliseconds) and their sum.
type PassReadings = (Vec<(String, f64)>, f64);

/// The newest readings, shared by the two worlds: the span and the step in milliseconds as `f64`
/// bits with whether the main world has not yet taken each, and the newest per-pass readings until
/// the main world takes them.
#[derive(Resource, Clone, Default)]
struct GpuClockReading {
    values: Arc<[(AtomicU64, AtomicBool); 2]>,
    passes: Arc<Mutex<Option<PassReadings>>>,
}

impl GpuClockReading {
    fn put(&self, which: usize, ms: f64) {
        self.values[which].0.store(ms.to_bits(), Ordering::Relaxed);
        self.values[which].1.store(true, Ordering::Release);
    }

    fn take(&self, which: usize) -> Option<f64> {
        self.values[which]
            .1
            .swap(false, Ordering::Acquire)
            .then(|| f64::from_bits(self.values[which].0.load(Ordering::Relaxed)))
    }

    fn put_passes(&self, readings: PassReadings) {
        if let Ok(mut passes) = self.passes.lock() {
            *passes = Some(readings);
        }
    }

    fn take_passes(&self) -> Option<PassReadings> {
        self.passes.lock().ok().and_then(|mut passes| passes.take())
    }
}

/// One per-pass stamp of a frame: its index among the frame's pass stamps, the order of the camera
/// whose graph took it, and the node it follows (`None`: the stamp that starts the camera's graph).
#[derive(Clone, Copy)]
struct PassMark {
    index: u32,
    order: isize,
    after: Option<InternedRenderLabel>,
}

/// One slot's state.
enum Slot {
    /// Free to record into.
    Free,
    /// Its frame recorded its stamps (the frame's count, the queries it used, the per-pass
    /// marks); it waits [`RESOLVE_DELAY`] frames to be resolved.
    Recorded(u64, u32, Vec<PassMark>),
    /// Its readback is out for the frame of this count; the state turns when the mapping ends
    /// ([`MAP_OK`] or [`MAP_FAILED`]); the marks name the frame's per-pass stamps.
    Mapping(u64, Arc<AtomicU8>, Vec<PassMark>),
}

/// The clock in the render world.
#[derive(Resource)]
struct GpuClock {
    queries: wgpu::QuerySet,
    resolve_buffer: Buffer,
    /// The recorded slot this frame's end node resolves (its index and the queries it used).
    due: Option<(usize, u32)>,
    readback: Vec<Buffer>,
    /// The clock's own one-pixel picture, the first stamp's pass's attachment.
    own: TextureView,
    slots: Vec<Slot>,
    /// The slot this frame records into, if one was free.
    current: Option<usize>,
    /// Whether this frame's end node ran (a frame with no graph run records nothing).
    ended: Arc<AtomicBool>,
    next: usize,
    /// This frame's count, and the count and second stamp of the last frame read.
    frame: u64,
    last_end: Option<(u64, u64)>,
    period_ns: f64,
    /// This frame's per-pass stamps: how many the nodes asked for, and their marks.
    pass_count: AtomicU32,
    marks: Mutex<Vec<PassMark>>,
    /// The clock's health since the last log line: the readings delivered, the pairs out of order,
    /// the mappings that failed and the ones cancelled as stuck, the frames that found no free slot;
    /// and when the window began.
    health: ClockHealth,
}

/// The clock's own health over one log window ([`HEALTH_LOG_S`]), with the raw ticks of the last
/// pair out of order and of the last good pair (the evidence of WHICH stamp goes wrong).
struct ClockHealth {
    reads: u64,
    unordered: u64,
    failed: u64,
    stuck: u64,
    skipped: u64,
    last_unordered: Option<(u64, u64, u64)>,
    last_good: Option<(u64, u64, u64)>,
    since: std::time::Instant,
}

impl ClockHealth {
    fn new() -> ClockHealth {
        ClockHealth {
            reads: 0,
            unordered: 0,
            failed: 0,
            stuck: 0,
            skipped: 0,
            last_unordered: None,
            last_good: None,
            since: std::time::Instant::now(),
        }
    }
}

/// The clock's plugin: the two graph nodes around the cameras, the choice of a slot before the
/// graph runs, the readbacks after it, and the main world's publication.
pub(crate) struct GpuClockPlugin;

impl Plugin for GpuClockPlugin {
    fn build(&self, app: &mut App) {
        let reading = GpuClockReading::default();
        app.register_diagnostic(Diagnostic::new(GPU_FRAME_PATH).with_suffix("ms"))
            .register_diagnostic(Diagnostic::new(GPU_STEP_PATH).with_suffix("ms"))
            .insert_resource(reading.clone())
            .add_systems(Update, publish);
        if gpu_clock_passes() {
            app.register_diagnostic(Diagnostic::new(GPU_PASS_SUM_PATH).with_suffix("ms"));
        }
        let Some(render_app) = app.get_sub_app_mut(RenderApp) else {
            return;
        };
        render_app.insert_resource(reading).add_systems(
            Render,
            (
                choose_slot.in_set(RenderSystems::PrepareResources),
                read_back.in_set(RenderSystems::Cleanup),
            ),
        );
        if gpu_clock_passes() {
            render_app.add_systems(Render, chain_once.in_set(RenderSystems::PrepareResources));
        }
    }

    fn finish(&self, app: &mut App) {
        let Some(render_app) = app.get_sub_app_mut(RenderApp) else {
            return;
        };
        let device = render_app.world().resource::<RenderDevice>().clone();
        if !device.features().contains(wgpu::Features::TIMESTAMP_QUERY) {
            tracing::warn!("VD_GPU_CLOCK=1 — the device offers no timestamps; the clock stays off");
            return;
        }
        let period_ns = f64::from(
            render_app
                .world()
                .resource::<RenderQueue>()
                .get_timestamp_period(),
        );
        let queries = device
            .wgpu_device()
            .create_query_set(&wgpu::QuerySetDescriptor {
                label: Some("vd_gpu_clock"),
                ty: wgpu::QueryType::Timestamp,
                count: SLOTS as u32 * SLOT_QUERIES,
            });
        let resolve = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("vd_gpu_clock_resolve"),
            size: SLOT_BYTES * SLOTS as u64,
            usage: wgpu::BufferUsages::QUERY_RESOLVE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let readback = (0..SLOTS)
            .map(|_| {
                device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some("vd_gpu_clock_readback"),
                    size: u64::from(SLOT_QUERIES) * 8,
                    usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                    mapped_at_creation: false,
                })
            })
            .collect();
        let own = device
            .create_texture(&wgpu::TextureDescriptor {
                label: Some("vd_gpu_clock_own"),
                size: wgpu::Extent3d {
                    width: 1,
                    height: 1,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: OWN_FORMAT,
                usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
                view_formats: &[],
            })
            .create_view(&wgpu::TextureViewDescriptor::default());
        render_app.insert_resource(GpuClock {
            queries,
            resolve_buffer: resolve,
            due: None,
            readback,
            own,
            slots: (0..SLOTS).map(|_| Slot::Free).collect(),
            current: None,
            ended: Arc::new(AtomicBool::new(false)),
            next: 0,
            frame: 0,
            last_end: None,
            period_ns,
            pass_count: AtomicU32::new(0),
            marks: Mutex::new(Vec::new()),
            health: ClockHealth::new(),
        });
        let end = EndNode::from_world(render_app.world_mut());
        let mut graph = render_app.world_mut().resource_mut::<RenderGraph>();
        // ★ AROUND EVERY NODE OF THE FRAME'S GRAPH, not the cameras alone: MEASURED headless with the
        // cameras alone, the span read 19.3–20.2 ms of a 23.4–25.0 ms frame the GPU filled — the
        // other nodes ran outside it. A node a later plugin adds is not wrapped (none does here: the
        // switch's plugin is added after the engine's and the overlay's).
        let others: Vec<_> = graph.iter_nodes().map(|n| n.label).collect();
        graph.add_node(GpuClockBegin, BeginNode);
        graph.add_node(GpuClockEnd, end);
        for label in others {
            graph.add_node_edge(GpuClockBegin, label);
            graph.add_node_edge(label, GpuClockEnd);
        }
        if graph.get_node_state(CameraDriverLabel).is_err() {
            tracing::warn!(
                "VD_GPU_CLOCK=1 — the graph has no camera driver; the clock reads the bare graph"
            );
        }
        tracing::info!(
            period_ns,
            "VD_GPU_CLOCK=1 — the GPU's own time for each frame, as render/gpu_frame and render/gpu_step"
        );
    }
}

/// ★ THE PER-PASS CHAIN IS BUILT ONCE, IN THE RENDER WORLD, AT ITS FIRST FRAME: after every plugin
/// has finished, and before the graph first runs.
/// - MEASURED (2026-09-29, the first headless run): a chain built in the plugin's `finish` hung the
///   capture client at its first frame. Its render thread spun at 100 % and no frame ended, because
///   plugins that finish after this one still add edges to the cameras' graph. The chain orders two
///   nodes that were independent when it was built; a later edge between them the other way round
///   closes a cycle, and the engine's graph runner re-queues a node whose inputs have not run, forever.
/// - MEASURED (the second run): a chain built in the plugin's `cleanup` was never built, because the
///   engine's pipelined renderer moves the render world onto its own thread in ITS cleanup, which
///   runs first; after that no plugin can reach the graph.
fn chain_once(mut graph: ResMut<RenderGraph>, mut done: Local<bool>) {
    if *done {
        return;
    }
    *done = true;
    let chain = chain_camera_graph(&mut graph);
    tracing::info!(
        stamps = chain.len(),
        chain = ?chain,
        "VD_GPU_CLOCK=passes — a stamp after each working node of the cameras' graph, as \
         render/gpu_pass_c<order>_<node> and render/gpu_pass_sum"
    );
}

/// Puts the nodes of the cameras' graph and a stamp after each working node into ONE chain, in an
/// order that the graph's own edges allow, and returns the stamped nodes in the chain's order.
///
/// ★ THE CHAIN IS CHECKED: after the chain is built, every node of the graph must still have its
/// place in an order that the edges allow. A chain that closed a cycle would hang the render
/// thread at its first frame, so it stops the client at its start with the nodes that have no
/// place.
fn chain_camera_graph(graph: &mut RenderGraph) -> Vec<InternedRenderLabel> {
    let Some(camera) = graph.get_sub_graph_mut(Core3d) else {
        tracing::warn!("VD_GPU_CLOCK=passes — the graph has no camera graph; no per-pass stamps");
        return Vec::new();
    };
    let markers = MARKER_NODES.map(|node| node.intern());
    let order = node_order(camera);
    camera.add_node(GpuPassStamp(0), PassStampNode { after: None });
    let mut previous = GpuPassStamp(0).intern();
    let mut stamped = Vec::new();
    for label in order {
        chain_edge(camera, previous, label);
        previous = label;
        if markers.contains(&label) {
            continue;
        }
        let stamp = GpuPassStamp(stamped.len() + 1);
        camera.add_node(stamp.clone(), PassStampNode { after: Some(label) });
        chain_edge(camera, label, stamp.intern());
        previous = stamp.intern();
        stamped.push(label);
    }
    let placed = node_order(camera);
    let unplaced: Vec<InternedRenderLabel> = camera
        .iter_nodes()
        .map(|node| node.label)
        .filter(|label| !placed.contains(label))
        .collect();
    assert!(
        unplaced.is_empty(),
        "VD_GPU_CLOCK=passes — the per-pass chain closed a cycle in the cameras' graph; these nodes \
         have no place in any order: {unplaced:?}"
    );
    stamped
}

/// One edge of the chain. An edge that the graph already holds stays as it is.
fn chain_edge(graph: &mut RenderGraph, from: InternedRenderLabel, to: InternedRenderLabel) {
    match graph.try_add_node_edge(from, to) {
        Ok(()) | Err(RenderGraphError::EdgeAlreadyExists(_)) => {}
        Err(e) => tracing::warn!(?from, ?to, %e, "VD_GPU_CLOCK=passes — a chain edge is refused"),
    }
}

/// An order of the graph's nodes in which every node comes after all its inputs (Kahn's), the ties
/// broken by the node's name so that two runs chain the same order.
fn node_order(graph: &RenderGraph) -> Vec<InternedRenderLabel> {
    let mut left: Vec<(String, InternedRenderLabel, Vec<InternedRenderLabel>)> = graph
        .iter_nodes()
        .map(|node| {
            let inputs = node
                .edges
                .input_edges()
                .iter()
                .map(Edge::get_output_node)
                .collect();
            (format!("{:?}", node.label), node.label, inputs)
        })
        .collect();
    left.sort_by(|a, b| a.0.cmp(&b.0));
    let mut done: Vec<InternedRenderLabel> = Vec::with_capacity(left.len());
    // A cycle ends the walk: the engine refuses a graph with a cycle before it ever runs.
    while let Some(i) = left
        .iter()
        .position(|(_, _, inputs)| inputs.iter().all(|input| done.contains(input)))
    {
        done.push(left.remove(i).1);
    }
    done
}

/// Before the graph runs: this frame records into the next slot when it is free.
fn choose_slot(clock: Option<ResMut<GpuClock>>) {
    let Some(mut clock) = clock else {
        return;
    };
    clock.frame += 1;
    clock.ended.store(false, Ordering::Relaxed);
    clock.pass_count.store(0, Ordering::Relaxed);
    if let Ok(marks) = clock.marks.get_mut() {
        marks.clear();
    }
    // The oldest recorded slot whose frame is at least RESOLVE_DELAY frames old.
    let now = clock.frame;
    clock.due = clock
        .slots
        .iter()
        .enumerate()
        .filter_map(|(slot, state)| match state {
            Slot::Recorded(frame, used, _) if frame + RESOLVE_DELAY <= now => {
                Some((slot, *frame, *used))
            }
            _ => None,
        })
        .min_by_key(|(_, frame, _)| *frame)
        .map(|(slot, _, used)| (slot, used));
    let next = clock.next;
    clock.current = match clock.slots[next] {
        Slot::Free => {
            clock.next = (next + 1) % SLOTS;
            Some(next)
        }
        Slot::Recorded(..) | Slot::Mapping(..) => {
            clock.health.skipped += 1;
            None
        }
    };
}

/// An empty pass on `view` that keeps its picture, stamped at its start or its end — or not stamped
/// (a camera's output waited for, not the last one): the device refuses a stamp request that names
/// no stamp (MEASURED: the capture client's two cameras, a validation error at the first frame).
fn stamp_pass(
    encoder: &mut wgpu::CommandEncoder,
    view: &wgpu::TextureView,
    queries: &wgpu::QuerySet,
    begin: Option<u32>,
    end: Option<u32>,
) {
    let stamped = begin.is_some() || end.is_some();
    let pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
        label: Some("vd_gpu_clock_stamp"),
        color_attachments: &[Some(wgpu::RenderPassColorAttachment {
            view,
            depth_slice: None,
            resolve_target: None,
            ops: wgpu::Operations {
                load: wgpu::LoadOp::Load,
                store: wgpu::StoreOp::Store,
            },
        })],
        depth_stencil_attachment: None,
        timestamp_writes: stamped.then_some(wgpu::RenderPassTimestampWrites {
            query_set: queries,
            beginning_of_pass_write_index: begin,
            end_of_pass_write_index: end,
        }),
        occlusion_query_set: None,
    });
    drop(pass);
}

/// The first stamp: the start of an empty pass on the clock's own picture, before the cameras.
struct BeginNode;

impl Node for BeginNode {
    fn run<'w>(
        &self,
        _graph: &mut RenderGraphContext,
        render_context: &mut RenderContext<'w>,
        world: &'w World,
    ) -> Result<(), NodeRunError> {
        if let Some(clock) = world.get_resource::<GpuClock>()
            && let Some(slot) = clock.current
        {
            let first = slot as u32 * SLOT_QUERIES;
            stamp_pass(
                render_context.command_encoder(),
                &clock.own,
                &clock.queries,
                Some(first),
                None,
            );
        }
        Ok(())
    }
}

/// A per-pass stamp in a camera's graph: the end of an empty pass that loads and stores the
/// camera's picture, after the node it follows (or at the start of the graph).
struct PassStampNode {
    after: Option<InternedRenderLabel>,
}

impl Node for PassStampNode {
    fn run<'w>(
        &self,
        graph: &mut RenderGraphContext,
        render_context: &mut RenderContext<'w>,
        world: &'w World,
    ) -> Result<(), NodeRunError> {
        let Some(clock) = world.get_resource::<GpuClock>() else {
            return Ok(());
        };
        let (Some(slot), Some(view)) = (clock.current, graph.get_view_entity()) else {
            return Ok(());
        };
        let Some(target) = world.get::<ViewTarget>(view) else {
            return Ok(());
        };
        let index = clock.pass_count.fetch_add(1, Ordering::Relaxed);
        if index >= MAX_PASS_STAMPS {
            return Ok(());
        }
        let order = world.get::<ExtractedCamera>(view).map_or(0, |c| c.order);
        if let Ok(mut marks) = clock.marks.lock() {
            marks.push(PassMark {
                index,
                order,
                after: self.after,
            });
        }
        stamp_pass(
            render_context.command_encoder(),
            target.main_texture_view(),
            &clock.queries,
            None,
            Some(slot as u32 * SLOT_QUERIES + 2 + index),
        );
        Ok(())
    }
}

/// The second stamp: the end of an empty pass on each camera's output picture, after the cameras,
/// and the pair's way to its readback buffer.
struct EndNode {
    views: QueryState<&'static ViewTarget>,
}

impl FromWorld for EndNode {
    fn from_world(world: &mut World) -> EndNode {
        EndNode {
            views: QueryState::new(world),
        }
    }
}

impl Node for EndNode {
    fn update(&mut self, world: &mut World) {
        self.views.update_archetypes(world);
    }

    fn run<'w>(
        &self,
        _graph: &mut RenderGraphContext,
        render_context: &mut RenderContext<'w>,
        world: &'w World,
    ) -> Result<(), NodeRunError> {
        let Some(clock) = world.get_resource::<GpuClock>() else {
            return Ok(());
        };
        let encoder = render_context.command_encoder();
        // ★ THE SLOT OF A FRAME THREE FRAMES OLD is resolved here (RESOLVE_DELAY), never this frame's.
        if let Some((due, used)) = clock.due {
            let first = due as u32 * SLOT_QUERIES;
            let at = SLOT_BYTES * due as u64;
            encoder.resolve_query_set(
                &clock.queries,
                first..first + used,
                &clock.resolve_buffer,
                at,
            );
            encoder.copy_buffer_to_buffer(
                &clock.resolve_buffer,
                at,
                &clock.readback[due],
                0,
                u64::from(used) * 8,
            );
        }
        clock.ended.store(true, Ordering::Relaxed);
        let Some(slot) = clock.current else {
            return Ok(());
        };
        let first = slot as u32 * SLOT_QUERIES;
        let outputs: Vec<&TextureView> = self
            .views
            .iter_manual(world)
            .map(ViewTarget::out_texture)
            .collect();
        // Every camera's output is waited for; the last pass carries the stamp. No camera: the
        // clock's own picture carries it, and the pair reads the graph's bare cost.
        match outputs.split_last() {
            Some((last, rest)) => {
                for view in rest {
                    stamp_pass(encoder, view, &clock.queries, None, None);
                }
                stamp_pass(encoder, last, &clock.queries, None, Some(first + 1));
            }
            None => stamp_pass(encoder, &clock.own, &clock.queries, None, Some(first + 1)),
        }
        Ok(())
    }
}

/// After the frame's submission: ask for this frame's stamps, and read every slot that returned.
fn read_back(
    clock: Option<ResMut<GpuClock>>,
    reading: Res<GpuClockReading>,
    device: Res<RenderDevice>,
) {
    let Some(mut clock) = clock else {
        return;
    };
    let clock = &mut *clock;
    let ended = clock.ended.load(Ordering::Relaxed);
    // This frame's slot holds its stamps now; it is resolved RESOLVE_DELAY frames later.
    if let Some(slot) = clock.current.take()
        && ended
    {
        let used = 2 + clock
            .pass_count
            .load(Ordering::Relaxed)
            .min(MAX_PASS_STAMPS);
        let marks = clock
            .marks
            .get_mut()
            .map(std::mem::take)
            .unwrap_or_default();
        clock.slots[slot] = Slot::Recorded(clock.frame, used, marks);
    }
    // The slot this frame's end node resolved: its readback is asked for.
    if let Some((slot, _)) = clock.due.take()
        && ended
        && let Slot::Recorded(frame, _, marks) =
            std::mem::replace(&mut clock.slots[slot], Slot::Free)
    {
        let ready = Arc::new(AtomicU8::new(MAP_PENDING));
        let state = Arc::clone(&ready);
        clock.readback[slot]
            .slice(..)
            .map_async(wgpu::MapMode::Read, move |result| {
                // A failed mapping says so, so that its slot is used again (it never came back
                // before, and the clock stopped with it).
                let ended = if result.is_ok() { MAP_OK } else { MAP_FAILED };
                state.store(ended, Ordering::Release);
            });
        clock.slots[slot] = Slot::Mapping(frame, ready, marks);
    }
    // ★ POLL THE DEVICE ONCE A FRAME, without blocking, so the readbacks' callbacks run. MEASURED
    // (2026-09-29, the T0 window session): the window never polls the device, so a readback's
    // callback ran only when the driver happened to maintain the device. The clock's four slots
    // stayed out, and it recorded nothing for seconds: one reading stood in ALL 12 samples of the
    // session's control mode, and in 10 of 12 samples of the morning's all-looks mode (its 32.25 ms).
    // The headless capture polls the device every frame for its own picture copies, which is why its
    // readings were fresh.
    if let Err(e) = device.poll(wgpu::PollType::Poll) {
        tracing::warn!(%e, "VD_GPU_CLOCK — the device poll failed; the clock's readings may lag");
    }
    for slot in 0..SLOTS {
        let Slot::Mapping(frame, ready, _) = &clock.slots[slot] else {
            continue;
        };
        let frame = *frame;
        match ready.load(Ordering::Acquire) {
            MAP_OK => {}
            MAP_FAILED => {
                clock.health.failed += 1;
                clock.slots[slot] = Slot::Free;
                continue;
            }
            _ if clock.frame.saturating_sub(frame) > STUCK_FRAMES => {
                // Unmapping cancels the pending mapping; the slot is used again.
                clock.readback[slot].unmap();
                clock.health.stuck += 1;
                clock.slots[slot] = Slot::Free;
                continue;
            }
            _ => continue,
        }
        let Slot::Mapping(_, _, marks) = std::mem::replace(&mut clock.slots[slot], Slot::Free)
        else {
            continue;
        };
        let (begin, end, stamps) = {
            let bytes = clock.readback[slot].slice(..).get_mapped_range();
            let word = |i: usize| {
                let mut b = [0u8; 8];
                b.copy_from_slice(&bytes[8 * i..8 * i + 8]);
                u64::from_le_bytes(b)
            };
            let stamps: Vec<u64> = marks.iter().map(|m| word(2 + m.index as usize)).collect();
            (word(0), word(1), stamps)
        };
        clock.readback[slot].unmap();
        // A pair out of order (a stamp not written) is no reading.
        if end <= begin {
            clock.health.unordered += 1;
            clock.health.last_unordered = Some((frame, begin, end));
            continue;
        }
        clock.health.reads += 1;
        clock.health.last_good = Some((frame, begin, end));
        let ms = |ticks: u64| ticks as f64 * clock.period_ns / 1.0e6;
        reading.put(0, ms(end - begin));
        if let Some((last_frame, last_end)) = clock.last_end
            && last_frame + 1 == frame
            && end > last_end
        {
            reading.put(1, ms(end - last_end));
        }
        clock.last_end = Some((frame, end));
        if !marks.is_empty() {
            reading.put_passes(pass_readings(&marks, &stamps, clock.period_ns));
        }
    }
    let window_s = clock.health.since.elapsed().as_secs_f64();
    if window_s >= HEALTH_LOG_S {
        let h = &clock.health;
        tracing::info!(
            reads = h.reads,
            unordered = h.unordered,
            failed = h.failed,
            stuck = h.stuck,
            skipped = h.skipped,
            last_unordered = ?h.last_unordered,
            last_good = ?h.last_good,
            window_s,
            "VD_GPU_CLOCK health — readings delivered, pairs out of order, mappings failed and \
             stuck, frames with no free slot"
        );
        clock.health = ClockHealth::new();
    }
}

/// A frame's per-pass readings: each node's time is its stamp less the stamp before it in the same
/// camera's graph, added up by camera and node, and the sum of them all. A stamp that is not later
/// than the one before it (a stamp not written) gives no reading.
fn pass_readings(marks: &[PassMark], stamps: &[u64], period_ns: f64) -> PassReadings {
    let mut by_node: BTreeMap<String, f64> = BTreeMap::new();
    let mut sum = 0.0;
    let mut previous: Option<u64> = None;
    let mut in_order: Vec<(&PassMark, u64)> = marks.iter().zip(stamps.iter().copied()).collect();
    in_order.sort_by_key(|(mark, _)| mark.index);
    for (mark, t) in in_order {
        if let (Some(label), Some(p)) = (mark.after, previous)
            && t > p
        {
            let ms = (t - p) as f64 * period_ns / 1.0e6;
            *by_node
                .entry(format!("gpu_pass_c{}_{label:?}", mark.order))
                .or_insert(0.0) += ms;
            sum += ms;
        }
        previous = Some(t);
    }
    (by_node.into_iter().collect(), sum)
}

/// In the main world: the newest readings into the diagnostics store, once each. The system writes
/// the store alone: the engine's `Diagnostics` parameter reads the same store, and the engine refuses
/// the two together (MEASURED: the capture client stopped at its first frame). A node's per-pass
/// path joins the store the first time it reads, because the cameras' nodes are known only when
/// their graphs run.
fn publish(reading: Res<GpuClockReading>, mut store: ResMut<DiagnosticsStore>) {
    let now = Instant::now();
    if let Some(ms) = reading.take(0) {
        measure(&mut store, &GPU_FRAME_PATH, now, ms);
    }
    if let Some(ms) = reading.take(1) {
        measure(&mut store, &GPU_STEP_PATH, now, ms);
    }
    if let Some((passes, sum)) = reading.take_passes() {
        for (name, ms) in passes {
            let path = DiagnosticPath::new(format!("render/{name}/elapsed_gpu"));
            measure(&mut store, &path, now, ms);
        }
        measure(&mut store, &GPU_PASS_SUM_PATH, now, sum);
    }
}

/// One reading into its diagnostic, which joins the store the first time.
fn measure(store: &mut DiagnosticsStore, path: &DiagnosticPath, time: Instant, value: f64) {
    if store.get(path).is_none() {
        store.add(Diagnostic::new(path.clone()).with_suffix("ms"));
    }
    if let Some(diagnostic) = store.get_mut(path) {
        diagnostic.add_measurement(DiagnosticMeasurement { time, value });
    }
}
