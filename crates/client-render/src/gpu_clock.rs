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

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};

use bevy::diagnostic::{Diagnostic, DiagnosticPath, Diagnostics, RegisterDiagnostic};
use bevy::prelude::*;
use bevy::render::graph::CameraDriverLabel;
use bevy::render::render_graph::{
    Node, NodeRunError, RenderGraph, RenderGraphContext, RenderLabel,
};
use bevy::render::render_resource::{Buffer, TextureView};
use bevy::render::renderer::{RenderContext, RenderDevice, RenderQueue};
use bevy::render::view::ViewTarget;
use bevy::render::{Render, RenderApp, RenderSystems};

/// The switch: `VD_GPU_CLOCK=1`.
pub(crate) const GPU_CLOCK_ENV: &str = "VD_GPU_CLOCK";

/// Whether the clock runs.
pub(crate) fn gpu_clock() -> bool {
    std::env::var(GPU_CLOCK_ENV).is_ok_and(|v| v == "1")
}

/// How many frames' pairs may be in flight at once: a readback returns a few frames after its
/// frame, and a frame whose slot is still out records nothing.
const SLOTS: usize = 4;

/// The byte step between two slots in the resolve buffer: the resolve's destination offset must
/// be a multiple of it.
const RESOLVE_STEP: u64 = wgpu::QUERY_RESOLVE_BUFFER_ALIGNMENT;

/// Two timestamps, eight bytes each.
const PAIR_BYTES: u64 = 16;

/// The span from the first stamp to the second, in milliseconds.
pub(crate) const GPU_FRAME_PATH: DiagnosticPath =
    DiagnosticPath::const_new("render/gpu_frame/elapsed_gpu");

/// The time from one read frame's second stamp to the next frame's, in milliseconds.
pub(crate) const GPU_STEP_PATH: DiagnosticPath =
    DiagnosticPath::const_new("render/gpu_step/elapsed_gpu");

#[derive(Debug, Hash, PartialEq, Eq, Clone, RenderLabel)]
struct GpuClockBegin;

#[derive(Debug, Hash, PartialEq, Eq, Clone, RenderLabel)]
struct GpuClockEnd;

/// The newest readings, shared by the two worlds: the span and the step in milliseconds as `f64`
/// bits, and whether the main world has not yet taken each.
#[derive(Resource, Clone, Default)]
struct GpuClockReading(Arc<[(AtomicU64, AtomicBool); 2]>);

impl GpuClockReading {
    fn put(&self, which: usize, ms: f64) {
        self.0[which].0.store(ms.to_bits(), Ordering::Relaxed);
        self.0[which].1.store(true, Ordering::Release);
    }

    fn take(&self, which: usize) -> Option<f64> {
        self.0[which]
            .1
            .swap(false, Ordering::Acquire)
            .then(|| f64::from_bits(self.0[which].0.load(Ordering::Relaxed)))
    }
}

/// One slot's state.
enum Slot {
    /// Free to record into.
    Free,
    /// Its readback is out for the frame of this count; the flag turns when the pair can be read.
    Mapping(u64, Arc<AtomicBool>),
}

/// The clock in the render world.
#[derive(Resource)]
struct GpuClock {
    queries: wgpu::QuerySet,
    resolve: Buffer,
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
                count: (2 * SLOTS) as u32,
            });
        let resolve = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("vd_gpu_clock_resolve"),
            size: RESOLVE_STEP * SLOTS as u64,
            usage: wgpu::BufferUsages::QUERY_RESOLVE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let readback = (0..SLOTS)
            .map(|_| {
                device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some("vd_gpu_clock_readback"),
                    size: PAIR_BYTES,
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
                format: wgpu::TextureFormat::Rgba8Unorm,
                usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
                view_formats: &[],
            })
            .create_view(&wgpu::TextureViewDescriptor::default());
        render_app.insert_resource(GpuClock {
            queries,
            resolve,
            readback,
            own,
            slots: (0..SLOTS).map(|_| Slot::Free).collect(),
            current: None,
            ended: Arc::new(AtomicBool::new(false)),
            next: 0,
            frame: 0,
            last_end: None,
            period_ns,
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

/// Before the graph runs: this frame records into the next slot when it is free.
fn choose_slot(clock: Option<ResMut<GpuClock>>) {
    let Some(mut clock) = clock else {
        return;
    };
    clock.frame += 1;
    clock.ended.store(false, Ordering::Relaxed);
    let next = clock.next;
    clock.current = match clock.slots[next] {
        Slot::Free => {
            clock.next = (next + 1) % SLOTS;
            Some(next)
        }
        Slot::Mapping(..) => None,
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
            let first = (2 * slot) as u32;
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
        let Some(slot) = clock.current else {
            return Ok(());
        };
        let first = (2 * slot) as u32;
        let outputs: Vec<&TextureView> = self
            .views
            .iter_manual(world)
            .map(ViewTarget::out_texture)
            .collect();
        let encoder = render_context.command_encoder();
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
        let at = RESOLVE_STEP * slot as u64;
        encoder.resolve_query_set(&clock.queries, first..first + 2, &clock.resolve, at);
        encoder.copy_buffer_to_buffer(&clock.resolve, at, &clock.readback[slot], 0, PAIR_BYTES);
        clock.ended.store(true, Ordering::Relaxed);
        Ok(())
    }
}

/// After the frame's submission: ask for this frame's pair, and read every pair that returned.
fn read_back(clock: Option<ResMut<GpuClock>>, reading: Res<GpuClockReading>) {
    let Some(mut clock) = clock else {
        return;
    };
    let clock = &mut *clock;
    if let Some(slot) = clock.current.take()
        && clock.ended.load(Ordering::Relaxed)
    {
        let ready = Arc::new(AtomicBool::new(false));
        let flag = Arc::clone(&ready);
        clock.readback[slot]
            .slice(..)
            .map_async(wgpu::MapMode::Read, move |result| {
                if result.is_ok() {
                    flag.store(true, Ordering::Release);
                }
            });
        clock.slots[slot] = Slot::Mapping(clock.frame, ready);
    }
    for slot in 0..SLOTS {
        let Slot::Mapping(frame, ready) = &clock.slots[slot] else {
            continue;
        };
        if !ready.load(Ordering::Acquire) {
            continue;
        }
        let frame = *frame;
        let (begin, end) = {
            let bytes = clock.readback[slot].slice(..).get_mapped_range();
            let word = |i: usize| {
                let mut b = [0u8; 8];
                b.copy_from_slice(&bytes[8 * i..8 * i + 8]);
                u64::from_le_bytes(b)
            };
            (word(0), word(1))
        };
        clock.readback[slot].unmap();
        clock.slots[slot] = Slot::Free;
        // A pair out of order (a stamp not written) is no reading.
        if end <= begin {
            continue;
        }
        let ms = |ticks: u64| ticks as f64 * clock.period_ns / 1.0e6;
        reading.put(0, ms(end - begin));
        if let Some((last_frame, last_end)) = clock.last_end
            && last_frame + 1 == frame
            && end > last_end
        {
            reading.put(1, ms(end - last_end));
        }
        clock.last_end = Some((frame, end));
    }
}

/// In the main world: the newest readings into the diagnostics, once each.
fn publish(reading: Res<GpuClockReading>, mut diagnostics: Diagnostics) {
    if let Some(ms) = reading.take(0) {
        diagnostics.add_measurement(&GPU_FRAME_PATH, || ms);
    }
    if let Some(ms) = reading.take(1) {
        diagnostics.add_measurement(&GPU_STEP_PATH, || ms);
    }
}
