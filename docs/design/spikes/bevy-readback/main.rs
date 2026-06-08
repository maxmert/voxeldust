//! THROWAWAY de-risking spike for Voxeldust P1.5 Slice 3 (renderer).
//!
//! ONE question, on THIS machine (Apple Silicon / macOS / Metal): can Bevy 0.18
//! give us the wgpu readback the HR6 capture pipeline needs?
//!
//! It must have 4 properties, each checked as a structural assertion on the
//! read-back RGBA8 bytes:
//!   1. OFFSCREEN render-to-image -> CPU readback, 256-byte row padding stripped.
//!   2. HEADLESS (no WinitPlugin, ScheduleRunnerPlugin, RenderTarget::Image).
//!   3. POST-egui composite: the readback INCLUDES a bevy_egui panel (issue #16689).
//!   4. Apple-Silicon correctness: a known-color quad at a known pixel reads back
//!      uncorrupted, no offset (wgpu#6827 is upload-only; validate readback anyway).
//!
//! Structure is the canonical Bevy 0.18 `headless_renderer` example (render-graph
//! copy node reading the camera's RenderTarget::Image), with bevy_egui drawing
//! INTO that same camera's image target via EguiContext + EguiMultipassSchedule,
//! so egui pixels are part of the camera pass that the copy node reads back. This
//! deliberately AVOIDS the `Screenshot` component path that #16689 is about.

use bevy::{
    app::{AppExit, ScheduleRunnerPlugin},
    camera::RenderTarget,
    core_pipeline::tonemapping::Tonemapping,
    ecs::schedule::ScheduleLabel,
    image::TextureFormatPixelInfo,
    prelude::*,
    render::{
        render_asset::RenderAssets,
        render_graph::{self, NodeRunError, RenderGraph, RenderGraphContext, RenderLabel},
        render_resource::{
            Buffer, BufferDescriptor, BufferUsages, CommandEncoderDescriptor, Extent3d, MapMode,
            PollType, TexelCopyBufferInfo, TexelCopyBufferLayout, TextureFormat, TextureUsages,
        },
        renderer::{RenderContext, RenderDevice, RenderQueue},
        Extract, Render, RenderApp, RenderSystems,
    },
    window::ExitCondition,
    winit::WinitPlugin,
};
use bevy_egui::{
    EguiContext, EguiGlobalSettings, EguiMultipassSchedule, EguiPlugin, PrimaryEguiContext,
};
use crossbeam_channel::{Receiver, Sender};
use std::{
    sync::{
        atomic::{AtomicBool, Ordering},
        Arc,
    },
    time::Duration,
};

// ---- spike parameters (modest size = fast readback) ----------------------------
// Width chosen so width*4 is NOT a multiple of 256: 500*4 = 2000, padded to 2048
// => 48 bytes of row padding. This FORCES the 256-byte padding-strip code path
// (the "#1 readback bug"). Use an odd-ish height too.
const WIDTH: u32 = 500;
const HEIGHT: u32 = 480;
// Non-magenta clear color so the NO-MAGENTA check is meaningful. Dark teal.
const CLEAR_R: u8 = 12;
const CLEAR_G: u8 = 30;
const CLEAR_B: u8 = 40;
// The known quad: a solid red sprite centered at a known world (== screen) position.
const QUAD_R: u8 = 220;
const QUAD_G: u8 = 20;
const QUAD_B: u8 = 20;
const QUAD_SIZE: f32 = 120.0;
// Number of frames to render before reading back (let the render world warm up).
const PRE_ROLL: u32 = 8;

/// Custom schedule label for the offscreen-camera egui context (multi-pass mode).
#[derive(ScheduleLabel, Clone, Debug, PartialEq, Eq, Hash)]
struct OffscreenEguiPass;

fn main() {
    App::new()
        .insert_resource(ClearColor(Color::srgb_u8(CLEAR_R, CLEAR_G, CLEAR_B)))
        .add_plugins(
            DefaultPlugins
                .set(WindowPlugin {
                    primary_window: None,
                    exit_condition: ExitCondition::DontExit,
                    ..default()
                })
                // WinitPlugin would create/require a window/display server — headless.
                .disable::<WinitPlugin>(),
        )
        .add_plugins(EguiPlugin::default())
        .add_plugins(ImageCopyPlugin)
        .add_plugins(ScheduleRunnerPlugin::run_loop(Duration::from_secs_f64(
            1.0 / 60.0,
        )))
        .init_resource::<ReadbackState>()
        // Headless: no window => no primary egui context. Don't let the plugin try.
        .add_systems(PreStartup, disable_primary_egui_context)
        .add_systems(Startup, setup)
        .add_systems(OffscreenEguiPass, draw_egui_panel)
        .add_systems(Update, tick_and_readback)
        .run();
}

#[derive(Resource, Default)]
struct ReadbackState {
    frames: u32,
    done: bool,
}

fn disable_primary_egui_context(mut settings: ResMut<EguiGlobalSettings>) {
    settings.auto_create_primary_context = false;
}

/// Channel resources (render world -> main world), as in the headless example.
#[derive(Resource, Deref)]
struct MainWorldReceiver(Receiver<Vec<u8>>);
#[derive(Resource, Deref)]
struct RenderWorldSender(Sender<Vec<u8>>);

/// Marker holding the handle of the offscreen render target image (main world).
#[derive(Resource)]
struct RenderTargetImage(Handle<Image>);

fn setup(mut commands: Commands, mut images: ResMut<Assets<Image>>, render_device: Res<RenderDevice>) {
    let size = Extent3d {
        width: WIDTH,
        height: HEIGHT,
        ..default()
    };

    // The texture the camera renders into.
    let mut render_target_image =
        Image::new_target_texture(WIDTH, HEIGHT, TextureFormat::bevy_default(), None);
    render_target_image.texture_descriptor.usage |= TextureUsages::COPY_SRC;
    let image_handle = images.add(render_target_image);
    commands.insert_resource(RenderTargetImage(image_handle.clone()));

    // The render-graph copy node copies this image -> a CPU-mappable buffer each frame.
    commands.spawn(ImageCopier::new(image_handle.clone(), size, &render_device));

    // A 2D camera rendering to the offscreen image. egui draws into the SAME image
    // via EguiContext + EguiMultipassSchedule(OffscreenEguiPass), so the readback
    // composites scene + egui. PrimaryEguiContext is NOT used (no window).
    commands.spawn((
        Camera2d,
        Camera {
            clear_color: ClearColorConfig::Custom(Color::srgb_u8(CLEAR_R, CLEAR_G, CLEAR_B)),
            ..default()
        },
        // In Bevy 0.18 RenderTarget is a SEPARATE component, not a Camera field.
        RenderTarget::Image(image_handle.into()),
        Tonemapping::None,
        // EguiMultipassSchedule causes bevy_egui to create+manage an EguiContext on
        // this entity and render its passes into THIS camera's image target. No
        // RenderLayers::none() here: we WANT the 2D scene rendered too, with egui
        // composited on top (that is the post-egui composite we are validating).
        EguiMultipassSchedule::new(OffscreenEguiPass),
    ));

    // Known-pattern quad: solid red, centered at screen center (world origin for a
    // default 2D camera). Used for the known-pixel correctness check.
    commands.spawn((
        Sprite {
            color: Color::srgb_u8(QUAD_R, QUAD_G, QUAD_B),
            custom_size: Some(Vec2::splat(QUAD_SIZE)),
            ..default()
        },
        Transform::from_xyz(0.0, 0.0, 0.0),
    ));
}

/// Draws ONE egui panel into the offscreen camera's egui context.
/// A bottom panel with the default egui dark fill, near the bottom edge — a region
/// where the scene clear color would otherwise dominate, so the EGUI-PRESENT check
/// is unambiguous.
fn draw_egui_panel(mut ctx: Single<&mut EguiContext, Without<PrimaryEguiContext>>) {
    let ctx = ctx.get_mut();
    bevy_egui::egui::TopBottomPanel::bottom("debug")
        .min_height(80.0)
        .show(ctx, |ui| {
            ui.heading("VOXELDUST DEBUG PANEL");
            ui.label("HR6 capture smoke — egui composite check");
            ui.label(format!("res {WIDTH}x{HEIGHT}"));
        });
}

fn tick_and_readback(
    mut state: ResMut<ReadbackState>,
    receiver: Res<MainWorldReceiver>,
    target: Res<RenderTargetImage>,
    images: Res<Assets<Image>>,
    mut exit: MessageWriter<AppExit>,
) {
    if state.done {
        return;
    }
    state.frames += 1;

    // Drain whatever the render world has sent so we use the latest frame.
    let mut latest: Option<Vec<u8>> = None;
    while let Ok(data) = receiver.try_recv() {
        latest = Some(data);
    }

    if state.frames < PRE_ROLL {
        return;
    }

    let Some(raw) = latest else {
        return; // not ready yet, keep ticking
    };

    // ---- strip the 256-byte COPY_BYTES_PER_ROW_ALIGNMENT padding ----------------
    let img = images.get(&target.0).expect("target image present");
    let pixel_size = img.texture_descriptor.format.pixel_size().unwrap(); // 4 for RGBA8
    let row_bytes = WIDTH as usize * pixel_size; // unpadded width*4
    let aligned_row_bytes = RenderDevice::align_copy_bytes_per_row(row_bytes); // padded to 256
    let unpadded: Vec<u8> = if row_bytes == aligned_row_bytes {
        raw.clone()
    } else {
        raw.chunks(aligned_row_bytes)
            .take(HEIGHT as usize)
            .flat_map(|row| &row[..row_bytes.min(row.len())])
            .copied()
            .collect()
    };

    // sanity: image format reported by bevy
    println!("---- BEVY 0.18 HEADLESS READBACK SPIKE ----");
    println!(
        "target image format = {:?}, pixel_size = {} bytes",
        img.texture_descriptor.format, pixel_size
    );
    println!(
        "raw buffer len = {} bytes; padded row = {} bytes; unpadded row = {} bytes; padding/row = {} bytes",
        raw.len(),
        aligned_row_bytes,
        row_bytes,
        aligned_row_bytes - row_bytes
    );
    println!(
        "unpadded len = {} bytes (expected {} = {}x{}x4)",
        unpadded.len(),
        WIDTH as usize * HEIGHT as usize * 4,
        WIDTH,
        HEIGHT
    );
    assert_eq!(
        unpadded.len(),
        WIDTH as usize * HEIGHT as usize * 4,
        "PROP1 FAIL: unpadded length wrong — padding strip is broken"
    );

    run_assertions(&unpadded, img.texture_descriptor.format);

    // Also dump the PNG so a human can eyeball it.
    if let Ok(dyn_img) = {
        let mut copy = img.clone();
        copy.data = Some(unpadded.clone());
        copy.try_into_dynamic()
    } {
        let _ = dyn_img.to_rgba8().save("spike_readback.png");
        println!("[saved spike_readback.png]");
    }

    state.done = true;
    exit.write(AppExit::Success);
}

/// The four G-RENDER-SMOKE structural checks. `fmt` tells us if bevy chose an sRGB
/// format (Rgba8UnormSrgb) so we interpret the bytes correctly for the known-color
/// check (the framebuffer stores sRGB-encoded bytes; our input colors were srgb_u8,
/// so the stored bytes should be ~ the same u8 values, modulo rounding).
fn run_assertions(px: &[u8], fmt: TextureFormat) {
    let w = WIDTH as usize;
    let h = HEIGHT as usize;
    let at = |x: usize, y: usize| -> (u8, u8, u8, u8) {
        let i = (y * w + x) * 4;
        (px[i], px[i + 1], px[i + 2], px[i + 3])
    };

    // (a) NO-MAGENTA: count exact (255,0,255,*) pixels.
    let mut magenta = 0usize;
    for p in px.chunks_exact(4) {
        if p[0] == 255 && p[1] == 0 && p[2] == 255 {
            magenta += 1;
        }
    }
    println!("(a) NO-MAGENTA   : magenta pixels = {magenta}  -> {}", pf(magenta == 0));

    // (b) CONTENT-PRESENT: % of pixels differing from clear color > threshold.
    // Tolerance because sRGB framebuffer rounding can nudge the clear bytes by ~1.
    let mut differ = 0usize;
    for p in px.chunks_exact(4) {
        let d = (p[0] as i32 - CLEAR_R as i32).abs()
            + (p[1] as i32 - CLEAR_G as i32).abs()
            + (p[2] as i32 - CLEAR_B as i32).abs();
        if d > 12 {
            differ += 1;
        }
    }
    let pct = differ as f64 / (w * h) as f64 * 100.0;
    println!(
        "(b) CONTENT      : {differ} px ({pct:.2}%) differ from clear  -> {}",
        pf(pct > 1.0)
    );

    // (c) EGUI-PRESENT (#16689 check): the egui bottom panel occupies roughly the
    // bottom ~80 px. Egui's default dark panel fill is a near-uniform dark gray that
    // is NOT our teal clear color. Detect panel-like pixels in the bottom band.
    let band_top = h.saturating_sub(70);
    let mut egui_like = 0usize;
    let mut band_total = 0usize;
    for y in band_top..h {
        for x in 0..w {
            band_total += 1;
            let (r, g, b, _) = at(x, y);
            // egui dark panel: gray-ish, r≈g≈b, and clearly not the teal clear (which
            // has b noticeably > r). Require near-neutral gray and brightness above clear.
            let near_gray = (r as i32 - g as i32).abs() < 16 && (g as i32 - b as i32).abs() < 16;
            let not_clear = (r as i32 - CLEAR_R as i32).abs()
                + (g as i32 - CLEAR_G as i32).abs()
                + (b as i32 - CLEAR_B as i32).abs()
                > 18;
            if near_gray && not_clear {
                egui_like += 1;
            }
        }
    }
    let egui_pct = if band_total > 0 {
        egui_like as f64 / band_total as f64 * 100.0
    } else {
        0.0
    };
    println!(
        "(c) EGUI-PRESENT : {egui_like}/{band_total} px ({egui_pct:.2}%) egui-panel-like in bottom band  -> {}",
        pf(egui_pct > 30.0)
    );

    // (d) KNOWN-PIXEL: the red quad covers QUAD_SIZE px centered at screen center.
    // Sample the exact center pixel; it must read back ~red, no offset/corruption.
    let (cr, cg, cb, ca) = at(w / 2, h / 2);
    // sRGB-stored bytes of an srgb_u8 input ≈ the same u8 (identity sRGB roundtrip).
    let red_ok = cr > 180 && cg < 70 && cb < 70 && ca > 200;
    println!(
        "(d) KNOWN-PIXEL  : center=({cr},{cg},{cb},{ca}) expected ~({QUAD_R},{QUAD_G},{QUAD_B},255)  fmt={fmt:?}  -> {}",
        pf(red_ok)
    );

    // Corner pixel should be the clear color (proves no horizontal/vertical offset).
    let (tr, tg, tb, _) = at(2, 2);
    let corner_ok = (tr as i32 - CLEAR_R as i32).abs() < 16
        && (tg as i32 - CLEAR_G as i32).abs() < 16
        && (tb as i32 - CLEAR_B as i32).abs() < 16;
    println!(
        "    corner(2,2) = ({tr},{tg},{tb}) expected ~({CLEAR_R},{CLEAR_G},{CLEAR_B}) [no-offset]  -> {}",
        pf(corner_ok)
    );

    let all = magenta == 0 && pct > 1.0 && egui_pct > 30.0 && red_ok && corner_ok;
    println!("==== OVERALL: {} ====", if all { "ALL PASS" } else { "SOME FAIL" });
}

fn pf(b: bool) -> &'static str {
    if b {
        "PASS"
    } else {
        "FAIL"
    }
}

// ================= render-world image copy (from headless_renderer example) =======

struct ImageCopyPlugin;
impl Plugin for ImageCopyPlugin {
    fn build(&self, app: &mut App) {
        let (s, r) = crossbeam_channel::unbounded();
        let render_app = app
            .insert_resource(MainWorldReceiver(r))
            .sub_app_mut(RenderApp);

        let mut graph = render_app.world_mut().resource_mut::<RenderGraph>();
        graph.add_node(ImageCopyLabel, ImageCopyDriver);
        graph.add_node_edge(bevy::render::graph::CameraDriverLabel, ImageCopyLabel);

        render_app
            .insert_resource(RenderWorldSender(s))
            .add_systems(ExtractSchedule, image_copy_extract)
            .add_systems(
                Render,
                receive_image_from_buffer.after(RenderSystems::Render),
            );
    }
}

#[derive(Clone, Default, Resource, Deref, DerefMut)]
struct ImageCopiers(Vec<ImageCopier>);

#[derive(Clone, Component)]
struct ImageCopier {
    buffer: Buffer,
    enabled: Arc<AtomicBool>,
    src_image: Handle<Image>,
}

impl ImageCopier {
    fn new(src_image: Handle<Image>, size: Extent3d, render_device: &RenderDevice) -> ImageCopier {
        let padded_bytes_per_row =
            RenderDevice::align_copy_bytes_per_row((size.width) as usize) * 4;
        let cpu_buffer = render_device.create_buffer(&BufferDescriptor {
            label: None,
            size: padded_bytes_per_row as u64 * size.height as u64,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        ImageCopier {
            buffer: cpu_buffer,
            src_image,
            enabled: Arc::new(AtomicBool::new(true)),
        }
    }
    fn enabled(&self) -> bool {
        self.enabled.load(Ordering::Relaxed)
    }
}

fn image_copy_extract(mut commands: Commands, image_copiers: Extract<Query<&ImageCopier>>) {
    commands.insert_resource(ImageCopiers(image_copiers.iter().cloned().collect()));
}

#[derive(Debug, PartialEq, Eq, Clone, Hash, RenderLabel)]
struct ImageCopyLabel;

#[derive(Default)]
struct ImageCopyDriver;

impl render_graph::Node for ImageCopyDriver {
    fn run(
        &self,
        _graph: &mut RenderGraphContext,
        render_context: &mut RenderContext,
        world: &World,
    ) -> Result<(), NodeRunError> {
        let image_copiers = world.get_resource::<ImageCopiers>().unwrap();
        let gpu_images = world
            .get_resource::<RenderAssets<bevy::render::texture::GpuImage>>()
            .unwrap();

        for image_copier in image_copiers.iter() {
            if !image_copier.enabled() {
                continue;
            }
            let Some(src_image) = gpu_images.get(&image_copier.src_image) else {
                continue;
            };

            let mut encoder = render_context
                .render_device()
                .create_command_encoder(&CommandEncoderDescriptor::default());

            let block_dimensions = src_image.texture_format.block_dimensions();
            let block_size = src_image.texture_format.block_copy_size(None).unwrap();
            let padded_bytes_per_row = RenderDevice::align_copy_bytes_per_row(
                (src_image.size.width as usize / block_dimensions.0 as usize) * block_size as usize,
            );

            encoder.copy_texture_to_buffer(
                src_image.texture.as_image_copy(),
                TexelCopyBufferInfo {
                    buffer: &image_copier.buffer,
                    layout: TexelCopyBufferLayout {
                        offset: 0,
                        bytes_per_row: Some(
                            std::num::NonZero::<u32>::new(padded_bytes_per_row as u32)
                                .unwrap()
                                .into(),
                        ),
                        rows_per_image: None,
                    },
                },
                src_image.size,
            );

            let render_queue = world.get_resource::<RenderQueue>().unwrap();
            render_queue.submit(std::iter::once(encoder.finish()));
        }
        Ok(())
    }
}

fn receive_image_from_buffer(
    image_copiers: Res<ImageCopiers>,
    render_device: Res<RenderDevice>,
    sender: Res<RenderWorldSender>,
) {
    for image_copier in image_copiers.0.iter() {
        if !image_copier.enabled() {
            continue;
        }
        let buffer_slice = image_copier.buffer.slice(..);
        let (s, r) = crossbeam_channel::bounded(1);
        buffer_slice.map_async(MapMode::Read, move |res| match res {
            Ok(res) => s.send(res).expect("Failed to send map update"),
            Err(err) => panic!("Failed to map buffer {err}"),
        });
        render_device
            .poll(PollType::wait_indefinitely())
            .expect("Failed to poll device for map async");
        r.recv().expect("Failed to receive the map_async message");
        let _ = sender.send(buffer_slice.get_mapped_range().to_vec());
        image_copier.buffer.unmap();
    }
}
