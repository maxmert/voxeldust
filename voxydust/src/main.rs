//! Voxydust — first-person game client with server-authoritative movement.

mod mesh;
mod network;

use std::net::SocketAddr;
use std::sync::Arc;

use clap::Parser;
use glam::{DQuat, DVec3, Mat4, Vec3};
use tokio::sync::mpsc;
use tracing::info;
use winit::application::ApplicationHandler;
use winit::event::{DeviceEvent, DeviceId, WindowEvent};
use winit::event_loop::{ActiveEventLoop, EventLoop};
use winit::keyboard::{KeyCode, PhysicalKey};
use winit::window::{CursorGrabMode, Window, WindowAttributes, WindowId};

use voxeldust_core::client_message::{PlayerInputData, WorldStateData};

use crate::mesh::IcoSphere;
use crate::network::NetEvent;

#[derive(Parser, Debug)]
#[command(name = "voxydust", about = "Voxydust game client")]
struct Args {
    #[arg(long, default_value = "127.0.0.1:7777")]
    gateway: SocketAddr,
    #[arg(long, default_value = "Player")]
    name: String,
    #[arg(long)]
    direct: Option<String>,
}

const MAX_OBJECTS: usize = 256;
// Eye height offset from player position. Player capsule center is at ~1.0m,
// so eye is at capsule center + 0.5m (roughly head height of 1.5m total).
const EYE_HEIGHT: f64 = 0.5;

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct ObjectUniforms {
    mvp: [[f32; 4]; 4],
    color: [f32; 4],
    _pad0: [f32; 4], _pad1: [f32; 4], _pad2: [f32; 4], _pad3: [f32; 4],
    _pad4: [f32; 4], _pad5: [f32; 4], _pad6: [f32; 4], _pad7: [f32; 4],
    _pad8: [f32; 4], _pad9: [f32; 4], _pad10: [f32; 4],
}
const _: () = assert!(std::mem::size_of::<ObjectUniforms>() == 256);

struct App {
    args: Args,
    window: Option<Arc<Window>>,
    gpu: Option<GpuState>,
    net_event_rx: Option<mpsc::UnboundedReceiver<NetEvent>>,
    input_tx: Option<mpsc::UnboundedSender<PlayerInputData>>,

    // Game state from server.
    latest_world_state: Option<WorldStateData>,
    player_position: DVec3,
    player_velocity: DVec3,
    current_shard_type: u8,
    reference_position: DVec3,
    reference_rotation: DQuat,

    // Camera (first-person, follows player position).
    camera_yaw: f64,
    camera_pitch: f64,
    /// Pilot yaw/pitch rate: -1.0 to 1.0, set by mouse movement, decays to 0.
    /// Sent every tick as the desired turn rate. No accumulation needed.
    pilot_yaw_rate: f64,
    pilot_pitch_rate: f64,
    /// Ship rotation from WorldState (used as camera heading when piloting).
    ship_rotation: DQuat,
    /// Whether the player is currently piloting (from WorldState grounded flag).
    is_piloting: bool,

    // Input state.
    keys_held: std::collections::HashSet<KeyCode>,
    mouse_grabbed: bool,
    connected: bool,
    frame_count: u64,
}

struct GpuState {
    surface: wgpu::Surface<'static>,
    device: wgpu::Device,
    queue: wgpu::Queue,
    config: wgpu::SurfaceConfiguration,
    pipeline: wgpu::RenderPipeline,
    sphere_vertex_buf: wgpu::Buffer,
    sphere_index_buf: wgpu::Buffer,
    sphere_index_count: u32,
    box_vertex_buf: wgpu::Buffer,
    box_index_buf: wgpu::Buffer,
    box_index_count: u32,
    depth_view: wgpu::TextureView,
    uniform_buf: wgpu::Buffer,
    bind_group: wgpu::BindGroup,
    egui_ctx: egui::Context,
    egui_winit: egui_winit::State,
    egui_renderer: egui_wgpu::Renderer,
}

impl App {
    fn new(args: Args) -> Self {
        Self {
            args,
            window: None,
            gpu: None,
            net_event_rx: None,
            input_tx: None,
            latest_world_state: None,
            player_position: DVec3::new(0.0, 1.0, 0.0),
            player_velocity: DVec3::ZERO,
            current_shard_type: 255,
            reference_position: DVec3::ZERO,
            reference_rotation: DQuat::IDENTITY,
            camera_yaw: 0.0,
            camera_pitch: 0.0,
            pilot_yaw_rate: 0.0,
            pilot_pitch_rate: 0.0,
            ship_rotation: DQuat::IDENTITY,
            is_piloting: false,
            keys_held: std::collections::HashSet::new(),
            mouse_grabbed: false,
            connected: false,
            frame_count: 0,
        }
    }

    fn init_gpu(&mut self, window: Arc<Window>) {
        let size = window.inner_size();
        let instance = wgpu::Instance::default();
        let surface = instance.create_surface(window.clone()).unwrap();

        let (adapter, device, queue) = pollster::block_on(async {
            let adapter = instance.request_adapter(&wgpu::RequestAdapterOptions {
                compatible_surface: Some(&surface), ..Default::default()
            }).await.expect("no GPU adapter");
            info!(adapter = ?adapter.get_info().name, "GPU");
            let (device, queue) = adapter.request_device(&wgpu::DeviceDescriptor::default(), None)
                .await.expect("no device");
            (adapter, device, queue)
        });

        let caps = surface.get_capabilities(&adapter);
        let format = caps.formats[0];
        let config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT, format,
            width: size.width.max(1), height: size.height.max(1),
            present_mode: wgpu::PresentMode::AutoVsync,
            alpha_mode: caps.alpha_modes[0], view_formats: vec![],
            desired_maximum_frame_latency: 2,
        };
        surface.configure(&device, &config);

        // Uniform buffer.
        let uniform_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("uniforms"), size: 256 * MAX_OBJECTS as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: None, entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0, visibility: wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform, has_dynamic_offset: true,
                    min_binding_size: wgpu::BufferSize::new(256),
                }, count: None,
            }],
        });

        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None, layout: &bind_group_layout, entries: &[wgpu::BindGroupEntry {
                binding: 0, resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                    buffer: &uniform_buf, offset: 0, size: wgpu::BufferSize::new(256),
                }),
            }],
        });

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("shader"), source: wgpu::ShaderSource::Wgsl(include_str!("sphere.wgsl").into()),
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: None, bind_group_layouts: &[&bind_group_layout], push_constant_ranges: &[],
        });

        let depth_format = wgpu::TextureFormat::Depth32Float;
        let depth_view = create_depth_texture(&device, config.width, config.height, depth_format);

        let vertex_layout = wgpu::VertexBufferLayout {
            array_stride: 12, step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &[wgpu::VertexAttribute { format: wgpu::VertexFormat::Float32x3, offset: 0, shader_location: 0 }],
        };

        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("pipeline"), layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader, entry_point: Some("vs_main"), buffers: &[vertex_layout],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader, entry_point: Some("fs_main"),
                targets: &[Some(wgpu::ColorTargetState { format, blend: Some(wgpu::BlendState::REPLACE), write_mask: wgpu::ColorWrites::ALL })],
                compilation_options: Default::default(),
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                front_face: wgpu::FrontFace::Ccw,
                cull_mode: Some(wgpu::Face::Back),
                ..Default::default()
            },
            depth_stencil: Some(wgpu::DepthStencilState {
                format: depth_format, depth_write_enabled: true,
                depth_compare: wgpu::CompareFunction::Less,
                stencil: Default::default(), bias: Default::default(),
            }),
            multisample: Default::default(), multiview: None, cache: None,
        });

        // Sphere mesh.
        let sphere = IcoSphere::generate(4);
        use wgpu::util::DeviceExt;
        let sphere_vertex_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("sphere_vb"), contents: bytemuck::cast_slice(&sphere.vertices),
            usage: wgpu::BufferUsages::VERTEX,
        });
        let sphere_index_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("sphere_ib"), contents: bytemuck::cast_slice(&sphere.indices),
            usage: wgpu::BufferUsages::INDEX,
        });

        // Box mesh (ship interior: 6 faces, 12 triangles).
        let (box_verts, box_idxs) = generate_box_mesh(4.0, 3.0, 8.0);
        let box_vertex_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("box_vb"), contents: bytemuck::cast_slice(&box_verts),
            usage: wgpu::BufferUsages::VERTEX,
        });
        let box_index_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("box_ib"), contents: bytemuck::cast_slice(&box_idxs),
            usage: wgpu::BufferUsages::INDEX,
        });

        // egui.
        let egui_ctx = egui::Context::default();
        let egui_winit = egui_winit::State::new(egui_ctx.clone(), egui::ViewportId::ROOT, &window, Some(window.scale_factor() as f32), None, None);
        let egui_renderer = egui_wgpu::Renderer::new(&device, format, None, 1, false);

        self.window = Some(window);
        self.gpu = Some(GpuState {
            surface, device, queue, config, pipeline,
            sphere_vertex_buf, sphere_index_buf, sphere_index_count: sphere.indices.len() as u32,
            box_vertex_buf, box_index_buf, box_index_count: box_idxs.len() as u32,
            depth_view, uniform_buf, bind_group,
            egui_ctx, egui_winit, egui_renderer,
        });
    }

    fn start_networking(&mut self) {
        let (event_tx, event_rx) = mpsc::unbounded_channel();
        let (input_tx, input_rx) = mpsc::unbounded_channel();
        let input_rx = Arc::new(tokio::sync::Mutex::new(input_rx));

        let gateway = self.args.gateway;
        let name = self.args.name.clone();
        let direct = self.args.direct.clone();

        std::thread::spawn(move || {
            let rt = tokio::runtime::Runtime::new().unwrap();
            rt.block_on(network::run_network(gateway, name, event_tx, input_rx, direct));
        });

        self.net_event_rx = Some(event_rx);
        self.input_tx = Some(input_tx);
    }

    fn poll_network(&mut self) {
        if let Some(rx) = &mut self.net_event_rx {
            while let Ok(event) = rx.try_recv() {
                match event {
                    NetEvent::WorldState(ws) => {
                        // Update player position from server.
                        if let Some(p) = ws.players.first() {
                            self.player_position = p.position;
                            self.player_velocity = p.velocity;
                            let was_piloting = self.is_piloting;
                            self.is_piloting = !p.grounded;

                            if was_piloting != self.is_piloting {
                                info!(piloting = self.is_piloting, grounded = p.grounded, frame = self.frame_count, "pilot mode changed");
                            }

                            if self.is_piloting {
                                self.ship_rotation = p.rotation;

                                // On first frame of piloting: sync camera yaw to ship heading
                                // so there's no visual snap.
                                if !was_piloting {
                                    let fwd = p.rotation * DVec3::NEG_Z;
                                    self.camera_yaw = fwd.z.atan2(fwd.x) as f64;
                                    self.camera_pitch = fwd.y.asin() as f64;
                                }
                            }
                        }
                        if self.latest_world_state.is_none() {
                            info!(bodies = ws.bodies.len(), tick = ws.tick, "first WorldState");
                        }
                        self.latest_world_state = Some(ws);
                    }
                    NetEvent::Connected { shard_type, reference_position, reference_rotation, .. } => {
                        self.current_shard_type = shard_type;
                        self.reference_position = reference_position;
                        self.reference_rotation = reference_rotation;
                        self.connected = true;
                        let shard_name = match shard_type { 0 => "Planet", 1 => "System", 2 => "Ship", _ => "?" };
                        info!(shard_name, "connected to shard");
                    }
                    NetEvent::Disconnected(reason) => {
                        info!(%reason, "disconnected");
                        self.connected = false;
                    }
                    NetEvent::Transitioning => {
                        info!("transitioning to new shard...");
                    }
                }
            }
        }
    }

    fn send_input(&mut self) {
        if let Some(tx) = &self.input_tx {
            let mut movement = [0.0f32; 3];
            if self.keys_held.contains(&KeyCode::KeyW) { movement[2] += 1.0; }
            if self.keys_held.contains(&KeyCode::KeyS) { movement[2] -= 1.0; }
            if self.keys_held.contains(&KeyCode::KeyD) { movement[0] += 1.0; }
            if self.keys_held.contains(&KeyCode::KeyA) { movement[0] -= 1.0; }

            let jump = self.keys_held.contains(&KeyCode::Space);
            let action = if self.keys_held.contains(&KeyCode::KeyE) { 3 } else { 0 };
            let free_look = self.keys_held.contains(&KeyCode::AltLeft);

            // When piloting: send yaw/pitch rate (-1 to 1) for ship torque.
            // When walking: send absolute camera yaw/pitch.
            let (look_yaw, look_pitch) = if self.is_piloting && !free_look {
                (self.pilot_yaw_rate as f32, self.pilot_pitch_rate as f32)
            } else {
                (self.camera_yaw as f32, self.camera_pitch as f32)
            };

            // Decay pilot rates toward zero (virtual spring centering).
            // 0.95 per frame at 60fps = half-life ~0.23s. Feels responsive but not twitchy.
            self.pilot_yaw_rate *= 0.95;
            self.pilot_pitch_rate *= 0.95;

            let _ = tx.send(PlayerInputData {
                movement,
                look_yaw,
                look_pitch,
                jump,
                fly_toggle: false,
                speed_tier: 0,
                action,
                block_type: 0,
                tick: self.frame_count,
            });
        }
    }

    fn render(&mut self) {
        self.poll_network();
        self.send_input();
        self.frame_count += 1;

        let gpu = match &mut self.gpu { Some(g) => g, None => return };

        let frame = match gpu.surface.get_current_texture() { Ok(f) => f, Err(_) => return };
        let view = frame.texture.create_view(&wgpu::TextureViewDescriptor::default());

        // First-person camera. Player position is in ship-local coords when in a ship.
        // Rotate into world space for consistent rendering with the view matrix.
        let free_look = self.keys_held.contains(&KeyCode::AltLeft);
        let cam_pos = if self.is_piloting || self.current_shard_type == 2 {
            // Ship: player_position is ship-local. Rotate by ship_rotation to get world-relative.
            let player_local = self.player_position + DVec3::new(0.0, EYE_HEIGHT, 0.0);
            self.ship_rotation * player_local
        } else {
            // Planet/System: player_position is already in the shard's world frame.
            self.player_position + DVec3::new(0.0, EYE_HEIGHT, 0.0)
        };

        let cam_fwd = if self.is_piloting && !free_look {
            // Camera locked to ship heading.
            let fwd = self.ship_rotation * DVec3::NEG_Z;
            let f = fwd.as_vec3().normalize();
            if self.frame_count % 60 == 0 {
                info!(
                    "LOCKED cam: fwd=({:.3},{:.3},{:.3}) rot=({:.3},{:.3},{:.3},{:.3})",
                    f.x, f.y, f.z,
                    self.ship_rotation.x, self.ship_rotation.y, self.ship_rotation.z, self.ship_rotation.w
                );
            }
            f
        } else {
            // Free camera (walking or free-look).
            let (sy, cy) = (self.camera_yaw as f32).sin_cos();
            let (sp, cp) = (self.camera_pitch as f32).sin_cos();
            let f = Vec3::new(cy * cp, sp, sy * cp).normalize();
            if self.frame_count % 60 == 0 {
                info!(
                    "FREE cam: yaw={:.3} pitch={:.3} fwd=({:.3},{:.3},{:.3}) is_piloting={} free_look={}",
                    self.camera_yaw, self.camera_pitch, f.x, f.y, f.z, self.is_piloting, free_look
                );
            }
            f
        };
        let aspect = gpu.config.width as f32 / gpu.config.height as f32;
        let proj = Mat4::perspective_rh(70.0_f32.to_radians(), aspect, 0.1, 1e13);
        let cam_up = if self.is_piloting && !free_look {
            (self.ship_rotation * DVec3::Y).as_vec3().normalize()
        } else {
            Vec3::Y
        };
        let view_mat = Mat4::look_to_rh(Vec3::ZERO, cam_fwd, cam_up);
        let vp = proj * view_mat;

        // Pre-compute uniforms.
        let mut uniform_data: Vec<ObjectUniforms> = vec![bytemuck::Zeroable::zeroed(); MAX_OBJECTS];
        let mut object_count = 0usize;
        let mut ship_interior_start = 0usize;
        let mut ship_interior_count = 0usize;

        if let Some(ref ws) = self.latest_world_state {
            // Celestial bodies.
            for body in &ws.bodies {
                if object_count >= MAX_OBJECTS { break; }
                let offset = (body.position - cam_pos).as_vec3();
                let scale = (body.radius as f32).max(1.0);
                let model = Mat4::from_translation(offset) * Mat4::from_scale(Vec3::splat(scale));
                let mvp = vp * model;
                let alpha = if body.body_id == 0 { 1.0 } else { 0.4 };
                let mut obj: ObjectUniforms = bytemuck::Zeroable::zeroed();
                obj.mvp = mvp.to_cols_array_2d();
                obj.color = [body.color[0], body.color[1], body.color[2], alpha];
                uniform_data[object_count] = obj;
                object_count += 1;
            }

            // Ship interior (only when in ship shard).
            if self.current_shard_type == 2 {
                ship_interior_start = object_count;

                // Ship interior rendering — standard space game approach:
                // Camera is parented to ship (view matrix contains inverse(ship_rotation)).
                // Interior objects use model = translate(ship_origin_to_cam) * ship_rot_mat.
                // The ship rotation in model cancels with inverse in view → walls stay fixed.
                // Exterior objects have no ship rotation in model → they rotate when ship turns.
                let ship_rot_mat = Mat4::from_quat(self.ship_rotation.as_quat());

                // Ship origin (0,0,0 in ship-local) relative to camera.
                // Camera is at player_position + eye_height in ship-local, then rotated.
                // Ship origin offset in ship-local = -(player_position + eye_height).
                let origin_local = -(self.player_position + DVec3::new(0.0, EYE_HEIGHT, 0.0));
                let ship_origin_offset = (self.ship_rotation * origin_local).as_vec3();

                // Ship box walls: vertices are in ship-local space.
                let model = Mat4::from_translation(ship_origin_offset) * ship_rot_mat;
                let mvp = vp * model;
                if object_count < MAX_OBJECTS {
                    let mut obj: ObjectUniforms = bytemuck::Zeroable::zeroed();
                    obj.mvp = mvp.to_cols_array_2d();
                    obj.color = [0.3, 0.3, 0.35, 0.4];
                    uniform_data[object_count] = obj;
                    ship_interior_count = 1;
                    object_count += 1;
                }

                // Pilot seat marker: ship-local (0, 0.5, -3) → rotate + offset.
                let seat_local = Vec3::new(0.0, 0.5, -3.0);
                let seat_model = Mat4::from_translation(ship_origin_offset) * ship_rot_mat
                    * Mat4::from_translation(seat_local) * Mat4::from_scale(Vec3::splat(0.3));
                if object_count < MAX_OBJECTS {
                    let mut obj: ObjectUniforms = bytemuck::Zeroable::zeroed();
                    obj.mvp = (vp * seat_model).to_cols_array_2d();
                    obj.color = [0.2, 0.5, 1.0, 1.0];
                    uniform_data[object_count] = obj;
                    object_count += 1;
                }

                // Exit door marker: ship-local (2.0, 0.5, 0) → rotate + offset.
                let door_local = Vec3::new(2.0, 0.5, 0.0);
                let door_model = Mat4::from_translation(ship_origin_offset) * ship_rot_mat
                    * Mat4::from_translation(door_local) * Mat4::from_scale(Vec3::splat(0.3));
                if object_count < MAX_OBJECTS {
                    let mut obj: ObjectUniforms = bytemuck::Zeroable::zeroed();
                    obj.mvp = (vp * door_model).to_cols_array_2d();
                    obj.color = [0.0, 1.0, 0.3, 1.0]; // green emissive
                    uniform_data[object_count] = obj;
                    object_count += 1;
                }
            }
        }

        // Upload uniforms.
        if object_count > 0 {
            gpu.queue.write_buffer(&gpu.uniform_buf, 0, bytemuck::cast_slice(&uniform_data[..object_count]));
        }

        // Render pass.
        let mut encoder = gpu.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("main"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &view, resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color { r: 0.005, g: 0.005, b: 0.02, a: 1.0 }),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: &gpu.depth_view,
                    depth_ops: Some(wgpu::Operations { load: wgpu::LoadOp::Clear(1.0), store: wgpu::StoreOp::Store }),
                    stencil_ops: None,
                }),
                ..Default::default()
            });

            pass.set_pipeline(&gpu.pipeline);

            // Draw celestial bodies + ship markers (spheres).
            pass.set_vertex_buffer(0, gpu.sphere_vertex_buf.slice(..));
            pass.set_index_buffer(gpu.sphere_index_buf.slice(..), wgpu::IndexFormat::Uint32);

            // Celestial bodies: indices 0..ship_interior_start.
            let sphere_end = if ship_interior_count > 0 { ship_interior_start } else { object_count };
            for i in 0..sphere_end {
                pass.set_bind_group(0, &gpu.bind_group, &[(i as u32) * 256]);
                pass.draw_indexed(0..gpu.sphere_index_count, 0, 0..1);
            }

            // Ship markers (seat + door spheres): indices ship_interior_start+1..object_count.
            if ship_interior_count > 0 {
                for i in (ship_interior_start + 1)..object_count {
                    pass.set_bind_group(0, &gpu.bind_group, &[(i as u32) * 256]);
                    pass.draw_indexed(0..gpu.sphere_index_count, 0, 0..1);
                }
            }

            // Draw ship interior box: index ship_interior_start.
            if ship_interior_count > 0 {
                pass.set_vertex_buffer(0, gpu.box_vertex_buf.slice(..));
                pass.set_index_buffer(gpu.box_index_buf.slice(..), wgpu::IndexFormat::Uint32);
                pass.set_bind_group(0, &gpu.bind_group, &[(ship_interior_start as u32) * 256]);
                pass.draw_indexed(0..gpu.box_index_count, 0, 0..1);
            }
        }

        // egui HUD.
        let window = self.window.as_ref().unwrap();
        let scale_factor = window.scale_factor() as f32;
        let logical_w = gpu.config.width as f32 / scale_factor;
        let logical_h = gpu.config.height as f32 / scale_factor;

        let raw_input = gpu.egui_winit.take_egui_input(window);
        let full_output = gpu.egui_ctx.run(raw_input, |ctx| {
            let layer = egui::LayerId::new(egui::Order::Foreground, egui::Id::new("hud"));
            let painter = ctx.layer_painter(layer);

            // Body labels.
            if let Some(ref ws) = self.latest_world_state {
                let body_names = ["Star", "P1", "P2", "P3", "P4", "P5", "P6", "P7", "P8"];
                for body in &ws.bodies {
                    let offset = (body.position - cam_pos).as_vec3();
                    let clip = vp * glam::Vec4::new(offset.x, offset.y, offset.z, 1.0);
                    if clip.w <= 0.0 { continue; }
                    let ndc_x = clip.x / clip.w;
                    let ndc_y = clip.y / clip.w;
                    if ndc_x.abs() > 1.2 || ndc_y.abs() > 1.2 { continue; }
                    let sx = (ndc_x * 0.5 + 0.5) * logical_w;
                    let sy = (1.0 - (ndc_y * 0.5 + 0.5)) * logical_h;
                    let name = body_names.get(body.body_id as usize).unwrap_or(&"?");
                    let dist = offset.length() as f64;
                    let label = if dist > 1e9 { format!("{} {:.1}Gm", name, dist/1e9) }
                        else if dist > 1e6 { format!("{} {:.1}Mm", name, dist/1e6) }
                        else { format!("{} {:.0}km", name, dist/1e3) };
                    let fov_half = 70.0_f64.to_radians() / 2.0;
                    let cr = ((body.radius / dist).atan() / fov_half * logical_h as f64 * 0.5).max(6.0).min(200.0) as f32;
                    let color = if body.body_id == 0 { egui::Color32::from_rgb(255, 220, 100) } else { egui::Color32::from_rgb(100, 200, 255) };
                    painter.circle_stroke(egui::pos2(sx, sy), cr, egui::Stroke::new(1.0, color));
                    painter.text(egui::pos2(sx + cr + 4.0, sy - 6.0), egui::Align2::LEFT_CENTER, &label, egui::FontId::proportional(11.0), color);
                }
            }

            // Info panel.
            let shard_name = match self.current_shard_type { 0 => "Planet", 1 => "System", 2 => "Ship", _ => "?" };
            egui::Area::new(egui::Id::new("info")).fixed_pos(egui::pos2(10.0, 10.0)).show(ctx, |ui| {
                ui.style_mut().visuals.override_text_color = Some(egui::Color32::from_rgb(200, 200, 200));
                ui.label(format!("Shard: {} | Connected: {}", shard_name, self.connected));

                if self.current_shard_type == 2 {
                    // Ship shard HUD.
                    let speed = self.player_velocity.length();
                    let is_piloting = speed > 0.01 || self.player_velocity != DVec3::ZERO;
                    let grounded = self.latest_world_state.as_ref()
                        .and_then(|ws| ws.players.first())
                        .map(|p| p.grounded)
                        .unwrap_or(true);
                    let piloting = !grounded;

                    if piloting {
                        // Pilot HUD — ship parameters.
                        ui.separator();
                        ui.colored_label(egui::Color32::from_rgb(100, 200, 255), "PILOTING");

                        // Velocity.
                        let speed_text = if speed > 1e6 {
                            format!("Speed: {:.2} Mm/s", speed / 1e6)
                        } else if speed > 1e3 {
                            format!("Speed: {:.2} km/s", speed / 1e3)
                        } else {
                            format!("Speed: {:.1} m/s", speed)
                        };
                        ui.label(&speed_text);

                        // Acceleration (thrust / mass = 50kN / 10t = 5 m/s²).
                        ui.label("Thrust: 50 kN | Mass: 10 t | Accel: 5 m/s²");

                        // Ship system position.
                        if let Some(ref ws) = self.latest_world_state {
                            let ship_pos = ws.origin;
                            ui.label(format!("Ship pos: ({:.2e}, {:.2e}, {:.2e})",
                                ship_pos.x, ship_pos.y, ship_pos.z));
                        }

                        // Nearest body.
                        if let Some(ref ws) = self.latest_world_state {
                            if let Some(nearest) = ws.bodies.iter().min_by(|a, b| {
                                let da = a.position.length_squared();
                                let db = b.position.length_squared();
                                da.partial_cmp(&db).unwrap()
                            }) {
                                let body_names = ["Star", "P1", "P2", "P3", "P4", "P5", "P6", "P7", "P8"];
                                let name = body_names.get(nearest.body_id as usize).unwrap_or(&"?");
                                let dist = nearest.position.length();
                                let dist_text = if dist > 1e9 { format!("{:.1} Gm", dist / 1e9) }
                                    else if dist > 1e6 { format!("{:.1} Mm", dist / 1e6) }
                                    else { format!("{:.0} km", dist / 1e3) };

                                // ETA at current speed.
                                let eta = if speed > 0.1 {
                                    let secs = dist / speed;
                                    if secs > 3600.0 { format!("ETA: {:.1}h", secs / 3600.0) }
                                    else if secs > 60.0 { format!("ETA: {:.0}m", secs / 60.0) }
                                    else { format!("ETA: {:.0}s", secs) }
                                } else {
                                    "ETA: --".to_string()
                                };

                                ui.label(format!("Nearest: {} ({}) {}", name, dist_text, eta));
                            }
                        }

                        ui.separator();
                        ui.label("WASD=thrust  E=exit seat");
                    } else {
                        // Walking inside ship.
                        ui.label(format!("Pos: ({:.1}, {:.1}, {:.1})",
                            self.player_position.x, self.player_position.y, self.player_position.z));

                        let seat = DVec3::new(0.0, 0.5, -3.0);
                        let door = DVec3::new(2.0, 0.5, 0.0);
                        let dist_seat = (self.player_position - seat).length();
                        let dist_door = (self.player_position - door).length();
                        ui.label(format!("Pilot seat: {:.1}m | Exit: {:.1}m", dist_seat, dist_door));
                        if dist_seat < 1.5 || dist_door < 1.5 {
                            ui.colored_label(egui::Color32::YELLOW, ">> Press E to interact <<");
                        }
                        ui.label("WASD=walk  E=interact  Space=jump");
                    }
                } else if self.current_shard_type == 0 {
                    // Planet shard HUD.
                    ui.label(format!("Pos: ({:.1}, {:.1}, {:.1})",
                        self.player_position.x, self.player_position.y, self.player_position.z));
                    ui.label("WASD=walk  Space=jump  Mouse=look");
                } else {
                    // System/other.
                    ui.label(format!("Pos: ({:.1}, {:.1}, {:.1})",
                        self.player_position.x, self.player_position.y, self.player_position.z));
                    ui.label("WASD=move  Mouse=look");
                }
            });

            // Crosshair.
            let center = egui::pos2(logical_w / 2.0, logical_h / 2.0);
            painter.circle_stroke(center, 3.0, egui::Stroke::new(1.0, egui::Color32::from_rgba_premultiplied(200, 200, 200, 100)));
        });

        let screen_descriptor = egui_wgpu::ScreenDescriptor {
            size_in_pixels: [gpu.config.width, gpu.config.height],
            pixels_per_point: scale_factor,
        };
        let clipped = gpu.egui_ctx.tessellate(full_output.shapes, full_output.pixels_per_point);
        for (id, delta) in &full_output.textures_delta.set { gpu.egui_renderer.update_texture(&gpu.device, &gpu.queue, *id, delta); }
        gpu.egui_renderer.update_buffers(&gpu.device, &gpu.queue, &mut encoder, &clipped, &screen_descriptor);
        {
            let egui_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("egui"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &view, resolve_target: None,
                    ops: wgpu::Operations { load: wgpu::LoadOp::Load, store: wgpu::StoreOp::Store },
                })],
                depth_stencil_attachment: None,
                ..Default::default()
            });
            let mut egui_pass = egui_pass.forget_lifetime();
            gpu.egui_renderer.render(&mut egui_pass, &clipped, &screen_descriptor);
        }
        for id in &full_output.textures_delta.free { gpu.egui_renderer.free_texture(id); }

        gpu.queue.submit(std::iter::once(encoder.finish()));
        frame.present();
    }
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.gpu.is_none() {
            let attrs = WindowAttributes::default()
                .with_title("Voxydust")
                .with_inner_size(winit::dpi::LogicalSize::new(1280, 720));
            let window = Arc::new(event_loop.create_window(attrs).expect("window"));
            self.init_gpu(window);
            self.start_networking();
            info!("Voxydust ready");
        }
    }

    fn window_event(&mut self, event_loop: &ActiveEventLoop, _id: WindowId, event: WindowEvent) {
        match event {
            WindowEvent::CloseRequested => event_loop.exit(),
            WindowEvent::Resized(size) => {
                if let Some(gpu) = &mut self.gpu {
                    gpu.config.width = size.width.max(1);
                    gpu.config.height = size.height.max(1);
                    gpu.surface.configure(&gpu.device, &gpu.config);
                    gpu.depth_view = create_depth_texture(&gpu.device, gpu.config.width, gpu.config.height, wgpu::TextureFormat::Depth32Float);
                }
            }
            WindowEvent::KeyboardInput { event, .. } => {
                if let PhysicalKey::Code(key) = event.physical_key {
                    if event.state.is_pressed() {
                        self.keys_held.insert(key);
                        if key == KeyCode::Escape {
                            if self.mouse_grabbed {
                                if let Some(ref w) = self.window {
                                    let _ = w.set_cursor_grab(CursorGrabMode::None);
                                    w.set_cursor_visible(true);
                                    self.mouse_grabbed = false;
                                }
                            } else {
                                event_loop.exit();
                            }
                        }
                    } else {
                        self.keys_held.remove(&key);
                    }
                }
            }
            WindowEvent::MouseInput { state, button, .. } => {
                if state.is_pressed() && button == winit::event::MouseButton::Left && !self.mouse_grabbed {
                    if let Some(ref w) = self.window {
                        if w.set_cursor_grab(CursorGrabMode::Locked).is_err() {
                            let _ = w.set_cursor_grab(CursorGrabMode::Confined);
                        }
                        w.set_cursor_visible(false);
                        self.mouse_grabbed = true;
                    }
                }
            }
            WindowEvent::RedrawRequested => {
                self.render();
                if let Some(ref window) = self.window { window.request_redraw(); }
            }
            _ => {}
        }
    }

    fn device_event(&mut self, _el: &ActiveEventLoop, _did: DeviceId, event: DeviceEvent) {
        if !self.mouse_grabbed { return; }
        if let DeviceEvent::MouseMotion { delta } = event {
            let sensitivity = 0.003;
            let free_look = self.keys_held.contains(&KeyCode::AltLeft);

            if self.is_piloting && !free_look {
                // Piloting: set yaw/pitch rate from mouse movement.
                // Rate is proportional to mouse velocity, clamped to [-1, 1].
                self.pilot_yaw_rate = (self.pilot_yaw_rate + delta.0 * sensitivity * 5.0).clamp(-1.0, 1.0);
                self.pilot_pitch_rate = (self.pilot_pitch_rate - delta.1 * sensitivity * 5.0).clamp(-1.0, 1.0);
            } else {
                // Walking or free-look: move camera yaw/pitch directly.
                self.camera_yaw += delta.0 * sensitivity;
                self.camera_pitch -= delta.1 * sensitivity;
                self.camera_pitch = self.camera_pitch.clamp(
                    -std::f64::consts::FRAC_PI_2 + 0.01,
                    std::f64::consts::FRAC_PI_2 - 0.01,
                );
            }
        }
    }
}

/// Generate box mesh (interior visible — normals face inward).
fn generate_box_mesh(width: f32, height: f32, length: f32) -> (Vec<[f32; 3]>, Vec<u32>) {
    let hw = width / 2.0;
    let hl = length / 2.0;
    #[rustfmt::skip]
    let vertices: Vec<[f32; 3]> = vec![
        // Floor (y=0)
        [-hw, 0.0, -hl], [hw, 0.0, -hl], [hw, 0.0, hl], [-hw, 0.0, hl],
        // Ceiling (y=height)
        [-hw, height, -hl], [hw, height, -hl], [hw, height, hl], [-hw, height, hl],
        // Left wall (x=-hw)
        [-hw, 0.0, -hl], [-hw, 0.0, hl], [-hw, height, hl], [-hw, height, -hl],
        // Right wall (x=+hw)
        [hw, 0.0, -hl], [hw, 0.0, hl], [hw, height, hl], [hw, height, -hl],
        // Back wall (z=+hl)
        [-hw, 0.0, hl], [hw, 0.0, hl], [hw, height, hl], [-hw, height, hl],
        // Front wall (z=-hl) — window
        [-hw, 0.0, -hl], [hw, 0.0, -hl], [hw, height, -hl], [-hw, height, -hl],
    ];
    // Inward-facing triangles (CW from outside = CCW from inside).
    #[rustfmt::skip]
    let indices: Vec<u32> = vec![
        0, 2, 1, 0, 3, 2,       // floor (viewed from above)
        4, 5, 6, 4, 6, 7,       // ceiling (viewed from below)
        8, 10, 9, 8, 11, 10,    // left wall
        12, 13, 14, 12, 14, 15, // right wall
        16, 17, 18, 16, 18, 19, // back wall
        20, 22, 21, 20, 23, 22, // front wall (window)
    ];
    (vertices, indices)
}

fn create_depth_texture(device: &wgpu::Device, width: u32, height: u32, format: wgpu::TextureFormat) -> wgpu::TextureView {
    let texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("depth"),
        size: wgpu::Extent3d { width, height, depth_or_array_layers: 1 },
        mip_level_count: 1, sample_count: 1, dimension: wgpu::TextureDimension::D2,
        format, usage: wgpu::TextureUsages::RENDER_ATTACHMENT, view_formats: &[],
    });
    texture.create_view(&Default::default())
}

fn main() {
    let args = Args::parse();
    tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::EnvFilter::try_from_default_env().unwrap_or_else(|_| "info".into()))
        .init();
    info!("Voxydust starting");
    let event_loop = EventLoop::new().expect("event loop");
    let mut app = App::new(args);
    event_loop.run_app(&mut app).expect("event loop error");
}
