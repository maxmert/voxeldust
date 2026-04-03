//! Voxydust — first-person game client with server-authoritative movement.

mod camera;
mod gpu;
mod hud;
mod input;
mod mesh;
mod network;
mod render;

use std::net::SocketAddr;
use std::sync::Arc;

use clap::Parser;
use glam::{DQuat, DVec3};
use tokio::sync::mpsc;
use tracing::info;
use winit::application::ApplicationHandler;
use winit::event::{DeviceEvent, DeviceId, WindowEvent};
use winit::event_loop::{ActiveEventLoop, EventLoop};
use winit::keyboard::{KeyCode, PhysicalKey};
use winit::window::{CursorGrabMode, Window, WindowAttributes, WindowId};

use voxeldust_core::client_message::{PlayerInputData, WorldStateData};

use crate::gpu::{GpuState, ObjectUniforms, MAX_OBJECTS};
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

struct App {
    args: Args,
    window: Option<Arc<Window>>,
    gpu: Option<GpuState>,
    net_event_rx: Option<mpsc::UnboundedReceiver<NetEvent>>,
    input_tx: Option<mpsc::UnboundedSender<PlayerInputData>>,

    // Game state from server (primary shard).
    latest_world_state: Option<WorldStateData>,
    player_position: DVec3,
    player_velocity: DVec3,
    current_shard_type: u8,
    reference_position: DVec3,
    reference_rotation: DQuat,

    // Secondary shard state (for dual-shard compositing during transitions).
    secondary_world_state: Option<WorldStateData>,
    secondary_shard_type: Option<u8>,

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

    // Autopilot.
    system_params: Option<voxeldust_core::system::SystemParams>,
    autopilot_target: Option<usize>,
    selected_thrust_tier: u8,
    trajectory_plan: Option<voxeldust_core::autopilot::TrajectoryPlan>,
    /// Timestamp of last T key press (for double-tap orbit detection).
    last_t_press: Option<std::time::Instant>,
    /// Engine cutoff — when true, no thrust or gravity compensation is sent. Toggle with X.
    engines_off: bool,
    /// Autopilot mode for the current engagement.
    autopilot_mode: voxeldust_core::autopilot::AutopilotMode,
    /// Pre-allocated uniform data buffer (reused every frame to avoid 512KB/frame alloc).
    uniform_data: Vec<ObjectUniforms>,
    /// Last frame timestamp for dt computation.
    last_frame_time: std::time::Instant,
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
            secondary_world_state: None,
            secondary_shard_type: None,
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
            system_params: None,
            autopilot_target: None,
            selected_thrust_tier: 3, // default Long Range
            last_t_press: None,
            engines_off: false,
            autopilot_mode: voxeldust_core::autopilot::AutopilotMode::DirectApproach,
            trajectory_plan: None,
            uniform_data: vec![bytemuck::Zeroable::zeroed(); MAX_OBJECTS],
            last_frame_time: std::time::Instant::now(),
        }
    }

    fn init_gpu(&mut self, window: Arc<Window>) {
        let gpu_state = gpu::init_gpu(window.clone());
        self.window = Some(window);
        self.gpu = Some(gpu_state);
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

                            // Always update ship_rotation on a ship shard — needed for
                            // correct floating-origin rendering of celestial bodies
                            // even while walking inside the ship.
                            if self.current_shard_type == 2 {
                                self.ship_rotation = p.rotation;
                            }

                            if self.is_piloting && !was_piloting {
                                // First frame of piloting: sync camera yaw to ship heading
                                // so there's no visual snap.
                                let fwd = p.rotation * DVec3::NEG_Z;
                                self.camera_yaw = fwd.z.atan2(fwd.x) as f64;
                                self.camera_pitch = fwd.y.asin() as f64;
                            }

                            // Pilot → walk transition: reset camera to ship-local forward.
                            if was_piloting && !self.is_piloting {
                                self.camera_yaw = -std::f64::consts::FRAC_PI_2;
                                self.camera_pitch = 0.0;
                                self.autopilot_target = None;
                                self.trajectory_plan = None;
                            }
                        }
                        if self.latest_world_state.is_none() {
                            info!(bodies = ws.bodies.len(), tick = ws.tick, "first WorldState");
                        }
                        self.latest_world_state = Some(ws);
                    }
                    NetEvent::Connected { shard_type, reference_position, reference_rotation, system_seed, .. } => {
                        self.current_shard_type = shard_type;
                        self.reference_position = reference_position;
                        self.reference_rotation = reference_rotation;
                        if shard_type == 2 {
                            self.ship_rotation = reference_rotation;
                        }
                        self.connected = true;
                        if system_seed > 0 {
                            self.system_params = Some(voxeldust_core::system::SystemParams::from_seed(system_seed));
                        }
                        let shard_name = match shard_type { 0 => "Planet", 1 => "System", 2 => "Ship", _ => "?" };
                        info!(shard_name, "connected to shard");
                    }
                    NetEvent::Disconnected(reason) => {
                        info!(%reason, "disconnected");
                        self.connected = false;
                    }
                    NetEvent::SecondaryConnected { shard_type, seed, reference_position, reference_rotation } => {
                        let shard_name = match shard_type { 0 => "Planet", 2 => "Ship", _ => "?" };
                        info!(shard_name, seed, "secondary shard connected for dual compositing");
                        self.secondary_shard_type = Some(shard_type);
                    }
                    NetEvent::SecondaryWorldState(ws) => {
                        self.secondary_world_state = Some(ws);
                    }
                    NetEvent::Transitioning => {
                        info!("transitioning to new shard...");
                        if self.secondary_world_state.is_some() {
                            // Seamless: promote secondary to primary.
                            info!("seamless transition — secondary data available");
                            self.latest_world_state = self.secondary_world_state.take();
                            if let Some(st) = self.secondary_shard_type.take() {
                                self.current_shard_type = st;
                            }
                            self.camera_yaw = -std::f64::consts::FRAC_PI_2;
                            self.camera_pitch = 0.0;
                            self.is_piloting = false;
                        } else {
                            // Hard transition: clear and reconnect.
                            self.latest_world_state = None;
                        }
                        self.secondary_world_state = None;
                        self.secondary_shard_type = None;
                    }
                }
            }
        }
    }

    /// Engage autopilot to the nearest planet in the ship's forward direction.
    fn engage_autopilot_to_nearest(&mut self, mode: voxeldust_core::autopilot::AutopilotMode) {
        if let Some(ref ws) = self.latest_world_state {
            let ship_fwd = self.ship_rotation * DVec3::NEG_Z;
            let ship_pos = ws.origin;
            let mut best: Option<(usize, f64)> = None;
            for body in &ws.bodies {
                if body.body_id == 0 { continue; }
                let to_body = (body.position - ship_pos).normalize_or_zero();
                let d = ship_fwd.dot(to_body);
                if d > best.map(|(_, bd)| bd).unwrap_or(0.7) {
                    best = Some(((body.body_id - 1) as usize, d));
                }
            }
            if let Some((idx, _)) = best {
                self.autopilot_target = Some(idx);
                self.autopilot_mode = mode;
                info!(planet = idx, mode = ?mode, "autopilot engaged");
            }
        }
    }

    /// Check for double-tap T timeout — if 400ms elapsed since first tap, engage DirectApproach.
    fn check_autopilot_tap_timeout(&mut self) {
        if let Some(press_time) = self.last_t_press {
            if press_time.elapsed() >= std::time::Duration::from_millis(400) {
                self.last_t_press = None;
                self.engage_autopilot_to_nearest(voxeldust_core::autopilot::AutopilotMode::DirectApproach);
            }
        }
    }

    fn send_input_with_dt(&mut self, dt: f64) {
        input::send_input_with_dt(
            &self.input_tx,
            &self.keys_held,
            self.engines_off,
            self.is_piloting,
            self.camera_yaw,
            self.camera_pitch,
            &mut self.pilot_yaw_rate,
            &mut self.pilot_pitch_rate,
            &mut self.selected_thrust_tier,
            self.frame_count,
            dt,
        );
    }

    fn render(&mut self) {
        let now = std::time::Instant::now();
        let dt = (now - self.last_frame_time).as_secs_f64();
        self.last_frame_time = now;

        self.poll_network();
        self.send_input_with_dt(dt);
        self.frame_count += 1;

        let gpu = match &mut self.gpu { Some(g) => g, None => return };

        // Compute camera.
        let cam = camera::compute_camera(
            self.player_position,
            self.camera_yaw,
            self.camera_pitch,
            self.is_piloting,
            self.current_shard_type,
            self.ship_rotation,
            &self.keys_held,
            gpu.config.width,
            gpu.config.height,
            self.latest_world_state.as_ref(),
        );

        // Compute autopilot trajectory for HUD.
        // Throttle to every 10 frames (~6Hz) to prevent visual jitter.
        if self.frame_count % 10 == 0 || self.trajectory_plan.is_none() {
            if let (Some(system), Some(ws), Some(target_idx)) =
                (&self.system_params, &self.latest_world_state, self.autopilot_target)
            {
                let ship_pos = ws.origin;
                let ship_vel = self.player_velocity;
                self.trajectory_plan = voxeldust_core::autopilot::plan_trajectory(
                    ship_pos, ship_vel, target_idx, system,
                    ws.game_time, self.selected_thrust_tier, 200,
                );
            } else if self.autopilot_target.is_none() {
                self.trajectory_plan = None;
            }
        }

        let window = self.window.as_ref().unwrap();

        render::render_frame(
            gpu,
            window,
            &mut self.uniform_data,
            &cam,
            self.latest_world_state.as_ref(),
            self.secondary_world_state.as_ref(),
            self.current_shard_type,
            self.player_position,
            self.player_velocity,
            self.ship_rotation,
            self.is_piloting,
            self.connected,
            self.selected_thrust_tier,
            self.engines_off,
            self.autopilot_target,
            self.trajectory_plan.as_ref(),
            self.system_params.as_ref(),
            self.frame_count,
        );
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
                    gpu.depth_view = gpu::create_depth_texture(&gpu.device, gpu.config.width, gpu.config.height, wgpu::TextureFormat::Depth32Float);
                }
            }
            WindowEvent::KeyboardInput { event, .. } => {
                if let PhysicalKey::Code(key) = event.physical_key {
                    if event.state.is_pressed() {
                        self.keys_held.insert(key);
                        // Autopilot: double-tap T = orbit, single-tap T = direct approach.
                        if key == KeyCode::KeyT && self.is_piloting {
                            if self.autopilot_target.is_some() {
                                // Disengage.
                                self.autopilot_target = None;
                                self.trajectory_plan = None;
                                self.autopilot_mode = voxeldust_core::autopilot::AutopilotMode::DirectApproach;
                            } else {
                                let now = std::time::Instant::now();
                                let is_double = self.last_t_press
                                    .map(|prev| now.duration_since(prev) < std::time::Duration::from_millis(400))
                                    .unwrap_or(false);

                                if is_double {
                                    // Double-tap: orbit insertion mode — engage immediately.
                                    self.last_t_press = None;
                                    self.engage_autopilot_to_nearest(voxeldust_core::autopilot::AutopilotMode::OrbitInsertion);
                                } else {
                                    // First tap: record time, wait for possible second tap.
                                    self.last_t_press = Some(now);
                                }
                            }
                        }
                        // WASD cancels autopilot.
                        if self.autopilot_target.is_some() && matches!(key,
                            KeyCode::KeyW | KeyCode::KeyA | KeyCode::KeyS | KeyCode::KeyD) {
                            self.autopilot_target = None;
                            self.trajectory_plan = None;
                            self.autopilot_mode = voxeldust_core::autopilot::AutopilotMode::DirectApproach;
                        }
                        // X key: toggle engine cutoff (SC-style decoupled mode).
                        if key == KeyCode::KeyX && self.is_piloting {
                            self.engines_off = !self.engines_off;
                            info!(engines_off = self.engines_off, "engine cutoff toggled");
                        }
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
                self.check_autopilot_tap_timeout();
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
                self.pilot_yaw_rate = (self.pilot_yaw_rate - delta.0 * sensitivity * 5.0).clamp(-1.0, 1.0);
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
