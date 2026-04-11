//! egui HUD rendering: body labels, autopilot trajectory, ship/planet info panels, crosshair.

use glam::{DVec3, Mat4};

use crate::gpu::GpuState;

/// All state the HUD needs to read (borrowed from App).
pub struct HudContext<'a> {
    pub latest_world_state: Option<&'a voxeldust_core::client_message::WorldStateData>,
    /// Interpolated body positions (lerped with the same t as the camera).
    pub interpolated_bodies: &'a [voxeldust_core::client_message::CelestialBodyData],
    pub cam_system_pos: DVec3,
    pub vp: Mat4,
    pub player_position: DVec3,
    pub player_velocity: DVec3,
    pub current_shard_type: u8,
    pub is_piloting: bool,
    pub connected: bool,
    pub selected_thrust_tier: u8,
    pub engines_off: bool,
    pub cruise_active: bool,
    pub atmo_comp_active: bool,
    pub autopilot_target: Option<usize>,
    pub trajectory_plan: Option<&'a voxeldust_core::autopilot::TrajectoryPlan>,
    pub server_autopilot: Option<&'a voxeldust_core::shard_message::AutopilotSnapshotData>,
    pub system_params: Option<&'a voxeldust_core::system::SystemParams>,
    pub frame_count: u64,
    pub warp_target_star: Option<WarpTargetInfo>,
    /// Functional block indicators: (ship-local position, kind, letter).
    pub block_indicators: &'a [(glam::Vec3, u8, char)],
}

/// Info about the currently targeted star for warp HUD.
pub struct WarpTargetInfo {
    pub star_index: u32,
    pub star_class_name: &'static str,
    pub distance_gu: f64,
    pub luminosity: f32,
    /// Direction to the star in system-space (for screen-space projection).
    pub direction: DVec3,
}

/// Run the egui HUD pass: body labels, trajectory, info panel, crosshair.
/// Returns the egui `FullOutput` ready for tessellation.
/// Result from the config panel: whether Save or Cancel was clicked.
pub enum ConfigPanelAction {
    None,
    Save,
    Close,
}

pub fn run_hud(
    gpu: &mut GpuState,
    window: &winit::window::Window,
    ctx: &HudContext,
    mut config_state: Option<&mut voxeldust_core::signal::config::BlockSignalConfig>,
) -> (egui::FullOutput, ConfigPanelAction) {
    let scale_factor = window.scale_factor() as f32;
    let logical_w = gpu.config.width as f32 / scale_factor;
    let logical_h = gpu.config.height as f32 / scale_factor;

    let raw_input = gpu.egui_winit.take_egui_input(window);
    let mut panel_action = ConfigPanelAction::None;

    let output = gpu.egui_ctx.run(raw_input, |ectx| {
        let layer = egui::LayerId::new(egui::Order::Foreground, egui::Id::new("hud"));
        let painter = ectx.layer_painter(layer);

        // Body labels.
        draw_body_labels(&painter, ctx, logical_w, logical_h);

        // Warp target reticle (purple circle + label on targeted star).
        draw_warp_target_reticle(&painter, ctx, logical_w, logical_h);

        // Autopilot trajectory line.
        draw_trajectory(&painter, ctx, logical_w, logical_h);

        // Block status indicators (functional blocks).
        // TODO: revisit indicator design — disabled for now.
        // if config_state.is_none() {
        //     draw_block_indicators(&painter, ctx, logical_w, logical_h);
        // }

        // Info panel.
        draw_info_panel(ectx, ctx);

        // Block config panel (when open).
        if let Some(ref mut config) = config_state {
            panel_action = draw_block_config_panel(ectx, config);
        }

        // Crosshair (only when config panel is not visible).
        if config_state.is_none() {
            let center = egui::pos2(logical_w / 2.0, logical_h / 2.0);
            painter.circle_stroke(center, 3.0, egui::Stroke::new(1.0, egui::Color32::from_rgba_premultiplied(200, 200, 200, 100)));
        }
    });
    (output, panel_action)
}

/// Tessellate and render the egui output into the command encoder.
pub fn render_egui(
    gpu: &mut GpuState,
    encoder: &mut wgpu::CommandEncoder,
    view: &wgpu::TextureView,
    full_output: egui::FullOutput,
) {
    let scale_factor = gpu.config.height as f32; // dummy; we use the real descriptor below
    let _ = scale_factor;
    let screen_descriptor = egui_wgpu::ScreenDescriptor {
        size_in_pixels: [gpu.config.width, gpu.config.height],
        pixels_per_point: full_output.pixels_per_point,
    };
    let clipped = gpu.egui_ctx.tessellate(full_output.shapes, full_output.pixels_per_point);
    for (id, delta) in &full_output.textures_delta.set {
        gpu.egui_renderer.update_texture(&gpu.device, &gpu.queue, *id, delta);
    }
    gpu.egui_renderer.update_buffers(&gpu.device, &gpu.queue, encoder, &clipped, &screen_descriptor);
    {
        let egui_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("egui"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view, resolve_target: None,
                ops: wgpu::Operations { load: wgpu::LoadOp::Load, store: wgpu::StoreOp::Store },
            })],
            depth_stencil_attachment: None,
            ..Default::default()
        });
        let mut egui_pass = egui_pass.forget_lifetime();
        gpu.egui_renderer.render(&mut egui_pass, &clipped, &screen_descriptor);
    }
    for id in &full_output.textures_delta.free {
        gpu.egui_renderer.free_texture(id);
    }
}

// ---------------------------------------------------------------------------
// Private helpers
// ---------------------------------------------------------------------------

fn draw_body_labels(painter: &egui::Painter, ctx: &HudContext, logical_w: f32, logical_h: f32) {
    {
        let body_names = ["Star", "P1", "P2", "P3", "P4", "P5", "P6", "P7", "P8"];
        for body in ctx.interpolated_bodies {
            let offset = (body.position - ctx.cam_system_pos).as_vec3();
            let clip = ctx.vp * glam::Vec4::new(offset.x, offset.y, offset.z, 1.0);
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
}

/// Draw small status indicators on functional blocks visible within ~15m.
fn draw_block_indicators(painter: &egui::Painter, ctx: &HudContext, logical_w: f32, logical_h: f32) {
    if ctx.current_shard_type != 2 || ctx.is_piloting {
        return; // Only show when walking inside a ship.
    }

    let accent = egui::Color32::from_rgba_unmultiplied(60, 200, 255, 160);
    let glow = egui::Color32::from_rgba_unmultiplied(60, 200, 255, 40);

    for &(block_pos, kind, letter) in ctx.block_indicators {
        // block_pos is ship-local. For the VP projection we need camera-relative offset.
        // In floating-origin, camera is at origin, so offset = block_pos - render_position.
        // But render_position was already subtracted during camera computation,
        // so we need the offset from the camera position (which is at origin in view space).
        // The VP matrix expects positions relative to the camera.
        let offset = block_pos - ctx.player_position.as_vec3();
        let dist = offset.length();

        // Distance cull: only show within 15m.
        if dist > 15.0 || dist < 0.5 {
            continue;
        }

        // Project to screen.
        let clip = ctx.vp * glam::Vec4::new(offset.x, offset.y, offset.z, 1.0);
        if clip.w <= 0.0 { continue; } // Behind camera.
        let ndc_x = clip.x / clip.w;
        let ndc_y = clip.y / clip.w;
        if ndc_x.abs() > 1.1 || ndc_y.abs() > 1.1 { continue; } // Off screen.
        let sx = (ndc_x * 0.5 + 0.5) * logical_w;
        let sy = (1.0 - (ndc_y * 0.5 + 0.5)) * logical_h;

        // Scale indicator size by distance (larger when closer).
        let size = (6.0 / dist).clamp(3.0, 12.0);

        // Glow halo.
        painter.circle_filled(egui::pos2(sx, sy), size + 3.0, glow);
        // Indicator dot.
        painter.circle_filled(egui::pos2(sx, sy), size, accent);
        // Kind letter.
        painter.text(
            egui::pos2(sx, sy),
            egui::Align2::CENTER_CENTER,
            letter.to_string(),
            egui::FontId::proportional(size * 1.5),
            egui::Color32::from_rgb(10, 20, 40),
        );
    }
}

fn draw_warp_target_reticle(painter: &egui::Painter, ctx: &HudContext, logical_w: f32, logical_h: f32) {
    let wt = match &ctx.warp_target_star {
        Some(wt) => wt,
        None => return,
    };

    // Project star direction to screen space.
    // Stars are at infinity in skybox mode — use direction * large distance.
    let dir = wt.direction.as_vec3();
    let far_pos = dir * 500.0; // same distance as skybox in shader
    let clip = ctx.vp * glam::Vec4::new(far_pos.x, far_pos.y, far_pos.z, 1.0);
    if clip.w <= 0.0 { return; } // behind camera

    let ndc_x = clip.x / clip.w;
    let ndc_y = clip.y / clip.w;
    if ndc_x.abs() > 1.5 || ndc_y.abs() > 1.5 { return; } // off screen
    let sx = (ndc_x * 0.5 + 0.5) * logical_w;
    let sy = (1.0 - (ndc_y * 0.5 + 0.5)) * logical_h;

    let purple = egui::Color32::from_rgb(180, 100, 255);

    // Draw diamond reticle (4 lines forming a diamond shape).
    let r = 18.0_f32;
    let center = egui::pos2(sx, sy);
    let stroke = egui::Stroke::new(1.5, purple);
    painter.line_segment([egui::pos2(sx, sy - r), egui::pos2(sx + r, sy)], stroke);
    painter.line_segment([egui::pos2(sx + r, sy), egui::pos2(sx, sy + r)], stroke);
    painter.line_segment([egui::pos2(sx, sy + r), egui::pos2(sx - r, sy)], stroke);
    painter.line_segment([egui::pos2(sx - r, sy), egui::pos2(sx, sy - r)], stroke);

    // Corner brackets (outer, larger).
    let br = 26.0_f32;
    let bl = 8.0_f32; // bracket line length
    let bracket_stroke = egui::Stroke::new(1.0, purple);
    // Top-left bracket
    painter.line_segment([egui::pos2(sx - br, sy - br), egui::pos2(sx - br + bl, sy - br)], bracket_stroke);
    painter.line_segment([egui::pos2(sx - br, sy - br), egui::pos2(sx - br, sy - br + bl)], bracket_stroke);
    // Top-right bracket
    painter.line_segment([egui::pos2(sx + br, sy - br), egui::pos2(sx + br - bl, sy - br)], bracket_stroke);
    painter.line_segment([egui::pos2(sx + br, sy - br), egui::pos2(sx + br, sy - br + bl)], bracket_stroke);
    // Bottom-left bracket
    painter.line_segment([egui::pos2(sx - br, sy + br), egui::pos2(sx - br + bl, sy + br)], bracket_stroke);
    painter.line_segment([egui::pos2(sx - br, sy + br), egui::pos2(sx - br, sy + br - bl)], bracket_stroke);
    // Bottom-right bracket
    painter.line_segment([egui::pos2(sx + br, sy + br), egui::pos2(sx + br - bl, sy + br)], bracket_stroke);
    painter.line_segment([egui::pos2(sx + br, sy + br), egui::pos2(sx + br, sy + br - bl)], bracket_stroke);

    // Label: star class + distance.
    let dist_text = if wt.distance_gu > 1000.0 { format!("{:.0} GU", wt.distance_gu) }
        else { format!("{:.1} GU", wt.distance_gu) };
    let eta = wt.distance_gu / 25.0;
    let eta_text = if eta > 60.0 { format!("{:.0}m", eta / 60.0) }
        else { format!("{:.0}s", eta) };
    let label = format!("WARP: {} | {} | ~{}", wt.star_class_name, dist_text, eta_text);
    painter.text(
        egui::pos2(sx + br + 6.0, sy - 6.0),
        egui::Align2::LEFT_CENTER,
        &label,
        egui::FontId::proportional(11.0),
        purple,
    );
}

fn draw_trajectory(painter: &egui::Painter, ctx: &HudContext, logical_w: f32, logical_h: f32) {
    let plan = match ctx.trajectory_plan {
        Some(p) => p,
        None => return,
    };

    use voxeldust_core::autopilot::FlightPhase;
    let accel_color = egui::Color32::from_rgb(100, 200, 255);
    let brake_color = egui::Color32::from_rgb(255, 150, 50);
    let flip_color = egui::Color32::from_rgb(255, 255, 100);

    // Project trajectory points to screen space.
    let mut screen_pts: Vec<(egui::Pos2, FlightPhase)> = Vec::new();
    for pt in &plan.points {
        let offset = (pt.position - ctx.cam_system_pos).as_vec3();
        let clip = ctx.vp * glam::Vec4::new(offset.x, offset.y, offset.z, 1.0);
        if clip.w <= 0.0 { continue; }
        let ndc_x = clip.x / clip.w;
        let ndc_y = clip.y / clip.w;
        if ndc_x.abs() > 2.0 || ndc_y.abs() > 2.0 { continue; }
        let sx = (ndc_x * 0.5 + 0.5) * logical_w;
        let sy = (1.0 - (ndc_y * 0.5 + 0.5)) * logical_h;
        screen_pts.push((egui::pos2(sx, sy), pt.phase));
    }

    // Catmull-Rom subdivision + dashed rendering.
    if screen_pts.len() >= 4 {
        let dash_offset = (ctx.frame_count / 3) as usize;
        let mut seg_idx = 0usize;

        for i in 0..screen_pts.len().saturating_sub(1) {
            let p0 = screen_pts[i.saturating_sub(1)].0;
            let p1 = screen_pts[i].0;
            let p2 = screen_pts[(i + 1).min(screen_pts.len() - 1)].0;
            let p3 = screen_pts[(i + 2).min(screen_pts.len() - 1)].0;
            let phase = screen_pts[i].1;
            let color = match phase {
                FlightPhase::Accelerate => accel_color,
                FlightPhase::Flip => flip_color,
                FlightPhase::Brake => brake_color,
                FlightPhase::Arrived => egui::Color32::GREEN,
                FlightPhase::SoiApproach | FlightPhase::CircularizeBurn => egui::Color32::from_rgb(100, 200, 255),
                FlightPhase::StableOrbit => egui::Color32::from_rgb(80, 180, 255),
                FlightPhase::DeorbitBurn => egui::Color32::from_rgb(255, 200, 50),
                FlightPhase::AtmosphericEntry => egui::Color32::from_rgb(255, 150, 50),
                FlightPhase::TerminalDescent | FlightPhase::Landing => egui::Color32::from_rgb(255, 100, 50),
                FlightPhase::Landed => egui::Color32::from_rgb(100, 255, 100),
                FlightPhase::Liftoff | FlightPhase::GravityTurn | FlightPhase::AscentBurn => egui::Color32::from_rgb(100, 255, 200),
                FlightPhase::EscapeBurn => egui::Color32::from_rgb(200, 100, 255),
                FlightPhase::WarpAlign | FlightPhase::WarpAccelerate => egui::Color32::from_rgb(150, 100, 255),
                FlightPhase::WarpCruise => egui::Color32::from_rgb(100, 150, 255),
                FlightPhase::WarpDecelerate => egui::Color32::from_rgb(100, 200, 255),
                FlightPhase::WarpArrival => egui::Color32::from_rgb(100, 255, 200),
            };

            // Subdivide into 4 sub-segments for smoothness.
            for s in 0..4 {
                let t0 = s as f32 / 4.0;
                let t1 = (s + 1) as f32 / 4.0;
                let a = catmull_rom(p0, p1, p2, p3, t0);
                let b = catmull_rom(p0, p1, p2, p3, t1);

                // Dash pattern: draw 3, skip 3.
                if (seg_idx + dash_offset) % 6 < 3 {
                    painter.line_segment([a, b], egui::Stroke::new(2.0, color));
                }
                seg_idx += 1;
            }
        }

        // Flip point marker.
        if plan.flip_index < screen_pts.len() {
            let (fp, _) = screen_pts[plan.flip_index];
            painter.circle_filled(fp, 4.0, flip_color);
            painter.text(egui::pos2(fp.x + 8.0, fp.y),
                egui::Align2::LEFT_CENTER, "FLIP",
                egui::FontId::proportional(10.0), flip_color);
        }

        // Intercept point marker.
        if let Some(&(last, _)) = screen_pts.last() {
            painter.circle_stroke(last, 6.0, egui::Stroke::new(2.0, egui::Color32::GREEN));
            painter.text(egui::pos2(last.x + 8.0, last.y),
                egui::Align2::LEFT_CENTER, "SOI",
                egui::FontId::proportional(10.0), egui::Color32::GREEN);
        }
    }
}

fn draw_info_panel(ectx: &egui::Context, ctx: &HudContext) {
    let shard_name = match ctx.current_shard_type { 0 => "Planet", 1 => "System", 2 => "Ship", 3 => "Galaxy", _ => "?" };
    egui::Area::new(egui::Id::new("info")).fixed_pos(egui::pos2(10.0, 10.0)).show(ectx, |ui| {
        ui.style_mut().visuals.override_text_color = Some(egui::Color32::from_rgb(200, 200, 200));
        ui.label(format!("Shard: {} | Connected: {}", shard_name, ctx.connected));

        if ctx.current_shard_type == voxeldust_core::client_message::shard_type::SHIP {
            draw_ship_hud(ui, ctx);
        } else if ctx.current_shard_type == 3 {
            // Galaxy shard HUD — warp travel.
            draw_galaxy_hud(ui, ctx);
        } else if ctx.current_shard_type == 0 {
            // Planet shard HUD.
            ui.label(format!("Pos: ({:.1}, {:.1}, {:.1})",
                ctx.player_position.x, ctx.player_position.y, ctx.player_position.z));
            ui.label("WASD=walk  Space=jump  Mouse=look");
        } else {
            // System/other.
            ui.label(format!("Pos: ({:.1}, {:.1}, {:.1})",
                ctx.player_position.x, ctx.player_position.y, ctx.player_position.z));
            ui.label("WASD=move  Mouse=look");
        }
    });
}

fn draw_galaxy_hud(ui: &mut egui::Ui, ctx: &HudContext) {
    ui.separator();
    ui.colored_label(egui::Color32::from_rgb(150, 100, 255), "WARP TRAVEL");

    let speed = ctx.player_velocity.length();
    if speed > 0.01 {
        ui.label(format!("Speed: {:.2} GU/s", speed));
    }

    ui.label(format!("Pos: ({:.1}, {:.1}, {:.1}) GU",
        ctx.player_position.x, ctx.player_position.y, ctx.player_position.z));
}

fn draw_ship_hud(ui: &mut egui::Ui, ctx: &HudContext) {
    let speed = ctx.player_velocity.length();
    let grounded = ctx.latest_world_state
        .and_then(|ws| ws.players.first())
        .map(|p| p.grounded)
        .unwrap_or(true);
    let piloting = !grounded;

    if piloting {
        draw_pilot_hud(ui, ctx, speed);
    } else {
        // Walking inside ship.
        ui.label(format!("Pos: ({:.1}, {:.1}, {:.1})",
            ctx.player_position.x, ctx.player_position.y, ctx.player_position.z));

        let seat = DVec3::new(0.0, 0.5, -3.0);
        let door = DVec3::new(2.0, 0.5, 0.0);
        let dist_seat = (ctx.player_position - seat).length();
        let dist_door = (ctx.player_position - door).length();
        ui.label(format!("Pilot seat: {:.1}m | Exit: {:.1}m", dist_seat, dist_door));
        if dist_seat < 1.5 || dist_door < 1.5 {
            ui.colored_label(egui::Color32::YELLOW, ">> Press E to interact <<");
        }
        ui.label("WASD=walk  E=interact  Space=jump");
    }
}

fn draw_pilot_hud(ui: &mut egui::Ui, ctx: &HudContext, speed: f64) {
    // Pilot HUD -- ship parameters.
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

    // Acceleration (thrust / mass = 50kN / 10t = 5 m/s^2).
    ui.label("Thrust: 50 kN | Mass: 10 t | Accel: 5 m/s\u{00B2}");

    // Ship system position.
    if let Some(ws) = ctx.latest_world_state {
        let ship_pos = ws.origin;
        ui.label(format!("Ship pos: ({:.2e}, {:.2e}, {:.2e})",
            ship_pos.x, ship_pos.y, ship_pos.z));
    }

    // Nearest body.
    if let Some(ws) = ctx.latest_world_state {
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
    let tier_label = thrust_tier_label(ctx.selected_thrust_tier);

    // Autopilot status.
    if let Some(plan) = ctx.trajectory_plan {
        draw_autopilot_active(ui, ctx, plan, tier_label);
    } else if ctx.autopilot_target.is_some() {
        // Autopilot target selected but trajectory computation failed.
        let body_names_ap = ["Star", "P1", "P2", "P3", "P4", "P5", "P6", "P7", "P8"];
        let idx = ctx.autopilot_target.unwrap();
        let target_name = body_names_ap.get(idx + 1).unwrap_or(&"?");
        ui.colored_label(egui::Color32::from_rgb(255, 200, 100),
            format!("AUTOPILOT: targeting {} (computing...)", target_name));
        ui.label(format!("Thrust: {}", tier_label));
        ui.label("T=disengage | 1-5=thrust | WASD=cancel");
    } else {
        if ctx.cruise_active {
            ui.colored_label(egui::Color32::from_rgb(60, 200, 255), "CRUISE ACTIVE (C to disengage)");
        }
        if ctx.atmo_comp_active {
            // Show hover status with gravity and speed from nearest planet.
            let grav_info = ctx.system_params.and_then(|sp| {
                ctx.latest_world_state.and_then(|ws| {
                    ws.bodies.iter().filter(|b| b.body_id > 0).filter_map(|b| {
                        let pi = (b.body_id - 1) as usize;
                        let planet = sp.planets.get(pi)?;
                        let dist = (ws.origin - b.position).length();
                        let alt = dist - planet.radius_m;
                        if alt < planet.atmosphere.atmosphere_height && planet.atmosphere.has_atmosphere {
                            Some((planet.surface_gravity, alt))
                        } else {
                            None
                        }
                    }).next()
                })
            });
            if let Some((g, alt)) = grav_info {
                let speed = ctx.player_velocity.length();
                let alt_text = if alt > 1000.0 { format!("{:.1} km", alt / 1000.0) }
                    else { format!("{:.0} m", alt) };
                ui.colored_label(
                    egui::Color32::from_rgb(100, 200, 255),
                    format!("ATMO COMP  {:.1}g  Alt: {}  {:.0} m/s  (H=off)", g / 9.81, alt_text, speed),
                );
            } else {
                ui.colored_label(egui::Color32::YELLOW, "ATMO COMP: no atmosphere (H=off)");
            }
        }
        if ctx.engines_off {
            ui.colored_label(egui::Color32::RED, "ENGINES OFF (X to restart)");
        }
        ui.label("WASD=thrust  E=exit seat  X=engine cutoff");
        ui.label("T=autopilot | TT=orbit | G=warp target");
        ui.label(format!("Thrust: {} (1-5 to change)", tier_label));

        // Warp target info (also shown in the reticle, but summary here too).
        if let Some(ref wt) = ctx.warp_target_star {
            ui.separator();
            ui.colored_label(egui::Color32::from_rgb(180, 100, 255),
                format!("WARP TARGET: Star #{}", wt.star_index));
            ui.label("G=cycle | Enter=engage warp | Esc=cancel");
        }
    }

    // Orbital elements display when near a planet.
    draw_orbital_elements(ui, ctx, speed);
}

fn draw_autopilot_active(
    ui: &mut egui::Ui,
    ctx: &HudContext,
    plan: &voxeldust_core::autopilot::TrajectoryPlan,
    tier_label: &str,
) {
    use voxeldust_core::autopilot::FlightPhase;
    let phase_name = match plan.current_phase {
        FlightPhase::Accelerate => "ACCEL",
        FlightPhase::Flip => "FLIP",
        FlightPhase::Brake => "BRAKE",
        FlightPhase::Arrived => "ARRIVED",
        FlightPhase::SoiApproach => "SOI APPROACH",
        FlightPhase::CircularizeBurn => "CIRCULARIZE",
        FlightPhase::StableOrbit => "ORBIT",
        FlightPhase::DeorbitBurn => "DEORBIT",
        FlightPhase::AtmosphericEntry => "ATMO ENTRY",
        FlightPhase::TerminalDescent => "DESCENT",
        FlightPhase::Landing => "LANDING",
        FlightPhase::Landed => "LANDED",
        FlightPhase::Liftoff => "LIFTOFF",
        FlightPhase::GravityTurn => "GRAVITY TURN",
        FlightPhase::AscentBurn => "ASCENT",
        FlightPhase::EscapeBurn => "ESCAPE",
        FlightPhase::WarpAlign => "WARP ALIGN",
        FlightPhase::WarpAccelerate => "WARP ACCEL",
        FlightPhase::WarpCruise => "WARP CRUISE",
        FlightPhase::WarpDecelerate => "WARP DECEL",
        FlightPhase::WarpArrival => "WARP ARRIVAL",
    };
    let body_names_ap = ["Star", "P1", "P2", "P3", "P4", "P5", "P6", "P7", "P8"];
    let target_name = body_names_ap.get(plan.target_planet_index + 1).unwrap_or(&"?");

    ui.colored_label(egui::Color32::from_rgb(100, 255, 200),
        format!("AUTOPILOT: {} -> {}", phase_name, target_name));
    let dampener_str = if plan.dampener_active { "DAMPENER ON" } else { "" };
    ui.label(format!("{} | Felt: {:.1}g {}", tier_label, plan.felt_g, dampener_str));

    // Prefer server-authoritative ETA when available.
    let eta = ctx.server_autopilot
        .map(|ap| ap.eta_real_seconds)
        .filter(|&e| e > 0.0)
        .unwrap_or(plan.eta_real_seconds);
    let eta_text = if eta > 3600.0 { format!("ETA: {:.1}h", eta / 3600.0) }
        else if eta > 60.0 { format!("ETA: {:.0}m {:.0}s", (eta / 60.0).floor(), eta % 60.0) }
        else { format!("ETA: {:.1}s", eta) };
    ui.label(&eta_text);

    ui.separator();
    ui.label("T=disengage | 1-5=thrust | WASD=cancel");
}

fn draw_orbital_elements(ui: &mut egui::Ui, ctx: &HudContext, speed: f64) {
    if let (Some(ws), Some(sp)) = (ctx.latest_world_state, ctx.system_params) {
        let ship_pos = ws.origin;
        for body in &ws.bodies {
            if body.body_id == 0 { continue; }
            let pi = (body.body_id - 1) as usize;
            if pi >= sp.planets.len() { continue; }
            let planet = &sp.planets[pi];
            let soi = voxeldust_core::system::compute_soi_radius(planet, &sp.star);
            let dist = (ship_pos - body.position).length();
            if dist < soi {
                ui.separator();
                let body_names = ["Star", "P1", "P2", "P3", "P4", "P5", "P6", "P7", "P8"];
                let name = body_names.get(pi + 1).unwrap_or(&"?");
                let alt = dist - planet.radius_m;
                let alt_text = if alt > 1e6 { format!("{:.0} km", alt / 1000.0) }
                    else { format!("{:.0} m", alt) };
                let soi_text = if soi > 1e6 { format!("{:.0} km", soi / 1000.0) }
                    else { format!("{:.0} m", soi) };
                let v_circ = voxeldust_core::autopilot::circular_orbit_velocity(planet, alt);
                let v_esc = voxeldust_core::autopilot::escape_velocity(planet, alt);
                ui.colored_label(egui::Color32::from_rgb(100, 200, 255),
                    format!("SOI: {}  Alt: {}  g: {:.1} m/s\u{00B2}", name, alt_text, planet.surface_gravity));
                ui.label(format!("SOI: {}  v_circ: {:.0} m/s  v_esc: {:.0} m/s",
                    soi_text, v_circ, v_esc));
                if planet.atmosphere.has_atmosphere {
                    let density = planet.atmosphere.density_at_altitude(alt);
                    let atmo_text = if alt < planet.atmosphere.atmosphere_height {
                        format!("ATMO: rho={:.4} kg/m\u{00B3}", density)
                    } else {
                        "ABOVE ATMOSPHERE".to_string()
                    };
                    ui.label(atmo_text);
                }
                // Landing indicator based on altitude and speed.
                let landing_gear = 1.5; // from ShipPhysicalProperties::starter_ship
                if alt < landing_gear + 1.0 && speed < 1.0 {
                    ui.colored_label(egui::Color32::GREEN, "LANDED");
                }
                break;
            }
        }
    }
}

fn thrust_tier_label(tier: u8) -> &'static str {
    match tier {
        0 => "MANEUVER 0.5g",
        1 => "IMPULSE 3g",
        2 => "CRUISE 5000g",
        3 => "LONG RANGE 50000g",
        4 => "EMERGENCY 250000g",
        _ => "?",
    }
}

/// Catmull-Rom spline interpolation between p1 and p2, using p0 and p3 as control points.
fn catmull_rom(p0: egui::Pos2, p1: egui::Pos2, p2: egui::Pos2, p3: egui::Pos2, t: f32) -> egui::Pos2 {
    let t2 = t * t;
    let t3 = t2 * t;
    let x = 0.5
        * ((2.0 * p1.x)
            + (-p0.x + p2.x) * t
            + (2.0 * p0.x - 5.0 * p1.x + 4.0 * p2.x - p3.x) * t2
            + (-p0.x + 3.0 * p1.x - 3.0 * p2.x + p3.x) * t3);
    let y = 0.5
        * ((2.0 * p1.y)
            + (-p0.y + p2.y) * t
            + (2.0 * p0.y - 5.0 * p1.y + 4.0 * p2.y - p3.y) * t2
            + (-p0.y + 3.0 * p1.y - 3.0 * p2.y + p3.y) * t3);
    egui::pos2(x, y)
}

// ---------------------------------------------------------------------------
// Block signal config panel
// ---------------------------------------------------------------------------

use voxeldust_core::signal::components::SeatControl;
use voxeldust_core::signal::config::BlockSignalConfig;
use voxeldust_core::signal::types::SignalProperty;

const SIGNAL_PROPERTY_NAMES: &[(&str, SignalProperty)] = &[
    ("Active", SignalProperty::Active),
    ("Throttle", SignalProperty::Throttle),
    ("Angle", SignalProperty::Angle),
    ("Extension", SignalProperty::Extension),
    ("Pressure", SignalProperty::Pressure),
    ("Speed", SignalProperty::Speed),
    ("Level", SignalProperty::Level),
    ("SwitchState", SignalProperty::SwitchState),
    ("Boost", SignalProperty::Boost),
    ("Status", SignalProperty::Status),
];

fn functional_kind_name(kind: u8) -> &'static str {
    // Must match the enum discriminant order in registry.rs.
    const NAMES: &[(u8, &str)] = &[
        (0, "Thruster"), (1, "Reactor"), (2, "Battery"), (3, "Solar Panel"),
        (4, "Power Conduit"), (5, "Seat"), (6, "Gravity Gen"), (7, "Shield Emitter"),
        (8, "Shield Gen"), (9, "Air Compressor"), (10, "Antenna"), (11, "Rotor"),
        (12, "Piston"), (13, "Rail"), (14, "Rail Junction"), (15, "Rail Signal"),
        (16, "Signal Converter"), (17, "Sensor"), (18, "Computer"),
    ];
    NAMES.iter().find(|(k, _)| *k == kind).map(|(_, n)| *n).unwrap_or("Unknown")
}

fn property_name(prop: SignalProperty) -> &'static str {
    SIGNAL_PROPERTY_NAMES.iter()
        .find(|(_, p)| *p == prop)
        .map(|(name, _)| *name)
        .unwrap_or("Unknown")
}

/// Text input for signal channel names with inline autocomplete from available channels.
fn channel_name_input(
    ui: &mut egui::Ui,
    id_salt: &str,
    channel_name: &mut String,
    available_channels: &[String],
) {
    let response = ui.add(
        egui::TextEdit::singleline(channel_name)
            .desired_width(120.0)
            .id(egui::Id::new(id_salt)),
    );

    // When focused and typing, show matching suggestions as clickable labels below.
    if response.has_focus() && !channel_name.is_empty() {
        let query = channel_name.to_lowercase();
        let mut shown = 0;
        for ch in available_channels {
            if ch.to_lowercase().contains(&query) && ch.as_str() != channel_name.as_str() {
                if ui.small_button(
                    egui::RichText::new(format!("  {ch}"))
                        .color(egui::Color32::from_rgb(100, 180, 240))
                        .size(10.0),
                ).clicked() {
                    *channel_name = ch.clone();
                }
                shown += 1;
                if shown >= 5 { break; }
            }
        }
    }
}

fn draw_block_config_panel(
    ectx: &egui::Context,
    config: &mut BlockSignalConfig,
) -> ConfigPanelAction {
    let mut action = ConfigPanelAction::None;

    let kind_label = functional_kind_name(config.kind);

    let screen = ectx.screen_rect();
    let accent = egui::Color32::from_rgba_unmultiplied(60, 180, 255, 180);
    let accent_dim = egui::Color32::from_rgba_unmultiplied(60, 180, 255, 40);

    // Dim the game world behind the panel (vignette effect).
    let bg_layer = egui::LayerId::new(egui::Order::Background, egui::Id::new("config_bg_dim"));
    let bg_painter = ectx.layer_painter(bg_layer);
    bg_painter.rect_filled(screen, 0.0, egui::Color32::from_rgba_unmultiplied(0, 0, 0, 120));

    // Panel dimensions — fit within the block face projection.
    // Camera is 1m from a 1m×1m face at 70° vertical FOV.
    // Block face angular size: 2*atan(0.5/1.0) ≈ 53° → covers 53/70 ≈ 75% of screen height.
    // Use 65% with inner padding so content stays within the face edges.
    let face_frac = 0.65;
    let panel_h = (screen.height() * face_frac).min(screen.height() - 40.0);
    let panel_w = panel_h.min(screen.width() * face_frac); // keep roughly square like the face
    let panel_x = (screen.width() - panel_w) / 2.0;
    let panel_y = (screen.height() - panel_h) / 2.0;
    let panel_rect = egui::Rect::from_min_size(
        egui::pos2(screen.left() + panel_x, screen.top() + panel_y),
        egui::vec2(panel_w, panel_h),
    );

    // Draw AR frame: corner brackets around the panel (like a HUD targeting reticle).
    let frame_layer = egui::LayerId::new(egui::Order::Foreground, egui::Id::new("config_frame"));
    let frame_painter = ectx.layer_painter(frame_layer);
    let corner_len = 20.0;
    let bracket_stroke = egui::Stroke::new(2.0, accent);
    let r = panel_rect.expand(6.0);
    // Top-left
    frame_painter.line_segment([r.left_top(), egui::pos2(r.left() + corner_len, r.top())], bracket_stroke);
    frame_painter.line_segment([r.left_top(), egui::pos2(r.left(), r.top() + corner_len)], bracket_stroke);
    // Top-right
    frame_painter.line_segment([r.right_top(), egui::pos2(r.right() - corner_len, r.top())], bracket_stroke);
    frame_painter.line_segment([r.right_top(), egui::pos2(r.right(), r.top() + corner_len)], bracket_stroke);
    // Bottom-left
    frame_painter.line_segment([r.left_bottom(), egui::pos2(r.left() + corner_len, r.bottom())], bracket_stroke);
    frame_painter.line_segment([r.left_bottom(), egui::pos2(r.left(), r.bottom() - corner_len)], bracket_stroke);
    // Bottom-right
    frame_painter.line_segment([r.right_bottom(), egui::pos2(r.right() - corner_len, r.bottom())], bracket_stroke);
    frame_painter.line_segment([r.right_bottom(), egui::pos2(r.right(), r.bottom() - corner_len)], bracket_stroke);

    // Subtle horizontal scanlines across the panel area (holographic effect).
    let scanline_color = egui::Color32::from_rgba_unmultiplied(60, 180, 255, 8);
    let mut y = panel_rect.top();
    while y < panel_rect.bottom() {
        frame_painter.line_segment(
            [egui::pos2(panel_rect.left(), y), egui::pos2(panel_rect.right(), y)],
            egui::Stroke::new(1.0, scanline_color),
        );
        y += 4.0;
    }

    // Panel content frame.
    let panel_frame = egui::Frame::default()
        .fill(egui::Color32::from_rgba_unmultiplied(5, 15, 30, 210))
        .stroke(egui::Stroke::new(1.0, accent_dim))
        .rounding(4.0)
        .inner_margin(egui::Margin::same(16.0));

    egui::CentralPanel::default()
        .frame(egui::Frame::none())
        .show(ectx, |ui| {
            ui.allocate_ui_at_rect(panel_rect, |ui| {
                    panel_frame.show(ui, |ui| {
                        // AR title with accent color.
                        ui.horizontal(|ui| {
                            ui.label(
                                egui::RichText::new(format!("// SIGNAL CONFIG — {}", kind_label))
                                    .color(egui::Color32::from_rgb(60, 200, 255))
                                    .size(16.0)
                                    .strong(),
                            );
                            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                                if ui.small_button(
                                    egui::RichText::new("X")
                                        .color(egui::Color32::from_rgb(255, 100, 100))
                                        .strong(),
                                ).clicked() {
                                    action = ConfigPanelAction::Close;
                                }
                            });
                        });

                        ui.add_space(4.0);
                        // Thin accent separator.
                        let rect = ui.available_rect_before_wrap();
                        ui.painter().line_segment(
                            [egui::pos2(rect.left(), rect.top()), egui::pos2(rect.right(), rect.top())],
                            egui::Stroke::new(1.0, egui::Color32::from_rgba_unmultiplied(60, 180, 255, 80)),
                        );
                        ui.add_space(6.0);

                        ui.label(
                            egui::RichText::new(format!(
                                "POS [{}, {}, {}]",
                                config.block_pos.x, config.block_pos.y, config.block_pos.z
                            ))
                            .color(egui::Color32::from_rgb(140, 160, 180))
                            .size(11.0),
                        );
                        ui.add_space(8.0);

            // --- Publish Bindings ---
            let accent = egui::Color32::from_rgb(60, 200, 255);
            let dim_text = egui::Color32::from_rgb(160, 180, 200);

            egui::CollapsingHeader::new(
                egui::RichText::new("PUBLISH CHANNELS").color(accent).size(13.0),
            )
                .default_open(true)
                .show(ui, |ui| {
                    let mut remove_idx = None;
                    for (i, binding) in config.publish_bindings.iter_mut().enumerate() {
                        ui.horizontal(|ui| {
                            ui.label(egui::RichText::new("CH").color(dim_text).size(11.0));
                            channel_name_input(ui, &format!("pub_ch_{i}"), &mut binding.channel_name, &config.available_channels);
                            ui.label(egui::RichText::new("PROP").color(dim_text).size(11.0));
                            egui::ComboBox::from_id_salt(format!("pub_prop_{i}"))
                                .selected_text(property_name(binding.property))
                                .show_ui(ui, |ui| {
                                    for &(name, prop) in SIGNAL_PROPERTY_NAMES {
                                        ui.selectable_value(&mut binding.property, prop, name);
                                    }
                                });
                            if ui.small_button(egui::RichText::new("-").color(egui::Color32::from_rgb(255, 100, 100))).clicked() {
                                remove_idx = Some(i);
                            }
                        });
                    }
                    if let Some(idx) = remove_idx {
                        config.publish_bindings.remove(idx);
                    }
                    if ui.button(egui::RichText::new("+ Add Publish").color(accent)).clicked() {
                        config.publish_bindings.push(
                            voxeldust_core::signal::config::PublishBindingConfig {
                                channel_name: "new-channel".into(),
                                property: SignalProperty::Active,
                            },
                        );
                    }
                });

            ui.add_space(4.0);

            // --- Subscribe Bindings ---
            egui::CollapsingHeader::new(
                egui::RichText::new("SUBSCRIBE CHANNELS").color(accent).size(13.0),
            )
                .default_open(true)
                .show(ui, |ui| {
                    let mut remove_idx = None;
                    for (i, binding) in config.subscribe_bindings.iter_mut().enumerate() {
                        ui.horizontal(|ui| {
                            ui.label(egui::RichText::new("CH").color(dim_text).size(11.0));
                            channel_name_input(ui, &format!("sub_ch_{i}"), &mut binding.channel_name, &config.available_channels);
                            ui.label(egui::RichText::new("PROP").color(dim_text).size(11.0));
                            egui::ComboBox::from_id_salt(format!("sub_prop_{i}"))
                                .selected_text(property_name(binding.property))
                                .show_ui(ui, |ui| {
                                    for &(name, prop) in SIGNAL_PROPERTY_NAMES {
                                        ui.selectable_value(&mut binding.property, prop, name);
                                    }
                                });
                            if ui.small_button(egui::RichText::new("-").color(egui::Color32::from_rgb(255, 100, 100))).clicked() {
                                remove_idx = Some(i);
                            }
                        });
                    }
                    if let Some(idx) = remove_idx {
                        config.subscribe_bindings.remove(idx);
                    }
                    if ui.button(egui::RichText::new("+ Add Subscribe").color(accent)).clicked() {
                        config.subscribe_bindings.push(
                            voxeldust_core::signal::config::SubscribeBindingConfig {
                                channel_name: "new-channel".into(),
                                property: SignalProperty::Throttle,
                            },
                        );
                    }
                });

            ui.add_space(4.0);

            // --- Converter Rules ---
            if !config.converter_rules.is_empty() || config.kind == voxeldust_core::block::FunctionalBlockKind::SignalConverter as u8 {
                egui::CollapsingHeader::new(
                    egui::RichText::new("CONVERTER RULES").color(accent).size(13.0),
                )
                    .default_open(true)
                    .show(ui, |ui| {
                        let mut remove_idx = None;
                        for (i, rule) in config.converter_rules.iter_mut().enumerate() {
                            ui.group(|ui| {
                                ui.horizontal(|ui| {
                                    ui.label(egui::RichText::new("IF").color(egui::Color32::from_rgb(255, 200, 60)).strong());
                                    ui.text_edit_singleline(&mut rule.input_channel);
                                });
                                ui.horizontal(|ui| {
                                    ui.label(egui::RichText::new("THEN").color(egui::Color32::from_rgb(100, 255, 100)).strong());
                                    ui.text_edit_singleline(&mut rule.output_channel);
                                });
                                if ui.small_button(egui::RichText::new("- Remove").color(egui::Color32::from_rgb(255, 100, 100))).clicked() {
                                    remove_idx = Some(i);
                                }
                            });
                        }
                        if let Some(idx) = remove_idx {
                            config.converter_rules.remove(idx);
                        }
                        if ui.button(egui::RichText::new("+ Add Rule").color(accent)).clicked() {
                            config.converter_rules.push(
                                voxeldust_core::signal::config::SignalRuleConfig {
                                    input_channel: "input".into(),
                                    condition: voxeldust_core::signal::converter::SignalCondition::Always,
                                    output_channel: "output".into(),
                                    expression: voxeldust_core::signal::converter::SignalExpression::PassThrough,
                                },
                            );
                        }
                    });
            }

            ui.add_space(4.0);

            // --- Seat Mappings ---
            if !config.seat_mappings.is_empty() || config.kind == voxeldust_core::block::FunctionalBlockKind::Seat as u8 {
                egui::CollapsingHeader::new(
                    egui::RichText::new("SEAT CONTROL MAPPINGS").color(accent).size(13.0),
                )
                    .default_open(true)
                    .show(ui, |ui| {
                        let mut remove_idx = None;
                        for (i, mapping) in config.seat_mappings.iter_mut().enumerate() {
                            ui.horizontal(|ui| {
                                // Control dropdown.
                                egui::ComboBox::from_id_salt(format!("seat_ctrl_{i}"))
                                    .selected_text(mapping.control.label())
                                    .width(120.0)
                                    .show_ui(ui, |ui| {
                                        for &ctrl in SeatControl::available_for_kind(
                                            voxeldust_core::block::FunctionalBlockKind::Seat,
                                        ) {
                                            ui.selectable_value(&mut mapping.control, ctrl, ctrl.label());
                                        }
                                    });
                                ui.label(egui::RichText::new("->").color(accent));
                                // Channel name (free text).
                                channel_name_input(ui, &format!("seat_ch_{i}"), &mut mapping.channel_name, &config.available_channels);
                                // Property dropdown.
                                egui::ComboBox::from_id_salt(format!("seat_prop_{i}"))
                                    .selected_text(property_name(mapping.property))
                                    .width(80.0)
                                    .show_ui(ui, |ui| {
                                        for &(name, prop) in SIGNAL_PROPERTY_NAMES {
                                            ui.selectable_value(&mut mapping.property, prop, name);
                                        }
                                    });
                                if ui.small_button(egui::RichText::new("-").color(egui::Color32::from_rgb(255, 100, 100))).clicked() {
                                    remove_idx = Some(i);
                                }
                            });
                        }
                        if let Some(idx) = remove_idx {
                            config.seat_mappings.remove(idx);
                        }
                        if ui.button(egui::RichText::new("+ Add Binding").color(accent)).clicked() {
                            config.seat_mappings.push(
                                voxeldust_core::signal::config::SeatInputBindingConfig {
                                    control: SeatControl::ThrustForward,
                                    channel_name: "new-channel".into(),
                                    property: SignalProperty::Throttle,
                                },
                            );
                        }
                    });
            }

            // ----- POWER CIRCUITS (reactor blocks only) -----
            if config.power_source.is_some() || config.kind == voxeldust_core::block::FunctionalBlockKind::Reactor as u8 {
                let power_color = egui::Color32::from_rgb(255, 200, 60); // amber for power
                egui::CollapsingHeader::new(
                    egui::RichText::new("POWER CIRCUITS").color(power_color).strong(),
                )
                .default_open(true)
                .show(ui, |ui| {
                    let ps = config.power_source.get_or_insert_with(Default::default);
                    let mut remove_idx = None;
                    for (i, circuit) in ps.circuits.iter_mut().enumerate() {
                        ui.horizontal(|ui| {
                            ui.label(egui::RichText::new("NAME").color(dim_text).small());
                            ui.add(egui::TextEdit::singleline(&mut circuit.name).desired_width(100.0));
                            ui.label(egui::RichText::new("FRAC").color(dim_text).small());
                            ui.add(egui::DragValue::new(&mut circuit.fraction)
                                .range(0.0..=1.0).speed(0.01).fixed_decimals(2));
                            if ui.small_button(
                                egui::RichText::new("-").color(egui::Color32::from_rgb(255, 100, 100)),
                            ).clicked() {
                                remove_idx = Some(i);
                            }
                        });
                    }
                    if let Some(idx) = remove_idx {
                        ps.circuits.remove(idx);
                    }
                    if ui.button(egui::RichText::new("+ Add Circuit").color(power_color)).clicked() {
                        use voxeldust_core::signal::config::PowerCircuitConfig;
                        ps.circuits.push(PowerCircuitConfig { name: "new".into(), fraction: 0.5 });
                    }
                    ui.add_space(4.0);
                    ui.separator();
                    ui.horizontal(|ui| {
                        ui.label(egui::RichText::new("ACCESS").color(dim_text).small());
                        use voxeldust_core::signal::config::PowerAccessConfig;
                        let access_label = match &ps.access {
                            PowerAccessConfig::OwnerOnly => "Owner Only",
                            PowerAccessConfig::AllowList(_) => "Allow List",
                            PowerAccessConfig::Open => "Open",
                        };
                        egui::ComboBox::from_id_salt("power_access")
                            .selected_text(access_label)
                            .show_ui(ui, |ui| {
                                if ui.selectable_label(matches!(ps.access, PowerAccessConfig::OwnerOnly), "Owner Only").clicked() {
                                    ps.access = PowerAccessConfig::OwnerOnly;
                                }
                                if ui.selectable_label(matches!(ps.access, PowerAccessConfig::AllowList(_)), "Allow List").clicked() {
                                    if !matches!(ps.access, PowerAccessConfig::AllowList(_)) {
                                        ps.access = PowerAccessConfig::AllowList(Vec::new());
                                    }
                                }
                                if ui.selectable_label(matches!(ps.access, PowerAccessConfig::Open), "Open").clicked() {
                                    ps.access = PowerAccessConfig::Open;
                                }
                            });
                    });
                    if let voxeldust_core::signal::config::PowerAccessConfig::AllowList(ref mut names) = ps.access {
                        let mut remove_name = None;
                        for (i, name) in names.iter_mut().enumerate() {
                            ui.horizontal(|ui| {
                                ui.add(egui::TextEdit::singleline(name).desired_width(120.0));
                                if ui.small_button(
                                    egui::RichText::new("-").color(egui::Color32::from_rgb(255, 100, 100)),
                                ).clicked() {
                                    remove_name = Some(i);
                                }
                            });
                        }
                        if let Some(idx) = remove_name {
                            names.remove(idx);
                        }
                        if ui.button(egui::RichText::new("+ Add Player").color(power_color)).clicked() {
                            names.push(String::new());
                        }
                    }
                });
            }

            // ----- POWER SOURCE (consumer blocks) -----
            if config.power_consumer.is_some() || !config.nearby_reactors.is_empty() {
                let power_color = egui::Color32::from_rgb(255, 200, 60);
                egui::CollapsingHeader::new(
                    egui::RichText::new("POWER SOURCE").color(power_color).strong(),
                )
                .default_open(true)
                .show(ui, |ui| {
                    let pc = config.power_consumer.get_or_insert_with(Default::default);

                    // Reactor dropdown.
                    ui.horizontal(|ui| {
                        ui.label(egui::RichText::new("REACTOR").color(dim_text).small());
                        let current_label = match pc.reactor_pos {
                            Some(pos) => {
                                config.nearby_reactors.iter()
                                    .find(|r| r.pos == pos)
                                    .map(|r| format!("{} ({},{},{}) — {:.0}m", r.label, r.pos.x, r.pos.y, r.pos.z, r.distance))
                                    .unwrap_or_else(|| format!("({},{},{})", pos.x, pos.y, pos.z))
                            }
                            None => "None".to_string(),
                        };
                        egui::ComboBox::from_id_salt("power_reactor")
                            .selected_text(&current_label)
                            .width(250.0)
                            .show_ui(ui, |ui| {
                                if ui.selectable_label(pc.reactor_pos.is_none(), "None").clicked() {
                                    pc.reactor_pos = None;
                                    pc.circuit.clear();
                                }
                                for reactor in &config.nearby_reactors {
                                    let label = format!("{} ({},{},{}) — {:.0}m",
                                        reactor.label, reactor.pos.x, reactor.pos.y, reactor.pos.z, reactor.distance);
                                    let selected = pc.reactor_pos == Some(reactor.pos);
                                    if ui.selectable_label(selected, &label).clicked() {
                                        pc.reactor_pos = Some(reactor.pos);
                                        // Auto-select first circuit if none selected.
                                        if pc.circuit.is_empty() {
                                            if let Some(first) = reactor.circuits.first() {
                                                pc.circuit = first.clone();
                                            }
                                        }
                                    }
                                }
                            });
                    });

                    // Circuit dropdown (from selected reactor).
                    if let Some(reactor_pos) = pc.reactor_pos {
                        if let Some(reactor_info) = config.nearby_reactors.iter().find(|r| r.pos == reactor_pos) {
                            ui.horizontal(|ui| {
                                ui.label(egui::RichText::new("CIRCUIT").color(dim_text).small());
                                egui::ComboBox::from_id_salt("power_circuit")
                                    .selected_text(if pc.circuit.is_empty() { "—" } else { &pc.circuit })
                                    .show_ui(ui, |ui| {
                                        for c in &reactor_info.circuits {
                                            let selected = pc.circuit == *c;
                                            if ui.selectable_label(selected, c).clicked() {
                                                pc.circuit = c.clone();
                                            }
                                        }
                                    });
                            });
                        }
                        ui.label(egui::RichText::new("Connected").color(egui::Color32::from_rgb(60, 255, 120)).small());
                    } else {
                        ui.label(egui::RichText::new("No reactor selected").color(egui::Color32::from_rgb(255, 200, 60)).small());
                    }
                });
            }

            ui.add_space(12.0);
            // Bottom accent line.
            let rect = ui.available_rect_before_wrap();
            ui.painter().line_segment(
                [egui::pos2(rect.left(), rect.top()), egui::pos2(rect.right(), rect.top())],
                egui::Stroke::new(1.0, egui::Color32::from_rgba_unmultiplied(60, 180, 255, 80)),
            );
            ui.add_space(8.0);

            ui.horizontal(|ui| {
                let save_btn = egui::Button::new(
                    egui::RichText::new("SAVE").color(egui::Color32::from_rgb(60, 255, 120)).strong(),
                );
                if ui.add(save_btn).clicked() {
                    action = ConfigPanelAction::Save;
                }
                ui.add_space(12.0);
                let cancel_btn = egui::Button::new(
                    egui::RichText::new("CANCEL").color(egui::Color32::from_rgb(200, 200, 200)),
                );
                if ui.add(cancel_btn).clicked() {
                    action = ConfigPanelAction::Close;
                }
            });
                    }); // panel_frame
                },
            ); // allocate_ui_at_rect
        }); // CentralPanel

    action
}
