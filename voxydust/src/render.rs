//! Render pass: uniform building, draw calls, scene lighting, egui integration.

use glam::{DVec3, DQuat, Mat4, Quat, Vec3};

use crate::camera::CameraParams;
use crate::gpu::{GpuState, ObjectUniforms, SceneLighting, EYE_HEIGHT, MAX_OBJECTS};
use crate::hud::{self, HudContext};

use voxeldust_core::client_message::WorldStateData;

/// Intermediate state for the render pass: object counts and special indices.
struct RenderObjects {
    object_count: usize,
    ship_interior_start: usize,
    ship_interior_count: usize,
    inside_sphere_indices: Vec<usize>,
}

/// Build object uniforms into the pre-allocated buffer. Returns counts for draw calls.
fn build_uniforms(
    uniform_data: &mut [ObjectUniforms],
    cam: &CameraParams,
    ws: Option<&WorldStateData>,
    current_shard_type: u8,
    player_position: DVec3,
    ship_rotation: DQuat,
) -> RenderObjects {
    let mut object_count = 0usize;
    let mut ship_interior_start = 0usize;
    let mut ship_interior_count = 0usize;
    let mut inside_sphere_indices: Vec<usize> = Vec::new();

    if let Some(ws) = ws {
        // Celestial bodies -- unified rendering for all shards.
        // Track which bodies the camera is inside (for inside-sphere pipeline).
        for body in &ws.bodies {
            if object_count >= MAX_OBJECTS { break; }
            let offset_f64 = body.position - cam.cam_system_pos;
            let dist_to_center = offset_f64.length();
            let camera_inside = dist_to_center < body.radius;
            let offset = offset_f64.as_vec3();
            let scale = (body.radius as f32).max(1.0);
            // Frustum cull bodies the camera is NOT inside.
            if !camera_inside && !cam.frustum.contains_sphere(offset, scale) {
                continue;
            }
            let model = Mat4::from_translation(offset) * Mat4::from_scale(Vec3::splat(scale));
            let mvp = cam.vp * model;
            let alpha = if body.body_id == 0 { 1.0 } else { 0.4 };
            let mut obj: ObjectUniforms = bytemuck::Zeroable::zeroed();
            obj.mvp = mvp.to_cols_array_2d();
            obj.model = model.to_cols_array_2d();
            obj.color = [body.color[0], body.color[1], body.color[2], alpha];
            // Stars (body_id 0) are emissive; planets are rocky (metallic=0.0, roughness=0.7).
            obj.material = if body.body_id == 0 { [0.0, 0.5, 0.0, 0.0] } else { [0.0, 0.7, 0.0, 0.0] };
            if camera_inside {
                inside_sphere_indices.push(object_count);
            }
            uniform_data[object_count] = obj;
            object_count += 1;
        }

        // Surface reference grid -- ship-sized spots near the camera for movement feedback.
        for body in &ws.bodies {
            if body.body_id == 0 { continue; } // skip star
            if object_count + 200 >= MAX_OBJECTS { break; }

            let body_center = body.position;
            let r = body.radius;
            let to_cam = cam.cam_system_pos - body_center;
            let cam_dist = to_cam.length();
            if cam_dist < 1.0 || cam_dist > r * 3.0 { continue; } // too close/far

            let altitude = cam_dist - r;
            let cam_dir = to_cam / cam_dist;

            // Spot size: ship-length (8m). Grid spacing: 4x spot size for checkerboard density.
            let spot_size = 8.0_f64; // meters -- matches ship length
            let grid_spacing = spot_size * 4.0; // 32m between grid lines
            // Angular spacing on the sphere surface.
            let angle_step = grid_spacing / r; // radians per grid cell

            // Visible angular radius from camera's sub-surface point.
            // At low altitude, show a wide patch. At high altitude, show more but
            // limited by MAX_OBJECTS budget.
            let visible_angle = (altitude / r).min(0.3).max(0.005); // radians
            let half_steps = ((visible_angle / angle_step) as i32).min(30).max(3);

            // Snap grid center to the nearest grid intersection so spots are
            // fixed to the planet surface, not the camera. As the ship moves,
            // spots stay put and new ones appear at the edges.
            let cam_lat = cam_dir.y.asin();
            let cam_lon = cam_dir.z.atan2(cam_dir.x);
            let center_lat_i = (cam_lat / angle_step).round() as i32;
            let center_lon_i = (cam_lon / angle_step).round() as i32;

            for lat_i in (center_lat_i - half_steps)..=(center_lat_i + half_steps) {
                for lon_i in (center_lon_i - half_steps)..=(center_lon_i + half_steps) {
                    if object_count >= MAX_OBJECTS { break; }
                    // Checkerboard: skip half the spots for visual clarity.
                    if (lat_i + lon_i) % 2 != 0 { continue; }

                    let lat = lat_i as f64 * angle_step;
                    let lon = lon_i as f64 * angle_step;

                    // Spherical to cartesian (on planet surface).
                    let cos_lat = lat.cos();
                    let spot_dir = DVec3::new(
                        cos_lat * lon.cos(),
                        lat.sin(),
                        cos_lat * lon.sin(),
                    );
                    let spot_pos = body_center + spot_dir * (r + 0.02); // 2cm above surface

                    let offset = (spot_pos - cam.cam_system_pos).as_vec3();
                    let dist_to_spot = offset.length();
                    // Cull spots behind the horizon or too far to see.
                    if dist_to_spot > (altitude * 3.0) as f32 + 500.0 { continue; }
                    // Frustum cull surface spots.
                    if !cam.frustum.contains_sphere(offset, spot_size as f32) { continue; }

                    // Orient flat on surface.
                    let spot_up = spot_dir.as_vec3().normalize();
                    let spot_rot = Quat::from_rotation_arc(Vec3::Y, spot_up);
                    let spot_render = spot_size as f32;
                    let model = Mat4::from_translation(offset)
                        * Mat4::from_quat(spot_rot)
                        * Mat4::from_scale(Vec3::new(spot_render, 0.01, spot_render));
                    let mvp = cam.vp * model;
                    let mut obj: ObjectUniforms = bytemuck::Zeroable::zeroed();
                    obj.mvp = mvp.to_cols_array_2d();
                    obj.model = model.to_cols_array_2d();
                    // Alternating bright/dim for depth perception.
                    let shade = if (lat_i.abs() + lon_i.abs()) % 4 == 0 { 0.8 } else { 0.4 };
                    obj.color = [shade * body.color[0], shade * body.color[1], shade * body.color[2], 0.9];
                    obj.material = [0.0, 0.8, 0.0, 0.0]; // matte surface marker
                    uniform_data[object_count] = obj;
                    object_count += 1;
                }
            }
        }

        for ship in &ws.ships {
            if object_count >= MAX_OBJECTS { break; }
            let offset = (ship.position - cam.cam_system_pos).as_vec3();
            // Frustum cull ships (bounding sphere radius ~8m for ship length).
            if !cam.frustum.contains_sphere(offset, 8.0) { continue; }
            let ship_rot = Mat4::from_quat(ship.rotation.as_quat());
            // Ship box: 4m wide, 3m tall, 8m long (same dimensions as interior).
            let scale = Mat4::from_scale(Vec3::new(4.0, 3.0, 8.0));
            let model = Mat4::from_translation(offset) * ship_rot * scale;
            let mvp = cam.vp * model;
            let mut obj: ObjectUniforms = bytemuck::Zeroable::zeroed();
            obj.mvp = mvp.to_cols_array_2d();
            obj.model = model.to_cols_array_2d();
            obj.color = if ship.is_own_ship {
                [0.4, 0.5, 0.6, 0.7] // own ship: lighter
            } else {
                [0.3, 0.3, 0.35, 0.7] // other ships
            };
            obj.material = [0.8, 0.3, 0.0, 0.0]; // metallic hull
            uniform_data[object_count] = obj;
            object_count += 1;
        }

        // Ship interior (only when in ship shard).
        if current_shard_type == 2 {
            ship_interior_start = object_count;

            // Ship interior rendering -- standard space game approach:
            // Camera is parented to ship (view matrix contains inverse(ship_rotation)).
            // Interior objects use model = translate(ship_origin_to_cam) * ship_rot_mat.
            // The ship rotation in model cancels with inverse in view -> walls stay fixed.
            // Exterior objects have no ship rotation in model -> they rotate when ship turns.
            let ship_rot_mat = Mat4::from_quat(ship_rotation.as_quat());

            // Ship origin (0,0,0 in ship-local) relative to camera.
            // Camera is at player_position + eye_height in ship-local, then rotated.
            // Ship origin offset in ship-local = -(player_position + eye_height).
            let origin_local = -(player_position + DVec3::new(0.0, EYE_HEIGHT, 0.0));
            let ship_origin_offset = (ship_rotation * origin_local).as_vec3();

            // Ship box walls: vertices are in ship-local space.
            let model = Mat4::from_translation(ship_origin_offset) * ship_rot_mat;
            let mvp = cam.vp * model;
            if object_count < MAX_OBJECTS {
                let mut obj: ObjectUniforms = bytemuck::Zeroable::zeroed();
                obj.mvp = mvp.to_cols_array_2d();
                obj.model = model.to_cols_array_2d();
                obj.color = [0.3, 0.3, 0.35, 0.4];
                obj.material = [0.6, 0.4, 0.0, 0.0]; // metallic interior
                uniform_data[object_count] = obj;
                ship_interior_count = 1;
                object_count += 1;
            }

            // Pilot seat marker: ship-local (0, 0.5, -3) -> rotate + offset.
            let seat_local = Vec3::new(0.0, 0.5, -3.0);
            let seat_model = Mat4::from_translation(ship_origin_offset) * ship_rot_mat
                * Mat4::from_translation(seat_local) * Mat4::from_scale(Vec3::splat(0.3));
            if object_count < MAX_OBJECTS {
                let mut obj: ObjectUniforms = bytemuck::Zeroable::zeroed();
                obj.mvp = (cam.vp * seat_model).to_cols_array_2d();
                obj.model = seat_model.to_cols_array_2d();
                obj.color = [0.2, 0.5, 1.0, 1.0];
                obj.material = [0.0, 0.5, 0.0, 0.0];
                uniform_data[object_count] = obj;
                object_count += 1;
            }

            // Exit door marker: ship-local (2.0, 0.5, 0) -> rotate + offset.
            let door_local = Vec3::new(2.0, 0.5, 0.0);
            let door_model = Mat4::from_translation(ship_origin_offset) * ship_rot_mat
                * Mat4::from_translation(door_local) * Mat4::from_scale(Vec3::splat(0.3));
            if object_count < MAX_OBJECTS {
                let mut obj: ObjectUniforms = bytemuck::Zeroable::zeroed();
                obj.mvp = (cam.vp * door_model).to_cols_array_2d();
                obj.model = door_model.to_cols_array_2d();
                obj.color = [0.0, 1.0, 0.3, 1.0]; // green emissive
                obj.material = [0.0, 0.5, 0.0, 0.0];
                uniform_data[object_count] = obj;
                object_count += 1;
            }
        }
    }

    RenderObjects {
        object_count,
        ship_interior_start,
        ship_interior_count,
        inside_sphere_indices,
    }
}

/// Compute orthographic light view-projection matrix for directional shadow mapping.
/// Centers the shadow frustum on the camera position, looking along the sun direction.
fn compute_light_vp(sun_dir: Vec3) -> Mat4 {
    let half_extent = 200.0_f32; // 200m half-width covers nearby scene

    // Sun shines from `sun_dir` toward the origin. The light "eye" is far along sun_dir.
    let light_eye = sun_dir * 500.0; // 500m away from center
    let light_target = Vec3::ZERO;
    // Choose a stable up vector that isn't parallel to sun_dir.
    let light_up = if sun_dir.y.abs() > 0.99 { Vec3::Z } else { Vec3::Y };

    let light_view = Mat4::look_at_rh(light_eye, light_target, light_up);
    // Standard depth (near=0, far=1). The light frustum is 1000m deep (light at 500m, so 500m behind + 500m in front).
    let light_proj = Mat4::orthographic_rh(
        -half_extent, half_extent,
        -half_extent, half_extent,
        0.1, 1000.0,
    );
    light_proj * light_view
}

/// Build SceneLighting from the current WorldState.
fn build_scene_lighting(ws: Option<&WorldStateData>) -> SceneLighting {
    if let Some(ws) = ws {
        if let Some(ref l) = ws.lighting {
            let dir = l.sun_direction.normalize().as_vec3();
            let light_vp = compute_light_vp(dir);
            SceneLighting {
                sun_direction: [dir.x, dir.y, dir.z, 0.0],
                sun_color: [l.sun_color[0], l.sun_color[1], l.sun_color[2], l.sun_intensity],
                ambient: [l.ambient, 0.0, 0.0, 0.0],
                camera_pos: [0.0, 0.0, 0.0, 0.0],
                light_vp: light_vp.to_cols_array_2d(),
            }
        } else {
            default_scene_lighting()
        }
    } else {
        default_scene_lighting()
    }
}

fn default_scene_lighting() -> SceneLighting {
    let dir = Vec3::new(0.182, 0.607, 0.303);
    let light_vp = compute_light_vp(dir);
    SceneLighting {
        sun_direction: [0.182, 0.607, 0.303, 0.0], // normalize(0.3, 1.0, 0.5)
        sun_color: [1.0, 1.0, 1.0, 1.0],
        ambient: [0.1, 0.0, 0.0, 0.0],
        camera_pos: [0.0, 0.0, 0.0, 0.0],
        light_vp: light_vp.to_cols_array_2d(),
    }
}

/// Execute the full render frame: build uniforms, 3D pass, egui HUD, present.
pub fn render_frame(
    gpu: &mut GpuState,
    window: &winit::window::Window,
    uniform_data: &mut [ObjectUniforms],
    cam: &CameraParams,
    latest_world_state: Option<&WorldStateData>,
    current_shard_type: u8,
    player_position: DVec3,
    player_velocity: DVec3,
    ship_rotation: DQuat,
    is_piloting: bool,
    connected: bool,
    selected_thrust_tier: u8,
    engines_off: bool,
    autopilot_target: Option<usize>,
    trajectory_plan: Option<&voxeldust_core::autopilot::TrajectoryPlan>,
    system_params: Option<&voxeldust_core::system::SystemParams>,
    frame_count: u64,
) {
    let frame = match gpu.surface.get_current_texture() { Ok(f) => f, Err(_) => return };
    let view = frame.texture.create_view(&wgpu::TextureViewDescriptor::default());

    // Build object uniforms.
    let ro = build_uniforms(
        uniform_data, cam, latest_world_state, current_shard_type,
        player_position, ship_rotation,
    );

    // Write scene lighting data from server WorldState.
    let scene_lighting = build_scene_lighting(latest_world_state);
    gpu.queue.write_buffer(&gpu.scene_lighting_buf, 0, bytemuck::bytes_of(&scene_lighting));

    let light_vp = Mat4::from_cols_array_2d(&scene_lighting.light_vp);

    // -- Shadow pass --
    // Overwrite MVPs with light-space projections for the shadow depth pass.
    if ro.object_count > 0 {
        // Compute shadow-space MVPs: replace each object's mvp with light_vp * model.
        for i in 0..ro.object_count {
            let model = Mat4::from_cols_array_2d(&uniform_data[i].model);
            uniform_data[i].mvp = (light_vp * model).to_cols_array_2d();
        }
        gpu.queue.write_buffer(&gpu.uniform_buf, 0, bytemuck::cast_slice(&uniform_data[..ro.object_count]));
    }

    let mut encoder = gpu.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });

    // Shadow render pass: depth-only into the shadow texture.
    {
        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("shadow"),
            color_attachments: &[],
            depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                view: &gpu.shadow_texture_view,
                depth_ops: Some(wgpu::Operations { load: wgpu::LoadOp::Clear(1.0), store: wgpu::StoreOp::Store }),
                stencil_ops: None,
            }),
            ..Default::default()
        });
        pass.set_pipeline(&gpu.shadow_pipeline);
        pass.set_vertex_buffer(0, gpu.sphere_vertex_buf.slice(..));
        pass.set_index_buffer(gpu.sphere_index_buf.slice(..), wgpu::IndexFormat::Uint32);

        // Draw all non-emissive sphere objects into the shadow map.
        let sphere_end = if ro.ship_interior_count > 0 { ro.ship_interior_start } else { ro.object_count };
        for i in 0..sphere_end {
            // Skip emissive objects (stars) -- they cast no shadows.
            if uniform_data[i].color[3] > 0.5 { continue; }
            pass.set_bind_group(0, &gpu.bind_group, &[(i as u32) * 256]);
            pass.draw_indexed(0..gpu.sphere_index_count, 0, 0..1);
        }

        // Shadow for ship exterior boxes (ship_interior objects).
        if ro.ship_interior_count > 0 {
            pass.set_vertex_buffer(0, gpu.box_vertex_buf.slice(..));
            pass.set_index_buffer(gpu.box_index_buf.slice(..), wgpu::IndexFormat::Uint32);
            pass.set_bind_group(0, &gpu.bind_group, &[(ro.ship_interior_start as u32) * 256]);
            pass.draw_indexed(0..gpu.box_index_count, 0, 0..1);
        }
    }

    // -- Main pass --
    // Restore camera-space MVPs by re-running the uniform build.
    let ro = build_uniforms(
        uniform_data, cam, latest_world_state, current_shard_type,
        player_position, ship_rotation,
    );

    // Upload camera-space uniforms for the main pass.
    if ro.object_count > 0 {
        gpu.queue.write_buffer(&gpu.uniform_buf, 0, bytemuck::cast_slice(&uniform_data[..ro.object_count]));
    }

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
                depth_ops: Some(wgpu::Operations { load: wgpu::LoadOp::Clear(0.0), store: wgpu::StoreOp::Store }),
                stencil_ops: None,
            }),
            ..Default::default()
        });

        // Set scene-wide lighting uniform (group 1) -- constant for entire pass.
        pass.set_bind_group(1, &gpu.scene_bind_group, &[]);
        // Set shadow map (group 2) -- constant for entire pass.
        pass.set_bind_group(2, &gpu.shadow_bind_group, &[]);

        // Draw celestial bodies + ship markers (spheres).
        pass.set_vertex_buffer(0, gpu.sphere_vertex_buf.slice(..));
        pass.set_index_buffer(gpu.sphere_index_buf.slice(..), wgpu::IndexFormat::Uint32);

        // Celestial bodies: indices 0..ship_interior_start.
        // Use inside-sphere pipeline (no backface culling) for bodies the camera is inside.
        let sphere_end = if ro.ship_interior_count > 0 { ro.ship_interior_start } else { ro.object_count };
        for i in 0..sphere_end {
            if ro.inside_sphere_indices.contains(&i) {
                pass.set_pipeline(&gpu.sphere_inside_pipeline);
            } else {
                pass.set_pipeline(&gpu.pipeline);
            }
            pass.set_bind_group(0, &gpu.bind_group, &[(i as u32) * 256]);
            pass.draw_indexed(0..gpu.sphere_index_count, 0, 0..1);
        }
        pass.set_pipeline(&gpu.pipeline);

        // Ship markers (seat + door spheres): indices ship_interior_start+1..object_count.
        if ro.ship_interior_count > 0 {
            for i in (ro.ship_interior_start + 1)..ro.object_count {
                pass.set_bind_group(0, &gpu.bind_group, &[(i as u32) * 256]);
                pass.draw_indexed(0..gpu.sphere_index_count, 0, 0..1);
            }
        }

        // Draw ship interior box: index ship_interior_start.
        if ro.ship_interior_count > 0 {
            pass.set_vertex_buffer(0, gpu.box_vertex_buf.slice(..));
            pass.set_index_buffer(gpu.box_index_buf.slice(..), wgpu::IndexFormat::Uint32);
            pass.set_bind_group(0, &gpu.bind_group, &[(ro.ship_interior_start as u32) * 256]);
            pass.draw_indexed(0..gpu.box_index_count, 0, 0..1);
        }
    }

    // Compute autopilot trajectory for HUD (done in main.rs before calling render_frame).
    // egui HUD.
    let hud_ctx = HudContext {
        latest_world_state,
        cam_system_pos: cam.cam_system_pos,
        vp: cam.vp,
        player_position,
        player_velocity,
        current_shard_type,
        is_piloting,
        connected,
        selected_thrust_tier,
        engines_off,
        autopilot_target,
        trajectory_plan,
        system_params,
        frame_count,
    };
    let full_output = hud::run_hud(gpu, window, &hud_ctx);
    hud::render_egui(gpu, &mut encoder, &view, full_output);

    gpu.queue.submit(std::iter::once(encoder.finish()));
    frame.present();
}
