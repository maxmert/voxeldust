//! Render pass: uniform building, draw calls, scene lighting, egui integration.

use glam::{DVec3, DQuat, Mat4, Quat, Vec3};

use crate::block_render::{BlockRenderer, ChunkGpuMesh};
use crate::camera::CameraParams;
use crate::gpu::{GpuState, ObjectUniforms, SceneLighting, EYE_HEIGHT, MAX_OBJECTS};
use crate::hud::{self, HudContext};

use voxeldust_core::block::palette::CHUNK_SIZE;
use voxeldust_core::client_message::WorldStateData;

/// Transform for a ship that should be rendered with block chunks.
struct BlockShipTransform {
    /// Camera-relative base transform: translates chunk-local vertices to world space.
    base_transform: Mat4,
}

/// Intermediate state for the render pass: object counts and special indices.
struct RenderObjects {
    object_count: usize,
    /// Start index in the uniform buffer reserved for block chunk uniforms.
    block_uniform_start: usize,
    /// Ships to render with block chunks (interior own ship + exterior ships with cached data).
    block_ships: Vec<BlockShipTransform>,
    inside_sphere_indices: Vec<usize>,
}

/// Build object uniforms into the pre-allocated buffer. Returns counts for draw calls.
fn build_uniforms(
    uniform_data: &mut [ObjectUniforms],
    cam: &CameraParams,
    ws: Option<&WorldStateData>,
    secondary_ws: Option<&WorldStateData>,
    current_shard_type: u8,
    player_position: DVec3,
    ship_rotation: DQuat,
) -> RenderObjects {
    let mut object_count = 0usize;
    let mut inside_sphere_indices: Vec<usize> = Vec::new();

    if let Some(ws) = ws {
        // Celestial bodies -- unified rendering for all shards.
        // Track which bodies the camera is inside (for inside-sphere pipeline).
        // The primary always renders all celestial bodies (sphere + spots) — these are
        // in sync with the camera's coordinate frame. The secondary shard contributes
        // only surface-detail data (ships, future: terrain chunks) that the primary lacks.
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
    }

    // Collect ship transforms for block-based rendering.
    // All ships (interior and exterior) are rendered from block chunks.
    let block_uniform_start = object_count;
    let mut block_ships: Vec<BlockShipTransform> = Vec::new();

    // Interior ship (player is inside — shard_type 2).
    if current_shard_type == voxeldust_core::client_message::shard_type::SHIP {
        let ship_rot_mat = Mat4::from_quat(ship_rotation.as_quat());
        let origin_local = -(player_position + DVec3::new(0.0, EYE_HEIGHT, 0.0));
        let ship_origin_offset = (ship_rotation * origin_local).as_vec3();
        let base_transform = Mat4::from_translation(ship_origin_offset) * ship_rot_mat;
        block_ships.push(BlockShipTransform { base_transform });
    }

    // Exterior ships from primary WorldState.
    if let Some(ws) = ws {
        for ship in &ws.ships {
            let offset = (ship.position - cam.cam_system_pos).as_vec3();
            if !cam.frustum.contains_sphere(offset, 16.0) { continue; }
            let ship_rot_mat = Mat4::from_quat(ship.rotation.as_quat());
            let base_transform = Mat4::from_translation(offset) * ship_rot_mat;
            block_ships.push(BlockShipTransform { base_transform });
        }
    }

    // Exterior ships from secondary WorldState (dual-shard compositing).
    if let Some(sec_ws) = secondary_ws {
        for ship in &sec_ws.ships {
            let ship_system_pos = sec_ws.origin + ship.position;
            let offset = (ship_system_pos - cam.cam_system_pos).as_vec3();
            if !cam.frustum.contains_sphere(offset, 16.0) { continue; }
            let ship_rot_mat = Mat4::from_quat(ship.rotation.as_quat());
            let base_transform = Mat4::from_translation(offset) * ship_rot_mat;
            block_ships.push(BlockShipTransform { base_transform });
        }
    }

    RenderObjects {
        object_count,
        block_uniform_start,
        block_ships,
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
    // Dim interstellar ambient — used as fallback when no host is sending
    // scene updates (e.g., during warp HostSwitch gap before galaxy shard
    // scene data arrives). Must not be bright white.
    let dir = Vec3::new(0.0, -1.0, 0.0);
    let light_vp = compute_light_vp(dir);
    SceneLighting {
        sun_direction: [0.0, -1.0, 0.0, 0.0],
        sun_color: [0.4, 0.4, 0.5, 0.15],
        ambient: [0.06, 0.0, 0.0, 0.0],
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
    secondary_world_state: Option<&WorldStateData>,
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
    server_autopilot: Option<&voxeldust_core::shard_message::AutopilotSnapshotData>,
    system_params: Option<&voxeldust_core::system::SystemParams>,
    frame_count: u64,
    star_instance_count: u32,
    warp_target_star: Option<hud::WarpTargetInfo>,
    block_renderer: Option<&BlockRenderer>,
    block_target: Option<&voxeldust_core::block::raycast::BlockHit>,
    block_indicators: &[(glam::Vec3, u8, char)],
    config_state: Option<&mut voxeldust_core::signal::config::BlockSignalConfig>,
) -> hud::ConfigPanelAction {
    let frame = match gpu.surface.get_current_texture() { Ok(f) => f, Err(_) => return hud::ConfigPanelAction::None };
    let view = frame.texture.create_view(&wgpu::TextureViewDescriptor::default());

    // Build object uniforms.
    let ro = build_uniforms(
        uniform_data, cam, latest_world_state, secondary_world_state,
        current_shard_type, player_position, ship_rotation,
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
        for i in 0..ro.object_count {
            if uniform_data[i].color[3] > 0.5 { continue; }
            pass.set_bind_group(0, &gpu.bind_group, &[(i as u32) * 256]);
            pass.draw_indexed(0..gpu.sphere_index_count, 0, 0..1);
        }

        // TODO: block chunk shadow pass (requires computing shadow-space MVPs for chunks).
    }

    // -- Star pass (before main pass, additive blend, no depth write) --
    {
        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("stars"),
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

        if star_instance_count > 0 {
            pass.set_pipeline(&gpu.star_pipeline);
            pass.set_bind_group(0, &gpu.star_bind_group, &[]);
            pass.set_vertex_buffer(0, gpu.star_quad_vertex_buf.slice(..));
            pass.set_vertex_buffer(1, gpu.star_instance_buf.slice(..));
            pass.set_index_buffer(gpu.star_quad_index_buf.slice(..), wgpu::IndexFormat::Uint32);
            pass.draw_indexed(0..6, 0, 0..star_instance_count);
        }
    }

    // -- Main pass --
    // Restore camera-space MVPs by re-running the uniform build.
    let ro = build_uniforms(
        uniform_data, cam, latest_world_state, secondary_world_state,
        current_shard_type, player_position, ship_rotation,
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
                    load: wgpu::LoadOp::Load, // preserve stars from previous pass
                    store: wgpu::StoreOp::Store,
                },
            })],
            depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                view: &gpu.depth_view,
                depth_ops: Some(wgpu::Operations { load: wgpu::LoadOp::Load, store: wgpu::StoreOp::Store }),
                stencil_ops: None,
            }),
            ..Default::default()
        });

        // Set scene-wide lighting uniform (group 1) -- constant for entire pass.
        pass.set_bind_group(1, &gpu.scene_bind_group, &[]);
        // Set shadow map (group 2) -- constant for entire pass.
        pass.set_bind_group(2, &gpu.shadow_bind_group, &[]);

        // Draw celestial bodies (spheres).
        pass.set_vertex_buffer(0, gpu.sphere_vertex_buf.slice(..));
        pass.set_index_buffer(gpu.sphere_index_buf.slice(..), wgpu::IndexFormat::Uint32);

        for i in 0..ro.object_count {
            if ro.inside_sphere_indices.contains(&i) {
                pass.set_pipeline(&gpu.sphere_inside_pipeline);
            } else {
                pass.set_pipeline(&gpu.pipeline);
            }
            pass.set_bind_group(0, &gpu.bind_group, &[(i as u32) * 256]);
            pass.draw_indexed(0..gpu.sphere_index_count, 0, 0..1);
        }

        // Draw all ships as block chunks. Each ship has a base transform and
        // all cached chunks are rendered with per-chunk offsets from that base.
        if let Some(br) = block_renderer {
            if br.has_chunks() && !ro.block_ships.is_empty() {
                let block_start = ro.block_uniform_start;
                let mut block_draws: Vec<(usize, &ChunkGpuMesh)> = Vec::new();
                let mut chunk_idx = 0usize;

                // For each ship, draw all cached chunks with that ship's transform.
                // Currently all sources share the same chunk data (own ship).
                // In the future, each ship would have its own chunk source.
                for ship_transform in &ro.block_ships {
                    for source in br.active_sources() {
                        for (chunk_pos, chunk_mesh) in br.chunks_for_source(source) {
                            let uniform_idx = block_start + chunk_idx;
                            if uniform_idx >= MAX_OBJECTS { break; }

                            let model = BlockRenderer::chunk_model_matrix(
                                ship_transform.base_transform,
                                chunk_pos,
                                CHUNK_SIZE as f64,
                            );
                            let mvp = cam.vp * model;

                            let mut obj: ObjectUniforms = bytemuck::Zeroable::zeroed();
                            obj.mvp = mvp.to_cols_array_2d();
                            obj.model = model.to_cols_array_2d();
                            obj.color = [1.0, 1.0, 1.0, 0.0];
                            obj.material = [0.1, 0.7, 0.0, 0.0];
                            uniform_data[uniform_idx] = obj;

                            block_draws.push((uniform_idx, chunk_mesh));
                            chunk_idx += 1;
                        }
                    }
                }

                // Upload all chunk uniforms in one batch.
                if chunk_idx > 0 {
                    gpu.queue.write_buffer(
                        &gpu.uniform_buf,
                        (block_start as u64) * 256,
                        bytemuck::cast_slice(&uniform_data[block_start..block_start + chunk_idx]),
                    );

                    // Record draw calls.
                    pass.set_pipeline(&br.pipeline);
                    pass.set_bind_group(1, &gpu.scene_bind_group, &[]);
                    pass.set_bind_group(2, &gpu.shadow_bind_group, &[]);

                    for &(uniform_idx, chunk_mesh) in &block_draws {
                        pass.set_bind_group(0, &gpu.bind_group, &[(uniform_idx as u32) * 256]);
                        pass.set_vertex_buffer(0, chunk_mesh.vertex_buf.slice(..));
                        pass.set_index_buffer(chunk_mesh.index_buf.slice(..), wgpu::IndexFormat::Uint32);
                        pass.draw_indexed(0..chunk_mesh.index_count, 0, 0..1);
                    }
                }
            }
        }

        // Block highlight: render a translucent cube at the targeted block.
        if let Some(target) = block_target {
            if current_shard_type == voxeldust_core::client_message::shard_type::SHIP && !ro.block_ships.is_empty() {
                let base_transform = ro.block_ships[0].base_transform;

                // Block center in ship-local space.
                let block_center = Vec3::new(
                    target.world_pos.x as f32 + 0.5,
                    target.world_pos.y as f32 + 0.5,
                    target.world_pos.z as f32 + 0.5,
                );
                let model = base_transform
                    * Mat4::from_translation(block_center)
                    * Mat4::from_scale(Vec3::splat(0.505)); // slightly larger than 0.5 half-extent
                let mvp = cam.vp * model;

                // Use a uniform slot after all block chunks.
                let highlight_idx = ro.block_uniform_start + 64; // safe margin
                if highlight_idx < MAX_OBJECTS {
                    let mut obj: ObjectUniforms = bytemuck::Zeroable::zeroed();
                    obj.mvp = mvp.to_cols_array_2d();
                    obj.model = model.to_cols_array_2d();
                    obj.color = [0.8, 0.9, 1.0, 0.0]; // bright white-blue
                    obj.material = [0.0, 0.3, 1.0, 0.0]; // glass = 1.0 (screen-door transparency)
                    uniform_data[highlight_idx] = obj;
                    gpu.queue.write_buffer(
                        &gpu.uniform_buf,
                        (highlight_idx as u64) * 256,
                        bytemuck::bytes_of(&uniform_data[highlight_idx]),
                    );

                    // Draw as sphere with inside pipeline (no backface culling).
                    pass.set_pipeline(&gpu.sphere_inside_pipeline);
                    pass.set_bind_group(1, &gpu.scene_bind_group, &[]);
                    pass.set_bind_group(2, &gpu.shadow_bind_group, &[]);
                    pass.set_vertex_buffer(0, gpu.sphere_vertex_buf.slice(..));
                    pass.set_index_buffer(gpu.sphere_index_buf.slice(..), wgpu::IndexFormat::Uint32);
                    pass.set_bind_group(0, &gpu.bind_group, &[(highlight_idx as u32) * 256]);
                    pass.draw_indexed(0..gpu.sphere_index_count, 0, 0..1);
                }
            }
        }
    }

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
        server_autopilot,
        system_params,
        frame_count,
        warp_target_star,
        block_indicators,
    };
    let (full_output, panel_action) = hud::run_hud(gpu, window, &hud_ctx, config_state);
    hud::render_egui(gpu, &mut encoder, &view, full_output);

    gpu.queue.submit(std::iter::once(encoder.finish()));
    frame.present();

    panel_action
}
