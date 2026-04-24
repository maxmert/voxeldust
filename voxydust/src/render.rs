//! Render pass: uniform building, draw calls, scene lighting, egui integration.

use glam::{DVec3, DQuat, Mat4, Quat, Vec3, Vec4};
use std::sync::atomic::AtomicU64;

/// Unix-ms wall-clock deadline until which render-frame diagnostics are
/// logged every frame instead of every 40. Set by main.rs's Transitioning
/// handler to `now + 500 ms` so the full shard-switch window is captured.
pub static TRACE_RENDER_UNTIL_MS: AtomicU64 = AtomicU64::new(0);

use crate::block_render::{BlockRenderer, ChunkGpuMesh};
use crate::camera::CameraParams;

use voxeldust_core::block::client_chunks::ChunkSourceId;
use crate::gpu::{
    GpuState, ObjectUniforms, SceneLighting, ShadowCascades, AtmosphereUniforms,
    EYE_HEIGHT, MAX_OBJECTS, SHADOW_MAP_SIZE, NUM_CASCADES,
};
use crate::graphics_settings::{GraphicsSettings, RenderConfig};
use crate::hud::{self, HudContext};

use voxeldust_core::block::palette::CHUNK_SIZE;
use voxeldust_core::client_message::WorldStateData;
use voxeldust_core::system::{SystemParams, PlanetParams};

/// Cascade split distances (view-space Z thresholds) and the corresponding
/// orthographic light VP matrices for cascaded shadow maps.
const CASCADE_SPLITS: [f32; 4] = [15.0, 60.0, 200.0, 800.0];
const CASCADE_NEAR: f32 = 0.1;

/// Compute the 4 cascade light-space view-projection matrices for CSM.
///
/// Each cascade is a sphere centered on the **camera position** with radius
/// equal to the cascade's far split distance. This is fully rotation-invariant:
/// rotating the camera does not move the shadow at all.
fn compute_cascade_matrices(
    sun_dir: Vec3,
    _cam: &CameraParams,
) -> [Mat4; 4] {
    let mut matrices = [Mat4::IDENTITY; 4];

    // Stable up vector not parallel to sun direction.
    let light_up = if sun_dir.y.abs() > 0.99 { Vec3::Z } else { Vec3::Y };

    // Camera is always at origin in floating-origin rendering.
    let center = Vec3::ZERO;

    for i in 0..4 {
        // Radius = cascade far distance. The sphere covers everything within
        // that distance from the camera, regardless of look direction.
        let radius = CASCADE_SPLITS[i];

        // Texel size for snapping (prevents sub-texel shimmer on camera translation).
        let texel_size = (radius * 2.0) / SHADOW_MAP_SIZE as f32;

        // Light view: look at the camera position from the sun direction.
        let light_eye = center + sun_dir * (radius + 500.0);
        let light_view = Mat4::look_at_rh(light_eye, center, light_up);

        // Snap the ortho projection to world-space texel boundaries.
        // Project camera position to light space and round to texel grid.
        let center_ls = (light_view * center.extend(1.0)).truncate();
        let snapped_x = (center_ls.x / texel_size).round() * texel_size;
        let snapped_y = (center_ls.y / texel_size).round() * texel_size;
        let offset_x = snapped_x - center_ls.x;
        let offset_y = snapped_y - center_ls.y;

        let light_proj = Mat4::orthographic_rh(
            -radius + offset_x, radius + offset_x,
            -radius + offset_y, radius + offset_y,
            0.1, radius * 2.0 + 1000.0,
        );

        matrices[i] = light_proj * light_view;
    }

    matrices
}

/// Transform for a ship that should be rendered with block chunks.
struct BlockShipTransform {
    /// Camera-relative base transform: translates chunk-local vertices to world space.
    base_transform: Mat4,
    /// Which chunk source provides this ship's blocks. Pre-resolved so the
    /// render loop draws chunks from exactly one source per ship — no more
    /// all-sources-at-all-transforms cross product (which only coincidentally
    /// worked for the single-ship case).
    source: ChunkSourceId,
}

/// View of a ship the client just left (SHIP → !SHIP transition) that
/// should be rendered in absolute system-space during the grace window,
/// independent of the current primary's coordinate frame. This is what
/// makes the exit seamless: the ship's system-space position doesn't
/// change when the client's "primary shard" does, so rendering against
/// it is blink-free.
#[derive(Clone, Copy)]
pub struct DepartedShipView {
    pub source: ChunkSourceId,
    pub world_position: DVec3,
    pub rotation: DQuat,
    pub bounding_radius: f32,
    /// Absolute system-space camera position, used to compute the offset
    /// for rendering. The caller must compute this consistently with
    /// whatever frame the current shard's `cam_system_pos` is in —
    /// typically: SHIP → already system-space; PLANET → add `ws.origin`.
    pub cam_world: DVec3,
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
    grace_fallback_ws: Option<&WorldStateData>,
    interpolated_bodies: &[voxeldust_core::client_message::CelestialBodyData],
    current_shard_type: u8,
    shard_seed: u64,
    player_position: DVec3,
    ship_rotation: DQuat,
    primary_source: Option<ChunkSourceId>,
    secondary_ship_sources: &std::collections::HashMap<u64, ChunkSourceId>,
    grace_ship_sources: &std::collections::HashMap<u64, ChunkSourceId>,
    departed_ship: Option<DepartedShipView>,
) -> RenderObjects {
    let mut object_count = 0usize;
    let mut inside_sphere_indices: Vec<usize> = Vec::new();

    {
        // Celestial bodies -- unified rendering for all shards.
        // Uses interpolated body positions (lerped with the same t as the camera)
        // so body offsets and camera share the same virtual time instant.
        // The secondary shard contributes only surface-detail data (ships, future:
        // terrain chunks) that the primary lacks.
        for body in interpolated_bodies {
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
        for body in interpolated_bodies {
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

    // Phase 1: collect ship block-mesh transforms. This populates
    // `ship_ids_with_meshes`, which the LOD-sphere pass below uses to skip
    // ships that already have streamed blocks. The legacy `ws.ships` field
    // is always `vec![]` in every shard — ships are emitted via `entities[]`
    // with `EntityKind::Ship`.
    //
    // Source resolution priority:
    //   - Own ship on SHIP shard: `primary_source`, Interior transform.
    //   - Any other ship entity (from ws / sec_ws / last_primary entities):
    //     `secondary_ship_sources[entity_id]`. Ships without a streamed source
    //     still render as LOD-proxy spheres in Phase 2.
    let mut ship_ids_with_meshes: std::collections::HashSet<u64> =
        std::collections::HashSet::new();
    let mut block_ships: Vec<BlockShipTransform> = Vec::new();

    use voxeldust_core::client_message::EntityKind;
    // Interior ship (player is inside — shard_type 2).
    // Resolve the ship's chunk source across the boarding seam: the client
    // may be in any of three states simultaneously:
    //   * primary_source is set (steady state after Connected) — use it.
    //   * primary_source is None but the ship is still reachable via its
    //     secondary observer source (post-Transitioning, pre-Connected gap).
    //   * same but the source was moved into grace after role change.
    // `shard_seed` flips one tick after `current_shard_type` (Connected vs
    // Transitioning), so during the boarding gap it still holds the PREV
    // primary's seed. Prefer the authoritative id carried on the own-ship
    // entity in `ws` / `secondary_ws` — that matches exactly what boarding
    // promoted. Fall back to `shard_seed` for the steady state.
    if current_shard_type == voxeldust_core::client_message::shard_type::SHIP {
        let own_ship_id: Option<u64> = ws
            .and_then(|w| w.entities.iter().find(|e| e.is_own
                && matches!(e.kind, EntityKind::Ship))
                .map(|e| e.entity_id))
            .or_else(|| secondary_ws
                .and_then(|w| w.entities.iter().find(|e| e.is_own
                    && matches!(e.kind, EntityKind::Ship))
                    .map(|e| e.entity_id)));
        let resolve_id = own_ship_id.unwrap_or(shard_seed);
        // Same preference as the external-ship path: grace-pinned before
        // secondary, because a freshly-created secondary source may be
        // waiting on chunk snapshots (renders empty ship = blink). The
        // grace source is the ship's previous primary/secondary that
        // was just role-switched — its chunks are already on the GPU.
        let source = primary_source
            .or_else(|| grace_ship_sources.get(&resolve_id).copied())
            .or_else(|| secondary_ship_sources.get(&resolve_id).copied());
        if let Some(source) = source {
            let ship_rot_mat = Mat4::from_quat(ship_rotation.as_quat());
            let origin_local = -(player_position + DVec3::new(0.0, EYE_HEIGHT, 0.0));
            let ship_origin_offset = (ship_rotation * origin_local).as_vec3();
            let base_transform = Mat4::from_translation(ship_origin_offset) * ship_rot_mat;
            block_ships.push(BlockShipTransform { base_transform, source });
            ship_ids_with_meshes.insert(resolve_id);
        }
    }

    // Render every ship as its actual block mesh — no LOD / proxy / sphere
    // substitution at any distance. If a ship's observer source isn't
    // connected yet or its chunks haven't arrived, we render nothing for
    // that ship this frame; the proactive pre-connect pipeline guarantees
    // the gap is sub-second and (per policy) happens before the ship is
    // close enough to be visually relevant.
    let mut push_ship_entity = |block_ships: &mut Vec<BlockShipTransform>,
                                 ship_ids_with_meshes: &mut std::collections::HashSet<u64>,
                                 e: &voxeldust_core::client_message::ObservableEntityData,
                                 origin: DVec3| {
        if !matches!(e.kind, EntityKind::Ship) { return; }
        if ship_ids_with_meshes.contains(&e.entity_id) { return; }
        // Skip own ship when interior branch already rendered it.
        if current_shard_type == voxeldust_core::client_message::shard_type::SHIP
            && e.entity_id == shard_seed
        {
            return;
        }
        // Source resolution: prefer grace-pinned (old primary / old
        // secondary whose chunks are already fully uploaded) over a
        // freshly-allocated secondary that may have zero chunks
        // populated yet. Observed bug: on SHIP→SYSTEM exit, the system
        // shard pre-connects the ship as a NEW secondary — a source
        // entry is created immediately but the chunk snapshots take
        // ~30-90 ms to arrive. During that window, falling back to
        // `secondary_ship_sources` picks a source with zero chunks and
        // the ship renders invisibly — the "blink." Checking grace
        // FIRST keeps the fully-populated old source active until the
        // grace window expires, by which time the new secondary has
        // finished downloading and both paths agree.
        let Some(source) = grace_ship_sources
            .get(&e.entity_id)
            .copied()
            .or_else(|| secondary_ship_sources.get(&e.entity_id).copied())
        else {
            return;
        };
        // System-space offset. `origin + e.position` is always the
        // entity's absolute system-space position (ship-shard shifts
        // entities by `-exterior.position`, planet-shard by
        // `-planet_pos`, so the sum undoes the shift). `cam.cam_system_pos`
        // is system-space on SHIP (camera.rs adds origin) but
        // shard-local on PLANET/SYSTEM — so we promote to system-space
        // here by adding origin when the current shard is not SHIP. On
        // SYSTEM the origin is 0 (no-op), on PLANET it's the planet's
        // system-space position (the missing piece that caused the
        // exit-to-planet blink: the ship rendered at `system_space -
        // planet_local` which was on the other side of the solar system
        // and always frustum-culled).
        let ship_system_pos = origin + e.position;
        let cam_system = if current_shard_type == voxeldust_core::client_message::shard_type::SHIP {
            cam.cam_system_pos
        } else {
            // Use the PRIMARY WS's origin for cam frame. When ws is
            // None (first frame after transition before any WS arrived)
            // fall back to the passed `origin` which at least matches
            // the entity we're computing for.
            ws.map(|w| w.origin).unwrap_or(origin) + cam.cam_system_pos
        };
        let offset = (ship_system_pos - cam_system).as_vec3();
        if !cam.frustum.contains_sphere(offset, e.bounding_radius.max(16.0) as f32) { return; }
        let ship_rot_mat = Mat4::from_quat(e.rotation.as_quat());
        let base_transform = Mat4::from_translation(offset) * ship_rot_mat;
        block_ships.push(BlockShipTransform { base_transform, source });
        ship_ids_with_meshes.insert(e.entity_id);
    };
    if let Some(ws) = ws {
        for e in &ws.entities {
            push_ship_entity(&mut block_ships, &mut ship_ids_with_meshes, e, ws.origin);
        }
    }
    if let Some(sec_ws) = secondary_ws {
        for e in &sec_ws.entities {
            push_ship_entity(&mut block_ships, &mut ship_ids_with_meshes, e, sec_ws.origin);
        }
    }
    // Grace-window fallback: during a seamless promotion, the new primary
    // marks its own ship `is_own` (skipped below) but we still want the
    // block mesh on screen; the stashed `last_primary` carries the ship as
    // a normal Ship entity at its system-space position.
    if let Some(grace_ws) = grace_fallback_ws {
        for e in &grace_ws.entities {
            push_ship_entity(&mut block_ships, &mut ship_ids_with_meshes, e, grace_ws.origin);
        }
    }

    // Departed ship: rendered in ABSOLUTE SYSTEM-SPACE so the ship stays
    // on screen during the exit grace window regardless of what shard
    // frame the new primary's camera is in. The entity-based paths above
    // all use `origin + e.position` which resolves to system-space only
    // when origin is in system-space AND cam_system_pos is in system-space
    // — that invariant breaks on planet-shard-primary where
    // `cam_system_pos` is planet-local. This path sidesteps the frame
    // mismatch by receiving the ship's already-system-space pose and
    // the caller-resolved system-space camera position.
    if let Some(ref dep) = departed_ship {
        let offset = (dep.world_position - dep.cam_world).as_vec3();
        let dist = offset.length();
        let in_frustum = cam.frustum.contains_sphere(offset, dep.bounding_radius.max(16.0));
        if in_frustum {
            let ship_rot_mat = Mat4::from_quat(dep.rotation.as_quat());
            let base_transform = Mat4::from_translation(offset) * ship_rot_mat;
            block_ships.push(BlockShipTransform {
                base_transform,
                source: dep.source,
            });
        }
        // Diagnostic: one line per frame during the grace window
        // (~90 lines for the 1.5 s window — acceptable until verified).
        tracing::info!(
            source = dep.source.0,
            offset = ?(offset.x, offset.y, offset.z),
            dist,
            bounding_radius = dep.bounding_radius,
            in_frustum,
            "departed-ship render"
        );
    }

    // Phase 2: LOD-sphere / avatar entity rendering. Runs AFTER block_ships
    // so `ship_ids_with_meshes` reflects which ships are already rendered as
    // full meshes and therefore shouldn't also get an orange proxy sphere.
    //
    // Skip is_own entries (own avatar — camera is mounted there).
    // `drawn_ids` tracks entity_ids already rendered so the grace-fallback
    // pass doesn't double-draw entities that also appear in the live WS.
    let mut drawn_ids: std::collections::HashSet<u64> = std::collections::HashSet::new();
    let draw_entities =
        |uniform_data: &mut [ObjectUniforms], object_count: &mut usize,
         drawn: &mut std::collections::HashSet<u64>,
         entities: &[voxeldust_core::client_message::ObservableEntityData],
         origin: DVec3,
         ships_with_meshes: &std::collections::HashSet<u64>| {
            for e in entities {
                if e.is_own {
                    continue;
                }
                if !drawn.insert(e.entity_id) {
                    continue;
                }
                let (color, radius) = match e.kind {
                    // Ships are ALWAYS rendered as their actual block mesh
                    // via `push_ship_entity` — never as a proxy sphere. If
                    // the mesh isn't on the GPU yet, render nothing this
                    // frame (streaming catches up within sub-second, and
                    // proactive pre-connect means this only happens at very
                    // distant approach where invisibility is acceptable).
                    EntityKind::Ship => continue,
                    EntityKind::EvaPlayer => ([1.0, 0.25, 0.85, 1.0], 1.0),
                    EntityKind::GroundedPlayer => ([0.25, 0.65, 1.0, 1.0], 1.0),
                    EntityKind::Seated => ([0.25, 1.0, 0.75, 1.0], 1.0),
                };
                let _ = ships_with_meshes; // kept for API stability; no longer consulted
                if *object_count >= MAX_OBJECTS {
                    break;
                }
                let world_pos = origin + e.position;
                let offset = (world_pos - cam.cam_system_pos).as_vec3();
                if !cam.frustum.contains_sphere(offset, radius) {
                    continue;
                }
                let model = Mat4::from_translation(offset) * Mat4::from_scale(Vec3::splat(radius));
                let mvp = cam.vp * model;
                let mut obj: ObjectUniforms = bytemuck::Zeroable::zeroed();
                obj.mvp = mvp.to_cols_array_2d();
                obj.model = model.to_cols_array_2d();
                obj.color = color;
                // Matte material (roughness 0.7, metallic 0) for now.
                obj.material = [0.0, 0.7, 0.0, 0.0];
                uniform_data[*object_count] = obj;
                *object_count += 1;
            }
        };
    if let Some(ws) = ws {
        draw_entities(
            uniform_data,
            &mut object_count,
            &mut drawn_ids,
            &ws.entities,
            ws.origin,
            &ship_ids_with_meshes,
        );
    }
    if let Some(sec_ws) = secondary_ws {
        draw_entities(
            uniform_data,
            &mut object_count,
            &mut drawn_ids,
            &sec_ws.entities,
            sec_ws.origin,
            &ship_ids_with_meshes,
        );
    }
    // Grace-window fallback: during a seamless primary-secondary promotion,
    // the old primary's WS is stashed briefly so its LOD-proxy entities (e.g.,
    // the ship the user just touched, which the new primary marks `is_own`
    // and therefore skips) keep drawing for `LAST_PRIMARY_GRACE_SECS`. Dedupe
    // against entities already drawn from `ws`/`sec_ws` by entity_id.
    if let Some(grace_ws) = grace_fallback_ws {
        draw_entities(
            uniform_data,
            &mut object_count,
            &mut drawn_ids,
            &grace_ws.entities,
            grace_ws.origin,
            &ship_ids_with_meshes,
        );
    }

    // Block chunk uniforms begin right after the sphere/avatar draws. The
    // shadow + color passes fill `uniform_data[block_uniform_start..]` with
    // one entry per (block_ships transform × chunks-for-source).
    let block_uniform_start = object_count;

    RenderObjects {
        object_count,
        block_uniform_start,
        block_ships,
        inside_sphere_indices,
    }
}

/// Find the planet whose atmosphere the camera is closest to.
/// Returns (planet_index, planet_body_position) or None if no atmosphere nearby.
fn find_atmosphere_planet(
    cam_system_pos: DVec3,
    system_params: &SystemParams,
    bodies: &[voxeldust_core::client_message::CelestialBodyData],
) -> Option<(usize, DVec3)> {
    let mut best: Option<(usize, f64, DVec3)> = None;

    for body in bodies {
        if body.body_id == 0 { continue; } // skip star
        let planet_idx = (body.body_id - 1) as usize;
        if planet_idx >= system_params.planets.len() { continue; }

        let planet = &system_params.planets[planet_idx];
        if !planet.atmosphere.has_atmosphere { continue; }

        let dist = (cam_system_pos - body.position).length();
        let atmo_top = planet.radius_m + planet.atmosphere.atmosphere_height;

        // Inside atmosphere sphere: use this planet immediately.
        if dist < atmo_top {
            return Some((planet_idx, body.position));
        }

        // Outside but close enough to render limb glow (within 3× atmosphere radius).
        if dist < atmo_top * 3.0 {
            match &best {
                None => best = Some((planet_idx, dist, body.position)),
                Some((_, best_dist, _)) if dist < *best_dist => {
                    best = Some((planet_idx, dist, body.position));
                }
                _ => {}
            }
        }
    }

    best.map(|(idx, _, pos)| (idx, pos))
}

/// Build AtmosphereUniforms from planet parameters and camera state.
/// All scattering coefficients come from the planet's seed-derived AtmosphereParams.
/// Units are converted to kilometers (matching Hillaire 2020 convention).
fn build_atmosphere_uniforms(
    planet: &PlanetParams,
    planet_pos: DVec3,
    cam: &CameraParams,
    scene_lighting: &SceneLighting,
    gfx: &GraphicsSettings,
) -> AtmosphereUniforms {
    let atmo = &planet.atmosphere;

    // Observer position relative to planet center, in km.
    let observer_m = cam.cam_system_pos - planet_pos;
    let observer_km = observer_m * (1.0 / 1000.0);

    let planet_r_km = (planet.radius_m / 1000.0) as f32;
    let atmo_r_km = ((planet.radius_m + atmo.atmosphere_height) / 1000.0) as f32;

    // Scale heights: convert from meters to km, then to exp scale (-1/H_km).
    let rayleigh_h_km = (atmo.rayleigh_scale_height / 1000.0) as f32;
    let mie_h_km = (atmo.mie_scale_height / 1000.0) as f32;

    // Scattering coefficients: convert from 1/m to 1/km (multiply by 1000).
    let r_coeff = atmo.rayleigh_coeff;
    let rayleigh_scatter_km = [
        (r_coeff[0] * 1000.0) as f32,
        (r_coeff[1] * 1000.0) as f32,
        (r_coeff[2] * 1000.0) as f32,
    ];
    let mie_scatter_km = (atmo.mie_coeff * 1000.0) as f32;
    let mie_absorb_km = (atmo.mie_absorption * 1000.0) as f32;

    let ozone_km = [
        (atmo.ozone_coeff[0] * 1000.0) as f32,
        (atmo.ozone_coeff[1] * 1000.0) as f32,
        (atmo.ozone_coeff[2] * 1000.0) as f32,
    ];
    let ozone_center_km = (atmo.ozone_center_altitude / 1000.0) as f32;
    let ozone_width_km = (atmo.ozone_width / 1000.0) as f32;

    // Sun direction and color from scene lighting.
    let sun_dir = [scene_lighting.sun_direction[0], scene_lighting.sun_direction[1], scene_lighting.sun_direction[2]];
    let sun_intensity = scene_lighting.sun_color[3]; // intensity in alpha
    let sun_color = [
        scene_lighting.sun_color[0] * sun_intensity,
        scene_lighting.sun_color[1] * sun_intensity,
        scene_lighting.sun_color[2] * sun_intensity,
    ];

    // Inverse VP for depth reconstruction.
    let inv_vp = (cam.vp).inverse();

    // Atmosphere-space basis: local up = planet radial. We need a world-fixed
    // azimuth reference that stays stable under tiny per-frame changes in
    // `observer_km` — otherwise the sky-view LUT reparameterizes slightly
    // each frame and the sky visibly shakes.
    //
    // The sun direction is a natural world-fixed reference: it barely changes
    // frame-to-frame (server-driven sun) and is always meaningful. Project it
    // onto the tangent plane at the observer to get a stable azimuth axis.
    // Only degenerate case: sun is exactly along `up`; pick an arbitrary
    // world-fixed fallback and ensure no discontinuity as we transition.
    let observer_len = observer_km.length();
    let up = if observer_len > 1e-6 {
        (observer_km / observer_len).as_vec3()
    } else {
        Vec3::Y
    };
    let sun_world = Vec3::new(sun_dir[0], sun_dir[1], sun_dir[2]);
    // Project sun onto tangent plane: tangent_sun = sun - (sun·up)·up.
    let tangent_sun_raw = sun_world - up * sun_world.dot(up);
    let tangent_sun_len = tangent_sun_raw.length();
    let right = if tangent_sun_len > 1e-4 {
        // Right axis perpendicular to both up and tangent_sun (cross product).
        // Stable because tangent_sun is world-fixed (or nearly so).
        up.cross(tangent_sun_raw).normalize_or(Vec3::X)
    } else {
        // Sun is colinear with up (zenith/nadir). Fall back to a world-fixed
        // axis that's perpendicular to up, choosing deterministically.
        let fallback = if up.x.abs() < 0.9 { Vec3::X } else { Vec3::Z };
        up.cross(fallback).normalize_or(Vec3::X)
    };
    let fwd = right.cross(up).normalize_or(Vec3::NEG_Z);
    let world_from_atmo_rot = Mat4::from_cols(
        right.extend(0.0),
        up.extend(0.0),
        (-fwd).extend(0.0),
        Vec4::new(0.0, 0.0, 0.0, 1.0),
    );
    let atmo_from_world_rot = world_from_atmo_rot.transpose();

    // Aerial-perspective LUT spans the visible distance range inside the
    // atmosphere. Use atmosphere height × 1.5 as a sensible cap — covers the
    // full horizon sweep without wasting LUT slices on far-past-atmosphere.
    let ap_max_km = ((atmo.atmosphere_height / 1000.0) as f32) * 1.5;

    // Sun angular radius (half-angle). If not set by the server, pick a
    // sensible default matching a Sol-like 0.25° disc.
    let sun_angular_radius = 0.00465_f32;

    AtmosphereUniforms {
        observer_pos: [observer_km.x as f32, observer_km.y as f32, observer_km.z as f32, 0.0],
        radii: [planet_r_km, atmo_r_km, 1.0, sun_angular_radius],
        rayleigh: [rayleigh_scatter_km[0], rayleigh_scatter_km[1], rayleigh_scatter_km[2],
                   -1.0 / rayleigh_h_km],
        mie: [mie_scatter_km, mie_absorb_km, -1.0 / mie_h_km, atmo.mie_anisotropy as f32],
        ozone: [ozone_km[0], ozone_km[1], ozone_km[2], ozone_center_km],
        ozone_extra: [ozone_width_km, 0.3, 1.0, 0.0], // ground_albedo, multi_scatter_factor
        sun_dir: [sun_dir[0], sun_dir[1], sun_dir[2], 0.0],
        sun_color: [sun_color[0], sun_color[1], sun_color[2], 0.0],
        inv_vp: inv_vp.to_cols_array_2d(),
        screen: [cam.aspect * 100.0, 100.0, 0.0, 0.0],
        quality: [
            atmo.weather_mie_multiplier as f32,
            atmo.weather_sun_occlusion as f32,
            ap_max_km,
            gfx.atmosphere_samples as f32,
        ],
        atmo_from_world: atmo_from_world_rot.to_cols_array_2d(),
        world_from_atmo: world_from_atmo_rot.to_cols_array_2d(),
    }
}

/// Build a disabled AtmosphereUniforms (no atmosphere rendering).
fn disabled_atmosphere_uniforms() -> AtmosphereUniforms {
    bytemuck::Zeroable::zeroed()
}

/// Compute eclipse darkening factor. Returns 1.0 (fully lit) to 0.0 (total eclipse).
/// Checks if any planet is between the camera and the star, using angular radii
/// for physically correct umbra/penumbra.
fn compute_eclipse(
    cam_system_pos: DVec3,
    bodies: &[voxeldust_core::client_message::CelestialBodyData],
    star_radius: f64,
) -> f32 {
    // Star is always body_id 0.
    let star_pos = bodies.iter()
        .find(|b| b.body_id == 0)
        .map(|b| b.position)
        .unwrap_or(DVec3::ZERO);

    let to_star = star_pos - cam_system_pos;
    let star_dist = to_star.length();
    if star_dist < 1.0 { return 1.0; }
    let star_dir = to_star / star_dist;

    let star_angular = (star_radius / star_dist).atan();

    let mut best_factor = 1.0_f32;

    for body in bodies {
        if body.body_id == 0 { continue; } // skip star

        let to_body = body.position - cam_system_pos;
        let body_dist = to_body.length();
        if body_dist < 1.0 { continue; }

        // Body must be between camera and star (projection along star direction > 0).
        let proj = to_body.dot(star_dir);
        if proj < 0.0 || proj > star_dist { continue; }

        // Closest approach of the camera→star ray to the body center.
        let perp = (to_body - star_dir * proj).length();

        // Angular radii from camera.
        let body_angular = (body.radius / body_dist).atan();
        let separation = (perp / body_dist).atan();

        if separation < (body_angular - star_angular).max(0.0) {
            // Total eclipse: body fully covers star disk.
            best_factor = 0.0;
        } else if separation < body_angular + star_angular {
            // Partial eclipse: penumbra region.
            let penumbra_width = 2.0 * star_angular;
            let t = ((separation - (body_angular - star_angular).max(0.0)) / penumbra_width)
                .clamp(0.0, 1.0) as f32;
            best_factor = best_factor.min(t);
        }
    }

    best_factor
}

/// Lightning flash envelope. Deterministic from (game_time, planet_seed) so
/// every player on the planet sees the identical flash pattern. Returns the
/// flash intensity in [0, 1], typically > 0 only for a few frames at a time.
///
/// Model: 0.5-second time buckets; each bucket independently rolls a trigger
/// event with probability scaled by `storm_strength` (coverage×type×precip at
/// the player's location) and `altitude_factor` (how far into the storm the
/// player is). When triggered, a 160-ms double-pulse envelope replays —
/// mimicking the characteristic "bright, momentary darkness, brighter" pattern
/// of real cloud-to-cloud lightning seen from below the storm.
fn lightning_flash_envelope(
    game_time: f64,
    planet_seed: u64,
    storm_strength: f32,
    altitude_factor: f32,
) -> f32 {
    if storm_strength <= 0.0 || altitude_factor <= 0.0 { return 0.0; }
    const BUCKET_SECS: f64 = 0.5;
    const FLASH_LEN: f64 = 0.16;
    let current_bucket = (game_time / BUCKET_SECS).floor() as i64;
    let trigger_prob = (storm_strength * altitude_factor * 0.35).clamp(0.0, 0.45);

    // Check up to 2 recent buckets in case an active flash started before the
    // current bucket (edges of 160 ms envelope can cross bucket boundaries).
    let mut max_flash = 0.0_f32;
    for back in 0..2 {
        let bucket = current_bucket.saturating_sub(back);
        let bucket_start_time = bucket as f64 * BUCKET_SECS;
        // 64-bit splitmix hash: bucket_index XOR planet_seed.
        let mixed = (bucket as u64).wrapping_mul(0x9E3779B97F4A7C15).wrapping_add(planet_seed);
        let roll = (mixed.wrapping_mul(0xBF58476D1CE4E5B9)) ^ (mixed >> 27);
        let r01 = (roll as f64) / (u64::MAX as f64);
        if r01 < trigger_prob as f64 {
            let t_since = game_time - bucket_start_time;
            if t_since >= 0.0 && t_since < FLASH_LEN {
                let x = (t_since / FLASH_LEN) as f32;
                // Double-pulse: quick bright peak, short dimming, smaller rebound.
                let env = if x < 0.15 {
                    1.0 - x / 0.15
                } else if x < 0.4 {
                    0.0
                } else if x < 0.7 {
                    (0.5 * (1.0 - (x - 0.4) / 0.3)).max(0.0)
                } else {
                    0.0
                };
                if env > max_flash { max_flash = env; }
            }
        }
    }
    max_flash
}

/// Build SceneLighting from the current WorldState.
fn build_scene_lighting(ws: Option<&WorldStateData>) -> SceneLighting {
    if let Some(ws) = ws {
        if let Some(ref l) = ws.lighting {
            let dir = l.sun_direction.normalize().as_vec3();
            SceneLighting {
                sun_direction: [dir.x, dir.y, dir.z, 0.0],
                sun_color: [l.sun_color[0], l.sun_color[1], l.sun_color[2], l.sun_intensity],
                ambient: [l.ambient, 0.0, 0.0, 0.0],
                camera_pos: [0.0, 0.0, 0.0, 0.0],
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
    SceneLighting {
        sun_direction: [0.0, -1.0, 0.0, 0.0],
        sun_color: [0.4, 0.4, 0.5, 0.15],
        ambient: [0.06, 0.0, 0.0, 0.0],
        camera_pos: [0.0, 0.0, 0.0, 0.0],
    }
}

/// Execute the full render frame: build uniforms, 3D pass, egui HUD, present.
pub fn render_frame(
    gpu: &mut GpuState,
    window: &winit::window::Window,
    uniform_data: &mut [ObjectUniforms],
    chunk_uniform_map: &mut std::collections::HashMap<voxeldust_core::block::client_chunks::ChunkKey, usize>,
    cam: &CameraParams,
    latest_world_state: Option<&WorldStateData>,
    secondary_world_state: Option<&WorldStateData>,
    grace_fallback_world_state: Option<&WorldStateData>,
    interpolated_bodies: &[voxeldust_core::client_message::CelestialBodyData],
    current_shard_type: u8,
    shard_seed: u64,
    player_position: DVec3,
    player_velocity: DVec3,
    ship_rotation: DQuat,
    is_piloting: bool,
    connected: bool,
    selected_thrust_tier: u8,
    engines_off: bool,
    cruise_active: bool,
    atmo_comp_active: bool,
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
    gfx_settings: &GraphicsSettings,
    sub_grid_transforms: &std::collections::HashMap<u32, voxeldust_core::client_message::SubGridTransformData>,
    sub_grid_assignments: &std::collections::HashMap<glam::IVec3, u32>,
    primary_source: Option<ChunkSourceId>,
    secondary_ship_sources: &std::collections::HashMap<u64, ChunkSourceId>,
    grace_ship_sources: &std::collections::HashMap<u64, ChunkSourceId>,
    departed_ship: Option<DepartedShipView>,
    cloud_system: Option<&crate::cloud_system::CloudSystem>,
) -> hud::ConfigPanelAction {
    let frame = match gpu.surface.get_current_texture() { Ok(f) => f, Err(_) => return hud::ConfigPanelAction::None };
    let view = frame.texture.create_view(&wgpu::TextureViewDescriptor::default());

    // Build object uniforms.
    let ro = build_uniforms(
        uniform_data, cam, latest_world_state, secondary_world_state, grace_fallback_world_state,
        interpolated_bodies, current_shard_type, shard_seed, player_position, ship_rotation,
        primary_source, secondary_ship_sources, grace_ship_sources, departed_ship,
    );

    // Diagnostic: log ship rendering state every 40 frames (~0.7s at 60fps)
    // so we can correlate "ship disappeared" reports with what the render
    // pipeline saw. Emits: block_ships count, per-ship source+chunk count,
    // number of Ship entities in each WS, and the full secondary source map.
    let trace_active = {
        use std::sync::atomic::Ordering;
        let now_ms = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_millis() as u64)
            .unwrap_or(0);
        let until = TRACE_RENDER_UNTIL_MS.load(Ordering::Relaxed);
        now_ms < until
    };
    if trace_active || frame_count % 40 == 0 {
        let ws_ship_entities = latest_world_state.map_or(0, |ws| {
            ws.entities.iter().filter(|e| matches!(e.kind, voxeldust_core::client_message::EntityKind::Ship)).count()
        });
        let sec_ws_ship_entities = secondary_world_state.map_or(0, |ws| {
            ws.entities.iter().filter(|e| matches!(e.kind, voxeldust_core::client_message::EntityKind::Ship)).count()
        });
        let grace_ws_ship_entities = grace_fallback_world_state.map_or(0, |ws| {
            ws.entities.iter().filter(|e| matches!(e.kind, voxeldust_core::client_message::EntityKind::Ship)).count()
        });
        let block_ship_sources: Vec<u32> = ro.block_ships
            .iter()
            .map(|bs| bs.source.0)
            .collect();
        let block_ship_chunk_counts: Vec<(u32, usize)> = block_renderer
            .map(|br| {
                ro.block_ships.iter()
                    .map(|bs| (bs.source.0, br.chunks_for_source(bs.source).count()))
                    .collect()
            })
            .unwrap_or_default();
        let grace_src_map: Vec<(u64, u32)> = grace_ship_sources.iter().map(|(k, v)| (*k, v.0)).collect();
        let ws_origin = latest_world_state.map(|ws| (ws.origin.x as i64, ws.origin.y as i64, ws.origin.z as i64));
        let grace_origin = grace_fallback_world_state.map(|gws| (gws.origin.x as i64, gws.origin.y as i64, gws.origin.z as i64));
        let cam_sys = (cam.cam_system_pos.x as i64, cam.cam_system_pos.y as i64, cam.cam_system_pos.z as i64);
        let primary_ship_entities: Vec<(u64, f64, bool)> = latest_world_state
            .map(|ws| ws.entities.iter()
                .filter(|e| matches!(e.kind, voxeldust_core::client_message::EntityKind::Ship))
                .map(|e| (e.entity_id, e.bounding_radius as f64, e.is_own))
                .collect())
            .unwrap_or_default();
        tracing::info!(
            frame = frame_count,
            shard_type = current_shard_type,
            shard_seed,
            ws_ships = ws_ship_entities,
            sec_ws_ships = sec_ws_ship_entities,
            grace_ws_ships = grace_ws_ship_entities,
            block_ships = ro.block_ships.len(),
            block_ship_sources = ?block_ship_sources,
            block_ship_chunks = ?block_ship_chunk_counts,
            primary_source = primary_source.map(|s| s.0),
            sec_src_map = ?secondary_ship_sources.iter().map(|(k, v)| (*k, v.0)).collect::<Vec<_>>(),
            grace_src_map = ?grace_src_map,
            ws_origin = ?ws_origin,
            grace_origin = ?grace_origin,
            cam_sys = ?cam_sys,
            ws_ships_info = ?primary_ship_entities,
            "ship-render diagnostic"
        );
    }

    // Write scene lighting data from server WorldState, with eclipse darkening.
    let mut scene_lighting = build_scene_lighting(latest_world_state);
    if gfx_settings.eclipse_shadows_enabled {
        if let Some(sys) = system_params {
            if !interpolated_bodies.is_empty() {
                let eclipse = compute_eclipse(cam.cam_system_pos, interpolated_bodies, sys.star.radius_m);
                scene_lighting.sun_color[3] *= eclipse; // modulate sun intensity
            }
        }
    }
    // Cloud shadow attenuation + lightning flashes + wetness. All only apply
    // when the player is actually outside — piloting a cockpit means the camera
    // is sealed inside an interior and storm modulation shouldn't leak into
    // the shared SceneLighting buffer (otherwise interior walls become "wet"
    // and flash with lightning, which looks awful).
    if !is_piloting {
        scene_lighting.ambient[1] = 0.0;
    }
    if !is_piloting {
    if let (Some(cs), Some(sys)) = (cloud_system, system_params) {
        if let Some(weather) = &cs.current_weather_cpu {
            if let Some((planet_idx, planet_pos)) = find_atmosphere_planet(cam.cam_system_pos, sys, interpolated_bodies) {
                let planet = &sys.planets[planet_idx];
                let observer_m = cam.cam_system_pos - planet_pos;
                let altitude_m = observer_m.length() - planet.radius_m;
                let cloud_top_m = planet.clouds.cloud_base_altitude + planet.clouds.cloud_layer_thickness;
                if altitude_m < cloud_top_m && planet.clouds.has_clouds {
                    let sample = weather.sample(observer_m);
                    let coverage = sample[0];
                    let cloud_type = sample[1];
                    let precip = sample[2];
                    let alt_factor = if altitude_m < planet.clouds.cloud_base_altitude {
                        1.0
                    } else {
                        let t = (altitude_m - planet.clouds.cloud_base_altitude) / planet.clouds.cloud_layer_thickness;
                        (1.0 - t).clamp(0.0, 1.0) as f32
                    };
                    let max_darken = 0.75 * cloud_type.max(0.4);
                    let factor = 1.0 - coverage * alt_factor * max_darken;
                    scene_lighting.sun_color[3] *= factor;

                    // Lightning: cumulonimbus + precipitation + coverage produces
                    // electrically active storm cells. Deterministic hashing of
                    // 0.5-second time buckets fires a short double-pulse flash
                    // with ~15% probability in any given bucket. Effect is
                    // additive to ambient so shadow side still glows.
                    // Wet-surface factor: surfaces stay wet during rain and
                    // for a short decay after. For initial implementation we
                    // use the instantaneous precipitation at the player's
                    // location; a proper puddle-decay buffer would remember
                    // "where did it recently rain" across chunks, a future
                    // addition. Clamp to the same altitude fade so the player
                    // only gets wet when they're actually below/in the cloud
                    // layer.
                    let wetness = (precip * alt_factor).clamp(0.0, 1.0);
                    scene_lighting.ambient[1] = wetness;

                    let storm_strength = coverage * cloud_type.max(0.0) * precip.max(0.0);
                    if storm_strength > 0.08 {
                        if let Some(ws) = latest_world_state {
                            let game_time = ws.game_time;
                            let flash = lightning_flash_envelope(game_time, planet.planet_seed, storm_strength, alt_factor);
                            if flash > 0.0 {
                                // Cool-white flash color, added to ambient term
                                // (the existing ambient channel lives in
                                // scene_lighting.ambient[0] as a scalar; bump
                                // it directly). Also brighten sun slightly to
                                // simulate the secondary illumination from
                                // deep-cloud scattering.
                                let f = flash.min(1.0);
                                scene_lighting.ambient[0] += f * 0.3;
                                scene_lighting.sun_color[3] *= 1.0 + f * 0.25;
                            }
                        }
                    }
                }
            }
        }
    }
    } // end if !is_piloting
    gpu.queue.write_buffer(&gpu.scene_lighting_buf, 0, bytemuck::bytes_of(&scene_lighting));

    // Write render configuration from graphics settings.
    let render_config = RenderConfig::from_settings(gfx_settings, frame_count);
    gpu.queue.write_buffer(&gpu.render_config_buf, 0, bytemuck::bytes_of(&render_config));

    // Upload camera-space uniforms.
    if ro.object_count > 0 {
        gpu.queue.write_buffer(&gpu.uniform_buf, 0, bytemuck::cast_slice(&uniform_data[..ro.object_count]));
    }

    let mut encoder = gpu.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });

    // -- CSM Shadow pass: render depth into 4 cascade layers --
    let sun_dir = Vec3::new(scene_lighting.sun_direction[0], scene_lighting.sun_direction[1], scene_lighting.sun_direction[2]);
    if sun_dir.length_squared() > 0.001 {
        let cascade_matrices = compute_cascade_matrices(sun_dir.normalize(), cam);


        // Upload cascade uniforms.
        let cascades = ShadowCascades {
            light_vp: [
                cascade_matrices[0].to_cols_array_2d(),
                cascade_matrices[1].to_cols_array_2d(),
                cascade_matrices[2].to_cols_array_2d(),
                cascade_matrices[3].to_cols_array_2d(),
            ],
            splits: CASCADE_SPLITS,
            _pad: [0.0; 4],
        };
        if let Some(buf) = &gpu.shadow_cascade_buf {
            gpu.queue.write_buffer(buf, 0, bytemuck::bytes_of(&cascades));
        }

        // Render depth for each cascade with per-cascade command buffer submission.
        // Each cascade reuses the same uniform buffer offsets — the per-cascade submit
        // ensures the GPU consumes the data before the next cascade overwrites it.
        if let Some(shadow_pipeline) = &gpu.shadow_pipeline {
            for cascade_idx in 0..NUM_CASCADES as usize {
                // Write shadow-space MVPs for spheres.
                for i in 0..ro.object_count {
                    let model = Mat4::from_cols_array_2d(&uniform_data[i].model);
                    uniform_data[i].mvp = (cascade_matrices[cascade_idx] * model).to_cols_array_2d();
                }
                if ro.object_count > 0 {
                    gpu.queue.write_buffer(&gpu.uniform_buf, 0, bytemuck::cast_slice(&uniform_data[..ro.object_count]));
                }

                // Write shadow-space MVPs for block chunks. Each ship in
                // `block_ships` pairs its transform with its own chunk source,
                // so we iterate (transform × that source's chunks) — not the
                // old (transform × all active sources) cross product.
                if let Some(br) = block_renderer {
                    if br.has_chunks() && !ro.block_ships.is_empty() {
                        let block_start = ro.block_uniform_start;
                        let mut chunk_idx = 0usize;
                        for ship_transform in &ro.block_ships {
                            for (chunk_pos, _) in br.chunks_for_source(ship_transform.source) {
                                let uniform_idx = block_start + chunk_idx;
                                if uniform_idx >= MAX_OBJECTS { break; }
                                let chunk_model = crate::block_render::BlockRenderer::chunk_model_matrix(
                                    ship_transform.base_transform, chunk_pos, CHUNK_SIZE as f64,
                                );
                                uniform_data[uniform_idx].mvp = (cascade_matrices[cascade_idx] * chunk_model).to_cols_array_2d();
                                uniform_data[uniform_idx].model = chunk_model.to_cols_array_2d();
                                chunk_idx += 1;
                            }
                        }
                        if chunk_idx > 0 {
                            gpu.queue.write_buffer(
                                &gpu.uniform_buf,
                                (block_start as u64) * 256,
                                bytemuck::cast_slice(&uniform_data[block_start..block_start + chunk_idx]),
                            );
                        }
                    }
                }

                // Submit this cascade's render pass.
                let mut shadow_encoder = gpu.device.create_command_encoder(
                    &wgpu::CommandEncoderDescriptor { label: Some("shadow_cascade") },
                );
                {
                    let mut pass = shadow_encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                        label: Some("shadow"),
                        color_attachments: &[],
                        depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                            view: &gpu.shadow_cascade_views[cascade_idx],
                            depth_ops: Some(wgpu::Operations {
                                load: wgpu::LoadOp::Clear(1.0),
                                store: wgpu::StoreOp::Store,
                            }),
                            stencil_ops: None,
                        }),
                        ..Default::default()
                    });

                    pass.set_pipeline(shadow_pipeline);
                    pass.set_vertex_buffer(0, gpu.sphere_vertex_buf.slice(..));
                    pass.set_index_buffer(gpu.sphere_index_buf.slice(..), wgpu::IndexFormat::Uint32);
                    for i in 0..ro.object_count {
                        if uniform_data[i].color[3] > 0.5 { continue; }
                        pass.set_bind_group(0, &gpu.bind_group, &[(i as u32) * 256]);
                        pass.draw_indexed(0..gpu.sphere_index_count, 0, 0..1);
                    }

                    if let Some(br) = block_renderer {
                        if br.has_chunks() && !ro.block_ships.is_empty() {
                            pass.set_pipeline(&br.shadow_pipeline);
                            let block_start = ro.block_uniform_start;
                            let mut chunk_idx = 0usize;
                            for ship_transform in &ro.block_ships {
                                for (_, chunk_mesh) in br.chunks_for_source(ship_transform.source) {
                                    let uniform_idx = block_start + chunk_idx;
                                    if uniform_idx >= MAX_OBJECTS { break; }
                                    pass.set_bind_group(0, &gpu.bind_group, &[(uniform_idx as u32) * 256]);
                                    pass.set_vertex_buffer(0, chunk_mesh.vertex_buf.slice(..));
                                    pass.set_index_buffer(chunk_mesh.index_buf.slice(..), wgpu::IndexFormat::Uint32);
                                    pass.draw_indexed(0..chunk_mesh.index_count, 0, 0..1);
                                    chunk_idx += 1;
                                }
                            }
                        }
                    }
                }
                gpu.queue.submit(std::iter::once(shadow_encoder.finish()));
            }
        }

        // Restore camera-space MVPs for the main pass.
        let ro = build_uniforms(
            uniform_data, cam, latest_world_state, secondary_world_state, grace_fallback_world_state,
            interpolated_bodies, current_shard_type, shard_seed, player_position, ship_rotation,
            primary_source, secondary_ship_sources, grace_ship_sources, departed_ship,
        );
        if ro.object_count > 0 {
            gpu.queue.write_buffer(&gpu.uniform_buf, 0, bytemuck::cast_slice(&uniform_data[..ro.object_count]));
        }
    }

    // 3D passes render to the HDR texture when enabled, swapchain otherwise.
    let color_target = if gpu.hdr_enabled {
        gpu.hdr_view.as_ref().unwrap()
    } else {
        &view
    };

    // -- Star pass (before main pass, additive blend, no depth write) --
    {
        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("stars"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: color_target, resolve_target: None,
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
    {
        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("main"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: color_target, resolve_target: None,
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
        // Set voxel volume (group 2) -- constant for entire pass.
        if let Some(ref voxel_bg) = gpu.voxel_bind_group {
            pass.set_bind_group(2, voxel_bg, &[]);
        }

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
                let mut block_draws: Vec<(usize, &ChunkGpuMesh)> = Vec::with_capacity(128);
                let mut chunk_idx = 0usize;

                // For each ship, draw all cached chunks with that ship's transform.
                // Build a map of (source, chunk_pos) → uniform_idx for sub-block draw reuse.
                // Reuses caller-owned HashMap to avoid per-frame allocation.
                chunk_uniform_map.clear();

                for ship_transform in &ro.block_ships {
                    for (chunk_pos, chunk_mesh) in br.chunks_for_source(ship_transform.source) {
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

                        let key = voxeldust_core::block::client_chunks::ChunkKey {
                            source: ship_transform.source,
                            chunk: chunk_pos,
                        };
                        chunk_uniform_map.insert(key, uniform_idx);

                        block_draws.push((uniform_idx, chunk_mesh));
                        chunk_idx += 1;
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
                    if let Some(ref vbg) = gpu.voxel_bind_group { pass.set_bind_group(2, vbg, &[]); }

                    for &(uniform_idx, chunk_mesh) in &block_draws {
                        pass.set_bind_group(0, &gpu.bind_group, &[(uniform_idx as u32) * 256]);
                        pass.set_vertex_buffer(0, chunk_mesh.vertex_buf.slice(..));
                        pass.set_index_buffer(chunk_mesh.index_buf.slice(..), wgpu::IndexFormat::Uint32);
                        pass.draw_indexed(0..chunk_mesh.index_count, 0, 0..1);
                    }

                    // Draw sub-block meshes (same pipeline + uniform as their
                    // parent chunk). Iterate per-ship sources so we draw
                    // sub-blocks only for ships we actually rendered.
                    for ship_transform in &ro.block_ships {
                        for (chunk_pos, sub_mesh) in br.sub_blocks_for_source(ship_transform.source) {
                            let key = voxeldust_core::block::client_chunks::ChunkKey {
                                source: ship_transform.source,
                                chunk: chunk_pos,
                            };
                            if let Some(&uniform_idx) = chunk_uniform_map.get(&key) {
                                pass.set_bind_group(0, &gpu.bind_group, &[(uniform_idx as u32) * 256]);
                                pass.set_vertex_buffer(0, sub_mesh.vertex_buf.slice(..));
                                pass.set_index_buffer(sub_mesh.index_buf.slice(..), wgpu::IndexFormat::Uint32);
                                pass.draw_indexed(0..sub_mesh.index_count, 0, 0..1);
                            }
                        }
                    }

                    // Draw sub-grid meshes with per-sub-grid transforms (mechanical mounts).
                    // Two-pass approach: first compute all uniforms, then batch-upload,
                    // then issue draw calls — avoids per-sub-grid write_buffer overhead.
                    if !sub_grid_transforms.is_empty() {
                        let sg_uniform_start = block_start + chunk_idx;
                        let mut sg_draws: Vec<(usize, &ChunkGpuMesh)> = Vec::with_capacity(32);

                        for ship_transform in &ro.block_ships {
                            for (sg_key, sg_mesh) in br.sub_grid_meshes_for_source(ship_transform.source) {
                                let uniform_idx = block_start + chunk_idx;
                                if uniform_idx >= MAX_OBJECTS { break; }

                                // Compute sub-grid transform from world-space body isometry.
                                // T(world_pos) * R(world_rot) * T(-anchor_root)
                                // anchor_root is the fixed root-space anchor; world_pos/rot
                                // include parent chain (composed on server).
                                let sg_local = sub_grid_transforms.get(&sg_key.sub_grid_id)
                                    .map(|sgt| {
                                        Mat4::from_translation(sgt.translation)
                                            * Mat4::from_quat(sgt.rotation)
                                            * Mat4::from_translation(-sgt.anchor)
                                    })
                                    .unwrap_or(Mat4::IDENTITY);

                                let chunk_offset = Vec3::new(
                                    sg_key.chunk.x as f32 * CHUNK_SIZE as f32,
                                    sg_key.chunk.y as f32 * CHUNK_SIZE as f32,
                                    sg_key.chunk.z as f32 * CHUNK_SIZE as f32,
                                );
                                let model = ship_transform.base_transform
                                    * sg_local
                                    * Mat4::from_translation(chunk_offset);
                                let mvp = cam.vp * model;

                                let mut obj: ObjectUniforms = bytemuck::Zeroable::zeroed();
                                obj.mvp = mvp.to_cols_array_2d();
                                obj.model = model.to_cols_array_2d();
                                obj.color = [1.0, 1.0, 1.0, 0.0];
                                obj.material = [0.1, 0.7, 0.0, 0.0];
                                uniform_data[uniform_idx] = obj;

                                sg_draws.push((uniform_idx, sg_mesh));
                                chunk_idx += 1;
                            }
                        }

                        // Batch-upload all sub-grid uniforms in one write_buffer call.
                        let sg_uniform_end = block_start + chunk_idx;
                        if sg_uniform_end > sg_uniform_start {
                            gpu.queue.write_buffer(
                                &gpu.uniform_buf,
                                (sg_uniform_start as u64) * 256,
                                bytemuck::cast_slice(&uniform_data[sg_uniform_start..sg_uniform_end]),
                            );

                            for &(uniform_idx, sg_mesh) in &sg_draws {
                                pass.set_bind_group(0, &gpu.bind_group, &[(uniform_idx as u32) * 256]);
                                pass.set_vertex_buffer(0, sg_mesh.vertex_buf.slice(..));
                                pass.set_index_buffer(sg_mesh.index_buf.slice(..), wgpu::IndexFormat::Uint32);
                                pass.draw_indexed(0..sg_mesh.index_count, 0, 0..1);
                            }
                        }
                    }

                    // Draw piston arm meshes — one per prismatic sub-grid.
                    if let Some(ref arm_mesh) = br.piston_arm_mesh {
                        for ship_transform in &ro.block_ships {
                            for (sg_id, sgt) in sub_grid_transforms {
                                if sgt.joint_type != 1 { continue; } // only prismatic
                                if sgt.current_value.abs() < 0.001 { continue; } // no arm when retracted

                                let uniform_idx = block_start + chunk_idx;
                                if uniform_idx >= MAX_OBJECTS { break; }

                                // Compute arm model matrix:
                                // 1. Translate to mount block center
                                // 2. Rotate Z axis to align with face normal
                                // 3. Scale Z by current extension
                                let mount_center = Vec3::new(
                                    sgt.mount_pos.x as f32 + 0.5,
                                    sgt.mount_pos.y as f32 + 0.5,
                                    sgt.mount_pos.z as f32 + 0.5,
                                );
                                let face_normal = match sgt.mount_face {
                                    0 => Vec3::X,
                                    1 => Vec3::NEG_X,
                                    2 => Vec3::Y,
                                    3 => Vec3::NEG_Y,
                                    4 => Vec3::Z,
                                    5 => Vec3::NEG_Z,
                                    _ => Vec3::Y,
                                };
                                // Rotation that maps +Z to face_normal.
                                let face_rot = glam::Quat::from_rotation_arc(Vec3::Z, face_normal);
                                // Arm starts at the face surface (block center + 0.5 along normal)
                                // and extends current_value meters toward the child blocks.
                                let arm_model = ship_transform.base_transform
                                    * Mat4::from_translation(mount_center + face_normal * 0.5)
                                    * Mat4::from_quat(face_rot)
                                    * Mat4::from_scale(Vec3::new(1.0, 1.0, sgt.current_value));

                                let mvp = cam.vp * arm_model;
                                let mut obj: ObjectUniforms = bytemuck::Zeroable::zeroed();
                                obj.mvp = mvp.to_cols_array_2d();
                                obj.model = arm_model.to_cols_array_2d();
                                obj.color = [1.0, 1.0, 1.0, 0.0];
                                obj.material = [0.1, 0.7, 0.0, 0.0];
                                uniform_data[uniform_idx] = obj;

                                gpu.queue.write_buffer(
                                    &gpu.uniform_buf,
                                    (uniform_idx as u64) * 256,
                                    bytemuck::cast_slice(&uniform_data[uniform_idx..uniform_idx + 1]),
                                );
                                pass.set_bind_group(0, &gpu.bind_group, &[(uniform_idx as u32) * 256]);
                                pass.set_vertex_buffer(0, arm_mesh.vertex_buf.slice(..));
                                pass.set_index_buffer(arm_mesh.index_buf.slice(..), wgpu::IndexFormat::Uint32);
                                pass.draw_indexed(0..arm_mesh.index_count, 0, 0..1);

                                chunk_idx += 1;
                            }
                        }
                    }
                }
            }
        }

        // Block highlight: render a translucent cube at the targeted block.
        if let Some(target) = block_target {
            if current_shard_type == voxeldust_core::client_message::shard_type::SHIP && !ro.block_ships.is_empty() {
                let base_transform = ro.block_ships[0].base_transform;

                // Block center in ship-local space (root grid coordinates).
                let block_center = Vec3::new(
                    target.world_pos.x as f32 + 0.5,
                    target.world_pos.y as f32 + 0.5,
                    target.world_pos.z as f32 + 0.5,
                );
                // Apply sub-grid transform if the block belongs to a sub-grid.
                let sg_local = sub_grid_assignments.get(&target.world_pos)
                    .and_then(|sg_id| sub_grid_transforms.get(sg_id))
                    .map(|sgt| {
                        Mat4::from_translation(sgt.translation)
                            * Mat4::from_quat(sgt.rotation)
                            * Mat4::from_translation(-sgt.anchor)
                    })
                    .unwrap_or(Mat4::IDENTITY);
                let model = base_transform
                    * sg_local
                    * Mat4::from_translation(block_center)
                    * Mat4::from_scale(Vec3::splat(0.505));
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
                    if let Some(ref vbg) = gpu.voxel_bind_group { pass.set_bind_group(2, vbg, &[]); }
                    pass.set_vertex_buffer(0, gpu.sphere_vertex_buf.slice(..));
                    pass.set_index_buffer(gpu.sphere_index_buf.slice(..), wgpu::IndexFormat::Uint32);
                    pass.set_bind_group(0, &gpu.bind_group, &[(highlight_idx as u32) * 256]);
                    pass.draw_indexed(0..gpu.sphere_index_count, 0, 0..1);
                }
            }
        }
    }

    // -- Upload atmosphere uniforms --
    if let Some(atmo_buf) = &gpu.atmosphere_buf {
        let atmo_uniforms = if gfx_settings.atmosphere_enabled {
            if let Some(sys) = system_params {
                if let Some((planet_idx, planet_pos)) = find_atmosphere_planet(cam.cam_system_pos, sys, interpolated_bodies) {
                    build_atmosphere_uniforms(
                        &sys.planets[planet_idx], planet_pos, cam, &scene_lighting, gfx_settings,
                    )
                } else {
                    disabled_atmosphere_uniforms()
                }
            } else {
                disabled_atmosphere_uniforms()
            }
        } else {
            disabled_atmosphere_uniforms()
        };
        gpu.queue.write_buffer(atmo_buf, 0, bytemuck::bytes_of(&atmo_uniforms));

        // Upload cloud uniforms for the same planet (if it has clouds).
        if let (Some(sys), Some(ws)) = (system_params, latest_world_state) {
            if let Some((planet_idx, planet_pos)) = find_atmosphere_planet(cam.cam_system_pos, sys, interpolated_bodies) {
                let planet = &sys.planets[planet_idx];
                if planet.clouds.has_clouds {
                    let observer_m = cam.cam_system_pos - planet_pos;
                    let observer_km = [
                        (observer_m.x / 1000.0) as f32,
                        (observer_m.y / 1000.0) as f32,
                        (observer_m.z / 1000.0) as f32,
                    ];
                    let game_time = ws.game_time;
                    let sun_dir = [scene_lighting.sun_direction[0], scene_lighting.sun_direction[1], scene_lighting.sun_direction[2]];
                    let sun_int = scene_lighting.sun_color[3];
                    let sun_col = [scene_lighting.sun_color[0], scene_lighting.sun_color[1], scene_lighting.sun_color[2]];

                    let cloud_uni = crate::cloud_system::CloudSystem::build_uniforms(
                        planet, observer_km, game_time, sun_dir, sun_int, sun_col,
                    );
                    if let Some(cloud_buf) = &gpu.cloud_uniform_buf {
                        gpu.queue.write_buffer(cloud_buf, 0, bytemuck::bytes_of(&cloud_uni));
                    }

                    // Log cloud params once for debugging.
                    static CLOUD_LOGGED: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
                    if !CLOUD_LOGGED.load(std::sync::atomic::Ordering::Relaxed) {
                        CLOUD_LOGGED.store(true, std::sync::atomic::Ordering::Relaxed);
                        tracing::info!(
                            planet_r_km = cloud_uni.geometry[0],
                            cloud_base_km = cloud_uni.geometry[1],
                            cloud_thick_km = cloud_uni.geometry[2],
                            coverage = cloud_uni.density_params[0],
                            density_scale = cloud_uni.density_params[1],
                            cloud_type = cloud_uni.density_params[2],
                            shape_scale = cloud_uni.noise_params[0],
                            detail_scale = cloud_uni.noise_params[1],
                            weather_scale_km = cloud_uni.scatter[3],
                            "cloud uniforms"
                        );
                    }
                }
            }
        }
    }

    // -- Atmosphere LUT compute passes (Hillaire 2020 four-LUT pipeline, minus
    // the multi-scatter LUT which is deferred to a later phase).
    //
    // Only dispatched when the atmosphere is actually enabled — when the
    // disabled-atmosphere uniform is uploaded, the LUTs still get rebuilt but
    // the composite shader early-outs via the `radii.z > 0.5` check.
    //
    // Transmittance LUT: depends only on planet parameters. Rebuilding it
    // every frame is a tiny (256×64 = 16384 invocations × ~40 samples) job —
    // far simpler than tracking "planet changed" state.
    // Sky-view + aerial-view: depend on observer position + sun + view, so
    // must be rebuilt every frame.
    if gpu.hdr_enabled {
        if let (
            Some(t_pipe), Some(t_bg),
            Some(m_pipe), Some(m_bg),
            Some(s_pipe), Some(s_bg),
            Some(a_pipe), Some(a_bg),
        ) = (
            &gpu.atmo_transmittance_pipeline, &gpu.atmo_transmittance_bind_group,
            &gpu.atmo_multiscatter_pipeline, &gpu.atmo_multiscatter_bind_group,
            &gpu.atmo_skyview_pipeline, &gpu.atmo_skyview_bind_group,
            &gpu.atmo_aerial_pipeline, &gpu.atmo_aerial_bind_group,
        ) {
            let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("atmo_lut_dispatch"),
                timestamp_writes: None,
            });
            // Transmittance LUT (8×8 workgroups).
            cpass.set_pipeline(t_pipe);
            cpass.set_bind_group(0, t_bg, &[]);
            cpass.dispatch_workgroups(
                crate::gpu::TRANSMITTANCE_LUT_W.div_ceil(8),
                crate::gpu::TRANSMITTANCE_LUT_H.div_ceil(8),
                1,
            );
            // Multi-scatter LUT: workgroup_size (1, 1, 64) — one workgroup per
            // (r, mu) output texel, 64 sample rays reduced in shared memory.
            cpass.set_pipeline(m_pipe);
            cpass.set_bind_group(0, m_bg, &[]);
            cpass.dispatch_workgroups(
                crate::gpu::MULTISCATTER_LUT_W,
                crate::gpu::MULTISCATTER_LUT_H,
                1,
            );
            // Sky-view LUT (depends on transmittance + multi-scatter).
            cpass.set_pipeline(s_pipe);
            cpass.set_bind_group(0, s_bg, &[]);
            cpass.dispatch_workgroups(
                crate::gpu::SKY_VIEW_LUT_W.div_ceil(8),
                crate::gpu::SKY_VIEW_LUT_H.div_ceil(8),
                1,
            );
            // Aerial-view LUT (3D; outer loop over slices happens inside the shader).
            cpass.set_pipeline(a_pipe);
            cpass.set_bind_group(0, a_bg, &[]);
            cpass.dispatch_workgroups(
                crate::gpu::AERIAL_LUT_W.div_ceil(8),
                crate::gpu::AERIAL_LUT_H.div_ceil(8),
                1,
            );
        }
    }

    // -- HDR composite pass: tonemap HDR → swapchain --
    if gpu.hdr_enabled {
        if let (Some(pipeline), Some(bind_group), Some(params_buf), Some(lut_bg)) =
            (&gpu.composite_pipeline, &gpu.composite_bind_group, &gpu.composite_params_buf, &gpu.atmo_lut_bind_group)
        {
            // Upload composite params (screen dimensions).
            let w = gpu.config.width as f32;
            let h = gpu.config.height as f32;
            let params = crate::gpu::CompositeParams {
                screen_dims: [w, h, 1.0 / w, 1.0 / h],
            };
            gpu.queue.write_buffer(params_buf, 0, bytemuck::bytes_of(&params));

            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("composite"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                ..Default::default()
            });
            pass.set_pipeline(pipeline);
            pass.set_bind_group(0, bind_group, &[]);
            pass.set_bind_group(1, lut_bg, &[]);
            pass.draw(0..3, 0..1); // vertexless fullscreen triangle
        }
    }

    // egui HUD.
    let hud_ctx = HudContext {
        latest_world_state,
        secondary_world_state,
        interpolated_bodies,
        cam_system_pos: cam.cam_system_pos,
        vp: cam.vp,
        player_position,
        player_velocity,
        current_shard_type,
        is_piloting,
        connected,
        selected_thrust_tier,
        engines_off,
        cruise_active,
        atmo_comp_active,
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
