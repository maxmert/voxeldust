//! Solar lighting — a single `DirectionalLight` driven entirely by the
//! server-broadcast stellar physics state. No diagnostic colour, no magic
//! illuminance constants, no client-side seed derivation:
//!
//! 1. The star body (the `CelestialBodySnapshot` with `body_id == 0`) carries
//!    a `StellarState` payload populated by the server (system-shard for
//!    that observer's authoritative system, propagated to ship/planet shards
//!    via `SystemSceneUpdate.bodies`). The client reads from primary first,
//!    falling back to the SYSTEM secondary.
//! 2. Direction: per-camera, recomputed every frame as
//!    `(star_pos − camera_world_pos).normalize()`. Two players at the same
//!    world coordinate compute the same direction. Two players in different
//!    parts of the system see the sun where it physically should be from
//!    their own position.
//! 3. Colour: `body.stellar.color_linear_rgb` — the server pre-computed this
//!    via the Tanner-Helland blackbody approximation in `core::blackbody`
//!    from the star's effective temperature.
//! 4. Illuminance (lux): `L_W · η_sun / (4π · r²)` where `L_W = body.stellar.luminosity_w`,
//!    `η_sun = LUMINOUS_EFFICACY_SUN`, and `r = |star_pos − camera_world_pos|`.
//!    Inverse-square law, no fudge factors.
//! 5. Cascade shadows: configured from `LightingFidelity` in `GameConfig`.
//!    No literals here.
//!
//! Three quality presets see the same physics — Ultra and Low produce
//! identical sun colour, identical illuminance — they only differ in
//! cascade resolution / count.

use bevy::camera::visibility::{CascadesVisibleEntities, ViewVisibility};
use bevy::light::cascade::Cascade;
use bevy::light::{
    CascadeShadowConfig, CascadeShadowConfigBuilder, Cascades, DirectionalLightShadowMap,
    SimulationLightSystems, VolumetricLight,
};
use bevy::math::Vec3A;
use bevy::prelude::*;
use bevy::transform::TransformSystems;
use glam::DVec3;

use voxeldust_core::physics_constants::{AU_M, L_SUN_W, LUMINOUS_EFFICACY_SUN, PI};
use voxeldust_core::stellar::StellarState;

use crate::config::GameConfig;
use crate::shard::{
    CameraWorldPos, PrimaryShard, PrimaryWorldState, SecondaryWorldStates, ShardOriginSet,
};
use crate::shard_types::system::SYSTEM_SHARD_TYPE;

/// Bevy 0.18's GPU shadow-buffer hard cap on cascades per directional
/// light (`bevy_pbr/src/render/light.rs:191`,
/// `MAX_CASCADES_PER_LIGHT = 4`). `CascadeShadowConfigBuilder::build`
/// happily constructs N frusta for any N, but the GPU buffer stores
/// only the first 4 — fragments past cascade 3 sample uninitialised
/// slots and read garbage shadow values (often "no shadow"), so
/// distant lit surfaces (planets, far hulls) blow out into a glowing
/// halo. Clamping in [`spawn_solar_light`] is a defensive guard that
/// converts a silent miscompile into a loud `tracing::warn!`.
const MAX_CASCADES_PER_LIGHT_U8: u8 = 4;

#[derive(Component, Debug, Clone, Copy)]
pub struct SolarLight;

pub struct SolarLightPlugin;

impl Plugin for SolarLightPlugin {
    fn build(&self, app: &mut App) {
        app.add_systems(Startup, spawn_solar_light)
            // **Run AFTER `ShardOriginSet`** so that:
            //   * `CameraWorldPos` has been updated by `apply_worldstate_pose`,
            //   * `ChunkSource.Transform` has been rebased to the camera-relative
            //     frame by `rebase_shard_transforms`,
            //   * and `TransformPropagate` (which runs in `PostUpdate`) will
            //     see consistent state when it computes children's
            //     `GlobalTransform`.
            // Without this ordering, `update_solar_light` could write the
            // light's rotation from a stale `CameraWorldPos`, and the
            // cascade-shadow extract would read a transient state where
            // the light direction lags the chunks' world position by one
            // frame — producing the "lit during shard transition, dark in
            // steady state" symptom.
            .add_systems(Update, update_solar_light.after(ShardOriginSet));
    }
}

fn spawn_solar_light(mut commands: Commands, config: Res<GameConfig>) {
    let fidelity = &config.graphics.lighting;
    // Defensive clamp — see `MAX_CASCADES_PER_LIGHT_U8` doc-comment.
    // No active preset overshoots the cap; the warn fires only if a
    // user-facing override (Phase 9 settings UI) sets a higher value.
    let cascade_count = fidelity.cascade_count.min(MAX_CASCADES_PER_LIGHT_U8) as usize;
    if (fidelity.cascade_count as usize) > MAX_CASCADES_PER_LIGHT_U8 as usize {
        tracing::warn!(
            requested = fidelity.cascade_count,
            clamped_to = cascade_count,
            "lighting fidelity preset requests more cascades than Bevy's \
             MAX_CASCADES_PER_LIGHT cap — clamped to avoid garbage shadow reads"
        );
    }
    let shadow_config: CascadeShadowConfig = CascadeShadowConfigBuilder {
        num_cascades: cascade_count,
        minimum_distance: fidelity.cascade_minimum_distance_m,
        maximum_distance: fidelity.cascade_max_distance_m,
        first_cascade_far_bound: fidelity.first_cascade_far_bound_m,
        overlap_proportion: fidelity.cascade_overlap_proportion,
    }
    .build();

    // Spawn with a *non-zero* placeholder illuminance and a real (non-IDENTITY)
    // rotation. Bevy 0.18's CSM machinery initialises its cascade frusta and
    // shadow-map allocations from these values on the first frame; spawning
    // with `illuminance: 0` or `Transform::IDENTITY` can leave the light
    // entity in a half-initialised state where the shadow map is degenerate
    // (depth = 0 everywhere → every fragment reads as "in shadow" → black
    // hull regardless of subsequent illuminance updates).
    //
    // The placeholders are overwritten on the very next frame by
    // `update_solar_light` once `WorldState.bodies[body_id == 0]` lands —
    // so they never affect rendered output, only initialisation.
    // Match Bevy 0.18's canonical `examples/3d/lighting.rs` spawn pattern:
    // only `DirectionalLight + Transform + CascadeShadowConfig + Name`.
    // Do NOT manually insert `GlobalTransform::IDENTITY` —
    // `DirectionalLight`'s `#[require(GlobalTransform, …)]` chain inserts
    // a fresh `GlobalTransform::default()` AND `TransformPropagate`
    // populates it from `Transform` *before* the cascade-shadow render
    // extract runs. Manually pinning `GlobalTransform::IDENTITY` here
    // overrode that propagation for the first frame, leaving the cascade
    // extract reading the light pointing along world `-Z` and producing
    // a degenerate shadow map. (Verified against Bevy v0.18.1
    // examples/3d/lighting.rs and shadow_caster_receiver.rs.)
    let mut entity = commands.spawn((
        SolarLight,
        DirectionalLight {
            color: Color::WHITE,
            // Sol-at-Earth equivalent: physically meaningful placeholder
            // (`L_SUN_W · η_sun / (4π · AU_M²)`) so CSM initialises with
            // realistic frusta scales. Update overrides immediately.
            illuminance: (L_SUN_W * LUMINOUS_EFFICACY_SUN
                / (4.0 * PI * AU_M * AU_M)) as f32,
            shadows_enabled: fidelity.directional_shadows_enabled,
            // Voxel-tuned biases — see `LightingFidelity::shadow_depth_bias`
            // for the rationale. Re-asserted each frame in `update_solar_light`.
            shadow_depth_bias: fidelity.shadow_depth_bias,
            shadow_normal_bias: fidelity.shadow_normal_bias,
            ..default()
        },
        shadow_config,
        Transform::from_rotation(Quat::from_rotation_x(-std::f32::consts::FRAC_PI_4)),
        Name::new("solar_light"),
    ));
    // Phase 5: tag the sun as a volumetric scatterer so god rays (light
    // shafts) emerge from it when the camera also has `VolumetricFog`.
    // Quality-gated: enabling `volumetric_fog_enabled` on the camera
    // without also tagging the sun produces a fog with no scatterers,
    // i.e. flat ambient — so we mirror the same boolean here.
    if fidelity.volumetric_fog_enabled {
        entity.insert(VolumetricLight);
    }
}

#[allow(clippy::too_many_arguments)]
fn update_solar_light(
    primary: Res<PrimaryShard>,
    primary_ws: Res<PrimaryWorldState>,
    secondary_ws: Res<SecondaryWorldStates>,
    camera_world: Res<CameraWorldPos>,
    config: Res<GameConfig>,
    time: Res<Time>,
    mut log_acc: Local<f32>,
    cameras: Query<(&Transform, &GlobalTransform), With<crate::MainCamera>>,
    mut lights: Query<
        (
            &mut Transform,
            &mut DirectionalLight,
            &CascadesVisibleEntities,
            &ViewVisibility,
            &GlobalTransform,
        ),
        (With<SolarLight>, Without<crate::MainCamera>),
    >,
) {
    let fidelity = &config.graphics.lighting;
    let Ok((mut tf, mut light, cascades_visible, light_view_vis, light_gt)) =
        lights.single_mut()
    else {
        return;
    };
    *log_acc += time.delta_secs();
    let log_now = *log_acc >= 1.0;
    if log_now {
        *log_acc = 0.0;
    }

    // Find the star body. The dual-shard compositor keeps both primary's
    // and SYSTEM-secondary's bodies live; system-shard's authoritative
    // `stellar` payload propagates through both. We **prefer the source
    // that carries stellar state** — primary may be a SHIP shard whose
    // SystemSceneUpdate cache is briefly stale on connect, in which case
    // its star body lacks `stellar` while secondary still has it.
    let primary_body_pos_stellar = primary_ws
        .latest
        .as_ref()
        .and_then(|ws| ws.bodies.iter().find(|b| b.body_id == 0))
        .map(|b| (b.position, b.stellar));
    let secondary_body_pos_stellar = secondary_ws
        .by_shard_type
        .get(&SYSTEM_SHARD_TYPE)
        .and_then(|(ws, _)| ws.bodies.iter().find(|b| b.body_id == 0))
        .map(|b| (b.position, b.stellar));

    // Pick the candidate that has stellar populated; if neither has it,
    // fall back to whichever has a position (so direction at least
    // tracks the star even if illuminance can't be computed).
    let with_stellar: Option<(DVec3, StellarState)> =
        [primary_body_pos_stellar, secondary_body_pos_stellar]
            .iter()
            .filter_map(|c| c.as_ref())
            .find_map(|(pos, st)| st.as_ref().map(|s| (*pos, *s)));
    let position_only_fallback: Option<DVec3> =
        primary_body_pos_stellar
            .or(secondary_body_pos_stellar)
            .map(|(p, _)| p);

    let (star_pos, stellar): (DVec3, Option<StellarState>) = match (with_stellar, position_only_fallback) {
        (Some((p, s)), _) => (p, Some(s)),
        (None, Some(p)) => (p, None),
        (None, None) => {
            light.illuminance = 0.0;
            if log_now {
                tracing::warn!(
                    "solar_light: no star body (body_id=0) in primary or SYSTEM-secondary WorldState"
                );
            }
            return;
        }
    };

    // Per-camera sun direction. Star at server-authoritative position; camera
    // at server-authoritative position; one normalised subtraction.
    //
    // **Critical**: normalize in DOUBLE PRECISION first, then cast the unit
    // vector to f32. The naïve `Vec3::new(to_star.x as f32, ...)
    // .normalize_or_zero()` casts a 1.1×10¹⁰-magnitude vector to f32 BEFORE
    // normalizing. f32 precision at 10¹⁰ is ~1.3 km; as the ship moves at
    // 60 km/s the cast jitters by ~1 LSB per frame, producing direction
    // noise that propagates into the directional light's
    // `Transform.rotation` → cascade frustum → shadow map → visible
    // "shadows shifting randomly with camera movement" even when the ship
    // sits still. Normalising the f64 vector first bounds components to
    // [-1, 1] where f32 has full sub-millionth precision.
    let to_star = star_pos - camera_world.pos;
    let to_star_len = to_star.length();
    let to_star_unit = to_star.normalize_or_zero();
    let sun_dir_f32 = Vec3::new(
        to_star_unit.x as f32,
        to_star_unit.y as f32,
        to_star_unit.z as f32,
    );
    if sun_dir_f32.length_squared() > 0.0 {
        // Bevy's DirectionalLight emits along the entity-local -Z. Rotating
        // local +Z onto the to-star vector makes -Z = -(to_star) = the
        // direction the photons travel from star to observer.
        tf.rotation = Quat::from_rotation_arc(Vec3::Z, sun_dir_f32);
    }

    // Photometric colour + illuminance. Both come from the broadcast
    // physics-derived stellar state. We have two paths:
    //
    //   * Stellar present (the normal Phase 1 path): use the spectrally-
    //     correct linear-sRGB colour and inverse-square photometric
    //     illuminance derived from `body.stellar.luminosity_w`.
    //
    //   * Stellar missing (server-shard not rebuilt, or first-frame race
    //     before SystemSceneUpdate arrives, or legacy-format peer): fall
    //     back to the broadcast `body.color` and an illuminance computed
    //     from a single solar luminosity at the same observer distance.
    //     This produces a consistent, physics-grounded fallback (`L_SUN_W`
    //     and `AU_M` are named physics constants) so the ship is always
    //     visibly lit even when the system is in a transitional state.
    //
    // The fallback path uses the *legacy* body.color (which after the
    // physics-correct `SystemParams::from_seed` rewrite is now also
    // linear-sRGB physics-derived; in older builds it's the legacy 3-bucket
    // stylised triplet). Either way the colour is meaningful.
    // Clamp r² floor to 1 m² to prevent division by zero when the camera
    // happens to land exactly on the star's coordinate origin (unphysical
    // but possible at game-start before the player is placed).
    let r_squared = (to_star_len * to_star_len).max(1.0);
    let (color_linear_rgb, lux) = match stellar {
        Some(s) => {
            let lux = (s.luminosity_w as f64) * LUMINOUS_EFFICACY_SUN
                / (4.0 * PI * r_squared);
            (s.color_linear_rgb, lux)
        }
        None => {
            // Treat the star as Sol-equivalent at the same observer distance.
            // The colour is whatever the server broadcast in `body.color`.
            // This is a pure-physics fallback — `L_SUN_W` is a named CODATA
            // value, not a magic number.
            let lux = L_SUN_W * LUMINOUS_EFFICACY_SUN / (4.0 * PI * r_squared);
            // Find the broadcast body.color from whichever source we used
            // for star_pos. We re-read because the position-only path lost
            // colour up-stream — refetch the same body.
            let fallback_color = primary_ws
                .latest
                .as_ref()
                .and_then(|ws| ws.bodies.iter().find(|b| b.body_id == 0).map(|b| b.color))
                .or_else(|| {
                    secondary_ws
                        .by_shard_type
                        .get(&SYSTEM_SHARD_TYPE)
                        .and_then(|(ws, _)| {
                            ws.bodies.iter().find(|b| b.body_id == 0).map(|b| b.color)
                        })
                })
                .unwrap_or([1.0, 1.0, 1.0]);
            (fallback_color, lux)
        }
    };

    light.color = Color::linear_rgb(
        color_linear_rgb[0],
        color_linear_rgb[1],
        color_linear_rgb[2],
    );
    // `max_directional_illuminance_lux` is a render-stability guard:
    //   * HDR presets ship with `1.0e9` (effectively no clamp) — AgX
    //     handles physical Sol-at-Earth (~1.27e5 lux) cleanly.
    //   * Low/LDR preset ships with `30_000` to land the lit-face value
    //     at the LDR clip threshold under `ev100 = 11`.
    // Either way the gameplay/photometric `lux` is preserved for any
    // non-rendering consumer that wants the true reading.
    light.illuminance = (lux as f32).min(fidelity.max_directional_illuminance_lux);
    // Hot-reload the shadows knob in case the user toggled it via the
    // graphics config UI mid-session. Cheap to set unconditionally.
    light.shadows_enabled = fidelity.directional_shadows_enabled;
    // Re-assert voxel-tuned shadow biases each frame in case any other
    // system somehow touches the DirectionalLight component. See
    // `LightingFidelity::shadow_depth_bias` for the rationale (Bevy
    // defaults are calibrated for organic geometry; voxel cubes need
    // much more aggressive biases at the 90° corners).
    light.shadow_depth_bias = fidelity.shadow_depth_bias;
    light.shadow_normal_bias = fidelity.shadow_normal_bias;

    if log_now {
        let primary_shard_type = primary.current.map(|k| k.shard_type);
        let tf_rot = (tf.rotation.x, tf.rotation.y, tf.rotation.z, tf.rotation.w);
        let cascade_caster_count: usize = cascades_visible
            .entities
            .values()
            .map(|cascades| cascades.iter().map(|c| c.len()).sum::<usize>())
            .sum();
        let cascade_views = cascades_visible.entities.len();
        // **CRITICAL DIAGNOSTIC**: light's ViewVisibility — if false,
        // `extract_lights` removes the directional light from the render
        // world entirely (`bevy_pbr-0.18.1/src/render/light.rs:520-526`),
        // and PBR rendering computes zero directional contribution. This
        // matches the "nothing is lit by the sun" symptom.
        let light_view_visible = light_view_vis.get();
        let light_gt_translation = light_gt.translation();
        let light_gt_rotation = light_gt.rotation();
        // Camera's GlobalTransform — used by the cascade computation to
        // build the frustum corners. If this differs from the camera's
        // local Transform (e.g., due to TransformPropagate not running
        // for the camera), cascades are computed against stale data.
        let (cam_tf, cam_gt) = cameras
            .iter()
            .next()
            .map(|(t, g)| {
                (
                    (t.translation, t.rotation),
                    (g.translation(), g.rotation()),
                )
            })
            .unwrap_or((
                (Vec3::ZERO, Quat::IDENTITY),
                (Vec3::ZERO, Quat::IDENTITY),
            ));
        if stellar.is_some() {
            tracing::info!(
                primary_shard_type = ?primary_shard_type,
                cam_world = ?(camera_world.pos.x, camera_world.pos.y, camera_world.pos.z),
                star = ?(star_pos.x, star_pos.y, star_pos.z),
                dist_m = to_star_len,
                sun_dir = ?(sun_dir_f32.x, sun_dir_f32.y, sun_dir_f32.z),
                tf_rot = ?tf_rot,
                color = ?color_linear_rgb,
                photometric_lux = lux,
                applied_lux = light.illuminance,
                shadows_enabled = light.shadows_enabled,
                shadow_depth_bias = light.shadow_depth_bias,
                shadow_normal_bias = light.shadow_normal_bias,
                cascade_caster_count,
                cascade_views,
                light_view_visible,
                light_gt_translation = ?(light_gt_translation.x, light_gt_translation.y, light_gt_translation.z),
                light_gt_rotation = ?(light_gt_rotation.x, light_gt_rotation.y, light_gt_rotation.z, light_gt_rotation.w),
                cam_local_translation = ?cam_tf.0,
                cam_local_rotation = ?cam_tf.1,
                cam_global_translation = ?cam_gt.0,
                cam_global_rotation = ?cam_gt.1,
                clamp_lux = fidelity.max_directional_illuminance_lux,
                "solar_light: physics path (stellar present)"
            );
        } else {
            tracing::warn!(
                primary_shard_type = ?primary_shard_type,
                cam_world = ?(camera_world.pos.x, camera_world.pos.y, camera_world.pos.z),
                star = ?(star_pos.x, star_pos.y, star_pos.z),
                dist_m = to_star_len,
                tf_rot = ?tf_rot,
                primary_has_body = primary_ws.latest.is_some(),
                secondary_has_system = secondary_ws.by_shard_type.contains_key(&SYSTEM_SHARD_TYPE),
                color = ?color_linear_rgb,
                illum_lux = light.illuminance,
                "solar_light: stellar-fallback path (server-shard not yet shipping StellarState — \
                 rebuild server with `./dev-cluster.sh rebuild` to get spectrally-correct sun)"
            );
        }
    }
}

/// Replace Bevy's auto-built cascade frusta with cube-bounded ones centered
/// at world origin (= camera position in our floating-origin scene).
///
/// **Why**: Bevy 0.18's `build_directional_light_cascades` projects the
/// camera's view frustum corners into light space and snaps the orthographic
/// projection's center to texel grid for stability
/// (`bevy_light-0.18.1/src/cascade.rs:241-296`). The snap stabilizes
/// camera *translation* (within texel granularity) but not *rotation* — a
/// 1° camera rotation per frame shifts the cascade center by ~70 mm in
/// light space, many texels at our 4 mm/texel near-cascade resolution,
/// producing the visible "shadows swim with mouse motion" artifact even
/// when the ship is stationary.
///
/// **Fix**: derive the cascade from a *cube* of side `2 · far_bound`
/// centered at world origin. The cube is invariant under camera rotation,
/// so the cascade is rotation-invariant. Trade-off is texel resolution:
/// the worst-case projected diameter is `2 · far_bound · √3` instead of
/// the camera frustum's tight bound, so the near cascade goes from
/// ~3 mm/texel to ~7 mm/texel — still far finer than our 1 m voxels and
/// an unconditional win over visible swimming.
fn stabilize_solar_cascades(
    cameras: Query<(Entity, &Camera), With<crate::MainCamera>>,
    mut lights: Query<
        (&GlobalTransform, &CascadeShadowConfig, &mut Cascades),
        With<SolarLight>,
    >,
    shadow_map: Res<DirectionalLightShadowMap>,
) {
    let Some((camera_entity, _)) = cameras.iter().find(|(_, c)| c.is_active) else {
        return;
    };
    let map_size = shadow_map.size as f32;

    for (light_transform, config, mut cascades) in &mut lights {
        // Directional lights ignore translation; cascade math uses rotation
        // only (Bevy convention — `cascade.rs:227`).
        let world_from_light = Mat4::from_quat(light_transform.rotation());

        // Mirror Bevy's per-cascade near/far derivation
        // (`cascade.rs:232-238`): cascade N's near bound is cascade N-1's
        // far bound scaled by the overlap factor.
        let overlap_factor = 1.0 - config.overlap_proportion;
        let mut near_bound = config.minimum_distance;
        let mut new_cascades = Vec::with_capacity(config.bounds.len());
        for &far_bound in &config.bounds {
            new_cascades.push(build_stable_cube_cascade(
                near_bound,
                far_bound,
                world_from_light,
                map_size,
            ));
            near_bound = overlap_factor * far_bound;
        }
        cascades.cascades.insert(camera_entity, new_cascades);
    }
}

fn build_stable_cube_cascade(
    _near_bound: f32,
    far_bound: f32,
    world_from_light: Mat4,
    shadow_map_size: f32,
) -> Cascade {
    // Cube of side 2*far_bound around world origin. The 8 corners are
    // axis-aligned and rotation-invariant — that's the point.
    let r = far_bound;
    let corners = [
        Vec3A::new(r, r, r),
        Vec3A::new(r, r, -r),
        Vec3A::new(r, -r, r),
        Vec3A::new(r, -r, -r),
        Vec3A::new(-r, r, r),
        Vec3A::new(-r, r, -r),
        Vec3A::new(-r, -r, r),
        Vec3A::new(-r, -r, -r),
    ];

    let light_to_world_inverse = world_from_light.transpose();

    let mut min = Vec3A::splat(f32::MAX);
    let mut max = Vec3A::splat(f32::MIN);
    for corner in corners {
        let corner_light = light_to_world_inverse.transform_point3a(corner);
        min = min.min(corner_light);
        max = max.max(corner_light);
    }

    // **Critical**: the X/Y scale in `clip_from_cascade` (= `2/cascade_diameter`)
    // must match the actual `min/max` extents — otherwise fragments in the
    // "padding" region of the orthographic frustum that don't have a caster
    // get the cleared shadow-map value (which is "everything occluded" with
    // reverse-Z) and read as in-shadow even though no caster covers them.
    // The previous attempt used `2r√3` (worst-case cube body diagonal)
    // here while min/max came from actual cube-corner projections (often
    // ~`2r`), creating a mismatched "in-shadow halo" that turned the
    // entire scene black.
    let extent_x = max.x - min.x;
    let extent_y = max.y - min.y;
    let cascade_diameter = extent_x.max(extent_y).ceil();
    let cascade_texel_size = cascade_diameter / shadow_map_size;

    // Center at the actual cube projection's centroid (≈ world origin in
    // light space, but exactly correct for any light direction). Snap to
    // texel grid for sub-pixel stability — same convention as
    // `cascade.rs:292-296`.
    let near_plane_center = Vec3A::new(
        ((0.5 * (min.x + max.x)) / cascade_texel_size).floor() * cascade_texel_size,
        ((0.5 * (min.y + max.y)) / cascade_texel_size).floor() * cascade_texel_size,
        max.z,
    );

    // Faithful matrix construction (`cascade.rs:299-326`) so the shader-side
    // shadow-lookup ABI is unchanged.
    let world_from_light_transpose = world_from_light.transpose();
    let cascade_from_world = Mat4::from_cols(
        world_from_light_transpose.x_axis,
        world_from_light_transpose.y_axis,
        world_from_light_transpose.z_axis,
        (-near_plane_center).extend(1.0),
    );
    let world_from_cascade = Mat4::from_cols(
        world_from_light.x_axis,
        world_from_light.y_axis,
        world_from_light.z_axis,
        world_from_light * near_plane_center.extend(1.0),
    );

    let r_recip = (max.z - min.z).recip();
    let clip_from_cascade = Mat4::from_cols(
        Vec4::new(2.0 / cascade_diameter, 0.0, 0.0, 0.0),
        Vec4::new(0.0, 2.0 / cascade_diameter, 0.0, 0.0),
        Vec4::new(0.0, 0.0, r_recip, 0.0),
        Vec4::new(0.0, 0.0, 1.0, 1.0),
    );

    let clip_from_world = clip_from_cascade * cascade_from_world;

    Cascade {
        world_from_cascade,
        clip_from_cascade,
        clip_from_world,
        texel_size: cascade_texel_size,
    }
}
