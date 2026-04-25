//! Solar lighting — spawns a single `DirectionalLight` driven by
//! whichever WorldState's `lighting: LightingData` is currently
//! authoritative. Works for every shard type:
//!
//! * **SHIP primary** → system secondary's `LightingData` (the
//!   system's star). Ship-shard itself also publishes the lighting
//!   it computed for the ship's exterior position, so we prefer the
//!   primary when present.
//! * **PLANET primary** → planet shard's lighting (sun filtered
//!   through planet-provided atmospheric tint).
//! * **SYSTEM primary (EVA)** → system shard's lighting direct.
//! * **GALAXY primary (warp)** → no sun (stars become the light
//!   source visually; no direct illumination needed).
//!
//! The light supports shadow casting with Bevy's cascaded shadow
//! maps (CSM), so the star's rays are occluded by ship hull,
//! creating proper patches of light/shade through windows and open
//! cockpits.

use bevy::light::{CascadeShadowConfig, CascadeShadowConfigBuilder};
use bevy::prelude::*;
use glam::DVec3;

use crate::shard::{CameraWorldPos, PrimaryWorldState, SecondaryWorldStates};
use crate::shard_types::system::SYSTEM_SHARD_TYPE;

#[derive(Component, Debug, Clone, Copy)]
pub struct SolarLight;

pub struct SolarLightPlugin;

impl Plugin for SolarLightPlugin {
    fn build(&self, app: &mut App) {
        app.add_systems(Startup, spawn_solar_light)
            .add_systems(Update, update_solar_light);
    }
}

fn spawn_solar_light(mut commands: Commands) {
    // Cascaded shadow maps: 4 cascades spanning ~0.1 m (closest
    // voxel detail) up to ~200 m (entire ship + parked neighbours).
    // Tight first cascade keeps block-level shadow edges crisp.
    let shadow_config: CascadeShadowConfig = CascadeShadowConfigBuilder {
        num_cascades: 4,
        minimum_distance: 0.1,
        maximum_distance: 250.0,
        first_cascade_far_bound: 4.0,
        overlap_proportion: 0.2,
    }
    .build();

    // Minimal spawn — DirectionalLight has `#[require(Cascades,
    // CascadesFrusta, CascadeShadowConfig, CascadesVisibleEntities,
    // Transform, Visibility, VisibilityClass)]` in Bevy 0.18.
    // Providing some of those explicitly while omitting others
    // (notably the Cascades triple) can leave the entity in a
    // half-initialised state where it appears in the world but the
    // render extract skips it. Letting `#[require]` populate
    // defaults (and the `on_add` hook add VisibilityClass) is the
    // Bevy-idiomatic way; we override only what we actually need.
    commands.spawn((
        SolarLight,
        DirectionalLight {
            // **DIAGNOSTIC RED**: bright red so any face the
            // directional light actually reaches is unmistakably
            // RED-tinted. If the planet / chunks still look like
            // their material colors (no red wash), the light is NOT
            // being applied to the meshes at all — render-pipeline
            // / extract issue, not calibration.
            color: Color::srgb(1.0, 0.0, 0.0),
            illuminance: 80_000.0,
            shadows_enabled: true,
            ..default()
        },
        shadow_config,
        Transform::from_rotation(Quat::from_rotation_x(-std::f32::consts::FRAC_PI_4)),
        Name::new("solar_light"),
    ));
}

fn update_solar_light(
    primary_ws: Res<PrimaryWorldState>,
    secondary_ws: Res<SecondaryWorldStates>,
    camera_world: Res<CameraWorldPos>,
    time: Res<Time>,
    mut log_acc: Local<f32>,
    mut lights: Query<
        (&mut Transform, &mut DirectionalLight),
        With<SolarLight>,
    >,
) {
    let Ok((mut tf, mut light)) = lights.single_mut() else {
        return;
    };
    *log_acc += time.delta_secs();
    let log_now = *log_acc >= 1.0;
    if log_now {
        *log_acc = 0.0;
    }

    // Color / intensity / ambient stay server-authoritative.
    // Source-of-truth precedence: SYSTEM secondary (the star owner)
    // first, falling back to the primary's cached lighting if no
    // SYSTEM is open (rare).
    let lighting = secondary_ws
        .by_shard_type
        .get(&SYSTEM_SHARD_TYPE)
        .and_then(|ws| ws.lighting.as_ref())
        .or_else(|| primary_ws.latest.as_ref().and_then(|ws| ws.lighting.as_ref()));

    let Some(l) = lighting else {
        light.illuminance = 0.0;
        if log_now {
            tracing::warn!("solar_light: no lighting data in primary or SYSTEM secondary WS");
        }
        return;
    };

    // **DIAGNOSTIC**: keep color RED (set at spawn) and don't touch
    // illuminance — we want the user to see whether the directional
    // light reaches the chunks at all. If sun-facing faces tint RED,
    // lighting is reaching them and we restore color/intensity from
    // the server's `LightingData` afterwards. If they're still
    // cream/white/pink (vertex color_hint values), the directional
    // light isn't being applied to the chunks at all.
    let _ = l;

    // **Sun direction is recomputed client-side every frame** rather
    // than read from `l.sun_direction`. Server convention computes
    // `sun_direction = (star_pos - observer_pos).normalize()` for ONE
    // observer (the first hosted ship for SHIP shards, the first EVA
    // player for SYSTEM shards, or the `(1e11, 0, 0)` fallback when
    // neither exists). Every other ship / EVA player in the same
    // system shares that single value, so a ship parked far from the
    // chosen observer sees the sun coming from the wrong angle.
    //
    // Camera-world position + the star's authoritative system-space
    // position (broadcast as `WorldState.bodies[]` with `body_id == 0`)
    // are both available client-side, so the correct per-camera
    // sun direction is one subtraction away. Bevy's directional light
    // emits along its entity-local -Z; rotating local +Z onto the
    // sun_direction makes -Z = -sun_direction = star→observer travel.
    //
    // The bodies array is searched on the SECONDARY first too —
    // the SYSTEM secondary always carries the freshest celestial
    // body data; SHIP-primary's cached `scene.bodies` lags by the
    // SystemSceneUpdate cadence.
    let star_pos = secondary_ws
        .by_shard_type
        .get(&SYSTEM_SHARD_TYPE)
        .and_then(|ws| {
            ws.bodies
                .iter()
                .find(|b| b.body_id == 0)
                .map(|b| DVec3::new(b.position.x, b.position.y, b.position.z))
        })
        .or_else(|| {
            primary_ws.latest.as_ref().and_then(|ws| {
                ws.bodies
                    .iter()
                    .find(|b| b.body_id == 0)
                    .map(|b| DVec3::new(b.position.x, b.position.y, b.position.z))
            })
        });

    let Some(star_pos) = star_pos else {
        if log_now {
            tracing::warn!(
                "solar_light: no star body (body_id=0) in WS — direction not updated, light using stale rotation"
            );
        }
        return;
    };

    let to_star = star_pos - camera_world.pos;
    let to_star_len = to_star.length();
    let sun_dir_f32 = Vec3::new(
        to_star.x as f32,
        to_star.y as f32,
        to_star.z as f32,
    )
    .normalize_or_zero();
    if sun_dir_f32.length_squared() > 0.0 {
        tf.rotation = Quat::from_rotation_arc(Vec3::Z, sun_dir_f32);
    }

    if log_now {
        tracing::info!(
            cam_world = ?(camera_world.pos.x, camera_world.pos.y, camera_world.pos.z),
            star = ?(star_pos.x, star_pos.y, star_pos.z),
            dist_m = to_star_len,
            sun_dir = ?(sun_dir_f32.x, sun_dir_f32.y, sun_dir_f32.z),
            illum_lux = light.illuminance,
            "solar_light: updated"
        );
    }
}
