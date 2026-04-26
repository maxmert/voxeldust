//! Atmospheric scattering — Phase 3.
//!
//! Consumes the server-broadcast [`PlanetGeophysicalState`] (in
//! `WorldStateData.bodies[i].planetary`) and configures Bevy 0.18's
//! [`bevy::pbr::Atmosphere`] + [`bevy::pbr::ScatteringMedium`] +
//! [`bevy::light::AtmosphereEnvironmentMapLight`] on the lighting camera.
//!
//! ## Determinism + BE-authority
//!
//! Every value used to build the `ScatteringMedium` (Rayleigh/Mie/Ozone
//! coefficients, scale heights, phase asymmetry, ground albedo, planet
//! bottom / top radius) is broadcast verbatim by the server. The client
//! does not re-derive any geophysical quantity from a seed — the
//! `tests/no_client_seed_derivation.rs` harness enforces this.
//!
//! Two clients that receive byte-identical broadcast bytes build
//! byte-identical `ScatteringMedium` assets and produce visually
//! identical atmospheric scattering at every wall-clock moment.
//!
//! ## Seamless transitions (per the original Phase 3 plan)
//!
//! The component lifecycle is keyed to **shard CONNECTION**, not primary
//! status. The orchestrator pre-connects a PLANET shard as a *secondary*
//! during approach (well before primary promotion); the moment that
//! happens, this module's `update_atmosphere` system sees a connected
//! PLANET shard with `planetary` data on its home body, builds the
//! `ScatteringMedium`, and inserts the `Atmosphere` component. The medium
//! parameters are static for the planet's session lifetime, so no swap
//! happens on the subsequent primary promotion — the transition is
//! invisible to lighting.
//!
//! On exit, the orchestrator keeps the PLANET secondary live for a grace
//! window; altitude rises through Bevy's exponential `ScatteringMedium`
//! falloff so the atmosphere becomes optically thin naturally; finally,
//! when the orchestrator drops the PLANET shard, this module removes
//! `Atmosphere` + co. and the IBL state machine swaps back to the
//! starfield environment map. The optical-depth ramp masks the boolean
//! transition.

use std::hash::{Hash, Hasher};

use bevy::asset::Assets;
use bevy::light::AtmosphereEnvironmentMapLight;
use bevy::math::Vec3;
use bevy::pbr::{
    Atmosphere, AtmosphereMode, AtmosphereSettings, Falloff, PhaseFunction, ScatteringMedium,
    ScatteringTerm,
};
use bevy::prelude::*;

use voxeldust_core::client_message::WorldStateData;
use voxeldust_core::geophysics::PlanetGeophysicalState;

use crate::config::{AtmosphereMethod, GameConfig};
use crate::lighting::camera::LightingCamera;
use crate::shard::registry::{PrimaryShard, Secondaries};
use crate::shard::worldstate::{PrimaryWorldState, SecondaryWorldStates};
use crate::shard_types::planet::PLANET_SHARD_TYPE;

// ──────────────────────────────────────────────────────────────────────────
// Internal constants
// ──────────────────────────────────────────────────────────────────────────

/// Falloff scale resolution for [`ScatteringMedium`] LUTs. 256 is the
/// engine-wide default and matches Bevy's `earthlike` constructor; raising
/// it costs memory + bake time without visible improvement at typical
/// viewing distances.
const SCATTERING_MEDIUM_FALLOFF_RESOLUTION: u32 = 256;

/// Phase-function resolution for the Mie / Rayleigh angular distribution.
/// Same engine-wide default.
const SCATTERING_MEDIUM_PHASE_RESOLUTION: u32 = 256;

/// Threshold below which an ozone term is treated as absent (avoids
/// inserting a `ScatteringTerm` whose coefficients are all zero —
/// Bevy's prefilter accepts that but it's wasted work).
/// Units: m⁻¹.
const OZONE_PRESENCE_THRESHOLD: f32 = 1.0e-9;

// ──────────────────────────────────────────────────────────────────────────
// Resource + plugin
// ──────────────────────────────────────────────────────────────────────────

/// Tracks the active scattering-medium asset and the geophysics that
/// produced it. Re-uploads the asset only when broadcast params change
/// (e.g. on shard transitions to a different planet); a static planet
/// reuses the same handle across frames.
#[derive(Resource, Default)]
pub struct AtmosphereState {
    /// The live medium handle attached to the camera's `Atmosphere`.
    medium_handle: Option<Handle<ScatteringMedium>>,
    /// Hash of the broadcast `PlanetGeophysicalState` the medium was built
    /// from. `None` until the first build.
    last_built_hash: Option<u64>,
    /// Bottom radius (planet radius, m) used for the live `Atmosphere`
    /// component. Cached to detect radius changes (planet swaps).
    last_bottom_radius_m: Option<f32>,
}

pub struct AtmospherePlugin;

impl Plugin for AtmospherePlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<AtmosphereState>()
            .add_systems(Update, update_atmosphere);
    }
}

// ──────────────────────────────────────────────────────────────────────────
// Update system
// ──────────────────────────────────────────────────────────────────────────

/// The single system that owns Atmosphere lifecycle on the camera.
///
/// Each frame:
///   1. Find any connected PLANET shard's WorldState (primary or secondary).
///   2. Locate the "home planet" body (closest to that shard's origin)
///      and check that it carries broadcast `planetary` data with
///      `has_atmosphere == true`.
///   3. If yes, build / reuse the `ScatteringMedium` asset and
///      insert `Atmosphere`, `AtmosphereSettings`, and
///      `AtmosphereEnvironmentMapLight` on the lighting camera.
///   4. If no, ensure those three components are absent.
fn update_atmosphere(
    mut commands: Commands,
    config: Res<GameConfig>,
    primary: Res<PrimaryShard>,
    secondaries: Res<Secondaries>,
    primary_ws: Res<PrimaryWorldState>,
    secondary_ws: Res<SecondaryWorldStates>,
    mut media: ResMut<Assets<ScatteringMedium>>,
    mut state: ResMut<AtmosphereState>,
    cameras: Query<
        (Entity, Option<&Atmosphere>, Option<&AtmosphereEnvironmentMapLight>),
        With<LightingCamera>,
    >,
) {
    let active = find_active_planet(&primary, &secondaries, &primary_ws, &secondary_ws);

    for (entity, existing_atmosphere, existing_envmap) in &cameras {
        match &active {
            Some(planet) if planet.geophysics.has_atmosphere => {
                let target_bottom_radius_m = planet.radius_m as f32;
                let new_hash = hash_geophysics(&planet.geophysics);

                // Re-upload the medium asset only when the broadcast
                // params change. Same planet, same handle, no GPU work.
                let medium_handle = if state.last_built_hash != Some(new_hash) {
                    let medium = build_scattering_medium(&planet.geophysics);
                    let handle = media.add(medium);
                    state.medium_handle = Some(handle.clone());
                    state.last_built_hash = Some(new_hash);
                    handle
                } else {
                    state
                        .medium_handle
                        .clone()
                        .expect("medium handle must exist when last_built_hash is Some")
                };

                let target_settings = build_atmosphere_settings(&config);
                let target_envmap_intensity =
                    config.graphics.lighting.ibl_intensity_atmosphere;

                let needs_atmosphere_insert = match existing_atmosphere {
                    Some(a) => {
                        a.bottom_radius != target_bottom_radius_m
                            || a.medium != medium_handle
                            || state.last_bottom_radius_m != Some(target_bottom_radius_m)
                    }
                    None => true,
                };
                if needs_atmosphere_insert {
                    let atmosphere = Atmosphere {
                        bottom_radius: target_bottom_radius_m,
                        top_radius: target_bottom_radius_m
                            + planet.geophysics.atmosphere_top_altitude_m,
                        ground_albedo: Vec3::from_array(
                            planet.geophysics.ground_albedo_linear_rgb,
                        ),
                        medium: medium_handle,
                    };
                    commands.entity(entity).insert(atmosphere);
                    commands.entity(entity).insert(target_settings.clone());
                    state.last_bottom_radius_m = Some(target_bottom_radius_m);
                    tracing::info!(
                        bottom_radius_m = target_bottom_radius_m,
                        top_radius_m = target_bottom_radius_m
                            + planet.geophysics.atmosphere_top_altitude_m,
                        scale_height_m = planet.geophysics.scale_height_m,
                        surface_pressure_pa = planet.geophysics.surface_pressure_pa,
                        composition_n2 = planet.geophysics.composition_n2,
                        composition_o2 = planet.geophysics.composition_o2,
                        composition_co2 = planet.geophysics.composition_co2,
                        "atmosphere: inserted on camera"
                    );
                }

                let needs_envmap = match existing_envmap {
                    Some(e) => (e.intensity - target_envmap_intensity).abs() > f32::EPSILON,
                    None => true,
                };
                if needs_envmap {
                    commands.entity(entity).insert(AtmosphereEnvironmentMapLight {
                        intensity: target_envmap_intensity,
                        ..AtmosphereEnvironmentMapLight::default()
                    });
                }
            }
            _ => {
                if existing_atmosphere.is_some() {
                    commands.entity(entity).remove::<Atmosphere>();
                    commands.entity(entity).remove::<AtmosphereSettings>();
                    state.last_bottom_radius_m = None;
                    tracing::info!("atmosphere: removed from camera");
                }
                if existing_envmap.is_some() {
                    commands
                        .entity(entity)
                        .remove::<AtmosphereEnvironmentMapLight>();
                }
            }
        }
    }
}

// ──────────────────────────────────────────────────────────────────────────
// Active-planet selection
// ──────────────────────────────────────────────────────────────────────────

/// Resolved "what planet are we standing on / approaching" info derived
/// from connected shards' WorldState broadcasts.
struct ActivePlanet {
    /// Planet radius in metres. Used as `Atmosphere::bottom_radius`.
    radius_m: f64,
    /// Server-broadcast atmospheric / bulk-physics state.
    geophysics: PlanetGeophysicalState,
}

/// Returns the home-planet info from the highest-priority connected
/// PLANET shard's WorldState (primary first, then any PLANET secondary),
/// or `None` if no PLANET shard is currently connected with broadcast
/// `planetary` data.
///
/// "Home planet" within a WorldState = the body whose position is
/// closest to that shard's origin. By construction
/// (`planet-shard/src/main.rs::sync_celestial_bodies`) this is the
/// planet the shard is anchored on; its `position` is `≈ 0` after the
/// `planet_sys_pos − planet_pos.0` subtraction.
fn find_active_planet(
    primary: &PrimaryShard,
    secondaries: &Secondaries,
    primary_ws: &PrimaryWorldState,
    secondary_ws: &SecondaryWorldStates,
) -> Option<ActivePlanet> {
    if primary
        .current
        .as_ref()
        .is_some_and(|k| k.shard_type == PLANET_SHARD_TYPE)
    {
        if let Some(ws) = primary_ws.latest.as_ref() {
            if let Some(p) = active_planet_from_world_state(ws) {
                return Some(p);
            }
        }
    }
    for (key, _) in secondaries.runtimes.iter() {
        if key.shard_type != PLANET_SHARD_TYPE {
            continue;
        }
        if let Some((ws, _)) = secondary_ws.by_shard_type.get(&PLANET_SHARD_TYPE) {
            if let Some(p) = active_planet_from_world_state(ws) {
                return Some(p);
            }
        }
    }
    None
}

/// Pick the planet body from `ws` whose position is closest to that
/// shard's origin and which carries broadcast `planetary` data.
fn active_planet_from_world_state(ws: &WorldStateData) -> Option<ActivePlanet> {
    ws.bodies
        .iter()
        .filter(|b| b.body_id != 0)
        .filter_map(|b| b.planetary.as_ref().map(|p| (b, p)))
        .min_by(|(a, _), (b, _)| {
            a.position
                .length_squared()
                .partial_cmp(&b.position.length_squared())
                .unwrap_or(std::cmp::Ordering::Equal)
        })
        .map(|(b, p)| ActivePlanet {
            radius_m: b.radius,
            geophysics: *p,
        })
}

// ──────────────────────────────────────────────────────────────────────────
// ScatteringMedium construction from broadcast params
// ──────────────────────────────────────────────────────────────────────────

/// Build a Bevy `ScatteringMedium` from the broadcast geophysical state.
///
/// One [`ScatteringTerm`] per scattering mechanism — Rayleigh (gas),
/// Mie (aerosol), and Ozone (when the broadcast says present). Each
/// term's vertical profile uses the Bevy `Falloff::Exponential` /
/// `Tent` form normalised against `atmosphere_top_altitude_m` so the
/// shader can sample in `[0, 1]`-parameter space.
fn build_scattering_medium(g: &PlanetGeophysicalState) -> ScatteringMedium {
    let top_altitude = g.atmosphere_top_altitude_m.max(f32::EPSILON);
    let rayleigh = ScatteringTerm {
        scattering: Vec3::from_array(g.rayleigh_scattering_per_lambda),
        absorption: Vec3::ZERO,
        falloff: Falloff::Exponential {
            scale: g.rayleigh_scale_height_m / top_altitude,
        },
        phase: PhaseFunction::Rayleigh,
    };
    let mie = ScatteringTerm {
        scattering: Vec3::from_array(g.mie_scattering_per_lambda),
        absorption: Vec3::from_array(g.mie_absorption_per_lambda),
        falloff: Falloff::Exponential {
            scale: g.mie_scale_height_m / top_altitude,
        },
        phase: PhaseFunction::Mie {
            asymmetry: g.mie_phase_g,
        },
    };
    let mut terms = vec![rayleigh, mie];
    let ozone_max = g
        .ozone_absorption_per_lambda
        .iter()
        .copied()
        .fold(0.0, f32::max);
    if ozone_max > OZONE_PRESENCE_THRESHOLD {
        terms.push(ScatteringTerm {
            absorption: Vec3::from_array(g.ozone_absorption_per_lambda),
            scattering: Vec3::ZERO,
            falloff: Falloff::Tent {
                center: g.ozone_layer_centre_m / top_altitude,
                width: g.ozone_layer_width_m / top_altitude,
            },
            phase: PhaseFunction::Isotropic,
        });
    }
    ScatteringMedium::new(
        SCATTERING_MEDIUM_FALLOFF_RESOLUTION,
        SCATTERING_MEDIUM_PHASE_RESOLUTION,
        terms,
    )
}

/// Build the per-camera `AtmosphereSettings` from the active rendering
/// fidelity. The default LUT sizes work at any planet scale; the only
/// fidelity dimension we expose is the `rendering_method` choice.
fn build_atmosphere_settings(config: &GameConfig) -> AtmosphereSettings {
    let rendering_method = match config.graphics.lighting.atmosphere_method {
        AtmosphereMethod::LookupTexture => AtmosphereMode::LookupTexture,
        AtmosphereMethod::Raymarched => AtmosphereMode::Raymarched,
    };
    AtmosphereSettings {
        rendering_method,
        ..AtmosphereSettings::default()
    }
}

// ──────────────────────────────────────────────────────────────────────────
// Hashing the broadcast geophysics — used to detect when the medium needs
// to be rebuilt
// ──────────────────────────────────────────────────────────────────────────

/// Hash every f32 field by its bit-representation so byte-identical
/// broadcasts produce the same hash. Returns a `u64` for cheap comparison.
fn hash_geophysics(g: &PlanetGeophysicalState) -> u64 {
    let mut h = std::collections::hash_map::DefaultHasher::new();
    g.has_atmosphere.hash(&mut h);
    hash_f32_array(&[
        g.surface_gravity_ms2,
        g.equilibrium_temp_k,
        g.surface_pressure_pa,
        g.ground_albedo_linear_rgb[0],
        g.ground_albedo_linear_rgb[1],
        g.ground_albedo_linear_rgb[2],
        g.composition_n2,
        g.composition_o2,
        g.composition_co2,
        g.composition_ch4,
        g.composition_h2o,
        g.composition_h2,
        g.composition_he,
        g.composition_dust,
        g.scale_height_m,
        g.atmosphere_top_altitude_m,
        g.mean_molecular_mass_kg,
        g.atmospheric_mass_kg,
        g.rayleigh_scattering_per_lambda[0],
        g.rayleigh_scattering_per_lambda[1],
        g.rayleigh_scattering_per_lambda[2],
        g.rayleigh_scale_height_m,
        g.mie_scattering_per_lambda[0],
        g.mie_scattering_per_lambda[1],
        g.mie_scattering_per_lambda[2],
        g.mie_absorption_per_lambda[0],
        g.mie_absorption_per_lambda[1],
        g.mie_absorption_per_lambda[2],
        g.mie_scale_height_m,
        g.mie_phase_g,
        g.ozone_absorption_per_lambda[0],
        g.ozone_absorption_per_lambda[1],
        g.ozone_absorption_per_lambda[2],
        g.ozone_layer_centre_m,
        g.ozone_layer_width_m,
    ], &mut h);
    h.finish()
}

fn hash_f32_array<H: Hasher>(values: &[f32], h: &mut H) {
    for v in values {
        v.to_bits().hash(h);
    }
}
