//! THE SKY ON SCREEN (slice 8s, `docs/investigation/2026-09-08/landforms/slice_8s_design.md`).
//!
//! The engine's atmosphere (Hillaire 2020 as Bevy 0.18 ships it, with the planet-centre patch in
//! `vendor/bevy_pbr`) draws ONE body's air a frame as a volume around a sphere: the sky dome, the
//! aerial perspective on every ground pixel, the sun's transmittance on the ground, the sun disc,
//! and the dome as the light that fills the shadows. This module WIRES it and computes nothing
//! physical: every number comes from `vd_client::sky` (the laws over the charter's words).
//!
//! Which body (design §7): among the bodies with air in the window, the one whose shell subtends
//! the largest angle at the eye — `top_radius / distance`. A body without air is never a
//! candidate; no candidate ⇒ no atmosphere on the camera, and the picture is what it was.
//!
//! Example: a pilot standing on an airless moon sees its planet's blue rim in the sky, because the
//! planet's shell is the largest in view; the ray to the moon's own ground crosses none of that
//! air, so the moon's horizon stays sharp and black-skied. No code names the moon or the planet.

use std::collections::BTreeMap;

use bevy::light::{AtmosphereEnvironmentMapLight, SunDisk};
use bevy::pbr::{
    Atmosphere, AtmosphereMode, AtmosphereSettings, Falloff, PhaseFunction, ScatteringMedium,
    ScatteringTerm,
};
use bevy::prelude::*;
use vd_client::sky as laws;
use vd_core::glam::DVec3;
use vd_core::look::BodyCharter;
use vd_core::pose::RealmId;

use crate::terrain::{Terrain, env_or};

/// `VD_SKY=0` draws no atmosphere and no HDR (the picture as before 8s: the airless control).
const SKY_ENV: &str = "VD_SKY";
/// `VD_SKY_MODE=lut|ray`: the engine's lookup mode or its raymarched mode (design §6 measures).
const SKY_MODE_ENV: &str = "VD_SKY_MODE";
/// `VD_SKY_ENV_PX`: the sky-light environment map's edge, pixels.
const SKY_ENV_PX_ENV: &str = "VD_SKY_ENV_PX";
/// ★ `VD_SKY_FLAT=1`: THE CONTROL (design §3.3, G-SKY-CONTROL) — the planet centre as upstream Bevy
/// fixes it, `(0, −R, 0)` under the world origin, so the unpatched reading can be measured against
/// the patched one from the same stand. A measurement knob, never a mode anybody flies.
const SKY_FLAT_ENV: &str = "VD_SKY_FLAT";

/// What the flags said at startup: the operational knobs, in ONE struct (no inline literals).
#[derive(Resource, Clone, Copy, Debug, PartialEq, Eq)]
pub struct SkyConfig {
    pub enabled: bool,
    pub raymarched: bool,
    pub env_map_px: u32,
    pub flat: bool,
}

impl SkyConfig {
    #[must_use]
    pub fn from_env() -> SkyConfig {
        let mode: String = env_or(SKY_MODE_ENV, "ray".to_owned());
        SkyConfig {
            enabled: env_or(SKY_ENV, 1u8) != 0,
            raymarched: mode != "lut",
            // The engine's own default edge, read rather than typed.
            env_map_px: env_or(
                SKY_ENV_PX_ENV,
                AtmosphereEnvironmentMapLight::default().size.x,
            ),
            flat: env_or(SKY_FLAT_ENV, 0u8) != 0,
        }
    }
}

/// One medium asset per body (a charter is fixed for a realm's life).
#[derive(Resource, Default)]
pub struct SkyMedia {
    by_realm: BTreeMap<RealmId, Handle<ScatteringMedium>>,
}

/// A body in the window with a charter, as `sync_terrain` lists it every frame: its centre in the
/// render frame (metres, the eye at the origin), its ground radius and its charter.
#[derive(Clone, Debug)]
pub(crate) struct SkyBody {
    pub realm: RealmId,
    pub centre: DVec3,
    pub radius_m: f64,
    pub charter: BodyCharter,
}

/// The engine's term from the law's term: a narrowing to `f32` and nothing else.
fn term_of(t: &laws::Term) -> ScatteringTerm {
    let narrow = |v: [f64; 3]| Vec3::new(v[0] as f32, v[1] as f32, v[2] as f32);
    ScatteringTerm {
        absorption: narrow(t.absorption_per_m),
        scattering: narrow(t.scattering_per_m),
        falloff: match t.falloff {
            laws::Falloff::Exponential { scale_share } => Falloff::Exponential {
                scale: scale_share as f32,
            },
            laws::Falloff::Tent { center, width } => Falloff::Tent {
                center: center as f32,
                width: width as f32,
            },
        },
        phase: match t.phase {
            laws::Phase::Rayleigh => PhaseFunction::Rayleigh,
            laws::Phase::Mie { asymmetry } => PhaseFunction::Mie {
                asymmetry: asymmetry as f32,
            },
            laws::Phase::Isotropic => PhaseFunction::Isotropic,
        },
    }
}

/// A body's medium: its Rayleigh, its aerosol and (where the body is earth-like) its ozone, at
/// the engine's default table resolutions.
fn medium_of(terms: &laws::SkyTerms, realm: RealmId) -> ScatteringMedium {
    let defaults = ScatteringMedium::default();
    let mut list = vec![term_of(&terms.rayleigh), term_of(&terms.mie)];
    if let Some(ozone) = terms.ozone.as_ref() {
        list.push(term_of(ozone));
    }
    ScatteringMedium::new(defaults.falloff_resolution, defaults.phase_resolution, list)
        .with_label(format!("sky of {realm:?}"))
}

/// The pick (design §7): the body with air whose shell subtends the largest angle at the eye.
fn pick(bodies: &[SkyBody]) -> Option<(&SkyBody, laws::SkyTerms)> {
    bodies
        .iter()
        .filter_map(|b| laws::sky_terms(&b.charter, b.radius_m).map(|t| (b, t)))
        .max_by(|(a, ta), (b, tb)| {
            laws::shell_angle(ta.top_radius_m, a.centre.length())
                .total_cmp(&laws::shell_angle(tb.top_radius_m, b.centre.length()))
        })
}

/// The picture's cameras with whatever air they hold: the atmosphere and its settings, if any.
type SkyCameras<'w, 's> = Query<
    'w,
    's,
    (
        Entity,
        Option<&'static mut Atmosphere>,
        Option<&'static mut AtmosphereSettings>,
    ),
    With<super::FollowCam>,
>;

/// THE SKY SYSTEM: after the terrain listed the bodies and placed the sun, put the chosen body's
/// air on the picture's camera (or take it off), move its centre with the eye, size the sun disc,
/// and state it all on the stamp.
pub(crate) fn sync_sky(
    config: Res<SkyConfig>,
    mut terrain: ResMut<Terrain>,
    mut media: ResMut<SkyMedia>,
    mut assets: ResMut<Assets<ScatteringMedium>>,
    mut cams: SkyCameras,
    mut commands: Commands,
) {
    if !config.enabled {
        return;
    }
    let chosen = pick(&terrain.sky_bodies).map(|(b, t)| (b.clone(), t));
    let Some((body, terms)) = chosen else {
        for (cam, present, _) in &mut cams {
            if present.is_some() {
                commands.entity(cam).remove::<(
                    Atmosphere,
                    AtmosphereSettings,
                    AtmosphereEnvironmentMapLight,
                )>();
            }
        }
        if let (Some(sun), Some(_)) = (terrain.sun, terrain.sun_disk.take()) {
            commands.entity(sun).remove::<SunDisk>();
        }
        return;
    };
    let bottom = terms.bottom_radius_m as f32;
    let planet_center = if config.flat {
        Vec3::new(0.0, -bottom, 0.0)
    } else {
        Vec3::new(
            body.centre.x as f32,
            body.centre.y as f32,
            body.centre.z as f32,
        )
    };
    let medium = media
        .by_realm
        .entry(body.realm)
        .or_insert_with(|| assets.add(medium_of(&terms, body.realm)))
        .clone();
    let same_body = terrain.sky_realm == Some(body.realm);
    // ★ THE LOOKUP MODE'S REACH (design §6, S5): the engine's table of the air in front of the
    // ground covers a fixed distance (32 km by default) and repeats its last value beyond it, so
    // it is set PER FRAME to the drawn radius — the table reaches as far as the drawn ground. The
    // stamp states it; the measurement judges the mode.
    let lut_far_m = terrain.stamp.as_ref().map_or(
        AtmosphereSettings::default().aerial_view_lut_max_distance,
        |s| s.drawn_radius_m as f32,
    );
    for (cam, present, settings) in &mut cams {
        match (present, settings) {
            // The same body's air: only its centre and the table's reach move with the eye.
            (Some(mut atmosphere), Some(mut settings)) if same_body => {
                atmosphere.planet_center = planet_center;
                settings.aerial_view_lut_max_distance = lut_far_m;
            }
            _ => {
                commands.entity(cam).insert((
                    Atmosphere {
                        bottom_radius: bottom,
                        top_radius: terms.top_radius_m as f32,
                        ground_albedo: Vec3::splat(terms.ground_albedo as f32),
                        medium: medium.clone(),
                        planet_center,
                    },
                    AtmosphereSettings {
                        rendering_method: if config.raymarched {
                            AtmosphereMode::Raymarched
                        } else {
                            AtmosphereMode::LookupTexture
                        },
                        aerial_view_lut_max_distance: lut_far_m,
                        ..default()
                    },
                    AtmosphereEnvironmentMapLight {
                        size: UVec2::splat(config.env_map_px),
                        ..default()
                    },
                ));
            }
        }
    }
    terrain.sky_realm = Some(body.realm);
    // The sun disc: the star's luminosity over the body's insolation, on the sun the terrain placed.
    let disk = terrain
        .sun_luma
        .and_then(|luma| laws::sun_disk_angle_rad(luma, body.charter.insolation_q12))
        .map(|a| a as f32);
    if let Some(sun) = terrain.sun
        && disk != terrain.sun_disk
    {
        match disk {
            Some(angular_size) => {
                commands.entity(sun).insert(SunDisk {
                    angular_size,
                    intensity: 1.0,
                });
            }
            None => {
                commands.entity(sun).remove::<SunDisk>();
            }
        }
        terrain.sun_disk = disk;
    }
    let eye_r_m = body.centre.length();
    if let Some(stamp) = terrain.stamp.as_mut() {
        stamp.sky = Some(vd_devproto::state::DevSkyStamp {
            realm: format!("{:?}", body.realm),
            bottom_m: terms.bottom_radius_m,
            top_m: terms.top_radius_m,
            eye_r_m,
            raymarched: config.raymarched,
            lut_far_m: f64::from(lut_far_m),
            flat: config.flat,
            sun_disk_deg: disk.map_or(0.0, |a| f64::from(a).to_degrees()),
            rayleigh_ratio: terms.rayleigh_550_per_m / laws::EARTH_RAYLEIGH_550_PER_M,
            ozone: terms.ozone.is_some(),
        });
    }
}
