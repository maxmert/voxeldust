#define_import_path bevy_pbr::atmosphere::types

struct Atmosphere {
    ground_albedo: vec3<f32>,
    // Radius of the planet
    bottom_radius: f32, // units: m
    // Radius at which we consider the atmosphere to 'end' for out calculations (from center of planet)
    top_radius: f32, // units: m
    // VOXELDUST PATCH (planet centre): the planet's centre in the camera's world, in metres. Upstream
    // fixes it at (0, -bottom_radius, 0) under the world origin; a spherical world states it.
    planet_center: vec3<f32>, // units: m
    // VOXELDUST PATCH (the air under its ground, 2026-09-30; `AtmosphereBelow`): under the bottom sphere
    // every term holds its value at the sphere and one term (the molecules) grows by its exponential law.
    below_scattering: vec3<f32>, // the growing term's scattering at the sphere, units: 1/m
    below_scale_height: f32, // its scale height, units: m
    below_extinction: vec3<f32>, // the whole medium's extinction at the sphere, units: 1/m
    below_floor_radius: f32, // no drawn ground stands lower; a pixel's sky ray ends here, units: m
}

struct AtmosphereSettings {
    transmittance_lut_size: vec2<u32>,
    multiscattering_lut_size: vec2<u32>,
    sky_view_lut_size: vec2<u32>,
    aerial_view_lut_size: vec3<u32>,
    transmittance_lut_samples: u32,
    multiscattering_lut_dirs: u32,
    multiscattering_lut_samples: u32,
    sky_view_lut_samples: u32,
    aerial_view_lut_samples: u32,
    aerial_view_lut_max_distance: f32,
    scene_units_to_m: f32,
    sky_max_samples: u32,
    rendering_method: u32,
    // VOXELDUST PATCH (the points pass): one where the sky pass adds the sun's disc on sky pixels; nought
    // where the points pass draws it after the reconstruction.
    sun_disk_on_sky: u32,
}

// "Atmosphere space" is just the view position with y=0 and oriented horizontally,
// so the horizon stays a horizontal line in our luts
struct AtmosphereTransforms {
    world_from_atmosphere: mat4x4<f32>,
}

struct AtmosphereData {
    atmosphere: Atmosphere,
    settings: AtmosphereSettings,
}