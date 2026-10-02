#import bevy_pbr::{
    atmosphere::{
        bindings::settings,
        functions::{
            get_view_position, raymarch_atmosphere, RAY_TABLE_SKY,
            max_atmosphere_distance, direction_atmosphere_to_world,
            sky_view_lut_uv_to_zenith_azimuth, zenith_azimuth_to_ray_dir,
        },
    }
}

#import bevy_render::{
    view::View,
    maths::HALF_PI,
}
#import bevy_core_pipeline::fullscreen_vertex_shader::FullscreenVertexOutput

@group(0) @binding(13) var sky_view_lut_out: texture_storage_2d<rgba16float, write>;

@compute
@workgroup_size(16, 16, 1)
fn main(@builtin(global_invocation_id) idx: vec3<u32>) {
    let uv = vec2<f32>(idx.xy) / vec2<f32>(settings.sky_view_lut_size);

    let cam_pos = get_view_position();
    let r = length(cam_pos);
    var zenith_azimuth = sky_view_lut_uv_to_zenith_azimuth(r, uv);

    let ray_dir_as = zenith_azimuth_to_ray_dir(zenith_azimuth.x, zenith_azimuth.y);
    let ray_dir_ws = direction_atmosphere_to_world(ray_dir_as);

    // VOXELDUST PATCH (the table's eye, 2026-10-02): the march starts at the eye's own place from the
    // planet's centre in the WORLD's axes, the axes the ray's direction is in. Upstream put it at
    // (0, r, 0), the eye in the atmosphere's axes, beside a world direction — which agrees only while the
    // world's +Y is the eye's up (upstream's flat world with its centre under the origin). With the
    // world's +Y anywhere else the table read a world component as the zenith and marched a world ray
    // from an atmosphere point. MEASURED on the home planet's own picture (its world +Y the polar axis,
    // the sun at -0.999 along it): the environment map built from the table gave the ground no sky light.
    let world_pos = cam_pos;
    let up = normalize(world_pos);
    let mu = dot(ray_dir_ws, up);
    let t_max = max_atmosphere_distance(r, mu);

    let result = raymarch_atmosphere(world_pos, ray_dir_ws, t_max, settings.sky_view_lut_samples, uv, RAY_TABLE_SKY);

    textureStore(sky_view_lut_out, idx.xy, vec4(result.inscattering, 1.0));
}
