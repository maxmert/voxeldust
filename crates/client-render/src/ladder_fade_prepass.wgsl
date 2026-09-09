// THE CROSSFADE'S PREPASS (slice 8 step 3): the ground's vertex stage for the shadow map and every
// other prepass — THE SAME GEOMORPH AND SINK as `ladder_fade.wgsl`, so the shadows fall from the
// surface the picture shows — and its fragment stage with THE SAME FAR EDGE, so past its edge a
// rung casts no shadow either (two coincident casters would fight in the shadow map). The base
// material is alpha-masked so the engine runs this fragment stage in the shadow pass.
//
// MEASURED before the prepass stage: the shadow pass ran the engine's own vertex stage on the
// unmorphed mesh, and at every fade-out edge the unmorphed finer rung stood over the
// morphed surface and shadowed it — 34 dark specks on the hill picture at 950 m, 1.9 km and
// 3.8 km, exactly the three edges inside the shadow's reach.
//
// The distance that drives the morph is read from the render frame's ORIGIN, where the eye stands
// (the floating origin: every chunk is placed against the eye), because this stage also runs in
// the shadow pass, whose view is the light's. The main stages read the same origin, so the three
// agree by construction.

#import bevy_pbr::{
    mesh_functions,
    prepass_io::{VertexOutput, FragmentOutput},
    view_transformations::position_world_to_clip,
}

struct LadderFade {
    // The bands: (in_lo, sink_end, out_lo, out_hi), metres from the eye.
    bands: vec4<f32>,
}

@group(#{MATERIAL_BIND_GROUP}) @binding(100) var<uniform> fade: LadderFade;

struct FadeVertex {
    @builtin(instance_index) instance_index: u32,
    @location(0) position: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @location(8) morph: vec3<f32>,
    @location(9) sink: vec3<f32>,
}

fn whole(d: f32) -> f32 {
    return clamp((fade.bands.w - d) / (fade.bands.w - fade.bands.z), 0.0, 1.0);
}

fn risen(d: f32) -> f32 {
    return clamp((d - fade.bands.x) / (fade.bands.y - fade.bands.x), 0.0, 1.0);
}

@vertex
fn vertex(vertex: FadeVertex) -> VertexOutput {
    var out: VertexOutput;
    let world_from_local = mesh_functions::get_world_from_local(vertex.instance_index);
    let own = mesh_functions::mesh_position_local_to_world(world_from_local, vec4<f32>(vertex.position, 1.0));
    let d = length(own.xyz);
    let local = mix(vertex.morph, vertex.position, whole(d)) - vertex.sink * (1.0 - risen(d));
    out.world_position = mesh_functions::mesh_position_local_to_world(world_from_local, vec4<f32>(local, 1.0));
    out.position = position_world_to_clip(out.world_position.xyz);
#ifdef UNCLIPPED_DEPTH_ORTHO_EMULATION
    out.unclipped_depth = out.position.z;
    out.position.z = min(out.position.z, 1.0);
#endif
#ifdef NORMAL_PREPASS_OR_DEFERRED_PREPASS
    out.world_normal = mesh_functions::mesh_normal_local_to_world(vertex.normal, vertex.instance_index);
#endif
#ifdef MOTION_VECTOR_PREPASS
    // The same morphed position under last frame's placement: the ground itself does not move.
    let previous_from_local = mesh_functions::get_previous_world_from_local(vertex.instance_index);
    out.previous_world_position = mesh_functions::mesh_position_local_to_world(previous_from_local, vec4<f32>(local, 1.0));
#endif
#ifdef VERTEX_OUTPUT_INSTANCE_INDEX
    out.instance_index = vertex.instance_index;
#endif
    return out;
}

// THE FAR EDGE in every prepass: with a target (a normal, a motion vector, an emulated depth) the
// stage returns it; with none — the shadow map, depth only — it returns nothing, as the engine's
// own alpha-mask prepass does, and only the discard matters.
#ifdef PREPASS_FRAGMENT
@fragment
fn fragment(in: VertexOutput) -> FragmentOutput {
    if length(in.world_position.xyz) >= fade.bands.w {
        discard;
    }
    var out: FragmentOutput;
#ifdef NORMAL_PREPASS
    out.normal = vec4<f32>(in.world_normal * 0.5 + vec3<f32>(0.5), 1.0);
#endif
#ifdef UNCLIPPED_DEPTH_ORTHO_EMULATION
    out.frag_depth = in.unclipped_depth;
#endif
    return out;
}
#else
@fragment
fn fragment(in: VertexOutput) {
    if length(in.world_position.xyz) >= fade.bands.w {
        discard;
    }
}
#endif
