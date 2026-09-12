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
    mesh_view_bindings::view,
    prepass_io::{VertexOutput, FragmentOutput},
    view_transformations::position_world_to_clip,
}

struct LadderFade {
    // The bands: (in_lo, sink_end, out_lo, out_hi), metres from the eye; the sink ramp ends a
    // little past the fade-in edge, so at the edge one finer cell of sink remains.
    bands: vec4<f32>,
    // The rung's sink in metres (x), step 5.
    sink: vec4<f32>,
    // The rung's cell in metres (x): a splat's width (D8-8's measurement).
    splat: vec4<f32>,
}

@group(#{MATERIAL_BIND_GROUP}) @binding(100) var<uniform> fade: LadderFade;

struct FadeVertex {
    @builtin(instance_index) instance_index: u32,
    @location(0) position: vec3<f32>,
#ifdef OCT_NORMAL
    @location(1) oct_normal: vec2<f32>,
#else
    @location(1) normal: vec3<f32>,
#endif
    // THE MORPH METRE (step 5): how far along its own radial the vertex stands from the next
    // coarser rung's surface; the target is the vertex plus its radial times this.
    @location(8) morph_m: f32,
    // THE RADIAL (step 5): the vertex's unit direction from the body's centre in the chunk's
    // frame.
    @location(9) radial: vec3<f32>,
    // THE MORPH NORMAL (item 20): the shade the next coarser rung draws at this vertex's target,
    // blended with the own normal across the fade-out band as the position is.
    @location(11) morph_normal: vec2<f32>,
#ifdef SPLAT
    // THE SPLAT CORNER (D8-8's measurement): which corner of the camera-facing square this copy
    // of the vertex is.
    @location(10) corner: vec2<f32>,
#endif
}

fn whole(d: f32) -> f32 {
    return clamp((fade.bands.w - d) / (fade.bands.w - fade.bands.z), 0.0, 1.0);
}

fn risen(d: f32) -> f32 {
    return clamp((d - fade.bands.x) / (fade.bands.y - fade.bands.x), 0.0, 1.0);
}

// THE PACKED NORMAL (ruling V18): two signed 16-bit numbers on the octahedron, unfolded to the
// unit normal — the library's `oct_decode`, step for step.
fn oct_decode(e: vec2<f32>) -> vec3<f32> {
    var n = vec3<f32>(e.x, e.y, 1.0 - abs(e.x) - abs(e.y));
    let t = clamp(-n.z, 0.0, 1.0);
    n.x = n.x + select(t, -t, n.x >= 0.0);
    n.y = n.y + select(t, -t, n.y >= 0.0);
    return normalize(n);
}

@vertex
fn vertex(vertex: FadeVertex) -> VertexOutput {
    var out: VertexOutput;
    let world_from_local = mesh_functions::get_world_from_local(vertex.instance_index);
    let own = mesh_functions::mesh_position_local_to_world(world_from_local, vec4<f32>(vertex.position, 1.0));
    let d = length(own.xyz);
    // The vertex's radial (its own attribute, turned into the world like a normal), its target
    // along it, and its sink along it — the morph in world space (step 5: one metre and one
    // radial per vertex, the sink one number per rung; no centre of the body, so no material
    // is rewritten as the eye moves).
    let radial = normalize(mesh_functions::mesh_normal_local_to_world(vertex.radial, vertex.instance_index));
#ifdef LIGHT_CASTER
    // THE LIGHT CASTER (the shadow ladder): the caster skips the morph and stands under the drawn
    // ground by the caster's own sink (the two rungs' bound), so the fine ground never shades
    // itself against a coarse surface that stands above it. MEASURED alone (round five): the work
    // per vertex is not the shadow's cost; the caster count is, which the ladder's coarse casters
    // cut. THE CASTER'S CROSSFADE: the bands here are the DRAWN rung's (the rung this caster casts
    // for), so the sink scales with that rung's wholeness — full where the finer chunk stands whole,
    // zero at the band's end where it has morphed onto this surface and the caster gives way to the
    // drawn chunk casting itself, unsunk: the shadow's edge moves with the crossfade, never at it.
    let morphed = own.xyz - radial * fade.splat.y * whole(d);
#else
    let coarser = own.xyz + radial * vertex.morph_m;
    let sink = radial * fade.sink.x;
    let morphed = mix(coarser, own.xyz, whole(d)) - sink * (1.0 - risen(d));
#endif
#ifdef SPLAT
    // THE SPLAT: the vertex as a camera-facing square one cell wide, spread in view space after
    // the morph and the sink, so it faces the eye by construction.
    let view_pos = view.view_from_world * vec4<f32>(morphed, 1.0);
    // Two cells wide (MEASURED at one cell: 3 183 crack pixels between splats on the aloft
    // stand — surface-nets vertices stand up to 1.7 cells apart on a diagonal).
    let splat_pos = view_pos.xyz + vec3<f32>(vertex.corner * fade.splat.x, 0.0);
    out.world_position = view.world_from_view * vec4<f32>(splat_pos, 1.0);
    out.position = view.clip_from_view * vec4<f32>(splat_pos, 1.0);
#else
    out.world_position = vec4<f32>(morphed, 1.0);
    out.position = position_world_to_clip(out.world_position.xyz);
#endif
#ifdef UNCLIPPED_DEPTH_ORTHO_EMULATION
    out.unclipped_depth = out.position.z;
    out.position.z = min(out.position.z, 1.0);
#endif
#ifdef NORMAL_PREPASS_OR_DEFERRED_PREPASS
#ifdef OCT_NORMAL
    let own_normal = oct_decode(vertex.oct_normal);
#else
    let own_normal = vertex.normal;
#endif
    // THE SHADE HANDS OVER WITH THE SHAPE (item 20): the same blend as the position's.
    let local_normal = normalize(mix(oct_decode(vertex.morph_normal), own_normal, whole(d)));
    out.world_normal = mesh_functions::mesh_normal_local_to_world(local_normal, vertex.instance_index);
#endif
#ifdef MOTION_VECTOR_PREPASS
    // The same morph under last frame's placement: the ground itself does not move, so last
    // frame's own position carries this frame's displacement.
    let previous_from_local = mesh_functions::get_previous_world_from_local(vertex.instance_index);
    let previous_own = mesh_functions::mesh_position_local_to_world(previous_from_local, vec4<f32>(vertex.position, 1.0));
    out.previous_world_position = vec4<f32>(previous_own.xyz + (morphed - own.xyz), 1.0);
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
