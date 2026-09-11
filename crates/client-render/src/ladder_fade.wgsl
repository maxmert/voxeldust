// THE CROSSFADE (the voxel foundation, slice 8 step 3; ruling V14 D8-2, §5) — the ground's own lit
// material, extended with the GEOMORPH that carries a rung onto the next coarser one across a band
// of distance, THE SINK that drops it under the next finer one, and THE FAR EDGE where it ends.
//
// Every vertex of a chunk at rung L carries ONE METRE (`morph_m`, step 5): how far along its own
// radial it stands from the rung L + 1 surface (computed by the client library from the same
// recipe when the chunk is built); the radial is the vertex's world position less the body's
// centre (the uniform), and the target is the vertex plus the radial times the metre. Across the
// band `(out_lo, out_hi)` around the rung's switch distance, the vertex slides from its own
// position to that target, on its own distance from the eye: whole at `out_lo`, on the coarser
// surface at `out_hi`. At `out_hi` the two surfaces coincide, so the finer ends there (the
// fragment stage discards it at and past `out_hi`) without a pop, and the coarser, present
// everywhere, carries on.
//
// THE SINK. Nearer than its own fade-in edge `in_hi` — which is the finer rung's `out_hi`, the same
// line — the chunk drops along each vertex's radial by the rung's sink (the uniform's `w`: the
// recipe's bound on the gap between the two fields plus a cell of each rung for the extractor's
// placement), from nothing at `in_hi` to the whole drop at `in_lo`. Under that drop the coarser mesh is certainly below the
// finer one, so it never shows through where the finer is whole, and where the finer is still
// building the sunk coarser shows instead of a hole (coarse before fine, SL8). Nothing is ever
// discarded on the near side: MEASURED before the sink, a partition by distance between two
// different meshes left one pixel of nothing on the line, where a grazing ray read the finer just
// past it and the coarser just before it; and before that, a coarse column drawn whole showed
// through the finer rung's territory wherever its surface stood over the finer (the dropped
// octaves cut both ways) — a tenth of the hill's pixels outside the bands on the wrong rung.
//
// Why a morph and not a dither. A dither gives every pixel to one of two rungs; MEASURED on the
// first crossfade flights, two surfaces that stand apart along a pixel's ray cannot share a
// weight (116 holes, all in bands), a weight read on the pixel's ray against a reference sphere
// is no distance where ground has relief (895 holes on the horizon rows), and with the coarser
// rung whole beneath a dithering finer one, a finer hill seen over a near crest dithers out onto
// a coarser surface that stands below the sightline (47 holes at 13–15 km). The morph makes the
// two surfaces the same before the switch; a dither between them then has nothing to hide.
//
// The bands are the Tier-A `vd_client::ladder_view::fade_bands`, handed in as a uniform; the
// hysteresis pair that sizes them lives there too. The probe's own material carries the same
// morph, sink and far edge (`probe.wgsl`), so what the probe reads is what the picture shows, and
// the shadow pass the same morph and sink (`ladder_fade_prepass.wgsl`), so the shadows fall from
// the surface the picture shows. Every distance is read from the render frame's ORIGIN, where the
// eye stands (the floating origin: every chunk is placed against the eye) — the shadow pass's view
// is the light's, so a distance read from the view would morph the shadow caster elsewhere.

#import bevy_pbr::{
    mesh_functions,
    forward_io::{VertexOutput, FragmentOutput},
    view_transformations::position_world_to_clip,
    pbr_fragment::pbr_input_from_standard_material,
    pbr_functions::{alpha_discard, apply_pbr_lighting, main_pass_post_lighting_processing},
}

struct LadderFade {
    // The bands: (in_lo, sink_end, out_lo, out_hi), metres from the eye; the sink ramp ends a
    // little past the fade-in edge, so at the edge one finer cell of sink remains.
    bands: vec4<f32>,
    // The rung's sink in metres (x), step 5.
    sink: vec4<f32>,
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
}

// How whole the rung is at a distance: one below its fade-out band, zero past it.
fn whole(d: f32) -> f32 {
    return clamp((fade.bands.w - d) / (fade.bands.w - fade.bands.z), 0.0, 1.0);
}

// How far the rung has risen from under the finer one: zero below its fade-in band, one where
// the sink ramp ends, a little past the fade-in edge.
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
    let coarser = own.xyz + radial * vertex.morph_m;
    let sink = radial * fade.sink.x;
    let morphed = mix(coarser, own.xyz, whole(d)) - sink * (1.0 - risen(d));
    out.world_position = vec4<f32>(morphed, 1.0);
    out.position = position_world_to_clip(out.world_position.xyz);
#ifdef OCT_NORMAL
    let local_normal = oct_decode(vertex.oct_normal);
#else
    let local_normal = vertex.normal;
#endif
    out.world_normal = mesh_functions::mesh_normal_local_to_world(local_normal, vertex.instance_index);
#ifdef VERTEX_OUTPUT_INSTANCE_INDEX
    out.instance_index = vertex.instance_index;
#endif
    return out;
}

// THE FAR EDGE: the rung ends where it has become the coarser one. The distance is the morphed
// fragment's own; the vertex stage read the unmorphed vertex's. The two differ by the morph's
// displacement, so the fringe ends a little before the surfaces coincide — harmless, because the
// coarser rung is never discarded on its near side and stands there.
@fragment
fn fragment(in: VertexOutput, @builtin(front_facing) is_front: bool) -> FragmentOutput {
    if length(in.world_position.xyz) >= fade.bands.w {
        discard;
    }
    var pbr_input = pbr_input_from_standard_material(in, is_front);
    pbr_input.material.base_color = alpha_discard(pbr_input.material, pbr_input.material.base_color);
    var out: FragmentOutput;
    out.color = apply_pbr_lighting(pbr_input);
    out.color = main_pass_post_lighting_processing(pbr_input, out.color);
    return out;
}
