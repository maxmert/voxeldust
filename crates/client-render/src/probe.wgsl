// THE PROBE (the voxel foundation, slice 8p; ruling V14 D8-7) — a material that writes a CODE
// instead of a colour, into the second picture every judged capture carries. Each pixel says WHAT
// drew it and HOW FAR it is:
//
//   R byte:  kind << 5 | rung        (the uniform's `code`, built by `vd_client_harness::probe::probe_byte`)
//   G, B:    the distance from the eye in CELLS of the rung, big-endian u16
//   A:       two marks as bits on 255 (the temporal path's T1 item 7,
//            `vd_client_harness::probe::probe_alpha`): 128 clear where the surface MOVES against its
//            body in the frame, 64 clear where its chunk is NEW in the frame (landed or rebuilt)
//
// ★ THE BYTES ARE EXACT THROUGH AN sRGB TARGET. The picture's target is 8-bit sRGB, and the engine
// encodes a linear fragment value on store and decodes it on the blit to the output, then encodes it
// again: byte = round(255 · enc(dec(round(255 · enc(v)) / 255))). A fragment that emits the linear
// value `dec(b / 255)` therefore stores exactly `b` — the same round trip `Color::srgb` relies on for
// every marker gate. The probe camera runs no tonemapping, no dither and no multisampling, so no
// stage between this fragment and the byte mixes two values (MEASURED by the picture gate: every
// terrain pixel's rung byte must equal the rung the flag named).
//
// The vertex stage carries THE SAME GEOMORPH AND SINK as the ground's material (`ladder_fade.wgsl`)
// and the fragment stage THE SAME FAR EDGE, so the probe reads exactly the surface the picture
// shows.
//
// The render frame's eye stands at the origin (`camera_transform`; the floating origin, every chunk
// placed against the eye), so a world position's length IS its distance from the eye — the same
// reading the ground's material and its shadow pass make.

#import bevy_pbr::{
    mesh_functions,
    mesh_view_bindings::view,
    view_transformations::position_world_to_clip,
}

struct ProbeUniform {
    // kind << 5 | rung, as a float 0..255.
    code: f32,
    // One cell of the rung, in metres.
    cell_m: f32,
    // The rung's sink in metres (x), step 5; (z) this probe's row of the band table (the
    // always-passed row for the ruler).
    sink: vec4<f32>,
}

@group(#{MATERIAL_BIND_GROUP}) @binding(0) var<uniform> probe: ProbeUniform;

// ★ THE REALM'S BAND TABLE (slice R1a): the same buffer the ground's material reads, so the probe
// fades on the band the picture fades on. One layout with `ladder_fade.wgsl` (its comment says what
// each part holds); the probe reads this frame's rows alone.
struct BandTable {
    clock: vec4<u32>,
    step: array<vec4<f32>, 3>,
    // ★ THE REALM'S FRAME (ruling B89): the body's centre in this frame's render space (xyz), and its spin axis there
    // (xyz) — a vertex's radial is its own place less the centre, so no vertex carries one.
    centre: vec4<f32>,
    pole: vec4<f32>,
    rows: array<vec4<f32>>,
}

@group(#{MATERIAL_BIND_GROUP}) @binding(1) var<storage, read> table: BandTable;

// This probe's row of the table.
fn bands() -> vec4<f32> {
    return table.rows[u32(probe.sink.z)];
}

// This probe's row of LAST frame's table (T1): last frame's rows follow this frame's.
fn last_bands() -> vec4<f32> {
    return table.rows[table.clock.w + u32(probe.sink.z)];
}

// THE REALM'S STEP (T1; `vd_client::motion::RealmStep`): where a point drawn at `y` this frame stood
// last frame.
fn step_point(y: vec3<f32>) -> vec3<f32> {
    return vec3<f32>(
        dot(table.step[0].xyz, y) + table.step[0].w,
        dot(table.step[1].xyz, y) + table.step[1].w,
        dot(table.step[2].xyz, y) + table.step[2].w,
    );
}

// THE TAG's parts (`vd_client::arrival`): the clock of the frame the chunk appeared in, 28 bits; the
// flag of a chunk that rises from its stand-in (without it, whole); ★ the own-height stand-in's three
// flags (2026-09-30): the chunk's stand-in stands at its own height, the chunk is drawn as a stand-in,
// and the stand-in is covered (it sinks over the ramp from its clock).
const RAMP_FLAG: u32 = 0x80000000u;
const OWN_HEIGHT_FLAG: u32 = 0x40000000u;
const STAND_IN_FLAG: u32 = 0x20000000u;
const COVERED_FLAG: u32 = 0x10000000u;
const CLOCK_MASK: u32 = 0x0fffffffu;

// The tag's age at this frame's clock over the ramp (`vd_client::arrival::ramp_age`), wrapping.
fn age_now(tag: u32) -> f32 {
    let age = f32((table.clock.x - (tag & CLOCK_MASK)) & CLOCK_MASK) / f32(max(table.clock.y, 1u));
    return clamp(age, 0.0, 1.0);
}

// The tag's age at LAST frame's clock, read SIGNED (T1; `vd_client::motion::previous_age`).
fn age_last(tag: u32) -> f32 {
    let diff = (table.clock.z - (tag & CLOCK_MASK)) & CLOCK_MASK;
    let age = f32(bitcast<i32>(diff << 4u) >> 4u) / f32(max(table.clock.y, 1u));
    return clamp(age, 0.0, 1.0);
}

// ★ THE ARRIVAL RAMP'S WEIGHT (the far-water design §10, slice R1c; `vd_client::arrival::ramp_weight`,
// step for step): nought in the frame a chunk landed, one half a second later, and one at once for a
// chunk that entered whole (its tag without the ramp's flag).
fn arrival(instance_index: u32) -> f32 {
    let tag = mesh_functions::get_tag(instance_index);
    return select(1.0, age_now(tag), (tag & RAMP_FLAG) != 0u);
}

// THE RAMP'S WEIGHT AT LAST FRAME'S CLOCK (T1; `vd_client::motion::previous_ramp_weight`): a chunk that
// landed after last frame's clock reads nought, where it stood.
fn last_arrival(instance_index: u32) -> f32 {
    let tag = mesh_functions::get_tag(instance_index);
    return select(1.0, age_last(tag), (tag & RAMP_FLAG) != 0u);
}

// ★ THE SINK'S WEIGHT (`vd_client::arrival::sink_weight`, step for step), at an age.
fn sink_weight_of(tag: u32, age: f32) -> f32 {
    let covered = select(0.0, age, (tag & COVERED_FLAG) != 0u);
    return select(1.0, covered, (tag & STAND_IN_FLAG) != 0u);
}

// ★ THE SHARE OF THE STAND-IN'S SINK a ramp starts under (`vd_client::arrival::start_sink_share`).
fn start_share(tag: u32) -> f32 {
    return select(1.0, 0.0, (tag & OWN_HEIGHT_FLAG) != 0u);
}

// THE FOURTH BYTE'S BITS (`vd_client_harness::probe::PROBE_BIT_STILL` and `PROBE_BIT_OLD`): each is set
// on 255 and cleared by its mark.
const BIT_STILL: f32 = 128.0;
const BIT_OLD: f32 = 64.0;

// The probe's stage output: the drawn place (for the far edge and the distance) and the two marks.
struct ProbeOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) world_position: vec4<f32>,
    @location(1) moving: f32,
    @location(2) fresh: f32,
}

// Whether a crossfade's weight stands strictly inside its band.
fn inside(t: f32) -> bool {
    return (t > 0.0) && (t < 1.0);
}

// The ground's vertex layout, whole (`ground_vertex_layout`); the probe draws no shade, so it reads no
// normal.
struct ProbeVertex {
    @builtin(instance_index) instance_index: u32,
    @location(0) position: vec3<f32>,
#ifdef OCT_NORMAL
    @location(1) oct_normal: vec2<f32>,
#else
    @location(1) normal: vec3<f32>,
#endif
    // THE MORPH METRE (step 5): the vertex's distance along its own radial to the next coarser
    // rung's surface.
    @location(8) morph_m: f32,
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
    return whole_in(bands(), d);
}

fn risen(d: f32) -> f32 {
    return risen_in(bands(), d);
}

// The wholeness and the rise under a given band row (this frame's or last frame's).
fn whole_in(b: vec4<f32>, d: f32) -> f32 {
    return clamp((b.w - d) / (b.w - b.z), 0.0, 1.0);
}

fn risen_in(b: vec4<f32>, d: f32) -> f32 {
    return clamp((d - b.x) / (b.y - b.x), 0.0, 1.0);
}

@vertex
fn vertex(vertex: ProbeVertex) -> ProbeOutput {
    var out: ProbeOutput;
    let world_from_local = mesh_functions::get_world_from_local(vertex.instance_index);
    let own = mesh_functions::mesh_position_local_to_world(world_from_local, vec4<f32>(vertex.position, 1.0));
    let d = length(own.xyz);
    // The vertex's radial — ★ its own place less the body's centre in this frame's render space (ruling B89: the realm's
    // band table carries the centre, rewritten every frame, so no vertex carries a radial and no material is rewritten
    // as the eye moves) — its target along it, and its sink along it: the morph in world space (step 5: one metre per
    // vertex, the sink one number per rung).
    let radial = normalize(own.xyz - table.centre.xyz);
    let coarser = own.xyz + radial * vertex.morph_m;
    let sink = radial * probe.sink.x;
    // ★ THE ARRIVAL RAMP (slice R1c) and THE OWN-HEIGHT STAND-IN: the twin rises and sinks as the
    // picture's chunk does.
    let tag = mesh_functions::get_tag(vertex.instance_index);
    let ramp = arrival(vertex.instance_index);
    let weight = sink_weight_of(tag, age_now(tag));
    let morphed = mix(
        coarser - radial * probe.sink.y * whole(d) * start_share(tag),
        mix(coarser, own.xyz, whole(d)) - sink * (1.0 - risen(d)) * weight,
        ramp,
    );
#ifdef SPLAT
    // THE SPLAT: the vertex as a camera-facing square one cell wide, spread in view space after
    // the morph and the sink, so it faces the eye by construction.
    let view_pos = view.view_from_world * vec4<f32>(morphed, 1.0);
    // Two cells wide (MEASURED at one cell: 3 183 crack pixels between splats on the aloft
    // stand — surface-nets vertices stand up to 1.7 cells apart on a diagonal).
    let splat_pos = view_pos.xyz + vec3<f32>(vertex.corner * probe.cell_m, 0.0);
    out.world_position = view.world_from_view * vec4<f32>(splat_pos, 1.0);
    out.position = view.clip_from_view * vec4<f32>(splat_pos, 1.0);
#else
    out.world_position = vec4<f32>(morphed, 1.0);
    out.position = position_world_to_clip(out.world_position.xyz);
#endif
    // ★ THE MOVING MARK (the temporal path's T1 item 7): the surface moves against its body this
    // frame when the vertex stands inside a band's morph or sink in this frame or the last (at last
    // frame's row and at its distance then, carried by the realm's step), or when its ramp weight
    // changed since last frame. The motion truth gate judges only the ground that holds still
    // against its body; the moving ground is judged by the Tier-A rule (`vd_client::motion`).
    let last_d = length(step_point(own.xyz));
    let last_b = last_bands();
    // A covered stand-in sinking under its finer ground moves as its ramp does.
    let moving = inside(whole(d)) || inside(risen(d)) || inside(whole_in(last_b, last_d))
        || inside(risen_in(last_b, last_d)) || (ramp != last_arrival(vertex.instance_index))
        || (weight != sink_weight_of(tag, age_last(tag)));
    out.moving = select(0.0, 1.0, moving);
    // ★ THE NEW MARK: the chunk appeared in this very frame, landing or rebuilt (its tag's clock is the
    // table's clock; `vd_client::arrival::appeared_at`), the design's new-chunk rows, which the motion
    // truth judges apart. An untagged mesh (nought) is never new: the clock never reads nought.
    out.fresh = select(0.0, 1.0, (tag & CLOCK_MASK) == table.clock.x);
    return out;
}

// The inverse of the sRGB transfer function: the linear value whose stored byte is `b / 255`.
fn srgb_to_linear(c: f32) -> f32 {
    if c <= 0.04045 {
        return c / 12.92;
    }
    return pow((c + 0.055) / 1.055, 2.4);
}

@fragment
fn fragment(in: ProbeOutput) -> @location(0) vec4<f32> {
    let d = length(in.world_position.xyz);
    if d >= bands().w {
        discard;
    }
    let cells = clamp(floor(d / probe.cell_m + 0.5), 0.0, 65535.0);
    let q = u32(cells);
    let hi = f32(q >> 8u) / 255.0;
    let lo = f32(q & 255u) / 255.0;
    // The fourth byte is linear in an sRGB target, so it stores `round(255 · a)` exactly. A pixel is
    // moving when any vertex of its triangle is.
    let byte = 255.0 - BIT_STILL * select(0.0, 1.0, in.moving > 0.0)
        - BIT_OLD * select(0.0, 1.0, in.fresh > 0.0);
    return vec4<f32>(
        srgb_to_linear(probe.code / 255.0),
        srgb_to_linear(hi),
        srgb_to_linear(lo),
        byte / 255.0,
    );
}
