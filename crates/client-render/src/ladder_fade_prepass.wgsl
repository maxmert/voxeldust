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
    prepass_bindings,
    prepass_io::FragmentOutput,
    view_transformations::position_world_to_clip,
}

struct LadderFade {
    // The rung's sink in metres (x), step 5; (z) this material's row of the band table.
    sink: vec4<f32>,
    // The rung's cell in metres (x): a splat's width (D8-8's measurement); (y) a caster's sink.
    splat: vec4<f32>,
}

@group(#{MATERIAL_BIND_GROUP}) @binding(100) var<uniform> fade: LadderFade;

// ★ THE REALM'S BAND TABLE (the far-water design §10, slice R1a): every rung's crossfade for this
// realm — one row per rung, (in_lo, sink_end, out_lo, out_hi) metres from the eye, and one row that
// is always passed — which the renderer rewrites in its prepare stage every frame, in step with
// the placements. The material holds only its row (`fade.sink.z`), so a moving horizon rewrites
// this buffer and never a material. `clock` is the arrival ramp's (slice R1c): this frame's clock (x)
// and the ramp's ticks (y); (z) LAST frame's clock and (w) the count of rows (the temporal path's T1).
// `step` is the realm's step from this frame's render space to last frame's (T1, a 3 × 4 matrix by
// rows), and `rows` holds this frame's rows and then LAST frame's, the same count.
struct BandTable {
    clock: vec4<u32>,
    step: array<vec4<f32>, 3>,
    rows: array<vec4<f32>>,
}

@group(#{MATERIAL_BIND_GROUP}) @binding(101) var<storage, read> table: BandTable;

// THE CENSUS'S DEPTH STEP: a thousandth of the depth (2⁻¹⁰), far above the depth buffer's own
// rounding (a 32-bit float keeps 2⁻²³) and far below any visible shift.
const CENSUS_DEPTH_STEP: f32 = 0.0009765625;

// THE MOTION CENSUS'S SENTINEL (T1; `vd_client_harness::motion_census::MOTION_SENTINEL`, the value the
// renderer clears the motion target to): the largest finite half float. Only the census's control
// writes it here.
const MOTION_SENTINEL: f32 = 65504.0;

// This material's row of the table.
fn bands() -> vec4<f32> {
    return table.rows[u32(fade.sink.z)];
}

// This material's row of LAST frame's table (T1): last frame's rows follow this frame's.
fn last_bands() -> vec4<f32> {
    return table.rows[table.clock.w + u32(fade.sink.z)];
}

// ★ THE REALM'S STEP (the temporal path's T1; `vd_client::motion::RealmStep`, step for step): a point
// drawn at `y` this frame stood at `step_point(y)` last frame, and a direction turns by
// `step_turn(v)`.
fn step_point(y: vec3<f32>) -> vec3<f32> {
    return step_turn(y) + vec3<f32>(table.step[0].w, table.step[1].w, table.step[2].w);
}

fn step_turn(v: vec3<f32>) -> vec3<f32> {
    return vec3<f32>(
        dot(table.step[0].xyz, v),
        dot(table.step[1].xyz, v),
        dot(table.step[2].xyz, v),
    );
}

// THE TAG's parts (`vd_client::arrival`): the clock of the frame the chunk appeared in, 28 bits; the
// flag of a chunk that rises from its stand-in (without it, whole); ★ the own-height stand-in's three
// flags (2026-09-30): the chunk's stand-in stands at its own height (its ramp starts on the coarser
// surface itself), the chunk is drawn as a stand-in (it stands at its own height), and the stand-in is
// covered (its finer ground has all arrived: it sinks over the ramp from its clock).
const RAMP_FLAG: u32 = 0x80000000u;
const OWN_HEIGHT_FLAG: u32 = 0x40000000u;
const STAND_IN_FLAG: u32 = 0x20000000u;
const COVERED_FLAG: u32 = 0x10000000u;
const CLOCK_MASK: u32 = 0x0fffffffu;

// The tag's age at this frame's clock over the ramp (`vd_client::arrival::ramp_age`): a wrapping 28-bit
// difference, so the clock's wrap moves nothing.
fn age_now(tag: u32) -> f32 {
    let age = f32((table.clock.x - (tag & CLOCK_MASK)) & CLOCK_MASK) / f32(max(table.clock.y, 1u));
    return clamp(age, 0.0, 1.0);
}

// ★ The tag's age at LAST frame's clock (T1; `vd_client::motion::previous_age`): read SIGNED, so a tag
// newer than last frame's clock reads nought — where the chunk stood. Read wrapping, that age would be
// about 2²⁸ ticks and read one: the vertex would seem to jump.
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

// ★ THE RAMP'S WEIGHT AT LAST FRAME'S CLOCK (T1; `vd_client::motion::previous_ramp_weight`, step for
// step): a chunk that landed after last frame's clock reads nought, its stand-in's surface.
fn last_arrival(instance_index: u32) -> f32 {
    let tag = mesh_functions::get_tag(instance_index);
    return select(1.0, age_last(tag), (tag & RAMP_FLAG) != 0u);
}

// ★ THE SINK'S WEIGHT (`vd_client::arrival::sink_weight`, step for step): one for a chunk of its own
// ring, nought for a stand-in at its own height, and for a covered stand-in its age over the ramp — at
// this frame's clock, and at last frame's (`vd_client::motion::previous_sink_weight`).
fn sink_weight_of(tag: u32, age: f32) -> f32 {
    let covered = select(0.0, age, (tag & COVERED_FLAG) != 0u);
    return select(1.0, covered, (tag & STAND_IN_FLAG) != 0u);
}

// ★ THE SHARE OF THE STAND-IN'S SINK a ramp starts under (`vd_client::arrival::start_sink_share`):
// nought when the stand-in stands at its own height.
fn start_share(tag: u32) -> f32 {
    return select(1.0, 0.0, (tag & OWN_HEIGHT_FLAG) != 0u);
}

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

// ★ THE STAGE'S OUTPUT: the engine's prepass output (`bevy_pbr`'s `prepass_io.wgsl`, the same
// locations) with its previous place cut to the smallest EXACT one (the temporal path's §5.2): last
// frame's clip x, y and w, three numbers in place of the engine's four-number world place. The clip
// place is linear in the world place, so it interpolates across the triangle as the world place does,
// and the fragment divides by w; without w the motion would bend across a large triangle, and the
// depth is not needed. The motion's output goes through memory for every drawn vertex, and the
// vertices do not shrink with the render scale, so each number counts.
struct FadeOutput {
    @builtin(position) position: vec4<f32>,
#ifdef NORMAL_PREPASS_OR_DEFERRED_PREPASS
    @location(2) world_normal: vec3<f32>,
#endif
    @location(4) world_position: vec4<f32>,
#ifdef MOTION_VECTOR_PREPASS
    @location(5) previous_clip: vec3<f32>,
#endif
#ifdef UNCLIPPED_DEPTH_ORTHO_EMULATION
    @location(6) unclipped_depth: f32,
#endif
#ifdef VERTEX_OUTPUT_INSTANCE_INDEX
    @location(7) instance_index: u32,
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
fn vertex(vertex: FadeVertex) -> FadeOutput {
    var out: FadeOutput;
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
    // A caster enters whole (slice R1c): the shadow ladder's own chunks carry no ramp.
    let ramp = 1.0;
#else
    let coarser = own.xyz + radial * vertex.morph_m;
    let sink = radial * fade.sink.x;
    // ★ THE ARRIVAL RAMP (slice R1c) and THE OWN-HEIGHT STAND-IN: the same start and rise and the same
    // sink's weight as the picture's vertex stage, so a late chunk's shadow rises with it.
    let tag = mesh_functions::get_tag(vertex.instance_index);
    let ramp = arrival(vertex.instance_index);
    let morphed = mix(
        coarser - radial * fade.sink.y * whole(d) * start_share(tag),
        mix(coarser, own.xyz, whole(d)) - sink * (1.0 - risen(d)) * sink_weight_of(tag, age_now(tag)),
        ramp,
    );
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
    // THE PICTURE CENSUS'S CONTROL (the temporal path's T0; `TerrainConfig::census_control_rung`): the
    // control rung stands one depth step nearer here than in the main pass (reverse depth: nearer is
    // larger), so the main pass's depth test leaves its pixels unwritten and the census must count
    // them. `fade.sink.w` is nought on every other rung, so this multiplies by one.
    out.position.z = out.position.z * (1.0 + fade.sink.w * CENSUS_DEPTH_STEP);
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
    let local_normal = normalize(mix(oct_decode(vertex.morph_normal), own_normal, whole(d) * ramp));
    out.world_normal = mesh_functions::mesh_normal_local_to_world(local_normal, vertex.instance_index);
#endif
#ifdef MOTION_VECTOR_PREPASS
    // ★ THE PREVIOUS PLACE (the temporal path's T1; `vd_client::motion::previous_placed`, step for
    // step): the vertex's own place and its radial carried into last frame's render space by its
    // realm's step, and placed there by the same rule as `morphed` above, on LAST frame's band row
    // and ramp weight. The eye stands at the origin in both frames, so the distance is the carried
    // place's length. The engine's previous transform is not read: a chunk spawned this frame has
    // none (the engine reads its place now, a still pixel under a moving eye), and a chunk's own
    // morph and ramp move it inside the realm, which no transform carries.
#ifdef MOTION_CONTROL_ENGINE
    // ★ THE DESIGN'S CONTROL 1 (T1, `VD_MOTION_CONTROL=engine`): TODAY'S FORMULA — last frame's
    // placement from the engine's previous transform, moved by THIS frame's displacement. The motion
    // truth gate must FAIL it on the rows of chunks spawned in the frame, which hold no previous
    // transform in the engine. A measurement's switch; never the picture's path.
    let previous_from_local = mesh_functions::get_previous_world_from_local(vertex.instance_index);
    let previous_own = mesh_functions::mesh_position_local_to_world(previous_from_local, vec4<f32>(vertex.position, 1.0));
    let last_world = previous_own.xyz + (out.world_position.xyz - own.xyz);
#else
    let last_own = step_point(own.xyz);
    let last_radial = step_turn(radial);
    let last_d = length(last_own);
    let last_b = last_bands();
#ifdef LIGHT_CASTER
    let last_placed = last_own - last_radial * fade.splat.y * whole_in(last_b, last_d);
#else
    let last_coarser = last_own + last_radial * vertex.morph_m;
    let last_tag = mesh_functions::get_tag(vertex.instance_index);
    let last_placed = mix(
        last_coarser - last_radial * fade.sink.y * whole_in(last_b, last_d) * start_share(last_tag),
        mix(last_coarser, last_own, whole_in(last_b, last_d))
            - last_radial * fade.sink.x * (1.0 - risen_in(last_b, last_d))
                * sink_weight_of(last_tag, age_last(last_tag)),
        last_arrival(vertex.instance_index),
    );
#endif
#ifdef SPLAT
    // The splat's corner: last frame's square faced last frame's eye, so its offset is the corner
    // spread in last frame's view space and turned back into the world (a rigid view: the inverse
    // of its turn is its transpose).
    let last_view = prepass_bindings::previous_view_uniforms.view_from_world;
    let last_turn = mat3x3<f32>(last_view[0].xyz, last_view[1].xyz, last_view[2].xyz);
    let last_corner = transpose(last_turn) * vec3<f32>(vertex.corner * fade.splat.x, 0.0);
    let last_world = last_placed + last_corner;
#else
    let last_world = last_placed;
#endif
#endif
    let last_clip = prepass_bindings::previous_view_uniforms.clip_from_world * vec4<f32>(last_world, 1.0);
    out.previous_clip = last_clip.xyw;
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
fn fragment(in: FadeOutput) -> FragmentOutput {
    if length(in.world_position.xyz) >= bands().w {
        discard;
    }
    var out: FragmentOutput;
#ifdef NORMAL_PREPASS
    out.normal = vec4<f32>(in.world_normal * 0.5 + vec3<f32>(0.5), 1.0);
#endif
#ifdef UNCLIPPED_DEPTH_ORTHO_EMULATION
    out.frag_depth = in.unclipped_depth;
#endif
#ifdef MOTION_VECTOR_PREPASS
    // ★ THE MOTION (T1): the engine's own rule, step for step (`bevy_pbr`'s `prepass.wgsl`): this
    // frame's unjittered clip place less last frame's clip place (carried from the vertex stage as
    // x, y and w), in the texture's half units with its V downward.
    let clip_position_t = view.unjittered_clip_from_world * in.world_position;
    let clip_position = clip_position_t.xy / clip_position_t.w;
    let previous_clip_position = in.previous_clip.xy / in.previous_clip.z;
    let motion = (clip_position - previous_clip_position) * vec2<f32>(0.5, -0.5);
    // THE MOTION CENSUS'S CONTROL (T1): the census's control rung writes the sentinel as its motion,
    // so the motion census must count its pixels as uncovered. `fade.sink.w` is nought on every
    // other rung, and the select keeps their motion exactly.
    out.motion_vector = select(motion, vec2<f32>(MOTION_SENTINEL), fade.sink.w > 0.5);
#endif
    return out;
}
#else
@fragment
fn fragment(in: FadeOutput) {
    if length(in.world_position.xyz) >= bands().w {
        discard;
    }
}
#endif
