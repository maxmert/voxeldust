// THE CROSSFADE (the voxel foundation, slice 8 step 3; ruling V14 D8-2, §5) — the ground's own lit
// material, extended with the GEOMORPH that carries a rung onto the next coarser one across a band
// of distance, THE SINK that drops it under the next finer one, and THE FAR EDGE where it ends.
//
// Every vertex of a chunk at rung L carries ONE METRE (`morph_m`, step 5): how far along its own
// radial it stands from the rung L + 1 surface (computed by the client library from the same
// recipe when the chunk is built); the radial is the vertex's own unit direction from the body's
// centre (since ruling B89 its place less the centre the band table carries), and the target is the vertex plus the radial times the metre. Across the
// band `(out_lo, out_hi)` around the rung's switch distance, the vertex slides from its own
// position to that target, on its own distance from the eye: whole at `out_lo`, on the coarser
// surface at `out_hi`. At `out_hi` the two surfaces coincide, so the finer ends there (the
// fragment stage discards it at and past `out_hi`) without a pop, and the coarser, present
// everywhere, carries on.
//
// THE SINK. Nearer than its own fade-in edge `in_hi` — which is the finer rung's `out_hi`, the same
// line — the chunk drops along each vertex's radial by the rung's sink (`fade.sink.x`: the
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
// The bands are the Tier-A `vd_client::ladder_view::band_table`, one row per rung of the realm's
// ONE table buffer, rewritten every frame (slice R1a); the hysteresis pair that sizes them lives
// there too. The probe's own material carries the same
// morph, sink and far edge (`probe.wgsl`), so what the probe reads is what the picture shows, and
// the shadow pass the same morph and sink (`ladder_fade_prepass.wgsl`), so the shadows fall from
// the surface the picture shows. Every distance is read from the render frame's ORIGIN, where the
// eye stands (the floating origin: every chunk is placed against the eye) — the shadow pass's view
// is the light's, so a distance read from the view would morph the shadow caster elsewhere.

#import bevy_pbr::{
    mesh_functions,
    mesh_view_bindings::view,
    forward_io::{VertexOutput, FragmentOutput},
    view_transformations::position_world_to_clip,
    pbr_fragment::pbr_input_from_standard_material,
    pbr_functions::{alpha_discard, apply_pbr_lighting, main_pass_post_lighting_processing},
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
// rows), and `rows` holds this frame's rows and then LAST frame's, the same count: the motion
// prepass reads them (`ladder_fade_prepass.wgsl`); this stage reads this frame's alone.
struct BandTable {
    clock: vec4<u32>,
    step: array<vec4<f32>, 3>,
    // ★ THE REALM'S FRAME (ruling B89): the body's centre in this frame's render space (xyz), and its spin axis there
    // (xyz) — a vertex's radial is its own place less the centre, so no vertex carries one.
    centre: vec4<f32>,
    pole: vec4<f32>,
    rows: array<vec4<f32>>,
}

@group(#{MATERIAL_BIND_GROUP}) @binding(101) var<storage, read> table: BandTable;

// This material's row of the table.
fn bands() -> vec4<f32> {
    return table.rows[u32(fade.sink.z)];
}

// THE TAG's parts (`vd_client::arrival`): the clock of the frame the chunk appeared in, 28 bits; the
// flag of a chunk that rises from its stand-in (without it, whole); ★ the own-height stand-in's three
// flags (2026-09-30; the owner: "Own height"): the chunk's stand-in stands at its own height (its ramp
// starts on the coarser surface itself), the chunk is drawn as a stand-in (it stands at its own height),
// and the stand-in is covered (its finer ground has all arrived: it sinks over the ramp from its clock).
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

// ★ THE ARRIVAL RAMP'S WEIGHT (the far-water design §10, slice R1c; `vd_client::arrival::ramp_weight`,
// step for step): nought in the frame a chunk landed, one half a second later, and one at once for a
// chunk that entered whole (its tag without the ramp's flag).
fn arrival(instance_index: u32) -> f32 {
    let tag = mesh_functions::get_tag(instance_index);
    return select(1.0, age_now(tag), (tag & RAMP_FLAG) != 0u);
}

// ★ THE SINK'S WEIGHT (`vd_client::arrival::sink_weight`, step for step): one for a chunk of its own
// ring, nought for a stand-in at its own height, and for a covered stand-in its age over the ramp.
fn sink_weight(tag: u32) -> f32 {
    let covered = select(0.0, age_now(tag), (tag & COVERED_FLAG) != 0u);
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
    // THE MORPH NORMAL (item 20): the shade the next coarser rung draws at this vertex's target,
    // blended with the own normal across the fade-out band as the position is.
    @location(11) morph_normal: vec2<f32>,
#ifdef SPLAT
    // THE SPLAT CORNER (D8-8's measurement): which corner of the camera-facing square this copy
    // of the vertex is.
    @location(10) corner: vec2<f32>,
#endif
#ifdef GROUND_SURFACE
    // ★ THE SURFACE (ruling B75 item 4): the bare rock's share at the vertex (x) and at its morph target (y).
    @location(12) surface: vec4<f32>,
    // ★ THE HEIGHT over the ladder radius, metres; the morph target stands `morph_m` higher on the same radial.
    @location(13) height_m: f32,
#endif
#ifdef WATER_SURFACE
    // ★ THE SEA'S WORDS (ruling B109): the depth at the point (x) and at its morph target (y), metres; the surface's slope
    // variance (z).
    @location(14) water: vec4<f32>,
    // ★ THE WIND OVER THE POINT (slice S3): a vector in the planet's frame along the sea's plane, m/s.
    @location(15) wind: vec3<f32>,
#endif
}

// ★ THE WATER'S OWN OPTICS (ruling B109), at the sRGB primaries' dominant wavelengths 612, 549 and 464 nm: pure water's
// absorption (Pope & Fry 1997), per metre, and its backscattering, half of Morel's (1974) b_w = 0.00288 (λ/500)^-4.32.
const WATER_ABSORB: vec3<f32> = vec3<f32>(0.2644, 0.0565, 0.0101);
const WATER_BACKSCATTER: vec3<f32> = vec3<f32>(0.000601, 0.000962, 0.001989);
// Water's refractive index: the sun's ray bends into the water by Snell's law.
const WATER_INDEX: f32 = 1.333;
// Austin (1974): the surface passes (1 - 0.043) / n² of the light that rises to it under the water up into the air.
const WATER_UP: f32 = 0.5386;
// Morel & Prieur (1977): deep water's irradiance reflectance under its surface, R∞ = 0.33 · b_b / a.
const WATER_DEEP_SHARE: f32 = 0.33;

// ★ THE PAINT TABLE (ruling B75 item 4): each cover's albedo, linear — a first table; the art has its own step.
const ROCK_ALBEDO: vec3<f32> = vec3<f32>(0.30, 0.29, 0.27);
const SOIL_ALBEDO: vec3<f32> = vec3<f32>(0.17, 0.12, 0.08);
const SNOW_ALBEDO: vec3<f32> = vec3<f32>(0.85, 0.87, 0.90);

// ★ OVER THE SNOW LINE (ruling B75 item 4): the realm's line rides the band table's last row (`table.rows[2·count]`,
// `vd_terrain::surface::SnowLine`): its height over the ladder radius at a latitude's sine `s` is
// `x + y·s² + z·|s|`, and `w` is one where the realm draws snow at all. One over the line, nought under it.
fn snow_over(height_m: f32, sin_lat: f32) -> f32 {
    let line = table.rows[2u * table.clock.w];
    let at = line.x + line.y * sin_lat * sin_lat + line.z * abs(sin_lat);
    return select(0.0, 1.0, (line.w > 0.5) && (height_m >= at));
}

// How whole the rung is at a distance: one below its fade-out band, zero past it.
fn whole(d: f32) -> f32 {
    let b = bands();
    return clamp((b.w - d) / (b.w - b.z), 0.0, 1.0);
}

// How far the rung has risen from under the finer one: zero below its fade-in band, one where
// the sink ramp ends, a little past the fade-in edge.
fn risen(d: f32) -> f32 {
    let b = bands();
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

// ★ THE SEA'S WAVES (slice S3; rulings B107, B110; `docs/investigation/2026-10-18/sea_waves_design.md`). The laws:
// Pierson & Moskowitz (1964), the fully developed wind sea, S(k) = (α/2) k⁻³ exp(−β (k_p/k)²) with ω_p = 0.855 g/U and
// β = 5/4 — the law's own form at its peak (their β = 0.74 goes with ω₀ = g/U₁₉.₅ and U₁₉.₅ ≈ 1.026 U₁₀: the same law);
// the TMA factor of the shallow water (Bouws et al. 1985, Kitaigorodskii's form); the spread of Hasselmann et al. (1980),
// cos^{2s}(θ/2), its exponents at the fully developed sea (U/c_p = 0.855); Cox & Munk (1954) the whole slope variance.
const WAVE_ALPHA: f32 = 0.0081;
const WAVE_BETA: f32 = 1.25;
const WAVE_PEAK: f32 = 0.855;
const COX_MUNK_FLOOR: f32 = 0.003;
const COX_MUNK_RATE: f32 = 0.00512;
const WAVE_DIRECTIONS: u32 = 48u;

// One sample of a band's profile's slope (four samples a row).
fn wave_profile_slope(row0: u32, i: u32) -> f32 {
    return table.rows[row0 + (i >> 2u)][i & 3u];
}

// A band's profile's slope at `s` along its direction, read between its samples, its period wrapped.
fn wave_slope_at(row0: u32, s: f32, period: f32, samples: u32) -> f32 {
    let x = fract(s / period) * f32(samples);
    let i0 = u32(floor(x)) % samples;
    return mix(wave_profile_slope(row0, i0), wave_profile_slope(row0, (i0 + 1u) % samples), fract(x));
}

// ★ THE SEA'S NORMAL AND ITS GLINT'S SLOPE VARIANCE at a point `p` (render space) under the wind `wind` (render space,
// m/s) over `depth` metres of water, for a pixel `foot` metres wide: each band adds its waves along its directions in the
// sea's plane, weighed by the spread at their angle to the wind and by THE PIXEL'S OWN FILTER — the footprint's box by its
// second moment, a wave of wavenumber k keeping H² = exp(−(k·foot)²/12) of its slope variance in the drawn normal — and
// the rest stays in Cox & Munk's variance, which the glint keeps less what the drawn waves show: one continuous rule with
// no step at any distance (W16, B16).
fn sea_waves(p: vec3<f32>, wind: vec3<f32>, depth: f32, foot: f32) -> vec4<f32> {
    let n = normalize(p - table.centre.xyz);
    let base = 2u * table.clock.w + 2u;
    let head = table.rows[base];
    let speed = length(wind);
    let cox_munk = COX_MUNK_FLOOR + COX_MUNK_RATE * speed;
    if head.x < 0.5 || speed < 0.1 {
        return vec4<f32>(n, cox_munk);
    }
    let g = head.y;
    let bands = u32(head.z);
    let samples = u32(head.w);
    let omega_p = WAVE_PEAK * g / speed;
    let kp2 = (omega_p * omega_p / g) * (omega_p * omega_p / g);
    var grad = vec3<f32>(0.0);
    var resolved = 0.0;
    let lists = base + 1u + 2u * bands;
    let row0 = lists + 2u * bands * WAVE_DIRECTIONS;
    for (var b = 0u; b < bands; b++) {
        let band = table.rows[base + 1u + 2u * b];
        let count = u32(table.rows[base + 2u + 2u * b].x);
        let k1 = band.y;
        let k2 = band.z;
        // The band's variance (the spectrum's closed integral over [k1, k2]) and the shallow water's factor.
        let km = sqrt(k1 * k2);
        let wh = sqrt(km * max(depth, 0.0));
        let tma = select(select(1.0, 1.0 - 0.5 * (2.0 - wh) * (2.0 - wh), wh <= 2.0), 0.5 * wh * wh, wh <= 1.0);
        let e = WAVE_ALPHA / (4.0 * WAVE_BETA * kp2) * (exp(-WAVE_BETA * kp2 / (k2 * k2)) - exp(-WAVE_BETA * kp2 / (k1 * k1))) * tma;
        // The share the pixel shows; a band under a ten-thousandth of it (its slopes a hundredth of their size, far under
        // one level of the picture) costs no reads.
        let h2 = exp(-(km * foot) * (km * foot) / 12.0);
        if e < 1e-8 || h2 < 1e-4 {
            continue;
        }
        resolved += h2 * e * k1 * k2;
        // The band's list: the directions its spread keeps under the eye's wind, each with the root of its share.
        var band_grad = vec3<f32>(0.0);
        let entry0 = lists + 2u * b * WAVE_DIRECTIONS;
        let profile = row0 + b * samples / 4u;
        for (var i = 0u; i < count; i++) {
            let dr = table.rows[entry0 + 2u * i];
            let share = table.rows[entry0 + 2u * i + 1u].x;
            let dt = dr.xyz - dot(dr.xyz, n) * n;
            band_grad += share * wave_slope_at(profile, dot(p, dr.xyz) + dr.w, band.x, samples) * dt;
        }
        grad += sqrt(h2 * e) * band_grad;
    }
    return vec4<f32>(normalize(n - grad), max(cox_munk - resolved, COX_MUNK_FLOOR));
}

@vertex
fn vertex(vertex: FadeVertex) -> VertexOutput {
    var out: VertexOutput;
    let world_from_local = mesh_functions::get_world_from_local(vertex.instance_index);
    let own = mesh_functions::mesh_position_local_to_world(world_from_local, vec4<f32>(vertex.position, 1.0));
    let d = length(own.xyz);
    // The vertex's radial — ★ its own place less the body's centre in this frame's render space (ruling B89: the realm's
    // band table carries the centre, rewritten every frame, so no vertex carries a radial and no material is rewritten
    // as the eye moves) — its target along it, and its sink along it: the morph in world space (step 5: one metre per
    // vertex, the sink one number per rung).
    let radial = normalize(own.xyz - table.centre.xyz);
    let coarser = own.xyz + radial * vertex.morph_m;
    let sink = radial * fade.sink.x;
    // ★ THE ARRIVAL RAMP (slice R1c; `vd_client::arrival::RampVertex::placed`): a chunk that landed
    // late starts on its stand-in's drawn surface — the coarser surface on its radial, sunk by the
    // stand-in's sink (`fade.sink.y`) times this rung's wholeness, which IS the stand-in's unrisen
    // share — and rises into its own place over the ramp; a chunk that entered whole reads a = 1.
    // ★ THE OWN-HEIGHT STAND-IN (2026-09-30; `vd_client::stand_in`): a stand-in stands at its own
    // height (its sink's weight nought) until its finer ground has all arrived, then sinks under it
    // over the ramp; a chunk landing on a stand-in at its own height starts on the coarser surface.
    let tag = mesh_functions::get_tag(vertex.instance_index);
    let ramp = arrival(vertex.instance_index);
    let morphed = mix(
        coarser - radial * fade.sink.y * whole(d) * start_share(tag),
        mix(coarser, own.xyz, whole(d)) - sink * (1.0 - risen(d)) * sink_weight(tag),
        ramp,
    );
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
#ifdef OCT_NORMAL
    let own_normal = oct_decode(vertex.oct_normal);
#else
    let own_normal = vertex.normal;
#endif
    // THE SHADE HANDS OVER WITH THE SHAPE (item 20): the same blend as the position's — and a late
    // chunk starts on its stand-in's shade, which is its morph normal (slice R1c).
    let local_normal = normalize(mix(oct_decode(vertex.morph_normal), own_normal, whole(d) * ramp));
    out.world_normal = mesh_functions::mesh_normal_local_to_world(local_normal, vertex.instance_index);
#ifdef GROUND_SURFACE
    // ★ THE SURFACE HANDS OVER WITH THE SHAPE (ruling B75 item 4): the rock by the same blend as the shade, the height
    // by the same blend as the position, and the snow over the realm's line at that height — on the latitude's sine,
    // the radial's part along the body's spin axis (`vd_terrain::surface::POLE_AXIS`, carried by the band table). The
    // soil is the rest.
    let w = whole(d) * ramp;
    let rock = mix(vertex.surface.y, vertex.surface.x, w);
    let snow = (1.0 - rock) * snow_over(vertex.height_m + vertex.morph_m * (1.0 - w), dot(radial, table.pole.xyz));
    let ground = ROCK_ALBEDO * rock + SNOW_ALBEDO * snow + SOIL_ALBEDO * (1.0 - rock - snow);
    // ★ A NARROWER RIVER'S SHARE (slice W2): the part of the cell its water covers, by the same blend as the rock, seen as
    // deep water's own colour (the medium's law with no floor showing, as the sheet draws it far from a shore).
    let river = mix(vertex.surface.w, vertex.surface.z, w);
    let water_body = WATER_UP * WATER_DEEP_SHARE * WATER_BACKSCATTER / WATER_ABSORB;
    out.color = vec4<f32>(mix(ground, water_body, river), 1.0);
#endif
#ifdef WATER_SURFACE
    // ★ THE WATER'S COLOUR BY THE MEDIUM'S LAW (ruling B109): the floor's paint (above) seen through the water's depth —
    // the irradiance reflectance under the surface of water `H` deep over a floor of albedo `A` (Maritorena, Morel &
    // Gentili 1994), `R = R∞ (1 − e^(−2KH)) + A e^(−2KH)`, with the diffuse attenuation `K = (a + b_b) / μ` along the sun's
    // ray bent into the water, then passed up through the surface (Austin's factor). The depth hands over with the
    // shape, as the paint does; the alpha carries the slope variance to the fragment stage.
    let depth = mix(vertex.water.y, vertex.water.x, w);
    // The sun's row (the band table's last, after the snow's): the direction to it, and one where a sun shines.
    let sun = table.rows[2u * table.clock.w + 1u];
    let cos_sun = select(1.0, max(dot(radial, sun.xyz), 0.0), sun.w > 0.5);
    let mu = sqrt(1.0 - (1.0 - cos_sun * cos_sun) / (WATER_INDEX * WATER_INDEX));
    let through = exp(-2.0 * (WATER_ABSORB + WATER_BACKSCATTER) / mu * depth);
    let deep = WATER_DEEP_SHARE * WATER_BACKSCATTER / WATER_ABSORB;
    // The alpha carries the depth (the waves' shallow-water law reads it); the tangent's slot carries the wind, turned
    // into render space by the piece's own placement (a turn and a shift: its length is kept).
    out.color = vec4<f32>(WATER_UP * mix(deep, out.color.rgb, through), depth);
    out.world_tangent = vec4<f32>((world_from_local * vec4<f32>(vertex.wind, 0.0)).xyz, 0.0);
#endif
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
#ifdef WATER_SURFACE
    // The pixel's footprint on the sea, metres (before any discard: a derivative wants every pixel of its quad).
    let water_foot = max(length(dpdx(in.world_position.xyz)), length(dpdy(in.world_position.xyz)));
#endif
    if length(in.world_position.xyz) >= bands().w {
        discard;
    }
    var pbr_input = pbr_input_from_standard_material(in, is_front);
#ifdef GROUND_SURFACE
    // ★ THE PAINT IS THE COLOUR (ruling B75 item 4): the engine multiplied it by the material's own colour, which stays
    // the look of a mesh with no surface.
    pbr_input.material.base_color = vec4<f32>(in.color.rgb, 1.0);
#endif
#ifdef WATER_SURFACE
    // ★ THE WAVES AND THE GLINT (slice S3; ruling B109): the normal from the waves a pixel can show, and GGX's α the RMS
    // slope of the rest (√σ²), the engine's perceptual roughness √α.
    let sea = sea_waves(in.world_position.xyz, in.world_tangent.xyz, in.color.a, water_foot);
    pbr_input.N = sea.xyz;
    pbr_input.material.perceptual_roughness = sqrt(sqrt(sea.w));
#endif
    pbr_input.material.base_color = alpha_discard(pbr_input.material, pbr_input.material.base_color);
    var out: FragmentOutput;
    out.color = apply_pbr_lighting(pbr_input);
    out.color = main_pass_post_lighting_processing(pbr_input, out.color);
    return out;
}
