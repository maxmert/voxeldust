// Starfield: dual-mode (skybox / parallax) instanced billboards.
//
// One mesh holds every visible star as a quad (4 verts, 6 indices).
// Vertex shader expands each quad into a camera-facing billboard and
// lerps between two world positions:
//   * skybox  — direction from the *anchor* to the star, placed at fixed
//               skybox_radius. Camera position has no effect, so it
//               looks like a static sky dome.
//   * parallax — direction + distance from the *camera*, clamped to
//                1.5× skybox_radius. As the camera moves through galaxy
//                space, near stars sweep past while far ones hold roughly
//                still.
// `parallax_amount` (0..1) crossfades between modes via smoothstep.
//
// Fragment outputs a premultiplied-alpha colour with alpha = 0 so
// `AlphaMode::Add` (BLEND_PREMULTIPLIED_ALPHA = src + dst*(1-srcA))
// reduces to pure additive `dst += rgb`. Overlapping faint stars
// accumulate into a Milky Way band; bright cores saturate to white in
// LDR (no HDR / Bloom / AgX wired up yet — see client/src/main.rs).

#import bevy_pbr::mesh_view_bindings::view

struct StarfieldUniform {
    // camera_galaxy_pos − mesh_anchor, in metres (f32).
    cam_offset: vec3<f32>,
    // 0.0 = skybox, 1.0 = parallax. Lerped on shard transitions.
    parallax_amount: f32,
    // Far-field radius in metres. Skybox places stars at this distance;
    // parallax clamps at 1.5× this.
    skybox_radius: f32,
}

@group(#{MATERIAL_BIND_GROUP}) @binding(0) var<uniform> sf: StarfieldUniform;

struct VertexInput {
    // Star's galaxy-space position relative to the mesh anchor (m).
    @location(0) star_pos: vec3<f32>,
    // rgb = star-class colour (sRGB-ish, premultiplied later by glow);
    // a   = brightness multiplier in [0,1] (luminosity-derived).
    @location(1) color: vec4<f32>,
    // xy = quad corner ∈ {-1, +1}² ; z = apparent half-size in metres
    // (luminosity-derived); w = reserved.
    @location(2) params: vec4<f32>,
}

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) corner: vec2<f32>,
    @location(1) color: vec4<f32>,
}

@vertex
fn vertex(in: VertexInput) -> VertexOutput {
    let corner = in.params.xy;
    let size = in.params.z;

    // Skybox: ignore camera offset → direction is anchor→star.
    let skybox_dir = normalize(in.star_pos);
    let skybox_pos = skybox_dir * sf.skybox_radius;

    // Parallax: camera→star direction + clamped real distance.
    let rel = in.star_pos - sf.cam_offset;
    let dist = length(rel);
    // max(dist, 1.0) avoids /0 if a star coincides with the camera.
    let parallax_pos = rel * (min(dist, sf.skybox_radius * 1.5) / max(dist, 1.0));

    let t = smoothstep(0.0, 1.0, sf.parallax_amount);
    let world_pos = mix(skybox_pos, parallax_pos, t);

    // Camera-facing billboard. view.world_from_view's columns 0/1 are
    // the camera's right/up axes in world space.
    let right = normalize(view.world_from_view[0].xyz);
    let up    = normalize(view.world_from_view[1].xyz);
    let billboard_pos = world_pos + (corner.x * right + corner.y * up) * size;

    var out: VertexOutput;
    out.clip_position = view.clip_from_world * vec4(billboard_pos, 1.0);
    out.corner = corner;
    out.color = in.color;
    return out;
}

@fragment
fn fragment(in: VertexOutput) -> @location(0) vec4<f32> {
    // Radial falloff: tight Gaussian core + soft halo.
    let r = length(in.corner);
    let core = exp(-r * r * 12.0);
    let halo = 0.15 / (1.0 + r * r * 8.0);
    let glow = min(core + halo, 1.0);

    let brightness = in.color.a;
    let rgb = in.color.rgb * glow * brightness;

    // Cull near-zero pixels so the quad's outer rim doesn't waste
    // fillrate or contribute imperceptible alpha.
    if max(rgb.r, max(rgb.g, rgb.b)) < 0.002 { discard; }

    // Pure additive: alpha = 0 → BLEND_PREMULTIPLIED_ALPHA collapses to
    // `dst.rgb += rgb`. See header.
    return vec4(rgb, 0.0);
}
