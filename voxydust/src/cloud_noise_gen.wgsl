// Cloud noise texture generation — compute shader.
//
// Generates the 3D noise textures needed for Nubis-style volumetric clouds:
// - Shape texture (128³ Rgba8Unorm): R=Perlin-Worley, GBA=Worley fBm 3 octaves
// - Detail texture (32³ Rgba8Unorm): RGB=Worley fBm 3 octaves
//
// All noise functions are seeded for deterministic output.
// Dispatched once per planet load.
//
// References:
// - Schneider 2015: "Real-time Volumetric Cloudscapes of Horizon Zero Dawn"
// - OfenPower/RealtimeVolumetricCloudRenderer (OpenGL noise gen reference)

struct NoiseParams {
    seed: vec4<u32>,          // x = planet seed lo, y = planet seed hi, z = texture_size, w = pass_type (0=shape, 1=detail)
    frequency: vec4<f32>,     // x = base frequency, y = persistence, z = lacunarity, w = unused
}

@group(0) @binding(0)
var output_tex: texture_storage_3d<rgba8unorm, write>;

@group(0) @binding(1)
var<uniform> params: NoiseParams;

// ---------------------------------------------------------------------------
// Hash functions for seeded noise (no external state, fully deterministic)
// ---------------------------------------------------------------------------

fn hash_u32(x: u32) -> u32 {
    var v = x;
    v = v ^ (v >> 16u);
    v = v * 0x45d9f3bu;
    v = v ^ (v >> 16u);
    v = v * 0x45d9f3bu;
    v = v ^ (v >> 16u);
    return v;
}

fn hash_uvec3(p: vec3<u32>) -> u32 {
    return hash_u32(p.x ^ hash_u32(p.y ^ hash_u32(p.z ^ params.seed.x)));
}

fn hash_to_float(h: u32) -> f32 {
    return f32(h & 0xFFFFFFu) / f32(0xFFFFFF);
}

fn hash3_to_vec3(p: vec3<i32>) -> vec3<f32> {
    let h = hash_uvec3(vec3<u32>(bitcast<u32>(p.x), bitcast<u32>(p.y), bitcast<u32>(p.z)));
    return vec3(
        hash_to_float(h),
        hash_to_float(hash_u32(h + 1u)),
        hash_to_float(hash_u32(h + 2u))
    );
}

// ---------------------------------------------------------------------------
// Perlin noise (3D, gradient-based)
// ---------------------------------------------------------------------------

fn fade(t: f32) -> f32 {
    return t * t * t * (t * (t * 6.0 - 15.0) + 10.0); // quintic smoothstep
}

fn grad(hash: u32, x: f32, y: f32, z: f32) -> f32 {
    let h = hash & 15u;
    let u = select(y, x, h < 8u);
    let v = select(select(x, z, h == 12u || h == 14u), y, h < 4u);
    return select(-u, u, (h & 1u) == 0u) + select(-v, v, (h & 2u) == 0u);
}

fn perlin_noise(p: vec3<f32>) -> f32 {
    let pi = vec3<i32>(floor(p));
    let pf = fract(p);

    let u = fade(pf.x);
    let v = fade(pf.y);
    let w = fade(pf.z);

    // 8 corner gradients
    let h000 = hash_uvec3(vec3<u32>(bitcast<u32>(pi.x),     bitcast<u32>(pi.y),     bitcast<u32>(pi.z)));
    let h100 = hash_uvec3(vec3<u32>(bitcast<u32>(pi.x + 1), bitcast<u32>(pi.y),     bitcast<u32>(pi.z)));
    let h010 = hash_uvec3(vec3<u32>(bitcast<u32>(pi.x),     bitcast<u32>(pi.y + 1), bitcast<u32>(pi.z)));
    let h110 = hash_uvec3(vec3<u32>(bitcast<u32>(pi.x + 1), bitcast<u32>(pi.y + 1), bitcast<u32>(pi.z)));
    let h001 = hash_uvec3(vec3<u32>(bitcast<u32>(pi.x),     bitcast<u32>(pi.y),     bitcast<u32>(pi.z + 1)));
    let h101 = hash_uvec3(vec3<u32>(bitcast<u32>(pi.x + 1), bitcast<u32>(pi.y),     bitcast<u32>(pi.z + 1)));
    let h011 = hash_uvec3(vec3<u32>(bitcast<u32>(pi.x),     bitcast<u32>(pi.y + 1), bitcast<u32>(pi.z + 1)));
    let h111 = hash_uvec3(vec3<u32>(bitcast<u32>(pi.x + 1), bitcast<u32>(pi.y + 1), bitcast<u32>(pi.z + 1)));

    let g000 = grad(h000, pf.x,       pf.y,       pf.z);
    let g100 = grad(h100, pf.x - 1.0, pf.y,       pf.z);
    let g010 = grad(h010, pf.x,       pf.y - 1.0, pf.z);
    let g110 = grad(h110, pf.x - 1.0, pf.y - 1.0, pf.z);
    let g001 = grad(h001, pf.x,       pf.y,       pf.z - 1.0);
    let g101 = grad(h101, pf.x - 1.0, pf.y,       pf.z - 1.0);
    let g011 = grad(h011, pf.x,       pf.y - 1.0, pf.z - 1.0);
    let g111 = grad(h111, pf.x - 1.0, pf.y - 1.0, pf.z - 1.0);

    // Trilinear interpolation of gradients
    let x0 = mix(g000, g100, u);
    let x1 = mix(g010, g110, u);
    let x2 = mix(g001, g101, u);
    let x3 = mix(g011, g111, u);
    let y0 = mix(x0, x1, v);
    let y1 = mix(x2, x3, v);
    return mix(y0, y1, w); // range approximately [-1, 1]
}

fn perlin_fbm(p: vec3<f32>, octaves: u32) -> f32 {
    var freq = 1.0;
    var amp = 1.0;
    var value = 0.0;
    var total_amp = 0.0;
    let persistence = params.frequency.y;
    let lacunarity = params.frequency.z;
    for (var i = 0u; i < octaves; i++) {
        value += perlin_noise(p * freq) * amp;
        total_amp += amp;
        freq *= lacunarity;
        amp *= persistence;
    }
    return value / total_amp; // normalized to ~[-1, 1]
}

// ---------------------------------------------------------------------------
// Worley (cellular) noise — distance to nearest feature point
// ---------------------------------------------------------------------------

fn worley_noise(p: vec3<f32>) -> f32 {
    let cell = vec3<i32>(floor(p));
    let local = fract(p);

    var min_dist = 1.0;

    // Search 3×3×3 neighbourhood
    for (var x = -1; x <= 1; x++) {
        for (var y = -1; y <= 1; y++) {
            for (var z = -1; z <= 1; z++) {
                let neighbor = cell + vec3(x, y, z);
                // Deterministic feature point position within this cell
                let feature = hash3_to_vec3(neighbor);
                let diff = vec3<f32>(vec3(x, y, z)) + feature - local;
                let dist = length(diff);
                min_dist = min(min_dist, dist);
            }
        }
    }
    return min_dist; // range [0, ~1.5], typically [0, 1]
}

fn worley_fbm(p: vec3<f32>, octaves: u32) -> f32 {
    var freq = 1.0;
    var amp = 1.0;
    var value = 0.0;
    var total_amp = 0.0;
    let persistence = params.frequency.y;
    let lacunarity = params.frequency.z;
    for (var i = 0u; i < octaves; i++) {
        value += worley_noise(p * freq) * amp;
        total_amp += amp;
        freq *= lacunarity;
        amp *= persistence;
    }
    return value / total_amp;
}

// ---------------------------------------------------------------------------
// Perlin-Worley combination (Nubis base shape)
//
// Creates connected, billowy shapes: Perlin provides the large-scale structure,
// Worley provides the puffy cellular detail. The combination remaps Perlin
// into the range [worley, 1.0] to ensure connected structures.
// ---------------------------------------------------------------------------

fn perlin_worley(p: vec3<f32>) -> f32 {
    let perl = perlin_fbm(p, 4u) * 0.5 + 0.5; // remap [-1,1] to [0,1]
    let worl = 1.0 - worley_fbm(p, 3u);        // invert: 0=at feature, 1=far from feature
    // Remap Perlin into [worley, 1.0]: connected where Worley is high, broken where low
    return saturate(remap_f(perl, worl - 1.0, 1.0, 0.0, 1.0));
}

fn remap_f(value: f32, old_lo: f32, old_hi: f32, new_lo: f32, new_hi: f32) -> f32 {
    return new_lo + (value - old_lo) / (old_hi - old_lo) * (new_hi - new_lo);
}

// ---------------------------------------------------------------------------
// Main compute kernel
// ---------------------------------------------------------------------------

@compute @workgroup_size(4, 4, 4)
fn gen_shape(@builtin(global_invocation_id) gid: vec3<u32>) {
    let size = params.seed.z;
    if any(gid >= vec3(size)) { return; }

    let uv = vec3<f32>(gid) / f32(size);
    let freq = params.frequency.x;

    // R: Perlin-Worley (connected billowy base shape)
    let pw = perlin_worley(uv * freq);

    // GBA: Worley fBm at 3 different frequencies (for detail erosion blending)
    let w1 = 1.0 - worley_fbm(uv * freq * 1.0, 3u);
    let w2 = 1.0 - worley_fbm(uv * freq * 2.0, 3u);
    let w3 = 1.0 - worley_fbm(uv * freq * 4.0, 3u);

    textureStore(output_tex, vec3<i32>(gid), vec4(pw, w1, w2, w3));
}

@compute @workgroup_size(4, 4, 4)
fn gen_detail(@builtin(global_invocation_id) gid: vec3<u32>) {
    let size = params.seed.z;
    if any(gid >= vec3(size)) { return; }

    let uv = vec3<f32>(gid) / f32(size);
    let freq = params.frequency.x;

    // RGB: Worley fBm at 3 octaves (fine erosion detail)
    let w1 = 1.0 - worley_fbm(uv * freq * 1.0, 3u);
    let w2 = 1.0 - worley_fbm(uv * freq * 2.0, 3u);
    let w3 = 1.0 - worley_fbm(uv * freq * 4.0, 3u);

    textureStore(output_tex, vec3<i32>(gid), vec4(w1, w2, w3, 1.0));
}
