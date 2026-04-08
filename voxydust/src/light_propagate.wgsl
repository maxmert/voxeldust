// Block light propagation — iterative flood fill from emissive voxels.
//
// Dispatched N iterations per frame (ping-pong between two 3D textures).
// Each iteration propagates light one block outward from all currently lit
// voxels, attenuating by 1/15 per block (matching the 0-15 emission range).

struct VoxelVolumeParams {
    world_to_volume: mat4x4<f32>,   // not used in compute shader
    sun_dir_and_size: vec4<f32>,    // w = volume size (used here)
    inv_volume_size: vec4<f32>,     // x = 1/size
}

@group(0) @binding(0)
var voxel_occupancy: texture_3d<u32>;

@group(0) @binding(1)
var light_prev: texture_3d<f32>;

@group(0) @binding(2)
var light_out: texture_storage_3d<rgba16float, write>;

@group(0) @binding(3)
var<uniform> voxel_params: VoxelVolumeParams;

// Emission colors derived from block type lower 8 bits.
// Maps known emissive block IDs to their characteristic color.
fn emission_color(block_type_lo: u32) -> vec3<f32> {
    // Lava (BlockId::LAVA = 200): warm orange-red
    if block_type_lo == 200u {
        return vec3(1.0, 0.4, 0.1);
    }
    // Magma rock (BlockId::MAGMA_ROCK = 40): deep red-orange
    if block_type_lo == 40u {
        return vec3(0.9, 0.3, 0.1);
    }
    // Sulfur (BlockId::SULFUR = 42): yellow-green glow
    if block_type_lo == 42u {
        return vec3(0.8, 0.9, 0.2);
    }
    // Energy crystal (BlockId::ENERGY_CRYSTAL = 111): purple-blue
    if block_type_lo == 111u {
        return vec3(0.5, 0.3, 1.0);
    }
    // Default: warm white (generic emissive)
    return vec3(1.0, 0.9, 0.8);
}

@compute @workgroup_size(4, 4, 4)
fn propagate(@builtin(global_invocation_id) gid: vec3<u32>) {
    let vol_size = u32(voxel_params.sun_dir_and_size.w);
    if any(gid >= vec3(vol_size)) { return; }

    let coord = vec3<i32>(gid);

    // Read occupancy: R = solid (0 or 255), G = emission (0-255), B = block type lo.
    let occ = textureLoad(voxel_occupancy, coord, 0);
    let is_solid = occ.r > 128u;
    let emission_raw = f32(occ.g) / 255.0; // 0.0 to 1.0 (maps from 0-15 * 17)
    let block_type_lo = occ.b;

    // Solid non-emissive blocks: no light propagates through. Store zero.
    if is_solid && emission_raw < 0.01 {
        textureStore(light_out, coord, vec4(0.0));
        return;
    }

    // Self-emission contribution.
    var self_light = vec3(0.0);
    if emission_raw > 0.01 {
        self_light = emission_color(block_type_lo) * emission_raw;
    }

    // If this voxel is solid (but emissive), just store self-emission.
    if is_solid {
        textureStore(light_out, coord, vec4(self_light, 0.0));
        return;
    }

    // Air voxel: sample 6 face neighbors from previous iteration.
    let vol_size_i = i32(vol_size);
    var max_light = vec3(0.0);

    let offsets = array<vec3<i32>, 6>(
        vec3(1, 0, 0), vec3(-1, 0, 0),
        vec3(0, 1, 0), vec3(0, -1, 0),
        vec3(0, 0, 1), vec3(0, 0, -1)
    );

    for (var i = 0u; i < 6u; i++) {
        let neighbor = coord + offsets[i];
        // Bounds check.
        if any(neighbor < vec3(0)) || any(neighbor >= vec3(vol_size_i)) { continue; }
        let neighbor_light = textureLoad(light_prev, neighbor, 0).rgb;
        // Take component-wise max (colored lights mix by maximum, not addition).
        max_light = max(max_light, neighbor_light);
    }

    // Attenuation: lose 1/15 per block traversed (Minecraft-style linear falloff).
    let attenuated = max_light * (14.0 / 15.0);

    // Final light = max of attenuated neighbor light and self-emission.
    let final_light = max(attenuated, self_light);

    textureStore(light_out, coord, vec4(final_light, 0.0));
}
