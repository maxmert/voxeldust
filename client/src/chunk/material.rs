//! Shared chunk material(s).
//!
//! Current strategy: one `StandardMaterial` with `base_color: WHITE`;
//! per-block color comes from vertex colors (baked into each quad by
//! the greedy mesher reading `BlockDef.color_hint`). This is simpler
//! than per-block-type material handles and avoids the Bevy render
//! pipeline shader-permutation proliferation a per-material approach
//! triggers. A future phase can swap in per-material PBR config if
//! block materials need to differ in roughness / metallic / emissive.

use bevy::prelude::*;

/// Cached handle for the shared chunk material. Lazy-initialised on
/// first mesh spawn.
#[derive(Resource, Default)]
pub struct ChunkMaterialCache {
    pub opaque: Option<Handle<StandardMaterial>>,
}

/// Component alias for consumers.
pub type ChunkMaterial = StandardMaterial;

pub fn ensure_chunk_material(
    cache: &mut ChunkMaterialCache,
    materials: &mut Assets<StandardMaterial>,
) -> Handle<StandardMaterial> {
    if let Some(ref h) = cache.opaque {
        return h.clone();
    }
    // `reflectance: 0.0` (default 0.5) — kills dielectric F0 specular
    // entirely. Voxel hulls are matte painted surfaces; future block types
    // (metal, glass) opt in to specular via per-block-type materials.
    //
    // Without this, Fresnel ramps the F0 = 0.04 specular toward 1.0 at
    // glancing angles, and the GGX spike on rough surfaces under physical
    // ~127 000-lux Sol produces single-pixel super-bright fragments that
    // Bloom's mip pyramid smears into visible "fireflies" on otherwise
    // dark hull faces.
    let handle = materials.add(StandardMaterial {
        base_color: Color::WHITE,
        perceptual_roughness: 1.0,
        metallic: 0.0,
        reflectance: 0.0,
        ..default()
    });
    cache.opaque = Some(handle.clone());
    handle
}
