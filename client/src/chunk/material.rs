//! Shared chunk material(s).
//!
//! `ChunkMaterial` is `ExtendedMaterial<StandardMaterial, EclipseExt>` —
//! Bevy 0.18's stock PBR material extended with the Phase 6 eclipse
//! occlusion uniform. The base half drives the normal voxel shading
//! pipeline (white base + per-quad vertex colour from `BlockDef.color_hint`,
//! roughness = 1, metallic = 0, reflectance = 0); the extension half
//! adds a single shared per-frame uniform with the sun + planet/moon
//! positions so the WGSL fragment can subtract the directional sun's
//! contribution at any fragment that sits in a planet's shadow.
//!
//! All chunks (terrain + ship hulls) share a single `Handle<ChunkMaterial>`,
//! so updating the eclipse uniform once per frame propagates to every
//! sun-lit voxel surface in the scene. Future per-block-type variants
//! (metals, glass, holos) opt in to their own material handles.
//!
//! Sub-block emissive proxies (lamp glow cubes, thruster glow planes)
//! continue to use plain `Assets<StandardMaterial>` because they're
//! `unlit: true` — they don't receive the directional sun and therefore
//! don't need the eclipse extension.

use bevy::pbr::{ExtendedMaterial, StandardMaterial};
use bevy::prelude::*;

use crate::lighting::eclipse::EclipseExt;

/// Component alias for consumers (chunk meshes, ship hull meshes).
pub type ChunkMaterial = ExtendedMaterial<StandardMaterial, EclipseExt>;

/// Cached handle for the shared chunk material. Lazy-initialised on
/// first mesh spawn.
#[derive(Resource, Default)]
pub struct ChunkMaterialCache {
    pub opaque: Option<Handle<ChunkMaterial>>,
}

pub fn ensure_chunk_material(
    cache: &mut ChunkMaterialCache,
    materials: &mut Assets<ChunkMaterial>,
) -> Handle<ChunkMaterial> {
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
    let handle = materials.add(ChunkMaterial {
        base: StandardMaterial {
            base_color: Color::WHITE,
            perceptual_roughness: 1.0,
            metallic: 0.0,
            reflectance: 0.0,
            ..default()
        },
        extension: EclipseExt::default(),
    });
    cache.opaque = Some(handle.clone());
    handle
}
