//! The source-agnostic realm-geometry projection + the box→block-mesh lowering IR
//! (Visual Crossing Playground, Slice V0).
//!
//! A dev-config (`Vec<RealmBoundary>`) — and, later, a reviewed wire arm — projects to
//! ONE [`RealmScene`] (`BTreeMap<RealmId, RealmBox>`); the renderer binds to the scene and
//! its lowering [`MeshPrim`]s, NEVER to `RealmBoundary` (which leaks the shard-authority-internal
//! `to_realm`/`band`/`effect`) and NEVER to a source. This crate carries NO Bevy types: a
//! `MeshPrim` is a plain vertex buffer + a translucent color + a translation/scale transform,
//! so the Tier-B renderer does `Mesh::from(prim.vertices)` with ZERO shape branch — the seam
//! that keeps A→B (dev-config→wire) and box→block-mesh (P4 greedy quads) non-cornering.
//!
//! HR3 (one tooling, never `match` on a realm KIND): [`stable_seed`] hashes the
//! POSTCARD BYTES of the whole [`RealmId`] — discriminant + payload uniformly — so the hue
//! is a pure function of identity with no per-kind arm (adversary H3). H4: the shape→vertex
//! tessellation lives HERE in Tier-A, not in the coverage-exempt renderer.

use std::collections::BTreeMap;

use glam::DVec3;
use vd_core::geometry::{Boundary, RealmBoundary};
use vd_core::pose::{FrameRef, RealmId};

/// The render-relevant shape of one realm's extent. A `RealmBoundary::shape` projects to this,
/// DROPPING the metric band/effect: [`Boundary::Shell`]→[`BoxShape::Sphere`],
/// [`Boundary::Aabb`]→[`BoxShape::Box`], [`Boundary::Obb`]→[`BoxShape::Box`] (orientation
/// DEFERRED — see the module NOTE below; the proxy renders axis-aligned until the Obb-orient
/// arm lands with the block-mesh work).
///
/// NOTE (Obb orientation deferred): an `Obb`'s `orient` is dropped at projection today. The
/// box→block-mesh successor (P4/P8) that emits real hull geometry carries the orientation into
/// the vertex tessellation; until then a station volume renders as its axis-aligned bounding
/// proxy, which is correct for the membership verdict (world-space) and only approximate for
/// the pixels (corroborating only).
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum BoxShape {
    /// A spherical extent of radius `r` (an SOI / planet-descent shell).
    Sphere { r: f64 },
    /// A box extent with per-axis half-extents `half` (a station/area/ship volume).
    Box { half: DVec3 },
}

/// One realm's render description: its render-relevant shape, its center OFFSET from the realm's
/// frame origin, its parent link (nesting), its nesting `depth` (0 = top level), and its
/// TRANSLUCENT color. Carries ONLY render-relevant fields — the `RealmBoundary`'s
/// `to_realm`/`band`/`effect` are shard-authority-internal and are dropped at projection.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RealmBox {
    /// The render-relevant shape (sphere or box).
    pub shape: BoxShape,
    /// The realm's authoritative reference frame — the box's `center_offset` is expressed in THIS
    /// frame, so the renderer places the box by composing the frame ORIGIN through the ONE
    /// `DeliveredView::world_pos` chokepoint (identity for the world-origin frames through P3; a
    /// Station-hull-borne box composes through its hull at P8, WITHOUT changing this shape). Carried
    /// here so the render glue never reconstructs a `FrameRef` from a `RealmId` — the `Area` arm
    /// can't (it needs the parent planet seed) and it would be a per-KIND match in a feature path.
    pub frame: FrameRef,
    /// The box center as a frame-local offset from the realm's frame origin.
    pub center_offset: DVec3,
    /// The parent realm, when this box nests inside another (`None` at the top level).
    pub parent: Option<RealmId>,
    /// The nesting depth (0 = top level, +1 per parent hop) — the deterministic parent-walk result.
    pub depth: u8,
    /// The TRANSLUCENT render color (identity-derived hue; alpha ~0.25 so nested boxes show through).
    pub color_rgba: [f32; 4],
}

/// The source-agnostic scene: one [`RealmBox`] per realm, keyed by [`RealmId`] for a
/// deterministic (order-independent) iteration order. The single chokepoint the renderer
/// binds to — a dev-config loader and a (future) wire decoder both produce this identical shape.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct RealmScene(BTreeMap<RealmId, RealmBox>);

/// Why a `Vec<RealmBoundary>` (or a `boxes.json` dev-config) failed to project to a [`RealmScene`].
#[derive(Clone, Debug, PartialEq, Eq, thiserror::Error)]
pub enum SceneError {
    /// Two boundaries named the same realm — the map key would collide, so the source is rejected
    /// loudly rather than silently keeping whichever the input happened to list first.
    #[error("duplicate realm id in the boundary set")]
    DuplicateRealm,
    /// A parent link formed a cycle, or the parent chain exceeded [`MAX_NEST_DEPTH`] — a bounded,
    /// loud stop instead of an unbounded walk (adversary H7 cycle/again-guard).
    #[error("parent chain cycles or exceeds the max nesting depth")]
    CycleOrDepthExceeded,
    /// The `boxes.json` dev-config was not a valid JSON array of `RealmBoundary` — the (owned)
    /// serde message so a bad hand-authored file fails LOUD at load, never silently empty.
    #[error("malformed boxes.json: {0}")]
    MalformedJson(String),
}

/// The nesting-depth ceiling: the parent walk stops (loud, [`SceneError::CycleOrDepthExceeded`])
/// past this many hops, so a cyclic or pathological parent chain cannot loop forever. Generous
/// for the real topology (station ⊃ ship ⊃ player is depth 2).
pub const MAX_NEST_DEPTH: u8 = 16;

/// The alpha of every projected box — TRANSLUCENT so nested boxes show through their parents
/// (the multi-mesh "mesh inside mesh" render). A named const (no magic number).
pub const BOX_ALPHA: f32 = 0.25;

impl RealmScene {
    /// Project a boundary set into the render scene: build the `BTreeMap<RealmId, RealmBox>`
    /// FIRST (rejecting a duplicate realm id, [`SceneError::DuplicateRealm`]), then compute each
    /// box's `depth` by walking its `parent` links via MAP LOOKUP (not a linear Vec scan — H7),
    /// bounded by [`MAX_NEST_DEPTH`] (a cycle/over-deep chain is a loud
    /// [`SceneError::CycleOrDepthExceeded`], not an infinite loop). Order-independent: the depth is
    /// a pure function of the realm SET, never the input Vec order.
    ///
    /// # Errors
    /// [`SceneError::DuplicateRealm`] on a repeated realm id; [`SceneError::CycleOrDepthExceeded`]
    /// on a cyclic/over-deep parent chain.
    pub fn from_boundaries(boundaries: &[RealmBoundary]) -> Result<RealmScene, SceneError> {
        // Pass 1: the parent map (realm → its parent), rejecting duplicates. Keyed by RealmId so the
        // depth walk below is a LOOKUP, never a linear find (order-independent, H7).
        let mut parents: BTreeMap<RealmId, Option<RealmId>> = BTreeMap::new();
        for b in boundaries {
            if parents.insert(b.realm, b.parent).is_some() {
                return Err(SceneError::DuplicateRealm);
            }
        }
        // Pass 2: project each boundary, computing depth over the map (a pure function of the set).
        let mut boxes: BTreeMap<RealmId, RealmBox> = BTreeMap::new();
        for b in boundaries {
            let depth = depth_of(b.realm, &parents)?;
            boxes.insert(
                b.realm,
                RealmBox {
                    shape: shape_of(b.shape),
                    frame: frame_of_realm(b.realm, b.parent),
                    center_offset: b.center.offset(),
                    parent: b.parent,
                    depth,
                    color_rgba: color_from_seed(stable_seed(b.realm)),
                },
            );
        }
        Ok(RealmScene(boxes))
    }

    /// Project a `boxes.json` dev-config into the render scene. The JSON is a plain array of
    /// [`RealmBoundary`] — the IDENTICAL `Vec<RealmBoundary>` the shard plants into its
    /// `RealmBoundaries` resource (single-sourced: the same authored file feeds both the shard
    /// authority and the client render), so a box's extent can never disagree with the shard's
    /// crossing geometry. Deserializes then delegates to [`RealmScene::from_boundaries`] — so a
    /// malformed file and a duplicate/cyclic set both fail LOUD ([`SceneError`]), never a silent
    /// empty scene.
    ///
    /// # Errors
    /// [`SceneError::MalformedJson`] if the text is not a valid `RealmBoundary` array;
    /// [`SceneError::DuplicateRealm`] / [`SceneError::CycleOrDepthExceeded`] as
    /// [`RealmScene::from_boundaries`].
    pub fn from_boxes_json(json: &str) -> Result<RealmScene, SceneError> {
        let boundaries = parse_boundaries(json)?;
        RealmScene::from_boundaries(&boundaries)
    }

    /// The box for a realm, if present.
    #[must_use]
    pub fn get(&self, realm: RealmId) -> Option<&RealmBox> {
        self.0.get(&realm)
    }

    /// Iterate the boxes in deterministic (`RealmId`-`Ord`) order — the render loop + the
    /// membership verdicts consume this.
    pub fn iter(&self) -> impl Iterator<Item = (RealmId, &RealmBox)> {
        self.0.iter().map(|(r, b)| (*r, b))
    }

    /// The number of boxes in the scene.
    #[must_use]
    pub fn len(&self) -> usize {
        self.0.len()
    }

    /// Whether the scene has no boxes.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }
}

/// Parse a `boxes.json` text into a `Vec<RealmBoundary>`, mapping the serde error into an owned
/// [`SceneError::MalformedJson`] — a monomorphic helper so the fallible decode + error map live
/// here, off [`RealmScene::from_boxes_json`] (which stays a straight-line delegate).
fn parse_boundaries(json: &str) -> Result<Vec<RealmBoundary>, SceneError> {
    serde_json::from_str(json).map_err(|e| SceneError::MalformedJson(e.to_string()))
}

/// The nesting depth of `realm` by walking `parents` up to the root — a monomorphic, bounded
/// map-lookup walk (all branching here, so [`RealmScene::from_boundaries`] stays a straight-line
/// shim). Stops loud past [`MAX_NEST_DEPTH`] hops so a cycle or an over-deep chain cannot loop.
fn depth_of(
    realm: RealmId,
    parents: &BTreeMap<RealmId, Option<RealmId>>,
) -> Result<u8, SceneError> {
    let mut depth: u8 = 0;
    let mut cursor = realm;
    loop {
        // Only a parent PRESENT in the scene contributes a nesting level: a genuine `None`, OR a
        // parent link to a realm NOT in the set (an out-of-scene parent has no visible box to nest
        // inside), terminates the chain as a root. So depth counts PRESENT ancestors only.
        let parent = match parents.get(&cursor) {
            Some(Some(p)) if parents.contains_key(p) => *p,
            _ => return Ok(depth),
        };
        if depth == MAX_NEST_DEPTH {
            return Err(SceneError::CycleOrDepthExceeded);
        }
        depth += 1;
        cursor = parent;
    }
}

/// Project a core [`Boundary`] to its render [`BoxShape`], dropping the metric band and the Obb
/// orientation (see the [`BoxShape`] NOTE). A monomorphic helper (the KIND branch lives here,
/// off the generic projection path).
fn shape_of(boundary: Boundary) -> BoxShape {
    match boundary {
        Boundary::Shell { r } => BoxShape::Sphere { r },
        Boundary::Aabb { half } => BoxShape::Box { half },
        // Obb orientation DEFERRED: render the axis-aligned bounding proxy for now.
        Boundary::Obb { half, orient: _ } => BoxShape::Box { half },
    }
}

/// The realm's authoritative [`FrameRef`] — the frame its box's `center_offset` is expressed in,
/// so the render glue composes the box through the ONE `world_pos` chokepoint. A monomorphic helper
/// (the realm-KIND destructure lives HERE in Tier-A, never in the Tier-B render feature path). An
/// `Area` frame needs the PARENT planet seed (a `RealmId::Area` alone can't carry it): the parent is
/// the boundary's `parent` link when it is a `Planet`, else `0` (a top-level area — defensive, no
/// panic). A `Ship`'s frame is keyed by its hull entity; the rest map their seed directly.
fn frame_of_realm(realm: RealmId, parent: Option<RealmId>) -> FrameRef {
    match realm {
        RealmId::Planet(planet_seed) => FrameRef::PlanetCentered { planet_seed },
        RealmId::System(system_seed) => FrameRef::SystemSpace { system_seed },
        RealmId::Ship(ship) => FrameRef::ShipLocal { ship },
        RealmId::Station(station_seed) => FrameRef::StationLocal { station_seed },
        RealmId::Area(area_seed) => FrameRef::AreaLocal {
            planet_seed: parent_planet_seed(parent),
            area_seed,
        },
    }
}

/// The parent planet's seed for an `Area` frame: the `parent` link when it names a `Planet`, else
/// `0` (a top-level or non-planet-parented area — defensive). Monomorphic so both arms are covered.
fn parent_planet_seed(parent: Option<RealmId>) -> u64 {
    match parent {
        Some(RealmId::Planet(seed)) => seed,
        _ => 0,
    }
}

/// The FNV-1a offset basis (64-bit).
const FNV_OFFSET: u64 = 0xcbf2_9ce4_8422_2325;
/// The FNV-1a prime (64-bit).
const FNV_PRIME: u64 = 0x0000_0100_0000_01b3;

/// A KIND-AGNOSTIC stable seed for a realm: hash the POSTCARD-serialized bytes of the WHOLE
/// [`RealmId`] (discriminant + payload) with FNV-1a. There is NO `match` on the realm KIND —
/// the enum arms are never destructured here, so `System(5)` and `Planet(5)` hash differently
/// (their discriminant byte differs) and a new realm arm needs no change (adversary H3). Pure
/// and deterministic (postcard v1 + a fixed-seed hash), identical in every process/run.
#[must_use]
pub fn stable_seed(realm: RealmId) -> u64 {
    // postcard v1 encodes the enum discriminant then the payload; encoding a `RealmId` cannot fail
    // (all arms are plain integers), but we never unwrap silently — an empty encoding still hashes.
    let bytes = postcard::to_allocvec(&realm).unwrap_or_default();
    fnv1a(&bytes)
}

/// FNV-1a over a byte slice — a monomorphic, branchless-per-byte deterministic hash (no dep).
fn fnv1a(bytes: &[u8]) -> u64 {
    let mut hash = FNV_OFFSET;
    for &b in bytes {
        hash ^= u64::from(b);
        hash = hash.wrapping_mul(FNV_PRIME);
    }
    hash
}

/// The golden-ratio conjugate — successive multiples spread hues maximally around the wheel, so
/// adjacent realm seeds get visually distinct colors.
const GOLDEN_RATIO_CONJUGATE: f64 = 0.618_033_988_749_895;

/// A TRANSLUCENT RGBA color from a stable seed: the seed picks a hue on the golden-ratio-spaced
/// wheel (fixed saturation/value for vivid, legible boxes), converted to sRGB with [`BOX_ALPHA`].
/// Pure — the same seed always yields the same color (determinism is load-bearing for captures).
#[must_use]
pub fn color_from_seed(seed: u64) -> [f32; 4] {
    // Map the seed into [0,1) and advance by the golden-ratio conjugate for a well-spread hue.
    let unit = (seed as f64) / (u64::MAX as f64);
    let hue = (unit + GOLDEN_RATIO_CONJUGATE).fract();
    let [r, g, b] = hsv_to_rgb(hue, 0.65, 0.95);
    [r, g, b, BOX_ALPHA]
}

/// HSV→RGB for `h,s,v ∈ [0,1]` → linear-ish `[r,g,b] ∈ [0,1]` (the standard 6-sector formula).
/// A monomorphic helper so every sector branch is covered here, off the pure `color_from_seed`.
fn hsv_to_rgb(h: f64, s: f64, v: f64) -> [f32; 3] {
    let sector = (h * 6.0).floor();
    let f = h * 6.0 - sector;
    let p = v * (1.0 - s);
    let q = v * (1.0 - s * f);
    let t = v * (1.0 - s * (1.0 - f));
    // sector is floor(h*6), h∈[0,1) ⇒ sector∈{0..5}; the `_` arm (sector 6 at h==1.0 exactly, or a
    // pathological value) folds back to the red sector so the fn is total.
    let (r, g, b) = match sector as i64 {
        0 => (v, t, p),
        1 => (q, v, p),
        2 => (p, v, t),
        3 => (p, q, v),
        4 => (t, p, v),
        _ => (v, p, q),
    };
    #[allow(clippy::cast_possible_truncation)]
    [r as f32, g as f32, b as f32]
}

/// One render vertex — position + normal, in the box's LOCAL space (the transform places it).
/// Plain arrays, no Bevy types: the renderer builds a `Mesh` from a `Vec<Vertex>`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Vertex {
    pub pos: [f32; 3],
    pub normal: [f32; 3],
}

/// A plain translation+uniform-per-axis scale transform (Bevy-FREE) — how a [`MeshPrim`]'s
/// local vertices are placed into the world. The renderer applies `translation` + `scale` with
/// no rotation (Obb orientation is deferred, see the [`BoxShape`] NOTE).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PrimTransform {
    pub translation: [f32; 3],
    pub scale: [f32; 3],
}

/// A renderer-agnostic mesh primitive: a vertex buffer + a translucent color + a place transform.
/// The renderer consumes ONLY this — it never learns the word "sphere" or "box" (adversary H4).
/// P4 terrain later emits greedy-quad vertices through this SAME type with zero renderer change.
#[derive(Clone, Debug, PartialEq)]
pub struct MeshPrim {
    pub vertices: Vec<Vertex>,
    pub color_rgba: [f32; 4],
    pub transform: PrimTransform,
}

/// The coarse UV-sphere longitude/latitude resolution for the proxy (low-poly, deterministic —
/// NO LOD). Named consts (no magic numbers).
pub const SPHERE_SECTORS: usize = 12;
pub const SPHERE_STACKS: usize = 8;

/// Lower a [`RealmBox`] to its render primitives at `world_center` (the box's world-space origin,
/// computed by the caller through the `world_pos` composition seam). TESSELLATES the shape into
/// VERTICES here in Tier-A (adversary H4): a `Box` → a unit cuboid (12 triangles) scaled by its
/// half-extents; a `Sphere` → a coarse UV sphere scaled by `r`. Exactly one prim per box today;
/// the block-mesh successor emits more prims (or more vertices) through the same shape.
#[must_use]
pub fn to_render_prims(rbox: &RealmBox, world_center: DVec3) -> Vec<MeshPrim> {
    let center = world_center + rbox.center_offset;
    let translation = [center.x as f32, center.y as f32, center.z as f32];
    match rbox.shape {
        BoxShape::Box { half } => {
            vec![MeshPrim {
                vertices: unit_cuboid_vertices(),
                color_rgba: rbox.color_rgba,
                transform: PrimTransform {
                    translation,
                    scale: [half.x as f32, half.y as f32, half.z as f32],
                },
            }]
        }
        BoxShape::Sphere { r } => {
            let r32 = r as f32;
            vec![MeshPrim {
                vertices: unit_sphere_vertices(),
                color_rgba: rbox.color_rgba,
                transform: PrimTransform {
                    translation,
                    scale: [r32, r32, r32],
                },
            }]
        }
    }
}

/// The 8 corners of the unit cube `[-1,1]^3`.
fn cube_corners() -> [[f32; 3]; 8] {
    [
        [-1.0, -1.0, -1.0],
        [1.0, -1.0, -1.0],
        [1.0, 1.0, -1.0],
        [-1.0, 1.0, -1.0],
        [-1.0, -1.0, 1.0],
        [1.0, -1.0, 1.0],
        [1.0, 1.0, 1.0],
        [-1.0, 1.0, 1.0],
    ]
}

/// A unit cuboid (`[-1,1]^3`) as 12 triangles (36 vertices) with per-face outward normals — the
/// `Box` tessellation. Placed by the [`PrimTransform`] scale (the box half-extents).
fn unit_cuboid_vertices() -> Vec<Vertex> {
    let c = cube_corners();
    // Six faces, each two triangles (CCW when viewed from outside), with the face's outward normal.
    let faces: [([usize; 4], [f32; 3]); 6] = [
        ([0, 3, 2, 1], [0.0, 0.0, -1.0]), // -Z
        ([4, 5, 6, 7], [0.0, 0.0, 1.0]),  // +Z
        ([0, 4, 7, 3], [-1.0, 0.0, 0.0]), // -X
        ([1, 2, 6, 5], [1.0, 0.0, 0.0]),  // +X
        ([0, 1, 5, 4], [0.0, -1.0, 0.0]), // -Y
        ([3, 7, 6, 2], [0.0, 1.0, 0.0]),  // +Y
    ];
    let mut verts = Vec::with_capacity(faces.len() * 6);
    for (quad, normal) in faces {
        // Two triangles per quad: (a,b,c) and (a,c,d).
        for &i in &[quad[0], quad[1], quad[2], quad[0], quad[2], quad[3]] {
            verts.push(Vertex { pos: c[i], normal });
        }
    }
    verts
}

/// A coarse unit UV sphere (radius 1) as triangles — the `Sphere` proxy tessellation. Deterministic
/// [`SPHERE_SECTORS`]×[`SPHERE_STACKS`] grid; the outward normal equals the unit position. Placed
/// by the [`PrimTransform`] scale (the sphere radius).
fn unit_sphere_vertices() -> Vec<Vertex> {
    use std::f64::consts::PI;
    // Grid of positions (stacks from the +Y pole to the -Y pole, sectors around Y).
    let mut grid: Vec<Vec<[f32; 3]>> = Vec::with_capacity(SPHERE_STACKS + 1);
    for i in 0..=SPHERE_STACKS {
        let stack_angle = PI / 2.0 - (i as f64) * PI / (SPHERE_STACKS as f64); // +pi/2 .. -pi/2
        let xz = stack_angle.cos();
        let y = stack_angle.sin();
        let mut row = Vec::with_capacity(SPHERE_SECTORS + 1);
        for j in 0..=SPHERE_SECTORS {
            let sector_angle = (j as f64) * 2.0 * PI / (SPHERE_SECTORS as f64);
            let x = xz * sector_angle.cos();
            let z = xz * sector_angle.sin();
            row.push([x as f32, y as f32, z as f32]);
        }
        grid.push(row);
    }
    // Two triangles per grid quad, skipping the degenerate pole triangles.
    let mut verts = Vec::new();
    for i in 0..SPHERE_STACKS {
        for j in 0..SPHERE_SECTORS {
            let a = grid[i][j];
            let b = grid[i + 1][j];
            let c = grid[i + 1][j + 1];
            let d = grid[i][j + 1];
            // Top cap (i==0): the a/d row collapses to the pole → one triangle (b,c,pole).
            if i != 0 {
                push_tri(&mut verts, a, b, d);
            }
            // Bottom cap (i==STACKS-1): the b/c row collapses to the pole → one triangle (a,c... ).
            if i != SPHERE_STACKS - 1 {
                push_tri(&mut verts, b, c, d);
            }
        }
    }
    verts
}

/// Push one triangle (three vertices, each normal = its unit position for a sphere).
fn push_tri(verts: &mut Vec<Vertex>, a: [f32; 3], b: [f32; 3], c: [f32; 3]) {
    for pos in [a, b, c] {
        verts.push(Vertex { pos, normal: pos });
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use glam::{DQuat, DVec3};
    use vd_core::EntityId;
    use vd_core::geometry::{CrossEffect, RealmBoundary};
    use vd_core::pose::{LatticePos, RealmId};

    /// A `Shell` boundary for `realm` at `center_offset`, radius `r`, parent `parent`.
    fn shell_boundary(
        realm: RealmId,
        center: DVec3,
        r: f64,
        parent: Option<RealmId>,
    ) -> RealmBoundary {
        RealmBoundary::shell(
            realm,
            LatticePos::local(center),
            r,
            1.15,
            1.30,
            0.0,
            0.05,
            0.5,
            1.0,
            parent,
            realm,
            CrossEffect::Authority,
        )
    }

    /// An `Aabb` boundary for `realm`, half-extents `half`, parent `parent`.
    fn aabb_boundary(realm: RealmId, half: DVec3, parent: Option<RealmId>) -> RealmBoundary {
        RealmBoundary::aabb(
            realm,
            LatticePos::local(DVec3::ZERO),
            half,
            1.15,
            1.30,
            0.0,
            0.05,
            0.5,
            1.0,
            parent,
            realm,
            CrossEffect::Authority,
        )
        .expect("valid aabb band")
    }

    /// An `Obb` boundary for `realm` (orientation should be DROPPED at projection).
    fn obb_boundary(realm: RealmId, half: DVec3) -> RealmBoundary {
        RealmBoundary::boxed(
            realm,
            LatticePos::local(DVec3::ZERO),
            half,
            Some(DQuat::from_rotation_z(0.5)),
            1.15,
            1.30,
            None,
            realm,
            CrossEffect::Interest,
        )
        .expect("valid obb band")
    }

    #[test]
    fn shell_projects_to_a_sphere_and_aabb_to_a_box() {
        let scene = RealmScene::from_boundaries(&[
            shell_boundary(RealmId::System(7), DVec3::new(1.0, 2.0, 3.0), 1000.0, None),
            aabb_boundary(RealmId::Station(9), DVec3::new(10.0, 20.0, 30.0), None),
        ])
        .expect("projects");
        assert_eq!(scene.len(), 2);
        assert!(!scene.is_empty());
        let sys = scene.get(RealmId::System(7)).expect("system box");
        assert_eq!(sys.shape, BoxShape::Sphere { r: 1000.0 });
        assert_eq!(sys.center_offset, DVec3::new(1.0, 2.0, 3.0));
        let stn = scene.get(RealmId::Station(9)).expect("station box");
        assert_eq!(
            stn.shape,
            BoxShape::Box {
                half: DVec3::new(10.0, 20.0, 30.0)
            }
        );
    }

    #[test]
    fn obb_projects_to_an_axis_aligned_box_dropping_orientation() {
        // The Obb-orient arm of shape_of: the orientation is deferred, so an Obb renders as its
        // axis-aligned bounding proxy (Box{half}).
        let scene = RealmScene::from_boundaries(&[obb_boundary(
            RealmId::Area(3),
            DVec3::new(4.0, 5.0, 6.0),
        )])
        .expect("projects");
        let area = scene.get(RealmId::Area(3)).expect("area box");
        assert_eq!(
            area.shape,
            BoxShape::Box {
                half: DVec3::new(4.0, 5.0, 6.0)
            }
        );
    }

    #[test]
    fn frame_of_realm_maps_every_kind_and_derives_the_area_parent_seed() {
        use vd_core::entity_kind::EntityKind;
        // Every realm KIND maps to its authoritative frame (the Tier-A destructure, all arms).
        assert_eq!(
            frame_of_realm(RealmId::Planet(4), None),
            FrameRef::PlanetCentered { planet_seed: 4 }
        );
        assert_eq!(
            frame_of_realm(RealmId::System(5), None),
            FrameRef::SystemSpace { system_seed: 5 }
        );
        let hull = EntityId::pack(EntityKind::Player, 1, 1, 1);
        assert_eq!(
            frame_of_realm(RealmId::Ship(hull), None),
            FrameRef::ShipLocal { ship: hull }
        );
        assert_eq!(
            frame_of_realm(RealmId::Station(6), None),
            FrameRef::StationLocal { station_seed: 6 }
        );
        // An Area under a Planet parent carries the parent's planet seed.
        assert_eq!(
            frame_of_realm(RealmId::Area(9), Some(RealmId::Planet(7))),
            FrameRef::AreaLocal {
                planet_seed: 7,
                area_seed: 9
            }
        );
        // An Area with a NON-planet parent (or no parent) defaults the planet seed to 0 — the
        // `_ => 0` arm of parent_planet_seed.
        assert_eq!(
            frame_of_realm(RealmId::Area(9), Some(RealmId::System(3))),
            FrameRef::AreaLocal {
                planet_seed: 0,
                area_seed: 9
            }
        );
        assert_eq!(
            frame_of_realm(RealmId::Area(9), None),
            FrameRef::AreaLocal {
                planet_seed: 0,
                area_seed: 9
            }
        );
    }

    #[test]
    fn a_projected_box_carries_its_realm_frame() {
        // The projection captures each realm's authoritative frame on its box (so the render glue
        // composes through the chokepoint without reconstructing a FrameRef).
        let scene = RealmScene::from_boundaries(&[shell_boundary(
            RealmId::System(7),
            DVec3::ZERO,
            1000.0,
            None,
        )])
        .expect("projects");
        assert_eq!(
            scene.get(RealmId::System(7)).expect("box").frame,
            FrameRef::SystemSpace { system_seed: 7 }
        );
    }

    #[test]
    fn a_duplicate_realm_id_is_rejected() {
        let err = RealmScene::from_boundaries(&[
            shell_boundary(RealmId::System(7), DVec3::ZERO, 100.0, None),
            aabb_boundary(RealmId::System(7), DVec3::splat(5.0), None),
        ])
        .expect_err("duplicate realm must reject");
        assert_eq!(err, SceneError::DuplicateRealm);
    }

    #[test]
    fn depth_walks_the_parent_chain_three_deep() {
        // station ⊃ ship ⊃ player: depths 0/1/2, computed via map LOOKUP (order-independent).
        let station = RealmId::Station(1);
        let ship = RealmId::System(2); // any distinct realm id stands in for the middle
        let player = RealmId::Planet(3);
        let scene = RealmScene::from_boundaries(&[
            // Deliberately NOT in depth order, to prove order-independence.
            aabb_boundary(player, DVec3::splat(1.0), Some(ship)),
            aabb_boundary(station, DVec3::splat(100.0), None),
            aabb_boundary(ship, DVec3::splat(10.0), Some(station)),
        ])
        .expect("projects");
        assert_eq!(scene.get(station).expect("station").depth, 0);
        assert_eq!(scene.get(ship).expect("ship").depth, 1);
        assert_eq!(scene.get(player).expect("player").depth, 2);
    }

    #[test]
    fn a_parent_link_to_an_out_of_scene_realm_is_a_root() {
        // The `Some(None-in-map)` terminal arm of depth_of: a parent not present in the set ends
        // the chain (that realm is a root for depth purposes) — depth 0.
        let scene = RealmScene::from_boundaries(&[aabb_boundary(
            RealmId::Station(5),
            DVec3::splat(1.0),
            Some(RealmId::System(99)), // System(99) is NOT in the set
        )])
        .expect("projects");
        assert_eq!(scene.get(RealmId::Station(5)).expect("box").depth, 0);
    }

    #[test]
    fn a_parent_cycle_is_a_bounded_loud_error() {
        // A ⇄ B cycle: the walk must stop loud at MAX_NEST_DEPTH, never loop forever (H7 guard).
        let a = RealmId::System(1);
        let b = RealmId::System(2);
        let err = RealmScene::from_boundaries(&[
            aabb_boundary(a, DVec3::splat(1.0), Some(b)),
            aabb_boundary(b, DVec3::splat(1.0), Some(a)),
        ])
        .expect_err("a cycle must reject");
        assert_eq!(err, SceneError::CycleOrDepthExceeded);
    }

    #[test]
    fn stable_seed_is_kind_agnostic_and_deterministic() {
        // Deterministic: the same realm always seeds the same.
        assert_eq!(
            stable_seed(RealmId::System(5)),
            stable_seed(RealmId::System(5))
        );
        // KIND-agnostic: System(5) and Planet(5) share a payload but differ in KIND, and hash
        // DIFFERENTLY (the discriminant byte differs) — no per-kind arm collides them.
        assert_ne!(
            stable_seed(RealmId::System(5)),
            stable_seed(RealmId::Planet(5))
        );
        // Distinct payloads within one kind also differ.
        assert_ne!(
            stable_seed(RealmId::System(5)),
            stable_seed(RealmId::System(6))
        );
        // Station/Area (the appended arms) hash without a special case.
        assert_ne!(
            stable_seed(RealmId::Station(1)),
            stable_seed(RealmId::Area(1))
        );
    }

    #[test]
    fn fnv1a_is_the_documented_offset_basis_for_the_empty_input() {
        // The empty-slice branch of the byte loop (no iterations) returns the offset basis.
        assert_eq!(fnv1a(&[]), FNV_OFFSET);
        // A one-byte input applies exactly one xor+mul step.
        assert_eq!(fnv1a(&[0]), FNV_OFFSET.wrapping_mul(FNV_PRIME));
    }

    #[test]
    fn color_from_seed_is_translucent_deterministic_and_in_gamut() {
        let c = color_from_seed(stable_seed(RealmId::System(7)));
        assert_eq!(c, color_from_seed(stable_seed(RealmId::System(7))));
        assert_eq!(c[3], BOX_ALPHA, "alpha is the translucent const");
        for ch in &c[..3] {
            assert!((0.0..=1.0).contains(ch), "channel {ch} in gamut");
        }
        // Distinct realms get distinct colors (the hue spread).
        assert_ne!(
            color_from_seed(stable_seed(RealmId::System(7))),
            color_from_seed(stable_seed(RealmId::System(8)))
        );
    }

    #[test]
    fn hsv_to_rgb_covers_every_sector() {
        // Six representative hues, one per sector 0..5, plus the h==1.0 fold-back (`_` arm).
        let sector_hues = [0.02, 0.19, 0.35, 0.52, 0.69, 0.85, 1.0];
        for h in sector_hues {
            let [r, g, b] = hsv_to_rgb(h, 0.65, 0.95);
            for ch in [r, g, b] {
                assert!((0.0..=1.0).contains(&ch), "hue {h} channel {ch} in gamut");
            }
        }
        // Pure primaries pin two specific sector arms by value.
        assert_eq!(hsv_to_rgb(0.0, 1.0, 1.0), [1.0, 0.0, 0.0]); // sector 0: red
        // s==0 → grey (v,v,v) regardless of hue (p==q==t==v).
        assert_eq!(hsv_to_rgb(0.4, 0.0, 0.5), [0.5, 0.5, 0.5]);
    }

    #[test]
    fn box_lowers_to_one_cuboid_prim_scaled_by_the_half_extents() {
        let rbox = RealmBox {
            shape: BoxShape::Box {
                half: DVec3::new(2.0, 3.0, 4.0),
            },
            frame: FrameRef::SystemSpace { system_seed: 1 },
            center_offset: DVec3::new(1.0, 0.0, 0.0),
            parent: None,
            depth: 0,
            color_rgba: [0.1, 0.2, 0.3, BOX_ALPHA],
        };
        let prims = to_render_prims(&rbox, DVec3::new(10.0, 0.0, 0.0));
        assert_eq!(prims.len(), 1);
        let p = &prims[0];
        // 6 faces × 2 tris × 3 verts = 36.
        assert_eq!(p.vertices.len(), 36);
        assert_eq!(p.color_rgba, [0.1, 0.2, 0.3, BOX_ALPHA]);
        assert_eq!(p.transform.scale, [2.0, 3.0, 4.0]);
        // world_center (10) + center_offset (1) = 11 on x.
        assert_eq!(p.transform.translation, [11.0, 0.0, 0.0]);
        // Every cuboid vertex is a unit-cube corner with a unit face normal. `abs()==1.0` is a
        // single condition (no `||` short-circuit branch — HR5: no uncoverable region in a helper).
        for v in &p.vertices {
            for c in v.pos {
                assert_eq!(c.abs(), 1.0, "cuboid corner component ±1");
            }
            let n = DVec3::new(v.normal[0] as f64, v.normal[1] as f64, v.normal[2] as f64);
            assert!((n.length() - 1.0).abs() < 1e-6, "unit face normal");
        }
    }

    #[test]
    fn sphere_lowers_to_one_prim_of_unit_positions_scaled_by_radius() {
        let rbox = RealmBox {
            shape: BoxShape::Sphere { r: 5.0 },
            frame: FrameRef::SystemSpace { system_seed: 1 },
            center_offset: DVec3::ZERO,
            parent: None,
            depth: 0,
            color_rgba: [0.4, 0.5, 0.6, BOX_ALPHA],
        };
        let prims = to_render_prims(&rbox, DVec3::new(0.0, 7.0, 0.0));
        assert_eq!(prims.len(), 1);
        let p = &prims[0];
        assert_eq!(p.transform.scale, [5.0, 5.0, 5.0]);
        assert_eq!(p.transform.translation, [0.0, 7.0, 0.0]);
        // Deterministic vertex count: caps contribute 1 tri per sector, middle stacks 2.
        // stacks=8, sectors=12 → top cap 12 tris + bottom cap 12 tris + 6 middle stacks × 12 × 2.
        let expected_tris =
            SPHERE_SECTORS + SPHERE_SECTORS + (SPHERE_STACKS - 2) * SPHERE_SECTORS * 2;
        assert_eq!(p.vertices.len(), expected_tris * 3);
        // Every sphere vertex is on the unit sphere and its normal equals its position.
        for v in &p.vertices {
            let pos = DVec3::new(v.pos[0] as f64, v.pos[1] as f64, v.pos[2] as f64);
            assert!((pos.length() - 1.0).abs() < 1e-6, "on the unit sphere");
            assert_eq!(v.pos, v.normal, "sphere normal == unit position");
        }
    }

    #[test]
    fn iter_yields_boxes_in_realmid_order() {
        let scene = RealmScene::from_boundaries(&[
            aabb_boundary(RealmId::System(9), DVec3::splat(1.0), None),
            aabb_boundary(RealmId::Planet(2), DVec3::splat(1.0), None),
        ])
        .expect("projects");
        let realms: Vec<RealmId> = scene.iter().map(|(r, _)| r).collect();
        // RealmId Ord: Planet(_) < System(_) by declaration order.
        assert_eq!(realms, vec![RealmId::Planet(2), RealmId::System(9)]);
    }

    #[test]
    fn an_empty_boundary_set_projects_to_an_empty_scene() {
        let scene = RealmScene::from_boundaries(&[]).expect("empty projects");
        assert!(scene.is_empty());
        assert_eq!(scene.len(), 0);
        assert!(scene.get(RealmId::System(1)).is_none());
        assert_eq!(scene, RealmScene::default());
    }

    #[test]
    fn from_boxes_json_loads_the_same_scene_as_the_boundaries_it_serializes() {
        // The dev-config path is SINGLE-SOURCED with the shard plant: the same `Vec<RealmBoundary>`
        // serialized to JSON must load to the byte-identical scene `from_boundaries` builds.
        let boundaries = vec![
            shell_boundary(RealmId::System(7), DVec3::new(1.0, 2.0, 3.0), 1000.0, None),
            aabb_boundary(RealmId::Station(9), DVec3::new(4.0, 5.0, 6.0), None),
        ];
        let json = serde_json::to_string(&boundaries).expect("serialize boundaries");
        let from_json = RealmScene::from_boxes_json(&json).expect("loads");
        let from_vec = RealmScene::from_boundaries(&boundaries).expect("projects");
        assert_eq!(from_json, from_vec, "the dev-config load matches the plant");
        // And it really carries the boxes (not a silent empty).
        assert_eq!(from_json.len(), 2);
        assert_eq!(
            from_json.get(RealmId::System(7)).expect("system").shape,
            BoxShape::Sphere { r: 1000.0 }
        );
    }

    #[test]
    fn from_boxes_json_rejects_malformed_json_loud() {
        // Not JSON at all → a MalformedJson error carrying the serde message (never a silent empty).
        // Discriminant equality (not `assert!(matches!(..))`, whose `_ => false` arm is an
        // uncoverable region — CLAUDE.md HR5) to check the variant without the (varying) payload.
        let malformed = std::mem::discriminant(&SceneError::MalformedJson(String::new()));
        let err =
            RealmScene::from_boxes_json("{not valid json").expect_err("malformed must reject");
        assert_eq!(std::mem::discriminant(&err), malformed, "got {err:?}");
        // Valid JSON but the WRONG shape (an object, not a RealmBoundary array) also rejects.
        let err = RealmScene::from_boxes_json("{}").expect_err("wrong shape must reject");
        assert_eq!(std::mem::discriminant(&err), malformed, "got {err:?}");
    }

    #[test]
    fn from_boxes_json_propagates_a_duplicate_realm_from_the_projection() {
        // A well-formed JSON array that still violates the projection contract (duplicate realm)
        // surfaces the SAME `from_boundaries` error through the JSON entrypoint.
        let boundaries = vec![
            shell_boundary(RealmId::System(7), DVec3::ZERO, 100.0, None),
            aabb_boundary(RealmId::System(7), DVec3::splat(5.0), None),
        ];
        let json = serde_json::to_string(&boundaries).expect("serialize");
        let err = RealmScene::from_boxes_json(&json).expect_err("duplicate must reject");
        assert_eq!(err, SceneError::DuplicateRealm);
    }

    #[test]
    fn scene_error_variants_render_their_loud_display_messages() {
        // The thiserror `#[error(...)]` Display arms — rendered so a bad dev-config `boxes.json`
        // fails LOUD at load (never a silent empty). The other tests compare by value/discriminant
        // and never format, so without this the Display arms stay uncovered.
        assert_eq!(
            SceneError::DuplicateRealm.to_string(),
            "duplicate realm id in the boundary set"
        );
        assert_eq!(
            SceneError::CycleOrDepthExceeded.to_string(),
            "parent chain cycles or exceeds the max nesting depth"
        );
        assert_eq!(
            SceneError::MalformedJson("bad".into()).to_string(),
            "malformed boxes.json: bad"
        );
        // Exercise the derived Clone + PartialEq FIELD comparison for the payload variant (the two
        // regions V2's `MalformedJson(String)` added — the other tests compare unit variants or use
        // `matches!`, so the String-carrying arm's clone/eq stays uncovered without this).
        let m = SceneError::MalformedJson("x".into());
        assert_eq!(m.clone(), m);
        assert_ne!(m, SceneError::MalformedJson("y".into()));
    }
}
