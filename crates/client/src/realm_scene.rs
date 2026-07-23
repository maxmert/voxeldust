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
use vd_core::geometry::{Boundary, RealmBoundary, RealmRegion};
use vd_core::pose::{FrameRef, RealmId};
use vd_core::worldgen::MAX_RENDERABLE_EXTENT_M;

use crate::realm_view::RealmView;

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
                    color_rgba: color_for_realm(b.realm),
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

    /// Project the SEED-DERIVED containment [`RealmRegion`] forest (`worldgen::realm_regions_for`) into
    /// the render scene — the C-6b SINGLE-SOURCE so the client draws EXACTLY the sim's containment
    /// geometry (no authored `boxes.json`, no drift). Only FINITE LEAF realms are drawn: a region whose
    /// shape extent (`Boundary::finite_extent`) is `<= worldgen::MAX_RENDERABLE_EXTENT_M` (systems r=40,
    /// planets r=10) becomes a box; the ~unbounded ambient shells (Galaxy r=1000, Universe r=1e9) are
    /// SKIPPED — the between-space is FELT, not framed. Depth is computed over the FULL forest's parent
    /// links (so a rendered System keeps its true nesting depth even though its Galaxy parent is skipped),
    /// then only the renderable subset is kept. A `RealmRegion` has NO `to_realm`/`effect` to leak (unlike
    /// `RealmBoundary`), so this is the cleaner projection; it produces the identical [`RealmBox`] type.
    ///
    /// # Errors
    /// [`SceneError::DuplicateRealm`] on a repeated realm id (the seed forest guarantees uniqueness — a
    /// duplicate is a generator bug); [`SceneError::CycleOrDepthExceeded`] on a cyclic/over-deep chain.
    pub fn from_regions(regions: &[RealmRegion]) -> Result<RealmScene, SceneError> {
        // Pass 1: the parent map over the WHOLE forest (renderable + ambient), rejecting duplicates — so
        // the depth walk is a LOOKUP over the true nesting, not just the renderable subset.
        let mut parents: BTreeMap<RealmId, Option<RealmId>> = BTreeMap::new();
        for r in regions {
            if parents.insert(r.realm, r.parent).is_some() {
                return Err(SceneError::DuplicateRealm);
            }
        }
        // Pass 2: project ONLY the finite renderable regions (skip the ambient Galaxy/Universe shells),
        // computing depth over the FULL map so a rendered System keeps its true depth.
        let mut boxes: BTreeMap<RealmId, RealmBox> = BTreeMap::new();
        for r in regions {
            if r.shape.finite_extent() > MAX_RENDERABLE_EXTENT_M {
                continue; // ambient (non-renderable) shell — felt, not framed
            }
            let depth = depth_of(r.realm, &parents)?;
            boxes.insert(
                r.realm,
                RealmBox {
                    shape: shape_of(r.shape),
                    frame: r.frame,
                    center_offset: r.center.offset(),
                    parent: r.parent,
                    depth,
                    color_rgba: color_for_realm(r.realm),
                },
            );
        }
        Ok(RealmScene(boxes))
    }

    /// Project a `regions.json` dev-config (a JSON array of [`RealmRegion`] — the IDENTICAL forest the
    /// shard computes from `worldgen::realm_regions_for(seed)` and plants into its `RealmRegions`) into the
    /// render scene. The C-6b SINGLE-SOURCE for the playground `--realm-boxes`: the client draws EXACTLY the
    /// sim's containment geometry (byte-identical to what the shard's detector evaluates). Deserializes then
    /// delegates to [`RealmScene::from_regions`] — a malformed file and a duplicate/cyclic set both fail
    /// LOUD ([`SceneError`]), never a silent empty scene.
    ///
    /// # Errors
    /// [`SceneError::MalformedJson`] if the text is not a valid `RealmRegion` array;
    /// [`SceneError::DuplicateRealm`] / [`SceneError::CycleOrDepthExceeded`] as [`RealmScene::from_regions`].
    pub fn from_regions_json(json: &str) -> Result<RealmScene, SceneError> {
        let regions = parse_regions(json)?;
        RealmScene::from_regions(&regions)
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

    /// Overlay a [`RealmView`]'s streamed live placements onto this BOOT scene (D-45(a) FA-2c-3.3): each
    /// boot box whose realm the feed has streamed gets its `frame` AND `center_offset` REPLACED by the
    /// latest server-shipped pose. BOTH move because the pose is authored in the shard's PARENT frame
    /// (`SystemSpace` for a planet), NOT the realm's own boot frame (`PlanetCentered`) — the boot frame is
    /// only correct for a STATIC realm; a MOVING realm must render in the frame it was authored in (this is
    /// invisible through P3 where all such frames are world-origin identity, but load-bearing at P4/P5).
    /// Shape / parent / depth / color are boot config and are kept. A boot box the feed never names stays
    /// boot-static. Callers skip this when the view is empty (walk scale → the boot scene, byte-identical).
    #[must_use]
    pub fn overlaid(&self, view: &RealmView) -> RealmScene {
        let boxes = self
            .0
            .iter()
            .map(|(&realm, boot)| {
                let overlaid = match view.realm_latest(realm) {
                    Some(live) => RealmBox {
                        frame: live.frame,
                        center_offset: live.pos,
                        ..*boot
                    },
                    None => *boot,
                };
                (realm, overlaid)
            })
            .collect();
        RealmScene(boxes)
    }
}

/// Parse a `boxes.json` text into a `Vec<RealmBoundary>`, mapping the serde error into an owned
/// [`SceneError::MalformedJson`] — a monomorphic helper so the fallible decode + error map live
/// here, off [`RealmScene::from_boxes_json`] (which stays a straight-line delegate).
fn parse_boundaries(json: &str) -> Result<Vec<RealmBoundary>, SceneError> {
    serde_json::from_str(json).map_err(|e| SceneError::MalformedJson(e.to_string()))
}

/// Parse a `regions.json` text into a `Vec<RealmRegion>`, mapping the serde error into an owned
/// [`SceneError::MalformedJson`] — the twin of [`parse_boundaries`] for the C-6b seed-forest single-source
/// (all fallible decode + error map here, so [`RealmScene::from_regions_json`] stays a straight-line delegate).
fn parse_regions(json: &str) -> Result<Vec<RealmRegion>, SceneError> {
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

/// The value (brightness) tint spread (FA-5 visual test): a realm's brightness is nudged DOWN by up to
/// this much by its seed, so co-located realms of one kind stay distinguishable WITHOUT leaving the role's
/// hue family (every Planet stays blue). Small enough that the family hue always reads.
const VALUE_TINT_SPREAD: f64 = 0.30;

/// A TRANSLUCENT RGBA render color for a realm by its ROLE (kind), tinted within the role's family by
/// seed. Planets are the BLUE family (the visual-universe test's blue spheres); each other role gets its
/// own base hue. PURE + cosmetic — the same realm always yields the same color (determinism is load-bearing
/// for captures) and the look is a pure CLIENT law of the realm's kind, NEVER on the wire (the server ships
/// no color; a realm's appearance is not authority).
#[must_use]
pub fn color_for_realm(realm: RealmId) -> [f32; 4] {
    let (hue, sat, val_base) = role_hsv(realm);
    // Distinguish within the family by nudging VALUE (brightness) by a small seed amount — the HUE (the
    // family) is FIXED, so every Planet stays blue. `seed_unit ∈ [0,1)` keeps `val` in a legible band.
    let val = val_base - VALUE_TINT_SPREAD * seed_unit(stable_seed(realm));
    let [r, g, b] = hsv_to_rgb(hue, sat, val);
    [r, g, b, BOX_ALPHA]
}

/// The base `(hue, sat, val)` for a realm ROLE — the ONE place a realm KIND maps to a look (a cosmetic
/// table, not a feature branch — HR3-safe because it drives only rendering, never behaviour). Planet =
/// blue; System (also the Galaxy/Universe `System` stand-ins) = warm star; Ship = amber; Station = steel;
/// Area = green. Monomorphic (the kind match is covered here, off the pure [`color_for_realm`]).
fn role_hsv(realm: RealmId) -> (f64, f64, f64) {
    match realm {
        RealmId::Planet(_) => (0.60, 0.75, 0.95),
        RealmId::System(_) => (0.13, 0.55, 0.98),
        RealmId::Ship(_) => (0.08, 0.80, 0.95),
        RealmId::Station(_) => (0.58, 0.10, 0.85),
        RealmId::Area(_) => (0.33, 0.60, 0.88),
    }
}

/// A seed → `[0,1)` unit for the brightness tint, via the shared `SplitMix64` avalanche (deterministic +
/// portable — determinism is load-bearing for captures). Reused, NOT a raw `seed >> 11`: a bare shift can
/// COLLAPSE adjacent small-input hashes (the visual test's `Planet(7)`/`Planet(8)` FNV seeds differ only
/// in bits the shift drops), giving two planets the identical shade; the avalanche spreads any bit.
fn seed_unit(seed: u64) -> f64 {
    vd_core::rng::SplitMix64::new(seed).next_f64()
}

/// HSV→RGB for `h,s,v ∈ [0,1]` → linear-ish `[r,g,b] ∈ [0,1]` (the standard 6-sector formula).
/// A monomorphic helper so every sector branch is covered here, off the pure [`color_for_realm`].
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
            // Wound (a,d,b)/(b,d,c) → CCW when viewed from OUTSIDE (front face outward), matching the
            // outward vertex normals AND the cuboid convention, so `cull_mode: Face::Back` uniformly
            // culls the inside: the container realm you sit inside (System SOI) drops away instead of
            // washing the whole view with its translucent tint.
            // Top cap (i==0): the a/d row collapses to the pole → one triangle (b,pole,c).
            if i != 0 {
                push_tri(&mut verts, a, d, b);
            }
            // Bottom cap (i==STACKS-1): the b/c row collapses to the pole → one triangle (a,d,pole).
            if i != SPHERE_STACKS - 1 {
                push_tri(&mut verts, b, d, c);
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
    fn overlaid_moves_a_streamed_box_frame_and_offset_keeps_static_and_empty_is_boot_identical() {
        use vd_core::pose::StampedPose;
        use vd_core::{TickId, UniverseTick};
        use vd_wire::channels::{RealmSnap, RealmSnapshotDatagram, SubId};
        // A boot scene: Planet 1 (boot frame PlanetCentered) + Station 2, both boot-static.
        let boot = RealmScene::from_boundaries(&[
            shell_boundary(
                RealmId::Planet(1),
                DVec3::new(20.0, 0.0, 0.0),
                5.0,
                Some(RealmId::System(7)),
            ),
            shell_boundary(
                RealmId::Station(2),
                DVec3::new(-25.0, 0.0, 0.0),
                3.0,
                Some(RealmId::System(7)),
            ),
        ])
        .expect("boot scene");
        // Stream a LIVE pose for Planet 1 ONLY, authored in the PARENT (System) frame at a new offset.
        let streamed_frame = FrameRef::SystemSpace { system_seed: 7 };
        let streamed_offset = DVec3::new(1.0e9, 5.0e8, 0.0);
        let mut view = RealmView::default();
        view.on_realm_snapshot(RealmSnapshotDatagram {
            sub: SubId(0),
            frame_id: 1,
            source_tick: TickId(1),
            universe_tick: UniverseTick(10),
            realms: vec![RealmSnap {
                realm: RealmId::Planet(1),
                pose: StampedPose::at_rest(streamed_frame, streamed_offset, UniverseTick(10)),
            }],
        });
        let scene = boot.overlaid(&view);
        // Planet 1 moved — BOTH its frame and center_offset are the streamed (parent-frame) values, and
        // its frame CHANGED from the boot PlanetCentered (the must-fix: a moving realm renders in the
        // frame it was authored in, not its own boot frame).
        let moved = scene.get(RealmId::Planet(1)).expect("planet box");
        assert_eq!(moved.center_offset, streamed_offset);
        assert_eq!(moved.frame, streamed_frame);
        assert_ne!(
            moved.frame,
            boot.get(RealmId::Planet(1)).expect("boot planet").frame,
            "the overlay REPLACES the boot frame (PlanetCentered → the authored SystemSpace)",
        );
        // Station 2 (not streamed) stays boot-static.
        assert_eq!(
            scene.get(RealmId::Station(2)),
            boot.get(RealmId::Station(2))
        );
        // An EMPTY view overlays to the boot scene byte-identical (the walk-scale case).
        assert_eq!(boot.overlaid(&RealmView::default()), boot);
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
    fn color_for_realm_is_blue_for_planets_role_based_translucent_and_in_gamut() {
        // Every ROLE (kind) is exercised (covers all `role_hsv` arms): translucent + in-gamut + deterministic.
        let realms = [
            RealmId::Planet(7),
            RealmId::System(7),
            RealmId::Ship(vd_core::ids::EntityId(1)),
            RealmId::Station(7),
            RealmId::Area(7),
        ];
        for realm in realms {
            let c = color_for_realm(realm);
            assert_eq!(c[3], BOX_ALPHA, "alpha is the translucent const");
            for ch in &c[..3] {
                assert!((0.0..=1.0).contains(ch), "channel {ch} in gamut");
            }
            assert_eq!(
                color_for_realm(realm),
                c,
                "deterministic (load-bearing for captures)"
            );
        }
        // A PLANET is BLUE — the blue channel dominates red + green (the visual test's blue spheres).
        let [pr, pg, pb, _] = color_for_realm(RealmId::Planet(7));
        assert!(pb > pr, "planet blue > red");
        assert!(pb > pg, "planet blue > green");
        // A same-kind realm stays in the family (still blue) but is a distinguishable shade.
        let [qr, qg, qb, _] = color_for_realm(RealmId::Planet(8));
        assert!(qb > qr, "planet 8 still blue > red");
        assert!(qb > qg, "planet 8 still blue > green");
        assert_ne!(
            color_for_realm(RealmId::Planet(7)),
            color_for_realm(RealmId::Planet(8)),
            "seed tints the shade within the blue family",
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
    fn from_regions_draws_only_finite_leaf_realms_and_skips_the_ambient_shells() {
        // C-6b SINGLE-SOURCE: the client's scene is projected from the SAME seed forest the sim's
        // containment detector consumes (`worldgen::realm_regions_for`). The finite realms are drawn —
        // System 7/8 (r=40), Planet 7 (r=10), Station 7 (half=5), Area 7 (half=3) — AND the Galaxy (r=180)
        // as the CONTAINING box around the systems, so an entity in the between-space is visibly still
        // inside a realm (never orphaned). Only the ~unbounded Universe (r=1e9) is SKIPPED (extent > thresh).
        let regions = vd_core::worldgen::realm_regions_for(0);
        let scene = RealmScene::from_regions(&regions).expect("the seed forest projects");
        // The finite renderable realms are present.
        assert!(scene.get(RealmId::System(7)).is_some(), "System 7 renders");
        assert!(scene.get(RealmId::System(8)).is_some(), "System 8 renders");
        assert!(scene.get(RealmId::Planet(7)).is_some(), "Planet 7 renders");
        assert!(
            scene.get(RealmId::Station(7)).is_some(),
            "Station 7 renders"
        );
        assert!(scene.get(RealmId::Area(7)).is_some(), "Area 7 renders");
        // The Galaxy IS drawn now — the CONTAINING box (a Sphere of its radius) at depth 1, enclosing both
        // systems; an entity in the gap between them is visibly inside it (never orphaned).
        let galaxy = scene
            .get(RealmId::System(1))
            .expect("the Galaxy renders as the containing box");
        assert_eq!(galaxy.shape, BoxShape::Sphere { r: 180.0 });
        assert_eq!(galaxy.depth, 1, "the Galaxy is depth 1 (Universe ⊃ Galaxy)");
        // Only the ~unbounded ambient Universe root is SKIPPED (felt, not framed).
        assert!(
            scene.get(RealmId::System(0)).is_none(),
            "the Universe ambient root is NOT rendered"
        );
        assert_eq!(
            scene.len(),
            6,
            "the 5 finite leaf realms + the Galaxy containing box"
        );
        // A rendered System keeps its TRUE nesting depth (Universe 0 ⊃ Galaxy 1 ⊃ System 2), even though
        // its Galaxy parent is skipped from the drawn set — depth is over the FULL forest.
        assert_eq!(
            scene.get(RealmId::System(7)).expect("system 7 box").depth,
            2,
            "System 7 is depth 2 (Universe ⊃ Galaxy ⊃ System) even with the ambient parents skipped"
        );
        assert_eq!(
            scene.get(RealmId::Planet(7)).expect("planet 7 box").depth,
            3,
            "Planet 7 is depth 3 (… ⊃ System ⊃ Planet)"
        );
        // A System renders as a sphere of its SOI radius; the box carries the realm's own frame.
        assert_eq!(
            scene.get(RealmId::System(7)).expect("system 7 box").shape,
            BoxShape::Sphere { r: 40.0 }
        );
        assert_eq!(
            scene.get(RealmId::System(7)).expect("system 7 box").frame,
            FrameRef::SystemSpace { system_seed: 7 }
        );
        // The Aabb→Box projection is the "client render is FREE" proof: a Station/Area PLANTED as a first-
        // class box realm draws — with ZERO station/area-specific render code — as a Box of its half-extents
        // at its true forest depth (Station 3 under System 7; Area 4 under Planet 7, the deepest realm).
        let station = scene.get(RealmId::Station(7)).expect("station 7 box");
        assert_eq!(
            station.shape,
            BoxShape::Box {
                half: DVec3::splat(5.0)
            }
        );
        assert_eq!(
            station.depth, 3,
            "Station 7 is depth 3 (… ⊃ System ⊃ Station)"
        );
        assert_eq!(
            station.frame,
            FrameRef::StationLocal { station_seed: 7 },
            "the Station box carries its own StationLocal frame"
        );
        let area = scene.get(RealmId::Area(7)).expect("area 7 box");
        assert_eq!(
            area.shape,
            BoxShape::Box {
                half: DVec3::splat(3.0)
            }
        );
        assert_eq!(
            area.depth, 4,
            "Area 7 is depth 4 (… ⊃ Planet ⊃ Area) — the deepest realm"
        );
        assert_eq!(
            area.frame,
            FrameRef::AreaLocal {
                planet_seed: 7,
                area_seed: 7
            },
            "the Area box carries its AreaLocal frame (parent planet seed 7 from the Planet parent link)"
        );
    }

    #[test]
    fn from_regions_json_loads_the_same_scene_as_the_seed_forest_it_serializes() {
        // The playground `--realm-boxes` single-source: the seed forest serialized to `regions.json` loads
        // to the byte-identical scene `from_regions` builds directly — the client draws EXACTLY the sim's
        // containment geometry.
        let regions = vd_core::worldgen::realm_regions_for(0);
        let json = serde_json::to_string(&regions).expect("serialize regions");
        let from_json = RealmScene::from_regions_json(&json).expect("loads");
        let from_vec = RealmScene::from_regions(&regions).expect("projects");
        assert_eq!(
            from_json, from_vec,
            "the regions.json load matches the plant"
        );
        assert_eq!(
            from_json.len(),
            6,
            "the 5 finite leaf realms + the Galaxy containing box"
        );
    }

    #[test]
    fn from_regions_json_rejects_malformed_json_loud() {
        // Not a RealmRegion array → a MalformedJson error (never a silent empty). Discriminant equality
        // (HR5: matches!'s _ => false arm is uncoverable).
        let malformed = std::mem::discriminant(&SceneError::MalformedJson(String::new()));
        let err =
            RealmScene::from_regions_json("{not valid json").expect_err("malformed must reject");
        assert_eq!(std::mem::discriminant(&err), malformed, "got {err:?}");
    }

    #[test]
    fn from_regions_rejects_a_duplicate_realm_in_the_forest() {
        // A duplicate realm in the forest is a generator bug — rejected LOUD (not silently keeping the
        // first). Covers the DuplicateRealm arm of from_regions.
        let mut regions = vd_core::worldgen::realm_regions_for(0);
        let dup = *regions.first().expect("non-empty forest");
        regions.push(dup);
        let err = RealmScene::from_regions(&regions).expect_err("a duplicate realm must reject");
        assert_eq!(err, SceneError::DuplicateRealm);
    }

    #[test]
    fn from_regions_rejects_a_cyclic_renderable_chain_loud() {
        // A FINITE (renderable) region whose parent chain CYCLES → the pass-2 depth walk stops loud at
        // MAX_NEST_DEPTH and `from_regions` propagates it (the `depth_of(..)?` Err arm — distinct from
        // pass-1's DuplicateRealm, which returns BEFORE the depth walk). Mutate the seed forest so
        // System 7 ⇄ System 8 (both finite, r=40) point at each other: no duplicate (pass 1 is clean), so
        // the cycle is caught only in the depth walk of a renderable region — exactly the `?` under test.
        let mut regions = vd_core::worldgen::realm_regions_for(0);
        regions
            .iter_mut()
            .find(|r| r.realm == RealmId::System(7))
            .expect("System 7 in the seed forest")
            .parent = Some(RealmId::System(8));
        regions
            .iter_mut()
            .find(|r| r.realm == RealmId::System(8))
            .expect("System 8 in the seed forest")
            .parent = Some(RealmId::System(7));
        let err =
            RealmScene::from_regions(&regions).expect_err("a cyclic renderable chain must reject");
        assert_eq!(err, SceneError::CycleOrDepthExceeded);
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
