//! The realm-scene projection + the box→block-mesh lowering IR.
//!
//! ONE source since the flag day (proto_minor 18, `docs/design/window_lane.md` §2.4/§2.11 —
//! owner-approved 2026-08-15/16 items 1/9/10): the COMPOSED STREAM. A
//! [`ServerControlMsg::RealmRegistry`] level (and its deltas) of [`SceneRow`]s projects to ONE
//! [`RealmScene`] (`BTreeMap<RealmId, RealmBox>`); the renderer binds to the scene and its
//! lowering [`MeshPrim`]s, never to a source. The `--realm-boxes` boot file and its JSON loaders
//! are DELETED (D-LANE-6 🟩, owner decision 10 — THE DRAW LAW): a realm that is not running
//! cannot be drawn, so the drawn set comes only from what the stream states. This crate carries
//! NO Bevy types: a `MeshPrim` is a plain vertex buffer + a translucent color + a
//! translation/scale transform, so the Tier-B renderer does `Mesh::from(prim.vertices)` with
//! ZERO shape branch.
//!
//! HR3 (one tooling, never `match` on a realm KIND): [`stable_seed`] hashes the
//! POSTCARD BYTES of the whole [`RealmId`] — discriminant + payload uniformly — so the hue
//! is a pure function of identity with no per-kind arm (adversary H3). H4: the shape→vertex
//! tessellation lives HERE in Tier-A, not in the coverage-exempt renderer.

use std::collections::BTreeMap;

use glam::DVec3;
use vd_core::geometry::Boundary;
use vd_core::pose::{LatticePos, RealmId, Tier};
use vd_wire::channels::SceneRow;

use crate::interp::stated_tier;
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

/// Which lawful author drew a box's pixels — THE DRAW LAW's two arms (owner decision 10), decided
/// by DATA PRESENCE on the row's bag, never a kind or an if-running flag. There is deliberately
/// no third variant: the wire types cannot represent a third pixel source.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BodyKind {
    /// The realm's OWN self-authored outline (`TAG_LOOK`) — a running realm draws itself.
    Look,
    /// The parent's photometric point-of-light datum (`TAG_LUMA`) — a sleeping realm appears
    /// only as its parent's placement marker. Pixel rendering (luma-driven point sprites) lands
    /// in Slice D; until then a marker is a tracked zero-extent point on the diagnosis surface.
    Marker,
}

/// One realm's render description: its render-relevant shape, its center OFFSET from the realm's
/// frame origin, its parent link (nesting), its nesting `depth` (0 = top level), and its
/// TRANSLUCENT color. Carries ONLY render-relevant fields — the `RealmBoundary`'s
/// `to_realm`/`band`/`effect` are shard-authority-internal and are dropped at projection.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RealmBox {
    /// The render-relevant shape (sphere or box). A MARKER body is a POINT — a zero-radius
    /// sphere — until Slice D's luma-driven point sprites land (the drawn footprint of a
    /// sleeping star at these distances IS sub-pixel; a placeholder mesh would be a lie).
    pub shape: BoxShape,
    /// WHICH LAWFUL AUTHOR drew this pixel (owner decision 10, THE DRAW LAW — by data presence,
    /// never a kind flag): [`BodyKind::Look`] = the realm's OWN self-authored outline (`TAG_LOOK`
    /// in the row's bag); [`BodyKind::Marker`] = its parent's photometric point-of-light datum
    /// (`TAG_LUMA`). A third source is unrepresentable in the wire types. The diagnosis surface
    /// (`DevState.realm_boxes[].body_kind`) reads this.
    pub body: BodyKind,
    /// The photometric datum `(class_code, luma_lsun)` — Slice D's point-sprite input on a
    /// MARKER body. On a LOOK body it is usually `None`, but a RUNNING star's OWN look bag
    /// lawfully carries `TAG_LUMA` beside `TAG_LOOK` (THE STAR-LOOK EXTENSION SEAM, owner
    /// ruling 2026-08-19), so a look row may state its own light.
    pub luma: Option<(u8, f64)>,
    /// THE UNIT `center`'s integer cell is counted in — the one number `draw_center` multiplies
    /// by. Stated by whoever shipped the value, never picked here: the composed level/delta row
    /// states it on `pose.frame` (§2.4: tier rides `pose.frame` explicitly), and the live
    /// per-tick overlay re-states it per pose ([`crate::interp::stated_tier`]). The old
    /// static-shape tier-inference gap is GONE: every row now carries the statement.
    pub tier: Tier,
    /// The box centre as the server shipped it — a FULL tiered position (coarse cell + fine offset),
    /// NOT a bare metre vector.
    ///
    /// WHY THE TYPE CHANGED (slice 5). This used to be a `DVec3` built by calling `.offset()` on the
    /// streamed position, i.e. the coarse half was DISCARDED at every constructor. Drawing then worked
    /// only because the whole world currently sits at cell zero — the drawn point was the centre with
    /// its cell thrown away, which equals the exact position ONLY while that cell is zero. The client
    /// must be able to CARRY the coarse half before the server ever emits a non-zero one, or every box
    /// lands wrong by a whole cell.
    ///
    /// Reduce it for drawing through the ONE chokepoint ([`RealmBox::draw_center`]), which flattens the
    /// tiered position in exact integer-cell arithmetic. Never do that arithmetic by hand here — that is
    /// the class of bug this field's type now prevents.
    pub center: LatticePos,
    /// The parent realm, when this box nests inside another (`None` at the top level).
    pub parent: Option<RealmId>,
    /// The nesting depth (0 = top level, +1 per parent hop) — the deterministic parent-walk result.
    pub depth: u8,
    /// The TRANSLUCENT render color (identity-derived hue; alpha ~0.25 so nested boxes show through).
    pub color_rgba: [f32; 4],
    /// ★ WHICH WAY THIS REALM FACES (D-MOVE-2), as its parent authored it.
    ///
    /// It was always on the wire — a realm's pose has carried a facing since the pose type existed —
    /// and the drawn row simply dropped it, because nothing a realm did could be seen from outside.
    /// A ship changes that: a hull that turns and shows the same face is not a ship, it is a marker.
    ///
    /// Identity for everything that does not turn, so every existing body draws exactly as before.
    pub facing: [f32; 4],
}

impl RealmBox {
    /// This box's centre in RENDER space — the tiered position flattened to metres, exactly.
    ///
    /// THE ONE WAY A BOX BECOMES DRAWABLE (slice 5). Every consumer — the renderer, the capture
    /// camera, the containment verdicts, the diagnosis surface — asks for this instead of doing its own
    /// arithmetic. Before, four call sites each spelled their own and two of them got it subtly wrong
    /// (dropping the coarse half, or mixing a reduced point against a raw centre), which is invisible
    /// while the world sits at cell zero and wrong the moment it does not.
    ///
    /// It used to take a server-told render ORIGIN to subtract. Nothing is subtracted now: the chain of
    /// shards restated this centre from the centre of the realm the session is standing in, the same
    /// space the occupants standing in this box are measured in, which is what makes the box and its
    /// riders agree on screen.
    ///
    /// Same primitive the entity path uses (`LatticePos::delta_m`), so a box and a player standing on
    /// it are reduced identically — and, like the entity path, it multiplies by a unit it was HANDED
    /// (`self.tier`) rather than looking one up off a frame name at the moment of drawing.
    #[must_use]
    pub fn draw_center(&self) -> DVec3 {
        self.center.delta_m(LatticePos::default(), self.tier)
    }
}

/// The source-agnostic scene: one [`RealmBox`] per realm, keyed by [`RealmId`] for a
/// deterministic (order-independent) iteration order. The single chokepoint the renderer
/// binds to — a dev-config loader and a (future) wire decoder both produce this identical shape.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct RealmScene(BTreeMap<RealmId, RealmBox>);

/// Why a composed level/delta failed to project to a [`RealmScene`].
#[derive(Clone, Debug, PartialEq, Eq, thiserror::Error)]
pub enum SceneError {
    /// Two rows named the same realm — the map key would collide, so the source is rejected
    /// loudly rather than silently keeping whichever the input happened to list first.
    #[error("duplicate realm id in the scene rows")]
    DuplicateRealm,
    /// A parent link formed a cycle, or the parent chain exceeded [`MAX_NEST_DEPTH`] — a bounded,
    /// loud stop instead of an unbounded walk (adversary H7 cycle/again-guard).
    #[error("parent chain cycles or exceeds the max nesting depth")]
    CycleOrDepthExceeded,
}

/// The nesting-depth ceiling: the parent walk stops (loud, [`SceneError::CycleOrDepthExceeded`])
/// past this many hops, so a cyclic or pathological parent chain cannot loop forever. Generous
/// for the real topology (station ⊃ ship ⊃ player is depth 2).
pub const MAX_NEST_DEPTH: u8 = 16;

/// The alpha of every projected box — TRANSLUCENT so nested boxes show through their parents
/// (the multi-mesh "mesh inside mesh" render). A named const (no magic number).
pub const BOX_ALPHA: f32 = 0.25;

impl RealmScene {
    /// Project one COMPOSED LEVEL's rows (`ServerControlMsg::RealmRegistry`, proto_minor 18 —
    /// the flag day, `docs/design/window_lane.md` §2.4) into the render scene. Per row the bag
    /// decides BY PRESENCE (THE DRAW LAW, owner decision 10): `TAG_LOOK` ⇒ a body (the realm's
    /// own outline; an outline wider than `MAX_RENDERABLE_EXTENT_M` is an ambient shell — felt,
    /// not framed — and is skipped); else `TAG_LUMA` ⇒ a marker point; else the row is TRACKED
    /// but NOT DRAWN (a missing statement means the thing is not drawn) — unknown tags are
    /// skipped, so signals extend forever with zero change here. Depth is computed over the FULL
    /// level's parent links, so a drawn realm keeps its true nesting depth even when its ambient
    /// parent is skipped.
    ///
    /// # Errors
    /// [`SceneError::DuplicateRealm`] on a repeated realm id (the composer dedups — a duplicate
    /// is a server bug); [`SceneError::CycleOrDepthExceeded`] on a cyclic/over-deep parent chain.
    pub fn from_scene_rows(rows: &[SceneRow]) -> Result<RealmScene, SceneError> {
        // Pass 1: the parent map over the WHOLE level (drawn + tracked), rejecting duplicates —
        // the depth walk is a LOOKUP over the true nesting, not just the drawn subset.
        let mut parents: BTreeMap<RealmId, Option<RealmId>> = BTreeMap::new();
        for r in rows {
            if parents.insert(r.realm, r.parent).is_some() {
                return Err(SceneError::DuplicateRealm);
            }
        }
        // Pass 2: project the DRAWN rows (a look or a marker), depth over the full map.
        let mut boxes: BTreeMap<RealmId, RealmBox> = BTreeMap::new();
        for r in rows {
            let Some(rbox) = row_box(r, depth_of(r.realm, &parents)?) else {
                continue; // no drawable statement — tracked, never drawn
            };
            boxes.insert(r.realm, rbox);
        }
        Ok(RealmScene(boxes))
    }

    /// Apply an INCREMENTAL composed update (`ServerControlMsg::RealmSceneDelta`, same epoch —
    /// the caller gates the epoch) — rows that ENTERED the drawn set (complete [`SceneRow`]s,
    /// drawable on arrival) and realms that LEFT (`removed`) — returning a NEW scene (ATOMIC: the
    /// caller swaps only on `Ok`, so a malformed delta leaves the live scene intact). A row whose
    /// bag carries no drawable statement REMOVES any previous box for that realm (its look was
    /// withdrawn — e.g. a body that fell out of the membership band keeps only its marker or
    /// nothing). Every box's nesting depth is RECOMPUTED over the merged parent links.
    ///
    /// # Errors
    /// [`SceneError::CycleOrDepthExceeded`] if the merged parent links form a cycle / over-deep
    /// chain (a server bug) — the caller counts it and keeps the previous scene, never a partial
    /// mutation.
    pub fn with_delta(
        &self,
        added: &[SceneRow],
        removed: &[RealmId],
    ) -> Result<RealmScene, SceneError> {
        let mut boxes = self.0.clone();
        for realm in removed {
            boxes.remove(realm);
        }
        for r in added {
            match row_box(r, 0) {
                Some(rbox) => {
                    boxes.insert(r.realm, rbox); // depth recomputed below
                }
                None => {
                    boxes.remove(&r.realm); // the drawable statement was withdrawn
                }
            }
        }
        // Recompute every box's depth over the merged parent links (the drawn set is bounded, so
        // a full pass is cheap and covers the out-of-order-arrival case in one place).
        let parents: BTreeMap<RealmId, Option<RealmId>> =
            boxes.iter().map(|(r, b)| (*r, b.parent)).collect();
        for (realm, b) in &mut boxes {
            b.depth = depth_of(*realm, &parents)?;
        }
        Ok(RealmScene(boxes))
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
    /// boot box whose realm the feed has streamed gets its `center` REPLACED by the latest
    /// server-shipped pose, and with it the UNIT that centre's integer cell is counted in. Shape /
    /// frame / parent / depth / color are boot config and are kept. A boot box the feed never names
    /// stays boot-static. Callers skip this when the view is empty (walk scale → the boot scene,
    /// byte-identical).
    ///
    /// WHAT CHANGED, AND WHY IT IS NOT A BEHAVIOUR CHANGE. This used to replace the box's `frame` as
    /// well, with the frame the streamed pose was authored in. Its stated reason was that a moving realm
    /// must render in the frame it was authored in — but nothing here renders in a frame; the only thing
    /// the field was ever read for downstream was `.tier()`, to pick metres-per-cell in `draw_center`.
    /// So the replacement was really about the UNIT, and it bought that at the cost of `frame` meaning
    /// the realm's own frame before the first streamed row and something else after — a field whose
    /// meaning depends on how long you have been connected. The unit now moves on its own (`tier`, taken
    /// from the pose's own label, the same statement the old code was reaching through `frame` to get),
    /// the drawn point is bit-identical, and `frame` keeps one meaning for the whole session.
    #[must_use]
    pub fn overlaid_at(&self, view: &RealmView, cursor: f64) -> RealmScene {
        let boxes = self
            .0
            .iter()
            .map(|(&realm, boot)| {
                let overlaid = match view.realm_pose(realm, cursor) {
                    Some(live) => RealmBox {
                        // The streamed pose carries its coarse cell SEPARATELY from its fine offset;
                        // recombine both. Taking `live.pos` alone (as this did) silently dropped the
                        // cell every time the feed moved a realm.
                        center: LatticePos::at(live.cell, live.pos),
                        // THE UNIT COMES WITH THE VALUE: `RenderPose::tier` was stamped from the label
                        // the shipping shard put on this very pose, so a placement that arrives counted
                        // in a different lattice than the boot shape draws at its true distance.
                        tier: live.tier,
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

/// One composed row's drawable box, or `None` when the row carries no drawable statement — THE
/// DRAW LAW's presence gate in one monomorphic place (HR5: every branch here, off the two
/// straight-line scene builders). `TAG_LOOK` wins (a running realm draws itself); an outline
/// wider than `MAX_RENDERABLE_EXTENT_M` is an ambient shell (felt, not framed); else the
/// point-of-light marker arm (look_horizon.md slice 1, Q2 APPROVED): `TAG_LUMA` (the parent's
/// photometric datum) and/or `TAG_EXTENT` (the parent's one stated radius — the presence floor:
/// a non-glowing subject still draws as a correctly-sized point instead of vanishing); the
/// marker's shape carries the stated extent, so the diagnosis surface and the sprite sizing read
/// ONE number. Else — including unknown future tags, skipped by the TLV codec's own law — the
/// row is not drawn.
fn row_box(r: &SceneRow, depth: u8) -> Option<RealmBox> {
    let (shape, body, luma) = if let Ok(outline) = vd_core::look::look_of(&r.bag) {
        // (The metre cut is DELETED — real-scale design §3.0: "is this drawable" is DATA
        // PRESENCE. An ambient realm carries `look = None` and ships no self-look at all, so
        // nothing arrives here to cull; a stated look is a statement of intent to be drawn.)
        // Ruling C: a running realm's own bag may carry TAG_LUMA — the star keeps its colour
        // through the wake handover (absent tag = no luma, never a default).
        (
            shape_of(outline),
            BodyKind::Look,
            vd_core::look::luma_of(&r.bag).ok(),
        )
    } else {
        let luma = vd_core::look::luma_of(&r.bag).ok();
        let extent = vd_core::look::extent_of(&r.bag).ok();
        if luma.is_none() && extent.is_none() {
            return None; // no drawable statement — tracked, never drawn
        }
        (
            BoxShape::Sphere {
                r: extent.unwrap_or(0.0),
            },
            BodyKind::Marker,
            luma,
        )
    };
    Some(RealmBox {
        shape,
        body,
        luma,
        // §2.4: the tier rides the row's own pose frame, stated beside the value it counts.
        tier: stated_tier(r.pose.frame),
        center: r.pose.pos,
        parent: r.parent,
        depth,
        color_rgba: color_for_realm(r.realm),
        // Straight off the authored pose, in the order the renderer wants it. The PARENT wrote this —
        // a realm never states its own facing any more than its own position (SL1).
        facing: [
            r.pose.orient.x as f32,
            r.pose.orient.y as f32,
            r.pose.orient.z as f32,
            r.pose.orient.w as f32,
        ],
    })
}

/// The nesting depth of `realm` by walking `parents` up to the root — a monomorphic, bounded
/// map-lookup walk (all branching here, so [`RealmScene::from_scene_rows`] stays a straight-line
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

// ---------------------------------------------------------------------------
// THE MARKER POINT SPRITE (window lane Slice D — `docs/design/window_lane.md` §2.8/§2.10)
// ---------------------------------------------------------------------------

/// The world radius the client draws a UNIT-luminosity (1 L☉) point source at — and the base
/// radius of the avatar dot, which `vd_client_render::DOT_RADIUS` takes FROM here so the two
/// cannot drift. THE convention it states, once: *a solar-luminosity point of light is drawn the
/// same size as one player marker*; every other luminosity scales off it by
/// [`marker_look`]'s √L law, and every point then passes through the ONE apparent-size floor
/// (`vd_client_harness::camera::marker_world_radius`), so a point of light is never sub-pixel.
///
/// A rendering convention, not a world number: nothing on the wire and nothing in the sim reads
/// it. The physically-exposed successor (an HDR apparent-magnitude exposure model) is one of the
/// owner-pending rendering decisions and is registered in DEFERRED, not guessed at here.
pub const POINT_SOURCE_BASE_RADIUS_M: f64 = vd_core::look::OCCUPANT_FIGURE_EXTENT_M;

/// The drawn sRGB of each Morgan-Keenan spectral class, indexed by the `TAG_LUMA` bag's
/// `class_code` (`vd_physics::taxonomy::SpectralClass`: `O=0, B=1, A=2, F=3, G=4, K=5, M=6`).
/// These are the blackbody colours of each class's effective-temperature band (Charity's
/// blackbody→sRGB table at ~30 000 / 15 000 / 8 500 / 6 600 / 5 700 / 4 400 / 3 200 K) — the
/// published sequence, not a picked palette. A REFLECTOR (a planet) carries its illuminator's
/// class by construction (`reflected_photometrics`), so a moon reads in its star's colour, which
/// is what reflected light does.
///
/// A class code the wire never mints (≥ 7) draws as the coolest class rather than vanishing —
/// the presence law is "never zero", so an unknown colour is still a point of light.
pub const MARKER_CLASS_SRGB: [[f32; 3]; 7] = [
    [0.608, 0.686, 1.000], // O — blue
    [0.671, 0.749, 1.000], // B — blue-white
    [0.792, 0.843, 1.000], // A — white
    [0.973, 0.969, 1.000], // F — yellow-white
    [1.000, 0.957, 0.918], // G — yellow
    [1.000, 0.839, 0.667], // K — orange
    [1.000, 0.714, 0.427], // M — red
];

/// How one MARKER body is drawn: an opaque, unlit point sprite of [`MarkerLook::color_rgba`] at
/// [`MarkerLook::base_radius_m`] — the two things the parent's `TAG_LUMA` datum lawfully states
/// about a sleeping child (§1.1 item 3b: brightness and colour, and nothing else).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct MarkerLook {
    /// The sprite's colour. OPAQUE (alpha 1.0), unlike a shell's [`BOX_ALPHA`]: a point of light
    /// emits, it does not enclose.
    pub color_rgba: [f32; 4],
    /// The sprite's base world radius BEFORE the shared apparent-size floor. `√L · `
    /// [`POINT_SOURCE_BASE_RADIUS_M`] — the equal-surface-brightness radius (a sphere whose
    /// radius goes as √L has flux ∝ L), i.e. the classic "size encodes magnitude" convention.
    /// It is the ONLY star convention the client has: the decorative backdrop field is DELETED
    /// (owner ruling 2026-08-20), so every point of light on screen is a real streamed realm.
    pub base_radius_m: f64,
}

/// The point sprite for one `TAG_LUMA` datum — the client half of THE DRAW LAW's marker arm
/// (`docs/design/window_lane.md` §2.8: "only the LOOK payload upgrades: marker ⇒ self-look").
/// Pure and monomorphic (HR5): the two branches — an out-of-range class code, and a
/// negative/NaN luminosity — are both exercised here, so the renderer stays a straight-line shim.
///
/// The apparent SIZE floor is deliberately NOT applied here: it needs the camera, and it is the
/// ONE shared `vd_client_harness::camera::marker_world_radius` both the renderer and the pixel
/// gates call, so the drawn footprint and the asserted rectangle cannot disagree.
#[must_use]
pub fn marker_look(class_code: u8, luma_lsun: f64) -> MarkerLook {
    let last = MARKER_CLASS_SRGB.len() - 1;
    let [r, g, b] = MARKER_CLASS_SRGB[usize::from(class_code).min(last)];
    MarkerLook {
        color_rgba: [r, g, b, 1.0],
        // `f64::max` returns the non-NaN side, so a malformed datum floors at zero luminosity
        // (a point at the apparent-size floor) rather than producing a NaN transform.
        base_radius_m: POINT_SOURCE_BASE_RADIUS_M * luma_lsun.max(0.0).sqrt(),
    }
}

/// One render shape's CIRCUMSCRIBED extent in metres (a sphere's radius; a box's half-diagonal —
/// the radius of the sphere around it): the ONE number the diagnosis surface (`DevRealmBox
/// .extent_m`), the camera fitting and the marker sizing all read. A marker box carries its
/// parent-stated extent as a sphere (`row_box` mints it so), so this returns that stated radius
/// exactly.
#[must_use]
pub fn shape_extent_m(shape: BoxShape) -> f64 {
    match shape {
        BoxShape::Sphere { r } => r,
        BoxShape::Box { half } => half.length(),
    }
}

/// THE MARKER'S BASE WORLD RADIUS before the shared apparent-size floor (look_horizon.md
/// slice 1, Q2 APPROVED): the LARGER of the photometric √L radius and the parent's one stated
/// circumscribed extent — so a point of light's angular size is never smaller than the thing it
/// stands for, and the marker→body handover has no size step (the pre-slice ~3.8× pop at the
/// three-pixel floor). ONE expression, shared by the renderer's sprite scale and the pixel
/// gates' rectangle (via `vd_client_harness::camera::marker_world_radius`), so the drawn
/// footprint and the asserted rectangle cannot disagree. A luma-less marker (a non-glowing
/// subject — the presence floor) sizes by its extent alone; an extent-less marker (an old
/// ★ HOW BRIGHT AND HOW BIG ONE STAR IS DRAWN — the point-source law (2026-08-29).
///
/// ★ WHY THE OLD ANSWER WAS "ALL OF THEM, THE SAME". Every star was clamped to the shared 3-pixel
/// presence floor, and the fragment stage returned a flat disc. MEASURED: at a typical neighbour
/// distance the floor is worth 4.1e14 m of drawn radius against a sun-like star's 0.5 m — the floor
/// wins by 8e14, for every star, always. So luminosity was computed, carried across the wire, and
/// then thrown away at the last step. 233 220 identical dots.
///
/// ★ WHAT A STAR ACTUALLY IS. A point. What you SEE is your own eye's blur, and that blur has the
/// same shape and width for every star — measured in PIXELS, not metres. Only its AMPLITUDE differs.
/// A bright star looks bigger because its faint outer wings stay above the visibility floor further
/// out, NOT because its core is wider. That single inversion is the whole fix.
///
/// So the sprite is a CROP of one fixed profile, cut where that profile fades below what a screen can
/// show. The idea is `godot-starlight`'s (Tiffany Bennett, MIT) — crop, never scale. Its own trick of
/// moving bright stars further away to tame them is deliberately NOT taken: that is a lie about where
/// a star is, and SL1 forbids it. Exposure is the honest knob.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct StarDraw {
    /// The sprite's half-size in PIXELS — where the profile fades below `cull_level`.
    pub crop_px: f64,
    /// The profile's peak, before the core clips. Above 1.0 the centre saturates to white and the
    /// colour survives only in the wings, which is what a bright star looks like.
    pub amplitude: f64,
    /// The star's FLUX before the response curve — exactly `L/d²` up to the exposure. Kept separate
    /// so the physics can be asserted on its own: the flux obeys the inverse-square law exactly, and
    /// the amplitude is a deliberately compressed VIEW of it.
    pub flux: f64,
}

/// The numbers the star law reads. ONE struct, no inline literals, so a look change is one place.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct StarTuning {
    /// THE EXPOSURE. Scales flux before the response curve below. A LOOK decision, not a constant.
    pub flux_gain: f64,
    /// ★ HOW HARD THE BRIGHTNESS IS COMPRESSED — the exponent of the response curve.
    ///
    /// Flux across a galaxy spans about a MILLION to one. Drawn linearly, a near star's whole sprite
    /// overflows to white: MEASURED at amplitude 2 900, the core stayed fully saturated out to 3.6 px
    /// and the halo to 9.6 px — a solid white ball 19 px across, which is exactly what the owner saw.
    ///
    /// Eyes, film and star charts are all LOGARITHMIC — that is what stellar magnitude IS. A power
    /// law with a small exponent is the same compression in closed form: at 0.3, a million-to-one
    /// flux range becomes about sixty to one in drawn brightness. Bright stars stay bright and stop
    /// being discs.
    pub response_exponent: f64,
    /// The width of the faint wings, in pixels — the eye's own blur.
    pub halo_sigma_px: f64,
    /// How much of the amplitude the wings carry.
    pub halo_weight: f64,
    /// The width of the bright core, in pixels.
    pub core_sigma_px: f64,
    /// The smallest sprite, in pixels. A star FADES below this rather than shrinking to nothing:
    /// a quad smaller than a pixel misses the pixel centre and BLINKS as the camera moves, which
    /// Stellarium and Celestia both name as the thing to avoid.
    pub min_crop_px: f64,
    /// Below this amplitude a star is not drawn. Set above one 8-bit step so a culled star was
    /// genuinely invisible, never merely dim.
    pub cull_level: f64,
}

impl Default for StarTuning {
    fn default() -> Self {
        Self {
            // ★ THE EXPOSURE — a LOOK decision, calibrated from the world's own distances and to be
            // judged by looking (owner-agreed 2026-08-29).
            //
            // MEASURED against the galaxy this world actually has: `base_radius_m` is 0.5·√L, so a
            // sun-like star gives `s = 0.5/d`, and the gain that puts amplitude at 1 is `1/s²`:
            //   nearest neighbour, ~0.24 ly   2.0e31
            //   typical spacing               5.8e34   ← chosen
            //   right across the galaxy       8.5e37
            //
            // The typical spacing is the right anchor: an ordinary star reads as an ordinary star.
            // A near neighbour then lands near 2 900, which CLIPS its core to white — correct, that
            // is what a bright near star looks like. Stars across the galaxy fall far below one
            // display step individually, which is also correct: no single one is visible from there.
            // They are not wasted, because the blending ADDS — a crowd of them is the milky band,
            // which is what a galaxy's glow actually is.
            flux_gain: 5.8e34,
            // A million-to-one flux range becomes about sixty to one drawn — bright stars are bright,
            // and none of them is a disc.
            response_exponent: 0.3,
            // TIGHT. The core is about a pixel and the halo a few — a star is a POINT seen through a
            // small blur, not a glowing sphere. A wide halo is what made the bright ones read as balls.
            halo_sigma_px: 1.6,
            halo_weight: 0.03,
            core_sigma_px: 0.6,
            min_crop_px: 1.0,
            // ★ LOW, BECAUSE FAINT STARS ARE NOT WASTE — THEY ARE THE BAND. A single star at 0.003 is
            // below one 8-bit step and invisible alone. A thousand of them along one line of sight
            // SUM to 3.0 under additive blending, and that sum is the milky glow of a galaxy seen
            // edge-on. Culling at a level where one star is invisible would delete the band with it.
            //
            // The floor still exists so a star that contributes nothing at all costs no overdraw.
            cull_level: 0.003,
        }
    }
}

/// The point-source draw law: what one star's flux comes to, and how large a sprite carries it.
///
/// `base_radius_m` is `POINT_SOURCE_BASE_RADIUS_M · √L`, so `(base/d)²` IS `L/d²` up to a constant —
/// the inverse-square law, from data the vertex already carries. No new attribute, no wire change.
#[must_use]
pub fn star_draw(base_radius_m: f64, dist_m: f64, t: &StarTuning) -> StarDraw {
    let s = base_radius_m / dist_m.max(1.0);
    let flux = s * s * t.flux_gain;
    // ★ COMPRESSED, BECAUSE SIGHT IS. See `response_exponent`: linear flux made near stars into
    // saturated white balls, because a galaxy's flux range is a million to one and a screen's is 255.
    let amplitude = flux.powf(t.response_exponent);
    if amplitude <= t.cull_level {
        // CULLED ON BRIGHTNESS, NEVER ON SIZE. A star disappears because it faded, not because it
        // became small — which is the difference between a sky that thins and one that pops.
        return StarDraw {
            crop_px: 0.0,
            amplitude: 0.0,
            flux,
        };
    }
    // WHERE THE WINGS FADE BELOW VISIBILITY, in closed form from the Gaussian:
    //   A·w·exp(-r²/2σ²) = cull   ⇒   r = σ·√(2·ln(A·w/cull))
    let ratio = amplitude * t.halo_weight / t.cull_level;
    let crop = if ratio > 1.0 {
        t.halo_sigma_px * (2.0 * ratio.ln()).sqrt()
    } else {
        0.0
    };
    StarDraw {
        crop_px: crop.max(t.min_crop_px),
        amplitude,
        flux,
    }
}

/// luma-only bag) by its √L radius alone, the floor carrying the rest.
#[must_use]
pub fn marker_base_radius_m(luma: Option<(u8, f64)>, extent_m: f64) -> f64 {
    let photometric = luma.map_or(0.0, |(class_code, luma_lsun)| {
        marker_look(class_code, luma_lsun).base_radius_m
    });
    photometric.max(extent_m.max(0.0))
}

/// THE MARKER'S DRAWN COLOUR: the photometric datum's blackbody class colour when the subject
/// glows; the box's own pure role colour (at a point of light's OPAQUE alpha) when it does
/// not — a non-glowing subject's appearance is a CLIENT cosmetic law exactly like a shell's,
/// because the parent's bag lawfully carries one radius and nothing else (the one-radius law).
#[must_use]
pub fn marker_color_rgba(role_rgba: [f32; 4], luma: Option<(u8, f64)>) -> [f32; 4] {
    match luma {
        Some((class_code, luma_lsun)) => marker_look(class_code, luma_lsun).color_rgba,
        None => {
            let [r, g, b, _] = role_rgba;
            [r, g, b, 1.0]
        }
    }
}

/// The ONE unit point-sprite vertex buffer every marker shares — the same coarse unit sphere the
/// `Sphere` proxy tessellates to (H4: the tessellation lives in Tier-A, never in the renderer).
/// The renderer builds this ONCE and scales it per marker, so a point of light never costs a mesh
/// (§2.14's tier-0 rung: "a dot never costs a mesh").
#[must_use]
pub fn point_sprite_vertices() -> Vec<Vertex> {
    unit_sphere_vertices()
}

/// The base `(hue, sat, val)` for a realm ROLE — the ONE place a realm KIND maps to a look (a cosmetic
/// table, not a feature branch — HR3-safe because it drives only rendering, never behaviour). Planet =
/// blue; System = warm star; Ship = amber; Station = steel; Area = green. The galaxy and the universe
/// have their own identities since S9 (they used to arrive here as `System` stand-ins) and are listed for
/// totality — a containment boundary is undrawable under the two-body law, so nothing renders them. Monomorphic (the kind match is covered here, off the pure [`color_for_realm`]).
fn role_hsv(realm: RealmId) -> (f64, f64, f64) {
    match realm {
        RealmId::Planet(_) => (0.60, 0.75, 0.95),
        RealmId::System(_) => (0.13, 0.55, 0.98),
        RealmId::Ship(_) => (0.08, 0.80, 0.95),
        RealmId::Station(_) => (0.58, 0.10, 0.85),
        RealmId::Area(_) => (0.33, 0.60, 0.88),
        // A star's box hue — COSMETIC ONLY (taxonomy arc §6.2 site 8): the drawn photometric
        // colour rides the look bag's TAG_LUMA (ruling C), never this fallback family.
        RealmId::Star(_) => (0.10, 0.85, 1.00),
        // The two containers, which a player never sees as a box: a galaxy and the universe are
        // BOUNDARIES, and this project's two-body law makes a containment boundary undrawable. They get a
        // colour because this table is total, not because anything renders it.
        RealmId::Galaxy(_) => (0.75, 0.35, 0.70),
        RealmId::Universe => (0.00, 0.00, 0.20),
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
    /// ★ HOW THE PRIMITIVE IS TURNED, as `[x, y, z, w]` (D-MOVE-2). Identity for everything that does
    /// not turn, so every drawn body is byte-identical to before this field existed.
    pub rotation: [f32; 4],
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
/// ONE fixed tessellation today). Named consts (no magic numbers).
///
/// NOTE (2026-08-03): this comment previously read "NO LOD". That standing ban is RETRACTED — detail
/// levels are now required (`docs/investigation/block_system_design_addendum_2.md`). Nothing here changes yet:
/// the realm proxy stays a single fixed tessellation, and it becomes the COARSEST rung of the terrain
/// detail ladder when that lands at P4, rather than an exception to a rule.
pub const SPHERE_SECTORS: usize = 12;
pub const SPHERE_STACKS: usize = 8;

/// Lower a [`RealmBox`] to its render primitives at `draw_center` — the box's centre ALREADY flattened
/// to metres by the caller, through the ONE [`RealmBox::draw_center`] chokepoint.
///
/// ONE TERM, NOT TWO (slice 5). This used to take the box's FRAME ORIGIN in world space and add
/// `rbox.center_offset` to it — a composition performed here, on the client, mixing a reduced point
/// with a raw centre whose coarse half had been discarded. Nothing is composed and nothing is
/// subtracted now: the server ships the centre already measured from the realm this session stands in,
/// the caller flattens that one position once, and this places it. Anything that needs a drawable
/// point asks the chokepoint, never arithmetic of its own.
///
/// TESSELLATES the shape into VERTICES here in Tier-A (adversary H4): a `Box` → a unit cuboid (12
/// triangles) scaled by its half-extents; a `Sphere` → a coarse UV sphere scaled by `r`. Exactly one
/// prim per box today; the block-mesh successor emits more prims through the same shape.
#[must_use]
pub fn to_render_prims(rbox: &RealmBox, draw_center: DVec3) -> Vec<MeshPrim> {
    let center = draw_center;
    let translation = [center.x as f32, center.y as f32, center.z as f32];
    match rbox.shape {
        BoxShape::Box { half } => {
            vec![MeshPrim {
                vertices: unit_cuboid_vertices(),
                color_rgba: rbox.color_rgba,
                transform: PrimTransform {
                    translation,
                    scale: [half.x as f32, half.y as f32, half.z as f32],
                    // The facing its parent authored. A box is the shape whose turning can be SEEN,
                    // which is exactly why a ship is drawn as one.
                    rotation: rbox.facing,
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
                    // A sphere looks the same whichever way it is turned, so this changes no pixel
                    // today. Carried anyway: the day a body gets a surface the facing must already
                    // be right, and a field that appears later is a field somebody forgets to fill.
                    rotation: rbox.facing,
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
mod star_law_tests {
    use super::*;

    const SUN_BASE_M: f64 = POINT_SOURCE_BASE_RADIUS_M; // sqrt(1 L_sun) = 1

    /// ★ THE DEFECT THIS LAW EXISTS TO FIX: every star drew at the same size, so luminosity was
    /// computed, shipped and discarded. A brighter star must draw BIGGER and BRIGHTER than a dim one
    /// at the same distance — the property the old presence floor destroyed.
    #[test]
    fn a_brighter_star_draws_bigger_and_brighter_at_the_same_distance() {
        let t = StarTuning::default();
        let d = 1.0e17;
        let dim = star_draw(SUN_BASE_M * (0.01_f64).sqrt(), d, &t);
        let bright = star_draw(SUN_BASE_M * (100.0_f64).sqrt(), d, &t);
        assert!(
            bright.amplitude > dim.amplitude,
            "a brighter star must be brighter: {bright:?} vs {dim:?}"
        );
        assert!(
            bright.crop_px > dim.crop_px,
            "a brighter star's wings reach further, so its sprite is larger: {bright:?} vs {dim:?}"
        );
    }

    /// ★ THE PHYSICS IS EXACT AND THE PICTURE IS COMPRESSED — two different statements, both true.
    ///
    /// The FLUX obeys the inverse-square law to the last bit: twice as far is a quarter as much light.
    /// The DRAWN brightness deliberately does not, because a galaxy's flux range is a million to one
    /// and a screen's is 255. Drawing it linearly turned near stars into saturated white balls 19 px
    /// across. Eyes, film and star charts are all logarithmic — stellar MAGNITUDE is this compression.
    #[test]
    fn the_flux_is_exactly_inverse_square_and_the_drawn_brightness_is_compressed() {
        let t = StarTuning::default();
        let near = star_draw(SUN_BASE_M, 1.0e17, &t);
        let far = star_draw(SUN_BASE_M, 2.0e17, &t);
        let flux_ratio = near.flux / far.flux;
        assert!(
            (flux_ratio - 4.0).abs() < 1.0e-9,
            "the PHYSICS must be exact inverse square: expected 4x, got {flux_ratio}"
        );
        let drawn_ratio = near.amplitude / far.amplitude;
        // Split, because one `&&` hides a short-circuit arm no test can take (HR5).
        assert!(
            drawn_ratio > 1.0,
            "the near star must still draw brighter: got {drawn_ratio}"
        );
        assert!(
            drawn_ratio < flux_ratio,
            "the PICTURE must be compressed — brighter, but by less than four: got {drawn_ratio}"
        );
    }

    /// ★ AND THE COMPRESSION MUST ACTUALLY TAME THE RANGE. A near star may clip its centre to white;
    /// it may NOT be a white disc. Pinned as the measurement that caught it: at linear flux the core
    /// stayed saturated to 3.6 px and the halo to 9.6 px — a ball 19 px across.
    #[test]
    fn a_near_bright_star_is_not_a_white_ball() {
        let t = StarTuning::default();
        let near = star_draw(SUN_BASE_M * 5.0, 3.0e16, &t);
        assert!(near.amplitude > 1.0, "a near star still clips its centre");
        // Where the CORE stops being fully white: A·exp(-r²/2σ²) = 1 ⇒ r = σ·√(2·ln A).
        let white_px = t.core_sigma_px * (2.0 * near.amplitude.ln()).sqrt();
        assert!(
            white_px < 3.0,
            "a star's saturated core must stay small — got {white_px} px, which reads as a ball"
        );
    }

    /// ★ A STAR FADES OUT; IT NEVER BLINKS OUT. The sprite has a floor, so it can never shrink below
    /// a pixel and start missing pixel centres as the camera turns — the failure Stellarium and
    /// Celestia both name. What falls to zero is the BRIGHTNESS, which is continuous.
    #[test]
    fn a_fading_star_keeps_its_minimum_sprite_until_it_is_culled_on_brightness() {
        let t = StarTuning::default();
        let mut d = 1.0e17;
        let mut drew = 0;
        for _ in 0..200 {
            let s = star_draw(SUN_BASE_M, d, &t);
            if s.amplitude == 0.0 {
                assert_eq!(s.crop_px, 0.0, "a culled star draws nothing at all");
                break;
            }
            assert!(
                s.crop_px >= t.min_crop_px,
                "a visible star is never smaller than the anti-blink floor: {s:?}"
            );
            drew += 1;
            d *= 1.15;
        }
        assert!(drew > 10, "the walk must actually cross the visible range");
    }

    /// ★ THE CULL SITS BELOW ONE DISPLAY STEP, AND THE INEQUALITY RUNS THAT WAY ON PURPOSE.
    ///
    /// I first wrote this the other way round — cull ABOVE one 8-bit step, reasoning that a star
    /// removed should have been invisible anyway. That is right for a star ALONE and wrong for a
    /// galaxy, and the difference is the whole milky band: the blending ADDS, so a thousand stars at
    /// a third of a display step each sum to a bright glow. Culling where one star is invisible would
    /// have deleted the band along with them.
    ///
    /// So the cull is a bound on OVERDRAW, not a visibility test: it removes a star only once it is
    /// below what a screen could show even alone, so nothing visible is ever taken away.
    #[test]
    fn the_cull_sits_below_one_display_step_so_a_crowd_still_glows() {
        let t = StarTuning::default();
        assert!(
            t.cull_level < 1.0 / 255.0,
            "a star must never be culled while it could still be seen alone"
        );
        // AND THE CROWD IS THE POINT: stars this faint are individually invisible and collectively
        // bright, which is what the glow of a galaxy seen edge-on actually is.
        let crowd = 1_000.0 * t.cull_level;
        assert!(
            crowd > 1.0,
            "a thousand just-visible stars must sum to a bright band, got {crowd}"
        );
        let gone = star_draw(SUN_BASE_M, 1.0e30, &t);
        assert_eq!(gone.amplitude, 0.0);
        assert_eq!(gone.crop_px, 0.0);
    }

    /// A star bright enough to clip is what gives the white core: the wings keep the hue while the
    /// centre saturates. This pins that such stars EXIST at a realistic distance rather than being
    /// a theoretical arm of the law.
    #[test]
    fn a_near_bright_star_saturates_its_core() {
        let t = StarTuning::default();
        let s = star_draw(SUN_BASE_M * (25.0_f64).sqrt(), 3.0e16, &t);
        assert!(
            s.amplitude > 1.0,
            "a bright near star must clip to white at its centre: {s:?}"
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use glam::{DQuat, I64Vec3};
    use vd_core::UniverseTick;
    use vd_core::look::{look_bag, luma_bag};
    use vd_core::pose::{FrameRef, StampedPose};

    /// ONE COMPOSED ROW — the whole client-facing contract in four fields (§2.4). Every fixture row
    /// is stamped in one ORIGIN frame at one tick, which is the shape a composed level arrives in:
    /// the client applies no transform and asks no source anything.
    fn row(realm: RealmId, parent: Option<RealmId>, pos: DVec3, bag: Vec<u8>) -> SceneRow {
        row_framed(
            realm,
            parent,
            FrameRef::SystemSpace { system_seed: 7 },
            pos,
            bag,
        )
    }

    /// [`row`] with the origin frame stated explicitly. The frame is the row's statement of the UNIT
    /// its position's integer cell is counted in, so the tier pin has to be able to vary it.
    fn row_framed(
        realm: RealmId,
        parent: Option<RealmId>,
        frame: FrameRef,
        pos: DVec3,
        bag: Vec<u8>,
    ) -> SceneRow {
        SceneRow {
            realm,
            parent,
            pose: StampedPose::at_rest(frame, pos, UniverseTick(100)),
            bag,
        }
    }

    /// A `TAG_LOOK` bag for a spherical outline — what a RUNNING realm says about itself (SL3).
    fn look_shell(r: f64) -> Vec<u8> {
        look_bag(&Boundary::Shell { r })
    }

    /// A `TAG_LOOK` bag for a box outline (a station / area volume).
    fn look_aabb(half: DVec3) -> Vec<u8> {
        look_bag(&Boundary::Aabb { half })
    }

    /// THE ONE WORLD (SL5) as a composed level: every region of the seed forest becomes a row
    /// carrying its own outline. The client draws exactly what the world states — no second
    /// generator, no reduced fixture universe.
    fn one_world_level() -> Vec<SceneRow> {
        vd_physics::worldgen::realm_regions_for(0)
            .iter()
            .map(|r| {
                // Flatten the NORMALIZED centre (H-21: reading `.offset()` here would keep this
                // test green while placing every star at the origin — a green test that stopped
                // measuring is worse than a red one).
                //
                // THE BOUND/LOOK SPLIT: a realm STATES its LOOK — a look-less ambient states
                // nothing drawable (an empty bag: the row is tracked, never drawn), exactly as
                // the shard's own emit now behaves. The BOUND is never framed.
                let bag = match r.look {
                    Some(look) => look_bag(&look),
                    None => {
                        vd_core::tlv::TlvWriter::new(vd_core::look::WINDOW_BODY_SCHEMA).finish()
                    }
                };
                // ★ THE THIRTEENTH SITE (slice S9). This flattened the centre with the CHILD's own
                // step, so the galaxy row in this fixture was 2048× out — and nothing failed,
                // because the assertions that read it are about shapes and depths. `centre_m` reads
                // the step off the parent, and the wrong one can no longer be passed.
                let regions = vd_physics::worldgen::realm_regions_for(0);
                row(
                    r.realm,
                    r.parent,
                    r.centre_m(&regions)
                        .expect("THE world's forest holds every child's parent"),
                    bag,
                )
            })
            .collect()
    }

    #[test]
    fn overlaid_moves_a_streamed_box_centre_keeps_static_and_empty_is_boot_identical() {
        use vd_core::TickId;
        use vd_wire::channels::{RealmSnap, RealmSnapshotDatagram, SubId};
        // A level: Planet 1 + Station 2, both drawn from their own look statements.
        let level = RealmScene::from_scene_rows(&[
            row(
                RealmId::Planet(1),
                Some(RealmId::System(7)),
                DVec3::new(20.0, 0.0, 0.0),
                look_shell(5.0),
            ),
            row(
                RealmId::Station(2),
                Some(RealmId::System(7)),
                DVec3::new(-25.0, 0.0, 0.0),
                look_shell(3.0),
            ),
        ])
        .expect("the level projects");
        // Stream a LIVE pose for Planet 1 ONLY, authored in the origin frame at a new offset.
        let streamed_frame = FrameRef::SystemSpace { system_seed: 7 };
        // A COARSE-CELL position: the cell is the half of the coordinate the client used to discard.
        let streamed_center = LatticePos::at(I64Vec3::new(4, -2, 9), DVec3::new(1.0e9, 5.0e8, 0.0));
        let mut view = RealmView::default();
        view.on_realm_snapshot(
            None,
            RealmSnapshotDatagram {
                sub: SubId(0),
                frame_id: 1,
                source_tick: TickId(1),
                universe_tick: UniverseTick(10),
                origin_epoch: 0,
                sky_anchor: None,
                realms: vec![RealmSnap {
                    realm: RealmId::Planet(1),
                    // The edge HEAD: Planet 1's OWN frame; `pose.frame` below is the TAIL, the frame
                    // its parent authored the placement in.
                    frame: FrameRef::PlanetCentered { planet_seed: 1 },
                    pose: StampedPose {
                        frame: streamed_frame,
                        pos: streamed_center,
                        vel: DVec3::ZERO,
                        orient: DQuat::IDENTITY,
                        universe_tick: UniverseTick(10),
                    },
                }],
            },
        );
        let scene = level.overlaid_at(&view, f64::INFINITY);
        // Planet 1 moved: its CENTRE is the streamed value, carrying the streamed COARSE CELL as well
        // as the offset (dropping the cell here is the slice-5 defect), and the UNIT that cell is
        // counted in is the one the streamed pose stated.
        let moved = scene.get(RealmId::Planet(1)).expect("planet box");
        assert_eq!(moved.center, streamed_center);
        assert_eq!(moved.center.cell(), I64Vec3::new(4, -2, 9));
        assert_eq!(moved.tier, stated_tier(streamed_frame));
        // EVERYTHING THE ROW ITSELF AUTHORED SURVIVES the overlay — the live feed restates a
        // placement, never an appearance (SL3: the parent says where, the realm says how it looks).
        let stated = level.get(RealmId::Planet(1)).expect("the level's planet");
        assert_eq!(moved.shape, stated.shape);
        assert_eq!(moved.body, stated.body);
        assert_eq!(moved.luma, stated.luma);
        assert_eq!(moved.parent, stated.parent);
        assert_eq!(moved.depth, stated.depth);
        assert_eq!(moved.color_rgba, stated.color_rgba);
        // Station 2 (not streamed) stays exactly as the level stated it.
        assert_eq!(
            scene.get(RealmId::Station(2)),
            level.get(RealmId::Station(2))
        );
        // An EMPTY view overlays to the level scene byte-identical (the walk-scale case).
        assert_eq!(
            level.overlaid_at(&RealmView::default(), f64::INFINITY),
            level
        );
    }

    /// THE STATION-LAPSE GATE's client half (look_horizon.md slice 1 — declared RED before the
    /// slice; the red was MEASURED at the producer: a non-glowing child stated no marker at all,
    /// so a station whose own picture lapsed VANISHED from the drawn set). A station's look draws
    /// while it runs; when its picture lapses the composed row's bag falls to its parent's
    /// extent-only marker — and the station stays DRAWN: a point of light sized by the ONE
    /// stated radius (the one-radius law), coloured by the client's own role law, NEVER an empty
    /// frame.
    #[test]
    fn a_lapsed_station_degrades_to_a_correctly_sized_marker_never_to_nothing() {
        let station = RealmId::Station(7);
        let pos = DVec3::new(60.0, -4.0, 9.0);
        // RUNNING: the station's own look draws it (SL3 — a realm draws itself).
        let running = RealmScene::from_scene_rows(&[row(
            station,
            None,
            pos,
            look_aabb(DVec3::new(10.0, 20.0, 30.0)),
        )])
        .expect("projects");
        assert_eq!(running.get(station).map(|b| b.body), Some(BodyKind::Look));
        // LAPSED: the bag is now the parent's extent-only marker — the station is STILL drawn.
        let extent = 40.0;
        let lapsed = RealmScene::from_scene_rows(&[row(
            station,
            None,
            pos,
            vd_core::look::marker_bag(None, extent),
        )])
        .expect("projects");
        let b = lapsed
            .get(station)
            .expect("a lapsed station is drawn as a marker, never dropped from the scene");
        assert_eq!(b.body, BodyKind::Marker);
        assert_eq!(b.luma, None, "a non-glowing subject states no photometrics");
        assert_eq!(
            b.shape,
            BoxShape::Sphere { r: extent },
            "the marker carries the parent's ONE stated radius"
        );
        // CORRECTLY SIZED: the sprite's base radius is exactly the stated extent (the shared
        // sizing expression the renderer and the pixel gates both call)…
        assert_eq!(
            marker_base_radius_m(b.luma, shape_extent_m(b.shape)),
            extent
        );
        // …and the sizing law: the base is the LARGER of the photometric radius and the extent.
        assert_eq!(marker_base_radius_m(Some((4, 1.0)), 0.2), 0.5, "√L wins");
        assert_eq!(
            marker_base_radius_m(Some((4, 1.0)), 40.0),
            40.0,
            "extent wins"
        );
        assert_eq!(marker_base_radius_m(None, -3.0), 0.0, "garbage floors at 0");
        // DRAWN IN THE ROLE COLOUR at a point of light's opaque alpha; a glowing marker keeps
        // its blackbody class colour.
        let quiet = marker_color_rgba(b.color_rgba, b.luma);
        assert_eq!(quiet[3], 1.0);
        assert_eq!(quiet[..3], b.color_rgba[..3]);
        assert_eq!(
            marker_color_rgba(b.color_rgba, Some((4, 1.0))),
            marker_look(4, 1.0).color_rgba
        );
        // The extent accessor's box arm: a box shape's circumscribed radius (its half-diagonal).
        assert_eq!(
            shape_extent_m(BoxShape::Box {
                half: DVec3::new(3.0, 4.0, 12.0)
            }),
            13.0
        );
        // And a GLOWING marker bag (datum + extent) decodes BOTH halves into one box: the
        // photometric datum rides `luma`, the one radius rides the shape.
        let glowing = RealmScene::from_scene_rows(&[row(
            RealmId::Planet(9),
            None,
            pos,
            vd_core::look::marker_bag(Some((6, 0.25)), 3.954),
        )])
        .expect("projects");
        let g = glowing.get(RealmId::Planet(9)).expect("drawn");
        assert_eq!(g.body, BodyKind::Marker);
        assert_eq!(g.luma, Some((6, 0.25)));
        assert_eq!(g.shape, BoxShape::Sphere { r: 3.954 });
    }

    #[test]
    fn a_look_bag_projects_a_shell_to_a_sphere_and_an_aabb_to_a_box() {
        let scene = RealmScene::from_scene_rows(&[
            row(
                RealmId::System(7),
                None,
                DVec3::new(1.0, 2.0, 3.0),
                look_shell(100.0),
            ),
            row(
                RealmId::Station(9),
                None,
                DVec3::ZERO,
                look_aabb(DVec3::new(10.0, 20.0, 30.0)),
            ),
        ])
        .expect("projects");
        assert_eq!(scene.len(), 2);
        assert!(!scene.is_empty());
        let sys = scene.get(RealmId::System(7)).expect("system box");
        assert_eq!(sys.shape, BoxShape::Sphere { r: 100.0 });
        assert_eq!(
            sys.center,
            LatticePos::from_metres(DVec3::new(1.0, 2.0, 3.0), Tier::Fine)
        );
        // A look body is the realm's OWN statement and carries no photometrics (a look never
        // carries luma — the two statements are structurally exclusive).
        assert_eq!(sys.body, BodyKind::Look);
        assert_eq!(sys.luma, None);
        let stn = scene.get(RealmId::Station(9)).expect("station box");
        assert_eq!(
            stn.shape,
            BoxShape::Box {
                half: DVec3::new(10.0, 20.0, 30.0)
            }
        );
    }

    #[test]
    fn an_obb_look_projects_to_an_axis_aligned_box_dropping_orientation() {
        // The Obb-orient arm of shape_of: the orientation is deferred, so an Obb renders as its
        // axis-aligned bounding proxy (Box{half}).
        let bag = look_bag(&Boundary::Obb {
            half: DVec3::new(4.0, 5.0, 6.0),
            orient: DQuat::from_rotation_z(0.5),
        });
        let scene = RealmScene::from_scene_rows(&[row(RealmId::Area(3), None, DVec3::ZERO, bag)])
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
    fn a_marker_row_is_a_zero_radius_point_carrying_its_photometrics() {
        // THE DRAW LAW's second arm (owner decision 10): a SLEEPING realm cannot draw itself, so its
        // parent states a point-of-light datum instead. The box is a tracked POINT — no placeholder
        // mesh, because a sleeping star's drawn footprint at these distances IS sub-pixel.
        let scene = RealmScene::from_scene_rows(&[
            row(
                RealmId::System(7),
                None,
                DVec3::new(3.0, 0.0, 0.0),
                luma_bag(6, 0.000_972_607_424_178_079_9),
            ),
            row(
                RealmId::Planet(7),
                Some(RealmId::System(7)),
                DVec3::ZERO,
                look_shell(10.0),
            ),
        ])
        .expect("projects");
        let marker = scene.get(RealmId::System(7)).expect("marker box");
        assert_eq!(marker.body, BodyKind::Marker);
        assert_eq!(marker.luma, Some((6, 0.000_972_607_424_178_079_9)));
        assert_eq!(marker.shape, BoxShape::Sphere { r: 0.0 });
        // A marker is a full row otherwise: it holds its placement and its nesting like any other.
        assert_eq!(
            marker.center,
            LatticePos::from_metres(DVec3::new(3.0, 0.0, 0.0), Tier::Fine)
        );
        assert_eq!(marker.depth, 0);
        // The two authors are DISTINGUISHABLE on the box (the diagnosis surface reads exactly this).
        let look = scene.get(RealmId::Planet(7)).expect("look box");
        assert_eq!(look.body, BodyKind::Look);
        assert_eq!(look.luma, None);
        assert_ne!(look.body, marker.body);
    }

    #[test]
    fn a_markers_point_sprite_is_its_class_colour_at_the_root_luminosity_radius() {
        // THE MARKER POINT SPRITE (window lane Slice D §2.8). A unit-luminosity source is drawn at
        // exactly the shared point-source base radius — the convention `POINT_SOURCE_BASE_RADIUS_M`
        // states — and in its spectral class's own colour, OPAQUE (a point of light emits; it does
        // not enclose like a translucent shell).
        let sun = marker_look(4, 1.0);
        assert_eq!(sun.base_radius_m, POINT_SOURCE_BASE_RADIUS_M);
        let [r, g, b] = MARKER_CLASS_SRGB[4];
        assert_eq!(sun.color_rgba, [r, g, b, 1.0]);
        // BRIGHTNESS DRIVES SIZE by the equal-surface-brightness law (radius ∝ √L, so flux ∝ L):
        // a quarter-luminosity source is drawn at HALF the radius, exactly.
        assert_eq!(
            marker_look(4, 0.25).base_radius_m,
            POINT_SOURCE_BASE_RADIUS_M * 0.5
        );
        // THE world's pinned M-dwarf datum (`System(7)`): class 6 draws red, and its faint
        // luminosity lands far below the base radius — which is why the shared apparent-size floor
        // exists at all (`marker_world_radius`), not because the sprite has a minimum of its own.
        let dwarf = marker_look(6, 0.000_972_607_424_178_079_9);
        assert_eq!(dwarf.color_rgba, [1.000, 0.714, 0.427, 1.0]);
        assert!(
            dwarf.base_radius_m < POINT_SOURCE_BASE_RADIUS_M,
            "a sub-solar source is drawn smaller: {}",
            dwarf.base_radius_m
        );
        // A REFLECTOR keeps its illuminator's class, so a planet reads in its star's colour — the
        // same class code yields the same colour whatever the luminosity.
        assert_eq!(marker_look(6, 1e-9).color_rgba, dwarf.color_rgba);
    }

    #[test]
    fn a_malformed_marker_datum_still_draws_a_point_and_never_a_nan() {
        // THE PRESENCE LAW is "never zero", so neither malformed input may make a marker vanish or
        // produce a NaN transform. A class code the wire never mints draws as the coolest class.
        let coolest = MARKER_CLASS_SRGB[MARKER_CLASS_SRGB.len() - 1];
        assert_eq!(marker_look(7, 1.0).color_rgba[0], coolest[0]);
        assert_eq!(
            marker_look(u8::MAX, 1.0).color_rgba,
            marker_look(6, 1.0).color_rgba
        );
        // A negative or NaN luminosity floors at zero radius (the apparent-size floor then carries
        // it), rather than yielding NaN through `sqrt`.
        assert_eq!(marker_look(0, -1.0).base_radius_m, 0.0);
        assert_eq!(marker_look(0, f64::NAN).base_radius_m, 0.0);
    }

    #[test]
    fn every_point_sprite_shares_one_unit_vertex_buffer() {
        // "A dot never costs a mesh" (§2.14's tier-0 rung): the renderer builds THIS buffer once
        // and scales it per marker, so the buffer cannot depend on the marker.
        let a = point_sprite_vertices();
        assert_eq!(a, point_sprite_vertices());
        assert!(!a.is_empty(), "the point sprite has geometry");
        // It is the same unit sphere the `Sphere` proxy tessellates to, at unit radius — the
        // tessellation lives here in Tier-A, never in the coverage-exempt renderer (H4).
        let far = a
            .iter()
            .map(|v| {
                f64::from(v.pos[0])
                    .hypot(f64::from(v.pos[1]))
                    .hypot(f64::from(v.pos[2]))
            })
            .fold(0.0_f64, f64::max);
        assert!((far - 1.0).abs() < 1e-6, "unit radius, got {far}");
    }

    #[test]
    fn every_realm_kind_projects_through_the_same_row_and_no_frame_is_derived_here() {
        // WHAT THIS REPLACES: the client used to DERIVE each realm's frame from its KIND (a match
        // over Planet/System/Ship/Station/Area, plus an Area's parent-planet-seed lookup). That
        // derivation is gone — a row states its own frame — so five different KINDS carrying the
        // identical row project to identical boxes, and an Area needs no planet parent to exist.
        let hull = vd_core::ids::EntityId(1);
        let kinds = [
            RealmId::Planet(4),
            RealmId::System(5),
            RealmId::Ship(hull),
            RealmId::Station(6),
            RealmId::Area(9),
        ];
        for realm in kinds {
            let scene = RealmScene::from_scene_rows(&[row(
                realm,
                None,
                DVec3::new(1.0, 0.0, 0.0),
                look_shell(12.0),
            )])
            .expect("projects");
            let b = scene.get(realm).expect("box");
            assert_eq!(b.shape, BoxShape::Sphere { r: 12.0 });
            assert_eq!(b.tier, Tier::Fine);
            assert_eq!(
                b.center,
                LatticePos::from_metres(DVec3::new(1.0, 0.0, 0.0), Tier::Fine)
            );
            assert_eq!(b.depth, 0);
            assert_eq!(b.body, BodyKind::Look);
            assert_eq!(b.parent, None);
        }
        // An Area under a SYSTEM and an Area under a PLANET project identically — the old code had
        // to ask what kind the parent was to fill in a planet seed; nothing asks now.
        let under_system = RealmScene::from_scene_rows(&[
            row(RealmId::System(3), None, DVec3::ZERO, look_shell(40.0)),
            row(
                RealmId::Area(9),
                Some(RealmId::System(3)),
                DVec3::ZERO,
                look_shell(3.0),
            ),
        ])
        .expect("projects");
        let under_planet = RealmScene::from_scene_rows(&[
            row(RealmId::Planet(7), None, DVec3::ZERO, look_shell(40.0)),
            row(
                RealmId::Area(9),
                Some(RealmId::Planet(7)),
                DVec3::ZERO,
                look_shell(3.0),
            ),
        ])
        .expect("projects");
        let a = under_system
            .get(RealmId::Area(9))
            .expect("area under a system");
        let p = under_planet
            .get(RealmId::Area(9))
            .expect("area under a planet");
        assert_eq!(a.shape, p.shape);
        assert_eq!(a.tier, p.tier);
        assert_eq!(a.depth, p.depth);
        assert_eq!(a.color_rgba, p.color_rgba);
    }

    /// THE UNIT IS STATED, NEVER INFERRED (§2.4) — the successor of the deleted `frame` field.
    ///
    /// A box used to carry the realm's own frame and the drawing side had to work the unit out from
    /// it. Now every composed row states the space its position is measured in, and the tier is read
    /// off THAT row: two rows for the same realm, identical but for the frame they were authored in,
    /// draw a whole tier apart. What the unit then does to the drawn point is pinned by
    /// `draw_center_multiplies_by_the_unit_it_was_told_not_by_one_it_picks`.
    #[test]
    fn a_boxs_unit_is_stated_by_its_own_rows_pose_frame() {
        let galaxy = RealmScene::from_scene_rows(&[row_framed(
            RealmId::System(1),
            None,
            FrameRef::GalaxySpace { galaxy_seed: 0 },
            DVec3::ZERO,
            look_shell(100.0),
        )])
        .expect("projects");
        let system = RealmScene::from_scene_rows(&[row(
            RealmId::System(1),
            None,
            DVec3::ZERO,
            look_shell(100.0),
        )])
        .expect("projects");
        let coarse = galaxy.get(RealmId::System(1)).expect("galaxy-framed box");
        let fine = system.get(RealmId::System(1)).expect("system-framed box");
        assert_eq!(coarse.tier, Tier::Galaxy);
        assert_eq!(fine.tier, Tier::Fine);
        assert_ne!(coarse.tier, fine.tier);
        // The box reads exactly what `stated_tier` reads off the row's own frame — one statement,
        // one reader, no context.
        assert_eq!(
            coarse.tier,
            stated_tier(FrameRef::GalaxySpace { galaxy_seed: 0 })
        );
        assert_eq!(
            fine.tier,
            stated_tier(FrameRef::SystemSpace { system_seed: 7 })
        );
        // Nothing else about the two rows differs, so nothing else can explain the difference.
        assert_eq!(coarse.shape, fine.shape);
        assert_eq!(coarse.center, fine.center);
    }

    #[test]
    fn a_duplicate_realm_id_is_rejected() {
        let err = RealmScene::from_scene_rows(&[
            row(RealmId::System(7), None, DVec3::ZERO, look_shell(100.0)),
            row(
                RealmId::System(7),
                None,
                DVec3::ZERO,
                look_aabb(DVec3::splat(5.0)),
            ),
        ])
        .expect_err("duplicate realm must reject");
        assert_eq!(err, SceneError::DuplicateRealm);
    }

    #[test]
    fn a_duplicate_is_caught_over_the_whole_level_even_when_neither_row_is_drawn() {
        // Pass 1 runs over the WHOLE level — drawn and merely-tracked rows alike — so a composer bug
        // that repeats a realm is loud even when neither copy would have produced a box. (This is
        // what the old JSON entrypoint's duplicate propagation guarded, on the lane that replaced it.)
        let err = RealmScene::from_scene_rows(&[
            row(RealmId::System(7), None, DVec3::ZERO, Vec::new()),
            row(RealmId::System(7), None, DVec3::ZERO, Vec::new()),
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
        let scene = RealmScene::from_scene_rows(&[
            // Deliberately NOT in depth order, to prove order-independence.
            row(
                player,
                Some(ship),
                DVec3::ZERO,
                look_aabb(DVec3::splat(1.0)),
            ),
            row(station, None, DVec3::ZERO, look_aabb(DVec3::splat(100.0))),
            row(
                ship,
                Some(station),
                DVec3::ZERO,
                look_aabb(DVec3::splat(10.0)),
            ),
        ])
        .expect("projects");
        assert_eq!(scene.get(station).expect("station").depth, 0);
        assert_eq!(scene.get(ship).expect("ship").depth, 1);
        assert_eq!(scene.get(player).expect("player").depth, 2);
    }

    #[test]
    fn a_parent_link_to_an_out_of_scene_realm_is_a_root() {
        // The `Some(not-in-map)` terminal arm of depth_of: a parent not present in the level ends the
        // chain (that realm is a root for depth purposes) — depth 0.
        let scene = RealmScene::from_scene_rows(&[row(
            RealmId::Station(5),
            Some(RealmId::System(99)), // System(99) is NOT in the level
            DVec3::ZERO,
            look_aabb(DVec3::splat(1.0)),
        )])
        .expect("projects");
        assert_eq!(scene.get(RealmId::Station(5)).expect("box").depth, 0);
    }

    #[test]
    fn a_parent_cycle_is_a_bounded_loud_error() {
        // A ⇄ B cycle among DRAWN rows: the pass-2 depth walk stops loud at MAX_NEST_DEPTH and
        // from_scene_rows propagates it (the `depth_of(..)?` Err arm — distinct from pass 1's
        // DuplicateRealm, which returns BEFORE the walk). Never an unbounded loop (H7 guard).
        let a = RealmId::System(1);
        let b = RealmId::System(2);
        let err = RealmScene::from_scene_rows(&[
            row(a, Some(b), DVec3::ZERO, look_aabb(DVec3::splat(1.0))),
            row(b, Some(a), DVec3::ZERO, look_aabb(DVec3::splat(1.0))),
        ])
        .expect_err("a cycle must reject");
        assert_eq!(err, SceneError::CycleOrDepthExceeded);
    }

    #[test]
    fn an_over_deep_parent_chain_stops_loud_at_the_ceiling() {
        // The OTHER way past the ceiling: an acyclic but pathologically deep chain. Same bounded,
        // loud stop — the walk is capped by hops, not by whether it ever revisits a realm.
        const CHAIN: u64 = MAX_NEST_DEPTH as u64 + 2;
        let level: Vec<SceneRow> = (0..CHAIN)
            .map(|i| {
                let parent = if i + 1 < CHAIN {
                    Some(RealmId::System(i + 1))
                } else {
                    None
                };
                row(
                    RealmId::System(i),
                    parent,
                    DVec3::ZERO,
                    look_aabb(DVec3::splat(1.0)),
                )
            })
            .collect();
        let err = RealmScene::from_scene_rows(&level).expect_err("an over-deep chain must reject");
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
            // T2: a star is a realm, and its fallback family is a role like any other (the
            // DRAWN photometric colour rides the look bag's TAG_LUMA — ruling C — never this).
            RealmId::Star(7),
            // ★ THE TWO CONTAINERS (slice S9). A player never sees either as a box — a galaxy and
            // the universe are BOUNDARIES, and this project's two-body law makes a containment
            // boundary undrawable. They are exercised anyway, because the table is TOTAL and an arm
            // nothing drives is an arm nobody has checked stays in gamut.
            RealmId::Galaxy(1),
            RealmId::Universe,
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

    /// SLICE 5 — the box's centre flattens to metres by EXACT INTEGER CELL arithmetic, the same way an
    /// entity's does, so a box and an occupant standing on it agree to the bit. The old client kept only
    /// the box's offset and dropped its cell, which is right ONLY while every cell is zero — this asserts
    /// the case that used to be wrong.
    #[test]
    fn draw_center_carries_the_integer_cell_not_just_the_metre_offset() {
        let rbox = RealmBox {
            shape: BoxShape::Sphere { r: 1.0 },
            body: BodyKind::Look,
            luma: None,
            tier: Tier::Fine,
            center: LatticePos::at(I64Vec3::new(3, 0, 0), DVec3::new(0.25, 0.0, 0.0)),
            parent: None,
            depth: 0,
            color_rgba: [0.0, 0.0, 0.0, BOX_ALPHA],
            facing: [0.0, 0.0, 0.0, 1.0],
        };
        let edge = Tier::Fine.cell_edge_m();
        assert_eq!(rbox.draw_center(), DVec3::new(3.0 * edge + 0.25, 0.0, 0.0));
        // Dropping the cell (the pre-slice-5 arithmetic) would have drawn it at 0.25 m — the box would
        // sit on top of the camera instead of 3 cells away.
        assert_ne!(rbox.draw_center(), rbox.center.offset());
    }

    /// THE SHIPPER DECIDES THE UNIT, on the box lane too — the twin of `view.rs`'s
    /// `world_pos_multiplies_by_the_unit_it_was_told_not_by_one_it_picks`.
    ///
    /// Two boxes with the IDENTICAL integer cell, differing only in the unit stated for that cell, must
    /// draw a whole tier apart. A `draw_center` that looked the unit up for itself at drawing time would
    /// return the same point for both.
    #[test]
    fn draw_center_multiplies_by_the_unit_it_was_told_not_by_one_it_picks() {
        use vd_core::pose::{FINE_CELL_EDGE_M, Tier};
        let at = |tier| RealmBox {
            shape: BoxShape::Sphere { r: 1.0 },
            body: BodyKind::Look,
            luma: None,
            tier,
            center: LatticePos::at(I64Vec3::new(1, 0, 0), DVec3::ZERO),
            parent: None,
            depth: 0,
            color_rgba: [0.0, 0.0, 0.0, BOX_ALPHA],
            facing: [0.0, 0.0, 0.0, 1.0],
        };
        assert_eq!(
            at(Tier::Fine).draw_center(),
            DVec3::new(FINE_CELL_EDGE_M, 0.0, 0.0)
        );
        assert_eq!(
            at(Tier::Galaxy).draw_center(),
            DVec3::new(Tier::Galaxy.cell_edge_m(), 0.0, 0.0)
        );
    }

    // RETIRED (Slice C1 flag day, owner-approved 2026-08-15/16 items 1/9/10 — window_lane.md §2.4):
    // the tier-agreement pin test guarded the shape lane's tier-inference gap ("a static shape states
    // no tail"). The gap is CLOSED structurally: every composed row states its unit on its own
    // pose.frame, read per row. The RealmBox tier fields are pinned per-row by
    // `a_boxs_unit_is_stated_by_its_own_rows_pose_frame`.

    #[test]
    fn box_lowers_to_one_cuboid_prim_scaled_by_the_half_extents() {
        let rbox = RealmBox {
            shape: BoxShape::Box {
                half: DVec3::new(2.0, 3.0, 4.0),
            },
            body: BodyKind::Look,
            luma: None,
            tier: Tier::Fine,
            center: LatticePos::from_metres(DVec3::new(1.0, 0.0, 0.0), Tier::Fine),
            parent: None,
            depth: 0,
            color_rgba: [0.1, 0.2, 0.3, BOX_ALPHA],
            facing: [0.0, 0.0, 0.0, 1.0],
        };
        // The centre is flattened ONCE, by the caller, through the one chokepoint; the prim lands
        // exactly there (slice 5: ONE term, no composition in here).
        let draw_center = rbox.draw_center();
        assert_eq!(draw_center, DVec3::new(1.0, 0.0, 0.0));
        let prims = to_render_prims(&rbox, draw_center);
        assert_eq!(prims.len(), 1);
        let p = &prims[0];
        // 6 faces × 2 tris × 3 verts = 36.
        assert_eq!(p.vertices.len(), 36);
        assert_eq!(p.color_rgba, [0.1, 0.2, 0.3, BOX_ALPHA]);
        assert_eq!(p.transform.scale, [2.0, 3.0, 4.0]);
        // The prim translation IS the reduced centre — nothing is added to it.
        assert_eq!(p.transform.translation, [1.0, 0.0, 0.0]);
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
            body: BodyKind::Look,
            luma: None,
            tier: Tier::Fine,
            center: LatticePos::ORIGIN,
            parent: None,
            depth: 0,
            color_rgba: [0.4, 0.5, 0.6, BOX_ALPHA],
            facing: [0.0, 0.0, 0.0, 1.0],
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
        let scene = RealmScene::from_scene_rows(&[
            row(
                RealmId::System(9),
                None,
                DVec3::ZERO,
                look_aabb(DVec3::splat(1.0)),
            ),
            row(
                RealmId::Planet(2),
                None,
                DVec3::ZERO,
                look_aabb(DVec3::splat(1.0)),
            ),
        ])
        .expect("projects");
        let realms: Vec<RealmId> = scene.iter().map(|(r, _)| r).collect();
        // RealmId Ord: Planet(_) < System(_) by declaration order.
        assert_eq!(realms, vec![RealmId::Planet(2), RealmId::System(9)]);
    }

    #[test]
    fn an_empty_level_projects_to_an_empty_scene() {
        let scene = RealmScene::from_scene_rows(&[]).expect("an empty level projects");
        assert!(scene.is_empty());
        assert_eq!(scene.len(), 0);
        assert_eq!(scene.get(RealmId::System(1)), None);
        assert_eq!(scene, RealmScene::default());
    }

    #[test]
    fn a_level_projects_deterministically_and_the_same_rows_as_a_delta_agree() {
        // WHAT THIS REPLACES: the deleted `boxes.json` loader's "the dev-config load matches the
        // plant" pin. There is one lane now, so the equivalent claim is that the lane is a FUNCTION —
        // the same rows always project to the same scene — and that the level and delta entrypoints
        // share the one row projection rather than each having their own.
        let rows = vec![
            row(
                RealmId::System(7),
                None,
                DVec3::new(1.0, 2.0, 3.0),
                look_shell(100.0),
            ),
            row(
                RealmId::Station(9),
                Some(RealmId::System(7)),
                DVec3::new(4.0, 5.0, 6.0),
                look_aabb(DVec3::splat(2.0)),
            ),
        ];
        let once = RealmScene::from_scene_rows(&rows).expect("projects");
        let twice = RealmScene::from_scene_rows(&rows).expect("projects");
        assert_eq!(once, twice, "the projection is a function of the rows");
        let as_delta = RealmScene::default()
            .with_delta(&rows, &[])
            .expect("the same rows apply as a delta");
        assert_eq!(as_delta, once, "one row projection, two entrypoints");
        // And it really carries the boxes (not a silent empty).
        assert_eq!(once.len(), 2);
        assert_eq!(
            once.get(RealmId::System(7)).expect("system").shape,
            BoxShape::Sphere { r: 100.0 }
        );
        assert_eq!(once.get(RealmId::Station(9)).expect("station").depth, 1);
    }

    #[test]
    fn a_bag_with_no_drawable_statement_is_tracked_but_never_drawn() {
        // WHAT THIS REPLACES: the deleted JSON loader's loud malformed-input rejection. The composed
        // lane's law is the opposite and stronger — a row whose bag states nothing drawable is not an
        // error, it is simply NOT DRAWN (every pixel has exactly one lawful author, so an absent
        // statement can never be guessed at). An EMPTY bag and a CORRUPT one both land there; the
        // unknown-tag case rides the TLV codec's own skip law (pinned in `vd_core::look`).
        let scene = RealmScene::from_scene_rows(&[
            row(RealmId::System(7), None, DVec3::ZERO, Vec::new()),
            row(
                RealmId::Planet(7),
                None,
                DVec3::ZERO,
                vec![0xFF, 0x00, 0x7F],
            ),
            row(RealmId::Station(9), None, DVec3::ZERO, look_shell(5.0)),
        ])
        .expect("a level of undrawable rows is not an error");
        assert_eq!(
            scene.get(RealmId::System(7)),
            None,
            "an empty bag draws nothing"
        );
        assert_eq!(
            scene.get(RealmId::Planet(7)),
            None,
            "a corrupt bag draws nothing — never a guessed shape"
        );
        assert_eq!(scene.len(), 1, "only the row that stated a look is drawn");
        assert_eq!(
            scene.get(RealmId::Station(9)).expect("the drawn row").shape,
            BoxShape::Sphere { r: 5.0 }
        );
    }

    #[test]
    fn the_one_worlds_level_draws_the_finite_realms_and_skips_the_ambient_shells() {
        // SL5 SINGLE SOURCE: the client's scene is projected from rows carrying the SAME seed forest
        // the sim's containment detector consumes. The finite realms are drawn — System 7/8 (r=40),
        // Planet 7 (r=10), Station 7 (half=5), Area 7 (half=3) — AND the Galaxy (r=180) as the
        // CONTAINING box around the systems, so an entity in the between-space is visibly still inside
        // a realm (never orphaned). Only the ~unbounded Universe (r=1e9) is SKIPPED (ambient: felt,
        // not framed).
        //
        // ★ SEVEN LEAVES, NOT FIVE (2026-09-01). Two more planets joined the walk world when the
        // second star system gained children. This test counts what the ONE world holds, so it moved
        // with the world — which is the test doing its job. It went red the moment the world changed
        // and stayed red until somebody looked, which is exactly what a census gate is for.
        let scene =
            RealmScene::from_scene_rows(&one_world_level()).expect("the one world projects");
        assert!(scene.get(RealmId::System(7)).is_some(), "System 7 renders");
        assert!(scene.get(RealmId::System(8)).is_some(), "System 8 renders");
        assert!(scene.get(RealmId::Planet(7)).is_some(), "Planet 7 renders");
        assert!(
            scene.get(RealmId::Station(7)).is_some(),
            "Station 7 renders"
        );
        assert!(scene.get(RealmId::Area(7)).is_some(), "Area 7 renders");
        // ★ S9: the Galaxy is named as a Galaxy, not as the `System(1)` stand-in it used to borrow.
        let galaxy = scene
            .get(RealmId::Galaxy(1))
            .expect("the Galaxy renders as the containing box");
        assert_eq!(galaxy.shape, BoxShape::Sphere { r: 180.0 });
        assert_eq!(galaxy.depth, 1, "the Galaxy is depth 1 (Universe ⊃ Galaxy)");
        assert_eq!(
            scene.get(RealmId::Universe),
            None,
            "the Universe ambient root is NOT rendered"
        );
        assert!(
            scene.get(vd_core::worldgen::PLANET_B).is_some(),
            "Planet B renders"
        );
        assert!(
            scene.get(vd_core::worldgen::PLANET_C).is_some(),
            "Planet C renders"
        );
        assert_eq!(
            scene.len(),
            8,
            "the 7 finite leaf realms + the Galaxy containing box"
        );
        // A rendered System keeps its TRUE nesting depth (Universe 0 ⊃ Galaxy 1 ⊃ System 2), even
        // though its Universe ancestor is skipped from the drawn set — depth is over the FULL level.
        assert_eq!(
            scene.get(RealmId::System(7)).expect("system 7 box").depth,
            2,
            "System 7 is depth 2 even with the ambient parent skipped"
        );
        assert_eq!(
            scene.get(RealmId::Planet(7)).expect("planet 7 box").depth,
            3,
            "Planet 7 is depth 3 (… ⊃ System ⊃ Planet)"
        );
        assert_eq!(
            scene.get(RealmId::System(7)).expect("system 7 box").shape,
            BoxShape::Sphere { r: 40.0 }
        );
        // The Aabb→Box projection is the "client render is FREE" proof: a Station/Area PLANTED as a
        // first-class box realm draws — with ZERO station/area-specific render code — as a Box of its
        // half-extents at its true forest depth.
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
    }

    #[test]
    fn the_drawn_set_is_exactly_the_finite_subset_of_the_one_worlds_forest() {
        // WHAT THIS REPLACES: the deleted `regions.json` single-source pin. The claim survives the
        // lane change — the client draws EXACTLY the sim's geometry — but it is now stated as a
        // property computed from the forest rather than a file round-trip: a region is drawn iff its
        // own outline fits the renderable extent, with no per-realm list anywhere.
        let regions = vd_physics::worldgen::realm_regions_for(0);
        let scene = RealmScene::from_scene_rows(&one_world_level()).expect("projects");
        let mut expected: Vec<RealmId> = regions
            .iter()
            .filter(|r| r.look.is_some())
            .map(|r| r.realm)
            .collect();
        expected.sort_unstable();
        let drawn: Vec<RealmId> = scene.iter().map(|(realm, _)| realm).collect();
        assert_eq!(drawn, expected, "the drawn set is the finite subset");
        assert!(!drawn.is_empty(), "a vacuous scene would assert nothing");
    }

    #[test]
    fn a_duplicate_realm_in_the_one_worlds_level_rejects() {
        // A repeated realm in the composed level is a composer bug — rejected LOUD (never silently
        // keeping whichever copy the input happened to list first).
        let mut level = one_world_level();
        level.push(level.first().expect("a non-empty world").clone());
        let err = RealmScene::from_scene_rows(&level).expect_err("a duplicate realm must reject");
        assert_eq!(err, SceneError::DuplicateRealm);
    }

    #[test]
    fn with_delta_adds_finite_skips_ambient_and_removes() {
        // VU AoI: a delta ADDS the realms that entered view (finite outlines framed, ambient shells
        // felt-not-framed) and REMOVES those that left. Atomic — returns a new scene the caller swaps.
        let base = RealmScene::from_scene_rows(&[row(
            RealmId::System(7),
            None,
            DVec3::ZERO,
            look_shell(40.0),
        )])
        .expect("base");
        let added = base
            .with_delta(
                &[
                    row(
                        RealmId::Planet(7),
                        Some(RealmId::System(7)),
                        DVec3::new(10.0, 0.0, 0.0),
                        look_shell(10.0),
                    ), // finite outline — drawn
                    // The ambient root STATES NO LOOK (the bound/look split: its shell is a
                    // containment promise the source never frames) — an empty bag: tracked,
                    // never drawn.
                    row(
                        RealmId::System(0),
                        None,
                        DVec3::ZERO,
                        vd_core::tlv::TlvWriter::new(vd_core::look::WINDOW_BODY_SCHEMA).finish(),
                    ),
                ],
                &[],
            )
            .expect("a well-formed add applies");
        assert!(
            added.get(RealmId::Planet(7)).is_some(),
            "the entered leaf is framed"
        );
        assert_eq!(
            added.get(RealmId::System(0)),
            None,
            "the ambient shell is felt, not framed"
        );
        assert_eq!(added.len(), 2, "System 7 (base) + Planet 7 (entered)");
        // A follow-up delta removes Planet 7 (it left AoI).
        let removed = added
            .with_delta(&[], &[RealmId::Planet(7)])
            .expect("a remove applies");
        assert_eq!(
            removed.get(RealmId::Planet(7)),
            None,
            "the departed realm is removed"
        );
        assert_eq!(removed.len(), 1);
    }

    #[test]
    fn with_delta_withdraws_a_box_when_its_row_stops_stating_one() {
        // THE WITHDRAWAL ARM. A row can arrive for a realm the client is already drawing and state
        // nothing drawable — the realm went to sleep, so its own outline is gone and no parent datum
        // replaced it. The previous box must go with the statement that authored it: keeping it would
        // leave a picture on screen that nothing on the wire claims any more.
        let drawn = RealmScene::from_scene_rows(&[row(
            RealmId::Planet(7),
            None,
            DVec3::new(10.0, 0.0, 0.0),
            look_shell(10.0),
        )])
        .expect("the look is drawn");
        assert_eq!(drawn.len(), 1);
        let withdrawn = drawn
            .with_delta(
                &[row(
                    RealmId::Planet(7),
                    None,
                    DVec3::new(10.0, 0.0, 0.0),
                    Vec::new(),
                )],
                &[],
            )
            .expect("a withdrawal applies");
        assert_eq!(withdrawn.get(RealmId::Planet(7)), None);
        assert!(withdrawn.is_empty());
        // A row that states a MARKER instead is a replacement, not a withdrawal — the sleeping realm
        // keeps a point of light. (Same arm of with_delta, the other outcome.)
        let marked = drawn
            .with_delta(
                &[row(
                    RealmId::Planet(7),
                    None,
                    DVec3::new(10.0, 0.0, 0.0),
                    luma_bag(3, 1.5),
                )],
                &[],
            )
            .expect("a marker applies");
        assert_eq!(
            marked.get(RealmId::Planet(7)).expect("marker box").body,
            BodyKind::Marker
        );
    }

    #[test]
    fn with_delta_recomputes_depth_when_a_parent_arrives_after_its_child() {
        // Independent streams carry no cross-stream order: a child's delta can land BEFORE its
        // parent's. Depth is recomputed each delta, so once the parent arrives the child nests correctly.
        let child_first = RealmScene::default()
            .with_delta(
                &[row(
                    RealmId::Planet(7),
                    Some(RealmId::System(7)),
                    DVec3::ZERO,
                    look_shell(10.0),
                )],
                &[],
            )
            .expect("the child arrives first");
        assert_eq!(
            child_first.get(RealmId::Planet(7)).expect("present").depth,
            0,
            "no parent in the scene yet ⇒ the child renders as a root (depth 0)"
        );
        let with_parent = child_first
            .with_delta(
                &[row(RealmId::System(7), None, DVec3::ZERO, look_shell(40.0))],
                &[],
            )
            .expect("the parent arrives next");
        assert_eq!(
            with_parent.get(RealmId::System(7)).expect("present").depth,
            0,
            "the arrived parent is the root"
        );
        assert_eq!(
            with_parent.get(RealmId::Planet(7)).expect("present").depth,
            1,
            "the child now nests one level under its arrived parent"
        );
    }

    #[test]
    fn with_delta_rejects_a_cyclic_parent_chain_loud() {
        // A delta whose merged parent links CYCLE is a server bug → the depth recompute stops loud and
        // the caller keeps the previous scene (atomic — never a partial mutation).
        let err = RealmScene::default()
            .with_delta(
                &[
                    row(
                        RealmId::System(7),
                        Some(RealmId::System(8)),
                        DVec3::ZERO,
                        look_shell(40.0),
                    ),
                    row(
                        RealmId::System(8),
                        Some(RealmId::System(7)),
                        DVec3::ZERO,
                        look_shell(40.0),
                    ),
                ],
                &[],
            )
            .expect_err("a cyclic delta must reject");
        assert_eq!(err, SceneError::CycleOrDepthExceeded);
    }

    #[test]
    fn scene_error_variants_render_their_loud_display_messages() {
        // The thiserror `#[error(...)]` Display arms — rendered so a malformed composed level fails
        // LOUD (never a silent empty). The other tests compare by value and never format, so without
        // this the Display arms stay uncovered.
        assert_eq!(
            SceneError::DuplicateRealm.to_string(),
            "duplicate realm id in the scene rows"
        );
        assert_eq!(
            SceneError::CycleOrDepthExceeded.to_string(),
            "parent chain cycles or exceeds the max nesting depth"
        );
        // Exercise the derived Clone + PartialEq (the caller counts and logs these).
        let e = SceneError::CycleOrDepthExceeded;
        assert_eq!(e.clone(), e);
        assert_ne!(e, SceneError::DuplicateRealm);
    }
}
