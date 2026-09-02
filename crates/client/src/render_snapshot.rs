//! The immutable render hand-off (P1.5 Slice 3 T4): the core thread publishes one of
//! these per step (lock-free, via `ArcSwap`); the windowed renderer (Tier-B
//! `vd-client-render`) loads it wait-free and samples it at its OWN display-time cursor.
//!
//! This is the seam that DECOUPLES the deterministic 20 Hz sim/netcode cadence (on its
//! own thread) from the variable display refresh rate: the renderer never touches the
//! live core, and the core never waits on the window. Motion stays smooth at ANY refresh
//! because the renderer interpolates at its display cursor — and it is still
//! no-prediction (the cursor is clamped into each entity's window by
//! [`EntityTrack::sample`](crate::interp::EntityTrack::sample); past the freshest tick an
//! entity FREEZES, never coasts). DRY: it carries the SAME [`DeliveredView`] +
//! [`RenderClock`] the headless [`DevState`](vd_devproto::DevState) path already uses, so
//! the window and `vdctl state` can never disagree about what was delivered.

use std::sync::Arc;

use glam::DVec3;
use vd_core::EntityId;
use vd_core::pose::FrameRef;
use vd_wire::channels::SubId;

use crate::interp::RenderPose;
use crate::net::ClientPhase;
use crate::realm_scene::RealmScene;
use crate::realm_view::RealmView;
use crate::render_clock::RenderClock;
use crate::view::DeliveredView;

/// An immutable snapshot of the delivered world + the render clock at one core step.
/// Cheap to clone (the view is `BTreeMap`s of `Copy` tracks; the scene is a shared `Arc`);
/// published every step and read wait-free by the renderer.
#[derive(Clone, Debug)]
pub struct RenderSnapshot {
    view: DeliveredView,
    clock: RenderClock,
    phase: ClientPhase,
    /// The boot-loaded realm-geometry projection (Visual Crossing Playground V2), carried on the
    /// SAME render seam as the poses so the renderer draws the realm boxes lock-free alongside the
    /// entity dots. Boot-loaded ONCE (a dev-config `boxes.json`) and shared by `Arc` — a per-step
    /// clone of the snapshot is a pointer bump, never a deep copy of the box map. Default EMPTY
    /// (no boxes until a scene is loaded), so every existing pose-only path is unchanged.
    scene: Arc<RealmScene>,
    /// The streamed LIVE realm placements (SLICE 6 S4). Carried RAW — resolved to positions only at
    /// [`RenderSnapshot::scene_at`], on the DISPLAY cursor. Previously the core thread baked an
    /// overlaid scene once per 20 Hz step from whatever had most recently arrived, with no cursor, so
    /// the ground was on a different time axis from the player standing on it.
    realm_view: RealmView,
    /// THE STAR CATALOGUE THE CLIENT HOLDS (S11) — the galaxy, ready to draw.
    ///
    /// ★ BEHIND AN `Arc`, AND THAT IS THE WHOLE DESIGN. A snapshot is cloned every step. At the target
    /// census this holds 150,000 rows, so a by-value field would deep-copy 7.0 MB twenty times a
    /// second for a galaxy that never moves. The pointer bump costs nothing, and the sky is replaced
    /// only when its generation changes — which is almost never.
    ///
    /// `None` until the sky is whole. A partial sky is never carried here: half a galaxy is not a
    /// smaller galaxy, it is a galaxy with holes, and the holes look exactly like stars that are not
    /// there.
    sky: Option<Arc<SkyDraw>>,
    /// THE REALM THE SESSION IS STANDING IN (S11) — the composed scene's own origin.
    ///
    /// Carried so the renderer can find the observer's ANCHOR in the star catalogue. It adds no wire
    /// data: the gateway already states it on `ServerControlMsg::RealmRegistry.origin`, and the client
    /// already stores it. It was simply private, with no way to reach the render path.
    origin: Option<vd_core::pose::RealmId>,
    /// THE SKY ANCHOR (owner ruling 2026-09-02 R1): the origin realm placed in the galaxy's frame,
    /// as the gateway last stated it on the per-tick lane; `None` until the chain reaches the
    /// galaxy. The renderer places its one star cloud by it — nothing else moves the sky.
    sky_anchor: Option<vd_core::pose::StampedPose>,
}

/// The sky, as the renderer receives it (S11): the rows, and the generation that identifies them.
///
/// The generation is carried so the renderer can tell "the same sky again" from "a different sky" with
/// one integer compare, instead of walking 150,000 rows to find out that nothing changed.
#[derive(Debug, Clone, PartialEq)]
pub struct SkyDraw {
    /// The stars, in the catalogue's own order.
    pub rows: Vec<vd_core::look::StarRow>,
    /// The generation these rows fold to.
    pub generation: u64,
}

/// ONE STAR, READY TO DRAW (S11): where it is relative to the observer, and what colour and size it
/// should be.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct StarPoint {
    /// Metres from the observer's own star system, in the galaxy's frame.
    pub pos_m: vd_core::glam::DVec3,
    /// The Morgan-Keenan spectral class code, which picks the colour.
    pub class_code: u8,
    /// Brightness in solar luminosities, which sizes the point.
    pub luma_lsun: f64,
}

impl SkyDraw {
    /// THE CLOUD'S VERTICES, BUILT ONCE (owner ruling 2026-09-02 R1): every star of the catalogue as
    /// metres from `reference`, a galaxy cell the renderer picks when it first builds the cloud and
    /// keeps for the cloud's life. Nothing here depends on where the observer stands — that is the
    /// whole point. The observer's position enters ONLY as the cloud's transform, per frame, from
    /// the sky anchor ([`sky_cloud_transform`]); the vertex buffer is uploaded once per catalogue.
    ///
    /// ★ NO STAR IS LEFT OUT. This used to drop the observer's own star system ("you are inside it")
    /// and to re-anchor on it, which is what turned the sky black in any realm the catalogue does not
    /// name — a hull, a planet, a station. The observer's own star is a point of light at its true
    /// place; when the star realm runs it draws its own body over that point at the same brightness.
    ///
    /// ★ THE SUBTRACTION IS EXACT, AND THAT IS WHY THE CLIENT MAY DO IT. Both values are whole cells
    /// at ONE tier — the galaxy's two-metre step — and every generated system is exactly
    /// cell-aligned. `i128` for the difference: two cells near opposite edges of the galaxy would
    /// overflow an `i64` subtraction, and an overflow here is a star drawn behind you.
    #[must_use]
    pub fn points_from(&self, reference: vd_core::glam::I64Vec3) -> Vec<StarPoint> {
        let edge_m = vd_core::pose::Tier::Galaxy.cell_edge_m();
        self.rows
            .iter()
            .map(|r| StarPoint {
                pos_m: vd_core::glam::DVec3::new(
                    (i128::from(r.cell.x) - i128::from(reference.x)) as f64 * edge_m,
                    (i128::from(r.cell.y) - i128::from(reference.y)) as f64 * edge_m,
                    (i128::from(r.cell.z) - i128::from(reference.z)) as f64 * edge_m,
                ),
                class_code: r.class_code,
                luma_lsun: r.luma_lsun,
            })
            .collect()
    }
}

/// THE CLOUD'S PLACEMENT, PER FRAME (owner ruling 2026-09-02 R1): the rigid transform that carries
/// a cloud built from `reference` ([`SkyDraw::points_from`]) into the render space — the origin
/// realm's axes, centred on the eye. The `anchor` is the gateway's statement of the origin realm in
/// the galaxy's frame; `eye_m` is the eye in the origin realm's frame, in metres.
///
/// A star at galaxy position `g` (metres) sits in the origin's frame at `qᐨ¹·(g − a)` where `a` is
/// the anchor's position and `q` the origin frame's orientation in the galaxy; relative to the eye
/// it is that minus `eye_m`. With vertices `v = g − R·edge` this is `qᐨ¹·v + [qᐨ¹·(R·edge − a) − eye_m]`,
/// so the entity's rotation is `qᐨ¹` and its translation the bracket. Computed in f64, narrowed once.
///
/// ★ PARALLAX IS THIS TRANSLATION MOVING. Nothing draws it on purpose: when the origin realm warps
/// across the galaxy, the anchor walks, the bracket walks with it, and the near stars slide past the
/// far ones because they sit at different depths in one cloud.
#[must_use]
pub fn sky_cloud_transform(
    reference: vd_core::glam::I64Vec3,
    anchor: &vd_core::pose::StampedPose,
    eye_m: vd_core::glam::DVec3,
) -> (vd_core::glam::DVec3, vd_core::glam::DQuat) {
    let edge_m = vd_core::pose::Tier::Galaxy.cell_edge_m();
    let cell = anchor.pos.cell();
    let reference_from_anchor_m = vd_core::glam::DVec3::new(
        (i128::from(reference.x) - i128::from(cell.x)) as f64 * edge_m,
        (i128::from(reference.y) - i128::from(cell.y)) as f64 * edge_m,
        (i128::from(reference.z) - i128::from(cell.z)) as f64 * edge_m,
    ) - anchor.pos.offset();
    let rotation = anchor.orient.inverse();
    (rotation * reference_from_anchor_m - eye_m, rotation)
}

/// THE ONE LIGHT YEAR NO TWO STAR SYSTEMS COME CLOSER THAN (owner ruling 2026-08-24, Q3): the
/// nearest any star can be to an observer standing in another system, and so the depth against
/// which the cloud's single-precision placement error is judged.
const SKY_NEAREST_STAR_FLOOR_M: f64 = vd_core::units::LIGHT_YEAR_M;

/// The largest angular error the cloud's placement may commit: a tenth of a pixel at the capture
/// frame's field of view — invisible by construction, and a fixed fraction of the smallest thing a
/// frame can show.
const SKY_DIRECTION_TOLERANCE_RAD: f64 = 1.0e-4;

/// ★ WHEN THE CLOUD MUST BE REBASED. The cloud's translation is narrowed to `f32` for the GPU, and an
/// `f32` of magnitude `T` carries an error of about `T · ε`. A star at the nearest lawful distance
/// then moves on screen by `T · ε / floor` radians, so the translation may grow to
/// `floor · tolerance / ε` before that error reaches the tolerance — about 8 × 10¹⁸ m, which is
/// wider than the galaxy. Inside one galaxy a cloud is therefore built ONCE and never again; the
/// rebase exists so that a galaxy-to-galaxy journey, when it comes, cannot quietly smear the sky.
#[must_use]
pub fn sky_rebase_bound_m() -> f64 {
    SKY_NEAREST_STAR_FLOOR_M * SKY_DIRECTION_TOLERANCE_RAD / f64::from(f32::EPSILON)
}

/// Does this translation exceed the rebase bound? See [`sky_rebase_bound_m`].
#[must_use]
pub fn sky_rebase_due(translation_m: vd_core::glam::DVec3) -> bool {
    translation_m.length() > sky_rebase_bound_m()
}

#[derive(Debug, Clone, PartialEq, Default)]
pub struct StarCloud {
    /// The star centre, repeated for each of its four corners. Metres from the cloud's reference cell (see `SkyDraw::points_from`).
    pub positions: Vec<[f32; 3]>,
    /// Which corner of the sprite this vertex is: (-1,-1), (1,-1), (1,1), (-1,1).
    pub corners: Vec<[f32; 2]>,
    /// The class colour, from [`crate::realm_scene::MARKER_CLASS_SRGB`].
    pub colors: Vec<[f32; 4]>,
    /// The star's own world radius before the apparent-size floor is applied — from
    /// [`crate::realm_scene::marker_look`], so the √L convention is stated in ONE place.
    pub base_radius_m: Vec<f32>,
    /// Two triangles per star.
    pub indices: Vec<u32>,
}

impl StarCloud {
    /// The corner signs, in the order the two triangles below index them.
    const CORNERS: [[f32; 2]; 4] = [[-1.0, -1.0], [1.0, -1.0], [1.0, 1.0], [-1.0, 1.0]];

    /// How many stars this cloud holds.
    #[must_use]
    pub fn len(&self) -> usize {
        self.base_radius_m.len() / 4
    }

    /// Whether the cloud holds no stars.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.base_radius_m.is_empty()
    }

    /// Build the cloud from placed stars.
    ///
    /// ★ THE COLOUR AND THE RADIUS COME FROM [`crate::realm_scene::marker_look`], never from a second
    /// rule written here. That function owns the √L convention (a sphere whose radius goes as the
    /// square root of luminosity has flux proportional to luminosity) and the class palette. Two
    /// spellings of one convention is how a starfield and its own pixel gate come to disagree.
    #[must_use]
    pub fn build(points: &[StarPoint]) -> StarCloud {
        let n = points.len();
        let mut cloud = StarCloud {
            positions: Vec::with_capacity(n * 4),
            corners: Vec::with_capacity(n * 4),
            colors: Vec::with_capacity(n * 4),
            base_radius_m: Vec::with_capacity(n * 4),
            indices: Vec::with_capacity(n * 6),
        };
        for (i, p) in points.iter().enumerate() {
            let look = crate::realm_scene::marker_look(p.class_code, p.luma_lsun);
            let pos = p.pos_m.as_vec3().to_array();
            let base = look.base_radius_m as f32;
            for corner in StarCloud::CORNERS {
                cloud.positions.push(pos);
                cloud.corners.push(corner);
                cloud.colors.push(look.color_rgba);
                cloud.base_radius_m.push(base);
            }
            let v = (i * 4) as u32;
            cloud
                .indices
                .extend_from_slice(&[v, v + 1, v + 2, v, v + 2, v + 3]);
        }
        cloud
    }
}

impl RenderSnapshot {
    /// A snapshot with NO realm boxes (the pose-only default — every pre-V2 construction site).
    #[must_use]
    pub fn new(view: DeliveredView, clock: RenderClock, phase: ClientPhase) -> RenderSnapshot {
        RenderSnapshot::with_scene(view, clock, phase, Arc::new(RealmScene::default()))
    }

    /// A snapshot carrying a boot-loaded [`RealmScene`] + the streamed realm placements on the render
    /// seam (V2). The `Arc` is the one the core holds — cloning the snapshot each step just bumps its
    /// refcount.
    #[must_use]
    pub fn with_scene(
        view: DeliveredView,
        clock: RenderClock,
        phase: ClientPhase,
        scene: Arc<RealmScene>,
    ) -> RenderSnapshot {
        RenderSnapshot::with_realms(view, clock, phase, scene, RealmView::default())
    }

    /// A snapshot carrying the boot scene AND the streamed live realm placements (SLICE 6 S4). Both
    /// ride the render seam unresolved; [`RenderSnapshot::scene_at`] evaluates them at the display
    /// cursor, so the ground and the entities standing on it are read at ONE instant.
    #[must_use]
    pub fn with_realms(
        view: DeliveredView,
        clock: RenderClock,
        phase: ClientPhase,
        scene: Arc<RealmScene>,
        realm_view: RealmView,
    ) -> RenderSnapshot {
        RenderSnapshot {
            view,
            clock,
            phase,
            scene,
            realm_view,
            sky: None,
            origin: None,
            sky_anchor: None,
        }
    }

    /// The same snapshot, carrying the sky the client holds (S11).
    ///
    /// Separate from the constructors above rather than a seventh parameter on each: every existing
    /// construction site means "no sky", and threading `None` through all of them would say nothing
    /// while touching everything.
    #[must_use]
    pub fn with_sky(mut self, sky: Option<Arc<SkyDraw>>) -> RenderSnapshot {
        self.sky = sky;
        self
    }

    /// The sky to draw, or `None` while the client holds no whole one.
    #[must_use]
    pub fn sky(&self) -> Option<&Arc<SkyDraw>> {
        self.sky.as_ref()
    }

    /// The realm the session is standing in — the composed scene's origin.
    #[must_use]
    pub fn origin(&self) -> Option<vd_core::pose::RealmId> {
        self.origin
    }

    /// The same snapshot, carrying the origin realm (S11).
    #[must_use]
    pub fn with_origin(mut self, origin: Option<vd_core::pose::RealmId>) -> RenderSnapshot {
        self.origin = origin;
        self
    }

    /// The sky anchor the gateway last stated (see [`RenderSnapshot::sky_anchor`]).
    #[must_use]
    pub fn sky_anchor(&self) -> Option<vd_core::pose::StampedPose> {
        self.sky_anchor
    }

    #[must_use]
    pub fn with_sky_anchor(mut self, anchor: Option<vd_core::pose::StampedPose>) -> RenderSnapshot {
        self.sky_anchor = anchor;
        self
    }

    /// The scene to DRAW at `cursor`: the boot geometry with every streamed realm's placement
    /// interpolated at that cursor — the SAME cursor [`RenderSnapshot::rendered`] samples entities at.
    ///
    /// This is the one-moment seam. An empty feed (walk/static scale, no moving realm) returns the boot
    /// scene by clone-of-pointer-content, so those rigs are unchanged. The map rebuild is a handful of
    /// boxes per drawn frame — deliberately preferred over caching, because a cache would have to be
    /// invalidated per cursor and the whole point is that the cursor moves every frame.
    #[must_use]
    pub fn scene_at(&self, cursor: f64) -> RealmScene {
        if self.realm_view.is_empty() {
            return (*self.scene).clone();
        }
        self.scene.overlaid_at(&self.realm_view, cursor)
    }

    /// The scene to DRAW at wall-time `now_s` — [`RenderSnapshot::scene_at`] on this snapshot's own
    /// render cursor. Before the clock is anchored (no snapshot has landed yet) there is no cursor and
    /// nothing has been streamed, so this is the boot scene. THE ONE place the unanchored case is
    /// decided, so no renderer site has to.
    #[must_use]
    pub fn scene_now(&self, now_s: f64) -> RealmScene {
        match self.cursor(now_s) {
            Some(cursor) => self.scene_at(cursor),
            None => (*self.scene).clone(),
        }
    }

    /// The freshest tick the REALM feed has delivered (diagnosis; see the dev state).
    #[must_use]
    pub fn realm_view(&self) -> &RealmView {
        &self.realm_view
    }

    /// The boot-loaded realm-box scene to draw (empty until a `boxes.json` is loaded). The renderer
    /// iterates this alongside [`RenderSnapshot::rendered`] to draw the translucent realm volumes.
    #[must_use]
    pub fn scene(&self) -> &RealmScene {
        &self.scene
    }

    /// The client lifecycle phase (so the renderer can show "connecting…" vs live).
    #[must_use]
    pub fn phase(&self) -> ClientPhase {
        self.phase
    }

    /// The render cursor at display wall-time `now_s` (the SAME shared-monotonic-epoch
    /// timeline the core thread anchored on); `None` until the first snapshot anchored it.
    #[must_use]
    pub fn cursor(&self, now_s: f64) -> Option<f64> {
        self.clock.cursor(now_s)
    }

    /// The composited poses to draw at display wall-time `now_s` — each entity exactly
    /// once, interpolated at the DISPLAY cursor (smooth at any refresh). Empty until the
    /// clock is anchored (nothing delivered yet).
    #[must_use]
    pub fn rendered(&self, now_s: f64) -> Vec<(EntityId, SubId, RenderPose)> {
        match self.clock.cursor(now_s) {
            Some(cursor) => self.view.rendered(cursor),
            None => Vec::new(),
        }
    }

    /// Flatten a delivered pose to the drawn point — a PASSTHROUGH ([`DeliveredView::world_pos`]). The
    /// chain of shards already expressed it from the centre of the realm this session stands in; the
    /// client subtracts nothing and picks no unit (the pose carries the one it was shipped with).
    #[must_use]
    pub fn world_pos(&self, pose: &RenderPose) -> DVec3 {
        self.view.world_pos(pose)
    }

    /// The freshest delivered universe tick (the run-stable capture-alignment quantity);
    /// `None` until the first snapshot anchored the clock.
    #[must_use]
    pub fn freshest_tick(&self) -> Option<u64> {
        self.clock.freshest_tick()
    }

    /// This client's own entity (so the renderer can highlight its dot), if known.
    #[must_use]
    pub fn own_entity(&self) -> Option<EntityId> {
        self.view.own_entity()
    }

    /// The player's LOCATION label (realm) for the stats HUD — the own entity's
    /// authoritative frame, NOT a raw shard id. `None` until the own entity has a
    /// delivered pose.
    #[must_use]
    pub fn location(&self) -> Option<String> {
        self.view.own_location_frame().map(FrameRef::label)
    }
}

#[cfg(test)]
mod tests {
    /// Flatten a RenderPose to world metres (poses ride the lattice normalized since the cell
    /// activation — `.pos` raw is a sub-cell residual).
    fn rp_world(p: &crate::interp::RenderPose) -> vd_core::glam::DVec3 {
        vd_core::pose::LatticePos::at(p.cell, p.pos)
            .delta_m(vd_core::pose::LatticePos::default(), p.tier)
    }
    use super::*;
    use glam::DVec3;
    use vd_core::entity_kind::EntityKind;
    use vd_core::geometry::Boundary;
    use vd_core::look::look_bag;
    use vd_core::pose::{FrameRef, RealmId, StampedPose};
    use vd_core::{TickId, UniverseTick};
    use vd_wire::channels::{EntitySnap, SceneRow, SnapshotDatagram};

    use crate::tuning::ClientInterpTuning;

    fn ent(seq: u64) -> EntityId {
        EntityId::pack(EntityKind::Player, 1, seq, seq as u32)
    }

    /// ONE COMPOSED SCENE ROW — a realm at `pos` that draws ITSELF as a sphere of radius `r`. The
    /// look bag is the whole appearance statement; the client applies no transform to it.
    fn scene_row(realm: RealmId, pos: DVec3, r: f64) -> SceneRow {
        SceneRow {
            realm,
            parent: None,
            pose: StampedPose::at_rest(
                FrameRef::SystemSpace { system_seed: 1 },
                pos,
                UniverseTick(100),
            ),
            bag: look_bag(&Boundary::Shell { r }),
        }
    }

    fn snap(frame_id: u64, tick: u64, entities: Vec<(EntityId, f64)>) -> SnapshotDatagram {
        SnapshotDatagram {
            sub: SubId(0),
            frame_id,
            source_tick: TickId(1),
            universe_tick: UniverseTick(tick),
            entities: entities
                .into_iter()
                .map(|(entity, x)| EntitySnap {
                    entity,
                    pose: StampedPose::at_rest(
                        FrameRef::SystemSpace { system_seed: 1 },
                        DVec3::new(x, 0.0, 0.0),
                        UniverseTick(tick),
                    ),
                })
                .collect(),
        }
    }

    #[test]
    fn a_default_snapshot_carries_an_empty_scene_and_with_scene_carries_the_streamed_one() {
        use crate::realm_scene::{BoxShape, RealmScene};
        use std::sync::Arc;
        use vd_core::pose::RealmId;

        // The pose-only `new` carries an empty scene (every pre-V2 path is unchanged).
        let bare = RenderSnapshot::new(
            DeliveredView::default(),
            RenderClock::new(ClientInterpTuning::DEFAULT),
            ClientPhase::Connecting,
        );
        assert!(bare.scene().is_empty(), "new() ⇒ no realm boxes");

        // `with_scene` carries the COMPOSED scene (one level of rows) through the render seam.
        let scene =
            RealmScene::from_scene_rows(&[scene_row(RealmId::System(7), DVec3::ZERO, 100.0)])
                .expect("scene");
        let s = RenderSnapshot::with_scene(
            DeliveredView::default(),
            RenderClock::new(ClientInterpTuning::DEFAULT),
            ClientPhase::Active,
            Arc::new(scene),
        );
        assert_eq!(s.scene().len(), 1, "the drawn box is carried on the seam");
        assert_eq!(
            s.scene().get(RealmId::System(7)).expect("box").shape,
            BoxShape::Sphere { r: 100.0 }
        );
    }

    #[test]
    fn before_any_snapshot_it_is_empty_and_locationless() {
        let snap = RenderSnapshot::new(
            DeliveredView::default(),
            RenderClock::new(ClientInterpTuning::DEFAULT),
            ClientPhase::Connecting,
        );
        assert_eq!(snap.phase(), ClientPhase::Connecting);
        assert_eq!(snap.cursor(5.0), None, "no anchor ⇒ no cursor");
        assert!(snap.rendered(5.0).is_empty(), "no anchor ⇒ nothing to draw");
        assert_eq!(snap.own_entity(), None);
        assert_eq!(snap.location(), None);
        // The realm feed's own freshness, read straight through — this is the diagnosis surface an agent
        // and the dev overlay both read to tell "the feed is quiet" apart from "the feed is dead", so it
        // must survive a refactor that has no other reason to keep it.
        assert!(
            snap.realm_view().is_empty(),
            "a fresh snapshot has heard nothing from the realm feed yet"
        );
    }

    #[test]
    fn an_unanchored_clock_suppresses_even_a_populated_view() {
        // The cursor==None gate is the ANCHOR, not an empty view: a view WITH a delivered
        // track but a clock that never observed a tick renders nothing — while own_entity
        // / location (cursor-independent) still read.
        let mut view = DeliveredView::default();
        view.on_snapshot(
            &std::collections::BTreeSet::from([SubId(0)]),
            snap(1, 10, vec![(ent(1), 0.0)]),
        );
        view.set_own_entity(ent(1));
        let snap = RenderSnapshot::new(
            view,
            RenderClock::new(ClientInterpTuning::DEFAULT), // never observed → unanchored
            ClientPhase::Active,
        );
        assert_eq!(snap.cursor(5.0), None, "clock unanchored");
        assert!(
            snap.rendered(5.0).is_empty(),
            "no cursor ⇒ nothing rendered, even with a delivered track"
        );
        assert_eq!(snap.own_entity(), Some(ent(1)), "own is cursor-independent");
        assert_eq!(snap.location(), Some("System 1".to_owned()));
    }

    #[test]
    fn exposes_freshest_tick_and_world_pos_when_anchored() {
        let mut view = DeliveredView::default();
        view.on_snapshot(
            &std::collections::BTreeSet::from([SubId(0)]),
            snap(1, 10, vec![(ent(1), 4.0)]),
        );
        view.set_own_entity(ent(1));
        let mut clock = RenderClock::new(ClientInterpTuning::DEFAULT);
        clock.observe(UniverseTick(10), 100.0);
        let s = RenderSnapshot::new(view, clock, ClientPhase::Active);
        assert_eq!(s.freshest_tick(), Some(10), "the anchored tick");
        let rendered = s.rendered(100.0);
        assert_eq!(rendered.len(), 1);
        // world_pos flattens the tiered pose — the shard chain already measured this from the
        // centre of the realm the session stands in, and the pose carries the unit its cell is
        // counted in; since the cell activation the value rides the integer half.
        let pose = rendered[0].2;
        assert_eq!(s.world_pos(&pose), rp_world(&pose));
    }

    #[test]
    fn freshest_tick_is_none_and_world_pos_falls_back_before_anchor() {
        let s = RenderSnapshot::new(
            DeliveredView::default(),
            RenderClock::new(ClientInterpTuning::DEFAULT), // unanchored
            ClientPhase::Connecting,
        );
        assert_eq!(s.freshest_tick(), None);
        // world_pos does not depend on the clock/cursor at all: it flattens the delivered position, so a
        // pose reduces to itself, finite and correct, even before the anchor.
        let pose = RenderPose {
            frame: FrameRef::SystemSpace { system_seed: 1 },
            cell: glam::I64Vec3::ZERO,
            pos: DVec3::new(1.0, 2.0, 3.0),
            orient: glam::DQuat::IDENTITY,
            tier: vd_core::pose::Tier::Fine,
        };
        assert_eq!(s.world_pos(&pose), DVec3::new(1.0, 2.0, 3.0));
    }

    #[test]
    fn it_interpolates_the_delivered_view_at_the_display_cursor() {
        // Two snapshots a tick-window apart; the renderer samples at ITS display cursor,
        // not the core's — so motion is smooth at any refresh rate.
        let mut view = DeliveredView::default();
        view.on_snapshot(
            &std::collections::BTreeSet::from([SubId(0)]),
            snap(1, 10, vec![(ent(1), 0.0)]),
        );
        view.on_snapshot(
            &std::collections::BTreeSet::from([SubId(0)]),
            snap(2, 12, vec![(ent(1), 10.0)]),
        );
        view.set_own_entity(ent(1));

        // Anchor the clock at tick 12 (the freshest) at wall-time 100.0, default
        // tuning (2.4-tick buffer). cursor(100.0) = 12 - 2.4 = 9.6 → before the window
        // → clamps to prev (x=0).
        let mut clock = RenderClock::new(ClientInterpTuning::DEFAULT);
        clock.observe(UniverseTick(12), 100.0);
        let snap = RenderSnapshot::new(view, clock, ClientPhase::Active);

        assert_eq!(snap.phase(), ClientPhase::Active);
        assert_eq!(snap.own_entity(), Some(ent(1)));
        assert_eq!(snap.location(), Some("System 1".to_owned()));

        // At the anchor instant the cursor sits behind the window → frozen at prev.
        assert_eq!(snap.cursor(100.0), Some(9.6));
        let early = snap.rendered(100.0);
        assert_eq!(early.len(), 1);
        assert_eq!(early[0].0, ent(1));
        assert_eq!(rp_world(&early[0].2), DVec3::new(0.0, 0.0, 0.0));

        // 0.13 s later the display cursor advances into the window (9.6 + 0.13*20 =
        // 12.2 → clamps to current x=10): a DIFFERENT sample from the same snapshot,
        // proving display-rate interpolation off one published frame.
        let later = snap.rendered(100.13);
        assert_eq!(rp_world(&later[0].2), DVec3::new(10.0, 0.0, 0.0));
    }
}

#[cfg(test)]
mod slice6_tests {
    use super::*;
    use vd_core::UniverseTick;
    use vd_core::geometry::Boundary;
    use vd_core::look::look_bag;
    use vd_core::pose::{RealmId, StampedPose};
    use vd_wire::channels::{RealmSnap, RealmSnapshotDatagram, SceneRow};

    use crate::realm_scene::RealmScene;
    use crate::tuning::ClientInterpTuning;

    const REALM: RealmId = RealmId::Planet(1);

    /// The scene as the composed LEVEL states it: one realm drawing its own outline at the origin.
    /// The streamed placements below move it; nothing here states where it is.
    fn level_scene() -> RealmScene {
        RealmScene::from_scene_rows(&[SceneRow {
            realm: REALM,
            parent: None,
            pose: StampedPose::at_rest(
                FrameRef::SystemSpace { system_seed: 1 },
                DVec3::ZERO,
                UniverseTick(0),
            ),
            bag: look_bag(&Boundary::Shell { r: 1.0 }),
        }])
        .expect("scene")
    }

    /// SLICE 6 S4 — THE SHAKE ACCEPTANCE. A realm moving at a constant 1 m/tick is streamed on
    /// CONSECUTIVE ticks. Sampled at a continuous sweep of render cursors, the drawn centre must track
    /// the cursor CONTINUOUSLY — no staircase.
    ///
    /// Before this slice the scene was baked from the newest ARRIVAL with no cursor at all, so the
    /// drawn centre held flat between deliveries and jumped when one landed, while the player standing
    /// on it moved smoothly at the cursor. That difference — a step against a slide — is the shake.
    #[test]
    fn slice6_the_drawn_realm_centre_tracks_the_cursor_continuously() {
        let mut realms = RealmView::default();
        for t in 10..=30u64 {
            realms.on_realm_snapshot(
                None,
                RealmSnapshotDatagram {
                    sub: SubId(0),
                    frame_id: t,
                    source_tick: vd_core::TickId(t),
                    universe_tick: UniverseTick(t),
                    origin_epoch: 0,
                    sky_anchor: None,
                    realms: vec![RealmSnap {
                        realm: REALM,
                        // The edge HEAD (proto_minor 8): REALM's own frame; `pose.frame` is the TAIL.
                        frame: vd_core::pose::frame_for_realm(REALM, None)
                            .expect("a seeded realm resolves"),
                        pose: StampedPose::at_rest(
                            FrameRef::SystemSpace { system_seed: 1 },
                            DVec3::new(t as f64, 0.0, 0.0),
                            UniverseTick(t),
                        ),
                    }],
                },
            );
        }
        let snap = RenderSnapshot::with_realms(
            DeliveredView::default(),
            RenderClock::new(ClientInterpTuning::DEFAULT),
            ClientPhase::Active,
            Arc::new(level_scene()),
            realms,
        );
        // Sweep the cursor across a whole tick in tenths and require the centre to follow it exactly.
        let mut prev = f64::NEG_INFINITY;
        for step in 0..=10 {
            let cursor = 20.0 + f64::from(step) / 10.0;
            let rbox = snap.scene_at(cursor);
            let rbox = rbox.get(REALM).expect("box");
            // Flatten the tiered centre (H-21): the streamed pose rides the lattice, so `.offset()`
            // alone would read only the sub-cell residual.
            let centre = rbox
                .center
                .delta_m(vd_core::pose::LatticePos::default(), rbox.tier)
                .x;
            assert!(
                (centre - cursor).abs() < 1e-9,
                "cursor {cursor} should draw the realm at {cursor}, got {centre}",
            );
            assert!(
                centre > prev,
                "the centre must advance with the cursor, not step"
            );
            prev = centre;
        }
    }

    /// SLICE 6 S4 — a scene with NO streamed placements is the level scene as it arrived, so
    /// static/walk-scale rigs are unchanged by the whole slice.
    #[test]
    fn slice6_an_unstreamed_scene_is_the_level_scene() {
        let level = level_scene();
        let snap = RenderSnapshot::with_realms(
            DeliveredView::default(),
            RenderClock::new(ClientInterpTuning::DEFAULT),
            ClientPhase::Active,
            Arc::new(level.clone()),
            RealmView::default(),
        );
        assert_eq!(snap.scene_at(123.0), level);
        // And before the clock is anchored there is no cursor at all — still the level scene.
        assert_eq!(snap.scene_now(9.0), level);
    }

    /// ★ THE SKY IS PLACED RELATIVE TO THE OBSERVER, EXACTLY (S11).
    ///
    /// This is the whole reason the catalogue may ship RAW — one sky for every player in the galaxy,
    /// sent once and never again. The observer's own anchor is already IN it, because the catalogue
    /// applies no observer filter. So a crossing costs a lookup and nothing on the wire.
    #[test]
    fn the_sky_is_placed_around_the_observers_own_system_by_exact_integer_subtraction() {
        use vd_core::glam::{DVec3, I64Vec3};
        use vd_core::pose::RealmId;
        let star = |n: u64, cell: I64Vec3| vd_core::look::StarRow {
            realm: RealmId::System(n),
            cell,
            class_code: 6,
            luma_lsun: 0.25,
        };
        // Three systems on the galaxy's two-metre lattice. System(2) is where the observer stands.
        let sky = SkyDraw {
            rows: vec![
                star(1, I64Vec3::new(100, 0, 0)),
                star(2, I64Vec3::new(400, -50, 7)),
                star(3, I64Vec3::new(400, -50, 107)),
            ],
            generation: 1,
        };

        // THE CLOUD IS BUILT FROM A REFERENCE CELL, and every star is in it — the reference's own
        // star included, at zero. Take System(2)'s cell as the reference.
        let pts = sky.points_from(I64Vec3::new(400, -50, 7));
        assert_eq!(pts.len(), 3, "every star of the catalogue, nobody left out");

        let edge = vd_core::pose::Tier::Galaxy.cell_edge_m();
        // System(2) sits AT the reference.
        let here = pts
            .iter()
            .find(|p| p.pos_m == DVec3::ZERO)
            .expect("the reference's own star");
        assert_eq!(here.class_code, 6);
        // System(3) sits exactly 100 cells along +z from the reference.
        let up = pts.iter().find(|p| p.pos_m.z > 0.0).expect("the +z one");
        assert_eq!(up.pos_m, DVec3::new(0.0, 0.0, 100.0 * edge));
        // System(1) is 300 cells back in x, 50 up in y, 7 down in z.
        let back = pts.iter().find(|p| p.pos_m.x < 0.0).expect("the -x one");
        assert_eq!(
            back.pos_m,
            DVec3::new(-300.0 * edge, 50.0 * edge, -7.0 * edge)
        );
        // ...and the photometrics ride along, because they colour and size the point. Compared as
        // VALUES rather than through an `all` predicate: the predicate's false arm is a region no
        // passing test can reach, and an unreachable region is an uncoverable one (HR5).
        assert_eq!(
            pts.iter().map(|p| p.class_code).collect::<Vec<_>>(),
            vec![6, 6, 6]
        );
        assert_eq!(
            pts.iter().map(|p| p.luma_lsun).collect::<Vec<_>>(),
            vec![0.25, 0.25, 0.25]
        );

        // ★ THE OBSERVER ENTERS ONLY THROUGH THE TRANSFORM (owner ruling 2026-09-02 R1). An anchor
        // stating the origin realm AT the reference cell, unrotated, with the eye at the origin's
        // centre, carries the cloud by nothing at all...
        let at_reference = vd_core::pose::StampedPose::at_rest(
            vd_core::pose::FrameRef::GalaxySpace { galaxy_seed: 1 },
            DVec3::ZERO,
            vd_core::ids::UniverseTick(0),
        );
        let at_reference = vd_core::pose::StampedPose {
            pos: vd_core::pose::LatticePos::at(I64Vec3::new(400, -50, 7), DVec3::ZERO),
            ..at_reference
        };
        let (t, q) = sky_cloud_transform(I64Vec3::new(400, -50, 7), &at_reference, DVec3::ZERO);
        assert_eq!(t, DVec3::ZERO);
        assert_eq!(q, vd_core::glam::DQuat::IDENTITY);
        // ...an origin ten cells further along +x, with a half-metre residual and an eye standing
        // 3 m from the origin's centre, carries it back by exactly that much...
        let moved = vd_core::pose::StampedPose {
            pos: vd_core::pose::LatticePos::at(
                I64Vec3::new(410, -50, 7),
                DVec3::new(0.5, 0.0, 0.0),
            ),
            ..at_reference
        };
        let (t, _) =
            sky_cloud_transform(I64Vec3::new(400, -50, 7), &moved, DVec3::new(3.0, 0.0, 0.0));
        assert_eq!(t, DVec3::new(-10.0 * edge - 0.5 - 3.0, 0.0, 0.0));
        // ...and an origin frame turned half a circle about z turns the cloud the other way, so a
        // star ahead in the galaxy is behind the observer.
        let turned = vd_core::pose::StampedPose {
            orient: vd_core::glam::DQuat::from_xyzw(0.0, 0.0, 1.0, 0.0),
            ..moved
        };
        let (t, q) = sky_cloud_transform(I64Vec3::new(400, -50, 7), &turned, DVec3::ZERO);
        assert_eq!(q, vd_core::glam::DQuat::from_xyzw(0.0, 0.0, -1.0, 0.0));
        assert_eq!(t, DVec3::new(10.0 * edge + 0.5, 0.0, 0.0));
    }

    /// ★ THE REBASE BOUND IS DERIVED, AND IT IS WIDER THAN THE GALAXY (owner ruling 2026-09-02 R8/5).
    #[test]
    fn the_cloud_is_rebased_only_beyond_a_derived_bound_wider_than_the_galaxy() {
        use vd_core::glam::DVec3;
        let bound = sky_rebase_bound_m();
        // One light year, a tenth of a pixel, single precision: about 7.9 × 10¹⁸ m.
        assert!(
            bound > 5.0e18,
            "the bound clears the galaxy's own radius by construction: {bound:e}"
        );
        assert!(!sky_rebase_due(DVec3::new(bound * 0.5, 0.0, 0.0)));
        assert!(sky_rebase_due(DVec3::new(bound * 1.5, 0.0, 0.0)));
    }

    /// ★ THE CLOUD IS FOUR VERTICES AND TWO TRIANGLES PER STAR, AND ITS SIZE IS A NUMBER (S11).
    #[test]
    fn the_star_cloud_is_one_upload_and_reuses_the_one_marker_convention() {
        use vd_core::glam::DVec3;
        let pt = |luma, class| StarPoint {
            pos_m: DVec3::new(1.0, 2.0, 3.0),
            class_code: class,
            luma_lsun: luma,
        };
        let cloud = StarCloud::build(&[pt(4.0, 0), pt(1.0, 6)]);

        assert_eq!(cloud.len(), 2);
        assert!(!cloud.is_empty());
        assert_eq!(cloud.positions.len(), 8, "four vertices per star");
        assert_eq!(cloud.indices.len(), 12, "two triangles per star");
        // The second star's quad indexes ITS OWN four vertices, never the first star's.
        assert_eq!(&cloud.indices[6..], &[4, 5, 6, 4, 6, 7]);
        // All four corners of one star share its centre; the shader moves them apart.
        assert!(cloud.positions[..4].iter().all(|p| *p == [1.0, 2.0, 3.0]));
        assert_eq!(cloud.corners[..4], StarCloud::CORNERS);

        // ★ THE RADIUS AND COLOUR COME FROM `marker_look`, not from a rule restated here. A second
        // spelling is how a starfield and its own pixel gate come to disagree.
        let bright = crate::realm_scene::marker_look(0, 4.0);
        assert_eq!(cloud.base_radius_m[0], bright.base_radius_m as f32);
        assert_eq!(cloud.colors[0], bright.color_rgba);
        // √L: four times the luminosity is twice the radius, so the two stars differ by exactly 2.
        let dim = crate::realm_scene::marker_look(6, 1.0);
        assert_eq!(cloud.base_radius_m[4], dim.base_radius_m as f32);
        assert_eq!(
            cloud.base_radius_m[0],
            2.0 * cloud.base_radius_m[4],
            "the square-root-of-luminosity convention, carried through unchanged"
        );
        // ...and the two classes really do differ, so the colour is per star and not a constant.
        assert_ne!(cloud.colors[0], cloud.colors[4]);

        // AN EMPTY SKY BUILDS AN EMPTY CLOUD, and does not panic on the index arithmetic.
        let none = StarCloud::build(&[]);
        assert!(none.is_empty());
        assert_eq!(none.len(), 0);
        assert!(none.indices.is_empty());
    }

    /// ★ THE CENSUS FITS, AND `u16` INDICES WOULD NOT (S11).
    ///
    /// Not a style note: 150,000 stars need 600,000 vertices, and a `u16` index tops out at 65,535.
    /// Choosing `u16` would silently draw the first 16,383 stars and nothing else.
    #[test]
    fn the_index_width_is_forced_by_the_census() {
        const CENSUS: usize = 150_000;
        let verts = CENSUS * 4;
        assert!(
            verts > usize::from(u16::MAX),
            "600,000 vertices overflow a u16 index — U32 is mandatory, not a preference"
        );
        // The buffer sizes this design was chosen on, stated so a later change has to face them.
        assert_eq!(
            verts * 24,
            14_400_000,
            "14.4 MB of vertices at 24 bytes each"
        );
        assert_eq!(CENSUS * 6 * 4, 3_600_000, "3.6 MB of u32 indices");
    }

    /// ★ CELLS AT OPPOSITE EDGES OF THE LATTICE DO NOT WRAP (S11).
    ///
    /// The subtraction runs in `i128` for exactly this case. In `i64` it would overflow, and a wrapped
    /// difference draws a star BEHIND the observer instead of in front — a picture that looks fine and
    /// is precisely inverted.
    #[test]
    fn two_stars_at_opposite_edges_of_the_lattice_do_not_wrap() {
        use vd_core::glam::I64Vec3;
        use vd_core::pose::RealmId;
        let far = i64::MAX / 2;
        let row = |n: u64, x: i64| vd_core::look::StarRow {
            realm: RealmId::System(n),
            cell: I64Vec3::new(x, 0, 0),
            class_code: 6,
            luma_lsun: 1.0,
        };
        let sky = SkyDraw {
            rows: vec![row(1, -far), row(2, far)],
            generation: 2,
        };
        // Referenced on System(2)'s own cell: System(1) is two `far`s back, exact in i128.
        let pts = sky.points_from(I64Vec3::new(far, 0, 0));
        assert_eq!(pts.len(), 2);
        let expect = -(2.0 * far as f64) * vd_core::pose::Tier::Galaxy.cell_edge_m();
        let behind = pts.iter().find(|p| p.pos_m.x != 0.0).expect("the far one");
        assert_eq!(behind.pos_m.x, expect);
        assert!(behind.pos_m.x < 0.0, "behind us, not in front");
    }

    /// ★ THE SKY RIDES THE RENDER SEAM BY POINTER, AND ONLY WHEN IT IS WHOLE (S11).
    ///
    /// A snapshot is cloned every step. At the target census the sky is 150,000 rows, so carrying it
    /// by value would deep-copy 7.0 MB twenty times a second for a galaxy that never moves. The clone
    /// must be a refcount bump, and this proves it is.
    #[test]
    fn the_sky_rides_the_seam_by_pointer_and_only_when_whole() {
        use std::sync::Arc;
        let bare = RenderSnapshot::new(
            DeliveredView::default(),
            RenderClock::new(ClientInterpTuning::DEFAULT),
            ClientPhase::Connecting,
        );
        // NO SKY until one is given. Half a galaxy is never carried here, so a renderer that sees
        // `None` knows there is nothing to draw rather than drawing holes.
        assert!(bare.sky().is_none());

        let sky = Arc::new(SkyDraw {
            rows: vec![vd_core::look::StarRow {
                realm: vd_core::pose::RealmId::System(1),
                cell: vd_core::glam::I64Vec3::new(1_313_684_865_644_610_304, 7, -3),
                class_code: 6,
                luma_lsun: 0.25,
            }],
            generation: 0xABCD,
        });
        let carried = bare.with_sky(Some(Arc::clone(&sky)));
        let held = carried.sky().expect("the sky is carried");
        assert_eq!(held.generation, 0xABCD);
        assert_eq!(held.rows.len(), 1);

        // ★ THE POINTER TEST. Cloning the snapshot must not copy the rows — it must share them.
        let before = Arc::strong_count(&sky);
        let cloned = carried.clone();
        assert_eq!(
            Arc::strong_count(&sky),
            before + 1,
            "a snapshot clone bumps the refcount; it does not copy the galaxy"
        );
        assert!(Arc::ptr_eq(cloned.sky().expect("carried"), &sky));

        // And it can be taken away again — the generation exchange may replace a sky.
        assert!(carried.with_sky(None).sky().is_none());

        // ★ THE ORIGIN RIDES THE SAME SEAM (S11), and it is what makes the sky drawable at all: the
        // renderer finds the observer's ANCHOR by looking up this realm in the catalogue. Carried
        // here rather than only read by the renderer, which is coverage-exempt — a datum only the
        // exempt tier reads is a datum nothing proves.
        let bare2 = RenderSnapshot::new(
            DeliveredView::default(),
            RenderClock::new(ClientInterpTuning::DEFAULT),
            ClientPhase::Connecting,
        );
        assert_eq!(bare2.origin(), None, "no standing realm stated yet");
        let placed = bare2.with_origin(Some(vd_core::pose::RealmId::System(7)));
        assert_eq!(placed.origin(), Some(vd_core::pose::RealmId::System(7)));
        assert!(
            placed.with_origin(None).origin().is_none(),
            "and it can be cleared — a session between realms states none"
        );
    }
}
