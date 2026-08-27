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
    }
}
