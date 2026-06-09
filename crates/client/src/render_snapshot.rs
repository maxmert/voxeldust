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

use vd_core::EntityId;
use vd_core::pose::FrameRef;
use vd_wire::channels::SubId;

use crate::interp::RenderPose;
use crate::net::ClientPhase;
use crate::render_clock::RenderClock;
use crate::view::DeliveredView;

/// An immutable snapshot of the delivered world + the render clock at one core step.
/// Cheap to clone (the view is `BTreeMap`s of `Copy` tracks); published every step and
/// read wait-free by the renderer.
#[derive(Clone, Debug)]
pub struct RenderSnapshot {
    view: DeliveredView,
    clock: RenderClock,
    phase: ClientPhase,
}

impl RenderSnapshot {
    #[must_use]
    pub fn new(view: DeliveredView, clock: RenderClock, phase: ClientPhase) -> RenderSnapshot {
        RenderSnapshot { view, clock, phase }
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
    use super::*;
    use glam::DVec3;
    use vd_core::entity_kind::EntityKind;
    use vd_core::pose::{FrameRef, StampedPose};
    use vd_core::{TickId, UniverseTick};
    use vd_wire::channels::{EntitySnap, SnapshotDatagram};

    use crate::tuning::ClientInterpTuning;

    fn ent(seq: u64) -> EntityId {
        EntityId::pack(EntityKind::Player, 1, seq, seq as u32)
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
    }

    #[test]
    fn an_unanchored_clock_suppresses_even_a_populated_view() {
        // The cursor==None gate is the ANCHOR, not an empty view: a view WITH a delivered
        // track but a clock that never observed a tick renders nothing — while own_entity
        // / location (cursor-independent) still read.
        let mut view = DeliveredView::default();
        view.on_snapshot(Some(SubId(0)), snap(1, 10, vec![(ent(1), 0.0)]));
        view.set_authority(ent(1), SubId(0));
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
    fn it_interpolates_the_delivered_view_at_the_display_cursor() {
        // Two snapshots a tick-window apart; the renderer samples at ITS display cursor,
        // not the core's — so motion is smooth at any refresh rate.
        let mut view = DeliveredView::default();
        view.on_snapshot(Some(SubId(0)), snap(1, 10, vec![(ent(1), 0.0)]));
        view.on_snapshot(Some(SubId(0)), snap(2, 12, vec![(ent(1), 10.0)]));
        view.set_authority(ent(1), SubId(0));

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
        assert_eq!(early[0].2.pos, DVec3::new(0.0, 0.0, 0.0));

        // 0.13 s later the display cursor advances into the window (9.6 + 0.13*20 =
        // 12.2 → clamps to current x=10): a DIFFERENT sample from the same snapshot,
        // proving display-rate interpolation off one published frame.
        let later = snap.rendered(100.13);
        assert_eq!(later[0].2.pos, DVec3::new(10.0, 0.0, 0.0));
    }
}
