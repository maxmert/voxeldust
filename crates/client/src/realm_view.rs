//! The client's delivered REALM view (D-45(a) realm-unification FA-2c) — the streamed authoritative
//! placements for the MOVING realm boxes a shard parents (an orbiting planet / station / ship). The
//! `RealmId`-keyed twin of [`crate::view::DeliveredView`]: a realm is just another positioned thing the
//! pure-renderer client draws at its last SERVER-SHIPPED pose (NO prediction).
//!
//! Reuses [`EntityTrack`] VERBATIM (no-prediction interpolation, frame-change window-collapse,
//! freeze-on-loss) — so a moving realm box interpolates at the 20 Hz feed rate and FREEZES (never coasts)
//! on loss, exactly like an entity. Latest-wins by a SINGLE per-FEED monotone high-water: the realm
//! observer feed is SUB-AGNOSTIC (the gateway fans it to every subscriber with no held-sub check) and
//! carries one `frame_id` per frame, so there is no per-sub high-water and no foreign-sub arm — that is
//! the ONE place the realm gate genuinely differs from the entity gate (hence [`RealmVerdict`], not
//! `SnapshotVerdict`), while sharing the strictly-older [`is_stale`] scalar.
//!
//! EMPTY until a `RealmSnapshot` datagram arrives; at walk/static scale the server ships none.

use std::collections::BTreeMap;

use vd_core::pose::RealmId;
use vd_wire::channels::{RealmSnapshotDatagram, is_stale};

use crate::interp::{EntityTrack, RenderPose};

/// What the realm-feed gate decides for one [`RealmSnapshotDatagram`] — the twin of
/// `SnapshotVerdict` MINUS `DropForeignSub` (the realm feed is sub-agnostic, so there is no
/// held-sub arm to gate). The caller anchors the render clock only on [`RealmVerdict::Apply`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RealmVerdict {
    /// Fresh (or a sibling chunk of the current frame): folded into the per-realm tracks.
    Apply,
    /// A strictly-older `frame_id` — a late straggler, dropped + counted.
    DropStale,
}

/// The decoded, delivered realm-placement view.
#[derive(Clone, Debug, Default)]
pub struct RealmView {
    /// ONE track per realm (latest-wins) — the moving box's interpolated pose.
    placements: BTreeMap<RealmId, EntityTrack>,
    /// The per-FEED staleness high-water for the §6.3 gate (a SINGLE scalar — the realm feed is
    /// sub-agnostic and carries one `frame_id` per frame, unlike the entity path's per-sub map).
    high_water: Option<u64>,
    /// Realm frames ACCEPTED by the gate (the realm liveness signal a `wait-until` predicate polls —
    /// the realm twin of `snapshots_applied`; STAYS 0 at walk scale where no realm frame ships).
    frames_applied: u64,
    /// FAULT count: strictly-older realm frames dropped by the gate.
    stale_frames_dropped: u64,
    /// FAULT count: delivered realm poses that carried a non-finite (NaN/Inf) component and had to be
    /// sanitized at ingress (a corrupt/diverged shard is a real fault — COUNTED, never silently fixed).
    nonfinite_poses: u64,
}

impl RealmView {
    /// Fold one delivered realm frame in through the shared strictly-older gate ([`is_stale`]): on Apply
    /// advance the high-water and fold each `RealmSnap` into its per-`RealmId` track (sanitize-at-ingress,
    /// latest-wins); on a stale frame count the drop. An EQUAL `frame_id` (a sibling chunk of a
    /// partitioned frame) is NOT stale and applies — `EntityTrack::observe` updates in place without
    /// collapsing the interp window. Returns the verdict so the caller anchors the render clock on Apply.
    pub fn on_realm_snapshot(&mut self, snap: RealmSnapshotDatagram) -> RealmVerdict {
        if is_stale(self.high_water, snap.frame_id) {
            self.stale_frames_dropped += 1;
            return RealmVerdict::DropStale;
        }
        self.high_water = Some(snap.frame_id);
        // Count the ACCEPTED frame — strictly AFTER the is_stale gate, so a DropStale never inflates it.
        self.frames_applied += 1;
        for row in snap.realms {
            // Sanitize at the decode-ingress chokepoint (the DeliveredView discipline, view.rs): a
            // non-finite realm pose must never reach the render transforms; count the fault.
            let raw = row.pose;
            let pose = raw.sanitized();
            if pose != raw {
                self.nonfinite_poses += 1;
            }
            self.placements
                .entry(row.realm)
                .and_modify(|track| track.observe(pose))
                .or_insert_with(|| EntityTrack::new(pose));
        }
        RealmVerdict::Apply
    }

    /// The pose to render for `realm` at `cursor` (universe-tick f64 units) — `None` for a realm the feed
    /// never streamed (its box stays boot-static). The scene overlay reads this to move a box live.
    #[must_use]
    pub fn realm_pose(&self, realm: RealmId, cursor: f64) -> Option<RenderPose> {
        self.placements
            .get(&realm)
            .map(|track| track.sample(cursor))
    }

    /// The LATEST streamed pose for `realm` (no interpolation) — `None` for a realm the feed never
    /// streamed. The scene overlay reads this to move a boot box to its live server-shipped placement;
    /// the latest (not a cursor-interpolated blend) keeps the overlay cursor-free (FA-2c; render-side
    /// interpolation is an FA-5+ smoothness refinement).
    #[must_use]
    pub fn realm_latest(&self, realm: RealmId) -> Option<RenderPose> {
        self.placements
            .get(&realm)
            .map(|track| track.current_render_pose())
    }

    /// Whether the feed has streamed ANY realm placement yet — the byte-identity gate for the scene
    /// overlay (empty ⇒ the published scene is the boot scene by pointer-bump, walk-scale unchanged).
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.placements.is_empty()
    }

    /// Realm frames accepted by the gate — the realm liveness signal surfaced on DevState (the realm
    /// twin of `snapshots_applied`, what a `WaitUntil{RealmFramesApplied >= 1}` closed-loop e2e polls).
    #[must_use]
    pub fn frames_applied(&self) -> u64 {
        self.frames_applied
    }

    /// Strictly-older realm frames dropped (a fault surfaced on DevState).
    #[must_use]
    pub fn stale_frames_dropped(&self) -> u64 {
        self.stale_frames_dropped
    }

    /// Delivered realm poses sanitized at ingress (a fault surfaced on DevState).
    #[must_use]
    pub fn nonfinite_poses(&self) -> u64 {
        self.nonfinite_poses
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use vd_core::glam::DVec3;
    use vd_core::pose::{FrameRef, StampedPose};
    use vd_core::{TickId, UniverseTick};
    use vd_wire::channels::{RealmSnap, SubId};

    const SYS: FrameRef = FrameRef::SystemSpace { system_seed: 7 };

    fn pose(offset: DVec3, tick: u64) -> StampedPose {
        StampedPose::at_rest(SYS, offset, UniverseTick(tick))
    }

    fn frame(frame_id: u64, tick: u64, rows: Vec<(RealmId, StampedPose)>) -> RealmSnapshotDatagram {
        RealmSnapshotDatagram {
            sub: SubId(0),
            frame_id,
            source_tick: TickId(1),
            universe_tick: UniverseTick(tick),
            realms: rows
                .into_iter()
                .map(|(realm, pose)| RealmSnap { realm, pose })
                .collect(),
        }
    }

    #[test]
    fn apply_folds_a_track_per_realm_and_advances_the_high_water() {
        let mut v = RealmView::default();
        let verdict = v.on_realm_snapshot(frame(
            3,
            10,
            vec![
                (RealmId::Planet(1), pose(DVec3::new(1.0e9, 0.0, 0.0), 10)),
                (RealmId::Station(2), pose(DVec3::new(0.0, 2.0e8, 0.0), 10)),
            ],
        ));
        assert_eq!(verdict, RealmVerdict::Apply);
        assert_eq!(v.high_water, Some(3));
        assert!(v.realm_pose(RealmId::Planet(1), 10.0).is_some());
        assert!(v.realm_pose(RealmId::Station(2), 10.0).is_some());
        // A realm the feed never streamed has no pose (its box stays boot-static).
        assert_eq!(v.realm_pose(RealmId::Planet(99), 10.0), None);
        // realm_latest (the cursor-free overlay reader) mirrors: Some for a streamed realm, None else.
        assert_eq!(
            v.realm_latest(RealmId::Planet(1)).map(|p| p.pos),
            Some(DVec3::new(1.0e9, 0.0, 0.0)),
        );
        assert_eq!(v.realm_latest(RealmId::Planet(99)), None);
        assert!(!v.is_empty());
    }

    #[test]
    fn a_strictly_older_frame_is_dropped_and_counted_but_an_equal_sibling_applies() {
        let mut v = RealmView::default();
        assert_eq!(
            v.on_realm_snapshot(frame(5, 10, vec![(RealmId::Planet(1), pose(DVec3::X, 10))])),
            RealmVerdict::Apply,
        );
        // A STRICTLY-older frame is a late straggler — dropped + counted, the placement unchanged.
        assert_eq!(
            v.on_realm_snapshot(frame(
                4,
                10,
                vec![(RealmId::Planet(1), pose(DVec3::ZERO, 10))]
            )),
            RealmVerdict::DropStale,
        );
        assert_eq!(v.stale_frames_dropped, 1);
        assert_eq!(v.high_water, Some(5));
        // frames_applied counts ONLY the accepted frame, not the stale drop.
        assert_eq!(v.frames_applied(), 1);
        // An EQUAL frame_id (a sibling chunk of the partitioned frame 5) is NOT stale and applies —
        // a second realm lands from the same frame.
        assert_eq!(
            v.on_realm_snapshot(frame(
                5,
                10,
                vec![(RealmId::Station(2), pose(DVec3::Y, 10))]
            )),
            RealmVerdict::Apply,
        );
        assert!(v.realm_pose(RealmId::Station(2), 10.0).is_some());
        // The equal-sibling Apply advanced frames_applied to 2 (the DropStale between did not).
        assert_eq!(v.frames_applied(), 2);
    }

    #[test]
    fn a_realm_absent_from_a_fresh_frame_persists_its_track_no_vanish() {
        let mut v = RealmView::default();
        v.on_realm_snapshot(frame(1, 10, vec![(RealmId::Planet(1), pose(DVec3::X, 10))]));
        // Frame 2 names ONLY Station 2 — Planet 1 is absent (packet loss is indistinguishable from
        // "not in this chunk"), so its track PERSISTS (never evicted on datagram-absence).
        v.on_realm_snapshot(frame(
            2,
            20,
            vec![(RealmId::Station(2), pose(DVec3::Y, 20))],
        ));
        assert!(
            v.realm_pose(RealmId::Planet(1), 20.0).is_some(),
            "no-vanish"
        );
        assert!(v.realm_pose(RealmId::Station(2), 20.0).is_some());
    }

    #[test]
    fn a_nonfinite_pose_is_sanitized_and_counted_then_a_clean_pose_is_not() {
        let mut v = RealmView::default();
        v.on_realm_snapshot(frame(
            1,
            10,
            vec![(RealmId::Planet(1), pose(DVec3::new(f64::NAN, 0.0, 0.0), 10))],
        ));
        assert_eq!(
            v.nonfinite_poses, 1,
            "a non-finite realm pose is a counted fault"
        );
        // The rendered pose is finite (sanitized), never NaN.
        let rp = v.realm_pose(RealmId::Planet(1), 10.0).expect("a track");
        assert!(rp.pos.is_finite(), "the sanitized pose is finite");
        // A subsequent CLEAN pose does NOT bump the fault count (the `pose != raw` false arm).
        v.on_realm_snapshot(frame(2, 20, vec![(RealmId::Planet(1), pose(DVec3::X, 20))]));
        assert_eq!(v.nonfinite_poses, 1);
    }

    #[test]
    fn fault_counts_are_exposed_and_the_view_is_clone_debug_and_empty_by_default() {
        // A fresh view is empty; the accessor methods + the Clone/Debug derives are exercised.
        let fresh = RealmView::default();
        assert!(fresh.is_empty());
        assert_eq!(fresh.stale_frames_dropped(), 0);
        assert_eq!(fresh.nonfinite_poses(), 0);
        let mut v = RealmView::default();
        v.on_realm_snapshot(frame(
            2,
            10,
            vec![(
                RealmId::Planet(1),
                pose(DVec3::new(f64::INFINITY, 0.0, 0.0), 10),
            )],
        ));
        v.on_realm_snapshot(frame(1, 10, vec![(RealmId::Planet(1), pose(DVec3::X, 10))])); // stale
        assert_eq!(v.nonfinite_poses(), 1);
        assert_eq!(v.stale_frames_dropped(), 1);
        let cloned = v.clone();
        assert_eq!(cloned.nonfinite_poses(), 1);
        assert!(!format!("{v:?}").is_empty());
    }

    #[test]
    fn realm_pose_reflects_the_latest_streamed_pose_frozen_past_the_window() {
        let mut v = RealmView::default();
        v.on_realm_snapshot(frame(
            1,
            10,
            vec![(RealmId::Planet(1), pose(DVec3::new(1.0, 0.0, 0.0), 10))],
        ));
        v.on_realm_snapshot(frame(
            2,
            20,
            vec![(RealmId::Planet(1), pose(DVec3::new(2.0, 0.0, 0.0), 20))],
        ));
        // Past the freshest tick the box FREEZES at the latest delivered pose (no extrapolation).
        let rp = v.realm_pose(RealmId::Planet(1), 1_000.0).expect("a track");
        assert_eq!(rp.pos, DVec3::new(2.0, 0.0, 0.0));
        assert_eq!(rp.frame, SYS);
    }
}
