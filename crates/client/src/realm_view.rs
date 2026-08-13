//! The client's delivered REALM view (D-45(a) realm-unification FA-2c) — the streamed authoritative
//! placements for the MOVING realm boxes a shard parents (an orbiting planet / station / ship). The
//! `RealmId`-keyed twin of [`crate::view::DeliveredView`]: a realm is just another positioned thing the
//! pure-renderer client draws at its last SERVER-SHIPPED pose (NO prediction).
//!
//! Reuses [`EntityTrack`] VERBATIM (no-prediction interpolation, frame-change window-collapse,
//! freeze-on-loss) — so a moving realm box interpolates at the 20 Hz feed rate and FREEZES (never coasts)
//! on loss, exactly like an entity. Latest-wins by a PER-`RealmId` monotone high-water (each realm is
//! authored by exactly ONE shard, so its `frame_id` stream is monotone; keying the gate per single-owner
//! realm decouples the independent per-shard `RealmFrameCounter`s so two co-subscribed mover shards never
//! freeze each other — see [`RealmView::high_water`]). The realm gate has no per-sub/foreign-sub arm (the
//! feed is sub-agnostic — the gateway fans it to every subscriber), hence [`RealmVerdict`] (not
//! `SnapshotVerdict`), while sharing the strictly-older [`is_stale`] scalar with the entity gate.
//!
//! EMPTY until a `RealmSnapshot` datagram arrives; at walk/static scale the server ships none.

use std::collections::BTreeMap;

use vd_core::UniverseTick;
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
    /// The per-REALM staleness high-water for the §6.3 gate — keyed by the single-owner `RealmId`,
    /// NOT one feed-global scalar. Each `RealmId` is authored by exactly ONE shard (its parent), and
    /// each shard runs its OWN monotone `RealmFrameCounter` from 0; a feed-global high-water would let a
    /// higher-counter shard's `frame_id` ratchet past a co-subscribed lower-counter shard's and FREEZE
    /// that shard's boxes forever (two mover shards in one client's AoI — the node-per-realm Forest case,
    /// e.g. a System 7 → Galaxy → System 8 warp). Keying per single-owner `RealmId` decouples the
    /// independent counters (the render is already per-`RealmId` latest-wins). A straggler across an
    /// authority HANDOFF (a realm re-homing between shards, two transient co-authors) still needs the
    /// `realm_fence` — an FA-6 concern (D-45 owed), not reachable until a realm reparents.
    high_water: BTreeMap<RealmId, u64>,
    /// Realm frames ACCEPTED by the gate (the realm liveness signal a `wait-until` predicate polls —
    /// the realm twin of `snapshots_applied`; STAYS 0 at walk scale where no realm frame ships).
    frames_applied: u64,
    /// FAULT count: strictly-older realm frames dropped by the gate.
    stale_frames_dropped: u64,
    /// FAULT count: delivered realm poses that carried a non-finite (NaN/Inf) component and had to be
    /// sanitized at ingress (a corrupt/diverged shard is a real fault — COUNTED, never silently fixed).
    nonfinite_poses: u64,
    /// Realm rows SKIPPED because their pose is stated in a space other than the one this client
    /// stands in (the ONE-SPACE rule, crossing-render slice §4x): during a hand-off the client
    /// legitimately holds two subs, and the OLD home keeps shipping rows in the OLD realm's frame
    /// until its sub closes — folding those beside the new home's rows draws two spaces into one
    /// picture (the measured jitter). Not a fault at a crossing; a non-zero steady-state value means
    /// a shard is shipping rows in a space its observer does not stand in (the server-side bug).
    foreign_space_rows: u64,
}

impl RealmView {
    /// Fold one delivered realm frame in through the shared strictly-older gate ([`is_stale`]), gated
    /// PER-`RealmId` (not one feed-global scalar — see [`RealmView::high_water`]): for each `RealmSnap`, a
    /// `frame_id` strictly older than THAT realm's high-water is a straggler (dropped + counted); a
    /// fresh-or-equal one advances the realm's high-water and folds the pose into its track
    /// (sanitize-at-ingress, latest-wins; an EQUAL `frame_id` — a sibling chunk of a partitioned frame —
    /// applies without collapsing the interp window). The datagram verdict is [`RealmVerdict::Apply`] iff
    /// ≥1 realm landed (so the caller anchors the render clock on real progress), else `DropStale`.
    ///
    /// THE ONE-SPACE RULE (crossing-render slice): `standing_in` is the frame the client's own avatar
    /// currently stands in (`DeliveredView::own_location_frame`). When known, a row whose POSE is
    /// stated in any other frame is skipped + counted (`foreign_space_rows`) — the shipping side
    /// restates every row into its observer's space (rehome_one_mechanism §4c), so a row in another
    /// space is by definition from the OLD home's still-draining feed (or a server bug), and folding
    /// it would draw two spaces into one picture. `None` (no avatar yet — a spectator/boot client)
    /// applies no filter, byte-identical to the pre-slice behaviour.
    pub fn on_realm_snapshot(
        &mut self,
        standing_in: Option<vd_core::pose::FrameRef>,
        snap: RealmSnapshotDatagram,
    ) -> RealmVerdict {
        let mut any_applied = false;
        for row in snap.realms {
            if standing_in.is_some_and(|own| row.pose.frame != own) {
                self.foreign_space_rows += 1;
                continue;
            }
            // Per-REALM staleness: a co-subscribed higher-counter shard must never ratchet a
            // lower-counter shard's realm past `frame_id` (the freeze bug a feed-global scalar caused).
            if is_stale(self.high_water.get(&row.realm).copied(), snap.frame_id) {
                self.stale_frames_dropped += 1;
                continue;
            }
            // Sanitize at the decode-ingress chokepoint (the DeliveredView discipline, view.rs): a
            // non-finite realm pose must never reach the render transforms; count the fault.
            let raw = row.pose;
            let pose = raw.sanitized();
            if pose != raw {
                self.nonfinite_poses += 1;
            }
            self.high_water.insert(row.realm, snap.frame_id);
            self.placements
                .entry(row.realm)
                .and_modify(|track| track.observe(pose))
                .or_insert_with(|| EntityTrack::new(pose));
            any_applied = true;
        }
        if any_applied {
            // Count the ACCEPTED frame — strictly AFTER the gate, so an all-stale drop never inflates it.
            self.frames_applied += 1;
            RealmVerdict::Apply
        } else {
            RealmVerdict::DropStale
        }
    }

    /// FORGET every stored placement and high-water (the ONE-SPACE rule's other half, crossing-render
    /// slice): called at the own-location frame flip — a crossing moved this client into a new realm,
    /// and every stored track is a position in the OLD realm's space, meaningless in the new one. The
    /// realm the client now stands in is the sharpest case: its own per-tick row NEVER ships again
    /// (the realm you occupy draws itself — its outline arrives on the scene lane at your origin), so
    /// without this its stale pre-crossing track would override that outline FOREVER (the measured
    /// "I landed on the planet and I am outside it"). The new space refills within one feed period;
    /// until then each box draws from its streamed scene outline (`RealmScene`), never from a stale
    /// track. High-waters clear with the tracks: they gate a space that no longer exists, and the
    /// one-space ingress filter keeps DIFFERENT-space stragglers out. HONEST RESIDUAL (review,
    /// minor): a fast A→B→A re-entry re-arrives in the SAME frame as before, so one reordered old-A
    /// straggler can pass the cleared gate and draw one stale placement for at most one feed period
    /// (the realm's single-author `frame_id` stream is monotone, so the next fresh row overwrites) —
    /// transient by construction, never a freeze or a permanent poisoning.
    pub fn forget_space(&mut self) {
        self.placements.clear();
        self.high_water.clear();
    }

    /// Realm rows skipped by the one-space ingress rule — see [`RealmView::on_realm_snapshot`].
    #[must_use]
    pub fn foreign_space_rows(&self) -> u64 {
        self.foreign_space_rows
    }

    /// The streamed pose for `realm` AT `cursor` (universe-tick f64 units) — interpolated on the same
    /// render clock, and by the same primitive, that every entity uses. `None` for a realm the feed
    /// never streamed (its box stays boot-static).
    ///
    /// SLICE 6 S4 — this existed and was CORRECT, but nothing called it: the scene overlay read
    /// `realm_latest` instead, taking whatever had most recently arrived with no cursor at all. That
    /// put the ground on a different time axis from the player standing on it, so their RELATIVE
    /// geometry moved every time either feed delivered. Reading BOTH at one cursor makes the drawn
    /// difference between them the same difference the server computed — which is the shake fix.
    #[must_use]
    pub fn realm_pose(&self, realm: RealmId, cursor: f64) -> Option<RenderPose> {
        self.placements
            .get(&realm)
            .map(|track| track.sample(cursor))
    }

    /// SLICE 6 S5 — how the streamed realm placements classify at `cursor`; see
    /// [`crate::view::DeliveredView::window_census`].
    #[must_use]
    pub fn window_census(&self, cursor: f64) -> (u32, u32, u32) {
        crate::interp::census(self.placements.values().map(|t| t.window_at(cursor)))
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

    /// The newest universe tick this realm's track has been fed — `None` for a realm the feed never
    /// streamed. SHAKE DIAGNOSIS: the realm feed's rows are authored by the realm's PARENT shard while
    /// an occupant's pose is composed on the shard it lives on, and each shard advances its own sense
    /// of universe time only when a clock sync ARRIVES. So the two numbers on one screen can drift
    /// apart, and their difference is the quantity that has to be measured before anything is built.
    #[must_use]
    pub fn realm_newest_tick(&self, realm: RealmId) -> Option<UniverseTick> {
        self.placements.get(&realm).map(|t| t.newest_tick())
    }

    /// The newest universe tick across EVERY streamed realm — the realm feed's freshness as one
    /// number, to be compared against the entity feed's. `None` before the first realm frame.
    #[must_use]
    pub fn newest_tick(&self) -> Option<UniverseTick> {
        self.placements.values().map(|t| t.newest_tick()).max()
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
                .map(|(realm, pose)| RealmSnap {
                    realm,
                    // The edge HEAD (proto_minor 8): the CHILD's own frame. `RealmView` folds a track
                    // per realm and does not read it, but a row without it is not a placement edge.
                    frame: vd_core::pose::frame_for_realm(realm, None)
                        .expect("a seeded realm resolves"),
                    pose,
                })
                .collect(),
        }
    }

    /// THE ONE-SPACE RULE (crossing-render slice): a row stated in a space other than the one the
    /// avatar stands in is skipped + counted, never folded — the old home's still-draining feed
    /// must not draw its space into the new one. Rows in the standing space fold as always, and a
    /// `None` standing space (spectator/boot) applies no filter.
    #[test]
    fn a_row_in_another_space_is_skipped_and_counted() {
        let mut v = RealmView::default();
        let planet_space = FrameRef::PlanetCentered { planet_seed: 9 };
        // Standing in the planet's space: a SYSTEM-space row (the old home's feed) is skipped...
        let verdict = v.on_realm_snapshot(
            Some(planet_space),
            frame(1, 10, vec![(RealmId::Planet(1), pose(DVec3::X, 10))]),
        );
        assert_eq!(verdict, RealmVerdict::DropStale, "nothing folded");
        assert_eq!(
            v.foreign_space_rows(),
            1,
            "the foreign-space row is counted"
        );
        assert_eq!(v.realm_pose(RealmId::Planet(1), 10.0), None);
        // ...while a row STATED in the standing space folds as always.
        let in_space = StampedPose::at_rest(planet_space, DVec3::Y, UniverseTick(11));
        let verdict = v.on_realm_snapshot(
            Some(planet_space),
            frame(2, 11, vec![(RealmId::Planet(2), in_space)]),
        );
        assert_eq!(verdict, RealmVerdict::Apply);
        assert!(v.realm_pose(RealmId::Planet(2), 11.0).is_some());
        assert_eq!(
            v.foreign_space_rows(),
            1,
            "an in-space row is never counted foreign"
        );
    }

    /// The space flip forgets EVERYTHING (placements + high-waters): every stored value is a
    /// position in the old space. The new space then refills from zero — a fresh feed's LOWER
    /// frame_id must be admitted, which is why the high-waters clear with the tracks.
    #[test]
    fn forget_space_clears_tracks_and_high_waters() {
        let mut v = RealmView::default();
        v.on_realm_snapshot(
            None,
            frame(9, 10, vec![(RealmId::Planet(1), pose(DVec3::X, 10))]),
        );
        assert!(!v.is_empty());
        v.forget_space();
        assert!(v.is_empty(), "every stored placement is forgotten");
        assert_eq!(v.realm_pose(RealmId::Planet(1), 10.0), None);
        // The new space's feed starts its own counter: a frame_id BELOW the forgotten high-water folds.
        let verdict = v.on_realm_snapshot(
            None,
            frame(1, 12, vec![(RealmId::Planet(1), pose(DVec3::Y, 12))]),
        );
        assert_eq!(
            verdict,
            RealmVerdict::Apply,
            "the forgotten high-water no longer gates"
        );
        assert!(v.realm_pose(RealmId::Planet(1), 12.0).is_some());
    }

    #[test]
    fn apply_folds_a_track_per_realm_and_advances_the_high_water() {
        let mut v = RealmView::default();
        let verdict = v.on_realm_snapshot(
            None,
            frame(
                3,
                10,
                vec![
                    (RealmId::Planet(1), pose(DVec3::new(1.0e9, 0.0, 0.0), 10)),
                    (RealmId::Station(2), pose(DVec3::new(0.0, 2.0e8, 0.0), 10)),
                ],
            ),
        );
        assert_eq!(verdict, RealmVerdict::Apply);
        // Per-realm high-water: BOTH streamed realms are stamped at the datagram's frame_id.
        assert_eq!(v.high_water.get(&RealmId::Planet(1)), Some(&3));
        assert_eq!(v.high_water.get(&RealmId::Station(2)), Some(&3));
        assert!(v.realm_pose(RealmId::Planet(1), 10.0).is_some());
        assert!(v.realm_pose(RealmId::Station(2), 10.0).is_some());
        // A realm the feed never streamed has no pose (its box stays boot-static).
        assert_eq!(v.realm_pose(RealmId::Planet(99), 10.0), None);
        // realm_latest (the cursor-free overlay reader) mirrors: Some for a streamed realm, None else.
        assert_eq!(
            v.realm_pose(RealmId::Planet(1), f64::INFINITY)
                .map(|p| p.pos),
            Some(DVec3::new(1.0e9, 0.0, 0.0)),
        );
        assert_eq!(v.realm_pose(RealmId::Planet(99), f64::INFINITY), None);
        assert!(!v.is_empty());
    }

    #[test]
    fn two_mover_shards_with_divergent_frame_ids_both_stay_live_no_cross_shard_freeze() {
        // The FA-5 multi-emitter case (the holistic /goal audit HIGH, wf_c9444997): two shards each
        // author their OWN disjoint realms with INDEPENDENT RealmFrameCounters — shard A far ahead
        // (frame_id 500), shard B fresh (frame_id 30). A feed-GLOBAL high-water would let A(500) ratchet
        // past B and DROP every B frame as stale forever (B's boxes FREEZE — the node-per-realm Forest
        // System 7 -> Galaxy -> System 8 warp bug). Per-RealmId keying decouples them: BOTH stay live.
        let mut v = RealmView::default();
        // Shard A (System 7): its planet at a high counter.
        assert_eq!(
            v.on_realm_snapshot(
                None,
                frame(500, 10, vec![(RealmId::Planet(7), pose(DVec3::X, 10))])
            ),
            RealmVerdict::Apply,
        );
        // Shard B (System 8): its planet at a LOW counter — must NOT be rejected as "stale" vs A's 500.
        assert_eq!(
            v.on_realm_snapshot(
                None,
                frame(30, 10, vec![(RealmId::Planet(8), pose(DVec3::Y, 10))])
            ),
            RealmVerdict::Apply,
            "shard B's low-counter frame must apply — no cross-shard high-water conflation",
        );
        assert!(v.realm_pose(RealmId::Planet(7), 10.0).is_some());
        assert!(v.realm_pose(RealmId::Planet(8), 10.0).is_some());
        // B keeps advancing independently of A's counter — it is not frozen.
        assert_eq!(
            v.on_realm_snapshot(
                None,
                frame(
                    31,
                    20,
                    vec![(RealmId::Planet(8), pose(DVec3::new(0.0, 3.0, 0.0), 20))]
                )
            ),
            RealmVerdict::Apply,
        );
        assert_eq!(
            v.realm_pose(RealmId::Planet(8), f64::INFINITY)
                .map(|p| p.pos),
            Some(DVec3::new(0.0, 3.0, 0.0)),
            "B moved to its frame-31 pose — not frozen",
        );
        // A per-realm stale straggler (B at 30 after B advanced to 31) is STILL dropped.
        assert_eq!(
            v.on_realm_snapshot(
                None,
                frame(30, 10, vec![(RealmId::Planet(8), pose(DVec3::ZERO, 10))])
            ),
            RealmVerdict::DropStale,
        );
        assert_eq!(v.stale_frames_dropped(), 1);
    }

    #[test]
    fn a_strictly_older_frame_is_dropped_and_counted_but_an_equal_sibling_applies() {
        let mut v = RealmView::default();
        assert_eq!(
            v.on_realm_snapshot(
                None,
                frame(5, 10, vec![(RealmId::Planet(1), pose(DVec3::X, 10))])
            ),
            RealmVerdict::Apply,
        );
        // A STRICTLY-older frame is a late straggler — dropped + counted, the placement unchanged.
        assert_eq!(
            v.on_realm_snapshot(
                None,
                frame(4, 10, vec![(RealmId::Planet(1), pose(DVec3::ZERO, 10))])
            ),
            RealmVerdict::DropStale,
        );
        assert_eq!(v.stale_frames_dropped, 1);
        assert_eq!(v.high_water.get(&RealmId::Planet(1)), Some(&5));
        // frames_applied counts ONLY the accepted frame, not the stale drop.
        assert_eq!(v.frames_applied(), 1);
        // An EQUAL frame_id (a sibling chunk of the partitioned frame 5) is NOT stale and applies —
        // a second realm lands from the same frame.
        assert_eq!(
            v.on_realm_snapshot(
                None,
                frame(5, 10, vec![(RealmId::Station(2), pose(DVec3::Y, 10))])
            ),
            RealmVerdict::Apply,
        );
        assert!(v.realm_pose(RealmId::Station(2), 10.0).is_some());
        // The equal-sibling Apply advanced frames_applied to 2 (the DropStale between did not).
        assert_eq!(v.frames_applied(), 2);
    }

    #[test]
    fn a_realm_absent_from_a_fresh_frame_persists_its_track_no_vanish() {
        let mut v = RealmView::default();
        v.on_realm_snapshot(
            None,
            frame(1, 10, vec![(RealmId::Planet(1), pose(DVec3::X, 10))]),
        );
        // Frame 2 names ONLY Station 2 — Planet 1 is absent (packet loss is indistinguishable from
        // "not in this chunk"), so its track PERSISTS (never evicted on datagram-absence).
        v.on_realm_snapshot(
            None,
            frame(2, 20, vec![(RealmId::Station(2), pose(DVec3::Y, 20))]),
        );
        assert!(
            v.realm_pose(RealmId::Planet(1), 20.0).is_some(),
            "no-vanish"
        );
        assert!(v.realm_pose(RealmId::Station(2), 20.0).is_some());
    }

    #[test]
    fn a_nonfinite_pose_is_sanitized_and_counted_then_a_clean_pose_is_not() {
        let mut v = RealmView::default();
        v.on_realm_snapshot(
            None,
            frame(
                1,
                10,
                vec![(RealmId::Planet(1), pose(DVec3::new(f64::NAN, 0.0, 0.0), 10))],
            ),
        );
        assert_eq!(
            v.nonfinite_poses, 1,
            "a non-finite realm pose is a counted fault"
        );
        // The rendered pose is finite (sanitized), never NaN.
        let rp = v.realm_pose(RealmId::Planet(1), 10.0).expect("a track");
        assert!(rp.pos.is_finite(), "the sanitized pose is finite");
        // A subsequent CLEAN pose does NOT bump the fault count (the `pose != raw` false arm).
        v.on_realm_snapshot(
            None,
            frame(2, 20, vec![(RealmId::Planet(1), pose(DVec3::X, 20))]),
        );
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
        v.on_realm_snapshot(
            None,
            frame(
                2,
                10,
                vec![(
                    RealmId::Planet(1),
                    pose(DVec3::new(f64::INFINITY, 0.0, 0.0), 10),
                )],
            ),
        );
        v.on_realm_snapshot(
            None,
            frame(1, 10, vec![(RealmId::Planet(1), pose(DVec3::X, 10))]),
        ); // stale
        assert_eq!(v.nonfinite_poses(), 1);
        assert_eq!(v.stale_frames_dropped(), 1);
        let cloned = v.clone();
        assert_eq!(cloned.nonfinite_poses(), 1);
        assert!(!format!("{v:?}").is_empty());
    }

    #[test]
    fn per_realm_freshness_is_readable_and_distinct_from_the_feeds_overall_freshness() {
        // THE SHAKE DIAGNOSIS PAIR, and the reason both exist. A realm's rows are authored by its PARENT
        // shard, while an occupant's pose is composed on the shard it lives on — and each shard advances
        // its own sense of universe time only when a clock sync arrives. So two numbers on one screen can
        // legitimately disagree, and telling ONE realm's freshness apart from the feed's newest is what
        // makes that difference measurable instead of a guess.
        let mut v = RealmView::default();
        assert_eq!(
            v.newest_tick(),
            None,
            "before the first frame the feed has no freshness at all"
        );
        assert_eq!(
            v.realm_newest_tick(RealmId::Planet(1)),
            None,
            "and neither does a realm it has never streamed"
        );

        v.on_realm_snapshot(
            None,
            frame(1, 10, vec![(RealmId::Planet(1), pose(DVec3::X, 10))]),
        );
        v.on_realm_snapshot(
            None,
            frame(2, 40, vec![(RealmId::Planet(2), pose(DVec3::Y, 40))]),
        );

        // Each realm reports ITS OWN newest tick — the older one is not dragged forward by the newer.
        assert_eq!(
            v.realm_newest_tick(RealmId::Planet(1)),
            Some(UniverseTick(10))
        );
        assert_eq!(
            v.realm_newest_tick(RealmId::Planet(2)),
            Some(UniverseTick(40))
        );
        // …while the feed's overall freshness is the newest across all of them.
        assert_eq!(v.newest_tick(), Some(UniverseTick(40)));
        // A realm the feed never streamed stays None even once the feed is live — absence of a track is
        // not the same as a stale one, and conflating them would hide a realm that never arrived.
        assert_eq!(v.realm_newest_tick(RealmId::Planet(9)), None);
    }

    #[test]
    fn realm_pose_reflects_the_latest_streamed_pose_frozen_past_the_window() {
        let mut v = RealmView::default();
        v.on_realm_snapshot(
            None,
            frame(
                1,
                10,
                vec![(RealmId::Planet(1), pose(DVec3::new(1.0, 0.0, 0.0), 10))],
            ),
        );
        v.on_realm_snapshot(
            None,
            frame(
                2,
                20,
                vec![(RealmId::Planet(1), pose(DVec3::new(2.0, 0.0, 0.0), 20))],
            ),
        );
        // Past the freshest tick the box FREEZES at the latest delivered pose (no extrapolation).
        let rp = v.realm_pose(RealmId::Planet(1), 1_000.0).expect("a track");
        assert_eq!(rp.pos, DVec3::new(2.0, 0.0, 0.0));
        assert_eq!(rp.frame, SYS);
    }
}
