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
    /// A datagram from a PREVIOUS scene epoch — a straggler from the scene the client already
    /// swapped away from (§2.7): dropped, its rows counted (`stale_epoch_rows`).
    DropStaleEpoch,
    /// A datagram from the NEXT scene epoch that raced its own level on the reliable lane —
    /// HELD one beat (§2.7) and replayed by [`RealmView::swap_epoch`] when the level lands.
    HeldNextEpoch,
}

/// The decoded, delivered realm-placement view.
#[derive(Clone, Debug, Default)]
pub struct RealmView {
    /// ONE track per realm (latest-wins) — the moving box's interpolated pose.
    placements: BTreeMap<RealmId, EntityTrack>,
    /// THE ONE FEED COUNTER (proto_minor 18, §2.4): the composed feed has a SINGLE author — the
    /// session's own connection plane, stamping one monotone `frame_id` — so the old per-realm
    /// high-water map (which decoupled many independent shard counters) collapses to one scalar
    /// plus the epoch. `None` before the first applied datagram of the current epoch; cleared on
    /// every epoch swap (a new scene is a new stream).
    high_water: Option<u64>,
    /// The CURRENT scene epoch (§2.7): datagrams apply only at this epoch; an older one is a
    /// straggler from the swapped-away scene (dropped + counted), a newer one raced its level on
    /// the reliable lane (held one beat below).
    epoch: u64,
    /// The one-beat hold (§2.7): the newest early next-epoch datagram, replayed by
    /// [`RealmView::swap_epoch`] the moment its level lands. Newest-wins (latest-wins feed).
    held: Option<RealmSnapshotDatagram>,
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
    /// ALIVE through C1 by design (§4.5 Topic 4: a guard dies only in the commit that removes its
    /// cause — the still-open dual-sub overlap); it retires in C2 with the lanes that feed it.
    foreign_space_rows: u64,
    /// FAULT/diagnosis count: rows dropped because their datagram's `origin_epoch` predates the
    /// client's current scene (§2.6.6's `stale_epoch_rows` row). Expected briefly at a crossing.
    stale_epoch_rows: u64,
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
        // THE EPOCH GATE (§2.7): older epoch ⇒ a straggler from the swapped-away scene, dropped +
        // counted per row; newer epoch ⇒ it raced its own level on the reliable lane, HELD one
        // beat (newest wins) and replayed at the swap.
        if snap.origin_epoch < self.epoch {
            self.stale_epoch_rows += snap.realms.len() as u64;
            return RealmVerdict::DropStaleEpoch;
        }
        if snap.origin_epoch > self.epoch {
            let newer = self
                .held
                .as_ref()
                .is_none_or(|h| (snap.origin_epoch, snap.frame_id) >= (h.origin_epoch, h.frame_id));
            if newer {
                self.held = Some(snap);
            }
            return RealmVerdict::HeldNextEpoch;
        }
        // THE ONE FEED COUNTER (§2.4): a single author stamps one monotone `frame_id`, so one
        // strictly-older scalar gates the whole feed (an equal id is a sibling chunk — applied).
        if is_stale(self.high_water, snap.frame_id) {
            self.stale_frames_dropped += 1;
            return RealmVerdict::DropStale;
        }
        let mut any_applied = false;
        for row in snap.realms {
            if standing_in.is_some_and(|own| row.pose.frame != own) {
                self.foreign_space_rows += 1;
                continue;
            }
            // Sanitize at the decode-ingress chokepoint (the DeliveredView discipline, view.rs): a
            // non-finite realm pose must never reach the render transforms; count the fault.
            let raw = row.pose;
            let pose = raw.sanitized();
            if pose != raw {
                self.nonfinite_poses += 1;
            }
            self.high_water = Some(snap.frame_id);
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

    /// THE EPOCH SWAP (§2.7 — the flag day's replacement of the old `forget_space` INFERENCE):
    /// called when a composed LEVEL lands with a new `origin_epoch`. Every stored track is a
    /// position in the OLD origin's frame, meaningless in the new one: forget them all, clear the
    /// feed counter (a new scene is a new stream), adopt the epoch — and hand back the one-beat
    /// HELD datagram if it belongs to the scene just adopted, for the caller to replay through
    /// the normal ingest (the level itself carried every row's pose, so nothing is undrawn while
    /// the replay lands). A level RE-SENT at the current epoch swaps nothing — the scene it
    /// describes is the scene the tracks already animate.
    #[must_use]
    pub fn swap_epoch(&mut self, epoch: u64) -> Option<RealmSnapshotDatagram> {
        if epoch == self.epoch {
            return None;
        }
        self.placements.clear();
        self.high_water = None;
        self.epoch = epoch;
        match self.held.take() {
            Some(h) if h.origin_epoch == epoch => Some(h),
            // A held datagram for a STILL-newer epoch stays held; an older one is dead.
            Some(h) if h.origin_epoch > epoch => {
                self.held = Some(h);
                None
            }
            _ => None,
        }
    }

    /// The CURRENT scene epoch (§2.7) — the reliable-lane delta gate reads this.
    #[must_use]
    pub fn epoch(&self) -> u64 {
        self.epoch
    }

    /// Rows dropped for carrying a PREVIOUS scene epoch (a fault/diagnosis count, §2.6.6).
    #[must_use]
    pub fn stale_epoch_rows(&self) -> u64 {
        self.stale_epoch_rows
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
    /// Flatten a RenderPose to world metres (normalized lattice since the cell activation).
    fn rpw(p: &crate::interp::RenderPose) -> DVec3 {
        vd_core::pose::LatticePos::at(p.cell, p.pos)
            .delta_m(vd_core::pose::LatticePos::default(), p.tier)
    }

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
        frame_at_epoch(0, frame_id, tick, rows)
    }

    /// The epoch-carrying fixture (§2.7): the view boots at epoch 0, so `frame` stays applicable
    /// without a swap; the epoch tests drive this one directly.
    fn frame_at_epoch(
        origin_epoch: u64,
        frame_id: u64,
        tick: u64,
        rows: Vec<(RealmId, StampedPose)>,
    ) -> RealmSnapshotDatagram {
        RealmSnapshotDatagram {
            sub: SubId(0),
            frame_id,
            source_tick: TickId(1),
            universe_tick: UniverseTick(tick),
            origin_epoch,
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

    /// THE EPOCH SWAP forgets EVERYTHING (placements + the feed counter): every stored value is a
    /// position in the old origin's frame. The new scene then refills from zero — a fresh feed's
    /// LOWER frame_id must be admitted, which is why the counter clears with the tracks. A level
    /// RE-SENT at the current epoch swaps nothing; an old-epoch straggler is dropped + counted;
    /// an EARLY next-epoch datagram is held one beat and handed back at the swap (newest wins).
    #[test]
    fn the_epoch_swap_clears_tracks_holds_early_datagrams_and_drops_stale_epochs() {
        let mut v = RealmView::default();
        assert_eq!(v.epoch(), 0, "the view boots at epoch 0");
        v.on_realm_snapshot(
            None,
            frame(9, 10, vec![(RealmId::Planet(1), pose(DVec3::X, 10))]),
        );
        assert!(!v.is_empty());
        // A level RE-SENT at the current epoch swaps nothing (the tracks already animate it).
        assert_eq!(v.swap_epoch(0), None);
        assert!(!v.is_empty(), "a same-epoch level forgets nothing");
        // An EARLY epoch-1 datagram races its level: HELD, not applied, not dropped.
        assert_eq!(
            v.on_realm_snapshot(
                None,
                frame_at_epoch(1, 1, 12, vec![(RealmId::Planet(2), pose(DVec3::Y, 12))]),
            ),
            RealmVerdict::HeldNextEpoch,
        );
        // A newer early datagram REPLACES the held one (latest-wins feed)...
        assert_eq!(
            v.on_realm_snapshot(
                None,
                frame_at_epoch(1, 2, 13, vec![(RealmId::Planet(2), pose(DVec3::Z, 13))]),
            ),
            RealmVerdict::HeldNextEpoch,
        );
        // ...and an OLDER one does not (the is_none_or false arm).
        assert_eq!(
            v.on_realm_snapshot(
                None,
                frame_at_epoch(1, 1, 12, vec![(RealmId::Planet(2), pose(DVec3::Y, 12))]),
            ),
            RealmVerdict::HeldNextEpoch,
        );
        assert_eq!(v.realm_pose(RealmId::Planet(2), f64::INFINITY), None);
        // THE SWAP: tracks + counter forgotten; the held epoch-1 datagram is handed back.
        let held = v.swap_epoch(1).expect("the held datagram replays");
        assert_eq!(
            held.frame_id, 2,
            "the NEWEST early datagram was the one held"
        );
        assert!(v.is_empty(), "every stored placement is forgotten");
        assert_eq!(v.epoch(), 1);
        // The new scene's feed starts its own counter: a frame_id BELOW the forgotten one folds.
        let verdict = v.on_realm_snapshot(None, held);
        assert_eq!(
            verdict,
            RealmVerdict::Apply,
            "the forgotten counter no longer gates; the replayed datagram folds"
        );
        assert!(v.realm_pose(RealmId::Planet(2), 13.0).is_some());
        // An OLD-epoch straggler (the swapped-away scene) is dropped, its rows counted.
        assert_eq!(
            v.on_realm_snapshot(
                None,
                frame_at_epoch(0, 10, 14, vec![(RealmId::Planet(1), pose(DVec3::X, 14))]),
            ),
            RealmVerdict::DropStaleEpoch,
        );
        assert_eq!(v.stale_epoch_rows(), 1);
        assert_eq!(v.realm_pose(RealmId::Planet(1), f64::INFINITY), None);
    }

    /// The swap's held-datagram disposition arms the big test above cannot reach: a held datagram
    /// for a STILL-newer epoch stays held across an intermediate swap; one for a dead epoch drops.
    #[test]
    fn a_held_datagram_outlives_an_intermediate_swap_only_while_its_epoch_is_ahead() {
        let mut v = RealmView::default();
        // Held for epoch 2 while the view sits at 0.
        assert_eq!(
            v.on_realm_snapshot(
                None,
                frame_at_epoch(2, 1, 10, vec![(RealmId::Planet(1), pose(DVec3::X, 10))]),
            ),
            RealmVerdict::HeldNextEpoch,
        );
        // Swapping to epoch 1 keeps it held (its scene has not arrived yet)...
        assert_eq!(v.swap_epoch(1), None);
        // ...and swapping to epoch 2 hands it back.
        assert!(v.swap_epoch(2).is_some());
        // A held datagram for an epoch the view has swapped PAST is dead: hold one for 3, then
        // swap straight to 4 — nothing replays.
        assert_eq!(
            v.on_realm_snapshot(
                None,
                frame_at_epoch(3, 1, 11, vec![(RealmId::Planet(1), pose(DVec3::X, 11))]),
            ),
            RealmVerdict::HeldNextEpoch,
        );
        assert_eq!(v.swap_epoch(4), None, "a dead-epoch hold never replays");
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
        // ONE feed counter (§2.4 — the composed feed has a single author): the datagram's
        // frame_id is the whole feed's high-water.
        assert_eq!(v.high_water, Some(3));
        assert!(v.realm_pose(RealmId::Planet(1), 10.0).is_some());
        assert!(v.realm_pose(RealmId::Station(2), 10.0).is_some());
        // A realm the feed never streamed has no pose (its box stays boot-static).
        assert_eq!(v.realm_pose(RealmId::Planet(99), 10.0), None);
        // realm_latest (the cursor-free overlay reader) mirrors: Some for a streamed realm, None else.
        assert_eq!(
            v.realm_pose(RealmId::Planet(1), f64::INFINITY)
                .map(|p| rpw(&p)),
            Some(DVec3::new(1.0e9, 0.0, 0.0)),
        );
        assert_eq!(v.realm_pose(RealmId::Planet(99), f64::INFINITY), None);
        assert!(!v.is_empty());
    }

    #[test]
    fn one_author_one_counter_the_composed_feed_never_regresses() {
        // THE COLLAPSE (§2.4): the old per-realm high-water existed because many shards each ran
        // their OWN RealmFrameCounter and a feed-global scalar let one ratchet past another (the
        // FA-5 two-mover freeze). The composed feed has exactly ONE author — the session's own
        // connection plane, stamping one monotone frame_id across every realm it composes — so
        // one scalar is now CORRECT: a lower id after a higher one is a genuine straggler even
        // when it names a different realm, because the same author stamped both.
        let mut v = RealmView::default();
        assert_eq!(
            v.on_realm_snapshot(
                None,
                frame(500, 10, vec![(RealmId::Planet(7), pose(DVec3::X, 10))])
            ),
            RealmVerdict::Apply,
        );
        // A lower-id datagram — even for a DIFFERENT realm — is a straggler from the one author.
        assert_eq!(
            v.on_realm_snapshot(
                None,
                frame(30, 9, vec![(RealmId::Planet(8), pose(DVec3::Y, 9))])
            ),
            RealmVerdict::DropStale,
        );
        assert_eq!(v.stale_frames_dropped(), 1);
        assert_eq!(v.realm_pose(RealmId::Planet(8), 10.0), None);
        // The author's next fresh id carries BOTH realms forward — nothing freezes.
        assert_eq!(
            v.on_realm_snapshot(
                None,
                frame(
                    501,
                    20,
                    vec![
                        (RealmId::Planet(7), pose(DVec3::new(2.0, 0.0, 0.0), 20)),
                        (RealmId::Planet(8), pose(DVec3::new(0.0, 3.0, 0.0), 20)),
                    ]
                )
            ),
            RealmVerdict::Apply,
        );
        assert_eq!(
            v.realm_pose(RealmId::Planet(8), f64::INFINITY)
                .map(|p| rpw(&p)),
            Some(DVec3::new(0.0, 3.0, 0.0)),
            "the fresh composed tick carries every realm — not frozen",
        );
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
        assert_eq!(v.high_water, Some(5));
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
        assert_eq!(rpw(&rp), DVec3::new(2.0, 0.0, 0.0));
        assert_eq!(rp.frame, SYS);
    }
}
