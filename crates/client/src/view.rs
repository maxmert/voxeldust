//! The client's DELIVERED world view — decoded snapshots only, wire truth (the
//! same honesty rule as the `WireMonitor`: the client renders what it was actually
//! sent, never internal hope).
//!
//! Poses are kept as per-`(SubId, EntityId)` [`EntityTrack`]s so the shape survives
//! P2 cross-shard overlap (an entity visible through two subscriptions). The
//! composited render pass picks ONE authoritative sub per entity and suppresses the
//! duplicate copies; in P1.5's single-subscription world that is simply the one
//! sub, but the seam is already correct for P2.
//!
//! ## Track eviction — sound signals only (NEVER datagram-absence)
//! A track is dropped ONLY on a RELIABLE signal, never on absence from a datagram:
//! snapshots ride UNRELIABLE datagrams, so an entity merely ABSENT from one frame is
//! indistinguishable from packet loss — evicting on absence would delete live entities.
//! Two reliable signals bound the view to its live set:
//!   * per-SUB teardown — [`DeliveredView::drop_sub`], driven by the gateway's reliable
//!     `SubscriptionClosing`; this is wired NOW (it keeps memory + per-step clone cost
//!     proportional to the LIVE subscription set as AoI churns on the end-goal cross-shard
//!     path, not to every sub ever seen).
//!   * per-ENTITY teardown within a still-open sub — `EventMsg::EntityRemoved`; this is the
//!     remaining P2 piece, gated on routing the reliable EventMsg class to the client
//!     (there is no Bulk/Event arm on the client inbound seam yet). Until then a still-open
//!     sub's tracks persist, which is correct (lossy datagrams cannot prove departure).
//!
//! (Raised by the Slice-0–T3 + P1.5-foundation audits; drop_sub closes the per-sub half.)

use std::collections::BTreeMap;

use glam::DVec3;
use vd_core::EntityId;
use vd_core::pose::FrameRef;
use vd_wire::channels::{SnapshotDatagram, SnapshotVerdict, SubId, classify_snapshot};

use crate::interp::{EntityTrack, RenderPose};

/// The decoded, delivered world view.
#[derive(Clone, Debug, Default)]
pub struct DeliveredView {
    tracks: BTreeMap<(SubId, EntityId), EntityTrack>,
    high_water: BTreeMap<SubId, u64>,
    /// Explicit render authority per entity (from `AuthorityChanged`); overrides the
    /// default lowest-sub pick when an entity is visible through several subs (P2).
    authoritative_sub: BTreeMap<EntityId, SubId>,
    own_entity: Option<EntityId>,
    stale_frames_dropped: u64,
    /// FAULT count: delivered poses that carried a non-finite (NaN/Inf) component and had
    /// to be sanitized at ingress. A corrupt/diverged sender is a real fault, so it is
    /// COUNTED (not silently fixed) — the codebase's "never silent" discipline.
    nonfinite_poses: u64,
}

impl DeliveredView {
    /// Fold one delivered snapshot in, through the shared §6.3 gate
    /// ([`classify_snapshot`]): apply the entities (feeding their per-sub tracks) or
    /// count the drop. `held_subs` is the SET of subscriptions the client currently holds
    /// (Track R / 1d.2d): during a cross-shard transfer overlap the client holds BOTH the
    /// source and dest subs, so a frame on either is admitted — the composited render pass
    /// ([`DeliveredView::render`]) still picks ONE authoritative sub per entity (via
    /// `AuthorityChanged`/`set_authority`), so the avatar is rendered exactly once.
    /// Returns the verdict so the caller anchors the render clock only on `Apply`.
    pub fn on_snapshot(
        &mut self,
        held_subs: &std::collections::BTreeSet<SubId>,
        snap: SnapshotDatagram,
    ) -> SnapshotVerdict {
        let high_water = self.high_water.get(&snap.sub).copied();
        let verdict = classify_snapshot(held_subs, high_water, snap.sub, snap.frame_id);
        match verdict {
            SnapshotVerdict::Apply => {
                self.high_water.insert(snap.sub, snap.frame_id);
                for entity in snap.entities {
                    // Sanitize at the decode-ingress chokepoint: a corrupt/diverged
                    // sender could ship a non-finite pose, which must never reach the
                    // render transforms (it would poison the scene). Guarding here keeps
                    // EVERY downstream consumer (interp, render snapshot, DevState) finite.
                    // A sanitized pose is a real FAULT — count it (never silently fix).
                    let raw = entity.pose;
                    let pose = raw.sanitized();
                    if pose != raw {
                        self.nonfinite_poses += 1;
                    }
                    self.tracks
                        .entry((snap.sub, entity.entity))
                        .and_modify(|track| track.observe(pose))
                        .or_insert_with(|| EntityTrack::new(pose));
                }
            }
            SnapshotVerdict::DropForeignSub | SnapshotVerdict::DropStale => {
                self.stale_frames_dropped += 1;
            }
        }
        verdict
    }

    /// Record render authority for an entity (from `AuthorityChanged`): which sub
    /// renders it, and that it is THIS client's own entity.
    pub fn set_authority(&mut self, entity: EntityId, sub: SubId) {
        self.authoritative_sub.insert(entity, sub);
        self.own_entity = Some(entity);
    }

    /// Drop every track (+ high-water + authority pointer) for a subscription the gateway
    /// has RELIABLY closed (`SubscriptionClosing`). The SOUND eviction signal — reliable +
    /// per-sub, unlike datagram-absence — so the view (and the per-step clone of it) stays
    /// bounded to the live subscription set as AoI churns. An entity whose authority pointed
    /// at the closed sub loses that override and falls back to its lowest remaining sub (or
    /// is no longer rendered if it had only this one). `own_entity` is retained: it is an id,
    /// and `own_location_frame` already reports `None` once the own track is gone.
    pub fn drop_sub(&mut self, sub: SubId) {
        self.tracks.retain(|(s, _), _| *s != sub);
        self.high_water.remove(&sub);
        self.authoritative_sub.retain(|_, s| *s != sub);
    }

    /// The composited render poses at `cursor`: each entity rendered EXACTLY ONCE,
    /// from its authoritative sub (explicit `AuthorityChanged`, else the lowest sub
    /// it appears in — unambiguous in P1.5's single sub, disambiguated by authority
    /// in P2). Cross-sub duplicates are suppressed.
    #[must_use]
    pub fn render(&self, cursor: f64) -> BTreeMap<EntityId, RenderPose> {
        self.rendered(cursor)
            .into_iter()
            .map(|(entity, _sub, pose)| (entity, pose))
            .collect()
    }

    /// Like [`DeliveredView::render`] but also reports each entity's authoritative
    /// sub — the dev-control diagnosis surface (which sub a rendered entity came
    /// from; in P1.5 always the one sub, disambiguated by authority in P2).
    #[must_use]
    pub fn rendered(&self, cursor: f64) -> Vec<(EntityId, SubId, RenderPose)> {
        self.chosen_subs()
            .into_iter()
            .filter_map(|(entity, sub)| {
                self.tracks
                    .get(&(sub, entity))
                    .map(|track| (entity, sub, track.sample(cursor)))
            })
            .collect()
    }

    /// Each entity's authoritative sub: explicit `AuthorityChanged`, else the lowest
    /// sub it appears on. Suppresses cross-sub duplicates (each entity once).
    fn chosen_subs(&self) -> BTreeMap<EntityId, SubId> {
        let mut chosen: BTreeMap<EntityId, SubId> = BTreeMap::new();
        for (sub, entity) in self.tracks.keys() {
            match self.authoritative_sub.get(entity) {
                Some(auth) => {
                    chosen.insert(*entity, *auth);
                }
                None => {
                    chosen.entry(*entity).or_insert(*sub);
                }
            }
        }
        chosen
    }

    #[must_use]
    pub fn own_entity(&self) -> Option<EntityId> {
        self.own_entity
    }

    /// The subs that currently hold a delivered TRACK for `entity` (ascending). A pure read-only
    /// diagnosis surface (it touches NO render/eviction state) — the dev-control + transfer-gate
    /// anti-vacuity probe: during a cross-shard overlap the avatar has a track on BOTH the source
    /// AND the dest sub here, even though [`DeliveredView::rendered`] composites it to ONE. So a gate
    /// can prove the two-holder window was REAL (both tracks live) before asserting it still rendered
    /// exactly once — otherwise "rendered once" is vacuously true because no overlap ever occurred.
    #[must_use]
    pub fn subs_holding(&self, entity: EntityId) -> Vec<SubId> {
        self.tracks
            .keys()
            .filter(|(_, e)| *e == entity)
            .map(|(sub, _)| *sub)
            .collect()
    }

    /// The frame the OWN entity is currently in — its authoritative track's leading
    /// edge — i.e. the player's LOCATION (realm), the basis for the player-stats HUD and
    /// the `vdctl` location readout. `None` until the own entity is known AND has a
    /// delivered track on its authoritative sub. Cursor-free (frame does not interpolate)
    /// and authority-disambiguated by the SAME `chosen_subs` the render path uses, so the
    /// location can never disagree with the rendered pose. It is the delivered FrameRef,
    /// so it is fence-validated and changes only on a real cross-realm move.
    #[must_use]
    pub fn own_location_frame(&self) -> Option<FrameRef> {
        let own = self.own_entity?;
        let sub = self.chosen_subs().get(&own).copied()?;
        self.tracks
            .get(&(sub, own))
            .map(|track| track.current_frame())
    }

    #[must_use]
    pub fn stale_frames_dropped(&self) -> u64 {
        self.stale_frames_dropped
    }

    /// FAULT count of delivered poses that were non-finite and had to be sanitized.
    #[must_use]
    pub fn nonfinite_poses(&self) -> u64 {
        self.nonfinite_poses
    }

    /// The frame-evaluation seam: map a rendered pose's FRAME-LOCAL position into world
    /// space, composing through the view as the frame requires. Planet/System/Galaxy frames
    /// are world-origin in P1.5 (identity). A `ShipLocal` interior pose composes through its
    /// hull entity's LIVE pose — `hull.pos + hull.orient * interior.pos` — sampled at the
    /// SAME `cursor` so interior and hull agree in time ("walk inside a flying ship", P8);
    /// until a hull is delivered (no ships pre-P8) the lookup is `None` and the interior
    /// renders at its frame origin. This is THE single chokepoint for that composition (and
    /// the P10 galaxy ly-cell offset). Never panics; always finite (poses sanitized at ingress).
    ///
    /// ⚠️ TIME-coherent only, NOT version-matched (DEFERRED D-35). When the ship rides its own
    /// shard (P8), hull + interior arrive on DIFFERENT lossy subs; §8.1 (`transfer_protocol.md`,
    /// `sealed_shards.md:132`) requires holding the newer stream until both match `(source_tick,
    /// version)` or the passenger visibly JUMPS relative to the hull under datagram loss. That
    /// needs a NEW `parent_version` wire field on `EntitySnap`/`SnapshotDatagram` + a hold-buffer
    /// here — a snapshot-decode RESHAPE, NOT free. The chokepoint shape (one level) is right; only
    /// the version gate is owed at P8.
    #[must_use]
    pub fn world_pos(&self, pose: &RenderPose, cursor: f64) -> DVec3 {
        match pose.frame {
            FrameRef::PlanetCentered { .. }
            | FrameRef::SystemSpace { .. }
            | FrameRef::GalaxySpace => pose.pos,
            FrameRef::ShipLocal { ship } => self
                .hull_pose(ship, cursor)
                .map_or(pose.pos, |hull| hull.pos + hull.orient * pose.pos),
        }
    }

    /// The hull entity's rendered pose at `cursor` — the basis a `ShipLocal` interior pose
    /// composes against. The hull is just-another-delivered-entity (the id named by the
    /// frame), resolved on its authoritative sub via the SAME `chosen_subs` the render path
    /// uses. `None` until the hull is delivered there (no ships pre-P8, or authority points
    /// at a sub we hold no track for) ⇒ the interior renders at its frame origin.
    fn hull_pose(&self, hull: EntityId, cursor: f64) -> Option<RenderPose> {
        let sub = self.chosen_subs().get(&hull).copied()?;
        self.tracks
            .get(&(sub, hull))
            .map(|track| track.sample(cursor))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::BTreeSet;
    use vd_core::entity_kind::EntityKind;
    use vd_core::pose::{FrameRef, StampedPose};
    use vd_core::{TickId, UniverseTick};
    use vd_wire::channels::EntitySnap;

    /// A single-element held-sub set (the common single-subscription case).
    fn s(sub: SubId) -> BTreeSet<SubId> {
        BTreeSet::from([sub])
    }

    fn ent(seq: u64) -> EntityId {
        EntityId::pack(EntityKind::Player, 1, seq, seq as u32)
    }

    fn snap(
        sub: SubId,
        frame_id: u64,
        tick: u64,
        entities: Vec<(EntityId, f64)>,
    ) -> SnapshotDatagram {
        SnapshotDatagram {
            sub,
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
    fn applied_snapshots_build_tracks_and_render_once_per_entity() {
        let mut view = DeliveredView::default();
        let held = s(SubId(0));
        view.on_snapshot(
            &held,
            snap(SubId(0), 1, 10, vec![(ent(1), 0.0), (ent(2), 100.0)]),
        );
        view.on_snapshot(
            &held,
            snap(SubId(0), 2, 12, vec![(ent(1), 10.0), (ent(2), 110.0)]),
        );
        let rendered = view.render(11.0); // tick-11 cursor → window midpoint
        assert_eq!(rendered.len(), 2);
        assert_eq!(rendered[&ent(1)].pos, DVec3::new(5.0, 0.0, 0.0));
        assert_eq!(rendered[&ent(2)].pos, DVec3::new(105.0, 0.0, 0.0));
        assert_eq!(view.stale_frames_dropped(), 0);
        // world_pos is the identity for P1.5 static (system) frames.
        assert_eq!(
            view.world_pos(&rendered[&ent(1)], 11.0),
            DVec3::new(5.0, 0.0, 0.0)
        );
    }

    #[test]
    fn drop_sub_evicts_only_that_subs_tracks_and_keeps_the_view_bounded() {
        // Two entities through two subs; closing sub 0 evicts ONLY its tracks + authority,
        // leaving sub 1 intact — the reliable per-sub eviction that bounds the view.
        let mut view = DeliveredView::default();
        view.on_snapshot(&s(SubId(0)), snap(SubId(0), 1, 10, vec![(ent(1), 1.0)]));
        view.on_snapshot(&s(SubId(1)), snap(SubId(1), 1, 10, vec![(ent(2), 2.0)]));
        view.set_authority(ent(1), SubId(0));
        assert_eq!(view.render(10.0).len(), 2);

        view.drop_sub(SubId(0));
        let r = view.render(10.0);
        assert_eq!(r.len(), 1, "only sub 1 survives");
        assert!(r.contains_key(&ent(2)));
        // high-water for the dropped sub is gone, so a fresh frame_id on a re-opened sub 0
        // is not rejected as stale.
        view.on_snapshot(&s(SubId(0)), snap(SubId(0), 1, 12, vec![(ent(3), 3.0)]));
        assert!(view.render(12.0).contains_key(&ent(3)));
        // Dropping a sub that was never delivered is a harmless no-op.
        view.drop_sub(SubId(7));
    }

    #[test]
    fn world_pos_is_identity_for_world_frames_and_composes_shiplocal_through_its_hull() {
        use glam::DQuat;
        use vd_core::pose::StampedPose;
        let mut view = DeliveredView::default();
        // System-frame entity → identity.
        view.on_snapshot(&s(SubId(0)), snap(SubId(0), 1, 10, vec![(ent(1), 7.0)]));
        let r = view.render(10.0);
        assert_eq!(view.world_pos(&r[&ent(1)], 10.0), DVec3::new(7.0, 0.0, 0.0));

        // A hull (ent 1) at world x=100 and an interior entity in ShipLocal{ship=ent(1)} at
        // local x=5 → composes to world x=105 (hull orient identity).
        let hull = ent(1);
        let interior = ent(2);
        view.on_snapshot(&s(SubId(0)), snap(SubId(0), 2, 11, vec![(hull, 100.0)]));
        view.on_snapshot(
            &s(SubId(0)),
            SnapshotDatagram {
                sub: SubId(0),
                frame_id: 3,
                source_tick: TickId(1),
                universe_tick: UniverseTick(11),
                entities: vec![EntitySnap {
                    entity: interior,
                    pose: StampedPose::at_rest(
                        FrameRef::ShipLocal { ship: hull },
                        DVec3::new(5.0, 0.0, 0.0),
                        UniverseTick(11),
                    ),
                }],
            },
        );
        let r = view.render(11.0);
        assert_eq!(
            view.world_pos(&r[&interior], 11.0),
            DVec3::new(105.0, 0.0, 0.0),
            "interior composes through the hull pose"
        );

        // ShipLocal whose hull is NOT delivered → falls back to the frame-local pos.
        let orphan = RenderPose {
            frame: FrameRef::ShipLocal { ship: ent(99) },
            pos: DVec3::new(3.0, 0.0, 0.0),
            orient: DQuat::IDENTITY,
        };
        assert_eq!(view.world_pos(&orphan, 11.0), DVec3::new(3.0, 0.0, 0.0));

        // ShipLocal whose hull authority points at a sub with NO track → also falls back
        // (defensive: no panic, no stale pose) — exercises hull_pose's trackless-sub arm.
        view.set_authority(hull, SubId(9));
        let interior_pose = RenderPose {
            frame: FrameRef::ShipLocal { ship: hull },
            pos: DVec3::new(5.0, 0.0, 0.0),
            orient: DQuat::IDENTITY,
        };
        assert_eq!(
            view.world_pos(&interior_pose, 11.0),
            DVec3::new(5.0, 0.0, 0.0)
        );

        // Galaxy + Planet frames are identity too (the combined world-frame arm).
        let gal = RenderPose {
            frame: FrameRef::GalaxySpace,
            pos: DVec3::new(1.0, 2.0, 3.0),
            orient: DQuat::IDENTITY,
        };
        assert_eq!(view.world_pos(&gal, 11.0), DVec3::new(1.0, 2.0, 3.0));
        let planet = RenderPose {
            frame: FrameRef::PlanetCentered { planet_seed: 1 },
            pos: DVec3::new(4.0, 0.0, 0.0),
            orient: DQuat::IDENTITY,
        };
        assert_eq!(view.world_pos(&planet, 11.0), DVec3::new(4.0, 0.0, 0.0));
    }

    #[test]
    fn world_pos_rotates_a_shiplocal_interior_by_the_hull_orientation() {
        // Pins the `hull.orient * interior.pos` ROTATION term (the literal "walk inside a
        // flying ship" math) — not just the position add. A 180°-about-Y hull is
        // sign-convention-independent: local +x (5,0,0) maps to (-5,0,0), so the interior
        // lands at hull.pos + (-5,0,0) = (95,0,0) — vs (105,0,0) if orient were ignored.
        use glam::DQuat;
        use std::f64::consts::PI;
        use vd_core::pose::StampedPose;
        let mut view = DeliveredView::default();
        let hull = ent(1);
        let interior = ent(2);
        let mut hull_pose = StampedPose::at_rest(
            FrameRef::SystemSpace { system_seed: 1 },
            DVec3::new(100.0, 0.0, 0.0),
            UniverseTick(11),
        );
        hull_pose.orient = DQuat::from_rotation_y(PI); // 180° about Y
        view.on_snapshot(
            &s(SubId(0)),
            SnapshotDatagram {
                sub: SubId(0),
                frame_id: 1,
                source_tick: TickId(1),
                universe_tick: UniverseTick(11),
                entities: vec![EntitySnap {
                    entity: hull,
                    pose: hull_pose,
                }],
            },
        );
        view.on_snapshot(
            &s(SubId(0)),
            SnapshotDatagram {
                sub: SubId(0),
                frame_id: 2,
                source_tick: TickId(1),
                universe_tick: UniverseTick(11),
                entities: vec![EntitySnap {
                    entity: interior,
                    pose: StampedPose::at_rest(
                        FrameRef::ShipLocal { ship: hull },
                        DVec3::new(5.0, 0.0, 0.0),
                        UniverseTick(11),
                    ),
                }],
            },
        );
        let r = view.render(11.0);
        let world = view.world_pos(&r[&interior], 11.0);
        assert!(
            (world - DVec3::new(95.0, 0.0, 0.0)).length() < 1e-9,
            "the hull orientation must rotate the interior offset, got {world:?}"
        );
    }

    #[test]
    fn foreign_sub_and_stale_frames_are_dropped_and_counted() {
        let mut view = DeliveredView::default();
        let held = s(SubId(0));
        view.on_snapshot(&held, snap(SubId(0), 5, 10, vec![(ent(1), 1.0)]));
        // Foreign sub: dropped.
        view.on_snapshot(&held, snap(SubId(3), 6, 11, vec![(ent(1), 9.0)]));
        // Strictly-older frame on the held sub: dropped.
        view.on_snapshot(&held, snap(SubId(0), 4, 9, vec![(ent(1), 9.0)]));
        assert_eq!(view.stale_frames_dropped(), 2);
        // Only the first frame applied: one track, frozen at its sole pose.
        assert_eq!(view.render(10.0)[&ent(1)].pos, DVec3::new(1.0, 0.0, 0.0));
    }

    #[test]
    fn a_non_finite_delivered_pose_is_sanitized_at_ingress() {
        // A corrupt sender ships NaN/Inf coordinates; the view must sanitize them at
        // ingress so the renderer never gets a poisoned transform.
        let mut view = DeliveredView::default();
        let mut pose = StampedPose::at_rest(
            FrameRef::SystemSpace { system_seed: 1 },
            DVec3::new(f64::NAN, 1.0, 0.0),
            UniverseTick(10),
        );
        pose.pos.z = f64::INFINITY;
        view.on_snapshot(
            &s(SubId(0)),
            SnapshotDatagram {
                sub: SubId(0),
                frame_id: 1,
                source_tick: TickId(1),
                universe_tick: UniverseTick(10),
                entities: vec![EntitySnap {
                    entity: ent(1),
                    pose,
                }],
            },
        );
        let rendered = view.render(10.0);
        let pos = rendered[&ent(1)].pos;
        assert!(pos.is_finite(), "ingress sanitized the pose, got {pos:?}");
        assert_eq!(pos, DVec3::new(0.0, 1.0, 0.0));
        // The corruption is COUNTED, not silently fixed (the "never silent" rule).
        assert_eq!(view.nonfinite_poses(), 1);
        // A subsequent FINITE pose does NOT bump the fault counter.
        view.on_snapshot(&s(SubId(0)), snap(SubId(0), 2, 11, vec![(ent(1), 3.0)]));
        assert_eq!(view.nonfinite_poses(), 1, "a clean pose is not a fault");
    }

    #[test]
    fn authority_changed_sets_own_entity() {
        let mut view = DeliveredView::default();
        assert_eq!(view.own_entity(), None);
        view.set_authority(ent(7), SubId(0));
        assert_eq!(view.own_entity(), Some(ent(7)));
    }

    #[test]
    fn an_entity_seen_through_two_subs_renders_once_from_its_authoritative_sub() {
        // 1d.2d overlap: with BOTH subs in the held SET, a frame on EITHER is admitted, so
        // entity 1 arrives on BOTH sub 0 and sub 1. Without authority it renders from the
        // lowest sub; AuthorityChanged moves it to the named sub — either way it renders
        // EXACTLY ONCE (the source copy is composited-suppressed — the FORK 0a guarantee).
        let mut view = DeliveredView::default();
        let both = BTreeSet::from([SubId(0), SubId(1)]); // the transfer-overlap held set
        view.on_snapshot(&both, snap(SubId(0), 1, 10, vec![(ent(1), 0.0)]));
        view.on_snapshot(&both, snap(SubId(1), 1, 10, vec![(ent(1), 50.0)]));
        assert_eq!(
            view.stale_frames_dropped(),
            0,
            "BOTH subs admitted from the held set (no foreign-sub drop in the overlap)"
        );
        // No authority yet → lowest sub (0).
        let r = view.render(10.0);
        assert_eq!(r.len(), 1, "rendered exactly once");
        assert_eq!(r[&ent(1)].pos, DVec3::new(0.0, 0.0, 0.0), "from sub 0");
        // AuthorityChanged moves render authority to sub 1 (the dest — FORK 0a re-point).
        view.set_authority(ent(1), SubId(1));
        let r = view.render(10.0);
        assert_eq!(r.len(), 1, "still exactly once");
        assert_eq!(
            r[&ent(1)].pos,
            DVec3::new(50.0, 0.0, 0.0),
            "now from sub 1 (the dest)"
        );
        // The anti-vacuity probe SEES both tracks even though render composites to one: the
        // two-holder overlap was REAL at the wire.
        assert_eq!(
            view.subs_holding(ent(1)),
            vec![SubId(0), SubId(1)],
            "both the source and dest subs hold a track during the overlap"
        );
        // An entity with no delivered track at all → empty (no panic).
        assert!(view.subs_holding(ent(2)).is_empty());
    }

    #[test]
    fn own_location_frame_is_the_own_entitys_current_frame() {
        let mut view = DeliveredView::default();
        // (a) No own entity yet → None.
        assert_eq!(view.own_location_frame(), None);
        // (b) Own entity known but NO delivered track yet (authority arrived before any
        // snapshot) → None: it is not in chosen_subs.
        view.set_authority(ent(1), SubId(0));
        assert_eq!(view.own_location_frame(), None);
        // (c) A snapshot delivers the own entity → its frame IS the player's location.
        view.on_snapshot(&s(SubId(0)), snap(SubId(0), 1, 10, vec![(ent(1), 0.0)]));
        assert_eq!(
            view.own_location_frame(),
            Some(FrameRef::SystemSpace { system_seed: 1 })
        );
        // (d) Authority moved to a sub with no track for it → None (mirrors render: we
        // show nothing for an entity we have no delivered pose for).
        view.set_authority(ent(1), SubId(9));
        assert_eq!(view.own_location_frame(), None);
    }

    #[test]
    fn authority_on_a_sub_without_a_track_renders_nothing_for_that_entity() {
        // Defensive: authority names a sub the entity has not appeared on → the
        // entity is simply not rendered (no panic, no stale pose from another sub).
        let mut view = DeliveredView::default();
        view.on_snapshot(&s(SubId(0)), snap(SubId(0), 1, 10, vec![(ent(1), 1.0)]));
        view.set_authority(ent(1), SubId(9)); // a sub with no track for ent(1)
        assert!(view.render(10.0).is_empty());
    }
}
