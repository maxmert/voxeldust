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
//! ## Track eviction is a P2 wire-contract decision (intentionally deferred)
//! Tracks are NEVER removed here, and that is deliberate at P1.5: snapshots ride
//! UNRELIABLE datagrams, so an entity merely ABSENT from one datagram is
//! indistinguishable from packet loss — evicting on absence would delete live entities
//! whenever a frame dropped. Correct eviction needs an explicit signal the wire does not
//! carry yet: a per-sub despawn/leave-AoI event (reliable) or full-membership snapshot
//! semantics. Both arrive with cross-shard overlap + interest management in P2, where
//! "out of range ⇒ rendered nowhere" and bounded memory become real requirements; until
//! then the world is a handful of dots through one sub and growth is non-issue. (Raised
//! by the Slice-0–T3 audit as a latent scale/ghost-render gap; the fix is gated on that
//! P2 wire decision, not bolted on here as an unsound TTL.)

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
    /// count the drop. `held_sub` is the subscription the client currently holds.
    /// Returns the verdict so the caller anchors the render clock only on `Apply`.
    pub fn on_snapshot(
        &mut self,
        held_sub: Option<SubId>,
        snap: SnapshotDatagram,
    ) -> SnapshotVerdict {
        let high_water = self.high_water.get(&snap.sub).copied();
        let verdict = classify_snapshot(held_sub, high_water, snap.sub, snap.frame_id);
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
}

/// The frame-evaluation seam: map a rendered pose's frame-local position into world
/// space. In P1.5 every frame is static (system/planet origin fixed), so this is
/// the identity — but it is THE single chokepoint where P8 composes a `ShipLocal`
/// pose through its moving hull and P10 applies the galaxy ly-cell offset, without
/// touching any call site.
#[must_use]
pub fn world_pos(pose: &RenderPose) -> DVec3 {
    pose.pos
}

#[cfg(test)]
mod tests {
    use super::*;
    use vd_core::entity_kind::EntityKind;
    use vd_core::pose::{FrameRef, StampedPose};
    use vd_core::{TickId, UniverseTick};
    use vd_wire::channels::EntitySnap;

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
        let held = Some(SubId(0));
        view.on_snapshot(
            held,
            snap(SubId(0), 1, 10, vec![(ent(1), 0.0), (ent(2), 100.0)]),
        );
        view.on_snapshot(
            held,
            snap(SubId(0), 2, 12, vec![(ent(1), 10.0), (ent(2), 110.0)]),
        );
        let rendered = view.render(11.0); // tick-11 cursor → window midpoint
        assert_eq!(rendered.len(), 2);
        assert_eq!(rendered[&ent(1)].pos, DVec3::new(5.0, 0.0, 0.0));
        assert_eq!(rendered[&ent(2)].pos, DVec3::new(105.0, 0.0, 0.0));
        assert_eq!(view.stale_frames_dropped(), 0);
        // world_pos is the identity for P1.5 static frames.
        assert_eq!(world_pos(&rendered[&ent(1)]), DVec3::new(5.0, 0.0, 0.0));
    }

    #[test]
    fn foreign_sub_and_stale_frames_are_dropped_and_counted() {
        let mut view = DeliveredView::default();
        let held = Some(SubId(0));
        view.on_snapshot(held, snap(SubId(0), 5, 10, vec![(ent(1), 1.0)]));
        // Foreign sub: dropped.
        view.on_snapshot(held, snap(SubId(3), 6, 11, vec![(ent(1), 9.0)]));
        // Strictly-older frame on the held sub: dropped.
        view.on_snapshot(held, snap(SubId(0), 4, 9, vec![(ent(1), 9.0)]));
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
            Some(SubId(0)),
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
        view.on_snapshot(Some(SubId(0)), snap(SubId(0), 2, 11, vec![(ent(1), 3.0)]));
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
        // P2 overlap: entity 1 arrives on BOTH sub 0 and sub 1. Without authority it
        // renders from the lowest sub; AuthorityChanged moves it to the named sub —
        // either way it renders EXACTLY ONCE (no duplicate).
        let mut view = DeliveredView::default();
        // (held_sub only gates the foreign-sub drop; here both subs are "held" for
        // the test by passing the snapshot's own sub.)
        view.on_snapshot(Some(SubId(0)), snap(SubId(0), 1, 10, vec![(ent(1), 0.0)]));
        view.on_snapshot(Some(SubId(1)), snap(SubId(1), 1, 10, vec![(ent(1), 50.0)]));
        // No authority yet → lowest sub (0).
        let r = view.render(10.0);
        assert_eq!(r.len(), 1, "rendered exactly once");
        assert_eq!(r[&ent(1)].pos, DVec3::new(0.0, 0.0, 0.0), "from sub 0");
        // AuthorityChanged moves render authority to sub 1.
        view.set_authority(ent(1), SubId(1));
        let r = view.render(10.0);
        assert_eq!(r.len(), 1, "still exactly once");
        assert_eq!(r[&ent(1)].pos, DVec3::new(50.0, 0.0, 0.0), "now from sub 1");
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
        view.on_snapshot(Some(SubId(0)), snap(SubId(0), 1, 10, vec![(ent(1), 0.0)]));
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
        view.on_snapshot(Some(SubId(0)), snap(SubId(0), 1, 10, vec![(ent(1), 1.0)]));
        view.set_authority(ent(1), SubId(9)); // a sub with no track for ent(1)
        assert!(view.render(10.0).is_empty());
    }
}
