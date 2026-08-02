//! The client's DELIVERED world view — decoded snapshots only, wire truth (the
//! same honesty rule as the `WireMonitor`: the client renders what it was actually
//! sent, never internal hope).
//!
//! ## Pure renderer — ONE track per `EntityId`, latest-wins (S6)
//! Poses are kept per-[`EntityId`] (NOT per-`(SubId, EntityId)`): the client is a PURE
//! RENDERER that never learns which node owns an entity. It renders EVERY entity at its
//! authoritative coordinate, and when the server re-homes authority the entity simply keeps
//! arriving (on whichever sub the new owner rides) into the SAME per-entity track — latest
//! delivered pose wins ([`EntityTrack::observe`], which collapses the interp window on a
//! frame change so a cross-realm re-home is clean, never a blended garbage intermediate).
//! `held_subs` stays a SET on the client (multi-realm AoI legitimately holds several subs —
//! a player in a ship docked at a station on a planet holds Ship+Station+Planet+System subs
//! at once); what was DELETED is the per-entity authority disambiguation
//! (`authoritative_sub`/`set_authority`/`chosen_subs`) whose only purpose was to de-dup an
//! entity the SERVER chose to emit twice. The server sends each entity once (in its
//! authoritative realm); the client just renders it.
//!
//! ## Track eviction — sound signals only (NEVER datagram-absence)
//! A track is dropped ONLY on a RELIABLE signal, never on absence from a datagram:
//! snapshots ride UNRELIABLE datagrams, so an entity merely ABSENT from one frame is
//! indistinguishable from packet loss — evicting on absence would delete live entities.
//! The reliable signals that bound the view to its live set:
//!   * per-ENTITY teardown — [`DeliveredView::remove_entity`], driven by the reliable
//!     `EventMsg::EntityRemoved` (S6): the server-authoritative "this entity left your view"
//!     signal. With per-ENTITY tracks this is THE eviction primitive (a de-owned copy the
//!     send-once server no longer streams).
//!   * per-SUB drop — [`DeliveredView::drop_sub`], driven by the reliable
//!     `SubscriptionClosing`: forgets that sub's staleness high-water so a re-opened sub id
//!     is not rejected as stale. It no longer evicts entity tracks (they are EntityId-keyed,
//!     shared across subs) — `EntityRemoved` owns per-entity eviction.

use std::collections::BTreeMap;

use glam::DVec3;
use vd_core::EntityId;
use vd_core::pose::{FrameRef, LatticePos};
use vd_wire::channels::{SnapshotDatagram, SnapshotVerdict, SubId, classify_snapshot};

use crate::interp::{EntityTrack, RenderPose};

/// The inert per-row "authoritative sub" a pure-renderer client reports for its diagnosis
/// surface (`DevEntityRow.authoritative_sub`): a node-agnostic client no longer has a
/// per-entity authoritative sub, so [`DeliveredView::rendered`] reports this constant. Kept
/// as a stable diagnostic field (not deleted) so `vdctl`/process-parity decode unchanged.
pub const RENDERED_SUB: SubId = SubId(0);

/// The decoded, delivered world view.
#[derive(Clone, Debug, Default)]
pub struct DeliveredView {
    /// ONE track per entity (latest-wins) — the pure-renderer shape. An entity re-homing
    /// across shards keeps arriving into the SAME track regardless of which sub carries it.
    tracks: BTreeMap<EntityId, EntityTrack>,
    /// Per-sub staleness high-water for the §6.3 gate — `held_subs` stays a SET (multi-realm
    /// AoI), so a strictly-older `frame_id` on a given sub is still dropped per sub.
    high_water: BTreeMap<SubId, u64>,
    own_entity: Option<EntityId>,
    stale_frames_dropped: u64,
    /// FAULT count: delivered poses that carried a non-finite (NaN/Inf) component and had
    /// to be sanitized at ingress. A corrupt/diverged sender is a real fault, so it is
    /// COUNTED (not silently fixed) — the codebase's "never silent" discipline.
    nonfinite_poses: u64,
    /// A5 — the SERVER-TOLD render origin (the last `pin_abs` from a `RealmRegistry`/`RealmSceneDelta`). Every
    /// drawn point + the camera subtract it via [`DeliveredView::world_pos`]. Defaults to the identity (ZERO):
    /// a walk cluster never sends a pin, and a walk pose is already root-absolute (identity fold), so
    /// subtracting ZERO renders it verbatim (byte-identical). Once a pin is received it is CARRIED indefinitely
    /// — never reset to ZERO — so a re-home whose fresh registry is momentarily in flight keeps the last origin
    /// (never a teleport to the world origin; the pin rides the reliable Control lane).
    render_origin: LatticePos,
}

impl DeliveredView {
    /// Fold one delivered snapshot in, through the shared §6.3 gate
    /// ([`classify_snapshot`]): apply the entities into their per-ENTITY tracks (latest-wins)
    /// or count the drop. `held_subs` is the SET of subscriptions the client currently holds
    /// (multi-realm AoI): a frame on ANY held sub is admitted; a frame on a sub the client
    /// does not hold drops. Because tracks are keyed by [`EntityId`] (NOT by sub), an entity
    /// that re-homes to a new owner keeps folding into the SAME track no matter which sub now
    /// carries it — [`EntityTrack::observe`] collapses the window on the cross-realm frame
    /// change, so the render is clean without any client-side node awareness.
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
                        .entry(entity.entity)
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

    /// Record which entity is THIS client's own avatar (from the node-agnostic
    /// `ServerControlMsg::OwnEntity`). It names ONLY the entity — never a sub / owning node —
    /// so the pure-renderer client learns which entity to center on WITHOUT learning which
    /// shard simulates it.
    pub fn set_own_entity(&mut self, entity: EntityId) {
        self.own_entity = Some(entity);
    }

    /// Evict one entity's track (from the reliable `EventMsg::EntityRemoved`) — the SOUND
    /// per-entity eviction signal (reliable + explicit, unlike datagram-absence). If the
    /// removed entity is the own avatar, `own_entity` is cleared too (the server told us it
    /// left our view). A remove for an entity we hold no track for is a harmless no-op.
    pub fn remove_entity(&mut self, entity: EntityId) {
        self.tracks.remove(&entity);
        if self.own_entity == Some(entity) {
            self.own_entity = None;
        }
    }

    /// Forget a subscription's staleness high-water when the gateway RELIABLY closes it
    /// (`SubscriptionClosing`), so a later re-opened sub id is not rejected as stale. It does
    /// NOT evict entity tracks — those are EntityId-keyed and shared across subs, so per-entity
    /// eviction is [`DeliveredView::remove_entity`]'s job (`EventMsg::EntityRemoved`). A close
    /// for a sub never delivered is a harmless no-op.
    pub fn drop_sub(&mut self, sub: SubId) {
        self.high_water.remove(&sub);
    }

    /// The render poses at `cursor`: each entity rendered EXACTLY ONCE from its single
    /// per-entity track. A pure-renderer client has one track per entity by construction, so
    /// there is no cross-sub duplicate to suppress.
    #[must_use]
    pub fn render(&self, cursor: f64) -> BTreeMap<EntityId, RenderPose> {
        self.tracks
            .iter()
            .map(|(entity, track)| (*entity, track.sample(cursor)))
            .collect()
    }

    /// Like [`DeliveredView::render`] but shaped for the dev-control diagnosis surface, which
    /// carries a per-row `authoritative_sub`. A node-agnostic client has no per-entity
    /// authoritative sub, so every row reports the inert [`RENDERED_SUB`] constant (kept so
    /// `vdctl`/process-parity decode the row unchanged).
    #[must_use]
    pub fn rendered(&self, cursor: f64) -> Vec<(EntityId, SubId, RenderPose)> {
        self.tracks
            .iter()
            .map(|(entity, track)| (*entity, RENDERED_SUB, track.sample(cursor)))
            .collect()
    }

    #[must_use]
    pub fn own_entity(&self) -> Option<EntityId> {
        self.own_entity
    }

    /// The subs that currently hold a delivered track for `entity` — a pure-renderer client
    /// keeps ONE per-entity track (not per-sub), so this reports the inert [`RENDERED_SUB`]
    /// when a track exists and nothing otherwise. Retained as a read-only diagnosis surface
    /// (the transfer-gate anti-vacuity probe) whose semantics collapse with the send-once
    /// pure-renderer model: an entity either has a delivered track or it does not.
    #[must_use]
    pub fn subs_holding(&self, entity: EntityId) -> Vec<SubId> {
        if self.tracks.contains_key(&entity) {
            vec![RENDERED_SUB]
        } else {
            Vec::new()
        }
    }

    /// The frame the OWN entity is currently in — its track's leading edge — i.e. the
    /// player's LOCATION (realm), the basis for the player-stats HUD and the `vdctl` location
    /// readout. `None` until the own entity is known AND has a delivered track. Cursor-free
    /// (frame does not interpolate); it is the delivered FrameRef, so it is fence-validated
    /// and changes only on a real cross-realm move (the re-home flips it, node-agnostically).
    #[must_use]
    pub fn own_location_frame(&self) -> Option<FrameRef> {
        let own = self.own_entity?;
        self.tracks.get(&own).map(|track| track.current_frame())
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

    /// The server-told render origin the renderer subtracts (A5). Identity (ZERO) until the first pin.
    #[must_use]
    pub fn render_origin(&self) -> LatticePos {
        self.render_origin
    }

    /// Adopt a fresh server-told render origin (a `RealmRegistry`/`RealmSceneDelta` `pin_abs`). CARRY-last-pin:
    /// it only ever moves to a real told value, never resets to identity, so a re-anchor never teleports the
    /// scene to the world origin between the trigger and the fresh registry's arrival.
    pub fn set_render_origin(&mut self, origin: LatticePos) {
        self.render_origin = origin;
    }

    /// The RENDER range-reduction (A5 — the server is authoritative). The server COMPOSES every entity's
    /// final position and ships it ROOT-ABSOLUTE (minor 7); the client only DRAWS. So this subtracts the
    /// server-told render `origin` in EXACT lattice arithmetic — NO compose, NO fold, NO per-viewer position
    /// derivation, NO fallback-to-origin. Through P4 the cell is ZERO and this reduces to `pose.pos - origin`
    /// in metres; the cell-aware rebase (non-zero galaxy cells) lands at A8. At `origin == identity` it returns
    /// `pose.pos` BIT-EXACT (the byte-floor — a walk pin folds to identity). Never panics; always finite (poses
    /// sanitized at the decode ingress). The camera eye and every drawn point subtract this SAME `origin`, so a
    /// re-pin shifts camera and content by one identical vector (invisible); it is computed FRESHLY per object,
    /// never as a cached-scene translation (which would contaminate near objects with the far delta's rounding).
    #[must_use]
    pub fn world_pos(&self, pose: &RenderPose, origin: LatticePos) -> DVec3 {
        LatticePos::at(pose.cell, pose.pos).delta_m(origin, pose.frame.tier())
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
        // world_pos at the identity origin returns the pose verbatim (A5 pure range-reduction).
        assert_eq!(
            view.world_pos(&rendered[&ent(1)], LatticePos::local(DVec3::ZERO)),
            DVec3::new(5.0, 0.0, 0.0)
        );
    }

    #[test]
    fn drop_sub_forgets_only_the_high_water_and_keeps_entity_tracks() {
        // The pure-renderer drop_sub forgets a sub's STALENESS high-water (so a re-opened sub id
        // is not rejected as stale) but does NOT evict entity tracks — those are EntityId-keyed and
        // per-entity eviction is remove_entity's job (EventMsg::EntityRemoved).
        let mut view = DeliveredView::default();
        view.on_snapshot(&s(SubId(0)), snap(SubId(0), 5, 10, vec![(ent(1), 1.0)]));
        assert_eq!(view.render(10.0).len(), 1);

        // Drop sub 0: the entity track SURVIVES (tracks are not sub-keyed), only the high-water is
        // forgotten — so a re-opened sub 0 with a LOW frame_id is admitted, not rejected as stale.
        view.drop_sub(SubId(0));
        assert_eq!(
            view.render(10.0).len(),
            1,
            "entity track survives a drop_sub"
        );
        view.on_snapshot(&s(SubId(0)), snap(SubId(0), 1, 12, vec![(ent(2), 3.0)]));
        assert!(
            view.render(12.0).contains_key(&ent(2)),
            "the low frame_id is admitted after the high-water was forgotten"
        );
        // Dropping a sub that was never delivered is a harmless no-op.
        view.drop_sub(SubId(7));
    }

    #[test]
    fn world_pos_subtracts_the_render_origin_and_never_composes() {
        use glam::DQuat;
        // A5 — the server ships ABSOLUTE positions; world_pos is a PURE range-reduction against the
        // server-told render origin, IDENTICAL for every frame kind (no compose, no per-frame branch, no
        // fallback-to-origin, no dependence on which realms are streamed). At the identity origin it returns
        // pose.pos BIT-EXACT (the byte-floor a walk pin folds to).
        let view = DeliveredView::default();
        let pose = |frame: FrameRef, pos: DVec3| RenderPose {
            frame,
            cell: glam::I64Vec3::ZERO,
            pos,
            orient: DQuat::IDENTITY,
        };
        // Identity origin ⇒ pose.pos verbatim, for EVERY frame kind (a ShipLocal whose hull is undelivered no
        // longer gets special-cased — the client never composes).
        for (frame, pos) in [
            (
                FrameRef::SystemSpace { system_seed: 7 },
                DVec3::new(7.0, 0.0, 0.0),
            ),
            (FrameRef::GalaxySpace, DVec3::new(1.0, 2.0, 3.0)),
            (
                FrameRef::PlanetCentered { planet_seed: 1 },
                DVec3::new(4.0, 0.0, 0.0),
            ),
            (
                FrameRef::AreaLocal {
                    planet_seed: 1,
                    area_seed: 2,
                },
                DVec3::new(5.0, 6.0, 7.0),
            ),
            (
                FrameRef::StationLocal { station_seed: 9 },
                DVec3::new(8.0, 0.0, 0.0),
            ),
            (
                FrameRef::ShipLocal { ship: ent(99) },
                DVec3::new(3.0, 0.0, 0.0),
            ),
        ] {
            assert_eq!(
                view.world_pos(&pose(frame, pos), LatticePos::local(DVec3::ZERO)),
                pos
            );
        }
        // A non-identity origin subtracts EXACTLY and never invents a value from a streamed realm: a Planet
        // occupant at (4,0,0) with the origin at (100,0,0) renders at (-96,0,0) — never (4,0,0) (the old
        // silent-identity-on-miss bug) nor a composed (104,..).
        assert_eq!(
            view.world_pos(
                &pose(
                    FrameRef::PlanetCentered { planet_seed: 1 },
                    DVec3::new(4.0, 0.0, 0.0)
                ),
                LatticePos::local(DVec3::new(100.0, 0.0, 0.0)),
            ),
            DVec3::new(-96.0, 0.0, 0.0)
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
        let pose = StampedPose::at_rest(
            FrameRef::SystemSpace { system_seed: 1 },
            DVec3::new(f64::NAN, 1.0, f64::INFINITY),
            UniverseTick(10),
        );
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
    fn own_entity_records_the_avatar_and_is_node_agnostic() {
        // OwnEntity names ONLY the entity — the pure-renderer own-avatar cue, with no sub/owner.
        let mut view = DeliveredView::default();
        assert_eq!(view.own_entity(), None);
        view.set_own_entity(ent(7));
        assert_eq!(view.own_entity(), Some(ent(7)));
        // Re-announcing (idempotent — the dest re-point re-confirms the same avatar) is a no-op.
        view.set_own_entity(ent(7));
        assert_eq!(view.own_entity(), Some(ent(7)));
    }

    #[test]
    fn a_re_homed_entity_renders_once_latest_wins_across_subs() {
        // THE pure-renderer node-agnostic property: an entity arrives on the source sub, then the
        // SAME entity arrives on the DEST sub (a different realm frame) after the server re-homes
        // it. With per-ENTITY tracks it folds into ONE track — the frame change collapses the interp
        // window to the dest pose (EntityTrack::observe), so it renders EXACTLY ONCE at the dest
        // coordinate, and the client never learned which node owns it.
        let mut view = DeliveredView::default();
        let both = BTreeSet::from([SubId(0), SubId(1)]); // multi-realm AoI: both subs held
        view.on_snapshot(&both, snap(SubId(0), 1, 10, vec![(ent(1), 0.0)]));
        // Same tick, same system frame, DEST sub, farther along: latest-wins updates the pose.
        view.on_snapshot(&both, snap(SubId(1), 1, 10, vec![(ent(1), 50.0)]));
        assert_eq!(
            view.stale_frames_dropped(),
            0,
            "both held subs admitted (no foreign-sub drop across the AoI)"
        );
        let r = view.render(10.0);
        assert_eq!(r.len(), 1, "rendered exactly once (one track per entity)");
        assert_eq!(
            r[&ent(1)].pos,
            DVec3::new(50.0, 0.0, 0.0),
            "latest-wins: the dest-sub pose replaced the source copy"
        );
        // The diagnosis probe reports the inert RENDERED_SUB when a track exists, else nothing.
        assert_eq!(view.subs_holding(ent(1)), vec![RENDERED_SUB]);
        assert!(view.subs_holding(ent(2)).is_empty());
    }

    #[test]
    fn a_cross_realm_re_home_flips_the_frame_cleanly() {
        // The re-home from realm A (system 1) to realm B (system 8): the dest frame DIFFERS, so
        // EntityTrack::observe collapses the window and the render reports the DEST realm frame —
        // node-agnostically (the client only ever saw two poses for one EntityId).
        let mut view = DeliveredView::default();
        let both = BTreeSet::from([SubId(0), SubId(1)]);
        view.set_own_entity(ent(1));
        view.on_snapshot(&both, snap(SubId(0), 1, 10, vec![(ent(1), 0.0)]));
        assert_eq!(
            view.own_location_frame(),
            Some(FrameRef::SystemSpace { system_seed: 1 }),
            "starts in realm A"
        );
        // A dest frame (system 8) on the OTHER sub — the re-home. The frame flips cleanly.
        view.on_snapshot(
            &both,
            SnapshotDatagram {
                sub: SubId(1),
                frame_id: 1,
                source_tick: TickId(1),
                universe_tick: UniverseTick(11),
                entities: vec![EntitySnap {
                    entity: ent(1),
                    pose: StampedPose::at_rest(
                        FrameRef::SystemSpace { system_seed: 8 },
                        DVec3::new(9.0, 0.0, 0.0),
                        UniverseTick(11),
                    ),
                }],
            },
        );
        assert_eq!(
            view.own_location_frame(),
            Some(FrameRef::SystemSpace { system_seed: 8 }),
            "the re-home flipped the location to realm B — no node info needed"
        );
        assert_eq!(
            view.render(11.0)[&ent(1)].pos,
            DVec3::new(9.0, 0.0, 0.0),
            "renders the dest pose"
        );
    }

    #[test]
    fn remove_entity_evicts_the_track_and_clears_own_when_it_is_the_avatar() {
        // EventMsg::EntityRemoved: the reliable per-entity eviction. Removing a non-own entity drops
        // just its track; removing the OWN avatar also clears own_entity (the server said it left).
        let mut view = DeliveredView::default();
        view.set_own_entity(ent(1));
        view.on_snapshot(
            &s(SubId(0)),
            snap(SubId(0), 1, 10, vec![(ent(1), 1.0), (ent(2), 2.0)]),
        );
        assert_eq!(view.render(10.0).len(), 2);

        // Remove a NON-own entity: its track goes, own_entity is untouched.
        view.remove_entity(ent(2));
        let r = view.render(10.0);
        assert_eq!(r.len(), 1, "only ent(2) evicted");
        assert!(r.contains_key(&ent(1)));
        assert_eq!(view.own_entity(), Some(ent(1)), "own unchanged");

        // Remove the OWN avatar: its track goes AND own_entity clears.
        view.remove_entity(ent(1));
        assert!(view.render(10.0).is_empty(), "the avatar track was evicted");
        assert_eq!(
            view.own_entity(),
            None,
            "own cleared when the avatar is removed"
        );
        // Removing an entity we hold no track for is a harmless no-op.
        view.remove_entity(ent(9));
    }

    #[test]
    fn own_location_frame_is_the_own_entitys_current_frame() {
        let mut view = DeliveredView::default();
        // (a) No own entity yet → None.
        assert_eq!(view.own_location_frame(), None);
        // (b) Own entity known but NO delivered track yet (OwnEntity arrived before any snapshot)
        // → None: there is no track to read a frame from.
        view.set_own_entity(ent(1));
        assert_eq!(view.own_location_frame(), None);
        // (c) A snapshot delivers the own entity → its frame IS the player's location.
        view.on_snapshot(&s(SubId(0)), snap(SubId(0), 1, 10, vec![(ent(1), 0.0)]));
        assert_eq!(
            view.own_location_frame(),
            Some(FrameRef::SystemSpace { system_seed: 1 })
        );
        // (d) The own entity is removed → None again (no track to read).
        view.remove_entity(ent(1));
        assert_eq!(view.own_location_frame(), None);
    }
}
