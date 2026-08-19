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
use vd_core::pose::{FrameRef, LatticePos};
use vd_core::{EntityId, UniverseTick};
use vd_wire::channels::{SnapshotDatagram, SnapshotVerdict, SubId, classify_snapshot};

use crate::interp::{EntityTrack, RenderPose, census};

/// The inert per-row "authoritative sub" a pure-renderer client reports for its diagnosis
/// surface (`DevEntityRow.authoritative_sub`): a node-agnostic client no longer has a
/// per-entity authoritative sub, so [`DeliveredView::rendered`] reports this constant. Kept
/// as a stable diagnostic field (not deleted) so `vdctl`/process-parity decode unchanged.
pub const RENDERED_SUB: SubId = SubId(0);

/// How long (in SECONDS) a removed entity's resurrect guard outlives its removal, measured against
/// the newest APPLIED snapshot tick at the wire-taught tick rate. DERIVED, not fitted: the guard
/// exists for the unreliable lane's reorder window, whose client-side model is the interpolation
/// buffer (`ClientInterpTuning`, 100–150 ms — anything later is unusable to the interpolator), with
/// one order of magnitude of headroom for a genuinely pathological link: 0.15 s × 10. Converting at
/// the ACTUAL rate (not the compiled default) keeps the wall-clock meaning when the cluster runs
/// faster or slower than 20 Hz.
const RESURRECT_GUARD_S: f64 = 1.5;

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
    /// Entity rows SKIPPED by the one-space rule (crossing-render slice): a NON-own entity whose
    /// pose is stated in a space other than the one this client stands in — the old home's
    /// still-draining feed during a crossing's grace window. The own avatar is exempt (its new-frame
    /// row is the crossing signal). Non-zero at a crossing is normal; non-zero at steady state means
    /// a shard ships entities in a space its observer does not stand in.
    foreign_space_rows: u64,
    /// THE ECHO SPACE (crossing-render slice, measured on the round-trip ride): the space the avatar
    /// most recently LEFT. During the demote grace the OLD home's anti-vanish relay streams the
    /// leaver's LIVE row — old space, the same advancing ticks as the new home's — to the leaver's
    /// own client through the still-open old sub, and an own row is otherwise exempt from the
    /// one-space filter (a new-frame row IS the crossing signal). So the echo is named precisely:
    /// own rows stated in THIS space are dropped while it is set, and it CLEARS when any held sub
    /// closes (`drop_sub` — the reliable end of the old feed, which is the echo's only source). A
    /// faster-than-the-grace return crossing is blocked at most until that close — bounded,
    /// self-healing. Step 5 slice E deleted the RELAYED half of the echo (the live restated copy);
    /// the surviving source is the retained ghost's frozen own-frame row, which keeps this pin
    /// load-bearing until slice F retires that emit at hold closure — it retires WITH slice F.
    echo_space: Option<FrameRef>,
    /// Own-entity rows dropped as the ECHO (the old home's relayed copy of the leaver) — see
    /// `echo_space`. Non-zero during a crossing's grace window is normal.
    echo_rows_dropped: u64,
    /// THE RESURRECT GUARD (the remove message, proto_minor 14): per removed entity, the universe
    /// tick its removal was stamped at. The reliable removal races the unreliable snapshot lane, so
    /// a straggler row for a removed entity whose stamp is `<= at` is refused here — without this,
    /// one reordered datagram silently re-creates the track the removal just evicted, and it
    /// freezes forever (the exact defect the remove message exists to close). A STRICTLY NEWER row
    /// is a genuine return (the player flew back): it clears the guard and folds normally. Bounded:
    /// entries older than the newest applied tick by more than [`RESURRECT_GUARD_TICKS`] are pruned
    /// (any row that stale is unusable to the interpolator regardless).
    removed_at: BTreeMap<EntityId, UniverseTick>,
    /// Rows refused by the resurrect guard — a straggler for a removed entity. Non-zero right
    /// after a removal is normal (the unreliable lane drains); growth at steady state means a
    /// shard keeps emitting an entity it told us was removed.
    resurrect_rows_dropped: u64,
    /// The universe-tick rate the wire taught this client (`ServerControlMsg::UniverseRate`) —
    /// what converts [`RESURRECT_GUARD_S`] into ticks. `None` until taught; the guard prune then
    /// converts at the tuning DEFAULT rate, which is only ever wrong by the same factor the
    /// render cursor is until the same message arrives.
    tick_hz: Option<f64>,
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
        // THE ONE-SPACE RULE, entity lane (crossing-render review, MAJOR finding): the space this
        // client stands in, read BEFORE the fold so the own row cannot re-anchor the filter that
        // judges its siblings within one datagram. During a crossing's grace window the old home
        // keeps emitting entities in the OLD realm's frame on its still-open sub; folding those
        // beside the new home's rows draws two spaces into one picture (the measured jitter, cured
        // on the realm lane by the same rule). The OWN avatar is EXEMPT: its strictly-newer row in
        // a new frame IS the crossing signal (the track's own frame-stability guard ignores
        // non-newer foreign rows; the residual is a rare one-tick flicker when datagram loss lets
        // the relayed old-frame copy of a tick outrun the new home's copy — bounded, self-healing).
        let standing = self.own_location_frame();
        match verdict {
            SnapshotVerdict::Apply => {
                self.high_water.insert(snap.sub, snap.frame_id);
                for entity in snap.entities {
                    // THE RESURRECT GUARD (see `removed_at`): a row at or before the entity's
                    // removal is the straggler the reliable removal already superseded — refused,
                    // counted. A strictly newer row is a genuine return — but the guard CLEARS
                    // only at the fold below, never here: a filter downstream may still discard
                    // this row (the one-space rule during a dual-sub window), and a guard
                    // disarmed by a row that never folded would let the NEXT straggler re-create
                    // the track with no removal left to evict it (review-caught, the exact
                    // permanent-phantom this guard exists against).
                    if let Some(&at) = self.removed_at.get(&entity.entity)
                        && entity.pose.universe_tick <= at
                    {
                        self.resurrect_rows_dropped += 1;
                        continue;
                    }
                    let own = self.own_entity == Some(entity.entity);
                    if !own && standing.is_some_and(|s| entity.pose.frame != s) {
                        self.foreign_space_rows += 1;
                        continue;
                    }
                    // The ECHO drop (see `echo_space`): the old home's relayed copy of the OWN
                    // avatar, in the space just left, must not fold — its ticks tie the new home's
                    // and whichever row lands first would own each tick, leaving the drawn player
                    // flickering between two spaces for the whole grace window (measured).
                    if own
                        && self.echo_space.is_some_and(|e| entity.pose.frame == e)
                        && standing.is_some_and(|s| s != entity.pose.frame)
                    {
                        self.echo_rows_dropped += 1;
                        continue;
                    }
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
                    // The genuine return CLEARS the guard exactly where the row folds — every
                    // filter above has passed, so the track this row creates is a live one.
                    self.removed_at.remove(&entity.entity);
                    self.tracks
                        .entry(entity.entity)
                        .and_modify(|track| track.observe(pose))
                        .or_insert_with(|| EntityTrack::new(pose));
                }
                // The CUT detector: the avatar's delivered frame changed — remember the space just
                // left as the echo space (armed until the old sub closes, see `drop_sub`).
                let now_standing = self.own_location_frame();
                if now_standing != standing && standing.is_some() {
                    self.echo_space = standing;
                }
                // BOUND the resurrect guard, honestly: the prune trades an unbounded map for a
                // bounded re-open window. An entry older than the window CAN in principle still
                // matter (there is no second removal to re-evict a post-prune straggler), so the
                // window is sized in wall-clock terms — the transport-reorder bound, ten times
                // the interpolation buffer — at the tick rate the wire actually TAUGHT this
                // client (`set_tick_hz`), not a compiled-in default. A datagram held longer than
                // that by any real transport is a fault; the frozen figure it would draw is that
                // fault surfacing, counted at the shard as it keeps NOT emitting the entity.
                let horizon = snap
                    .universe_tick
                    .0
                    .saturating_sub(self.resurrect_guard_ticks());
                self.removed_at.retain(|_, at| at.0 >= horizon);
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

    /// Adopt the wire-taught universe-tick rate — the resurrect guard's seconds→ticks factor.
    pub fn set_tick_hz(&mut self, tick_hz: f64) {
        self.tick_hz = Some(tick_hz);
    }

    /// [`RESURRECT_GUARD_S`] in ticks, at the wire-taught rate (the tuning default until taught).
    fn resurrect_guard_ticks(&self) -> u64 {
        let hz = self
            .tick_hz
            .unwrap_or(crate::tuning::ClientInterpTuning::DEFAULT.tick_hz);
        (RESURRECT_GUARD_S * hz).ceil() as u64
    }

    /// Evict one entity's track (from the reliable `EventMsg::EntityRemoved`) — the SOUND
    /// per-entity eviction signal (reliable + explicit, unlike datagram-absence). A remove for
    /// an entity we hold no track for still ARMS the resurrect guard (the removal may simply
    /// have outrun every row) — otherwise a harmless no-op. `at` is the removing shard's
    /// universe tick — see `removed_at` for the race it closes.
    ///
    /// `own_entity` is DELIBERATELY NOT cleared: the marker names IDENTITY, not presence. The
    /// one client that legitimately receives a removal of its own avatar is the LEAVER's — its
    /// session still subscribes to the realm it just left, whose band-exit teardown removes the
    /// avatar THERE while the new home streams it live. Its own newer rows re-fold through the
    /// resurrect guard (a genuine return), but nothing would ever re-say `OwnEntity` — clearing
    /// here would permanently orphan the one-space filter and the render centering. A client
    /// whose avatar is removed because IT is leaving (logout) is on a closing session, where a
    /// stale marker beside an evicted track draws nothing.
    pub fn remove_entity(&mut self, entity: EntityId, at: UniverseTick) {
        self.tracks.remove(&entity);
        // Last-wins on the stamp: a second removal (the entity returned and left again) must
        // never REGRESS the guard to an older tick and re-open the straggler window.
        let guard = self.removed_at.entry(entity).or_insert(at);
        *guard = (*guard).max(at);
    }

    /// Forget a subscription's staleness high-water when the gateway RELIABLY closes it
    /// (`SubscriptionClosing`), so a later re-opened sub id is not rejected as stale. It does
    /// NOT evict entity tracks — those are EntityId-keyed and shared across subs, so per-entity
    /// eviction is [`DeliveredView::remove_entity`]'s job (`EventMsg::EntityRemoved`). ⚠ KNOWN
    /// residual (D-4, slice F/VU-6): the remove message covers the ENTITY-leaves case only — when
    /// the OBSERVER leaves (this client crosses away), the old realm's bystanders' tracks stay
    /// here frozen in the departed frame (their rows stop at the one-space filter); the VU-6
    /// authoritative scene re-stream on the crossing is the owed cure. A close for a sub never
    /// delivered is a harmless no-op.
    pub fn drop_sub(&mut self, sub: SubId) {
        self.high_water.remove(&sub);
        // A held sub closing is the reliable END of the old home's feed — the echo's only source —
        // so the echo pin lifts here, re-arming the own avatar's next genuine crossing cut.
        self.echo_space = None;
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

    /// The newest universe tick across every delivered entity track — the ENTITY feed's freshness as
    /// one number. SHAKE DIAGNOSIS: compared against the realm feed's equivalent, this is what shows
    /// whether the ground under a player and the player themselves are being drawn from the same
    /// moment. `None` before the first snapshot.
    #[must_use]
    pub fn newest_tick(&self) -> Option<vd_core::UniverseTick> {
        self.tracks.values().map(|t| t.newest_tick()).max()
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

    /// SLICE 6 S5 — how the tracks classify at `cursor`: `(blended, clamped_old, clamped_new)`.
    /// A feed stuck on `clamped_old` is the condition that WAS the shake (the cursor falling behind
    /// the retained history, so nothing ever blends); `clamped_new` counts stalled/frozen tracks.
    #[must_use]
    pub fn window_census(&self, cursor: f64) -> (u32, u32, u32) {
        census(self.tracks.values().map(|t| t.window_at(cursor)))
    }

    /// FAULT count of delivered poses that were non-finite and had to be sanitized.
    #[must_use]
    pub fn nonfinite_poses(&self) -> u64 {
        self.nonfinite_poses
    }

    /// Entity rows skipped by the one-space rule — see [`DeliveredView::on_snapshot`].
    #[must_use]
    pub fn foreign_space_rows(&self) -> u64 {
        self.foreign_space_rows
    }

    /// Own-entity rows dropped as the old home's echo — see `echo_space`.
    #[must_use]
    pub fn echo_rows_dropped(&self) -> u64 {
        self.echo_rows_dropped
    }

    /// Rows refused by the resurrect guard — see `removed_at`.
    #[must_use]
    pub fn resurrect_rows_dropped(&self) -> u64 {
        self.resurrect_rows_dropped
    }

    /// THE DRAWN POINT — and it is now a PASSTHROUGH, which is the whole of the client's job.
    ///
    /// Every chain of shards between the world and this session has already restated the value from the
    /// centre of the realm the session is standing in — each level subtracting the ONE placement it
    /// authored — so the number that arrives is already the number to draw. There is nothing to compose,
    /// nothing to fold, nothing to subtract, and no per-viewer derivation of any kind. What is left is a
    /// FLATTENING: a delivered position is an exact integer cell plus a metre offset, and the drawn point
    /// is the two added up, which is what `delta_m` from the identity is.
    ///
    /// It used to subtract a server-told RENDER ORIGIN: a pin realm's own absolute position, sent on the
    /// scene lane, because the server shipped every position measured from the universe root. That whole
    /// arrangement is gone — no realm has an absolute any more, because no realm is entitled to know where
    /// it sits — and with it goes the client's last piece of position arithmetic. If a subtraction ever
    /// reappears here, something upstream has started shipping numbers measured from somewhere other than
    /// the realm this session is in, and THAT is the bug to fix.
    ///
    /// THE UNIT IS TOLD, NOT CHOSEN. Adding the cell in needs metres-per-cell, and that is `pose.tier` —
    /// stamped onto the render pose from the label the shipper attached to this very value (see
    /// [`crate::interp::stated_tier`]). This function looks up nothing: it multiplies by what it was
    /// handed. Picking the unit here instead, from whatever frame name happened to ride along, is wrong by
    /// a light-year per cell the first time a galaxy-tier value reaches a client standing in a system.
    #[must_use]
    pub fn world_pos(&self, pose: &RenderPose) -> DVec3 {
        LatticePos::at(pose.cell, pose.pos).delta_m(LatticePos::default(), pose.tier)
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
        assert_eq!(rpw(&rendered[&ent(1)]), DVec3::new(5.0, 0.0, 0.0));
        assert_eq!(rpw(&rendered[&ent(2)]), DVec3::new(105.0, 0.0, 0.0));
        assert_eq!(view.stale_frames_dropped(), 0);
        // world_pos is a passthrough: the pose ships already measured from the realm this session is in.
        assert_eq!(
            view.world_pos(&rendered[&ent(1)]),
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
    fn world_pos_draws_what_it_is_handed_and_never_composes() {
        use glam::DQuat;
        use vd_core::pose::Tier;
        // The server ships every position already measured from the realm this session stands in; world_pos
        // DRAWS it, IDENTICALLY for every frame kind — no compose, no per-frame branch, no
        // fallback-to-origin, no dependence on which realms are streamed, and no origin to subtract. A
        // cell-zero pose comes back BIT-EXACT. If this test ever needs an origin argument again, the server
        // has stopped converting.
        let view = DeliveredView::default();
        let pose = |frame: FrameRef, pos: DVec3| RenderPose {
            frame,
            cell: glam::I64Vec3::ZERO,
            pos,
            orient: DQuat::IDENTITY,
            tier: crate::interp::stated_tier(frame),
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
            assert_eq!(view.world_pos(&pose(frame, pos)), pos);
        }
        // The INTEGER CELL still reaches the drawn point: a pose three cells out draws three cell-edges
        // away, not at its bare metre remainder. That half of the arithmetic survives the removal of the
        // origin subtraction — it is a flattening, not a conversion.
        let edge = Tier::Fine.cell_edge_m();
        assert_eq!(
            view.world_pos(&RenderPose {
                frame: FrameRef::PlanetCentered { planet_seed: 1 },
                cell: glam::I64Vec3::new(3, 0, 0),
                pos: DVec3::new(4.0, 0.0, 0.0),
                orient: DQuat::IDENTITY,
                tier: Tier::Fine,
            }),
            DVec3::new(3.0 * edge + 4.0, 0.0, 0.0)
        );
    }

    /// THE SHIPPER DECIDES THE UNIT — and this is the assertion that can tell "told" from "guessed"
    /// apart, which a same-tier test never can.
    ///
    /// Two poses with the IDENTICAL frame label and the IDENTICAL integer cell, differing only in the
    /// unit that was stated for that cell, must draw a whole tier apart. A `world_pos` that picked the
    /// unit off the frame name (`FrameRef::tier`, which answers `Fine` for every in-system frame) would
    /// return the same point for both, and this fails by a factor of ~10^19. That is exactly the failure
    /// waiting for the first galaxy-tier value delivered under an in-system label.
    #[test]
    fn world_pos_multiplies_by_the_unit_it_was_told_not_by_one_it_picks() {
        use glam::DQuat;
        use vd_core::pose::{COARSE_CELL_EDGE_M, FINE_CELL_EDGE_M, Tier};
        let view = DeliveredView::default();
        let at = |tier| RenderPose {
            // The SAME label on both — so nothing about the label can explain the difference below.
            frame: FrameRef::SystemSpace { system_seed: 1 },
            cell: glam::I64Vec3::new(1, 0, 0),
            pos: DVec3::ZERO,
            orient: DQuat::IDENTITY,
            tier,
        };
        assert_eq!(
            view.world_pos(&at(Tier::Fine)),
            DVec3::new(FINE_CELL_EDGE_M, 0.0, 0.0)
        );
        assert_eq!(
            view.world_pos(&at(Tier::Coarse)),
            DVec3::new(COARSE_CELL_EDGE_M, 0.0, 0.0)
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
        assert_eq!(rpw(&view.render(10.0)[&ent(1)]), DVec3::new(1.0, 0.0, 0.0));
    }

    #[test]
    fn a_non_finite_delivered_pose_is_sanitized_at_ingress() {
        // A corrupt sender ships NaN/Inf coordinates; the view must sanitize them at
        // ingress so the renderer never gets a poisoned transform.
        let mut view = DeliveredView::default();
        // Built RAW (LatticePos::at), never through the normalizing constructors: a hostile wire
        // pose carries whatever halves the sender wrote — normalizing an Inf here would saturate
        // the carry into a huge-but-finite cell and test the wrong thing.
        let pose = StampedPose {
            frame: FrameRef::SystemSpace { system_seed: 1 },
            pos: vd_core::pose::LatticePos::at(
                vd_core::glam::I64Vec3::new(0, 1024, 0),
                DVec3::new(f64::NAN, 0.0, f64::INFINITY),
            ),
            vel: DVec3::ZERO,
            orient: vd_core::glam::DQuat::IDENTITY,
            universe_tick: UniverseTick(10),
        };
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
        let pos = rpw(&rendered[&ent(1)]);
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

    /// THE ONE-SPACE RULE on the entity lane (crossing-render review, MAJOR): a NON-own entity's
    /// row stated in a space other than the one the avatar stands in is skipped + counted; the OWN
    /// avatar is exempt (its new-frame row is the crossing signal); with no avatar the filter is
    /// inert (spectator, byte-identical).
    #[test]
    fn a_foreign_space_entity_row_is_skipped_unless_it_is_the_own_avatar() {
        let planet = FrameRef::PlanetCentered { planet_seed: 7 };
        let mut view = DeliveredView::default();
        let held = s(SubId(0));
        // No avatar yet: rows in any frame fold (the spectator arm).
        view.on_snapshot(&held, snap(SubId(0), 1, 10, vec![(ent(2), 5.0)]));
        assert_eq!(view.foreign_space_rows(), 0);
        assert_eq!(view.render(10.0).len(), 1);

        // The avatar lands in the SYSTEM space; a sibling row in the SAME space folds...
        view.set_own_entity(ent(1));
        view.on_snapshot(
            &held,
            snap(SubId(0), 2, 11, vec![(ent(1), 0.0), (ent(2), 6.0)]),
        );
        assert_eq!(
            view.foreign_space_rows(),
            0,
            "in-space rows are never counted"
        );
        // ...while a NON-own row in ANOTHER space is skipped + counted.
        view.on_snapshot(
            &held,
            SnapshotDatagram {
                sub: SubId(0),
                frame_id: 3,
                source_tick: TickId(1),
                universe_tick: UniverseTick(12),
                entities: vec![EntitySnap {
                    entity: ent(2),
                    pose: StampedPose::at_rest(planet, DVec3::new(9.0, 0.0, 0.0), UniverseTick(12)),
                }],
            },
        );
        assert_eq!(
            view.foreign_space_rows(),
            1,
            "the foreign-space row is counted"
        );
        assert_eq!(
            rpw(&view.render(12.0)[&ent(2)]),
            DVec3::new(6.0, 0.0, 0.0),
            "the foreign-space row never folded"
        );
        // The OWN avatar's new-frame row IS the crossing signal — admitted, and the location flips.
        view.on_snapshot(
            &held,
            SnapshotDatagram {
                sub: SubId(0),
                frame_id: 4,
                source_tick: TickId(1),
                universe_tick: UniverseTick(13),
                entities: vec![EntitySnap {
                    entity: ent(1),
                    pose: StampedPose::at_rest(planet, DVec3::new(0.1, 0.0, 0.0), UniverseTick(13)),
                }],
            },
        );
        assert_eq!(view.foreign_space_rows(), 1, "the own avatar is exempt");
        assert_eq!(
            view.own_location_frame(),
            Some(planet),
            "the crossing flipped the location"
        );
    }

    /// THE ECHO PIN (crossing-render slice, the round-trip ride's measurement): after the avatar's
    /// cut into a new space, the old home keeps relaying the leaver's LIVE row in the OLD space with
    /// the SAME advancing ticks through the still-open old sub. Those rows are dropped (whichever
    /// row landed first would otherwise own each tick — a two-space flicker for the whole grace);
    /// the pin lifts when a held sub closes, re-arming the next genuine crossing.
    #[test]
    fn the_old_homes_echo_of_the_own_avatar_is_dropped_until_the_old_sub_closes() {
        let planet = FrameRef::PlanetCentered { planet_seed: 7 };
        let sys = FrameRef::SystemSpace { system_seed: 1 };
        let mut view = DeliveredView::default();
        let both = BTreeSet::from([SubId(0), SubId(1)]);
        view.set_own_entity(ent(1));
        // Standing in the system…
        view.on_snapshot(&both, snap(SubId(0), 1, 10, vec![(ent(1), 17.0)]));
        assert_eq!(view.own_location_frame(), Some(sys));
        // …the crossing cuts into the planet's space (the dest sub's row).
        view.on_snapshot(
            &both,
            SnapshotDatagram {
                sub: SubId(1),
                frame_id: 1,
                source_tick: TickId(1),
                universe_tick: UniverseTick(11),
                entities: vec![EntitySnap {
                    entity: ent(1),
                    pose: StampedPose::at_rest(planet, DVec3::new(0.4, 0.0, 0.0), UniverseTick(11)),
                }],
            },
        );
        assert_eq!(view.own_location_frame(), Some(planet), "the cut landed");
        // The ECHO: the old sub relays the leaver's live row in the OLD space at a NEWER tick —
        // dropped, counted; the standing frame and the drawn pose never flip back.
        view.on_snapshot(&both, snap(SubId(0), 2, 12, vec![(ent(1), 17.4)]));
        assert_eq!(view.echo_rows_dropped(), 1, "the echo is dropped");
        assert_eq!(view.own_location_frame(), Some(planet), "no backward flip");
        assert_eq!(
            rpw(&view.render(12.0)[&ent(1)]),
            DVec3::new(0.4, 0.0, 0.0),
            "the drawn pose stays in the new space"
        );
        // The old sub closes (the echo's end): the pin lifts, and a GENUINE return crossing back
        // into the system space folds again.
        view.drop_sub(SubId(0));
        view.on_snapshot(
            &both,
            SnapshotDatagram {
                sub: SubId(1),
                frame_id: 2,
                source_tick: TickId(1),
                universe_tick: UniverseTick(20),
                entities: vec![EntitySnap {
                    entity: ent(1),
                    pose: StampedPose::at_rest(sys, DVec3::new(18.0, 0.0, 0.0), UniverseTick(20)),
                }],
            },
        );
        assert_eq!(
            view.own_location_frame(),
            Some(sys),
            "the return cut folds after the close"
        );
    }

    /// The echo pin's THIRD condition, its other answer (HR5): the pin protects a LIVE standing
    /// frame. When the reliable removal has evicted the own avatar's track there IS no standing
    /// frame, so a strictly-newer own row in the echo space is the genuine return the resurrect
    /// guard admits — eating it as an echo would leave the avatar permanently undrawn (the track
    /// is gone and no other row is coming).
    #[test]
    fn an_own_row_in_the_echo_space_folds_when_no_standing_frame_remains() {
        let planet = FrameRef::PlanetCentered { planet_seed: 7 };
        let sys = FrameRef::SystemSpace { system_seed: 1 };
        let mut view = DeliveredView::default();
        let both = BTreeSet::from([SubId(0), SubId(1)]);
        view.set_own_entity(ent(1));
        // Standing in the system…
        view.on_snapshot(&both, snap(SubId(0), 1, 10, vec![(ent(1), 17.0)]));
        // …the cut into the planet arms the echo pin (echo space = the system).
        view.on_snapshot(
            &both,
            SnapshotDatagram {
                sub: SubId(1),
                frame_id: 1,
                source_tick: TickId(1),
                universe_tick: UniverseTick(11),
                entities: vec![EntitySnap {
                    entity: ent(1),
                    pose: StampedPose::at_rest(planet, DVec3::new(0.4, 0.0, 0.0), UniverseTick(11)),
                }],
            },
        );
        assert_eq!(view.own_location_frame(), Some(planet), "the cut landed");
        // The reliable removal evicts the track — no standing frame remains; the pin stays armed
        // (only a sub close clears it).
        view.remove_entity(ent(1), UniverseTick(12));
        assert_eq!(view.own_location_frame(), None, "no standing frame");
        // A strictly-newer own row in the ECHO space is the genuine return, not an echo: it folds.
        view.on_snapshot(&both, snap(SubId(0), 2, 13, vec![(ent(1), 18.0)]));
        assert_eq!(
            view.echo_rows_dropped(),
            0,
            "no live standing frame — nothing for the echo pin to protect"
        );
        assert_eq!(
            view.own_location_frame(),
            Some(sys),
            "the return row folded"
        );
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
            rpw(&r[&ent(1)]),
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
            rpw(&view.render(11.0)[&ent(1)]),
            DVec3::new(9.0, 0.0, 0.0),
            "renders the dest pose"
        );
    }

    #[test]
    fn remove_entity_evicts_the_track_but_never_the_own_identity_marker() {
        // EventMsg::EntityRemoved: the reliable per-entity eviction. Removing a non-own entity
        // drops just its track; removing the OWN avatar drops its track but KEEPS `own_entity` —
        // the marker names identity, not presence (the leaver's own client legitimately receives
        // its avatar's removal from the realm it just left, and nothing would re-say OwnEntity).
        let mut view = DeliveredView::default();
        view.set_own_entity(ent(1));
        view.on_snapshot(
            &s(SubId(0)),
            snap(SubId(0), 1, 10, vec![(ent(1), 1.0), (ent(2), 2.0)]),
        );
        assert_eq!(view.render(10.0).len(), 2);

        // Remove a NON-own entity: its track goes, own_entity is untouched.
        view.remove_entity(ent(2), UniverseTick(10));
        let r = view.render(10.0);
        assert_eq!(r.len(), 1, "only ent(2) evicted");
        assert!(r.contains_key(&ent(1)));
        assert_eq!(view.own_entity(), Some(ent(1)), "own unchanged");

        // Remove the OWN avatar: its track goes, the identity marker STAYS — the new home's
        // strictly-newer rows re-fold through the resurrect guard and the client keeps knowing
        // which entity is itself.
        view.remove_entity(ent(1), UniverseTick(10));
        assert!(view.render(10.0).is_empty(), "the avatar track was evicted");
        assert_eq!(
            view.own_entity(),
            Some(ent(1)),
            "the identity marker survives its own removal"
        );
        view.on_snapshot(&s(SubId(0)), snap(SubId(0), 2, 11, vec![(ent(1), 3.0)]));
        assert_eq!(
            view.own_location_frame(),
            Some(FrameRef::SystemSpace { system_seed: 1 }),
            "the newer own row re-folds and the one-space filter stays anchored"
        );
        // Removing an entity we hold no track for still ARMS the resurrect guard (the removal may
        // have outrun every row of a just-entered entity) — otherwise a harmless no-op.
        view.remove_entity(ent(9), UniverseTick(10));
    }

    #[test]
    fn the_resurrect_guard_refuses_stragglers_admits_returns_and_stays_bounded() {
        let mut view = DeliveredView::default();
        view.on_snapshot(&s(SubId(0)), snap(SubId(0), 1, 10, vec![(ent(1), 1.0)]));
        assert_eq!(view.render(10.0).len(), 1);
        // The reliable removal at tick 12 evicts the track and arms the guard.
        view.remove_entity(ent(1), UniverseTick(12));
        assert!(view.render(12.0).is_empty());
        // A STRAGGLER: a reordered datagram whose rows predate the removal (tick 11 <= 12). It
        // must NOT re-create the track — that is the frozen-phantom defect the message closes.
        view.on_snapshot(&s(SubId(0)), snap(SubId(0), 2, 11, vec![(ent(1), 5.0)]));
        assert!(
            view.render(12.0).is_empty(),
            "a straggler row never resurrects a removed track"
        );
        assert_eq!(view.resurrect_rows_dropped(), 1);
        // A GENUINE RETURN: a strictly newer row (tick 13 > 12) clears the guard and folds.
        view.on_snapshot(&s(SubId(0)), snap(SubId(0), 3, 13, vec![(ent(1), 7.0)]));
        assert_eq!(view.render(13.0).len(), 1, "the player flew back");
        // And once cleared, the old stamp is forgotten — a fresh row folds unguarded.
        view.on_snapshot(&s(SubId(0)), snap(SubId(0), 4, 14, vec![(ent(1), 8.0)]));
        assert_eq!(view.resurrect_rows_dropped(), 1, "no further refusals");

        // A SECOND REMOVAL never regresses the guard (last-wins on the stamp)…
        view.remove_entity(ent(1), UniverseTick(20));
        view.remove_entity(ent(1), UniverseTick(15));
        view.on_snapshot(&s(SubId(0)), snap(SubId(0), 5, 18, vec![(ent(1), 9.0)]));
        assert!(
            view.render(18.0).is_empty(),
            "tick 18 <= the newest removal (20): still refused"
        );
        // …and the guard PRUNES once the applied clock moves past the derived window: a removal
        // this stale protects against nothing (the row would be unusable to the interpolator),
        // so an ancient stamp no longer blocks a fold.
        view.on_snapshot(&s(SubId(0)), snap(SubId(0), 6, 60, vec![(ent(2), 1.0)]));
        view.on_snapshot(&s(SubId(0)), snap(SubId(0), 7, 19, vec![(ent(1), 9.5)]));
        assert_eq!(
            view.render(60.0).len(),
            2,
            "the pruned guard no longer refuses; the map stays sized to recent departures"
        );
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
        view.remove_entity(ent(1), UniverseTick(10));
        assert_eq!(view.own_location_frame(), None);
    }
}
