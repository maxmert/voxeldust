//! THE CONTAINMENT SCAN: which realm each subject is inside, and the crossing that follows when
//! the answer changes.
//!
//! Owns: the per-entity hysteretic membership bitset, the crossing cooldown and the durable
//! re-drive latch, the per-subject evaluation itself, and the ONE fan-out that dispatches a
//! committed re-home by the subject's durability class.
//!
//! Does NOT own: any notion of what a subject IS. A ship, a station, a moon and a rock cross by
//! identical code because that code cannot tell them apart — there is no orbit, gravity or thrust
//! symbol on this path, and a "does this child move?" test inside a placement lookup would be a
//! specific and is forbidden (SL4). Nor does it drive the saga: it emits a request and the
//! orchestrator owns what happens next.

use super::{
    Dots, HandoffHolds, HoldRole, OwnedTransients, Placements, RealmAuthority, RealmRegions,
    StubConfig, StubStats, book_anchor, close_hold, push_demand,
};
use crate::io::MsgClass;
use crate::runtime::{ClockSample, OutboundBox};
use bevy_ecs::prelude::{Res, ResMut, Resource};
use std::collections::{BTreeMap, BTreeSet};
use vd_core::child_index::ChildIndex;
use vd_core::entity_kind::{DurabilityClass, durability_of};
use vd_core::geometry::{DepthKey, RealmRegion, container, should_rehome};
use vd_core::placement::PlacementBook;
use vd_core::pose::{FrameRef, LatticePos, RealmId, StampedPose};
use vd_core::{EntityId, Fence, NodeId, SessionId, TickId, TransferId, UniverseTick};
use vd_wire::intershard::{
    CrossingAborted, CrossingRequest, DemandVerb, ExteriorCrossingRequest, InterShardFlow,
    TransientCrossingRequest, crossing_transfer_id,
};
use vd_wire::seams::directory::DirectoryKey;

/// The re-emit payload a held durable `RequestInFlight` latch carries so `redrive_stranded_crossings`
/// (3f-D4) can re-mint the SAME `CrossingRequest` without recovering these fields from the live
/// `Dots`/geometry (`RequestInFlight` holds only `EntityId → TransferId`). Captured at the emit site
/// (`fan_out_crossing`'s durable `Vacant` arm) alongside the latch. All `Copy` (so `CrossingState`
/// stays `Copy`+`Default`); `from_realm` is `config.realm` (constant) and `subject` is the map key, so
/// only the three genuinely-per-crossing fields ride here. `None` when no latch is held.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct LatchedCrossing {
    /// The destination realm the crossing resolved (`winner.to_realm`) — re-emitted verbatim.
    pub to_realm: RealmId,
    /// The subject's authority fence at latch time (the `CrossingRequest.subject_fence` + the second
    /// component of the deterministic [`crossing_transfer_id`], so the re-drive re-mints the SAME id).
    pub subject_fence: Fence,
    /// The subject's session (the durable crossing carries it so the orchestrator's saga can
    /// `PrepareSubscribe` to the client's gateway). `SessionId::NONE` for an exterior.
    pub session: SessionId,
    /// The ruler switch, slice 1 — the latch is a driven child's EXTERIOR, so the re-drive re-mints an
    /// `ExteriorCrossingRequest`, never a session-bearing one.
    pub exterior: bool,
    /// The dest realm's PARENT provenance (the container region's `parent`) — re-emitted VERBATIM so the
    /// stranded-latch re-drive re-mints the byte-identical `CrossingRequest` (an Area dest's frame still
    /// forms on the re-drive). `None` for a non-Area dest.
    pub to_parent: Option<RealmId>,
}

/// The per-entity crossing state the CONTAINMENT trigger carries between ticks — the post-commit cooldown
/// plus the durable re-drive latch. (The per-region HYSTERETIC membership lives in the separate
/// [`ContainmentProgress`] bitset, task #135 §2.3; this struct no longer holds the old single-winner
/// dwell.) Keyed by the subject entity (owned dots AND held transients); lazily evicted when it leaves.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct CrossingState {
    /// The `local_tick` a crossing last committed (arms the symmetric post-commit cooldown,
    /// [`should_rehome`]). `None` = never committed. PER-ENTITY anti-thrash; reset on abort so the
    /// re-home can re-fire without a physical re-cross (the container still differs from the owner).
    pub last_commit_tick: Option<TickId>,
    /// The per-entity crossing-ATTEMPT counter (Slice 3f-D, H2). Bumped ONLY when a NEW `RequestInFlight`
    /// latch is taken (`fan_out_crossing`'s `Vacant` arm), i.e. only AFTER the prior latch cleared on
    /// abort/commit — so it feeds a UNIQUE [`crossing_transfer_id`] per attempt: a stale `CrossingAborted`
    /// for an old attempt can never wrong-clear a same-fence re-cross's fresh latch. A lost-enqueue keeps the
    /// latch held (no bump); a crash evicts `CrossingProgress` → resets to `0` (the restored in-band dot
    /// re-mints the same first-attempt id — idempotent). Not reset on a winner change (per-entity, like the
    /// cooldown).
    pub crossing_attempt: u32,
    /// The re-emit payload for the durable latch this state armed (3f-D4). `Some` iff a durable
    /// `RequestInFlight` latch is currently held for this subject — set alongside the latch in
    /// `fan_out_crossing`'s durable `Vacant` arm, read by `redrive_stranded_crossings` to re-mint the
    /// SAME `CrossingRequest`. Left `None` on a transient crossing (no latch) and cleared by the
    /// pre-CAS abort ([`abort_crossing_latch`] — wire OR local-exhaustion). `Option` keeps
    /// `LatchedCrossing` (a non-`Default` payload) off the `Default` path.
    pub latched_crossing: Option<LatchedCrossing>,
    /// D-WORLD-2 cure — the ttl re-drives ALREADY SPENT on the currently-held latch. Reset to `0`
    /// when a new latch is taken (`fan_out_crossing`'s `Vacant` arm) and by the pre-CAS abort clear;
    /// incremented per re-emit by `redrive_stranded_crossings`. Once it reaches
    /// `StubConfig::crossing_redrive_budget` the next ttl expiry takes the LOCAL exhaustion abort
    /// instead of another re-emit. Meaningful only while the latch is held.
    pub redrives_spent: u32,
}

/// The containment trigger's per-entity cooldown/latch state (task #135). One entry per evaluated
/// subject; lazily evicted when the subject leaves the shard.
#[derive(Resource, Debug, Default)]
pub struct CrossingProgress(pub BTreeMap<EntityId, CrossingState>);

/// Per-entity HYSTERETIC containment membership: THE SET OF REALMS this occupant is currently a member
/// of (task #135 §2.3).
///
/// **This was a fixed-width `u64` — one bit per watched region — and that single word is what capped a
/// parent at 64 children (SL9, retired 2026-08-24).** It is now the short list it always described. The
/// set holds only TRUE members, so its size is the occupant's own chain (about six: the realm it stands
/// in and its ancestors) plus, briefly, the sibling whose release edge it has not yet left during a
/// crossing. It does NOT grow with the parent's breadth: a galaxy watching a hundred and fifty thousand
/// star systems stores the same handful per occupant as a planet watching one moon, because an occupant
/// can only be inside the ones that actually contain it.
///
/// PER-ENTITY, so there is still zero cross-entity contention (the `par_iter` precondition the `Copy`
/// word was chosen for is preserved — only the "one word" part is gone). An EMPTY set allocates nothing,
/// which is the state of every occupant this shard does not host. Each realm's membership advances
/// INDEPENDENTLY by its own [`vd_core::geometry::ContainmentBand`] — there is no single-winner slot to
/// corrupt (the old `winner_ix` reset was actively wrong here).
#[derive(Clone, Debug, Default, PartialEq)]
pub struct RegionMembership {
    realms: BTreeSet<RealmId>,
}

impl RegionMembership {
    /// The realms this subject was a member of at the last evaluation — re-asked to decide release.
    fn members(&self) -> impl Iterator<Item = RealmId> + '_ {
        self.realms.iter().copied()
    }

    /// Is the entity a hysteretic member of `realm`?
    fn get(&self, realm: RealmId) -> bool {
        self.realms.contains(&realm)
    }

    /// Record (or clear) membership of `realm`. Clearing REMOVES, so the set only ever holds true
    /// members and stays the size of the occupant's own chain however wide the parent is.
    pub(crate) fn set(&mut self, realm: RealmId, member: bool) {
        if member {
            self.realms.insert(realm);
        } else {
            self.realms.remove(&realm);
        }
    }
}

/// Per-entity containment-membership bitsets (task #135). Keyed by the subject entity; lazily evicted
/// with the other per-entity ledgers when the subject leaves.
#[derive(Resource, Debug, Default)]
pub struct ContainmentProgress(pub BTreeMap<EntityId, RegionMembership>);

/// The per-entity DURABLE-crossing latch (Slice 3d): a durable entity that has emitted a
/// `CrossingRequest` is latched here (keyed by subject entity → the deterministic
/// [`crossing_transfer_id`]) so the trigger emits EXACTLY ONE request per crossing. Cleared POSITIVELY
/// by the saga terminal — `on_saga_demote` on a durable COMMIT, or `CrossingAborted` on a pre-CAS abort
/// (the 3f abort egress) — and, since the D-WORLD-2 cure, by the SOURCE-LOCAL exhaustion abort: a
/// request whose dest never resolves (no saga ever starts, so no terminal can ever arrive) is
/// re-driven `crossing_redrive_budget` times at the `request_ttl_ticks` cadence and then aborted
/// locally through the SAME pre-CAS clear ([`abort_crossing_latch`]), so an unhosted-dest graze can
/// never suppress this entity's crossings forever. Lazily evicted when the subject leaves.
#[derive(Resource, Debug, Default)]
pub struct RequestInFlight(pub BTreeMap<EntityId, TransferId>);

/// THE ONE PRE-CAS CROSSING-ABORT LATCH CLEAR (Slice 3f-D + the D-WORLD-2 cure) — the body shared by
/// the wire `CrossingAborted` consumer ([`on_crossing_aborted`], a saga that STARTED and aborted) and
/// the SOURCE-LOCAL exhaustion abort ([`redrive_stranded_crossings`], a request that never started a
/// saga because its dest never resolved). It clears the `RequestInFlight` latch, closes any source
/// hand-off hold (an abort is pre-CAS, so a hold from THIS transfer cannot exist — holds open at the
/// post-CAS demote; the only hold this could find is a NEWER committed crossing's, and both callers
/// gate against that: the wire path by the exact id match, the local path by holding the live latch),
/// BUMPS the attempt (H2 — the re-latch mints a FRESH [`crossing_transfer_id`], so a stale
/// wire abort arriving later is the counted `crossing_abort_stale` no-op, never a wrong-clear),
/// drops the re-emit payload + the spent-re-drive count, and sets the containment cooldown to
/// `cooldown`: `None` re-fires the re-home NEXT tick (the wire abort — the container still differs,
/// audit-L1); `Some(now)` arms the EXISTING `should_rehome` `k_dwell` dwell (the exhaustion abort —
/// a dot PARKED inside an unresolved region must re-fire at a bounded cadence, never per-tick).
/// `entry().or_default()` (not `if let`) so there is no uncoverable `None` region — a latched entity
/// always has a `CrossingState`, but a default is harmless if absent. NOT a drop: dropping would
/// reset the attempt to 0 → id aliasing. Monomorphic, straight-line (HR5).
fn abort_crossing_latch(
    entity: EntityId,
    cooldown: Option<TickId>,
    in_flight: &mut RequestInFlight,
    progress: &mut CrossingProgress,
    holds: &mut HandoffHolds,
    stats: &mut StubStats,
) {
    close_hold(holds, (entity, HoldRole::Source));
    in_flight.0.remove(&entity);
    let st = progress.0.entry(entity).or_default();
    st.crossing_attempt = st.crossing_attempt.saturating_add(1);
    st.latched_crossing = None;
    st.redrives_spent = 0;
    st.last_commit_tick = cooldown;
    stats.crossing_latches_cleared += 1;
}

/// SOURCE consumer of the orchestrator's `CrossingAborted` (Slice 3f-D): the crossing resolve/start saga
/// aborted pre-CAS, so CLEAR the subject's `RequestInFlight` latch (the positive re-cross signal) — but
/// ONLY if it still holds THIS aborted transfer id (a stale abort for a superseded /
/// re-latched transfer must not free a live crossing; the per-attempt id makes that exact).
///
/// ALWAYS acks `CrossingAbortedAck` (all three paths): the orchestrator keeps its durable
/// `pending_abort_replies` entry alive — re-emitting `CrossingAborted` every `scan_deadlines` window,
/// crash-durable via the store — until THIS ack drops it. So a lost first ack whose re-emit finds the latch
/// already cleared must STILL re-ack, else the ORCHESTRATOR entry leaks (the ack-gate must not merely
/// relocate the strand). On the id-match clear it also RE-ARMS: reset ONLY the dwell (so a still-in-band
/// entity re-requests without physically re-crossing — audit-L1) + BUMP the attempt (so the re-latch mints a
/// FRESH id — H2). Monomorphic.
#[allow(clippy::too_many_arguments)]
pub(crate) fn on_crossing_aborted(
    abort: CrossingAborted,
    in_flight: &mut RequestInFlight,
    progress: &mut CrossingProgress,
    holds: &mut HandoffHolds,
    driven: &mut crate::stub::drive::DrivenChildren,
    stats: &mut StubStats,
    outbox: &mut OutboundBox,
    orchestrator: NodeId,
) {
    // UNCONDITIONAL ack (all paths, before the no-entity early-return). `abort` is `Copy`, so it also drives
    // the id-match below.
    outbox.push_flow(
        orchestrator,
        MsgClass::Saga,
        &InterShardFlow::CrossingAbortedAck(abort),
    );
    // The ruler switch, slice 2: an aborted EXTERIOR crossing thaws the child the flush froze — it is
    // this realm's to drive again — and clears its latch like any subject's.
    let entity = match abort.subject {
        DirectoryKey::Ship(entity) => {
            if let Some(d) = driven.0.get_mut(&RealmId::Ship(entity)) {
                d.frozen = false;
            }
            entity
        }
        other => {
            let Some(entity) = other.transfer_subject_entity() else {
                stats.crossing_abort_no_entity += 1;
                return;
            };
            entity
        }
    };
    // Clear ONLY on an exact id match (`== Some(&abort.transfer)`) — equality over the value so the
    // false arm is a covered no-op, not an uncoverable `matches!` region (HR5(d)). The clear itself
    // is THE shared pre-CAS abort body ([`abort_crossing_latch`] — hold close under the match, latch
    // clear, attempt bump). `cooldown: None` (audit-L1): the container still differs from the owner
    // (the entity did not move), so `should_rehome` re-fires NEXT tick with a fresh id.
    if in_flight.0.get(&entity) == Some(&abort.transfer) {
        abort_crossing_latch(entity, None, in_flight, progress, holds, stats);
    } else {
        stats.crossing_abort_stale += 1;
    }
}

/// Slice 3e — THE per-shard geometric transfer-TRIGGER: evaluate every owned dot + held transient
/// against the realm REGIONS this shard carries (task #135): compute the DEEPEST region CONTAINING each
/// owned subject (`container`) and, when that differs from the realm this shard owns it in, fan out ONE
/// authority re-home. SYMMETRIC + direction-free — escaping a realm and entering one are the same rule.
/// INERT in production through C-3: `RealmRegions` is empty (the seed boot-population is C-5/C-6), so it
/// early-returns (behaviour-identical); its logic is exercised only by the sim integration tests.
///
/// A BRANCHLESS SHIM (HR5): the system body is iterate → delegate; ALL branching lives in the monomorphic
/// helpers [`evaluate_one_subject`] / [`fan_out_crossing`] / [`retain_live`]. The geometry lives in
/// `vd_core::geometry` (`container` / `region_verdict` / `should_rehome` / `ContainmentBand`).
#[allow(clippy::too_many_arguments)]
pub(crate) fn evaluate_realm_boundaries(
    config: Res<StubConfig>,
    clock: Res<ClockSample>,
    authority: Res<RealmAuthority>,
    regions: Res<RealmRegions>,
    placements: Res<Placements>,
    mut dots: ResMut<Dots>,
    mut owned_transients: ResMut<OwnedTransients>,
    mut progress: ResMut<CrossingProgress>,
    mut membership: ResMut<ContainmentProgress>,
    mut in_flight: ResMut<RequestInFlight>,
    mut stats: ResMut<StubStats>,
    mut outbox: ResMut<OutboundBox>,
    // The ruler switch, slice 1 — the DRIVEN CHILDREN this realm authors are subjects too.
    driven: Res<crate::stub::drive::DrivenChildren>,
    exterior: Res<crate::stub::realm_head::ExteriorAuthority>,
    mut exterior_scan: ResMut<ExteriorScan>,
) {
    // Authority gate (mirrors `emit_transient_batch`): a shard without its realm lease triggers no
    // re-home — the `realm_fence` is the durable `subject_fence` and the transient `src_realm_fence`.
    let Some(realm_fence) = authority.0 else {
        return;
    };
    // Inert until regions are planted (production through C-3): nothing to evaluate containment against.
    if regions.is_empty() {
        return;
    }
    // EVICTION (lazy retain, DRY single site): drop per-entity state for any subject no longer resident,
    // so the maps never leak. The live set is the union of owned-dot entities and held-transient keys.
    let live: BTreeSet<EntityId> = dots
        .0
        .values()
        .map(|d| d.entity)
        .chain(owned_transients.0.keys().copied())
        .chain(
            driven
                .0
                .keys()
                .filter_map(|r| DirectoryKey::exterior_entity(*r)),
        )
        .collect();
    retain_live(&mut progress.0, &live);
    retain_live(&mut membership.0, &live);
    retain_live(&mut in_flight.0, &live);
    retain_live(&mut exterior_scan.0, &live);

    let ctx = CrossingCtx {
        config: &config,
        clock: &clock,
        regions: &regions.regions,
        depths: &regions.depths,
        ancestor_chain: &regions.ancestor_chain,
        child_index: regions.child_index(),
        own_frame: regions.own_frame(config.realm),
        ix_of: &regions.ix_of,
        unindexed: regions.unindexed_rows(),
        root_realm: regions.root_realm,
        realm_fence,
    };
    // Per owned dot (only those this shard SIMULATES) — the durable subjects. `.iter_mut()` (not
    // `.values_mut()`) so the `SessionId` key is in scope: a durable crossing carries the subject's
    // session for the saga's gateway-routed `PrepareSubscribe` (Slice 3f).
    for (session, dot) in dots.0.iter_mut().filter(|(_, d)| d.authority.simulates()) {
        let pose = dot.pose;
        debug_assert_nonlatched_stamp_is_current(
            &in_flight.0,
            dot.entity,
            pose.universe_tick,
            clock.universe_tick,
        );
        // A DURABLE dot reads the HEAD book — the world of NOW. For a non-latched dot the head's
        // instant IS the pose's stamp (the `readvance_dots` guarantee pinned above), bit-identical to
        // the old per-stamp solve; a LATCHED dot's frozen position is measured against the current
        // world, the same one-instant rule the flush follows (its scan outputs are suppressed by the
        // standing latch either way).
        let anchor = book_anchor(&regions.ix_of, config.realm, owning_realm(pose.frame));
        let Some(book) = placements.0.head(anchor) else {
            // No book for a subject's anchor is a writer gap, never a silent skip: counted, and the
            // subject is simply not evaluated this tick (the next authored tick picks it up).
            stats.placement_book_miss += 1;
            continue;
        };
        let eval = evaluate_one_subject(
            &ctx,
            book,
            dot.entity,
            &pose,
            dot.prev_offset,
            dot.authority.fence(),
            SubjectLane::ByKind(Some(*session)),
            None,
            &mut progress.0,
            &mut membership.0,
            &mut in_flight.0,
            &mut stats,
            &mut outbox,
        );
        dot.prev_offset = eval.prev_offset;
    }
    // Per HELD transient (`is_held()` — the counted tier; Arriving/Departing are excluded) — the
    // transient subjects. The transient carries no per-entity authority fence; its egress rides the
    // shard's `src_realm_fence`, so the helper's `subject_fence` argument is the realm fence.
    for (entity, t) in owned_transients
        .0
        .iter_mut()
        .filter(|(_, t)| t.status.is_held())
    {
        let pose = t.pose;
        // A HELD TRANSIENT reads the book at ITS OWN stamp (row 18 of the instant table): it
        // re-advances AFTER this scan, so its stamp is the previously authored instant — inside the
        // window by construction (the `+1` in `placement_window_ticks`). Bit-identical to the old
        // per-stamp solve. Do NOT reorder the re-advance ahead of this scan in this arc.
        let anchor = book_anchor(&regions.ix_of, config.realm, owning_realm(pose.frame));
        let book = match placements.0.at(anchor, pose.universe_tick) {
            Ok(b) => b,
            Err(_) => {
                stats.placement_book_miss += 1;
                continue;
            }
        };
        let eval = evaluate_one_subject(
            &ctx,
            book,
            *entity,
            &pose,
            t.prev_offset,
            realm_fence,
            // A transient carries no session (its batch handoff emits no session-bearing command); it
            // dispatches to the Transient arm, which never reads this. Slice 3f.
            SubjectLane::ByKind(None),
            None,
            &mut progress.0,
            &mut membership.0,
            &mut in_flight.0,
            &mut stats,
            &mut outbox,
        );
        t.prev_offset = eval.prev_offset;
    }
    // ★ THE THIRD LANE — MY DRIVEN CHILDREN (the ruler switch, slice 1; owner 2026-09-02). A hull is a
    // realm the parent authors, and until now it was only an OBJECT the scan measured against, never a
    // SUBJECT it measured: a hull left its star system's shell and nobody noticed (MEASURED: two
    // flights to the end of the millimetre ruler). Now each driven child is asked the same question
    // as a dot, on the same swept lookup and the same container fold: which of my children holds it,
    // or has it left me altogether. Its ledger key is the entity its exterior is named by, so the
    // latch, the membership and the cooldown are the ones every subject has.
    //
    // THE EARLY START (M-D; D-MOVE-3 piece 2): a saga needs ticks and a shell is crossed inside one
    // at the speeds the owner flies, so the point asked about is the position LED by the velocity the
    // parent itself authored, over the request ttl — the saga's own expected duration, derived, never
    // a literal. The pose the source ships at the flush is the truth at the flush tick, so an early
    // request never plants the hull where it is not.
    //
    // Nothing here names the child's motion, drive or kind (SL4): the state read is the placement
    // the physics pass produced, and the lane is the one every subject takes.
    let own_frame = regions.own_frame(config.realm);
    let lead_s = f64::from(config.request_ttl_ticks) * config.tick_dt_s;
    for (child, held) in driven.0.iter() {
        let Some(entity) = DirectoryKey::exterior_entity(*child) else {
            continue; // a driven child with no exterior key has no author to move: never a subject
        };
        let Some(exterior_fence) = exterior.0.get(child).copied() else {
            stats.exterior_scan_unleased += 1;
            continue;
        };
        let Some(region) = regions.direct_child(config.realm, *child) else {
            continue; // authored but not on my roster this tick: nothing to measure against
        };
        let pose = exterior_pose(region, &held.state, own_frame, clock.universe_tick);
        let led = led_pose(&pose, lead_s);
        let prev = exterior_scan.0.get(&entity).copied().unwrap_or(led.pos);
        let anchor = book_anchor(&regions.ix_of, config.realm, config.realm);
        let Some(book) = placements.0.head(anchor) else {
            stats.placement_book_miss += 1;
            continue;
        };
        let eval = evaluate_one_subject(
            &ctx,
            book,
            entity,
            &led,
            prev,
            exterior_fence,
            SubjectLane::Exterior,
            // A child is always inside its own bound; the question is which OTHER region holds it.
            Some(*child),
            &mut progress.0,
            &mut membership.0,
            &mut in_flight.0,
            &mut stats,
            &mut outbox,
        );
        exterior_scan.0.insert(entity, eval.prev_offset);
    }
}

/// The ruler switch, slice 1 — where each driven child's LED point was at the previous scan, keyed by
/// the entity its exterior is named by (the swept prior every subject carries; a dot keeps its own on
/// the `Dot`). Evicted with the live set like every per-subject ledger.
#[derive(Resource, Debug, Default)]
pub struct ExteriorScan(pub BTreeMap<EntityId, LatticePos>);

/// A driven child's authored placement as a stamped pose in the PARENT's own frame — the same row
/// `placement_row` writes, read as the point the scan measures: the berth cell plus the travel the
/// physics pass produced, the velocity the parent wrote, the facing it wrote.
pub(crate) fn exterior_pose(
    region: &vd_core::geometry::RealmRegion,
    state: &crate::stub::drive::DrivenState,
    frame: FrameRef,
    at: vd_core::ids::UniverseTick,
) -> StampedPose {
    StampedPose {
        frame,
        pos: region
            .center
            .in_parents_frame()
            .translated(state.pos_m, frame.tier()),
        vel: state.vel_mps,
        orient: state.orient,
        universe_tick: at,
    }
}

/// The pose LED by its own velocity over `lead_s` seconds — the early-start point (M-D). A zero
/// velocity or a zero lead is the pose itself, bit for bit.
fn led_pose(pose: &StampedPose, lead_s: f64) -> StampedPose {
    StampedPose {
        pos: pose.pos.translated(pose.vel * lead_s, pose.frame.tier()),
        ..*pose
    }
}

/// Which arm of the fan-out a subject takes — chosen by the LANE that scanned it, never by a kind
/// test alone: a dot and a transient dispatch on their entity's durability class as before; a driven
/// child's exterior is its own policy (HR2: policy fan-out on one machinery).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum SubjectLane {
    ByKind(Option<SessionId>),
    Exterior,
}

/// Read-only per-tick context shared by every subject evaluation (task #135).
struct CrossingCtx<'a> {
    config: &'a StubConfig,
    clock: &'a ClockSample,
    /// The shard's realm REGIONS (own realm + ancestor chain + a bounded child set).
    regions: &'a [RealmRegion],
    /// The boot-computed depth key per region (index-aligned to `regions`) — the `container` fold reads
    /// these, never re-walking parents (O(entities × regions), not O(N·M²)).
    depths: &'a [DepthKey],
    /// The ambient-ROOT realm (the `parent: None` region) — the `container` fold identity. Always `Some`
    /// here (the system early-returns on an empty registry before building the ctx).
    root_realm: Option<RealmId>,
    /// The realm-authority fence — the durable subject's `subject_fence` fallback AND the transient
    /// `src_realm_fence`.
    realm_fence: Fence,

    /// The boot-computed SELF ∪ ANCESTORS realm SET per region index, and the realm→index map to address
    /// it (task #177). Together they give the DERIVED hysteresis prior: a subject is always a member of
    /// the realm it is authoritatively in and of every ancestor of it, so an arriving shard never starts
    /// from a blank memory that contradicts the one the source held. Sized by DEPTH, never by the
    /// parent's breadth (SL9).
    ancestor_chain: &'a [BTreeSet<RealmId>],
    ix_of: &'a BTreeMap<RealmId, usize>,
    /// The rows the child index does not answer for (`RealmRegions::unindexed_rows`).
    unindexed: &'a [RealmId],
    /// THE CHILD INDEX (SL9): which of this shard's static direct children could hold a given point, so
    /// the fold below asks a handful instead of the whole forest. EMPTY on a rig that never named its
    /// own realm, in which case nothing is indexed, nothing is skipped, and the fold is the full scan it
    /// has always been.
    child_index: &'a ChildIndex,
    /// THIS SHARD'S OWN frame — the one the child index is built in, so a subject is expressed there
    /// once per tick instead of once per child.
    own_frame: FrameRef,
}

/// Is this region worth evaluating for a subject whose candidate set is `candidates`?
///
/// YES unless ALL of these hold: the index actually answers for this realm (so a miss is INFORMATION,
/// not ignorance), the lookup did not name it, the subject is not already a remembered member of it, and
/// it is not on the subject's own derived chain. The last two matter because membership is HYSTERETIC —
/// a region the subject is inside must be re-asked every tick to decide RELEASE, however far the
/// geometry says it now is. (The index ABSTAINED on a wide step while it was a grid; the tree answers
/// every span, so that arm is gone.)
///
/// Monomorphic and written as four named terms rather than one chained `&&`, so every arm is coverable
/// on its own (HR5) and so the reason a region survived the filter can be read off the code.
pub(crate) fn worth_asking(
    index: &ChildIndex,
    candidates: &[RealmId],
    remembered: &RegionMembership,
    chain: &BTreeSet<RealmId>,
    realm: RealmId,
) -> bool {
    if !index.answers_for(realm) {
        return true; // not indexed (an ancestor, this realm itself, a mover) — always evaluated
    }
    if candidates.contains(&realm) {
        return true; // the lookup named it
    }
    if remembered.get(realm) {
        return true; // already a member — must be re-asked to decide release
    }
    chain.contains(&realm) // on the derived prior's chain — likewise
}

/// The prior endpoint expressed in THIS SHARD'S OWN frame, for the candidate lookup. `None` for "no
/// usable prior" — the same two cases the verdict's own prior recognises, decided the same way.
///
/// The degeneracy test is on CELLS, not on the whole position: `LatticePos` compares an f64 offset, so
/// a whole-position equality would be a float comparison, and a subject that has not moved would then
/// depend on a float comparing equal to itself.
fn prior_in_own_frame(
    pose: &StampedPose,
    prev_pos: LatticePos,
    own_frame: FrameRef,
    book: &PlacementBook,
) -> Option<LatticePos> {
    if prev_pos.cell() == pose.pos.cell() {
        return None;
    }
    let prev = StampedPose {
        pos: prev_pos,
        ..*pose
    }
    .sanitized();
    vd_core::frame::transfer_frame(&prev, own_frame, book)
        .ok()
        .map(|p| p.pos)
}

/// The DERIVED hysteresis prior for a subject authoritatively in `owning` — SELF ∪ ANCESTORS as a set of
/// realms, or EMPTY for a realm this shard does not host. Monomorphic so its branches are covered once.
fn owned_prior_chain<'a>(
    ix_of: &BTreeMap<RealmId, usize>,
    chains: &'a [BTreeSet<RealmId>],
    owning: RealmId,
) -> &'a BTreeSet<RealmId> {
    const EMPTY: &BTreeSet<RealmId> = &BTreeSet::new();
    ix_of
        .get(&owning)
        .and_then(|ix| chains.get(*ix))
        .unwrap_or(EMPTY)
}

/// Retain only the entries whose key is a LIVE subject (the DRY eviction primitive, Slice 3e; RLM Step 2
/// widened over `K` for `AoiMembership`'s `RealmPath` keys). A branchless `retain` over any
/// `BTreeMap<K, V>` — no per-monomorphization branch trap (HR5).
pub(crate) fn retain_live<K: Ord, V>(map: &mut BTreeMap<K, V>, live: &BTreeSet<K>) {
    map.retain(|k, _| live.contains(k));
}

/// S0 of the placement arc — the Stage-B4 guarantee the instant table stands on, PINNED where the
/// detector reads a dot: a non-latched simulating dot was re-stamped THIS tick by `readvance_dots`
/// (group A, strictly before the detector), so the scan measures it against the world of NOW. A
/// LATCHED dot (an in-flight crossing) is exempt: its stamp is deliberately frozen (the
/// up-observation ride measurement). A monomorphic helper so the detector stays a branchless shim
/// and every arm of the invariant — the latch exemption, the holding stamp, and the panic — is
/// coverable by name (HR5); debug-only, like every stated invariant.
pub(crate) fn debug_assert_nonlatched_stamp_is_current(
    in_flight: &BTreeMap<EntityId, TransferId>,
    entity: EntityId,
    stamp: UniverseTick,
    now: UniverseTick,
) {
    debug_assert!(
        in_flight.contains_key(&entity) || stamp == now,
        "a non-latched simulating dot's stamp must equal the clock at the detector",
    );
}

/// The result of one subject's containment evaluation — the frame-local offset the caller records as
/// `prev_offset`. (Every re-home is now the uniform orchestrator saga — source==dest is the degenerate
/// case — so the detector emits a `CrossingRequest`/`TransientCrossingRequest` and NEVER rewrites the pose
/// in place; the crossed pose is placed at the DEST's adopt via `place_arriving_pose`, which is the same
/// node's adopt on a co-hosted re-home.)
struct SubjectEval {
    prev_offset: LatticePos,
}

/// The subject's OWNING realm, derived from its POSE FRAME (not a co-hosting relabel — that machinery is
/// deleted; every re-home is now the uniform orchestrator saga).
///
/// ★ IT NO LONGER TAKES A FALLBACK, AND THE FALLBACK IS WHY IT EXISTED. This used to be
/// `frame.realm().unwrap_or(config_realm)`, wrapped in its own function precisely so the `None` arm could
/// be driven by a direct unit test — its own doc said the fallback was *"otherwise uncoverable, no live
/// shard uses GalaxySpace"*. That was the whole justification, and it rested on galaxy space naming no
/// realm.
///
/// A galaxy names one now, so every frame does, and there is nothing left to fall back FROM. Keeping the
/// shim would mean keeping a second realm-of-a-frame answer that could disagree with the first.
#[must_use]
pub(crate) fn owning_realm(frame: FrameRef) -> RealmId {
    frame.realm()
}

/// WHO is outside this shard — a fact of the shard's OWN LINEAGE, never a search of the world. An occupant
/// that has left everything this shard holds goes to this shard's parent, one level up.
///
/// Two sources, asked in order, because they answer the same question with different reliability:
///   1. the shard's own coordinate, which carries its full lineage — when whoever launched it set one;
///   2. its own row in the seed-derived region list, which always carries its parent.
///
/// The second is NOT a convenience for tests. The shipped launchers emit no lineage at all, so a shard
/// started any way other than by the demand spawner boots with a root-shaped identity whose parent is
/// empty — and every occupant leaving it was thrown all the way to the ambient root instead of one level
/// up. The same shard's own region row named its true parent the whole time; nothing read it.
///
/// Only the TRUE root has no parent in either source, and only there is the ambient root the honest
/// answer. Note this reads the shard's own ANCESTRY, never a position: it learns whose child it is, which
/// is what it must know to hand an occupant upward, and still never learns where it itself sits.
///
/// AND IT IS ASKED ABOUT THE OCCUPANT'S REALM, not the shard's. You fall out of the realm you are STANDING
/// IN — which on a shard co-hosting a chain (System ⊃ Planet ⊃ Area) is often not the realm the shard is
/// named after. Asking the shard's lineage there would throw somebody leaving a co-hosted area past the
/// planet it sits on and all the way to the star's parent. For the shard's own realm the two sources are
/// consulted in their original order (own coord first, roster second), so a single-realm shard behaves
/// exactly as before.
pub(crate) fn outward_dest(
    config: &StubConfig,
    regions: &[RealmRegion],
    root_realm: RealmId,
    owning: RealmId,
) -> RealmId {
    let roster_parent = |realm: RealmId| {
        regions
            .iter()
            .find(|r| r.realm == realm)
            .and_then(|r| r.parent)
    };
    if owning != config.realm
        && let Some(parent) = roster_parent(owning)
    {
        return parent;
    }
    if let Some(parent) = config.own_coord.parent() {
        return parent.lowered();
    }
    if let Some(parent) = roster_parent(config.realm) {
        return parent;
    }
    root_realm
}

/// The FULL per-subject CONTAINMENT evaluation (task #135), monomorphic so every branch is covered ONCE
/// here (HR5). Full-scan the shard's regions: advance each region's per-entity hysteretic membership bit
/// (its own [`vd_core::geometry::ContainmentBand`] over `signed_distance`), fold the members into the
/// DEEPEST containing realm ([`container`], total by construction — no `None`), and — when that differs
/// from the realm this shard owns the subject in AND the post-commit cooldown elapsed ([`should_rehome`])
/// — fan ONE re-home out by the subject's `DurabilityClass`. Returns `cur` for the caller's `prev_offset`.
///
/// UNIFORM re-home (the un-hosted-child cure, task #149): there is NO node-placement short-circuit. Whether
/// `head(Realm(dest))` resolves to a FOREIGN node or to THIS node (source==dest, a co-hosted child), the
/// detector emits the SAME `CrossingRequest`/`TransientCrossingRequest` and the ONE orchestrator saga
/// carries it — a same-node saga completes post-S3 (the gateway self-acks the cut, the CAS bumps the fence,
/// the route swap is an idempotent no-op). So co-hosting is now PURELY a placement (the grant/affirm that
/// makes `head(Realm(child))` resolve here); the crossed pose is rebound at the DEST's adopt (same node on a
/// co-hosted re-home), never rewritten in place. `to_parent` (the container region's `parent`) rides the
/// request so an `Area` dest's frame forms.
///
/// SYMMETRIC + direction-free: escaping a realm (System→Galaxy) and entering one (Galaxy→System) are the
/// identical path — the container simply changed. The per-region band hysteresis IS the anti-flap dwell.
#[allow(clippy::too_many_arguments)]
fn evaluate_one_subject(
    ctx: &CrossingCtx<'_>,
    // The authored book this subject is measured through — selected by the CALLER per lane (head for
    // a durable dot, the pose's own instant for a held transient), never by a clock in here (SL4).
    book: &PlacementBook,
    entity: EntityId,
    pose: &StampedPose,
    // WHERE THIS SUBJECT WAS AT THE PREVIOUS EVALUATION, in the same frame as `pose`. Every membership
    // question below is asked about the SEGMENT from here to `pose.pos`, not about the endpoint alone,
    // so a subject that travels through a child within one tick is still seen. Equal to `pose.pos`
    // means "no prior", which is what every construction site seeds and what the first tick after a
    // spawn, an adopt or an arrival carries; the verdict then collapses to the point answer exactly.
    prev_pos: LatticePos,
    subject_fence: Fence,
    // The lane that scanned this subject: a dot carries its session (the durable `CrossingRequest`
    // carries it so the orchestrator's saga can `PrepareSubscribe` to the client's gateway), a
    // transient none, and a driven child's exterior is its own arm. Slice 3f; the ruler switch, slice 1.
    lane: SubjectLane,
    // A region the fold must NOT ask about: a driven child's OWN region, which always holds its own
    // centre and would otherwise be its own deepest container. `None` for a dot.
    exclude: Option<RealmId>,
    progress: &mut BTreeMap<EntityId, CrossingState>,
    membership: &mut BTreeMap<EntityId, RegionMembership>,
    in_flight: &mut BTreeMap<EntityId, TransferId>,
    stats: &mut StubStats,
    outbox: &mut OutboundBox,
) -> SubjectEval {
    let cur = pose.pos;
    // The ambient root seeds the `container` fold (always `Some` — the system gated on a non-empty
    // registry before building the ctx; the `let-else` keeps this branchless of an `unwrap`).
    let Some(root_realm) = ctx.root_realm else {
        return SubjectEval { prev_offset: cur };
    };
    // FULL SCAN: advance each region's hysteretic membership bit and collect the members' depth keys.
    // Per-region + per-entity (the bitset) → each region's bit advances INDEPENDENTLY (no winner slot to
    // corrupt). Zips regions with their boot-computed depth keys (no per-tick parent walk, no index panic).
    // The subject's OWNING realm from its POSE FRAME, hoisted ABOVE the scan because the derived prior
    // below needs it. (`owning_realm` is unit-tested on both arms; on a single-realm shard the pose frame
    // IS `config.frame`.)
    let owning = owning_realm(pose.frame);
    // THE DERIVED HYSTERESIS PRIOR (task #177). A subject is ALWAYS a hysteretic member of the realm it is
    // authoritatively in and of every ancestor of it — regardless of what this shard happens to remember.
    //
    // This is what stops the boundary re-firing. The stored bitset is per-shard and starts BLANK on an
    // arriving shard, so after a hand-off the two sides held OPPOSITE priors for the same subject at the
    // same place: the source thought "inside, stay inside", the destination thought "outside, must acquire".
    // Anything resting in the band between the acquire and release edges therefore oscillated forever. OR-ing
    // the derived prior in makes the answer a total function of COMMITTED state, so it is identical on both
    // sides of a hand-off and survives a crash, a replay, a re-drive and a lease change — and, unlike the
    // RAM-only bitset, a shard restart no longer blanks every resident at once.
    //
    // It is an OR, never a replacement: the stored bit still carries genuine hysteresis for regions BELOW the
    // owning realm (a child you have entered but not yet re-homed into), which ancestry cannot express.
    let owned_chain = owned_prior_chain(ctx.ix_of, ctx.ancestor_chain, owning);
    if owned_chain.is_empty() {
        // The realm named by the pose frame is not one this shard hosts, so there is no ancestry to derive
        // and we fall back to today's stored-only prior. That is a SAFE degrade, not a normal path — it means
        // a rebind upstream handed us a pose in a frame we cannot place — so it is counted and said out loud
        // rather than silently reverting to the behaviour this slice exists to remove.
        stats.containment_prior_unhosted += 1;
        tracing::warn!(
            realm = ?owning,
            entity = entity.0,
            "containment prior: the pose frame names a realm this shard does not host — \
             falling back to the stored-only prior, which is the pre-fix behaviour for this subject"
        );
    }
    // THE MEASUREMENT POSE: the subject's position, stamp-aligned to the book's own instant. For a
    // non-latched dot and a held transient the two instants are already equal (the caller's selection),
    // so this is the identity; for a LATCHED dot (frozen stamp, head book) it is the deliberate
    // one-instant rule — the frozen position measured against the world of NOW.
    let measured = StampedPose {
        universe_tick: book.at(),
        ..*pose
    };
    let bits = membership.entry(entity).or_default();
    // THE CANDIDATE LOOKUP (SL9), ONCE per subject rather than once per child. The index is built in
    // this shard's OWN frame, so the subject is expressed there once and asked once. A subject this
    // shard cannot place in its own frame yields NO candidates — and because `worth_asking` only ever
    // skips a realm the index positively answers for, an empty answer for an indexed child is a real
    // "not here", while an unbuildable point simply leaves every child unindexed-and-evaluated below.
    // ★ THE POSITION STAYS ON THE LATTICE (slice S4). It used to be flattened from the frame origin
    // before the lookup, and at galaxy magnitudes that flatten rounds by a quarter of a metre — over a
    // hundred times the whole containment band. A key built from a rounded position can name a cell the
    // child was never registered in, and this index's answer is trusted as a positive "not here", so
    // the caller would then SKIP the one child that actually holds the point.
    // ★ THE LOOKUP ASKS ABOUT THE SEGMENT, NOT THE POINT (slice S5). A child the subject travels
    // THROUGH within one tick holds neither endpoint, so a point lookup would skip it before the swept
    // verdict could ever be asked — the arithmetic would be correct and unreachable. A segment of zero
    // length IS the point answer, so a stationary subject pays exactly today's lookup. The tree answers
    // every span (the grid it replaced abstained on a wide one), so the answer is always information.
    let own_point = vd_core::frame::transfer_frame(&measured, ctx.own_frame, book).map(|p| p.pos);
    let own_prev = prior_in_own_frame(&measured, prev_pos, ctx.own_frame, book);
    let mut seg_candidates: Vec<RealmId> = Vec::new();
    let candidates: &[RealmId] = match (&own_point, &own_prev) {
        (Ok(p), Some(q)) => {
            ctx.child_index
                .candidates_segment(*q, *p, ctx.own_frame.tier(), &mut seg_candidates);
            &seg_candidates
        }
        // No usable prior: the point lookup, unchanged.
        (Ok(p), None) => {
            ctx.child_index
                .candidates_into(*p, ctx.own_frame.tier(), &mut seg_candidates);
            &seg_candidates
        }
        // A subject this shard cannot place in its own frame: no candidates, and every indexed child
        // therefore falls through to the unconditional evaluation below.
        // A cross-unit refusal lands here too, and it is COUNTED rather than swallowed: the answer
        // (no candidates, so every indexed child is evaluated unconditionally) is conservative and
        // stays exactly as it was, but the reason is no longer invisible.
        (Err(e), _) => {
            stats.cross_tier_refused += u64::from(matches!(
                e,
                vd_core::frame::FrameError::CrossTierCrossing(_)
            ));
            &[]
        }
    };
    let mut members: Vec<DepthKey> = Vec::new();
    // ★ ASK ONLY THE ROWS THAT CAN ANSWER (SL9, MEASURED on the fifth flight, 2026-09-03): the rows
    // the index does not answer for (the ancestors, this realm, the movers), the index's candidates
    // for this subject, the rows it was a member of, and its prior's chain. Every other row is an
    // indexed child the lookup already ruled out — `worth_asking` said no to each of them, one at a
    // time, 233 220 times per subject per tick on the galaxy: 325 ms a tick, a stale window, a frozen
    // picture.
    let remembered: Vec<RealmId> = bits.members().collect();
    let mut ask: Vec<usize> =
        Vec::with_capacity(ctx.unindexed.len() + candidates.len() + remembered.len());
    ask.extend(
        ctx.unindexed
            .iter()
            .filter_map(|r| ctx.ix_of.get(r).copied()),
    );
    ask.extend(candidates.iter().filter_map(|r| ctx.ix_of.get(r).copied()));
    ask.extend(remembered.iter().filter_map(|r| ctx.ix_of.get(r).copied()));
    ask.extend(owned_chain.iter().filter_map(|r| ctx.ix_of.get(r).copied()));
    ask.sort_unstable();
    ask.dedup();
    for ix in ask {
        let region = &ctx.regions[ix];
        let depth_key = ctx.depths[ix];
        if Some(region.realm) == exclude {
            continue; // the subject's own region: it holds its own centre by construction
        }
        if !worth_asking(ctx.child_index, candidates, bits, owned_chain, region.realm) {
            // NOT a member, and not remembered as one, so there is nothing to advance: `set(realm, false)`
            // on an absent realm is a no-op and `members` gains nothing. Skipping is therefore identical
            // to evaluating, which is what `the_index_decides_exactly_what_the_full_scan_decides` proves.
            continue;
        }
        // The prior is STORED-OR-DERIVED (see `owned_mask` above): the remembered bit, OR the fact
        // that this region is the subject's owning realm or an ancestor of it. Hoisted above the
        // verdict because the hysteresis arm (acquire vs release edge) is an INPUT to it.
        let was_member = bits.get(region.realm) | owned_chain.contains(&region.realm);
        // THE INTEGER CONTAINMENT VERDICT (real-scale addendum §A4.8 row 12): re-express the pose
        // into the region's frame through the shard's own AUTHORED placement book (the input-side
        // frame seam, §2.5/FA-1), then decide membership ON INTEGER CELLS for Shell/Aabb — exact at
        // every magnitude, bit-identical across hosts, no float and no square root on the deciding
        // path. The f64 signed distance rides beside it as the LOG GAUGE only. A frame the shard
        // cannot name (`Err`) SAFE-DEGRADES to non-member — never a spurious container.
        let (now, sd) = match vd_core::geometry::region_verdict(
            &measured, prev_pos, region, book, was_member,
        ) {
            Ok(v) => (v.member, v.signed_distance_m),
            // SAFE-DEGRADE, UNCHANGED — but a cross-unit refusal is counted, because "I cannot say" and
            // "definitely not a member" are different answers and only one of them is true here.
            Err(e) => {
                stats.cross_tier_refused += u64::from(matches!(
                    e,
                    vd_core::frame::FrameError::CrossTierCrossing(_)
                ));
                (false, f64::MAX)
            }
        };
        // EVERY REGION'S ANSWER, AND THE PLACEMENT IT WAS MEASURED AGAINST, whenever a NON-OWNED region
        // claims this point. A wrong container is chosen somewhere on the live path and the label follows
        // the decision: an occupant in the star was measured carrying a frame of a planet it is nowhere
        // near. If several children report negative here for one point, they are all sitting at their
        // parent's origin, which is the original two-answers defect alive on the surviving path.
        // Bitwise `&` (not `&&`): all three operands are cheap + pure, and a short-circuit would
        // leave the tail terms regions HR5 cannot cover from a false-LHS side (the discipline
        // `AoiConfig::in_range` uses).
        if (sd <= 0.0) & (region.realm != owning) & (region.parent == Some(owning)) {
            tracing::debug!(
                entity = entity.0,
                // WHOSE SCAN, AND AT WHAT INSTANT. Two sibling realms were seen claiming one point; without
                // these two fields their claims cannot be grouped into a single scan, so it was impossible
                // to tell one shard setting two bits from two shards each answering for the same occupant.
                scanned_by = ?ctx.config.realm,
                at_tick = pose.universe_tick.0,
                // THE SHARD'S OWN CLOCK beside the pose's stamp. The gap between them IS how stale the
                // position being re-decided is: a pose that stops advancing while placements keep moving
                // can be found inside first one sibling and then another without the occupant moving at all.
                shard_tick = ctx.clock.universe_tick.0,
                claimed_by = ?region.realm,
                signed_distance = sd,
                // THE FULL POSITION, never the sub-cell residual. `offset()` is a remainder: at a 1/1024 m
                // cell edge an occupant twenty metres out carries twenty thousand cells and almost no
                // remainder, so reading it as a position says "at the origin" for something far away. That
                // single habit produced three wrong root causes in this arc.
                pose_full = %vd_core::pose::describe(pose.pos, pose.frame),
                placement = ?book.of(region.frame).map(|p| p.origin),
                pose_at = %vd_core::pose::describe(pose.pos, pose.frame),
                pose_frame = ?pose.frame,
                // THE REFRAMED POSITION ITSELF — the number the boundary is actually measured against.
                // Without it the line above cannot distinguish "the placement was wrong" from "the
                // subtraction was wrong", and the two need different fixes.
                reframed = ?vd_core::frame::transfer_frame(&measured, region.frame, book)
                    .map(|p| {
                        let full = p.pos.delta_m(vd_core::pose::LatticePos::ORIGIN, p.frame.tier());
                        (full, p.pos.cell(), full.length())
                    }),
                region_shape = ?region.shape,
                region_frame = ?region.frame,
                "CONTAINMENT: a child claims this point",
            );
        }
        // MEMBERSHIP, which is what the container fold actually reads — not "geometrically inside", which
        // is what the line below used to report. The band's edges straddle the region's surface (an inset
        // acquire edge inside it, an outset release edge past it — STATIC widths: the shipped band is built
        // at v_rel = 0, and the speed-sized widening is still OWED, D-WORLD-4b), so where two siblings sit
        // closer than those dead-zones an occupant can be a member of TWO siblings at once without being
        // inside either. Two same-depth members are then resolved by `depth_beats`, which breaks the tie on
        // realm id — the LOWER number, not the nearer realm.
        if now && region.parent == Some(owning) {
            tracing::debug!(
                entity = entity.0,
                scanned_by = ?ctx.config.realm,
                at_tick = pose.universe_tick.0,
                member_of = ?region.realm,
                signed_distance = sd,
                was_member,
                band_inset = region.band.inset(),
                band_outset = region.band.outset(),
                "CONTAINMENT: a child holds MEMBERSHIP of this occupant",
            );
        }
        bits.set(region.realm, now);
        if now {
            members.push(depth_key);
        }
    }
    // LEAVING IS NOT A SEARCH. If the occupant is inside this realm or one of its children, the deepest
    // member IS the container. If it is inside NOTHING this shard holds, it has left — and the destination
    // is this shard's PARENT, taken from its own lineage.
    //
    // It used to fall back to the ambient ROOT, which required scanning ancestors to find "which enclosing
    // realm contains me" — and that scan is a leak: it needs every ancestor's position and shape. A realm
    // does not know what is outside it and does not need to ([[ground rule]]: only a parent knows where its
    // children are). "Am I outside my own boundary?" is the whole question; WHO is outside is a fact of the
    // shard's own lineage, and WHERE the occupant lands there is the parent's arithmetic, not this shard's.
    let outside_dest = outward_dest(ctx.config, ctx.regions, root_realm, owning);
    let container_realm = container(outside_dest, &members);
    // The container region's PARENT provenance — ★DEAD on the wire: the consumer it was appended for
    // (`rebind_pose_to_dest`) is DELETED (D-PLACE-1); the dest forms an `Area` frame from its own
    // ROSTER (`arrival_frame` — the region carries the planet parent there). Still stamped because the
    // frozen request shape carries the field (postcard is positional); flag-day removal D-WIRE-1.
    let to_parent = ctx
        .regions
        .iter()
        .find(|r| r.realm == container_realm)
        .and_then(|r| r.parent);
    // (`owning` was computed above the region scan — the derived hysteresis prior needs it.)
    // The post-commit cooldown from the per-entity CrossingState.
    let state = progress.entry(entity).or_default();
    let since_commit = state
        .last_commit_tick
        // `.min(u32::MAX)` before the cast: `saturating_sub` is u64, and a raw `as u32` truncation is
        // non-monotone (a gap of `2^32 + k` would read as `k` and falsely re-suppress). (Defensive.)
        .map(|t| (ctx.clock.local_tick.0.saturating_sub(t.0)).min(u32::MAX as u64) as u32);
    // The ONE symmetric re-home decision: re-home iff the deepest container differs from the OWNING realm
    // (past the cooldown). `dest` is the container realm — derived from position, never authored. It ALWAYS
    // fans out the crossing — a same-node dest (a co-hosted child) is the degenerate case of the same saga.
    if let Some(dest) = should_rehome(owning, container_realm, since_commit, &ctx.config.boundary) {
        // THE CROSSING-START LINE (rehome_one_mechanism §4u Stage A). One line per decision, BEFORE the
        // latch is consulted: who decided (this shard), from what label (`owning` — the pose's frame), to
        // where (`dest` — the derived container), at which instants (the pose's stamp beside this shard's
        // clock — the gap IS the staleness), and whether a latch already stands for this entity. Racing
        // crossings and stranded latches are diagnosed by joining these lines across the per-node logs.
        tracing::info!(
            entity = entity.0,
            scanned_by = ?ctx.config.realm,
            owning = ?owning,
            dest = ?dest,
            pose_frame = ?pose.frame,
            at_tick = pose.universe_tick.0,
            shard_tick = ctx.clock.universe_tick.0,
            since_commit = ?since_commit,
            latched = ?in_flight.get(&entity),
            "CROSSING DECIDED: the container differs from the owning realm",
        );
        fan_out_crossing(
            ctx,
            entity,
            subject_fence,
            lane,
            dest,
            to_parent,
            state,
            in_flight,
            stats,
            outbox,
        );
    }
    SubjectEval { prev_offset: cur }
}

/// Fan the committed RE-HOME out by the subject's `DurabilityClass` — the ONE dispatch site (HR2 policy
/// fan-out on one machinery, NO `match` on realm kind, so stations/ships/signals inherit it unchanged).
/// `to_realm` is the DERIVED container realm ([`container`]), a value flowing unmodified through the
/// frozen wire. Monomorphic so both arms are covered once:
/// - `Durable` → emit ONE `CrossingRequest` (latched in `RequestInFlight`, suppressed if already in
///   flight) + arm the cooldown + carry the re-drive payload.
/// - `Transient` → emit a `TransientCrossingRequest` (batched-grant path, no per-entity latch) + arm the cooldown.
#[allow(clippy::too_many_arguments)]
fn fan_out_crossing(
    ctx: &CrossingCtx<'_>,
    entity: EntityId,
    subject_fence: Fence,
    lane: SubjectLane,
    to_realm: RealmId,
    // ★DEAD wire field, threaded for shape only: the consumer it was appended for
    // (`rebind_pose_to_dest`) is DELETED (D-PLACE-1) — the dest forms its frame from its own ROSTER
    // (`arrival_frame`). Flag-day removal ledgered D-WIRE-1.
    to_parent: Option<RealmId>,
    state: &mut CrossingState,
    in_flight: &mut BTreeMap<EntityId, TransferId>,
    stats: &mut StubStats,
    outbox: &mut OutboundBox,
) {
    let from_realm = ctx.config.realm;
    // ★ THE EXTERIOR ARM (the ruler switch, slice 1). The verdict stands and is counted; the request
    // that carries it to the orchestrator is the third request arm on the wire (beside the durable
    // and the transient one), which under SL6 waits for the owner's word. The cooldown is armed so
    // a hull dwelling on a shell is decided once per dwell, not once per tick.
    let SubjectLane::ByKind(subject_session) = lane else {
        stats.exterior_crossings_decided += 1;
        tracing::info!(
            entity = entity.0,
            from_realm = ?from_realm,
            to_realm = ?to_realm,
            subject_fence = ?subject_fence,
            "EXTERIOR CROSSING DECIDED: a driven child left this realm or entered a sibling",
        );
        // Latched exactly like a durable subject: one request per crossing, re-driven on the ttl,
        // suppressed while in flight. The subject is the child's EXTERIOR key, and the fence is the
        // exterior lease this realm holds — the saga's CAS expectation.
        use std::collections::btree_map::Entry;
        match in_flight.entry(entity) {
            Entry::Vacant(slot) => {
                let subject = DirectoryKey::Ship(entity);
                let attempt = state.crossing_attempt;
                let transfer = crossing_transfer_id(subject, subject_fence, attempt);
                slot.insert(transfer);
                outbox.push_flow(
                    ctx.config.orchestrator,
                    MsgClass::Saga,
                    &InterShardFlow::ExteriorCrossingRequest(ExteriorCrossingRequest {
                        subject,
                        from_realm,
                        to_realm,
                        subject_fence,
                        attempt,
                    }),
                );
                state.last_commit_tick = Some(ctx.clock.local_tick);
                state.latched_crossing = Some(LatchedCrossing {
                    to_realm,
                    subject_fence,
                    session: SessionId::NONE,
                    to_parent,
                    exterior: true,
                });
                state.redrives_spent = 0;
                stats.exterior_crossings_requested += 1;
                tracing::info!(
                    entity = entity.0,
                    transfer = ?transfer,
                    from_realm = ?from_realm,
                    to_realm = ?to_realm,
                    attempt,
                    subject_fence = ?subject_fence,
                    "EXTERIOR CROSSING REQUEST EMITTED and latched",
                );
            }
            Entry::Occupied(_) => {
                stats.crossings_suppressed_in_flight += 1;
            }
        }
        return;
    };
    match durability_of(entity) {
        DurabilityClass::Durable => {
            // A durable crossing subject is normally a session-owned dot (only the dot loop passes
            // `Some(session)`). But `durability_of` keys on the entity's KIND TAG, not the loop of origin:
            // a Durable-TAGGED item in the held-transient set would reach this arm with `None`. DEGRADE,
            // never panic (an unroutable crossing must not start a saga): count it and emit nothing. Gated
            // BEFORE the latch insert so no orphan latch is left.
            let Some(session) = subject_session else {
                stats.crossing_durable_no_session += 1;
                return;
            };
            // The durable transfer request — latched so exactly ONE fires per crossing. The `Vacant`
            // arm inserts the deterministic latch id + emits + arms the cooldown; the `Occupied` arm
            // (already in-flight) is the SUPPRESS no-op. `Entry` (not `contains_key`+`insert`) so there
            // is one map lookup and clippy's map_entry lint is satisfied.
            use std::collections::btree_map::Entry;
            match in_flight.entry(entity) {
                Entry::Vacant(slot) => {
                    let subject = DirectoryKey::Entity(entity);
                    // 3f-D (H2): the id is stamped with the current attempt so each latch is UNIQUE. The
                    // attempt is bumped ONLY when this latch later CLEARS on abort (`on_crossing_aborted`),
                    // NOT here — so the still-latched id is stable for the ttl-re-drive, and a crash (which
                    // evicts `CrossingProgress` → attempt back to 0) re-mints the same first-attempt id.
                    let attempt = state.crossing_attempt;
                    let transfer = crossing_transfer_id(subject, subject_fence, attempt);
                    slot.insert(transfer);
                    outbox.push_flow(
                        ctx.config.orchestrator,
                        MsgClass::Saga,
                        &InterShardFlow::CrossingRequest(CrossingRequest {
                            subject,
                            from_realm,
                            to_realm,
                            subject_fence,
                            session,
                            attempt,
                            to_parent,
                        }),
                    );
                    state.last_commit_tick = Some(ctx.clock.local_tick);
                    // 3f-D4: carry the re-emit payload so `redrive_stranded_crossings` can re-mint the SAME
                    // request for a delivered-but-unresolved dest (a stranded latch with no rising edge).
                    state.latched_crossing = Some(LatchedCrossing {
                        to_realm,
                        subject_fence,
                        session,
                        to_parent,
                        exterior: false,
                    });
                    // D-WORLD-2: a fresh latch starts a fresh re-drive budget.
                    state.redrives_spent = 0;
                    stats.crossings_requested += 1;
                    // Stage A: the emit line pairs with "CROSSING DECIDED" above and with the
                    // orchestrator's "CROSSING SAGA STARTED" via the transfer id.
                    tracing::info!(
                        entity = entity.0,
                        transfer = ?transfer,
                        from_realm = ?from_realm,
                        to_realm = ?to_realm,
                        attempt,
                        subject_fence = ?subject_fence,
                        "CROSSING REQUEST EMITTED and latched",
                    );
                }
                Entry::Occupied(held) => {
                    stats.crossings_suppressed_in_flight += 1;
                    // Stage A: fires per tick while the occupant dwells over a boundary with a latch
                    // standing. Bounded since the D-WORLD-2 cure: a latch nothing terminates is
                    // re-driven `crossing_redrive_budget` times at the `request_ttl_ticks` cadence and
                    // then aborted locally, so these lines can never run forever.
                    tracing::debug!(
                        entity = entity.0,
                        latched = ?held.get(),
                        wanted_to = ?to_realm,
                        from_realm = ?from_realm,
                        "CROSSING SUPPRESSED: a request is already in flight for this entity",
                    );
                }
            }
        }
        DurabilityClass::Transient => {
            // The transient crossing request (the batched-grant path — no per-entity latch; the batch
            // idempotency lives in the grant/adopt journal). `src_realm_fence` is the realm fence.
            outbox.push_flow(
                ctx.config.orchestrator,
                MsgClass::Saga,
                &InterShardFlow::TransientCrossingRequest(TransientCrossingRequest {
                    subject: DirectoryKey::Entity(entity),
                    from_realm,
                    to_realm,
                    src_realm_fence: ctx.realm_fence,
                    to_parent,
                }),
            );
            state.last_commit_tick = Some(ctx.clock.local_tick);
            stats.transient_crossings_requested += 1;
        }
    }
}

/// Slice 3f-D4 (DEFERRED D-43 #2) + the D-WORLD-2 exhaustion cure — the per-tick RE-DRIVE of a
/// STRANDED durable crossing latch, BOUNDED. A DELIVERED-but-unresolved dest (`head(Realm(to))`
/// absent — an UNHOSTED realm on a static cluster, or a realm shard mid-lease/partitioned at P4/P5)
/// leaves the `RequestInFlight` latch standing with NO re-emit: under CONTAINMENT (task #135)
/// `should_rehome` returns `Some(container)` every tick the container differs, but the still-
/// latched dot hits `fan_out_crossing`'s `Occupied` SUPPRESS arm each tick, so nothing re-fires while the
/// dot dwells in the (unresolved) region. This scan re-emits the SAME latched
/// `CrossingRequest` (same `(subject, subject_fence, attempt)` → byte-identical [`crossing_transfer_id`];
/// the orchestrator's `contains_key` guard absorbs a dup that already started, an unresolved one re-tries
/// the head reads) once `local_tick - last_commit_tick >= request_ttl_ticks`, then re-arms the ttl timer —
/// but only `crossing_redrive_budget` times: a latch still standing at the NEXT expiry has spent one
/// full destructive-abort window per re-drive with no terminal, so the dest is UNRESOLVABLE and the
/// source takes the LOCAL pre-CAS abort ([`abort_crossing_latch`], the same body the wire
/// `CrossingAborted` runs): latch cleared, attempt bumped, containment cooldown armed at the abort tick
/// (the `k_dwell` dwell bounds a parked dweller's re-fire; a graze that already left simply continues —
/// its container matches its owner again, so nothing re-fires). The entity stays simulated at the
/// source throughout — no saga ever started, so there is nothing to thaw beyond the latch itself.
/// `request_ttl_ticks == 0` (disarmed — unit rigs only since the D-WORLD-2 cure armed every launcher)
/// → an early return before any iteration; the exhaustion arm sits behind the same ttl gate.
/// Authority-gated exactly like `evaluate_realm_boundaries`. A BRANCHLESS shim over the
/// re-emit payload the latch carried at emit time ([`LatchedCrossing`]) — no recovery from live geometry;
/// the only branches are the ttl early-return, the authority gate, the `>= ttl` window, and the
/// budget split, each covered once (HR5). Determinism: `in_flight.0` / `progress.0` are `BTreeMap`s
/// (ordered iteration; the exhausted set is collected in that order and drained after the scan), the
/// re-mint is a pure fn of the latched fields + the universe clock, `.min(u32::MAX)` guards the cast
/// (mirrors `evaluate_one_subject`'s `since_commit` compute). The `.expect()`s are STRAIGHT-LINE
/// invariants (a held latch always carries its `CrossingState` — `retain_live` syncs both on the same
/// live set — with a `latched_crossing` payload + an armed `last_commit_tick`), matching the existing
/// session `.expect()` shape, so they add no coverable false arm.
#[allow(clippy::too_many_arguments)]
pub(crate) fn redrive_stranded_crossings(
    config: Res<StubConfig>,
    clock: Res<ClockSample>,
    authority: Res<RealmAuthority>,
    regions: Res<RealmRegions>,
    mut in_flight: ResMut<RequestInFlight>,
    mut progress: ResMut<CrossingProgress>,
    mut holds: ResMut<HandoffHolds>,
    mut stats: ResMut<StubStats>,
    mut outbox: ResMut<OutboundBox>,
) {
    let Some(realm_fence) = authority.0 else {
        return;
    };
    // On an ARMED (demand-scale) shard, KEEP an INWARD crossing DEST — and, via `ancestor_close`, its
    // whole Universe→…→dest chain — demand-alive for EXACTLY the crossing window, so a player crossing
    // INTO a realm cannot have it reaped out from under them (the Planet→System return-freeze fix). The
    // in-flight latch IS the lifecycle: taken at `fan_out_crossing`, cleared POSITIVELY at commit /
    // pre-CAS abort / vanish. A cleared latch is simply not scanned ⇒ the keep-alive stops and the dest
    // self-heals — on commit the arriving player's own arm-B liveness carries it (the last keep-alive's
    // `demand_ttl` overlaps, no gap); on abort it ages out after at most `demand_ttl`, a bounded tail,
    // never a leak. An OUTWARD crossing's dest is this shard's PARENT, which a shard may never demand
    // (SL7 — a realm speaks about itself or a direct child, never upward): `push_demand`'s structural
    // gate refuses it, counted, and the parent stays alive anyway — the latch keeps the departing
    // occupant in this shard's observer fold (`speaks_for`), so this realm never reports `Empty`, arm B
    // of `desired_alive` holds it, and `ancestor_close` pulls the parent chain. Measured by the
    // return-crossing gate (`a_planet_to_system_return_commits_both_rehomes_and_the_player_rides`).
    let armed = regions.aoi_live();
    let ttl = config.request_ttl_ticks;
    // Walk/static scale (inert AoI) with the ttl re-drive disabled is byte-identical to the old early-out.
    if !armed && ttl == 0 {
        return;
    }
    // `in_flight` and `progress` are DISTINCT resources — no aliasing — so the scan reads the held
    // latches while mutating each subject's `CrossingState` in one deterministic pass. Exhausted
    // latches are collected (in the same `BTreeMap` order) and aborted AFTER the pass — the abort
    // removes from `in_flight`, which must not happen under its own iteration borrow.
    let mut exhausted: Vec<EntityId> = Vec::new();
    for (entity, _latched) in in_flight.0.iter() {
        let state = progress
            .0
            .get_mut(entity)
            .expect("a held latch always has a CrossingState (retain_live syncs both)");
        let lc = state
            .latched_crossing
            .expect("a held durable latch carries its re-emit payload");
        if armed {
            // The full lineage coord of the dest, when this shard's forest can name it. A destination it
            // cannot name — a built realm it does not host, the case the old `expect` here named as the
            // P8 ship-realm work — gets no keep-alive from here and is COUNTED, never assumed (the ruler
            // switch, slice 1).
            match regions.coord_of(lc.to_realm) {
                Some(coord) => push_demand(
                    &mut outbox,
                    config.orchestrator,
                    &config.own_coord,
                    coord,
                    realm_fence,
                    DemandVerb::KeepAlive,
                    clock.universe_tick,
                    &mut stats,
                ),
                None => stats.crossing_keepalive_unnamed += 1,
            }
        }
        if ttl != 0 {
            let last = state
                .last_commit_tick
                .expect("a held latch armed the cooldown at emit time");
            let elapsed = clock
                .local_tick
                .0
                .saturating_sub(last.0)
                .min(u32::MAX as u64) as u32;
            if elapsed >= ttl {
                if state.redrives_spent < config.crossing_redrive_budget {
                    // The byte-identical request again, on the arm the latch was minted for: the SAME
                    // attempt (it bumps only on a post-abort re-latch) and the SAME parent provenance.
                    let flow = if lc.exterior {
                        InterShardFlow::ExteriorCrossingRequest(ExteriorCrossingRequest {
                            subject: DirectoryKey::Ship(*entity),
                            from_realm: config.realm,
                            to_realm: lc.to_realm,
                            subject_fence: lc.subject_fence,
                            attempt: state.crossing_attempt,
                        })
                    } else {
                        InterShardFlow::CrossingRequest(CrossingRequest {
                            subject: DirectoryKey::Entity(*entity),
                            from_realm: config.realm,
                            to_realm: lc.to_realm,
                            subject_fence: lc.subject_fence,
                            session: lc.session,
                            attempt: state.crossing_attempt,
                            to_parent: lc.to_parent,
                        })
                    };
                    outbox.push_flow(config.orchestrator, MsgClass::Saga, &flow);
                    // Re-arm the ttl timer so the next re-drive is another `ttl` ticks out, and spend
                    // one unit of the D-WORLD-2 budget.
                    state.last_commit_tick = Some(clock.local_tick);
                    state.redrives_spent = state.redrives_spent.saturating_add(1);
                    stats.crossings_redriven += 1;
                } else {
                    // D-WORLD-2 EXHAUSTION: the budget is spent and the latch STILL stands — every
                    // re-drive window (each ≥ the worst healthy saga resolve) passed with no terminal,
                    // so the dest is unresolvable from here. Collected; aborted after the pass.
                    exhausted.push(*entity);
                }
            }
        }
    }
    for entity in exhausted {
        // The SOURCE-LOCAL pre-CAS abort (the D-WORLD-2 cure): the same clear the wire `CrossingAborted`
        // runs, minus the wire (no saga ever started — SL6: nothing new crosses a boundary here). The
        // cooldown arms at THIS tick, so a dot parked inside the unresolved region re-fires only past
        // the existing `should_rehome` `k_dwell` dwell — the bounded sit-inside cadence — while a graze
        // that already left continues unharmed (its container matches its owner again). WARN, never
        // silent: this is a realm the cluster could not resolve for a whole `(budget+1)·ttl` window.
        tracing::warn!(
            entity = entity.0,
            from_realm = ?config.realm,
            wanted_to = ?progress.0.get(&entity).and_then(|st| st.latched_crossing).map(|lc| lc.to_realm),
            redrives_spent = config.crossing_redrive_budget,
            ttl,
            "CROSSING EXHAUSTED: dest never resolved — aborting locally, latch cleared, entity stays at the source",
        );
        abort_crossing_latch(
            entity,
            Some(clock.local_tick),
            &mut in_flight,
            &mut progress,
            &mut holds,
            &mut stats,
        );
        stats.crossings_exhausted += 1;
    }
}
