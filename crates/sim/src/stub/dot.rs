//! THE OCCUPANT: one avatar as this shard holds it, and the input that moves it.
//!
//! Owns: the [`Dot`] record and the shard's dot table, the entity mint (ids are minted HERE, from
//! seed-derived entropy, never from a clock), the [`InputLog`] that is the INPUT-CONSERVATION
//! oracle's ground truth, and the integrate/govern arithmetic that turns one delivered input
//! datagram into a new pose.
//!
//! Does NOT own: where the dot goes next (`containment`), who is told about it (`frames`,
//! `window`), or how it crosses a realm boundary (`conversion`, `saga_arms`). The integrator knows
//! the realm it is inside and nothing above it — a dot's pose is measured from its own realm's
//! centre, full stop (SL1).

use super::{RealmRegions, StubConfig, flight_tuning};
use crate::authority::Authority;
use crate::runtime::ClockSample;
use bevy_ecs::prelude::Resource;
use std::collections::{BTreeMap, VecDeque};
use vd_core::entity_kind::EntityKind;
use vd_core::flight::{self, FlightTuning};
use vd_core::kinematics::{self};
use vd_core::placement::{PlacementBook, PlacementLedger};
use vd_core::pose::{FrameRef, LatticePos, RealmId, StampedPose};
use vd_core::rng::SplitMix64;
use vd_core::{AccountId, EntityId, Fence, NodeId, SessionId};
use vd_wire::channels::InputDatagram;

/// One connected avatar.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Dot {
    /// ★ THE STICK THIS PILOT LAST HELD (D-MOVE-2, the temporary control seam) — kept on the PILOT
    /// rather than on the realm, because that is whose it is.
    ///
    /// A player's keys already travel to whichever shard holds that player, and the route swaps on
    /// every crossing. So a pilot who steps aboard a ship has their keys arrive at the ship's own
    /// shard with NO routing change: the seam is only that the ship reads them.
    ///
    /// Later a SEAT decides which pilot is at the controls; today a self-driven realm reads the one it
    /// holds. `None` until a first input arrives, which is why a realm holding a silent pilot pushes
    /// nothing rather than pushing zero.
    pub last_stick: Option<([f32; 3], [f32; 3])>,
    pub entity: EntityId,
    pub account: AccountId,
    /// The Session-key fence the owning gateway holds; stale input is dropped.
    pub session_fence: Fence,
    /// The gateway this session arrived through (reply route — NEVER an address).
    pub gateway: NodeId,
    /// The directory-record PREDICATE truth (NOT the authority truth — `authority` is that):
    /// `granted` answers "is this entity's grant recorded here / which directory op is owed",
    /// keyed on by `pending_grant_op`/`foreign_takeover_target`/`crossing_target`/`flush_target`.
    /// Strictly weaker poll-bookkeeping that 1d.5b's saga-pushed Promote/Demote DELETES; it never
    /// re-decides emit/input/oracle (those are `authority.simulates()`). An adopted-but-not-crossed
    /// dot and a retained source Ghost are both `granted==true && simulates()==false` (FG-2 interim).
    pub granted: bool,
    /// A TRANSFER-DESTINATION input slot may APPLY input before its full directory grant
    /// (1c.5): the gateway sends `OpenInputSlot` only AFTER the directory committed authority
    /// to this shard (the saga's `commit_cas`), so this is the gateway's commit-time
    /// attestation that the shard owns the session's INPUT. It is NARROWER than `granted`:
    /// an `input_active` dot applies input (so the post-marker cut buffer drains here, input-
    /// conservation) but is NOT rendered and holds NO directory record — the real per-entity
    /// `Authority` attach + render + ghost is 1d (D-27), which sets `granted`. A regular
    /// attach sets `granted` (and `input_active` stays false; `granted` alone permits input).
    pub input_active: bool,
    /// This dot is a TRANSFER-DESTINATION ADOPT (set by `OpenInputSlot` carrying the transfer
    /// subject): its `entity` is the SUBJECT id (not a fresh mint), and it ADOPTS the existing
    /// directory record (a `HeadRead`, never a `LeaseGrant` the CAS fence would Refuse). Cleared
    /// on the grant flip. The adopt dot is born `Ghost` (the frozen ghost mirror) and STAYS Ghost
    /// (renders NOTHING) until the saga `Promote` flips it `Ghost→Owned` in `on_saga_promote`
    /// (1d.5b.3b — strict demote-before-promote; `apply_crossing` only STORES the crossed pose, it no
    /// longer promotes); the grant flip carries NO `SessionAttached` (the source still owns the client
    /// connection — R2).
    pub adopting: bool,
    /// The per-entity authority TRUTH (`authority.rs` FSM, attached 1d.4b/D-27): `Owned`
    /// simulates+holds, `Ghost` is a retained read-only mirror, `Frozen` is mid-transfer.
    /// `authority.simulates()` is the SINGLE answer to "does this shard ACCEPT-BY-AUTHORITY / HOLD
    /// this entity" — half the `apply_input` gate and the oracle held-set. EMIT-eligibility (slice F)
    /// is the strictly-DERIVED `simulates() | (is_retained_ghost & Source-hold-open)` (`emits`): a
    /// retained Ghost emits its kinematic mirror ONLY while the hand-off hold is open — the leaver
    /// vanishes at hold closure — and the fed-ghost lane is DELETED (slice F). It integrates/accepts
    /// NOTHING (FG-2 — emit-eligibility is derived from authority + the hold,
    /// never a competing authority store). Login AND the transfer-dest both mint
    /// `Ghost{GENESIS}` (simulate nothing
    /// pre-grant) and Promote `Ghost→Owned` via the IDENTICAL machinery (login at the grant fence,
    /// dest at the crossing fence) — kind-generic: a ship/block/signal entity uses the SAME states
    /// (no per-kind fork). The source self-fence demotes `Owned→Frozen→Ghost` and RETAINS the dot.
    pub authority: Authority,
    /// The mirror on release: a detached dot stays HELD (authoritative) until
    /// the directory confirms its revoke — authority is released AT the
    /// directory, never by local despawn.
    pub departing: bool,
    /// The directory-RECORDED PREDICATE fence for this entity (FENCE-1/5/8, the LeaseRevoke key + the
    /// crossing/grant fence): every grant/revoke uses THIS, never a hardcoded literal, so a transfer
    /// that advanced the fence past genesis cannot wedge a logout forever. NOT kept in sync with
    /// `authority.fence()`: on a retained source Ghost they intentionally diverge (`entity_fence` =
    /// the dot's OWN old grant fence; `authority.fence()` = the new owner's). `authority.fence()` is
    /// the FSM truth; `entity_fence` is the recorded-predicate fence (RETAINED — only the 1c.8 poll
    /// that read it for the granted-key HeadRead was torn out in 1d.5b.2).
    pub entity_fence: Fence,
    pub pose: StampedPose,
    pub yaw: f64,
    pub pitch: f64,
    pub last_applied_seq: Option<u64>,
    /// The frame-local offset (`pose.pos.offset()`) at the END of the previous tick, written LAST each
    /// evaluation tick to `cur`. INERT under CONTAINMENT (task #135): the detector uses POINT membership
    /// (`region_signed_distance` at `cur`), not a swept segment, so `prev_offset` is written-but-unread —
    /// RESERVED for the deferred additive swept tunnel-guard (DEFERRED D-45). Seeded to the spawn offset
    /// at every construction site.
    pub prev_offset: LatticePos,
}

/// All avatars on this shard, in deterministic session order. The key set IS the
/// shard's held-set for the AUTHORITY-UNIQUE oracle.
#[derive(Resource, Debug, Default)]
pub struct Dots(pub BTreeMap<SessionId, Dot>);

/// Entity minting state: a per-shard monotonic sequence + seed-derived entropy.
#[derive(Resource, Debug)]
pub struct EntityMint {
    pub(crate) seq: u64,
    pub(crate) rng: SplitMix64,
}

/// Why an input was not applied (typed — never a stringly warn-and-drop).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DiscardReason {
    /// Carried a fence below the session's highest-seen (stale gateway — R2 guard).
    StaleFence,
    /// seq at or below the last applied (duplicate or reordered; latest-wins).
    DuplicateSeq,
    /// No avatar for that session on this shard.
    UnknownSession,
    /// The avatar exists but its directory grant is not yet confirmed — it has
    /// no authority to consume input (fence rule 2).
    PendingAuthority,
    /// The avatar is being released (detach received, revoke in flight) — it no
    /// longer consumes input.
    Departing,
    /// The payload failed to decode.
    MalformedInput,
    /// The payload decoded but carried a non-finite (NaN/Inf) movement/look component —
    /// a forged or corrupt client. Integrating it would PERMANENTLY poison the
    /// authoritative pose (NaN sticks through every later tick), so it is discarded
    /// and counted at the ingress gate (`InputDatagram::is_finite`), mirroring the
    /// client's delivered-pose `sanitized()` chokepoint.
    NonFiniteInput,
}

/// The INPUT-CONSERVATION ground truth: every delivered input lands here, applied or
/// discarded-with-reason. BOUNDED (SCALE-3): a day-long shard run cannot grow this
/// without limit — the windows hold the most recent `capacity` entries and the
/// `*_total` counters are EXACT for metrics. The harness sets a large window so the
/// oracle still sees a whole short run; production sets a small one.
#[derive(Resource, Debug)]
pub struct InputLog {
    applied: VecDeque<(SessionId, u64)>,
    discarded: VecDeque<(SessionId, Option<u64>, DiscardReason)>,
    capacity: usize,
    /// Exact lifetime totals (never lossy — the honest metric).
    pub applied_total: u64,
    pub discarded_total: u64,
    /// Entries evicted from the windows because the consumer fell behind the
    /// `capacity` window (in production nobody drains; counted, never an OOM).
    pub window_evictions: u64,
}

impl InputLog {
    #[must_use]
    pub fn new(capacity: usize) -> InputLog {
        InputLog {
            applied: VecDeque::new(),
            discarded: VecDeque::new(),
            capacity: capacity.max(1),
            applied_total: 0,
            discarded_total: 0,
            window_evictions: 0,
        }
    }

    pub(crate) fn record_applied(&mut self, session: SessionId, seq: u64) {
        self.applied_total += 1;
        if self.applied.len() >= self.capacity {
            self.applied.pop_front();
            self.window_evictions += 1;
        }
        self.applied.push_back((session, seq));
    }

    pub(crate) fn record_discarded(
        &mut self,
        session: SessionId,
        seq: Option<u64>,
        reason: DiscardReason,
    ) {
        self.discarded_total += 1;
        if self.discarded.len() >= self.capacity {
            self.discarded.pop_front();
            self.window_evictions += 1;
        }
        self.discarded.push_back((session, seq, reason));
    }

    /// The applied window, oldest→newest (the oracle's ground truth).
    #[must_use]
    pub fn applied(&self) -> Vec<(SessionId, u64)> {
        self.applied.iter().copied().collect()
    }

    /// The discarded window, oldest→newest.
    #[must_use]
    pub fn discarded(&self) -> Vec<(SessionId, Option<u64>, DiscardReason)> {
        self.discarded.iter().copied().collect()
    }
}

/// Per-shard monotonic snapshot frame counter.
#[derive(Resource, Debug, Default)]
pub struct FrameCounter(pub u64);

/// Mint a fresh avatar id: `{kind, mint_shard, seq, rand24}` — monotonic sequence
/// plus seed-derived entropy; NEVER time-derived (R7).
pub(crate) fn mint_entity(mint: &mut EntityMint, node: NodeId) -> EntityId {
    let seq = mint.seq;
    mint.seq += 1;
    let rand24 = (mint.rng.next_u64() & 0x00FF_FFFF) as u32;
    // NodeIds fit u32 in every real topology; the mask documents the packing.
    let mint_shard = (node.0 & u64::from(u32::MAX)) as u32;
    EntityId::pack(EntityKind::Player, mint_shard, seq, rand24)
}

/// The input path: fence gate → decode → seq gate → integrate. Every outcome lands
/// in the [`InputLog`].
#[allow(clippy::too_many_arguments)]
pub(crate) fn apply_input(
    config: &StubConfig,
    clock: &ClockSample,
    regions: &RealmRegions,
    placements: &PlacementLedger,
    dots: &mut Dots,
    log: &mut InputLog,
    session: SessionId,
    fence: Fence,
    input_bytes: &[u8],
) {
    let Some(dot) = dots.0.get_mut(&session) else {
        log.record_discarded(session, None, DiscardReason::UnknownSession);
        return;
    };
    // A dot may apply input once it is the AUTHORITY for the session's input: either it SIMULATES
    // (`Authority::Owned`) OR a committed transfer-destination input slot (`input_active`, 1c.5 —
    // the gateway opened it only post-directory-commit, so the post-marker cut buffer drains+
    // integrates here even though the dest is still a Ghost; subtlety 4). A purely provisional dot
    // (neither) drops input as PendingAuthority — incl. a RETAINED source Ghost (`simulates()==false`,
    // `input_active==false`), which correctly drops late input (was UnknownSession under the old
    // `dots.remove`; both record `seq=None` so they are INPUT-CONSERVATION-equivalent).
    if !dot.authority.simulates() && !dot.input_active {
        log.record_discarded(session, None, DiscardReason::PendingAuthority);
        return;
    }
    if dot.departing {
        log.record_discarded(session, None, DiscardReason::Departing);
        return;
    }
    if fence.is_stale_against(dot.session_fence) {
        log.record_discarded(session, None, DiscardReason::StaleFence);
        return;
    }
    // A higher fence means the gateway re-granted (P3 adoption); track it.
    if fence > dot.session_fence {
        dot.session_fence = fence;
    }
    let Ok(input) = postcard::from_bytes::<InputDatagram>(input_bytes) else {
        log.record_discarded(session, None, DiscardReason::MalformedInput);
        return;
    };
    // The authoritative-ingress finite gate (never trust network input): a NaN/Inf
    // component would integrate into the pose and STICK. Discarded + counted; the seq
    // does NOT advance (the input was never applied — INPUT-CONSERVATION holds).
    if !input.is_finite() {
        log.record_discarded(session, Some(input.seq), DiscardReason::NonFiniteInput);
        return;
    }
    if dot.last_applied_seq.is_some_and(|last| input.seq <= last) {
        log.record_discarded(session, Some(input.seq), DiscardReason::DuplicateSeq);
        return;
    }
    dot.last_applied_seq = Some(input.seq);
    // The pilot's own frame stick, kept for the realm's drive producer. The turn is built here rather
    // than in the producer so the two consumers of `look` cannot drift: pitch turns about the right
    // axis, yaw about the up axis, and roll has no key yet.
    dot.last_stick = Some((input.movement, [input.look[1], input.look[0], 0.0]));
    integrate(dot, &input, config, clock, regions, placements);
    log.record_applied(session, input.seq);
}

/// THE APPROACH GOVERNOR's answer for one subject (§4.2(c), ★OQ-2 owner-ruled universal): the
/// fastest speed the CONTAINING realm permits at `pos` — the realm's own ceiling, lowered toward
/// each direct child's own ceiling as the subject nears that child's bound
/// ([`flight::approach_ceiling_mps`]), so nothing can arrive fast or fly through anything.
///
/// EVERY input is something the realm lawfully holds (SL1/SL2/SL6 answer: NONE crosses): its own
/// region's bound, its children's bounds (it authors their placements), and the placement rows it
/// authored THIS tick — read through the ledger's head book and the ONE `child_rows` join (H2: no
/// second position path; SL4: a book row cannot say how a child moves). The child distance is the
/// f64 `delta_m` flatten — a CONTROL input (a speed limit), read at AoI-class precision, exactly
/// like the AoI range test (addendum §A6.2 item 3 records the f64 posture at galaxy magnitude);
/// the authority verdict stays integer and is untouched here.
///
/// Three graceful arms answer `None` — "this realm states NO ceiling here": an unhosted frame, a
/// shard with no region forest (walk/static rigs through C-3), and a not-yet-authored ledger
/// (pre-sync). `None` is the PRE-LAW posture and each consumer maps it to ITS OWN pre-law
/// behaviour — the throttle to the plain foot speed (a commanded speed never exceeded it), the
/// transient clamp to no clamp at all (a seeded velocity was never cut) — so a forestless rig is
/// byte-identical on BOTH paths. A stated ceiling is structurally `>= v_foot` on every arm — the
/// law can only ever RAISE the ceiling above the foot floor, never cut into it.
pub(crate) fn governed_ceiling_for_frame(
    regions: &RealmRegions,
    placements: &PlacementLedger,
    frame: FrameRef,
    pos: LatticePos,
    tuning: &FlightTuning,
) -> Option<f64> {
    let realm = regions.realm_of_frame(frame)?;
    let book = placements.head(realm)?;
    governed_ceiling_in_book(regions, realm, book, pos, tuning)
}

/// The ceiling arithmetic over ONE authored book — split from the frame/ledger resolve so the unit
/// tier measures it against `author_book`-built books directly (the crate's `publish` ban stands:
/// no test mints a rival ledger history). `None` when `realm` has no region here — the same
/// no-law answer as the outer resolve.
pub(crate) fn governed_ceiling_in_book(
    regions: &RealmRegions,
    realm: RealmId,
    book: &PlacementBook,
    pos: LatticePos,
    tuning: &FlightTuning,
) -> Option<f64> {
    let own = regions.regions.iter().find(|r| r.realm == realm)?;
    let tier = own.frame.tier();
    // THE CRUISE CEILING — the only term the throwaway overdrive touches (see `cruise_overdrive`).
    let lawful = flight::realm_speed_cap_mps(
        own.shape.finite_extent(),
        tuning.v_foot_mps,
        tuning.traverse_s,
    );
    let mut v = lawful * regions.cruise_overdrive;
    // ★THE OUTWARD ARM (test-instrument safety, measured 2026-08-20 from a flight that left the world).
    //
    // The child arms below slow you INWARD. Nothing slowed you toward your OWN realm's exit, because
    // the lawful ceiling never needed it: the bands that catch an outward crossing are sized against
    // that same lawful ceiling. An overdriven cruise breaks exactly that assumption — a subject crosses
    // its own exit band in a fraction of a tick, the crossing never latches, and it keeps going with
    // nothing outside to catch it. Measured: a pilot at 64x left the universe and stuck at the domain
    // edge.
    //
    // So the exit is governed like everything else, by the SAME falling-ceiling law and the SAME lawful
    // target: your ceiling decays to the realm's own lawful cap as you near your own shell. Deep inside
    // you cruise overdriven; at the boundary you are lawful, so the band is crossed at the speed it was
    // sized for. A realm is centred on itself, so the distance to its own shell is its extent less your
    // own distance from its centre — no parent, no ancestor, nothing it does not already hold (SL1).
    //
    // At the lawful `1.0` this arm is INERT by construction: `v` is already `lawful`, and the arm can
    // only ever return `lawful + something non-negative`, so the `min` cannot bite.
    let dist_to_own_shell =
        own.shape.finite_extent() - pos.delta_m(LatticePos::ORIGIN, tier).length();
    v = v.min(flight::approach_ceiling_mps(
        lawful,
        dist_to_own_shell,
        tuning.tau_s,
    ));
    for (child, placed) in regions.child_rows(realm, book) {
        let child_cap = flight::realm_speed_cap_mps(
            child.shape.finite_extent(),
            tuning.v_foot_mps,
            tuning.traverse_s,
        );
        let dist_to_bound = pos.delta_m(placed.pos, tier).length() - child.shape.finite_extent();
        v = v.min(flight::approach_ceiling_mps(
            child_cap,
            dist_to_bound,
            tuning.tau_s,
        ));
    }
    Some(v)
}

/// Kinematic point integration: axes are clamped to [-1, 1], displacement is
/// speed·dt in the dot's yaw-rotated heading. Pure f64 closed-form per tick.
///
/// THE SPEED LAW rides here as ONE scale factor (S3, the design's one integrator seam): the
/// commanded speed is the geometric throttle map under the realm's GOVERNED ceiling
/// ([`governed_ceiling_for_frame`]) further capped by the proportional RAMP
/// ([`flight::ramp_cap_mps`], seeded from the pose's own carried velocity — no new state, and the
/// ramp memory crosses shards with the pose). Wherever the ceiling clamps to the foot speed —
/// every sub-45 km realm, every region-less rig — the scale is `1.0` EXACTLY and the step below is
/// the pre-law arithmetic bit-for-bit (the inertness pin
/// `the_speed_law_is_bit_inert_wherever_the_ceiling_clamps` measures it). Deceleration is
/// deliberately instant (P3 "stopped means stopped"); the gradual arrival slow-down is the
/// governor's falling ceiling, not a ramp state.
fn integrate(
    dot: &mut Dot,
    input: &InputDatagram,
    config: &StubConfig,
    clock: &ClockSample,
    regions: &RealmRegions,
    placements: &PlacementLedger,
) {
    dot.yaw += f64::from(input.look[0]);
    // Pitch is CLAMPED to the valid look range (WB-1): unbounded accumulation would wrap
    // past the ±π/2 gimbal pole and silently corrupt authoritative orientation. Yaw wraps
    // freely (no pole). ONE shared bound (`vd_core::kinematics::PITCH_LIMIT`).
    dot.pitch = kinematics::clamp_pitch(dot.pitch + f64::from(input.look[1]));
    dot.orient_from_angles();
    // The movement-axis map is the ONE shared input convention (vd_core::kinematics) —
    // the client's nav/camera invert the SAME definition (no hand-re-encoded drift).
    let axes = kinematics::local_axes_from_movement(input.movement);
    let tuning = flight_tuning(config);
    // `None` (no stated ceiling) maps to the foot speed HERE: the throttle's pre-law commanded
    // speed never exceeded it, and `min(v_foot, ramp) == v_foot` exactly (the ramp floor sits
    // strictly above the foot), so the scale below is `1.0` bit-for-bit on a forestless rig.
    let v_allowed =
        governed_ceiling_for_frame(regions, placements, dot.pose.frame, dot.pose.pos, &tuning)
            .unwrap_or(tuning.v_foot_mps);
    let ramp = flight::ramp_cap_mps(
        dot.pose.vel.length(),
        tuning.v_foot_mps,
        tuning.tick_dt_s,
        tuning.tau_s,
    );
    let scale = flight::throttle_axes_scale(axes.length(), tuning.v_foot_mps, v_allowed.min(ramp));
    // OCCUPANT movement runs in the realm's SUBJECTIVE time: `move_speed · dt · time_multiplier`. At the
    // default `1.0` this is byte-identical; a slow-time realm (`< 1.0`) moves its occupants slower.
    // The time multiplier stays OUTSIDE the speed law deliberately (§4.2's own line): the cap means
    // GEOMETRY, the multiplier means SUBJECTIVE TIME — conflating them would make a slow-time zone
    // also a slow-warp zone.
    let step = dot.pose.orient
        * axes
        * (scale * (config.move_speed_mps * config.tick_dt_s * config.time_multiplier));
    // Integrate through the ONE in-frame move (`translated` — displace, then re-bucket): the same
    // two operations in the same order as the former `map_offset(..).normalize(..)` spelling, so
    // this is byte-identical by construction (real-scale addendum §A4.5 — the map_offset foot-gun
    // is deleted; the bounded-offset invariant can no longer be forgotten at a call site).
    // THE FOLD. Adding the step into a raw metre triple is what made motion precision depend on how far
    // you are from the origin: at star-system distances the result snaps to a ~2 mm grid, so walking
    // carries a direction-dependent speed error and anything slower than ~4.9 cm/s never moves at all
    // while still reporting the commanded speed. Folding the leftover into the whole number each tick
    // means every tick only ever adds a ~2 cm step to a sub-millimetre leftover — the error is the same
    // everywhere in the universe.
    dot.pose.pos = dot.pose.pos.translated(step, dot.pose.frame.tier());
    dot.pose.vel = step / config.tick_dt_s;
    dot.pose.universe_tick = clock.universe_tick;
}

impl Dot {
    fn orient_from_angles(&mut self) {
        self.pose.orient = kinematics::orient_from_yaw_pitch(self.yaw, self.pitch);
    }
}
