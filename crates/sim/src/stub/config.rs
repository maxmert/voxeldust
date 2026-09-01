//! THE SHARD'S DIALS, AND EVERY NUMBER DERIVED FROM THEM.
//!
//! Owns: [`StubConfig`] — the one composer-provided parameter block a stub shard is built from —
//! plus [`PlacementCarry`] and the derivations that turn a tick rate and a cadence into the TTLs,
//! windows and budgets the lanes obey. Nothing in this crate writes an operational literal at a use
//! site; it asks here, and the answer is always a function of something the deployment already
//! stated (the no-magic-numbers law).
//!
//! Does NOT own: any state, any system, any decision. Nothing here touches a `World`, and no number
//! here is CHOSEN — a value that cannot be derived does not belong in this file.

use bevy_ecs::prelude::Resource;
use std::collections::{BTreeMap, BTreeSet};
use vd_core::flight::FlightTuning;
use vd_core::pose::{FrameRef, RealmId, StampedPose};
use vd_core::realm_coord::RealmCoord;
use vd_core::realm_path::RealmPath;
use vd_core::worldgen::level_of;
use vd_core::{AccountId, NodeId, TickId};

/// HOW FAR A MOVING REALM PLACEMENT MAY BE CARRIED along its own velocity before the row that needed it is
/// OMITTED instead of guessed at — a SIMULATION-AUTHORITY policy, held by the crate that authors the
/// velocity.
///
/// Applying a placement at a row's OWN instant means carrying that realm's placement forward or back from
/// the tick its author stamped it at. That carry is a straight line along the authored velocity, and a
/// straight line is only honest over a short arc of an orbit. Past this many ticks the honest answer is "I
/// do not know where that realm was", and a wrong answer about where a planet was puts the player inside
/// its crust.
///
/// WHERE IT USED TO LIVE, AND WHY THAT WAS WRONG. It was a field on the gateway's `TransportTuning`, set
/// from the gateway bin. The gateway carried placements because it was the party composing every position
/// out of a world-wide placement graph; it composes nothing now — each level applies the ONE placement it
/// authored, in the shard that authored it — so an accuracy bound on somebody else's velocity had no
/// business being a transport parameter. A router cannot decide how far a simulation may extrapolate.
///
/// ⚠ NO CONSUMER TODAY, and that is the honest state rather than an oversight. Every level currently
/// resolves its own child's placement CLOSED-FORM at the row's own instant (`ChildFrame.child_at`,
/// `active_children`, the entity lane's per-batch instant), so nothing linearly carries anything and the
/// cap is unreachable by construction. It is kept, measured and re-derived because the moment any lane
/// stops being able to re-derive a placement at an arbitrary instant — a checkpoint-carried rapier body
/// (Category C), a signal-driven ship — the carry comes back and this is the bound it must respect. The
/// derivation is re-measured every run against the closed-form orbits by
/// `vd-tests`' `frame_conversion_e2e::the_linear_carry_chord_error_sizes_the_placement_skew_cap`; it is
/// not a chosen number and must not become one.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PlacementCarry {
    /// The cap in UNIVERSE TICKS for the cluster this was derived for.
    pub max_skew_ticks: u64,
}

impl PlacementCarry {
    /// The wall-clock budget [`Self::for_tick_rate`] converts into ticks.
    ///
    /// WHAT IT HAS TO ABSORB: the entity lane and the realm lane are separate unreliable datagrams
    /// authored by different shards whose universe clocks advance only when a sync arrives, so rows that
    /// belong to one instant can reach a receiver a few hundred milliseconds apart under load. The carry
    /// exists to put them back on one time axis.
    ///
    /// WHAT BOUNDS IT, and this is MEASURED rather than felt: the carry is a straight line at the authored
    /// velocity, i.e. a chord across an arc, and past some skew that chord stops describing the orbit. The
    /// criterion is that the error the carry introduces over the WHOLE window stays below the distance the
    /// realm itself moves in ONE tick — below that the carry is worth less than the quantisation of the
    /// feed that produced it; above it, it is inventing motion. That criterion needs no invented constant:
    /// it compares the mechanism against its own input.
    ///
    /// MEASURED over every orbiting body the shipped presets generate across sixteen universe seeds,
    /// against the closed-form `orbital_state`. The tightest body is the innermost visual-scale planet
    /// (period 12.4 s, 8.7 m/s), and it admits **13 ticks at 50 Hz (260 ms)** and **8 ticks at 20 Hz
    /// (400 ms)**. The binding number is therefore 260 ms, and this constant is the largest round step
    /// comfortably inside it at every shipped rate. Its predecessor, 500 ms, was an arrival-skew budget
    /// wearing an accuracy label and was nearly twice what the orbits actually allow.
    ///
    /// The gate re-measures on every run and fails if this constant creeps back past the bound, so raising
    /// it means moving the measurement, not the number.
    pub const BUDGET_MS: u64 = 200;

    /// The policy for a cluster running at `tick_hz`. Rounds UP, so a slow tick rate never produces a cap
    /// of zero (which would refuse every moving placement that arrived even one tick late).
    #[must_use]
    pub fn for_tick_rate(tick_hz: u32) -> PlacementCarry {
        PlacementCarry {
            max_skew_ticks: Self::skew_ticks_for(tick_hz),
        }
    }

    /// The cap in ticks alone — the same derivation, for a caller that wants the scalar.
    #[must_use]
    pub fn skew_ticks_for(tick_hz: u32) -> u64 {
        let hz = u64::from(tick_hz.max(1));
        (hz * Self::BUDGET_MS).div_ceil(1_000).max(1)
    }
}

/// Stub-shard configuration (composer-provided; world params seed-derived, no
/// inline literals in systems).
///
/// NOT `Copy` (the `held_realms` `BTreeSet` is heap-backed): it is only ever borrowed
/// (`Res<StubConfig>` / `&StubConfig`), constructed once per shard and moved into the world at
/// `register_stub_shard`. Prefer `StubConfig::single_realm` for the byte-identical single-realm case.
#[derive(Resource, Clone, Debug)]
pub struct StubConfig {
    pub realm: RealmId,
    /// The FULL set of realms this shard HOSTS — its own `realm` PLUS any deeper CHILD realms it
    /// CO-HOSTS (the un-hosted-child cure). A shard's `realm_neighbourhood_for` already includes its
    /// owned children as evaluated regions; co-hosting makes the shard the actual HEAD of those child
    /// realms so a durable dot that walks into a child (e.g. a planet's SOI nested in the shard's
    /// system) RE-HOMES LOCALLY (a realm-label update, no `CrossingRequest`) instead of stranding on an
    /// un-hosted `head(Realm(child))`. Default (`single_realm`) = `{realm}`, so every single-realm shard
    /// and test is byte-identical (the extra-realm grant/affirm/short-circuit paths are all inert when
    /// the set is the lone `realm`).
    pub held_realms: BTreeSet<RealmId>,
    pub frame: FrameRef,
    /// Dot walk speed, meters per second.
    pub move_speed_mps: f64,
    /// Simulation tick length, seconds.
    pub tick_dt_s: f64,
    /// The realm's SUBJECTIVE time factor (D-45(a)): how fast time passes INSIDE this shard's realm vs
    /// universe time. `1.0` = universe rate (default, byte-identical). `< 1.0` DILATES — an occupant who
    /// enters a slow-time realm moves slower (a time-dilation zone); `> 1.0` speeds them up. Applied to
    /// OCCUPANT MOVEMENT ([`integrate`]) — NOT to the realm's own celestial orbit (that is parent-authored
    /// in OBJECTIVE universe time; celestial mechanics are not subjective). Boot-constant now
    /// (`VD_TIME_MULTIPLIER` global default + `VD_REALM_TIME_MULTIPLIER` per-realm override); a DYNAMIC
    /// (runtime-changing, orchestrator-propagated) factor accumulates local time on this same seam later.
    pub time_multiplier: f64,
    /// The orchestrator's node id (directory seam peer).
    pub orchestrator: NodeId,
    /// Seed for the entity-mint entropy tail (NEVER wall-clock — R7).
    pub mint_seed: u64,
    /// Bounded window for the input-conservation log (SCALE-3): production sets a
    /// small ring; the harness sets a large one so the oracle sees a whole run.
    pub input_log_capacity: usize,
    /// How often (ticks) a shard re-reads its realm head to OBSERVE a lost lease
    /// (fence rule 4 self-fence — FENCE-1/5/8). 0 disables (single-shard P1 tests).
    pub realm_recheck_interval: u64,
    /// D-3 lease-liveness heartbeat cadence (the holder's LOCAL ticks): how often the shard re-sends
    /// `LeaseRenew` for its Realm + every granted Entity, keeping leases alive against the orchestrator's
    /// reaper. `0` = INERT (no heartbeat — the pre-D-3 default). The holder's local copy of
    /// `DirectoryTuning::lease_renew_interval_ticks` (set from the same env knob), so the producer gates
    /// on `local_tick` without reaching across the directory seam.
    pub lease_renew_interval_ticks: u64,
    /// Per-datagram byte budget for snapshot partitioning (audit GW-1): a full-world
    /// snapshot is split into chunks each encoding under this, so none exceeds the
    /// QUIC datagram MTU. Operational param (never an inline literal in systems).
    pub snapshot_datagram_budget: usize,
    /// D-3 Slice 5 — the PROACTIVE self-fence grace (the holder's LOCAL ticks). When the realm is held
    /// but its lease has gone un-CONFIRMED for longer than this (`local_tick - last realm-head
    /// round-trip > grace`), the shard hard-stops its own realm authority (fence rule 4) BEFORE the
    /// orchestrator's reassign window opens — the split-brain cure a partitioned holder needs (the
    /// reactive `realm_recheck` reply never arrives under partition). `0` = INERT (the pre-D-3 default).
    /// The holder's local copy of `DirectoryTuning::self_fence_grace_ticks`; the split-brain-safe timing
    /// (`lease_ttl < grace` AND the THETA_MAX-scaled `THETA_MAX*grace < lease_ttl + max_self_fence_grace`, so
    /// `should_reap`'s `ttl+max` reassign horizon outlasts even a throttled self-fence) is enforced
    /// orchestrator-side by `DirectoryTuning::validate`. REQUIRES `realm_recheck_interval > 0` as the confirmation channel —
    /// the timer is inert without it (no round-trip ⇒ no `last_confirmed` ⇒ nothing to measure).
    pub self_fence_grace_ticks: u64,
    /// Slice 3d/3e — the per-shard boundary hysteresis tuning consumed by `evaluate_realm_boundaries`
    /// (`should_commit`'s dwell/cooldown counts + the velocity pad + cell size). Validated once at
    /// `register_stub_shard` (fail-loud like the tick-pair guard). Default [`BoundaryTuning::DEFAULT`].
    /// INERT in production through P3: the trigger is gated on a NON-EMPTY `RealmBoundaries` registry,
    /// which the composer leaves empty (behaviour-identical); only tests populate it.
    pub boundary: vd_core::geometry::BoundaryTuning,
    /// Slice 3d → ARMED by the D-WORLD-2 cure: how long (local ticks) a `RequestInFlight` crossing
    /// latch may stand before `redrive_stranded_crossings` re-emits the SAME `CrossingRequest` (on top
    /// of the POSITIVE saga-terminal clear via `on_saga_demote` / `CrossingAborted`, which covers every
    /// request that actually STARTED a saga — this ttl covers the one that never did: a
    /// delivered-but-UNRESOLVED dest, dropped with no reply). DERIVED, never a literal:
    /// [`crate::saga::derive_request_ttl_ticks`] = `abort_deadline_ticks + POST_COMMIT_STEPS ·
    /// redrive_deadline_ticks + 1` — strictly outlasting the worst HEALTHY saga resolve, so a re-drive
    /// (and the exhaustion abort behind it) never races a live-but-slow saga. Every launcher arms it
    /// (`VD_CROSSING_TTL_TICKS` in the bins; the derivation call in the fixtures); `0` = disarmed
    /// (unit-rig only — a disarmed unresolved-dest drop is the D-WORLD-2 permanent strand).
    pub request_ttl_ticks: u32,
    /// D-WORLD-2 cure — how many ttl re-drives a stranded latch spends before the source declares the
    /// dest UNRESOLVABLE and takes the LOCAL pre-CAS abort ([`abort_crossing_latch`], the same
    /// machinery the wire `CrossingAborted` consumer runs): the latch clears, the attempt bumps (a
    /// later crossing of the same entity mints a fresh id and fires), and the containment cooldown
    /// arms at the abort tick — so a dot PARKED inside an unresolved region re-fires at the bounded
    /// cadence of the EXISTING `should_rehome` `k_dwell` dwell plus the full ttl cycle, never a
    /// per-tick storm, while a GRAZE (already outside) simply continues unharmed. DERIVED, never a
    /// literal: [`crate::saga::derive_crossing_redrive_budget`] = `abort_deadline_ticks /
    /// redrive_deadline_ticks` (the saga's own patience ratio, `.max(1)`). Inert while
    /// `request_ttl_ticks == 0` (the exhaustion path sits behind the ttl arm).
    pub crossing_redrive_budget: u32,
    /// How long (local ticks) a shard keeps acting on behalf of a subject it is handing over — keeping the
    /// destination's children warm and keeping the parent informed — after it stops owning them. `0` =
    /// INERT (every literal in the tree today), which makes the whole hand-off ledger unreachable and the
    /// build byte-identical. The armed budget is DERIVED, never picked, and lands with its consumer.
    pub handoff_hold_ttl_ticks: u32,
    /// The shard's OWN full lifecycle coord (RLM Step 2) — the AoI loop names children via
    /// `own_coord.child(level)` and keys demands on `child.path()`, unbuildable from the lossy `realm`.
    /// Default = the single-realm ROOT coord for `realm` ([`StubConfig::root_coord`], byte-identity: inert
    /// AoI never reads it); a live-AoI (visual) shard's boot sets the FULL seed lineage.
    pub own_coord: RealmCoord,
    /// F7 predictive horizon (ticks): the AoI loop projects `pos + vel·(boot_ticks_p99·tick_dt_s)` so a
    /// fast occupant demands spin-up before it arrives (boot latency masked). `0` ⇒ no predictive term
    /// (default; byte-identity). Never an inline literal.
    pub boot_ticks_p99: u32,
    /// THIS REALM'S OWN stored spawn poses, keyed by account: where an account's avatar stands **measured
    /// from this realm's centre**, in this realm's frame. The shape a durable per-realm pose store holds,
    /// which is what P7 fills this from — a store owned by one realm cannot hold anything else, because a
    /// realm has no way to express a position outside itself.
    ///
    /// It used to hold poses that were ABSOLUTE, in a universe-root frame, loaded from a cluster-wide env
    /// var that every shard read a copy of. [`resolve_spawn_pose`] "converted" them by relabelling, which
    /// planted a player stored above a planet next to that planet's star. Nothing on the boot path fills
    /// this any more: the position a login spawns at now arrives in `AttachSession`, converted by the
    /// gateway, which is the only party holding the whole forest and therefore the only one that CAN
    /// convert. A pose here in the wrong frame is refused and counted, never relabelled.
    ///
    /// DEFAULT EMPTY ⇒ origin-at-rest ⇒ byte-identical to every existing rig. A `BTreeMap` (O(log n)
    /// lookup, scales to a real roster).
    pub spawn_poses: BTreeMap<AccountId, StampedPose>,
}

impl StubConfig {
    /// The default single-realm ROOT coord for `realm` — a one-level lineage (the inert-AoI default; a
    /// live-AoI shard's boot replaces it with the full seed lineage).
    ///
    /// ★ THE "NEVER A SHIP" CAVEAT IS GONE (2026-09-01). This used to carry an `expect` reading "a
    /// shard realm is a seed-lineage realm" — honest while a ship had no level, and a promise that
    /// would have aborted a whole shard the day one did. Every kind resolves now, so there is nothing
    /// left to expect.
    #[must_use]
    pub fn root_coord(realm: RealmId) -> RealmCoord {
        RealmCoord::from_path(RealmPath::from_levels(vec![level_of(realm)]))
            .expect("a one-level path has a leaf")
    }

    /// The `held_realms` for a SINGLE-realm shard: exactly `{realm}`. The default co-hosting set —
    /// every single-realm construction site passes this so its behaviour is byte-identical to the
    /// pre-co-hosting `RealmAuthority(Option<Fence>)` model (the extra-realm grant/affirm/short-circuit
    /// paths are all inert when the set is the lone `realm`).
    #[must_use]
    pub fn single_realm(realm: RealmId) -> BTreeSet<RealmId> {
        BTreeSet::from([realm])
    }

    /// The realms this shard hosts BEYOND its own `realm` — the co-hosted CHILD realms. Empty for a
    /// single-realm shard (`held_realms == {realm}`). The grant/affirm/renewal paths iterate THIS so the
    /// primary `realm` keeps its existing single-realm machinery untouched (byte-identical) and only the
    /// EXTRA realms take the additive co-host path.
    pub fn cohosted_realms(&self) -> impl Iterator<Item = RealmId> + '_ {
        self.held_realms
            .iter()
            .copied()
            .filter(move |r| *r != self.realm)
    }
}

/// The ledger's backward window, in ticks: wide enough that every message-carried instant a lane can
/// legitimately ask for is still retained — the hand-off hold window (a retained ghost's stamp is at
/// most `handoff_hold_ttl_ticks` old) and the relay retain TTL (an observed-interior row is pruned
/// past it), plus ONE tick for the held transients' stamp-lags-by-one shape (they re-advance AFTER
/// the containment scan reads them). Derived, never a magic number.
pub(crate) fn placement_window_ticks(config: &StubConfig) -> u32 {
    u32::try_from(
        u64::from(config.handoff_hold_ttl_ticks)
            .max(retain_ttl_ticks(config))
            .saturating_add(1),
    )
    .unwrap_or(u32::MAX)
}

/// THE SPEED LAW's tuning for THIS shard (real-scale design §4 / addendum §A3, slice S3) —
/// [`vd_core::flight::FlightTuning`] derived from what the shard already holds and nothing else:
/// its foot speed, its tick, its own AoI demand beat and its measured boot p99. ONE derivation;
/// the process gates (`warp_pixels::wake_budget_ticks`) derive the identical τ, so the ramp the
/// integrator flies and the wake budget the gates assert are the same number by construction.
pub(crate) fn flight_tuning(config: &StubConfig) -> FlightTuning {
    FlightTuning::derive(
        config.move_speed_mps,
        config.tick_dt_s,
        aoi_recheck_cadence(config),
        config.boot_ticks_p99,
    )
}

/// THE WINDOW KEEP-ALIVE TTL, a tick COUNT (owner law 3(a), `docs/design/
/// owner_decisions_2026-08-15.md` item 3: retention is FOREVER DERIVED — at least two cadences
/// plus one, never a free literal): [`RETAIN_TTL_CADENCE_BEATS`] beats of the SAME AoI cadence
/// the SL7 bit and the shape lane already beat on ([`aoi_recheck_cadence`] — one cadence source,
/// HR3), plus one tick of intra-tick prune-vs-receive slack. A subscriber re-asserting once per
/// cadence survives ONE lost keep-alive; a dead one expires within ~two beats and its fan dies
/// with it (`docs/design/window_lane.md` §2.3).
pub(crate) fn window_ttl_ticks(config: &StubConfig) -> u64 {
    RETAIN_TTL_CADENCE_BEATS * aoi_recheck_cadence(config) + 1
}

/// The AoI parent-resolution / occupant up-relay cadence — the observation cascade rides it to learn (and keep
/// fresh) its active children. DECOUPLED from the self-fence realm recheck: when that channel is ARMED (`> 0`)
/// it IS the cadence (one HeadRead round-trip serves both, HR3); when it is DISARMED (`0` — the DevTest profile
/// turns the self-fence off) the up-relay STILL must run in demand mode, so it falls back to a tick-DERIVED
/// demand cadence (~half a second at the shard's tick rate) — never a magic literal, never a self-fence
/// dependency. Monomorphic (all branching HERE, HR5), so `parent_headread_due` stays a straight expression.
/// `pub` for ONE reader: the e2e beat-rate gate asserts the bit lands at 1/cadence against THIS expression,
/// so the gate and the production rate cannot drift apart (finding 41).
pub fn aoi_recheck_cadence(config: &StubConfig) -> u64 {
    if config.realm_recheck_interval > 0 {
        config.realm_recheck_interval
    } else {
        (((1.0 / config.tick_dt_s).round() as u64) / 2).max(1)
    }
}

/// The survive-one-lost-datagram floor for the retain TTL (a tick COUNT) every TTL-pruned store shares. The lane
/// is `Unreliable`/`FireAndForget`, so one loss is a one-tick gap; retaining ≥ 2 ticks past `last_seen`
/// bridges it regardless of intra-tick prune-vs-receive order. The cadence-beats term now always sits
/// above it (2 beats + 1 ≥ 3), so this floor is the stated invariant's own name rather than a live
/// bound — kept so the derivation says all three of its reasons. Independent of `GRACE_TICKS_FLOOR`.
const RETAIN_TTL_FLOOR: u64 = 2;

/// How many CADENCE BEATS the retain TTL must span, now that the SL7 bit beats on the AoI cadence
/// rather than per tick (lane cure, finding 41): ≥ 2, so ONE lost beat is bridged by the next — the
/// same one-loss posture [`RETAIN_TTL_FLOOR`] states in ticks. The `+ 1` in [`retain_ttl_ticks`] is the
/// intra-tick prune-vs-receive slack; without it, 2 beats against a TTL of exactly 2 cadences is a
/// zero-slack coincidence, not a derivation (the shipped 50 Hz / recheck-25 profile landed on exactly
/// that equality before this was derived).
pub(crate) const RETAIN_TTL_CADENCE_BEATS: u64 = 2;

/// The ONE retain TTL as a tick COUNT (a duration, not an instant — hence `u64`, not `TickId`): prunes the
/// `ChildLiveness` bits, the observed-interior stores and the foreign-entity holding bay alike.
/// DERIVED — never a literal — as the WIDEST of: the ONE loiter constant ([`WALK_DEMAND_AOI_GRACE_S`],
/// ~1 s, via the SAME converter the region grace uses, so it stays consistent with an armed region's
/// `grace_ticks`); [`RETAIN_TTL_CADENCE_BEATS`] beats of the bit's own cadence plus one tick of slack
/// (the bit beats per cadence, not per tick — finding 41); and [`RETAIN_TTL_FLOOR`]. `max` only ever
/// WIDENS, so every store sharing this TTL keeps at least its previous window.
pub(crate) fn retain_ttl_ticks(config: &StubConfig) -> u64 {
    u64::from(vd_core::worldgen::grace_ticks_from_seconds(
        vd_core::worldgen::WALK_DEMAND_AOI_GRACE_S,
        config.tick_dt_s,
    ))
    .max(RETAIN_TTL_CADENCE_BEATS * aoi_recheck_cadence(config) + 1)
    .max(RETAIN_TTL_FLOOR)
}

/// A retained entry is ALIVE iff its last arrival was no more than `ttl` ticks ago, on the holder's OWN
/// local clock (`saturating_sub` so a clock not yet past `last_seen` reads age 0). Monomorphic (HR5).
pub(crate) fn ttl_alive(last_seen: TickId, now: TickId, ttl: u64) -> bool {
    now.0.saturating_sub(last_seen.0) <= ttl
}
