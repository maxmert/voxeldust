//! StubShard simulation systems: points in empty space (P1–P3 proving ground;
//! `docs/design/sealed_shards.md` §stub, roadmap P1). NO voxels, NO physics, NO
//! rendering — the point of a stub shard is that the CONNECTION and (later)
//! TRANSFER machinery around it is real.
//!
//! Binding behaviors encoded here:
//! - The shard NEVER sees a ticket: it acts on `(SessionId, Fence)` handed over the
//!   gateway↔shard flow, and drops stale-fence input with a logged reason — the
//!   stale-gateway drop branch exists (and is covered) from day one even though it
//!   cannot fire with a single gateway.
//! - Every avatar `EntityId` is minted HERE (`EntityId::pack`, seed-derived entropy,
//!   never time-derived — R7).
//! - Every applied or discarded input lands in [`InputLog`] — the ground truth the
//!   INPUT-CONSERVATION oracle audits against what the fabric delivered.
//! - Frames are emitted only while the shard HOLDS its realm authority (fence
//!   granted via the Directory seam), stamped with that fence.

use std::collections::{BTreeMap, BTreeSet, VecDeque};

use bevy_ecs::prelude::{IntoScheduleConfigs, Res, ResMut, Resource, Schedule, World};
use vd_core::celestial::{OrbitalElements, orbital_state, secs_since_epoch};
use vd_core::collections::DetHashMap;
use vd_core::entity_kind::{DurabilityClass, EntityKind, continuity_of, durability_of};
use vd_core::frame::{FramePlacement, LocalFrames, rebind_pose_to_dest};
use vd_core::geometry::{
    DepthKey, OverlapBand, RealmRegion, container, region_depth, region_signed_distance,
    should_rehome,
};
use vd_core::glam::DVec3;
use vd_core::kinematics;
use vd_core::pose::{FrameRef, RealmId, StampedPose};
use vd_core::realm_coord::RealmCoord;
use vd_core::realm_path::{RealmLevel, RealmPath};
use vd_core::rng::SplitMix64;
use vd_core::worldgen::level_of;
use vd_core::{
    AccountId, EntityId, EpochId, Fence, NodeId, SessionId, TickId, TransferId, UniverseTick,
};
use vd_wire::channels::{
    EntitySnap, InputDatagram, RealmSnap, RealmSnapshotDatagram, SnapshotDatagram, SubId,
    partition_entities, partition_realms,
};
use vd_wire::intershard::{
    CrossingAborted, CrossingRequest, DemandVerb, DemoteCmd, FlushSource, GhostFlow,
    InterShardFlow, PROMOTE_STEP, PromoteCmd, RE_HOME_STEP, ReHomeCmd, ReHomeState, RealmDemand,
    STUB_CROSSING_STEP, TRANSFER_SCHEMA_VERSION, TRANSIENT_ABANDON_STEP, TRANSIENT_BATCH_STEP,
    TRANSIENT_COMPLETE_STEP, TRANSIENT_DISCARD_STEP, TRANSIENT_DROP_STEP, TRANSIENT_RELEASE_STEP,
    TransferAck, TransferEnvelope, TransientCrossingGrant, TransientCrossingRequest,
    TransientHandoff, TransientItem, TransitionPayload, crossing_transfer_id,
};
use vd_wire::seams::directory::{AuthorityRef, DirectoryKey, DirectoryOp, DirectoryReply};
use vd_wire::seams::transfer_control::TransferControlAck;
use vd_wire::session_flow::{GatewayToShard, ShardToGateway};

use crate::authority::{Authority, AuthorityCmd};
use crate::io::{Durability, Inbound, MsgClass};
use crate::runtime::{ClockSample, InboundBox, NodeIdentity, OutboundBox};

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
    /// Slice 3d — how long (ticks) a `RequestInFlight` crossing latch may stand before the geometric
    /// trigger is permitted to re-request (a belt-and-braces bound on top of the POSITIVE saga terminal
    /// clear via `on_saga_demote` / `CrossingAborted`). `0` = INERT (the positive clear is the sole
    /// driver; the pre-3f default and every current rig). Reserved for the 3f abort/TTL egress — carried
    /// now so the config surface is frozen before the consumer lands.
    pub request_ttl_ticks: u32,
    /// The shard's OWN full lifecycle coord (RLM Step 2) — the AoI loop names children via
    /// `own_coord.child(level)` and keys demands on `child.path()`, unbuildable from the lossy `realm`.
    /// Default = the single-realm ROOT coord for `realm` ([`StubConfig::root_coord`], byte-identity: inert
    /// AoI never reads it); a live-AoI (visual) shard's boot sets the FULL seed lineage.
    pub own_coord: RealmCoord,
    /// F7 predictive horizon (ticks): the AoI loop projects `pos + vel·(boot_ticks_p99·tick_dt_s)` so a
    /// fast occupant demands spin-up before it arrives (boot latency masked). `0` ⇒ no predictive term
    /// (default; byte-identity). Never an inline literal.
    pub boot_ticks_p99: u32,
    /// RLM 5f-3b — the per-account STORED spawn poses: a login admits the account's avatar at ITS stored
    /// pose (organic bootstrap), not the origin. Keyed by the [`AccountId`] the `AttachSession` arm already
    /// carries, so the pose loads SHARD-SIDE with NO frozen-wire grow. Each pose is ABSOLUTE — expressed in
    /// the Universe-root frame `container_coord_at` reads (`SystemSpace{system_seed: 0}`, worldgen.rs) — and
    /// [`resolve_spawn_pose`] REBINDS it into THIS shard's realm frame via the ONE frame machinery (HR3)
    /// crossings/re-homes use (`rebind_pose_to_dest`), never a raw-offset replant. DEFAULT EMPTY ⇒
    /// origin-at-rest ⇒ BYTE-IDENTICAL to the pre-5f-3b admit for every existing rig (the `None` arm is the
    /// old literal). A `BTreeMap` (O(log n) lookup, scales to a real roster); a config/env STAND-IN
    /// (`vd_bins::resolve_spawn_poses`) fills it today and the P7 durable per-account pose store swaps in
    /// behind this SAME map with zero caller reshape.
    pub spawn_poses: BTreeMap<AccountId, StampedPose>,
}

impl StubConfig {
    /// The default single-realm ROOT coord for `realm` — a one-level lineage (the inert-AoI default; a
    /// live-AoI shard's boot replaces it with the full seed lineage). `realm` is always a seed-lineage
    /// realm here (never an entity-backed ship), so `level_of` resolves.
    #[must_use]
    pub fn root_coord(realm: RealmId) -> RealmCoord {
        RealmCoord::from_path(RealmPath::from_levels(vec![
            level_of(realm).expect("a shard realm is a seed-lineage realm"),
        ]))
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

/// One connected avatar.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Dot {
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
    /// this entity" — half the `apply_input` gate and the oracle held-set. EMIT-eligibility (1d.5b.3b)
    /// is the strictly-DERIVED `simulates() | is_fed_ghost | is_retained_ghost` (`emit_frames`): a fed
    /// or retained Ghost emits its kinematic mirror to keep the cross-shard handoff seamless but
    /// integrates/accepts NOTHING (FG-2 — emit-eligibility is derived from authority + the feed
    /// registration, never a competing authority store). Login AND the transfer-dest both mint
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
    pub prev_offset: DVec3,
}

/// All avatars on this shard, in deterministic session order. The key set IS the
/// shard's held-set for the AUTHORITY-UNIQUE oracle.
#[derive(Resource, Debug, Default)]
pub struct Dots(pub BTreeMap<SessionId, Dot>);

/// The PRIMARY realm authority this shard holds (`config.realm`'s fence; None until the directory
/// grants it — no frames are emitted unowned). The self-fence / emit-frame / promote-guard machinery
/// all key on THIS, unchanged from the single-realm model. CO-HOSTED child realms (co-hosting) carry
/// their own fences in [`CoHostedAuthority`]; the local re-home short-circuit reads the union via
/// `CrossingCtx::held_here`.
#[derive(Resource, Debug, Default)]
pub struct RealmAuthority(pub Option<Fence>);

/// The fences for the CO-HOSTED CHILD realms this shard hosts beyond its own `config.realm` (the
/// un-hosted-child cure). Populated by the same directory grant/affirm path as `RealmAuthority` but
/// for `config.cohosted_realms()` — a per-realm map so each child's head is affirmed independently.
/// Default EMPTY: a single-realm shard (`held_realms == {realm}`) never populates it, so every
/// single-realm rig is byte-identical (the map is only touched when `cohosted_realms()` is non-empty).
#[derive(Resource, Debug, Default)]
pub struct CoHostedAuthority(pub BTreeMap<RealmId, Fence>);

/// D-3 Slice 5 — the `local_tick` of the last realm-head ROUND-TRIP confirmation (the reply at which
/// the directory affirmed this shard still owns its realm). The partition detector for the proactive
/// self-fence: it advances only when a `realm_recheck` reply lands, so under a partition (the
/// orchestrator unreachable, no reply) it FREEZES while `local_tick` keeps advancing — the gap is the
/// evidence the holder has lost contact. Meaningful only while `RealmAuthority` is `Some` (the
/// self-fence timer guards on that); reset to the current `local_tick` on every (re)confirmation, so a
/// re-granted realm never inherits a stale deadline.
#[derive(Resource, Debug, Default)]
pub struct RealmConfirmedAt(pub TickId);

/// One ghost-neighbor this shard FEEDS (1d.5b.3b): the node hosting a kinematic ghost of an entity
/// we OWN, plus the monotone egress `seq` stamped on each `GhostFlow::Delta`. Holds NO pose/authority
/// — the fed pose is read LIVE from the owned `Dot` each tick (FG-2 single-truth).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GhostNeighbor {
    /// The node hosting the ghost (the transfer SOURCE in 1d.5b.3b; any band-neighbor at P-band).
    pub source: NodeId,
    /// The next `GhostFlow::Delta.seq` to stamp (monotone per neighbor — the lossy-stream cursor).
    pub seq: u64,
    /// The boundary ANCHOR (1d.5b.3c): the fixed pose POSITION at which the entity crossed into this
    /// realm, captured once at promote. The dest measures the owned entity's distance from here each
    /// tick; when it exits the overlap band the dest Despawns the ghost (band-exit). IMMUTABLE
    /// bookkeeping — a reference point, NEVER a live pose (FG-2: the live pose is read from the `Dot`).
    /// Interim: a local-boundary approximation (the stub has no realm-center/SOI geometry); the real
    /// SOI band anchors at the realm center (`for_planet_soi`/`for_system_soi`, P4/P5).
    pub anchor: DVec3,
}

/// DEST-side ghost FEED registry (1d.5b.3b): for each entity this shard OWNS, the ghost-host
/// neighbor(s) to feed `GhostFlow::Delta`. Populated by the relocated promote (`on_saga_promote`)
/// in 1d.5b.3b — the dest, on becoming owner, registers the transfer source as a ghost-host. The
/// SAME machinery serves the future band-driven multi-neighbor ghost (the owner fans `Delta` to
/// every overlap neighbor — the registration generalizes to a neighbor SET, an additive extension).
/// Torn down on band-exit `Despawn` (1d.5b.3c). One entry per owned, ghosted entity.
#[derive(Resource, Debug, Default)]
pub struct GhostColliderRegistration(pub BTreeMap<EntityId, GhostNeighbor>);

/// One fed source-ghost's freshness/dedup state (1d.5b.3b): the feeding owner `from`, the
/// last-seen `Delta.seq` (lossy latest-wins dedup), and `fed` = has-ever-been-fed-and-not-despawned
/// (the `is_fed_ghost` emit-eligibility latch — NOT fed-this-tick, since `GhostDelta` is lossy and a
/// dropped delta must not blink the avatar). Holds NO pose/authority copy — the fed pose is written
/// INTO the retained ghost `Dot.pose` + `AuthorityCmd::GhostRefresh` (FG-2 single-truth).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GhostFeedState {
    /// The owner feeding this ghost (validated on `Despawn`).
    pub from: NodeId,
    /// The highest `Delta.seq` applied — a `seq <= last_seq` redelivery is a counted stale drop.
    pub last_seq: u64,
    /// Has-ever-been-fed-and-not-despawned (the `is_fed_ghost` emit latch).
    pub fed: bool,
}

/// SOURCE-side mirror of the ghosts this shard HOSTS for owners elsewhere (1d.5b.3b): per entity,
/// the feed freshness/dedup state. The retained source `Dot` IS the ghost (1d.4b kept it); this is
/// pure bookkeeping beside it (no second pose/authority store). One entry per hosted ghost.
#[derive(Resource, Debug, Default)]
pub struct SourceGhostMirror(pub BTreeMap<EntityId, GhostFeedState>);

/// One TRANSIENT (debris/projectile) this shard tracks (D-7). Pose-only for D-7a — the ballistic
/// `(pose0, v0)` blob that lets the dest re-advance closed-form is D-7b (`TransientItem.state`). The
/// `Held` subset (Arriving EXCLUDED) is the authoritative ground truth; a transient is NEVER a
/// directory `OwnerRecord` (a 1000-debris burst writes ZERO directory rows — burst isolation, HR2).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Transient {
    pub pose: StampedPose,
    /// The realm-lease fence the set is anchored to (set at adopt = the dest realm fence). The
    /// `TRANSIENT-AUTHORITY-HELD` oracle cross-checks this against the shard's realm fence + a
    /// committed go-token.
    pub anchor_fence: Fence,
    pub status: TransientStatus,
    /// Slice 3d — the frame-local offset at the END of the previous tick, the START endpoint of THIS
    /// tick's swept boundary segment in `evaluate_realm_boundaries` (the transient twin of
    /// `Dot::prev_offset`). Seeded to the pose offset at every construction site so tick-1's segment
    /// is degenerate, then written LAST each evaluation tick — anti-tunneling over the whole segment.
    /// INERT in production through P3 (the boundary registry is EMPTY).
    pub prev_offset: DVec3,
}

/// A transient's lifecycle tier (D-7) — the Held-vs-Arriving split is the transient twin of the
/// durable Owned-vs-Ghost split that makes adopt-before-drop free of a double-held tick.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TransientStatus {
    /// AUTHORITATIVELY held by this shard (COUNTED + rendered). `outbound` is `None` for a settled
    /// transient; it is set to the batch id once the item has been EMITTED in a crossing batch — the
    /// item stays authoritative (adopt-before-drop) until the orchestrator's `TransientRelease` for
    /// that batch flips it to `Departing`.
    Held { outbound: Option<TransferId> },
    /// A SOURCE-side pending crossing (the TEST-seeded boundary-heuristic stand-in — the autonomous
    /// geometric trigger is P4/P5): `emit_transient_batch` drains it into ONE `TransientBatch`
    /// envelope to `dest` and transitions the item to `Held{outbound: Some(batch)}`. The item is
    /// still COUNTED here (it has not left yet).
    Crossing {
        dest: NodeId,
        to_realm: RealmId,
        dst_realm_fence: Fence,
        batch: TransferId,
        /// The dest realm's PARENT provenance — carried from the [`TransientCrossingGrant`] so
        /// `emit_transient_batch`'s `rebind_pose_to_dest` forms an `Area` frame. `None` for a non-Area dest.
        to_parent: Option<RealmId>,
    },
    /// A mid-flight adopted copy at the DEST (UNCOUNTED — the Ghost analogue: excluded from the
    /// conservation count AND from rendering), tagged with its batch. Flips to `Held{outbound: None}`
    /// on the batch's `TransientDrop` (= PROMOTE; the transient twin of the ordered Ghost→Owned).
    Arriving { batch: TransferId },
    /// A SOURCE-side RELEASED copy (D-7b), retained UNCOUNTED and UNRENDERED across the handoff gap:
    /// on `TransientRelease` the source flips `Held{outbound:Some(b)}→Departing{b}` (so it stops both
    /// counting and rendering BEFORE the dest promotes — the holder set is never `{source,dest}`), and
    /// removes it only on `ReleaseComplete` (after the dest's promote-confirm). Retaining it (rather
    /// than removing on release) lets a dest-crash-mid-promote re-drive the promote against a
    /// still-extant copy. The source-side twin of the durable retained Ghost.
    Departing { batch: TransferId },
}

impl TransientStatus {
    /// Is this transient AUTHORITATIVELY held by its shard (COUNTED for conservation + render)? `Held`
    /// and the pre-emit `Crossing` count (the source still owns a crossing item until release);
    /// `Arriving` (dest mid-flight) and `Departing` (source released, retained) do NOT — the two
    /// UNCOUNTED tiers that make the holder set transit `{source}→{}→{dest}`, never `{source,dest}`.
    /// Monomorphic — the SINGLE answer to "does this shard hold this transient", mirroring
    /// `Authority::simulates` for durable dots.
    #[must_use]
    pub fn is_held(&self) -> bool {
        match self {
            TransientStatus::Held { .. } | TransientStatus::Crossing { .. } => true,
            TransientStatus::Arriving { .. } | TransientStatus::Departing { .. } => false,
        }
    }

    /// Is this transient MID-TRANSFER (D-7b.3) — i.e. losing it to a realm self-fence is an in-flight
    /// transfer LOSS (counted against the kind's `LossBudget`), NOT a resident eviction? Everything
    /// EXCEPT a settled `Held{outbound: None}`: a source item flagged-to-cross (`Crossing`), emitted
    /// and awaiting release (`Held{outbound: Some}`), released-and-retained (`Departing`), or a dest
    /// mid-flight copy (`Arriving`). Monomorphic predicate — the SINGLE answer to "is this a handover
    /// loss".
    #[must_use]
    pub fn is_in_handover(&self) -> bool {
        match self {
            TransientStatus::Held { outbound: None } => false,
            TransientStatus::Held { outbound: Some(_) }
            | TransientStatus::Crossing { .. }
            | TransientStatus::Arriving { .. }
            | TransientStatus::Departing { .. } => true,
        }
    }
}

/// The transients this shard tracks, by entity id (D-7). The `is_held()` subset is the
/// `TRANSIENT-AUTHORITY-HELD` oracle ground truth — anchored to the realm-lease fence, never the
/// directory. Sits beside `Dots`/`GhostColliderRegistration` (a sibling held-set, not a fork).
#[derive(Resource, Debug, Default)]
pub struct OwnedTransients(pub BTreeMap<EntityId, Transient>);

// ---------------------------------------------------------------------------
// Slice 3d/3e — the per-shard geometric transfer-TRIGGER state (`evaluate_realm_boundaries`).
// INERT in production through P3: the trigger early-returns on an EMPTY `RealmBoundaries` registry
// (the composer plants none — behaviour-identical), so ALL of the state below stays untouched in a
// real run; the tests populate `RealmBoundaries` to exercise every arm.
// ---------------------------------------------------------------------------

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
    /// `PrepareSubscribe` to the client's gateway).
    pub session: SessionId,
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
    /// SAME `CrossingRequest`. Left `None` on a transient crossing (no latch) and after an abort clear
    /// (the abort resets the whole `CrossingState`). `Option` keeps `LatchedCrossing` (a non-`Default`
    /// payload) off the `Default` path.
    pub latched_crossing: Option<LatchedCrossing>,
}

/// The containment trigger's per-entity cooldown/latch state (task #135). One entry per evaluated
/// subject; lazily evicted when the subject leaves the shard.
#[derive(Resource, Debug, Default)]
pub struct CrossingProgress(pub BTreeMap<EntityId, CrossingState>);

/// Per-entity HYSTERETIC containment membership: bit `ix` = "hysteretic member of `RealmRegions[ix]`"
/// (task #135 §2.3). A fixed-width `u64` (`MAX_REGIONS` = 64: the shard's own realm + its ~4 ancestors +
/// a bounded child set; the HUNDREDS of sibling child realms resolve via the directory, NOT a local bit —
/// the scale answer). `Copy`, ONE word per entity → zero cross-entity contention (the `par_iter`
/// precondition). Each region's bit advances INDEPENDENTLY by its own [`vd_core::geometry::ContainmentBand`]
/// — there is no single-winner slot to corrupt (the old `winner_ix` reset was actively wrong here).
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct RegionMembership {
    bits: u64,
}

impl RegionMembership {
    /// Is the entity a hysteretic member of region `ix`? (Precondition `ix < MAX_REGIONS`.)
    fn get(self, ix: usize) -> bool {
        self.bits & (1u64 << ix) != 0
    }

    /// Set the membership bit for region `ix` (precondition `ix < MAX_REGIONS`).
    fn set(&mut self, ix: usize, member: bool) {
        let mask = 1u64 << ix;
        if member {
            self.bits |= mask;
        } else {
            self.bits &= !mask;
        }
    }
}

/// Per-entity containment-membership bitsets (task #135). Keyed by the subject entity; lazily evicted
/// with the other per-entity ledgers when the subject leaves.
#[derive(Resource, Debug, Default)]
pub struct ContainmentProgress(pub BTreeMap<EntityId, RegionMembership>);

/// Per-CHILD AoI hysteresis + grace (RLM Step 2), keyed by child `RealmPath` (globally unique — Step 3
/// dedups on `child.path()`). Twin of [`ContainmentProgress`], but CHILD-path-keyed (not entity-keyed).
/// `BTreeMap` (no default-hasher HashMap in sim — determinism). Default empty; lazily evicted each tick
/// to the current direct-child roster (`retain_live`).
#[derive(Resource, Debug, Default)]
pub struct AoiMembership(pub BTreeMap<RealmPath, AoiState>);

/// One direct child's AoI state: `was_in` (acquired — for the hysteresis) + `grace_remaining` (ticks a
/// would-be release is held after the last in-range observation).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct AoiState {
    was_in: bool,
    grace_remaining: u32,
}

/// The per-entity DURABLE-crossing latch (Slice 3d): a durable entity that has emitted a
/// `CrossingRequest` is latched here (keyed by subject entity → the deterministic
/// [`crossing_transfer_id`]) so the trigger emits EXACTLY ONE request per crossing. Cleared POSITIVELY
/// by the saga terminal — `on_saga_demote` on a durable COMMIT, or `CrossingAborted` on a pre-CAS abort
/// (the 3f abort egress) — never by a bounded TTL window. Lazily evicted when the subject leaves.
#[derive(Resource, Debug, Default)]
pub struct RequestInFlight(pub BTreeMap<EntityId, TransferId>);

/// The bitset width bound: a shard's region set exceeding this fails LOUD at boot (`guard_regions_nest`,
/// C-5). The scale answer is NOT a wider bitset — it is the own-realm + ~4-ancestor scoping (children via
/// the directory, §2.3), so 64 is generous headroom, not a ceiling on how crowded a realm can be.
pub const MAX_REGIONS: usize = 64;

/// The seed-derived realm REGIONS this shard evaluates CONTAINMENT against (task #135): the shard's own
/// realm + its ancestor chain (+ a bounded child set). Holds the regions, their BOOT-COMPUTED depth keys
/// (so the per-tick `container` fold never re-walks parents — O(entities × regions), not O(N·M²)), and the
/// ambient-ROOT realm (the `parent: None` region — the `container` fold identity). DEFAULT EMPTY → the
/// detector early-returns (INERT in production through C-3; the seed boot-population + `guard_regions_nest`
/// land C-5/C-6). Replaces the directional `RealmBoundaries` portal registry.
#[derive(Resource, Debug, Default)]
pub struct RealmRegions {
    regions: Vec<RealmRegion>,
    depths: Vec<DepthKey>,
    root_realm: Option<RealmId>,
    /// The shard's DIRECT MOVING children (FA-2b): realm → its `OrbitalElements`. A region in this map
    /// is AUTHORED live each tick from its ephemeris (`frame_context` registers it `with_moving_child`);
    /// a region absent from it rides its static `center` at the identity placement. EMPTY at walk/static
    /// scale (`worldgen::moving_children_for` returns none — all `StaticOffset`), so the frame context is
    /// byte-identical to FA-1; the canonical seed generation (P4/FA-5) is what populates it.
    moving: BTreeMap<RealmId, OrbitalElements>,
}

impl RealmRegions {
    /// Build the resource from a region forest, computing the depth-key cache + the ambient-root realm
    /// ONCE (regions are static at P3). The per-tick detector reads the cache; it never re-walks parents.
    /// The moving-child roster starts EMPTY — [`with_moving_children`](Self::with_moving_children) adds it.
    #[must_use]
    pub fn new(regions: Vec<RealmRegion>) -> RealmRegions {
        let depths = regions
            .iter()
            .enumerate()
            .map(|(ix, r)| (region_depth(&regions, r.realm), r.realm, ix))
            .collect();
        let root_realm = regions.iter().find(|r| r.parent.is_none()).map(|r| r.realm);
        RealmRegions {
            regions,
            depths,
            root_realm,
            moving: BTreeMap::new(),
        }
    }

    /// Register the shard's DIRECT MOVING children (FA-2b): the `(realm, elements)` roster
    /// `worldgen::moving_children_for` derives for the hosted realm. A builder (not a `new` arg) so the
    /// many `RealmRegions::new` call sites stay unchanged and byte-identical (the walk roster passes an
    /// empty map). Only the frames of registered realms author live; every other region stays static.
    #[must_use]
    pub fn with_moving_children(
        mut self,
        moving: BTreeMap<RealmId, OrbitalElements>,
    ) -> RealmRegions {
        self.moving = moving;
        self
    }

    /// The detector short-circuits (inert) when no regions are planted — production through C-3.
    fn is_empty(&self) -> bool {
        self.regions.is_empty()
    }

    /// The per-shard ephemeris [`FrameContext`] (D-45(a) frame-authority FA-1/FA-2b). A region in the
    /// [`moving`](Self::moving) roster is AUTHORED live from its `OrbitalElements` each tick
    /// (`with_moving_child` — `placement()` re-derives its pose from `tick`); every other region rides
    /// its static `center` at the identity placement (`with_placed`). At walk/static scale the moving
    /// roster is EMPTY, so every region takes the identity arm ⇒ byte-equivalent to [`IdentityFrames`]
    /// for the container decision (the FA-1 seam-swap regression gate). `own` is the ambient-root
    /// region's frame (defaulting to `GalaxySpace` for an empty forest, which the detector never
    /// evaluates — it short-circuits on `is_empty`).
    #[must_use]
    pub fn frame_context(&self, tick_hz: f64) -> LocalFrames {
        let own = self
            .regions
            .iter()
            .find(|r| r.parent.is_none())
            .map_or(FrameRef::GalaxySpace, |r| r.frame);
        let mut ctx = LocalFrames::new(own, tick_hz);
        for r in &self.regions {
            ctx = match self.moving.get(&r.realm) {
                Some(elements) => ctx.with_moving_child(r.frame, *elements),
                None => ctx.with_placed(r.frame, FramePlacement::identity()),
            };
        }
        ctx
    }

    /// The shard's authored placements for its MOVING children as [`RealmSnap`] observer rows (FA-2c) —
    /// each computed LIVE from its `OrbitalElements` at `tick` (the closed-form ephemeris `frame_context`
    /// would derive; a passive orbiting body is LAW-1's zero-signal case). The pose is stamped in the
    /// shard's OWN (ambient-root) frame — the parent authors its children THERE. BRANCHLESS + EMPTY at
    /// walk/static scale (`moving` is empty ⇒ no rows ⇒ `emit_realm_frames` sends nothing ⇒ byte-identical).
    /// Authored (signal-driven, `with_placed`) children join here when they exist (P6/P9); today only the
    /// orbital roster moves.
    #[must_use]
    pub fn authored_realm_snaps(
        &self,
        own_realm: RealmId,
        tick_hz: f64,
        tick: UniverseTick,
    ) -> Vec<RealmSnap> {
        // Movers-only view of the unified placements (RLM Step 2, H2): the observer feed ships only the
        // moving children (a static child rides its `center`, not a live snap). At walk/static scale
        // `moving` is empty ⇒ this drops every row ⇒ EMPTY ⇒ byte-identical to the pre-refactor feed.
        self.child_placements(own_realm, tick_hz, tick)
            .into_iter()
            .filter(|(r, _)| self.moving.contains_key(&r.realm))
            .map(|(r, pose)| RealmSnap {
                realm: r.realm,
                pose,
            })
            .collect()
    }

    /// The UNIFIED per-tick placement of EVERY DIRECT child (RLM Step 2, H2): a mover authored from its
    /// `OrbitalElements`, a static child at its region `center`. ONE position code-path both the observer
    /// feed AND the AoI loop consume — no third position path. DIRECT children only (`parent == own`);
    /// ancestor/self/root regions excluded. Poses stamped in the shard's OWN (ambient-root) frame — the
    /// SAME frame `authored_realm_snaps` used, so the AoI distance and the feed measure one geometry
    /// (H-1). Returns `(&RealmRegion, StampedPose)` so callers read extent/aoi/parent without a re-scan.
    #[must_use]
    pub fn child_placements(
        &self,
        own_realm: RealmId,
        tick_hz: f64,
        tick: UniverseTick,
    ) -> Vec<(&RealmRegion, StampedPose)> {
        // Two DIFFERENT anchors: the pose FRAME is the ambient-ROOT frame (`parent: None` — the parent
        // authors its children THERE), while CHILD SELECTION is by the shard's OWN realm (`own_realm ==
        // config.realm`). They are NOT the same region — a forest nests own several levels below the root
        // (the escape/undock 3-level topology), so filtering by the root would drop every real child.
        let own_frame = self
            .regions
            .iter()
            .find(|r| r.parent.is_none())
            .map_or(FrameRef::GalaxySpace, |r| r.frame);
        let secs = secs_since_epoch(tick.0, tick_hz);
        self.regions
            .iter()
            .filter(|r| is_direct_child(r.parent, own_realm))
            .map(|r| (r, place_child(&self.moving, r, own_frame, secs, tick)))
            .collect()
    }
}

/// A region is a DIRECT child iff its parent IS the shard's own realm. A branchless equality (HR5): the
/// `==` is covered true (a child) and false (the root's `None`, an ancestor, a sibling) by any nested
/// forest.
fn is_direct_child(region_parent: Option<RealmId>, own_realm: RealmId) -> bool {
    region_parent == Some(own_realm)
}

/// Place ONE child (RLM Step 2, H2): a mover from its ephemeris (`orbital_state`), a static child at its
/// `center` (both in the parent's OWN frame — cell-0 identity through P3). The `match` is covered once
/// here (a mover test + a static test), NOT per generic monomorphization (HR5).
fn place_child(
    moving: &BTreeMap<RealmId, OrbitalElements>,
    r: &RealmRegion,
    own: FrameRef,
    secs: f64,
    tick: UniverseTick,
) -> StampedPose {
    match moving.get(&r.realm) {
        Some(elements) => {
            let st = orbital_state(elements, secs);
            let mut p = StampedPose::at_rest(own, st.position, tick);
            p.vel = st.velocity;
            p
        }
        None => StampedPose::at_rest(own, r.center.offset(), tick),
    }
}

/// Entity minting state: a per-shard monotonic sequence + seed-derived entropy.
#[derive(Resource, Debug)]
pub struct EntityMint {
    seq: u64,
    rng: SplitMix64,
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

    fn record_applied(&mut self, session: SessionId, seq: u64) {
        self.applied_total += 1;
        if self.applied.len() >= self.capacity {
            self.applied.pop_front();
            self.window_evictions += 1;
        }
        self.applied.push_back((session, seq));
    }

    fn record_discarded(&mut self, session: SessionId, seq: Option<u64>, reason: DiscardReason) {
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

/// Per-shard monotonic REALM-snapshot frame counter (FA-2c) — the realm observer feed's own `frame_id`,
/// independent of the entity [`FrameCounter`] so the client's per-feed staleness gates never cross.
#[derive(Resource, Debug, Default)]
pub struct RealmFrameCounter(pub u64);

/// Counters for conditions that are tolerated but must never be silent.
#[derive(Resource, Debug, Default, PartialEq, Eq)]
pub struct StubStats {
    /// Attach requests that arrived before the realm lease was granted; the
    /// gateway retries attach until it sees `SessionAttached` (at-least-once).
    pub attaches_deferred: u64,
    /// Inter-shard frames that failed to decode at ingress (a malformed/garbage
    /// gateway→shard or directory-reply payload). Tolerated — the frame is dropped,
    /// never mis-applied — but counted so a decode regression is observable rather
    /// than log-only (ROB-E2E-1; mirrors the gateway's `undecodable`). 0 in any
    /// healthy run.
    pub undecodable: u64,
    /// `OpenInputSlot` arrived before this shard holds its realm lease — dropped + counted.
    /// **1c.5 has NO re-drive**: the gateway emits `OpenInputSlot` exactly once (at
    /// `apply_commit`) and the saga is forward-only past `Committed`, so a deferred slot means
    /// the post-marker buffer the gateway already take-drained is PERMANENTLY lost — the
    /// re-drive / durable-backstop producer is owed 1d/P3 (DEFERRED D-8 + the 1c.7 conservation
    /// gate). 0 in a healthy single-process run where the dest realm lease precedes the commit.
    pub input_slots_deferred: u64,
    /// `OpenInputSlot` carrying a fence BELOW the dot's session fence — a replay or a
    /// partitioned old gateway; dropped + counted, input never re-armed (the day-one
    /// stale-gateway-drop rule, `wire::session_flow`). 0 in a healthy run.
    pub input_slots_stale: u64,
    /// `OpenInputSlot` carrying a NON-Entity transfer subject (1c.8): the dest only ADOPTS an
    /// `Entity` transfer, so a Realm/Session/Ship subject is a counted no-op (no extraction
    /// panic, no adopt). 0 in a healthy Entity-transfer run.
    pub input_slots_malformed: u64,
    /// The `resume_from_seq` of the most recent HONORED `OpenInputSlot` — the watermark this
    /// shard last opened a transfer-dest input slot at. Observability latch (NOT a counter): it
    /// makes the gateway-EMITTED resume value visible to the cross-cut conservation gate (the
    /// emitted value must equal the client's own CUT_MARKER seq, threaded through the real saga —
    /// D-28), and is the operational answer to "what seq did this shard resume a handed-off
    /// session at". `None` until the first honored slot.
    pub last_input_slot_resume: Option<u64>,
    /// 1d.1 entity-state crossings APPLIED (`StubCrossing` pose stored on the adopted dot, first
    /// delivery of a `(transfer, step)`). The headline 1d.1 counter.
    pub crossings_applied: u64,
    /// Crossings BUFFERED because the adopt grant had not flipped yet — drained + applied at the
    /// flip (the crossing, emitted at CAS, races ahead of the dest's adopt under the 1c.8
    /// promote-before-demote model). 0 only if every crossing arrived after adopt.
    pub crossings_buffered: u64,
    /// Crossings DROPPED for a fence below the dot's recorded authority fence (fence rule 1 — a
    /// stale leftover from a superseded transfer). 0 in a healthy single-transfer run.
    pub crossings_stale: u64,
    /// `Transfer` envelopes whose payload kind 1d.1 does not consume (`InitialSpawn` /
    /// `TransientBatch`) — a counted no-op, never a panic. 0 in a 1d.1 crossing run.
    pub crossings_unhandled: u64,
    /// `Transfer` envelopes carrying a `universe_epoch` that does NOT match this shard's current
    /// clock epoch — REFUSED at ingress (never applied, never buffered, never acked). The
    /// transfer_protocol §3.3 fail-safe: "a leg whose epoch_id mismatches the current epoch is
    /// discarded, not resumed — no entity placed at a stale celestial position" (the R6/R7
    /// epoch-reset class the rebuild exists to kill). The source stamps `clock.epoch` on every
    /// envelope; pre-this-gate the field was carried-but-unread at the receiver. Mirrors the
    /// follower clock's `epoch_mismatches` counted-ignore. 0 in any single-epoch run (P3 runs one
    /// persisted orchestrator epoch that never resets mid-run); becomes load-bearing once a clean
    /// re-genesis or a delayed redelivery (the owed redelivering transport) can carry a prior-epoch
    /// envelope across a restart. `schema_version` version-floor validation rides the TLV-blob
    /// handshake owed at D-31 (a distinct, deliberately-deferred concern).
    pub crossings_epoch_mismatch: u64,
    /// SOURCE self-fence redeliveries that found the dot ALREADY demoted to a Ghost — a counted
    /// no-op (the idempotency guard's taken arm). 1d.4b retains the source as a Ghost instead of
    /// `dots.remove`, so a second foreign-owner reply must NOT re-demote; this proves the guard.
    pub self_fence_skipped: u64,
    /// SOURCE saga-pushed `Demote` (1d.5b.1) received for a NON-Entity subject (Realm/Session/Ship)
    /// — there is no local Entity dot to demote, so the flip is skipped (the `DemoteAck` is still
    /// sent). 0 in a healthy Entity-transfer run (the stub's transfer subject is always an Entity).
    pub saga_demote_no_entity: u64,
    /// DEST saga-pushed `Promote` (1d.5b.3b) received and APPLIED for the first time — the headline
    /// ordered-promote counter. 1d.5b.3b RELOCATED the real `Ghost→Owned` flip here (out of
    /// `apply_crossing`), so this handler does the flip + announces the dest sub + registers the
    /// ghost feed + acks `PromoteAck`.
    pub promotes_confirmed: u64,
    /// DEST saga-pushed `Promote` REDELIVERIES (already-journaled `(transfer, PROMOTE_STEP)`) — a
    /// counted re-ack-only no-op (at-least-once delivery). 0 in a healthy single-delivery run.
    pub promotes_redelivered: u64,
    /// DEST `Promote` (1d.5b.3b) that found NO dot for the subject entity — a counted no-op (still
    /// acks). 0 in a healthy run (the crossing creates the dest dot before the Promote round-trip).
    pub promote_no_dot: u64,
    /// DEST `Promote` (1d.5b.3b) whose crossing pose has NOT yet landed (`STUB_CROSSING_STEP` not
    /// journaled): the flip is SKIPPED (pose-before-promote — never a poseless origin frame); the
    /// saga `Promoting`-timeout re-emits. 0 in the happy path (the crossing precedes the Promote).
    pub promote_before_crossing: u64,
    /// DEST `Promote` that arrived while this shard does NOT hold its realm (a realm self-fence raced
    /// the Promote) — DROPPED as a counted no-op (degrade, never panic), the saga re-drives. 0 in P2
    /// (no realm-revoke producer); reachable only at P8/P10 realm mobility (the re-drive is owed).
    pub promote_without_realm: u64,
    /// D-37 forward re-home ADOPTS: a `ReHome` CREATED an Owned dot at the target from `ReHomeState`
    /// (no pre-existing ghost — the target is fresh, unlike a `Promote` flip). `> 0` proves CELL-2
    /// recovery actually adopted the re-homed entity.
    pub re_home_adopted: u64,
    /// `ReHome` REDELIVERIES (already-journaled `(transfer, RE_HOME_STEP)`) — a counted re-ack-only no-op
    /// (at-least-once). 0 in a healthy single-delivery run.
    pub re_home_redelivered: u64,
    /// `ReHome` for a non-Entity subject (no `transfer_subject_entity`) — a counted no-op (still acks).
    /// 0 in P3 (CELL-2 re-homes an Entity); reachable when Realm/Ship re-home lands (Slice 4).
    pub re_home_no_entity: u64,
    /// `ReHome` that arrived while this target does NOT hold its realm — DROPPED as a counted no-op
    /// (degrade, never panic; the re-home target normally holds its realm, committed by the orchestrator).
    pub re_home_without_realm: u64,
    /// `ReHome` adopt carrying a `universe_epoch` that does NOT match this shard's current clock epoch —
    /// REFUSED at ingress (never journaled/adopted/acked). The transfer_protocol §3.3 fail-safe applied
    /// UNIFORMLY to the SECOND pose-placing ingress (the first is the crossing →
    /// [`StubStats::crossings_epoch_mismatch`]); mirrors it exactly. 0 in any single-epoch run (P3 runs one
    /// persisted orchestrator epoch that never resets mid-run); load-bearing once a clean re-genesis or a
    /// delayed redelivery (the owed redelivering transport) can carry a prior-epoch `ReHomeCmd` across a
    /// restart — exactly the player/ship re-home + P7 checkpoint-reload paths that must never place an
    /// entity at a stale celestial position.
    pub re_home_epoch_mismatch: u64,
    /// SOURCE `GhostFlow::Delta` (1d.5b.3b) APPLIED — the fed pose + `GhostRefresh` written into the
    /// retained ghost dot. The headline ghost-feed counter.
    pub ghost_delta_applied: u64,
    /// SOURCE `GhostFlow::Delta` DROPPED as stale — `seq <= last_seq` (lossy latest-wins dedup) or
    /// no mirror entry for the entity. Expected under datagram reorder/loss; 0 in lockstep.
    pub ghost_delta_stale: u64,
    /// SOURCE `GhostFlow::Delta` whose `source_fence` was STALE against the ghost dot (the
    /// `GhostRefresh` `is_stale_against` guard rejected it) — a counted no-op. 0 in a healthy run.
    pub ghost_refresh_stale: u64,
    /// SOURCE `GhostFlow::Despawn` (1d.5b.3c band-exit) received and TORN DOWN — the `SourceGhostMirror`
    /// entry + the retained ghost `Dot` removed (the ghost lifecycle ENDS; the source stops self-emitting
    /// + stops being a collider). The headline band-exit teardown counter.
    pub ghost_despawns: u64,
    /// SOURCE `GhostFlow::Despawn` that tore down NOTHING (1d.5b.3c) — no mirror entry AND no retained
    /// ghost dot to remove: an at-least-once REDELIVERY after teardown, or a stale Despawn for an entity
    /// the source has since RE-OWNED (the `Owned` dot is structurally refused — only a retained Ghost is
    /// torn down). A counted idempotent no-op, never a panic. 0 in a healthy single-delivery run.
    pub ghost_despawn_no_host: u64,
    /// DEST band-exit (1d.5b.3c): the owned entity left the overlap band, so the dest emitted
    /// `GhostFlow::Despawn` to the ghost-host and DEREGISTERED the feed (`GhostColliderRegistration`).
    /// The headline band-exit DETECTION counter (the dest half of the source's `ghost_despawns`).
    pub ghost_band_exits: u64,
    /// DEST feed pass skipped a registration whose entity is NOT currently owned here (no dot, or a
    /// non-`simulates()` dot) — a counted no-op (only an Owned dot's live pose is fed). 0 steady-state.
    pub ghost_feed_skipped: u64,
    /// SOURCE: `TransientBatch` envelopes EMITTED (D-7) — one per (dest realm, tick) batch regardless
    /// of item count (G-TIER). The headline transient-egress counter.
    pub transients_emitted: u64,
    /// DEST: transient items ADOPTED as `Arriving` on the FIRST delivery of a `TransientBatch` (the
    /// uncounted mid-flight tier). The headline transient-adopt counter.
    pub transients_adopted: u64,
    /// DEST: `TransientBatch` REDELIVERIES (already-journaled `(transfer, TRANSIENT_BATCH_STEP)`) — a
    /// counted re-ack-only no-op (at-least-once). 0 in a healthy single-delivery run.
    pub transients_adopt_redelivered: u64,
    /// DEST: `Arriving→Held` PROMOTIONS on a `TransientDrop` (D-7b: the transient twin of Ghost→Owned,
    /// reachable only after the source released). The headline transient-promote counter.
    pub transients_promoted: u64,
    /// SOURCE: `Held{outbound}` items RELEASED to `Departing` on a `TransientRelease` (D-7b: the source
    /// goes uncounted BEFORE the dest promotes — the clean hand-off, NOT a loss). The source half of
    /// the structural drop-before-promote.
    pub transients_handed_off: u64,
    /// `TransientDrop` (promote) REDELIVERIES (already-journaled `(transfer, TRANSIENT_DROP_STEP)`) — a
    /// counted idempotent no-op (at-least-once). 0 in a healthy single-delivery run.
    pub transient_drop_noop: u64,
    /// `TransientRelease` + `ReleaseComplete` REDELIVERIES (already-journaled
    /// `(transfer, TRANSIENT_RELEASE_STEP)`) — a counted idempotent no-op (at-least-once). 0 healthy.
    pub transient_release_noop: u64,
    /// GROSS transients DROPPED on a realm self-fence — EVERY tier (handover + settled-resident). Ops
    /// visibility only; 0 on the happy path. NOT the loss-budget gate (that reads
    /// `transients_lost_in_handover` — a 1000-resident burst eviction would dwarf a tiny budget here).
    pub transients_dropped: u64,
    /// D-3 Slice 5 — PROACTIVE self-fences: the holder dropped its realm authority because the lease
    /// went un-confirmed past `self_fence_grace_ticks` (a detected partition from the orchestrator). Ops
    /// visibility / partition signal; `0` on the happy path and inert (`self_fence_grace_ticks == 0`).
    pub realm_self_fenced_lapsed: u64,
    /// HANDOVER-attributable LOSS per kind (D-7b.3): on a realm self-fence, ONLY transients in a
    /// handover status (`is_in_handover()` — emitted/Departing/Crossing source items + Arriving dest
    /// items) are an in-flight transfer LOSS bucketed here; a settled `Held{outbound:None}` dropped on
    /// eviction is a resident-eviction event, OUT of transfer-budget scope. The `verify_transient_
    /// loss_budget` gate compares this per-kind count to the kind's `LossBudget` — so a mixed-kind
    /// burst can never spuriously trip a single-kind budget. `DetHashMap` (fixed-seed → deterministic).
    pub transients_lost_in_handover: DetHashMap<EntityKind, u64>,
    /// D-7d — transients dropped by `on_transient_abandon` (the dead-DEST resolution: the dest died
    /// mid-handoff, so the source's retained copy is dropped as an ACCOUNTED loss). 0 on the happy path
    /// and on a dead-SOURCE resolution (which self-promotes, no loss); `> 0` is the DEST-kill cell's
    /// anti-vacuity proof. Each increment ALSO feeds `transients_lost_in_handover` (the SAME budget the
    /// realm self-fence feeds, DRY) — so `verify_transient_loss_budget` sees a deterministic, named loss.
    pub transients_departure_cancelled: u64,
    /// R-6d3c — transients DISCARDED at the DEST by `on_transient_discard` (the source died in
    /// `BatchHandoff::AwaitAdopt` PRE-adopt, so a late-replayed `Arriving` copy is removed as an ACCOUNTED
    /// loss + the adopt is poisoned). 0 on every happy path and on a RESTART recovery (which adopts
    /// normally); `> 0` is the never-restart cell's anti-vacuity proof. Each increment WITH A DECODABLE
    /// kind also feeds `transients_lost_in_handover`; a corrupt-tag item is removed + counted HERE but NOT
    /// attributed (the `from_tag` Err arm — HR2, never decode-to-default).
    pub transients_discarded_source_crash: u64,
    /// CA-1 S3/S4 — DEST: `BatchAdopted` acks RE-DRIVEN by `redrive_pending_adoptions` (one per DISTINCT
    /// `Arriving` batch per tick). The LIVENESS half of the over-discard guarantee: it keeps re-presenting
    /// the adopt evidence to the orchestrator's `dest_adopted` latch so a transiently-lost single ack never
    /// strands the batch. `> 0` whenever a batch sits `Arriving` for more than the initial adopt tick.
    pub batch_adopts_redriven: u64,
    /// CA-1 S3 — SOURCE: `ReSolicitBatch` liveness probes RECEIVED (a counted NO-OP — the probe's signal is
    /// its SEND outcome at the orchestrator, not this handler; a LIVE source simply acknowledges receipt by
    /// existing). Ops visibility that the AwaitAdopt probe reached a live source.
    pub re_solicits_received: u64,
    /// Slice 3e — DURABLE geometric crossings REQUESTED: `evaluate_realm_boundaries` committed a durable
    /// entity across an `Authority` boundary and emitted ONE `CrossingRequest` (latched in
    /// `RequestInFlight`). The headline durable-trigger counter. `0` in prod through P3 (empty registry).
    pub crossings_requested: u64,
    /// Slice 3e — durable crossings SUPPRESSED because the subject was already latched in-flight (a second
    /// `should_commit` edge before the saga terminal cleared the latch). Proves exactly-one-request. `0`
    /// steady-state.
    pub crossings_suppressed_in_flight: u64,
    /// Slice 3f-D4 — a STRANDED durable latch RE-DRIVEN by `redrive_stranded_crossings`: a
    /// delivered-but-unresolved dest (`head(Realm(to))` transiently absent at P4/P5) left the
    /// `RequestInFlight` latch standing with no rising edge to re-emit it, so the per-tick latch-scan
    /// re-emitted the SAME `CrossingRequest` once `local_tick - last_commit_tick >= request_ttl_ticks`.
    /// `0` while `request_ttl_ticks == 0` (the INERT default / every current rig) and on every happy path
    /// (the saga terminal clears the latch before the ttl elapses).
    pub crossings_redriven: u64,
    /// Slice 3e (robustness, goal-audit L4) — a Durable-TAGGED subject reached the durable crossing arm from
    /// the held-transient loop (a kind/loop mismatch — e.g. a mis-tagged batch item) and so carried no
    /// session: DEGRADED (counted, emitted nothing) instead of panicking. Must be `0` in a well-formed mesh;
    /// a non-zero value flags a mis-tagged transient batch from a peer.
    pub crossing_durable_no_session: u64,
    /// Slice 3e — TRANSIENT geometric crossings REQUESTED: a transient committed across an `Authority`
    /// boundary and emitted a `TransientCrossingRequest` (the batched-grant path). `0` in prod through P3.
    pub transient_crossings_requested: u64,
    /// Slice 3e — `RequestInFlight` latches CLEARED by a saga terminal at the SOURCE: `on_saga_demote` on a
    /// durable COMMIT, or a `CrossingAborted` demux on a pre-CAS abort. Proves the POSITIVE (never-TTL)
    /// clear. `0` until a triggered crossing resolves.
    pub crossing_latches_cleared: u64,
    /// Slice 3e — SOURCE `TransientCrossingGrant`s APPLIED: a granted transient flipped `Held → Crossing`
    /// so `emit_transient_batch` ships it. `0` in prod through P3 (no transient crossings triggered).
    pub transient_grants_applied: u64,
    /// Slice 3e — `TransientCrossingGrant` counted NO-OPS: an unknown transient OR one already flipped (a
    /// redelivery after the flip). `0` in a healthy single-delivery run.
    pub transient_grant_noop: u64,
    /// Slice 3e — `TransientCrossingGrant` for a NON-Entity subject (Realm/Session/Ship) — a counted no-op.
    /// `0` in a healthy run (a transient crossing subject is always an Entity).
    pub transient_grant_no_entity: u64,
    /// Slice 3e — `CrossingAborted` for a NON-Entity subject — a counted no-op. `0` in a healthy run.
    pub crossing_abort_no_entity: u64,
    /// Slice 3e — `CrossingAborted` whose transfer id did NOT match the subject's current latch (a stale
    /// abort for a superseded / re-latched crossing) — a counted no-op, the latch is preserved. `0` healthy.
    pub crossing_abort_stale: u64,
}

/// The outcome of journaling one transferred-entity-state step (1d.0).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StepOutcome {
    /// This `(transfer, step_id)` was not yet applied — the caller MUST apply the effect.
    FirstApply,
    /// A redelivery of an already-applied step — the caller MUST NOT re-apply (re-ack only).
    AlreadyApplied,
}

/// The dest shard's idempotency journal for transferred-entity-STATE steps (1d.0): the IN-MEMORY
/// backing of the frozen `IdempotencyKey::TransferStep{transfer, step_id}` dedup. It keys on the
/// `(TransferId, step_id)` tuple — the SAME key the gateway's per-session RAM journal uses
/// (`gateway.rs` `recorded`/`journal`) and the durable redb `applied_steps` table will use at P3
/// (HR3 ONE machinery, many stores; the store differs per altitude, the KEY and the
/// consult-before-effect / record-after-effect discipline do not — DEFERRED D-22). It is NEVER a
/// dest-local key.
///
/// 1d.0 lands ONLY this primitive (NO Transfer-arm receiver yet). The 1d.1 receiver consults it
/// BEFORE applying a `StubCrossing` step and records AFTER — the ordering-discipline gate is 1d.1;
/// here we only prove the primitive is idempotent. (An in-mem backing cannot crash mid-step — the
/// durable crash window is a P3 concern, D-22.)
///
/// OWED (1d.1, when the receiver feeds this): a RETENTION BOUND — drop a transfer's steps on its
/// terminal (as the gateway RAM journal does, dropped-whole on terminal/`Bye`), so a long-lived
/// dest shard does not accumulate one entry per `(transfer, step)` forever. The set cannot grow in
/// 1d.0 (nothing journals into it yet); the bound belongs with the receiver that knows terminality.
#[derive(Resource, Debug, Default)]
pub struct AppliedSteps(BTreeSet<(TransferId, u32)>);

impl AppliedSteps {
    /// Journal one transfer step by its `IdempotencyKey::TransferStep` components (passed as the
    /// canonical `(transfer, step_id)`, never a dest-local id). Idempotent: the FIRST call records
    /// and returns [`StepOutcome::FirstApply`]; every redelivery of the same key returns
    /// [`StepOutcome::AlreadyApplied`] WITHOUT re-effect. A distinct `step_id` or `transfer` is
    /// independent.
    pub fn journal_step(&mut self, transfer: TransferId, step_id: u32) -> StepOutcome {
        if self.0.insert((transfer, step_id)) {
            StepOutcome::FirstApply
        } else {
            StepOutcome::AlreadyApplied
        }
    }

    /// Non-mutating probe: has `(transfer, step_id)` been journaled? Used by the relocated dest
    /// promote (1d.5b.3b) to gate the `Ghost→Owned` flip on the crossing pose having LANDED
    /// (`STUB_CROSSING_STEP` applied) — pose-before-promote, so a `Promote` racing ahead of its
    /// crossing never flips a poseless dot. Read-only (unlike `journal_step`, which records).
    #[must_use]
    pub fn is_applied(&self, transfer: TransferId, step_id: u32) -> bool {
        self.0.contains(&(transfer, step_id))
    }
}

/// One entity-state crossing held until its dot is adopted (1d.1). The `StubCrossing`, emitted by
/// the saga at the CAS, races AHEAD of the dest's adopt (which is gated behind
/// `OpenInputSlot`→`HeadRead`→grant-flip under the 1c.8 promote-before-demote model), so a
/// crossing that arrives before the grant flips is BUFFERED here and drained at the flip — the
/// arrival/adopt ordering is decoupled within a SINGLE delivery (no re-emit needed).
#[derive(Debug, Clone, Copy, PartialEq)]
struct PendingCrossing {
    transfer: TransferId,
    step_id: u32,
    fence: Fence,
    pose: StampedPose,
}

/// Crossings buffered awaiting their dot's adopt grant-flip, keyed by the subject entity (1d.1).
/// Drained by `drain_pending_crossing` on the flip. In normal operation it holds ≤(concurrent
/// inbound transfers) entries transiently — the adopt lands within a few ticks.
///
/// OWED (alongside the 1d.0 `AppliedSteps` retention bound, D-22): a cleanup for an entry whose
/// entity NEVER adopts (a misrouted crossing). Both need terminal-awareness the dest shard does
/// not yet have, so both land with the journal-retention slice; neither grows per-tick in a
/// healthy run.
#[derive(Resource, Debug, Default)]
pub struct PendingCrossings(BTreeMap<EntityId, PendingCrossing>);

/// Install the stub-shard systems and resources onto a node's world + schedule.
/// Called EXPLICITLY by the shard bin (RLM Step 5a: for a `NodeKind::Shard(profile)`; was
/// `NodeKind::StubShard` pre-5a) — never by feature code, and NEVER gated on the node kind (the
/// carried `ShardProfile` is capability-inert at P1–P3, proven byte-identical by the 5a inertness gate).
pub fn register_stub_shard(world: &mut World, schedule: &mut Schedule, config: StubConfig) {
    // Slice 3e — FAIL-LOUD boundary-tuning validation at boot (mirroring the tick-pair guard): a
    // zero dwell/pad/cell would silently disable anti-flap or divide the cell rebase. The registry is
    // EMPTY in prod (the trigger is inert), but the tuning is validated regardless so a misconfigured
    // deployment never boots a half-armed trigger.
    config
        .boundary
        .validate()
        .expect("StubConfig.boundary is a valid BoundaryTuning (n_entry/k_dwell/pad/cell > 0)");
    // RLM Step 2 (L3 co-hosting hygiene): the AoI loop (`aoi_decide`) builds exactly ONE `own_coord` per
    // shard from `config.realm`, so it names the PRIMARY realm's direct children — correct for
    // node-per-realm (`held_realms == {realm}`, the base; co-hosting is D-44 KEPT-unused). A co-hosting
    // shard's CO-HOSTED realms' children are simply not AoI-evaluated here (a per-held-realm coord loop is
    // owed if/when D-44 is revived) — an incompleteness, NOT a mis-key, so it is safe for dormant infra
    // and NOT a boot tripwire (an unconditional `held_realms.len() == 1` panic would break the dormant
    // co-hosting grant/affirm/re-home tests, which legitimately build multi-realm shards). The live-AoI
    // composer (RLM Step 5/6) builds node-per-realm shards, so the primary path is always complete.
    // RLM Step 2 (M-2 single-source): the AoI hysteresis BANDS are baked into each `RealmRegion.aoi` at
    // GENERATION from `UniverseConfig.interest` (occupant_v_max_mps + tick_dt_s), and the runtime horizon
    // reads `StubConfig.tick_dt_s` — so the two must agree on `tick_dt_s`. `register_stub_shard` sees only
    // `StubConfig` (never `UniverseConfig`), so the cross-check belongs at the COMPOSER boot where both
    // meet (the live-AoI shard wiring, RLM Step 5/6); `InterestConfig` already carries the inputs for it.
    // RLM 5f-4a: `UniverseConfig::walk_demand(occupant_v_max_mps, tick_dt_s)` now takes BOTH as RUNTIME
    // arguments (no hardcoded `AOI_TICK_DT_S` — which is 0.05 and would be WRONG at the dev cluster's 50 Hz =
    // 0.02), so the composer (5f-4b) passes the live cluster's `tick_dt_s`/`move_speed·time_multiplier` and
    // the two homes agree BY CONSTRUCTION; the `debug_assert!` cross-check lands with that composer wiring.
    // Capture the scalar params read AFTER the config move (`StubConfig` is no longer `Copy` — the
    // `held_realms` set is heap-backed).
    let mint_seed = config.mint_seed;
    let input_log_capacity = config.input_log_capacity;
    world.insert_resource(config);
    world.insert_resource(Dots::default());
    world.insert_resource(RealmAuthority::default());
    world.insert_resource(CoHostedAuthority::default());
    world.insert_resource(RealmConfirmedAt::default());
    world.insert_resource(EntityMint {
        seq: 0,
        rng: SplitMix64::new(mint_seed),
    });
    world.insert_resource(InputLog::new(input_log_capacity));
    world.insert_resource(FrameCounter::default());
    world.insert_resource(RealmFrameCounter::default());
    world.insert_resource(StubStats::default());
    world.insert_resource(AppliedSteps::default());
    world.insert_resource(PendingCrossings::default());
    world.insert_resource(GhostColliderRegistration::default());
    world.insert_resource(SourceGhostMirror::default());
    world.insert_resource(OwnedTransients::default());
    // task #135 — the CONTAINMENT trigger state. `RealmRegions` defaults EMPTY, so
    // `evaluate_realm_boundaries` early-returns in prod (inert through C-3; seed boot-population is
    // C-5/C-6); tests populate it via `RealmRegions::new`.
    world.insert_resource(CrossingProgress::default());
    world.insert_resource(ContainmentProgress::default());
    world.insert_resource(RequestInFlight::default());
    world.insert_resource(RealmRegions::default());
    // RLM Step 2 — the per-CHILD AoI hysteresis ledger. Defaults EMPTY; `evaluate_realm_aoi` is inert
    // (early-returns) until regions are planted AND the shard is clock-synced, so this is byte-identical
    // through walk/canonical scale (inert AoI ⇒ no demand ⇒ never touched).
    world.insert_resource(AoiMembership::default());
    // `feed_source_ghosts` runs AFTER `process_inbound` (this tick's promote has registered the
    // neighbor + the dest dot is Owned) and BEFORE `emit_frames` (the source consumes the Delta it
    // received this tick before emitting) — the dest→source ghost collider feed (1d.5b.3b).
    // `readvance_transients` (D-7b) runs AFTER `process_inbound` (this tick's promote/drop settled the
    // held-set) and BEFORE `emit_transient_batch` (a crossing item is emitted with its CURRENT
    // re-advanced pose, keeping the source/dest origins consistent). `emit_transient_batch` (D-7) runs
    // AFTER `process_inbound` and is independent of the ghost/frame egress — it ships the source's
    // pending transient crossings as ONE batch per dest realm.
    // `self_fence_lapsed_realm` (D-3 Slice 5) runs AFTER `process_inbound` (so THIS tick's realm-head
    // confirmation has already refreshed `RealmConfirmedAt` — a fresh reply pre-empts a spurious
    // self-fence) and BEFORE the egress systems (a holder that self-fences this tick drops its
    // transients and emits NO frames this tick — a stale partitioned owner stops affecting clients at
    // once). INERT unless `self_fence_grace_ticks > 0` (the pre-D-3 default and every single-shard rig).
    // `redrive_pending_adoptions` (CA-1 S3/S4) runs AFTER `process_inbound` (this tick's promote/discard
    // settled the Arriving set — a just-promoted item is Held, not re-driven) and AFTER
    // `self_fence_lapsed_realm` (a self-fenced holder already dropped its Arriving items, so it re-drives
    // nothing). It is an orchestrator-bound egress like `emit_transient_batch`.
    // `evaluate_realm_boundaries` (Slice 3e) runs AFTER `self_fence_lapsed_realm` (a holder that lost its
    // realm this tick emits NO crossing request — the authority gate short-circuits) and BEFORE
    // `readvance_transients` / `emit_transient_batch`: a TRANSIENT crossing it flags emits a
    // `TransientCrossingRequest` this tick, and the source's own position (read from the settled
    // post-inbound pose) drives the swept segment. Its `CrossingRequest`/`TransientCrossingRequest`
    // egress is orchestrator-bound like `emit_transient_batch`. INERT unless `RealmBoundaries` is
    // non-empty (the composer plants none through P3 — behaviour-identical).
    // The full per-tick order is a single strict chain. Bevy's system-tuple `.chain()` supports at most
    // 8 direct elements, so the 9 systems are expressed as two chained groups joined by
    // `.after(self_fence_lapsed_realm)` — the SAME total order as one flat `.chain()` (group A ends at
    // `self_fence_lapsed_realm`; group B is itself chained and runs strictly after it). Splitting here is
    // a mechanical arity workaround, NOT a semantics change.
    schedule.add_systems(
        (
            request_pending_grants,
            process_inbound,
            self_fence_lapsed_realm,
        )
            .chain(),
    );
    // RLM Step 2 (M-1 retrofit, behaviour-CHANGING by design): `evaluate_realm_boundaries` and
    // `emit_realm_frames` are the two AUTHORING systems — both previously gated ONLY on `RealmAuthority`
    // (`authority.0`), the D-Finding-1 hole where a fresh shard authors at tick 0 BEFORE its first
    // `ClockSync` (a pre-sync celestial pose is wrong). `.run_if(has_synced)` closes it. Both early-return
    // on empty snaps/regions, so at walk/static scale this is inert (byte-identical); at visual scale the
    // first-sync boundary is exactly what must be gated. `ClockSample` is persistent and both systems sit
    // AFTER `observe_clock_syncs` on the shared schedule, so reading `synced` is order-correct.
    schedule.add_systems(
        (
            evaluate_realm_boundaries.run_if(has_synced),
            redrive_stranded_crossings,
            readvance_transients,
            emit_transient_batch,
            redrive_pending_adoptions,
            feed_source_ghosts,
            emit_frames,
            emit_realm_frames.run_if(has_synced),
        )
            .chain()
            .after(self_fence_lapsed_realm),
    );
    // RLM Step 2 — the demand-driven realm-lifecycle detector, a THIRD chained group (group B is at
    // Bevy's 8-`.chain()` arity limit). Runs strictly AFTER `emit_realm_frames` (the author tail — the AoI
    // decision reads the SAME per-tick child placements the observer feed just shipped, H-1) and
    // `.run_if(has_synced)` (a fresh shard demands nothing pre-sync — determinism).
    schedule.add_systems(
        (evaluate_realm_aoi,)
            .chain()
            .after(emit_realm_frames)
            .run_if(has_synced),
    );
}

/// Until the directory has granted this shard its realm — and every provisional
/// dot its entity — keep requesting (grants are idempotent by fence; a lost
/// reply costs one tick).
fn request_pending_grants(
    config: Res<StubConfig>,
    identity: Res<NodeIdentity>,
    clock: Res<ClockSample>,
    authority: Res<RealmAuthority>,
    cohosted: Res<CoHostedAuthority>,
    dots: Res<Dots>,
    mut outbox: ResMut<OutboundBox>,
) {
    // Co-hosting (the un-hosted-child cure): keep requesting the head of every co-hosted CHILD realm not
    // yet affirmed (the un-affirmed child self-grants at `GENESIS.next`, exactly like the primary realm's
    // `None` arm; a re-request of an already-held child is idempotent by fence). INERT for a single-realm
    // shard — `cohosted_realms()` is empty, so this loop never bodies (byte-identical). A BRANCHLESS shim:
    // the sole branch is the `contains_key` membership, covered by both a held and an un-held child.
    for realm in config.cohosted_realms() {
        if !cohosted.0.contains_key(&realm) {
            outbox.push_flow(
                config.orchestrator,
                MsgClass::Saga,
                &InterShardFlow::Directory(DirectoryOp::LeaseGrant {
                    key: DirectoryKey::Realm(realm),
                    owner: AuthorityRef::Shard(identity.node_id),
                    fence: Fence::GENESIS.next(),
                }),
            );
        }
    }
    match authority.0 {
        None => {
            let op = DirectoryOp::LeaseGrant {
                key: DirectoryKey::Realm(config.realm),
                owner: AuthorityRef::Shard(identity.node_id),
                fence: Fence::GENESIS.next(),
            };
            outbox.push_flow(
                config.orchestrator,
                MsgClass::Saga,
                &InterShardFlow::Directory(op),
            );
        }
        Some(realm_fence) => {
            // Periodically re-read the realm head: the reply reveals a lost lease so
            // the shard self-fences (the loss-reaction is otherwise unreachable). It is ALSO the
            // round-trip that re-arms `RealmConfirmedAt` for the proactive self-fence (D-3 Slice 5).
            if crate::directory::due_this_tick(config.realm_recheck_interval, clock.local_tick.0) {
                let op = DirectoryOp::HeadRead {
                    key: DirectoryKey::Realm(config.realm),
                };
                outbox.push_flow(
                    config.orchestrator,
                    MsgClass::Saga,
                    &InterShardFlow::Directory(op),
                );
            }
            // D-3 lease-renewal heartbeat: keep the Realm + every granted, non-departing Entity lease
            // alive on the holder's own LOCAL cadence (the orchestrator's reaper revokes a lapsed lease).
            // INERT when `lease_renew_interval_ticks == 0` (pre-D-3 default). A departing dot is excluded
            // (its lease is about to be revoked by the logout `LeaseRevoke`, not renewed).
            if crate::directory::due_this_tick(
                config.lease_renew_interval_ticks,
                clock.local_tick.0,
            ) {
                // The primary realm + every CO-HOSTED child realm this shard actually holds + every
                // granted non-departing Entity. The co-host chain is EMPTY for a single-realm shard
                // (`cohosted.0` is never populated), so the renewal set is byte-identical there.
                let renewals = std::iter::once((DirectoryKey::Realm(config.realm), realm_fence))
                    .chain(
                        cohosted
                            .0
                            .iter()
                            .map(|(realm, fence)| (DirectoryKey::Realm(*realm), *fence)),
                    )
                    .chain(
                        dots.0
                            .values()
                            .filter(|d| d.granted && !d.departing)
                            .map(|d| (DirectoryKey::Entity(d.entity), d.entity_fence)),
                    );
                outbox.push_renewals(renewals, config.orchestrator);
            }
        }
    }
    // Per-dot grant/revoke requests (login LeaseGrant, adopt HeadRead, logout LeaseRevoke). The
    // 1c.8 SOURCE granted-key poll is GONE (1d.5b.2): the saga-pushed `Demote` (on_saga_demote) is
    // now the SOLE source-demote driver — the source no longer polls the directory to DISCOVER a
    // foreign takeover.
    for dot in dots.0.values() {
        if let Some(op) = pending_grant_op(dot, identity.node_id) {
            outbox.push_flow(
                config.orchestrator,
                MsgClass::Saga,
                &InterShardFlow::Directory(op),
            );
        }
    }
}

/// The per-dot directory op `request_pending_grants` should (re)issue, as a monomorphic 3-way so
/// the system loop stays a branchless shim (HR5):
/// - adopting + !granted → `HeadRead{Entity}` (1c.8): ADOPT the record the transfer CAS moved
///   here; a `LeaseGrant` at `GENESIS.next` would be Refused (the record is past genesis).
/// - !adopting + !granted → `LeaseGrant{Entity}` at `GENESIS.next` (the login fresh-mint path).
/// - departing → `LeaseRevoke{Entity}` at the RECORDED fence (FENCE-1/5/8; a literal would be
///   Refused once a transfer advanced the fence, stranding the logout).
/// - otherwise (granted, non-departing) → `None`. The 1c.8 source granted-key poll branch is GONE
///   (1d.5b.2): the source self-fence is driven by the saga-pushed `Demote`, not a directory poll.
#[must_use]
fn pending_grant_op(dot: &Dot, node: NodeId) -> Option<DirectoryOp> {
    if !dot.granted {
        if dot.adopting {
            Some(DirectoryOp::HeadRead {
                key: DirectoryKey::Entity(dot.entity),
            })
        } else {
            Some(DirectoryOp::LeaseGrant {
                key: DirectoryKey::Entity(dot.entity),
                owner: AuthorityRef::Shard(node),
                fence: Fence::GENESIS.next(),
            })
        }
    } else if dot.departing {
        Some(DirectoryOp::LeaseRevoke {
            key: DirectoryKey::Entity(dot.entity),
            fence: dot.entity_fence,
        })
    } else {
        None
    }
}

/// Drain and dispatch everything delivered this tick.
#[allow(clippy::too_many_arguments)]
fn process_inbound(
    config: Res<StubConfig>,
    identity: Res<NodeIdentity>,
    clock: Res<ClockSample>,
    inbox: Res<InboundBox>,
    mut dots: ResMut<Dots>,
    // Bundled realm-authority tuple `SystemParam` (bevy's 16-param ceiling): the primary realm fence, its
    // confirmation timestamp, AND the co-hosted child-realm fence map are all the realm-authority stores,
    // so grouping them is a mechanical arity fix. Destructured to three `&mut` below.
    realm_auth: (
        ResMut<RealmAuthority>,
        ResMut<RealmConfirmedAt>,
        ResMut<CoHostedAuthority>,
    ),
    mut mint: ResMut<EntityMint>,
    mut log: ResMut<InputLog>,
    mut stats: ResMut<StubStats>,
    mut applied: ResMut<AppliedSteps>,
    mut pending: ResMut<PendingCrossings>,
    // Bundled into ONE tuple `SystemParam` (bevy caps a system at 16 top-level params; Slice 3e's
    // `RequestInFlight` addition would make 17). Both are the ghost-path stores, so grouping them is a
    // mechanical arity fix, not a coupling change — destructured back to the two `&mut` at the call sites.
    ghost_state: (ResMut<GhostColliderRegistration>, ResMut<SourceGhostMirror>),
    mut owned_transients: ResMut<OwnedTransients>,
    // Bundled tuple `SystemParam` (bevy's 16-param ceiling): 3f-D threads `CrossingProgress` into the
    // `CrossingAborted` demux (for the L1 dwell re-arm + attempt bump) — both are the crossing-trigger
    // latch/dwell stores, so grouping them is a mechanical arity fix. Destructured to two `&mut` below.
    crossing: (ResMut<RequestInFlight>, ResMut<CrossingProgress>),
    mut outbox: ResMut<OutboundBox>,
) {
    let (mut registration, mut mirror) = ghost_state;
    let (mut in_flight, mut progress) = crossing;
    let (mut authority, mut confirmed, mut cohosted) = realm_auth;
    for msg in &inbox.0 {
        let Inbound::Wire { from, class, bytes } = msg else {
            // Unreachability notices are observed by the node shell (TickReport);
            // the stub has no retry obligations in P1.
            continue;
        };
        match class {
            MsgClass::Control | MsgClass::Input => {
                let ctx = GatewayMsgCtx {
                    config: &config,
                    identity: &identity,
                    clock: &clock,
                    realm_fence: authority.0,
                };
                on_gateway_msg(
                    bytes,
                    *from,
                    &ctx,
                    &mut dots,
                    &mut mint,
                    &mut log,
                    &mut stats,
                    &mut outbox,
                );
            }
            MsgClass::Saga => on_directory_reply(
                bytes,
                &identity,
                &config,
                &clock,
                &mut authority,
                &mut confirmed,
                &mut cohosted,
                &mut dots,
                &mut applied,
                &mut pending,
                &mut registration,
                &mut owned_transients,
                &mut in_flight,
                &mut progress,
                &mut stats,
                &mut outbox,
            ),
            // 1d.5b.3b: the SOURCE-side ghost feed consumer — Spawn/Delta/Despawn from the dest owner
            // refresh this shard's RETAINED ghost dot (kinematic collider; pose+GhostRefresh, never a
            // second authority store). On the dedicated Ghost carriers, NOT the Saga dispatch.
            MsgClass::GhostReliable | MsgClass::GhostDelta => {
                on_ghost_flow(*from, bytes, &mut dots, &mut mirror, &mut stats);
            }
            // Membership (clock sync) is consumed by the node-level follower system;
            // Snapshot / RealmSnapshot are gateway→client render datagrams and never target a shard.
            MsgClass::Membership | MsgClass::Snapshot | MsgClass::RealmSnapshot => {}
        }
    }
}

/// Read-only context for one gateway-message dispatch.
struct GatewayMsgCtx<'a> {
    config: &'a StubConfig,
    identity: &'a NodeIdentity,
    clock: &'a ClockSample,
    realm_fence: Option<Fence>,
}

/// Resolve the BIRTH pose for a login-admitted avatar (RLM 5f-3b) — THE one admit-pose seam (HR3): every
/// login births its dot through here, with NO origin/home fork (only the pose value differs). Two arms:
///
/// - **`Some` (stored):** the account has a stored spawn pose. It is an ABSOLUTE pose in the Universe-root
///   frame [`container_coord_at`](vd_core::worldgen::container_coord_at) reads
///   (`SystemSpace{system_seed: 0}`). We [`StampedPose::sanitized`] it (a config/P7-store pose can never
///   poison the sim), RE-STAMP its `universe_tick` to the current clock `tick`, then REBIND it into `leaf`'s
///   realm frame via [`rebind_pose_to_dest`] — the SAME machinery the durable crossing / D-37 re-home /
///   transient batch use. This is deliberately NOT a raw-offset replant of the stored offset into the leaf
///   frame: routing through `rebind_pose_to_dest` keeps the pose FRAME-COHERENT (under P3 [`IdentityFrames`]
///   the position is unchanged and only the frame LABEL flips to `leaf`; when the P4 ephemeris `FrameContext`
///   lands, the real absolute→leaf transform flows in here for free, no caller reshape).
/// - **`None` (absent):** origin-at-rest in `config.frame` — BYTE-IDENTICAL to the pre-5f-3b admit literal.
///
/// Concrete (non-generic) + branchless-per-arm: both `Some`/`None` arms are exercised by the unit tests,
/// no wall-clock/rng (the `tick` is the clock's). `leaf` is this shard's own lineage coord: `leaf.lowered()`
/// is the dest realm and `leaf.parent()` its parent provenance (the one field a rebind into an `Area` needs).
fn resolve_spawn_pose(
    config: &StubConfig,
    account: AccountId,
    leaf: &RealmCoord,
    tick: UniverseTick,
) -> StampedPose {
    config
        .spawn_poses
        .get(&account)
        .map(|stored| {
            let absolute = StampedPose {
                universe_tick: tick,
                ..stored.sanitized()
            };
            rebind_pose_to_dest(absolute, leaf.lowered(), leaf.parent().map(|p| p.lowered()))
        })
        .unwrap_or_else(|| StampedPose::at_rest(config.frame, DVec3::ZERO, tick))
}

/// Handle one gateway→shard message.
#[allow(clippy::too_many_arguments)]
fn on_gateway_msg(
    bytes: &[u8],
    from: NodeId,
    ctx: &GatewayMsgCtx<'_>,
    dots: &mut Dots,
    mint: &mut EntityMint,
    log: &mut InputLog,
    stats: &mut StubStats,
    outbox: &mut OutboundBox,
) {
    let Ok(msg) = postcard::from_bytes::<GatewayToShard>(bytes) else {
        stats.undecodable += 1;
        tracing::error!("undecodable gateway->shard message");
        return;
    };
    match msg {
        GatewayToShard::AttachSession {
            session,
            fence,
            account,
        } => {
            // An avatar cannot exist on a shard that doesn't own its realm yet:
            // defer (counted); the gateway retries attach until it sees the reply.
            let Some(realm_fence) = ctx.realm_fence else {
                stats.attaches_deferred += 1;
                return;
            };
            let dot = dots.0.entry(session).or_insert_with(|| {
                let entity = mint_entity(mint, ctx.identity.node_id);
                // RLM 5f-3b: birth at the account's STORED spawn pose, rebound into THIS shard's realm
                // frame via the ONE frame machinery (HR3), or origin-at-rest when none is stored (empty
                // `spawn_poses` ⇒ byte-identical to the pre-5f-3b literal). ONE admit path — only the pose
                // value changes; the Ghost birth → LeaseGrant → Promote → SessionAttached flow is unchanged.
                let pose = resolve_spawn_pose(
                    ctx.config,
                    account,
                    &ctx.config.own_coord,
                    ctx.clock.universe_tick,
                );
                Dot {
                    entity,
                    account,
                    session_fence: fence,
                    gateway: from,
                    granted: false,
                    input_active: false,
                    adopting: false,
                    // Born a Ghost: a pre-grant login simulates NOTHING (`simulates()==false`),
                    // so it emits no frame until the LoggedIn grant flip Promotes it Ghost→Owned
                    // at the recorded grant fence (the IDENTICAL Promote the transfer-dest uses).
                    authority: Authority::Ghost {
                        source_fence: Fence::GENESIS,
                        since_tick: ctx.clock.local_tick,
                    },
                    departing: false,
                    entity_fence: Fence::GENESIS,
                    pose,
                    yaw: 0.0,
                    pitch: 0.0,
                    last_applied_seq: None,
                    // Seed to the spawn offset: tick-1's swept segment is degenerate. Origin when no stored
                    // pose (byte-identical to the old `DVec3::ZERO`); the stored offset otherwise.
                    prev_offset: pose.pos.offset(),
                }
            });
            // Idempotent re-attach: refresh the fence if the gateway's advanced.
            if fence > dot.session_fence {
                dot.session_fence = fence;
            }
            if dot.granted {
                // Authority recorded: confirm (idempotently, on every retry).
                let reply = ShardToGateway::SessionAttached {
                    session,
                    entity: dot.entity,
                    frame: ctx.config.frame,
                    realm_fence,
                };
                push_session_reply(outbox, from, &reply);
            } else {
                // The avatar exists ONLY provisionally until the directory
                // records its grant (requested by `request_pending_grants`,
                // retried every tick). No reply yet — the gateway retries attach.
                let op = DirectoryOp::LeaseGrant {
                    key: DirectoryKey::Entity(dot.entity),
                    owner: AuthorityRef::Shard(ctx.identity.node_id),
                    fence: Fence::GENESIS.next(),
                };
                outbox.push_flow(
                    ctx.config.orchestrator,
                    MsgClass::Saga,
                    &InterShardFlow::Directory(op),
                );
            }
        }
        GatewayToShard::SessionInput {
            session,
            fence,
            input_bytes,
        } => {
            apply_input(
                ctx.config,
                ctx.clock,
                dots,
                log,
                session,
                fence,
                &input_bytes,
            );
        }
        GatewayToShard::DetachSession { session, fence } => {
            match dots.0.get_mut(&session) {
                Some(dot) if fence.is_stale_against(dot.session_fence) => {
                    log.record_discarded(session, None, DiscardReason::StaleFence);
                }
                Some(dot) if dot.granted => {
                    // Two-phase release: the dot stays HELD until the directory
                    // confirms the revoke (authority is released AT the
                    // directory, mirroring the grant). The revoke is (re)sent by
                    // `request_pending_grants`; the SessionDetached reply waits
                    // for the confirmation.
                    dot.departing = true;
                }
                Some(_) => {
                    // A provisional dot has no directory record: safe to drop now.
                    dots.0.remove(&session);
                    push_session_reply(outbox, from, &ShardToGateway::SessionDetached { session });
                }
                None => {
                    // Idempotent: detaching an unknown session still confirms.
                    push_session_reply(outbox, from, &ShardToGateway::SessionDetached { session });
                }
            }
        }
        GatewayToShard::OpenInputSlot {
            session,
            fence,
            account,
            resume_from_seq,
            subject,
        } => {
            // A transfer-DESTINATION input slot. The gateway sends this only AFTER the
            // directory committed authority to this shard (the saga's `commit_cas`), so the
            // shard may APPLY this session's input even before its own per-entity grant
            // records: the post-marker cut buffer the gateway held drains here. The dot is
            // `input_active` and ADOPTS the transfer subject (1c.8): its entity becomes the
            // SUBJECT id (the record the CAS moved here), so the adopt HeadRead lands on that
            // record and flips `granted` (authority-held). It STAYS a Ghost (`simulates()==false`)
            // — it renders nowhere (no pose carried) until the saga `Promote` flips it in
            // `on_saga_promote` (1d.5b.3b; `apply_crossing` only STORES the pose) — and gets
            // no `SessionAttached` reply (the source still owns the client connection — R2).
            let Some(_realm_fence) = ctx.realm_fence else {
                // No realm lease yet: drop + count. NO re-drive in 1c.5 (the gateway already
                // take-drained the buffer; OpenInputSlot is emitted once, the saga is
                // forward-only past Committed) — the recovery producer is owed 1d/P3 (D-8).
                stats.input_slots_deferred += 1;
                return;
            };
            // Extract the SUBJECT EntityId to ADOPT. A non-Entity subject (e.g. a Realm-subject
            // saga, which the FSM proptests drive through CommitAuthority) is a COUNTED no-op —
            // never an extraction panic: the dest only adopts an Entity transfer.
            let Some(subject_entity) = subject.transfer_subject_entity() else {
                stats.input_slots_malformed += 1;
                tracing::warn!(
                    ?subject,
                    "OpenInputSlot carried a non-Entity subject — no adopt (counted no-op)"
                );
                return;
            };
            let dot = dots.0.entry(session).or_insert_with(|| Dot {
                entity: subject_entity, // 1c.8 ADOPT: the transferred subject id, not a fresh mint
                account,
                session_fence: fence,
                gateway: from,
                granted: false,
                input_active: false,
                adopting: true,
                // THE frozen ghost mirror, born Ghost (NOT Frozen) so the later promote is a legal
                // Promote (Ghost→Owned); `source_fence: GENESIS` is strictly stale vs the CAS
                // fence, so the `on_saga_promote` Promote succeeds (1d.5b.3b — relocated out of
                // `apply_crossing`, which now only STORES the crossed pose).
                authority: Authority::Ghost {
                    source_fence: Fence::GENESIS,
                    since_tick: ctx.clock.local_tick,
                },
                departing: false,
                entity_fence: Fence::GENESIS, // the adopt HeadRead fills the real CAS fence
                pose: StampedPose::at_rest(ctx.config.frame, DVec3::ZERO, ctx.clock.universe_tick),
                yaw: 0.0,
                pitch: 0.0,
                last_applied_seq: None,
                // Seed to the spawn offset (origin): tick-1's swept segment is degenerate.
                prev_offset: DVec3::ZERO,
            });
            // STALE-GATEWAY-DROP (the binding day-one rule, `wire::session_flow`): a slot whose
            // fence is BELOW the dot's session fence is a replay or a partitioned old gateway —
            // drop it, never re-arm input (1d adds departing/revoking states this guards). A
            // fresh mint set `session_fence := fence`, so it is never stale against itself.
            if fence.is_stale_against(dot.session_fence) {
                stats.input_slots_stale += 1;
                return;
            }
            // SECURITY / HR1: only arm input from the gateway that owns the session — a shard must
            // never apply input for a session it was not legitimately routed. `!granted` gates: only
            // a fresh adopt-Ghost slot arms `input_active` + re-seeds its watermark; a granted dot
            // owns its own input watermark and must not be re-seeded (the `!granted` guard false arm).
            if !dot.granted && dot.gateway == from {
                dot.input_active = true;
                // OBSERVABILITY: record the gateway-EMITTED resume watermark (unmutated) so the
                // cross-cut conservation gate can assert it equals the client's CUT_MARKER seq
                // (D-28). Distinct from `dot.last_applied_seq`, which advances as the drained
                // batch applies — this latch holds the AS-RECEIVED value.
                stats.last_input_slot_resume = Some(resume_from_seq);
                // SEED the dedup watermark to `resume_from_seq` (= marker_seq): the drained
                // resume batch (marker+1..) applies in order; a `seq <= marker` replay is
                // rejected (the source already applied it). MAX-merge — never LOWER it.
                dot.last_applied_seq = Some(
                    dot.last_applied_seq
                        .map_or(resume_from_seq, |s| s.max(resume_from_seq)),
                );
                if fence > dot.session_fence {
                    dot.session_fence = fence;
                }
            }
        }
    }
}

/// Mint a fresh avatar id: `{kind, mint_shard, seq, rand24}` — monotonic sequence
/// plus seed-derived entropy; NEVER time-derived (R7).
fn mint_entity(mint: &mut EntityMint, node: NodeId) -> EntityId {
    let seq = mint.seq;
    mint.seq += 1;
    let rand24 = (mint.rng.next_u64() & 0x00FF_FFFF) as u32;
    // NodeIds fit u32 in every real topology; the mask documents the packing.
    let mint_shard = (node.0 & u64::from(u32::MAX)) as u32;
    EntityId::pack(EntityKind::Player, mint_shard, seq, rand24)
}

fn push_session_reply(outbox: &mut OutboundBox, to: NodeId, reply: &ShardToGateway) {
    let bytes = postcard::to_allocvec(reply).expect("closed wire enums serialize infallibly");
    outbox.0.push((
        to,
        MsgClass::Control,
        crate::io::bytes(bytes),
        Durability::Ephemeral,
    ));
}

/// The input path: fence gate → decode → seq gate → integrate. Every outcome lands
/// in the [`InputLog`].
fn apply_input(
    config: &StubConfig,
    clock: &ClockSample,
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
    integrate(dot, &input, config, clock);
    log.record_applied(session, input.seq);
}

/// Kinematic point integration: axes are clamped to [-1, 1], displacement is
/// speed·dt in the dot's yaw-rotated heading. Pure f64 closed-form per tick.
fn integrate(dot: &mut Dot, input: &InputDatagram, config: &StubConfig, clock: &ClockSample) {
    dot.yaw += f64::from(input.look[0]);
    // Pitch is CLAMPED to the valid look range (WB-1): unbounded accumulation would wrap
    // past the ±π/2 gimbal pole and silently corrupt authoritative orientation. Yaw wraps
    // freely (no pole). ONE shared bound (`vd_core::kinematics::PITCH_LIMIT`).
    dot.pitch = kinematics::clamp_pitch(dot.pitch + f64::from(input.look[1]));
    dot.orient_from_angles();
    // The movement-axis map is the ONE shared input convention (vd_core::kinematics) —
    // the client's nav/camera invert the SAME definition (no hand-re-encoded drift).
    let axes = kinematics::local_axes_from_movement(input.movement);
    // OCCUPANT movement runs in the realm's SUBJECTIVE time: `move_speed · dt · time_multiplier`. At the
    // default `1.0` this is byte-identical; a slow-time realm (`< 1.0`) moves its occupants slower.
    let step = dot.pose.orient
        * axes
        * (config.move_speed_mps * config.tick_dt_s * config.time_multiplier);
    // Integrate the frame-local offset, PRESERVING the cell anchor (`map_offset`, NOT `local` which
    // would zero it). Through P3 cell is ZERO so this is the full local position += step,
    // behaviour-identical to the pre-lattice `pos += step` (D-41); the per-tick normalize/re-centering
    // that re-buckets a drifted offset back into the cell (the bounded-offset invariant) lands with
    // P4/P5 re-centering (galaxy ly-cells at P10), and is a PURE ADDITION here (a `.normalize()` on the
    // result) because the cell is already carried — NOT a clobber-and-replace.
    dot.pose.pos = dot.pose.pos.map_offset(|o| o + step);
    dot.pose.vel = step / config.tick_dt_s;
    dot.pose.universe_tick = clock.universe_tick;
}

impl Dot {
    fn orient_from_angles(&mut self) {
        self.pose.orient = kinematics::orient_from_yaw_pitch(self.yaw, self.pitch);
    }
}

/// A `SessionAttached` reply plus the gateway it routes to (the login grant-flip's egress).
struct AttachEgress {
    gateway: NodeId,
    reply: ShardToGateway,
}

/// The outcome of a grant-flip — three mutually-exclusive caller obligations (1d.1/1d.2c).
enum GrantFlip {
    /// A LOGIN attach flipped: push the `SessionAttached` egress.
    LoggedIn(AttachEgress),
    /// A transfer-dest ADOPT flipped (R2 — no re-home): the dot becomes granted + STAYS a Ghost
    /// (`simulates()==false`, emits no client frames). 1d.5b.3b: the dest read-sub is NO LONGER
    /// announced here — `SubscriptionReady` RELOCATED to `on_saga_promote` (announced only when the
    /// dest genuinely promotes Ghost→Owned), so until promote the client's render authority stays on
    /// the SOURCE sub. The caller only drains the flipped session's buffered crossing (if any); the
    /// `session` is carried so the drain needs no second lookup.
    Adopted { session: SessionId },
    /// No ungranted dot matched (a duplicate grant head): idempotent no-op.
    NoOp,
}

/// Flip the matching provisional dot to `granted` at the recorded fence. Monomorphic (the loop +
/// the login/adopt branching live here) so the reply arm stays a branchless dispatch (HR5).
///
/// - login (`!adopting`): granted := true, Promote Ghost→Owned (now `simulates()`) → `LoggedIn`
///   (push SessionAttached).
/// - adopt (`adopting`): granted := true, adopting cleared, the dot STAYS Ghost (NO Promote here —
///   the saga `Promote` flips it in `on_saga_promote`, 1d.5b.3b; `apply_crossing` only STORES the
///   pose), NO SessionAttached (R2 — the source owns the client) → `Adopted` (the caller drains the crossing).
///
/// Guarded by `!dot.granted` so a duplicate grant head is idempotent (`NoOp`, no second flip).
fn flip_grant(
    dots: &mut BTreeMap<SessionId, Dot>,
    entity: EntityId,
    fence: Fence,
    config: &StubConfig,
    realm_fence: Fence,
) -> GrantFlip {
    for (session, dot) in dots.iter_mut() {
        if !dot_grant_target(dot, entity) {
            continue;
        }
        dot.granted = true;
        dot.entity_fence = fence; // the recorded authority fence (== the CAS new_fence)
        if dot.adopting {
            // Transfer-dest adopt: held but not rendered, no re-home (1c.8). Clear the adopt marker
            // so the dot becomes a normal granted holder (the dest dot). The dot STAYS Ghost
            // (`simulates()==false`) — it emits no client frames yet. 1d.5b.3b: the dest read-sub is
            // NOT announced here — `SubscriptionReady` moved to `on_saga_promote` (announced only at
            // the genuine Ghost→Owned promote), so the client stays on the SOURCE sub until then and
            // the demote-before-promote ordering is strict. The caller only drains the crossing.
            dot.adopting = false;
            return GrantFlip::Adopted { session: *session };
        }
        // Login Promote: Ghost{GENESIS} → Owned at the recorded grant fence (strictly > GENESIS,
        // so infallible) — the IDENTICAL Ghost→Owned machinery the transfer-dest uses (kind-generic).
        dot.authority = dot
            .authority
            .apply(AuthorityCmd::Promote { new_fence: fence })
            .expect(
                "login Ghost{GENESIS} promotes at the recorded grant fence (strictly > GENESIS)",
            );
        return GrantFlip::LoggedIn(AttachEgress {
            gateway: dot.gateway,
            reply: ShardToGateway::SessionAttached {
                session: *session,
                entity,
                frame: config.frame,
                realm_fence,
            },
        });
    }
    GrantFlip::NoOp
}

/// Whether a dot is the (single) ungranted holder of `entity` awaiting its grant flip.
/// Monomorphic predicate (the `&&` short-circuit is covered once here, not in the loop body).
#[must_use]
fn dot_grant_target(dot: &Dot, entity: EntityId) -> bool {
    (dot.entity == entity) & !dot.granted
}

/// The SOURCE self-fence machinery (1c.8/1d.4b): DEMOTE the local granted, non-departing holder of
/// `entity` to a RETAINED Ghost (`Owned→Frozen→Ghost`) at `new_owner_fence` (the post-CAS owner
/// fence, strictly newer than the source's own grant fence — the directory CAS is fence-monotone),
/// NO directory write (the record is the dest's now), `granted` KEPT. 1d.4b RETAINS the dot (the
/// first ghost) instead of `dots.remove`-ing it, so it stops emitting (`simulates()==false`) but
/// survives for the 1d.5b ghost-as-collider. Idempotent WITHOUT IllegalTransition-as-control-flow:
/// an already-Ghost redelivery is a counted no-op via the `simulates()` guard; a no-match (the
/// entity is not held here) is a clean no-op. Monomorphic.
///
/// 1d.5b.2 — the saga-pushed ordered `Demote` ([`on_saga_demote`]) is now the SOLE driver of this
/// transition; the 1c.8 cooperative granted-key poll (which DISCOVERED a foreign takeover from a
/// directory head) is TORN OUT, so `transfer` is always the real `Demote` transfer (no inert
/// fallback). 1d.5b.3b RELOCATED the dest's autonomous `apply_crossing` promote into `on_saga_promote`
/// (strict demote-before-promote), so the source demotes to a retained Ghost BEFORE the dest is Owned:
/// there is NO two-holder window (a brief ZERO-Owned handoff gap instead, excused mid-flight by the
/// 1d.5b.3d per-tick oracle). The retained source Ghost is a live collider fed by the dest (the
/// `GhostFlow` feed, 1d.5b.3b) and torn down on band-exit (1d.5b.3c).
fn self_fence_foreign_entity(
    dots: &mut BTreeMap<SessionId, Dot>,
    entity: EntityId,
    new_owner_fence: Fence,
    transfer: TransferId,
    at_tick: TickId,
    stats: &mut StubStats,
) {
    let Some(session) = dots
        .iter()
        .find(|(_, d)| foreign_takeover_target(d, entity))
        .map(|(session, _)| *session)
    else {
        return; // no matching dot: never held here — a clean no-op
    };
    let dot = dots
        .get_mut(&session)
        .expect("foreign_takeover_target just matched this session");
    // Idempotency WITHOUT relying on IllegalTransition as control flow: only an Owned source
    // demotes; an already-Ghost redelivery is a COUNTED no-op (the covered guard arm).
    if !dot.authority.simulates() {
        stats.self_fence_skipped += 1;
        return;
    }
    // Owned → Frozen (always legal) → Ghost (the foreign CAS fence is strictly newer than the
    // source's own grant fence — the directory CAS is fence-monotone). An Err here is a real
    // invariant break, so panic; the refusal arms are proptested in `authority.rs`.
    dot.authority = dot
        .authority
        .apply(AuthorityCmd::Freeze { transfer })
        .and_then(|frozen| {
            frozen.apply(AuthorityCmd::Demote {
                new_owner_fence,
                at_tick,
            })
        })
        .expect("an Owned source freezes infallibly and demotes at the strictly-newer CAS fence");
    tracing::warn!(
        "entity {entity} now held at fence {new_owner_fence} — source self-demoted to a retained Ghost"
    );
    // KEEP the dot (no remove) and KEEP `granted` == true — it survives as the retained Ghost.
}

/// SOURCE consumer of the saga-pushed ordered `Demote` (1d.5b.1, D-2): the binding fence-enforced
/// `Owned→Frozen→Ghost` demote, the FIRST half of demote-before-promote, and (since 1d.5b.2) the
/// SOLE driver of the source self-fence. It REUSES [`self_fence_foreign_entity`]'s machinery (Freeze
/// at the real `cmd.transfer`, then Demote at `cmd.new_owner_fence` — the post-CAS owner fence). An
/// already-Ghost dot (a redelivery) is the `!simulates()` counted no-op (`self_fence_skipped`); an
/// unheld entity is a clean no-match no-op. The `DemoteAck` is sent UNCONDITIONALLY (outside the flip
/// guard): even on those no-op paths the saga MUST still get its ack or it would wedge in `Demoting`.
/// Monomorphic. (A non-Entity subject has no local Entity dot here — counted + skipped, still acked.)
fn on_saga_demote(
    cmd: DemoteCmd,
    config: &StubConfig,
    clock: &ClockSample,
    dots: &mut Dots,
    in_flight: &mut RequestInFlight,
    stats: &mut StubStats,
    outbox: &mut OutboundBox,
) {
    match cmd.subject.transfer_subject_entity() {
        Some(entity) => {
            self_fence_foreign_entity(
                &mut dots.0,
                entity,
                cmd.new_owner_fence,
                cmd.transfer,
                clock.local_tick,
                stats,
            );
            // Slice 3e: the durable COMMIT terminal at the source — POSITIVELY clear the crossing
            // latch (the triggered durable crossing committed; the entity may trigger again). A latch
            // present-and-removed increments the cleared counter; an absent latch is a clean no-op.
            if in_flight.0.remove(&entity).is_some() {
                stats.crossing_latches_cleared += 1;
            }
        }
        None => stats.saga_demote_no_entity += 1,
    }
    // Ack DemoteAck UNCONDITIONALLY — the saga's demote-before-promote ordering gates the dest
    // Promote on THIS ack; a missing ack on a no-op path (already-Ghost / unheld) would wedge it.
    outbox.push_flow(
        config.orchestrator,
        MsgClass::Saga,
        &InterShardFlow::SagaAck(TransferControlAck::DemoteAck {
            transfer: cmd.transfer,
        }),
    );
}

/// SOURCE consumer of the orchestrator's `TransientCrossingGrant` (Slice 3e): the grant carries the
/// resolved dest + fence + batch id for a transient this shard flagged via `TransientCrossingRequest`,
/// so flip the SOURCE Transient `Held → Crossing{dest, to_realm, dst_realm_fence, batch, to_parent}` — the
/// five grant fields match `TransientStatus::Crossing` EXACTLY (a straight field assign). `emit_transient_
/// batch` then ships it. A grant for a NON-Entity subject, an UNKNOWN transient, or a non-`Held` one (a
/// redelivery after the flip) is a counted no-op (degrade, never panic). Monomorphic so every arm is
/// covered once (HR5).
fn on_transient_crossing_grant(
    grant: TransientCrossingGrant,
    owned: &mut OwnedTransients,
    stats: &mut StubStats,
) {
    let Some(entity) = grant.subject.transfer_subject_entity() else {
        stats.transient_grant_no_entity += 1;
        return;
    };
    // Flip ONLY a SETTLED `Held{outbound: None}` transient → Crossing. A `Held{outbound: Some}` (already
    // emitted this batch), a `Crossing` (already flipped), or an Arriving/Departing (mid-handoff) item is
    // a counted no-op — so a REDELIVERED grant never re-flips + re-emits an in-flight batch (at-least-once).
    match owned.0.get_mut(&entity) {
        Some(t) if matches!(t.status, TransientStatus::Held { outbound: None }) => {
            t.status = TransientStatus::Crossing {
                dest: grant.dest,
                to_realm: grant.to_realm,
                dst_realm_fence: grant.dst_realm_fence,
                batch: grant.batch,
                to_parent: grant.to_parent,
            };
            stats.transient_grants_applied += 1;
        }
        // Unknown transient OR one already crossing/emitted/handing-off (a redelivery) — counted no-op.
        Some(_) | None => stats.transient_grant_noop += 1,
    }
}

/// SOURCE consumer of the orchestrator's `CrossingAborted` (Slice 3f-D): the crossing resolve/start saga
/// aborted pre-CAS, so CLEAR the subject's `RequestInFlight` latch (the positive re-cross signal, never a
/// bounded TTL) — but ONLY if it still holds THIS aborted transfer id (a stale abort for a superseded /
/// re-latched transfer must not free a live crossing; the per-attempt id makes that exact).
///
/// ALWAYS acks `CrossingAbortedAck` (all three paths): the orchestrator keeps its durable
/// `pending_abort_replies` entry alive — re-emitting `CrossingAborted` every `scan_deadlines` window,
/// crash-durable via the store — until THIS ack drops it. So a lost first ack whose re-emit finds the latch
/// already cleared must STILL re-ack, else the ORCHESTRATOR entry leaks (the ack-gate must not merely
/// relocate the strand). On the id-match clear it also RE-ARMS: reset ONLY the dwell (so a still-in-band
/// entity re-requests without physically re-crossing — audit-L1) + BUMP the attempt (so the re-latch mints a
/// FRESH id — H2). Monomorphic.
fn on_crossing_aborted(
    abort: CrossingAborted,
    in_flight: &mut RequestInFlight,
    progress: &mut CrossingProgress,
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
    let Some(entity) = abort.subject.transfer_subject_entity() else {
        stats.crossing_abort_no_entity += 1;
        return;
    };
    // Clear ONLY on an exact id match (`== Some(&abort.transfer)`) — equality over the value so the
    // false arm is a covered no-op, not an uncoverable `matches!` region (HR5(d)).
    if in_flight.0.get(&entity) == Some(&abort.transfer) {
        in_flight.0.remove(&entity);
        // Re-arm (audit-L1 + H2): reset the cooldown so the re-home can re-fire, and BUMP the attempt so the
        // re-latch mints a fresh id. `entry().or_default()` (not `if let`) so
        // there is no uncoverable `None` region — a latched entity always has a `CrossingState`, but a
        // default is harmless if absent. NOT a drop: dropping would reset the attempt to 0 → aliasing.
        let st = progress.0.entry(entity).or_default();
        st.crossing_attempt = st.crossing_attempt.saturating_add(1);
        // Reset the cooldown so the re-home can re-fire without a physical re-cross — the container still
        // differs from the owner (the entity did not move), so `should_rehome` fires again next tick.
        st.last_commit_tick = None;
        stats.crossing_latches_cleared += 1;
    } else {
        stats.crossing_abort_stale += 1;
    }
}

/// DEST consumer of the saga-pushed ordered `Promote` (1d.5b.3b, D-2): the SECOND half of
/// demote-before-promote and — since 1d.5b.3b — the REAL `Ghost→Owned` promoter (the flip RELOCATED
/// here out of `apply_crossing`, so the ordering is STRICT: the dest becomes Owned only on this
/// command, after the source has demoted). On the FIRST delivery it: flips the dest dot Ghost→Owned
/// at `cmd.new_fence` (gated pose-before-promote on the crossing having landed), announces the dest
/// read-sub to the gateway (`SubscriptionReady` — RELOCATED here from the adopt flip, so the
/// client's render authority moves to the dest only NOW, ~promote-time), registers the transfer
/// `source` as a ghost-neighbor, and SPAWNS the source ghost (`GhostFlow::Spawn` → the dest drives
/// the collider feed). It ALWAYS acks `PromoteAck` (outside the FirstApply gate) so a redelivery
/// re-acks without re-flipping/re-spawning. Journal-gate + ack here; the branchy flip lives in the
/// monomorphic [`promote_apply`] (HR5). `realm_fence` is the dest's realm authority (held by the
/// invariant that the orchestrator only routes a Promote to the realm's owner).
#[allow(clippy::too_many_arguments)]
fn on_saga_promote(
    cmd: PromoteCmd,
    config: &StubConfig,
    self_node: NodeId,
    clock: &ClockSample,
    dots: &mut Dots,
    applied: &mut AppliedSteps,
    registration: &mut GhostColliderRegistration,
    realm_fence: Fence,
    stats: &mut StubStats,
    outbox: &mut OutboundBox,
) {
    match applied.journal_step(cmd.transfer, PROMOTE_STEP) {
        StepOutcome::FirstApply => promote_apply(
            cmd,
            config,
            self_node,
            clock,
            dots,
            applied,
            registration,
            realm_fence,
            stats,
            outbox,
        ),
        StepOutcome::AlreadyApplied => stats.promotes_redelivered += 1,
    }
    // Ack UNCONDITIONALLY — the release gate (PromoteAcked AND DestDelivered) needs this ack even on
    // a redelivery / a deferred (pose-not-yet-landed) flip; never wedge the saga in Promoting.
    outbox.push_flow(
        config.orchestrator,
        MsgClass::Saga,
        &InterShardFlow::SagaAck(TransferControlAck::PromoteAck {
            transfer: cmd.transfer,
        }),
    );
}

/// The DEST promote effect (1d.5b.3b), monomorphic so every branch is covered ONCE here, not in a
/// generic body (HR5). Find the dest Ghost dot for the subject entity; if the crossing pose has not
/// yet landed (`STUB_CROSSING_STEP` not journaled) DEFER (count, no flip — pose-before-promote);
/// else flip Ghost→Owned at `cmd.new_fence` (infallible: the dest Ghost's GENESIS `source_fence` is
/// strictly below the post-CAS fence), announce the dest sub, register the ghost-neighbor, and spawn
/// the source ghost. A non-Entity subject or a missing dot is a counted no-op (`promote_no_dot`).
#[allow(clippy::too_many_arguments)]
fn promote_apply(
    cmd: PromoteCmd,
    config: &StubConfig,
    self_node: NodeId,
    clock: &ClockSample,
    dots: &mut Dots,
    applied: &mut AppliedSteps,
    registration: &mut GhostColliderRegistration,
    realm_fence: Fence,
    stats: &mut StubStats,
    outbox: &mut OutboundBox,
) {
    let Some(entity) = cmd.subject.transfer_subject_entity() else {
        stats.promote_no_dot += 1;
        return;
    };
    let Some((session, dot)) = dots.0.iter_mut().find(|(_, d)| crossing_target(d, entity)) else {
        stats.promote_no_dot += 1;
        return;
    };
    if !applied.is_applied(cmd.transfer, STUB_CROSSING_STEP) {
        // The crossing pose has not landed yet — flipping now would emit a poseless origin frame. DEFER.
        // ⚠️ RECOVERY (DEFERRED.md D-6 #1, the durable entity-STATE crossing as a producer-less phase):
        // re-driving the PROMOTE does NOT cure a crossing that the at-most-once mesh LOST — Promote only
        // re-acks + re-defers here. The lost-crossing case needs the CROSSING re-driven (a Demoting/Promoting
        // Timeout that also re-emits `EmitCrossing`, idempotent via this `STUB_CROSSING_STEP` journal dedup —
        // the proven D-37-2d template) OR the owed redelivering transport. Under the harness at-least-once
        // FaultFabric the crossing always redelivers, so this DEFER is reached only as the transient
        // promote-before-crossing race, never a permanent wedge; the permanent case is a deploy precondition.
        stats.promote_before_crossing += 1;
        return;
    }
    let session = *session;
    // Ghost→Owned at the post-CAS fence. TWO cases, split by whether this is a SOURCE==DEST re-home (task
    // #149) — a co-hosted-child crossing whose `head(Realm(dest))` resolved to THIS node:
    // - CROSS-NODE (`cmd.source != self_node`): the DEST's Ghost holds GENESIS `source_fence`, strictly < the
    //   CAS fence, so the standard strict-newer `AuthorityCmd::Promote` is infallible.
    // - SOURCE==DEST (`cmd.source == self_node`): the ordered `Demote` already self-fenced THIS SAME dot to
    //   `Ghost{source_fence: cmd.new_fence}` (the demote lands the CAS fence), so a strict-newer Promote at
    //   the SAME fence would `StaleFence`. The directory CAS committed THIS node as the owner at
    //   `cmd.new_fence`, so RE-OWN the dot at that exact fence directly (the route swap completing as the
    //   idempotent no-op the source==dest saga is). This monomorphic 2-arm `if` keeps both paths covered
    //   (HR5): the cross-node strict-newer promote AND the source==dest equal-fence re-own.
    dot.authority = if cmd.source == self_node {
        Authority::Owned {
            fence: cmd.new_fence,
        }
    } else {
        dot.authority
            .apply(AuthorityCmd::Promote {
                new_fence: cmd.new_fence,
            })
            .expect(
                "cross-node dest Ghost promotes at the post-CAS fence (strictly newer than GENESIS)",
            )
    };
    let gateway = dot.gateway;
    let pose = dot.pose;
    stats.promotes_confirmed += 1;
    // RELOCATED here from the adopt flip (1d.5b.3b): the dest read-sub is announced ONLY now, at
    // promote — so the client's render authority moves to the dest only after it is genuinely Owned.
    push_session_reply(
        outbox,
        gateway,
        &ShardToGateway::SubscriptionReady {
            session,
            entity,
            frame: config.frame,
            realm_fence,
        },
    );
    // The dest (owner) registers the source as a ghost-neighbor + spawns the source ghost; the ANCHOR is
    // THIS promote pose (= the crossed pose, the boundary the entity entered through), from which the dest
    // measures band membership + Despawns the ghost on band-exit (1d.5b.3c). Shared tail — see the fn doc.
    // `self_node` guards the source==dest same-node re-home (no self-ghost feed loop).
    register_and_spawn_source_ghost(
        entity,
        cmd.source,
        self_node,
        pose,
        cmd.new_fence,
        clock.local_tick,
        registration,
        outbox,
    );
}

/// Register the transfer/re-home SOURCE as a ghost-neighbor + SPAWN the source ghost so this shard (the new
/// owner) DRIVES the `GhostFlow` collider feed to the source (owner → ghost-host) — keeping the retained
/// source ghost a live collider + the render seamless (`feed_source_ghosts` streams Delta after). The ANCHOR
/// is the applied (crossed / re-homed) pose. Shared by [`promote_apply`] (a `Ghost→Owned` flip) and
/// [`re_home_apply`] (a fresh Owned build): the two DIVERGE above this tail, which was byte-identical (the
/// old ⚠️ DRY-PIN, now extracted so a [[D-39]].6 ghost-blob / band-driven multi-neighbor edit touches ONE
/// place). A Spawn to a possibly-dead source is harmless FireAndForget (HR1: the shard cannot see the
/// liveness set — never special-case it).
///
/// SOURCE==DEST GUARD (task #149): a same-node re-home (a co-hosted-child crossing whose `head(Realm(dest))`
/// resolves to THIS node) drives the SAME orchestrator saga, so it reaches promote with `source == self_node`.
/// Registering `self` as a ghost-neighbor of itself + Spawning a ghost to itself would create a self-ghost
/// feed loop (the owner would `GhostFlow::Delta` its own retained copy). Skip BOTH here — the dot is already
/// Owned+rendered on this node; there is no foreign owner to feed. A cross-node source (`source != self_node`)
/// takes the register+Spawn path exactly as before. Monomorphic, both arms covered.
#[allow(clippy::too_many_arguments)]
fn register_and_spawn_source_ghost(
    entity: EntityId,
    source: NodeId,
    self_node: NodeId,
    pose: StampedPose,
    source_fence: Fence,
    since_tick: TickId,
    registration: &mut GhostColliderRegistration,
    outbox: &mut OutboundBox,
) {
    if source == self_node {
        // Same-node re-home: no foreign owner to feed. Skip the self-ghost register + Spawn (else a
        // self-feed loop). The dot is already Owned + rendered here — nothing else is owed.
        return;
    }
    registration.0.insert(
        entity,
        GhostNeighbor {
            source,
            seq: 0,
            anchor: pose.pos.offset(),
        },
    );
    outbox.push_flow(
        source,
        MsgClass::GhostReliable,
        &InterShardFlow::Ghost(GhostFlow::Spawn {
            entity,
            pose,
            source_fence,
            since_tick,
        }),
    );
}

/// D-37 forward re-home ADOPT (the journal-gate + UNCONDITIONAL-ack shell, modelled on `on_saga_promote`).
/// The fresh target RECEIVES a `ReHome` after the orchestrator re-homed a permanently-killed owner's
/// committed entity here; it CREATES the entity as an Owned dot from the carried pose (no pre-existing
/// ghost to flip — unlike `Promote`'s `Ghost→Owned`). Acks `PromoteAck` UNCONDITIONALLY (outside the
/// journal gate) — the saga's `Promoting` release gate needs the ack even on a redelivery, else it wedges.
#[allow(clippy::too_many_arguments)]
fn on_re_home(
    cmd: ReHomeCmd,
    config: &StubConfig,
    self_node: NodeId,
    clock: &ClockSample,
    dots: &mut Dots,
    applied: &mut AppliedSteps,
    registration: &mut GhostColliderRegistration,
    stats: &mut StubStats,
    outbox: &mut OutboundBox,
) {
    // §3.3 epoch fail-safe, UNIFORM with the crossing ingress (`on_transfer_envelope`): a re-home adopt
    // carrying a `universe_epoch` that does not match this shard's current epoch is REFUSED — never
    // journaled, adopted, or acked — so no entity is reconstructed at a stale celestial position. The
    // re-home is the SECOND pose-placing ingress; guarding it here makes §3.3 cover BOTH. A cross-epoch
    // re-home is a defunct OLD-epoch saga (only reachable across a re-genesis / a delayed redelivery): NOT
    // acking is correct — the current-epoch orchestrator is not waiting on it (same rationale as the
    // crossing arm's no-ack-on-mismatch).
    if cmd.universe_epoch != clock.epoch {
        stats.re_home_epoch_mismatch += 1;
        return;
    }
    let transfer = cmd.transfer; // `ReHomeCmd` is Clone-not-Copy (the pose payload) — capture before the move
    match applied.journal_step(transfer, RE_HOME_STEP) {
        StepOutcome::FirstApply => re_home_apply(
            cmd,
            config,
            self_node,
            clock,
            dots,
            registration,
            stats,
            outbox,
        ),
        StepOutcome::AlreadyApplied => stats.re_home_redelivered += 1,
    }
    outbox.push_flow(
        config.orchestrator,
        MsgClass::Saga,
        &InterShardFlow::SagaAck(TransferControlAck::PromoteAck { transfer }),
    );
}

/// The D-37 re-home effect (monomorphic so every branch is covered ONCE here — HR5). DIVERGES from
/// `promote_apply` by CONSTRUCTING an Owned dot from the payload pose rather than flipping a ghost: the
/// re-home target is FRESH (the killed dest never replicated a ghost here). A non-Entity subject is a
/// counted no-op (`re_home_no_entity`). The created dot is clientless (no session route — the client
/// re-subscribes via the D-37/D-36 connection-plane path, owed): `AccountId(0)` + the orchestrator as an
/// inert reply sentinel, `granted` so it simulates + emits frames, born `Owned` at `cmd.new_fence`.
#[allow(clippy::too_many_arguments)]
fn re_home_apply(
    cmd: ReHomeCmd,
    config: &StubConfig,
    self_node: NodeId,
    clock: &ClockSample,
    dots: &mut Dots,
    registration: &mut GhostColliderRegistration,
    stats: &mut StubStats,
    outbox: &mut OutboundBox,
) {
    let Some(entity) = cmd.subject.transfer_subject_entity() else {
        stats.re_home_no_entity += 1;
        return;
    };
    // Never trust the network: sanitize the carried pose to finite at this ingress. (P7 grows
    // `ReHomeState::Snapshot` — this `let` becomes a `match` whose new arm needs its own coverage. TODO.)
    let ReHomeState::PoseOnly(raw) = cmd.state;
    let pose = raw.sanitized();
    // Deterministic clientless session key (entity id ↦ session) so seed-replay stays byte-identical and
    // the oracle held-set sees exactly one Owned dot for this entity.
    let session = SessionId(entity.0);
    dots.0.insert(
        session,
        Dot {
            entity,
            account: AccountId(0), // orphan: no client account until the session re-homes (D-37/D-36)
            session_fence: Fence::GENESIS,
            gateway: config.orchestrator, // inert reply sentinel — push_session_reply is never called here
            granted: true,
            input_active: false,
            adopting: false,
            authority: Authority::Owned {
                fence: cmd.new_fence,
            },
            departing: false,
            entity_fence: cmd.new_fence,
            pose,
            yaw: 0.0,
            pitch: 0.0,
            last_applied_seq: None,
            // Seed to the re-homed pose offset: this tick's swept segment is degenerate.
            prev_offset: pose.pos.offset(),
        },
    );
    stats.re_home_adopted += 1;
    // Register the (re-home) source as a ghost-neighbor + spawn its ghost — the target (owner) now drives
    // the collider feed to it, exactly as `promote_apply` does. Shared tail — see the fn doc. `self_node`
    // guards the source==dest same-node re-home (no self-ghost feed loop).
    register_and_spawn_source_ghost(
        entity,
        cmd.source,
        self_node,
        pose,
        cmd.new_fence,
        clock.local_tick,
        registration,
        outbox,
    );
}

/// Refresh a hosted ghost dot from a fed pose (1d.5b.3b): write the (sanitized) pose INTO `dot.pose`
/// and advance the kinematic mirror via `AuthorityCmd::GhostRefresh` (fence-monotone; a stale
/// `source_fence` is refused). FG-2: the dot IS the single authority/pose truth — this is the only
/// writer of a hosted ghost's pose, never a second store. Returns `true` on apply, `false` on a
/// refused (stale-fence / non-Ghost) refresh — ALL the branching lives in THIS monomorphic helper.
fn refresh_source_ghost(dot: &mut Dot, pose: StampedPose, source_fence: Fence) -> bool {
    match dot
        .authority
        .apply(AuthorityCmd::GhostRefresh { source_fence })
    {
        Ok(refreshed) => {
            dot.authority = refreshed;
            dot.pose = pose.sanitized();
            true
        }
        Err(_) => false,
    }
}

/// SOURCE-side ghost feed consumer (1d.5b.3b): the dest (new owner) drives `GhostFlow` to this shard
/// (the ghost-host) — `Spawn` establishes the feed, `Delta` is the 20Hz latest-wins pose stream,
/// `Despawn` ends it (1d.5b.3c band-exit). Each refreshes the RETAINED ghost dot's pose + fence
/// (`refresh_source_ghost`) so it stays a live kinematic collider AND keeps emitting on the source
/// sub across the handoff (the seamless fill). `Delta` is deduped by `seq` (lossy datagram). A
/// malformed body is counted + dropped, never mis-applied. Monomorphic.
fn on_ghost_flow(
    from: NodeId,
    bytes: &[u8],
    dots: &mut Dots,
    mirror: &mut SourceGhostMirror,
    stats: &mut StubStats,
) {
    let flow = match postcard::from_bytes::<InterShardFlow>(bytes) {
        Ok(InterShardFlow::Ghost(flow)) => flow,
        // Any other arm misrouted onto a Ghost carrier, or a decode failure: counted + dropped.
        _ => {
            stats.undecodable += 1;
            return;
        }
    };
    match flow {
        GhostFlow::Spawn {
            entity,
            pose,
            source_fence,
            ..
        } => {
            // Establish the feed (reliable): the mirror's `fed` latch + the dedup cursor. Refresh the
            // retained ghost dot's pose from the spawn snapshot (it may already be live from the
            // self-emit fill; this overwrites with the owner's authoritative pose).
            mirror.0.insert(
                entity,
                GhostFeedState {
                    from,
                    last_seq: 0,
                    fed: true,
                },
            );
            if let Some(dot) = dots.0.values_mut().find(|d| d.entity == entity) {
                let _ = refresh_source_ghost(dot, pose, source_fence);
            }
        }
        GhostFlow::Delta {
            entity,
            pose,
            source_fence,
            seq,
            ..
        } => {
            // Latest-wins dedup: no mirror entry, or a `seq <= last_seq` redelivery, is a stale drop.
            let Some(state) = mirror.0.get_mut(&entity) else {
                stats.ghost_delta_stale += 1;
                return;
            };
            if seq <= state.last_seq {
                stats.ghost_delta_stale += 1;
                return;
            }
            state.last_seq = seq;
            let applied = dots
                .0
                .values_mut()
                .find(|d| d.entity == entity)
                .is_some_and(|dot| refresh_source_ghost(dot, pose, source_fence));
            if applied {
                stats.ghost_delta_applied += 1;
            } else {
                // A fresh-seq Delta whose `source_fence` was stale against the ghost (or no hosted
                // dot) — the `GhostRefresh` guard refused it; counted, pose unchanged.
                stats.ghost_refresh_stale += 1;
            }
        }
        GhostFlow::Despawn { entity, .. } => {
            // Band-exit TEARDOWN (1d.5b.3c): the ghost lifecycle ENDS — remove the mirror bookkeeping
            // AND the retained ghost DOT (the source stops self-emitting + stops being a collider).
            // IDEMPOTENT: a reliable redelivery, or a stale Despawn for an entity the source has since
            // RE-OWNED (the `Owned` dot is structurally refused by `remove_retained_ghost`), tears down
            // nothing — a counted no-op (`ghost_despawn_no_host`), never a panic.
            let had_mirror = mirror.0.remove(&entity).is_some();
            let removed_dot = remove_retained_ghost(dots, entity);
            if had_mirror | removed_dot {
                stats.ghost_despawns += 1;
            } else {
                stats.ghost_despawn_no_host += 1;
            }
        }
    }
}

/// Remove the RETAINED source ghost dot for `entity` (1d.5b.3c band-exit teardown), if one is hosted
/// here. Removes ONLY a dot whose authority is a `Ghost` (a retained source ghost); a dot the source
/// has RE-OWNED (`Owned` — a re-acquisition transfer brought the entity back) is structurally REFUSED,
/// so a stale Despawn from an earlier transfer can never tear out a live owner. This Ghost-only guard
/// is the shard-LOCAL stand-in for the orchestrator's in-transfer refusal (a `vd-sim` shard cannot see
/// the orchestrator live-saga set — `DEFERRED.md` D-2). Returns whether a dot was removed. Monomorphic
/// (the find + the `matches!` false arm — a non-Ghost dot — are covered here, not in the decode arm).
#[must_use]
fn remove_retained_ghost(dots: &mut Dots, entity: EntityId) -> bool {
    let Some(session) = dots
        .0
        .iter()
        .find(|(_, d)| (d.entity == entity) & matches!(d.authority, Authority::Ghost { .. }))
        .map(|(s, _)| *s)
    else {
        return false;
    };
    dots.0.remove(&session);
    true
}

/// Whether a dot is the local granted, non-departing holder of `entity` (the self-fence target).
/// Monomorphic predicate so the chained `&&`s are covered in one helper, not the reply arm.
///
/// ⚠️ INTENTIONALLY identical-bodied to [`crossing_target`] AND [`flush_target`] — a TRIPLET, DO NOT
/// merge them (the `twin-D1` reconciliation; unrelated to DEFERRED `D-1`). The NAMES carry distinct
/// intents at disjoint call sites/shards: this finds the SOURCE's held dot to DEMOTE on the saga
/// `Demote`; `crossing_target` finds the DEST dot to APPLY a crossing; `flush_target` finds the
/// SOURCE's held dot to SHIP its pose. AUTHORITY-UNIQUE bounds each to ≤1 dot — see [`crossing_target`].
#[must_use]
fn foreign_takeover_target(dot: &Dot, entity: EntityId) -> bool {
    (dot.entity == entity) & dot.granted & !dot.departing
}

/// SOURCE side of the 1d.1 pose flush: ship the authoritative pose of the held subject dot back to
/// the orchestrator (`TransferAck::SourceFlushed`), so the saga can stamp the crossing. Read-only —
/// authority is unchanged (the abort-path `ThawSource` is the only stateful source undo). Three
/// total paths, all covered: a non-Entity subject → no-op; the subject not held here → counted
/// no-op (a stale/misrouted flush); held → ship. Monomorphic (the finder + ship are hoisted out of
/// the decode arm — HR5 branchless shim).
fn on_flush_source(flush: FlushSource, config: &StubConfig, dots: &Dots, outbox: &mut OutboundBox) {
    let Some(entity) = flush.subject.transfer_subject_entity() else {
        return; // a non-Entity subject is not a per-entity pose flush
    };
    let Some(dot) = dots.0.values().find(|d| flush_target(d, entity)) else {
        tracing::warn!(%entity, "FlushSource for an entity this shard does not hold — no pose to ship");
        return;
    };
    outbox.push_flow(
        config.orchestrator,
        MsgClass::Saga,
        &InterShardFlow::TransferAck(TransferAck::SourceFlushed {
            transfer_id: flush.transfer,
            step_id: flush.step_id,
            pose: dot.pose,
            // The source's own input drain watermark (observability; the saga's CAS watermark is
            // the GATEWAY's SourceFrozen seq, not this).
            drained_seq: dot.last_applied_seq.unwrap_or(0),
        }),
    );
}

/// Whether a dot is the local held holder of `entity` whose pose the source flushes. Monomorphic
/// (the `&`s are covered once here, not the decode arm). The THIRD member of the identical-bodied
/// `twin-D1` triplet ([`foreign_takeover_target`], [`crossing_target`]) — DO NOT merge (distinct
/// intent: this SHIPS a pose; disjoint call site; AUTHORITY-UNIQUE bounds it to ≤1 dot).
#[must_use]
fn flush_target(dot: &Dot, entity: EntityId) -> bool {
    (dot.entity == entity) & dot.granted & !dot.departing
}

/// DEST side of the 1d.1 crossing: adopt the crossed entity STATE (pose only in 1d.1). Consumes
/// only `StubCrossing` (other payload kinds are a counted no-op). If the adopt grant has not
/// flipped yet (the crossing, emitted at CAS, races ahead of the dest's adopt under the 1c.8
/// promote-before-demote model), BUFFER it for the flip to drain — do NOT journal/ack until the
/// pose is actually applied.
#[allow(clippy::too_many_arguments)]
fn on_transfer_envelope(
    env: TransferEnvelope,
    current_epoch: EpochId,
    config: &StubConfig,
    dots: &mut Dots,
    applied: &mut AppliedSteps,
    pending: &mut PendingCrossings,
    owned: &mut OwnedTransients,
    stats: &mut StubStats,
    outbox: &mut OutboundBox,
) {
    // transfer_protocol §3.3 fail-safe: a crossing leg whose `universe_epoch` does not match this
    // shard's current epoch is REFUSED — discarded, never applied/buffered/acked — so no entity is
    // ever placed at a stale celestial position. Checked here (before the payload match) so it
    // covers every payload of THIS (`Transfer`) arm — durable crossing, transient batch, initial spawn —
    // and a stale-epoch envelope cannot even strand a `PendingCrossings` entry. The OTHER pose-placing
    // ingress, `on_re_home` (the `InterShardFlow::ReHome` arm), carries the SAME guard, so §3.3 is UNIFORM
    // across both. Counted + fail-loud, mirroring the follower clock's `epoch_mismatches` counted-ignore.
    // (Epoch is exact-match by construction — unlike `schema_version`, a version FLOOR owed at D-31's
    // TLV-blob handshake.)
    if env.universe_epoch != current_epoch {
        stats.crossings_epoch_mismatch += 1;
        return;
    }
    let (entity, pose) = match env.payload {
        TransitionPayload::StubCrossing { entity, pose, .. } => (entity, pose),
        // D-7: the DEST adopts a transient batch into its uncounted `Arriving` tier + acks
        // `BatchAdopted` (the gate that lets the orchestrator emit the adopt-before-drop
        // `TransientDrop`). Handled fully here — never the durable StubCrossing dot machinery.
        TransitionPayload::TransientBatch {
            dst_realm_fence,
            items,
            ..
        } => {
            adopt_transient_batch(
                env.transfer_id,
                dst_realm_fence,
                items,
                config.orchestrator,
                owned,
                applied,
                stats,
                outbox,
            );
            return;
        }
        TransitionPayload::InitialSpawn { .. } => {
            stats.crossings_unhandled += 1;
            return;
        }
    };
    match dots
        .0
        .iter()
        .find(|(_, d)| crossing_target(d, entity))
        .map(|(s, _)| *s)
    {
        Some(session) => {
            let dot = dots.0.get_mut(&session).expect("just found");
            apply_crossing(
                dot,
                env.transfer_id,
                env.step_id,
                env.fence,
                pose,
                applied,
                config,
                stats,
                outbox,
            );
        }
        None => {
            // The adopt grant has not flipped yet: buffer for the flip to drain (no journal/ack).
            // Count a genuine FIRST buffering only — the saga re-emits the crossing at-least-once
            // (saga.rs Swapping timeout), and each redelivery while still-buffered overwrites the
            // SAME key (bounded), so an unconditional bump would inflate the counter for ONE
            // crossing. `insert` returning `None` is the first-insert signal (the same
            // FirstApply-vs-AlreadyApplied shape as `journal_step`).
            let first = pending
                .0
                .insert(
                    entity,
                    PendingCrossing {
                        transfer: env.transfer_id,
                        step_id: env.step_id,
                        fence: env.fence,
                        pose,
                    },
                )
                .is_none();
            if first {
                stats.crossings_buffered += 1;
            }
        }
    }
}

/// Whether a dot is the ADOPTED dest dot for `entity`: authority-held (`granted`), not departing.
/// Monomorphic predicate.
///
/// 1d.3 dropped a former `simulates()`-style term, and 1d.5b.3b made that DROP load-bearing: the
/// adopt dot stays `Ghost` (NOT simulating) from adopt all the way until the saga `Promote` flips it
/// in `on_saga_promote` (`apply_crossing` only STORES the pose now — it no longer promotes), so a
/// `!simulates()`-style guard would NEVER match the adopt dot and the crossing could never land. With
/// the term dropped, both the pre-promote Ghost AND a post-promote `Owned` redelivery match; the
/// journal returns `AlreadyApplied` on the redelivery, so it re-acks WITHOUT re-applying (a guard that
/// missed it would fall to `on_transfer_envelope`'s `None` arm and re-buffer into `PendingCrossings`
/// forever — a permanent strand + `crossings_buffered` inflation).
/// Sound because AUTHORITY-UNIQUE guarantees at most one granted non-departing dot per entity here,
/// AND the saga addresses the crossing envelope (`InterShardFlow::Transfer`) to the DEST node only,
/// so a widened predicate can never misland on the SOURCE's render-ready dot.
///
/// ⚠️ INTENTIONALLY identical-bodied to [`foreign_takeover_target`] AND [`flush_target`] — the
/// `twin-D1` triplet, DO NOT merge them (the tag is local DRY-legibility bookkeeping, NOT the
/// unrelated DEFERRED `D-1`). The NAMES are the documentation of intent: this finds the DEST dot to
/// APPLY a crossing; `foreign_takeover_target` finds the source's holder to DEMOTE on a takeover;
/// `flush_target` finds the source's holder to SHIP its pose. They serve disjoint call sites on
/// different shards and never fire on the same dot, so the shared body is safe — but a "DRY" merge
/// would break each call site's legibility.
#[must_use]
fn crossing_target(dot: &Dot, entity: EntityId) -> bool {
    (dot.entity == entity) & dot.granted & !dot.departing
}

/// Apply one crossing to its adopted dot (the shared immediate + drained path). Fence rule 1: a
/// crossing below the dot's recorded authority fence is a stale leftover (counted, dropped, NOT
/// acked). Otherwise consult-before-effect via the 1d.0 journal (FirstApply ⇒ sanitize + store;
/// redelivery ⇒ re-ack only), then ack the step either way.
#[allow(clippy::too_many_arguments)]
fn apply_crossing(
    dot: &mut Dot,
    transfer: TransferId,
    step_id: u32,
    fence: Fence,
    pose: StampedPose,
    applied: &mut AppliedSteps,
    config: &StubConfig,
    stats: &mut StubStats,
    outbox: &mut OutboundBox,
) {
    if fence.is_stale_against(dot.entity_fence) {
        stats.crossings_stale += 1;
        return;
    }
    if applied.journal_step(transfer, step_id) == StepOutcome::FirstApply {
        // Never trust the network: sanitize to finite at this ingress before storing. The crossing
        // STORES the pose but the dot STAYS Ghost — the `Ghost→Owned` promote RELOCATED to
        // `on_saga_promote` (1d.5b.3b) so the demote-before-promote ordering is STRICT (the dest
        // becomes Owned only on the saga `Promote`, after the source has demoted). `on_saga_promote`
        // gates its flip on THIS journal entry (`STUB_CROSSING_STEP` applied) — pose-before-promote,
        // so the relocated promote still never emits a poseless origin-default frame.
        dot.pose = pose.sanitized();
        stats.crossings_applied += 1;
    }
    outbox.push_flow(
        config.orchestrator,
        MsgClass::Saga,
        &InterShardFlow::TransferAck(TransferAck::Accepted {
            transfer_id: transfer,
            step_id,
        }),
    );
}

/// Drain a crossing buffered before its dot adopted (1d.1): on the adopt grant-flip, apply the
/// stored pose to the now-granted dot. No buffered crossing for the entity ⇒ no-op (an adopt with
/// nothing pending — e.g. a crossing that arrived after adopt, or none at all).
#[allow(clippy::too_many_arguments)]
fn drain_pending_crossing(
    dots: &mut BTreeMap<SessionId, Dot>,
    session: SessionId,
    entity: EntityId,
    applied: &mut AppliedSteps,
    pending: &mut PendingCrossings,
    config: &StubConfig,
    stats: &mut StubStats,
    outbox: &mut OutboundBox,
) {
    let Some(crossing) = pending.0.remove(&entity) else {
        return;
    };
    let dot = dots.get_mut(&session).expect("the just-adopted dot");
    apply_crossing(
        dot,
        crossing.transfer,
        crossing.step_id,
        crossing.fence,
        crossing.pose,
        applied,
        config,
        stats,
        outbox,
    );
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
/// `vd_core::geometry` (`container` / `region_signed_distance` / `should_rehome` / `ContainmentBand`).
#[allow(clippy::too_many_arguments)]
fn evaluate_realm_boundaries(
    config: Res<StubConfig>,
    clock: Res<ClockSample>,
    authority: Res<RealmAuthority>,
    regions: Res<RealmRegions>,
    mut dots: ResMut<Dots>,
    mut owned_transients: ResMut<OwnedTransients>,
    mut progress: ResMut<CrossingProgress>,
    mut membership: ResMut<ContainmentProgress>,
    mut in_flight: ResMut<RequestInFlight>,
    mut stats: ResMut<StubStats>,
    mut outbox: ResMut<OutboundBox>,
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
        .collect();
    retain_live(&mut progress.0, &live);
    retain_live(&mut membership.0, &live);
    retain_live(&mut in_flight.0, &live);

    // FA-1: the per-shard ephemeris frame context. At walk/static scale it is byte-equivalent to
    // IdentityFrames; FA-4 refreshes moving-child placements per tick. `tick_hz = 1/tick_dt_s`.
    let frames = regions.frame_context(1.0 / config.tick_dt_s);
    let ctx = CrossingCtx {
        config: &config,
        clock: &clock,
        regions: &regions.regions,
        depths: &regions.depths,
        root_realm: regions.root_realm,
        realm_fence,
        frames: &frames,
    };
    // Per owned dot (only those this shard SIMULATES) — the durable subjects. `.iter_mut()` (not
    // `.values_mut()`) so the `SessionId` key is in scope: a durable crossing carries the subject's
    // session for the saga's gateway-routed `PrepareSubscribe` (Slice 3f).
    for (session, dot) in dots.0.iter_mut().filter(|(_, d)| d.authority.simulates()) {
        let pose = dot.pose;
        let eval = evaluate_one_subject(
            &ctx,
            dot.entity,
            &pose,
            dot.authority.fence(),
            Some(*session),
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
        let eval = evaluate_one_subject(
            &ctx,
            *entity,
            &pose,
            realm_fence,
            // A transient carries no session (its batch handoff emits no session-bearing command); it
            // dispatches to the Transient arm, which never reads this. Slice 3f.
            None,
            &mut progress.0,
            &mut membership.0,
            &mut in_flight.0,
            &mut stats,
            &mut outbox,
        );
        t.prev_offset = eval.prev_offset;
    }
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
    /// The per-shard ephemeris frame context (FA-1): the input seam re-expresses a subject pose into
    /// each region's (possibly moving) frame through this before the signed distance. Byte-equivalent
    /// to [`IdentityFrames`] at walk/static scale (every region frame at the identity placement).
    frames: &'a LocalFrames,
}

/// Retain only the entries whose key is a LIVE subject (the DRY eviction primitive, Slice 3e; RLM Step 2
/// widened over `K` for `AoiMembership`'s `RealmPath` keys). A branchless `retain` over any
/// `BTreeMap<K, V>` — no per-monomorphization branch trap (HR5).
fn retain_live<K: Ord, V>(map: &mut BTreeMap<K, V>, live: &BTreeSet<K>) {
    map.retain(|k, _| live.contains(k));
}

/// The result of one subject's containment evaluation — the frame-local offset the caller records as
/// `prev_offset`. (Every re-home is now the uniform orchestrator saga — source==dest is the degenerate
/// case — so the detector emits a `CrossingRequest`/`TransientCrossingRequest` and NEVER rewrites the pose
/// in place; the crossed pose is rebound at the DEST's adopt via `rebind_pose_to_dest`, which is the same
/// node's adopt on a co-hosted re-home.)
struct SubjectEval {
    prev_offset: DVec3,
}

/// The subject's OWNING realm derived from its POSE FRAME (not a co-hosting relabel — that machinery is
/// deleted; every re-home is now the uniform orchestrator saga). `FrameRef::realm()` is `Some` for every
/// LIVE frame (System/Planet/Area/…), so the fallback to `config_realm` is reached ONLY by a frame with no
/// nameable realm (`FrameRef::GalaxySpace`, which no live shard hosts). Extracted as a MONOMORPHIC helper so
/// BOTH arms — the `Some` (a System/Planet/Area frame) and the `None` fallback — are covered by direct unit
/// tests (HR5: the fallback is otherwise uncoverable, no live shard uses `GalaxySpace`).
#[must_use]
fn owning_realm(frame: FrameRef, config_realm: RealmId) -> RealmId {
    frame.realm().unwrap_or(config_realm)
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
    entity: EntityId,
    pose: &StampedPose,
    subject_fence: Fence,
    // The subject's session — `Some` for a durable dot (the durable `CrossingRequest` carries it so the
    // orchestrator's saga can `PrepareSubscribe` to the client's gateway), `None` for a transient (whose
    // batch path emits no session-bearing command and never reaches the durable arm). Slice 3f.
    subject_session: Option<SessionId>,
    progress: &mut BTreeMap<EntityId, CrossingState>,
    membership: &mut BTreeMap<EntityId, RegionMembership>,
    in_flight: &mut BTreeMap<EntityId, TransferId>,
    stats: &mut StubStats,
    outbox: &mut OutboundBox,
) -> SubjectEval {
    let cur = pose.pos.offset();
    // The ambient root seeds the `container` fold (always `Some` — the system gated on a non-empty
    // registry before building the ctx; the `let-else` keeps this branchless of an `unwrap`).
    let Some(root_realm) = ctx.root_realm else {
        return SubjectEval { prev_offset: cur };
    };
    // FULL SCAN: advance each region's hysteretic membership bit and collect the members' depth keys.
    // Per-region + per-entity (the bitset) → each region's bit advances INDEPENDENTLY (no winner slot to
    // corrupt). Zips regions with their boot-computed depth keys (no per-tick parent walk, no index panic).
    let bits = membership.entry(entity).or_default();
    let mut members: Vec<DepthKey> = Vec::new();
    for (ix, (region, &depth_key)) in ctx.regions.iter().zip(ctx.depths.iter()).enumerate() {
        // The input-side frame seam (§2.5, FA-1): re-express the pose into the region's frame BEFORE the
        // signed distance, through the shard's own `LocalFrames` ephemeris (`ctx.frames`). At walk/static
        // scale every region is at identity so this is byte-equal to the retired `IdentityFrames`; FA-4
        // gives moving direct children a live orbital placement per tick. A frame the shard cannot name
        // (`Err`) SAFE-DEGRADES to non-member (`f64::MAX`) — never a spurious container.
        let sd = region_signed_distance(pose, region, ctx.frames).unwrap_or(f64::MAX);
        let now = region.band.member(bits.get(ix), sd);
        bits.set(ix, now);
        if now {
            members.push(depth_key);
        }
    }
    let container_realm = container(root_realm, &members);
    // The container region's PARENT provenance — the one field a `rebind_pose_to_dest` into an `Area` needs
    // (its enclosing `Planet`). A deterministic worldgen fact carried on every `RealmRegion`; threaded onto
    // the crossing request so the dest's Area frame forms. `None` for a non-Area dest (a one-field lift) and
    // when the container is the root (no matching region — never re-homes to it anyway).
    let to_parent = ctx
        .regions
        .iter()
        .find(|r| r.realm == container_realm)
        .and_then(|r| r.parent);
    // The subject's OWNING realm derived from its POSE FRAME (the relabel map is deleted): `pose.frame`'s
    // realm, else `config.realm` for a frame with no nameable realm (`owning_realm`, unit-tested both arms).
    // On a single-realm shard the pose frame IS `config.frame` (== `config.realm`'s frame) verbatim.
    let owning = owning_realm(pose.frame, ctx.config.realm);
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
        fan_out_crossing(
            ctx,
            entity,
            subject_fence,
            subject_session,
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
    subject_session: Option<SessionId>,
    to_realm: RealmId,
    // The dest realm's PARENT provenance (the container region's `parent`) — carried onto BOTH class arms'
    // requests so the dest's `rebind_pose_to_dest` forms an `Area` frame. `None` for a non-Area dest.
    to_parent: Option<RealmId>,
    state: &mut CrossingState,
    in_flight: &mut BTreeMap<EntityId, TransferId>,
    stats: &mut StubStats,
    outbox: &mut OutboundBox,
) {
    let from_realm = ctx.config.realm;
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
                    });
                    stats.crossings_requested += 1;
                }
                Entry::Occupied(_) => stats.crossings_suppressed_in_flight += 1,
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

/// Slice 3f-D4 (DEFERRED D-43 #2) — the per-tick RE-DRIVE of a STRANDED durable crossing latch. A
/// DELIVERED-but-unresolved dest (`head(Realm(to))` transiently absent — a realm shard mid-lease /
/// partitioned at P4/P5) leaves the `RequestInFlight` latch standing with NO re-emit: under CONTAINMENT
/// (task #135) `should_rehome` returns `Some(container)` every tick the container differs, but the still-
/// latched dot hits `fan_out_crossing`'s `Occupied` SUPPRESS arm each tick, so nothing re-fires while the
/// dot dwells in the (unresolved) region. This scan re-emits the SAME latched
/// `CrossingRequest` (same `(subject, subject_fence, attempt)` → byte-identical [`crossing_transfer_id`];
/// the orchestrator's `contains_key` guard absorbs a dup that already started, an unresolved one re-tries
/// the head reads) once `local_tick - last_commit_tick >= request_ttl_ticks`, then re-arms the ttl timer.
/// `request_ttl_ticks == 0` (the INERT default / every current rig) → an early return before any
/// iteration. Authority-gated exactly like `evaluate_realm_boundaries`. A BRANCHLESS shim over the
/// re-emit payload the latch carried at emit time ([`LatchedCrossing`]) — no recovery from live geometry;
/// the only branches are the ttl early-return, the authority gate, and the `>= ttl` window, each covered
/// once (HR5). Determinism: `in_flight.0` / `progress.0` are `BTreeMap`s (ordered iteration), the
/// re-mint is a pure fn of the latched fields + the universe clock, `.min(u32::MAX)` guards the cast
/// (mirrors `evaluate_one_subject`'s `since_commit` compute). The `.expect()`s are STRAIGHT-LINE
/// invariants (a held latch always carries its `CrossingState` — `retain_live` syncs both on the same
/// live set — with a `latched_crossing` payload + an armed `last_commit_tick`), matching the existing
/// session `.expect()` shape, so they add no coverable false arm.
#[allow(clippy::too_many_arguments)]
fn redrive_stranded_crossings(
    config: Res<StubConfig>,
    clock: Res<ClockSample>,
    authority: Res<RealmAuthority>,
    in_flight: Res<RequestInFlight>,
    mut progress: ResMut<CrossingProgress>,
    mut stats: ResMut<StubStats>,
    mut outbox: ResMut<OutboundBox>,
) {
    let ttl = config.request_ttl_ticks;
    if ttl == 0 {
        return;
    }
    let Some(_realm_fence) = authority.0 else {
        return;
    };
    // `in_flight` (immutable) and `progress` (mutable) are DISTINCT resources — no aliasing — so the scan
    // reads the held latches while mutating each subject's `CrossingState` in one deterministic pass.
    for (entity, _latched) in in_flight.0.iter() {
        let state = progress
            .0
            .get_mut(entity)
            .expect("a held latch always has a CrossingState (retain_live syncs both)");
        let lc = state
            .latched_crossing
            .expect("a held durable latch carries its re-emit payload");
        let last = state
            .last_commit_tick
            .expect("a held latch armed the cooldown at emit time");
        let elapsed = clock
            .local_tick
            .0
            .saturating_sub(last.0)
            .min(u32::MAX as u64) as u32;
        if elapsed >= ttl {
            let subject = DirectoryKey::Entity(*entity);
            outbox.push_flow(
                config.orchestrator,
                MsgClass::Saga,
                &InterShardFlow::CrossingRequest(CrossingRequest {
                    subject,
                    from_realm: config.realm,
                    to_realm: lc.to_realm,
                    subject_fence: lc.subject_fence,
                    session: lc.session,
                    // The SAME attempt (unchanged since the latch — the attempt bumps only on a
                    // post-abort re-latch), so the re-emitted id is byte-identical to the standing latch.
                    attempt: state.crossing_attempt,
                    // The SAME parent provenance the latch captured — so the re-drive re-mints the identical
                    // request (an Area dest's frame still forms on the re-drive).
                    to_parent: lc.to_parent,
                }),
            );
            // Re-arm the ttl timer so the next re-drive is another `ttl` ticks out.
            state.last_commit_tick = Some(clock.local_tick);
            stats.crossings_redriven += 1;
        }
    }
}

/// Per-shard system (D-7b): RE-ADVANCE every AUTHORITATIVELY-HELD transient's pose by its closed-form
/// continuity each tick — debris is a MOVING object, so a frozen-on-cut pose would teleport. Runs on
/// BOTH source and dest, each from its OWN stamped origin (no double-advance: the dest re-advances
/// from the pose it ADOPTED, which the source already advanced to its emit tick, so by composability
/// of constant-velocity motion the dest lands exactly where the source's trajectory would). The
/// UNCOUNTED `Arriving` (dest mid-flight) + `Departing` (source released) tiers are SKIPPED — they are
/// not rendered, and the dest catches up in one closed-form step at promote. Per-shard-LOCAL over
/// `owned.0`, zero cross-shard read → `par_iter_mut`-ready for the D-7c burst (zero logic change). The
/// f64 pose feeds ONLY render + the batch payload, NEVER a control discriminant (Category-A; the
/// crossing trigger compares integer cells, P4/P5).
fn readvance_transients(
    config: Res<StubConfig>,
    clock: Res<ClockSample>,
    mut owned: ResMut<OwnedTransients>,
) {
    let now = clock.universe_tick;
    for (entity, t) in owned.0.iter_mut() {
        if t.status.is_held() {
            // The SINGLE tick_dt_s chokepoint (no inline literal); `saturating_sub` enforces
            // monotonic-forward-only — a backward target yields dt=0 (no motion), never negative time.
            // accel = ZERO: a stub is empty space with no gravity field (P5's SphericalSpace introduces
            // the seed-derived analytic gravity — same primitive, non-zero accel, no system rewrite).
            let dt_s = now.0.saturating_sub(t.pose.universe_tick.0) as f64 * config.tick_dt_s;
            t.pose = kinematics::advance_continuity(
                continuity_of(*entity),
                t.pose,
                DVec3::ZERO,
                dt_s,
                now,
            );
        }
    }
}

/// SOURCE system (D-7): drain the pending transient crossings (`TransientStatus::Crossing`, the
/// TEST-seeded boundary-heuristic stand-in — the autonomous geometric trigger is P4/P5) into ONE
/// `TransientBatch` envelope per dest realm (G-TIER: one envelope per batch, never per item), and
/// transition each emitted item to `Held{outbound: Some(batch)}` (still authoritative — the source
/// holds it until the orchestrator's `TransientDrop`, adopt-before-drop). A shard without its realm
/// lease ships nothing (the Crossing items were dropped on the self-fence).
fn emit_transient_batch(
    config: Res<StubConfig>,
    clock: Res<ClockSample>,
    authority: Res<RealmAuthority>,
    mut owned: ResMut<OwnedTransients>,
    mut stats: ResMut<StubStats>,
    mut outbox: ResMut<OutboundBox>,
) {
    let Some(src_realm_fence) = authority.0 else {
        return;
    };
    // One pass: collect each Crossing item into its batch group AND mark it sent in place (no second
    // fallible lookup — the in-place mutate avoids an uncoverable None arm, HR5).
    struct Group {
        dest: NodeId,
        to_realm: RealmId,
        dst_realm_fence: Fence,
        items: Vec<TransientItem>,
    }
    let mut batches: BTreeMap<TransferId, Group> = BTreeMap::new();
    for (entity, t) in owned.0.iter_mut() {
        if let TransientStatus::Crossing {
            dest,
            to_realm,
            dst_realm_fence,
            batch,
            // The dest realm's parent is consumed IMMEDIATELY by the per-item rebind below (each item's
            // pose forms its Area frame at push time), so it rides no further into the `Group`.
            to_parent,
        } = t.status
        {
            batches
                .entry(batch)
                .or_insert(Group {
                    dest,
                    to_realm,
                    dst_realm_fence,
                    items: Vec::new(),
                })
                .items
                .push(TransientItem {
                    entity: *entity,
                    // Rebind the SOURCE-frame pose into the DEST realm's frame via the ONE machinery
                    // (HR3) the durable crossing + D-37 re-home also use. Identity through P3
                    // (position/velocity/orientation UNCHANGED, only the frame flips to `to_realm`), so
                    // the adopting shard reads the pose already expressed in its own frame. `to_parent`
                    // supplies the enclosing Planet so an `Area` dest's frame forms.
                    pose: rebind_pose_to_dest(t.pose, to_realm, to_parent),
                    state: Vec::new(),
                });
            t.status = TransientStatus::Held {
                outbound: Some(batch),
            };
        }
    }
    for (batch, g) in batches {
        let env = TransferEnvelope {
            transfer_id: batch,
            universe_epoch: clock.epoch,
            schema_version: TRANSFER_SCHEMA_VERSION,
            fence: g.dst_realm_fence,
            step_id: TRANSIENT_BATCH_STEP,
            class: DurabilityClass::Transient,
            payload: TransitionPayload::TransientBatch {
                from_realm: config.realm,
                to_realm: g.to_realm,
                src_realm_fence,
                dst_realm_fence: g.dst_realm_fence,
                source_tick: clock.local_tick,
                items: g.items,
            },
        };
        outbox.push_flow_durable(
            g.dest,
            MsgClass::Saga,
            &InterShardFlow::Transfer(env),
            Durability::Retained,
        );
        stats.transients_emitted += 1;
    }
}

/// DEST adopt of a transient batch (D-7): journal the batch step idempotently; on FIRST delivery,
/// insert each item into `OwnedTransients` as the uncounted `Arriving` tier (anchored to the batch's
/// committed realm fence); ALWAYS ack `BatchAdopted` to the orchestrator (at-least-once — the ack
/// GATES the adopt-before-drop `TransientDrop`, so a lost ack must be re-ackable). A redelivery
/// re-acks WITHOUT re-adopting (the items are already Arriving/Held).
#[allow(clippy::too_many_arguments)]
fn adopt_transient_batch(
    transfer: TransferId,
    dst_realm_fence: Fence,
    items: Vec<TransientItem>,
    orchestrator: NodeId,
    owned: &mut OwnedTransients,
    applied: &mut AppliedSteps,
    stats: &mut StubStats,
    outbox: &mut OutboundBox,
) {
    match applied.journal_step(transfer, TRANSIENT_BATCH_STEP) {
        StepOutcome::FirstApply => {
            let adopted = items.len() as u64;
            for item in items {
                // SANITIZE network input at the decode-ingress chokepoint (D-7b): a corrupt /
                // diverged sender could carry NaN/Inf, which would poison the ballistic
                // re-advance + the render — never trust the wire pose.
                let pose = item.pose.sanitized();
                owned.0.insert(
                    item.entity,
                    Transient {
                        pose,
                        anchor_fence: dst_realm_fence,
                        status: TransientStatus::Arriving { batch: transfer },
                        // Seed to the adopted pose offset: the first evaluation segment is degenerate.
                        prev_offset: pose.pos.offset(),
                    },
                );
            }
            stats.transients_adopted += adopted;
        }
        StepOutcome::AlreadyApplied => stats.transients_adopt_redelivered += 1,
    }
    outbox.push_flow(
        orchestrator,
        MsgClass::Saga,
        &InterShardFlow::TransferAck(TransferAck::BatchAdopted {
            transfer_id: transfer,
            step_id: TRANSIENT_BATCH_STEP,
        }),
    );
}

/// CA-1 S3/S4 DEST LIVENESS — re-emit `BatchAdopted` every tick for each DISTINCT batch this shard still
/// holds `Arriving`. This is the liveness half of the over-discard guarantee: the orchestrator's
/// `dest_adopted` latch is set from a `BatchAdopted` in the inbox, so re-presenting the ack every tick means
/// a transiently-lost single ack never strands the batch (the latch is re-established the next tick the ack
/// lane delivers) — and on the exact budget-maturity tick the ack is in the inbox to be latched pre-scan.
/// Gated PURELY on `Arriving`-presence (adopt is LEASE-FREE — NO authority/realm gate; a `self_fence`
/// would have already dropped the `Arriving` items via `self_fence_drop_transients`, so a self-fenced
/// holder naturally re-drives nothing). DISTINCT batches only (`BTreeSet`, deterministic order): a batch of
/// N items yields ONE ack, not N — the MMO-scale discipline (a 1000-bullet batch is one re-drive, never
/// 1000). Idempotent at the orchestrator: a re-driven `BatchAdopted` on a saga past `AwaitAdopt` is absorbed
/// by the FSM catch-all. TERMINATION: a permanently-orphaned `Arriving` item is dropped by the realm
/// self-fence (`self_fence_grace_ticks > 0`, enforced at prod boot) → the re-drive then stops (nothing
/// Arriving); in a finite test rig it is bounded by the run length.
fn redrive_pending_adoptions(
    config: Res<StubConfig>,
    owned: Res<OwnedTransients>,
    mut stats: ResMut<StubStats>,
    mut outbox: ResMut<OutboundBox>,
) {
    let mut batches: BTreeSet<TransferId> = BTreeSet::new();
    for t in owned.0.values() {
        if let TransientStatus::Arriving { batch } = t.status {
            batches.insert(batch);
        }
    }
    for batch in batches {
        outbox.push_flow(
            config.orchestrator,
            MsgClass::Saga,
            &InterShardFlow::TransferAck(TransferAck::BatchAdopted {
                transfer_id: batch,
                step_id: TRANSIENT_BATCH_STEP,
            }),
        );
        stats.batch_adopts_redriven += 1;
    }
}

/// SOURCE — phase 1 of the structural drop-before-promote (D-7b): on `TransientRelease` flip this
/// batch's `Held{outbound: Some(b)}` items to the UNCOUNTED `Departing{b}` tier (the source stops
/// counting + rendering BEFORE the dest promotes — a clean hand-off, NOT a loss) and ALWAYS ack
/// `DropApplied`(`TRANSIENT_RELEASE_STEP`) to gate the dest promote. Journaled idempotent: a
/// redelivery re-acks WITHOUT re-flipping (at-least-once). The item is RETAINED (not removed) so a
/// dest-crash-mid-promote can re-drive against it; `on_release_complete` retires it later.
#[allow(clippy::too_many_arguments)]
fn on_transient_release(
    rel: TransientHandoff,
    owned: &mut OwnedTransients,
    applied: &mut AppliedSteps,
    stats: &mut StubStats,
    config: &StubConfig,
    outbox: &mut OutboundBox,
) {
    match applied.journal_step(rel.transfer, TRANSIENT_RELEASE_STEP) {
        StepOutcome::FirstApply => {
            for t in owned.0.values_mut() {
                if let TransientStatus::Held { outbound: Some(b) } = t.status
                    && b == rel.transfer
                {
                    t.status = TransientStatus::Departing { batch: b };
                    stats.transients_handed_off += 1;
                }
            }
        }
        StepOutcome::AlreadyApplied => stats.transient_release_noop += 1,
    }
    // ALWAYS ack (at-least-once — the ack GATES the dest promote, so a lost ack must be re-ackable).
    outbox.push_flow(
        config.orchestrator,
        MsgClass::Saga,
        &InterShardFlow::TransferAck(TransferAck::DropApplied {
            transfer_id: rel.transfer,
            step_id: TRANSIENT_RELEASE_STEP,
        }),
    );
}

/// DEST — phase 2 (PROMOTE, D-7b): on `TransientDrop` flip this batch's `Arriving{b}` items to
/// authoritative `Held{outbound: None}`, re-anchoring to the batch's commit fence, and ALWAYS ack
/// `DropApplied`(`TRANSIENT_DROP_STEP`) (the promote-confirm that drives the source `ReleaseComplete`).
/// Reachable only after the source released (the orchestrator gates it on the source's `DropApplied`),
/// so the source is already uncounted — the holder set is never `{source, dest}`. Journaled idempotent.
#[allow(clippy::too_many_arguments)]
fn on_transient_promote(
    promote: TransientHandoff,
    owned: &mut OwnedTransients,
    applied: &mut AppliedSteps,
    stats: &mut StubStats,
    config: &StubConfig,
    outbox: &mut OutboundBox,
) {
    match applied.journal_step(promote.transfer, TRANSIENT_DROP_STEP) {
        StepOutcome::FirstApply => {
            for t in owned.0.values_mut() {
                if let TransientStatus::Arriving { batch } = t.status
                    && batch == promote.transfer
                {
                    t.status = TransientStatus::Held { outbound: None };
                    t.anchor_fence = promote.fence;
                    stats.transients_promoted += 1;
                }
            }
        }
        StepOutcome::AlreadyApplied => stats.transient_drop_noop += 1,
    }
    outbox.push_flow(
        config.orchestrator,
        MsgClass::Saga,
        &InterShardFlow::TransferAck(TransferAck::DropApplied {
            transfer_id: promote.transfer,
            step_id: TRANSIENT_DROP_STEP,
        }),
    );
}

/// SOURCE — phase 3 (D-7b): on `ReleaseComplete` (after the dest's promote-confirm) RETIRE this
/// batch's retained `Departing{b}` items. STATE-idempotent (like the durable retained-ghost teardown,
/// `remove_retained_ghost`): a redelivery finds no `Departing` item and is a counted no-op
/// (`transient_release_noop`). D-7d: it now ALWAYS acks `DropApplied`(`TRANSIENT_COMPLETE_STEP`) — the
/// `SourceRetired` signal that drives the saga's `BatchHandoff` tail to `Done` (so a lost
/// `ReleaseComplete` is RE-DRIVEN by the saga's `AwaitComplete` Timeout, not left until a realm
/// self-fence). The ack fires on BOTH paths (removed + already-gone) so a redelivery is still ackable;
/// the orchestrator's tombstoned saga absorbs the duplicate as a no-op.
fn on_release_complete(
    rc: TransientHandoff,
    owned: &mut OwnedTransients,
    stats: &mut StubStats,
    config: &StubConfig,
    outbox: &mut OutboundBox,
) {
    let mut to_remove: Vec<EntityId> = Vec::new();
    for (entity, t) in owned.0.iter() {
        if let TransientStatus::Departing { batch } = t.status
            && batch == rc.transfer
        {
            to_remove.push(*entity);
        }
    }
    if to_remove.is_empty() {
        stats.transient_release_noop += 1;
    } else {
        for entity in to_remove {
            owned.0.remove(&entity);
        }
    }
    outbox.push_flow(
        config.orchestrator,
        MsgClass::Saga,
        &InterShardFlow::TransferAck(TransferAck::DropApplied {
            transfer_id: rc.transfer,
            step_id: TRANSIENT_COMPLETE_STEP,
        }),
    );
}

/// SOURCE — D-7d dead-DEST resolution: on `TransientAbandon` (the dest died mid-handoff, so the only
/// promote target is gone) DROP this batch's retained items — `Departing{b}` (already released) OR
/// `Held{outbound: Some(b)}` (the dest died before the source released) — as an ACCOUNTED loss: each is
/// removed + bucketed into `transients_lost_in_handover` by kind (the SAME budget the realm self-fence
/// feeds — DRY, so `verify_transient_loss_budget` reads ONE honest population) + counted in
/// `transients_departure_cancelled`. Journaled idempotent (`TRANSIENT_ABANDON_STEP`): a redelivery
/// short-circuits, so the loss counts EXACTLY once. A corrupt kind tag (HR2 — never decode-to-default)
/// is still removed but NOT bucketed (no kind to attribute the loss to — the `Err` arm).
fn on_transient_abandon(
    abandon: TransientHandoff,
    owned: &mut OwnedTransients,
    applied: &mut AppliedSteps,
    stats: &mut StubStats,
) {
    match applied.journal_step(abandon.transfer, TRANSIENT_ABANDON_STEP) {
        StepOutcome::FirstApply => {
            let mut to_remove: Vec<EntityId> = Vec::new();
            for (entity, t) in owned.0.iter() {
                let in_batch = match t.status {
                    TransientStatus::Departing { batch } => batch == abandon.transfer,
                    TransientStatus::Held {
                        outbound: Some(batch),
                    } => batch == abandon.transfer,
                    _ => false,
                };
                if in_batch {
                    to_remove.push(*entity);
                }
            }
            for entity in to_remove {
                owned.0.remove(&entity);
                stats.transients_departure_cancelled += 1;
                if let Ok(kind) = EntityKind::from_tag(entity.kind_tag()) {
                    *stats.transients_lost_in_handover.entry(kind).or_insert(0) += 1;
                }
            }
        }
        StepOutcome::AlreadyApplied => stats.transient_release_noop += 1,
    }
}

/// DEST — R-6d3c NEVER-restart closure: on `TransientDiscard` (the source died in
/// `BatchHandoff::AwaitAdopt`, PRE-adopt, so the batch is being counted lost) REMOVE any
/// `Arriving{batch==transfer}` item as an ACCOUNTED loss AND POISON `(transfer, TRANSIENT_BATCH_STEP)`
/// — so a LATE outbox replay of the batch adopts as `AlreadyApplied` and never re-inserts an orphan
/// (the exact silent-loss interleave; `adopt_transient_batch` hits its `AlreadyApplied` arm,
/// `stub.rs` `TRANSIENT_BATCH_STEP`). Journaled idempotent by `(transfer, TRANSIENT_DISCARD_STEP)`: a
/// redelivery short-circuits, so the loss counts EXACTLY once. Ack-FREE — the resolving saga is
/// terminal (mirroring `on_transient_abandon`). A corrupt kind tag (HR2 — never decode-to-default) is
/// still removed + counted in `transients_discarded_source_crash` but NOT bucketed (the `Err` arm).
fn on_transient_discard(
    discard: TransientHandoff,
    owned: &mut OwnedTransients,
    applied: &mut AppliedSteps,
    stats: &mut StubStats,
) {
    match applied.journal_step(discard.transfer, TRANSIENT_DISCARD_STEP) {
        StepOutcome::FirstApply => {
            // POISON the adopt: record `(transfer, TRANSIENT_BATCH_STEP)` so a late replayed
            // `adopt_transient_batch` is `AlreadyApplied` (never re-inserts). The return value is
            // ignored — we only need the row present. (If the adopt already ran, this is a harmless
            // no-op insert; its `Arriving` items are then removed by the loop below.)
            let _ = applied.journal_step(discard.transfer, TRANSIENT_BATCH_STEP);
            let mut to_remove: Vec<EntityId> = Vec::new();
            for (entity, t) in owned.0.iter() {
                if let TransientStatus::Arriving { batch } = t.status
                    && batch == discard.transfer
                {
                    to_remove.push(*entity);
                }
            }
            for entity in to_remove {
                owned.0.remove(&entity);
                stats.transients_discarded_source_crash += 1;
                if let Ok(kind) = EntityKind::from_tag(entity.kind_tag()) {
                    *stats.transients_lost_in_handover.entry(kind).or_insert(0) += 1;
                }
            }
        }
        StepOutcome::AlreadyApplied => stats.transient_release_noop += 1,
    }
}

/// On a realm SELF-FENCE (the lease was taken over / revoked), DROP every transient this shard
/// tracked (D-7) — they were anchored to the now-lost lease with NO hand-off: a counted LOSS (the
/// declared-loss path; D-7b's `LossBudget` gate reads `transients_dropped`). Durable dots are
/// RETAINED (authority.rs owns them); only the held-set-anchored transients are lost. The `+= 0` on
/// an empty set is a covered straight-line no-op (the happy path never loses the realm).
fn self_fence_drop_transients(owned: &mut OwnedTransients, stats: &mut StubStats) {
    // GROSS eviction count (ops visibility) — every tier.
    stats.transients_dropped += owned.0.len() as u64;
    // HANDOVER-attributable per-kind LOSS (the D-7b.3 budget gate reads THIS, never the gross count):
    // only `is_in_handover()` items are an in-flight transfer loss; a settled `Held{outbound: None}`
    // is a resident eviction OUT of budget scope. A corrupt kind tag (HR2 — never decode-to-default)
    // is counted GROSS but NOT bucketed (the `Err` arm — it has no kind to attribute the loss to).
    for (entity, t) in owned.0.iter() {
        if t.status.is_in_handover()
            && let Ok(kind) = EntityKind::from_tag(entity.kind_tag())
        {
            *stats.transients_lost_in_handover.entry(kind).or_insert(0) += 1;
        }
    }
    owned.0.clear();
}

/// D-3 Slice 5 — the PROACTIVE self-fence (fence rule 4, the partition cure). When this shard HOLDS its
/// realm but has had no round-trip confirmation (`RealmConfirmedAt`) within `self_fence_grace_ticks` of
/// its own `local_tick` — i.e. it is partitioned from the orchestrator, so the reactive `realm_recheck`
/// reply never arrives — it HARD-STOPS its own authority BEFORE the orchestrator's reassign horizon
/// opens (`should_reap` reaps only past `lease_expires + max_self_fence_grace`; the split-brain-safe
/// ordering `lease_ttl < grace` with `THETA_MAX*grace < lease_ttl + max` is enforced orchestrator-side by
/// `DirectoryTuning::validate`). The schedule runs
/// this before the egress systems, so a holder that self-fences this tick drops its held transients and
/// emits NO frames — two owners can never both reach clients. The reactive `realm_recheck` reply path
/// (a reply showing a takeover) remains the prompt cure while the link is ALIVE; this is the only path
/// that fires when it is NOT. `local_tick` is the partition-surviving clock (it advances every
/// `step_tick` regardless of `ClockSync`), so the measurement holds even with the universe clock frozen.
fn self_fence_lapsed_realm(
    config: Res<StubConfig>,
    clock: Res<ClockSample>,
    confirmed: Res<RealmConfirmedAt>,
    mut authority: ResMut<RealmAuthority>,
    mut owned_transients: ResMut<OwnedTransients>,
    mut stats: ResMut<StubStats>,
) {
    // The split-brain-critical predicate is the ONE shared `lease_self_fence_due` (DRY across every
    // authority holder — the shard's Realm here, the gateway's Session in connection-plane — so the
    // fence-rule-4 timing can never drift between them). `held` = this shard holds its realm.
    if crate::directory::lease_self_fence_due(
        authority.0.is_some(),
        config.self_fence_grace_ticks,
        config.realm_recheck_interval,
        clock.local_tick,
        confirmed.0,
    ) {
        authority.0 = None;
        self_fence_drop_transients(&mut owned_transients, &mut stats);
        stats.realm_self_fenced_lapsed += 1;
    }
}

/// Apply a realm-head affirm to the right authority store — the PRIMARY realm (`config.realm` ↦
/// `RealmAuthority`, the single-realm machinery, byte-identical) OR a CO-HOSTED CHILD realm
/// (`CoHostedAuthority`, the additive co-host store). Monomorphic so every branch is covered ONCE
/// (HR5). `record` is the head at the moment of the read (`None` = the record was revoked):
/// - PRIMARY realm, OURS  → hold `RealmAuthority` at the recorded fence + re-arm the self-fence clock.
/// - PRIMARY realm, FOREIGN/None → self-fence (drop `RealmAuthority` + declare the transients lost).
/// - CO-HOSTED realm, OURS → hold this child's `CoHostedAuthority` fence (independent of the primary).
/// - CO-HOSTED realm, FOREIGN/None → drop this child's entry (the shard no longer co-hosts it). The
///   transient loss path is PRIMARY-only (transients are anchored to `RealmAuthority`, never a child).
#[allow(clippy::too_many_arguments)]
fn affirm_realm_head(
    realm: RealmId,
    record: Option<&vd_wire::seams::directory::OwnerRecord>,
    identity: &NodeIdentity,
    config: &StubConfig,
    clock: &ClockSample,
    authority: &mut RealmAuthority,
    confirmed: &mut RealmConfirmedAt,
    cohosted: &mut CoHostedAuthority,
    owned_transients: &mut OwnedTransients,
    stats: &mut StubStats,
) {
    let ours = record.map(|r| r.authority) == Some(AuthorityRef::Shard(identity.node_id));
    if realm == config.realm {
        // The PRIMARY realm — the single-realm authority path, UNCHANGED.
        if ours {
            authority.0 = record.map(|r| r.fence);
            // D-3 Slice 5: a round-trip that AFFIRMS ownership re-arms the self-fence deadline — the
            // holder has just heard from the directory, so it is provably not partitioned now.
            confirmed.0 = clock.local_tick;
        } else {
            // Taken over (P2 transfer / reassignment) or revoked (record None): SELF-FENCE immediately
            // (fence rule 4) — drop authority and stop emitting so a stale old owner cannot affect clients.
            tracing::warn!("realm lease no longer held by this shard — self-fencing");
            authority.0 = None;
            // D-7: the transients were anchored to the now-lost lease, with no hand-off — a counted LOSS
            // (the declared-loss path; durable dots are retained by authority.rs).
            self_fence_drop_transients(owned_transients, stats);
        }
    } else if ours {
        // A CO-HOSTED CHILD realm we still hold: refresh its own fence (independent of the primary).
        cohosted
            .0
            .insert(realm, record.map_or(Fence::GENESIS, |r| r.fence));
    } else {
        // A CO-HOSTED CHILD realm taken over or revoked: drop the co-host entry (no transient loss — a
        // child realm never anchors this shard's transients; those ride `RealmAuthority`).
        cohosted.0.remove(&realm);
    }
}

/// Handle a directory reply: realm-lease and entity-grant confirmations.
#[allow(clippy::too_many_arguments)]
fn on_directory_reply(
    bytes: &[u8],
    identity: &NodeIdentity,
    config: &StubConfig,
    clock: &ClockSample,
    authority: &mut RealmAuthority,
    confirmed: &mut RealmConfirmedAt,
    cohosted: &mut CoHostedAuthority,
    dots: &mut Dots,
    applied: &mut AppliedSteps,
    pending: &mut PendingCrossings,
    registration: &mut GhostColliderRegistration,
    owned_transients: &mut OwnedTransients,
    in_flight: &mut RequestInFlight,
    progress: &mut CrossingProgress,
    stats: &mut StubStats,
    outbox: &mut OutboundBox,
) {
    // Decode the Saga-class envelope once and dispatch by arm. The orchestrator wraps a directory
    // answer in DirectoryReply; the 1d.1 transfer machinery adds the SOURCE's FlushSource request
    // and the DEST's Transfer (StubCrossing) envelope. D-7 adds the DEST's `TransientBatch` adopt
    // (inside `on_transfer_envelope`) + the orchestrator's `TransientDrop`. Other arms
    // (Ghost/Directory/Saga/SagaAck/TransferAck) never target a stub inbound and are ignored.
    let reply = match postcard::from_bytes::<InterShardFlow>(bytes) {
        Ok(InterShardFlow::DirectoryReply(reply)) => reply,
        // SOURCE: ship the held subject's pose (1d.1).
        Ok(InterShardFlow::FlushSource(flush)) => {
            on_flush_source(flush, config, dots, outbox);
            return;
        }
        // DEST: adopt the crossed entity state — a durable `StubCrossing` (1d.1) OR a `TransientBatch`
        // adopt-as-Arriving (D-7); both ride the `Transfer` arm, split by payload inside.
        Ok(InterShardFlow::Transfer(env)) => {
            on_transfer_envelope(
                env,
                clock.epoch,
                config,
                dots,
                applied,
                pending,
                owned_transients,
                stats,
                outbox,
            );
            return;
        }
        // D-7b structural drop-before-promote: SOURCE release (`Held→Departing` + ack), DEST promote
        // (`Arriving→Held` + ack), SOURCE complete (retire the `Departing` copy). The holder set
        // transits `{source}→{}→{dest}`, never `{source,dest}`.
        Ok(InterShardFlow::TransientRelease(rel)) => {
            on_transient_release(rel, owned_transients, applied, stats, config, outbox);
            return;
        }
        Ok(InterShardFlow::TransientDrop(promote)) => {
            on_transient_promote(promote, owned_transients, applied, stats, config, outbox);
            return;
        }
        Ok(InterShardFlow::ReleaseComplete(rc)) => {
            on_release_complete(rc, owned_transients, stats, config, outbox);
            return;
        }
        // SOURCE: the D-7d dead-DEST resolution — abandon this batch's retained copy as an accounted
        // loss (no ack: the resolving saga is already terminal; the fabric delivers reliably to the
        // live source, and a lost abandon falls back to the realm self-fence loss path).
        Ok(InterShardFlow::TransientAbandon(abandon)) => {
            on_transient_abandon(abandon, owned_transients, applied, stats);
            return;
        }
        // DEST: the R-6d3c NEVER-restart resolution — the source died in AwaitAdopt (pre-adopt), so
        // remove any late-replayed Arriving copy of this batch AND poison the adopt as an accounted loss
        // (no ack: the resolving saga is already terminal, exactly like TransientAbandon).
        Ok(InterShardFlow::TransientDiscard(discard)) => {
            on_transient_discard(discard, owned_transients, applied, stats);
            return;
        }
        // SOURCE: the saga-pushed ordered Demote (1d.5b.1) — Owned→Frozen→Ghost + DemoteAck. Slice 3e:
        // the durable COMMIT terminal at the source, so it also POSITIVELY clears the subject's
        // `RequestInFlight` crossing latch (a triggered durable crossing has committed — the entity is
        // free to trigger again from its new home).
        Ok(InterShardFlow::Demote(cmd)) => {
            on_saga_demote(cmd, config, clock, dots, in_flight, stats, outbox);
            return;
        }
        // SOURCE (Slice 3e): the orchestrator GRANTED a resolved dest for a `TransientCrossingRequest`
        // — flip the source Transient to `Crossing` so `emit_transient_batch` ships it this/next tick.
        // The grant's four fields EXACTLY match `TransientStatus::Crossing`. A grant for a
        // non-Entity subject or an unknown/settled transient is a counted no-op (degrade, never panic).
        Ok(InterShardFlow::TransientCrossingGrant(g)) => {
            on_transient_crossing_grant(g, owned_transients, stats);
            return;
        }
        // SOURCE (Slice 3e): the crossing resolve/start saga ABORTED pre-CAS — CLEAR the subject's
        // `RequestInFlight` latch IF it still holds THIS transfer id (the positive re-cross signal; the
        // full abort EGRESS is Slice 3f, but the consumer arm lands now so the wire arm is used). A
        // mismatch (a stale abort for a superseded/re-latched transfer) is a counted no-op.
        Ok(InterShardFlow::CrossingAborted(a)) => {
            on_crossing_aborted(a, in_flight, progress, stats, outbox, config.orchestrator);
            return;
        }
        // DEST: the saga-pushed ordered Promote (1d.5b.3b) — the REAL Ghost→Owned promoter + the
        // dest read-sub announce + the source-ghost feed registration/Spawn + PromoteAck. The dest
        // normally holds its realm before the orchestrator routes a Promote to it (the realm-owner
        // invariant); if it does NOT (a realm self-fence raced the Promote — unreachable in P2, no
        // realm-revoke producer; reachable only at P8/P10 multi-realm mobility), DROP the Promote as
        // a counted no-op so the saga's `Promoting`-timeout re-drives it — DEGRADE, never panic
        // (mirroring every sibling handler; the realm-mobility re-drive is owed with D-3/Slice-2).
        Ok(InterShardFlow::Promote(cmd)) => {
            let Some(realm_fence) = authority.0 else {
                stats.promote_without_realm += 1;
                return;
            };
            on_saga_promote(
                cmd,
                config,
                identity.node_id,
                clock,
                dots,
                applied,
                registration,
                realm_fence,
                stats,
                outbox,
            );
            return;
        }
        // D-37 forward re-home ADOPT (the target creates the entity Owned from the carried pose). The
        // realm guard mirrors Promote (the re-home target is committed by the orchestrator CAS so it
        // normally holds its realm; the guard is a counted degrade-never-panic no-op, never an unwrap).
        Ok(InterShardFlow::ReHome(cmd)) => {
            let Some(_realm_fence) = authority.0 else {
                stats.re_home_without_realm += 1;
                return;
            };
            on_re_home(
                cmd,
                config,
                identity.node_id,
                clock,
                dots,
                applied,
                registration,
                stats,
                outbox,
            );
            return;
        }
        // CA-1 S3: the orchestrator's AwaitAdopt liveness PROBE. A COUNTED NO-OP — the probe's whole signal
        // is its SEND OUTCOME at the orchestrator (a dead source's send fails → `NodeUnreachable`); a LIVE
        // source that receives it need do nothing but exist (its regular lease-renewal inbound is what clears
        // stale unreachable evidence). Counted for ops visibility; never a state change.
        Ok(InterShardFlow::ReSolicitBatch(_)) => {
            stats.re_solicits_received += 1;
            return;
        }
        Ok(_) => return,
        Err(_) => {
            stats.undecodable += 1;
            tracing::error!("undecodable saga-class message");
            return;
        }
    };
    match reply {
        DirectoryReply::Head {
            key: DirectoryKey::Realm(realm),
            record,
        } => {
            // A realm-head affirm for the PRIMARY realm drives the single-realm authority machinery
            // (unchanged); one for a CO-HOSTED CHILD realm drives only its own map entry — so the
            // co-hosting affirm reuses THIS one arm (no new dispatch arm), byte-identical for a
            // single-realm shard (the `else` branch is never reached when `held_realms == {realm}`).
            // Branching is hoisted into the monomorphic `affirm_realm_head` (HR5): the arm body is a
            // branchless shim.
            affirm_realm_head(
                realm,
                record.as_ref(),
                identity,
                config,
                clock,
                authority,
                confirmed,
                cohosted,
                owned_transients,
                stats,
            );
        }
        DirectoryReply::Head {
            key: DirectoryKey::Entity(entity),
            record: Some(record),
        } => {
            // The avatar's authority is now RECORDED: it becomes held, visible,
            // and attachable (fence rule 2 — authority derives from the directory).
            let Some(realm_fence) = authority.0 else {
                return; // grant raced ahead of the realm lease: retry resolves it
            };
            if record.authority == AuthorityRef::Shard(identity.node_id) {
                // OURS: flip granted + stamp the recorded fence. A login attach pushes
                // SessionAttached and Promotes Ghost→Owned (now simulates); a transfer-dest ADOPT
                // (1c.8) suppresses the attach (R2 — source owns the client) and STAYS a Ghost (no
                // pose carried; the saga `Promote` flips it in `on_saga_promote`, 1d.5b.3b —
                // `apply_crossing` only STORES the crossed pose). The
                // branching is hoisted into `flip_grant` → `GrantFlip`: LoggedIn pushes
                // SessionAttached; Adopted drains the buffered crossing (if any) NOW; NoOp is a
                // duplicate-grant idempotent no-op.
                match flip_grant(&mut dots.0, entity, record.fence, config, realm_fence) {
                    GrantFlip::LoggedIn(egress) => {
                        push_session_reply(outbox, egress.gateway, &egress.reply);
                    }
                    GrantFlip::Adopted { session } => {
                        // 1d.5b.3b: NO SubscriptionReady here (it moved to on_saga_promote). Just
                        // drain any buffered crossing onto the now-adopted Ghost dot (it stays Ghost
                        // and emits nothing until the saga Promote flips it Owned).
                        drain_pending_crossing(
                            &mut dots.0,
                            session,
                            entity,
                            applied,
                            pending,
                            config,
                            stats,
                            outbox,
                        );
                    }
                    GrantFlip::NoOp => {}
                }
            } else {
                // FOREIGN owner of an entity this shard holds: IGNORED here (1d.5b.2). The source
                // self-fence is now driven SOLELY by the saga-pushed ordered `Demote`
                // (`on_saga_demote`) — the 1c.8 cooperative poll that DISCOVERED a foreign takeover
                // from a directory head is gone. The EXPLICIT else keeps the outer `if`'s FALSE arm
                // a covered branch (a pre-grant race still surfaces a foreign-owner head, exercised by
                // `foreign_entity_grants_and_pregrant_races_are_survived`) — never an implicit
                // uncovered else.
            }
        }
        DirectoryReply::Head {
            key: DirectoryKey::Entity(entity),
            record: None,
        } => {
            // The revoke is RECORDED (no record remains): finish the release —
            // despawn and confirm to the gateway.
            let departed: Vec<SessionId> = dots
                .0
                .iter()
                .filter(|(_, d)| (d.entity == entity) & d.departing)
                .map(|(s, _)| *s)
                .collect();
            for session in departed {
                let dot = dots.0.remove(&session).expect("just found");
                push_session_reply(
                    outbox,
                    dot.gateway,
                    &ShardToGateway::SessionDetached { session },
                );
            }
        }
        // Headless realm reads, CAS results, clock answers: no obligation in P1.
        _ => {}
    }
}

/// DEST-side ghost collider FEED (1d.5b.3b): for each registered ghost-neighbor, stream a
/// `GhostFlow::Delta` of the OWNED entity's live pose to the ghost-host (the transfer source). The
/// owner DRIVES the feed (owner → ghost-host). Runs AFTER `process_inbound` (this tick's promote
/// registered the neighbor + the dot is Owned) and BEFORE `emit_frames` (the source consumes the
/// delta it received this tick before emitting). A registration whose entity is not currently Owned
/// here (no dot / a non-`simulates()` dot) is a counted no-op. SCALE: single-neighbor today; a
/// band-driven multi-neighbor owner fans Delta to every neighbor — the `seq` is per-neighbor in the
/// body, so multi-neighbor breaks encode-once (a band-ghost-era optimization, DEFERRED).
fn feed_source_ghosts(
    config: Res<StubConfig>,
    clock: Res<ClockSample>,
    dots: Res<Dots>,
    mut registration: ResMut<GhostColliderRegistration>,
    mut stats: ResMut<StubStats>,
    mut outbox: ResMut<OutboundBox>,
) {
    // The overlap band, seed-derived from the shard's per-tick travel (no inline literal — the
    // factors live in `core::geometry`). Velocity-safe by construction; its destroy edge is many
    // per-tick steps out, so a ghost SPAWNED in-band at the crossing exits only after the entity has
    // walked well past the demote→promote→release handoff (band-exit is strictly POST-release).
    let band = OverlapBand::for_motion(config.move_speed_mps * config.tick_dt_s);
    let mut exited: Vec<EntityId> = Vec::new();
    for (entity, neighbor) in registration.0.iter_mut() {
        let Some(dot) = dots.0.values().find(|d| d.entity == *entity) else {
            stats.ghost_feed_skipped += 1;
            continue;
        };
        if !dot.authority.simulates() {
            stats.ghost_feed_skipped += 1;
            continue;
        }
        if ghost_band_exited(&band, neighbor.anchor, &dot.pose) {
            // BAND-EXIT (1d.5b.3c): the owned entity left the overlap band — DESPAWN the ghost on the
            // RELIABLE carrier (a lost Despawn would leak the collider) + DEREGISTER the feed. The
            // source tears the ghost down on receipt (`on_ghost_flow`). No vanish: the dest is the
            // sole render source by now (the destroy edge is sized past the handoff window).
            outbox.push_flow_durable(
                neighbor.source,
                MsgClass::GhostReliable,
                &InterShardFlow::Ghost(GhostFlow::Despawn {
                    entity: *entity,
                    source_fence: dot.authority.fence(),
                }),
                Durability::Retained,
            );
            exited.push(*entity);
            stats.ghost_band_exits += 1;
        } else {
            outbox.push_flow(
                neighbor.source,
                MsgClass::GhostDelta,
                &InterShardFlow::Ghost(GhostFlow::Delta {
                    entity: *entity,
                    pose: dot.pose,
                    source_fence: dot.authority.fence(),
                    source_tick: clock.local_tick,
                    seq: neighbor.seq,
                }),
            );
            neighbor.seq += 1;
        }
    }
    // Deregister the exited ghosts (the feed stops; the source teardown is driven by the Despawn).
    for entity in exited {
        registration.0.remove(&entity);
    }
}

/// Whether the owned entity has EXITED the overlap band anchored at its boundary crossing (1d.5b.3c):
/// it is no longer a member — it has travelled past the band's destroy edge from `anchor`. A
/// branchless monomorphic shim: the membership hysteresis lives in `OverlapBand::update_membership`
/// (fully covered in `core`), so the dest's band-exit decision adds no uncovered branch here. The
/// ghost was SPAWNED in-band at the crossing (distance 0 = a member), so `was_member` is always `true`.
#[must_use]
fn ghost_band_exited(band: &OverlapBand, anchor: DVec3, pose: &StampedPose) -> bool {
    !band.update_membership(true, (pose.pos.offset() - anchor).length())
}

/// Whether a hosted ghost has a LIVE feed (1d.5b.3b): has-ever-been-fed-and-not-despawned (the
/// mirror's `fed` latch). NOT fed-this-tick — `GhostDelta` is lossy, so a dropped delta must NOT
/// blink the avatar (it holds its last fed pose). Monomorphic (the Some/None split is here).
#[must_use]
fn is_fed_ghost(mirror: &SourceGhostMirror, d: &Dot) -> bool {
    mirror.0.get(&d.entity).is_some_and(|s| s.fed)
}

/// Whether a dot is a RETAINED source ghost (1d.5b.3b): a granted, non-departing `Ghost` whose
/// `source_fence` is POST-GENESIS — it holds its last-Owned pose, so it SELF-EMITS that pose on its
/// sub from the demote instant, filling the demote→Promote window the dest feed cannot reach
/// (nothing is Owned then to produce the pose). The `source_fence != GENESIS` term EXCLUDES the
/// pre-promote DEST-adopt Ghost (which holds `GENESIS` and has no real pose — it must stay silent
/// until `on_saga_promote` flips it Owned); `granted & !departing` excludes a pre-grant provisional
/// Ghost. Monomorphic (the state destructure + guard live here, not in the filter closure).
#[must_use]
fn is_retained_ghost(d: &Dot) -> bool {
    let retained_source = matches!(
        d.authority,
        Authority::Ghost { source_fence, .. } if source_fence != Fence::GENESIS
    );
    retained_source & d.granted & !d.departing
}

/// EMIT-eligibility (1d.5b.3b): a dot's pose is emitted if it SIMULATES (Owned — the authority
/// truth) OR is a fed ghost OR is a retained source ghost — the DERIVED union via BITWISE `|` (never
/// `||`, so each operand's false arm stays coverable; HR5). Authority/input/oracle truth remains
/// `simulates()` ALONE — a ghost emits its kinematic mirror but integrates/accepts NOTHING (FG-2).
#[must_use]
fn emits(mirror: &SourceGhostMirror, d: &Dot) -> bool {
    d.authority.simulates() | is_fed_ghost(mirror, d) | is_retained_ghost(d)
}

/// Emit one fence-stamped frame per tick to every gateway with an attached session.
/// No realm authority ⇒ no frames (an unowned shard is silent, never speculative).
fn emit_frames(
    config: Res<StubConfig>,
    clock: Res<ClockSample>,
    authority: Res<RealmAuthority>,
    dots: Res<Dots>,
    mirror: Res<SourceGhostMirror>,
    mut counter: ResMut<FrameCounter>,
    mut outbox: ResMut<OutboundBox>,
) {
    let Some(realm_fence) = authority.0 else {
        return;
    };
    // EMIT-eligibility is the DERIVED `simulates() | is_fed_ghost | is_retained_ghost` (1d.5b.3b):
    // an Owned dot (the authority truth) emits; ALSO a fed ghost (the dest-driven collider feed) and
    // a retained source ghost (self-emitting its last-Owned pose to fill the demote→Promote handoff)
    // emit their kinematic mirror — so the avatar renders CONTINUOUSLY across the strict handoff (no
    // vanish). A Frozen or pre-grant-provisional Ghost still emits NOTHING.
    let mut gateways: Vec<NodeId> = dots
        .0
        .values()
        .filter(|d| emits(&mirror, d))
        .map(|d| d.gateway)
        .collect();
    gateways.sort_unstable();
    gateways.dedup();
    if gateways.is_empty() {
        return;
    }
    let entities: Vec<EntitySnap> = dots
        .0
        .values()
        .filter(|d| emits(&mirror, d))
        .map(|d| EntitySnap {
            entity: d.entity,
            pose: d.pose,
        })
        .collect();
    // Partition BY CONTENT so no datagram exceeds the MTU budget (audit GW-1): a
    // full-world snapshot ships as several independent self-contained frames. Per
    // connection_plane.md §6.3 EVERY chunk of one tick carries the SAME frame_id +
    // celestial_tick (each chunk self-contained, latest-wins) — the client merges
    // them and treats only a STRICTLY older frame_id as stale, so a reordered
    // sibling chunk of the same tick is never dropped. The counter advances once per
    // tick. The shared partitioner is the ONE place every shard type does this.
    let frame_id = counter.0;
    counter.0 += 1;
    for chunk in partition_entities(&entities, config.snapshot_datagram_budget) {
        let snapshot = SnapshotDatagram {
            // The shard always stamps sub 0; the gateway re-tags per session.
            sub: SubId(0),
            frame_id,
            source_tick: clock.local_tick,
            universe_tick: clock.universe_tick,
            entities: chunk,
        };
        let snapshot_bytes =
            postcard::to_allocvec(&snapshot).expect("closed wire enums serialize infallibly");
        let frame = ShardToGateway::Frame {
            realm_fence,
            source_tick: clock.local_tick,
            snapshot_bytes,
        };
        // ONE shared body per chunk, cloned (refcount bump) to every subscribing
        // gateway — never an O(entities) copy per gateway (SCALE-1).
        let bytes = crate::io::bytes(
            postcard::to_allocvec(&frame).expect("closed wire enums serialize infallibly"),
        );
        for &gateway in &gateways {
            outbox.0.push((
                gateway,
                MsgClass::Snapshot,
                bytes.clone(),
                Durability::Ephemeral,
            ));
        }
    }
}

/// Emit the shard's authored REALM placements (moving planet/station/ship boxes) to observers each tick
/// (FA-2c) — the render-plane twin of [`emit_frames`]. No realm authority ⇒ silent. NO moving child ⇒
/// silent (byte-identical at walk/static scale — `authored_realm_snaps` is empty). Otherwise it ships a
/// [`RealmSnapshotDatagram`] (MTU-partitioned by the shared `partition_realms`) as a
/// [`MsgClass::RealmSnapshot`] datagram to every gateway with an EMITTING observer — the SAME recipients
/// the entity snapshot reaches. Latest-wins, unreliable; realms are world observation, never authority.
#[allow(clippy::too_many_arguments)]
fn emit_realm_frames(
    config: Res<StubConfig>,
    clock: Res<ClockSample>,
    authority: Res<RealmAuthority>,
    regions: Res<RealmRegions>,
    dots: Res<Dots>,
    mirror: Res<SourceGhostMirror>,
    mut counter: ResMut<RealmFrameCounter>,
    mut outbox: ResMut<OutboundBox>,
) {
    let Some(realm_fence) = authority.0 else {
        return;
    };
    // The authored moving-child rows — EMPTY at walk/static scale ⇒ nothing ships (byte-identical).
    let realms =
        regions.authored_realm_snaps(config.realm, 1.0 / config.tick_dt_s, clock.universe_tick);
    if realms.is_empty() {
        return;
    }
    // The observers present: the SAME gateway set the entity snapshot reaches (the emitting dots). No
    // observer ⇒ no realm ship (a shard authors its children's motion regardless, but ships only to a
    // recipient — the `emit_frames` gating, verbatim).
    let mut gateways: Vec<NodeId> = dots
        .0
        .values()
        .filter(|d| emits(&mirror, d))
        .map(|d| d.gateway)
        .collect();
    gateways.sort_unstable();
    gateways.dedup();
    if gateways.is_empty() {
        return;
    }
    let frame_id = counter.0;
    counter.0 += 1;
    // Partition BY CONTENT so no realm datagram exceeds the MTU budget (audit GW-1) — the SAME shared
    // partitioner every shard's entity snapshot uses; each chunk is a self-contained latest-wins frame.
    for chunk in partition_realms(&realms, config.snapshot_datagram_budget) {
        let snapshot = RealmSnapshotDatagram {
            sub: SubId(0),
            frame_id,
            source_tick: clock.local_tick,
            universe_tick: clock.universe_tick,
            realms: chunk,
        };
        let realm_snapshot_bytes =
            postcard::to_allocvec(&snapshot).expect("closed wire enums serialize infallibly");
        let frame = ShardToGateway::RealmFrame {
            realm_fence,
            source_tick: clock.local_tick,
            realm_snapshot_bytes,
        };
        // ONE shared body per chunk, refcount-cloned to every subscribing gateway (SCALE-1).
        let bytes = crate::io::bytes(
            postcard::to_allocvec(&frame).expect("closed wire enums serialize infallibly"),
        );
        for &gateway in &gateways {
            outbox.0.push((
                gateway,
                MsgClass::RealmSnapshot,
                bytes.clone(),
                Durability::Ephemeral,
            ));
        }
    }
}

/// RLM Step 2 — the run-condition: a fresh shard AUTHORS NOTHING (its celestial poses, boundary crossings,
/// AoI demands) until its clock is LIVE (D-Finding-1). `ClockSample.synced` latches `true` on the first
/// `ClockSync` and never resets, so once synced every subsequent tick runs the gated authors. Reading the
/// persistent `ClockSample` after `observe_clock_syncs` (same schedule) is order-correct.
fn has_synced(clock: Res<ClockSample>) -> bool {
    clock.synced
}

/// RLM Step 2 — the demand-driven realm-lifecycle detector (the SIBLING of [`evaluate_realm_boundaries`]):
/// per tick, this shard computes which of its DIRECT children an occupant's Area-of-Interest reaches and
/// emits a [`RealmDemand`] toward the orchestrator so those child realms spin up (and self-reports its own
/// realm's emptiness so it can be torn down). This realizes the decentralized RLM policy: EVERY realm's
/// shard is the AoI authority for ITS children — no central AoI scan (HR2/HR3 generic; one loop, no
/// match-on-realm-kind). A BRANCHLESS system shim: only the two guard `else`s (both mirroring the covered
/// detector), then delegate — ALL hysteresis/predictive/emit branching lives in the monomorphic
/// [`aoi_decide`]/[`aoi_transition`]/[`aoi_min_dist`] helpers (HR5 per-monomorphization discipline).
///
/// Gated `.run_if(has_synced)` + on `RealmAuthority` + on a non-empty region set, so it is INERT (emits
/// nothing) at walk/canonical scale where the seed AoI bands are [`AoiConfig::inert`] — byte-identical.
/// Step 2 NEVER emits a parent `TearDown` (REVISION 1 R2 supersedes the §2.2 pseudocode): a child leaving
/// AoI simply STOPS being demanded (its key drops after grace); the Step-3 reconciler closure is the sole
/// kill authority.
#[allow(clippy::too_many_arguments)]
fn evaluate_realm_aoi(
    config: Res<StubConfig>,
    clock: Res<ClockSample>,
    authority: Res<RealmAuthority>,
    regions: Res<RealmRegions>,
    dots: Res<Dots>,
    owned_transients: Res<OwnedTransients>,
    mut membership: ResMut<AoiMembership>,
    mut outbox: ResMut<OutboundBox>,
) {
    // Authority gate (verbatim `evaluate_realm_boundaries`): a shard without its realm lease demands
    // nothing — the `realm_fence` is the emitter's authority proof carried on every demand.
    let Some(realm_fence) = authority.0 else {
        return;
    };
    // Inert until regions are planted (production through canonical scale): no children to evaluate.
    if regions.is_empty() {
        return;
    }
    // L3: `aoi_decide` names the PRIMARY realm's (`config.realm`) direct children via ONE `own_coord`.
    // For node-per-realm (the base) that is complete; a co-hosting shard's co-hosted realms' children are
    // simply not evaluated here (D-44 dormant — see `register_stub_shard`). No panic: dormant co-hosting
    // tests legitimately run this with `held_realms.len() > 1` and inert bands, emitting nothing.
    aoi_decide(
        &config,
        &clock,
        &regions,
        &dots,
        &owned_transients,
        realm_fence,
        &mut membership.0,
        &mut outbox,
    );
}

/// The monomorphic AoI decision (ALL branching HERE, HR5). For each direct child: reduce the occupant set
/// to a scalar min distance (live + F7 predictive), run the per-child hysteresis + grace machine, and emit
/// the resulting verb. With ZERO occupants the shard self-reports `Empty{own_coord}` (the sole occupancy
/// authority a sealed parent cannot see — Step-3 EDGE 1). Deterministic: occupants reduce to a scalar min
/// BEFORE any emit (iteration order cannot leak into the demand set, H-2); children fold in the stable
/// seed-derived `child_placements` order.
#[allow(clippy::too_many_arguments)]
fn aoi_decide(
    config: &StubConfig,
    clock: &ClockSample,
    regions: &RealmRegions,
    dots: &Dots,
    owned: &OwnedTransients,
    realm_fence: Fence,
    membership: &mut BTreeMap<RealmPath, AoiState>,
    outbox: &mut OutboundBox,
) {
    let own_coord = &config.own_coord;
    let tick = clock.universe_tick;
    let tick_hz = 1.0 / config.tick_dt_s;
    let horizon_s = f64::from(config.boot_ticks_p99) * config.tick_dt_s; // F7 predictive horizon

    // Occupants = owned durable dots this shard SIMULATES ∪ held transients (mirror
    // `evaluate_realm_boundaries`). Each reduced to frame-local `(pos, vel)` — the SAME own frame the
    // child placements use (H-1) — so the AoI distance and the observer feed measure one geometry.
    let occupants: Vec<(DVec3, DVec3)> = dots
        .0
        .values()
        .filter(|d| d.authority.simulates())
        .map(|d| (d.pose.pos.offset(), d.pose.vel))
        .chain(
            owned
                .0
                .values()
                .filter(|t| t.status.is_held())
                .map(|t| (t.pose.pos.offset(), t.pose.vel)),
        )
        .collect();

    // Zero-occupant self-report: this CHILD shard tells the orchestrator its OWN realm holds nobody
    // (`child == own_coord`). Fence = this shard's own realm authority over its emptiness.
    if occupants.is_empty() {
        push_demand(
            outbox,
            config.orchestrator,
            own_coord.clone(),
            realm_fence,
            DemandVerb::Empty,
            tick,
        );
        return;
    }

    // The UNIFIED direct-child placements (movers authored from ephemeris, static children at `center`),
    // stable seed-derived Vec order — the SAME code-path the observer feed reads (H-1/H-2, no reorder).
    let placements = regions.child_placements(config.realm, tick_hz, tick);
    let mut live_paths = BTreeSet::<RealmPath>::new();
    for (region, pose) in &placements {
        let level = region_level(region);
        let child_coord = own_coord.child(level);
        let key = child_coord.path().clone();
        live_paths.insert(key.clone());
        let child_pos = pose.pos.offset(); // own frame (== the placements' frame)

        let min_dist_eff = aoi_min_dist(&occupants, child_pos, horizon_s);
        let state = membership.get(&key).copied().unwrap_or_default();
        let now_in = region.aoi.in_range(state.was_in, min_dist_eff);
        let (verb, next) = aoi_transition(state, now_in, region.aoi.grace_ticks());
        if let Some(v) = verb {
            debug_assert!(
                v != DemandVerb::TearDown,
                "Step 2 never emits parent TearDown — the Step-3 closure is the sole kill authority (M-1)"
            );
            push_demand(
                outbox,
                config.orchestrator,
                child_coord,
                realm_fence,
                v,
                tick,
            );
        }
        match next {
            Some(s) => {
                membership.insert(key, s);
            }
            None => {
                membership.remove(&key);
            }
        }
    }
    // Evict any child-path no longer in the roster (no leak) — the DRY primitive, keyed by RealmPath.
    retain_live(membership, &live_paths);
}

/// The smallest distance from ANY occupant to `child_pos`, taking the LESSER of the live distance and the
/// F7 predictive distance (`occ_pos + occ_vel·horizon_s`) — so a fast occupant demands spin-up BEFORE it
/// arrives (boot latency masked). A STATIC occupant has `vel == 0` ⇒ `pred == live` ⇒ no predictive term.
/// Straight-line fold (monomorphic, HR5); the scalar min is order-independent.
fn aoi_min_dist(occupants: &[(DVec3, DVec3)], child_pos: DVec3, horizon_s: f64) -> f64 {
    let mut min = f64::MAX;
    for (pos, vel) in occupants {
        let live = (child_pos - *pos).length();
        let pred = (child_pos - (*pos + *vel * horizon_s)).length();
        min = min.min(live).min(pred);
    }
    min
}

/// The per-child hysteresis + grace state machine (ALL of it monomorphic — each arm a covered region,
/// HR5). Returns `(verb_to_emit, next_state)`. Step 2 NEVER returns `TearDown`: a child leaving range
/// holds for `grace_ticks` (emitting `KeepAlive`), then drops its key and emits NOTHING — the Step-3
/// reconciler closure tears it down (M-1, REVISION 1 R2 supersedes the §2.2 pseudocode).
fn aoi_transition(
    state: AoiState,
    now_in: bool,
    grace_ticks: u32,
) -> (Option<DemandVerb>, Option<AoiState>) {
    match (state.was_in, now_in) {
        (false, true) => (
            Some(DemandVerb::SpinUp),
            Some(AoiState {
                was_in: true,
                grace_remaining: grace_ticks,
            }),
        ),
        (true, true) => (
            Some(DemandVerb::KeepAlive),
            Some(AoiState {
                was_in: true,
                grace_remaining: grace_ticks,
            }),
        ),
        (true, false) => {
            if state.grace_remaining > 0 {
                (
                    Some(DemandVerb::KeepAlive),
                    Some(AoiState {
                        was_in: true,
                        grace_remaining: state.grace_remaining - 1,
                    }),
                )
            } else {
                (None, None) // drop the key; NO TearDown
            }
        }
        (false, false) => (None, None),
    }
}

/// The `RealmId → RealmLevel` for a hosted child region — sourced un-lossily from the seed via
/// [`level_of`] (the `System(0)`/`System(1)` stand-ins recover to Universe/Galaxy; keyed kinds pass
/// through). Monomorphic (HR5: the kind match is covered once inside `level_of`). A hosted child region is
/// ALWAYS a seed-lineage realm — never an entity-backed ship — so the `None` arm is unreachable here.
fn region_level(region: &RealmRegion) -> RealmLevel {
    level_of(region.realm)
        .expect("a hosted child region is a seed-lineage realm (never an entity-backed ship)")
}

/// Emit ONE [`RealmDemand`] toward the orchestrator (the RLM emit seam). Rides `MsgClass::Saga` (Reliable)
/// so the side-effecting-flow guard passes; `push_flow` defaults `Ephemeral`, which is CORRECT — a
/// `RealmDemand` is `ReDriven` (self-heals on the next re-assertion), NOT producer-less-reliable, so the
/// durability guard passes.
///
/// The wire field is named `parent_fence` but carries the EMITTER's authority fence: the PARENT's realm
/// fence for SpinUp/KeepAlive (proving authority over the child), the CHILD-shard's own realm fence for
/// Empty (its authority over its own emptiness). Step 3 keys the Empty idempotency on `parent_fence`
/// accordingly (`IdempotencyKey::FencedKey`).
fn push_demand(
    outbox: &mut OutboundBox,
    orch: NodeId,
    child: RealmCoord,
    fence: Fence,
    verb: DemandVerb,
    tick: UniverseTick,
) {
    outbox.push_flow(
        orch,
        MsgClass::Saga,
        &InterShardFlow::RealmDemand(RealmDemand {
            child,
            parent_fence: fence,
            verb,
            universe_tick: tick,
        }),
    );
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::capability::NodeKind;
    use vd_core::frame::IdentityFrames; // the retired identity ctx — tests use it as the byte-identity oracle
    use vd_core::pose::LatticePos; // only the tests construct a LatticePos directly; prod uses .map_offset/.offset
    use vd_core::{MsgId, UniverseTick};
    use vd_wire::intershard::{
        DEMOTE_STEP, FLUSH_SOURCE_STEP, RE_SOLICIT_STEP, STUB_CROSSING_STEP,
    };

    const SHARD: NodeId = NodeId(10);
    const GATEWAY: NodeId = NodeId(20);
    const ORCH: NodeId = NodeId(30);
    const SESSION: SessionId = SessionId(0xAA);

    fn config() -> StubConfig {
        StubConfig {
            realm: RealmId::System(7),
            held_realms: StubConfig::single_realm(RealmId::System(7)),
            frame: FrameRef::SystemSpace { system_seed: 7 },
            move_speed_mps: 2.0,
            tick_dt_s: 0.05,
            time_multiplier: 1.0,
            orchestrator: ORCH,
            mint_seed: 99,
            input_log_capacity: 1024,
            realm_recheck_interval: 0,
            lease_renew_interval_ticks: 0,
            self_fence_grace_ticks: 0,
            snapshot_datagram_budget: 1100,
            boundary: vd_core::geometry::BoundaryTuning::DEFAULT,
            request_ttl_ticks: 0,
            own_coord: StubConfig::root_coord(RealmId::System(7)),
            boot_ticks_p99: 0,
            // 5f-3b: no stored spawn poses ⇒ every login births origin-at-rest (byte-identical).
            spawn_poses: BTreeMap::new(),
        }
    }

    struct Rig {
        world: World,
        schedule: Schedule,
    }

    impl Rig {
        fn new() -> Rig {
            Rig::with_config(config())
        }

        fn with_config(cfg: StubConfig) -> Rig {
            Rig::with_config_and_kind(cfg, NodeKind::StubShard)
        }

        /// As [`with_config`](Self::with_config) but with an explicit [`NodeKind`] — the RLM Step-5a
        /// capability-inertness gate parameterizes the shard's carried profile over this.
        fn with_config_and_kind(cfg: StubConfig, kind: NodeKind) -> Rig {
            let mut world = World::new();
            world.insert_resource(InboundBox::default());
            world.insert_resource(OutboundBox::default());
            world.insert_resource(NodeIdentity {
                node_id: SHARD,
                kind,
            });
            world.insert_resource(ClockSample {
                local_tick: vd_core::TickId(1),
                universe_tick: UniverseTick(100),
                epoch: vd_core::EpochId(1),
                // RLM Step 2 (M-1 rig sweep): the shared rig is SYNCED so the newly-gated authors
                // (`evaluate_realm_boundaries`/`emit_realm_frames`/`evaluate_realm_aoi`) actually run —
                // else they silently no-op and every author assertion goes false-green.
                synced: true,
            });
            let mut schedule = Schedule::default();
            register_stub_shard(&mut world, &mut schedule, cfg);
            Rig { world, schedule }
        }

        /// Set the rig's local tick (the test rig runs the schedule directly, so it
        /// must drive the clock the node shell would normally advance).
        fn set_local_tick(&mut self, tick: u64) {
            self.world.resource_mut::<ClockSample>().local_tick = vd_core::TickId(tick);
        }

        /// Run one tick with the given inbound; returns everything sent.
        fn tick(&mut self, inbound: Vec<Inbound>) -> Vec<(NodeId, MsgClass, Vec<u8>)> {
            self.world.resource_mut::<InboundBox>().0 = inbound;
            self.schedule.run(&mut self.world);
            std::mem::take(&mut self.world.resource_mut::<OutboundBox>().0)
                .into_iter()
                .map(|(to, class, bytes, _)| (to, class, bytes.to_vec()))
                .collect()
        }

        /// Like [`tick`](Self::tick) but PRESERVES each frame's [`Durability`] — for the R-6d marker
        /// conformance (verifying a producer-less one-shot's push site carries `Retained`).
        fn tick_raw(
            &mut self,
            inbound: Vec<Inbound>,
        ) -> Vec<(NodeId, MsgClass, InterShardFlow, Durability)> {
            self.world.resource_mut::<InboundBox>().0 = inbound;
            self.schedule.run(&mut self.world);
            std::mem::take(&mut self.world.resource_mut::<OutboundBox>().0)
                .into_iter()
                .filter_map(|(to, class, bytes, dur)| {
                    postcard::from_bytes::<InterShardFlow>(&bytes)
                        .ok()
                        .map(|flow| (to, class, flow, dur))
                })
                .collect()
        }

        fn grant_realm(&mut self) {
            let reply = DirectoryReply::Head {
                key: DirectoryKey::Realm(config().realm),
                record: Some(vd_wire::seams::directory::OwnerRecord {
                    authority: AuthorityRef::Shard(SHARD),
                    fence: Fence(1),
                    lease_expires: UniverseTick(1_000),
                    in_transfer: None,
                }),
            };
            let bytes = crate::io::bytes(
                postcard::to_allocvec(&InterShardFlow::DirectoryReply(reply)).expect("encode"),
            );
            let _ = self.tick(vec![Inbound::Wire {
                from: ORCH,
                class: MsgClass::Saga,
                bytes,
            }]);
        }

        /// Send the attach request only (the dot stays provisional).
        fn attach_request(
            &mut self,
            session: SessionId,
            gateway: NodeId,
        ) -> Vec<(NodeId, MsgClass, Vec<u8>)> {
            let msg = GatewayToShard::AttachSession {
                session,
                fence: Fence(1),
                account: AccountId(5),
            };
            self.tick(vec![wire_msg(gateway, MsgClass::Control, &msg)])
        }

        /// Deliver the directory's entity-grant confirmation for `session`'s dot.
        fn confirm_entity_grant(&mut self, session: SessionId) -> Vec<(NodeId, MsgClass, Vec<u8>)> {
            let entity = self.world.resource::<Dots>().0[&session].entity;
            let reply = DirectoryReply::Head {
                key: DirectoryKey::Entity(entity),
                record: Some(vd_wire::seams::directory::OwnerRecord {
                    authority: AuthorityRef::Shard(SHARD),
                    fence: Fence(1),
                    lease_expires: UniverseTick(1_000),
                    in_transfer: None,
                }),
            };
            self.tick(vec![wire_msg(
                ORCH,
                MsgClass::Saga,
                &InterShardFlow::DirectoryReply(reply),
            )])
        }

        /// The full attach flow: request, then grant confirmation.
        fn attach(&mut self) -> Vec<(NodeId, MsgClass, Vec<u8>)> {
            let _ = self.attach_request(SESSION, GATEWAY);
            self.confirm_entity_grant(SESSION)
        }
    }

    fn wire_msg<T: serde::Serialize>(from: NodeId, class: MsgClass, msg: &T) -> Inbound {
        Inbound::Wire {
            from,
            class,
            bytes: postcard::to_allocvec(msg).expect("encode").into(),
        }
    }

    // ---- D-7 (transient/debris transfer) -------------------------------------------------------

    const DEST_NODE: NodeId = NodeId(40);

    fn transient_pose() -> StampedPose {
        StampedPose::at_rest(config().frame, DVec3::new(1.0, 2.0, 3.0), UniverseTick(5))
    }

    /// Decode an `OutboundBox` into `(target, InterShardFlow)` pairs (the Saga-class egress).
    fn decode_flows(outbox: &mut OutboundBox) -> Vec<(NodeId, InterShardFlow)> {
        std::mem::take(&mut outbox.0)
            .into_iter()
            .map(|(to, _class, bytes, _)| {
                (
                    to,
                    postcard::from_bytes::<InterShardFlow>(&bytes).expect("decode"),
                )
            })
            .collect()
    }

    #[test]
    fn shard_profile_swap_is_capability_inert_for_every_profile_kind() {
        // RLM Step 5a ACCEPTANCE GATE. The shard boot swaps `NodeKind::StubShard` →
        // `NodeKind::Shard(profile_for(coord.profile_kind()))`. That swap MUST be CAPABILITY-INERT at P1–P3:
        // no system reads the carried `ShardProfile` caps to decide behavior yet, so a shard carrying ANY
        // profile emits BYTE-IDENTICAL authored frames + grants to the zero-cap StubShard shard, at the same
        // synced tick, with identical inputs. PROVEN here (not asserted) over EVERY `ProfileKind` — incl.
        // `Galaxy` (the one that carries `signal_relay`, which will uncorner P9 Signals). A future
        // `match kind` / capability gate that changes authored output would break this guard.
        use vd_core::taxonomy::ProfileKind;
        let baseline = Rig::with_config_and_kind(config(), NodeKind::StubShard).tick(vec![]);
        assert!(
            !baseline.is_empty(),
            "a synced shard authors + self-grants on an empty tick — a non-vacuous inertness baseline"
        );
        for k in ProfileKind::ALL {
            let profile =
                crate::capability::profile_for(k).expect("every ProfileKind yields a profile");
            let out = Rig::with_config_and_kind(config(), NodeKind::Shard(profile)).tick(vec![]);
            assert_eq!(
                out, baseline,
                "Shard({k:?}) authored output diverged from StubShard — the profile swap is NOT inert"
            );
        }
    }

    #[test]
    fn transient_status_is_held_excludes_arriving_and_departing() {
        // Held + the pre-emit Crossing are authoritatively held (counted); Arriving (dest mid-flight)
        // AND Departing (source released, retained) are the two UNCOUNTED tiers (D-7b).
        assert!(TransientStatus::Held { outbound: None }.is_held());
        assert!(
            TransientStatus::Held {
                outbound: Some(TransferId(1))
            }
            .is_held()
        );
        assert!(
            TransientStatus::Crossing {
                dest: DEST_NODE,
                to_realm: RealmId::System(8),
                dst_realm_fence: Fence(2),
                batch: TransferId(1),
                to_parent: None,
            }
            .is_held()
        );
        assert!(
            !TransientStatus::Arriving {
                batch: TransferId(1)
            }
            .is_held()
        );
        assert!(
            !TransientStatus::Departing {
                batch: TransferId(1)
            }
            .is_held()
        );
    }

    #[test]
    fn realm_lease_lapsed_requires_held_armed_channel_and_stale() {
        // The shared proactive self-fence predicate (HR5 branchless shim). Self-fence iff ALL hold: the
        // realm is HELD, the timer is ARMED (grace != 0), the confirmation CHANNEL exists (recheck != 0),
        // and the last confirmation is STALER than the grace. Each guard alone vetoes; the boundary
        // (`== grace`) is NOT lapsed (a strict `>`), so the holder keeps authority for the whole window.
        use crate::directory::lease_self_fence_due;
        let now = TickId(100);
        assert!(lease_self_fence_due(true, 5, 2, now, TickId(94))); // 6 > 5 ⇒ lapsed
        assert!(!lease_self_fence_due(false, 5, 2, now, TickId(94))); // not held
        assert!(!lease_self_fence_due(true, 0, 2, now, TickId(94))); // timer disarmed (pre-D-3 default)
        assert!(!lease_self_fence_due(true, 5, 0, now, TickId(94))); // no confirmation channel
        assert!(!lease_self_fence_due(true, 5, 2, now, TickId(95))); // 100-95 = 5 == grace ⇒ within, holds
    }

    #[test]
    fn a_partitioned_holder_proactively_self_fences_then_re_arms_on_re_grant() {
        // D-3 Slice 5: the realm is granted (a round-trip confirmation at tick 1), then NO further
        // realm-head reply arrives (a partition from the orchestrator). Once `local_tick - confirmed`
        // exceeds the grace, the holder hard-stops its OWN authority and drops its held transients —
        // before the orchestrator's reassign window opens — so a stale owner can never affect clients.
        // A later re-grant re-arms the deadline, so a re-granted realm never inherits the stale one.
        let mut rig = Rig::with_config(StubConfig {
            realm_recheck_interval: 2, // the round-trip confirmation channel is active
            self_fence_grace_ticks: 5, // rig-local; ttl < grace <= ttl + max is validated orch-side
            ..config()
        });
        rig.grant_realm(); // confirm at local_tick 1 ⇒ RealmConfirmedAt(1), authority Some(Fence(1))
        let debris = EntityId::pack(EntityKind::Debris, 1, 7, 1);
        rig.world.resource_mut::<OwnedTransients>().0.insert(
            debris,
            Transient {
                pose: StampedPose::at_rest(config().frame, DVec3::ZERO, UniverseTick(98)),
                anchor_fence: Fence(1),
                status: TransientStatus::Held { outbound: None },
                prev_offset: DVec3::ZERO,
            },
        );

        // Within the grace (local 6 - confirmed 1 = 5, NOT > 5): the holder KEEPS authority.
        rig.set_local_tick(6);
        let _ = rig.tick(vec![]);
        assert_eq!(rig.world.resource::<RealmAuthority>().0, Some(Fence(1)));
        assert_eq!(
            rig.world.resource::<StubStats>().realm_self_fenced_lapsed,
            0
        );
        assert!(
            rig.world
                .resource::<OwnedTransients>()
                .0
                .contains_key(&debris)
        );

        // Past the grace (local 7 - confirmed 1 = 6 > 5): SELF-FENCE — authority dropped, transient lost.
        rig.set_local_tick(7);
        let _ = rig.tick(vec![]);
        assert_eq!(
            rig.world.resource::<RealmAuthority>().0,
            None,
            "authority hard-stopped"
        );
        assert_eq!(
            rig.world.resource::<StubStats>().realm_self_fenced_lapsed,
            1
        );
        assert_eq!(
            rig.world.resource::<StubStats>().transients_dropped,
            1,
            "the held transient anchored to the lost lease is a counted loss"
        );
        assert!(rig.world.resource::<OwnedTransients>().0.is_empty());

        // A re-grant at local 7 re-confirms (RealmConfirmedAt ⇒ 7); at local 11 (11-7 = 4 <= 5) the
        // holder is STILL authoritative — proof the re-granted realm did NOT inherit the stale deadline
        // (had `confirmed` stayed 1, 11-1 = 10 > 5 would have re-fenced it immediately).
        rig.grant_realm();
        rig.set_local_tick(11);
        let _ = rig.tick(vec![]);
        assert_eq!(
            rig.world.resource::<RealmAuthority>().0,
            Some(Fence(1)),
            "re-grant re-armed the timer"
        );
        assert_eq!(
            rig.world.resource::<StubStats>().realm_self_fenced_lapsed,
            1,
            "no second self-fence"
        );
    }

    #[test]
    fn readvance_advances_held_transients_and_skips_the_uncounted_tiers() {
        // D-7b: an AUTHORITATIVELY-held debris re-advances by its closed-form ballistic motion each
        // tick (vel·dt); the uncounted Arriving/Departing tiers are SKIPPED (not rendered, caught up
        // at promote). The Rig clock is universe_tick=100; a pose stamped at tick 98 advances dt =
        // (100-98)·tick_dt_s(0.05) = 0.1s.
        let mut rig = Rig::new();
        rig.grant_realm();
        let held = EntityId::pack(EntityKind::Debris, 1, 7, 1);
        let arriving = EntityId::pack(EntityKind::Debris, 1, 7, 2);
        let departing = EntityId::pack(EntityKind::Debris, 1, 7, 3);
        let moving = StampedPose {
            vel: DVec3::new(10.0, 0.0, 0.0),
            ..StampedPose::at_rest(config().frame, DVec3::ZERO, UniverseTick(98))
        };
        {
            let mut owned = rig.world.resource_mut::<OwnedTransients>();
            owned.0.insert(
                held,
                Transient {
                    pose: moving,
                    anchor_fence: Fence(1),
                    status: TransientStatus::Held { outbound: None },
                    prev_offset: DVec3::ZERO,
                },
            );
            owned.0.insert(
                arriving,
                Transient {
                    pose: moving,
                    anchor_fence: Fence(1),
                    status: TransientStatus::Arriving {
                        batch: TransferId(1),
                    },
                    prev_offset: DVec3::ZERO,
                },
            );
            owned.0.insert(
                departing,
                Transient {
                    pose: moving,
                    anchor_fence: Fence(1),
                    status: TransientStatus::Departing {
                        batch: TransferId(1),
                    },
                    prev_offset: DVec3::ZERO,
                },
            );
        }
        let _ = rig.tick(vec![]);
        let owned = rig.world.resource::<OwnedTransients>();
        assert_eq!(
            owned.0[&held].pose.pos.offset(),
            DVec3::new(1.0, 0.0, 0.0),
            "the held debris advanced by vel·dt (10 · 0.1)"
        );
        assert_eq!(
            owned.0[&held].pose.universe_tick,
            UniverseTick(100),
            "re-stamped to now"
        );
        assert_eq!(
            owned.0[&arriving].pose.pos.offset(),
            DVec3::ZERO,
            "the uncounted Arriving tier is NOT advanced"
        );
        assert_eq!(
            owned.0[&departing].pose.pos.offset(),
            DVec3::ZERO,
            "the uncounted Departing tier is NOT advanced"
        );
    }

    #[test]
    fn emit_transient_batch_ships_one_envelope_and_marks_outbound() {
        let mut rig = Rig::new();
        rig.grant_realm(); // authority.0 = Some(Fence(1))
        let entity = EntityId::pack(EntityKind::Debris, 1, 7, 1);
        let batch = TransferId(0xB3);
        rig.world.resource_mut::<OwnedTransients>().0.insert(
            entity,
            Transient {
                pose: transient_pose(),
                anchor_fence: Fence(1),
                status: TransientStatus::Crossing {
                    dest: DEST_NODE,
                    to_realm: RealmId::System(8),
                    dst_realm_fence: Fence(2),
                    batch,
                    to_parent: None,
                },
                prev_offset: DVec3::ZERO,
            },
        );
        let sent = rig.tick(vec![]);
        // Exactly ONE TransientBatch envelope to the dest (G-TIER: one per batch), at the dst fence.
        let to_dest: Vec<_> = sent
            .iter()
            .filter(|(to, _, _)| *to == DEST_NODE)
            .map(|(_, _, bytes)| postcard::from_bytes::<InterShardFlow>(bytes).expect("decode"))
            .collect();
        assert_eq!(
            to_dest,
            vec![InterShardFlow::Transfer(TransferEnvelope {
                transfer_id: batch,
                universe_epoch: vd_core::EpochId(1),
                schema_version: TRANSFER_SCHEMA_VERSION,
                fence: Fence(2),
                step_id: TRANSIENT_BATCH_STEP,
                class: DurabilityClass::Transient,
                payload: TransitionPayload::TransientBatch {
                    from_realm: config().realm,
                    to_realm: RealmId::System(8),
                    src_realm_fence: Fence(1),
                    dst_realm_fence: Fence(2),
                    source_tick: vd_core::TickId(1),
                    items: vec![TransientItem {
                        entity,
                        // D-7b: `readvance_transients` ran BEFORE emit (Crossing is_held), re-stamping
                        // the pose to the source's current universe-tick (100); a rest pose's position
                        // is unchanged (vel ZERO), only the stamp moves. Then `emit_transient_batch`
                        // REBINDS it into the dest realm's frame (HR3 one machinery) — identity through
                        // P3, so position/velocity/orientation stay put and only the FRAME flips {7}→{8}.
                        pose: StampedPose {
                            frame: FrameRef::SystemSpace { system_seed: 8 },
                            universe_tick: UniverseTick(100),
                            ..transient_pose()
                        },
                        state: vec![],
                    }],
                },
            })],
        );
        // The emitted item is now Held{outbound} (still authoritative — adopt-before-drop).
        assert_eq!(
            rig.world.resource::<OwnedTransients>().0[&entity].status,
            TransientStatus::Held {
                outbound: Some(batch)
            }
        );
        assert_eq!(rig.world.resource::<StubStats>().transients_emitted, 1);
    }

    #[test]
    fn emit_transient_batch_ships_nothing_without_a_realm_lease() {
        let mut rig = Rig::new(); // NO grant_realm → authority.0 = None
        let entity = EntityId::pack(EntityKind::Debris, 1, 7, 1);
        let crossing = TransientStatus::Crossing {
            dest: DEST_NODE,
            to_realm: RealmId::System(8),
            dst_realm_fence: Fence(2),
            batch: TransferId(0xB3),
            to_parent: None,
        };
        rig.world.resource_mut::<OwnedTransients>().0.insert(
            entity,
            Transient {
                pose: transient_pose(),
                anchor_fence: Fence(1),
                status: crossing,
                prev_offset: DVec3::ZERO,
            },
        );
        let sent = rig.tick(vec![]);
        assert!(
            sent.iter().all(|(to, _, _)| *to != DEST_NODE),
            "a shard without its realm lease ships no transient batch"
        );
        // The Crossing item is UNCHANGED (it retries when the lease arrives).
        assert_eq!(
            rig.world.resource::<OwnedTransients>().0[&entity].status,
            crossing
        );
        assert_eq!(rig.world.resource::<StubStats>().transients_emitted, 0);
    }

    #[test]
    fn adopt_transient_batch_adopts_arriving_acks_and_dedups() {
        let batch = TransferId(0xB1);
        let entity = EntityId::pack(EntityKind::Debris, 1, 7, 0);
        let mut owned = OwnedTransients::default();
        let mut applied = AppliedSteps::default();
        let mut stats = StubStats::default();
        let mut outbox = OutboundBox::default();
        let items = vec![TransientItem {
            entity,
            pose: transient_pose(),
            state: vec![],
        }];

        // FIRST delivery: adopt as Arriving (uncounted) anchored to the dst fence + ack BatchAdopted.
        adopt_transient_batch(
            batch,
            Fence(5),
            items.clone(),
            ORCH,
            &mut owned,
            &mut applied,
            &mut stats,
            &mut outbox,
        );
        assert_eq!(owned.0[&entity].status, TransientStatus::Arriving { batch });
        assert_eq!(owned.0[&entity].anchor_fence, Fence(5));
        assert_eq!(stats.transients_adopted, 1);
        assert_eq!(stats.transients_adopt_redelivered, 0);
        assert_eq!(
            decode_flows(&mut outbox),
            vec![(
                ORCH,
                InterShardFlow::TransferAck(TransferAck::BatchAdopted {
                    transfer_id: batch,
                    step_id: TRANSIENT_BATCH_STEP,
                })
            )]
        );

        // REDELIVERY: no re-adopt, re-ack only (at-least-once — the ack may have been lost).
        adopt_transient_batch(
            batch,
            Fence(5),
            items,
            ORCH,
            &mut owned,
            &mut applied,
            &mut stats,
            &mut outbox,
        );
        assert_eq!(stats.transients_adopted, 1, "not re-adopted");
        assert_eq!(stats.transients_adopt_redelivered, 1);
        assert_eq!(decode_flows(&mut outbox).len(), 1, "re-acked exactly once");
    }

    #[test]
    fn transient_release_promote_complete_lifecycle_and_dedup() {
        // The full D-7b source/dest handler lifecycle on ONE mixed owned set (the handlers walk by
        // status; source-vs-dest is just which statuses are present in production): RELEASE flips
        // `Held{Some}→Departing` (uncounted), PROMOTE flips `Arriving→Held` (re-anchored), COMPLETE
        // retires `Departing` — each journaled/state idempotent (redelivery = counted no-op + re-ack).
        let this = TransferId(0xB2);
        let other = TransferId(0xB9);
        let held_this = EntityId::pack(EntityKind::Debris, 1, 7, 1);
        let held_other = EntityId::pack(EntityKind::Debris, 1, 7, 2);
        let held_settled = EntityId::pack(EntityKind::Debris, 1, 7, 3);
        let arriving_this = EntityId::pack(EntityKind::Debris, 1, 7, 4);
        let arriving_other = EntityId::pack(EntityKind::Debris, 1, 7, 5);
        let crossing = EntityId::pack(EntityKind::Debris, 1, 7, 6);
        let departing_other = EntityId::pack(EntityKind::Debris, 1, 7, 7);
        let mk = |status| Transient {
            pose: transient_pose(),
            anchor_fence: Fence(2),
            status,
            prev_offset: DVec3::ZERO,
        };
        let mut owned = OwnedTransients::default();
        owned.0.insert(
            held_this,
            mk(TransientStatus::Held {
                outbound: Some(this),
            }),
        );
        owned.0.insert(
            held_other,
            mk(TransientStatus::Held {
                outbound: Some(other),
            }),
        );
        owned
            .0
            .insert(held_settled, mk(TransientStatus::Held { outbound: None }));
        owned
            .0
            .insert(arriving_this, mk(TransientStatus::Arriving { batch: this }));
        owned.0.insert(
            arriving_other,
            mk(TransientStatus::Arriving { batch: other }),
        );
        owned.0.insert(
            crossing,
            mk(TransientStatus::Crossing {
                dest: DEST_NODE,
                to_realm: RealmId::System(8),
                dst_realm_fence: Fence(9),
                batch: other,
                to_parent: None,
            }),
        );
        owned.0.insert(
            departing_other,
            mk(TransientStatus::Departing { batch: other }),
        );
        let mut applied = AppliedSteps::default();
        let mut stats = StubStats::default();
        let cfg = config();
        let mut outbox = OutboundBox::default();

        // RELEASE (SOURCE): `Held{Some(this)}` → uncounted `Departing`; `Held{Some(other)}` and every
        // other status untouched; ack `DropApplied`(RELEASE_STEP) to the orchestrator.
        let rel = TransientHandoff {
            transfer: this,
            step_id: TRANSIENT_RELEASE_STEP,
            fence: Fence(5),
        };
        on_transient_release(rel, &mut owned, &mut applied, &mut stats, &cfg, &mut outbox);
        assert_eq!(
            owned.0[&held_this].status,
            TransientStatus::Departing { batch: this },
            "the source's Held item for THIS batch is released to the uncounted Departing tier"
        );
        assert_eq!(
            owned.0[&held_other].status,
            TransientStatus::Held {
                outbound: Some(other)
            },
            "a Held item for ANOTHER batch is untouched"
        );
        assert_eq!(
            owned.0[&held_settled].status,
            TransientStatus::Held { outbound: None }
        );
        assert_eq!(stats.transients_handed_off, 1);
        assert_eq!(
            decode_flows(&mut outbox),
            vec![(
                ORCH,
                InterShardFlow::TransferAck(TransferAck::DropApplied {
                    transfer_id: this,
                    step_id: TRANSIENT_RELEASE_STEP,
                })
            )]
        );
        // RELEASE REDELIVERY: re-ack, no re-flip.
        on_transient_release(rel, &mut owned, &mut applied, &mut stats, &cfg, &mut outbox);
        assert_eq!(stats.transient_release_noop, 1);
        assert_eq!(stats.transients_handed_off, 1, "not re-released");
        assert_eq!(decode_flows(&mut outbox).len(), 1, "re-acked exactly once");

        // PROMOTE (DEST): `Arriving{this}` → `Held{None}` re-anchored to the commit fence;
        // `Arriving{other}` untouched; ack `DropApplied`(DROP_STEP).
        let promote = TransientHandoff {
            transfer: this,
            step_id: TRANSIENT_DROP_STEP,
            fence: Fence(5),
        };
        on_transient_promote(
            promote,
            &mut owned,
            &mut applied,
            &mut stats,
            &cfg,
            &mut outbox,
        );
        assert_eq!(
            owned.0[&arriving_this].status,
            TransientStatus::Held { outbound: None },
            "THIS batch's Arriving is promoted to authoritative Held"
        );
        assert_eq!(
            owned.0[&arriving_this].anchor_fence,
            Fence(5),
            "the promoted item re-anchors to the batch's commit fence"
        );
        assert_eq!(
            owned.0[&arriving_other].status,
            TransientStatus::Arriving { batch: other },
            "another batch's Arriving is untouched"
        );
        assert!(
            owned.0.contains_key(&crossing),
            "a pending Crossing is untouched"
        );
        assert_eq!(stats.transients_promoted, 1);
        assert_eq!(
            decode_flows(&mut outbox),
            vec![(
                ORCH,
                InterShardFlow::TransferAck(TransferAck::DropApplied {
                    transfer_id: this,
                    step_id: TRANSIENT_DROP_STEP,
                })
            )]
        );
        // PROMOTE REDELIVERY: re-ack, no re-flip.
        on_transient_promote(
            promote,
            &mut owned,
            &mut applied,
            &mut stats,
            &cfg,
            &mut outbox,
        );
        assert_eq!(stats.transient_drop_noop, 1);
        assert_eq!(stats.transients_promoted, 1, "not re-promoted");
        assert_eq!(decode_flows(&mut outbox).len(), 1, "re-acked exactly once");

        // COMPLETE (SOURCE): retire THIS batch's `Departing` copy (held_this); a `Departing` for
        // ANOTHER batch (departing_other) is untouched; D-7d: ALWAYS ack DropApplied(COMPLETE_STEP) —
        // the `SourceRetired` signal that drives the saga's BatchHandoff tail to Done.
        let rc = TransientHandoff {
            transfer: this,
            step_id: TRANSIENT_RELEASE_STEP,
            fence: Fence(5),
        };
        on_release_complete(rc, &mut owned, &mut stats, &cfg, &mut outbox);
        assert!(
            !owned.0.contains_key(&held_this),
            "the retained Departing copy for THIS batch is retired"
        );
        assert_eq!(
            owned.0[&departing_other].status,
            TransientStatus::Departing { batch: other },
            "a Departing copy for ANOTHER batch is untouched"
        );
        assert_eq!(
            decode_flows(&mut outbox),
            vec![(
                ORCH,
                InterShardFlow::TransferAck(TransferAck::DropApplied {
                    transfer_id: this,
                    step_id: TRANSIENT_COMPLETE_STEP,
                })
            )],
            "ReleaseComplete acks the retire-complete so the saga tail reaches Done (SourceRetired)"
        );
        // COMPLETE REDELIVERY: no Departing for THIS batch → counted no-op, but STILL acks
        // (at-least-once — the orchestrator's tombstoned saga absorbs the duplicate).
        on_release_complete(rc, &mut owned, &mut stats, &cfg, &mut outbox);
        assert_eq!(
            stats.transient_release_noop, 2,
            "the redelivered complete is a counted no-op"
        );
        assert_eq!(
            decode_flows(&mut outbox).len(),
            1,
            "the redelivery still acks (at-least-once)"
        );
    }

    #[test]
    fn self_fence_drops_held_transients_as_a_counted_loss() {
        // A realm takeover (the lease now held by someone else) self-fences the shard AND drops its
        // transients (anchored to the now-lost lease) as a counted LOSS — durable dots are retained.
        let mut rig = Rig::new();
        rig.grant_realm();
        let entity = EntityId::pack(EntityKind::Debris, 1, 7, 1);
        rig.world.resource_mut::<OwnedTransients>().0.insert(
            entity,
            Transient {
                pose: transient_pose(),
                anchor_fence: Fence(1),
                status: TransientStatus::Held { outbound: None },
                prev_offset: DVec3::ZERO,
            },
        );
        let takeover = DirectoryReply::Head {
            key: DirectoryKey::Realm(config().realm),
            record: Some(vd_wire::seams::directory::OwnerRecord {
                authority: AuthorityRef::Shard(NodeId(99)),
                fence: Fence(2),
                lease_expires: UniverseTick(1_000),
                in_transfer: None,
            }),
        };
        let _ = rig.tick(vec![wire_msg(
            ORCH,
            MsgClass::Saga,
            &InterShardFlow::DirectoryReply(takeover),
        )]);
        assert!(
            rig.world.resource::<OwnedTransients>().0.is_empty(),
            "transients dropped on self-fence"
        );
        assert_eq!(rig.world.resource::<StubStats>().transients_dropped, 1);
        assert!(
            rig.world.resource::<RealmAuthority>().0.is_none(),
            "the shard self-fenced its realm"
        );
    }

    #[test]
    fn self_fence_drops_held_transients_on_realm_revoke() {
        // The revoked arm (the realm record is GONE) also self-fences + drops transients — the second
        // `self_fence_drop_transients` call site (a revoke vs a takeover).
        let mut rig = Rig::new();
        rig.grant_realm();
        let entity = EntityId::pack(EntityKind::Debris, 1, 7, 2);
        rig.world.resource_mut::<OwnedTransients>().0.insert(
            entity,
            Transient {
                pose: transient_pose(),
                anchor_fence: Fence(1),
                status: TransientStatus::Held { outbound: None },
                prev_offset: DVec3::ZERO,
            },
        );
        let revoked = DirectoryReply::Head {
            key: DirectoryKey::Realm(config().realm),
            record: None,
        };
        let _ = rig.tick(vec![wire_msg(
            ORCH,
            MsgClass::Saga,
            &InterShardFlow::DirectoryReply(revoked),
        )]);
        assert!(rig.world.resource::<OwnedTransients>().0.is_empty());
        assert_eq!(rig.world.resource::<StubStats>().transients_dropped, 1);
        assert!(rig.world.resource::<RealmAuthority>().0.is_none());
    }

    #[test]
    fn transient_status_is_in_handover_excludes_only_settled_held() {
        // D-7b.3: every tier EXCEPT a settled `Held{outbound: None}` is a handover loss if dropped.
        assert!(
            !TransientStatus::Held { outbound: None }.is_in_handover(),
            "a settled Held is a resident eviction, not a handover loss"
        );
        assert!(
            TransientStatus::Held {
                outbound: Some(TransferId(1))
            }
            .is_in_handover()
        );
        assert!(
            TransientStatus::Crossing {
                dest: DEST_NODE,
                to_realm: RealmId::System(8),
                dst_realm_fence: Fence(2),
                batch: TransferId(1),
                to_parent: None,
            }
            .is_in_handover()
        );
        assert!(
            TransientStatus::Arriving {
                batch: TransferId(1)
            }
            .is_in_handover()
        );
        assert!(
            TransientStatus::Departing {
                batch: TransferId(1)
            }
            .is_in_handover()
        );
    }

    #[test]
    fn self_fence_buckets_handover_loss_per_kind_and_excludes_resident_and_corrupt() {
        // D-7b.3: a realm self-fence buckets ONLY handover-status items into the per-kind
        // handover-loss counter (the budget gate's input), keyed by kind; a settled `Held{None}` is a
        // resident eviction (NOT bucketed); a corrupt kind tag is counted GROSS but NOT bucketed (the
        // `from_tag` Err arm). The gross `transients_dropped` still counts EVERY tier.
        let t = TransferId(0xC0);
        let mk = |status| Transient {
            pose: transient_pose(),
            anchor_fence: Fence(2),
            status,
            prev_offset: DVec3::ZERO,
        };
        let mut owned = OwnedTransients::default();
        // Four Debris in handover (one per tier).
        owned.0.insert(
            EntityId::pack(EntityKind::Debris, 1, 1, 0),
            mk(TransientStatus::Held { outbound: Some(t) }),
        );
        owned.0.insert(
            EntityId::pack(EntityKind::Debris, 1, 2, 0),
            mk(TransientStatus::Crossing {
                dest: DEST_NODE,
                to_realm: RealmId::System(8),
                dst_realm_fence: Fence(2),
                batch: t,
                to_parent: None,
            }),
        );
        owned.0.insert(
            EntityId::pack(EntityKind::Debris, 1, 3, 0),
            mk(TransientStatus::Arriving { batch: t }),
        );
        owned.0.insert(
            EntityId::pack(EntityKind::Debris, 1, 4, 0),
            mk(TransientStatus::Departing { batch: t }),
        );
        // A Projectile in handover (proves per-kind disentangling).
        owned.0.insert(
            EntityId::pack(EntityKind::Projectile, 1, 5, 0),
            mk(TransientStatus::Held { outbound: Some(t) }),
        );
        // A SETTLED Debris (resident eviction — NOT a handover loss).
        owned.0.insert(
            EntityId::pack(EntityKind::Debris, 1, 6, 0),
            mk(TransientStatus::Held { outbound: None }),
        );
        // A corrupt kind tag (99) in handover — counted gross, NOT bucketed (the from_tag Err arm).
        owned.0.insert(
            EntityId(99u128 << 120),
            mk(TransientStatus::Departing { batch: t }),
        );
        let mut stats = StubStats::default();

        self_fence_drop_transients(&mut owned, &mut stats);
        assert!(
            owned.0.is_empty(),
            "every transient is dropped on self-fence"
        );
        assert_eq!(stats.transients_dropped, 7, "gross counts EVERY tier");
        assert_eq!(
            stats.transients_lost_in_handover.get(&EntityKind::Debris),
            Some(&4),
            "4 Debris in handover bucketed (the settled one + the corrupt one excluded)"
        );
        assert_eq!(
            stats
                .transients_lost_in_handover
                .get(&EntityKind::Projectile),
            Some(&1),
            "the Projectile is bucketed under its OWN kind (per-kind disentangling)"
        );
    }

    #[test]
    fn on_transient_abandon_drops_the_batch_as_accounted_loss_and_is_idempotent() {
        // D-7d dead-DEST resolution: abandon drops THIS batch's retained items — both `Departing{this}`
        // (already released) AND `Held{Some(this)}` (the dest died before release) — bucketing each into
        // the per-kind loss budget + counting departure_cancelled. A `Held{None}` resident, a
        // `Departing{other}` batch, and a `Held{Some(other)}` batch are UNTOUCHED. A corrupt kind tag is
        // removed but NOT bucketed (the from_tag Err arm). A redelivery is a journaled no-op.
        let this = TransferId(0xD7D);
        let other = TransferId(0xBEEF);
        let mk = |status| Transient {
            pose: transient_pose(),
            anchor_fence: Fence(3),
            status,
            prev_offset: DVec3::ZERO,
        };
        let mut owned = OwnedTransients::default();
        let dep_this = EntityId::pack(EntityKind::Debris, 1, 1, 0);
        let held_this = EntityId::pack(EntityKind::Debris, 1, 2, 0);
        let resident = EntityId::pack(EntityKind::Debris, 1, 3, 0);
        let dep_other = EntityId::pack(EntityKind::Debris, 1, 4, 0);
        let held_other = EntityId::pack(EntityKind::Debris, 1, 5, 0);
        let corrupt = EntityId(99u128 << 120); // tag 99 → from_tag Err (removed, not bucketed)
        owned
            .0
            .insert(dep_this, mk(TransientStatus::Departing { batch: this }));
        owned.0.insert(
            held_this,
            mk(TransientStatus::Held {
                outbound: Some(this),
            }),
        );
        owned
            .0
            .insert(resident, mk(TransientStatus::Held { outbound: None }));
        owned
            .0
            .insert(dep_other, mk(TransientStatus::Departing { batch: other }));
        owned.0.insert(
            held_other,
            mk(TransientStatus::Held {
                outbound: Some(other),
            }),
        );
        owned
            .0
            .insert(corrupt, mk(TransientStatus::Departing { batch: this }));
        let mut applied = AppliedSteps::default();
        let mut stats = StubStats::default();
        let abandon = TransientHandoff {
            transfer: this,
            step_id: TRANSIENT_ABANDON_STEP,
            fence: Fence(3),
        };

        on_transient_abandon(abandon, &mut owned, &mut applied, &mut stats);
        // THIS batch's Departing + Held{Some} + the corrupt one are dropped.
        assert!(!owned.0.contains_key(&dep_this));
        assert!(!owned.0.contains_key(&held_this));
        assert!(!owned.0.contains_key(&corrupt));
        // The resident + the OTHER batch's copies survive (not this batch).
        assert!(owned.0.contains_key(&resident));
        assert!(owned.0.contains_key(&dep_other));
        assert!(owned.0.contains_key(&held_other));
        assert_eq!(
            stats.transients_departure_cancelled, 3,
            "3 items abandoned (2 Debris + 1 corrupt)"
        );
        assert_eq!(
            stats.transients_lost_in_handover.get(&EntityKind::Debris),
            Some(&2),
            "only the 2 Debris are bucketed (the corrupt kind is removed but not attributable)"
        );

        // REDELIVERY: the journal short-circuits — no double-count, the survivors are untouched.
        on_transient_abandon(abandon, &mut owned, &mut applied, &mut stats);
        assert_eq!(
            stats.transients_departure_cancelled, 3,
            "the redelivery did not re-count"
        );
        assert_eq!(
            stats.transient_release_noop, 1,
            "the redelivery is a counted journal no-op"
        );
    }

    #[test]
    fn on_transient_discard_removes_arriving_poisons_adopt_and_is_idempotent() {
        // R-6d3c NEVER-restart resolution (DEST role): the source died in AwaitAdopt pre-adopt, so the
        // discard REMOVES this batch's `Arriving{this}` items as an accounted loss — a decodable Debris
        // is bucketed, a corrupt kind tag is removed but NOT bucketed (the from_tag Err arm). An
        // `Arriving{other}` (batch mismatch), a `Held{None}` resident, and a `Departing{this}` (non-
        // Arriving) are UNTOUCHED — the false arms. A redelivery is a journaled no-op (loss counts once).
        let this = TransferId(0x6D3C);
        let other = TransferId(0xBEEF);
        let mk = |status| Transient {
            pose: transient_pose(),
            anchor_fence: Fence(4),
            status,
            prev_offset: DVec3::ZERO,
        };
        let mut owned = OwnedTransients::default();
        let arr_this = EntityId::pack(EntityKind::Debris, 1, 1, 0);
        let corrupt = EntityId(99u128 << 120); // tag 99 → from_tag Err (removed, not bucketed)
        let arr_other = EntityId::pack(EntityKind::Debris, 1, 2, 0);
        let resident = EntityId::pack(EntityKind::Debris, 1, 3, 0);
        let dep_this = EntityId::pack(EntityKind::Debris, 1, 4, 0);
        owned
            .0
            .insert(arr_this, mk(TransientStatus::Arriving { batch: this }));
        owned
            .0
            .insert(corrupt, mk(TransientStatus::Arriving { batch: this }));
        owned
            .0
            .insert(arr_other, mk(TransientStatus::Arriving { batch: other }));
        owned
            .0
            .insert(resident, mk(TransientStatus::Held { outbound: None }));
        owned
            .0
            .insert(dep_this, mk(TransientStatus::Departing { batch: this }));
        let mut applied = AppliedSteps::default();
        let mut stats = StubStats::default();
        let discard = TransientHandoff {
            transfer: this,
            step_id: TRANSIENT_DISCARD_STEP,
            fence: Fence(4),
        };

        on_transient_discard(discard, &mut owned, &mut applied, &mut stats);
        // THIS batch's Arriving items (the decodable + the corrupt) are removed.
        assert!(!owned.0.contains_key(&arr_this));
        assert!(!owned.0.contains_key(&corrupt));
        // The OTHER batch's Arriving, the settled resident, and a Departing item survive (false arms).
        assert!(owned.0.contains_key(&arr_other));
        assert!(owned.0.contains_key(&resident));
        assert!(owned.0.contains_key(&dep_this));
        assert_eq!(
            stats.transients_discarded_source_crash, 2,
            "2 Arriving items discarded (1 Debris + 1 corrupt)"
        );
        assert_eq!(
            stats.transients_lost_in_handover.get(&EntityKind::Debris),
            Some(&1),
            "only the decodable Debris is bucketed (the corrupt kind is removed but not attributable)"
        );

        // REDELIVERY: the journal short-circuits — no double-count, the survivors are untouched.
        on_transient_discard(discard, &mut owned, &mut applied, &mut stats);
        assert_eq!(
            stats.transients_discarded_source_crash, 2,
            "the redelivery did not re-count"
        );
        assert_eq!(
            stats.transient_release_noop, 1,
            "the redelivery is a counted journal no-op"
        );
        assert_eq!(
            owned.0.len(),
            3,
            "the survivors are untouched by the redelivery"
        );
    }

    #[test]
    fn discard_before_adopt_poisons_so_a_late_replay_never_orphans() {
        // THE target interleave (Defect A closed): the discard fires FIRST on an EMPTY owned set (the
        // dest never received the batch — the source died pre-adopt) → it removes nothing but POISONS
        // `(transfer, TRANSIENT_BATCH_STEP)`. A LATE outbox replay of the batch then adopts as
        // `AlreadyApplied` — inserting NOTHING — so no `Arriving` orphan is ever stranded.
        let this = TransferId(0x6D3C);
        let entity = EntityId::pack(EntityKind::Debris, 1, 7, 0);
        let mut owned = OwnedTransients::default();
        let mut applied = AppliedSteps::default();
        let mut stats = StubStats::default();
        let mut outbox = OutboundBox::default();

        let discard = TransientHandoff {
            transfer: this,
            step_id: TRANSIENT_DISCARD_STEP,
            fence: Fence(4),
        };
        on_transient_discard(discard, &mut owned, &mut applied, &mut stats);
        assert_eq!(
            stats.transients_discarded_source_crash, 0,
            "nothing to remove on an empty dest — the discard only poisons the adopt"
        );

        // The LATE batch replay: the adopt hits its `AlreadyApplied` arm (poisoned) — no insert.
        adopt_transient_batch(
            this,
            Fence(5),
            vec![TransientItem {
                entity,
                pose: transient_pose(),
                state: vec![],
            }],
            ORCH,
            &mut owned,
            &mut applied,
            &mut stats,
            &mut outbox,
        );
        assert!(
            owned.0.is_empty(),
            "the poisoned adopt inserts nothing — no Arriving orphan"
        );
        assert_eq!(
            stats.transients_adopt_redelivered, 1,
            "the adopt short-circuited on the poisoned step"
        );
        assert_eq!(stats.transients_adopted, 0, "no item was ever adopted");
    }

    #[test]
    fn transient_discard_flows_through_the_inbound_dispatch() {
        // Covers the `on_directory_reply` DISPATCH arm for `TransientDiscard` (the direct-call tests
        // above cover the handler itself): a DEST holding an `Arriving` copy receives the discard, drops
        // it as an accounted loss, and emits NOTHING (ack-FREE — the resolving saga is terminal).
        let mut rig = Rig::new();
        rig.grant_realm(); // authority.0 = Some(Fence(1))
        let debris = EntityId::pack(EntityKind::Debris, 1, 7, 0);
        let batch = TransferId(0xB7);
        rig.world.resource_mut::<OwnedTransients>().0.insert(
            debris,
            Transient {
                pose: transient_pose(),
                anchor_fence: Fence(1),
                status: TransientStatus::Arriving { batch },
                prev_offset: DVec3::ZERO,
            },
        );
        let discard = TransientHandoff {
            transfer: batch,
            step_id: TRANSIENT_DISCARD_STEP,
            fence: Fence(1),
        };
        let sent = rig.tick(vec![wire_msg(
            ORCH,
            MsgClass::Saga,
            &InterShardFlow::TransientDiscard(discard),
        )]);
        assert!(
            !rig.world
                .resource::<OwnedTransients>()
                .0
                .contains_key(&debris),
            "the Arriving copy was discarded"
        );
        assert_eq!(
            rig.world
                .resource::<StubStats>()
                .transients_discarded_source_crash,
            1
        );
        assert_eq!(sent, vec![], "the discard is ack-free (terminal saga)");
    }

    #[test]
    fn redrive_pending_adoptions_re_emits_one_ack_per_distinct_arriving_batch() {
        // CA-1 S3/S4 LIVENESS: `redrive_pending_adoptions` re-emits `BatchAdopted` every tick for each
        // DISTINCT `Arriving` batch (dedup — a batch of N items yields ONE ack, the MMO-scale discipline),
        // and SKIPS settled `Held` items. Seed the tiers DIRECTLY + tick with an empty inbox so the ONLY
        // egress is the re-drive (adopt itself is not exercised here).
        let mut rig = Rig::new();
        rig.grant_realm();
        let batch = TransferId(0xB7);
        let other = TransferId(0xB8); // 0xB7 < 0xB8 → deterministic BTreeSet emit order
        let arriving = |b| Transient {
            pose: transient_pose(),
            anchor_fence: Fence(1),
            status: TransientStatus::Arriving { batch: b },
            prev_offset: DVec3::ZERO,
        };
        {
            let mut owned = rig.world.resource_mut::<OwnedTransients>();
            owned
                .0
                .insert(EntityId::pack(EntityKind::Debris, 1, 7, 0), arriving(batch));
            // Second item, SAME batch → still ONE ack for `batch` (distinct-batch dedup).
            owned
                .0
                .insert(EntityId::pack(EntityKind::Debris, 1, 7, 1), arriving(batch));
            owned
                .0
                .insert(EntityId::pack(EntityKind::Debris, 1, 7, 2), arriving(other));
            // A SETTLED resident (Held) is NOT re-driven — the `if let Arriving` false arm.
            owned.0.insert(
                EntityId::pack(EntityKind::Debris, 1, 7, 3),
                Transient {
                    pose: transient_pose(),
                    anchor_fence: Fence(1),
                    status: TransientStatus::Held { outbound: None },
                    prev_offset: DVec3::ZERO,
                },
            );
        }
        let sent = rig.tick(vec![]);
        let ack = |t| {
            postcard::to_allocvec(&InterShardFlow::TransferAck(TransferAck::BatchAdopted {
                transfer_id: t,
                step_id: TRANSIENT_BATCH_STEP,
            }))
            .expect("encode")
        };
        // Exactly ONE ack per DISTINCT batch, in ascending BTreeSet order.
        assert_eq!(
            sent,
            vec![
                (ORCH, MsgClass::Saga, ack(batch)),
                (ORCH, MsgClass::Saga, ack(other)),
            ]
        );
        assert_eq!(rig.world.resource::<StubStats>().batch_adopts_redriven, 2);
    }

    #[test]
    fn redrive_pending_adoptions_is_a_noop_when_nothing_is_arriving() {
        // The empty-`Arriving` path (the `for batch in batches` empty loop + `if let` all-false): a shard
        // holding only a settled `Held` transient re-drives nothing and emits no egress.
        let mut rig = Rig::new();
        rig.grant_realm();
        rig.world.resource_mut::<OwnedTransients>().0.insert(
            EntityId::pack(EntityKind::Debris, 1, 7, 0),
            Transient {
                pose: transient_pose(),
                anchor_fence: Fence(1),
                status: TransientStatus::Held { outbound: None },
                prev_offset: DVec3::ZERO,
            },
        );
        let sent = rig.tick(vec![]);
        assert_eq!(sent, vec![], "nothing Arriving → no re-drive");
        assert_eq!(rig.world.resource::<StubStats>().batch_adopts_redriven, 0);
    }

    #[test]
    fn re_solicit_batch_is_a_counted_noop_at_the_source() {
        // CA-1 S3: the orchestrator's AwaitAdopt liveness PROBE arriving at a (live) SOURCE is a counted
        // no-op — no state change, no egress (the probe's signal is its SEND outcome at the orchestrator,
        // not this handler). No `Arriving` items, so the re-drive adds nothing to `sent`.
        let mut rig = Rig::new();
        rig.grant_realm();
        let probe = TransientHandoff {
            transfer: TransferId(0xB7),
            step_id: RE_SOLICIT_STEP,
            fence: Fence(1),
        };
        let sent = rig.tick(vec![wire_msg(
            ORCH,
            MsgClass::Saga,
            &InterShardFlow::ReSolicitBatch(probe),
        )]);
        assert_eq!(rig.world.resource::<StubStats>().re_solicits_received, 1);
        assert_eq!(
            sent,
            vec![],
            "the probe is a pure no-op — no reply, no state change"
        );
    }

    #[test]
    fn transient_batch_and_drop_flow_through_the_inbound_dispatch() {
        // Covers the inbound DISPATCH into the transient handlers (the direct-call tests above cover
        // the handlers themselves): on_directory_reply → on_transfer_envelope's `TransientBatch` arm
        // (adopt + ack) AND on_directory_reply's `TransientDrop` arm (promote). The DEST adopts a
        // batch, acks BatchAdopted, then promotes Arriving→Held on the drop.
        let mut rig = Rig::new();
        rig.grant_realm(); // authority.0 = Some(Fence(1))
        let debris = EntityId::pack(EntityKind::Debris, 1, 7, 0);
        let batch = TransferId(0xB7);
        let env = TransferEnvelope {
            transfer_id: batch,
            universe_epoch: vd_core::EpochId(1),
            schema_version: TRANSFER_SCHEMA_VERSION,
            fence: Fence(1),
            step_id: TRANSIENT_BATCH_STEP,
            class: DurabilityClass::Transient,
            payload: TransitionPayload::TransientBatch {
                from_realm: RealmId::System(8),
                to_realm: config().realm,
                src_realm_fence: Fence(1),
                dst_realm_fence: Fence(1),
                source_tick: vd_core::TickId(1),
                items: vec![TransientItem {
                    entity: debris,
                    pose: transient_pose(),
                    state: vec![],
                }],
            },
        };
        let sent = rig.tick(vec![wire_msg(
            NodeId(50),
            MsgClass::Saga,
            &InterShardFlow::Transfer(env),
        )]);
        // Adopted as the uncounted Arriving tier, anchored to the envelope's dst realm fence.
        assert_eq!(
            rig.world.resource::<OwnedTransients>().0[&debris].status,
            TransientStatus::Arriving { batch }
        );
        assert_eq!(
            rig.world.resource::<OwnedTransients>().0[&debris].anchor_fence,
            Fence(1)
        );
        // The egress is the BatchAdopted ack TWICE (exact-vec equality — no filter/any closure with an
        // uncoverable short-circuit arm, the HR5 test discipline): the adopt handler acks it once (in
        // `process_inbound`), then CA-1 S3/S4's `redrive_pending_adoptions` re-emits it the SAME tick (the
        // item is now `Arriving`) — the liveness re-drive. Both are byte-identical; the orchestrator absorbs
        // the duplicate (idempotent). Order is adopt-ack THEN re-drive (chain order).
        let expected_ack =
            postcard::to_allocvec(&InterShardFlow::TransferAck(TransferAck::BatchAdopted {
                transfer_id: batch,
                step_id: TRANSIENT_BATCH_STEP,
            }))
            .expect("encode");
        assert_eq!(
            sent,
            vec![
                (ORCH, MsgClass::Saga, expected_ack.clone()),
                (ORCH, MsgClass::Saga, expected_ack),
            ]
        );

        // The TransientDrop dispatch arm PROMOTES the Arriving item → Held + acks DropApplied(DROP).
        let promote = TransientHandoff {
            transfer: batch,
            step_id: TRANSIENT_DROP_STEP,
            fence: Fence(1),
        };
        let sent = rig.tick(vec![wire_msg(
            ORCH,
            MsgClass::Saga,
            &InterShardFlow::TransientDrop(promote),
        )]);
        assert_eq!(
            rig.world.resource::<OwnedTransients>().0[&debris].status,
            TransientStatus::Held { outbound: None }
        );
        assert_eq!(rig.world.resource::<StubStats>().transients_promoted, 1);
        let promote_ack =
            postcard::to_allocvec(&InterShardFlow::TransferAck(TransferAck::DropApplied {
                transfer_id: batch,
                step_id: TRANSIENT_DROP_STEP,
            }))
            .expect("encode");
        assert_eq!(sent, vec![(ORCH, MsgClass::Saga, promote_ack)]);

        // The TransientRelease + ReleaseComplete dispatch arms (SOURCE role): seed a Held source item
        // for a 2nd batch, release it → Departing + DropApplied(RELEASE) ack, then complete → retired.
        let src_batch = TransferId(0xB8);
        let src_item = EntityId::pack(EntityKind::Debris, 1, 7, 9);
        rig.world.resource_mut::<OwnedTransients>().0.insert(
            src_item,
            Transient {
                pose: transient_pose(),
                anchor_fence: Fence(1),
                status: TransientStatus::Held {
                    outbound: Some(src_batch),
                },
                prev_offset: DVec3::ZERO,
            },
        );
        let rel = TransientHandoff {
            transfer: src_batch,
            step_id: TRANSIENT_RELEASE_STEP,
            fence: Fence(1),
        };
        let sent = rig.tick(vec![wire_msg(
            ORCH,
            MsgClass::Saga,
            &InterShardFlow::TransientRelease(rel),
        )]);
        assert_eq!(
            rig.world.resource::<OwnedTransients>().0[&src_item].status,
            TransientStatus::Departing { batch: src_batch }
        );
        let release_ack =
            postcard::to_allocvec(&InterShardFlow::TransferAck(TransferAck::DropApplied {
                transfer_id: src_batch,
                step_id: TRANSIENT_RELEASE_STEP,
            }))
            .expect("encode");
        assert_eq!(sent, vec![(ORCH, MsgClass::Saga, release_ack)]);

        let rc = TransientHandoff {
            transfer: src_batch,
            step_id: TRANSIENT_RELEASE_STEP,
            fence: Fence(1),
        };
        let sent = rig.tick(vec![wire_msg(
            ORCH,
            MsgClass::Saga,
            &InterShardFlow::ReleaseComplete(rc),
        )]);
        assert!(
            !rig.world
                .resource::<OwnedTransients>()
                .0
                .contains_key(&src_item),
            "ReleaseComplete retired the Departing copy"
        );
        let complete_ack =
            postcard::to_allocvec(&InterShardFlow::TransferAck(TransferAck::DropApplied {
                transfer_id: src_batch,
                step_id: TRANSIENT_COMPLETE_STEP,
            }))
            .expect("encode");
        assert_eq!(
            sent,
            vec![(ORCH, MsgClass::Saga, complete_ack)],
            "D-7d: ReleaseComplete acks the retire-complete (SourceRetired drives the saga tail to Done)"
        );

        // The D-7d TransientAbandon dispatch arm (SOURCE role): seed a fresh Departing item for a 3rd
        // batch (the dead-dest case), abandon it → dropped + bucketed as an accounted loss, NO ack (the
        // resolving saga is already terminal).
        let abandon_batch = TransferId(0xB9);
        let abandon_item = EntityId::pack(EntityKind::Debris, 1, 8, 9);
        rig.world.resource_mut::<OwnedTransients>().0.insert(
            abandon_item,
            Transient {
                pose: transient_pose(),
                anchor_fence: Fence(1),
                status: TransientStatus::Departing {
                    batch: abandon_batch,
                },
                prev_offset: DVec3::ZERO,
            },
        );
        let sent = rig.tick(vec![wire_msg(
            ORCH,
            MsgClass::Saga,
            &InterShardFlow::TransientAbandon(TransientHandoff {
                transfer: abandon_batch,
                step_id: TRANSIENT_ABANDON_STEP,
                fence: Fence(1),
            }),
        )]);
        assert!(
            !rig.world
                .resource::<OwnedTransients>()
                .0
                .contains_key(&abandon_item),
            "TransientAbandon dropped the retained Departing copy"
        );
        assert_eq!(
            rig.world
                .resource::<StubStats>()
                .transients_lost_in_handover
                .get(&EntityKind::Debris),
            Some(&1),
            "the abandoned Debris is bucketed as an accounted loss"
        );
        assert!(sent.is_empty(), "TransientAbandon is terminal — no ack");
    }

    fn input_msg(seq: u64, fence: Fence, movement: [f32; 3], look: [f32; 2]) -> Inbound {
        // Every existing caller keeps the inert action_bits:0 default via this delegation (DRY — ONE
        // InputDatagram construction site); only the action_bits tripwire below varies the bits.
        input_msg_bits(seq, fence, movement, look, 0)
    }

    /// Like [`input_msg`] but with an EXPLICIT `action_bits` — the D-41/D-39.1 tripwire is the sole
    /// caller that needs to vary the one field `input_msg` otherwise hardcodes to 0.
    fn input_msg_bits(
        seq: u64,
        fence: Fence,
        movement: [f32; 3],
        look: [f32; 2],
        action_bits: u32,
    ) -> Inbound {
        let input = InputDatagram {
            seq,
            is_cut_marker: false,
            client_tick: vd_core::TickId(2),
            movement,
            look,
            action_bits,
        };
        let msg = GatewayToShard::SessionInput {
            session: SESSION,
            fence,
            input_bytes: postcard::to_allocvec(&input).expect("encode input"),
        };
        wire_msg(GATEWAY, MsgClass::Input, &msg)
    }

    fn decode_frames(sent: &[(NodeId, MsgClass, Vec<u8>)]) -> Vec<SnapshotDatagram> {
        sent.iter()
            .filter(|(_, class, _)| *class == MsgClass::Snapshot)
            .map(|(_, _, bytes)| {
                let frame: ShardToGateway = postcard::from_bytes(bytes).expect("frame");
                let snapshot_bytes = frame
                    .into_snapshot_bytes()
                    .expect("snapshot class carries Frame");
                postcard::from_bytes::<SnapshotDatagram>(&snapshot_bytes).expect("snapshot")
            })
            .collect()
    }

    /// The realm-observer twin of [`decode_frames`] (FA-2c): every `RealmSnapshotDatagram` on the
    /// `RealmSnapshot` class, decoded from its `ShardToGateway::RealmFrame` carrier.
    fn decode_realm_frames(sent: &[(NodeId, MsgClass, Vec<u8>)]) -> Vec<RealmSnapshotDatagram> {
        sent.iter()
            .filter(|(_, class, _)| *class == MsgClass::RealmSnapshot)
            .map(|(_, _, bytes)| {
                let frame: ShardToGateway = postcard::from_bytes(bytes).expect("frame");
                let realm_bytes = frame
                    .into_realm_snapshot_bytes()
                    .expect("realm class carries RealmFrame");
                postcard::from_bytes::<RealmSnapshotDatagram>(&realm_bytes).expect("realm snapshot")
            })
            .collect()
    }

    #[test]
    fn boot_requests_the_realm_lease_until_granted_then_stops() {
        let mut rig = Rig::new();
        let expected_request =
            postcard::to_allocvec(&InterShardFlow::Directory(DirectoryOp::LeaseGrant {
                key: DirectoryKey::Realm(config().realm),
                owner: AuthorityRef::Shard(SHARD),
                fence: Fence(1),
            }))
            .expect("encode");
        // Two unanswered ticks: two identical idempotent requests.
        for _ in 0..2 {
            let sent = rig.tick(vec![]);
            assert_eq!(sent, vec![(ORCH, MsgClass::Saga, expected_request.clone())]);
        }
        rig.grant_realm();
        assert_eq!(rig.world.resource::<RealmAuthority>().0, Some(Fence(1)));
        // Granted: no more requests (and no frames — no sessions yet).
        assert_eq!(rig.tick(vec![]), vec![]);
    }

    #[test]
    fn foreign_realm_grant_is_rejected_loudly_and_authority_stays_none() {
        let mut rig = Rig::new();
        let reply = DirectoryReply::Head {
            key: DirectoryKey::Realm(config().realm),
            record: Some(vd_wire::seams::directory::OwnerRecord {
                authority: AuthorityRef::Shard(NodeId(99)),
                fence: Fence(1),
                lease_expires: UniverseTick(1_000),
                in_transfer: None,
            }),
        };
        let _ = rig.tick(vec![wire_msg(
            ORCH,
            MsgClass::Saga,
            &InterShardFlow::DirectoryReply(reply),
        )]);
        assert_eq!(rig.world.resource::<RealmAuthority>().0, None);
    }

    #[test]
    fn a_lost_realm_lease_self_fences_the_shard() {
        // FENCE-1/5/8: once granted, a realm head showing a FOREIGN owner (a P2
        // takeover) or NO record makes the shard drop authority and stop frames —
        // a stale old owner cannot affect clients (fence rule 4).
        let mut rig = Rig::new();
        rig.grant_realm();
        assert_eq!(rig.world.resource::<RealmAuthority>().0, Some(Fence(1)));
        // A foreign owner head: self-fence.
        let foreign = DirectoryReply::Head {
            key: DirectoryKey::Realm(config().realm),
            record: Some(vd_wire::seams::directory::OwnerRecord {
                authority: AuthorityRef::Shard(NodeId(99)),
                fence: Fence(2),
                lease_expires: UniverseTick(1_000),
                in_transfer: None,
            }),
        };
        let _ = rig.tick(vec![wire_msg(
            ORCH,
            MsgClass::Saga,
            &InterShardFlow::DirectoryReply(foreign),
        )]);
        assert_eq!(
            rig.world.resource::<RealmAuthority>().0,
            None,
            "self-fenced"
        );

        // Re-grant, then a headless realm read (record gone): also self-fence.
        rig.grant_realm();
        assert_eq!(rig.world.resource::<RealmAuthority>().0, Some(Fence(1)));
        let gone = DirectoryReply::Head {
            key: DirectoryKey::Realm(config().realm),
            record: None,
        };
        let _ = rig.tick(vec![wire_msg(
            ORCH,
            MsgClass::Saga,
            &InterShardFlow::DirectoryReply(gone),
        )]);
        assert_eq!(rig.world.resource::<RealmAuthority>().0, None);
    }

    #[test]
    fn a_granted_shard_periodically_re_reads_its_realm_head() {
        // With a re-check interval, a granted shard sends a HeadRead so a revoked
        // lease is OBSERVED (the self-fence reaction is otherwise unreachable).
        let mut rig = Rig::with_config(StubConfig {
            realm_recheck_interval: 2,
            ..config()
        });
        rig.grant_realm();
        let is_head_read = |sent: &[(NodeId, MsgClass, Vec<u8>)]| {
            sent.iter()
                .filter(|(to, _, _)| *to == ORCH)
                .any(|(_, _, bytes)| {
                    let flow: InterShardFlow =
                        postcard::from_bytes(bytes).expect("directory flow decodes");
                    flow == InterShardFlow::Directory(DirectoryOp::HeadRead {
                        key: DirectoryKey::Realm(config().realm),
                    })
                })
        };
        // An EVEN tick (local_tick % 2 == 0) re-reads the realm head.
        rig.set_local_tick(2);
        assert!(is_head_read(&rig.tick(vec![])), "even tick re-reads");
        // An ODD tick does not (the interval gate's other branch).
        rig.set_local_tick(3);
        assert!(!is_head_read(&rig.tick(vec![])), "odd tick is quiet");
    }

    #[test]
    fn a_granted_shard_renews_its_realm_and_granted_entities_on_cadence() {
        // D-3 heartbeat: on the renew cadence a granted shard re-sends LeaseRenew for its Realm AND
        // every GRANTED, NON-DEPARTING Entity — never a non-granted (still-granting) or departing
        // (logging-out) dot. INERT off-cadence + when the interval is 0.
        let mut rig = Rig::with_config(StubConfig {
            lease_renew_interval_ticks: 4,
            ..config()
        });
        rig.grant_realm();
        let mk = |entity: EntityId, granted: bool, departing: bool| Dot {
            entity,
            account: AccountId(1),
            session_fence: Fence(1),
            gateway: GATEWAY,
            granted,
            input_active: false,
            adopting: false,
            authority: Authority::Owned { fence: Fence(1) },
            departing,
            entity_fence: Fence(1),
            pose: StampedPose::at_rest(config().frame, DVec3::ZERO, UniverseTick(0)),
            yaw: 0.0,
            pitch: 0.0,
            last_applied_seq: None,
            prev_offset: DVec3::ZERO,
        };
        let granted_e = EntityId::pack(EntityKind::Player, 10, 1, 1);
        let provisional_e = EntityId::pack(EntityKind::Player, 10, 2, 2);
        let departing_e = EntityId::pack(EntityKind::Player, 10, 3, 3);
        {
            let mut dots = rig.world.resource_mut::<Dots>();
            dots.0.insert(SessionId(1), mk(granted_e, true, false));
            dots.0.insert(SessionId(2), mk(provisional_e, false, false)); // emits LeaseGrant
            dots.0.insert(SessionId(3), mk(departing_e, true, true)); // emits LeaseRevoke
        }
        let renew_keys = |sent: &[(NodeId, MsgClass, Vec<u8>)]| -> Vec<DirectoryKey> {
            sent.iter()
                .filter(|(to, _, _)| *to == ORCH)
                .filter_map(
                    |(_, _, b)| match postcard::from_bytes::<InterShardFlow>(b) {
                        Ok(InterShardFlow::Directory(DirectoryOp::LeaseRenew { key, fence })) => {
                            assert_eq!(fence, Fence(1), "renews at the held fence");
                            Some(key)
                        }
                        _ => None,
                    },
                )
                .collect()
        };
        // A multiple tick renews the Realm + the granted, non-departing entity ONLY.
        rig.set_local_tick(4);
        let on = renew_keys(&rig.tick(vec![]));
        assert!(
            on.contains(&DirectoryKey::Realm(config().realm)),
            "the realm lease is renewed"
        );
        assert!(
            on.contains(&DirectoryKey::Entity(granted_e)),
            "a granted, non-departing entity lease is renewed"
        );
        assert!(
            !on.contains(&DirectoryKey::Entity(provisional_e)),
            "a non-granted (still-granting) entity is NOT renewed"
        );
        assert!(
            !on.contains(&DirectoryKey::Entity(departing_e)),
            "a departing (logging-out) entity is NOT renewed"
        );
        // Off-cadence: no heartbeat at all (the interval gate's modulo branch). The inert branch
        // (interval == 0) is covered by every other granted-shard test (all run config() at interval 0).
        rig.set_local_tick(5);
        assert!(
            renew_keys(&rig.tick(vec![])).is_empty(),
            "an off-cadence tick emits no LeaseRenew"
        );
    }

    #[test]
    fn a_cohosting_shard_renews_its_child_realm_lease_on_cadence() {
        // D-3 heartbeat, co-hosting (task #149): a MULTI-realm shard renews its PRIMARY realm AND every
        // CO-HOSTED CHILD realm it holds (the `cohosted.0` chain in the renewal set). Drives the co-host
        // renewal closure that a single-realm shard never reaches (`cohosted.0` empty). Boot System 7
        // (primary) + Planet 7 (co-hosted child), then a cadence tick renews BOTH realm keys.
        let mut rig = Rig::with_config(StubConfig {
            lease_renew_interval_ticks: 4,
            ..cohost_planet_config()
        });
        boot_cohost_planet(&mut rig); // primary System 7 on RealmAuthority + child Planet 7 on CoHostedAuthority
        assert_eq!(
            rig.world
                .resource::<CoHostedAuthority>()
                .0
                .get(&RealmId::Planet(7)),
            Some(&Fence(1)),
            "precondition: the co-hosted Planet-7 head is held (so the renewal chain has a child to emit)",
        );
        // A still-granting (provisional) dot so the cadence tick ALSO emits a `LeaseGrant` to ORCH — a
        // NON-`LeaseRenew` op that exercises the extractor's fall-through arm (no uncoverable `_ => None`).
        rig.world.resource_mut::<Dots>().0.insert(
            SessionId(1),
            Dot {
                entity: EntityId::pack(EntityKind::Player, 7, 1, 1),
                account: AccountId(1),
                session_fence: Fence(1),
                gateway: GATEWAY,
                granted: false, // provisional ⇒ emits LeaseGrant, NOT LeaseRenew
                input_active: false,
                adopting: false,
                authority: Authority::Owned { fence: Fence(1) },
                departing: false,
                entity_fence: Fence(1),
                pose: StampedPose::at_rest(config().frame, DVec3::ZERO, UniverseTick(0)),
                yaw: 0.0,
                pitch: 0.0,
                last_applied_seq: None,
                prev_offset: DVec3::ZERO,
            },
        );
        let renew_keys = |sent: &[(NodeId, MsgClass, Vec<u8>)]| -> Vec<DirectoryKey> {
            sent.iter()
                .filter(|(to, _, _)| *to == ORCH)
                .filter_map(
                    |(_, _, b)| match postcard::from_bytes::<InterShardFlow>(b) {
                        Ok(InterShardFlow::Directory(DirectoryOp::LeaseRenew { key, .. })) => {
                            Some(key)
                        }
                        _ => None,
                    },
                )
                .collect()
        };
        rig.set_local_tick(4); // on the renew cadence
        let on = renew_keys(&rig.tick(vec![]));
        assert!(
            on.contains(&DirectoryKey::Realm(RealmId::System(7))),
            "the PRIMARY realm lease is renewed: {on:?}",
        );
        assert!(
            on.contains(&DirectoryKey::Realm(RealmId::Planet(7))),
            "the CO-HOSTED CHILD realm lease is ALSO renewed (the co-host chain): {on:?}",
        );
    }

    #[test]
    fn logout_revokes_at_the_recorded_entity_fence_not_a_literal() {
        // FENCE-1/5/8: a dot granted at a NON-genesis fence revokes at THAT fence on
        // logout — a hardcoded literal would be Refused and strand the logout.
        let mut rig = Rig::new();
        rig.grant_realm();
        let _ = rig.attach_request(SESSION, GATEWAY);
        let entity = rig.world.resource::<Dots>().0[&SESSION].entity;
        // The directory granted the entity at fence 5 (a transfer advanced it).
        let granted_at_5 = DirectoryReply::Head {
            key: DirectoryKey::Entity(entity),
            record: Some(vd_wire::seams::directory::OwnerRecord {
                authority: AuthorityRef::Shard(SHARD),
                fence: Fence(5),
                lease_expires: UniverseTick(1_000),
                in_transfer: None,
            }),
        };
        let _ = rig.tick(vec![wire_msg(
            ORCH,
            MsgClass::Saga,
            &InterShardFlow::DirectoryReply(granted_at_5),
        )]);
        assert_eq!(
            rig.world.resource::<Dots>().0[&SESSION].entity_fence,
            Fence(5)
        );
        // Detach, then the retry driver revokes at the RECORDED fence 5.
        let detach = GatewayToShard::DetachSession {
            session: SESSION,
            fence: Fence(1),
        };
        let _ = rig.tick(vec![wire_msg(GATEWAY, MsgClass::Control, &detach)]);
        let sent = rig.tick(vec![]);
        let expected_revoke = InterShardFlow::Directory(DirectoryOp::LeaseRevoke {
            key: DirectoryKey::Entity(entity),
            fence: Fence(5),
        });
        let to_orch: Vec<InterShardFlow> = sent
            .iter()
            .filter(|(to, _, _)| *to == ORCH)
            .map(|(_, _, bytes)| postcard::from_bytes(bytes).expect("flow decodes"))
            .collect();
        assert!(
            to_orch.contains(&expected_revoke),
            "revoke at the recorded fence 5, not a literal: {to_orch:?}"
        );
    }

    #[test]
    fn non_head_directory_replies_are_ignored() {
        // CAS results / clock answers carry no shard obligation (the catch-all arm).
        let mut rig = Rig::new();
        rig.grant_realm();
        let cas = DirectoryReply::CasResult {
            key: DirectoryKey::Realm(config().realm),
            outcome: vd_wire::seams::directory::CasOutcome::Won {
                new_fence: Fence(9),
            },
        };
        let clock = DirectoryReply::ClockNow {
            universe_tick: UniverseTick(5),
            epoch: vd_core::EpochId(1),
        };
        // A non-reply InterShardFlow arm misdirected to the stub on Saga (a SagaAck — the
        // gateway→saga ack) is also ignored: the stub handles ONLY DirectoryReply, every
        // other arm is a no-op (the dispatch `Ok(_) => return`), never a panic or a decode
        // error. (The real Transfer arm to a dest shard lands at Slice 1d.)
        let stray = InterShardFlow::SagaAck(
            vd_wire::seams::transfer_control::TransferControlAck::Committed {
                transfer: vd_core::TransferId(9),
            },
        );
        let _ = rig.tick(vec![
            wire_msg(ORCH, MsgClass::Saga, &InterShardFlow::DirectoryReply(cas)),
            wire_msg(ORCH, MsgClass::Saga, &InterShardFlow::DirectoryReply(clock)),
            wire_msg(ORCH, MsgClass::Saga, &stray),
        ]);
        // Authority unaffected by non-Head replies and the stray arm.
        assert_eq!(rig.world.resource::<RealmAuthority>().0, Some(Fence(1)));
    }

    #[test]
    fn unrelated_directory_replies_are_ignored() {
        let mut rig = Rig::new();
        // A headless realm read and an entity head: neither grants authority.
        let none_head = DirectoryReply::Head {
            key: DirectoryKey::Realm(config().realm),
            record: None,
        };
        let entity_head = DirectoryReply::Head {
            key: DirectoryKey::Entity(EntityId(1)),
            record: None,
        };
        let _ = rig.tick(vec![
            wire_msg(
                ORCH,
                MsgClass::Saga,
                &InterShardFlow::DirectoryReply(none_head),
            ),
            wire_msg(
                ORCH,
                MsgClass::Saga,
                &InterShardFlow::DirectoryReply(entity_head),
            ),
        ]);
        assert_eq!(rig.world.resource::<RealmAuthority>().0, None);
    }

    #[test]
    fn undecodable_messages_are_survived() {
        let mut rig = Rig::new();
        let garbage = vec![0xFF, 0x00, 0x13, 0x37];
        let _ = rig.tick(vec![
            Inbound::Wire {
                from: ORCH,
                class: MsgClass::Saga,
                bytes: garbage.clone().into(),
            },
            Inbound::Wire {
                from: GATEWAY,
                class: MsgClass::Control,
                bytes: garbage.into(),
            },
            // Non-wire inbound is skipped by the dispatcher.
            Inbound::NodeUnreachable {
                to: GATEWAY,
                class: MsgClass::Snapshot,
                undelivered: MsgId(1),
            },
            // Snapshot/Membership classes carry nothing for the stub dispatcher.
            Inbound::Wire {
                from: GATEWAY,
                class: MsgClass::Snapshot,
                bytes: vec![1].into(),
            },
            Inbound::Wire {
                from: ORCH,
                class: MsgClass::Membership,
                bytes: vec![2].into(),
            },
        ]);
        assert_eq!(rig.world.resource::<Dots>().0.len(), 0);
        // Both decode failures (Saga + Control) are COUNTED, never silent (ROB-E2E-1);
        // the Snapshot/Membership garbage is not decoded by the stub, so it adds nothing.
        assert_eq!(rig.world.resource::<StubStats>().undecodable, 2);
    }

    /// The absolute Universe-root frame a STORED spawn pose lives in (`container_coord_at`'s frame).
    fn root_frame() -> FrameRef {
        FrameRef::SystemSpace { system_seed: 0 }
    }

    #[test]
    fn resolve_spawn_pose_rebinds_a_stored_absolute_pose_into_the_leaf_realm_frame() {
        // 5f-3b `Some` arm: a login WITH a stored spawn pose is admitted at THAT pose, rebound into the leaf
        // realm's frame via the ONE frame machinery — NOT the origin, and NOT a raw-offset replant. The leaf
        // here is an AREA under a Planet: an `Area` frame is LOSSY (it needs its enclosing Planet as parent
        // PROVENANCE), so this proves the parent from `leaf.parent()` is threaded into `rebind_pose_to_dest`
        // to FORM the `AreaLocal{planet_seed, area_seed}` frame — the frame-coherence path P4's ephemeris
        // rides. Under P3 IdentityFrames the position is UNCHANGED; only the frame LABEL flips.
        use vd_core::realm_path::RealmKindTag;
        let account = AccountId(0x5F3B);
        let stored_pos = DVec3::new(100.0, 200.0, 300.0);
        // ABSOLUTE stored pose: the Universe-root frame `container_coord_at` reads (SystemSpace{0}).
        let stored = StampedPose::at_rest(root_frame(), stored_pos, UniverseTick(0));
        // A multi-level leaf: Area(99) under Planet(7) — so `leaf.parent()` is `Some` (exercises the
        // `.map(|p| p.lowered())` provenance closure) AND the dest frame is the lossy Area arm.
        let planet = RealmCoord::from_path(RealmPath::from_levels(vec![RealmLevel::new(
            RealmKindTag::Planet,
            7,
        )]))
        .expect("a one-level path has a leaf");
        let leaf = planet.child(RealmLevel::new(RealmKindTag::Area, 99));
        let mut cfg = config();
        cfg.spawn_poses.insert(account, stored);
        let got = resolve_spawn_pose(&cfg, account, &leaf, UniverseTick(77));
        // Frame LABEL flipped to the leaf realm's AreaLocal frame (formed from the Planet parent
        // provenance), NOT the stored absolute SystemSpace{0}.
        assert_eq!(
            got.frame,
            FrameRef::AreaLocal {
                planet_seed: 7,
                area_seed: 99,
            }
        );
        // Position is the STORED pos (NOT origin), unchanged under the P3 identity rebind.
        assert_eq!(got.pos.offset(), stored_pos);
        // Re-stamped to the current clock tick; at rest (the stored pose was at rest).
        assert_eq!(got.universe_tick, UniverseTick(77));
        assert_eq!(got.vel, DVec3::ZERO);
    }

    #[test]
    fn resolve_spawn_pose_without_an_entry_is_origin_at_rest_byte_identical() {
        // 5f-3b `None` arm: a login WITHOUT a stored pose is origin-at-rest in this shard's frame —
        // BYTE-IDENTICAL to the pre-5f-3b admit literal (`at_rest(config.frame, ZERO, tick)`).
        let cfg = config(); // empty spawn_poses
        let got = resolve_spawn_pose(&cfg, AccountId(0xAB), &cfg.own_coord, UniverseTick(42));
        assert_eq!(
            got,
            StampedPose::at_rest(cfg.frame, DVec3::ZERO, UniverseTick(42))
        );
    }

    #[test]
    fn login_admits_the_dot_at_its_stored_spawn_pose_through_the_one_admit_path() {
        // The admit-path proof (the wiring, not just the helper): a real AttachSession for an account WITH
        // a stored spawn pose births its dot at the rebound stored pose — same Ghost-birth admit path, only
        // the pose value differs. `attach_request` attaches AccountId(5), so key the stand-in on it.
        let stored_pos = DVec3::new(11.0, -22.0, 33.0);
        let stored = StampedPose::at_rest(root_frame(), stored_pos, UniverseTick(0));
        let mut rig = Rig::with_config(StubConfig {
            spawn_poses: BTreeMap::from([(AccountId(5), stored)]),
            ..config()
        });
        rig.grant_realm();
        let _ = rig.attach_request(SESSION, GATEWAY);
        let dot = rig.world.resource::<Dots>().0[&SESSION];
        // Born at the STORED pose, rebound into the shard's realm frame (System 7), re-stamped to the
        // rig clock (universe_tick 100) — NOT origin-at-rest.
        assert_eq!(dot.pose.frame, FrameRef::SystemSpace { system_seed: 7 });
        assert_eq!(dot.pose.pos.offset(), stored_pos);
        assert_eq!(dot.pose.universe_tick, UniverseTick(100));
        // `prev_offset` is seeded from the SAME stored offset (not the old hardcoded ZERO).
        assert_eq!(dot.prev_offset, stored_pos);
    }

    #[test]
    fn login_without_a_stored_pose_births_at_the_origin_unchanged() {
        // The admit-path `None` proof: an account with no stand-in entry births origin-at-rest — the
        // byte-identical pre-5f-3b behaviour through the real attach path.
        let mut rig = Rig::new(); // config() ⇒ empty spawn_poses
        rig.grant_realm();
        let _ = rig.attach_request(SESSION, GATEWAY);
        let dot = rig.world.resource::<Dots>().0[&SESSION];
        assert_eq!(
            dot.pose,
            StampedPose::at_rest(config().frame, DVec3::ZERO, UniverseTick(100))
        );
        assert_eq!(dot.prev_offset, DVec3::ZERO);
    }

    #[test]
    fn attach_before_realm_grant_is_deferred_and_counted() {
        let mut rig = Rig::new();
        let sent = rig.attach_request(SESSION, GATEWAY);
        // Only the lease re-request went out — no attach reply.
        assert_eq!(sent.len(), 1);
        assert_eq!(rig.world.resource::<StubStats>().attaches_deferred, 1);
        assert_eq!(rig.world.resource::<Dots>().0.len(), 0);
    }

    #[test]
    fn attach_is_two_phase_authority_derives_from_the_directory() {
        let mut rig = Rig::new();
        rig.grant_realm();

        // Phase 1: the attach request spawns a PROVISIONAL dot and asks the
        // directory for its entity grant — no attach reply, no frames yet.
        let sent = rig.attach_request(SESSION, GATEWAY);
        let dot = rig.world.resource::<Dots>().0[&SESSION];
        assert!(!dot.granted, "provisional until the directory records it");
        assert_eq!(dot.account, AccountId(5));
        assert_eq!(dot.gateway, GATEWAY);
        assert_eq!(dot.entity.kind_tag(), EntityKind::Player as u8);
        assert_eq!(dot.entity.mint_shard(), 10, "minted by THIS shard");
        let grant: InterShardFlow = postcard::from_bytes(&sent[0].2).expect("decode");
        assert_eq!(
            grant,
            InterShardFlow::Directory(DirectoryOp::LeaseGrant {
                key: DirectoryKey::Entity(dot.entity),
                owner: AuthorityRef::Shard(SHARD),
                fence: Fence(1),
            })
        );
        assert!(sent.iter().all(|(to, _, _)| *to == ORCH), "directory only");
        assert_eq!(
            decode_frames(&sent).len(),
            0,
            "provisional dots are invisible"
        );
        // The grant is retried every tick until confirmed (idempotent by fence).
        let sent = rig.tick(vec![]);
        assert_eq!(sent.len(), 1);
        assert_eq!(sent[0].0, ORCH);

        // Phase 2: the grant confirmation makes the dot HELD: SessionAttached
        // (with the REAL realm fence) and the first frame flow the same tick.
        // (The retry system also fires one last pre-grant request that tick.)
        let sent = rig.confirm_entity_grant(SESSION);
        assert!(rig.world.resource::<Dots>().0[&SESSION].granted);
        let to_gateway: Vec<ShardToGateway> = sent
            .iter()
            .filter(|(to, class, _)| (*to == GATEWAY) & (*class == MsgClass::Control))
            .map(|(_, _, bytes)| postcard::from_bytes(bytes).expect("decode"))
            .collect();
        assert_eq!(
            to_gateway,
            vec![ShardToGateway::SessionAttached {
                session: SESSION,
                entity: dot.entity,
                frame: config().frame,
                realm_fence: Fence(1),
            }]
        );
        assert_eq!(decode_frames(&sent).len(), 1, "held dots render");
        // A duplicate grant head is idempotent (no second attach reply).
        let sent = rig.confirm_entity_grant(SESSION);
        let attach_replies = sent
            .iter()
            .filter(|(_, class, _)| *class == MsgClass::Control)
            .count();
        assert_eq!(attach_replies, 0);
    }

    #[test]
    fn foreign_entity_grants_and_pregrant_races_are_survived() {
        let mut rig = Rig::new();
        rig.grant_realm();
        let _ = rig.attach_request(SESSION, GATEWAY);
        let entity = rig.world.resource::<Dots>().0[&SESSION].entity;
        // The directory says ANOTHER shard owns the entity: loud, not granted.
        let foreign = DirectoryReply::Head {
            key: DirectoryKey::Entity(entity),
            record: Some(vd_wire::seams::directory::OwnerRecord {
                authority: AuthorityRef::Shard(NodeId(99)),
                fence: Fence(1),
                lease_expires: UniverseTick(1_000),
                in_transfer: None,
            }),
        };
        let _ = rig.tick(vec![wire_msg(
            ORCH,
            MsgClass::Saga,
            &InterShardFlow::DirectoryReply(foreign),
        )]);
        assert!(!rig.world.resource::<Dots>().0[&SESSION].granted);
    }

    #[test]
    fn entity_grant_racing_ahead_of_the_realm_lease_waits_for_retry() {
        // The realm lease is NOT granted yet; an entity head arriving anyway
        // cannot activate the dot (no realm fence to stamp) — the per-tick retry
        // resolves it once the realm lease lands.
        let mut rig = Rig::new();
        rig.grant_realm();
        let _ = rig.attach_request(SESSION, GATEWAY);
        let entity = rig.world.resource::<Dots>().0[&SESSION].entity;
        rig.world.resource_mut::<RealmAuthority>().0 = None;
        let head = DirectoryReply::Head {
            key: DirectoryKey::Entity(entity),
            record: Some(vd_wire::seams::directory::OwnerRecord {
                authority: AuthorityRef::Shard(SHARD),
                fence: Fence(1),
                lease_expires: UniverseTick(1_000),
                in_transfer: None,
            }),
        };
        let _ = rig.tick(vec![wire_msg(
            ORCH,
            MsgClass::Saga,
            &InterShardFlow::DirectoryReply(head),
        )]);
        assert!(!rig.world.resource::<Dots>().0[&SESSION].granted);
    }

    #[test]
    fn reattach_is_idempotent_and_a_higher_fence_upgrades() {
        let mut rig = Rig::new();
        rig.grant_realm();
        let _ = rig.attach();
        let first_entity = rig.world.resource::<Dots>().0[&SESSION].entity;
        // Same-fence re-attach: same entity, still one dot.
        let _ = rig.attach();
        assert_eq!(rig.world.resource::<Dots>().0.len(), 1);
        assert_eq!(
            rig.world.resource::<Dots>().0[&SESSION].entity,
            first_entity
        );
        // Higher-fence re-attach upgrades the stored fence.
        let msg = GatewayToShard::AttachSession {
            session: SESSION,
            fence: Fence(3),
            account: AccountId(5),
        };
        let _ = rig.tick(vec![wire_msg(GATEWAY, MsgClass::Control, &msg)]);
        assert_eq!(
            rig.world.resource::<Dots>().0[&SESSION].session_fence,
            Fence(3)
        );
    }

    #[test]
    fn applied_input_moves_the_dot_and_is_logged() {
        let mut rig = Rig::new();
        rig.grant_realm();
        let _ = rig.attach();
        // Forward input, yaw 0: heading is -Z.
        let _ = rig.tick(vec![input_msg(1, Fence(1), [1.0, 0.0, 0.0], [0.0, 0.0])]);
        let dot = rig.world.resource::<Dots>().0[&SESSION];
        let expected_step = 2.0 * 0.05; // speed * dt
        assert!(
            (dot.pose.pos.offset().z + expected_step).abs() < 1e-12,
            "moved -Z"
        );
        assert_eq!(dot.pose.pos.offset().x, 0.0);
        assert_eq!(dot.last_applied_seq, Some(1));
        assert_eq!(
            rig.world.resource::<InputLog>().applied(),
            vec![(SESSION, 1)]
        );
        // Velocity is displacement over dt.
        assert!((dot.pose.vel.z + 2.0).abs() < 1e-12);
    }

    #[test]
    fn yaw_rotates_the_heading() {
        let mut rig = Rig::new();
        rig.grant_realm();
        let _ = rig.attach();
        // Look 90° left (yaw = +π/2), then walk forward: heading becomes -X.
        let half_pi = std::f32::consts::FRAC_PI_2;
        let _ = rig.tick(vec![input_msg(
            1,
            Fence(1),
            [1.0, 0.0, 0.0],
            [half_pi, 0.0],
        )]);
        let dot = rig.world.resource::<Dots>().0[&SESSION];
        let expected_step = 2.0 * 0.05;
        assert!(
            (dot.pose.pos.offset().x + expected_step).abs() < 1e-6,
            "moved -X"
        );
        assert!(dot.pose.pos.offset().z.abs() < 1e-6);
    }

    #[test]
    fn pitch_is_clamped_at_the_gimbal_pole_never_wraps_past_vertical() {
        // WB-1: a huge look-up delta must NOT accumulate past ±π/2 (which would flip the
        // authoritative orientation). Two big up-pitches in a row stay clamped at the limit.
        let mut rig = Rig::new();
        rig.grant_realm();
        let _ = rig.attach();
        let _ = rig.tick(vec![input_msg(1, Fence(1), [0.0, 0.0, 0.0], [0.0, 3.0])]);
        let _ = rig.tick(vec![input_msg(2, Fence(1), [0.0, 0.0, 0.0], [0.0, 3.0])]);
        let dot = rig.world.resource::<Dots>().0[&SESSION];
        assert_eq!(
            dot.pitch,
            vd_core::kinematics::PITCH_LIMIT,
            "accumulated pitch is held at the limit, never wrapped past vertical"
        );
    }

    #[test]
    fn strafe_and_vertical_axes_integrate() {
        let mut rig = Rig::new();
        rig.grant_realm();
        let _ = rig.attach();
        // Strafe right + up, no forward; clamp catches the out-of-range axis.
        let _ = rig.tick(vec![input_msg(1, Fence(1), [0.0, 2.0, 1.0], [0.0, 0.0])]);
        let dot = rig.world.resource::<Dots>().0[&SESSION];
        let expected_step = 2.0 * 0.05;
        assert!(
            (dot.pose.pos.offset().x - expected_step).abs() < 1e-12,
            "clamped strafe"
        );
        assert!(
            (dot.pose.pos.offset().y - expected_step).abs() < 1e-12,
            "vertical"
        );
    }

    #[test]
    fn occupant_movement_dilates_with_the_realm_time_multiplier() {
        // A realm's SUBJECTIVE time factor scales OCCUPANT movement: a slow-time realm (0.5) advances the
        // dot HALF as far per tick; the default (1.0) is byte-identical to the un-scaled `speed·dt` step.
        let step_len_at = |mult: f64| {
            let mut rig = Rig::with_config(StubConfig {
                time_multiplier: mult,
                ..config()
            });
            rig.grant_realm();
            let _ = rig.attach();
            // Pure forward input (movement = [forward, strafe, up]).
            let _ = rig.tick(vec![input_msg(1, Fence(1), [1.0, 0.0, 0.0], [0.0, 0.0])]);
            rig.world.resource::<Dots>().0[&SESSION]
                .pose
                .pos
                .offset()
                .length()
        };
        // Default 1.0 = the un-multiplied step (`move_speed 2.0 · dt 0.05` = 0.1) — byte-identical.
        assert!((step_len_at(1.0) - 2.0 * 0.05).abs() < 1e-12);
        // The 0.5-multiplier realm moved EXACTLY half as far (time dilation).
        assert!((step_len_at(0.5) - step_len_at(1.0) * 0.5).abs() < 1e-12);
    }

    #[test]
    fn every_discard_reason_is_logged() {
        let mut rig = Rig::new();
        rig.grant_realm();
        let _ = rig.attach();
        let _ = rig.tick(vec![input_msg(5, Fence(1), [1.0, 0.0, 0.0], [0.0, 0.0])]);

        // DuplicateSeq: same seq again.
        let _ = rig.tick(vec![input_msg(5, Fence(1), [1.0, 0.0, 0.0], [0.0, 0.0])]);
        // StaleFence: fence below the session's.
        let _ = rig.tick(vec![input_msg(
            6,
            Fence::GENESIS,
            [1.0, 0.0, 0.0],
            [0.0, 0.0],
        )]);
        // MalformedInput: undecodable payload.
        let bad = GatewayToShard::SessionInput {
            session: SESSION,
            fence: Fence(1),
            input_bytes: vec![0xFF],
        };
        let _ = rig.tick(vec![wire_msg(GATEWAY, MsgClass::Input, &bad)]);
        // UnknownSession.
        let unknown = GatewayToShard::SessionInput {
            session: SessionId(0xBB),
            fence: Fence(1),
            input_bytes: vec![],
        };
        let _ = rig.tick(vec![wire_msg(GATEWAY, MsgClass::Input, &unknown)]);
        // PendingAuthority: input for a provisional (ungranted) dot.
        let _ = rig.attach_request(SessionId(0xCC), GATEWAY);
        let pending = GatewayToShard::SessionInput {
            session: SessionId(0xCC),
            fence: Fence(1),
            input_bytes: vec![],
        };
        let _ = rig.tick(vec![wire_msg(GATEWAY, MsgClass::Input, &pending)]);
        // NonFiniteInput: a forged NaN component is caught by the finite gate.
        let _ = rig.tick(vec![input_msg(
            6,
            Fence(1),
            [f32::NAN, 0.0, 0.0],
            [0.0, 0.0],
        )]);

        let log = rig.world.resource::<InputLog>();
        assert_eq!(log.applied(), vec![(SESSION, 5)]);
        assert_eq!(
            log.discarded(),
            vec![
                (SESSION, Some(5), DiscardReason::DuplicateSeq),
                (SESSION, None, DiscardReason::StaleFence),
                (SESSION, None, DiscardReason::MalformedInput),
                (SessionId(0xBB), None, DiscardReason::UnknownSession),
                (SessionId(0xCC), None, DiscardReason::PendingAuthority),
                (SESSION, Some(6), DiscardReason::NonFiniteInput),
            ]
        );
    }

    #[test]
    fn non_finite_input_is_discarded_and_never_poisons_the_pose() {
        // ROB-1 (whole-codebase audit): a forged/corrupt NaN or Inf input must NEVER
        // integrate — NaN sticks in the authoritative pose forever and fans out to every
        // observer. The finite gate discards + counts it, the pose stays untouched, and
        // the seq does NOT advance (the input was never applied), so a subsequent FINITE
        // datagram at the same seq applies normally — the session is not wedged.
        let mut rig = Rig::new();
        rig.grant_realm();
        let _ = rig.attach();
        let _ = rig.tick(vec![input_msg(
            1,
            Fence(1),
            [f32::NAN, 0.0, 0.0],
            [f32::INFINITY, 0.0],
        )]);
        let dot = rig.world.resource::<Dots>().0[&SESSION];
        assert_eq!(
            dot.pose.pos.offset(),
            vd_core::glam::DVec3::ZERO,
            "pose untouched"
        );
        // Split asserts (no `&&` short-circuit branch — the HR5 coverage discipline).
        assert_eq!(dot.yaw, 0.0, "yaw untouched");
        assert_eq!(dot.pitch, 0.0, "pitch untouched");

        // The same seq, now finite: applies (the poisoned datagram never consumed it).
        let _ = rig.tick(vec![input_msg(1, Fence(1), [1.0, 0.0, 0.0], [0.0, 0.0])]);
        let log = rig.world.resource::<InputLog>();
        assert_eq!(log.applied(), vec![(SESSION, 1)]);
        assert_eq!(
            log.discarded(),
            vec![(SESSION, Some(1), DiscardReason::NonFiniteInput)]
        );
        let dot = rig.world.resource::<Dots>().0[&SESSION];
        assert!(
            dot.pose.pos.offset().is_finite(),
            "authoritative pose finite"
        );
        assert!(
            dot.pose.pos.offset().z < 0.0,
            "the finite input integrated (moved -Z)"
        );
    }

    #[test]
    fn action_bits_are_inert_two_datagrams_differing_only_in_action_bits_integrate_identically() {
        // D-41 / D-39.1 MECHANICAL-GUARD TRIPWIRE (exists-to-be-flipped). `action_bits` is INERT in the
        // current integrator — `integrate` reads ONLY `look` + `movement`, never `action_bits` — so two
        // inputs identical EXCEPT for `action_bits` MUST integrate to the identical authoritative pose
        // today. This flips RED the day the reliable client→shard discrete-action arm makes the sim
        // consume `action_bits` (its named consumers: P6 block-edit-forward + P11 PvP fire-registration,
        // DEFERRED D-39.1) — a hard guard that world-mutating actions (esp. PvP fire-reg) can NEVER be
        // silently gated onto the lossy UNRELIABLE input datagram that `action_bits` rides.
        let mut inert = Rig::new();
        inert.grant_realm();
        let _ = inert.attach();
        let mut set = Rig::new();
        set.grant_realm();
        let _ = set.attach();

        // A non-trivial input (movement + look both non-zero) so the pose actually MOVES — proving the
        // two agree on a REAL integration, not on a shared do-nothing origin.
        let movement = [1.0f32, 0.5, -0.25];
        let look = [0.3f32, 0.1];
        let _ = inert.tick(vec![input_msg_bits(1, Fence(1), movement, look, 0)]);
        let _ = set.tick(vec![input_msg_bits(1, Fence(1), movement, look, u32::MAX)]);

        let dot_inert = inert.world.resource::<Dots>().0[&SESSION];
        let dot_set = set.world.resource::<Dots>().0[&SESSION];
        // Dot is Copy + PartialEq (pose + yaw + pitch + vel): ONE equality assert is the strongest,
        // HR5-coverage-safe identical-pose check (no `matches!` false-arm, no `&&` short-circuit).
        assert_eq!(
            dot_inert, dot_set,
            "action_bits is inert: 0 vs u32::MAX must not change the integrated pose"
        );
    }

    #[test]
    fn a_higher_input_fence_upgrades_the_session() {
        let mut rig = Rig::new();
        rig.grant_realm();
        let _ = rig.attach();
        let _ = rig.tick(vec![input_msg(1, Fence(4), [0.0, 0.0, 0.0], [0.0, 0.0])]);
        assert_eq!(
            rig.world.resource::<Dots>().0[&SESSION].session_fence,
            Fence(4)
        );
        assert_eq!(
            rig.world.resource::<InputLog>().applied(),
            vec![(SESSION, 1)]
        );
    }

    #[test]
    fn detach_is_two_phase_release_via_the_directory() {
        let mut rig = Rig::new();
        rig.grant_realm();
        let _ = rig.attach();
        let entity = rig.world.resource::<Dots>().0[&SESSION].entity;
        let detach = GatewayToShard::DetachSession {
            session: SESSION,
            fence: Fence(1),
        };
        // Phase 1: the dot stays HELD (departing); the retry driver sends the
        // revoke on the following tick (and every tick until confirmed).
        let _ = rig.tick(vec![wire_msg(GATEWAY, MsgClass::Control, &detach)]);
        let dot = rig.world.resource::<Dots>().0[&SESSION];
        assert!(dot.departing, "held until the directory releases it");
        let sent = rig.tick(vec![]);
        let revokes: Vec<InterShardFlow> = sent
            .iter()
            .filter(|(to, _, _)| *to == ORCH)
            .map(|(_, _, bytes)| postcard::from_bytes(bytes).expect("decode"))
            .collect();
        assert!(
            revokes.contains(&InterShardFlow::Directory(DirectoryOp::LeaseRevoke {
                key: DirectoryKey::Entity(entity),
                fence: Fence(1),
            })),
            "the entity revoke is on its way"
        );
        // Departing dots no longer consume input.
        let _ = rig.tick(vec![input_msg(9, Fence(1), [1.0, 0.0, 0.0], [0.0, 0.0])]);
        assert_eq!(
            rig.world.resource::<InputLog>().discarded().last().copied(),
            Some((SESSION, None, DiscardReason::Departing))
        );
        // Phase 2: the headless entity head confirms the revoke — despawn + reply.
        let gone = DirectoryReply::Head {
            key: DirectoryKey::Entity(entity),
            record: None,
        };
        let sent = rig.tick(vec![wire_msg(
            ORCH,
            MsgClass::Saga,
            &InterShardFlow::DirectoryReply(gone),
        )]);
        assert_eq!(rig.world.resource::<Dots>().0.len(), 0);
        let confirms: Vec<ShardToGateway> = sent
            .iter()
            .filter(|(to, class, _)| (*to == GATEWAY) & (*class == MsgClass::Control))
            .map(|(_, _, bytes)| postcard::from_bytes(bytes).expect("decode"))
            .collect();
        assert_eq!(
            confirms,
            vec![ShardToGateway::SessionDetached { session: SESSION }]
        );
        // Unknown-session detach still confirms immediately (idempotent).
        let sent = rig.tick(vec![wire_msg(GATEWAY, MsgClass::Control, &detach)]);
        let confirm: ShardToGateway = postcard::from_bytes(&sent[0].2).expect("decode");
        assert_eq!(
            confirm,
            ShardToGateway::SessionDetached { session: SESSION }
        );
        assert_eq!(
            confirm.into_snapshot_bytes(),
            None,
            "only frames carry snapshot bytes"
        );
        // A provisional (ungranted) dot detaches immediately — no record exists.
        let _ = rig.attach_request(SessionId(0xDD), GATEWAY);
        let detach_pending = GatewayToShard::DetachSession {
            session: SessionId(0xDD),
            fence: Fence(1),
        };
        let sent = rig.tick(vec![wire_msg(GATEWAY, MsgClass::Control, &detach_pending)]);
        assert!(
            !rig.world
                .resource::<Dots>()
                .0
                .contains_key(&SessionId(0xDD))
        );
        let confirms = sent
            .iter()
            .filter(|(to, class, _)| (*to == GATEWAY) & (*class == MsgClass::Control))
            .count();
        assert_eq!(confirms, 1);
    }

    #[test]
    fn stale_fence_detach_is_discarded_with_reason() {
        let mut rig = Rig::new();
        rig.grant_realm();
        let _ = rig.attach();
        // Upgrade the session fence, then detach with the old one.
        let _ = rig.tick(vec![input_msg(1, Fence(4), [0.0, 0.0, 0.0], [0.0, 0.0])]);
        let stale = GatewayToShard::DetachSession {
            session: SESSION,
            fence: Fence(1),
        };
        let _ = rig.tick(vec![wire_msg(GATEWAY, MsgClass::Control, &stale)]);
        assert_eq!(rig.world.resource::<Dots>().0.len(), 1, "dot survives");
        let log = rig.world.resource::<InputLog>();
        assert_eq!(
            log.discarded().last().copied(),
            Some((SESSION, None, DiscardReason::StaleFence))
        );
    }

    #[test]
    fn frames_carry_all_dots_and_count_monotonically() {
        let mut rig = Rig::new();
        rig.grant_realm();
        let _ = rig.attach();
        // A second session through the same gateway (full grant flow).
        let _ = rig.attach_request(SessionId(0xBB), GATEWAY);
        let _ = rig.confirm_entity_grant(SessionId(0xBB));
        let sent = rig.tick(vec![]);
        let frames = decode_frames(&sent);
        assert_eq!(frames.len(), 1, "one gateway, one frame");
        let snap = &frames[0];
        assert_eq!(snap.sub, SubId(0));
        assert_eq!(snap.entities.len(), 2, "both dots present");
        assert_eq!(snap.source_tick, vd_core::TickId(1));
        assert_eq!(snap.universe_tick, UniverseTick(100));
        // Frame ids increment.
        let next = decode_frames(&rig.tick(vec![]));
        assert_eq!(next[0].frame_id, snap.frame_id + 1);
    }

    #[test]
    fn a_large_world_partitions_into_multiple_under_budget_frames() {
        // GW-1: many dots exceed the datagram budget, so the snapshot ships as
        // several same-frame_id sibling chunks, each encoding under the budget,
        // together carrying EVERY entity (no silent MTU drop).
        let mut rig = Rig::with_config(StubConfig {
            snapshot_datagram_budget: 300,
            ..config()
        });
        rig.grant_realm();
        // Insert 12 granted dots directly (bypassing the attach handshake).
        {
            let mut dots = rig.world.resource_mut::<Dots>();
            for n in 0..12u64 {
                dots.0.insert(
                    SessionId(u128::from(n) + 1),
                    Dot {
                        entity: EntityId::pack(EntityKind::Player, 10, n, n as u32),
                        account: AccountId(n as u128),
                        session_fence: Fence(1),
                        gateway: GATEWAY,
                        granted: true,
                        input_active: false,
                        adopting: false,
                        authority: Authority::Owned { fence: Fence(1) },
                        departing: false,
                        entity_fence: Fence(1),
                        pose: StampedPose::at_rest(config().frame, DVec3::ZERO, UniverseTick(100)),
                        yaw: 0.0,
                        pitch: 0.0,
                        last_applied_seq: None,
                        prev_offset: DVec3::ZERO,
                    },
                );
            }
        }
        let frames = decode_frames(&rig.tick(vec![]));
        assert!(
            frames.len() > 1,
            "12 dots must partition into multiple frames"
        );
        // Every chunk is under budget and the union is all 12 entities.
        let mut all_entities = std::collections::BTreeSet::new();
        for f in &frames {
            let encoded = postcard::to_allocvec(f).expect("encode").len();
            assert!(encoded <= 300, "chunk encodes to {encoded} > 300");
            for e in &f.entities {
                all_entities.insert(e.entity);
            }
        }
        assert_eq!(all_entities.len(), 12, "no entity lost across chunks");
        // §6.3: every chunk of ONE tick shares the SAME frame_id (each self-contained
        // latest-wins) so a reordered sibling chunk is never dropped as stale.
        let ids: std::collections::BTreeSet<u64> = frames.iter().map(|f| f.frame_id).collect();
        assert_eq!(ids.len(), 1, "all chunks of one tick share a frame_id");
        let tick0_id = *ids.iter().next().expect("at least one chunk");
        // The next tick's chunks all share a STRICTLY GREATER frame_id: the counter
        // advances exactly once per tick (monotonic between ticks, stable within one).
        let next = decode_frames(&rig.tick(vec![]));
        assert!(next.len() > 1, "still partitioned the next tick");
        let next_ids: std::collections::BTreeSet<u64> = next.iter().map(|f| f.frame_id).collect();
        assert_eq!(
            next_ids.len(),
            1,
            "next tick's chunks also share one frame_id"
        );
        assert_eq!(
            *next_ids.iter().next().expect("chunk"),
            tick0_id + 1,
            "frame_id advances exactly once per tick"
        );
    }

    #[test]
    fn two_gateways_each_get_the_frame() {
        let mut rig = Rig::new();
        rig.grant_realm();
        let _ = rig.attach();
        let other_gateway = NodeId(21);
        let _ = rig.attach_request(SessionId(0xBB), other_gateway);
        let _ = rig.confirm_entity_grant(SessionId(0xBB));
        let sent = rig.tick(vec![]);
        let snapshot_targets: Vec<NodeId> = sent
            .iter()
            .filter(|(_, class, _)| *class == MsgClass::Snapshot)
            .map(|(to, _, _)| *to)
            .collect();
        assert_eq!(snapshot_targets, vec![GATEWAY, other_gateway]);
    }

    #[test]
    fn minted_entities_are_unique_and_structured() {
        let mut mint = EntityMint {
            seq: 0,
            rng: SplitMix64::new(1),
        };
        let a = mint_entity(&mut mint, SHARD);
        let b = mint_entity(&mut mint, SHARD);
        assert_ne!(a, b);
        assert_eq!(a.seq(), 0);
        assert_eq!(b.seq(), 1);
        assert_eq!(a.kind_tag(), EntityKind::Player as u8);
        assert_eq!(a.mint_shard(), 10);
    }

    #[test]
    fn input_log_is_a_bounded_window_with_exact_totals() {
        // SCALE-3: a small window holds only the NEWEST entries (no unbounded
        // growth), while the totals are EXACT and evictions are counted.
        let mut log = InputLog::new(3);
        for seq in 0..10u64 {
            log.record_applied(SessionId(1), seq);
        }
        for seq in 0..4u64 {
            log.record_discarded(SessionId(2), Some(seq), DiscardReason::DuplicateSeq);
        }
        // Window holds the last 3 of each; totals count everything.
        assert_eq!(log.applied().len(), 3);
        assert_eq!(
            log.applied(),
            vec![(SessionId(1), 7), (SessionId(1), 8), (SessionId(1), 9)]
        );
        assert_eq!(log.discarded().len(), 3);
        assert_eq!(log.applied_total, 10);
        assert_eq!(log.discarded_total, 4);
        // 7 applied + 1 discarded evicted from the windows.
        assert_eq!(log.window_evictions, 8);
        // Capacity floors at 1.
        let mut tiny = InputLog::new(0);
        tiny.record_applied(SessionId(9), 1);
        tiny.record_applied(SessionId(9), 2);
        assert_eq!(tiny.applied(), vec![(SessionId(9), 2)]);
    }

    // ---- Slice 1c.5: the dest-side OpenInputSlot (transfer-destination input slot) ------

    /// The transfer subject the OpenInputSlot test helpers carry (the avatar the dest adopts).
    const SUBJECT: EntityId = EntityId(0xBEEF);

    fn open_input_slot_subj(
        session: SessionId,
        gateway: NodeId,
        resume_from_seq: u64,
        fence: Fence,
        subject: DirectoryKey,
    ) -> Inbound {
        wire_msg(
            gateway,
            MsgClass::Control,
            &GatewayToShard::OpenInputSlot {
                session,
                fence,
                account: AccountId(5),
                resume_from_seq,
                subject,
            },
        )
    }

    fn open_input_slot_f(
        session: SessionId,
        gateway: NodeId,
        resume_from_seq: u64,
        fence: Fence,
    ) -> Inbound {
        open_input_slot_subj(
            session,
            gateway,
            resume_from_seq,
            fence,
            DirectoryKey::Entity(SUBJECT),
        )
    }

    fn open_input_slot(session: SessionId, gateway: NodeId, resume_from_seq: u64) -> Inbound {
        open_input_slot_f(session, gateway, resume_from_seq, Fence(1))
    }

    fn input_for(session: SessionId, seq: u64, gateway: NodeId) -> Inbound {
        wire_msg(
            gateway,
            MsgClass::Input,
            &GatewayToShard::SessionInput {
                session,
                fence: Fence(1),
                input_bytes: postcard::to_allocvec(&InputDatagram {
                    seq,
                    is_cut_marker: false,
                    client_tick: vd_core::TickId(2),
                    movement: [1.0, 0.0, 0.0],
                    look: [0.0, 0.0],
                    action_bits: 0,
                })
                .expect("encode"),
            },
        )
    }

    #[test]
    fn open_input_slot_adopts_the_subject_input_active_without_attaching() {
        let mut rig = Rig::new();
        rig.grant_realm();
        let sent = rig.tick(vec![open_input_slot(SESSION, GATEWAY, 5)]);
        let dot = rig.world.resource::<Dots>().0[&SESSION];
        assert!(dot.input_active, "the slot is input-active");
        // 1c.8: the dot ADOPTS the subject — its entity IS the subject id (not a fresh mint),
        // it is `adopting`, NOT granted (the adopt HeadRead flips that), and a Ghost (no simulate).
        assert_eq!(dot.entity, SUBJECT, "the dot adopted the transfer subject");
        assert!(dot.adopting, "the dot is a transfer-dest adopt");
        assert!(
            !dot.granted,
            "the adopt HeadRead has not flipped granted yet"
        );
        assert!(
            !dot.authority.simulates(),
            "the adopt is a Ghost (the frozen mirror) — does not simulate, renders nothing"
        );
        assert_eq!(
            dot.authority,
            Authority::Ghost {
                source_fence: Fence::GENESIS,
                since_tick: TickId(1),
            },
            "the dest adopt is born the GENESIS frozen ghost mirror (Promoted to Owned by the crossing)"
        );
        assert_eq!(
            dot.entity_fence,
            Fence::GENESIS,
            "the adopt HeadRead fills the real fence"
        );
        assert_eq!(
            dot.last_applied_seq,
            Some(5),
            "seeded to the resume watermark"
        );
        assert_eq!(
            rig.world.resource::<StubStats>().last_input_slot_resume,
            Some(5),
            "the as-received resume watermark is latched for the conservation gate"
        );
        // The slot is a SILENT inbound state change: it emits NOTHING — no SessionAttached,
        // no re-home (the source still owns the client connection — R2), and the Ghost adopt dot
        // does not simulate so it renders no snapshot frame either.
        assert!(sent.is_empty(), "the input slot emits nothing back");
    }

    #[test]
    fn an_adopting_dot_head_reads_the_record_never_lease_grants() {
        // 1c.8 HR5: the request_pending_grants 3-way ADOPT arm — an adopting !granted dot emits
        // HeadRead{Entity} (to adopt the record the CAS moved here), NEVER a LeaseGrant (which
        // the directory would Refuse at the post-genesis CAS fence).
        let mut rig = Rig::new();
        rig.grant_realm();
        let _ = rig.tick(vec![open_input_slot(SESSION, GATEWAY, 5)]);
        let sent = rig.tick(vec![]);
        let to_orch: Vec<InterShardFlow> = sent
            .iter()
            .filter(|(to, _, _)| *to == ORCH)
            .map(|(_, _, bytes)| postcard::from_bytes(bytes).expect("flow decodes"))
            .collect();
        assert!(
            to_orch.contains(&InterShardFlow::Directory(DirectoryOp::HeadRead {
                key: DirectoryKey::Entity(SUBJECT),
            })),
            "the adopting dot HeadReads its subject: {to_orch:?}"
        );
        // Value-compare (NOT matches!, whose match-success arm would be an uncoverable region):
        // the EXACT LeaseGrant an adopting dot must NEVER send (it adopts via HeadRead instead).
        assert!(
            !to_orch.contains(&InterShardFlow::Directory(DirectoryOp::LeaseGrant {
                key: DirectoryKey::Entity(SUBJECT),
                owner: AuthorityRef::Shard(SHARD),
                fence: Fence::GENESIS.next(),
            })),
            "an adopting dot NEVER LeaseGrants its entity: {to_orch:?}"
        );
    }

    #[test]
    fn the_adopt_grant_flip_holds_authority_without_announcing_the_sub_until_promote() {
        // 1d.5b.3b: the adopt grant-flip sets granted + stamps entity_fence + STAYS Ghost, but NO
        // LONGER announces the dest sub — `SubscriptionReady` RELOCATED to on_saga_promote (announced
        // only at the genuine Ghost→Owned promote). So the client stays on the SOURCE sub until then,
        // and demote-before-promote is strict. It still pushes NO SessionAttached (R2).
        let mut rig = Rig::new();
        rig.grant_realm();
        let _ = rig.tick(vec![open_input_slot(SESSION, GATEWAY, 5)]);
        let sent = rig.tick(vec![adopted_head(Fence(2))]);
        let dot = rig.world.resource::<Dots>().0[&SESSION];
        // Split asserts (each &&-short-circuit false arm is uncoverable — HR5).
        assert!(dot.granted, "the adopt flips granted (authority-held)");
        assert_eq!(dot.entity_fence, Fence(2), "stamped the recorded CAS fence");
        assert!(
            !dot.authority.simulates(),
            "an adopt STAYS a Ghost (renders nothing) until the saga Promote flips it Owned"
        );
        assert!(!dot.adopting, "adopting is cleared on the flip");
        // NO SubscriptionReady at the adopt (it moved to the promote) and NO frame rendered.
        assert!(
            gw_replies(&sent).is_empty(),
            "the adopt announces NO gateway reply (the sub moved to the promote): {sent:?}"
        );
        assert_eq!(
            decode_frames(&sent).len(),
            0,
            "an adopted Ghost dot renders nothing (simulates()==false)"
        );

        // After the crossing lands, the saga Promote DOES announce the dest sub (the relocated one).
        let _ = rig.tick(vec![crossing_msg(TransferId(7), Fence(2), crossing_pose())]);
        let sent = rig.tick(vec![promote_msg(Fence(2), NodeId(99))]);
        assert_eq!(
            gw_replies(&sent),
            vec![ShardToGateway::SubscriptionReady {
                session: SESSION,
                entity: SUBJECT,
                frame: FrameRef::SystemSpace { system_seed: 7 },
                realm_fence: Fence(1), // the DEST realm fence (grant_realm set Fence(1))
            }],
            "the Promote announces exactly one SubscriptionReady, never a SessionAttached"
        );
        assert!(
            rig.world.resource::<Dots>().0[&SESSION]
                .authority
                .simulates(),
            "the Promote flips the dest Ghost→Owned"
        );
    }

    #[test]
    fn a_non_entity_subject_open_input_slot_is_a_counted_noop() {
        // 1c.8 HR5: the DirectoryKey::transfer_subject_entity None arm — a non-Entity subject (e.g.
        // a Realm saga driven through CommitAuthority) does NOT adopt; counted no-op, never a panic.
        let mut rig = Rig::new();
        rig.grant_realm();
        let sent = rig.tick(vec![open_input_slot_subj(
            SESSION,
            GATEWAY,
            5,
            Fence(1),
            DirectoryKey::Realm(RealmId::System(99)),
        )]);
        assert!(
            !rig.world.resource::<Dots>().0.contains_key(&SESSION),
            "a non-Entity subject mints no dot"
        );
        assert_eq!(rig.world.resource::<StubStats>().input_slots_malformed, 1);
        assert!(sent.is_empty(), "the no-op emits nothing");
    }

    #[test]
    fn applied_step_redelivery_is_idempotent() {
        // 1d.0 PERMANENT GATE: the dest applied_steps journal dedups by the frozen
        // IdempotencyKey::TransferStep (transfer, step_id). The FIRST apply records + returns
        // FirstApply; every redelivery of the SAME key returns AlreadyApplied with no re-effect;
        // a distinct step_id OR transfer is independent.
        let mut steps = AppliedSteps::default();
        let t = TransferId(1);
        assert_eq!(steps.journal_step(t, 0), StepOutcome::FirstApply);
        assert_eq!(
            steps.journal_step(t, 0),
            StepOutcome::AlreadyApplied,
            "a redelivered (transfer, step_id) is a no-op — never a second effect"
        );
        assert_eq!(
            steps.journal_step(t, 1),
            StepOutcome::FirstApply,
            "a distinct step_id is journaled independently"
        );
        assert_eq!(
            steps.journal_step(TransferId(2), 0),
            StepOutcome::FirstApply,
            "a distinct transfer is journaled independently"
        );
    }

    /// Whether the orchestrator-bound egress carries a specific `SagaAck` (value-compare; the
    /// `*to == ORCH` filter restricts decode to ack/directory traffic — frames go to gateways).
    fn saga_ack_to_orch(sent: &[(NodeId, MsgClass, Vec<u8>)], ack: TransferControlAck) -> bool {
        sent.iter()
            .filter(|(to, _, _)| *to == ORCH)
            .any(|(_, _, bytes)| {
                postcard::from_bytes::<InterShardFlow>(bytes).ok()
                    == Some(InterShardFlow::SagaAck(ack))
            })
    }

    #[test]
    fn the_saga_demote_flips_the_source_to_ghost_and_acks_unconditionally() {
        // 1d.5b.2 SOURCE consumer of the ordered Demote (the SOLE source-demote driver): drive
        // Owned→Frozen→Ghost (REUSING self_fence_foreign_entity) at the new owner fence, then ack
        // DemoteAck. A redelivered Demote finds an already-Ghost dot → the !simulates() counted no-op
        // — but STILL acks (the unconditional ack: never wedge the saga in Demoting).
        let mut rig = Rig::new();
        rig.grant_realm();
        let _ = rig.attach(); // a granted, locally-owned dot at Fence(1)
        let entity = rig.world.resource::<Dots>().0[&SESSION].entity;
        let demote = InterShardFlow::Demote(DemoteCmd {
            transfer: TransferId(7),
            subject: DirectoryKey::Entity(entity),
            new_owner_fence: Fence(2),
            step_id: DEMOTE_STEP,
        });
        let sent = rig.tick(vec![wire_msg(ORCH, MsgClass::Saga, &demote)]);
        assert_eq!(
            rig.world.resource::<Dots>().0[&SESSION].authority,
            Authority::Ghost {
                source_fence: Fence(2),
                since_tick: TickId(1),
            },
            "the ordered Demote drives Owned{{1}}→Frozen→Ghost at the new owner fence (2)"
        );
        // The demote keys `authority.fence()` to the new owner (2) but does NOT touch `entity_fence`
        // — it stays the dot's OWN old grant fence (1); the intended divergence on a retained Ghost.
        assert_eq!(
            rig.world.resource::<Dots>().0[&SESSION].entity_fence,
            Fence(1),
            "the demote leaves entity_fence at the dot's own grant fence (authority.fence() diverges)"
        );
        assert!(
            saga_ack_to_orch(
                &sent,
                TransferControlAck::DemoteAck {
                    transfer: TransferId(7)
                }
            ),
            "DemoteAck is sent to the orchestrator: {sent:?}"
        );
        // The self-fence is purely LOCAL — no directory write (no LeaseRevoke at the stale fence,
        // no delete of the dest's record): the saga demote's ONLY orch-bound emission is the
        // DemoteAck. (Preserves the deleted poll-era test's no-directory-write guard.)
        assert_eq!(
            sent.iter().filter(|(to, _, _)| *to == ORCH).count(),
            1,
            "the saga demote writes nothing to the directory — only the DemoteAck: {sent:?}"
        );
        // Redelivery on the already-Ghost dot: counted skip, but STILL acks.
        let sent = rig.tick(vec![wire_msg(ORCH, MsgClass::Saga, &demote)]);
        assert_eq!(
            rig.world.resource::<StubStats>().self_fence_skipped,
            1,
            "the already-Ghost redelivery is the !simulates() counted no-op"
        );
        assert!(
            saga_ack_to_orch(
                &sent,
                TransferControlAck::DemoteAck {
                    transfer: TransferId(7)
                }
            ),
            "the redelivery STILL acks DemoteAck (never wedge the saga)"
        );
    }

    #[test]
    fn the_saga_demote_on_an_unheld_entity_is_a_clean_noop_but_acks() {
        // The no-match arm of self_fence_foreign_entity (`foreign_takeover_target` finds no dot) —
        // now reachable ONLY via on_saga_demote, since the poll that used to drive it is torn out
        // (1d.5b.2). A Demote for an Entity this shard does not hold is a clean no-op (no flip, no
        // panic) but STILL acks DemoteAck (never wedge the saga in Demoting).
        let mut rig = Rig::new();
        rig.grant_realm(); // realm held, but NO dot attached → this shard holds no entity
        let unheld = EntityId::pack(EntityKind::Player, 99, 99, 99);
        let demote = InterShardFlow::Demote(DemoteCmd {
            transfer: TransferId(7),
            subject: DirectoryKey::Entity(unheld),
            new_owner_fence: Fence(2),
            step_id: DEMOTE_STEP,
        });
        let sent = rig.tick(vec![wire_msg(ORCH, MsgClass::Saga, &demote)]);
        assert!(
            rig.world.resource::<Dots>().0.is_empty(),
            "an unheld-entity Demote creates or mutates no dot"
        );
        assert_eq!(
            rig.world.resource::<StubStats>().self_fence_skipped,
            0,
            "the no-match arm is distinct from the already-Ghost skip arm"
        );
        assert!(
            saga_ack_to_orch(
                &sent,
                TransferControlAck::DemoteAck {
                    transfer: TransferId(7)
                }
            ),
            "an unheld-entity Demote STILL acks DemoteAck: {sent:?}"
        );
    }

    #[test]
    fn the_saga_demote_on_a_non_entity_subject_is_a_counted_noop_but_acks() {
        // The None arm of subject→entity: a Realm/Session/Ship subject has no local Entity dot to
        // demote — counted (`saga_demote_no_entity`), no flip, but the DemoteAck is STILL sent.
        let mut rig = Rig::new();
        rig.grant_realm();
        let demote = InterShardFlow::Demote(DemoteCmd {
            transfer: TransferId(7),
            subject: DirectoryKey::Realm(config().realm),
            new_owner_fence: Fence(2),
            step_id: DEMOTE_STEP,
        });
        let sent = rig.tick(vec![wire_msg(ORCH, MsgClass::Saga, &demote)]);
        assert_eq!(rig.world.resource::<StubStats>().saga_demote_no_entity, 1);
        assert!(
            saga_ack_to_orch(
                &sent,
                TransferControlAck::DemoteAck {
                    transfer: TransferId(7)
                }
            ),
            "a non-Entity Demote still acks: {sent:?}"
        );
    }

    #[test]
    fn the_saga_promote_flips_owned_announces_the_sub_and_spawns_the_ghost() {
        // 1d.5b.3b: on_saga_promote is the REAL promoter. Set up a transfer-dest Ghost dot (adopt +
        // crossing stores the pose, the dot STAYS Ghost), then the saga Promote: flips Ghost→Owned,
        // announces the dest read-sub (RELOCATED from adopt), registers the source ghost-neighbor,
        // and Spawns the source ghost (the dest DRIVES the feed). Redelivery re-acks only.
        let mut rig = Rig::new();
        rig.grant_realm();
        let _ = rig.tick(vec![open_input_slot(SESSION, GATEWAY, 5)]); // adopting dot for SUBJECT
        let _ = rig.tick(vec![adopted_head(Fence(2))]); // flip → granted Ghost
        let _ = rig.tick(vec![crossing_msg(TransferId(7), Fence(2), crossing_pose())]);
        assert!(
            !rig.world.resource::<Dots>().0[&SESSION]
                .authority
                .simulates(),
            "still Ghost after the crossing (the autonomous promote is gone)"
        );

        rig.set_local_tick(5); // pins the Spawn's since_tick deterministically
        let source = NodeId(99);
        let sent = rig.tick(vec![promote_msg(Fence(2), source)]);
        assert_eq!(
            rig.world.resource::<Dots>().0[&SESSION].authority,
            Authority::Owned { fence: Fence(2) },
            "the Promote flips the dest Ghost→Owned at the new fence"
        );
        assert_eq!(rig.world.resource::<StubStats>().promotes_confirmed, 1);
        assert!(
            saga_ack_to_orch(
                &sent,
                TransferControlAck::PromoteAck {
                    transfer: TransferId(7)
                }
            ),
            "PromoteAck is sent: {sent:?}"
        );
        // The dest read-sub is announced NOW (relocated from the adopt flip).
        assert_eq!(
            gw_replies(&sent),
            vec![ShardToGateway::SubscriptionReady {
                session: SESSION,
                entity: SUBJECT,
                frame: FrameRef::SystemSpace { system_seed: 7 },
                realm_fence: Fence(1),
            }],
            "the Promote announces exactly one SubscriptionReady"
        );
        // The source ghost-neighbor is registered (the feed pass has already advanced `seq` once this
        // tick — it runs after process_inbound), and the source ghost is Spawned (the feed started).
        assert_eq!(
            rig.world
                .resource::<GhostColliderRegistration>()
                .0
                .get(&SUBJECT)
                .map(|n| n.source),
            Some(source),
            "the source ghost-neighbor is registered"
        );
        assert!(
            flows_to(&sent, source).contains(&InterShardFlow::Ghost(GhostFlow::Spawn {
                entity: SUBJECT,
                pose: crossing_pose().sanitized(),
                source_fence: Fence(2),
                since_tick: vd_core::TickId(5),
            })),
            "the source ghost is Spawned at the crossed pose + the new fence: {sent:?}"
        );

        // Redelivery: re-ack only, NO re-flip / re-register / re-Spawn — the journal returns
        // AlreadyApplied, so `promote_apply` (which holds the flip + register + Spawn) is NOT entered.
        // `promotes_confirmed` staying 1 proves it. (The ongoing feed `Delta` to `source` continues
        // every tick — that is the running collider feed, not a re-Spawn.)
        let sent = rig.tick(vec![promote_msg(Fence(2), source)]);
        assert_eq!(rig.world.resource::<StubStats>().promotes_redelivered, 1);
        assert_eq!(
            rig.world.resource::<StubStats>().promotes_confirmed,
            1,
            "no re-flip / re-Spawn on redelivery (promote_apply gated on FirstApply)"
        );
        assert!(saga_ack_to_orch(
            &sent,
            TransferControlAck::PromoteAck {
                transfer: TransferId(7)
            }
        ));
    }

    #[test]
    fn the_saga_promote_re_owns_a_source_equals_dest_ghost_at_the_exact_cas_fence() {
        // The SOURCE==DEST re-own arm of `promote_apply` (task #149): a co-hosted-child crossing whose
        // `head(Realm(dest))` resolves to THIS node reaches `on_saga_promote` with `cmd.source == self_node`
        // (= the Rig's own SHARD id). Unlike the cross-node case (a GENESIS-fenced Ghost, strictly older than
        // the CAS fence, promoted via `AuthorityCmd::Promote`), the ordered same-node `Demote` already
        // self-fenced THIS dot to `Ghost{source_fence: cmd.new_fence}`, so a strict-newer Promote at the SAME
        // fence would `StaleFence`. The `if cmd.source == self_node` arm RE-OWNS the dot directly at that exact
        // fence — the idempotent route-swap the degenerate saga is. Same Ghost-dot fixture as the cross-node
        // test but with `source = SHARD`, asserting `Owned { fence: <the promote's new_fence> }`.
        let mut rig = Rig::new();
        rig.grant_realm();
        let _ = rig.tick(vec![open_input_slot(SESSION, GATEWAY, 5)]); // adopting dot for SUBJECT
        let _ = rig.tick(vec![adopted_head(Fence(2))]); // flip → granted Ghost
        let _ = rig.tick(vec![crossing_msg(TransferId(7), Fence(2), crossing_pose())]); // lands STUB_CROSSING_STEP
        assert!(
            !rig.world.resource::<Dots>().0[&SESSION]
                .authority
                .simulates(),
            "still Ghost after the crossing (source==dest re-own has not run yet)"
        );
        rig.set_local_tick(5);
        // `source == SHARD` (the Rig's own node id) ⇒ `cmd.source == self_node` ⇒ the direct re-own arm.
        let sent = rig.tick(vec![promote_msg(Fence(2), SHARD)]);
        assert_eq!(
            rig.world.resource::<Dots>().0[&SESSION].authority,
            Authority::Owned { fence: Fence(2) },
            "the source==dest Promote RE-OWNS the dot at the exact CAS fence (no strict-newer StaleFence)"
        );
        assert_eq!(rig.world.resource::<StubStats>().promotes_confirmed, 1);
        assert!(
            saga_ack_to_orch(
                &sent,
                TransferControlAck::PromoteAck {
                    transfer: TransferId(7)
                }
            ),
            "the source==dest Promote still acks PromoteAck: {sent:?}"
        );
        // The self-ghost tail is skipped on source==dest (no self-feed loop) — mirrors the re-home guard.
        assert_eq!(
            rig.world
                .resource::<GhostColliderRegistration>()
                .0
                .get(&SUBJECT),
            None,
            "a source==dest promote registers NO self-ghost neighbor",
        );
        assert!(
            flows_to(&sent, SHARD).is_empty(),
            "no GhostFlow::Spawn is sent to self on a source==dest promote: {sent:?}",
        );
    }

    #[test]
    fn on_re_home_creates_an_owned_dot_from_the_pose_acks_and_spawns_the_ghost() {
        // D-37 CELL 2 adopt: the FRESH target receives a `ReHome` and CREATES an Owned dot from the pose
        // (no pre-existing ghost to flip, unlike Promote). Acks PromoteAck, registers + Spawns the source
        // ghost, but emits NO SubscriptionReady (clientless until the session re-homes — D-37/D-36).
        let mut rig = Rig::new();
        rig.grant_realm();
        rig.set_local_tick(5); // pins the Spawn's since_tick deterministically
        let source = NodeId(99);
        let session = SessionId(SUBJECT.0); // the deterministic clientless session key
        let sent = rig.tick(vec![re_home_msg(
            Fence(2),
            source,
            DirectoryKey::Entity(SUBJECT),
        )]);
        assert_eq!(
            rig.world.resource::<Dots>().0[&session].authority,
            Authority::Owned { fence: Fence(2) },
            "the re-home CREATES an Owned dot born at the new fence"
        );
        assert_eq!(rig.world.resource::<Dots>().0[&session].entity, SUBJECT);
        assert_eq!(rig.world.resource::<StubStats>().re_home_adopted, 1);
        assert!(
            saga_ack_to_orch(
                &sent,
                TransferControlAck::PromoteAck {
                    transfer: TransferId(7)
                }
            ),
            "PromoteAck is sent: {sent:?}"
        );
        assert!(
            gw_replies(&sent).is_empty(),
            "a clientless re-home announces NO SubscriptionReady"
        );
        assert_eq!(
            rig.world
                .resource::<GhostColliderRegistration>()
                .0
                .get(&SUBJECT)
                .map(|n| n.source),
            Some(source),
            "the source ghost-neighbor is registered"
        );
        assert!(
            flows_to(&sent, source).contains(&InterShardFlow::Ghost(GhostFlow::Spawn {
                entity: SUBJECT,
                pose: crossing_pose().sanitized(),
                source_fence: Fence(2),
                since_tick: vd_core::TickId(5),
            })),
            "the source ghost is Spawned at the re-homed pose + the new fence: {sent:?}"
        );

        // Redelivery: re-ack only, NO re-adopt (journal AlreadyApplied ⇒ re_home_apply not entered).
        let sent = rig.tick(vec![re_home_msg(
            Fence(2),
            source,
            DirectoryKey::Entity(SUBJECT),
        )]);
        assert_eq!(rig.world.resource::<StubStats>().re_home_redelivered, 1);
        assert_eq!(
            rig.world.resource::<StubStats>().re_home_adopted,
            1,
            "no re-adopt on redelivery (re_home_apply gated on FirstApply)"
        );
        assert!(saga_ack_to_orch(
            &sent,
            TransferControlAck::PromoteAck {
                transfer: TransferId(7)
            }
        ));
    }

    #[test]
    fn a_same_node_re_home_registers_no_self_ghost_and_spawns_none() {
        // GHOST-FEED GUARD (task #149): a SOURCE==DEST re-home (the co-hosted-child crossing whose
        // `head(Realm(dest))` resolves to THIS node) reaches `re_home_apply` with `cmd.source == self_node`.
        // `register_and_spawn_source_ghost` must SKIP the self-ghost registration + the `GhostFlow::Spawn`
        // (else a self-feed loop: the owner would `Delta` its own retained copy). The dot is still adopted
        // Owned + acked; ONLY the ghost tail is skipped. `source == SHARD` (the Rig's own node id).
        let mut rig = Rig::new();
        rig.grant_realm();
        rig.set_local_tick(5);
        let session = SessionId(SUBJECT.0);
        let sent = rig.tick(vec![re_home_msg(
            Fence(2),
            SHARD, // source == this node's id (the source==dest degenerate saga)
            DirectoryKey::Entity(SUBJECT),
        )]);
        // The dot is still adopted Owned + PromoteAck'd (the re-home body ran) …
        assert_eq!(
            rig.world.resource::<Dots>().0[&session].authority,
            Authority::Owned { fence: Fence(2) },
            "a same-node re-home still adopts the dot Owned",
        );
        assert!(saga_ack_to_orch(
            &sent,
            TransferControlAck::PromoteAck {
                transfer: TransferId(7)
            }
        ));
        // … but NO self-ghost was registered and NO Spawn was emitted to self.
        assert_eq!(
            rig.world
                .resource::<GhostColliderRegistration>()
                .0
                .get(&SUBJECT),
            None,
            "a source==dest re-home registers NO self-ghost neighbor",
        );
        assert!(
            flows_to(&sent, SHARD).is_empty(),
            "no GhostFlow::Spawn is sent to self on a same-node re-home: {sent:?}",
        );
    }

    #[test]
    fn on_re_home_no_ops_for_a_non_entity_subject_or_a_target_without_its_realm() {
        // re_home_apply BAILS (still acks) on a non-Entity subject; the dispatch BAILS (no adopt) when the
        // target does not hold its realm — both counted degrade-never-panic no-ops.
        let mut rig = Rig::new();
        rig.grant_realm();
        let sent = rig.tick(vec![re_home_msg(
            Fence(2),
            NodeId(99),
            DirectoryKey::Realm(config().realm),
        )]);
        assert_eq!(rig.world.resource::<StubStats>().re_home_no_entity, 1);
        assert_eq!(rig.world.resource::<StubStats>().re_home_adopted, 0);
        assert!(
            saga_ack_to_orch(
                &sent,
                TransferControlAck::PromoteAck {
                    transfer: TransferId(7)
                }
            ),
            "a non-Entity re-home still acks (never wedges the saga)"
        );

        // A re-home delivered while the target does NOT hold its realm (no grant_realm) → no adopt.
        let mut rig2 = Rig::new();
        let _ = rig2.tick(vec![re_home_msg(
            Fence(2),
            NodeId(99),
            DirectoryKey::Entity(SUBJECT),
        )]);
        assert_eq!(rig2.world.resource::<StubStats>().re_home_without_realm, 1);
        assert_eq!(rig2.world.resource::<StubStats>().re_home_adopted, 0);
    }

    #[test]
    fn the_dest_feed_pass_streams_monotone_deltas_and_skips_unowned_registrations() {
        // 1d.5b.3b: feed_source_ghosts streams GhostFlow::Delta to each registered neighbor whose
        // entity is OWNED here, with a MONOTONE seq; a registration with no Owned dot is skipped.
        let mut rig = Rig::new();
        rig.grant_realm();
        let _ = rig.tick(vec![open_input_slot(SESSION, GATEWAY, 5)]);
        let _ = rig.tick(vec![adopted_head(Fence(2))]);
        let _ = rig.tick(vec![crossing_msg(TransferId(7), Fence(2), crossing_pose())]);
        let source = NodeId(99);
        let _ = rig.tick(vec![promote_msg(Fence(2), source)]); // Owned + registered; first feed (seq 0)

        // The next tick feeds the SECOND delta (seq 1) of the Owned dot's live pose to the source.
        rig.set_local_tick(6);
        let sent = rig.tick(vec![]);
        assert!(
            flows_to(&sent, source).contains(&InterShardFlow::Ghost(GhostFlow::Delta {
                entity: SUBJECT,
                pose: crossing_pose().sanitized(),
                source_fence: Fence(2),
                source_tick: vd_core::TickId(6),
                seq: 1,
            })),
            "the feed streams the next monotone-seq Delta of the Owned pose: {sent:?}"
        );

        // A registration whose entity is NOT owned here (no dot) is a counted no-op (no feed).
        rig.world
            .resource_mut::<GhostColliderRegistration>()
            .0
            .insert(
                EntityId(0xABCD),
                GhostNeighbor {
                    source,
                    seq: 0,
                    anchor: DVec3::ZERO,
                },
            );
        let skipped_before = rig.world.resource::<StubStats>().ghost_feed_skipped;
        let _ = rig.tick(vec![]);
        assert_eq!(
            rig.world.resource::<StubStats>().ghost_feed_skipped,
            skipped_before + 1,
            "a registration with no Owned dot is skipped"
        );
    }

    #[test]
    fn the_dest_feed_pass_skips_a_ghost_dot_registration() {
        // The simulates()-gate negative arm: a registration whose entity dot is a Ghost (not Owned)
        // is skipped — only an OWNED entity's live pose is fed.
        let mut rig = Rig::new();
        let entity = make_retained_ghost(&mut rig, Fence(2)); // a granted Ghost dot for `entity`
        rig.world
            .resource_mut::<GhostColliderRegistration>()
            .0
            .insert(
                entity,
                GhostNeighbor {
                    source: NodeId(88),
                    seq: 0,
                    anchor: DVec3::ZERO,
                },
            );
        let sent = rig.tick(vec![]);
        assert!(
            flows_to(&sent, NodeId(88)).is_empty(),
            "no feed for a Ghost (non-Owned) dot: {sent:?}"
        );
        assert!(rig.world.resource::<StubStats>().ghost_feed_skipped >= 1);
    }

    #[test]
    fn the_source_ghost_consumer_refreshes_and_dedups_the_feed() {
        // 1d.5b.3b: the SOURCE ghost-host consumes the dest-driven feed — Spawn establishes it +
        // refreshes the retained ghost's pose; a fresh Delta refreshes; a stale-seq / no-mirror /
        // stale-fence Delta is dropped; Despawn clears the latch; a malformed body is undecodable.
        let mut rig = Rig::new();
        let entity = make_retained_ghost(&mut rig, Fence(2)); // retained source Ghost{Fence(2)}
        let pose_a = crossing_pose();
        let pose_b = StampedPose::at_rest(
            FrameRef::SystemSpace { system_seed: 7 },
            DVec3::new(9.0, 8.0, 7.0),
            UniverseTick(3),
        );

        // Spawn: establish the mirror (fed) + refresh the hosted ghost's pose at the owner fence.
        let _ = rig.tick(vec![ghost_lifecycle(GhostFlow::Spawn {
            entity,
            pose: pose_a,
            source_fence: Fence(2),
            since_tick: vd_core::TickId(0),
        })]);
        assert_eq!(
            rig.world.resource::<Dots>().0[&SESSION].pose,
            pose_a.sanitized(),
            "Spawn refreshes the hosted ghost pose"
        );
        assert!(
            rig.world
                .resource::<SourceGhostMirror>()
                .0
                .get(&entity)
                .is_some_and(|s| s.fed),
            "the feed is established (fed)"
        );

        // A fresh Delta (seq 1 > 0, fence 2 >= 2) applies the new pose.
        let _ = rig.tick(vec![ghost_delta(entity, pose_b, Fence(2), 1)]);
        assert_eq!(
            rig.world.resource::<Dots>().0[&SESSION].pose,
            pose_b.sanitized(),
            "a fresh Delta refreshes the pose"
        );
        assert_eq!(rig.world.resource::<StubStats>().ghost_delta_applied, 1);

        // A stale-seq Delta (seq 1 <= last_seq 1) is dropped — the pose is unchanged.
        let _ = rig.tick(vec![ghost_delta(entity, pose_a, Fence(2), 1)]);
        assert_eq!(
            rig.world.resource::<Dots>().0[&SESSION].pose,
            pose_b.sanitized(),
            "a stale-seq Delta is dropped"
        );
        assert_eq!(rig.world.resource::<StubStats>().ghost_delta_stale, 1);

        // A fresh-seq Delta with a STALE fence (1 < the ghost's 2) is REFUSED by GhostRefresh.
        let _ = rig.tick(vec![ghost_delta(entity, pose_a, Fence(1), 2)]);
        assert_eq!(
            rig.world.resource::<Dots>().0[&SESSION].pose,
            pose_b.sanitized(),
            "a stale-fence Delta is refused (pose unchanged)"
        );
        assert_eq!(rig.world.resource::<StubStats>().ghost_refresh_stale, 1);

        // A Delta for an UNregistered entity (no mirror) is dropped.
        let _ = rig.tick(vec![ghost_delta(EntityId(0xCAFE), pose_a, Fence(2), 9)]);
        assert_eq!(
            rig.world.resource::<StubStats>().ghost_delta_stale,
            2,
            "a no-mirror Delta is dropped"
        );

        // Despawn TEARS DOWN the hosted ghost (1d.5b.3c): the mirror entry AND the retained ghost dot
        // are removed — the ghost lifecycle ENDS (the source stops self-emitting + being a collider).
        let _ = rig.tick(vec![ghost_lifecycle(GhostFlow::Despawn {
            entity,
            source_fence: Fence(2),
        })]);
        assert!(
            !rig.world
                .resource::<SourceGhostMirror>()
                .0
                .contains_key(&entity),
            "Despawn removes the mirror entry"
        );
        // The retained ghost dot (keyed by SESSION via `make_retained_ghost`) is GONE. A direct key
        // check, NOT `.values().any(|d| ...)`: after teardown the map is empty, so an `any` closure
        // would never run (an uncoverable region) — assert absence by the key that was removed.
        assert!(
            !rig.world.resource::<Dots>().0.contains_key(&SESSION),
            "Despawn removes the retained ghost dot (the lifecycle ends)"
        );
        assert_eq!(rig.world.resource::<StubStats>().ghost_despawns, 1);

        // A Spawn for an entity this shard does NOT host (no dot) records the mirror but refreshes
        // nothing — the no-dot arm of the Spawn handler.
        let _ = rig.tick(vec![ghost_lifecycle(GhostFlow::Spawn {
            entity: EntityId(0xDEAD),
            pose: pose_a,
            source_fence: Fence(2),
            since_tick: vd_core::TickId(0),
        })]);
        assert!(
            rig.world
                .resource::<SourceGhostMirror>()
                .0
                .contains_key(&EntityId(0xDEAD)),
            "a Spawn with no hosted dot still records the mirror"
        );

        // A Despawn for an entity with NO mirror AND no hosted dot tears down nothing — a counted
        // idempotent no-op (`ghost_despawn_no_host`), never a panic, and never bumps the teardown count.
        let no_host_before = rig.world.resource::<StubStats>().ghost_despawn_no_host;
        let despawns_before = rig.world.resource::<StubStats>().ghost_despawns;
        let _ = rig.tick(vec![ghost_lifecycle(GhostFlow::Despawn {
            entity: EntityId(0x12345),
            source_fence: Fence(2),
        })]);
        assert_eq!(
            rig.world.resource::<StubStats>().ghost_despawn_no_host,
            no_host_before + 1,
            "a Despawn for an unhosted entity is a counted no-op"
        );
        assert_eq!(
            rig.world.resource::<StubStats>().ghost_despawns,
            despawns_before,
            "...and does NOT bump the teardown counter"
        );

        // A malformed ghost body is counted undecodable, never mis-applied.
        let undec_before = rig.world.resource::<StubStats>().undecodable;
        let _ = rig.tick(vec![Inbound::Wire {
            from: DEST_OWNER,
            class: MsgClass::GhostDelta,
            bytes: crate::io::bytes(vec![0xFF, 0xFF, 0xFF]),
        }]);
        assert_eq!(
            rig.world.resource::<StubStats>().undecodable,
            undec_before + 1,
            "a malformed ghost body is counted undecodable"
        );
    }

    #[test]
    fn the_dest_feed_despawns_on_band_exit_and_deregisters() {
        // 1d.5b.3c: the dest (owner) drives the source-ghost lifecycle END. While the owned entity is
        // IN the overlap band (anchored at its crossing) the feed streams Delta; once it walks PAST the
        // band's destroy edge the dest emits GhostFlow::Despawn (reliable) + DEREGISTERS the feed.
        let mut rig = Rig::new();
        rig.grant_realm();
        let _ = rig.tick(vec![open_input_slot(SESSION, GATEWAY, 5)]);
        let _ = rig.tick(vec![adopted_head(Fence(2))]);
        let _ = rig.tick(vec![crossing_msg(TransferId(7), Fence(2), crossing_pose())]);
        let source = NodeId(99);
        // Owned + registered; the anchor is the crossed pose position (the boundary it entered through).
        let _ = rig.tick(vec![promote_msg(Fence(2), source)]);

        // IN-BAND (the dot is at the anchor, distance 0): the feed streams the next monotone Delta and
        // KEEPS the registration — the `else` (feed) arm of the band-exit decision.
        rig.set_local_tick(6);
        let sent = rig.tick(vec![]);
        assert!(
            flows_to(&sent, source).contains(&InterShardFlow::Ghost(GhostFlow::Delta {
                entity: SUBJECT,
                pose: crossing_pose().sanitized(),
                source_fence: Fence(2),
                source_tick: vd_core::TickId(6),
                seq: 1,
            })),
            "in-band: the feed streams a Delta (not a Despawn): {sent:?}"
        );
        assert!(
            rig.world
                .resource::<GhostColliderRegistration>()
                .0
                .contains_key(&SUBJECT),
            "in-band: the registration is kept"
        );
        let exits_before = rig.world.resource::<StubStats>().ghost_band_exits;

        // BAND-EXIT: move the owned dot well past the destroy edge from the crossing anchor. The band
        // is `for_motion(move_speed*dt)` = `for_motion(0.1)`, destroy_above = 20*0.1 = 2.0 m; +3 m exits.
        let exit_pos = crossing_pose().pos.offset() + DVec3::new(3.0, 0.0, 0.0);
        rig.world
            .resource_mut::<Dots>()
            .0
            .get_mut(&SESSION)
            .expect("the owned dot")
            .pose
            .pos = LatticePos::local(exit_pos);
        let sent = rig.tick(vec![]);
        assert!(
            flows_to(&sent, source).contains(&InterShardFlow::Ghost(GhostFlow::Despawn {
                entity: SUBJECT,
                source_fence: Fence(2),
            })),
            "band-exit: the dest Despawns the ghost on the reliable carrier: {sent:?}"
        );
        assert!(
            !rig.world
                .resource::<GhostColliderRegistration>()
                .0
                .contains_key(&SUBJECT),
            "band-exit: the feed is deregistered"
        );
        assert_eq!(
            rig.world.resource::<StubStats>().ghost_band_exits,
            exits_before + 1
        );

        // ...and the feed truly STOPS: a further tick sends nothing to the source (no Delta, no re-Despawn).
        let sent = rig.tick(vec![]);
        assert!(
            flows_to(&sent, source).is_empty(),
            "after deregistration the feed is silent: {sent:?}"
        );
    }

    #[test]
    fn a_band_exit_despawn_refuses_to_remove_a_reowned_owned_dot() {
        // 1d.5b.3c structural refusal — the shard-LOCAL stand-in for the orchestrator in-transfer gate
        // (a `vd-sim` shard cannot see the live-saga set). A stale Despawn for an entity the source has
        // since RE-OWNED removes nothing: only a retained Ghost is torn down, never a live `Owned` dot.
        // Covers `remove_retained_ghost`'s `matches!(Ghost{..})` FALSE arm + the `no_host` counter.
        let mut rig = Rig::new();
        rig.grant_realm();
        let _ = rig.attach(); // SESSION's dot is granted + Owned (a re-acquisition would land here)
        let entity = rig.world.resource::<Dots>().0[&SESSION].entity;
        assert!(
            rig.world.resource::<Dots>().0[&SESSION]
                .authority
                .simulates(),
            "precondition: the dot is Owned"
        );

        let despawns_before = rig.world.resource::<StubStats>().ghost_despawns;
        let _ = rig.tick(vec![ghost_lifecycle(GhostFlow::Despawn {
            entity,
            source_fence: Fence(1),
        })]);
        assert!(
            rig.world.resource::<Dots>().0[&SESSION]
                .authority
                .simulates(),
            "the re-owned Owned dot is structurally refused (kept, still simulating)"
        );
        assert_eq!(
            rig.world.resource::<StubStats>().ghost_despawn_no_host,
            1,
            "the stale Despawn is a counted no-op"
        );
        assert_eq!(
            rig.world.resource::<StubStats>().ghost_despawns,
            despawns_before,
            "...and tears nothing down"
        );
    }

    #[test]
    fn a_retained_ghost_self_emits_but_an_unfed_genesis_ghost_does_not() {
        // 1d.5b.3b emit-widening: a RETAINED source Ghost (post-GENESIS fence) SELF-EMITS its
        // last-Owned pose (the demote→Promote fill); a pre-promote DEST-adopt Ghost (GENESIS, no real
        // pose) emits NOTHING.
        // (a) retained source ghost → emits.
        let mut rig = Rig::new();
        let _entity = make_retained_ghost(&mut rig, Fence(2));
        let sent = rig.tick(vec![]);
        assert_eq!(
            decode_frames(&sent).len(),
            1,
            "the retained source Ghost self-emits its last-Owned pose (no vanish): {sent:?}"
        );

        // (b) a pre-promote DEST-adopt Ghost (GENESIS) → emits nothing.
        let mut rig = Rig::new();
        rig.grant_realm();
        let _ = rig.tick(vec![open_input_slot(SESSION, GATEWAY, 5)]);
        let sent = rig.tick(vec![adopted_head(Fence(2))]); // granted Ghost{GENESIS}, no pose
        assert_eq!(
            decode_frames(&sent).len(),
            0,
            "an unfed GENESIS dest-adopt Ghost renders nothing until the Promote"
        );
    }

    #[test]
    fn emit_frames_renders_an_emitting_dot_and_filters_a_silent_one_in_the_same_tick() {
        // Covers the entity-collect filter's FALSE arm: a tick with BOTH a retained source Ghost
        // (emits) AND a pre-grant GENESIS adopt Ghost (silent). emit_frames does NOT early-return
        // (the emitter makes gateways non-empty) AND the entity-collect filter EXCLUDES the silent
        // dot — exactly the emitting one renders.
        let mut rig = Rig::new();
        let emitter = make_retained_ghost(&mut rig, Fence(2)); // SESSION: retained Ghost, emits
        let _ = rig.tick(vec![open_input_slot(SessionId(0xBB), GATEWAY, 5)]); // a pre-grant GENESIS adopt Ghost, silent
        let sent = rig.tick(vec![]);
        let entities: Vec<EntityId> = decode_frames(&sent)
            .into_iter()
            .flat_map(|f| f.entities)
            .map(|e| e.entity)
            .collect();
        assert_eq!(
            entities,
            vec![emitter],
            "only the emitting retained Ghost renders; the silent GENESIS adopt Ghost is filtered out"
        );
    }

    fn orbit() -> OrbitalElements {
        OrbitalElements {
            sma: 1.5e11,
            ecc: 0.1,
            inclination: 0.4,
            raan: 0.3,
            arg_periapsis: 0.9,
            mean_anomaly_epoch: 0.2,
            central_mass: 1.989e30,
        }
    }

    #[test]
    fn authored_realm_snaps_computes_a_moving_child_pose_in_the_own_frame_and_is_empty_when_static()
    {
        // FA-2c: a moving child's authored `RealmSnap` is its ephemeris pose (position + velocity) at the
        // tick, stamped in the shard's OWN (ambient-root) frame; a forest with NO moving child yields none.
        let elements = orbit();
        let mut moving = BTreeMap::new();
        moving.insert(OTHER_REALM, elements);
        let regions = RealmRegions::new(vec![root_region(), own_region(), child_region()])
            .with_moving_children(moving);
        let (tick_hz, tick) = (20.0, UniverseTick(1_000));
        let snaps = regions.authored_realm_snaps(OWN_REALM, tick_hz, tick);
        let state = orbital_state(&elements, secs_since_epoch(tick.0, tick_hz));
        assert_eq!(snaps.len(), 1);
        assert_eq!(snaps[0].realm, OTHER_REALM);
        assert_eq!(
            snaps[0].pose.frame,
            frame_of(ROOT_REALM),
            "authored in the own frame"
        );
        assert_eq!(snaps[0].pose.pos.offset(), state.position);
        assert_eq!(snaps[0].pose.vel, state.velocity);
        assert_eq!(snaps[0].pose.universe_tick, tick);
        // A static forest (no moving roster) authors NO realm snap — the byte-identity case.
        assert!(
            RealmRegions::new(vec![root_region(), own_region()])
                .authored_realm_snaps(OWN_REALM, tick_hz, tick)
                .is_empty()
        );
    }

    #[test]
    fn emit_realm_frames_ships_a_moving_child_only_to_a_present_observer_with_authority() {
        // FA-2c: the shard ships a moving child's box to observers ONLY when it (a) holds its realm, (b)
        // authors >=1 moving child, and (c) has an emitting observer — the emit_frames gating, verbatim.
        // Covers all four arms of emit_realm_frames.
        let plant = |rig: &mut Rig| {
            let mut moving = BTreeMap::new();
            moving.insert(OTHER_REALM, orbit());
            *rig.world.resource_mut::<RealmRegions>() =
                RealmRegions::new(vec![root_region(), own_region(), child_region()])
                    .with_moving_children(moving);
        };
        // A dot INSIDE own but OUTSIDE the child (container == owning ⇒ it emits, never re-homes).
        let observer = |rig: &mut Rig, tag: u32| {
            insert_owned_dot(
                rig,
                TRIG_SESSION,
                EntityId::pack(EntityKind::Player, 10, 1, tag),
                DVec3::new(5_000.0, 0.0, 0.0),
            );
        };
        let realms = |sent: &[(NodeId, MsgClass, Vec<u8>)]| -> Vec<RealmId> {
            decode_realm_frames(sent)
                .into_iter()
                .flat_map(|f| f.realms)
                .map(|s| s.realm)
                .collect()
        };

        // (a) HAPPY: granted + moving child + emitting observer ⇒ the moving realm ships.
        let mut rig = Rig::new();
        rig.grant_realm();
        plant(&mut rig);
        observer(&mut rig, 5);
        assert_eq!(realms(&rig.tick(vec![])), vec![OTHER_REALM]);

        // (b) NO OBSERVER: granted + moving child but no emitting dot ⇒ silent (gateways empty).
        let mut rig = Rig::new();
        rig.grant_realm();
        plant(&mut rig);
        assert!(realms(&rig.tick(vec![])).is_empty());

        // (c) STATIC SCALE: granted + observer but EMPTY roster ⇒ silent (the byte-identity arm).
        let mut rig = Rig::new();
        rig.grant_realm();
        observer(&mut rig, 6);
        assert!(realms(&rig.tick(vec![])).is_empty());

        // (d) NO AUTHORITY: an ungranted shard is silent even with a moving child + observer.
        let mut rig = Rig::new();
        plant(&mut rig);
        observer(&mut rig, 7);
        assert!(realms(&rig.tick(vec![])).is_empty());
    }

    #[test]
    fn the_saga_promote_on_an_unheld_or_non_entity_subject_is_a_counted_noop_but_acks() {
        // on_saga_promote's no-dot arms: a Promote for an entity not held here, and for a non-Entity
        // (Realm) subject, each flip nothing (counted `promote_no_dot`) but STILL ack PromoteAck.
        let mut rig = Rig::new();
        rig.grant_realm();
        // (a) unheld Entity subject.
        let sent = rig.tick(vec![promote_msg(Fence(2), NodeId(99))]); // SUBJECT not held here
        assert_eq!(rig.world.resource::<StubStats>().promote_no_dot, 1);
        assert!(saga_ack_to_orch(
            &sent,
            TransferControlAck::PromoteAck {
                transfer: TransferId(7)
            }
        ));
        // (b) non-Entity (Realm) subject.
        let realm_promote = InterShardFlow::Promote(PromoteCmd {
            transfer: TransferId(8),
            subject: DirectoryKey::Realm(RealmId::System(9)),
            new_fence: Fence(2),
            step_id: PROMOTE_STEP,
            source: NodeId(99),
        });
        let sent = rig.tick(vec![wire_msg(ORCH, MsgClass::Saga, &realm_promote)]);
        assert_eq!(rig.world.resource::<StubStats>().promote_no_dot, 2);
        assert!(saga_ack_to_orch(
            &sent,
            TransferControlAck::PromoteAck {
                transfer: TransferId(8)
            }
        ));
    }

    #[test]
    fn the_saga_promote_before_the_crossing_lands_defers_the_flip_but_acks() {
        // pose-before-promote: a Promote arriving BEFORE the crossing journaled does NOT flip (no
        // poseless origin frame); counted `promote_before_crossing`, still acks. The dot stays Ghost.
        let mut rig = Rig::new();
        rig.grant_realm();
        let _ = rig.tick(vec![open_input_slot(SESSION, GATEWAY, 5)]);
        let _ = rig.tick(vec![adopted_head(Fence(2))]); // granted Ghost, NO crossing yet
        let sent = rig.tick(vec![promote_msg(Fence(2), NodeId(99))]);
        assert!(
            !rig.world.resource::<Dots>().0[&SESSION]
                .authority
                .simulates(),
            "the Promote does NOT flip a dot whose crossing pose has not landed (stays Ghost)"
        );
        assert_eq!(rig.world.resource::<StubStats>().promote_before_crossing, 1);
        assert!(saga_ack_to_orch(
            &sent,
            TransferControlAck::PromoteAck {
                transfer: TransferId(7)
            }
        ));
    }

    #[test]
    fn the_saga_promote_without_a_realm_is_a_counted_noop_never_a_panic() {
        // The realm-owner guard's None arm (1d.5b.3b audit hardening): a Promote arriving while this
        // shard does NOT hold its realm is DROPPED as a counted no-op (DEGRADE, never panic), no ack
        // — the saga re-drives. Mirrors every sibling handler's degrade-not-crash discipline.
        let mut rig = Rig::new(); // NO grant_realm → the shard holds no realm (authority.0 == None)
        let sent = rig.tick(vec![promote_msg(Fence(2), NodeId(99))]);
        assert_eq!(rig.world.resource::<StubStats>().promote_without_realm, 1);
        assert!(
            !saga_ack_to_orch(
                &sent,
                TransferControlAck::PromoteAck {
                    transfer: TransferId(7)
                }
            ),
            "a realm-less Promote is dropped (no ack) — the saga re-drives: {sent:?}"
        );
    }

    #[test]
    fn the_dest_applies_post_marker_input_and_rejects_replays_at_the_marker() {
        let mut rig = Rig::new();
        rig.grant_realm();
        let _ = rig.tick(vec![open_input_slot(SESSION, GATEWAY, 10)]); // watermark = 10
        // marker+1, marker+2 apply in order; a seq <= marker is a counted DuplicateSeq.
        let _ = rig.tick(vec![
            input_for(SESSION, 11, GATEWAY),
            input_for(SESSION, 12, GATEWAY),
            input_for(SESSION, 9, GATEWAY),
        ]);
        let log = rig.world.resource::<InputLog>();
        assert_eq!(log.applied(), vec![(SESSION, 11), (SESSION, 12)]);
        assert_eq!(
            log.discarded(),
            vec![(SESSION, Some(9), DiscardReason::DuplicateSeq)],
            "a seq <= marker was already applied at the source"
        );
    }

    #[test]
    fn open_input_slot_before_the_realm_lease_is_deferred_and_counted() {
        let mut rig = Rig::new(); // NO grant_realm
        let _ = rig.tick(vec![open_input_slot(SESSION, GATEWAY, 5)]);
        assert!(
            !rig.world.resource::<Dots>().0.contains_key(&SESSION),
            "no slot yet"
        );
        assert_eq!(rig.world.resource::<StubStats>().input_slots_deferred, 1);
    }

    #[test]
    fn open_input_slot_max_merges_the_watermark_and_guards_a_granted_dot() {
        let mut rig = Rig::new();
        rig.grant_realm();
        let _ = rig.tick(vec![open_input_slot(SESSION, GATEWAY, 20)]);
        // A re-sent slot with a LOWER watermark never lowers it (max-merge).
        let _ = rig.tick(vec![open_input_slot(SESSION, GATEWAY, 5)]);
        assert_eq!(
            rig.world.resource::<Dots>().0[&SESSION].last_applied_seq,
            Some(20)
        );
        // A slot from a NON-owning gateway does not touch the dot (security guard false arm).
        let other_gateway = NodeId(999);
        let _ = rig.tick(vec![open_input_slot(SESSION, other_gateway, 100)]);
        let dot = rig.world.resource::<Dots>().0[&SESSION];
        assert_eq!(
            dot.last_applied_seq,
            Some(20),
            "a foreign gateway cannot move the watermark"
        );
        assert_eq!(dot.gateway, GATEWAY, "ownership unchanged");
    }

    #[test]
    fn open_input_slot_bumps_the_session_fence_then_is_inert_on_a_granted_dot() {
        let mut rig = Rig::new();
        rig.grant_realm();
        // Mint the provisional slot at fence 1, watermark 5.
        let _ = rig.tick(vec![open_input_slot_f(SESSION, GATEWAY, 5, Fence(1))]);
        // A slot at a HIGHER fence (as 1d/P3 will re-issue under a fresher realm lease) bumps
        // the session fence and MAX-merges the watermark.
        let _ = rig.tick(vec![open_input_slot_f(SESSION, GATEWAY, 7, Fence(4))]);
        {
            let dot = rig.world.resource::<Dots>().0[&SESSION];
            assert_eq!(dot.session_fence, Fence(4), "bumped to the fresher lease");
            assert_eq!(dot.last_applied_seq, Some(7), "watermark advanced");
        }
        // A STALE slot (fence below the dot's session fence — a replay or partitioned old
        // gateway) is dropped + counted, never re-arming input (the day-one stale-gateway rule).
        let _ = rig.tick(vec![open_input_slot_f(SESSION, GATEWAY, 999, Fence(1))]);
        {
            let dot = rig.world.resource::<Dots>().0[&SESSION];
            assert_eq!(
                dot.session_fence,
                Fence(4),
                "stale slot does not touch the fence"
            );
            assert_eq!(
                dot.last_applied_seq,
                Some(7),
                "stale slot does not move the watermark"
            );
            assert_eq!(rig.world.resource::<StubStats>().input_slots_stale, 1);
        }
        // Once the dot is GRANTED (the 1d promotion), a stray OpenInputSlot is inert —
        // the granted entity owns its own input watermark (the `!granted` guard false arm).
        rig.world
            .resource_mut::<Dots>()
            .0
            .get_mut(&SESSION)
            .expect("the provisional dot was minted above")
            .granted = true;
        let _ = rig.tick(vec![open_input_slot_f(SESSION, GATEWAY, 999, Fence(9))]);
        let dot = rig.world.resource::<Dots>().0[&SESSION];
        assert_eq!(dot.session_fence, Fence(4), "granted dot's fence untouched");
        assert_eq!(
            dot.last_applied_seq,
            Some(7),
            "granted dot's watermark untouched"
        );
    }

    // ---- Slice 1d.1: the pose-only entity-state crossing (source flush + dest adopt) ---------

    const FROM_REALM: RealmId = RealmId::System(7);
    const TO_REALM: RealmId = RealmId::System(8);

    /// A non-origin crossing pose (so a test can tell an applied crossing from the origin-adopt).
    fn crossing_pose() -> StampedPose {
        StampedPose::at_rest(
            FrameRef::SystemSpace { system_seed: 8 },
            DVec3::new(4.0, -5.0, 6.0),
            UniverseTick(200),
        )
    }

    /// The directory head that flips the adopted dot to `granted` at `fence` (dest-owned record).
    fn adopted_head(fence: Fence) -> Inbound {
        wire_msg(
            ORCH,
            MsgClass::Saga,
            &InterShardFlow::DirectoryReply(DirectoryReply::Head {
                key: DirectoryKey::Entity(SUBJECT),
                record: Some(vd_wire::seams::directory::OwnerRecord {
                    authority: AuthorityRef::Shard(SHARD),
                    fence,
                    lease_expires: UniverseTick(1_000),
                    in_transfer: None,
                }),
            }),
        )
    }

    /// A `StubCrossing` envelope for `SUBJECT` at `fence` carrying `pose`.
    fn crossing_msg(transfer: TransferId, fence: Fence, pose: StampedPose) -> Inbound {
        wire_msg(
            ORCH,
            MsgClass::Saga,
            &InterShardFlow::Transfer(TransferEnvelope {
                transfer_id: transfer,
                universe_epoch: vd_core::EpochId(1),
                schema_version: vd_wire::intershard::TRANSFER_SCHEMA_VERSION,
                fence,
                step_id: STUB_CROSSING_STEP,
                class: vd_core::entity_kind::DurabilityClass::Durable,
                payload: TransitionPayload::StubCrossing {
                    entity: SUBJECT,
                    from_realm: FROM_REALM,
                    to_realm: TO_REALM,
                    pose,
                    state: vec![],
                },
            }),
        )
    }

    fn flush_msg(entity: EntityId) -> Inbound {
        wire_msg(
            ORCH,
            MsgClass::Saga,
            &InterShardFlow::FlushSource(FlushSource {
                transfer: TransferId(7),
                subject: DirectoryKey::Entity(entity),
                step_id: FLUSH_SOURCE_STEP,
            }),
        )
    }

    /// The flows the shard sent to the orchestrator this tick.
    fn to_orch(sent: &[(NodeId, MsgClass, Vec<u8>)]) -> Vec<InterShardFlow> {
        sent.iter()
            .filter(|(to, _, _)| *to == ORCH)
            .map(|(_, _, b)| postcard::from_bytes(b).expect("flow decodes"))
            .collect()
    }

    /// Whether the shard acked a crossing step (value-compare, never `matches!` — HR5).
    fn acked(sent: &[(NodeId, MsgClass, Vec<u8>)], transfer: TransferId) -> bool {
        to_orch(sent).contains(&InterShardFlow::TransferAck(TransferAck::Accepted {
            transfer_id: transfer,
            step_id: STUB_CROSSING_STEP,
        }))
    }

    /// A saga `Promote` for `SUBJECT` at `new_fence`, naming `source` as the ghost-host (1d.5b.3b).
    fn promote_msg(new_fence: Fence, source: NodeId) -> Inbound {
        wire_msg(
            ORCH,
            MsgClass::Saga,
            &InterShardFlow::Promote(PromoteCmd {
                transfer: TransferId(7),
                subject: DirectoryKey::Entity(SUBJECT),
                new_fence,
                step_id: PROMOTE_STEP,
                source,
            }),
        )
    }

    fn re_home_msg(new_fence: Fence, source: NodeId, subject: DirectoryKey) -> Inbound {
        wire_msg(
            ORCH,
            MsgClass::Saga,
            &InterShardFlow::ReHome(ReHomeCmd {
                transfer: TransferId(7),
                universe_epoch: vd_core::EpochId(1),
                subject,
                new_fence,
                step_id: RE_HOME_STEP,
                state: ReHomeState::PoseOnly(crossing_pose()),
                source,
            }),
        )
    }

    /// The `ShardToGateway` Control replies this shard sent to `GATEWAY` this tick (value-compare).
    fn gw_replies(sent: &[(NodeId, MsgClass, Vec<u8>)]) -> Vec<ShardToGateway> {
        sent.iter()
            .filter(|(to, class, _)| (*to == GATEWAY) & (*class == MsgClass::Control))
            .map(|(_, _, bytes)| postcard::from_bytes(bytes).expect("decode"))
            .collect()
    }

    /// The `InterShardFlow`s this shard sent to `target` this tick (value-compare; `.ok()` drops the
    /// rare non-flow / decode failure, of which there are none on a ghost/orchestrator peer).
    fn flows_to(sent: &[(NodeId, MsgClass, Vec<u8>)], target: NodeId) -> Vec<InterShardFlow> {
        sent.iter()
            .filter(|(to, _, _)| *to == target)
            .filter_map(|(_, _, bytes)| postcard::from_bytes::<InterShardFlow>(bytes).ok())
            .collect()
    }

    /// One `GhostFlow::Delta` as the SOURCE ghost-host receives it (for the consumer tests).
    fn ghost_delta(entity: EntityId, pose: StampedPose, source_fence: Fence, seq: u64) -> Inbound {
        Inbound::Wire {
            from: DEST_OWNER,
            class: MsgClass::GhostDelta,
            bytes: crate::io::bytes(
                postcard::to_allocvec(&InterShardFlow::Ghost(GhostFlow::Delta {
                    entity,
                    pose,
                    source_fence,
                    source_tick: vd_core::TickId(0),
                    seq,
                }))
                .expect("encode"),
            ),
        }
    }

    /// One `GhostFlow::Spawn` / `Despawn` as the SOURCE ghost-host receives it (reliable carrier).
    fn ghost_lifecycle(flow: GhostFlow) -> Inbound {
        Inbound::Wire {
            from: DEST_OWNER,
            class: MsgClass::GhostReliable,
            bytes: crate::io::bytes(
                postcard::to_allocvec(&InterShardFlow::Ghost(flow)).expect("encode"),
            ),
        }
    }

    /// Demote the SESSION dot to a RETAINED source Ghost at `new_owner_fence` (the consumer/emit
    /// fixture): a granted Owned dot → the saga `Demote` → `Ghost`. Returns the dot's entity.
    fn make_retained_ghost(rig: &mut Rig, new_owner_fence: Fence) -> EntityId {
        rig.grant_realm();
        let _ = rig.attach();
        let entity = rig.world.resource::<Dots>().0[&SESSION].entity;
        let demote = InterShardFlow::Demote(DemoteCmd {
            transfer: TransferId(7),
            subject: DirectoryKey::Entity(entity),
            new_owner_fence,
            step_id: vd_wire::intershard::DEMOTE_STEP,
        });
        let _ = rig.tick(vec![wire_msg(ORCH, MsgClass::Saga, &demote)]);
        entity
    }

    /// The DEST owner node that feeds this shard's hosted ghosts in the consumer tests.
    const DEST_OWNER: NodeId = NodeId(99);

    #[test]
    fn flush_source_ships_the_held_dots_pose() {
        let mut rig = Rig::new();
        rig.grant_realm();
        let _ = rig.attach(); // a granted login dot for SESSION
        let entity = rig.world.resource::<Dots>().0[&SESSION].entity;

        // (a) BEFORE any input: the dot is at origin with NO applied seq → drained_seq defaults 0.
        let dot0 = rig.world.resource::<Dots>().0[&SESSION];
        let sent0 = rig.tick(vec![flush_msg(entity)]);
        assert_eq!(
            to_orch(&sent0),
            vec![InterShardFlow::TransferAck(TransferAck::SourceFlushed {
                transfer_id: TransferId(7),
                step_id: FLUSH_SOURCE_STEP,
                pose: dot0.pose,
                drained_seq: 0,
            })],
            "ships the held dot's pose; an unset watermark defaults to 0"
        );

        // (b) AFTER applying seq 1: the pose moved and the watermark is Some(1).
        let _ = rig.tick(vec![input_for(SESSION, 1, GATEWAY)]);
        let dot1 = rig.world.resource::<Dots>().0[&SESSION];
        assert_ne!(
            dot1.pose.pos.offset(),
            DVec3::ZERO,
            "the dot moved on input"
        );
        let sent1 = rig.tick(vec![flush_msg(entity)]);
        assert_eq!(
            to_orch(&sent1),
            vec![InterShardFlow::TransferAck(TransferAck::SourceFlushed {
                transfer_id: TransferId(7),
                step_id: FLUSH_SOURCE_STEP,
                pose: dot1.pose,
                drained_seq: 1,
            })],
            "ships the moved pose + the real drain watermark"
        );
    }

    #[test]
    fn flush_source_for_an_unheld_or_non_entity_subject_ships_nothing() {
        let mut rig = Rig::new();
        rig.grant_realm();
        let _ = rig.attach();
        // An entity this shard does not hold → no ship (counted no-op).
        let sent = rig.tick(vec![flush_msg(EntityId(0xDEAD))]);
        assert!(to_orch(&sent).is_empty(), "no pose for an unheld entity");
        // A non-Entity subject → no ship.
        let sent = rig.tick(vec![wire_msg(
            ORCH,
            MsgClass::Saga,
            &InterShardFlow::FlushSource(FlushSource {
                transfer: TransferId(7),
                subject: DirectoryKey::Realm(RealmId::System(9)),
                step_id: FLUSH_SOURCE_STEP,
            }),
        )]);
        assert!(
            to_orch(&sent).is_empty(),
            "no pose for a non-Entity subject"
        );
    }

    #[test]
    fn a_crossing_to_an_adopted_dot_stores_the_pose_stays_ghost_then_promote_flips_owned() {
        let mut rig = Rig::new();
        rig.grant_realm();
        let _ = rig.tick(vec![open_input_slot(SESSION, GATEWAY, 5)]); // adopting dot
        let _ = rig.tick(vec![adopted_head(Fence(2))]); // flip → granted, still a Ghost (not simulating)
        let pose = crossing_pose();

        // 1d.5b.3b: the crossing STORES the pose but leaves the dot a GHOST — the Ghost→Owned promote
        // RELOCATED to on_saga_promote (strict demote-before-promote). No autonomous flip here.
        let sent = rig.tick(vec![crossing_msg(TransferId(7), Fence(2), pose)]);
        let dot = rig.world.resource::<Dots>().0[&SESSION];
        assert_eq!(dot.pose, pose.sanitized(), "the crossed pose stored");
        assert!(
            !dot.authority.simulates(),
            "the dot STAYS Ghost after the crossing (the autonomous promote is gone)"
        );
        assert_eq!(rig.world.resource::<StubStats>().crossings_applied, 1);
        assert!(acked(&sent, TransferId(7)), "the crossing step is acked");

        // The saga Promote flips it Ghost→Owned (pose-before-promote satisfied — the crossing landed).
        let _ = rig.tick(vec![promote_msg(Fence(2), NodeId(99))]);
        assert_eq!(
            rig.world.resource::<Dots>().0[&SESSION].authority,
            Authority::Owned { fence: Fence(2) },
            "the Promote flips the dest Ghost→Owned at the recorded CAS fence"
        );

        // A crossing redelivery AFTER the flip still matches `crossing_target` (now Owned, still
        // granted+non-departing) → the journal returns `AlreadyApplied`: re-ack WITHOUT re-applying,
        // NOT re-buffered (no strand).
        let buffered_before = rig.world.resource::<StubStats>().crossings_buffered;
        let sent = rig.tick(vec![crossing_msg(TransferId(7), Fence(2), pose)]);
        assert_eq!(
            rig.world.resource::<StubStats>().crossings_applied,
            1,
            "no re-apply on a post-flip redelivery"
        );
        assert_eq!(
            rig.world.resource::<StubStats>().crossings_buffered,
            buffered_before,
            "a post-flip redelivery re-acks via the journal — it is NOT re-buffered (no strand)"
        );
        assert!(
            acked(&sent, TransferId(7)),
            "a post-flip redelivery still re-acks"
        );
    }

    #[test]
    fn a_crossing_before_adopt_is_buffered_then_applied_on_the_grant_flip() {
        let mut rig = Rig::new();
        rig.grant_realm();
        let _ = rig.tick(vec![open_input_slot(SESSION, GATEWAY, 5)]); // adopting dot, NOT granted
        let pose = crossing_pose();

        // The crossing arrives BEFORE the adopt flip → BUFFERED (no ack, dot unchanged).
        let sent = rig.tick(vec![crossing_msg(TransferId(7), Fence(2), pose)]);
        assert!(
            !acked(&sent, TransferId(7)),
            "a buffered crossing is not acked yet"
        );
        let dot = rig.world.resource::<Dots>().0[&SESSION];
        assert_eq!(
            dot.pose.pos.offset(),
            DVec3::ZERO,
            "buffered, not applied (still adopting)"
        );
        assert_eq!(rig.world.resource::<StubStats>().crossings_buffered, 1);
        assert_eq!(rig.world.resource::<StubStats>().crossings_applied, 0);

        // A redelivery WHILE STILL BUFFERED (the saga re-emits at-least-once) overwrites the same
        // key and must NOT inflate the buffered count (audit F-3 — the counter is per-crossing, not
        // per-redelivery).
        let sent = rig.tick(vec![crossing_msg(TransferId(7), Fence(2), pose)]);
        assert!(
            !acked(&sent, TransferId(7)),
            "still buffered, still not acked"
        );
        assert_eq!(
            rig.world.resource::<StubStats>().crossings_buffered,
            1,
            "a still-buffered redelivery does not inflate the buffered count"
        );

        // The adopt grant-flip DRAINS the buffer → applies the pose + acks, but the dot STAYS Ghost
        // (1d.5b.3b — the drained path shares `apply_crossing`, whose autonomous promote is gone).
        let sent = rig.tick(vec![adopted_head(Fence(2))]);
        let dot = rig.world.resource::<Dots>().0[&SESSION];
        assert_eq!(
            dot.pose,
            pose.sanitized(),
            "the buffered crossing applied on the flip"
        );
        assert!(
            !dot.authority.simulates(),
            "the drained crossing leaves the dot a Ghost (the promote relocated to on_saga_promote)"
        );
        assert_eq!(rig.world.resource::<StubStats>().crossings_applied, 1);
        assert!(acked(&sent, TransferId(7)), "the drained crossing is acked");

        // The saga Promote then flips it Ghost→Owned (the buffered-drain path also satisfies
        // pose-before-promote — the crossing journaled on the drain).
        let _ = rig.tick(vec![promote_msg(Fence(2), NodeId(99))]);
        assert_eq!(
            rig.world.resource::<Dots>().0[&SESSION].authority,
            Authority::Owned { fence: Fence(2) },
            "the Promote flips Owned on the buffered-drain path"
        );

        // A redelivery AFTER the drained crossing was journaled hits the IMMEDIATE path (now Owned,
        // still matched by `crossing_target`) and the journal dedups it across the boundary — re-ack
        // only, NO second apply, NO re-buffer.
        let buffered_before = rig.world.resource::<StubStats>().crossings_buffered;
        let sent = rig.tick(vec![crossing_msg(TransferId(7), Fence(2), pose)]);
        assert_eq!(
            rig.world.resource::<StubStats>().crossings_applied,
            1,
            "a redelivery after a DRAINED crossing does not re-apply (journal spans the boundary)"
        );
        assert_eq!(
            rig.world.resource::<StubStats>().crossings_buffered,
            buffered_before,
            "a post-drain redelivery re-acks via the journal — it is NOT re-buffered",
        );
        assert!(
            acked(&sent, TransferId(7)),
            "the post-drain redelivery still re-acks"
        );
    }

    #[test]
    fn a_stale_fence_crossing_is_dropped() {
        let mut rig = Rig::new();
        rig.grant_realm();
        let _ = rig.tick(vec![open_input_slot(SESSION, GATEWAY, 5)]);
        let _ = rig.tick(vec![adopted_head(Fence(2))]); // adopted at Fence(2)
        // A crossing at Fence(1) is BELOW the dot's recorded authority fence → stale (fence rule 1).
        let sent = rig.tick(vec![crossing_msg(TransferId(7), Fence(1), crossing_pose())]);
        assert_eq!(rig.world.resource::<StubStats>().crossings_stale, 1);
        assert_eq!(rig.world.resource::<StubStats>().crossings_applied, 0);
        let dot = rig.world.resource::<Dots>().0[&SESSION];
        assert_eq!(
            dot.pose.pos.offset(),
            DVec3::ZERO,
            "a stale crossing does not move the dot"
        );
        assert!(
            !acked(&sent, TransferId(7)),
            "a stale crossing is not acked"
        );
    }

    #[test]
    fn a_non_crossing_transfer_payload_is_a_counted_noop() {
        let mut rig = Rig::new();
        rig.grant_realm();
        let _ = rig.tick(vec![open_input_slot(SESSION, GATEWAY, 5)]);
        let _ = rig.tick(vec![adopted_head(Fence(2))]);
        // An InitialSpawn payload is not a 1d.1 dest concern → counted no-op (no apply, no ack).
        let env = InterShardFlow::Transfer(TransferEnvelope {
            transfer_id: TransferId(7),
            universe_epoch: vd_core::EpochId(1),
            schema_version: vd_wire::intershard::TRANSFER_SCHEMA_VERSION,
            fence: Fence(2),
            step_id: STUB_CROSSING_STEP,
            class: vd_core::entity_kind::DurabilityClass::Durable,
            payload: TransitionPayload::InitialSpawn {
                entity: SUBJECT,
                to_realm: TO_REALM,
                pose: crossing_pose(),
                state: vec![],
            },
        });
        let sent = rig.tick(vec![wire_msg(ORCH, MsgClass::Saga, &env)]);
        assert_eq!(rig.world.resource::<StubStats>().crossings_unhandled, 1);
        assert_eq!(rig.world.resource::<StubStats>().crossings_applied, 0);
        assert!(
            !acked(&sent, TransferId(7)),
            "an unhandled payload is not acked"
        );
    }

    #[test]
    fn a_stale_epoch_crossing_is_refused() {
        let mut rig = Rig::new();
        rig.grant_realm();
        let _ = rig.tick(vec![open_input_slot(SESSION, GATEWAY, 5)]);
        let _ = rig.tick(vec![adopted_head(Fence(2))]);
        // A crossing minted under epoch 2 while this shard's clock epoch is 1 (the rig default) — a
        // delayed redelivery across a re-genesis, or a clock-desync bug. Refused at the ingress
        // BEFORE the payload match and BEFORE find-dot: counted, never applied, never buffered,
        // never acked (transfer_protocol §3.3 fail-safe — no entity placed at a stale celestial
        // position). This exercises the mismatch arm of the epoch gate; every other crossing test
        // exercises the match arm (rig clock epoch == envelope epoch == EpochId(1)).
        let env = InterShardFlow::Transfer(TransferEnvelope {
            transfer_id: TransferId(7),
            universe_epoch: vd_core::EpochId(2),
            schema_version: vd_wire::intershard::TRANSFER_SCHEMA_VERSION,
            fence: Fence(2),
            step_id: STUB_CROSSING_STEP,
            class: vd_core::entity_kind::DurabilityClass::Durable,
            payload: TransitionPayload::StubCrossing {
                entity: SUBJECT,
                from_realm: FROM_REALM,
                to_realm: TO_REALM,
                pose: crossing_pose(),
                state: vec![],
            },
        });
        let sent = rig.tick(vec![wire_msg(ORCH, MsgClass::Saga, &env)]);
        assert_eq!(
            rig.world.resource::<StubStats>().crossings_epoch_mismatch,
            1
        );
        assert_eq!(rig.world.resource::<StubStats>().crossings_applied, 0);
        assert_eq!(rig.world.resource::<StubStats>().crossings_buffered, 0);
        assert_eq!(
            rig.world.resource::<Dots>().0[&SESSION].pose.pos.offset(),
            DVec3::ZERO,
            "a stale-epoch crossing does not move the dot"
        );
        assert!(
            !acked(&sent, TransferId(7)),
            "a stale-epoch crossing is not acked"
        );
    }

    #[test]
    fn a_stale_epoch_re_home_is_refused() {
        let mut rig = Rig::new();
        rig.grant_realm();
        let session = SessionId(SUBJECT.0); // the deterministic clientless re-home session key
        // A re-home minted under epoch 2 while this shard's clock epoch is 1 (the rig default) — a delayed
        // redelivery across a re-genesis. Refused at the ingress BEFORE journal/adopt/ack: counted, no dot
        // created, no PromoteAck. The §3.3 fail-safe, UNIFORM with the crossing arm
        // (a_stale_epoch_crossing_is_refused). Exercises the mismatch arm of the re-home epoch gate; every
        // other re-home test exercises the match arm (rig clock epoch == cmd epoch == EpochId(1)).
        let env = InterShardFlow::ReHome(ReHomeCmd {
            transfer: TransferId(7),
            universe_epoch: vd_core::EpochId(2),
            subject: DirectoryKey::Entity(SUBJECT),
            new_fence: Fence(2),
            step_id: RE_HOME_STEP,
            state: ReHomeState::PoseOnly(crossing_pose()),
            source: NodeId(99),
        });
        let sent = rig.tick(vec![wire_msg(ORCH, MsgClass::Saga, &env)]);
        assert_eq!(rig.world.resource::<StubStats>().re_home_epoch_mismatch, 1);
        assert_eq!(rig.world.resource::<StubStats>().re_home_adopted, 0);
        assert!(
            !rig.world.resource::<Dots>().0.contains_key(&session),
            "a stale-epoch re-home creates no dot"
        );
        assert!(
            !saga_ack_to_orch(
                &sent,
                TransferControlAck::PromoteAck {
                    transfer: TransferId(7)
                }
            ),
            "a stale-epoch re-home is not acked"
        );
    }

    #[test]
    fn a_duplicate_entity_grant_head_is_an_idempotent_noop() {
        // GrantFlip::NoOp: a SECOND grant head for an already-granted dot flips nothing.
        let mut rig = Rig::new();
        rig.grant_realm();
        let _ = rig.tick(vec![open_input_slot(SESSION, GATEWAY, 5)]);
        let _ = rig.tick(vec![adopted_head(Fence(2))]); // first flip → granted
        let before = rig.world.resource::<Dots>().0[&SESSION];
        let sent = rig.tick(vec![adopted_head(Fence(2))]); // duplicate → NoOp
        let after = rig.world.resource::<Dots>().0[&SESSION];
        assert_eq!(after, before, "a duplicate grant head changes nothing");
        assert!(!acked(&sent, TransferId(7)), "no ack on a duplicate grant");
    }

    #[test]
    fn producer_less_reliable_flows_push_with_the_retained_marker() {
        // R-6d §7 CONFORMANCE: the two `FlowDurabilityClass::ProducerLessReliable` flows — the source-shard
        // `TransientBatch` emit (D-6 #1) and the band-exit `Ghost::Despawn` — have NO scan_deadlines
        // re-driver, so their push MUST carry `Durability::Retained` (the R-6d durable outbox mirrors +
        // replays them across a source crash). The `send`/`push_flow` default is Ephemeral, so THIS test is
        // the guarantee that these two sites opted into durability; a future producer-less flow (compile-
        // forced-classified by `durability_class`, R-6d2a) whose author forgets the marker trips this.
        use vd_wire::intershard::FlowDurabilityClass;
        let producer_less = |sent: &[(NodeId, MsgClass, InterShardFlow, Durability)]| {
            sent.iter()
                .find(|(_, _, f, _)| {
                    f.durability_class() == FlowDurabilityClass::ProducerLessReliable
                })
                .map(|(_, _, _, dur)| *dur)
        };

        // --- (a) TransientBatch (the emit_transient_batch producer-less one-shot) ---
        let mut rig = Rig::new();
        rig.grant_realm();
        rig.world.resource_mut::<OwnedTransients>().0.insert(
            EntityId::pack(EntityKind::Debris, 1, 7, 1),
            Transient {
                pose: transient_pose(),
                anchor_fence: Fence(1),
                status: TransientStatus::Crossing {
                    dest: DEST_NODE,
                    to_realm: RealmId::System(8),
                    dst_realm_fence: Fence(2),
                    batch: TransferId(0xB3),
                    to_parent: None,
                },
                prev_offset: DVec3::ZERO,
            },
        );
        assert_eq!(
            producer_less(&rig.tick_raw(vec![])),
            Some(Durability::Retained),
            "the TransientBatch emit MUST push Durability::Retained (no re-driver, D-6 #1)"
        );

        // --- (b) band-exit Ghost::Despawn (reuses the_dest_feed_despawns...'s setup) ---
        let mut rig = Rig::new();
        rig.grant_realm();
        let _ = rig.tick(vec![open_input_slot(SESSION, GATEWAY, 5)]);
        let _ = rig.tick(vec![adopted_head(Fence(2))]);
        let _ = rig.tick(vec![crossing_msg(TransferId(7), Fence(2), crossing_pose())]);
        let _ = rig.tick(vec![promote_msg(Fence(2), NodeId(99))]);
        rig.set_local_tick(6);
        let _ = rig.tick(vec![]); // in-band: streams a Delta, keeps the feed
        let exit_pos = crossing_pose().pos.offset() + DVec3::new(3.0, 0.0, 0.0);
        rig.world
            .resource_mut::<Dots>()
            .0
            .get_mut(&SESSION)
            .expect("the owned dot")
            .pose
            .pos = LatticePos::local(exit_pos);
        assert_eq!(
            producer_less(&rig.tick_raw(vec![])),
            Some(Durability::Retained),
            "the band-exit Ghost::Despawn MUST push Durability::Retained (no re-driver)"
        );
    }

    // ---- Slice 3d/3e/4b + C-3 (the CONTAINMENT re-home trigger) --------------------------------

    use vd_core::geometry::{Boundary, BoundaryTuning, ContainmentBand, RealmRegion};
    use vd_core::pose::frame_for_realm;

    /// The shard's OWN realm (mirrors `config().realm`); a re-home fires when the container differs.
    const OWN_REALM: RealmId = RealmId::System(7);
    /// The DEEPER child region a dot docks INTO (its container becomes `OTHER_REALM` ⇒ re-home here).
    const OTHER_REALM: RealmId = RealmId::Planet(42);
    /// The ambient ROOT realm (`parent: None`) — the `container` fold identity. A large Shell covering
    /// everything, so an entity is ALWAYS in ≥1 realm (the "no way we will not be in any realm" mandate).
    const ROOT_REALM: RealmId = RealmId::System(0);
    /// The PARENT an OWN-realm child undocks OUTWARD to (distinct from `OWN_REALM` and `OTHER_REALM`, so
    /// the undock destination a test reads is unambiguous). In the escape forest `OWN_REALM` nests under it.
    const PARENT_REALM: RealmId = RealmId::System(77);
    const TRIG_SESSION: SessionId = SessionId(0xBB);

    /// Build a velocity-safe containment band (slow v_rel so it is inset/outset-sized, not widened). The
    /// generous inset/outset (metres) means a dot placed clearly inside/outside a region's shell resolves
    /// membership without any per-tick flap.
    fn band() -> ContainmentBand {
        ContainmentBand::for_containment_velocity_safe(50.0, 100.0, 1.0, 0.05, 1.0)
            .expect("valid containment band")
    }

    /// The frame `realm`'s region is expressed in. System/Planet always resolve; under `IdentityFrames`
    /// the frame is a no-op at P3, so positions are frame-invariant regardless.
    fn frame_of(realm: RealmId) -> FrameRef {
        frame_for_realm(realm, None).expect("System/Planet realm always resolves a frame")
    }

    /// One `RealmRegion` shell at `center` of radius `r`, in `realm`'s own frame, nested under `parent`.
    fn region(realm: RealmId, parent: Option<RealmId>, center: DVec3, r: f64) -> RealmRegion {
        RealmRegion {
            realm,
            center: LatticePos::local(center),
            frame: frame_of(realm),
            shape: Boundary::Shell { r },
            band: band(),
            aoi: vd_core::geometry::AoiConfig::inert(),
            parent,
        }
    }

    /// One `RealmRegion` axis-aligned BOX at `center` with per-axis `half`, nested under `parent`.
    fn region_box(
        realm: RealmId,
        parent: Option<RealmId>,
        center: DVec3,
        half: DVec3,
    ) -> RealmRegion {
        RealmRegion {
            realm,
            center: LatticePos::local(center),
            frame: frame_of(realm),
            shape: Boundary::Aabb { half },
            band: band(),
            aoi: vd_core::geometry::AoiConfig::inert(),
            parent,
        }
    }

    /// The AMBIENT-ROOT region: a huge shell at the origin covering all placements a test uses, so the
    /// container fold is total (every point is at least in the root). `parent: None`.
    fn root_region() -> RealmRegion {
        region(ROOT_REALM, None, DVec3::ZERO, 1.0e9)
    }

    /// The shard's OWN-realm region (`System(7)`): a large shell at the origin nested under the root. A dot
    /// inside only this (and the root) has container == `OWN_REALM` ⇒ NO re-home.
    fn own_region() -> RealmRegion {
        region(OWN_REALM, Some(ROOT_REALM), DVec3::ZERO, 100_000.0)
    }

    /// The DEEPER child region (`OTHER_REALM` = `Planet(42)`): a small shell at the origin nested under
    /// `OWN_REALM`. A dot clearly inside it ⇒ container == `OTHER_REALM` ⇒ re-home INWARD to `OTHER_REALM`.
    fn child_region() -> RealmRegion {
        region(OTHER_REALM, Some(OWN_REALM), DVec3::ZERO, 1000.0)
    }

    /// Plant the STANDARD 3-level dock forest (root ⊃ own ⊃ child) into the world's `RealmRegions`. A dot
    /// inside the child re-homes to `OTHER_REALM`; a dot only inside own (outside child) stays.
    fn plant_dock_regions(rig: &mut Rig) {
        *rig.world.resource_mut::<RealmRegions>() =
            RealmRegions::new(vec![root_region(), own_region(), child_region()]);
    }

    /// Plant the ESCAPE forest: root(`System(0)`) ⊃ parent(`PARENT_REALM`) ⊃ own(`OWN_REALM`, a SMALL
    /// shell). The shard's config realm is `OWN_REALM`. A dot INSIDE the small own-shell has container ==
    /// `OWN_REALM` (no re-home); a dot OUTSIDE it but still inside the parent shell has container ==
    /// `PARENT_REALM` ⇒ re-home OUTWARD to `PARENT_REALM` (the undock). Symmetric with the dock — the same
    /// containment machinery, no "direction".
    fn plant_escape_regions(rig: &mut Rig) {
        let parent = region(PARENT_REALM, Some(ROOT_REALM), DVec3::ZERO, 100_000.0);
        let own_small = region(OWN_REALM, Some(PARENT_REALM), DVec3::ZERO, 1000.0);
        *rig.world.resource_mut::<RealmRegions>() =
            RealmRegions::new(vec![root_region(), parent, own_small]);
    }

    /// Insert an OWNED dot (a durable Player by default) at frame-local `offset`, its `prev_offset`
    /// seeded to the SAME offset (a fresh spawn — the first evaluated segment is degenerate).
    fn insert_owned_dot(rig: &mut Rig, session: SessionId, entity: EntityId, offset: DVec3) {
        rig.world.resource_mut::<Dots>().0.insert(
            session,
            Dot {
                entity,
                account: AccountId(1),
                session_fence: Fence(1),
                gateway: GATEWAY,
                granted: true,
                input_active: false,
                adopting: false,
                authority: Authority::Owned { fence: Fence(1) },
                departing: false,
                entity_fence: Fence(1),
                pose: StampedPose::at_rest(config().frame, offset, UniverseTick(100)),
                yaw: 0.0,
                pitch: 0.0,
                last_applied_seq: None,
                prev_offset: offset,
            },
        );
    }

    /// Move an owned dot's frame-local offset (its render/trigger position) without touching
    /// `prev_offset` (the trigger writes that itself each tick).
    fn move_dot(rig: &mut Rig, session: SessionId, offset: DVec3) {
        rig.world
            .resource_mut::<Dots>()
            .0
            .get_mut(&session)
            .expect("the owned dot")
            .pose
            .pos = LatticePos::local(offset);
    }

    /// Every `CrossingRequest` in the outbox (decoded), so a test asserts the exact count.
    fn crossing_requests(sent: &[(NodeId, MsgClass, Vec<u8>)]) -> Vec<CrossingRequest> {
        sent.iter()
            .filter_map(
                |(_, _, b)| match postcard::from_bytes::<InterShardFlow>(b) {
                    Ok(InterShardFlow::CrossingRequest(r)) => Some(r),
                    _ => None,
                },
            )
            .collect()
    }

    /// Every `CrossingAbortedAck` in the outbox (decoded) — the source's latch-clear confirm (3f-D).
    fn crossing_aborted_acks(sent: &[(NodeId, MsgClass, Vec<u8>)]) -> Vec<CrossingAborted> {
        sent.iter()
            .filter_map(
                |(_, _, b)| match postcard::from_bytes::<InterShardFlow>(b) {
                    Ok(InterShardFlow::CrossingAbortedAck(a)) => Some(a),
                    _ => None,
                },
            )
            .collect()
    }

    /// Every `TransientCrossingRequest` in the outbox (decoded).
    fn transient_crossing_requests(
        sent: &[(NodeId, MsgClass, Vec<u8>)],
    ) -> Vec<TransientCrossingRequest> {
        sent.iter()
            .filter_map(
                |(_, _, b)| match postcard::from_bytes::<InterShardFlow>(b) {
                    Ok(InterShardFlow::TransientCrossingRequest(r)) => Some(r),
                    _ => None,
                },
            )
            .collect()
    }

    #[test]
    fn frame_context_places_every_region_at_identity_and_defaults_own_to_galaxyspace() {
        use vd_core::frame::FrameContext;
        // (a) EMPTY forest → `own` defaults to `GalaxySpace` (the detector short-circuits on `is_empty`
        // before ever building this, but the default must still be well-formed): `GalaxySpace` resolves to
        // the identity via the `own` arm, and NO other frame is placed, so any other frame is `None`.
        let empty = RealmRegions::new(vec![]);
        let ectx = empty.frame_context(20.0);
        assert_eq!(
            ectx.placement(FrameRef::GalaxySpace, UniverseTick(0)),
            Some(FramePlacement::identity()),
            "empty-forest own defaults to GalaxySpace ⇒ identity",
        );
        assert_eq!(
            ectx.placement(frame_of(OWN_REALM), UniverseTick(7)),
            None,
            "an unplaced frame in an empty forest resolves to None",
        );
        // (b) The standard dock forest (root ⊃ own ⊃ child): EVERY region frame resolves to the identity
        // placement at EVERY tick (static walk scale) — the byte-identity guarantee vs the retired
        // `IdentityFrames`. FA-4 will hand MOVING direct children a live orbital placement; here all static.
        let ctx = RealmRegions::new(vec![root_region(), own_region(), child_region()])
            .frame_context(20.0);
        for r in [root_region(), own_region(), child_region()] {
            for tick in [UniverseTick(0), UniverseTick(1_000_000)] {
                assert_eq!(
                    ctx.placement(r.frame, tick),
                    Some(FramePlacement::identity()),
                    "region {:?} sits at the identity placement at tick {} (static walk scale)",
                    r.realm,
                    tick.0,
                );
            }
        }
    }

    #[test]
    fn frame_context_authors_a_registered_moving_child_from_its_ephemeris() {
        use vd_core::celestial::{orbital_state, secs_since_epoch};
        use vd_core::frame::FrameContext;
        // FA-2b: a region in the MOVING roster is authored LIVE from its `OrbitalElements` each tick
        // (`with_moving_child`) — its placement TRACKS the orbit; a region ABSENT from the roster stays
        // static at the identity (the byte-identity arm). The moving child is `child_region`
        // (`OTHER_REALM` = Planet 42); the roster maps its realm to a Kepler orbit.
        let elements = OrbitalElements {
            sma: 1.5e11,
            ecc: 0.1,
            inclination: 0.4,
            raan: 0.3,
            arg_periapsis: 0.9,
            mean_anomaly_epoch: 0.2,
            central_mass: 1.989e30,
        };
        let mut moving = BTreeMap::new();
        moving.insert(OTHER_REALM, elements);
        let regions = RealmRegions::new(vec![root_region(), own_region(), child_region()])
            .with_moving_children(moving);
        let tick_hz = 20.0;
        let ctx = regions.frame_context(tick_hz);
        // The MOVING child: authored from the ephemeris at the sampled tick (the `with_moving_child` arm).
        let tick = UniverseTick(1_000);
        let state = orbital_state(&elements, secs_since_epoch(tick.0, tick_hz));
        assert_eq!(
            ctx.placement(frame_of(OTHER_REALM), tick),
            Some(FramePlacement::moving(state.position, state.velocity)),
            "a registered moving child is authored live from its orbit, not the static identity",
        );
        // A region ABSENT from the roster (`own`) stays static at the identity — the byte-identity arm.
        assert_eq!(
            ctx.placement(frame_of(OWN_REALM), tick),
            Some(FramePlacement::identity()),
            "a non-roster region stays at the identity placement (byte-identical to FA-1)",
        );
    }

    #[test]
    fn an_unnameable_pose_frame_safe_degrades_to_non_member_never_a_spurious_container() {
        // FA-1 safe-degrade: a subject whose pose frame the shard cannot NAME (not among its regions) has
        // `region_signed_distance` → `Err` → `f64::MAX` for EVERY region, so it is a member of NONE and its
        // deepest container folds to the ambient ROOT — never a spurious INNER container, never a panic. The
        // discriminator: at the origin under the OWN (registered) frame the deepest container is the child
        // `OTHER_REALM` (a re-home INWARD); under an un-nameable frame the child must NOT be entered.
        let mut rig = Rig::new(); // owns System 7 (config().realm == OWN_REALM)
        rig.grant_realm();
        plant_dock_regions(&mut rig); // root(1e9) ⊃ own(1e5) ⊃ child(1000), all shells at the origin
        let entity = EntityId::pack(EntityKind::Player, 10, 1, 77);
        // A Station frame the dock forest NEVER planted — un-nameable to any dock region.
        insert_owned_dot_framed(
            &mut rig,
            TRIG_SESSION,
            entity,
            FrameRef::StationLocal { station_seed: 999 },
            DVec3::ZERO,
        );
        let mut all: Vec<(NodeId, MsgClass, Vec<u8>)> = Vec::new();
        for t in 2..6 {
            rig.set_local_tick(t);
            all.extend(rig.tick(vec![]));
        }
        assert!(
            crossing_requests(&all)
                .iter()
                .all(|r| r.to_realm != OTHER_REALM),
            "an un-nameable pose frame safe-degrades to non-member ⇒ NEVER re-homes into the inner child",
        );
    }

    #[test]
    fn owning_realm_reads_the_pose_frame_when_nameable() {
        // The `Some` arm (the LIVE path — every real shard frame is nameable): the owning realm is the
        // pose FRAME's realm, IGNORING the `config_realm` fallback. Covers a System, Planet, and Area frame.
        assert_eq!(
            super::owning_realm(
                FrameRef::SystemSpace { system_seed: 7 },
                RealmId::System(99)
            ),
            RealmId::System(7),
            "a System frame owns System(system_seed), not the config fallback",
        );
        assert_eq!(
            super::owning_realm(
                FrameRef::PlanetCentered { planet_seed: 3 },
                RealmId::System(99)
            ),
            RealmId::Planet(3),
            "a Planet frame owns Planet(planet_seed)",
        );
        assert_eq!(
            super::owning_realm(
                FrameRef::AreaLocal {
                    planet_seed: 3,
                    area_seed: 8,
                },
                RealmId::System(99),
            ),
            RealmId::Area(8),
            "an Area frame owns Area(area_seed) — the very frame the to_parent fix makes form",
        );
    }

    #[test]
    fn owning_realm_falls_back_to_config_realm_for_an_unnameable_frame() {
        // The `None` fallback arm (otherwise UNCOVERABLE — no live shard uses GalaxySpace): a frame whose
        // `FrameRef::realm()` is None (`GalaxySpace`) falls back to the shard's `config_realm`.
        assert_eq!(
            super::owning_realm(FrameRef::GalaxySpace, RealmId::System(42)),
            RealmId::System(42),
            "GalaxySpace has no realm → the config_realm fallback",
        );
    }

    #[test]
    fn durable_dot_dwelling_in_band_triggers_exactly_one_crossing_request() {
        let mut rig = Rig::new();
        rig.grant_realm();
        // The dock forest (root ⊃ own(System(7)) ⊃ child(Planet(42))).
        plant_dock_regions(&mut rig);
        let entity = EntityId::pack(EntityKind::Player, 10, 1, 1);
        // Deep inside the CHILD region's shell (|100| ≪ 1000 - inset): container == OTHER_REALM ⇒ re-home.
        insert_owned_dot(&mut rig, TRIG_SESSION, entity, DVec3::new(100.0, 0.0, 0.0));

        // Under containment the band IS the dwell: the re-home fires as soon as the container differs;
        // the RequestInFlight latch then suppresses every later tick → exactly ONE request across the dwell.
        let mut all: Vec<(NodeId, MsgClass, Vec<u8>)> = Vec::new();
        for t in 2..8 {
            rig.set_local_tick(t);
            all.extend(rig.tick(vec![]));
        }
        let reqs = crossing_requests(&all);
        assert_eq!(
            reqs.len(),
            1,
            "exactly ONE CrossingRequest across the dwell"
        );
        assert_eq!(reqs[0].subject, DirectoryKey::Entity(entity));
        assert_eq!(reqs[0].from_realm, config().realm);
        assert_eq!(reqs[0].to_realm, OTHER_REALM);
        assert_eq!(reqs[0].subject_fence, Fence(1));
        // Slice 3f: the source threads the dot's session (its `Dots` map key) so the orchestrator's
        // saga can `PrepareSubscribe` to the client's gateway.
        assert_eq!(reqs[0].session, TRIG_SESSION);
        // 3f-D: the FIRST crossing uses attempt 0.
        assert_eq!(reqs[0].attempt, 0);
        // The latch is set to the deterministic id BOTH ends derive (attempt 0).
        assert_eq!(
            rig.world.resource::<RequestInFlight>().0.get(&entity),
            Some(&crossing_transfer_id(
                DirectoryKey::Entity(entity),
                Fence(1),
                0
            )),
        );
        let stats = rig.world.resource::<StubStats>();
        assert_eq!(stats.crossings_requested, 1);
        // The cooldown was armed (`last_commit_tick` set) on the emit.
        assert!(
            rig.world
                .resource::<CrossingProgress>()
                .0
                .get(&entity)
                .expect("progress")
                .last_commit_tick
                .is_some(),
            "the commit armed the cooldown",
        );
    }

    #[test]
    fn the_in_flight_latch_suppresses_a_second_request() {
        let mut rig = Rig::new();
        rig.grant_realm();
        plant_dock_regions(&mut rig);
        let entity = EntityId::pack(EntityKind::Player, 10, 1, 2);
        insert_owned_dot(&mut rig, TRIG_SESSION, entity, DVec3::new(100.0, 0.0, 0.0));
        let mut all: Vec<(NodeId, MsgClass, Vec<u8>)> = Vec::new();
        // Deep inside the child ⇒ the FIRST re-home fires (sets the latch + arms the cooldown).
        for t in 2..8 {
            rig.set_local_tick(t);
            all.extend(rig.tick(vec![]));
        }
        assert_eq!(crossing_requests(&all).len(), 1, "the first crossing fired");
        assert!(
            rig.world
                .resource::<RequestInFlight>()
                .0
                .contains_key(&entity)
        );
        // WITHOUT sending the Demote terminal (so the latch STAYS set), drive a SECOND container change:
        // leave the child region (container reverts to own ⇒ no re-home) long enough for the cooldown to
        // lapse, then re-enter the child (container flips back to OTHER_REALM). This second re-home
        // decision reaches `fan_out_crossing` — but the latch is still held, so it takes the SUPPRESS arm.
        for t in 8..12 {
            rig.set_local_tick(t);
            // Outside the child shell (sd 1000 > outset) but still inside own ⇒ container == own.
            move_dot(&mut rig, TRIG_SESSION, DVec3::new(2000.0, 0.0, 0.0));
            all.extend(rig.tick(vec![]));
        }
        for t in 12..18 {
            rig.set_local_tick(t);
            move_dot(&mut rig, TRIG_SESSION, DVec3::new(100.0, 0.0, 0.0)); // back inside the child
            all.extend(rig.tick(vec![]));
        }
        assert_eq!(
            crossing_requests(&all).len(),
            1,
            "the RequestInFlight latch holds it to ONE request across the second container change",
        );
        assert!(
            rig.world
                .resource::<StubStats>()
                .crossings_suppressed_in_flight
                > 0,
            "the second container change hit the in-flight SUPPRESS arm",
        );
    }

    #[test]
    fn a_transient_crossing_emits_a_transient_crossing_request() {
        let mut rig = Rig::new();
        rig.grant_realm();
        plant_dock_regions(&mut rig);
        // A Debris entity is Transient → the transient fan-out arm.
        let entity = EntityId::pack(EntityKind::Debris, 10, 1, 3);
        rig.world.resource_mut::<OwnedTransients>().0.insert(
            entity,
            Transient {
                pose: StampedPose::at_rest(
                    config().frame,
                    DVec3::new(100.0, 0.0, 0.0),
                    UniverseTick(100),
                ),
                anchor_fence: Fence(1),
                status: TransientStatus::Held { outbound: None },
                prev_offset: DVec3::new(100.0, 0.0, 0.0),
            },
        );
        // A transient carries NO per-entity latch (its batch journal dedups instead), so the ONLY
        // anti-thrash is the symmetric `k_dwell` cooldown (§2.7). Run WITHIN the cooldown window
        // (commit at tick 2, k_dwell = 5) so the container-differs re-home fires exactly once.
        let mut all: Vec<(NodeId, MsgClass, Vec<u8>)> = Vec::new();
        for t in 2..7 {
            rig.set_local_tick(t);
            all.extend(rig.tick(vec![]));
        }
        let reqs = transient_crossing_requests(&all);
        assert_eq!(reqs.len(), 1, "exactly ONE TransientCrossingRequest");
        assert_eq!(reqs[0].subject, DirectoryKey::Entity(entity));
        assert_eq!(reqs[0].to_realm, OTHER_REALM);
        assert_eq!(reqs[0].src_realm_fence, Fence(1));
        assert_eq!(
            rig.world
                .resource::<StubStats>()
                .transient_crossings_requested,
            1
        );
        // Transients are NOT latched in RequestInFlight (the batch journal dedups instead).
        assert!(rig.world.resource::<RequestInFlight>().0.is_empty());
    }

    #[test]
    fn a_durable_tagged_transient_degrades_instead_of_panicking() {
        // REGRESSION (goal-audit L4): a Durable-TAGGED entity in the held-transient set (a kind/loop
        // mismatch — e.g. a mis-tagged batch item) reaches the durable crossing arm from the transient
        // loop, which passes `None` for the session. The arm must DEGRADE (count + emit nothing), NOT
        // panic on `subject_session.expect`, and must leave NO orphan latch.
        let mut rig = Rig::new();
        rig.grant_realm();
        plant_dock_regions(&mut rig);
        // A Player entity is DURABLE, but we place it (wrongly) in the transient set.
        let entity = EntityId::pack(EntityKind::Player, 10, 1, 9);
        rig.world.resource_mut::<OwnedTransients>().0.insert(
            entity,
            Transient {
                pose: StampedPose::at_rest(
                    config().frame,
                    DVec3::new(100.0, 0.0, 0.0),
                    UniverseTick(100),
                ),
                anchor_fence: Fence(1),
                status: TransientStatus::Held { outbound: None },
                prev_offset: DVec3::new(100.0, 0.0, 0.0),
            },
        );
        let mut all: Vec<(NodeId, MsgClass, Vec<u8>)> = Vec::new();
        for t in 2..8 {
            rig.set_local_tick(t);
            all.extend(rig.tick(vec![])); // must not panic
        }
        assert!(
            crossing_requests(&all).is_empty(),
            "a session-less durable subject emits NO CrossingRequest",
        );
        assert!(
            rig.world.resource::<RequestInFlight>().0.is_empty(),
            "the degradation leaves no orphan latch",
        );
        assert!(
            rig.world
                .resource::<StubStats>()
                .crossing_durable_no_session
                > 0,
            "the kind/loop mismatch is counted, not panicked",
        );
    }

    #[test]
    fn a_dot_inside_only_its_own_realm_triggers_no_re_home() {
        // The "no re-home" case (§4): a dot inside the OWN region (System(7)) but OUTSIDE the deeper child
        // (Planet(42)) has container == OWN_REALM == the realm the shard owns it in ⇒ `should_rehome`
        // returns None. No CrossingRequest, no latch — symmetric with the empty-registry inert path.
        let mut rig = Rig::new();
        rig.grant_realm();
        plant_dock_regions(&mut rig);
        let entity = EntityId::pack(EntityKind::Player, 10, 1, 4);
        // Outside the child shell (sd 3000 > outset) but well inside own ⇒ container == OWN_REALM.
        insert_owned_dot(&mut rig, TRIG_SESSION, entity, DVec3::new(4000.0, 0.0, 0.0));
        let mut all: Vec<(NodeId, MsgClass, Vec<u8>)> = Vec::new();
        for t in 2..8 {
            rig.set_local_tick(t);
            all.extend(rig.tick(vec![]));
        }
        // Split the two emptiness checks (HR5(d): a single `assert!(a && b)` leaves the short-circuit
        // false-branch of `a` uncovered).
        assert!(
            crossing_requests(&all).is_empty(),
            "container == own ⇒ NO durable CrossingRequest",
        );
        assert!(
            transient_crossing_requests(&all).is_empty(),
            "container == own ⇒ NO TransientCrossingRequest",
        );
        assert!(
            rig.world.resource::<RequestInFlight>().0.is_empty(),
            "no re-home ⇒ no latch",
        );
    }

    #[test]
    fn a_rootless_region_forest_is_a_safe_no_op() {
        // DEFENSIVE (HR5): a NON-EMPTY forest with NO ambient root (every region has a parent — a
        // malformed set that `guard_regions_nest` rejects at boot in C-5) leaves `root_realm == None`, so
        // `evaluate_one_subject` cannot seed the `container` fold and returns EARLY — no re-home, never a
        // panic. Covers the `let Some(root_realm) = ctx.root_realm else { return cur }` guard.
        let mut rig = Rig::new();
        rig.grant_realm();
        // A single region whose `parent` is `Some(..)` ⇒ NO `parent: None` root ⇒ `root_realm == None`.
        *rig.world.resource_mut::<RealmRegions>() = RealmRegions::new(vec![own_region()]);
        let entity = EntityId::pack(EntityKind::Player, 10, 1, 5);
        insert_owned_dot(&mut rig, TRIG_SESSION, entity, DVec3::ZERO);
        let mut all: Vec<(NodeId, MsgClass, Vec<u8>)> = Vec::new();
        for t in 2..6 {
            rig.set_local_tick(t);
            all.extend(rig.tick(vec![]));
        }
        assert!(
            crossing_requests(&all).is_empty(),
            "a rootless forest cannot seed the container fold ⇒ NO re-home",
        );
    }

    /// The instantaneous (hysteresis-free) CONTAINER realm of `pos` over the SEED-DERIVED forest
    /// (`worldgen::realm_regions_for` — the exact geometry a shard boots): the members are the regions
    /// whose surface `pos` is on/inside (`signed_distance <= 0`), folded from the Universe root. This is
    /// the containment CONTRACT — the sim's stateful band membership layers hysteresis on top. Asserting
    /// this (NOT the emitted `to_realm`) proves the containment MODEL routes a sibling crossing through the
    /// shared parent (task #135 C-6c).
    fn seed_container_at(pos: DVec3) -> RealmId {
        let regions = vd_core::worldgen::realm_regions_for(0);
        let stamped = |p: DVec3| StampedPose {
            frame: FrameRef::SystemSpace { system_seed: 0 },
            pos: LatticePos::local(p),
            vel: DVec3::ZERO,
            orient: vd_core::glam::DQuat::IDENTITY,
            universe_tick: UniverseTick(0),
        };
        let members: Vec<DepthKey> = regions
            .iter()
            .enumerate()
            .filter(|(_, r)| {
                region_signed_distance(&stamped(pos), r, &IdentityFrames)
                    .expect("IdentityFrames never errors")
                    <= 0.0
            })
            .map(|(ix, r)| (region_depth(&regions, r.realm), r.realm, ix))
            .collect();
        container(RealmId::System(0), &members)
    }

    #[test]
    fn symmetric_recross_resolves_the_full_container_sequence_both_ways() {
        // C-6c — THE CONTAINER-SEQUENCE round-trip over the SEED forest: a dot scripted +X
        // origin(System 7) → x=50(Galaxy gap) → x=100(System 8) → x=50(Galaxy) → origin(System 7) resolves
        // the CONTAINER-realm sequence [System 7, Galaxy, System 8, Galaxy, System 7]. This is the pure
        // geometric proof the sibling crossing routes THROUGH the shared Galaxy parent BOTH ways (leaving a
        // system lands in the Galaxy ancestor, entering the next is the Galaxy shard's job). We assert the
        // CONTAINER (`container()`), NOT the emitted `to_realm` — the model's symmetric truth.
        const GALAXY: RealmId = RealmId::System(1);
        const SYSTEM_7: RealmId = RealmId::System(7);
        const SYSTEM_8: RealmId = RealmId::System(8);
        let waypoints = [
            (DVec3::new(0.0, 0.0, 0.0), SYSTEM_7), // origin: inside System 7's SOI (r=40)
            (DVec3::new(50.0, 0.0, 0.0), GALAXY),  // the gap: outside 7 (r=40) and 8 (@100, r=40)
            (DVec3::new(100.0, 0.0, 0.0), SYSTEM_8), // inside System 8's SOI
            (DVec3::new(50.0, 0.0, 0.0), GALAXY),  // back in the gap (the RETURN leg)
            (DVec3::new(0.0, 0.0, 0.0), SYSTEM_7), // home (the full reverse-cross)
        ];
        let seq: Vec<RealmId> = waypoints
            .iter()
            .map(|(p, _)| seed_container_at(*p))
            .collect();
        let expected: Vec<RealmId> = waypoints.iter().map(|(_, r)| *r).collect();
        assert_eq!(
            seq, expected,
            "the container sequence routes System 7 → Galaxy → System 8 → Galaxy → System 7 BOTH ways",
        );
    }

    #[test]
    fn escape_soi_lands_in_the_immediate_parent_not_a_skipped_ancestor() {
        // C-6c — the ESCAPE-SOI mandate: leaving a region lands you in its IMMEDIATE parent, never a
        // skipped ancestor. Leaving Planet 7 (SOI r=10 at (20,0,0)) lands in System 7 (the star system it
        // orbits), NOT the Galaxy; leaving System 7 (SOI r=40) lands in the Galaxy (the between-systems
        // space), NOT the Universe. The container fold picks the DEEPEST containing realm at each step.
        const GALAXY: RealmId = RealmId::System(1);
        const SYSTEM_7: RealmId = RealmId::System(7);
        const PLANET_7: RealmId = RealmId::Planet(7);
        // Inside Planet 7's SOI (centered at (20,0,0), r=10) → the PLANET.
        assert_eq!(seed_container_at(DVec3::new(20.0, 0.0, 0.0)), PLANET_7);
        // Just OUTSIDE Planet 7 but still inside System 7 (at the star, the origin) → the STAR SYSTEM (the
        // immediate parent), NOT the Galaxy grandparent.
        assert_eq!(seed_container_at(DVec3::ZERO), SYSTEM_7);
        // Outside System 7's SOI (x=50, in the gap) → the GALAXY (the immediate parent), NOT the Universe.
        assert_eq!(seed_container_at(DVec3::new(50.0, 0.0, 0.0)), GALAXY);
    }

    #[test]
    fn the_live_containment_scan_holds_a_dense_crowd_in_one_realm_without_a_spurious_re_home() {
        // C-6c SCALE — the "hundreds in ONE location" mandate at the DETECTOR tier. The O(subjects × regions)
        // container fold runs over the WHOLE owned crowd EVERY tick; this proves it stays bounded + CORRECT
        // at crowd scale with the REAL production neighbourhood planted. (The e2e density gate
        // `p1_volume_dense_hundreds_walk_under_invariants` runs the detector INERT — empty `RealmRegions`,
        // early-return — so the live full-scan is only exercised at N ≥ 128 HERE.)
        const CROWD: usize = 128; // the "hundreds in one location" floor (N ≥ 128)
        let mut rig = Rig::new();
        rig.grant_realm();
        // The REAL seed neighbourhood `shard.rs` boots for System 7 — {Universe, Galaxy, System 7, Planet 7,
        // Station 7} (Station 7 is System 7's first-class child, task #133), a 5-region fold per subject — NOT
        // a hand-authored fixture, so this exercises the scan the bins run. The crowd clears the Station box.
        *rig.world.resource_mut::<RealmRegions>() = RealmRegions::new(
            vd_core::worldgen::realm_neighbourhood_for(0, RealmId::System(7)),
        );
        // Pack N dots into a tight ~3.5 m cube at the star (origin): every dot is well inside System 7 (r=40)
        // and ≥ ~16 m from Planet 7's centre (20,0,0) ⇒ its deepest container is System 7 == its owning realm
        // (no re-home). Overlapping positions are fine — a crowd IS hundreds in one place; the entities differ.
        for i in 0..CROWD {
            let offset = DVec3::new(
                (i % 5) as f64 - 2.0,
                ((i / 5) % 5) as f64 - 2.0,
                ((i / 25) % 5) as f64 - 2.0,
            );
            insert_owned_dot(
                &mut rig,
                SessionId(i as u128),
                EntityId::pack(EntityKind::Player, i as u32, 1, i as u32),
                offset,
            );
        }
        // Re-scan the whole crowd for several ticks.
        let mut all: Vec<(NodeId, MsgClass, Vec<u8>)> = Vec::new();
        for t in 0..5 {
            rig.set_local_tick(2 + t);
            all.extend(rig.tick(vec![]));
        }
        // The scan TOUCHED every dot (membership computed for all N) — proof the full-scan RAN at scale, not
        // an inert early-return (an empty `RealmRegions` leaves this map empty).
        assert_eq!(
            rig.world.resource::<ContainmentProgress>().0.len(),
            CROWD,
            "the live containment scan evaluated all {CROWD} dots (not an inert early-return)",
        );
        // And it stayed CORRECT at scale: the whole crowd is contained in System 7 ⇒ ZERO re-homes fire.
        let spurious = crossing_requests(&all).len();
        assert_eq!(
            spurious, 0,
            "a dense crowd inside one realm triggers NO spurious re-home under the O(N×regions) scan",
        );
    }

    // ---- task #133: the first-class Station/Area realm re-home GATE (kind-agnostic detector) --------

    /// The System-7 seed neighbourhood the LIVE `shard.rs` boots — now including the first-class Station 7
    /// child (task #133). Planting THIS (not a hand-authored fixture) proves the Station region is one the
    /// production bins actually load.
    const STATION_A: RealmId = RealmId::Station(7);

    /// Grant `realm` to THIS shard (a parameterized [`Rig::grant_realm`], which hardcodes `config().realm`)
    /// so a non-default-realm rig (e.g. a Station-owning shard) can take authority. Mirrors `grant_realm`'s
    /// round-trip directory confirmation, keyed on the passed realm.
    fn grant_realm_for(rig: &mut Rig, realm: RealmId) {
        grant_realm_for_at(rig, realm, Fence(1));
    }

    /// Like [`grant_realm_for`] but at an explicit `fence` — for asserting a RE-affirm refreshes the
    /// co-hosted child's recorded fence (the `else if ours` refresh arm of `affirm_realm_head`).
    fn grant_realm_for_at(rig: &mut Rig, realm: RealmId, fence: Fence) {
        let reply = DirectoryReply::Head {
            key: DirectoryKey::Realm(realm),
            record: Some(vd_wire::seams::directory::OwnerRecord {
                authority: AuthorityRef::Shard(SHARD),
                fence,
                lease_expires: UniverseTick(1_000),
                in_transfer: None,
            }),
        };
        let bytes = crate::io::bytes(
            postcard::to_allocvec(&InterShardFlow::DirectoryReply(reply)).expect("encode"),
        );
        let _ = rig.tick(vec![Inbound::Wire {
            from: ORCH,
            class: MsgClass::Saga,
            bytes,
        }]);
    }

    /// Affirm a realm head that is NO LONGER this shard's — a REVOKED record (`None`, the reaper dropped
    /// the lease). For a CO-HOSTED child this drives the FOREIGN/None `else` arm of `affirm_realm_head`
    /// (`cohosted.0.remove(&realm)`) — the child dropped from `CoHostedAuthority`.
    fn revoke_realm_for(rig: &mut Rig, realm: RealmId) {
        let reply = DirectoryReply::Head {
            key: DirectoryKey::Realm(realm),
            record: None,
        };
        let bytes = crate::io::bytes(
            postcard::to_allocvec(&InterShardFlow::DirectoryReply(reply)).expect("encode"),
        );
        let _ = rig.tick(vec![Inbound::Wire {
            from: ORCH,
            class: MsgClass::Saga,
            bytes,
        }]);
    }

    /// Insert an OWNED durable dot at frame-local `offset` expressed in `frame` (the frame-aware sibling of
    /// [`insert_owned_dot`], which pins the pose frame to `config().frame`). Needed for a Station-owning
    /// shard, whose dots live in the StationLocal frame. Under `IdentityFrames` the frame is inert for the
    /// container decision, but carrying the OWNING realm's frame keeps the fixture honest.
    fn insert_owned_dot_framed(
        rig: &mut Rig,
        session: SessionId,
        entity: EntityId,
        frame: FrameRef,
        offset: DVec3,
    ) {
        rig.world.resource_mut::<Dots>().0.insert(
            session,
            Dot {
                entity,
                account: AccountId(1),
                session_fence: Fence(1),
                gateway: GATEWAY,
                granted: true,
                input_active: false,
                adopting: false,
                authority: Authority::Owned { fence: Fence(1) },
                departing: false,
                entity_fence: Fence(1),
                pose: StampedPose::at_rest(frame, offset, UniverseTick(100)),
                yaw: 0.0,
                pitch: 0.0,
                last_applied_seq: None,
                prev_offset: offset,
            },
        );
    }

    #[test]
    fn a_dot_moving_into_the_station_box_re_homes_into_the_first_class_station_realm() {
        // THE task #133 headline: the SAME kind-agnostic containment detector re-homes a dot into a
        // first-class STATION realm with ZERO station-specific code (HR3). The shard OWNS System 7 and boots
        // the REAL seed neighbourhood (now {Universe, Galaxy, System 7, Planet 7, STATION 7}). A dot starts
        // at the origin (container == System 7 == owning ⇒ NO re-home), then walks into the Station BOX at
        // (-25,0,0) (a Cartesian `Aabb`, not an SOI shell) — its deepest container flips to Station 7 ≠ the
        // owning System 7, so ONE re-home fires whose `to_realm` is the Station. The box `signed_distance`
        // feeds the identical `ContainmentBand` the shells use — the Station is detected by geometry alone.
        let mut rig = Rig::new(); // owns System 7 (config().realm)
        rig.grant_realm();
        *rig.world.resource_mut::<RealmRegions>() = RealmRegions::new(
            vd_core::worldgen::realm_neighbourhood_for(0, RealmId::System(7)),
        );
        let entity = EntityId::pack(EntityKind::Player, 10, 1, 33);
        // Origin: well inside System 7 (r=40), clear of the Station box (x∈[-30,-20]) ⇒ container == System 7.
        insert_owned_dot(&mut rig, TRIG_SESSION, entity, DVec3::ZERO);
        rig.set_local_tick(2);
        let at_origin = crossing_requests(&rig.tick(vec![]));
        assert_eq!(
            at_origin.len(),
            0,
            "a dot at the origin is contained in System 7 (== owning) ⇒ NO re-home",
        );
        // Walk INTO the Station box centre — the deepest container becomes Station 7.
        let mut into: Vec<(NodeId, MsgClass, Vec<u8>)> = Vec::new();
        for t in 3..8 {
            rig.set_local_tick(t);
            move_dot(&mut rig, TRIG_SESSION, DVec3::new(-25.0, 0.0, 0.0)); // the Station box centre
            into.extend(rig.tick(vec![]));
        }
        let reqs = crossing_requests(&into);
        assert_eq!(
            reqs.len(),
            1,
            "exactly ONE re-home into the Station (latched thereafter)"
        );
        assert_eq!(
            reqs[0].to_realm, STATION_A,
            "the kind-agnostic detector re-homes into the first-class Station realm",
        );
        assert_eq!(
            reqs[0].from_realm,
            config().realm,
            "leaving the owning System 7"
        );
        assert_eq!(reqs[0].subject, DirectoryKey::Entity(entity));
    }

    #[test]
    fn a_station_owning_shard_re_homes_a_dot_that_leaves_the_station_back_to_system_7() {
        // The RETURN leg of the round-trip, proven as a GENUINE emission (symmetric detector, no direction):
        // a shard that OWNS Station 7 boots Station 7's seed neighbourhood — its own realm + its ANCESTOR
        // chain {System 7, Galaxy, Universe}. A dot INSIDE the Station box is contained in Station 7 (==
        // owning ⇒ NO re-home); when it LEAVES the box (to the origin, still inside System 7's r=40 SOI) its
        // deepest container becomes System 7 ≠ the owning Station 7, so ONE re-home fires whose `to_realm` is
        // System 7. This is the same machinery as the inbound gate above, run from the Station's authority —
        // the "both ways" proof that a Station is a first-class realm on the identical kind-agnostic path.
        let station_cfg = StubConfig {
            realm: STATION_A,
            held_realms: StubConfig::single_realm(STATION_A),
            frame: FrameRef::StationLocal { station_seed: 7 },
            ..config()
        };
        let station_frame = station_cfg.frame; // FrameRef is Copy — capture before the config move
        let mut rig = Rig::with_config(station_cfg);
        grant_realm_for(&mut rig, STATION_A);
        *rig.world.resource_mut::<RealmRegions>() =
            RealmRegions::new(vd_core::worldgen::realm_neighbourhood_for(0, STATION_A));
        let entity = EntityId::pack(EntityKind::Player, 10, 1, 34);
        // Inside the Station box (its centre) ⇒ container == Station 7 == owning ⇒ NO re-home.
        insert_owned_dot_framed(
            &mut rig,
            TRIG_SESSION,
            entity,
            station_frame,
            DVec3::new(-25.0, 0.0, 0.0),
        );
        rig.set_local_tick(2);
        let inside = crossing_requests(&rig.tick(vec![]));
        assert_eq!(
            inside.len(),
            0,
            "a dot inside the Station box is contained in Station 7 (== owning) ⇒ NO re-home",
        );
        // LEAVE the Station box to the origin — still inside System 7 (r=40), outside the Station box
        // (x∈[-30,-20]) ⇒ the deepest container becomes System 7, the immediate parent.
        let mut out: Vec<(NodeId, MsgClass, Vec<u8>)> = Vec::new();
        for t in 3..8 {
            rig.set_local_tick(t);
            move_dot(&mut rig, TRIG_SESSION, DVec3::ZERO); // out of the box, into open System 7 space
            out.extend(rig.tick(vec![]));
        }
        let reqs = crossing_requests(&out);
        assert_eq!(
            reqs.len(),
            1,
            "exactly ONE re-home back to System 7 (latched thereafter)"
        );
        assert_eq!(
            reqs[0].to_realm,
            RealmId::System(7),
            "leaving the Station re-homes back to the enclosing System 7 (symmetric, same detector)",
        );
        assert_eq!(
            reqs[0].from_realm, STATION_A,
            "leaving the owning Station 7"
        );
        assert_eq!(reqs[0].subject, DirectoryKey::Entity(entity));
    }

    #[test]
    fn a_dot_moving_into_the_area_box_re_homes_into_the_first_class_area_realm() {
        // The AREA analog of the Station inbound gate (task #133), proving the SAME kind-agnostic detector
        // re-homes into a first-class AREA realm — the DEEPEST region in the seed forest (depth 4). A shard
        // that OWNS Planet 7 boots Planet 7's seed neighbourhood (its own realm + ancestors {System 7,
        // Galaxy, Universe} + its child AREA 7). A dot starts at Planet 7's centre (20,0,0) (container ==
        // Planet 7 == owning ⇒ NO re-home — this is ALSO the escape-SOI probe point, which must NOT resolve
        // to the Area), then walks into the Area BOX at (25,0,0) — its deepest container flips to Area 7 ≠
        // the owning Planet 7, so ONE re-home fires whose `to_realm` is the Area. Zero area-specific code.
        let planet_cfg = StubConfig {
            realm: RealmId::Planet(7),
            held_realms: StubConfig::single_realm(RealmId::Planet(7)),
            frame: FrameRef::PlanetCentered { planet_seed: 7 },
            ..config()
        };
        let planet_frame = planet_cfg.frame; // FrameRef is Copy — capture before the config move
        let mut rig = Rig::with_config(planet_cfg);
        grant_realm_for(&mut rig, RealmId::Planet(7));
        *rig.world.resource_mut::<RealmRegions>() = RealmRegions::new(
            vd_core::worldgen::realm_neighbourhood_for(0, RealmId::Planet(7)),
        );
        let entity = EntityId::pack(EntityKind::Player, 10, 1, 35);
        // Planet 7's centre (20,0,0): inside Planet 7 (r=10), OUTSIDE the Area box (x∈[22,28]) ⇒ container
        // == Planet 7 == owning ⇒ NO re-home (and the escape-SOI probe still resolves to Planet 7).
        insert_owned_dot_framed(
            &mut rig,
            TRIG_SESSION,
            entity,
            planet_frame,
            DVec3::new(20.0, 0.0, 0.0),
        );
        rig.set_local_tick(2);
        let at_centre = crossing_requests(&rig.tick(vec![]));
        assert_eq!(
            at_centre.len(),
            0,
            "the dot at Planet 7's centre is contained in Planet 7 (== owning) ⇒ NO re-home",
        );
        // Walk INTO the Area box centre (25,0,0) — inside Planet 7 (r=10 sphere) AND the Area box ⇒ the
        // deepest container becomes Area 7.
        let mut into: Vec<(NodeId, MsgClass, Vec<u8>)> = Vec::new();
        for t in 3..8 {
            rig.set_local_tick(t);
            move_dot(&mut rig, TRIG_SESSION, DVec3::new(25.0, 0.0, 0.0)); // the Area box centre
            into.extend(rig.tick(vec![]));
        }
        let reqs = crossing_requests(&into);
        assert_eq!(
            reqs.len(),
            1,
            "exactly ONE re-home into the Area (latched thereafter)"
        );
        assert_eq!(
            reqs[0].to_realm,
            RealmId::Area(7),
            "the kind-agnostic detector re-homes into the first-class Area realm (the deepest region)",
        );
        assert_eq!(
            reqs[0].from_realm,
            RealmId::Planet(7),
            "leaving the owning Planet 7"
        );
        assert_eq!(reqs[0].subject, DirectoryKey::Entity(entity));
    }

    /// Co-hosting config: a shard hosting System 7 that ALSO CO-HOSTS Planet 7 (the un-hosted-child cure).
    /// Its `held_realms` is `{System(7), Planet(7)}` — the primary realm plus the co-hosted child. The
    /// region union (`realm_neighbourhood_for_held`) gives it BOTH neighbourhoods so the detector can
    /// evaluate Planet 7's SOI from the System-7 authority.
    fn cohost_planet_config() -> StubConfig {
        StubConfig {
            held_realms: BTreeSet::from([RealmId::System(7), RealmId::Planet(7)]),
            ..config()
        }
    }

    /// Grant the PRIMARY realm (System 7) AND affirm the CO-HOSTED child (Planet 7), populating both
    /// `RealmAuthority` and `CoHostedAuthority` — the co-hosting boot state. Plant the UNION region set.
    fn boot_cohost_planet(rig: &mut Rig) {
        grant_realm_for(rig, RealmId::System(7)); // primary → RealmAuthority
        grant_realm_for(rig, RealmId::Planet(7)); // co-hosted child → CoHostedAuthority
        *rig.world.resource_mut::<RealmRegions>() =
            RealmRegions::new(vd_core::worldgen::realm_neighbourhood_for_held(
                0,
                &BTreeSet::from([RealmId::System(7), RealmId::Planet(7)]),
            ));
    }

    #[test]
    fn a_cohosting_shard_affirms_the_child_realm_head_into_its_own_authority_map() {
        // The multi-realm AFFIRM path (co-hosting): a shard co-hosting Planet 7 must land the Planet-7
        // realm-head reply in `CoHostedAuthority` (independent of the primary `RealmAuthority`), so the
        // short-circuit's `held_here` answers `Some` for the child. This is the grant/affirm half of the cure.
        let mut rig = Rig::with_config(cohost_planet_config());
        grant_realm_for(&mut rig, RealmId::System(7));
        assert_eq!(
            rig.world.resource::<RealmAuthority>().0,
            Some(Fence(1)),
            "the primary System-7 realm is held on RealmAuthority",
        );
        assert!(
            !rig.world
                .resource::<CoHostedAuthority>()
                .0
                .contains_key(&RealmId::Planet(7)),
            "the child is NOT held until its own head affirms",
        );
        grant_realm_for(&mut rig, RealmId::Planet(7));
        assert_eq!(
            rig.world
                .resource::<CoHostedAuthority>()
                .0
                .get(&RealmId::Planet(7)),
            Some(&Fence(1)),
            "the co-hosted Planet-7 head lands in CoHostedAuthority (not RealmAuthority)",
        );
        assert_eq!(
            rig.world.resource::<RealmAuthority>().0,
            Some(Fence(1)),
            "the child affirm leaves the primary realm untouched",
        );
        // RE-AFFIRM the SAME child at a NEWER fence: the child is ALREADY in `CoHostedAuthority`, so this hits
        // the `else if ours` REFRESH arm (not the first insert). The stored fence must update to the refreshed
        // value — the periodic round-trip re-arming a still-held co-hosted child head.
        grant_realm_for_at(&mut rig, RealmId::Planet(7), Fence(2));
        assert_eq!(
            rig.world
                .resource::<CoHostedAuthority>()
                .0
                .get(&RealmId::Planet(7)),
            Some(&Fence(2)),
            "the re-affirm REFRESHES the co-hosted Planet-7 fence to the newer value",
        );
        assert_eq!(
            rig.world.resource::<RealmAuthority>().0,
            Some(Fence(1)),
            "the child re-affirm still leaves the primary realm untouched",
        );
    }

    #[test]
    fn a_revoked_cohosted_child_head_is_dropped_from_the_cohost_authority_map() {
        // The FOREIGN/None `else` arm of `affirm_realm_head` (`cohosted.0.remove(&realm)`): a co-hosted CHILD
        // realm this shard held is TAKEN OVER or REVOKED (here `record: None` — the reaper dropped the lease),
        // so its `CoHostedAuthority` entry is DROPPED (no transient loss — a child never anchors this shard's
        // transients; those ride the PRIMARY `RealmAuthority`). This is distinct from the primary self-fence
        // (which drops `RealmAuthority` + declares transients lost). Boot with Planet 7 HELD, then revoke it.
        let mut rig = Rig::with_config(cohost_planet_config());
        boot_cohost_planet(&mut rig); // primary System 7 + co-hosted child Planet 7 both held
        assert_eq!(
            rig.world
                .resource::<CoHostedAuthority>()
                .0
                .get(&RealmId::Planet(7)),
            Some(&Fence(1)),
            "precondition: the co-hosted Planet-7 head is held",
        );
        revoke_realm_for(&mut rig, RealmId::Planet(7)); // record None ⇒ not ours ⇒ the remove arm
        assert!(
            !rig.world
                .resource::<CoHostedAuthority>()
                .0
                .contains_key(&RealmId::Planet(7)),
            "the revoked co-hosted child is DROPPED from CoHostedAuthority",
        );
        // The PRIMARY realm is UNTOUCHED — the child-revoke path never self-fences the primary lease.
        assert_eq!(
            rig.world.resource::<RealmAuthority>().0,
            Some(Fence(1)),
            "revoking the co-hosted child leaves the primary System-7 realm held",
        );
    }

    #[test]
    fn a_dot_re_homing_into_a_cohosted_child_emits_a_crossing_request_with_its_parent() {
        // THE UNIVERSAL re-home assertion (task #149, re-baselined from the old relabel test): a durable dot
        // that walks from System 7 into CO-HOSTED Planet 7's SOI emits the SAME `CrossingRequest` as a
        // foreign crossing — there is NO local short-circuit. `head(Realm(Planet 7))` resolves to THIS node
        // (source==dest), which the ONE orchestrator saga handles as the degenerate case. The request carries
        // Planet 7 as `to_realm` and its enclosing System 7 as `to_parent` (the container region's parent) so
        // the dest's `rebind_pose_to_dest` forms the child frame. The pose is NOT rewritten in place here (the
        // detector only requests); the frame flips at the dest's adopt (same node on a co-hosted re-home).
        let mut rig = Rig::with_config(cohost_planet_config());
        boot_cohost_planet(&mut rig);
        let entity = EntityId::pack(EntityKind::Player, 10, 1, 51);
        // Start at the origin — inside System 7 (r=40), OUTSIDE Planet 7 (centre 20, r=10) ⇒ container ==
        // System 7 == owning ⇒ NO re-home.
        insert_owned_dot(&mut rig, TRIG_SESSION, entity, DVec3::ZERO);
        rig.set_local_tick(2);
        let at_origin = crossing_requests(&rig.tick(vec![]));
        assert_eq!(
            at_origin.len(),
            0,
            "at the origin the dot is in System 7 (== owning)"
        );
        // Walk INTO Planet 7's centre (20,0,0) — its deepest container flips to Planet 7, a realm THIS
        // shard CO-HOSTS ⇒ ONE CrossingRequest (the uniform saga; a co-hosted dest is source==dest).
        let mut out: Vec<(NodeId, MsgClass, Vec<u8>)> = Vec::new();
        for t in 3..8 {
            rig.set_local_tick(t);
            move_dot(&mut rig, TRIG_SESSION, DVec3::new(20.0, 0.0, 0.0));
            out.extend(rig.tick(vec![]));
        }
        let reqs = crossing_requests(&out);
        assert_eq!(
            reqs.len(),
            1,
            "a re-home into a CO-HOSTED child emits ONE CrossingRequest (source==dest — the uniform saga)",
        );
        assert_eq!(
            reqs[0].to_realm,
            RealmId::Planet(7),
            "the crossing targets the co-hosted Planet 7 realm",
        );
        assert_eq!(
            reqs[0].to_parent,
            Some(RealmId::System(7)),
            "the request carries Planet 7's enclosing System 7 as to_parent (so the dest's Area/child frame forms)",
        );
    }

    #[test]
    fn a_dot_re_homing_into_a_non_cohosted_child_emits_a_crossing_request() {
        // The SAME co-hosting shard, a dot walking into Station 7 — a child it does NOT co-host (`held_realms`
        // is `{System(7), Planet(7)}`, no Station). Post-task-#149 this is IDENTICAL to the co-hosted case:
        // ONE `CrossingRequest` (the node-placement branch is deleted — held-here vs foreign no longer
        // matters, both are the uniform saga). Kept as a second geometry to prove the request fires for any
        // container change, co-hosted or not.
        let mut rig = Rig::with_config(cohost_planet_config());
        boot_cohost_planet(&mut rig);
        // The region union for {System 7, Planet 7} DOES include Station 7 (a child of the held System 7).
        let entity = EntityId::pack(EntityKind::Player, 10, 1, 52);
        insert_owned_dot(&mut rig, TRIG_SESSION, entity, DVec3::ZERO);
        let mut out: Vec<(NodeId, MsgClass, Vec<u8>)> = Vec::new();
        for t in 3..9 {
            rig.set_local_tick(t);
            move_dot(&mut rig, TRIG_SESSION, DVec3::new(-25.0, 0.0, 0.0)); // Station 7 box centre
            out.extend(rig.tick(vec![]));
        }
        let reqs = crossing_requests(&out);
        assert_eq!(
            reqs.len(),
            1,
            "a re-home into a NON-co-hosted child emits ONE CrossingRequest (the uniform saga)",
        );
        assert_eq!(
            reqs[0].to_realm,
            RealmId::Station(7),
            "the crossing targets the Station 7 realm",
        );
        assert_eq!(
            reqs[0].to_parent,
            Some(RealmId::System(7)),
            "the request carries Station 7's enclosing System 7 as to_parent",
        );
    }

    #[test]
    fn a_held_transient_re_homing_into_a_cohosted_child_emits_a_transient_request_with_its_parent()
    {
        // The TRANSIENT twin (HR2 — the SAME machinery, no per-kind fork, no local short-circuit): a held
        // Debris transient inside CO-HOSTED Planet 7 emits ONE `TransientCrossingRequest` carrying Planet 7's
        // enclosing System 7 as `to_parent` (so the batch's `rebind_pose_to_dest` forms the child frame). The
        // pose is not rewritten in place; the frame flips at the dest's adopt.
        let mut rig = Rig::with_config(cohost_planet_config());
        boot_cohost_planet(&mut rig);
        let entity = EntityId::pack(EntityKind::Debris, 10, 1, 61);
        rig.world.resource_mut::<OwnedTransients>().0.insert(
            entity,
            Transient {
                // Inside Planet 7's SOI (centre x=20, r=10) — container == Planet 7 (co-hosted).
                pose: StampedPose::at_rest(
                    config().frame,
                    DVec3::new(20.0, 0.0, 0.0),
                    UniverseTick(100),
                ),
                anchor_fence: Fence(1),
                status: TransientStatus::Held { outbound: None },
                prev_offset: DVec3::new(20.0, 0.0, 0.0),
            },
        );
        let mut all: Vec<(NodeId, MsgClass, Vec<u8>)> = Vec::new();
        for t in 2..7 {
            rig.set_local_tick(t);
            all.extend(rig.tick(vec![]));
        }
        let reqs = transient_crossing_requests(&all);
        assert_eq!(
            reqs.len(),
            1,
            "a transient re-home into a CO-HOSTED child emits ONE TransientCrossingRequest (the uniform saga)",
        );
        assert_eq!(reqs[0].to_realm, RealmId::Planet(7));
        assert_eq!(
            reqs[0].to_parent,
            Some(RealmId::System(7)),
            "the transient request carries Planet 7's enclosing System 7 as to_parent",
        );
        assert_eq!(
            rig.world
                .resource::<StubStats>()
                .transient_crossings_requested,
            1,
        );
    }

    #[test]
    fn an_aborted_re_home_re_fires_while_the_dot_is_still_in_the_region() {
        // POSITIVE proof of the abort RE-FIRE (`on_crossing_aborted` resets `last_commit_tick=None`): a dot
        // whose crossing ABORTED is STILL geometrically in the deeper region, so `container != owning` and
        // `should_rehome` fires AGAIN with a FRESH id — the re-home self-heals without a physical re-cross.
        // A green tree that DISARMS the detector during the abort (the e2e tests) cannot catch a broken
        // re-fire; this drives the abort WITH the regions still armed and asserts the attempt-1 re-emit.
        let mut rig = Rig::new();
        rig.grant_realm();
        plant_dock_regions(&mut rig);
        let entity = EntityId::pack(EntityKind::Player, 10, 1, 9);
        insert_owned_dot(&mut rig, TRIG_SESSION, entity, DVec3::ZERO); // inside the child ⇒ re-homes
        rig.set_local_tick(2);
        let first = crossing_requests(&rig.tick(vec![]));
        assert_eq!(first.len(), 1, "the in-region dot re-homes once");
        assert_eq!(first[0].attempt, 0, "the first re-home is attempt 0");
        // The MATCHING abort clears the latch + resets the cooldown so the re-home can re-fire.
        let transfer = crossing_transfer_id(DirectoryKey::Entity(entity), Fence(1), 0);
        let mut after: Vec<(NodeId, MsgClass, Vec<u8>)> = Vec::new();
        rig.set_local_tick(3);
        after.extend(rig.tick(vec![wire_msg(
            ORCH,
            MsgClass::Saga,
            &InterShardFlow::CrossingAborted(CrossingAborted {
                subject: DirectoryKey::Entity(entity),
                transfer,
            }),
        )]));
        rig.set_local_tick(4);
        after.extend(rig.tick(vec![])); // still in the region ⇒ re-fires with a fresh id
        let refired = crossing_requests(&after).iter().any(|r| r.attempt == 1);
        assert!(
            refired,
            "after the abort the still-in-region dot RE-FIRES with attempt==1 (self-heal)",
        );
    }

    #[test]
    fn a_dot_jittering_across_a_region_surface_within_the_band_never_re_homes() {
        // The BAND (not the latch) proves anti-flap. A dot whose signed distance to the child region
        // oscillates ACROSS the surface (sd ∈ [-40, +40]) but stays inside the acquire-hysteresis dead-zone
        // (acquire only at sd ≤ -inset = -50) NEVER acquires membership, so `container` stays the OWN realm
        // and ZERO re-homes fire — latch-INDEPENDENT (no re-home ⇒ no latch to mask a flap). A broken band
        // (naive point membership `sd ≤ 0`) would acquire the child on every sd<0 tick and re-home.
        let mut rig = Rig::new();
        rig.grant_realm();
        plant_dock_regions(&mut rig); // child (Planet 42) at origin, r=1000, band inset=50/outset=100
        let entity = EntityId::pack(EntityKind::Player, 10, 1, 11);
        insert_owned_dot(&mut rig, TRIG_SESSION, entity, DVec3::new(1040.0, 0.0, 0.0)); // sd_child = +40
        let mut all: Vec<(NodeId, MsgClass, Vec<u8>)> = Vec::new();
        // Jitter x ∈ [960, 1040] ⇒ sd_child ∈ [-40, +40], crossing the surface but never reaching -inset,
        // for well over `k_dwell` ticks.
        for (i, &x) in [960.0, 1040.0, 970.0, 1030.0, 965.0, 1035.0, 962.0, 1038.0]
            .iter()
            .enumerate()
        {
            rig.set_local_tick(2 + i as u64);
            move_dot(&mut rig, TRIG_SESSION, DVec3::new(x, 0.0, 0.0));
            all.extend(rig.tick(vec![]));
        }
        assert!(
            crossing_requests(&all).is_empty(),
            "the acquire-hysteresis dead-zone keeps the dot a non-member ⇒ ZERO re-homes (band, not latch)",
        );
    }

    #[test]
    fn an_empty_registry_or_no_realm_triggers_nothing() {
        // (a) EMPTY region registry (the production-inert path): no candidates, no emit, no state.
        let mut rig = Rig::new();
        rig.grant_realm();
        let entity = EntityId::pack(EntityKind::Player, 10, 1, 5);
        insert_owned_dot(&mut rig, TRIG_SESSION, entity, DVec3::new(100.0, 0.0, 0.0));
        let mut all: Vec<(NodeId, MsgClass, Vec<u8>)> = Vec::new();
        for t in 2..8 {
            rig.set_local_tick(t);
            all.extend(rig.tick(vec![]));
        }
        assert!(crossing_requests(&all).is_empty());
        assert!(rig.world.resource::<RequestInFlight>().0.is_empty());
        assert!(rig.world.resource::<CrossingProgress>().0.is_empty());

        // (b) NO realm authority (never granted): the authority gate short-circuits BEFORE any work —
        // even with a populated registry + an in-band dot. A NON-ZERO `request_ttl_ticks` here so the
        // 3f-D4 `redrive_stranded_crossings` scan ALSO reaches (and covers) its authority-gate `else`
        // arm (past its own `ttl==0` early-return), mirroring the trigger's gate.
        let mut rig2 = Rig::with_config(config_with_ttl(3));
        plant_dock_regions(&mut rig2);
        // A dot cannot normally exist without a realm, but the trigger's gate must be authority-first:
        // force one in and confirm the early-return fires.
        insert_owned_dot(&mut rig2, TRIG_SESSION, entity, DVec3::new(100.0, 0.0, 0.0));
        assert_eq!(rig2.world.resource::<RealmAuthority>().0, None);
        let mut all2: Vec<(NodeId, MsgClass, Vec<u8>)> = Vec::new();
        for t in 2..8 {
            rig2.set_local_tick(t);
            all2.extend(rig2.tick(vec![]));
        }
        assert!(
            crossing_requests(&all2).is_empty(),
            "no realm authority ⇒ the trigger + the ttl re-drive both early-return",
        );
        assert!(rig2.world.resource::<CrossingProgress>().0.is_empty());
        assert_eq!(
            rig2.world.resource::<StubStats>().crossings_redriven,
            0,
            "no realm authority ⇒ the ttl scan never re-drives",
        );
    }

    #[test]
    fn eviction_drops_state_for_a_departed_subject() {
        let mut rig = Rig::new();
        rig.grant_realm();
        plant_dock_regions(&mut rig);
        let entity = EntityId::pack(EntityKind::Player, 10, 1, 6);
        insert_owned_dot(&mut rig, TRIG_SESSION, entity, DVec3::new(100.0, 0.0, 0.0));
        // Deep inside the child ⇒ a re-home fires, so ALL FOUR per-entity ledgers hold a row (the
        // ContainmentProgress bitset is written by the full-scan every tick a subject is evaluated).
        for t in 2..8 {
            rig.set_local_tick(t);
            let _ = rig.tick(vec![]);
        }
        assert!(
            rig.world
                .resource::<CrossingProgress>()
                .0
                .contains_key(&entity)
        );
        assert!(
            rig.world
                .resource::<RequestInFlight>()
                .0
                .contains_key(&entity)
        );
        assert!(
            rig.world
                .resource::<ContainmentProgress>()
                .0
                .contains_key(&entity),
            "the per-region membership bitset holds a row for the evaluated subject",
        );
        // The dot logs out (removed): next evaluation tick evicts its per-entity state (the DRY retain).
        rig.world.resource_mut::<Dots>().0.remove(&TRIG_SESSION);
        rig.set_local_tick(8);
        let _ = rig.tick(vec![]);
        assert!(
            !rig.world
                .resource::<CrossingProgress>()
                .0
                .contains_key(&entity)
        );
        assert!(
            !rig.world
                .resource::<RequestInFlight>()
                .0
                .contains_key(&entity)
        );
        assert!(
            !rig.world
                .resource::<ContainmentProgress>()
                .0
                .contains_key(&entity)
        );
    }

    // ---- Slice 4a (task #133): D-38 discharge + ledger-eviction + cell-carry SHAPE -----------------

    /// The DEEPER CHILD region as a Spherical Shell (SOI descent) — the shape the G-IDENTICAL fixture
    /// varies. Radius 1000, at the origin, nested under `OWN_REALM`, realm `OTHER_REALM`.
    fn child_region_shell() -> RealmRegion {
        child_region()
    }

    /// The DEEPER CHILD region as a Cartesian Aabb (station volume) — mirrors `child_region_shell`
    /// arg-for-arg (same realm/parent/center/band; the ONLY difference is the SHAPE). Half-extents 200 m.
    fn child_region_aabb() -> RealmRegion {
        region_box(
            OTHER_REALM,
            Some(OWN_REALM),
            DVec3::ZERO,
            DVec3::new(200.0, 200.0, 200.0),
        )
    }

    /// The ONE crossing-feature fixture, driven by a CHILD-REGION-CONSTRUCTOR closure so its BODY is
    /// written exactly once and run over two region shapes (§9's "same fixture on a Shell now, box at
    /// P4"). Plants root ⊃ own ⊃ CHILD(make_child()) in `RealmRegions`, inserts an OWNED DURABLE dot
    /// clearly OUTSIDE the child (signed_distance > 0 — no vacuous "already inside" pass), then drives it
    /// INWARD across the child's acquire edge on a DIAGONAL segment (a velocity with ≥2 non-zero axis
    /// components — so the Aabb run genuinely exercises `box_signed_distance`'s corner metric, which a
    /// pure axis-aligned approach would skip; a shell's `|p|` distance handles the same diagonal unchanged,
    /// so the body is identical). Returns the emitted `CrossingRequest`s + the dot's start/end SIGNED
    /// DISTANCE to the child (the anti-vacuity guarantee — the region actually GATED the re-home, it was
    /// not a no-op that fired on an already-inside dot) + the child shape.
    fn drive_inward_crossing_feature(
        make_child: impl Fn() -> RealmRegion,
    ) -> (Vec<CrossingRequest>, f64, f64, Boundary) {
        let mut rig = Rig::new();
        rig.grant_realm();
        let child = make_child();
        let to_realm = child.realm; // OTHER_REALM
        let shape = child.shape;
        *rig.world.resource_mut::<RealmRegions>() =
            RealmRegions::new(vec![root_region(), own_region(), child]);
        // A distinct durable Player, placed on a DIAGONAL well OUTSIDE the child region.
        let entity = EntityId::pack(EntityKind::Player, 10, 1, 0x4A);
        let outside = DVec3::new(1000.0, 1000.0, 0.0);
        insert_owned_dot(&mut rig, TRIG_SESSION, entity, outside);
        // Signed distance of the START pose to the child (proves it began OUTSIDE — positive distance).
        let start_sd = child_signed_distance(&child, outside);
        let mut all: Vec<(NodeId, MsgClass, Vec<u8>)> = Vec::new();
        // Approach inward along the diagonal, then hold deep inside (the band IS the dwell — one member
        // tick suffices, but hold a few so the run mirrors a real approach).
        let waypoints = [
            DVec3::new(700.0, 700.0, 0.0),
            DVec3::new(400.0, 400.0, 0.0),
            DVec3::new(50.0, 50.0, 0.0),
            DVec3::new(50.0, 50.0, 0.0),
            DVec3::new(50.0, 50.0, 0.0),
            DVec3::new(50.0, 50.0, 0.0),
        ];
        for (i, wp) in waypoints.iter().enumerate() {
            rig.set_local_tick(2 + i as u64);
            move_dot(&mut rig, TRIG_SESSION, *wp);
            all.extend(rig.tick(vec![]));
        }
        let inside = DVec3::new(50.0, 50.0, 0.0);
        let end_sd = child_signed_distance(&child, inside);
        // Only requests whose destination is THIS child's realm — filtering here proves the collection is
        // this feature's, not incidental noise.
        let reqs: Vec<CrossingRequest> = crossing_requests(&all)
            .into_iter()
            .filter(|r| r.to_realm == to_realm)
            .collect();
        (reqs, start_sd, end_sd, shape)
    }

    /// The signed distance from a frame-local `offset` to a region's surface (the exact scalar the
    /// containment band consumes). Under `IdentityFrames` at P3 the reframe is a no-op, so this is the
    /// same value the detector reads.
    fn child_signed_distance(region: &RealmRegion, offset: DVec3) -> f64 {
        use vd_core::geometry::region_signed_distance;
        let pose = StampedPose::at_rest(config().frame, offset, UniverseTick(100));
        region_signed_distance(&pose, region, &IdentityFrames)
            .expect("identity reframe never errors")
    }

    /// DISCHARGES D-38: the G-IDENTICAL assert_feature_anywhere — ONE containment re-home fixture,
    /// identical whether the deeper child region is a Spherical (Shell) or a Cartesian (Aabb) volume.
    #[test]
    fn assert_feature_anywhere() {
        // Run (a): a Spherical PLANET profile ⇔ a Shell child region (SOI descent). The dot descends the
        // diagonal across the shell's acquire edge and commits exactly ONE re-home to OTHER_REALM.
        let planet = crate::capability::profiles::planet().expect("planet profile");
        assert_eq!(
            planet.voxel(),
            Some(crate::capability::VoxelGeometry::Spherical),
            "the Shell run is tied to the Spherical profile",
        );
        let (shell_reqs, shell_start_sd, shell_end_sd, shell_shape) =
            drive_inward_crossing_feature(child_region_shell);
        // Spherical ⇔ Shell: the run's region IS a Shell (compared by equality, not `matches!`, so there
        // is no uncoverable false arm — HR5(d)). `child_region` uses r 1000.
        assert_eq!(shell_shape, Boundary::Shell { r: 1000.0 });
        // Run (b): a Cartesian STATION profile ⇔ an Aabb child region (station volume). The IDENTICAL body
        // drives the SAME diagonal descent across the box's acquire edge → exactly ONE re-home.
        let station = crate::capability::profiles::station().expect("station profile");
        assert_eq!(
            station.voxel(),
            Some(crate::capability::VoxelGeometry::Cartesian),
            "the Aabb run is tied to the Cartesian profile",
        );
        let (aabb_reqs, aabb_start_sd, aabb_end_sd, aabb_shape) =
            drive_inward_crossing_feature(child_region_aabb);
        // Cartesian ⇔ Aabb: the run's region IS an Aabb (equality, no `matches!` false arm — HR5(d)).
        assert_eq!(
            aabb_shape,
            Boundary::Aabb {
                half: DVec3::new(200.0, 200.0, 200.0),
            }
        );

        // (i) EXACTLY ONE re-home CrossingRequest per run, whose destination is the child's realm — the
        // box/shell actually GATED the re-home (not a bare count of "something fired").
        assert_eq!(
            shell_reqs.len(),
            1,
            "the Shell run emits exactly one re-home"
        );
        assert_eq!(aabb_reqs.len(), 1, "the Aabb run emits exactly one re-home");
        assert_eq!(
            shell_reqs[0].to_realm, OTHER_REALM,
            "the Shell re-home gated to OTHER_REALM"
        );
        assert_eq!(
            aabb_reqs[0].to_realm, OTHER_REALM,
            "the Aabb re-home gated to OTHER_REALM"
        );
        // IDENTICAL feature behavior across the two profiles: same subject, same source realm, same
        // destination, same attempt. The two runs differ ONLY in region shape, never in the feature.
        assert_eq!(shell_reqs[0].subject, aabb_reqs[0].subject);
        assert_eq!(shell_reqs[0].from_realm, aabb_reqs[0].from_realm);
        assert_eq!(shell_reqs[0].to_realm, aabb_reqs[0].to_realm);
        assert_eq!(shell_reqs[0].attempt, aabb_reqs[0].attempt);
        assert_eq!(shell_reqs[0].session, aabb_reqs[0].session);

        // (ii) Anti-vacuity: the signed distance ACTUALLY crossed the acquire edge in BOTH runs — the dot
        // started OUTSIDE the surface (sd > 0) and ended at least `inset` (50 m) INSIDE (sd <= -inset), a
        // real membership acquire, never a no-op that fired on an already-inside dot.
        assert!(
            shell_start_sd > 0.0,
            "Shell start OUTSIDE the surface (sd {shell_start_sd})"
        );
        assert!(
            shell_end_sd <= -50.0,
            "Shell end at least the inset INSIDE (sd {shell_end_sd})"
        );
        assert!(
            aabb_start_sd > 0.0,
            "Aabb start OUTSIDE the surface (sd {aabb_start_sd})"
        );
        assert!(
            aabb_end_sd <= -50.0,
            "Aabb end at least the inset INSIDE (sd {aabb_end_sd})"
        );
    }

    /// Slice 4a: the ledger-eviction-on-leave gate (the InputLog-leak class). The `retain_live` machinery
    /// already exists (`stub.rs` `evaluate_realm_boundaries`); this PROVES all THREE per-entity ledgers
    /// evicted by the detector (CrossingProgress + RequestInFlight + ContainmentProgress — the C-3 bitset
    /// is the 4th `retain_live` monomorphization that REPLACES the deleted `InterestZones`) are evicted
    /// when the subject leaves. Uses `assert_eq!(.get(), None)` (not `assert!(matches!)`) so the None
    /// equality is the covered arm.
    #[test]
    fn slice4a_all_three_ledgers_evict_when_the_subject_leaves() {
        let mut rig = Rig::new();
        rig.grant_realm();
        plant_dock_regions(&mut rig);
        let entity = EntityId::pack(EntityKind::Player, 10, 1, 0x4B);
        insert_owned_dot(&mut rig, TRIG_SESSION, entity, DVec3::new(100.0, 0.0, 0.0));
        // Deep inside the child ⇒ a durable re-home commits, so CrossingProgress (cooldown), RequestInFlight
        // (latch) AND ContainmentProgress (bitset) all hold a row for the subject.
        for t in 2..8 {
            rig.set_local_tick(t);
            let _ = rig.tick(vec![]);
        }
        // All three ledgers now hold an entry (setup pre-conditions; the load-bearing asserts are the
        // three `None` equalities after the subject leaves).
        assert!(
            rig.world
                .resource::<CrossingProgress>()
                .0
                .contains_key(&entity)
        );
        assert!(
            rig.world
                .resource::<RequestInFlight>()
                .0
                .contains_key(&entity)
        );
        assert!(
            rig.world
                .resource::<ContainmentProgress>()
                .0
                .contains_key(&entity)
        );
        // The dot logs out (removed from Dots). Not in OwnedTransients either, so `live` excludes it.
        rig.world.resource_mut::<Dots>().0.remove(&TRIG_SESSION);
        rig.set_local_tick(8);
        let _ = rig.tick(vec![]);
        // All THREE per-entity ledgers no longer contain the subject — split into three `is_none()`
        // asserts (HR5: never a collapsed `assert!(a && b)`), equality-form (not `matches!`).
        assert_eq!(
            rig.world.resource::<CrossingProgress>().0.get(&entity),
            None
        );
        assert_eq!(rig.world.resource::<RequestInFlight>().0.get(&entity), None);
        assert_eq!(
            rig.world.resource::<ContainmentProgress>().0.get(&entity),
            None
        );
    }

    /// Slice 4a (adversary MEDIUM-3 — the cell-carry SHAPE assertion, NOT an integer-gate flip). Proves
    /// the dot's `LatticePos.cell` (a DISTINCT NON-ZERO cell) is CARRIED UNCHANGED through the trigger
    /// AND the offset-based crossing still fires. This proves the cell is PRESERVED (shape only). It does
    /// NOT prove the "integer in/out DECISION flips across a cell boundary": `should_commit` /
    /// `evaluate_one_subject` read ONLY `LatticePos.offset()` today (`stub.rs`), never comparing the cell,
    /// so a cross-cell in/out DECISION test is BLOCKED on the P4/P5 rebase math (D-41) and is deliberately
    /// NOT written here.
    #[test]
    fn slice4a_nonzero_cell_is_carried_unchanged_and_the_offset_crossing_still_fires() {
        use vd_core::glam::I64Vec3;
        let mut rig = Rig::new();
        rig.grant_realm();
        plant_dock_regions(&mut rig);
        let entity = EntityId::pack(EntityKind::Player, 10, 1, 0x4C);
        // A DISTINCT non-zero cell anchor. The trigger reads only `offset()`, so the crossing decision is
        // driven by the (in-band) offset exactly as if the cell were zero.
        let cell = I64Vec3::new(5, -7, 11);
        let offset_inside = DVec3::new(100.0, 0.0, 0.0); // sd = 100 - 1000 ≪ -inset ⇒ inside the child
        rig.world.resource_mut::<Dots>().0.insert(
            TRIG_SESSION,
            Dot {
                entity,
                account: AccountId(1),
                session_fence: Fence(1),
                gateway: GATEWAY,
                granted: true,
                input_active: false,
                adopting: false,
                authority: Authority::Owned { fence: Fence(1) },
                departing: false,
                entity_fence: Fence(1),
                pose: StampedPose {
                    // Plant the non-zero cell anchor on an otherwise-rest pose (avoids naming DQuat).
                    pos: LatticePos::at(cell, offset_inside),
                    ..StampedPose::at_rest(config().frame, offset_inside, UniverseTick(100))
                },
                yaw: 0.0,
                pitch: 0.0,
                last_applied_seq: None,
                prev_offset: offset_inside,
            },
        );
        let mut all: Vec<(NodeId, MsgClass, Vec<u8>)> = Vec::new();
        for t in 2..8 {
            rig.set_local_tick(t);
            all.extend(rig.tick(vec![]));
        }
        // The OFFSET-based crossing still fires: exactly one inward CrossingRequest.
        assert_eq!(
            crossing_requests(&all).len(),
            1,
            "the offset-based crossing fires regardless of the (non-zero) cell anchor",
        );
        // The non-zero cell is CARRIED UNCHANGED through the trigger — PROVING the cell is PRESERVED
        // (shape). (It is NOT compared in the in/out decision — that cell-aware gate is P4/P5 D-41 math.)
        assert_eq!(
            rig.world
                .resource::<Dots>()
                .0
                .get(&TRIG_SESSION)
                .expect("the owned dot")
                .pose
                .pos
                .cell(),
            cell,
            "the non-zero LatticePos.cell is carried through the trigger unchanged (SHAPE preserved)",
        );
    }

    #[test]
    fn the_grant_demux_flips_a_source_transient_to_crossing() {
        let mut rig = Rig::new();
        rig.grant_realm();
        let entity = EntityId::pack(EntityKind::Debris, 10, 1, 7);
        rig.world.resource_mut::<OwnedTransients>().0.insert(
            entity,
            Transient {
                pose: transient_pose(),
                anchor_fence: Fence(1),
                status: TransientStatus::Held { outbound: None },
                prev_offset: transient_pose().pos.offset(),
            },
        );
        let grant = TransientCrossingGrant {
            subject: DirectoryKey::Entity(entity),
            dest: DEST_NODE,
            to_realm: OTHER_REALM,
            dst_realm_fence: Fence(3),
            batch: TransferId(77),
            to_parent: None,
        };
        let _ = rig.tick(vec![wire_msg(
            ORCH,
            MsgClass::Saga,
            &InterShardFlow::TransientCrossingGrant(grant),
        )]);
        // The grant flipped Held → Crossing; `emit_transient_batch` (later in the SAME schedule tick)
        // drained the Crossing into ONE batch and left it Held{outbound: Some(batch)}. The flip is
        // proven by BOTH the emit having fired and the resulting outbound tag.
        assert_eq!(
            rig.world.resource::<StubStats>().transient_grants_applied,
            1
        );
        assert_eq!(rig.world.resource::<StubStats>().transients_emitted, 1);
        assert_eq!(
            rig.world.resource::<OwnedTransients>().0[&entity].status,
            TransientStatus::Held {
                outbound: Some(TransferId(77)),
            },
            "the flipped Crossing was emitted, leaving Held{{outbound: Some(batch)}}",
        );
        // A REDELIVERED grant now finds Held{outbound: Some} (not a settled Held{None}) — a counted
        // no-op, so no re-flip + re-emit.
        let _ = rig.tick(vec![wire_msg(
            ORCH,
            MsgClass::Saga,
            &InterShardFlow::TransientCrossingGrant(grant),
        )]);
        assert_eq!(rig.world.resource::<StubStats>().transient_grant_noop, 1);
        assert_eq!(
            rig.world.resource::<StubStats>().transients_emitted,
            1,
            "the redelivered grant did NOT re-emit",
        );
    }

    #[test]
    fn the_crossing_aborted_demux_clears_the_latch_on_an_id_match_only() {
        let mut rig = Rig::new();
        rig.grant_realm();
        let entity = EntityId::pack(EntityKind::Player, 10, 1, 8);
        // A live owned dot for the entity so the trigger's eviction retain keeps its latch (the empty
        // registry means no crossing is triggered; the dot only keeps the subject alive).
        insert_owned_dot(&mut rig, TRIG_SESSION, entity, DVec3::new(0.0, 0.0, 0.0));
        let transfer = crossing_transfer_id(DirectoryKey::Entity(entity), Fence(1), 0);
        rig.world
            .resource_mut::<RequestInFlight>()
            .0
            .insert(entity, transfer);
        // A STALE abort (wrong id) preserves the latch (a superseded / re-latched crossing) — but STILL acks
        // (3f-D: the orchestrator keeps its durable entry until the ack, so every delivery must re-ack).
        let stale_out = rig.tick(vec![wire_msg(
            ORCH,
            MsgClass::Saga,
            &InterShardFlow::CrossingAborted(CrossingAborted {
                subject: DirectoryKey::Entity(entity),
                transfer: TransferId(999),
            }),
        )]);
        assert!(
            rig.world
                .resource::<RequestInFlight>()
                .0
                .contains_key(&entity),
            "a mismatched abort does NOT clear the latch",
        );
        assert_eq!(rig.world.resource::<StubStats>().crossing_abort_stale, 1);
        assert_eq!(
            crossing_aborted_acks(&stale_out).len(),
            1,
            "even a stale abort is acked (all paths ack — else the orch entry leaks)",
        );
        // The MATCHING abort clears the latch, acks, and RE-ARMS (bumps the attempt + resets the dwell).
        let match_out = rig.tick(vec![wire_msg(
            ORCH,
            MsgClass::Saga,
            &InterShardFlow::CrossingAborted(CrossingAborted {
                subject: DirectoryKey::Entity(entity),
                transfer,
            }),
        )]);
        assert!(
            !rig.world
                .resource::<RequestInFlight>()
                .0
                .contains_key(&entity)
        );
        assert_eq!(
            rig.world.resource::<StubStats>().crossing_latches_cleared,
            1
        );
        assert_eq!(
            crossing_aborted_acks(&match_out).len(),
            1,
            "the match acks too"
        );
        // The re-arm bumped the attempt (0 -> 1), so a re-cross mints a FRESH id (H2).
        assert_eq!(
            rig.world
                .resource::<CrossingProgress>()
                .0
                .get(&entity)
                .map(|st| st.crossing_attempt),
            Some(1),
        );
    }

    /// 3f-D4 config: the shared trigger fixture with a NON-ZERO `request_ttl_ticks` so
    /// `redrive_stranded_crossings` is armed (every other rig uses the INERT `0`).
    fn config_with_ttl(ttl: u32) -> StubConfig {
        StubConfig {
            request_ttl_ticks: ttl,
            ..config()
        }
    }

    #[test]
    fn a_stranded_durable_latch_redrives_after_the_ttl() {
        // 3f-D4: a delivered-but-unresolved dest leaves the durable latch STANDING with no rising edge
        // (the dot dwells statically in-band; `evaluate_realm_boundaries`'s `Occupied` arm only
        // suppresses). The per-tick latch-scan MUST re-emit the SAME `CrossingRequest` once the ttl
        // elapses — the C1 fix (the `Entry::Occupied` re-drive the DEFERRED text prescribed is
        // unreachable for a static dweller).
        const TTL: u32 = 3;
        let mut rig = Rig::with_config(config_with_ttl(TTL));
        rig.grant_realm();
        plant_dock_regions(&mut rig);
        let entity = EntityId::pack(EntityKind::Player, 10, 1, 42);
        // Deep inside the child from spawn (sd = 100 - 1000 ≪ -inset) so the first commit latches it; there
        // is no orchestrator in this rig, so the latch is never cleared → it STRANDS (the tested condition).
        insert_owned_dot(&mut rig, TRIG_SESSION, entity, DVec3::new(100.0, 0.0, 0.0));

        // Inside the child from tick 2 ⇒ the re-home commits on the FIRST evaluated tick (the band IS the
        // dwell) → the durable `Vacant` arm emits ONE request + arms `last_commit_tick = 2`.
        let mut first: Vec<(NodeId, MsgClass, Vec<u8>)> = Vec::new();
        for t in 2..=4 {
            rig.set_local_tick(t);
            first.extend(rig.tick(vec![]));
        }
        let first_reqs = crossing_requests(&first);
        assert_eq!(
            first_reqs.len(),
            1,
            "exactly ONE request from the rising edge"
        );
        assert_eq!(
            first_reqs[0].attempt, 0,
            "the first crossing uses attempt 0"
        );
        let latched_id = crossing_transfer_id(DirectoryKey::Entity(entity), Fence(1), 0);
        assert_eq!(
            rig.world.resource::<RequestInFlight>().0.get(&entity),
            Some(&latched_id),
            "the latch is held (no orchestrator ever cleared it)",
        );
        // The commit armed `last_commit_tick = 2`. Ticks 3/4 (already ticked in the loop above) were still
        // < TTL, so no re-drive fired there — proven by `first_reqs.len() == 1`.
        assert_eq!(
            rig.world.resource::<StubStats>().crossings_redriven,
            0,
            "the ttl window has NOT elapsed yet through tick 4 (the `>= ttl` false arm)",
        );

        // Tick 5: local_tick - 2 == 3 >= TTL → the scan re-emits the SAME request (the `>= ttl` TRUE arm).
        rig.set_local_tick(5);
        let redrive_out = rig.tick(vec![]);
        let redrive_reqs = crossing_requests(&redrive_out);
        assert_eq!(
            redrive_reqs.len(),
            1,
            "the ttl re-drive re-emitted exactly one request"
        );
        // The re-drive is BYTE-IDENTICAL to the standing latch: same subject/to_realm/fence/session/attempt.
        assert_eq!(redrive_reqs[0].subject, DirectoryKey::Entity(entity));
        assert_eq!(redrive_reqs[0].to_realm, OTHER_REALM);
        assert_eq!(redrive_reqs[0].from_realm, config().realm);
        assert_eq!(redrive_reqs[0].subject_fence, Fence(1));
        assert_eq!(redrive_reqs[0].session, TRIG_SESSION);
        assert_eq!(redrive_reqs[0].attempt, 0, "same attempt → same id");
        assert_eq!(
            crossing_transfer_id(
                redrive_reqs[0].subject,
                redrive_reqs[0].subject_fence,
                redrive_reqs[0].attempt,
            ),
            latched_id,
            "the re-emitted id equals the standing latch id (idempotent at the orchestrator)",
        );
        assert!(
            rig.world.resource::<StubStats>().crossings_redriven >= 1,
            "the re-drive counter bumped",
        );
        assert_eq!(
            rig.world.resource::<RequestInFlight>().0.get(&entity),
            Some(&latched_id),
            "the latch is STILL held after the re-drive (only a saga terminal clears it)",
        );

        // Tick 6: the re-drive re-armed `last_commit_tick = 5`, so 6 - 5 == 1 < TTL → NO third emit yet
        // (the `>= ttl` false arm again, proving the timer re-arms and does not storm every tick).
        rig.set_local_tick(6);
        let after = rig.tick(vec![]);
        assert_eq!(
            crossing_requests(&after).len(),
            0,
            "the re-drive re-armed the ttl timer — no storm before the next window",
        );
    }

    #[test]
    fn the_ttl_redrive_is_inert_when_request_ttl_ticks_is_zero() {
        // 3f-D4: with the DEFAULT `request_ttl_ticks == 0` the scan EARLY-RETURNS (the inert default of
        // every current rig) — a held latch is NEVER re-driven, even past many ticks.
        let mut rig = Rig::new(); // request_ttl_ticks == 0
        rig.grant_realm();
        plant_dock_regions(&mut rig);
        let entity = EntityId::pack(EntityKind::Player, 10, 1, 43);
        insert_owned_dot(&mut rig, TRIG_SESSION, entity, DVec3::new(100.0, 0.0, 0.0));

        // Commit on the first in-band tick, then advance well past any plausible ttl window.
        let mut all: Vec<(NodeId, MsgClass, Vec<u8>)> = Vec::new();
        for t in 2..=40 {
            rig.set_local_tick(t);
            all.extend(rig.tick(vec![]));
        }
        assert_eq!(
            crossing_requests(&all).len(),
            1,
            "exactly the ONE rising-edge request — the ttl scan is inert (never re-drives)",
        );
        assert_eq!(
            rig.world.resource::<StubStats>().crossings_redriven,
            0,
            "the ttl==0 early-return means a held latch is never re-driven",
        );
        assert!(
            rig.world
                .resource::<RequestInFlight>()
                .0
                .contains_key(&entity),
            "the latch stands (no positive clear), but is never re-emitted",
        );
    }

    #[test]
    fn a_crossing_aborted_for_a_non_entity_subject_still_acks() {
        // The THIRD unconditional-ack path (3f-D): a `CrossingAborted` whose subject is not an Entity (a
        // malformed/realm subject) has no latch to clear, but MUST still ack — else the orchestrator's
        // durable `pending_abort_replies` entry would leak.
        let mut rig = Rig::new();
        rig.grant_realm();
        let out = rig.tick(vec![wire_msg(
            ORCH,
            MsgClass::Saga,
            &InterShardFlow::CrossingAborted(CrossingAborted {
                subject: DirectoryKey::Realm(RealmId::System(7)),
                transfer: TransferId(5),
            }),
        )]);
        assert_eq!(
            rig.world.resource::<StubStats>().crossing_abort_no_entity,
            1
        );
        assert_eq!(
            crossing_aborted_acks(&out).len(),
            1,
            "the no-entity path acks too",
        );
    }

    #[test]
    fn a_saga_demote_clears_the_durable_crossing_latch() {
        let mut rig = Rig::new();
        rig.grant_realm();
        plant_dock_regions(&mut rig);
        let entity = EntityId::pack(EntityKind::Player, 10, 1, 9);
        insert_owned_dot(&mut rig, TRIG_SESSION, entity, DVec3::new(100.0, 0.0, 0.0));
        for t in 2..8 {
            rig.set_local_tick(t);
            let _ = rig.tick(vec![]);
        }
        assert!(
            rig.world
                .resource::<RequestInFlight>()
                .0
                .contains_key(&entity)
        );
        // The durable COMMIT terminal at the source (the saga-pushed Demote) POSITIVELY clears it.
        let demote = DemoteCmd {
            transfer: crossing_transfer_id(DirectoryKey::Entity(entity), Fence(1), 0),
            subject: DirectoryKey::Entity(entity),
            new_owner_fence: Fence(2),
            step_id: DEMOTE_STEP,
        };
        let _ = rig.tick(vec![wire_msg(
            ORCH,
            MsgClass::Saga,
            &InterShardFlow::Demote(demote),
        )]);
        assert!(
            !rig.world
                .resource::<RequestInFlight>()
                .0
                .contains_key(&entity),
            "the durable Demote terminal cleared the crossing latch",
        );
        assert!(rig.world.resource::<StubStats>().crossing_latches_cleared >= 1);
    }

    /// Drive an entity OUTWARD across a shell in ONE move (prev inside → cur far outside ⇒ swept
    /// Outward, which `should_commit` commits IMMEDIATELY, no dwell). Returns the outbound frames of the
    /// exit tick. Shared by the nested-undock and top-level-degrade tests so they differ ONLY in the
    /// planted boundary's `parent`.
    fn drive_outward_exit(rig: &mut Rig, entity: EntityId) -> Vec<(NodeId, MsgClass, Vec<u8>)> {
        // ESCAPE forest: root ⊃ parent(PARENT_REALM) ⊃ own(OWN_REALM, small shell r=1000). The shard owns
        // OWN_REALM; a dot leaving the small own-shell (but inside the parent) undocks OUTWARD to the parent.
        plant_escape_regions(rig);
        // Start INSIDE the own-shell (sd = 500 - 1000 ≪ -inset ⇒ container == OWN_REALM, no re-home yet).
        insert_owned_dot(rig, TRIG_SESSION, entity, DVec3::new(500.0, 0.0, 0.0));
        // Prime membership with one in-band tick.
        rig.set_local_tick(2);
        let _ = rig.tick(vec![]);
        // Step OUTSIDE the own-shell (sd = 5000 - 1000 ≫ +outset ⇒ own membership releases) but still
        // deep inside the PARENT shell ⇒ container flips to PARENT_REALM ⇒ re-home OUTWARD to the parent.
        move_dot(rig, TRIG_SESSION, DVec3::new(5000.0, 0.0, 0.0));
        rig.set_local_tick(3);
        rig.tick(vec![])
    }

    #[test]
    fn slice4b_an_undock_re_homes_outward_to_the_parent_realm() {
        // Slice 4b (retargeted to CONTAINMENT) — the UNDOCK leg: the shard OWNS a CHILD realm (OWN_REALM,
        // a small shell nested under PARENT_REALM). An entity leaving the child region (container flips to
        // the parent) re-homes OUTWARD to PARENT_REALM. Symmetric with the dock — the SAME machinery; there
        // is no "direction", only "the container changed".
        let mut rig = Rig::new();
        rig.grant_realm();
        let entity = EntityId::pack(EntityKind::Player, 10, 2, 1);
        let out = drive_outward_exit(&mut rig, entity);
        let reqs = crossing_requests(&out);
        assert_eq!(
            reqs.len(),
            1,
            "leaving the owned child region re-homes exactly once",
        );
        // THE LOAD-BEARING ASSERT: the undock re-homes to the PARENT (the deepest region still containing
        // the dot after it left the child). NON-VACUOUS because PARENT_REALM != OWN_REALM != OTHER_REALM.
        assert_eq!(
            reqs[0].to_realm, PARENT_REALM,
            "the undock re-homes OUTWARD to the parent realm (the new deepest container)",
        );
        assert_ne!(
            reqs[0].to_realm, OTHER_REALM,
            "the undock destination is NOT the inward child interior",
        );
        assert_eq!(
            reqs[0].from_realm, OWN_REALM,
            "the re-home leaves the realm the shard owns the subject in",
        );
    }

    #[test]
    fn slice4b_a_dock_and_undock_resolve_contrasting_destinations() {
        // Slice 4b (retargeted to CONTAINMENT) — DOCK vs UNDOCK contrast: entering a DEEPER child region
        // re-homes to the child interior (OTHER_REALM); leaving an OWNED child region re-homes to the
        // parent (PARENT_REALM). Both fall out of `container()`; the two destinations DIFFER (no accidental
        // alias) — proving the symmetric rule distinguishes the two container changes without a direction.

        // DOCK (into a deeper child) — the standard dock forest; the dot inside the child re-homes to it.
        let mut dock = Rig::new();
        dock.grant_realm();
        plant_dock_regions(&mut dock);
        let dock_entity = EntityId::pack(EntityKind::Player, 10, 2, 2);
        // Deep inside the child from spawn (sd ≪ -inset) → container == OTHER_REALM ≠ own ⇒ re-home INWARD.
        insert_owned_dot(
            &mut dock,
            TRIG_SESSION,
            dock_entity,
            DVec3::new(100.0, 0.0, 0.0),
        );
        let mut dock_out: Vec<(NodeId, MsgClass, Vec<u8>)> = Vec::new();
        for t in 2..8 {
            dock.set_local_tick(t);
            dock_out.extend(dock.tick(vec![]));
        }
        let dock_reqs = crossing_requests(&dock_out);
        assert_eq!(dock_reqs.len(), 1, "the dock commits exactly one re-home");
        assert_eq!(
            dock_reqs[0].to_realm, OTHER_REALM,
            "the dock docks INWARD to the deeper child region's realm",
        );

        // UNDOCK (out of an owned child) — the escape forest; leaving the own-shell undocks to the parent.
        let mut undock = Rig::new();
        undock.grant_realm();
        let undock_entity = EntityId::pack(EntityKind::Player, 10, 2, 3);
        let undock_out = drive_outward_exit(&mut undock, undock_entity);
        let undock_reqs = crossing_requests(&undock_out);
        assert_eq!(
            undock_reqs.len(),
            1,
            "the undock commits exactly one re-home"
        );
        assert_eq!(
            undock_reqs[0].to_realm, PARENT_REALM,
            "the undock re-homes OUTWARD to the parent realm",
        );

        // THE CONTRAST: the two container changes resolve DISTINCT destinations.
        assert_ne!(
            dock_reqs[0].to_realm, undock_reqs[0].to_realm,
            "a dock and an undock resolve DIFFERENT destinations (deeper child vs parent)",
        );
    }

    #[test]
    fn a_nested_inner_boundary_wins_over_its_parent() {
        // Retargeted to CONTAINMENT (the `deepest_wins` property): a dot inside BOTH the own region
        // (System(7)) AND a coincident DEEPER child (Planet(42), nested under own) has container == the
        // INNER child (depth beats the parent), so the re-home targets the inner realm — exercising the
        // boot-cached `region_depth` argmax.
        let mut rig = Rig::new();
        rig.grant_realm();
        plant_dock_regions(&mut rig); // root ⊃ own(System 7) ⊃ child(Planet 42), all origin-coincident
        let entity = EntityId::pack(EntityKind::Player, 10, 2, 5);
        // At |100| the dot is inside own (100000) AND the child (1000): the deeper child wins.
        insert_owned_dot(&mut rig, TRIG_SESSION, entity, DVec3::new(100.0, 0.0, 0.0));
        let mut all: Vec<(NodeId, MsgClass, Vec<u8>)> = Vec::new();
        for t in 2..8 {
            rig.set_local_tick(t);
            all.extend(rig.tick(vec![]));
        }
        let reqs = crossing_requests(&all);
        assert_eq!(reqs.len(), 1);
        assert_eq!(
            reqs[0].to_realm, OTHER_REALM,
            "the INNER (deeper) region wins the depth argmax",
        );
    }

    #[test]
    fn a_boundary_hovering_dot_does_not_flap() {
        // Retargeted to CONTAINMENT (`hysteresis_no_flap`): a dot jittering INSIDE the child region's band
        // for many ticks stays a member (its bit never releases), so container is stable and the latch caps
        // the emits at ≤ 1 across the whole window (no per-tick flap).
        let mut rig = Rig::new();
        rig.grant_realm();
        plant_dock_regions(&mut rig);
        let entity = EntityId::pack(EntityKind::Player, 10, 2, 2);
        // Hover deep inside the child (sd stays ≪ -inset even with the jitter): a member every tick.
        insert_owned_dot(&mut rig, TRIG_SESSION, entity, DVec3::new(200.0, 0.0, 0.0));
        let mut all: Vec<(NodeId, MsgClass, Vec<u8>)> = Vec::new();
        for t in 2..15 {
            rig.set_local_tick(t);
            // Jitter within the child region each tick (still a member: sd stays deeply negative).
            let jitter = if t % 2 == 0 { 210.0 } else { 190.0 };
            move_dot(&mut rig, TRIG_SESSION, DVec3::new(jitter, 0.0, 0.0));
            all.extend(rig.tick(vec![]));
        }
        assert!(
            crossing_requests(&all).len() <= 1,
            "a region-hovering dot emits at most one request (no flap)",
        );
    }

    #[test]
    fn a_non_entity_grant_and_abort_are_counted_no_ops() {
        // The `transfer_subject_entity() == None` arms of the two demuxes (a Realm subject).
        let mut rig = Rig::new();
        rig.grant_realm();
        let _ = rig.tick(vec![wire_msg(
            ORCH,
            MsgClass::Saga,
            &InterShardFlow::TransientCrossingGrant(TransientCrossingGrant {
                subject: DirectoryKey::Realm(RealmId::System(7)),
                dest: DEST_NODE,
                to_realm: OTHER_REALM,
                dst_realm_fence: Fence(3),
                batch: TransferId(1),
                to_parent: None,
            }),
        )]);
        assert_eq!(
            rig.world.resource::<StubStats>().transient_grant_no_entity,
            1
        );
        let _ = rig.tick(vec![wire_msg(
            ORCH,
            MsgClass::Saga,
            &InterShardFlow::CrossingAborted(CrossingAborted {
                subject: DirectoryKey::Realm(RealmId::System(7)),
                transfer: TransferId(1),
            }),
        )]);
        assert_eq!(
            rig.world.resource::<StubStats>().crossing_abort_no_entity,
            1
        );
    }

    #[test]
    fn a_grant_for_an_unknown_transient_is_a_counted_no_op() {
        let mut rig = Rig::new();
        rig.grant_realm();
        let entity = EntityId::pack(EntityKind::Debris, 10, 2, 3);
        let _ = rig.tick(vec![wire_msg(
            ORCH,
            MsgClass::Saga,
            &InterShardFlow::TransientCrossingGrant(TransientCrossingGrant {
                subject: DirectoryKey::Entity(entity),
                dest: DEST_NODE,
                to_realm: OTHER_REALM,
                dst_realm_fence: Fence(3),
                batch: TransferId(1),
                to_parent: None,
            }),
        )]);
        assert_eq!(rig.world.resource::<StubStats>().transient_grant_noop, 1);
    }

    #[test]
    fn a_dot_outside_the_deeper_child_but_inside_own_stays() {
        // Retargeted from the portal "StaysOutside is not a candidate": a dot that is OUTSIDE the deeper
        // child region (its bit never acquires) but inside the OWN region has container == OWN_REALM — the
        // realm the shard owns it in — so NO re-home fires and no crossing latch is created.
        let mut rig = Rig::new();
        rig.grant_realm();
        plant_dock_regions(&mut rig);
        let entity = EntityId::pack(EntityKind::Player, 10, 2, 4);
        insert_owned_dot(&mut rig, TRIG_SESSION, entity, DVec3::new(9000.0, 0.0, 0.0));
        let mut all: Vec<(NodeId, MsgClass, Vec<u8>)> = Vec::new();
        for t in 2..8 {
            rig.set_local_tick(t);
            // Move around, but always well outside the child region (and inside own).
            move_dot(
                &mut rig,
                TRIG_SESSION,
                DVec3::new(9000.0 + t as f64, 0.0, 0.0),
            );
            all.extend(rig.tick(vec![]));
        }
        assert!(crossing_requests(&all).is_empty());
        assert!(rig.world.resource::<RequestInFlight>().0.is_empty());
        // The subject is evaluated every tick (so it carries a ContainmentProgress bit + a CrossingProgress
        // cooldown row), but since container == own NO re-home ever COMMITTED — the cooldown was never
        // armed (the `last_commit_tick` is None).
        assert_eq!(
            rig.world
                .resource::<CrossingProgress>()
                .0
                .get(&entity)
                .and_then(|st| st.last_commit_tick),
            None,
            "no re-home committed ⇒ the cooldown was never armed",
        );
    }

    #[test]
    #[should_panic(expected = "valid BoundaryTuning")]
    fn a_zero_dwell_boundary_tuning_fails_loud_at_boot() {
        let bad = StubConfig {
            boundary: BoundaryTuning {
                k_dwell: 0,
                ..BoundaryTuning::DEFAULT
            },
            ..config()
        };
        // The fail-loud validation in `register_stub_shard` (mirrors the tick-pair guard). `k_dwell` is the
        // post-commit cooldown `should_rehome` still reads (§2.7); zero fails `BoundaryTuning::validate`.
        let _ = Rig::with_config(bad);
    }

    // ===== RLM Step 2 — demand-driven realm lifecycle (AoI) ==========================================

    /// A live AoI band (base 1000 ⇒ spin_up 1000 m, tear_down 2000 m, `grace` ticks). The velocity-safe
    /// ctor is the only way to build a live band; `v_rel = 0` here (the tests place occupants by geometry).
    fn aoi_band(grace: u32) -> vd_core::geometry::AoiConfig {
        vd_core::geometry::AoiConfig::for_velocity_safe(1000.0, 1.0, 2.0, 0.0, 0.05, grace, 0.0)
            .expect("0 < spin_up < tear_down ⇒ a valid live band")
    }

    /// A Shell region with an EXPLICIT frame — `region`/`frame_of` resolves via `frame_for_realm(realm,
    /// None)`, which returns `None` for an `Area` (Area needs its `Planet` parent); the HR4 fixture
    /// supplies the frame here.
    fn region_framed(
        realm: RealmId,
        parent: Option<RealmId>,
        center: DVec3,
        r: f64,
        frame: FrameRef,
    ) -> RealmRegion {
        RealmRegion {
            realm,
            center: LatticePos::local(center),
            frame,
            shape: Boundary::Shell { r },
            band: band(),
            aoi: vd_core::geometry::AoiConfig::inert(),
            parent,
        }
    }

    /// A small direct-child region (containment shell radius `r`) carrying a LIVE AoI band — frame via
    /// `frame_of`. An occupant OUTSIDE the small containment shell but INSIDE the 1000 m AoI is the
    /// "reaches before it enters" case the whole feature exists for.
    fn aoi_child(realm: RealmId, parent: RealmId, r: f64, grace: u32) -> RealmRegion {
        RealmRegion {
            aoi: aoi_band(grace),
            ..region(realm, Some(parent), DVec3::ZERO, r)
        }
    }

    /// Like [`aoi_child`] but with an EXPLICIT frame (for the `Area` child `frame_of` cannot resolve).
    fn aoi_child_framed(
        realm: RealmId,
        parent: Option<RealmId>,
        frame: FrameRef,
        grace: u32,
    ) -> RealmRegion {
        RealmRegion {
            aoi: aoi_band(grace),
            ..region_framed(realm, parent, DVec3::ZERO, 100.0, frame)
        }
    }

    fn plant_aoi(rig: &mut Rig, regions: Vec<RealmRegion>) {
        *rig.world.resource_mut::<RealmRegions>() = RealmRegions::new(regions);
    }

    /// Every `RealmDemand` in the outbox (decoded). The `_ => None` arm is exercised too — an AoI tick also
    /// ships entity snapshots (non-`InterShardFlow` bytes) alongside the demand.
    fn demands(sent: &[(NodeId, MsgClass, Vec<u8>)]) -> Vec<RealmDemand> {
        sent.iter()
            .filter_map(
                |(_, _, b)| match postcard::from_bytes::<InterShardFlow>(b) {
                    Ok(InterShardFlow::RealmDemand(d)) => Some(d),
                    _ => None,
                },
            )
            .collect()
    }

    /// The child coord the loop names for `own_realm`'s child `child_realm` — `own_coord.child(level)`.
    fn child_coord_of(own_realm: RealmId, child_realm: RealmId) -> RealmCoord {
        StubConfig::root_coord(own_realm).child(level_of(child_realm).expect("seed-lineage child"))
    }

    fn player(tag: u32) -> EntityId {
        EntityId::pack(EntityKind::Player, 10, 1, tag)
    }

    #[test]
    fn aoi_transition_covers_every_hysteresis_arm() {
        let g = 3;
        // (false,true) ACQUIRE ⇒ SpinUp, grace armed.
        assert_eq!(
            aoi_transition(AoiState::default(), true, g),
            (
                Some(DemandVerb::SpinUp),
                Some(AoiState {
                    was_in: true,
                    grace_remaining: g
                })
            ),
        );
        // (true,true) HOLD-IN ⇒ KeepAlive, grace re-armed.
        assert_eq!(
            aoi_transition(
                AoiState {
                    was_in: true,
                    grace_remaining: 1
                },
                true,
                g
            ),
            (
                Some(DemandVerb::KeepAlive),
                Some(AoiState {
                    was_in: true,
                    grace_remaining: g
                })
            ),
        );
        // (true,false) grace > 0 ⇒ KeepAlive, grace decrements.
        assert_eq!(
            aoi_transition(
                AoiState {
                    was_in: true,
                    grace_remaining: 2
                },
                false,
                g
            ),
            (
                Some(DemandVerb::KeepAlive),
                Some(AoiState {
                    was_in: true,
                    grace_remaining: 1
                })
            ),
        );
        // (true,false) grace == 0 ⇒ DROP the key, NO demand (never a TearDown — M-1).
        assert_eq!(
            aoi_transition(
                AoiState {
                    was_in: true,
                    grace_remaining: 0
                },
                false,
                g
            ),
            (None, None),
        );
        // (false,false) never-in ⇒ nothing.
        assert_eq!(aoi_transition(AoiState::default(), false, g), (None, None));
    }

    #[test]
    fn aoi_min_dist_takes_the_lesser_of_live_and_predictive() {
        let child = DVec3::ZERO;
        // A STATIC occupant: pred == live == its distance.
        assert_eq!(
            aoi_min_dist(&[(DVec3::new(500.0, 0.0, 0.0), DVec3::ZERO)], child, 1.0),
            500.0
        );
        // A MOVING occupant closing in: pred (300) beats live (1500).
        let occ = (DVec3::new(1500.0, 0.0, 0.0), DVec3::new(-1200.0, 0.0, 0.0));
        assert_eq!(aoi_min_dist(&[occ], child, 1.0), 300.0);
        // The MIN across two occupants — order-independent (H-2).
        let a = (DVec3::new(900.0, 0.0, 0.0), DVec3::ZERO);
        let b = (DVec3::new(400.0, 0.0, 0.0), DVec3::ZERO);
        assert_eq!(aoi_min_dist(&[a, b], child, 1.0), 400.0);
        assert_eq!(aoi_min_dist(&[b, a], child, 1.0), 400.0);
    }

    #[test]
    fn region_level_recovers_seed_lineage_kinds() {
        use vd_core::realm_path::{RealmKindTag, RealmLevel};
        assert_eq!(
            region_level(&region(
                RealmId::Planet(42),
                Some(OWN_REALM),
                DVec3::ZERO,
                1.0
            )),
            RealmLevel::new(RealmKindTag::Planet, 42)
        );
        assert_eq!(
            region_level(&region(
                RealmId::System(7),
                Some(ROOT_REALM),
                DVec3::ZERO,
                1.0
            )),
            RealmLevel::new(RealmKindTag::System, 7)
        );
    }

    #[test]
    fn child_placements_unifies_movers_and_static() {
        // A STATIC direct child of OWN (not in the moving roster) rides its region center — place_child's
        // `None` arm — and is_direct_child accepts ONLY the OWN child (root/own excluded).
        let regions = RealmRegions::new(vec![
            root_region(),
            own_region(),
            region(
                OTHER_REALM,
                Some(OWN_REALM),
                DVec3::new(10.0, 0.0, 0.0),
                100.0,
            ),
        ]);
        let placements = regions.child_placements(OWN_REALM, 20.0, UniverseTick(5));
        assert_eq!(placements.len(), 1);
        assert_eq!(placements[0].0.realm, OTHER_REALM);
        assert_eq!(placements[0].1.pos.offset(), DVec3::new(10.0, 0.0, 0.0));
        assert_eq!(placements[0].1.vel, DVec3::ZERO);
    }

    #[test]
    fn evaluate_realm_aoi_inert_without_authority() {
        // No realm lease ⇒ the authority `else` short-circuits ⇒ no demand (even with a live child + dot).
        let mut rig = Rig::new();
        plant_aoi(
            &mut rig,
            vec![
                root_region(),
                own_region(),
                aoi_child(OTHER_REALM, OWN_REALM, 100.0, 3),
            ],
        );
        insert_owned_dot(
            &mut rig,
            TRIG_SESSION,
            player(7),
            DVec3::new(500.0, 0.0, 0.0),
        );
        assert!(demands(&rig.tick(vec![])).is_empty());
    }

    #[test]
    fn evaluate_realm_aoi_inert_empty_regions() {
        // Granted, but NO regions ⇒ the `is_empty` guard short-circuits ⇒ no demand.
        let mut rig = Rig::new();
        rig.grant_realm();
        insert_owned_dot(
            &mut rig,
            TRIG_SESSION,
            player(7),
            DVec3::new(500.0, 0.0, 0.0),
        );
        assert!(demands(&rig.tick(vec![])).is_empty());
    }

    #[test]
    fn evaluate_realm_aoi_unsynced_authors_nothing() {
        // D-Finding-1: a shard whose clock is NOT yet synced authors nothing (`has_synced` skips the
        // system) — no pre-sync demand even with authority + a live child + an in-range occupant.
        let mut rig = Rig::new();
        rig.grant_realm();
        rig.world.resource_mut::<ClockSample>().synced = false;
        plant_aoi(
            &mut rig,
            vec![
                root_region(),
                own_region(),
                aoi_child(OTHER_REALM, OWN_REALM, 100.0, 3),
            ],
        );
        insert_owned_dot(
            &mut rig,
            TRIG_SESSION,
            player(7),
            DVec3::new(500.0, 0.0, 0.0),
        );
        assert!(demands(&rig.tick(vec![])).is_empty());
    }

    #[test]
    fn evaluate_realm_aoi_empty_self_report() {
        // Zero occupants ⇒ the CHILD shard self-reports its OWN realm holds nobody: exactly one
        // `Empty { child = own_coord }` (Step-3's occupancy authority — a sealed parent cannot see inside).
        let mut rig = Rig::new();
        rig.grant_realm();
        plant_aoi(
            &mut rig,
            vec![
                root_region(),
                own_region(),
                aoi_child(OTHER_REALM, OWN_REALM, 100.0, 3),
            ],
        );
        assert_eq!(
            demands(&rig.tick(vec![])),
            vec![RealmDemand {
                child: StubConfig::root_coord(OWN_REALM),
                parent_fence: Fence(1),
                verb: DemandVerb::Empty,
                universe_tick: UniverseTick(100),
            }]
        );
    }

    #[test]
    fn evaluate_realm_aoi_spinup_then_keepalive() {
        // AOI-2: an occupant reaching the child emits exactly one SpinUp keyed on `child.path()`; the next
        // tick (still in range) emits KeepAlive. The FULL demand is asserted (child, fence, verb, tick).
        let mut rig = Rig::new();
        rig.grant_realm();
        plant_aoi(
            &mut rig,
            vec![
                root_region(),
                own_region(),
                aoi_child(OTHER_REALM, OWN_REALM, 100.0, 3),
            ],
        );
        insert_owned_dot(
            &mut rig,
            TRIG_SESSION,
            player(7),
            DVec3::new(500.0, 0.0, 0.0),
        );
        let want = child_coord_of(OWN_REALM, OTHER_REALM);
        assert_eq!(
            demands(&rig.tick(vec![])),
            vec![RealmDemand {
                child: want.clone(),
                parent_fence: Fence(1),
                verb: DemandVerb::SpinUp,
                universe_tick: UniverseTick(100),
            }]
        );
        assert_eq!(
            demands(&rig.tick(vec![])),
            vec![RealmDemand {
                child: want,
                parent_fence: Fence(1),
                verb: DemandVerb::KeepAlive,
                universe_tick: UniverseTick(100),
            }]
        );
    }

    #[test]
    fn evaluate_realm_aoi_grace_then_drop() {
        // An occupant LEAVES: while grace remains the child stays demanded (KeepAlive), then its key drops
        // and it stops being demanded — and NO parent TearDown is EVER emitted (M-1 locks REVISION-1 R2).
        let mut rig = Rig::new();
        rig.grant_realm();
        plant_aoi(
            &mut rig,
            vec![
                root_region(),
                own_region(),
                aoi_child(OTHER_REALM, OWN_REALM, 100.0, 2),
            ],
        );
        insert_owned_dot(
            &mut rig,
            TRIG_SESSION,
            player(7),
            DVec3::new(500.0, 0.0, 0.0),
        );
        let mut seen = Vec::new();
        seen.extend(demands(&rig.tick(vec![]))); // SpinUp (grace armed to 2)
        move_dot(&mut rig, TRIG_SESSION, DVec3::new(3000.0, 0.0, 0.0)); // OUT of the 2000 m tear-down
        seen.extend(demands(&rig.tick(vec![]))); // KeepAlive (grace 2 → 1)
        seen.extend(demands(&rig.tick(vec![]))); // KeepAlive (grace 1 → 0)
        let after_grace = demands(&rig.tick(vec![])); // grace 0 ⇒ drop, silent
        assert!(
            after_grace.is_empty(),
            "grace expired ⇒ the child stops being demanded"
        );
        assert_eq!(
            seen.iter().map(|d| d.verb).collect::<Vec<_>>(),
            vec![
                DemandVerb::SpinUp,
                DemandVerb::KeepAlive,
                DemandVerb::KeepAlive
            ]
        );
        assert!(
            !seen.iter().any(|d| d.verb == DemandVerb::TearDown),
            "Step 2 never emits a parent TearDown"
        );
    }

    #[test]
    fn evaluate_realm_aoi_predictive_spinup() {
        // F7: an occupant OUTSIDE the spin-up radius but whose `pos + vel·horizon` lands inside ⇒ SpinUp
        // (boot latency masked); a STATIC occupant at the same pos ⇒ NO demand.
        let horizon = StubConfig {
            boot_ticks_p99: 20,
            ..config()
        }; // horizon_s = 20 · 0.05 = 1.0
        let mut rig = Rig::with_config(horizon);
        rig.grant_realm();
        plant_aoi(
            &mut rig,
            vec![
                root_region(),
                own_region(),
                aoi_child(OTHER_REALM, OWN_REALM, 100.0, 3),
            ],
        );
        insert_owned_dot(
            &mut rig,
            TRIG_SESSION,
            player(7),
            DVec3::new(1500.0, 0.0, 0.0),
        );
        rig.world
            .resource_mut::<Dots>()
            .0
            .get_mut(&TRIG_SESSION)
            .expect("the dot")
            .pose
            .vel = DVec3::new(-1000.0, 0.0, 0.0); // pred: 1500 − 1000·1.0 = 500 < 1000
        assert_eq!(
            demands(&rig.tick(vec![]))
                .iter()
                .map(|d| d.verb)
                .collect::<Vec<_>>(),
            vec![DemandVerb::SpinUp]
        );

        // STATIC occupant (vel 0) at the same 1500 m ⇒ live == pred == 1500 > 1000 ⇒ silent.
        let mut still = Rig::with_config(StubConfig {
            boot_ticks_p99: 20,
            ..config()
        });
        still.grant_realm();
        plant_aoi(
            &mut still,
            vec![
                root_region(),
                own_region(),
                aoi_child(OTHER_REALM, OWN_REALM, 100.0, 3),
            ],
        );
        insert_owned_dot(
            &mut still,
            TRIG_SESSION,
            player(7),
            DVec3::new(1500.0, 0.0, 0.0),
        );
        assert!(demands(&still.tick(vec![])).is_empty());
    }

    #[test]
    fn evaluate_realm_aoi_demand_order_is_stable() {
        // H-2: two children + two occupants — the emitted `Vec<RealmDemand>` is IDENTICAL regardless of the
        // occupants' `Dots` insertion order (occupants reduce to a scalar min BEFORE any emit).
        let build = |sessions: &[(SessionId, DVec3)]| -> Vec<RealmDemand> {
            let mut rig = Rig::new();
            rig.grant_realm();
            plant_aoi(
                &mut rig,
                vec![
                    root_region(),
                    own_region(),
                    aoi_child(OTHER_REALM, OWN_REALM, 100.0, 3),
                    aoi_child(RealmId::Planet(43), OWN_REALM, 100.0, 3),
                ],
            );
            for (i, (s, pos)) in sessions.iter().enumerate() {
                insert_owned_dot(&mut rig, *s, player(i as u32), *pos);
            }
            demands(&rig.tick(vec![]))
        };
        let a = build(&[
            (SessionId(1), DVec3::new(500.0, 0.0, 0.0)),
            (SessionId(2), DVec3::new(700.0, 0.0, 0.0)),
        ]);
        let b = build(&[
            (SessionId(2), DVec3::new(700.0, 0.0, 0.0)),
            (SessionId(1), DVec3::new(500.0, 0.0, 0.0)),
        ]);
        assert_eq!(a, b);
        assert_eq!(
            a.len(),
            2,
            "both children reached ⇒ two SpinUps in a stable order"
        );
    }

    #[test]
    fn evaluate_realm_aoi_evicts_a_departed_child() {
        // The AoI ledger is lazily evicted to the current roster: a child removed from the forest drops its
        // membership entry (retain_live, keyed by RealmPath) — no leak.
        let mut rig = Rig::new();
        rig.grant_realm();
        plant_aoi(
            &mut rig,
            vec![
                root_region(),
                own_region(),
                aoi_child(OTHER_REALM, OWN_REALM, 100.0, 3),
            ],
        );
        insert_owned_dot(
            &mut rig,
            TRIG_SESSION,
            player(7),
            DVec3::new(500.0, 0.0, 0.0),
        );
        let _ = rig.tick(vec![]); // SpinUp ⇒ the child's AoI state is recorded
        assert_eq!(rig.world.resource::<AoiMembership>().0.len(), 1);
        // The child leaves the roster (region removed); its state is evicted next tick.
        plant_aoi(&mut rig, vec![root_region(), own_region()]);
        let _ = rig.tick(vec![]);
        assert!(
            rig.world.resource::<AoiMembership>().0.is_empty(),
            "a departed child's AoI state is evicted"
        );
    }

    /// Drive ONE AoI tick for a shard hosting `own_realm` with a live `child` region, an in-range occupant,
    /// and a granted lease — returning every demand. The HR4 fixture runs this on two realm kinds.
    fn drive_aoi_spinup(
        own_realm: RealmId,
        own_frame: FrameRef,
        child: RealmRegion,
    ) -> Vec<RealmDemand> {
        let cfg = StubConfig {
            realm: own_realm,
            held_realms: StubConfig::single_realm(own_realm),
            frame: own_frame,
            own_coord: StubConfig::root_coord(own_realm),
            ..config()
        };
        let mut rig = Rig::with_config(cfg);
        grant_realm_for(&mut rig, own_realm);
        let root = region(ROOT_REALM, None, DVec3::ZERO, 1.0e9);
        let own = region_framed(
            own_realm,
            Some(ROOT_REALM),
            DVec3::ZERO,
            100_000.0,
            own_frame,
        );
        plant_aoi(&mut rig, vec![root, own, child]);
        insert_owned_dot(
            &mut rig,
            TRIG_SESSION,
            player(7),
            DVec3::new(500.0, 0.0, 0.0),
        );
        demands(&rig.tick(vec![]))
    }

    #[test]
    fn assert_realm_aoi_feature_anywhere() {
        // HR4 G-IDENTICAL: the IDENTICAL AoI feature (an occupant reaching a child ⇒ exactly ONE SpinUp
        // keyed on the child's coord) fires byte-identically on a SYSTEM shard (child = Planet) AND a
        // PLANET shard (child = Area). The loop is kind-BLIND — `own_coord.child(level_of(child))`, no
        // match-on-realm-kind — so the two runs differ ONLY in the child realm named.
        let a = drive_aoi_spinup(
            OWN_REALM,
            FrameRef::SystemSpace { system_seed: 7 },
            aoi_child_framed(
                OTHER_REALM,
                Some(OWN_REALM),
                FrameRef::PlanetCentered { planet_seed: 42 },
                3,
            ),
        );
        let b = drive_aoi_spinup(
            RealmId::Planet(42),
            FrameRef::PlanetCentered { planet_seed: 42 },
            aoi_child_framed(
                RealmId::Area(99),
                Some(RealmId::Planet(42)),
                FrameRef::AreaLocal {
                    planet_seed: 42,
                    area_seed: 99,
                },
                3,
            ),
        );
        assert_eq!(a.len(), 1, "the System shard emits exactly one SpinUp");
        assert_eq!(b.len(), 1, "the Planet shard emits exactly one SpinUp");
        assert_eq!(a[0].verb, DemandVerb::SpinUp);
        assert_eq!(b[0].verb, DemandVerb::SpinUp);
        // IDENTICAL structure — same verb, same emitter fence, same tick; only the child KIND differs.
        assert_eq!(a[0].parent_fence, b[0].parent_fence);
        assert_eq!(a[0].universe_tick, b[0].universe_tick);
        // Each names its OWN child through the coord machinery (System→Planet, Planet→Area).
        assert_eq!(a[0].child, child_coord_of(OWN_REALM, OTHER_REALM));
        assert_eq!(
            b[0].child,
            child_coord_of(RealmId::Planet(42), RealmId::Area(99))
        );
    }
}
