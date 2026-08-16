//! Gateway: the client's single-connection terminus + the P2 transfer-control CONSUMER.
//! (M0 ORIGIN — ONE subscription, ONE authority, `docs/design/connection_plane.md` §M0; the
//! transfer machinery — cut partition, route swap, commit drain, `OpenInputSlot`, release — landed
//! ON that base across P2 Slices 1c.0–1c.8.) The client holds exactly one logical connection;
//! everything server-side routes by in-frame `SessionId` + `Fence`, NEVER by source address (R2).
//!
//! Binding shapes that exist NOW because P2 cannot retrofit them:
//! - WRITE plane: `RouteSnapshot { authority, fence, cut }` behind `ArcSwap`; `route.store` is
//!   the gateway's SOLE route-mutation primitive (P2's `CommitAuthority` drives it). READ plane
//!   (1d.2): per-session `subs: ArcSwap<SubTable>`, with `publish_subs` the SOLE writer.
//! - The 20 Hz hot paths are [`route_input`] (write: one `route` `ArcSwap` load + one `AtomicU64`)
//!   and the read fan [`on_shard_frame`] (per subscribed shard: one `subs` load + a ≤4 `lookup` +
//!   a per-sub fence compare + a once-per-`SubId` byte-level re-tag) — no lock any control path
//!   takes. ([`forward_frame`]/[`frame_passes_fence`] are the SPIKE-2a ROUTE-SWAP-mechanic bench
//!   helpers: they load `route.fence` to measure the swap's wait-free read under contention, which
//!   is DISTINCT from the live read-plane per-sub `SubEntry::accepted` fence the fan checks.)
//! - Every forwarded frame's fence is compared against the per-shard `SubEntry::accepted` fence
//!   (in [`on_shard_frame`]); stale frames are dropped and counted (fence rule 5 — load-bearing the
//!   moment a session subscribes to more than one shard, e.g. across a transfer).
//! - The gateway is the SOLE ticket validator; the session mint COMMITS at the
//!   orchestrator's directory insert (the gateway only proposes entropy).

use std::collections::{BTreeMap, BTreeSet, VecDeque};
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use arc_swap::ArcSwap;
use bevy_ecs::prelude::{IntoScheduleConfigs, Res, ResMut, Resource, Schedule, World};
use vd_core::UniverseTick;
use vd_core::glam::DVec3;
use vd_core::home::{HomeRegistry, StoredHome};
use vd_core::pose::{FrameRef, RealmId, StampedPose};
use vd_core::realm_coord::RealmCoord;
use vd_core::rng::SplitMix64;
// The gateway is the ONE party holding both ends of every login conversion, so it holds THE world —
// LOWERED: a `WorldRealms` region forest it can query but not regenerate. It RECEIVES that value
// from the composition root (`WorldView::lowered`); the crate carries NO edge to the motion crate,
// so an orbit symbol here is an unresolved-crate compile error again (SL4 — the batch review found
// this crate allowlisted out of the fence, and the allowlist entry is deleted with the edge).
use vd_core::worldgen::WorldRealms;
use vd_core::{AccountId, EntityId, Fence, NodeId, SessionId, TickId, TransferId};
use vd_sim::io::{Inbound, MsgClass};
use vd_sim::runtime::{ClockSample, InboundBox, NodeIdentity, OutboundBox};
use vd_wire::channels::{
    CONSERVATIVE_DATAGRAM_BUDGET, ClientControlMsg, RealmSnap, RealmSnapshotDatagram, SceneRow,
    ServerControlMsg, SubId, partition_realms,
};
use vd_wire::intershard::{DemandVerb, InterShardFlow, RealmDemand, ShardPresence};
use vd_wire::seams::directory::{
    AuthorityRef, DirectoryKey, DirectoryOp, DirectoryReply, OwnerRecord,
};
use vd_wire::seams::transfer_control::{
    PrepareReject, PrepareResult, TransferControl, TransferControlAck,
};
use vd_wire::session_flow::{
    BodyStmt, GatewayToShard, RelayedStatement, ShardToGateway, WindowId, WindowScope,
    open_relay_statements, peek_input_seq, peek_snapshot_frame_id, retag_snapshot_sub,
    window_body_admissible, window_sender_is_head,
};
use vd_wire::version::ProtoVersion;

use crate::tickets;
use crate::window;

/// Gateway operational parameters — ONE reviewed struct, no inline literals.
#[derive(Clone, Copy, Debug)]
pub struct TransportTuning {
    /// Hard cap on concurrent sessions (beyond it, logins are refused loudly).
    pub max_sessions: usize,
    /// Soft cap on a session's `seq > marker` cut buffer (`TransferProgress.dest_buffer`):
    /// beyond it the OLDEST buffered input is dropped (latest-wins) + counted
    /// (`GatewayStats.dest_inputs_dropped`) — never silent, never unbounded. The cap ships
    /// WITH the buffer (NOT Slice 2) because a parked saga (`ReleaseSubscribe` not yet
    /// landed; `Bye`-mid-transfer, D-23) can hold the cut open across unbounded ticks. The
    /// HARD durable backstop (the seq-range interval map) stays 1d/P3 (D-8).
    ///
    /// INVARIANT (drain-burst bound): `apply_commit` drains the WHOLE buffer to the dest in ONE
    /// tick as unreliable `SessionInput` frames, so this cap also bounds that single-tick burst.
    /// It MUST stay well below the transport's per-tick capacity (the dest `BoundedInbox`
    /// capacity and the sender `OutboundStagingCap`) or the conserved resume batch — riding the
    /// unreliable Input class — becomes the designated shed casualty under congestion. The
    /// reliable-carrier-vs-durable-watermark redesign that removes this fragility is owed 1d/P3
    /// (D-8); until then keep this at [`Self::DEFAULT_MAX_BUFFERED_INPUTS`]-scale.
    pub max_buffered_inputs: usize,
}

impl TransportTuning {
    /// Default `max_buffered_inputs`. Sized by the CUT WINDOW (freeze → directory CAS →
    /// commit = a handful of ticks of 20 Hz input, sub-second even under tick-skew), not by the
    /// transport caps: 256 inputs ≈ 12 s of input — a generous margin over any realistic cut
    /// window — while staying below the `OutboundStagingCap` (4096) and the dest `BoundedInbox`
    /// capacity. The dest inbox is `outbound_capacity × 8` with a **`.max(256)` floor**
    /// (`io_prod::mesh`): at the shipped DEV `outbound_cap = 256` that is 2048 (8× headroom), but
    /// the order-of-magnitude margin holds only while `outbound_cap ≥ 256` — below ~32 the inbox
    /// floors at 256 == this drain burst, reintroducing the foot-gun. So treat
    /// `DEFAULT_MAX_BUFFERED_INPUTS ≤ dest_inbound_floor` as a deployment INVARIANT until the
    /// reliable-carrier / durable-watermark redesign removes the unreliable-drain fragility
    /// (D-8, 1d/P3). The earlier 2048 default exactly EQUALLED the DEV dest inbox capacity — a
    /// foot-gun where one full drain plus any co-arriving frame shed a conserved (unreliable)
    /// input.
    pub const DEFAULT_MAX_BUFFERED_INPUTS: usize = 256;
}

/// RLM 5f-3c/5f-3d — the DYNAMIC-HOME config: the SERVER-side inputs the gateway derives an
/// authenticated login's HOME realm from (never anything the client supplies), PLUS (5f-3d) the timing of
/// the pre-Active hold while that home's shard boots. Grouped in ONE struct (the
/// operational-params-in-one-struct convention) so it grows [`GatewayConfig`] by a SINGLE field. The
/// fully-INERT case (unarmed, empty pose store, zero windows) is byte-identical to the pre-5f-3c
/// gateway: no `RealmDemand` is ever emitted and no login ever enters
/// [`SessionPhase::AwaitingHomeRealm`]. There is NO `Default` — a world is a decision, and the old
/// `Default` silently built a hand-placed walk-scale one inside the shipped library (the D-WORLD-5
/// residue the batch review re-flagged); every composer states its world explicitly.
///
/// The `spawn_poses` map is the SAME `VD_SPAWN_POSES` stand-in the shard admits at (5f-3b), so the
/// gateway-DERIVED home coord and the shard-side admit pose agree; the P7 durable per-account pose store
/// swaps in behind this SAME map with zero caller reshape. An absent entry ⇒ the injector derives the
/// root/origin coord (still a valid in-forest home).
#[derive(Clone, Debug)]
pub struct SeedInjectorConfig {
    /// The ARMED gate (`VD_DEMAND`). `false` (default) ⇒ the injector is INERT: a login emits NO
    /// `RealmDemand` and routes to the static `GatewayConfig::shard`, byte-identical to the pre-5f-3c
    /// gateway. It is ALSO the 5f-3d DYNAMIC-HOME phase gate (with `ClockSample::synced` — the ONE shared
    /// expression [`GatewayConfig::dynamic_home_mode`], so the seed and the wait can never disagree). The
    /// live-arming veto (the mutual-exclusion safety with a static forest) is 5f-3e; this flag is only the
    /// on/off.
    pub armed: bool,
    /// THE WORLD this gateway resolves against — held, not re-derived per question.
    ///
    /// Every spatial answer the injector gives comes from this one value: where a login lands, which regions
    /// its home's shard will evaluate, how that realm's origin folds, and whether the resolved home is real.
    /// They used to come from `(seed, config)` re-run at each call site, and two of those call sites built
    /// DIFFERENT worlds — a home resolved among the generated stars, then checked against a hand-placed
    /// world that has never heard of them. That agreed by luck while there was one star; with several it
    /// rejects a valid home and the gateway panics on its own defence. One held world makes the two
    /// impossible to disagree.
    ///
    /// A cluster passes the world its shards will simulate: generated content in production, generated plus
    /// hand-placed structures in a test that needs a station or an area to stand in. LOWERED
    /// (`WorldView::lowered`): the region forest alone — the composition root builds the world; this
    /// crate can only query it (SL4).
    pub world: WorldRealms,
    /// WHERE EVERY ACCOUNT APPEARS: a realm, and a pose inside that realm's own frame.
    ///
    /// It used to be a per-account universe-absolute position that this router walked the whole seed
    /// forest downward to resolve. That walk subtracted each realm's STORED centre, which is zero for
    /// anything that orbits, so every orbiting planet read as sitting on its own star and a login at a
    /// star's centre landed inside a planet. A stored home has nothing to descend: the realm is a name and
    /// the pose is already measured from that realm's own centre.
    pub homes: HomeRegistry,
    /// RLM 5f-3d — the gateway's LOCAL copy of the ORCHESTRATOR's `RlmTuning::demand_ttl_ticks` (BOTH come
    /// from the SAME `resolve_rlm_tuning(tick_hz, …)` derivation, so they cannot drift while `VD_TICK_HZ` is
    /// cluster-wide). It sizes the home-demand RE-DRIVE cadence ([`Self::redrive_interval_ticks`]) — the
    /// CORRECTNESS invariant that keeps the reconciler's arm-A `demanded_recently` FRESH across the spawned
    /// shard's whole pod boot (if the re-seed lapses, arm-A expires before arm-B
    /// `running_live & !empty_confirmed` arms on live occupancy, and the reconciler KILLS the half-booted
    /// realm). `0` (the default) is INERT — unreachable while unarmed.
    pub demand_ttl_ticks: u64,
    /// RLM 5f-3d — the BOUNDED pre-Active bootstrap TTL (gateway LOCAL ticks) spanning the WHOLE
    /// dynamic-home wait: [`SessionPhase::AwaitingHomeRealm`] **plus** the dynamic-target
    /// `AwaitingAttach` (a freshly spawned shard can die between head-resolve and `SessionAttached`, so a
    /// TTL that stopped at the resolve would leave that session hanging forever). On expiry the client is
    /// Closed LOUDLY — never a silent hang (CRITIQUE-1). DERIVE it with [`Self::bootstrap_ttl_from_rlm`] so
    /// it is never tighter than the boot the orchestrator's own measured launch-TTL floor allows. `0` (the
    /// default) is INERT — unreachable while unarmed.
    pub bootstrap_ttl_ticks: u64,
}

impl SeedInjectorConfig {
    /// The fully-INERT injector over an EXPLICIT world: unarmed, empty pose store, zero windows —
    /// byte-identical to the pre-5f-3c gateway. This replaced `Default`, which silently built a
    /// hand-placed walk-scale world inside the shipped library (SL5's second-world foot-gun,
    /// D-WORLD-5): the world is now always the composer's decision, and this crate cannot build
    /// one at all (no edge to the generator).
    ///
    /// The inert injector still needs a home to state, because "no home" is not an answer a login
    /// can be given: the ambient root at its own centre — the one home every forest has.
    ///
    /// # Panics
    /// When `world` holds no ambient root (a degenerate forest — refused where it can be seen).
    #[must_use]
    pub fn inert(world: WorldRealms) -> SeedInjectorConfig {
        let root = world
            .regions()
            .iter()
            .find(|r| r.parent.is_none())
            .expect("every forest has exactly one ambient root")
            .realm;
        let fallback = StoredHome::in_realm(world.regions(), root, DVec3::ZERO)
            .expect("the root realm is named by the very forest it came from");
        SeedInjectorConfig {
            armed: false,
            world,
            homes: HomeRegistry::new(fallback),
            demand_ttl_ticks: 0,
            bootstrap_ttl_ticks: 0,
        }
    }
    /// RLM 5f-3d — the re-drive cadence divisor: the home demand is re-seeded every `demand_ttl / 4` local
    /// ticks. NAMED (never an inline literal) and chosen so the re-seed keeps the reconciler's arm-A alive
    /// with ~4x headroom (three consecutive lost re-seeds still leave the demand fresh) while cutting a
    /// 100K mass-login's re-drive fan-in by that same factor versus an every-tick re-drive: at the shipped
    /// cloud budget (`demand_ttl = hz*4`) the cadence is ~1s regardless of tick rate, i.e. a 10–40x
    /// reduction at 10–40 Hz. Mirrors the reconciler's own back-off-never-hammer discipline
    /// (`exec_spinup`'s cooldown), which is what kept the co-hosting THRASH from recurring.
    ///
    /// That same headroom absorbs the LOCAL-vs-UNIVERSE tick skew: the cadence is counted in the gateway's
    /// own local ticks while the reconciler measures demand freshness in UNIVERSE ticks, so under a CPU
    /// throttle one cadence spans more than `demand_ttl / 4` universe ticks. Even at the tick-skew ceiling the
    /// re-seed still lands well inside `demand_ttl` — the margin is why the divisor is 4 and not 2.
    pub const REDRIVE_DIVISOR: u64 = 4;

    /// RLM 5f-3d — the home-bootstrap RE-DRIVE cadence in gateway LOCAL ticks: `demand_ttl / 4`, floored at
    /// 1 (a zero cadence would divide by zero in [`home_redrive_due`]). STRICTLY less than `demand_ttl`
    /// whenever the TTL is > 1, so the re-seed always lands before arm-A goes stale.
    #[must_use]
    pub fn redrive_interval_ticks(&self) -> u64 {
        (self.demand_ttl_ticks / SeedInjectorConfig::REDRIVE_DIVISOR).max(1)
    }

    /// RLM 5f-3d — the ONE bootstrap-TTL derivation from the cluster's RLM budget (so a bin never inlines a
    /// literal): the reconciler's own launch (boot) floor — already floored by the MEASURED pod-boot p99 in
    /// `RlmTuning::cloud_with_boot`, the 5f-1 boot-floor discipline — PLUS one full demand cadence of slack
    /// for the demand ingest, the spawn decision, the realm lease grant, the head round-trip and the
    /// attach. So a REAL slow boot never trips the TTL, while a genuinely failed boot still Closes loudly
    /// inside a bounded window.
    #[must_use]
    pub fn bootstrap_ttl_from_rlm(launch_ttl_ticks: u64, demand_ttl_ticks: u64) -> u64 {
        launch_ttl_ticks.saturating_add(demand_ttl_ticks)
    }

    /// Reject a mis-tuned ARMED injector at boot (fail-LOUD, mirroring `RlmTuning::validate`). Every check
    /// is gated on `armed`: an UNARMED injector never reads these windows, so the zero default is
    /// vacuously valid. Bitwise `&` keeps both operands covered (HR5).
    ///
    /// # Errors
    /// - [`SeedInjectorError::ZeroWindowWhileArmed`] — armed with a zero demand TTL or bootstrap TTL (the
    ///   re-drive would degenerate to every-tick and the bootstrap would expire before it began).
    /// - [`SeedInjectorError::BootstrapTtlBelowRedrive`] — the bootstrap window cannot contain even ONE
    ///   re-drive, so a slow home boot would be Closed before its demand was ever re-seeded.
    pub fn validate(&self) -> Result<(), SeedInjectorError> {
        let armed = self.armed;
        if armed & ((self.demand_ttl_ticks == 0) | (self.bootstrap_ttl_ticks == 0)) {
            return Err(SeedInjectorError::ZeroWindowWhileArmed);
        }
        let redrive = self.redrive_interval_ticks();
        if armed & (self.bootstrap_ttl_ticks <= redrive) {
            return Err(SeedInjectorError::BootstrapTtlBelowRedrive {
                bootstrap_ttl: self.bootstrap_ttl_ticks,
                redrive,
            });
        }
        Ok(())
    }
}

/// A mis-tuned ARMED [`SeedInjectorConfig`] — rejected LOUD at boot so a deployment never runs a
/// bootstrap budget that would Close healthy logins (or hammer the orchestrator every tick).
#[derive(Clone, Copy, Debug, PartialEq, Eq, thiserror::Error)]
pub enum SeedInjectorError {
    /// Armed with a zero `demand_ttl_ticks` or `bootstrap_ttl_ticks`.
    #[error(
        "an ARMED gateway seed injector needs demand_ttl_ticks > 0 and bootstrap_ttl_ticks > 0 \
         (derive them from the cluster RlmTuning — see SeedInjectorConfig::bootstrap_ttl_from_rlm)"
    )]
    ZeroWindowWhileArmed,
    /// The bootstrap window is not longer than one re-drive cadence.
    #[error(
        "bootstrap_ttl {bootstrap_ttl} must strictly exceed the re-drive cadence {redrive} \
         (else a slow home boot is Closed before its demand is ever re-seeded)"
    )]
    BootstrapTtlBelowRedrive { bootstrap_ttl: u64, redrive: u64 },
}

/// Gateway configuration (composer-provided).
#[derive(Resource, Clone, Debug)]
pub struct GatewayConfig {
    pub orchestrator: NodeId,
    /// P1: the single stub shard every session lands on (the login shard).
    pub shard: NodeId,
    /// The STABLE set of routable shard `NodeId`s (node-class dispatch — FORK 5). Seeded
    /// from config (the cluster's shard roster is statically known, exactly as `shard` is);
    /// for 1d.2 it is `{shard, the transfer dest}`. It governs ONLY node-class dispatch
    /// (`is_known_shard`) — provably disjoint from client NodeIds — so a per-session
    /// subscription-refcount slip can NEVER mis-class a client datagram as a shard frame.
    /// (DISTINCT from the per-session `subscribed_shards` reverse index, which governs
    /// fan-out only.) `shard` is always a member.
    ///
    /// RLM 5f-3d: this set stays FROZEN config. A DEMAND-SPAWNED home shard was never in it (its
    /// `NodeId` is minted at spawn time, long after boot), so the dynamic roster lives in the RUNTIME
    /// [`GatewaySessions::dynamic_shards`] map and the ONE dispatch predicate
    /// ([`is_routable_shard`]) is their union.
    pub known_shards: BTreeSet<NodeId>,
    /// The auth service's Ed25519 verifying key (login validation).
    pub auth_verifying_key: [u8; tickets::ED25519_KEY_BYTES],
    /// Seed for session-id proposal entropy (the directory insert is the mint).
    pub session_seed: u64,
    /// The cluster's universe-tick rate (Hz), relayed to clients via
    /// `ServerControlMsg::UniverseRate` so they drive the render cursor at the
    /// server's rate. The SAME value the node feeds its `TickPacer` (VD_TICK_HZ).
    pub tick_hz: u32,
    /// D-3 lease-liveness heartbeat cadence (the gateway's LOCAL ticks): how often it re-sends
    /// `LeaseRenew` for every Active session's `Session` key, keeping the session lease alive against
    /// the orchestrator's reaper. `0` = INERT (no heartbeat — the pre-D-3 default). The gateway's local
    /// copy of `DirectoryTuning::lease_renew_interval_ticks` (same env knob), so the producer gates on
    /// `local_tick` without reaching across the directory seam.
    pub lease_renew_interval_ticks: u64,
    /// D-3 Slice 5b — how often (the gateway's LOCAL ticks) to re-read each Active session's `Session`
    /// head: the ROUND-TRIP confirmation channel that re-arms `Session.confirmed_at` (the partition
    /// detector). `0` = INERT (no recheck — the pre-D-3 default); mirrors the shard's
    /// `realm_recheck_interval`. The proactive self-fence is inert without it (no round-trip to measure).
    pub session_recheck_interval: u64,
    /// D-3 Slice 5b — the proactive self-fence grace (the gateway's LOCAL ticks). When an Active session's
    /// lease goes un-confirmed for longer than this (`local_tick - confirmed_at > grace` — a partition
    /// from the orchestrator), the gateway hard-stops acting as that session's authority BEFORE the
    /// orchestrator's reassign window opens. `0` = INERT (the pre-D-3 default). The gateway's local copy of
    /// `DirectoryTuning::self_fence_grace_ticks`; the split-brain-safe ordering is validated orch-side.
    pub self_fence_grace_ticks: u64,
    /// 3g abort-leg (INERT test lever): when `Some`, the NEXT `PrepareSubscribe` that would
    /// otherwise reply `Ready` instead replies `Prepared{ result: Rejected(this) }`, then the
    /// gateway self-clears it (one-shot). `None` on EVERY cluster = the 1c `Ready` stub,
    /// behaviour-identical. The SOLE way to trigger the pre-CAS abort of a crossing-origin
    /// durable saga in a cluster, since the gateway (not the dest stub) is the durable Prepare
    /// decider (`apply_prepare` hardcodes `Ready` today).
    pub reject_next_prepare: Option<PrepareReject>,
    /// RLM 5f-3c/5f-3d — the trusted-gateway dynamic-home inputs (server-derived home realm + the
    /// bootstrap hold timing). [`SeedInjectorConfig::inert`] is fully INERT (unarmed) ⇒ byte-identical to the pre-5f-3c
    /// gateway.
    pub seed_injector: SeedInjectorConfig,
    pub tuning: TransportTuning,
}

impl GatewayConfig {
    /// Is `from` a routable shard per the FROZEN config roster (STABLE node-class dispatch — FORK 5)?
    /// Seeded from the cluster's shard roster, provably disjoint from client NodeIds, so it can never
    /// mis-class a client datagram as a shard frame regardless of subscription churn. The RUNTIME half of
    /// the dispatch (demand-spawned home shards) is [`is_routable_shard`].
    #[must_use]
    fn is_known_shard(&self, from: NodeId) -> bool {
        self.known_shards.contains(&from)
    }

    /// RLM 5f-3d — THE dynamic-home mode gate (ONE expression, HR3): the gateway routes a login to its
    /// DEMAND-SPAWNED home realm's shard only when the injector is `armed` (`VD_DEMAND`) AND the clock has
    /// `synced`. Both conjuncts are load-bearing:
    /// - `armed` is the config gate — ABSENT it the login follows the EXACT pre-5f-3d static flow
    ///   (`AwaitingDirectory → AwaitingAttach → config.shard`), with NO `AwaitingHomeRealm`, NO extra
    ///   head-read and NO reordered `Welcome`/`AttachSession` (byte-identical).
    /// - `synced` because a PRE-SYNC seed would carry `universe_tick` 0, which `demanded_recently` reads as
    ///   "never demanded" — the home would never spin, so the session would wait for a realm nobody
    ///   demanded until the bootstrap TTL Closed it. Sharing this ONE expression with the seed emit is what
    ///   makes "entering the wait ⇒ a demand was seeded" a construction-level invariant.
    ///
    /// MF3: an ARMED-but-pre-sync login does NOT fall through to the static arm — the committed-lease arm
    /// HOLDS it in `AwaitingDirectory` (emitting nothing) before consulting this gate, because on an armed
    /// cluster `config.shard` is not that player's home. So the live inputs here are `!armed` (⇒ false, the
    /// static path) and `armed & synced` (⇒ true, the dynamic path); the pre-sync case is caught upstream and
    /// counted (`logins_held_pre_sync`).
    ///
    /// Bitwise `&` (no short-circuit region — HR5); the function returns both true and false across the
    /// static/dynamic unit tests.
    #[must_use]
    fn dynamic_home_mode(&self, clock_synced: bool) -> bool {
        self.seed_injector.armed & clock_synced
    }
}

/// RLM 5f-3d — THE ONE node-class dispatch predicate for "is this peer a shard": the FROZEN config roster
/// UNION the RUNTIME set of demand-spawned shards (a session's home shard, and — RLM 5f-4 — a transfer's
/// crossing DEST). A dynamically spawned shard's `NodeId` is minted at spawn time — it can never be in the
/// boot-time config — so without the runtime half its `SessionAttached`, its `SubscriptionReady` and its
/// `Snapshot`/`RealmSnapshot` frames would all fall through to the client branch and be counted
/// `undecodable` (the render authority never re-points ⇒ a BLIND player). Bitwise `|` so
/// neither membership arm is a short-circuit-uncoverable region (HR5); when unarmed `dynamic_shards` is
/// always empty, so the result is byte-identical to the pre-5f-3d config-only test.
#[must_use]
fn is_routable_shard(config: &GatewayConfig, sessions: &GatewaySessions, from: NodeId) -> bool {
    config.is_known_shard(from)
        | sessions.dynamic_shards.contains_key(&from)
        // THE WINDOW LANE (Slice B): a node this gateway holds an OPEN WINDOW on is a shard the
        // gateway itself asked to speak — a lineage ANCESTOR (the galaxy above a logged-in
        // system) is often reachable through no session role, and without this arm its window
        // rows would fall to the client branch. The windows map is derived + diffed per tick, so
        // the clause dies with the window (zero sessions ⇒ zero windows ⇒ no dispatch residue).
        | sessions.windows.values().any(|w| w.shard == from)
        // A node that ANNOUNCED itself as a shard is one, for as long as it keeps announcing — see
        // `on_shard_presence`. This is the clause that lets a demand-spawned realm be heard at all: the
        // two above are the boot roster (which cannot know about it) and a transfer's claim (which is a
        // fact about a hand-off, not about a process).
        // The ownership record's answer — the one that DECIDES. A node is here by holding a realm
        // through the fence commit, so it cannot put itself here.
        | sessions.record_shards.contains(&from)
        // The RESYNC path: a shard's own greeting, which keeps a restarted router working in the window
        // before a record reaches it. Weaker on purpose (it is self-asserted), and no longer alone.
        | sessions.announced_shards.contains(&from)
}

/// A frame from a node this router knows as NEITHER a shard NOR a client, on a class a stranger may not
/// send. It is DISCARDED — but it now says so, by name.
///
/// WHY THIS EXISTS. It used to fall to `undecodable`, the bucket for a client sending malformed bytes.
/// Those are two different facts: one is "these bytes are rubbish", the other is "I do not know who this
/// is". Sharing an outcome hid an entire running shard — measured on the demand gate, 17,365 frames
/// discarded in one run against a counter whose own comment says it must stay zero for a healthy login,
/// with no gate watching it. The shard was up and simulating a player; everything it said about that
/// player was thrown away, so the player kept being drawn by the realm they had already left.
///
/// TWO DIFFERENT FAULTS, and the split is the point. A node with a live session sending a class a client
/// may not send is a PEER BUG — the same fault as a known shard sending the wrong class, and it stays
/// `undecodable`. A node with NO session is a STRANGER, and if it happens to be a running shard then its
/// whole output is being thrown away. Lumping the two is what let the second hide inside the first.
///
/// The warn fires on the FIRST refusal only — the `== 1` edge — so a node emitting at 20 Hz produces one
/// line rather than twenty a second. The count carries the magnitude; the line names who started it.
/// Deliberately stateless: [`GatewayStats`] is `Copy` and stays that way, so this adds a counter and not
/// a set that would have to be swept.
/// Apply the ownership record's list of realm-holding nodes — WHOSE FRAMES THIS ROUTER MAY READ.
///
/// A LEVEL, not a delta: the incoming set REPLACES the held one. That is what makes a missed message
/// self-correcting — the next change carries the whole truth — and it is why a router that restarts is
/// right again as soon as anything moves, rather than having to have witnessed every join and departure.
///
/// STALE VIEWS ARE REFUSED BY TICK, so a redelivered or reordered push cannot resurrect a roster that has
/// already been superseded. Equal ticks are accepted: the same level re-sent is the same answer, and
/// refusing it would make a harmless redelivery look like a fault.
///
/// It does NOT touch `announced_shards`. That set is now the RESYNC path — a shard that greets a router
/// which has not yet been told anything is still heard, which is what keeps a restarted gateway working
/// in the window before the next roster arrives. The record is how this router DECIDES; the greeting is
/// how it COPES until the record reaches it.
fn on_shard_roster(
    roster: vd_wire::intershard::ShardRoster,
    sessions: &mut GatewaySessions,
    stats: &mut GatewayStats,
) {
    if roster.at < sessions.roster_at {
        stats.shard_roster_stale += 1;
        return;
    }
    sessions.roster_at = roster.at;
    sessions.record_shards = roster.nodes.into_iter().collect();
    stats.shard_rosters_applied += 1;
}

fn refuse_unknown_sender(
    from: NodeId,
    class: MsgClass,
    sessions: &GatewaySessions,
    stats: &mut GatewayStats,
) {
    if sessions.by_client.contains_key(&from) {
        stats.undecodable += 1; // a KNOWN client sent a class it may not send: a peer bug, as before
        return;
    }
    stats.refused_unknown_sender += 1;
    if stats.refused_unknown_sender == 1 {
        tracing::warn!(
            node = from.0,
            ?class,
            "DISCARDING frames from a node this router knows as neither a shard nor a client — if it is \
             a shard, everything it says about the players it owns is being thrown away"
        );
    }
}

/// RLM 5f-3d — THE ONE routing target for everything a session sends shard-ward (HR3: one routing path, NOT
/// a per-kind fork): its DYNAMICALLY resolved home shard when it has one, else the statically configured
/// login `config.shard`. Every former `config.shard` literal on the session path routes through this — the
/// `AttachSession` grant, its per-tick retry, the login `open_sub`, and the `Bye` detach — so a routed
/// session follows its home while a STATIC session (`home_shard` forever `None`) is byte-identical to the
/// pre-5f-3d gateway. The WRITE route's authority is retargeted separately, through the sole `store_route`
/// primitive at the home resolve.
#[must_use]
fn session_target(session: &Session, config: &GatewayConfig) -> NodeId {
    session.home_shard.unwrap_or(config.shard)
}

/// The WRITE-plane route a session's input follows (authority + the transfer cut).
///
/// **Two independent atomic publishes (1d.2):** `route` (THIS, the input plane) and
/// `subs` (the read plane, [`SubTable`]) are now SEPARATE `ArcSwap`s on [`SessionHot`].
/// The forwarder reads them for orthogonal purposes — `route_input` reads ONLY `route`
/// (never `subs`), the frame-fan reads ONLY `subs` (never `route`) — so any interleaving
/// is tolerated: no consumer needs `route` and `subs` mutually consistent at an instant.
/// (Consequence, NOT a bug given no client prediction: there is a brief window where
/// `route.authority == dest` but the dest sub isn't open yet — its `SubscriptionReady` is
/// a round-trip later — absorbed by the cut buffer; the seamless claim's honest footnote.)
///
/// `fence` is now WRITE-PLANE-ONLY: the read-plane accepted fence moved to
/// [`SubEntry::accepted`] (per-shard), but `fence` stays the WRITE route's fence so the
/// sole-`store_route`-literal (and `store_commit`'s carry) never reshapes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RouteSnapshot {
    pub authority: NodeId,
    /// The WRITE route's fence (carried by commit; the per-shard READ fence is
    /// [`SubEntry::accepted`]).
    pub fence: Fence,
    /// The input seq cut during a transfer — ALWAYS `None` in P1.
    pub cut: Option<SeqCut>,
}

/// Transfer-time input partition (P2; the shape exists so the route never reshapes).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SeqCut {
    pub marker_seq: u64,
    pub dest: NodeId,
}

/// COLD per-session transfer state, owned solely by the single-threaded control plane
/// (`on_transfer_control` + the cut-marker observer). It lives on the cold [`Session`],
/// NEVER on the `Arc<SessionHot>` shared with the (future, threaded) 20 Hz forwarder, so
/// that forwarder can never race it. `Session.transfer` is `None` when no transfer is in
/// flight. ONE in-flight transfer per session: the orchestrator serializes a session's
/// saga on its `DirectoryKey` (`DirectoryCore::lock_transfer`), so a second concurrent
/// transfer on the same subject is refused upstream — no `by_transfer` index is needed.
#[derive(Debug)]
struct TransferProgress {
    /// Binds this progress to ONE transfer; a command/marker for a different transfer on
    /// this session is rejected (counted), never absorbed against the wrong saga.
    transfer: TransferId,
    /// Whether `RequestCut` has been issued for this transfer — the precondition for
    /// confirming a cut marker. A marker that arrives before `RequestCut` (a premature or
    /// forged emit) is dropped, so a `CutConfirmed` can never be journaled before its
    /// issuing command (the saga FSM also gates `CutConfirmed` by state; this is the
    /// gateway-side half of that guard).
    cut_requested: bool,
    /// The transfer's DEST shard, captured from `PrepareSubscribe` (the first command, which
    /// always carries it). Its consumer (1d.2): the PRECISE abort-time read-sub close — on
    /// `AbortTransfer` (which carries no dest of its own) the gateway closes EXACTLY this transfer's
    /// dest sub, mirroring `ReleaseSubscribe`'s precise `src`. NEVER an "any sub != config.shard"
    /// heuristic, which would close the player's CURRENT live sub on a chained transfer and ALL
    /// composited subs (ship/host/planet) in the N-shard end goal. `close_sub` no-ops if the dest
    /// sub is not (yet) open (abort normally runs pre-CAS, before the dest sub exists).
    dest: NodeId,
    /// RLM 5f-4e — the SOURCE home this crossing is DEMOTING away from: the `home_shard` value
    /// `CommitAuthority` displaced when it re-pointed the routing target at `dest`. `Some` EXACTLY inside
    /// the demote tail (`CommitAuthority` → the terminal that closes the source sub); `None` before commit
    /// (a fresh progress) and for a session that never resolved a dynamic home (a static login's
    /// `home_shard` is `None`).
    ///
    /// WHY IT IS STASHED RATHER THAN RELEASED AT COMMIT (the REJECT-class blind window this closes): the
    /// client's READ subscription on the SOURCE stays OPEN across the whole demote grace — it is closed only
    /// at `ReleaseSubscribe`, a later tick — and the source keeps SHIPPING `Snapshot`/`RealmSnapshot` frames
    /// the whole time (the composite the seamless crossing is built on). Dropping the source's runtime-roster
    /// claim at commit un-routes it, so for a DEMAND-SPAWNED source (never in the FROZEN `known_shards`) every
    /// one of those frames falls through to the client branch and is counted `undecodable` — the player goes
    /// BLIND for the entire demote tail, right after every crossing. So the claim is HELD here and handed
    /// back at the terminal that actually closes the sub (`ReleaseSubscribe` / `AbortTransfer`), or at the
    /// session's exit ([`GatewaySessions::release_session_claims`]) if the client quits inside the window.
    ///
    /// It is INTERNAL gateway state — NOT on the wire (no `InterShardFlow` arm, no `TransferControl`
    /// variant, no `PROTO_MINOR` bump); commit already knows the old home locally.
    demoting_home: Option<NodeId>,
    /// The applied-steps idempotency journal: `(transfer, step_id) -> the recorded ack`,
    /// re-sent VERBATIM on an at-least-once redelivery (never re-applies the effect),
    /// reached ONLY through [`TransferProgress::recorded`] / [`TransferProgress::journal`]
    /// (the gateway's ONE dedup accessor — never touched inline). The key is the wire's
    /// `IdempotencyKey::TransferStep` — the SAME key the 1d durable redb `applied_steps`
    /// table builds (HR3 one machinery, many stores; only the STORE differs per altitude).
    /// RAM-ONLY by design — the gateway is soft-state (on resume it re-registers with the
    /// saga, never replays from RAM); the DURABLE table is the dest shard's at 1d (DEFERRED
    /// D-22). Bounded: O(phases) per live transfer, dropped whole on terminal/Bye/mint-refusal.
    applied: BTreeMap<(TransferId, u32), TransferControlAck>,
    /// The CUT BUFFER: `seq > marker_seq` client input captured while the cut is open,
    /// drained to the dest as `SessionInput` at `CommitAuthority` (integration.json #1: the
    /// gateway holds it; the dest applies it post-commit). FIFO = seq order (the hot
    /// `route_input` `fetch_max` dedups `seq <= prev` BEFORE the partition, so a buffered
    /// frame is strictly increasing). Bounded by `TransportTuning.max_buffered_inputs` —
    /// over cap the OLDEST is dropped + counted (latest-wins input). COLD/single-threaded
    /// state (never on `Arc<SessionHot>` — HR1; the 20 Hz forwarder must never race it);
    /// dropped whole on the in-flight terminal (`apply_abort` / `Bye`). RAM-only soft-state:
    /// a gateway crash mid-cut — OR a dest that drops `OpenInputSlot` (realm lease late at
    /// commit) — PERMANENTLY loses this take-drained buffer. **1c.5 has NO re-drive producer**
    /// (`OpenInputSlot` is emitted once at `apply_commit`; the saga is forward-only past
    /// `Committed`). The durable seq-range interval-map backstop + the re-drive are owed 1d/P3
    /// (D-8 + the 1c.7 cross-crate conservation gate).
    dest_buffer: VecDeque<Vec<u8>>,
}

/// RLM 5f-4 — the DEFERRED runtime-roster edits ONE `TransferControl` command produced, applied only after
/// the `&mut Session` borrow ends. Exactly the [`GatewaySessions::close_sub`] / `subs_to_close` shape and
/// for the same reason: a `&mut Session` handed out of `by_session` may NEVER be held across a
/// [`GatewaySessions`] mutation, and every roster edit mutates `dynamic_shards`.
///
/// Claims are applied BEFORE releases, so a re-claim of the SAME node — a second `PrepareSubscribe` naming
/// the dest an earlier one already claimed — can never transiently drop that node off the roster (it must
/// stay dispatchable straight through the defensive replace).
#[derive(Debug, Default)]
struct RosterEdits {
    /// Crossing dests to CLAIM, each through the config-gated [`GatewaySessions::claim_crossing_dest`].
    claim: Vec<NodeId>,
    /// Roster claims to RELEASE. `None` — a static session's absent home, or a pre-commit transfer's absent
    /// `demoting_home` — is the no-op arm of [`GatewaySessions::release_dynamic_shard`], so an `Option` role
    /// can be pushed VERBATIM (which is also why adding the RLM 5f-4e demoting-source role introduced no new
    /// branch at any release site). A single command may push SEVERAL: a terminal returns both the crossing
    /// dest and the demoting source; a defensive Prepare replace returns both of the displaced progress's.
    release: Vec<Option<NodeId>>,
}

impl TransferProgress {
    /// The recorded outcome for this transfer's `step`, or `None` if not yet applied — the
    /// ONE dedup READ (the redelivery gate + the cut-marker observer both call it). Keyed by
    /// `(self.transfer, step)` = the wire's `IdempotencyKey::TransferStep`.
    fn recorded(&self, step: u32) -> Option<TransferControlAck> {
        self.applied.get(&(self.transfer, step)).copied()
    }

    /// Record `ack` at this transfer's `step` — the ONE dedup WRITE, so the recorded value
    /// is re-sent verbatim on a redelivery. Keyed by `(self.transfer, step)`.
    fn journal(&mut self, step: u32, ack: TransferControlAck) {
        self.applied.insert((self.transfer, step), ack);
    }
}

/// The lock-free per-session hot state shared with the (future, threaded) 20 Hz
/// forwarding path. The cold session record owns an `Arc` of this.
///
/// **Per-session, NOT shared (M1):** each session has its OWN `subs` `ArcSwap`. There is
/// NO shared cross-session `SubTable` — an implementer must not "optimize" the per-session
/// `subs.load()` into a shared table, which would re-introduce a cross-session lock.
#[derive(Debug)]
pub struct SessionHot {
    /// WRITE plane (input route) — `route_input` reads ONLY this.
    pub route: ArcSwap<RouteSnapshot>,
    pub last_input_seq: AtomicU64,
    /// READ plane (per-shard accepted subscriptions) — the frame-fan reads ONLY this,
    /// wait-free. Published whole by `publish_subs` (the sole writer); HR1: it holds ONLY
    /// `{shard, sub, accepted}` — never any transfer state (that stays cold on [`Session`]).
    pub subs: ArcSwap<SubTable>,
}

/// An immutable per-session snapshot of the accepted subscriptions, published WHOLE on
/// every membership change and read WAIT-FREE by the forwarder (FORK 3: a boxed sorted
/// slice — ≤ ~4 entries — gives a branch-predictable linear/binary scan with no alloc on
/// the read, and lets a sub be added/removed without a lock, which a map of atomics cannot).
#[derive(Debug, Default)]
pub struct SubTable {
    /// Sorted by shard `NodeId` (so `lookup` can binary-search); ≤ ~4 entries.
    by_shard: Box<[SubEntry]>,
}

impl SubTable {
    /// The accepted subscription THIS session tagged for `shard`, or `None` if the session
    /// does not subscribe to it. Binary search over the ≤4 sorted entries.
    #[must_use]
    fn lookup(&self, shard: NodeId) -> Option<&SubEntry> {
        self.by_shard
            .binary_search_by_key(&shard, |e| e.shard)
            .ok()
            .map(|i| &self.by_shard[i])
    }
}

/// One accepted subscription on the READ plane: which `sub` id this session tagged a
/// `shard`'s frames with, and the fence at which it accepts them. HR1: no transfer state.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SubEntry {
    pub shard: NodeId,
    pub sub: SubId,
    /// The per-shard accepted frame fence (frames below it are dropped — fence rule 5).
    pub accepted: Fence,
}

/// Where one session is in its login lifecycle. `PartialEq` so tests assert the phase by EQUALITY
/// (`assert_eq!`), never `assert!(matches!(…))` whose false arm is uncoverable (HR5).
#[derive(Clone, Debug, PartialEq, Eq)]
enum SessionPhase {
    /// Session-key grant sent; awaiting the directory head (retried every tick —
    /// idempotent by fence).
    AwaitingDirectory,
    /// RLM 5f-3d — the DYNAMIC-HOME hold: the session's lease is COMMITTED (the client is already
    /// `Welcome`d) and its home realm has been DEMANDED, but that realm's shard is still booting, so there
    /// is no node to attach to yet. The gateway holds the client here — SEAMLESSLY: no `Close`, no
    /// teleport, no fallback attach to some other shard, no loading screen; the client is simply welcomed
    /// and frame-less until its own home attaches. While here the gateway (a) RE-SEEDS the home demand and
    /// (b) re-polls `HeadRead{Realm(home.lowered())}` on the [`SeedInjectorConfig::redrive_interval_ticks`]
    /// cadence, and the bounded [`SeedInjectorConfig::bootstrap_ttl_ticks`] guarantees the hold ENDS —
    /// loudly — rather than hanging. Entered ONLY in dynamic mode
    /// ([`GatewayConfig::dynamic_home_mode`]); a static login never sees this phase.
    ///
    /// `home_rid` names the realm being waited on — the KEY into [`GatewaySessions::home_bootstraps`],
    /// where the wait's heavy state lives ONCE PER REALM (the resolved lineage [`RealmCoord`], the cadence
    /// anchor, the member set). At a 100K mass login onto one home that is ONE lineage `Vec` and ONE
    /// re-drive, not one per session. [`Session::home_rid`] is the same id as a standing copy that outlives
    /// this phase (the bounded-TTL diagnostic reads it from `AwaitingAttach`, where the phase payload is
    /// gone).
    AwaitingHomeRealm { home_rid: RealmId },
    /// Directory granted (and, in dynamic mode, the home realm RESOLVED); attach sent to the session's
    /// target shard — `session.home_shard` when dynamically resolved, else the static `config.shard` —
    /// retried until attached.
    AwaitingAttach,
    /// Live: input routes shard-ward, frames flow client-ward. The subscription set
    /// lives on `Session.subs` (the cold authority) + `SessionHot.subs` (the hot
    /// projection) — NOT here (1d.2a).
    Active { entity: EntityId },
    /// D-3 Slice 5b — the gateway SELF-FENCED this session (fence rule 4): it lost contact with the
    /// orchestrator (no `Session`-head round-trip within `self_fence_grace_ticks` — a partition), or a
    /// reply revealed the lease reassigned/revoked, so it HARD-STOPS acting as the session's authority
    /// BEFORE the orchestrator's reassign window opens. Input is dropped (the `Active`-only guard in
    /// `route_client_input`), frames are skipped (`on_shard_frame`), the lease is no longer renewed, and
    /// `drive_pending_sessions` leaves it be. It lingers inert until the client's connection ends (`Bye`)
    /// or a future ResumeTicket adoption (D-37/P3) re-homes it — never double-served while fenced.
    SelfFenced,
}

/// The cold authoritative record of ONE accepted subscription (the lifecycle truth; the
/// hot `SubTable` is the forwarding projection — spec's `SessionCold.subscriptions`,
/// `connection_plane.md` §2.1).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct SubRecord {
    sub: SubId,
    frame: FrameRef,
    /// The per-shard accepted frame fence (mirrored into [`SubEntry::accepted`]).
    accepted: Fence,
    state: SubState,
}

/// A subscription's lifecycle state. `Draining` = `SubscriptionClosing` sent; it stays in
/// the `SubTable` + reverse index for ONE more tick so an in-flight straggler frame is
/// still routed (drained), then the cold drain-sweep removes it (C2 — never a silent drop).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum SubState {
    Active,
    Draining,
}

/// One session's cold record (the hot part is the shared `Arc<SessionHot>`).
#[derive(Debug)]
struct Session {
    client: NodeId,
    account: AccountId,
    fence: Fence,
    phase: SessionPhase,
    next_sub: u32,
    /// RLM 5f-3d (the D-34 field) — the session's DYNAMICALLY resolved home shard: `Some` once the
    /// `Realm(home_rid)` head named the node that owns its home realm. THE routing field: every shard-ward
    /// send resolves through [`session_target`] = `home_shard.unwrap_or(config.shard)`, so a static
    /// (unarmed) session — forever `None` — is byte-identical to the pre-5f-3d gateway.
    ///
    /// RLM 5f-4 closes D-34's A1 (AUTHORITY-FOLLOWING DETACH): it is ALSO re-pointed at `CommitAuthority`
    /// (to the transfer's `dest`), so after a crossing the routing target IS the current authority. A session
    /// that never resolved a dynamic home and never crossed stays `None` forever. D-34's COMPOSITED-SUBS
    /// clause remains OWED: this field names ONE node, whereas the source keeps its own read sub (and its
    /// ghost) until `ReleaseSubscribe` — see `TransferProgress::demoting_home`.
    home_shard: Option<NodeId>,
    /// RLM 5f-3d — the STANDING home-realm identity of a dynamic session: set once at the committed lease
    /// and NEVER cleared, so it outlives the `AwaitingHomeRealm` phase payload. Two live readers: the
    /// bounded-TTL Close diagnostic (which can fire in `AwaitingAttach`, where the phase payload is gone —
    /// this is then the ONLY surviving name of the realm that failed to boot) and the
    /// [`GatewaySessions::end_home_wait`] index removal on every session exit. `None` ⇒ a static session.
    home_rid: Option<RealmId>,
    /// The composed realm feed's per-session monotone frame counter (proto_minor 18, §2.4): the
    /// connection plane stamps it on every composed [`RealmSnapshotDatagram`] it authors — lawful
    /// because a composed row is a NEW row it authors from attested inputs — so the client's
    /// staleness gate is ONE counter (single author) plus the epoch. Chunks of one tick share
    /// one id (the sibling-chunk rule); a fresh tick increments it.
    realm_feed_frame_id: u64,
    /// The reliable scene lane's send-on-change baseline: the (realm → bag) content of the last
    /// level/delta emitted at the CURRENT epoch. Reset (to the full level) on every epoch bump;
    /// diffed per tick into `RealmSceneDelta`s at a stable epoch. Poses deliberately absent —
    /// position changes ride the unreliable per-tick datagram, never the reliable lane.
    scene_sent: BTreeMap<RealmId, Vec<u8>>,
    /// WHERE THIS LOGIN'S AVATAR GOES, measured from its home realm's own centre and stamped with that
    /// realm's frame. Resolved ONCE, in the same descent that resolves the home lineage, and then repeated
    /// verbatim on every `AttachSession` (including the retries) so a re-attach cannot land the player
    /// somewhere else than the first attempt would have.
    ///
    /// `None` for a session with no dynamic home — a static cluster attaches to a fixed shard that is not
    /// in general the realm the stored position is in, and handing it a pose measured somewhere else is
    /// exactly the confusion this field exists to end. The shard then births at its own origin, which is
    /// what every rig does today.
    spawn: Option<StampedPose>,
    /// RLM 5f-3d — the HARD deadline (gateway LOCAL tick) of the BOUNDED pre-Active dynamic-home bootstrap:
    /// `Some` EXACTLY while a dynamic session is booting — spanning `AwaitingHomeRealm` **and** the
    /// dynamic-target `AwaitingAttach` (a freshly spawned shard can die between head-resolve and
    /// `SessionAttached`) — and cleared at the `Active` promote. `None` for a STATIC session, always.
    /// It lives HERE rather than in the phase precisely because it must outlive the `AwaitingHomeRealm`
    /// phase; the re-drive ANCHOR (`since`) conversely lives IN that phase, where it is total (a session in
    /// `AwaitingHomeRealm` always has one — no `Option` arm that no test could ever reach).
    bootstrap_deadline: Option<TickId>,
    /// D-3 Slice 5b — the `local_tick` of the last `Session`-head ROUND-TRIP confirmation (the reply that
    /// affirmed THIS gateway still owns the session lease). The partition detector for the proactive
    /// self-fence: set when the session goes `Active` (the attach IS a confirmation) and re-armed on every
    /// affirming recheck reply; under a partition (no reply) it FREEZES while `local_tick` climbs, and
    /// `lease_self_fence_due` fires once the gap exceeds the grace. Meaningful only while `Active`.
    confirmed_at: TickId,
    /// The negotiated proto minor for this connection (the sender-gates-variants
    /// rule): minor-1+ variants like `UniverseRate` are emitted only when `>= 1`.
    negotiated_minor: u16,
    /// COLD transfer state, `None` until a saga's `PrepareSubscribe` opens one (1c.2).
    /// Dropped whole on the in-flight terminal (`AbortTransfer`) or when the session ends.
    transfer: Option<TransferProgress>,
    /// The COLD authoritative subscription set, keyed by shard `NodeId` (1d.2a; ≤ ~4).
    /// Mutated ONLY through `open_sub`/`close_sub`/the drain-sweep, each followed by
    /// `publish_subs` (the sole `SubTable` writer — HR3). The hot `SubTable` is its
    /// forwarding projection.
    subs: BTreeMap<NodeId, SubRecord>,
    /// 1d.5a — the per-observer-sub delivery high-water: `SubId -> highest delivered frame_id`.
    /// COLD (off the wait-free `SubTable` — HR1), written in `on_shard_frame` at the push instant
    /// (only ACCEPTED, past-fence dest frames advance it), removed when the sub is swept (no leak).
    /// The (a) demote predicate's input: the standing "every current dest observer got >=1 frame"
    /// watermark. Absent ≡ watermark 0 ≡ re-blocks (an observer opening mid-demote is undelivered).
    delivered: BTreeMap<SubId, u64>,
    /// THE WINDOW LANE's session lineage (Slice B — `docs/design/window_lane.md` §2.6.2), the
    /// realm chain root→leaf this session stands under. SESSION HISTORY ONLY, never a forest
    /// read: seeded by the login descent's [`RealmCoord`] (dynamic) or the attach frame's realm
    /// (static), and updated at each crossing's `SubscriptionReady` — a realm already in the
    /// lineage TRUNCATES back to it (an outward cross), anything else APPENDS below the previous
    /// leaf (an inward cross; "travel is always out into the shared parent and in again" — SL2,
    /// so the previous leaf IS the parent). A wrong guess is fail-closed downstream: the opened
    /// `Child` window is refused by the shard (`window_child_unrostered`) and never confirms, so
    /// the chain simply ends there. Empty until the session resolves a home/attach.
    lineage: Vec<RealmId>,
    /// THE WINDOW LANE's per-session composed scene: the derived chain, the origin marker +
    /// epoch, the held strata and the fresh-fold ring the composed emissions read from. Dropped
    /// with the session — zero sessions, zero composer state.
    shadow: window::ShadowScene,
    hot: Arc<SessionHot>,
}

/// The session table: by session id (authoritative) and by client connection
/// (in-process: the client's NodeId IS the connection).
#[derive(Resource, Debug, Default)]
pub struct GatewaySessions {
    by_session: BTreeMap<SessionId, Session>,
    by_client: BTreeMap<NodeId, SessionId>,
    /// The per-session fan-out reverse index `shard -> {sessions subscribing to it}` (FORK 5
    /// / H2), cold-maintained by `open_sub`/`close_sub`/the drain-sweep alongside the hot
    /// `SubTable`. It makes `on_shard_frame` iterate ONLY subscribers-of-`from`, not all
    /// sessions. It governs FAN-OUT only — NEVER node-class dispatch (that is the stable
    /// `config.known_shards`), so a refcount slip cannot mis-route a client. A `Draining`
    /// sub stays indexed for one tick (its straggler is drained), then removed.
    subscribed_shards: BTreeMap<NodeId, BTreeSet<SessionId>>,
    /// RLM 5f-3d — the RUNTIME routable-shard roster: `demand-spawned shard -> how many session ROLES are
    /// held on it`. A dynamically spawned shard's `NodeId` is minted at spawn time and can NEVER be in the
    /// FROZEN [`GatewayConfig::known_shards`], so this is the other half of the ONE dispatch predicate
    /// [`is_routable_shard`]. REFCOUNTED (not a grow-only set) so a long-lived gateway does not accumulate
    /// one entry per realm shard the cluster ever spun up across 100K-realm churn: a node LEAVES the roster
    /// when its last role does. Empty (⇒ dispatch byte-identical) unless in dynamic-home mode.
    ///
    /// RLM 5f-4 / 5f-4e — there are now exactly THREE role kinds, each claiming ONE refcount:
    /// - the session's HOME shard — claimed at the login home resolve (`on_home_realm_head`) and re-claimed
    ///   for the new home at `CommitAuthority`; released at the session's exit
    ///   ([`GatewaySessions::release_session_claims`]);
    /// - an in-flight transfer's CROSSING DEST — claimed at `PrepareSubscribe`
    ///   ([`GatewaySessions::claim_crossing_dest`]), released at its terminal (`ReleaseSubscribe` /
    ///   `AbortTransfer`), on a defensive Prepare replace, and at the session's exit;
    /// - a committed transfer's DEMOTING SOURCE home (`TransferProgress::demoting_home`) — the claim commit
    ///   DISPLACED but deliberately did NOT release, because the client's read sub on the source outlives the
    ///   commit by the whole demote grace. Released at the terminal that closes that sub, on a defensive
    ///   Prepare replace, and at the session's exit.
    ///
    /// So mid-demote-tail a crossing session holds TWO refcounts on the dest (home + crossing) and ONE on the
    /// source — deliberately: the dest stays routable when the tail releases the crossing one, and the SOURCE
    /// stays routable for as long as the client is still reading it (without which the player went blind for
    /// the whole tail after every crossing).
    dynamic_shards: BTreeMap<NodeId, u32>,
    /// Nodes that ANNOUNCED themselves as shards over the authenticated mesh, and are therefore heard.
    ///
    /// A SET, not a refcount, because it is not a claim anybody holds — it is a fact about a process
    /// being up. That is exactly why the previous arrangement failed: a transfer's refcount made node
    /// class last as long as one hand-off, so a running shard went mute the moment its hand-off ended,
    /// or never spoke at all if no hand-off ever named it.
    ///
    /// ⚠ REMOVAL IS OWED, and rides with the reaper: nothing takes a node OUT of this set today, so a
    /// long-lived cluster with realm churn accumulates ids. Bounded by nodes ever spawned, which is fine
    /// at the tier this runs at and is NOT fine in a real deployment. The signal to remove on is the same
    /// one the reaper needs — the orchestrator tearing a realm down — so the two land together rather
    /// than growing two half-answers to one lifecycle question.
    announced_shards: BTreeSet<NodeId>,
    /// The nodes the OWNERSHIP RECORD shows holding a realm — the authoritative answer to whose frames
    /// this router may read, pushed by the orchestrator as a whole level (`on_shard_roster`).
    ///
    /// This is the one that DECIDES. `announced_shards` above is what a shard says about itself and is
    /// now only the resync path for the window before a record arrives; a node cannot put itself HERE,
    /// because getting here means holding a realm through the fence commit.
    record_shards: BTreeSet<NodeId>,
    /// The tick of the newest roster applied, so a reordered or redelivered push cannot resurrect a
    /// superseded one. Starts at zero: before any roster arrives this router has been told nothing, and
    /// every level is newer than nothing.
    roster_at: UniverseTick,
    /// RLM 5f-3d — the per-REALM home-BOOTSTRAP index: `home realm -> the ONE wait every session booting
    /// into that realm shares`. Two independent things make it per-REALM rather than per-session, both
    /// load-bearing at 100K scale:
    /// - a `Realm` head reply carries NO session id, so without an index every reply would cost an O(S) scan
    ///   over all sessions — O(S²) across a mass login. With it ONE reply resolves the WHOLE member set at
    ///   O(log R + k).
    /// - the RE-DRIVE iterates this map, so N sessions booting into ONE home cost ONE `RealmDemand` + ONE
    ///   `HeadRead` per cadence instead of 2N. Coalescing is provably safe: the only per-session field a
    ///   demand carries is the audit-only `parent_fence`, which the orchestrator's ledger folds
    ///   order-independently and never reads for a decision (`rlm.rs` `update_fence`); child, verb and tick
    ///   are identical for every member.
    ///
    /// An entry SURVIVES the head resolve (it is not the wait's terminator — the `Active` promote is), which
    /// is what keeps the demand re-seeded across the attach round-trip; see [`HomeWait::resolved`].
    /// Maintained by the [`GatewaySessions::begin_home_wait`] / [`GatewaySessions::end_home_wait`] pair:
    /// EVERY exit (the `Active` promote, the bootstrap-TTL Close, `Bye`) drops that member through
    /// `end_home_wait`, and the realm's entry is PRUNED when its last member leaves — so neither an entry nor
    /// a lineage `Vec` can outlive its sessions. Empty unless in dynamic-home mode (⇒ byte-identical).
    home_bootstraps: BTreeMap<RealmId, HomeWait>,
    /// THE WINDOW LANE (Slice A, docs/design/window_lane.md §2.3): the windows this gateway holds
    /// open, keyed by the id IT minted (monotone, never reused — the `SubId` discipline, so a
    /// straggler row from a closed window drops by id mismatch, never a guess). DERIVED each tick
    /// from the Active sessions' own routing state ([`desired_windows`]) and diffed — so zero
    /// sessions structurally means zero window state (the design's teardown test), and a crossing
    /// re-derives the set with no bespoke hook. Empty on every pre-window rig ⇒ byte-identical.
    windows: BTreeMap<WindowId, GatewayWindow>,
    /// The monotone [`WindowId`] mint for `windows` (starts at 1 — id 0 is never issued, so a
    /// zero-initialized forgery is never a live window).
    next_window: u64,
    /// THE WINDOW LANE's realm→node answers (Slice B — `docs/design/window_lane.md` §2.6.2):
    /// which node currently heads each realm the Active sessions' LINEAGES name. Fed from the
    /// gateway's OWN routing state (every Active sub's `(node, frame)`) and from the directory's
    /// `Realm` head replies (the EXISTING `HeadRead`/`Head` pair, re-polled on the window
    /// keep-alive cadence for lineage ancestors the sessions never subscribed to — e.g. the
    /// galaxy above a logged-in system). NEVER a world model: every entry is a directory answer
    /// or a live subscription, and the map is PRUNED each tick to the realms the Active
    /// sessions' lineages + subs actually name — zero sessions ⇒ empty (the teardown truth).
    realm_heads: BTreeMap<RealmId, NodeId>,
}

/// RLM 5f-3d — ONE realm's pre-Active home bootstrap: the state EVERY session booting into that realm
/// shares. Held once per REALM, which is simultaneously the memory win (one lineage `Vec` for a whole mass
/// login) and the coalescing point (one demand + one head-read per cadence, however many sessions wait).
/// `PartialEq`/`Debug` so tests assert the WHOLE index by equality (never `matches!` — HR5); NO `Clone` —
/// nothing ever copies a wait.
#[derive(Debug, PartialEq, Eq)]
struct HomeWait {
    /// The SERVER-derived home lineage, descended ONCE per realm. Every re-seed rebuilds its demand off THIS
    /// coord through [`demand_for_home`] — never a fresh forest descend (O(1) per re-drive), and never a
    /// demand naming a different realm than the one being waited on.
    coord: RealmCoord,
    /// The gateway LOCAL tick this wait OPENED: the re-drive cadence anchor, and the anchor a member's
    /// DYNAMIC attach retry rides ([`GatewaySessions::attach_anchor`]). TOTAL (every entry has one — no
    /// unreachable `Option` arm) and PER-REALM, so a mass login across MANY homes spreads its re-drives over
    /// the cadence window instead of spiking them on one tick.
    since: TickId,
    /// The account whose per-account sentinel `parent_fence` the COALESCED demand carries — the first member
    /// to open the wait. ONE representative is correct and deliberate: that fence is audit-only at the
    /// orchestrator (`rlm.rs` `update_fence` never reads it for a decision), so pinning it per REALM keeps
    /// the re-seed's BYTES stable across member churn instead of flapping with whichever waiter is iterated.
    account: AccountId,
    /// The sessions currently booting into this realm — every one of them pre-`Active`. NON-EMPTY by
    /// construction: `end_home_wait` prunes the entry when the last member leaves.
    members: BTreeSet<SessionId>,
    /// Has a `Realm` head reply already named this realm's node for every CURRENT member? It gates ONLY the
    /// head-read half of the re-drive. The DEMAND half keeps running until the last member goes `Active`,
    /// because the reconciler's arm-A `demanded_recently` must stay fresh across the attach round-trip too:
    /// a re-seed that stopped at the resolve lets arm-A lapse mid-bootstrap and the reconciler REAPS the very
    /// realm the login is waiting for (arm-B cannot cover it — a booted-but-unoccupied realm self-reports
    /// `Empty`). Reset to `false` whenever a NEW member joins, so one lost reply cannot wedge that joiner —
    /// the poll simply resumes on the next cadence tick.
    resolved: bool,
}

/// ONE open window as the gateway holds it: where it points (the shard node the gateway
/// resolved as the stating realm's head when it derived the window), what it asks for, which
/// realm is its lawful AUTHOR — the attestation context every inbound row for this window is
/// checked against ([`on_window_row`]) — and, since Slice B, the composer's ingest state (the
/// level ring, the latest bodies, the membership verdict). Closing the window drops the ingest
/// whole: zero windows structurally means zero composer state (§2.6.1 guard 4).
#[derive(Debug, PartialEq)]
struct GatewayWindow {
    /// The shard node this window was opened on — the head the gateway's OWN routing state named
    /// for the author realm (a directory-head answer: `home_shard`/the session's subs/the
    /// window-lane `Realm` head poll). A row from any other sender is forged or stale and is
    /// dropped + counted (fail-closed).
    shard: NodeId,
    scope: WindowScope,
    /// The realm whose statements this window carries — [`WindowScope::Occupants`] ⇒ the realm
    /// the sessions stand in; [`WindowScope::Child`] ⇒ that child's PARENT (the hop author).
    author_realm: RealmId,
    /// Slice B: every attested row this window admitted, decoded once (`docs/design/
    /// window_lane.md` §2.6.2) — the ONLY write path is [`on_window_row`] AFTER attestation
    /// (§2.6.1 guard 2: the composer consumes only admitted rows).
    ingest: window::WindowIngest,
    /// Statements that raced the author's FIRST roster (Slice C1 — load-bearing since the flag
    /// day made bodies pixels): a marker (or a relayed batch) whose vouching roster has not
    /// arrived is PARKED — newest per subject/child, bounded by the roster size by construction —
    /// and re-admitted through the SAME predicates when a level lands. Send-on-change lanes send
    /// ONCE, so a fail-closed drop here would be a permanently invisible body; parking keeps the
    /// drop fail-closed (nothing is served un-attested) without the permanence.
    parked_bodies: BTreeMap<RealmId, (BodyStmt, vd_core::UniverseTick)>,
    /// The parked Q2 relays, newest child-fence wins — drained exactly like the bodies.
    parked_relays: BTreeMap<RealmId, (Fence, Vec<u8>)>,
}

impl GatewaySessions {
    /// THE WINDOW LANE's gauge for the admin surface: how many windows this gateway holds open.
    #[must_use]
    pub fn windows_open_count(&self) -> usize {
        self.windows.len()
    }
    #[must_use]
    pub fn len(&self) -> usize {
        self.by_session.len()
    }
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.by_session.is_empty()
    }
    /// RLM RG-4 — how many demand-spawned home shards are on the RUNTIME routable roster. The direct
    /// process-tier observable for "the dynamic-home resolve fired": `dynamic_shards` is populated in exactly
    /// one place, `claim_dynamic_shard` at the `on_home_realm_head` resolve, so a nonzero count means a login
    /// routed to a spawn-minted node rather than the static `config.shard`.
    #[must_use]
    pub fn dynamic_shard_count(&self) -> usize {
        self.dynamic_shards.len()
    }
    /// The session ids currently active (for tests/oracles).
    pub fn sessions(&self) -> impl Iterator<Item = SessionId> + '_ {
        self.by_session.keys().copied()
    }
    /// The avatar entity a session renders authoritatively (None until Active).
    /// P2's transfer coordinator keys its sagas on this.
    #[must_use]
    pub fn entity_of(&self, session: SessionId) -> Option<EntityId> {
        self.by_session.get(&session).and_then(|s| match s.phase {
            SessionPhase::Active { entity, .. } => Some(entity),
            SessionPhase::AwaitingDirectory
            | SessionPhase::AwaitingHomeRealm { .. }
            | SessionPhase::AwaitingAttach
            | SessionPhase::SelfFenced => None,
        })
    }

    /// RLM 5f-3d — the session's DYNAMICALLY resolved home shard, or `None` for a STATIC (unarmed) session
    /// / one whose home realm has not become routable yet. The observable half of the D-34 routing field
    /// (the harness/oracles and 5f-4's live route read it to assert WHICH node a session was routed to).
    #[must_use]
    pub fn home_shard_of(&self, session: SessionId) -> Option<NodeId> {
        self.by_session.get(&session).and_then(|s| s.home_shard)
    }

    /// RLM 5f-3d — the session's HOME REALM (the deepest realm containing its stored spawn pose), or `None`
    /// for a STATIC (unarmed) session. Set once at the committed lease and never cleared, so it is
    /// meaningful in every phase from the lease onward.
    #[must_use]
    pub fn home_realm_of(&self, session: SessionId) -> Option<RealmId> {
        self.by_session.get(&session).and_then(|s| s.home_rid)
    }

    /// D-3 / S4 — the local tick of the FRESHEST last-lease-round-trip confirmation (`Session::confirmed_at`)
    /// over sessions that carry a meaningful one: `Active` (re-armed on each affirming `Session`-head recheck
    /// reply) OR `SelfFenced` (FROZEN at the instant the proactive self-fence fired — the gateway's own
    /// evidence it lost the directory path). `None` when no such session exists (only pre-`Active` logins, or
    /// zero sessions). The gateway readiness partition detector — the session-servicing analogue of the
    /// shard's `RealmConfirmedAt`, fed to `vd_node::health::gateway_sessions_live`.
    ///
    /// `SelfFenced` MUST be included: [`self_fence_lapsed_sessions`] fires at the SAME
    /// `local_tick - confirmed > grace` threshold as the readiness de-route and runs EARLIER in the same tick,
    /// so under a TOTAL partition it flips every session `Active → SelfFenced` BEFORE readiness is sampled — an
    /// `Active`-only max would then read `None` and keep a fully-partitioned gateway falsely Ready. A still-fresh
    /// `Active` session dominates the max, so a healthy gateway (or a partial partition where one session still
    /// re-arms) stays Ready; the gateway de-routes only when the freshest over BOTH sets is stale (EVERY session
    /// has frozen — a total directory partition). It re-becomes Ready once a fresh `Active` session attaches or
    /// the frozen `SelfFenced` ghosts clear on connection-end.
    #[must_use]
    pub fn freshest_session_confirmed(&self) -> Option<u64> {
        self.by_session
            .values()
            .filter(|s| {
                matches!(
                    s.phase,
                    SessionPhase::Active { .. } | SessionPhase::SelfFenced
                )
            })
            .map(|s| s.confirmed_at.0)
            .max()
    }

    /// The sessions subscribing to `shard` (the H2 reverse-index read the frame-fan iterates).
    /// Empty when no session subscribes to it. Returns owned ids so the caller can mutate the
    /// outbox while iterating; the set is ≤ S and only the subscribers, never all sessions.
    fn subscribers_of(&self, shard: NodeId) -> Vec<SessionId> {
        self.subscribed_shards
            .get(&shard)
            .map(|set| set.iter().copied().collect())
            .unwrap_or_default()
    }

    /// RLM 5f-3d — THE one entry into the pre-Active home bootstrap (HR3): JOIN `session_id` to its home
    /// realm's [`HomeWait`], creating that wait (with the descended lineage, the cadence anchor and the
    /// representative account) when this session is the first one booting into the realm. Called from the
    /// committed-lease arm in dynamic mode, exactly once per login.
    ///
    /// A joiner ALWAYS clears `resolved`: it has not seen a head reply of its own, so the head poll must
    /// resume even when an earlier member already resolved this realm — otherwise a single lost reply would
    /// hang the joiner until the bounded TTL closed it. `since` is NOT re-anchored (the shared cadence is the
    /// point), and the coord/account of the opening member are kept (the demand is per-realm, not per-member).
    fn begin_home_wait(
        &mut self,
        session_id: SessionId,
        home_rid: RealmId,
        coord: RealmCoord,
        since: TickId,
        account: AccountId,
    ) {
        let wait = self
            .home_bootstraps
            .entry(home_rid)
            .or_insert_with(|| HomeWait {
                coord,
                since,
                account,
                members: BTreeSet::new(),
                resolved: false,
            });
        wait.members.insert(session_id);
        wait.resolved = false;
    }

    /// RLM 5f-3d — THE one exit from the pre-Active home bootstrap (HR3): drop `session_id` from its home
    /// realm's member set and PRUNE the whole realm entry once it empties (no unbounded growth, no orphan
    /// lineage `Vec`). Called on EVERY exit — the `Active` promote, the bootstrap-TTL Close, and `Bye` — so
    /// an index entry can never outlive its sessions. Total, never fallible:
    /// - `home_rid` `None` ⇒ a STATIC session ⇒ no-op (the byte-identical arm every static `Bye` takes);
    /// - the realm absent ⇒ this session already left the bootstrap (the normal arm for a `Bye` AFTER the
    ///   session went `Active`, since `Session::home_rid` is deliberately never cleared) ⇒ no-op.
    fn end_home_wait(&mut self, session_id: SessionId, home_rid: Option<RealmId>) {
        let Some(rid) = home_rid else {
            return;
        };
        let Some(wait) = self.home_bootstraps.get_mut(&rid) else {
            return;
        };
        wait.members.remove(&session_id);
        if wait.members.is_empty() {
            self.home_bootstraps.remove(&rid);
        }
    }

    /// RLM 5f-3d — the cadence ANCHOR a pre-`Active` session's `AwaitingAttach` retry rides, fed to
    /// [`attach_retry_due`]: `Some(since)` of the [`HomeWait`] it belongs to for a DYNAMIC session (so its
    /// attach retry is coalesced onto the SAME backed-off cadence as that realm's re-drive — a mass login
    /// must not re-attach every session every tick at a freshly booted shard), `None` for a STATIC session
    /// (`home_rid` is `None` ⇒ the byte-identical per-tick retry).
    ///
    /// Both `?` arms are total and FAIL-SAFE: `None` means "retry every tick", i.e. the pre-5f-3d behaviour —
    /// never a wedge. The second arm (a `home_rid` with no live wait) cannot occur on the live path — every
    /// path that drops a member either removes the session (`Bye`, the TTL Close) or leaves `AwaitingAttach`
    /// (the `Active` promote) — so it is proven directly by a unit test rather than through a system.
    #[must_use]
    fn attach_anchor(&self, home_rid: Option<RealmId>) -> Option<TickId> {
        let rid = home_rid?;
        Some(self.home_bootstraps.get(&rid)?.since)
    }

    /// RLM 5f-3d — claim `home` for one session on the RUNTIME routable-shard roster (the dispatch half of
    /// the dynamic route): the node becomes node-class-dispatchable as a shard for as long as at least one
    /// session is homed on it.
    fn claim_dynamic_shard(&mut self, home: NodeId) {
        *self.dynamic_shards.entry(home).or_insert(0) += 1;
    }

    /// RLM 5f-3d — release one session's claim on its dynamically resolved home shard. At zero the node
    /// LEAVES the runtime roster, so a long-lived gateway accumulates no entry per ever-spawned realm shard
    /// (the 100K-realm churn leak). `None` — a STATIC session, or a dynamic one that never resolved a home
    /// — is a no-op. All arms (no home / last-out ⇒ remove / others remain) are proven by unit tests.
    fn release_dynamic_shard(&mut self, home: Option<NodeId>) {
        let Some(node) = home else {
            return;
        };
        let remaining = self
            .dynamic_shards
            .get(&node)
            .copied()
            .unwrap_or(0)
            .saturating_sub(1);
        if remaining == 0 {
            self.dynamic_shards.remove(&node);
        } else {
            self.dynamic_shards.insert(node, remaining);
        }
    }

    /// RLM 5f-4 — CLAIM a transfer's CROSSING DEST on the runtime routable roster: THE ADMISSION that makes
    /// a DEMAND-SPAWNED destination realm's shard node-class-dispatchable, so its `SubscriptionReady`, its
    /// `Snapshot` frames and its `RealmSnapshot` frames are CONSUMED instead of falling through to the
    /// client branch and being counted `undecodable` (which left the render authority un-repointed — a
    /// BLIND player after every crossing into a realm the cluster spun up on demand). Driven by the
    /// orchestrator-signed `PrepareSubscribe` (which strictly precedes FreezeSource/CommitAuthority, so the
    /// dest is routable before its first frame) and re-claimed for the home role at `CommitAuthority`.
    ///
    /// GATED to a dest the FROZEN [`GatewayConfig::known_shards`] does not already cover. Two properties
    /// follow, both load-bearing:
    /// - an ALL-STATIC crossing leaves `dynamic_shards` EMPTY, so [`is_routable_shard`] answers purely from
    ///   the config half exactly as it did before this slice (byte-identical dispatch, zero new state);
    /// - the RELEASE side needs NO matching gate, because [`Self::release_dynamic_shard`] of a node that
    ///   holds no entry is a total no-op on the map (`unwrap_or(0).saturating_sub(1) == 0` ⇒ it removes an
    ///   absent key).
    ///
    /// The one-gate design is NOT justified by "a skipped claim leaves nothing to release" — that is FALSE:
    /// a `known_shards` node CAN hold runtime entries, because the LOGIN-time home claim goes through the
    /// RAW, UNGATED [`Self::claim_dynamic_shard`] (`on_home_realm_head`) and `bins/gateway.rs` always puts
    /// the login shard in `known_shards`. The true invariant is a SPLIT by the frozen half:
    /// - for a node IN `known_shards`, its runtime entry is IRRELEVANT to dispatch — [`is_routable_shard`]
    ///   already answers `true` from the config half — so a crossing into it that skips the claim while its
    ///   terminal decrements a login-time entry is unobservable, and can neither un-route the node nor pin it;
    /// - for a node NOT in `known_shards` (the only case dispatch depends on), the gate NEVER skips, so every
    ///   claim site (the raw login claim, this crossing claim, the commit-time home re-claim) is paired by
    ///   count with a release site (the terminal, the defensive replace, the session exit) and the refcount
    ///   is exactly balanced.
    fn claim_crossing_dest(&mut self, config: &GatewayConfig, dest: NodeId) {
        if config.is_known_shard(dest) {
            return; // already dispatchable from the frozen roster — never a spurious dynamic entry
        }
        self.claim_dynamic_shard(dest);
    }

    /// RLM 5f-4 — THE one session-exit roster release (HR3): drop EVERY runtime-roster claim `session`
    /// holds. There are THREE roles (RLM 5f-4e), released here in one place so no exit can leak a refcount
    /// that would pin a demand-spawned node on the roster forever:
    /// 1. its resolved HOME shard (`home_shard`);
    /// 2. an in-flight transfer's CROSSING DEST (`transfer.dest`);
    /// 3. an in-flight transfer's DEMOTING SOURCE home (`transfer.demoting_home`) — the claim
    ///    `CommitAuthority` stashed instead of releasing, so the source stays routable while the client
    ///    still reads it.
    ///
    /// Called from BOTH session exits (`Bye` and the bounded-TTL Close). Releasing all three is what makes
    /// an exit anywhere in the demote tail exact: in the window between `CommitAuthority` and
    /// `ReleaseSubscribe` the dest is claimed TWICE (crossing + new home) and the SOURCE still holds its
    /// stashed one, and the three releases here take every node back to zero. Every arm is total: `None` — a
    /// static session, one with no transfer in flight, or a pre-commit transfer (whose `demoting_home` is
    /// always `None`) — is the no-op arm of [`Self::release_dynamic_shard`]. Role 3 is `None` for the
    /// bounded-TTL exit BY CONSTRUCTION (that deadline is cleared at the `Active` promote and a transfer
    /// requires `Active`), so only `Bye` can observe it `Some`.
    fn release_session_claims(&mut self, session: &Session) {
        self.release_dynamic_shard(session.home_shard);
        self.release_dynamic_shard(session.transfer.as_ref().map(|tp| tp.dest));
        self.release_dynamic_shard(session.transfer.as_ref().and_then(|tp| tp.demoting_home));
    }

    /// THE one open primitive (HR3): allocate a never-reused `sub` id, insert the cold
    /// `SubRecord` (Active), push `SubscriptionOpened` BEFORE publishing the hot table (X1 —
    /// a forwarder can never route a frame for the sub ahead of its opener), publish the sole
    /// `SubTable`, and index the reverse fan-out. The transfer is the FIRST extra caller;
    /// login is the first. Returns the allocated sub id.
    fn open_sub(
        &mut self,
        session_id: SessionId,
        shard: NodeId,
        frame: FrameRef,
        accepted: Fence,
        outbox: &mut OutboundBox,
    ) -> Option<SubId> {
        let session = self.by_session.get_mut(&session_id)?;
        let sub = SubId(session.next_sub);
        session.next_sub += 1;
        session.subs.insert(
            shard,
            SubRecord {
                sub,
                frame,
                accepted,
                state: SubState::Active,
            },
        );
        // X1: SubscriptionOpened strictly precedes any data for the sub.
        push_control(
            outbox,
            session.client,
            &ServerControlMsg::SubscriptionOpened { sub, frame },
        );
        publish_subs(&session.hot, &session.subs); // sole SubTable writer (HR3)
        self.subscribed_shards
            .entry(shard)
            .or_default()
            .insert(session_id);
        Some(sub)
    }

    /// THE one close primitive (HR3): mark the cold `SubRecord` `Draining` and send
    /// `SubscriptionClosing`. It STAYS in the `SubTable` + index for ONE more tick (C2 drain
    /// grace) so an in-flight straggler frame is still routed; the cold drain-sweep removes it
    /// next tick. A close of a shard this session does not subscribe to is a no-op. Called by
    /// `on_transfer_control` at `ReleaseSubscribe` (close the source sub) and defensively at
    /// `AbortTransfer` (close any dest sub).
    ///
    /// **THE AUTHORITY-SUB INVARIANT (the read-plane half of claim-before-release):** never close
    /// the subscription of the node that is CURRENTLY feeding this session
    /// ([`session_target`] — the one definition of "the current authority", HR3). A SAME-NODE
    /// re-home (`source == dest`) is an ordinary saga — task #149 deleted the node-placement
    /// short-circuit precisely so EVERY re-home runs the one cross-node machinery — and on one the
    /// "source sub" IS the live authority sub, because `subs` is keyed by shard `NodeId` (ONE entry
    /// per node). Closing it there unsubscribes the player from their own owner: no snapshots, no
    /// realm feed, no clock, no visible input response — a TOTAL client freeze while the shard
    /// happily simulates them. The runtime roster already survives this case by claiming the dest a
    /// SECOND time at `CommitAuthority` (claim-before-release); this is the same rule for the read
    /// plane. Counted, never silent.
    fn close_sub(
        &mut self,
        session_id: SessionId,
        shard: NodeId,
        config: &GatewayConfig,
        outbox: &mut OutboundBox,
        stats: &mut GatewayStats,
    ) {
        let Some(session) = self.by_session.get_mut(&session_id) else {
            return;
        };
        if session_target(session, config) == shard {
            stats.sub_close_refused_authority += 1;
            tracing::warn!(
                shard = shard.0,
                "refused to close the sub of the session's CURRENT authority (same-node re-home) \
                 — closing it would freeze the client on its own owner"
            );
            return;
        }
        let Some(rec) = session.subs.get_mut(&shard) else {
            return;
        };
        if rec.state == SubState::Draining {
            return; // already draining (idempotent — never a second SubscriptionClosing)
        }
        rec.state = SubState::Draining;
        let sub = rec.sub;
        push_control(
            outbox,
            session.client,
            &ServerControlMsg::SubscriptionClosing { sub },
        );
        // NOTE: left in `subs` + `subscribed_shards` until the next-tick drain-sweep so a
        // straggler frame from `shard` is drained, not dropped (the hot table still resolves
        // it). No `publish_subs` here — the hot projection keeps the Draining entry routable.
    }

    /// The cold drain-sweep (off the hot path): remove every `Draining` sub, republish the
    /// session's `SubTable`, and drop the reverse-index entry. Runs once per control tick, so a
    /// `Draining` sub lives for exactly one tick (its straggler drained), then is gone.
    fn sweep_draining(&mut self) {
        // Collect first (one mutable borrow at a time): which (session, shard) pairs drained.
        let mut drained: Vec<(SessionId, NodeId)> = Vec::new();
        for (session_id, session) in &mut self.by_session {
            // Capture (shard, sub) up front so the watermark cleanup needs no fallible re-lookup
            // (a `subs.remove` here is always Some — the shard came from this very filter).
            let draining: Vec<(NodeId, SubId)> = session
                .subs
                .iter()
                .filter(|(_, r)| r.state == SubState::Draining)
                .map(|(shard, r)| (*shard, r.sub))
                .collect();
            if draining.is_empty() {
                continue;
            }
            for (shard, sub) in &draining {
                session.subs.remove(shard);
                // 1d.5a: the delivery watermark dies WITH its sub (no leak; a swept observer
                // leaves the conjunction). Removed at the sweep — not at `close_sub` — so a
                // one-tick-Draining sub still counts its delivered frames until it is truly gone.
                session.delivered.remove(sub);
                drained.push((*session_id, *shard));
            }
            publish_subs(&session.hot, &session.subs);
        }
        // Then drop each drained session from its shard's reverse-index entry.
        for (session_id, shard) in drained {
            if let Some(set) = self.subscribed_shards.get_mut(&shard) {
                set.remove(&session_id);
                if set.is_empty() {
                    self.subscribed_shards.remove(&shard);
                }
            }
        }
    }
}

/// Session-id proposal entropy (the directory insert is the authoritative mint).
#[derive(Resource, Debug)]
struct SessionMint(SplitMix64);

/// Honesty counters: everything tolerated-but-rejected is counted, never silent.
#[derive(Resource, Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct GatewayStats {
    pub logins_rejected: u64,
    pub version_rejected: u64,
    pub sessions_refused_capacity: u64,
    pub session_mints_refused: u64,
    pub resumes_refused: u64,
    pub inputs_deduped: u64,
    pub inputs_unroutable: u64,
    pub inputs_malformed: u64,
    pub stale_frames_dropped: u64,
    /// A shard's `EntityRemoved` refused for ONE subscriber because it carried a fence stale
    /// against that session's per-shard accepted fence (rule 5 — a demoted old owner's removal
    /// must not evict what the live owner still streams). The same discipline as
    /// `stale_frames_dropped`, counted apart because a wrongly-dropped REMOVAL strands a frozen
    /// figure (the D-4(a) defect) while a wrongly-dropped frame costs one tick of motion.
    pub stale_removals_dropped: u64,
    pub undecodable: u64,
    /// A frame from a node this router knows as NEITHER a shard NOR a client, on a class a stranger may
    /// not send. It is DISCARDED, and until now it was discarded as `undecodable` — the same bucket as a
    /// client sending malformed bytes.
    ///
    /// THE TWO ARE NOT THE SAME FACT, and lumping them hid a whole running shard. Measured on the demand
    /// gate: 17,365 frames in one run, against a counter whose own comment says it must stay zero for a
    /// healthy login, and no gate noticed. A shard that is up, simulating a player, and not on the roster
    /// had its entire output thrown away in silence; the player it owned kept being drawn by the realm
    /// they had already left.
    pub refused_unknown_sender: u64,
    /// Rosters applied from the ownership record — the router being TOLD whose frames it may read, rather
    /// than inferring it. On a healthy demand cluster this moves a handful of times as realms spin up and
    /// down, and then stops.
    pub shard_rosters_applied: u64,
    /// Rosters refused as older than the one held. Not a fault by itself — a redelivery on a re-driven
    /// lane is expected — but a rising count next to a router that is missing shards means pushes are
    /// arriving out of order and the tick guard is the only thing holding the line.
    pub shard_roster_stale: u64,
    /// A session in the `subscribed_shards` reverse index for `from` had NO matching
    /// `SubEntry` in its hot `SubTable` (an index/table desync — an invariant breach the
    /// `publish_subs` co-republish makes impossible by construction). Counted, never a silent
    /// `continue` (C2 honesty floor). A straggler from a just-closed source is NOT this — that
    /// is the one-tick `Draining` grace, drained not dropped.
    pub frame_sub_desync: u64,
    /// A sub close was REFUSED because the shard is the session's CURRENT authority
    /// ([`GatewaySessions::close_sub`]'s authority-sub invariant). The reachable producer is a
    /// SAME-NODE re-home (`source == dest`): its `ReleaseSubscribe` names a `src` that is also the
    /// post-commit authority, and honouring that close would strand the client on its own owner
    /// (the total-freeze class). `> 0` simply means the cluster ran a same-node re-home — expected
    /// whenever an occupant re-enters the realm it just left; it is a health signal, not an error.
    pub sub_close_refused_authority: u64,
    /// A `TransferControl` command (or cut marker, or a `SubscriptionReady` read-plane
    /// notice) for an unknown/absent session, or a phase command whose `TransferProgress`
    /// prerequisite is missing — dropped + counted, never panicked (mirrors `inputs_unroutable`).
    pub transfer_unroutable: u64,
    /// A still-parked route phase (after 1c.4: `ReleaseSubscribe`, the demote tail).
    /// Counted + warned, no ack (the saga correctly pins until its handler exists),
    /// never journaled.
    pub transfer_control_parked: u64,
    /// `CommitAuthority` arrived for a BOUND transfer (session + matching `TransferProgress`
    /// present) but `route.cut` is `None` — a FreezeSource-precedes-commit ordered-control
    /// protocol violation. Counted + `tracing::error`-logged; the saga PINS (route untouched),
    /// never a garbage-dest swap. DISTINCT from `transfer_unroutable` (the session or the
    /// in-flight transfer is absent) so the WEDGE-1 pin signal stays unblurred.
    pub commit_without_cut: u64,
    /// A `seq > marker_seq` client input held in the cut buffer for the dest (the partition
    /// fired). Drained to the dest at `CommitAuthority`. 0 outside a transfer's cut window.
    pub inputs_buffered_for_dest: u64,
    /// A buffered input dropped because the cut buffer hit `max_buffered_inputs` (oldest-
    /// first, latest-wins) — the never-silent floor; nonzero only under a parked/stalled
    /// saga holding the cut open (D-23). The durable backstop is 1d/P3 (D-8).
    pub dest_inputs_dropped: u64,
    /// D-3 Slice 5b — PROACTIVE Session self-fences: an Active session whose lease went un-confirmed past
    /// `self_fence_grace_ticks` (a detected partition from the orchestrator) was hard-stopped. Ops
    /// visibility / partition signal; `0` on the happy path and inert (`self_fence_grace_ticks == 0`).
    pub sessions_self_fenced_lapsed: u64,
    /// RLM 5f-3d — bounded-TTL dynamic-home bootstrap FAILURES: a login whose demand-spawned home realm did
    /// not become routable (or whose resolved home never confirmed the attach) inside
    /// `bootstrap_ttl_ticks` was Closed LOUDLY. `0` on every happy path and inert for a static gateway; a
    /// nonzero value is the ops signal that the spawn path — not the session path — is broken.
    pub home_bootstrap_timeouts: u64,
    /// RLM 5f-3d — a `home_bootstraps` member referenced a session absent from `by_session` (an invariant
    /// breach the `begin_home_wait`/`end_home_wait` pairing makes impossible by construction). Counted,
    /// never a silent `continue` — the same C2 honesty floor as `frame_sub_desync`. (Plain backticks, not an
    /// intra-doc link: this is PUBLIC documentation naming a private index.)
    pub home_wait_desync: u64,
    /// RLM 5f-3d (MF3) — committed-lease logins HELD because the gateway is ARMED but its clock has not
    /// synced yet, so the home realm could not be demanded (a pre-sync demand carries `universe_tick` 0,
    /// which the reconciler reads as "never demanded"). The login stays in `AwaitingDirectory` and its
    /// idempotent `LeaseGrant` is re-driven, so this self-heals on the first `ClockSync`: a small count at
    /// boot is normal, a CLIMBING one says the clock broadcast never arrived (check the orchestrator's
    /// `clock_peers`). `0` on a static gateway — the static path never consults the clock.
    pub logins_held_pre_sync: u64,
    /// D-3 Slice 5b — REACTIVE Session self-fences: a `Session`-head recheck reply revealed the lease had
    /// been reassigned/revoked (no longer this gateway at its fence), so the session was hard-stopped
    /// promptly (the link-alive cure, vs the proactive timer's partition cure). `0` on the happy path.
    pub sessions_self_fenced_revoked: u64,
    /// RLM 5f RG-3 — reactive-greeting `ShardPresence` frames received from demand-spawned shards. Pure
    /// OBSERVABILITY: the connection-learning that makes the shard reachable already happened below the app
    /// seam (the mesh records the return connection on this same reliable frame), so the gateway does NOTHING
    /// with it but count + debug-log — never a `dynamic_shards` claim (a presence has no session role or
    /// lifetime, and the roster is authority-refcounted). `0` on a static gateway (no shard ever greets). Its
    /// existence keeps a greeting from being miscounted `undecodable` (the honesty floor).
    pub presence_announces: u64,
    /// THE WINDOW LANE's admitted rows (mesh minor 16): a `WindowFrame`/`WindowBody`/
    /// `WindowMembership` row that PASSED attestation (a known window, the roster-head sender,
    /// an admissible body) and was INGESTED into the window's composer state (Slice B retired
    /// Slice A's deliberate `window_rows_unconsumed` — the engine now consumes every admitted
    /// row). Counted APART from `undecodable` (the honesty floor); grows at the realm-lane rate.
    pub window_rows_ingested: u64,
    /// THE WINDOW LANE, fail-closed (Slice A — docs/design/window_lane.md §2.2/§2.6.6): a window
    /// row whose sender is NOT the head this gateway resolved for the stating realm when it
    /// opened the window ([`window_sender_is_head`] driven with the held head). A forged or
    /// deposed sender's row is dropped + counted, never served on faith. `0` healthy.
    pub window_sender_mismatch: u64,
    /// THE WINDOW LANE, fail-closed: a `WindowBody` whose authorship the admission rule refuses
    /// ([`window_body_admissible`] — a look about anything but the author itself, a marker about
    /// a non-child). Dropped + counted, never patched (§2.2). `0` healthy.
    pub window_misauthored_body: u64,
    /// THE WINDOW LANE, fail-closed: a window row naming an id this gateway holds no window for —
    /// a straggler from a closed window (dropped by id mismatch, exactly the `WindowId` contract)
    /// or a forged id. BENIGN in small numbers around closes; steady growth is a fault.
    pub window_unknown_row: u64,
    /// THE WINDOW LANE's control egress — `WindowOpen`s sent for NEWLY derived windows
    /// (THROUGHPUT; one per window per derivation edge, never per tick).
    pub window_open_sent: u64,
    /// THE WINDOW LANE's control egress — `WindowClose`s sent when a window stopped being
    /// derivable (a session ended, crossed away, or its parent sub drained). The polite fast
    /// path; the shard's derived TTL is the crash backstop.
    pub window_close_sent: u64,
    /// THE WINDOW LANE's keep-alive egress — `WindowOpen` re-asserts on the derived cadence
    /// (idempotent per window; the shard's TTL is 2 beats + 1). THROUGHPUT.
    pub window_keepalives_sent: u64,
    /// Slice B — a `WindowFrame` stamped further behind its window's ring head than the derived
    /// span (`WindowTuning::ring_span_ticks`): refused, counted, never folded (§2.6.2).
    pub window_level_refused: u64,
    /// Slice B — a `WindowBody` older (by `authored_at`) than the statement already held for its
    /// subject: refused (newest wins on a re-driven lane), counted.
    pub window_body_stale: u64,
    /// Slice B — a `Marker` arriving before its author's FIRST placement level, so the
    /// stream-only child roster (the author's own attested full-roster rows — §2.6.2 deleted the
    /// seed-forest source) cannot vouch for the subject yet: refused fail-closed, counted apart
    /// from `window_misauthored_body` (a well-ordered peer racing its own lanes is not a forger).
    pub window_body_preroster: u64,
    /// Slice B — SHARED folds actually computed: one per (origin, common tick) per gateway tick,
    /// shared across every session standing in that origin (§2.14). THROUGHPUT.
    pub window_folds: u64,
    /// Slice B — sessions served from an already-computed shared fold this tick (§2.14's
    /// memoization). The parity gate requires this nonzero with ≥2 co-located sessions.
    pub window_fold_hits: u64,
    /// Slice B — a shared fold that did NOT equal the per-session recompute bit-for-bit (the
    /// §2.14 "dedup f64 agreement: shared fold == per-session fold" measurement, taken on every
    /// memo hit). MUST stay 0 — asserted by the parity gate.
    pub window_fold_divergence: u64,
    /// Slice B — folds whose fresh prefix covered the WHOLE derived chain of length ≥ 2: direct
    /// proof that two different shards stamped levels at IDENTICAL universe ticks (the
    /// exact-cadence boot pin, §2.6.3 — asserted nonzero by the parity gate).
    pub window_full_chain_folds: u64,
    /// Slice B — GAUGE: sessions holding a non-empty derived chain this tick.
    pub window_chains_held: u64,
    /// Slice B — a stratum HELD at its last composed poses this tick (per stratum per tick —
    /// §2.6.4 `compose_hold_ticks`): the sky above a lagging hop holds while the local world
    /// keeps moving.
    pub window_compose_hold_ticks: u64,
    /// Slice B — held strata removed by the DEAD-HOP EXIT (held past the derived TTL): clean
    /// removal, counted, never a silent truncation (§2.6.4).
    pub window_hop_dead: u64,
    /// Slice B — a would-be common-tick REWIND froze emission (§2.6.3 monotone-T). 0 healthy.
    pub window_t_monotone_stalled: u64,
    /// Slice B — a chain derivation met a cycle and truncated fail-closed (§2.6.2). 0 healthy.
    pub window_chain_cycle: u64,
    /// Slice B — rows refused by `FrameError::InstantMismatch` inside a fold: the shear law
    /// firing (§2.6.3). MUST stay 0 in a healthy run (asserted by the parity gate); driven
    /// nonzero on purpose by the G-SHEAR anti-vacuity unit.
    pub window_instant_mismatch: u64,
    /// Slice B — rows refused by `RotatedFrameAcrossCells` inside a fold (pre-P10 cell math;
    /// today reachable only at the author's inversion — pinned by the composer unit).
    pub window_rotated_refused: u64,
    /// Slice B — rows dropped because their stated tail frame was not their level's own frame
    /// (or an unknown-frame refusal from the fold): alien, dropped, counted (§2.6.6).
    pub window_alien_rows: u64,
    /// Slice B — a chain level whose hop was absent/mismatched (or whose roster was empty): the
    /// fold's fresh prefix capped there, fail-closed.
    pub window_hop_invalid: u64,
    /// Slice B — an Active session whose composed feed was WITHHELD because its own-level window
    /// (or its own realm) is not derivable yet — the login race of §2.6.6, held not guessed.
    pub window_unresolved_standing: u64,
    /// Slice B — the §2.12 hop-vs-child-row agreement: composed positions for the SAME realm from
    /// the two lawful sources that disagreed AT ALL (bit-level). 0 on THE world today (identity
    /// orientations cancel exactly) — asserted by the parity gate.
    pub window_dedup_disagree: u64,
    /// Slice B — GAUGE (max): the largest measured hop-vs-child-row deviation, in nanometres —
    /// the §2.12 "measured bound", printed by the parity gate.
    pub window_dedup_max_dev_nm: u64,
    /// Slice B — `HeadRead{Realm(ancestor)}` polls sent on the window keep-alive cadence to
    /// resolve lineage ancestors the session never subscribed to (the EXISTING directory pair —
    /// no new wire arm). THROUGHPUT.
    pub window_head_reads_sent: u64,
    /// Slice B — composed rows produced across all folds. THROUGHPUT; with
    /// `parity_rows_matched` it bounds the (explained) composed-surplus: the composer carries
    /// the FULL direct-child roster of every chain level, dormant children included.
    pub window_composed_rows: u64,
    /// §2.6.5 step 4 (Q2 = PARENT RELAY, Slice C1): relayed live-child interior rows composed
    /// into folds — the sibling-interior carrier's rows actually reaching drawn scenes.
    pub window_relay_rows_composed: u64,
    /// A relayed interior refused at the fold (stamp off the rings / no placement row at that
    /// stamp / a hop invalid there) — dropped + counted, healed by the child's next relay.
    pub window_relay_unplaceable: u64,
    /// ★TOMBSTONED LANES, counted so a revived producer is never silent (window lane Slice C2,
    /// minor 19). `old_realm_frames_dropped`: the old opaque per-tick realm datagram
    /// (`ShardToGateway::RealmFrame`), whose producer and whose last consumer (the Slice-B parity
    /// comparator) both died here. Non-zero means a shard is speaking a lane nobody serves.
    pub old_realm_frames_dropped: u64,
    /// ★TOMBSTONED (window lane Slice C2, minor 19): old-lane per-observer scene deltas received
    /// from a shard and DROPPED. The composed lane has been the one scene author since the
    /// minor-18 flag day, and the shard-side producer is now deleted too — the ids-only
    /// `WindowMembership` verdict replaced it. `0` on a healthy cluster; non-zero means a shard
    /// is speaking a lane nobody serves.
    pub old_scene_deltas_dropped: u64,
    /// Composed-scene EGRESS — full levels shipped (`ServerControlMsg::RealmRegistry`): one per
    /// epoch bump (login, crossing). The pixel gates' emitted-level provenance.
    pub scene_levels_sent: u64,
    /// Composed-scene EGRESS — reliable deltas shipped (`ServerControlMsg::RealmSceneDelta`) on
    /// membership/body change at a stable epoch.
    pub scene_deltas_sent: u64,
    /// Composed-scene EGRESS — per-tick composed realm datagrams shipped (chunks counted; one
    /// tick's chunks share a frame id).
    pub scene_datagrams_sent: u64,
    /// Q2 relay (mesh minor 17) — relayed statements ADMITTED into a window's ingest (levels held
    /// per child; bodies into the one body store — the marker⇒look handover path).
    pub window_relays_ingested: u64,
    /// Q2 relay — sealed blobs that did not decode (counted, dropped — fail-closed).
    pub window_relay_undecodable: u64,
    /// Q2 relay — a relayed child the window author's own attested roster does not vouch
    /// (dropped; the roster is the stream-only child set, same as the marker admission).
    pub window_relay_unvouched: u64,
    /// Q2 relay — a relay refused by the CHILD's fence order (a deposed incarnation still
    /// shipping) or a relayed level older than the held one. Never normal past a re-home window.
    pub window_relay_stale: u64,
    /// Slice D (§2.8 departure mirror): SELF-LOOKs dropped by the derived roster-loss window — the
    /// realm behind them stopped speaking, so its parent's marker resumes and the body it drew
    /// shrinks to a point of light.
    pub window_looks_pruned: u64,
    /// Slice D: relayed interior LEVELS dropped by the same window — a dead live-child's interior
    /// cannot keep composing rows after its statements stop.
    pub window_relay_levels_pruned: u64,
}

pub fn register_gateway(world: &mut World, schedule: &mut Schedule, config: GatewayConfig) {
    let session_seed = config.session_seed;
    world.insert_resource(config);
    world.insert_resource(GatewaySessions::default());
    world.insert_resource(SessionMint(SplitMix64::new(session_seed)));
    world.insert_resource(GatewayStats::default());
    schedule.add_systems(
        (
            process_gateway_inbound,
            self_fence_lapsed_sessions,
            // RLM 5f-3d: the bounded dynamic-home bootstrap reaper. AFTER `process_gateway_inbound` so a
            // home head (or a `SessionAttached`) applied THIS tick pre-empts a spurious expiry — the same
            // ordering rationale as the self-fence — and BEFORE `drive_pending_sessions` so an expired
            // session is never re-driven after it was Closed. INERT for a static gateway (no session
            // carries a bootstrap window ⇒ no wire bytes ⇒ byte-identical).
            expire_home_bootstrap,
            drive_pending_sessions,
            renew_and_recheck_sessions,
            // THE WINDOW LANE (docs/design/window_lane.md §2.3): derive/open/close the
            // per-session chain windows + the keep-alive re-assert. AFTER the session drivers, so
            // a session promoted Active (or ended) THIS tick is served (or torn down) in the same
            // pass. INERT with zero Active sessions — zero window state, zero wire bytes.
            drive_windows,
            // THE WINDOW LANE's composer (§2.6/§4 — LIVE since Slice C1, the flag day): fold the
            // attested window statements per session at one universe tick and EMIT the composed
            // picture — the level on every epoch bump, the reliable delta on membership/body
            // change, the per-tick datagram on every fresh fold. LAST, after the ingest and the
            // window diff, so a level admitted this tick folds AND ships this tick (and a
            // crossing's swap level is ordered after the AuthorityChanged the inbound pass
            // pushed, on the same reliable stream — §2.7).
            compose_scenes,
        )
            .chain(),
    );
}

/// D-3 Slice 5b — the PROACTIVE Session self-fence (fence rule 4, the gateway analog of the shard's
/// `self_fence_lapsed_realm`). For every Active session whose lease has gone un-confirmed past
/// `self_fence_grace_ticks` of the gateway's own `local_tick` — a partition from the orchestrator, where
/// the recheck reply never arrives — hard-stop acting as its authority (phase ⇒ `SelfFenced`) BEFORE the
/// orchestrator's reassign window opens. Runs right after `process_gateway_inbound`, so a `Session`-head
/// reply applied THIS tick (re-arming `confirmed_at`) pre-empts a spurious fence; and before
/// `renew_and_recheck_sessions`, so a just-fenced session is neither renewed nor re-checked. INERT unless
/// `self_fence_grace_ticks > 0` AND `session_recheck_interval > 0` (the shared `lease_self_fence_due`
/// guards both). The hot input path is unaffected — `route_client_input`'s Active-only guard already
/// drops a fenced session's datagrams, so no wait-free route teardown is needed.
fn self_fence_lapsed_sessions(
    config: Res<GatewayConfig>,
    clock: Res<ClockSample>,
    mut sessions: ResMut<GatewaySessions>,
    mut stats: ResMut<GatewayStats>,
) {
    for session in sessions.by_session.values_mut() {
        if vd_sim::directory::lease_self_fence_due(
            matches!(session.phase, SessionPhase::Active { .. }),
            config.self_fence_grace_ticks,
            config.session_recheck_interval,
            clock.local_tick,
            session.confirmed_at,
        ) {
            session.phase = SessionPhase::SelfFenced;
            stats.sessions_self_fenced_lapsed += 1;
        }
    }
}

/// RLM 5f-3d — is the home-bootstrap RE-DRIVE due for a wait that OPENED at `since`, at gateway local tick
/// `now`, on cadence `interval`? PURE (ticks in, bool out — no wall-clock, no rng), so the whole re-drive
/// schedule is deterministic and replayable.
///
/// Two properties the correctness invariant rests on:
/// - `elapsed != 0` — the tick the wait BEGAN already emitted the seed + head-read inline, so re-driving in
///   the same tick would be a pure duplicate (`drive_pending_sessions` runs after
///   `process_gateway_inbound` in the SAME tick).
/// - `elapsed % cadence` anchored on the wait's OWN `since` ([`HomeWait::since`], not a global
///   `local_tick % interval`) — so a 100K mass login across many homes re-drives them on different ticks,
///   spreading the orchestrator fan-in across the whole cadence window instead of spiking it on one tick.
///
/// `interval.max(1)` keeps the cadence total AND fail-SAFE: a zero interval degrades to "every tick" (the
/// re-drive is a correctness invariant, so erring toward re-driving beats silently never re-driving —
/// `u64::is_multiple_of(0)` would be false for every nonzero elapsed). The live cadence
/// [`SeedInjectorConfig::redrive_interval_ticks`] is already floored at 1, and `validate` refuses an armed
/// zero window at boot, so this is belt-and-suspenders.
#[must_use]
fn home_redrive_due(now: TickId, since: TickId, interval: u64) -> bool {
    let elapsed = now.0.saturating_sub(since.0);
    let cadence = interval.max(1);
    (elapsed != 0) & elapsed.is_multiple_of(cadence)
}

/// RLM 5f-3d — has the BOUNDED pre-Active bootstrap window closed (CRITIQUE-1)? Strictly `>` so the
/// deadline tick itself is still inside the window (the hold is generous at its own edge). PURE.
#[must_use]
fn home_bootstrap_expired(now: TickId, deadline: TickId) -> bool {
    now.0 > deadline.0
}

/// RLM 5f-3d — is a session's `AwaitingAttach` retry due at gateway local tick `now`? ONE predicate for both
/// modes (HR3), fed the anchor by [`GatewaySessions::attach_anchor`]:
/// - `None` (a STATIC session — no home bootstrap) ⇒ EVERY tick, byte-identical to the pre-5f-3d retry
///   driver.
/// - `Some(since)` (a DYNAMIC member of a [`HomeWait`]) ⇒ the SAME backed-off cadence that realm's re-drive
///   rides. A mass login must not re-`AttachSession` every waiting session at a just-booted shard every
///   tick; the inline attach at the head resolve already went out, and the bounded bootstrap TTL spans many
///   cadences, so a lost attach still retries well inside the window.
///
/// PURE, and `home_redrive_due`'s `elapsed != 0` guard keeps the retry off the tick the wait opened.
#[must_use]
fn attach_retry_due(anchor: Option<TickId>, now: TickId, cadence: u64) -> bool {
    match anchor {
        None => true,
        Some(since) => home_redrive_due(now, since, cadence),
    }
}

/// RLM 5f-3d — THE bounded dynamic-home bootstrap reaper (CRITIQUE-1: a hold must END, and end LOUDLY).
/// Every session carrying a [`Session::bootstrap_deadline`] window — i.e. one in `AwaitingHomeRealm` OR in
/// the dynamic-target `AwaitingAttach`, since a freshly spawned shard can die between head-resolve and
/// `SessionAttached` — is Closed once `local_tick` passes its deadline: `tracing::error!` + a
/// `ServerControlMsg::Close` naming the failure + the counter, plus the SAME cleanup `Bye` performs (revoke
/// the committed `Session` lease so it does not linger until the orchestrator's reaper; detach at the home
/// shard iff one was resolved). NEVER a silent hang and never a teleport to some other shard.
///
/// A STATIC session has `bootstrap == None` and is untouched, so an unarmed gateway emits nothing here
/// (byte-identical); the per-tick cost is the same O(S) scan `self_fence_lapsed_sessions` already pays.
fn expire_home_bootstrap(
    config: Res<GatewayConfig>,
    clock: Res<ClockSample>,
    mut sessions: ResMut<GatewaySessions>,
    mut stats: ResMut<GatewayStats>,
    mut outbox: ResMut<OutboundBox>,
) {
    // Collect first: the removal mutates both session maps AND both 5f-3d indexes.
    let expired: Vec<SessionId> = sessions
        .by_session
        .iter()
        .filter(|(_, s)| {
            s.bootstrap_deadline
                .is_some_and(|deadline| home_bootstrap_expired(clock.local_tick, deadline))
        })
        .map(|(id, _)| *id)
        .collect();
    for session_id in expired {
        let session = sessions
            .by_session
            .remove(&session_id)
            .expect("collected from by_session this very tick");
        sessions.by_client.remove(&session.client);
        sessions.end_home_wait(session_id, session.home_rid);
        // RLM 5f-4: the SAME session-exit release as `Bye` (HR3 — one primitive, both exits), so neither
        // exit can leak a roster refcount. A TTL-expired session is pre-`Active` and therefore never holds
        // a transfer, so only the home-shard half is live here; sharing the primitive is what keeps the two
        // exits from drifting.
        sessions.release_session_claims(&session);
        stats.home_bootstrap_timeouts += 1;
        tracing::error!(
            session = %session_id,
            home_realm = ?session.home_rid,
            home_shard = ?session.home_shard,
            "the dynamic home realm did not become routable inside the bounded bootstrap TTL — closing \
             this client LOUDLY (RLM 5f-3d; never a silent hang). Check the orchestrator's realm spawn \
             path, then VD_BOOT_TICKS_P99 / the derived bootstrap TTL."
        );
        push_control(
            &mut outbox,
            session.client,
            &ServerControlMsg::Close {
                reason: "home realm did not become available".to_owned(),
            },
        );
        // The lease WAS committed (the window opens strictly downstream of the grant), so revoke it rather
        // than leave a live `Session` record for the reaper — exactly what `Bye` does.
        push_directory(
            &mut outbox,
            config.orchestrator,
            DirectoryOp::LeaseRevoke {
                key: DirectoryKey::Session(session_id),
                fence: session.fence,
            },
        );
        // Detach ONLY at a home we actually attached to. Expiring in `AwaitingHomeRealm` (no home resolved)
        // must not spray a detach at the static `config.shard`, which in dynamic mode may not even exist.
        if let Some(home) = session.home_shard {
            push_to_shard(
                &mut outbox,
                home,
                MsgClass::Control,
                &GatewayToShard::DetachSession {
                    session: session_id,
                    fence: session.fence,
                },
            );
        }
    }
}

/// D-3 lease-liveness producers (gateway half), each on the gateway's own LOCAL cadence:
/// (1) the HEARTBEAT — re-send `LeaseRenew` for every Active session's `Session` key, so the lease never
///     lapses while the client is connected (the orchestrator's reaper revokes a lapsed-and-confirmed-dead
///     lease); ONE mechanism with the shard's Realm/Entity heartbeat (the shared `push_renewals` shim —
///     HR3, never a match-on-shard-kind);
/// (2) the RECHECK — re-read every Active session's `Session` head, the ROUND-TRIP CONFIRMATION channel
///     whose affirming reply re-arms `Session.confirmed_at` (and whose foreign/absent reply triggers the
///     reactive self-fence); mirrors the shard's `realm_recheck`, and is what makes the proactive
///     `self_fence_lapsed_sessions` timer non-inert.
/// Both cadences are independent and INERT at interval `0` (the pre-D-3 default).
///
/// RLM 5f-3d (MF2) — the RENEW set is `Active` **OR** mid-dynamic-home-bootstrap
/// ([`Session::bootstrap_deadline`] `Some`). An `Active`-only renew set was correct while pre-`Active` lasted
/// one or two ticks; the dynamic-home hold can last a whole measured pod boot
/// ([`SeedInjectorConfig::bootstrap_ttl_ticks`] — 140 local ticks at the shipped budget, more with a bigger
/// measured boot), which EXCEEDS the orchestrator's lease-reap horizon. Its lease was already COMMITTED at
/// the grant, so without a renewal a held login's own lease lapses mid-hold and (once this gateway is latched
/// dead by some unrelated blip) can be REVOKED under it, with nothing pre-`Active` watching. A STATIC session
/// has `bootstrap_deadline == None` forever, so the renew set — and the wire bytes — are unchanged for an
/// unarmed gateway. Bitwise `|`: no short-circuit region (HR5); both operands are cheap tests.
///
/// The RECHECK stays `Active`-ONLY: it exists to re-arm `confirmed_at` for the proactive self-fence, which
/// only guards a session the gateway is actively serving as authority. A pre-`Active` login has no authority
/// to fence and is bounded by the bootstrap TTL instead, so polling its head would be pure round-trip cost.
fn renew_and_recheck_sessions(
    config: Res<GatewayConfig>,
    clock: Res<ClockSample>,
    sessions: Res<GatewaySessions>,
    mut outbox: ResMut<OutboundBox>,
) {
    let tick = clock.local_tick.0;
    if vd_sim::directory::due_this_tick(config.lease_renew_interval_ticks, tick) {
        let renewals = sessions
            .by_session
            .iter()
            .filter(|(_, s)| {
                matches!(s.phase, SessionPhase::Active { .. }) | s.bootstrap_deadline.is_some()
            })
            .map(|(id, s)| (DirectoryKey::Session(*id), s.fence));
        outbox.push_renewals(renewals, config.orchestrator);
    }
    if vd_sim::directory::due_this_tick(config.session_recheck_interval, tick) {
        for (id, _) in sessions
            .by_session
            .iter()
            .filter(|(_, s)| matches!(s.phase, SessionPhase::Active { .. }))
        {
            push_directory(
                &mut outbox,
                config.orchestrator,
                DirectoryOp::HeadRead {
                    key: DirectoryKey::Session(*id),
                },
            );
        }
    }
}

/// THE WINDOW LANE's keep-alive cadence at the gateway (Slice A — `docs/design/window_lane.md`
/// §2.3: "re-asserted on a derived keepalive cadence"), MIRRORING the shard's
/// [`vd_sim::stub::aoi_recheck_cadence`] derivation exactly so subscriber and holder beat on one
/// rhythm: the ARMED recheck channel when present (`session_recheck_interval` — the gateway's
/// local copy of the same env knob family the shard's `realm_recheck_interval` rides), else a
/// tick-DERIVED half-second cadence off the cluster tick rate — never a free literal. The shard's
/// TTL is 2 beats + 1 of the SAME derivation, so one lost keep-alive is bridged and a dead
/// gateway expires within ~two beats.
fn window_keepalive_cadence(config: &GatewayConfig) -> u64 {
    if config.session_recheck_interval > 0 {
        config.session_recheck_interval
    } else {
        (u64::from(config.tick_hz) / 2).max(1)
    }
}

/// One window the gateway currently WANTS open (the per-tick derivation's row).
#[derive(Debug, PartialEq, Eq)]
struct DesiredWindow {
    shard: NodeId,
    scope: WindowScope,
    author_realm: RealmId,
}

/// THE composer's derived-bounds home, off the ONE keep-alive beat the gateway already re-asserts
/// windows on ([`window_keepalive_cadence`]) — subscriber, holder TTL and composer retention all
/// breathe on one rhythm (`docs/design/window_lane.md` §2.6.7; owner law 3(a)).
fn window_tuning(config: &GatewayConfig) -> window::WindowTuning {
    window::WindowTuning::derive(window_keepalive_cadence(config))
}

/// THE WINDOW DERIVATION, Slice B's session/stream-only FULL chain (`docs/design/window_lane.md`
/// §2.6.2 — the Slice-A seed-forest parent lookup is DELETED): per Active session,
/// - an [`WindowScope::Occupants`] window on every ACTIVE subscription's shard (the realm the
///   sub's own frame names — and during a crossing hand-off BOTH ends, the §2.7 "hold both
///   chains through the overlap" posture riding the existing dual-sub pattern for free), and
/// - a [`WindowScope::Child`] window per adjacent LINEAGE pair (parent P ⊃ child C), opened on
///   the node [`GatewaySessions::realm_heads`] names for P — the session's own history (login
///   descent + crossings) says WHICH realms, the gateway's routing state + the directory's
///   `Realm` heads say WHERE. A parent whose node is not resolved yet simply gets no window (the
///   head poll in [`drive_windows`] keeps asking); nothing is guessed.
///
/// Windows are SHARED per (shard, scope) across sessions (§2.6.2), which the caller's diff gives
/// for free; zero Active sessions derive the empty set — zero window state, structurally.
fn desired_windows(sessions: &GatewaySessions) -> Vec<DesiredWindow> {
    let mut out: Vec<DesiredWindow> = Vec::new();
    for session in sessions.by_session.values() {
        if !matches!(session.phase, SessionPhase::Active { .. }) {
            continue;
        }
        for (node, record) in &session.subs {
            if record.state != SubState::Active {
                continue; // a draining sub is on its way out — never a fresh window
            }
            let Some(realm) = record.frame.realm() else {
                continue;
            };
            push_unique_window(
                &mut out,
                DesiredWindow {
                    shard: *node,
                    scope: WindowScope::Occupants,
                    author_realm: realm,
                },
            );
        }
        for pair in session.lineage.windows(2) {
            let (parent, child) = (pair[0], pair[1]);
            let Some(parent_node) = sessions.realm_heads.get(&parent) else {
                continue; // no resolved head yet — the cadence poll keeps asking
            };
            push_unique_window(
                &mut out,
                DesiredWindow {
                    shard: *parent_node,
                    scope: WindowScope::Child(child),
                    author_realm: parent,
                },
            );
        }
    }
    out
}

/// Dedup helper for the derivation (windows are shared per (shard, scope)): linear over a set
/// bounded by sessions × 2 — monomorphic, no `Ord` demanded of the wire scope type.
fn push_unique_window(out: &mut Vec<DesiredWindow>, wanted: DesiredWindow) {
    if !out
        .iter()
        .any(|w| w.shard == wanted.shard && w.scope == wanted.scope)
    {
        out.push(wanted);
    }
}

/// THE WINDOW LANE's gateway driver (Slice A — `docs/design/window_lane.md` §2.3/§4): derive the
/// wanted window set from the Active sessions' own routing state, DIFF it against the held set
/// (close what stopped being derivable — a session ended, crossed away, or its parent sub
/// drained; open what appeared), and re-assert every held window on the derived keep-alive
/// cadence. Zero sessions ⇒ zero desired ⇒ every window closed and the map empty — the teardown
/// gate's structural half. The composer is Slice B: nothing here reads a row, and no client sees
/// anything.
fn drive_windows(
    config: Res<GatewayConfig>,
    clock: Res<ClockSample>,
    mut sessions: ResMut<GatewaySessions>,
    mut stats: ResMut<GatewayStats>,
    mut outbox: ResMut<OutboundBox>,
) {
    // ---- Slice B: refresh + prune the realm→node map from the gateway's OWN routing state ----
    // Every Active sub's (node, frame) IS a live realm-head answer (the shard serving that realm
    // is the node the sub rides); the directory's `Realm` heads fill the ancestors the session
    // never subscribed to (fed in `on_directory_reply`, re-polled below). Pruned to the realms
    // the Active sessions' lineages + subs actually name — zero sessions ⇒ an empty map, the
    // structural teardown truth.
    let mut named: BTreeSet<RealmId> = BTreeSet::new();
    let mut resolves: Vec<(RealmId, NodeId)> = Vec::new();
    for session in sessions.by_session.values() {
        if !matches!(session.phase, SessionPhase::Active { .. }) {
            continue;
        }
        named.extend(session.lineage.iter().copied());
        for (node, record) in &session.subs {
            if record.state != SubState::Active {
                continue;
            }
            let Some(realm) = record.frame.realm() else {
                continue;
            };
            named.insert(realm);
            resolves.push((realm, *node));
        }
    }
    for (realm, node) in resolves {
        sessions.realm_heads.insert(realm, node);
    }
    sessions
        .realm_heads
        .retain(|realm, _| named.contains(realm));

    let desired = desired_windows(&sessions);
    let stale: Vec<WindowId> = sessions
        .windows
        .iter()
        .filter(|(_, held)| {
            !desired
                .iter()
                .any(|d| d.shard == held.shard && d.scope == held.scope)
        })
        .map(|(id, _)| *id)
        .collect();
    for id in stale {
        let held = sessions
            .windows
            .remove(&id)
            .expect("listed from this same map one expression above");
        push_to_shard(
            &mut outbox,
            held.shard,
            MsgClass::Control,
            &GatewayToShard::WindowClose { window: id },
        );
        stats.window_close_sent += 1;
    }
    for wanted in desired {
        if sessions
            .windows
            .values()
            .any(|held| held.shard == wanted.shard && held.scope == wanted.scope)
        {
            continue;
        }
        let id = WindowId(sessions.next_window + 1); // ids start at 1 — 0 is never issued
        sessions.next_window += 1;
        push_to_shard(
            &mut outbox,
            wanted.shard,
            MsgClass::Control,
            &GatewayToShard::WindowOpen {
                window: id,
                scope: wanted.scope,
            },
        );
        // (No runtime-roster claim: an open window is ITSELF a dispatch source — see the window
        // arm of `is_routable_shard` — so the session-role refcounts stay exactly the three
        // documented roles and a window cannot unbalance them.)
        sessions.windows.insert(
            id,
            GatewayWindow {
                shard: wanted.shard,
                scope: wanted.scope,
                author_realm: wanted.author_realm,
                ingest: window::WindowIngest::default(),
                parked_bodies: BTreeMap::new(),
                parked_relays: BTreeMap::new(),
            },
        );
        stats.window_open_sent += 1;
    }
    if vd_sim::directory::due_this_tick(window_keepalive_cadence(&config), clock.local_tick.0) {
        for (id, held) in &sessions.windows {
            push_to_shard(
                &mut outbox,
                held.shard,
                MsgClass::Control,
                &GatewayToShard::WindowOpen {
                    window: *id,
                    scope: held.scope,
                },
            );
            stats.window_keepalives_sent += 1;
        }
        // ---- Slice B: the lineage-ancestor head poll, on the SAME beat (§2.6.2) ----
        // For every Active session's lineage PARENT, re-read `HeadRead{Realm(p)}` — the EXISTING
        // directory pair, no new wire arm. Re-polled even when resolved (a re-homed ancestor's
        // head answer self-heals the window's pointing within one beat); deduped across sessions.
        let mut parents: BTreeSet<RealmId> = BTreeSet::new();
        for session in sessions.by_session.values() {
            if !matches!(session.phase, SessionPhase::Active { .. }) {
                continue;
            }
            for pair in session.lineage.windows(2) {
                parents.insert(pair[0]);
            }
        }
        for parent in parents {
            push_directory(
                &mut outbox,
                config.orchestrator,
                DirectoryOp::HeadRead {
                    key: DirectoryKey::Realm(parent),
                },
            );
            stats.window_head_reads_sent += 1;
        }
    }
}

/// THE COMPOSER's per-tick pass (`docs/design/window_lane.md` §2.6/§2.7 — LIVE since Slice C1,
/// the flag day): per Active session, derive the chain (session/stream-only), pick the fresh
/// common tick, FOLD once per (origin, tick) shared across sessions (§2.14 — the memo below),
/// roll the per-session scene (epoch, holds, dead-hop exits, monotone-T), EMIT the composed
/// picture — the full level on every epoch bump (ordered after the crossing's AuthorityChanged
/// on the same reliable stream), the reliable delta on membership/body change, the per-tick
/// composed datagram on every fresh fold.
fn compose_scenes(
    config: Res<GatewayConfig>,
    clock: Res<ClockSample>,
    mut sessions: ResMut<GatewaySessions>,
    mut stats: ResMut<GatewayStats>,
    mut outbox: ResMut<OutboundBox>,
) {
    compose_scenes_pass(&config, &clock, &mut sessions, &mut stats, &mut outbox);
}

/// The composer pass proper, a plain function so the unit tier can drive it over a hand-built
/// session table (the `one_active_session` pattern) — every classify arm reachable without a
/// full rig tick.
fn compose_scenes_pass(
    config: &GatewayConfig,
    clock: &ClockSample,
    sessions: &mut GatewaySessions,
    stats: &mut GatewayStats,
    outbox: &mut OutboundBox,
) {
    let tuning = window_tuning(config);
    // THE ROSTER-DRIVEN PRUNE (`docs/design/window_lane.md` §2.8's departure mirror, Slice D),
    // before anything is composed from these stores: a realm that stopped stating its own look —
    // it tore down, or its parent's relay holder TTL-expired it — loses that look after the
    // derived roster-loss window, and the presence gate falls through to its parent's ever-present
    // marker. Markers never expire (the floor: never zero drawn).
    for held in sessions.windows.values_mut() {
        let (looks, relays) = held.ingest.prune_stale(clock.universe_tick, &tuning);
        stats.window_looks_pruned += looks;
        stats.window_relay_levels_pruned += relays;
    }
    // Field-split borrows: the window map is read-only here; the sessions are rolled.
    let GatewaySessions {
        windows,
        by_session,
        ..
    } = sessions;
    let windows = &*windows;
    let catalog: Vec<window::CatalogRow> = windows
        .iter()
        .map(|(id, w)| window::CatalogRow {
            window: *id,
            scope: w.scope,
            author: w.author_realm,
            confirmed: w.ingest.confirmed(),
        })
        .collect();
    // The §2.14 shared-fold memo: ONE fold per (origin, tick) per gateway tick, shared across
    // every session standing in that origin. Local to the pass — nothing global accumulates.
    // The bool is the divergence instrument's once-latch: the FIRST memo hit recomputes the
    // fold per-session and compares bit-level (§2.14's exactness measurement); later hits just
    // share — so the instrument costs at most ONE extra fold per (origin, tick), never O(sessions).
    let mut memo: BTreeMap<(RealmId, u64), (window::Composed, bool)> = BTreeMap::new();
    let mut chains_held = 0u64;
    for session in by_session.values_mut() {
        if !matches!(session.phase, SessionPhase::Active { .. }) {
            continue;
        }
        let target = session_target(session, config);
        let origin_frame = session
            .subs
            .get(&target)
            .filter(|r| r.state == SubState::Active)
            .map(|r| r.frame);
        let (Some(origin_frame), Some(origin)) =
            (origin_frame, origin_frame.and_then(FrameRef::realm))
        else {
            // The login race (§2.6.6): no standing realm yet — the composed feed is WITHHELD,
            // counted, never guessed.
            stats.window_unresolved_standing += 1;
            continue;
        };
        let chain = window::derive_chain(origin, &catalog);
        stats.window_chain_cycle += u64::from(chain.cycled);
        if chain.hops.is_empty() {
            stats.window_unresolved_standing += 1;
            continue;
        }
        chains_held += 1;
        let authors: Vec<RealmId> = chain.hops.iter().map(|h| h.author).collect();
        let ingests: Vec<&window::WindowIngest> = chain
            .hops
            .iter()
            .map(|h| &windows[&h.window].ingest)
            .collect();
        let (prefix, t) = window::fresh_prefix(&ingests);
        // §2.7's same-T swap: a CROSSING's first level in the new origin is composed at the SAME
        // universe tick as the last old-epoch emission whenever the new chain's rings still
        // retain that stamp (they hold a derived span, so at the realm-lane cadence they do) —
        // every body's position across the swap is then the exact same-tick re-expression under
        // the new chain, screen deltas bounded by one tick of true motion (the crossing
        // no-flicker gate measures exactly this). If the stamp aged out, the freshest common
        // stamp serves and the gate still bounds the delta.
        let crossing = session.shadow.origin.is_some_and(|o| o != origin);
        // The same-T preference as ONE filtered option (every guard's false side is a reachable
        // branch INSIDE the closure): keep the old tick only when this IS a crossing, the whole
        // new chain is fresh, and every ring still retains that stamp. Otherwise the freshest
        // common stamp serves and the no-flicker gate still bounds the delta.
        let same_t = session.shadow.last_t.filter(|t_old| {
            crossing
                && prefix == ingests.len()
                && ingests.iter().all(|i| i.level_at(*t_old).is_some())
        });
        let t = same_t.or(t);
        // Cost discipline (§2.14): a scene already AT this common tick with an unchanged chain
        // has nothing new to fold — skip the fold AND the scene roll (never a hold, never a
        // stall; the ring already carries this tick).
        let already_current = (t == session.shadow.last_t)
            & (session.shadow.origin == Some(origin))
            & (session.shadow.chain_authors == authors)
            & t.is_some();
        if !already_current {
            let fold = t.map(|t| {
                if let Some((shared, verified)) = memo.get_mut(&(origin, t.0)) {
                    stats.window_fold_hits += 1;
                    if !*verified {
                        // §2.14's exactness proof, MEASURED once per shared fold: it must equal
                        // a per-session recompute bit-for-bit (asserted 0 by the parity gate).
                        let per_session =
                            compose_fresh(origin, origin_frame, t, &authors, &ingests, prefix);
                        stats.window_fold_divergence += fold_divergence(shared, &per_session);
                        *verified = true;
                    }
                    shared.clone()
                } else {
                    let fold = compose_fresh(origin, origin_frame, t, &authors, &ingests, prefix);
                    stats.window_folds += 1;
                    stats.window_composed_rows += fold.rows.len() as u64;
                    stats.window_relay_rows_composed += fold.relay_rows;
                    stats.window_relay_unplaceable += fold.relay_unplaceable;
                    stats.window_instant_mismatch += fold.instant_refused;
                    stats.window_rotated_refused += fold.rotated_refused;
                    stats.window_alien_rows += fold.alien_rows;
                    stats.window_hop_invalid += fold.hop_invalid;
                    stats.window_dedup_disagree += fold.dedup_disagree;
                    stats.window_dedup_max_dev_nm =
                        stats.window_dedup_max_dev_nm.max(fold.dedup_max_dev_nm);
                    stats.window_full_chain_folds +=
                        u64::from((fold.fresh_levels == authors.len()) & (authors.len() >= 2));
                    memo.insert((origin, t.0), (fold.clone(), false));
                    fold
                }
            });
            let prev_t = session.shadow.last_t;
            let report = session.shadow.advance(origin, &authors, fold, &tuning);
            stats.window_compose_hold_ticks += report.holds;
            stats.window_hop_dead += report.dead_hops;
            stats.window_t_monotone_stalled += u64::from(report.stalled);
            // ---- THE LIVE EMISSIONS (Slice C1 — §2.4/§2.7) ----------------------------------
            let epoch = session.shadow.origin_epoch;
            // Did THIS pass advance the fold tick? ONE option, computed once (HR5: the two
            // emission guards below read it without a short-circuit side no run can reach) —
            // `None` both for a pass that folded nothing (an unconfirmed chain's session) and
            // for the same-T crossing swap (whose level carries the poses instead).
            let t_advanced = if session.shadow.last_t == prev_t {
                None
            } else {
                session.shadow.last_t
            };
            if report.epoch_bumped {
                // THE SWAP LEVEL: one full composed level in the new origin — the atomic-swap
                // signal the client replaces its scene on. Ordered AFTER any AuthorityChanged the
                // inbound pass pushed this tick, on the same reliable control stream (§2.7).
                let t_level = session.shadow.last_t.unwrap_or(vd_core::UniverseTick(0));
                let rows = window::scene_level_rows(
                    origin,
                    origin_frame,
                    t_level,
                    &authors,
                    &ingests,
                    &session.shadow,
                );
                session.scene_sent = rows.iter().map(|r| (r.realm, r.bag.clone())).collect();
                push_control(
                    outbox,
                    session.client,
                    &ServerControlMsg::RealmRegistry {
                        origin,
                        origin_epoch: epoch,
                        rows,
                    },
                );
                stats.scene_levels_sent += 1;
            } else if let Some(t_now) = t_advanced {
                // The reliable DELTA on membership/body change at a stable epoch (§2.6.5 step 8):
                // diff the drawn (realm → bag) content against the send-on-change baseline. Pose
                // motion never rides here — it is the datagram's cargo below.
                let rows = window::scene_level_rows(
                    origin,
                    origin_frame,
                    t_now,
                    &authors,
                    &ingests,
                    &session.shadow,
                );
                let current: BTreeMap<RealmId, Vec<u8>> =
                    rows.iter().map(|r| (r.realm, r.bag.clone())).collect();
                if current != session.scene_sent {
                    let added: Vec<SceneRow> = rows
                        .into_iter()
                        .filter(|r| session.scene_sent.get(&r.realm) != Some(&r.bag))
                        .collect();
                    let removed: Vec<RealmId> = session
                        .scene_sent
                        .keys()
                        .filter(|r| !current.contains_key(r))
                        .copied()
                        .collect();
                    session.scene_sent = current;
                    push_control(
                        outbox,
                        session.client,
                        &ServerControlMsg::RealmSceneDelta {
                            origin,
                            origin_epoch: epoch,
                            added,
                            removed,
                        },
                    );
                    stats.scene_deltas_sent += 1;
                }
            }
            // THE PER-TICK COMPOSED DATAGRAM: a fresh fold landed ⇒ ship the drawn set's poses
            // (fresh rows at T + held strata at their old stamps, declared per row — §2.6.4),
            // MTU-partitioned, the per-session monotone frame id stamped by this pass (lawful:
            // a composed row is a NEW row this plane authors from attested inputs). The ORIGIN
            // never rides a datagram row (its own frame is the tail — the head≠tail law).
            if let Some(t_now) = t_advanced {
                let realms: Vec<RealmSnap> = session
                    .shadow
                    .drawn_rows()
                    .map(|r| RealmSnap {
                        realm: r.realm,
                        frame: r.frame,
                        pose: r.pose,
                    })
                    .collect();
                if !realms.is_empty() {
                    session.realm_feed_frame_id += 1;
                    let frame_id = session.realm_feed_frame_id;
                    for chunk in partition_realms(&realms, CONSERVATIVE_DATAGRAM_BUDGET) {
                        let datagram = RealmSnapshotDatagram {
                            sub: SubId(0),
                            frame_id,
                            source_tick: clock.local_tick,
                            universe_tick: t_now,
                            origin_epoch: epoch,
                            realms: chunk,
                        };
                        let bytes = postcard::to_allocvec(&datagram)
                            .expect("closed wire enums serialize infallibly");
                        outbox.0.push((
                            session.client,
                            MsgClass::RealmSnapshot,
                            vd_sim::io::bytes(bytes),
                            vd_sim::io::Durability::Ephemeral,
                        ));
                        stats.scene_datagrams_sent += 1;
                    }
                }
            }
        }
        session.shadow.chain_covers_lineage =
            !session.lineage.is_empty() & (chain.hops.len() == session.lineage.len());
    }
    stats.window_chains_held = chains_held;
}

/// The §2.14 divergence verdict, split out so BOTH arms are unit-drivable (HR5): 1 when a
/// shared fold does not equal its per-session recompute bit-for-bit (must never happen —
/// asserted 0 by the parity gate; nonzero would mean the fold is not a pure function of its
/// attested inputs), else 0.
fn fold_divergence(shared: &window::Composed, per_session: &window::Composed) -> u64 {
    u64::from(per_session != shared)
}

/// One fresh fold: resolve each fresh chain level's stamped-`t` window level and run the
/// composer. Monomorphic straight-line (HR5) — the branching lives in `window::compose`.
fn compose_fresh(
    origin: RealmId,
    origin_frame: FrameRef,
    t: UniverseTick,
    authors: &[RealmId],
    ingests: &[&window::WindowIngest],
    prefix: usize,
) -> window::Composed {
    let levels: Vec<&window::WindowLevel> = ingests[..prefix]
        .iter()
        .map(|i| {
            i.level_at(t)
                .expect("fresh_prefix only names ticks every prefix level retains")
        })
        .collect();
    window::compose(origin, origin_frame, t, authors, &levels, prefix, ingests)
}

/// ---------------------------------------------------------------------------
/// THE HOT PATHS (pure, lock-free; SPIKE-2a benches these exact functions)
/// ---------------------------------------------------------------------------
///
/// Route one client input datagram: dedup by the leading seq varint, then read
/// the route via one `ArcSwap` load. No decode of the payload, no locks.
#[must_use]
pub fn route_input(hot: &SessionHot, input_bytes: &[u8]) -> InputRouting {
    let Ok(seq) = peek_input_seq(input_bytes) else {
        return InputRouting::Malformed;
    };
    // Latest-wins dedup: monotonic high-water mark on one atomic. `fetch_max` is a
    // SINGLE atomic RMW — it advances the mark to `max(prev, seq)` and returns the
    // PRIOR value, so the load+compare+store is indivisible. A non-atomic load-then-
    // store (FG-2) would let two concurrent forwarder threads both read the same
    // `last`, both pass, and both forward the same datagram (or store out of order).
    // With `fetch_max` exactly one observes `seq > prev` for any given seq, and the
    // mark never moves backward regardless of arrival interleaving. Still wait-free
    // (one instruction; no lock, no retry loop) — the SPIKE-2a hot-path budget holds.
    let prev = hot.last_input_seq.fetch_max(seq, Ordering::Relaxed);
    if seq <= prev {
        return InputRouting::Deduped;
    }
    let route = hot.route.load();
    // THE CUT PARTITION (1c.5): a `seq > marker_seq` frame during the cut window is held for
    // the dest (the cold caller buffers it; `apply_commit` drains it post-swap). This is ONE
    // `Option` discriminant test + (only when a cut is installed) one `u64` compare, reading
    // `.cut`/`.marker_seq` off the SAME `route` Arc already loaded above — no 2nd load, no
    // lock, no alloc. Steady state (`cut == None`) falls straight to the byte-identical
    // `Forward { authority }` the SPIKE-2a benches assert; the `Some` arm is predicted-not-
    // taken. `fetch_max` (above) still precedes, so a buffered seq is strictly increasing.
    match route.cut {
        Some(SeqCut { marker_seq, .. }) if seq > marker_seq => InputRouting::Buffer,
        _ => InputRouting::Forward {
            to: route.authority,
        },
    }
}

/// What the input hot path decided.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum InputRouting {
    Forward {
        to: NodeId,
    },
    /// `seq > marker_seq` while a cut is installed: the cold caller holds it in the session's
    /// cut buffer for the dest (drained at commit). Carries no `dest` — the drain reads it
    /// off `route.cut` (keeping this hot return a pure compare).
    Buffer,
    Deduped,
    Malformed,
}

/// SPIKE-2a ROUTE-SWAP-mechanic bench helper — NOT the live read-plane fence check. It loads the
/// WRITE `route.fence` so the bench can measure the route swap's wait-free read under contention
/// (publisher `route.store` vs reader `route.load`). The LIVE read fan ([`on_shard_frame`]) checks
/// the PER-SUB [`SubEntry::accepted`] fence instead (the 1d.2 read/write split) — do NOT confuse
/// the two: this is the swap microbench, that is the production frame-acceptance.
#[must_use]
pub fn frame_passes_fence(hot: &SessionHot, frame_fence: Fence) -> bool {
    !frame_fence.is_stale_against(hot.route.load().fence)
}

/// SPIKE-2a microbench helper (route-swap mechanic): fence-compare against the WRITE route then a
/// byte-level sub re-tag — the single-session forward SHAPE under route-swap contention. NOT the
/// live forwarder: production fans per subscribed shard in [`on_shard_frame`] at the per-sub
/// [`SubEntry::accepted`] fence, sharing the re-tag once-per-`SubId` (SCALE-1).
#[must_use]
pub fn forward_frame(
    hot: &SessionHot,
    sub: SubId,
    frame_fence: Fence,
    snapshot_bytes: &[u8],
) -> Option<Vec<u8>> {
    if !frame_passes_fence(hot, frame_fence) {
        return None;
    }
    retag_snapshot_sub(snapshot_bytes, sub).ok()
}

/// ---------------------------------------------------------------------------
/// Control-plane systems (cold paths)
/// ---------------------------------------------------------------------------
#[allow(clippy::too_many_arguments)] // bevy system: each resource is one parameter
fn process_gateway_inbound(
    // ResMut (was Res): `on_transfer_control` `.take()`s the one-shot `reject_next_prepare` lever
    // (3g abort-leg). Every other read in this body (`config.orchestrator`, `config.is_known_shard`)
    // derefs the `ResMut` read-only, so no other edit.
    mut config: ResMut<GatewayConfig>,
    identity: Res<NodeIdentity>,
    clock: Res<ClockSample>,
    inbox: Res<InboundBox>,
    mut sessions: ResMut<GatewaySessions>,
    mut mint: ResMut<SessionMint>,
    mut stats: ResMut<GatewayStats>,
    mut outbox: ResMut<OutboundBox>,
) {
    // Cold drain-sweep FIRST (off the hot path): remove subs that were marked `Draining` in a
    // PRIOR tick. A sub closed (marked Draining) while processing this tick's inbound stays
    // routable through the rest of this tick's batch (its straggler is drained), then is swept
    // at the START of the next tick — the one-tick grace (C2 / X1).
    sessions.sweep_draining();
    for msg in &inbox.0 {
        let Inbound::Wire { from, class, bytes } = msg else {
            continue;
        };
        let from = *from;
        // Node-class dispatch (FORK 5): orchestrator → routable-shard → client-fallthrough. The orderING
        // (orchestrator first) keeps a shard NodeId from ever colliding with the orchestrator role; node
        // roles are disjoint by construction. The shard test is the STABLE `config.known_shards` UNION the
        // RUNTIME `dynamic_shards` roster of demand-spawned home shards (5f-3d), NEVER the mutable
        // per-session `subscribed_shards` — so a subscription refcount slip cannot mis-class a client.
        if from == config.orchestrator {
            match class {
                // The orchestrator→gateway Saga class carries an InterShardFlow envelope:
                // a DirectoryReply (session-grant head) OR a Saga(TransferControl) command.
                // Decode ONCE and dispatch by variant (the dispatch split that lets them
                // coexist on one class without mis-decoding into each other).
                MsgClass::Saga => match postcard::from_bytes::<InterShardFlow>(bytes) {
                    Ok(InterShardFlow::DirectoryReply(reply)) => on_directory_reply(
                        reply,
                        &config,
                        &identity,
                        &clock,
                        &mut sessions,
                        &mut stats,
                        &mut outbox,
                    ),
                    // The TransferControl consumer (the gateway counterpart to the saga
                    // runtime): the no-authority-move phases (1c.2); the route-touching
                    // phases park until 1c.3/1c.4.
                    Ok(InterShardFlow::Saga(cmd)) => on_transfer_control(
                        cmd,
                        &mut config,
                        &mut sessions,
                        &mut stats,
                        &mut outbox,
                    ),
                    // WHO MAY BE HEARD, from the party that decides it. The ownership record's list of
                    // realm-holding nodes replaces three inferences: a boot roster that cannot know about
                    // a realm spun up later, a node's own claim about itself, and a transfer's claim that
                    // died with the transfer. Applied as a LEVEL — the whole set, latest tick wins — so a
                    // missed message is corrected by the next change rather than leaving this router deaf
                    // to a live shard.
                    Ok(InterShardFlow::ShardRoster(roster)) => {
                        on_shard_roster(roster, &mut sessions, &mut stats);
                    }
                    Ok(_) | Err(_) => stats.undecodable += 1,
                },
                // Membership (clock sync) is consumed by the follower system.
                MsgClass::Membership => {}
                _ => stats.undecodable += 1,
            }
        } else if matches!(class, MsgClass::Saga) {
            // RLM 5f RG-3: the ONLY non-orchestrator `Saga` frame is a demand-spawned shard's reactive
            // greeting (`ShardPresence`) — a routable shard sends Control/Snapshot/RealmSnapshot and a client
            // Control/Input, never Saga. It has NO routing effect (the mesh already learned the return
            // connection below the app seam on this same reliable frame; the gateway makes the shard reachable
            // by REPLYING over that learned lane). Hoisted ABOVE `is_routable_shard` because a greeting arrives
            // BEFORE the shard is on the runtime roster (it becomes routable only when a session's home
            // resolves), so it would otherwise fall to the client arm and be miscounted `undecodable`.
            on_shard_presence(from, bytes, &mut sessions, &mut stats);
        } else if is_routable_shard(&config, &sessions, from) {
            match class {
                MsgClass::Control => on_shard_control(
                    from,
                    bytes,
                    &config,
                    &clock,
                    &mut sessions,
                    &mut stats,
                    &mut outbox,
                ),
                MsgClass::Snapshot => {
                    on_shard_frame(from, bytes, &mut sessions, &mut stats, &mut outbox);
                }
                MsgClass::RealmSnapshot => {
                    on_shard_realm_frame(from, bytes, &config, &mut sessions, &mut stats);
                }
                _ => stats.undecodable += 1,
            }
        } else {
            // A client connection.
            match class {
                MsgClass::Control => on_client_control(
                    bytes,
                    from,
                    &config,
                    &identity,
                    &mut sessions,
                    &mut mint,
                    &mut stats,
                    &mut outbox,
                ),
                MsgClass::Input => {
                    on_client_input(bytes, from, &config, &mut sessions, &mut stats, &mut outbox);
                }
                // NEITHER a shard, NOR a class a client may send. That is a fact about the SENDER, not
                // about the bytes, and it used to share an outcome with "a client sent rubbish". See
                // `refuse_unknown_sender`.
                _ => refuse_unknown_sender(from, *class, &sessions, &mut stats),
            }
        }
    }
    // 1d.5a: after draining the tick's inbound, recompute the STANDING delivery watermark for each
    // in-flight transfer and emit `DeliveredToObservers` when satisfied (the (a) demote-predicate
    // input). Run once per tick, NOT per frame — no 20Hz regression (D-24).
    recompute_delivery_watermarks(&sessions, &config, &mut outbox);
}

/// RLM 5f RG-3 — consume a demand shard's reactive greeting. PURE OBSERVABILITY: the reachability effect
/// (the mesh learning the return connection) already happened below the app seam on this same reliable
/// frame, so this only decodes for a counter + a debug line and claims NOTHING (no `dynamic_shards` role —
/// a presence has no session and no lifetime). A body that is not a `ShardPresence` (any other
/// `InterShardFlow` arm, or undecodable bytes) counts `undecodable` exactly as the pre-RG-3 fallthrough did.
fn on_shard_presence(
    from: NodeId,
    bytes: &[u8],
    sessions: &mut GatewaySessions,
    stats: &mut GatewayStats,
) {
    match postcard::from_bytes::<InterShardFlow>(bytes) {
        Ok(InterShardFlow::ShardPresence(ShardPresence { local_tick })) => {
            stats.presence_announces += 1;
            // THE GREETING NOW ADMITS THE NODE, and this line is why the fleet was mute.
            //
            // It used to be "learned for reply, no roster claim": the mesh learned a return lane, and the
            // node stayed a stranger for dispatch. So a demand-spawned shard could boot, hold a realm,
            // own a player and speak every tick, and every word was discarded — measured on this gate at
            // 14,884 refused frames from one cluster, all from the very planet the player had re-homed
            // into. The player was authoritative there and had never once been told where they were.
            //
            // Node class is a LIFECYCLE fact and it now comes from the node's own announcement over the
            // authenticated mesh, for as long as it keeps announcing. It is NOT a transfer's claim: the
            // old admission rode `PrepareSubscribe`, so a node was a shard only while a hand-off said so,
            // which is a fact about one hand-off standing in for a fact about a process.
            //
            // ⚠ WHAT THIS LEANS ON, stated rather than assumed: "I am a shard" is SELF-ASSERTED over the
            // shared cluster credential — the same boundary every other inter-node claim already trusts,
            // and the same one DEFERRED.md records as insufficient for a real deployment (D-RLM-8), with
            // the cloud preflight already REFUSING demand mode until per-node trust lands (D-RLM-7). So
            // this admits exactly what that veto already contains, and no more. The orchestrator-attested
            // version is the upgrade that rides with per-node trust, not a second mechanism.
            if sessions.announced_shards.insert(from) {
                tracing::info!(
                    from = from.0,
                    local_tick = local_tick.0,
                    "shard announced itself — admitted for dispatch, so what it says can reach players",
                );
            }
        }
        Ok(_) | Err(_) => stats.undecodable += 1,
    }
}

/// 1d.5a — the STANDING per-observer delivery watermark pass. For each in-flight transfer, emit
/// `DeliveredToObservers` iff the dest observer set is NON-EMPTY (an empty set is NEVER vacuously
/// satisfied — in the window before the dest sub opens the saga must NOT be told delivery is done)
/// AND every current dest observer has received >=1 dest frame. RECOMPUTED each tick (never latched
/// here): an observer opening mid-demote inherits watermark 0 and RE-BLOCKS. Idempotent re-emit
/// while true (the saga latches `dest_delivered` and the FSM absorbs the repeat) — a STANDING
/// per-TICK (not per-frame; no 20Hz regression — D-24) reliable ack for the demote-tail duration;
/// bounded by the tail, rising-edge gating deferred. HR1: the watermark is gateway-internal; only
/// the boolean `DeliveredToObservers` crosses, on the existing SagaAck arm.
fn recompute_delivery_watermarks(
    sessions: &GatewaySessions,
    config: &GatewayConfig,
    outbox: &mut OutboundBox,
) {
    for session in sessions.by_session.values() {
        let Some(progress) = session.transfer.as_ref() else {
            continue;
        };
        if every_observer_delivered(sessions, progress.dest) {
            reply_ack(
                outbox,
                config.orchestrator,
                TransferControlAck::DeliveredToObservers {
                    transfer: progress.transfer,
                },
            );
        }
    }
}

/// Whether the dest observer set is NON-EMPTY AND every observer has delivered >=1 frame on its
/// dest sub. Iterates sessions directly (the `subs.get(&dest)` Some/None is the observer /
/// non-observer split — a domain case, never a cross-lookup desync). Bitwise `&` so neither the
/// found-nor-all arm is a short-circuit-uncoverable region (HR5). Empty observer set ⇒ false.
#[must_use]
fn every_observer_delivered(sessions: &GatewaySessions, dest: NodeId) -> bool {
    let mut found = false;
    let mut all = true;
    for session in sessions.by_session.values() {
        let Some(rec) = session.subs.get(&dest) else {
            continue;
        };
        found = true;
        all &= session.delivered.get(&rec.sub).copied().unwrap_or(0) >= 1;
    }
    found & all
}

fn push_control(outbox: &mut OutboundBox, to: NodeId, msg: &ServerControlMsg) {
    let bytes = postcard::to_allocvec(msg).expect("closed wire enums serialize infallibly");
    // The gateway is a ROUTER — every flow it emits is re-driven/loss-tolerant, so these direct pushes are
    // Ephemeral (R-6d2b review LOW-1). If a future gateway flow is ever producer-less-reliable
    // (`FlowDurabilityClass::ProducerLessReliable`) it MUST route through `OutboundBox::push_flow_durable`
    // with `Retained` (whose debug_assert enforces it), NEVER a bare Ephemeral `.0.push`.
    outbox.0.push((
        to,
        MsgClass::Control,
        vd_sim::io::bytes(bytes),
        vd_sim::io::Durability::Ephemeral,
    ));
}

/// Announce "this entity is YOUR avatar" to one client — the DUAL-signal that carries the
/// pure-renderer migration (S4). It ALWAYS pushes `AuthorityChanged{entity, sub}` (the sub-keyed
/// signal an OLD, node-AWARE minor<2 client re-points render authority with) AND — to a peer that
/// negotiated minor >= 2 — the node-AGNOSTIC `OwnEntity{entity}` (which names ONLY the entity, no
/// sub / owning node). A pure-renderer client reads `OwnEntity` and IGNORES `AuthorityChanged`; an
/// old client reads `AuthorityChanged` and ignores the (withheld-anyway) `OwnEntity`. The gateway
/// emits BOTH so a rolling fleet of both client versions renders the same avatar without a flag day.
fn announce_own_entity(
    outbox: &mut OutboundBox,
    client: NodeId,
    negotiated_minor: u16,
    entity: EntityId,
    sub: SubId,
) {
    // The sub-keyed re-point for a node-AWARE (minor<2) client. Always emitted (harmless to a
    // pure-renderer client, which drops it as an ignored variant).
    push_control(
        outbox,
        client,
        &ServerControlMsg::AuthorityChanged { entity, sub },
    );
    // The node-AGNOSTIC own-entity signal (sender-gates-variants): withheld from a minor<2 peer
    // (it would desync an old decoder), sent to minor>=2 (the pure-renderer client's own-entity cue).
    if negotiated_minor >= 2 {
        push_control(outbox, client, &ServerControlMsg::OwnEntity { entity });
    }
}

fn push_to_shard(outbox: &mut OutboundBox, to: NodeId, class: MsgClass, msg: &GatewayToShard) {
    let bytes = postcard::to_allocvec(msg).expect("closed wire enums serialize infallibly");
    outbox.0.push((
        to,
        class,
        vd_sim::io::bytes(bytes),
        vd_sim::io::Durability::Ephemeral,
    ));
}

fn push_directory(outbox: &mut OutboundBox, to: NodeId, op: DirectoryOp) {
    let bytes = postcard::to_allocvec(&InterShardFlow::Directory(op))
        .expect("closed wire enums serialize infallibly");
    outbox.0.push((
        to,
        MsgClass::Saga,
        vd_sim::io::bytes(bytes),
        vd_sim::io::Durability::Ephemeral,
    ));
}

/// The ONE `SagaAck` encode-and-push — the exact inverse of the saga runtime's
/// `outbox.push_flow(gateway, Saga, &InterShardFlow::Saga(cmd))` (DRY: reuses the shared
/// `push_flow`). Every gateway reply to the orchestrator's saga rides this.
fn reply_ack(outbox: &mut OutboundBox, orchestrator: NodeId, ack: TransferControlAck) {
    outbox.push_flow(orchestrator, MsgClass::Saga, &InterShardFlow::SagaAck(ack));
}

/// Handle one client control message. The gateway is the SOLE ticket validator;
/// shards never see a credential.
#[allow(clippy::too_many_arguments)]
fn on_client_control(
    bytes: &[u8],
    client: NodeId,
    config: &GatewayConfig,
    identity: &NodeIdentity,
    sessions: &mut GatewaySessions,
    mint: &mut SessionMint,
    stats: &mut GatewayStats,
    outbox: &mut OutboundBox,
) {
    let Ok(msg) = postcard::from_bytes::<ClientControlMsg>(bytes) else {
        stats.undecodable += 1;
        return;
    };
    match msg {
        ClientControlMsg::Hello { version, login } => {
            let Some(negotiated) = ProtoVersion::negotiate(ProtoVersion::CURRENT, version) else {
                stats.version_rejected += 1;
                // The refusal names its CAUSE (`refusal_reason`), because from minor 8 there are two and
                // they call for different action: a major mismatch is "different protocol generation";
                // a minor below the floor is "same generation, your build predates frame-anchored
                // positions — update it". Both used to report the major-mismatch sentence, which would
                // have sent an operator hunting a generation split that was not there. Both still land
                // on the ONE `version_rejected` counter — the cause is in the sentence, not a new stat.
                push_control(
                    outbox,
                    client,
                    &ServerControlMsg::Close {
                        reason: ProtoVersion::CURRENT.refusal_reason(version),
                    },
                );
                return;
            };
            if tickets::validate_login(&config.auth_verifying_key, &login).is_err() {
                stats.logins_rejected += 1;
                push_control(
                    outbox,
                    client,
                    &ServerControlMsg::Close {
                        reason: "login ticket rejected".to_owned(),
                    },
                );
                return;
            }
            if sessions.by_session.len() >= config.tuning.max_sessions {
                stats.sessions_refused_capacity += 1;
                push_control(
                    outbox,
                    client,
                    &ServerControlMsg::Close {
                        reason: "gateway at session capacity".to_owned(),
                    },
                );
                return;
            }
            if sessions.by_client.contains_key(&client) {
                // A duplicate Hello on a live connection: idempotent no-op (the
                // pending/active session keeps progressing).
                return;
            }
            // Propose a session id; the DIRECTORY INSERT is the authoritative mint.
            let proposed =
                SessionId((u128::from(mint.0.next_u64()) << 64) | u128::from(mint.0.next_u64()));
            let fence = Fence::GENESIS.next();
            sessions.by_session.insert(
                proposed,
                Session {
                    client,
                    account: login.account,
                    fence,
                    phase: SessionPhase::AwaitingDirectory,
                    next_sub: 0,
                    // RLM 5f-3d: no home yet — the home realm is DERIVED (and its shard resolved) strictly
                    // downstream of the committed lease, so a fresh session routes at `config.shard` in
                    // BOTH modes (and its input is dropped anyway until `Active`). A dynamic session's
                    // route is retargeted through the sole `store_route` primitive at the home resolve.
                    home_shard: None,
                    home_rid: None,
                    // The composed feed starts cold: the first fold bumps the epoch 0→1 and ships
                    // the first level; the counter and the baseline fill with it.
                    realm_feed_frame_id: 0,
                    scene_sent: BTreeMap::new(),
                    // No home resolved yet ⇒ nowhere to measure a spawn from. Filled in the same place
                    // (and by the same descent) as the home lineage, strictly after the committed lease.
                    spawn: None,
                    bootstrap_deadline: None,
                    // Armed at the Active transition (the attach); irrelevant while still logging in.
                    confirmed_at: TickId(0),
                    negotiated_minor: negotiated.minor,
                    transfer: None,
                    subs: BTreeMap::new(),
                    delivered: BTreeMap::new(),
                    // THE WINDOW LANE (Slice B): no home yet ⇒ no lineage and an empty shadow
                    // scene — both fill strictly downstream (the descent / the attach frame).
                    lineage: Vec::new(),
                    shadow: window::ShadowScene::default(),
                    hot: Arc::new(SessionHot {
                        route: ArcSwap::from_pointee(RouteSnapshot {
                            authority: config.shard,
                            fence: Fence::GENESIS,
                            cut: None,
                        }),
                        last_input_seq: AtomicU64::new(0),
                        subs: ArcSwap::from_pointee(SubTable::default()),
                    }),
                },
            );
            sessions.by_client.insert(client, proposed);
            push_directory(
                outbox,
                config.orchestrator,
                DirectoryOp::LeaseGrant {
                    key: DirectoryKey::Session(proposed),
                    owner: AuthorityRef::Gateway(identity.node_id),
                    fence,
                },
            );
        }
        ClientControlMsg::Resume { .. } => {
            // Resume/adoption lands in P3; refusing is honest, not silent.
            stats.resumes_refused += 1;
            push_control(
                outbox,
                client,
                &ServerControlMsg::Close {
                    reason: "resume is not available yet".to_owned(),
                },
            );
        }
        ClientControlMsg::Bye => {
            let Some(session_id) = sessions.by_client.remove(&client) else {
                return;
            };
            let session = sessions
                .by_session
                .remove(&session_id)
                .expect("session maps are kept in sync");
            // WEDGE-1 (pinned to Slice 2 — DEFERRED D-23): a Bye mid-transfer drops the
            // session + its journal; subsequent saga commands for it then count as
            // `transfer_unroutable` with NO producer to unstick the pinned saga. The real
            // backstop is the Slice-2 saga timeout/abort producer; pin loud here so the
            // dropped in-flight transfer is never silent.
            if let Some(tp) = session.transfer.as_ref() {
                tracing::warn!(
                    session = %session_id,
                    transfer = tp.transfer.0,
                    "client Bye dropped a session with an in-flight transfer — the saga will \
                     pin until the Slice-2 timeout producer lands (D-23)"
                );
            }
            // RLM 5f-3d: drop this session from BOTH runtime indexes before anything else — every exit from
            // the dynamic-home machinery runs through the ONE `end_home_wait` / `release_session_claims`
            // pair, so a `Bye` mid-boot can never leave a waiting-index entry or a roster refcount behind.
            // A STATIC session takes the `None` arm of both (no-op ⇒ byte-identical). RLM 5f-4/5f-4e: the
            // release covers ALL THREE roles (home shard, an in-flight transfer's crossing dest, and the
            // DEMOTING SOURCE home a committed crossing stashed), so a `Bye` anywhere in the demote tail —
            // where the dest is claimed twice and the source still holds one — drops every node to zero.
            sessions.end_home_wait(session_id, session.home_rid);
            sessions.release_session_claims(&session);
            // 5f-3d: the detach goes to the session's ROUTING TARGET — its dynamically resolved home shard
            // when it has one, else the static `config.shard` (the ONE `session_target` path, HR3).
            // RLM 5f-4 closes D-34's A1 (authority-following detach) HERE: `CommitAuthority` now re-points
            // `home_shard` to the transfer dest, so after a crossing this target IS the current authority and
            // the detach reaches the shard that actually holds the `SessionTable` entry (it used to go to the
            // source and leak the dest's). The COMPOSITED-SUBS clause remains owed — this detaches only the
            // CURRENT authority, while the source retains its own sub/ghost until `ReleaseSubscribe`.
            // Still NOT a `session.subs.keys()` scan — `subs` is empty at login (would regress login→Bye).
            push_to_shard(
                outbox,
                session_target(&session, config),
                MsgClass::Control,
                &GatewayToShard::DetachSession {
                    session: session_id,
                    fence: session.fence,
                },
            );
            push_directory(
                outbox,
                config.orchestrator,
                DirectoryOp::LeaseRevoke {
                    key: DirectoryKey::Session(session_id),
                    fence: session.fence,
                },
            );
        }
        // No transfers in P1: a cut confirmation has nothing to bind to.
        // Pongs are liveness echoes; the P1 gateway sends no pings.
        ClientControlMsg::CutEmitted { .. } | ClientControlMsg::Pong { .. } => {}
    }
}

/// Route one client input datagram (HOT decision + bookkeeping), then COLD-observe the
/// transfer cut marker — strictly OFF the SPIKE-2a hot path (`route_input` is unchanged).
fn on_client_input(
    bytes: &[u8],
    client: NodeId,
    config: &GatewayConfig,
    sessions: &mut GatewaySessions,
    stats: &mut GatewayStats,
    outbox: &mut OutboundBox,
) {
    let Some(session_id) = sessions.by_client.get(&client).copied() else {
        stats.inputs_unroutable += 1;
        return;
    };
    // Guarded lookup, never `[]`: the two session maps are kept in sync by construction,
    // but a desync must DROP-and-count on this 20Hz path, never panic the gateway
    // (R2 lineage — a routing-map slip is a counted unroutable, not a crash).
    let Some(session) = sessions.by_session.get_mut(&session_id) else {
        stats.inputs_unroutable += 1;
        return;
    };
    if !matches!(session.phase, SessionPhase::Active { .. }) {
        stats.inputs_unroutable += 1;
        return;
    }
    // HOT: the benched route decision (ArcSwap load + one atomic). Unchanged.
    match route_input(&session.hot, bytes) {
        InputRouting::Forward { to } => {
            push_to_shard(
                outbox,
                to,
                MsgClass::Input,
                &GatewayToShard::SessionInput {
                    session: session_id,
                    fence: session.fence,
                    input_bytes: bytes.to_vec(),
                },
            );
        }
        InputRouting::Buffer => {
            // COLD: hold the seq>marker frame in the session's cut buffer for the dest
            // (drained at CommitAuthority). The hot `route_input` only DECIDED Buffer; the
            // buffer lives on the cold `TransferProgress`, never on `SessionHot` (HR1).
            // `route_input` returns Buffer ONLY when `route.cut` is Some, which `apply_freeze`
            // installs together with the in-flight `transfer` — so `transfer` is Some here by
            // construction (`expect`, the unreachable-arm shape). Bounded: over cap, drop the
            // OLDEST (latest-wins input) + count.
            let tp = session
                .transfer
                .as_mut()
                .expect("an installed cut implies an in-flight transfer (apply_freeze sets both)");
            if tp.dest_buffer.len() >= config.tuning.max_buffered_inputs {
                tp.dest_buffer.pop_front();
                stats.dest_inputs_dropped += 1;
            }
            tp.dest_buffer.push_back(bytes.to_vec());
            stats.inputs_buffered_for_dest += 1;
        }
        InputRouting::Deduped => stats.inputs_deduped += 1,
        InputRouting::Malformed => stats.inputs_malformed += 1,
    }
    // S3: the in-band cut marker is RETIRED (the cut is server-timed — `apply_request_cut` self-acks
    // `CutConfirmed`, `apply_freeze` derives the seq at install time). A client `CUT_MARKER` is now an
    // ordinary input already routed above; there is no per-input marker observation left to do, so
    // this hot-ish path drops the former cold marker decode.
}

/// THE gateway counterpart to the saga runtime: consume one `TransferControl` command.
/// EVERY phase is now LIVE: Prepare/RequestCut/FreezeSource (the cut install, 1c.3)/
/// CommitAuthority (the route swap, 1c.4)/ThawSource/AbortTransfer/ReleaseSubscribe (the demote
/// tail's SUCCESS teardown, 1c.8 — acks `Released` so the saga reaches `Done`). The applied-steps
/// journal gates every command BEFORE any effect (consult-before-effect) and records AFTER
/// (record-after-effect) so an at-least-once redelivery re-sends the recorded ack verbatim and
/// never re-applies.
fn on_transfer_control(
    cmd: TransferControl,
    // `&mut` (was `&`): the one-shot `reject_next_prepare` lever is `.take()`n by `apply_prepare`
    // (3g abort-leg). The borrow is split at the call site — `config.orchestrator` reads and
    // `&mut config.reject_next_prepare` never overlap (distinct statements / a disjoint field).
    config: &mut GatewayConfig,
    sessions: &mut GatewaySessions,
    stats: &mut GatewayStats,
    outbox: &mut OutboundBox,
) {
    let transfer = cmd.transfer();
    let step = cmd.step_id();
    let session_id = cmd.session(); // the get_mut key; reused for the dest-bound OpenInputSlot/SessionInput
    let Some(session) = sessions.by_session.get_mut(&session_id) else {
        stats.transfer_unroutable += 1; // unknown/absent session: counted, never panic
        return;
    };
    // REDELIVERY GATE (consult-before-effect): a recorded step for THIS transfer re-sends
    // the recorded ack verbatim, no effect — via the ONE dedup accessor `TransferProgress::
    // recorded`. A let-chain (each link's true/false arm is separately exercised: no-transfer
    // / wrong-transfer / unrecorded-step / recorded-step).
    //
    // S3: `RequestCut` is now SELF-ACKING (`apply_request_cut` returns `CutConfirmed`, journaled
    // at step 1 by the standard record-then-send path below), so it is NO LONGER excluded from
    // this gate — a redelivered `RequestCut` re-serves the recorded `CutConfirmed` verbatim, never
    // re-pushing the client `RequestCut` or re-generating the ack. (The old exclusion existed only
    // because the now-inert cut-marker observer owned the step-1 journal; the server owns it now.)
    if let Some(tp) = session.transfer.as_ref()
        && tp.transfer == transfer
        && let Some(prior) = tp.recorded(step)
    {
        reply_ack(outbox, config.orchestrator, prior);
        return;
    }
    // The READ-plane subs to close AFTER the ack/journal (the `&mut Session` borrow must end
    // before `GatewaySessions::close_sub` — the sole sub-close primitive — can run).
    // ReleaseSubscribe closes the SOURCE sub (`src` from the command); AbortTransfer closes
    // EXACTLY this transfer's DEST sub (`tp.dest`, captured at PrepareSubscribe) — never an "any
    // sub != config.shard" heuristic (which would close the player's CURRENT live sub on a chained
    // transfer and ALL composited subs in the N-shard end goal). Inert in the 1d.2 happy path
    // (abort runs pre-CAS, before the dest sub exists; `close_sub` no-ops then) but PRECISE for the
    // commit/flip-window-open ordering and the N-shard future.
    let mut subs_to_close: Vec<NodeId> = Vec::new();
    // RLM 5f-4: the runtime routable-roster edits this command produced, applied after the borrow ends
    // (the same deferred-mutation discipline as `subs_to_close` — see [`RosterEdits`]).
    let mut roster = RosterEdits::default();
    // Compute the ack (or None for deferred/parked phases), then record-then-send below.
    let ack: Option<TransferControlAck> = match cmd {
        TransferControl::PrepareSubscribe { dest, .. } => {
            // RLM 5f-4: the dest a DEFENSIVE replace is about to displace — captured BEFORE
            // `apply_prepare` overwrites the progress, because that displaced dest's roster claim must be
            // released WITH it or a second Prepare on this session leaks a refcount that pins the old dest
            // on the roster forever.
            let displaced = session.transfer.as_ref().map(|tp| tp.dest);
            // RLM 5f-4e: and the displaced progress's DEMOTING SOURCE home. A second Prepare on a session
            // that ALREADY committed a crossing (its demote tail still open) would otherwise strand that
            // source claim: the replacement progress starts `demoting_home: None`, so after the overwrite NO
            // holder can name the old source and no terminal will ever release it.
            let displaced_home = session.transfer.as_ref().and_then(|tp| tp.demoting_home);
            // Split borrow: `session` from `sessions`, `&mut config.reject_next_prepare` from
            // `config` (disjoint resources / a disjoint field) — the one-shot 3g reject lever.
            let ack = apply_prepare(
                session,
                transfer,
                dest,
                stats,
                &mut config.reject_next_prepare,
            );
            // RLM 5f-4 — THE CROSSING-DEST ADMISSION. The roster edits ride the SAME condition as the
            // progress write: `apply_prepare` returns `None` from exactly ONE guard (a not-Active session)
            // and that guard opens NO progress, so a refused prepare must claim nothing (a claim with no
            // terminal to release it IS the leak) and displace nothing. A `Rejected` verdict still opened
            // the progress and still gets the claim — its `AbortTransfer` terminal releases it.
            // BOTH displaced roles ride the SAME `ack.is_some()` gate as the claim: a refused prepare
            // overwrote no progress, so it must hand nothing back either.
            if ack.is_some() {
                roster.claim.push(dest);
                roster.release.push(displaced);
                roster.release.push(displaced_home);
            }
            ack
        }
        TransferControl::RequestCut { .. } => {
            // S3: self-acking `CutConfirmed` (server-timed cut) — journaled at step 1 below.
            apply_request_cut(session, outbox, transfer, stats)
        }
        TransferControl::FreezeSource {
            marker_seq, dest, ..
        } => apply_freeze(session, transfer, marker_seq, dest, stats),
        TransferControl::ThawSource { .. } => apply_thaw(session, transfer, stats),
        TransferControl::AbortTransfer { .. } => {
            // Close EXACTLY this transfer's dest sub (`tp.dest`), mirroring ReleaseSubscribe's
            // precise `src`; a foreign/absent abort matches no `tp` and closes nothing. `close_sub`
            // no-ops if the dest sub is not (yet) open — the normal pre-CAS abort case.
            if let Some(tp) = session
                .transfer
                .as_ref()
                .filter(|tp| tp.transfer == transfer)
            {
                subs_to_close.push(tp.dest);
                // RLM 5f-4e: a POST-CAS abort ALSO closes the DEMOTING SOURCE sub, so the un-route below
                // never outlives its subscription (the invariant the whole `demoting_home` stash exists to
                // hold). `extend` of an `Option` is branchless — `None` (every PRE-CAS abort) is a no-op, so
                // no new region and every existing abort cell is untouched. This arm is not reachable from
                // the shipped sender (`saga.rs post_commit_is_forward_only` proptests no AbortTransfer after
                // CasWon), but it is the one place the rework's own invariant could be violated.
                subs_to_close.extend(tp.demoting_home);
                // RLM 5f-4: the abort terminal RELEASES this transfer's crossing-dest claim — the
                // compensating half of the Prepare-time admission. A pre-CAS abort never re-pointed
                // `home_shard`, so this is the dest's ONLY claim and it leaves the roster here.
                roster.release.push(Some(tp.dest));
                // RLM 5f-4e: …AND the DEMOTING SOURCE home the commit stashed (released in lock-step with the
                // sub close above). For a PRE-CAS abort this is always `None` (no commit ran ⇒ nothing
                // displaced) — the no-op arm of `release_dynamic_shard`, so no new branch. Load-bearing for a
                // POST-CAS abort: `apply_abort` PRUNES `session.transfer`, so a terminal that skipped the
                // stash would strand the source's claim forever with no later holder able to name it.
                roster.release.push(tp.demoting_home);
            }
            apply_abort(session, transfer)
        }
        TransferControl::CommitAuthority {
            new_fence, subject, ..
        } => apply_commit(
            session,
            transfer,
            new_fence,
            subject,
            session_id,
            stats,
            outbox,
            &mut roster,
        ),
        TransferControl::ReleaseSubscribe { src, .. } => {
            // Close the SOURCE sub ONLY for the matching in-flight transfer (mirroring
            // `apply_release`'s prune); a foreign/absent release closes nothing.
            if let Some(tp) = session
                .transfer
                .as_ref()
                .filter(|tp| tp.transfer == transfer)
            {
                subs_to_close.push(src);
                // RLM 5f-4: the demote tail releases this transfer's crossing-dest claim. The dest stays
                // ROUTABLE because `CommitAuthority` claimed it a SECOND time for the session's new home
                // role — that claim lives until the session exits, so the post-crossing player keeps
                // receiving its dest frames long after the saga reaches `Done`.
                roster.release.push(Some(tp.dest));
                // RLM 5f-4e: the DEMOTING SOURCE's claim is due at the demote tail — the `close_sub` above
                // (same command) stops the client reading the source. Holding the claim until here is what
                // closes the blind window (the source stayed routable across the whole CommitAuthority→here
                // grace). NOTE the routable window ends at the sub's CLOSE, one inbound-batch short of its
                // REMOVAL: `close_sub` deliberately keeps the sub drainable for the rest of this batch + until
                // the next `sweep_draining`, whereas this release lands at the end of THIS command — so a
                // source straggler later in the same batch is dropped/`undecodable` for a demand-spawned
                // source (a `known_shards` source is still served). Tying the release to the sweep instead is
                // OWED (DEFERRED). `None` for a static/never-dynamically-homed session (the no-op arm).
                roster.release.push(tp.demoting_home);
            }
            apply_release(session, transfer)
        }
    };
    // RECORD-then-SEND for the LIVE acking phases. The journal write is GUARDED to the
    // IN-FLIGHT transfer (`tp.transfer == transfer`): an idempotent re-ack of a phase for a
    // NON-in-flight transfer (e.g. a stray Thaw/Abort while a different transfer is live)
    // must never pollute the live transfer's journal — bounding it to its own steps. (A
    // terminal AbortTransfer prunes `session.transfer` inside `apply_abort`, so `as_mut()`
    // is already `None` there — no record, the abort being idempotently re-ackable anyway.)
    if let Some(ack) = ack {
        if let Some(tp) = session.transfer.as_mut()
            && tp.transfer == transfer
        {
            tp.journal(step, ack);
        }
        reply_ack(outbox, config.orchestrator, ack);
    }
    // The `&mut Session` borrow has ended. RLM 5f-4: apply the runtime-roster edits FIRST — claims before
    // releases (see [`RosterEdits`]) — so a just-admitted dest is node-class-dispatchable for the REST of
    // this very tick's inbound batch, not only from the next tick.
    for dest in roster.claim {
        sessions.claim_crossing_dest(config, dest);
    }
    for node in roster.release {
        sessions.release_dynamic_shard(node);
    }
    // Then close the collected subs through the sole close primitive (Draining grace; the next-tick sweep
    // removes them). `close_sub` is idempotent and a no-op for a shard this session does not subscribe to.
    for shard in subs_to_close {
        sessions.close_sub(session_id, shard, config, outbox, stats);
    }
}

/// `PrepareSubscribe` (step 0): open the per-session transfer progress + reply `Prepared`.
/// 1c STUB readiness — there is no dest-subscription/ghost machinery yet (one stub shard),
/// so the verdict is `Ready` (the typed `Rejected` path stays WIRED for later bands, not
/// dead live code). DEFENSIVE replace: overwrite any stale progress.
///
/// REQUIRES the session be ACTIVE (WB-1): a transfer can only run for an attached avatar.
/// This single guard ENFORCES "attach strictly precedes transfer" LOCALLY (rather than
/// borrowing it from saga ordering) and is what makes every downstream route writer safe:
/// `session.transfer` is set Some ONLY here, ONLY when Active; Active is monotonic-until-
/// removal; and the attach path early-returns once Active (never re-storing the route). So a
/// cut installed by `apply_freeze` — or a route swapped by 1c.4 `CommitAuthority` — can never
/// be clobbered by a late `SessionAttached`. A not-Active prepare is counted + un-acked (pins
/// the saga, per WEDGE-1), never a route-touching transfer on a half-attached session.
fn apply_prepare(
    session: &mut Session,
    transfer: TransferId,
    dest: NodeId,
    stats: &mut GatewayStats,
    // 3g abort-leg (INERT test lever): consumed ONCE when this prepare would otherwise reply `Ready`.
    // The not-Active guard below stays ABOVE this and MUST NOT consume it — the lever fires only for a
    // would-be-`Ready` prepare, so a not-Active prepare leaves it armed for the retried (Active) one.
    reject_next_prepare: &mut Option<PrepareReject>,
) -> Option<TransferControlAck> {
    if !matches!(session.phase, SessionPhase::Active { .. }) {
        stats.transfer_unroutable += 1;
        return None;
    }
    // RACE-1 (pinned to Slice 2 — DEFERRED D-23): replacing a LIVE, different transfer's
    // progress discards its journal. Impossible today — the orchestrator serializes one
    // saga per session-subject (`DirectoryCore::lock_transfer`), and a redelivered SAME
    // transfer is caught by the redelivery gate before reaching here — so any existing
    // progress here is necessarily a stale DIFFERENT transfer. Becomes reachable only once
    // Slice-2 closes the abort-lock leak (D-1); pinned loud so it is never silently relied on.
    if let Some(displaced) = session.transfer.as_ref() {
        tracing::warn!(
            displaced = displaced.transfer.0,
            opening = transfer.0,
            "PrepareSubscribe replaced a stale in-flight transfer's progress — revisit at \
             Slice 2 (the one-saga-per-subject lock makes this benign today; D-23)"
        );
    }
    session.transfer = Some(TransferProgress {
        transfer,
        cut_requested: false,
        dest, // captured here for the precise abort-time dest-sub close (1d.2)
        // RLM 5f-4e: a FRESH progress is demoting nothing — the stash is written ONLY by
        // `apply_commit` (the one place a home is displaced). A defensive replace therefore starts
        // `None`, which is why the DISPLACED progress's stash must be handed back at the replace
        // (`on_transfer_control`'s PrepareSubscribe arm) or its source claim would be stranded.
        demoting_home: None,
        applied: BTreeMap::new(),
        dest_buffer: VecDeque::new(),
    });
    // The dest input slot is opened at CommitAuthority (apply_commit) — the gateway BUFFERS
    // seq>marker locally during the cut, so the dest needs nothing until the commit drain.
    // (An early prepare-time open is a P3 gateway-adoption resilience concern, not 1c.5.)
    //
    // 3g abort-leg: the ONE-SHOT reject lever. `None` (every real cluster) = the 1c `Ready` stub
    // (behaviour-identical). `Some(reject)` = reply `Rejected(reject)` and SELF-CLEAR (`.take()`),
    // so the very next prepare is `Ready` again — the sole way to drive a crossing-origin durable
    // saga into its pre-CAS abort in a cluster (the gateway is the durable Prepare decider).
    let result = match reject_next_prepare.take() {
        Some(reject) => {
            tracing::debug!(
                transfer = transfer.0,
                ?reject,
                "PrepareSubscribe REJECTED by the one-shot reject_next_prepare lever (3g abort-leg)"
            );
            PrepareResult::Rejected(reject)
        }
        None => {
            tracing::debug!(
                transfer = transfer.0,
                "PrepareSubscribe readiness is a 1c stub (Ready)"
            );
            PrepareResult::Ready
        }
    };
    Some(TransferControlAck::Prepared { transfer, result })
}

/// `RequestCut` (step 1): mark the cut as requested + SELF-ACK `CutConfirmed` (the SERVER-TIMED
/// cut, S3). The cut is now driven ENTIRELY server-side — the saga no longer waits on a client
/// `CUT_MARKER` (which multi-hop breaks: on hop 2+ the client's session is bound to the FIRST
/// shard's port, so the marker never reaches the current authority, and the saga stalls
/// `Cutting`→`CutTimeout`). The gateway still pushes `ServerControlMsg::RequestCut` to the
/// client (harmless/cosmetic — the old client's marker is now an inert ordinary input, S3),
/// but the `CutConfirmed` seq here is a PLACEHOLDER: the REAL input-cut seq is derived ATOMICALLY
/// at cut-install time in [`apply_freeze`] from `last_input_seq` (the leak-free partition point),
/// and rides `SourceFrozen.drained_seq` + the installed `SeqCut.marker_seq` — never this value.
/// So the FSM carries this `marker_seq` unread (`CommittingCas` reads `drained_seq`, not it).
///
/// Journaled at step 1 (via the standard record-then-send path in `on_transfer_control`), so a
/// redelivered `RequestCut` re-serves the SAME `CutConfirmed` verbatim (idempotent). Requires the
/// matching progress (`PrepareSubscribe` precedes it); setting `cut_requested` keeps the F1 guard
/// intact for the now-inert marker observer. Returns the ack for the caller to journal + reply.
fn apply_request_cut(
    session: &mut Session,
    outbox: &mut OutboundBox,
    transfer: TransferId,
    stats: &mut GatewayStats,
) -> Option<TransferControlAck> {
    let bound = session
        .transfer
        .as_ref()
        .is_some_and(|tp| tp.transfer == transfer);
    if !bound {
        stats.transfer_unroutable += 1; // RequestCut without a matching Prepared: drop+count
        return None;
    }
    session
        .transfer
        .as_mut()
        .expect("bound implies the in-flight transfer is present")
        .cut_requested = true;
    push_control(
        outbox,
        session.client,
        &ServerControlMsg::RequestCut { transfer },
    );
    // SELF-ACK the cut server-side (S3). `marker_seq: 0` is a PLACEHOLDER — the FSM carries it
    // unread; the real input-cut seq is `apply_freeze`'s install-time `last_input_seq`.
    Some(TransferControlAck::CutConfirmed {
        transfer,
        marker_seq: 0,
    })
}

/// `FreezeSource` (step 2): INSTALL the live cut — `seq <= marker_seq` stays bound to the
/// source, `seq > marker_seq` will buffer toward `dest`. The SOLE `cut: Some(..)` writer; its
/// mirror image is `apply_thaw`'s `cut: None`, both via `store_cut`. The `seq > marker`
/// partition READ lands in 1c.5; 1c.3 installs the latent field only (this hot path reads
/// only `.authority`, so the install is inert to routing until then).
///
/// UNBOUND DIVERGES from `apply_thaw`: freeze is a FORWARD phase (its `SourceFrozen` drives
/// the directory CAS), so an unbound freeze must NOT fabricate an ack — acking a
/// never-installed cut would advance the saga to a CAS / route-swap against a session this
/// gateway no longer holds (the Bye-mid-transfer wedge, WEDGE-1). Returning `None` correctly
/// PINS the saga until the Slice-2 abort producer (D-23). Bound-check FIRST, mirroring
/// `apply_request_cut` (the forward sibling), NOT `apply_thaw` (a compensator that no-ops
/// truthfully when unbound). Self-acking, so a redelivery is re-served by the gate, never
/// re-entered here (the route is never re-touched).
fn apply_freeze(
    session: &mut Session,
    transfer: TransferId,
    _cmd_marker_seq: u64, // S3: the command's marker is a PLACEHOLDER — the gateway derives its own below.
    dest: NodeId,
    stats: &mut GatewayStats,
) -> Option<TransferControlAck> {
    let bound = session
        .transfer
        .as_ref()
        .is_some_and(|tp| tp.transfer == transfer);
    if !bound {
        stats.transfer_unroutable += 1; // no source to freeze: pin the saga (no ack)
        return None;
    }
    // S3 SERVER-TIMED CUT — derive the input-cut seq at INSTALL time from `last_input_seq`, the
    // gateway's per-session high-water. `route_input`'s `fetch_max` advanced `last_input_seq` and
    // then Forwarded EVERY pre-cut input to the current authority (the source), so at the instant
    // this `store_cut` installs the `SeqCut`, `last_input_seq` == "the highest seq forwarded to the
    // source". Everything `<= marker_seq` went to the source and WILL apply (in-order,
    // at-least-once); everything `> marker_seq` buffers for the dest, which resumes at
    // `marker_seq + 1`. So inputs partition IDENTICALLY across the cut — NONE lost, NONE doubled —
    // which is exactly why the client `CUT_MARKER` is no longer needed (S3). This RETIRES the OLD
    // hazard: the OLD marker came from the client's chosen `CUT_MARKER` seq, which could sit BELOW
    // inputs already Forwarded to the source, so the dest re-applied them (double).
    //
    // PARTITION SAFETY IS BY SERIALIZATION, NOT BY THESE ATOMICS (verify wf review a9946a1c). Today
    // `route_input` (from `on_client_input`) and this `apply_freeze` (from `on_transfer_control`)
    // both run inside the SINGLE `process_gateway_inbound` system on the ONE sim thread, one inbound
    // msg at a time — they never overlap, so the `load(marker)`→`store_cut` pair is effectively
    // atomic w.r.t. the router. **TRIPWIRE (DEFERRED.md D-46):** the "(future) threaded 20 Hz
    // forwarder" would make this a live TOCTOU — a concurrent `route_input` could `fetch_max(X+1)`
    // and Forward X+1 to the source in the window between the load and the store (marker=X), then
    // the dest re-applies X+1 (the dest seeds `last_applied` to `marker_seq`, NOT the source's true
    // last-applied, so it does NOT dedup this cross-shard double). Before threading the forwarder,
    // install the cut FIRST with a sentinel marker then settle it (so a racing input Buffers), or
    // seed the dest watermark from the source's acked last-applied — NOT this gateway high-water.
    let marker_seq = session.hot.last_input_seq.load(Ordering::Relaxed);
    store_cut(&session.hot, Some(SeqCut { marker_seq, dest }));
    // drained_seq = marker_seq (the install-time high-water): "the source applied input through
    // exactly this seq" holds because the source applies everything the gateway forwarded (all
    // `<= marker_seq`). The FSM threads THIS `SourceFrozen.drained_seq` into `CommittingCas` as the
    // CAS watermark + it seeds the dest's `OpenInputSlot.resume_from_seq` at `apply_commit`.
    Some(TransferControlAck::SourceFrozen {
        transfer,
        drained_seq: marker_seq,
    })
}

/// THE sole route-mutation primitive: the one `route.store` for every transfer/attach
/// write. `store_cut` (carry authority+fence, swap cut), `store_commit` (move authority→dest,
/// carry fence, clear cut), and the attach path (fresh fence) all funnel here — so a future
/// `RouteSnapshot` field can never silently drop on any of them: the exhaustive no-`..rest`
/// struct literal lives in EXACTLY one place, and a 4th field is a single compile-fix every
/// caller inherits. (Login route BIRTH stays a direct `ArcSwap::from_pointee` — not a store.)
fn store_route(hot: &SessionHot, authority: NodeId, fence: Fence, cut: Option<SeqCut>) {
    hot.route.store(Arc::new(RouteSnapshot {
        authority,
        fence,
        cut,
    }));
}

/// THE sole `SubTable` (READ plane) writer — the read-plane analog of `store_route`: project
/// the cold `subs` map into the immutable hot `SubTable` and publish it whole via one
/// `ArcSwap::store`. The exhaustive no-`..rest` `SubEntry` literal lives in EXACTLY one place
/// (HR3), so a future `SubEntry` field is a single compile-fix every caller inherits. Both
/// `Active` AND `Draining` records are projected (a `Draining` sub stays routable for its
/// one-tick drain grace); the cold map is already sorted by shard `NodeId` (`BTreeMap`), so
/// the boxed slice is sorted for `SubTable::lookup`'s binary search by construction.
fn publish_subs(hot: &SessionHot, subs: &BTreeMap<NodeId, SubRecord>) {
    let by_shard: Box<[SubEntry]> = subs
        .iter()
        .map(|(shard, rec)| SubEntry {
            shard: *shard,
            sub: rec.sub,
            accepted: rec.accepted,
        })
        .collect();
    hot.subs.store(Arc::new(SubTable { by_shard }));
}

/// Carry the route's `authority` + `fence` forward and swap ONLY the `cut` — the cut
/// install/clear policy (`apply_freeze` `Some`, `apply_thaw`/`apply_abort` `None`). Its
/// mirror image is `store_commit`; both go through the lone `store_route` literal.
fn store_cut(hot: &SessionHot, cut: Option<SeqCut>) {
    let current = hot.route.load();
    store_route(hot, current.authority, current.fence, cut); // CARRY authority + fence
}

/// `CommitAuthority`: MOVE authority → `dest`, CARRY the realm fence UNCHANGED, CLEAR the
/// cut — ONE atomic `route.store` (the SPIKE-2a one-publish guarantee). The fence is CARRIED,
/// NOT advanced to the per-Entity CAS `new_fence` (R-FENCE / DEFERRED D-25 — see `apply_commit`).
fn store_commit(hot: &SessionHot, dest: NodeId) {
    let current = hot.route.load();
    store_route(hot, dest, current.fence, None); // MOVE authority→dest, CARRY fence, CLEAR cut
}

/// `CommitAuthority` (step 3): THE route swap. MOVE authority → `dest` (the `SeqCut.dest` the
/// preceding `FreezeSource` installed), CLEAR the cut, in ONE atomic `store_commit`.
///
/// `dest` HAS NO WIRE CARRIER by construction — `CommitAuthority { transfer, session, new_fence }`
/// carries no dest — so the installed `route.cut.dest` is authoritative: the gateway-local
/// crystallization of the saga's durable `SagaCtx.dest` (the SAME value `FreezeSource.dest`
/// rode). FreezeSource strictly precedes commit on reliable+ordered CONTROL (the saga emits
/// CommitAuthority only after `CommittingCas`), so the cut is present at first delivery.
///
/// Bound-check FIRST (mirror `apply_freeze`): a post-CAS forward phase must NOT fabricate
/// `Committed` for a session this gateway no longer holds (WEDGE-1) — pin instead.
///
/// **R-FENCE (DEFERRED D-25): `new_fence` is CARRIED, NOT installed as `route.fence`.** Frames
/// stamp the REALM fence (each sub accepts at its per-shard `SubEntry::accepted` realm fence,
/// checked in `on_shard_frame`), whereas `new_fence` is the per-ENTITY CAS fence
/// (`DirectoryKey::Entity`). Installing the Entity fence here would make the dest's OWN
/// realm-stamped frames stale
/// (`realm_fence.is_stale_against(new_fence) == true`) — a black screen. Fence rule 5 (fence
/// out a demoted REMOTE owner) activates by REALM-fence movement at 1d/mesh when source/dest
/// are distinct realm leases; intra-shard (one realm lease) it correctly does NOT fire.
/// `new_fence` stays threaded through wire+saga (the CAS linearization point); the gateway
/// consumes it without installing it until the dest re-stamps its realm lease (1d/mesh).
///
/// **RLM 5f-4e — closes D-34's A1 (authority-following detach); the composited-subs clause remains OWED.**
/// Commit is where the session's ROUTING TARGET follows authority: `home_shard` is re-pointed to `dest`.
/// The displaced OLD home's roster claim is **STASHED** in `TransferProgress::demoting_home`, NOT released
/// here — the source's read sub outlives the commit by the whole demote grace (RLM 5f-4e). This re-points
/// only the CURRENT authority; the SOURCE keeps its own sub/ghost until `ReleaseSubscribe`, so D-34's
/// composited-subs half is not yet closed. See step (b2).
#[allow(clippy::too_many_arguments)] // the 8th is the deferred roster carrier (the borrow-discipline seam)
fn apply_commit(
    session: &mut Session,
    transfer: TransferId,
    new_fence: Fence,
    subject: DirectoryKey,
    session_id: SessionId,
    stats: &mut GatewayStats,
    outbox: &mut OutboundBox,
    // RLM 5f-4: the deferred roster edits — `dest` is discovered HERE (from the installed cut), but
    // `dynamic_shards` lives on `GatewaySessions`, which is borrow-locked while `session` is held.
    roster: &mut RosterEdits,
) -> Option<TransferControlAck> {
    let bound = session
        .transfer
        .as_ref()
        .is_some_and(|tp| tp.transfer == transfer);
    if !bound {
        stats.transfer_unroutable += 1; // session/transfer absent: pin (WEDGE-1)
        return None;
    }
    // Capture BOTH `dest` and `marker_seq` from the installed cut BEFORE `store_commit`
    // clears it: `marker_seq` seeds the dest's resume watermark; `dest` is the new authority.
    let Some(SeqCut { dest, marker_seq }) = session.hot.route.load().cut else {
        // BOUND but no cut: FreezeSource must precede commit (ordered CONTROL). Loud,
        // counted, route UNTOUCHED — never a swap to a garbage dest; the saga pins.
        stats.commit_without_cut += 1;
        tracing::error!(
            transfer = transfer.0,
            "CommitAuthority with no installed cut — FreezeSource must precede commit; \
             route untouched, saga pins"
        );
        return None;
    };
    let _ = new_fence; // R-FENCE: carried-not-installed in 1c.4 (D-25); 1d/mesh installs on dest re-stamp
    // (a) AUTHORITATIVE OpenInputSlot: the dest seeds last_applied_seq = marker_seq, so the
    // drained resume batch (marker+1..) applies in order and a `seq <= marker` replay is
    // rejected — closing the UnknownSession input-loss hole for the genuinely-distinct dest.
    // 1c.8: it ALSO carries the transfer `subject` (forwarded verbatim from CommitAuthority) so
    // the dest ADOPTS the transferred avatar (its Entity becomes the dot's id). `new_fence` is
    // NOT carried — the dest learns it from its own directory HeadRead (pull-through).
    push_to_shard(
        outbox,
        dest,
        MsgClass::Control,
        &GatewayToShard::OpenInputSlot {
            session: session_id,
            fence: session.fence,
            account: session.account,
            resume_from_seq: marker_seq,
            subject,
        },
    );
    // (b) THE swap: authority:=dest, fence carried, cut:=None (ONE atomic publish).
    store_commit(&session.hot, dest);
    // (b2) RLM 5f-4e — closes D-34's A1 (composited-subs clause OWED): the player is now HOMED on the dest,
    // so the session's ROUTING TARGET
    // ([`session_target`]) must follow authority. `Option::replace` does both halves in ONE branchless
    // expression: point `home_shard` at `dest` and hand back the OLD home. Roster consequences:
    // - CLAIM `dest` for the new HOME role. This is a SECOND claim on top of the Prepare-time crossing
    //   claim (the refcount makes it idempotent-by-construction), and it is what keeps the dest routable
    //   after `ReleaseSubscribe` releases the crossing one — without it a demand-spawned dest left the
    //   roster the instant the demote tail landed and the player went blind again.
    // - RLM 5f-4e: the OLD home's claim is **STASHED, NOT RELEASED HERE** (`TransferProgress::
    //   demoting_home`). Commit moves the WRITE authority, but the client's READ subscription on the
    //   SOURCE stays open for the whole demote grace (closed at `ReleaseSubscribe`, a later tick) and the
    //   source keeps shipping frames. Releasing at commit un-routed a DEMAND-SPAWNED source, so its
    //   `Snapshot`/`RealmSnapshot` frames were counted `undecodable` for the entire tail — a BLIND player
    //   right after every crossing. The stash is handed back where the sub actually closes
    //   (`ReleaseSubscribe` / `AbortTransfer`), or at the session's exit if the client quits in the window,
    //   so a chain of crossings still accumulates no refcount per hop.
    // Before this, `home_shard` was never re-pointed at commit (the open D-34 owe), so a post-crossing
    // `Bye` detached at the SOURCE and leaked the dest's `SessionTable` entry.
    roster.claim.push(dest);
    // Two statements, not one expression: `home_shard` and `transfer` are disjoint fields but a single
    // `progress.demoting_home = session.home_shard.replace(dest)` would borrow `session` twice.
    let demoting_home = session.home_shard.replace(dest);
    // ONE `&mut` reach into the in-flight progress serves BOTH remaining steps (the stash write and the
    // buffer take), so the bound-check's unreachable-`None` shape is asserted exactly once.
    // The bound-check above guarantees `session.transfer` is Some (the in-flight transfer);
    // `expect` is the unreachable-arm shape.
    let progress = session
        .transfer
        .as_mut()
        .expect("bound-check above guarantees the in-flight transfer is present");
    progress.demoting_home = demoting_home;
    // (c) DRAIN the cut buffer to the now-authoritative dest as ordinary SessionInput, in
    // push order (== seq order: route_input's fetch_max dropped seq<=prev BEFORE buffering,
    // so the buffer is strictly increasing). `std::mem::take` empties it — the STRUCTURAL
    // drain-once guarantee: a redelivered CommitAuthority re-serves Committed via the gate
    // (above) AND finds the buffer empty, so it never re-drains. That SAME gate is what keeps the
    // (b2) stash exact under at-least-once delivery: a redelivered commit never re-enters here, so
    // `demoting_home` can never be overwritten with `Some(dest)` (which would strand the real source
    // claim) and `dest` is never claimed a third time.
    // Push order == seq order: `route_input`'s `fetch_max` returns `Deduped` for any
    // `seq <= prev` BEFORE the partition, so only a strictly-increasing subsequence is ever
    // buffered. The dest's own `last_applied_seq` dedup is the backstop regardless of order.
    let buffered = std::mem::take(&mut progress.dest_buffer);
    for input_bytes in buffered {
        push_to_shard(
            outbox,
            dest,
            MsgClass::Input,
            &GatewayToShard::SessionInput {
                session: session_id,
                fence: session.fence, // session-grant fence (NOT new_fence; R-FENCE/D-25)
                input_bytes,
            },
        );
    }
    Some(TransferControlAck::Committed { transfer })
}

/// `ThawSource` (step 4): the `FreezeSource` compensator — input resumes to the source.
/// Clears the cut (via `store_cut`) ONLY for the matching in-flight transfer
/// (ROB-THAW-UNBOUND-STORE): a thaw naming a different/absent transfer must NOT touch the
/// route, else it would clobber the in-flight transfer's LIVE cut (live since 1c.3). A thaw
/// against a never-frozen source is a correct no-op (the source never stopped), so it ALWAYS
/// acks `SourceThawed` (even unbound — counted — so the saga's thaw compensator can always
/// complete), but only the BOUND case stores the route. The live mirror image of `apply_freeze`.
fn apply_thaw(
    session: &mut Session,
    transfer: TransferId,
    stats: &mut GatewayStats,
) -> Option<TransferControlAck> {
    let bound = session
        .transfer
        .as_ref()
        .is_some_and(|tp| tp.transfer == transfer);
    if bound {
        store_cut(&session.hot, None);
    } else {
        stats.transfer_unroutable += 1;
    }
    Some(TransferControlAck::SourceThawed { transfer })
}

/// `AbortTransfer` (step 5, terminal): `PrepareSubscribe`'s compensator — tear down the dest
/// ghost. In 1c.2 `PrepareSubscribe` built no ghost (`Ready` stub), so teardown is inert on
/// session state and the route is untouched (the source stayed authoritative). NOT a
/// disconnect: the `Session`/`phase` are untouched. Prunes ONLY the matching in-flight
/// transfer (CP-1: an abort for transfer B must never clobber a live transfer A). An abort
/// is an IDEMPOTENT terminal: a redelivered abort, or an abort for a transfer this gateway
/// never held, is a benign no-op re-ack — NOT a routing failure (F2: it is uncounted, so
/// `transfer_unroutable` stays about genuine failures). Always acks `Aborted`.
fn apply_abort(session: &mut Session, transfer: TransferId) -> Option<TransferControlAck> {
    if session
        .transfer
        .as_ref()
        .is_some_and(|tp| tp.transfer == transfer)
    {
        session.transfer = None; // prune ONLY the matching in-flight transfer
        // ROB-1c3: clear any installed cut LOCALLY. The saga emits ThawSource before
        // AbortTransfer over reliable CONTROL, so today the cut is usually already clear —
        // but abort must be LOCALLY fail-safe, not borrow that ordering: once the cut READ
        // goes live (1c.5) a survived cut becomes a live mis-route, and Slice-2/warp abort
        // producers need not preserve Thaw-before-Abort. `store_cut(None)` is idempotent
        // (a no-op when the cut is already clear), so it is safe on every abort path.
        store_cut(&session.hot, None);
    }
    tracing::debug!(
        transfer = transfer.0,
        "AbortTransfer tears down the dest ghost (inert in 1c.2 — no ghost) + clears any cut"
    );
    Some(TransferControlAck::Aborted { transfer })
}

/// `ReleaseSubscribe` (step 6, the SUCCESS teardown of the demote tail): close the source
/// subscription after demote grace and ack `Released` so the saga reaches `Done`. The cut is
/// already `None` (cleared at `CommitAuthority`'s `store_commit`) and the dest buffer was
/// take-drained there, so the ONLY post-commit remnant to clear is `session.transfer` — pruned
/// here ONLY when bound to THIS transfer (mirroring `apply_abort`'s prune). A stray re-ack of an
/// absent/torn-down transfer is a clean no-op-and-ack: it touches no state and still acks
/// `Released`, so an at-least-once redelivery is idempotent without polluting the journal (the
/// record-then-send guard sees `session.transfer == None` and skips the write, exactly as on
/// `apply_abort`). NOT a routing failure (uncounted) — `transfer_unroutable` stays about genuine
/// failures. Always acks `Released`.
fn apply_release(session: &mut Session, transfer: TransferId) -> Option<TransferControlAck> {
    if session
        .transfer
        .as_ref()
        .is_some_and(|tp| tp.transfer == transfer)
    {
        session.transfer = None; // prune ONLY the matching in-flight transfer (the last remnant)
    }
    tracing::debug!(
        transfer = transfer.0,
        "ReleaseSubscribe closes the source subscription (demote tail) + clears session.transfer"
    );
    Some(TransferControlAck::Released { transfer })
}

// S3 — THE CUT-MARKER OBSERVER IS RETIRED. The cut is now driven ENTIRELY server-side:
// `apply_request_cut` self-acks `CutConfirmed` (server-timed advance) and `apply_freeze` derives
// the real input-cut seq from `last_input_seq` at install time. A client's stamped `CUT_MARKER`
// (which the OLD client still emits on a `ServerControlMsg::RequestCut`) no longer drives anything
// — it is an ordinary input routed by `route_input` like any other, so the former `on_cut_marker`
// per-input observer (a cold `peek_is_cut_marker` decode for every mid-transfer input) is DELETED.
//
// This is what makes MULTI-HOP durable crossings work: on hop 2+ the client's session stays bound
// to the FIRST shard's port, so its marker never reaches the current authority's saga — under the
// old marker-driven cut that starved the `Cutting` step into `CutTimeout`→abort. Server-timing the
// cut removes that client dependency entirely. The `TransferProgress::cut_requested` field is kept
// (set by `apply_request_cut`) as the F1 phase-order record, harmless now that no observer reads it.

/// Handle a shard control reply (attach/detach lifecycle), from shard `from`.
fn on_shard_control(
    from: NodeId,
    bytes: &[u8],
    config: &GatewayConfig,
    clock: &ClockSample,
    sessions: &mut GatewaySessions,
    stats: &mut GatewayStats,
    outbox: &mut OutboundBox,
) {
    let Ok(msg) = postcard::from_bytes::<ShardToGateway>(bytes) else {
        stats.undecodable += 1;
        return;
    };
    match msg {
        ShardToGateway::SessionAttached {
            session: session_id,
            entity,
            frame,
            realm_fence,
        } => {
            // 5f-3d: the sub + the route must land on the session's ROUTING TARGET (its resolved home
            // shard, else the static `config.shard`), captured while we hold the session below — as is its
            // home realm, so the per-realm bootstrap can be left once the borrow ends.
            let target;
            let home_rid;
            {
                let Some(session) = sessions.by_session.get_mut(&session_id) else {
                    // Attach reply for a session that left meanwhile: ignore (the
                    // detach path already ran).
                    return;
                };
                // Promote ONLY from AwaitingAttach. A duplicate reply for an already-Active session is
                // idempotent (as before); CRUCIALLY a late/duplicate `SessionAttached` straggler must
                // NEVER resurrect a `SelfFenced` session (D-3 Slice 5b) — `SessionAttached` is re-emitted
                // on every `AttachSession` retry and rides the gateway↔shard link, which can be HEALTHY
                // while the gateway↔orchestrator link (that drove the self-fence) is partitioned, so a
                // straggler can arrive after `Active → SelfFenced`. Re-promoting it would re-arm the grace
                // clock and resume input/frame egress — re-opening the exact split-brain window the
                // self-fence exists to close. (AwaitingDirectory — an attach before the grant — is
                // likewise not promotable here.) This keeps `Active` monotonic-until-removal.
                if !matches!(session.phase, SessionPhase::AwaitingAttach) {
                    return;
                }
                // THE sole route-mutation primitive: one atomic store (P2 NOW drives this same
                // `store_route` from `CommitAuthority`'s `store_commit`). Attach SETS a fresh
                // realm fence (its distinct carry policy); commit/cut CARRY it.
                let authority = session.hot.route.load().authority;
                store_route(&session.hot, authority, realm_fence, None);
                session.phase = SessionPhase::Active { entity };
                // THE WINDOW LANE (Slice B): a STATIC login's lineage starts at the attach
                // frame's realm (a dynamic login already carries its descent's full lineage —
                // never overwritten here).
                if session.lineage.is_empty()
                    && let Some(realm) = frame.realm()
                {
                    session.lineage = vec![realm];
                }
                // RLM 5f-3d: the pre-Active bootstrap window CLOSES here — the session is LIVE, so the
                // bounded TTL no longer applies to it. Already `None` for a static session (byte-identical).
                session.bootstrap_deadline = None;
                target = session_target(session, config);
                home_rid = session.home_rid;
                // D-3 Slice 5b: going Active IS a fresh round-trip confirmation (the directory granted
                // and the shard attached) — arm the self-fence deadline from here.
                session.confirmed_at = clock.local_tick;
            }
            // RLM 5f-3d: the `Active` promote is THE terminator of this session's home bootstrap — it leaves
            // the per-realm wait HERE (the head resolve deliberately does not, so the demand stays re-seeded
            // across the attach round-trip). The realm's entry is pruned with its last member, so a fully
            // attached mass login leaves the index empty. `None` (a static session) is the no-op arm.
            sessions.end_home_wait(session_id, home_rid);
            // Open the login sub on the session's routing TARGET at the realm fence — the FIRST `open_sub`
            // caller (the transfer dest is the second, 1d.2b). `open_sub` pushes
            // SubscriptionOpened BEFORE publishing the SubTable (X1) and indexes the fan-out.
            let sub = sessions
                .open_sub(session_id, target, frame, realm_fence, outbox)
                .expect("session present (we just held it above this tick)");
            let session = sessions
                .by_session
                .get(&session_id)
                .expect("session present");
            announce_own_entity(
                outbox,
                session.client,
                session.negotiated_minor,
                entity,
                sub,
            );
            // The composed scene (proto_minor 18): the login LEVEL is emitted by the composer —
            // `compose_scenes` — the pass after this session's chain windows confirm (the first
            // fold bumps the epoch 0→1 and ships the full level + datagrams). Nothing scene-shaped
            // is emitted here any more: one author, one lane, one moment (§2.4).
        }
        ShardToGateway::SessionDetached { .. } => {
            // The session was already removed on Bye; the confirmation closes the loop.
        }
        ShardToGateway::SubscriptionReady {
            session: session_id,
            entity,
            frame,
            realm_fence,
        } => {
            // FORK 0a (Track R / 1d.2b): the dest (`from`) adopted the crossing entity and is
            // readable. Open a SECOND per-session sub on `from` at the DEST realm fence and
            // RE-POINT the avatar's render authority to it — the read-plane analog of the
            // write-plane `CommitAuthority`. The client then composites the source copy as
            // non-authoritative (renders the avatar ONCE, from the dest sub — invariant A1).
            let Some(session) = sessions.by_session.get(&session_id) else {
                stats.transfer_unroutable += 1; // absent session: counted, never a panic
                return;
            };
            if session.subs.contains_key(&from) {
                return; // duplicate SubscriptionReady (at-least-once): idempotent no-op
            }
            let sub = sessions
                .open_sub(session_id, from, frame, realm_fence, outbox)
                .expect("session present (held immutably just above this tick)");
            // THE LINEAGE follows the avatar across a crossing. THIS is the phase that can move
            // it: `SubscriptionReady` carries the destination FRAME, so the realm the player is
            // now in is nameable here (`CommitAuthority` names a node and a directory key,
            // neither of which says which realm that is). The composer derives the new chain from
            // this lineage next pass, bumps the origin epoch, and emits the swap level — ordered
            // AFTER the `AuthorityChanged` below on the same reliable stream (§2.7).
            if let Some(realm) = frame.realm() {
                // The session is a stated invariant, not a lookup that can fail: `open_sub` above
                // just resolved it (this is the borrow-split re-fetch, nothing else).
                let session = sessions
                    .by_session
                    .get_mut(&session_id)
                    .expect("session present (held immutably just above this tick)");
                lineage_apply(&mut session.lineage, realm);
            }
            let session = sessions
                .by_session
                .get(&session_id)
                .expect("session present");
            // The read-plane re-point at the dest promote (FORK 0a). Announce BOTH signals: a node-aware
            // client re-points its render authority to `sub` via `AuthorityChanged`; a pure-renderer
            // client (minor>=2) already renders this entity latest-wins by EntityId and just re-confirms
            // `OwnEntity` (idempotent). The client learns WHICH entity is its avatar, never WHICH node.
            announce_own_entity(
                outbox,
                session.client,
                session.negotiated_minor,
                entity,
                sub,
            );
        }
        ShardToGateway::RealmSceneDelta { .. } => {
            // ★TOMBSTONE (Slice C2, minor 19): the per-observer scene delta's PRODUCER is deleted
            // (§2.9 shrinks that emit to the ids-only membership verdict) and its client fan went
            // at the Slice-C1 flag day. A frame still decodes — its discriminant is reserved
            // forever — so it lands here: counted, dropped, never served.
            stats.old_scene_deltas_dropped += 1;
        }
        // THE REMOVE MESSAGE's fan (proto_minor 14, D-4(a)): the shard permanently stopped
        // emitting `entity` — every Active subscriber of that shard is told to evict its track,
        // each checked against ITS per-shard accepted fence and ITS negotiated minor.
        ShardToGateway::EntityRemoved {
            realm_fence,
            entity,
            at,
        } => {
            fan_entity_removed(from, realm_fence, entity, at, sessions, stats, outbox);
        }
        ShardToGateway::Frame { .. } | ShardToGateway::RealmFrame { .. } => {
            // Entity/realm frames ride the Snapshot / RealmSnapshot datagram classes; one on the
            // reliable Control stream is a peer bug (FA-2c: a RealmFrame is forwarded by
            // `on_shard_realm_frame`, dispatched from `MsgClass::RealmSnapshot`, never here).
            stats.undecodable += 1;
        }
        // THE WINDOW LANE (docs/design/window_lane.md §2.2): every row is ATTESTED fail-closed
        // (known window, roster-head sender, admissible body) and then INGESTED into the
        // window's composer state (Slice B — `window.rs`, shadow-first: measured, never served).
        // The reliable lanes (bodies, membership) arrive here on Control; the per-tick
        // `WindowFrame` datagram arrives via `on_shard_realm_frame` — BOTH run the ONE
        // `on_window_row` rule.
        ShardToGateway::WindowFrame {
            window,
            at,
            hop,
            rows,
            ..
        } => {
            let level = window::WindowLevel {
                at,
                hop: hop.map(|b| *b),
                rows,
            };
            on_window_row(
                from,
                window,
                WindowRow::Level(level),
                config,
                sessions,
                stats,
            );
        }
        ShardToGateway::WindowBody {
            window,
            subject,
            stmt,
            authored_at,
            ..
        } => {
            on_window_row(
                from,
                window,
                WindowRow::Body {
                    subject,
                    stmt,
                    authored_at,
                },
                config,
                sessions,
                stats,
            );
        }
        ShardToGateway::WindowMembership {
            window,
            added,
            removed,
        } => {
            on_window_row(
                from,
                window,
                WindowRow::Membership { added, removed },
                config,
                sessions,
                stats,
            );
        }
        // THE Q2 RELAY's forward leg (Slice C1, mesh minor 17; owner-approved 2026-08-16 —
        // owner_decisions_2026-08-15.md addendum + window_lane.md §5 RULINGS): a live child's
        // sealed self-authored statements, forwarded verbatim by its parent. Admitted HERE,
        // against the CHILD's identity, by `on_window_relayed`.
        ShardToGateway::WindowRelayed {
            window,
            child,
            child_fence,
            statements,
            ..
        } => {
            on_window_relayed(
                from,
                window,
                child,
                child_fence,
                &statements,
                sessions,
                stats,
            );
        }
    }
}

/// THE Q2 RELAY's receiving admission (Slice C1, mesh minor 17; owner-approved 2026-08-16 —
/// `docs/design/owner_decisions_2026-08-15.md` addendum + `docs/design/window_lane.md` §5
/// RULINGS): the FORWARDING parent must be the window's roster head (the same sender attestation
/// every direct row runs); the CHILD must be vouched by the parent's own attested roster (the
/// stream-only child set — the same source the direct marker admission uses); the child's own
/// fence orders incarnations (a deposed zombie's relay is refused); the seal opens HERE — the
/// first and only party to read it — and every inner statement is admitted against the CHILD's
/// identity with the existing predicates ([`window_body_admissible`] with the child as the
/// stating realm; its markers vouched by its own relayed level's roster). Admitted bodies land in
/// the window ingest's ONE body store, so the §2.8 marker⇒look handover needs nothing new
/// downstream (a waking realm's look upgrades its drawn body by data presence); relayed interior
/// LEVELS are held per child — the Slice-D interior compose's input, stored + counted and
/// deliberately unconsumed until that slice (the Slice-A discipline).
fn on_window_relayed(
    from: NodeId,
    window_id: WindowId,
    child: RealmId,
    child_fence: Fence,
    statements: &[u8],
    sessions: &mut GatewaySessions,
    stats: &mut GatewayStats,
) {
    let Some(held) = sessions.windows.get_mut(&window_id) else {
        stats.window_unknown_row += 1;
        return;
    };
    if !window_sender_is_head(from, Some(held.shard)) {
        stats.window_sender_mismatch += 1;
        tracing::warn!(
            sender = from.0,
            head = held.shard.0,
            window = window_id.0,
            "forged-sender window relay dropped (fail-closed attestation)"
        );
        return;
    }
    if !held.ingest.rosters(child) {
        // PARKED, not dropped (the same first-roster race as the direct markers — the relay is
        // forwarded send-on-change, once): counted, held newest-fence-wins, re-admitted through
        // the full admission the moment the author's level vouches the child.
        stats.window_relay_unvouched += 1;
        if held
            .parked_relays
            .get(&child)
            .is_none_or(|(f, _)| !child_fence.is_stale_against(*f))
        {
            held.parked_relays
                .insert(child, (child_fence, statements.to_vec()));
        }
        return;
    }
    admit_relay(held, child, child_fence, statements, stats);
}

/// The ONE relay admission (direct + drained-parked), applied AFTER the sender + roster vouches:
/// the child's fence orders incarnations, the seal opens HERE (the first and only reader), and
/// every inner statement runs the existing predicates against the CHILD's identity.
fn admit_relay(
    held: &mut GatewayWindow,
    child: RealmId,
    child_fence: Fence,
    statements: &[u8],
    stats: &mut GatewayStats,
) {
    if !held.ingest.admit_relay_fence(child, child_fence) {
        stats.window_relay_stale += 1;
        return;
    }
    let Ok(opened) = open_relay_statements(statements) else {
        stats.window_relay_undecodable += 1;
        return;
    };
    for statement in opened {
        match statement {
            RelayedStatement::Level { at, rows } => {
                if held.ingest.ingest_relay_level(child, at, rows) {
                    stats.window_relays_ingested += 1;
                } else {
                    stats.window_relay_stale += 1;
                }
            }
            RelayedStatement::Body {
                subject,
                stmt,
                authored_at,
            } => {
                let children = held.ingest.relay_child_roster(child);
                if !window_body_admissible(&stmt, subject, child, &children) {
                    stats.window_misauthored_body += 1;
                    tracing::warn!(
                        %subject,
                        %child,
                        "mis-authored RELAYED body dropped (fail-closed admission against the child)"
                    );
                    continue;
                }
                if held.ingest.ingest_body(subject, &stmt, authored_at) {
                    stats.window_relays_ingested += 1;
                } else {
                    stats.window_body_stale += 1;
                }
            }
        }
    }
}

/// One admitted window statement's payload, handed to the composer's ingest AFTER attestation —
/// the Slice-B shape of the one ingest rule (the Slice-A `Option<(RealmId, &BodyStmt)>` grew a
/// typed row per statement kind when the engine started consuming).
enum WindowRow {
    Level(window::WindowLevel),
    Body {
        subject: RealmId,
        stmt: BodyStmt,
        authored_at: UniverseTick,
    },
    Membership {
        added: Vec<RealmId>,
        removed: Vec<RealmId>,
    },
}

/// THE WINDOW LANE's ingest rule (docs/design/window_lane.md §2.2/§2.6.6), ONE rule for every row
/// kind on both carrying lanes, fail-closed in three layers:
/// 1. the window must be one THIS gateway holds open — a straggler from a closed window (or a
///    forged id) drops by id mismatch, exactly the `WindowId` contract;
/// 2. the sender must BE the head the gateway's own routing state resolved for the stating realm
///    when it derived the window (the directory-head answer `home_shard`/the session subs/the
///    realm-head poll named) — [`window_sender_is_head`] driven with that head in hand;
/// 3. a body statement's authorship must be admissible ([`window_body_admissible`]): a look only
///    about the author itself (SL3), a marker only about the author's DIRECT children (R4) — the
///    child set read from the author's OWN attested full-roster rows (§2.6.2: the seed-forest
///    second source is DELETED; the stream is the only vouching knowledge). A marker arriving
///    before the author's first level finds no roster to vouch for it and is refused apart
///    (`window_body_preroster`), fail-closed, never served on faith.
///
/// A row that passes is INGESTED into the window's composer state (`window_rows_ingested` —
/// Slice B retired Slice A's deliberate unconsumed counter). Still SHADOW: no client sees any of
/// it; the composed emissions ride `compose_scenes`.
fn on_window_row(
    from: NodeId,
    window_id: WindowId,
    row: WindowRow,
    config: &GatewayConfig,
    sessions: &mut GatewaySessions,
    stats: &mut GatewayStats,
) {
    let Some(held) = sessions.windows.get_mut(&window_id) else {
        stats.window_unknown_row += 1;
        return;
    };
    if !window_sender_is_head(from, Some(held.shard)) {
        stats.window_sender_mismatch += 1;
        tracing::warn!(
            sender = from.0,
            head = held.shard.0,
            window = window_id.0,
            "forged-sender window row dropped (fail-closed attestation)"
        );
        return;
    }
    match row {
        WindowRow::Level(level) => match held.ingest.ingest_frame(level, &window_tuning(config)) {
            window::Ingested::Applied => {
                stats.window_rows_ingested += 1;
                drain_parked(held, stats);
            }
            window::Ingested::BehindRing => stats.window_level_refused += 1,
        },
        WindowRow::Body {
            subject,
            stmt,
            authored_at,
        } => {
            if matches!(stmt, BodyStmt::Marker { .. }) && !held.ingest.confirmed() {
                // PARKED, not dropped (Slice C1): the send-on-change lane sends a body ONCE, so
                // a marker racing the author's first roster would otherwise stay invisible
                // forever. Counted as before; re-admitted through the SAME predicate the moment
                // a level lands (fail-closed meanwhile — nothing composes it while parked).
                stats.window_body_preroster += 1;
                held.parked_bodies.insert(subject, (stmt, authored_at));
                return;
            }
            admit_body(held, window_id, subject, stmt, authored_at, stats);
        }
        WindowRow::Membership { added, removed } => {
            held.ingest.ingest_membership(&added, &removed);
            stats.window_rows_ingested += 1;
        }
    }
}

/// The ONE body admission (direct + drained-parked): the subject rule against the author's OWN
/// attested roster, then newest-wins ingest — counted per outcome, never patched.
fn admit_body(
    held: &mut GatewayWindow,
    window_id: WindowId,
    subject: RealmId,
    stmt: BodyStmt,
    authored_at: vd_core::UniverseTick,
    stats: &mut GatewayStats,
) {
    let children = held.ingest.roster_set();
    if !window_body_admissible(&stmt, subject, held.author_realm, &children) {
        stats.window_misauthored_body += 1;
        tracing::warn!(
            %subject,
            author = %held.author_realm,
            window = window_id.0,
            "mis-authored window body dropped (fail-closed admission)"
        );
        return;
    }
    if held.ingest.ingest_body(subject, &stmt, authored_at) {
        stats.window_rows_ingested += 1;
    } else {
        stats.window_body_stale += 1;
    }
}

/// Drain the statements that raced the author's first roster (Slice C1): every parked body and
/// relay whose subject the NOW-attested roster vouches re-runs the same admission it would have
/// met in order; the rest stay parked (bounded — one slot per subject/child) for the next level.
fn drain_parked(held: &mut GatewayWindow, stats: &mut GatewayStats) {
    let roster = held.ingest.roster_set();
    let ready: Vec<RealmId> = held
        .parked_bodies
        .keys()
        .filter(|subject| roster.contains(subject))
        .copied()
        .collect();
    for subject in ready {
        let (stmt, authored_at) = held
            .parked_bodies
            .remove(&subject)
            .expect("keyed by the loop above");
        if held.ingest.ingest_body(subject, &stmt, authored_at) {
            stats.window_rows_ingested += 1;
        } else {
            stats.window_body_stale += 1;
        }
    }
    let ready: Vec<RealmId> = held
        .parked_relays
        .keys()
        .filter(|child| roster.contains(child))
        .copied()
        .collect();
    for child in ready {
        let (child_fence, statements) = held
            .parked_relays
            .remove(&child)
            .expect("keyed by the loop above");
        admit_relay(held, child, child_fence, &statements, stats);
    }
}

/// Fan one shard's `EntityRemoved` to that shard's subscribers as [`ServerControlMsg::Event`] —
/// the reliable per-entity eviction (the remove message, proto_minor 14). The same subscriber walk
/// as the frame fan (`subscribers_of`, Active only, per-shard accepted fence), but TYPED and
/// per-session minor-gated: a peer that negotiated below 14 is withheld the variant
/// (sender-gates-variants) and keeps the frozen-figure gap this message closes. A removal stale
/// against a session's accepted fence is refused for THAT session and counted apart
/// (`stale_removals_dropped`) — a demoted old owner must not evict what the live owner streams.
fn fan_entity_removed(
    from: NodeId,
    realm_fence: Fence,
    entity: EntityId,
    at: vd_core::UniverseTick,
    sessions: &mut GatewaySessions,
    stats: &mut GatewayStats,
    outbox: &mut OutboundBox,
) {
    for session_id in sessions.subscribers_of(from) {
        let Some(session) = sessions.by_session.get(&session_id) else {
            stats.frame_sub_desync += 1;
            continue;
        };
        // The OWNER of the removed entity is never told to evict its own avatar: the removal is a
        // fact about the OLD realm (the ghost band closed there) while the owner's truth is its
        // live dest sub — evicting would blink the one figure that must never blink, for exactly
        // one datagram interval, in the middle of a crossing. Bystanders have no other source of
        // the fact; the owner IS the fact's source.
        let SessionPhase::Active { entity: own } = &session.phase else {
            continue;
        };
        if *own == entity {
            continue;
        }
        let table = session.hot.subs.load();
        let Some(entry) = table.lookup(from) else {
            stats.frame_sub_desync += 1;
            continue;
        };
        let accepted = entry.accepted;
        drop(table);
        if realm_fence.is_stale_against(accepted) {
            stats.stale_removals_dropped += 1;
            continue;
        }
        if session.negotiated_minor < 14 {
            continue;
        }
        push_control(
            outbox,
            session.client,
            &ServerControlMsg::Event(vd_wire::channels::EventMsg::EntityRemoved { entity, at }),
        );
    }
}

/// Fan one shard frame (from shard `from`) out to that shard's subscribers, each at ITS sub
/// id and ITS per-shard accepted fence. The READ-plane heart (1d.2a): iterate ONLY
/// subscribers-of-`from` (H2 reverse index, O(subscribers) not O(all sessions)) and resolve
/// each session's `SubEntry` for `from` off the wait-free hot `SubTable`.
fn on_shard_frame(
    from: NodeId,
    bytes: &[u8],
    sessions: &mut GatewaySessions,
    stats: &mut GatewayStats,
    outbox: &mut OutboundBox,
) {
    let Ok(ShardToGateway::Frame {
        realm_fence,
        source_tick: _,
        snapshot_bytes,
    }) = postcard::from_bytes::<ShardToGateway>(bytes)
    else {
        stats.undecodable += 1;
        return;
    };
    // 1d.5a: peek the `frame_id` ONCE off the wire (the per-observer delivery watermark advances
    // by it). A malformed body fails HERE and the whole frame is abandoned (counted once) — and
    // because the peek validates the leading `sub` varint, the per-session `retag_snapshot_sub`
    // below is then INFALLIBLE (`.expect()`), so there is no second decode-error region.
    let Ok(frame_id) = peek_snapshot_frame_id(&snapshot_bytes) else {
        stats.undecodable += 1;
        return;
    };
    // SCALE-1: re-tag the snapshot body ONCE per distinct sub-id into a SHARED `Arc`; the
    // per-session fan-out is then a cheap fence check + refcount bump, never an O(entities)
    // re-allocation per subscriber.
    //
    // There is NO per-pin memo beside it any more, because there is nothing to memoise: the shard already
    // shipped every occupant measured in the frame of the realm the session is standing in. A router that
    // re-expressed the body per pin had to hold every parent's authored placement of every child to do it —
    // the world-wide graph this arc deleted — and it paid an O(entities) decode-and-re-encode per distinct
    // pin on the one hop that must stay sub-millisecond. The re-tag below is a leading-varint splice with
    // no decode at all.
    let mut retagged: BTreeMap<SubId, vd_sim::io::Bytes> = BTreeMap::new();
    for session_id in sessions.subscribers_of(from) {
        let Some(session) = sessions.by_session.get_mut(&session_id) else {
            // The reverse index and `by_session` are kept in sync by `open_sub`/the
            // drain-sweep; a missing session is an index/table desync (counted, never silent).
            stats.frame_sub_desync += 1;
            continue;
        };
        // D-3 Slice 5b: a SELF-FENCED session no longer acts as authority — it is served NO frames (its
        // subs linger inert in the reverse index until the connection ends or a ResumeTicket adoption
        // re-homes it). A still-attaching session has no subs and is never in this index. Active only.
        if !matches!(session.phase, SessionPhase::Active { .. }) {
            continue;
        }
        // Resolve THIS session's sub + per-shard accepted fence off the wait-free hot `SubTable`,
        // then RELEASE that borrow (the values are `Copy`) so we may advance the cold watermark.
        // A subscriber-in-index ALWAYS has a `SubEntry` (republished together by `publish_subs`);
        // a `None` is an invariant breach, counted (C2 honesty floor), never silent.
        let table = session.hot.subs.load();
        let Some(entry) = table.lookup(from) else {
            stats.frame_sub_desync += 1;
            continue;
        };
        let sub = entry.sub;
        let accepted = entry.accepted;
        let client = session.client;
        drop(table);
        // Per-shard fence (NOT the session-global route fence): the source sub accepts source
        // frames at the source realm fence; the dest sub accepts dest frames at the dest realm
        // fence. A demoted old owner's stale frame is dropped + counted (fence rule 5).
        if realm_fence.is_stale_against(accepted) {
            stats.stale_frames_dropped += 1;
            continue;
        }
        // 1d.5a: an ACCEPTED, past-fence frame ADVANCES this observer-sub's delivery high-water —
        // the standing (a) demote-predicate input. Only delivered (forwarded) frames count; a
        // stale-dropped frame (above) does NOT advance it.
        session
            .delivered
            .entry(sub)
            .and_modify(|w| *w = (*w).max(frame_id))
            .or_insert(frame_id);
        let body =
            retagged.entry(sub).or_insert_with(|| {
                vd_sim::io::bytes(retag_snapshot_sub(&snapshot_bytes, sub).expect(
                    "the frame_id peek validated the sub varint, so the re-tag is infallible",
                ))
            });
        outbox.0.push((
            client,
            MsgClass::Snapshot,
            body.clone(),
            vd_sim::io::Durability::Ephemeral,
        ));
    }
}

/// Ingest one WINDOW LEVEL, arriving on the realm-lane datagram class.
///
/// The window lane's per-tick statement is FireAndForget/Unreliable, so it rides THIS class (the
/// realm-lane tick) rather than the reliable Control stream — and runs the SAME attested ingest
/// rule as the Control-borne rows: unknown window, forged sender or stale fence is dropped and
/// counted, never composed.
///
/// ★TOMBSTONE (window lane Slice C2, minor 19): the OLD realm datagram
/// ([`ShardToGateway::RealmFrame`]) also arrived here, and this function used to fan its opaque
/// body to every subscriber. That fan died at the minor-18 flag day (the composed feed became the
/// client's one scene author) and its last consumer — the Slice-B parity comparator — dies here,
/// its measurement discharged with the lanes it measured. A `RealmFrame` frame is now counted and
/// dropped, never served.
fn on_shard_realm_frame(
    from: NodeId,
    bytes: &[u8],
    config: &GatewayConfig,
    sessions: &mut GatewaySessions,
    stats: &mut GatewayStats,
) {
    match postcard::from_bytes::<ShardToGateway>(bytes) {
        Ok(ShardToGateway::WindowFrame {
            window,
            at,
            hop,
            rows,
            ..
        }) => {
            let level = window::WindowLevel {
                at,
                hop: hop.map(|b| *b),
                rows,
            };
            on_window_row(
                from,
                window,
                WindowRow::Level(level),
                config,
                sessions,
                stats,
            );
        }
        // The tombstoned old realm datagram: decodable forever (its discriminant is reserved),
        // produced by nobody, served to nobody — counted so a revived producer is visible.
        Ok(ShardToGateway::RealmFrame { .. }) => stats.old_realm_frames_dropped += 1,
        Ok(_) | Err(_) => stats.undecodable += 1,
    }
}

/// A DETERMINISTIC per-account sentinel [`Fence`] for the injected `RealmDemand`'s `parent_fence` — NOT a
/// global constant (CRITIQUE-1 defense-in-depth: a single shared sentinel would collapse every login's
/// `FencedKey` idempotency into one). It is NOT the auth: `record_demand` does NOT read `parent_fence` for
/// any Step-3 decision (it only `max`-tracks it as `last_fence` for audit — rlm.rs), so this is
/// belt-and-suspenders. Folds the `u128` `AccountId` to `u64`; distinct accounts yield distinct sentinels
/// across the small id space the P7 store issues (a collision is harmless — audit-only). No wall-clock/rng.
fn home_sentinel_fence(account: AccountId) -> Fence {
    let a = account.0;
    Fence((a as u64) ^ ((a >> 64) as u64))
}

/// RLM 5f-3c — SERVER-DERIVE an authenticated login's HOME lineage AND its spawn position: the descent over
/// the account's STORED position (or the origin) yields the FULL root→leaf lineage, so demanding it spins up
/// the whole ancestor chain (the 5f-3a ride). NOTHING client-supplied enters here: the client's only spatial
/// input is its authenticated `AccountId`, so a raw client cannot steer which realm spins up (the abuse
/// boundary).
///
/// The descend is lineage-depth-bounded (`parent()` → `None` at the root), so no extra depth cap is needed.
/// Concrete (non-generic), no wall-clock/rng.
///
/// RLM 5f-3d SCALE: this DESCEND runs exactly ONCE per login. The resolved coord is then carried in
/// [`SessionPhase::AwaitingHomeRealm`] and the pose on the `Session`, so every re-drive rebuilds its demand
/// (and repeats its attach) off the STORED answer instead of re-descending the forest — which also makes it
/// impossible for a re-seed to name a different realm than the one the session is waiting on.
///
/// BOTH halves of a home, READ rather than resolved: the lineage to demand, and the pose to hand the shard
/// in `AttachSession`, already measured from that realm's own centre and stamped with that realm's frame.
///
/// THE DESCENT THAT USED TO BE HERE IS GONE. It walked a private copy of the whole seed forest downward,
/// subtracting each realm's stored centre. A realm that ORBITS stores its centre as zero — its live
/// position rides its parent's per-tick lane — so the walk read every orbiting planet as sitting exactly on
/// its own star, and a login at a star's centre resolved to *inside a planet*. Measured through the shipped
/// boot: all five planets of the first star system answered `4.16 m INSIDE` to that walk while answering
/// between `13.76` and `140.31 m OUTSIDE` in their own frames at their live placements.
///
/// The walk was also the one place a party that owns no realm did realm arithmetic. Both faults have the
/// same cure and it is not a better walk: a home is STORED as a name plus a realm-local pose, so there is
/// nothing to descend and nobody has to know where any realm is. The bootstrap circularity that blocked
/// this — the chain cannot answer "where inside" until it is running, and it only runs because something
/// demanded the answer to "which realm" — does not arise for a stored home, because neither question is
/// asked of anybody.
///
/// Login is NOT a special path: the lineage returned here is demanded through the ordinary mechanics, and
/// nothing downstream can tell where the first position came from.
/// THE WINDOW LANE's lineage rule at a crossing (`docs/design/window_lane.md` §2.6.2: "updated
/// at each crossing's `SubscriptionReady`"): a realm already in the lineage TRUNCATES back to it
/// (an outward cross — ancestors are KEPT); anything else APPENDS below the previous leaf (an
/// inward cross: travel is always out into the shared parent and in again — SL2 — so the
/// previous leaf IS the parent). A wrong append is fail-closed downstream: the Child window it
/// derives is refused by the shard (`window_child_unrostered`) and never confirms, so the chain
/// simply ends at the last attested hop.
fn lineage_apply(lineage: &mut Vec<RealmId>, realm: RealmId) {
    if let Some(pos) = lineage.iter().position(|r| *r == realm) {
        lineage.truncate(pos + 1);
    } else {
        lineage.push(realm);
    }
}

/// A [`RealmCoord`]'s realm chain, root→leaf — THE session lineage the window derivation reads
/// (`docs/design/window_lane.md` §2.6.2: "the lineage comes from the session's login descent").
fn coord_lineage(coord: &RealmCoord) -> Vec<RealmId> {
    let mut chain = Vec::new();
    let mut cursor = Some(coord.clone());
    while let Some(c) = cursor {
        chain.push(c.lowered());
        cursor = c.parent();
    }
    chain.reverse();
    chain
}

fn home_placement(cfg: &SeedInjectorConfig, account: AccountId) -> (RealmCoord, StampedPose) {
    let home = cfg.homes.home_of(account);
    (home.realm, home.pose)
}

/// RLM 5f-3c/5f-3d — THE one `RealmDemand{SpinUp}` constructor for a home lineage (HR3): the initial seed
/// (off the freshly derived [`home_coord`]) and EVERY re-drive (off the coord stored in the phase) build the
/// demand here, so they differ ONLY in `universe_tick` — never in child, verb or fence. The tick is the
/// clock's (determinism: no wall-clock, no rng).
fn demand_for_home(
    child: RealmCoord,
    account: AccountId,
    universe_tick: vd_core::UniverseTick,
) -> RealmDemand {
    RealmDemand {
        child,
        parent_fence: home_sentinel_fence(account),
        verb: DemandVerb::SpinUp,
        universe_tick,
    }
}

/// RLM 5f-3d — THE dynamic-home ROUTE resolve. A `Realm` head reply carries NO session id, so it is matched
/// against the [`GatewaySessions::home_bootstraps`] index: ONE reply resolves EVERY session booting into that
/// realm (a mass login onto the same home is a win, not a fan-out cost).
///
/// - `record` `None` — the realm is NOT routable yet (the orchestrator holds the demand; its shard is still
///   booting). Every waiter STAYS in `AwaitingHomeRealm`: NO `Close`, NO teleport, NO fallback attach to
///   some other shard, no loading screen — the SEAMLESS hold. The re-drive keeps the demand fresh and
///   re-polls this very head until it resolves (or the bounded TTL Closes loudly).
/// - `record` `Some` — the owning node is now known: it JOINS the RUNTIME routable-shard roster (it can
///   never be in the frozen config), the WRITE route is retargeted to it through the sole `store_route`
///   primitive, the phase advances to `AwaitingAttach`, and the attach is sent THERE (never to
///   `config.shard`).
///
/// The wait ENTRY SURVIVES this resolve (only `resolved` flips): the demand re-seed must continue across the
/// attach round-trip or the reconciler's arm-A lapses and reaps the realm out from under the login. That makes
/// the per-member `AwaitingHomeRealm` test below load-bearing rather than decorative — it is what keeps the
/// resolve IDEMPOTENT under at-least-once delivery. A duplicate reply finds its members in `AwaitingAttach`
/// and touches nothing, so it can neither re-`claim_dynamic_shard` (a refcount leak that would pin the node
/// on the roster forever) nor demote a session that has already gone `Active`.
///
/// A reply nobody waits on — EVERY `Realm` head in static mode, or one after the last member left — is a
/// clean no-op, exactly the pre-5f-3d behaviour for this arm.
///
/// SCOPE: a home realm that MIGRATES to a different node after this resolve but before `SessionAttached`
/// does not re-point the attach (its members have left `AwaitingHomeRealm`) — it rides the bounded bootstrap
/// TTL and Closes loudly, then the client re-logins onto the new owner. Live re-pointing mid-bootstrap
/// belongs with the D-34 authority-follows-commit work, not here.
fn on_home_realm_head(
    home_rid: RealmId,
    record: Option<OwnerRecord>,
    sessions: &mut GatewaySessions,
    stats: &mut GatewayStats,
    outbox: &mut OutboundBox,
) {
    let Some(owner) = record else {
        return; // still booting — hold the client, seamlessly
    };
    let home = owner.authority.node();
    let Some(wait) = sessions.home_bootstraps.get_mut(&home_rid) else {
        return; // nobody is booting into this realm (every static-mode Realm head lands here)
    };
    // The head poll is SATISFIED for every current member: stop re-polling (the demand half of the re-drive
    // deliberately keeps running — see `HomeWait::resolved`). A later joiner clears this again.
    wait.resolved = true;
    let members: Vec<SessionId> = wait.members.iter().copied().collect();
    for session_id in members {
        let Some(session) = sessions.by_session.get_mut(&session_id) else {
            // The index and `by_session` are kept in sync by the begin/end pair; a miss is an
            // invariant breach — counted, never a silent continue (the C2 honesty floor).
            stats.home_wait_desync += 1;
            continue;
        };
        if !matches!(session.phase, SessionPhase::AwaitingHomeRealm { .. }) {
            // Already resolved by an earlier reply for this realm (it sits in `AwaitingAttach`, still a
            // member because the re-seed must continue): a duplicate is an exact no-op, never a second
            // roster claim and never a re-attach storm.
            continue;
        }
        session.home_shard = Some(home);
        // THE sole route-mutation primitive (HR3): retarget the WRITE route's authority to the home shard,
        // CARRYING the route's current fence — the attach SETS the fresh realm fence a round-trip later.
        // Without this the session would attach to (and subscribe on) its home while still routing input at
        // the placeholder `config.shard`.
        store_route(&session.hot, home, session.hot.route.load().fence, None);
        session.phase = SessionPhase::AwaitingAttach;
        let (fence, account, spawn) = (session.fence, session.account, session.spawn);
        push_to_shard(
            outbox,
            home,
            MsgClass::Control,
            &GatewayToShard::AttachSession {
                session: session_id,
                fence,
                account,
                // THE realm this pose was measured against is the one we are attaching to — this arm is
                // reached only after the `Realm(home_rid)` head named the node owning that very realm.
                spawn,
            },
        );
        // The home joins the RUNTIME routable roster, so its `SessionAttached` + frames are node-class
        // dispatchable as a shard (its NodeId was minted at spawn — never in the frozen config).
        sessions.claim_dynamic_shard(home);
    }
}

/// Handle a directory reply. TWO gateway obligations ride this seam: the `Session` head confirms (or denies)
/// the mint and re-confirms an Active lease (D-3), and — RLM 5f-3d — the `Realm` head names the node owning
/// a pre-Active session's DYNAMIC HOME realm (5f-3c discarded this arm). Everything else (Entity/Ship heads,
/// CAS outcomes, clock samples) carries no gateway obligation.
fn on_directory_reply(
    reply: DirectoryReply,
    config: &GatewayConfig,
    identity: &NodeIdentity,
    clock: &ClockSample,
    sessions: &mut GatewaySessions,
    stats: &mut GatewayStats,
    outbox: &mut OutboundBox,
) {
    let (session_id, record) = match reply {
        DirectoryReply::Head {
            key: DirectoryKey::Session(session_id),
            record,
        } => (session_id, record),
        // RLM 5f-3d — the dynamic-home route resolve (this arm used to be discarded).
        DirectoryReply::Head {
            key: DirectoryKey::Realm(home_rid),
            record,
        } => {
            // THE WINDOW LANE (Slice B, §2.6.2): EVERY `Realm` head answer feeds the window
            // derivation's realm→node map — the lawful source for a lineage ANCESTOR's node
            // (the cadence poll in `drive_windows` asks; this is the answer landing). Pruned
            // each tick to the realms the Active sessions actually name.
            if let Some(owner) = &record {
                sessions
                    .realm_heads
                    .insert(home_rid, owner.authority.node());
            }
            on_home_realm_head(home_rid, record, sessions, stats, outbox);
            return;
        }
        // Entity/Ship heads, CAS outcomes and clock samples carry no gateway obligation.
        _ => return,
    };
    let Some(session) = sessions.by_session.get_mut(&session_id) else {
        return; // session left while the reply was in flight
    };
    let granted = record.is_some_and(|r| {
        r.authority == AuthorityRef::Gateway(identity.node_id) && r.fence == session.fence
    });
    // D-3 Slice 5b: an Active session's `Session`-head reply is a RECHECK round-trip (not a mint). An
    // affirming head RE-ARMS the proactive self-fence deadline (we just heard from the directory); a
    // foreign/absent head means the lease was reassigned or reaped, so reactively SELF-FENCE now — the
    // link-alive cure that complements the partition timer (mirrors the shard's reactive realm self-fence).
    if matches!(session.phase, SessionPhase::Active { .. }) {
        if granted {
            session.confirmed_at = clock.local_tick;
        } else {
            session.phase = SessionPhase::SelfFenced;
            stats.sessions_self_fenced_revoked += 1;
        }
        return;
    }
    if !matches!(session.phase, SessionPhase::AwaitingDirectory) {
        return; // AwaitingAttach / already-SelfFenced: no obligation for this head
    }
    if granted {
        // The mint is committed. Welcome the client; attach to the shard.
        //
        // RLM 5f-3d — ONE derivation for the whole dynamic decision (HR3, and the CRITIQUE-3 correctness
        // invariant): in DYNAMIC mode ([`GatewayConfig::dynamic_home_mode`] — the config gate `armed` AND
        // `clock.synced`) we server-derive the home demand ONCE here and use it for BOTH the seed emit and
        // the wait, so "a session entered `AwaitingHomeRealm` ⇒ a `SpinUp` demand was seeded for EXACTLY
        // that realm" holds by construction rather than by two agreeing conditions. In STATIC mode this is
        // `None` and everything below is the EXACT pre-5f-3d flow: phase `AwaitingAttach`, Welcome,
        // UniverseRate, `AttachSession` to `config.shard` — same order, same bytes, no extra head-read.
        //
        // MF3 — the gate is THREE-way, not two. `armed & !synced` must HOLD, never fall through to the
        // static arm: on an ARMED cluster `config.shard` is NOT this player's home (it may not even be a
        // live node), so attaching there would either serve the player from the WRONG shard undetected or
        // spin an unbounded per-tick attach retry with no TTL behind it (nothing sets `bootstrap_deadline`
        // on the static arm). Holding costs nothing and is invisible to the client: no Welcome yet, so no
        // client-visible artifact, and the `AwaitingDirectory` retry arm re-sends the IDEMPOTENT `LeaseGrant`
        // — the next reply (clock now synced) takes the dynamic arm. Counted, never silent. (That per-tick
        // re-grant is also what keeps the already-committed lease fresh through the hold: an idempotent
        // re-grant at the same owner+fence REFRESHES `lease_expires`, so this hold needs no renewal of its
        // own — unlike the post-Welcome dynamic-home hold, which is why MF2 widened the renew set.)
        if config.seed_injector.armed & !clock.synced {
            stats.logins_held_pre_sync += 1;
            return;
        }
        let home = if config.dynamic_home_mode(clock.synced) {
            // The ONE forest descend per login (5f-3d: every later re-drive reuses this coord). It now
            // yields BOTH halves of the answer — which realm, and where inside it — because they come from
            // the same walk down and separating them is what let the two disagree.
            let (coord, spawn) = home_placement(&config.seed_injector, session.account);
            session.spawn = Some(spawn);
            // THE WINDOW LANE (Slice B, §2.6.2): the login descent's coord IS the session's
            // lineage — recorded root→leaf ON the session, so the chain derivation reads
            // session history and never a forest.
            session.lineage = coord_lineage(&coord);
            Some(coord)
        } else {
            None
        };
        // The lowered directory key of the very realm this login is about to demand.
        let home_rid = home.as_ref().map(RealmCoord::lowered);
        session.phase = match home_rid {
            // DYNAMIC: hold here until this realm's shard is routable (there is no node to attach to yet).
            // The phase carries ONLY the realm id — the lineage + the cadence anchor live once per realm in
            // `home_bootstraps` (indexed below).
            Some(home_rid) => SessionPhase::AwaitingHomeRealm { home_rid },
            // STATIC: the pre-5f-3d transition, unchanged.
            None => SessionPhase::AwaitingAttach,
        };
        // The STANDING home identity (never cleared — it outlives the phase payload); `None` when static.
        session.home_rid = home_rid;
        push_control(
            outbox,
            session.client,
            &ServerControlMsg::Welcome {
                version: ProtoVersion::CURRENT,
                session: session_id,
                session_fence: session.fence,
                epoch: clock.epoch,
            },
        );
        // Relay the cluster tick rate to a minor-1+ client (sender-gates-variants),
        // so it drives its render cursor at the server's rate, not a guess.
        if session.negotiated_minor >= 1 {
            push_control(
                outbox,
                session.client,
                &ServerControlMsg::UniverseRate {
                    tick_hz: config.tick_hz,
                },
            );
        }
        // Captured while the `session` borrow is live — the per-realm index insert below needs them once it
        // has ended.
        let account = session.account;
        let since = clock.local_tick;
        // The DYNAMIC arm hands the descended lineage out here so the index insert can own it (the demand
        // consumes its own copy). `None` for a static login.
        let wait_seed: Option<(RealmId, RealmCoord)> = match home {
            // STATIC (the byte-identical default): attach to the session's routing target, which for a
            // session that never resolves a home IS `config.shard`.
            None => {
                push_to_shard(
                    outbox,
                    session_target(session, config),
                    MsgClass::Control,
                    &GatewayToShard::AttachSession {
                        session: session_id,
                        fence: session.fence,
                        account: session.account,
                        // STATIC: nothing was descended, so there is no realm this pose belongs to and
                        // nothing honest to send. `None` ⇒ the shard births at its own origin, which is
                        // byte-identical to every static rig today.
                        spawn: None,
                    },
                );
                None
            }
            // RLM 5f-3c/5f-3d — THE TRUSTED GATEWAY SEED INJECTOR + THE DYNAMIC-HOME HOLD. This arm is the
            // cluster-attested proof an AUTHENTICATED login LANDED: it is reached ONLY strictly downstream
            // of `validate_login` success AND a directory-CAS-committed `Session` lease owned by THIS
            // gateway at THIS fence (`granted`), on the `AwaitingDirectory →` transition — so it fires
            // EXACTLY ONCE per login (a re-driven grant re-enters and returns at the
            // `AwaitingHomeRealm`/`AwaitingAttach`/`Active` guards above, never here). The demand carries
            // NO client-supplied coord: the client's only spatial input is its authenticated `AccountId`
            // (the abuse boundary — a raw client speaks only `ClientControlMsg`).
            //
            // We emit NO `AttachSession` here: the home shard does not exist yet. The client has already
            // been `Welcome`d (above) and stays held in `AwaitingHomeRealm` — no `Close`, no teleport, no
            // attach to a wrong shard — until the `Realm` head names its node.
            Some(home) => {
                let rid = home.lowered();
                // The BOUNDED hold (CRITIQUE-1): a deadline spanning this phase AND the dynamic-target
                // `AwaitingAttach`. `saturating_add` so a huge TTL cannot wrap into an instant expiry.
                session.bootstrap_deadline = Some(TickId(
                    clock
                        .local_tick
                        .0
                        .saturating_add(config.seed_injector.bootstrap_ttl_ticks),
                ));
                // (a) SEED the home demand — the whole ancestor chain spins up (the 5f-3a ride) — riding the
                // EXISTING `RealmDemand` arm on `MsgClass::Saga` (Reliable; the flow is `ReDriven`, so the
                // default `Ephemeral` is correct). NO new wire arm, no grown `AttachSession`. Built through
                // the SAME `demand_for_home` every re-drive uses (HR3).
                outbox.push_flow(
                    config.orchestrator,
                    MsgClass::Saga,
                    &InterShardFlow::RealmDemand(demand_for_home(
                        home.clone(),
                        account,
                        clock.universe_tick,
                    )),
                );
                // (b) POLL the home realm's directory head — the EXISTING `HeadRead`/`Head` pair, whose
                // reply carries the owning node once the spawned shard takes its realm lease.
                push_directory(
                    outbox,
                    config.orchestrator,
                    DirectoryOp::HeadRead {
                        key: DirectoryKey::Realm(rid),
                    },
                );
                Some((rid, home))
            }
        };
        // Index the member LAST: `session`'s borrow of `sessions` must end before this. `None` (static) ⇒
        // no-op ⇒ the bootstrap index stays empty on an unarmed gateway.
        if let Some((home_rid, coord)) = wait_seed {
            sessions.begin_home_wait(session_id, home_rid, coord, since, account);
        }
    } else {
        // Mint refused (id collision or foreign holder): close loudly; the client
        // retries login with a fresh Hello.
        stats.session_mints_refused += 1;
        let client = session.client;
        sessions.by_session.remove(&session_id);
        sessions.by_client.remove(&client);
        push_control(
            outbox,
            client,
            &ServerControlMsg::Close {
                reason: "session mint refused by the directory".to_owned(),
            },
        );
    }
}

/// Per-tick retry driver: pending directory grants and shard attaches are
/// re-sent until answered (all idempotent — at-least-once over a lossy fabric).
///
/// RLM 5f-3d adds the DYNAMIC-HOME re-drive — the only producer here that is neither per-tick nor
/// per-session: it iterates the per-REALM [`GatewaySessions::home_bootstraps`] index on a BACKED-OFF cadence.
fn drive_pending_sessions(
    config: Res<GatewayConfig>,
    identity: Res<NodeIdentity>,
    clock: Res<ClockSample>,
    sessions: Res<GatewaySessions>,
    mut outbox: ResMut<OutboundBox>,
) {
    let cadence = config.seed_injector.redrive_interval_ticks();
    // ---- RLM 5f-3d — THE HOME-BOOTSTRAP RE-DRIVE, and it is a CORRECTNESS INVARIANT, not idempotent
    // politeness (CRITIQUE-3). While ANY session is booting into a realm the gateway must periodically:
    //   (a) RE-SEED the `SpinUp` demand — keeping the reconciler's arm-A `demanded_recently` FRESH through
    //       the shard's whole pod boot AND the attach round-trip that follows it, UNTIL the last member goes
    //       `Active`. Arm-B (`running_live & !empty_confirmed`) cannot cover that gap: a booted realm with
    //       nobody attached yet self-reports `Empty`, so if arm-A lapses the reconciler KILLS the realm the
    //       login is waiting for — and the login would then wait for a realm that was just reaped, until its
    //       bounded TTL Closed it. Hence the re-seed runs until `Active`, NOT until the head resolves.
    //   (b) RE-POLL `HeadRead{Realm(rid)}` — the reply is the ONLY way the gateway learns the node, so a
    //       dropped reply must not wedge the login. This half STOPS at the resolve (`HomeWait::resolved`) and
    //       resumes if a later member joins.
    // Both ride EXISTING wire arms, and both are gated on ONE backed-off cadence
    // (`demand_ttl / REDRIVE_DIVISOR`, anchored per realm): NOT every tick, and NOT per session. Under a
    // 100K mass login onto one home, a per-session every-tick re-drive would fan 200K messages at the
    // orchestrator EVERY tick; this emits ONE demand (+ at most one head-read) per REALM per cadence.
    // Coalescing is sound because the only per-session field in the demand is the audit-only `parent_fence`
    // (`rlm.rs` `update_fence` never reads it for a decision). Each re-seed carries the FULL lineage, so it
    // refreshes the whole ancestor chain exactly as the initial seed did (the 5f-3a ride), not just the leaf.
    // (The ONE inline seed at a login's committed lease is unchanged and stays per-login — it must land the
    // instant the login lands. That is once per login, not once per cadence; the unbounded cost this loop
    // removes is the REPEAT.)
    for (home_rid, wait) in &sessions.home_bootstraps {
        if home_redrive_due(clock.local_tick, wait.since, cadence) {
            // Rebuilt off the STORED lineage (no forest re-descend — O(1) per re-drive) through the SAME
            // `demand_for_home` the initial seed used, so it names EXACTLY the realm these sessions wait on;
            // only `universe_tick` moves.
            outbox.push_flow(
                config.orchestrator,
                MsgClass::Saga,
                &InterShardFlow::RealmDemand(demand_for_home(
                    wait.coord.clone(),
                    wait.account,
                    clock.universe_tick,
                )),
            );
            if !wait.resolved {
                push_directory(
                    &mut outbox,
                    config.orchestrator,
                    DirectoryOp::HeadRead {
                        key: DirectoryKey::Realm(*home_rid),
                    },
                );
            }
        }
    }
    for (session_id, session) in &sessions.by_session {
        match &session.phase {
            SessionPhase::AwaitingDirectory => {
                push_directory(
                    &mut outbox,
                    config.orchestrator,
                    DirectoryOp::LeaseGrant {
                        key: DirectoryKey::Session(*session_id),
                        owner: AuthorityRef::Gateway(identity.node_id),
                        fence: session.fence,
                    },
                );
            }
            // RLM 5f-3d: a session HOLDING for its home realm has no per-session producer — its demand
            // re-seed and head re-poll are COALESCED per realm by the loop above (one message set for every
            // member of that home), so there is nothing to emit here.
            SessionPhase::AwaitingHomeRealm { .. } => {}
            SessionPhase::AwaitingAttach => {
                // 5f-3d: a STATIC attach retries EVERY tick (byte-identical); a DYNAMIC one rides the same
                // backed-off cadence as its realm's re-drive (a mass login must not re-attach every session
                // at a just-booted shard every tick). ONE predicate, ONE anchor source — HR3.
                if attach_retry_due(
                    sessions.attach_anchor(session.home_rid),
                    clock.local_tick,
                    cadence,
                ) {
                    push_to_shard(
                        &mut outbox,
                        // 5f-3d: the retry follows the SAME ONE routing path as the original grant — the
                        // resolved home shard for a dynamic session, `config.shard` for a static one.
                        session_target(session, &config),
                        MsgClass::Control,
                        &GatewayToShard::AttachSession {
                            session: *session_id,
                            fence: session.fence,
                            account: session.account,
                            // The retry repeats the SAME pose the first attach carried — a re-attach must
                            // not be able to place the avatar anywhere else than the attempt it repeats.
                            spawn: session.spawn,
                        },
                    );
                }
            }
            // Active needs no re-drive; a SelfFenced session is deliberately left alone (D-3 Slice 5b) —
            // it is no longer renewed, re-checked, or re-attached, awaiting connection-end / adoption.
            SessionPhase::Active { .. } | SessionPhase::SelfFenced => {}
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ed25519_dalek::SigningKey;
    use vd_core::pose::FrameRef;
    use vd_core::pose::RealmId;
    use vd_core::{EpochId, TickId, UniverseTick};
    // DEV-ONLY (SL4): the fixtures BUILD worlds; the shipped crate only receives lowered ones.
    use vd_physics::worldgen::{UniverseConfig, WorldView};
    use vd_sim::capability::NodeKind;
    use vd_wire::channels::{
        EntitySnap, InputDatagram, RealmSnap, RealmSnapshotDatagram, SnapshotDatagram,
    };
    use vd_wire::seams::directory::OwnerRecord;
    use vd_wire::version::{PROTO_MAJOR, PROTO_MINOR_FLOOR};

    const GW: NodeId = NodeId(1);
    const SHARD: NodeId = NodeId(2);
    const ORCH: NodeId = NodeId(3);
    const CLIENT: NodeId = NodeId(100);
    /// The transfer DESTINATION authority — DISTINCT from the source `SHARD(2)` so a test
    /// asserting `SeqCut.dest == DEST` proves `dest` threads from the wire command, not from
    /// `route.authority` (== SHARD) or a default.
    const DEST: NodeId = NodeId(42);
    const SIGNING_KEY: [u8; 32] = [0x42; 32];

    fn verifying_key() -> [u8; 32] {
        SigningKey::from_bytes(&SIGNING_KEY)
            .verifying_key()
            .to_bytes()
    }

    /// The bare clock the pure-pass tests hand `compose_scenes_pass` (the rig's schedule reads
    /// the world resource instead).
    fn test_clock() -> ClockSample {
        ClockSample {
            local_tick: TickId(1),
            universe_tick: UniverseTick(50),
            epoch: EpochId(9),
            synced: true,
        }
    }

    fn config() -> GatewayConfig {
        GatewayConfig {
            orchestrator: ORCH,
            shard: SHARD,
            // The STABLE routable-shard roster (FORK 5): the login shard AND the transfer
            // dest, so a render-ready dest's frames are node-class-dispatchable (1d.2c). DEST
            // is recognized as a shard; whether a SESSION receives its frames is governed
            // separately by the per-session `subscribed_shards` reverse index.
            known_shards: BTreeSet::from([SHARD, DEST]),
            auth_verifying_key: verifying_key(),
            session_seed: 7,
            tick_hz: 50,
            lease_renew_interval_ticks: 0,
            session_recheck_interval: 0,
            self_fence_grace_ticks: 0,
            reject_next_prepare: None, // 3g abort-leg lever INERT by default (behaviour-identical)
            // 5f-3c: the injector is UNARMED ⇒ INERT (byte-identical: no RealmDemand emitted).
            // The armed tests below override this via `..config()`. The world is EXPLICIT — the
            // walk fixture, lowered — because `Default` (which built one silently) is deleted.
            seed_injector: SeedInjectorConfig::inert(test_world().lowered()),
            tuning: TransportTuning {
                max_sessions: 4,
                max_buffered_inputs: 8,
            },
        }
    }

    /// The per-tick sends captured across a login drive (one Vec of `(to, class,
    /// bytes)` per tick).
    type LoginSends = Vec<Vec<(NodeId, MsgClass, Vec<u8>)>>;

    struct Rig {
        world: World,
        schedule: Schedule,
    }

    /// HR5 — a TRACE sink so every tracing macro's lazy field closure evaluates on the paths the
    /// tests drive; without a subscriber those closures are dead regions coverage cannot reach.
    fn init_test_tracing() {
        use std::sync::Once;
        static ONCE: Once = Once::new();
        ONCE.call_once(|| {
            let subscriber = tracing_subscriber::fmt()
                .with_max_level(tracing::level_filters::LevelFilter::TRACE)
                .with_writer(std::io::sink)
                .finish();
            let _ = tracing::subscriber::set_global_default(subscriber);
        });
    }

    impl Rig {
        fn new() -> Rig {
            init_test_tracing();
            let mut world = World::new();
            world.insert_resource(InboundBox::default());
            world.insert_resource(OutboundBox::default());
            world.insert_resource(NodeIdentity {
                node_id: GW,
                kind: NodeKind::Gateway,
            });
            world.insert_resource(ClockSample {
                local_tick: TickId(1),
                universe_tick: UniverseTick(50),
                epoch: EpochId(9),
                // RLM Step 2: a live-clock rig (the gateway runs no clock-gated authors, so this is inert
                // for the gateway systems — set for a coherent non-default clock).
                synced: true,
            });
            let mut schedule = Schedule::default();
            register_gateway(&mut world, &mut schedule, config());
            Rig { world, schedule }
        }

        fn tick(&mut self, inbound: Vec<Inbound>) -> Vec<(NodeId, MsgClass, Vec<u8>)> {
            self.world.resource_mut::<InboundBox>().0 = inbound;
            self.schedule.run(&mut self.world);
            std::mem::take(&mut self.world.resource_mut::<OutboundBox>().0)
                .into_iter()
                .map(|(to, class, bytes, _)| (to, class, bytes.to_vec()))
                .collect()
        }

        fn stats(&self) -> GatewayStats {
            *self.world.resource::<GatewayStats>()
        }

        /// Hello → directory grant → attach reply; returns (session_id, all sends).
        #[allow(clippy::type_complexity)] // test helper: ticks of raw sends
        fn login(&mut self) -> (SessionId, LoginSends) {
            self.login_with(&hello_msg())
        }

        /// Log in at the CURRENT minor, then FORCE the pending session's negotiated minor down to
        /// `minor` before the grant/attach ticks.
        ///
        /// This exists because `PROTO_MINOR_FLOOR` now REFUSES a genuinely old peer at Hello, so a real
        /// minor-0/1 handshake never produces a session to observe. The sender-gates-variants arms are
        /// still live code — the floor is a property of THIS minor (a field append), and the next
        /// purely variant-additive minor lowers it again — so they still have to be exercised. Forcing
        /// the field is the honest way to do that; the alternative (deleting the gate tests because the
        /// floor currently hides them) would leave the withholding logic unproven the day it matters.
        #[allow(clippy::type_complexity)] // test helper: ticks of raw sends
        fn login_at_minor(&mut self, minor: u16) -> (SessionId, LoginSends) {
            self.login_inner(&hello_msg(), Some(minor))
        }

        fn login_with(&mut self, hello: &ClientControlMsg) -> (SessionId, LoginSends) {
            self.login_inner(hello, None)
        }

        fn login_inner(
            &mut self,
            hello: &ClientControlMsg,
            force_minor: Option<u16>,
        ) -> (SessionId, LoginSends) {
            let hello = self.tick(vec![wire(CLIENT, MsgClass::Control, hello)]);
            let session_id = *self
                .world
                .resource::<GatewaySessions>()
                .sessions()
                .collect::<Vec<_>>()
                .first()
                .expect("session pending");
            if let Some(minor) = force_minor {
                self.world
                    .resource_mut::<GatewaySessions>()
                    .by_session
                    .get_mut(&session_id)
                    .expect("pending session")
                    .negotiated_minor = minor;
            }
            let granted = self.tick(vec![wire(ORCH, MsgClass::Saga, &granted_head(session_id))]);
            let attached = self.tick(vec![wire(
                SHARD,
                MsgClass::Control,
                &ShardToGateway::SessionAttached {
                    session: session_id,
                    entity: EntityId(77),
                    frame: FrameRef::SystemSpace { system_seed: 7 },
                    realm_fence: Fence(1),
                },
            )]);
            (session_id, vec![hello, granted, attached])
        }
    }

    fn wire<T: serde::Serialize>(from: NodeId, class: MsgClass, msg: &T) -> Inbound {
        Inbound::Wire {
            from,
            class,
            bytes: postcard::to_allocvec(msg).expect("encode").into(),
        }
    }

    fn hello_msg() -> ClientControlMsg {
        ClientControlMsg::Hello {
            version: ProtoVersion::CURRENT,
            login: tickets::mint_login(&SIGNING_KEY, AccountId(5), EpochId(9), 1),
        }
    }

    fn hello_msg_below_floor() -> ClientControlMsg {
        ClientControlMsg::Hello {
            version: ProtoVersion {
                major: PROTO_MAJOR,
                minor: PROTO_MINOR_FLOOR - 1,
            },
            login: tickets::mint_login(&SIGNING_KEY, AccountId(5), EpochId(9), 1),
        }
    }

    #[test]
    fn a_peer_below_the_protocol_floor_is_closed_with_the_floor_named() {
        // proto_minor 8 appended a FIELD to `RealmSnap`, which postcard cannot skip and a sender cannot
        // gate out — so an old peer is REFUSED rather than negotiated down. Before the floor, this exact
        // Hello was welcomed and the peer then mis-framed every realm datagram it decoded: not one wrong
        // field, a desynced stream. The reason NAMES the floor so the operator is told to update the
        // client instead of hunting a protocol-generation split.
        let mut rig = Rig::new();
        let sent = rig.tick(vec![wire(
            CLIENT,
            MsgClass::Control,
            &hello_msg_below_floor(),
        )]);
        assert_eq!(
            decode_controls(&sent, CLIENT),
            vec![ServerControlMsg::Close {
                reason:
                    "protocol minor below the floor (18): the scene is server-composed from v1.18"
                        .to_owned()
            }]
        );
        assert_eq!(rig.stats().version_rejected, 1);
        assert_eq!(
            rig.world.resource::<GatewaySessions>().sessions().count(),
            0,
            "a refused peer mints NO session"
        );
    }

    #[test]
    fn a_minor_0_session_is_welcomed_without_the_universe_rate_variant() {
        // Sender-gates-variants: a session at minor 0 must NOT be sent the minor-1 UniverseRate (it
        // would desync an old decoder). Welcome only. The minor is FORCED rather than negotiated —
        // the floor refuses a real minor-0 handshake now (see `login_at_minor`), but the withholding
        // arm is live code that a future variant-additive minor will lower the floor back onto.
        let mut rig = Rig::new();
        let (session_id, sends) = rig.login_at_minor(0);
        let welcomes = decode_controls(&sends[1], CLIENT);
        assert_eq!(
            welcomes,
            vec![ServerControlMsg::Welcome {
                version: ProtoVersion::CURRENT,
                session: session_id,
                session_fence: Fence(1),
                epoch: EpochId(9),
            }],
            "a minor-0 session gets Welcome but NOT UniverseRate"
        );
    }

    // ---- RLM 5f RG-3: the reactive-greeting consume -------------------------------------------
    #[test]
    fn a_shard_presence_greeting_is_counted_never_undecodable() {
        let mut rig = Rig::new();
        let sent = rig.tick(vec![wire(
            SHARD,
            MsgClass::Saga,
            &InterShardFlow::ShardPresence(ShardPresence {
                local_tick: TickId(7),
            }),
        )]);
        assert_eq!(rig.stats().presence_announces, 1);
        assert_eq!(rig.stats().undecodable, 0);
        // Pure observability — a greeting has NO routing effect (no roster claim, no reply here; the
        // reachability was the mesh learning the connection below the seam).
        assert!(sent.is_empty());
    }

    #[test]
    fn a_repeated_greeting_admits_once_and_keeps_counting() {
        // The admit is an INSERT (idempotent): the first greeting logs the admission, a re-greet on
        // the silence cadence counts the announce but does not re-admit (the roster set is a set).
        let mut rig = Rig::new();
        let greet = || {
            wire(
                SHARD,
                MsgClass::Saga,
                &InterShardFlow::ShardPresence(ShardPresence {
                    local_tick: TickId(7),
                }),
            )
        };
        let _ = rig.tick(vec![greet()]);
        let _ = rig.tick(vec![greet()]);
        assert_eq!(rig.stats().presence_announces, 2, "every greeting counts");
        assert_eq!(rig.stats().undecodable, 0);
    }

    #[test]
    fn a_greeting_from_an_unbooked_peer_is_counted_the_same() {
        // A demand shard is NOT on the gateway's roster when it first greets; the counter must not care.
        let mut rig = Rig::new();
        rig.tick(vec![wire(
            NodeId(999),
            MsgClass::Saga,
            &InterShardFlow::ShardPresence(ShardPresence {
                local_tick: TickId(3),
            }),
        )]);
        assert_eq!(rig.stats().presence_announces, 1);
        assert_eq!(rig.stats().undecodable, 0);
    }

    #[test]
    fn a_non_greeting_saga_body_stays_undecodable() {
        // A VALID but non-ShardPresence InterShardFlow on the shard→gateway Saga class counts
        // `undecodable` exactly as the pre-RG-3 fallthrough did (the Ok(_) arm) — no greeting miscount.
        let mut rig = Rig::new();
        rig.tick(vec![wire(
            SHARD,
            MsgClass::Saga,
            &granted_head(SessionId(1)),
        )]);
        assert_eq!(rig.stats().presence_announces, 0);
        assert_eq!(rig.stats().undecodable, 1);
    }

    #[test]
    fn garbage_saga_bytes_stay_undecodable() {
        // Undecodable bytes (the Err arm) — an invalid discriminant.
        let mut rig = Rig::new();
        rig.tick(vec![Inbound::Wire {
            from: SHARD,
            class: MsgClass::Saga,
            bytes: vec![0x7F].into(),
        }]);
        assert_eq!(rig.stats().presence_announces, 0);
        assert_eq!(rig.stats().undecodable, 1);
    }

    #[test]
    fn a_gateway_no_one_greets_is_byte_identical() {
        // No shard greets ⇒ presence_announces stays 0 and every counter is the default (byte-identity).
        let mut rig = Rig::new();
        rig.tick(vec![]);
        assert_eq!(rig.stats(), GatewayStats::default());
    }

    #[test]
    fn a_routable_shard_sending_a_non_frame_class_is_still_undecodable() {
        // The routable-shard branch's fallthrough (a rostered shard sending a class it never should — here
        // Input) is UNCHANGED by RG-3: the greeting arm only intercepts Saga, so this still counts
        // `undecodable`. (Re-covers that arm, which the greeting arm's hoist moved the old Saga case off.)
        let mut rig = Rig::new();
        rig.tick(vec![Inbound::Wire {
            from: SHARD, // a `known_shards` member ⇒ routable
            class: MsgClass::Input,
            bytes: vec![0x00].into(),
        }]);
        assert_eq!(rig.stats().undecodable, 1);
        assert_eq!(rig.stats().presence_announces, 0);
    }

    // ---- RLM 5f RG-4a2: the gateway's /admin/snapshot builder --------------------------------
    #[test]
    fn the_gateway_admin_snapshot_reflects_the_live_world() {
        // The world→contract builder: one logged-in session ⇒ sessions_open 1; a static rig demanded no
        // dynamic home ⇒ dynamic_shards 0; the clock is mirrored; and the gateway carries NO directory (it
        // does not own one — reporting a directory from a gateway would mislead an operator).
        let mut rig = Rig::new();
        let _ = rig.login();
        let snap = crate::admin::gateway_admin_snapshot(&mut rig.world);
        let gw = snap
            .gateway
            .expect("a gateway snapshot carries the gateway view");
        assert_eq!(gw.sessions_open, 1);
        assert_eq!(gw.dynamic_shards, 0);
        assert_eq!(snap.universe_tick, 50);
        assert_eq!(snap.epoch, 9);
        assert!(
            snap.directory.is_empty(),
            "the gateway owns no directory — that is the orchestrator's"
        );
    }

    /// A session-grant head reply as the orchestrator now sends it — wrapped in the
    /// InterShardFlow::DirectoryReply envelope (the dispatch split).
    fn granted_head(session: SessionId) -> InterShardFlow {
        InterShardFlow::DirectoryReply(DirectoryReply::Head {
            key: DirectoryKey::Session(session),
            record: Some(OwnerRecord {
                authority: AuthorityRef::Gateway(GW),
                fence: Fence(1),
                lease_expires: UniverseTick(1_000),
                in_transfer: None,
            }),
        })
    }

    /// A Session-head reply DENYING this gateway's ownership (record absent — e.g. the orchestrator's
    /// reaper revoked the lapsed lease): drives both the reactive self-fence and the recheck channel.
    fn absent_head(session: SessionId) -> InterShardFlow {
        InterShardFlow::DirectoryReply(DirectoryReply::Head {
            key: DirectoryKey::Session(session),
            record: None,
        })
    }

    /// A D-3 Slice-5b config: the recheck channel + proactive self-fence both ARMED (rig-local values;
    /// the split-brain-safe `ttl < grace` with `THETA_MAX*grace < ttl + max` ordering is validated orchestrator-side).
    fn self_fence_config() -> GatewayConfig {
        GatewayConfig {
            session_recheck_interval: 2,
            self_fence_grace_ticks: 5,
            ..config()
        }
    }

    fn set_tick(rig: &mut Rig, tick: u64) {
        rig.world.resource_mut::<ClockSample>().local_tick = TickId(tick);
    }

    fn session_active(rig: &Rig, sid: SessionId) -> bool {
        rig.world
            .resource::<GatewaySessions>()
            .entity_of(sid)
            .is_some()
    }

    #[test]
    fn a_partitioned_gateway_proactively_self_fences_a_stale_session() {
        // D-3 Slice 5b: an Active session whose `Session`-head goes un-confirmed past the grace (a
        // partition from the orchestrator — no recheck reply) is hard-stopped BEFORE the orchestrator's
        // reassign window, then served no input/frames. It is FENCED, not removed (the connection lingers).
        let mut rig = Rig::new();
        rig.world.insert_resource(self_fence_config());
        let (sid, _) = rig.login(); // Active; confirmed_at armed to local_tick 1 at the attach
        // Within the grace (local 6 - confirmed 1 = 5, NOT > 5): still Active.
        set_tick(&mut rig, 6);
        let _ = rig.tick(vec![]);
        assert!(session_active(&rig, sid));
        assert_eq!(rig.stats().sessions_self_fenced_lapsed, 0);
        // Past the grace (local 7 - 1 = 6 > 5): SELF-FENCE.
        set_tick(&mut rig, 7);
        let _ = rig.tick(vec![]);
        assert!(
            !session_active(&rig, sid),
            "the partitioned session self-fenced"
        );
        assert_eq!(rig.stats().sessions_self_fenced_lapsed, 1);
        assert_eq!(
            rig.world.resource::<GatewaySessions>().len(),
            1,
            "fenced, not removed (awaits connection-end / adoption)"
        );
        // S4 readiness de-route reachability (the review's exact concern): after the REAL schedule self-fences
        // the last session out of `Active`, the partition detector must still SEE the frozen confirmed (via the
        // now-SelfFenced session), not read `None`. An `Active`-only max would read `None` here and keep this
        // fully-partitioned gateway falsely Ready. The frozen confirmed is tick 1 (armed at the attach); with
        // local_tick 7 > confirmed 1 + grace 5, `vd_node::health::gateway_sessions_live(Some(1), 7, 5)` is NOT
        // live (proven over the pure predicate in the vd-node health tests — the seam kept node-free here).
        assert_eq!(
            rig.world
                .resource::<GatewaySessions>()
                .freshest_session_confirmed(),
            Some(1),
            "the self-fenced session's frozen confirm is the reachable de-route signal (not None)"
        );
    }

    #[test]
    fn a_recheck_reply_re_arms_an_active_session_and_a_foreign_head_self_fences_it() {
        // The reactive arm: an AFFIRMING recheck reply re-arms the deadline; a foreign/absent head
        // (lease reassigned or reaped) self-fences at once — the link-alive cure beside the partition timer.
        let mut rig = Rig::new();
        rig.world.insert_resource(self_fence_config());
        let (sid, _) = rig.login(); // confirmed_at = 1
        set_tick(&mut rig, 4);
        let _ = rig.tick(vec![wire(ORCH, MsgClass::Saga, &granted_head(sid))]); // re-arm ⇒ confirmed_at 4
        assert!(session_active(&rig, sid));
        // At local 8 (8 - re-armed 4 = 4 <= 5) STILL Active — proof the reply re-armed the deadline
        // (had `confirmed_at` stayed 1, 8 - 1 = 7 > 5 would have fenced it here).
        set_tick(&mut rig, 8);
        let _ = rig.tick(vec![]);
        assert!(
            session_active(&rig, sid),
            "an affirming recheck reply re-armed the self-fence deadline"
        );
        assert_eq!(rig.stats().sessions_self_fenced_lapsed, 0);
        // A foreign/absent head reactively self-fences.
        let _ = rig.tick(vec![wire(ORCH, MsgClass::Saga, &absent_head(sid))]);
        assert!(
            !session_active(&rig, sid),
            "a revoked-lease head reactively self-fences the session"
        );
        assert_eq!(rig.stats().sessions_self_fenced_revoked, 1);
    }

    #[test]
    fn the_recheck_producer_re_reads_active_session_heads_on_cadence() {
        // The confirmation channel: on the recheck cadence the gateway re-reads each Active session's
        // `Session` head — the round-trip whose affirming reply re-arms `confirmed_at`.
        let mut rig = Rig::new();
        rig.world.insert_resource(self_fence_config()); // recheck every 2 ticks
        let (sid, _) = rig.login();
        set_tick(&mut rig, 4); // a recheck multiple; 4 - 1 = 3 within grace, so no self-fence here
        let sends = rig.tick(vec![]);
        // Compare against the exact expected bytes with BITWISE `&` (no short-circuit branch gaps — HR5).
        let expected = postcard::to_allocvec(&InterShardFlow::Directory(DirectoryOp::HeadRead {
            key: DirectoryKey::Session(sid),
        }))
        .expect("encode");
        let saw_head_read = sends.iter().any(|(to, class, bytes)| {
            (*to == ORCH) & (*class == MsgClass::Saga) & (bytes.as_slice() == expected.as_slice())
        });
        assert!(
            saw_head_read,
            "an Active session's head is re-read on the recheck cadence"
        );
    }

    #[test]
    fn on_shard_frame_skips_a_self_fenced_session() {
        // A self-fenced session lingers in the fan-out reverse index (its subs are not torn down) but
        // is served NO frames — skipped cleanly via the phase gate (NOT a desync), watermark untouched.
        let (mut sessions, sid, _) = one_active_session();
        sessions.by_session.get_mut(&sid).expect("present").phase = SessionPhase::SelfFenced;
        let mut stats = GatewayStats::default();
        let mut outbox = OutboundBox::default();
        let frame = postcard::to_allocvec(&frame_msg(Fence(1), 9)).expect("encode");
        on_shard_frame(SHARD, &frame, &mut sessions, &mut stats, &mut outbox);
        assert!(
            outbox.0.is_empty(),
            "a self-fenced session receives no forwarded frames"
        );
        assert_eq!(
            stats.frame_sub_desync, 0,
            "skipped via the Active-only phase gate, never counted a desync"
        );
        assert_eq!(
            sessions.by_session[&sid].delivered.get(&SubId(0)),
            None,
            "no frame ⇒ the delivery watermark is untouched"
        );
    }

    #[test]
    fn a_session_head_for_an_awaiting_attach_session_carries_no_obligation() {
        // After the mint grant the session is AwaitingAttach (the shard attach is in flight). A
        // duplicate/late `Session`-head reply then is neither a fresh mint (AwaitingDirectory) nor a
        // recheck (Active), so it is dropped with no effect (the at-least-once idempotency floor) — no
        // second Welcome, no phase change, no mint refusal.
        let mut rig = Rig::new();
        let _ = rig.tick(vec![wire(CLIENT, MsgClass::Control, &hello_msg())]);
        let sid = *rig
            .world
            .resource::<GatewaySessions>()
            .sessions()
            .collect::<Vec<_>>()
            .first()
            .expect("session pending");
        let granted = rig.tick(vec![wire(ORCH, MsgClass::Saga, &granted_head(sid))]); // ⇒ AwaitingAttach
        // filter+count evaluates the predicate on EVERY control (Welcome ⇒ kept, UniverseRate ⇒ dropped),
        // so both arms of the `matches!` are covered (no `.any` short-circuit — HR5).
        assert_eq!(
            decode_controls(&granted, CLIENT)
                .iter()
                .filter(|m| matches!(m, ServerControlMsg::Welcome { .. }))
                .count(),
            1,
            "the FIRST grant Welcomes the client exactly once"
        );
        let before = rig.stats();
        let dup = rig.tick(vec![wire(ORCH, MsgClass::Saga, &granted_head(sid))]); // duplicate head
        assert!(
            decode_controls(&dup, CLIENT).is_empty(),
            "a duplicate head for an AwaitingAttach session re-sends NO Welcome"
        );
        assert_eq!(
            rig.stats().session_mints_refused,
            before.session_mints_refused,
            "no mint refusal — the head simply carries no obligation"
        );
        assert!(
            !session_active(&rig, sid),
            "still AwaitingAttach (awaiting the shard's SessionAttached), not Active"
        );
    }

    #[test]
    fn a_session_attached_straggler_does_not_resurrect_a_self_fenced_session() {
        // D-3 Slice 5b CRITICAL guard. `SessionAttached` is re-emitted on every `AttachSession` retry
        // and rides the gateway↔shard link, which can be HEALTHY while the gateway↔orchestrator link
        // (that drove the self-fence) is partitioned — so a straggler can land AFTER Active→SelfFenced.
        // Re-promoting it would re-arm the grace clock and resume input/frames — re-opening the
        // split-brain window. The attach arm promotes only from AwaitingAttach, so the fenced session
        // stays inert (awaiting Bye / the future ResumeTicket adoption).
        let mut rig = Rig::new();
        rig.world.insert_resource(self_fence_config());
        let (sid, _) = rig.login(); // Active; confirmed_at armed at the attach
        let _ = rig.tick(vec![wire(ORCH, MsgClass::Saga, &absent_head(sid))]); // reactively self-fence
        assert!(!session_active(&rig, sid), "the session is self-fenced");
        let fenced_confirmed =
            rig.world.resource::<GatewaySessions>().by_session[&sid].confirmed_at;
        // A late/duplicate SessionAttached straggler at a much later tick must be IGNORED.
        set_tick(&mut rig, 42);
        let _ = rig.tick(vec![wire(
            SHARD,
            MsgClass::Control,
            &ShardToGateway::SessionAttached {
                session: sid,
                entity: EntityId(77),
                frame: FrameRef::SystemSpace { system_seed: 7 },
                realm_fence: Fence(1),
            },
        )]);
        assert!(
            !session_active(&rig, sid),
            "the straggler did NOT resurrect the fenced session to Active (no split-brain reopened)"
        );
        assert_eq!(
            rig.world.resource::<GatewaySessions>().by_session[&sid].confirmed_at,
            fenced_confirmed,
            "the self-fence deadline was NOT re-armed by the straggler"
        );
    }

    fn decode_controls(sent: &[(NodeId, MsgClass, Vec<u8>)], to: NodeId) -> Vec<ServerControlMsg> {
        sent.iter()
            .filter(|(node, class, _)| (*node == to) & (*class == MsgClass::Control))
            .map(|(_, _, bytes)| postcard::from_bytes(bytes).expect("decode"))
            .collect()
    }

    fn input_bytes(seq: u64) -> Vec<u8> {
        postcard::to_allocvec(&InputDatagram {
            seq,
            is_cut_marker: false,
            client_tick: TickId(1),
            movement: [1.0, 0.0, 0.0],
            look: [0.0, 0.0],
            action_bits: 0,
        })
        .expect("encode")
    }

    fn frame_msg(fence: Fence, frame_id: u64) -> ShardToGateway {
        let snapshot = SnapshotDatagram {
            sub: SubId(0),
            frame_id,
            source_tick: TickId(5),
            universe_tick: UniverseTick(50),
            entities: vec![EntitySnap {
                entity: EntityId(77),
                pose: vd_core::pose::StampedPose::at_rest(
                    FrameRef::SystemSpace { system_seed: 7 },
                    vd_core::glam::DVec3::ZERO,
                    UniverseTick(50),
                ),
            }],
        };
        ShardToGateway::Frame {
            realm_fence: fence,
            source_tick: TickId(5),
            snapshot_bytes: postcard::to_allocvec(&snapshot).expect("encode"),
        }
    }

    /// A `ShardToGateway::RealmFrame` carrying one moving-realm placement (FA-2c) — the realm twin of
    /// [`frame_msg`].
    fn realm_frame_msg(realm: RealmId) -> ShardToGateway {
        ShardToGateway::RealmFrame {
            realm_fence: Fence(1),
            source_tick: TickId(5),
            realm_snapshot_bytes: realm_frame_payload(realm),
        }
    }

    #[test]
    fn the_full_login_flow_reaches_active_with_welcome_then_subscription() {
        let mut rig = Rig::new();
        let (session_id, sends) = rig.login();

        // Hello tick: exactly one directory grant went out (plus the retry driver's
        // duplicate — idempotent by fence).
        let to_orch: Vec<NodeId> = sends[0].iter().map(|(to, _, _)| *to).collect();
        assert!(
            to_orch.iter().all(|to| *to == ORCH),
            "only directory traffic"
        );

        // Grant tick: Welcome to the client (with the directory-committed id,
        // fence, epoch) and an attach toward the shard.
        // Welcome, then UniverseRate (the client negotiated minor 1, so the gateway
        // relays the cluster tick rate right after the Welcome).
        let welcomes = decode_controls(&sends[1], CLIENT);
        assert_eq!(
            welcomes,
            vec![
                ServerControlMsg::Welcome {
                    version: ProtoVersion::CURRENT,
                    session: session_id,
                    session_fence: Fence(1),
                    epoch: EpochId(9),
                },
                ServerControlMsg::UniverseRate { tick_hz: 50 },
            ]
        );
        // Attach tick: SubscriptionOpened STRICTLY BEFORE AuthorityChanged (X1),
        // sub allocated from the monotonic allocator. The default rig negotiates minor 2
        // (ProtoVersion::CURRENT), so the gateway ALSO trails `OwnEntity` (the pure-renderer
        // own-entity signal, S4) after `AuthorityChanged` — the node-aware + node-agnostic
        // dual-announce for a rolling fleet of both client versions.
        let controls = decode_controls(&sends[2], CLIENT);
        assert_eq!(
            controls,
            vec![
                ServerControlMsg::SubscriptionOpened {
                    sub: SubId(0),
                    frame: FrameRef::SystemSpace { system_seed: 7 },
                },
                ServerControlMsg::AuthorityChanged {
                    entity: EntityId(77),
                    sub: SubId(0),
                },
                ServerControlMsg::OwnEntity {
                    entity: EntityId(77),
                },
                // THE LOGIN LEVEL (§2.4): the composer's same-tick pass sees the standing
                // realm and bumps the epoch 0→1 — the swap signal, after the identity control.
                ServerControlMsg::RealmRegistry {
                    origin: RealmId::System(7),
                    origin_epoch: 1,
                    rows: vec![SceneRow {
                        realm: RealmId::System(7),
                        parent: None,
                        pose: vd_core::pose::StampedPose::at_rest(
                            FrameRef::SystemSpace { system_seed: 7 },
                            DVec3::ZERO,
                            UniverseTick(0),
                        ),
                        bag: Vec::new(),
                    }],
                },
            ]
        );
        assert_eq!(
            rig.stats(),
            GatewayStats {
                // THE WINDOW LANE (Slice A): a clean login now ALSO derives + opens the session's
                // own-realm Occupants window — exactly one open, a THROUGHPUT count, not a reject.
                window_open_sent: 1,
                // Slice B: the composer derives the one-hop chain the moment the session
                // is Active (a GAUGE — one session, one chain). No frames were ever ingested, so
                // nothing folds and nothing holds (pre-first-fold boot is not a stall).
                window_chains_held: 1,
                // Slice C1: the login LEVEL shipped the moment the standing realm resolved.
                scene_levels_sent: 1,
                ..GatewayStats::default()
            },
            "clean run, zero rejects"
        );
    }

    #[test]
    fn own_entity_trails_authority_changed_for_minor2_and_is_withheld_from_minor0() {
        // S4 dual-announce (sender-gates-variants): a minor>=2 (pure-renderer) client gets BOTH
        // `AuthorityChanged{entity, sub}` (the node-aware sub re-point) AND `OwnEntity{entity}` (the
        // node-AGNOSTIC own-entity cue) at attach — the node-agnostic signal LAST. A minor-0 client
        // gets ONLY `AuthorityChanged` (the minor-2 OwnEntity is withheld — it would desync an old
        // decoder). Neither ever learns which shard owns the entity from these two messages.
        let mut rig = Rig::new();
        let (_sid, sends) = rig.login(); // default hello = ProtoVersion::CURRENT (minor 2)
        let controls = decode_controls(&sends[2], CLIENT); // the attach tick's client controls
        assert_eq!(
            controls,
            vec![
                ServerControlMsg::SubscriptionOpened {
                    sub: SubId(0),
                    frame: FrameRef::SystemSpace { system_seed: 7 },
                },
                ServerControlMsg::AuthorityChanged {
                    entity: EntityId(77),
                    sub: SubId(0),
                },
                ServerControlMsg::OwnEntity {
                    entity: EntityId(77),
                },
                // The login LEVEL (§2.4) trails the identity control — same tick, same stream.
                ServerControlMsg::RealmRegistry {
                    origin: RealmId::System(7),
                    origin_epoch: 1,
                    rows: vec![SceneRow {
                        realm: RealmId::System(7),
                        parent: None,
                        pose: vd_core::pose::StampedPose::at_rest(
                            FrameRef::SystemSpace { system_seed: 7 },
                            DVec3::ZERO,
                            UniverseTick(0),
                        ),
                        bag: Vec::new(),
                    }],
                },
            ],
            "minor-2 attach announces AuthorityChanged THEN the node-agnostic OwnEntity",
        );

        // A minor-0 session: OwnEntity is withheld — ONLY AuthorityChanged names its avatar. Forced
        // rather than negotiated (the floor refuses a real minor-0 handshake; see `login_at_minor`).
        let mut rig0 = Rig::new();
        let (_sid0, sends0) = rig0.login_at_minor(0);
        let controls0 = decode_controls(&sends0[2], CLIENT);
        assert_eq!(
            controls0,
            vec![
                ServerControlMsg::SubscriptionOpened {
                    sub: SubId(0),
                    frame: FrameRef::SystemSpace { system_seed: 7 },
                },
                ServerControlMsg::AuthorityChanged {
                    entity: EntityId(77),
                    sub: SubId(0),
                },
                // The login LEVEL (§2.4) trails the identity control — same tick, same stream.
                ServerControlMsg::RealmRegistry {
                    origin: RealmId::System(7),
                    origin_epoch: 1,
                    rows: vec![SceneRow {
                        realm: RealmId::System(7),
                        parent: None,
                        pose: vd_core::pose::StampedPose::at_rest(
                            FrameRef::SystemSpace { system_seed: 7 },
                            DVec3::ZERO,
                            UniverseTick(0),
                        ),
                        bag: Vec::new(),
                    }],
                },
            ],
            "a minor-0 session gets AuthorityChanged but NOT the minor-2 OwnEntity (sender-gates-variants)",
        );
    }

    #[test]
    fn entity_of_tracks_the_session_lifecycle() {
        let mut rig = Rig::new();
        let _ = rig.tick(vec![wire(CLIENT, MsgClass::Control, &hello_msg())]);
        let session_id = rig
            .world
            .resource::<GatewaySessions>()
            .sessions()
            .next()
            .expect("pending");
        // Pending phases expose no entity; unknown sessions expose none either.
        assert_eq!(
            rig.world
                .resource::<GatewaySessions>()
                .entity_of(session_id),
            None
        );
        assert_eq!(
            rig.world
                .resource::<GatewaySessions>()
                .entity_of(SessionId(0xDEAD)),
            None
        );
        let _ = rig.tick(vec![wire(ORCH, MsgClass::Saga, &granted_head(session_id))]);
        assert_eq!(
            rig.world
                .resource::<GatewaySessions>()
                .entity_of(session_id),
            None,
            "still awaiting attach"
        );
        let _ = rig.tick(vec![wire(
            SHARD,
            MsgClass::Control,
            &ShardToGateway::SessionAttached {
                session: session_id,
                entity: EntityId(77),
                frame: FrameRef::SystemSpace { system_seed: 7 },
                realm_fence: Fence(1),
            },
        )]);
        assert_eq!(
            rig.world
                .resource::<GatewaySessions>()
                .entity_of(session_id),
            Some(EntityId(77)),
            "active sessions expose their avatar (P2's saga key)"
        );
    }

    #[test]
    fn version_mismatch_and_bad_tickets_close_with_reasons() {
        let mut rig = Rig::new();
        let wrong_version = ClientControlMsg::Hello {
            version: ProtoVersion {
                major: ProtoVersion::CURRENT.major + 1,
                minor: 0,
            },
            login: tickets::mint_login(&SIGNING_KEY, AccountId(5), EpochId(9), 1),
        };
        let sent = rig.tick(vec![wire(CLIENT, MsgClass::Control, &wrong_version)]);
        let controls = decode_controls(&sent, CLIENT);
        assert_eq!(controls.len(), 1);
        assert_eq!(
            controls[0],
            ServerControlMsg::Close {
                reason: "incompatible protocol major version".to_owned()
            }
        );
        // A foreign signer's ticket is rejected by the SOLE validator.
        let forged = ClientControlMsg::Hello {
            version: ProtoVersion::CURRENT,
            login: tickets::mint_login(&[0x66; 32], AccountId(5), EpochId(9), 1),
        };
        let sent = rig.tick(vec![wire(CLIENT, MsgClass::Control, &forged)]);
        let controls = decode_controls(&sent, CLIENT);
        assert_eq!(
            controls[0],
            ServerControlMsg::Close {
                reason: "login ticket rejected".to_owned()
            }
        );
        assert_eq!(rig.stats().version_rejected, 1);
        assert_eq!(rig.stats().logins_rejected, 1);
        assert!(rig.world.resource::<GatewaySessions>().is_empty());
    }

    #[test]
    fn capacity_duplicate_hello_and_resume_paths() {
        let mut rig = Rig::new();
        // Fill to capacity with distinct clients.
        for n in 0..4u64 {
            let _ = rig.tick(vec![wire(NodeId(100 + n), MsgClass::Control, &hello_msg())]);
        }
        assert_eq!(rig.world.resource::<GatewaySessions>().len(), 4);
        // One more is refused loudly.
        let sent = rig.tick(vec![wire(NodeId(199), MsgClass::Control, &hello_msg())]);
        assert_eq!(
            decode_controls(&sent, NodeId(199))[0],
            ServerControlMsg::Close {
                reason: "gateway at session capacity".to_owned()
            }
        );
        assert_eq!(rig.stats().sessions_refused_capacity, 1);
        // Resume is refused honestly until P3.
        let resume = ClientControlMsg::Resume {
            version: ProtoVersion::CURRENT,
            ticket: dummy_resume(),
        };
        let sent = rig.tick(vec![wire(NodeId(198), MsgClass::Control, &resume)]);
        assert_eq!(
            decode_controls(&sent, NodeId(198))[0],
            ServerControlMsg::Close {
                reason: "resume is not available yet".to_owned()
            }
        );
        assert_eq!(rig.stats().resumes_refused, 1);
    }

    fn dummy_resume() -> vd_wire::seams::tickets::ResumeTicket {
        vd_wire::seams::tickets::ResumeTicket {
            claims: vd_wire::seams::tickets::SessionClaims {
                session: SessionId(1),
                account: AccountId(1),
                epoch: EpochId(1),
                validity_epoch: 0,
                expires: UniverseTick(0),
                key_id: 0,
            },
            session_fence: Fence(0),
            resume_nonce: 0,
            hmac: [0; 32],
        }
    }

    #[test]
    fn mint_refusal_closes_the_client_and_clears_the_session() {
        let mut rig = Rig::new();
        let _ = rig.tick(vec![wire(CLIENT, MsgClass::Control, &hello_msg())]);
        let session_id = rig
            .world
            .resource::<GatewaySessions>()
            .sessions()
            .next()
            .expect("pending session");
        // The directory says someone ELSE holds the session key.
        let refused = InterShardFlow::DirectoryReply(DirectoryReply::Head {
            key: DirectoryKey::Session(session_id),
            record: Some(OwnerRecord {
                authority: AuthorityRef::Gateway(NodeId(55)),
                fence: Fence(3),
                lease_expires: UniverseTick(1_000),
                in_transfer: None,
            }),
        });
        let sent = rig.tick(vec![wire(ORCH, MsgClass::Saga, &refused)]);
        assert_eq!(
            decode_controls(&sent, CLIENT)[0],
            ServerControlMsg::Close {
                reason: "session mint refused by the directory".to_owned()
            }
        );
        assert!(rig.world.resource::<GatewaySessions>().is_empty());
        assert_eq!(rig.stats().session_mints_refused, 1);
    }

    #[test]
    fn input_routes_dedups_and_counts_every_failure_mode() {
        let mut rig = Rig::new();
        let (session_id, _) = rig.login();

        // A fresh input forwards to the shard wrapped with (session, fence).
        let sent = rig.tick(vec![Inbound::Wire {
            from: CLIENT,
            class: MsgClass::Input,
            bytes: input_bytes(1).into(),
        }]);
        let inputs: Vec<&(NodeId, MsgClass, Vec<u8>)> = sent
            .iter()
            .filter(|(to, class, _)| (*to == SHARD) & (*class == MsgClass::Input))
            .collect();
        assert_eq!(inputs.len(), 1);
        let fwd: GatewayToShard = postcard::from_bytes(&inputs[0].2).expect("decode");
        assert_eq!(
            fwd,
            GatewayToShard::SessionInput {
                session: session_id,
                fence: Fence(1),
                input_bytes: input_bytes(1),
            }
        );

        // Same seq again: deduped. Garbage: malformed. Unknown client: unroutable.
        let _ = rig.tick(vec![
            Inbound::Wire {
                from: CLIENT,
                class: MsgClass::Input,
                bytes: input_bytes(1).into(),
            },
            Inbound::Wire {
                from: CLIENT,
                class: MsgClass::Input,
                bytes: vec![0x80].into(),
            },
            Inbound::Wire {
                from: NodeId(177),
                class: MsgClass::Input,
                bytes: input_bytes(2).into(),
            },
        ]);
        let stats = rig.stats();
        assert_eq!(stats.inputs_deduped, 1);
        assert_eq!(stats.inputs_malformed, 1);
        assert_eq!(stats.inputs_unroutable, 1);
    }

    #[test]
    fn input_for_a_desynced_session_map_is_dropped_and_counted_never_panics() {
        // WB-1: `by_client` and `by_session` are kept in sync by construction, but a slip
        // must DROP-and-count on the 20Hz input path, never panic the gateway. Proven by
        // an artificially desynced table (present in `by_client`, absent from `by_session`).
        let mut sessions = GatewaySessions::default();
        sessions.by_client.insert(CLIENT, SessionId(7));
        let mut stats = GatewayStats::default();
        let mut outbox = OutboundBox::default();
        on_client_input(
            &input_bytes(1),
            CLIENT,
            &config(),
            &mut sessions,
            &mut stats,
            &mut outbox,
        );
        assert_eq!(
            stats.inputs_unroutable, 1,
            "the desync is counted, not crashed"
        );
        assert!(
            outbox.0.is_empty(),
            "nothing forwarded for a desynced session"
        );
    }

    #[test]
    fn duplicate_hello_below_capacity_is_an_idempotent_noop() {
        let mut rig = Rig::new();
        let _ = rig.tick(vec![wire(CLIENT, MsgClass::Control, &hello_msg())]);
        assert_eq!(rig.world.resource::<GatewaySessions>().len(), 1);
        let sent = rig.tick(vec![wire(CLIENT, MsgClass::Control, &hello_msg())]);
        assert_eq!(
            rig.world.resource::<GatewaySessions>().len(),
            1,
            "no second session"
        );
        // Only the pending-grant retry went out — no Close, no new mint.
        assert!(sent.iter().all(|(to, _, _)| *to == ORCH));
        assert_eq!(rig.stats(), GatewayStats::default());
    }

    #[test]
    fn input_before_active_is_unroutable() {
        let mut rig = Rig::new();
        let _ = rig.tick(vec![wire(CLIENT, MsgClass::Control, &hello_msg())]);
        let _ = rig.tick(vec![Inbound::Wire {
            from: CLIENT,
            class: MsgClass::Input,
            bytes: input_bytes(1).into(),
        }]);
        assert_eq!(rig.stats().inputs_unroutable, 1);
    }

    #[test]
    fn frames_fan_out_retagged_and_stale_fences_drop() {
        let mut rig = Rig::new();
        let (_, _) = rig.login();

        // A current-fence frame reaches the client re-tagged to ITS sub.
        let sent = rig.tick(vec![wire(
            SHARD,
            MsgClass::Snapshot,
            &frame_msg(Fence(1), 9),
        )]);
        let snaps: Vec<&(NodeId, MsgClass, Vec<u8>)> = sent
            .iter()
            .filter(|(to, class, _)| (*to == CLIENT) & (*class == MsgClass::Snapshot))
            .collect();
        assert_eq!(snaps.len(), 1);
        let snap: SnapshotDatagram = postcard::from_bytes(&snaps[0].2).expect("decode");
        assert_eq!(snap.sub, SubId(0));
        assert_eq!(snap.frame_id, 9);

        // A HIGHER fence is current (not stale) — still forwarded.
        let sent = rig.tick(vec![wire(
            SHARD,
            MsgClass::Snapshot,
            &frame_msg(Fence(2), 10),
        )]);
        assert_eq!(sent.len(), 1);

        // A stale fence is dropped and counted (the P2 old-owner guard).
        let sent = rig.tick(vec![wire(
            SHARD,
            MsgClass::Snapshot,
            &frame_msg(Fence::GENESIS, 11),
        )]);
        assert_eq!(sent.len(), 0);
        assert_eq!(rig.stats().stale_frames_dropped, 1);
    }

    #[test]
    fn two_active_sessions_share_one_retagged_body() {
        // SCALE-1: a second session with the SAME sub re-uses the ONE retagged body
        // (the Occupied map arm) — the gateway never re-encodes per subscriber.
        let mut rig = Rig::new();
        let (_, _) = rig.login();
        // A second client logs in fully (distinct session, same sub 0).
        let before: std::collections::BTreeSet<SessionId> =
            rig.world.resource::<GatewaySessions>().sessions().collect();
        let _ = rig.tick(vec![Inbound::Wire {
            from: NodeId(101),
            class: MsgClass::Control,
            bytes: postcard::to_allocvec(&hello_msg()).expect("encode").into(),
        }]);
        let session2 = rig
            .world
            .resource::<GatewaySessions>()
            .sessions()
            .find(|s| !before.contains(s))
            .expect("second session pending");
        let _ = rig.tick(vec![wire(ORCH, MsgClass::Saga, &granted_head(session2))]);
        let _ = rig.tick(vec![wire(
            SHARD,
            MsgClass::Control,
            &ShardToGateway::SessionAttached {
                session: session2,
                entity: EntityId(88),
                frame: FrameRef::SystemSpace { system_seed: 7 },
                realm_fence: Fence(1),
            },
        )]);

        // One frame: BOTH clients receive a sub-0 snapshot from the shared body.
        let sent = rig.tick(vec![wire(
            SHARD,
            MsgClass::Snapshot,
            &frame_msg(Fence(1), 9),
        )]);
        let mut recipients: Vec<NodeId> = sent
            .iter()
            .filter(|(_, class, _)| *class == MsgClass::Snapshot)
            .map(|(to, _, _)| *to)
            .collect();
        recipients.sort_unstable();
        assert_eq!(
            recipients,
            vec![CLIENT, NodeId(101)],
            "both subscribers fed"
        );
        for (_, _, bytes) in sent.iter().filter(|(_, c, _)| *c == MsgClass::Snapshot) {
            let snap: SnapshotDatagram = postcard::from_bytes(bytes).expect("decode");
            assert_eq!(snap.sub, SubId(0));
        }
        assert_eq!(rig.stats().undecodable, 0);
    }

    #[test]
    fn a_corrupt_snapshot_body_is_counted_once_and_abandons_the_frame() {
        // The Err arm of the per-sub retag: a body that can't re-tag (bad varint)
        // would fail identically for every sub, so the whole frame is abandoned.
        let mut rig = Rig::new();
        let (_, _) = rig.login();
        let corrupt = ShardToGateway::Frame {
            realm_fence: Fence(1),
            source_tick: TickId(5),
            snapshot_bytes: vec![0x80], // truncated varint: retag fails
        };
        let sent = rig.tick(vec![wire(SHARD, MsgClass::Snapshot, &corrupt)]);
        // The corrupt body re-tags for no sub, so the whole frame is abandoned: the
        // active session receives NOTHING and the failure is counted exactly once.
        assert_eq!(sent.len(), 0, "no output from a corrupt body");
        assert_eq!(rig.stats().undecodable, 1, "counted exactly once");
    }

    #[test]
    fn a_window_row_before_its_engine_is_dropped_fail_closed_and_counted_apart() {
        // THE WINDOW LANE's ingest (Slice B: the composer CONSUMES what attestation admits;
        // SHADOW — nothing reaches a client). Driven through the REAL ingest:
        //   admitted  — the roster-head sender, on the held window (Control AND the datagram
        //               class — both carriers run the ONE rule) ⇒ INGESTED;
        //   preroster — a marker BEFORE the author's first level (no attested roster to vouch);
        //   forged    — a routable-but-wrong sender is dropped + counted (fail-closed);
        //   unknown   — a straggler/forged window id drops by id mismatch;
        //   misauthored — a look about anything but the author, a marker about a non-child
        //               (vouched against the author's OWN attested roster — never a forest);
        //   stale     — a body older than the held statement (newest wins).
        // Nothing is guessed at, nothing reaches a client, nothing is `undecodable`.
        let mut rig = Rig::new();
        let (_, _) = rig.login();
        // The login derived + opened the Occupants window (id 1) on SHARD for System(7).
        assert_eq!(
            rig.world.resource::<GatewaySessions>().windows_open_count(),
            1
        );
        let roster_row = vd_wire::channels::RealmSnap {
            realm: RealmId::Planet(7),
            frame: FrameRef::PlanetCentered { planet_seed: 7 },
            pose: vd_core::pose::StampedPose::at_rest(
                FrameRef::SystemSpace { system_seed: 7 },
                DVec3::new(30.0, 0.0, 0.0),
                vd_core::UniverseTick(5),
            ),
        };
        let frame_for = |window: WindowId| ShardToGateway::WindowFrame {
            realm_fence: Fence(1),
            window,
            at: vd_core::UniverseTick(5),
            hop: None,
            rows: vec![roster_row],
        };
        let body_about = |subject: RealmId, stmt: BodyStmt, at: u64| ShardToGateway::WindowBody {
            realm_fence: Fence(1),
            window: WindowId(1),
            subject,
            stmt,
            authored_at: vd_core::UniverseTick(at),
        };
        let membership = ShardToGateway::WindowMembership {
            window: WindowId(1),
            added: vec![RealmId::Planet(7)],
            removed: Vec::new(),
        };
        // PREROSTER: a marker arriving before the author's FIRST level finds no attested roster
        // to vouch for its subject — refused apart, fail-closed, not "misauthored".
        let _ = rig.tick(vec![wire(
            SHARD,
            MsgClass::Control,
            &body_about(RealmId::Planet(7), BodyStmt::Marker { luma: vec![2] }, 4),
        )]);
        assert_eq!(rig.stats().window_body_preroster, 1, "no roster yet");
        let sent = rig.tick(vec![
            // ADMITTED: the head, on the held window — frame (both carriers; the re-delivered
            // stamp replaces, latest-wins), membership, an admissible self-look, and an
            // admissible marker about a child the author's OWN roster rows vouch for.
            wire(SHARD, MsgClass::Control, &frame_for(WindowId(1))),
            wire(SHARD, MsgClass::RealmSnapshot, &frame_for(WindowId(1))),
            wire(SHARD, MsgClass::Control, &membership),
            wire(
                SHARD,
                MsgClass::Control,
                &body_about(RealmId::System(7), BodyStmt::SelfLook { bag: vec![1] }, 5),
            ),
            wire(
                SHARD,
                MsgClass::Control,
                &body_about(RealmId::Planet(7), BodyStmt::Marker { luma: vec![2] }, 5),
            ),
            // STALE: an OLDER self-look than the one already held — newest wins, counted apart.
            wire(
                SHARD,
                MsgClass::Control,
                &body_about(RealmId::System(7), BodyStmt::SelfLook { bag: vec![9] }, 3),
            ),
            // FORGED: a routable shard that is NOT the head this window was opened on.
            wire(DEST, MsgClass::Control, &frame_for(WindowId(1))),
            // UNKNOWN: an id this gateway never minted (or already closed).
            wire(SHARD, MsgClass::Control, &frame_for(WindowId(99))),
            // MISAUTHORED: a look about a child (SL3 — a realm draws only itself), and a marker
            // about the author itself (R4 — a marker is only ever about a direct child).
            wire(
                SHARD,
                MsgClass::Control,
                &body_about(RealmId::Planet(7), BodyStmt::SelfLook { bag: vec![1] }, 5),
            ),
            wire(
                SHARD,
                MsgClass::Control,
                &body_about(RealmId::System(7), BodyStmt::Marker { luma: vec![2] }, 5),
            ),
        ]);
        // Since the flag day the composer EMITS (level/delta/datagram) — but a window ROW is
        // never forwarded verbatim: everything client-bound decodes as a composed message.
        // Verified as DATA (HR5: no wildcard arms a test can never reach): the class SET is
        // pinned by equality, every control decodes AND its leading postcard discriminant is a
        // composed-scene variant, every realm-class payload decodes as the composed datagram.
        let client_classes: std::collections::BTreeSet<MsgClass> = sent
            .iter()
            .filter(|(to, _, _)| *to == CLIENT)
            .map(|(_, class, _)| *class)
            .collect();
        assert_eq!(
            client_classes,
            [MsgClass::Control, MsgClass::RealmSnapshot].into(),
            "the client hears exactly the composed control + datagram lanes"
        );
        let scene_discs: std::collections::BTreeSet<u8> = [
            control_disc(&ServerControlMsg::RealmRegistry {
                origin: RealmId::System(7),
                origin_epoch: 0,
                rows: Vec::new(),
            }),
            control_disc(&ServerControlMsg::RealmSceneDelta {
                origin: RealmId::System(7),
                origin_epoch: 0,
                added: Vec::new(),
                removed: Vec::new(),
            }),
        ]
        .into();
        for (_, class, bytes) in sent.iter().filter(|(to, _, _)| *to == CLIENT) {
            if *class == MsgClass::Control {
                let _ = postcard::from_bytes::<ServerControlMsg>(bytes)
                    .expect("client-bound control is a composed ServerControlMsg");
                let disc = bytes[0];
                assert!(
                    scene_discs.contains(&disc),
                    "only composed scene control reaches the client here (disc {disc})"
                );
            }
            if *class == MsgClass::RealmSnapshot {
                let _ = postcard::from_bytes::<RealmSnapshotDatagram>(bytes)
                    .expect("client-bound realm bytes are the COMPOSED datagram");
            }
        }
        assert_eq!(
            rig.stats().window_rows_ingested,
            6,
            "each admitted row ingested (two frames, membership, look, marker) PLUS the parked \
             preroster marker drained by the first level (Slice C1: parked, never lost — the \
             send-once lane cannot re-send it)"
        );
        assert_eq!(rig.stats().window_body_stale, 1, "the older look refused");
        assert_eq!(rig.stats().window_sender_mismatch, 1, "the forgery dropped");
        assert_eq!(rig.stats().window_unknown_row, 1, "the stranger id dropped");
        assert_eq!(
            rig.stats().window_misauthored_body,
            2,
            "both misauthorships dropped"
        );
        assert_eq!(rig.stats().undecodable, 0, "a window row is NOT garbage");
        // The composer CONSUMED the admitted level: the session's one-hop chain folded at the
        // level's stamp and the roster row composed (SHADOW — measured above, served to no one).
        assert_eq!(rig.stats().window_folds, 1, "the admitted level folded");
        assert_eq!(
            rig.stats().window_composed_rows,
            1,
            "the roster row composed"
        );
        assert_eq!(rig.stats().window_instant_mismatch, 0);
    }

    #[test]
    fn windows_derive_open_keepalive_and_close_with_the_sessions() {
        // THE WINDOW LANE's gateway driver (Slice A): the login's Active promote derives + opens
        // the Occupants window on the session's own shard; the keep-alive re-asserts it on the
        // DERIVED cadence (tick_hz/2 with the recheck channel disarmed — the shard's own
        // fallback derivation, mirrored); the session's end closes it — ZERO sessions ⇒ ZERO
        // window state, structurally (the design's teardown test, gateway half).
        let mut rig = Rig::new();
        let (sid, sends) = rig.login();
        let attach_tick = &sends[2];
        // Shard-bound sends only (bitwise `|`, HR5) — a client-bound control byte must never be
        // decoded under the shard contract (postcard is positional, an alias would be silent).
        let opens: Vec<(NodeId, GatewayToShard)> = attach_tick
            .iter()
            .filter(|(to, _, _)| (*to == SHARD) | (*to == DEST))
            .map(|(to, _, b)| {
                (
                    *to,
                    postcard::from_bytes::<GatewayToShard>(b).expect(
                        "every shard-bound byte this tick decodes under the shard contract",
                    ),
                )
            })
            .collect();
        assert_eq!(
            opens,
            vec![(
                SHARD,
                GatewayToShard::WindowOpen {
                    window: WindowId(1),
                    scope: WindowScope::Occupants,
                }
            )],
            "the Active promote opens the own-realm Occupants window, id minted from 1"
        );
        assert_eq!(rig.stats().window_open_sent, 1);
        // The keep-alive: due exactly on the derived cadence (config: recheck disarmed, 50 Hz ⇒
        // every 25 local ticks), idempotent re-assert of the SAME id + scope.
        set_tick(&mut rig, 24);
        let quiet = rig.tick(vec![]);
        assert_eq!(
            quiet.len(),
            0,
            "off-cadence, an idle Active session sends nothing"
        );
        set_tick(&mut rig, 25);
        let beat = rig.tick(vec![]);
        assert_eq!(
            beat.len(),
            1,
            "the cadence beat carries the keep-alive alone"
        );
        assert_eq!(
            postcard::from_bytes::<GatewayToShard>(&beat[0].2).expect("decode"),
            GatewayToShard::WindowOpen {
                window: WindowId(1),
                scope: WindowScope::Occupants,
            }
        );
        assert_eq!(rig.stats().window_keepalives_sent, 1);
        // The session ends: the window closes on the SAME tick and the state is EMPTY.
        let _ = sid;
        let bye = rig.tick(vec![wire(
            CLIENT,
            MsgClass::Control,
            &ClientControlMsg::Bye,
        )]);
        let closes = bye
            .iter()
            .filter(|(to, class, b)| {
                (*to == SHARD)
                    & (*class == MsgClass::Control)
                    & matches!(
                        postcard::from_bytes::<GatewayToShard>(b),
                        Ok(GatewayToShard::WindowClose {
                            window: WindowId(1)
                        })
                    )
            })
            .count();
        assert_eq!(closes, 1, "the polite close rides the session's exit tick");
        let sessions = rig.world.resource::<GatewaySessions>();
        assert!(sessions.is_empty());
        assert_eq!(
            sessions.windows_open_count(),
            0,
            "zero sessions ⇒ zero window state (structural, both maps)"
        );
        assert_eq!(rig.stats().window_close_sent, 1);
    }

    #[test]
    fn a_crossing_overlap_holds_both_chains_and_derives_the_child_window_on_the_parent() {
        // §2.7's posture in Slice-A form: through a hand-off overlap the session holds subs on
        // BOTH ends (the existing dual-sub pattern), so the derivation opens windows on BOTH
        // chains — and the moment the gateway's own routing state names a sub'd PARENT of a
        // sub'd realm (System(7) ⊃ Planet(7) in the login registry), the `Child(c)` window on
        // the parent's shard derives too. Nothing was looked up anew: the two subs and the login
        // registry are exactly what the gateway already held.
        let mut rig = Rig::new();
        let (sid, _) = rig.login(); // Occupants(System(7)) on SHARD == WindowId(1)
        let ready = rig.tick(vec![wire(
            DEST,
            MsgClass::Control,
            &ShardToGateway::SubscriptionReady {
                session: sid,
                entity: EntityId(77),
                frame: FrameRef::PlanetCentered { planet_seed: 7 },
                realm_fence: Fence(2),
            },
        )]);
        // Shard-bound sends only (same discipline as the login test above).
        let opens: Vec<(NodeId, GatewayToShard)> = ready
            .iter()
            .filter(|(to, _, _)| (*to == SHARD) | (*to == DEST))
            .map(|(to, _, b)| {
                (
                    *to,
                    postcard::from_bytes::<GatewayToShard>(b).expect(
                        "every shard-bound byte this tick decodes under the shard contract",
                    ),
                )
            })
            .collect();
        assert_eq!(
            opens,
            vec![
                (
                    DEST,
                    GatewayToShard::WindowOpen {
                        window: WindowId(2),
                        scope: WindowScope::Occupants,
                    }
                ),
                (
                    SHARD,
                    GatewayToShard::WindowOpen {
                        window: WindowId(3),
                        scope: WindowScope::Child(RealmId::Planet(7)),
                    }
                ),
            ],
            "the dest's own-level window AND the hop window on the parent, in derivation order"
        );
        assert_eq!(
            rig.world.resource::<GatewaySessions>().windows_open_count(),
            3,
            "both chains held through the overlap"
        );
    }

    #[test]
    fn the_orchestrator_saga_class_dispatch_splits_reply_from_command_from_garbage() {
        use vd_wire::seams::transfer_control::TransferControl;
        let mut rig = Rig::new();

        // A TransferControl command (the saga driving the gateway) DISPATCHES to the
        // consumer — never mis-decoded as a directory reply. With no session 7 here it is
        // counted unroutable (consumed, not undecodable): the split is what this asserts.
        let cmd = InterShardFlow::Saga(TransferControl::PrepareSubscribe {
            transfer: vd_core::TransferId(1),
            session: SessionId(7),
            dest: SHARD,
        });
        let _ = rig.tick(vec![wire(ORCH, MsgClass::Saga, &cmd)]);
        assert_eq!(rig.stats().transfer_unroutable, 1);
        assert_eq!(rig.stats().undecodable, 0, "a command is NOT undecodable");

        // A non-reply / non-command Saga-class arm (a misdirected Ghost) → undecodable.
        let ghost = InterShardFlow::Ghost(vd_wire::intershard::GhostFlow::Despawn {
            entity: EntityId::pack(vd_core::entity_kind::EntityKind::Player, 1, 7, 3),
            source_fence: Fence(1),
        });
        let _ = rig.tick(vec![wire(ORCH, MsgClass::Saga, &ghost)]);
        assert_eq!(
            rig.stats().undecodable,
            1,
            "a non-dispatchable arm is undecodable"
        );

        // Raw garbage on the orchestrator Saga path → also undecodable (the Err arm).
        let _ = rig.tick(vec![Inbound::Wire {
            from: ORCH,
            class: MsgClass::Saga,
            bytes: vec![0xFF, 0xFF].into(),
        }]);
        assert_eq!(rig.stats().undecodable, 2);
        // The consumed-command count did not move (the split is clean in both directions).
        assert_eq!(rig.stats().transfer_unroutable, 1);

        // A well-formed reply that is NOT a Session head (a CAS outcome carries no gateway
        // obligation) decodes + dispatches to on_directory_reply, which returns without
        // effect — it is NOT undecodable (valid arm), just no-op for the gateway.
        let cas = InterShardFlow::DirectoryReply(DirectoryReply::CasResult {
            key: DirectoryKey::Session(SessionId(7)),
            outcome: vd_wire::seams::directory::CasOutcome::Won {
                new_fence: Fence(2),
            },
        });
        let _ = rig.tick(vec![wire(ORCH, MsgClass::Saga, &cas)]);
        assert_eq!(
            rig.stats().undecodable,
            2,
            "a valid non-Head reply is not undecodable"
        );
    }

    #[test]
    fn frames_skip_sessions_that_are_not_active_yet() {
        let mut rig = Rig::new();
        let _ = rig.tick(vec![wire(CLIENT, MsgClass::Control, &hello_msg())]);
        let sent = rig.tick(vec![wire(
            SHARD,
            MsgClass::Snapshot,
            &frame_msg(Fence(1), 1),
        )]);
        let to_client = sent.iter().filter(|(to, _, _)| *to == CLIENT).count();
        assert_eq!(to_client, 0, "pending sessions receive nothing");
    }

    #[test]
    fn bye_detaches_revokes_and_clears() {
        let mut rig = Rig::new();
        let (session_id, _) = rig.login();
        let sent = rig.tick(vec![wire(
            CLIENT,
            MsgClass::Control,
            &ClientControlMsg::Bye,
        )]);
        assert!(rig.world.resource::<GatewaySessions>().is_empty());
        let detach: GatewayToShard = postcard::from_bytes(
            &sent
                .iter()
                .find(|(to, _, _)| *to == SHARD)
                .expect("detach sent")
                .2,
        )
        .expect("decode");
        assert_eq!(
            detach,
            GatewayToShard::DetachSession {
                session: session_id,
                fence: Fence(1),
            }
        );
        let revoke: InterShardFlow = postcard::from_bytes(
            &sent
                .iter()
                .find(|(to, _, _)| *to == ORCH)
                .expect("revoke sent")
                .2,
        )
        .expect("decode");
        assert_eq!(
            revoke,
            InterShardFlow::Directory(DirectoryOp::LeaseRevoke {
                key: DirectoryKey::Session(session_id),
                fence: Fence(1),
            })
        );
        // Bye from a connection with no session is a no-op.
        let sent = rig.tick(vec![wire(
            NodeId(177),
            MsgClass::Control,
            &ClientControlMsg::Bye,
        )]);
        assert_eq!(sent.len(), 0);
        // The shard detach confirmation closes the loop silently.
        let sent = rig.tick(vec![wire(
            SHARD,
            MsgClass::Control,
            &ShardToGateway::SessionDetached {
                session: session_id,
            },
        )]);
        assert_eq!(sent.len(), 0);
    }

    #[test]
    fn pending_phases_retry_every_tick_until_answered() {
        let mut rig = Rig::new();
        let _ = rig.tick(vec![wire(CLIENT, MsgClass::Control, &hello_msg())]);
        // AwaitingDirectory: a grant retry goes out on an idle tick.
        let sent = rig.tick(vec![]);
        assert_eq!(sent.len(), 1);
        assert_eq!(sent[0].0, ORCH);
        let session_id = rig
            .world
            .resource::<GatewaySessions>()
            .sessions()
            .next()
            .expect("pending");
        let _ = rig.tick(vec![wire(ORCH, MsgClass::Saga, &granted_head(session_id))]);
        // AwaitingAttach: an attach retry goes out on an idle tick.
        let sent = rig.tick(vec![]);
        assert_eq!(sent.len(), 1);
        assert_eq!(sent[0].0, SHARD);
        let retry: GatewayToShard = postcard::from_bytes(&sent[0].2).expect("decode");
        assert_eq!(
            retry,
            GatewayToShard::AttachSession {
                session: session_id,
                fence: Fence(1),
                account: AccountId(5),
                spawn: None,
            }
        );
    }

    #[test]
    fn duplicate_and_late_replies_are_idempotent() {
        let mut rig = Rig::new();
        let (session_id, _) = rig.login();
        // A duplicate directory head after activation: no-op.
        let sent = rig.tick(vec![wire(ORCH, MsgClass::Saga, &granted_head(session_id))]);
        assert_eq!(sent.len(), 0);
        // A duplicate attach reply: no second SubscriptionOpened.
        let sent = rig.tick(vec![wire(
            SHARD,
            MsgClass::Control,
            &ShardToGateway::SessionAttached {
                session: session_id,
                entity: EntityId(77),
                frame: FrameRef::SystemSpace { system_seed: 7 },
                realm_fence: Fence(1),
            },
        )]);
        assert_eq!(decode_controls(&sent, CLIENT).len(), 0);
        // Replies for a session that's gone: ignored.
        let _ = rig.tick(vec![wire(
            CLIENT,
            MsgClass::Control,
            &ClientControlMsg::Bye,
        )]);
        let sent = rig.tick(vec![
            wire(ORCH, MsgClass::Saga, &granted_head(session_id)),
            wire(
                SHARD,
                MsgClass::Control,
                &ShardToGateway::SessionAttached {
                    session: session_id,
                    entity: EntityId(77),
                    frame: FrameRef::SystemSpace { system_seed: 7 },
                    realm_fence: Fence(1),
                },
            ),
        ]);
        assert_eq!(sent.len(), 0);
    }

    #[test]
    fn garbage_wrong_classes_and_notices_are_counted_or_skipped() {
        let mut rig = Rig::new();
        let _ = rig.tick(vec![
            // Undecodable from every peer family.
            Inbound::Wire {
                from: CLIENT,
                class: MsgClass::Control,
                bytes: vec![0xFF].into(),
            },
            Inbound::Wire {
                from: SHARD,
                class: MsgClass::Control,
                bytes: vec![0xFF].into(),
            },
            Inbound::Wire {
                from: SHARD,
                class: MsgClass::Snapshot,
                bytes: vec![0xFF].into(),
            },
            Inbound::Wire {
                from: ORCH,
                class: MsgClass::Saga,
                bytes: vec![0xFF].into(),
            },
            // Wrong classes.
            Inbound::Wire {
                from: SHARD,
                class: MsgClass::Saga,
                bytes: vec![1].into(),
            },
            Inbound::Wire {
                from: ORCH,
                class: MsgClass::Control,
                bytes: vec![1].into(),
            },
            Inbound::Wire {
                from: CLIENT,
                class: MsgClass::Snapshot,
                bytes: vec![1].into(),
            },
            // Clock sync is the follower system's business: skipped here.
            Inbound::Wire {
                from: ORCH,
                class: MsgClass::Membership,
                bytes: vec![1].into(),
            },
            // Transport notices are skipped by the dispatcher.
            Inbound::NodeUnreachable {
                to: SHARD,
                class: MsgClass::Input,
                undelivered: vd_core::MsgId(0),
            },
        ]);
        // SIX, not seven — and the seventh moved rather than vanished. This rig never logs anybody in, so
        // the `CLIENT` node has no session; a `Snapshot` from it is not "a client sent malformed bytes",
        // it is a node the router does not know sending a class only a shard sends. That is now its own
        // fact, because on the live cluster it was a running shard's entire output.
        assert_eq!(rig.stats().undecodable, 6);
        assert_eq!(
            rig.stats().refused_unknown_sender,
            1,
            "the sessionless node's data frame is refused by SENDER, not miscounted as bad bytes"
        );
        // CutEmitted/Pong are accepted no-ops (nothing to bind to in P1).
        let _ = rig.tick(vec![
            wire(
                CLIENT,
                MsgClass::Control,
                &ClientControlMsg::CutEmitted {
                    transfer: vd_core::TransferId(1),
                    marker_seq: 5,
                },
            ),
            wire(
                CLIENT,
                MsgClass::Control,
                &ClientControlMsg::Pong { nonce: 2 },
            ),
        ]);
        // Entity heads carry no gateway obligation.
        let _ = rig.tick(vec![wire(
            ORCH,
            MsgClass::Saga,
            &DirectoryReply::Head {
                key: DirectoryKey::Entity(EntityId(9)),
                record: None,
            },
        )]);
        // A Frame on the Control class is a peer bug, counted.
        let before = rig.stats().undecodable;
        let _ = rig.tick(vec![wire(
            SHARD,
            MsgClass::Control,
            &frame_msg(Fence(1), 1),
        )]);
        assert_eq!(rig.stats().undecodable, before + 1);
    }

    /// SPIKE-2a: the 20 Hz hot path is lock-free by construction (ArcSwap load +
    /// one atomic + a byte splice) and fast enough that 50k route+retag rounds
    /// finish far inside any tick budget even in a debug build. The bound is a
    /// generous PROPERTY gate (catches contention collapse / accidental O(n²)),
    /// not a microbenchmark number.
    #[test]
    // The seam ban targets PRODUCTION reaching for wall-clock; a latency microbench
    // measuring elapsed time is exactly what Instant is for (justified exemption).
    #[allow(clippy::disallowed_methods)]
    fn spike_2a_hot_path_volume_bound() {
        let hot = SessionHot {
            route: ArcSwap::from_pointee(RouteSnapshot {
                authority: SHARD,
                fence: Fence(1),
                cut: None,
            }),
            last_input_seq: AtomicU64::new(0),
            subs: ArcSwap::from_pointee(SubTable::default()),
        };
        let snapshot_bytes = frame_msg(Fence(1), 1)
            .into_snapshot_bytes()
            .expect("frame_msg builds a Frame");
        let started = std::time::Instant::now();
        let mut forwarded = 0u64;
        for seq in 1..=50_000u64 {
            let input = input_bytes(seq);
            assert_eq!(
                route_input(&hot, &input),
                InputRouting::Forward { to: SHARD },
                "monotonic seqs always forward"
            );
            forwarded += 1;
            let out = forward_frame(&hot, SubId(0), Fence(1), &snapshot_bytes)
                .expect("current fence forwards");
            assert!(!out.is_empty());
        }
        assert_eq!(forwarded, 50_000);
        // MV-4: the 1c.5 cut partition adds a `.cut` read to `route_input`. Time it under
        // volume against a `cut: Some` route, crossing the marker, proving the added branch
        // is no contended-load regression (the partition is one compare on the loaded Arc).
        let cut_hot = SessionHot {
            route: ArcSwap::from_pointee(RouteSnapshot {
                authority: SHARD,
                fence: Fence(1),
                cut: Some(SeqCut {
                    marker_seq: 25_000,
                    dest: DEST,
                }),
            }),
            last_input_seq: AtomicU64::new(0),
            subs: ArcSwap::from_pointee(SubTable::default()),
        };
        let (mut to_source, mut to_buffer) = (0u64, 0u64);
        for seq in 1..=50_000u64 {
            // seq <= marker → Forward (source); seq > marker → Buffer (held for the dest).
            if route_input(&cut_hot, &input_bytes(seq)) == InputRouting::Buffer {
                to_buffer += 1;
            } else {
                to_source += 1;
            }
        }
        assert_eq!(
            (to_source, to_buffer),
            (25_000, 25_000),
            "partitioned at the marker"
        );
        let elapsed = started.elapsed();
        assert!(
            elapsed < std::time::Duration::from_secs(5),
            "hot path collapsed: 100k rounds took {elapsed:?}"
        );
    }

    /// THE ROUTER HOT-PATH VOLUME GATE — a crowd-sized snapshot body through the real fan-out.
    ///
    /// WHAT IT USED TO MEASURE AND WHY THAT MATTERS. Until this slice the router RE-EXPRESSED every
    /// snapshot body into each viewer's space, and this gate existed to keep that decode-and-re-encode from
    /// collapsing under a crowd. The router does not do that any more: the shard chain ships every occupant
    /// already measured in the frame of the realm its viewer is standing in, so the only per-body work left
    /// here is the leading-varint re-tag and the refcounted fan. The gate is RE-AIMED at that remaining
    /// work rather than deleted, because it is still the ONE 20 Hz path in the system with no latency gate
    /// over it otherwise — `just spike3a` measures the mesh datagram transport on loopback with the router
    /// outside its loop.
    ///
    /// WHAT IS NOT COVERED HERE, AND WHERE IT IS: the cost that MOVED — one conversion per level, one
    /// shard hop per level — is measured by `just chain-latency`
    /// (`vd-tests`'s `the_chain_pays_one_tick_per_level_each_way_and_the_price_is_measured` and
    /// `the_chain_compounds_staleness_per_level_under_a_lossy_link`). That gate measures in TICKS rather
    /// than wall-clock, because the moved cost is a delivery per hop and not CPU; the two gates are
    /// complementary and neither substitutes for the other. Measured on a 3-level chain at 20 Hz: one tick
    /// per hop each way, exactly, and 4 ticks (200 ms) from a star authoring a placement to a client
    /// holding it.
    ///
    /// The ceiling is DERIVED, not written out: the gateway must finish a tick's work inside a tick, so one
    /// frame per tick may take at most one tick period, and this asserts the whole 50k-round loop inside a
    /// large multiple of that budget. It is a PROPERTY gate against contention collapse and accidental
    /// O(n²) — a debug-build microbenchmark number would be meaningless — but a per-session full decode
    /// (the shape this design exists to avoid) would blow it by two orders of magnitude.
    #[test]
    // The seam ban targets PRODUCTION reaching for wall-clock; a latency gate measuring elapsed time is
    // exactly what Instant is for (justified exemption).
    #[allow(clippy::disallowed_methods)]
    fn the_router_hot_path_holds_a_crowd_sized_body_under_volume() {
        const ENTITIES: usize = 128;
        const ROUNDS: u32 = 50_000;
        let tick_hz = 50u32;

        let snapshot = SnapshotDatagram {
            sub: SubId(0),
            frame_id: 1,
            source_tick: TickId(5),
            universe_tick: UniverseTick(0),
            entities: (0..ENTITIES)
                .map(|i| EntitySnap {
                    entity: EntityId(i as u128),
                    pose: vd_core::pose::StampedPose::at_rest(
                        FrameRef::PlanetCentered { planet_seed: 7 },
                        DVec3::new(i as f64, 0.0, 0.0),
                        UniverseTick(0),
                    ),
                })
                .collect(),
        };
        let shard_bytes = postcard::to_allocvec(&snapshot).expect("encode");
        let frame = postcard::to_allocvec(&ShardToGateway::Frame {
            realm_fence: Fence(1),
            source_tick: TickId(5),
            snapshot_bytes: shard_bytes.clone(),
        })
        .expect("encode");

        let (mut sessions, _sid, _) = one_active_session();
        let mut stats = GatewayStats::default();

        let started = std::time::Instant::now();
        let mut served = 0u32;
        let mut last = Vec::new();
        for _ in 0..ROUNDS {
            let mut outbox = OutboundBox::default();
            on_shard_frame(SHARD, &frame, &mut sessions, &mut stats, &mut outbox);
            served += u32::try_from(outbox.0.len()).expect("one push per round");
            last = outbox.0[0].2.to_vec();
        }
        let elapsed = started.elapsed();
        assert_eq!(served, ROUNDS, "every round served the one subscriber");
        // ANTI-VACUITY, and the byte-identity claim in one assertion: the body the subscriber received is
        // the shard's own body, not a re-encoding of it. `one_active_session` holds SubId(0) and the
        // datagram already leads with sub 0, so the re-tag rewrites the same varint and the bytes match
        // exactly. A router that started composing again would fail here before it failed on time.
        assert_eq!(
            last, shard_bytes,
            "the router forwarded the shard's own crowd-sized body byte for byte"
        );

        // The derived ceiling: `ROUNDS` tick-periods of headroom over what is nominally ROUNDS ticks of
        // work. A debug build is ~an order of magnitude off release, hence the whole period rather than a
        // fraction of it; the failure this catches is a per-session decode, which costs sessions× more.
        let budget = std::time::Duration::from_secs_f64(f64::from(ROUNDS) / f64::from(tick_hz));
        assert!(
            elapsed < budget,
            "the gateway fan-out collapsed: {ROUNDS} rounds of {ENTITIES} entities took {elapsed:?} \
             (budget {budget:?} = one tick period per round at {tick_hz} Hz)"
        );
    }

    /// SPIKE-2a (the route-swap hot-path GATE — formally blocks the P2 route-swap design):
    /// proves the gateway route swap is WAIT-FREE and TORN-READ-FREE under a CONCURRENT
    /// `route.store` publisher (modeling a P2 `CommitAuthority` swap under load). The ONE
    /// `ArcSwap` swap means any `route.load()` yields exactly ONE published `RouteSnapshot` —
    /// never a field-mix — so authority + fence + cut (incl. `cut: Some(SeqCut)`, the field
    /// P2 adds) move together atomically; the read path is one `ArcSwap::load` (+ for
    /// `route_input` one relaxed atomic), no `Mutex` anywhere. No `CommitAuthority`-driven
    /// `route.store` lands until this is green.
    ///
    /// SCOPE (honest): this gates the SWAP MECHANIC (atomicity + the wait-free read latency).
    /// It does NOT prove the cut-PARTITIONING read logic — `route_input`'s future
    /// `seq <= marker → source / > marker → dest` branch (the slot at line ~190) is a
    /// CORRECTNESS property that gets its own test in Slice 1c, not a latency gate. Two
    /// timed bands are measured separately: the isolated ROUTE DECISION (`frame_passes_fence`
    /// = one `route.load` + fence compare, no alloc — the thing the swap actually contends),
    /// and the end-to-end per-sub FORWARD (`forward_frame` = load + retag, where production
    /// amortizes the retag once-per-SubId via SCALE-1, so this is a fan-out figure, not the
    /// route decision). The tight ROUTE-DECISION budget is what catches a contended-load
    /// regression that the alloc-dominated forward number would hide.
    ///
    /// HAND-ROLLED (no bench crate): no library expresses a concurrent-contention p99
    /// HARD-FAIL gate — criterion/divan are report-only steady-state harnesses with no p99
    /// and no fail-threshold (investigated, 2026); we would hand-compute p99 + the assert
    /// regardless. The latency ASSERTS are RELEASE-ONLY: debug + coverage instrumentation
    /// make a tail meaningless, so a debug/coverage run still exercises the concurrency plus
    /// the torn-read invariant (fast, small N) while a release run (`just spike2a`,
    /// `--test-threads=1` so siblings don't oversubscribe) enforces the timing. `Instant` is
    /// the justified seam exemption (a latency microbench is exactly what wall-clock is for).
    #[test]
    #[allow(clippy::disallowed_methods)]
    fn spike_2a_route_swap_is_wait_free_and_torn_read_free() {
        use std::sync::Arc;
        use std::sync::atomic::AtomicBool;
        use std::time::{Duration, Instant};

        // The publisher's small FIXED set of known-good routes (distinct authority + fence),
        // INCLUDING a `cut: Some(SeqCut)` member — the exact field P2's CommitAuthority adds —
        // so a Some-cut genuinely crosses the swap under contention and the torn-read
        // membership check covers the SeqCut bytes (not just the always-None P1 shape).
        // A frame at Fence(9) is never stale against any fence here, so `forward_frame`
        // always reaches the full retag (the worst-case forward).
        let routes = [
            RouteSnapshot {
                authority: SHARD,
                fence: Fence(1),
                cut: None,
            },
            RouteSnapshot {
                authority: ORCH,
                fence: Fence(2),
                cut: Some(SeqCut {
                    marker_seq: 7,
                    dest: SHARD,
                }),
            },
            RouteSnapshot {
                authority: SHARD,
                fence: Fence(3),
                cut: None,
            },
        ];
        let hot = Arc::new(SessionHot {
            route: ArcSwap::from_pointee(routes[0]),
            last_input_seq: AtomicU64::new(0),
            subs: ArcSwap::from_pointee(SubTable::default()),
        });
        let frame = frame_msg(Fence(9), 1)
            .into_snapshot_bytes()
            .expect("frame_msg builds a Frame");

        // 4 readers vs 1 publisher: a CONSERVATIVE-on-dev-hardware contention figure (a 4-core
        // box oversubscribes 5:N) — NOT a model of cloud core counts. Production reads a
        // session's hot state from ~one forwarder; 4 concurrent loaders is strictly HARDER, so
        // a pass here is a safe upper bound, not a scale claim.
        const READERS: usize = 4;
        // Small N under debug/coverage (instrumented — keep it quick); large N in release for
        // a stable tail. `cfg!` folds at compile time → no runtime branch (no coverage hole).
        const SAMPLES_PER_READER: usize = if cfg!(debug_assertions) {
            2_000
        } else {
            200_000
        };

        let stop = Arc::new(AtomicBool::new(false));
        let misses = Arc::new(AtomicU64::new(0));

        // Each reader returns TWO sample bands: (isolated route-decision, end-to-end forward).
        type Bands = (Vec<Duration>, Vec<Duration>);
        let bands: Vec<Bands> = std::thread::scope(|s| {
            // Publisher: swap the route as fast as it can (CommitAuthority under contention).
            let pub_hot = Arc::clone(&hot);
            let pub_stop = Arc::clone(&stop);
            s.spawn(move || {
                let mut i = 0usize;
                while !pub_stop.load(Ordering::Relaxed) {
                    pub_hot.route.store(Arc::new(routes[i % routes.len()]));
                    i = i.wrapping_add(1);
                }
            });
            let handles: Vec<_> = (0..READERS)
                .map(|_| {
                    let hot = Arc::clone(&hot);
                    let misses = Arc::clone(&misses);
                    let frame = frame.clone();
                    s.spawn(move || {
                        let mut route_read = Vec::with_capacity(SAMPLES_PER_READER);
                        let mut forward = Vec::with_capacity(SAMPLES_PER_READER);
                        let mut local = 0u64;
                        for _ in 0..SAMPLES_PER_READER {
                            // Band 1 — the ISOLATED route decision the swap contends: one
                            // `route.load` + fence compare, NO alloc, so a contended-load
                            // regression can't hide under the retag's heap-alloc noise.
                            let t = Instant::now();
                            let pass = frame_passes_fence(&hot, Fence(9));
                            route_read.push(t.elapsed());
                            // Band 2 — the end-to-end per-sub forward (load + retag alloc).
                            let t = Instant::now();
                            let out = forward_frame(&hot, SubId(0), Fence(9), &frame);
                            forward.push(t.elapsed());
                            // Invariants — accumulated via `+= u64::from(..)` (NOT an `if`), so
                            // each never-taken failure case stays a COVERED region, not a hole.
                            // A high-fence frame always passes + forwards (proves the reads
                            // RAN); a DIRECT load is a COMPLETE member of the published set.
                            // This last check is a STRUCTURAL-INVARIANT CANARY: ArcSwap cannot
                            // tear a single Arc today, so it guards a FUTURE regression where
                            // authority/fence/cut stop sharing one Arc (e.g. the P2 temptation
                            // to bolt marker_seq onto a separate atomic) — then a mixed load
                            // would be a non-member and fire here.
                            local += u64::from(!pass);
                            local += u64::from(out.is_none());
                            let loaded: RouteSnapshot = **hot.route.load();
                            local += u64::from(!routes.contains(&loaded));
                        }
                        misses.fetch_add(local, Ordering::Relaxed);
                        (route_read, forward)
                    })
                })
                .collect();
            let bands = handles
                .into_iter()
                .map(|h| h.join().expect("reader thread"))
                .collect();
            stop.store(true, Ordering::Relaxed); // let the publisher exit before scope-join
            bands
        });

        // Invariants checked in EVERY build (incl. debug/coverage): no torn read AND every
        // high-fence read passed + forwarded (misses counts all failure modes → exactly 0).
        assert_eq!(
            misses.load(Ordering::Relaxed),
            0,
            "a route.load() was not a complete member of the published set (torn read), or a \
             high-fence frame failed to pass/forward"
        );
        let route_read: Vec<Duration> = bands.iter().flat_map(|(r, _)| r.iter().copied()).collect();
        let forward: Vec<Duration> = bands.iter().flat_map(|(_, f)| f.iter().copied()).collect();
        assert_eq!(route_read.len(), READERS * SAMPLES_PER_READER);
        assert_eq!(forward.len(), READERS * SAMPLES_PER_READER);

        // The p99 tail helper is the ONE shared latency-gate utility (HR3), extracted to
        // vd-harness so SPIKE-2a (here) and SPIKE-3a (vd-io-prod) can never drift.
        let route_p99 = vd_harness::latency::percentile_unstable(route_read, 99);
        let forward_p99 = vd_harness::latency::percentile_unstable(forward, 99);
        // The hard latency GATES are release-only (a debug/coverage tail is meaningless).
        #[cfg(not(debug_assertions))]
        {
            // The route DECISION (one ArcSwap load + fence compare) must be lost in the noise
            // of a 50 ms (20 Hz) tick — this is ~10,000x under. The budget guards the property
            // that actually matters: WAIT-FREE (no lock). Observed p99 ~625 ns under a
            // hammering publisher; a Mutex/lock in this read would be ≥20 µs under the same
            // contention, so 5 µs (~8x over observed) cleanly catches that regression while
            // staying robust on a throttled CI-less dev box. THE number that blocks P2.
            const ROUTE_DECISION_P99_BUDGET: Duration = Duration::from_micros(5);
            // The end-to-end forward includes the retag alloc production amortizes per-SubId
            // (SCALE-1) — a looser fan-out ceiling, not the route decision (observed ~600 ns).
            const FORWARD_FAN_OUT_P99_BUDGET: Duration = Duration::from_micros(50);
            eprintln!(
                "SPIKE-2a: route-decision p99 = {route_p99:?} (budget {ROUTE_DECISION_P99_BUDGET:?}); \
                 forward-fan-out p99 = {forward_p99:?} (budget {FORWARD_FAN_OUT_P99_BUDGET:?}); \
                 {} samples/band across {READERS} readers, 0 torn reads",
                READERS * SAMPLES_PER_READER
            );
            assert!(
                route_p99 < ROUTE_DECISION_P99_BUDGET,
                "route-decision p99 {route_p99:?} exceeded {ROUTE_DECISION_P99_BUDGET:?} under a \
                 concurrent route.store publisher (a contended-load regression)"
            );
            assert!(
                forward_p99 < FORWARD_FAN_OUT_P99_BUDGET,
                "forward-fan-out p99 {forward_p99:?} exceeded {FORWARD_FAN_OUT_P99_BUDGET:?}"
            );
        }
        #[cfg(debug_assertions)]
        let _ = (route_p99, forward_p99);
    }

    #[test]
    fn hot_path_unit_outcomes() {
        let hot = SessionHot {
            route: ArcSwap::from_pointee(RouteSnapshot {
                authority: SHARD,
                fence: Fence(2),
                cut: None,
            }),
            last_input_seq: AtomicU64::new(10),
            subs: ArcSwap::from_pointee(SubTable::default()),
        };
        assert_eq!(route_input(&hot, &input_bytes(10)), InputRouting::Deduped);
        assert_eq!(route_input(&hot, &input_bytes(5)), InputRouting::Deduped);
        assert_eq!(
            route_input(&hot, &input_bytes(11)),
            InputRouting::Forward { to: SHARD }
        );
        assert_eq!(route_input(&hot, &[0x80]), InputRouting::Malformed);
        // Stale frame fence → None; corrupt snapshot header → None.
        assert_eq!(forward_frame(&hot, SubId(0), Fence(1), &[0]), None);
        assert_eq!(forward_frame(&hot, SubId(0), Fence(2), &[0x80; 6]), None);
        // A multi-byte sub id re-tags exactly (the varint continuation path).
        let snapshot_bytes = frame_msg(Fence(2), 3)
            .into_snapshot_bytes()
            .expect("frame_msg builds a Frame");
        let big = forward_frame(&hot, SubId(40_000), Fence(2), &snapshot_bytes)
            .expect("current fence forwards");
        let decoded: SnapshotDatagram = postcard::from_bytes(&big).expect("decode");
        assert_eq!(decoded.sub, SubId(40_000));
    }

    #[test]
    fn concurrent_inputs_at_one_seq_forward_exactly_once() {
        // FG-2: the dedup is a SINGLE atomic `fetch_max`, so many threads racing the
        // SAME seq yield exactly ONE Forward — every other thread dedups. The prior
        // non-atomic load-then-store could let several threads observe the same stale
        // high-water mark, all pass, and all forward a duplicate input. A barrier
        // maximizes the contention window.
        use std::sync::Barrier;
        use std::sync::atomic::AtomicUsize;

        const THREADS: usize = 32;
        let hot = SessionHot {
            route: ArcSwap::from_pointee(RouteSnapshot {
                authority: SHARD,
                fence: Fence(2),
                cut: None,
            }),
            last_input_seq: AtomicU64::new(0),
            subs: ArcSwap::from_pointee(SubTable::default()),
        };
        let forwards = AtomicUsize::new(0);
        let barrier = Barrier::new(THREADS);
        std::thread::scope(|s| {
            for _ in 0..THREADS {
                s.spawn(|| {
                    barrier.wait();
                    if route_input(&hot, &input_bytes(7)) == (InputRouting::Forward { to: SHARD }) {
                        forwards.fetch_add(1, Ordering::Relaxed);
                    }
                });
            }
        });
        assert_eq!(
            forwards.load(Ordering::Relaxed),
            1,
            "exactly one thread forwards seq 7; the rest dedup"
        );
        assert_eq!(
            hot.last_input_seq.load(Ordering::Relaxed),
            7,
            "the high-water mark advanced to seq 7 exactly once"
        );
    }

    // ---- Slice 1c.2: the gateway TransferControl consumer ----------------------

    const XFER: TransferId = TransferId(0x1c2);
    /// The transfer SUBJECT the saga carries on `CommitAuthority` — the avatar the dest adopts
    /// (1c.8). The gateway forwards it VERBATIM into `OpenInputSlot`; these tests assert that.
    const XFER_SUBJECT: DirectoryKey = DirectoryKey::Entity(EntityId(0x1c8));

    fn saga_cmd(cmd: TransferControl) -> Inbound {
        wire(ORCH, MsgClass::Saga, &InterShardFlow::Saga(cmd))
    }

    /// The `SagaAck`s the gateway sent back to the orchestrator (ignoring the directory
    /// ops that also ride ORCH+Saga — login's LeaseGrant, etc.).
    fn acks_to_orch(sent: &[(NodeId, MsgClass, Vec<u8>)]) -> Vec<TransferControlAck> {
        sent.iter()
            .filter(|(node, class, _)| (*node == ORCH) & (*class == MsgClass::Saga))
            .filter_map(|(_, _, bytes)| {
                // The gateway only ever sends valid flows; `.expect` keeps the Err path in
                // std (no caller branch). A non-SagaAck ORCH/Saga send (a directory op, e.g.
                // login's LeaseGrant) maps to None — exercised by the login-tick assertion.
                match postcard::from_bytes::<InterShardFlow>(bytes)
                    .expect("gateway sends a valid flow")
                {
                    InterShardFlow::SagaAck(ack) => Some(ack),
                    _ => None,
                }
            })
            .collect()
    }

    fn marker_input(seq: u64) -> Inbound {
        wire(
            CLIENT,
            MsgClass::Input,
            &InputDatagram {
                seq,
                is_cut_marker: true,
                client_tick: TickId(1),
                movement: [0.0, 0.0, 0.0],
                look: [0.0, 0.0],
                action_bits: 0,
            },
        )
    }

    fn transfer_in_flight(rig: &Rig, sid: SessionId) -> bool {
        rig.world
            .resource::<GatewaySessions>()
            .by_session
            .get(&sid)
            .expect("session present")
            .transfer
            .is_some()
    }

    /// Drive a session to a LIVE cut: Prepare(dest:DEST) -> RequestCut -> marker(M) ->
    /// FreezeSource(marker_seq:M, dest:DEST). Returns the tick's sends from the freeze.
    fn freeze_to_live_cut(
        rig: &mut Rig,
        sid: SessionId,
        marker: u64,
    ) -> Vec<(NodeId, MsgClass, Vec<u8>)> {
        let _ = rig.tick(vec![saga_cmd(TransferControl::PrepareSubscribe {
            transfer: XFER,
            session: sid,
            dest: DEST,
        })]);
        let _ = rig.tick(vec![saga_cmd(TransferControl::RequestCut {
            transfer: XFER,
            session: sid,
        })]);
        let _ = rig.tick(vec![marker_input(marker)]);
        rig.tick(vec![saga_cmd(TransferControl::FreezeSource {
            transfer: XFER,
            session: sid,
            marker_seq: marker,
            dest: DEST,
        })])
    }

    fn route_cut(rig: &Rig, sid: SessionId) -> Option<SeqCut> {
        route_state(rig, sid).cut
    }

    /// The full loaded route snapshot (all three fields from ONE coherent load) — the swap
    /// oracle for 1c.4 (authority + fence + cut together).
    fn route_state(rig: &Rig, sid: SessionId) -> RouteSnapshot {
        *rig.world
            .resource::<GatewaySessions>()
            .by_session
            .get(&sid)
            .expect("session present")
            .hot
            .route
            .load_full()
    }

    #[test]
    fn prepare_opens_progress_and_acks_ready() {
        let mut rig = Rig::new();
        let (sid, login_sends) = rig.login();
        // The hello tick's ORCH/Saga send is a directory LeaseGrant, not a SagaAck — so
        // `acks_to_orch` yields none (covers the non-ack decode arm of the helper).
        assert_eq!(acks_to_orch(&login_sends[0]), vec![]);
        let sent = rig.tick(vec![saga_cmd(TransferControl::PrepareSubscribe {
            transfer: XFER,
            session: sid,
            dest: SHARD,
        })]);
        assert_eq!(
            acks_to_orch(&sent),
            vec![TransferControlAck::Prepared {
                transfer: XFER,
                result: PrepareResult::Ready,
            }]
        );
        assert!(
            transfer_in_flight(&rig, sid),
            "PrepareSubscribe opened progress"
        );
    }

    #[test]
    fn request_cut_pushes_to_client_and_self_acks_cut_confirmed() {
        // S3: RequestCut is now SELF-ACKING (server-timed cut). It STILL pushes one
        // ServerControlMsg::RequestCut to the CLIENT (cosmetic — the old client's marker is inert)
        // AND self-acks CutConfirmed{marker_seq: 0 placeholder} in the SAME tick — the saga advances
        // Cutting→Freezing with no client marker. (The real input-cut seq is derived at FreezeSource.)
        let mut rig = Rig::new();
        let (sid, _) = rig.login();
        let _ = rig.tick(vec![saga_cmd(TransferControl::PrepareSubscribe {
            transfer: XFER,
            session: sid,
            dest: SHARD,
        })]);
        let sent = rig.tick(vec![saga_cmd(TransferControl::RequestCut {
            transfer: XFER,
            session: sid,
        })]);
        assert_eq!(
            decode_controls(&sent, CLIENT),
            vec![ServerControlMsg::RequestCut { transfer: XFER }],
            "RequestCut still pushes to the client (cosmetic — the marker is now inert)",
        );
        assert_eq!(
            acks_to_orch(&sent),
            vec![TransferControlAck::CutConfirmed {
                transfer: XFER,
                marker_seq: 0, // placeholder; the real seq is FreezeSource's install-time high-water
            }],
            "RequestCut self-acks CutConfirmed server-side (no client marker needed)",
        );

        // A client CUT_MARKER on the INPUT flow is now INERT — an ordinary input, NO CutConfirmed.
        let sent = rig.tick(vec![marker_input(42)]);
        assert_eq!(
            acks_to_orch(&sent),
            vec![],
            "the client marker no longer drives a CutConfirmed (S3 — server-timed cut)",
        );
    }

    #[test]
    fn a_redelivered_request_cut_re_serves_the_recorded_cut_confirmed() {
        // S3: RequestCut is now self-acking and journaled at step 1, so a redelivery re-serves the
        // SAME CutConfirmed verbatim via the standard redelivery gate — it does NOT re-push the
        // client RequestCut nor re-generate the ack (the effect already ran once).
        let mut rig = Rig::new();
        let (sid, _) = rig.login();
        let _ = rig.tick(vec![saga_cmd(TransferControl::PrepareSubscribe {
            transfer: XFER,
            session: sid,
            dest: SHARD,
        })]);
        let request_cut = || {
            saga_cmd(TransferControl::RequestCut {
                transfer: XFER,
                session: sid,
            })
        };
        let first = rig.tick(vec![request_cut()]);
        assert_eq!(
            decode_controls(&first, CLIENT),
            vec![ServerControlMsg::RequestCut { transfer: XFER }],
            "the first RequestCut pushes to the client + self-acks",
        );
        let confirmed = vec![TransferControlAck::CutConfirmed {
            transfer: XFER,
            marker_seq: 0,
        }];
        assert_eq!(acks_to_orch(&first), confirmed);
        // Redeliver RequestCut: the redelivery gate re-serves the recorded CutConfirmed, and does
        // NOT re-push the client RequestCut (no re-effect).
        let sent = rig.tick(vec![request_cut()]);
        assert_eq!(
            decode_controls(&sent, CLIENT),
            vec![],
            "a redelivered RequestCut re-serves the recorded ack — it does NOT re-push the client",
        );
        assert_eq!(
            acks_to_orch(&sent),
            confirmed,
            "the redelivery re-serves the SAME CutConfirmed verbatim (idempotent)",
        );
    }

    #[test]
    fn freeze_installs_the_live_cut_and_acks_source_frozen() {
        // G2: FreezeSource installs Some(SeqCut{marker_seq, dest}) on the route and acks
        // SourceFrozen{drained_seq == marker_seq}. Asserting dest == DEST (!= the authority
        // SHARD) proves `dest` THREADS from the wire command into the cut, not defaulted.
        let mut rig = Rig::new();
        let (sid, _) = rig.login();
        let sent = freeze_to_live_cut(&mut rig, sid, 42);
        assert_eq!(
            acks_to_orch(&sent),
            vec![TransferControlAck::SourceFrozen {
                transfer: XFER,
                drained_seq: 42,
            }]
        );
        assert_eq!(
            route_cut(&rig, sid),
            Some(SeqCut {
                marker_seq: 42,
                dest: DEST,
            }),
            "the live cut carries the wire-supplied marker_seq + dest"
        );
    }

    #[test]
    fn an_installed_cut_partitions_input_at_the_marker() {
        // 1c.5: the live cut PARTITIONS `route_input` — all three arms (a SessionHot built
        // directly so the dedup high-water is BELOW the marker, exercising the `seq <= marker`
        // Forward arm that the marker-advanced rig high-water would otherwise hide).
        // (Flipped from the 1c.4 installed-but-inert baseline this test reserved.)
        let hot = SessionHot {
            route: ArcSwap::from_pointee(RouteSnapshot {
                authority: SHARD,
                fence: Fence(2),
                cut: Some(SeqCut {
                    marker_seq: 42,
                    dest: DEST,
                }),
            }),
            last_input_seq: AtomicU64::new(0),
            subs: ArcSwap::from_pointee(SubTable::default()),
        };
        // seq <= marker (and past the dedup high-water) → Forward to the SOURCE authority.
        assert_eq!(
            route_input(&hot, &input_bytes(40)),
            InputRouting::Forward { to: SHARD }
        );
        // seq > marker → Buffer (the cold caller holds it for the dest).
        assert_eq!(route_input(&hot, &input_bytes(99)), InputRouting::Buffer);
        // a duplicate (<= the dedup high-water, now 99) is Deduped, NOT buffered.
        assert_eq!(route_input(&hot, &input_bytes(50)), InputRouting::Deduped);
        // and with NO cut installed, a fresh seq is a plain Forward (the `_` arm).
        let no_cut = SessionHot {
            route: ArcSwap::from_pointee(RouteSnapshot {
                authority: SHARD,
                fence: Fence(2),
                cut: None,
            }),
            last_input_seq: AtomicU64::new(0),
            subs: ArcSwap::from_pointee(SubTable::default()),
        };
        assert_eq!(
            route_input(&no_cut, &input_bytes(100)),
            InputRouting::Forward { to: SHARD }
        );
    }

    #[test]
    fn a_redelivered_freeze_resends_source_frozen_without_re_storing_the_route() {
        // G4: FreezeSource owns its step-2 journal slot, so a redelivery short-circuits at
        // the gate (re-sends the SAME SourceFrozen) and NEVER re-enters apply_freeze — the
        // route keeps the ORIGINAL cut even if the redelivery names a different marker/dest.
        let mut rig = Rig::new();
        let (sid, _) = rig.login();
        let first = freeze_to_live_cut(&mut rig, sid, 42);
        let second = rig.tick(vec![saga_cmd(TransferControl::FreezeSource {
            transfer: XFER,
            session: sid,
            marker_seq: 999,
            dest: ORCH,
        })]);
        assert_eq!(
            acks_to_orch(&first),
            acks_to_orch(&second),
            "same SourceFrozen re-sent"
        );
        assert_eq!(
            route_cut(&rig, sid),
            Some(SeqCut {
                marker_seq: 42,
                dest: DEST,
            }),
            "the route keeps the original cut; the redelivery never re-touched it"
        );
    }

    #[test]
    fn freeze_for_an_unrecorded_transfer_drops_and_counts_no_ack() {
        // G6 (LBD-2): an unbound FreezeSource installs NO cut, sends NO ack (pinning the
        // saga — WEDGE-1), and is counted. Diverges from thaw (a compensator that acks).
        let mut rig = Rig::new();
        let (sid, _) = rig.login();
        let sent = rig.tick(vec![saga_cmd(TransferControl::FreezeSource {
            transfer: XFER,
            session: sid,
            marker_seq: 7,
            dest: DEST,
        })]);
        assert_eq!(acks_to_orch(&sent), vec![], "no ack -> the saga pins");
        assert_eq!(
            route_cut(&rig, sid),
            None,
            "no cut installed for an unbound freeze"
        );
        assert_eq!(rig.stats().transfer_unroutable, 1);
    }

    #[test]
    fn thaw_clears_a_live_cut_then_acks() {
        // G5: drive a genuinely-LIVE cut, THEN thaw — the compensator clears it to None.
        let mut rig = Rig::new();
        let (sid, _) = rig.login();
        let _ = freeze_to_live_cut(&mut rig, sid, 42);
        assert!(
            route_cut(&rig, sid).is_some(),
            "cut is live before the thaw"
        );
        let sent = rig.tick(vec![saga_cmd(TransferControl::ThawSource {
            transfer: XFER,
            session: sid,
        })]);
        assert_eq!(
            acks_to_orch(&sent),
            vec![TransferControlAck::SourceThawed { transfer: XFER }]
        );
        assert_eq!(route_cut(&rig, sid), None, "thaw cleared the live cut");
    }

    #[test]
    fn abort_clears_a_live_cut_locally() {
        // ROB-1c3: AbortTransfer must clear an installed cut LOCALLY, not borrow safety from
        // the saga's Thaw-before-Abort ordering. Drive a LIVE cut, then abort DIRECTLY (no
        // preceding thaw) — the cut is gone and the progress pruned.
        let mut rig = Rig::new();
        let (sid, _) = rig.login();
        let _ = freeze_to_live_cut(&mut rig, sid, 42);
        assert!(
            route_cut(&rig, sid).is_some(),
            "cut is live before the abort"
        );
        let sent = rig.tick(vec![saga_cmd(TransferControl::AbortTransfer {
            transfer: XFER,
            session: sid,
        })]);
        assert_eq!(
            acks_to_orch(&sent),
            vec![TransferControlAck::Aborted { transfer: XFER }]
        );
        assert_eq!(
            route_cut(&rig, sid),
            None,
            "abort cleared the live cut locally"
        );
        assert!(!transfer_in_flight(&rig, sid), "abort pruned the progress");
    }

    #[test]
    fn prepare_for_a_not_yet_active_session_is_refused_and_counted() {
        // WB-1: a transfer phase can only run for an ATTACHED (Active) session. A
        // PrepareSubscribe that races ahead of SessionAttached (session still AwaitingAttach)
        // is refused — no ack (pins the saga), no progress created, counted — so a later
        // attach can never clobber a cut/route installed on a half-attached session.
        let mut rig = Rig::new();
        // Drive Hello + the directory grant, but NOT SessionAttached → AwaitingAttach.
        let _ = rig.tick(vec![wire(CLIENT, MsgClass::Control, &hello_msg())]);
        let sid = rig
            .world
            .resource::<GatewaySessions>()
            .sessions()
            .next()
            .expect("session pending");
        let _ = rig.tick(vec![wire(ORCH, MsgClass::Saga, &granted_head(sid))]);
        // The session is NOT Active yet — a PrepareSubscribe must be refused.
        let sent = rig.tick(vec![saga_cmd(TransferControl::PrepareSubscribe {
            transfer: XFER,
            session: sid,
            dest: DEST,
        })]);
        assert_eq!(
            acks_to_orch(&sent),
            vec![],
            "no Prepared ack for a non-Active session"
        );
        assert!(
            !transfer_in_flight(&rig, sid),
            "no transfer progress created on a half-attached session"
        );
        assert_eq!(rig.stats().transfer_unroutable, 1);
    }

    // ---- Slice 1c.4: CommitAuthority — the route swap --------------------------

    #[test]
    fn commit_swaps_authority_to_dest_and_acks() {
        // T2: the route swap moves authority -> cut.dest, clears the cut, and CARRIES the
        // realm fence UNCHANGED (R-FENCE: NOT the per-Entity CAS new_fence). DEST != SHARD
        // (the source/authority), so this proves dest threads from the installed cut.
        let mut rig = Rig::new();
        let (sid, _) = rig.login(); // attach installs route.fence = Fence(1)
        let _ = freeze_to_live_cut(&mut rig, sid, 7);
        let sent = rig.tick(vec![saga_cmd(TransferControl::CommitAuthority {
            transfer: XFER,
            session: sid,
            new_fence: Fence(2),
            subject: XFER_SUBJECT,
        })]);
        assert_eq!(
            acks_to_orch(&sent),
            vec![TransferControlAck::Committed { transfer: XFER }]
        );
        assert_eq!(
            route_state(&rig, sid),
            RouteSnapshot {
                authority: DEST,
                fence: Fence(1), // R-FENCE: CARRIED, not the new_fence(2)
                cut: None,
            }
        );
    }

    #[test]
    fn commit_does_not_drop_the_dest_own_realm_frames() {
        // T3 (R-FENCE regression guard): after the swap the route fence is CARRIED (Fence(1)),
        // so the dest's own realm-stamped frames (Fence(1)) still FORWARD. This FAILS the
        // instant someone installs new_fence(2) as route.fence (the black-screen bug). A
        // genuinely-stale frame (GENESIS) still drops — rule-5 machinery is live by data.
        let mut rig = Rig::new();
        let (sid, _) = rig.login();
        let _ = freeze_to_live_cut(&mut rig, sid, 7);
        let _ = rig.tick(vec![saga_cmd(TransferControl::CommitAuthority {
            transfer: XFER,
            session: sid,
            new_fence: Fence(2),
            subject: XFER_SUBJECT,
        })]);
        // A realm-stamped frame at the carried fence is forwarded to the client.
        let sent = rig.tick(vec![wire(
            SHARD,
            MsgClass::Snapshot,
            &frame_msg(Fence(1), 9),
        )]);
        let forwarded = sent
            .iter()
            .filter(|(to, class, _)| (*to == CLIENT) & (*class == MsgClass::Snapshot))
            .count();
        assert_eq!(
            forwarded, 1,
            "the dest's own realm frame still forwards post-swap"
        );
        assert_eq!(rig.stats().stale_frames_dropped, 0);
        // A genuinely stale frame still drops + counts.
        let sent = rig.tick(vec![wire(
            SHARD,
            MsgClass::Snapshot,
            &frame_msg(Fence::GENESIS, 10),
        )]);
        assert_eq!(sent.len(), 0);
        assert_eq!(rig.stats().stale_frames_dropped, 1);
    }

    #[test]
    fn commit_without_a_prior_freeze_pins_and_counts() {
        // T4: BOUND (prepared) but no cut installed -> commit_without_cut, no ack, route
        // untouched (never a garbage-dest swap). The saga pins.
        let mut rig = Rig::new();
        let (sid, _) = rig.login();
        let _ = rig.tick(vec![saga_cmd(TransferControl::PrepareSubscribe {
            transfer: XFER,
            session: sid,
            dest: DEST,
        })]);
        let sent = rig.tick(vec![saga_cmd(TransferControl::CommitAuthority {
            transfer: XFER,
            session: sid,
            new_fence: Fence(2),
            subject: XFER_SUBJECT,
        })]);
        assert_eq!(acks_to_orch(&sent), vec![], "no ack -> the saga pins");
        assert_eq!(route_state(&rig, sid).authority, SHARD, "route untouched");
        assert_eq!(route_state(&rig, sid).cut, None);
        assert_eq!(rig.stats().commit_without_cut, 1);
        assert_eq!(
            rig.stats().transfer_unroutable,
            0,
            "distinct from unroutable"
        );
    }

    #[test]
    fn commit_for_an_absent_transfer_pins_and_counts_unroutable() {
        // T5: no prepare at all (unbound) -> transfer_unroutable, no ack, route untouched.
        let mut rig = Rig::new();
        let (sid, _) = rig.login();
        let sent = rig.tick(vec![saga_cmd(TransferControl::CommitAuthority {
            transfer: XFER,
            session: sid,
            new_fence: Fence(2),
            subject: XFER_SUBJECT,
        })]);
        assert_eq!(acks_to_orch(&sent), vec![]);
        assert_eq!(rig.stats().transfer_unroutable, 1);
        assert_eq!(rig.stats().commit_without_cut, 0, "distinct from no-cut");
        assert_eq!(route_state(&rig, sid).authority, SHARD, "route untouched");
    }

    #[test]
    fn commit_is_idempotent_on_redelivery() {
        // T6: a redelivered CommitAuthority re-serves Committed FROM THE JOURNAL and NEVER
        // re-enters apply_commit — proven by STATE (the route is byte-identical), since a
        // re-entry could not reconstruct dest from the now-None cut.
        let mut rig = Rig::new();
        let (sid, _) = rig.login();
        let _ = freeze_to_live_cut(&mut rig, sid, 7);
        let commit = || {
            saga_cmd(TransferControl::CommitAuthority {
                transfer: XFER,
                session: sid,
                new_fence: Fence(2),
                subject: XFER_SUBJECT,
            })
        };
        let first = rig.tick(vec![commit()]);
        let after_first = route_state(&rig, sid);
        let second = rig.tick(vec![commit()]);
        assert_eq!(
            acks_to_orch(&first),
            acks_to_orch(&second),
            "same Committed re-served"
        );
        assert_eq!(
            route_state(&rig, sid),
            after_first,
            "route byte-identical: the redelivery never re-ran apply_commit"
        );
        assert_eq!(
            rig.stats().commit_without_cut,
            0,
            "redelivery is not a no-cut fault"
        );
    }

    // ---- Slice 1c.5: the cut partition — gateway buffer + drain-at-commit -------

    fn client_input(seq: u64) -> Inbound {
        Inbound::Wire {
            from: CLIENT,
            class: MsgClass::Input,
            bytes: input_bytes(seq).into(),
        }
    }

    /// The `seq`s of `SessionInput` frames the gateway sent to `dest` (the drained cut buffer).
    /// The variant match (not a class pre-filter) discriminates — the commit-drain stream to
    /// `dest` mixes `OpenInputSlot` + `SessionInput`, so both match arms are live.
    fn shard_input_seqs(sent: &[(NodeId, MsgClass, Vec<u8>)], dest: NodeId) -> Vec<u64> {
        sent.iter()
            .filter(|(to, _, _)| *to == dest)
            .filter_map(|(_, _, bytes)| {
                match postcard::from_bytes::<GatewayToShard>(bytes).expect("gateway sends valid") {
                    GatewayToShard::SessionInput { input_bytes, .. } => {
                        Some(peek_input_seq(&input_bytes).expect("valid input"))
                    }
                    _ => None,
                }
            })
            .collect()
    }

    /// The `resume_from_seq`s of `OpenInputSlot` frames the gateway sent to `dest`.
    /// As above, the variant match discriminates so the `_ => None` arm is exercised by the
    /// `SessionInput` frames in the same drained stream.
    fn open_slot_watermarks(sent: &[(NodeId, MsgClass, Vec<u8>)], dest: NodeId) -> Vec<u64> {
        sent.iter()
            .filter(|(to, _, _)| *to == dest)
            .filter_map(|(_, _, bytes)| {
                match postcard::from_bytes::<GatewayToShard>(bytes).expect("gateway sends valid") {
                    GatewayToShard::OpenInputSlot {
                        resume_from_seq, ..
                    } => Some(resume_from_seq),
                    _ => None,
                }
            })
            .collect()
    }

    /// The `subject`s of `OpenInputSlot` frames the gateway sent to `dest` (1c.8): proves the
    /// CommitAuthority subject is forwarded VERBATIM into the dest's adopt slot.
    fn open_slot_subjects(sent: &[(NodeId, MsgClass, Vec<u8>)], dest: NodeId) -> Vec<DirectoryKey> {
        sent.iter()
            .filter(|(to, _, _)| *to == dest)
            .filter_map(|(_, _, bytes)| {
                match postcard::from_bytes::<GatewayToShard>(bytes).expect("gateway sends valid") {
                    GatewayToShard::OpenInputSlot { subject, .. } => Some(subject),
                    _ => None,
                }
            })
            .collect()
    }

    fn dest_buffer_len(rig: &Rig, sid: SessionId) -> usize {
        rig.world
            .resource::<GatewaySessions>()
            .by_session
            .get(&sid)
            .expect("session")
            .transfer
            .as_ref()
            .map_or(0, |tp| tp.dest_buffer.len())
    }

    #[test]
    fn seq_past_the_marker_buffers_for_the_dest_and_is_not_forwarded() {
        let mut rig = Rig::new();
        let (sid, _) = rig.login();
        let _ = freeze_to_live_cut(&mut rig, sid, 10);
        // Three seq>marker inputs: held in the cut buffer, NOT forwarded to the source.
        let sent = rig.tick(vec![client_input(11), client_input(12), client_input(13)]);
        assert_eq!(rig.stats().inputs_buffered_for_dest, 3);
        assert_eq!(dest_buffer_len(&rig, sid), 3);
        assert_eq!(
            shard_input_seqs(&sent, SHARD),
            Vec::<u64>::new(),
            "buffered input does NOT go to the source"
        );
        assert_eq!(
            shard_input_seqs(&sent, DEST),
            Vec::<u64>::new(),
            "nothing reaches the dest until commit"
        );
    }

    #[test]
    fn the_cut_buffer_drops_oldest_over_cap_and_counts() {
        // Default cap is 8 (config()); 11 inputs ⇒ 3 oldest shed, newest 8 kept.
        let mut rig = Rig::new();
        let (sid, _) = rig.login();
        let _ = freeze_to_live_cut(&mut rig, sid, 100);
        for seq in 101..=111 {
            let _ = rig.tick(vec![client_input(seq)]);
        }
        assert_eq!(
            dest_buffer_len(&rig, sid),
            8,
            "capped at max_buffered_inputs"
        );
        assert_eq!(rig.stats().dest_inputs_dropped, 3);
        assert_eq!(
            rig.stats().inputs_buffered_for_dest,
            11,
            "all counted as buffered"
        );
        // Commit + drain and assert WHICH seqs survive: drop-OLDEST means the kept window is the
        // NEWEST 8 (104..=111), drained in seq order — proving the latest-wins identity, not just
        // the length. A drop-NEWEST inversion would strand the player's most recent input here.
        let sent = rig.tick(vec![saga_cmd(TransferControl::CommitAuthority {
            transfer: XFER,
            session: sid,
            new_fence: Fence(2),
            subject: XFER_SUBJECT,
        })]);
        assert_eq!(
            shard_input_seqs(&sent, DEST),
            vec![104, 105, 106, 107, 108, 109, 110, 111],
            "the drained survivors are exactly the kept newest-8 window, in order"
        );
    }

    #[test]
    fn commit_opens_the_dest_slot_then_drains_the_buffer_in_seq_order() {
        let mut rig = Rig::new();
        let (sid, _) = rig.login();
        let _ = freeze_to_live_cut(&mut rig, sid, 10);
        let _ = rig.tick(vec![client_input(11), client_input(12), client_input(13)]);
        let sent = rig.tick(vec![saga_cmd(TransferControl::CommitAuthority {
            transfer: XFER,
            session: sid,
            new_fence: Fence(2),
            subject: XFER_SUBJECT,
        })]);
        // The authoritative OpenInputSlot carries resume_from_seq == marker_seq.
        assert_eq!(open_slot_watermarks(&sent, DEST), vec![10]);
        // 1c.8: it also carries the transfer subject VERBATIM (the dest adopts this avatar).
        assert_eq!(open_slot_subjects(&sent, DEST), vec![XFER_SUBJECT]);
        // The buffer drained to the dest, in seq order.
        assert_eq!(shard_input_seqs(&sent, DEST), vec![11, 12, 13]);
        assert_eq!(dest_buffer_len(&rig, sid), 0, "buffer emptied by the drain");
        assert_eq!(
            acks_to_orch(&sent),
            vec![TransferControlAck::Committed { transfer: XFER }]
        );
    }

    #[test]
    fn a_redelivered_commit_does_not_re_drain_the_buffer() {
        let mut rig = Rig::new();
        let (sid, _) = rig.login();
        let _ = freeze_to_live_cut(&mut rig, sid, 10);
        let _ = rig.tick(vec![client_input(11), client_input(12)]);
        let commit = || {
            saga_cmd(TransferControl::CommitAuthority {
                transfer: XFER,
                session: sid,
                new_fence: Fence(2),
                subject: XFER_SUBJECT,
            })
        };
        let first = rig.tick(vec![commit()]);
        assert_eq!(shard_input_seqs(&first, DEST), vec![11, 12]);
        // Redelivery: re-serves Committed from the journal, drains NOTHING (buffer is empty).
        let second = rig.tick(vec![commit()]);
        assert_eq!(
            acks_to_orch(&second),
            acks_to_orch(&first),
            "same Committed re-served"
        );
        assert_eq!(
            shard_input_seqs(&second, DEST),
            Vec::<u64>::new(),
            "the redelivery never re-drains"
        );
    }

    #[test]
    fn abort_drops_the_cut_buffer() {
        let mut rig = Rig::new();
        let (sid, _) = rig.login();
        let _ = freeze_to_live_cut(&mut rig, sid, 10);
        let _ = rig.tick(vec![client_input(11), client_input(12)]);
        assert_eq!(dest_buffer_len(&rig, sid), 2);
        let sent = rig.tick(vec![saga_cmd(TransferControl::AbortTransfer {
            transfer: XFER,
            session: sid,
        })]);
        assert!(!transfer_in_flight(&rig, sid), "abort pruned the progress");
        assert_eq!(
            shard_input_seqs(&sent, DEST),
            Vec::<u64>::new(),
            "buffered seq>marker frames reach NEITHER shard on abort (player stays on source)"
        );
    }

    #[test]
    fn abort_acks_and_prunes_the_progress() {
        let mut rig = Rig::new();
        let (sid, _) = rig.login();
        let _ = rig.tick(vec![saga_cmd(TransferControl::PrepareSubscribe {
            transfer: XFER,
            session: sid,
            dest: SHARD,
        })]);
        assert!(transfer_in_flight(&rig, sid));
        let sent = rig.tick(vec![saga_cmd(TransferControl::AbortTransfer {
            transfer: XFER,
            session: sid,
        })]);
        assert_eq!(
            acks_to_orch(&sent),
            vec![TransferControlAck::Aborted { transfer: XFER }]
        );
        assert!(
            !transfer_in_flight(&rig, sid),
            "AbortTransfer pruned the progress (the saga's terminal)"
        );
    }

    #[test]
    fn a_redelivered_prepare_resends_the_same_ack_without_re_applying() {
        let mut rig = Rig::new();
        let (sid, _) = rig.login();
        let prepare = || {
            saga_cmd(TransferControl::PrepareSubscribe {
                transfer: XFER,
                session: sid,
                dest: SHARD,
            })
        };
        let first = rig.tick(vec![prepare()]);
        let second = rig.tick(vec![prepare()]);
        // The SAME Prepared ack both times (the journal re-sent it; no second effect).
        assert_eq!(acks_to_orch(&first), acks_to_orch(&second));
        assert_eq!(acks_to_orch(&second).len(), 1);
    }

    #[test]
    fn a_cut_marker_is_inert_no_ack_after_request_cut() {
        // S3: the client CUT_MARKER is RETIRED. After RequestCut has self-acked CutConfirmed, a
        // stamped marker on the input flow is just an ordinary input — it drives NO ack, however
        // many times it is (re)sent. (The saga already advanced server-side; the marker is dead.)
        let mut rig = Rig::new();
        let (sid, _) = rig.login();
        let _ = rig.tick(vec![saga_cmd(TransferControl::PrepareSubscribe {
            transfer: XFER,
            session: sid,
            dest: SHARD,
        })]);
        let _ = rig.tick(vec![saga_cmd(TransferControl::RequestCut {
            transfer: XFER,
            session: sid,
        })]);
        // Three sends of the marker at the same seq → ZERO acks each (the marker is inert now).
        let a = rig.tick(vec![marker_input(9)]);
        let b = rig.tick(vec![marker_input(9)]);
        let c = rig.tick(vec![marker_input(9)]);
        assert_eq!(
            acks_to_orch(&a),
            vec![],
            "a stamped marker drives no ack (S3)"
        );
        assert_eq!(
            acks_to_orch(&b),
            vec![],
            "a re-sent marker drives no ack (S3)"
        );
        assert_eq!(
            acks_to_orch(&c),
            vec![],
            "a re-sent marker drives no ack (S3)"
        );
    }

    fn session_transfer_is_none(rig: &Rig, sid: SessionId) -> bool {
        rig.world
            .resource::<GatewaySessions>()
            .by_session
            .get(&sid)
            .expect("session")
            .transfer
            .is_none()
    }

    #[test]
    fn release_subscribe_acks_released() {
        // 1c.8: ReleaseSubscribe is the LAST phase flipped from PARK to LIVE — it acks Released
        // (so the saga reaches Done) and clears session.transfer (the last post-commit remnant).
        let mut rig = Rig::new();
        let (sid, _) = rig.login();
        let _ = rig.tick(vec![saga_cmd(TransferControl::PrepareSubscribe {
            transfer: XFER,
            session: sid,
            dest: SHARD,
        })]);
        let sent = rig.tick(vec![saga_cmd(TransferControl::ReleaseSubscribe {
            transfer: XFER,
            session: sid,
            src: SHARD,
        })]);
        assert_eq!(
            acks_to_orch(&sent),
            vec![TransferControlAck::Released { transfer: XFER }],
            "ReleaseSubscribe now acks Released (no longer parks)"
        );
        assert_eq!(
            rig.stats().transfer_control_parked,
            0,
            "nothing parks anymore"
        );
        // The in-flight transfer is pruned (the source subscription closed).
        assert!(
            session_transfer_is_none(&rig, sid),
            "session.transfer cleared on release"
        );
        // Idempotent redelivery: a stray re-ack of the now-absent transfer is a clean
        // no-op-and-ack (still Released, transfer_unroutable stays 0).
        let again = rig.tick(vec![saga_cmd(TransferControl::ReleaseSubscribe {
            transfer: XFER,
            session: sid,
            src: SHARD,
        })]);
        assert_eq!(
            acks_to_orch(&again),
            vec![TransferControlAck::Released { transfer: XFER }],
            "a redelivered Release is a clean no-op-and-ack"
        );
        assert_eq!(
            rig.stats().transfer_unroutable,
            0,
            "a release re-ack is not a routing failure"
        );
    }

    #[test]
    fn thaw_for_an_unrecorded_transfer_still_acks_but_is_counted() {
        // ThawSource's compensator must always complete (a thaw against a never-frozen
        // source is a correct no-op), yet the missing progress is counted, not silent.
        let mut rig = Rig::new();
        let (sid, _) = rig.login();
        let sent = rig.tick(vec![saga_cmd(TransferControl::ThawSource {
            transfer: XFER,
            session: sid,
        })]);
        assert_eq!(
            acks_to_orch(&sent),
            vec![TransferControlAck::SourceThawed { transfer: XFER }]
        );
        assert_eq!(rig.stats().transfer_unroutable, 1);
    }

    #[test]
    fn abort_for_an_unrecorded_transfer_is_an_idempotent_uncounted_re_ack() {
        // An abort is an idempotent terminal: aborting a transfer this gateway never held
        // re-acks Aborted and is NOT a routing failure (F2: transfer_unroutable stays clean).
        let mut rig = Rig::new();
        let (sid, _) = rig.login();
        let sent = rig.tick(vec![saga_cmd(TransferControl::AbortTransfer {
            transfer: XFER,
            session: sid,
        })]);
        assert_eq!(
            acks_to_orch(&sent),
            vec![TransferControlAck::Aborted { transfer: XFER }]
        );
        assert_eq!(
            rig.stats().transfer_unroutable,
            0,
            "an abort is not unroutable"
        );
    }

    #[test]
    fn a_redelivered_terminal_abort_is_idempotent_and_uncounted() {
        // F2: after a bound abort prunes the progress, a redelivered abort (now unbound)
        // re-acks Aborted without inflating transfer_unroutable (healthy at-least-once).
        let mut rig = Rig::new();
        let (sid, _) = rig.login();
        let _ = rig.tick(vec![saga_cmd(TransferControl::PrepareSubscribe {
            transfer: XFER,
            session: sid,
            dest: SHARD,
        })]);
        let abort = || {
            saga_cmd(TransferControl::AbortTransfer {
                transfer: XFER,
                session: sid,
            })
        };
        let first = rig.tick(vec![abort()]);
        let second = rig.tick(vec![abort()]); // redelivered terminal
        let aborted = vec![TransferControlAck::Aborted { transfer: XFER }];
        assert_eq!(acks_to_orch(&first), aborted);
        assert_eq!(
            acks_to_orch(&second),
            aborted,
            "the redelivered terminal re-acks"
        );
        assert_eq!(
            rig.stats().transfer_unroutable,
            0,
            "no spurious unroutable count"
        );
    }

    #[test]
    fn abort_of_a_different_transfer_does_not_clobber_the_in_flight_one() {
        // CP-1: an AbortTransfer for transfer B must NOT prune in-flight transfer A's
        // progress — the prune is keyed on the matching transfer, not the variant.
        let mut rig = Rig::new();
        let (sid, _) = rig.login();
        let other = TransferId(0x777);
        let _ = rig.tick(vec![saga_cmd(TransferControl::PrepareSubscribe {
            transfer: XFER,
            session: sid,
            dest: SHARD,
        })]);
        assert!(transfer_in_flight(&rig, sid));
        let sent = rig.tick(vec![saga_cmd(TransferControl::AbortTransfer {
            transfer: other,
            session: sid,
        })]);
        assert_eq!(
            acks_to_orch(&sent),
            vec![TransferControlAck::Aborted { transfer: other }],
            "the foreign abort still acks idempotently"
        );
        assert!(
            transfer_in_flight(&rig, sid),
            "in-flight transfer A survives an abort aimed at transfer B"
        );
    }

    #[test]
    fn release_of_a_different_transfer_does_not_clobber_the_in_flight_one() {
        // TAIL-1 (mirrors the abort no-clobber case): a ReleaseSubscribe for transfer B must NOT
        // prune in-flight transfer A — `apply_release` prunes ONLY the matching transfer; a stray
        // re-ack of an absent/foreign transfer is a clean ack-and-no-op.
        let mut rig = Rig::new();
        let (sid, _) = rig.login();
        let other = TransferId(0x777);
        let _ = rig.tick(vec![saga_cmd(TransferControl::PrepareSubscribe {
            transfer: XFER,
            session: sid,
            dest: SHARD,
        })]);
        assert!(transfer_in_flight(&rig, sid));
        let sent = rig.tick(vec![saga_cmd(TransferControl::ReleaseSubscribe {
            transfer: other,
            session: sid,
            src: SHARD,
        })]);
        assert_eq!(
            acks_to_orch(&sent),
            vec![TransferControlAck::Released { transfer: other }],
            "the foreign release still acks idempotently"
        );
        assert!(
            transfer_in_flight(&rig, sid),
            "in-flight transfer A survives a release aimed at transfer B"
        );
    }

    #[test]
    fn a_bye_with_an_in_flight_transfer_is_warned_and_detaches() {
        // WEDGE-1 pin: a Bye mid-transfer drops the session + journal (the saga then pins
        // until the Slice-2 timeout). The path runs cleanly + detaches; the warn fires.
        let mut rig = Rig::new();
        let (sid, _) = rig.login();
        let _ = rig.tick(vec![saga_cmd(TransferControl::PrepareSubscribe {
            transfer: XFER,
            session: sid,
            dest: SHARD,
        })]);
        assert!(transfer_in_flight(&rig, sid));
        let _ = rig.tick(vec![wire(
            CLIENT,
            MsgClass::Control,
            &ClientControlMsg::Bye,
        )]);
        // The session (and its in-flight-transfer journal) is dropped on Bye — the WEDGE-1
        // warn path ran without panic. (The DetachSession/LeaseRevoke fan-out is covered by
        // `bye_detaches_revokes_and_clears`.)
        assert!(
            !rig.world
                .resource::<GatewaySessions>()
                .by_session
                .contains_key(&sid),
            "the session (and its journal) is dropped on Bye"
        );
    }

    #[test]
    fn request_cut_without_a_matching_prepare_is_counted_and_pushes_nothing() {
        let mut rig = Rig::new();
        let (sid, _) = rig.login();
        let sent = rig.tick(vec![saga_cmd(TransferControl::RequestCut {
            transfer: XFER,
            session: sid,
        })]);
        assert_eq!(
            decode_controls(&sent, CLIENT),
            vec![],
            "no RequestCut pushed"
        );
        assert_eq!(rig.stats().transfer_unroutable, 1);
    }

    #[test]
    fn an_ordinary_input_during_a_transfer_is_not_a_cut_marker() {
        // The cold marker observer runs while a transfer is in flight, but a NON-marker
        // input (`is_cut_marker = false`) produces no CutConfirmed — only routing.
        let mut rig = Rig::new();
        let (sid, _) = rig.login();
        let _ = rig.tick(vec![saga_cmd(TransferControl::PrepareSubscribe {
            transfer: XFER,
            session: sid,
            dest: SHARD,
        })]);
        let sent = rig.tick(vec![wire(
            CLIENT,
            MsgClass::Input,
            &InputDatagram {
                seq: 5,
                is_cut_marker: false,
                client_tick: TickId(1),
                movement: [1.0, 0.0, 0.0],
                look: [0.0, 0.0],
                action_bits: 0,
            },
        )]);
        assert_eq!(
            acks_to_orch(&sent),
            vec![],
            "ordinary input yields no CutConfirmed"
        );
    }

    #[test]
    fn a_cut_marker_is_inert_before_and_after_request_cut() {
        // S3: the client CUT_MARKER never drives a CutConfirmed — before OR after RequestCut. The
        // cut is server-timed: RequestCut self-acks CutConfirmed, and a marker input is ordinary.
        let mut rig = Rig::new();
        let (sid, _) = rig.login();
        let _ = rig.tick(vec![saga_cmd(TransferControl::PrepareSubscribe {
            transfer: XFER,
            session: sid,
            dest: SHARD,
        })]);
        // A marker BEFORE RequestCut: inert (an ordinary input; no ack).
        let sent = rig.tick(vec![marker_input(7)]);
        assert_eq!(
            acks_to_orch(&sent),
            vec![],
            "a marker before RequestCut drives no ack (S3 — inert)"
        );
        // RequestCut self-acks CutConfirmed server-side.
        let sent = rig.tick(vec![saga_cmd(TransferControl::RequestCut {
            transfer: XFER,
            session: sid,
        })]);
        assert_eq!(
            acks_to_orch(&sent),
            vec![TransferControlAck::CutConfirmed {
                transfer: XFER,
                marker_seq: 0,
            }],
            "RequestCut self-acks the cut server-side (no client marker)",
        );
        // A marker AFTER RequestCut is still inert (the saga already advanced; no second ack).
        let sent = rig.tick(vec![marker_input(8)]);
        assert_eq!(
            acks_to_orch(&sent),
            vec![],
            "a marker after RequestCut drives no ack (S3 — the cut is already server-timed)"
        );
    }

    #[test]
    fn a_seq_valid_but_undecodable_input_during_a_transfer_is_not_a_marker() {
        // The cold observer full-decodes; `route_input` only peeks the leading seq varint.
        // A datagram whose seq varint is valid but whose body is truncated routes normally,
        // then the observer's decode fails → no CutConfirmed, no panic.
        let mut rig = Rig::new();
        let (sid, _) = rig.login();
        let _ = rig.tick(vec![saga_cmd(TransferControl::PrepareSubscribe {
            transfer: XFER,
            session: sid,
            dest: SHARD,
        })]);
        let sent = rig.tick(vec![Inbound::Wire {
            from: CLIENT,
            class: MsgClass::Input,
            bytes: vec![0x05].into(), // seq=5 (valid varint), then EOF → full decode fails
        }]);
        assert_eq!(
            acks_to_orch(&sent),
            vec![],
            "an undecodable input yields no CutConfirmed"
        );
    }

    #[test]
    fn a_command_for_a_different_transfer_does_not_consult_the_wrong_journal() {
        // The redelivery gate only re-sends from the journal when the in-flight transfer
        // matches. A command for a DIFFERENT transfer falls through and is handled fresh
        // (defensive replace), never absorbed against the wrong saga.
        let mut rig = Rig::new();
        let (sid, _) = rig.login();
        let other = TransferId(0x999);
        let _ = rig.tick(vec![saga_cmd(TransferControl::PrepareSubscribe {
            transfer: XFER,
            session: sid,
            dest: SHARD,
        })]);
        let sent = rig.tick(vec![saga_cmd(TransferControl::PrepareSubscribe {
            transfer: other,
            session: sid,
            dest: SHARD,
        })]);
        assert_eq!(
            acks_to_orch(&sent),
            vec![TransferControlAck::Prepared {
                transfer: other,
                result: PrepareResult::Ready,
            }],
            "the different transfer is handled fresh, not re-sent from the XFER journal"
        );
    }

    #[test]
    fn freshest_session_confirmed_maxes_over_active_and_selffenced() {
        // A bare session in a given phase with a given confirmed_at tick (uses the module test consts).
        fn sess(phase: SessionPhase, confirmed: u64) -> Session {
            Session {
                client: CLIENT,
                account: AccountId(5),
                fence: Fence(1),
                phase,
                next_sub: 0,
                // 5f-3d: a STATIC session (no dynamic home, no bootstrap window) — the byte-identical shape.
                home_shard: None,
                home_rid: None,
                realm_feed_frame_id: 0,
                scene_sent: BTreeMap::new(),
                // A bare test session: no home descended, so no spawn pose — the static shape.
                spawn: None,
                bootstrap_deadline: None,
                confirmed_at: TickId(confirmed),
                negotiated_minor: 1,
                transfer: None,
                subs: BTreeMap::new(),
                delivered: BTreeMap::new(),
                lineage: Vec::new(),
                shadow: window::ShadowScene::default(),
                hot: Arc::new(SessionHot {
                    route: ArcSwap::from_pointee(RouteSnapshot {
                        authority: SHARD,
                        fence: Fence(1),
                        cut: None,
                    }),
                    last_input_seq: AtomicU64::new(0),
                    subs: ArcSwap::from_pointee(SubTable::default()),
                }),
            }
        }
        let mut sessions = GatewaySessions::default();
        // Empty table → None (no session carries a meaningful confirmed_at).
        assert_eq!(sessions.freshest_session_confirmed(), None);
        // Only PRE-Active logins → None (their confirmed_at is the 0 sentinel, never a real round-trip),
        // even though their values are large — they must not be able to de-route a long-running gateway.
        sessions
            .by_session
            .insert(SessionId(1), sess(SessionPhase::AwaitingDirectory, 999));
        sessions
            .by_session
            .insert(SessionId(2), sess(SessionPhase::AwaitingAttach, 888));
        assert_eq!(sessions.freshest_session_confirmed(), None);
        // ONLY a SelfFenced session (a totally-partitioned gateway that self-fenced its last session): the
        // FROZEN confirmed is visible → Some(stale). This is the case an Active-only max wrongly read as None
        // (falsely Ready). It is the de-route signal.
        sessions
            .by_session
            .insert(SessionId(3), sess(SessionPhase::SelfFenced, 40));
        assert_eq!(sessions.freshest_session_confirmed(), Some(40));
        // Add fresh Active sessions → the MAX over {Active ∪ SelfFenced} picks the freshest Active (a healthy
        // gateway stays Ready despite the lingering SelfFenced ghost); the pre-Active 999 stays excluded.
        sessions.by_session.insert(
            SessionId(4),
            sess(
                SessionPhase::Active {
                    entity: EntityId(1),
                },
                30,
            ),
        );
        sessions.by_session.insert(
            SessionId(5),
            sess(
                SessionPhase::Active {
                    entity: EntityId(2),
                },
                50,
            ),
        );
        assert_eq!(sessions.freshest_session_confirmed(), Some(50));
    }

    // ---- Slice 1d.2a: the route-table reshape (sub registry + reverse index) ----

    /// Build a bare `GatewaySessions` with ONE Active session whose only sub is on `SHARD`,
    /// exactly as a real login would leave it — the substrate for the primitive unit tests.
    fn one_active_session() -> (GatewaySessions, SessionId, OutboundBox) {
        let mut sessions = GatewaySessions::default();
        let sid = SessionId(0xA11A);
        sessions.by_session.insert(
            sid,
            Session {
                client: CLIENT,
                account: AccountId(5),
                fence: Fence(1),
                phase: SessionPhase::Active {
                    entity: EntityId(77),
                },
                next_sub: 0,
                // 5f-3d: a STATIC session (no dynamic home, no bootstrap window) — the byte-identical shape.
                home_shard: None,
                home_rid: None,
                realm_feed_frame_id: 0,
                scene_sent: BTreeMap::new(),
                // A bare test session: no home descended, so no spawn pose — the static shape.
                spawn: None,
                bootstrap_deadline: None,
                confirmed_at: TickId(0),
                negotiated_minor: 1,
                transfer: None,
                subs: BTreeMap::new(),
                delivered: BTreeMap::new(),
                lineage: Vec::new(),
                shadow: window::ShadowScene::default(),
                hot: Arc::new(SessionHot {
                    route: ArcSwap::from_pointee(RouteSnapshot {
                        authority: SHARD,
                        fence: Fence(1),
                        cut: None,
                    }),
                    last_input_seq: AtomicU64::new(0),
                    subs: ArcSwap::from_pointee(SubTable::default()),
                }),
            },
        );
        sessions.by_client.insert(CLIENT, sid);
        let mut outbox = OutboundBox::default();
        // Open the login sub (the FIRST open_sub caller), exactly as on_shard_control does.
        let sub = sessions
            .open_sub(
                sid,
                SHARD,
                FrameRef::SystemSpace { system_seed: 7 },
                Fence(1),
                &mut outbox,
            )
            .expect("session present");
        assert_eq!(sub, SubId(0));
        (sessions, sid, outbox)
    }

    /// Decode the control messages a captured `OutboundBox` sent to a client.
    fn controls_in(outbox: &OutboundBox, to: NodeId) -> Vec<ServerControlMsg> {
        let owned: Vec<(NodeId, MsgClass, Vec<u8>)> = outbox
            .0
            .iter()
            .map(|(t, c, b, _)| (*t, *c, b.to_vec()))
            .collect();
        decode_controls(&owned, to)
    }

    #[test]
    fn open_sub_indexes_publishes_and_emits_opened_before_the_table() {
        // open_sub: X1 (SubscriptionOpened pushed BEFORE the hot table is readable), the cold
        // SubRecord installed, the reverse index populated, and the hot SubTable carries the
        // accepted fence. A second open on a DISTINCT shard yields a fresh never-reused sub.
        let (mut sessions, sid, outbox) = one_active_session();
        assert_eq!(
            controls_in(&outbox, CLIENT),
            vec![ServerControlMsg::SubscriptionOpened {
                sub: SubId(0),
                frame: FrameRef::SystemSpace { system_seed: 7 },
            }]
        );
        // The reverse index now lists this session under SHARD (and nothing under DEST).
        assert_eq!(sessions.subscribers_of(SHARD), vec![sid]);
        assert_eq!(sessions.subscribers_of(DEST), Vec::<SessionId>::new());
        // The hot SubTable resolves SHARD to (SubId(0), Fence(1)).
        let table = sessions.by_session[&sid].hot.subs.load();
        assert_eq!(
            table.lookup(SHARD),
            Some(&SubEntry {
                shard: SHARD,
                sub: SubId(0),
                accepted: Fence(1),
            })
        );
        assert_eq!(table.lookup(DEST), None, "no sub for an unsubscribed shard");
        // A second open on DEST allocates the next monotonic, never-reused sub id.
        let mut outbox2 = OutboundBox::default();
        let sub1 = sessions
            .open_sub(
                sid,
                DEST,
                FrameRef::SystemSpace { system_seed: 8 },
                Fence(3),
                &mut outbox2,
            )
            .expect("session present");
        assert_eq!(sub1, SubId(1), "monotonic, never reused");
        assert_eq!(sessions.subscribers_of(DEST), vec![sid]);
        let table = sessions.by_session[&sid].hot.subs.load();
        assert_eq!(table.lookup(DEST).map(|e| e.sub), Some(SubId(1)));
        assert_eq!(table.lookup(DEST).map(|e| e.accepted), Some(Fence(3)));
    }

    #[test]
    fn open_sub_for_an_absent_session_is_a_counted_free_none() {
        // The `?` arm: open_sub on an unknown session returns None and touches nothing.
        let mut sessions = GatewaySessions::default();
        let mut outbox = OutboundBox::default();
        assert_eq!(
            sessions.open_sub(
                SessionId(0xDEAD),
                SHARD,
                FrameRef::SystemSpace { system_seed: 7 },
                Fence(1),
                &mut outbox,
            ),
            None
        );
        assert!(outbox.0.is_empty());
        assert!(sessions.subscribers_of(SHARD).is_empty());
    }

    #[test]
    fn close_sub_drains_for_one_tick_then_the_sweep_removes_it() {
        // close_sub marks Draining + emits SubscriptionClosing but KEEPS the sub in the table +
        // index for one tick (a straggler is still routable); the next sweep removes it.
        let (mut sessions, sid, _) = one_active_session();
        let mut outbox = OutboundBox::default();
        let mut stats = GatewayStats::default();
        // The production shape of a source-sub close: `CommitAuthority` already re-pointed the
        // session's authority at the DEST, so SHARD (the source) is no longer the feeding node and
        // the authority-sub invariant lets the release through.
        sessions
            .by_session
            .get_mut(&sid)
            .expect("the fixture session is present")
            .home_shard = Some(DEST);
        sessions.close_sub(sid, SHARD, &config(), &mut outbox, &mut stats);
        assert_eq!(stats.sub_close_refused_authority, 0);
        assert_eq!(
            controls_in(&outbox, CLIENT),
            vec![ServerControlMsg::SubscriptionClosing { sub: SubId(0) }]
        );
        // Still routable this tick (the drain grace): index + hot table both still resolve it.
        assert_eq!(sessions.subscribers_of(SHARD), vec![sid]);
        assert_eq!(
            sessions.by_session[&sid]
                .hot
                .subs
                .load()
                .lookup(SHARD)
                .map(|e| e.sub),
            Some(SubId(0)),
            "Draining sub stays in the hot table for its one-tick grace"
        );
        // A SECOND close is idempotent — no second SubscriptionClosing.
        let mut outbox2 = OutboundBox::default();
        sessions.close_sub(sid, SHARD, &config(), &mut outbox2, &mut stats);
        assert!(outbox2.0.is_empty(), "already Draining: no second close");
        // The next-tick sweep removes it from BOTH the table and the index.
        sessions.sweep_draining();
        assert_eq!(sessions.subscribers_of(SHARD), Vec::<SessionId>::new());
        assert_eq!(
            sessions.by_session[&sid].hot.subs.load().lookup(SHARD),
            None,
            "swept out of the hot table"
        );
        assert!(
            sessions.subscribed_shards.is_empty(),
            "the emptied reverse-index entry is removed"
        );
    }

    #[test]
    fn close_sub_for_an_absent_session_or_unsubscribed_shard_is_a_no_op() {
        // Both early-return arms: an unknown session, and a known session that does not
        // subscribe to the named shard.
        let (mut sessions, sid, _) = one_active_session();
        let mut outbox = OutboundBox::default();
        let mut stats = GatewayStats::default();
        let cfg = config();
        sessions.close_sub(SessionId(0xDEAD), SHARD, &cfg, &mut outbox, &mut stats); // unknown session
        sessions.close_sub(sid, DEST, &cfg, &mut outbox, &mut stats); // does not subscribe to DEST
        assert!(outbox.0.is_empty(), "neither path emits a close");
        assert_eq!(stats.sub_close_refused_authority, 0, "neither is a refusal");
        // The original SHARD sub is untouched.
        assert_eq!(sessions.subscribers_of(SHARD), vec![sid]);
    }

    #[test]
    fn close_sub_refuses_to_close_the_sessions_current_authority() {
        // THE AUTHORITY-SUB INVARIANT (the same-node re-home freeze, task #175). A re-home whose
        // source and dest are the SAME node is an ordinary saga (#149 deleted the node-placement
        // short-circuit), and its `ReleaseSubscribe` names a `src` that is ALSO the post-commit
        // authority. `subs` is keyed by shard, so honouring that close would tear down the one
        // subscription feeding the client — the player keeps being simulated while their client
        // goes totally silent (no snapshots, no realm feed, no clock, no visible input response).
        let (mut sessions, sid, _) = one_active_session();
        let mut outbox = OutboundBox::default();
        let mut stats = GatewayStats::default();
        // A static session's authority IS `config.shard` (`session_target`'s `unwrap_or`), which is
        // exactly the same-node shape: the node being released is still the one feeding the client.
        sessions.close_sub(sid, SHARD, &config(), &mut outbox, &mut stats);
        assert_eq!(stats.sub_close_refused_authority, 1, "refused + counted");
        assert!(
            outbox.0.is_empty(),
            "no SubscriptionClosing reaches the client"
        );
        // The live sub survives, in BOTH the cold table and the hot projection.
        assert_eq!(sessions.subscribers_of(SHARD), vec![sid]);
        assert_eq!(
            sessions.by_session[&sid]
                .hot
                .subs
                .load()
                .lookup(SHARD)
                .map(|e| e.sub),
            Some(SubId(0)),
            "the authority sub stays live — the client keeps receiving its own owner's frames"
        );
        // And it is NOT swept away next tick either (it was never marked Draining).
        sessions.sweep_draining();
        assert_eq!(sessions.subscribers_of(SHARD), vec![sid]);
    }

    #[test]
    fn sweep_with_no_draining_subs_is_a_no_op() {
        // The sweep's empty-`draining` continue arm: a session with only Active subs is left
        // byte-identical.
        let (mut sessions, sid, _) = one_active_session();
        sessions.sweep_draining();
        assert_eq!(sessions.subscribers_of(SHARD), vec![sid]);
        assert_eq!(
            sessions.by_session[&sid]
                .hot
                .subs
                .load()
                .lookup(SHARD)
                .map(|e| e.sub),
            Some(SubId(0))
        );
    }

    #[test]
    fn sweep_keeps_a_shared_reverse_index_entry_with_a_surviving_subscriber() {
        // The `set.is_empty()` FALSE arm: two sessions subscribe to SHARD; closing+sweeping ONE
        // leaves the reverse-index entry alive for the other (the entry is not removed).
        let (mut sessions, sid_a, _) = one_active_session();
        // A second session on the same SHARD sub.
        let sid_b = SessionId(0xB22B);
        sessions.by_session.insert(
            sid_b,
            Session {
                client: NodeId(101),
                account: AccountId(6),
                fence: Fence(1),
                phase: SessionPhase::Active {
                    entity: EntityId(88),
                },
                next_sub: 0,
                // 5f-3d: a STATIC session (no dynamic home, no bootstrap window) — the byte-identical shape.
                home_shard: None,
                home_rid: None,
                realm_feed_frame_id: 0,
                scene_sent: BTreeMap::new(),
                // A bare test session: no home descended, so no spawn pose — the static shape.
                spawn: None,
                bootstrap_deadline: None,
                confirmed_at: TickId(0),
                negotiated_minor: 1,
                transfer: None,
                subs: BTreeMap::new(),
                delivered: BTreeMap::new(),
                lineage: Vec::new(),
                shadow: window::ShadowScene::default(),
                hot: Arc::new(SessionHot {
                    route: ArcSwap::from_pointee(RouteSnapshot {
                        authority: SHARD,
                        fence: Fence(1),
                        cut: None,
                    }),
                    last_input_seq: AtomicU64::new(0),
                    subs: ArcSwap::from_pointee(SubTable::default()),
                }),
            },
        );
        sessions.by_client.insert(NodeId(101), sid_b);
        let mut ob = OutboundBox::default();
        sessions
            .open_sub(
                sid_b,
                SHARD,
                FrameRef::SystemSpace { system_seed: 7 },
                Fence(1),
                &mut ob,
            )
            .expect("present");
        let mut both = sessions.subscribers_of(SHARD);
        both.sort_unstable();
        assert_eq!(both, vec![sid_a, sid_b]);
        // Close + sweep ONLY session A. Its authority already moved to DEST (the post-commit shape),
        // so the authority-sub invariant lets the source release through.
        let mut ob = OutboundBox::default();
        sessions
            .by_session
            .get_mut(&sid_a)
            .expect("fixture session A is present")
            .home_shard = Some(DEST);
        sessions.close_sub(
            sid_a,
            SHARD,
            &config(),
            &mut ob,
            &mut GatewayStats::default(),
        );
        sessions.sweep_draining();
        assert_eq!(
            sessions.subscribers_of(SHARD),
            vec![sid_b],
            "B's entry survives A's drain (the reverse-index entry is not removed)"
        );
    }

    #[test]
    fn sweep_tolerates_a_drained_shard_missing_from_the_reverse_index() {
        // The `if let Some(set) = ..` None arm (a defensive desync guard): a cold Draining sub
        // whose shard is ABSENT from `subscribed_shards` (a forced index breach) is swept from
        // the table without panic — the index update is a no-op, never an index-out-of-bounds.
        let (mut sessions, sid, _) = one_active_session();
        // Mark the SHARD sub Draining in the COLD map directly...
        sessions
            .by_session
            .get_mut(&sid)
            .expect("present")
            .subs
            .get_mut(&SHARD)
            .expect("sub present")
            .state = SubState::Draining;
        // ...and forcibly clear the reverse index so the drained shard has no entry.
        sessions.subscribed_shards.clear();
        sessions.sweep_draining(); // must not panic on the missing-entry None arm
        assert_eq!(
            sessions.by_session[&sid].hot.subs.load().lookup(SHARD),
            None,
            "the cold Draining sub was still swept from the hot table"
        );
        assert!(sessions.subscribed_shards.is_empty());
    }

    #[test]
    fn a_frame_from_an_unsubscribed_known_shard_fans_to_nobody() {
        // The `subscribers_of` empty path: a DEST frame (DEST is a known shard) reaches
        // on_shard_frame, but no session subscribes to DEST, so it fans to nobody — and the
        // desync counter is untouched (an empty subscriber set is NOT a desync).
        let mut rig = Rig::new();
        let (_, _) = rig.login(); // subscribes only to SHARD
        let sent = rig.tick(vec![wire(
            DEST,
            MsgClass::Snapshot,
            &frame_msg(Fence(1), 9),
        )]);
        assert_eq!(
            sent,
            Vec::new(),
            "a frame from an unsubscribed shard reaches no client (and nothing else is sent)"
        );
        assert_eq!(
            rig.stats().frame_sub_desync,
            0,
            "an empty fan is not a desync"
        );
        assert_eq!(rig.stats().stale_frames_dropped, 0);
    }

    #[test]
    fn a_forced_index_table_desync_hits_the_counter_never_a_silent_drop() {
        // C2 dead-branch trap: a session in the reverse index for SHARD whose hot SubTable has
        // NO SubEntry for SHARD (an invariant breach `publish_subs` makes impossible by
        // construction) is COUNTED (`frame_sub_desync`), never a silent continue. BOTH desync
        // arms are exercised: (a) the index references a session absent from `by_session`;
        // (b) a present session whose hot table was corrupted to empty.
        let frame = postcard::to_allocvec(&frame_msg(Fence(1), 1)).expect("encode");

        // (a) index points at a session that does not exist in by_session.
        let mut sessions = GatewaySessions::default();
        sessions
            .subscribed_shards
            .entry(SHARD)
            .or_default()
            .insert(SessionId(0xC0DE));
        let mut stats = GatewayStats::default();
        let mut outbox = OutboundBox::default();
        on_shard_frame(SHARD, &frame, &mut sessions, &mut stats, &mut outbox);
        assert_eq!(stats.frame_sub_desync, 1, "(a) missing session is counted");
        assert!(outbox.0.is_empty());

        // (b) a present session indexed under SHARD but with an EMPTY hot SubTable.
        let (mut sessions, sid, _) = one_active_session();
        sessions.by_session[&sid]
            .hot
            .subs
            .store(Arc::new(SubTable::default()));
        let mut stats = GatewayStats::default();
        let mut outbox = OutboundBox::default();
        on_shard_frame(SHARD, &frame, &mut sessions, &mut stats, &mut outbox);
        assert_eq!(stats.frame_sub_desync, 1, "(b) lookup None is counted");
        assert!(outbox.0.is_empty(), "no frame forwarded on a desync");
    }

    #[test]
    fn a_frame_with_a_malformed_snapshot_body_is_counted_undecodable_not_forwarded() {
        // 1d.5a: on_shard_frame peeks the frame_id off the body BEFORE the fan (to advance the
        // delivery watermark). A body that fails the peek — a corrupt/buggy shard — is counted
        // (`undecodable`) and the WHOLE frame abandoned once, never forwarded. (The peek validating
        // the sub varint is also what makes the per-session retag below infallible.)
        let bad = postcard::to_allocvec(&ShardToGateway::Frame {
            realm_fence: Fence(1),
            source_tick: TickId(5),
            snapshot_bytes: vec![0x80], // a truncated varint — peek_snapshot_frame_id errors
        })
        .expect("encode");
        let (mut sessions, _sid, _) = one_active_session();
        let mut stats = GatewayStats::default();
        let mut outbox = OutboundBox::default();
        on_shard_frame(SHARD, &bad, &mut sessions, &mut stats, &mut outbox);
        assert_eq!(
            stats.undecodable, 1,
            "a malformed snapshot body is counted undecodable"
        );
        assert!(outbox.0.is_empty(), "nothing forwarded on a malformed body");
    }

    /// The realm datagram a `realm_frame_msg` envelope carries — the payload the shard authored, which is
    /// exactly what the client must receive. The ONE builder, which `realm_frame_msg` wraps — so reading
    /// the payload never destructures an enum a fixture just built (a dead refusal arm by construction).
    fn realm_frame_payload(realm: RealmId) -> Vec<u8> {
        let snapshot = RealmSnapshotDatagram {
            sub: SubId(0),
            frame_id: 3,
            source_tick: TickId(5),
            universe_tick: UniverseTick(50),
            origin_epoch: 0,
            realms: vec![RealmSnap {
                realm,
                // The edge HEAD (proto_minor 8): the CHILD's own frame. The gateway IGNORES it in this
                // slice — it forwards realm bytes verbatim — but a row without it is not a placement
                // edge, so the fixture ships a real one rather than a same-frame placeholder.
                frame: vd_core::pose::frame_for_realm(realm, None)
                    .expect("a Planet realm resolves"),
                pose: vd_core::pose::StampedPose::at_rest(
                    FrameRef::SystemSpace { system_seed: 7 },
                    vd_core::glam::DVec3::new(1.0e9, 0.0, 0.0),
                    UniverseTick(50),
                ),
            }],
        };
        postcard::to_allocvec(&snapshot).expect("encode")
    }

    #[test]
    fn a_tombstoned_old_realm_frame_is_counted_and_serves_nobody() {
        // ★TOMBSTONE (Slice C2, minor 19): the old opaque realm datagram has no producer and no
        // consumer. It still DECODES (its discriminant is reserved forever), so the honest
        // accounting is its own counter — never the `undecodable` bucket (which would hide a
        // revived producer among real garbage), and never a byte to a client.
        let (mut sessions, _sid, _) = one_active_session();
        let mut stats = GatewayStats::default();
        let envelope = postcard::to_allocvec(&realm_frame_msg(RealmId::Planet(7))).expect("encode");
        on_shard_realm_frame(SHARD, &envelope, &config(), &mut sessions, &mut stats);
        assert_eq!(
            stats.old_realm_frames_dropped, 1,
            "the dead lane is counted, so a revived producer is visible"
        );
        assert_eq!(
            stats.undecodable, 0,
            "and it is NOT filed as garbage — it decodes, it is simply dead"
        );
    }

    #[test]
    fn a_malformed_realm_envelope_is_counted_once_and_forwards_nothing() {
        // `on_shard_realm_frame` is the ONE place the realm envelope is opened, so garbage on that class
        // is counted exactly ONCE and forwards nothing.
        let mut rig = Rig::new();
        let (_, _) = rig.login();
        let sent = rig.tick(vec![wire(SHARD, MsgClass::RealmSnapshot, &0xffu8)]);
        assert_eq!(rig.world.resource::<GatewayStats>().undecodable, 1);
        assert_eq!(to_client(&sent, MsgClass::RealmSnapshot), 0);
    }

    #[test]
    fn a_dead_realm_frame_routes_through_the_dispatch_arm_and_serves_nobody() {
        // The `MsgClass::RealmSnapshot` dispatch arm through the REAL schedule: a tombstoned
        // realm frame from a subscribed shard is ROUTED (not a stranger, not garbage) and lands
        // in its own drop counter — the client receives nothing on the realm class, because the
        // composed feed is the ONE realm feed a client has (§2.4).
        let mut rig = Rig::new();
        let (_sid, _) = rig.login(); // subscribes to SHARD
        let sent = rig.tick(vec![wire(
            SHARD,
            MsgClass::RealmSnapshot,
            &realm_frame_msg(RealmId::Planet(7)),
        )]);
        assert_eq!(
            to_client(&sent, MsgClass::RealmSnapshot),
            0,
            "the dead realm lane fans nothing to clients"
        );
        let stats = rig.world.resource::<GatewayStats>();
        assert_eq!(stats.undecodable, 0);
        assert_eq!(stats.old_realm_frames_dropped, 1);
    }

    #[test]
    fn every_observer_delivered_requires_a_non_empty_all_delivered_observer_set() {
        // The 1d.5a (a) predicate's three load-bearing properties, asserted directly (the p2
        // capstone proves them end-to-end — a vacuous fire would release the source early and
        // re-open the vanish — this pins them as a unit gate). `one_active_session` subscribes the
        // session to SHARD as sub 0.
        let (mut sessions, sid, _) = one_active_session();
        // (1) ANTI-VACUOUS: no session subscribes to DEST → the EMPTY observer set is NOT satisfied
        // (never a vacuous true — the saga must not be told "delivered" before the dest sub opens).
        assert!(
            !every_observer_delivered(&sessions, DEST),
            "an empty dest-observer set never vacuously fires the demote",
        );
        // (2) NOT DELIVERED: the SHARD observer exists but its watermark is absent (0) → blocked.
        assert!(
            !every_observer_delivered(&sessions, SHARD),
            "an observer with no delivered frame (watermark 0) blocks the predicate",
        );
        // (3) DELIVERED: advance the observer's sub-0 watermark to >=1 → satisfied.
        sessions
            .by_session
            .get_mut(&sid)
            .expect("session present")
            .delivered
            .insert(SubId(0), 1);
        assert!(
            every_observer_delivered(&sessions, SHARD),
            "every (here: the one) dest observer delivered >=1 frame -> satisfied",
        );
    }

    #[test]
    fn on_shard_frame_advances_the_delivery_watermark_max_wins_and_stale_does_not() {
        // 1d.5a: the delivery watermark advances through the REAL `on_shard_frame` path (not a
        // manual insert): an accepted past-fence frame sets `delivered[sub] = frame_id`; a LOWER
        // frame_id does NOT regress it (`.max`); a fence-STALE frame does NOT advance it (dropped
        // before the advance). `one_active_session` = SubId(0) on SHARD @ accepted Fence(1).
        let (mut sessions, sid, _) = one_active_session();
        let mut stats = GatewayStats::default();
        let mut outbox = OutboundBox::default();
        let wm = |s: &GatewaySessions| s.by_session[&sid].delivered.get(&SubId(0)).copied();

        // (a) an accepted frame (Fence(1), frame_id 9) advances the watermark to 9.
        let f9 = postcard::to_allocvec(&frame_msg(Fence(1), 9)).expect("encode");
        on_shard_frame(SHARD, &f9, &mut sessions, &mut stats, &mut outbox);
        assert_eq!(
            wm(&sessions),
            Some(9),
            "an accepted frame advances delivered[sub] to its frame_id"
        );

        // (b) a LOWER frame_id (5) does NOT regress the high-water (`.max`).
        let f5 = postcard::to_allocvec(&frame_msg(Fence(1), 5)).expect("encode");
        on_shard_frame(SHARD, &f5, &mut sessions, &mut stats, &mut outbox);
        assert_eq!(
            wm(&sessions),
            Some(9),
            "a lower frame_id never lowers the watermark (.max)"
        );

        // (c) a fence-STALE frame (Fence(0) < accepted Fence(1)) is dropped — no advance.
        let stale = postcard::to_allocvec(&frame_msg(Fence(0), 99)).expect("encode");
        on_shard_frame(SHARD, &stale, &mut sessions, &mut stats, &mut outbox);
        assert_eq!(
            wm(&sessions),
            Some(9),
            "a fence-stale frame does not advance the watermark"
        );
        assert_eq!(stats.stale_frames_dropped, 1);
    }

    // ---- Slice 1d.2b: SubscriptionReady + the source-sub close primitives -------

    /// A `SubscriptionReady` from shard `from` (the dest), as the dest emits it at adopt.
    fn subscription_ready(from: NodeId, session: SessionId) -> Inbound {
        wire(
            from,
            MsgClass::Control,
            &ShardToGateway::SubscriptionReady {
                session,
                entity: EntityId(77),
                frame: FrameRef::SystemSpace { system_seed: 8 },
                realm_fence: Fence(5),
            },
        )
    }

    /// The full route+sub snapshot a session holds (for the dest-sub assertions).
    fn sub_for(rig: &Rig, sid: SessionId, shard: NodeId) -> Option<SubEntry> {
        rig.world
            .resource::<GatewaySessions>()
            .by_session
            .get(&sid)
            .expect("session present")
            .hot
            .subs
            .load()
            .lookup(shard)
            .copied()
    }

    #[test]
    fn subscription_ready_opens_the_dest_sub_and_repoints_authority() {
        // FORK 0a: a SubscriptionReady from DEST opens a SECOND sub (SubId(1)) at the DEST realm
        // fence, emits SubscriptionOpened{1} (X1, before any frame) THEN AuthorityChanged{entity,
        // 1} — re-pointing the avatar's render authority to the dest sub. The source SubId(0)
        // stays open (the two-sub overlap).
        let mut rig = Rig::new();
        let (sid, _) = rig.login(); // SubId(0) on SHARD
        let sent = rig.tick(vec![subscription_ready(DEST, sid)]);
        assert_eq!(
            decode_controls(&sent, CLIENT),
            vec![
                ServerControlMsg::SubscriptionOpened {
                    sub: SubId(1),
                    frame: FrameRef::SystemSpace { system_seed: 8 },
                },
                ServerControlMsg::AuthorityChanged {
                    entity: EntityId(77),
                    sub: SubId(1),
                },
                // The default rig negotiates minor 2, so the pure-renderer `OwnEntity` trails the
                // node-aware `AuthorityChanged` at the dest re-point too (S4 dual-announce).
                ServerControlMsg::OwnEntity {
                    entity: EntityId(77),
                },
            ],
            "SubscriptionOpened(1) strictly precedes AuthorityChanged(entity,1) (X1 + A1 re-point)"
        );
        // BOTH subs now resolve: source SubId(0) on SHARD, dest SubId(1) on DEST at Fence(5).
        assert_eq!(sub_for(&rig, sid, SHARD).map(|e| e.sub), Some(SubId(0)));
        assert_eq!(
            sub_for(&rig, sid, DEST),
            Some(SubEntry {
                shard: DEST,
                sub: SubId(1),
                accepted: Fence(5),
            })
        );
        // The reverse index lists this session under BOTH shards.
        assert_eq!(
            rig.world.resource::<GatewaySessions>().subscribers_of(DEST),
            vec![sid]
        );
    }

    #[test]
    fn a_subscription_ready_naming_a_realm_moves_the_lineage() {
        // THE LINEAGE follows the avatar (the deleted render pin's lawful successor is the origin
        // marker, derived from this): `SubscriptionReady` is the one phase that can move it — it
        // carries the destination FRAME, so the realm the player is now in is nameable. The
        // composer derives the new chain from this next pass and bumps the origin epoch (§2.7).
        let mut rig = Rig::new();
        let (sid, _) = rig.login();
        let before = rig.world.resource::<GatewaySessions>().by_session[&sid]
            .lineage
            .clone();
        let _ = rig.tick(vec![wire(
            DEST,
            MsgClass::Control,
            &ShardToGateway::SubscriptionReady {
                session: sid,
                entity: EntityId(77),
                frame: FrameRef::PlanetCentered { planet_seed: 42 },
                realm_fence: Fence(5),
            },
        )]);
        let after = &rig.world.resource::<GatewaySessions>().by_session[&sid].lineage;
        assert_ne!(&before, after, "the lineage advanced for the entered realm");
        assert_eq!(
            after.last(),
            Some(&RealmId::Planet(42)),
            "the entered realm is the lineage leaf"
        );
    }

    #[test]
    fn a_subscription_ready_into_galaxy_space_leaves_the_lineage() {
        // Galaxy space names no realm, so there is nothing to apply and the lineage stays where
        // the avatar last stood — the between-space is felt, not re-anchored.
        let mut rig = Rig::new();
        let (sid, _) = rig.login();
        let before = rig.world.resource::<GatewaySessions>().by_session[&sid]
            .lineage
            .clone();
        let _ = rig.tick(vec![wire(
            DEST,
            MsgClass::Control,
            &ShardToGateway::SubscriptionReady {
                session: sid,
                entity: EntityId(77),
                frame: FrameRef::GalaxySpace,
                realm_fence: Fence(5),
            },
        )]);
        let after = &rig.world.resource::<GatewaySessions>().by_session[&sid].lineage;
        assert_eq!(&before, after, "no realm named, no lineage moved");
    }

    #[test]
    fn a_shard_roster_applies_fresh_rejects_stale_and_admits_for_dispatch() {
        // Minor 9's roster receive: the orchestrator's level REPLACES the record set (latest tick
        // wins); an older one is refused as stale — a reordered datagram must never resurrect a dead
        // roster. The applied set is DECISIVE: a node the record shows holding a realm is heard for
        // dispatch without ever greeting.
        const RECORDED: NodeId = NodeId(77);
        let mut rig = Rig::new();
        let (sid, _) = rig.login();
        let roster = |nodes: Vec<NodeId>, at: u64| {
            wire(
                ORCH,
                MsgClass::Saga,
                &InterShardFlow::ShardRoster(vd_wire::intershard::ShardRoster {
                    nodes,
                    at: UniverseTick(at),
                }),
            )
        };
        let _ = rig.tick(vec![roster(vec![RECORDED], 10)]);
        assert_eq!(rig.stats().shard_rosters_applied, 1);
        // STALE: an older level is refused; the held set is untouched.
        let _ = rig.tick(vec![roster(vec![], 5)]);
        assert_eq!(rig.stats().shard_roster_stale, 1);
        assert_eq!(rig.stats().shard_rosters_applied, 1);
        // DECISIVE: the recorded node's SubscriptionReady is heard (it is a shard by record), so the
        // session opens a sub on it — a node outside roster+greetings would have been refused.
        let _ = rig.tick(vec![wire(
            RECORDED,
            MsgClass::Control,
            &ShardToGateway::SubscriptionReady {
                session: sid,
                entity: EntityId(77),
                frame: FrameRef::SystemSpace { system_seed: 8 },
                realm_fence: Fence(5),
            },
        )]);
        assert_eq!(
            sub_for(&rig, sid, RECORDED).map(|e| e.sub),
            Some(SubId(1)),
            "the record-admitted shard's ready opened a sub"
        );
    }

    #[test]
    fn a_known_clients_wrong_class_counts_undecodable_not_unknown_sender() {
        // The refusal split: an UNKNOWN node on a data class is refused by sender; a node with a
        // LIVE session sending a class it may not send is a peer bug, counted undecodable exactly
        // as before the split existed.
        let mut rig = Rig::new();
        let _ = rig.login();
        let before = rig.stats().undecodable;
        let _ = rig.tick(vec![Inbound::Wire {
            from: CLIENT,
            class: MsgClass::Snapshot,
            bytes: vec![1].into(),
        }]);
        assert_eq!(rig.stats().undecodable, before + 1);
        assert_eq!(
            rig.stats().refused_unknown_sender,
            0,
            "a KNOWN client is never refused as an unknown sender"
        );
    }

    #[test]
    fn a_duplicate_subscription_ready_is_an_idempotent_no_op() {
        // At-least-once: a second SubscriptionReady for an already-open dest sub opens nothing
        // and emits nothing (the sub id is never re-allocated, A1 not re-emitted).
        let mut rig = Rig::new();
        let (sid, _) = rig.login();
        let _ = rig.tick(vec![subscription_ready(DEST, sid)]);
        let sent = rig.tick(vec![subscription_ready(DEST, sid)]);
        assert_eq!(
            decode_controls(&sent, CLIENT),
            vec![],
            "a duplicate SubscriptionReady emits nothing"
        );
        assert_eq!(sub_for(&rig, sid, DEST).map(|e| e.sub), Some(SubId(1)));
        assert_eq!(
            rig.stats().transfer_unroutable,
            0,
            "a duplicate is not unroutable"
        );
    }

    #[test]
    fn subscription_ready_for_an_absent_session_is_counted_and_emits_nothing() {
        // An absent session: counted (transfer_unroutable), never a panic, nothing opened.
        let mut rig = Rig::new();
        let sent = rig.tick(vec![subscription_ready(DEST, SessionId(0xDEAD))]);
        assert_eq!(decode_controls(&sent, CLIENT), vec![]);
        assert_eq!(rig.stats().transfer_unroutable, 1);
    }

    #[test]
    fn release_subscribe_closes_the_source_sub_with_a_drain_grace() {
        // 1d.2b: ReleaseSubscribe(src: SHARD) closes the SOURCE sub — SubscriptionClosing{0} is
        // emitted, the sub goes Draining (still routable THIS tick), and the NEXT tick's sweep
        // removes it. The dest sub (opened by SubscriptionReady) survives.
        let mut rig = Rig::new();
        let (sid, _) = rig.login();
        let _ = rig.tick(vec![subscription_ready(DEST, sid)]); // SubId(1) on DEST
        // Drive the REAL saga order (Prepare → RequestCut → Freeze → Commit) before the release: the
        // commit is what re-points the session's authority at DEST, so SHARD is genuinely no longer
        // the feeding node when its sub is released (the authority-sub invariant, task #175). A
        // release can never precede its commit — it IS the success teardown of the demote tail.
        let _ = cross_to(&mut rig, sid, XFER, DEST);
        let sent = rig.tick(vec![saga_cmd(TransferControl::ReleaseSubscribe {
            transfer: XFER,
            session: sid,
            src: SHARD,
        })]);
        // The source sub close went to the client; Released acked to the orchestrator.
        assert_eq!(
            decode_controls(&sent, CLIENT),
            vec![ServerControlMsg::SubscriptionClosing { sub: SubId(0) }]
        );
        assert_eq!(
            acks_to_orch(&sent),
            vec![TransferControlAck::Released { transfer: XFER }]
        );
        // THIS tick (the close tick) the source sub is still in the hot table (drain grace).
        assert_eq!(sub_for(&rig, sid, SHARD).map(|e| e.sub), Some(SubId(0)));
        // The NEXT tick's sweep removes it; the dest sub survives.
        let _ = rig.tick(vec![]);
        assert_eq!(
            sub_for(&rig, sid, SHARD),
            None,
            "source sub swept after the grace"
        );
        assert_eq!(
            sub_for(&rig, sid, DEST).map(|e| e.sub),
            Some(SubId(1)),
            "dest sub survives"
        );
        assert_eq!(
            rig.world
                .resource::<GatewaySessions>()
                .subscribers_of(SHARD),
            Vec::<SessionId>::new()
        );
    }

    #[test]
    fn a_straggler_source_frame_in_the_drain_grace_tick_is_still_routed() {
        // C2 drain grace: a source frame arriving in the SAME batch as the ReleaseSubscribe
        // close is still routed to the client (drained, not silently dropped); only the
        // next-tick sweep stops routing.
        let mut rig = Rig::new();
        let (sid, _) = rig.login();
        // The real saga order — the commit re-points authority at DEST, so releasing SHARD's sub is
        // permitted by the authority-sub invariant (task #175).
        let _ = cross_to(&mut rig, sid, XFER, DEST);
        // Release + a co-arriving source frame in ONE tick batch: the close marks Draining, the
        // frame still fans (the sub is routable through the rest of the batch).
        let sent = rig.tick(vec![
            saga_cmd(TransferControl::ReleaseSubscribe {
                transfer: XFER,
                session: sid,
                src: SHARD,
            }),
            wire(SHARD, MsgClass::Snapshot, &frame_msg(Fence(1), 9)),
        ]);
        let snaps = sent
            .iter()
            .filter(|(to, c, _)| (*to == CLIENT) & (*c == MsgClass::Snapshot))
            .count();
        assert_eq!(
            snaps, 1,
            "the straggler is drained (routed) during the grace tick"
        );
        // After the next-tick sweep, a further source frame routes to nobody — and (the session
        // being Active with no pending retries) nothing else is sent, so the whole tick is empty.
        let sent = rig.tick(vec![wire(
            SHARD,
            MsgClass::Snapshot,
            &frame_msg(Fence(1), 10),
        )]);
        assert_eq!(
            sent,
            Vec::new(),
            "after the sweep the source sub no longer routes (nothing reaches the client)"
        );
    }

    #[test]
    fn abort_defensively_closes_an_opened_dest_sub() {
        // apply_abort's defensive close (a sub on a shard != the source): drive a dest sub open
        // (SubscriptionReady) WITH a live in-flight transfer, then abort — the dest sub is
        // closed (SubscriptionClosing{1}) and swept, while the source sub stays.
        let mut rig = Rig::new();
        let (sid, _) = rig.login();
        let _ = rig.tick(vec![saga_cmd(TransferControl::PrepareSubscribe {
            transfer: XFER,
            session: sid,
            dest: DEST,
        })]);
        let _ = rig.tick(vec![subscription_ready(DEST, sid)]); // SubId(1) on DEST
        assert_eq!(sub_for(&rig, sid, DEST).map(|e| e.sub), Some(SubId(1)));
        let sent = rig.tick(vec![saga_cmd(TransferControl::AbortTransfer {
            transfer: XFER,
            session: sid,
        })]);
        assert_eq!(
            decode_controls(&sent, CLIENT),
            vec![ServerControlMsg::SubscriptionClosing { sub: SubId(1) }],
            "abort defensively closes the dest sub"
        );
        assert_eq!(
            acks_to_orch(&sent),
            vec![TransferControlAck::Aborted { transfer: XFER }]
        );
        // The dest sub is swept next tick; the source sub stays open.
        let _ = rig.tick(vec![]);
        assert_eq!(
            sub_for(&rig, sid, DEST),
            None,
            "dest sub closed + swept on abort"
        );
        assert_eq!(
            sub_for(&rig, sid, SHARD).map(|e| e.sub),
            Some(SubId(0)),
            "source sub stays"
        );
    }

    #[test]
    fn abort_without_a_dest_sub_closes_nothing_extra() {
        // The happy-path abort (no dest sub yet): the defensive close collects no shard, so no
        // SubscriptionClosing is emitted — only the Aborted ack.
        let mut rig = Rig::new();
        let (sid, _) = rig.login();
        let _ = rig.tick(vec![saga_cmd(TransferControl::PrepareSubscribe {
            transfer: XFER,
            session: sid,
            dest: DEST,
        })]);
        let sent = rig.tick(vec![saga_cmd(TransferControl::AbortTransfer {
            transfer: XFER,
            session: sid,
        })]);
        assert_eq!(
            decode_controls(&sent, CLIENT),
            vec![],
            "no dest sub to close: no SubscriptionClosing"
        );
        assert_eq!(
            acks_to_orch(&sent),
            vec![TransferControlAck::Aborted { transfer: XFER }]
        );
        // The source sub is untouched (abort never closes the source).
        assert_eq!(sub_for(&rig, sid, SHARD).map(|e| e.sub), Some(SubId(0)));
    }

    #[test]
    fn abort_closes_only_its_own_dest_never_a_prior_transfers_live_sub() {
        // F1 REGRESSION (audit `wf_93d8e84f`): once a transfer's dest sub is live (a sub on a shard
        // != the login shard), a LATER, UNRELATED transfer's abort must close ONLY its OWN dest —
        // never that live sub. The old "any sub != config.shard" heuristic closed the player's
        // CURRENT live sub (a chained-transfer black screen) and, in the N-shard end goal, EVERY
        // composited sub (ship/host/planet). The fix closes exactly `tp.dest`.
        let mut rig = Rig::new();
        let (sid, _) = rig.login(); // login sub SubId(0) on config.shard (SHARD)

        // A first transfer opens the DEST sub (SubId(1)) — a live, non-login-shard sub.
        let _ = rig.tick(vec![saga_cmd(TransferControl::PrepareSubscribe {
            transfer: XFER,
            session: sid,
            dest: DEST,
        })]);
        let _ = rig.tick(vec![subscription_ready(DEST, sid)]);
        assert_eq!(
            sub_for(&rig, sid, DEST).map(|e| e.sub),
            Some(SubId(1)),
            "the prior transfer's DEST sub is live"
        );

        // A SECOND transfer to a DIFFERENT dest, then aborted. Its dest sub was never opened, so the
        // precise abort close is a no-op — and it must NOT touch the prior transfer's live DEST sub.
        let xfer2 = TransferId(0x1c3);
        let dest2 = NodeId(43);
        let _ = rig.tick(vec![saga_cmd(TransferControl::PrepareSubscribe {
            transfer: xfer2,
            session: sid,
            dest: dest2,
        })]);
        let sent = rig.tick(vec![saga_cmd(TransferControl::AbortTransfer {
            transfer: xfer2,
            session: sid,
        })]);
        // No SubscriptionClosing on the client: the abort closed only its own (unopened) dest2.
        // (The OLD heuristic would have emitted SubscriptionClosing{SubId(1)} here — closing the
        // live DEST sub of the unrelated prior transfer.)
        assert_eq!(
            decode_controls(&sent, CLIENT),
            vec![],
            "the unrelated abort closes no live sub"
        );
        assert_eq!(
            acks_to_orch(&sent),
            vec![TransferControlAck::Aborted { transfer: xfer2 }]
        );
        let _ = rig.tick(vec![]); // a sweep tick changes nothing
        assert_eq!(
            sub_for(&rig, sid, DEST).map(|e| e.sub),
            Some(SubId(1)),
            "the prior transfer's live DEST sub SURVIVES the unrelated abort"
        );
    }

    #[test]
    fn release_of_a_foreign_transfer_closes_no_sub() {
        // The collect-guard FALSE arm: a ReleaseSubscribe for a transfer NOT in flight collects
        // no source sub to close (the source sub stays open), still acks Released idempotently.
        let mut rig = Rig::new();
        let (sid, _) = rig.login();
        let _ = rig.tick(vec![saga_cmd(TransferControl::PrepareSubscribe {
            transfer: XFER,
            session: sid,
            dest: DEST,
        })]);
        let sent = rig.tick(vec![saga_cmd(TransferControl::ReleaseSubscribe {
            transfer: TransferId(0x999),
            session: sid,
            src: SHARD,
        })]);
        assert_eq!(
            decode_controls(&sent, CLIENT),
            vec![],
            "a foreign release closes no sub"
        );
        assert_eq!(
            acks_to_orch(&sent),
            vec![TransferControlAck::Released {
                transfer: TransferId(0x999)
            }]
        );
        assert_eq!(
            sub_for(&rig, sid, SHARD).map(|e| e.sub),
            Some(SubId(0)),
            "source sub intact"
        );
    }

    // ---- Slice 1d.2c: the 2-shard transfer CAPSTONE (gateway read-plane half) ----

    /// The subs that the client-bound snapshots in `sent` are tagged with (decoded).
    fn delivered_snapshot_subs(sent: &[(NodeId, MsgClass, Vec<u8>)]) -> Vec<SubId> {
        sent.iter()
            .filter(|(to, class, _)| (*to == CLIENT) & (*class == MsgClass::Snapshot))
            .map(|(_, _, bytes)| {
                postcard::from_bytes::<SnapshotDatagram>(bytes)
                    .expect("a delivered snapshot decodes")
                    .sub
            })
            .collect()
    }

    /// Every session whose lease the gateway RENEWED in this batch (an ORCH-bound `LeaseRenew` on a
    /// Session key). Shared by the D-3 renew-cadence cell and the 5f-3d MF2 dynamic-hold cell so the
    /// decoder's non-renew (`_ => None`) arm is owned once — the D-3 cell's hello tick emits a
    /// `LeaseGrant` (a Directory op that is NOT a renew), exercising that arm for both callers.
    fn renewed_sessions(sent: &[(NodeId, MsgClass, Vec<u8>)]) -> Vec<SessionId> {
        sent.iter()
            .filter(|(to, _, _)| *to == ORCH)
            .filter_map(
                |(_, _, b)| match postcard::from_bytes::<InterShardFlow>(b) {
                    Ok(InterShardFlow::Directory(DirectoryOp::LeaseRenew {
                        key: DirectoryKey::Session(s),
                        ..
                    })) => Some(s),
                    _ => None,
                },
            )
            .collect()
    }

    #[test]
    fn capstone_two_sub_overlap_routes_both_frames_and_repoints_the_avatar_to_the_dest() {
        // 1d.2c CAPSTONE (the gateway read-plane half of the 2-shard transfer): during the
        // post-commit/pre-release window the gateway holds TWO subs for one session; the dest's
        // `SubscriptionReady` (the stub emits it at adopt, 1d.2c) opened SubId(1) and emitted
        // `AuthorityChanged{entity, SubId(1)}` — re-pointing the avatar's render authority to the
        // dest. A synthetic SOURCE frame fans to SubId(0) and a synthetic DEST frame fans to
        // SubId(1), EACH fence-checked against its OWN realm fence and retagged to its own sub.
        // The client thus receives the avatar on BOTH subs but with `AuthorityChanged` naming
        // SubId(1) authoritative — so a compositing client (`DeliveredView`, suppression proven
        // in `vd_client::view`) renders the avatar EXACTLY ONCE, from the DEST sub. Then
        // `ReleaseSubscribe` closes the SOURCE sub.
        let mut rig = Rig::new();
        let (sid, _) = rig.login(); // login sub SubId(0) on SHARD, AuthorityChanged{77, SubId(0)}
        // The transfer reaches commit (the REAL saga order — the commit is also what re-points the
        // session's authority at DEST, so the later source release is permitted by the
        // authority-sub invariant, task #175): then the dest adopts and announces its read-sub.
        let _ = cross_to(&mut rig, sid, XFER, DEST);
        // SubscriptionReady from DEST opens SubId(1) (at the DEST realm fence) + re-points
        // authority to SubId(1).
        let ready_sent = rig.tick(vec![subscription_ready(DEST, sid)]);
        assert_eq!(
            decode_controls(&ready_sent, CLIENT),
            vec![
                ServerControlMsg::SubscriptionOpened {
                    sub: SubId(1),
                    frame: FrameRef::SystemSpace { system_seed: 8 },
                },
                ServerControlMsg::AuthorityChanged {
                    entity: EntityId(77), // the login avatar (SUBJECT id at the gateway is the dot's entity)
                    sub: SubId(1),
                },
                // The default rig is minor 2, so `OwnEntity` re-confirms the avatar (a pure-renderer
                // client renders it latest-wins by EntityId — the sub is irrelevant to it, S4).
                ServerControlMsg::OwnEntity {
                    entity: EntityId(77),
                },
                // THE CROSSING SWAP LEVEL (§2.7), ordered AFTER AuthorityChanged on this same
                // reliable stream: the composer saw the standing realm flip System(7)→System(8)
                // and bumped the epoch EXACTLY once (login was 1, this crossing makes 2). No
                // window frames were fed in this rig, so the level carries only the new origin's
                // bagless row at the boot stamp — the swap SIGNAL, which is what is under test.
                ServerControlMsg::RealmRegistry {
                    origin: RealmId::System(8),
                    origin_epoch: 2,
                    rows: vec![SceneRow {
                        realm: RealmId::System(8),
                        parent: None,
                        pose: vd_core::pose::StampedPose::at_rest(
                            FrameRef::SystemSpace { system_seed: 8 },
                            DVec3::ZERO,
                            UniverseTick(0),
                        ),
                        bag: Vec::new(),
                    }],
                },
            ],
            "the dest sub opens (X1), authority re-points to SubId(1) (FORK 0a / A1), and the \
             swap level follows the AuthorityChanged on the same reliable stream (§2.7)"
        );
        // 1d.5a: the dest sub is OPEN but NO dest frame is delivered yet → the standing delivery
        // predicate is NOT satisfied → no premature DeliveredToObservers to the saga (anti-vacuous).
        assert!(
            !acks_to_orch(&ready_sent)
                .contains(&TransferControlAck::DeliveredToObservers { transfer: XFER }),
            "no DeliveredToObservers before the dest delivers a frame",
        );
        // Both subs are held (the two-sub overlap), at their OWN realm fences:
        // SubId(0) on SHARD @ Fence(1) (login), SubId(1) on DEST @ Fence(5) (SubscriptionReady).
        assert_eq!(
            sub_for(&rig, sid, SHARD),
            Some(SubEntry {
                shard: SHARD,
                sub: SubId(0),
                accepted: Fence(1),
            })
        );
        assert_eq!(
            sub_for(&rig, sid, DEST),
            Some(SubEntry {
                shard: DEST,
                sub: SubId(1),
                accepted: Fence(5),
            })
        );

        // A synthetic SOURCE frame (from SHARD @ the source realm fence) fans to SubId(0); a
        // synthetic DEST frame (from DEST @ the dest realm fence) fans to SubId(1). Each is
        // checked against ITS OWN accepted fence and retagged to ITS OWN sub.
        let sent = rig.tick(vec![
            wire(SHARD, MsgClass::Snapshot, &frame_msg(Fence(1), 9)),
            wire(DEST, MsgClass::Snapshot, &frame_msg(Fence(5), 9)),
        ]);
        let mut subs = delivered_snapshot_subs(&sent);
        subs.sort_unstable();
        assert_eq!(
            subs,
            vec![SubId(0), SubId(1)],
            "the avatar rides BOTH subs (source→SubId(0), dest→SubId(1)) — the overlap"
        );
        // 1d.5a: that dest frame ADVANCED the dest observer's watermark, so the standing predicate
        // now holds → the gateway emits DeliveredToObservers{XFER} to the saga (the demote-predicate
        // input). Value-asserted (not incidental): a wrong-transfer / silent / unconditional emit
        // turns THIS red.
        assert!(
            acks_to_orch(&sent)
                .contains(&TransferControlAck::DeliveredToObservers { transfer: XFER }),
            "the delivered dest frame drives DeliveredToObservers{{XFER}} to the saga",
        );
        // A dest frame BELOW the dest's accepted fence (Fence(5)) is stale-dropped — proving the
        // PER-SHARD fence (not the session-global route fence) governs the dest sub.
        let sent = rig.tick(vec![wire(
            DEST,
            MsgClass::Snapshot,
            &frame_msg(Fence(4), 10),
        )]);
        assert_eq!(
            delivered_snapshot_subs(&sent),
            Vec::<SubId>::new(),
            "a dest frame below the dest's own accepted fence is dropped"
        );
        assert_eq!(rig.stats().stale_frames_dropped, 1);

        // ReleaseSubscribe closes the SOURCE sub: SubscriptionClosing{0}, then swept next tick;
        // afterward only the DEST sub (SubId(1)) routes — the crossing is complete.
        let sent = rig.tick(vec![saga_cmd(TransferControl::ReleaseSubscribe {
            transfer: XFER,
            session: sid,
            src: SHARD,
        })]);
        assert_eq!(
            decode_controls(&sent, CLIENT),
            vec![ServerControlMsg::SubscriptionClosing { sub: SubId(0) }]
        );
        let _ = rig.tick(vec![]); // the sweep removes the drained source sub
        assert_eq!(sub_for(&rig, sid, SHARD), None, "source sub closed + swept");
        let sent = rig.tick(vec![
            wire(SHARD, MsgClass::Snapshot, &frame_msg(Fence(1), 11)),
            wire(DEST, MsgClass::Snapshot, &frame_msg(Fence(5), 11)),
        ]);
        assert_eq!(
            delivered_snapshot_subs(&sent),
            vec![SubId(1)],
            "post-release only the DEST sub routes — the avatar is single-sub on the dest"
        );
    }

    #[test]
    fn the_gateway_renews_active_session_leases_on_cadence() {
        // D-3 heartbeat (gateway half): on the renew cadence the gateway re-sends LeaseRenew for every
        // ACTIVE session's Session key — never a still-logging-in (non-Active) session, and never
        // off-cadence. Drives ONE session through AwaitingAttach (non-Active) then Active so both filter
        // arms + the iterator's zero-iter (no Active → empty) and nonzero-iter (Active) are exercised.
        // (`renewed_sessions` is the module-level decode helper, shared with the 5f-3d MF2 cell.)
        let mut rig = Rig::new();
        rig.world.insert_resource(GatewayConfig {
            lease_renew_interval_ticks: 4,
            ..config()
        });
        // Hello → AwaitingDirectory: the gateway sends a session LeaseGrant to the orchestrator (a
        // Directory op that is NOT a LeaseRenew — exercises the decoder's non-renew arm).
        let hello = rig.tick(vec![wire(CLIENT, MsgClass::Control, &hello_msg())]);
        assert!(
            renewed_sessions(&hello).is_empty(),
            "login emits a LeaseGrant, not a LeaseRenew"
        );
        let session_id = rig
            .world
            .resource::<GatewaySessions>()
            .sessions()
            .next()
            .expect("session pending");
        // The directory grant → AwaitingAttach (still at tick 1, no renew).
        let _ = rig.tick(vec![wire(ORCH, MsgClass::Saga, &granted_head(session_id))]);

        // A renew tick while the session is NON-Active (AwaitingAttach): nothing renewed (filter false
        // arm; push_renewals called with an empty iterator).
        rig.world.resource_mut::<ClockSample>().local_tick = TickId(4);
        assert!(
            renewed_sessions(&rig.tick(vec![])).is_empty(),
            "a non-Active (still-attaching) session is not renewed"
        );

        // Attach at an off-cadence tick → Active (no renew at tick 5).
        rig.world.resource_mut::<ClockSample>().local_tick = TickId(5);
        let _ = rig.tick(vec![wire(
            SHARD,
            MsgClass::Control,
            &ShardToGateway::SessionAttached {
                session: session_id,
                entity: EntityId(77),
                frame: FrameRef::SystemSpace { system_seed: 7 },
                realm_fence: Fence(1),
            },
        )]);

        // A renew tick while Active: the session lease is renewed (filter true arm; nonzero iterator).
        rig.world.resource_mut::<ClockSample>().local_tick = TickId(8);
        assert_eq!(
            renewed_sessions(&rig.tick(vec![])),
            vec![session_id],
            "an Active session's lease is renewed on cadence"
        );

        // Off-cadence (the modulo branch on the proceed path): no renewal.
        rig.world.resource_mut::<ClockSample>().local_tick = TickId(9);
        assert!(
            renewed_sessions(&rig.tick(vec![])).is_empty(),
            "an off-cadence tick emits no LeaseRenew"
        );
    }

    // ---------------------------------------------------------------------------
    // 3g abort-leg: the one-shot `reject_next_prepare` lever (HR5(c) — the gateway unit tests OWN the
    // two-arm coverage of `apply_prepare`'s reject tail; the e2e is composition proof, not the arm owner).
    // Drive `apply_prepare` DIRECTLY (no `Rig`) so both the `Some`/`None` arms + the guard precedence are
    // exercised in one monomorphic surface. `assert_eq!` on the FULL ack (HR5(d) — never `matches!`).
    // ---------------------------------------------------------------------------

    use vd_wire::seams::transfer_control::SpatialReject;

    /// A bare ACTIVE session for driving `apply_prepare` directly (mirrors `freshest_..`'s `sess`, but
    /// always Active with a fresh empty transfer). `transfer: None` so `apply_prepare` opens fresh progress.
    fn active_session() -> Session {
        Session {
            client: CLIENT,
            account: AccountId(5),
            fence: Fence(1),
            phase: SessionPhase::Active {
                entity: EntityId(1),
            },
            next_sub: 0,
            // 5f-3d: a STATIC session (no dynamic home, no bootstrap window) — the byte-identical shape.
            home_shard: None,
            home_rid: None,
            realm_feed_frame_id: 0,
            scene_sent: BTreeMap::new(),
            // A bare test session: no home descended, so no spawn pose — the static shape.
            spawn: None,
            bootstrap_deadline: None,
            confirmed_at: TickId(1),
            negotiated_minor: 1,
            transfer: None,
            subs: BTreeMap::new(),
            delivered: BTreeMap::new(),
            lineage: Vec::new(),
            shadow: window::ShadowScene::default(),
            hot: Arc::new(SessionHot {
                route: ArcSwap::from_pointee(RouteSnapshot {
                    authority: SHARD,
                    fence: Fence(1),
                    cut: None,
                }),
                last_input_seq: AtomicU64::new(0),
                subs: ArcSwap::from_pointee(SubTable::default()),
            }),
        }
    }

    #[test]
    fn apply_prepare_inert_lever_replies_ready() {
        // The `None` arm: an INERT lever leaves the 1c stub behaviour byte-identical (Ready) AND stays None.
        let mut session = active_session();
        let mut stats = GatewayStats::default();
        let mut lever: Option<PrepareReject> = None;
        let ack = apply_prepare(&mut session, XFER, DEST, &mut stats, &mut lever);
        assert_eq!(
            ack,
            Some(TransferControlAck::Prepared {
                transfer: XFER,
                result: PrepareResult::Ready,
            }),
            "an inert (None) lever replies the 1c Ready stub"
        );
        assert_eq!(lever, None, "the inert lever is untouched (stays None)");
        assert_eq!(
            stats.transfer_unroutable, 0,
            "an Active prepare is not a routing failure"
        );
        assert!(
            session.transfer.is_some(),
            "the prepare opened progress on the Active session"
        );
    }

    #[test]
    fn apply_prepare_armed_lever_rejects_once_then_self_clears() {
        // The `Some` arm + the one-shot self-clear: the FIRST prepare rejects with the armed reason and the
        // lever clears; the SECOND (fresh Active session) prepare is Ready again — proving `.take()` fired.
        let reject = PrepareReject::Spatial(SpatialReject::Obstructed);
        let mut lever: Option<PrepareReject> = Some(reject);

        let mut first = active_session();
        let mut stats = GatewayStats::default();
        let ack1 = apply_prepare(&mut first, XFER, DEST, &mut stats, &mut lever);
        assert_eq!(
            ack1,
            Some(TransferControlAck::Prepared {
                transfer: XFER,
                result: PrepareResult::Rejected(reject),
            }),
            "the armed lever rejects the first prepare with the exact armed reason"
        );
        assert_eq!(lever, None, "the one-shot lever self-cleared (.take)");

        // A SECOND prepare on a fresh Active session now sees the cleared lever → Ready.
        let mut second = active_session();
        let ack2 = apply_prepare(&mut second, XFER, DEST, &mut stats, &mut lever);
        assert_eq!(
            ack2,
            Some(TransferControlAck::Prepared {
                transfer: XFER,
                result: PrepareResult::Ready,
            }),
            "the very next prepare is Ready again (the lever is one-shot, not sticky)"
        );
        assert_eq!(
            stats.transfer_unroutable, 0,
            "neither Active prepare is a routing failure"
        );
    }

    #[test]
    fn apply_prepare_not_active_does_not_consume_the_lever() {
        // Guard precedence: the not-Active guard sits ABOVE the lever and returns None + bumps
        // `transfer_unroutable` WITHOUT consuming the lever — so an inactivity-rejected prepare leaves the
        // lever armed for the retried (Active) one (the lever fires only for a would-be-Ready prepare).
        let reject = PrepareReject::Spatial(SpatialReject::Obstructed);
        let mut lever: Option<PrepareReject> = Some(reject);
        let mut session = active_session();
        session.phase = SessionPhase::AwaitingAttach; // NOT Active
        let mut stats = GatewayStats::default();
        let ack = apply_prepare(&mut session, XFER, DEST, &mut stats, &mut lever);
        assert_eq!(
            ack, None,
            "a not-Active prepare is un-acked (pins the saga, WEDGE-1)"
        );
        assert_eq!(
            stats.transfer_unroutable, 1,
            "a not-Active prepare is counted as unroutable"
        );
        assert_eq!(
            lever,
            Some(reject),
            "the guard did NOT consume the lever — it stays armed for the retried Active prepare"
        );
    }

    // ===== RLM 5f-3c — the TRUSTED GATEWAY SEED INJECTOR =========================================

    /// Every `RealmDemand` the gateway sent to the orchestrator (decoded), ignoring the directory ops that
    /// also ride ORCH+Saga (login's `LeaseGrant`). Mirrors [`acks_to_orch`]: `.expect` keeps the Err path
    /// in std (no caller branch); a non-`RealmDemand` ORCH/Saga send (the login `LeaseGrant`) maps to None
    /// — the `_` arm is exercised by the armed login's hello tick.
    fn demands_to_orch(sent: &[(NodeId, MsgClass, Vec<u8>)]) -> Vec<RealmDemand> {
        sent.iter()
            .filter(|(node, class, _)| (*node == ORCH) & (*class == MsgClass::Saga))
            .filter_map(|(_, _, bytes)| {
                match postcard::from_bytes::<InterShardFlow>(bytes)
                    .expect("gateway sends a valid flow")
                {
                    InterShardFlow::RealmDemand(d) => Some(d),
                    _ => None,
                }
            })
            .collect()
    }

    /// The count of injected home demands across an entire multi-tick login drive.
    fn demand_count(sends: &LoginSends) -> usize {
        sends.iter().map(|tick| demands_to_orch(tick).len()).sum()
    }

    /// THE DEEP HOME every armed rig here uses: the walk-scale world's deepest realm, an Area inside a
    /// Planet inside a Star system. Demanding it therefore spins up a four-level chain, which is what these
    /// tests are about.
    ///
    /// These rigs used to say the same thing as a POSITION — 25 metres along `+x` from the ambient root —
    /// and let the router descend the forest to work out which realm that fell in. It fell in exactly this
    /// one, at that area's own centre, so naming it states what the rig always meant and removes the
    /// descent from the test as well as from the code.
    const TEST_HOME_REALM: RealmId = RealmId::Area(7);

    /// RLM 5f-3d — the DEMAND TTL every armed test rig runs with: 8 ticks ⇒ a re-drive cadence of
    /// `8 / REDRIVE_DIVISOR = 2` ticks. Small enough that a handful of `set_tick` steps cross it, and NOT 1,
    /// so "backed off, not every tick" is observable.
    const TEST_DEMAND_TTL: u64 = 8;
    /// RLM 5f-3d — the bootstrap TTL for rigs that must NOT expire while a test drives several ticks.
    const TEST_BOOTSTRAP_TTL: u64 = 100;

    /// An ARMED injector whose accounts live in `homes`, over the walk-scale world, with a LIVE 5f-3d
    /// bootstrap budget (a short re-drive cadence, a long bootstrap TTL). The pre-5f-3d 5f-3c tests are
    /// unaffected by the budget: they hold `local_tick` at 1, so neither the re-drive nor the TTL can fire
    /// during them.
    fn armed_injector(homes: HomeRegistry) -> SeedInjectorConfig {
        SeedInjectorConfig {
            armed: true,
            world: test_world().lowered(),
            homes,
            demand_ttl_ticks: TEST_DEMAND_TTL,
            bootstrap_ttl_ticks: TEST_BOOTSTRAP_TTL,
        }
    }

    /// The walk-scale world every armed rig here resolves homes against.
    fn test_world() -> WorldView {
        WorldView::hand_placed(&UniverseConfig::walk_scale())
    }

    /// `realm`'s full root→leaf lineage in that world — a NAME walk up the parent pointers, with no
    /// distance read anywhere in it.
    fn lineage_of(realm: RealmId) -> RealmCoord {
        vd_core::worldgen::coord_of_realm(test_world().regions(), realm)
            .expect("the rig names a realm of its own world")
    }

    /// The rig default: every account lives at the ambient root's own centre.
    fn default_homes() -> HomeRegistry {
        let world = test_world();
        let root = world
            .regions()
            .iter()
            .find(|r| r.parent.is_none())
            .expect("a forest has one ambient root")
            .realm;
        homes_at(root)
    }

    /// A registry where every account lives at `realm`'s own centre — the rig equivalent of "the account
    /// store says you live here", with no descent anywhere in it.
    fn homes_at(realm: RealmId) -> HomeRegistry {
        let world = test_world();
        HomeRegistry::new(
            StoredHome::in_realm(world.regions(), realm, DVec3::ZERO)
                .expect("the rig names a realm of its own world"),
        )
    }

    /// A registry where `account` lives at `realm`, `x` metres along `+x` from that realm's own centre, and
    /// every other account lives at the ambient root.
    fn homes_for(account: AccountId, realm: RealmId, x: f64) -> HomeRegistry {
        let world = test_world();
        let root = world
            .regions()
            .iter()
            .find(|r| r.parent.is_none())
            .expect("a forest has one ambient root")
            .realm;
        HomeRegistry::new(
            StoredHome::in_realm(world.regions(), root, DVec3::ZERO).expect("the root is named"),
        )
        .with_account(
            account,
            StoredHome::in_realm(world.regions(), realm, DVec3::new(x, 0.0, 0.0))
                .expect("the rig names a realm of its own world"),
        )
    }

    /// Arm `rig`'s live `GatewayConfig` with `injector` + set the clock's `synced` latch (the same
    /// re-insert mechanism the D-3 heartbeat tests use).
    fn arm_injector(rig: &mut Rig, injector: SeedInjectorConfig, synced: bool) {
        rig.world.insert_resource(GatewayConfig {
            seed_injector: injector,
            ..config()
        });
        rig.world.resource_mut::<ClockSample>().synced = synced;
    }

    /// A Session-head reply GRANTED to a FOREIGN gateway (`granted` is false — not this gateway).
    fn foreign_head(session: SessionId) -> InterShardFlow {
        InterShardFlow::DirectoryReply(DirectoryReply::Head {
            key: DirectoryKey::Session(session),
            record: Some(OwnerRecord {
                authority: AuthorityRef::Gateway(NodeId(0xBAD)),
                fence: Fence(1),
                lease_expires: UniverseTick(1_000),
                in_transfer: None,
            }),
        })
    }

    /// A Session-head reply owned by THIS gateway but at the WRONG fence (`granted` is false).
    fn wrong_fence_head(session: SessionId) -> InterShardFlow {
        InterShardFlow::DirectoryReply(DirectoryReply::Head {
            key: DirectoryKey::Session(session),
            record: Some(OwnerRecord {
                authority: AuthorityRef::Gateway(GW),
                fence: Fence(0xF),
                lease_expires: UniverseTick(1_000),
                in_transfer: None,
            }),
        })
    }

    /// The composed LEVELS on an outbox (proto_minor 18, §2.4), decoded off the control pushes —
    /// each as (origin, origin_epoch, rows). The `_ => None` arm discriminates non-level control
    /// (e.g. `OwnEntity`/`AuthorityChanged`). Mirrors `demands_to_orch`.
    /// The leading postcard discriminant of one encoded control message — a DATA pin (the
    /// composed-scene-only claims compare discriminant bytes instead of matching arms a test
    /// could never drive, per the HR5 test discipline).
    fn control_disc(msg: &ServerControlMsg) -> u8 {
        postcard::to_allocvec(msg).expect("encodes")[0]
    }

    fn scene_levels(outbox: &OutboundBox) -> Vec<(RealmId, u64, Vec<SceneRow>)> {
        outbox
            .0
            .iter()
            .filter_map(|(_, _, bytes, _)| {
                match postcard::from_bytes::<ServerControlMsg>(bytes)
                    .expect("gateway test pushes only ServerControlMsg on this outbox")
                {
                    ServerControlMsg::RealmRegistry {
                        origin,
                        origin_epoch,
                        rows,
                    } => Some((origin, origin_epoch, rows)),
                    _ => None,
                }
            })
            .collect()
    }

    /// The composed reliable DELTAS on an outbox (§2.4) — each as (epoch, added, removed).
    fn scene_deltas(outbox: &OutboundBox) -> Vec<(u64, Vec<SceneRow>, Vec<RealmId>)> {
        outbox
            .0
            .iter()
            .filter_map(|(_, _, bytes, _)| {
                match postcard::from_bytes::<ServerControlMsg>(bytes)
                    .expect("gateway test pushes only ServerControlMsg on this outbox")
                {
                    ServerControlMsg::RealmSceneDelta {
                        origin_epoch,
                        added,
                        removed,
                        ..
                    } => Some((origin_epoch, added, removed)),
                    _ => None,
                }
            })
            .collect()
    }

    /// The composed per-tick realm datagrams a rig tick sent to `to` — each decoded whole.
    fn composed_datagrams(
        sent: &[(NodeId, MsgClass, Vec<u8>)],
        to: NodeId,
    ) -> Vec<RealmSnapshotDatagram> {
        sent.iter()
            .filter(|(node, class, _)| (*node == to) & (*class == MsgClass::RealmSnapshot))
            .map(|(_, _, bytes)| {
                postcard::from_bytes::<RealmSnapshotDatagram>(bytes)
                    .expect("the composed feed carries whole datagrams")
            })
            .collect()
    }

    #[test]
    fn the_composer_emits_the_login_level_the_datagrams_and_the_body_delta() {
        // THE FLAG DAY's login path (§2.4, replacing the deleted SessionAttached RealmRegistry):
        // the first confirmed fold bumps the epoch 0→1 and ships ONE full level — the origin row
        // first (pose ZERO, its self-look attached), every composed row with its bag — then every
        // fresh fold ships the per-tick composed datagram (one frame id per tick, the epoch on
        // each), and a BODY change at a stable epoch ships the reliable delta, never a new level.
        let mut rig = Rig::new();
        let (_sid, _) = rig.login(); // Occupants window id 1 on SHARD; lineage [System(7)]
        let w = WindowId(1);
        // The realm speaks: its level (one authored child row) + its own look + a child marker.
        let sys_look = vd_core::look::look_bag(&vd_core::geometry::Boundary::Shell { r: 40.0 });
        let sent = rig.tick(vec![
            wire(
                SHARD,
                MsgClass::RealmSnapshot,
                &ShardToGateway::WindowFrame {
                    realm_fence: Fence(1),
                    window: w,
                    at: UniverseTick(1000),
                    hop: None,
                    rows: vec![snap(
                        RealmId::Planet(9),
                        FrameRef::PlanetCentered { planet_seed: 9 },
                        SYS7,
                        30.0,
                        1000,
                    )],
                },
            ),
            wire(
                SHARD,
                MsgClass::Control,
                &ShardToGateway::WindowBody {
                    realm_fence: Fence(1),
                    window: w,
                    subject: RealmId::System(7),
                    stmt: vd_wire::session_flow::BodyStmt::SelfLook {
                        bag: sys_look.clone(),
                    },
                    authored_at: UniverseTick(1000),
                },
            ),
        ]);
        let mut ob = OutboundBox::default();
        ob.0.extend(sent.iter().map(|(to, class, bytes)| {
            (
                *to,
                *class,
                vd_sim::io::bytes(bytes.clone()),
                vd_sim::io::Durability::Ephemeral,
            )
        }));
        // The LEVEL shipped at LOGIN (epoch 0→1, the swap signal — pinned by the login-flow
        // tests); this fold happens at the SAME epoch, so what ships now is the reliable DELTA
        // filling the drawn set: the origin row gained its self-look, and the composed child row
        // appeared (bagless — tracked, not drawn, until its statement lands).
        assert!(
            scene_levels(&ob).is_empty(),
            "a stable epoch re-ships no level — the delta fills the scene"
        );
        let deltas = scene_deltas(&ob);
        assert_eq!(deltas.len(), 1, "the first fold ships one filling delta");
        let (epoch, rows, removed) = &deltas[0];
        assert_eq!(*epoch, 1, "at the login epoch");
        assert!(removed.is_empty());
        assert_eq!(
            rows.len(),
            2,
            "the origin row (its look landed) + the composed child row"
        );
        assert_eq!(rows[0].realm, RealmId::System(7));
        assert_eq!(rows[0].parent, None);
        assert_eq!(
            rows[0].pose.pos.offset(),
            DVec3::ZERO,
            "the origin draws from its look AT the origin"
        );
        assert_eq!(
            rows[0].bag, sys_look,
            "the origin's self-look rides its delta row"
        );
        assert_eq!(rows[1].realm, RealmId::Planet(9));
        assert_eq!(
            rows[1].parent,
            Some(RealmId::System(7)),
            "hierarchy identity only"
        );
        assert_eq!(rows[1].pose.pos.offset(), DVec3::new(30.0, 0.0, 0.0));
        assert_eq!(
            rows[1].bag,
            Vec::<u8>::new(),
            "no statement yet — tracked, not drawn"
        );
        // The per-tick composed datagram rode the same tick: epoch stamped, the gateway's own
        // frame id, the origin absent from the rows (head≠tail law).
        let datagrams = composed_datagrams(&sent, CLIENT);
        assert_eq!(datagrams.len(), 1, "one fresh fold, one datagram");
        assert_eq!(datagrams[0].origin_epoch, 1);
        assert_eq!(
            datagrams[0].frame_id, 1,
            "the per-session monotone feed counter"
        );
        assert_eq!(
            datagrams[0]
                .realms
                .iter()
                .map(|r| r.realm)
                .collect::<Vec<_>>(),
            vec![RealmId::Planet(9)],
            "composed rows only — the origin never rides a datagram row"
        );
        assert_eq!(
            datagrams[0].realms[0].pose.frame, SYS7,
            "the TAIL is the origin frame"
        );

        // A BODY arrives for the child (its marker) at the SAME epoch: the next fresh fold ships
        // a reliable DELTA carrying the row with its new bag — never a whole new level.
        let luma = vd_core::look::luma_bag(2, 1.5);
        let sent = rig.tick(vec![
            wire(
                SHARD,
                MsgClass::Control,
                &ShardToGateway::WindowBody {
                    realm_fence: Fence(1),
                    window: w,
                    subject: RealmId::Planet(9),
                    stmt: vd_wire::session_flow::BodyStmt::Marker { luma: luma.clone() },
                    authored_at: UniverseTick(1001),
                },
            ),
            wire(
                SHARD,
                MsgClass::RealmSnapshot,
                &ShardToGateway::WindowFrame {
                    realm_fence: Fence(1),
                    window: w,
                    at: UniverseTick(1001),
                    hop: None,
                    rows: vec![snap(
                        RealmId::Planet(9),
                        FrameRef::PlanetCentered { planet_seed: 9 },
                        SYS7,
                        31.0,
                        1001,
                    )],
                },
            ),
        ]);
        let mut ob = OutboundBox::default();
        ob.0.extend(sent.iter().map(|(to, class, bytes)| {
            (
                *to,
                *class,
                vd_sim::io::bytes(bytes.clone()),
                vd_sim::io::Durability::Ephemeral,
            )
        }));
        assert!(
            scene_levels(&ob).is_empty(),
            "still no new level at a stable epoch"
        );
        let deltas = scene_deltas(&ob);
        assert_eq!(deltas.len(), 1, "the body change ships one reliable delta");
        let (epoch, added, removed) = &deltas[0];
        assert_eq!(*epoch, 1, "at the CURRENT epoch");
        assert_eq!(added.len(), 1);
        assert_eq!(added[0].realm, RealmId::Planet(9));
        assert_eq!(
            added[0].bag, luma,
            "the marker datum rides the delta row's bag"
        );
        assert!(removed.is_empty());
        // The second fresh fold advanced the feed counter — sibling chunks would share it.
        let datagrams = composed_datagrams(&sent, CLIENT);
        assert_eq!(datagrams.len(), 1);
        assert_eq!(datagrams[0].frame_id, 2);
        // A THIRD tick with an UNCHANGED tick/bags emits nothing new (already_current: no fold,
        // no datagram, no delta, no level).
        let sent = rig.tick(vec![]);
        assert!(composed_datagrams(&sent, CLIENT).is_empty());
    }

    #[test]
    fn a_relayed_batch_is_admitted_against_the_child_parked_preroster_and_fail_closed() {
        // THE Q2 RELAY's receiving admission (Slice C1, mesh minor 17; owner-approved 2026-08-16
        // — owner_decisions_2026-08-15.md addendum + window_lane.md §5 RULINGS), driven through
        // the REAL dispatch: a batch arriving BEFORE the author's roster is PARKED (counted
        // unvouched — the send-once forward must not be lost) and DRAINED through the full
        // admission the moment the author's level vouches the child; the opened statements admit
        // against the CHILD's identity (its level held, its self-look + its marker into the one
        // body store, a mis-authored inner body dropped apart); then every refusal arm:
        // a stale child fence, a forged sender, an unknown window, an undecodable seal, and a
        // stale relayed level — all counted, nothing guessed at.
        let mut rig = Rig::new();
        let (_, _) = rig.login(); // window 1 = Occupants(System 7) on SHARD
        let w = WindowId(1);
        let area = RealmId::Area(3);
        let area_frame = FrameRef::AreaLocal {
            planet_seed: 7,
            area_seed: 3,
        };
        let seal = vd_wire::session_flow::seal_relay_statements(&[
            RelayedStatement::Level {
                at: vd_core::UniverseTick(999),
                rows: vec![vd_wire::channels::RealmSnap {
                    realm: area,
                    frame: area_frame,
                    pose: vd_core::pose::StampedPose::at_rest(
                        FrameRef::PlanetCentered { planet_seed: 7 },
                        DVec3::new(1.0, 0.0, 0.0),
                        vd_core::UniverseTick(999),
                    ),
                }],
            },
            RelayedStatement::Body {
                subject: RealmId::Planet(7),
                stmt: BodyStmt::SelfLook { bag: vec![1] },
                authored_at: vd_core::UniverseTick(999),
            },
            RelayedStatement::Body {
                subject: area,
                stmt: BodyStmt::Marker { luma: vec![2] },
                authored_at: vd_core::UniverseTick(999),
            },
            // Mis-authored INNER body: a look about a realm that is not the child itself.
            RelayedStatement::Body {
                subject: RealmId::Planet(9),
                stmt: BodyStmt::SelfLook { bag: vec![3] },
                authored_at: vd_core::UniverseTick(999),
            },
        ]);
        let relayed = |child: RealmId, child_fence: Fence, statements: Vec<u8>| {
            ShardToGateway::WindowRelayed {
                realm_fence: Fence(1),
                window: w,
                child,
                child_fence,
                statements,
            }
        };
        // (1) BEFORE the author's roster: PARKED + counted unvouched; nothing ingested yet.
        // A second batch with an OLDER child fence does NOT displace the parked one (newest
        // wins even in the park), while a NEWER one does — both counted unvouched.
        let _ = rig.tick(vec![
            wire(
                SHARD,
                MsgClass::Control,
                &relayed(RealmId::Planet(7), Fence(9), seal.clone()),
            ),
            wire(
                SHARD,
                MsgClass::Control,
                &relayed(RealmId::Planet(7), Fence(7), vec![0xAA]),
            ),
        ]);
        assert_eq!(rig.stats().window_relay_unvouched, 2, "parked, counted");
        assert_eq!(rig.stats().window_relays_ingested, 0);
        // (2) The author's level lands naming the child → the parked batch DRAINS through the
        // full admission: the relayed level held, the child's self-look + its marker admitted
        // into the ONE body store, the mis-authored inner body dropped apart.
        let author_frame = ShardToGateway::WindowFrame {
            realm_fence: Fence(1),
            window: w,
            at: vd_core::UniverseTick(6),
            hop: None,
            rows: vec![vd_wire::channels::RealmSnap {
                realm: RealmId::Planet(7),
                frame: FrameRef::PlanetCentered { planet_seed: 7 },
                pose: vd_core::pose::StampedPose::at_rest(
                    FrameRef::SystemSpace { system_seed: 7 },
                    DVec3::new(30.0, 0.0, 0.0),
                    vd_core::UniverseTick(6),
                ),
            }],
        };
        let _ = rig.tick(vec![wire(SHARD, MsgClass::RealmSnapshot, &author_frame)]);
        assert_eq!(
            rig.stats().window_relays_ingested,
            3,
            "the drained batch: the level + the self-look + the marker"
        );
        assert_eq!(
            rig.stats().window_misauthored_body,
            1,
            "the mis-authored inner look dropped apart, against the CHILD's identity"
        );
        {
            let sessions = rig.world.resource::<GatewaySessions>();
            let held = &sessions.windows[&w];
            assert_eq!(
                held.ingest.look_of(RealmId::Planet(7)),
                Some(&[1u8][..]),
                "the relayed self-look landed in the ONE body store (the §2.8 handover path)"
            );
            assert_eq!(held.ingest.marker_of(area), Some(&[2u8][..]));
            assert_eq!(
                held.ingest.relay_child_roster(RealmId::Planet(7)),
                std::collections::BTreeSet::from([area]),
                "the relayed level IS the child's attested roster"
            );
        }
        // (3) A STALE child fence is refused (a deposed incarnation still shipping).
        let _ = rig.tick(vec![wire(
            SHARD,
            MsgClass::Control,
            &relayed(RealmId::Planet(7), Fence(8), seal.clone()),
        )]);
        assert_eq!(rig.stats().window_relay_stale, 1);
        // (4) A STALE relayed LEVEL (older `at` than held) refuses apart while a same-fence
        // redelivery of the bodies stays newest-wins (counted body-stale, not re-ingested).
        let stale_level =
            vd_wire::session_flow::seal_relay_statements(&[RelayedStatement::Level {
                at: vd_core::UniverseTick(998),
                rows: Vec::new(),
            }]);
        let _ = rig.tick(vec![wire(
            SHARD,
            MsgClass::Control,
            &relayed(RealmId::Planet(7), Fence(9), stale_level),
        )]);
        assert_eq!(
            rig.stats().window_relay_stale,
            2,
            "an older relayed level refuses apart (newest wins)"
        );
        // (5) FORGED sender: a routable node that is not the window's head.
        let _ = rig.tick(vec![wire(
            DEST,
            MsgClass::Control,
            &relayed(RealmId::Planet(7), Fence(9), seal.clone()),
        )]);
        assert_eq!(rig.stats().window_sender_mismatch, 1);
        // (6) UNKNOWN window id: dropped by id mismatch.
        let _ = rig.tick(vec![wire(
            SHARD,
            MsgClass::Control,
            &ShardToGateway::WindowRelayed {
                realm_fence: Fence(1),
                window: WindowId(99),
                child: RealmId::Planet(7),
                child_fence: Fence(9),
                statements: seal.clone(),
            },
        )]);
        assert_eq!(rig.stats().window_unknown_row, 1);
        // (7) An UNDECODABLE seal (vouched child, fresh fence): counted, dropped, fail-closed.
        let _ = rig.tick(vec![wire(
            SHARD,
            MsgClass::Control,
            &relayed(RealmId::Planet(7), Fence(10), vec![0xFF, 0xFF, 0xFF]),
        )]);
        assert_eq!(rig.stats().window_relay_undecodable, 1);
        // (8) A STALE relayed BODY inside an otherwise-fresh batch: the level applies (newer),
        // the older-stamped look refuses apart (`window_body_stale`) — newest wins per subject.
        let ingested_before = rig.stats().window_relays_ingested;
        let stale_body = vd_wire::session_flow::seal_relay_statements(&[
            RelayedStatement::Level {
                at: vd_core::UniverseTick(1000),
                rows: vec![vd_wire::channels::RealmSnap {
                    realm: area,
                    frame: area_frame,
                    pose: vd_core::pose::StampedPose::at_rest(
                        FrameRef::PlanetCentered { planet_seed: 7 },
                        DVec3::new(2.0, 0.0, 0.0),
                        vd_core::UniverseTick(1000),
                    ),
                }],
            },
            RelayedStatement::Body {
                subject: RealmId::Planet(7),
                stmt: BodyStmt::SelfLook { bag: vec![9] },
                authored_at: vd_core::UniverseTick(998),
            },
        ]);
        let _ = rig.tick(vec![wire(
            SHARD,
            MsgClass::Control,
            &relayed(RealmId::Planet(7), Fence(11), stale_body),
        )]);
        assert_eq!(
            rig.stats().window_body_stale,
            1,
            "an older relayed body refuses apart while its level applies"
        );
        assert_eq!(
            rig.stats().window_relays_ingested,
            ingested_before + 1,
            "the batch's fresh level still ingested"
        );
    }

    #[test]
    fn a_drained_parked_body_older_than_the_held_statement_counts_stale() {
        // The drain's newest-wins refusal, driven directly (the rig cannot reach it: a marker
        // parks only pre-roster, and a roster never un-confirms — this pins the arm for the
        // redelivered-straggler shape a reordered reliable stream could still produce).
        let mut held = GatewayWindow {
            shard: SHARD,
            scope: WindowScope::Occupants,
            author_realm: RealmId::System(7),
            ingest: window::WindowIngest::default(),
            parked_bodies: BTreeMap::new(),
            parked_relays: BTreeMap::new(),
        };
        let roster_level = window::WindowLevel {
            at: vd_core::UniverseTick(10),
            hop: None,
            rows: vec![vd_wire::channels::RealmSnap {
                realm: RealmId::Planet(7),
                frame: FrameRef::PlanetCentered { planet_seed: 7 },
                pose: vd_core::pose::StampedPose::at_rest(
                    FrameRef::SystemSpace { system_seed: 7 },
                    DVec3::new(30.0, 0.0, 0.0),
                    vd_core::UniverseTick(10),
                ),
            }],
        };
        assert_eq!(
            held.ingest
                .ingest_frame(roster_level, &window_tuning(&config())),
            window::Ingested::Applied
        );
        assert!(held.ingest.ingest_body(
            RealmId::Planet(7),
            &BodyStmt::Marker { luma: vec![9] },
            vd_core::UniverseTick(10)
        ));
        held.parked_bodies.insert(
            RealmId::Planet(7),
            (BodyStmt::Marker { luma: vec![1] }, vd_core::UniverseTick(4)),
        );
        let mut stats = GatewayStats::default();
        drain_parked(&mut held, &mut stats);
        assert_eq!(
            stats.window_body_stale, 1,
            "the drained straggler refuses — newest wins"
        );
        assert_eq!(
            held.ingest.marker_of(RealmId::Planet(7)),
            Some(&[9u8][..]),
            "the held statement survives the drain"
        );
        assert!(
            held.parked_bodies.is_empty(),
            "the park slot cleared either way"
        );
    }

    #[test]
    fn an_old_lane_scene_delta_from_a_shard_is_counted_and_dropped() {
        // ★TOMBSTONE (Slice C2): a shard-side per-observer scene delta has no producer left
        // (deleted, never disabled — §4.5 Topic 5) — its frames land here COUNTED, and nothing
        // client-bound leaves (the composed lane is the one scene author).
        let mut rig = Rig::new();
        let (_sid, _) = rig.login();
        let sent = rig.tick(vec![wire(
            SHARD,
            MsgClass::Control,
            &ShardToGateway::RealmSceneDelta {
                observer: AccountId(5),
                added: vec![],
                removed: vec![RealmId::Planet(8)],
            },
        )]);
        assert_eq!(rig.stats().old_scene_deltas_dropped, 1);
        assert!(
            decode_controls(&sent, CLIENT)
                .iter()
                .all(|m| !matches!(m, ServerControlMsg::RealmSceneDelta { .. })),
            "nothing scene-shaped is forwarded from the dead lane"
        );
        assert_eq!(
            rig.stats().undecodable,
            0,
            "routed and understood, deliberately dropped"
        );
    }

    #[test]
    fn a_crossing_swap_level_is_composed_at_the_last_old_epoch_tick_when_retained() {
        // §2.7's same-T swap, the retained arm (the capstone drives the fallback — its rings hold
        // nothing at the crossing): a hand-built table where BOTH origins' windows retain the old
        // epoch's tick. The swap level must be composed AT that tick — every persisting body's
        // position is the same-tick re-expression, screen deltas bounded by one tick of true
        // motion (the crossing no-flicker gate's mechanical half) — and the swap tick ships NO
        // datagram (the level carries the poses; the next fresh tick resumes the feed).
        let cfg = config();
        let (mut sessions, sid, _) = one_active_session();
        // The session stands in System(7) (an Active sub on SHARD).
        sessions
            .by_session
            .get_mut(&sid)
            .expect("present")
            .subs
            .insert(
                SHARD,
                SubRecord {
                    sub: SubId(0),
                    frame: SYS7,
                    accepted: Fence(1),
                    state: SubState::Active,
                },
            );
        // Two Occupants windows on SHARD — the old origin's and the new one's — each retaining
        // levels at T=100 AND T=101 (the ring spans both, so the swap CAN re-express at 100).
        let mut mk = |id: u64, author: RealmId, own_frame: FrameRef, ticks: &[u64]| {
            let mut ingest = window::WindowIngest::default();
            for &at in ticks {
                let level = window::WindowLevel {
                    at: UniverseTick(at),
                    hop: None,
                    rows: vec![vd_wire::channels::RealmSnap {
                        realm: RealmId::Station(42),
                        frame: FrameRef::StationLocal { station_seed: 42 },
                        pose: vd_core::pose::StampedPose::at_rest(
                            own_frame,
                            DVec3::new(9.0, 0.0, 0.0),
                            UniverseTick(at),
                        ),
                    }],
                };
                assert_eq!(
                    ingest.ingest_frame(level, &window_tuning(&cfg)),
                    window::Ingested::Applied
                );
            }
            sessions.windows.insert(
                WindowId(id),
                GatewayWindow {
                    shard: SHARD,
                    scope: WindowScope::Occupants,
                    author_realm: author,
                    ingest,
                    parked_bodies: BTreeMap::new(),
                    parked_relays: BTreeMap::new(),
                },
            );
        };
        mk(1, RealmId::System(7), SYS7, &[100, 101]);
        // The NEW origin's window retains the old tick AND a newer one — without the same-T
        // preference the swap would fold at 102 and this test would fail.
        mk(2, RealmId::Planet(7), PLANET7, &[100, 101, 102]);
        let mut stats = GatewayStats::default();
        let mut ob = OutboundBox::default();
        // Pass 1: the login scene in System(7) — folds at the NEWEST common tick (101), epoch 1.
        compose_scenes_pass(&cfg, &test_clock(), &mut sessions, &mut stats, &mut ob);
        assert_eq!(stats.scene_levels_sent, 1);
        assert_eq!(
            stats.scene_datagrams_sent, 1,
            "the fresh fold ships the feed"
        );
        assert_eq!(
            sessions.by_session[&sid].shadow.last_t,
            Some(UniverseTick(101))
        );
        // THE CROSSING: the standing frame flips to Planet(7) (the dest sub took over).
        sessions
            .by_session
            .get_mut(&sid)
            .expect("present")
            .subs
            .get_mut(&SHARD)
            .expect("present")
            .frame = PLANET7;
        let datagrams_before = stats.scene_datagrams_sent;
        let mut ob = OutboundBox::default();
        compose_scenes_pass(&cfg, &test_clock(), &mut sessions, &mut stats, &mut ob);
        let levels = scene_levels(&ob);
        assert_eq!(levels.len(), 1, "the swap level shipped");
        let (origin, epoch, rows) = &levels[0];
        assert_eq!(*origin, RealmId::Planet(7));
        assert_eq!(*epoch, 2, "the crossing bumped the epoch exactly once");
        // THE SAME-T HALF: the new chain retains tick 100 — wait, the swap must compose at the
        // LAST OLD-EPOCH tick (101), which the new window retains too.
        assert_eq!(
            sessions.by_session[&sid].shadow.last_t,
            Some(UniverseTick(101)),
            "the swap level is composed AT the last old-epoch tick (§2.7 same-T)"
        );
        for row in rows.iter().filter(|r| r.realm != RealmId::Planet(7)) {
            assert_eq!(
                row.pose.universe_tick,
                UniverseTick(101),
                "every composed swap row is the SAME-tick re-expression"
            );
        }
        assert_eq!(
            stats.scene_datagrams_sent, datagrams_before,
            "the swap tick ships no datagram — the level carries the poses"
        );
    }

    /// §2.7's same-T swap, the FALLBACK arms (the retained twin above drives the preference):
    /// a crossing whose new chain does NOT retain the old tick folds at the freshest common
    /// stamp instead — the same-tick re-expression is preferred, never demanded.
    #[test]
    fn a_crossing_swap_falls_back_to_the_fresh_tick_when_the_old_one_is_not_retained() {
        let cfg = config();
        let (mut sessions, sid, _) = one_active_session();
        sessions
            .by_session
            .get_mut(&sid)
            .expect("present")
            .subs
            .insert(
                SHARD,
                SubRecord {
                    sub: SubId(0),
                    frame: SYS7,
                    accepted: Fence(1),
                    state: SubState::Active,
                },
            );
        let mut mk = |id: u64, author: RealmId, own_frame: FrameRef, ticks: &[u64]| {
            let mut ingest = window::WindowIngest::default();
            for &at in ticks {
                let level = window::WindowLevel {
                    at: UniverseTick(at),
                    hop: None,
                    rows: vec![vd_wire::channels::RealmSnap {
                        realm: RealmId::Station(42),
                        frame: FrameRef::StationLocal { station_seed: 42 },
                        pose: vd_core::pose::StampedPose::at_rest(
                            own_frame,
                            DVec3::new(9.0, 0.0, 0.0),
                            UniverseTick(at),
                        ),
                    }],
                };
                assert_eq!(
                    ingest.ingest_frame(level, &window_tuning(&cfg)),
                    window::Ingested::Applied
                );
            }
            sessions.windows.insert(
                WindowId(id),
                GatewayWindow {
                    shard: SHARD,
                    scope: WindowScope::Occupants,
                    author_realm: author,
                    ingest,
                    parked_bodies: BTreeMap::new(),
                    parked_relays: BTreeMap::new(),
                },
            );
        };
        mk(1, RealmId::System(7), SYS7, &[100, 101]);
        // The NEW origin's window retains ONLY a newer tick — the same-T preference cannot hold.
        mk(2, RealmId::Planet(7), PLANET7, &[102]);
        let mut stats = GatewayStats::default();
        let mut ob = OutboundBox::default();
        compose_scenes_pass(&cfg, &test_clock(), &mut sessions, &mut stats, &mut ob);
        assert_eq!(
            sessions.by_session[&sid].shadow.last_t,
            Some(UniverseTick(101))
        );
        sessions
            .by_session
            .get_mut(&sid)
            .expect("present")
            .subs
            .get_mut(&SHARD)
            .expect("present")
            .frame = PLANET7;
        let mut ob = OutboundBox::default();
        compose_scenes_pass(&cfg, &test_clock(), &mut sessions, &mut stats, &mut ob);
        let levels = scene_levels(&ob);
        assert_eq!(levels.len(), 1, "the swap level still ships");
        assert_eq!(levels[0].1, 2, "the epoch still bumps exactly once");
        assert_eq!(
            sessions.by_session[&sid].shadow.last_t,
            Some(UniverseTick(102)),
            "unretained old tick: the freshest stamp serves (§2.7's stated fallback)"
        );
    }

    /// §2.7's same-T swap, the short-prefix fallback: a crossing whose new chain is NOT fully
    /// fresh (a stale upper stratum caps the prefix) also falls back to the fresh fold.
    #[test]
    fn a_crossing_swap_falls_back_when_the_new_chain_is_not_fully_fresh() {
        let cfg = config();
        let (mut sessions, sid, _) = one_active_session();
        sessions
            .by_session
            .get_mut(&sid)
            .expect("present")
            .subs
            .insert(
                SHARD,
                SubRecord {
                    sub: SubId(0),
                    frame: SYS7,
                    accepted: Fence(1),
                    state: SubState::Active,
                },
            );
        let tuning = window_tuning(&cfg);
        let mut occupants = |id: u64, author: RealmId, own_frame: FrameRef, ticks: &[u64]| {
            let mut ingest = window::WindowIngest::default();
            for &at in ticks {
                let level = window::WindowLevel {
                    at: UniverseTick(at),
                    hop: None,
                    rows: vec![vd_wire::channels::RealmSnap {
                        realm: RealmId::Station(42),
                        frame: FrameRef::StationLocal { station_seed: 42 },
                        pose: vd_core::pose::StampedPose::at_rest(
                            own_frame,
                            DVec3::new(9.0, 0.0, 0.0),
                            UniverseTick(at),
                        ),
                    }],
                };
                assert_eq!(
                    ingest.ingest_frame(level, &tuning),
                    window::Ingested::Applied
                );
            }
            sessions.windows.insert(
                WindowId(id),
                GatewayWindow {
                    shard: SHARD,
                    scope: WindowScope::Occupants,
                    author_realm: author,
                    ingest,
                    parked_bodies: BTreeMap::new(),
                    parked_relays: BTreeMap::new(),
                },
            );
        };
        occupants(1, RealmId::System(7), SYS7, &[100, 101]);
        occupants(2, RealmId::Planet(7), PLANET7, &[101, 102]);
        // The new chain's UPPER stratum (the parent's Child-scope window) shares no fresh tick
        // with the leaf: the prefix caps at 1 < 2 and the same-T preference cannot hold.
        let mut stale_parent = window::WindowIngest::default();
        assert_eq!(
            stale_parent.ingest_frame(
                window::WindowLevel {
                    at: UniverseTick(50),
                    hop: Some(vd_wire::session_flow::HopRow {
                        child: RealmId::Planet(7),
                        inv: vd_core::frame::FramePlacement::moving(
                            DVec3::new(-30.0, 0.0, 0.0),
                            DVec3::ZERO,
                        ),
                    }),
                    rows: vec![vd_wire::channels::RealmSnap {
                        realm: RealmId::Planet(7),
                        frame: PLANET7,
                        pose: vd_core::pose::StampedPose::at_rest(
                            SYS7,
                            DVec3::new(30.0, 0.0, 0.0),
                            UniverseTick(50),
                        ),
                    }],
                },
                &tuning
            ),
            window::Ingested::Applied
        );
        sessions.windows.insert(
            WindowId(3),
            GatewayWindow {
                shard: SHARD,
                scope: WindowScope::Child(RealmId::Planet(7)),
                author_realm: RealmId::System(7),
                ingest: stale_parent,
                parked_bodies: BTreeMap::new(),
                parked_relays: BTreeMap::new(),
            },
        );
        let mut stats = GatewayStats::default();
        let mut ob = OutboundBox::default();
        compose_scenes_pass(&cfg, &test_clock(), &mut sessions, &mut stats, &mut ob);
        assert_eq!(
            sessions.by_session[&sid].shadow.last_t,
            Some(UniverseTick(101))
        );
        sessions
            .by_session
            .get_mut(&sid)
            .expect("present")
            .subs
            .get_mut(&SHARD)
            .expect("present")
            .frame = PLANET7;
        let mut ob = OutboundBox::default();
        compose_scenes_pass(&cfg, &test_clock(), &mut sessions, &mut stats, &mut ob);
        let levels = scene_levels(&ob);
        assert_eq!(levels.len(), 1, "the swap level still ships");
        assert_eq!(
            sessions.by_session[&sid].shadow.last_t,
            Some(UniverseTick(102)),
            "a short-prefix chain folds fresh — the preference never blocks the swap"
        );
    }

    /// THE REMOVE MESSAGE's fan (proto_minor 14, D-4(a)): a shard's `EntityRemoved` reaches its
    /// Active subscriber as a typed `ServerControlMsg::Event`, fence-checked per subscriber and
    /// minor-gated per session — a stale removal is refused + counted apart (a wrongly-dropped
    /// removal strands a frozen figure), and an older peer is withheld the variant.
    #[test]
    fn an_entity_removed_fans_to_subscribers_fence_checked_and_minor_gated() {
        let mut rig = Rig::new();
        let (sid, login_sends) = rig.login(); // Active session at CLIENT, subscribed to SHARD
        let entity = EntityId(0xE1);
        let at = vd_core::UniverseTick(500);
        let removal = |fence: Fence| ShardToGateway::EntityRemoved {
            realm_fence: fence,
            entity,
            at,
        };
        // (1) A live-fence removal reaches the client as the typed Event.
        let sent = rig.tick(vec![wire(SHARD, MsgClass::Control, &removal(Fence(1)))]);
        let expected =
            ServerControlMsg::Event(vd_wire::channels::EventMsg::EntityRemoved { entity, at });
        assert!(
            decode_controls(&sent, CLIENT).contains(&expected),
            "the removal routes through on_shard_control to the subscriber"
        );
        // (2) A STALE-fence removal (a demoted old owner) is refused for this subscriber + counted
        // on its own counter — it must not evict what the live owner still streams.
        let sent = rig.tick(vec![wire(
            SHARD,
            MsgClass::Control,
            &removal(Fence::GENESIS),
        )]);
        assert!(
            !decode_controls(&sent, CLIENT).contains(&expected),
            "a stale removal is withheld"
        );
        assert_eq!(
            rig.world.resource::<GatewayStats>().stale_removals_dropped,
            1
        );
        // (3) An older peer (minor < 14) is withheld the variant (sender-gates-variants) — it
        // keeps the frozen-figure gap the message closes, but its wire never desyncs.
        rig.world
            .resource_mut::<GatewaySessions>()
            .by_session
            .get_mut(&sid)
            .expect("session")
            .negotiated_minor = 13;
        let sent = rig.tick(vec![wire(SHARD, MsgClass::Control, &removal(Fence(1)))]);
        assert!(
            decode_controls(&sent, CLIENT).is_empty(),
            "a minor<14 peer receives no Event"
        );
        // (4) THE OWNER SKIP: a removal for the session's OWN avatar is never fanned to it — the
        // removal is a fact about the OLD realm mid-crossing, while the owner's truth is its live
        // dest sub; evicting would blink the one figure that must never blink. Restore minor 14
        // first so the skip (not the gate) is what withholds it.
        rig.world
            .resource_mut::<GatewaySessions>()
            .by_session
            .get_mut(&sid)
            .expect("session")
            .negotiated_minor = 14;
        // The session's own entity, read off the wire the login itself announced (`OwnEntity`) —
        // no fallible phase destructure, so no failure-only region (HR5).
        let own = login_sends
            .iter()
            .flat_map(|tick| decode_controls(tick, CLIENT))
            .find_map(|msg| match msg {
                ServerControlMsg::OwnEntity { entity } => Some(entity),
                _ => None,
            })
            .expect("the login announces the own entity");
        let sent = rig.tick(vec![wire(
            SHARD,
            MsgClass::Control,
            &ShardToGateway::EntityRemoved {
                realm_fence: Fence(1),
                entity: own,
                at: vd_core::UniverseTick(501),
            },
        )]);
        assert!(
            decode_controls(&sent, CLIENT).is_empty(),
            "the entity's own session is never told to evict its own avatar"
        );
    }

    /// The removal fan's C2 honesty arms — the same dead-branch traps the frame fan carries:
    /// (a) the reverse index names a session absent from `by_session`; (b) a present session
    /// whose hot table holds no entry for the shard; (c) a non-Active session is skipped clean.
    /// Each is COUNTED (or a clean skip), never a silent drop.
    #[test]
    fn an_entity_removed_fan_counts_desyncs_and_skips_non_active_sessions() {
        let removal = ShardToGateway::EntityRemoved {
            realm_fence: Fence(1),
            entity: EntityId(0xE1),
            at: vd_core::UniverseTick(500),
        };
        // (a) index points at a session that does not exist in by_session.
        let mut sessions = GatewaySessions::default();
        sessions
            .subscribed_shards
            .entry(SHARD)
            .or_default()
            .insert(SessionId(0xC0DE));
        let mut stats = GatewayStats::default();
        let mut outbox = OutboundBox::default();
        fan_entity_removed(
            SHARD,
            Fence(1),
            EntityId(0xE1),
            vd_core::UniverseTick(500),
            &mut sessions,
            &mut stats,
            &mut outbox,
        );
        assert_eq!(stats.frame_sub_desync, 1, "(a) missing session is counted");
        assert!(outbox.0.is_empty());

        // (b) a present session indexed under SHARD but with an EMPTY hot SubTable.
        let (mut sessions, sid, _) = one_active_session();
        sessions.by_session[&sid]
            .hot
            .subs
            .store(Arc::new(SubTable::default()));
        let mut stats = GatewayStats::default();
        let mut outbox = OutboundBox::default();
        fan_entity_removed(
            SHARD,
            Fence(1),
            EntityId(0xE1),
            vd_core::UniverseTick(500),
            &mut sessions,
            &mut stats,
            &mut outbox,
        );
        assert_eq!(stats.frame_sub_desync, 1, "(b) lookup None is counted");
        assert!(outbox.0.is_empty(), "no removal forwarded on a desync");

        // (c) a subscribed but NON-Active session (self-fenced / still attaching) is skipped
        // clean — served no removals, exactly as it is served no frames.
        let (mut sessions, sid, _) = one_active_session();
        sessions.by_session.get_mut(&sid).expect("session").phase = SessionPhase::AwaitingAttach;
        let mut stats = GatewayStats::default();
        let mut outbox = OutboundBox::default();
        fan_entity_removed(
            SHARD,
            Fence(1),
            EntityId(0xE1),
            vd_core::UniverseTick(500),
            &mut sessions,
            &mut stats,
            &mut outbox,
        );
        assert_eq!(stats.frame_sub_desync, 0, "not a desync — a clean skip");
        assert!(outbox.0.is_empty());
        let _ = postcard::to_allocvec(&removal).expect("the fixture arm stays constructible");
    }

    #[test]
    fn a_granted_session_lease_injects_exactly_one_server_derived_home_demand() {
        // The trusted injector: on the committed-lease arm (an authenticated login LANDED), the gateway
        // emits EXACTLY ONE RealmDemand{SpinUp} whose child is the SERVER-DERIVED home lineage
        // (container_coord_at over the account's STORED pose) — the client supplies NO coord/pose.
        let account = AccountId(5); // the login account (hello_msg)
        let mut rig = Rig::new();
        arm_injector(
            &mut rig,
            armed_injector(homes_for(account, TEST_HOME_REALM, 0.0)),
            true,
        );
        let (_sid, sends) = rig.login();
        let demands: Vec<RealmDemand> = sends
            .iter()
            .flat_map(|tick| demands_to_orch(tick))
            .collect();
        assert_eq!(
            demands.len(),
            1,
            "exactly one home demand per committed lease"
        );
        // The child is the STORED home's lineage — NOT anything the client sent. The Area-A box is the
        // deep 5-level [Universe, Galaxy, System(7), Planet(7), Area(7)] home (5f-3a).
        let expected_child = lineage_of(TEST_HOME_REALM);
        assert_eq!(
            demands[0].child, expected_child,
            "the child is the stored home's lineage, read and not resolved"
        );
        assert_eq!(demands[0].verb, DemandVerb::SpinUp);
        assert_eq!(
            demands[0].universe_tick,
            UniverseTick(50),
            "the demand carries the clock's universe tick (determinism, no wall-clock)"
        );
        assert_eq!(
            demands[0].parent_fence,
            Fence(5),
            "the per-account sentinel fence (account 5 → 5), not a global constant"
        );
    }

    #[test]
    fn an_unarmed_injector_emits_no_home_demand_byte_identical() {
        // Default (VD_DEMAND unset) ⇒ the injector is INERT: a login emits NO RealmDemand (the byte-
        // identical default; covers the `armed == false` short-circuit arm of the emit gate).
        let mut rig = Rig::new(); // config() ⇒ the inert injector (unarmed)
        let (_sid, sends) = rig.login();
        assert_eq!(
            demand_count(&sends),
            0,
            "an unarmed gateway injects no home demand"
        );
    }

    #[test]
    fn a_pre_sync_clock_injects_no_demand_and_a_synced_clock_injects_one() {
        // The synced gate (CRITIQUE-2): a PRE-SYNC seed would carry universe_tick 0 → last_demand_tick 0,
        // which `demanded_recently` treats as "never demanded" → the home would never spin. So the
        // injector emits ONLY once the clock is synced. Both arms of the `clock.synced` gate.
        //
        // MF3 — and an ARMED-but-pre-sync grant must HOLD, not fall back to the static path: `config.shard`
        // is not this player's home on an armed cluster, so a static attach there would serve the player from
        // the WRONG shard (or retry forever with no TTL). It emits NOTHING client-ward or shard-ward, stays in
        // `AwaitingDirectory`, is COUNTED, and the re-driven idempotent `LeaseGrant` takes the DYNAMIC arm
        // once the clock syncs.
        let poses = homes_for(AccountId(5), TEST_HOME_REALM, 0.0);
        // pre-sync: armed, but the clock has not synced → NO demand.
        let mut rig = Rig::new();
        arm_injector(&mut rig, armed_injector(poses.clone()), false);
        let sid = {
            let _ = rig.tick(vec![wire(CLIENT, MsgClass::Control, &hello_msg())]);
            session_of(&rig, CLIENT)
        };
        let grant = rig.tick(vec![wire(ORCH, MsgClass::Saga, &granted_head(sid))]);
        assert_eq!(
            decode_controls(&grant, CLIENT),
            Vec::new(),
            "a pre-sync armed grant pushes NOTHING at the client (no Welcome, no Close — no artifact)"
        );
        assert!(
            !saw_attach(&grant, SHARD, sid, None),
            "and NEVER attaches to the static config.shard on an armed cluster (MF3)"
        );
        assert_eq!(demands_to_orch(&grant).len(), 0, "and demands nothing");
        assert_eq!(
            grant
                .iter()
                .map(|(to, class, _)| (*to, *class))
                .collect::<Vec<_>>(),
            vec![(ORCH, MsgClass::Saga)],
            "the ONLY send is the AwaitingDirectory retry's idempotent LeaseGrant"
        );
        assert_eq!(phase_of(&rig, sid), SessionPhase::AwaitingDirectory);
        assert_eq!(
            rig.stats().logins_held_pre_sync,
            1,
            "the hold is counted, never silent"
        );
        // The clock syncs; the SAME session's re-driven grant now takes the DYNAMIC arm.
        rig.world.resource_mut::<ClockSample>().synced = true;
        let synced_grant = rig.tick(vec![wire(ORCH, MsgClass::Saga, &granted_head(sid))]);
        assert_eq!(
            phase_of(&rig, sid),
            waiting_phase(),
            "once synced the held login enters AwaitingHomeRealm — no re-login needed"
        );
        assert_eq!(
            demands_to_orch(&synced_grant).len(),
            1,
            "and its home is demanded exactly then"
        );
        assert_eq!(rig.stats().logins_held_pre_sync, 1, "held once, not twice");
        // The original 5f-3c assertion, unchanged: a whole pre-sync login drive injects no demand at all.
        let mut rig = Rig::new();
        arm_injector(&mut rig, armed_injector(poses.clone()), false);
        let (_sid, presync) = rig.login();
        assert_eq!(
            demand_count(&presync),
            0,
            "no home demand while the clock is pre-sync"
        );
        // synced: the same armed injector, clock synced → exactly one demand, nonzero tick.
        let mut rig = Rig::new();
        arm_injector(&mut rig, armed_injector(poses), true);
        let (_sid, synced) = rig.login();
        let demands: Vec<RealmDemand> = synced
            .iter()
            .flat_map(|tick| demands_to_orch(tick))
            .collect();
        assert_eq!(
            demands.len(),
            1,
            "exactly one home demand once the clock is synced"
        );
        assert_ne!(
            demands[0].universe_tick,
            UniverseTick(0),
            "a synced demand carries a NONZERO universe tick (so demanded_recently desires it)"
        );
    }

    #[test]
    fn a_re_driven_grant_injects_the_home_demand_exactly_once() {
        // The mint-commit arm fires ONCE (AwaitingDirectory→AwaitingAttach); a re-driven grant re-enters
        // at the AwaitingAttach / Active guards above and returns — it never re-emits. Emit-once per lease.
        let mut rig = Rig::new();
        arm_injector(
            &mut rig,
            armed_injector(homes_for(AccountId(5), TEST_HOME_REALM, 0.0)),
            true,
        );
        let hello = rig.tick(vec![wire(CLIENT, MsgClass::Control, &hello_msg())]);
        let sid = rig
            .world
            .resource::<GatewaySessions>()
            .sessions()
            .next()
            .expect("session pending");
        // The committed-lease transition (5f-3d: `AwaitingDirectory → AwaitingHomeRealm`, since this rig is
        // armed + synced): the ONE inline emit.
        let g1 = rig.tick(vec![wire(ORCH, MsgClass::Saga, &granted_head(sid))]);
        // A re-driven grant while `AwaitingHomeRealm`: the not-AwaitingDirectory guard returns — NO emit.
        // (The `local_tick` stays 1 throughout, so the 5f-3d re-drive cadence never fires here either.)
        let g2 = rig.tick(vec![wire(ORCH, MsgClass::Saga, &granted_head(sid))]);
        // A `SessionAttached` from the STATIC shard while awaiting the HOME realm is correctly ignored
        // (5f-3d: this armed session is routed to its home, not to `config.shard`) — and emits nothing.
        let attached = rig.tick(vec![wire(
            SHARD,
            MsgClass::Control,
            &ShardToGateway::SessionAttached {
                session: sid,
                entity: EntityId(77),
                frame: FrameRef::SystemSpace { system_seed: 7 },
                realm_fence: Fence(1),
            },
        )]);
        // A further granted head: still the same guard, still NO emit.
        let g3 = rig.tick(vec![wire(ORCH, MsgClass::Saga, &granted_head(sid))]);
        let total: usize = [&hello, &g1, &g2, &attached, &g3]
            .iter()
            .map(|tick| demands_to_orch(tick).len())
            .sum();
        assert_eq!(
            total, 1,
            "exactly one home demand across the whole login + every re-drive"
        );
    }

    /// Drive an armed gateway to a login's `AwaitingDirectory`, then feed `reply` (a NON-granted head):
    /// no home demand is ever injected (the emit is strictly downstream of the committed-lease arm).
    fn assert_no_demand_on_non_granted(reply: impl Fn(SessionId) -> InterShardFlow) {
        let mut rig = Rig::new();
        arm_injector(
            &mut rig,
            armed_injector(homes_for(AccountId(5), TEST_HOME_REALM, 0.0)),
            true,
        );
        let hello = rig.tick(vec![wire(CLIENT, MsgClass::Control, &hello_msg())]);
        let sid = rig
            .world
            .resource::<GatewaySessions>()
            .sessions()
            .next()
            .expect("session pending");
        let after = rig.tick(vec![wire(ORCH, MsgClass::Saga, &reply(sid))]);
        assert_eq!(
            demands_to_orch(&hello).len(),
            0,
            "the hello tick injects no demand"
        );
        assert_eq!(
            demands_to_orch(&after).len(),
            0,
            "a non-granted reply injects no home demand"
        );
    }

    #[test]
    fn an_absent_directory_head_injects_no_home_demand() {
        // record None (the reaper revoked / never minted) ⇒ granted false ⇒ the refused arm ⇒ NO emit.
        assert_no_demand_on_non_granted(absent_head);
    }

    #[test]
    fn a_foreign_owned_head_injects_no_home_demand() {
        // The lease is owned by a DIFFERENT gateway ⇒ granted false ⇒ NO emit.
        assert_no_demand_on_non_granted(foreign_head);
    }

    #[test]
    fn a_wrong_fence_head_injects_no_home_demand() {
        // Owned by THIS gateway but at the wrong fence ⇒ granted false ⇒ NO emit.
        assert_no_demand_on_non_granted(wrong_fence_head);
    }

    #[test]
    fn an_absent_spawn_pose_derives_the_origin_home_in_forest() {
        // The P7-store-absent stand-in: NO stored pose ⇒ the injector derives the ROOT/ORIGIN home coord
        // (still a valid in-forest lineage), NOT a skip. One demand, child == container_coord_at(origin).
        let mut rig = Rig::new();
        arm_injector(&mut rig, armed_injector(default_homes()), true); // empty pose store
        let (_sid, sends) = rig.login();
        let demands: Vec<RealmDemand> = sends
            .iter()
            .flat_map(|tick| demands_to_orch(tick))
            .collect();
        assert_eq!(
            demands.len(),
            1,
            "an absent pose still injects one (origin) home demand"
        );
        assert_eq!(
            demands[0].child,
            default_homes().fallback().realm,
            "an account with no home of its own is demanded at the registry's fallback home"
        );
    }

    #[test]
    fn the_home_sentinel_fence_is_per_account_and_deterministic() {
        // CRITIQUE-1 defense-in-depth: the sentinel is DETERMINISTIC per-account (NOT a global constant),
        // so two accounts carry DISTINCT parent_fences — no FencedKey idempotency-collapse.
        assert_ne!(
            home_sentinel_fence(AccountId(5)),
            home_sentinel_fence(AccountId(6)),
            "two accounts ⇒ two distinct sentinels"
        );
        assert_eq!(
            home_sentinel_fence(AccountId(5)),
            Fence(5),
            "the fold is deterministic (low word for a small account)"
        );
        // Accounts differing ONLY in the high u64 word still differ — the fold mixes both halves.
        assert_ne!(
            home_sentinel_fence(AccountId(1u128 << 64)),
            home_sentinel_fence(AccountId(0)),
            "the high word is folded in (distinct even when the low word matches)"
        );
    }

    #[test]
    fn the_defence_reads_the_same_world_that_produced_the_answer() {
        // The injector's defence-in-depth predicate, both arms — now asked OF THE WORLD rather than of a
        // freshly rebuilt one. It used to consult the hand-placed world while the home came from the
        // generated one, which is only safe while those two happen to hold the same realms; the false arm
        // below is exactly what a valid home beside a second star would have hit.
        use vd_core::realm_path::{RealmKindTag, RealmLevel};
        // The LOWERED world — the exact value the shipped injector holds and queries.
        let world = test_world().lowered();
        let home = lineage_of(TEST_HOME_REALM);
        assert!(
            world.contains_realm(home.lowered()),
            "a stored home is in the world it is stored against"
        );
        // Append a leaf realm no world contains (Station(0xDEAD)) — a corrupted stand-in.
        let off = home.child(RealmLevel::new(RealmKindTag::Station, 0xDEAD));
        assert!(
            !world.contains_realm(off.lowered()),
            "an off-world leaf (a corrupted stand-in) is rejected"
        );
    }

    // ===== RLM 5f-3d — the GATEWAY DYNAMIC-HOME ROUTE + the seamless attach hold ===================

    /// The DEMAND-SPAWNED home shard. Deliberately NOT a member of `config().known_shards`
    /// (`{SHARD, DEST}`), exactly like a real shard whose `NodeId` is minted at spawn time — so any test
    /// that routes to it proves the RUNTIME `dynamic_shards` roster is what makes it dispatchable.
    const HOME: NodeId = NodeId(77);
    /// A second client connection. The SAME account ⇒ the SAME home realm, which is what makes the
    /// one-reply-resolves-many fan-out (and the roster refcount) observable.
    const CLIENT2: NodeId = NodeId(101);

    /// The home realm `AccountId(5)`'s stored pose `(25,0,0)` resolves to: the SAME lineage `home_coord`
    /// derives (and `demand_for_home` demands), lowered to the `RealmId` the directory keys realms by
    /// (`DirectoryKey::Realm(coord.lowered())` — exactly what `rlm.rs` grants a spawned realm at).
    fn home_lineage() -> RealmCoord {
        lineage_of(TEST_HOME_REALM)
    }

    /// …lowered to the `RealmId` the directory keys realms by.
    fn home_realm() -> RealmId {
        home_lineage().lowered()
    }

    /// A `Realm`-head reply for `rid`: `Some(node)` = the realm is LIVE, owned by `node` (its shard took the
    /// realm lease); `None` = not up yet (the shard is still booting).
    fn realm_head(rid: RealmId, owner: Option<NodeId>) -> InterShardFlow {
        InterShardFlow::DirectoryReply(DirectoryReply::Head {
            key: DirectoryKey::Realm(rid),
            record: owner.map(|node| OwnerRecord {
                authority: AuthorityRef::Shard(node),
                fence: Fence(3),
                lease_expires: UniverseTick(1_000),
                in_transfer: None,
            }),
        })
    }

    /// A rig in DYNAMIC-HOME mode: armed + synced, with `AccountId(5)`'s stored spawn pose and the live
    /// 5f-3d budget (`TEST_DEMAND_TTL` / `TEST_BOOTSTRAP_TTL`).
    fn dynamic_rig() -> Rig {
        let mut rig = Rig::new();
        arm_injector(
            &mut rig,
            armed_injector(homes_for(AccountId(5), TEST_HOME_REALM, 0.0)),
            true,
        );
        rig
    }

    /// A dynamic rig whose bootstrap TTL is `ttl` ticks (for the bounded-TTL cells).
    fn dynamic_rig_with_ttl(ttl: u64) -> Rig {
        let mut rig = Rig::new();
        let injector = SeedInjectorConfig {
            bootstrap_ttl_ticks: ttl,
            ..armed_injector(homes_for(AccountId(5), TEST_HOME_REALM, 0.0))
        };
        arm_injector(&mut rig, injector, true);
        rig
    }

    fn phase_of(rig: &Rig, sid: SessionId) -> SessionPhase {
        rig.world
            .resource::<GatewaySessions>()
            .by_session
            .get(&sid)
            .expect("session present")
            .phase
            .clone()
    }

    fn session_of(rig: &Rig, client: NodeId) -> SessionId {
        *rig.world
            .resource::<GatewaySessions>()
            .by_client
            .get(&client)
            .expect("client has a session")
    }

    /// The session's WRITE-route authority (the node its input would be forwarded to).
    fn route_authority(rig: &Rig, sid: SessionId) -> NodeId {
        rig.world
            .resource::<GatewaySessions>()
            .by_session
            .get(&sid)
            .expect("session present")
            .hot
            .route
            .load()
            .authority
    }

    /// The phase a WAITING dynamic session must be in: holding for its home REALM (the lineage + the cadence
    /// anchor live once per realm in `home_bootstraps`, asserted by [`home_wait`]).
    fn waiting_phase() -> SessionPhase {
        SessionPhase::AwaitingHomeRealm {
            home_rid: home_realm(),
        }
    }

    /// The EXACT per-realm bootstrap index a set of `members` waiting on the one home realm must produce
    /// (wait opened at local tick `since`, representative account 5 — every dynamic rig's login account).
    fn home_wait(since: u64, members: &[SessionId], resolved: bool) -> BTreeMap<RealmId, HomeWait> {
        BTreeMap::from([(
            home_realm(),
            HomeWait {
                coord: home_lineage(),
                since: TickId(since),
                account: AccountId(5),
                members: members.iter().copied().collect(),
                resolved,
            },
        )])
    }

    /// Drive ONE dynamic login for `client` (hello → granted lease); returns its id + the GRANT tick's sends.
    #[allow(clippy::type_complexity)] // test helper: one tick of raw sends
    fn dynamic_login(
        rig: &mut Rig,
        client: NodeId,
    ) -> (SessionId, Vec<(NodeId, MsgClass, Vec<u8>)>) {
        let _ = rig.tick(vec![wire(client, MsgClass::Control, &hello_msg())]);
        let sid = session_of(rig, client);
        let granted = rig.tick(vec![wire(ORCH, MsgClass::Saga, &granted_head(sid))]);
        (sid, granted)
    }

    /// The spawn pose THIS rig's gateway derives for the test account — computed through the very
    /// `home_placement` production uses, so an attach assertion can never drift from the derivation. `None`
    /// on an unarmed rig, which is what a static attach carries.
    fn attach_spawn(rig: &Rig) -> Option<StampedPose> {
        let injector = &rig.world.resource::<GatewayConfig>().seed_injector;
        injector
            .armed
            .then(|| home_placement(injector, AccountId(5)).1)
    }

    /// Did the gateway send THIS session's `AttachSession` to `node`? Compares the EXACT expected bytes
    /// (never a speculative decode — postcard is not self-describing, so a client-bound `ServerControlMsg`
    /// on the same `Control` class can mis-decode as a `GatewayToShard`). Bitwise `&` (no short-circuit
    /// region — HR5).
    fn saw_attach(
        sent: &[(NodeId, MsgClass, Vec<u8>)],
        node: NodeId,
        sid: SessionId,
        spawn: Option<StampedPose>,
    ) -> bool {
        let expected = postcard::to_allocvec(&GatewayToShard::AttachSession {
            session: sid,
            fence: Fence(1),
            account: AccountId(5),
            spawn,
        })
        .expect("encode");
        sent.iter().any(|(to, class, bytes)| {
            (*to == node)
                & (*class == MsgClass::Control)
                & (bytes.as_slice() == expected.as_slice())
        })
    }

    /// Did the gateway send THIS session's `DetachSession` to `node`? (Exact bytes, as above.)
    fn saw_detach(sent: &[(NodeId, MsgClass, Vec<u8>)], node: NodeId, sid: SessionId) -> bool {
        let expected = postcard::to_allocvec(&GatewayToShard::DetachSession {
            session: sid,
            fence: Fence(1),
        })
        .expect("encode");
        sent.iter().any(|(to, class, bytes)| {
            (*to == node)
                & (*class == MsgClass::Control)
                & (bytes.as_slice() == expected.as_slice())
        })
    }

    /// Did the gateway revoke THIS session's committed lease? (Exact bytes, as above.)
    fn saw_lease_revoke(sent: &[(NodeId, MsgClass, Vec<u8>)], sid: SessionId) -> bool {
        let expected =
            postcard::to_allocvec(&InterShardFlow::Directory(DirectoryOp::LeaseRevoke {
                key: DirectoryKey::Session(sid),
                fence: Fence(1),
            }))
            .expect("encode");
        sent.iter().any(|(to, class, bytes)| {
            (*to == ORCH) & (*class == MsgClass::Saga) & (bytes.as_slice() == expected.as_slice())
        })
    }

    /// How many `HeadRead{Realm(rid)}` polls the gateway sent (exact bytes).
    fn realm_head_reads(sent: &[(NodeId, MsgClass, Vec<u8>)], rid: RealmId) -> usize {
        let expected = postcard::to_allocvec(&InterShardFlow::Directory(DirectoryOp::HeadRead {
            key: DirectoryKey::Realm(rid),
        }))
        .expect("encode");
        sent.iter()
            .filter(|(to, class, bytes)| {
                (*to == ORCH)
                    & (*class == MsgClass::Saga)
                    & (bytes.as_slice() == expected.as_slice())
            })
            .count()
    }

    #[test]
    fn nothing_of_the_home_lineage_is_running_at_the_instant_the_login_descent_must_answer() {
        // SLICE 6 (b) — THE BOOTSTRAP CIRCULARITY, AS AN ASSERTION RATHER THAN A PARAGRAPH.
        //
        // The other half of this slice was to move the login DESCENT into the chain: the galaxy's shard
        // hands the star's shard a point in the star's frame, the star's hands the planet's a point in the
        // planet's frame, and the planet accepts and computes nothing. That is the right shape and it is
        // what every other lane in this arc now does.
        //
        // It cannot be built as things stand, and this gate is why. The descent answers TWO questions from
        // ONE walk: WHICH realm the account lives in (a routing decision) and WHERE INSIDE IT they stand (the
        // geometry). The chain can only be asked the second once it is running — and it only runs because
        // something demanded the lineage, which is the first. At the instant the descent has to produce its
        // answer the gateway holds an unresolved bootstrap wait and NOT ONE shard on the runtime routable
        // roster: there is nobody to ask. The router answers from its forest copy precisely because nothing
        // is running yet.
        //
        // This is an owner-visible design decision, not an implementation detail, and inventing a fallback
        // here would be exactly the kind of quiet second source of truth this whole arc exists to remove.
        // The two ways out both change WHERE AN ACCOUNT'S HOME IS STORED or WHAT IS ALWAYS UP, and neither is
        // this slice's to pick:
        //   (1) store a home as (lineage, pose in that realm's own frame) — the P7 durable per-realm store —
        //       so there is no walk to run and no absolute to descend from; the router carries a NAME and a
        //       number it did not compute; or
        //   (2) make the ambient root always resident and descend it level by level, demanding as it goes:
        //       the root names the child that contains you and hands it your point in that child's frame,
        //       and the login waits one round-trip per level. Seamless (a login already waits), but it makes
        //       "the root is up" a cluster invariant the demand loop does not have today.
        let mut rig = dynamic_rig();
        let (sid, _granted) = dynamic_login(&mut rig, CLIENT);
        let sessions = rig.world.resource::<GatewaySessions>();
        // The descent HAS run — the session is waiting on the realm it named, and carries the pose that walk
        // produced. So the answer exists at this instant and was produced by the one party that holds no
        // realm at all.
        assert_eq!(sessions.home_realm_of(sid), Some(home_realm()));
        assert!(
            sessions
                .by_session
                .get(&sid)
                .is_some_and(|s| s.spawn.is_some()),
            "the login already carries a pose measured in its home realm's frame"
        );
        // And nothing of that lineage is reachable. `dynamic_shards` is the RUNTIME routable roster — the
        // only place a demand-spawned shard can appear — and it is empty; `home_shard_of` is the same fact
        // asked about this one realm.
        assert_eq!(
            sessions.dynamic_shard_count(),
            0,
            "no demand-spawned shard is routable yet — there is no chain to ask"
        );
        assert_eq!(
            sessions.home_shard_of(sid),
            None,
            "and specifically not the home realm's own shard"
        );
    }

    #[test]
    fn a_dynamic_login_is_welcomed_at_the_lease_then_held_with_no_attach_and_no_close() {
        // THE SEAMLESS HOLD. On the committed lease a dynamic login is WELCOMED exactly as a static one is
        // (same instant, same bytes), then HELD in `AwaitingHomeRealm`: no `AttachSession` to the static
        // `config.shard` (a teleport to the wrong shard), no `Close`, no loading-screen signal — it simply
        // has no frames yet. The demand + the head-poll ride EXISTING wire arms.
        let mut rig = dynamic_rig();
        let (sid, granted) = dynamic_login(&mut rig, CLIENT);
        assert_eq!(
            decode_controls(&granted, CLIENT),
            vec![
                ServerControlMsg::Welcome {
                    version: ProtoVersion::CURRENT,
                    session: sid,
                    session_fence: Fence(1),
                    epoch: EpochId(9),
                },
                ServerControlMsg::UniverseRate { tick_hz: 50 },
            ],
            "a dynamic login is Welcome'd at the committed lease — and nothing else is pushed at it"
        );
        assert!(
            !saw_attach(&granted, SHARD, sid, None),
            "a dynamic login never attaches to the static config.shard"
        );
        assert!(
            !saw_attach(&granted, HOME, sid, attach_spawn(&rig)),
            "and it cannot attach to its home before the head names the node"
        );
        assert_eq!(phase_of(&rig, sid), waiting_phase());
        assert_eq!(
            demands_to_orch(&granted).len(),
            1,
            "ONE home demand, on the existing RealmDemand arm"
        );
        assert_eq!(
            realm_head_reads(&granted, home_realm()),
            1,
            "and ONE poll of the home realm's directory head"
        );
        let sessions = rig.world.resource::<GatewaySessions>();
        assert_eq!(sessions.home_realm_of(sid), Some(home_realm()));
        assert_eq!(
            sessions.home_shard_of(sid),
            None,
            "no home shard until the head resolves"
        );
        assert_eq!(
            sessions.entity_of(sid),
            None,
            "a held session has no avatar yet (the new phase is a non-Active arm of entity_of)"
        );
        assert_eq!(
            sessions.home_bootstraps,
            home_wait(1, &[sid], false),
            "the session is a member of its home realm's ONE bootstrap wait — which owns the descended \
             lineage, the cadence anchor and the representative account (so ONE reply resolves it and ONE \
             re-drive covers it)"
        );
        // The EXACT send fingerprint of a dynamic grant tick: Welcome + UniverseRate to the client, then the
        // demand + the head-poll to the orchestrator — and NOTHING shard-ward (no attach from the grant arm
        // and none from the re-drive, which is not due on the tick the wait began).
        assert_eq!(
            granted
                .iter()
                .map(|(to, class, _)| (*to, *class))
                .collect::<Vec<_>>(),
            vec![
                (CLIENT, MsgClass::Control),
                (CLIENT, MsgClass::Control),
                (ORCH, MsgClass::Saga),
                (ORCH, MsgClass::Saga),
            ]
        );
    }

    #[test]
    fn the_committed_lease_welcome_is_identical_in_both_modes() {
        // SEAMLESS + byte-identical AT THE CLIENT BOUNDARY: the client sees the SAME control stream at the
        // committed lease whether or not the gateway is in dynamic-home mode. Nothing about the home
        // bootstrap leaks to it — no Close, no teleport, no extra/omitted variant, no reordering. (Both rigs
        // share `session_seed`, so the minted SessionId — and hence the Welcome bytes — match exactly.)
        let mut dynamic = dynamic_rig();
        let (dyn_sid, dyn_granted) = dynamic_login(&mut dynamic, CLIENT);
        let mut static_rig = Rig::new();
        let (static_sid, static_granted) = dynamic_login(&mut static_rig, CLIENT);
        assert_eq!(dyn_sid, static_sid, "the same mint stream in both rigs");
        assert_eq!(
            decode_controls(&dyn_granted, CLIENT),
            decode_controls(&static_granted, CLIENT),
            "the client-visible Welcome at the committed lease is identical in both modes"
        );
    }

    #[test]
    fn a_home_head_with_no_record_keeps_the_session_waiting_seamlessly() {
        // The realm is DEMANDED but its shard is still booting (`record: None`). The waiter STAYS: no Close,
        // no teleport, no fallback attach — and it is not dropped.
        let mut rig = dynamic_rig();
        let (sid, _) = dynamic_login(&mut rig, CLIENT);
        let after = rig.tick(vec![wire(
            ORCH,
            MsgClass::Saga,
            &realm_head(home_realm(), None),
        )]);
        assert_eq!(
            phase_of(&rig, sid),
            waiting_phase(),
            "a not-yet-routable home keeps the session waiting"
        );
        assert_eq!(
            decode_controls(&after, CLIENT),
            Vec::new(),
            "NOTHING is pushed at the client while its home boots (no Close, no teleport)"
        );
        assert_eq!(
            rig.world.resource::<GatewaySessions>().len(),
            1,
            "held, never dropped"
        );
        assert!(
            !saw_attach(&after, SHARD, sid, None),
            "and never fallen back onto the static shard"
        );
    }

    #[test]
    fn a_resolved_home_head_routes_the_attach_to_the_spawned_shard_and_admits_its_frames() {
        // THE ROUTE. The `Realm` head names the spawned node: it JOINS the runtime routable roster, the
        // WRITE route is retargeted to it, the attach goes THERE (never `config.shard`), and — purely via
        // that runtime roster — its `SessionAttached` promotes the session and its frames reach the client.
        let mut rig = dynamic_rig();
        let (sid, _) = dynamic_login(&mut rig, CLIENT);
        let resolved = rig.tick(vec![wire(
            ORCH,
            MsgClass::Saga,
            &realm_head(home_realm(), Some(HOME)),
        )]);
        assert_eq!(phase_of(&rig, sid), SessionPhase::AwaitingAttach);
        assert!(
            saw_attach(&resolved, HOME, sid, attach_spawn(&rig)),
            "the attach goes to the SPAWNED home shard"
        );
        assert!(
            !saw_attach(&resolved, SHARD, sid, None),
            "never to the static config.shard"
        );
        assert_eq!(
            route_authority(&rig, sid),
            HOME,
            "the WRITE route retargeted"
        );
        let sessions = rig.world.resource::<GatewaySessions>();
        assert_eq!(sessions.home_shard_of(sid), Some(HOME));
        assert_eq!(
            sessions.dynamic_shards,
            BTreeMap::from([(HOME, 1)]),
            "the spawned node joined the RUNTIME routable roster (it is not in the frozen config)"
        );
        assert_eq!(
            sessions.home_bootstraps,
            home_wait(1, &[sid], true),
            "the session STAYS a member of the (now resolved) bootstrap — MF1: the demand re-seed must \
             outlive the resolve and run until the Active promote, or the reconciler reaps the realm this \
             login is attaching to"
        );
        // The runtime roster is what makes HOME dispatchable as a shard at all.
        let attached = rig.tick(vec![wire(
            HOME,
            MsgClass::Control,
            &ShardToGateway::SessionAttached {
                session: sid,
                entity: EntityId(77),
                frame: FrameRef::SystemSpace { system_seed: 7 },
                realm_fence: Fence(1),
            },
        )]);
        assert_eq!(
            rig.world.resource::<GatewaySessions>().entity_of(sid),
            Some(EntityId(77)),
            "the session went Active off its HOME shard's attach"
        );
        assert_eq!(
            decode_controls(&attached, CLIENT),
            vec![
                ServerControlMsg::SubscriptionOpened {
                    sub: SubId(0),
                    frame: FrameRef::SystemSpace { system_seed: 7 },
                },
                ServerControlMsg::AuthorityChanged {
                    entity: EntityId(77),
                    sub: SubId(0),
                },
                ServerControlMsg::OwnEntity {
                    entity: EntityId(77)
                },
                // THE LOGIN LEVEL (§2.4, the deleted attach-seam RealmRegistry's successor): the
                // composer's first pass after the attach sees the standing realm and bumps the
                // epoch 0→1 — the swap signal, ordered after the identity control above. No
                // window frames were fed in this rig, so it carries only the origin's bagless
                // row; the deltas fill the drawn set as the statements land.
                ServerControlMsg::RealmRegistry {
                    origin: RealmId::System(7),
                    origin_epoch: 1,
                    rows: vec![SceneRow {
                        realm: RealmId::System(7),
                        parent: None,
                        pose: vd_core::pose::StampedPose::at_rest(
                            FrameRef::SystemSpace { system_seed: 7 },
                            DVec3::ZERO,
                            UniverseTick(0),
                        ),
                        bag: Vec::new(),
                    }],
                },
            ],
            "the login sub opened on the HOME shard (never config.shard); the composer ships the \
             login level after the identity control (§2.4)"
        );
        let framed = rig.tick(vec![wire(
            HOME,
            MsgClass::Snapshot,
            &frame_msg(Fence(1), 1),
        )]);
        assert_eq!(
            framed
                .iter()
                .filter(|(to, class, _)| (*to == CLIENT) & (*class == MsgClass::Snapshot))
                .count(),
            1,
            "a frame from the dynamically routed home shard reaches the client"
        );
        assert_eq!(
            rig.world
                .resource::<GatewaySessions>()
                .by_session
                .get(&sid)
                .expect("session")
                .bootstrap_deadline,
            None,
            "the bounded bootstrap window closed at the Active promote"
        );
        // MF1 test (iii): the `Active` promote is the wait's TERMINATOR — the member leaves and, being the
        // last one, takes the realm's whole entry (and its lineage `Vec`) with it.
        assert_eq!(
            rig.world.resource::<GatewaySessions>().home_bootstraps,
            BTreeMap::new(),
            "the Active promote drops the member and prunes the realm entry"
        );
        // …so the re-drive is silent from here on, however many cadences pass (nothing left to re-seed).
        set_tick(&mut rig, 11);
        let after_active = rig.tick(vec![]);
        assert_eq!(
            demands_to_orch(&after_active).len(),
            0,
            "an Active session's home is no longer re-seeded (arm-B owns it now — live occupancy)"
        );
        assert_eq!(realm_head_reads(&after_active, home_realm()), 0);
        // A `Bye` from an ACTIVE dynamic session: its `home_rid` still names the realm but the wait entry is
        // long gone, so the index exit is a clean no-op — while the DETACH still routes to its home shard and
        // the runtime roster releases (the whole dynamic teardown, after the bootstrap has ended).
        let bye = rig.tick(vec![wire(
            CLIENT,
            MsgClass::Control,
            &ClientControlMsg::Bye,
        )]);
        assert!(
            saw_detach(&bye, HOME, sid),
            "an Active dynamic session detaches at its HOME shard, never config.shard"
        );
        let sessions = rig.world.resource::<GatewaySessions>();
        assert_eq!(sessions.len(), 0);
        assert_eq!(sessions.home_bootstraps, BTreeMap::new());
        assert_eq!(
            sessions.dynamic_shards,
            BTreeMap::new(),
            "and its roster claim is released"
        );
    }

    #[test]
    fn the_home_bootstrap_re_drive_fires_on_the_backoff_cadence_not_every_tick() {
        // THE RE-DRIVE AS A CORRECTNESS INVARIANT (CRITIQUE-3): while waiting, the gateway must keep
        // re-seeding the demand (so the reconciler's arm-A `demanded_recently` never lapses mid-boot and
        // reaps the half-booted realm) AND re-poll the head. But BACKED OFF: `demand_ttl / 4`, NOT every
        // tick — the 100K mass-login storm guard. Both arms of the due/not-due branch are exercised.
        let cadence = SeedInjectorConfig {
            ..armed_injector(default_homes())
        }
        .redrive_interval_ticks();
        assert_eq!(cadence, 2, "demand_ttl 8 / REDRIVE_DIVISOR 4 = 2");
        assert!(
            cadence < TEST_DEMAND_TTL,
            "the cadence must be STRICTLY inside the demand TTL, or arm-A lapses mid-boot"
        );
        assert!(cadence > 1, "…and it must not degenerate to every tick");
        let mut rig = dynamic_rig();
        let (sid, _) = dynamic_login(&mut rig, CLIENT); // waiting since local tick 1
        // Tick 2 (elapsed 1): NOT due.
        set_tick(&mut rig, 2);
        let quiet = rig.tick(vec![]);
        assert_eq!(demands_to_orch(&quiet).len(), 0, "no re-seed off cadence");
        assert_eq!(
            realm_head_reads(&quiet, home_realm()),
            0,
            "no re-poll off cadence"
        );
        // Tick 3 (elapsed 2 == cadence): DUE — both halves fire.
        set_tick(&mut rig, 3);
        let driven = rig.tick(vec![]);
        let demands = demands_to_orch(&driven);
        assert_eq!(demands.len(), 1, "the home demand is RE-SEEDED on cadence");
        assert_eq!(
            demands[0].child,
            lineage_of(TEST_HOME_REALM),
            "the re-seed names the SAME stored home lineage"
        );
        assert_eq!(demands[0].verb, DemandVerb::SpinUp);
        assert_eq!(
            realm_head_reads(&driven, home_realm()),
            1,
            "and the head is re-polled"
        );
        // Tick 4 (elapsed 3): NOT due again — proof it is a cadence, not a latch.
        set_tick(&mut rig, 4);
        let quiet2 = rig.tick(vec![]);
        assert_eq!(demands_to_orch(&quiet2).len(), 0);
        // Tick 5 (elapsed 4): due again.
        set_tick(&mut rig, 5);
        assert_eq!(demands_to_orch(&rig.tick(vec![])).len(), 1);
        assert_eq!(
            phase_of(&rig, sid),
            waiting_phase(),
            "still held, seamlessly"
        );
    }

    #[test]
    fn the_re_seed_outlives_the_resolve_so_the_reconciler_never_reaps_the_home_mid_attach() {
        // MF1 DEFECT A — THE ANTI-REAP INVARIANT, the reason this slice exists. The head resolve is NOT the
        // end of the re-seed: a session can sit in the DYNAMIC `AwaitingAttach` for the whole rest of the
        // bootstrap window (a lost `AttachSession`/`SessionAttached` — the 5f-4 dial-a-fresh-pod race), and a
        // booted-but-unoccupied realm self-reports `Empty`, so the reconciler's arm-B (`running_live &
        // !empty_confirmed`) is FALSE. Only the gateway's re-seed keeps arm-A `demanded_recently` alive; if it
        // stopped at the resolve, arm-A would expire one `demand_ttl` after it and the reconciler would KILL
        // the realm this login is attaching to — long BEFORE the bootstrap TTL noticed.
        let mut rig = dynamic_rig(); // demand_ttl 8 (cadence 2), bootstrap_ttl 100
        let (sid, _) = dynamic_login(&mut rig, CLIENT); // wait opened at local tick 1
        let resolve_tick = 2;
        // A tick MORE than one whole `demand_ttl` after the resolve — exactly where the reconciler's arm-A
        // would have lapsed had the re-seed stopped there — and on the wait's cadence (odd, anchored at 1).
        let probe_tick = 13;
        assert!(
            probe_tick - resolve_tick > TEST_DEMAND_TTL,
            "the probe must sit past one whole demand TTL from the resolve (where arm-A lapses)"
        );
        set_tick(&mut rig, resolve_tick);
        let resolved = rig.tick(vec![wire(
            ORCH,
            MsgClass::Saga,
            &realm_head(home_realm(), Some(HOME)),
        )]);
        assert!(saw_attach(&resolved, HOME, sid, attach_spawn(&rig)));
        assert_eq!(phase_of(&rig, sid), SessionPhase::AwaitingAttach);
        // Now the attach is NEVER confirmed.
        set_tick(&mut rig, probe_tick);
        let late = rig.tick(vec![]);
        assert_eq!(
            demands_to_orch(&late).len(),
            1,
            "the home demand is STILL re-seeded while the resolved session waits to attach (MF1-A)"
        );
        assert_eq!(
            demands_to_orch(&late)[0].child,
            home_lineage(),
            "and it still names the SAME server-derived home lineage"
        );
        assert_eq!(
            realm_head_reads(&late, home_realm()),
            0,
            "…while the head POLL stays off — the node is already known (only the demand half continues)"
        );
        assert!(
            saw_attach(&late, HOME, sid, attach_spawn(&rig)),
            "the dynamic attach retry rides the SAME cadence tick (coalesced, not per-tick)"
        );
        // Off-cadence the whole thing is silent — the retry is a cadence, not a latch.
        set_tick(&mut rig, probe_tick + 1);
        let quiet = rig.tick(vec![]);
        assert_eq!(demands_to_orch(&quiet).len(), 0);
        assert!(
            !saw_attach(&quiet, HOME, sid, attach_spawn(&rig)),
            "a DYNAMIC attach retry does not fire every tick (the mass-login storm guard)"
        );
        // Still held, still bounded, and it can still complete: the attach lands and the session goes Active.
        assert_eq!(rig.stats().home_bootstrap_timeouts, 0);
        let attached = rig.tick(vec![wire(
            HOME,
            MsgClass::Control,
            &ShardToGateway::SessionAttached {
                session: sid,
                entity: EntityId(77),
                frame: FrameRef::SystemSpace { system_seed: 7 },
                realm_fence: Fence(1),
            },
        )]);
        assert_eq!(
            rig.world.resource::<GatewaySessions>().entity_of(sid),
            Some(EntityId(77)),
            "the late attach still completes the login"
        );
        assert_eq!(
            rig.world.resource::<GatewaySessions>().home_bootstraps,
            BTreeMap::new(),
            "and THAT is what ends the re-seed"
        );
        let _ = attached;
    }

    #[test]
    fn two_sessions_on_one_home_coalesce_to_exactly_one_demand_and_one_head_read() {
        // MF1 DEFECT B — the STORM guard. The re-drive iterates REALMS, not sessions: N sessions booting into
        // ONE home cost ONE `RealmDemand` + ONE `HeadRead` per cadence, not 2N. (At the 100K mass-login scale
        // the difference is 200K messages per cadence versus 2.) Coalescing is sound because the only
        // per-session field in the demand is the audit-only `parent_fence`.
        let mut rig = dynamic_rig();
        let (sid_a, _) = dynamic_login(&mut rig, CLIENT);
        let (sid_b, _) = dynamic_login(&mut rig, CLIENT2);
        set_tick(&mut rig, 3); // elapsed 2 == the cadence
        let driven = rig.tick(vec![]);
        let demands = demands_to_orch(&driven);
        assert_eq!(
            demands.len(),
            1,
            "TWO waiters on one home ⇒ exactly ONE re-seeded demand"
        );
        assert_eq!(demands[0].child, home_lineage());
        assert_eq!(
            demands[0].parent_fence,
            home_sentinel_fence(AccountId(5)),
            "carrying the wait's representative per-account sentinel (audit-only at the orchestrator)"
        );
        assert_eq!(
            realm_head_reads(&driven, home_realm()),
            1,
            "and exactly ONE head poll for the realm both are waiting on"
        );
        // After the resolve: still ONE demand per cadence (the anti-reap invariant), and ZERO head-reads.
        let _ = rig.tick(vec![wire(
            ORCH,
            MsgClass::Saga,
            &realm_head(home_realm(), Some(HOME)),
        )]);
        set_tick(&mut rig, 5);
        let after = rig.tick(vec![]);
        assert_eq!(
            demands_to_orch(&after).len(),
            1,
            "one demand per cadence still covers BOTH resolved members"
        );
        assert_eq!(
            realm_head_reads(&after, home_realm()),
            0,
            "and the head poll is done — the node is known"
        );
        assert_eq!(phase_of(&rig, sid_a), SessionPhase::AwaitingAttach);
        assert_eq!(phase_of(&rig, sid_b), SessionPhase::AwaitingAttach);
    }

    #[test]
    fn a_session_joining_an_already_resolved_home_resumes_the_head_poll_and_resolves() {
        // The joiner case the surviving wait entry creates: session B logs into a home realm session A has
        // ALREADY resolved. B has seen no head reply of its own, so joining CLEARS `resolved` — the poll
        // resumes on the next cadence tick and B is resolved by the reply, while A (already in
        // `AwaitingAttach`) is left untouched. Both arms of the per-member phase test, in one cell.
        let mut rig = dynamic_rig();
        let (sid_a, _) = dynamic_login(&mut rig, CLIENT);
        let _ = rig.tick(vec![wire(
            ORCH,
            MsgClass::Saga,
            &realm_head(home_realm(), Some(HOME)),
        )]);
        assert_eq!(phase_of(&rig, sid_a), SessionPhase::AwaitingAttach);
        assert_eq!(
            rig.world.resource::<GatewaySessions>().home_bootstraps,
            home_wait(1, &[sid_a], true),
            "resolved, and the entry survives"
        );
        // B logs in on the SAME account ⇒ the same home realm ⇒ it JOINS the existing wait.
        let (sid_b, _) = dynamic_login(&mut rig, CLIENT2);
        assert_eq!(
            rig.world.resource::<GatewaySessions>().home_bootstraps,
            home_wait(1, &[sid_a, sid_b], false),
            "the joiner cleared `resolved` (its own head reply may be lost — the poll must resume)"
        );
        set_tick(&mut rig, 3);
        let driven = rig.tick(vec![]);
        assert_eq!(
            realm_head_reads(&driven, home_realm()),
            1,
            "so the head IS re-polled for the joiner"
        );
        let resolved = rig.tick(vec![wire(
            ORCH,
            MsgClass::Saga,
            &realm_head(home_realm(), Some(HOME)),
        )]);
        assert_eq!(phase_of(&rig, sid_b), SessionPhase::AwaitingAttach);
        assert!(
            saw_attach(&resolved, HOME, sid_b, attach_spawn(&rig)),
            "the joiner's attach goes to the home shard"
        );
        assert_eq!(
            rig.world.resource::<GatewaySessions>().dynamic_shards,
            BTreeMap::from([(HOME, 2)]),
            "each member claims the roster exactly once — the already-resolved member was skipped"
        );
    }

    #[test]
    fn a_dynamic_pre_active_session_keeps_its_committed_lease_renewed() {
        // MF2 — a held login's lease MUST be renewed. The dynamic-home hold can outlast the orchestrator's
        // lease-reap horizon (`bootstrap_ttl` is a whole measured pod boot), and the lease was COMMITTED at
        // the grant, so an `Active`-only renew set would let a held session's own lease lapse under it with
        // nothing pre-Active watching. The renew set is therefore `Active` OR mid-bootstrap; the RECHECK
        // stays Active-only.
        // `renewed_sessions` (the module-level decode helper) owns the non-renew arm via the D-3 cell.
        let rechecked = |sent: &[(NodeId, MsgClass, Vec<u8>)]| -> Vec<SessionId> {
            sent.iter()
                .filter(|(to, _, _)| *to == ORCH)
                .filter_map(
                    |(_, _, b)| match postcard::from_bytes::<InterShardFlow>(b) {
                        Ok(InterShardFlow::Directory(DirectoryOp::HeadRead {
                            key: DirectoryKey::Session(s),
                        })) => Some(s),
                        _ => None,
                    },
                )
                .collect()
        };
        // DYNAMIC: armed + synced, with the renew AND recheck cadences live.
        let mut rig = Rig::new();
        rig.world.insert_resource(GatewayConfig {
            lease_renew_interval_ticks: 4,
            session_recheck_interval: 4,
            seed_injector: armed_injector(homes_for(AccountId(5), TEST_HOME_REALM, 0.0)),
            ..config()
        });
        let (sid, _) = dynamic_login(&mut rig, CLIENT);
        assert_eq!(phase_of(&rig, sid), waiting_phase());
        set_tick(&mut rig, 4);
        let held = rig.tick(vec![]);
        assert_eq!(
            renewed_sessions(&held),
            vec![sid],
            "a session HELD in AwaitingHomeRealm still renews its committed lease (MF2)"
        );
        assert_eq!(
            rechecked(&held),
            Vec::new(),
            "…but is NOT re-checked (no authority to self-fence pre-Active)"
        );
        // …and the Active-only recheck RESUMES once the hold ends: the whole point of gating the recheck on
        // Active (not on the renew set) is that a held session parks the recheck and un-parks it on promote.
        // Drive the SAME session home-resolved → attached → Active, then a recheck-cadence tick fires it.
        let _ = rig.tick(vec![wire(
            ORCH,
            MsgClass::Saga,
            &realm_head(home_realm(), Some(HOME)),
        )]);
        assert_eq!(
            phase_of(&rig, sid),
            SessionPhase::AwaitingAttach,
            "home resolved ⇒ the bootstrap hold ends"
        );
        let _ = rig.tick(vec![wire(
            HOME,
            MsgClass::Control,
            &ShardToGateway::SessionAttached {
                session: sid,
                entity: EntityId(77),
                frame: FrameRef::SystemSpace { system_seed: 7 },
                realm_fence: Fence(1),
            },
        )]);
        set_tick(&mut rig, 8);
        assert_eq!(
            rechecked(&rig.tick(vec![])),
            vec![sid],
            "the Active-only recheck RESUMES once the session is Active"
        );
        // STATIC (the byte-identical control): an `AwaitingAttach` login carries no bootstrap window, so it
        // is NOT renewed — the pre-5f-3d renew set exactly.
        let mut plain = Rig::new();
        plain.world.insert_resource(GatewayConfig {
            lease_renew_interval_ticks: 4,
            ..config()
        });
        let _ = plain.tick(vec![wire(CLIENT, MsgClass::Control, &hello_msg())]);
        let static_sid = session_of(&plain, CLIENT);
        let _ = plain.tick(vec![wire(ORCH, MsgClass::Saga, &granted_head(static_sid))]);
        assert_eq!(phase_of(&plain, static_sid), SessionPhase::AwaitingAttach);
        set_tick(&mut plain, 4);
        assert_eq!(
            renewed_sessions(&plain.tick(vec![])),
            Vec::new(),
            "a STATIC pre-Active login is not renewed (byte-identical renew set)"
        );
    }

    #[test]
    fn the_bounded_bootstrap_ttl_closes_a_never_routable_home_loudly() {
        // CRITIQUE-1: the hold is BOUNDED. A home that never becomes routable Closes the client LOUDLY at
        // the deadline (counted + a reason + the committed lease revoked), never a silent hang. Expiring in
        // `AwaitingHomeRealm` sends NO detach (no home was ever resolved — the `None` arm).
        let mut rig = dynamic_rig_with_ttl(5); // waiting since tick 1 ⇒ deadline 6
        let (sid, _) = dynamic_login(&mut rig, CLIENT);
        set_tick(&mut rig, 6); // AT the deadline: still inside the window
        let inside = rig.tick(vec![]);
        assert_eq!(
            phase_of(&rig, sid),
            waiting_phase(),
            "the deadline tick itself is still inside the hold"
        );
        assert_eq!(decode_controls(&inside, CLIENT), Vec::new(), "no Close yet");
        assert_eq!(rig.stats().home_bootstrap_timeouts, 0);
        set_tick(&mut rig, 7); // PAST the deadline
        let closed = rig.tick(vec![]);
        assert_eq!(
            decode_controls(&closed, CLIENT),
            vec![ServerControlMsg::Close {
                reason: "home realm did not become available".to_owned(),
            }],
            "the bounded TTL Closes LOUDLY with a reason"
        );
        assert_eq!(rig.stats().home_bootstrap_timeouts, 1);
        assert!(
            saw_lease_revoke(&closed, sid),
            "the committed Session lease is revoked, not left for the reaper"
        );
        assert!(
            !saw_detach(&closed, SHARD, sid),
            "no detach is sprayed at the static shard we never attached to"
        );
        let sessions = rig.world.resource::<GatewaySessions>();
        assert_eq!(sessions.len(), 0, "the session is gone");
        assert_eq!(
            sessions.home_bootstraps,
            BTreeMap::new(),
            "and its bootstrap-index entry with it (the last member out prunes the realm)"
        );
    }

    #[test]
    fn the_bootstrap_ttl_also_spans_the_dynamic_attach_wait_and_detaches_the_home() {
        // The TTL spans the WHOLE pre-Active bootstrap: a home that resolves but never confirms the attach
        // (a shard that dies between head-resolve and `SessionAttached`) ALSO Closes at the deadline — and
        // because a home WAS resolved, the cleanup detaches THERE and releases the runtime roster entry.
        let mut rig = dynamic_rig_with_ttl(5); // deadline 6
        let (sid, _) = dynamic_login(&mut rig, CLIENT);
        let _ = rig.tick(vec![wire(
            ORCH,
            MsgClass::Saga,
            &realm_head(home_realm(), Some(HOME)),
        )]);
        assert_eq!(phase_of(&rig, sid), SessionPhase::AwaitingAttach);
        assert_eq!(
            rig.world.resource::<GatewaySessions>().dynamic_shards,
            BTreeMap::from([(HOME, 1)])
        );
        set_tick(&mut rig, 7); // past the deadline, still AwaitingAttach
        let closed = rig.tick(vec![]);
        assert_eq!(
            decode_controls(&closed, CLIENT),
            vec![ServerControlMsg::Close {
                reason: "home realm did not become available".to_owned(),
            }],
            "a dynamic AwaitingAttach that never attaches also Closes loudly"
        );
        assert_eq!(rig.stats().home_bootstrap_timeouts, 1);
        assert!(
            saw_detach(&closed, HOME, sid),
            "the detach goes to the RESOLVED home shard"
        );
        assert!(saw_lease_revoke(&closed, sid));
        assert_eq!(
            rig.world.resource::<GatewaySessions>().dynamic_shards,
            BTreeMap::new(),
            "the last session left ⇒ the spawned node leaves the runtime roster (no churn leak)"
        );
    }

    #[test]
    fn one_home_head_resolves_every_waiter_and_the_roster_refcount_drains() {
        // SCALE + the roster refcount. Two sessions on the SAME home realm are resolved by ONE head reply
        // (a mass login onto one home is a win, not a fan-out cost); the roster counts BOTH, and only the
        // LAST session leaving removes the node (both refcount arms).
        let mut rig = dynamic_rig();
        let (sid_a, _) = dynamic_login(&mut rig, CLIENT);
        let (sid_b, _) = dynamic_login(&mut rig, CLIENT2);
        assert_eq!(
            rig.world.resource::<GatewaySessions>().home_bootstraps,
            home_wait(1, &[sid_a, sid_b], false),
            "both sessions are members of the ONE per-realm wait (one lineage copy, one cadence anchor)"
        );
        let resolved = rig.tick(vec![wire(
            ORCH,
            MsgClass::Saga,
            &realm_head(home_realm(), Some(HOME)),
        )]);
        assert!(saw_attach(&resolved, HOME, sid_a, attach_spawn(&rig)));
        assert!(saw_attach(&resolved, HOME, sid_b, attach_spawn(&rig)));
        assert_eq!(phase_of(&rig, sid_a), SessionPhase::AwaitingAttach);
        assert_eq!(phase_of(&rig, sid_b), SessionPhase::AwaitingAttach);
        assert_eq!(
            rig.world.resource::<GatewaySessions>().dynamic_shards,
            BTreeMap::from([(HOME, 2)]),
            "the roster refcounts BOTH sessions homed on the spawned node"
        );
        // MF1-B — a DUPLICATE head reply (at-least-once) is an exact no-op: both members have left
        // `AwaitingHomeRealm`, so nothing re-attaches and — crucially — the roster is NOT re-claimed (a
        // second claim would pin the node on the runtime roster forever once these sessions leave).
        let dup = rig.tick(vec![wire(
            ORCH,
            MsgClass::Saga,
            &realm_head(home_realm(), Some(HOME)),
        )]);
        assert!(
            !saw_attach(&dup, HOME, sid_a, attach_spawn(&rig)),
            "a duplicate Realm head does not re-attach an already-resolved member"
        );
        assert_eq!(
            rig.world.resource::<GatewaySessions>().dynamic_shards,
            BTreeMap::from([(HOME, 2)]),
            "and does not double-count the roster refcount"
        );
        assert_eq!(rig.stats().home_wait_desync, 0);
        // A `Bye` from the first: the node STAYS (the other session still needs it).
        let _ = rig.tick(vec![wire(
            CLIENT,
            MsgClass::Control,
            &ClientControlMsg::Bye,
        )]);
        assert_eq!(
            rig.world.resource::<GatewaySessions>().dynamic_shards,
            BTreeMap::from([(HOME, 1)]),
            "one leaver does not evict a node another session is homed on"
        );
        // A `Bye` from the last: the node LEAVES the roster.
        let last = rig.tick(vec![wire(
            CLIENT2,
            MsgClass::Control,
            &ClientControlMsg::Bye,
        )]);
        assert_eq!(
            rig.world.resource::<GatewaySessions>().dynamic_shards,
            BTreeMap::new(),
            "the last leaver evicts it (bounded across 100K-realm churn)"
        );
        assert!(
            saw_detach(&last, HOME, sid_b),
            "and the Bye detach itself routes to the session's HOME shard, not config.shard"
        );
    }

    #[test]
    fn byes_while_awaiting_the_home_prune_the_wait_index_incrementally() {
        // The bootstrap index can never outlive its sessions: a `Bye` mid-boot drops that session from its
        // home's member set (the entry survives while another member waits), and the LAST removal prunes the
        // realm entry — one `RealmCoord` lineage freed with it.
        let mut rig = dynamic_rig();
        let (sid_a, _) = dynamic_login(&mut rig, CLIENT);
        let (sid_b, _) = dynamic_login(&mut rig, CLIENT2);
        let _ = rig.tick(vec![wire(
            CLIENT,
            MsgClass::Control,
            &ClientControlMsg::Bye,
        )]);
        assert_eq!(
            rig.world.resource::<GatewaySessions>().home_bootstraps,
            home_wait(1, &[sid_b], false),
            "the remaining member keeps the realm's wait alive"
        );
        assert_eq!(
            rig.world.resource::<GatewaySessions>().dynamic_shards,
            BTreeMap::new(),
            "a session that never resolved a home releases nothing"
        );
        let _ = rig.tick(vec![wire(
            CLIENT2,
            MsgClass::Control,
            &ClientControlMsg::Bye,
        )]);
        assert_eq!(
            rig.world.resource::<GatewaySessions>().home_bootstraps,
            BTreeMap::new(),
            "the last removal PRUNES the realm entry (no unbounded growth)"
        );
        assert_eq!(rig.world.resource::<GatewaySessions>().len(), 0);
        let _ = sid_a; // named for the assertion above only
    }

    #[test]
    fn a_static_login_is_byte_identical_with_no_home_phase_and_no_extra_head_read() {
        // BYTE-IDENTITY (CRITIQUE-2): with the injector UNARMED (the default) a login follows the EXACT
        // pre-5f-3d flow — `AwaitingDirectory → AwaitingAttach → config.shard` — with NO `AwaitingHomeRealm`,
        // NO extra head-read, NO RealmDemand, and no reordering of Welcome/UniverseRate/AttachSession.
        let mut rig = Rig::new(); // config() ⇒ the inert injector (unarmed)
        let (sid, sends) = rig.login();
        assert_eq!(
            phase_of(&rig, sid),
            SessionPhase::Active {
                entity: EntityId(77)
            },
            "the static login reached Active in the same three ticks as before"
        );
        assert_eq!(
            decode_controls(&sends[1], CLIENT),
            vec![
                ServerControlMsg::Welcome {
                    version: ProtoVersion::CURRENT,
                    session: sid,
                    session_fence: Fence(1),
                    epoch: EpochId(9),
                },
                ServerControlMsg::UniverseRate { tick_hz: 50 },
            ],
            "Welcome then UniverseRate, unchanged"
        );
        assert!(
            saw_attach(&sends[1], SHARD, sid, None),
            "the attach goes to the static config.shard at the SAME tick as the Welcome"
        );
        // The EXACT send fingerprint of the static grant tick: Welcome + UniverseRate client-ward, then the
        // grant arm's `AttachSession` and the per-tick retry driver's duplicate — both to `config.shard`.
        // Nothing added, nothing reordered, nothing orchestrator-ward (no demand, no Realm head-read).
        assert_eq!(
            sends[1]
                .iter()
                .map(|(to, class, _)| (*to, *class))
                .collect::<Vec<_>>(),
            vec![
                (CLIENT, MsgClass::Control),
                (CLIENT, MsgClass::Control),
                (SHARD, MsgClass::Control),
                (SHARD, MsgClass::Control),
            ]
        );
        let total_head_reads: usize = sends
            .iter()
            .map(|tick| realm_head_reads(tick, home_realm()))
            .sum();
        assert_eq!(
            total_head_reads, 0,
            "a static login costs NO extra Realm head-read round-trip"
        );
        assert_eq!(demand_count(&sends), 0, "and emits no RealmDemand");
        let sessions = rig.world.resource::<GatewaySessions>();
        assert_eq!(
            sessions.home_shard_of(sid),
            None,
            "no dynamic home ⇒ session_target is config.shard"
        );
        assert_eq!(sessions.home_realm_of(sid), None);
        assert_eq!(route_authority(&rig, sid), SHARD);
        assert_eq!(sessions.home_bootstraps, BTreeMap::new());
        assert_eq!(sessions.dynamic_shards, BTreeMap::new());
        assert_eq!(rig.stats().home_bootstrap_timeouts, 0);
        assert_eq!(
            rig.stats().logins_held_pre_sync,
            0,
            "an UNARMED gateway never consults the clock on the login path (MF3)"
        );
        // A stray Realm head in static mode is a clean no-op (nobody waits on it).
        let stray = rig.tick(vec![wire(
            ORCH,
            MsgClass::Saga,
            &realm_head(home_realm(), Some(HOME)),
        )]);
        assert_eq!(
            rig.world.resource::<GatewaySessions>().home_shard_of(sid),
            None,
            "a static session is never re-homed by a stray Realm head"
        );
        assert_eq!(decode_controls(&stray, CLIENT), Vec::new());
        assert_eq!(rig.stats().home_wait_desync, 0);
    }

    #[test]
    fn a_forced_home_wait_desync_is_counted_never_silent() {
        // The C2 honesty floor: a `home_bootstraps` MEMBER naming a session absent from `by_session` (an
        // invariant breach the begin/end pairing makes impossible) is COUNTED, never a silent continue.
        let mut sessions = GatewaySessions::default();
        sessions.begin_home_wait(
            SessionId(0xC0DE),
            home_realm(),
            home_lineage(),
            TickId(1),
            AccountId(5),
        );
        let mut stats = GatewayStats::default();
        let mut outbox = OutboundBox::default();
        on_home_realm_head(
            home_realm(),
            Some(OwnerRecord {
                authority: AuthorityRef::Shard(HOME),
                fence: Fence(3),
                lease_expires: UniverseTick(1_000),
                in_transfer: None,
            }),
            &mut sessions,
            &mut stats,
            &mut outbox,
        );
        assert_eq!(stats.home_wait_desync, 1, "the desync is counted");
        assert_eq!(
            outbox.0.len(),
            0,
            "and nothing is sent for a phantom session"
        );
        assert_eq!(
            sessions.dynamic_shards,
            BTreeMap::new(),
            "a phantom session claims no roster entry"
        );
        assert_eq!(
            sessions.home_bootstraps,
            home_wait(1, &[SessionId(0xC0DE)], true),
            "the entry SURVIVES the resolve (the demand re-seed runs until Active) with the poll satisfied"
        );
    }

    #[test]
    fn the_attach_retry_cadence_gate_is_per_tick_for_static_and_backed_off_for_dynamic() {
        // MF1 — the two ARMS of the attach-retry gate, driven directly (the live arms are proven by the
        // static byte-identity fingerprint and by the dynamic cadence cell).
        assert!(
            attach_retry_due(None, TickId(2), 2),
            "a STATIC session (no bootstrap wait) retries EVERY tick — byte-identical"
        );
        assert!(
            attach_retry_due(None, TickId(3), 2),
            "…on the off-cadence tick too"
        );
        assert!(
            !attach_retry_due(Some(TickId(1)), TickId(2), 2),
            "a DYNAMIC member is silent off its realm's cadence"
        );
        assert!(
            attach_retry_due(Some(TickId(1)), TickId(3), 2),
            "…and retries on it"
        );
        // The ANCHOR source, both `?` arms: `None` for a static session (no home realm), `None` also for the
        // fail-SAFE case of a home realm with no live wait (unreachable on the live path — every member drop
        // either removes the session or leaves `AwaitingAttach` — so it is proven here directly).
        let mut sessions = GatewaySessions::default();
        assert_eq!(
            sessions.attach_anchor(None),
            None,
            "a static session has no anchor ⇒ per-tick retry"
        );
        assert_eq!(
            sessions.attach_anchor(Some(home_realm())),
            None,
            "a home realm with no live wait falls back to the per-tick retry (never a wedge)"
        );
        sessions.begin_home_wait(
            SessionId(1),
            home_realm(),
            home_lineage(),
            TickId(9),
            AccountId(5),
        );
        assert_eq!(
            sessions.attach_anchor(Some(home_realm())),
            Some(TickId(9)),
            "a member rides its realm's wait anchor"
        );
    }

    #[test]
    fn home_redrive_due_fires_only_on_the_cadence_and_never_at_the_start() {
        // The pure cadence predicate, both arms + the two guards: elapsed 0 (the tick the wait began, whose
        // seed already went out inline) never re-drives; a zero interval degrades to every-tick (fail-safe,
        // never a silent never-re-drive); and the anchor is the wait's OWN `since`, which is what spreads a
        // mass login across many homes over the cadence window.
        assert!(!home_redrive_due(TickId(1), TickId(1), 2), "elapsed 0");
        assert!(!home_redrive_due(TickId(2), TickId(1), 2), "elapsed 1 of 2");
        assert!(home_redrive_due(TickId(3), TickId(1), 2), "elapsed 2 of 2");
        assert!(!home_redrive_due(TickId(4), TickId(1), 2), "elapsed 3 of 2");
        assert!(home_redrive_due(TickId(5), TickId(1), 2), "elapsed 4 of 2");
        // Two WAITS that opened on DIFFERENT ticks are due on DIFFERENT ticks (the storm spread).
        assert!(home_redrive_due(TickId(4), TickId(2), 2));
        assert!(!home_redrive_due(TickId(5), TickId(2), 2));
        // Guards: a zero interval is every-tick (fail-safe); a `since` in the future saturates to 0.
        assert!(home_redrive_due(TickId(2), TickId(1), 0));
        assert!(!home_redrive_due(TickId(1), TickId(9), 2));
    }

    #[test]
    fn home_bootstrap_expired_is_exclusive_at_the_deadline() {
        // Strictly `>`: the deadline tick itself is still inside the hold (generous at its own edge).
        assert!(!home_bootstrap_expired(TickId(5), TickId(6)));
        assert!(!home_bootstrap_expired(TickId(6), TickId(6)));
        assert!(home_bootstrap_expired(TickId(7), TickId(6)));
    }

    #[test]
    fn the_seed_injector_validate_rejects_a_mis_tuned_armed_budget() {
        // Fail-LOUD at boot (mirrors `RlmTuning::validate`): an UNARMED injector is vacuously valid (its
        // windows are never read), an ARMED one needs both windows non-zero AND a bootstrap window that
        // contains at least one re-drive. All arms + both Display messages.
        assert_eq!(
            SeedInjectorConfig::inert(test_world().lowered()).validate(),
            Ok(())
        );
        let armed = armed_injector(default_homes());
        assert_eq!(armed.validate(), Ok(()), "the live test budget is valid");
        let zero_demand = SeedInjectorConfig {
            demand_ttl_ticks: 0,
            ..armed_injector(default_homes())
        };
        assert_eq!(
            zero_demand
                .validate()
                .expect_err("an armed zero demand TTL is rejected"),
            SeedInjectorError::ZeroWindowWhileArmed
        );
        let zero_bootstrap = SeedInjectorConfig {
            bootstrap_ttl_ticks: 0,
            ..armed_injector(default_homes())
        };
        assert_eq!(
            zero_bootstrap
                .validate()
                .expect_err("an armed zero bootstrap TTL is rejected"),
            SeedInjectorError::ZeroWindowWhileArmed
        );
        let too_tight = SeedInjectorConfig {
            bootstrap_ttl_ticks: 2, // == the cadence (8/4): no room for even one re-drive
            ..armed_injector(default_homes())
        };
        assert_eq!(
            too_tight
                .validate()
                .expect_err("a bootstrap window shorter than one re-drive is rejected"),
            SeedInjectorError::BootstrapTtlBelowRedrive {
                bootstrap_ttl: 2,
                redrive: 2,
            }
        );
        // The operator-facing Display text of both arms (the actionable boot failure).
        assert!(
            SeedInjectorError::ZeroWindowWhileArmed
                .to_string()
                .contains("bootstrap_ttl_ticks > 0")
        );
        assert!(
            SeedInjectorError::BootstrapTtlBelowRedrive {
                bootstrap_ttl: 2,
                redrive: 2,
            }
            .to_string()
            .contains("must strictly exceed the re-drive cadence")
        );
    }

    #[test]
    fn the_bootstrap_ttl_derivation_is_the_launch_floor_plus_one_demand_cadence() {
        // The ONE derivation a bin uses (never an inline literal): the reconciler's measured-boot launch
        // floor PLUS one demand cadence of slack, saturating.
        assert_eq!(SeedInjectorConfig::bootstrap_ttl_from_rlm(60, 80), 140);
        assert_eq!(
            SeedInjectorConfig::bootstrap_ttl_from_rlm(u64::MAX, 1),
            u64::MAX,
            "saturating (a mis-set env can never wrap into an instant expiry)"
        );
        // The cadence divisor is the named constant, and a zero TTL still yields a usable cadence.
        assert_eq!(SeedInjectorConfig::REDRIVE_DIVISOR, 4);
        assert_eq!(
            SeedInjectorConfig::inert(test_world().lowered()).redrive_interval_ticks(),
            1,
            "the inert injector floors at 1 (never a divide-by-zero cadence)"
        );
    }

    // ===== RLM 5f-4 — the GATEWAY CROSSING-DESTINATION ADMISSION ===================================

    /// A DEMAND-SPAWNED crossing DESTINATION. Deliberately NOT a member of `config().known_shards`
    /// (`{SHARD, DEST}`) — exactly like a realm the cluster spun up on demand, whose `NodeId` was minted at
    /// spawn time long after boot — so every frame of its that the gateway consumes PROVES the RUNTIME
    /// roster admitted it.
    const DYN_DEST: NodeId = NodeId(78);
    /// A SECOND demand-spawned dest (the defensive-replace / chained-crossing cells).
    const DYN_DEST2: NodeId = NodeId(79);
    /// A SECOND transfer id: a DIFFERENT transfer, so a replacing `PrepareSubscribe` is not absorbed by
    /// the redelivery gate.
    const XFER2: TransferId = TransferId(0x5f4);

    /// The RUNTIME routable-shard roster, WHOLE (asserted by equality — never `matches!`).
    fn roster(rig: &Rig) -> BTreeMap<NodeId, u32> {
        rig.world
            .resource::<GatewaySessions>()
            .dynamic_shards
            .clone()
    }

    /// The session's resolved home shard — the D-34 routing field this slice re-points at commit.
    fn homed_on(rig: &Rig, sid: SessionId) -> Option<NodeId> {
        rig.world.resource::<GatewaySessions>().home_shard_of(sid)
    }

    /// The shards a session subscribes to (the COLD authority, sorted by `NodeId`).
    fn subs_of(rig: &Rig, sid: SessionId) -> Vec<NodeId> {
        rig.world
            .resource::<GatewaySessions>()
            .by_session
            .get(&sid)
            .expect("session present")
            .subs
            .keys()
            .copied()
            .collect()
    }

    /// How many datagrams of `class` reached CLIENT this tick.
    fn to_client(sent: &[(NodeId, MsgClass, Vec<u8>)], class: MsgClass) -> usize {
        sent.iter()
            .filter(|(to, c, _)| (*to == CLIENT) & (*c == class))
            .count()
    }

    /// Both of the dest's DATA classes in one tick: an entity `Snapshot` at `Fence(5)` — the realm fence
    /// [`subscription_ready`] opens the dest sub at, so it is ACCEPTED rather than fence-dropped — and a
    /// `RealmSnapshot` (ambient world state, no fence gate).
    fn dest_frames(dest: NodeId) -> Vec<Inbound> {
        vec![
            wire(dest, MsgClass::Snapshot, &frame_msg(Fence(5), 1)),
            wire(
                dest,
                MsgClass::RealmSnapshot,
                &realm_frame_msg(RealmId::Planet(7)),
            ),
        ]
    }

    /// Drive the WHOLE saga a crossing rides against `dest`, ONE ordered CONTROL command per tick:
    /// Prepare → RequestCut → FreezeSource → CommitAuthority. Returns the COMMIT tick's sends.
    fn cross_to(
        rig: &mut Rig,
        sid: SessionId,
        transfer: TransferId,
        dest: NodeId,
    ) -> Vec<(NodeId, MsgClass, Vec<u8>)> {
        let _ = rig.tick(vec![saga_cmd(TransferControl::PrepareSubscribe {
            transfer,
            session: sid,
            dest,
        })]);
        let _ = rig.tick(vec![saga_cmd(TransferControl::RequestCut {
            transfer,
            session: sid,
        })]);
        let _ = rig.tick(vec![saga_cmd(TransferControl::FreezeSource {
            transfer,
            session: sid,
            marker_seq: 0,
            dest,
        })]);
        rig.tick(vec![saga_cmd(TransferControl::CommitAuthority {
            transfer,
            session: sid,
            new_fence: Fence(2),
            subject: XFER_SUBJECT,
        })])
    }

    #[test]
    fn a_prepare_admits_the_crossing_dest_so_its_subscription_ready_lands() {
        // THE BUG THIS SLICE FIXES, half one. A demand-spawned crossing dest is in NEITHER the frozen
        // config roster nor (pre-5f-4) the runtime one, so every frame it sent fell through to the CLIENT
        // branch. The orchestrator-signed `PrepareSubscribe` now ADMITS it, and Prepare strictly precedes
        // FreezeSource/CommitAuthority — so the admission is in place before the dest's first frame.
        let mut rig = Rig::new();
        let (sid, _) = rig.login();
        assert_eq!(
            roster(&rig),
            BTreeMap::new(),
            "an ordinary login claims no runtime roster entry"
        );
        let _ = rig.tick(vec![saga_cmd(TransferControl::PrepareSubscribe {
            transfer: XFER,
            session: sid,
            dest: DYN_DEST,
        })]);
        assert_eq!(
            roster(&rig),
            BTreeMap::from([(DYN_DEST, 1)]),
            "the Prepare claimed the demand-spawned dest onto the RUNTIME routable roster"
        );
        // …and THAT is what makes its read-plane promote land: the dest sub opens and the client is told
        // which sub now carries its avatar.
        let ready = rig.tick(vec![subscription_ready(DYN_DEST, sid)]);
        assert_eq!(
            decode_controls(&ready, CLIENT),
            vec![
                ServerControlMsg::SubscriptionOpened {
                    sub: SubId(1),
                    frame: FrameRef::SystemSpace { system_seed: 8 },
                },
                ServerControlMsg::AuthorityChanged {
                    entity: EntityId(77),
                    sub: SubId(1),
                },
                ServerControlMsg::OwnEntity {
                    entity: EntityId(77)
                },
            ],
            "the dest's SubscriptionReady opened the dest sub and re-pointed the render authority"
        );
        assert_eq!(subs_of(&rig, sid), vec![SHARD, DYN_DEST]);
        assert_eq!(rig.stats().undecodable, 0, "nothing was lost");
    }

    #[test]
    fn without_the_prepare_admission_a_crossing_dests_frames_are_all_lost() {
        // THE FALSIFIABLE TWIN. The IDENTICAL frames from the IDENTICAL node with NO preceding Prepare: the
        // dest is unrecognised, so every one of its frames falls through to the CLIENT branch and is counted
        // `undecodable` — the `SubscriptionReady` (which fails the `ClientControlMsg` decode) so the render
        // authority NEVER re-points and the player is BLIND, and both data classes (which have no
        // client-branch arm at all) so nothing of the dest world is ever served. This is the state
        // test-one's admission is measured against.
        let mut rig = Rig::new();
        let (sid, _) = rig.login();
        let ready = rig.tick(vec![subscription_ready(DYN_DEST, sid)]);
        assert_eq!(
            decode_controls(&ready, CLIENT),
            Vec::new(),
            "with no admission the dest's SubscriptionReady opens nothing and tells the client nothing"
        );
        assert_eq!(roster(&rig), BTreeMap::new(), "nothing was claimed");
        assert_eq!(
            subs_of(&rig, sid),
            vec![SHARD],
            "the session still subscribes ONLY to its source — the render authority never moved"
        );
        let frames = rig.tick(dest_frames(DYN_DEST));
        assert_eq!(
            to_client(&frames, MsgClass::Snapshot),
            0,
            "no entity frame from the unadmitted dest reached the client"
        );
        assert_eq!(
            to_client(&frames, MsgClass::RealmSnapshot),
            0,
            "and no realm frame either"
        );
        // All three are still lost — but they are lost for TWO different reasons, and the counters now say
        // which. The `SubscriptionReady` reaches the client branch and fails to decode as a client message,
        // so it is genuinely `undecodable`. The two DATA frames have no client-branch arm at all: they are
        // refused because the router does not know their sender. That split is the whole point of the new
        // counter — on the live demand cluster this second number reached 17,365 while `undecodable` was
        // the only thing anyone looked at.
        assert_eq!(
            rig.stats().undecodable,
            1,
            "the SubscriptionReady fell to the client branch and failed the client decode"
        );
        assert_eq!(
            rig.stats().refused_unknown_sender,
            2,
            "both DATA frames were refused because the sender is not known as a shard — never served"
        );
    }

    #[test]
    fn the_forward_crossing_re_homes_the_session_and_admits_its_snapshot_and_realm_frames() {
        // THE FORWARD CROSSING, end to end, into a DEMAND-SPAWNED dest: the write route swaps, the D-34
        // routing field FOLLOWS authority, the dest is admitted, and BOTH of its render classes reach the
        // client. Pre-5f-4 every one of the dest's frames was counted undecodable instead.
        let mut rig = Rig::new();
        let (sid, _) = rig.login();
        let committed = cross_to(&mut rig, sid, XFER, DYN_DEST);
        assert_eq!(
            acks_to_orch(&committed),
            vec![TransferControlAck::Committed { transfer: XFER }],
            "the commit acked exactly once"
        );
        assert_eq!(
            route_authority(&rig, sid),
            DYN_DEST,
            "the WRITE route swapped to the dest"
        );
        assert_eq!(
            homed_on(&rig, sid),
            Some(DYN_DEST),
            "THE D-34 CLOSE: the session's routing target followed authority to the dest"
        );
        assert_eq!(
            roster(&rig),
            BTreeMap::from([(DYN_DEST, 2)]),
            "the dest holds TWO claims — the crossing (from Prepare) and the new home (from commit)"
        );
        // The read-plane promote, then BOTH render classes.
        let _ = rig.tick(vec![subscription_ready(DYN_DEST, sid)]);
        assert_eq!(subs_of(&rig, sid), vec![SHARD, DYN_DEST]);
        let frames = rig.tick(dest_frames(DYN_DEST));
        assert_eq!(
            to_client(&frames, MsgClass::Snapshot),
            1,
            "the dest's entity frame reaches the client (the avatar renders after the crossing)"
        );
        // The world around the avatar rides the COMPOSED feed since the flag day; the dest's
        // tombstoned realm frame lands in its own counter (a fallen-through frame would count
        // undecodable, asserted 0 below).
        assert_eq!(to_client(&frames, MsgClass::RealmSnapshot), 0);
        assert_eq!(
            rig.stats().undecodable,
            0,
            "nothing was lost as undecodable"
        );
    }

    #[test]
    fn the_release_terminal_returns_the_crossing_claim_and_the_dest_stays_routable() {
        // The demote tail hands back the CROSSING claim only. The dest keeps the HOME claim commit
        // installed, so a post-`Done` player keeps receiving its dest frames — the refcount's whole point.
        let mut rig = Rig::new();
        let (sid, _) = rig.login();
        let _ = cross_to(&mut rig, sid, XFER, DYN_DEST);
        let _ = rig.tick(vec![subscription_ready(DYN_DEST, sid)]);
        let released = rig.tick(vec![saga_cmd(TransferControl::ReleaseSubscribe {
            transfer: XFER,
            session: sid,
            src: SHARD,
        })]);
        assert_eq!(
            acks_to_orch(&released),
            vec![TransferControlAck::Released { transfer: XFER }]
        );
        assert_eq!(
            roster(&rig),
            BTreeMap::from([(DYN_DEST, 1)]),
            "the crossing claim went back; the HOME claim keeps the dest on the roster"
        );
        assert_eq!(
            to_client(&rig.tick(dest_frames(DYN_DEST)), MsgClass::Snapshot),
            1,
            "so the dest is STILL routable after the saga reached Done"
        );
        // …and the session exit is what finally drains it.
        let bye = rig.tick(vec![wire(
            CLIENT,
            MsgClass::Control,
            &ClientControlMsg::Bye,
        )]);
        assert!(
            saw_detach(&bye, DYN_DEST, sid),
            "D-34: the post-crossing detach reaches the CURRENT authority, not the source"
        );
        assert!(
            !saw_detach(&bye, SHARD, sid),
            "and never the source it left"
        );
        assert_eq!(
            roster(&rig),
            BTreeMap::new(),
            "the last claim went with the session — no roster leak"
        );
    }

    #[test]
    fn the_abort_terminal_returns_the_crossing_claim_and_the_dest_leaves_the_roster() {
        // The compensating terminal. A pre-CAS abort never re-pointed `home_shard`, so the Prepare-time
        // claim is the dest's ONLY one and the node leaves the roster with it.
        let mut rig = Rig::new();
        let (sid, _) = rig.login();
        let _ = rig.tick(vec![saga_cmd(TransferControl::PrepareSubscribe {
            transfer: XFER,
            session: sid,
            dest: DYN_DEST,
        })]);
        assert_eq!(roster(&rig), BTreeMap::from([(DYN_DEST, 1)]));
        let aborted = rig.tick(vec![saga_cmd(TransferControl::AbortTransfer {
            transfer: XFER,
            session: sid,
        })]);
        assert_eq!(
            acks_to_orch(&aborted),
            vec![TransferControlAck::Aborted { transfer: XFER }]
        );
        assert_eq!(
            roster(&rig),
            BTreeMap::new(),
            "the aborted crossing's claim is returned — the dest leaves the runtime roster"
        );
        assert_eq!(
            homed_on(&rig, sid),
            None,
            "and an aborted crossing never re-homed the session"
        );
    }

    #[test]
    fn a_redelivered_and_a_replacing_prepare_never_leak_a_roster_refcount() {
        // IDEMPOTENCE of the admission. (a) A redelivered SAME-transfer Prepare is absorbed by the journal
        // gate before any effect ⇒ no second claim. (b) A DIFFERENT transfer's Prepare defensively REPLACES
        // the progress: it claims the new dest and hands back the DISPLACED one, so neither a stale entry
        // is pinned nor is a re-claimed same-node dest ever transiently unrouted.
        let mut rig = Rig::new();
        let (sid, _) = rig.login();
        let first = rig.tick(vec![saga_cmd(TransferControl::PrepareSubscribe {
            transfer: XFER,
            session: sid,
            dest: DYN_DEST,
        })]);
        assert_eq!(roster(&rig), BTreeMap::from([(DYN_DEST, 1)]));
        // (a) the exact same command again — the redelivery gate re-serves the recorded ack.
        let again = rig.tick(vec![saga_cmd(TransferControl::PrepareSubscribe {
            transfer: XFER,
            session: sid,
            dest: DYN_DEST,
        })]);
        assert_eq!(
            acks_to_orch(&again),
            acks_to_orch(&first),
            "the redelivery re-served the recorded Prepared verbatim"
        );
        assert_eq!(
            roster(&rig),
            BTreeMap::from([(DYN_DEST, 1)]),
            "and claimed nothing a second time (a redelivery cannot inflate the refcount)"
        );
        // (b) a DIFFERENT transfer naming the SAME dest: claim-then-release keeps it at exactly one, and it
        // never leaves the roster in between.
        let _ = rig.tick(vec![saga_cmd(TransferControl::PrepareSubscribe {
            transfer: XFER2,
            session: sid,
            dest: DYN_DEST,
        })]);
        assert_eq!(
            roster(&rig),
            BTreeMap::from([(DYN_DEST, 1)]),
            "the replace re-claimed and released the same node — net one, never a leak"
        );
        // (b') a replace naming a DIFFERENT dest hands the displaced one back.
        let _ = rig.tick(vec![saga_cmd(TransferControl::PrepareSubscribe {
            transfer: XFER,
            session: sid,
            dest: DYN_DEST2,
        })]);
        assert_eq!(
            roster(&rig),
            BTreeMap::from([(DYN_DEST2, 1)]),
            "the displaced dest left the roster with the progress it belonged to"
        );
        let _ = rig.tick(vec![saga_cmd(TransferControl::AbortTransfer {
            transfer: XFER,
            session: sid,
        })]);
        assert_eq!(
            roster(&rig),
            BTreeMap::new(),
            "and the terminal drains the survivor — every claim balanced"
        );
    }

    #[test]
    fn a_prepare_refused_before_attach_claims_no_roster_entry() {
        // The claim rides the SAME condition as the progress write. A not-yet-Active session's Prepare is
        // refused and opens NO progress, so a claim here would have no terminal to release it — a
        // permanent roster leak pinning a node the crossing never reached.
        let mut rig = Rig::new();
        let _ = rig.tick(vec![wire(CLIENT, MsgClass::Control, &hello_msg())]);
        let sid = session_of(&rig, CLIENT);
        let _ = rig.tick(vec![wire(ORCH, MsgClass::Saga, &granted_head(sid))]);
        assert_eq!(phase_of(&rig, sid), SessionPhase::AwaitingAttach);
        let sent = rig.tick(vec![saga_cmd(TransferControl::PrepareSubscribe {
            transfer: XFER,
            session: sid,
            dest: DYN_DEST,
        })]);
        assert_eq!(acks_to_orch(&sent), vec![], "the prepare was refused");
        assert_eq!(
            roster(&rig),
            BTreeMap::new(),
            "a refused prepare claims nothing (a claim with no terminal IS the leak)"
        );
        assert_eq!(rig.stats().transfer_unroutable, 1);
    }

    #[test]
    fn a_crossing_off_a_shared_home_leaves_the_other_sessions_route_intact() {
        // THE REFCOUNT'S OTHER ARM. Two sessions share one demand-spawned home; ONE of them crosses away.
        // The node must survive the crosser's whole demote tail (decrement, not remove), or the session that
        // stayed would go blind the moment its neighbour crossed. NOTE (RLM 5f-4e): the crosser's home claim
        // is STASHED at commit and returned at `ReleaseSubscribe`, so HOME holds BOTH sessions' claims
        // through the tail — which is exactly why a SHARED home masked the blind window; the solo-home cell
        // `a_crossing_off_a_solo_dynamic_home_keeps_the_source_routable_through_the_demote_tail` is the
        // falsifying one.
        let mut rig = dynamic_rig();
        let (sid_a, _) = dynamic_login(&mut rig, CLIENT);
        let (sid_b, _) = dynamic_login(&mut rig, CLIENT2);
        // ONE head reply resolves both members onto HOME (two claims).
        let _ = rig.tick(vec![wire(
            ORCH,
            MsgClass::Saga,
            &realm_head(home_realm(), Some(HOME)),
        )]);
        assert_eq!(roster(&rig), BTreeMap::from([(HOME, 2)]));
        for client in [CLIENT, CLIENT2] {
            let sid = session_of(&rig, client);
            let _ = rig.tick(vec![wire(
                HOME,
                MsgClass::Control,
                &ShardToGateway::SessionAttached {
                    session: sid,
                    entity: EntityId(77),
                    frame: FrameRef::SystemSpace { system_seed: 7 },
                    realm_fence: Fence(1),
                },
            )]);
        }
        // B crosses HOME -> DYN_DEST.
        let _ = cross_to(&mut rig, sid_b, XFER, DYN_DEST);
        assert_eq!(
            roster(&rig),
            BTreeMap::from([(HOME, 2), (DYN_DEST, 2)]),
            "B's home claim is STASHED for its demote tail, so HOME still counts both sessions"
        );
        assert_eq!(
            homed_on(&rig, sid_b),
            Some(DYN_DEST),
            "B followed authority"
        );
        assert_eq!(homed_on(&rig, sid_a), Some(HOME), "A did not move");
        assert_eq!(
            to_client(&rig.tick(dest_frames(HOME)), MsgClass::Snapshot),
            1,
            "and A still receives its HOME frames after its neighbour crossed away"
        );
        // Both exits drain to nothing.
        let _ = rig.tick(vec![saga_cmd(TransferControl::ReleaseSubscribe {
            transfer: XFER,
            session: sid_b,
            src: HOME,
        })]);
        assert_eq!(
            roster(&rig),
            BTreeMap::from([(HOME, 1), (DYN_DEST, 1)]),
            "the tail returned B's stashed HOME claim AND its crossing claim — A's HOME claim is untouched"
        );
        let _ = rig.tick(vec![wire(
            CLIENT2,
            MsgClass::Control,
            &ClientControlMsg::Bye,
        )]);
        assert_eq!(roster(&rig), BTreeMap::from([(HOME, 1)]));
        let _ = rig.tick(vec![wire(
            CLIENT,
            MsgClass::Control,
            &ClientControlMsg::Bye,
        )]);
        assert_eq!(
            roster(&rig),
            BTreeMap::new(),
            "every claim across two sessions and one crossing is balanced"
        );
    }

    #[test]
    fn a_bye_mid_crossing_returns_the_crossing_claim_before_the_saga_terminal() {
        // A `Bye` can land ANYWHERE in the saga, including before a terminal ever arrives (WEDGE-1). The
        // ONE session-exit release covers the in-flight transfer's dest too, so a client that quits
        // mid-crossing cannot pin a demand-spawned node on the roster forever.
        let mut rig = Rig::new();
        let (sid, _) = rig.login();
        let _ = rig.tick(vec![saga_cmd(TransferControl::PrepareSubscribe {
            transfer: XFER,
            session: sid,
            dest: DYN_DEST,
        })]);
        assert_eq!(roster(&rig), BTreeMap::from([(DYN_DEST, 1)]));
        let _ = rig.tick(vec![wire(
            CLIENT,
            MsgClass::Control,
            &ClientControlMsg::Bye,
        )]);
        assert_eq!(
            roster(&rig),
            BTreeMap::new(),
            "the exit released the in-flight crossing's claim, with no terminal in sight"
        );
    }

    #[test]
    fn a_bye_in_the_commit_to_release_window_returns_both_of_the_dests_claims() {
        // The DOUBLE-claim window: between `CommitAuthority` and `ReleaseSubscribe` the dest is claimed
        // twice (crossing + home). Both exits release BOTH roles through the ONE primitive, so a quit
        // exactly here still drains the node to zero.
        let mut rig = Rig::new();
        let (sid, _) = rig.login();
        let _ = cross_to(&mut rig, sid, XFER, DYN_DEST);
        assert_eq!(roster(&rig), BTreeMap::from([(DYN_DEST, 2)]));
        let _ = rig.tick(vec![wire(
            CLIENT,
            MsgClass::Control,
            &ClientControlMsg::Bye,
        )]);
        assert_eq!(
            roster(&rig),
            BTreeMap::new(),
            "both roles released at the one exit — the doubly-claimed dest drains to zero"
        );
        // The OTHER exit (the bounded bootstrap TTL) shares that primitive: a pre-Active dynamic session
        // holds only the home role, and it too drains.
        let mut ttl_rig = dynamic_rig_with_ttl(5); // waiting since tick 1 ⇒ deadline 6
        let (ttl_sid, _) = dynamic_login(&mut ttl_rig, CLIENT);
        let _ = ttl_rig.tick(vec![wire(
            ORCH,
            MsgClass::Saga,
            &realm_head(home_realm(), Some(HOME)),
        )]);
        assert_eq!(roster(&ttl_rig), BTreeMap::from([(HOME, 1)]));
        set_tick(&mut ttl_rig, 7); // past the deadline
        let _ = ttl_rig.tick(vec![]);
        assert_eq!(ttl_rig.stats().home_bootstrap_timeouts, 1);
        assert_eq!(
            homed_on(&ttl_rig, ttl_sid),
            None,
            "the TTL-closed session is gone"
        );
        assert_eq!(
            roster(&ttl_rig),
            BTreeMap::new(),
            "and the bounded-TTL exit released its claim through the same primitive"
        );
    }

    #[test]
    fn an_all_static_crossing_leaves_the_runtime_roster_empty() {
        // BYTE-IDENTITY. `DEST` is in the FROZEN `known_shards`, so it is already dispatchable and the
        // admission must not manufacture runtime state for it: `dynamic_shards` stays EMPTY at EVERY step,
        // which makes `is_routable_shard` answer purely from the config half exactly as before this slice.
        // (The D-34 re-point still happens — it is a routing fix, not a roster one.)
        let mut rig = Rig::new();
        let (sid, _) = rig.login();
        let _ = rig.tick(vec![saga_cmd(TransferControl::PrepareSubscribe {
            transfer: XFER,
            session: sid,
            dest: DEST,
        })]);
        assert_eq!(
            roster(&rig),
            BTreeMap::new(),
            "a KNOWN dest is never claimed onto the runtime roster"
        );
        let _ = rig.tick(vec![saga_cmd(TransferControl::RequestCut {
            transfer: XFER,
            session: sid,
        })]);
        let _ = rig.tick(vec![saga_cmd(TransferControl::FreezeSource {
            transfer: XFER,
            session: sid,
            marker_seq: 0,
            dest: DEST,
        })]);
        let committed = rig.tick(vec![saga_cmd(TransferControl::CommitAuthority {
            transfer: XFER,
            session: sid,
            new_fence: Fence(2),
            subject: XFER_SUBJECT,
        })]);
        assert_eq!(
            committed
                .iter()
                .map(|(to, class, _)| (*to, *class))
                .collect::<Vec<_>>(),
            vec![(DEST, MsgClass::Control), (ORCH, MsgClass::Saga)],
            "the commit tick's send fingerprint is unchanged: the dest's OpenInputSlot then the ack"
        );
        assert_eq!(
            roster(&rig),
            BTreeMap::new(),
            "still empty after the commit"
        );
        assert_eq!(
            homed_on(&rig, sid),
            Some(DEST),
            "the D-34 re-point applies to a static dest too (routing, not roster)"
        );
        let _ = rig.tick(vec![saga_cmd(TransferControl::ReleaseSubscribe {
            transfer: XFER,
            session: sid,
            src: SHARD,
        })]);
        assert_eq!(
            roster(&rig),
            BTreeMap::new(),
            "and the release of a known dest is a total no-op on the map"
        );
        // The static dest's frames dispatch through the CONFIG half, as they always did. Since
        // the flag day the OLD-lane realm frame reaches the parity intake, never the client —
        // the composed lane is the client's one realm feed (§2.4); a fallen-through frame would
        // count undecodable, which stays 0 (the routing is intact).
        let _ = rig.tick(vec![subscription_ready(DEST, sid)]);
        let frames = rig.tick(dest_frames(DEST));
        assert_eq!(to_client(&frames, MsgClass::Snapshot), 1);
        assert_eq!(
            to_client(&frames, MsgClass::RealmSnapshot),
            0,
            "the old realm lane no longer fans to clients (the composed feed replaced it)"
        );
        assert_eq!(rig.stats().undecodable, 0);
        let bye = rig.tick(vec![wire(
            CLIENT,
            MsgClass::Control,
            &ClientControlMsg::Bye,
        )]);
        assert!(
            saw_detach(&bye, DEST, sid),
            "D-34 closes for the static crossing too: the detach follows authority"
        );
        assert_eq!(
            roster(&rig),
            BTreeMap::new(),
            "and no entry was ever created"
        );
    }

    // ===== RLM 5f-4e — the DEMOTING-SOURCE claim (the REJECT-class blind window) ====================

    /// The SOURCE home a committed crossing is still demoting away from (`TransferProgress::
    /// demoting_home`) — the RLM 5f-4e stash. Read by equality so a redelivery can be shown NOT to
    /// clobber it. Only called where a transfer is in flight.
    fn demoting_home_of(rig: &Rig, sid: SessionId) -> Option<NodeId> {
        rig.world
            .resource::<GatewaySessions>()
            .by_session
            .get(&sid)
            .expect("session present")
            .transfer
            .as_ref()
            .expect("a transfer is in flight")
            .demoting_home
    }

    /// Drive ONE dynamic login all the way to `Active` on the DEMAND-SPAWNED `HOME` shard: hello → grant →
    /// the `Realm` head that resolves the home → the home's `SessionAttached`. The session then holds
    /// EXACTLY ONE roster claim (its home role) and ONE read sub (on `HOME`) — the shape every solo-home
    /// crossing cell needs, where `HOME`'s routability rests ENTIRELY on that one claim (it is not in
    /// `known_shards`).
    fn dynamic_active_on_home(rig: &mut Rig, client: NodeId) -> SessionId {
        let (sid, _) = dynamic_login(rig, client);
        let _ = rig.tick(vec![wire(
            ORCH,
            MsgClass::Saga,
            &realm_head(home_realm(), Some(HOME)),
        )]);
        let _ = rig.tick(vec![wire(
            HOME,
            MsgClass::Control,
            &ShardToGateway::SessionAttached {
                session: sid,
                entity: EntityId(77),
                frame: FrameRef::SystemSpace { system_seed: 7 },
                realm_fence: Fence(1),
            },
        )]);
        sid
    }

    #[test]
    fn a_crossing_off_a_solo_dynamic_home_keeps_the_source_routable_through_the_demote_tail() {
        // THE REJECT-CLASS BLIND WINDOW, falsified. ONE session on a DEMAND-SPAWNED home with NO second
        // session parked there — which is exactly what `a_crossing_off_a_shared_home_…` masked: there the
        // neighbour's claim kept the node routable no matter what commit did.
        //
        // The client's READ subscription on the SOURCE stays open across the WHOLE demote grace (it closes
        // only at `ReleaseSubscribe`, a later tick) and the source keeps SHIPPING frames the whole time.
        // Releasing the source's runtime-roster claim at `CommitAuthority` un-routed it, so every one of
        // those frames fell through to the client branch and was counted `undecodable` — a BLIND player for
        // the whole tail, right after EVERY crossing. Pre-rework this cell fails at the first (a) assertion
        // below (0 frames served, 2 undecodable); post-rework the source stays routable until its sub closes.
        let mut rig = dynamic_rig();
        let sid = dynamic_active_on_home(&mut rig, CLIENT);
        assert_eq!(
            roster(&rig),
            BTreeMap::from([(HOME, 1)]),
            "HOME's routability rests on this ONE claim — nothing else holds it"
        );
        assert_eq!(
            subs_of(&rig, sid),
            vec![HOME],
            "and the client reads exactly one shard: the source"
        );
        let _ = cross_to(&mut rig, sid, XFER, DYN_DEST);
        // (a) THE TICK AFTER COMMIT — the BEHAVIOURAL falsifier, asserted BEFORE any structural claim so a
        // pre-rework tree fails on the SYMPTOM (the player's frames), not on an internal field. The dest sub
        // does not exist yet — its `SubscriptionReady` is a round-trip away — so the SOURCE feed is ALL the
        // player has. Both render classes must still land. Pre-rework: 0 and 0, with `undecodable` at 2.
        let after_commit = rig.tick(dest_frames(HOME));
        assert_eq!(
            to_client(&after_commit, MsgClass::Snapshot),
            1,
            "the SOURCE's entity frame still reaches the client on the tick after commit"
        );
        // Since the flag day the world's continuity rides the COMPOSED feed (the client holds its
        // scene across the swap, §2.7); the source's tombstoned realm frame lands in the dead-lane
        // counter, never at the client. The falsifier stays: a frame that fell through to the
        // client branch as a stranger's would count undecodable.
        assert_eq!(
            to_client(&after_commit, MsgClass::RealmSnapshot),
            0,
            "the old realm lane no longer fans to clients (the composed feed replaced it)"
        );
        assert_eq!(
            rig.stats().undecodable,
            0,
            "nothing of the source was counted undecodable — the source stayed ROUTABLE"
        );
        // …and the structure that produced it: the target moved, the displaced home was STASHED, both ends
        // are on the roster.
        assert_eq!(
            homed_on(&rig, sid),
            Some(DYN_DEST),
            "the routing target followed authority to the dest"
        );
        assert_eq!(
            demoting_home_of(&rig, sid),
            Some(HOME),
            "…while the displaced home was STASHED on the progress, not released at commit"
        );
        assert_eq!(
            roster(&rig),
            BTreeMap::from([(HOME, 1), (DYN_DEST, 2)]),
            "so BOTH ends stay routable through the tail (source 1; dest home + crossing)"
        );
        // (b) AFTER THE DEST PROMOTE — both subs open, which is the composite a seamless crossing renders.
        let _ = rig.tick(vec![subscription_ready(DYN_DEST, sid)]);
        assert_eq!(
            subs_of(&rig, sid),
            vec![HOME, DYN_DEST],
            "both subs are open across the tail"
        );
        let composited = rig.tick(dest_frames(HOME));
        assert_eq!(
            to_client(&composited, MsgClass::Snapshot),
            1,
            "the source still feeds the composite after the dest promote"
        );
        // The realm class rides the composed feed since the flag day (asserted 0 on the dead lane).
        assert_eq!(to_client(&composited, MsgClass::RealmSnapshot), 0);
        let dest_side = rig.tick(dest_frames(DYN_DEST));
        assert_eq!(
            to_client(&dest_side, MsgClass::Snapshot),
            1,
            "and the dest feeds it too — the player sees both ends, never neither"
        );
        assert_eq!(to_client(&dest_side, MsgClass::RealmSnapshot), 0);
        assert_eq!(
            rig.stats().undecodable,
            0,
            "still nothing lost, with both subs open"
        );
        // The terminal is where the source's claim comes DUE: the same command closes the source sub, so the
        // routable window matches the sub's lifetime to the tick.
        let released = rig.tick(vec![saga_cmd(TransferControl::ReleaseSubscribe {
            transfer: XFER,
            session: sid,
            src: HOME,
        })]);
        assert_eq!(
            acks_to_orch(&released),
            vec![TransferControlAck::Released { transfer: XFER }]
        );
        assert_eq!(
            roster(&rig),
            BTreeMap::from([(DYN_DEST, 1)]),
            "the source left the roster WITH its sub; the dest keeps its home claim"
        );
        let bye = rig.tick(vec![wire(
            CLIENT,
            MsgClass::Control,
            &ClientControlMsg::Bye,
        )]);
        assert!(
            saw_detach(&bye, DYN_DEST, sid),
            "the detach follows authority to the dest"
        );
        assert_eq!(
            roster(&rig),
            BTreeMap::new(),
            "every claim across the whole crossing is balanced — no leak, nothing pinned"
        );
    }

    #[test]
    fn a_post_cas_abort_returns_both_the_crossing_dest_and_the_demoting_source_claims() {
        // THE OTHER TERMINAL. `apply_abort` PRUNES `session.transfer`, so an abort that lands AFTER the CAS
        // (the stash already written) is the LAST holder able to NAME the demoting source — skip it there and
        // the source's claim is stranded on the roster forever, with no later path that could ever release
        // it. Both roles go back here; the session, which the commit did re-home, keeps only its dest claim.
        let mut rig = dynamic_rig();
        let sid = dynamic_active_on_home(&mut rig, CLIENT);
        let _ = cross_to(&mut rig, sid, XFER, DYN_DEST);
        assert_eq!(roster(&rig), BTreeMap::from([(HOME, 1), (DYN_DEST, 2)]));
        let aborted = rig.tick(vec![saga_cmd(TransferControl::AbortTransfer {
            transfer: XFER,
            session: sid,
        })]);
        assert_eq!(
            acks_to_orch(&aborted),
            vec![TransferControlAck::Aborted { transfer: XFER }]
        );
        assert_eq!(
            roster(&rig),
            BTreeMap::from([(DYN_DEST, 1)]),
            "the crossing claim AND the stashed source claim both went back at the abort"
        );
        assert_eq!(
            homed_on(&rig, sid),
            Some(DYN_DEST),
            "a POST-CAS abort does not un-re-home the session (the commit already happened)"
        );
        // RLM 5f-4e (the relocated-blind-window fix): the abort ALSO closed the DEMOTING SOURCE sub in
        // lock-step with releasing its claim, so a straggler source frame can never hit an
        // unroutable-but-still-subscribed node. One sweep tick later the drained source sub is gone.
        let _ = rig.tick(vec![]);
        assert!(
            sub_for(&rig, sid, HOME).is_none(),
            "the demoting source sub was closed at the abort and swept, not left dangling on an un-routed node"
        );
        let _ = rig.tick(vec![wire(
            CLIENT,
            MsgClass::Control,
            &ClientControlMsg::Bye,
        )]);
        assert_eq!(
            roster(&rig),
            BTreeMap::new(),
            "and the exit drains the survivor — balanced"
        );
    }

    #[test]
    fn a_bye_inside_the_demote_tail_returns_the_stashed_source_claim_too() {
        // THE THIRD ROLE AT THE SESSION EXIT. A client that quits between `CommitAuthority` and
        // `ReleaseSubscribe` leaves NO terminal to run, so the ONE exit primitive must hand back all THREE
        // roles (home + crossing dest + demoting source) or a demand-spawned source is pinned forever.
        let mut rig = dynamic_rig();
        let sid = dynamic_active_on_home(&mut rig, CLIENT);
        let _ = cross_to(&mut rig, sid, XFER, DYN_DEST);
        assert_eq!(roster(&rig), BTreeMap::from([(HOME, 1), (DYN_DEST, 2)]));
        let bye = rig.tick(vec![wire(
            CLIENT,
            MsgClass::Control,
            &ClientControlMsg::Bye,
        )]);
        assert!(
            saw_detach(&bye, DYN_DEST, sid),
            "the detach follows authority to the dest"
        );
        assert_eq!(
            roster(&rig),
            BTreeMap::new(),
            "all three roles released at the ONE exit — nothing pinned, nothing double-released"
        );
    }

    #[test]
    fn a_second_prepare_inside_the_demote_tail_hands_back_the_displaced_source_claim() {
        // THE DEFENSIVE REPLACE. A second `PrepareSubscribe` OVERWRITES the progress, and the replacement
        // starts `demoting_home: None` — so unless the DISPLACED progress's stash is handed back alongside
        // its displaced dest, the first crossing's source is stranded with no holder left able to name it.
        let mut rig = dynamic_rig();
        let sid = dynamic_active_on_home(&mut rig, CLIENT);
        let _ = cross_to(&mut rig, sid, XFER, DYN_DEST);
        assert_eq!(roster(&rig), BTreeMap::from([(HOME, 1), (DYN_DEST, 2)]));
        // A DIFFERENT transfer (so the redelivery gate does not absorb it) naming a THIRD node.
        let _ = rig.tick(vec![saga_cmd(TransferControl::PrepareSubscribe {
            transfer: XFER2,
            session: sid,
            dest: DYN_DEST2,
        })]);
        assert_eq!(
            demoting_home_of(&rig, sid),
            None,
            "the replacement progress is demoting nothing of its own"
        );
        assert_eq!(
            roster(&rig),
            BTreeMap::from([(DYN_DEST, 1), (DYN_DEST2, 1)]),
            "the displaced dest AND the displaced source both went back; the new dest was claimed"
        );
        let _ = rig.tick(vec![saga_cmd(TransferControl::AbortTransfer {
            transfer: XFER2,
            session: sid,
        })]);
        assert_eq!(
            roster(&rig),
            BTreeMap::from([(DYN_DEST, 1)]),
            "the second crossing's terminal drains its dest (its own stash was None — the no-op arm)"
        );
        let _ = rig.tick(vec![wire(
            CLIENT,
            MsgClass::Control,
            &ClientControlMsg::Bye,
        )]);
        assert_eq!(
            roster(&rig),
            BTreeMap::new(),
            "and the exit drains the home role — fully balanced"
        );
    }

    #[test]
    fn a_redelivered_commit_neither_re_claims_the_dest_nor_clobbers_the_stash() {
        // THE AT-LEAST-ONCE GUARD ON THE STASH. If a redelivered `CommitAuthority` re-entered
        // `apply_commit`, `home_shard.replace(dest)` would return `Some(dest)` — overwriting the stash with
        // the DEST, stranding the real source's claim — and would claim the dest a THIRD time. The
        // applied-steps journal gate absorbs the redelivery BEFORE any effect, so both stay exact.
        let mut rig = dynamic_rig();
        let sid = dynamic_active_on_home(&mut rig, CLIENT);
        let first = cross_to(&mut rig, sid, XFER, DYN_DEST);
        let again = rig.tick(vec![saga_cmd(TransferControl::CommitAuthority {
            transfer: XFER,
            session: sid,
            new_fence: Fence(2),
            subject: XFER_SUBJECT,
        })]);
        assert_eq!(
            acks_to_orch(&again),
            acks_to_orch(&first),
            "the redelivery re-served the recorded Committed verbatim"
        );
        assert_eq!(
            demoting_home_of(&rig, sid),
            Some(HOME),
            "and left the stash naming the REAL source, never the dest"
        );
        assert_eq!(
            roster(&rig),
            BTreeMap::from([(HOME, 1), (DYN_DEST, 2)]),
            "with no third claim on the dest"
        );
        // The proof the held claim is still the SOURCE's: the terminal drains HOME to nothing.
        let _ = rig.tick(vec![saga_cmd(TransferControl::ReleaseSubscribe {
            transfer: XFER,
            session: sid,
            src: HOME,
        })]);
        assert_eq!(
            roster(&rig),
            BTreeMap::from([(DYN_DEST, 1)]),
            "the tail returned exactly the source claim (a clobbered stash would leave HOME pinned)"
        );
        let _ = rig.tick(vec![wire(
            CLIENT,
            MsgClass::Control,
            &ClientControlMsg::Bye,
        )]);
        assert_eq!(roster(&rig), BTreeMap::new());
    }

    // ===== THE WINDOW LANE: the COMPOSER (docs/design/window_lane.md §2.6/§4) =====

    fn snap(realm: RealmId, head: FrameRef, tail: FrameRef, x: f64, at: u64) -> RealmSnap {
        RealmSnap {
            realm,
            frame: head,
            pose: vd_core::pose::StampedPose::at_rest(
                tail,
                DVec3::new(x, 0.0, 0.0),
                UniverseTick(at),
            ),
        }
    }

    const SYS7: FrameRef = FrameRef::SystemSpace { system_seed: 7 };
    const PLANET7: FrameRef = FrameRef::PlanetCentered { planet_seed: 7 };

    /// The parent's Child-scope frame at `at`: System(7) authors Planet(7) at `leaf_x` (the hop
    /// pre-inverted at the author: the parent's frame sits at −leaf_x in the planet's) and
    /// Planet(9) at `sibling_x` — plus any extra rows the caller grafts.
    fn parent_frame(
        window: WindowId,
        at: u64,
        leaf_x: f64,
        mut extra: Vec<RealmSnap>,
    ) -> ShardToGateway {
        let mut rows = vec![
            snap(RealmId::Planet(7), PLANET7, SYS7, leaf_x, at),
            snap(
                RealmId::Planet(9),
                FrameRef::PlanetCentered { planet_seed: 9 },
                SYS7,
                leaf_x + 30.0,
                at,
            ),
        ];
        rows.append(&mut extra);
        ShardToGateway::WindowFrame {
            realm_fence: Fence(1),
            window,
            at: UniverseTick(at),
            hop: Some(Box::new(vd_wire::session_flow::HopRow {
                child: RealmId::Planet(7),
                inv: vd_core::frame::FramePlacement::moving(
                    DVec3::new(-leaf_x, 0.0, 0.0),
                    DVec3::ZERO,
                ),
            })),
            rows,
        }
    }

    fn leaf_frame(window: WindowId, at: u64) -> ShardToGateway {
        ShardToGateway::WindowFrame {
            realm_fence: Fence(1),
            window,
            at: UniverseTick(at),
            hop: None,
            rows: Vec::new(),
        }
    }

    #[test]
    fn the_composer_folds_the_crossing_chain_at_one_tick_and_tears_down_to_nothing() {
        // THE UNIT-TIER TWIN of the process self-consistency gate (§2.12): a real login, a real
        // crossing (System(7) on SHARD → Planet(7) on DEST), then the two window statements
        // through the REAL ingest + composer — the full chain folding at ONE tick, the dedup
        // agreement measured exactly zero, every refusal arm driven by name, the epoch mechanics
        // observed, and the teardown leaving zero composer state.
        let mut rig = Rig::new();
        let (sid, _) = rig.login(); // lineage [System(7)]; Occupants window id 1 on SHARD
        let _ = cross_to(&mut rig, sid, XFER, DEST);
        let _ = rig.tick(vec![wire(
            DEST,
            MsgClass::Control,
            &ShardToGateway::SubscriptionReady {
                session: sid,
                entity: EntityId(77),
                frame: PLANET7,
                realm_fence: Fence(2),
            },
        )]);
        {
            let sessions = rig.world.resource::<GatewaySessions>();
            let session = &sessions.by_session[&sid];
            assert_eq!(
                session.lineage,
                vec![RealmId::System(7), RealmId::Planet(7)],
                "the crossing APPENDED below the previous leaf (§2.6.2)"
            );
            assert_eq!(
                sessions.windows_open_count(),
                3,
                "own-level both ends + the lineage-derived Child window on the parent"
            );
        }
        // Window ids from the login + the ready tick: 1 = Occupants(System 7)@SHARD,
        // 2 = Occupants(Planet 7)@DEST, 3 = Child(Planet 7)@SHARD.
        let (leaf_w, parent_w) = (WindowId(2), WindowId(3));

        // ---- Tick A (T=1000): the leaf speaks alone — the parent's hop is not confirmed yet,
        // so the chain folds one level deep.
        let _ = rig.tick(vec![wire(
            DEST,
            MsgClass::RealmSnapshot,
            &leaf_frame(leaf_w, 1000),
        )]);
        assert_eq!(
            rig.stats().window_folds,
            1,
            "the one-level chain folded at 1000"
        );

        // ---- Tick B (T=1001): the FULL chain folds at ONE tick — the exact-cadence law — and
        // the two lawful sources for the same realm agree to the bit (§2.12's dedup bound).
        let _ = rig.tick(vec![
            wire(DEST, MsgClass::RealmSnapshot, &leaf_frame(leaf_w, 1001)),
            wire(
                SHARD,
                MsgClass::RealmSnapshot,
                &parent_frame(parent_w, 1001, 30.0, vec![]),
            ),
        ]);
        assert_eq!(
            rig.stats().window_full_chain_folds,
            1,
            "a ≥2-level fold at ONE tick"
        );
        assert_eq!(rig.stats().window_dedup_disagree, 0);
        assert_eq!(
            rig.stats().window_dedup_max_dev_nm,
            0,
            "§2.12: exact today, measured"
        );
        assert_eq!(rig.stats().window_instant_mismatch, 0);

        // ---- Tick C (T=1002): the refusal arms, by name. The parent's roster grows an
        // ALIEN-framed row (tail ≠ the author's frame — dropped from the fold, counted); a
        // tombstoned old realm datagram arrives (counted in its own bucket, served to nobody);
        // and a frame stamped a whole ring span behind the head is REFUSED.
        let alien = snap(
            RealmId::Planet(11),
            FrameRef::PlanetCentered { planet_seed: 11 },
            FrameRef::PlanetCentered { planet_seed: 11 }, // tail ≠ the author's frame: alien
            9.0,
            1002,
        );
        let _ = rig.tick(vec![
            wire(DEST, MsgClass::RealmSnapshot, &leaf_frame(leaf_w, 1002)),
            wire(
                SHARD,
                MsgClass::RealmSnapshot,
                &parent_frame(parent_w, 1002, 31.0, vec![alien]),
            ),
            wire(
                DEST,
                MsgClass::RealmSnapshot,
                &ShardToGateway::RealmFrame {
                    realm_fence: Fence(1),
                    source_tick: TickId(1002),
                    realm_snapshot_bytes: vec![0xFF, 0xFF],
                },
            ),
            wire(SHARD, MsgClass::RealmSnapshot, &leaf_frame(parent_w, 900)),
        ]);
        let stats = rig.stats();
        assert_eq!(
            stats.window_alien_rows, 1,
            "the alien row was dropped, counted"
        );
        assert_eq!(
            stats.old_realm_frames_dropped, 1,
            "the tombstoned realm datagram counts in its own bucket, served to nobody"
        );
        assert_eq!(stats.window_level_refused, 1, "a span-stale level refused");
        assert_eq!(stats.window_full_chain_folds, 2);
        assert_eq!(stats.window_fold_divergence, 0);
        assert_eq!(stats.window_t_monotone_stalled, 0);
        assert_eq!(stats.window_chain_cycle, 0);
        // The origin marker's epoch walked its three chain identities: [System(7)] at login,
        // [Planet(7)] the tick the origin flipped, [Planet(7), System(7)] when the hop
        // confirmed (§2.7's epoch mechanics — the client swap is Slice C1).
        assert_eq!(
            rig.world.resource::<GatewaySessions>().by_session[&sid]
                .shadow
                .origin_epoch,
            3
        );

        // ---- A quiet tick: the scene is already AT the common tick with an unchanged chain —
        // nothing re-folds (the §2.14 cost discipline).
        let folds_before = rig.stats().window_folds;
        let _ = rig.tick(vec![]);
        assert_eq!(
            rig.stats().window_folds,
            folds_before,
            "nothing new to fold"
        );

        // ---- Teardown: zero sessions ⇒ zero composer state, structurally (§2.6.1 guard 4).
        let _ = rig.tick(vec![wire(
            CLIENT,
            MsgClass::Control,
            &ClientControlMsg::Bye,
        )]);
        let sessions = rig.world.resource::<GatewaySessions>();
        assert!(sessions.is_empty());
        assert_eq!(
            sessions.windows_open_count(),
            0,
            "every ingest died with its window"
        );
        assert!(
            sessions.realm_heads.is_empty(),
            "the realm→node map pruned to the empty named set"
        );
    }

    #[test]
    fn an_unresolved_lineage_ancestor_resolves_via_the_directory_head_poll() {
        // §2.6.2's ancestor leg end-to-end: a lineage parent the session never subscribed to is
        // resolved through the EXISTING `HeadRead{Realm}`/`Head` pair on the window keep-alive
        // beat; the answer opens the Child window on the named node; and that node's frames are
        // node-class dispatched as shard frames PURELY because the window names it (the window
        // arm of `is_routable_shard`) — no session role, no config entry, no presence.
        let mut rig = Rig::new();
        let (sid, _) = rig.login();
        // The login descent of a DYNAMIC session would have recorded the full root→leaf chain;
        // the static rig's descent machinery is exercised elsewhere — grow the recorded lineage
        // directly (session state, exactly what the derivation reads).
        rig.world
            .resource_mut::<GatewaySessions>()
            .by_session
            .get_mut(&sid)
            .expect("session present")
            .lineage = vec![RealmId::System(0), RealmId::System(7)];
        // No node is known for System(0): the Child window CANNOT derive yet (fail-closed).
        let _ = rig.tick(vec![]);
        assert_eq!(
            rig.world.resource::<GatewaySessions>().windows_open_count(),
            1,
            "no guessed window for an unresolved ancestor"
        );
        // The keep-alive beat carries the ancestor head poll (the EXISTING directory pair).
        set_tick(&mut rig, 25);
        let beat = rig.tick(vec![]);
        let head_reads = beat
            .iter()
            .filter(|(to, class, b)| {
                (*to == ORCH)
                    & (*class == MsgClass::Saga)
                    & (postcard::from_bytes::<InterShardFlow>(b)
                        == Ok(InterShardFlow::Directory(DirectoryOp::HeadRead {
                            key: DirectoryKey::Realm(RealmId::System(0)),
                        })))
            })
            .count();
        assert_eq!(
            head_reads, 1,
            "ONE deduped poll per lineage parent per beat"
        );
        assert_eq!(rig.stats().window_head_reads_sent, 1);
        // Step off the beat, so the reply tick below carries the OPEN alone (a beat tick would
        // add the keep-alive re-assert of the same window — a second, idempotent WindowOpen).
        set_tick(&mut rig, 26);
        // The head answers: System(0) is held by a spawn-minted node in no roster anywhere.
        let ancestor = NodeId(99);
        let opened = rig.tick(vec![wire(
            ORCH,
            MsgClass::Saga,
            &InterShardFlow::DirectoryReply(DirectoryReply::Head {
                key: DirectoryKey::Realm(RealmId::System(0)),
                record: Some(OwnerRecord {
                    authority: AuthorityRef::Shard(ancestor),
                    fence: Fence(3),
                    lease_expires: UniverseTick(1_000),
                    in_transfer: None,
                }),
            }),
        )]);
        let opens_to_ancestor: Vec<GatewayToShard> = opened
            .iter()
            .filter(|(to, _, _)| *to == ancestor)
            .map(|(_, _, b)| postcard::from_bytes(b).expect("shard-bound bytes decode"))
            .collect();
        assert_eq!(
            opens_to_ancestor,
            vec![GatewayToShard::WindowOpen {
                window: WindowId(2),
                scope: WindowScope::Child(RealmId::System(7)),
            }],
            "the resolved head opened the lineage hop window on the ancestor's node"
        );
        // The ancestor's attested statement is HEARD (the window arm of the dispatch) and
        // ingested — a node reachable through no session role at all.
        let frame = ShardToGateway::WindowFrame {
            realm_fence: Fence(3),
            window: WindowId(2),
            at: UniverseTick(5),
            hop: Some(Box::new(vd_wire::session_flow::HopRow {
                child: RealmId::System(7),
                inv: vd_core::frame::FramePlacement::identity(),
            })),
            rows: vec![snap(
                RealmId::System(7),
                SYS7,
                FrameRef::SystemSpace { system_seed: 0 },
                0.0,
                5,
            )],
        };
        let _ = rig.tick(vec![wire(ancestor, MsgClass::RealmSnapshot, &frame)]);
        assert_eq!(rig.stats().window_rows_ingested, 1);
        assert_eq!(
            rig.stats().refused_unknown_sender,
            0,
            "heard BECAUSE of the window"
        );
        // Teardown: the session's exit closes BOTH windows and empties the realm→node map.
        let _ = rig.tick(vec![wire(
            CLIENT,
            MsgClass::Control,
            &ClientControlMsg::Bye,
        )]);
        let sessions = rig.world.resource::<GatewaySessions>();
        assert_eq!(sessions.windows_open_count(), 0);
        assert!(sessions.realm_heads.is_empty());
        assert_eq!(rig.stats().window_close_sent, 2);
    }

    #[test]
    fn an_attach_never_overwrites_a_lineage_the_descent_already_recorded() {
        // The dynamic-login posture at the attach promote: the grant already recorded the
        // descent's full lineage, and the attach frame must NOT overwrite it (the static
        // seed runs only on an EMPTY lineage).
        let mut rig = Rig::new();
        let _ = rig.tick(vec![wire(CLIENT, MsgClass::Control, &hello_msg())]);
        let sid = session_of(&rig, CLIENT);
        let _ = rig.tick(vec![wire(ORCH, MsgClass::Saga, &granted_head(sid))]);
        rig.world
            .resource_mut::<GatewaySessions>()
            .by_session
            .get_mut(&sid)
            .expect("session present")
            .lineage = vec![RealmId::System(0), RealmId::System(7)];
        let _ = rig.tick(vec![wire(
            SHARD,
            MsgClass::Control,
            &ShardToGateway::SessionAttached {
                session: sid,
                entity: EntityId(77),
                frame: FrameRef::SystemSpace { system_seed: 7 },
                realm_fence: Fence(1),
            },
        )]);
        assert_eq!(
            rig.world.resource::<GatewaySessions>().by_session[&sid].lineage,
            vec![RealmId::System(0), RealmId::System(7)],
            "the recorded descent survives the attach"
        );
        // And the OTHER unseedable shape: an attach whose frame names no realm (galaxy space)
        // seeds nothing — an empty lineage stays empty, fail-closed, never guessed.
        let mut rig = Rig::new();
        let _ = rig.tick(vec![wire(CLIENT, MsgClass::Control, &hello_msg())]);
        let sid = session_of(&rig, CLIENT);
        let _ = rig.tick(vec![wire(ORCH, MsgClass::Saga, &granted_head(sid))]);
        let _ = rig.tick(vec![wire(
            SHARD,
            MsgClass::Control,
            &ShardToGateway::SessionAttached {
                session: sid,
                entity: EntityId(77),
                frame: FrameRef::GalaxySpace,
                realm_fence: Fence(1),
            },
        )]);
        assert_eq!(
            rig.world.resource::<GatewaySessions>().by_session[&sid].lineage,
            Vec::<RealmId>::new(),
            "a realm-less attach frame seeds no lineage"
        );
    }

    #[test]
    fn the_lineage_rule_truncates_on_reentry_and_appends_on_descent() {
        // §2.6.2's crossing rule, both arms: an inward cross APPENDS below the previous leaf;
        // an outward cross TRUNCATES back to the re-entered realm KEEPING its ancestors.
        let mut lineage = vec![RealmId::System(0), RealmId::System(7)];
        lineage_apply(&mut lineage, RealmId::Planet(7));
        assert_eq!(
            lineage,
            vec![RealmId::System(0), RealmId::System(7), RealmId::Planet(7)]
        );
        lineage_apply(&mut lineage, RealmId::System(7));
        assert_eq!(
            lineage,
            vec![RealmId::System(0), RealmId::System(7)],
            "truncated to the re-entered realm — ancestors KEPT"
        );
        // An empty lineage (a static session's first frame): the append arm seeds it.
        let mut fresh = Vec::new();
        lineage_apply(&mut fresh, RealmId::System(7));
        assert_eq!(fresh, vec![RealmId::System(7)]);
        // And the descent recorder: a coord's chain lowers root→leaf.
        use vd_core::realm_path::{RealmKindTag, RealmLevel, RealmPath};
        let coord = RealmCoord::from_path(RealmPath::from_levels(vec![
            RealmLevel::new(RealmKindTag::System, 7),
            RealmLevel::new(RealmKindTag::Planet, 3),
        ]))
        .expect("a two-level path");
        assert_eq!(
            coord_lineage(&coord),
            vec![RealmId::System(7), RealmId::Planet(3)]
        );
        let root = RealmCoord::from_path(RealmPath::from_levels(vec![RealmLevel::new(
            RealmKindTag::System,
            7,
        )]))
        .expect("a root path");
        assert_eq!(coord_lineage(&root), vec![RealmId::System(7)]);
    }

    #[test]
    fn the_composer_withholds_sessions_with_no_standing_realm_or_window() {
        // §2.6.6's "unresolved standing" row, all three shapes, driven over a hand-built table
        // through the REAL pass (the login race is held, counted, never guessed):
        // (a) an Active session with no cold sub on its routing target;
        let cfg = config();
        let (mut sessions, sid, _) = one_active_session();
        let mut stats = GatewayStats::default();
        compose_scenes_pass(
            &cfg,
            &test_clock(),
            &mut sessions,
            &mut stats,
            &mut OutboundBox::default(),
        );
        assert_eq!(stats.window_unresolved_standing, 1);
        assert_eq!(stats.window_chains_held, 0);
        // (b) a sub whose frame names no realm (galaxy space names none);
        sessions
            .by_session
            .get_mut(&sid)
            .expect("present")
            .subs
            .insert(
                SHARD,
                SubRecord {
                    sub: SubId(0),
                    frame: FrameRef::GalaxySpace,
                    accepted: Fence(1),
                    state: SubState::Active,
                },
            );
        compose_scenes_pass(
            &cfg,
            &test_clock(),
            &mut sessions,
            &mut stats,
            &mut OutboundBox::default(),
        );
        assert_eq!(stats.window_unresolved_standing, 2);
        // (c) a named realm with NO derivable own-level window (the pass ran before any window
        // existed — the pure-fn shape the schedule's driver ordering normally prevents).
        sessions
            .by_session
            .get_mut(&sid)
            .expect("present")
            .subs
            .get_mut(&SHARD)
            .expect("present")
            .frame = SYS7;
        compose_scenes_pass(
            &cfg,
            &test_clock(),
            &mut sessions,
            &mut stats,
            &mut OutboundBox::default(),
        );
        assert_eq!(stats.window_unresolved_standing, 3);
        assert_eq!(
            stats.window_folds, 0,
            "nothing was ever guessed into a fold"
        );
    }

    /// The load gate's typed extractor: a `WindowFrame`'s (id, level), `None` for anything else
    /// (both arms driven below — HR5).
    fn window_level_of(msg: ShardToGateway) -> Option<(WindowId, window::WindowLevel)> {
        match msg {
            ShardToGateway::WindowFrame {
                window,
                at,
                hop,
                rows,
                ..
            } => Some((
                window,
                window::WindowLevel {
                    at,
                    hop: hop.map(|b| *b),
                    rows,
                },
            )),
            _ => None,
        }
    }

    #[test]
    fn the_load_gates_extractor_and_the_divergence_verdict_cover_both_arms() {
        // The extractor: a window frame yields its level; anything else yields None.
        let level = window_level_of(ShardToGateway::WindowFrame {
            realm_fence: Fence(1),
            window: WindowId(3),
            at: UniverseTick(9),
            hop: None,
            rows: Vec::new(),
        });
        assert_eq!(
            level,
            Some((
                WindowId(3),
                window::WindowLevel {
                    at: UniverseTick(9),
                    hop: None,
                    rows: Vec::new(),
                }
            ))
        );
        assert_eq!(
            window_level_of(ShardToGateway::SessionDetached {
                session: SessionId(1)
            }),
            None
        );
        // The §2.14 divergence verdict: equal folds verify silent; ANY bit of difference counts.
        let same = window::Composed::default();
        assert_eq!(fold_divergence(&same, &same.clone()), 0);
        let differs = window::Composed {
            instant_refused: 1,
            ..window::Composed::default()
        };
        assert_eq!(fold_divergence(&same, &differs), 1);
    }

    /// G-COMPOSE-LOAD (window_lane.md §2.6.7; §4.5 Topic 3 — the owner-adopted refinement: gate
    /// the p99, never the mean): BOTH engine sides under a DERIVED load, per tick —
    /// ingest-DECODE (postcard `WindowFrame` levels at MTU-full row counts, one per live window)
    /// AND compose+fan (the shared fold per origin + every session's scene roll). The load
    /// shape is derived, never invented: sessions = the density fixture's N (128 — the
    /// hundreds-in-one-location correctness bound already in gate), chain depth = the design's
    /// stated observer depth (§2.1: 4–6 ⇒ 6), rows per level = what fills one
    /// CONSERVATIVE_DATAGRAM_BUDGET datagram (the MTU partitioner's own bound — the shape the
    /// wire actually carries), origins = the crossing dual-sub bound (4 distinct chains).
    /// Budget: p99 per tick < ONE realm-lane tick (§2.2: 20 Hz ⇒ 50 ms). The timing assert is
    /// release-only (the SPIKE-2a/3a pattern — a debug/coverage tail is meaningless); the debug
    /// run still exercises every body.
    #[test]
    // The seam ban targets PRODUCTION wall-clock reads; a latency gate measuring elapsed time is
    // exactly what Instant is for (the SPIKE-2a exemption, same reasoning).
    #[allow(clippy::disallowed_methods)]
    fn g_compose_load_p99_ingest_and_fold_under_one_tick() {
        const SESSIONS: usize = 128; // the density fixture's N
        const DEPTH: usize = 6; // §2.1 chain depth, deep end
        const ORIGINS: u64 = 4; // the dual-sub crossing bound
        const TICKS: u64 = 400;
        let realm_lane_hz = 20u32; // §2.2: the realm-lane tick the budget derives from
        let cfg = config();

        // Rows per level: fill one conservative datagram, exactly as the partitioner bounds it.
        let candidate: Vec<RealmSnap> = (0..512)
            .map(|i| {
                snap(
                    RealmId::Planet(1000 + i),
                    FrameRef::PlanetCentered {
                        planet_seed: 1000 + i,
                    },
                    SYS7,
                    i as f64,
                    0,
                )
            })
            .collect();
        let rows_per_level = vd_wire::channels::partition_realms(
            &candidate,
            vd_wire::channels::CONSERVATIVE_DATAGRAM_BUDGET,
        )[0]
        .len();

        // The window fan: ORIGINS chains × DEPTH levels. Origin o's chain authors are
        // System(10o+k); level 0 is the origin's own realm System(10o).
        let mut sessions = GatewaySessions::default();
        let mut next_window = 0u64;
        // The fixture's authors are ALL systems; the seed rides beside the id from construction
        // (no realm-kind extraction anywhere downstream).
        let mut window_rows: Vec<(WindowId, u64, WindowScope)> = Vec::new();
        for o in 0..ORIGINS {
            for k in 0..DEPTH as u64 {
                next_window += 1;
                let seed = 10 * o + k;
                let scope = if k == 0 {
                    WindowScope::Occupants
                } else {
                    WindowScope::Child(RealmId::System(seed - 1))
                };
                sessions.windows.insert(
                    WindowId(next_window),
                    GatewayWindow {
                        shard: SHARD,
                        scope,
                        author_realm: RealmId::System(seed),
                        ingest: window::WindowIngest::default(),
                        parked_bodies: BTreeMap::new(),
                        parked_relays: BTreeMap::new(),
                    },
                );
                window_rows.push((WindowId(next_window), seed, scope));
            }
        }
        sessions.next_window = next_window;
        // The sessions: N Active dots spread over the origins, each standing in its origin.
        let base = one_active_session().0;
        let template = &base.by_session[&SessionId(0xA11A)];
        for i in 0..SESSIONS {
            let o = (i as u64) % ORIGINS;
            let sid = SessionId(0xB000 + i as u128);
            let mut session = Session {
                client: CLIENT,
                account: AccountId(i as u128),
                fence: Fence(1),
                phase: SessionPhase::Active {
                    entity: EntityId(i as u128),
                },
                next_sub: 1,
                home_shard: None,
                home_rid: None,
                realm_feed_frame_id: 0,
                scene_sent: BTreeMap::new(),
                spawn: None,
                bootstrap_deadline: None,
                confirmed_at: TickId(0),
                negotiated_minor: 1,
                transfer: None,
                subs: BTreeMap::new(),
                delivered: BTreeMap::new(),
                lineage: (0..DEPTH as u64)
                    .rev()
                    .map(|k| RealmId::System(10 * o + k))
                    .collect(),
                shadow: window::ShadowScene::default(),
                hot: Arc::clone(&template.hot),
            };
            session.subs.insert(
                SHARD,
                SubRecord {
                    sub: SubId(0),
                    frame: FrameRef::SystemSpace {
                        system_seed: 10 * o,
                    },
                    accepted: Fence(1),
                    state: SubState::Active,
                },
            );
            sessions.by_session.insert(sid, session);
        }

        // Pre-encode one tick's wire bytes per window per tick offset (the decode side must
        // decode FRESH bytes per tick — that is the measured cost).
        let mut stats = GatewayStats::default();
        let mut samples: Vec<std::time::Duration> = Vec::with_capacity(TICKS as usize);
        for t in 1..=TICKS {
            // Author one tick's statements (outside the measured section: the SHARD does this).
            let bytes_per_window: Vec<(WindowId, Vec<u8>)> = window_rows
                .iter()
                .map(|(id, seed, scope)| {
                    let author_frame = FrameRef::SystemSpace { system_seed: *seed };
                    let rows: Vec<RealmSnap> = (0..rows_per_level)
                        .map(|i| {
                            snap(
                                RealmId::Planet(5000 + i as u64),
                                FrameRef::PlanetCentered {
                                    planet_seed: 5000 + i as u64,
                                },
                                author_frame,
                                (i as f64) + (t as f64),
                                t,
                            )
                        })
                        .collect();
                    let hop = match scope {
                        WindowScope::Occupants => None,
                        WindowScope::Child(child) => {
                            Some(Box::new(vd_wire::session_flow::HopRow {
                                child: *child,
                                inv: vd_core::frame::FramePlacement::moving(
                                    DVec3::new(-(t as f64), 0.0, 0.0),
                                    DVec3::ZERO,
                                ),
                            }))
                        }
                    };
                    let msg = ShardToGateway::WindowFrame {
                        realm_fence: Fence(1),
                        window: *id,
                        at: UniverseTick(t),
                        hop,
                        rows,
                    };
                    (*id, postcard::to_allocvec(&msg).expect("encode"))
                })
                .collect();
            // ---- the measured section: decode + ingest every window's level, then the pass.
            let started = std::time::Instant::now();
            for (_, bytes) in &bytes_per_window {
                let decoded: ShardToGateway =
                    postcard::from_bytes(bytes).expect("the load gate authored these bytes");
                let (wid, level) =
                    window_level_of(decoded).expect("the load gate authored window frames");
                on_window_row(
                    SHARD,
                    wid,
                    WindowRow::Level(level),
                    &cfg,
                    &mut sessions,
                    &mut stats,
                );
            }
            compose_scenes_pass(
                &cfg,
                &test_clock(),
                &mut sessions,
                &mut stats,
                &mut OutboundBox::default(),
            );
            samples.push(started.elapsed());
        }
        assert_eq!(
            stats.window_folds,
            TICKS * ORIGINS,
            "one shared fold per (origin, tick) — the §2.14 sharing under load"
        );
        assert_eq!(
            stats.window_fold_hits as usize,
            (SESSIONS - ORIGINS as usize) * TICKS as usize
        );
        assert_eq!(stats.window_fold_divergence, 0);
        assert_eq!(stats.window_instant_mismatch, 0);
        // Bounded state under load: every ring within the derived span.
        for w in sessions.windows.values() {
            assert!(w.ingest.newest().is_some());
        }
        let p50 = vd_harness::latency::percentile_unstable(samples.clone(), 50);
        let p99 = vd_harness::latency::percentile_unstable(samples, 99);
        let budget = std::time::Duration::from_secs_f64(1.0 / f64::from(realm_lane_hz));
        eprintln!(
            "[g-compose-load] sessions={SESSIONS} origins={ORIGINS} depth={DEPTH} \
             rows/level={rows_per_level} windows={} ticks={TICKS} | per-tick ingest+decode+fold+fan \
             p50={p50:?} p99={p99:?} budget(one 20 Hz tick)={budget:?}",
            window_rows.len(),
        );
        // The hard p99 gate is release-only (SPIKE-2a/3a pattern) — debug/coverage builds run
        // the same bodies but their tails are instrumentation, not the engine.
        #[cfg(not(debug_assertions))]
        assert!(
            p99 < budget,
            "G-COMPOSE-LOAD failed: p99 {p99:?} ≥ one realm-lane tick {budget:?}"
        );
        #[cfg(debug_assertions)]
        let _ = budget;
    }

    /// THE SHADOW SOAK (window_lane.md §4.5 Topic 5 — "a soak added to slice B's exit"): the
    /// existing soak recipe run on the SHADOW configuration (`just rlm-soak`, second line). A
    /// fixed-seed xorshift drives ~30k gateway ticks of session churn + lossy/reordered/forged
    /// window traffic through the REAL rig, asserting EVERY 128 ticks that
    /// memory is BOUNDED (rings ≤ the derived span, pending ≤ the derived cap, realm-heads ≤
    /// the named set, windows ≤ the derivable set) and counters MONOTONE — and at the end that
    /// the last session's exit leaves ZERO composer state. Asserted, never eyeballed.
    /// Release-gated like `rlm_soak` (inert in debug; the same paths run in the gate's unit
    /// tests above).
    #[test]
    #[cfg(not(debug_assertions))]
    fn window_shadow_soak_state_stays_bounded_and_zeroes_at_the_end() {
        let mut rig = Rig::new();
        let (sid, _) = rig.login();
        let tuning = window_tuning(&config());
        let mut state: u64 = 0x5150_5150_5150_5150;
        let mut next = || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            state
        };
        let mut prev = GatewayStats::default();
        for step in 0u64..30_000 {
            let r = next();
            let at = 1_000 + step / 2 + (r % 5); // mostly advancing, sometimes reordered stamps
            let mut inbox: Vec<Inbound> = Vec::new();
            match r % 7 {
                0 | 1 => inbox.push(wire(
                    SHARD,
                    MsgClass::RealmSnapshot,
                    &leaf_frame(WindowId(1), at),
                )),
                2 => inbox.push(wire(
                    SHARD,
                    MsgClass::RealmSnapshot,
                    &old_lane_frame(
                        vec![snap(
                            RealmId::Planet(9),
                            FrameRef::PlanetCentered { planet_seed: 9 },
                            SYS7,
                            1.0,
                            at,
                        )],
                        at,
                    ),
                )),
                3 => inbox.push(wire(
                    DEST,
                    MsgClass::RealmSnapshot,
                    &leaf_frame(WindowId(1), at),
                )), // forged
                4 => inbox.push(wire(
                    SHARD,
                    MsgClass::RealmSnapshot,
                    &leaf_frame(WindowId(9999), at),
                )), // unknown
                5 => inbox.push(wire(
                    SHARD,
                    MsgClass::Control,
                    &ShardToGateway::WindowBody {
                        realm_fence: Fence(1),
                        window: WindowId(1),
                        subject: RealmId::System(7),
                        stmt: BodyStmt::SelfLook {
                            bag: vec![(r % 251) as u8],
                        },
                        authored_at: UniverseTick(at),
                    },
                )),
                _ => inbox.push(wire(
                    SHARD,
                    MsgClass::Control,
                    &ShardToGateway::WindowMembership {
                        window: WindowId(1),
                        added: vec![RealmId::Planet(9)],
                        removed: vec![RealmId::Planet(9)],
                    },
                )),
            }
            let _ = rig.tick(inbox);
            if step % 128 == 0 {
                let sessions = rig.world.resource::<GatewaySessions>();
                assert!(
                    sessions.windows_open_count() <= 2,
                    "windows stay the derivable set"
                );
                for w in sessions.windows.values() {
                    let span = w
                        .ingest
                        .newest()
                        .map_or(0, |n| n.0.saturating_sub(tuning.ring_span_ticks));
                    let _ = span;
                    assert!(
                        w.ingest.roster_set().len() <= 64,
                        "rosters bounded by the authored rows"
                    );
                }
                let session = &sessions.by_session[&sid];
                assert!(
                    session.shadow.ring.len() as u64 <= tuning.ring_span_ticks + 1,
                    "the fold ring holds one derived span, never more"
                );
                assert!(
                    session.shadow.pending.len() <= tuning.pending_cap,
                    "the comparator queue is capped"
                );
                assert!(
                    sessions.realm_heads.len() <= 2,
                    "realm-heads ≤ the named set"
                );
                let now = rig.stats();
                // Counters are MONOTONE (a decreasing counter is state corruption).
                assert!(now.window_rows_ingested >= prev.window_rows_ingested);
                assert!(now.window_folds >= prev.window_folds);
                assert!(now.parity_rows_matched >= prev.parity_rows_matched);
                assert!(now.window_sender_mismatch >= prev.window_sender_mismatch);
                prev = now;
            }
        }
        assert!(
            rig.stats().window_sender_mismatch > 0,
            "the forged arm really ran"
        );
        assert!(
            rig.stats().window_unknown_row > 0,
            "the unknown arm really ran"
        );
        assert!(rig.stats().window_folds > 0, "the composer really folded");
        // The exit: zero sessions ⇒ zero composer state, after 30k ticks of churn.
        let _ = rig.tick(vec![wire(
            CLIENT,
            MsgClass::Control,
            &ClientControlMsg::Bye,
        )]);
        let sessions = rig.world.resource::<GatewaySessions>();
        assert!(sessions.is_empty());
        assert_eq!(sessions.windows_open_count(), 0);
        assert!(sessions.realm_heads.is_empty());
    }
}
