//! THE GATEWAY'S DIALS, AND THE NUMBERS DERIVED FROM THEM.
//!
//! Owns: the transport tuning, the dynamic-home injector config with the boot-time validation that
//! rejects a mis-ordered budget LOUD rather than running one, and the cadences the window lane and
//! the retries beat on. Every one is a reviewed field on a config struct, never a literal at a use
//! site.
//!
//! Does NOT own: any accuracy bound on somebody else's simulation. A router cannot decide how far a
//! simulation may extrapolate its own velocity — that policy lives with the crate that authors the
//! velocity, and it used to be a field here, which was the wrong home for it.

use crate::tickets;
use crate::window;
use bevy_ecs::prelude::Resource;
use std::collections::BTreeSet;
use vd_core::NodeId;
use vd_core::glam::DVec3;
use vd_core::home::{HomeRegistry, StoredHome};
use vd_core::worldgen::WorldRealms;
use vd_wire::seams::transfer_control::PrepareReject;

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
    /// THE GALAXY'S STARS (S11) — folded ONCE at boot, and stated to each client once.
    ///
    /// ★ THE OWNER RULED THIS ON 2026-08-27: *"we're passing the Galaxy just once over reliable lane."*
    /// ONCE means one sky, the same for every player, sent one time — not once per shard, not once per
    /// realm, and not again when a player crosses.
    ///
    /// ★ WHY THE GATEWAY AND NOT A SHARD. A shard folds its sky from the realms IT booted, so the sky
    /// you received depended on which shard you were subscribed to: a home system shard states ONE star
    /// — its own — and your own star is never drawn, because you are standing inside it. MEASURED
    /// 2026-08-27 on a dual cluster: the galaxy shard held 3 stars, the client held 1, and drew 0. The
    /// gateway is the one party that holds every session and the whole forest, so it is where a thing
    /// that crosses ONCE belongs.
    ///
    /// ★ ROWS, NEVER A WORLD (SL4). The boot folds these because only the boot sees both the generator
    /// and the simulation. The routing plane receives a finished list of stars and cannot name a body,
    /// an orbit or a motion.
    ///
    /// Empty on a gateway booted without a world — it then states no sky, rather than a wrong one.
    pub sky: Vec<vd_core::look::StarRow>,
    /// The generation [`Self::sky`] folds to — the identity the client compares against what it holds.
    pub sky_generation: u64,
    /// The frame the sky is stated in — the galaxy's — so the composer can lift each observer's
    /// origin into it and place the star cloud (owner ruling 2026-09-02 R1). `None` with no sky.
    pub sky_frame: Option<vd_core::pose::FrameRef>,
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
    /// ★ THE PER-TICK TRACE (2026-09-05, owner: *"lets measure and understand better how all
    /// works"*): when `Some(kind)`, every drawn row of that realm kind is logged once per fold per
    /// session (`target: "vd_trace"`) with its composed pose, velocity, distance, source (fresh or
    /// held) and the chain's state. `None` (the default; `VD_TRACE_REALM` unset) logs nothing.
    pub trace_realm_kind: Option<vd_core::realm_path::RealmKindTag>,
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

/// `VD_TRACE_REALM`'s value → the realm kind to trace; an unknown word is `None` (nothing traced,
/// never a guess). Case-sensitive, the kind's own name.
#[must_use]
pub fn parse_realm_kind(word: &str) -> Option<vd_core::realm_path::RealmKindTag> {
    use vd_core::realm_path::RealmKindTag as K;
    match word {
        "Universe" => Some(K::Universe),
        "Galaxy" => Some(K::Galaxy),
        "System" => Some(K::System),
        "Planet" => Some(K::Planet),
        "Station" => Some(K::Station),
        "Area" => Some(K::Area),
        "Star" => Some(K::Star),
        _ => None,
    }
}

impl GatewayConfig {
    /// Is `from` a routable shard per the FROZEN config roster (STABLE node-class dispatch — FORK 5)?
    /// Seeded from the cluster's shard roster, provably disjoint from client NodeIds, so it can never
    /// mis-class a client datagram as a shard frame regardless of subscription churn. The RUNTIME half of
    /// the dispatch (demand-spawned home shards) is [`is_routable_shard`].
    #[must_use]
    pub(crate) fn is_known_shard(&self, from: NodeId) -> bool {
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
    pub(crate) fn dynamic_home_mode(&self, clock_synced: bool) -> bool {
        self.seed_injector.armed & clock_synced
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
pub(crate) fn window_keepalive_cadence(config: &GatewayConfig) -> u64 {
    if config.session_recheck_interval > 0 {
        config.session_recheck_interval
    } else {
        (u64::from(config.tick_hz) / 2).max(1)
    }
}

/// THE composer's derived-bounds home, off the ONE keep-alive beat the gateway already re-asserts
/// windows on ([`window_keepalive_cadence`]) — subscriber, holder TTL and composer retention all
/// breathe on one rhythm (`docs/design/window_lane.md` §2.6.7; owner law 3(a)).
pub(crate) fn window_tuning(config: &GatewayConfig) -> window::WindowTuning {
    window::WindowTuning::derive(window_keepalive_cadence(config), config.tick_hz)
}
