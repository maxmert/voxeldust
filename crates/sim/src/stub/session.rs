//! THE GATEWAY↔SHARD SESSION LANE: attach, admit, adopt, detach.
//!
//! Owns: the dispatch of every [`GatewayToShard`] message, the ONE admit-pose seam a login births
//! its dot through, the transfer-destination input slot (including the one buffered until this
//! realm's lease lands), the grant-flip that turns a provisional dot into an authoritative one, and
//! the replies that go back the other way.
//!
//! Does NOT own: the credential. A shard NEVER sees a ticket — it acts on a `(SessionId, Fence)`
//! handed over this lane and drops a stale-fence input with a logged reason. Nor does it own the
//! directory round-trip that makes a grant real (`realm_head`).

use super::{
    DiscardReason, Dot, Dots, EntityMint, InputLog, OpenWindows, RealmRegions, StubConfig,
    StubStats, apply_input, mint_entity, on_window_close, on_window_open,
};
use crate::authority::{Authority, AuthorityCmd};
use crate::io::{Durability, MsgClass};
use crate::runtime::{ClockSample, NodeIdentity, OutboundBox};
use bevy_ecs::prelude::Resource;
use std::collections::BTreeMap;
use vd_core::glam::DVec3;
use vd_core::kinematics::{self};
use vd_core::placement::PlacementLedger;
use vd_core::pose::{LatticePos, StampedPose};
use vd_core::{AccountId, EntityId, Fence, NodeId, SessionId, UniverseTick};
use vd_wire::intershard::InterShardFlow;
use vd_wire::seams::directory::{AuthorityRef, DirectoryKey, DirectoryOp};
use vd_wire::session_flow::{GatewayToShard, ShardToGateway};

/// One transfer-destination input slot held until this shard's realm lease lands (Stage B2 — closes
/// DEFERRED D-8's app-drop loss point). The gateway emits `OpenInputSlot` exactly ONCE (at
/// `apply_commit`) and the saga is forward-only past the commit, so dropping the slot for a late
/// lease stranded the adopt PERMANENTLY — the reproduced fence-7 strand (rehome_one_mechanism §4v,
/// fact 3). Held here instead and drained the moment the lease is affirmed; the adopt HeadRead pump
/// then flips the grant and `PendingCrossings` drains as usual.
#[derive(Debug, Clone, Copy, PartialEq)]
struct PendingInputSlot {
    fence: Fence,
    account: AccountId,
    resume_from_seq: u64,
    subject: DirectoryKey,
    from: NodeId,
}

/// Input slots buffered awaiting the realm lease, keyed by session (Stage B2). Holds ≤(concurrent
/// inbound transfers) entries transiently; an entry whose lease never lands shares
/// [`PendingCrossings`]' retention posture (owed with the journal-retention slice, D-22).
#[derive(Resource, Debug, Default)]
pub struct PendingInputSlots(BTreeMap<SessionId, PendingInputSlot>);

/// Read-only context for one gateway-message dispatch.
pub(crate) struct GatewayMsgCtx<'a> {
    pub(crate) config: &'a StubConfig,
    pub(crate) identity: &'a NodeIdentity,
    pub(crate) clock: &'a ClockSample,
    pub(crate) realm_fence: Option<Fence>,
    /// The speed law's read-only inputs (S3): the region forest and the authored placement ledger
    /// the input integrator's governor reads — the SAME stores every other consumer reads.
    pub(crate) regions: &'a RealmRegions,
    pub(crate) placements: &'a PlacementLedger,
    /// ★ DOES THIS REALM FLY ON AN OCCUPANT'S STICK? A hull with engines and a stored body does.
    /// An occupant of such a realm is AT THE CONTROLS: its stick becomes the realm's push, and its
    /// own body does not also walk. Derived at the one call site from the capability plus the stored
    /// body — never from what kind of realm this is (HR3/SL4).
    pub(crate) flies_on_a_stick: bool,
}

/// Resolve the BIRTH pose for a login-admitted avatar — THE one admit-pose seam (HR3): every login births
/// its dot through here, with NO origin/home fork (only the pose value differs).
///
/// TWO SOURCES, asked in order, and BOTH must already be measured from THIS realm's own centre:
///   1. `offered` — the pose the gateway put in `AttachSession`, already measured from this realm's centre.
///      It is descended there over a private copy of the whole seed forest, which is the one place in the
///      login path where a party that owns no realm still does realm arithmetic. That is a KNOWN OPEN
///      BREACH, not a design: the descent belongs level by level in the chain, and the reason it is not
///      there yet is a bootstrap circularity measured in `home_placement` (gateway.rs) and asserted by
///      `nothing_of_the_home_lineage_is_running_at_the_instant_the_login_descent_must_answer`. Whoever
///      offers the pose, THIS end is unchanged: it is checked, never converted.
///   2. `config.spawn_poses` — this realm's OWN stored poses, keyed by account. Empty today; the P7 durable
///      per-realm pose store fills exactly this map, and a per-realm store's poses are realm-local by
///      construction. Nothing on the boot path fills it any more.
///
/// Neither ⇒ origin-at-rest in `config.frame` — the literal every rig has always taken.
///
/// A POSE IN THE WRONG FRAME IS REFUSED, COUNTED AND LOGGED, and the avatar births at the origin instead.
/// This is the whole point of the change. It used to be handed to [`rebind_pose_to_dest`] with an identity
/// frame context, which RENAMED the frame and moved no number — so a stored universe-absolute (145 m from
/// its star, 3 m above its planet) was planted 3 m… from the star, 145 m from where the player left. The
/// number was never converted; it just started claiming to be local. After the fold's removal there is no
/// universe-root frame anybody may measure in at all, so there is nothing left to relabel FROM: a pose that
/// is not in this shard's frame is a routing mistake, and the honest answer is to say so out loud rather
/// than plant the player somewhere arbitrary.
///
/// Concrete (non-generic); every arm is exercised by the unit tests; no wall-clock/rng (the `tick` is the
/// clock's). The chosen pose is [`StampedPose::sanitized`] (a config / P7-store / wire pose can never
/// poison the sim) and RE-STAMPED to the current tick.
pub(crate) fn resolve_spawn_pose(
    config: &StubConfig,
    account: AccountId,
    offered: Option<StampedPose>,
    tick: UniverseTick,
    stats: &mut StubStats,
) -> StampedPose {
    let origin = StampedPose::at_rest(config.frame, DVec3::ZERO, tick);
    let Some(stored) = offered.or_else(|| config.spawn_poses.get(&account).copied()) else {
        return origin;
    };
    let stored = StampedPose {
        universe_tick: tick,
        ..stored.sanitized()
    };
    if stored.frame != config.frame {
        stats.spawn_poses_refused += 1;
        tracing::error!(
            offered_frame = ?stored.frame,
            own_frame = ?config.frame,
            realm = %config.realm,
            "a spawn pose measured in a frame this realm is not — birthing at the origin instead. \
             A realm cannot convert a position into its own frame: it does not know where it itself \
             sits, and the party that put it there must state the pose in this frame or not at all.",
        );
        return origin;
    }
    stored
}

/// Handle one gateway→shard message.
#[allow(clippy::too_many_arguments)]
pub(crate) fn on_gateway_msg(
    bytes: &[u8],
    from: NodeId,
    ctx: &GatewayMsgCtx<'_>,
    dots: &mut Dots,
    mint: &mut EntityMint,
    log: &mut InputLog,
    pending_slots: &mut PendingInputSlots,
    windows: &mut OpenWindows,
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
            spawn,
        } => {
            // An avatar cannot exist on a shard that doesn't own its realm yet:
            // defer (counted); the gateway retries attach until it sees the reply.
            let Some(realm_fence) = ctx.realm_fence else {
                stats.attaches_deferred += 1;
                return;
            };
            let dot = dots.0.entry(session).or_insert_with(|| {
                let entity = mint_entity(mint, ctx.identity.node_id);
                // Birth at the pose the gateway measured against THIS realm (or this realm's own stored
                // one), or at the origin when there is none. ONE admit path — only the pose value changes;
                // the Ghost birth → LeaseGrant → Promote → SessionAttached flow is unchanged.
                let pose =
                    resolve_spawn_pose(ctx.config, account, spawn, ctx.clock.universe_tick, stats);
                Dot {
                    last_stick: None,
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
                    // The angles come FROM the pose, never from zero: the integrator rebuilds the
                    // orientation from them every tick, so a birth that zeroed them would face the
                    // dot at the frame's default no matter what pose it was handed
                    // (`kinematics::yaw_pitch_from_orient` — the two stores of one truth).
                    yaw: kinematics::yaw_pitch_from_orient(pose.orient).0,
                    pitch: kinematics::yaw_pitch_from_orient(pose.orient).1,
                    last_applied_seq: None,
                    // Seed to the spawn offset: tick-1's swept segment is degenerate. Origin when no stored
                    // pose (byte-identical to the old `DVec3::ZERO`); the stored offset otherwise.
                    prev_offset: pose.pos,
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
                ctx.regions,
                ctx.placements,
                ctx.flies_on_a_stick,
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
                    // A provisional dot has no directory record: safe to drop now. NO remove
                    // message — a pre-grant provisional dot never emitted (`emits()` excludes
                    // it), so no bystander's client ever held a track to evict.
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
                // No realm lease yet: BUFFER + count (Stage B2 — closes DEFERRED D-8's app-drop loss
                // point, reproduced live as the fence-7 permanent strand, rehome_one_mechanism §4v
                // fact 3). The gateway emits `OpenInputSlot` exactly once and the saga is forward-only
                // past the commit, so this shard is the only party that can still complete the adopt:
                // hold the slot, and `drain_pending_input_slots` replays it the moment the lease lands.
                stats.input_slots_deferred += 1;
                pending_slots.0.insert(
                    session,
                    PendingInputSlot {
                        fence,
                        account,
                        resume_from_seq,
                        subject,
                        from,
                    },
                );
                tracing::warn!(
                    ?session,
                    ?subject,
                    "OpenInputSlot BUFFERED — no realm lease yet; the adopt drains when the lease lands"
                );
                return;
            };
            adopt_input_slot(
                session,
                fence,
                account,
                resume_from_seq,
                subject,
                from,
                ctx.config,
                ctx.clock,
                dots,
                stats,
            );
        }
        GatewayToShard::WindowOpen {
            window,
            scope,
            static_held,
        } => {
            on_window_open(
                windows,
                from,
                window,
                scope,
                static_held,
                ctx.clock.local_tick,
                stats,
            );
        }
        GatewayToShard::WindowClose { window } => {
            on_window_close(windows, from, window, stats);
        }
        // ASK FOR THE SKY — RETIRED (S11, owner ruling 2026-08-27). A shard states no sky at all now:
        // the GATEWAY holds the galaxy and states it ONCE. The arm stays on the wire because deleting a
        // variant renumbers every later one, and a positional test guards that. Still COUNTED, so a
        // gateway that has not caught up is visible rather than silently ignored.
        GatewayToShard::SkyRequest => {
            stats.sky_requests_taken += 1;
        }
    }
}

/// ADOPT a transfer-destination input slot — the body of `GatewayToShard::OpenInputSlot`, extracted so
/// the lease-affirm drain replays a buffered slot through the IDENTICAL path (one adopt rule, two
/// callers — Stage B2). Mints the adopt-Ghost dot, gates staleness, arms input from the owning
/// gateway, seeds the resume watermark.
#[allow(clippy::too_many_arguments)]
fn adopt_input_slot(
    session: SessionId,
    fence: Fence,
    account: AccountId,
    resume_from_seq: u64,
    subject: DirectoryKey,
    from: NodeId,
    config: &StubConfig,
    clock: &ClockSample,
    dots: &mut Dots,
    stats: &mut StubStats,
) {
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
        last_stick: None,
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
            since_tick: clock.local_tick,
        },
        departing: false,
        entity_fence: Fence::GENESIS, // the adopt HeadRead fills the real CAS fence
        pose: StampedPose::at_rest(config.frame, DVec3::ZERO, clock.universe_tick),
        yaw: 0.0,
        pitch: 0.0,
        last_applied_seq: None,
        // Seed to the spawn offset (origin): tick-1's swept segment is degenerate.
        prev_offset: LatticePos::ORIGIN,
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

/// Drain every input slot buffered while the realm lease was absent, through the IDENTICAL adopt
/// path (Stage B2). Called at the lease-affirm None→Some flip; the adopt HeadRead pump then flips
/// each dot's grant, and `PendingCrossings` drains at that flip — completing a crossing that the
/// dropped-slot posture stranded permanently (rehome_one_mechanism §4v fact 3).
pub(crate) fn drain_pending_input_slots(
    pending_slots: &mut PendingInputSlots,
    config: &StubConfig,
    clock: &ClockSample,
    dots: &mut Dots,
    stats: &mut StubStats,
) {
    for (session, slot) in std::mem::take(&mut pending_slots.0) {
        stats.input_slots_drained += 1;
        tracing::info!(
            ?session,
            subject = ?slot.subject,
            "draining a buffered OpenInputSlot — the realm lease just landed"
        );
        adopt_input_slot(
            session,
            slot.fence,
            slot.account,
            slot.resume_from_seq,
            slot.subject,
            slot.from,
            config,
            clock,
            dots,
            stats,
        );
    }
}

pub(crate) fn push_session_reply(outbox: &mut OutboundBox, to: NodeId, reply: &ShardToGateway) {
    let bytes = postcard::to_allocvec(reply).expect("closed wire enums serialize infallibly");
    outbox.0.push((
        to,
        MsgClass::Control,
        crate::io::bytes(bytes),
        Durability::Ephemeral,
    ));
}

/// THE REMOVE MESSAGE's shard emit (proto_minor 14, D-4(a)): this shard PERMANENTLY stopped
/// emitting `entity`, so every gateway still holding a subscribed session here is told once, on the
/// reliable Control lane, stamped with this realm's fence (the gateway refuses a demoted old
/// owner's stale removal per subscriber) and this shard's universe tick (the client's resurrect
/// guard against a straggler datagram). The fan is the REMAINING dots' gateways — after the
/// removal, exactly the bystanders whose clients hold the departed figure's track. A shard whose
/// last dot just left has nobody left to tell, and the sub teardown that follows is the client's
/// realm-scene exit anyway. Called ONLY from a stop-emitting site: the HOLD-CLOSURE pair (the
/// SpawnV2 take-over proof; the hold-TTL expiry — slice F's vanish moment, while the retained dot
/// lives on silently as the return target), the band-exit ghost despawn (idempotent on clients
/// that already evicted at closure), and a detach completing at the directory. Never from the
/// demote itself — the retained ghost emits through the whole hold window.
pub(crate) fn push_entity_removed(
    dots: &Dots,
    realm_fence: Fence,
    entity: EntityId,
    at: UniverseTick,
    outbox: &mut OutboundBox,
) {
    let mut gateways: Vec<NodeId> = dots.0.values().map(|d| d.gateway).collect();
    gateways.sort_unstable();
    gateways.dedup();
    for gateway in gateways {
        // RETAINED, not Ephemeral (review-caught): this is a PRODUCER-LESS reliable one-shot —
        // neither emit site re-fires (a redelivered Despawn finds the dot gone; a redelivered
        // directory head finds nobody departing) — so a shard crash between the emit and the
        // send would otherwise lose the removal FOREVER, and a lost removal is a permanent
        // frozen figure on every bystander's screen. The same argument that puts
        // `GhostFlow::Despawn` on the durable outbox (io/mod.rs `Durability`) puts this there.
        let bytes = postcard::to_allocvec(&ShardToGateway::EntityRemoved {
            realm_fence,
            entity,
            at,
        })
        .expect("closed wire enums serialize infallibly");
        outbox.0.push((
            gateway,
            MsgClass::Control,
            crate::io::bytes(bytes),
            Durability::Retained,
        ));
    }
}

/// A `SessionAttached` reply plus the gateway it routes to (the login grant-flip's egress).
pub(crate) struct AttachEgress {
    pub(crate) gateway: NodeId,
    pub(crate) reply: ShardToGateway,
}

/// The outcome of a grant-flip — three mutually-exclusive caller obligations (1d.1/1d.2c).
pub(crate) enum GrantFlip {
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
pub(crate) fn flip_grant(
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
