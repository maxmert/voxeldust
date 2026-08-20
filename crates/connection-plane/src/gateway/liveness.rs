//! LIVENESS: the self-fence, the retries, and the bootstrap that must end.
//!
//! Owns: the PROACTIVE session self-fence (a gateway that has lost contact with the directory stops
//! believing its own leases before anybody else may be given them), the heartbeat and re-check
//! cadences, the bounded pre-Active home bootstrap with its LOUD expiry, and the per-tick re-send of
//! everything still unanswered.
//!
//! Does NOT own: the answers it waits for. Every retry here is idempotent because the transport is
//! at-least-once and the reply may already be in flight; nothing in this module may have an effect
//! that a second copy of itself would double.

use super::{
    GatewayConfig, GatewaySessions, GatewayStats, SessionPhase, demand_for_home, push_control,
    push_directory, push_to_shard, session_target,
};
use bevy_ecs::prelude::{Res, ResMut};
use vd_core::{SessionId, TickId};
use vd_sim::io::MsgClass;
use vd_sim::runtime::{ClockSample, NodeIdentity, OutboundBox};
use vd_wire::channels::ServerControlMsg;
use vd_wire::intershard::InterShardFlow;
use vd_wire::seams::directory::{AuthorityRef, DirectoryKey, DirectoryOp};
use vd_wire::session_flow::GatewayToShard;

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
pub(crate) fn self_fence_lapsed_sessions(
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
pub(crate) fn home_redrive_due(now: TickId, since: TickId, interval: u64) -> bool {
    let elapsed = now.0.saturating_sub(since.0);
    let cadence = interval.max(1);
    (elapsed != 0) & elapsed.is_multiple_of(cadence)
}

/// RLM 5f-3d — has the BOUNDED pre-Active bootstrap window closed (CRITIQUE-1)? Strictly `>` so the
/// deadline tick itself is still inside the window (the hold is generous at its own edge). PURE.
#[must_use]
pub(crate) fn home_bootstrap_expired(now: TickId, deadline: TickId) -> bool {
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
pub(crate) fn attach_retry_due(anchor: Option<TickId>, now: TickId, cadence: u64) -> bool {
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
pub(crate) fn expire_home_bootstrap(
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
pub(crate) fn renew_and_recheck_sessions(
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

/// Per-tick retry driver: pending directory grants and shard attaches are
/// re-sent until answered (all idempotent — at-least-once over a lossy fabric).
///
/// RLM 5f-3d adds the DYNAMIC-HOME re-drive — the only producer here that is neither per-tick nor
/// per-session: it iterates the per-REALM [`GatewaySessions::home_bootstraps`] index on a BACKED-OFF cadence.
pub(crate) fn drive_pending_sessions(
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
