//! THE DIRECTORY REPLIES: the session mint confirmed, and the realm heads the gateway routes on.
//!
//! Owns: the reply arms for the two obligations that ride this seam — the `Session` head that
//! confirms or denies a mint and re-confirms a live lease, and the `Realm` head that resolves where
//! a session's home is currently running.
//!
//! Does NOT own: the directory's decision. A reply is believed on its fence, never on its
//! plausibility; a stale or refused record leaves the gateway exactly where it was.

use super::{
    GatewayConfig, GatewaySessions, GatewayStats, SessionPhase, coord_lineage, demand_for_home,
    home_placement, on_home_realm_head, push_control, push_directory, push_to_shard,
    session_target,
};
use vd_core::TickId;
use vd_core::pose::RealmId;
use vd_core::realm_coord::RealmCoord;
use vd_sim::io::MsgClass;
use vd_sim::runtime::{ClockSample, NodeIdentity, OutboundBox};
use vd_wire::channels::ServerControlMsg;
use vd_wire::intershard::InterShardFlow;
use vd_wire::seams::directory::{AuthorityRef, DirectoryKey, DirectoryOp, DirectoryReply};
use vd_wire::session_flow::GatewayToShard;
use vd_wire::version::ProtoVersion;

/// Handle a directory reply. TWO gateway obligations ride this seam: the `Session` head confirms (or denies)
/// the mint and re-confirms an Active lease (D-3), and — RLM 5f-3d — the `Realm` head names the node owning
/// a pre-Active session's DYNAMIC HOME realm (5f-3c discarded this arm). Everything else (Entity/Ship heads,
/// CAS outcomes, clock samples) carries no gateway obligation.
pub(crate) fn on_directory_reply(
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
        // The ONE forest descend per login (5f-3d: every later re-drive reuses this coord). It
        // yields BOTH halves of the answer — which realm, and where inside it. The POSE half is
        // recorded in EVERY mode (T2's forced re-derivation: since the star became a body at
        // the home centre, an attach with no pose would drop the account inside the Star realm
        // — the static chain measured exactly that as a frozen unresolvable crossing); the
        // COORD half still gates the dynamic demand alone.
        let (coord, spawn) = home_placement(&config.seed_injector, session.account);
        session.spawn = Some(spawn);
        let home = if config.dynamic_home_mode(clock.synced) {
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
