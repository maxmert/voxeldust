//! THE CLIENT'S TWO DIRECTIONS: what it says, and what it is told.
//!
//! Owns: the control conversation (hello, resume, subscribe, close) with the gateway as the SOLE
//! ticket validator, the input datagram's hot routing decision plus the cold cut-marker observation
//! that follows it, and the fan that turns one shard frame into one tagged frame per subscriber at
//! that subscriber's own sub id and accepted fence.
//!
//! Does NOT own: the credential's meaning (`crate::tickets`) or the route it uses (`routing`). A
//! shard never sees a credential, and the fan never invents a fence.

use super::{
    GatewayConfig, GatewaySessions, GatewayStats, InputRouting, RouteSnapshot, Session, SessionHot,
    SessionMint, SessionPhase, SubTable, push_control, push_directory, push_to_shard, route_input,
    session_target,
};
use crate::tickets;
use crate::window;
use arc_swap::ArcSwap;
use std::collections::BTreeMap;
use std::sync::Arc;
use std::sync::atomic::AtomicU64;
use vd_core::{EntityId, Fence, NodeId, SessionId, TickId};
use vd_sim::io::MsgClass;
use vd_sim::runtime::{NodeIdentity, OutboundBox};
use vd_wire::channels::{ClientControlMsg, ServerControlMsg, SubId};
use vd_wire::seams::directory::{AuthorityRef, DirectoryKey, DirectoryOp};
use vd_wire::session_flow::{
    GatewayToShard, ShardToGateway, peek_snapshot_frame_id, retag_snapshot_sub,
};
use vd_wire::version::ProtoVersion;

/// Announce "this entity is YOUR avatar" to one client — the DUAL-signal that carries the
/// pure-renderer migration (S4). It ALWAYS pushes `AuthorityChanged{entity, sub}` (the sub-keyed
/// signal an OLD, node-AWARE minor<2 client re-points render authority with) AND — to a peer that
/// negotiated minor >= 2 — the node-AGNOSTIC `OwnEntity{entity}` (which names ONLY the entity, no
/// sub / owning node). A pure-renderer client reads `OwnEntity` and IGNORES `AuthorityChanged`; an
/// old client reads `AuthorityChanged` and ignores the (withheld-anyway) `OwnEntity`. The gateway
/// emits BOTH so a rolling fleet of both client versions renders the same avatar without a flag day.
pub(crate) fn announce_own_entity(
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

/// Handle one client control message. The gateway is the SOLE ticket validator;
/// shards never see a credential.
#[allow(clippy::too_many_arguments)]
pub(crate) fn on_client_control(
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
                    // A fresh session holds no sky until it says otherwise (S11).
                    sky_held: None,
                    sky_parts_sent: 0,
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
        // THE RECEIVER STATES WHAT IT HOLDS (S11). Recorded against the session, because the session
        // is the thing that holds a sky — the gateway does not, and the shard must not try to.
        //
        // A session that never says this is treated as holding nothing, which is the safe default: it
        // is served the sky it may already have, rather than being denied one it does not.
        ClientControlMsg::SkyHeld { generation } => {
            if let Some(entry) = sessions
                .by_client
                .get(&client)
                .copied()
                .and_then(|session| sessions.by_session.get_mut(&session))
            {
                entry.sky_held = Some(generation);
                stats.sky_held_stated += 1;
            }
        }
        ClientControlMsg::CutEmitted { .. } | ClientControlMsg::Pong { .. } => {}
    }
}

/// Route one client input datagram (HOT decision + bookkeeping), then COLD-observe the
/// transfer cut marker — strictly OFF the SPIKE-2a hot path (`route_input` is unchanged).
pub(crate) fn on_client_input(
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

/// Fan one shard's `EntityRemoved` to that shard's subscribers as [`ServerControlMsg::Event`] —
/// the reliable per-entity eviction (the remove message, proto_minor 14). The same subscriber walk
/// as the frame fan (`subscribers_of`, Active only, per-shard accepted fence), but TYPED and
/// per-session minor-gated: a peer that negotiated below 14 is withheld the variant
/// (sender-gates-variants) and keeps the frozen-figure gap this message closes. A removal stale
/// against a session's accepted fence is refused for THAT session and counted apart
/// (`stale_removals_dropped`) — a demoted old owner must not evict what the live owner streams.
pub(crate) fn fan_entity_removed(
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
pub(crate) fn on_shard_frame(
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
