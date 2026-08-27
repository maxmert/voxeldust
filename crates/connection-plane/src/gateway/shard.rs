//! WHAT A SHARD SAYS BACK: the attach/detach lifecycle and a demand shard's greeting.
//!
//! Owns: the shard-control reply arms — a session attached, detached, refused — and the consumption
//! of a demand-spawned shard's reactive presence, which is PURE OBSERVABILITY here: the reachability
//! effect happened in the mesh below the seam, and this lane only records that it did.
//!
//! Does NOT own: the shard's authority. Which node holds a realm is the directory's answer, learned
//! through `directory`; nothing here promotes a peer by having heard from it.

use super::{
    GatewayConfig, GatewaySessions, GatewayStats, SessionPhase, WindowRow, announce_own_entity,
    fan_entity_removed, fan_star_catalogue, lineage_apply, on_window_relayed, on_window_row,
    session_target, store_route,
};
use crate::window;
use vd_core::NodeId;
use vd_sim::runtime::{ClockSample, OutboundBox};
use vd_wire::intershard::{InterShardFlow, ShardPresence};
use vd_wire::session_flow::ShardToGateway;

/// RLM 5f RG-3 — consume a demand shard's reactive greeting. PURE OBSERVABILITY: the reachability effect
/// (the mesh learning the return connection) already happened below the app seam on this same reliable
/// frame, so this only decodes for a counter + a debug line and claims NOTHING (no `dynamic_shards` role —
/// a presence has no session and no lifetime). A body that is not a `ShardPresence` (any other
/// `InterShardFlow` arm, or undecodable bytes) counts `undecodable` exactly as the pre-RG-3 fallthrough did.
pub(crate) fn on_shard_presence(
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

/// Handle a shard control reply (attach/detach lifecycle), from shard `from`.
pub(crate) fn on_shard_control(
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
                // A frame always names its realm since S9, so the only condition left is whether the
                // lineage is empty — the `Some` arm this used to also test could not fail.
                if session.lineage.is_empty() {
                    session.lineage = vec![frame.realm()];
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
            {
                let realm = frame.realm();
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
        ShardToGateway::StarCatalogue {
            generation,
            part,
            parts,
            rows,
        } => {
            fan_star_catalogue(from, generation, part, parts, rows, sessions, stats, outbox);
        }
        ShardToGateway::WindowStaticRows { window, rows, .. } => {
            on_window_row(
                from,
                window,
                WindowRow::StaticRows(rows),
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
            interior,
            ..
        } => {
            on_window_relayed(
                from,
                window,
                child,
                child_fence,
                &statements,
                interior,
                config,
                sessions,
                stats,
            );
        }
    }
}
