//! THE WORLD BUILD AND THE INBOUND DISPATCH: where the gateway's systems are installed, and where
//! each delivered message is handed to one.
//!
//! Owns: the one registration a gateway binary calls, the schedule order, and the per-tick drain
//! that refuses a stranger's frame before any lane sees it.
//!
//! Does NOT own: any per-kind fork. The same systems are installed everywhere; configuration, not
//! code selection, decides what a deployment does.

use super::{
    GatewayConfig, GatewaySessions, GatewayStats, SessionMint, compose_scenes,
    drive_pending_sessions, drive_windows, expire_home_bootstrap, is_routable_shard,
    on_client_control, on_client_input, on_directory_reply, on_shard_control, on_shard_frame,
    on_shard_presence, on_shard_realm_frame, on_shard_roster, on_transfer_control,
    recompute_delivery_watermarks, refuse_unknown_sender, renew_and_recheck_sessions,
    self_fence_lapsed_sessions,
};
use bevy_ecs::prelude::{IntoScheduleConfigs, Res, ResMut, Schedule, World};
use vd_core::rng::SplitMix64;
use vd_sim::io::{Inbound, MsgClass};
use vd_sim::runtime::{ClockSample, InboundBox, NodeIdentity, OutboundBox};
use vd_wire::intershard::InterShardFlow;

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
                    // ★ THE SKY FOLLOWS THE HULL (the ruler switch, slice 5): a realm on some session's
                    // chain moved house; the chains aboard it take the new ancestry.
                    Ok(InterShardFlow::ExteriorMoved(moved)) => {
                        super::home::on_exterior_moved(moved, &mut sessions, &mut stats);
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
