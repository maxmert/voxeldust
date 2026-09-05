//! THE WINDOW LANE, gateway side: which windows to hold open, what may be admitted through them,
//! and the fold that becomes a client's scene.
//!
//! Owns: the per-tick derivation of the wanted window set from the Active sessions' own lineages,
//! the keep-alive that re-asserts them, the admission rules every arriving statement must pass
//! (sender is the head, subject is on the author's own attested roster, newest-wins), the park-and-
//! drain for statements that raced their author's first roster, and the composer pass.
//!
//! Does NOT own: the world. The gateway composes from statements REALMS authored about themselves
//! and about their own direct children; it never regenerates a placement, never opens a sealed
//! relay, and never invents a row for a realm that is not running (SL3).

use super::{
    GatewayConfig, GatewaySessions, GatewayStats, GatewayWindow, SessionPhase, SubState,
    push_control, push_directory, push_to_shard, session_target, window_keepalive_cadence,
    window_tuning,
};
use crate::window;
use bevy_ecs::prelude::{Res, ResMut};
use std::collections::{BTreeMap, BTreeSet};
use vd_core::UniverseTick;
use vd_core::pose::{FrameRef, RealmId};
use vd_core::{Fence, NodeId};
use vd_sim::io::MsgClass;
use vd_sim::runtime::{ClockSample, OutboundBox};
use vd_wire::channels::{
    CONSERVATIVE_DATAGRAM_BUDGET, RealmSnap, RealmSnapshotDatagram, SceneRow, ServerControlMsg,
    SubId, partition_realms,
};
use vd_wire::intershard::InteriorRelay;
use vd_wire::seams::directory::{DirectoryKey, DirectoryOp};
use vd_wire::session_flow::{
    BodyStmt, GatewayToShard, RelayedStatement, ShardToGateway, WindowId, WindowScope,
    open_relay_statements, window_body_admissible, window_sender_is_head,
};

/// One window the gateway currently WANTS open (the per-tick derivation's row).
#[derive(Debug, PartialEq, Eq)]
struct DesiredWindow {
    shard: NodeId,
    scope: WindowScope,
    author_realm: RealmId,
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
            // A frame always names its realm since S9 — the skip this used to need could not fire.
            let realm = record.frame.realm();
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
/// STATE THE GALAXY to every client that does not hold it (S11).
///
/// ★ ONE SKY, ONE AUTHOR. The gateway folded this at boot from the whole world; no shard states a sky
/// any more. A shard's sky was its OWN realms', so the sky a player received depended on which shard
/// they were subscribed to — a home system shard states one star, its own, and a player never draws
/// their own star because they are standing inside it. Measured on a dual cluster: the galaxy shard
/// held 3, the client held 1, and drew 0.
///
/// ★ AND IT ALSO BEATS. The liveness beat says "this sky is still current", so it must come from the
/// party that authors the sky. It is unconditional, because on a galaxy that never changes SILENCE is
/// the healthy case and is indistinguishable from a dead emitter without it.
fn emit_sky(
    config: &GatewayConfig,
    sessions: &mut GatewaySessions,
    stats: &mut GatewayStats,
    outbox: &mut OutboundBox,
) {
    if config.sky.is_empty() {
        return; // a gateway booted without a world states no sky, rather than a wrong one
    }
    // CUT ONCE, PER GENERATION — see `GatewaySessions::sky_cut` for what this cost every beat.
    if sessions.sky_cut.as_ref().map(|(g, _)| *g) != Some(config.sky_generation) {
        sessions.sky_cut = Some((
            config.sky_generation,
            vd_wire::channels::partition_stars(&config.sky, SKY_PART_BUDGET_BYTES),
        ));
    }
    // Split the borrow by FIELD so the cut can be read while the session table is walked mutably.
    let GatewaySessions {
        by_session,
        sky_cut,
        ..
    } = sessions;
    let parts = &sky_cut.as_ref().expect("the cut was just made").1;
    let total = parts.len() as u32;
    for session in by_session.values_mut() {
        // THE BEAT goes to everyone, held sky or not: it is the statement that nothing changed.
        push_control(
            outbox,
            session.client,
            &ServerControlMsg::SkyAlive {
                generation: config.sky_generation,
            },
        );
        stats.sky_alive_beats_sent += 1;
        if session.sky_held == Some(config.sky_generation) {
            stats.sky_parts_skipped += total as u64;
            continue; // this client holds this sky — the whole point of the exchange
        }
        // ★ PACED, NOT FIRE-HOSED (2026-08-29). This sent EVERY part on EVERY beat until the client
        // confirmed. MEASURED at S12's census: 233 220 rows = 10.72 MB in 1 309 parts, re-sent whole
        // on each beat, per client. A client cannot ingest that in one beat, so it never confirms, so
        // it is sent again — and the flood drowns the login handshake on the same reliable link. The
        // symptom is a client stuck at "authenticating" with the gateway logging a MUST-BE-0
        // contiguity alert thousands of times.
        //
        // The register predicted exactly this: "at 14.2 MB the one-time transfer needs parts,
        // acknowledgement and resumption, and none of that can be exercised until the census rises at
        // S12." The census has risen.
        //
        // So a beat carries a BOUNDED number of parts, and the client's own held-mark decides when to
        // stop. A part the client already holds costs nothing to skip; a part it lacks arrives on a
        // later beat. The transfer still completes — it simply does not shout the whole sky at a
        // socket that cannot drink it.
        let first = session.sky_parts_sent.min(total);
        let last = (first + SKY_PARTS_PER_BEAT).min(total);
        // ★ EACH PART IS SENT ONCE. NO WRAP (2026-08-29).
        //
        // This used to restart at part 0 after a full pass, as insurance against a lost part. That
        // insurance is now both unnecessary and harmful:
        //
        //  * UNNECESSARY — the reliable inbound path no longer discards anything. It waits for room
        //    and lets QUIC's flow control slow the sender, so a part put on the wire arrives.
        //  * HARMFUL — a client that has not finished assembling gets the whole catalogue again, and
        //    again, forever. That is 256 KB decoded on its main thread every beat with nothing to
        //    show for it, which is exactly the movement stutter the owner reported.
        //
        // A client that holds the sky says so and is skipped by the check above. A client that does
        // not is still receiving its first pass. Either way there is no reason to say it twice.
        session.sky_parts_sent = last;
        for (i, rows) in parts
            .iter()
            .enumerate()
            .skip(first as usize)
            .take((last - first) as usize)
        {
            push_control(
                outbox,
                session.client,
                &ServerControlMsg::StarCatalogue {
                    generation: config.sky_generation,
                    part: i as u32,
                    parts: total,
                    rows: rows.clone(),
                },
            );
            stats.star_catalogue_parts_sent += 1;
        }
    }
}

/// How large one part of the catalogue may be. Stated once, here, so a test can reason about it.
const SKY_PART_BUDGET_BYTES: usize = 8 * 1024;

/// How many catalogue parts one beat may carry to one client.
///
/// ★ A PACE, NOT A CAP — the whole sky still arrives, over as many beats as it takes. At 8 KB a part
/// this is 256 KB per beat per client, which a reliable link carries without starving the login
/// handshake that shares it. The unpaced version sent 10.72 MB per beat and starved exactly that.
const SKY_PARTS_PER_BEAT: u32 = 32;

pub(crate) fn drive_windows(
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
            // A frame always names its realm since S9 — the skip this used to need could not fire.
            let realm = record.frame.realm();
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
                // A FRESH WINDOW HOLDS NOTHING, so it is served the whole roster (S11).
                static_held: None,
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
    // ★ THE GALAXY CROSSES ONCE (S11; owner ruling 2026-08-27). The gateway holds the whole sky and
    // states it to a client that does not hold it — on the same beat the keep-alive uses, so it needs
    // no timer of its own and repairs itself if a part is lost.
    //
    // "ONCE" is enforced by the client's own statement, not by a memory here: a session that has
    // confirmed this generation is skipped, and a session that crosses to another star system is still
    // holding the same sky, so nothing crosses again. That is the whole saving.
    if vd_sim::directory::due_this_tick(window_keepalive_cadence(&config), clock.local_tick.0) {
        emit_sky(&config, &mut sessions, &mut stats, &mut outbox);
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
                    // ★ THE COUNTER THE KEEP-ALIVE COMPARES (S11). Stating what we hold is what stops
                    // the shard re-sending the whole static roster twice a second, for ever — 28.4 MB/s
                    // per window at the S12 census, on the reliable lane.
                    static_held: held.ingest.static_digest(),
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
pub(crate) fn compose_scenes(
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
pub(crate) fn compose_scenes_pass(
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
    // marker. Markers never expire (the floor: never zero drawn). The prune clock is each
    // window's OWN ring head (look_horizon.md §3.5 C6) — the same head the ring trims on —
    // never this gateway's clock, so a gateway a few ticks adrift cannot evict a stamp early.
    for held in sessions.windows.values_mut() {
        let (looks, relays) = held.ingest.prune_stale(&tuning);
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
    let mut sky_anchored = 0u64;
    let mut stamp_gap_max = 0u64;
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
        let (Some(origin_frame), Some(origin)) = (origin_frame, origin_frame.map(FrameRef::realm))
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
        // THE STAMP GAP, measured: how far each hop's newest level sits from the origin's.
        if let Some(origin_newest) = ingests.first().and_then(|i| i.newest()) {
            for hop in &ingests[1..] {
                let gap = hop
                    .newest()
                    .map_or(u64::MAX, |h| origin_newest.0.abs_diff(h.0));
                stamp_gap_max = stamp_gap_max.max(gap);
            }
        }
        let (prefix, t) = window::fresh_prefix(&ingests);
        // ★ SAY WHERE THE CHAIN STOPS (2026-09-02), once per keep-alive beat: a chain whose fresh
        // prefix is shorter than its length folds no sky, and every refusal counter reads zero.
        if prefix < ingests.len()
            && clock
                .local_tick
                .0
                .is_multiple_of(window_keepalive_cadence(config))
        {
            let stamps: Vec<(RealmId, Option<u64>, Option<u64>, usize)> = chain
                .hops
                .iter()
                .zip(ingests.iter())
                .map(|(h, i)| {
                    let ticks: Vec<u64> = i.ticks().map(|t| t.0).collect();
                    (
                        h.author,
                        ticks.first().copied(),
                        ticks.last().copied(),
                        ticks.len(),
                    )
                })
                .collect();
            tracing::warn!(
                prefix,
                hops = ingests.len(),
                ?stamps,
                "the chain's fresh prefix stops short — per hop: (author, oldest, newest, count)"
            );
        }
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
                        let per_session = compose_fresh(
                            origin,
                            origin_frame,
                            t,
                            &authors,
                            &ingests,
                            prefix,
                            config.sky_frame,
                        );
                        stats.window_fold_divergence += fold_divergence(shared, &per_session);
                        *verified = true;
                    }
                    shared.clone()
                } else {
                    let fold = compose_fresh(
                        origin,
                        origin_frame,
                        t,
                        &authors,
                        &ingests,
                        prefix,
                        config.sky_frame,
                    );
                    stats.window_folds += 1;
                    stats.window_composed_rows += fold.rows.len() as u64;
                    stats.window_relay_rows_composed += fold.relay_rows;
                    stats.window_relay_descent_refused += fold.relay_descent_refused;
                    stats.window_relay_stamp_missing += fold.relay_stamp_missing;
                    stats.window_relay_unrostered += fold.relay_unrostered;
                    stats.window_relay_stamp_skew_ticks = stats
                        .window_relay_stamp_skew_ticks
                        .max(fold.relay_stamp_skew_ticks);
                    stats.window_relay_depth_max =
                        stats.window_relay_depth_max.max(fold.relay_depth_max);
                    stats.window_instant_mismatch += fold.instant_refused;
                    stats.window_rotated_refused += fold.rotated_refused;
                    stats.window_far_rows += fold.far_rows;
                    stats.window_alien_rows += fold.alien_rows;
                    stats.window_hop_invalid += fold.hop_invalid;
                    stats.window_dedup_disagree += fold.dedup_disagree;
                    stats.window_dedup_max_dev_cells = stats
                        .window_dedup_max_dev_cells
                        .max(fold.dedup_max_dev_cells);
                    stats.window_full_chain_folds +=
                        u64::from((fold.fresh_levels == authors.len()) & (authors.len() >= 2));
                    memo.insert((origin, t.0), (fold.clone(), false));
                    fold
                }
            });
            let prev_t = session.shadow.last_t;
            let covers_lineage =
                !session.lineage.is_empty() & (chain.hops.len() == session.lineage.len());
            let hop_pending = window::hop_pending(&session.lineage, chain.hops.len(), &catalog);
            let report = session.shadow.advance_covering(
                origin,
                &authors,
                fold,
                &tuning,
                covers_lineage,
                hop_pending,
            );
            stats.window_compose_hold_ticks += report.holds;
            stats.window_hop_dead += report.dead_hops;
            stats.window_t_monotone_stalled += u64::from(report.stalled);
            stats.window_origin_swap_deferred += u64::from(report.swap_deferred);
            stats.window_origin_swap_forced += u64::from(report.swap_forced);
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
                let mut rows = window::scene_level_rows(
                    origin,
                    origin_frame,
                    t_level,
                    &authors,
                    &ingests,
                    &session.shadow,
                );
                stats.window_looks_carried +=
                    session
                        .look_shelf
                        .carry(&mut rows, t_level, tuning.hold_ttl_ticks);
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
                let mut rows = window::scene_level_rows(
                    origin,
                    origin_frame,
                    t_now,
                    &authors,
                    &ingests,
                    &session.shadow,
                );
                // THE LOOK SHELF (2026-09-04): a drawn row whose chain windows state no look yet
                // (the hand-over churn) keeps the last look this session emitted, for one hold.
                stats.window_looks_carried +=
                    session
                        .look_shelf
                        .carry(&mut rows, t_now, tuning.hold_ttl_ticks);
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
                // ★ THE SKY ANCHOR RIDES EVERY CHUNK OF THE TICK (owner ruling 2026-09-02 R1), and
                // a tick with an anchor and NO drawn row still ships one datagram: a pilot alone in
                // a hull with nothing else in range must still be told where the galaxy is.
                let sky_anchor = session.shadow.sky_anchor();
                sky_anchored += u64::from(sky_anchor.is_some());
                let chunks = if realms.is_empty() {
                    sky_anchor.map(|_| Vec::new()).into_iter().collect()
                } else {
                    partition_realms(&realms, CONSERVATIVE_DATAGRAM_BUDGET)
                };
                if !chunks.is_empty() {
                    session.realm_feed_frame_id += 1;
                    let frame_id = session.realm_feed_frame_id;
                    for chunk in chunks {
                        let datagram = RealmSnapshotDatagram {
                            sub: SubId(0),
                            frame_id,
                            source_tick: clock.local_tick,
                            universe_tick: t_now,
                            origin_epoch: epoch,
                            sky_anchor,
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
    stats.window_sky_anchored = sky_anchored;
    stats.window_chain_stamp_gap_max = stamp_gap_max;
}

/// The §2.14 divergence verdict, split out so BOTH arms are unit-drivable (HR5): 1 when a
/// shared fold does not equal its per-session recompute bit-for-bit (must never happen —
/// asserted 0 by the parity gate; nonzero would mean the fold is not a pure function of its
/// attested inputs), else 0.
pub(crate) fn fold_divergence(shared: &window::Composed, per_session: &window::Composed) -> u64 {
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
    sky_frame: Option<FrameRef>,
) -> window::Composed {
    let levels: Vec<&window::WindowLevel> = ingests[..prefix]
        .iter()
        .map(|i| {
            i.level_at(t)
                .expect("fresh_prefix only names ticks every prefix level retains")
        })
        .collect();
    let mut fold = window::compose_in(
        origin,
        origin_frame,
        t,
        authors,
        &levels,
        prefix,
        ingests,
        sky_frame,
    );
    if let Some(sky) = sky_frame {
        window::place_sky_anchor(&mut fold, origin_frame, sky);
    }
    fold
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
#[allow(clippy::too_many_arguments)]
pub(crate) fn on_window_relayed(
    from: NodeId,
    window_id: WindowId,
    child: RealmId,
    child_fence: Fence,
    statements: &[u8],
    interior: Vec<InteriorRelay>,
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
            "forged-sender window relay dropped (fail-closed attestation)"
        );
        return;
    }
    if !held.ingest.rosters(child) {
        // PARKED, not dropped (the same first-roster race as the direct markers — the relay is
        // forwarded send-on-change, once): counted, held newest-fence-wins, re-admitted through
        // the full admission the moment the author's level vouches the child. The interior
        // parks WITH its relay (slice 3) — same seal, same race, same drain.
        stats.window_relay_unvouched += 1;
        if held
            .parked_relays
            .get(&child)
            .is_none_or(|(f, _, _)| !child_fence.is_stale_against(*f))
        {
            held.parked_relays
                .insert(child, (child_fence, statements.to_vec(), interior));
        }
        return;
    }
    admit_relay(
        held,
        child,
        child_fence,
        statements,
        &interior,
        &window_tuning(config),
        stats,
    );
}

/// The ONE relay admission (direct + drained-parked), applied AFTER the sender + roster vouches:
/// the child's fence orders incarnations, the seal opens HERE (the first and only reader), and
/// every inner statement runs the existing predicates against the CHILD's identity. Since look
/// horizon slice 3 (owner-approved 2026-08-17 — look_horizon.md RULINGS + §2 ASK A) the relay
/// also carries the child's held GRANDCHILD batches, sealed the whole way: each is vouched
/// against the child's own attested roster, fence-ordered by the existing per-realm map, opened
/// HERE, and admits ONLY the author's own picture — see [`admit_interior_relay`].
#[allow(clippy::too_many_arguments)]
fn admit_relay(
    held: &mut GatewayWindow,
    child: RealmId,
    child_fence: Fence,
    statements: &[u8],
    interior: &[InteriorRelay],
    tuning: &window::WindowTuning,
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
                if held.ingest.ingest_relay_level(child, at, rows, tuning) {
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
    // THE SEALED INTERIOR FORWARD's admission (look_horizon.md §3.2, AFTER the child's own
    // batch: its freshly ingested Level IS the roster the grandchildren are vouched against).
    for entry in interior {
        admit_interior_relay(held, child, entry, stats);
    }
}

/// ONE forwarded grandchild batch's admission (look_horizon.md §3.2's three rules, monomorphic —
/// every refusal arm its own named unit, HR5):
/// 1. the named grandchild must appear in the RELAYING child's own attested roster
///    ([`window::WindowIngest::relay_child_roster`] — the existing vouch); otherwise refuse +
///    count `window_relay_interior_unvouched` — a VIOLATION counter, gates assert 0;
/// 2. its fence orders against the existing per-realm fence map
///    ([`window::WindowIngest::admit_relay_fence`], already keyed by realm id — it generalises
///    unchanged); a deposed grandchild incarnation counts `window_relay_stale`;
/// 3. from the opened batch, admit ONLY the author's own picture (a `SelfLook` whose subject IS
///    the author): its `Level` and its markers describe depth-3 subjects, for which no row can
///    exist — dropped + counted `window_relay_interior_filtered` (an EXPECTED-nonzero counter,
///    never asserted zero); a `SelfLook` about anyone else is a mis-authored body
///    (`window_misauthored_body`), same as every other lane.
///
/// An admitted picture lands in the ONE body store (the marker⇒look handover needs nothing new
/// downstream) and the subject joins the window's interior-admitted set — the third disjunct of
/// the bag's member test (§3.4.5: the forward gate was the middle realm's membership decision,
/// so arrival IS the verdict).
fn admit_interior_relay(
    held: &mut GatewayWindow,
    child: RealmId,
    entry: &InteriorRelay,
    stats: &mut GatewayStats,
) {
    if !held.ingest.relay_child_roster(child).contains(&entry.child) {
        stats.window_relay_interior_unvouched += 1;
        tracing::warn!(
            grandchild = %entry.child,
            %child,
            "interior relay naming an unrostered grandchild refused (fail-closed vouch)"
        );
        return;
    }
    if !held
        .ingest
        .admit_relay_fence(entry.child, entry.child_fence)
    {
        stats.window_relay_stale += 1;
        return;
    }
    let Ok(opened) = open_relay_statements(&entry.own) else {
        stats.window_relay_undecodable += 1;
        return;
    };
    for statement in opened {
        match statement {
            RelayedStatement::Body {
                subject,
                stmt: stmt @ BodyStmt::SelfLook { .. },
                authored_at,
            } if subject == entry.child => {
                if held.ingest.ingest_body(subject, &stmt, authored_at) {
                    held.ingest.admit_interior(subject, authored_at);
                    stats.window_relays_ingested += 1;
                } else {
                    stats.window_body_stale += 1;
                }
            }
            RelayedStatement::Body {
                stmt: BodyStmt::SelfLook { .. },
                ..
            } => {
                stats.window_misauthored_body += 1;
            }
            RelayedStatement::Level { .. } | RelayedStatement::Body { .. } => {
                stats.window_relay_interior_filtered += 1;
            }
        }
    }
}

/// One admitted window statement's payload, handed to the composer's ingest AFTER attestation —
/// the Slice-B shape of the one ingest rule (the Slice-A `Option<(RealmId, &BodyStmt)>` grew a
/// typed row per statement kind when the engine started consuming).
pub(crate) enum WindowRow {
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
    /// The author's STATIC roster (slice S10) — its direct children that do not move, stated once on
    /// the reliable lane instead of riding every tick's frame.
    StaticRows(Vec<vd_wire::channels::RealmSnap>),
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
pub(crate) fn on_window_row(
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
                drain_parked(held, &window_tuning(config), stats);
            }
            window::Ingested::BehindRing => stats.window_level_refused += 1,
        },
        WindowRow::Body {
            subject,
            stmt,
            authored_at,
        } => {
            if matches!(stmt, BodyStmt::Marker { .. }) && !held.ingest.rosters(subject) {
                // PARKED, not dropped (Slice C1): the send-on-change lane sends a body ONCE, so
                // a marker racing the author's first roster would otherwise stay invisible
                // forever. Counted as before; re-admitted through the SAME predicate the moment
                // either door lands (fail-closed meanwhile — nothing composes it while parked).
                //
                // ★ THE QUESTION IS "DOES THE ROSTER NAME THIS ONE", NOT "HAS ANY ROSTER ARRIVED"
                // (fixed 2026-08-28). This read `!held.ingest.confirmed()`, and `confirmed()` is
                // true as soon as EITHER door has spoken. A realm's children reach us by two:
                // movers ride the per-tick level, statics ride the reliable roster. The shard
                // sends them level -> markers -> static roster, so a STATIC child's marker always
                // arrives in the gap. The level had landed, so the gate said "I have a roster" and
                // judged the marker against a roster that structurally could not contain it. The
                // marker was dropped as mis-authored, and the roster lane never says the same
                // thing twice, so the realm was never drawn again.
                //
                // MEASURED, driving that exact order: after the level, `confirmed()` = true while
                // `rosters(star)` = false; the marker counted `window_misauthored_body = 1` with
                // zero parked; and after the static roster landed the held marker was still None.
                // The home star vanished from the drawn scene while all nine sibling planets drew
                // — `render_boxes_smoke` red on the drawn-vs-world set.
                //
                // `rosters` is the predicate this always wanted: it reads BOTH doors, and its own
                // comment warned that consulting the level alone "would refuse every static
                // child's marker as unvouched — silently, and only at real scale." It was written
                // for the admission and never wired to the park gate.
                //
                // THE TRADE, STATED: a marker whose subject is genuinely NOT a child now parks in
                // one slot instead of being dropped and counted. The park map is bounded at one
                // slot per subject, which is the same bound the pre-roster case has always had,
                // and nothing composes a parked statement.
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
        WindowRow::StaticRows(rows) => {
            held.ingest.ingest_static_rows(rows);
            stats.window_rows_ingested += 1;
            // A roster landing can un-park a marker that was waiting for one: the static lane is the
            // only roster a realm with no moving children ever states.
            drain_parked(held, &window_tuning(config), stats);
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
pub(crate) fn drain_parked(
    held: &mut GatewayWindow,
    tuning: &window::WindowTuning,
    stats: &mut GatewayStats,
) {
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
        let (child_fence, statements, interior) = held
            .parked_relays
            .remove(&child)
            .expect("keyed by the loop above");
        admit_relay(
            held,
            child,
            child_fence,
            &statements,
            &interior,
            tuning,
            stats,
        );
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
pub(crate) fn on_shard_realm_frame(
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
