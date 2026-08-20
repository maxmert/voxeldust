//! THE TWO PLANES AND THE EGRESS: where a frame is decided to go, and the only primitives that
//! move one.
//!
//! Owns: the WRITE plane (one atomic route load plus one counter on the hot input path) and the
//! READ plane (one table load plus a per-sub fence compare on the fan), the SOLE route-mutation and
//! subscription-publication primitives every control path must go through, the node-class predicate
//! that decides whether a peer may speak at all, and the small push helpers that encode one message
//! onto the outbox.
//!
//! Does NOT own: any address. Everything routes by in-frame `SessionId` and `Fence`, never by
//! source address — which is what lets a shard transfer be a server-side route swap the client's
//! transport never observes.

use super::{
    GatewayConfig, GatewaySessions, GatewayStats, RouteSnapshot, SeqCut, Session, SessionHot,
    SubEntry, SubRecord, SubTable,
};
use std::collections::BTreeMap;
use std::sync::Arc;
use std::sync::atomic::Ordering;
use vd_core::{Fence, NodeId};
use vd_sim::io::MsgClass;
use vd_sim::runtime::OutboundBox;
use vd_wire::channels::{ServerControlMsg, SubId};
use vd_wire::intershard::InterShardFlow;
use vd_wire::seams::directory::DirectoryOp;
use vd_wire::seams::transfer_control::TransferControlAck;
use vd_wire::session_flow::{GatewayToShard, peek_input_seq, retag_snapshot_sub};

/// RLM 5f-3d — THE ONE node-class dispatch predicate for "is this peer a shard": the FROZEN config roster
/// UNION the RUNTIME set of demand-spawned shards (a session's home shard, and — RLM 5f-4 — a transfer's
/// crossing DEST). A dynamically spawned shard's `NodeId` is minted at spawn time — it can never be in the
/// boot-time config — so without the runtime half its `SessionAttached`, its `SubscriptionReady` and its
/// `Snapshot`/`RealmSnapshot` frames would all fall through to the client branch and be counted
/// `undecodable` (the render authority never re-points ⇒ a BLIND player). Bitwise `|` so
/// neither membership arm is a short-circuit-uncoverable region (HR5); when unarmed `dynamic_shards` is
/// always empty, so the result is byte-identical to the pre-5f-3d config-only test.
#[must_use]
pub(crate) fn is_routable_shard(
    config: &GatewayConfig,
    sessions: &GatewaySessions,
    from: NodeId,
) -> bool {
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
pub(crate) fn on_shard_roster(
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

pub(crate) fn refuse_unknown_sender(
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
pub(crate) fn session_target(session: &Session, config: &GatewayConfig) -> NodeId {
    session.home_shard.unwrap_or(config.shard)
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

pub(crate) fn push_control(outbox: &mut OutboundBox, to: NodeId, msg: &ServerControlMsg) {
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

pub(crate) fn push_to_shard(
    outbox: &mut OutboundBox,
    to: NodeId,
    class: MsgClass,
    msg: &GatewayToShard,
) {
    let bytes = postcard::to_allocvec(msg).expect("closed wire enums serialize infallibly");
    outbox.0.push((
        to,
        class,
        vd_sim::io::bytes(bytes),
        vd_sim::io::Durability::Ephemeral,
    ));
}

pub(crate) fn push_directory(outbox: &mut OutboundBox, to: NodeId, op: DirectoryOp) {
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
pub(crate) fn reply_ack(outbox: &mut OutboundBox, orchestrator: NodeId, ack: TransferControlAck) {
    outbox.push_flow(orchestrator, MsgClass::Saga, &InterShardFlow::SagaAck(ack));
}

/// THE sole route-mutation primitive: the one `route.store` for every transfer/attach
/// write. `store_cut` (carry authority+fence, swap cut), `store_commit` (move authority→dest,
/// carry fence, clear cut), and the attach path (fresh fence) all funnel here — so a future
/// `RouteSnapshot` field can never silently drop on any of them: the exhaustive no-`..rest`
/// struct literal lives in EXACTLY one place, and a 4th field is a single compile-fix every
/// caller inherits. (Login route BIRTH stays a direct `ArcSwap::from_pointee` — not a store.)
pub(crate) fn store_route(hot: &SessionHot, authority: NodeId, fence: Fence, cut: Option<SeqCut>) {
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
pub(crate) fn publish_subs(hot: &SessionHot, subs: &BTreeMap<NodeId, SubRecord>) {
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
pub(crate) fn store_cut(hot: &SessionHot, cut: Option<SeqCut>) {
    let current = hot.route.load();
    store_route(hot, current.authority, current.fence, cut); // CARRY authority + fence
}

/// `CommitAuthority`: MOVE authority → `dest`, CARRY the realm fence UNCHANGED, CLEAR the
/// cut — ONE atomic `route.store` (the SPIKE-2a one-publish guarantee). The fence is CARRIED,
/// NOT advanced to the per-Entity CAS `new_fence` (R-FENCE / DEFERRED D-25 — see `apply_commit`).
pub(crate) fn store_commit(hot: &SessionHot, dest: NodeId) {
    let current = hot.route.load();
    store_route(hot, dest, current.fence, None); // MOVE authority→dest, CARRY fence, CLEAR cut
}
