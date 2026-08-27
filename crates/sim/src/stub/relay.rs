//! THE UP-RELAY AND THE INTEREST BYTE: the two things that cross a realm boundary that are not a
//! placement.
//!
//! Owns: the child-side ship of this realm's own sealed statements, the parent-side holder that
//! keeps the latest blob per live child UNOPENED, the forward gate that is also the membership
//! gate, and the one-byte assertion that something outside may be looking in.
//!
//! Does NOT own: the contents. The parent's entire lawful vocabulary here is forward-or-drop — it
//! never opens, merges, reads or re-states a child's statement, so a sealed parent structurally
//! cannot consult a value it never sees (HR1). No occupant pose is in any of it (SL2), and every
//! new datum on this path is an ASK before it is code (SL6).

use super::{
    ChildRealmNodes, OpenWindows, ParentRealmNode, StubConfig, StubStats, push_session_reply,
    retain_ttl_ticks, ttl_alive,
};
use crate::io::MsgClass;
use crate::runtime::{ClockSample, OutboundBox};
use bevy_ecs::prelude::Resource;
use std::collections::{BTreeMap, BTreeSet};
use vd_core::pose::RealmId;
use vd_core::realm_coord::RealmCoord;
use vd_core::{Fence, NodeId, TickId, UniverseTick};
use vd_wire::intershard::InterShardFlow;
use vd_wire::seams::directory::{DirectoryKey, DirectoryOp};
use vd_wire::session_flow::ShardToGateway;

/// THE Q2 RELAY MEMO (window lane Slice C1, mesh minor 17; owner-approved 2026-08-16 —
/// docs/design/owner_decisions_2026-08-15.md addendum + docs/design/window_lane.md §5 RULINGS):
/// the CHILD-side send-on-change baseline for the up-relay of this realm's verbatim self-authored
/// statements — the (parent, fingerprint-of-LEVEL-rows-and-bodies) pair last shipped. A fresh
/// ship happens when the authored level or the look/marker set changes (orbiting children ⇒
/// every tick, the rate of the lane this relay replaces; a static interior stays quiet), when
/// the parent resolves or moves (the relay-subscription moment), or on the AoI-cadence
/// re-assert. DEFAULT `None` ⇒ the first parent resolve ships.
#[derive(Resource, Debug, Default)]
pub struct RelayShip(pub(crate) Option<(NodeId, Vec<u8>)>);

/// THE Q2 RELAY HOLDER (PARENT side, same citation): the LATEST sealed statement blob per live
/// direct child, held UNOPENED — the parent's whole lawful vocabulary is forward-or-drop (no
/// store-merge, no read, no re-state; this crate deliberately never calls
/// `open_relay_statements`, so a parent structurally cannot consult a value it never sees).
/// Pruned on the derived retain TTL (2 cadences + 1 — owner law 3(a)); forwarded to every open
/// window's subscriber send-on-change (a fresh window is served everything currently held once).
#[derive(Resource, Debug, Default)]
pub struct RelayHeld(pub(crate) BTreeMap<RealmId, RelayHeldEntry>);

impl RelayHeld {
    /// The sealed OWN statement blob currently held for `child`, if any — a READ-ONLY view for
    /// harnesses and gates. The blob stays SEALED: this hands back the bytes, and only
    /// `vd_wire::session_flow::open_relay_statements` can look inside them. Production code in
    /// this crate deliberately never calls that — pinned by the G-STRUCTURAL-SEAL source scan,
    /// which is what makes forward-or-drop structural rather than a promise (the Q2 ruling,
    /// `docs/design/window_lane.md` §5 RULINGS; two hops since look_horizon.md §2 ASK A).
    #[must_use]
    pub fn statements_for(&self, child: RealmId) -> Option<Vec<u8>> {
        self.0.get(&child).map(|e| e.own.clone())
    }

    /// The held INTERIOR half for `child` (look horizon slice 3 — the grandchild batches the
    /// child forwarded sealed), a READ-ONLY view for harnesses and gates. Same seal discipline
    /// as [`RelayHeld::statements_for`]: bytes out, never values.
    #[must_use]
    pub fn interior_for(&self, child: RealmId) -> Option<Vec<vd_wire::intershard::InteriorRelay>> {
        self.0.get(&child).map(|e| e.interior.clone())
    }
}

/// One held relay, SPLIT (look_horizon.md §2 ASK A / §6 slice 3): the child's OWN half and the
/// interior half (its forwarded grandchild batches), beside freshness (the TTL clock) and the
/// child's own fence (zombie ordering — a deposed incarnation's batch is refused). THE FORWARD
/// READS ONLY THE OWN HALF for this realm's own up-relay statement (`build_relay_interior`);
/// the interior half is forwarded SEALED to the gateway and never rides further up — there is
/// no field it could ride in (`InteriorRelay` carries no `interior`). `digest` is the §5.4
/// send-on-change baseline over (fence, own, interior), computed ONCE at receive so a
/// grandchild's picture change is never invisible to send-on-change even when the child's own
/// statements are unchanged.
#[derive(Debug, PartialEq)]
pub(crate) struct RelayHeldEntry {
    pub(crate) seen: TickId,
    pub(crate) fence: Fence,
    pub(crate) own: Vec<u8>,
    pub(crate) interior: Vec<vd_wire::intershard::InteriorRelay>,
    pub(crate) digest: u64,
}

/// THE REALM-SIDE SHARED IN-BAND VERDICT (look_horizon.md §3.4.5 — the forward gate IS the
/// membership gate): the union, over every observer this realm's AoI fold ran, of the direct
/// children inside that observer's band this tick. Written by `aoi_decide` (the ONE existing
/// fold — never a second AoI derivation, HR3; cleared at the zero-observer return), read by the
/// up-relay ship: a held child's sealed batch joins the `interior` forward ONLY while that child
/// is in this verdict, so the arrival of a grandchild's picture at the grandparent's gateway is
/// already conditioned on the middle realm's own membership decision — one fewer boundary
/// crossing than shipping any verdict upward. One tick of staleness (the emitter runs before
/// the fold on the schedule) is absorbed by the band's own hysteresis and grace.
#[derive(Resource, Debug, Default)]
pub struct InBandVerdict(pub(crate) BTreeSet<RealmId>);

/// THE INTEREST BYTE's holder (look horizon slice 4; Q1 APPROVED, owner-approved 2026-08-17 —
/// docs/design/look_horizon.md §2 ASK B): the latest lawfully admitted parent assertion about
/// whether something outside may be looking in. One byte, two values, nothing else — held under
/// the derived retain TTL and DECAYING to "nobody is watching" on silence (the safe direction:
/// the entry is pruned whole, and with it the down-proxy observer it powers). The `0` value is
/// stored rather than clearing, so the (fence, at) ordering survives an explicit switch-off and
/// a deposed incarnation's later `1` still refuses. DEFAULT `None` ⇒ byte-identical everywhere
/// the lane never fires.
#[derive(Resource, Debug, Default)]
pub struct InterestHeld(pub(crate) Option<InterestEntry>);

/// One held interest assertion: the sender's fence + tick (zombie/freshness ordering), the
/// local-tick TTL base, and the byte itself.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct InterestEntry {
    pub(crate) fence: Fence,
    pub(crate) at: UniverseTick,
    pub(crate) seen: TickId,
    /// How far away the nearest outside looker is, or `None` for nobody. See the wire field for why
    /// this is a distance and deliberately not a direction.
    pub(crate) from_m: Option<f64>,
}

/// THE INTEREST EMISSION's per-child hysteresis latch (look horizon slice 4, the emitter half):
/// which direct children this realm currently asserts interest INTO — present = inside the
/// child's interior band (spin-up at the child's interior reach, release at reach + the derived
/// lead). The falling edge ships one explicit `0`; silence is the receiver's backstop.
///
/// ★ KEYED BY THE CHILD'S OWN `RealmId` (slice S10), like the AoI membership ledger it mirrors — both
/// were keyed by the full lineage path, which allocated a list per lookup for a question about identity.
/// This latch only ever holds one shard's own direct children, so an id cannot collide. Pruned to the
/// live child roster each pass.
#[derive(Resource, Debug, Default)]
pub struct InterestEmitLatch(pub(crate) BTreeSet<RealmId>);

/// THE Q2 RELAY FORWARD (Slice C1 — the parent half; owner-approved 2026-08-16, cited on
/// [`RelayHeld`]): prune the holder on the derived retain TTL, then forward every held sealed
/// batch to every open window's subscriber, send-on-change per (window, child) — a fresh window
/// is served everything currently held once. The bytes go out EXACTLY as received
/// (forward-or-drop): same seal, same child fence; this shard adds only its own envelope fence.
/// ReDriven/reliable — the session-reply lane, mirroring `WindowBody`'s reasoning (a lost relay
/// is an invisible realm at exactly the no-flicker moment G-HANDOVER measures).
pub(crate) fn emit_window_relays(
    config: &StubConfig,
    clock: &ClockSample,
    realm_fence: Fence,
    held: &mut RelayHeld,
    windows: &mut OpenWindows,
    stats: &mut StubStats,
    outbox: &mut OutboundBox,
) {
    let ttl = retain_ttl_ticks(config);
    held.0
        .retain(|_, e| ttl_alive(e.seen, clock.local_tick, ttl));
    for ((gateway, window), w) in &mut windows.0 {
        // A held entry that vanished (TTL / teardown) clears its baseline so a re-held child
        // re-ships to this window rather than being remembered as already-served.
        w.sent_relays.retain(|c, _| held.0.contains_key(c));
        for (child, entry) in &held.0 {
            if w.sent_relays.get(child) == Some(&entry.digest) {
                continue; // unchanged — send-on-change holds its tongue
            }
            push_session_reply(
                outbox,
                *gateway,
                &ShardToGateway::WindowRelayed {
                    realm_fence,
                    window: *window,
                    child: *child,
                    child_fence: entry.fence,
                    statements: entry.own.clone(),
                    // The interior half, forwarded SEALED and verbatim (look horizon slice 3
                    // — G-VERBATIM pins the byte-identity). The gateway is the first opener.
                    interior: entry.interior.clone(),
                },
            );
            w.sent_relays.insert(*child, entry.digest);
            stats.window_relays_forwarded += 1;
        }
    }
}

/// The 8-byte send-on-change baseline digest over a held relay's (fence, own sealed statements,
/// interior batches) — look_horizon.md §5.4's improvement to the per-(window, child) blob
/// compare, with the slice-3 interior JOINED as the third component: a grandchild's picture
/// change with the child's own statements unchanged re-keys the baseline, so it can never be
/// invisible to send-on-change (the silent staleness bug §5.4 names). FNV-1a 64: deterministic,
/// dependency-free, branchless (one fold expression over the concatenated parts; each interior
/// entry contributes its child id via the closed postcard codec, its fence bytes and its sealed
/// bytes). Computed ONCE at receive (stored on [`RelayHeldEntry::digest`]), never per tick. A
/// (astronomically unlikely) collision costs one deferred re-send healed by the keep-alive
/// re-assert (`reset_baselines`), never a wrong byte on the wire.
pub(crate) fn relay_entry_digest(
    fence: Fence,
    own: &[u8],
    interior: &[vd_wire::intershard::InteriorRelay],
) -> u64 {
    let h = fnv1a_from(FNV_OFFSET, &fence.0.to_le_bytes());
    let h = fnv1a_from(h, own);
    interior.iter().fold(h, |h, e| {
        let h = fnv1a_from(
            h,
            &postcard::to_allocvec(&e.child).expect("closed wire enums serialize infallibly"),
        );
        let h = fnv1a_from(h, &e.child_fence.0.to_le_bytes());
        fnv1a_from(h, &e.own)
    })
}

// ★ THE FOLD ITSELF MOVED TO THE CORE (slice S11). It existed here AND in the saved-data label's
// generation — two copies of one arithmetic, which is two chances for them to stop being the same
// arithmetic. `vd_core::digest` is now the only definition in the tree.
use vd_core::digest::{FNV_OFFSET, fnv1a as fnv1a_from};

/// A send-on-change baseline for ONE statement's bytes (slice S10).
///
/// ★ WHY THE BODY BASELINE IS A DIGEST NOW. It used to keep the WHOLE BAG per subject — one heap
/// vector per direct child, held for the life of the window, compared byte-by-byte every tick and
/// cloned again on every change. At the target census that is 150,000 vectors held per subscriber,
/// 150,000 comparisons per tick, and 150,000 frees plus 150,000 allocations on every keep-alive beat,
/// because the beat clears the baselines by design.
///
/// The relay lane one field below already solved this exact problem this exact way, and its own note
/// carries the safety argument verbatim: a collision "costs one deferred re-send healed by the
/// keep-alive re-assert, never a wrong byte on the wire". The keep-alive that made the old form
/// expensive is the same beat that heals the new form's one failure mode.
pub(crate) fn statement_digest(bytes: &[u8]) -> u64 {
    fnv1a_from(FNV_OFFSET, bytes)
}

/// THE Q2 RELAY RECEIVE (Slice C1 — the parent's ingress; owner-approved 2026-08-16, cited on
/// [`RelayHeld`]): hold a live direct child's sealed statements UNOPENED for the per-tick
/// forwarder. Mis-route and attestation are the same admission every up-lane uses (fail closed,
/// counted); the child's own fence orders incarnations — a deposed zombie's batch is refused.
/// Monomorphic (all branching HERE, HR5).
pub(crate) fn on_window_relay(
    wr: vd_wire::intershard::WindowRelay,
    from: NodeId,
    config: &StubConfig,
    clock: &ClockSample,
    child_nodes: &ChildRealmNodes,
    held: &mut RelayHeld,
    stats: &mut StubStats,
) {
    let parent_ok = wr.child.parent().map(|p| p.lowered()) == Some(config.realm);
    if !parent_ok {
        stats.window_relay_misrouted += 1;
        return;
    }
    let child = wr.child.lowered();
    // Lane attestation (findings 0/43, up half): same admission compare as the bit and the rows;
    // fail closed, count only (the bit's refusal arms the head re-read).
    if child_nodes.0.get(&child) != Some(&from) {
        stats.window_relay_unattested += 1;
        return;
    }
    if held
        .0
        .get(&child)
        .is_some_and(|e| wr.realm_fence.is_stale_against(e.fence))
    {
        stats.window_relay_stale += 1;
        return;
    }
    stats.window_relays_received += 1;
    let digest = relay_entry_digest(wr.realm_fence, &wr.own, &wr.interior);
    held.0.insert(
        child,
        RelayHeldEntry {
            seen: clock.local_tick,
            fence: wr.realm_fence,
            own: wr.own,
            interior: wr.interior,
            digest,
        },
    );
}

/// THE INTEREST BYTE's receive (look horizon slice 4; Q1 APPROVED, owner-approved 2026-08-17 —
/// docs/design/look_horizon.md §2 ASK B, fail-closed shape): mis-route, unattested sender and a
/// deposed incarnation's stale byte are refused + counted, MIRRORING [`retain_child_live`]'s
/// admission (the unattested arm arms the same lazy head re-read — here for the PARENT, the
/// re-home backstop); a value outside {0, 1} is refused apart ("nothing else is lawful"). A
/// lawful byte is held whole — value included — under the derived retain TTL; silence decays it
/// to "nobody is watching" (the prune in [`aoi_decide`]). Monomorphic: ALL branching here
/// (HR5).
#[allow(clippy::too_many_arguments)]
pub(crate) fn on_realm_interest(
    ri: vd_wire::intershard::RealmInterest,
    from: NodeId,
    config: &StubConfig,
    clock: &ClockSample,
    parent_node: &ParentRealmNode,
    held: &mut InterestHeld,
    stats: &mut StubStats,
    outbox: &mut OutboundBox,
) {
    if ri.child.lowered() != config.realm {
        stats.realm_interest_misrouted += 1;
        return;
    }
    if parent_node.0 != Some(from) {
        stats.realm_interest_unattested += 1;
        if let Some(parent) = config.own_coord.parent() {
            outbox.push_flow(
                config.orchestrator,
                MsgClass::Saga,
                &InterShardFlow::Directory(DirectoryOp::HeadRead {
                    key: DirectoryKey::Realm(parent.lowered()),
                }),
            );
        }
        return;
    }
    // ★ THE LAWFUL-VALUE GUARD, unchanged in spirit (S10): it refused a byte above 1; it now refuses a
    // distance that is not a distance. `is_some_and` so the `None` case — nobody looking — is lawful.
    if ri
        .look_inside_from_m
        .is_some_and(|d| !d.is_finite() | (d < 0.0))
    {
        stats.realm_interest_unlawful += 1;
        return;
    }
    let fresh = held
        .0
        .is_none_or(|h| (ri.parent_fence, ri.at) >= (h.fence, h.at));
    if !fresh {
        stats.realm_interest_stale += 1;
        return;
    }
    stats.realm_interest_received += 1;
    held.0 = Some(InterestEntry {
        fence: ri.parent_fence,
        at: ri.at,
        seen: clock.local_tick,
        from_m: ri.look_inside_from_m,
    });
}

/// THE INTERIOR SELECTION (look horizon slice 3 — §3.2's forward rule + §3.4.5's forward gate,
/// monomorphic so both filter arms are covered by named tests): each held child's OWN sealed
/// half, byte-for-byte (G-VERBATIM pins the identity across both hops), for children inside
/// this realm's own in-band verdict. Reads `own` and NEVER `held[child].interior` — the held
/// interior halves stop here structurally, because [`vd_wire::intershard::InteriorRelay`] has
/// no field to put them in.
pub(crate) fn build_relay_interior(
    held: &RelayHeld,
    verdict: &BTreeSet<RealmId>,
) -> Vec<vd_wire::intershard::InteriorRelay> {
    held.0
        .iter()
        .filter(|(child, _)| verdict.contains(*child))
        .map(|(child, e)| vd_wire::intershard::InteriorRelay {
            child: *child,
            child_fence: e.fence,
            own: e.own.clone(),
        })
        .collect()
}

/// THE INTEREST EMISSION for one direct child (look horizon slice 4; Q1 APPROVED, owner
/// 2026-08-17 — §2 ASK B; monomorphic, ALL branching here, both arms of every predicate driven
/// by named tests). The COARSE decision of §3.4.2: judged by the realm HOLDING the observer,
/// from its own authored placement for the child and the child's boot-derived interior band —
/// spin-up at the child's interior reach (`444.104489631` m for a star system on THE world),
/// release at reach + the derived lead (`469.104489631` m), both bracketing the child's shell
/// so interiors wake strictly before any crossing. `min_dist` is the least distance from any
/// OCCUPANCY-derived observer (the structural cascade cap's source half: a synthetic
/// interest-derived observer contributes `INFINITY`, so an interest-only realm emits nothing
/// downward — an observer's realm wakes its children's interiors and STOPS).
///
/// Emission shape: `1` on the RISING edge immediately (the transition itself — the same
/// finding-41 doctrine the occupancy bit's adopt edge rides) and on every AoI beat while the
/// band holds (the re-assert the receiver's TTL is sized against); ONE explicit `0` on the
/// falling edge (silence remains the receiver's backstop). Routed direct to the child's
/// attested head — no route resolved yet ⇒ nothing sent, the latch untouched, the next beat
/// retries (fail-closed, never through the orchestrator).
#[allow(clippy::too_many_arguments)]
pub(crate) fn emit_realm_interest(
    band: &vd_core::geometry::AoiConfig,
    realm: RealmId,
    child_coord: &RealmCoord,
    route: Option<&NodeId>,
    min_dist: f64,
    bit_due: bool,
    latch: &mut BTreeSet<RealmId>,
    realm_fence: Fence,
    tick: UniverseTick,
    outbox: &mut OutboundBox,
    stats: &mut StubStats,
) {
    if band.spin_up_r_m() <= 0.0 {
        return; // a leaf (or an inert world): no interior, no interest — the byte never exists
    }
    let was = latch.contains(&realm);
    let now = band.in_range(was, min_dist);
    let Some(&node) = route else {
        return; // head not resolved yet — the eager cadence read lands it; the next beat sends
    };
    let rising = now & !was;
    let falling = !now & was;
    if (now & (bit_due | rising)) | falling {
        outbox.push_flow(
            node,
            MsgClass::Saga,
            &InterShardFlow::RealmInterest(vd_wire::intershard::RealmInterest {
                child: child_coord.clone(),
                parent_fence: realm_fence,
                at: tick,
                // ★ THE DISTANCE ITSELF (S10). `min_dist` is the least distance from any occupancy
                // observer to this child — the number this function ALREADY computed the band test
                // from and then threw away. Sending it costs nothing new to derive and is what lets
                // the child wake a few of its own children instead of all of them.
                look_inside_from_m: now.then_some(min_dist),
            }),
        );
        stats.realm_interest_sent += 1;
    }
    if now {
        latch.insert(realm);
    } else {
        latch.remove(&realm);
    }
}
