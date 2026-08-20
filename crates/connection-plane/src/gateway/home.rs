//! THE DYNAMIC HOME: where an authenticated login is put, derived on the server.
//!
//! Owns: the descent of an account's stored position into a realm lineage and a spawn pose, the one
//! spin-up demand constructor that lineage seeds, and the head resolve that turns the demanded realm
//! into a route once something is actually running there.
//!
//! Does NOT own: the fact that this descent happens at the gateway at all. It is a KNOWN OPEN
//! BREACH of "only the parent knows positions" (SL1), kept because of a bootstrap circularity that
//! is measured, not assumed — nothing of the home lineage is running at the instant the answer is
//! needed — and the receiving end checks the pose rather than trusting it.

use super::{
    GatewaySessions, GatewayStats, SeedInjectorConfig, SessionPhase, push_to_shard, store_route,
};
use vd_core::pose::{RealmId, StampedPose};
use vd_core::realm_coord::RealmCoord;
use vd_core::{AccountId, Fence, SessionId};
use vd_sim::io::MsgClass;
use vd_sim::runtime::OutboundBox;
use vd_wire::intershard::{DemandVerb, RealmDemand};
use vd_wire::seams::directory::OwnerRecord;
use vd_wire::session_flow::GatewayToShard;

/// A DETERMINISTIC per-account sentinel [`Fence`] for the injected `RealmDemand`'s `parent_fence` — NOT a
/// global constant (CRITIQUE-1 defense-in-depth: a single shared sentinel would collapse every login's
/// `FencedKey` idempotency into one). It is NOT the auth: `record_demand` does NOT read `parent_fence` for
/// any Step-3 decision (it only `max`-tracks it as `last_fence` for audit — rlm.rs), so this is
/// belt-and-suspenders. Folds the `u128` `AccountId` to `u64`; distinct accounts yield distinct sentinels
/// across the small id space the P7 store issues (a collision is harmless — audit-only). No wall-clock/rng.
pub(crate) fn home_sentinel_fence(account: AccountId) -> Fence {
    let a = account.0;
    Fence((a as u64) ^ ((a >> 64) as u64))
}

/// RLM 5f-3c — SERVER-DERIVE an authenticated login's HOME lineage AND its spawn position: the descent over
/// the account's STORED position (or the origin) yields the FULL root→leaf lineage, so demanding it spins up
/// the whole ancestor chain (the 5f-3a ride). NOTHING client-supplied enters here: the client's only spatial
/// input is its authenticated `AccountId`, so a raw client cannot steer which realm spins up (the abuse
/// boundary).
///
/// The descend is lineage-depth-bounded (`parent()` → `None` at the root), so no extra depth cap is needed.
/// Concrete (non-generic), no wall-clock/rng.
///
/// RLM 5f-3d SCALE: this DESCEND runs exactly ONCE per login. The resolved coord is then carried in
/// [`SessionPhase::AwaitingHomeRealm`] and the pose on the `Session`, so every re-drive rebuilds its demand
/// (and repeats its attach) off the STORED answer instead of re-descending the forest — which also makes it
/// impossible for a re-seed to name a different realm than the one the session is waiting on.
///
/// BOTH halves of a home, READ rather than resolved: the lineage to demand, and the pose to hand the shard
/// in `AttachSession`, already measured from that realm's own centre and stamped with that realm's frame.
///
/// THE DESCENT THAT USED TO BE HERE IS GONE. It walked a private copy of the whole seed forest downward,
/// subtracting each realm's stored centre. A realm that ORBITS stores its centre as zero — its live
/// position rides its parent's per-tick lane — so the walk read every orbiting planet as sitting exactly on
/// its own star, and a login at a star's centre resolved to *inside a planet*. Measured through the shipped
/// boot: all five planets of the first star system answered `4.16 m INSIDE` to that walk while answering
/// between `13.76` and `140.31 m OUTSIDE` in their own frames at their live placements.
///
/// The walk was also the one place a party that owns no realm did realm arithmetic. Both faults have the
/// same cure and it is not a better walk: a home is STORED as a name plus a realm-local pose, so there is
/// nothing to descend and nobody has to know where any realm is. The bootstrap circularity that blocked
/// this — the chain cannot answer "where inside" until it is running, and it only runs because something
/// demanded the answer to "which realm" — does not arise for a stored home, because neither question is
/// asked of anybody.
///
/// Login is NOT a special path: the lineage returned here is demanded through the ordinary mechanics, and
/// nothing downstream can tell where the first position came from.
/// THE WINDOW LANE's lineage rule at a crossing (`docs/design/window_lane.md` §2.6.2: "updated
/// at each crossing's `SubscriptionReady`"): a realm already in the lineage TRUNCATES back to it
/// (an outward cross — ancestors are KEPT); anything else APPENDS below the previous leaf (an
/// inward cross: travel is always out into the shared parent and in again — SL2 — so the
/// previous leaf IS the parent). A wrong append is fail-closed downstream: the Child window it
/// derives is refused by the shard (`window_child_unrostered`) and never confirms, so the chain
/// simply ends at the last attested hop.
pub(crate) fn lineage_apply(lineage: &mut Vec<RealmId>, realm: RealmId) {
    if let Some(pos) = lineage.iter().position(|r| *r == realm) {
        lineage.truncate(pos + 1);
    } else {
        lineage.push(realm);
    }
}

/// A [`RealmCoord`]'s realm chain, root→leaf — THE session lineage the window derivation reads
/// (`docs/design/window_lane.md` §2.6.2: "the lineage comes from the session's login descent").
pub(crate) fn coord_lineage(coord: &RealmCoord) -> Vec<RealmId> {
    let mut chain = Vec::new();
    let mut cursor = Some(coord.clone());
    while let Some(c) = cursor {
        chain.push(c.lowered());
        cursor = c.parent();
    }
    chain.reverse();
    chain
}

pub(crate) fn home_placement(
    cfg: &SeedInjectorConfig,
    account: AccountId,
) -> (RealmCoord, StampedPose) {
    let home = cfg.homes.home_of(account);
    (home.realm, home.pose)
}

/// RLM 5f-3c/5f-3d — THE one `RealmDemand{SpinUp}` constructor for a home lineage (HR3): the initial seed
/// (off the freshly derived [`home_coord`]) and EVERY re-drive (off the coord stored in the phase) build the
/// demand here, so they differ ONLY in `universe_tick` — never in child, verb or fence. The tick is the
/// clock's (determinism: no wall-clock, no rng).
pub(crate) fn demand_for_home(
    child: RealmCoord,
    account: AccountId,
    universe_tick: vd_core::UniverseTick,
) -> RealmDemand {
    RealmDemand {
        child,
        parent_fence: home_sentinel_fence(account),
        verb: DemandVerb::SpinUp,
        universe_tick,
    }
}

/// RLM 5f-3d — THE dynamic-home ROUTE resolve. A `Realm` head reply carries NO session id, so it is matched
/// against the [`GatewaySessions::home_bootstraps`] index: ONE reply resolves EVERY session booting into that
/// realm (a mass login onto the same home is a win, not a fan-out cost).
///
/// - `record` `None` — the realm is NOT routable yet (the orchestrator holds the demand; its shard is still
///   booting). Every waiter STAYS in `AwaitingHomeRealm`: NO `Close`, NO teleport, NO fallback attach to
///   some other shard, no loading screen — the SEAMLESS hold. The re-drive keeps the demand fresh and
///   re-polls this very head until it resolves (or the bounded TTL Closes loudly).
/// - `record` `Some` — the owning node is now known: it JOINS the RUNTIME routable-shard roster (it can
///   never be in the frozen config), the WRITE route is retargeted to it through the sole `store_route`
///   primitive, the phase advances to `AwaitingAttach`, and the attach is sent THERE (never to
///   `config.shard`).
///
/// The wait ENTRY SURVIVES this resolve (only `resolved` flips): the demand re-seed must continue across the
/// attach round-trip or the reconciler's arm-A lapses and reaps the realm out from under the login. That makes
/// the per-member `AwaitingHomeRealm` test below load-bearing rather than decorative — it is what keeps the
/// resolve IDEMPOTENT under at-least-once delivery. A duplicate reply finds its members in `AwaitingAttach`
/// and touches nothing, so it can neither re-`claim_dynamic_shard` (a refcount leak that would pin the node
/// on the roster forever) nor demote a session that has already gone `Active`.
///
/// A reply nobody waits on — EVERY `Realm` head in static mode, or one after the last member left — is a
/// clean no-op, exactly the pre-5f-3d behaviour for this arm.
///
/// SCOPE: a home realm that MIGRATES to a different node after this resolve but before `SessionAttached`
/// does not re-point the attach (its members have left `AwaitingHomeRealm`) — it rides the bounded bootstrap
/// TTL and Closes loudly, then the client re-logins onto the new owner. Live re-pointing mid-bootstrap
/// belongs with the D-34 authority-follows-commit work, not here.
pub(crate) fn on_home_realm_head(
    home_rid: RealmId,
    record: Option<OwnerRecord>,
    sessions: &mut GatewaySessions,
    stats: &mut GatewayStats,
    outbox: &mut OutboundBox,
) {
    let Some(owner) = record else {
        return; // still booting — hold the client, seamlessly
    };
    let home = owner.authority.node();
    let Some(wait) = sessions.home_bootstraps.get_mut(&home_rid) else {
        return; // nobody is booting into this realm (every static-mode Realm head lands here)
    };
    // The head poll is SATISFIED for every current member: stop re-polling (the demand half of the re-drive
    // deliberately keeps running — see `HomeWait::resolved`). A later joiner clears this again.
    wait.resolved = true;
    let members: Vec<SessionId> = wait.members.iter().copied().collect();
    for session_id in members {
        let Some(session) = sessions.by_session.get_mut(&session_id) else {
            // The index and `by_session` are kept in sync by the begin/end pair; a miss is an
            // invariant breach — counted, never a silent continue (the C2 honesty floor).
            stats.home_wait_desync += 1;
            continue;
        };
        if !matches!(session.phase, SessionPhase::AwaitingHomeRealm { .. }) {
            // Already resolved by an earlier reply for this realm (it sits in `AwaitingAttach`, still a
            // member because the re-seed must continue): a duplicate is an exact no-op, never a second
            // roster claim and never a re-attach storm.
            continue;
        }
        session.home_shard = Some(home);
        // THE sole route-mutation primitive (HR3): retarget the WRITE route's authority to the home shard,
        // CARRYING the route's current fence — the attach SETS the fresh realm fence a round-trip later.
        // Without this the session would attach to (and subscribe on) its home while still routing input at
        // the placeholder `config.shard`.
        store_route(&session.hot, home, session.hot.route.load().fence, None);
        session.phase = SessionPhase::AwaitingAttach;
        let (fence, account, spawn) = (session.fence, session.account, session.spawn);
        push_to_shard(
            outbox,
            home,
            MsgClass::Control,
            &GatewayToShard::AttachSession {
                session: session_id,
                fence,
                account,
                // THE realm this pose was measured against is the one we are attaching to — this arm is
                // reached only after the `Realm(home_rid)` head named the node owning that very realm.
                spawn,
            },
        );
        // The home joins the RUNTIME routable roster, so its `SessionAttached` + frames are node-class
        // dispatchable as a shard (its NodeId was minted at spawn — never in the frozen config).
        sessions.claim_dynamic_shard(home);
    }
}
