//! AREA OF INTEREST: which of MY direct children must stay alive, decided by ME, about THEM.
//!
//! Owns: the observer set this realm folds (the occupants it holds, plus each OCCUPIED CHILD
//! treated at the placement this realm authored for it), the per-`(observer, child)` hysteresis and
//! grace, the demand emitted from it, and the one occupancy bit a child reports upward.
//!
//! Does NOT own: any statement about ITSELF. A realm never reaches upward and has zero control over
//! its parent; liveness is decided by looking only at itself and one level down, and area of
//! interest is decided by the PARENT, never by the realm about itself (SL7). Exactly one bit
//! crosses — occupancy, upward. No occupant pose, no entity set, no depth or hop count, anywhere.

use super::{
    ChildRealmNodes, Dots, HandoffHolds, InBandVerdict, InterestEmitLatch, InterestHeld,
    OpenWindows, OwnedTransients, ParentRealmNode, Placements, RealmAuthority, RealmRegions,
    StubConfig, StubStats, aoi_recheck_cadence, emit_realm_interest, emit_window_membership,
    head_reads_due, parent_headread_due, region_level, retain_live, retain_ttl_ticks, speaks_for,
    ttl_alive,
};
use crate::io::MsgClass;
use crate::runtime::{ClockSample, OutboundBox};
use bevy_ecs::prelude::{Res, ResMut, Resource};
use std::collections::{BTreeMap, BTreeSet};
use vd_core::glam::DVec3;
use vd_core::placement::PlacementBook;
use vd_core::pose::{LatticePos, RealmId, Tier};
use vd_core::realm_coord::RealmCoord;
use vd_core::{AccountId, EntityId, Fence, NodeId, SessionId, TickId, UniverseTick};
use vd_wire::intershard::{DemandVerb, InterShardFlow, RealmDemand};
use vd_wire::seams::directory::{DirectoryKey, DirectoryOp};

/// One direct child's SL7 occupancy bit as this PARENT holds it (Step 5 slice A) — written by the
/// `ChildLive` receive arm, pruned on the ONE retain TTL (`retain_ttl_ticks`). Presence within the
/// TTL IS the bit; `fence`+`at` are the last-wins ordering key (a deposed incarnation's heartbeat
/// is rejected); `home` is the sender `NodeId` — the HR1-preserving return address for everything
/// shipped back down (a re-homed child re-targets its parent's down-lanes with its first heartbeat).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ChildLiveEntry {
    pub home: NodeId,
    pub fence: Fence,
    pub at: UniverseTick,
    pub last_seen: TickId,
}

/// The parent-side store of its direct children's occupancy bits (Step 5 slice A), keyed by the
/// child's lowered realm. Replaced the pose-derived activity the deleted retained-occupant lane
/// carried (slice D) — a child is LIVE here iff its bit is fresh, which is all SL7 lets a parent
/// know.
#[derive(Resource, Debug, Default)]
pub struct ChildLiveness(pub BTreeMap<RealmId, ChildLiveEntry>);

/// Step 5 lane cure (finding 41) — HAS THE PARENT BEEN TOLD this realm is occupied, since it last went
/// empty? The SL7 bit beats on the AoI cadence (its contract's stated rate), and this latch is what
/// derives the ADOPT EDGE with no hook in any adopt path: `aoi_decide` flips it false at the Empty
/// self-report and true when a bit actually ships, so the first occupied tick the parent is resolved
/// emits immediately — the occupancy transition itself, which is what the bit MEANS — and every later
/// tick waits for the cadence. Deliberately "told", not "was occupied": a realm whose parent is still
/// unresolved keeps the edge armed, so the first resolve ships the bit at once rather than waiting out
/// a cadence. PURE emit bookkeeping — no fence, never persisted. Never flips at walk/static (no parent
/// ever resolves, no bit ever ships) ⇒ byte-identical.
#[derive(Resource, Debug, Default)]
pub struct WasOccupied(pub bool);

/// Identity of ONE AoI observer — the key that gives each occupant its OWN per-child hysteresis latch
/// (VU S0: the cull is now PER-OBSERVER, not a global scalar-min over all occupants). The observers this
/// shard evaluates are the occupants it SIMULATES — a durable dot (keyed by its `SessionId`) or a held
/// transient (keyed by its `EntityId`) — plus its occupied direct CHILDREN (the third arm below). The
/// per-occupant PROXY arm an earlier slice put here was the SL2 breach; it died with its lane (Step 5
/// slice D), and a deep observer's siblings are culled by its ancestor treating each occupied child AS
/// the observer instead.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum ObserverId {
    Dot(SessionId),
    Transient(EntityId),
    /// Step 5 slice B — an OCCUPIED DIRECT CHILD (a fresh SL7 bit), standing at the placement this
    /// shard authors for it and reaching as far as its own extent (SL7's occupied-child proxy: the
    /// child stands in for whoever is inside it, at exactly the resolution this shard's decision is
    /// meaningful at). Runs the AoI/latch math like any observer but never enters an `Occupants`
    /// verdict (the parent hosts no client for it — gated on `render_routes`); its in-range set is
    /// the verdict a [`WindowScope::Child`] window is served instead.
    Child(RealmId),
    /// Look horizon slice 4 (§3.4.3, THE DOWN-PROXY — SL7 read in the other direction): the ONE
    /// synthetic observer a realm holding a live interest byte inserts, at its own centre, with
    /// reach equal to its own extent — an interested realm is its outside observers' proxy, at
    /// its own scale, with the error bounded by its own extent (SL7's own bound, same reason).
    /// Runs the existing fold unchanged; INTEREST-derived at construction, so it never produces
    /// interest for the next level down (the structural cascade cap) and never counts as
    /// occupancy (no Empty suppression, no `ChildLive` bit — a byte from outside must never
    /// manufacture an occupancy fact).
    Interest,
}

/// WHERE an AoI observer entry CAME FROM (look horizon §3.4.3 — the flag set at construction
/// that IS the structural cascade cap): occupancy-derived (a real dot, a held transient, an
/// occupied-child proxy) or interest-derived (the one synthetic down-proxy). ONLY
/// occupancy-derived observers produce interest for the next level down, count at the Empty
/// gate, or beat the `ChildLive` bit — so an observer's realm wakes its children's interiors
/// and STOPS: live realms below the observer, exactly two, with no depth number and no hop
/// count crossing any boundary.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum ObserverOrigin {
    Occupancy,
    Interest,
}

/// One entry of the observer list [`aoi_decide`] folds — the tuple grew a name when the origin
/// flag landed (look horizon slice 4). `reach` is the observer's own physical extent, a GENERIC
/// parameter (no observer-type branch): a point occupant reaches 0 m; an occupied child (and
/// the interest down-proxy) reaches to its own surface, so warming errs early, never late.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct AoiObserver {
    pub(crate) id: ObserverId,
    pub(crate) pos: LatticePos,
    pub(crate) vel: DVec3,
    pub(crate) reach: f64,
    /// SET ONLY FOR THE INTEREST PROXY (S10): how far the outside looker is from THIS realm's centre.
    /// `Some` flips the distance rule below from "how far is the child from me" to "how near could the
    /// looker be to that child" — see the proxy's construction for why the direction is absent by law.
    pub(crate) outside_from_m: Option<f64>,
    pub(crate) origin: ObserverOrigin,
}

/// Per-`(observer, child)` AoI hysteresis + grace (RLM Step 2 → VU S0 per-observer). Keyed by the
/// OBSERVER plus the child's own `RealmId`, so each
/// occupant carries its own acquire/grace latch and the child-level demand is the UNION over observers
/// ([`union_verb`]). Twin of [`ContainmentProgress`]. `BTreeMap` (no default-hasher HashMap in sim —
/// determinism). Default empty; lazily evicted each tick to the current (observer × direct-child) roster
/// (`retain_live`).
#[derive(Resource, Debug, Default)]
/// ★ KEYED BY IDENTITY, NOT BY LINEAGE (slice S10). This was keyed by the child's full `RealmPath` — a
/// heap-allocated list of levels, CLONED TWICE PER (observer, child) PAIR every tick and compared level by
/// level on every lookup. At the target census of 150,000 children that is 300,000 list allocations per
/// tick for a question about identity. A direct child's `RealmId` answers the same question: this map is
/// SHARD-LOCAL and only ever holds one shard's own direct children, so no two keys can collide, and the
/// path is a pure function of the id for the one or two children that actually get a message.
///
/// It also makes the map contiguous per observer, which is the primitive the S12 fold inversion needs
/// (D-MOVE-1 / M8): "which children does THIS observer hold a latch for" becomes a range read.
pub struct AoiMembership(pub BTreeMap<(ObserverId, RealmId), AoiState>);

/// One direct child's AoI state: `was_in` (acquired — for the hysteresis) + `grace_remaining` (ticks a
/// would-be release is held after the last in-range observation).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct AoiState {
    pub(crate) was_in: bool,
    pub(crate) grace_remaining: u32,
}

impl AoiState {
    /// Is this (observer, child) pair inside the band right now — the latched verdict the SL7
    /// membership emit ships and the demand union reads? A READ-ONLY diagnosis accessor: the
    /// latch itself is written only by `aoi_transition`, so there is exactly one author.
    #[must_use]
    pub fn in_band(&self) -> bool {
        self.was_in
    }
}

/// The SL7/window stores bundled into ONE tuple `SystemParam` (bevy's 16-param ceiling; a mechanical
/// arity fix, destructured right back inside [`evaluate_realm_aoi`]): the children's occupancy bits
/// (pruned + folded as observers), the window registry whose membership baselines the AoI pass
/// diffs against, and (look horizon slice 3) the realm-side shared in-band verdict the up-relay's
/// interior forward gates on (§3.4.5 — written HERE, by the one fold, never derived twice). The
/// three scene stores that used to ride here — the per-live-child reflect cache, the from-above
/// holding and the observed interior outlines — died with their lanes (window lane Slice C2,
/// minor 19).
type Sl7Stores<'w> = (
    ResMut<'w, ChildLiveness>,
    ResMut<'w, OpenWindows>,
    ResMut<'w, InBandVerdict>,
    // Look horizon slice 4 — the interest byte's two ends, both consulted by the ONE fold: the
    // CHILD-side holder (TTL-pruned here; a live `1` powers the down-proxy observer) and the
    // PARENT-side per-child emission latch (the interior-band hysteresis).
    ResMut<'w, InterestHeld>,
    ResMut<'w, InterestEmitLatch>,
);

/// RLM Step 2 — the demand-driven realm-lifecycle detector (the SIBLING of [`evaluate_realm_boundaries`]):
/// per tick, this shard computes which of its DIRECT children an occupant's Area-of-Interest reaches and
/// emits a [`RealmDemand`] toward the orchestrator so those child realms spin up (and self-reports its own
/// realm's emptiness so it can be torn down). This realizes the decentralized RLM policy: EVERY realm's
/// shard is the AoI authority for ITS children — no central AoI scan (HR2/HR3 generic; one loop, no
/// match-on-realm-kind). A BRANCHLESS system shim: only the two guard `else`s (both mirroring the covered
/// detector), then delegate — ALL hysteresis/predictive/emit branching lives in the monomorphic
/// [`aoi_decide`]/[`aoi_transition`]/[`aoi_min_dist`] helpers (HR5 per-monomorphization discipline).
///
/// Gated `.run_if(has_synced)` + on `RealmAuthority` + on a non-empty region set, so it is INERT (emits
/// nothing) at walk/canonical scale where the seed AoI bands are [`AoiConfig::inert`] — byte-identical.
/// Step 2 NEVER emits a parent `TearDown` (REVISION 1 R2 supersedes the §2.2 pseudocode): a child leaving
/// AoI simply STOPS being demanded (its key drops after grace); the Step-3 reconciler closure is the sole
/// kill authority.
#[allow(clippy::too_many_arguments)]
pub(crate) fn evaluate_realm_aoi(
    config: Res<StubConfig>,
    clock: Res<ClockSample>,
    authority: Res<RealmAuthority>,
    regions: Res<RealmRegions>,
    ledger: Res<Placements>,
    dots: Res<Dots>,
    owned_transients: Res<OwnedTransients>,
    // The hand-offs this shard is still party to — read-only here; `process_inbound` owns the writes.
    holds: Res<HandoffHolds>,
    mut membership: ResMut<AoiMembership>,
    mut outbox: ResMut<OutboundBox>,
    // The resolved parent node (from the cadence HeadRead) — the up-lanes' relay target. `None` at a
    // root shard / before the first parent Head reply ⇒ nothing upward fires this tick.
    parent: Res<ParentRealmNode>,
    // Step 5 — the SL7/scene stores bundled into ONE tuple `SystemParam` (bevy's 16-param ceiling; a
    // mechanical arity fix, destructured right back below): the children's occupancy bits (pruned +
    // folded as observers), the per-live-child send-on-change cache, the from-above holding (folded
    // into every occupant's scene + restated onward), and the observed interior outlines (READ-ONLY —
    // `emit_realm_frames` owns their writes/prune, and it runs earlier on the same schedule).
    sl7: Sl7Stores,
    // Lane cure (finding 41) — the bit's "parent has been told" latch (the adopt-edge derivation).
    mut was_occupied: ResMut<WasOccupied>,
    // Look horizon slice 4 — the interest emission's route: the directory-attested child head
    // map the admission reads already maintain (READ-ONLY here; `process_inbound` owns the
    // writes). The byte goes direct to the child's head on the route the parent already
    // resolves — never further, never sideways, never through the orchestrator (§2 ASK B).
    child_nodes: Res<ChildRealmNodes>,
    // The chain's emit-side observability: read nowhere in the sim — a counter, never a control input.
    mut stats: ResMut<StubStats>,
) {
    let (mut child_liveness, mut open_windows, mut verdict, mut interest_held, mut interest_latch) =
        sl7;
    // Authority gate (verbatim `evaluate_realm_boundaries`): a shard without its realm lease demands
    // nothing — the `realm_fence` is the emitter's authority proof carried on every demand.
    let Some(realm_fence) = authority.0 else {
        return;
    };
    // Inert until regions are planted (production through canonical scale): no children to evaluate.
    if regions.is_empty() {
        return;
    }
    // L3: `aoi_decide` names the PRIMARY realm's (`config.realm`) direct children via ONE `own_coord`.
    // For node-per-realm (the base) that is complete; a co-hosting shard's co-hosted realms' children are
    // simply not evaluated here (D-44 dormant — see `register_stub_shard`). No panic: dormant co-hosting
    // tests legitimately run this with `held_realms.len() > 1` and inert bands, emitting nothing.
    // The HEAD book — the writer ran first on this same gated schedule, so the invariant is stated.
    let head = ledger
        .0
        .head(config.realm)
        .expect("the writer authors every held anchor before the AoI pass runs");
    aoi_decide(
        &config,
        &clock,
        &regions,
        head,
        &dots,
        &owned_transients,
        &holds,
        realm_fence,
        &mut membership.0,
        &mut outbox,
        parent.0,
        &mut was_occupied.0,
        &mut child_liveness.0,
        &mut open_windows,
        &mut verdict.0,
        &mut interest_held,
        &mut interest_latch.0,
        &child_nodes,
        &mut stats,
    );
}

/// The monomorphic AoI decision (ALL branching HERE, HR5). PER-OBSERVER (VU S0): each occupant carries its
/// OWN acquire/grace latch keyed `(observer, child.path())`, and the child-level demand is the UNION over
/// observers ([`union_verb`]) — SpinUp the tick a child FIRST becomes demanded by anyone, KeepAlive while
/// sustained. With ZERO occupants the shard self-reports `Empty{own_coord}` (the sole occupancy authority a
/// sealed parent cannot see — Step-3 EDGE 1). Deterministic: at most ONE demand per child, emitted AFTER
/// the observer fold (occupant iteration order cannot leak into the demand set, H-2); children fold in the
/// stable seed-derived `child_placements` order.
#[allow(clippy::too_many_arguments)]
fn aoi_decide(
    config: &StubConfig,
    clock: &ClockSample,
    regions: &RealmRegions,
    book: &PlacementBook,
    dots: &Dots,
    owned: &OwnedTransients,
    // The hand-offs this shard is still party to — the subjects it has let go of but not yet seen taken
    // over. They count as present here exactly as an owned dot does (`speaks_for`). EMPTY at a disarmed
    // budget ⇒ every filter below reduces to the old ownership test.
    holds: &HandoffHolds,
    realm_fence: Fence,
    membership: &mut BTreeMap<(ObserverId, RealmId), AoiState>,
    outbox: &mut OutboundBox,
    parent_node: Option<NodeId>,
    // Lane cure (finding 41) — the "parent has been told" latch that derives the bit's adopt edge;
    // false again at every Empty self-report.
    was_occupied: &mut bool,
    // Step 5 slice B — the direct children's SL7 occupancy bits: pruned here on the derived TTL,
    // then folded as one synthetic observer per FRESH bit (SL7's occupied-child proxy).
    child_liveness: &mut BTreeMap<RealmId, ChildLiveEntry>,
    // THE WINDOW LANE (§2.9 step 5): the open windows whose SL7 membership verdicts
    // this SAME fold ships — never a second AoI derivation (HR3).
    windows: &mut OpenWindows,
    // Look horizon slice 3 (§3.4.5) — the realm-side SHARED in-band verdict: the union over
    // every observer of its in-band direct children, written by THIS fold (the one AoI
    // derivation) for the up-relay's interior forward gate to read. Cleared at the
    // zero-observer return (a vacated realm forwards no interior).
    shared_verdict: &mut BTreeSet<RealmId>,
    // Look horizon slice 4 — the interest byte's two ends: the CHILD-side holder (TTL-pruned
    // here; a live `1` powers the ONE down-proxy observer) and the PARENT-side per-child
    // emission latch (the interior-band hysteresis); plus the attested child-head routes the
    // emission sends on.
    interest: &mut InterestHeld,
    interest_latch: &mut BTreeSet<RealmId>,
    child_nodes: &ChildRealmNodes,
    stats: &mut StubStats,
) {
    let own_coord = &config.own_coord;
    let tick = clock.universe_tick;
    let horizon_s = f64::from(config.boot_ticks_p99) * config.tick_dt_s; // F7 predictive horizon
    // The hand-off budget, read once — every `speaks_for` below asks the same question of the same clock.
    let hold_ttl = config.handoff_hold_ttl_ticks;
    let local = clock.local_tick;
    // The cell size the AoI arithmetic subtracts in. Observers and child placements are BOTH measured in
    // this shard's own ambient frame (H-1), so one tier serves every distance below.
    let own_tier = config.frame.tier();

    // Step 5 slice B — EXPIRE stale child bits on the ONE derived TTL: a child that stopped
    // heartbeating stops counting as an occupied-child observer and stops being named in any
    // verdict — all from this one prune. The TTL window bridges a lost `Unreliable` heartbeat so a
    // single missed datagram never blinks a warmed neighbourhood. Inert at walk/static (the store
    // is empty — no bit is ever received — so the prune is a no-op ⇒ byte-identical).
    let retain_ttl = retain_ttl_ticks(config);
    child_liveness.retain(|_, e| ttl_alive(e.last_seen, clock.local_tick, retain_ttl));
    // Look horizon slice 4 — the interest byte DECAYS on the same derived TTL: silence means
    // "nobody is watching" (the safe direction; §3.6 — two lost beats expire it, then the
    // teardown cooldown and grace still apply downstream).
    if interest
        .0
        .is_some_and(|e| !ttl_alive(e.seen, clock.local_tick, retain_ttl))
    {
        interest.0 = None;
    }

    // THE LANE CADENCE, resolved once per pass (lane cure, findings 41 + 0/43): the SL7 bit beats on
    // it (`bit_due`, plus the adopt edge — the same expression the shape ship uses, HR3), and the
    // admission head-reads ride it behind the same `aoi_live` gate the parent resolve uses
    // (`heads_due` — one fold, `head_reads_due`).
    let bit_due = crate::directory::due_this_tick(aoi_recheck_cadence(config), clock.local_tick.0);
    let heads_due = head_reads_due(config, regions, clock);

    // VU AoI S2a-2b-i — resolve THIS realm's PARENT node. Periodically HeadRead the parent's directory
    // record so its authoritative node lands in `ParentRealmNode` (written from the reply in
    // `on_directory_reply`) — the relay target of BOTH occupant up-lanes and the up-observation lanes
    // (rows + outlines). Reuses the SAME directory HeadRead/cadence the own-realm recheck uses (HR3),
    // keyed on the parent `RealmCoord` (`lowered()`, like the directory — the boot guard rules out a
    // self-aliasing parent). Inert at walk/static: `aoi_live()` is false ⇒ `parent_headread_due`
    // returns `None` ⇒ nothing emitted.
    //
    // DELIBERATELY ABOVE THE EMPTY RETURN, and that placement is a measured fix, not a tidy-up. An
    // armed realm that holds NOBODY still authors its interior and still owes its parent the
    // observation of it — that is what lets an approaching traveller see a neighbour realm's contents
    // before ever crossing in (SL3: visibility is the spin-up trigger). This block used to sit below
    // the zero-occupant return, so a realm nobody had ever entered could not resolve its parent and
    // its whole up-observation lane stayed mute: flown 2026-08-13 as "approaching another star
    // system, its planets were not loading" — the exited system kept working only because its parent
    // node had been resolved WHILE it was occupied and stayed cached.
    if let Some(parent_coord) = parent_headread_due(config, regions, clock) {
        outbox.push_flow(
            config.orchestrator,
            MsgClass::Saga,
            &InterShardFlow::Directory(DirectoryOp::HeadRead {
                key: DirectoryKey::Realm(parent_coord.lowered()),
            }),
        );
    }

    // The UNIFIED direct-child placements — the HEAD book's rows joined back onto the child regions,
    // stable seed-derived Vec order — the SAME authored table the observer feed reads (H-1/H-2, no
    // reorder). Resolved BEFORE the observer fold: the SL7 child observers below stand AT these rows.
    let placements = regions.child_rows(config.realm, book);

    // Step 5 slice B — THE SL7 OCCUPIED-CHILD PROXY, verbatim: for each of MY direct children with a
    // FRESH occupancy bit, one synthetic observer at the placement AND velocity I already author for
    // it, reaching as far as the child's own extent. Zero new data crosses: the placement and the
    // extent are mine, the bit already arrived. The child stands in for whoever is inside it; the
    // error is bounded by the child's size, which is the resolution my decision is meaningful at.
    // EMPTY at walk/static (no bit ever arrives) ⇒ byte-identical.
    let child_observers = placements
        .iter()
        .filter(|(region, _)| child_liveness.contains_key(&region.realm))
        .map(|(region, pose)| AoiObserver {
            id: ObserverId::Child(region.realm),
            pos: pose.pos,
            vel: pose.vel,
            reach: region.shape.circumscribed_extent(),
            outside_from_m: None,
            origin: ObserverOrigin::Occupancy,
        });

    // Observers = durable dots this shard SPEAKS FOR — owned, or mid-hand-off — ∪ held transients (mirror
    // `evaluate_realm_boundaries`) ∪ the OCCUPIED direct children (SL7, slice B), each TAGGED with its
    // `ObserverId` + reduced to frame-local `(pos, vel, reach)` — the SAME own frame the child placements
    // use (H-1) — so the AoI distance and the observer feed measure one geometry. `reach` is the
    // observer's own physical extent, a GENERIC parameter (no observer-type branch): a point occupant
    // reaches 0 m; an occupied child reaches to its own surface, so warming errs early, never late.
    //
    // A dot mid-hand-off is counted HERE, which is what stops a realm calling itself empty while somebody is
    // still leaving it (the emptiness gate a few lines down reads this very list). It also holds the children
    // that occupant could see warm across the window, so a crossing that aborts finds its neighbourhood
    // exactly as it left it instead of re-spinning it. And an OCCUPIED CHILD is counted here too, which
    // is SL7's liveness rule made structural: no occupants AND no live child is exactly
    // `observers.is_empty()`.
    // Look horizon slice 4 (§3.4.3, THE DOWN-PROXY): a realm holding a live interest signal inserts
    // ONE synthetic observer at its OWN centre (the origin of its own frame — the one position a realm
    // lawfully has) and runs this very fold unchanged. INTEREST-derived at construction — the
    // structural cascade cap: it drives demand and the shared verdict, but never the Empty report,
    // never the bit, and never an interest emission of its own.
    //
    // ★ IT NO LONGER WAKES EVERY CHILD (slice S10, owner-approved under SL6 2026-08-26). It used to
    // carry `reach = my own extent`, so every child's distance clamped to ZERO and every band admitted
    // it — 150,000 realms woken every tick at the target census, by construction rather than by
    // accident, and unfixable by any change of loop shape.
    //
    // The signal now carries HOW FAR the looker is (`from_m`). The looker's DIRECTION is unknown by
    // law (SL2 forbids a pose crossing), so the honest distance to one of my children is the NEAREST
    // the looker could possibly be: its distance to me, less the child's distance from my centre. That
    // is computed in the pair loop below.
    //
    // The visibility test itself is the child's OWN existing band — which worldgen already derives
    // from angular size — so nothing new decides what is worth waking, and no threshold constant had
    // to cross into this crate.
    // The `own_shape` precondition is KEPT (S10) even though the reach no longer comes from it: a realm
    // that cannot state its own boundary has no business reasoning about what is inside it, and dropping
    // the guard with the reach would have been a silent behaviour change in the degenerate boot.
    let interest_proxy = interest
        .0
        .and_then(|e| e.from_m)
        .filter(|_| regions.own_shape(config.realm).is_some())
        .map(|from_m| AoiObserver {
            id: ObserverId::Interest,
            pos: LatticePos::ORIGIN,
            vel: DVec3::ZERO,
            reach: 0.0,
            outside_from_m: Some(from_m),
            origin: ObserverOrigin::Interest,
        });
    let observers: Vec<AoiObserver> = dots
        .0
        .iter()
        .filter(|(_, d)| speaks_for(holds, d, local, hold_ttl))
        .map(|(s, d)| AoiObserver {
            id: ObserverId::Dot(*s),
            pos: d.pose.pos,
            vel: d.pose.vel,
            reach: 0.0,
            outside_from_m: None,
            origin: ObserverOrigin::Occupancy,
        })
        .chain(
            owned
                .0
                .iter()
                .filter(|(_, t)| t.status.is_held())
                .map(|(e, t)| AoiObserver {
                    id: ObserverId::Transient(*e),
                    pos: t.pose.pos,
                    vel: t.pose.vel,
                    reach: 0.0,
                    outside_from_m: None,
                    origin: ObserverOrigin::Occupancy,
                }),
        )
        .chain(child_observers)
        .chain(interest_proxy)
        .collect();

    // THE OCCUPANCY TRUTH (look horizon slice 4 splits it from mere list-emptiness): does this
    // realm hold occupants or a live child — the ONLY facts the Empty self-report and the SL7
    // bit may ever state? The interest down-proxy deliberately does NOT count: a byte from
    // outside must never manufacture an occupancy fact (SL7's bit means occupancy, nothing
    // else), so an interest-held vacated realm keeps reporting Empty — truthfully — while its
    // parent's own demand (the observer stands well inside the AoI band whenever it stands
    // inside the interior band) is what keeps it alive to serve pictures.
    let occupied = observers
        .iter()
        .any(|o| o.origin == ObserverOrigin::Occupancy);
    // Zero-occupant self-report: this CHILD shard tells the orchestrator its OWN realm holds nobody
    // (`child == own_coord`). Fence = this shard's own realm authority over its emptiness.
    if !occupied {
        // Lane cure (finding 41): going empty re-arms the adopt edge — the NEXT occupant's first tick
        // ships the bit immediately, off-cadence, because that transition is what the bit means.
        *was_occupied = false;
        push_demand(
            outbox,
            config.orchestrator,
            own_coord,
            own_coord.clone(),
            realm_fence,
            DemandVerb::Empty,
            tick,
            stats,
        );
    }
    if observers.is_empty() {
        // THE WINDOW LANE (Slice A): with zero observers every window's SL7 verdict IS the empty
        // set — the diff below ships the removals once, so a subscriber's drawn-set gate never
        // holds bodies for a realm whose occupants all left (the same emptiness truth the
        // self-report above states to the orchestrator). The shared verdict clears with it
        // (look horizon slice 3): a vacated realm forwards no interior.
        shared_verdict.clear();
        emit_window_membership(windows, &BTreeMap::new(), &BTreeMap::new(), outbox, stats);
        return;
    }

    // Step 5 slice A — THE SL7 OCCUPANCY BIT, upward: observers are non-empty from here (the Empty
    // arm returned above), so this realm is LIVE and says so to its parent — one heartbeat, presence
    // is the bit. ON THE AoI CADENCE, plus immediately at the occupancy transition (lane cure,
    // finding 41 — the rate its own contract states): `was_occupied` latches "the parent has been
    // told", so the first tick somebody is inside (an adopted crosser, a login) ships the bit at once
    // — the transition itself, derived here with no hook in any adopt path — and every later tick
    // waits for the cadence beat. The receiver's TTL is sized in cadences (`retain_ttl_ticks`), so a
    // lost datagram still only costs staleness, never a blink; the reconciler's `ancestor_close`
    // backstops liveness centrally. A shard whose parent is unresolved (the root; a lease race at
    // boot) emits nothing and keeps the edge armed — the first resolve ships the bit immediately.
    // Bitwise `|` (both operands pure, HR5).
    // Gated on the OCCUPANCY truth, not list non-emptiness (look horizon slice 4): the interest
    // down-proxy alone must never beat the bit — that would tell the parent "occupied" about a
    // realm holding nobody, and the parent's occupied-child proxy would cascade on a fiction.
    if occupied && let Some(parent) = parent_node {
        let adopt_edge = !*was_occupied;
        if bit_due | adopt_edge {
            let bit = InterShardFlow::ChildLive(vd_wire::intershard::ChildLive {
                child: own_coord.clone(),
                fence: realm_fence,
                at: tick,
            });
            outbox.push_flow(parent, MsgClass::SignalDelta, &bit);
            *was_occupied = true;
        }
    }

    // VU AoI S1b — the RENDER route for each DOT observer (a player with a client): its DURABLE id + the
    // gateway to reach it. Only dots receive render deltas; held transients keep realms warm (lifecycle) but
    // have no client to draw. Each dot's LEVEL set is accumulated below and diffed into one delta per dot.
    let render_routes: BTreeMap<ObserverId, (AccountId, NodeId)> = dots
        .0
        .iter()
        .filter(|(_, d)| d.authority.simulates())
        .map(|(s, d)| (ObserverId::Dot(*s), (d.account, d.gateway)))
        .collect();
    // THE WINDOW LANE (§2.9 step 5) — the per-observer `next_in` transitions, accumulated as IDS
    // ONLY: per DOT observer its in-band DIRECT children (the `Occupants`-window fold), per
    // OCCUPIED-CHILD observer its in-band siblings (the SL7 proxy fold — the occupied child stands
    // in for whoever is inside it). This IS the whole render output of the fold since Slice C2:
    // the verdict says WHICH realms an observer's band admits, and each realm states its own look
    // on its own window (SL3). The two OUTLINE accumulators that used to sit here — the per-dot
    // `RealmSceneDelta` push and the per-live-child `ChildSceneSet` reflect — are deleted with
    // their lanes; a parent has no business authoring what its children look like.
    let mut in_band: BTreeMap<ObserverId, BTreeSet<RealmId>> = BTreeMap::new();
    // Look horizon slice 3 (§3.4.5) — the SHARED verdict accumulator: the union over EVERY
    // observer (a dot, a held transient, an occupied child) of its in-band children this tick.
    // Same transitions, same bands, same grace as the demand — read, never re-derived.
    let mut fold_verdict: BTreeSet<RealmId> = BTreeSet::new();

    let mut live_keys = BTreeSet::<(ObserverId, RealmId)>::new();
    // Look horizon slice 4 — the children seen this pass (the interest latch's evict set).
    let mut live_realms = BTreeSet::<RealmId>::new();
    for (region, pose) in &placements {
        // A Ship child has no lineage coord to demand/draw by until P8 (D-SHIP-1): excluded from
        // the AoI/demand/render fold, counted, never a panic — see `region_level`.
        let Some(level) = region_level(region) else {
            stats.ship_child_regions_excluded += 1;
            continue;
        };
        let child_coord = own_coord.child(level);
        live_realms.insert(region.realm);
        let child_pos = pose.pos; // own frame (== the placements' frame), carried WHOLE

        // Was the child kept-alive by ANY observer at tick START (its acquire latch held)? — the input to
        // the SpinUp-vs-KeepAlive union split, read BEFORE this tick's latch updates.
        let was_demanded = observers.iter().any(|o| {
            membership
                .get(&(o.id, region.realm))
                .is_some_and(|s| s.was_in)
        });
        // Fold each observer's own hysteresis machine; a child is demanded THIS tick iff ANY observer's
        // machine keeps it live (SpinUp | KeepAlive).
        let mut now_demanded = false;
        // Look horizon slice 4 — the interest emission's input: the LEAST distance from any
        // OCCUPANCY-derived observer to this child (the cascade cap at the source: the
        // interest-derived down-proxy never produces interest for the next level down) —
        // EXCLUDING the child's OWN occupied-child proxy. The byte exists to make "something
        // OUTSIDE may be looking in" representable (Q1's exact sentence); a child's own proxy
        // stands FOR its occupants, who are INSIDE — the child already holds them and already
        // wakes its own interior for them. Without this exclusion the chain is measured
        // unlawful (flown as the warp gate's red, 2026-08-17): the universe flags its occupied
        // galaxy, the galaxy's down-proxy reaches its own full 12.5 km extent, and EVERY star
        // system wakes — the exact "wake every interior" alternative §2 ASK B priced and
        // rejected. Occupants, transients and OTHER children's proxies still produce interest
        // (SL7's sibling-warming stays). Bitwise `&` (both operands pure, HR5).
        let mut interest_min_dist = f64::INFINITY;
        for o in &observers {
            let key = (o.id, region.realm);
            live_keys.insert(key);
            // The observer's own extent SHORTENS the distance (never below zero): an occupied child
            // reaches to its own surface, so it warms a sibling as soon as any of its occupants
            // could be that close — the SL7 error bound made operational (warm early, never late).
            let raw = occupant_child_dist(o.pos, o.vel, child_pos, own_tier, horizon_s);
            let dist = match o.outside_from_m {
                // THE LOOKER IS OUTSIDE THIS REALM, and its direction is unknown by law. `raw` is how
                // far this child sits from my centre; `d_out` is how far the looker sits from my
                // centre. The nearest the two could be is the difference — the conservative reading,
                // which over-wakes and never under-wakes.
                Some(d_out) => (d_out - raw).max(0.0),
                None => (raw - o.reach).max(0.0),
            };
            if (o.origin == ObserverOrigin::Occupancy) & (o.id != ObserverId::Child(region.realm)) {
                interest_min_dist = interest_min_dist.min(dist);
            }
            let state = membership.get(&key).copied().unwrap_or_default();
            let now_in = region.aoi.in_range(state.was_in, dist);
            let (verb, next) = aoi_transition(state, now_in, region.aoi.grace_ticks());
            if matches!(verb, Some(DemandVerb::SpinUp | DemandVerb::KeepAlive)) {
                now_demanded = true;
            }
            // THE VERDICT rides the SAME per-observer band membership as the demand (no new AoI
            // math, HR3): a child is ADMITTED for this observer iff it is in-AoI this tick
            // (`next_in` — the grace latch holds it true across a passed realm's whole grace
            // window, so no flicker). Render and demand read the SAME transition on the SAME
            // band. Only a DOT observer (a client this shard hosts) has a window to be told; a
            // PROXY occupant (or held transient) runs the AoI/latch math above but names nobody
            // here — it warms the sibling (demand) without a mis-routed verdict. Bitwise `&` (not
            // `&&`): both operands are pure bools, and a short-circuit would leave the RHS a
            // region HR5 can never cover from the false-LHS side (the discipline
            // `AoiConfig::in_range` and every AoI fold here use).
            let next_in = next.is_some_and(|s| s.was_in);
            // Look horizon slice 3 (§3.4.5): ANY observer's in-band child joins the shared
            // verdict — the up-relay's interior forward gate.
            if next_in {
                fold_verdict.insert(region.realm);
            }
            if render_routes.contains_key(&o.id) & next_in {
                in_band.entry(o.id).or_default().insert(region.realm);
            }
            // The OCCUPIED CHILD's own in-range set (SL7: the child stands in for its occupants,
            // at the placement this shard authored for it) — the `Child`-scope window's verdict,
            // shared by every observer under that child.
            if let ObserverId::Child(_) = o.id
                && next_in
            {
                in_band.entry(o.id).or_default().insert(region.realm);
            }
            match next {
                Some(s) => {
                    membership.insert(key, s);
                }
                None => {
                    membership.remove(&key);
                }
            }
        }
        // ONE demand per child = the observer union (deterministic — independent of observer order).
        let verb = union_verb(was_demanded, now_demanded);
        if let Some(v) = verb {
            debug_assert!(
                v != DemandVerb::TearDown,
                "Step 2 never emits parent TearDown — the Step-3 closure is the sole kill authority (M-1)"
            );
            push_demand(
                outbox,
                config.orchestrator,
                own_coord,
                child_coord.clone(),
                realm_fence,
                v,
                tick,
                stats,
            );
        }
        // Lane cure (findings 0/43, up half) — EAGER admission pre-resolve: on the same cadence as the
        // parent head-read (one `head_reads_due`, HR3), read the directory head of every child this
        // shard is currently DEMANDING or holds a LIVE BIT for. Eager-for-demanded is what makes
        // fail-closed free: the head resolves DURING the spin-up boot window, so it is in hand before
        // the child's first bit arrives — zero added spin-up-to-visible latency — and the read count is
        // bounded by in-band children, never roster size. The lazy re-read armed by an unattested bit
        // (`retain_child_live`) is the re-home backstop (D-RLM-6 mechanism C). Bitwise `|`/`&` (HR5).
        if heads_due & (verb.is_some() | child_liveness.contains_key(&region.realm)) {
            outbox.push_flow(
                config.orchestrator,
                MsgClass::Saga,
                &InterShardFlow::Directory(DirectoryOp::HeadRead {
                    key: DirectoryKey::Realm(region.realm),
                }),
            );
        }
        // Look horizon slice 4 — THE INTEREST EMISSION (§2 ASK B / §3.4.2 coarse): is this
        // direct child's interior worth waking, judged from the placement this realm authors
        // and the child's boot-derived interior band? One byte, to the child's attested head,
        // on the AoI beat (plus the rising edge), an explicit 0 on the falling edge.
        emit_realm_interest(
            &region.interior_band,
            region.realm,
            &child_coord,
            child_nodes.0.get(&region.realm),
            interest_min_dist,
            bit_due,
            interest_latch,
            realm_fence,
            tick,
            outbox,
            stats,
        );
    }
    // Evict interest latches for children no longer on the roster (mirrors `retain_live`).
    interest_latch.retain(|r| live_realms.contains(r));
    // THE WINDOW LANE (§2.9 step 5): ship each open window its SL7 membership verdict — the fold
    // above, diffed per window against what THAT window was already told (ids only; the receiver
    // never re-derives AoI). Hysteresis, bands, grace and the demand path are untouched: this
    // reads the verdict, it never makes one.
    //
    // THIS IS THE WHOLE OF WHAT A PARENT SAYS ABOUT ITS CHILDREN'S VISIBILITY, and it is a list of
    // names. Two lanes used to leave from here as well — a per-dot push of the OUTLINES in band,
    // and a per-live-child reflect of the surroundings DOWN into that child's process. Both are
    // deleted (window lane Slice C2, minor 19). The first made a parent the author of its
    // children's look, which is SL3's own violation; the second was the one message that could
    // tell a realm about itself, and its SL1 self-placement filter retires WITH it by the owner's
    // Q3 amendment (2026-08-16, docs/design/window_lane.md §5 RULINGS) — no realm-inbound message
    // type carries a placement at all now, which is strictly stronger than filtering one lane.
    // Look horizon slice 3 (§3.4.5): publish this tick's shared verdict for the up-relay's
    // interior forward gate (the emitter reads it next tick — one tick of staleness the band's
    // own hysteresis absorbs).
    *shared_verdict = fold_verdict;
    emit_window_membership(windows, &render_routes, &in_band, outbox, stats);
    // THE OCCUPANT UP-RELAY IS GONE (Step 5 slice D), and what replaced it is already above: the ONE
    // occupancy bit (SL7 verbatim — emitted at this function's non-empty gate) is everything a parent
    // may know about who is inside, and the occupied-child observer fold is how it warms a
    // traveller's next neighbourhood without a single pose ever crossing a realm boundary again (SL2
    // restored on this lane). A dot mid-hand-off keeps its parent's interest through `speaks_for` —
    // it counts at the Empty gate, so the bit keeps beating while somebody is still leaving.
    //
    // Evict any (observer, child) pair no longer live (an observer that left OR a child dropped from
    // the roster) — the DRY primitive, keyed by `(ObserverId, RealmId)`.
    retain_live(membership, &live_keys);
}

/// The effective AoI distance from ONE occupant to `child_pos`: the LESSER of the live distance and the F7
/// predictive distance (`occ_pos + occ_vel·horizon_s`) — so a fast occupant demands spin-up BEFORE it
/// arrives (boot latency masked). A STATIC occupant has `vel == 0` ⇒ `pred == live` ⇒ no predictive term.
/// Straight-line (monomorphic, HR5); the cross-observer UNION now lives in [`aoi_decide`]/[`union_verb`].
///
/// BOTH ENDPOINTS ARE POSITIONS, and the ONE way to turn two positions into metres is `delta_m` — the
/// exact-integer subtraction, the same rule the containment decision was corrected to in a35c294. They
/// used to arrive here as bare metre triples, which silently discarded each one's whole-number part: from
/// a player's first input the integrator folds their position into that part, so every occupant measured
/// as standing at their realm's origin and the entire demand loop stopped depending on where anyone was.
pub(crate) fn occupant_child_dist(
    occ_pos: LatticePos,
    occ_vel: DVec3,
    child_pos: LatticePos,
    tier: Tier,
    horizon_s: f64,
) -> f64 {
    let to_child = child_pos.delta_m(occ_pos, tier);
    let live = to_child.length();
    let pred = (to_child - occ_vel * horizon_s).length();
    live.min(pred)
}

/// The child-level demand verb = the UNION over observers of the per-observer hysteresis machines.
/// `was` = any observer kept the child demanded at tick start; `now` = any observer keeps it demanded this
/// tick. SpinUp the tick a child FIRST becomes demanded by anyone, KeepAlive while sustained, nothing when
/// no observer wants it. The Step-3 reconciler folds SpinUp and KeepAlive IDENTICALLY (`refresh_demand`);
/// the split preserves the single-observer byte-shape (one occupant ⇒ exactly today's SpinUp→KeepAlive).
/// Monomorphic, every arm a covered region (HR5).
pub(crate) fn union_verb(was: bool, now: bool) -> Option<DemandVerb> {
    match (was, now) {
        (false, true) => Some(DemandVerb::SpinUp),
        (true, true) => Some(DemandVerb::KeepAlive),
        (_, false) => None,
    }
}

/// The per-child hysteresis + grace state machine (ALL of it monomorphic — each arm a covered region,
/// HR5). Returns `(verb_to_emit, next_state)`. Step 2 NEVER returns `TearDown`: a child leaving range
/// holds for `grace_ticks` (emitting `KeepAlive`), then drops its key and emits NOTHING — the Step-3
/// reconciler closure tears it down (M-1, REVISION 1 R2 supersedes the §2.2 pseudocode).
pub(crate) fn aoi_transition(
    state: AoiState,
    now_in: bool,
    grace_ticks: u32,
) -> (Option<DemandVerb>, Option<AoiState>) {
    match (state.was_in, now_in) {
        (false, true) => (
            Some(DemandVerb::SpinUp),
            Some(AoiState {
                was_in: true,
                grace_remaining: grace_ticks,
            }),
        ),
        (true, true) => (
            Some(DemandVerb::KeepAlive),
            Some(AoiState {
                was_in: true,
                grace_remaining: grace_ticks,
            }),
        ),
        (true, false) => {
            if state.grace_remaining > 0 {
                (
                    Some(DemandVerb::KeepAlive),
                    Some(AoiState {
                        was_in: true,
                        grace_remaining: state.grace_remaining - 1,
                    }),
                )
            } else {
                (None, None) // drop the key; NO TearDown
            }
        }
        (false, false) => (None, None),
    }
}

/// Emit ONE [`RealmDemand`] toward the orchestrator (the RLM emit seam). Rides `MsgClass::Saga` (Reliable)
/// so the side-effecting-flow guard passes; `push_flow` defaults `Ephemeral`, which is CORRECT — a
/// `RealmDemand` is `ReDriven` (self-heals on the next re-assertion), NOT producer-less-reliable, so the
/// durability guard passes.
///
/// THE ONE STRUCTURAL GATE ON WHAT A DEMAND MAY NAME (lane cure, finding 37 — L-3: a realm speaks about
/// itself or a direct child, never upward). SL7 allows a shard exactly two demand shapes: its OWN realm
/// (the Empty self-report) and a DIRECT CHILD (the AoI union). Anything else — above all a crossing
/// dest that is this shard's PARENT on an outward hand-off — is refused HERE, counted, one place (HR3).
/// The parent needs no upward demand to stay alive: while the hand-off latch stands the source still
/// speaks for the departing occupant (`speaks_for`), so it never reports `Empty`, arm B of
/// `desired_alive` holds it, and `ancestor_close` pulls its whole chain — measured by the return-
/// crossing gate, not argued. Monomorphic; bitwise `|` (both shape tests are pure and covered, HR5).
///
/// The wire field is named `parent_fence` but carries the EMITTER's authority fence: the PARENT's realm
/// fence for SpinUp/KeepAlive (proving authority over the child), the CHILD-shard's own realm fence for
/// Empty (its authority over its own emptiness). Step 3 keys the Empty idempotency on `parent_fence`
/// accordingly (`IdempotencyKey::FencedKey`).
#[allow(clippy::too_many_arguments)]
pub(crate) fn push_demand(
    outbox: &mut OutboundBox,
    orch: NodeId,
    own: &RealmCoord,
    child: RealmCoord,
    fence: Fence,
    verb: DemandVerb,
    tick: UniverseTick,
    stats: &mut StubStats,
) {
    let names_self = child == *own;
    let names_direct_child = child.parent().as_ref() == Some(own);
    if !(names_self | names_direct_child) {
        stats.demand_refused_not_own_or_child += 1;
        tracing::warn!(
            named = %child.lowered(),
            own = %own.lowered(),
            ?verb,
            "demand refused: a shard demands only itself or a direct child (SL7)",
        );
        return;
    }
    outbox.push_flow(
        orch,
        MsgClass::Saga,
        &InterShardFlow::RealmDemand(RealmDemand {
            child,
            parent_fence: fence,
            verb,
            universe_tick: tick,
        }),
    );
}

/// Step 5 slice A — upsert a direct child's SL7 occupancy bit. Mis-routes drop by the same
/// `lowered()` compare every up-lane uses (the child's PARENT link must be this realm); an UNATTESTED
/// sender drops fail-closed (findings 0/43: the directory head is the admission authority — the stored
/// `home` route is believed only once attested) and arms a lazy head re-read (D-RLM-6 mechanism C, the
/// re-home backstop; the eager cadence read normally resolves during the spin-up boot window); a stale
/// `(fence, at)` never regresses a fresher entry (a deposed incarnation's heartbeat is rejected);
/// `last_seen` is the local-tick TTL base every retained store in this file prunes on. This is what
/// bounds the ZOMBIE window deterministically: a deposed child still beating fails the head compare, its
/// `home` is never refreshed, and the entry TTLs out independent of the child.
#[allow(clippy::too_many_arguments)]
pub(crate) fn retain_child_live(
    store: &mut ChildLiveness,
    config: &StubConfig,
    cl: vd_wire::intershard::ChildLive,
    now: TickId,
    home: NodeId,
    child_nodes: &ChildRealmNodes,
    stats: &mut StubStats,
    outbox: &mut OutboundBox,
) {
    let parent_ok = cl.child.parent().map(|p| p.lowered()) == Some(config.realm);
    if !parent_ok {
        stats.child_live_misrouted += 1;
        return;
    }
    let child = cl.child.lowered();
    if child_nodes.0.get(&child) != Some(&home) {
        stats.child_live_unattested += 1;
        outbox.push_flow(
            config.orchestrator,
            MsgClass::Saga,
            &InterShardFlow::Directory(DirectoryOp::HeadRead {
                key: DirectoryKey::Realm(child),
            }),
        );
        return;
    }
    let fresh = store
        .0
        .get(&child)
        .is_none_or(|held| (cl.fence, cl.at) >= (held.fence, held.at));
    if !fresh {
        stats.child_live_stale += 1;
        return;
    }
    stats.child_live_received += 1;
    store.0.insert(
        child,
        ChildLiveEntry {
            home,
            fence: cl.fence,
            at: cl.at,
            last_seen: now,
        },
    );
}
