//! THE WINDOW LANE, shard side: what a realm STATES about itself to whoever is looking in.
//!
//! Owns: the window registry and its derived keep-alive TTL (a gateway that dies can never leak a
//! fan), the per-tick placement statements a realm authors for its direct children, its own
//! look/marker bodies, and the membership verdict each window is served as a diff.
//!
//! Does NOT own: how anything outside this realm looks. A realm draws ITSELF — the parent authors
//! WHERE a child is and the child authors HOW IT LOOKS (SL3) — so every row shipped here is either
//! this realm's own statement or a placement this realm authored, and never a fact about a
//! grandchild (that arrives sealed, through `relay`, and is forwarded unopened).

use super::{
    InBandVerdict, ObserverId, ParentRealmNode, Placements, RealmAuthority, RealmRegions,
    RelayHeld, RelayShip, StubConfig, StubStats, aoi_recheck_cadence, build_relay_interior,
    emit_window_relays, push_session_reply, ttl_alive, window_ttl_ticks,
};
use crate::io::{Durability, MsgClass};
use crate::runtime::{ClockSample, OutboundBox};
use bevy_ecs::prelude::{Res, ResMut, Resource};
use std::collections::{BTreeMap, BTreeSet};
use vd_core::frame::FramePlacement;
use vd_core::placement::PlacementBook;
use vd_core::pose::{FrameRef, RealmId};
use vd_core::{AccountId, Fence, NodeId, TickId};
use vd_wire::channels::RealmSnap;
use vd_wire::intershard::InterShardFlow;
use vd_wire::session_flow::{BodyStmt, ShardToGateway, WindowId, WindowScope};

/// THE WINDOW REGISTRY (Slice A — docs/design/window_lane.md §2.3/§2.9): the windows this shard
/// currently serves, keyed `(opened_by, window)` — the OPENER rides the key because a `WindowId`
/// is minted per SUBSCRIBER (monotone per gateway, the `SubId` discipline), so two gateways'
/// ids may collide and only the pair is unambiguous. Rows ship back to `opened_by` alone.
///
/// Consumed from [`GatewayToShard::WindowOpen`]/[`GatewayToShard::WindowClose`] (idempotent: a
/// duplicate open refreshes the TTL AND re-asserts every send-on-change lane (Slice D — see
/// [`OpenWindow::reset_baselines`]); a close is the polite fast path). The crash backstop is the
/// DERIVED keep-alive TTL ([`window_ttl_ticks`], 2 beats + 1 — owner law 3(a)): a window not
/// re-asserted within it is dropped by the emitter's per-tick prune, so a dead gateway can never
/// leak a fan. EMPTY by default ⇒ every existing rig is byte-identical (no window, no emission).
#[derive(Resource, Debug, Default)]
pub struct OpenWindows(pub(crate) BTreeMap<(NodeId, WindowId), OpenWindow>);

/// One open window's shard-side state: its scope, its keep-alive freshness, and the two
/// send-on-change baselines its reliable lanes diff against (bodies + the SL7 membership
/// verdict). The baselines reset with the window — a re-opened window is re-served its full
/// body set and full verdict, exactly what a fresh subscriber needs.
#[derive(Debug, PartialEq)]
pub(crate) struct OpenWindow {
    pub(crate) scope: WindowScope,
    /// The holder's LOCAL tick of the last `WindowOpen` (open or keep-alive re-assert).
    pub(crate) last_refresh: TickId,
    /// Send-on-change: a DIGEST of the last [`ShardToGateway::WindowBody`] bag shipped per subject
    /// (the own look under this realm's id, one marker per direct child — disjoint keys).
    ///
    /// ★ A DIGEST SINCE S10, not the whole bag — see [`statement_digest`] for the cost this removes
    /// and for the collision argument, which the relay baseline below already relies on.
    sent_bodies: BTreeMap<RealmId, u64>,
    /// Send-on-change: the last SL7 membership verdict shipped (ids only).
    membership_sent: BTreeSet<RealmId>,
    /// Send-on-change: the (child fence, sealed bytes) last FORWARDED per live child on the Q2
    /// relay leg (`ShardToGateway::WindowRelayed`). Resets with the window — a re-opened window
    /// is re-served everything currently held, exactly what a fresh subscriber needs.
    /// The per-(window, child) relay send-on-change baseline as an 8-BYTE DIGEST over the held
    /// entry's (fence, own sealed statements) — look_horizon.md §5.4: the digest re-keys the
    /// baseline without storing or memcmp'ing the whole blob per tick, and when the slice-3
    /// interior forward lands, the interior field JOINS the digest input, so a grandchild's
    /// change with the child's own statements unchanged can never be invisible to
    /// send-on-change (the silent staleness bug §5.4 names).
    pub(crate) sent_relays: BTreeMap<RealmId, u64>,
    /// Send-on-change: a DIGEST of the STATIC roster last shipped (slice S10). `None` before the first
    /// send and after every keep-alive re-assert. One `u64` for the whole set, because the set is sent
    /// whole or not at all — there is no per-row diff to keep.
    sent_static: Option<u64>,
}

impl OpenWindow {
    pub(crate) fn opened(scope: WindowScope, now: TickId) -> OpenWindow {
        OpenWindow {
            scope,
            last_refresh: now,
            sent_bodies: BTreeMap::new(),
            membership_sent: BTreeSet::new(),
            sent_relays: BTreeMap::new(),
            sent_static: None,
        }
    }

    /// Forget every send-on-change baseline that the gateway can LOSE, so this subscriber is served
    /// the full body set, the full membership verdict and every held relay batch again. Called on the
    /// derived keep-alive re-assert (window lane Slice D) — the beat that makes a gateway-side
    /// roster-loss TTL sound.
    ///
    /// ★ THE STATIC ROSTER IS NOT CLEARED HERE, and the difference is measured, not stylistic. The
    /// three lanes above expire at the gateway (`WindowIngest::prune_stale` walks `look_of`,
    /// `interior_admitted` and `relay_levels`), so re-serving them on the beat is what makes silence
    /// mean "the realm stopped speaking". **`static_rows` never expires** — it is whole-set
    /// replacement with no TTL — so clearing it bought nothing and cost the whole roster, twice a
    /// second, for ever. At the S12 census that is 28.4 MB/s per window on the reliable lane.
    ///
    /// The roster is instead compared against the digest the SUBSCRIBER states on its open
    /// (`GatewayToShard::WindowOpen::static_held`), because the subscriber is the only party that
    /// knows what it holds — the same lesson that deleted `SkyStatedTo` from the sky lane.
    fn reset_baselines(&mut self) {
        self.sent_bodies.clear();
        self.membership_sent.clear();
        self.sent_relays.clear();
    }
}

/// THE MARKER ROSTER (Slice A — docs/design/window_lane.md §2.2/§2.8, the owner-ruled R4 datum):
/// per DIRECT child, the photometric datum `(class_code, luma_lsun)` drawn from the child's own
/// generation stream (`vd-physics` `system_photometrics_for_config` → `marker_datum`), planted by
/// the BOOT the way the region forest and the motion roster are — the boot/config path, never a
/// wire-struct change, and never a `vd-physics` edge from this crate (the datum arrives as two
/// plain scalars). A child with no entry (a hand-placed walk body; a Ship until P8) simply GLOWS
/// NOT — since look_horizon.md slice 1 (Q2 APPROVED) it still states a point-of-light marker
/// carrying its circumscribed extent alone (the presence floor: body → marker, never body →
/// nothing), built by [`current_bodies`] through the ONE `vd_core::look::marker_bag` codec.
/// Since Slice C1 planets carry the REFLECTED draw (§1.1 item 3b — per direct child).
/// DEFAULT EMPTY ⇒ byte-identical rigs.
#[derive(Resource, Debug, Default)]
pub struct ChildLuma(pub BTreeMap<RealmId, (u8, f64)>);

/// THE WINDOW LANE's subscription open (Slice A — docs/design/window_lane.md §2.3): register the
/// window under `(opener, id)` — the opener is the transport sender, exactly the attestation
/// source every other gateway-lane arm trusts (`dot.gateway = from`). IDEMPOTENT per the wire
/// contract: a duplicate open of the SAME scope is the derived keep-alive — it refreshes the TTL
/// AND re-asserts every send-on-change lane (Slice D); an open REUSING a live id under a
/// DIFFERENT scope replaces the window whole (fresh
/// send-on-change baselines — the subscriber is served its full set again), because the id mint
/// is monotone per subscriber and a reuse is a new subscription, never a refresh.
pub(crate) fn on_window_open(
    windows: &mut OpenWindows,
    from: NodeId,
    window: WindowId,
    scope: WindowScope,
    static_held: Option<u64>,
    now: TickId,
    stats: &mut StubStats,
) {
    match windows.0.entry((from, window)) {
        std::collections::btree_map::Entry::Occupied(mut held) if held.get().scope == scope => {
            let w = held.get_mut();
            w.last_refresh = now;
            // THE KEEP-ALIVE IS ALSO THE RE-ASSERT (window lane Slice D — §2.3's "re-asserted on a
            // derived keepalive cadence", made real for the send-on-change lanes). Clearing the
            // baselines re-serves this subscriber its whole body set, its whole membership verdict
            // and every held relay batch on the beat the SUBSCRIBER chose, so the gateway's
            // roster-loss window (`WindowTuning::look_ttl_ticks`, two of ITS OWN beats + 1) is
            // sound with no cross-process cadence agreement: a statement that stops arriving means
            // the realm behind it stopped speaking, not that nothing changed. Without this a
            // static realm would state its look once, forever, and a torn-down realm's last look
            // would be indistinguishable from a live one's silence.
            w.reset_baselines();
            // ★ THE COUNTER COMPARISON (S11, owner-approved 2026-08-27). The subscriber states the
            // roster digest it holds; the shard adopts that as its baseline and re-sends only if the
            // roster has actually changed. `None` — a fresh or restarted gateway — is served
            // everything, which is what makes a re-open safe.
            //
            // ★ MEASURED, not argued. With this assignment removed, the test fails on the FIRST
            // keep-alive ("tick 2: a keep-alive must not re-ship an unchanged roster"). Note that
            // restoring the old `sent_static = None` inside `reset_baselines` does NOT reproduce the
            // defect on its own, because this line runs after it and overwrites it — which is exactly
            // why the first attempt at that proof passed and proved nothing.
            w.sent_static = static_held;
            stats.window_reasserted += 1;
        }
        std::collections::btree_map::Entry::Occupied(mut held) => {
            *held.get_mut() = OpenWindow::opened(scope, now);
            held.get_mut().sent_static = static_held;
            stats.windows_opened += 1;
            trace_window_opened(from, window, scope);
        }
        std::collections::btree_map::Entry::Vacant(fresh) => {
            fresh.insert(OpenWindow::opened(scope, now));
            stats.windows_opened += 1;
            trace_window_opened(from, window, scope);
        }
    }
}

/// ★ SAY WHO IS WATCHING, AND WHAT THEY ASKED FOR (2026-09-01).
///
/// Nothing anywhere recorded a window's SCOPE. A counter says four windows are open; it cannot say
/// whether one of them names your ship, and that is the whole difference between "the parent was
/// never asked" and "the parent answered and the answer was dropped". A live diagnosis stalled on
/// exactly that gap, twice in one evening.
///
/// ON THE OPEN ONLY, never on the keep-alive: a window is opened once and re-asserted twice a second
/// for as long as somebody looks, so logging the re-assert would drown the file it is meant to
/// explain.
fn trace_window_opened(from: NodeId, window: WindowId, scope: WindowScope) {
    match scope {
        WindowScope::Occupants => tracing::info!(
            gateway = from.0,
            window = window.0,
            "a window opened on THIS realm — somebody is standing inside me"
        ),
        WindowScope::Child(child) => tracing::info!(
            gateway = from.0,
            window = window.0,
            %child,
            "a window opened on a CHILD of mine — somebody is looking out from inside it"
        ),
    }
}

/// THE WINDOW LANE's subscription close (Slice A): the polite fast path — the DERIVED TTL is the
/// crash backstop. Closing an unknown window is a COUNTED no-op per the wire contract (a close
/// racing the TTL, or a malformed id — never a panic, never guessed at).
pub(crate) fn on_window_close(
    windows: &mut OpenWindows,
    from: NodeId,
    window: WindowId,
    stats: &mut StubStats,
) {
    if windows.0.remove(&(from, window)).is_none() {
        stats.window_close_unknown += 1;
    }
}

/// Drop every window whose keep-alive lapsed (the crash backstop — a dead gateway can never leak
/// a fan; chaos-pinned). Runs at the head of the per-tick emitter, BEFORE anything ships, so an
/// expired window leaks zero emissions past its TTL. Counted per drop.
fn prune_expired_windows(
    windows: &mut OpenWindows,
    config: &StubConfig,
    now: TickId,
    stats: &mut StubStats,
) {
    let ttl = window_ttl_ticks(config);
    let before = windows.0.len();
    windows.0.retain(|_, w| ttl_alive(w.last_refresh, now, ttl));
    stats.window_ttl_expired += (before - windows.0.len()) as u64;
}

/// THE HOP ROW's placement: the child's placement in THIS realm's frame, read off the head book
/// this realm authored — the same row the roster already carries for that child, stated once more
/// beside the level so the gateway's chain fold needs no roster search (owner ruling 2026-09-02 R1).
///
/// ★ NO INVERSION ANY MORE. This used to re-express THIS realm's origin in the child's frame, at the
/// child's step, and that number does not exist at galaxy scale: a galaxy's origin counted in a star
/// system's millimetre cells overflows every lattice, so a galaxy shard refused its own hop on every
/// tick and the observer chain stopped one level short of the sky. The parent's own number about
/// its child is always representable, because a parent contains its child.
pub(crate) fn hop_placement(child: FrameRef, book: &PlacementBook) -> Option<FramePlacement> {
    book.of(child)
}

#[allow(clippy::too_many_arguments)]
fn emit_window_frames(
    config: &StubConfig,
    clock: &ClockSample,
    realm_fence: Fence,
    regions: &RealmRegions,
    head: &PlacementBook,
    realms: &[RealmSnap],
    windows: &OpenWindows,
    stats: &mut StubStats,
    outbox: &mut OutboundBox,
) {
    for ((gateway, window), held) in &windows.0 {
        let hop = match held.scope {
            // The observer's-own-level window: an occupant already stands in this frame — no hop.
            WindowScope::Occupants => None,
            WindowScope::Child(child) => {
                // The hop row exists only for a DIRECT child this shard actually authors: a
                // subscriber naming a stranger gets nothing, counted, never a guess.
                let Some(region) = regions
                    .direct_children(config.realm)
                    .find(|r| r.realm == child)
                else {
                    stats.window_child_unrostered += 1;
                    tracing::warn!(
                        %child,
                        realm = %config.realm,
                        "Child-scope window names a realm this shard does not parent — no frame"
                    );
                    continue;
                };
                // ★ A SHIP IS NO LONGER REFUSED A WINDOW (2026-09-01). This lane excluded a ship and
                // counted it, because a ship had no lineage to serve a window by. It has one now, so a
                // subscriber may watch a ship exactly as it watches a planet — which is what has to be
                // true before a pilot can see their own hull.
                // The book was authored over this same child roster one expression above, so a
                // rostered child always has a row: the `None` arm IS the unrostered case, refused
                // and counted just above.
                match hop_placement(region.frame, head) {
                    Some(placement) => {
                        Some(Box::new(vd_wire::session_flow::HopRow { child, placement }))
                    }
                    None => {
                        // A rostered child with no row in the book this same tick authored: a
                        // defect in the authoring, never a quiet tick. Counted and said, once
                        // per keep-alive beat.
                        stats.window_hop_missing += 1;
                        if clock
                            .local_tick
                            .0
                            .is_multiple_of(aoi_recheck_cadence(config))
                        {
                            tracing::warn!(
                                %child,
                                realm = %config.realm,
                                frame = ?region.frame,
                                book_rows = head.rows().count(),
                                "the hop child has NO row in this tick's book — the frame is withheld"
                            );
                        }
                        continue;
                    }
                }
            }
        };
        // ★ UNDER THE DATAGRAM BUDGET, IN CHUNKS (2026-09-02). This frame rides the unreliable lane,
        // and a datagram over the path's budget is dropped by the transport — counted where no gate
        // reads, and said in a log nobody was reading. MEASURED on the home system: nine mover rows
        // plus the hop came to 1,420 bytes against 1,200, so EVERY frame of the hull's window was
        // dropped and the observer chain never had a stamp for the star system. The rows are split
        // exactly as the entity snapshots are; the hop rides EVERY chunk (one datagram may arrive
        // without the others) and the gateway merges the chunks of one stamp. A tick with no rows
        // still sends one frame: the stamp itself is what the composer aligns chains on.
        let mut chunks =
            vd_wire::channels::partition_realms(realms, config.snapshot_datagram_budget);
        if chunks.is_empty() {
            chunks.push(Vec::new());
        }
        for rows in chunks {
            let frame = ShardToGateway::WindowFrame {
                realm_fence,
                window: *window,
                at: clock.universe_tick,
                hop: hop.clone(),
                rows,
            };
            let bytes = crate::io::bytes(
                postcard::to_allocvec(&frame).expect("closed wire enums serialize infallibly"),
            );
            outbox.0.push((
                *gateway,
                MsgClass::RealmSnapshot,
                bytes,
                Durability::Ephemeral,
            ));
            stats.window_frames_sent += 1;
        }
        stats.window_frame_rows_sent += realms.len() as u64;
    }
}

// THE SHARD STATES NO SKY (S11, owner ruling 2026-08-27 — "we're passing the Galaxy just once over
// reliable lane"). `StarCatalogue` and `SkyRequests` lived here. A shard folded its sky from the
// realms IT booted, so the sky a player received depended on which shard they were subscribed to.
// The GATEWAY holds the galaxy now: see `gateway::window_lane::emit_sky`.

/// ★ THE CHILDREN ONE WINDOW CARRIES (owner decision 3, 2026-09-02 — R10): the children in the
/// window's range as the fold last stated them, every mover (a planet keeps its dot until reach gives
/// it a brightness radius), and the hop child (the chain climbs through it). `None` — everything —
/// when this realm has no live band: without a band nothing is out of range, which keeps the
/// walk-scale fixtures byte-identical.
///
/// A child in range runs and draws itself, so the gateway needs its placement. A child out of range
/// sleeps and needs no row; if it is a star, the star field already shows it. MEASURED before this
/// existed: the galaxy shipped 233,220 rows and 233,220 markers on every keep-alive, 22 MB, forever,
/// and the hop the sky needed was shed with the rest.
fn window_admitted(
    regions: &RealmRegions,
    held: &OpenWindow,
    movers: &BTreeSet<RealmId>,
) -> Option<BTreeSet<RealmId>> {
    if !regions.aoi_live() {
        return None;
    }
    let mut set = held.membership_sent.clone();
    set.extend(movers.iter().copied());
    if let WindowScope::Child(child) = held.scope {
        set.insert(child);
    }
    Some(set)
}

/// THE BODIES AND THE STATIC ROSTERS, per window (Slice A + C1; owner decision 3, 2026-09-02 — R10):
/// runs AFTER the range fold of the tick, so a freshly opened window is served the children in its
/// range on the tick it opened. Each window gets its own body set and its own roster, send-on-change
/// on both (a digest per subject; a digest over the whole roster).
#[allow(clippy::too_many_arguments)]
pub(crate) fn emit_window_rosters(
    config: Res<StubConfig>,
    clock: Res<ClockSample>,
    authority: Res<RealmAuthority>,
    regions: Res<RealmRegions>,
    placements: Res<Placements>,
    mut stats: ResMut<StubStats>,
    mut outbox: ResMut<OutboundBox>,
    lane: RosterLane,
) {
    let (mut windows, child_luma, driven) = lane;
    let Some(realm_fence) = authority.0 else {
        return;
    };
    if windows.0.is_empty() {
        return;
    }
    let head = placements
        .0
        .head(config.realm)
        .expect("the writer authors every held anchor before the feed runs");
    let movers: BTreeSet<RealmId> = regions.moving_children_of(config.realm, &driven);
    for ((gateway, window), held) in &mut windows.0 {
        let admitted = window_admitted(&regions, held, &movers);
        for (subject, stmt) in current_bodies(&config, &regions, &child_luma.0, admitted.as_ref()) {
            let bag = match &stmt {
                BodyStmt::SelfLook { bag } => bag,
                BodyStmt::Marker { luma } => luma,
            };
            let digest = crate::stub::relay::statement_digest(bag);
            if held.sent_bodies.get(&subject) == Some(&digest) {
                continue; // unchanged — send-on-change holds its tongue
            }
            push_session_reply(
                outbox.as_mut(),
                *gateway,
                &ShardToGateway::WindowBody {
                    realm_fence,
                    window: *window,
                    subject,
                    stmt,
                    authored_at: clock.universe_tick,
                },
            );
            held.sent_bodies.insert(subject, digest);
            stats.window_bodies_sent += 1;
        }
        // The static roster of this window: the admitted set less the movers (each one lookup),
        // or — with no live band — every static child, as before.
        let rows: Vec<RealmSnap> = match &admitted {
            Some(set) => regions.snaps_for(
                config.realm,
                head,
                set.iter().copied().filter(|r| !movers.contains(r)),
            ),
            None => regions
                .authored_realm_snaps(config.realm, head)
                .into_iter()
                .filter(|r| !movers.contains(&r.realm))
                .collect(),
        };
        let encoded = postcard::to_allocvec(&rows).expect("closed wire enums serialize infallibly");
        let digest = crate::stub::relay::statement_digest(&encoded);
        if held.sent_static == Some(digest) {
            continue; // unchanged — a static roster says nothing twice
        }
        push_session_reply(
            outbox.as_mut(),
            *gateway,
            &ShardToGateway::WindowStaticRows {
                realm_fence,
                window: *window,
                authored_at: clock.universe_tick,
                rows,
            },
        );
        held.sent_static = Some(digest);
        stats.window_static_rows_sent += 1;
    }
}

/// The realm's CURRENT self-authored body set — its OWN look (from its boot extent via
/// [`RealmRegions::own_shape`], never any parent's row about it — SL3) plus one point-of-light
/// marker per DIRECT child, GLOWING OR NOT (look_horizon.md slice 1, Q2 APPROVED — the presence
/// floor): a glowing child's bag carries its photometric datum plus its circumscribed extent, a
/// non-glowing child's the extent ALONE — one radius, nothing else (the one-radius law: a bound
/// is a promise about space; a look is a statement about appearance; the child's own picture
/// supersedes the whole bag the instant it runs). The extent is the SAME
/// `circumscribed_extent()` the parent's SL7 proxy fold already reads. Shared by
/// [`emit_window_bodies`] and the Q2 relay ship, so the direct lane and the relayed lane can
/// never state different bodies.
pub(crate) fn current_bodies(
    config: &StubConfig,
    regions: &RealmRegions,
    child_luma: &BTreeMap<RealmId, (u8, f64)>,
    admitted: Option<&BTreeSet<RealmId>>,
) -> Vec<(RealmId, BodyStmt)> {
    let mut bodies: Vec<(RealmId, BodyStmt)> = Vec::new();
    // THE BOUND/LOOK SPLIT (real-scale design §3.0): what a realm STATES about its appearance
    // is its LOOK — the bound is a containment promise and is never drawn. A look-less realm
    // (the ambient Universe/Galaxy) states nothing: undrawable structurally, not by a cull.
    if let Some(look) = regions.own_look(config.realm) {
        bodies.push((
            config.realm,
            BodyStmt::SelfLook {
                // THE STAR-LOOK EXTENSION SEAM (ruling C): the running realm's own bag carries
                // its photometric datum beside the outline when it has one — the star (and its
                // system) keeps its colour through the wake handover. Future star parameters
                // are future tags on this same bag (skip-unknown makes them free).
                bag: vd_core::look::self_look_bag(&look, child_luma.get(&config.realm).copied()),
            },
        ));
    }
    // The markers: over the admitted set when there is one (each a lookup), else every direct
    // child — the roster walk only when nothing is out of range.
    let mut marker = |region: &vd_core::geometry::RealmRegion| {
        let Some(look) = region.look else { return };
        bodies.push((
            region.realm,
            BodyStmt::Marker {
                luma: vd_core::look::marker_bag(
                    child_luma.get(&region.realm).copied(),
                    look.circumscribed_extent(),
                ),
            },
        ));
    };
    match admitted {
        Some(set) => {
            for realm in set {
                if let Some(region) = regions.direct_child(config.realm, *realm) {
                    marker(region);
                }
            }
        }
        None => {
            for region in regions.direct_children(config.realm) {
                marker(region);
            }
        }
    }
    bodies
}

/// THE WINDOW LANE's per-tick emitter (`docs/design/window_lane.md` §2.9): a realm STATES what it
/// lawfully owns, once per tick, to whoever holds a window on it. No realm authority ⇒ silent.
///
/// It states four things and nothing else: where its DIRECT CHILDREN are, in its own frame
/// ([`ShardToGateway::WindowFrame`], with the pre-inverted hop row on a `Child`-scope window); what
/// IT looks like and one photometric marker per dormant child ([`ShardToGateway::WindowBody`],
/// send-on-change); its SL7 membership verdict as ids ([`ShardToGateway::WindowMembership`], from
/// the AoI pass); and, one hop up, its own statements SEALED for its parent to forward verbatim
/// ([`InterShardFlow::WindowRelay`] — the Q2 ruling). Every statement goes to the connection plane.
///
/// WHAT IT NO LONGER DOES (window lane Slice C2, minor 19 — the deletion): it does not cascade
/// anyone's rows DOWN into a child's process, it does not ship its rows UP for a parent to restate
/// and re-fan, it does not fan a child's held interior, and it does not emit the old opaque
/// per-tick realm datagram (`ShardToGateway::RealmFrame`) — the `Occupants` window frame subsumes
/// that emit (§2.9 step 3), and the composed feed has been the client's one scene author since the
/// minor-18 flag day. Nothing of the picture passes between realms any more.
#[allow(clippy::too_many_arguments)]
pub(crate) fn emit_realm_frames(
    config: Res<StubConfig>,
    clock: Res<ClockSample>,
    authority: Res<RealmAuthority>,
    regions: Res<RealmRegions>,
    placements: Res<Placements>,
    // THE STAR CATALOGUE this shard states (S11) — planted by the boot, shipped here, never derived.
    // The resolved parent the Q2 relay ships to (`None` at a root shard / before the first parent
    // Head reply ⇒ nothing goes up this tick).
    parent_node: Res<ParentRealmNode>,
    mut stats: ResMut<StubStats>,
    mut outbox: ResMut<OutboundBox>,
    // THE WINDOW LANE (Slice A + C1) — the registry this emitter serves, the boot-planted marker
    // roster, and the Q2 relay's two stores (the child-side ship memo + the parent-side holder),
    // bundled as ONE tuple `SystemParam` (bevy's 16-param ceiling — the established mechanical
    // arity fix). Destructured below.
    window_lane: WindowLaneStores,
) {
    let (mut windows, child_luma, mut relay_ship, mut relay_held, verdict, driven) = window_lane;
    let Some(realm_fence) = authority.0 else {
        return;
    };

    // The authored child rows — one per DIRECT child, static and moving alike (owner Q3; the movers-
    // only filter was the last rival motion test). EMPTY only for a childless leaf. Each row is the
    // child's placement in THIS shard's own frame, shipped as authored — the shard composes nothing
    // and never learns where it sits.
    // The HEAD book — authored this same tick by the one writer (`author_placements` runs first on
    // the schedule and this system shares its `has_synced` gate), so the invariant is stated, never
    // hoped. Every statement below reads THESE rows.
    let head = placements
        .0
        .head(config.realm)
        .expect("the writer authors every held anchor before the feed runs");
    // ★ THE MOVERS, BY LOOKUP (owner ruling 2026-09-02 R8 item 1; SL9): the orbiting and the driven
    // are a named set, and their rows are each one lookup — the roster is never walked here.
    let movers: BTreeSet<RealmId> = regions.moving_children_of(config.realm, &driven);
    let mover_rows: Vec<RealmSnap> = regions.snaps_for(config.realm, head, movers.iter().copied());

    prune_expired_windows(&mut windows, &config, clock.local_tick, &mut stats);
    stats.windows_open = windows.0.len() as u64;
    if !windows.0.is_empty() {
        emit_window_frames(
            &config,
            &clock,
            realm_fence,
            &regions,
            head,
            &mover_rows,
            &windows,
            &mut stats,
            &mut outbox,
        );
        // The bodies and the static rosters ship from `emit_window_rosters`, which runs AFTER the
        // range fold of this same tick, so a window opened this tick is served the children in
        // its range on this tick (owner decision 3, 2026-09-02 — R10).
    }
    // ★ NO SKY IS STATED HERE ANY MORE (S11, owner ruling 2026-08-27 — "we're passing the Galaxy just
    // once over reliable lane"). A shard folded its sky from the realms IT booted, so the sky a player
    // received depended on which shard they were subscribed to: a home star system states ONE star, its
    // own, and a player never draws their own star because they are standing inside it. MEASURED on a
    // dual cluster — the galaxy shard held 3, the client held 1, and drew 0.
    //
    // "ONCE" means one sky, the same for everybody. The GATEWAY holds it now: it is the one party that
    // sees every session and the whole forest. See `gateway::window_lane::emit_sky`.
    // THE Q2 RELAY FORWARD (Slice C1 — the parent half; prunes the holder even with zero windows
    // so a reaped child's stale seal cannot linger past the derived TTL).
    emit_window_relays(
        &config,
        &clock,
        realm_fence,
        &mut relay_held,
        &mut windows,
        &mut stats,
        &mut outbox,
    );
    // THE Q2 RELAY SHIP (Slice C1 — the child half; the tombstoned shape lane's successor,
    // owner-approved 2026-08-16: docs/design/owner_decisions_2026-08-15.md addendum +
    // docs/design/window_lane.md §5 RULINGS). This realm's VERBATIM self-authored statements —
    // its authored interior level (also the roster its markers are vouched against), its own
    // look, its markers — SEALED (`seal_relay_statements`) and shipped ONE hop up for the parent
    // to hold unopened and forward. Send-on-change on the (parent, LEVEL ROWS + bodies) — the
    // level is part of the statement batch, so a realm with ORBITING children ships every tick
    // (the rate of the up-observation lane this relay replaced; a static interior stays quiet) —
    // re-asserted on the AoI cadence (a restarted parent holds relays only in RAM); the parent
    // resolving IS the relay-subscription moment (the memo never matches a fresh parent). A
    // childless LEAF still owes its look to outside observers, so this runs unconditionally.
    let relay_due =
        crate::directory::due_this_tick(aoi_recheck_cadence(&config), clock.local_tick.0);
    if let Some(parent) = parent_node.0 {
        // ★ THE RELAY CARRIES THE CHILDREN IN RANGE (owner decision 3, 2026-09-02 — R10): the fold
        // verdict of this realm (the union over every looker), plus every mover, exactly as a window
        // does — a galaxy relaying 233,220 sleeping systems to the universe is the same flood by
        // another road. With no live band nothing is out of range, as today.
        let relay_admitted: Option<BTreeSet<RealmId>> = regions.aoi_live().then(|| {
            let mut set = verdict.0.clone();
            set.extend(movers.iter().copied());
            set
        });
        let realms: Vec<RealmSnap> = match &relay_admitted {
            Some(set) => regions.snaps_for(config.realm, head, set.iter().copied()),
            None => regions.authored_realm_snaps(config.realm, head),
        };
        let bodies = current_bodies(&config, &regions, &child_luma.0, relay_admitted.as_ref());
        if !bodies.is_empty() {
            // THE SEALED INTERIOR FORWARD (look horizon slice 3, owner-approved 2026-08-17 —
            // look_horizon.md RULINGS + §2 ASK A): each held child batch's OWN half, verbatim,
            // membership-gated by this realm's own in-band verdict (§3.4.5). The held INTERIOR
            // halves never join — `build_relay_interior` reads only `own`, and the carried type
            // has no field a third level could ride in. The interior joins the ship
            // fingerprint, so a grandchild's picture change re-ships even when this realm's
            // own statements are unchanged (§5.4's staleness law, up-leg half).
            let interior = build_relay_interior(&relay_held, &verdict.0);
            let fingerprint = postcard::to_allocvec(&(&realms, &bodies, &interior))
                .expect("closed wire enums serialize infallibly");
            let current = (parent, fingerprint);
            if relay_due || relay_ship.0.as_ref() != Some(&current) {
                let mut statements = vec![vd_wire::session_flow::RelayedStatement::Level {
                    at: clock.universe_tick,
                    rows: realms.clone(),
                }];
                statements.extend(bodies.into_iter().map(|(subject, stmt)| {
                    vd_wire::session_flow::RelayedStatement::Body {
                        subject,
                        stmt,
                        authored_at: clock.universe_tick,
                    }
                }));
                outbox.push_flow(
                    parent,
                    MsgClass::Saga,
                    &InterShardFlow::WindowRelay(vd_wire::intershard::WindowRelay {
                        child: config.own_coord.clone(),
                        realm_fence,
                        own: vd_wire::session_flow::seal_relay_statements(&statements),
                        interior,
                    }),
                );
                stats.window_relays_sent += 1;
                relay_ship.0 = Some(current);
            }
        }
    }
}

/// THE WINDOW LANE's emitter stores bundled into ONE tuple `SystemParam` (bevy's 16-param
/// ceiling; a mechanical arity fix, destructured right back inside [`emit_realm_frames`]): the
/// registry, the boot-planted marker roster, the Q2 relay's two stores (the child-side ship
/// memo + the parent-side holder), and (look horizon slice 3) the read-only shared in-band
/// verdict the interior forward gates on (§3.4.5 — written by the AoI fold, read here).
/// The roster emitter's stores, bundled as ONE tuple `SystemParam` (bevy's 16-param ceiling).
type RosterLane<'w> = (
    ResMut<'w, OpenWindows>,
    Res<'w, ChildLuma>,
    Res<'w, crate::stub::drive::DrivenChildren>,
);

type WindowLaneStores<'w> = (
    ResMut<'w, OpenWindows>,
    Res<'w, ChildLuma>,
    ResMut<'w, RelayShip>,
    ResMut<'w, RelayHeld>,
    Res<'w, InBandVerdict>,
    // D-MOVE-2 — the driven children this realm holds. The lane split below must ask it, or a ship
    // under thrust is filed as STANDING STILL. See the split for what that costs.
    Res<'w, crate::stub::drive::DrivenChildren>,
);

/// One window's SL7 membership VERDICT (`docs/design/window_lane.md` §2.2/§2.9): for an
/// [`WindowScope::Occupants`] window, the union over the opener's OWN dots of their in-band
/// direct children (the per-dot fold — the same per-account verdict the render delta ships,
/// filtered to the one gateway the window belongs to); for a [`WindowScope::Child`] window, the
/// occupied child's own in-band set (the SL7 proxy fold, ids only). Monomorphic —
/// every arm a covered region (HR5).
fn window_verdict(
    scope: WindowScope,
    opener: NodeId,
    render_routes: &BTreeMap<ObserverId, (AccountId, NodeId)>,
    in_band: &BTreeMap<ObserverId, BTreeSet<RealmId>>,
) -> BTreeSet<RealmId> {
    match scope {
        WindowScope::Occupants => render_routes
            .iter()
            .filter(|(_, (_, gateway))| *gateway == opener)
            .filter_map(|(obs, _)| in_band.get(obs))
            .flat_map(|ids| ids.iter().copied())
            .collect(),
        WindowScope::Child(child) => in_band
            .get(&ObserverId::Child(child))
            .cloned()
            .unwrap_or_default(),
    }
}

/// Ship each open window its membership DIFF (`added`/`removed` vs what THAT window already
/// holds), reliable on the session-reply lane — send-on-change, so a stable verdict ships
/// nothing and a fresh window is served its full set as one `added` batch. The scope note the
/// wire contract states holds here by construction: the verdict gates BODIES and interiors
/// downstream, never the placement/marker rows (those always ship the full roster).
pub(crate) fn emit_window_membership(
    windows: &mut OpenWindows,
    render_routes: &BTreeMap<ObserverId, (AccountId, NodeId)>,
    in_band: &BTreeMap<ObserverId, BTreeSet<RealmId>>,
    outbox: &mut OutboundBox,
    stats: &mut StubStats,
) {
    for ((gateway, window), held) in &mut windows.0 {
        let verdict = window_verdict(held.scope, *gateway, render_routes, in_band);
        let added: Vec<RealmId> = verdict.difference(&held.membership_sent).copied().collect();
        let removed: Vec<RealmId> = held.membership_sent.difference(&verdict).copied().collect();
        if !added.is_empty() | !removed.is_empty() {
            push_session_reply(
                outbox,
                *gateway,
                &ShardToGateway::WindowMembership {
                    window: *window,
                    added,
                    removed,
                },
            );
            held.membership_sent = verdict;
            stats.window_memberships_sent += 1;
        }
    }
}
