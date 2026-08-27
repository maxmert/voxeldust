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
    emit_window_relays, push_session_reply, region_level, ttl_alive, window_ttl_ticks,
};
use crate::io::{Durability, MsgClass};
use crate::runtime::{ClockSample, OutboundBox};
use bevy_ecs::prelude::{Res, ResMut, Resource};
use std::collections::{BTreeMap, BTreeSet};
use vd_core::frame::{FrameError, FramePlacement, transfer_frame};
use vd_core::glam::DVec3;
use vd_core::placement::PlacementBook;
use vd_core::pose::{FrameRef, RealmId, StampedPose};
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

    /// Forget every send-on-change baseline, so this subscriber is served the full body set, the
    /// full membership verdict and every held relay batch again. Called on the derived keep-alive
    /// re-assert (window lane Slice D) — the beat that makes a gateway-side roster-loss TTL sound.
    fn reset_baselines(&mut self) {
        self.sent_bodies.clear();
        self.membership_sent.clear();
        self.sent_relays.clear();
        self.sent_static = None;
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
            stats.window_reasserted += 1;
        }
        std::collections::btree_map::Entry::Occupied(mut held) => {
            *held.get_mut() = OpenWindow::opened(scope, now);
            stats.windows_opened += 1;
        }
        std::collections::btree_map::Entry::Vacant(fresh) => {
            fresh.insert(OpenWindow::opened(scope, now));
            stats.windows_opened += 1;
        }
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

/// THE HOP-ROW INVERSION AT THE AUTHOR (`docs/design/window_lane.md` §2.2, R1): "my frame
/// expressed in the child's frame at the book's instant" — computed THROUGH [`transfer_frame`]
/// itself, by re-expressing the author's own frame ORIGIN (its own body's seat,
/// INV-BODY-AT-ORIGIN) into the child's frame. One arithmetic, the frame core's own: the f64
/// path, the integer cell subtraction and the rotated-cross-cell REFUSAL are all inherited, never
/// a second implementation (the refusal is exactly the design's "inherits `transfer_frame`'s
/// refusal semantics"). The one field the pose transform does not carry is the angular term: the
/// parent's spin seen from the child is the child's own spin, reversed and re-axed into the
/// child's axes — `-(q⁻¹·ω)`, where `q⁻¹` IS the transferred orientation (the origin pose rides
/// in with the identity, so what comes back is exactly the child orientation's inverse).
pub(crate) fn invert_hop_placement(
    own: FrameRef,
    child: FrameRef,
    book: &PlacementBook,
) -> Result<FramePlacement, FrameError> {
    let child_at = book.of(child).ok_or(FrameError::UnknownDestFrame)?;
    let origin = StampedPose::at_rest(own, DVec3::ZERO, book.at());
    let inv = transfer_frame(&origin, child, book)?;
    Ok(FramePlacement {
        origin_cell: inv.pos.cell(),
        origin: inv.pos.offset(),
        velocity: inv.vel,
        orientation: inv.orient,
        angular_velocity: -(inv.orient * child_at.angular_velocity),
    })
}

/// THE WINDOW LANE's per-tick placement statements (`docs/design/window_lane.md` §2.9 steps 1–3):
/// the authored rows were built ONCE this tick (the caller's `realms` — the same Vec every old
/// lane reads); per open window they ship as ONE [`ShardToGateway::WindowFrame`] stamped at the
/// one universe tick, `hop: None` on an [`WindowScope::Occupants`] window, the pre-inverted
/// [`HopRow`] on a [`WindowScope::Child`] window. FireAndForget/Unreliable — the realm-lane
/// datagram class, full-state latest-wins (owner law 3(b)); a lost frame self-heals next tick.
/// Kind-BLIND: no realm-kind test anywhere on this path (HR3/HR4).
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
    let own_frame = regions.own_frame(config.realm);
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
                // A Ship child has no lineage coord to serve a window by until P8 (D-SHIP-1):
                // excluded, counted, never a panic — see `region_level`.
                if region_level(region).is_none() {
                    stats.ship_child_regions_excluded += 1;
                    continue;
                }
                match invert_hop_placement(own_frame, region.frame, head) {
                    Ok(inv) => Some(Box::new(vd_wire::session_flow::HopRow { child, inv })),
                    // The frame core's refusal (rotated frame across integer cells — owed with
                    // P10 cell math): dropped + counted, never shipped with folded numbers.
                    Err(refusal) => {
                        stats.window_hop_refused += 1;
                        tracing::warn!(
                            %child,
                            realm = %config.realm,
                            ?refusal,
                            "hop-row inversion refused — the window frame is withheld this tick"
                        );
                        continue;
                    }
                }
            }
        };
        let frame = ShardToGateway::WindowFrame {
            realm_fence,
            window: *window,
            at: clock.universe_tick,
            hop,
            rows: realms.to_vec(),
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
        stats.window_frame_rows_sent += realms.len() as u64;
    }
}

/// THE WINDOW LANE's look/marker statements (`docs/design/window_lane.md` §2.9 step 4):
/// send-on-change + on-open, NEVER per-tick. The realm's CURRENT body set is its OWN look —
/// derived from what it knows about ITSELF (its boot-config extent via [`RealmRegions::own_shape`],
/// never any parent's row about it — SL3) — plus one photometric marker per DIRECT child the
/// boot roster carries a luma bag for (the owner-ruled R4 datum; absence of a bag is absence of
/// data, never a default). Each window diffs the set against what IT was already sent, so a fresh
/// window is served everything once and a static world then ships nothing UNTIL its next keep-alive
/// re-assert, which clears the baseline on the SUBSCRIBER's own beat (Slice D — the gateway's
/// roster-loss window is derived from that beat, so silence past it means the realm behind a
/// statement stopped speaking, not that nothing changed). ReDriven/reliable —
/// the session-reply lane (a lost look is an invisible realm at exactly the no-flicker moment).
/// THE WINDOW LANE's STATIC ROSTER (slice S10; owner-approved 2026-08-27): the author's direct
/// children that DO NOT MOVE, shipped on the RELIABLE session lane, send-on-change — which for a realm
/// whose children are static means exactly ONCE, plus the keep-alive re-assert as the repair.
///
/// ★ WHY THIS IS NOT ON THE PER-TICK FRAME. The frame is latest-wins and UNRELIABLE, and that is right
/// for a mover: a dropped row heals next tick and a resend of a stale position would be worse than the
/// loss. It is exactly wrong for a star, which never moves — repeating it costs 285 MB/s per subscriber
/// at the target census (150,000 rows × 95 bytes × 20 Hz) for bytes that are identical every time, and
/// the 14.2 MB message does not fit a datagram at all.
///
/// Send-on-change was REFUSED on the frame for a good reason — over an unreliable lane a dropped row
/// loses a star forever and a late joiner is never served. **Reliability is what makes send-on-change
/// lawful here**, which is why this lane and this cadence had to arrive together.
///
/// The baseline is a DIGEST of the whole set, so an unchanged roster costs one hash and no bytes; see
/// [`crate::stub::relay::statement_digest`] for the collision argument (a miss defers one re-send to the
/// keep-alive beat, never a wrong byte).
fn emit_window_static_rows(
    clock: &ClockSample,
    realm_fence: Fence,
    statics: &[RealmSnap],
    windows: &mut OpenWindows,
    stats: &mut StubStats,
    outbox: &mut OutboundBox,
) {
    let encoded = postcard::to_allocvec(statics).expect("closed wire enums serialize infallibly");
    let digest = crate::stub::relay::statement_digest(&encoded);
    for ((gateway, window), held) in &mut windows.0 {
        if held.sent_static == Some(digest) {
            continue; // unchanged — a static roster says nothing twice
        }
        push_session_reply(
            outbox,
            *gateway,
            &ShardToGateway::WindowStaticRows {
                realm_fence,
                window: *window,
                authored_at: clock.universe_tick,
                rows: statics.to_vec(),
            },
        );
        held.sent_static = Some(digest);
        stats.window_static_rows_sent += 1;
    }
}

#[allow(clippy::too_many_arguments)]
fn emit_window_bodies(
    config: &StubConfig,
    clock: &ClockSample,
    realm_fence: Fence,
    regions: &RealmRegions,
    child_luma: &BTreeMap<RealmId, (u8, f64)>,
    windows: &mut OpenWindows,
    stats: &mut StubStats,
    outbox: &mut OutboundBox,
) {
    // The current body set, built once per tick (subjects are disjoint by construction: a realm
    // is never its own direct child) — the ONE authoring expression the direct lane and the Q2
    // relay ship share.
    let bodies = current_bodies(config, regions, child_luma);
    for ((gateway, window), held) in &mut windows.0 {
        for (subject, stmt) in &bodies {
            let bag = match stmt {
                BodyStmt::SelfLook { bag } => bag,
                BodyStmt::Marker { luma } => luma,
            };
            let digest = crate::stub::relay::statement_digest(bag);
            if held.sent_bodies.get(subject) == Some(&digest) {
                continue; // unchanged — send-on-change holds its tongue
            }
            push_session_reply(
                outbox,
                *gateway,
                &ShardToGateway::WindowBody {
                    realm_fence,
                    window: *window,
                    subject: *subject,
                    stmt: stmt.clone(),
                    authored_at: clock.universe_tick,
                },
            );
            held.sent_bodies.insert(*subject, digest);
            stats.window_bodies_sent += 1;
        }
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
    for region in regions.direct_children(config.realm) {
        // The marker radius the parent authors is the child's LOOK extent — the size the child
        // would draw at, never its authority bound (at true scale the two differ by orders).
        let Some(look) = region.look else { continue };
        bodies.push((
            region.realm,
            BodyStmt::Marker {
                luma: vd_core::look::marker_bag(
                    child_luma.get(&region.realm).copied(),
                    look.circumscribed_extent(),
                ),
            },
        ));
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
    let (mut windows, child_luma, mut relay_ship, mut relay_held, verdict) = window_lane;
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
    let realms = regions.authored_realm_snaps(config.realm, head);
    // ★ THE TWO LANES, SPLIT BY WHETHER THE CHILD MOVES (slice S10; owner ruling 2026-08-27).
    // A mover's row belongs on the per-tick frame, where repetition IS the loss story and a resend of
    // last tick's value would be worse than useless. A STATIC child's row belongs on the reliable lane
    // exactly once — MEASURED, repeating it costs 285 MB/s per subscriber at the target census for data
    // that never changes, and 14.2 MB does not fit a datagram at all.
    //
    // `realms` (the whole roster) is kept for the PARENT-ward relay below, which is a different lane to
    // a different consumer and is not split here.
    let (movers, statics): (Vec<RealmSnap>, Vec<RealmSnap>) = realms
        .iter()
        .cloned()
        .partition(|r| regions.child_moves(r.realm));

    // ===== THE WINDOW LANE (docs/design/window_lane.md §2.9): per tick, per open window, ONE
    // code path for every realm kind (HR3/HR4; ships excluded + counted per the D-SHIP-1
    // pattern). There is NO authored-empty return: a childless LEAF realm authors no rows, yet
    // its own-level window still owes its per-tick stamp (`at` is what the composer aligns chains
    // on) and its self-look. Since Slice C2 this is the WHOLE of the shard's picture egress —
    // the old lanes it used to run beside are deleted.
    prune_expired_windows(&mut windows, &config, clock.local_tick, &mut stats);
    stats.windows_open = windows.0.len() as u64;
    if !windows.0.is_empty() {
        emit_window_frames(
            &config,
            &clock,
            realm_fence,
            &regions,
            head,
            &movers,
            &windows,
            &mut stats,
            &mut outbox,
        );
        emit_window_bodies(
            &config,
            &clock,
            realm_fence,
            &regions,
            &child_luma.0,
            &mut windows,
            &mut stats,
            &mut outbox,
        );
        emit_window_static_rows(
            &clock,
            realm_fence,
            &statics,
            &mut windows,
            &mut stats,
            &mut outbox,
        );
    }
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
        let bodies = current_bodies(&config, &regions, &child_luma.0);
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
type WindowLaneStores<'w> = (
    ResMut<'w, OpenWindows>,
    Res<'w, ChildLuma>,
    ResMut<'w, RelayShip>,
    ResMut<'w, RelayHeld>,
    Res<'w, InBandVerdict>,
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
