//! ★ THE FAR-VIEW SHIP — the gateway's side (the landform arc, slice 8c; SL3: a realm draws itself,
//! whoever looks; SL2's clarification: the gateway composes every observer's picture and is not a
//! realm).
//!
//! A realm states the digest of the artifact its shard serves in its own look bag (`TAG_ARTIFACT`),
//! which reaches this gateway the way its surface does — through the realm's own window when a
//! session stands inside it, through its parent's window by relay when a session looks at it from
//! outside. Once a beat, for every session's composed picture, the gateway reads which drawn realms
//! state an artifact and:
//!
//! 1. ASKS the realm's shard once (`GatewayToShard::ArtifactWant`) for an artifact it does not hold
//!    at that digest, resolving the shard the way the window lane resolves a lineage ancestor (the
//!    directory's `Realm` head, polled on the same beat);
//! 2. CACHES the head and the pyramid parts the shard answers with (`BulkFor { audience: Realm }`),
//!    opaque, ONE copy per realm however many sessions draw it — the way it holds the sky;
//! 3. SERVES them, paced, to each session that draws the realm and has not stated it holds them
//!    (`ClientControlMsg::ArtifactHeld`), the head first, each part once — the sky's own pacing;
//! 4. FORGETS a realm's artifact when no session draws the realm any more.
//!
//! The gateway never decodes a tile; the head and the pyramid parts are read only for their realm,
//! digest and part indices, so the cache knows when it is whole.
//!
//! **Example.** A pilot in a hull in orbit sees the home planet. The planet's look bag, relayed by
//! the star system's window, names an artifact. Her gateway asks the planet's shard, holds the six
//! levels it answers with, and hands them to her a megabyte a beat; a second pilot logging in behind
//! the same gateway gets them from the cache and the shard is not asked again.

use std::collections::{BTreeMap, BTreeSet};

use vd_core::pose::LatticePos;
use vd_core::pose::RealmId;
use vd_core::{NodeId, TickId};
use vd_sim::io::MsgClass;
use vd_sim::runtime::{ClockSample, OutboundBox};
use vd_wire::channels::{BulkMsg, ServerControlMsg};
use vd_wire::session_flow::{ArtifactView, GatewayToShard};

use super::routing::{push_control, push_to_shard};
use super::{GatewayConfig, GatewaySessions, GatewayStats, Session, SessionPhase};

/// How many artifact parts one beat may carry to one client: 32 of 32 KB is a megabyte a beat,
/// the sky's own pace — a pace, never a cap.
pub(crate) const ARTIFACT_PARTS_PER_BEAT: u32 = 32;

/// ★ AN ARTIFACT THE GATEWAY HOLDS for the realms its sessions draw: the shard it came from, the
/// digest, the head's bytes and the pyramid parts' bytes in arrival order, and what makes it whole.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct GatewayArtifact {
    pub(crate) shard: NodeId,
    pub(crate) digest: [u64; 2],
    pub(crate) levels: u32,
    /// ★ HOW MANY COAST PARTS THE HEAD ANNOUNCED (2026-09-22, ruling W10): the cache is whole only
    /// when every one of them is here, so a session is never served a shoreline with a hole in it.
    pub(crate) coast_parts: u32,
    /// The head's bytes: an entry exists only once a head arrived.
    pub(crate) head: Vec<u8>,
    pub(crate) parts: Vec<Vec<u8>>,
    /// When the last head or part landed: a transfer that stalls a beat is asked for again.
    pub(crate) last_part_at: TickId,
    /// Each level's part count, as its parts state it.
    level_parts: BTreeMap<u32, u32>,
    /// The `(level, part)` pairs held.
    seen: BTreeSet<(u32, u32)>,
    /// The coast parts held, by index.
    coast_seen: BTreeSet<u32>,
    /// ★ THE TILES HELD (the one tile path): the shard's answers to the viewers' wants, by tile,
    /// shared by every session that draws the realm.
    pub(crate) tiles: BTreeMap<(u8, u32, u32), Vec<u8>>,
}

impl GatewayArtifact {
    /// Whether the head, every part of every level AND every coast part are here.
    #[must_use]
    pub(crate) fn whole(&self) -> bool {
        let expected: u32 = (1..=self.levels)
            .map(|l| self.level_parts.get(&l).copied().unwrap_or(u32::MAX))
            .fold(0u32, u32::saturating_add);
        // A head of no levels announces nothing and is never whole.
        self.levels > 0
            && expected != u32::MAX
            && self.seen.len() as u32 == expected
            // ★ THE COAST MASK too (2026-09-22, ruling W10): a client that draws a far rung with
            // half a mask draws half a shoreline, so the cache is not whole until every part is in.
            && self.coast_seen.len() as u32 == self.coast_parts
    }
}

/// ★ ONE PART ARRIVES from a shard on the realm audience: a head opens (or replaces, at a new
/// digest) the realm's cache; a pyramid part lands in it. A part for a realm with no head, from a
/// node that is not the head's sender, or of a shape the head did not announce is counted and
/// dropped. Nothing else in the bulk shapes is a realm's (a tile is an occupant's).
pub(crate) fn cache_artifact_part(
    from: NodeId,
    realm: RealmId,
    bytes: Vec<u8>,
    now: TickId,
    sessions: &mut GatewaySessions,
    stats: &mut GatewayStats,
) {
    match postcard::from_bytes::<BulkMsg>(&bytes) {
        Ok(BulkMsg::ArtifactHead {
            realm: named,
            digest,
            levels,
            coast_parts,
            ..
        }) if named == realm => {
            match sessions.artifacts.get_mut(&realm) {
                // The same head again (a re-answer): the transfer is alive, nothing restarts.
                Some(cache) if cache.digest == digest && cache.shard == from => {
                    cache.last_part_at = now;
                }
                // A new artifact, or the same one from another node: a new transfer for every
                // session that draws the realm.
                _ => {
                    tracing::info!(
                        realm = %realm,
                        from = from.0,
                        levels,
                        "the artifact's head is cached; the pyramid's parts follow"
                    );
                    sessions.artifacts.insert(
                        realm,
                        GatewayArtifact {
                            shard: from,
                            digest,
                            levels,
                            coast_parts,
                            head: bytes,
                            parts: Vec::new(),
                            last_part_at: now,
                            level_parts: BTreeMap::new(),
                            seen: BTreeSet::new(),
                            coast_seen: BTreeSet::new(),
                            tiles: BTreeMap::new(),
                        },
                    );
                    for session in sessions.by_session.values_mut() {
                        session.artifact_parts_sent.remove(&realm);
                        session.artifact_tiles_sent.remove(&realm);
                    }
                }
            }
            stats.artifact_parts_cached += 1;
        }
        Ok(BulkMsg::ArtifactPyramid {
            realm: named,
            level,
            part,
            parts,
            ..
        }) if named == realm => {
            let Some(cache) = sessions.artifacts.get_mut(&realm) else {
                stats.artifact_parts_stray += 1;
                return;
            };
            if cache.shard != from
                || level == 0
                || level > cache.levels
                || parts == 0
                || part >= parts
                || cache.level_parts.get(&level).is_some_and(|&p| p != parts)
                || !cache.seen.insert((level, part))
            {
                stats.artifact_parts_stray += 1;
                return;
            }
            cache.level_parts.insert(level, parts);
            cache.parts.push(bytes);
            cache.last_part_at = now;
            stats.artifact_parts_cached += 1;
        }
        // ★ A COAST PART (2026-09-22, ruling W10): the realm's sea bits, held with the head and
        // served to every session that draws the realm, like a pyramid part. A part for a realm
        // with no head, from another node, or of a shape the head did not announce is a stray.
        Ok(BulkMsg::ArtifactCoast {
            realm: named,
            part,
            parts,
            ..
        }) if named == realm => {
            let Some(cache) = sessions.artifacts.get_mut(&realm) else {
                stats.artifact_parts_stray += 1;
                return;
            };
            if cache.shard != from
                || parts != cache.coast_parts
                || part >= parts
                || !cache.coast_seen.insert(part)
            {
                stats.artifact_parts_stray += 1;
                return;
            }
            cache.parts.push(bytes);
            cache.last_part_at = now;
            stats.artifact_parts_cached += 1;
        }
        // ★ A TILE (the one tile path): the shard's answer to a viewer's want, held with the
        // realm's cache for every session within reach of it; a tile for a realm with no head,
        // or from another node, is a stray.
        Ok(BulkMsg::ArtifactTile {
            realm: named,
            face,
            tx,
            ty,
            ..
        }) if named == realm => {
            let Some(cache) = sessions.artifacts.get_mut(&realm) else {
                stats.artifact_parts_stray += 1;
                return;
            };
            if cache.shard != from {
                stats.artifact_parts_stray += 1;
                return;
            }
            cache.tiles.insert((face, tx, ty), bytes);
            cache.last_part_at = now;
            stats.artifact_parts_cached += 1;
        }
        _ => stats.artifact_parts_stray += 1,
    }
}

/// ★ A SESSION'S VIEW OF A DRAWN REALM (the one tile path): the eye's direction and distance from
/// the realm's centre, in the realm's own frame, read off the row the gateway composed for the
/// session — the realm's centre and facing in the picture's frame, whose origin is the session's
/// own realm. The eye stands at the picture's origin within its own realm's size: a pilot in a
/// hull is placed by the hull, the SL7 proxy. `None` for the session's own realm (its shard ships
/// its occupants' tiles itself), for a row at the centre, and for a row whose length is not a
/// number — a diverged shard may state a non-finite pose, and a NaN reach names every tile.
fn session_view(session: &Session, realm: RealmId) -> Option<ArtifactView> {
    if session.shadow.origin == Some(realm) {
        return None;
    }
    let row = session.shadow.drawn_rows().find(|r| r.realm == realm)?;
    let centre = row
        .pose
        .pos
        .delta_m(LatticePos::ORIGIN, row.pose.frame.tier());
    let eye = row.pose.orient.inverse() * (-centre);
    let radial_m = eye.length();
    if !radial_m.is_finite() || radial_m <= 0.0 {
        return None;
    }
    let dir = eye / radial_m;
    Some(ArtifactView {
        dir: [dir.x, dir.y, dir.z],
        radial_m,
    })
}

/// The body a realm's look bag states, for the tile reach: its seed off the surface's frame, its
/// radius off the look, its two charter words. `None` for a realm that states no surface, no
/// charter, or a surface that is not a PLANET's — a star states its surface in its own
/// star-centred frame — and such a realm draws no tiles.
fn body_of_bag(bag: &[u8]) -> Option<vd_terrain::BodyDefinition> {
    let surface = vd_core::look::surface_of(bag).ok().flatten()?;
    let charter = vd_core::look::charter_of_bag(bag).ok().flatten()?;
    let RealmId::Planet(seed) = surface.frame.realm() else {
        return None;
    };
    let radius_m = vd_core::look::look_of(bag).ok()?.finite_extent();
    vd_terrain::BodyDefinition::from_seed(
        seed,
        radius_m,
        vd_terrain::BodyFacts::new(charter.gravity_mm_s2, charter.bulk_density_kgm3),
    )
}

/// The tiles a view reaches on a body, by the terrain crate's own rule (the shard's rule too).
///
/// ★ `pub(crate)` FOR THE PEBBLE (2026-09-21). A body the bag's charter builds always HAS a macro
/// lattice, because the divisor rule and `BodyDefinition::from_seed` read the same cell count: the
/// only body without one is the pebble, which `from_seed` refuses. So the no-lattice arm cannot be
/// driven through a look bag, and the gateway's unit test calls this with a REAL body the terrain
/// crate strips the lattice off (`BodyDefinition::without_macro_lattice`, which that crate states
/// for exactly this). Example: a 5 km rock a pilot flies past states a surface and a charter, and
/// the gateway hands her no tiles for it — it draws from its own recipe.
pub(crate) fn tiles_in_reach(
    body: &vd_terrain::BodyDefinition,
    levels: u32,
    view: ArtifactView,
) -> Vec<(u8, u32, u32)> {
    let Some(lattice) = body.macro_lattice() else {
        return Vec::new();
    };
    let reach_m = vd_terrain::artifact::tile_reach_m(
        &lattice,
        levels,
        body.ladder().rungs - 1,
        body.radius_m(),
        body.relief_bound_m(0),
        view.radial_m,
        vd_core::geometry::drawable_theta_min_rad(),
    );
    vd_terrain::artifact::tiles_within(&lattice, view.dir, reach_m)
}

/// ★ THE ARTIFACTS THE PICTURES NAME: for every Active session's drawn realms, the digest the
/// realm's own look bag states (through any window this gateway holds). A realm that states none
/// is not in the map: it is drawn from its look alone.
pub(crate) fn drawn_artifacts(sessions: &GatewaySessions) -> BTreeMap<RealmId, [u64; 2]> {
    let mut out = BTreeMap::new();
    for session in sessions.by_session.values() {
        if !matches!(session.phase, SessionPhase::Active { .. }) {
            continue;
        }
        for realm in drawn_realms(session) {
            // A realm two sessions draw is read twice and inserted once: the bag is the same.
            let stated = sessions
                .windows
                .values()
                .find_map(|w| w.ingest.look_of(realm))
                .and_then(|bag| vd_core::look::artifact_of_bag(bag).ok().flatten());
            if let Some(digest) = stated {
                out.insert(realm, digest);
            }
        }
    }
    out
}

/// ★ THE REALMS A SESSION DRAWS, for the artifact ship: the realm the occupant STANDS IN first,
/// then every composed row. The composed rows hold a body row for every ancestor and a row for
/// every child, and none for the origin — its row is synthesised for the client alone
/// (`scene_level_rows`). MEASURED on the far-view ship's first flights (2026-09-20): the gateway
/// asked for every planet in the sky except the one the pilot stood over, and the globe under
/// the pilot switched to its solved shape only when a boarded hull made the planet an ancestor.
/// A realm draws itself whoever looks (SL3), the occupant inside it included.
fn drawn_realms(session: &Session) -> impl Iterator<Item = RealmId> + '_ {
    let mut seen: BTreeSet<RealmId> = BTreeSet::new();
    session
        .shadow
        .origin
        .into_iter()
        .chain(session.shadow.drawn_rows().map(|r| r.realm))
        .filter(move |r| seen.insert(*r))
}

/// ★ THE BEAT: ask for what is not held, forget what nobody draws, serve what is whole.
pub(crate) fn serve_artifacts(
    config: &GatewayConfig,
    clock: &ClockSample,
    drawn: &BTreeMap<RealmId, [u64; 2]>,
    sessions: &mut GatewaySessions,
    stats: &mut GatewayStats,
    outbox: &mut OutboundBox,
) {
    // Forget the artifacts and the wants of realms no picture names any more.
    let before = sessions.artifacts.len();
    sessions
        .artifacts
        .retain(|realm, _| drawn.contains_key(realm));
    stats.artifact_caches_evicted += (before - sessions.artifacts.len()) as u64;
    sessions
        .artifact_wants
        .retain(|realm, _| drawn.contains_key(realm));
    for session in sessions.by_session.values_mut() {
        session
            .artifact_parts_sent
            .retain(|realm, _| drawn.contains_key(realm));
        session
            .artifact_tiles_sent
            .retain(|realm, _| drawn.contains_key(realm));
    }
    // The wants: a realm whose artifact is not held at the stated digest — or held in part with
    // nothing landing for a beat (a stalled answer) — is asked for, once a beat at most, from the
    // shard the directory names for it; a realm without a resolved head waits for the head poll
    // on this same beat.
    let cadence = super::config::window_keepalive_cadence(config);
    for (realm, digest) in drawn {
        let held = sessions.artifacts.get(realm).is_some_and(|a| {
            a.digest == *digest
                && (a.whole() || clock.local_tick.0.saturating_sub(a.last_part_at.0) < cadence)
        });
        if held {
            continue;
        }
        let asked_recently = sessions
            .artifact_wants
            .get(realm)
            .is_some_and(|at| clock.local_tick.0.saturating_sub(at.0) < cadence);
        if asked_recently {
            continue;
        }
        let Some(shard) = sessions.realm_heads.get(realm).copied() else {
            stats.artifact_wants_unresolved += 1;
            continue;
        };
        tracing::info!(
            realm = %realm,
            shard = shard.0,
            digest = ?digest,
            "the artifact of a drawn realm is wanted from its shard"
        );
        push_to_shard(
            outbox,
            shard,
            MsgClass::Control,
            &GatewayToShard::ArtifactWant {
                realm: *realm,
                digest: *digest,
                view: None,
            },
        );
        sessions
            .artifact_wants
            .insert(*realm, TickId(clock.local_tick.0));
        stats.artifact_wants_sent += 1;
    }
    // ★ THE VIEWERS' TILE WANTS (the one tile path): for every session and every drawn realm
    // whose head is cached, the session's view of the realm goes to the realm's shard once a
    // beat; the shard answers the tiles the view reaches, each once per gateway.
    let mut views: Vec<(NodeId, RealmId, [u64; 2], ArtifactView)> = Vec::new();
    for session in sessions.by_session.values() {
        if !matches!(session.phase, SessionPhase::Active { .. }) {
            continue;
        }
        for (realm, digest) in drawn {
            let Some(cache) = sessions.artifacts.get(realm) else {
                continue;
            };
            if cache.digest != *digest {
                continue;
            }
            if let Some(view) = session_view(session, *realm) {
                views.push((cache.shard, *realm, *digest, view));
            }
        }
    }
    for (shard, realm, digest, view) in views {
        push_to_shard(
            outbox,
            shard,
            MsgClass::Control,
            &GatewayToShard::ArtifactWant {
                realm,
                digest,
                view: Some(view),
            },
        );
        stats.artifact_wants_sent += 1;
    }
    // The serve: each session that draws a realm whose artifact is whole here, and has not stated
    // it holds it, gets the head then the parts, paced, each once.
    let GatewaySessions {
        by_session,
        artifacts,
        windows,
        ..
    } = sessions;
    for session in by_session.values_mut() {
        if !matches!(session.phase, SessionPhase::Active { .. }) {
            continue;
        }
        let drawn_here: Vec<RealmId> = drawn_realms(session).collect();
        for realm in drawn_here {
            let Some(cache) = artifacts.get(&realm).filter(|a| a.whole()) else {
                continue;
            };
            // ★ THE TILES WITHIN THIS SESSION'S REACH (the one tile path), each once, in the
            // beat's budget beside the parts: the view the gateway composed for the session and
            // the terrain crate's own reach rule pick them out of the realm's cache.
            let mut budget = ARTIFACT_PARTS_PER_BEAT;
            if let Some(view) = session_view(session, realm)
                && let Some(bag) = windows.values().find_map(|w| w.ingest.look_of(realm))
                && let Some(body) = body_of_bag(bag)
            {
                let sent = session.artifact_tiles_sent.entry(realm).or_default();
                // ★ THE BEAT'S BUDGET IS A TAKE, NOT A TEST (2026-09-21). The pace is the
                // head's own: at most `ARTIFACT_PARTS_PER_BEAT` tiles a beat. A `budget == 0`
                // break inside this walk is a branch NO VIEW CAN DRIVE, so it states the pace
                // as a `take` instead. MEASURED: over 1 158 body radii from 10 km to
                // 1 000 000 km, four altitudes each and four directions each, the most tiles
                // one view reaches is NINE, and the budget is thirty-two — the reach is the
                // rung's switch distance (about 445 km) and a lattice node is about 8 192 m on
                // every body of THE world, so a view spans at most three tiles of sixty-four
                // nodes each way. Example: a pilot in orbit over the home planet reaches nine
                // tiles; the gateway hands her all nine in one beat and still has room for the
                // pyramid's parts. A finer lattice would make the break live again: restore it.
                let paced: Vec<_> = tiles_in_reach(&body, cache.levels, view)
                    .into_iter()
                    .filter(|tile| !sent.contains(tile))
                    .filter_map(|tile| cache.tiles.get(&tile).map(|bytes| (tile, bytes)))
                    .take(budget as usize)
                    .collect();
                for (tile, bytes) in paced {
                    push_control(
                        outbox,
                        session.client,
                        &ServerControlMsg::ArtifactPart {
                            bytes: bytes.clone(),
                        },
                    );
                    sent.insert(tile);
                    stats.artifact_parts_served += 1;
                    budget -= 1;
                }
            }
            if session.artifact_held.get(&realm) == Some(&cache.digest) {
                stats.artifact_parts_skipped += 1;
                continue;
            }
            let total = 1 + cache.parts.len() as u32;
            let first = session
                .artifact_parts_sent
                .get(&realm)
                .copied()
                .unwrap_or(0)
                .min(total);
            let last = (first + budget).min(total);
            if first == 0 {
                tracing::info!(
                    realm = %realm,
                    client = ?session.client,
                    total,
                    "the artifact is served to a session that draws the realm"
                );
            }
            for i in first..last {
                let bytes = if i == 0 {
                    cache.head.clone()
                } else {
                    cache.parts[(i - 1) as usize].clone()
                };
                push_control(
                    outbox,
                    session.client,
                    &ServerControlMsg::ArtifactPart { bytes },
                );
                stats.artifact_parts_served += 1;
            }
            session.artifact_parts_sent.insert(realm, last);
        }
    }
}
