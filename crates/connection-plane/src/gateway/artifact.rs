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

use vd_core::glam::{DQuat, DVec3};
use vd_core::pose::LatticePos;
use vd_core::pose::RealmId;
use vd_core::{EntityId, NodeId, TickId};
use vd_sim::io::MsgClass;
use vd_sim::runtime::{ClockSample, OutboundBox};
use vd_wire::channels::{BulkMsg, ServerControlMsg, SnapshotDatagram};
use vd_wire::session_flow::{ArtifactView, GatewayToShard};

use super::routing::{push_control, push_to_shard};
use super::{GatewayConfig, GatewaySessions, GatewayStats, Session, SessionPhase};

/// ★ The minor a session must have negotiated to read a fine record (the levels' record, minor 38; ruling B86 item 3).
pub(crate) const FINE_MINOR: u16 = 38;
/// The fine levels a finished tile carries (its square and five mean levels): a session that holds none of a tile
/// holds "from past the coarsest".
const FINE_LEVELS: u8 = vd_terrain::fine::read::MEAN_LEVELS as u8 + 1;

/// How many artifact bytes one beat may carry to one client, over every realm it draws: a
/// megabyte a beat (thirty-two parts of 32 KB), the sky's own pace — a pace, never a cap. A beat
/// sends while any of it is left, so one part larger than what is left still goes, and every beat
/// with something due sends at least one part.
///
/// ★ ONE BUDGET PER CLIENT (2026-09-27, the solved-world walk): the budget was thirty-two parts for
/// EACH REALM, so a pilot who logged in while the gateway held five planets' whole artifacts was
/// sent about five megabytes in one half-second beat against the transport's four-megabyte retry
/// buffer; 452 sends were shed, and three far planets' pyramids were never whole for that pilot.
pub(crate) const ARTIFACT_BYTES_PER_BEAT: usize = 32 * 32 * 1024;

/// ★ WHAT A SHED DOES TO A SESSION'S SERVE THIS BEAT ([`super::session::ArtifactRedrive`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Redrive {
    /// No shed is pending: the serve goes on.
    Idle,
    /// A shed came within the last beat: nothing goes to this client this beat, so its retry
    /// buffer drains.
    Hold,
    /// A shed came a beat or more ago: send again what the client may lack, then serve.
    Now,
}

/// The re-drive's state for a session whose newest shed toward its client came at `shed_at`.
#[must_use]
pub(crate) fn redrive_state(shed_at: Option<TickId>, now: TickId, cadence: u64) -> Redrive {
    match shed_at {
        None => Redrive::Idle,
        Some(at) if now.0.saturating_sub(at.0) < cadence => Redrive::Hold,
        Some(_) => Redrive::Now,
    }
}

/// ★ THE SESSION FORGETS A REALM'S ARTIFACT (2026-09-27): what its client stated it holds, how far
/// it was served, the tiles it was sent and every re-drive record of the realm — sent in the same
/// pass as the scene delta that tells the client the realm left, which makes the client forget
/// the same. Example: a far planet leaves a pilot's picture for one tick and comes back; the next
/// beat serves the pilot its head and pyramid again.
pub(crate) fn forget_for_session(session: &mut Session, realm: RealmId) {
    session.artifact_held.remove(&realm);
    session.artifact_parts_sent.remove(&realm);
    session.artifact_tiles_sent.remove(&realm);
    session.fine_sent.remove(&realm);
    session.fine_acked.remove(&realm);
    session.artifact_redrive.repass.remove(&realm);
}

/// ★ A SEND TOWARD A NODE WAS SHED (2026-09-27): when the node is a client that holds a session,
/// that session's re-drive is armed at `now`; a shed toward any other node (a shard, a client with
/// no session) changes nothing here. Example: the transport sheds a pyramid part toward a pilot's
/// client, and the next beats hold, then serve that pilot again.
pub(crate) fn note_send_shed(
    to: NodeId,
    now: TickId,
    sessions: &mut GatewaySessions,
    stats: &mut GatewayStats,
) {
    let named = sessions.by_client.get(&to).copied();
    let Some(session) = named.and_then(|sid| sessions.by_session.get_mut(&sid)) else {
        return;
    };
    session.artifact_redrive.shed_at = Some(now);
    stats.client_sends_shed += 1;
}

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
    /// ★ THE HEAD'S THREE NUMBERS THE FINE REACH READS (ruling B87): the drawn ground's lowest and highest metres and
    /// the sea — the shard names its answers by the same three.
    pub(crate) band: (i32, i32, i16),
    /// ★ THE FINE LEVELS HELD (ruling B87): each tile's levels the shard answered, one record a level, as the wire's
    /// own part messages in order, shared by every session that draws the realm; and the record assembling now, by tile
    /// (its level, read off its first part, and its parts so far).
    pub(crate) fine: BTreeMap<(FineTile, u8), Vec<Vec<u8>>>,
    fine_assembling: BTreeMap<(u8, u32, u32), FineAssembling>,
}

/// A fine tile's name: its face and its index there.
pub(crate) type FineTile = (u8, u32, u32);

/// One fine record assembling on the gateway: its level once its first part is here, and its parts so far.
#[derive(Debug, Default, Clone, PartialEq, Eq)]
struct FineAssembling {
    level: Option<u8>,
    parts: Vec<Option<Vec<u8>>>,
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
            sea_m,
            low_m,
            high_m,
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
                            band: (low_m, high_m, sea_m),
                            fine: BTreeMap::new(),
                            fine_assembling: BTreeMap::new(),
                        },
                    );
                    for session in sessions.by_session.values_mut() {
                        session.artifact_parts_sent.remove(&realm);
                        session.artifact_tiles_sent.remove(&realm);
                        session.fine_sent.remove(&realm);
                        session.fine_acked.remove(&realm);
                        session.artifact_redrive.repass.remove(&realm);
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
        // ★ A FINE RECORD'S PART (ruling B87): the shard's answer to the viewers' views, one level a record, held whole
        // in the realm's cache for every session whose distance reads it. The record's level is read off its first
        // part's own label (the levels' record's form and first level: never a height); a part out of order, past its
        // count, for a realm with no head, or from another node is a stray.
        Ok(BulkMsg::FineTilePart {
            realm: named,
            face,
            tx,
            ty,
            part,
            parts,
            bytes: piece,
            ..
        }) if named == realm => {
            let Some(cache) = sessions.artifacts.get_mut(&realm) else {
                stats.artifact_parts_stray += 1;
                return;
            };
            if cache.shard != from || parts == 0 || part >= parts {
                stats.artifact_parts_stray += 1;
                return;
            }
            let tile = (face, tx, ty);
            let assembling = cache.fine_assembling.entry(tile).or_default();
            if assembling.parts.len() != parts as usize || part == 0 {
                // A record's first part, or another cut: the record starts over.
                *assembling = FineAssembling {
                    level: None,
                    parts: vec![None; parts as usize],
                };
            }
            if part == 0 {
                assembling.level = vd_terrain::fine::tile::range_label(&piece).map(|(lo, _)| lo);
            }
            assembling.parts[part as usize] = Some(bytes);
            let whole = assembling.parts.iter().all(Option::is_some);
            if let (true, Some(level)) = (whole, assembling.level) {
                let done = std::mem::take(&mut assembling.parts);
                cache.fine_assembling.remove(&tile);
                cache
                    .fine
                    .insert((tile, level), done.into_iter().flatten().collect());
            }
            cache.last_part_at = now;
            stats.artifact_parts_cached += 1;
        }
        _ => stats.artifact_parts_stray += 1,
    }
}

/// ★ A SESSION'S VIEW OF A DRAWN REALM (the one tile path; ruling B88: ONE MACHINERY FOR EVERY REALM): the eye's
/// direction and distance from the realm's centre, in the realm's own frame, by one formula for every realm the session
/// draws — the realm it stands in among them. The eye is first placed in the frame of the realm the session stands in
/// (the picture's origin), then carried into the drawn realm by the placement the picture already holds for it: its
/// centre and facing in the origin's frame (for the origin itself, its own centre and no turn).
///
/// The eye in the origin's frame is the avatar's own position for the origin itself — the realm's own snapshot states
/// it ([`SessionEye`](super::session::SessionEye)), and the view goes back to that realm alone — and the origin's centre
/// for every other realm: a session inside a hull is placed by the hull (SL7: an occupied realm stands in for whoever
/// is inside it; SL2: no occupant pose enters another realm). The proxy's error is the origin's own size, and the
/// fine reach carries one fine tile (eight 8 km nodes) past its edge. `None` until the avatar is read, for an eye at a
/// centre, and for a length that is not a number — a diverged shard may state a non-finite pose, and a NaN reach names
/// every tile.
///
/// **Example.** A pilot sits in a hull berthed on the home belt; her eye is the hull's centre carried into the planet's
/// frame. She walks out onto the belt: her eye is now her avatar's position, which the planet's snapshot states. Both
/// views go to the planet's shard the same way, and the same call names the same pieces from the same spot.
pub(crate) fn session_view(session: &Session, realm: RealmId) -> Option<ArtifactView> {
    let (eye_at_origin, centre, orient) = if session.shadow.origin == Some(realm) {
        let (stated_in, at) = session.eye.at?;
        if stated_in != realm {
            return None; // the avatar was last read in another realm: a crossing's first beat
        }
        (DVec3::from_array(at), DVec3::ZERO, DQuat::IDENTITY)
    } else {
        let row = session.shadow.drawn_rows().find(|r| r.realm == realm)?;
        let centre = row
            .pose
            .pos
            .delta_m(LatticePos::ORIGIN, row.pose.frame.tier());
        (DVec3::ZERO, centre, row.pose.orient)
    };
    let eye = orient.inverse() * (eye_at_origin - centre);
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

/// ★ THE AVATAR IN A FRAME'S BODY (ruling B88): the realm whose frame it stands in and its metres from that realm's
/// centre, read off the snapshot body the realm's shard sent this gateway for the session's client. The body is
/// decoded at most once per frame (`decoded`), and only when a session behind it wants its eye; a body that does not
/// decode, does not hold the avatar or states a position that is not a number gives nothing.
pub(crate) fn avatar_at(
    decoded: &mut Option<Option<SnapshotDatagram>>,
    body: &[u8],
    entity: EntityId,
) -> Option<(RealmId, [f64; 3])> {
    let snapshot = decoded
        .get_or_insert_with(|| postcard::from_bytes::<SnapshotDatagram>(body).ok())
        .as_ref()?;
    let pose = &snapshot.entities.iter().find(|e| e.entity == entity)?.pose;
    let at = pose.pos.delta_m(LatticePos::ORIGIN, pose.frame.tier());
    at.is_finite().then(|| (pose.frame.realm(), at.to_array()))
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

/// The tiles a view reaches on a body, by the terrain crate's own rule: THE WHOLE TILE RING
/// (`vd_terrain::artifact::tile_ring_m`), the finest tile-reading rung's ask ring with no horizon.
///
/// ★ THE WHOLE RING, NEVER THE HORIZON (2026-10-01): the client's ladder reads its horizon over the
/// SOLVED ground under the eye, and the gateway relays the tiles undecoded, so it cannot read that
/// ground. A horizon read over the radius stood short of the client's wherever the solved ground lay
/// far under it — MEASURED on the shard's own copy of the rule at the picture gate's seam stand: 6
/// tiles named, 8 more waited on. The whole ring holds every reach over any ground, so a session is
/// served every cached tile it can ask; the shard, which holds the ground, ships the exact set.
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
    // ★ THE SAME FLOOR THE CLIENT ASKS AT (ruling W17): the finest tile-reading rung's handover
    // step, octaves and field together.
    let rung = vd_terrain::artifact::tile_rung(&lattice, levels, body.ladder().rungs - 1);
    let step_m =
        body.step_bound_m(rung) + vd_terrain::artifact::field_step_m(body, &lattice, levels, rung);
    let ring_m = vd_terrain::artifact::tile_ring_m(
        &lattice,
        levels,
        body.ladder().rungs - 1,
        vd_core::geometry::drawable_theta_min_rad(),
        step_m,
    );
    vd_terrain::artifact::tiles_within(&lattice, view.dir, ring_m)
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
        session
            .artifact_redrive
            .repass
            .retain(|realm| drawn.contains_key(realm));
        // ★ THE EYE IS READ AGAIN for the next beat (ruling B88), off the next frame that holds the avatar.
        if matches!(session.phase, SessionPhase::Active { .. }) {
            session.eye.wanted = true;
        }
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
    // ★ THE FINE LEVELS EACH VIEW READS (ruling B87), by the one call the realm's shard names its answers by — for every
    // session and drawn realm whose view went to the shard this beat; a tile no view reads any more leaves the realm's
    // cache (the shard lets it go from its record by the same views).
    let pixel_rad = vd_core::geometry::drawable_theta_min_rad();
    let mut fine_needs: BTreeMap<(vd_core::SessionId, RealmId), Vec<(FineTile, u8)>> =
        BTreeMap::new();
    for (sid, session) in &sessions.by_session {
        if !matches!(session.phase, SessionPhase::Active { .. }) {
            continue;
        }
        for (realm, digest) in drawn {
            let Some(cache) = sessions
                .artifacts
                .get(realm)
                .filter(|c| c.digest == *digest)
            else {
                continue;
            };
            if let Some(view) = session_view(session, *realm)
                && let Some(bag) = sessions
                    .windows
                    .values()
                    .find_map(|w| w.ingest.look_of(*realm))
                && let Some(body) = body_of_bag(bag)
            {
                let needs = vd_terrain::artifact::fine_needs_of_head(
                    &body,
                    cache.band,
                    view.dir,
                    view.radial_m,
                    pixel_rad,
                );
                fine_needs.insert((*sid, *realm), needs);
            }
        }
    }
    let read: BTreeSet<(RealmId, (u8, u32, u32))> = fine_needs
        .iter()
        .flat_map(|((_, realm), needs)| needs.iter().map(move |(t, _)| (*realm, *t)))
        .collect();
    for (realm, cache) in sessions.artifacts.iter_mut() {
        cache
            .fine
            .retain(|(tile, _), _| read.contains(&(*realm, *tile)));
    }
    // The serve: each session that draws a realm whose artifact is whole here, and has not stated
    // it holds it, gets the head then the parts, paced, each once — within ONE budget a beat for the
    // whole session ([`ARTIFACT_BYTES_PER_BEAT`]), each drawn realm's tiles and parts in the picture's
    // order.
    let GatewaySessions {
        by_session,
        artifacts,
        windows,
        ..
    } = sessions;
    for (sid, session) in by_session.iter_mut() {
        if !matches!(session.phase, SessionPhase::Active { .. }) {
            continue;
        }
        match redrive_state(session.artifact_redrive.shed_at, clock.local_tick, cadence) {
            Redrive::Idle => {}
            Redrive::Hold => {
                stats.artifact_serves_held += 1;
                continue;
            }
            Redrive::Now => redrive(session, artifacts, stats),
        }
        let mut budget = ARTIFACT_BYTES_PER_BEAT;
        // ★ COARSE FIRST, EVERY REALM, THEN THE FINE LEVELS (rulings B86 item 3 and B88): every drawn realm's tiles and its
        // head and pyramid — the ground every rung stands on, and a far planet's whole look — go before any realm's fine
        // levels, which only refine that ground. MEASURED before (the walker on the home belt, the first run on one path):
        // the fine levels went first inside the home planet's turn and took the beat's megabyte for about 30 s, the
        // pyramid waited, the client built 16 pieces of ground in those 30 s, and the far planets' heads waited 20 s.
        let drawn_here: Vec<RealmId> = drawn_realms(session).collect();
        for &realm in &drawn_here {
            let Some(cache) = artifacts.get(&realm).filter(|a| a.whole()) else {
                continue;
            };
            // ★ THE TILES WITHIN THIS SESSION'S REACH (the one tile path), each once, in the
            // beat's budget beside the parts: the view the gateway composed for the session and
            // the terrain crate's own reach rule pick them out of the realm's cache. Example: a
            // pilot in orbit over the home planet reaches nine tiles; the gateway hands her as
            // many as the beat's megabyte holds, and the rest the next beat.
            if let Some(view) = session_view(session, realm)
                && let Some(bag) = windows.values().find_map(|w| w.ingest.look_of(realm))
                && let Some(body) = body_of_bag(bag)
            {
                let sent = session.artifact_tiles_sent.entry(realm).or_default();
                let due: Vec<_> = tiles_in_reach(&body, cache.levels, view)
                    .into_iter()
                    .filter(|tile| !sent.contains(tile))
                    .filter_map(|tile| cache.tiles.get(&tile).map(|bytes| (tile, bytes)))
                    .collect();
                for (tile, bytes) in due {
                    if budget == 0 {
                        break;
                    }
                    push_control(
                        outbox,
                        session.client,
                        &ServerControlMsg::ArtifactPart {
                            bytes: bytes.clone(),
                        },
                    );
                    sent.insert(tile);
                    stats.artifact_parts_served += 1;
                    budget = budget.saturating_sub(bytes.len());
                }
            }
            if session.artifact_held.get(&realm) == Some(&cache.digest) {
                session.artifact_redrive.repass.remove(&realm);
                stats.artifact_parts_skipped += 1;
                continue;
            }
            let total = 1 + cache.parts.len() as u32;
            let mut next = session
                .artifact_parts_sent
                .get(&realm)
                .copied()
                .unwrap_or(0)
                .min(total);
            // ★ A PASS OWED BY A RE-DRIVE starts from the head once the pass in flight has ended,
            // so a pass always reaches its tail.
            if next == total && session.artifact_redrive.repass.remove(&realm) {
                next = 0;
            }
            if next == 0 && budget > 0 {
                tracing::info!(
                    realm = %realm,
                    client = ?session.client,
                    total,
                    "the artifact is served to a session that draws the realm"
                );
            }
            while next < total && budget > 0 {
                let bytes = if next == 0 {
                    &cache.head
                } else {
                    &cache.parts[(next - 1) as usize]
                };
                push_control(
                    outbox,
                    session.client,
                    &ServerControlMsg::ArtifactPart {
                        bytes: bytes.clone(),
                    },
                );
                stats.artifact_parts_served += 1;
                budget = budget.saturating_sub(bytes.len());
                next += 1;
            }
            session.artifact_parts_sent.insert(realm, next);
        }
        for &realm in &drawn_here {
            let Some(cache) = artifacts.get(&realm).filter(|a| a.whole()) else {
                continue;
            };
            // ★ THE FINE LEVELS BY THIS SESSION'S OWN DISTANCE (ruling B87), whatever realm it stands in: the nearest tile
            // first, each tile's levels the session lacks the coarsest first, from the realm's cache, in the beat's
            // budget; a level not cached yet waits with every finer level of its tile, and the session draws the coarser
            // ones meanwhile. Only to a session that reads them. The nearest tile whole first, as the shard answers them
            // (`vd_sim::stub::fine_ship`): the near pieces the client builds first need its finest levels first.
            if session.negotiated_minor >= FINE_MINOR
                && let Some(needs) = fine_needs.get(&(*sid, realm))
            {
                let named: BTreeSet<FineTile> = needs.iter().map(|(t, _)| *t).collect();
                let sent = session.fine_sent.entry(realm).or_default();
                sent.retain(|tile, _| named.contains(tile));
                for &(tile, need) in needs {
                    let mut held = sent.get(&tile).copied().unwrap_or(FINE_LEVELS);
                    while (budget > 0) & (held > need) {
                        let Some(record) = cache.fine.get(&(tile, held - 1)) else {
                            break;
                        };
                        for bytes in record {
                            push_control(
                                outbox,
                                session.client,
                                &ServerControlMsg::ArtifactPart {
                                    bytes: bytes.clone(),
                                },
                            );
                            stats.artifact_parts_served += 1;
                            budget = budget.saturating_sub(bytes.len());
                        }
                        held -= 1;
                        sent.insert(tile, held);
                    }
                    if budget == 0 {
                        break;
                    }
                }
            }
        }
    }
}

/// ★ THE RE-DRIVE, ONCE (2026-09-27): a shed toward the session's client came a beat or more ago.
/// Every realm it was served and has not stated whole owes one more pass from the head, and every
/// tile served from a view goes out again (ruling B88: every tile comes from a realm's cache); the
/// client drops what it already holds.
fn redrive(
    session: &mut Session,
    artifacts: &BTreeMap<RealmId, GatewayArtifact>,
    stats: &mut GatewayStats,
) {
    // ★ The fine levels served go out again from the realms' caches (ruling B87) — only those the client has not
    // stated it holds (ruling B81 Q3): the levels it acknowledged stand.
    session.fine_sent = session.fine_acked.clone();
    let held = &session.artifact_held;
    let owed: Vec<RealmId> = session
        .artifact_parts_sent
        .keys()
        .filter(|realm| {
            artifacts
                .get(realm)
                .is_none_or(|cache| held.get(realm) != Some(&cache.digest))
        })
        .copied()
        .collect();
    let redrive = &mut session.artifact_redrive;
    redrive.shed_at = None;
    redrive.repass.extend(owed);
    session.artifact_tiles_sent.clear();
    stats.artifact_redrives += 1;
    tracing::info!(
        client = ?session.client,
        repass = redrive.repass.len(),
        "a send toward the client was shed: its artifacts are served again"
    );
}
