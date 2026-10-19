//! ★ THE FINE SHIP — the shard's side (the fine ground's way in, B2; rulings B81 Q2–Q3, B85, B86 item 3 and ★ B87: the
//! fine ground reaches EVERY viewer by its distance, whatever realm the viewer stands in; SL3: a realm draws itself,
//! whoever looks).
//!
//! The sim never names the generator: the composition root injects a [`FineTiles`] provider (the shape of
//! [`super::artifact_ship::TileSource`]) once the artifact is here and the fine level applies to the body. Every tick:
//!
//! 1. **THE WORK** — the provider lands its workers' answers ([`FineTiles::tick`]) and writes them to this realm's own
//!    store, the one this shard already holds (`RealmStore`).
//! 2. **THE VIEWS** — ONE PATH for every viewer (ruling B88, the owner: *"one way only — through gateway"*): each
//!    gateway states once a beat the VIEW every session behind it has of this realm (`GatewayToShard::ArtifactWant
//!    { view }`: a direction and a distance from the realm's centre, never a pose entering its simulation, SL2) — the
//!    realm's own occupants (their avatars' positions, which the gateway reads off the snapshot this realm sends their
//!    clients), a pilot aboard a hull over it, a station in its orbit, anyone who draws it. A gateway's views of one
//!    beat replace the last's and lapse with the window keep-alive's own lifetime (two beats and a tick). For each new
//!    view set the provider names the fine tiles the views read, each with its finest level
//!    (`vd_terrain::artifact::fine_needs`: by distance alone) — the realm's REASON to solve them; on the ticks between,
//!    the named tiles are held as reasons ([`FineTiles::hold`]).
//! 3. **THE ANSWER** — each named tile goes to the gateway's cache for this realm (`BulkAudience::Realm`) ONE LEVEL A
//!    RECORD, the nearest tile first, each tile's levels the coarsest first and down to the finest any of its sessions
//!    reads, each level once per gateway, in its parts, [`FINE_BYTES_PER_TICK`] a tick a gateway; the gateway serves
//!    every session the levels its own distance reads, in the same order. A tile not finished yet is passed over and
//!    answered when it is. No record goes to a session directly.
//!
//! Nothing crosses UPWARD and no pose leaves this realm (SL2). A tile no view of a gateway reads any more is let go
//! from that gateway's record (the gateway drops it from its cache by the same views); a gateway that states no view for
//! a keep-alive's lifetime is forgotten; a gateway's head want (a new cache) starts its record over.
//!
//! **Example.** A pilot aboard a hull 10 m over the home belt and a walker beside it are two views from one gateway: the
//! planet's shard names the 128 m tiles both read, solves the missing ones, and answers each tile's levels 5, 4, … down
//! to the finest either reads into the gateway's cache; the gateway hands each of them the levels its own distance reads.

use std::collections::BTreeMap;
use std::sync::Arc;

use bevy_ecs::prelude::*;
use vd_core::TickId;
use vd_wire::session_flow::{ArtifactView, BulkAudience};

use super::artifact_ship::ArtifactWants;
use super::config::{ttl_alive, window_ttl_ticks};
use super::exterior::RealmStore;
use super::{RealmAuthority, StubConfig, StubStats};
use crate::io::{Bytes, Store};
use crate::runtime::{ClockSample, OutboundBox};

/// ★ THE PACE: how many bytes one tick answers to one gateway — 256 KiB a tick, 5 MiB a second at twenty ticks, beside the
/// artifact's own tiles. In BYTES, not parts (ruling B88): a far tile's coarse level is one small part, and a pace of
/// eight parts a tick spent a whole tick on eight such parts. A stated pace, never a cap: every part arrives, and at least
/// one goes every tick.
pub const FINE_BYTES_PER_TICK: usize = 256 * 1024;

/// A fine tile's name on this side of the seam: its fine level's number (the record's `level` byte: 1 the 128 m level, 2
/// the 16 m level), its face and its index there (ruling B95: one machinery for every level).
pub type FineName = (u8, u8, u32, u32);

/// The level a gateway holds nothing of a tile from: past the coarsest.
pub const NONE_HELD: u8 = u8::MAX;

/// The provider of the fine tiles' bytes and their work, injected by the composition root; the sim reads bytes.
pub trait FineTiles: Send + Sync {
    /// ★ THE WORK, once a tick before the answers: the workers' answers landed and written to `store` (this realm's
    /// own, when it has one), and the next work started from the reasons this tick named or held.
    fn tick(&self, store: Option<&mut dyn Store>);
    /// ★ THE TILES a view at `radial_m` from the realm's centre along the unit direction `dir` (the realm's own frame)
    /// reads, the nearest first, each with the finest level it reads — and the provider's reason to solve them.
    fn want(&self, dir: [f64; 3], radial_m: f64) -> Vec<(FineName, u8)>;
    /// Holds `tiles` as reasons this tick (a view set named them on an earlier tick), each with the finest level a view
    /// reads it at — level 0 is the tile read whole, a coarser level only its far copies (which the provider serves when
    /// stored and never solves, send-ahead's first rule).
    fn hold(&self, tiles: &[(FineName, u8)]);
    /// The mean levels a finished tile of the fine level `level` (its number) carries, plus one (the level count).
    fn levels(&self, level: u8) -> u8;
    /// The parts of one finished tile's levels `lo` up to (not including) `hi` — `hi` [`NONE_HELD`] for the levels to
    /// the coarsest — each encoded as the wire's `BulkMsg::FineTilePart`; `None` while the tile is not finished.
    fn parts(&self, tile: FineName, lo: u8, hi: u8) -> Option<Arc<[Bytes]>>;
    /// ★ THE FAR LAKES' PARTS (ruling B116 part 2): the realm's far lake list, each part encoded as the wire's
    /// `BulkMsg::ArtifactLakes`; `None` until the provider has folded every lake's tiles.
    fn far_lakes(&self) -> Option<Arc<[Bytes]>>;
}

/// The injected provider: `None` until the artifact is here and the fine level applies (a dry body, a body still
/// solving, a test rig).
#[derive(Resource, Default)]
pub struct FineSource(pub Option<Box<dyn FineTiles>>);

impl std::fmt::Debug for FineSource {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("FineSource")
            .field("present", &self.0.is_some())
            .finish()
    }
}

/// ★ ONE GATEWAY'S FINE RECORD (ruling B87): the views its sessions stated in their last beat and when, whether they
/// are new, the tiles they read (each with the finest level any of them reads, the nearest first), the finest level of
/// each tile already answered to the gateway's cache, and the record on its way — the tile, its level, the parts sent.
#[derive(Debug, Default, PartialEq)]
pub struct FineGateway {
    pub views: Vec<ArtifactView>,
    pub views_at: TickId,
    pub fresh: bool,
    pub named: Vec<(FineName, u8)>,
    pub sent: BTreeMap<FineName, u8>,
    pub current: Option<(FineName, u8, usize)>,
    /// ★ Whether the realm's far lake list went to this gateway (ruling B116 part 2): once, when it exists.
    pub far_sent: bool,
}

impl FineGateway {
    /// One view stated at `now`: a view of a new beat starts the set over.
    pub fn state(&mut self, view: ArtifactView, now: TickId) {
        if self.views_at != now {
            self.views.clear();
            self.views_at = now;
        }
        self.views.push(view);
        self.fresh = true;
    }
}

/// ★ ONE TICK'S ANSWERS to one gateway: up to [`FINE_BYTES_PER_TICK`] of `named`'s levels the gateway lacks — at least one
/// part — the record on its way first, then [`next_record`]'s. Each part goes through `send`; returns the records finished.
fn answer(
    source: &dyn FineTiles,
    named: &[(FineName, u8)],
    sent: &mut BTreeMap<FineName, u8>,
    current: &mut Option<(FineName, u8, usize)>,
    send: &mut dyn FnMut(Bytes),
) -> u64 {
    let mut budget = FINE_BYTES_PER_TICK;
    let mut finished = 0u64;
    while budget > 0 {
        let Some((tile, level, from)) = current.take().or_else(|| next_record(source, named, sent))
        else {
            break;
        };
        let Some(parts) = source.parts(tile, level, level + 1) else {
            continue;
        };
        let mut to = from;
        while (to < parts.len()) & (budget > 0) {
            budget = budget.saturating_sub(parts[to].len());
            send(parts[to].clone());
            to += 1;
        }
        if to == parts.len() {
            sent.insert(tile, level);
            finished += 1;
        } else {
            *current = Some((tile, level, to));
        }
    }
    finished
}

/// ★ THE NEXT RECORD: the nearest named tile that still lacks a level down to the finest it is read at, and its next level
/// — the one under the finest it holds, so its levels join, the coarsest first; a tile not finished is passed over.
/// `None` when every named tile holds what it is read at. THE NEAREST TILE WHOLE FIRST, never every tile's coarsest level
/// first: MEASURED (the walker on the home belt, ruling B88's runs) the level-by-level order built 2 157 pieces of ground
/// on the rough view and rebuilt them as the finer levels came, the frame at 93 ms and 3 305 pieces drawn at 90 s; the
/// near pieces the client builds first need the near tiles' finest levels first.
fn next_record(
    source: &dyn FineTiles,
    named: &[(FineName, u8)],
    sent: &BTreeMap<FineName, u8>,
) -> Option<(FineName, u8, usize)> {
    named.iter().find_map(|&(tile, need)| {
        let next = sent
            .get(&tile)
            .copied()
            .unwrap_or_else(|| source.levels(tile.0))
            .checked_sub(1)?;
        (next >= need)
            .then(|| source.parts(tile, next, next + 1))
            .flatten()
            .map(|_| (tile, next, 0))
    })
}

/// ★ THE EMITTER, once a tick after the artifact's: the reasons, the work, the answers.
#[allow(clippy::too_many_arguments)]
pub(crate) fn emit_fine(
    config: Res<StubConfig>,
    clock: Res<ClockSample>,
    authority: Res<RealmAuthority>,
    source: Res<FineSource>,
    mut store: ResMut<RealmStore>,
    mut wants: ResMut<ArtifactWants>,
    mut stats: ResMut<StubStats>,
    mut outbox: ResMut<OutboundBox>,
) {
    let Some(source) = source.0.as_deref() else {
        return;
    };
    // The views lapse with a window keep-alive's lifetime: a gateway that stopped stating them has no viewer left.
    let ttl = window_ttl_ticks(&config);
    let now = clock.local_tick;
    wants.2.retain(|_, g| ttl_alive(g.views_at, now, ttl));
    // ★ THE REASONS, before the work: a new view set is named, an older one held.
    for g in wants.2.values_mut() {
        if g.fresh {
            let mut merged: BTreeMap<FineName, u8> = BTreeMap::new();
            let mut order: Vec<FineName> = Vec::new();
            for view in &g.views {
                for (tile, need) in source.want(view.dir, view.radial_m) {
                    let level = merged.entry(tile).or_insert_with(|| {
                        order.push(tile);
                        need
                    });
                    *level = (*level).min(need);
                }
            }
            g.named = order.into_iter().map(|t| (t, merged[&t])).collect();
            // A tile no view reads any more is let go on both sides: the gateway drops it from its cache by the same
            // views, and answers it again when a view reads it later.
            g.sent.retain(|t, _| merged.contains_key(t));
            g.fresh = false;
        } else {
            source.hold(&g.named);
        }
    }
    source.tick(store.0.as_deref_mut().map(|s| s as &mut dyn Store));
    let Some(realm_fence) = authority.0 else {
        return;
    };
    // ★ THE ANSWERS: each gateway's named tiles to its cache for this realm, one level a record, the coarsest first.
    for (gateway, g) in &mut wants.2 {
        let mut send = |part: Bytes| {
            super::artifact_ship::push_bulk(
                &mut outbox,
                *gateway,
                realm_fence,
                BulkAudience::Realm(config.realm),
                part,
            );
            stats.fine_parts_sent += 1;
        };
        let finished = answer(source, &g.named, &mut g.sent, &mut g.current, &mut send);
        // ★ THE FAR LAKES (ruling B116 part 2): the realm's list once to each gateway, the moment it exists.
        if !g.far_sent
            && let Some(parts) = source.far_lakes()
        {
            for part in parts.iter() {
                send(part.clone());
            }
            g.far_sent = true;
        }
        stats.fine_tiles_sent += finished;
    }
}
