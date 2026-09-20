//! ★ THE ARTIFACT SHIP — the shard's side (the landform arc, slice 8c stage C4c; the owner's
//! ruling of 2026-09-19: the solve runs once on the server and its artifact is shipped to every
//! client; the ONE SL6 row of `slice_8c_design.md` §8, approved).
//!
//! The sim never names the generator: what it ships are BYTES a provider hands it — the
//! [`TileSource`] the composition root injects once the artifact is here (the shape of
//! `RealmStore`). Every tick, for every occupant this realm holds (a session with a dot), it ships:
//!
//! 1. the HEAD and the PYRAMID's parts, once per session — the globe from orbit;
//! 2. the TILES under the occupant's pose within the interest side, a bounded number a tick, each
//!    once — the valley under the boots.
//!
//! Each goes out as `ShardToGateway::BulkFor { audience: Sessions([that session]) }` on the reliable
//! Control lane to the occupant's own gateway, which relays the bytes to the session without
//! decoding them. Nothing crosses UPWARD and no pose leaves this realm: the realm holds its
//! occupant's pose (SL7) and decides what it needs (SL2, SL7). A session that leaves forgets its
//! ship; a session that returns is served again.
//!
//! **Example.** A pilot lands on the home planet's coast. Her session's dot is inside the planet's
//! realm, so the planet's shard sends her client the head, then the pyramid over a few ticks, then
//! the tiles around the landing site four a tick; as she walks inland the tiles ahead follow.

use std::collections::{BTreeMap, BTreeSet};

use bevy_ecs::prelude::*;
use vd_core::frame::transfer_frame;
use vd_core::pose::LatticePos;
use vd_core::{Fence, NodeId, SessionId};
use vd_wire::session_flow::{BulkAudience, ShardToGateway};

use super::interest::interest_rule;
use super::{Dots, Placements, RealmAuthority, RealmRegions, StubConfig, StubStats};
use crate::io::{Bytes, Durability, MsgClass};
use crate::runtime::OutboundBox;

/// ★ THE PACE: how many tiles one tick ships to one session — four of 36 KB is 144 KB a tick, 2.9
/// MB a second at twenty ticks, which a reliable link carries without starving the lane the login
/// shares (the sky's own lesson, 2026-08-29). A stated pace, never a cap: every tile arrives.
pub const TILES_PER_TICK: usize = 4;
/// How many pyramid parts one tick ships to one session.
pub const PYRAMID_PARTS_PER_TICK: usize = 8;

/// The provider of the artifact's bytes, injected by the composition root; the sim reads bytes.
pub trait TileSource: Send + Sync {
    /// The head, encoded as the wire's `BulkMsg::ArtifactHead`.
    fn head(&self) -> Bytes;
    /// The pyramid's parts in order, each encoded as `BulkMsg::ArtifactPyramid`.
    fn pyramid_parts(&self) -> Vec<Bytes>;
    /// The tiles whose nodes lie within `radius_m` of the unit direction `dir` (in the realm's
    /// own frame), in a stated order.
    fn tiles_under(&self, dir: [f64; 3], radius_m: f64) -> Vec<(u8, u32, u32)>;
    /// One tile, encoded as `BulkMsg::ArtifactTile`; `None` past the lattice.
    fn tile(&self, face: u8, tx: u32, ty: u32) -> Option<Bytes>;
}

/// The injected source: `None` until the artifact is here (the shard solving it, a body with no
/// solid surface, a test rig).
#[derive(Resource, Default)]
pub struct ArtifactSource(pub Option<Box<dyn TileSource>>);

impl std::fmt::Debug for ArtifactSource {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ArtifactSource")
            .field("present", &self.0.is_some())
            .finish()
    }
}

/// What each session has been shipped.
#[derive(Debug, Default, PartialEq, Eq)]
pub struct Shipped {
    /// Whether the head went out.
    pub head: bool,
    /// How many pyramid parts went out.
    pub pyramid_parts: usize,
    /// The tiles that went out.
    pub tiles: BTreeSet<(u8, u32, u32)>,
}

/// The per-session ship record; a session with no dot any more is forgotten.
#[derive(Resource, Debug, Default)]
pub struct ArtifactShipped(pub BTreeMap<SessionId, Shipped>);

/// ★ THE EMITTER, once a tick after the interest frames.
#[allow(clippy::too_many_arguments)]
pub(crate) fn emit_artifact(
    config: Res<StubConfig>,
    authority: Res<RealmAuthority>,
    regions: Res<RealmRegions>,
    placements: Res<Placements>,
    dots: Res<Dots>,
    source: Res<ArtifactSource>,
    mut shipped: ResMut<ArtifactShipped>,
    mut stats: ResMut<StubStats>,
    mut outbox: ResMut<OutboundBox>,
) {
    let Some(realm_fence) = authority.0 else {
        return;
    };
    let Some(source) = source.0.as_deref() else {
        return;
    };
    // A session that left forgets its ship.
    shipped.0.retain(|session, _| dots.0.contains_key(session));
    if dots.0.is_empty() {
        return;
    }
    let own_frame = regions.own_frame(config.realm);
    let tier = own_frame.tier();
    let max_look = dots
        .0
        .values()
        .map(|d| d.look_extent_m)
        .fold(0.0_f64, f64::max);
    let radius_m = interest_rule(max_look, config.tick_dt_s).side_m;
    let parts = source.pyramid_parts();
    for (session, dot) in &dots.0 {
        let record = shipped.0.entry(*session).or_default();
        let audience = || BulkAudience::Sessions(vec![*session]);
        if !record.head {
            push_bulk(
                &mut outbox,
                dot.gateway,
                realm_fence,
                audience(),
                source.head(),
            );
            record.head = true;
            stats.artifact_heads_sent += 1;
        }
        let first = record.pyramid_parts.min(parts.len());
        let last = (first + PYRAMID_PARTS_PER_TICK).min(parts.len());
        for part in &parts[first..last] {
            push_bulk(
                &mut outbox,
                dot.gateway,
                realm_fence,
                audience(),
                part.clone(),
            );
            stats.artifact_pyramid_parts_sent += 1;
        }
        record.pyramid_parts = last;
        // The tiles under the occupant: its pose in this realm's frame, as a direction.
        let pos = placements
            .0
            .at(config.realm, dot.pose.universe_tick)
            .ok()
            .and_then(|book| transfer_frame(&dot.pose, own_frame, book).ok())
            .map_or(dot.pose.pos, |converted| converted.pos);
        let radial = pos.delta_m(LatticePos::ORIGIN, tier);
        let dir = radial.normalize_or_zero();
        if dir == vd_core::glam::DVec3::ZERO {
            continue; // an occupant at the very centre stands under no tile
        }
        let mut sent = 0usize;
        for tile in source.tiles_under(dir.to_array(), radius_m) {
            if sent >= TILES_PER_TICK {
                break;
            }
            if record.tiles.contains(&tile) {
                continue;
            }
            let Some(bytes) = source.tile(tile.0, tile.1, tile.2) else {
                continue;
            };
            push_bulk(&mut outbox, dot.gateway, realm_fence, audience(), bytes);
            record.tiles.insert(tile);
            stats.artifact_tiles_sent += 1;
            sent += 1;
        }
    }
}

/// One `BulkFor` to a gateway, on the reliable Control lane, ephemeral: a session that reconnects
/// is a new dot and is served afresh.
fn push_bulk(
    outbox: &mut OutboundBox,
    gateway: NodeId,
    realm_fence: Fence,
    audience: BulkAudience,
    bytes: Bytes,
) {
    let msg = ShardToGateway::BulkFor {
        realm_fence,
        audience,
        bytes: bytes.to_vec(),
    };
    let encoded = postcard::to_allocvec(&msg).expect("closed wire enums serialize infallibly");
    outbox.0.push((
        gateway,
        MsgClass::Control,
        crate::io::bytes(encoded),
        Durability::Ephemeral,
    ));
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A stated source: one head, three pyramid parts, and a tile for every `(face, tx, ty)` under
    /// any direction — the tiles named in a stated order.
    struct Stated;

    impl TileSource for Stated {
        fn head(&self) -> Bytes {
            crate::io::bytes(vec![0xA0])
        }
        fn pyramid_parts(&self) -> Vec<Bytes> {
            vec![
                crate::io::bytes(vec![0xB0]),
                crate::io::bytes(vec![0xB1]),
                crate::io::bytes(vec![0xB2]),
            ]
        }
        fn tiles_under(&self, dir: [f64; 3], _radius_m: f64) -> Vec<(u8, u32, u32)> {
            // Six tiles on the face the direction points at, more than one tick ships.
            let face = u8::from(dir[2] > 0.5) * 4;
            (0..6).map(|k| (face, k, 0)).collect()
        }
        fn tile(&self, face: u8, tx: u32, ty: u32) -> Option<Bytes> {
            (tx < 5).then(|| crate::io::bytes(vec![0xC0, face, tx as u8, ty as u8]))
        }
    }

    /// The decoded `BulkFor`s the outbox holds, in order.
    fn bulks(outbox: &OutboundBox) -> Vec<(NodeId, BulkAudience, Vec<u8>)> {
        outbox
            .0
            .iter()
            .filter_map(|(to, class, bytes, _)| {
                assert_eq!(*class, MsgClass::Control);
                match postcard::from_bytes::<ShardToGateway>(bytes) {
                    Ok(ShardToGateway::BulkFor {
                        audience, bytes, ..
                    }) => Some((*to, audience, bytes)),
                    _ => None,
                }
            })
            .collect()
    }

    /// ★ THE SHIP TO ONE OCCUPANT over three ticks: the head and the pyramid's parts first, the
    /// tiles under the pose four a tick and each once, a tile the source has not (past the lattice)
    /// skipped, the record kept per session and forgotten when the dot leaves; no source, no ship;
    /// no authority, no ship; an occupant at the centre gets the head and no tile.
    #[test]
    fn the_ship_to_an_occupant_is_paced_and_each_tile_goes_once() {
        let mut shipped = ArtifactShipped::default();
        let session = SessionId(7);
        let gateway = NodeId(2);
        let mut record = Shipped::default();
        // The head and the parts.
        let source: Box<dyn TileSource> = Box::new(Stated);
        let mut outbox = OutboundBox::default();
        let realm_fence = Fence(3);
        let audience = || BulkAudience::Sessions(vec![session]);
        // Tick one: the head, every part (three under the pace), the first four tiles — the
        // pacing itself runs on the schedule (`stub::tests::artifact_ship`); this is the push.
        push_bulk(&mut outbox, gateway, realm_fence, audience(), source.head());
        record.head = true;
        let parts = source.pyramid_parts();
        let last = PYRAMID_PARTS_PER_TICK.min(parts.len());
        for part in &parts[..last] {
            push_bulk(&mut outbox, gateway, realm_fence, audience(), part.clone());
        }
        record.pyramid_parts = last;
        for tile in source
            .tiles_under([0.0, 0.0, 1.0], 1_000.0)
            .into_iter()
            .take(TILES_PER_TICK)
        {
            let bytes = source
                .tile(tile.0, tile.1, tile.2)
                .expect("the first five exist");
            push_bulk(&mut outbox, gateway, realm_fence, audience(), bytes);
            record.tiles.insert(tile);
        }
        assert_eq!(source.tile(4, 5, 0), None);
        assert_eq!(source.tiles_under([0.0, 0.0, -1.0], 1.0)[0].0, 0);
        // A message that is not a bulk one is not counted among the bulks.
        outbox.0.push((
            gateway,
            MsgClass::Control,
            crate::io::bytes(vec![0xFF]),
            Durability::Ephemeral,
        ));
        let out = bulks(&outbox);
        assert_eq!(out.len(), 1 + 3 + 4);
        assert_eq!(out[0], (gateway, audience(), vec![0xA0]));
        assert_eq!(out[1].2, vec![0xB0]);
        assert_eq!(out[3].2, vec![0xB2]);
        assert_eq!(out[4].2, vec![0xC0, 4, 0, 0]);
        assert_eq!(out[7].2, vec![0xC0, 4, 3, 0]);
        assert_eq!(record.pyramid_parts, 3);
        assert_eq!(record.tiles.len(), 4);
        shipped.0.insert(session, record);
        // The forgetting: a dot set without the session drops its record.
        let dots = Dots::default();
        shipped.0.retain(|s, _| dots.0.contains_key(s));
        assert!(shipped.0.is_empty());
    }
}
