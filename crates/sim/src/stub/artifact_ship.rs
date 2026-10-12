//! ★ THE ARTIFACT SHIP — the shard's side (the landform arc, slice 8c stage C4c and the far-view
//! ship; the owner's ruling of 2026-09-19: the solve runs once on the server and its artifact is
//! shipped to every client; SL3: a realm draws itself, whoever looks).
//!
//! The sim never names the generator: what it ships are BYTES a provider hands it — the
//! [`TileSource`] the composition root injects once the artifact is here (the shape of
//! `RealmStore`). Two ships run every tick:
//!
//! 1. **THE WANTS** — a gateway that composes a picture with this realm in it, at any rung, for any
//!    session (in orbit, on the ground, passing by) has read this realm's artifact digest off the
//!    realm's own look bag and asked for the artifact (`GatewayToShard::ArtifactWant`). The shard
//!    answers with the HEAD and the PYRAMID's parts, paced, as `BulkFor { audience: Realm }`: the
//!    gateway caches them once per realm and serves every session that draws the realm. A want at
//!    a digest this realm does not hold is dropped and counted.
//! 2. **THE TILES** — ONE PATH for every viewer (ruling B88, the owner: *"one way only — through gateway"*): each
//!    gateway states once a beat the VIEW every session behind it has of this realm (`GatewayToShard::ArtifactWant
//!    { view }`: a direction and a distance from the realm's centre) — the realm's own occupants among them, whose eye
//!    the gateway reads off the snapshot this realm sends their clients, a pilot aboard a hull over it, anyone who
//!    draws it. The tiles a view reaches go to the gateway's cache for this realm, a bounded number a tick, each once
//!    per gateway; the gateway serves every session the ones its own view reaches. No tile goes to a session directly.
//!
//! Each goes out on the reliable Control lane to the asking gateway, which caches the bytes without
//! decoding a tile. Nothing crosses UPWARD and no pose leaves this realm (SL2, SL7).
//!
//! **Example.** A pilot in a hull in orbit sees the home planet. Her gateway reads the planet's
//! artifact digest off the planet's look bag (relayed through the star system's window), asks the
//! planet's shard once, receives the head and the six pyramid levels over a few ticks, and serves
//! her the globe from orbit; when she lands and walks out, her view of the planet is her avatar's
//! own position, and the tiles around her follow it four a tick into the gateway's cache.

use std::collections::{BTreeMap, BTreeSet};

use bevy_ecs::prelude::*;
use vd_core::{Fence, NodeId};
use vd_wire::session_flow::{ArtifactView, BulkAudience, ShardToGateway};

use super::{RealmAuthority, StubConfig, StubStats};
use crate::io::{Bytes, Durability, MsgClass};
use crate::runtime::OutboundBox;

/// ★ THE PACE: how many tiles one tick ships to one gateway — four of 40 KB is 160 KB a tick, 3.2
/// MB a second at twenty ticks, which a reliable link carries without starving the lane the login
/// shares (the sky's own lesson, 2026-08-29). A stated pace, never a cap: every tile arrives.
pub const TILES_PER_TICK: usize = 4;
/// How many pyramid parts one tick ships to one asking gateway.
pub const PYRAMID_PARTS_PER_TICK: usize = 8;

/// The provider of the artifact's bytes, injected by the composition root; the sim reads bytes.
pub trait TileSource: Send + Sync {
    /// The artifact's digest — what the realm states under `TAG_ARTIFACT`, what a want names.
    fn digest(&self) -> [u64; 2];
    /// The head, encoded as the wire's `BulkMsg::ArtifactHead`.
    fn head(&self) -> Bytes;
    /// ★ THE REALM'S PARTS in order, ONE LIST the sim pages through (2026-09-22, the coast mask):
    /// the pyramid's parts first, each encoded as `BulkMsg::ArtifactPyramid`, then the coast
    /// mask's, each encoded as `BulkMsg::ArtifactCoast`. The sim never names a shape: it ships
    /// the list, paced, and the head says how many of each to expect.
    fn artifact_parts(&self) -> Vec<Bytes>;
    /// The tiles a view standing `radial_m` from the realm's centre along the unit direction `dir`
    /// (in the realm's own frame) reaches, in a stated order, as far as the source's own reach rule
    /// says the fine rungs read — the eye's height read over the ground the source's artifact states
    /// under it, the client's own ground (2026-10-01; over the radius, a pilot over a deep floor
    /// waited on tiles never sent). One rule for every viewer (ruling B88).
    fn tiles_under(&self, dir: [f64; 3], radial_m: f64) -> Vec<(u8, u32, u32)>;
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

/// ★ A GATEWAY'S WANT for this realm's artifact at a digest, and how far the answer has gone:
/// the head goes first, then the parts in order, paced.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Want {
    pub digest: [u64; 2],
    /// Whether the head went out.
    pub head_sent: bool,
    /// How many pyramid parts went out.
    pub parts_sent: usize,
}

/// ★ A GATEWAY'S TILE WANTS (the one tile path, 2026-09-20): the views its sessions stated since
/// the last tick, the tiles those views reach that have not gone out, and the tiles this gateway
/// holds already (it caches them per realm). A head want from the gateway starts the record over:
/// a new head means a new cache on the gateway's side.
#[derive(Debug, Default, PartialEq)]
pub struct TileWant {
    pub views: Vec<ArtifactView>,
    pub queued: std::collections::VecDeque<(u8, u32, u32)>,
    pub sent: BTreeSet<(u8, u32, u32)>,
}

/// The wants by asking gateway — ONE head want per gateway (a newer want replaces an older,
/// unfinished one), one tile record per gateway, and ★ one FINE record per gateway (ruling B87: the
/// views its sessions stated in their last beat, the fine tiles they read, what went to its cache).
#[derive(Resource, Debug, Default)]
pub struct ArtifactWants(
    pub BTreeMap<NodeId, Want>,
    pub BTreeMap<NodeId, TileWant>,
    pub BTreeMap<NodeId, super::fine_ship::FineGateway>,
);

/// ★ THE WANT ARRIVES (`GatewayToShard::ArtifactWant`): recorded for this realm, refused for any
/// other (the gateway asked the wrong shard — counted, never guessed at).
#[allow(clippy::too_many_arguments)]
pub(crate) fn on_artifact_want(
    from: NodeId,
    realm: vd_core::pose::RealmId,
    digest: [u64; 2],
    view: Option<ArtifactView>,
    now: vd_core::TickId,
    config: &StubConfig,
    wants: &mut ArtifactWants,
    stats: &mut StubStats,
) {
    if realm != config.realm {
        stats.artifact_wants_foreign += 1;
        return;
    }
    // A viewer's want: the tiles its view reaches, answered by the emitter when the source is here —
    // and ★ the fine tiles it reads (ruling B87), the gateway's views of one beat replacing the last's.
    if let Some(view) = view {
        wants.1.entry(from).or_default().views.push(view);
        wants.2.entry(from).or_default().state(view, now);
        stats.artifact_wants_received += 1;
        return;
    }
    // A head want starts the gateway's tile records over: its cache is new.
    wants.1.remove(&from);
    wants.2.remove(&from);
    wants.0.insert(
        from,
        Want {
            digest,
            head_sent: false,
            parts_sent: 0,
        },
    );
    stats.artifact_wants_received += 1;
}

/// ★ THE EMITTER, once a tick after the interest frames: the wants, then the viewers' tiles.
pub(crate) fn emit_artifact(
    config: Res<StubConfig>,
    authority: Res<RealmAuthority>,
    source: Res<ArtifactSource>,
    mut wants: ResMut<ArtifactWants>,
    mut stats: ResMut<StubStats>,
    mut outbox: ResMut<OutboundBox>,
) {
    let Some(realm_fence) = authority.0 else {
        return;
    };
    let Some(source) = source.0.as_deref() else {
        return;
    };
    // ★ THE WANTS: a want at another digest is stale (the gateway read an older statement, or
    // this realm re-solved) and is dropped; a matching one is answered head first, then parts,
    // paced, and forgotten when whole.
    //
    // THE PARTS ARE ASKED ONLY WHEN A WANT NEEDS THEM (measured on the second flight,
    // 2026-09-20): asking the source for every part on every tick cost the home planet's shard
    // 96 ms a tick — five times its budget — from the tick its artifact landed, with no want at
    // all; the source's parts are the wire's own bytes, and the bins source now holds them once.
    let digest = source.digest();
    let mut parts: Option<Vec<Bytes>> = None;
    let mut done: Vec<NodeId> = Vec::new();
    for (gateway, want) in &mut wants.0 {
        if want.digest != digest {
            stats.artifact_wants_stale += 1;
            done.push(*gateway);
            continue;
        }
        let parts = parts.get_or_insert_with(|| source.artifact_parts());
        if !want.head_sent {
            push_bulk(
                &mut outbox,
                *gateway,
                realm_fence,
                BulkAudience::Realm(config.realm),
                source.head(),
            );
            want.head_sent = true;
            stats.artifact_heads_sent += 1;
        }
        let first = want.parts_sent.min(parts.len());
        let last = (first + PYRAMID_PARTS_PER_TICK).min(parts.len());
        for part in &parts[first..last] {
            push_bulk(
                &mut outbox,
                *gateway,
                realm_fence,
                BulkAudience::Realm(config.realm),
                part.clone(),
            );
            stats.artifact_pyramid_parts_sent += 1;
        }
        want.parts_sent = last;
        if last == parts.len() {
            done.push(*gateway);
        }
    }
    for gateway in done {
        wants.0.remove(&gateway);
    }
    // ★ THE VIEWERS' TILES (the one tile path): every view a gateway stated reaches the tiles
    // the source's own rule names; the ones not yet sent to that gateway queue, nearest first,
    // and go out four a tick on the realm audience, each once per gateway.
    let mut stale_views: Vec<NodeId> = Vec::new();
    for (gateway, want) in &mut wants.1 {
        let views = std::mem::take(&mut want.views);
        for view in views {
            for tile in source.tiles_under(view.dir, view.radial_m) {
                if !want.sent.contains(&tile) && !want.queued.contains(&tile) {
                    want.queued.push_back(tile);
                }
            }
        }
        let mut sent = 0usize;
        while sent < TILES_PER_TICK {
            let Some(tile) = want.queued.pop_front() else {
                break;
            };
            let Some(bytes) = source.tile(tile.0, tile.1, tile.2) else {
                continue;
            };
            push_bulk(
                &mut outbox,
                *gateway,
                realm_fence,
                BulkAudience::Realm(config.realm),
                bytes,
            );
            want.sent.insert(tile);
            stats.artifact_tiles_sent += 1;
            sent += 1;
        }
        if want.queued.is_empty() && want.sent.is_empty() {
            stale_views.push(*gateway);
        }
    }
    for gateway in stale_views {
        wants.1.remove(&gateway);
    }
}

/// One `BulkFor` to a gateway, on the reliable Control lane, ephemeral: a gateway that restarts
/// asks afresh.
pub(crate) fn push_bulk(
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
    use vd_core::SessionId;

    /// The push: one bulk on the Control lane to the gateway named, ephemeral, with the audience
    /// and the bytes given.
    #[test]
    fn a_bulk_goes_to_the_gateway_on_the_control_lane() {
        let mut outbox = OutboundBox::default();
        push_bulk(
            &mut outbox,
            NodeId(2),
            Fence(3),
            BulkAudience::Realm(vd_core::pose::RealmId::Planet(7)),
            crate::io::bytes(vec![0xA0]),
        );
        push_bulk(
            &mut outbox,
            NodeId(2),
            Fence(3),
            BulkAudience::Sessions(vec![SessionId(7)]),
            crate::io::bytes(vec![0xC0]),
        );
        let (to, class, bytes, durability) = &outbox.0[0];
        assert_eq!(
            (*to, *class, *durability),
            (NodeId(2), MsgClass::Control, Durability::Ephemeral)
        );
        assert_eq!(
            postcard::from_bytes::<ShardToGateway>(bytes).expect("decodes"),
            ShardToGateway::BulkFor {
                realm_fence: Fence(3),
                audience: BulkAudience::Realm(vd_core::pose::RealmId::Planet(7)),
                bytes: vec![0xA0],
            }
        );
        assert_eq!(outbox.0.len(), 2);
        assert_eq!(
            format!("{:?}", ArtifactSource(None)),
            "ArtifactSource { present: false }"
        );
    }
}
