//! ★ THE FINE SHIP ON THE SCHEDULE (the fine ground's way in, B2; rulings B86 item 3, B87 and B88): the provider's work
//! runs every tick, with the realm's own store when it has one; every viewer is answered by the one call by distance
//! through its gateway's cache for the realm — ONE PATH (ruling B88): the realm's own occupants are shipped nothing of
//! their own — one level a record, the nearest tile first and its levels the coarsest first, a byte pace a tick a gateway
//! with at least one part, the record on its way first; a tile not finished is passed over; a record whose parts vanish
//! mid-way is let go; a gateway's views of one beat replace the last's and lapse with the window keep-alive's lifetime; a
//! head want starts a gateway's record over; nothing ships with no provider or no authority.

use super::*;
use crate::stub::artifact_ship::ArtifactWants;
use crate::stub::config::window_ttl_ticks;
use crate::stub::fine_ship::{FINE_BYTES_PER_TICK, FineName, FineSource, FineTiles, NONE_HELD};
use std::sync::atomic::{AtomicBool, AtomicU32, Ordering};
use std::sync::{Arc, Mutex};
use vd_wire::session_flow::{BulkAudience, ShardToGateway};

/// A second gateway, whose sessions read the realm from farther off.
const OTHER_GATEWAY: NodeId = NodeId(21);

/// A level-nought part of the first tile: a quarter of a tick's pace, so four fill a tick.
const LARGE: usize = FINE_BYTES_PER_TICK / 4;

/// A stated provider of six levels (its tiles 128 m tiles, the fine level 1): three tiles named under any direction off the centre — the first read at the level
/// `need0` (nought unless told), its level nought in ten [`LARGE`] parts and every other level in one small part; the
/// second read at level 2, one small part a level; the third not finished. It counts its ticks and the ticks that held a
/// store, keeps the tiles it was told to hold, and its first tile's parts vanish once told to. A part's first five bytes
/// name its tile, its index and its levels.
#[derive(Default)]
struct Stated {
    ticks: Arc<AtomicU32>,
    with_store: Arc<AtomicU32>,
    vanish: Arc<AtomicBool>,
    need0: Arc<AtomicU32>,
    held: Arc<Mutex<Vec<FineName>>>,
}

impl FineTiles for Stated {
    fn tick(&self, store: Option<&mut dyn crate::io::Store>) {
        self.ticks.fetch_add(1, Ordering::SeqCst);
        if store.is_some() {
            self.with_store.fetch_add(1, Ordering::SeqCst);
        }
    }
    fn want(&self, _dir: [f64; 3], radial_m: f64) -> Vec<(FineName, u8)> {
        assert!(radial_m > 0.0, "{radial_m}");
        vec![
            ((1, 2, 0, 0), self.need0.load(Ordering::SeqCst) as u8),
            ((1, 2, 1, 0), 2),
            ((1, 2, 2, 0), 0),
        ]
    }
    fn hold(&self, tiles: &[(FineName, u8)]) {
        self.held
            .lock()
            .expect("the held list")
            .extend(tiles.iter().map(|(tile, _)| *tile));
    }
    fn levels(&self, level: u8) -> u8 {
        assert_eq!(level, 1, "the stated tiles are 128 m tiles");
        6
    }
    fn parts(&self, tile: FineName, lo: u8, hi: u8) -> Option<Arc<[crate::io::Bytes]>> {
        let count = match tile {
            (1, 2, 0, 0) if !self.vanish.load(Ordering::SeqCst) => {
                if lo == 0 {
                    10u8
                } else {
                    1
                }
            }
            (1, 2, 1, 0) => 1,
            _ => return None,
        };
        let size = if (tile == (1, 2, 0, 0)) & (lo == 0) {
            LARGE
        } else {
            5
        };
        Some(
            (0..count)
                .map(|k| {
                    let mut part = vec![0xF0, tile.2 as u8, k, lo, hi];
                    part.resize(size, 0);
                    crate::io::bytes(part)
                })
                .collect(),
        )
    }
}

/// The bulk bytes the shard sent to `to` this tick with the audience given, in order.
fn bulks_to(
    sent: &[(NodeId, MsgClass, Vec<u8>)],
    to: NodeId,
    audience: &BulkAudience,
) -> Vec<Vec<u8>> {
    sent.iter()
        .filter(|(at, class, _)| *at == to && *class == MsgClass::Control)
        .filter_map(
            |(_, _, bytes)| match postcard::from_bytes::<ShardToGateway>(bytes) {
                Ok(ShardToGateway::BulkFor {
                    audience: a, bytes, ..
                }) if a == *audience => Some(bytes),
                _ => None,
            },
        )
        .collect()
}

/// The first five bytes of each bulk the shard sent to the gateway this tick with the audience given, in order: the
/// part's tile, index and levels.
fn bulks(sent: &[(NodeId, MsgClass, Vec<u8>)], audience: &BulkAudience) -> Vec<Vec<u8>> {
    heads(bulks_to(sent, GATEWAY, audience))
}

/// The first five bytes of each part.
fn heads(parts: Vec<Vec<u8>>) -> Vec<Vec<u8>> {
    parts.into_iter().map(|p| p[..5].to_vec()).collect()
}

/// A gateway's want for this realm: a viewer's view (a direction and a distance), or the head (`None`).
fn want_from(gateway: NodeId, view: Option<f64>) -> Inbound {
    wire_msg(
        gateway,
        MsgClass::Control,
        &GatewayToShard::ArtifactWant {
            realm: config().realm,
            digest: [1, 2],
            view: view.map(|radial_m| vd_wire::session_flow::ArtifactView {
                dir: [0.0, 0.0, 1.0],
                radial_m,
            }),
        },
    )
}

/// The gateway's want ([`want_from`] the rig's gateway).
fn want(view: Option<f64>) -> Inbound {
    want_from(GATEWAY, view)
}

/// ★ ONE PATH (ruling B88): an occupant standing over the tiles is shipped nothing of its own; its gateway's view of it is
/// answered by distance into the gateway's cache, one level a record, the coarsest first, the record on its way first; a
/// second gateway reading from farther off gets the coarser levels alone; a record whose parts vanish mid-way is let go.
#[test]
fn a_view_is_answered_by_distance_one_level_a_record_the_coarsest_first() {
    let mut rig = Rig::new();
    let cache = BulkAudience::Realm(config().realm);
    // No provider: nothing shipped.
    assert!(bulks(&rig.tick(vec![]), &cache).is_empty());
    // No authority yet: the provider's work runs, nothing ships.
    let first = Stated::default();
    let ticks = Arc::clone(&first.ticks);
    let source = FineSource(Some(Box::new(first)));
    assert_eq!(format!("{source:?}"), "FineSource { present: true }");
    *rig.world.resource_mut::<FineSource>() = source;
    let _ = rig.tick(vec![]);
    assert_eq!(ticks.load(Ordering::SeqCst), 1);
    // The realm granted and an occupant attached off the centre, then a fresh provider: nothing goes to her session.
    *rig.world.resource_mut::<FineSource>() = FineSource::default();
    rig.grant_realm();
    let _ = rig.attach();
    rig.world
        .resource_mut::<Dots>()
        .0
        .get_mut(&SESSION)
        .expect("the attached dot")
        .pose = StampedPose::at_rest(
        config().frame,
        DVec3::new(0.0, 0.0, 1_000.0),
        UniverseTick(5),
    );
    let stated = Stated::default();
    let (vanish, need0) = (Arc::clone(&stated.vanish), Arc::clone(&stated.need0));
    *rig.world.resource_mut::<FineSource>() = FineSource(Some(Box::new(stated)));
    let me = BulkAudience::Sessions(vec![SESSION]);
    let sent = rig.tick(vec![]);
    assert!(bulks(&sent, &me).is_empty());
    assert!(bulks(&sent, &cache).is_empty());
    // Her gateway's view of her, the nearest tile first. Tick one: the first tile's levels 5 … 1 and four of its level
    // nought's ten large parts (the tick's bytes spent); tick two: four more; tick three: the last two, then the second
    // tile's levels 5 … 2; then nothing (the third tile not finished).
    let sent = rig.tick(vec![want(Some(1_000.0))]);
    assert!(bulks(&sent, &me).is_empty());
    assert_eq!(
        bulks(&sent, &cache),
        vec![
            vec![0xF0, 0, 0, 5, 6],
            vec![0xF0, 0, 0, 4, 5],
            vec![0xF0, 0, 0, 3, 4],
            vec![0xF0, 0, 0, 2, 3],
            vec![0xF0, 0, 0, 1, 2],
            vec![0xF0, 0, 0, 0, 1],
            vec![0xF0, 0, 1, 0, 1],
            vec![0xF0, 0, 2, 0, 1],
            vec![0xF0, 0, 3, 0, 1],
        ]
    );
    assert_eq!(
        rig.world.resource::<ArtifactWants>().2[&GATEWAY].current,
        Some(((1, 2, 0, 0), 0, 4))
    );
    let out = bulks(&rig.tick(vec![]), &cache);
    assert_eq!(
        (out.len(), out[0].clone(), out[3].clone()),
        (4, vec![0xF0, 0, 4, 0, 1], vec![0xF0, 0, 7, 0, 1])
    );
    assert_eq!(
        bulks(&rig.tick(vec![]), &cache),
        vec![
            vec![0xF0, 0, 8, 0, 1],
            vec![0xF0, 0, 9, 0, 1],
            vec![0xF0, 1, 0, 5, 6],
            vec![0xF0, 1, 0, 4, 5],
            vec![0xF0, 1, 0, 3, 4],
            vec![0xF0, 1, 0, 2, 3],
        ]
    );
    assert!(bulks(&rig.tick(vec![]), &cache).is_empty());
    {
        let stats = rig.world.resource::<StubStats>();
        assert_eq!((stats.fine_parts_sent, stats.fine_tiles_sent), (19, 10));
        let record = &rig.world.resource::<ArtifactWants>().2[&GATEWAY];
        assert_eq!(
            record
                .sent
                .iter()
                .map(|(t, l)| (*t, *l))
                .collect::<Vec<_>>(),
            vec![((1, 2, 0, 0), 0), ((1, 2, 1, 0), 2)]
        );
        assert_eq!(record.current, None);
    }
    // A second gateway whose session reads the first tile only at level 3: its levels 5 … 3, then the second tile's
    // 5 … 2 — one tick.
    need0.store(3, Ordering::SeqCst);
    let out = heads(bulks_to(
        &rig.tick(vec![want_from(OTHER_GATEWAY, Some(5_000.0))]),
        OTHER_GATEWAY,
        &cache,
    ));
    assert_eq!(
        out,
        vec![
            vec![0xF0, 0, 0, 5, 6],
            vec![0xF0, 0, 0, 4, 5],
            vec![0xF0, 0, 0, 3, 4],
            vec![0xF0, 1, 0, 5, 6],
            vec![0xF0, 1, 0, 4, 5],
            vec![0xF0, 1, 0, 3, 4],
            vec![0xF0, 1, 0, 2, 3],
        ]
    );
    // The first gateway's next beat reads the first tile at level nought again after a head want (a new cache); its
    // level nought is on its way when its parts vanish — let go, and the second tile goes on.
    need0.store(0, Ordering::SeqCst);
    let _ = rig.tick(vec![want(None)]);
    let _ = rig.tick(vec![want(Some(1_000.0))]);
    assert_eq!(
        rig.world.resource::<ArtifactWants>().2[&GATEWAY].current,
        Some(((1, 2, 0, 0), 0, 4))
    );
    vanish.store(true, Ordering::SeqCst);
    assert_eq!(
        bulks(&rig.tick(vec![]), &cache),
        vec![
            vec![0xF0, 1, 0, 5, 6],
            vec![0xF0, 1, 0, 4, 5],
            vec![0xF0, 1, 0, 3, 4],
            vec![0xF0, 1, 0, 2, 3]
        ]
    );
    assert_eq!(
        rig.world.resource::<ArtifactWants>().2[&GATEWAY].current,
        None
    );
}

/// ★ B87: a viewer outside the realm — a pilot aboard a hull over it, a station in its orbit — is answered through its
/// gateway's cache for the realm by the same call, one level a record, the coarsest first. Each beat is its own tick (the
/// rig's clock stands still unless a test moves it, and a beat's views replace the last beat's by the tick they came in).
#[test]
fn a_gateways_views_are_answered_to_its_realm_cache_and_lapse_with_the_keep_alive() {
    let mut rig = Rig::new();
    rig.grant_realm();
    let stated = Stated::default();
    let held = Arc::clone(&stated.held);
    *rig.world.resource_mut::<FineSource>() = FineSource(Some(Box::new(stated)));
    let cache = BulkAudience::Realm(config().realm);
    let mut beat = 0u64;
    let mut at_next_tick = |rig: &mut Rig, inbound: Vec<Inbound>| {
        beat += 1;
        rig.set_local_tick(beat);
        rig.tick(inbound)
    };
    // A view: the first tile's levels 5 … 1 and four of its level-nought parts to the realm's cache this tick.
    let out = bulks(&at_next_tick(&mut rig, vec![want(Some(2_000.0))]), &cache);
    assert_eq!(
        (out.len(), out[0].clone(), out[5].clone()),
        (9, vec![0xF0, 0, 0, 5, 6], vec![0xF0, 0, 0, 0, 1])
    );
    // Between beats the named tiles are held as reasons; the answer goes on.
    let out = bulks(&at_next_tick(&mut rig, vec![]), &cache);
    assert_eq!((out.len(), out[0].clone()), (4, vec![0xF0, 0, 4, 0, 1]));
    assert!(held.lock().expect("the held list").contains(&(1, 2, 0, 0)));
    // The views of a new beat replace the last's; the same tiles add nothing new.
    let _ = at_next_tick(&mut rig, vec![want(Some(2_000.0)), want(Some(2_500.0))]);
    assert_eq!(
        rig.world.resource::<ArtifactWants>().2[&GATEWAY]
            .views
            .len(),
        2
    );
    let _ = at_next_tick(&mut rig, vec![]);
    assert!(bulks(&at_next_tick(&mut rig, vec![]), &cache).is_empty());
    // A head want starts the gateway's record over (its cache is new): the levels go again on the next view.
    let _ = at_next_tick(&mut rig, vec![want(None)]);
    assert!(
        !rig.world
            .resource::<ArtifactWants>()
            .2
            .contains_key(&GATEWAY)
    );
    let out = bulks(&at_next_tick(&mut rig, vec![want(Some(2_000.0))]), &cache);
    assert_eq!(out[0], vec![0xF0, 0, 0, 5, 6]);
    // No view for a keep-alive's lifetime: the gateway is forgotten.
    let ttl = window_ttl_ticks(&config());
    for _ in 0..=ttl + 1 {
        let _ = at_next_tick(&mut rig, vec![]);
    }
    assert!(rig.world.resource::<ArtifactWants>().2.is_empty());
    assert_eq!(NONE_HELD, u8::MAX);
}

/// The work reads the realm's own store when the shard holds one.
#[test]
fn the_providers_work_reads_the_realms_own_store() {
    let mut rig = Rig::new();
    rig.grant_realm();
    let stated = Stated::default();
    let with_store = Arc::clone(&stated.with_store);
    *rig.world.resource_mut::<FineSource>() = FineSource(Some(Box::new(stated)));
    *rig.world
        .resource_mut::<crate::stub::exterior::RealmStore>() =
        crate::stub::exterior::RealmStore(Some(Box::new(crate::io::mem::MemStore::default())));
    let _ = rig.tick(vec![]);
    assert_eq!(with_store.load(Ordering::SeqCst), 1);
}
