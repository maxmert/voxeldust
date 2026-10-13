//! ★ THE ARTIFACT SHIP ON THE SCHEDULE (slice 8c stage C4c and the far-view ship): a gateway's
//! want is answered with the head and the pyramid's parts, paced, on the realm audience, once; a
//! stale want is dropped; a want for another realm is refused; the tiles a viewer's view reaches go
//! four a tick and each once to that gateway's cache; ★ ONE PATH (ruling B88): an occupant is shipped
//! no tile of its own — its gateway's view of it is; nothing ships with no source or no authority.

use super::*;
use crate::stub::artifact_ship::{ArtifactSource, ArtifactWants, TileSource};
use vd_wire::session_flow::{BulkAudience, ShardToGateway};

/// A stated source: one head, eleven pyramid parts (more than one tick's pace), six tiles under
/// any direction off the centre (five the source holds). It counts how often its parts are asked
/// for: the emitter may ask only on a tick with a want it answers. The count is the source's OWN
/// (a shared static raced between tests of one binary: 3 against 2 in the coverage run).
#[derive(Default)]
struct Stated(std::sync::Arc<std::sync::atomic::AtomicU32>);

const DIGEST: [u64; 2] = [0xD1, 0xD2];

impl TileSource for Stated {
    fn digest(&self) -> [u64; 2] {
        DIGEST
    }
    fn head(&self) -> crate::io::Bytes {
        crate::io::bytes(vec![0xA0])
    }
    fn artifact_parts(&self) -> Vec<crate::io::Bytes> {
        self.0.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
        (0..11u8).map(|k| crate::io::bytes(vec![0xB0, k])).collect()
    }
    fn tiles_under(&self, dir: [f64; 3], radial_m: f64) -> Vec<(u8, u32, u32)> {
        // The view's distance from the centre is handed to the source, in metres of the realm's
        // frame.
        assert!(radial_m > 0.0, "{radial_m}");
        // A view of the far side (down -Z) reads ground this artifact's lattice does not cover:
        // the source names tiles past the lattice, and the gateway gets no tile at all.
        if dir[2] < 0.0 {
            return (5..7).map(|k| (2, k, 0)).collect();
        }
        (0..6).map(|k| (2, k, 0)).collect()
    }
    fn tile(&self, face: u8, tx: u32, ty: u32) -> Option<crate::io::Bytes> {
        (tx < 5).then(|| crate::io::bytes(vec![0xC0, face, tx as u8, ty as u8]))
    }
}

/// The bulk bytes the shard sent to the gateway this tick with the audience given, in order.
fn bulks(sent: &[(NodeId, MsgClass, Vec<u8>)], want: &BulkAudience) -> Vec<Vec<u8>> {
    sent.iter()
        .filter(|(to, class, _)| *to == GATEWAY && *class == MsgClass::Control)
        .filter_map(
            |(_, _, bytes)| match postcard::from_bytes::<ShardToGateway>(bytes) {
                Ok(ShardToGateway::BulkFor {
                    audience, bytes, ..
                }) if audience == *want => Some(bytes),
                _ => None,
            },
        )
        .collect()
}

fn want_msg(realm: RealmId, digest: [u64; 2]) -> Inbound {
    wire_msg(
        GATEWAY,
        MsgClass::Control,
        &GatewayToShard::ArtifactWant {
            realm,
            digest,
            view: None,
        },
    )
}

fn view_msg(realm: RealmId, digest: [u64; 2], radial_m: f64) -> Inbound {
    view_msg_dir(realm, digest, [0.0, 0.0, 1.0], radial_m)
}

/// The same viewer's want, from a stated direction: a gateway that composes the realm's far side
/// states the direction its session looks from.
fn view_msg_dir(realm: RealmId, digest: [u64; 2], dir: [f64; 3], radial_m: f64) -> Inbound {
    wire_msg(
        GATEWAY,
        MsgClass::Control,
        &GatewayToShard::ArtifactWant {
            realm,
            digest,
            view: Some(vd_wire::session_flow::ArtifactView { dir, radial_m }),
        },
    )
}

/// ★ THE ONE TILE PATH: a viewer's want reaches the tiles the source names under its view,
/// which go out four a tick on the realm audience, each once per gateway; a second view of the
/// same ground adds nothing; a head want starts the record over; a view with no source waits.
#[test]
fn a_viewers_want_is_answered_with_tiles_on_the_realm_audience_each_once() {
    let mut rig = Rig::new();
    rig.grant_realm();
    let realm = config().realm;
    let audience = BulkAudience::Realm(realm);
    // Before the source is here the view waits.
    let _ = rig.tick(vec![view_msg(realm, DIGEST, 1_000.0)]);
    assert!(bulks(&rig.tick(vec![]), &audience).is_empty());
    assert_eq!(rig.world.resource::<ArtifactWants>().1.len(), 1);
    *rig.world.resource_mut::<ArtifactSource>() = ArtifactSource(Some(Box::new(Stated::default())));
    // The source names six tiles (five it holds): four this tick, one the next, then nothing.
    let out = bulks(&rig.tick(vec![]), &audience);
    assert_eq!(out.len(), 4);
    assert_eq!(out[0], vec![0xC0, 2, 0, 0]);
    let out = bulks(&rig.tick(vec![]), &audience);
    assert_eq!(out, vec![vec![0xC0, 2, 4, 0]]);
    assert!(bulks(&rig.tick(vec![]), &audience).is_empty());
    // The same view again: every tile is held by that gateway already.
    let _ = rig.tick(vec![view_msg(realm, DIGEST, 1_000.0)]);
    assert!(bulks(&rig.tick(vec![]), &audience).is_empty());
    assert_eq!(rig.world.resource::<StubStats>().artifact_tiles_sent, 5);
    // A head want starts the record over: the tiles go again after the head and the parts.
    let out = bulks(
        &rig.tick(vec![
            want_msg(realm, DIGEST),
            view_msg(realm, DIGEST, 1_000.0),
        ]),
        &audience,
    );
    assert!(out.iter().any(|b| b == &vec![0xC0, 2, 0, 0]), "{out:?}");
}

#[test]
fn a_gateways_want_is_answered_paced_on_the_realm_audience_once() {
    let mut rig = Rig::new();
    rig.grant_realm();
    let realm = config().realm;
    let audience = BulkAudience::Realm(realm);
    // A want before the source is here waits; a want for another realm is refused.
    let _ = rig.tick(vec![
        want_msg(realm, DIGEST),
        want_msg(RealmId::Planet(99), DIGEST),
    ]);
    assert_eq!(rig.world.resource::<StubStats>().artifact_wants_received, 1);
    assert_eq!(rig.world.resource::<StubStats>().artifact_wants_foreign, 1);
    assert_eq!(rig.world.resource::<ArtifactWants>().0.len(), 1);
    let counter = std::sync::Arc::new(std::sync::atomic::AtomicU32::new(0));
    *rig.world.resource_mut::<ArtifactSource>() =
        ArtifactSource(Some(Box::new(Stated(std::sync::Arc::clone(&counter)))));
    let asked = || counter.load(std::sync::atomic::Ordering::SeqCst);
    let asked_before = asked();
    // Tick one: the head and eight parts; tick two: the last three, and the want is done.
    let out = bulks(&rig.tick(vec![]), &audience);
    assert_eq!(out.len(), 1 + 8);
    assert_eq!(out[0], vec![0xA0]);
    assert_eq!(out[1], vec![0xB0, 0]);
    assert_eq!(out[8], vec![0xB0, 7]);
    let out = bulks(&rig.tick(vec![]), &audience);
    assert_eq!(out, vec![vec![0xB0, 8], vec![0xB0, 9], vec![0xB0, 10]]);
    assert!(rig.world.resource::<ArtifactWants>().0.is_empty());
    assert_eq!(asked() - asked_before, 2, "once per answering tick");
    // ★ With no want the source is never asked for its parts (the 96 ms tick of the second
    // flight): a quiet tick costs nothing.
    assert!(bulks(&rig.tick(vec![]), &audience).is_empty());
    assert_eq!(asked() - asked_before, 2);
    // A stale want (another digest) is dropped and counted, and asks for no parts; a fresh one
    // is answered again.
    let _ = rig.tick(vec![want_msg(realm, [1, 1])]);
    assert!(bulks(&rig.tick(vec![]), &audience).is_empty());
    assert_eq!(rig.world.resource::<StubStats>().artifact_wants_stale, 1);
    assert_eq!(asked() - asked_before, 2);
    // A want that arrives with the source here is answered on the same tick (the handler runs
    // before the emitter): the head and eight parts at once.
    assert_eq!(
        bulks(&rig.tick(vec![want_msg(realm, DIGEST)]), &audience).len(),
        9
    );
    {
        let stats = rig.world.resource::<StubStats>();
        assert_eq!(stats.artifact_heads_sent, 2);
        assert_eq!(stats.artifact_pyramid_parts_sent, 11 + 8);
        assert_eq!(stats.artifact_wants_received, 3);
    }
}

/// ★ ONE PATH (ruling B88, the owner: *"one way only — through gateway"*): an occupant standing over
/// tiles is shipped none of its own, on any audience; the tiles under it go to its gateway's cache
/// when the gateway states its view, as every other viewer's do. Example: a walker stands on the home
/// belt; the planet's shard sends her nothing directly, and her gateway's view of her brings the
/// tiles under her boots into its cache, from which she is served.
#[test]
fn an_occupant_is_shipped_no_tile_of_its_own() {
    let mut rig = Rig::new();
    rig.grant_realm();
    let _ = rig.attach();
    let realm = config().realm;
    let source = ArtifactSource(Some(Box::new(Stated::default())));
    assert_eq!(format!("{source:?}"), "ArtifactSource { present: true }");
    *rig.world.resource_mut::<ArtifactSource>() = source;
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
    for _ in 0..3 {
        let sent = rig.tick(vec![]);
        assert!(bulks(&sent, &BulkAudience::Sessions(vec![SESSION])).is_empty());
        assert!(bulks(&sent, &BulkAudience::Realm(realm)).is_empty());
    }
    assert_eq!(rig.world.resource::<StubStats>().artifact_tiles_sent, 0);
    // Her gateway's view of her: the tiles under it go to the cache, four this tick.
    let out = bulks(
        &rig.tick(vec![view_msg(realm, DIGEST, 1_000.0)]),
        &BulkAudience::Realm(realm),
    );
    assert_eq!(out.len(), 4);
}

/// ★ A TILE QUEUES ONCE PER GATEWAY, whatever its sessions ask: two views of the same ground in
/// one tick name the same tiles, and the second view adds nothing the first one queued. Example:
/// a pilot and her wingman look at the same valley through one gateway; the planet's shard queues
/// that valley once and ships it once.
#[test]
fn two_views_in_one_tick_queue_each_tile_once() {
    let mut rig = Rig::new();
    rig.grant_realm();
    let realm = config().realm;
    let audience = BulkAudience::Realm(realm);
    *rig.world.resource_mut::<ArtifactSource>() = ArtifactSource(Some(Box::new(Stated::default())));
    // Both views name the same six tiles: the first queues them, the second finds them queued.
    let out = bulks(
        &rig.tick(vec![
            view_msg(realm, DIGEST, 1_000.0),
            view_msg(realm, DIGEST, 2_000.0),
        ]),
        &audience,
    );
    assert_eq!(out.len(), 4);
    assert_eq!(out[0], vec![0xC0, 2, 0, 0]);
    // The queue held two tiles, never eight: the next tick ships the fifth, and the sixth is past
    // the lattice.
    let out = bulks(&rig.tick(vec![]), &audience);
    assert_eq!(out, vec![vec![0xC0, 2, 4, 0]]);
    assert_eq!(bulks(&rig.tick(vec![]), &audience).len(), 0);
    assert_eq!(rig.world.resource::<StubStats>().artifact_tiles_sent, 5);
}

/// ★ A TILE RECORD THAT SHIPS NOTHING IS DROPPED: a view over ground this artifact does not cover
/// names tiles past the lattice, so the queue drains with no tile out and the shard forgets that
/// gateway's tile record. Example: a gateway composes the planet's far side; the planet's shard
/// holds no tile there, ships nothing, and keeps no record of the ask.
#[test]
fn a_tile_record_that_ships_nothing_is_dropped() {
    let mut rig = Rig::new();
    rig.grant_realm();
    let realm = config().realm;
    *rig.world.resource_mut::<ArtifactSource>() = ArtifactSource(Some(Box::new(Stated::default())));
    let sent = rig.tick(vec![view_msg_dir(realm, DIGEST, [0.0, 0.0, -1.0], 1_000.0)]);
    assert_eq!(bulks(&sent, &BulkAudience::Realm(realm)).len(), 0);
    assert_eq!(rig.world.resource::<StubStats>().artifact_tiles_sent, 0);
    // The gateway asked, the shard shipped nothing, and the record is gone.
    assert_eq!(rig.world.resource::<StubStats>().artifact_wants_received, 1);
    assert_eq!(rig.world.resource::<ArtifactWants>().1.len(), 0);
}

/// Without the realm's authority nothing ships, whatever the source holds or is asked.
#[test]
fn the_ship_waits_for_the_realms_authority() {
    let mut rig = Rig::new();
    *rig.world.resource_mut::<ArtifactSource>() = ArtifactSource(Some(Box::new(Stated::default())));
    let _ = rig.attach_request(SESSION, GATEWAY);
    let _ = rig.tick(vec![want_msg(config().realm, DIGEST)]);
    let sent = rig.tick(vec![]);
    assert!(bulks(&sent, &BulkAudience::Realm(config().realm)).is_empty());
    assert!(bulks(&sent, &BulkAudience::Sessions(vec![SESSION])).is_empty());
    assert_eq!(rig.world.resource::<StubStats>().artifact_heads_sent, 0);
}

/// A source whose ground under any view is twenty tiles, all held.
struct Twenty;

impl TileSource for Twenty {
    fn digest(&self) -> [u64; 2] {
        DIGEST
    }
    fn head(&self) -> crate::io::Bytes {
        crate::io::bytes(vec![0xA0])
    }
    fn artifact_parts(&self) -> Vec<crate::io::Bytes> {
        Vec::new()
    }
    fn tiles_under(&self, _dir: [f64; 3], _radial_m: f64) -> Vec<(u8, u32, u32)> {
        (0..20).map(|k| (1, k, 0)).collect()
    }
    fn tile(&self, face: u8, tx: u32, ty: u32) -> Option<crate::io::Bytes> {
        Some(crate::io::bytes(vec![0xC1, face, tx as u8, ty as u8]))
    }
}

/// ★ TWENTY TILES UNDER ONE VIEW SHIP FOUR A TICK, EACH ONCE: the queue drains over five ticks and
/// every tile goes to the gateway's cache once. Example: a pilot flies low along a coast; the shard
/// queues each new tile under her gateway's view and ships it once.
#[test]
fn twenty_tiles_under_one_view_ship_four_a_tick_each_once() {
    let mut rig = Rig::new();
    rig.grant_realm();
    let realm = config().realm;
    *rig.world.resource_mut::<ArtifactSource>() = ArtifactSource(Some(Box::new(Twenty)));
    let audience = BulkAudience::Realm(realm);
    let first = bulks(&rig.tick(vec![view_msg(realm, DIGEST, 1_000.0)]), &audience).len();
    let rest: usize = (0..5)
        .map(|_| bulks(&rig.tick(vec![]), &audience).len())
        .sum();
    assert_eq!(
        (first, first + rest),
        (4, 20),
        "four a tick, every one of the twenty"
    );
    assert_eq!(
        rig.world.resource::<ArtifactWants>().1[&GATEWAY].sent.len(),
        20
    );
    assert_eq!(rig.world.resource::<StubStats>().artifact_tiles_sent, 20);
}
