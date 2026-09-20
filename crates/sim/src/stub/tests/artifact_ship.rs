//! ★ THE ARTIFACT SHIP ON THE SCHEDULE (slice 8c stage C4c): the emitter runs after the window
//! rosters for every occupant the realm holds, ships the head and the pyramid once, the tiles
//! under the pose four a tick and each once, forgets a session that leaves, and ships nothing with
//! no source, no authority, or an occupant at the centre.

use super::*;
use crate::stub::artifact_ship::{ArtifactShipped, ArtifactSource, TileSource};
use vd_wire::session_flow::{BulkAudience, ShardToGateway};

/// A stated source: one head, three pyramid parts, six tiles under any direction off the centre
/// (five the source holds).
struct Stated;

impl TileSource for Stated {
    fn head(&self) -> crate::io::Bytes {
        crate::io::bytes(vec![0xA0])
    }
    fn pyramid_parts(&self) -> Vec<crate::io::Bytes> {
        vec![
            crate::io::bytes(vec![0xB0]),
            crate::io::bytes(vec![0xB1]),
            crate::io::bytes(vec![0xB2]),
        ]
    }
    fn tiles_under(&self, _dir: [f64; 3], _radius_m: f64) -> Vec<(u8, u32, u32)> {
        (0..6).map(|k| (2, k, 0)).collect()
    }
    fn tile(&self, face: u8, tx: u32, ty: u32) -> Option<crate::io::Bytes> {
        (tx < 5).then(|| crate::io::bytes(vec![0xC0, face, tx as u8, ty as u8]))
    }
}

/// The bulk bytes the shard sent to the gateway this tick, in order.
fn bulks(sent: &[(NodeId, MsgClass, Vec<u8>)]) -> Vec<Vec<u8>> {
    sent.iter()
        .filter(|(to, class, _)| *to == GATEWAY && *class == MsgClass::Control)
        .filter_map(
            |(_, _, bytes)| match postcard::from_bytes::<ShardToGateway>(bytes) {
                Ok(ShardToGateway::BulkFor {
                    audience, bytes, ..
                }) => {
                    assert_eq!(audience, BulkAudience::Sessions(vec![SESSION]));
                    Some(bytes)
                }
                _ => None,
            },
        )
        .collect()
}

#[test]
fn the_ship_runs_on_the_schedule_for_the_occupant_the_realm_holds() {
    let mut rig = Rig::new();
    rig.grant_realm();
    let _ = rig.attach();
    // No source: nothing shipped, and the record is empty.
    let sent = rig.tick(vec![]);
    assert!(bulks(&sent).is_empty());
    assert!(rig.world.resource::<ArtifactShipped>().0.is_empty());
    let source = ArtifactSource(Some(Box::new(Stated)));
    assert_eq!(format!("{source:?}"), "ArtifactSource { present: true }");
    assert_eq!(
        format!("{:?}", ArtifactSource(None)),
        "ArtifactSource { present: false }"
    );
    *rig.world.resource_mut::<ArtifactSource>() = source;
    // The occupant stands off the centre, so tiles are under it.
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
    // Tick one: the head, the three parts, four tiles.
    let out = bulks(&rig.tick(vec![]));
    assert_eq!(out.len(), 1 + 3 + 4);
    assert_eq!(out[0], vec![0xA0]);
    assert_eq!(out[1], vec![0xB0]);
    assert_eq!(out[3], vec![0xB2]);
    assert_eq!(out[4], vec![0xC0, 2, 0, 0]);
    assert_eq!(out[7], vec![0xC0, 2, 3, 0]);
    // Tick two: the last tile the source holds; the sixth is past the lattice and skipped.
    let out = bulks(&rig.tick(vec![]));
    assert_eq!(out, vec![vec![0xC0, 2, 4, 0]]);
    // Tick three: everything went once; nothing more.
    assert!(bulks(&rig.tick(vec![])).is_empty());
    {
        let stats = rig.world.resource::<StubStats>();
        assert_eq!(stats.artifact_heads_sent, 1);
        assert_eq!(stats.artifact_pyramid_parts_sent, 3);
        assert_eq!(stats.artifact_tiles_sent, 5);
        let shipped = rig.world.resource::<ArtifactShipped>();
        let record = shipped.0.get(&SESSION).expect("a record");
        assert!(record.head);
        assert_eq!(record.pyramid_parts, 3);
        assert_eq!(record.tiles.len(), 5);
    }
    // An occupant at the very centre stands under no tile: a fresh session there gets the head
    // and the parts only.
    let centre = SessionId(0xBB);
    let mut dot = rig.world.resource::<Dots>().0[&SESSION];
    dot.pose = StampedPose::at_rest(config().frame, DVec3::ZERO, UniverseTick(5));
    rig.world.resource_mut::<Dots>().0.insert(centre, dot);
    let sent = rig.tick(vec![]);
    let to_centre: Vec<Vec<u8>> = sent
        .iter()
        .filter_map(
            |(_, _, bytes)| match postcard::from_bytes::<ShardToGateway>(bytes) {
                Ok(ShardToGateway::BulkFor {
                    audience, bytes, ..
                }) if audience == BulkAudience::Sessions(vec![centre]) => Some(bytes),
                _ => None,
            },
        )
        .collect();
    assert_eq!(to_centre.len(), 4);
    assert!(to_centre.iter().all(|b| b[0] != 0xC0));
    // The session that leaves is forgotten; the one that stays keeps its record.
    rig.world.resource_mut::<Dots>().0.remove(&SESSION);
    let _ = rig.tick(vec![]);
    let shipped = rig.world.resource::<ArtifactShipped>();
    assert!(!shipped.0.contains_key(&SESSION));
    assert!(shipped.0.contains_key(&centre));
    // With every dot gone the record empties and nothing is shipped.
    rig.world.resource_mut::<Dots>().0.clear();
    assert!(bulks(&rig.tick(vec![])).is_empty());
    assert!(rig.world.resource::<ArtifactShipped>().0.is_empty());
}

/// Without the realm's authority nothing ships, whatever the source holds.
#[test]
fn the_ship_waits_for_the_realms_authority() {
    let mut rig = Rig::new();
    *rig.world.resource_mut::<ArtifactSource>() = ArtifactSource(Some(Box::new(Stated)));
    let _ = rig.attach_request(SESSION, GATEWAY);
    assert!(bulks(&rig.tick(vec![])).is_empty());
    assert_eq!(rig.world.resource::<StubStats>().artifact_heads_sent, 0);
}
