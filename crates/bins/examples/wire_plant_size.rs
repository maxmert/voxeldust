//! The wire plant's measurement (the voxel foundation, slice 4): the encoded size, in bytes, of every
//! planted shape with nothing in it but its own header, so slice 9's byte budgets start from facts.
//! No runtime cost exists until a producer does. Run:
//!
//! ```text
//! cargo run --release -p vd-bins --example wire_plant_size
//! ```

use vd_core::UniverseTick;
use vd_core::geometry::Boundary;
use vd_core::grid::{CellAddr, ChunkCoord, Face, Rung};
use vd_core::look::{SELF_LOOK_BUDGET_BYTES, SurfaceStmt, surface_look_bag};
use vd_core::pose::{FrameRef, RealmId};
use vd_core::realm_coord::RealmCoord;
use vd_core::realm_path::{RealmKindTag, RealmLevel, RealmPath};
use vd_core::{Fence, SessionId};
use vd_wire::channels::{
    BlockEdit, BulkMsg, ChunkRow, ClientControlMsg, ServerControlMsg, WorldAction,
};
use vd_wire::intershard::{ChildFelt, InterShardFlow};
use vd_wire::session_flow::{BulkAudience, GatewayToShard, ShardToGateway};

fn size<T: serde::Serialize>(what: &str, value: &T) -> usize {
    let n = postcard::to_allocvec(value).expect("encodes").len();
    println!("wire_plant_size: {what:<46} {n:>5} B");
    n
}

fn main() {
    let chunk = ChunkCoord {
        body: RealmId::Planet(2298),
        face: Face::PosY,
        rung: Rung::new(0).expect("rung"),
        x: 1_000_000,
        y: -1_000_000,
        z: 1_000_000,
    };
    let cell = CellAddr {
        body: RealmId::Planet(2298),
        face: Face::PosY,
        rung: Rung::new(0).expect("rung"),
        i: 60_000_000,
        j: -60_000_000,
        k: 60_000_000,
    };
    let empty_row = ChunkRow {
        key: chunk,
        at: UniverseTick(u64::MAX),
        bag: vd_core::chunk_row::chunk_row_bag(&[]),
    };
    let one_row = size(
        "ChunkRows, one empty row (the header)",
        &BulkMsg::ChunkRows {
            realm: RealmId::Planet(2298),
            rows: vec![empty_row.clone()],
        },
    );
    let bulk_for = size(
        "BulkFor, three recipients, no bytes",
        &ShardToGateway::BulkFor {
            realm_fence: Fence(u64::MAX),
            audience: BulkAudience::Sessions(vec![SessionId(u128::MAX); 3]),
            bytes: vec![],
        },
    );
    let surface = surface_look_bag(
        &Boundary::Shell { r: 1_737_400.0 },
        Some((5, 1.0)),
        &SurfaceStmt {
            frame: FrameRef::PlanetCentered { planet_seed: 2298 },
            generator: u64::MAX,
        },
        None,
    );
    let surface_b = surface.len();
    println!(
        "wire_plant_size: {:<46} {surface_b:>5} B",
        "self-look bag with luma and surface"
    );
    let felt = size(
        "ChildFelt",
        &InterShardFlow::ChildFelt(ChildFelt {
            child: RealmCoord::from_path(RealmPath::from_levels(vec![
                RealmLevel::new(RealmKindTag::Universe, 0),
                RealmLevel::new(RealmKindTag::Galaxy, 2),
                RealmLevel::new(RealmKindTag::System, 7),
            ]))
            .expect("a three-level path has a leaf"),
            parent_fence: Fence(u64::MAX),
            at: UniverseTick(u64::MAX),
            felt: [i64::MAX, i64::MIN, i64::MAX],
        }),
    );
    let action = size(
        "WorldAction, one placement",
        &ClientControlMsg::WorldAction {
            seq: u64::MAX,
            action: WorldAction::BlockEdit(BlockEdit {
                at: cell,
                kind: u16::MAX,
                variant: u8::MAX,
                scale: 3,
                orientation: 23,
                site: [7, 7, 7],
            }),
        },
    );
    let forwarded = size(
        "SessionAction, the same placement forwarded",
        &GatewayToShard::SessionAction {
            session: SessionId(u128::MAX),
            fence: Fence(u64::MAX),
            seq: u64::MAX,
            action: WorldAction::BlockEdit(BlockEdit {
                at: cell,
                kind: 0,
                variant: 0,
                scale: 0,
                orientation: 0,
                site: [0, 0, 0],
            }),
        },
    );
    let hello = size(
        "HelloWorld",
        &ClientControlMsg::HelloWorld {
            declared: u64::MAX,
            measured: u64::MAX,
        },
    );
    let refused = size(
        "WorldRefused",
        &ServerControlMsg::WorldRefused {
            ours: u64::MAX,
            theirs: u64::MAX,
            half: vd_wire::channels::WorldHalf::Declared,
        },
    );
    // The gates can fail: the surface bag must fit one datagram; every header stays under a hundred
    // bytes, because a header that costs more than the record it carries is a lane defect.
    assert!(
        surface_b <= SELF_LOOK_BUDGET_BYTES,
        "the surface bag exceeds the budget: {surface_b}"
    );
    for (what, n) in [
        ("ChunkRows header", one_row),
        ("BulkFor", bulk_for),
        ("ChildFelt", felt),
        ("WorldAction", action),
        ("SessionAction", forwarded),
        ("HelloWorld", hello),
        ("WorldRefused", refused),
    ] {
        assert!(n < 100, "{what} costs {n} bytes with nothing in it");
    }
    println!("wire_plant_size: PASS");
}
