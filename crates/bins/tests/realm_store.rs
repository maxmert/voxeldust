//! ★ A REALM REMEMBERS WHAT PEOPLE BUILT — through a real file on disk (D-MOVE-2).
//!
//! A shard has never opened a file before this. These gates drive the shipped opener and the shipped
//! reader, against a real store, so the round trip is proven end to end rather than in memory.
use vd_core::built::{Berth, BlueprintId, BuiltBody, BuiltFacts};
use vd_core::entity_kind::EntityKind;
use vd_core::fence::Fence;
use vd_core::geometry::Boundary;
use vd_core::glam::DVec3;
use vd_core::ids::{AccountId, EntityId};
use vd_core::pose::RealmId;
use vd_sim::io::Store as _;
use vd_sim::stub::built_store as bs;

fn ship(seq: u64) -> RealmId {
    RealmId::Ship(EntityId::pack(EntityKind::Ship, 1, seq, 0))
}

fn a_body(realm: RealmId) -> BuiltBody {
    BuiltBody {
        realm,
        owner: AccountId(1000),
        blueprint: BlueprintId(7),
        bound: Boundary::Shell { r: 20.0 },
        look: Boundary::Shell { r: 20.0 },
        facts: BuiltFacts {
            mass_g: 50_000_000,
            cross_section_mm2: 12_000_000,
            drag_micro: 820_000,
            max_push_micro_mps2: 98_100_000,
            max_turn_micro_radps2: 800_000,
        },
        fence: Fence(1),
    }
}

fn a_berth(child: RealmId, x: f64) -> Berth {
    Berth {
        child,
        offset_m: DVec3::new(x, 0.0, 0.0),
        bound: Boundary::Shell { r: 20.0 },
        look: Boundary::Shell { r: 20.0 },
        fence: Fence(1),
    }
}

/// Open a store at `path` with the realm role, as the shard does.
fn open(path: &std::path::Path) -> vd_io_prod::store::RedbStore {
    // ★ THE SAME LABEL THE SHIPPED BOOT BUILDS. A hand-written seed here would pass while production
    // failed — which is exactly what happened the first time this was written: the helper said 7, the
    // tool read the world's own seed, and the file correctly refused to open for a different world.
    let stamp = vd_bins::durable_stamp(
        &vd_bins::process_env(),
        vd_core::store_stamp::StoreRole::RealmStore,
        vd_core::EpochId(0),
    )
    .expect("the realm store's label");
    let (store, _d) = vd_io_prod::store::RedbStore::open(
        path,
        vd_io_prod::store::StoreTuning::default(),
        stamp,
    )
    .unwrap_or_else(|e| panic!("a realm store opens at {}: {e}", path.display()));
    store
}

#[test]
fn a_realm_reads_back_exactly_what_it_wrote() {
    let dir = std::env::temp_dir().join(format!("vd-realm-store-{}", std::process::id()));
    std::fs::create_dir_all(&dir).expect("a temp dir");
    let path = dir.join("realm.redb");
    let _ = std::fs::remove_file(&path);
    {
        let mut store = open(&path);
        store.put(&bs::body_key(), &bs::encode_body(&a_body(ship(1))).into());
        for (seq, x) in [(1_u64, 1000.0_f64), (2, 2000.0), (3, 3000.0)] {
            store.put(&bs::berth_key(ship(seq)), &bs::encode_berth(&a_berth(ship(seq), x)).into());
        }
        // `put` only STAGES. `commit` is the durability barrier, and `flush` blocks until that commit
        // is actually on disk — `commit` alone returns before its own write is fsync'd.
        store.commit();
        store.flush();
        // ★ THE FILE IS HELD EXCLUSIVELY, and the lock lives until the store's off-tick writer thread
        // has actually finished. Dropping the value is what joins that thread, so the release must be
        // explicit here — otherwise the reopen below races the writer and is refused with
        // "Database already open".
        //
        // That exclusivity is a PROPERTY worth having: two shards can never open one realm's file, so
        // a realm cannot be simulated twice by accident.
        drop(store);
    }
    // A NEW process opens the same file — which is the case that matters, because a store that only
    // works while its writer is alive is a cache, not a store.
    let store = open(&path);
    let (berths, body) = vd_bins::read_realm_store(&store).expect("the rows read back");
    let body = body.expect("the body was stored");
    assert_eq!(body.realm, ship(1), "the whole minted name survived the disk");
    assert_eq!(body.owner, AccountId(1000));
    assert_eq!(body.facts.mass_g, 50_000_000);
    assert_eq!(berths.len(), 3, "every berth came back in ONE scan");
    let mut xs: Vec<f64> = berths.iter().map(|b| b.offset_m.x).collect();
    xs.sort_by(f64::total_cmp);
    assert_eq!(xs, vec![1000.0, 2000.0, 3000.0]);
}

#[test]
fn a_file_from_a_different_world_is_refused_and_not_misread() {
    // ★ THE LABEL IS LOAD-BEARING. The rows carry no field names, so a file written by a world that
    // measures distance differently decodes WITHOUT COMPLAINT and puts a hangar in the wrong place.
    // The label is read before any row, so the refusal happens before anything is misplaced.
    let dir = std::env::temp_dir().join(format!("vd-realm-store-refuse-{}", std::process::id()));
    std::fs::create_dir_all(&dir).expect("a temp dir");
    let path = dir.join("realm.redb");
    let _ = std::fs::remove_file(&path);
    {
        let mut store = open(&path);
        store.put(&bs::body_key(), &bs::encode_body(&a_body(ship(1))).into());
        // `put` only STAGES. `commit` is the durability barrier, and `flush` blocks until that commit
        // is actually on disk — `commit` alone returns before its own write is fsync'd.
        store.commit();
        store.flush();
    }
    let other_world = vd_core::store_stamp::StoreStamp::new(
        vd_core::store_stamp::StoreRole::RealmStore,
        // A different universe seed: the same role, the same shapes, a different world. Stated by hand
        // HERE deliberately — this test is about a label that does NOT match, so it must not be built
        // by the same helper the writer used.
        999_999,
        vd_core::EpochId(0),
        &vd_physics::worldgen::world_shape_constants(),
    );
    let refused = vd_io_prod::store::RedbStore::open(
        &path,
        vd_io_prod::store::StoreTuning::default(),
        other_world,
    );
    assert!(
        refused.is_err(),
        "a store from another world must refuse to open, never be read"
    );
}

#[test]
fn a_realm_with_no_store_boots_exactly_as_it_always_did() {
    // Every existing cluster and every existing test states no path. They must get no file and behave
    // byte-identically — an absent store is not an empty one, it is the world before this existed.
    // SAFETY: this test process states no realm store, which is what every existing cluster does.
    unsafe { std::env::remove_var("VD_REALM_STORE") };
    let store = vd_bins::open_realm_store(&vd_bins::process_env())
        .expect("no path is not an error");
    assert!(store.is_none(), "no path means no store, and no change");
}

#[test]
fn the_shipyards_stand_in_writes_rows_the_shard_reads() {
    // ★ THE WHOLE POINT OF THE TOOL, end to end: it writes the two rows, and the SHIPPED reader — the
    // one a shard calls at boot — reads them back. Nothing here reaches inside the tool.
    //
    // When the shipyard replaces this tool it writes the same two rows into the same two files, so
    // this gate keeps passing without being rewritten. That is the growth claim, tested.
    let dir = std::env::temp_dir().join(format!("vd-build-ship-{}", std::process::id()));
    std::fs::create_dir_all(&dir).expect("a temp dir");
    let parent_store = dir.join("system.redb");
    let ship_store = dir.join("ship.redb");
    let _ = std::fs::remove_file(&parent_store);
    let _ = std::fs::remove_file(&ship_store);

    let out = std::process::Command::new(env!("CARGO_BIN_EXE_vd-build-ship"))
        .args([
            "--parent-store",
            &parent_store.display().to_string(),
            "--ship-store",
            &ship_store.display().to_string(),
            "--owner",
            "1000",
            "--berth-x-m",
            "1500",
        ])
        .output()
        .expect("the tool runs");
    assert!(
        out.status.success(),
        "the tool built a ship: {}",
        String::from_utf8_lossy(&out.stderr)
    );

    // The PARENT's file holds one berth, and nothing else.
    let (berths, no_body) = vd_bins::read_realm_store(&open(&parent_store)).expect("the parent's rows");
    assert_eq!(berths.len(), 1, "the parent authored exactly one berth");
    assert!((berths[0].offset_m.x - 1500.0).abs() < 1e-9, "at the berth the tool was given");
    assert!(no_body.is_none(), "a parent holds no body of its own here");

    // The SHIP's file holds the body, and no berths — it has authored none.
    let (no_berths, body) = vd_bins::read_realm_store(&open(&ship_store)).expect("the ship's rows");
    let body = body.expect("the hull has a body");
    assert!(no_berths.is_empty(), "a hull with no built children authored no berths");
    assert_eq!(body.owner, AccountId(1000), "and it belongs to somebody");
    assert_eq!(body.realm, berths[0].child, "the two rows name the SAME hull");
    assert!(
        body.facts.max_push_micro_mps2 > 0,
        "the hull states its own push, not a shared constant"
    );
}

#[test]
fn a_hull_written_by_the_tool_is_a_realm_in_the_world_the_shard_boots() {
    // ★ THE WHOLE OF STEP 4 AND 5, END TO END. The tool writes a berth. The shipped reader reads it.
    // The shipped world build lowers it. The hull is a realm in this world, with a band and an
    // interest radius it got from the code a planet gets them from.
    //
    // Nothing here reaches inside anything: the tool is a process, the reader and the world build are
    // the ones the shard calls at boot.
    let dir = std::env::temp_dir().join(format!("vd-built-world-{}", std::process::id()));
    std::fs::create_dir_all(&dir).expect("a temp dir");
    let parent_store = dir.join("system.redb");
    let ship_store = dir.join("ship.redb");
    let _ = std::fs::remove_file(&parent_store);
    let _ = std::fs::remove_file(&ship_store);

    let out = std::process::Command::new(env!("CARGO_BIN_EXE_vd-build-ship"))
        .args([
            "--parent-store", &parent_store.display().to_string(),
            "--ship-store", &ship_store.display().to_string(),
            "--owner", "1000",
            "--berth-x-m", "1000",
        ])
        .output()
        .expect("the tool runs");
    assert!(out.status.success(), "{}", String::from_utf8_lossy(&out.stderr));

    // What the shard does at boot: read the berths, then build the world with them.
    let (berths, _body) = vd_bins::read_realm_store(&open(&parent_store)).expect("the berths");
    assert_eq!(berths.len(), 1, "the tool wrote one berth");
    let parent = RealmId::System(7);
    let held: std::collections::BTreeSet<RealmId> = std::iter::once(parent).collect();
    let with: Vec<_> = berths.iter().map(|b| (parent, *b)).collect();

    let dev = vd_bins::DEV;
    let (regions, _m, _l) = vd_bins::boot_world_built(
        dev.universe_seed, &held, parent, dev.move_speed, dev.tick_dt, &held, &with,
    );
    let hull = regions
        .iter()
        .find(|r| r.realm == berths[0].child)
        .expect("the hull the tool built is a realm in this world");
    assert_eq!(hull.parent, Some(parent), "berthed in the realm that authored it");
    assert!(hull.aoi.spin_up_r_m() > 0.0, "and it wakes by the generic rule");

    // And with NO berths the same call gives the world that was there before any of this existed.
    let (plain, _m, _l) = vd_bins::boot_world_built(
        dev.universe_seed, &held, parent, dev.move_speed, dev.tick_dt, &held, &[],
    );
    assert_eq!(regions.len(), plain.len() + 1, "exactly one realm added, nothing else moved");
}
