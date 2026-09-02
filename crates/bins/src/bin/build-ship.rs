//! ★ `vd-build-ship` — the first writer of a built realm (D-MOVE-2; owner rulings 2026-09-01).
//!
//! **THIS IS THE SHIPYARD'S STAND-IN.** The owner's vision is that a player brings a blueprint to a
//! shipyard, pays a price calculated from the materials, and robots build the hull block by block —
//! and *"only when a ship is built is a realm created, and it is saved."*
//!
//! There is no shipyard yet. So an operator writes the same two rows, into the same two files, and the
//! rows do not change when the shipyard arrives. **Only who fills them in does, and that is the whole
//! design:** the boot path reads a record and never learns who wrote it.
//!
//! **IT RUNS WHILE THE CLUSTER IS STOPPED, and that is not a limitation to fix.** A realm's file is
//! held exclusively — two writers can never open one — so a single offline writer cannot race anybody.
//! No lock, no message, no new lane, no new law. When a shipyard writes these rows it will do so from
//! inside the running realm that owns the file, which is the same exclusivity seen from the other side.
//!
//! ```text
//! vd-build-ship --parent-store <path> --ship-store <path> --owner <id> --parent <realm>
//! ```
use std::process::ExitCode;
use vd_core::built::{Berth, BlueprintId, BuiltBody, BuiltFacts};
use vd_core::entity_kind::EntityKind;
use vd_core::fence::Fence;
use vd_core::geometry::Boundary;
use vd_core::glam::DVec3;
use vd_core::ids::{AccountId, EntityId};
use vd_core::pose::RealmId;
use vd_sim::io::Store as _;
use vd_sim::stub::built_store as bs;

fn main() -> ExitCode {
    match run() {
        Ok(realm) => {
            println!("built {realm}");
            ExitCode::SUCCESS
        }
        Err(e) => {
            eprintln!("vd-build-ship: {e}");
            ExitCode::FAILURE
        }
    }
}

fn run() -> Result<RealmId, String> {
    let args = Args::parse(std::env::args().skip(1))?;
    let realm = mint_ship(args.mint_shard, args.seq);
    // ★ A BOX, NOT A BALL — and the reason is a measurement, not a taste. The drawing code turns a
    // shell into a sphere, and a sphere looks identical however it is turned: a hull drawn as one shows
    // NOTHING when it flies. A box is drawn as a cuboid, so its length states its heading on screen.
    //
    // The long axis is Z, and forward is NEGATIVE Z — that is the stick's own mapping, not a choice
    // made here. So the -Z end is the nose.
    let bound = Boundary::Aabb {
        half: DVec3::new(args.half_x_m, args.half_y_m, args.half_z_m),
    };

    let body = BuiltBody {
        realm,
        owner: AccountId(args.owner),
        // Nothing reads a blueprint yet. It is written anyway: a row stored without one can never gain
        // it later, and every hull built before blueprints existed would be a hull nobody can repair.
        blueprint: BlueprintId(0),
        bound,
        look: bound,
        facts: BuiltFacts {
            mass_g: args.mass_g,
            cross_section_mm2: args.cross_section_mm2,
            drag_micro: args.drag_micro,
            max_push_micro_mps2: args.max_push_micro_mps2,
            max_turn_micro_radps2: args.max_turn_micro_radps2,
        },
        // GENESIS: nothing has moved this row yet.
        fence: Fence::GENESIS,
    };
    // Said out loud, because the rating never crosses to a screen: the pilot's panel shows the
    // fraction the throttle commands, and this line is where the whole number is read.
    println!(
        "rating: push {} m/s^2, turn {} rad/s^2, mass {} kg",
        args.max_push_micro_mps2 as f64 / 1.0e6,
        args.max_turn_micro_radps2 as f64 / 1.0e6,
        args.mass_g as f64 / 1.0e3
    );
    let berth = Berth {
        child: realm,
        // ★ WHERE IT STARTS, AND ONLY WHERE IT STARTS. From its first tick the parent authors this
        // hull's placement from the pushes it states, so this offset is its berth and nothing more.
        offset_m: DVec3::new(args.berth_x_m, args.berth_y_m, args.berth_z_m),
        bound,
        look: bound,
        fence: Fence::GENESIS,
    };

    // ★ THE CHILD'S ROW FIRST, THEN THE PARENT'S. If the second write fails, the world holds a hull
    // nobody can reach — which is recoverable by writing the berth again. The other order would leave
    // a parent pointing at a hull with no body, which is a berth that refuses every boot.
    write_one(&args.ship_store, &bs::body_key(), &bs::encode_body(&body))?;
    write_one(
        &args.parent_store,
        &bs::berth_key(realm),
        &bs::encode_berth(&berth),
    )?;
    Ok(realm)
}

/// One row, committed and flushed to its own file.
///
/// **STAGING IS NOT WRITING.** `put` stages, `commit` is the barrier, and `flush` waits for the disk —
/// `commit` alone returns before its own write is safe. A tool that stopped at `put` would print
/// success and store nothing.
fn write_one(path: &std::path::Path, key: &[u8], value: &[u8]) -> Result<(), String> {
    if let Some(parent) = path.parent().filter(|p| !p.as_os_str().is_empty()) {
        std::fs::create_dir_all(parent).map_err(|e| format!("{}: {e}", parent.display()))?;
    }
    let stamp = vd_bins::durable_stamp(
        &vd_bins::process_env(),
        vd_core::store_stamp::StoreRole::RealmStore,
        vd_core::EpochId(0),
    )
    .map_err(|e| format!("the store's label: {e}"))?;
    let (mut store, _durability) =
        vd_io_prod::store::RedbStore::open(path, vd_io_prod::store::StoreTuning::default(), stamp)
            .map_err(|e| format!("{}: {e}", path.display()))?;
    store.put(key, &value.to_vec().into());
    store.commit();
    store.flush();
    Ok(())
}

/// A minted name — 128 bits, because a built realm is named by whichever machine builds it with no
/// central agreement. A narrower name would let two machines pick one name for two PLACES.
fn mint_ship(mint_shard: u32, seq: u64) -> RealmId {
    RealmId::Ship(EntityId::pack(EntityKind::Ship, mint_shard, seq, 0))
}

struct Args {
    parent_store: std::path::PathBuf,
    ship_store: std::path::PathBuf,
    owner: u128,
    mint_shard: u32,
    seq: u64,
    half_x_m: f64,
    half_y_m: f64,
    half_z_m: f64,
    mass_g: u64,
    cross_section_mm2: u64,
    drag_micro: u32,
    max_push_micro_mps2: i64,
    max_turn_micro_radps2: i64,
    berth_x_m: f64,
    berth_y_m: f64,
    berth_z_m: f64,
}

impl Args {
    fn parse(argv: impl Iterator<Item = String>) -> Result<Args, String> {
        let mut a = Args {
            parent_store: std::path::PathBuf::new(),
            ship_store: std::path::PathBuf::new(),
            owner: 0,
            mint_shard: 1,
            seq: 1,
            // A 12 x 6 x 40 m hull: the owner's own case for a small multi-crew vessel, shaped so a
            // human eye reads its heading. ★ PER-HULL DATA, never a world default — a blueprint states
            // these three numbers, and until blueprints exist the operator does.
            half_x_m: 6.0,
            half_y_m: 3.0,
            half_z_m: 20.0,
            mass_g: 50_000_000,
            cross_section_mm2: 12_000_000,
            drag_micro: 820_000,
            // Ten gravities, the order of magnitude the warp arithmetic uses. ★ PER-HULL DATA: a real
            // hull derives this from the thrusters built into it, which needs blocks. Until then it is
            // stated here, on ONE hull's row — never a default every ship in the world shares.
            max_push_micro_mps2: 98_100_000,
            max_turn_micro_radps2: 800_000,
            // ★ A BERTH IS MEASURED FROM THE PARENT'S CENTRE, not from where a player arrives — and a
            // star system's centre is its STAR. A hull berthed at a small offset sits inside the star,
            // and a player who spawns in the system's clearing is 1.08e10 metres away from it.
            //
            // The caller states all three axes for that reason. The shipyard will work them out from
            // where the hull was ordered; today the operator says where.
            berth_x_m: 1_000.0,
            berth_y_m: 0.0,
            berth_z_m: 0.0,
        };
        let mut it = argv.peekable();
        while let Some(flag) = it.next() {
            let mut value = || it.next().ok_or_else(|| format!("{flag} needs a value"));
            match flag.as_str() {
                "--parent-store" => a.parent_store = value()?.into(),
                "--ship-store" => a.ship_store = value()?.into(),
                "--owner" => a.owner = num(&value()?, &flag)?,
                "--mint-shard" => a.mint_shard = num(&value()?, &flag)?,
                "--seq" => a.seq = num(&value()?, &flag)?,
                "--half-x-m" => a.half_x_m = num(&value()?, &flag)?,
                "--half-y-m" => a.half_y_m = num(&value()?, &flag)?,
                "--half-z-m" => a.half_z_m = num(&value()?, &flag)?,
                "--mass-g" => a.mass_g = num(&value()?, &flag)?,
                "--max-push-micro-mps2" => a.max_push_micro_mps2 = num(&value()?, &flag)?,
                "--max-turn-micro-radps2" => a.max_turn_micro_radps2 = num(&value()?, &flag)?,
                "--berth-x-m" => a.berth_x_m = num(&value()?, &flag)?,
                "--berth-y-m" => a.berth_y_m = num(&value()?, &flag)?,
                "--berth-z-m" => a.berth_z_m = num(&value()?, &flag)?,
                other => return Err(format!("{other} is not a flag this tool knows")),
            }
        }
        if a.parent_store.as_os_str().is_empty() || a.ship_store.as_os_str().is_empty() {
            return Err("--parent-store and --ship-store are both required".to_owned());
        }
        if a.owner == 0 {
            return Err("--owner is required: a built realm belongs to somebody".to_owned());
        }
        Ok(a)
    }
}

/// One number, or a refusal naming the flag — monomorphic error path (HR5).
fn num<T: std::str::FromStr>(s: &str, flag: &str) -> Result<T, String> {
    s.parse()
        .map_err(|_| format!("{flag}: {s:?} is not a number"))
}
