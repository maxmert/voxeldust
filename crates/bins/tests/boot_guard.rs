//! D-6 Slice D-delta — the orchestrator's HR1 ephemeral-store boot guard (always compiled; no feature).
//!
//! `VD_STORE_PATH` is REQUIRED and a temp-dir path is REJECTED LOUD unless the explicit dev/test opt-in
//! `VD_STORE_EPHEMERAL_OK` is set — the production-safety net against the old `/tmp/{shard_id}` data-loss
//! bug. This locks BOTH arms of that guard (reject without the opt-in, accept with it), so a refactor that
//! silently inverts the condition fails the gate.

use std::process::{Command, Stdio};
use std::time::{Duration, Instant};

use vd_bins::{
    Cluster, ClusterAddrs, DEV, admin_get_body, common_env, orchestrator_env, spawn_node,
};
use vd_io_prod::trust::ClusterTrust;

/// A PLANET of THE world, by seed.
///
/// It used to be hardcoded to `7`, which was a planet of the hand-placed walk forest. There is one world
/// now and it is generated, so its planets carry hash seeds and `7` names nothing — a shard booted on it
/// gets an EMPTY neighbourhood and fails the fence for the wrong reason ("0 ambient roots"), which is what
/// this test then mistook for the lineage fence firing.
///
/// ★ RE-DERIVED 2026-08-21 (the gate-pass arc). The seed was the LITERAL `0`, and on 2026-08-20 the
/// no-`VD_UNIVERSE_SEED` default became [`vd_physics::worldgen::HOME_SEED`] — which is what the
/// `vd-shard` this test spawns actually boots, because neither `shard_env` nor `common_env` names the
/// key. So the test picked a planet out of ONE world and asserted a fence in ANOTHER: the same
/// two-worlds-in-one-launch class the paragraph above was written about, re-opened by a default
/// moving underneath it. `DEV.universe_seed` IS that default, so the two ends cannot disagree again.
fn a_system_of_the_world() -> u64 {
    // ★ A STAR SYSTEM, NOT A PLANET (owner ruling 2026-08-30). These fences are about a realm that
    // HAS A PARENT, and a star system has one — the galaxy. A planet no longer serves as the subject:
    // a realm BELOW a star system cannot place itself from the seed, so its parent must name it, and
    // a planet launched by hand now refuses EARLIER with that message. The fences below would then
    // never be reached, and the tests would pass or fail for a reason none of them is about.
    //
    // Read from the star-system LAYER, which is what the fences read too — not from the full forest,
    // which builds every planet and moon in the galaxy to answer a question about a system.
    let world = vd_physics::worldgen::system_layer_view(
        vd_bins::DEV.universe_seed,
        &vd_physics::worldgen::UniverseConfig::world(vd_bins::DEV.move_speed, vd_bins::DEV.tick_dt),
    );
    world
        .regions()
        .iter()
        .find_map(|r| match r.realm {
            vd_core::pose::RealmId::System(seed) => Some(seed),
            _ => None,
        })
        .expect("the generated world holds star systems")
}

fn addrs() -> ClusterAddrs {
    ClusterAddrs::reserve()
}

#[test]
fn a_temp_store_without_ephemeral_ok_refuses_to_boot() {
    let addrs = addrs();
    let trust_dir = std::env::temp_dir().join(format!("vd-bootreject-{}", std::process::id()));
    ClusterTrust::generate("vd-bootreject")
        .expect("trust")
        .write_der_dir(&trust_dir)
        .expect("trust dir");
    // A temp-dir store path (the guard's reject target) — with the dev escape STRIPPED.
    let temp_store =
        std::env::temp_dir().join(format!("vd-bootreject-{}.redb", std::process::id()));
    let mut node_env = orchestrator_env(
        &addrs,
        &DEV,
        &temp_store.display().to_string(),
        vd_bins::ClusterShape::Single,
    );
    node_env.retain(|(k, _)| *k != "VD_STORE_EPHEMERAL_OK");
    let common = common_env(&trust_dir.display().to_string(), &DEV);

    // A REAL reserved VD_BIND so the mesh binds and execution actually REACHES the store guard (a garbage
    // bind would non-zero-exit for the WRONG reason — guarded against by the stderr-reason assertions).
    let mut cmd = Command::new(env!("CARGO_BIN_EXE_vd-orchestrator"));
    for (k, v) in common.iter().chain(node_env.iter()) {
        cmd.env(k, v);
    }
    cmd.stdout(Stdio::null()).stderr(Stdio::piped());
    let output = cmd
        .spawn()
        .expect("spawn orchestrator")
        .wait_with_output()
        .expect("wait orchestrator");

    assert!(
        !output.status.success(),
        "the orchestrator MUST exit non-zero on a temp store with no VD_STORE_EPHEMERAL_OK"
    );
    let err = String::from_utf8_lossy(&output.stderr);
    assert!(
        err.contains("Refusing to boot"),
        "the ephemeral guard must reject with its message; stderr: {err}"
    );
    // Rigor: prove it was the STORE GUARD, not an earlier trust/bind/config failure.
    assert!(
        !err.contains("VD_TRUST_DIR") && !err.contains("VD_BIND"),
        "rejected for the wrong reason (not the store guard); stderr: {err}"
    );
    let _ = std::fs::remove_dir_all(&trust_dir);
    let _ = std::fs::remove_file(&temp_store);
}

#[test]
fn a_temp_store_with_ephemeral_ok_boots_past_the_guard() {
    // FIRST statement: hold the process tier for the whole body, so it outlives the cluster reap
    // that frees the ports. See `vd_bins::cluster_tier`.
    let _tier = vd_bins::cluster_tier();
    // The ACCEPT arm: `orchestrator_env` keeps VD_STORE_EPHEMERAL_OK=1, so a temp store boots and serves
    // admin (the guard's accept + the "under a TEMP dir … dev/test" warn ran).
    let addrs = addrs();
    let admin_addr = addrs.admin;
    let trust_dir = std::env::temp_dir().join(format!("vd-bootaccept-{}", std::process::id()));
    ClusterTrust::generate("vd-bootaccept")
        .expect("trust")
        .write_der_dir(&trust_dir)
        .expect("trust dir");
    let store = std::env::temp_dir().join(format!("vd-bootaccept-{}.redb", std::process::id()));
    let _ = std::fs::remove_file(&store);
    let common = common_env(&trust_dir.display().to_string(), &DEV);
    let mut cluster = Cluster::new();
    cluster.push(
        "vd-orchestrator",
        spawn_node(
            env!("CARGO_BIN_EXE_vd-orchestrator"),
            &common,
            &orchestrator_env(
                &addrs,
                &DEV,
                &store.display().to_string(),
                vd_bins::ClusterShape::Single,
            ),
        )
        .expect("spawn orch"),
    );
    let _guard = cluster;
    let started = Instant::now();
    let mut served = false;
    while started.elapsed() < Duration::from_secs(10) {
        if admin_get_body(admin_addr, "/admin/snapshot", Some(Duration::from_secs(2))).is_some() {
            served = true;
            break;
        }
        std::thread::sleep(Duration::from_millis(100));
    }
    assert!(
        served,
        "with VD_STORE_EPHEMERAL_OK a temp-store orchestrator must boot past the guard + serve admin"
    );
    let _ = std::fs::remove_dir_all(&trust_dir);
    let _ = std::fs::remove_file(&store);
}

// ---- THE LINEAGE FENCE ---------------------------------------------------------------------------
//
// The owner's rule: we cannot boot a realm without its parents all the way to the root. A shard whose
// world gives it a parent while its declared lineage claims none MISROUTES every occupant that leaves it —
// leaving reads the shard's own lineage to decide who to hand the occupant to, so it hands them to the
// ambient root instead of to the star system twenty metres away. Nothing crashes; the player is simply
// somewhere else. Refusing to start beats that, and the guard existed for a year with no caller at all.

/// The env a lone `vd-shard` needs to reach the lineage fence, with `extra` appended (later keys win in
/// the process env, which is how the realm KIND and lineage are overridden per case).
fn shard_boot_env(
    addrs: &ClusterAddrs,
    extra: &[(&'static str, String)],
) -> Vec<(&'static str, String)> {
    // Each lineage case gets its own work dir, so one refused shard's outbox never meets the next.
    let work_dir = vd_bins::fresh_work_dir("vd-bootguard");
    let mut env = vd_bins::shard_env(addrs, &DEV, vd_bins::ClusterShape::Single, &work_dir);
    env.extend(extra.iter().cloned());
    env
}

/// A ROOT-SHAPED lineage for `realm` — one level, no parent. This is exactly what every launcher used to
/// hand every shard, and what a shard now only ever sees if somebody declares it explicitly.
fn root_shaped_coord(kind: vd_core::realm_path::RealmKindTag, seed: u64) -> String {
    vd_core::realm_path::RealmPath::from_levels(vec![vd_core::realm_path::RealmLevel::new(
        kind, seed,
    )])
    .to_env_string()
}

#[test]
fn a_shard_claiming_to_be_a_root_while_the_world_gives_it_a_parent_refuses_to_boot() {
    let _tier = vd_bins::cluster_tier();
    let addrs = addrs();
    let trust_dir = std::env::temp_dir().join(format!("vd-lineage-{}", std::process::id()));
    ClusterTrust::generate("vd-lineage")
        .expect("trust")
        .write_der_dir(&trust_dir)
        .expect("trust dir");
    let common = common_env(&trust_dir.display().to_string(), &DEV);
    // A STAR-SYSTEM shard, told its lineage names no parent. The seed world says a star system sits
    // inside the galaxy, so the two disagree and the fence must fire.
    let node_env = shard_boot_env(
        &addrs,
        &[
            ("VD_REALM_KIND", "system".to_owned()),
            ("VD_REALM_SEED", a_system_of_the_world().to_string()),
            (
                "VD_OWN_COORD",
                root_shaped_coord(
                    vd_core::realm_path::RealmKindTag::System,
                    a_system_of_the_world(),
                ),
            ),
        ],
    );
    let mut cmd = Command::new(env!("CARGO_BIN_EXE_vd-shard"));
    for (k, v) in common.iter().chain(node_env.iter()) {
        cmd.env(k, v);
    }
    cmd.stdout(Stdio::null()).stderr(Stdio::piped());
    let output = cmd
        .spawn()
        .expect("spawn shard")
        .wait_with_output()
        .expect("wait shard");
    assert!(
        !output.status.success(),
        "a shard whose lineage omits its parent MUST exit non-zero"
    );
    let err = String::from_utf8_lossy(&output.stderr);
    assert!(
        err.contains("but booted with a root lineage"),
        "the lineage fence must reject with its own message; stderr: {err}"
    );
    assert!(
        err.contains("refusing to boot"),
        "the lineage fence must refuse the boot out loud; stderr: {err}"
    );
    // Rigor: prove it was the LINEAGE fence and not an earlier trust/bind/config failure, and that it
    // named BOTH sides — the realm booted and the parent the world gives it.
    assert!(
        err.contains(&format!("System({})", a_system_of_the_world())),
        "the refusal must name the realm booted; stderr: {err}"
    );
    assert!(
        err.contains("whose parent is Galaxy("),
        "the refusal must name the parent the lineage omitted; stderr: {err}"
    );
    let _ = std::fs::remove_dir_all(&trust_dir);
}

#[test]
fn the_same_shard_with_no_declared_lineage_derives_one_and_boots_past_the_fence() {
    // THE ACCEPT TWIN, and the reason the fence is not simply a blanket refusal: with no `VD_OWN_COORD`
    // the shard DERIVES its lineage from the very forest it just planted, walking its own region's parent
    // pointers to the root. A derived lineage cannot disagree with the forest it came from, so every
    // shipped launcher — none of which emits a lineage — boots. Without this the fence would refuse the
    // whole fleet, since the seed forest gives even a star system a galaxy above it.
    let _tier = vd_bins::cluster_tier();
    let addrs = addrs();
    let probe = addrs.shard_probe;
    let trust_dir = std::env::temp_dir().join(format!("vd-lineage-ok-{}", std::process::id()));
    ClusterTrust::generate("vd-lineage-ok")
        .expect("trust")
        .write_der_dir(&trust_dir)
        .expect("trust dir");
    let common = common_env(&trust_dir.display().to_string(), &DEV);
    let node_env = shard_boot_env(
        &addrs,
        &[
            ("VD_REALM_KIND", "system".to_owned()),
            ("VD_REALM_SEED", a_system_of_the_world().to_string()),
        ],
    );
    let mut cluster = Cluster::new();
    cluster.push(
        "vd-shard",
        spawn_node(env!("CARGO_BIN_EXE_vd-shard"), &common, &node_env).expect("spawn shard"),
    );
    let _guard = cluster;
    let started = Instant::now();
    let mut live = false;
    // ★ RAISED FROM 15 s (2026-08-30). The property under test is that a shard with a DERIVED
    // lineage boots PAST the fence — not how fast it does so. Fifteen seconds was written when a
    // galaxy held three star systems; it now holds a quarter of a million, and a debug-build shard
    // folds the star-system layer before it can answer anything. The same reason raised the dev
    // cluster's own bring-up deadline.
    while started.elapsed() < Duration::from_secs(180) {
        if vd_bins::http_get_status(probe, "/healthz", Some(Duration::from_secs(1))) == Some(200) {
            live = true;
            break;
        }
        std::thread::sleep(Duration::from_millis(100));
    }
    assert!(
        live,
        "a shard with a DERIVED lineage must boot past the fence and serve /healthz"
    );
    let _ = std::fs::remove_dir_all(&trust_dir);
}

#[test]
fn an_aoi_armed_shard_without_the_handoff_hold_budget_refuses_to_boot() {
    // Step 5 slice B — the hold boot fence: THE world's interest bands are armed, so a shard booted
    // with a zero/absent hand-off hold budget would run half the hand-off ledger (a source falling
    // silent about a departing occupant while its parent still counts on it). It must refuse, out
    // loud, naming the missing key — never boot disarmed. (`common_env` carries the derived value for
    // every launcher; this test strips it to prove the fence is live.)
    let _tier = vd_bins::cluster_tier();
    let addrs = addrs();
    let trust_dir = std::env::temp_dir().join(format!("vd-hold-fence-{}", std::process::id()));
    ClusterTrust::generate("vd-hold-fence")
        .expect("trust")
        .write_der_dir(&trust_dir)
        .expect("trust dir");
    let common: Vec<(&'static str, String)> = common_env(&trust_dir.display().to_string(), &DEV)
        .into_iter()
        .filter(|(k, _)| *k != "VD_HANDOFF_HOLD_TICKS")
        .collect();
    let node_env = shard_boot_env(
        &addrs,
        &[
            ("VD_REALM_KIND", "system".to_owned()),
            ("VD_REALM_SEED", a_system_of_the_world().to_string()),
        ],
    );
    let mut cmd = Command::new(env!("CARGO_BIN_EXE_vd-shard"));
    for (k, v) in common.iter().chain(node_env.iter()) {
        cmd.env(k, v);
    }
    cmd.stdout(Stdio::null()).stderr(Stdio::piped());
    let output = cmd
        .spawn()
        .expect("spawn shard")
        .wait_with_output()
        .expect("wait shard");
    assert!(
        !output.status.success(),
        "an AoI-armed shard with no hold budget MUST exit non-zero"
    );
    let err = String::from_utf8_lossy(&output.stderr);
    assert!(
        err.contains("VD_HANDOFF_HOLD_TICKS"),
        "the refusal must name the missing key; stderr: {err}"
    );
    assert!(
        err.contains("refusing to boot"),
        "the hold fence must refuse the boot out loud; stderr: {err}"
    );
    let _ = std::fs::remove_dir_all(&trust_dir);
}

/// ★ THE NEW CONTRACT (owner ruling 2026-08-30): A REALM BELOW A STAR SYSTEM IS NAMED BY ITS PARENT.
///
/// A shard whose realm the star-system layer does NOT name cannot place itself. A planet's identifier
/// is a one-way hash of its system's, and a player-built station or area is not in the seed at all —
/// the generator emits neither — so no search will ever produce one.
///
/// It does not have to. A spawn demand carries a `RealmCoord`, which names every ancestor by kind and
/// seed, and the launcher hands it on as `VD_OWN_COORD`. This test is the case where nobody did:
/// a deep shard started by hand, with no lineage.
///
/// It must refuse, and the refusal must say WHAT is missing and HOW to supply it — not "the forest
/// has 0 ambient roots", which is true and useless.
#[test]
fn a_realm_below_a_star_system_launched_with_no_lineage_refuses_and_names_the_cure() {
    let _tier = vd_bins::cluster_tier();
    let addrs = addrs();
    let trust_dir = std::env::temp_dir().join(format!("vd-deep-nolineage-{}", std::process::id()));
    ClusterTrust::generate("vd-deep-nolineage")
        .expect("trust")
        .write_der_dir(&trust_dir)
        .expect("trust dir");
    let common = common_env(&trust_dir.display().to_string(), &DEV);
    // A PLANET of THE world, and deliberately NO `VD_OWN_COORD` — nothing tells it who contains it.
    let planet_seed = a_planet_of_the_world();
    let node_env = shard_boot_env(
        &addrs,
        &[
            ("VD_REALM_KIND", "planet".to_owned()),
            ("VD_REALM_SEED", planet_seed.to_string()),
        ],
    );
    let mut cmd = Command::new(env!("CARGO_BIN_EXE_vd-shard"));
    for (k, v) in common.iter().chain(node_env.iter()) {
        cmd.env(k, v);
    }
    cmd.stdout(Stdio::null()).stderr(Stdio::piped());
    let output = cmd
        .spawn()
        .expect("spawn shard")
        .wait_with_output()
        .expect("wait shard");
    assert!(
        !output.status.success(),
        "a deep shard with no lineage MUST exit non-zero"
    );
    let err = String::from_utf8_lossy(&output.stderr);
    assert!(
        err.contains(&format!("Planet({planet_seed})")),
        "the refusal must name the realm it could not place; stderr: {err}"
    );
    assert!(
        err.contains("its parent must name it"),
        "the refusal must say WHOSE job it is to name the realm; stderr: {err}"
    );
    assert!(
        err.contains("VD_OWN_COORD"),
        "the refusal must name the cure; stderr: {err}"
    );
    let _ = std::fs::remove_dir_all(&trust_dir);
}

/// The planet the test above cannot place. Read from the full world deliberately: this is the ONE
/// place a test needs a realm the star-system layer does not name.
fn a_planet_of_the_world() -> u64 {
    vd_bins::boot_world(
        vd_bins::DEV.universe_seed,
        vd_bins::DEV.move_speed,
        vd_bins::DEV.tick_dt,
    )
    .regions()
    .iter()
    .find_map(|r| match r.realm {
        vd_core::pose::RealmId::Planet(seed) => Some(seed),
        _ => None,
    })
    .expect("the generated world holds planets")
}
