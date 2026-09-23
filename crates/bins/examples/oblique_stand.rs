//! ★ AN OBLIQUE STAND — THE OWNER'S OWN FRAMING (2026-09-23, ruling W16 §9 item 2: *"a picture that
//! can judge a ring: an OBLIQUE stand — the ground running from under the nose out to the limb"*).
//!
//! **WHY IT EXISTS.** A 45° NADIR frame cannot hold a rung boundary at all: MEASURED in ruling
//! W16 §5, a 300 km nadir stand spans eye distances of 300 to 360 km while the nearest ring stands
//! at 489 to 513 km. Only an OBLIQUE frame — the nose pitched down toward the horizon — holds
//! several rung boundaries at once, and each one is then a roughly HORIZONTAL band across the
//! frame, which `ring_step`'s `rows` mode reads.
//!
//! **WHAT IT PICKS.** The land node nearest a stated AIM whose eight neighbours are land too (the
//! `land_stand` rule) and whose sun stands at least `VD_STAND_SUN_MIN_DEG` up, then the nearest SEA
//! node to it — the nose is aimed along the bearing toward that sea, so a COASTLINE lies on the
//! frame's own vertical centre line and `coast_line`'s oblique walk can read it.
//!
//! **THE PITCH IS STATED, not drawn.** The horizon of an eye `h` over a body of radius `R` stands
//! `dip = acos(R / (R + h))` below the local horizontal. The nose is pitched `dip + below` below
//! that horizontal, so the horizon stands `below` degrees ABOVE the view axis — and the reference
//! view's half field is 22.5°, so the horizon sits at `0.5 − tan(below) / tan(22.5°) / 2` of the
//! frame down from its top. At `below = 7.5°` that is a third.
//!
//! **AND A SEQUENCE.** With `steps > 1` it also prints ONE `VD_SPAWN_POSES` line holding `steps`
//! accounts (1000, 1001, …) standing `step_km` apart along the same bearing at `seq_alt_m`, each
//! with the same framing. One cluster boot then gives a sequence of frames of a MOVING eye over
//! standing ground: a luminance step that moves with the eye is a ring boundary, one that stays is
//! the landscape.
//!
//! `cargo run --release -p vd-bins --example oblique_stand -- <dx> <dy> <dz> <below_deg> <steps>
//! <step_km> <seq_alt_m> [alt_m ...]`, with `VD_STAND_SUN="x y z"` the star's direction in the
//! planet's frame.

use vd_bins::artifact_worker::{SolveJob, run_solve};
use vd_core::glam::{DMat3, DQuat, DVec3};
use vd_terrain::artifact::DRY_M;
use vd_terrain::home::{HOME_PLANET_SEED, home_planet, home_solve_words};
use vd_terrain::macro_lattice::{MacroLattice, NO_NODE};
use vd_terrain::solve::FACIES_SEA;

fn main() {
    let a: Vec<String> = std::env::args().skip(1).collect();
    let num = |k: usize, d: f64| -> f64 { a.get(k).and_then(|s| s.parse().ok()).unwrap_or(d) };
    let aim = DVec3::new(num(0, 0.617_270), num(1, -0.437_286), num(2, -0.654_033)).normalize();
    let below = num(3, 7.5);
    let steps = num(4, 1.0) as usize;
    let step_km = num(5, 20.0);
    let seq_alt = num(6, 150_000.0);
    let alts: Vec<f64> = if a.len() > 7 {
        a[7..]
            .iter()
            .map(|s| s.parse().expect("an altitude"))
            .collect()
    } else {
        vec![
            20_000.0,
            60_000.0,
            150_000.0,
            400_000.0,
            1_000_000.0,
            2_500_000.0,
        ]
    };
    let sun: Option<DVec3> = std::env::var("VD_STAND_SUN").ok().map(|v| {
        let w: Vec<f64> = v
            .split_whitespace()
            .map(|x| x.parse().expect("a sun component"))
            .collect();
        DVec3::new(w[0], w[1], w[2]).normalize()
    });
    let sun_min = std::env::var("VD_STAND_SUN_MIN_DEG")
        .ok()
        .and_then(|v| v.parse::<f64>().ok())
        .unwrap_or(10.0);
    let body = home_planet();
    let artifact = run_solve(&SolveJob {
        body,
        words: home_solve_words(),
    })
    .expect("the home planet solves");
    let body = home_planet().with_sea_m(artifact.sea());
    let lattice = MacroLattice::of(&body).expect("a lattice");
    let sea = artifact.sea().unwrap_or(0);
    let unit = (1u64 << vd_recipe::bend::DIR_BITS) as f64;
    let dir_of = |node: u32| -> DVec3 {
        let d = lattice.direction(node);
        DVec3::new(
            d[0].raw() as f64 / unit,
            d[1].raw() as f64 / unit,
            d[2].raw() as f64 / unit,
        )
        .normalize()
    };
    let land = |node: u32| {
        let r = artifact.rows[node as usize];
        r.water_m == DRY_M && i32::from(r.z_m) > sea
    };
    let day_min = sun_min.to_radians().sin();
    let mut best: Option<(f64, u32)> = None;
    for node in 0..lattice.node_count() as u32 {
        if !land(node)
            || !lattice
                .neighbours(node)
                .iter()
                .all(|&m| m == NO_NODE || land(m))
        {
            continue;
        }
        if sun.is_some_and(|s| dir_of(node).dot(s) < day_min) {
            continue;
        }
        let dot = dir_of(node).dot(aim);
        if best.is_none_or(|(b, _)| dot > b) {
            best = Some((dot, node));
        }
    }
    let (dot, node) = best.expect("the home planet has land");
    let d = dir_of(node);
    let row = artifact.rows[node as usize];
    // THE NEAREST SEA NODE, so the nose can be aimed at a coastline.
    let mut near_sea: Option<(f64, u32)> = None;
    for other in 0..artifact.node_count() as u32 {
        if artifact.rows[other as usize].receiver_facies >> vd_terrain::artifact::FACIES_SHIFT
            & FACIES_SEA
            == 0
        {
            continue;
        }
        let c = dir_of(other).dot(d).clamp(-1.0, 1.0);
        if near_sea.is_none_or(|(b, _)| c > b) {
            near_sea = Some((c, other));
        }
    }
    let radius = body.radius_m();
    let (sea_km, bearing) = match near_sea {
        Some((c, other)) => {
            let sd = dir_of(other);
            let t = (sd - d * d.dot(sd)).normalize();
            (c.acos() * radius / 1000.0, t)
        }
        None => {
            let seed = if d.z.abs() < 0.9 { DVec3::Z } else { DVec3::X };
            (f64::INFINITY, (seed - d * d.dot(seed)).normalize())
        }
    };
    println!(
        "oblique_stand: land node {node} at {:.2}° off the aim, the ground {} m over the ladder radius (the sea {sea} m), the star {:.1}° up; the nearest sea node {sea_km:.0} km away, and the nose is aimed along it",
        dot.clamp(-1.0, 1.0).acos().to_degrees(),
        row.z_m,
        sun.map_or(f64::NAN, |s| d.dot(s).asin().to_degrees())
    );
    println!(
        "the reference view's half field is 22.5°, so the horizon sits {:.3} of the frame down from its top",
        0.5 - below.to_radians().tan() / 22.5f64.to_radians().tan() / 2.0
    );
    let ground = radius + f64::from(row.z_m);
    // THE POSE at a ground direction `g`, a bearing `t` across it, and an altitude.
    let pose = |g: DVec3, t: DVec3, alt: f64| -> (DVec3, DQuat, f64, DVec3) {
        let eye = ground + alt;
        let dip = (radius / eye).clamp(-1.0, 1.0).acos();
        let beta = dip + below.to_radians();
        let forward = (t * beta.cos() - g * beta.sin()).normalize();
        let right = forward.cross(g).normalize();
        let up = right.cross(forward);
        let orient = DQuat::from_mat3(&DMat3::from_cols(right, up, -forward)).normalize();
        let position = g * eye;
        // ★ THE LOOK-AT TARGET: where the view axis meets the ground's own sphere, so a flight can
        // aim the nose with `vdctl look_at` and never depend on the spawn's facing alone.
        let theta = std::f64::consts::FRAC_PI_2 - beta;
        let s = eye * theta.sin();
        let reach = eye * theta.cos() - (ground * ground - s * s).max(0.0).sqrt();
        (
            position,
            orient,
            dip.to_degrees(),
            position + forward * reach,
        )
    };
    // ★ THE NOSE MAY BE TURNED AWAY FROM THE SEA (`VD_STAND_AWAY=1`): a ring gate wants a frame
    // of ONE material, and a bay across the middle of the frame is a second one. The coastline
    // gate wants the other bearing, so both stands are the same instrument with one word changed.
    let bearing = if std::env::var("VD_STAND_AWAY").is_ok_and(|v| v == "1") {
        -bearing
    } else {
        bearing
    };
    println!(
        "altitude m | the horizon's dip ° | the nose's pitch below the horizontal ° | the pose | the look-at target"
    );
    let mut stands: Vec<String> = Vec::new();
    for (k, alt) in alts.iter().enumerate() {
        let (p, q, _, look) = pose(d, bearing, *alt);
        stands.push(format!(
            "{}=Planet({HOME_PLANET_SEED}):{},{},{}@{},{},{},{}",
            1000 + k,
            p.x,
            p.y,
            p.z,
            q.x,
            q.y,
            q.z,
            q.w
        ));
        println!(
            "the stand's look-at target {k} ({alt:.0} m): {} {} {}",
            look.x, look.y, look.z
        );
    }
    println!(
        "THE STANDS AS ONE BOOT: VD_SPAWN_POSES='{}'",
        stands.join(";")
    );
    for alt in alts {
        let (p, q, dip, look) = pose(d, bearing, alt);
        println!(
            "{alt:.0}|{dip:.3}|{:.3}|1000=Planet({HOME_PLANET_SEED}):{},{},{}@{},{},{},{}|{} {} {}",
            dip + below,
            p.x,
            p.y,
            p.z,
            q.x,
            q.y,
            q.z,
            q.w,
            look.x,
            look.y,
            look.z
        );
    }
    if steps > 1 {
        let mut entries: Vec<String> = Vec::with_capacity(steps);
        for k in 0..steps {
            // The stand moved `k · step_km` along the bearing, as an arc of the body: the ground
            // direction and the bearing are both carried along it, so every frame is framed alike.
            let arc = (k as f64 * step_km * 1000.0) / radius;
            let g = (d * arc.cos() + bearing * arc.sin()).normalize();
            let t = (bearing * arc.cos() - d * arc.sin()).normalize();
            let (p, q, _, look) = pose(g, t, seq_alt);
            entries.push(format!(
                "{}=Planet({HOME_PLANET_SEED}):{},{},{}@{},{},{},{}",
                1000 + k,
                p.x,
                p.y,
                p.z,
                q.x,
                q.y,
                q.z,
                q.w
            ));
            println!(
                "the sequence's look-at target {k}: {} {} {}",
                look.x, look.y, look.z
            );
        }
        println!(
            "\nthe SEQUENCE at {seq_alt:.0} m, {steps} stands {step_km:.0} km apart along the bearing:"
        );
        println!("VD_SPAWN_POSES='{}'", entries.join(";"));
    }
}
