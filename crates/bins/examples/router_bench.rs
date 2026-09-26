//! ★ THE WASHBOARD BENCH — THE TWO CURES OF D8's GRID BIAS, MEASURED BEFORE R2 (ruling B10 item 2,
//! 2026-09-26; `docs/investigation/2026-09-24/terrain_believability_status.md` §8c and the plan's order).
//!
//! **WHY IT EXISTS.** The shoreline bench's panels showed parallel 45° ridges one node apart across
//! the high lake and the second pocket's floor, and chains of lakes on exact diagonals: D8's own bias
//! on smooth ground, carved by the solve's erosion into the 8 km field. The owner ruled that two cures
//! are measured first — a seeded spread of the drain direction (Rho8) and the two-neighbour split (D∞)
//! — and this bench measures them WITHOUT changing the world: `vd_bins::router_variants` rewrites the
//! drains of a routed state, and this example runs the solve's own pass loop around it.
//!
//! **TWO STAGES.**
//! 1. `--cone`: the home moon under Tarboton's cone at two poles (a face's middle, near a face's
//!    corner), each router's upslope-area error inward and outward (Tarboton's Table 2 quantity, in
//!    nodes²), Hyväluoma's rotation score at 15° and 45°, and the fourfold amplitude.
//! 2. `--planet <d8|rho8|dinf>`: the HOME PLANET solved by a line-for-line COPY of
//!    `vd_terrain::solve::solve_full` in which only the drains after each routing, the accumulation
//!    and (for D∞) the cut and the deposit follow the variant. The final routing, the lakes, the
//!    water's tree and the artifact are the solve's own, read over the variant's eroded field.
//!    ★ THE CONTROL: `--planet d8` must give the artifact the pin records
//!    (`crates/bins/tests/home_artifact_pin.rs`, generator 21) bit for bit, or the copy is not the
//!    solve and nothing the variants print can be trusted (`--check-stock` also runs the real solve and
//!    compares every height and every lake level).
//!
//! **WHAT IT PRINTS (stage 2)**, each a `CENSUS <GATE>.<variant> …` line: G-GRID's valleys (the census's
//! own rule — a land node of at least a hundred nodes of drainage area, its long axis over eight
//! receiver steps of the shipped tree) at four and twelve bins; the lakes' long axes (patches of four
//! nodes or more); ★ THE FACING — the ground's own slope direction over the land, its fourfold mode and
//! twelve-bin flatness, a reading no router makes; the lake share, the sea and the ocean share.
//! `--write-artifact <file>` writes the variant's artifact for the shoreline bench
//! (`--artifact-cache <file>`), which paints its stands.
//!
//! ```text
//! cargo run --release -p vd-bins --example router_bench -- --cone
//! cargo run --release -p vd-bins --example router_bench -- --planet rho8 --write-artifact rho8.art
//! ```
//!
//! **Example.** Over the second pocket's floor, D8's furrows run north-east one node apart; the census
//! line FACING reads a strong fourfold mode at phase 45°. After the cure the same floor's slopes face
//! every way, the mode falls toward the isotropic reading, and the shoreline bench's panel of that
//! floor shows hollows where the furrows stood.

use std::time::Instant;

use vd_bins::grid_bias::{
    AZIMUTHS, AxisHistogram, ConeError, cone_error, cone_z, patch_axis, polar_map, trunk_axis, unit,
};
use vd_bins::router_variants::{
    Dinf, Fans, FillOpts, basin_fill_deposit, basin_fill_shaped, dinf, dinf_accumulate, dinf_cut,
    dinf_cut_skipping, dinf_sweep, dominant, facing, paola_braided, paola_fan_offsets, rho8,
    single_links, units,
};
use vd_terrain::BodyDefinition;
use vd_terrain::artifact::Artifact;
use vd_terrain::home::{home_moon, home_planet, home_solve_words};
use vd_terrain::macro_lattice::{MacroLattice, NO_NODE};
use vd_terrain::solve::SolveWords;
use vd_terrain::solve::{
    AGE_GATE_MIN_NODES, DRY, GLACIAL_EPOCH_YR, LAKE_SEA_SOLVES, MacroSolve, RouteReport, Schedule,
    coast_band, envelope_steps, settle_lakes_sea, tan_repose_q16,
};

/// The artifact the pin records at generator 21 (`crates/bins/tests/home_artifact_pin.rs`).
const HOME_ARTIFACT_DIGEST: [u64; 2] = [0x1eb5_3c78_2459_e0df, 0xc2cc_7661_57d2_a0e4];

/// The cone's slope (grid_bias's own).
const CONE_SLOPE: f64 = 0.05;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Variant {
    D8,
    Rho8,
    Dinf,
    /// D8's routing with the BASIN FILL in place of the solve's deposit.
    Fill,
    /// D∞'s routing with the basin fill.
    DinfFill,
    /// D8's routing with Paola's sloped fans and the skip bit (the deposit's nodes stay uncut).
    Fan,
    /// The same without the skip bit: how much of a fan the next sweep grades away.
    FanNoSkip,
    /// D∞'s routing with Paola's sloped fans and the skip bit — ruling B11's candidate world change.
    DinfFan,
}

impl Variant {
    fn name(self) -> &'static str {
        match self {
            Variant::D8 => "d8",
            Variant::Rho8 => "rho8",
            Variant::Dinf => "dinf",
            Variant::Fill => "fill",
            Variant::DinfFill => "dinf+fill",
            Variant::Fan => "fan",
            Variant::FanNoSkip => "fan-noskip",
            Variant::DinfFan => "dinf+fan",
        }
    }
}

// ── STAGE 1: THE CONE ────────────────────────────────────────────────────────────────────────

fn cone_stage() {
    let body = home_moon();
    let lattice = MacroLattice::of(&body).expect("the home moon has a lattice");
    let radius_m = body.radius_m();
    let u = units(&lattice);
    let edge = lattice.edge as i32;
    let face = vd_seed::bend::Face::PosX;
    println!(
        "== STAGE 1: TARBOTON'S CONE on the home moon ({} nodes of {:.0} m) ==",
        lattice.node_count(),
        lattice.node_m()
    );
    for (pole, node) in [
        ("face-middle", lattice.index(face, edge / 2, edge / 2)),
        ("near-corner", lattice.index(face, edge / 6, edge / 6)),
    ] {
        let apex = unit(&lattice, node);
        let reference = unit(&lattice, lattice.neighbours(node)[4]);
        for v in [Variant::D8, Variant::Rho8, Variant::Dinf] {
            let mut state = MacroSolve::new(&body).expect("the moon has a solve state");
            state.z = cone_z(&lattice, apex, radius_m, CONE_SLOPE);
            let _ = state.route();
            match v {
                Variant::D8 | Variant::Fill | Variant::Fan | Variant::FanNoSkip => {
                    state.accumulate();
                }
                Variant::Rho8 => {
                    let seed = state.seed;
                    let _ = rho8(&mut state, &u, radius_m, seed);
                    state.accumulate();
                }
                Variant::Dinf | Variant::DinfFill | Variant::DinfFan => {
                    let d = dinf(&state, &u, radius_m);
                    dinf_accumulate(&mut state, &d);
                }
            }
            let band = |lo: f64, hi: f64| {
                cone_error(
                    &lattice,
                    &state.discharge,
                    apex,
                    radius_m,
                    (lo.to_radians(), hi.to_radians()),
                )
            };
            let inward = band(15.0, 45.0);
            let outward = band(135.0, 165.0);
            let map = polar_map(
                &lattice,
                &state.discharge,
                apex,
                reference,
                (20f64.to_radians(), 70f64.to_radians()),
            );
            let rot = |deg: usize| map.rotation_score(deg * AZIMUTHS / 360);
            let say = |e: ConeError| format!("mse{:.1}/rel{:.2}%", e.mse, e.relative_pct);
            println!(
                "CENSUS CONE.{pole}.{} inward:{} outward:{} rotation15:{:.3} rotation45:{:.3} fourfold:{:.2}% no-band[Tarboton-1997;Hyvaluoma-2017]",
                v.name(),
                say(inward),
                say(outward),
                rot(15),
                rot(45),
                map.fourfold_pct()
            );
        }
    }
}

// ── THE STAGE TRACE: WHERE DOES THE WASHBOARD COME FROM? ──────────────────────────────────────

/// ★ A WINDOW OF THE FACE GRID around an aim, shaded after each stage of the copy, with the window's
/// fine texture as numbers — so the stage whose picture first shows the 45° ridges names the step
/// that makes them (2026-09-26: neither routing cure removed them from the second pocket's floor).
struct Trace {
    lattice: MacroLattice,
    face: vd_seed::bend::Face,
    ci: i32,
    cj: i32,
    half: i32,
    dir: String,
}

impl Trace {
    fn new(lattice: &MacroLattice, aim: vd_core::glam::DVec3, half: i32, dir: &str) -> Trace {
        let aim = aim.normalize();
        let mut best = (f64::MIN, 0u32);
        for node in 0..lattice.node_count() as u32 {
            let d = unit(lattice, node).dot(aim);
            if d > best.0 {
                best = (d, node);
            }
        }
        let (face, ci, cj) = lattice.split(best.1);
        let _ = std::fs::create_dir_all(dir);
        println!(
            "TRACE: node {} (face {:?}, {ci}, {cj}), a window of {} × {} nodes, pictures in {dir}",
            best.1,
            face,
            2 * half + 1,
            2 * half + 1
        );
        Trace {
            lattice: *lattice,
            face,
            ci,
            cj,
            half,
            dir: dir.to_string(),
        }
    }

    /// The node at window offset `(a, b)`, or `None` off the face.
    fn node(&self, a: i32, b: i32) -> Option<u32> {
        let (i, j) = (self.ci + a, self.cj + b);
        let edge = self.lattice.edge as i32;
        ((0..edge).contains(&i) && (0..edge).contains(&j))
            .then(|| self.lattice.index(self.face, i, j))
    }

    /// The shaded picture and the texture numbers of `z` (sixteenths) at one stage.
    fn shot(&self, name: &str, z: &[i32]) {
        const PX: u32 = 4;
        let size = (2 * self.half + 1) as u32;
        let node_m = self.lattice.node_m();
        let mut img = image::RgbImage::new(size * PX, size * PX);
        // The light: from the north-west, 45° up (the panels' own light).
        let (az, alt) = (315f64.to_radians(), 45f64.to_radians());
        let light = [alt.cos() * az.cos(), alt.cos() * az.sin(), alt.sin()];
        let at = |a: i32, b: i32| self.node(a, b).map(|n| f64::from(z[n as usize]) / 16.0);
        let (mut rms, mut count) = (0.0f64, 0usize);
        let (mut c4, mut s4, mut sum) = (0.0f64, 0.0f64, 0.0f64);
        for b in -self.half..=self.half {
            for a in -self.half..=self.half {
                let (Some(c), Some(e), Some(w), Some(n), Some(so)) = (
                    at(a, b),
                    at(a + 1, b),
                    at(a - 1, b),
                    at(a, b + 1),
                    at(a, b - 1),
                ) else {
                    continue;
                };
                let gx = (e - w) / (2.0 * node_m);
                let gy = (n - so) / (2.0 * node_m);
                let len = (gx * gx + gy * gy + 1.0).sqrt();
                let shade = ((-gx * light[0] - gy * light[1] + light[2]) / len).clamp(0.0, 1.0);
                let g = (40.0 + 215.0 * shade) as u8;
                let (x0, y0) = (((a + self.half) as u32) * PX, ((self.half - b) as u32) * PX);
                for dy in 0..PX {
                    for dx in 0..PX {
                        img.put_pixel(x0 + dx, y0 + dy, image::Rgb([g, g, g]));
                    }
                }
                // The fine texture: the height less the mean of the eight neighbours.
                let ring: Vec<f64> = [
                    (-1, -1),
                    (0, -1),
                    (1, -1),
                    (-1, 0),
                    (1, 0),
                    (-1, 1),
                    (0, 1),
                    (1, 1),
                ]
                .iter()
                .filter_map(|&(da, db)| at(a + da, b + db))
                .collect();
                if ring.len() == 8 {
                    let fine = c - ring.iter().sum::<f64>() / 8.0;
                    rms += fine * fine;
                    count += 1;
                }
                let s = gx.hypot(gy);
                if s > 0.0 {
                    let th = gy.atan2(gx);
                    c4 += s * (4.0 * th).cos();
                    s4 += s * (4.0 * th).sin();
                    sum += s;
                }
            }
        }
        let path = format!("{}/{name}.png", self.dir);
        let ok = img.save(&path).is_ok();
        println!(
            "TRACE {name}: fine texture RMS {:.1} m over {count} nodes; the slopes' fourfold {:.1} % at phase {:.1}° (45° = the diagonals); {}",
            (rms / count.max(1) as f64).sqrt(),
            if sum > 0.0 {
                100.0 * c4.hypot(s4) / sum
            } else {
                f64::NAN
            },
            (s4.atan2(c4).to_degrees() / 4.0).rem_euclid(90.0),
            if ok {
                path
            } else {
                format!("{path} NOT WRITTEN")
            }
        );
    }
}

// ── STAGE 2: THE PLANET, THE SOLVE'S OWN LOOP AROUND THE VARIANT ─────────────────────────────

/// What the copy leaves: the final state, the facies, and the routes the lakes' fixed point read.
struct Solved {
    state: MacroSolve,
    facies: Vec<u8>,
    rewrites: Vec<String>,
}

/// ★ THE COPY of `vd_terrain::solve::solve_full` (generator 21), line for line, with the variant's
/// three hooks marked `VARIANT`. Every other line is the solve's own; the control run proves it.
#[allow(clippy::too_many_lines)]
fn solve_variant(
    body: &BodyDefinition,
    words: &SolveWords,
    schedule: Schedule,
    v: Variant,
    trace: Option<&Trace>,
) -> Option<Solved> {
    let lattice = body.macro_lattice()?;
    let land = vd_terrain::land::initial_land(body, &lattice, &words.land());
    if let Some(t) = trace {
        t.shot("00_land", &land.z);
    }
    let mut state = MacroSolve::from_land(body, &land)?;
    let gravity = body.facts().gravity_mm_s2;
    let radius_m = body.radius_m();
    let u = if matches!(v, Variant::D8 | Variant::Fill) {
        // (the fan variants read chords, so they take the units)
        Vec::new()
    } else {
        units(&lattice)
    };
    let mut rewrites = Vec::new();
    let pre_climate = vd_terrain::climate::climate(body, &lattice, words, &land.z, land.sea_view());
    let surface = vd_terrain::craters::Surface {
        crust: &land.crust,
        boundary_m: &land.boundary_m,
        plate: &land.plate,
        plates: &land.plates,
        province: &land.province,
        rain_mm_yr: &pre_climate.rain_mm_yr,
    };
    let craters = vd_terrain::craters::crater_population(body, &lattice, words, &surface);
    vd_terrain::craters::apply_craters(&mut state.z, &lattice, &craters, gravity);
    drop(pre_climate);
    let relief = envelope_steps(body);
    for z in &mut state.z {
        *z = (*z).clamp(-relief, relief);
    }
    for (u, &z) in state.uplift.iter_mut().zip(&state.z) {
        *u = (*u).clamp(-relief - z, relief - z);
    }
    state.z_flood.clone_from(&state.z);
    if let Some(t) = trace {
        t.shot("01_craters", &state.z);
    }
    let gain = schedule.gain();
    let alpha_m = vd_terrain::land::flexural_parameter_m(words.elastic_thickness_m, gravity);
    let mut since_route = schedule.flood_every;
    let mut since_climate = schedule.climate_every;
    let mut since_rebound = 0;
    let mut climate = None;
    let no_ice: Vec<i32> = if words.water_km3 == 0 {
        vec![i32::MAX; state.node_count()]
    } else {
        Vec::new()
    };
    let epoch_step = GLACIAL_EPOCH_YR / u64::from(schedule.passes.max(1));
    let epoch_left = GLACIAL_EPOCH_YR - epoch_step * u64::from(schedule.passes);
    let mut routes: Vec<RouteReport> = Vec::new();
    // The talus relaxations the loop ran (the solve's own `report.talus.len()`).
    let mut talus_done = 0u32;
    // VARIANT: the two links of the last routing, for D∞'s cut between routings.
    let mut links: Option<Dinf> = None;
    // VARIANT (fans): the nodes the deposit ever raised (the skip bit), the last pass's fans over
    // their spills with their volume, and a pass's own span of time.
    let mut deposited = vec![false; state.node_count()];
    let mut above: Option<(Vec<(u32, i32)>, f64)> = None;
    let dt_yr = schedule.age_yr as f64 / f64::from(schedule.passes.max(1));
    for pass in 0..schedule.passes {
        if since_climate >= schedule.climate_every {
            if let Some(sea) = vd_terrain::land::sea_level_connected(
                &lattice,
                &state.z,
                &state.area,
                words.water_km3,
                false,
            ) {
                state.sea_z = sea.level;
                state.ocean = sea.mask.ocean;
            }
            let c = vd_terrain::climate::climate(
                body,
                &lattice,
                words,
                &state.z,
                (words.water_km3 > 0).then_some(state.sea_view()),
            );
            state.rain.clone_from(&c.rain_mm_yr);
            climate = Some(c);
            since_climate = 0;
            since_route = schedule.flood_every;
        }
        since_climate += 1;
        if since_route >= schedule.flood_every {
            routes.push(state.route());
            // VARIANT 1 AND 2: the drains after the routing, and the accumulation over them.
            match v {
                Variant::D8 => state.accumulate(),
                Variant::Fill | Variant::Fan | Variant::FanNoSkip => {
                    state.accumulate();
                    links = Some(single_links(&state));
                }
                Variant::Rho8 => {
                    let seed = state.seed;
                    let r = rho8(&mut state, &u, radius_m, seed);
                    rewrites.push(format!(
                        "pass {pass}: rho8 read {} slope nodes, moved {}, rings {}",
                        r.nodes, r.changed, r.cyclic
                    ));
                    state.accumulate();
                }
                Variant::Dinf | Variant::DinfFill | Variant::DinfFan => {
                    let d = dinf(&state, &u, radius_m);
                    let rings = dominant(&mut state, &d);
                    rewrites.push(format!(
                        "pass {pass}: dinf {} facet nodes, {} split, rings {} (dominant tree rings {rings})",
                        d.facets, d.split, d.cyclic
                    ));
                    dinf_accumulate(&mut state, &d);
                    links = Some(d);
                }
            }
            since_route = 0;
        }
        since_route += 1;
        if let Some(c) = climate.as_ref() {
            if (pass + 1) * schedule.talus_passes / schedule.passes.max(1)
                > pass * schedule.talus_passes / schedule.passes.max(1)
            {
                let tan: Vec<u32> = c.aridity_q8.iter().map(|&a| tan_repose_q16(a)).collect();
                let _ = state.talus(&tan);
                talus_done += 1;
            }
            let line: &[i32] = if words.water_km3 == 0 {
                &no_ice
            } else {
                &c.ela_z
            };
            let _ = state.ice(line, &c.temperature_dk, c.lapse_mk_km, gravity, epoch_step);
        }
        // VARIANT 3: D∞'s cut and deposit over both links; the others cut with the solve's own sweep.
        match (v, links.as_ref()) {
            (Variant::Dinf, Some(d)) => {
                let _ = dinf_sweep(&mut state, d, gain, pass, schedule.passes);
            }
            (Variant::Fan | Variant::FanNoSkip | Variant::DinfFan, Some(d)) => {
                let skip_on = v != Variant::FanNoSkip;
                let _ = dinf_cut_skipping(
                    &mut state,
                    d,
                    gain,
                    pass,
                    schedule.passes,
                    skip_on.then_some(deposited.as_slice()),
                );
                // CHECK-FILL-SURVIVES: the last pass's fans over their spills, after this cut.
                if let Some((list, before)) = above.take() {
                    let after = volume_above(&state, &list);
                    if pass <= 3 || pass + 1 == schedule.passes {
                        rewrites.push(format!(
                            "pass {pass}: basin fill survives the cut: {:.3e} of {:.3e} m³ over the spills ({:.1} %)",
                            after, before, if before > 0.0 { 100.0 * after / before } else { f64::NAN }
                        ));
                    }
                }
                let c = climate
                    .as_ref()
                    .expect("the climate stands from the first pass");
                let talus: Vec<f64> = c
                    .aridity_q8
                    .iter()
                    .map(|&a| f64::from(tan_repose_q16(a)) / 65_536.0)
                    .collect();
                let fans = std::cell::RefCell::new(Fans::default());
                let shape = |st: &MacroSolve, nodes: &[u32], inl: &[(u32, u128, u128)]| {
                    paola_fan_offsets(
                        st,
                        nodes,
                        inl,
                        &u,
                        radius_m,
                        dt_yr,
                        &talus,
                        &mut fans.borrow_mut(),
                    )
                };
                let (f, marks) = basin_fill_shaped(
                    &mut state,
                    d,
                    &FillOpts {
                        shape: Some(&shape),
                        runoff: Some(&c.runoff_mm_yr),
                        cap_at_spill: false,
                    },
                );
                for &r in &marks.raised {
                    deposited[r as usize] = true;
                }
                let over = volume_above(&state, &marks.above);
                above = Some((marks.above, over));
                if pass < 3 || pass + 1 == schedule.passes {
                    let fans = fans.borrow();
                    let mut sl = fans.slopes.clone();
                    sl.sort_by(f64::total_cmp);
                    let q = |p: f64| {
                        sl.get(((sl.len() as f64 - 1.0) * p) as usize)
                            .copied()
                            .unwrap_or(f64::NAN)
                    };
                    let mut tops = fans.top_m.clone();
                    tops.sort_by(f64::total_cmp);
                    rewrites.push(format!(
                        "pass {pass}: basin fill (fans) {} hollows, {} fed, {} full, {} nodes raised, {} steps; {} inlets ({} dry, {} capped at the scree), slope p10 {:.2e} p50 {:.2e} p90 {:.2e}, fan top p50 {:.0} m max {:.0} m; {:.3e} m³ over the spills",
                        f.basins, f.fed, f.full, f.filled, f.total_fill, fans.inlets, fans.dry, fans.capped, q(0.1), q(0.5), q(0.9),
                        tops.get(tops.len() / 2).copied().unwrap_or(f64::NAN), tops.last().copied().unwrap_or(f64::NAN), over
                    ));
                }
            }
            (Variant::Fill | Variant::DinfFill, Some(d)) => {
                let _ = dinf_cut(&mut state, d, gain, pass, schedule.passes);
                let f = basin_fill_deposit(&mut state, d);
                if pass < 3 || pass + 1 == schedule.passes {
                    rewrites.push(format!(
                        "pass {pass}: basin fill {} hollows, {} fed, {} full, {} nodes raised, {} steps",
                        f.basins, f.fed, f.full, f.filled, f.total_fill
                    ));
                }
            }
            _ => {
                let _ = state.sweep(gain, pass, schedule.passes);
            }
        }
        since_rebound += 1;
        if since_rebound >= schedule.isostasy_every {
            let _ = state.rebound(alpha_m);
            since_rebound = 0;
        }
        if let Some(t) = trace
            && [0u32, 1, 4, 9, 19, schedule.passes - 1].contains(&pass)
        {
            t.shot(&format!("02_pass{:03}", pass + 1), &state.z);
        }
    }
    let climate = match climate {
        Some(c) => c,
        None => vd_terrain::climate::climate(
            body,
            &lattice,
            words,
            &state.z,
            (words.water_km3 > 0).then_some(state.sea_view()),
        ),
    };
    let tan: Vec<u32> = climate
        .aridity_q8
        .iter()
        .map(|&a| tan_repose_q16(a))
        .collect();
    // The talus relaxations the loop had no passes to hold: the count the schedule states runs.
    while talus_done < schedule.talus_passes {
        let _ = state.talus(&tan);
        talus_done += 1;
    }
    drop(tan);
    let _ = state.envelope(relief);
    if let Some(t) = trace {
        t.shot("03_envelope", &state.z);
    }
    if let Some(sea) = vd_terrain::land::sea_level_connected(
        &lattice,
        &state.z,
        &state.area,
        words.water_km3,
        false,
    ) {
        state.sea_z = sea.level;
    }
    routes.push(state.route());
    let ice_mask = {
        let line: &[i32] = if words.water_km3 == 0 {
            &no_ice
        } else {
            &climate.ela_z
        };
        let (mask, _, _, _) = state.ice(
            line,
            &climate.temperature_dk,
            climate.lapse_mk_km,
            gravity,
            epoch_left,
        );
        mask
    };
    let mut final_climate = climate;
    let mut budget = None;
    if words.water_km3 > 0 {
        let mut climate =
            vd_terrain::climate::climate(body, &lattice, words, &state.z, Some(state.sea_view()));
        let mut lakes = vd_terrain::lakes::budget(&state, &climate);
        let volume = lakes.report.volume_m3;
        let _ = settle_lakes_sea(
            &mut state,
            words.water_km3,
            LAKE_SEA_SOLVES,
            volume,
            &mut routes,
            &mut |s: &MacroSolve| {
                climate =
                    vd_terrain::climate::climate(body, &lattice, words, &s.z, Some(s.sea_view()));
                lakes = vd_terrain::lakes::budget(s, &climate);
                lakes.report.volume_m3
            },
        );
        state.water_z.clone_from(&lakes.water);
        final_climate = climate;
        budget = Some(lakes);
    }
    let _ = state.hypsometric_integrals(AGE_GATE_MIN_NODES);
    let _ = state.sediment_by_basin();
    let tree = match &budget {
        Some(lakes) => vd_terrain::lakes::water_tree(&state, lakes),
        None => vd_terrain::lakes::dry_tree(&state),
    };
    drop(budget);
    let _ = state.follow_the_water(tree, &final_climate.runoff_mm_yr);
    drop(final_climate);
    let facies = state.facies(&ice_mask, coast_band(words, gravity), words.water_km3 > 0);
    Some(Solved {
        state,
        facies,
        rewrites,
    })
}

/// The volume the listed nodes stand over their spills, in m³ (heights in sixteenths).
fn volume_above(state: &MacroSolve, list: &[(u32, i32)]) -> f64 {
    list.iter()
        .map(|&(v, spill)| {
            let i = v as usize;
            f64::from((state.z[i] - spill).max(0)) / 16.0 * state.area[i] as f64
        })
        .sum()
}

/// The artifact exactly as `vd_bins::artifact_worker::run_solve` makes it from a solved state.
fn artifact_of(body: &BodyDefinition, words: &SolveWords, s: &Solved) -> Artifact {
    let lattice = body.macro_lattice().expect("a lattice");
    let climate =
        vd_terrain::climate::climate(body, &lattice, words, &s.state.z, Some(s.state.sea_view()));
    Artifact::of(&s.state, &s.facies, &climate, words.water_km3 > 0)
}

/// ★ THE READINGS of a solved state: G-GRID's valleys and lakes (the census's own rule), the facing,
/// the lake share, the sea.
fn readings(tag: &str, state: &MacroSolve) {
    let lattice = &state.lattice;
    let n = state.node_count();
    // The drainage area over the shipped tree (one node's area a node), donors first.
    let mut drain: Vec<u64> = state.area.clone();
    for &node in &state.order {
        let r = state.receiver[node as usize];
        if r != NO_NODE {
            drain[r as usize] += drain[node as usize];
        }
    }
    let mut four = AxisHistogram::new(4);
    let mut twelve = AxisHistogram::new(12);
    for node in 0..n as u32 {
        let i = node as usize;
        if state.ocean[i] || drain[i] < state.area[i].saturating_mul(100) {
            continue;
        }
        if let Some(axis) = trunk_axis(lattice, &state.receiver, node, 8) {
            four.add(axis.degrees);
            twelve.add(axis.degrees);
        }
    }
    println!(
        "CENSUS G-GRID-VALLEYS.{tag} {:.3}four/{:.3}twelve/{}trunks ~1.006[W13.3] {}",
        four.flatness(),
        twelve.flatness(),
        four.total(),
        four.line()
    );
    // The lakes: patches of one level, four nodes or more.
    let lake = |i: usize| state.water_z[i] != DRY && !state.ocean[i];
    let mut seen = vec![false; n];
    let (mut l4, mut l12) = (AxisHistogram::new(4), AxisHistogram::new(12));
    let (mut lake_area, mut land_area, mut patches, mut largest) = (0u64, 0u64, 0usize, 0usize);
    for i in 0..n {
        if !state.ocean[i] {
            land_area += state.area[i];
        }
        if lake(i) {
            lake_area += state.area[i];
        }
        if seen[i] || !lake(i) {
            continue;
        }
        let level = state.water_z[i];
        let mut stack = vec![i as u32];
        let mut nodes = Vec::new();
        seen[i] = true;
        while let Some(v) = stack.pop() {
            nodes.push(v);
            for m in lattice.neighbours(v) {
                if m != NO_NODE
                    && !seen[m as usize]
                    && lake(m as usize)
                    && state.water_z[m as usize] == level
                {
                    seen[m as usize] = true;
                    stack.push(m);
                }
            }
        }
        patches += 1;
        largest = largest.max(nodes.len());
        if nodes.len() >= 4
            && let Some(axis) = patch_axis(lattice, &nodes)
        {
            l4.add(axis.degrees);
            l12.add(axis.degrees);
        }
    }
    println!(
        "CENSUS G-GRID-LAKES.{tag} {:.3}four/{:.3}twelve/{}patches-of-4+ no-band[a-reading] {}",
        l4.flatness(),
        l12.flatness(),
        l4.total(),
        l4.line()
    );
    let land = |i: usize| !state.ocean[i];
    let f = facing(lattice, &state.z, &land);
    println!(
        "CENSUS FACING.{tag} fourfold{:.2}%/phase{:.1}deg/twelve-flatness{:.3}/{}nodes no-band[the-ground's-own-slope-direction]",
        f.fourfold_pct,
        f.phase_deg,
        f.flatness(),
        f.nodes
    );
    println!(
        "CENSUS LAKES.{tag} share{:.3}%/{}patches/largest{}nodes; sea {} steps; ocean share {:.3} no-band[ruling-B9:4.828%-at-generator-21]",
        100.0 * lake_area as f64 / land_area.max(1) as f64,
        patches,
        largest,
        state.sea_z,
        state.ocean_share()
    );
}

fn planet_stage(
    v: Variant,
    write: Option<&str>,
    check_stock: bool,
    trace: Option<(vd_core::glam::DVec3, i32, String)>,
) {
    let body = home_planet();
    let words = home_solve_words();
    let schedule = Schedule::standard(words.age_yr);
    println!("== STAGE 2: THE HOME PLANET, variant {} ==", v.name());
    println!(
        "  Paola, Heller & Angevine 1992's braided inlet slope over the load-to-water ratio: {:.3}",
        paola_braided()
    );
    let t = Instant::now();
    let tracer = trace.map(|(aim, half, dir)| {
        let lattice = body.macro_lattice().expect("a lattice");
        Trace::new(&lattice, aim, half, &dir)
    });
    let solved =
        solve_variant(&body, &words, schedule, v, tracer.as_ref()).expect("the home planet solves");
    println!("the copy solved in {:.1} s", t.elapsed().as_secs_f64());
    let (fills, routes): (Vec<&String>, Vec<&String>) = solved
        .rewrites
        .iter()
        .partition(|l| l.contains("basin fill"));
    for line in routes.iter().take(3).chain(fills.iter()) {
        println!("  {line}");
    }
    if routes.len() > 3 {
        println!("  … and {} more routings", routes.len() - 3);
    }
    let artifact = artifact_of(&body, &words, &solved);
    let digest = artifact.digest();
    println!(
        "ARTIFACT {} digest [{:#018x}, {:#018x}]",
        v.name(),
        digest[0],
        digest[1]
    );
    if v == Variant::D8 {
        let ok = digest == HOME_ARTIFACT_DIGEST;
        println!(
            "CENSUS CHECK-COPY-IS-THE-SOLVE.d8 {} =the-pin[home_artifact_pin,generator-21] {}",
            if ok { "equal" } else { "DIFFERS" },
            if ok { "GREEN" } else { "RED" }
        );
    }
    if check_stock {
        let t = Instant::now();
        let (stock, _, _) =
            vd_terrain::solve::solve_full(&body, &words, schedule).expect("the planet solves");
        let dz = stock
            .z
            .iter()
            .zip(&solved.state.z)
            .filter(|(a, b)| a != b)
            .count();
        let dw = stock
            .water_z
            .iter()
            .zip(&solved.state.water_z)
            .filter(|(a, b)| a != b)
            .count();
        println!(
            "CENSUS CHECK-STOCK.{} {}heights/{}levels-differ-from-solve_full ({:.1} s) {}",
            v.name(),
            dz,
            dw,
            t.elapsed().as_secs_f64(),
            if v == Variant::D8 && dz == 0 && dw == 0 {
                "GREEN"
            } else if v == Variant::D8 {
                "RED"
            } else {
                "NO-BAND"
            }
        );
    }
    readings(v.name(), &solved.state);
    if let Some(path) = write {
        match vd_bins::artifact_file::write(path, &artifact) {
            Ok(()) => println!("wrote the {} artifact to {path}", v.name()),
            Err(e) => println!("could not write {path}: {e}"),
        }
    }
}

/// ★ THE READINGS OF AN ARTIFACT FILE (the shoreline bench's `--artifact-cache` format): the facing of
/// the land and — ★ THE WASHBOARD'S OWN READING — the facing of the land's FINE texture, each node's
/// height less the mean of its eight neighbours, so the continents' regional slopes (which the cube's
/// faces orient) leave the reading and the one-node furrows stay; and the lake share. The heights are
/// the rows' whole metres.
fn artifact_readings(path: &str, tag: &str) {
    use vd_terrain::artifact::{DRY_M, FACIES_SHIFT};
    use vd_terrain::solve::FACIES_SEA;
    let Some(a) = vd_bins::artifact_file::read(path) else {
        println!("cannot read the artifact {path}");
        return;
    };
    let body = home_planet();
    let lattice = body.macro_lattice().expect("a lattice");
    let n = lattice.node_count();
    let sea = |i: usize| (a.rows[i].receiver_facies >> FACIES_SHIFT) & FACIES_SEA != 0;
    let z: Vec<i32> = a.rows.iter().map(|r| i32::from(r.z_m)).collect();
    let land = |i: usize| !sea(i);
    let f = facing(&lattice, &z, &land);
    // The fine texture, in decimetres: the height less the mean of the eight neighbours.
    let mut fine = vec![0i32; n];
    let mut has = vec![false; n];
    for i in 0..n {
        let nb = lattice.neighbours(i as u32);
        if nb.contains(&NO_NODE) {
            continue;
        }
        let mean = nb.iter().map(|&m| f64::from(z[m as usize])).sum::<f64>() / 8.0;
        fine[i] = ((f64::from(z[i]) - mean) * 10.0).round() as i32;
        has[i] = true;
    }
    let fine_keep = |i: usize| {
        land(i)
            && has[i]
            && lattice
                .neighbours(i as u32)
                .iter()
                .all(|&m| m != NO_NODE && has[m as usize])
    };
    let g = facing(&lattice, &fine, &fine_keep);
    let (mut lake, mut dry) = (0u64, 0u64);
    for (i, r) in a.rows.iter().enumerate() {
        if sea(i) {
            continue;
        }
        let area = lattice.area_m2(i as u32);
        dry += area;
        if r.water_m != DRY_M {
            lake += area;
        }
    }
    println!(
        "CENSUS FACING-LAND.{tag} fourfold{:.2}%/phase{:.1}deg/twelve-flatness{:.3}/{}nodes no-band[the-ground's-own-slope-direction]",
        f.fourfold_pct,
        f.phase_deg,
        f.flatness(),
        f.nodes
    );
    println!(
        "CENSUS FACING-FINE.{tag} fourfold{:.2}%/phase{:.1}deg/twelve-flatness{:.3}/{}nodes no-band[the-washboard:one-node-furrows;phase-45=diagonals,0=rows]",
        g.fourfold_pct,
        g.phase_deg,
        g.flatness(),
        g.nodes
    );
    println!(
        "CENSUS LAKE-SHARE-ROWS.{tag} {:.3}% no-band[the-rows'-own-water]",
        100.0 * lake as f64 / dry.max(1) as f64
    );
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let mut cone = false;
    let mut planet: Option<Variant> = None;
    let mut write: Option<String> = None;
    let mut check_stock = false;
    let mut readings_done = false;
    let mut trace: Option<(vd_core::glam::DVec3, i32, String)> = None;
    let mut k = 0;
    while k < args.len() {
        match args[k].as_str() {
            "--cone" => cone = true,
            "--planet" => {
                k += 1;
                planet = Some(match args.get(k).map(String::as_str) {
                    Some("d8") => Variant::D8,
                    Some("rho8") => Variant::Rho8,
                    Some("dinf") => Variant::Dinf,
                    Some("fill") => Variant::Fill,
                    Some("dinf+fill") => Variant::DinfFill,
                    Some("fan") => Variant::Fan,
                    Some("fan-noskip") => Variant::FanNoSkip,
                    Some("dinf+fan") => Variant::DinfFan,
                    other => panic!(
                        "--planet d8|rho8|dinf|fill|dinf+fill|fan|fan-noskip|dinf+fan, not {other:?}"
                    ),
                });
            }
            "--write-artifact" => {
                k += 1;
                write = args.get(k).cloned();
            }
            "--check-stock" => check_stock = true,
            "--trace" => {
                let num = |o: usize| -> f64 {
                    args.get(k + o)
                        .and_then(|a| a.parse().ok())
                        .expect("--trace DX DY DZ HALF DIR")
                };
                let aim = vd_core::glam::DVec3::new(num(1), num(2), num(3));
                let half = num(4) as i32;
                let dir = args.get(k + 5).cloned().expect("--trace DX DY DZ HALF DIR");
                trace = Some((aim, half, dir));
                k += 5;
            }
            "--artifact-readings" => {
                let path = args
                    .get(k + 1)
                    .cloned()
                    .expect("--artifact-readings FILE TAG");
                let tag = args
                    .get(k + 2)
                    .cloned()
                    .expect("--artifact-readings FILE TAG");
                artifact_readings(&path, &tag);
                readings_done = true;
                k += 2;
            }
            other => panic!("unknown argument {other}"),
        }
        k += 1;
    }
    if cone {
        cone_stage();
    }
    if let Some(v) = planet {
        planet_stage(v, write.as_deref(), check_stock, trace.clone());
    }
    if !cone && planet.is_none() && !readings_done {
        println!(
            "usage: router_bench [--cone] [--planet d8|rho8|dinf] [--write-artifact FILE] [--check-stock]"
        );
    }
}
