//! ★ THE GRID'S SCRATCHES, MEASURED (ruling B2 step 3; gate G-GRID, 2026-09-22). Three readings of
//! the router on THE world's own home moon, printed as numbers a change can move:
//!
//! 1. **TARBOTON'S CONE** (1997, Water Resour. Res. 33(2), 309–319). The ground rises with the
//!    great-circle distance from one node, so every direction is equal and the upslope area of a
//!    point is an exact number. The error is stated in node widths squared, near the pole (the
//!    water converges) and near the antipode (it diverges). ★ A grid-aligned plane passes while the
//!    defect is whole, so no plane is measured here.
//! 2. **HYVÄLUOMA'S ROTATION** (2017, IJGIS 31(11), 2272–2285). The same cone, turned about its own
//!    pole and correlated with itself. A perfect router scores one at every angle.
//! 3. **THE FLAT** — the step the cure touches. A bowl in the side of that cone is filled by the
//!    priority flood into a flat disc with one spill, which is exactly a lake. Two readings of how
//!    the water crosses it: the DETOUR (the path down the receivers over the beeline to the exit; a
//!    straight run reads one) and the long-axis histogram of the trunks against the grid's own four
//!    directions (a spike at 45° names the tie-break).
//!
//! `cargo run --release -p vd-bins --example grid_bias`

use vd_bins::grid_bias::{
    AZIMUTHS, AxisHistogram, ConeError, axis_gap, cone_error, cone_z, pair_axis, polar_map,
    trunk_axis, unit,
};
use vd_terrain::home::home_moon;
use vd_terrain::macro_lattice::{MacroLattice, NO_NODE};
use vd_terrain::solve::MacroSolve;

/// The cone's slope: one part in a hundred, so the home moon's 353 km radius carries a field of a
/// few kilometres and no node's step is lost to the sixteenth of a metre the solve stores.
const SLOPE: f64 = 0.01;

/// The bowl in the cone's side: its angular radius and its depth in metres. The depth is deep
/// enough that the flood must fill it, and the radius wide enough that the flat holds thousands of
/// nodes — which is the size of the hollows the owner saw as ponds.
const BOWL_RADIANS: f64 = 0.45;
const BOWL_DEPTH_M: f64 = 4_000.0;

/// The trunk's length in receiver steps: one step is one of eight directions by construction, and
/// eight steps is a line on the ground whose angle is free.
const TRUNK_STEPS: usize = 8;

/// ★ THE OTHER ARM OF THE MEASUREMENT: the flat receivers as they were assigned BEFORE the cure of
/// 2026-09-22 — a breadth-first HOP COUNT out from the flat's draining shore, ties to the smaller
/// node index. It is written here, in the instrument, so one run shows both arms and the cure's
/// worth is a measurement that could have failed rather than a claim.
///
/// The flat's own nodes are read off the routed tree: a node whose receiver stands at its own
/// flooded height was assigned by the flat's rule; every other node is the flat's shore or an
/// outlet, exactly as the routing's own distance field seeds it.
fn hop_receivers(lattice: &MacroLattice, state: &MacroSolve) -> Vec<u32> {
    let n = state.node_count();
    let mut dist = vec![u32::MAX; n];
    for (i, d) in dist.iter_mut().enumerate() {
        let r = state.receiver[i];
        if r == NO_NODE || state.z_flood[r as usize] < state.z_flood[i] {
            *d = 0;
        }
    }
    let mut frontier: Vec<u32> = (0..n as u32).filter(|&i| dist[i as usize] == 0).collect();
    let mut level = 0u32;
    while !frontier.is_empty() {
        let mut next = Vec::new();
        for &node in &frontier {
            for m in lattice.neighbours(node) {
                if m == NO_NODE
                    || dist[m as usize] != u32::MAX
                    || state.z_flood[m as usize] != state.z_flood[node as usize]
                {
                    continue;
                }
                dist[m as usize] = level + 1;
                next.push(m);
            }
        }
        frontier = next;
        level += 1;
    }
    let mut receiver = state.receiver.clone();
    for i in 0..n {
        if dist[i] == 0 || dist[i] == u32::MAX {
            continue;
        }
        let mut best = NO_NODE;
        for m in lattice.neighbours(i as u32) {
            if m == NO_NODE
                || dist[m as usize] == u32::MAX
                || state.z_flood[m as usize] != state.z_flood[i]
            {
                continue;
            }
            if best == NO_NODE
                || dist[m as usize] < dist[best as usize]
                || (dist[m as usize] == dist[best as usize] && m < best)
            {
                best = m;
            }
        }
        receiver[i] = best;
    }
    receiver
}

fn main() {
    let body = home_moon();
    let lattice = MacroLattice::of(&body).expect("the home moon has a macro lattice");
    let radius_m = body.radius_m();
    println!(
        "the home moon: {} nodes of {:.0} m, radius {:.0} km",
        lattice.node_count(),
        lattice.node_m(),
        radius_m / 1_000.0
    );

    // ★ 1 AND 2 — THE CONE, at two poles: one at the middle of a face, where the grid's directions
    // stand square to the cone's own frame, and one near a face corner, where they do not. The
    // defect is the GRID's, so it must not follow the pole.
    let edge = lattice.edge as i32;
    let face = vd_seed::bend::Face::PosX;
    for (name, node) in [
        ("the face's middle", lattice.index(face, edge / 2, edge / 2)),
        (
            "near a face corner",
            lattice.index(face, edge / 6, edge / 6),
        ),
    ] {
        let apex = unit(&lattice, node);
        let mut state = MacroSolve::new(&body).expect("the moon has a solve state");
        state.z = cone_z(&lattice, apex, radius_m, SLOPE);
        let report = state.route();
        state.accumulate();
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
        let say = |e: ConeError| {
            format!(
                "MSE {:.1} nodes², bias {:+.2}, {:.2} % of the true area, {} nodes",
                e.mse, e.bias, e.relative_pct, e.nodes
            )
        };
        println!(
            "\nTARBOTON'S CONE, pole at {name}: flats {} of {} nodes, undrained {}\n  inward  (15°–45° from the pole):      {}\n  outward (15°–45° from the antipode):  {}",
            report.flat,
            lattice.node_count(),
            report.undrained,
            say(inward),
            say(outward)
        );
        let reference = unit(&lattice, lattice.neighbours(node)[4]);
        let map = polar_map(
            &lattice,
            &state.discharge,
            apex,
            reference,
            (20f64.to_radians(), 70f64.to_radians()),
        );
        let scores: Vec<String> = [0, 15, 30, 45, 60, 75, 90]
            .iter()
            .map(|&deg| {
                let slots = deg * AZIMUTHS / 360;
                format!("{deg}°:{:.3}", map.rotation_score(slots))
            })
            .collect();
        println!(
            "HYVÄLUOMA'S ROTATION, pole at {name}: {}\n  the fourfold amplitude {:.2} % of the ring's own mean (a perfect router: 0 %)",
            scores.join(" "),
            map.fourfold_pct()
        );
    }

    // ★ 3 — THE FLAT. A bowl in the cone's side, filled by the flood into a disc with one spill.
    // The pole stands on ANOTHER face, so the bowl sits at the middle of its own face and its whole
    // rim is one face's cells — the frame every long axis is stated in.
    let apex = unit(
        &lattice,
        lattice.index(vd_seed::bend::Face::PosZ, edge / 2, edge / 2),
    );
    let bowl = unit(&lattice, lattice.index(face, edge / 2, edge / 2));
    let mut state = MacroSolve::new(&body).expect("the moon has a solve state");
    let mut z = cone_z(&lattice, apex, radius_m, SLOPE);
    for (node, z) in z.iter_mut().enumerate() {
        let phi = unit(&lattice, node as u32)
            .dot(bowl)
            .clamp(-1.0, 1.0)
            .acos();
        if phi < BOWL_RADIANS {
            let share = 1.0 - (phi / BOWL_RADIANS) * (phi / BOWL_RADIANS);
            *z -= (BOWL_DEPTH_M * share * 16.0).round() as i32;
        }
    }
    state.z = z;
    let report = state.route();
    let flat: Vec<u32> = (0..lattice.node_count() as u32)
        .filter(|&node| state.pit[node as usize] > 0)
        .collect();
    println!(
        "\nTHE FLAT: the bowl filled to a disc of {} nodes ({} flat receivers, {} undrained)",
        flat.len(),
        report.flat,
        report.undrained
    );

    // ★ BOTH ARMS, ON THE ONE FILLED DISC: the hop count that drew the scratches and the summed
    // chord that replaced it.
    let arms = [
        ("the HOP COUNT (before)", hop_receivers(&lattice, &state)),
        ("the SUMMED CHORD (after)", state.receiver.clone()),
    ];
    for (name, receiver) in &arms {
        // The detour: the path down the receivers out of the flat, over the straight line to the
        // same exit. A flat whose water goes the way water goes reads one.
        let (mut detour, mut counted, mut worst) = (0.0f64, 0usize, 0.0f64);
        for &node in &flat {
            let mut at = node;
            let mut path = 0.0f64;
            let mut steps = 0usize;
            while state.pit[at as usize] > 0 && steps < 4 * lattice.edge as usize {
                let next = receiver[at as usize];
                if next == NO_NODE {
                    break;
                }
                path += f64::from(lattice.chord_m(at, next));
                at = next;
                steps += 1;
            }
            let beeline = radius_m
                * unit(&lattice, node)
                    .dot(unit(&lattice, at))
                    .clamp(-1.0, 1.0)
                    .acos();
            if beeline < lattice.node_m() {
                continue;
            }
            let ratio = path / beeline;
            detour += ratio;
            counted += 1;
            if ratio > worst {
                worst = ratio;
            }
        }
        // ★ THE BEARING. ★ ONE DISC WITH ONE SPILL IS PEAKED BY CONSTRUCTION: every drop on it must
        // run toward the same exit, so a FLAT histogram here would be a defect, not a gate. The gate
        // on one flat is whether the trunk points where the straight line to its own exit points.
        // The long-axis histogram's flatness is the PLANET's reading, over thousands of hollows
        // whose exits lie in every direction — `lake_census` prints that one.
        let (mut gap, mut gaps, mut worst_gap) = (0.0f64, 0usize, 0.0f64);
        let mut four = AxisHistogram::new(4);
        for &node in &flat {
            let Some(axis) = trunk_axis(&lattice, receiver, node, TRUNK_STEPS) else {
                continue;
            };
            four.add(axis.degrees);
            let mut at = node;
            for _ in 0..TRUNK_STEPS {
                at = receiver[at as usize];
            }
            if let Some(line) = pair_axis(&lattice, node, at) {
                let d = axis_gap(axis.degrees, line.degrees);
                gap += d;
                gaps += 1;
                if d > worst_gap {
                    worst_gap = d;
                }
            }
        }
        println!(
            "  {name}\n    THE DETOUR out of the flat: mean {:.4}, worst {:.3}, over {counted} nodes (a straight run reads 1.000)\n    THE BEARING against the straight line to the exit: mean {:.2}°, worst {:.1}°, over {gaps} trunks (a straight run reads 0.00°)\n    the trunks' own bearings (PEAKED by construction, one disc and one spill): {} | {:.3}",
            detour / counted.max(1) as f64,
            worst,
            gap / gaps.max(1) as f64,
            worst_gap,
            four.line(),
            four.flatness()
        );
    }
}
