//! M8-0 — THE RUNG DISAGREEMENT, IN PIXELS (slice 8, the ladder; `00_proposed_voxel_foundation.md`
//! slice 8's measurement: "runs BEFORE this slice builds the ladder").
//!
//! For every rung `L` of the home planet: the vertical disagreement between the height at rung `L`
//! and at rung `L + 1` along the same direction, over a grid of directions on every face, as the
//! maximum and the 99th percentile in metres, in CELLS of rung `L + 1`, and in PIXELS at the switch
//! distance — the distance at which one cell of rung `L + 1` subtends one pixel at the reference view
//! (45° over 720 rows, `vd_core::geometry::REFERENCE_VIEW_FOV_Y_RAD`), the drawable floor. The
//! ladder's coarsening law (ruling V9) promises the far view is the near hill with the fine octaves
//! left out; this is the number that says whether a rung switch can be SEEN. The pass line is the
//! slice's own: p99 ≤ 1 pixel, max ≤ 2 pixels.
//!
//! ```text
//! cargo run --release -p vd-bins --example rung_disagreement
//! ```

use std::process::ExitCode;

use vd_bins::DEV;
use vd_seed::bend::{Face, direction};
use vd_terrain::height::height_m;

/// Directions per face edge in the sample grid.
const GRID: i32 = 48;
/// The reference view's rows and vertical field of view (the drawable floor's own).
const VIEW_ROWS: f64 = 720.0;

fn main() -> ExitCode {
    let Some(body) = vd_bins::home_body(DEV.universe_seed) else {
        println!("rung_disagreement: REFUSED — the home system holds no planet the recipe accepts");
        return ExitCode::FAILURE;
    };
    let fov_y = vd_core::geometry::REFERENCE_VIEW_FOV_Y_RAD;
    // One pixel's angle at the reference view.
    let pixel_rad = fov_y / VIEW_ROWS;
    println!(
        "rung_disagreement: home planet {} — {} rungs, {} octaves, {} directions per face",
        body.seed(),
        body.ladder().rungs,
        body.octaves_at(0).len(),
        GRID * GRID
    );
    println!(
        "  rung L -> L+1   cell(L+1)   switch distance   max (m)   p99 (m)   max (cells)   p99 (cells)   max (px)   p99 (px)   bound (m)"
    );
    let faces = [
        Face::PosX,
        Face::NegX,
        Face::PosY,
        Face::NegY,
        Face::PosZ,
        Face::NegZ,
    ];
    let mut worst_px = 0.0f64;
    let mut worst_p99_px = 0.0f64;
    let mut rung = 0u8;
    while rung + 1 < body.ladder().rungs {
        let cell = f64::from(vd_seed::ladder::cell_m(rung + 1));
        // The switch distance: where one cell of the coarser rung is one pixel high.
        let switch_m = cell / pixel_rad;
        let mut diffs: Vec<f64> = Vec::with_capacity(6 * (GRID * GRID) as usize);
        for face in faces {
            let mut i = 0;
            while i < GRID {
                let mut j = 0;
                while j < GRID {
                    let a = (2.0 * (f64::from(i) + 0.5) / f64::from(GRID)) - 1.0;
                    let b = (2.0 * (f64::from(j) + 0.5) / f64::from(GRID)) - 1.0;
                    let d = direction(face, a, b);
                    let dg = [d[0], d[1], d[2]];
                    let fine = height_m(&body, dg, rung);
                    let coarse = height_m(&body, dg, rung + 1);
                    diffs.push((fine - coarse).abs());
                    j += 1;
                }
                i += 1;
            }
        }
        diffs.sort_by(|x, y| x.partial_cmp(y).expect("finite"));
        let max = diffs[diffs.len() - 1];
        let p99 = diffs[(diffs.len() as f64 * 0.99) as usize];
        // A vertical step of `m` metres at the switch distance subtends `m / switch_m` radians.
        let px = |m: f64| (m / switch_m) / pixel_rad;
        // ★ THE BOUND THE LADDER ACTUALLY USES (slice 8a stage 4): the body's own STEP BOUND, which
        // carries the cap-rock bench's Lipschitz word and its fade's own step. Before the bench it
        // was a difference of two value bounds and the two agreed; they no longer do, and the
        // column that matters is the one ruling T7's rules 2 and 3 read.
        let bound = body.step_bound_m(rung);
        println!(
            "  {rung:>2} -> {:>2}      {cell:>8.0} m   {:>10.0} m   {max:>7.2}   {p99:>7.2}   {:>11.2}   {:>11.2}   {:>8.2}   {:>8.2}   {bound:>9.2}",
            rung + 1,
            switch_m,
            max / cell,
            p99 / cell,
            px(max),
            px(p99)
        );
        worst_px = worst_px.max(px(max));
        worst_p99_px = worst_p99_px.max(px(p99));
        rung += 1;
    }
    println!(
        "rung_disagreement: worst max {worst_px:.2} px, worst p99 {worst_p99_px:.2} px at the switch distance (pass: max ≤ 2, p99 ≤ 1)"
    );
    ExitCode::SUCCESS
}
