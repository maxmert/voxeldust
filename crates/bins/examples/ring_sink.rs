//! ★ THE RING'S OWN NUMBER: HOW FAR A RUNG'S DRAWN GROUND STANDS UNDER ITS OWN RADIUS, AND OVER
//! HOW WIDE AN ANNULUS (2026-09-23; the owner, from 3 300 km: *"even when shape do not change from
//! far view the moving rings are visible, because for some reason they have different color"*;
//! ruling W16 fault D).
//!
//! **THE MECHANISM, read in the code and not searched for.** A rung's chunk is drawn SUNK along
//! each vertex's radial while the FINER rung still covers it: `ladder_fade.wgsl` subtracts
//! `sink · (1 − risen(d))`, where `risen` ramps from the fade-in band's inner edge `in_lo` to
//! `sink_end`. The ramp is LINEAR, and `sink_end` used to stand far PAST the fade-in edge `in_hi`
//! so that AT the edge the rung still stood one FINER CELL under its own surface — the chord a
//! finer triangle cuts under a coarser crease. The consequence is arithmetic: the residual decays
//! from one finer cell to nothing over `sink_end − in_hi` metres of eye distance, and that is an
//! ANNULUS of ground whose drawn surface stands under its own radius. Lambert reads a tilted
//! ground there, and the air between the eye and the ground is kilometres thicker, so the annulus
//! shades differently from the ring outside it — a ring you can see, which is a seam (SL8).
//!
//! **WHAT IT PRINTS, per rung of the home planet:** the fade-in band, the sink's own depth, where
//! the ramp ends, the residual AT the edge, the width of the annulus over which that residual
//! decays, and that annulus as a SHARE of the distance the rung is drawn from — which is how big
//! the ring is on the screen.
//!
//! `cargo run --release -p vd-bins --example ring_sink`

use vd_client::chunks::sink_m;
use vd_client::ladder_view::AskBound;
use vd_terrain::home::home_planet;

fn main() {
    let body = home_planet();
    let rungs = body.ladder().rungs;
    // ★ THE BANDS THE PICTURE REALLY DRAWS (ruling W17): the ladder's own bound, having read the
    // body AND the pyramid the artifact holds, so a rung whose handover step floors its switch is
    // read where it is really drawn.
    let lattice = body
        .macro_lattice()
        .expect("the home planet has a macro lattice");
    let mut levels = 0u32;
    while lattice.coarser(levels + 1).is_some() {
        levels += 1;
    }
    let bound = AskBound::unbounded().for_body(&body, rungs, levels);
    let fade_bands = |rung: u8, rungs: u8| bound.fade_bands(rung, rungs);
    let sink_end_m = |body: &vd_terrain::BodyDefinition, rung: u8, rungs: u8| {
        bound.sink_end_m(body, rung, rungs)
    };
    println!(
        "ring_sink: the home planet, {rungs} rungs. A rung is drawn from about its own switch \
         distance, so an annulus is read against that."
    );
    println!(
        "rung | cell m | fade-in lo m | fade-in hi m | sink m | ramp end m | residual at the edge m | the annulus past the edge m | as a share of the edge's own distance"
    );
    for rung in 1..rungs {
        let (fade_in, _) = fade_bands(rung, rungs);
        let sink = sink_m(&body, rung);
        let end = sink_end_m(&body, rung, rungs);
        // The shader's own arithmetic: `sink · (1 − risen(d))`, `risen` linear from `in_lo` to
        // `end`. At the fade-in edge the finer rung's last fragment is discarded.
        let risen = |d: f64| ((d - fade_in[0]) / (end - fade_in[0])).clamp(0.0, 1.0);
        let residual = sink * (1.0 - risen(fade_in[1]));
        let annulus = (end - fade_in[1]).max(0.0);
        println!(
            "{rung:>4} | {:>6} | {:>12.0} | {:>12.0} | {sink:>6.0} | {end:>10.0} | {residual:>22.0} | {annulus:>27.0} | {:>37.4}",
            vd_seed::ladder::cell_m(rung),
            fade_in[0],
            fade_in[1],
            annulus / fade_in[1].max(1.0),
        );
    }
}
