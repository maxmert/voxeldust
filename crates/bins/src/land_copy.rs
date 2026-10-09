//! ★ THE ONE COPY OF THE INITIAL LAND FOR THE BENCHES (rulings B58 item 2, B59–B61). A line-for-line copy of
//! `vd_terrain::land::initial_land` (generator 26) in which a bench may take each of the land's laws out or switch a
//! diagnostic in — IN THE COPY ONLY. Nothing in the game reads this module (the owner, 2026-10-01: no test seams in the
//! game; ruling B38: one bench, the binding rules unbroken). THE CONTROL: with [`LandLaws::default`] the copy rebuilds
//! the game's initial land bit for bit (the example `coast_bench` checks every row against
//! `vd_terrain::land::initial_land` on the home planet), or nothing it prints can be trusted.
//!
//! THE LAWS A BENCH MAY TAKE OUT (each a field of [`LandLaws`]; generator 26 holds them all, generator 25 none —
//! [`LandLaws::GENERATOR_25`]):
//!
//! 1. ★ THE CELL'S MEAN (`cell`): each node the mean of its plate terms over `K × K` sub-cells; `None` reads the node's
//!    centre alone, as generator 25 did.
//! 2. ★ THE MARGINS BY KIND (`margins`; `vd_terrain::land::margin_width_m`).
//! 3. ★ THE SOURCED SPREAD (`spread`): out, each plate keeps its rank in the draw on generator 25's unsourced range
//!    [`OLD_CRUST_SCATTER`] — a diagnostic of the old draw.
//! 4. ★ THE FIRST RELIEF ([`Relief`]): the game's `Spectrum` (the older landscape, `vd_terrain::land::older_landscape_m`),
//!    `None`, or the two forms ruling B60 did not take — (a) `Airy`, a NAMED crust-thickness scatter inside each plate
//!    ([`CRUST_FIELD_SPREAD`], [`CRUST_FIELD_LENGTH_M`]) through the land's own Airy isostasy; (b) `Flex`, the same
//!    field's root buoyancy through the solve's own flexural rebound.
//! 5. ★ THE CRUST SHARE READ CONTINUOUSLY (`share`): out, the room's profiles take a sub-cell's side from the half-share
//!    test, as generator 25 did (the belts' copy [`crate::belts_copy`] reads the same switch).
//!
//! The diagnostics of ruling B58 (`blend_m`, `scatter`) stay — measurements, never candidates (ruling B38 item 4).
//!
//! **Example.** `phase2-margins` in the belts' bench solves the home planet with every law of generator 26 but the
//! margins: its Andean coast falls in one node again, and the bench prints how much of the coastline steps.

use vd_recipe::Gi;
use vd_recipe::bend::{DIR_BITS, direction, inv_n_of};
use vd_recipe::height::{AMP_BITS, Octave, relief};
use vd_recipe::noise::NOISE_BITS;
use vd_seed::rng::child_seed;
use vd_terrain::BodyDefinition;
use vd_terrain::Gf;
use vd_terrain::land::{
    AFFINITY_OCTAVES, AFFINITY_WAVE_M, BELT_HALF_WIDTHS, Boundary, CELL_SUBCELLS,
    CONTINENTAL_CRUST_DENSITY_KGM3, CONTINENTAL_SHARE, EARTH_CONTINENT_SPREAD,
    EARTH_CONTINENTAL_CRUST_M, EARTH_RELIEF_CAP_M, Kind, Land, LandWords, MANTLE_DENSITY_KGM3,
    Plate, area_quantile, boundary_of, crust_share, draw_plates, dry_field, flexural_parameter_m,
    freeboard_factor, isostatic_parts_m, load_ocean, margin_width_m, older_landscape_m,
    plate_count, province_of, relief_cap_m, sea_level_connected, smoothstep, uplift_m,
};
use vd_terrain::macro_lattice::MacroLattice;
use vd_terrain::solve::{MacroSolve, Z_STEPS_PER_M};

/// ★ THE FIRST RELIEF'S FORM (rulings B59 item 1, B60; see the module).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum Relief {
    /// Generator 25's: no relief inside a plate.
    None,
    /// (a) The crust-thickness scatter through the land's own Airy isostasy.
    Airy,
    /// (b) The crust-thickness scatter's root buoyancy through the solve's own flexural rebound.
    Flex,
    /// (c) The older landscape's relief by the continents' published spectrum: THE GAME'S (ruling B60).
    #[default]
    Spectrum,
}

/// ★ WHAT THE COPY HOLDS (see the module); the default is the game's land.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct LandLaws {
    /// The cell's mean over K × K sub-cells; `None` the node's centre alone.
    pub cell: Option<u32>,
    /// The margins by kind.
    pub margins: bool,
    /// The sourced per-plate crust spread.
    pub spread: bool,
    /// The first relief's form.
    pub relief: Relief,
    /// ★ THE CRUST'S SHARE READ CONTINUOUSLY (ruling B61 item 1): each sub-cell's room profile blends the continental and
    /// the oceanic profile by its crust share, and the belts' wedge takes in rock by it ([`crate::belts_copy`]).
    pub share: bool,
    /// ★ A DIAGNOSTIC (ruling B58): the two plates' characters blended across the bisector by the quintic smoothstep
    /// of the signed distance over `±w` whole metres, half and half at the bisector — no law behind `w`.
    pub blend_m: Option<u32>,
    /// ★ A DIAGNOSTIC (ruling B58): every node's first height moved by a whole number of sixteenths in `[−a, a]`
    /// drawn from its own hash — no law behind it.
    pub scatter: Option<u32>,
}

impl LandLaws {
    /// Generator 25's land: the node's centre, no margins, the unsourced spread, no relief, the half-share test.
    pub const GENERATOR_25: LandLaws = LandLaws {
        cell: None,
        margins: false,
        spread: false,
        relief: Relief::None,
        share: false,
        blend_m: None,
        scatter: None,
    };
}

impl Default for LandLaws {
    /// THE GAME'S LAND (generator 26): every law in, no diagnostic.
    fn default() -> LandLaws {
        LandLaws {
            cell: Some(CELL_SUBCELLS),
            margins: true,
            spread: true,
            relief: Relief::Spectrum,
            share: true,
            blend_m: None,
            scatter: None,
        }
    }
}

/// Generator 25's per-plate crust scatter, which named no source (ruling B59 item 3): the diagnostic `spread` out.
pub const OLD_CRUST_SCATTER: (f64, f64) = (0.7, 1.4);
/// ★ THE CRUST-THICKNESS SCATTER INSIDE A CONTINENT (form (a)): σ 3.7 km on 39.5 km (the research's CRUST1.0 reading
/// over 9 008 stable, ice-free 1° cells), as a share of the plate's own crust.
pub const CRUST_FIELD_SPREAD: f64 = 3.7 / 39.5;
/// ★ THE SCATTER'S CORRELATION LENGTH, metres (form (a)): the exponential fit to the same reading.
pub const CRUST_FIELD_LENGTH_M: f64 = 365_000.0;
/// The crust field's shortest wave, metres: two nodes of the home planet's lattice.
pub const FIELD_SHORTEST_WAVE_M: f64 = 16_384.0;

/// The diagnostic scatter's salt (the bench's own stream; the ASCII of "TIES").
const SALT_TIES: u64 = 0x5449_4553;
/// The crust field's salt (the bench's own stream; the ASCII of "CRUSTF").
const SALT_CRUST_FIELD: u64 = 0x4352_5553_5446;

/// The land stream's salt, copied from `vd_terrain::body::salt::LAND` (crate-private there).
const SALT_LAND: u64 = 0x5e_ed_0a;

/// The copy of `vd_terrain::body::rounded` (crate-private): half up.
fn rounded(v: Gf) -> i64 {
    (v + Gf::HALF).floor().to_i64_floor()
}

/// The copy of `vd_terrain::body::frequency_of` (crate-private).
fn frequency_of(frequency: Gf) -> (Gi, Gi) {
    let whole = frequency.floor();
    (
        Gi::new(whole.to_i64_floor()),
        Gi::new(rounded((frequency - whole) * Gf::from_i64(1 << NOISE_BITS))),
    )
}

/// The copy of `vd_terrain::land::affinity_octaves` (private).
fn affinity_octaves(body: &BodyDefinition) -> [Octave; AFFINITY_OCTAVES] {
    let radius = Gf::from_f64(body.ladder().radius_m());
    let mut wave = Gf::from_i64(i64::from(AFFINITY_WAVE_M));
    let mut amp = 1i64 << AMP_BITS;
    let mut out = [Octave::smooth(0, Gi::ZERO, Gi::ZERO, Gi::ZERO); AFFINITY_OCTAVES];
    for (k, slot) in out.iter_mut().enumerate() {
        let (fi, ff) = frequency_of(radius / wave);
        *slot = Octave::smooth(
            child_seed(body.seed(), SALT_LAND, 1 + k as u64),
            fi,
            ff,
            Gi::new(amp),
        );
        wave = wave / Gf::TWO;
        amp >>= 1;
    }
    out
}

/// The copy of `vd_terrain::land::affinity_noise` (private).
fn affinity_noise(octaves: &[Octave], dir: [Gi; 3]) -> Gf {
    Gf::from_i64(relief(octaves, dir).raw()) / Gf::from_i64(1 << NOISE_BITS)
}

/// The copy of `vd_terrain::land::dir_of` (crate-private).
fn dir_of(dir: [Gi; 3]) -> [Gf; 3] {
    let one = Gf::from_i64(1 << DIR_BITS);
    [
        Gf::from_i64(dir[0].raw()) / one,
        Gf::from_i64(dir[1].raw()) / one,
        Gf::from_i64(dir[2].raw()) / one,
    ]
}

/// ★ A FIELD OF OCTAVES over the lattice, zero mean and unit variance by area (forms (a) and (b)): each octave
/// `(wave_m, amplitude)` is the recipe's smooth noise at that wave, seeded on `salt`.
#[must_use]
pub fn octave_field(
    body: &BodyDefinition,
    lattice: &MacroLattice,
    salt: u64,
    octaves: &[(f64, f64)],
) -> Vec<f64> {
    let radius = Gf::from_f64(body.ladder().radius_m());
    let built: Vec<Octave> = octaves
        .iter()
        .enumerate()
        .map(|(k, &(wave, amp))| {
            let (fi, ff) = frequency_of(radius / Gf::from_f64(wave));
            Octave::smooth(
                child_seed(body.seed(), salt, 1 + k as u64),
                fi,
                ff,
                Gi::new((amp * f64::from(1u32 << AMP_BITS)).round() as i64),
            )
        })
        .collect();
    let n = lattice.node_count();
    let mut field: Vec<f64> = (0..n as u32)
        .map(|node| affinity_noise(&built, lattice.direction(node)).to_f64())
        .collect();
    let (mut sum, mut sum2, mut total) = (0.0f64, 0.0f64, 0.0f64);
    for (node, &f) in field.iter().enumerate() {
        let a = lattice.area_m2(node as u32) as f64;
        sum += f * a;
        sum2 += f * f * a;
        total += a;
    }
    let mean = sum / total;
    let sd = (sum2 / total - mean * mean).sqrt();
    for f in &mut field {
        *f = (*f - mean) / sd;
    }
    field
}

/// The octaves from `longest_m` down to [`FIELD_SHORTEST_WAVE_M`], each half the wave of the one before and `ratio` of
/// its amplitude, the first of amplitude one.
fn octave_ladder(longest_m: f64, ratio: f64) -> Vec<(f64, f64)> {
    let mut out = Vec::new();
    let (mut wave, mut amp) = (longest_m, 1.0f64);
    while wave >= FIELD_SHORTEST_WAVE_M {
        out.push((wave, amp));
        wave /= 2.0;
        amp *= ratio;
    }
    out
}

/// ★ THE CRUST FIELD (relief forms (a) and (b)): an exponential covariance of length `a` has a flat spectrum over waves
/// longer than `2πa` and falls as `k⁻³` in two dimensions over shorter ones — an octave's amplitude `2^(−½)` of the one
/// before under the corner wave, half of the next one over it.
#[must_use]
pub fn crust_field(body: &BodyDefinition, lattice: &MacroLattice) -> Vec<f64> {
    let corner = std::f64::consts::TAU * CRUST_FIELD_LENGTH_M;
    let mut octaves = vec![(2.0 * corner, 0.5)];
    octaves.extend(octave_ladder(corner, 0.5f64.sqrt()));
    octave_field(body, lattice, SALT_CRUST_FIELD, &octaves)
}

/// The copy of `vd_terrain::land::initial_land` with the laws a bench may take out (see the module).
#[must_use]
pub fn initial_land_copy(
    body: &BodyDefinition,
    lattice: &MacroLattice,
    words: &LandWords,
    laws: LandLaws,
) -> Land {
    let mut plates = draw_plates(
        body.seed(),
        plate_count(body.ladder().radius_m(), words.elastic_thickness_m),
    );
    // DIAGNOSTIC (`spread` out): each plate's rank in the draw, read on generator 25's unsourced range.
    if !laws.spread {
        let half = Gf::from_f64(EARTH_CONTINENT_SPREAD) * Gf::from_i64(3).sqrt();
        let (lo, hi) = OLD_CRUST_SCATTER;
        for p in &mut plates {
            let u = (p.crust_scatter - (Gf::ONE - half)) / (Gf::TWO * half);
            p.crust_scatter = Gf::from_f64(lo) + u * Gf::from_f64(hi - lo);
        }
    }
    initial_land_from_copy(body, lattice, words, plates, laws)
}

/// The boundaries of node `node`'s sub-cells on a lattice `K` times finer (the recipe's own bend, so the sub-cells
/// tile the node's cell exactly), row by row; `None` the node's own centre boundary alone.
fn sub_boundaries(
    lattice: &MacroLattice,
    plates: &[Plate],
    node: u32,
    centre: Boundary,
    cell: Option<u32>,
    out: &mut Vec<Boundary>,
) {
    out.clear();
    let Some(k) = cell else {
        out.push(centre);
        return;
    };
    let (face, i, j) = lattice.split(node);
    let inv = inv_n_of(lattice.edge * k);
    let k = k as i32;
    for b in 0..k {
        for a in 0..k {
            let d = direction(i32::from(face.index()), i * k + a, j * k + b, inv);
            out.push(boundary_of(plates, dir_of(d)));
        }
    }
}

/// The copy of `vd_terrain::land::initial_land_from` with the laws a bench may take out (see the module). Every line
/// not marked `LAW OUT` or `DIAGNOSTIC` is the game's own.
#[allow(clippy::too_many_lines)]
#[must_use]
pub fn initial_land_from_copy(
    body: &BodyDefinition,
    lattice: &MacroLattice,
    words: &LandWords,
    plates: Vec<Plate>,
    laws: LandLaws,
) -> Land {
    let cell = laws.cell;
    let n = lattice.node_count();
    let gravity = body.facts().gravity_mm_s2;
    let cap_m = relief_cap_m(body);
    let octaves = affinity_octaves(body);
    let radius = Gf::from_f64(body.ladder().radius_m());
    let width =
        Gf::from_f64(flexural_parameter_m(words.elastic_thickness_m, gravity) * BELT_HALF_WIDTHS);
    let relief = Gf::from_f64(body.relief_m());
    // 1. The plates and the affinity at every node's centre.
    let mut boundaries = Vec::with_capacity(n);
    let mut affinity = Vec::with_capacity(n);
    let mut noise = Vec::with_capacity(n);
    let mut area = Vec::with_capacity(n);
    for node in 0..n as u32 {
        let gi = lattice.direction(node);
        let d = dir_of(gi);
        let b = boundary_of(&plates, d);
        let a = affinity_noise(&octaves, gi);
        affinity.push(a + plates[usize::from(b.plate)].affinity);
        noise.push(a);
        boundaries.push(b);
        area.push(lattice.area_m2(node));
    }
    // 2. The crust threshold: the continental share of the AREA, over the centres' field.
    let threshold = area_quantile(&affinity, &area, 1.0 - CONTINENTAL_SHARE);
    // The first estimate of the freeboard factor, which the Andean margin's width reads.
    let first = {
        let mut part_o = Vec::with_capacity(n);
        let mut part_c = Vec::with_capacity(n);
        let mut crust = Vec::with_capacity(n);
        for k in 0..n {
            let share = crust_share(affinity[k], threshold);
            let own = &plates[usize::from(boundaries[k].plate)];
            let (o, c) = isostatic_parts_m(share, cap_m, own.crust_scatter);
            part_o.push(o);
            part_c.push(c);
            crust.push((share * Gf::from_f64(255.0) + Gf::HALF).to_i64_floor() as u8);
        }
        let flat = vec![Gf::ZERO; n];
        freeboard_factor(
            lattice,
            &part_o,
            &part_c,
            &crust,
            &area,
            words.water_km3,
            &flat,
        )
    };
    // DIAGNOSTIC (form (a) or (b)): the crust field.
    let field: Vec<f64> = match laws.relief {
        Relief::Airy | Relief::Flex => crust_field(body, lattice),
        Relief::None | Relief::Spectrum => Vec::new(),
    };
    // One sub-cell's crust share and scatter: its own plate's, or across a margin, or the diagnostic blend.
    let terms = |k: usize, sb: &Boundary| -> (Gf, Gf) {
        let own = &plates[usize::from(sb.plate)];
        let own_share = crust_share(noise[k] + own.affinity, threshold);
        let Some(o) = sb.other else {
            return (own_share, own.crust_scatter);
        };
        let other = &plates[usize::from(o)];
        // DIAGNOSTIC (B58): the blend across the bisector.
        if let Some(w) = laws.blend_m {
            let t =
                smoothstep(Gf::HALF + radius * sb.sine / (Gf::TWO * Gf::from_i64(i64::from(w))));
            let offset = own.affinity * t + other.affinity * (Gf::ONE - t);
            return (
                crust_share(noise[k] + offset, threshold),
                own.crust_scatter * t + other.crust_scatter * (Gf::ONE - t),
            );
        }
        // LAW OUT (`margins`): the plate's own terms up to the bisector.
        if !laws.margins {
            return (own_share, own.crust_scatter);
        }
        let other_share = crust_share(noise[k] + other.affinity, threshold);
        let column = |share: Gf, scatter: Gf| -> Gf {
            let (o, c) = isostatic_parts_m(share, cap_m, scatter);
            o + first * c
        };
        if column(own_share, own.crust_scatter) <= column(other_share, other.crust_scatter) {
            return (own_share, own.crust_scatter);
        }
        let w = margin_width_m(sb.kind, own, cap_m, first, words.elastic_thickness_m);
        let t = (radius * sb.sine / w).clamp(Gf::ZERO, Gf::ONE);
        (
            other_share + (own_share - other_share) * t,
            other.crust_scatter + (own.crust_scatter - other.crust_scatter) * t,
        )
    };
    // 3. Isostasy, dry — each node the mean of its sub-cells' parts and shares.
    let count = Gf::from_i64(i64::from(cell.map_or(1, |k| k * k)));
    let mut subs = Vec::new();
    let mut part_o = Vec::with_capacity(n);
    let mut part_c = Vec::with_capacity(n);
    let mut crust = Vec::with_capacity(n);
    for k in 0..n {
        sub_boundaries(lattice, &plates, k as u32, boundaries[k], cell, &mut subs);
        let (mut o_sum, mut c_sum, mut s_sum) = (Gf::ZERO, Gf::ZERO, Gf::ZERO);
        // DIAGNOSTIC (form (a)): the node's crust thickness scattered by the field.
        let thick = match laws.relief {
            Relief::Airy => Gf::ONE + Gf::from_f64(CRUST_FIELD_SPREAD * field[k]),
            _ => Gf::ONE,
        };
        for sb in &subs {
            let (share, scatter) = terms(k, sb);
            let (o, c) = isostatic_parts_m(share, cap_m, scatter * thick);
            o_sum += o;
            c_sum += c;
            s_sum += share;
        }
        part_o.push(o_sum / count);
        part_c.push(c_sum / count);
        crust.push((s_sum / count * Gf::from_f64(255.0) + Gf::HALF).to_i64_floor() as u8);
    }
    let steps = Gf::from_i64(i64::from(Z_STEPS_PER_M));
    // The older landscape's relief by each node's crust share (LAW OUT: `relief` other than `Spectrum`, none).
    let older: Vec<Gf> = match laws.relief {
        Relief::Spectrum => older_landscape_m(body, lattice)
            .into_iter()
            .zip(&crust)
            .map(|(r, &c)| Gf::from_i64(i64::from(c)) / Gf::from_i64(255) * r)
            .collect(),
        _ => vec![Gf::ZERO; n],
    };
    let freeboard_factor = freeboard_factor(
        lattice,
        &part_o,
        &part_c,
        &crust,
        &area,
        words.water_km3,
        &older,
    );
    let (dry, tallest) = dry_field(&part_o, &part_c, &area, freeboard_factor);
    let room = (relief - tallest).greater(Gf::ZERO);
    // 4. Orogeny: the belts under the room — each node the mean of its sub-cells' profiles.
    let mut uplift = Vec::with_capacity(n);
    for k in 0..n {
        sub_boundaries(lattice, &plates, k as u32, boundaries[k], cell, &mut subs);
        let mut sum = Gf::ZERO;
        for sb in &subs {
            let own = &plates[usize::from(sb.plate)];
            // LAW OUT (`share`): the sub-cell's side from the half-share test — its own crust's word where the cell
            // is read, the node's where the centre is.
            let continental = match cell {
                Some(_) => {
                    let share = crust_share(noise[k] + own.affinity, threshold);
                    (share * Gf::from_f64(255.0) + Gf::HALF).to_i64_floor() >= 128
                }
                None => crust[k] >= 128,
            };
            let (other_continental, other_lower) = match sb.other {
                Some(o) => (plates[usize::from(o)].affinity >= Gf::ZERO, o < sb.plate),
                None => (false, false),
            };
            let amplitude = room
                * (sb.convergence.abs() / Gf::TWO).clamp(Gf::ZERO, Gf::ONE)
                * (Gf::ONE - own.age * Gf::HALF);
            let distance = radius * sb.sine;
            // The Andean margin's trench wall (LAW OUT with `margins`).
            let andean = laws.margins
                & (sb.kind == Kind::Convergent)
                & (if laws.share {
                    own.affinity >= Gf::ZERO
                } else {
                    continental
                })
                & sb.other
                    .is_some_and(|o| plates[usize::from(o)].affinity < Gf::ZERO);
            let (continental_side, oceanic_side) = if andean {
                let wall = uplift_m(
                    Kind::Convergent,
                    false,
                    true,
                    true,
                    distance,
                    amplitude,
                    width,
                );
                (wall, wall)
            } else {
                (
                    uplift_m(
                        sb.kind,
                        true,
                        other_continental,
                        !other_lower,
                        distance,
                        amplitude,
                        width,
                    ),
                    uplift_m(
                        sb.kind,
                        false,
                        other_continental,
                        !other_lower,
                        distance,
                        amplitude,
                        width,
                    ),
                )
            };
            sum += if laws.share {
                let s = terms(k, sb).0;
                continental_side * s + oceanic_side * (Gf::ONE - s)
            } else if continental {
                continental_side
            } else {
                oceanic_side
            };
        }
        uplift.push((sum / count * steps).to_i64_floor() as i32);
    }
    let mut z_dry: Vec<i32> = dry
        .iter()
        .zip(&older)
        .map(|(&h, &r)| (h * steps).to_i64_floor() as i32 + (r * steps).to_i64_floor() as i32)
        .collect();
    // DIAGNOSTIC (form (b)): the root through flexure added before the sea is solved.
    if laws.relief == Relief::Flex {
        let lift = flex_lift(
            body,
            &z_dry,
            &crust,
            &field,
            &plates,
            &boundaries,
            cap_m,
            freeboard_factor,
            flexural_parameter_m(words.elastic_thickness_m, gravity),
        );
        for (z, l) in z_dry.iter_mut().zip(lift) {
            *z += l;
        }
    }
    // 5. The sea, with the load, over the connected ocean only.
    let sea = sea_level_connected(lattice, &z_dry, &area, words.water_km3, true);
    let (sea_z, ocean, pockets, pocket_nodes) = match sea {
        Some(s) => (
            Some(s.level),
            s.mask.ocean,
            s.mask.pockets,
            s.mask.pocket_nodes,
        ),
        None => (None, vec![false; n], 0, 0),
    };
    let mut z: Vec<i32> = match sea_z {
        Some(level) => load_ocean(&z_dry, &ocean, level),
        None => z_dry,
    };
    // DIAGNOSTIC (B58): no platform an exact flat.
    if let Some(a) = laws.scatter {
        let span = u64::from(2 * a + 1);
        for (k, h) in z.iter_mut().enumerate() {
            let draw = child_seed(body.seed(), SALT_TIES, k as u64) % span;
            *h += draw as i32 - a as i32;
        }
    }
    let relief_steps = (relief * steps).to_i64_floor() as i32;
    for (u, &h) in uplift.iter_mut().zip(&z) {
        *u = (*u).clamp(-relief_steps - h, relief_steps - h);
    }
    let province: Vec<u8> = crust
        .iter()
        .zip(&uplift)
        .zip(&z)
        .zip(&boundaries)
        .map(|(((&c, &u), &h), b)| {
            let other_continental = b
                .other
                .is_some_and(|o| plates[usize::from(o)].affinity >= Gf::ZERO);
            province_of(c, u, h, sea_z, b.kind, other_continental).code()
        })
        .collect();
    Land {
        z,
        uplift,
        sea_z,
        province,
        plate: boundaries.iter().map(|b| b.plate).collect(),
        kind: boundaries.iter().map(|b| b.kind as u8).collect(),
        boundary_m: boundaries
            .iter()
            .map(|b| match b.other {
                Some(_) => (radius * b.sine).to_i64_floor() as u32,
                None => u32::MAX,
            })
            .collect(),
        crust,
        plates,
        threshold,
        freeboard_factor,
        ocean,
        pockets,
        pocket_nodes,
    }
}

/// ★ FORM (b), THE ROOT THROUGH FLEXURE, sixteenths a node: each node's crust-thickness scatter `δt` (the field times
/// [`CRUST_FIELD_SPREAD`] times the plate's continental crust, by the node's crust share) is a root whose buoyancy
/// `(ρ_m − ρ_c)·g·δt` lifts the plate; the solve's own flexural rebound lifts the land by `ρ_c/ρ_m` of a removal, so
/// a removal of `(ρ_m − ρ_c)/ρ_c · δt` gives the lift `(ρ_m − ρ_c)/ρ_m · δt` low-passed at the flexural parameter —
/// Airy's own number at the waves the plate cannot hold, nothing at the waves it can.
#[allow(clippy::too_many_arguments)]
fn flex_lift(
    body: &BodyDefinition,
    z_dry: &[i32],
    crust: &[u8],
    field: &[f64],
    plates: &[Plate],
    boundaries: &[Boundary],
    cap_m: f64,
    freeboard: Gf,
    alpha_m: f64,
) -> Vec<i32> {
    let n = z_dry.len();
    let provisional = Land {
        z: z_dry.to_vec(),
        uplift: vec![0; n],
        sea_z: None,
        province: vec![0; n],
        plate: boundaries.iter().map(|b| b.plate).collect(),
        kind: vec![0; n],
        boundary_m: vec![u32::MAX; n],
        crust: crust.to_vec(),
        plates: plates.to_vec(),
        threshold: Gf::ZERO,
        freeboard_factor: freeboard,
        ocean: vec![false; n],
        pockets: 0,
        pocket_nodes: 0,
    };
    let mut state = MacroSolve::from_land(body, &provisional).expect("a lattice");
    let cap_ratio = cap_m / EARTH_RELIEF_CAP_M;
    let removal =
        (MANTLE_DENSITY_KGM3 - CONTINENTAL_CRUST_DENSITY_KGM3) / CONTINENTAL_CRUST_DENSITY_KGM3;
    for k in 0..n {
        let own = &plates[usize::from(boundaries[k].plate)];
        let t_c =
            EARTH_CONTINENTAL_CRUST_M * cap_ratio * own.crust_scatter.to_f64() * freeboard.to_f64();
        let dt = CRUST_FIELD_SPREAD * field[k] * t_c * f64::from(crust[k]) / 255.0;
        state.pending[k] = (removal * dt * f64::from(Z_STEPS_PER_M)).round() as i32;
    }
    let before = state.z.clone();
    let _ = state.rebound(alpha_m);
    state.z.iter().zip(&before).map(|(a, b)| a - b).collect()
}
