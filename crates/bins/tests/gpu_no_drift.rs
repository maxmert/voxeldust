//! ★★ THE NO-DRIFT GATE OVER MANY KEYS (SL10, 2026-09-14): the card's box against the CPU's, on
//! the keys a real stand draws and not on eight golden ones.
//!
//! SL10 asks for no drift as a MEASUREMENT on every shipped target. The boot self-check measures
//! it ONCE, over the eight keys the world identity folds, and every one of them passed while the
//! seam stand drew 14 959 pixels of nothing under its ground (§26.9). A gate of eight keys is not
//! a gate over the world.
//!
//! **THE DEFECT THIS GATE PINS** (§26.10). The card's gear POOLS its buffers, and the pool used to
//! GROW and never shrink. A kernel walks a binding by ITS SLICE'S OWN LENGTH —
//! `vd_recipe::plan::column_row` walks the box's lattices that way — so a box with EIGHT lattices,
//! built on a gear whose lattice buffer still held SIXTEEN from an earlier box, read the earlier
//! box's rows. MEASURED: 1 022 of the seam stand's 6 049 boxes drifted through a reused gear, and
//! NOT ONE through a fresh one.
//!
//! **SO THIS GATE REUSES ONE GEAR, BY NAME.** A gate that builds every box on fresh buffers cannot
//! see the defect at all — which is exactly why the first eight-key check did not.
//!
//! ★ **WHERE THE PIN'S PAIR COMES FROM, SINCE 2026-09-16.** The SWEEP still flies the seam stand's
//! own wanted set, box after box. The PIN no longer takes its pair from that order: stage 1's
//! extended ladder moved the stand onto ground where every box it wants carries ONE cavern lattice,
//! so no narrow box stands behind a wide one there any more. The pair is now CONSTRUCTED from the
//! body's own ladder and CHOSEN BY THE TRAP — a face's CORNER box at rung 0, which carries three
//! cavern lattice rows, against the SAME corner at the TOP rung, where nothing is carved and the
//! box carries one row of no face while its columns still stand on both partner faces. A shrink
//! alone is not the trap, and MEASURING that was the day's second lesson. See
//! [`the_body_asks_for_a_narrow_box_right_behind_a_wide_one`].
//!
//! GPU-REQUIRED AND LOCAL, like the seam probe and the picture gate: it needs a working adapter,
//! it is not in `just gate`, and it is run with `just gpu-drift`. Without the `render` feature this
//! file compiles to zero tests.

#![cfg(feature = "render")]

use vd_client_render::gpu_check::BoxGear;
use vd_client_render::wgpu;
use vd_seed::bend::Face;
use vd_terrain::BodyDefinition;
use vd_terrain::chunk::{CHUNK_EDGE, ChunkKey, in_ladder};
use vd_terrain::digest::surface_chunk_z;
use vd_terrain::home::home_planet;

/// ★ THE SEAM STAND'S EYE, in the home planet's own frame — where the picture gate stands for its
/// fifth picture, and the stand whose hole found the drift. The SWEEP below flies its whole wanted
/// set, which is why this eye stays.
///
/// ⚠ ITS ORDER NO LONGER HOLDS A SHRINK. It once did — the stand sits on a cube face seam, and its
/// set held boxes carrying a PARTNER lattice beside boxes carrying none. On stage 1's extended
/// ladder every box it wants carries ONE. MEASURED, and printed by every run: see the finding on
/// [`the_body_asks_for_a_narrow_box_right_behind_a_wide_one`].
const SEAM_STAND_EYE_M: [f64; 3] = [
    4_280_492.264_785_528,
    -1_990_175.796_753_562_3,
    -4_280_492.264_785_528,
];

/// ★ WHAT ONE ORDER OF BOXES ASKS ITS GEAR FOR: each box's cavern-lattice row count in the order
/// a builder meets them, the histogram of those counts, and how many times a box asks for FEWER rows
/// than the box right before it. That SHRINK is the trap the pin holds — the wide box grew the gear's
/// lattice binding, the narrow box wrote fewer rows, and the kernel walked the binding by its slice's
/// own length, so every column of the narrow box found a lattice that box does not have.
struct Census {
    /// How many of the keys stand on the ladder and have a plan.
    planned: usize,
    /// Lattice words per box, against how many boxes ask for that many.
    histogram: std::collections::BTreeMap<usize, usize>,
    /// How many consecutive pairs shrink.
    shrinks: usize,
    /// The FIRST shrinking pair in the order: the wide box, then the narrow one behind it.
    pair: Option<(ChunkKey, ChunkKey)>,
}

impl Census {
    /// The order's own reading, in one line a reader can act on.
    fn state(&self, name: &str) -> String {
        format!(
            "{name}: {} boxes planned, lattice-word histogram {:?}, {} shrinks",
            self.planned, self.histogram, self.shrinks
        )
    }
}

/// The census of `keys`, read in the order given.
fn census(body: &BodyDefinition, keys: &[ChunkKey]) -> Census {
    let mut out = Census {
        planned: 0,
        histogram: std::collections::BTreeMap::new(),
        shrinks: 0,
        pair: None,
    };
    let mut previous: Option<(ChunkKey, usize)> = None;
    for key in keys {
        let Some(plan) = vd_terrain::gpu::plan(body, *key) else {
            continue;
        };
        let words = plan.lattice_words().len();
        out.planned += 1;
        *out.histogram.entry(words).or_default() += 1;
        if let Some((before, before_words)) = previous
            && words < before_words
        {
            out.shrinks += 1;
            if out.pair.is_none() {
                out.pair = Some((before, *key));
            }
        }
        previous = Some((*key, words));
    }
    out
}

/// The keys the seam stand itself asks for, in its own order — a MEASUREMENT the tests print, and no
/// longer where the pin's pair comes from. See the finding on
/// [`the_body_asks_for_a_narrow_box_right_behind_a_wide_one`].
fn seam_stand_keys(body: &BodyDefinition) -> Vec<ChunkKey> {
    let mut view = vd_client::ladder_view::LadderView::default();
    view.wanted(body, SEAM_STAND_EYE_M, None).keys
}

/// The TOP rung of a body: the coarsest, where no cavern is carved at all.
fn top_rung(body: &BodyDefinition) -> u8 {
    body.ladder().rungs.saturating_sub(1)
}

/// The MIDDLE chunk index along a face edge at rung 0 — the last chunk index, halved. A CHUNK
/// index, never a cell index: a cell index put every middle box off the ladder, and the census
/// counted twelve boxes where the construction names eighteen.
#[allow(
    clippy::integer_division,
    reason = "a test on the host: a chunk is 62 cells, which no shift divides"
)]
fn middle_chunk(body: &BodyDefinition) -> i32 {
    ((body.ladder().cells_per_edge(0) as i32 - 1) / CHUNK_EDGE as i32) >> 1
}

/// ★★ THE CANDIDATE ORDER, BUILT FROM THE LADDER SO THE PIN CANNOT GO QUIET (re-derived
/// 2026-09-16, stage 2 of slice 8a, when the ground moved the seam stand's whole wanted set into
/// ONE row count). Three boxes per face, and the pair search below picks the trap out of them:
///
/// - THE FACE'S CORNER AT RUNG 0. Caverns are carved at rung 0 and its halo reaches across TWO
///   seams, so it carries THREE cavern lattice rows: its own face and both partners.
/// - THE SAME CORNER AT THE TOP RUNG. No cavern is carved there at all, so the box carries ONE
///   stand-in row of NO FACE — while its columns STILL sit on all three faces.
/// - THE MIDDLE OF THE FACE AT RUNG 0, which carries one real row and whose columns all sit on its
///   own face.
fn shrinking_order(body: &BodyDefinition) -> Vec<ChunkKey> {
    let mut keys: Vec<ChunkKey> = Vec::new();
    let top = top_rung(body);
    let middle = middle_chunk(body);
    for face in Face::ALL {
        let corner = ChunkKey {
            face,
            rung: 0,
            x: 0,
            y: 0,
            z: surface_chunk_z(body, face, 0, 0, 0),
        };
        let coarse_corner = ChunkKey {
            face,
            rung: top,
            x: 0,
            y: 0,
            z: surface_chunk_z(body, face, top, 0, 0),
        };
        let mid = ChunkKey {
            face,
            rung: 0,
            x: middle,
            y: middle,
            z: surface_chunk_z(body, face, 0, middle, middle),
        };
        for candidate in [corner, coarse_corner, mid] {
            if in_ladder(body, candidate) {
                keys.push(candidate);
            }
        }
    }
    keys
}

/// ★★ WHETHER A PAIR IS THE TRAP, AND NOT MERELY A SHRINK — the whole lesson of 2026-09-16.
///
/// `vd_recipe::plan::column_row` takes the FIRST row whose face the column names (`has == 0` guards
/// every later row), so a stale row sitting BEHIND a row the box really has is read by nobody. The
/// trap needs a column of the narrow box whose face the narrow box's OWN rows do not carry, while
/// the wide box's rows carry it at an index the narrow box never wrote. Then the kernel walks the
/// binding by its slice's own length, finds the stale row, and reads a lattice this box does not
/// have.
///
/// **Example.** The pilot's client builds the corner box of face `+X` at rung 0: three rows — `+X`,
/// `+Y`, `−Z`. Next it builds the SAME corner at the top rung, where nothing is carved: one row of
/// NO FACE. On a gear that kept the first box's rows, every halo column of that box standing on
/// `+Y` finds the earlier box's `+Y` lattice and the player flies at a cave that is not there.
///
/// Answers the face a stale row would hand over, or `None` where the pair is safe by construction.
fn stale_face_of(
    wide: &vd_terrain::gpu::BoxPlan,
    narrow: &vd_terrain::gpu::BoxPlan,
) -> Option<i64> {
    let kept = narrow.lattices.len();
    if wide.lattices.len() <= kept {
        return None;
    }
    for site in &narrow.sites {
        if site.face == vd_terrain::lattice::CORNER_FACE {
            continue;
        }
        let want = i64::from(site.face);
        if narrow.lattices.iter().any(|b| b.face.raw() == want) {
            continue;
        }
        if wide.lattices[kept..].iter().any(|b| b.face.raw() == want) {
            return Some(want);
        }
    }
    None
}

/// ★ THE PIN'S PAIR, AND THE CENSUS THAT FOUND IT: the first ordered pair of [`shrinking_order`]
/// that is a TRAP by [`stale_face_of`]. The panic is the FINDING a body owes the reader — a body
/// on which no box can hand a stale row to another has nothing for the pooled-buffer pin to hold,
/// and the histogram says so in the same breath.
fn shrinking_pair(body: &BodyDefinition) -> (Census, ChunkKey, ChunkKey, i64) {
    let order = shrinking_order(body);
    let found = census(body, &order);
    let plans: Vec<(ChunkKey, vd_terrain::gpu::BoxPlan)> = order
        .iter()
        .filter_map(|k| vd_terrain::gpu::plan(body, *k).map(|p| (*k, p)))
        .collect();
    for (wi, (wide_key, wide)) in plans.iter().enumerate() {
        for (narrow_key, narrow) in plans.iter().skip(wi + 1) {
            if let Some(face) = stale_face_of(wide, narrow) {
                return (found, *wide_key, *narrow_key, face);
            }
        }
    }
    panic!(
        "NO BOX OF THIS BODY CAN HAND A STALE CAVERN LATTICE ROW TO ANOTHER: the pooled-buffer pin \
         has nothing to hold. {}",
        found.state("the constructed order")
    );
}

/// HOW MANY OF THE STAND'S OWN BOXES THE SWEEP TAKES: EVERY ONE. The whole wanted set is 6 049
/// boxes and MEASURED 30 s on this machine, which a local GPU gate can pay — and the 1 022 that
/// drifted before the cure are spread across it, so a gate that stopped early would judge a part
/// of the stand and call it the stand. `VD_GPU_DRIFT_BOXES=<n>` takes the first `n` for a quick
/// look while a cure is being hunted.
const SWEEP_BOXES: usize = 0;

/// The lanes the shipped builder flies (`CARD_IN_FLIGHT`): the gate builds the way the client does.
const LANES: usize = 2;

/// A minimal executor for the two async calls of device creation: poll with a no-op waker.
fn block_on<F: std::future::Future>(future: F) -> F::Output {
    let mut future = std::pin::pin!(future);
    let waker = std::task::Waker::noop();
    let mut cx = std::task::Context::from_waker(waker);
    loop {
        if let std::task::Poll::Ready(v) = future.as_mut().poll(&mut cx) {
            return v;
        }
        std::thread::yield_now();
    }
}

/// The card this machine offers, with 64-bit integers where it has them.
fn device() -> (wgpu::Device, wgpu::Queue) {
    let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor::default());
    let adapter = block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
        power_preference: wgpu::PowerPreference::HighPerformance,
        ..Default::default()
    }))
    .expect("this gate requires a working GPU adapter");
    let info = adapter.get_info();
    eprintln!(
        "gpu_no_drift: adapter {} ({:?}, {:?})",
        info.name, info.backend, info.device_type
    );
    assert!(
        adapter.features().contains(wgpu::Features::SHADER_INT64),
        "the recipe is 64-bit integer arithmetic: a card without SHADER_INT64 never builds"
    );
    let (device, queue) = block_on(adapter.request_device(&wgpu::DeviceDescriptor {
        label: Some("gpu_no_drift"),
        required_features: wgpu::Features::SHADER_INT64,
        required_limits: adapter.limits(),
        ..Default::default()
    }))
    .expect("a device");
    (device, queue)
}

/// The cells of a readback, as the client's geometry stage decodes them.
fn cells_of(bytes: &[u8]) -> Vec<u32> {
    bytes
        .chunks_exact(4)
        .map(|b| u32::from_le_bytes(b.try_into().unwrap_or([0; 4])))
        .collect()
}

/// ONE BOX ON THE CARD against the same box on this host: how many cell words differ, and the
/// first one that does. A short readback is every cell differing, because a box the card did not
/// finish is not the CPU's box either.
fn differing(plan: &vd_terrain::gpu::BoxPlan, card: &[u32]) -> (usize, Option<(usize, u32, u32)>) {
    let host = plan.cells_of(&plan.run());
    if host.len() != card.len() {
        return (host.len(), Some((0, 0, 0)));
    }
    let mut count = 0usize;
    let mut first = None;
    for (i, want) in host.iter().enumerate() {
        if *want != card[i] {
            count += 1;
            if first.is_none() {
                first = Some((i, card[i], *want));
            }
        }
    }
    (count, first)
}

/// ★ THE PIN: the two boxes that drifted, in their own order, on ONE gear. A gear that sizes its
/// bindings to the box it is building answers the CPU's own words; the gear that grew and never
/// shrank gave this box 128 834 cells of the PREVIOUS box's surface.
#[test]
fn the_smaller_box_after_a_bigger_one_is_still_the_cpus_box() {
    let (device, queue) = device();
    let body = home_planet();
    let mut gear = BoxGear::new(device, queue);
    let (found, predecessor, failing, stale_face) = shrinking_pair(&body);
    eprintln!("gpu_no_drift: {}", found.state("the constructed order"));
    assert!(
        found.shrinks >= 1,
        "the pin needs a SHRINK to hold anything. {}",
        found.state("the constructed order")
    );
    eprintln!(
        "gpu_no_drift: a gear that kept the wide box's rows hands face {stale_face} to the narrow \
         box's columns"
    );
    let before = vd_terrain::gpu::plan(&body, predecessor).expect("the key is on the ladder");
    let after = vd_terrain::gpu::plan(&body, failing).expect("the key is on the ladder");
    // The pair is only a trap while the second box asks for FEWER lattice rows than the first, which
    // is how `shrinking_pair` chose it; the gate states it again so the choice cannot go quiet.
    assert!(
        after.lattice_words().len() < before.lattice_words().len(),
        "the pin needs a SHRINK: {predecessor:?} asks for {} lattice words and {failing:?} for {}",
        before.lattice_words().len(),
        after.lattice_words().len()
    );
    eprintln!(
        "gpu_no_drift: the shrinking pair is {predecessor:?} ({} lattice words) then {failing:?} \
         ({} words)",
        before.lattice_words().len(),
        after.lattice_words().len()
    );
    let grown = gear
        .run(&before)
        .expect("the card answered for the bigger box");
    let (count, first) = differing(&before, &cells_of(&grown.cells));
    assert_eq!(count, 0, "the bigger box itself differs at {first:?}");
    // THE SAME GEAR, the smaller box.
    let run = gear
        .run(&after)
        .expect("the card answered for the smaller box");
    let card = cells_of(&run.cells);
    let (count, first) = differing(&after, &card);
    assert_eq!(
        count, 0,
        "THE CARD'S BOX IS NOT THE CPU'S at {failing:?}, first at {first:?}"
    );
    // AND AGAINST THE GENERATOR'S OWN READING, which is what the shard computes collision on and
    // what SL10 names: the plan's host run is one path to the box, `sample_box` is the other.
    let cpu =
        vd_terrain::lattice::sample_box(&body, None, failing).expect("the box is on the ladder");
    let built = after.box_of(&card, &vd_terrain::gpu::BoxPlan::dirs_of(&after.run()));
    let cells = cpu
        .cells
        .iter()
        .zip(built.cells.iter())
        .filter(|(a, b)| a != b)
        .count();
    assert_eq!(
        cells, 0,
        "the card's box is not `sample_box`'s at {failing:?}"
    );
}

/// ★ THE BOOT SELF-CHECK, WHICH THE CLIENT RUNS BEFORE IT LETS THE CARD BUILD AT ALL: it now
/// carries the pool's own pass — the eight golden boxes through ONE reused gear — beside the
/// columns and the fresh-gear cells. This gate states that the check the client trusts is green on
/// this machine.
#[test]
fn the_boot_self_check_is_green_over_a_reused_gear() {
    let (device, queue) = device();
    let body = home_planet();
    let check = vd_client_render::gpu_check::run(&device, &queue, &body);
    eprintln!(
        "gpu_no_drift: the boot self-check compared {} columns and {} cells in {} µs, and its pool \
         pass shrank a binding {} times",
        check.columns, check.cells, check.micros, check.pool_shrinks
    );
    assert_eq!(check.differing, 0, "a column direction is not the CPU's");
    assert_eq!(check.differing_cells, 0, "a cell word is not the CPU's");
    assert!(
        check.trusted(),
        "the client would refuse the card: {check:?}"
    );
    // ★ AND THE PASS COULD HAVE FAILED: a pool pass whose boxes all ask for the same rows never
    // meets the defect that drew the seam stand's hole. MEASURED: with the pool's old growing rule
    // restored, the eight golden keys alone stayed green and this check goes red.
    assert!(
        check.pool_shrinks > 0,
        "the boot check's pool pass never asked its reused gear for fewer rows: it cannot see a \
         stale row at all"
    );
}

/// ★★ THE SWEEP — the seam stand's OWN wanted set, box after box on the SAME gears the client
/// flies, at the lanes it flies. This is the gate SL10 asks for: the keys a stand draws, not the
/// keys a golden list names.
#[test]
fn the_card_builds_the_seam_stands_own_boxes_byte_for_byte() {
    let (device, queue) = device();
    let body: BodyDefinition = home_planet();
    let mut view = vd_client::ladder_view::LadderView::default();
    let wanted = view.wanted(&body, SEAM_STAND_EYE_M, None);
    // Zero is every box the stand wants; a count takes that many.
    let cap = std::env::var("VD_GPU_DRIFT_BOXES")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .unwrap_or(SWEEP_BOXES);
    let take = if cap == 0 {
        wanted.keys.len()
    } else {
        cap.min(wanted.keys.len())
    };
    assert!(take > 0, "the seam stand wants no chunks at all");
    let mut gears: Vec<BoxGear> = (0..LANES)
        .map(|_| BoxGear::new(device.clone(), queue.clone()))
        .collect();
    let started = std::time::Instant::now();
    let mut built = 0usize;
    let mut drifted: Vec<String> = Vec::new();
    let mut window: Vec<vd_terrain::gpu::BoxPlan> = Vec::new();
    let mut keys = wanted.keys.iter().take(take);
    loop {
        // Fill the lanes, then collect them in the same order: the builder's own shape.
        window.clear();
        while window.len() < gears.len() {
            let Some(key) = keys.next() else { break };
            let Some(plan) = vd_terrain::gpu::plan(&body, *key) else {
                continue;
            };
            gears[window.len()].submit(&plan);
            window.push(plan);
        }
        if window.is_empty() {
            break;
        }
        for (lane, plan) in window.iter().enumerate() {
            let run = gears[lane].collect().expect("the card answered");
            let (count, first) = differing(plan, &cells_of(&run.cells));
            built += 1;
            if count > 0 && drifted.len() < 8 {
                drifted.push(format!("{:?}: {count} cells, first {first:?}", plan.key));
            }
        }
    }
    eprintln!(
        "gpu_no_drift: {built} of the seam stand's {} boxes through the card's own gears, {} in \
         flight, in {:.1} s",
        wanted.keys.len(),
        gears.len(),
        started.elapsed().as_secs_f64()
    );
    assert!(
        drifted.is_empty(),
        "THE CARD'S BOX IS NOT THE CPU'S on {} of {built} boxes: {drifted:?}",
        drifted.len()
    );
}

/// ★ WHY THAT PAIR, AND NOT ANOTHER — and the FINDING the seam stand now states (2026-09-16).
///
/// **THE FINDING.** The seam stand's own wanted set NO LONGER HOLDS A SHRINK. Every box it asks
/// for carries ONE cavern lattice — eight lattice words — so no narrow box stands behind a wide
/// one anywhere in its order. MEASURED on stage 2's ground: 6 302 boxes wanted, 6 302 planned,
/// histogram `{8: 6302}`, 0 shrinks; and the SAME reading on the tree before stage 2 (5 907 boxes,
/// `{8: 5907}`, 0 shrinks), so it is stage 1's extended ladder that moved the stand, not the ridge.
/// The pin that read its order therefore stopped pinning anything, and it said so by failing.
///
/// **THE CURE, AND THE SECOND LESSON.** The pair is now CONSTRUCTED from the body's own ladder
/// ([`shrinking_order`]) and then CHOSEN BY THE TRAP ITSELF ([`stale_face_of`]), never by the
/// shrink alone. A shrink is not enough: `vd_recipe::plan::column_row` takes the FIRST row whose
/// face a column names, so a stale row behind a real one is read by nobody. MEASURED on this very
/// file — a first cure paired a rung-0 CORNER box (three rows) with a rung-0 MIDDLE box (one row),
/// a true shrink, and with the pool's old growing rule restored the pin stayed GREEN, because
/// every column of a middle box names its own face and takes row zero. The shipped pair is the
/// corner at rung 0 against the SAME corner at the TOP rung, where nothing is carved: one row of
/// NO FACE, and columns still standing on both partner faces, which the stale rows then hand over.
///
/// This test needs no card. It states the order and refuses a construction with no trap in it —
/// where the pin above would quietly hold nothing.
#[test]
fn the_body_asks_for_a_narrow_box_right_behind_a_wide_one() {
    let body = home_planet();
    // The stand's own reading, kept as a MEASUREMENT: the day the ground moves a shrink back into
    // its order, this line says so.
    let stand = census(&body, &seam_stand_keys(&body));
    eprintln!(
        "gpu_no_drift: {}",
        stand.state("the seam stand's own order")
    );
    let order = shrinking_order(&body);
    let (found, wide, narrow, stale_face) = shrinking_pair(&body);
    eprintln!("gpu_no_drift: {}", found.state("the constructed order"));
    assert!(
        found.shrinks >= 1,
        "the construction asks every gear for the same rows: the pin is idle. {}",
        found.state("the constructed order")
    );
    let before = vd_terrain::gpu::plan(&body, wide).expect("the wide box");
    let after = vd_terrain::gpu::plan(&body, narrow).expect("the narrow box");
    eprintln!(
        "gpu_no_drift: the constructed order holds {} boxes; the trap is {wide:?} ({} lattice \
         words, {} nodes) then {narrow:?} ({} words, {} nodes), and the stale row hands over face \
         {stale_face}",
        order.len(),
        before.lattice_words().len(),
        before.node_count,
        after.lattice_words().len(),
        after.node_count,
    );
    assert!(
        after.lattice_words().len() < before.lattice_words().len(),
        "the wide box no longer carries more lattice rows: the pin is idle"
    );
    assert!(
        after.node_count < before.node_count,
        "the wide box no longer carries more nodes: the pin is idle"
    );
    assert_eq!(
        stale_face_of(&before, &after),
        Some(stale_face),
        "the pair is a TRAP and not merely a shrink"
    );
}
