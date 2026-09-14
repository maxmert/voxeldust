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
//! GPU-REQUIRED AND LOCAL, like the seam probe and the picture gate: it needs a working adapter,
//! it is not in `just gate`, and it is run with `just gpu-drift`. Without the `render` feature this
//! file compiles to zero tests.

#![cfg(feature = "render")]

use vd_client_render::gpu_check::BoxGear;
use vd_client_render::wgpu;
use vd_seed::bend::Face;
use vd_terrain::BodyDefinition;
use vd_terrain::chunk::ChunkKey;
use vd_terrain::home::home_planet;

/// ★ THE SEAM STAND'S EYE, in the home planet's own frame — where the picture gate stands for its
/// fifth picture, and the stand whose hole found the drift. The stand sits ON a cube face seam, so
/// its wanted set holds the boxes that carry a PARTNER lattice beside the ones that carry none:
/// the very shrink the pool's old rule could not survive.
const SEAM_STAND_EYE_M: [f64; 3] = [
    4_280_492.264_785_528,
    -1_990_175.796_753_562_3,
    -4_280_492.264_785_528,
];

/// ★ THE KEY THE SEAM STAND DREW A HOLE FOR: the first box of the 1 022 that drifted. It carries
/// EIGHT lattices, and the box before it in the stand's own order carries SIXTEEN.
const FAILING_KEY: ChunkKey = ChunkKey {
    face: Face::PosX,
    rung: 9,
    x: 69,
    y: 7,
    z: 0,
};

/// ★ THE BOX BEFORE IT, which grew the pool: a rung-0 box with SIXTEEN lattices. The pair is the
/// whole defect in two submits — a bigger box, then a smaller one on the same gear.
const PREDECESSOR_KEY: ChunkKey = ChunkKey {
    face: Face::NegZ,
    rung: 0,
    x: 35_929,
    y: 161_395,
    z: 305,
};

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
    let before = vd_terrain::gpu::plan(&body, PREDECESSOR_KEY).expect("the key is on the ladder");
    let after = vd_terrain::gpu::plan(&body, FAILING_KEY).expect("the key is on the ladder");
    // The pair is only a trap while the second box asks for FEWER lattice rows than the first: the
    // gate states that, so a world whose ladder changes cannot leave this test passing on nothing.
    assert!(
        after.lattice_words().len() < before.lattice_words().len(),
        "the pin needs a SHRINK: {PREDECESSOR_KEY:?} asks for {} lattice words and {FAILING_KEY:?} \
         for {}",
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
        "THE CARD'S BOX IS NOT THE CPU'S at {FAILING_KEY:?}, first at {first:?}"
    );
    // AND AGAINST THE GENERATOR'S OWN READING, which is what the shard computes collision on and
    // what SL10 names: the plan's host run is one path to the box, `sample_box` is the other.
    let cpu =
        vd_terrain::lattice::sample_box(&body, FAILING_KEY).expect("the box is on the ladder");
    let built = after.box_of(&card, &vd_terrain::gpu::BoxPlan::dirs_of(&after.run()));
    let cells = cpu
        .cells
        .iter()
        .zip(built.cells.iter())
        .filter(|(a, b)| a != b)
        .count();
    assert_eq!(
        cells, 0,
        "the card's box is not `sample_box`'s at {FAILING_KEY:?}"
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
    let wanted = view.wanted(&body, SEAM_STAND_EYE_M);
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

/// ★ WHY THAT KEY, AND NOT ANOTHER — the stand's own order (§26.10). The seam stand asks for the
/// failing box IMMEDIATELY AFTER a box that carries TWO cavern lattices, and the second of those
/// two names the failing box's OWN FACE. That is the whole trap: the wide box grew the gear's
/// lattice binding to two rows, the narrow box wrote one, and the kernel walked the binding by its
/// slice's own length — so every column of the narrow box found a lattice that box does not have.
///
/// This test needs no card. It states the ORDER, so a ladder that ever stops putting a narrow box
/// behind a wide one tells us here, where the pin above would quietly stop pinning anything.
#[test]
fn the_stand_asks_for_the_narrow_box_right_behind_a_wide_one() {
    let body = home_planet();
    let mut view = vd_client::ladder_view::LadderView::default();
    let wanted = view.wanted(&body, SEAM_STAND_EYE_M);
    let at = wanted
        .keys
        .iter()
        .position(|k| *k == FAILING_KEY)
        .expect("the seam stand asks for the key that drew the hole");
    let before = vd_terrain::gpu::plan(&body, wanted.keys[at - 1]).expect("the box before it");
    let after = vd_terrain::gpu::plan(&body, FAILING_KEY).expect("the box that drifted");
    eprintln!(
        "gpu_no_drift: the seam stand wants {} boxes; {FAILING_KEY:?} stands at {at}, behind \
         {:?} which carries {} lattice words and {} nodes against its own {} and {}",
        wanted.keys.len(),
        before.key,
        before.lattice_words().len(),
        before.node_count,
        after.lattice_words().len(),
        after.node_count,
    );
    assert!(
        after.lattice_words().len() < before.lattice_words().len(),
        "the box before the failing one no longer carries more lattice rows: the pin is idle"
    );
    assert!(
        after.node_count < before.node_count,
        "the box before the failing one no longer carries more nodes: the pin is idle"
    );
}
