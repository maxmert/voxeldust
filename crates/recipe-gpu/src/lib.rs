//! ★ THE RECIPE ON THE GPU — the compute entry points (ruling F7/F8, decision 5: rust-gpu, one
//! crate, two compilations). This crate is the thin shell cargo-gpu compiles to SPIR-V: each entry
//! point reads its bindings and calls `vd-recipe`'s kernels, the SAME functions the server and the
//! client's CPU path call. No arithmetic lives here; a kernel here would be a second copy.
//!
//! The first entry point is the octave sum: the relief of one column per invocation over the same
//! integer inputs the CPU reads — the octaves IN PLACE from their buffer, as the recipe's own
//! `Octave` words (MEASURED: a per-thread copy of the table into private memory cost the module
//! three to six times the transcription's time). The second is THE CELL FIELD (step G1): one
//! invocation per cell of a box, each writing the cell's substance and gap as one word.
//!
//! **What the host still does for the cell field (G1 only).** The column pass — the direction, the
//! surface and the biome of each column — and the cavern lattice's node values are computed on the
//! CPU and uploaded; the host also resolves the lattice's TOPOLOGY (which face a column belongs to
//! across a seam, and whether it is a corner phantom) into plain indices. Steps G2 and after move
//! those to the card as well. The kernel's own arithmetic is already the recipe's.
//!
//! **Example.** The client asks the GPU for the 262 144 cells of chunk (face 2, rung 0, 19, 1, 4)
//! with its halo: 4 096 workgroups of 64 invocations, each reading its column, its radial layer and
//! the carvers that reach the box, and writing one word — the word the shard's CPU writes for the
//! same cell, which is why the ground the player walks on is the ground the card draws.

#![cfg_attr(target_arch = "spirv", no_std)]

use spirv_std::glam::UVec3;
use spirv_std::spirv;
use vd_recipe::Gi;
use vd_recipe::cell::{
    CellAt, CellCharter, Column, LAYER_ABOVE, LAYER_BELOW, LENGTH_BITS, Layer, Tube,
    above_cell_word, below_cell_word, cavern_at, cell_word,
};
use vd_recipe::height::{Octave, relief};

/// The relief of one column per invocation: `dirs` holds three words per column at 40 fraction
/// bits, `octaves` the body's live octaves as the recipe's own words, `out` one word per column.
#[spirv(compute(threads(256)))]
pub fn relief_columns(
    #[spirv(global_invocation_id)] id: UVec3,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 0)] dirs: &[i64],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 1)] octaves: &[Octave],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 2)] out: &mut [i64],
) {
    let i = id.x as usize;
    if i >= out.len() {
        return;
    }
    let dir = [
        Gi::new(dirs[i * 3]),
        Gi::new(dirs[i * 3 + 1]),
        Gi::new(dirs[i * 3 + 2]),
    ];
    out[i] = relief(octaves, dir).raw();
}

/// ★ THE CELL FIELD (step G1): one invocation per cell of a box. `id.x` is the cell along the
/// face's first axis, `id.y` along its second and `id.z` along the radial, so the dispatch is
/// `(edge / 64, edge, edge)` workgroups of 64 invocations for a box `edge` cells across.
///
/// The bindings, in order: the body's charter at this rung (which also states the box's edge), one
/// [`Layer`] per radial layer, one [`Column`] per column of the box, the cavern lattices' node
/// values laid end to end, the tube carvers that can reach the box (never an empty buffer: a
/// carver of no radius stands in, and it hollows nothing), and one output word per cell.
#[spirv(compute(threads(64)))]
pub fn cell_field(
    #[spirv(global_invocation_id)] id: UVec3,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 0)] charter: &CellCharter,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 1)] layers: &[Layer],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 2)] columns: &[Column],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 3)] nodes: &[Gi],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 4)] tubes: &[Tube],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 5)] out: &mut [u32],
) {
    let edge = charter.box_edge.raw() as u32;
    if (id.x >= edge) | (id.y >= edge) | (id.z >= edge) {
        return;
    }
    let index = ((id.z * edge + id.y) * edge + id.x) as usize;
    if index >= out.len() {
        return;
    }
    let layer = &layers[id.z as usize];
    let column = &columns[(id.y * edge + id.x) as usize];
    let rule = layer.rule.raw();
    out[index] = if rule == LAYER_BELOW {
        below_cell_word(charter)
    } else if rule == LAYER_ABOVE {
        above_cell_word(charter, layer.r_steps << LENGTH_BITS)
    } else {
        cell_word(
            charter,
            &CellAt {
                dir: column.dir,
                h: column.h,
                biome: column.biome,
                r_steps: layer.r_steps,
            },
            cavern_at(column, layer, nodes),
            tubes,
        )
    };
}

/// ★ A PROBE: the integer square root of each input word, so a host can measure the card's own
/// answer against the CPU's for ONE kernel. A box of a quarter of a million cells says only that
/// something differs; a probe says which function does. Both probes here found a real fault on
/// 2026-09-13 (the root's loop tail and the carvers' accumulator), and both stay as instruments —
/// the bench runs them before part 5, so a kernel rule broken again is named in one line.
#[spirv(compute(threads(256)))]
pub fn isqrt_probe(
    #[spirv(global_invocation_id)] id: UVec3,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 0)] input: &[i64],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 1)] out: &mut [i64],
) {
    let i = id.x as usize;
    if i >= out.len() {
        return;
    }
    out[i] = vd_recipe::root::isqrt(input[i] as u64) as i64;
}

/// ★ A PROBE: per invocation the hollow the carvers open at a point, the first carver's own
/// distance, and how many carvers the card sees — so a host can tell a wrong LIST from a wrong
/// DISTANCE. It was the second that failed and the first that did not, which is how the
/// accumulator's shape was found.
#[spirv(compute(threads(256)))]
pub fn hollow_probe(
    #[spirv(global_invocation_id)] id: UVec3,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 0)] tubes: &[Tube],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 1)] points: &[i64],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 2)] out: &mut [i64],
) {
    let i = id.x as usize;
    if i * 3 + 2 >= out.len() {
        return;
    }
    let p = [
        Gi::new(points[i * 3]),
        Gi::new(points[i * 3 + 1]),
        Gi::new(points[i * 3 + 2]),
    ];
    out[i * 3] = vd_recipe::cell::tube_hollow_steps(tubes, p).raw();
    out[i * 3 + 1] = tubes[0].distance_steps(p).raw();
    out[i * 3 + 2] = tubes.len() as i64;
}
