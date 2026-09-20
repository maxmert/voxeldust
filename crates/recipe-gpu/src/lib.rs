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
//! ★ **THE WHOLE CHAIN (step G2-A):** the COLUMN pass and the NODE pass now stand in front of the
//! cell field on the CARD, so a box's request carries its key and its charter and nothing that has
//! to be computed. Three compute passes of one command encoder: the cell field reads the column
//! buffer and the node buffer the first two filled, and neither ever crosses the bus.
//!
//! **What the host still does.** TOPOLOGY only: which face a column belongs to across a seam and
//! whether it is a corner phantom (`site_of`), which carvers reach the box, where each cavern
//! lattice stands, and the radial layers' rules. That is a few thousand integer comparisons against
//! a quarter of a million cells of arithmetic — MEASURED at 0.056 ms of a 2.01 ms box.
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
use vd_recipe::plan::{CAVERN_STRIDE, NodeBlock, PlanCharter, column_row, node_value};

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
        // The card holds no artifact row: every column's water is the body's sea.
        above_cell_word(charter, layer.r_steps << LENGTH_BITS, charter.sea_radius)
    } else {
        cell_word(
            charter,
            &CellAt {
                dir: column.dir,
                h: column.h,
                biome: column.biome,
                r_steps: layer.r_steps,
                water: charter.sea_radius,
            },
            cavern_at(column, layer, nodes),
            tubes,
        )
    };
}

/// ★ THE COLUMN PASS (step G2-A): one invocation per column of a box. It reads the column's SITE —
/// which face it belongs to and its cell there, or a corner phantom — and writes the whole column
/// row: the direction, the surface's radius, the biome, and where the column's cavern lattice sits.
///
/// The host no longer computes any of that. A box's request is now its key and its charter; the
/// 4 096 octave sums a box needs run here.
///
/// The bindings, in order: the plan charter, four 32-bit words per column (the face, the cell's two
/// indices, one of padding), the box's cavern lattices, one [`Column`] out per column, and the same
/// columns' DIRECTIONS as three words each — the one thing a host still reads back, because the
/// extractor places its vertices along them.
#[spirv(compute(threads(64)))]
pub fn column_pass(
    #[spirv(global_invocation_id)] id: UVec3,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 0)] charter: &PlanCharter,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 1)] sites: &[i32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 2)] lattices: &[NodeBlock],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 3)] columns: &mut [Column],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 4)] dirs: &mut [i64],
) {
    let i = id.x as usize;
    if i >= columns.len() {
        return;
    }
    let row = column_row(
        charter,
        lattices,
        sites[i * 4],
        sites[i * 4 + 1],
        sites[i * 4 + 2],
    );
    dirs[i * 3] = row.dir[0].raw();
    dirs[i * 3 + 1] = row.dir[1].raw();
    dirs[i * 3 + 2] = row.dir[2].raw();
    columns[i] = row;
}

/// ★ THE NODE PASS (step G2-A): one invocation per node of the box's cavern lattices. `id.x` is the
/// node along the lattice's first face axis, `id.y` along its second, and `id.z` names a SLICE —
/// which lattice and which radial node — through the small table the host lays down, so the kernel
/// divides nothing to find its place.
///
/// The bindings, in order: the plan charter, the box's lattices, two words per slice (the lattice
/// and the radial node index), the corner radius of each radial node, and the node values out.
#[spirv(compute(threads(32)))]
pub fn node_pass(
    #[spirv(global_invocation_id)] id: UVec3,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 0)] charter: &PlanCharter,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 1)] lattices: &[NodeBlock],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 2)] slices: &[i64],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 3)] radii: &[i64],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 4)] nodes: &mut [Gi],
) {
    // ★ EVERY READ IS GUARDED THE SAME WAY, and a failed guard WRITES NOTHING. A shader has no
    // bound check of its own: an index past a buffer reads a neighbouring allocation on one card and
    // zero on another, which is a drift SL10 could never measure. The three reads are the slice row,
    // the lattice it names, and the radial radius; each is tested before it is taken.
    let z = id.z as usize;
    if z * 2 + 1 >= slices.len() {
        return;
    }
    let lattice = slices[z * 2];
    let nc = slices[z * 2 + 1];
    if (lattice < 0) | (lattice as usize >= lattices.len()) | (nc < 0) | (nc as usize >= radii.len())
    {
        return;
    }
    let block = &lattices[lattice as usize];
    let face = block.face.raw();
    if face < 0 {
        return;
    }
    let d0 = block.dims[0].raw();
    let d1 = block.dims[1].raw();
    let (na, nb) = (id.x as i64, id.y as i64);
    if (na >= d0) | (nb >= d1) {
        return;
    }
    let index = (block.base.raw() + (nc * d1 + nb) * d0 + na) as usize;
    if index >= nodes.len() {
        return;
    }
    let stride = CAVERN_STRIDE as i64;
    nodes[index] = node_value(
        charter,
        face as i32,
        ((block.node0[0].raw() + na) * stride) as i32,
        ((block.node0[1].raw() + nb) * stride) as i32,
        Gi::new(radii[nc as usize]),
    );
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
