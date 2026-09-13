//! ★ THE RECIPE ON THE GPU — the compute entry points (ruling F7/F8, decision 5: rust-gpu, one
//! crate, two compilations). This crate is the thin shell cargo-gpu compiles to SPIR-V: each entry
//! point reads its bindings and calls `vd-recipe`'s kernels, the SAME functions the server and the
//! client's CPU path call. No arithmetic lives here; a kernel here would be a second copy.
//!
//! The first entry point is the bench's octave sum: the relief of one column per invocation over
//! the same integer inputs the CPU reads — the octaves IN PLACE from their buffer, as the recipe's
//! own `Octave` words (MEASURED: a per-thread copy of the table into private memory cost the
//! module three to six times the transcription's time). The client's runtime self-check (F8
//! decision 3) and the cell-field step (G1) grow from it.
//!
//! **Example.** The client asks the GPU for the relief of a chunk's 3 844 columns: one workgroup of
//! 256 invocations at a time, each reading its direction, summing the fourteen octaves through
//! `vd_recipe::height::relief`, and writing one word — the word the server's CPU writes for the
//! same column.

#![cfg_attr(target_arch = "spirv", no_std)]

use spirv_std::glam::UVec3;
use spirv_std::spirv;
use vd_recipe::Gi;
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
