//! ★ THE RECIPE ON THE GPU — the compute entry points (ruling F7/F8, decision 5: rust-gpu, one
//! crate, two compilations). This crate is the thin shell cargo-gpu compiles to SPIR-V: each entry
//! point reads its bindings and calls `vd-recipe`'s kernels, the SAME functions the server and the
//! client's CPU path call. No arithmetic lives here; a kernel here would be a second copy.
//!
//! The first entry point is the bench's octave sum: the relief of one column per invocation over
//! the same integer inputs the CPU reads. The client's runtime self-check (F8 decision 3) and the
//! cell-field step (G1) grow from it.
//!
//! **Example.** The client asks the GPU for the relief of a chunk's 3 844 columns: one workgroup of
//! 256 invocations at a time, each reading its direction, summing the fourteen octaves through
//! `vd_recipe::height::relief`, and writing one word — the word the server's CPU writes for the
//! same column.

#![cfg_attr(target_arch = "spirv", no_std)]

use spirv_std::glam::{UVec3, UVec4};
use spirv_std::spirv;
use vd_recipe::Gi;
use vd_recipe::height::{OCTAVES_CAP, Octave, relief_of_table};

/// The words one octave takes in the `octaves` buffer: the seed, the frequency's integer part, the
/// frequency's fraction, the amplitude — each one 64-bit word, the bench's packing.
pub const WORDS_PER_OCTAVE: usize = 4;
/// One octave from its four words.
fn octave_at(words: &[u64], k: usize) -> Octave {
    let base = k * WORDS_PER_OCTAVE;
    Octave {
        seed: words[base],
        frequency_int: Gi::new(words[base + 1] as i64),
        frequency_frac: Gi::new(words[base + 2] as i64),
        amplitude: Gi::new(words[base + 3] as i64),
    }
}

/// The relief of one column per invocation: `dirs` holds three words per column at 40 fraction
/// bits, `octaves` the packed octaves, `count.x` their number, `out` one word per column.
#[spirv(compute(threads(256)))]
pub fn relief_columns(
    #[spirv(global_invocation_id)] id: UVec3,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 0)] dirs: &[i64],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 1)] octaves: &[u64],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 2)] out: &mut [i64],
    #[spirv(uniform, descriptor_set = 0, binding = 3)] count: &UVec4,
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
    let mut n = count.x as usize;
    if n > OCTAVES_CAP {
        n = OCTAVES_CAP;
    }
    // The octaves as a fixed-size stack array: a shader holds no heap.
    let mut octs = [Octave {
        seed: 0,
        frequency_int: Gi::ZERO,
        frequency_frac: Gi::ZERO,
        amplitude: Gi::ZERO,
    }; OCTAVES_CAP];
    let mut k = 0;
    while k < n {
        octs[k] = octave_at(octaves, k);
        k += 1;
    }
    out[i] = relief_of_table(&octs, n, dir).raw();
}
