//! ★ THE RECIPE'S GPU COMPILATION AT BUILD TIME (ruling F8 decision 5): the client's build script
//! compiles the recipe's GPU shell (`crates/recipe-gpu`, the one source's entry points) to SPIR-V
//! with cargo-gpu — rust-gpu's own pinned nightly and back end, installed on first use — into the
//! build's output directory, where `include_bytes!` picks it up. No SPIR-V is committed: a binary
//! in the tree would be a second copy of the recipe.
//!
//! The prerequisite is the tool itself:
//! `cargo +nightly-2026-06-06 install --git https://github.com/Rust-GPU/rust-gpu cargo-gpu --locked`.
//! Without it the build stops here with that line, never with a stale module.

use std::path::PathBuf;
use std::process::Command;

fn main() {
    let manifest = PathBuf::from(std::env::var("CARGO_MANIFEST_DIR").expect("the manifest dir"));
    let shell = manifest.join("../recipe-gpu");
    let recipe = manifest.join("../recipe");
    let out = PathBuf::from(std::env::var("OUT_DIR").expect("the out dir")).join("recipe-gpu");
    println!("cargo:rerun-if-changed={}", shell.join("src").display());
    println!(
        "cargo:rerun-if-changed={}",
        shell.join("Cargo.toml").display()
    );
    println!("cargo:rerun-if-changed={}", recipe.join("src").display());
    println!(
        "cargo:rerun-if-changed={}",
        recipe.join("Cargo.toml").display()
    );
    let status = Command::new("cargo")
        .args(["gpu", "build", "--shader-crate"])
        .arg(&shell)
        .arg("--output-dir")
        .arg(&out)
        .args([
            "--target",
            "spirv-unknown-vulkan1.2",
            "--capabilities",
            "Int64",
            "--auto-install-rust-toolchain",
        ])
        .status();
    match status {
        Ok(s) if s.success() => {}
        Ok(s) => panic!("cargo gpu build failed with {s}: the recipe's GPU shell did not compile"),
        Err(e) => panic!(
            "cargo gpu is not installed ({e}): install it with `cargo +nightly-2026-06-06 install \
             --git https://github.com/Rust-GPU/rust-gpu cargo-gpu --locked` (ruling F8 decision 5)"
        ),
    }
    let spv = out.join("vd_recipe_gpu.spv");
    assert!(spv.is_file(), "cargo gpu build wrote no {}", spv.display());
    println!("cargo:rustc-env=VD_RECIPE_GPU_SPV={}", spv.display());
}
