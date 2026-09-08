//! ★ THE VOXEL FOUNDATION's CROSS-PINS: constants two crates state separately because the lower one
//! may not depend on the upper one, tied together here in the one crate that sees both.

/// The gap byte's density convention (slice 3, in the registry's identity digest) and the generator's
/// quantum (slice 5) are one number: a change on either side without the other would quantise a
/// saved trench in one step and read it in another.
#[test]
fn the_gap_steps_per_cell_are_one_number_in_the_registry_and_in_the_generator() {
    assert_eq!(
        i64::from(vd_core::registry::GAP_STEPS_PER_CELL),
        vd_terrain::chunk::GAP_STEPS_PER_CELL
    );
    assert_eq!(vd_terrain::chunk::GAP_STEPS_PER_CELL, 128);
}
