//! ★ THE REGISTRY — the catalogue of everything a cell can be (the voxel foundation, slice 3;
//! rulings V6 Part B and V7).
//!
//! A saved record names a kind by a sixteen-bit number. This module says what that number means:
//! what it is made of (a SUBSTANCE), what shape it has (a FORM), what it does (a FUNCTION), how many
//! styles it has, which small-block scales it admits, and how much it weighs and how it breaks —
//! the last two DERIVED per kind from the substance's cited physical facts, never typed per cell.
//!
//! Five tables behind one mechanism: substance, form, function, block kind, attachment kind. Every
//! table is a `const` array whose index IS the id; ids are dense, append-only and never reused; a
//! retired row keeps its place, marked, with a replacement pointer. A lookup of an unknown number is a
//! typed refusal, never a default (the decode-to-Default ban), and no id type has a `Default`.
//!
//! **Identity keys, not display names.** Every row carries a `key`, a stable identity string that
//! never changes once the row exists, beside a `name` that may be reworded freely. The identity
//! digest folds keys, never names: a rename refuses nothing, a meaning swap refuses everything.
//!
//! **What may change and what may not** (ruling B-13). The part of a row that gives a SAVED record its
//! meaning — the kind's triple by KEY, the form's geometry, an attachment's key and whether it makes a
//! body, the gap convention — is folded into the identity digest over the rows `0..n`. A build that
//! knows MORE rows still matches an older store's prefix. Everything that only bounds what a player may
//! place NEXT (a variant count, a scale mask, a legal-orientation mask, a slot count, any physical or
//! render column) stays out, so a new style refuses nothing anywhere.
//!
//! **There are no themes** (ruling V7). A kind is placeable anywhere any kind is placeable. The only
//! cap is the sixteen-bit id.
//!
//! **Example.** A titanium hull plate is one kind: substance titanium alloy, form cube, function none,
//! three styles, every small-block scale allowed. A saved record says "that kind, style 1". The
//! registry says the rest, and never differently: two titanium plates never weigh differently.

pub mod attachment;
pub mod digest;
pub mod form;
pub mod function;
pub mod kind;
pub mod rotation;
pub mod substance;

pub use attachment::{ATTACHMENTS, AttachmentDef, AttachmentKindId};
pub use digest::{RegistryMismatch, RegistryStamp, identity_prefix_digest};
pub use form::{CELL_VOLUME_UNITS, ColliderClass, FORMS, FormClass, FormDef, ShapeId};
pub use function::{FUNCTIONS, FunctionDef, FunctionId};
pub use kind::{
    BlockKindId, KIND_COUNT, KINDS, KindDef, KindFlags, RegistryError, is_terrain,
    placement_allowed,
};
pub use rotation::{ORIENTATIONS, Orientation, Rotation};
pub use substance::{Family, MatterState, Provenance, SUBSTANCES, SubstanceDef, SubstanceId};

/// THE DENSITY CONVENTION of a terrain cell's gap byte (ruling B-3): the signed RADIAL distance from
/// the cell's CENTRE to the ground, in this many steps per cell, NEGATIVE inside solid. Both the step
/// count and the convention tag are part of the identity digest, because changing either reinterprets
/// every trench ever dug.
pub const GAP_STEPS_PER_CELL: i32 = 128;
/// The convention's tag: 1 = "radial gap, sampled at the cell centre, negative inside solid". A change
/// of sign, sample point or meaning is a new tag, and it refuses every older store loudly.
pub const GAP_CONVENTION_TAG: u8 = 1;

/// How the game compresses real fracture energies, which span five orders, into integrity points that
/// span three. The design names it `TOUGHNESS_COMPRESS = ½`: a square root.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Compression {
    SquareRoot,
}

/// The one place the game's tuned material constants live, each a sentence a player would recognise
/// (`block_system_design.md` §2.2.1). Few, global, named; every per-material number derives from cited
/// facts through them.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct GameScale {
    /// A destroyed block breaks into roughly 3 cm fragments: square metres of new surface per cubic
    /// metre of broken block.
    pub fragment_surface_m2_per_m3: u64,
    /// With the compression below, calibrates granite to 1 198 integrity points.
    pub toughness_base: u64,
    /// The compression of fracture energy into points.
    pub toughness_compress: Compression,
}

impl GameScale {
    /// The shipped scale.
    pub const SHIPPED: GameScale = GameScale {
        fragment_surface_m2_per_m3: 200,
        toughness_base: 268,
        toughness_compress: Compression::SquareRoot,
    };
}

/// A hand-written registry table's totality witness: every id below `len` resolves, `len` itself is
/// refused. Shared by the five tables' tripwire tests.
#[cfg(test)]
pub(crate) fn assert_dense<T, F: Fn(u16) -> Option<T>>(len: usize, lookup: F) {
    for i in 0..len {
        let id = u16::try_from(i).expect("a table index fits u16");
        assert!(lookup(id).is_some(), "id {id} must resolve");
    }
    let past = u16::try_from(len).expect("a table length fits u16");
    assert!(
        lookup(past).is_none(),
        "id {past} is past the table and must be refused"
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_gap_convention_and_the_game_scale_are_the_stated_numbers() {
        assert_eq!(
            GAP_STEPS_PER_CELL, 128,
            "1/128 of a cell: eight fine cells at rung 0"
        );
        assert_eq!(GAP_CONVENTION_TAG, 1);
        assert_eq!(GameScale::SHIPPED.fragment_surface_m2_per_m3, 200);
        assert_eq!(GameScale::SHIPPED.toughness_base, 268);
        assert_eq!(
            GameScale::SHIPPED.toughness_compress,
            Compression::SquareRoot
        );
    }

    #[test]
    fn the_dense_witness_accepts_a_dense_table_and_refuses_a_hole() {
        assert_dense(3, |id| if id < 3 { Some(id) } else { None });
    }
}
