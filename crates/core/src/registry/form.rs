//! THE FORM TABLE — the shape a cell takes: the smooth terrain cell, the sub-grid of small blocks, the
//! landscape object, and the TWENTY square building shapes (ruling V7 S3-1: the half-cube, the
//! quarter post and the post belong to the sub-grid, so twenty exactly).
//!
//! Volumes are in units of the cell's `196 608`, from the base's catalogue
//! (`block_system_design.md` §4.3); every shape's volume is a multiple of the widest admitted
//! small-block divisor, so a small block at every admitted scale keeps a non-zero volume and weight
//! (the volume rule). Full faces use the grid's face numbering (`+X −X +Y −Y +Z −Z` as bits 0..=5).
//! The orientation orbit is the number of distinct placements under the 24 proper rotations, and the
//! legal-orientation mask says WHICH codes a record may carry (out of the digest: a bound on what a
//! player may place next).
//!
//! **Provisional columns, named.** The four connect families (wall, fence, railing, conduit) have a
//! volume that depends on which neighbours they connect to; the base says "per variant". The rows
//! carry one nominal volume, marked provisional, until the mesh bake (slice 12) states the rule. The
//! legal-orientation masks are provisional for every shape but the cube for the same reason: the
//! canonical set comes from the shape's own symmetry at the bake. The octant mask the pyramid
//! coarsens by is derived at the same bake and is not a column yet; adding it changes the digest,
//! which is free before the first world is saved (§3 rule 1).
//!
//! **Example.** A wedge is half a cell, full on its −Y and −X faces, one of twelve placements, one
//! convex hull. A shipwright places one on a bow at rotation code 17; the registry knows its faces and
//! its hull, and the rotation table turns both exactly.

/// A form's number: its index in [`FORMS`]. Dense, append-only, never reused. No `Default`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct ShapeId(pub u16);

/// A cell's volume in units.
pub const CELL_VOLUME_UNITS: u32 = 196_608;

/// The class of a form: what mechanism draws and collides it. APPEND ONLY — the identity digest folds
/// these numbers.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
pub enum FormClass {
    /// Nothing: the form of the Empty kind.
    Void = 0,
    /// A smooth terrain cell: substance plus the gap byte, meshed by the extractor.
    Terrain = 1,
    /// A cell holding small blocks on its sub-lattice.
    SubGrid = 2,
    /// A landscape object anchored at this cell (a tree), expanded from its parameters.
    Object = 3,
    /// One of the square building shapes.
    Shape = 4,
}

/// How a form collides. APPEND ONLY — the identity digest folds these numbers.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
pub enum ColliderClass {
    /// No collider at all.
    None = 0,
    /// The smooth extractor's surface.
    Smooth = 1,
    /// The union of the small blocks' boxes.
    Union = 2,
    /// The object's skeleton, expanded.
    Skeleton = 3,
    /// One axis-aligned box.
    Box = 4,
    /// One convex hull.
    Convex = 5,
    /// A compound of convex parts.
    Compound = 6,
}

/// One form.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct FormDef {
    /// The identity, folded into the digest. Never changes once the row exists.
    pub key: &'static str,
    /// The display word. May change freely.
    pub name: &'static str,
    pub class: FormClass,
    /// Volume in units of [`CELL_VOLUME_UNITS`]; 0 where the volume comes from elsewhere (a void, a
    /// sub-grid's members, an object's skeleton). A terrain cell is a full cell, and its gap byte says
    /// how much of it is rock.
    pub volume_units: u32,
    /// Whether the volume is a nominal stand-in until the mesh bake states the real rule.
    pub volume_provisional: bool,
    /// Which of the six faces the form covers completely (`+X −X +Y −Y +Z −Z` as bits 0..=5).
    pub full_faces: u8,
    /// Distinct placements under the 24 proper rotations.
    pub orientations: u8,
    /// Which orientation codes a record may carry: bit `c` for code `c`. Out of the digest.
    pub legal_orientations: u32,
    pub collider: ColliderClass,
}

/// Every one of the 24 codes.
pub const ALL_ORIENTATIONS: u32 = (1 << 24) - 1;
/// Only the identity code.
pub const IDENTITY_ONLY: u32 = 1;

#[allow(clippy::too_many_arguments)]
const fn row(
    key: &'static str,
    class: FormClass,
    volume_units: u32,
    volume_provisional: bool,
    full_faces: u8,
    orientations: u8,
    legal_orientations: u32,
    collider: ColliderClass,
) -> FormDef {
    FormDef {
        key,
        name: key,
        class,
        volume_units,
        volume_provisional,
        full_faces,
        orientations,
        legal_orientations,
        collider,
    }
}

const PX: u8 = 1 << 0;
const NX: u8 = 1 << 1;
const PY: u8 = 1 << 2;
const NY: u8 = 1 << 3;
const PZ: u8 = 1 << 4;
const NZ: u8 = 1 << 5;
const ALL_FACES: u8 = PX | NX | PY | NY | PZ | NZ;

use ColliderClass as C;
use FormClass as K;

/// THE FORM TABLE. The index is the id. APPEND ONLY once the first world is saved.
pub const FORMS: [FormDef; 24] = [
    // 0..=3: the non-shape forms. Their orientation is always the identity.
    row("void", K::Void, 0, false, 0, 1, IDENTITY_ONLY, C::None),
    row(
        "terrain",
        K::Terrain,
        CELL_VOLUME_UNITS,
        false,
        0,
        1,
        IDENTITY_ONLY,
        C::Smooth,
    ),
    row(
        "sub-grid",
        K::SubGrid,
        0,
        false,
        0,
        1,
        IDENTITY_ONLY,
        C::Union,
    ),
    row(
        "object",
        K::Object,
        0,
        false,
        0,
        1,
        IDENTITY_ONLY,
        C::Skeleton,
    ),
    // 4..=23: the twenty shapes, in the base's order with the three small boxes removed. The legal
    // masks are provisional (every code) until the bake states each shape's canonical set; the cube's
    // is exact, because every rotation of a cube is the cube.
    row(
        "cube",
        K::Shape,
        CELL_VOLUME_UNITS,
        false,
        ALL_FACES,
        1,
        IDENTITY_ONLY,
        C::Box,
    ),
    row(
        "slab",
        K::Shape,
        98_304,
        false,
        NY,
        6,
        ALL_ORIENTATIONS,
        C::Convex,
    ),
    row(
        "plate",
        K::Shape,
        49_152,
        false,
        NY,
        6,
        ALL_ORIENTATIONS,
        C::Convex,
    ),
    row(
        "panel",
        K::Shape,
        24_576,
        false,
        NY,
        6,
        ALL_ORIENTATIONS,
        C::Convex,
    ),
    row(
        "wedge",
        K::Shape,
        98_304,
        false,
        NY | NX,
        12,
        ALL_ORIENTATIONS,
        C::Convex,
    ),
    row(
        "ramp-low",
        K::Shape,
        49_152,
        false,
        NY,
        24,
        ALL_ORIENTATIONS,
        C::Convex,
    ),
    row(
        "ramp-high",
        K::Shape,
        147_456,
        false,
        NY | NX,
        24,
        ALL_ORIENTATIONS,
        C::Convex,
    ),
    row(
        "corner-out",
        K::Shape,
        65_536,
        false,
        NY,
        24,
        ALL_ORIENTATIONS,
        C::Convex,
    ),
    row(
        "corner-in",
        K::Shape,
        131_072,
        false,
        NY | NX | NZ,
        24,
        ALL_ORIENTATIONS,
        C::Compound,
    ),
    row(
        "ramp-low-out",
        K::Shape,
        32_768,
        false,
        NY,
        24,
        ALL_ORIENTATIONS,
        C::Convex,
    ),
    row(
        "ramp-low-in",
        K::Shape,
        65_536,
        false,
        NY,
        24,
        ALL_ORIENTATIONS,
        C::Compound,
    ),
    row(
        "ramp-high-out",
        K::Shape,
        131_072,
        false,
        NY,
        24,
        ALL_ORIENTATIONS,
        C::Convex,
    ),
    row(
        "ramp-high-in",
        K::Shape,
        163_840,
        false,
        NY | PX | PZ,
        24,
        ALL_ORIENTATIONS,
        C::Compound,
    ),
    row(
        "tetra",
        K::Shape,
        32_768,
        false,
        0,
        8,
        ALL_ORIENTATIONS,
        C::Convex,
    ),
    row(
        "tetra-in",
        K::Shape,
        163_840,
        false,
        PX | PY | PZ,
        8,
        ALL_ORIENTATIONS,
        C::Compound,
    ),
    // Stairs: the base says 24 codes, 8 legal; the eight are the bake's to name (provisional mask).
    row(
        "stairs",
        K::Shape,
        147_456,
        false,
        NY | NX,
        24,
        ALL_ORIENTATIONS,
        C::Compound,
    ),
    // The connect families: the base says "volume per variant"; one nominal volume, provisional.
    row(
        "wall",
        K::Shape,
        49_152,
        true,
        0,
        12,
        ALL_ORIENTATIONS,
        C::Compound,
    ),
    row(
        "fence",
        K::Shape,
        24_576,
        true,
        0,
        12,
        ALL_ORIENTATIONS,
        C::Compound,
    ),
    row(
        "railing",
        K::Shape,
        24_576,
        true,
        0,
        12,
        ALL_ORIENTATIONS,
        C::Compound,
    ),
    row(
        "conduit",
        K::Shape,
        49_152,
        true,
        0,
        12,
        ALL_ORIENTATIONS,
        C::Compound,
    ),
];

impl ShapeId {
    /// The form of the Empty kind.
    pub const VOID: ShapeId = ShapeId(0);
    /// The smooth terrain cell.
    pub const TERRAIN: ShapeId = ShapeId(1);
    /// A cell of small blocks.
    pub const SUB_GRID: ShapeId = ShapeId(2);
    /// A landscape object.
    pub const OBJECT: ShapeId = ShapeId(3);
    /// The first shape, the cube — the only shape placeable before slice 12.
    pub const CUBE: ShapeId = ShapeId(4);
    /// The number of square shapes.
    pub const SHAPE_COUNT: usize = 20;

    /// This form's row; `None` for a number the table does not hold (refuse, never default).
    #[must_use]
    pub fn def(self) -> Option<&'static FormDef> {
        FORMS.get(usize::from(self.0))
    }

    /// Whether this form is one of the square shapes.
    #[must_use]
    pub fn is_shape(self) -> bool {
        matches!(self.def(), Some(f) if f.class == FormClass::Shape)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The row builder runs at compile time, which no coverage tool sees; run it here too.
    #[test]
    fn the_row_builder_runs_at_runtime_and_matches_the_table() {
        assert_eq!(
            row(
                "cube",
                K::Shape,
                CELL_VOLUME_UNITS,
                false,
                ALL_FACES,
                1,
                IDENTITY_ONLY,
                C::Box
            ),
            FORMS[usize::from(ShapeId::CUBE.0)]
        );
        assert_eq!(
            row("void", K::Void, 0, false, 0, 1, IDENTITY_ONLY, C::None),
            FORMS[0]
        );
    }

    #[test]
    fn the_table_is_dense_and_holds_exactly_twenty_shapes() {
        crate::registry::assert_dense(FORMS.len(), |id| ShapeId(id).def());
        let shapes = FORMS.iter().filter(|f| f.class == FormClass::Shape).count();
        assert_eq!(
            shapes,
            ShapeId::SHAPE_COUNT,
            "twenty shapes exactly (ruling V7)"
        );
        assert_eq!(ShapeId::CUBE.def().map(|f| f.key), Some("cube"));
        assert!(ShapeId::CUBE.is_shape());
        assert!(!ShapeId::TERRAIN.is_shape());
        assert!(!ShapeId(99).is_shape(), "an unknown form is not a shape");
        let mut keys = std::collections::BTreeSet::new();
        for f in &FORMS {
            assert!(keys.insert(f.key), "duplicate form key {}", f.key);
        }
        for dropped in ["quarter", "post", "nub"] {
            assert!(!keys.contains(dropped), "{dropped} belongs to the sub-grid");
        }
        // The enum numbers the digest folds are the stated ones.
        assert_eq!((FormClass::Void as u8, FormClass::Shape as u8), (0, 4));
        assert_eq!(
            (ColliderClass::None as u8, ColliderClass::Compound as u8),
            (0, 6)
        );
    }

    #[test]
    fn every_shape_volume_divides_by_the_widest_admitted_scale_and_fits_the_cell() {
        // The volume rule: the divisor is DERIVED from the widest scale the kinds admit (1/8 m → 512),
        // so the rule and the mask can never disagree.
        let widest = crate::registry::kind::WIDEST_SCALE;
        let divisor = 1u32 << (3 * widest);
        assert_eq!(divisor, 512);
        for f in FORMS.iter().filter(|f| f.class == FormClass::Shape) {
            assert_eq!(f.volume_units % divisor, 0, "{} volume", f.key);
            assert!(f.volume_units > 0, "{} has volume", f.key);
            assert!(
                f.volume_units <= CELL_VOLUME_UNITS,
                "{} fits the cell",
                f.key
            );
        }
        assert_eq!(
            ShapeId::CUBE.def().map(|f| f.volume_units),
            Some(CELL_VOLUME_UNITS)
        );
        let provisional: Vec<&str> = FORMS
            .iter()
            .filter(|f| f.volume_provisional)
            .map(|f| f.key)
            .collect();
        assert_eq!(provisional, ["wall", "fence", "railing", "conduit"]);
    }

    #[test]
    fn full_faces_are_within_six_bits_orbits_divide_twenty_four_and_masks_are_within_the_table() {
        for f in &FORMS {
            assert_eq!(f.full_faces & !ALL_FACES, 0, "{} faces", f.key);
            assert_eq!(24 % u32::from(f.orientations), 0, "{} orbit", f.key);
            assert_eq!(
                f.legal_orientations & !ALL_ORIENTATIONS,
                0,
                "{} mask",
                f.key
            );
            assert_ne!(f.legal_orientations & 1, 0, "{} admits the identity", f.key);
            let non_shape_identity_only = f.class != FormClass::Shape;
            if non_shape_identity_only {
                assert_eq!(f.legal_orientations, IDENTITY_ONLY, "{} turns not", f.key);
            }
        }
        let cube = ShapeId::CUBE.def().expect("cube");
        assert_eq!(
            (
                cube.full_faces,
                cube.orientations,
                cube.collider,
                cube.legal_orientations
            ),
            (ALL_FACES, 1, ColliderClass::Box, IDENTITY_ONLY)
        );
        let convex = FORMS
            .iter()
            .filter(|f| f.collider == ColliderClass::Convex)
            .count();
        let compound = FORMS
            .iter()
            .filter(|f| f.collider == ColliderClass::Compound)
            .count();
        assert_eq!(
            (convex, compound),
            (10, 9),
            "one box, ten hulls, nine compounds"
        );
    }
}
