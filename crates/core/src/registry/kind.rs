//! THE BLOCK KIND TABLE — one row per number a saved record can name: a substance, a form, a function,
//! the flags, the style count, the small-block scales it admits, and a replacement pointer for a
//! retired row (ruling V7 S3-3).
//!
//! **The numbering is ONE ordered recipe, appended at the end only.** The frozen first batch is built
//! at compile time from three groups written on 2026-09-07 — the Empty kind, the terrain substances,
//! the building substances in every shape, the two holders — and those groups are FROZEN: a new
//! substance, a new shape or a new function never touches them. Every later entry goes into
//! [`APPENDED`], after the frozen batch, so no saved number ever moves. A pinned digest of the whole
//! recipe makes any reorder fail the build before a store can see it.
//!
//! Mass and integrity are DERIVED here from the substance's cited facts and the form's volume, through
//! the game's few named constants (`block_system_design.md` §2.2.1), and reproduce the design's own
//! worked table exactly. Nobody types a weight.
//!
//! **Example.** A granite cube weighs 2 700 kg and holds 1 198 integrity points. A half-scale titanium
//! wedge weighs 277 kg. A saved record names each by one number, and the number never changes.

use super::form::{CELL_VOLUME_UNITS, FORMS, FormClass, ShapeId};
use super::function::FunctionId;
use super::rotation::Orientation;
use super::substance::{Family, MatterState, SUBSTANCES, SubstanceId};
use super::{Compression, GameScale};

/// A block kind's number: its index in [`KINDS`]. Dense, append-only, never reused. No `Default`: a
/// zero is [`BlockKindId::EMPTY`], named.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct BlockKindId(pub u16);

/// A kind's flags.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct KindFlags(pub u8);

impl KindFlags {
    /// The row is retired: saved records still decode and draw; new placements are refused.
    pub const RETIRED: KindFlags = KindFlags(1 << 0);
    /// The kind is placeable in the current build (out of the digest: a bound on what a player may
    /// place next). Before slice 12 only the cube shape and the solid terrain cells are.
    pub const PLACEABLE: KindFlags = KindFlags(1 << 1);

    #[must_use]
    pub const fn has(self, flag: KindFlags) -> bool {
        self.0 & flag.0 != 0
    }
}

/// One block kind.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct KindDef {
    pub substance: SubstanceId,
    pub form: ShapeId,
    pub function: FunctionId,
    pub flags: KindFlags,
    /// How many styles a record may name: a record's variant must be below this.
    pub variants: u8,
    /// Which small-block scales the kind admits, bit `s − 1` for scale `1/2^s`. The record carries two
    /// bits of scale (ruling B: 0 = whole, 1 = half, 2 = quarter, 3 = eighth), so bits 0..=2 are the
    /// admissible set; bit 3 is reserved for a fifth scale the owner may open after the frame-cost
    /// measurement (D-2).
    pub sub_scales: u8,
    /// For a retired row, the kind that replaces it.
    pub replaced_by: Option<BlockKindId>,
}

/// The widest small-block scale a record can carry today: 3, one eighth of a metre.
pub const WIDEST_SCALE: u8 = 3;
/// Every scale a record can carry.
const ALL_SCALES: u8 = (1 << WIDEST_SCALE) - 1;

/// Why a kind lookup or a record's fields were refused.
#[derive(Clone, Copy, Debug, PartialEq, Eq, thiserror::Error)]
pub enum RegistryError {
    #[error("unknown block kind {0}")]
    UnknownKind(u16),
    #[error("variant {variant} is not below kind {kind}'s count of {count}")]
    VariantOutOfRange { kind: u16, variant: u8, count: u8 },
    #[error("kind {0} is retired and may not be placed")]
    Retired(u16),
    #[error("kind {0} is not placeable in this build")]
    NotPlaceable(u16),
    #[error("scale {scale} is not admitted by kind {kind}")]
    ScaleNotAdmitted { kind: u16, scale: u8 },
    #[error("orientation code {code} is not legal for kind {kind}")]
    OrientationNotLegal { kind: u16, code: u8 },
}

// ★ THE FROZEN FIRST BATCH (2026-09-07). These three groups are never edited again: a new substance,
// shape or function is appended to `APPENDED`, after them.
const FROZEN_TERRAIN: [u16; 40] = [
    2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, // bedrock
    13, 14, 15, 16, 17, 18, 19, 20, 21, // loose ground
    22, 23, 24, 25, 26, 27, 28, 29, // surface and water
    30, 31, 32, 33, 34, 35, 36, 37, 38, 39, // ores and minerals
    45, 46, // moss, fungus
];
const FROZEN_BUILDING: [u16; 24] = [
    40, 41, 42, 43, // woods and planks
    47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, // building
    44, // bark
    2, 3, 7, 6, 10,
    9, // the mason's stones: granite, basalt, sandstone, limestone, marble, slate
];
const FROZEN_COUNT: usize =
    1 + FROZEN_TERRAIN.len() + FROZEN_BUILDING.len() * ShapeId::SHAPE_COUNT + 2;

/// Every kind added after the frozen batch, in the order it was added: `(substance, form, function)`.
/// APPEND AT THE END ONLY.
pub const APPENDED: [(u16, u16, u16); 0] = [];

/// The table's length.
pub const KIND_COUNT: usize = FROZEN_COUNT + APPENDED.len();

// The registry is capped by the record's sixteen-bit kind field.
const _: () = assert!(KIND_COUNT <= u16::MAX as usize);

/// Whether a substance may be placed by hand as a terrain cell: solid or loose matter, never a
/// liquid, a gas, a void, or an ore (ruling 15.3: ore is live state, placed by the world).
const fn terrain_placeable(substance: u16) -> bool {
    let s = &SUBSTANCES[substance as usize];
    matches!(s.state, MatterState::Solid | MatterState::Loose) && !matches!(s.family, Family::Ore)
}

const fn entry(substance: SubstanceId, form: ShapeId, placeable: bool, sub_scales: u8) -> KindDef {
    KindDef {
        substance,
        form,
        function: FunctionId::NONE,
        flags: KindFlags(if placeable { KindFlags::PLACEABLE.0 } else { 0 }),
        variants: 1,
        sub_scales,
        replaced_by: None,
    }
}

const fn build() -> [KindDef; KIND_COUNT] {
    let mut out = [entry(SubstanceId::VOID, ShapeId::VOID, true, 0); KIND_COUNT];
    let mut n = 0;
    // 0: the Empty kind — the removal. Placing Empty IS the removal path. Never Air (ruling V4).
    out[n] = entry(SubstanceId::VOID, ShapeId::VOID, true, 0);
    n += 1;
    // The terrain cells.
    let mut t = 0;
    while t < FROZEN_TERRAIN.len() {
        let s = FROZEN_TERRAIN[t];
        out[n] = entry(SubstanceId(s), ShapeId::TERRAIN, terrain_placeable(s), 0);
        n += 1;
        t += 1;
    }
    // The building shapes: every building substance in every shape, cube first (the shape order).
    let mut b = 0;
    while b < FROZEN_BUILDING.len() {
        let mut s = 0;
        while s < ShapeId::SHAPE_COUNT {
            let form = ShapeId(ShapeId::CUBE.0 + s as u16);
            out[n] = entry(SubstanceId(FROZEN_BUILDING[b]), form, s == 0, ALL_SCALES);
            n += 1;
            s += 1;
        }
        b += 1;
    }
    // The landscape object holder and the sub-grid holder: their substance is void; the record's
    // object byte and the sub-lattice say the rest.
    out[n] = entry(SubstanceId::VOID, ShapeId::OBJECT, false, 0);
    n += 1;
    out[n] = entry(SubstanceId::VOID, ShapeId::SUB_GRID, false, 0);
    n += 1;
    // Everything appended after the frozen batch, in order.
    n = append_into(&mut out, n, &APPENDED);
    let _ = n;
    out
}

/// Write `appended` into `out` from `start`, in order, one row per triple; the next free index.
const fn append_into(out: &mut [KindDef], start: usize, appended: &[(u16, u16, u16)]) -> usize {
    let mut n = start;
    let mut a = 0;
    while a < appended.len() {
        let (s, f, func) = appended[a];
        out[n] = KindDef {
            substance: SubstanceId(s),
            form: ShapeId(f),
            function: FunctionId(func),
            flags: KindFlags(0),
            variants: 1,
            sub_scales: 0,
            replaced_by: None,
        };
        n += 1;
        a += 1;
    }
    n
}

/// Every row's substance, form and function name a row of its table. Checked at compile time, so the
/// lookups below index directly and have no unreachable refusal arm.
const fn recipe_is_sound(table: &[KindDef; KIND_COUNT]) -> bool {
    let mut i = 0;
    while i < KIND_COUNT {
        let k = &table[i];
        if k.substance.0 as usize >= SUBSTANCES.len()
            || k.form.0 as usize >= FORMS.len()
            || k.function.0 as usize >= super::function::FUNCTIONS.len()
        {
            return false;
        }
        i += 1;
    }
    true
}

/// THE BLOCK KIND TABLE. The index is the id. APPEND ONLY.
pub const KINDS: [KindDef; KIND_COUNT] = build();
const _: () = assert!(recipe_is_sound(&KINDS));

impl BlockKindId {
    /// The removal kind.
    pub const EMPTY: BlockKindId = BlockKindId(0);

    /// This kind's row, or a typed refusal for a number the table does not hold.
    pub fn def(self) -> Result<&'static KindDef, RegistryError> {
        KINDS
            .get(usize::from(self.0))
            .ok_or(RegistryError::UnknownKind(self.0))
    }

    /// The decode rule for a record's fields: the kind exists, the variant is below its count, the
    /// small-block scale (0 = whole cell) is admitted, and the orientation code is legal for the form.
    pub fn check(
        self,
        variant: u8,
        scale: u8,
        orientation: Orientation,
    ) -> Result<&'static KindDef, RegistryError> {
        let def = self.def()?;
        if variant >= def.variants {
            return Err(RegistryError::VariantOutOfRange {
                kind: self.0,
                variant,
                count: def.variants,
            });
        }
        if scale != 0 && (scale > WIDEST_SCALE || def.sub_scales & (1 << (scale - 1)) == 0) {
            return Err(RegistryError::ScaleNotAdmitted {
                kind: self.0,
                scale,
            });
        }
        let legal = FORMS[usize::from(def.form.0)].legal_orientations;
        if legal & (1u32 << orientation.code()) == 0 {
            return Err(RegistryError::OrientationNotLegal {
                kind: self.0,
                code: orientation.code(),
            });
        }
        Ok(def)
    }

    /// The placement rule: the decode rule plus "not retired" and "placeable in this build".
    pub fn check_placement(
        self,
        variant: u8,
        scale: u8,
        orientation: Orientation,
    ) -> Result<&'static KindDef, RegistryError> {
        let def = self.check(variant, scale, orientation)?;
        placement_allowed(self.0, def)?;
        Ok(def)
    }

    /// The mass of one whole cell of this kind, in grams: density × the form's volume. A terrain cell
    /// is counted as a full cell; its gap byte scales it at the mesh. Zero for a void or a gas.
    pub fn mass_g(self) -> Result<u64, RegistryError> {
        self.mass_g_at(0)
    }

    /// The mass of one block of this kind at small-block `scale` (0 = whole cell), in grams: the
    /// whole-cell mass divided by eight per scale step.
    pub fn mass_g_at(self, scale: u8) -> Result<u64, RegistryError> {
        let def = self.def()?;
        let substance = &SUBSTANCES[usize::from(def.substance.0)];
        let form = &FORMS[usize::from(def.form.0)];
        let whole = u64::from(substance.density_g_m3) * u64::from(form.volume_units)
            / u64::from(CELL_VOLUME_UNITS);
        Ok(whole >> (3 * u32::from(scale)))
    }

    /// The integrity of one whole cell of this kind, in integrity points, derived from the substance's
    /// work of fracture through the game scale (`block_system_design.md` §2.2.1):
    /// `fracture_work_kj = w × surface / 1000`, `toughness_dp_m3 = base × √fracture_work_kj`,
    /// `integrity = toughness × volume`, every step in fixed point so the design's worked table
    /// reproduces exactly (granite 1 198, packed soil 169, ice 207, glass 379). Zero for a void, a gas
    /// or a liquid.
    pub fn integrity_dp(self, scale: GameScale) -> Result<u64, RegistryError> {
        let def = self.def()?;
        let substance = &SUBSTANCES[usize::from(def.substance.0)];
        let form = &FORMS[usize::from(def.form.0)];
        if !matches!(substance.state, MatterState::Solid | MatterState::Loose) {
            return Ok(0);
        }
        // The square root of the fracture work in kJ/m³, at 2^20 fixed point, with the kilojoule
        // divide INSIDE the root so a soft substance never floors to zero.
        let joules = u128::from(substance.work_of_fracture_j_m2)
            * u128::from(scale.fragment_surface_m2_per_m3);
        let root_q20 = match scale.toughness_compress {
            Compression::SquareRoot => (joules << 40) / 1_000,
        }
        .isqrt();
        let toughness_dp_m3 = (u128::from(scale.toughness_base) * root_q20) >> 20;
        let dp = toughness_dp_m3 * u128::from(form.volume_units) / u128::from(CELL_VOLUME_UNITS);
        Ok(u64::try_from(dp).unwrap_or(u64::MAX))
    }
}

/// The placement half of the rule, on a row: not retired, and placeable in this build. A function of
/// the row so a retired row can be driven before any row in the table is retired.
pub fn placement_allowed(kind: u16, def: &KindDef) -> Result<(), RegistryError> {
    if def.flags.has(KindFlags::RETIRED) {
        return Err(RegistryError::Retired(kind));
    }
    if !def.flags.has(KindFlags::PLACEABLE) {
        return Err(RegistryError::NotPlaceable(kind));
    }
    Ok(())
}

/// Whether a kind's form is a terrain cell (a helper for the record slice).
#[must_use]
pub fn is_terrain(def: &KindDef) -> bool {
    matches!(def.form.def(), Some(f) if f.class == FormClass::Terrain)
}

/// The digest of the whole recipe — every row's triple in table order — the committed pin of the
/// numbering. A reorder, an insertion or a change of any row's triple moves it.
#[must_use]
pub fn recipe_digest() -> u64 {
    let mut acc = crate::digest::FNV_OFFSET;
    for k in &KINDS {
        acc = crate::digest::fnv1a(acc, &k.substance.0.to_le_bytes());
        acc = crate::digest::fnv1a(acc, &k.form.0.to_le_bytes());
        acc = crate::digest::fnv1a(acc, &k.function.0.to_le_bytes());
    }
    acc
}

#[cfg(test)]
mod tests {
    use super::*;

    fn find(substance: &str, form: &str) -> BlockKindId {
        let s = SUBSTANCES
            .iter()
            .position(|d| d.key == substance)
            .expect("substance") as u16;
        let f = FORMS.iter().position(|d| d.key == form).expect("form") as u16;
        let i = KINDS
            .iter()
            .position(|k| k.substance == SubstanceId(s) && k.form == ShapeId(f))
            .expect("kind");
        BlockKindId(i as u16)
    }

    const ID: Orientation = Orientation::IDENTITY;

    #[test]
    fn the_table_is_dense_every_row_resolves_and_the_recipe_is_pinned() {
        assert_eq!(KINDS.len(), 1 + 40 + 24 * 20 + 2);
        assert_eq!(
            build(),
            KINDS,
            "the compile-time table equals the runtime build"
        );
        assert!(recipe_is_sound(&KINDS));
        let mut broken = KINDS;
        broken[3].form = ShapeId(200);
        assert!(
            !recipe_is_sound(&broken),
            "a row naming a form past the table is unsound"
        );
        let mut broken = KINDS;
        broken[3].substance = SubstanceId(200);
        assert!(!recipe_is_sound(&broken), "a substance past the table");
        let mut broken = KINDS;
        broken[3].function = FunctionId(200);
        assert!(!recipe_is_sound(&broken), "a function past the table");
        // The append step, driven with rows, because the shipped list is still empty.
        let mut out = [KINDS[0]; 3];
        let next = append_into(&mut out, 1, &[(2, 4, 0), (3, 5, 0)]);
        assert_eq!(next, 3);
        assert_eq!(out[1].substance, SubstanceId(2));
        assert_eq!(out[2].form, ShapeId(5));
        assert_eq!(out[1].function, FunctionId::NONE);
        assert_eq!(
            append_into(&mut out, 1, &[]),
            1,
            "an empty list appends nothing"
        );
        for (i, k) in KINDS.iter().enumerate() {
            let id = BlockKindId(i as u16);
            assert_eq!(id.def(), Ok(k));
            assert!(k.variants >= 1, "kind {i} has at least one style");
            assert_eq!(k.replaced_by, None, "nothing is retired yet");
        }
        assert_eq!(
            BlockKindId(KIND_COUNT as u16).def(),
            Err(RegistryError::UnknownKind(KIND_COUNT as u16))
        );
        assert_eq!(
            BlockKindId::EMPTY.def().map(|k| (k.substance, k.form)),
            Ok((SubstanceId::VOID, ShapeId::VOID))
        );
        // Every (substance, form, function) triple is unique.
        let mut triples = std::collections::BTreeSet::new();
        for k in &KINDS {
            assert!(
                triples.insert((k.substance, k.form, k.function)),
                "duplicate triple"
            );
        }
        // THE PIN: the numbering of every row, frozen. Granite terrain is kind 1; the oak cube is 41.
        assert_eq!(find("granite", "terrain"), BlockKindId(1));
        assert_eq!(find("oak wood", "cube"), BlockKindId(41));
        assert_eq!(
            recipe_digest(),
            RECIPE_DIGEST,
            "the kind numbering moved: every saved record re-points"
        );
        assert!(is_terrain(find("granite", "terrain").def().expect("kind")));
        assert!(!is_terrain(&KINDS[0]));
        assert!(!is_terrain(&KindDef {
            form: ShapeId(200),
            ..KINDS[0]
        }));
        assert!(terrain_placeable(2), "granite may be placed");
        assert!(!terrain_placeable(25), "water may not");
        assert!(
            !terrain_placeable(30),
            "iron ore may not: the world places ore"
        );
    }

    #[test]
    fn placeability_follows_a_stated_rule_never_a_wholesale_grant() {
        for (i, k) in KINDS.iter().enumerate() {
            let s = &SUBSTANCES[usize::from(k.substance.0)];
            let f = &FORMS[usize::from(k.form.0)];
            let placeable = k.flags.has(KindFlags::PLACEABLE);
            match f.class {
                FormClass::Void => assert!(placeable, "Empty is the removal, kind {i}"),
                FormClass::Terrain => {
                    let expected = matches!(s.state, MatterState::Solid | MatterState::Loose)
                        && s.family != Family::Ore;
                    assert_eq!(placeable, expected, "terrain kind {i} ({})", s.key);
                }
                FormClass::Shape => assert_eq!(placeable, f.key == "cube", "shape kind {i}"),
                FormClass::SubGrid | FormClass::Object => assert!(!placeable, "holder {i}"),
            }
        }
        assert!(
            !find("water", "terrain")
                .def()
                .expect("k")
                .flags
                .has(KindFlags::PLACEABLE)
        );
        assert!(
            !find("lava", "terrain")
                .def()
                .expect("k")
                .flags
                .has(KindFlags::PLACEABLE)
        );
        assert!(
            !find("gold ore", "terrain")
                .def()
                .expect("k")
                .flags
                .has(KindFlags::PLACEABLE)
        );
        assert!(
            find("dirt", "terrain")
                .def()
                .expect("k")
                .flags
                .has(KindFlags::PLACEABLE)
        );
        assert!(
            find("granite", "cube")
                .def()
                .expect("k")
                .flags
                .has(KindFlags::PLACEABLE),
            "a mason lays granite"
        );
    }

    #[test]
    fn the_decode_and_placement_rules_refuse_by_name() {
        let steel_cube = find("structural steel", "cube");
        let steel_wedge = find("structural steel", "wedge");
        assert!(steel_cube.check(0, 0, ID).is_ok());
        assert!(
            steel_cube.check(0, 3, ID).is_ok(),
            "an eighth-scale steel block is admitted"
        );
        assert_eq!(
            steel_cube.check(1, 0, ID),
            Err(RegistryError::VariantOutOfRange {
                kind: steel_cube.0,
                variant: 1,
                count: 1
            })
        );
        assert_eq!(
            steel_cube.check(0, 4, ID),
            Err(RegistryError::ScaleNotAdmitted {
                kind: steel_cube.0,
                scale: 4
            }),
            "a sixteenth is reserved until the owner opens D-2"
        );
        let granite = find("granite", "terrain");
        assert_eq!(
            granite.check(0, 1, ID),
            Err(RegistryError::ScaleNotAdmitted {
                kind: granite.0,
                scale: 1
            }),
            "a terrain cell admits no small block"
        );
        let turned = Orientation::new(17).expect("code");
        assert_eq!(
            granite.check(0, 0, turned),
            Err(RegistryError::OrientationNotLegal {
                kind: granite.0,
                code: 17
            }),
            "a terrain cell never turns"
        );
        assert_eq!(
            steel_cube.check(0, 0, turned),
            Err(RegistryError::OrientationNotLegal {
                kind: steel_cube.0,
                code: 17
            }),
            "every rotation of a cube is the cube, so only code 0 is legal"
        );
        assert!(steel_wedge.check(0, 0, turned).is_ok(), "a wedge turns");
        assert!(steel_cube.check_placement(0, 0, ID).is_ok());
        assert_eq!(
            steel_wedge.check_placement(0, 0, ID),
            Err(RegistryError::NotPlaceable(steel_wedge.0)),
            "every shape but the cube waits for slice 12"
        );
        assert_eq!(
            BlockKindId(9_999).check_placement(0, 0, ID),
            Err(RegistryError::UnknownKind(9_999))
        );
        let mut retired = *steel_cube.def().expect("kind");
        retired.flags = KindFlags(KindFlags::RETIRED.0 | KindFlags::PLACEABLE.0);
        assert_eq!(
            placement_allowed(steel_cube.0, &retired),
            Err(RegistryError::Retired(steel_cube.0)),
            "a retired row refuses placement even though it is otherwise placeable"
        );
        let err = format!("{}", RegistryError::Retired(7));
        assert!(err.contains("retired"), "{err}");
    }

    #[test]
    fn mass_and_integrity_reproduce_the_designs_worked_table_exactly() {
        let dp = |s: &str, f: &str| find(s, f).integrity_dp(GameScale::SHIPPED).expect("dp");
        // block_system_design.md §2.2.1, the table with no per-material tuning at all.
        assert_eq!(dp("dirt", "terrain"), 169, "packed soil");
        assert_eq!(dp("ice", "terrain"), 207, "water ice");
        assert_eq!(dp("glass", "cube"), 379);
        assert_eq!(dp("granite", "terrain"), 1_198);
        assert_eq!(dp("concrete", "cube"), 1_312);
        assert_eq!(dp("basalt", "terrain"), 1_366);
        assert_eq!(dp("oak wood", "cube"), 11_985);
        assert_eq!(dp("structural steel", "cube"), 37_900);
        assert_eq!(dp("titanium alloy", "cube"), 41_518);
        // The soft substances the old kilojoule floor zeroed.
        assert!(dp("snow", "terrain") > 0);
        assert!(dp("sand", "terrain") > 0);
        assert_eq!(dp("water", "terrain"), 0, "a liquid has no integrity");
        assert_eq!(BlockKindId::EMPTY.integrity_dp(GameScale::SHIPPED), Ok(0));
        // Mass: the plan's own worked example.
        let granite = find("granite", "terrain");
        assert_eq!(
            granite.mass_g(),
            Ok(2_700_000),
            "a full granite cell is 2 700 kg"
        );
        assert_eq!(find("structural steel", "cube").mass_g(), Ok(7_850_000));
        let titanium_plate = find("titanium alloy", "plate");
        assert_eq!(
            titanium_plate.mass_g(),
            Ok(1_107_500),
            "a quarter of a cell of titanium"
        );
        let titanium_wedge = find("titanium alloy", "wedge");
        assert_eq!(
            titanium_wedge.mass_g_at(1),
            Ok(276_875),
            "a half-scale wedge: 277 kg"
        );
        assert_eq!(
            titanium_wedge.integrity_dp(GameScale::SHIPPED).expect("dp"),
            dp("titanium alloy", "cube") / 2,
            "half the cell, half the points"
        );
        assert_eq!(BlockKindId::EMPTY.mass_g(), Ok(0));
        assert_eq!(
            BlockKindId(9_999).mass_g(),
            Err(RegistryError::UnknownKind(9_999))
        );
        assert_eq!(
            BlockKindId(9_999).integrity_dp(GameScale::SHIPPED),
            Err(RegistryError::UnknownKind(9_999))
        );
    }

    /// THE COMMITTED PIN of the kind numbering.
    const RECIPE_DIGEST: u64 = 2_884_287_356_012_083_886;
}
