//! THE SUBSTANCE TABLE — what a cell is made of, with the cited physical facts every per-kind number
//! derives from (`block_system_design.md` §2.2.1; ruling V7: every substance the home planet can emit,
//! plus the builder's, append-only).
//!
//! Each row carries a KEY (its identity, folded into the digest, never changed) beside a NAME (the
//! display word, free to change), and a PROVENANCE: `Cited` names the design's table; `Provisional`
//! marks a value from a standard reference range that the geology topic (slice 5) firms before the
//! first world is saved; `Sentinel` marks a row whose numbers are not physics at all.
//!
//! **The ore convention.** An ore row is the ORE-BEARING ROCK at a typical grade — what a miner sees
//! and cuts — so its density and melting point are the host rock's, never the pure mineral's and never
//! the refined metal's. Coal is the rock itself.
//!
//! **Example.** Granite: 2 700 kg/m³, a work of fracture of 100 J/m². From those two numbers alone a
//! granite cube weighs 2 700 kg and holds 1 198 integrity points, and nobody typed either.

/// A substance's number: its index in [`SUBSTANCES`]. Dense, append-only, never reused. No `Default`:
/// a zero is named, never assumed.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct SubstanceId(pub u16);

/// What state of matter a substance is at ambient conditions.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MatterState {
    /// Nothing: the substance of the Empty kind.
    Void,
    Solid,
    /// Solid but loose: soil, sand, gravel, snow.
    Loose,
    Liquid,
    Gas,
}

/// The substance family: sound and particle sets, and the shatter policy, later.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Family {
    Void,
    Rock,
    Soil,
    Ice,
    Fluid,
    Ore,
    Wood,
    Plant,
    Masonry,
    Metal,
    Composite,
    Glass,
    Polymer,
    Atmosphere,
}

/// Where a row's numbers come from.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Provenance {
    /// The design's cited table, `block_system_design.md` §2.2.
    Cited,
    /// A standard reference range; firmed by the geology topic before the first world is saved.
    Provisional,
    /// Not physics: a sentinel row (the void).
    Sentinel,
}

/// One substance.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SubstanceDef {
    /// The identity, folded into the digest. Never changes once the row exists.
    pub key: &'static str,
    /// The display word. May change freely.
    pub name: &'static str,
    pub state: MatterState,
    pub family: Family,
    /// Density in grams per cubic metre, so titanium (4 430 000) fits a `u32` and air (1 200) is exact.
    pub density_g_m3: u32,
    /// The work of fracture in joules per square metre — the one number integrity derives from.
    pub work_of_fracture_j_m2: u32,
    /// The melting point in kelvin; 0 only where it does not apply: a void, a gas, and a substance
    /// that burns or decomposes before it melts (wood, plant matter, peat, coal, a composite).
    pub melting_k: u16,
    pub provenance: Provenance,
}

const fn row(
    key: &'static str,
    state: MatterState,
    family: Family,
    density_g_m3: u32,
    work_of_fracture_j_m2: u32,
    melting_k: u16,
    provenance: Provenance,
) -> SubstanceDef {
    SubstanceDef {
        key,
        name: key,
        state,
        family,
        density_g_m3,
        work_of_fracture_j_m2,
        melting_k,
        provenance,
    }
}

use Family as F;
use MatterState as S;
use Provenance::{Cited, Provisional, Sentinel};

/// THE SUBSTANCE TABLE. The index is the id. APPEND ONLY once the first world is saved.
pub const SUBSTANCES: [SubstanceDef; 60] = [
    // 0 — the substance of the Empty kind: nothing.
    row("void", S::Void, F::Void, 0, 0, 0, Sentinel),
    // 1 — the atmosphere. Never the removal kind (ruling V4, step 13).
    row("air", S::Gas, F::Atmosphere, 1_200, 0, 0, Provisional),
    // Bedrock, 2..=12.
    row("granite", S::Solid, F::Rock, 2_700_000, 100, 1_500, Cited),
    row("basalt", S::Solid, F::Rock, 2_900_000, 130, 1_450, Cited),
    row(
        "gabbro",
        S::Solid,
        F::Rock,
        3_000_000,
        120,
        1_450,
        Provisional,
    ),
    row(
        "andesite",
        S::Solid,
        F::Rock,
        2_600_000,
        100,
        1_400,
        Provisional,
    ),
    row(
        "limestone",
        S::Solid,
        F::Rock,
        2_500_000,
        60,
        1_100,
        Provisional,
    ),
    row(
        "sandstone",
        S::Solid,
        F::Rock,
        2_300_000,
        40,
        1_900,
        Provisional,
    ),
    row(
        "shale",
        S::Solid,
        F::Rock,
        2_400_000,
        30,
        1_500,
        Provisional,
    ),
    row(
        "slate",
        S::Solid,
        F::Rock,
        2_700_000,
        50,
        1_500,
        Provisional,
    ),
    row(
        "marble",
        S::Solid,
        F::Rock,
        2_700_000,
        60,
        1_100,
        Provisional,
    ),
    row(
        "quartzite",
        S::Solid,
        F::Rock,
        2_650_000,
        90,
        1_950,
        Provisional,
    ),
    row(
        "obsidian",
        S::Solid,
        F::Rock,
        2_400_000,
        10,
        1_300,
        Provisional,
    ),
    // Loose ground, 13..=21 (a soil is ground rock; it melts as rock does; peat burns).
    row("dirt", S::Loose, F::Soil, 1_500_000, 2, 1_500, Cited),
    row("loam", S::Loose, F::Soil, 1_400_000, 2, 1_500, Provisional),
    row("clay", S::Loose, F::Soil, 1_800_000, 5, 1_900, Provisional),
    row("sand", S::Loose, F::Soil, 1_600_000, 1, 1_950, Provisional),
    row(
        "gravel",
        S::Loose,
        F::Soil,
        1_700_000,
        1,
        1_500,
        Provisional,
    ),
    row("silt", S::Loose, F::Soil, 1_500_000, 1, 1_500, Provisional),
    row("mud", S::Loose, F::Soil, 1_700_000, 1, 1_500, Provisional),
    row("peat", S::Loose, F::Soil, 300_000, 3, 0, Provisional),
    row(
        "permafrost",
        S::Solid,
        F::Soil,
        1_800_000,
        20,
        273,
        Provisional,
    ),
    // Surface and water, 22..=29.
    row("snow", S::Loose, F::Ice, 300_000, 1, 273, Provisional),
    row("ice", S::Solid, F::Ice, 917_000, 3, 273, Cited),
    row("packed ice", S::Solid, F::Ice, 920_000, 3, 273, Provisional),
    row("water", S::Liquid, F::Fluid, 1_000_000, 0, 273, Provisional),
    row(
        "salt water",
        S::Liquid,
        F::Fluid,
        1_025_000,
        0,
        271,
        Provisional,
    ),
    row(
        "lava",
        S::Liquid,
        F::Fluid,
        2_700_000,
        0,
        1_400,
        Provisional,
    ),
    row("salt", S::Solid, F::Rock, 2_160_000, 5, 1_074, Provisional),
    row("ash", S::Loose, F::Soil, 700_000, 1, 1_400, Provisional),
    // Ores and minerals, 30..=39 — the ore-bearing ROCK at a typical grade, one convention for all.
    // Kinds in the registry; PLACED by live state (the seed ruling), never placeable by hand.
    row(
        "iron ore",
        S::Solid,
        F::Ore,
        3_500_000,
        80,
        1_500,
        Provisional,
    ),
    row(
        "copper ore",
        S::Solid,
        F::Ore,
        2_900_000,
        70,
        1_500,
        Provisional,
    ),
    row(
        "tin ore",
        S::Solid,
        F::Ore,
        2_800_000,
        60,
        1_500,
        Provisional,
    ),
    row("coal", S::Solid, F::Ore, 1_350_000, 20, 0, Provisional),
    row(
        "bauxite",
        S::Solid,
        F::Ore,
        2_500_000,
        40,
        2_300,
        Provisional,
    ),
    row(
        "gold ore",
        S::Solid,
        F::Ore,
        2_700_000,
        90,
        1_500,
        Provisional,
    ),
    row(
        "silver ore",
        S::Solid,
        F::Ore,
        2_800_000,
        90,
        1_500,
        Provisional,
    ),
    row(
        "uranium ore",
        S::Solid,
        F::Ore,
        2_800_000,
        90,
        1_500,
        Provisional,
    ),
    row("sulfur", S::Solid, F::Ore, 2_070_000, 5, 388, Provisional),
    row(
        "quartz",
        S::Solid,
        F::Ore,
        2_650_000,
        10,
        1_950,
        Provisional,
    ),
    // Grown and organic, as placed blocks, 40..=46 (they burn before they melt).
    row("oak wood", S::Solid, F::Wood, 700_000, 10_000, 0, Cited),
    row(
        "pine wood",
        S::Solid,
        F::Wood,
        500_000,
        6_000,
        0,
        Provisional,
    ),
    row(
        "birch wood",
        S::Solid,
        F::Wood,
        650_000,
        8_000,
        0,
        Provisional,
    ),
    row("planks", S::Solid, F::Wood, 600_000, 6_000, 0, Provisional),
    row("bark", S::Solid, F::Wood, 400_000, 1_000, 0, Provisional),
    row("moss", S::Solid, F::Plant, 300_000, 50, 0, Provisional),
    row("fungus", S::Solid, F::Plant, 400_000, 100, 0, Provisional),
    // Building, 47..=59.
    row(
        "brick",
        S::Solid,
        F::Masonry,
        1_900_000,
        80,
        1_900,
        Provisional,
    ),
    row(
        "concrete",
        S::Solid,
        F::Masonry,
        2_400_000,
        120,
        1_800,
        Cited,
    ),
    row(
        "mortar",
        S::Solid,
        F::Masonry,
        2_000_000,
        40,
        1_700,
        Provisional,
    ),
    row("glass", S::Solid, F::Glass, 2_500_000, 10, 1_700, Cited),
    row(
        "aluminium alloy",
        S::Solid,
        F::Metal,
        2_700_000,
        30_000,
        930,
        Provisional,
    ),
    row(
        "structural steel",
        S::Solid,
        F::Metal,
        7_850_000,
        100_000,
        1_700,
        Cited,
    ),
    row(
        "stainless steel",
        S::Solid,
        F::Metal,
        8_000_000,
        100_000,
        1_700,
        Provisional,
    ),
    row(
        "titanium alloy",
        S::Solid,
        F::Metal,
        4_430_000,
        120_000,
        1_900,
        Cited,
    ),
    row(
        "copper",
        S::Solid,
        F::Metal,
        8_960_000,
        60_000,
        1_358,
        Provisional,
    ),
    row(
        "carbon composite",
        S::Solid,
        F::Composite,
        1_600_000,
        5_000,
        0,
        Provisional,
    ),
    row(
        "ceramic",
        S::Solid,
        F::Masonry,
        3_000_000,
        50,
        2_300,
        Provisional,
    ),
    row(
        "polymer",
        S::Solid,
        F::Polymer,
        1_200_000,
        3_000,
        450,
        Provisional,
    ),
    row(
        "rubber",
        S::Solid,
        F::Polymer,
        1_100_000,
        10_000,
        0,
        Provisional,
    ),
];

/// The families whose members burn or decompose before they melt, so a melting point of 0 is right.
/// A thermoplastic polymer melts; vulcanised rubber, coal and peat decompose and are named singly.
#[cfg(test)]
const fn burns(family: Family) -> bool {
    matches!(family, F::Wood | F::Plant | F::Composite)
}

impl SubstanceId {
    /// The substance of the Empty kind.
    pub const VOID: SubstanceId = SubstanceId(0);
    /// The atmosphere.
    pub const AIR: SubstanceId = SubstanceId(1);

    /// This substance's row; `None` for a number the table does not hold (refuse, never default).
    #[must_use]
    pub fn def(self) -> Option<&'static SubstanceDef> {
        SUBSTANCES.get(usize::from(self.0))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The row builder runs at compile time, which no coverage tool sees; run it here too.
    #[test]
    fn the_row_builder_runs_at_runtime_and_matches_the_table() {
        assert_eq!(
            row("granite", S::Solid, F::Rock, 2_700_000, 100, 1_500, Cited),
            SUBSTANCES[2]
        );
        assert_eq!(
            row("void", S::Void, F::Void, 0, 0, 0, Sentinel),
            SUBSTANCES[0]
        );
        assert!(burns(F::Wood));
        assert!(!burns(F::Rock));
    }

    #[test]
    fn the_table_is_dense_and_every_key_is_unique() {
        crate::registry::assert_dense(SUBSTANCES.len(), |id| SubstanceId(id).def());
        let mut keys = std::collections::BTreeSet::new();
        for s in &SUBSTANCES {
            assert!(keys.insert(s.key), "duplicate substance key {}", s.key);
        }
        assert_eq!(SUBSTANCES.len(), 60);
        assert_eq!(SubstanceId::VOID.def().map(|s| s.key), Some("void"));
        assert_eq!(
            SubstanceId::AIR.def().map(|s| s.state),
            Some(MatterState::Gas)
        );
        assert_eq!(SubstanceId(60).def(), None, "past the table is refused");
        assert!(
            !keys.contains("cut stone"),
            "a mason builds with the bedrock rows themselves"
        );
    }

    #[test]
    fn the_cited_rows_carry_the_designs_numbers_and_the_void_is_a_sentinel() {
        let by_key = |k: &str| SUBSTANCES.iter().find(|s| s.key == k).expect("row");
        assert_eq!(by_key("granite").density_g_m3, 2_700_000);
        assert_eq!(by_key("granite").work_of_fracture_j_m2, 100);
        assert_eq!(by_key("basalt").work_of_fracture_j_m2, 130);
        assert_eq!(by_key("concrete").work_of_fracture_j_m2, 120);
        assert_eq!(by_key("structural steel").work_of_fracture_j_m2, 100_000);
        assert_eq!(by_key("titanium alloy").work_of_fracture_j_m2, 120_000);
        assert_eq!(by_key("oak wood").work_of_fracture_j_m2, 10_000);
        assert_eq!(by_key("glass").work_of_fracture_j_m2, 10);
        assert_eq!(by_key("ice").density_g_m3, 917_000);
        assert_eq!(by_key("dirt").work_of_fracture_j_m2, 2);
        let cited = SUBSTANCES
            .iter()
            .filter(|s| s.provenance == Provenance::Cited)
            .count();
        assert_eq!(
            cited, 9,
            "the design cites nine rows; the rest are provisional"
        );
        assert_eq!(by_key("void").provenance, Provenance::Sentinel);
    }

    #[test]
    fn every_row_states_what_it_must_and_no_packed_form_is_lighter_than_its_parent() {
        let by_key = |k: &str| SUBSTANCES.iter().find(|s| s.key == k).expect("row");
        for s in &SUBSTANCES {
            let is_matter = !matches!(s.state, MatterState::Void | MatterState::Gas);
            assert_eq!(s.density_g_m3 >= 100_000, is_matter, "{} density", s.key);
            let breaks = matches!(s.state, MatterState::Solid | MatterState::Loose);
            assert_eq!(s.work_of_fracture_j_m2 > 0, breaks, "{} fracture", s.key);
            // A solid or a liquid states a melting point, unless it burns or decomposes first.
            let decomposes = ["coal", "peat", "rubber"].contains(&s.key);
            let melts = matches!(
                s.state,
                MatterState::Solid | MatterState::Loose | MatterState::Liquid
            ) && !burns(s.family)
                && !decomposes;
            assert_eq!(s.melting_k > 0, melts, "{} melting point", s.key);
        }
        assert!(by_key("packed ice").density_g_m3 >= by_key("ice").density_g_m3);
        assert!(by_key("packed ice").density_g_m3 >= by_key("snow").density_g_m3);
        assert_eq!(by_key("lava").melting_k, 1_400, "lava freezes to basalt");
        assert!(by_key("peat").density_g_m3 <= 400_000, "dry peat is light");
        // The ore convention: the ore-bearing rock, never the pure mineral or the metal.
        for s in SUBSTANCES.iter().filter(|s| s.family == Family::Ore) {
            assert!(
                (1_300_000..=3_600_000).contains(&s.density_g_m3),
                "{} is an ore-bearing rock",
                s.key
            );
        }
    }
}
