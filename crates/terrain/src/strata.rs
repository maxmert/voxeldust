//! ★ THE STRATA — which common substance a cell holds, by its depth under the surface and the biome
//! above it (the seed law, rulings V4 and V9 S5-6: bulk stock only; every ore is live state; no
//! indicator material).
//!
//! The generator names substances by its OWN small enum, never by a registry number: the registry
//! (`vd_core::registry`) lives above this crate, and a host maps each stratum to a registry id by its
//! KEY. So a registry renumbering can never move a chunk's digest.
//!
//! **Example.** Two metres under a grassland hillside the strata say "dirt, then sediment, then the
//! body's bedrock". A hundred metres down they say "granite" on a granite moon and "basalt" on a basalt
//! moon, because the bedrock kind is one draw from the body's seed.

/// A substance the seed can decide: bulk stock, water and air. Ores are never here.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[repr(u8)]
pub enum Stratum {
    Air = 0,
    Water = 1,
    Snow = 2,
    Ice = 3,
    Sand = 4,
    Dirt = 5,
    Clay = 6,
    Gravel = 7,
    Permafrost = 8,
    Sandstone = 9,
    Limestone = 10,
    Shale = 11,
    Granite = 12,
    Basalt = 13,
    Gabbro = 14,
    Andesite = 15,
    Quartzite = 16,
    Salt = 17,
    /// THE REMOVAL (ruling V4): a mined cell holds `Empty`, never `Air`, which is the atmosphere.
    /// Appended in slice 6 at code 18; `is_solid` is false and the registry substance is `void`.
    Empty = 18,
    /// ★ THE ROCK MAP's two new bedrocks (slice 8d step 2), APPENDED so no code already stored
    /// moves: slate is the folded belt's own soft rock and marble the basement's.
    Slate = 19,
    Marble = 20,
}

impl Stratum {
    /// Every stratum, for the host's mapping test.
    pub const ALL: [Stratum; 21] = [
        Stratum::Air,
        Stratum::Water,
        Stratum::Snow,
        Stratum::Ice,
        Stratum::Sand,
        Stratum::Dirt,
        Stratum::Clay,
        Stratum::Gravel,
        Stratum::Permafrost,
        Stratum::Sandstone,
        Stratum::Limestone,
        Stratum::Shale,
        Stratum::Granite,
        Stratum::Basalt,
        Stratum::Gabbro,
        Stratum::Andesite,
        Stratum::Quartzite,
        Stratum::Salt,
        Stratum::Empty,
        Stratum::Slate,
        Stratum::Marble,
    ];

    /// The registry KEY this stratum maps to (`vd_core::registry::SUBSTANCES` names its rows by
    /// key). The host looks the id up by this string once at boot.
    #[must_use]
    pub const fn registry_key(self) -> &'static str {
        match self {
            Stratum::Air => "air",
            Stratum::Water => "water",
            Stratum::Snow => "snow",
            Stratum::Ice => "ice",
            Stratum::Sand => "sand",
            Stratum::Dirt => "dirt",
            Stratum::Clay => "clay",
            Stratum::Gravel => "gravel",
            Stratum::Permafrost => "permafrost",
            Stratum::Sandstone => "sandstone",
            Stratum::Limestone => "limestone",
            Stratum::Shale => "shale",
            Stratum::Granite => "granite",
            Stratum::Basalt => "basalt",
            Stratum::Gabbro => "gabbro",
            Stratum::Andesite => "andesite",
            Stratum::Quartzite => "quartzite",
            Stratum::Salt => "salt",
            Stratum::Empty => "void",
            Stratum::Slate => "slate",
            Stratum::Marble => "marble",
        }
    }

    /// Whether the cell is matter the surface passes through (not air, not water).
    #[must_use]
    pub const fn is_solid(self) -> bool {
        !matches!(self, Stratum::Air | Stratum::Water | Stratum::Empty)
    }

    /// The byte a digest folds and a record stores.
    #[must_use]
    pub const fn code(self) -> u8 {
        self as u8
    }

    /// The stratum of a code byte; `None` for a byte no stratum owns (a decoder REFUSES, never
    /// defaults).
    #[must_use]
    pub fn from_code(code: u8) -> Option<Stratum> {
        Stratum::ALL.get(usize::from(code)).copied()
    }
}

/// The biome above a column: what the topsoil is and how deep it goes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
pub enum Biome {
    /// Hot and dry: sand over sandstone.
    Desert = 0,
    /// Temperate: dirt over clay over sediment.
    Grassland = 1,
    /// Cold: snow over permafrost.
    Tundra = 2,
    /// Steep and high: bare rock, a little gravel.
    Highland = 3,
}

impl Biome {
    pub const ALL: [Biome; 4] = [
        Biome::Desert,
        Biome::Grassland,
        Biome::Tundra,
        Biome::Highland,
    ];
}

/// ★ THE RECIPE'S ROW COUNT IS THIS ENUM'S COUNT. The cell kernel picks a biome's strata row by a
/// MASK (`vd_recipe::cell::BIOME_MASK`), because a mask cannot be out of bounds and a shader has no
/// bound to test. A mask only names the right row while the rows are exactly the biomes: a FIFTH
/// biome added here without a fifth row there would fold onto the desert's row silently, and every
/// cell of that biome would read sand where it should read something else. This assertion makes
/// that a red BUILD, not a wrong world; `chunk::charter_of` fills the rows in this enum's own order
/// and `chunk`'s own test measures that each biome reads its own.
const _: () = assert!(vd_recipe::cell::BIOMES == Biome::ALL.len());

/// ★ THE ROCK PROVINCE of one node (slice 8d step 2; `slice_8d_design.md` §3.5; ruling W4 item 4,
/// the owner's SL6 approval of the datum). A hillside of flat beds reads as one desert everywhere,
/// so the beds need a REGION: which crust a node stands on and what the plates did to it. The solve
/// knows both; the word rides the artifact's node row, because the solve's crust and belt fields are
/// its own working state and are thrown away.
///
/// The province decides WHICH ROCK the beds are drawn from, and nothing else. It is a map of LOOK
/// (ruling S5-6): no ore is in it, and none ever will be, because a map a seed alone decides is a
/// map a wiki publishes.
///
/// **Example.** A miner digs under the shelf province and cuts shale, then limestone, then
/// sandstone, bed by bed. A miner digging the same wall in the rift province cuts basalt all the
/// way down. Neither can read where the copper is, because no copper is in the seed.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
#[repr(u8)]
pub enum Province {
    /// The old dry continent: a shield the plates left standing.
    CrystallineBasement = 0,
    /// A collision belt on continental crust: the rock is folded and metamorphosed.
    FoldedBelt = 1,
    /// Continental crust under its own sea: the drowned platform.
    FlatShelf = 2,
    /// Ground the plates are pulling apart or a floor they are building: new lava.
    RiftBasalt = 3,
    /// The still ocean floor and the trench that fills with it.
    DeepSediment = 4,
}

/// ★ THE TENSILE STRENGTH OF A ROCK, megapascals (2026-09-22, ruling W7 step 3): the one number
/// the stream-power erodibility reads about a rock — the cut goes as the INVERSE SQUARE of it
/// (Sklar & Dietrich 2001, Geology 29: `E ∝ σ_T⁻²`, measured in an abrasion mill on these rocks).
/// The values are the published typical ones for each rock (Sklar & Dietrich 2001 Table 1; the
/// Rock Mechanics handbooks' Brazilian-test ranges, their middles). Shale is the softest and is
/// the reference `K₀` was calibrated on. A rock the cell pass never places (air, water, soil,
/// snow) answers shale's, so nothing divides by zero and no soil is harder than rock.
#[must_use]
pub const fn tensile_strength_mpa(s: Stratum) -> u32 {
    match s {
        Stratum::Shale => 3,
        Stratum::Sandstone => 5,
        Stratum::Limestone => 6,
        Stratum::Slate => 8,
        Stratum::Marble => 7,
        Stratum::Granite => 10,
        Stratum::Andesite => 10,
        Stratum::Basalt => 12,
        Stratum::Gabbro => 12,
        Stratum::Quartzite => 15,
        Stratum::Salt => 2,
        _ => 3,
    }
}

impl Province {
    /// ★ THE ERODIBILITY OF A PROVINCE as a share of `K₀` in 1/256 (W7 step 3): the inverse
    /// square of its four rocks' mean tensile strength over shale's ([`tensile_strength_mpa`]),
    /// so a shield of granite and quartzite cuts about a tenth as fast as a shale shelf. An
    /// integer, one answer on every host; never over 256.
    #[must_use]
    #[allow(
        clippy::integer_division,
        reason = "five provinces, once per solve, never a kernel's path: an exact quotient a test states"
    )]
    pub fn erodibility_q8(self) -> u32 {
        let rocks = self.rocks();
        let mut sum = 0u32;
        let mut k = 0;
        while k < rocks.len() {
            sum += tensile_strength_mpa(rocks[k]);
            k += 1;
        }
        // Shale's strength, squared, times 256, over the mean's square — the mean is `sum / 4`,
        // so the square of the sum carries 16.
        let shale = tensile_strength_mpa(Stratum::Shale);
        let q8 = 256 * 16 * shale * shale / (sum * sum);
        if q8 > 256 { 256 } else { q8 }
    }

    pub const ALL: [Province; 5] = [
        Province::CrystallineBasement,
        Province::FoldedBelt,
        Province::FlatShelf,
        Province::RiftBasalt,
        Province::DeepSediment,
    ];

    /// The byte the artifact's row stores.
    #[must_use]
    pub const fn code(self) -> u8 {
        self as u8
    }

    /// The province of a code byte; `None` for a byte no province owns (a reader REFUSES, never
    /// defaults).
    #[must_use]
    pub fn from_code(code: u8) -> Option<Province> {
        Province::ALL.get(usize::from(code)).copied()
    }

    /// ★ THE FOUR ROCKS OF A PROVINCE, the SOFT pair first and the HARD pair second. THE SEED'S OWN
    /// IDENTITY CHOICE under ruling T9: which rock a province is made of is a choice like the body's
    /// tilt, not a physical number, so it is stated here as data and no law computes it. Every name
    /// is a registry KEY through [`Stratum::registry_key`], never a number typed by hand.
    ///
    /// A bed whose hardness passes the bench's cap threshold reads the hard pair, so a cap-rock
    /// tread stands on a rock that really is harder than the riser under it.
    ///
    /// **Example.** The pilot walks up a folded belt. The riser at her knee is quartzite and the
    /// tread she steps onto is slate, and both run along the whole hillside at one height.
    #[must_use]
    pub const fn rocks(self) -> [Stratum; 4] {
        match self {
            Province::CrystallineBasement => [
                Stratum::Slate,
                Stratum::Marble,
                Stratum::Granite,
                Stratum::Quartzite,
            ],
            Province::FoldedBelt => [
                Stratum::Shale,
                Stratum::Slate,
                Stratum::Granite,
                Stratum::Quartzite,
            ],
            Province::FlatShelf => [
                Stratum::Shale,
                Stratum::Limestone,
                Stratum::Sandstone,
                Stratum::Limestone,
            ],
            Province::RiftBasalt => [
                Stratum::Andesite,
                Stratum::Basalt,
                Stratum::Basalt,
                Stratum::Gabbro,
            ],
            Province::DeepSediment => [
                Stratum::Shale,
                Stratum::Sandstone,
                Stratum::Sandstone,
                Stratum::Limestone,
            ],
        }
    }
}

/// ★ THE PROVINCE A HOST READS WHERE IT HOLDS NO ROW: the basement. A body with no artifact — the
/// crate's own kernel tests, the card's plan, a chunk whose tile has not arrived — has no plates to
/// read, so every column stands on the oldest ground the map names. STATED, never guessed.
pub const DEFAULT_PROVINCE: Province = Province::CrystallineBasement;

/// ★ THE DEFAULT'S CODE IS ZERO, AND THE SHADER COUNTS ON IT. The GPU shell (`vd-recipe-gpu`) may
/// name nothing above the recipe, so it writes `Gi::ZERO` for a card's province. A default moved off
/// zero here would make the card read another province's rocks and the drift gate would go red for
/// a reason nobody could see in the diff. This assertion makes it a red BUILD instead.
const _: () = assert!(DEFAULT_PROVINCE.code() == 0);

/// ★ THE RECIPE'S PROVINCE ROW COUNT COVERS THIS ENUM. The cell kernel picks a province's row of
/// rocks by a MASK (`vd_recipe::cell::PROVINCE_MASK`), exactly as it picks a biome's row, so a
/// sixth province added here without room there would fold onto the basement's row in silence. This
/// assertion makes that a red BUILD, not a wrong world.
const _: () = assert!(vd_recipe::cell::PROVINCES >= Province::ALL.len());

/// The body's bedrock kind: one draw from the seed.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Bedrock {
    Granite,
    Basalt,
    Gabbro,
    Andesite,
    Quartzite,
}

impl Bedrock {
    pub const ALL: [Bedrock; 5] = [
        Bedrock::Granite,
        Bedrock::Basalt,
        Bedrock::Gabbro,
        Bedrock::Andesite,
        Bedrock::Quartzite,
    ];

    #[must_use]
    pub const fn stratum(self) -> Stratum {
        match self {
            Bedrock::Granite => Stratum::Granite,
            Bedrock::Basalt => Stratum::Basalt,
            Bedrock::Gabbro => Stratum::Gabbro,
            Bedrock::Andesite => Stratum::Andesite,
            Bedrock::Quartzite => Stratum::Quartzite,
        }
    }
}

/// The depth table of a body: how deep each layer runs, in whole metres.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct StrataTable {
    /// The topsoil's depth (dirt, sand or snow).
    pub topsoil_m: u32,
    /// The subsoil's depth below the topsoil (clay, gravel or permafrost).
    pub subsoil_m: u32,
    /// The sediment's depth below the subsoil (sandstone, limestone or shale).
    pub sediment_m: u32,
    /// Which sediment the body has.
    pub sediment: Stratum,
    /// The bedrock below everything.
    pub bedrock: Bedrock,
}

impl StrataTable {
    /// The deepest metre the strata change at; below it a cell is bedrock without a lookup.
    #[must_use]
    pub const fn max_depth_m(&self) -> u32 {
        self.topsoil_m + self.subsoil_m + self.sediment_m
    }

    /// The substance at `depth_m` whole metres under the surface (0 is the surface cell) in `biome`.
    #[must_use]
    pub fn at(&self, biome: Biome, depth_m: u32) -> Stratum {
        if depth_m < self.topsoil_m {
            return match biome {
                Biome::Desert => Stratum::Sand,
                Biome::Grassland => Stratum::Dirt,
                Biome::Tundra => Stratum::Snow,
                Biome::Highland => Stratum::Gravel,
            };
        }
        if depth_m < self.topsoil_m + self.subsoil_m {
            return match biome {
                Biome::Desert => Stratum::Sandstone,
                Biome::Grassland => Stratum::Clay,
                Biome::Tundra => Stratum::Permafrost,
                Biome::Highland => self.bedrock.stratum(),
            };
        }
        if depth_m < self.max_depth_m() {
            return self.sediment;
        }
        self.bedrock.stratum()
    }
}

#[cfg(test)]
mod tests {
    //! ★ A TEST MAY DIVIDE (ruling F7's rule is about the SHIPPED path, not the measurement): a test
    //! states the exact quotient a reciprocal stands for, and a fixture picks its sample columns with a
    //! remainder. Neither runs in a kernel.
    #![allow(
        clippy::integer_division,
        clippy::modulo_arithmetic,
        reason = "a test states an exact quotient or picks a sample column; never a kernel's path"
    )]
    use super::*;

    fn table() -> StrataTable {
        StrataTable {
            topsoil_m: 2,
            subsoil_m: 5,
            sediment_m: 40,
            sediment: Stratum::Limestone,
            bedrock: Bedrock::Basalt,
        }
    }

    #[test]
    fn every_stratum_has_a_distinct_code_and_a_registry_key() {
        let mut codes = std::collections::BTreeSet::new();
        let mut keys = std::collections::BTreeSet::new();
        for s in Stratum::ALL {
            assert!(codes.insert(s.code()), "{s:?} shares a code");
            assert!(keys.insert(s.registry_key()), "{s:?} shares a key");
        }
        assert_eq!(codes.len(), 21);
        assert!(!Stratum::Air.is_solid());
        assert!(!Stratum::Water.is_solid());
        assert!(Stratum::Granite.is_solid());
        for b in Bedrock::ALL {
            assert!(b.stratum().is_solid());
        }
        assert_eq!(Biome::ALL.len(), 4);
    }

    /// ★ FAILING FIRST (slice 8d step 2): EVERY PROVINCE HAS ITS OWN CODE, ITS OWN FOUR ROCKS, AND
    /// EVERY ROCK IS SOLID BEDROCK OF THE REGISTRY. Four statements. (1) The codes are dense and
    /// distinct, so the row's byte reads back to the province that wrote it. (2) A byte no province
    /// owns is refused, never defaulted. (3) Every rock is solid and names a registry key. (4) No
    /// two provinces state the same four rocks, so the map really is a map.
    /// ★ THE ERODIBILITY BY ROCK (W7 step 3), three statements: shale's own share is the whole
    /// (256); a shield of granite and quartzite cuts under a third as fast as a shale shelf; every
    /// province's share stands in `1..=256`, and an unplaced stratum reads shale's strength.
    #[test]
    fn the_erodibility_falls_with_the_rocks_tensile_strength() {
        assert_eq!(
            tensile_strength_mpa(Stratum::Shale),
            tensile_strength_mpa(Stratum::Air)
        );
        let shelf = Province::FlatShelf.erodibility_q8();
        let shield = Province::CrystallineBasement.erodibility_q8();
        assert!(shield * 3 < shelf, "shield {shield} against shelf {shelf}");
        for p in Province::ALL {
            let q = p.erodibility_q8();
            assert!((1..=256).contains(&q), "{p:?}: {q}");
        }
        // The shale-only reading: four shales would give exactly 256.
        let shale = tensile_strength_mpa(Stratum::Shale);
        assert_eq!(256 * 16 * shale * shale / ((4 * shale) * (4 * shale)), 256);
    }

    #[test]
    fn every_province_has_a_code_and_four_solid_rocks() {
        let mut codes = std::collections::BTreeSet::new();
        let mut rows = std::collections::BTreeSet::new();
        for p in Province::ALL {
            assert!(codes.insert(p.code()), "{p:?} shares a code");
            assert_eq!(Province::from_code(p.code()), Some(p));
            let rocks = p.rocks();
            for r in rocks {
                assert!(r.is_solid(), "{p:?} states {r:?}");
                assert!(!r.registry_key().is_empty());
            }
            assert!(rows.insert(rocks), "{p:?} shares its rocks");
        }
        assert_eq!(codes.len(), 5);
        assert_eq!(Province::from_code(5), None);
        assert_eq!(Province::from_code(255), None);
        assert_eq!(DEFAULT_PROVINCE, Province::CrystallineBasement);
        // The two rocks the rock map added read their own registry keys.
        assert_eq!(Stratum::Slate.registry_key(), "slate");
        assert_eq!(Stratum::Marble.registry_key(), "marble");
        assert_eq!(Stratum::from_code(20), Some(Stratum::Marble));
    }

    #[test]
    fn the_depth_table_reads_topsoil_subsoil_sediment_then_bedrock_in_every_biome() {
        let t = table();
        assert_eq!(t.max_depth_m(), 47);
        assert_eq!(t.at(Biome::Grassland, 0), Stratum::Dirt);
        assert_eq!(t.at(Biome::Grassland, 3), Stratum::Clay);
        assert_eq!(t.at(Biome::Grassland, 10), Stratum::Limestone);
        assert_eq!(t.at(Biome::Grassland, 47), Stratum::Basalt);
        assert_eq!(t.at(Biome::Desert, 1), Stratum::Sand);
        assert_eq!(t.at(Biome::Desert, 4), Stratum::Sandstone);
        assert_eq!(t.at(Biome::Tundra, 0), Stratum::Snow);
        assert_eq!(t.at(Biome::Tundra, 6), Stratum::Permafrost);
        assert_eq!(t.at(Biome::Highland, 1), Stratum::Gravel);
        assert_eq!(t.at(Biome::Highland, 2), Stratum::Basalt);
        assert_eq!(t.at(Biome::Highland, 500), Stratum::Basalt);
    }
}
