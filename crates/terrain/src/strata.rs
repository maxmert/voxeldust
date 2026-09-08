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
}

impl Stratum {
    /// Every stratum, for the host's mapping test.
    pub const ALL: [Stratum; 18] = [
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
        }
    }

    /// Whether the cell is matter the surface passes through (not air, not water).
    #[must_use]
    pub const fn is_solid(self) -> bool {
        !matches!(self, Stratum::Air | Stratum::Water)
    }

    /// The byte a digest folds and a record stores.
    #[must_use]
    pub const fn code(self) -> u8 {
        self as u8
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
        assert_eq!(codes.len(), 18);
        assert!(!Stratum::Air.is_solid());
        assert!(!Stratum::Water.is_solid());
        assert!(Stratum::Granite.is_solid());
        for b in Bedrock::ALL {
            assert!(b.stratum().is_solid());
        }
        assert_eq!(Biome::ALL.len(), 4);
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
