/// Block type identifier. 0 = Air (always).
///
/// A zero-cost wrapper around `u16`, supporting up to 65,535 block types.
/// Constructing from raw `u16` is allowed for deserialization; gameplay code
/// should use the named constants.
#[derive(Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Debug, Default)]
#[repr(transparent)]
pub struct BlockId(pub u16);

impl BlockId {
    // -----------------------------------------------------------------------
    // Basic (0–5)
    // -----------------------------------------------------------------------
    pub const AIR: Self = Self(0);
    pub const STONE: Self = Self(1);
    pub const DIRT: Self = Self(2);
    pub const GRASS: Self = Self(3);
    pub const SAND: Self = Self(4);
    pub const WATER: Self = Self(5);

    // -----------------------------------------------------------------------
    // Stone variants (6–13)
    // -----------------------------------------------------------------------
    pub const GRANITE: Self = Self(6);
    pub const BASALT: Self = Self(7);
    pub const LIMESTONE: Self = Self(8);
    pub const MARBLE: Self = Self(9);
    pub const SLATE: Self = Self(10);
    pub const SANDSTONE: Self = Self(11);
    pub const OBSIDIAN: Self = Self(12);
    pub const PUMICE: Self = Self(13);

    // -----------------------------------------------------------------------
    // Soil / Surface (20–25)
    // -----------------------------------------------------------------------
    pub const CLAY: Self = Self(20);
    pub const MUD: Self = Self(21);
    pub const GRAVEL: Self = Self(22);
    pub const PEAT: Self = Self(23);
    pub const RED_SAND: Self = Self(24);
    pub const DARK_SOIL: Self = Self(25);

    // -----------------------------------------------------------------------
    // Ice / Snow (30–33)
    // -----------------------------------------------------------------------
    pub const SNOW: Self = Self(30);
    pub const PACKED_ICE: Self = Self(31);
    pub const BLUE_ICE: Self = Self(32);
    pub const PERMAFROST: Self = Self(33);

    // -----------------------------------------------------------------------
    // Volcanic (40–42)
    // -----------------------------------------------------------------------
    pub const MAGMA_ROCK: Self = Self(40);
    pub const ASH: Self = Self(41);
    pub const SULFUR: Self = Self(42);

    // -----------------------------------------------------------------------
    // Ores (100–111)
    // -----------------------------------------------------------------------
    pub const COAL_ORE: Self = Self(100);
    pub const IRON_ORE: Self = Self(101);
    pub const COPPER_ORE: Self = Self(102);
    pub const GOLD_ORE: Self = Self(103);
    pub const DIAMOND_ORE: Self = Self(104);
    pub const EMERALD_ORE: Self = Self(105);
    pub const RUBY_ORE: Self = Self(106);
    pub const SAPPHIRE_ORE: Self = Self(107);
    pub const URANIUM_ORE: Self = Self(108);
    pub const TITANIUM_ORE: Self = Self(109);
    pub const MYTHRIL_ORE: Self = Self(110);
    pub const ENERGY_CRYSTAL: Self = Self(111);

    // -----------------------------------------------------------------------
    // Fluids (200–202)
    // -----------------------------------------------------------------------
    pub const LAVA: Self = Self(200);
    pub const SALT_WATER: Self = Self(201);
    pub const OIL: Self = Self(202);

    // -----------------------------------------------------------------------
    // Vegetation (300–311)
    // -----------------------------------------------------------------------
    pub const OAK_LOG: Self = Self(300);
    pub const OAK_LEAVES: Self = Self(301);
    pub const PINE_LOG: Self = Self(302);
    pub const PINE_LEAVES: Self = Self(303);
    pub const JUNGLE_LOG: Self = Self(304);
    pub const JUNGLE_LEAVES: Self = Self(305);
    pub const BIRCH_LOG: Self = Self(306);
    pub const BIRCH_LEAVES: Self = Self(307);
    pub const ACACIA_LOG: Self = Self(308);
    pub const ACACIA_LEAVES: Self = Self(309);
    pub const TALL_GRASS: Self = Self(310);
    pub const MOSS: Self = Self(311);
    pub const CACTUS: Self = Self(312);
    pub const DEAD_BUSH: Self = Self(313);

    // -----------------------------------------------------------------------
    // Functional blocks — Propulsion (1000–1099)
    // -----------------------------------------------------------------------
    pub const THRUSTER_SMALL_CHEMICAL: Self = Self(1000);
    pub const THRUSTER_MEDIUM_CHEMICAL: Self = Self(1001);
    pub const THRUSTER_LARGE_CHEMICAL: Self = Self(1002);
    pub const THRUSTER_SMALL_ION: Self = Self(1010);
    pub const THRUSTER_MEDIUM_ION: Self = Self(1011);
    pub const THRUSTER_LARGE_ION: Self = Self(1012);
    pub const THRUSTER_SMALL_FUSION: Self = Self(1020);
    pub const THRUSTER_MEDIUM_FUSION: Self = Self(1021);
    pub const THRUSTER_LARGE_FUSION: Self = Self(1022);

    // -----------------------------------------------------------------------
    // Functional blocks — Power (1100–1199)
    // -----------------------------------------------------------------------
    pub const REACTOR_SMALL: Self = Self(1100);
    pub const REACTOR_MEDIUM: Self = Self(1101);
    pub const REACTOR_LARGE: Self = Self(1102);
    pub const BATTERY: Self = Self(1110);
    pub const SOLAR_PANEL: Self = Self(1120);
    pub const POWER_CONDUIT: Self = Self(1130);

    // -----------------------------------------------------------------------
    // Functional blocks — Life Support (1200–1299)
    // -----------------------------------------------------------------------
    pub const AIR_COMPRESSOR: Self = Self(1200);
    pub const AIR_VENT: Self = Self(1201);

    // -----------------------------------------------------------------------
    // Functional blocks — Shields (1300–1399)
    // -----------------------------------------------------------------------
    pub const SHIELD_EMITTER: Self = Self(1300);
    pub const SHIELD_GENERATOR: Self = Self(1301);

    // -----------------------------------------------------------------------
    // Functional blocks — Mechanical (1400–1499)
    // -----------------------------------------------------------------------
    pub const ROTOR: Self = Self(1400);
    pub const PISTON: Self = Self(1401);

    // -----------------------------------------------------------------------
    // Functional blocks — Transport (1500–1599)
    // -----------------------------------------------------------------------
    pub const RAIL_STRAIGHT: Self = Self(1500);
    pub const RAIL_CURVE: Self = Self(1501);
    pub const RAIL_JUNCTION: Self = Self(1502);
    pub const RAIL_SIGNAL: Self = Self(1503);

    // -----------------------------------------------------------------------
    // Functional blocks — Logic (1600–1699)
    // -----------------------------------------------------------------------
    pub const SIGNAL_CONVERTER: Self = Self(1600);
    pub const SENSOR_PRESSURE: Self = Self(1610);
    pub const SENSOR_PROXIMITY: Self = Self(1611);
    pub const SENSOR_SPEED: Self = Self(1612);
    pub const SENSOR_POWER: Self = Self(1613);
    pub const SENSOR_DAMAGE: Self = Self(1614);
    pub const COMPUTER: Self = Self(1650);

    // -----------------------------------------------------------------------
    // Structural (1700–1799)
    // -----------------------------------------------------------------------
    pub const HULL_LIGHT: Self = Self(1700);
    pub const HULL_STANDARD: Self = Self(1701);
    pub const HULL_HEAVY: Self = Self(1702);
    pub const HULL_ARMORED: Self = Self(1703);
    pub const WINDOW: Self = Self(1710);
    pub const DOOR: Self = Self(1720);

    // -----------------------------------------------------------------------
    // Utility (1800–1899)
    // -----------------------------------------------------------------------
    pub const COCKPIT: Self = Self(1800);          // Pilot seat (pre-configured bindings)
    pub const OWNERSHIP_CORE: Self = Self(1801);
    pub const SEAT: Self = Self(1802);             // Generic seat (empty, player configures)
    pub const HUD_PANEL: Self = Self(1810);
    pub const GRAVITY_GENERATOR: Self = Self(1820);
    pub const ANTENNA: Self = Self(1830);

    // -----------------------------------------------------------------------
    // Ship system blocks (1840–1859)
    // -----------------------------------------------------------------------
    pub const FLIGHT_COMPUTER: Self = Self(1840);
    pub const HOVER_MODULE: Self = Self(1841);
    pub const AUTOPILOT: Self = Self(1842);
    pub const WARP_COMPUTER: Self = Self(1843);
    pub const ENGINE_CONTROLLER: Self = Self(1844);

    // Cruise drives (propulsion boost modules)
    pub const CRUISE_DRIVE_SMALL: Self = Self(1900);
    pub const CRUISE_DRIVE_MEDIUM: Self = Self(1910);
    pub const CRUISE_DRIVE_LARGE: Self = Self(1920);

    #[inline(always)]
    pub const fn as_u16(self) -> u16 {
        self.0
    }

    #[inline(always)]
    pub const fn is_air(self) -> bool {
        self.0 == 0
    }

    #[inline(always)]
    pub const fn from_u16(v: u16) -> Self {
        Self(v)
    }
}

impl std::fmt::Display for BlockId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "BlockId({})", self.0)
    }
}
