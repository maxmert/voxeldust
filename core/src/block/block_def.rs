/// Material category — affects mining speed multipliers, sounds, and particle effects.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum MaterialType {
    Stone = 0,
    Soil = 1,
    Wood = 2,
    Metal = 3,
    Glass = 4,
    Organic = 5,
    Fluid = 6,
    Crystal = 7,
    Ice = 8,
}

/// Tool type that receives a mining speed bonus.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum ToolType {
    None = 0,
    Pickaxe = 1,
    Shovel = 2,
    Axe = 3,
    Drill = 4,
}

/// Properties specific to liquid blocks. Use `LiquidProperties::NONE` for non-liquids.
#[derive(Clone, Copy, Debug)]
pub struct LiquidProperties {
    /// Gravity multiplier when entity is submerged. 0.0 = not a liquid.
    pub gravity_multiplier: f32,
    /// Drag coefficient opposing entity velocity in the fluid.
    pub drag: f32,
    /// Upward force when the swim/jump input is held.
    pub swim_force: f32,
}

impl LiquidProperties {
    pub const NONE: Self = Self {
        gravity_multiplier: 0.0,
        drag: 0.0,
        swim_force: 0.0,
    };

    #[inline]
    pub const fn is_liquid(&self) -> bool {
        // Using to_bits() for const comparison since f32 == is not const.
        self.gravity_multiplier.to_bits() != 0.0_f32.to_bits()
    }
}

/// Static, immutable properties for a block type. One entry per `BlockId` in the
/// registry. These do not change at runtime — per-instance state (health,
/// orientation) lives in `BlockMeta`.
#[derive(Clone, Debug)]
pub struct BlockDef {
    /// Human-readable name.
    pub name: &'static str,

    // -- Physics --
    /// Whether this block has collision (solid = blocks movement and ray casts).
    pub is_solid: bool,
    /// Whether light and vision pass through this block.
    pub is_transparent: bool,
    /// Whether this block falls when unsupported (sand, gravel).
    pub gravity_affected: bool,
    /// Material category.
    pub material_type: MaterialType,
    /// Density in kg/m^3. Used for mass calculations in ship aggregation.
    pub density: u16,

    // -- Mining --
    /// Base hardness. 0 = instant break (tall grass). 255 = hardest (obsidian).
    /// Actual hit points = hardness * HARDNESS_HP_MULTIPLIER.
    pub hardness: u8,
    /// Minimum tool tier required to mine this block (0 = hand).
    pub mining_tier: u8,
    /// Tool type that mines this block fastest.
    pub preferred_tool: ToolType,
    /// Resistance to explosions (0–255).
    pub blast_resistance: u8,

    // -- Visual --
    /// RGB color hint for the shader (used when no texture is available).
    pub color_hint: [u8; 3],
    /// Light emission level (0 = none, 15 = brightest).
    pub light_emission: u8,

    // -- Fire --
    /// How easily this block catches fire (0 = non-flammable, 255 = extremely).
    pub flammability: u8,
    /// How long this block resists burning (0 = burns instantly, 255 = fireproof).
    pub fire_resistance: u8,

    // -- Liquid --
    /// Liquid properties. `LiquidProperties::NONE` for non-liquid blocks.
    pub liquid: LiquidProperties,
}

/// Multiplier from `BlockDef::hardness` (u8) to actual hit points (u16).
/// hardness=1 → 10 HP, hardness=255 → 2550 HP.
pub const HARDNESS_HP_MULTIPLIER: u16 = 10;

/// Number of visual damage stages for the block-breaking animation.
pub const DAMAGE_STAGES: u8 = 10;

impl BlockDef {
    /// Default definition used for undefined block IDs (acts like air).
    pub const UNDEFINED: Self = Self {
        name: "undefined",
        is_solid: false,
        is_transparent: true,
        gravity_affected: false,
        material_type: MaterialType::Stone,
        density: 0,
        hardness: 0,
        mining_tier: 0,
        preferred_tool: ToolType::None,
        blast_resistance: 0,
        color_hint: [0, 0, 0],
        light_emission: 0,
        flammability: 0,
        fire_resistance: 0,
        liquid: LiquidProperties::NONE,
    };

    /// Maximum hit points for this block type.
    #[inline]
    pub const fn max_hp(&self) -> u16 {
        (self.hardness as u16) * HARDNESS_HP_MULTIPLIER
    }
}
