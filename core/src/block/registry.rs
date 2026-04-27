use super::block_def::{BlockDef, LiquidProperties, MaterialType, ToolType};
use super::block_id::BlockId;

/// Maximum number of block types supported by the u16 address space.
const MAX_BLOCK_TYPES: usize = 65536;

// ---------------------------------------------------------------------------
// Functional block types and interaction schemas
// ---------------------------------------------------------------------------

/// Category of functional block — determines which subsystems interact with it.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum FunctionalBlockKind {
    Thruster,
    Reactor,
    Battery,
    SolarPanel,
    PowerConduit,
    Seat,
    GravityGenerator,
    ShieldEmitter,
    ShieldGenerator,
    AirCompressor,
    Antenna,
    Rotor,
    Piston,
    Rail,
    RailJunction,
    RailSignal,
    SignalConverter,
    Sensor,
    Computer,
    CruiseDrive,
    FlightComputer,
    HoverModule,
    Autopilot,
    WarpComputer,
    EngineController,
}

/// Per-thruster-block static properties (thrust output, fuel consumption).
#[derive(Clone, Copy, Debug)]
pub struct ThrusterProps {
    /// Maximum thrust output in Newtons.
    pub thrust_n: f64,
    /// Fuel consumption rate (kg/s) — future use for fuel system.
    pub fuel_rate: f64,
}

/// Per-light-emitter-block specification driving the client's per-block
/// `PointLight` / `SpotLight` spawning. Server and client both read this
/// from the same shared registry, so a placement decision (where the
/// block sits in the chunk grid) deterministically implies a specific
/// light contribution everywhere.
///
/// Photometric units are physical:
///   * `lumens` — luminous flux Φ (lm). Bevy 0.18's `PointLight.intensity`
///     and `SpotLight.intensity` consume this value directly.
///   * `color_kelvin` — blackbody temperature for a thermal emitter
///     (incandescent / sun-like). Converted to linear sRGB on the client
///     via `core::blackbody::temperature_to_linear_rgb`.
///   * `tint_linear_rgb` — multiplicative tint applied on top of the
///     blackbody colour. Models real-world filter / LED chromaticity that
///     can't be expressed as a pure blackbody (`[1.0, 0.1, 0.05]` for a
///     red emergency lamp, `[1, 1, 1]` for a pure thermal source).
///   * `range_m` — physical falloff cutoff in metres.
///   * `emissive_radiance_w_per_m2` — companion value driving the
///     emissive proxy surface's HDR brightness (the visible "bulb face"
///     glow, independent of the spawned light's surrounding luminance).
///
/// `shadow_caster_class` is the priority gate the client's shadow-budget
/// enforcer uses: `0` = never casts shadows, higher = higher priority
/// for the configurable `LightingFidelity.max_shadow_casting_local_lights`
/// budget.
///
/// `emissive_geometry` chooses the shape of the visible emissive
/// surface:
///   * [`EmissiveGeometry::FullCube`] — glow from every air-facing side
///     (lamps, reactors).
///   * [`EmissiveGeometry::DirectionalFace`] — glow from a single face
///     aligned with the block's `BlockOrientation.facing_direction()`
///     (thruster exhaust, beam sources).
#[derive(Clone, Copy, Debug)]
pub struct LightSpec {
    pub lumens: f32,
    pub color_kelvin: f32,
    pub tint_linear_rgb: [f32; 3],
    pub range_m: f32,
    pub beam_kind: BeamKind,
    pub shadow_caster_class: u8,
    pub emissive_radiance_w_per_m2: f32,
    pub emissive_geometry: EmissiveGeometry,
}

/// Beam shape for a [`LightSpec`].
#[derive(Clone, Copy, Debug)]
pub enum BeamKind {
    /// Omni-directional point source. Maps to Bevy's `PointLight`.
    PointOmni,
    /// Bounded cone, angles in radians. Inner = full-intensity edge,
    /// outer = zero-intensity edge (Bevy interpolates between the two).
    /// Maps to Bevy's `SpotLight`.
    SpotCone {
        inner_angle_rad: f32,
        outer_angle_rad: f32,
    },
}

/// Visible emissive surface shape attached to a light-emitter block.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum EmissiveGeometry {
    /// All air-facing sides of the block glow (housekeeping fixture
    /// look — lamps, reactor cores). The proxy is a slightly outset
    /// cube; faces bordering adjacent solid blocks remain occluded by
    /// the neighbour's mesh face.
    FullCube,
    /// Only the face aligned with the block's
    /// `BlockOrientation.facing_direction()` glows — the canonical
    /// "exhaust nozzle" / "spotlight emitter" look. The proxy is a
    /// flat quad sitting just outside that single face.
    DirectionalFace,
}

/// Per-cruise-drive-block static properties.
#[derive(Clone, Copy, Debug)]
pub struct CruiseDriveProps {
    /// Thrust boost multiplier applied to subscribed thrusters when active.
    pub boost_multiplier: f64,
}

/// Joint type for mechanical mounts.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum JointType {
    /// Rotation around the mount face normal (rotors, hinges).
    Revolute,
    /// Translation along the mount face normal (pistons, sliders).
    Prismatic,
}

/// Per-block static mechanical properties for sub-block mounts (rotors, pistons).
#[derive(Clone, Copy, Debug)]
pub struct MechanicalProps {
    /// Joint type (determines Rapier joint API).
    pub joint_type: JointType,
    /// Maximum torque (N·m) for revolute or force (N) for prismatic.
    pub max_force: f64,
    /// Maximum speed: deg/s for revolute, m/s for prismatic.
    pub max_speed: f64,
    /// Maximum range: degrees for revolute (360 = unlimited), meters for prismatic.
    pub max_range: f64,
}

/// Per-block static power properties (generation, consumption, storage).
#[derive(Clone, Copy, Debug)]
pub struct PowerProps {
    /// Power generation capacity in Watts. 0 for consumers.
    pub generation_w: f64,
    /// Power consumption at full load in Watts. 0 for sources.
    pub consumption_w: f64,
    /// Energy storage capacity in Joules. 0 for non-batteries.
    pub storage_j: f64,
    /// Maximum charge/discharge rate in Watts. 0 for non-batteries.
    pub charge_rate_w: f64,
    /// Wireless power broadcast range in blocks. 0 for non-sources.
    pub broadcast_range: f32,
}

/// What an interaction does when triggered.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum InteractionType {
    /// Simple on/off toggle (seat, reactor, shield generator).
    Toggle,
    /// Cycle through discrete states (rail junction direction).
    CycleState,
    /// Open a UI panel on the client (signal converter rules, computer terminal).
    OpenUI,
    /// Custom interaction dispatched to a kind-specific handler.
    Custom(u8),
}

/// A single interaction that a functional block supports.
#[derive(Clone, Debug)]
pub struct InteractionDef {
    /// Input action code that triggers this (3=E key, 8=F key, etc.)
    pub action_key: u8,
    /// What this interaction does.
    pub interaction_type: InteractionType,
    /// Display label for client HUD prompt ("Sit", "Toggle Power", "Open Config").
    pub label: &'static str,
}

/// All interactions a block kind supports.
pub struct InteractionSchema {
    pub actions: &'static [InteractionDef],
}

/// No interactions — used for functional blocks that have no player-facing interaction.
static SCHEMA_NONE: InteractionSchema = InteractionSchema { actions: &[] };

static SCHEMA_SEAT: InteractionSchema = InteractionSchema {
    actions: &[InteractionDef { action_key: 3, interaction_type: InteractionType::Toggle, label: "Sit" }],
};

static SCHEMA_TOGGLE: InteractionSchema = InteractionSchema {
    actions: &[InteractionDef { action_key: 3, interaction_type: InteractionType::Toggle, label: "Toggle" }],
};

static SCHEMA_PROGRAM: InteractionSchema = InteractionSchema {
    actions: &[InteractionDef { action_key: 3, interaction_type: InteractionType::OpenUI, label: "Program" }],
};

static SCHEMA_JUNCTION: InteractionSchema = InteractionSchema {
    actions: &[
        InteractionDef { action_key: 3, interaction_type: InteractionType::CycleState, label: "Switch" },
    ],
};

// ---------------------------------------------------------------------------
// Block Registry
// ---------------------------------------------------------------------------

/// The global block property table. Indexed by `BlockId::as_u16()`.
///
/// Constructed once at startup via `BlockRegistry::new()`. Both server and
/// client build identical instances from the same static data.
pub struct BlockRegistry {
    defs: Vec<BlockDef>,
    /// Functional block kind per block ID. None = not functional (structural/natural block).
    functional_kinds: Vec<Option<FunctionalBlockKind>>,
    /// Per-thruster-block properties. None = not a thruster.
    thruster_props: Vec<Option<ThrusterProps>>,
    /// Per-block power properties. None = not power-relevant.
    power_props: Vec<Option<PowerProps>>,
    /// Per-cruise-drive properties. None = not a cruise drive.
    cruise_drive_props: Vec<Option<CruiseDriveProps>>,
    /// Per-mechanical-mount properties. None = not a mechanical mount.
    mechanical_props: Vec<Option<MechanicalProps>>,
    /// Per-light-emitter properties. `None` for blocks that don't emit
    /// dynamic light. Drives the client's per-block `PointLight` /
    /// `SpotLight` spawning.
    light_specs: Vec<Option<LightSpec>>,
}

impl BlockRegistry {
    /// Build the registry with all defined block types.
    pub fn new() -> Self {
        let mut defs = Vec::with_capacity(MAX_BLOCK_TYPES);
        defs.resize_with(MAX_BLOCK_TYPES, || BlockDef::UNDEFINED.clone());
        let mut functional_kinds = vec![None; MAX_BLOCK_TYPES];
        let mut thruster_props_vec: Vec<Option<ThrusterProps>> = vec![None; MAX_BLOCK_TYPES];
        let mut power_props_vec: Vec<Option<PowerProps>> = vec![None; MAX_BLOCK_TYPES];
        let mut cruise_drive_props_vec: Vec<Option<CruiseDriveProps>> = vec![None; MAX_BLOCK_TYPES];
        let mut mechanical_props_vec: Vec<Option<MechanicalProps>> = vec![None; MAX_BLOCK_TYPES];
        let mut light_specs_vec: Vec<Option<LightSpec>> = vec![None; MAX_BLOCK_TYPES];

        let r = &mut defs;

        // =================================================================
        // Basic (0–5)
        // =================================================================
        set(r, BlockId::AIR, BlockDef {
            name: "air",
            is_solid: false,
            is_transparent: true,
            density: 0,
            hardness: 0,
            color_hint: [0, 0, 0],
            ..BlockDef::UNDEFINED
        });
        set(r, BlockId::STONE, BlockDef {
            name: "stone",
            is_solid: true,
            is_transparent: false,
            material_type: MaterialType::Stone,
            density: 2700,
            hardness: 30,
            mining_tier: 1,
            preferred_tool: ToolType::Pickaxe,
            blast_resistance: 30,
            color_hint: [128, 128, 128],
            ..BlockDef::UNDEFINED
        });
        set(r, BlockId::DIRT, BlockDef {
            name: "dirt",
            is_solid: true,
            is_transparent: false,
            material_type: MaterialType::Soil,
            density: 1500,
            hardness: 5,
            preferred_tool: ToolType::Shovel,
            blast_resistance: 5,
            color_hint: [139, 90, 43],
            ..BlockDef::UNDEFINED
        });
        set(r, BlockId::GRASS, BlockDef {
            name: "grass",
            is_solid: true,
            is_transparent: false,
            material_type: MaterialType::Soil,
            density: 1500,
            hardness: 5,
            preferred_tool: ToolType::Shovel,
            blast_resistance: 5,
            color_hint: [86, 152, 42],
            ..BlockDef::UNDEFINED
        });
        set(r, BlockId::SAND, BlockDef {
            name: "sand",
            is_solid: true,
            is_transparent: false,
            gravity_affected: true,
            material_type: MaterialType::Soil,
            density: 1600,
            hardness: 3,
            preferred_tool: ToolType::Shovel,
            blast_resistance: 3,
            color_hint: [219, 201, 146],
            ..BlockDef::UNDEFINED
        });
        set(r, BlockId::WATER, BlockDef {
            name: "water",
            is_solid: false,
            is_transparent: true,
            material_type: MaterialType::Fluid,
            density: 1000,
            hardness: 0,
            color_hint: [32, 96, 200],
            liquid: LiquidProperties {
                gravity_multiplier: 0.4,
                drag: 0.8,
                swim_force: 10.0,
            },
            ..BlockDef::UNDEFINED
        });

        // =================================================================
        // Stone variants (6–13)
        // =================================================================
        set(r, BlockId::GRANITE, stone_variant("granite", 2750, [160, 120, 110]));
        set(r, BlockId::BASALT, stone_variant("basalt", 2900, [50, 50, 55]));
        set(r, BlockId::LIMESTONE, stone_variant("limestone", 2500, [210, 200, 180]));
        set(r, BlockId::MARBLE, stone_variant("marble", 2700, [230, 225, 220]));
        set(r, BlockId::SLATE, stone_variant("slate", 2750, [80, 80, 90]));
        set(r, BlockId::SANDSTONE, stone_variant("sandstone", 2300, [200, 170, 120]));
        set(r, BlockId::OBSIDIAN, BlockDef {
            name: "obsidian",
            is_solid: true,
            is_transparent: false,
            material_type: MaterialType::Stone,
            density: 2600,
            hardness: 255,
            mining_tier: 4,
            preferred_tool: ToolType::Pickaxe,
            blast_resistance: 255,
            color_hint: [15, 10, 20],
            ..BlockDef::UNDEFINED
        });
        set(r, BlockId::PUMICE, BlockDef {
            name: "pumice",
            is_solid: true,
            is_transparent: false,
            material_type: MaterialType::Stone,
            density: 800,
            hardness: 10,
            mining_tier: 0,
            preferred_tool: ToolType::Pickaxe,
            blast_resistance: 8,
            color_hint: [190, 180, 170],
            ..BlockDef::UNDEFINED
        });

        // =================================================================
        // Soil / Surface (20–25)
        // =================================================================
        set(r, BlockId::CLAY, soil("clay", 1800, [170, 140, 120]));
        set(r, BlockId::MUD, soil("mud", 1600, [90, 70, 50]));
        set(r, BlockId::GRAVEL, BlockDef {
            name: "gravel",
            is_solid: true,
            is_transparent: false,
            gravity_affected: true,
            material_type: MaterialType::Soil,
            density: 1800,
            hardness: 6,
            preferred_tool: ToolType::Shovel,
            blast_resistance: 6,
            color_hint: [140, 130, 125],
            ..BlockDef::UNDEFINED
        });
        set(r, BlockId::PEAT, soil("peat", 1100, [60, 40, 25]));
        set(r, BlockId::RED_SAND, BlockDef {
            name: "red_sand",
            is_solid: true,
            is_transparent: false,
            gravity_affected: true,
            material_type: MaterialType::Soil,
            density: 1600,
            hardness: 3,
            preferred_tool: ToolType::Shovel,
            blast_resistance: 3,
            color_hint: [190, 100, 50],
            ..BlockDef::UNDEFINED
        });
        set(r, BlockId::DARK_SOIL, soil("dark_soil", 1500, [50, 35, 20]));

        // =================================================================
        // Ice / Snow (30–33)
        // =================================================================
        set(r, BlockId::SNOW, BlockDef {
            name: "snow",
            is_solid: true,
            is_transparent: false,
            material_type: MaterialType::Ice,
            density: 300,
            hardness: 2,
            preferred_tool: ToolType::Shovel,
            blast_resistance: 2,
            color_hint: [240, 245, 255],
            ..BlockDef::UNDEFINED
        });
        set(r, BlockId::PACKED_ICE, ice("packed_ice", 900, 15, [160, 200, 230]));
        set(r, BlockId::BLUE_ICE, ice("blue_ice", 920, 20, [100, 150, 220]));
        set(r, BlockId::PERMAFROST, BlockDef {
            name: "permafrost",
            is_solid: true,
            is_transparent: false,
            material_type: MaterialType::Ice,
            density: 1800,
            hardness: 25,
            mining_tier: 1,
            preferred_tool: ToolType::Pickaxe,
            blast_resistance: 20,
            color_hint: [140, 155, 165],
            ..BlockDef::UNDEFINED
        });

        // =================================================================
        // Volcanic (40–42)
        // =================================================================
        set(r, BlockId::MAGMA_ROCK, BlockDef {
            name: "magma_rock",
            is_solid: true,
            is_transparent: false,
            material_type: MaterialType::Stone,
            density: 3000,
            hardness: 40,
            mining_tier: 2,
            preferred_tool: ToolType::Pickaxe,
            blast_resistance: 40,
            color_hint: [60, 20, 10],
            light_emission: 3,
            fire_resistance: 255,
            ..BlockDef::UNDEFINED
        });
        set(r, BlockId::ASH, BlockDef {
            name: "ash",
            is_solid: true,
            is_transparent: false,
            gravity_affected: true,
            material_type: MaterialType::Soil,
            density: 600,
            hardness: 2,
            preferred_tool: ToolType::Shovel,
            blast_resistance: 2,
            color_hint: [100, 95, 90],
            ..BlockDef::UNDEFINED
        });
        set(r, BlockId::SULFUR, BlockDef {
            name: "sulfur",
            is_solid: true,
            is_transparent: false,
            material_type: MaterialType::Crystal,
            density: 2000,
            hardness: 12,
            mining_tier: 1,
            preferred_tool: ToolType::Pickaxe,
            blast_resistance: 10,
            color_hint: [200, 190, 50],
            flammability: 200,
            ..BlockDef::UNDEFINED
        });

        // =================================================================
        // Ores (100–111)
        // =================================================================
        set(r, BlockId::COAL_ORE, ore("coal_ore", 1400, 15, 1, [40, 35, 30]));
        set(r, BlockId::IRON_ORE, ore("iron_ore", 3500, 25, 1, [160, 130, 110]));
        set(r, BlockId::COPPER_ORE, ore("copper_ore", 3200, 22, 1, [170, 110, 60]));
        set(r, BlockId::GOLD_ORE, ore("gold_ore", 5000, 30, 2, [220, 190, 50]));
        set(r, BlockId::DIAMOND_ORE, ore("diamond_ore", 3500, 60, 3, [100, 220, 230]));
        set(r, BlockId::EMERALD_ORE, ore("emerald_ore", 2800, 55, 3, [40, 180, 80]));
        set(r, BlockId::RUBY_ORE, ore("ruby_ore", 3200, 55, 3, [200, 30, 50]));
        set(r, BlockId::SAPPHIRE_ORE, ore("sapphire_ore", 3200, 55, 3, [30, 60, 200]));
        set(r, BlockId::URANIUM_ORE, ore("uranium_ore", 4000, 40, 3, [80, 180, 60]));
        set(r, BlockId::TITANIUM_ORE, ore("titanium_ore", 4500, 50, 3, [180, 190, 200]));
        set(r, BlockId::MYTHRIL_ORE, ore("mythril_ore", 3000, 80, 4, [120, 160, 230]));
        set(r, BlockId::ENERGY_CRYSTAL, BlockDef {
            name: "energy_crystal",
            is_solid: true,
            is_transparent: false,
            material_type: MaterialType::Crystal,
            density: 2500,
            hardness: 70,
            mining_tier: 4,
            preferred_tool: ToolType::Pickaxe,
            blast_resistance: 50,
            color_hint: [200, 100, 255],
            light_emission: 10,
            ..BlockDef::UNDEFINED
        });

        // =================================================================
        // Fluids (200–202)
        // =================================================================
        set(r, BlockId::LAVA, BlockDef {
            name: "lava",
            is_solid: false,
            is_transparent: true,
            material_type: MaterialType::Fluid,
            density: 3100,
            hardness: 0,
            color_hint: [220, 90, 10],
            light_emission: 15,
            fire_resistance: 255,
            liquid: LiquidProperties {
                gravity_multiplier: 0.2,
                drag: 3.0,
                swim_force: 4.0,
            },
            ..BlockDef::UNDEFINED
        });
        set(r, BlockId::SALT_WATER, BlockDef {
            name: "salt_water",
            is_solid: false,
            is_transparent: true,
            material_type: MaterialType::Fluid,
            density: 1025,
            hardness: 0,
            color_hint: [20, 80, 160],
            liquid: LiquidProperties {
                gravity_multiplier: 0.35,
                drag: 0.9,
                swim_force: 11.0,
            },
            ..BlockDef::UNDEFINED
        });
        set(r, BlockId::OIL, BlockDef {
            name: "oil",
            is_solid: false,
            is_transparent: true,
            material_type: MaterialType::Fluid,
            density: 870,
            hardness: 0,
            color_hint: [30, 25, 20],
            flammability: 250,
            liquid: LiquidProperties {
                gravity_multiplier: 0.3,
                drag: 1.5,
                swim_force: 8.0,
            },
            ..BlockDef::UNDEFINED
        });

        // =================================================================
        // Vegetation (300–313)
        // =================================================================
        set(r, BlockId::OAK_LOG, log("oak_log", [120, 80, 40]));
        set(r, BlockId::OAK_LEAVES, leaves("oak_leaves", [50, 120, 30]));
        set(r, BlockId::PINE_LOG, log("pine_log", [100, 70, 35]));
        set(r, BlockId::PINE_LEAVES, leaves("pine_leaves", [30, 90, 40]));
        set(r, BlockId::JUNGLE_LOG, log("jungle_log", [130, 90, 50]));
        set(r, BlockId::JUNGLE_LEAVES, leaves("jungle_leaves", [20, 100, 20]));
        set(r, BlockId::BIRCH_LOG, log("birch_log", [200, 195, 180]));
        set(r, BlockId::BIRCH_LEAVES, leaves("birch_leaves", [80, 140, 50]));
        set(r, BlockId::ACACIA_LOG, log("acacia_log", [160, 100, 60]));
        set(r, BlockId::ACACIA_LEAVES, leaves("acacia_leaves", [100, 150, 30]));
        set(r, BlockId::TALL_GRASS, BlockDef {
            name: "tall_grass",
            is_solid: false,
            is_transparent: true,
            material_type: MaterialType::Organic,
            density: 50,
            hardness: 0,
            color_hint: [70, 140, 40],
            flammability: 200,
            ..BlockDef::UNDEFINED
        });
        set(r, BlockId::MOSS, BlockDef {
            name: "moss",
            is_solid: false,
            is_transparent: true,
            material_type: MaterialType::Organic,
            density: 100,
            hardness: 1,
            color_hint: [50, 110, 40],
            flammability: 150,
            ..BlockDef::UNDEFINED
        });
        set(r, BlockId::CACTUS, BlockDef {
            name: "cactus",
            is_solid: true,
            is_transparent: false,
            material_type: MaterialType::Organic,
            density: 500,
            hardness: 4,
            color_hint: [40, 130, 50],
            ..BlockDef::UNDEFINED
        });
        set(r, BlockId::DEAD_BUSH, BlockDef {
            name: "dead_bush",
            is_solid: false,
            is_transparent: true,
            material_type: MaterialType::Organic,
            density: 30,
            hardness: 0,
            color_hint: [150, 120, 60],
            flammability: 250,
            ..BlockDef::UNDEFINED
        });

        // =================================================================
        // Structural (1700–1799)
        // =================================================================
        set(r, BlockId::HULL_LIGHT, hull("hull_light", 2000, 40, 40, [180, 185, 190]));
        set(r, BlockId::HULL_STANDARD, hull("hull_standard", 4000, 80, 80, [150, 155, 160]));
        set(r, BlockId::HULL_HEAVY, hull("hull_heavy", 6000, 140, 140, [120, 125, 130]));
        set(r, BlockId::HULL_ARMORED, hull("hull_armored", 8000, 220, 220, [90, 95, 100]));
        set(r, BlockId::WINDOW, BlockDef {
            name: "window",
            is_solid: true,
            is_transparent: true,
            material_type: MaterialType::Glass,
            density: 2500,
            hardness: 15,
            mining_tier: 0,
            preferred_tool: ToolType::None,
            blast_resistance: 5,
            color_hint: [200, 220, 240],
            ..BlockDef::UNDEFINED
        });
        set(r, BlockId::DOOR, BlockDef {
            name: "door",
            is_solid: true,
            is_transparent: false,
            material_type: MaterialType::Metal,
            density: 3000,
            hardness: 60,
            mining_tier: 1,
            preferred_tool: ToolType::None,
            blast_resistance: 60,
            color_hint: [100, 105, 110],
            ..BlockDef::UNDEFINED
        });

        // (Light fixtures used to live as full blocks at IDs 1730–1733.
        // Phase 8 v3 moved them to sub-block elements
        // (`SubBlockType::SurfaceLight` and friends) so they sit flush
        // against host-block faces rather than occupying a full 1 m
        // cube. See `core/src/block/sub_block.rs` and
        // `BlockRegistry::sub_block_light_spec`.)

        // =================================================================
        // Utility (1800–1899) — basic block properties only.
        // Functional behavior defined in the functional block registry.
        // =================================================================
        set(r, BlockId::COCKPIT, functional_block("cockpit", 2000, 50, [60, 80, 100]));
        set(r, BlockId::SEAT, functional_block("seat", 500, 20, [180, 180, 180]));
        set(r, BlockId::OWNERSHIP_CORE, functional_block("ownership_core", 5000, 200, [255, 215, 0]));

        // Ship system blocks
        set(r, BlockId::FLIGHT_COMPUTER, functional_block("flight_computer", 800, 30, [40, 120, 200]));
        set(r, BlockId::HOVER_MODULE, functional_block("hover_module", 1000, 40, [60, 180, 140]));
        set(r, BlockId::AUTOPILOT, functional_block("autopilot", 1200, 40, [180, 140, 60]));
        set(r, BlockId::WARP_COMPUTER, functional_block("warp_computer", 1500, 50, [140, 60, 200]));
        set(r, BlockId::ENGINE_CONTROLLER, functional_block("engine_controller", 600, 25, [200, 80, 40]));
        set(r, BlockId::HUD_PANEL, BlockDef {
            name: "hud_panel",
            is_solid: true,
            is_transparent: true,
            material_type: MaterialType::Glass,
            density: 2000,
            hardness: 20,
            blast_resistance: 10,
            color_hint: [180, 210, 240],
            ..BlockDef::UNDEFINED
        });
        set(r, BlockId::GRAVITY_GENERATOR, functional_block("gravity_generator", 6000, 100, [80, 60, 140]));
        set(r, BlockId::ANTENNA, functional_block("antenna", 1500, 30, [200, 200, 210]));

        // =================================================================
        // Propulsion (1000–1099)
        // =================================================================
        set(r, BlockId::THRUSTER_SMALL_CHEMICAL, functional_block("thruster_small_chemical", 500, 30, [200, 100, 50]));
        set(r, BlockId::THRUSTER_MEDIUM_CHEMICAL, functional_block("thruster_medium_chemical", 1500, 50, [200, 100, 50]));
        set(r, BlockId::THRUSTER_LARGE_CHEMICAL, functional_block("thruster_large_chemical", 4000, 80, [200, 100, 50]));
        set(r, BlockId::THRUSTER_SMALL_ION, functional_block("thruster_small_ion", 400, 25, [100, 150, 220]));
        set(r, BlockId::THRUSTER_MEDIUM_ION, functional_block("thruster_medium_ion", 1200, 45, [100, 150, 220]));
        set(r, BlockId::THRUSTER_LARGE_ION, functional_block("thruster_large_ion", 3500, 70, [100, 150, 220]));
        set(r, BlockId::THRUSTER_SMALL_FUSION, functional_block("thruster_small_fusion", 600, 40, [180, 80, 220]));
        set(r, BlockId::THRUSTER_MEDIUM_FUSION, functional_block("thruster_medium_fusion", 2000, 65, [180, 80, 220]));
        set(r, BlockId::THRUSTER_LARGE_FUSION, functional_block("thruster_large_fusion", 5000, 100, [180, 80, 220]));

        // =================================================================
        // Power (1100–1199)
        // =================================================================
        set(r, BlockId::REACTOR_SMALL, functional_block("reactor_small", 3000, 60, [50, 180, 80]));
        set(r, BlockId::REACTOR_MEDIUM, functional_block("reactor_medium", 8000, 100, [50, 180, 80]));
        set(r, BlockId::REACTOR_LARGE, functional_block("reactor_large", 15000, 160, [50, 180, 80]));
        set(r, BlockId::BATTERY, functional_block("battery", 2000, 40, [220, 180, 40]));
        set(r, BlockId::SOLAR_PANEL, BlockDef {
            name: "solar_panel",
            is_solid: true,
            is_transparent: false,
            material_type: MaterialType::Metal,
            density: 800,
            hardness: 15,
            blast_resistance: 10,
            color_hint: [30, 40, 100],
            ..BlockDef::UNDEFINED
        });
        set(r, BlockId::POWER_CONDUIT, functional_block("power_conduit", 1000, 20, [200, 160, 40]));

        // =================================================================
        // Life Support (1200–1299)
        // =================================================================
        set(r, BlockId::AIR_COMPRESSOR, functional_block("air_compressor", 2500, 50, [150, 200, 220]));
        set(r, BlockId::AIR_VENT, functional_block("air_vent", 1000, 20, [170, 190, 200]));

        // =================================================================
        // Shields (1300–1399)
        // =================================================================
        set(r, BlockId::SHIELD_EMITTER, functional_block("shield_emitter", 3000, 70, [80, 140, 220]));
        set(r, BlockId::SHIELD_GENERATOR, functional_block("shield_generator", 5000, 100, [60, 120, 200]));

        // =================================================================
        // Mechanical (1400–1499)
        // =================================================================
        set(r, BlockId::ROTOR, functional_block("rotor", 4000, 80, [160, 160, 170]));
        set(r, BlockId::PISTON, functional_block("piston", 4000, 80, [170, 160, 160]));

        // =================================================================
        // Transport (1500–1599)
        // =================================================================
        set(r, BlockId::RAIL_STRAIGHT, functional_block("rail_straight", 2000, 40, [140, 130, 120]));
        set(r, BlockId::RAIL_CURVE, functional_block("rail_curve", 2000, 40, [140, 130, 120]));
        set(r, BlockId::RAIL_JUNCTION, functional_block("rail_junction", 2500, 50, [140, 130, 120]));
        set(r, BlockId::RAIL_SIGNAL, functional_block("rail_signal", 1000, 20, [200, 60, 60]));

        // =================================================================
        // Logic (1600–1699)
        // =================================================================
        set(r, BlockId::SIGNAL_CONVERTER, functional_block("signal_converter", 1500, 30, [220, 200, 100]));
        set(r, BlockId::SENSOR_PRESSURE, functional_block("sensor_pressure", 800, 20, [100, 200, 180]));
        set(r, BlockId::SENSOR_PROXIMITY, functional_block("sensor_proximity", 800, 20, [100, 200, 180]));
        set(r, BlockId::SENSOR_SPEED, functional_block("sensor_speed", 800, 20, [100, 200, 180]));
        set(r, BlockId::SENSOR_POWER, functional_block("sensor_power", 800, 20, [100, 200, 180]));
        set(r, BlockId::SENSOR_DAMAGE, functional_block("sensor_damage", 800, 20, [100, 200, 180]));
        set(r, BlockId::COMPUTER, functional_block("computer", 2000, 40, [40, 50, 60]));
        // Cruise drives (propulsion boost modules)
        set(r, BlockId::CRUISE_DRIVE_SMALL, functional_block("cruise_drive_small", 3000, 80, [80, 200, 220]));
        set(r, BlockId::CRUISE_DRIVE_MEDIUM, functional_block("cruise_drive_medium", 8000, 120, [80, 200, 220]));
        set(r, BlockId::CRUISE_DRIVE_LARGE, functional_block("cruise_drive_large", 20000, 200, [80, 200, 220]));

        // Populate functional block kinds.
        use FunctionalBlockKind::*;
        let fk = &mut functional_kinds;
        // Propulsion
        for id in [BlockId::THRUSTER_SMALL_CHEMICAL, BlockId::THRUSTER_MEDIUM_CHEMICAL,
            BlockId::THRUSTER_LARGE_CHEMICAL, BlockId::THRUSTER_SMALL_ION,
            BlockId::THRUSTER_MEDIUM_ION, BlockId::THRUSTER_LARGE_ION,
            BlockId::THRUSTER_SMALL_FUSION, BlockId::THRUSTER_MEDIUM_FUSION,
            BlockId::THRUSTER_LARGE_FUSION] {
            fk[id.as_u16() as usize] = Some(Thruster);
        }
        // Cruise drives
        for id in [BlockId::CRUISE_DRIVE_SMALL, BlockId::CRUISE_DRIVE_MEDIUM, BlockId::CRUISE_DRIVE_LARGE] {
            fk[id.as_u16() as usize] = Some(CruiseDrive);
        }
        // Power
        for id in [BlockId::REACTOR_SMALL, BlockId::REACTOR_MEDIUM, BlockId::REACTOR_LARGE] {
            fk[id.as_u16() as usize] = Some(Reactor);
        }
        fk[BlockId::BATTERY.as_u16() as usize] = Some(Battery);
        fk[BlockId::SOLAR_PANEL.as_u16() as usize] = Some(SolarPanel);
        fk[BlockId::POWER_CONDUIT.as_u16() as usize] = Some(PowerConduit);
        // Seats
        fk[BlockId::COCKPIT.as_u16() as usize] = Some(Seat);
        fk[BlockId::SEAT.as_u16() as usize] = Some(Seat);
        // Ship system blocks
        fk[BlockId::FLIGHT_COMPUTER.as_u16() as usize] = Some(FlightComputer);
        fk[BlockId::HOVER_MODULE.as_u16() as usize] = Some(HoverModule);
        fk[BlockId::AUTOPILOT.as_u16() as usize] = Some(Autopilot);
        fk[BlockId::WARP_COMPUTER.as_u16() as usize] = Some(WarpComputer);
        fk[BlockId::ENGINE_CONTROLLER.as_u16() as usize] = Some(EngineController);
        // Utility
        fk[BlockId::GRAVITY_GENERATOR.as_u16() as usize] = Some(GravityGenerator);
        fk[BlockId::ANTENNA.as_u16() as usize] = Some(Antenna);
        // Shields
        fk[BlockId::SHIELD_EMITTER.as_u16() as usize] = Some(ShieldEmitter);
        fk[BlockId::SHIELD_GENERATOR.as_u16() as usize] = Some(ShieldGenerator);
        // Life support
        fk[BlockId::AIR_COMPRESSOR.as_u16() as usize] = Some(AirCompressor);
        // Mechanical
        fk[BlockId::ROTOR.as_u16() as usize] = Some(Rotor);
        fk[BlockId::PISTON.as_u16() as usize] = Some(Piston);
        // Transport
        fk[BlockId::RAIL_STRAIGHT.as_u16() as usize] = Some(Rail);
        fk[BlockId::RAIL_CURVE.as_u16() as usize] = Some(Rail);
        fk[BlockId::RAIL_JUNCTION.as_u16() as usize] = Some(RailJunction);
        fk[BlockId::RAIL_SIGNAL.as_u16() as usize] = Some(RailSignal);
        // Logic
        fk[BlockId::SIGNAL_CONVERTER.as_u16() as usize] = Some(SignalConverter);
        for id in [BlockId::SENSOR_PRESSURE, BlockId::SENSOR_PROXIMITY,
            BlockId::SENSOR_SPEED, BlockId::SENSOR_POWER, BlockId::SENSOR_DAMAGE] {
            fk[id.as_u16() as usize] = Some(Sensor);
        }
        fk[BlockId::COMPUTER.as_u16() as usize] = Some(Computer);

        // Populate thruster-specific properties.
        let tp = &mut thruster_props_vec;
        // Chemical: cheap, moderate thrust, high fuel consumption.
        // Small: 50 kN → 8 on starter ship (10t) = 400 kN total ≈ 40 m/s² (4g).
        tp[BlockId::THRUSTER_SMALL_CHEMICAL.as_u16() as usize] = Some(ThrusterProps { thrust_n: 50_000.0, fuel_rate: 5.0 });
        tp[BlockId::THRUSTER_MEDIUM_CHEMICAL.as_u16() as usize] = Some(ThrusterProps { thrust_n: 200_000.0, fuel_rate: 15.0 });
        tp[BlockId::THRUSTER_LARGE_CHEMICAL.as_u16() as usize] = Some(ThrusterProps { thrust_n: 800_000.0, fuel_rate: 50.0 });
        // Ion: efficient, low thrust, low fuel.
        tp[BlockId::THRUSTER_SMALL_ION.as_u16() as usize] = Some(ThrusterProps { thrust_n: 20_000.0, fuel_rate: 0.5 });
        tp[BlockId::THRUSTER_MEDIUM_ION.as_u16() as usize] = Some(ThrusterProps { thrust_n: 100_000.0, fuel_rate: 2.0 });
        tp[BlockId::THRUSTER_LARGE_ION.as_u16() as usize] = Some(ThrusterProps { thrust_n: 500_000.0, fuel_rate: 8.0 });
        // Fusion: best power/weight ratio, end-game.
        tp[BlockId::THRUSTER_SMALL_FUSION.as_u16() as usize] = Some(ThrusterProps { thrust_n: 100_000.0, fuel_rate: 1.0 });
        tp[BlockId::THRUSTER_MEDIUM_FUSION.as_u16() as usize] = Some(ThrusterProps { thrust_n: 500_000.0, fuel_rate: 4.0 });
        tp[BlockId::THRUSTER_LARGE_FUSION.as_u16() as usize] = Some(ThrusterProps { thrust_n: 2_000_000.0, fuel_rate: 12.0 });

        // Populate power properties.
        let pp = &mut power_props_vec;
        // Reactors: power sources with broadcast range.
        pp[BlockId::REACTOR_SMALL.as_u16() as usize]  = Some(PowerProps { generation_w: 500_000.0,     consumption_w: 0.0, storage_j: 0.0, charge_rate_w: 0.0, broadcast_range: 50.0 });
        pp[BlockId::REACTOR_MEDIUM.as_u16() as usize]  = Some(PowerProps { generation_w: 2_000_000.0,  consumption_w: 0.0, storage_j: 0.0, charge_rate_w: 0.0, broadcast_range: 100.0 });
        pp[BlockId::REACTOR_LARGE.as_u16() as usize]   = Some(PowerProps { generation_w: 10_000_000.0, consumption_w: 0.0, storage_j: 0.0, charge_rate_w: 0.0, broadcast_range: 200.0 });
        pp[BlockId::SOLAR_PANEL.as_u16() as usize]     = Some(PowerProps { generation_w: 50_000.0,     consumption_w: 0.0, storage_j: 0.0, charge_rate_w: 0.0, broadcast_range: 20.0 });
        // Battery: energy storage.
        pp[BlockId::BATTERY.as_u16() as usize]          = Some(PowerProps { generation_w: 0.0, consumption_w: 0.0, storage_j: 36_000_000.0, charge_rate_w: 500_000.0, broadcast_range: 0.0 });
        // Thrusters: power consumers (chemical cheap, ion moderate, fusion efficient).
        pp[BlockId::THRUSTER_SMALL_CHEMICAL.as_u16() as usize]  = Some(PowerProps { generation_w: 0.0, consumption_w: 100_000.0,   storage_j: 0.0, charge_rate_w: 0.0, broadcast_range: 0.0 });
        pp[BlockId::THRUSTER_MEDIUM_CHEMICAL.as_u16() as usize] = Some(PowerProps { generation_w: 0.0, consumption_w: 400_000.0,   storage_j: 0.0, charge_rate_w: 0.0, broadcast_range: 0.0 });
        pp[BlockId::THRUSTER_LARGE_CHEMICAL.as_u16() as usize]  = Some(PowerProps { generation_w: 0.0, consumption_w: 1_500_000.0, storage_j: 0.0, charge_rate_w: 0.0, broadcast_range: 0.0 });
        pp[BlockId::THRUSTER_SMALL_ION.as_u16() as usize]       = Some(PowerProps { generation_w: 0.0, consumption_w: 200_000.0,   storage_j: 0.0, charge_rate_w: 0.0, broadcast_range: 0.0 });
        pp[BlockId::THRUSTER_MEDIUM_ION.as_u16() as usize]      = Some(PowerProps { generation_w: 0.0, consumption_w: 800_000.0,   storage_j: 0.0, charge_rate_w: 0.0, broadcast_range: 0.0 });
        pp[BlockId::THRUSTER_LARGE_ION.as_u16() as usize]       = Some(PowerProps { generation_w: 0.0, consumption_w: 4_000_000.0, storage_j: 0.0, charge_rate_w: 0.0, broadcast_range: 0.0 });
        pp[BlockId::THRUSTER_SMALL_FUSION.as_u16() as usize]    = Some(PowerProps { generation_w: 0.0, consumption_w: 50_000.0,    storage_j: 0.0, charge_rate_w: 0.0, broadcast_range: 0.0 });
        pp[BlockId::THRUSTER_MEDIUM_FUSION.as_u16() as usize]   = Some(PowerProps { generation_w: 0.0, consumption_w: 200_000.0,   storage_j: 0.0, charge_rate_w: 0.0, broadcast_range: 0.0 });
        pp[BlockId::THRUSTER_LARGE_FUSION.as_u16() as usize]    = Some(PowerProps { generation_w: 0.0, consumption_w: 800_000.0,   storage_j: 0.0, charge_rate_w: 0.0, broadcast_range: 0.0 });
        // Other consumers.
        pp[BlockId::SHIELD_EMITTER.as_u16() as usize]    = Some(PowerProps { generation_w: 0.0, consumption_w: 300_000.0,   storage_j: 0.0, charge_rate_w: 0.0, broadcast_range: 0.0 });
        pp[BlockId::SHIELD_GENERATOR.as_u16() as usize]  = Some(PowerProps { generation_w: 0.0, consumption_w: 500_000.0,   storage_j: 0.0, charge_rate_w: 0.0, broadcast_range: 0.0 });
        pp[BlockId::GRAVITY_GENERATOR.as_u16() as usize] = Some(PowerProps { generation_w: 0.0, consumption_w: 200_000.0,   storage_j: 0.0, charge_rate_w: 0.0, broadcast_range: 0.0 });
        pp[BlockId::AIR_COMPRESSOR.as_u16() as usize]    = Some(PowerProps { generation_w: 0.0, consumption_w: 50_000.0,    storage_j: 0.0, charge_rate_w: 0.0, broadcast_range: 0.0 });
        pp[BlockId::ROTOR.as_u16() as usize]             = Some(PowerProps { generation_w: 0.0, consumption_w: 100_000.0,   storage_j: 0.0, charge_rate_w: 0.0, broadcast_range: 0.0 });
        pp[BlockId::PISTON.as_u16() as usize]            = Some(PowerProps { generation_w: 0.0, consumption_w: 80_000.0,    storage_j: 0.0, charge_rate_w: 0.0, broadcast_range: 0.0 });
        pp[BlockId::ANTENNA.as_u16() as usize]           = Some(PowerProps { generation_w: 0.0, consumption_w: 10_000.0,    storage_j: 0.0, charge_rate_w: 0.0, broadcast_range: 0.0 });
        pp[BlockId::COMPUTER.as_u16() as usize]          = Some(PowerProps { generation_w: 0.0, consumption_w: 20_000.0,    storage_j: 0.0, charge_rate_w: 0.0, broadcast_range: 0.0 });
        // Cruise drives: consume power when active. The real power cost comes from boosted thrusters'
        // increased consumption, but the drive itself also draws a base amount to stay operational.
        pp[BlockId::CRUISE_DRIVE_SMALL.as_u16() as usize]  = Some(PowerProps { generation_w: 0.0, consumption_w: 50_000.0,    storage_j: 0.0, charge_rate_w: 0.0, broadcast_range: 0.0 });
        pp[BlockId::CRUISE_DRIVE_MEDIUM.as_u16() as usize] = Some(PowerProps { generation_w: 0.0, consumption_w: 200_000.0,   storage_j: 0.0, charge_rate_w: 0.0, broadcast_range: 0.0 });
        pp[BlockId::CRUISE_DRIVE_LARGE.as_u16() as usize]  = Some(PowerProps { generation_w: 0.0, consumption_w: 1_000_000.0, storage_j: 0.0, charge_rate_w: 0.0, broadcast_range: 0.0 });

        // Ship system blocks — power consumption when active.
        pp[BlockId::FLIGHT_COMPUTER.as_u16() as usize]    = Some(PowerProps { generation_w: 0.0, consumption_w: 5_000.0,     storage_j: 0.0, charge_rate_w: 0.0, broadcast_range: 0.0 });
        pp[BlockId::HOVER_MODULE.as_u16() as usize]        = Some(PowerProps { generation_w: 0.0, consumption_w: 20_000.0,    storage_j: 0.0, charge_rate_w: 0.0, broadcast_range: 0.0 });
        pp[BlockId::AUTOPILOT.as_u16() as usize]           = Some(PowerProps { generation_w: 0.0, consumption_w: 5_000.0,     storage_j: 0.0, charge_rate_w: 0.0, broadcast_range: 0.0 });
        pp[BlockId::WARP_COMPUTER.as_u16() as usize]       = Some(PowerProps { generation_w: 0.0, consumption_w: 10_000.0,    storage_j: 0.0, charge_rate_w: 0.0, broadcast_range: 0.0 });
        pp[BlockId::ENGINE_CONTROLLER.as_u16() as usize]   = Some(PowerProps { generation_w: 0.0, consumption_w: 2_000.0,     storage_j: 0.0, charge_rate_w: 0.0, broadcast_range: 0.0 });

        // Cruise drive boost multipliers: Small=500x, Medium=2000x, Large=8000x.
        // At 500x boost, 8 × 1.25 MN chemical thrusters produce 8 × 625 MN = 5 GN
        // on a 2000t ship = 2500 m/s², reaching 100 km/s in 40 seconds.
        let cd = &mut cruise_drive_props_vec;
        cd[BlockId::CRUISE_DRIVE_SMALL.as_u16() as usize]  = Some(CruiseDriveProps { boost_multiplier: 500.0 });
        cd[BlockId::CRUISE_DRIVE_MEDIUM.as_u16() as usize] = Some(CruiseDriveProps { boost_multiplier: 2000.0 });
        cd[BlockId::CRUISE_DRIVE_LARGE.as_u16() as usize]  = Some(CruiseDriveProps { boost_multiplier: 8000.0 });

        // Mechanical mount properties.
        let mp = &mut mechanical_props_vec;
        mp[BlockId::ROTOR.as_u16() as usize] = Some(MechanicalProps {
            joint_type: JointType::Revolute,
            max_force: 10_000.0,    // 10 kN·m torque (strong enough to track fast targets)
            max_speed: 360.0,       // 360 deg/s (1 full rotation/s at full input)
            max_range: 360.0,       // full rotation (no limits)
        });
        mp[BlockId::PISTON.as_u16() as usize] = Some(MechanicalProps {
            joint_type: JointType::Prismatic,
            max_force: 5_000.0,     // 5 kN force (hydraulic actuator scale)
            max_speed: 2.0,         // 2 m/s
            max_range: 10.0,        // 10 meter max extension
        });

        // ─── Light specifications ────────────────────────────────────────
        //
        // Photometric values calibrated against Bevy 0.18's exposure
        // scale. The engine's `PointLight::default()` is 1 000 000 lm
        // ("very large cinema light"); the docs explicitly note that
        // "for indoor lighting with a lower exposure this would be way
        // too bright" — implying that against the AAA outdoor exposure
        // we ship (`ev100 = 11`, ≈ overcast-day target) realistic ship
        // interior fixtures need to sit in the tens-to-hundreds of
        // thousands of lumens, not residential household scale.
        //
        // Reference: a real industrial LED panel emits ~5 000 lm; under
        // an outdoor-calibrated camera that reads as roughly 5 cd/m² at
        // 2 m distance — invisibly dim against any non-zero ambient
        // floor. Bevy's exposure-aware scale needs ~50 000 lm to sit at
        // a comfortable indoor luminance under outdoor exposure.
        let ls = &mut light_specs_vec;

        // (Lamp specs moved to `BlockRegistry::sub_block_light_spec`
        // for the sub-block-mounted variants.)

        // Thrusters — plasma exhaust glow. Brighter classes scale up.
        // Photometric values calibrated against Bevy's exposure scale
        // (see lamp section above for the rationale): tens of thousands
        // of lumens for a small fixture-class light. Real rocket plumes
        // emit > 10⁹ lm at the throat — our values are conservative
        // pixel-budget approximations.
        // Chemical: orange-red flame plume (~2200 K, fuel-rich combustion).
        ls[BlockId::THRUSTER_SMALL_CHEMICAL.as_u16() as usize] = Some(LightSpec {
            lumens: 80_000.0,
            color_kelvin: 2200.0,
            tint_linear_rgb: [1.0, 0.55, 0.20],   // orange flame
            range_m: 8.0,
            beam_kind: BeamKind::PointOmni,
            shadow_caster_class: 0,                // exhaust is fill, not shadow
            emissive_radiance_w_per_m2: 12_000.0,
            emissive_geometry: EmissiveGeometry::DirectionalFace,
        });
        ls[BlockId::THRUSTER_MEDIUM_CHEMICAL.as_u16() as usize] = Some(LightSpec {
            lumens: 250_000.0,
            color_kelvin: 2200.0,
            tint_linear_rgb: [1.0, 0.55, 0.20],
            range_m: 12.0,
            beam_kind: BeamKind::PointOmni,
            shadow_caster_class: 0,
            emissive_radiance_w_per_m2: 24_000.0,
            emissive_geometry: EmissiveGeometry::DirectionalFace,
        });
        ls[BlockId::THRUSTER_LARGE_CHEMICAL.as_u16() as usize] = Some(LightSpec {
            lumens: 800_000.0,
            color_kelvin: 2200.0,
            tint_linear_rgb: [1.0, 0.55, 0.20],
            range_m: 18.0,
            beam_kind: BeamKind::PointOmni,
            shadow_caster_class: 0,
            emissive_radiance_w_per_m2: 60_000.0,
            emissive_geometry: EmissiveGeometry::DirectionalFace,
        });
        // Ion: cool blue-violet plume (~10 000 K equivalent, ionised xenon).
        ls[BlockId::THRUSTER_SMALL_ION.as_u16() as usize] = Some(LightSpec {
            lumens: 50_000.0,
            color_kelvin: 10_000.0,
            tint_linear_rgb: [0.55, 0.65, 1.0],
            range_m: 6.0,
            beam_kind: BeamKind::PointOmni,
            shadow_caster_class: 0,
            emissive_radiance_w_per_m2: 8_000.0,
            emissive_geometry: EmissiveGeometry::DirectionalFace,
        });
        ls[BlockId::THRUSTER_MEDIUM_ION.as_u16() as usize] = Some(LightSpec {
            lumens: 150_000.0,
            color_kelvin: 10_000.0,
            tint_linear_rgb: [0.55, 0.65, 1.0],
            range_m: 9.0,
            beam_kind: BeamKind::PointOmni,
            shadow_caster_class: 0,
            emissive_radiance_w_per_m2: 16_000.0,
            emissive_geometry: EmissiveGeometry::DirectionalFace,
        });
        ls[BlockId::THRUSTER_LARGE_ION.as_u16() as usize] = Some(LightSpec {
            lumens: 500_000.0,
            color_kelvin: 10_000.0,
            tint_linear_rgb: [0.55, 0.65, 1.0],
            range_m: 14.0,
            beam_kind: BeamKind::PointOmni,
            shadow_caster_class: 0,
            emissive_radiance_w_per_m2: 40_000.0,
            emissive_geometry: EmissiveGeometry::DirectionalFace,
        });
        // Fusion: brilliant near-white core (D-T fusion plasma).
        ls[BlockId::THRUSTER_SMALL_FUSION.as_u16() as usize] = Some(LightSpec {
            lumens: 200_000.0,
            color_kelvin: 6500.0,
            tint_linear_rgb: [1.0, 1.0, 1.0],
            range_m: 8.0,
            beam_kind: BeamKind::PointOmni,
            shadow_caster_class: 0,
            emissive_radiance_w_per_m2: 30_000.0,
            emissive_geometry: EmissiveGeometry::DirectionalFace,
        });
        ls[BlockId::THRUSTER_MEDIUM_FUSION.as_u16() as usize] = Some(LightSpec {
            lumens: 700_000.0,
            color_kelvin: 6500.0,
            tint_linear_rgb: [1.0, 1.0, 1.0],
            range_m: 12.0,
            beam_kind: BeamKind::PointOmni,
            shadow_caster_class: 0,
            emissive_radiance_w_per_m2: 80_000.0,
            emissive_geometry: EmissiveGeometry::DirectionalFace,
        });
        ls[BlockId::THRUSTER_LARGE_FUSION.as_u16() as usize] = Some(LightSpec {
            lumens: 2_500_000.0,
            color_kelvin: 6500.0,
            tint_linear_rgb: [1.0, 1.0, 1.0],
            range_m: 22.0,
            beam_kind: BeamKind::PointOmni,
            shadow_caster_class: 0,
            emissive_radiance_w_per_m2: 200_000.0,
            emissive_geometry: EmissiveGeometry::DirectionalFace,
        });

        // Reactors — soft warm glow from the core. Used as fill around
        // the engine room; less intense than lamps because the player
        // doesn't read by reactor light, but enough to define the engine
        // bay's shape.
        ls[BlockId::REACTOR_SMALL.as_u16() as usize] = Some(LightSpec {
            lumens: 30_000.0,
            color_kelvin: 4500.0,
            tint_linear_rgb: [1.0, 0.95, 0.85],
            range_m: 8.0,
            beam_kind: BeamKind::PointOmni,
            shadow_caster_class: 1,
            emissive_radiance_w_per_m2: 1_500.0,
            emissive_geometry: EmissiveGeometry::FullCube,
        });
        ls[BlockId::REACTOR_MEDIUM.as_u16() as usize] = Some(LightSpec {
            lumens: 100_000.0,
            color_kelvin: 4500.0,
            tint_linear_rgb: [1.0, 0.95, 0.85],
            range_m: 12.0,
            beam_kind: BeamKind::PointOmni,
            shadow_caster_class: 1,
            emissive_radiance_w_per_m2: 4_000.0,
            emissive_geometry: EmissiveGeometry::FullCube,
        });
        ls[BlockId::REACTOR_LARGE.as_u16() as usize] = Some(LightSpec {
            lumens: 350_000.0,
            color_kelvin: 4500.0,
            tint_linear_rgb: [1.0, 0.95, 0.85],
            range_m: 18.0,
            beam_kind: BeamKind::PointOmni,
            shadow_caster_class: 1,
            emissive_radiance_w_per_m2: 8_000.0,
            emissive_geometry: EmissiveGeometry::FullCube,
        });

        Self {
            defs, functional_kinds,
            thruster_props: thruster_props_vec,
            power_props: power_props_vec,
            cruise_drive_props: cruise_drive_props_vec,
            mechanical_props: mechanical_props_vec,
            light_specs: light_specs_vec,
        }
    }

    /// O(1) lookup by `BlockId`.
    #[inline(always)]
    pub fn get(&self, id: BlockId) -> &BlockDef {
        &self.defs[id.as_u16() as usize]
    }

    #[inline(always)]
    pub fn is_solid(&self, id: BlockId) -> bool {
        self.get(id).is_solid
    }

    #[inline(always)]
    pub fn is_transparent(&self, id: BlockId) -> bool {
        self.get(id).is_transparent
    }

    /// Whether this block type is a functional block (has ECS entity lifecycle).
    #[inline(always)]
    pub fn is_functional(&self, id: BlockId) -> bool {
        self.functional_kinds[id.as_u16() as usize].is_some()
    }

    /// Get the functional block kind for a block type. None = not functional.
    #[inline(always)]
    pub fn functional_kind(&self, id: BlockId) -> Option<FunctionalBlockKind> {
        self.functional_kinds[id.as_u16() as usize]
    }

    /// Get thruster-specific properties for a block ID. None = not a thruster.
    #[inline]
    pub fn thruster_props(&self, id: BlockId) -> Option<ThrusterProps> {
        self.thruster_props[id.as_u16() as usize]
    }

    /// Get power-specific properties for a block ID. None = not power-relevant.
    #[inline]
    pub fn power_props(&self, id: BlockId) -> Option<PowerProps> {
        self.power_props[id.as_u16() as usize]
    }

    pub fn cruise_drive_props(&self, id: BlockId) -> Option<CruiseDriveProps> {
        self.cruise_drive_props[id.as_u16() as usize]
    }

    pub fn mechanical_props(&self, id: BlockId) -> Option<MechanicalProps> {
        self.mechanical_props[id.as_u16() as usize]
    }

    /// Get the per-block light-emitter specification. `None` for blocks
    /// that don't emit dynamic light.
    #[inline]
    pub fn light_spec(&self, id: BlockId) -> Option<LightSpec> {
        self.light_specs[id.as_u16() as usize]
    }

    /// Per-sub-block-type light spec. Sub-block lamps (`SurfaceLight` and
    /// friends) sit flush against a host-block face — physically smaller
    /// than a full-cube lamp — so their geometry mode is always
    /// `DirectionalFace`. Photometric values are calibrated for ship-
    /// cabin illumination at the active outdoor exposure (see lamp
    /// section in `light_specs` initialisation for the rationale).
    pub fn sub_block_light_spec(
        &self,
        ty: super::sub_block::SubBlockType,
    ) -> Option<LightSpec> {
        use super::sub_block::SubBlockType;
        match ty {
            SubBlockType::SurfaceLight => Some(LightSpec {
                lumens: 60_000.0,
                color_kelvin: 3000.0,
                tint_linear_rgb: [1.0, 1.0, 1.0],
                range_m: 12.0,
                beam_kind: BeamKind::PointOmni,
                shadow_caster_class: 1,
                emissive_radiance_w_per_m2: 12_000.0,
                emissive_geometry: EmissiveGeometry::DirectionalFace,
            }),
            SubBlockType::RedSurfaceLight => Some(LightSpec {
                lumens: 40_000.0,
                color_kelvin: 3000.0,
                tint_linear_rgb: [1.0, 0.15, 0.05],
                range_m: 8.0,
                beam_kind: BeamKind::PointOmni,
                shadow_caster_class: 0,
                emissive_radiance_w_per_m2: 8_000.0,
                emissive_geometry: EmissiveGeometry::DirectionalFace,
            }),
            SubBlockType::BlueSurfaceLight => Some(LightSpec {
                lumens: 40_000.0,
                color_kelvin: 3000.0,
                tint_linear_rgb: [0.15, 0.30, 1.0],
                range_m: 8.0,
                beam_kind: BeamKind::PointOmni,
                shadow_caster_class: 0,
                emissive_radiance_w_per_m2: 8_000.0,
                emissive_geometry: EmissiveGeometry::DirectionalFace,
            }),
            SubBlockType::Floodlight => Some(LightSpec {
                lumens: 300_000.0,
                color_kelvin: 5500.0,
                tint_linear_rgb: [1.0, 1.0, 1.0],
                range_m: 40.0,
                beam_kind: BeamKind::SpotCone {
                    inner_angle_rad: 30.0_f32.to_radians(),
                    outer_angle_rad: 45.0_f32.to_radians(),
                },
                shadow_caster_class: 2,
                emissive_radiance_w_per_m2: 60_000.0,
                emissive_geometry: EmissiveGeometry::DirectionalFace,
            }),
            _ => None,
        }
    }

    /// Get the interaction schema for a functional block kind.
    /// Returns the set of interactions (action keys + types + labels) this kind supports.
    pub fn interaction_schema(&self, kind: FunctionalBlockKind) -> &'static InteractionSchema {
        match kind {
            FunctionalBlockKind::Seat => &SCHEMA_SEAT,
            FunctionalBlockKind::Reactor => &SCHEMA_TOGGLE,
            FunctionalBlockKind::Battery => &SCHEMA_TOGGLE,
            FunctionalBlockKind::ShieldGenerator => &SCHEMA_TOGGLE,
            FunctionalBlockKind::AirCompressor => &SCHEMA_TOGGLE,
            FunctionalBlockKind::SignalConverter => &SCHEMA_PROGRAM,
            FunctionalBlockKind::Computer => &SCHEMA_PROGRAM,
            FunctionalBlockKind::RailJunction => &SCHEMA_JUNCTION,
            // Blocks with no direct player interaction (config via F key panel).
            FunctionalBlockKind::CruiseDrive => &SCHEMA_NONE,
            _ => &SCHEMA_NONE,
        }
    }
}

// ---------------------------------------------------------------------------
// Helpers to reduce boilerplate in the registration above
// ---------------------------------------------------------------------------

fn set(defs: &mut [BlockDef], id: BlockId, def: BlockDef) {
    defs[id.as_u16() as usize] = def;
}

fn stone_variant(name: &'static str, density: u16, color: [u8; 3]) -> BlockDef {
    BlockDef {
        name,
        is_solid: true,
        is_transparent: false,
        material_type: MaterialType::Stone,
        density,
        hardness: 30,
        mining_tier: 1,
        preferred_tool: ToolType::Pickaxe,
        blast_resistance: 30,
        color_hint: color,
        ..BlockDef::UNDEFINED
    }
}

fn soil(name: &'static str, density: u16, color: [u8; 3]) -> BlockDef {
    BlockDef {
        name,
        is_solid: true,
        is_transparent: false,
        material_type: MaterialType::Soil,
        density,
        hardness: 5,
        preferred_tool: ToolType::Shovel,
        blast_resistance: 5,
        color_hint: color,
        ..BlockDef::UNDEFINED
    }
}

fn ice(name: &'static str, density: u16, hardness: u8, color: [u8; 3]) -> BlockDef {
    BlockDef {
        name,
        is_solid: true,
        is_transparent: false,
        material_type: MaterialType::Ice,
        density,
        hardness,
        mining_tier: 1,
        preferred_tool: ToolType::Pickaxe,
        blast_resistance: hardness,
        color_hint: color,
        ..BlockDef::UNDEFINED
    }
}

fn ore(name: &'static str, density: u16, hardness: u8, mining_tier: u8, color: [u8; 3]) -> BlockDef {
    BlockDef {
        name,
        is_solid: true,
        is_transparent: false,
        material_type: MaterialType::Stone,
        density,
        hardness,
        mining_tier,
        preferred_tool: ToolType::Pickaxe,
        blast_resistance: hardness,
        color_hint: color,
        ..BlockDef::UNDEFINED
    }
}

fn log(name: &'static str, color: [u8; 3]) -> BlockDef {
    BlockDef {
        name,
        is_solid: true,
        is_transparent: false,
        material_type: MaterialType::Wood,
        density: 600,
        hardness: 10,
        preferred_tool: ToolType::Axe,
        blast_resistance: 10,
        color_hint: color,
        flammability: 150,
        fire_resistance: 50,
        ..BlockDef::UNDEFINED
    }
}

fn leaves(name: &'static str, color: [u8; 3]) -> BlockDef {
    BlockDef {
        name,
        is_solid: true,
        is_transparent: true,
        material_type: MaterialType::Organic,
        density: 100,
        hardness: 2,
        preferred_tool: ToolType::None,
        blast_resistance: 1,
        color_hint: color,
        flammability: 200,
        fire_resistance: 20,
        ..BlockDef::UNDEFINED
    }
}

fn hull(name: &'static str, density: u16, hardness: u8, blast_resistance: u8, color: [u8; 3]) -> BlockDef {
    BlockDef {
        name,
        is_solid: true,
        is_transparent: false,
        material_type: MaterialType::Metal,
        density,
        hardness,
        mining_tier: 2,
        preferred_tool: ToolType::Drill,
        blast_resistance,
        color_hint: color,
        fire_resistance: 255,
        ..BlockDef::UNDEFINED
    }
}

fn functional_block(name: &'static str, density: u16, hardness: u8, color: [u8; 3]) -> BlockDef {
    BlockDef {
        name,
        is_solid: true,
        is_transparent: false,
        material_type: MaterialType::Metal,
        density,
        hardness,
        mining_tier: 1,
        preferred_tool: ToolType::Drill,
        blast_resistance: hardness,
        color_hint: color,
        fire_resistance: 200,
        ..BlockDef::UNDEFINED
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn air_is_not_solid() {
        let reg = BlockRegistry::new();
        assert!(!reg.is_solid(BlockId::AIR));
        assert!(reg.is_transparent(BlockId::AIR));
    }

    #[test]
    fn stone_is_solid_and_opaque() {
        let reg = BlockRegistry::new();
        assert!(reg.is_solid(BlockId::STONE));
        assert!(!reg.is_transparent(BlockId::STONE));
    }

    #[test]
    fn water_is_liquid() {
        let reg = BlockRegistry::new();
        let def = reg.get(BlockId::WATER);
        assert!(!def.is_solid);
        assert!(def.is_transparent);
        assert!(def.liquid.is_liquid());
    }

    #[test]
    fn all_named_blocks_have_names() {
        let reg = BlockRegistry::new();
        // Spot-check that all registered blocks have non-"undefined" names
        let named = [
            BlockId::AIR, BlockId::STONE, BlockId::DIRT, BlockId::GRASS,
            BlockId::SAND, BlockId::WATER, BlockId::GRANITE, BlockId::OBSIDIAN,
            BlockId::COAL_ORE, BlockId::DIAMOND_ORE, BlockId::LAVA,
            BlockId::OAK_LOG, BlockId::HULL_STANDARD, BlockId::COCKPIT,
            BlockId::THRUSTER_SMALL_CHEMICAL, BlockId::REACTOR_SMALL,
            BlockId::BATTERY, BlockId::ROTOR, BlockId::PISTON,
            BlockId::RAIL_STRAIGHT, BlockId::SIGNAL_CONVERTER,
            BlockId::COMPUTER, BlockId::OWNERSHIP_CORE, BlockId::HUD_PANEL,
            BlockId::ANTENNA, BlockId::SHIELD_EMITTER,
        ];
        for id in named {
            let def = reg.get(id);
            assert_ne!(def.name, "undefined", "Block {:?} should be defined", id);
        }
    }

    #[test]
    fn undefined_blocks_are_air_like() {
        let reg = BlockRegistry::new();
        // An ID that is not registered should have air-like properties
        let def = reg.get(BlockId::from_u16(9999));
        assert_eq!(def.name, "undefined");
        assert!(!def.is_solid);
        assert!(def.is_transparent);
    }

    #[test]
    fn hull_blocks_are_fireproof() {
        let reg = BlockRegistry::new();
        for id in [BlockId::HULL_LIGHT, BlockId::HULL_STANDARD, BlockId::HULL_HEAVY, BlockId::HULL_ARMORED] {
            let def = reg.get(id);
            assert_eq!(def.fire_resistance, 255, "{} should be fireproof", def.name);
        }
    }

    #[test]
    fn gravity_affected_blocks() {
        let reg = BlockRegistry::new();
        assert!(reg.get(BlockId::SAND).gravity_affected);
        assert!(reg.get(BlockId::GRAVEL).gravity_affected);
        assert!(reg.get(BlockId::RED_SAND).gravity_affected);
        assert!(reg.get(BlockId::ASH).gravity_affected);
        assert!(!reg.get(BlockId::STONE).gravity_affected);
    }

    #[test]
    fn hardness_to_hp() {
        let reg = BlockRegistry::new();
        assert_eq!(reg.get(BlockId::OBSIDIAN).max_hp(), 2550);
        assert_eq!(reg.get(BlockId::DIRT).max_hp(), 50);
        assert_eq!(reg.get(BlockId::AIR).max_hp(), 0);
    }

    #[test]
    fn solid_blocks_must_not_be_transparent() {
        // Guard against the UNDEFINED spread bug: if a block is solid, it must
        // not be transparent (except for special cases like leaves and windows).
        let reg = BlockRegistry::new();
        let exceptions = [
            BlockId::WINDOW, BlockId::HUD_PANEL,
            // Leaves are solid + transparent (light passes through, collision exists)
            BlockId::OAK_LEAVES, BlockId::PINE_LEAVES, BlockId::JUNGLE_LEAVES,
            BlockId::BIRCH_LEAVES, BlockId::ACACIA_LEAVES,
        ];
        for id_raw in 0..2000u16 {
            let id = BlockId::from_u16(id_raw);
            let def = reg.get(id);
            if def.name == "undefined" { continue; }
            if exceptions.contains(&id) { continue; }
            if def.is_solid && def.is_transparent {
                panic!(
                    "Block '{}' (id={}) is solid AND transparent — likely missing \
                     is_transparent: false in its definition",
                    def.name, id_raw
                );
            }
        }
    }

    #[test]
    fn structural_blocks_are_not_functional() {
        let reg = BlockRegistry::new();
        assert!(!reg.is_functional(BlockId::AIR));
        assert!(!reg.is_functional(BlockId::STONE));
        assert!(!reg.is_functional(BlockId::HULL_STANDARD));
        assert!(!reg.is_functional(BlockId::WINDOW));
        assert!(!reg.is_functional(BlockId::DIRT));
    }

    #[test]
    fn functional_blocks_have_kinds() {
        let reg = BlockRegistry::new();
        assert_eq!(reg.functional_kind(BlockId::THRUSTER_SMALL_CHEMICAL), Some(FunctionalBlockKind::Thruster));
        assert_eq!(reg.functional_kind(BlockId::REACTOR_SMALL), Some(FunctionalBlockKind::Reactor));
        assert_eq!(reg.functional_kind(BlockId::BATTERY), Some(FunctionalBlockKind::Battery));
        assert_eq!(reg.functional_kind(BlockId::COCKPIT), Some(FunctionalBlockKind::Seat));
        assert_eq!(reg.functional_kind(BlockId::ROTOR), Some(FunctionalBlockKind::Rotor));
        assert_eq!(reg.functional_kind(BlockId::PISTON), Some(FunctionalBlockKind::Piston));
        assert_eq!(reg.functional_kind(BlockId::RAIL_STRAIGHT), Some(FunctionalBlockKind::Rail));
        assert_eq!(reg.functional_kind(BlockId::SIGNAL_CONVERTER), Some(FunctionalBlockKind::SignalConverter));
        assert_eq!(reg.functional_kind(BlockId::COMPUTER), Some(FunctionalBlockKind::Computer));
    }

    #[test]
    fn interaction_schemas() {
        let reg = BlockRegistry::new();
        let seat_schema = reg.interaction_schema(FunctionalBlockKind::Seat);
        assert_eq!(seat_schema.actions.len(), 1);
        assert_eq!(seat_schema.actions[0].action_key, 3);
        assert_eq!(seat_schema.actions[0].label, "Sit");

        let thruster_schema = reg.interaction_schema(FunctionalBlockKind::Thruster);
        assert!(thruster_schema.actions.is_empty());

        let junction_schema = reg.interaction_schema(FunctionalBlockKind::RailJunction);
        assert_eq!(junction_schema.actions.len(), 1);
        assert_eq!(junction_schema.actions[0].interaction_type, InteractionType::CycleState);
    }
}
