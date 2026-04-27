//! Sub-block elements — thin functional elements placed on block surfaces.
//!
//! Sub-blocks attach to specific faces of solid blocks. Multiple elements can
//! coexist on different faces of the same host block. They don't replace the
//! host block — they sit ON its surface.
//!
//! Sub-blocks are the connection infrastructure for game systems:
//! - Power wires connect blocks in the power network graph
//! - Rails define paths for transport carts
//! - Mechanical mounts (rotor, piston, hinge) create Rapier physics joints
//! - Pipes route fluids between blocks
//! - Ladders modify player KCC physics (climbable)
//!
//! Sub-blocks are stored as data in ChunkStorage (no ECS entities).
//! Game systems create their own network entities by querying sub-block data.

use glam::IVec3;
use serde::{Deserialize, Serialize};

/// Type of sub-block element.
/// ID ranges are reserved per category for future expansion without renumbering.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum SubBlockType {
    // Power & signals (0-9)
    PowerWire       = 0,
    SignalWire      = 1,

    // Transport (10-19)
    Rail            = 10,
    RailJunction    = 11,
    RailStop        = 12,
    ConveyorBelt    = 13,

    // Mechanical joints (20-29)
    RotorMount      = 20,
    PistonMount     = 21,
    HingeMount      = 22,
    SliderMount     = 23,

    // Fluid (30-39)
    Pipe            = 30,
    PipeValve       = 31,
    PipePump        = 32,

    // Decorative & utility (40-49)
    Ladder          = 40,
    Handle          = 41,
    /// Warm-white interior fixture mounted flush against a host
    /// block's face. Light + emissive proxy spawn from
    /// `BlockRegistry::sub_block_light_spec`.
    SurfaceLight    = 42,
    Cable           = 43,
    Vent            = 44,
    /// Red emergency / alarm lamp — same fixture geometry as
    /// `SurfaceLight`, saturated red filter on the LightSpec.
    RedSurfaceLight = 45,
    /// Blue instrument-panel lamp — soft blue filter, low intensity.
    BlueSurfaceLight = 46,
    /// Industrial daylight floodlight — directional spot beam from
    /// the face. Same `DirectionalFace` emissive geometry, much
    /// higher lumens than the warm-white surface light.
    Floodlight      = 47,

    // Structural (50-59)
    Bracket         = 50,
    Seal            = 51,

    // HUD / UI (60-69)
    /// Transparent signal-driven HUD tile on a block face.
    /// Per-sub-block config (widget kind + channel/property) lives in
    /// the server-authored `BlockSignalConfig` payload keyed on
    /// `(block_pos, face)`.
    HudPanel        = 60,
}

impl SubBlockType {
    pub fn from_u8(v: u8) -> Option<Self> {
        match v {
            0  => Some(Self::PowerWire),
            1  => Some(Self::SignalWire),
            10 => Some(Self::Rail),
            11 => Some(Self::RailJunction),
            12 => Some(Self::RailStop),
            13 => Some(Self::ConveyorBelt),
            20 => Some(Self::RotorMount),
            21 => Some(Self::PistonMount),
            22 => Some(Self::HingeMount),
            23 => Some(Self::SliderMount),
            30 => Some(Self::Pipe),
            31 => Some(Self::PipeValve),
            32 => Some(Self::PipePump),
            40 => Some(Self::Ladder),
            41 => Some(Self::Handle),
            42 => Some(Self::SurfaceLight),
            43 => Some(Self::Cable),
            44 => Some(Self::Vent),
            45 => Some(Self::RedSurfaceLight),
            46 => Some(Self::BlueSurfaceLight),
            47 => Some(Self::Floodlight),
            50 => Some(Self::Bracket),
            51 => Some(Self::Seal),
            60 => Some(Self::HudPanel),
            _  => None,
        }
    }

    /// Human-readable name for UI display.
    pub fn label(self) -> &'static str {
        match self {
            Self::PowerWire     => "Power Wire",
            Self::SignalWire    => "Signal Wire",
            Self::Rail          => "Rail",
            Self::RailJunction  => "Rail Junction",
            Self::RailStop      => "Rail Stop",
            Self::ConveyorBelt  => "Conveyor Belt",
            Self::RotorMount    => "Rotor Mount",
            Self::PistonMount   => "Piston Mount",
            Self::HingeMount    => "Hinge Mount",
            Self::SliderMount   => "Slider Mount",
            Self::Pipe          => "Pipe",
            Self::PipeValve     => "Pipe Valve",
            Self::PipePump      => "Pipe Pump",
            Self::Ladder        => "Ladder",
            Self::Handle        => "Handle",
            Self::SurfaceLight  => "Surface Light",
            Self::Cable         => "Cable",
            Self::Vent          => "Vent",
            Self::RedSurfaceLight  => "Red Surface Light",
            Self::BlueSurfaceLight => "Blue Surface Light",
            Self::Floodlight       => "Floodlight",
            Self::Bracket       => "Bracket",
            Self::Seal          => "Seal",
            Self::HudPanel      => "HUD Panel",
        }
    }

    /// Whether this sub-block type forms chains (connects to adjacent sub-blocks
    /// of the same type on opposing faces).
    pub fn is_chainable(self) -> bool {
        matches!(self,
            Self::PowerWire | Self::SignalWire |
            Self::Rail | Self::ConveyorBelt |
            Self::Pipe | Self::Cable
        )
    }

    /// Color hint for rendering (RGB 0-255).
    pub fn color(self) -> [u8; 3] {
        match self {
            Self::PowerWire     => [255, 180, 0],    // orange
            Self::SignalWire    => [60, 200, 255],   // cyan
            Self::Rail          => [140, 140, 140],  // steel gray
            Self::RailJunction  => [140, 140, 160],
            Self::RailStop      => [200, 60, 60],    // red
            Self::ConveyorBelt  => [180, 160, 100],  // tan
            Self::RotorMount    => [100, 200, 100],  // green
            Self::PistonMount   => [100, 200, 100],
            Self::HingeMount    => [100, 200, 100],
            Self::SliderMount   => [100, 200, 100],
            Self::Pipe          => [80, 130, 200],   // blue
            Self::PipeValve     => [80, 130, 180],
            Self::PipePump      => [80, 150, 220],
            Self::Ladder        => [160, 120, 60],   // wood brown
            Self::Handle        => [180, 180, 180],  // light gray
            Self::SurfaceLight  => [255, 255, 200],  // warm white
            Self::Cable         => [40, 40, 40],     // dark gray
            Self::Vent          => [120, 130, 140],  // steel
            Self::RedSurfaceLight  => [255, 60, 30], // red emergency
            Self::BlueSurfaceLight => [60, 120, 255], // blue instrument
            Self::Floodlight       => [240, 240, 245], // daylight white
            Self::Bracket       => [100, 100, 110],
            Self::Seal          => [200, 200, 220],  // light steel
            Self::HudPanel      => [20, 30, 50],     // dark glass; real content drawn by widget
        }
    }
}

/// A single sub-block element attached to a block face.
/// Size: 4 bytes. Stored sparsely in ChunkStorage.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SubBlockElement {
    /// Which face of the host block (0-5: +X, -X, +Y, -Y, +Z, -Z).
    pub face: u8,
    /// Element type.
    pub element_type: SubBlockType,
    /// Rotation on the face (0-3: 0°, 90°, 180°, 270° around face normal).
    pub rotation: u8,
    /// State flags.
    pub flags: u8,
}

impl SubBlockElement {
    pub const FLAG_POWERED: u8 = 0x01;
    pub const FLAG_ACTIVE: u8 = 0x02;
}

// ---------------------------------------------------------------------------
// Lamp configuration (per-placed sub-block lamp)
// ---------------------------------------------------------------------------

/// Per-lamp player-editable configuration for `SubBlockType::SurfaceLight`,
/// `RedSurfaceLight`, `BlueSurfaceLight`, and `Floodlight` elements.
///
/// Stored on the `ShipGrid` keyed by `(IVec3, face)` and broadcast to
/// clients on chunk snapshot / delta. The default value reproduces the
/// stock `LightSpec` for the sub-block's type — players adjust through
/// the F-key config panel on each lamp:
///
///   * `subscribe_channel` — drives the lamp's on/off / brightness.
///     When non-empty, the client multiplies `LightSpec.lumens`
///     (× `intensity_scale`) by the channel's broadcast value.
///   * `publish_channel` — channel the lamp publishes its `Active`
///     state to, so other blocks can react to the lamp turning on/off.
///   * `color_kelvin` + `tint_linear_rgb` — overrides the spec's
///     blackbody colour. Identity when left at the spec defaults.
///   * `intensity_scale` — multiplier on `LightSpec.lumens` (1.0 = full
///     stock, 0.0 = off, > 1 brightens beyond spec).
///
/// All fields are static-per-session (serialised over FlatBuffers) and
/// the client's lamp spawn reads them when the chunk remeshes.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct LampConfig {
    pub subscribe_channel: String,
    pub publish_channel: String,
    pub color_kelvin: f32,
    pub tint_linear_rgb: [f32; 3],
    pub intensity_scale: f32,
}

impl LampConfig {
    /// Factory: build the default config that reproduces a given
    /// sub-block-light variant's stock `LightSpec`. Used by the server
    /// the first time a player F-keys a freshly-placed lamp.
    pub fn default_for(ty: SubBlockType) -> Self {
        match ty {
            SubBlockType::SurfaceLight => Self {
                subscribe_channel: String::new(),
                publish_channel: String::new(),
                color_kelvin: 3000.0,
                tint_linear_rgb: [1.0, 1.0, 1.0],
                intensity_scale: 1.0,
            },
            SubBlockType::RedSurfaceLight => Self {
                subscribe_channel: String::new(),
                publish_channel: String::new(),
                color_kelvin: 3000.0,
                tint_linear_rgb: [1.0, 0.15, 0.05],
                intensity_scale: 1.0,
            },
            SubBlockType::BlueSurfaceLight => Self {
                subscribe_channel: String::new(),
                publish_channel: String::new(),
                color_kelvin: 3000.0,
                tint_linear_rgb: [0.15, 0.30, 1.0],
                intensity_scale: 1.0,
            },
            SubBlockType::Floodlight => Self {
                subscribe_channel: String::new(),
                publish_channel: String::new(),
                color_kelvin: 5500.0,
                tint_linear_rgb: [1.0, 1.0, 1.0],
                intensity_scale: 1.0,
            },
            // Non-lamp sub-blocks return a neutral-white default; the
            // caller is expected to gate by `is_lamp(ty)` first.
            _ => Self {
                subscribe_channel: String::new(),
                publish_channel: String::new(),
                color_kelvin: 3000.0,
                tint_linear_rgb: [1.0, 1.0, 1.0],
                intensity_scale: 1.0,
            },
        }
    }
}

/// Whether a `SubBlockType` corresponds to a lamp / light fixture
/// that exposes a `LampConfig`.
pub fn is_lamp_sub_block(ty: SubBlockType) -> bool {
    matches!(
        ty,
        SubBlockType::SurfaceLight
            | SubBlockType::RedSurfaceLight
            | SubBlockType::BlueSurfaceLight
            | SubBlockType::Floodlight
    )
}

// ---------------------------------------------------------------------------
// Face utilities
// ---------------------------------------------------------------------------

/// Face indices: 0=+X, 1=-X, 2=+Y, 3=-Y, 4=+Z, 5=-Z.
pub const FACE_COUNT: u8 = 6;

/// Convert face index to offset vector.
pub fn face_to_offset(face: u8) -> IVec3 {
    match face {
        0 => IVec3::X,
        1 => IVec3::NEG_X,
        2 => IVec3::Y,
        3 => IVec3::NEG_Y,
        4 => IVec3::Z,
        5 => IVec3::NEG_Z,
        _ => IVec3::ZERO,
    }
}

/// Get the opposing face index (0↔1, 2↔3, 4↔5).
pub fn opposite_face(face: u8) -> u8 {
    face ^ 1
}

/// Convert an IVec3 face normal to face index (0-5).
pub fn face_from_normal(normal: IVec3) -> u8 {
    if normal.x > 0 { 0 }
    else if normal.x < 0 { 1 }
    else if normal.y > 0 { 2 }
    else if normal.y < 0 { 3 }
    else if normal.z > 0 { 4 }
    else if normal.z < 0 { 5 }
    else { 2 } // default +Y
}

// ---------------------------------------------------------------------------
// Connectivity
// ---------------------------------------------------------------------------

/// Check if two sub-blocks on adjacent block faces form a connection.
/// A wire on face +X of block A connects to a wire on face -X of block B (adjacent in +X).
pub fn are_connected(
    pos_a: IVec3, face_a: u8,
    pos_b: IVec3, face_b: u8,
) -> bool {
    let expected_offset = face_to_offset(face_a);
    let actual_offset = pos_b - pos_a;
    actual_offset == expected_offset && face_b == opposite_face(face_a)
}

// ---------------------------------------------------------------------------
// Bitmask auto-tiling
// ---------------------------------------------------------------------------

/// For each face, the 4 edge directions in the face plane.
/// Each edge direction is an IVec3 offset to the adjacent block.
/// Order: [+u, -u, +v, -v] in the face's tangent space.
/// Bits: 0=+u, 1=-u, 2=+v, 3=-v.
const FACE_EDGE_OFFSETS: [[IVec3; 4]; 6] = [
    // Face 0 (+X): tangent u=-Z, v=+Y
    [IVec3::NEG_Z, IVec3::Z, IVec3::Y, IVec3::NEG_Y],
    // Face 1 (-X): tangent u=+Z, v=+Y
    [IVec3::Z, IVec3::NEG_Z, IVec3::Y, IVec3::NEG_Y],
    // Face 2 (+Y): tangent u=+X, v=+Z
    [IVec3::X, IVec3::NEG_X, IVec3::Z, IVec3::NEG_Z],
    // Face 3 (-Y): tangent u=+X, v=-Z
    [IVec3::X, IVec3::NEG_X, IVec3::NEG_Z, IVec3::Z],
    // Face 4 (+Z): tangent u=+X, v=+Y
    [IVec3::X, IVec3::NEG_X, IVec3::Y, IVec3::NEG_Y],
    // Face 5 (-Z): tangent u=-X, v=+Y
    [IVec3::NEG_X, IVec3::X, IVec3::Y, IVec3::NEG_Y],
];

/// Compute the 4-bit connection mask for a chainable sub-block on a face.
/// Bit 0 = +u neighbor, bit 1 = -u neighbor, bit 2 = +v neighbor, bit 3 = -v neighbor.
///
/// Checks adjacent blocks on the same face plane for matching chainable sub-blocks.
/// The `get_sub_blocks` closure returns the sub-block elements at a given world position.
pub fn compute_connection_mask(
    pos: IVec3,
    face: u8,
    element_type: SubBlockType,
    get_sub_blocks: impl Fn(IVec3) -> Vec<SubBlockElement>,
) -> u8 {
    if face >= 6 { return 0; }
    let edges = &FACE_EDGE_OFFSETS[face as usize];
    let mut mask = 0u8;

    for (bit, &offset) in edges.iter().enumerate() {
        let neighbor_pos = pos + offset;
        let neighbor_subs = get_sub_blocks(neighbor_pos);
        // Check if neighbor has the same chainable type on the same face.
        if neighbor_subs.iter().any(|e| e.face == face && e.element_type == element_type) {
            mask |= 1 << bit;
        }
    }

    mask
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn face_offset_roundtrip() {
        for face in 0..6 {
            let offset = face_to_offset(face);
            let back = face_from_normal(offset);
            assert_eq!(face, back, "face {face} → offset {offset} → face {back}");
        }
    }

    #[test]
    fn opposite_face_pairs() {
        assert_eq!(opposite_face(0), 1); // +X ↔ -X
        assert_eq!(opposite_face(1), 0);
        assert_eq!(opposite_face(2), 3); // +Y ↔ -Y
        assert_eq!(opposite_face(3), 2);
        assert_eq!(opposite_face(4), 5); // +Z ↔ -Z
        assert_eq!(opposite_face(5), 4);
    }

    #[test]
    fn connectivity_adjacent_faces() {
        // Wire on +X face of (0,0,0) connects to wire on -X face of (1,0,0).
        assert!(are_connected(
            IVec3::new(0, 0, 0), 0,  // +X face
            IVec3::new(1, 0, 0), 1,  // -X face
        ));

        // Same faces (not opposing) don't connect.
        assert!(!are_connected(
            IVec3::new(0, 0, 0), 0,  // +X face
            IVec3::new(1, 0, 0), 0,  // +X face (wrong — should be -X)
        ));

        // Non-adjacent blocks don't connect.
        assert!(!are_connected(
            IVec3::new(0, 0, 0), 0,
            IVec3::new(2, 0, 0), 1,  // 2 blocks apart
        ));
    }

    #[test]
    fn sub_block_type_roundtrip() {
        for &sbt in &[
            SubBlockType::PowerWire, SubBlockType::Rail, SubBlockType::RotorMount,
            SubBlockType::Pipe, SubBlockType::Ladder, SubBlockType::Bracket,
        ] {
            let u = sbt as u8;
            assert_eq!(SubBlockType::from_u8(u), Some(sbt), "roundtrip failed for {sbt:?}");
        }
        assert_eq!(SubBlockType::from_u8(255), None);
    }
}
