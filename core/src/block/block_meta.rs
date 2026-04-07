/// Compact block orientation: 6 facing directions × 4 rotations = 24 states.
///
/// - Facing (0–5): +X, -X, +Y, -Y, +Z, -Z
/// - Rotation (0–3): 0°, 90°, 180°, 270° around the facing axis
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Default)]
#[repr(transparent)]
pub struct BlockOrientation(pub u8);

impl BlockOrientation {
    pub const DEFAULT: Self = Self(0); // facing +Y, no rotation

    /// Facing direction index (0–5): +X, -X, +Y, -Y, +Z, -Z.
    #[inline(always)]
    pub const fn facing(self) -> u8 {
        self.0 / 4
    }

    /// Rotation around facing axis (0–3): 0°, 90°, 180°, 270°.
    #[inline(always)]
    pub const fn rotation(self) -> u8 {
        self.0 % 4
    }

    /// Construct from facing direction and rotation.
    ///
    /// # Panics
    /// Debug-asserts that `facing < 6` and `rotation < 4`.
    pub const fn new(facing: u8, rotation: u8) -> Self {
        debug_assert!(facing < 6 && rotation < 4);
        Self(facing * 4 + rotation)
    }

    /// Total number of valid orientations.
    pub const COUNT: u8 = 24;

    /// Create orientation from a face normal direction (IVec3).
    /// The block faces in the direction of the normal — e.g., a thruster
    /// placed on the -Z face of a wall will face +Z (outward from the wall).
    pub fn from_face_normal(normal: glam::IVec3) -> Self {
        let facing = match (normal.x, normal.y, normal.z) {
            (1, 0, 0)  => 0,  // +X
            (-1, 0, 0) => 1,  // -X
            (0, 1, 0)  => 2,  // +Y
            (0, -1, 0) => 3,  // -Y
            (0, 0, 1)  => 4,  // +Z
            (0, 0, -1) => 5,  // -Z
            _ => 2,            // fallback: +Y (default)
        };
        Self::new(facing, 0)
    }

    /// Get the facing direction as a unit DVec3.
    /// Maps facing index 0-5 to axis-aligned direction vectors.
    pub fn facing_direction(self) -> glam::DVec3 {
        match self.facing() {
            0 => glam::DVec3::X,
            1 => glam::DVec3::NEG_X,
            2 => glam::DVec3::Y,
            3 => glam::DVec3::NEG_Y,
            4 => glam::DVec3::Z,
            5 => glam::DVec3::NEG_Z,
            _ => glam::DVec3::Y,
        }
    }

    /// Get the facing index (0-5) for a given axis direction.
    /// Inverse of `facing_direction()`.
    pub fn axis_to_facing(positive_axis: bool, axis: u8) -> u8 {
        match (axis, positive_axis) {
            (0, true)  => 0,  // +X
            (0, false) => 1,  // -X
            (1, true)  => 2,  // +Y
            (1, false) => 3,  // -Y
            (2, true)  => 4,  // +Z
            (2, false) => 5,  // -Z
            _ => 2,
        }
    }
}

/// Per-block metadata flags stored as a bitfield.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
#[repr(transparent)]
pub struct BlockFlags(pub u8);

impl BlockFlags {
    pub const DAMAGED: u8 = 0x01;
    pub const ON_FIRE: u8 = 0x02;
    pub const POWERED: u8 = 0x04;
    pub const WATERLOGGED: u8 = 0x08;

    #[inline(always)]
    pub const fn has(self, flag: u8) -> bool {
        self.0 & flag != 0
    }

    #[inline(always)]
    pub fn set(&mut self, flag: u8, value: bool) {
        if value {
            self.0 |= flag;
        } else {
            self.0 &= !flag;
        }
    }
}

/// Per-block instance metadata. Only allocated for blocks that need non-default
/// state (damaged, oriented, functional). Stored sparsely in `ChunkStorage`.
///
/// Size: 8 bytes.
#[derive(Clone, Debug, Default)]
pub struct BlockMeta {
    /// Current hit points. Only meaningful when `flags.has(DAMAGED)`.
    /// 0 = will break this tick.
    pub health: u16,
    /// Block orientation (facing + rotation).
    pub orientation: BlockOrientation,
    /// Boolean state flags.
    pub flags: BlockFlags,
    /// ECS Entity index for functional blocks. 0 = no associated entity.
    /// Uses u32 for compactness — the Entity generation is tracked separately
    /// by the ECS world.
    pub entity_index: u32,
}

impl BlockMeta {
    pub const EMPTY: Self = Self {
        health: 0,
        orientation: BlockOrientation::DEFAULT,
        flags: BlockFlags(0),
        entity_index: 0,
    };

    /// Whether this block has taken damage.
    #[inline]
    pub fn is_damaged(&self) -> bool {
        self.flags.has(BlockFlags::DAMAGED)
    }

    /// Whether this metadata entry is effectively empty (no meaningful state).
    /// Used to decide whether to remove the sparse entry.
    pub fn is_empty(&self) -> bool {
        self.flags.0 == 0
            && self.orientation == BlockOrientation::DEFAULT
            && self.entity_index == 0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn orientation_roundtrip() {
        for facing in 0..6u8 {
            for rot in 0..4u8 {
                let o = BlockOrientation::new(facing, rot);
                assert_eq!(o.facing(), facing);
                assert_eq!(o.rotation(), rot);
            }
        }
    }

    #[test]
    fn orientation_count() {
        assert_eq!(BlockOrientation::COUNT, 24);
        // Ensure all 24 values are distinct
        let mut seen = std::collections::HashSet::new();
        for facing in 0..6u8 {
            for rot in 0..4u8 {
                assert!(seen.insert(BlockOrientation::new(facing, rot)));
            }
        }
    }

    #[test]
    fn flags_set_and_check() {
        let mut f = BlockFlags(0);
        assert!(!f.has(BlockFlags::DAMAGED));
        f.set(BlockFlags::DAMAGED, true);
        assert!(f.has(BlockFlags::DAMAGED));
        assert!(!f.has(BlockFlags::ON_FIRE));
        f.set(BlockFlags::ON_FIRE, true);
        assert!(f.has(BlockFlags::DAMAGED));
        assert!(f.has(BlockFlags::ON_FIRE));
        f.set(BlockFlags::DAMAGED, false);
        assert!(!f.has(BlockFlags::DAMAGED));
        assert!(f.has(BlockFlags::ON_FIRE));
    }

    #[test]
    fn meta_empty_check() {
        let m = BlockMeta::EMPTY;
        assert!(m.is_empty());

        let mut m2 = BlockMeta::EMPTY;
        m2.flags.set(BlockFlags::DAMAGED, true);
        assert!(!m2.is_empty());

        let mut m3 = BlockMeta::EMPTY;
        m3.orientation = BlockOrientation::new(2, 1);
        assert!(!m3.is_empty());

        let mut m4 = BlockMeta::EMPTY;
        m4.entity_index = 42;
        assert!(!m4.is_empty());
    }

    #[test]
    fn from_face_normal_all_axes() {
        use glam::IVec3;
        assert_eq!(BlockOrientation::from_face_normal(IVec3::X).facing(), 0);
        assert_eq!(BlockOrientation::from_face_normal(IVec3::NEG_X).facing(), 1);
        assert_eq!(BlockOrientation::from_face_normal(IVec3::Y).facing(), 2);
        assert_eq!(BlockOrientation::from_face_normal(IVec3::NEG_Y).facing(), 3);
        assert_eq!(BlockOrientation::from_face_normal(IVec3::Z).facing(), 4);
        assert_eq!(BlockOrientation::from_face_normal(IVec3::NEG_Z).facing(), 5);
    }

    #[test]
    fn facing_direction_roundtrip() {
        use glam::DVec3;
        let dirs = [DVec3::X, DVec3::NEG_X, DVec3::Y, DVec3::NEG_Y, DVec3::Z, DVec3::NEG_Z];
        for (i, expected) in dirs.iter().enumerate() {
            let o = BlockOrientation::new(i as u8, 0);
            assert_eq!(o.facing_direction(), *expected, "facing {} should map to {:?}", i, expected);
        }
    }
}
