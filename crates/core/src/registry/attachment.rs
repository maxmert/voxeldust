//! THE ATTACHMENT KIND TABLE — the things that sit on a block's face and take no space (V2.5): a HUD,
//! a hinge, a piston, a rail clamp. A second typed table under the same mechanism (ruling B-8), so
//! "an attachment kind in a cell record" is a compile error and not a runtime refusal.
//!
//! What is IN the identity digest: the kind's KEY and whether it makes a second rigid body. What is
//! OUT: the display name, the slot count, the legal faces and the parameter budget, which bound what a
//! player may attach next and may be raised freely.
//!
//! **Example.** A pilot bolts a fuel gauge onto the aft face of a tank block, slot 0. Later she adds a
//! warning lamp on the same face, slot 1. Both rows sit beside the tank's record; neither takes a cell.

/// An attachment kind's number: its index in [`ATTACHMENTS`]. Dense, append-only, never reused. No
/// `Default`: a short read never becomes a HUD.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct AttachmentKindId(pub u16);

/// One attachment kind.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct AttachmentDef {
    /// The identity, folded into the digest. Never changes once the row exists.
    pub key: &'static str,
    /// The display word. May change freely.
    pub name: &'static str,
    /// Whether attaching one makes a second rigid body inside the realm (a joint).
    pub makes_body: bool,
    /// Which faces accept it (`+X −X +Y −Y +Z −Z` as bits 0..=5).
    pub legal_faces: u8,
    /// How many may share one face.
    pub slots: u8,
    /// The most parameter bytes one may carry.
    pub params_max: u16,
}

const ALL_FACES: u8 = 0b11_1111;

const fn row(key: &'static str, makes_body: bool, slots: u8, params_max: u16) -> AttachmentDef {
    AttachmentDef {
        key,
        name: key,
        makes_body,
        legal_faces: ALL_FACES,
        slots,
        params_max,
    }
}

/// THE ATTACHMENT KIND TABLE. The index is the id. APPEND ONLY.
pub const ATTACHMENTS: [AttachmentDef; 4] = [
    row("hud", false, 4, 64),
    row("hinge", true, 1, 16),
    row("piston", true, 1, 16),
    row("rail clamp", true, 1, 32),
];

impl AttachmentKindId {
    /// This kind's row; `None` for a number the table does not hold.
    #[must_use]
    pub fn def(self) -> Option<&'static AttachmentDef> {
        ATTACHMENTS.get(usize::from(self.0))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_table_is_dense_the_builder_runs_and_the_joints_make_bodies() {
        crate::registry::assert_dense(ATTACHMENTS.len(), |id| AttachmentKindId(id).def());
        assert_eq!(row("hud", false, 4, 64), ATTACHMENTS[0]);
        assert_eq!(
            AttachmentKindId(0).def().map(|a| (a.key, a.makes_body)),
            Some(("hud", false))
        );
        for a in &ATTACHMENTS[1..] {
            assert!(a.makes_body, "{} is a joint", a.key);
            assert_eq!(a.slots, 1, "one joint per face");
        }
        let mut keys = std::collections::BTreeSet::new();
        for a in &ATTACHMENTS {
            assert!(keys.insert(a.key), "duplicate attachment key {}", a.key);
            assert_eq!(a.legal_faces & !ALL_FACES, 0);
            assert!(a.slots >= 1);
        }
    }
}
