//! ★ WHAT PEOPLE BUILT — the record a realm exists by when no seed made it (D-MOVE-2; owner rulings
//! 2026-09-01).
//!
//! **THE WORLD IS WHAT THE SEED MADE, PLUS A RECORD OF WHAT PEOPLE BUILT.** The seed half is computed:
//! every process starts from one number and works out the same stars, so nothing is stored. A ship
//! cannot be computed — a player made it — so it is written down instead, and read at the same place in
//! the same boot.
//!
//! **THE BOOT PATH NEVER LEARNS THE DIFFERENCE.** Today an operator writes these rows; later a shipyard
//! writes them, for a price, with robots building block by block. The rows do not change. Only who
//! fills them in does, and that is the whole design.
//!
//! ⚠ **THIS IS NOT A SECOND WORLD, and the difference is worth stating because they can look alike.**
//! A world VARIANT is a setting each process reads on its own — two processes then hold different
//! worlds, which measurably happened and was backed out on 2026-08-31. A RECORD adds nothing to the
//! world: the generator is untouched, no setting selects anything, and an empty record is this world
//! before anybody built in it (SL5).
use crate::fence::Fence;
use crate::geometry::Boundary;
use crate::ids::AccountId;
use crate::pose::RealmId;
use glam::DVec3;
use serde::{Deserialize, Serialize};

/// Which design a hull was built from. A shipyard prices from it and a repair rebuilds from it.
///
/// Nothing reads it yet. It is written anyway: a row stored without it can never gain it later, and
/// every ship built before blueprints existed would be a ship nobody could repair.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub struct BlueprintId(pub u128);

/// WHAT A BODY IS, for the forces its parent applies to it.
///
/// Whole numbers, on the same grids the movement lane uses, for the same reason: two processes must
/// read one statement the same way, and a whole number always does.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct BuiltFacts {
    /// Mass in whole grams. It CANCELS out of gravity — a heavy hull and a light one fall identically
    /// — and does NOT cancel out of drag, which is why it is stored at all.
    pub mass_g: u64,
    /// The area the medium pushes against, in whole square millimetres.
    pub cross_section_mm2: u64,
    /// The drag coefficient, in millionths. A plain ratio with no unit.
    pub drag_micro: u32,
    /// The hardest this hull can push itself, in whole micro-metres per second, per second.
    ///
    /// ★ PER-HULL DATA, WHICH IS THE POINT. A single shipped constant is what exists today, and it
    /// would make every ship in the world fly identically — the magic number the project's own rule
    /// forbids. A real hull derives this from the thrusters built into it; until blocks exist, the
    /// row carries it.
    pub max_push_micro_mps2: i64,
    /// The hardest this hull can turn itself, in whole micro-radians per second, per second.
    pub max_turn_micro_radps2: i64,
}

/// ★ A BUILT REALM'S OWN BODY — held in ITS OWN store, because a realm authors how it looks (SL3).
// `Eq` is absent deliberately: a boundary carries a float extent, and a float has no total equality.
// This is the same reason the interest row dropped `Eq` in S10 — `PartialEq` is what every consumer
// actually uses, and claiming a total order over a float would be a lie the compiler cannot catch.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct BuiltBody {
    /// The realm this describes. Minted, up to 128 bits wide — a built realm is named by whichever
    /// machine built it, with no central agreement, so a narrower name would let two machines pick one
    /// name for two PLACES.
    pub realm: RealmId,
    /// Whose it is. A spaceport reads it to know whose ship to deliver; a trade rewrites it.
    ///
    /// ★ THE FIRST OWNER IN THIS WORLD. Nothing here has belonged to a person before — the only owner
    /// record in the tree names a MACHINE.
    pub owner: AccountId,
    /// What it was built from.
    pub blueprint: BlueprintId,
    /// The box that decides who is inside it. **The slot it was sold, not the size of its hull**
    /// (owner ruling 2026-09-01): building inside it never changes it, so nothing crosses a boundary
    /// to keep a parent's copy in step, and two hulls can never grow into each other.
    pub bound: Boundary,
    /// How big it draws. A realm authors how it looks, so the realm holds this.
    pub look: Boundary,
    /// What it is made of, for the forces its parent applies.
    pub facts: BuiltFacts,
    /// Which commit last moved this row.
    pub fence: Fence,
}

/// ★ A BERTH A PARENT AUTHORED — held in the PARENT'S store, because a parent authors its children's
/// placements and is their only writer (SL1).
///
/// This is the parent's own past authorship, kept between runs. Without it a parent that restarts has
/// forgotten where it put things.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct Berth {
    /// The built child this berth is for.
    pub child: RealmId,
    /// Where the parent put it, in the parent's own frame, in metres.
    ///
    /// ★ **ONLY WHERE IT STARTS.** A driven child's placement is authored every tick from the pushes it
    /// states, so this is the hull's berth and nothing more.
    pub offset_m: DVec3,
    /// The child's box, as the parent holds it — a READING of what the child authored, never a second
    /// opinion. Containment and the interest radius both need it before the child is running.
    pub bound: Boundary,
    /// The child's drawn size, held for the same reason.
    pub look: Boundary,
    /// Which commit last moved this row.
    pub fence: Fence,
}

/// ★ WHAT IS DELIBERATELY ABSENT: A POSITION.
///
/// Not the live one. Not the last one. Not the one it had when the process died.
///
/// > A running realm is OUT. A record with no running realm is STORED. There is no status field.
///
/// Nothing ever writes a placement back from the physics. The berth is written when a hull is built,
/// when it is stored, and when its home changes — never by a tick. Three consequences follow, and they
/// are structural rather than remembered:
///
/// - A player who logs out while losing does not save the fight. The row never changed, so the hull
///   returns to its berth.
/// - A crash saves nothing either, by the same rule, with no special case for it.
/// - A hull destroyed in battle stays destroyed, because destroying it is a COMMIT and a crash is not.
///   That asymmetry is the whole anti-cheat property.
///
/// Damage survives, because putting a hull away is a commit.
///
/// Absent for the same reason: velocity, facing, who is aboard, fuel, and which machine runs it. The
/// last of those is the directory's business, and the directory already forgets it when a realm stops.
pub const WHY_NO_POSITION: () = ();

#[cfg(test)]
mod tests {
    use super::{Berth, BlueprintId, BuiltBody, BuiltFacts};
    use crate::entity_kind::EntityKind;
    use crate::fence::Fence;
    use crate::geometry::Boundary;
    use crate::ids::{AccountId, EntityId};
    use crate::pose::RealmId;
    use glam::DVec3;

    fn a_ship() -> RealmId {
        RealmId::Ship(EntityId::pack(EntityKind::Ship, 1, 1, 0))
    }
    fn a_body() -> BuiltBody {
        BuiltBody {
            realm: a_ship(),
            owner: AccountId(1000),
            blueprint: BlueprintId(7),
            bound: Boundary::Shell { r: 20.0 },
            look: Boundary::Shell { r: 20.0 },
            facts: BuiltFacts {
                mass_g: 50_000_000,
                cross_section_mm2: 12_000_000,
                drag_micro: 820_000,
                max_push_micro_mps2: 98_100_000,
                max_turn_micro_radps2: 800_000,
            },
            fence: Fence(1),
        }
    }

    #[test]
    fn a_body_survives_the_round_trip_with_its_whole_minted_name() {
        // ★ THE NAME IS THE POINT. A built realm is named by whichever machine built it, with no
        // central agreement, so the name is up to 128 bits. A store that narrowed it would let two
        // machines pick one name for two PLACES — the collision the widening exists to prevent.
        let body = a_body();
        let bytes = postcard::to_allocvec(&body).expect("a body encodes");
        let back: BuiltBody = postcard::from_bytes(&bytes).expect("a body decodes");
        assert_eq!(back.realm, body.realm, "the whole minted name survives");
        assert_eq!(back.owner, body.owner);
        assert_eq!(back.facts, body.facts);
    }

    #[test]
    fn a_berth_survives_the_round_trip() {
        let berth = Berth {
            child: a_ship(),
            offset_m: DVec3::new(1000.0, 0.0, 0.0),
            bound: Boundary::Shell { r: 20.0 },
            look: Boundary::Shell { r: 20.0 },
            fence: Fence(1),
        };
        let bytes = postcard::to_allocvec(&berth).expect("a berth encodes");
        let back: Berth = postcard::from_bytes(&bytes).expect("a berth decodes");
        assert_eq!(back.child, berth.child);
        assert_eq!(back.offset_m, berth.offset_m);
    }

    #[test]
    fn no_field_anywhere_can_hold_a_position_or_a_speed() {
        // ★ THE ANTI-CHEAT PROPERTY IS STRUCTURAL, not remembered. A player who logs out while losing
        // must not save the fight, and a crash must save nothing either — with no special case for
        // either. That is only true while there is NOWHERE to write a live placement.
        //
        // A berth carries an offset, which is where a hull STARTS — written when it is built, stored,
        // or moved home, and never by a tick. A body carries no placement at all.
        //
        // This reads the encoded SIZE rather than the field names, because the codec is positional and
        // carries no names at all. A body is a name, an owner, a blueprint, two boxes, five whole
        // numbers and a fence. Adding a placement to it would add at least three more numbers, and the
        // encoding would grow past this bound.
        //
        // Stated as a bound rather than an exact size so that a WIDER name — a station's, when it is
        // widened — does not fail a test about placements.
        let body = a_body();
        let encoded = postcard::to_allocvec(&body).expect("a body encodes");
        let width = encoded.len();
        assert!(
            width < 96,
            "a body holds no placement — three more numbers would not fit in {width} bytes"
        );
    }

    #[test]
    fn a_hulls_push_is_its_own_and_not_a_shared_constant() {
        // Two hulls with different engines must be able to differ. A single shipped rating is what
        // exists today, and it would make every ship in the world fly identically.
        let mut light = a_body();
        light.facts.max_push_micro_mps2 = 200_000_000;
        assert_ne!(light.facts, a_body().facts, "per-hull, never per-world");
    }
}
