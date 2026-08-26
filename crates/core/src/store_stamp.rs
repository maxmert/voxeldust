//! THE SAVED-DATA STAMP — the label every durable file carries, and the pure decision that reads it.
//!
//! Owns: what a durable file states about the world that wrote it, and the four outcomes of comparing
//! that statement with the world opening it.
//!
//! Does NOT own: any file, any database, any input/output. This crate is pure by law, and the decision
//! lives here rather than beside the database on purpose — the coverage gate holds this crate to every
//! branch, and the decision is the part that must never be wrong. The plumbing that reads bytes and
//! calls [`StoreStamp::verify`] lives in the io crate at its ratcheted floor.
//!
//! # Why this exists (owner ruling 2026-08-24, Q1 condition 1; ledger D-48)
//!
//! Nothing this project writes to disk carries a format version. The galaxy's coordinate step is about
//! to change from a thousandth of a metre to two metres, and **a position written under one step and
//! read under another puts a player a thousand times too far out.** Nothing crashes. Nothing is logged.
//! The player is simply in the wrong place, and no test can tell.
//!
//! The stamp is the refusal that catches it. It is compared BEFORE any row is read, because our encoding
//! is positional and not self-describing: two different record shapes decode from the same bytes with no
//! error at all, so "read it and see if it looks right" is not available to us.
//!
//! # What is in it, and the rule for what may be added
//!
//! Every field here is COMPARED FOR EQUALITY. A field with a lenient check would be code no test can
//! drive to both answers, which the coverage rule forbids — so the rule is simple: **if it does not
//! change the meaning of the bytes, it does not belong in the stamp.** Provenance for a human belongs in
//! a log line, not here.
//!
//! The two generations are FOLDED over the values themselves rather than typed by hand. A number
//! somebody must remember to increment is a number somebody forgets: the coordinate generation moves
//! because a tier's metres-per-cell moved, and it cannot fail to.

use crate::ids::EpochId;
use crate::pose::Tier;
use serde::{Deserialize, Serialize};

/// The stamp's own layout version — the ONE number a human bumps, and only when the stamp's own shape
/// changes. Everything else in the stamp derives itself.
///
/// It exists because the stamp is read before anything else: if its own shape changed, the reader would
/// mis-decode the very record that was supposed to protect it.
pub const STAMP_LAYOUT_VERSION: u16 = 1;

/// Which durable file this is. A file states its own role so an outbox can never be opened as a saga
/// log — the encoding is positional, so that mistake reads as data rather than as an error.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[repr(u8)]
pub enum StoreRole {
    /// The orchestrator's directory + saga record.
    Directory = 0,
    /// A node's durable outbox.
    Outbox = 1,
    /// The orchestrator's launch ledger — the record of which processes it started.
    ///
    /// ★ ITS REFUSAL POLICY DIFFERS FROM THE OTHERS AND THAT IS DELIBERATE. Children are started into
    /// their own process group and are NOT killed when their parent drops them, and this ledger is the
    /// only record of what to reap. A parent that refuses to open it and exits ORPHANS every child it
    /// started. So a caller opening this role must reap before it exits — see the io crate's opener.
    LaunchLedger = 2,
    /// A client's on-disk cache of the star catalogue (owner Q4 condition 4). Named here rather than
    /// later because the whole reason this type lives in the pure crate is that the client shares it.
    ClientCatalogue = 3,
}

impl StoreRole {
    /// Every role — the totality list a test walks so a new role cannot be added without being driven.
    pub const ALL: [StoreRole; 4] = [
        StoreRole::Directory,
        StoreRole::Outbox,
        StoreRole::LaunchLedger,
        StoreRole::ClientCatalogue,
    ];

    /// The role's name for a refusal message an operator has to act on at three in the morning.
    #[must_use]
    pub fn label(self) -> &'static str {
        match self {
            StoreRole::Directory => "directory+saga",
            StoreRole::Outbox => "outbox",
            StoreRole::LaunchLedger => "launch-ledger",
            StoreRole::ClientCatalogue => "client-catalogue",
        }
    }
}

/// What a durable file states about the world that wrote it.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct StoreStamp {
    /// The stamp's own shape — see [`STAMP_LAYOUT_VERSION`].
    pub layout: u16,
    /// What this file is.
    pub role: StoreRole,
    /// The seed the world was generated from. Env-only today and recorded nowhere durable, which means
    /// a store from one world opens silently against another — every position in it referring to bodies
    /// that are not there.
    pub universe_seed: u64,
    /// The universe epoch this file's records were written under.
    pub epoch: EpochId,
    /// THE COORDINATE GENERATION — folded over the metres-per-cell of every tier. It moves when, and
    /// only when, a coordinate unit moves, so the change that would silently misplace a player by a
    /// factor of a thousand cannot be made without this number moving with it.
    pub coordinate_generation: u64,
    /// THE WORLD-LAW GENERATION — folded over the named constants that shape the forest. It moves when
    /// the world's own geometry moves, so records describing bodies at one set of distances are refused
    /// by a process that would place those bodies somewhere else.
    pub world_generation: u64,
}

impl StoreStamp {
    /// Build the label THIS build would write, from the facts that decide whether another build may
    /// read it.
    ///
    /// The two generations are derived here rather than passed in, so no caller can state one — a
    /// number a caller could state is a number a caller could state wrongly. `world_constants` is the
    /// caller's own list of the values that shape its world; this folds them and cannot ask what any of
    /// them means (SL4: nothing here may name a motion).
    #[must_use]
    pub fn new(
        role: StoreRole,
        universe_seed: u64,
        epoch: EpochId,
        world_constants: &[f64],
    ) -> StoreStamp {
        StoreStamp {
            layout: STAMP_LAYOUT_VERSION,
            role,
            universe_seed,
            epoch,
            coordinate_generation: coordinate_generation(),
            world_generation: world_generation(world_constants),
        }
    }
}

/// Why a durable file was refused. Each arm names the field and carries BOTH values, because a refusal
/// an operator cannot act on becomes an unofficial delete — which is the loss the refusal exists to
/// prevent.
#[derive(Clone, Copy, Debug, PartialEq, Eq, thiserror::Error)]
pub enum StampRefusal {
    /// The file holds records but no stamp: it was written before stamps existed, so the shape of its
    /// bytes is unknown. We refuse rather than guess (owner ruling 2026-08-24: refuse, do not convert).
    #[error(
        "this file holds data but carries no format label, so it was written before labels existed and \
         the shape of its contents cannot be known. Refusing rather than guessing. To start a fresh \
         world here, set VD_STORE_ALLOW_GENESIS=1, which DELETES what is in this file."
    )]
    UnstampedAndNotEmpty,
    /// The stamp's own shape differs, so nothing else in it can be trusted to mean what it says.
    #[error("the format label's own layout is {found}; this build writes {expected}")]
    Layout { found: u16, expected: u16 },
    /// The file is for a different job.
    #[error("this file is a {found} store; this process is opening it as a {expected} store")]
    Role {
        found: &'static str,
        expected: &'static str,
    },
    /// A different world.
    #[error(
        "this file was written for world seed {found}; this process is running world seed {expected}"
    )]
    UniverseSeed { found: u64, expected: u64 },
    /// A different epoch.
    #[error("this file was written in universe epoch {found}; this process is in epoch {expected}")]
    Epoch { found: u64, expected: u64 },
    /// THE ONE THIS WHOLE MECHANISM EXISTS FOR.
    #[error(
        "this file's positions are counted in different units than this build uses (coordinate \
         generation {found} against {expected}). A position written under one unit and read under \
         another is silently wrong by the ratio between them — nothing would crash, and a player would \
         simply be in the wrong place. Refusing. There is no conversion: delete the file, or set \
         VD_STORE_ALLOW_GENESIS to start it over. Everything it held is discarded either way."
    )]
    CoordinateGeneration { found: u64, expected: u64 },
    /// The world's own geometry moved.
    #[error(
        "this file describes a world whose distances differ from this build's (world generation \
         {found} against {expected}). Its records name bodies this process would place elsewhere."
    )]
    WorldGeneration { found: u64, expected: u64 },
}

/// What the opener should do, once the stamp has been read (or found absent).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum StampVerdict {
    /// The file carries no stamp and holds nothing. Write the stamp and proceed — this is a new file.
    WriteAndProceed,
    /// The stamp is present and every field matches. Proceed.
    Proceed,
}

/// THE DECISION, and the whole reason this module is in the pure crate.
///
/// `found` is what the file said, or `None` if it said nothing. `empty` is whether the file holds any
/// records at all — which is what separates "a new file" from "a file from before labels existed".
///
/// Written as a straight sequence of named comparisons rather than a derived equality so that a
/// refusal can say WHICH field disagreed and with what. `PartialEq` on the whole stamp would be one
/// line and would produce a message no operator could act on.
///
/// # Errors
/// [`StampRefusal`] naming the first field that disagreed.
pub fn verify(
    found: Option<StoreStamp>,
    expected: &StoreStamp,
    empty: bool,
) -> Result<StampVerdict, StampRefusal> {
    let Some(found) = found else {
        // ABSENT. The two cases differ by whether anything else is in the file, and conflating them is
        // exactly how a store from before this change would be adopted and mis-read.
        if empty {
            return Ok(StampVerdict::WriteAndProceed);
        }
        return Err(StampRefusal::UnstampedAndNotEmpty);
    };
    // LAYOUT FIRST: if the stamp's own shape differs, no field below means what it appears to mean.
    if found.layout != expected.layout {
        return Err(StampRefusal::Layout {
            found: found.layout,
            expected: expected.layout,
        });
    }
    if found.role != expected.role {
        return Err(StampRefusal::Role {
            found: found.role.label(),
            expected: expected.role.label(),
        });
    }
    if found.universe_seed != expected.universe_seed {
        return Err(StampRefusal::UniverseSeed {
            found: found.universe_seed,
            expected: expected.universe_seed,
        });
    }
    if found.epoch != expected.epoch {
        return Err(StampRefusal::Epoch {
            found: found.epoch.0,
            expected: expected.epoch.0,
        });
    }
    if found.coordinate_generation != expected.coordinate_generation {
        return Err(StampRefusal::CoordinateGeneration {
            found: found.coordinate_generation,
            expected: expected.coordinate_generation,
        });
    }
    if found.world_generation != expected.world_generation {
        return Err(StampRefusal::WorldGeneration {
            found: found.world_generation,
            expected: expected.world_generation,
        });
    }
    Ok(StampVerdict::Proceed)
}

/// THE COORDINATE GENERATION — folded over every tier's metres-per-cell.
///
/// Derived, never typed. Change a tier's unit and this moves; there is no step at which somebody could
/// forget. It reads [`Tier::ALL`], so ADDING a tier moves it too — which is correct, because a build
/// that knows a tier the writer did not cannot be trusted to read that writer's positions.
///
/// The fold is over the bit pattern of each edge, not its printed value: two units that differ below
/// printing precision are still different units.
///
/// `const` so it can be compared BEFORE a byte is exchanged: the protocol contract folds the same value
/// at compile time, which is what lets two processes refuse each other at the handshake rather than
/// discovering the disagreement in a position somebody has already acted on.
#[must_use]
pub const fn coordinate_generation() -> u64 {
    let mut acc = FNV_OFFSET;
    // An index loop rather than a `for`, because a `for` is not permitted in a `const fn`. Same fold,
    // same order, same value.
    let mut i = 0;
    while i < Tier::ALL.len() {
        acc = fnv_u64(acc, Tier::ALL[i].cell_edge_m().to_bits());
        i += 1;
    }
    acc
}

/// THE WORLD-LAW GENERATION — folded over the named numbers a caller states as the ones that shape its
/// world. Kept as an argument rather than reaching into the world generator because this crate carries
/// no edge to it, and because the crossing path may not name a motion (SL4): the caller states its
/// constants, this folds them, and nothing here can ask what they mean.
#[must_use]
pub fn world_generation(constants: &[f64]) -> u64 {
    let mut acc = FNV_OFFSET;
    for c in constants {
        acc = fnv_u64(acc, c.to_bits());
    }
    acc
}

/// FNV-1a's 64-bit offset basis.
const FNV_OFFSET: u64 = 0xcbf2_9ce4_8422_2325;
/// FNV-1a's 64-bit prime.
const FNV_PRIME: u64 = 0x0000_0100_0000_01b3;

/// One 64-bit value folded into an FNV-1a accumulator, byte by byte. Dependency-free and stable across
/// builds and machines — a generation that changed with the compiler would refuse every store after an
/// upgrade. The same reasoning the boot counter's checksum already follows.
const fn fnv_u64(mut acc: u64, v: u64) -> u64 {
    let bytes = v.to_le_bytes();
    let mut i = 0;
    while i < bytes.len() {
        acc ^= bytes[i] as u64;
        acc = acc.wrapping_mul(FNV_PRIME);
        i += 1;
    }
    acc
}

#[cfg(test)]
mod tests {
    use super::*;

    fn stamp() -> StoreStamp {
        StoreStamp::new(StoreRole::Directory, 2298, EpochId(7), &[1.0, 2.0])
    }

    #[test]
    fn the_constructor_derives_everything_a_caller_must_not_state() {
        // The two generations and the layout are DERIVED, so no caller can state one wrongly. Asserted
        // rather than assumed, because the whole mechanism rests on them being un-stateable.
        let s = StoreStamp::new(StoreRole::Outbox, 5, EpochId(1), &[3.0]);
        assert_eq!(s.layout, STAMP_LAYOUT_VERSION);
        assert_eq!(s.coordinate_generation, coordinate_generation());
        assert_eq!(s.world_generation, world_generation(&[3.0]));
        assert_eq!(s.role, StoreRole::Outbox);
        assert_eq!(s.universe_seed, 5);
        assert_eq!(s.epoch, EpochId(1));
    }

    #[test]
    fn an_empty_unstamped_file_is_written_and_proceeds() {
        assert_eq!(
            verify(None, &stamp(), true),
            Ok(StampVerdict::WriteAndProceed)
        );
    }

    #[test]
    fn an_unstamped_file_that_holds_data_is_refused() {
        // THE OWNER'S RULING, 2026-08-24: refuse, do not convert. Bytes written before labels existed
        // have an unknown shape, and our encoding is positional — reading them under today's shapes
        // yields records, not errors. Guessing is the one thing that must not happen here.
        assert_eq!(
            verify(None, &stamp(), false),
            Err(StampRefusal::UnstampedAndNotEmpty)
        );
    }

    #[test]
    fn a_matching_stamp_proceeds() {
        assert_eq!(
            verify(Some(stamp()), &stamp(), false),
            Ok(StampVerdict::Proceed)
        );
        // And an empty file with a matching stamp is the ordinary restart case.
        assert_eq!(
            verify(Some(stamp()), &stamp(), true),
            Ok(StampVerdict::Proceed)
        );
    }

    #[test]
    fn each_field_refuses_on_its_own_and_names_both_values() {
        // EVERY ARM DRIVEN SEPARATELY. A refusal that fired on the wrong field would send an operator
        // to the wrong cause, so each one is proven to fire for its own reason and to carry both
        // numbers — the thing that makes the message actionable.
        let want = stamp();

        let mut f = stamp();
        f.layout = 99;
        assert_eq!(
            verify(Some(f), &want, false),
            Err(StampRefusal::Layout {
                found: 99,
                expected: STAMP_LAYOUT_VERSION
            })
        );

        let mut f = stamp();
        f.role = StoreRole::Outbox;
        assert_eq!(
            verify(Some(f), &want, false),
            Err(StampRefusal::Role {
                found: "outbox",
                expected: "directory+saga"
            })
        );

        let mut f = stamp();
        f.universe_seed = 1;
        assert_eq!(
            verify(Some(f), &want, false),
            Err(StampRefusal::UniverseSeed {
                found: 1,
                expected: 2298
            })
        );

        let mut f = stamp();
        f.epoch = EpochId(8);
        assert_eq!(
            verify(Some(f), &want, false),
            Err(StampRefusal::Epoch {
                found: 8,
                expected: 7
            })
        );

        let mut f = stamp();
        f.coordinate_generation = 42;
        assert_eq!(
            verify(Some(f), &want, false),
            Err(StampRefusal::CoordinateGeneration {
                found: 42,
                expected: want.coordinate_generation
            })
        );

        let mut f = stamp();
        f.world_generation = 42;
        assert_eq!(
            verify(Some(f), &want, false),
            Err(StampRefusal::WorldGeneration {
                found: 42,
                expected: want.world_generation
            })
        );
    }

    #[test]
    fn the_layout_is_checked_before_anything_it_would_make_meaningless() {
        // ORDERING, ASSERTED. If the stamp's own shape differs, every other field is bytes read under
        // the wrong shape — so a file that disagrees about EVERYTHING must still refuse on layout, or
        // the message names a cause the operator cannot act on.
        let mut f = stamp();
        f.layout = 99;
        f.role = StoreRole::Outbox;
        f.universe_seed = 1;
        assert_eq!(
            verify(Some(f), &stamp(), false),
            Err(StampRefusal::Layout {
                found: 99,
                expected: STAMP_LAYOUT_VERSION
            })
        );
    }

    #[test]
    fn the_coordinate_generation_moves_when_a_unit_moves_and_not_otherwise() {
        // THE GATE THE WHOLE MECHANISM RESTS ON. It is derived from the tier table, so it cannot fail
        // to move when a unit moves. Proven here over the fold itself — the shipped table is one input
        // to it, and slice S8 changes that table.
        let shipped = coordinate_generation();
        assert_eq!(shipped, coordinate_generation(), "the fold must be stable");

        // A unit that moves by ONE BIT — far below anything printable — is a different unit.
        let a = world_generation(&[Tier::Fine.cell_edge_m()]);
        let nudged = f64::from_bits(Tier::Fine.cell_edge_m().to_bits() + 1);
        let b = world_generation(&[nudged]);
        assert_ne!(
            a, b,
            "a unit that differs below printing precision still differs"
        );

        // And the fold is order-sensitive, so a table whose entries were reordered is a different
        // table — which it is, because a tier's identity is its position in that list.
        assert_ne!(world_generation(&[1.0, 2.0]), world_generation(&[2.0, 1.0]));
    }

    #[test]
    fn the_world_generation_folds_what_it_is_given_and_nothing_else() {
        assert_eq!(world_generation(&[]), FNV_OFFSET);
        assert_ne!(world_generation(&[0.0]), world_generation(&[]));
        // Positive and negative zero are the same number and DIFFERENT bits. The fold reads bits, so it
        // separates them — stated because it is surprising, and because a world constant that flipped
        // sign to zero would otherwise refuse every store for no visible reason.
        assert_ne!(world_generation(&[0.0]), world_generation(&[-0.0]));
    }

    #[test]
    fn every_role_has_a_label_and_they_are_all_different() {
        // TOTALITY over the role list: a role added without a label, or sharing one, would produce a
        // refusal message naming the wrong file.
        let labels: Vec<&str> = StoreRole::ALL.iter().map(|r| r.label()).collect();
        assert_eq!(labels.len(), StoreRole::ALL.len());
        let mut sorted = labels.clone();
        sorted.sort_unstable();
        sorted.dedup();
        assert_eq!(
            sorted.len(),
            labels.len(),
            "two roles share a label: {labels:?}"
        );
    }

    #[test]
    fn a_stamp_survives_the_round_trip_it_will_actually_take() {
        // The stamp is written and read back with the same encoder every durable record uses. If it
        // could not round-trip, the refusal would fire on a file this build itself had just written.
        let s = stamp();
        let bytes = postcard::to_allocvec(&s).expect("stamp encodes");
        let back: StoreStamp = postcard::from_bytes(&bytes).expect("stamp decodes");
        assert_eq!(back, s);
    }
}
