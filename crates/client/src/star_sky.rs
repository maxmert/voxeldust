//! THE STAR CATALOGUE's client half (S11) — parts in, one sky out, and a damaged one refused.
//!
//! The galaxy's stars arrive in parts, because the whole sky is 7.0 MB at the target census and no
//! carrier takes that in one piece. This assembles them.
//!
//! ★ A SKY IS DRAWN ONLY WHEN IT IS WHOLE. Half a catalogue is not a smaller galaxy — it is a galaxy
//! with holes in it, and holes look exactly like stars that do not exist. So parts accumulate
//! unseen, and the sky becomes drawable in one step when the last part lands.
//!
//! ★ THE GENERATION IS THE IDENTITY, not the arrival order. It is folded from the catalogue's own
//! content, so two parts carrying the same generation belong to the same sky by construction and two
//! carrying different ones cannot be spliced into a galaxy that never existed. A newer generation
//! REPLACES a partial older one rather than merging with it.
//!
//! ★ EVERY REFUSAL IS COUNTED AND NAMED. The owner's condition is that a byte-flipped cache produces
//! a counted refusal and a re-request, NEVER a drawn frame — and the reason is the encoding: postcard
//! is positional, so damaged bytes do not fail to decode, they decode into NONSENSE. A star at 10^300
//! metres is a perfectly well-formed row. The bounds below are what turn nonsense back into a refusal.

use vd_core::look::StarRow;
use vd_core::pose::CELL_DOMAIN_MAX;

/// The most parts one catalogue may claim to have. A damaged header could otherwise announce four
/// billion parts and hold the assembler open forever, waiting for a sky that is not coming.
///
/// Sized well past what the census needs: 150,000 stars at ~47 bytes is 7.0 MB, which is thousands of
/// parts at any sane carrier budget, never millions.
pub const MAX_PARTS: u32 = 100_000;

/// The most stars one catalogue may carry. The census is 150,000; this leaves room for the galaxy to
/// grow without leaving room for a damaged length to allocate the machine out of memory.
pub const MAX_STARS: usize = 4_000_000;

/// Why a part was refused — each its own reason, so a refusal names itself rather than arriving as a
/// single "bad catalogue" that could mean anything.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SkyRefusal {
    /// `parts` was zero — a catalogue with no parts cannot ever complete.
    NoParts,
    /// `parts` claimed more than [`MAX_PARTS`].
    TooManyParts,
    /// `part >= parts` — a part outside the set it claims to belong to.
    PartOutOfRange,
    /// The accumulated stars would exceed [`MAX_STARS`].
    TooManyStars,
    /// A row carries a position outside the lawful cell domain, or a brightness that is not a number.
    /// This is the arm a byte-flipped cache lands in.
    Unrepresentable,
}

/// What accepting a part did.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SkyIngest {
    /// Stored. The sky is not whole yet.
    Applied,
    /// Stored, and that was the last part — [`StarSky::complete`] now answers.
    Completed,
    /// A newer generation arrived; the partial sky in hand was discarded for it.
    Superseded,
    /// Refused, and counted. Nothing was stored.
    Refused(SkyRefusal),
}

/// The client's assembling star catalogue.
#[derive(Debug, Default)]
pub struct StarSky {
    generation: Option<u64>,
    parts_expected: u32,
    /// Parts held by index, so a re-delivered part REPLACES rather than duplicating its stars.
    held: std::collections::BTreeMap<u32, Vec<StarRow>>,
    /// Every refusal, counted. A sky that never completes must be explainable from the counters
    /// alone — "the client shows no stars" is otherwise a report with nowhere to start.
    pub refused: u64,
    /// Skies abandoned because a newer generation arrived mid-assembly.
    pub superseded: u64,
}

impl StarSky {
    /// Take one part of a catalogue.
    pub fn accept(
        &mut self,
        generation: u64,
        part: u32,
        parts: u32,
        rows: Vec<StarRow>,
    ) -> SkyIngest {
        if let Some(reason) = header_refusal(part, parts) {
            self.refused += 1;
            return SkyIngest::Refused(reason);
        }
        if rows.iter().any(|r| !row_representable(r)) {
            self.refused += 1;
            return SkyIngest::Refused(SkyRefusal::Unrepresentable);
        }
        // A DIFFERENT SKY REPLACES, never merges: the generation is folded from content, so parts of
        // two catalogues describe two galaxies and splicing them would build a third that never existed.
        let superseded = self.generation.is_some_and(|g| g != generation);
        if superseded {
            self.superseded += 1;
            self.held.clear();
        }
        if self.generation != Some(generation) {
            self.generation = Some(generation);
            self.parts_expected = parts;
        }
        let held_stars: usize = self.held.values().map(Vec::len).sum();
        if held_stars.saturating_add(rows.len()) > MAX_STARS {
            self.refused += 1;
            return SkyIngest::Refused(SkyRefusal::TooManyStars);
        }
        self.held.insert(part, rows);
        if superseded {
            return SkyIngest::Superseded;
        }
        if self.held.len() as u32 == self.parts_expected {
            SkyIngest::Completed
        } else {
            SkyIngest::Applied
        }
    }

    /// The whole sky, or `None` while parts are still missing. Ordered by part, so the stars come out
    /// in exactly the order the catalogue stated them — which is what the generation was folded over.
    #[must_use]
    pub fn complete(&self) -> Option<Vec<StarRow>> {
        if self.generation.is_none() || self.held.len() as u32 != self.parts_expected {
            return None;
        }
        Some(self.held.values().flatten().copied().collect())
    }

    /// Which sky is in hand, whole or partial.
    #[must_use]
    pub fn generation(&self) -> Option<u64> {
        self.generation
    }
}

/// The header's own bounds — monomorphic, so every arm is covered once (HR5).
fn header_refusal(part: u32, parts: u32) -> Option<SkyRefusal> {
    if parts == 0 {
        return Some(SkyRefusal::NoParts);
    }
    if parts > MAX_PARTS {
        return Some(SkyRefusal::TooManyParts);
    }
    if part >= parts {
        return Some(SkyRefusal::PartOutOfRange);
    }
    None
}

/// Is this row a star, or is it damage? A position outside the lawful cell domain cannot be drawn at
/// all, and a brightness that is not a number poisons every size derived from it.
fn row_representable(r: &StarRow) -> bool {
    let c = r.cell;
    let in_domain = (c.x.unsigned_abs() <= CELL_DOMAIN_MAX.unsigned_abs())
        & (c.y.unsigned_abs() <= CELL_DOMAIN_MAX.unsigned_abs())
        & (c.z.unsigned_abs() <= CELL_DOMAIN_MAX.unsigned_abs());
    in_domain & r.luma_lsun.is_finite() & (r.luma_lsun >= 0.0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use vd_core::glam::I64Vec3;
    use vd_core::pose::RealmId;

    fn star(n: u64) -> StarRow {
        StarRow {
            realm: RealmId::System(n),
            cell: I64Vec3::new(1_313_684_865_644_610_304 + n as i64, 7, -3),
            class_code: 6,
            luma_lsun: 0.25,
        }
    }

    /// ★ A SKY IS DRAWN ONLY WHEN IT IS WHOLE. Half a catalogue is not a smaller galaxy — it is a
    /// galaxy with holes, and a hole looks exactly like a star that does not exist. So the assembler
    /// must answer NOTHING until the last part lands, and then everything at once.
    #[test]
    fn a_sky_is_invisible_until_its_last_part_arrives() {
        let mut sky = StarSky::default();
        assert_eq!(sky.accept(7, 0, 3, vec![star(1)]), SkyIngest::Applied);
        assert!(sky.complete().is_none(), "one part of three draws nothing");
        assert_eq!(sky.accept(7, 2, 3, vec![star(3)]), SkyIngest::Applied);
        assert!(sky.complete().is_none(), "a GAP draws nothing either");
        assert_eq!(sky.accept(7, 1, 3, vec![star(2)]), SkyIngest::Completed);
        // …and it comes out in the catalogue's own order, which is what the generation was folded
        // over — out of order it would be a different sky by its own identity.
        assert_eq!(
            sky.complete().expect("whole"),
            vec![star(1), star(2), star(3)]
        );
    }

    /// A RE-DELIVERED PART REPLACES, never duplicates. The carrier may deliver twice; a sky with one
    /// star drawn two thousand times is not a rendering problem, it is a wrong galaxy.
    #[test]
    fn a_re_delivered_part_replaces_rather_than_duplicating() {
        let mut sky = StarSky::default();
        assert_eq!(sky.accept(7, 0, 2, vec![star(1)]), SkyIngest::Applied);
        assert_eq!(sky.accept(7, 0, 2, vec![star(1)]), SkyIngest::Applied);
        assert_eq!(sky.accept(7, 1, 2, vec![star(2)]), SkyIngest::Completed);
        assert_eq!(sky.complete().expect("whole"), vec![star(1), star(2)]);
    }

    /// ★ TWO SKIES ARE NEVER SPLICED. The generation is folded from the catalogue's own content, so
    /// parts carrying different ones describe different galaxies. Merging them would build a third
    /// that never existed — and it would look perfectly plausible on screen.
    #[test]
    fn a_newer_generation_replaces_a_partial_sky_and_never_merges_with_it() {
        let mut sky = StarSky::default();
        assert_eq!(sky.accept(7, 0, 2, vec![star(1)]), SkyIngest::Applied);
        assert_eq!(sky.accept(8, 0, 1, vec![star(9)]), SkyIngest::Superseded);
        assert_eq!(sky.superseded, 1);
        assert_eq!(sky.generation(), Some(8));
        assert_eq!(
            sky.complete().expect("the new sky is whole"),
            vec![star(9)],
            "not one star of the abandoned sky survived into this one"
        );
    }

    /// ★ EVERY WAY A DAMAGED HEADER CAN LIE, refused by name and counted. postcard is POSITIONAL: a
    /// byte-flipped cache does not fail to decode, it decodes into nonsense — so these are the arms
    /// that turn nonsense back into a refusal instead of a drawn frame.
    #[test]
    fn a_damaged_header_is_refused_by_name_and_never_drawn() {
        let mut sky = StarSky::default();
        // ZERO PARTS: a catalogue that can never complete would hold the assembler open forever.
        assert_eq!(
            sky.accept(7, 0, 0, vec![star(1)]),
            SkyIngest::Refused(SkyRefusal::NoParts)
        );
        // A COUNT PAST THE CAP: four billion parts is a damaged length, not a big galaxy.
        assert_eq!(
            sky.accept(7, 0, MAX_PARTS + 1, vec![star(1)]),
            SkyIngest::Refused(SkyRefusal::TooManyParts)
        );
        // A PART OUTSIDE ITS OWN SET.
        assert_eq!(
            sky.accept(7, 3, 3, vec![star(1)]),
            SkyIngest::Refused(SkyRefusal::PartOutOfRange)
        );
        assert_eq!(sky.refused, 3, "each refusal counted");
        assert!(sky.complete().is_none(), "and nothing became drawable");
        assert_eq!(sky.generation(), None, "a refused part changes no state");
    }

    /// ★ A STAR THAT CANNOT BE DRAWN IS REFUSED, and this is the arm a byte-flipped cache lands in.
    /// One flipped byte turns a distance into 10^300 metres, or a brightness into "not a number" —
    /// both perfectly well-formed rows, and both unbounded draws.
    #[test]
    fn an_unrepresentable_star_is_refused_and_the_whole_part_with_it() {
        let mut sky = StarSky::default();
        let mut far = star(1);
        far.cell.x = i64::MAX; // past the lawful cell domain
        assert_eq!(
            sky.accept(7, 0, 1, vec![far]),
            SkyIngest::Refused(SkyRefusal::Unrepresentable)
        );
        let mut nan = star(2);
        nan.luma_lsun = f64::NAN;
        assert_eq!(
            sky.accept(7, 0, 1, vec![nan]),
            SkyIngest::Refused(SkyRefusal::Unrepresentable)
        );
        let mut neg = star(3);
        neg.luma_lsun = -1.0; // a negative brightness poisons every size derived from it
        assert_eq!(
            sky.accept(7, 0, 1, vec![neg]),
            SkyIngest::Refused(SkyRefusal::Unrepresentable)
        );
        assert_eq!(sky.refused, 3);
        // THE WHOLE PART IS REFUSED, not the bad row alone: a part with one star quietly removed is a
        // galaxy with a hole in it, which is the failure this is here to prevent, not a repair.
        assert!(sky.complete().is_none());
        // …and a clean part of the same sky still lands, so the refusal is about the damage and not
        // about the assembler having given up.
        assert_eq!(sky.accept(7, 0, 1, vec![star(1)]), SkyIngest::Completed);
    }

    /// A star count past the cap is refused rather than allocated — a damaged length must not be able
    /// to spend the machine's memory before anything notices it is damage.
    #[test]
    fn more_stars_than_the_cap_are_refused_rather_than_allocated() {
        let mut sky = StarSky::default();
        let huge: Vec<StarRow> = (0..8).map(star).collect();
        assert_eq!(sky.accept(7, 0, 2, huge.clone()), SkyIngest::Applied);
        // Force the accumulated count past the cap with a second part.
        let mut sky2 = StarSky::default();
        let cap_row = vec![star(1); 1];
        assert_eq!(sky2.accept(7, 0, 2, cap_row), SkyIngest::Applied);
        // A part claiming more stars than the cap allows, alone.
        let over: Vec<StarRow> = std::iter::repeat_n(star(1), MAX_STARS + 1).collect();
        assert_eq!(
            sky2.accept(7, 1, 2, over),
            SkyIngest::Refused(SkyRefusal::TooManyStars)
        );
        assert_eq!(sky2.refused, 1);
    }
}
