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

/// The largest one encoded [`StarRow`] can be. Derived, never guessed: a realm id is an enum tag plus
/// a u64 varint (11 bytes at most), three cell axes are zigzag varints (10 bytes each), the class is
/// one byte, and the brightness is a fixed eight. That totals 50. The value here rounds up and leaves
/// the rest as slack, because this number's only job is to refuse an absurd file unread.
const MAX_ROW_BYTES: usize = 64;

/// The most bytes a cache file may hold. A larger file is refused BEFORE it is decoded.
///
/// Decoding is the work an attacker gets for free from a file nobody validated, so the size test comes
/// first and reads no further than the length. postcard does not pre-allocate from a claimed length —
/// serde caps a sequence's reserved capacity — so the cost a damaged file can impose is the work of
/// decoding, and that work is bounded by this number.
pub const MAX_CACHE_BYTES: usize = MAX_STARS * MAX_ROW_BYTES;

/// Why a cache file was refused. Each reason is its own arm. A refusal names itself, so "the stars did
/// not come back" is a report with somewhere to start.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CacheRefusal {
    /// Larger than [`MAX_CACHE_BYTES`]. Refused unread.
    TooLarge,
    /// postcard could not read the bytes at all.
    Undecodable,
    /// The stamp disagrees with this build or this world. See [`vd_core::store_stamp::verify`].
    ForeignWorld,
    /// More stars than [`MAX_STARS`].
    TooManyStars,
    /// A row holds a position outside the lawful cell domain, or a brightness that is not a number.
    Unrepresentable,
    /// The file held a different number of stars than its header claimed.
    CountMismatch,
    /// The rows do not fold to the generation the file states.
    ///
    /// ★ THIS IS THE ARM THAT CATCHES AN EDIT. postcard is positional. A player who flips a byte in
    /// their own cache does not produce a file that fails to decode — they produce a file that decodes
    /// into a DIFFERENT, perfectly well-formed sky. Nothing above this test would notice. The fold is
    /// what notices, because the generation is derived from the content and the content moved.
    DigestMismatch,
}

/// A cache file's HEADER. The rows follow it, encoded separately.
///
/// ★ WHY THE FILE IS SPLIT. A header that can be read on its own is a header that can be DISBELIEVED
/// on its own. The claimed star count is tested against the cap before a single row is decoded, so a
/// damaged length is refused with the work of reading about forty bytes rather than the work of
/// materialising fifty million rows. Decoding is exactly what a hostile file wants to buy.
///
/// It also lets the fold run over the rows' OWN BYTES, as they lie in the file, rather than over a
/// re-encoding of them. That is the same comparison the server made when it folded the generation.
#[derive(Debug, serde::Serialize, serde::Deserialize)]
struct CacheHeader {
    stamp: vd_core::store_stamp::StoreStamp,
    generation: u64,
    /// What the file CLAIMS it holds. Checked against the cap first, then against the truth.
    stars: u64,
}

/// Write a sky to bytes, under the stamp of the world that produced it.
#[must_use]
pub fn encode_cache(
    stamp: vd_core::store_stamp::StoreStamp,
    generation: u64,
    rows: &[StarRow],
) -> Vec<u8> {
    let mut out = postcard::to_allocvec(&CacheHeader {
        stamp,
        generation,
        stars: rows.len() as u64,
    })
    .unwrap_or_default();
    out.extend_from_slice(&postcard::to_allocvec(rows).unwrap_or_default());
    out
}

/// Read a sky back, and prove it before any star is drawn from it.
///
/// ★ WHY A CACHE NEEDS PROVING AT ALL. The file sits on the player's own disk. The player may edit it.
/// The owner's condition is exact: a byte-flipped cache must produce a counted refusal and a fresh
/// request, and must NEVER produce a drawn frame. That condition cannot be met by decoding alone,
/// because postcard is positional — damaged bytes decode into nonsense rather than failing. A star at
/// 10^300 metres is a well-formed row.
///
/// The catalogue is a DRAWING AID and nothing more. The server validates every warp destination
/// itself. So a forged cache draws a star that nobody else can see, and flies its owner into empty
/// space. That is the whole of the harm, and it lands on the forger.
///
/// The tests run in order, cheapest first, and each one names its own refusal:
///
/// ```text
///   size    -> TooLarge          refused UNREAD; decoding is the work a bad file wants
///   header  -> Undecodable       postcard could not read even the header
///   stamp   -> ForeignWorld      another world, another epoch, or another build's units
///   claim   -> TooManyStars      the CLAIMED count, before one row is decoded
///   rows    -> Undecodable       the rows do not read, or the file holds more than it claimed
///   truth   -> CountMismatch     the file held a different number than it claimed
///   rows    -> Unrepresentable   a position or a brightness that cannot be drawn
///   fold    -> DigestMismatch    the rows are not the sky the file claims
/// ```
///
/// # Errors
/// [`CacheRefusal`] naming the first test that failed.
pub fn decode_cache(
    bytes: &[u8],
    expected: &vd_core::store_stamp::StoreStamp,
) -> Result<(u64, Vec<StarRow>), CacheRefusal> {
    if bytes.len() > MAX_CACHE_BYTES {
        return Err(CacheRefusal::TooLarge);
    }
    let (header, rest) = take_cache_header(bytes)?;
    // The stamp decides whether this file belongs to this world at all. Reused rather than re-stated:
    // a second set of comparisons here could drift from the one every other store uses.
    if vd_core::store_stamp::verify(Some(header.stamp), expected, false).is_err() {
        return Err(CacheRefusal::ForeignWorld);
    }
    // THE CLAIM, tested before the rows are read. This is the whole point of splitting the file.
    if header.stars > MAX_STARS as u64 {
        return Err(CacheRefusal::TooManyStars);
    }
    // Trailing bytes are a decode error in postcard, so a file holding MORE than it encoded is refused
    // here rather than being silently truncated to the part that parsed.
    let rows = decode_cache_rows(rest)?;
    if rows.len() as u64 != header.stars {
        return Err(CacheRefusal::CountMismatch);
    }
    if !rows.iter().all(row_representable) {
        return Err(CacheRefusal::Unrepresentable);
    }
    // THE FOLD, over the rows' OWN BYTES as they lie in the file — the same comparison the server made.
    if vd_core::look::catalogue_generation(rest) != header.generation {
        return Err(CacheRefusal::DigestMismatch);
    }
    Ok((header.generation, rows))
}

/// Hoisted so the generic decode stays a straight-line expression (HR5's monomorphization rule).
fn take_cache_header(bytes: &[u8]) -> Result<(CacheHeader, &[u8]), CacheRefusal> {
    postcard::take_from_bytes::<CacheHeader>(bytes).map_err(|_| CacheRefusal::Undecodable)
}

/// Hoisted for the same reason as [`take_cache_header`].
fn decode_cache_rows(bytes: &[u8]) -> Result<Vec<StarRow>, CacheRefusal> {
    postcard::from_bytes::<Vec<StarRow>>(bytes).map_err(|_| CacheRefusal::Undecodable)
}

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

/// What a liveness beat proved.
///
/// The three arms are exactly the three readings the beat exists to separate; see
/// [`StarSky::beat`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SkyBeat {
    /// The server names the sky this client already holds whole. **Nothing changed, and that is now
    /// PROVEN rather than assumed from silence.** The overwhelmingly common answer on a static galaxy.
    Current,
    /// The server names a DIFFERENT sky from the one held. The catalogue in hand is out of date and the
    /// stars on screen are not the stars the server will fly the player to.
    Stale,
    /// No whole sky is held yet — the client is still assembling, or has never been served. The beat
    /// proves the lane is alive, which is itself worth knowing while parts are still arriving.
    Unheld,
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
    /// Liveness beats that CONFIRMED the held sky. Expected to climb forever on a static galaxy — this
    /// is the counter whose flat line is the defect, not its growth. A caller holding a clock turns
    /// "stopped climbing" into the third reading, *"nobody is working"*.
    pub beats_current: u64,
    /// Liveness beats naming a sky this client does not hold.
    pub beats_stale: u64,
    /// Liveness beats that arrived while no whole sky was held.
    pub beats_unheld: u64,
    /// Cache files refused, by any of the reasons in [`CacheRefusal`]. The owner's condition is that a
    /// damaged cache produces a COUNTED refusal and never a drawn frame, so this is the half that makes
    /// the refusal visible — a player reporting "my stars are gone" is otherwise a report with nowhere
    /// to start.
    pub cache_refused: u64,
    /// Skies adopted from a cache rather than from the wire. Each one is 7.0 MB that did not cross.
    pub cache_adopted: u64,
}

impl StarSky {
    /// Take a cache file from disk, prove it, and adopt the sky it holds.
    ///
    /// This is the saving the cache exists for: a returning player draws the galaxy from their own disk
    /// instead of pulling 7.0 MB across the wire again.
    ///
    /// Every refusal is COUNTED here rather than only returned, because the owner's condition is that a
    /// damaged cache produces a counted refusal and never a drawn frame. A returned error that nobody
    /// records is a refusal that cannot be reported.
    ///
    /// A refused cache leaves the sky exactly as it was. It does not clear a good sky already held, and
    /// it does not half-adopt: the client either holds the whole cached galaxy or holds what it held
    /// before. What follows a refusal is a fresh request, which is the receiver-states-its-generation
    /// exchange and is still owed (D-S11-SKY).
    ///
    /// # Errors
    /// [`CacheRefusal`] naming the first test the file failed.
    pub fn adopt_cache(
        &mut self,
        bytes: &[u8],
        expected: &vd_core::store_stamp::StoreStamp,
    ) -> Result<u64, CacheRefusal> {
        let (generation, rows) = self.count_cache_refusal(decode_cache(bytes, expected))?;
        // Adopted as a ONE-PART sky, which is what a cache is: it was whole when it was written, so it
        // arrives whole. `complete` answers immediately and the galaxy draws in one step.
        self.generation = Some(generation);
        self.parts_expected = 1;
        self.held.clear();
        self.held.insert(0, rows);
        self.cache_adopted += 1;
        Ok(generation)
    }

    /// Hoisted so [`Self::adopt_cache`] stays a straight-line expression, and so the counter cannot be
    /// forgotten on a path that returns early.
    fn count_cache_refusal(
        &mut self,
        got: Result<(u64, Vec<StarRow>), CacheRefusal>,
    ) -> Result<(u64, Vec<StarRow>), CacheRefusal> {
        if got.is_err() {
            self.cache_refused += 1;
        }
        got
    }

    /// Take one liveness beat and say what it proved.
    ///
    /// ★ WHY THIS EXISTS AT ALL. The catalogue is send-on-change and a galaxy does not change, so the
    /// server is correctly SILENT on that lane essentially forever. Silence is therefore the signature
    /// of the healthy case AND of a dead emitter, and nothing in the bytes tells them apart:
    ///
    /// ```text
    ///   the sky did not change   ──┐
    ///                              ├──► ...both look EXACTLY like this: nothing arrives.
    ///   the emitter is broken    ──┘
    /// ```
    ///
    /// The beat makes the healthy case SAY something, which splits the two:
    ///
    /// ```text
    ///   Current  ──►  "nothing changed"       (healthy, and proven)
    ///   Stale    ──►  "I hold the wrong sky"
    ///   (silence) ─►  "nobody is working"     — absence, so the CALLER's clock decides it
    /// ```
    ///
    /// The third reading is deliberately not answered here: absence is not an event, and this type
    /// holds no clock. A caller that holds one reads [`Self::beats_current`] and treats a counter that
    /// stopped climbing as the third arm. The watchdog POLICY — what the client shows when the sky lane
    /// goes quiet — is owed with the renderer (D-S11-SKY).
    ///
    /// Deliberately does NOT act on `Stale`. Knowing the held sky is wrong is a different job from
    /// replacing it, and replacing it is the receiver-states-its-generation exchange this beat is a
    /// companion to, not a substitute for.
    pub fn beat(&mut self, generation: u64) -> SkyBeat {
        // `complete` is what makes a generation HELD — a half-assembled sky of the named generation is
        // not a sky the client can draw, so it answers `Unheld` and not `Current`.
        if self.complete().is_none() {
            self.beats_unheld += 1;
            return SkyBeat::Unheld;
        }
        if self.generation == Some(generation) {
            self.beats_current += 1;
            SkyBeat::Current
        } else {
            self.beats_stale += 1;
            SkyBeat::Stale
        }
    }

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

    /// ★ THE THREE READINGS A BEAT GIVES (S11; the SL6-approved arm, owner 2026-08-27).
    ///
    /// The catalogue lane is send-on-change and a galaxy does not change, so a HEALTHY server is silent
    /// there essentially forever. Silence therefore means two different things at once and the client
    /// cannot tell which it is looking at. The beat splits them by making the healthy case speak.
    #[test]
    fn a_beat_separates_nothing_changed_from_i_hold_the_wrong_sky() {
        let mut sky = StarSky::default();

        // UNHELD — a beat arriving before any sky is whole. It still proves the lane is alive, which is
        // worth knowing while parts are in flight, but it confirms nothing about a sky not yet drawn.
        assert_eq!(sky.beat(7), SkyBeat::Unheld);

        // ...and a HALF-assembled sky of the very generation named is STILL Unheld. A sky is drawn only
        // when whole, so a beat must not promote half a galaxy to "current" — the holes in it look
        // exactly like stars that do not exist.
        assert_eq!(sky.accept(7, 0, 2, vec![star(1)]), SkyIngest::Applied);
        assert_eq!(
            sky.beat(7),
            SkyBeat::Unheld,
            "half a sky is not a held sky, however confidently the server names it"
        );

        // CURRENT — the sky completes, and now the same beat proves it. This is the overwhelmingly
        // common answer in a running game, and the one that silence could never give.
        assert_eq!(sky.accept(7, 1, 2, vec![star(2)]), SkyIngest::Completed);
        assert_eq!(sky.beat(7), SkyBeat::Current);

        // STALE — the server names a different sky. The stars on screen are not the stars it will fly
        // the player to.
        assert_eq!(sky.beat(8), SkyBeat::Stale);

        // The beat does NOT act on that: knowing the held sky is wrong is a different job from
        // replacing it, and replacing it is the receiver-states-its-generation exchange (D-S11-SKY).
        assert_eq!(sky.generation(), Some(7), "the beat reads, it never writes");
        assert_eq!(sky.complete().map(|r| r.len()), Some(2));

        assert_eq!(sky.beats_unheld, 2);
        assert_eq!(sky.beats_current, 1);
        assert_eq!(sky.beats_stale, 1);
        // The counters are the whole liveness story: a caller holding a clock reads a `beats_current`
        // that STOPPED CLIMBING as the third reading — "nobody is working" — which absence alone,
        // having no event to fire, can never announce here.
        assert_eq!(sky.refused, 0, "and a beat is never a refusal");
    }

    use vd_core::store_stamp::{StoreRole, StoreStamp};

    const SEED: u64 = 0x5EED;
    const CONSTANTS: [f64; 2] = [1.0, 2.0];

    fn stamp() -> StoreStamp {
        StoreStamp::new(
            StoreRole::ClientCatalogue,
            SEED,
            vd_core::ids::EpochId(1),
            &CONSTANTS,
        )
    }

    /// Fold the generation the way the SERVER does, so a cache written here is a cache the server
    /// would have accepted. A test that folded it differently would prove only its own arithmetic.
    fn fold(rows: &[StarRow]) -> u64 {
        vd_core::look::catalogue_generation(&postcard::to_allocvec(rows).expect("encodes"))
    }

    /// ★ A CLEAN CACHE ROUND-TRIPS, AND THE SKY COMES BACK EXACTLY (S11).
    ///
    /// This is the saving the whole cache exists for: a returning player draws the galaxy from disk
    /// instead of pulling 7.0 MB across the wire again.
    #[test]
    fn a_clean_cache_returns_the_same_sky_it_stored() {
        let rows = vec![star(1), star(2), star(3)];
        let generation = fold(&rows);
        let bytes = encode_cache(stamp(), generation, &rows);
        let (got_gen, got_rows) =
            decode_cache(&bytes, &stamp()).expect("a clean cache is accepted");
        assert_eq!(got_gen, generation);
        assert_eq!(got_rows, rows, "unaltered, and in order");
    }

    /// ★ ADOPTING A CACHE IS 7.0 MB THAT DOES NOT CROSS (S11), AND A REFUSED ONE CHANGES NOTHING.
    #[test]
    fn a_cache_is_adopted_whole_and_a_refused_one_leaves_the_held_sky_alone() {
        let rows = vec![star(1), star(2), star(3)];
        let generation = fold(&rows);
        let mut sky = StarSky::default();

        assert_eq!(
            sky.adopt_cache(&encode_cache(stamp(), generation, &rows), &stamp()),
            Ok(generation)
        );
        // Whole in ONE step — a cache was whole when it was written, so it arrives whole.
        assert_eq!(sky.complete(), Some(rows.clone()));
        assert_eq!(sky.generation(), Some(generation));
        assert_eq!(sky.cache_adopted, 1);
        assert_eq!(sky.cache_refused, 0);

        // ...and the beat now CONFIRMS a sky that never crossed the wire. This is the whole point:
        // the client holds the galaxy, and the server agrees, having sent none of it.
        assert_eq!(sky.beat(generation), SkyBeat::Current);

        // A DAMAGED CACHE IS COUNTED AND CHANGES NOTHING. It must not clear a good sky — a player who
        // is already flying should not lose the stars because a stale file on disk was bad.
        assert_eq!(
            sky.adopt_cache(&[0xff, 0xff, 0xff], &stamp()),
            Err(CacheRefusal::Undecodable)
        );
        assert_eq!(sky.cache_refused, 1, "the refusal is counted, never silent");
        assert_eq!(sky.cache_adopted, 1, "and not counted as an adoption");
        assert_eq!(
            sky.complete(),
            Some(rows),
            "the sky already held survives a bad file"
        );
        assert_eq!(sky.generation(), Some(generation));
    }

    /// ★ THE EDIT THAT ONLY THE FOLD CATCHES (S11; the owner's condition, Q4).
    ///
    /// postcard is positional, so a flipped byte does not break the decode — it produces a DIFFERENT
    /// well-formed sky. Every test above the fold is satisfied by it: the stamp is untouched, the count
    /// is unchanged, and the rows are perfectly representable. Only the generation notices, because the
    /// generation is folded from the content and the content moved.
    #[test]
    fn a_byte_flipped_cache_is_refused_by_the_fold_and_never_drawn() {
        let rows = vec![star(1), star(2), star(3)];
        let bytes = encode_cache(stamp(), fold(&rows), &rows);

        // Flip ONE bit somewhere in the row payload, and hunt for a flip that still decodes — the
        // point is a file that survives every other test, not one that falls over early.
        let mut caught = 0usize;
        let mut checked = 0usize;
        for i in (bytes.len() / 2)..bytes.len() {
            let mut edited = bytes.clone();
            edited[i] ^= 0x01;
            let got = decode_cache(&edited, &stamp());
            assert!(got.is_err(), "a byte-flipped cache was drawn at byte {i}");
            caught += usize::from(got == Err(CacheRefusal::DigestMismatch));
            checked += 1;
        }
        assert!(checked > 0, "the loop must actually run");
        // ★ MEASURED, not assumed: EVERY flip in the row payload reaches the fold. Not one is turned
        // away by an earlier test. That is the point of having the fold at all — the tests above it
        // catch a file that is obviously broken, and a deliberate edit is not obviously broken.
        assert_eq!(
            caught, checked,
            "every edit reached the fold; an earlier test catching one would mean the fixture, not the \
             code, was doing the work"
        );
    }

    /// ★ EVERY REFUSAL ARM, DRIVEN AND NAMED (S11).
    ///
    /// A refusal that cannot be told from another refusal is a report with nowhere to start, so each
    /// arm is reached on its own and by its own cause.
    #[test]
    fn each_cache_refusal_is_reached_by_its_own_cause() {
        let rows = vec![star(1), star(2)];
        let good = encode_cache(stamp(), fold(&rows), &rows);

        // TOO LARGE — refused unread, before any decoding work is done for it.
        assert_eq!(
            decode_cache(&vec![0u8; MAX_CACHE_BYTES + 1], &stamp()),
            Err(CacheRefusal::TooLarge)
        );

        // UNDECODABLE — bytes postcard cannot read at all.
        assert_eq!(
            decode_cache(&[0xff, 0xff, 0xff], &stamp()),
            Err(CacheRefusal::Undecodable)
        );

        // FOREIGN WORLD — a cache from another seed. The stars in it name bodies this build does not
        // have, so drawing it would put a player's destination where nothing is.
        let other = StoreStamp::new(
            StoreRole::ClientCatalogue,
            SEED + 1,
            vd_core::ids::EpochId(1),
            &CONSTANTS,
        );
        assert_eq!(
            decode_cache(&encode_cache(other, fold(&rows), &rows), &stamp()),
            Err(CacheRefusal::ForeignWorld)
        );
        // ...and the clean file is accepted against its own stamp, so the arm above is the stamp
        // disagreeing and not the fixture being broken.
        assert!(decode_cache(&good, &stamp()).is_ok());

        // UNREPRESENTABLE — a row that cannot be drawn. Written with a CORRECT fold, so it reaches the
        // representability test rather than being turned away by the digest first.
        let bad_rows = vec![StarRow {
            realm: vd_core::pose::RealmId::System(1),
            cell: vd_core::glam::I64Vec3::new(i64::MAX, 0, 0),
            class_code: 6,
            luma_lsun: 0.25,
        }];
        assert_eq!(
            decode_cache(&encode_cache(stamp(), fold(&bad_rows), &bad_rows), &stamp()),
            Err(CacheRefusal::Unrepresentable)
        );

        // UNDECODABLE, THE SECOND DOOR — a header that reads perfectly, with rows behind it that do
        // not. Driven separately because the header and the rows are decoded apart, so one arm
        // reaching this refusal says nothing about the other.
        let mut bad_rows_file = postcard::to_allocvec(&CacheHeader {
            stamp: stamp(),
            generation: 0,
            stars: 2,
        })
        .expect("encodes");
        bad_rows_file.extend_from_slice(&[0xff, 0xff, 0xff]);
        assert_eq!(
            decode_cache(&bad_rows_file, &stamp()),
            Err(CacheRefusal::Undecodable)
        );

        // DIGEST MISMATCH — every other test satisfied, and the stated generation simply wrong.
        assert_eq!(
            decode_cache(&encode_cache(stamp(), fold(&rows) ^ 1, &rows), &stamp()),
            Err(CacheRefusal::DigestMismatch)
        );
    }

    /// ★ TOO MANY STARS IS REFUSED WITHOUT WALKING THEM (S11).
    ///
    /// Separated from the arms above because building the file is the expensive part. The cap stands
    /// between a damaged length and an unbounded draw.
    #[test]
    fn more_cached_stars_than_the_cap_are_refused() {
        // A header CLAIMING one star past the cap, and no rows behind it at all. That is the point of
        // the split: the claim is disbelieved for the price of reading the header, so proving the cap
        // costs nothing. Materialising four million rows to prove it would make the gate pay, on every
        // run, exactly the cost this test exists to show the code never pays.
        let bytes = postcard::to_allocvec(&CacheHeader {
            stamp: stamp(),
            generation: 0,
            stars: MAX_STARS as u64 + 1,
        })
        .expect("encodes");
        assert_eq!(
            decode_cache(&bytes, &stamp()),
            Err(CacheRefusal::TooManyStars)
        );

        // And a header that LIES the other way — claiming fewer than it holds — is caught too, so the
        // cheap check above cannot be walked past by simply understating the count.
        let rows = vec![star(1), star(2)];
        let mut lying = postcard::to_allocvec(&CacheHeader {
            stamp: stamp(),
            generation: 0,
            stars: 1,
        })
        .expect("encodes");
        lying.extend_from_slice(&postcard::to_allocvec(&rows).expect("encodes"));
        assert_eq!(
            decode_cache(&lying, &stamp()),
            Err(CacheRefusal::CountMismatch)
        );
    }
}
