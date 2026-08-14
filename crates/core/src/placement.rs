//! The placement BOOK and its LEDGER — where a parent's authored answers to "where are my direct
//! children" live, each at ONE NAMED INSTANT (SL4, the placement arc).
//!
//! The law this module is the shape of: *physics produces A PLACEMENT; re-home/containment CONSUMES
//! placements and may never ask HOW a thing moves.* Before it, the placement lookup was a trait whose
//! implementor took a CLOCK and ran the Kepler ephemeris inside itself — a "does this child have
//! orbital elements?" test on the crossing path, exactly what SL4 forbids. A [`PlacementBook`] cannot
//! do that: its instant is a PROPERTY OF THE TABLE, never a parameter of a read, and its rows are
//! plain values. A lookup that cannot see time cannot integrate motion — SL4 held by the compiler.
//!
//! One writer authors books at named instants into one [`PlacementLedger`]; every consumer selects a
//! book by an instant it already holds AS DATA (a pose's stamp, a datagram's tick), then reads rows
//! with no clock. The ledger is a small ring of EXACT books per anchor — the writer already computed
//! them, so retaining is exact where projecting would approximate.
//!
//! **SL1 as a shape:** there is no cell for the anchor's own position. [`PlacementBook::of`] answers
//! the identity for the anchor — computed, never stored — so a realm's own placement is unstateable
//! here, not merely unstated.

use std::collections::BTreeMap;
use std::collections::VecDeque;

use crate::frame::FramePlacement;
use crate::ids::UniverseTick;
use crate::pose::{FrameRef, RealmId};

/// THE INJECTED MOTION SEAM: an opaque closure from seconds-since-epoch to a placement row — how a
/// child's way of moving reaches the placement WRITER without the simulation ever being able to name
/// it (the `sim::io` seam discipline, applied to motion). The boot composition builds these from the
/// motion crate's discriminant and injects them; the writer RUNS them to author book rows; nothing on
/// the crossing path can ask what is inside (there is no type to match on and no crate edge to reach
/// one — SL4 held structurally).
#[derive(Clone)]
pub struct MotionFn(pub std::sync::Arc<dyn Fn(f64) -> FramePlacement + Send + Sync>);

impl std::fmt::Debug for MotionFn {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        // Deliberately opaque: the closure's contents are exactly what nothing here may know.
        f.write_str("MotionFn")
    }
}

/// ONE anchor's placements for its DIRECT children, at ONE named instant.
///
/// The instant is a PROPERTY OF THE TABLE: a consumer that wants a different instant selects a
/// different book (from the [`PlacementLedger`]), it never hands a clock to a read. `of` has THREE
/// answers and no fourth: the anchor itself (the identity, computed), a direct child (the stored
/// row), anything else (`None` — nobody told this anchor where a stranger sits, so there is no
/// arithmetic it could do).
#[derive(Clone, Debug, PartialEq)]
pub struct PlacementBook {
    /// The authoring realm's own frame — always at the identity (it IS its own origin).
    anchor: FrameRef,
    /// The one instant every row in this book is true at.
    at: UniverseTick,
    /// The direct children's placements, sorted by frame for the binary search. Frames are unique
    /// (each realm has exactly one frame). NO row for `anchor`: [`PlacementBook::new`] removes one —
    /// the anchor's own placement is the identity by definition and is never stored (SL1: a field's
    /// PRESENCE is the leak).
    rows: Vec<(FrameRef, FramePlacement)>,
}

impl PlacementBook {
    /// Build a book from an anchor, its instant, and the authored child rows (any order; sorted
    /// here). A row naming the anchor itself is REMOVED — see the field doc.
    #[must_use]
    pub fn new(
        anchor: FrameRef,
        at: UniverseTick,
        mut rows: Vec<(FrameRef, FramePlacement)>,
    ) -> PlacementBook {
        rows.retain(|(f, _)| *f != anchor);
        rows.sort_by_key(|(f, _)| *f);
        PlacementBook { anchor, at, rows }
    }

    /// The one instant this book speaks at.
    #[must_use]
    pub fn at(&self) -> UniverseTick {
        self.at
    }

    /// The authoring realm's own frame.
    #[must_use]
    pub fn anchor(&self) -> FrameRef {
        self.anchor
    }

    /// THREE ANSWERS, NO FOURTH — AND NO CLOCK ARGUMENT.
    ///
    /// The signature IS the law (SL4): a lookup that cannot see time cannot integrate motion. Pinned
    /// by a compile-fail doc-test — handing a read a clock stops compiling, today and every day:
    ///
    /// ```compile_fail,E0061
    /// use vd_core::ids::UniverseTick;
    /// use vd_core::placement::PlacementBook;
    /// use vd_core::pose::FrameRef;
    /// let book = PlacementBook::new(FrameRef::GalaxySpace, UniverseTick(0), Vec::new());
    /// // E0061: `of` takes ONE argument — there is no tick parameter to hand a solver.
    /// let _ = book.of(FrameRef::GalaxySpace, UniverseTick(0));
    /// ```
    #[must_use]
    pub fn of(&self, frame: FrameRef) -> Option<FramePlacement> {
        if self.anchor == frame {
            return Some(FramePlacement::identity());
        }
        self.rows
            .binary_search_by_key(&frame, |(f, _)| *f)
            .ok()
            .map(|i| self.rows[i].1)
    }

    /// The stored child rows, in frame order. The anchor never appears (its identity is computed).
    pub fn rows(&self) -> impl Iterator<Item = (FrameRef, FramePlacement)> + '_ {
        self.rows.iter().copied()
    }
}

/// Why a book selection missed: the wanted instant is outside the window the ledger retains for that
/// anchor (or the anchor has no books at all — `head: None`). A typed refusal, never a silently
/// substituted nearby instant: the consumer that holds the stamp decides what a miss means (drop the
/// batch and count it; refuse the arrival loudly).
#[derive(Clone, Copy, Debug, PartialEq, Eq, thiserror::Error)]
#[error("no placement book for {anchor:?} at tick {}: head {head:?}, window {span} back", wanted.0)]
pub struct PlacementMiss {
    /// The anchor whose book was wanted.
    pub anchor: RealmId,
    /// The instant the consumer asked for (a stamp it holds as data).
    pub wanted: UniverseTick,
    /// The newest retained instant, or `None` if the anchor has no books yet.
    pub head: Option<UniverseTick>,
    /// The backward window (ticks behind the head) the ledger retains.
    pub span: u32,
}

/// Every anchor this shard hosts × a bounded BACKWARD window of exact per-instant books.
///
/// One writer publishes; every consumer reads. `head` is infallible-by-construction once the writer
/// has run (an `Option` only for the pre-first-author boot instants); `at` is the fallible selection
/// for the lanes whose instant is message-carried and bounded by relay latency.
#[derive(Clone, Debug, Default)]
pub struct PlacementLedger {
    books: BTreeMap<RealmId, VecDeque<PlacementBook>>,
    span_back: u32,
}

impl PlacementLedger {
    /// A ledger retaining, per anchor, every book within `span_back` ticks behind the newest.
    #[must_use]
    pub fn new(span_back: u32) -> PlacementLedger {
        PlacementLedger {
            books: BTreeMap::new(),
            span_back,
        }
    }

    /// The backward window (ticks behind the head) this ledger retains.
    #[must_use]
    pub fn span_back(&self) -> u32 {
        self.span_back
    }

    /// Publish one authored book. Instants must arrive in non-decreasing order (the writer runs on
    /// the monotonic universe clock): a NEWER instant appends and trims the window's tail; the SAME
    /// instant replaces (a re-authored tick is idempotent); an OLDER instant is REFUSED — returned
    /// `false` so the caller can count it — because the window's whole meaning is exact books in tick
    /// order, and reordering history would let two consumers read two pasts.
    pub fn publish(&mut self, anchor: RealmId, book: PlacementBook) -> bool {
        let deque = self.books.entry(anchor).or_default();
        match deque.back().map(|b| b.at().cmp(&book.at())) {
            None | Some(std::cmp::Ordering::Less) => deque.push_back(book),
            Some(std::cmp::Ordering::Equal) => {
                *deque.back_mut().expect("compared against the back row") = book;
            }
            Some(std::cmp::Ordering::Greater) => return false,
        }
        let head_at = deque.back().expect("a book was just stored").at().0;
        while deque
            .front()
            .is_some_and(|b| b.at().0 + u64::from(self.span_back) < head_at)
        {
            deque.pop_front();
        }
        true
    }

    /// The NEWEST book for `anchor` — the "now" selection (no clock: the head IS now, because the
    /// writer runs at the head of every tick). `None` only before the writer's first pass.
    #[must_use]
    pub fn head(&self, anchor: RealmId) -> Option<&PlacementBook> {
        self.books.get(&anchor).and_then(|d| d.back())
    }

    /// The book for `anchor` at EXACTLY `stamp` — the selection for a lane whose instant is
    /// message-carried (an arriving pose's stamp, a relayed datagram's tick).
    ///
    /// # Errors
    /// [`PlacementMiss`] when `stamp` is outside the retained window (or the anchor has no books).
    pub fn at(
        &self,
        anchor: RealmId,
        stamp: UniverseTick,
    ) -> Result<&PlacementBook, PlacementMiss> {
        let deque = self.books.get(&anchor);
        let head = deque.and_then(|d| d.back()).map(PlacementBook::at);
        deque
            .and_then(|d| {
                d.binary_search_by_key(&stamp, PlacementBook::at)
                    .ok()
                    .and_then(|i| d.get(i))
            })
            .ok_or(PlacementMiss {
                anchor,
                wanted: stamp,
                head,
                span: self.span_back,
            })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use glam::{DQuat, DVec3, I64Vec3};

    fn sys() -> FrameRef {
        FrameRef::SystemSpace { system_seed: 1 }
    }
    fn planet(seed: u64) -> FrameRef {
        FrameRef::PlanetCentered { planet_seed: seed }
    }
    fn at_x(x: f64) -> FramePlacement {
        FramePlacement::moving(DVec3::new(x, 0.0, 0.0), DVec3::ZERO)
    }

    #[test]
    fn a_book_answers_the_anchor_a_child_and_nothing_else() {
        let book = PlacementBook::new(
            sys(),
            UniverseTick(7),
            vec![(planet(3), at_x(3.0)), (planet(2), at_x(2.0))],
        );
        assert_eq!(book.anchor(), sys());
        assert_eq!(book.at(), UniverseTick(7));
        // The anchor: the identity, computed.
        assert_eq!(book.of(sys()), Some(FramePlacement::identity()));
        // A direct child: the stored row (sorted, so both orders of construction resolve).
        assert_eq!(book.of(planet(2)), Some(at_x(2.0)));
        assert_eq!(book.of(planet(3)), Some(at_x(3.0)));
        // A stranger: None — nobody told this anchor where it sits.
        assert_eq!(book.of(planet(99)), None);
    }

    #[test]
    fn a_row_naming_the_anchor_is_removed_not_stored() {
        // SL1: a field's PRESENCE is the leak — the anchor's own placement is unstateable in a book.
        let book = PlacementBook::new(
            sys(),
            UniverseTick(0),
            vec![(sys(), at_x(999.0)), (planet(1), at_x(1.0))],
        );
        assert_eq!(book.of(sys()), Some(FramePlacement::identity()));
        assert_eq!(book.rows().count(), 1);
        assert_eq!(book.rows().next(), Some((planet(1), at_x(1.0))));
    }

    #[test]
    fn rows_iterates_in_frame_order() {
        let book = PlacementBook::new(
            sys(),
            UniverseTick(0),
            vec![(planet(9), at_x(9.0)), (planet(1), at_x(1.0))],
        );
        let rows: Vec<_> = book.rows().collect();
        assert_eq!(rows, vec![(planet(1), at_x(1.0)), (planet(9), at_x(9.0))]);
    }

    #[test]
    fn a_stored_row_keeps_its_full_placement() {
        // The row is a whole rigid placement (cell anchor, velocity, orientation, spin) — the book
        // stores it verbatim, no field is normalized away.
        let full = FramePlacement {
            origin_cell: I64Vec3::new(5, 0, -2),
            origin: DVec3::new(1.0, 2.0, 3.0),
            velocity: DVec3::new(0.1, 0.0, -0.4),
            orientation: DQuat::from_rotation_z(0.3),
            angular_velocity: DVec3::new(0.0, 0.0, 0.2),
        };
        let book = PlacementBook::new(sys(), UniverseTick(3), vec![(planet(1), full)]);
        assert_eq!(book.of(planet(1)), Some(full));
    }

    fn anchor_realm() -> RealmId {
        RealmId::System(1)
    }

    fn book_at(t: u64) -> PlacementBook {
        PlacementBook::new(sys(), UniverseTick(t), vec![(planet(1), at_x(t as f64))])
    }

    #[test]
    fn the_ledger_head_is_the_newest_book_and_none_before_the_first() {
        let mut ledger = PlacementLedger::new(4);
        assert_eq!(ledger.head(anchor_realm()), None);
        assert!(ledger.publish(anchor_realm(), book_at(10)));
        assert!(ledger.publish(anchor_realm(), book_at(11)));
        assert_eq!(ledger.head(anchor_realm()), Some(&book_at(11)));
        assert_eq!(ledger.span_back(), 4);
    }

    #[test]
    fn the_ledger_serves_exact_instants_inside_the_window_and_misses_outside() {
        let mut ledger = PlacementLedger::new(2);
        for t in 10..=14 {
            assert!(ledger.publish(anchor_realm(), book_at(t)));
        }
        // Window: head 14, span 2 ⇒ 12..=14 retained.
        assert_eq!(
            ledger.at(anchor_realm(), UniverseTick(12)),
            Ok(&book_at(12))
        );
        assert_eq!(
            ledger.at(anchor_realm(), UniverseTick(14)),
            Ok(&book_at(14))
        );
        // Behind the window: a typed miss carrying the head + span (never a nearby substitute).
        assert_eq!(
            ledger.at(anchor_realm(), UniverseTick(11)),
            Err(PlacementMiss {
                anchor: anchor_realm(),
                wanted: UniverseTick(11),
                head: Some(UniverseTick(14)),
                span: 2,
            })
        );
        // Ahead of the head: the same typed miss. Forward asks DO happen (a sender's ClockSync
        // phase can lead the receiver's) — the LANES answer them by clamping to the head
        // (`book_at_or_head`/`arrival_book` in vd-sim, counted + measured per direction); the
        // LEDGER itself stays exact-instant, and forward RETENTION is the ledgered non-feature
        // (D-PLACE-6).
        assert!(ledger.at(anchor_realm(), UniverseTick(15)).is_err());
        // An anchor with no books at all: head is None in the miss.
        assert_eq!(
            ledger.at(RealmId::System(99), UniverseTick(14)),
            Err(PlacementMiss {
                anchor: RealmId::System(99),
                wanted: UniverseTick(14),
                head: None,
                span: 2,
            })
        );
    }

    #[test]
    fn publishing_the_same_instant_replaces_and_an_older_one_is_refused() {
        let mut ledger = PlacementLedger::new(4);
        assert!(ledger.publish(anchor_realm(), book_at(10)));
        // Same instant: replaced, idempotently (a re-authored tick).
        let replacement =
            PlacementBook::new(sys(), UniverseTick(10), vec![(planet(1), at_x(42.0))]);
        assert!(ledger.publish(anchor_realm(), replacement.clone()));
        assert_eq!(ledger.head(anchor_realm()), Some(&replacement));
        // Older instant: refused — history is never reordered.
        assert!(!ledger.publish(anchor_realm(), book_at(9)));
        assert_eq!(ledger.head(anchor_realm()), Some(&replacement));
    }

    #[test]
    fn a_motion_fn_is_debug_opaque_and_runnable() {
        // The seam's two properties: it RUNS (a pure fn of seconds), and it says NOTHING about what
        // it is (the opaque Debug — the contents are exactly what a consumer may not know).
        let f = MotionFn(std::sync::Arc::new(|secs| {
            FramePlacement::moving(DVec3::new(secs, 0.0, 0.0), DVec3::ZERO)
        }));
        assert_eq!((f.0)(3.0).origin, DVec3::new(3.0, 0.0, 0.0));
        assert_eq!(format!("{f:?}"), "MotionFn");
        assert_eq!(format!("{:?}", f.clone()), "MotionFn");
    }

    #[test]
    fn the_miss_names_itself() {
        let miss = PlacementMiss {
            anchor: anchor_realm(),
            wanted: UniverseTick(5),
            head: Some(UniverseTick(9)),
            span: 3,
        };
        assert_eq!(
            miss.to_string(),
            "no placement book for System(1) at tick 5: head Some(UniverseTick(9)), window 3 back"
        );
    }
}
