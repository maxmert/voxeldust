//! ★ THE STAND-IN (2026-09-30; the owner: "Cause and cure now" — the login's stream-in pops,
//! `docs/design/DEFERRED.md`, the temporal path's T2 block).
//!
//! Every rung's ring covers only its own band of distance ([`crate::ladder_view`]: "every ring is
//! the same annulus in its own chunks"), so a coarser rung is drawn under a finer ring's ground only
//! inside the crossfade band. As the eye moves, the release hold keeps the old chunks until the new
//! ones arrive, so no ground is ever left to nobody. At a LOGIN nothing is drawn yet: until a finer
//! chunk lands, its ground is a hole, and the landing fills the hole in one frame — its arrival ramp
//! rises from its next rung's surface, which is not drawn there. MEASURED on the pilot's login (the
//! moving-eye gate's landing, the renderer's landing census): two landings in three, at every rung
//! from 0 to 8, rose with no next-rung chunk drawn under them, and a rung-2 dune 3.3 km off appeared
//! in one frame where the land 6.6 km behind it had shown through.
//!
//! **The rule.** While a ladder lands from nothing (until its wanted set has once been whole), a
//! wanted chunk that has not arrived is asked for only over DRAWN GROUND: its next-rung column
//! drawn and settled (the surface its ramp rises from), or a finer column drawn inside it; the top
//! rung needs none. Until then the next-rung column is asked for in its place, as a STAND-IN, on the
//! same rule — up the ladder to the first column that stands over drawn ground itself. A landed
//! ladder asks for every chunk at once, as before: the release hold keeps its ground as the eye
//! moves (MEASURED with the rule always on: a walk's revealed peaks waited on their stand-ins, and
//! the walk's band broke on 1 365 frames).
//!
//! **★ THE STAND-IN STANDS AT ITS OWN HEIGHT** (2026-09-30; the owner: "Own height" — the ring-sink
//! law, "a rung is drawn at its own radius wherever nothing covers it"). MEASURED with the band's
//! sink: a stand-in stood sunk by its rung's sink (121 m at rung 6, 583 m at rung 8), so the near
//! ground first stood low and rose in steps as the finer rungs landed — the near ridge's crest four
//! pixels in one frame. So a chunk drawn as a stand-in carries its tag's stand-in flag
//! ([`crate::arrival::STAND_IN_FLAG`]) and stands unsunk; a finer chunk that lands on it starts on its
//! surface ([`crate::arrival::OWN_HEIGHT_FLAG`]) and grows into its own shape over the ramp; when the
//! stand-in's own rise is done and every wanted chunk over its footprint has arrived, it is COVERED
//! ([`crate::arrival::covered_stamp`]) and sinks under that ground over the ramp, and only then does the
//! release hold let it go.
//!
//! **Example.** The pilot logs in on the dunes. The rung-3 chunk over the valley 3.3 km off lands
//! first and stands at its own height; the rung-2 dune is asked for once that chunk has settled, and
//! grows out of its surface over half a second; when every rung-2 chunk over it has landed, the
//! rung-3 chunk sinks away under them over half a second and is released.

use vd_terrain::BodyDefinition;
use vd_terrain::chunk::ChunkKey;

use crate::artifact_book::ArtifactCache;
use crate::ladder_view::{Column, column_span};

/// ★ WHETHER A COLUMN STANDS OVER DRAWN GROUND on a ladder of `rungs`, `drawn` naming the columns
/// that hold a drawn chunk: the top rung always; else its next-rung column drawn, or one of the four
/// finer columns inside it drawn.
#[must_use]
pub fn grounds(drawn: &dyn Fn(Column) -> bool, col: Column, rungs: u8) -> bool {
    let top = u16::from(col.rung) + 1 >= u16::from(rungs);
    let over = drawn(next_column(col));
    let under = finer_columns(col).iter().any(|c| drawn(*c));
    top | over | under
}

/// The next-rung column over `col`: the column one rung coarser that holds it.
#[must_use]
pub fn next_column(col: Column) -> Column {
    let (x, y) = col.coarser(1);
    Column {
        face: col.face,
        rung: col.rung.saturating_add(1),
        x,
        y,
    }
}

/// The four columns one rung finer inside `col`; none at rung 0.
#[must_use]
pub fn finer_columns(col: Column) -> Vec<Column> {
    col.rung.checked_sub(1).map_or_else(Vec::new, |rung| {
        [(0, 0), (1, 0), (0, 1), (1, 1)]
            .iter()
            .map(|(dx, dy)| Column {
                face: col.face,
                rung,
                x: col.x * 2 + dx,
                y: col.y * 2 + dy,
            })
            .collect()
    })
}

/// ★ THE ASK FOR A MISSING CHUNK: `None` asks for it now (it stands over drawn ground); `Some(col)`
/// names the stand-in column to ask for first — up the ladder from its next rung, the first column
/// that stands over drawn ground itself (the top rung always does).
#[must_use]
pub fn stand_in_for(drawn: &dyn Fn(Column) -> bool, key: ChunkKey, rungs: u8) -> Option<Column> {
    let mut col = Column::of(key);
    if grounds(drawn, col, rungs) {
        return None;
    }
    loop {
        col = next_column(col);
        if grounds(drawn, col, rungs) {
            return Some(col);
        }
    }
}

/// THE KEYS OF A COLUMN on what the client holds: every slice of its span up to the band's top (the
/// descent's own rule, [`crate::ladder_view::LadderView`]), and whether the span is provisional (read
/// on a stand-in field: read it again once the artifact is whole).
#[must_use]
pub fn column_keys(
    body: &BodyDefinition,
    artifact: Option<&ArtifactCache>,
    col: Column,
) -> (Vec<ChunkKey>, bool) {
    let (span, provisional) = column_span(body, artifact, col);
    let top_z = vd_terrain::digest::top_chunk_z(body, col.rung);
    let keys = (span.lo..=span.hi.min(top_z))
        .map(|z| ChunkKey {
            face: col.face,
            rung: col.rung,
            x: col.x,
            y: col.y,
            z,
        })
        .collect();
    (keys, provisional)
}

/// ★ A STAND-IN'S PRIORITY (the lower builds first; the layout of
/// [`crate::ladder_view::WantedSet::priority_of`]): the urgent class, by depth — so a stand-in builds
/// before the finer ground it stands in for, and a coarser one before a finer one.
#[must_use]
pub fn stand_in_priority(key: ChunkKey) -> u32 {
    63u32.saturating_sub(u32::from(key.rung)) << 24
}

#[cfg(test)]
mod tests {
    use super::*;
    use vd_seed::bend::Face;

    const RUNGS: u8 = 6;

    fn col(rung: u8, x: i32, y: i32) -> Column {
        Column {
            face: Face::PosX,
            rung,
            x,
            y,
        }
    }

    fn key(rung: u8, x: i32, y: i32) -> ChunkKey {
        ChunkKey {
            face: Face::PosX,
            rung,
            x,
            y,
            z: 3,
        }
    }

    fn drawn(cols: &[Column]) -> impl Fn(Column) -> bool + '_ {
        move |c| cols.contains(&c)
    }

    /// The next rung's column holds the finer one; the four finer columns tile it; rung 0 has none.
    #[test]
    fn the_next_column_holds_the_four_finer_ones() {
        assert_eq!(next_column(col(2, 5, 6)), col(3, 2, 3));
        let finer = finer_columns(col(3, 2, 3));
        assert_eq!(
            finer,
            vec![col(2, 4, 6), col(2, 5, 6), col(2, 4, 7), col(2, 5, 7)]
        );
        assert!(finer.iter().all(|c| next_column(*c) == col(3, 2, 3)));
        assert!(finer_columns(col(0, 1, 1)).is_empty());
    }

    /// ★ Over drawn ground: the top rung always; the next rung drawn; a finer column drawn inside.
    /// Nothing drawn: not grounded. A column drawn itself, or a sibling drawn, does not ground it.
    #[test]
    fn a_column_stands_over_its_next_rung_or_its_finer_ground() {
        let none = drawn(&[]);
        assert!(grounds(&none, col(RUNGS - 1, 0, 0), RUNGS));
        assert!(!grounds(&none, col(2, 5, 6), RUNGS));
        assert!(grounds(&drawn(&[col(3, 2, 3)]), col(2, 5, 6), RUNGS));
        assert!(grounds(&drawn(&[col(1, 11, 13)]), col(2, 5, 6), RUNGS));
        assert!(!grounds(&drawn(&[col(2, 5, 6)]), col(2, 5, 6), RUNGS));
        assert!(!grounds(&drawn(&[col(2, 4, 6)]), col(2, 5, 6), RUNGS));
        assert!(!grounds(&drawn(&[col(4, 1, 1)]), col(2, 5, 6), RUNGS));
    }

    /// ★ THE ASK: a chunk over its drawn next rung is asked for now; with nothing drawn the stand-in
    /// is the top rung's column; with rung 4 drawn over it, the rung-3 column (which stands over it).
    #[test]
    fn a_missing_chunk_asks_first_for_the_nearest_grounded_column_up_the_ladder() {
        let k = key(2, 5, 6);
        assert_eq!(stand_in_for(&drawn(&[col(3, 2, 3)]), k, RUNGS), None);
        assert_eq!(
            stand_in_for(&drawn(&[]), k, RUNGS),
            Some(col(RUNGS - 1, 0, 0))
        );
        assert_eq!(
            stand_in_for(&drawn(&[col(4, 1, 1)]), k, RUNGS),
            Some(col(3, 2, 3))
        );
        // A top-rung chunk needs no stand-in.
        assert_eq!(stand_in_for(&drawn(&[]), key(RUNGS - 1, 0, 0), RUNGS), None);
    }

    /// The column's keys are its span's slices up to the band's top, on the recipe's own relief while
    /// no artifact is held (provisional for a body that states one).
    #[test]
    fn a_column_asks_for_every_slice_of_its_span() {
        let body = vd_terrain::home::home_planet();
        let c = Column {
            face: Face::PosZ,
            rung: 6,
            x: 3,
            y: 4,
        };
        let (keys, provisional) = column_keys(&body, None, c);
        assert!(provisional);
        assert!(!keys.is_empty());
        let top_z = vd_terrain::digest::top_chunk_z(&body, 6);
        assert!(keys.iter().all(|k| Column::of(*k) == c));
        assert!(keys.iter().all(|k| k.z <= top_z));
        assert!(keys.windows(2).all(|w| w[1].z == w[0].z + 1));
    }

    /// A coarser stand-in builds first.
    #[test]
    fn a_coarser_stand_in_builds_first() {
        assert!(stand_in_priority(key(3, 0, 0)) < stand_in_priority(key(2, 0, 0)));
        assert_eq!(stand_in_priority(key(0, 0, 0)), 63 << 24);
    }
}
