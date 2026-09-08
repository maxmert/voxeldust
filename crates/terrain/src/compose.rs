//! ★ THE COMPOSITION ORDER (ruling V10 S6-6): what the seed did not decide, laid over the generated
//! shape in ONE fixed order before the extractor runs:
//!
//! ```text
//!   1 the generated shape       the recipe's cells, the halo included
//!   2 terrain cell edits        a mined cell → `Empty` (ruling V4; never `Air`, the atmosphere), gap +1;
//!                               placed dirt → dirt, gap −1
//!   3 catalogue blocks          the cell KEEPS its gap byte (Format B); the block changes the lane
//!   4 sub-metre blocks          the same, plus the seating rule (`seat`)
//!   5 attachments               no volume; they never touch the field
//! ```
//!
//! Rows are applied by RANK (2 before 3 before 4 before 5), and inside a rank in the order given;
//! the last row on a cell wins. Only rank 2 changes a byte the extractor reads: rows 3 to 5 keep
//! the gap byte, by the owner's ruling that a foundation cut into a slope shows the slope on
//! removal. The rows are an in-memory list here; the store that holds them is slice 9's and the lane
//! that carries them slice 10's.
//!
//! A row addresses a cell of the BOX, halo included, in local coordinates `−1..=62`: a host that
//! holds the neighbour chunk's edits lays them onto the halo, so a hole mined on a chunk edge is
//! smooth across it.
//!
//! **Example.** A player mines the cell under a rock ledge, then sets a steel foundation on the
//! cell beside it. Row 2 makes the mined cell air at the top code; row 3 changes nothing in the
//! field. The extractor sags the ledge's underside into the hole and runs the slope into the
//! foundation, where the opaque cube hides it.

use crate::chunk::Cell;
use crate::lattice::SampleBox;
use crate::strata::Stratum;

/// One row of what the seed did not decide, at a local box cell.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum EditRow {
    /// A terrain cell edit: the substance code and the gap byte the cell now holds.
    TerrainCell { cell: [i8; 3], code: u8, gap: i8 },
    /// A catalogue block on the cell: the field is untouched.
    Block { cell: [i8; 3] },
    /// A sub-metre block on the cell: the field is untouched.
    SubMetre { cell: [i8; 3] },
    /// An attachment: no volume, no cell.
    Attachment,
}

impl EditRow {
    /// The composition rank: the order of the five rows, the generated shape being rank 1.
    #[must_use]
    pub const fn rank(self) -> u8 {
        match self {
            EditRow::TerrainCell { .. } => 2,
            EditRow::Block { .. } => 3,
            EditRow::SubMetre { .. } => 4,
            EditRow::Attachment => 5,
        }
    }
}

/// Lay the rows over the box in composition order. A row naming a code no stratum owns, or a cell
/// outside the box, is counted in the returned refusal count and changes nothing (a decoder refuses,
/// never defaults).
pub fn compose(samples: &mut SampleBox, rows: &[EditRow]) -> u32 {
    let mut ordered: Vec<(u8, usize, EditRow)> = rows
        .iter()
        .enumerate()
        .map(|(n, r)| (r.rank(), n, *r))
        .collect();
    ordered.sort_by(|x, y| (x.0, x.1).cmp(&(y.0, y.1)));
    let mut refused = 0;
    for (_, _, row) in ordered {
        if let EditRow::TerrainCell { cell, code, gap } = row {
            let (a, b, c) = (i32::from(cell[0]), i32::from(cell[1]), i32::from(cell[2]));
            let span = -1..=62;
            let inside = span.contains(&a) & span.contains(&b) & span.contains(&c);
            match (inside, Stratum::from_code(code)) {
                (true, Some(stratum)) => {
                    let i = SampleBox::index(a, b, c);
                    samples.cells[i] = Cell { stratum, gap };
                }
                _ => refused += 1,
            }
        }
    }
    refused
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::extract::tests::synthetic;

    #[test]
    fn rows_apply_by_rank_then_by_order_and_only_terrain_rows_touch_the_field() {
        let mut sb = synthetic(|_, _, c| if c < 10 { -128 } else { 127 });
        let before = sb.cell(5, 5, 9);
        let rows = [
            EditRow::Attachment,
            EditRow::Block { cell: [5, 5, 9] },
            // Two edits on one cell: the later one wins inside rank 2.
            EditRow::TerrainCell {
                cell: [5, 5, 9],
                code: Stratum::Air.code(),
                gap: 127,
            },
            EditRow::TerrainCell {
                cell: [5, 5, 9],
                code: Stratum::Dirt.code(),
                gap: -3,
            },
            EditRow::SubMetre { cell: [6, 5, 9] },
            // A halo cell is a lawful target.
            EditRow::TerrainCell {
                cell: [-1, 0, 10],
                code: Stratum::Air.code(),
                gap: 100,
            },
        ];
        assert_eq!(compose(&mut sb, &rows), 0);
        assert_eq!(
            sb.cell(5, 5, 9),
            Cell {
                stratum: Stratum::Dirt,
                gap: -3
            }
        );
        assert_ne!(sb.cell(5, 5, 9), before);
        assert_eq!(sb.cell(-1, 0, 10).gap, 100);
        assert_eq!(
            sb.cell(6, 5, 9).gap,
            -128,
            "a sub-metre row keeps the field"
        );
        assert_eq!(EditRow::Attachment.rank(), 5);
        assert_eq!(EditRow::Block { cell: [0, 0, 0] }.rank(), 3);
        assert_eq!(EditRow::SubMetre { cell: [0, 0, 0] }.rank(), 4);
        // A block row after a terrain row in the list still comes AFTER it by rank, and does not
        // undo it.
        let mut sb2 = synthetic(|_, _, _| -128);
        let rows2 = [
            EditRow::Block { cell: [1, 1, 1] },
            EditRow::TerrainCell {
                cell: [1, 1, 1],
                code: Stratum::Air.code(),
                gap: 64,
            },
        ];
        assert_eq!(compose(&mut sb2, &rows2), 0);
        assert_eq!(sb2.cell(1, 1, 1).gap, 64);
    }

    /// A cell mined on a chunk's edge is one edit in the world: the owner lays it on its core, the
    /// neighbour lays it on its halo through `local_of_site`, and the two meshes still meet — every
    /// shared vertex the same bytes.
    #[test]
    fn a_neighbours_edit_reaches_the_halo_and_the_meshes_still_meet() {
        use crate::digest::surface_chunk_z;
        use crate::extract::extract;
        use crate::home::home_planet;
        use crate::lattice::{local_of_site, sample_box, site_of};
        use crate::position::vertex_position_m;
        use vd_seed::bend::Face;
        let m = home_planet();
        let z = surface_chunk_z(&m, Face::NegZ, 0, 40, 41);
        let key = |x: i32| crate::chunk::ChunkKey {
            face: Face::NegZ,
            rung: 0,
            x,
            y: 41,
            z,
        };
        let (ka, kb) = (key(40), key(41));
        let mut a = sample_box(&m, ka).expect("in the band");
        let mut b = sample_box(&m, kb).expect("in the band");
        // The edit: the last column of chunk A at a surface layer, mined to Empty. In world terms
        // it is a site; A holds it at local a = 61, B at local a = −1.
        let c = {
            // The lowest air cell of the column, by a fixed-count scan; the cell under it is rock.
            let mut lowest_air = 61;
            let mut c = 60;
            while c >= 0 {
                if a.cell(61, 30, c).gap >= 0 {
                    lowest_air = c;
                }
                c -= 1;
            }
            lowest_air.max(1) - 1
        };
        let site = site_of(&m, ka, 61, 30);
        let (la, lb) = local_of_site(&m, ka, site).expect("A holds it");
        let (na, nb) = local_of_site(&m, kb, site).expect("B holds it in its halo");
        assert_eq!((la, lb), (61, 30));
        assert_eq!((na, nb), (-1, 30));
        let row = |aa: i32, bb: i32| EditRow::TerrainCell {
            cell: [aa as i8, bb as i8, c as i8],
            code: Stratum::Empty.code(),
            gap: i8::MAX,
        };
        assert_eq!(compose(&mut a, &[row(la, lb)]), 0);
        assert_eq!(compose(&mut b, &[row(na, nb)]), 0);
        assert_eq!(a.cell(61, 30, c), b.cell(-1, 30, c), "one edit, both boxes");
        // Every group across the shared face, the mined cell's included, gives the same vertex
        // in metres from both boxes; and the mined cell moved a vertex.
        use crate::extract::vertex_of_group;
        let plain = sample_box(&m, ka).expect("in the band");
        let mut compared = 0;
        let mut moved = 0;
        let mut cc = -1;
        while cc < 62 {
            let mut bb = -1;
            while bb < 62 {
                let va = vertex_of_group(&a, [61, bb, cc]);
                let vb = vertex_of_group(&b, [-1, bb, cc]);
                assert_eq!(va.is_some(), vb.is_some());
                if let (Some(va), Some(vb)) = (va, vb) {
                    assert_eq!(
                        vertex_position_m(&m, &a, va),
                        vertex_position_m(&m, &b, vb),
                        "group (61, {bb}, {cc})"
                    );
                    compared += 1;
                    if vertex_of_group(&plain, [61, bb, cc]) != Some(va) {
                        moved += 1;
                    }
                }
                bb += 1;
            }
            cc += 1;
        }
        assert!(compared > 0);
        assert!(moved > 0, "the mined cell reshaped the surface");
        let _ = (extract(&a), extract(&b));
    }

    #[test]
    fn a_row_outside_the_box_or_with_an_unknown_code_is_refused_and_changes_nothing() {
        let mut sb = synthetic(|_, _, _| -128);
        let copy = sb.clone();
        let rows = [
            EditRow::TerrainCell {
                cell: [63, 0, 0],
                code: Stratum::Air.code(),
                gap: 1,
            },
            EditRow::TerrainCell {
                cell: [0, -2, 0],
                code: Stratum::Air.code(),
                gap: 1,
            },
            EditRow::TerrainCell {
                cell: [0, 0, 0],
                code: 250,
                gap: 1,
            },
        ];
        assert_eq!(compose(&mut sb, &rows), 3);
        assert_eq!(sb, copy);
    }
}
