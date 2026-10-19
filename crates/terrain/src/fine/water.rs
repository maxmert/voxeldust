//! ★ THE WATER MAP OF A FINE RASTER (the one-volume revision §3.4; rulings B105 Q1 (b) and Q2, B114; slice W2): which
//! cells of a level's raster hold water, and at which level. THE IDENTITY comes from the 8 km world (each node's standing
//! water: the sea at its level, a lake at its word), THE WET TEST from the raster's own ground, and CONNECTION is computed,
//! never assumed: each water spreads from its own nodes' cells whose ground stands under its level, through the cells
//! whose ground stands under it (four neighbours), so a hollow under a lake's level that no path joins to the lake stays
//! dry — no hanging pond. Where two waters reach one cell the higher level holds it (a contact; the gate G-DIVIDE counts
//! them). A carved river ([`super::carve`]) holds its channel's cells at its bank. Every level in quarter metres over the
//! sea, [`DRY`] where no water stands; integers only, so every host reads the same map.
//!
//! **Example.** On the coast stand's 128 m raster the bay's cells stand under the sea's level and join the ocean's
//! nodes, so they hold the sea at nought; a hollow 200 m inland stands 1 m under the sea's level, but no path of cells
//! under the level reaches it over the dune's crest, so it stays dry; the river's carved mouth holds its bank, which meets
//! the sea's level there with no step.

/// A cell where no water stands.
pub const DRY: i32 = i32::MIN;

/// ★ THE WATER OF A RASTER: `ground` the raster's heights (quarter metres over the sea, `side` a row; a cell no face holds
/// is [`DRY`] in `ground` too and never wet), `node_level` each cell's node's standing water level (quarter metres over
/// the sea) or `None`, and `river` each carved river cell's bank ([`DRY`] where none). Returns each cell's water level or
/// [`DRY`]. The waters spread in the order of their levels, lowest first, each from its own cells in raster order, so the
/// answer is one on every host.
#[must_use]
pub fn raster_water(
    ground: &[i32],
    side: usize,
    node_level: &[Option<i32>],
    river: &[i32],
) -> Vec<i32> {
    raster_water_within(ground, side, node_level, river, None)
}

/// ★ [`raster_water`] INSIDE A COARSER WATER'S REACH (W16: a finer level refines the coarser one and never contradicts it;
/// ruling B115): with `reach` given, a water of level `λ` spreads only into the cells whose `reach` is that same level —
/// the coarser level's water at the cell by the column rule (`FineBook::shore_water`) — so the 16 m water never runs out
/// through a 16 m valley the 128 m water does not hold. MEASURED need (2026-10-18): a lake at its 128 m spill stands at
/// that pass's height, and the 16 m detail lies a few metres under it in places, so the 16 m spread alone ran down the
/// valley beyond.
#[must_use]
pub fn raster_water_within(
    ground: &[i32],
    side: usize,
    node_level: &[Option<i32>],
    river: &[i32],
    reach: Option<&[i32]>,
) -> Vec<i32> {
    let n = ground.len();
    let mut water = vec![DRY; n];
    let mut levels: Vec<i32> = node_level.iter().flatten().copied().collect();
    levels.sort_unstable();
    levels.dedup();
    let under = |k: usize, lam: i32| {
        ground[k] != DRY && ground[k] < lam && reach.is_none_or(|r| r[k] == lam)
    };
    let mut seen = vec![false; n];
    // Each cell's column in its row, without a division (the crate's integer fence).
    let column: Vec<usize> = (0..n).map(|k| k % side).collect();
    for lam in levels {
        seen.iter_mut().for_each(|s| *s = false);
        let mut stack: Vec<usize> = (0..n)
            .filter(|&k| node_level[k] == Some(lam) && under(k, lam))
            .collect();
        for &k in &stack {
            seen[k] = true;
        }
        while let Some(k) = stack.pop() {
            water[k] = water[k].max(lam);
            let x = column[k];
            let around = [
                (x > 0).then(|| k - 1),
                (x + 1 < side).then(|| k + 1),
                (k >= side).then(|| k - side),
                (k + side < n).then(|| k + side),
            ];
            for m in around.into_iter().flatten() {
                if !seen[m] && under(m, lam) {
                    seen[m] = true;
                    stack.push(m);
                }
            }
        }
    }
    for (w, &bank) in water.iter_mut().zip(river) {
        if bank != DRY {
            *w = (*w).max(bank);
        }
    }
    water
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A lake's own cells under its level spread to the joined hollow, a hollow behind a crest stays dry, a river cell
    /// holds its bank, and where two waters meet the higher holds the cell.
    #[test]
    fn the_water_spreads_from_its_own_nodes_and_never_hangs() {
        // One row of nine cells: a lake node's cells at 0..3 (level 40), a crest at 4 (60), a hollow at 5 (10) the lake
        // cannot reach, a sea node's cell at 6 (level 0) whose ground stands at −8, a river cell at 7 (bank 20), dry land
        // at 8.
        let ground = [10, 20, 30, 35, 60, 10, -8, 12, 50];
        let lake = Some(40);
        let node_level = [lake, lake, lake, None, None, None, Some(0), None, None];
        let river = [DRY, DRY, DRY, DRY, DRY, DRY, DRY, 20, DRY];
        let w = raster_water(&ground, 9, &node_level, &river);
        assert_eq!(w, [40, 40, 40, 40, DRY, DRY, 0, 20, DRY]);
        // Two waters reaching one cell: the higher holds it.
        let ground2 = [0, 0, 0];
        let w2 = raster_water(&ground2, 3, &[Some(5), None, Some(9)], &[DRY; 3]);
        assert_eq!(w2, [9, 9, 9]);
        // A cell no face holds is never wet.
        let w3 = raster_water(&[DRY, 0], 2, &[Some(5), Some(5)], &[DRY; 2]);
        assert_eq!(w3, [DRY, 5]);
        // Inside a coarser water's reach: the lake at 40 stops where the coarser level holds no water at 40, though the
        // ground beyond stands under it.
        let reach = [40, 40, 40, 40, DRY, DRY, 0, DRY, DRY];
        let low = [10, 20, 30, 35, 30, 10, -8, 12, 50];
        let w4 = raster_water_within(&low, 9, &node_level, &river, Some(&reach));
        assert_eq!(w4, [40, 40, 40, 40, DRY, DRY, 0, 20, DRY]);
    }
}
