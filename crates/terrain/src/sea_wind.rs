//! ★ THE SEA'S WIND (ruling B109; the sea's look, `docs/investigation/2026-10-18/sea_light_design.md` Q2): the year's
//! surface wind of each climate cell, from the seasons' chain run once on a body's STORED ground — the planet's shard
//! computes it when it loads, keeps it beside the artifact and ships it (SL6 yes, B107 Q1). The sun's glint reads its
//! speed by Cox & Munk (1954); the waves (slice S3) read its direction. The same function on the same words gives the
//! same bytes on any number of cores (the chain joins its seasons in their order).
//!
//! **Example.** The home planet's shard loads its stored world, runs the chain once (about ten calls of spin-up on 12
//! cores), and ships 34 656 cells of six bytes: the trade wind east of the coast stand blows about 7 m/s from the east.

use crate::BodyDefinition;
use crate::artifact::{Artifact, DRY_M, FACIES_SHIFT};
use crate::gf::Gf;
use crate::macro_lattice::MacroLattice;
use crate::seasons::chain::{CLIMATE_ROWS, SeasonState, seasonal_climate};
use crate::solve::{FACIES_SEA, SolveWords, WORD_FLAG_TIDALLY_LOCKED, Z_STEPS_PER_M};

/// The surface wind's version word: a reader refuses any other.
pub const SEA_WIND_VERSION: u32 = 1;

/// ★ THE YEAR'S SURFACE WIND of each cell of the climate grid (`crate::seasons::grid::ClimateGrid` of [`SeaWind::rows`]
/// latitude bands over the body's radius), whole centimetres a second: the mean of the seasons' winds east and north,
/// and the mean of their speeds.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SeaWind {
    /// The climate grid's latitude bands.
    pub rows: u32,
    /// The year's mean wind toward the east, cm/s, per cell.
    pub east_cms: Vec<i16>,
    /// The year's mean wind toward the north, cm/s, per cell.
    pub north_cms: Vec<i16>,
    /// The year's mean wind speed, cm/s, per cell.
    pub speed_cms: Vec<u16>,
}

/// ★ WHETHER A WORLD HAS A SEA'S WIND: the seasons' chain runs on its body (air with a day of its own, not star-locked —
/// the solve's own climate source) and its stored world holds a sea. The one rule the shard reads to wait for it and
/// [`sea_wind`] reads to compute it.
#[must_use]
pub fn blows(words: &SolveWords, artifact: &Artifact) -> bool {
    let air = words.p_surf_pa.is_some_and(|p| p > 0) & words.day_s.is_some();
    let locked = words.flags & WORD_FLAG_TIDALLY_LOCKED != 0;
    air & !locked & (artifact.sea_m != DRY_M)
}

/// ★ THE SEA'S WIND of `artifact`, a body's stored world: `None` where the world has none ([`blows`]).
#[must_use]
pub fn sea_wind(
    body: &BodyDefinition,
    lattice: &MacroLattice,
    words: &SolveWords,
    artifact: &Artifact,
    threads: usize,
) -> Option<SeaWind> {
    if !blows(words, artifact) {
        return None;
    }
    // The stored ground in the solve's sixteenths, the connected ocean from the rows' own facies, the sea's level.
    let z: Vec<i32> = artifact
        .rows
        .iter()
        .map(|r| i32::from(r.z_m) * Z_STEPS_PER_M)
        .collect();
    let ocean: Vec<bool> = artifact
        .rows
        .iter()
        .map(|r| (r.receiver_facies >> FACIES_SHIFT) & FACIES_SEA != 0)
        .collect();
    let sea = crate::climate::SeaView {
        level: i32::from(artifact.sea_m) * Z_STEPS_PER_M,
        ocean: &ocean,
    };
    let mut state = SeasonState::new(lattice, threads);
    let _ = seasonal_climate(body, lattice, words, &z, Some(sea), threads, &mut state);
    Some(quantised(CLIMATE_ROWS as u32, &state.last_wind))
}

/// The year's winds in whole cm/s, rounded to the nearest, held inside their words.
fn quantised(rows: u32, wind: &[[Gf; 3]]) -> SeaWind {
    let cms = |v: Gf| (v * Gf::from_i64(100) + Gf::HALF).to_i64_floor();
    SeaWind {
        rows,
        east_cms: wind
            .iter()
            .map(|w| cms(w[0]).clamp(i64::from(i16::MIN), i64::from(i16::MAX)) as i16)
            .collect(),
        north_cms: wind
            .iter()
            .map(|w| cms(w[1]).clamp(i64::from(i16::MIN), i64::from(i16::MAX)) as i16)
            .collect(),
        speed_cms: wind
            .iter()
            .map(|w| cms(w[2]).clamp(0, i64::from(u16::MAX)) as u16)
            .collect(),
    }
}

impl SeaWind {
    /// The bytes the store keeps and the wire carries: the version, the rows, the cell count, then the three columns,
    /// little-endian.
    #[must_use]
    pub fn to_bytes(&self) -> Vec<u8> {
        let n = self.east_cms.len();
        let mut b = Vec::with_capacity(12 + 6 * n);
        b.extend_from_slice(&SEA_WIND_VERSION.to_le_bytes());
        b.extend_from_slice(&self.rows.to_le_bytes());
        b.extend_from_slice(&(n as u32).to_le_bytes());
        for v in &self.east_cms {
            b.extend_from_slice(&v.to_le_bytes());
        }
        for v in &self.north_cms {
            b.extend_from_slice(&v.to_le_bytes());
        }
        for v in &self.speed_cms {
            b.extend_from_slice(&v.to_le_bytes());
        }
        b
    }

    /// The wind out of [`SeaWind::to_bytes`]'s form; `None` for another version or a short or long record.
    #[must_use]
    pub fn from_bytes(b: &[u8]) -> Option<SeaWind> {
        let word = |i: usize| -> Option<u32> {
            Some(u32::from_le_bytes(
                b.get(4 * i..4 * i + 4)?.try_into().ok()?,
            ))
        };
        let n = word(2)? as usize;
        if (word(0)? != SEA_WIND_VERSION) | (b.len() != 12 + 6 * n) {
            return None;
        }
        let column = |k: usize| -> Vec<[u8; 2]> {
            b[12 + 2 * n * k..12 + 2 * n * (k + 1)]
                .chunks_exact(2)
                .map(|c| [c[0], c[1]])
                .collect()
        };
        Some(SeaWind {
            rows: word(1)?,
            east_cms: column(0).into_iter().map(i16::from_le_bytes).collect(),
            north_cms: column(1).into_iter().map(i16::from_le_bytes).collect(),
            speed_cms: column(2).into_iter().map(u16::from_le_bytes).collect(),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// ★ THE WIND'S BYTES ROUND-TRIP, and a record of another version or length is refused.
    ///
    /// **Example.** Three cells of the trades: east −700 cm/s, north 50, speed 720; read back word for word.
    #[test]
    fn the_winds_bytes_round_trip_and_a_wrong_record_is_refused() {
        let w = SeaWind {
            rows: 90,
            east_cms: vec![-700, 0, 300],
            north_cms: vec![50, -20, 0],
            speed_cms: vec![720, 30, 310],
        };
        let b = w.to_bytes();
        assert_eq!(b.len(), 12 + 18);
        assert_eq!(SeaWind::from_bytes(&b), Some(w));
        assert_eq!(SeaWind::from_bytes(&b[..b.len() - 1]), None);
        let mut other = b.clone();
        other[0] = 2;
        assert_eq!(SeaWind::from_bytes(&other), None);
        assert_eq!(SeaWind::from_bytes(&b[..8]), None);
    }

    /// ★ THE YEAR'S WIND IN WHOLE cm/s: rounded to the nearest and held inside its word.
    #[test]
    fn the_wind_is_rounded_to_whole_centimetres_and_held_in_its_word() {
        let w = quantised(
            90,
            &[
                [Gf::from_f64(-7.004), Gf::from_f64(0.506), Gf::from_f64(7.2)],
                [Gf::from_f64(1e4), Gf::from_f64(-1e4), Gf::from_f64(1e4)],
            ],
        );
        assert_eq!(w.east_cms, vec![-700, i16::MAX]);
        assert_eq!(w.north_cms, vec![51, i16::MIN]);
        assert_eq!(w.speed_cms, vec![720, u16::MAX]);
    }
}
