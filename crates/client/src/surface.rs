//! ★ THE FIRST SURFACE LOOK ON THE CLIENT (ruling B75 item 4, 2026-10-10): the client reads the planet's surface laws
//! (`vd_terrain::surface`) from what it already holds — the charter the realm stated, the artifact's head and its
//! pyramid — and nothing new crosses a realm boundary.
//! - **The rock** is a share of each ground vertex, from the slope the vertex is drawn at ([`vertex_rock`]); it needs
//!   no climate, so no chunk waits for it.
//! - **The snow line** is the planet's own, once per planet ([`snow_line_of`]): the climate law on the charter's words,
//!   its datum the head's sea, its season's heat store the sea's share the pyramid shows. The drawing paints snow over
//!   it.
//!
//! ★ INTERIMS (registered in `docs/design/DEFERRED.md`, "THE FIRST SURFACE LOOK"): the sea's share is read from the
//! finest pyramid level, never from the solve's own ocean mask (a block that holds a sea node and stands under the sea's
//! level counts whole); a body with no sea has no datum the client can read, so it draws no snow; the slope is the drawn
//! one (`vd_terrain::surface`'s interims).
//!
//! **Example.** The pilot logs in over the home planet. The head lands and the near chunks build at once, rock on their
//! steep sides. A second later the pyramid is whole, the planet's snow line is known, and the summits over it turn white
//! in the same frame everywhere.

use vd_core::look::BodyCharter;
use vd_terrain::BodyDefinition;
use vd_terrain::Gf;
use vd_terrain::climate::climate_laws;
use vd_terrain::solve::SolveWords;
use vd_terrain::surface::{COS_STEPS, SnowLine, rock_share};

use crate::artifact_book::ArtifactCache;

/// ★ THE WORDS OF A CHARTER (slice 8c stage C3): the ONE mapping from the census's charter to the generator's words —
/// shared by the shard's boot, the cross-pin, the benches (`vd_bins::solve_words_of`, which adds the census's system
/// age, a fact the charter does not state) and the client's surface laws, so a field can never be read into the wrong
/// word. The age here is nought: no climate law reads it.
#[must_use]
pub fn charter_words(c: &BodyCharter) -> SolveWords {
    SolveWords {
        water_km3: c.water_km3.unwrap_or(0),
        elastic_thickness_m: c.elastic_thickness_m.unwrap_or(0),
        insolation_q12: c.insolation_q12,
        t_eq_mk: c.t_eq_mk,
        t_surface_mk: c.t_surface_mk,
        bond_albedo_q12: c.bond_albedo_q12,
        mu_q8: c.mu_q8,
        scale_height_m: c.scale_height_m,
        p_surf_pa: c.p_surf_pa,
        tau_ir_q12: c.tau_ir_q12,
        day_s: c.day_s,
        obliquity_cos_q1024: c.obliquity_cos_q1024,
        ecc_q16: c.ecc_q16,
        year_s: c.year_s,
        flags: c.flags,
        age_yr: 0,
    }
}

/// The bare rock's share at a ground vertex, in 1/255: the slope between its unit `normal` and its unit `radial`
/// (`vd_terrain::surface::rock_share`). A normal that faces down reads as a vertical face.
#[must_use]
pub fn vertex_rock(normal: [f32; 3], radial: [f32; 3]) -> u8 {
    let cos = f64::from(normal[0]) * f64::from(radial[0])
        + f64::from(normal[1]) * f64::from(radial[1])
        + f64::from(normal[2]) * f64::from(radial[2]);
    rock_share((cos.clamp(0.0, 1.0) * f64::from(COS_STEPS)).round() as u32)
}

/// ★ THE PLANET'S SNOW LINE as the client reads it: `None` until the artifact is whole, and on a body with no sea or no
/// snow.
#[must_use]
pub fn snow_line_of(
    body: &BodyDefinition,
    charter: &BodyCharter,
    cache: &ArtifactCache,
) -> Option<SnowLine> {
    let sea_m = cache.head.sea().filter(|_| cache.whole())?;
    let laws = climate_laws(body, &charter_words(charter), wet_share_of(cache, sea_m));
    SnowLine::of(&laws, sea_m)
}

/// The sea's share of the surface as the finest pyramid level shows it: a block that holds a sea node (its water word is
/// the sea's level) AND whose mean stands under the sea's level counts whole. Nought with no pyramid.
#[must_use]
pub fn wet_share_of(cache: &ArtifactCache, sea_m: i32) -> Gf {
    let Some(level) = cache.level(1) else {
        return Gf::ZERO;
    };
    let wet = level
        .z_m
        .iter()
        .zip(&level.water_m)
        .filter(|&(&z, &w)| (i32::from(w) == sea_m) & (i32::from(z) < sea_m))
        .count();
    Gf::from_i64(wet as i64) / Gf::from_i64(level.z_m.len().max(1) as i64)
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::artifact_book::ArtifactReceiver;
    use vd_core::pose::RealmId;
    use vd_terrain::artifact::DRY_M;
    use vd_terrain::surface::FULL;
    use vd_wire::channels::BulkMsg;

    /// ★ A TINY ARTIFACT AT THE CLIENT: a body two nodes on an edge — one pyramid level of six blocks — whose sea
    /// stands at `sea_m` ([`DRY_M`] for none); with `level` the level's heights and water words are here, so the cache is
    /// whole, and without it only the head is.
    pub(crate) fn tiny_cache(
        realm: RealmId,
        sea_m: i16,
        level: Option<(Vec<i16>, Vec<i16>)>,
    ) -> ArtifactCache {
        let mut rx = ArtifactReceiver::default();
        rx.accept(BulkMsg::ArtifactHead {
            realm,
            world_tag: 9,
            version: vd_terrain::artifact::ARTIFACT_VERSION,
            edge: 2,
            digest: [3, u64::from(sea_m.unsigned_abs())],
            tiles_per_edge: 1,
            levels: 1,
            sea_m,
            coast_parts: 0,
            low_m: -100,
            high_m: 100,
            wind_parts: 0,
        });
        if let Some((z_m, water_m)) = level {
            rx.accept(BulkMsg::ArtifactPyramid {
                realm,
                level: 1,
                part: 0,
                parts: 1,
                z_m,
                water_m,
            });
        }
        rx.book().get(realm).cloned().expect("a cache")
    }

    /// Six blocks: two hold the sea and stand under it, one holds the sea but stands over it, one is dry under it, one
    /// is dry land and one a lake.
    pub(crate) fn six_blocks() -> (Vec<i16>, Vec<i16>) {
        (
            vec![-10, -10, 5, -10, 20, -5],
            vec![0, 0, 0, DRY_M, DRY_M, 7],
        )
    }

    /// The home planet's charter with a sea's worth of water: air AND water, so it snows.
    pub(crate) fn wet_charter() -> BodyCharter {
        BodyCharter {
            gravity_mm_s2: vd_terrain::home::HOME_PLANET_GRAVITY_MM_S2,
            bulk_density_kgm3: vd_terrain::home::HOME_PLANET_BULK_DENSITY_KGM3,
            escape_velocity_mps: 11_190,
            insolation_q12: 3_065,
            t_eq_mk: 236_795,
            t_surface_mk: Some(288_000),
            bond_albedo_q12: 1_228,
            mu_q8: Some(7_168),
            scale_height_m: Some(7_161),
            p_surf_pa: Some(101_325),
            tau_vis_q12: None,
            tau_ir_q12: Some(4_096),
            day_s: Some(86_400),
            obliquity_cos_q1024: Some(939),
            water_km3: Some(1_300_000_000),
            sea_offset_mm: None,
            elastic_thickness_m: Some(30_000),
            ecc_q16: 1_130,
            year_s: 34_766_100,
            flags: vd_core::look::CHARTER_FLAG_HAS_AIR | vd_core::look::CHARTER_FLAG_SOLID_SURFACE,
        }
    }

    #[test]
    fn the_words_are_the_charters_own_with_no_age() {
        let c = wet_charter();
        let w = charter_words(&c);
        assert_eq!(w.water_km3, 1_300_000_000);
        assert_eq!(w.elastic_thickness_m, 30_000);
        assert_eq!(w.t_surface_mk, Some(288_000));
        assert_eq!(w.obliquity_cos_q1024, Some(939));
        assert_eq!(w.year_s, 34_766_100);
        assert_eq!(w.age_yr, 0);
        let bare = charter_words(&BodyCharter {
            water_km3: None,
            elastic_thickness_m: None,
            ..c
        });
        assert_eq!((bare.water_km3, bare.elastic_thickness_m), (0, 0));
    }

    #[test]
    fn a_vertex_reads_its_rock_from_its_slope() {
        let up = [0.0, 0.0, 1.0];
        assert_eq!(vertex_rock(up, up), 0);
        assert_eq!(vertex_rock([0.0, 0.0, -1.0], up), FULL);
        let h = 0.5f32.sqrt();
        assert_eq!(vertex_rock([h, 0.0, h], up), FULL - 92);
    }

    #[test]
    fn the_seas_share_counts_the_blocks_that_hold_the_sea_under_its_level() {
        let realm = RealmId::Planet(7);
        let whole = tiny_cache(realm, 0, Some(six_blocks()));
        assert_eq!(wet_share_of(&whole, 0).to_f64(), 2.0 / 6.0);
        // Only the head: no level, no share.
        assert_eq!(wet_share_of(&tiny_cache(realm, 0, None), 0), Gf::ZERO);
    }

    #[test]
    fn the_snow_line_waits_for_the_whole_artifact_and_a_sea() {
        let realm = RealmId::Planet(7);
        let body = vd_terrain::home::home_planet();
        let c = wet_charter();
        // The head alone: not whole.
        assert_eq!(snow_line_of(&body, &c, &tiny_cache(realm, 0, None)), None);
        // A body with no sea: no datum to read.
        assert_eq!(
            snow_line_of(&body, &c, &tiny_cache(realm, DRY_M, Some(six_blocks()))),
            None
        );
        // Whole, with a sea: the climate law's own line on the sea's share the level shows.
        let whole = tiny_cache(realm, 0, Some(six_blocks()));
        let laws = climate_laws(&body, &charter_words(&c), wet_share_of(&whole, 0));
        let line = SnowLine::of(&laws, 0);
        assert!(line.is_some());
        assert_eq!(snow_line_of(&body, &c, &whole), line);
    }
}
