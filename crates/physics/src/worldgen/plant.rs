//! THE PLANTED FIXTURES: player-built content standing beside the generated bodies.
//!
//! Owns: which content a config plants, its derived spec, and the append that puts it into a forest
//! AFTER generation — so a plant can never perturb a draw and the generated world is bit-identical
//! with the plant on or off.
//!
//! Does NOT own: a special case anywhere else. A planted station is a realm like any other: the same
//! containment detector re-homes into it, the same fences judge it, and no live path can tell it
//! from a moon.

use super::{GeneratedBody, Placement, UniverseConfig, generate_system_forest, orbital_of};
use glam::DVec3;
use serde::{Deserialize, Serialize};
use vd_core::geometry::Boundary;
use vd_core::pose::RealmId;
use vd_core::rng::child_seed;
use vd_core::worldgen::GALAXY;

/// `child_seed` salt distinguishing FIXTURE-PLANTED player-built children (look_horizon.md slice 5
/// G-IDENTICAL — the SL5 fixture-forest doctrine) from every seed-generated kind: a planted
/// station's id is `f(its host system, this salt, index)` and a planted area's is `f(its host
/// planet, this salt, index)`, so plants can never collide with generated ids or with each other.
const FIXTURE_SALT: u64 = 0x0046_4958_5455_5245; // "FIXTURE"
/// The taxonomy-arc fixture plant sizes — the owner's Q3 build cases (real-scale design §6.3,
/// measured admission margins 8.51× and 4,261×): a 10 km CITY under the home system and a 20 m
/// STRUCTURE on the inner planet. Cited constants of the plant (player-built content the seed
/// never emits), not world knobs.
const FIXTURE_CITY_R_M: f64 = 1.0e4;
const FIXTURE_STRUCTURE_R_M: f64 = 20.0;

/// WHICH player-built content a config plants beside the generated bodies. `None` (default) is
/// byte-identical to the pre-plant world; the ONE named plant today is the slice-5 G-IDENTICAL
/// pair. A NAMED enumeration, deliberately not a geometry parameter: a free-form plant input would
/// be a second world generator wearing a config field (SL5 forbids it), while a named fixture is
/// content with one derivation, shared by every process that boots it.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub enum FixturePlant {
    /// Nothing built — THE world exactly as the seed generates it.
    #[default]
    None,
    /// The G-IDENTICAL pair (look_horizon.md slice 5): one player-built STATION under the home
    /// star system and one player-built AREA on that system's inner planet — see
    /// [`station_area_plant`] for every derived number.
    StationArea,
    /// ★ THE STATION/AREA PAIR **PLUS ONE BUILT SHIP** under the home star system (D-MOVE-2, owner
    /// ruling 2026-08-31: *"Ship is a separate Realm, but player built manually … we can't and should
    /// not generate the ship from the seed"*).
    ///
    /// A ship joins the plant rather than the generator for exactly the reason a station did: the seed
    /// does not make it, a player does. What the plant stands in for is PERSISTENCE — the built world
    /// nobody stores yet — and it is listed here so that when persistence lands, this arm is what it
    /// replaces.
    StationAreaShip,
}

/// The G-IDENTICAL plant's derived spec — public so the pixel gate's ORACLE derives its parks and
/// expectations from the SAME numbers the boots plant, out-of-band (never by reading the drawn
/// scene back).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct StationAreaPlant {
    /// The planted station's realm id: `Station(child_seed(home system, FIXTURE_SALT, 0))`.
    pub station: RealmId,
    /// The station's parent — the HOME star system (the root's first grandchild, the same lineage
    /// rule `default_home_realm` applies to the lowered forest).
    pub station_parent: RealmId,
    /// The station's static offset in its parent's frame.
    pub station_offset_m: DVec3,
    /// The station's shell radius.
    pub station_extent_m: f64,
    /// The planted area's realm id: `Area(child_seed(inner planet, FIXTURE_SALT, 0))`.
    pub area: RealmId,
    /// The area's parent — the home system's INNER planet (smallest semi-major axis, the same
    /// rule the world roster's `inner` uses). The frame law (`frame_for_realm`) requires an Area's
    /// parent to be a PLANET, which is WHY the pair is planted in the walk-forest shape (station
    /// under the system, area under a planet) and not as parent/child of each other.
    pub area_parent: RealmId,
    /// The area's static offset in the PLANET's frame.
    pub area_offset_m: DVec3,
    /// The area's shell radius.
    pub area_extent_m: f64,
}

/// The G-IDENTICAL plant, derived from a generated forest + its config. Every number is an
/// expression over THE world's own values, with its constraint stated (and pinned by this crate's
/// units — a plant that broke one would fail the boot fence loudly, not draw wrongly):
///
/// - **station extent** = the planet SOI radius (`planet.planet_soi_r_m`): planet-extent CLASS, so
///   every visibility bound the world already proves for a planet (reach 302.06 m, climb stops at
///   the galaxy) holds for the station verbatim, and the home system's interior band stays
///   PLANET-dominated (444.104489631 m — the station's `75 + 302.06 = 377.06 m` term is smaller).
/// - **station offset** = half the system shell radius up the `(1, 0, 2)/√5` tilted-polar
///   direction: `|offset| = 75 m` nests with a whole planet-orbit annulus of margin
///   (`75 + 3.954 < 150`); the `z = 67.1 m` component stands clear of the orbital plane (worst
///   planet `|z|` is apoapsis · sin(inclination), measured tiny against it in the units); the
///   `x = 33.5 m` component stands clear of BOTH ±Z polar flight axes (the licensed exit corridor
///   and the gate's own park legs) by far more than its extent.
/// - **area parent** = the INNER planet, forced by the physics, not chosen: under the OUTER planet
///   the area's worst-instant excursion (142.05 m at the eccentricity cap) leaves less system
///   slack than its own visibility reach, so its climb would be 3 and every boot would refuse
///   (the Q3 posture). Under the inner planet (excursion 16.07 m) the climb stops at the system:
///   levels 2, exactly what the carrier serves.
/// - **area extent** = a quarter of the planet SOI (`0.9885 m`): visibility reach
///   `76.39 × 0.9885 = 75.51 m` — the planet's interior band it induces brackets the planet's own
///   3.954 m shell (the park band exists), while the system-level slack
///   `150 − (16.07 + 1.977) − 0.99 = 130.97 m` stays far above that reach (the climb stops).
/// - **area offset** = half the planet SOI up +Z in the planet's frame: nests at `3/4` of the
///   planet's inscribed extent, a quarter-extent of margin.
#[must_use]
pub fn station_area_plant(seed_universe: u64, config: &UniverseConfig) -> StationAreaPlant {
    let mut base = *config;
    base.fixture_plant = FixturePlant::None;
    station_area_plant_spec(&generate_system_forest(seed_universe, &base))
}

/// The spec over an already-generated (plant-free) forest — the one derivation both
/// [`station_area_plant`] and the generator's own append share.
fn station_area_plant_spec(bodies: &[GeneratedBody]) -> StationAreaPlant {
    let home = bodies
        .iter()
        .find(|b| b.parent == Some(GALAXY))
        .expect("THE world generates at least one star system")
        .realm;
    let inner = bodies
        .iter()
        .filter(|b| b.parent == Some(home))
        .filter_map(|b| orbital_of(b.placement).map(|e| (b.realm, e.sma)))
        .min_by(|a, b| a.1.total_cmp(&b.1))
        .expect("THE home system generates orbiting planets")
        .0;
    let tilt = DVec3::new(1.0, 0.0, 2.0).normalize();
    let home_seed = plant_seed_of(home).expect("the home system is seed-keyed");
    let inner_seed = plant_seed_of(inner).expect("a generated planet is seed-keyed");
    // The plant's OFFSETS are proportions of the bodies it hangs under, read from the forest
    // itself (the config carried no in-system radii after the re-solve — the shells are
    // solved); its EXTENTS are the owner's Q3 build cases (the 10 km city, the 20 m structure).
    let home_shell_m = bodies
        .iter()
        .find(|b| b.realm == home)
        .expect("the home system is in the forest it was found in")
        .shape
        .finite_extent();
    let inner_shell_m = bodies
        .iter()
        .find(|b| b.realm == inner)
        .expect("the inner planet is in the forest it was found in")
        .shape
        .finite_extent();
    StationAreaPlant {
        station: RealmId::Station(child_seed(home_seed, FIXTURE_SALT, 0)),
        station_parent: home,
        station_offset_m: tilt * (home_shell_m * 0.5),
        station_extent_m: FIXTURE_CITY_R_M,
        area: RealmId::Area(child_seed(inner_seed, FIXTURE_SALT, 0)),
        area_parent: inner,
        area_offset_m: DVec3::new(0.0, 0.0, inner_shell_m * 0.5),
        area_extent_m: FIXTURE_STRUCTURE_R_M,
    }
}

/// The u64 seed of a SEED-LINEAGE realm (a system or a planet — the only parents a plant hangs
/// under), `None` for the entity/plant-keyed kinds. Monomorphic; both arms driven by named units.
pub(crate) fn plant_seed_of(realm: RealmId) -> Option<u64> {
    match realm {
        RealmId::System(s) | RealmId::Planet(s) => Some(s),
        // No fixture plants on a star (taxonomy arc §6.2 site 4): nothing is built inside the
        // dust-sublimation radius. And none directly in a galaxy or the universe: a plant hangs under a
        // seed-lineage parent that holds surfaces, and those two hold only other realms.
        RealmId::Ship(_)
        | RealmId::Station(_)
        | RealmId::Area(_)
        | RealmId::Star(_)
        | RealmId::Galaxy(_)
        | RealmId::Universe => None,
    }
}

/// ★ THE BUILT SHIP'S SPEC — derived from the world it is built in, never stated (D-MOVE-2).
///
/// **ITS IDENTITY IS MINTED, NOT SEEDED, and that is the whole point of the kind.** Every generated
/// realm's name comes from the world generator; a ship's comes from whichever shard built it, packed
/// `kind | mint_shard | seq | rand`. The fixture packs a FIXED one so the plant stays deterministic —
/// a built ship in the real game gets a live mint, and nothing else about it differs.
#[must_use]
pub fn built_ship_realm() -> RealmId {
    RealmId::Ship(vd_core::ids::EntityId::pack(
        vd_core::entity_kind::EntityKind::Ship,
        // A fixture's mint shard and sequence: fixed, so the plant is the same world every boot. The
        // 24 entropy bits are zero for the same reason — a fixture has nothing to guard against.
        1,
        1,
        0,
    ))
}

/// Append the named plant's bodies to a generated forest — called at the END of
/// [`generate_system_forest`], AFTER every generated body and every stream draw, so the additive
/// discipline holds: with a plant present every generated body, id, orbit and photometric draw is
/// byte-identical to the plant-free world.
pub(crate) fn append_fixture_plant(bodies: &mut Vec<GeneratedBody>, config: &UniverseConfig) {
    match config.fixture_plant {
        FixturePlant::None => {}
        FixturePlant::StationArea => {
            let plant = station_area_plant_spec(bodies);
            bodies.push(GeneratedBody {
                realm: plant.station,
                parent: Some(plant.station_parent),
                shape: Boundary::Shell {
                    r: plant.station_extent_m,
                },
                placement: Placement::StaticOffset(plant.station_offset_m),
                // A player-built structure has no seed stream and no photometric draw — the
                // presence floor (look_horizon slice 1) states its point of light from its
                // extent alone.
                photometrics: None,
                taxon: None,
                look: Some(Boundary::Shell {
                    r: plant.station_extent_m,
                }),
            });
            bodies.push(GeneratedBody {
                realm: plant.area,
                parent: Some(plant.area_parent),
                shape: Boundary::Shell {
                    r: plant.area_extent_m,
                },
                placement: Placement::StaticOffset(plant.area_offset_m),
                photometrics: None,
                taxon: None,
                look: Some(Boundary::Shell {
                    r: plant.area_extent_m,
                }),
            });
        }
        FixturePlant::StationAreaShip => {
            // The pair first, byte-for-byte as `StationArea` plants it — a ship is APPENDED to the
            // built world, never a different one. Recursing keeps ONE derivation of the pair, so the
            // two arms can never drift apart.
            let mut pair = config.clone();
            pair.fixture_plant = FixturePlant::StationArea;
            append_fixture_plant(bodies, &pair);
            let plant = station_area_plant_spec(bodies);
            bodies.push(GeneratedBody {
                realm: built_ship_realm(),
                // Under the home STAR SYSTEM, which is the parent the movement ruling's own worked
                // example uses: it adds the star's pull to the ship's push and authors where the ship
                // then is.
                parent: Some(plant.station_parent),
                shape: Boundary::Shell { r: SHIP_EXTENT_M },
                // ★ WHERE IT STARTS, AND ONLY WHERE IT STARTS. A ship is a DRIVEN child: from its
                // first tick its parent authors its placement from the pushes it states, so this
                // offset is the hull's starting berth and nothing more.
                //
                // ★ BERTHED WHERE A PLAYER ACTUALLY ARRIVES, AND THAT WAS MEASURED (2026-09-01). The
                // first version put it beside the STATION, because that is where a built ship would
                // sit. MEASURED: the station hangs elsewhere in the system, so the walk from the spawn
                // was 1.08e10 m — about 250 DAYS of continuous flying. A berth nobody can reach is a
                // ship nobody can fly.
                //
                // The berth is derived from the SAME clearing the spawn is derived from, so the two
                // cannot drift apart: move the spawn and the berth follows it.
                placement: Placement::StaticOffset(
                    spawn_clearing(bodies) + DVec3::new(SHIP_BERTH_OFFSET_M, 0.0, 0.0),
                ),
                // A built hull has no seed stream and no photometric draw, exactly like the station.
                photometrics: None,
                taxon: None,
                look: Some(Boundary::Shell { r: SHIP_EXTENT_M }),
            });
        }
    }
}

/// How big the fixture's hull is, across. A 40 m ship is the owner's own build case for a small
/// multi-crew vessel, and it sits well inside the station's own extent so the berth is lawful.
const SHIP_EXTENT_M: f64 = 20.0;

/// How far the berth sits to one side of where a player arrives: far enough that a hull is not sitting
/// on top of them, near enough to fly to in seconds. At 500 m/s this is a two-second hop.
const SHIP_BERTH_OFFSET_M: f64 = 1_000.0;

/// WHERE A PLAYER ARRIVES in the home system, derived exactly as `WorldView::default_home_offset_m`
/// derives it — twice the largest statically-placed body's own bound, along +Z.
///
/// **THE SAME DERIVATION, not a copy of the number.** A berth stated as a literal would silently stop
/// matching the spawn the day the home system's contents changed, and the ship would drift out of
/// reach again with nothing going red.
fn spawn_clearing(bodies: &[GeneratedBody]) -> DVec3 {
    let Some(home) = bodies
        .iter()
        .find(|b| b.parent == Some(GALAXY))
        .map(|b| b.realm)
    else {
        return DVec3::ZERO;
    };
    let clearing_z = bodies
        .iter()
        .filter(|b| b.parent == Some(home))
        .filter_map(|b| match b.placement {
            Placement::StaticOffset(at) => {
                let bound = b.shape.finite_extent();
                (at.length() < bound).then_some(2.0 * bound)
            }
            Placement::Orbital(_) => None,
        })
        .fold(0.0, f64::max);
    DVec3::new(0.0, 0.0, clearing_z)
}

// ===== T4 — THE EARTH-LIKE PREDICATE + THE SEED SEARCH (celestial_taxonomy_design §8) =======

#[cfg(test)]
mod ship_plant_tests {
    use super::{FixturePlant, built_ship_realm};
    use crate::worldgen::{UniverseConfig, generate_system_forest};

    fn worlds() -> (Vec<crate::worldgen::GeneratedBody>, Vec<crate::worldgen::GeneratedBody>) {
        let base = UniverseConfig::world(500.0, 0.02);
        let with_ship = base.clone().with_station_area_ship_plant();
        let pair_only = base.with_station_area_plant();
        (
            generate_system_forest(vd_physics_seed(), &pair_only),
            generate_system_forest(vd_physics_seed(), &with_ship),
        )
    }
    fn vd_physics_seed() -> u64 {
        crate::worldgen::HOME_SEED
    }

    #[test]
    fn the_ship_is_appended_and_the_built_world_under_it_is_untouched() {
        let (pair, with_ship) = worlds();
        // ★ ADDITIVE, exactly as the station and area are. Adding a ship must not move, renumber or
        // re-draw a single body that was already there — the discipline the plant exists to keep.
        assert_eq!(with_ship.len(), pair.len() + 1, "exactly one body appended");
        assert_eq!(
            with_ship[..pair.len()],
            pair[..],
            "every body before the ship is byte-identical"
        );
    }

    #[test]
    fn the_ship_hangs_under_the_home_star_system_which_does_the_physics() {
        let (_, with_ship) = worlds();
        let ship = with_ship.last().expect("the ship is last");
        assert_eq!(ship.realm, built_ship_realm());
        // Its parent is the system that integrates it — the parent in the movement ruling's own
        // worked example.
        let station = with_ship
            .iter()
            .find(|b| matches!(b.realm, vd_core::pose::RealmId::Station(_)))
            .expect("the pair plants a station");
        assert_eq!(ship.parent, station.parent, "berthed in the same realm as the station");
    }

    #[test]
    fn a_built_ship_has_no_seed_draw_because_no_seed_made_it() {
        let (_, with_ship) = worlds();
        let ship = with_ship.last().expect("the ship is last");
        // A generated body draws its light and its taxon from the seed stream. A built hull has
        // neither, which is what "the seed did not make this" means in the data.
        assert!(ship.photometrics.is_none());
        assert!(ship.taxon.is_none());
    }

    #[test]
    fn the_plain_world_still_plants_nothing() {
        let plain = UniverseConfig::world(500.0, 0.02);
        assert_eq!(plain.fixture_plant, FixturePlant::None);
    }
}

#[cfg(test)]
mod ship_wake_tests {
    use super::built_ship_realm;
    use crate::worldgen::{UniverseConfig, realm_regions_for_config};

    /// ★ DOES A BUILT SHIP WAKE LIKE EVERYTHING ELSE? The wake rule is generic — a realm states ONE
    /// radius, derived from its own size and its own speed, and the same verdict decides both "wake
    /// this child" and "tell this child a looker is near". A ship must need no special path.
    #[test]
    fn a_built_ship_states_a_wake_radius_like_every_other_realm() {
        let config = UniverseConfig::world(500.0, 0.02).with_station_area_ship_plant();
        let regions = realm_regions_for_config(crate::worldgen::HOME_SEED, &config);
        let ship = regions
            .iter()
            .find(|r| r.realm == built_ship_realm())
            .expect("the plant put a ship in the world");
        assert!(
            ship.aoi.spin_up_r_m() > 0.0,
            "a ship wakes by the same rule as a planet: {}",
            ship.aoi.spin_up_r_m()
        );
        assert!(
            ship.aoi.tear_down_r_m() > ship.aoi.spin_up_r_m(),
            "and it sleeps further out than it wakes, so it cannot flap"
        );
    }

    /// A hull is small, so its wake radius is small — you must be near it to see it, exactly as the
    /// owner expects for a body this size. Stated as a RELATION to the station beside it rather than
    /// as a number, so the assertion survives every re-solve of the wake law.
    #[test]
    fn a_ships_wake_reach_is_smaller_than_the_station_it_is_berthed_beside() {
        let config = UniverseConfig::world(500.0, 0.02).with_station_area_ship_plant();
        let regions = realm_regions_for_config(crate::worldgen::HOME_SEED, &config);
        let ship = regions
            .iter()
            .find(|r| r.realm == built_ship_realm())
            .expect("the ship is in the world");
        let station = regions
            .iter()
            .find(|r| matches!(r.realm, vd_core::pose::RealmId::Station(_)))
            .expect("the station is in the world");
        assert!(
            ship.aoi.spin_up_r_m() < station.aoi.spin_up_r_m(),
            "a 20 m hull is seen from closer than a station: ship {} vs station {}",
            ship.aoi.spin_up_r_m(),
            station.aoi.spin_up_r_m()
        );
    }
}

#[cfg(test)]
mod ship_reach_tests {
    use super::built_ship_realm;
    use crate::worldgen::{UniverseConfig, WorldView};

    /// ★ CAN A PLAYER ACTUALLY REACH THE SHIP? Measured rather than assumed, before anything is built
    /// on top of it: a berth nobody can fly to is a ship nobody can test.
    #[test]
    fn the_berth_is_within_reach_of_where_a_player_starts() {
        let config = UniverseConfig::world(500.0, 0.02).with_station_area_ship_plant();
        let world = WorldView::generated(crate::worldgen::HOME_SEED, &config);
        let ship = world
            .regions()
            .iter()
            .find(|r| r.realm == built_ship_realm())
            .expect("the plant put a ship in the world");
        let spawn = world.default_home_offset_m();
        // ★ THE WHOLE POSITION, cell anchor included. Reading only the leftover offset is the
        // parent-frame unit trap: at star-system scale the CELL carries nearly all the magnitude, so
        // an offset-only read reports a berth near the origin however far out the hull really is.
        let berth_pos = ship.center.in_parents_frame();
        let berth = berth_pos.offset()
            + berth_pos.cell().as_dvec3() * ship.frame.tier().cell_edge_m();
        println!(
            "[ship-reach] berth cell {:?} offset {:?}",
            berth_pos.cell(),
            berth_pos.offset()
        );
        println!("[ship-reach] spawn {spawn:?}");
        let gap_m = (berth - spawn).length();
        let minutes_at_walk = gap_m / 500.0 / 60.0;
        println!(
            "[ship-reach] spawn to berth: {gap_m:.3e} m = {minutes_at_walk:.1} minutes at 500 m/s"
        );
        assert!(
            minutes_at_walk < 5.0,
            "a player must be able to reach the ship in minutes, not hours: \
             {gap_m:.3e} m is {minutes_at_walk:.1} minutes at 500 m/s"
        );
    }
}
